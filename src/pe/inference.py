"""Predicting the test series with the slice models alone or with the series models (06_submission).

Per series (one StudyInstanceUID folder) every DICOM file is read once. Its headers give the stacks and the main stack as
in 02 (`pe.data.prepare`); the main-stack slices are decoded to the values of the png16 store, their lung boxes give the
series' box (`series_boxes`), and every main-stack slice is cut to that box and stretched back to 512 px, as the slice
models were trained (04). The models of a variant run in float16, the mean logit of each backbone is calibrated with the
fit of scripts/fit_slice_ensemble.py, the series labels are the training label rates, and the consistency fix follows.
A slice or series that cannot be read gets the prediction of a slice outside the main stack, never an error.

With the series models (`SeriesPredictor`, `predict_series`), each GPU holds every model of the variant and takes whole
series from a shared queue, so a variant with a single slice model uses every GPU too. Per series, the slice models of
fold k give the features of the series models of fold k (05); their probabilities are averaged over models and folds.
"""
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from scipy.special import expit
from torch.utils.data import DataLoader, Dataset

from pe.consistency import make_consistent
from pe.data.dataset import series_boxes
from pe.data.dicom import HU_MIN, as_text, dataset_hu
from pe.data.prepare import SERIES_LABELS, STORE, add_stacks, pixel_stats, slice_table
from pe.data.windows import WINDOWS
from pe.models.series_model import SeriesModel
from pe.models.slice_model import SliceModel
from pe.series_training import with_neighbours
from pe.training import to_windows

SIZE = 512
WINDOW_NAMES = ["pe", "mediastinal", "lung"]
# What slice_table reads (the file meta fields are looked up there first, as in scripts/extract_dicom_headers.py)
HEADER_FIELDS = ["SOPInstanceUID", "StudyInstanceUID", "InstanceNumber", "ImplementationVersionName", "TransferSyntaxUID", "BitsStored",
                 "PixelRepresentation", "RescaleIntercept", "SliceThickness", "PixelSpacing", "ImagePositionPatient",
                 "ImageOrientationPatient", "ConvolutionKernel"]
# Relative GPU time per slice, measured on a Kaggle T4 (05, step A): to spread the models over the GPUs
COST = {"seresnext101": 1 / 90, "convnext_tiny": 1 / 150}


def read_study(study_dir) -> tuple[pd.DataFrame, dict]:
    """The slice table of one series (`add_stacks`: stack, main, body order) and its read files by SOPInstanceUID."""
    rows, files = [], {}
    for path in sorted(Path(study_dir).glob("*/*.dcm")):
        try:
            ds = pydicom.dcmread(path)
        except Exception as error:
            print(f"not read: {path} ({error!r})", flush=True)
            continue
        row = {"path": str(path), "split": "test"}
        for field in HEADER_FIELDS:
            value = ds.file_meta.get(field) if field in ds.file_meta else ds.get(field)
            row[field] = "" if value is None else as_text(value)
        rows.append(row)
        files[row["SOPInstanceUID"]] = ds
    return add_stacks(slice_table(pd.DataFrame(rows))), files


def prepare_study(study_dir) -> dict:
    """The model input of one series: `crops` (main-stack slices x 512 x 512, int16, the stored values of 04) with `ok`
    (False where a slice could not be decoded), and every slice's `image` and `main` flag, main-stack slices first in
    body order."""
    slices, files = read_study(study_dir)
    main = slices[slices["main"]].reset_index(drop=True)
    if main.empty:  # no axial stack: nothing the models were trained on
        return {"study": Path(study_dir).name, "crops": torch.zeros(0, SIZE, SIZE, dtype=torch.int16), "ok": np.zeros(0, bool),
                "image": slices["image"].to_numpy(dtype="U"), "main": np.zeros(len(slices), bool)}
    stored, ok = [], []
    for image in main["image"]:
        try:
            stored.append(STORE.to_array(dataset_hu(files[image])))
            ok.append(True)
        except Exception as error:
            print(f"not decoded: {study_dir} {image} ({error!r})", flush=True)
            stored.append(np.zeros((SIZE, SIZE), np.uint16))
            ok.append(False)
    main = pd.concat([main, pd.DataFrame([pixel_stats(s.astype(np.int16) + HU_MIN) for s in stored])], axis=1)
    if (main["lung_top"] >= 0).any():
        top, bottom, left, right = series_boxes(main, "lung", SIZE)[0]
    else:
        top, bottom, left, right = 0, SIZE, 0, SIZE
    crops = np.stack([cv2.resize(s[top:bottom, left:right], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR) for s in stored]).astype(np.int16)
    others = slices.loc[~slices["main"], "image"]
    return {"study": Path(study_dir).name, "crops": torch.from_numpy(crops), "ok": np.array(ok),
            "image": np.r_[main["image"].to_numpy(dtype="U"), others.to_numpy(dtype="U")], "main": np.r_[np.ones(len(main), bool), np.zeros(len(others), bool)]}


def as_is(item):
    """The loader's collate function: the item unchanged (no conversion of its arrays to tensors)."""
    return item


class StudyDataset(Dataset):
    """One item per series folder: `prepare_study`, or the error if the series cannot be prepared at all."""

    def __init__(self, study_dirs):
        self.study_dirs = list(study_dirs)

    def __len__(self):
        return len(self.study_dirs)

    def __getitem__(self, i):
        try:
            return prepare_study(self.study_dirs[i])
        except Exception as error:
            return {"study": Path(self.study_dirs[i]).name, "error": repr(error)}


class SliceEnsemble:
    """The slice models of one variant spread over the GPUs; called on a series' crops, the calibrated slice logits.

    `variant`: from ensemble.json (backbones, folds, intercept, weights); `files[backbone][fold]`: the weights file.
    Each model runs in float16 with its head in float32; the logits of a backbone's folds are averaged."""

    def __init__(self, variant: dict, files: dict, devices: list[str], batch: int = 32):
        self.variant, self.batch = variant, batch
        self.windows = [WINDOWS[name] for name in WINDOW_NAMES]
        jobs = sorted(((b, k) for b in variant["backbones"] for k in variant["folds"]), key=lambda job: -COST[job[0]])
        self.models = {device: [] for device in devices}  # devices without a model are dropped below
        load = {device: 0.0 for device in devices}
        for backbone, fold in jobs:  # the costliest model first, to the GPU with the least work
            device = min(load, key=load.get)
            model = SliceModel(backbone, in_chans=len(self.windows), pretrained=False)
            model.load_state_dict(torch.load(files[backbone][fold], map_location="cpu"))
            model = model.half().eval().to(device, memory_format=torch.channels_last)
            model.head.float()
            self.models[device].append((backbone, model))
            load[device] += COST[backbone]
        self.models = {device: models for device, models in self.models.items() if models}
        self.pool = ThreadPoolExecutor(len(self.models))

    def _run(self, device, crops):
        sums = {}
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch):
                x = crops[start:start + self.batch, None].to(device)
                windows = to_windows(x, self.windows).half().contiguous(memory_format=torch.channels_last)
                for backbone, model in self.models[device]:
                    logits = model(windows)[0].float().cpu().numpy()
                    sums.setdefault(backbone, np.zeros(len(crops)))[start:start + len(logits)] += logits
        return sums

    def __call__(self, crops: torch.Tensor) -> np.ndarray:
        sums = {}
        for part in self.pool.map(lambda device: self._run(device, crops), list(self.models)):
            for backbone, total in part.items():
                sums[backbone] = sums.get(backbone, 0) + total
        means = [sums[b] / len(self.variant["folds"]) for b in self.variant["backbones"]]
        return self.variant["intercept"] + sum(w * m for w, m in zip(self.variant["weights"], means))


def predict_slices(study_dirs, ensemble: SliceEnsemble, workers: int = 4, log_every: int = 100) -> pd.DataFrame:
    """One row per slice found: study, image, the calibrated probability (NaN outside the main stack or where a slice
    could not be decoded) and whether it may be raised by the consistency fix."""
    # One series in flight per worker: a long series is ~0.5 GB of crops in shared memory
    loader = DataLoader(StudyDataset(study_dirs), batch_size=None, collate_fn=as_is, num_workers=workers, prefetch_factor=1 if workers else None)
    parts, start, done = [], time.time(), 0
    for item in loader:
        if "error" in item:
            print(f"series not prepared: {item['study']} ({item['error']})", flush=True)
            continue
        p = np.full(len(item["image"]), np.nan)
        n = len(item["crops"])
        if n:
            p[:n] = np.where(item["ok"], expit(ensemble(item["crops"])), np.nan)
        parts.append(pd.DataFrame({"study": item["study"], "image": item["image"], "p": p, "main": item["main"] & ~np.isnan(p)}))
        done += 1
        if done % log_every == 0:
            print(f"{done:,} series, {sum(len(x) for x in parts):,} slices, {(time.time() - start) / 60:.1f} min", flush=True)
    if not parts:
        return pd.DataFrame(columns=["study", "image", "p", "main"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("image")  # a file twice in a series: its main-stack row


class SeriesPredictor:
    """Every model of a series variant on one GPU; called on a series' crops, its nine series probabilities and its
    slices' probabilities, before the consistency fix.

    `slice_files[backbone][fold]`: slice model weights; `series_runs`: per series model, the run directory of each fold
    (config.json and model.pt of scripts/train_series.py); `folds`: the folds of the variant. The slice models run in
    float16 with their heads in float32, the series models in float32, one series at a time."""

    def __init__(self, slice_files: dict, series_runs: list[list], folds: list[int], device: str, batch: int = 32):
        self.device, self.batch, self.folds = device, batch, list(folds)
        self.windows = [WINDOWS[name] for name in WINDOW_NAMES]
        self.series = {k: [] for k in self.folds}  # fold: [(model, backbones, input, neighbours)]
        needed = set()
        for runs in series_runs:
            for k in self.folds:
                config = json.loads((Path(runs[k]) / "config.json").read_text())
                assert config["fold"] == k, runs[k]
                backbones = config["backbones"].split(",")
                width = sum({"embedding": NUM_FEATURES[b], "logit": 1, "both": NUM_FEATURES[b] + 1}[config["input"]] for b in backbones)
                neighbours = not config["no_neighbours"]
                model = SeriesModel(width * (3 if neighbours else 1), config["hidden"], len(SERIES_LABELS), config["dropout"])
                model.load_state_dict(torch.load(Path(runs[k]) / "model.pt", map_location="cpu"))
                self.series[k].append((model.eval().to(device), backbones, config["input"], neighbours))
                needed |= {(b, k) for b in backbones}
        self.slice_models = {}
        for backbone, fold in sorted(needed):
            model = SliceModel(backbone, in_chans=len(self.windows), pretrained=False)
            model.load_state_dict(torch.load(slice_files[backbone][fold], map_location="cpu"))
            model = model.half().eval().to(device, memory_format=torch.channels_last)
            model.head.float()
            self.slice_models[backbone, fold] = model

    @torch.inference_mode()
    def __call__(self, crops: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        n = len(crops)
        outputs = {key: (torch.empty(n, NUM_FEATURES[key[0]], device=self.device), torch.empty(n, 1, device=self.device)) for key in self.slice_models}
        for start in range(0, n, self.batch):
            x = crops[start:start + self.batch, None].to(self.device)
            windows = to_windows(x, self.windows).half().contiguous(memory_format=torch.channels_last)
            for key, model in self.slice_models.items():
                logits, features = model(windows)
                outputs[key][0][start:start + len(x)] = features.float()
                outputs[key][1][start:start + len(x), 0] = logits.float()
        p_series, p_image, count = np.zeros(len(SERIES_LABELS)), np.zeros(n), 0
        for k in self.folds:
            for model, backbones, kind, neighbours in self.series[k]:
                parts = [part for b in backbones for part, wanted in zip(outputs[b, k], (kind != "logit", kind != "embedding")) if wanted]
                x = torch.cat(parts, dim=1)[None]
                if neighbours:
                    x = with_neighbours(x)
                slice_logits, series_logits = model(x, torch.tensor([n]))
                p_image += torch.sigmoid(slice_logits[0]).cpu().numpy()
                p_series += torch.sigmoid(series_logits[0]).cpu().numpy()
                count += 1
        return p_series / count, p_image / count


# Embedding width of each backbone (timm num_features)
NUM_FEATURES = {"seresnext101": 2048, "convnext_tiny": 768}


def predict_series(study_dirs, predictors: list[SeriesPredictor], workers: int = 4, log_every: int = 100) -> tuple[pd.DataFrame, dict]:
    """One row per slice found (study, image, the probability: NaN outside the main stack or where a slice could not be
    decoded, and whether it may be raised by the fix) and the nine series probabilities of each predicted series. The
    series are prepared by `workers` processes and taken from a queue by one thread per predictor (GPU)."""
    loader = DataLoader(StudyDataset(study_dirs), batch_size=None, collate_fn=as_is, num_workers=workers, prefetch_factor=2 if workers else None)
    todo = queue.Queue(maxsize=2 * len(predictors))
    parts, series_p, lock, start = [], {}, threading.Lock(), time.time()
    busy = [0.0] * len(predictors)
    counts = [0] * len(predictors)
    errors = []

    def consume(i):
        while (item := todo.get()) is not None:
            try:
                p = np.full(len(item["image"]), np.nan)
                ok = np.flatnonzero(item["ok"])
                if len(ok):
                    t = time.time()
                    p_series, p_image = predictors[i](item["crops"][ok])
                    busy[i] += time.time() - t
                    p[ok] = p_image
                    with lock:
                        series_p[item["study"]] = p_series
                with lock:
                    parts.append(pd.DataFrame({"study": item["study"], "image": item["image"], "p": p, "main": item["main"] & ~np.isnan(p)}))
                    counts[i] += 1
                    done = sum(counts)
                if done % log_every == 0:
                    print(f"{done:,} series, {(time.time() - start) / 60:.1f} min, series per GPU {counts}", flush=True)
            except Exception as error:  # this series keeps the fallback; the others go on
                errors.append(f"{item['study']}: {error!r}")

    threads = [threading.Thread(target=consume, args=(i,), daemon=True) for i in range(len(predictors))]
    for thread in threads:
        thread.start()
    for item in loader:
        if "error" in item:
            print(f"series not prepared: {item['study']} ({item['error']})", flush=True)
            continue
        todo.put(item)
    for _ in threads:
        todo.put(None)
    for thread in threads:
        thread.join()
    for error in errors:
        print("series not predicted:", error, flush=True)
    wall = time.time() - start
    print(f"{sum(counts):,} series in {wall / 60:.1f} min; per GPU: series {counts}, busy {[f'{b / wall:.0%}' for b in busy]}", flush=True)
    if not parts:
        return pd.DataFrame(columns=["study", "image", "p", "main"]), series_p
    return pd.concat(parts, ignore_index=True).drop_duplicates("image"), series_p


def submission(test: pd.DataFrame, predicted: pd.DataFrame, label_rates: dict, other_stack: float, series_p: dict | None = None) -> pd.DataFrame:
    """The rows of the submission (id, label) for every slice and series of `test` (test.csv: StudyInstanceUID,
    SOPInstanceUID), after the consistency fix. Slices without a prediction get `other_stack`, series without one (or
    all, without `series_p`) the label rates; a series with no slice that may be raised lets the fix raise any of its
    slices."""
    table = test[["StudyInstanceUID", "SOPInstanceUID"]].drop_duplicates("SOPInstanceUID").merge(
        predicted.rename(columns={"study": "StudyInstanceUID", "image": "SOPInstanceUID"}), how="left", on=["StudyInstanceUID", "SOPInstanceUID"])
    studies, image_series = np.unique(table["StudyInstanceUID"].to_numpy(dtype="U"), return_inverse=True)
    p_image = table["p"].fillna(other_stack).to_numpy(float)
    can_raise = np.array(table["main"].eq(True), dtype=bool)
    has_raisable = np.bincount(image_series, weights=can_raise, minlength=len(studies)) > 0
    can_raise |= ~has_raisable[image_series]
    p_series = np.tile([label_rates[label] for label in SERIES_LABELS], (len(studies), 1))
    for i, study in enumerate(studies):
        if series_p and study in series_p:
            p_series[i] = series_p[study]
    fixed_series, fixed_image = make_consistent(p_series, image_series, p_image, can_raise)
    series_rows = pd.DataFrame({"id": [f"{s}_{label}" for s in studies for label in SERIES_LABELS], "label": fixed_series.ravel()})
    return pd.concat([series_rows, pd.DataFrame({"id": table["SOPInstanceUID"], "label": fixed_image})], ignore_index=True)
