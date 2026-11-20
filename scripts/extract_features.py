"""Run the slice models of one fold over every main-stack slice of train and test: features and logits (05_series_models).

Usage: python scripts/extract_features.py --fold K [--batch 64] [--workers 12] [--limit N]

The series model of fold K is trained on what the slice models of fold K give for the series of the other folds, and
scored on fold K, which those slice models never saw: in-sample features for training, out-of-fold ones for validation,
and the test series through the same pair. Both final slice models of 04 read the same input (one slice, windows pe,
mediastinal and lung, the series' lung box, 512 px), so one pass over the store feeds both. The models run as in
validation, under bf16 autocast.

Writes outputs/features/index.csv (series, image of every main-stack slice of train and test, in slices.csv order) and,
per model, outputs/features/<model>/f<K>.npy (features, float16) and logits_f<K>.npy (float32). Files are written under a
temporary name and renamed at the end, so a fold whose files exist is complete and is skipped. Before that, the logits
of the fold's validation slices are compared with the ones 04 saved.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pe.data.dataset import SliceDataset, neighbour_images, series_boxes
from pe.data.prepare import read_series, read_slices
from pe.data.windows import WINDOWS
from pe.models.slice_model import SliceModel
from pe.paths import load_paths
from pe.training import to_windows

STORE, SIZE, CHANNELS, CROP = "png16", 512, [0], "lung"
WINDOW_NAMES = ["pe", "mediastinal", "lung"]
# The final slice models of 04 (section 10): the run of each fold, whose epoch-1 weights count
RUNS = {"seresnext101": [f"s11_seresnext101_f{k}_s1" for k in range(5)],
        "convnext_tiny": ["s8e2_512_convnext_tiny_s1", *[f"s12_convnext_tiny_f{k}_s1" for k in range(1, 5)]]}


def worker_setup(_):
    import cv2
    cv2.setNumThreads(1)  # one core per worker


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)  # at most 12, always
    parser.add_argument("--limit", type=int, help="the first N slices only (smoke test); nothing is renamed into place")
    args = parser.parse_args()

    paths = load_paths()
    processed, out = paths["processed_data_dir"], paths["outputs_dir"] / "features"
    out.mkdir(parents=True, exist_ok=True)
    slices, series = read_slices(processed / "slices.csv"), read_series(processed / "series.csv")
    table = slices[slices["main"]].reset_index(drop=True)  # train and test, slices.csv order
    if not (out / "index.csv").exists():
        table[["series", "image"]].to_csv(out / "index.csv.tmp", index=False)
        os.replace(out / "index.csv.tmp", out / "index.csv")
    assert pd.read_csv(out / "index.csv", dtype=str).equals(table[["series", "image"]].astype(str)), "index.csv does not match slices.csv"
    if not args.limit and all((out / name / f"{stem}{args.fold}.npy").exists() for name in RUNS for stem in ("f", "logits_f")):
        print(f"fold {args.fold}: already written", flush=True)
        return

    # Each validation file of 04 must hold exactly the fold's main-stack slices, so that the check below means something
    train = series[series["split"] == "train"]
    fold_keys = table.loc[table["series"].isin(train.loc[train["fold"] == args.fold, "series"]), ["series", "image"]].astype(str)
    for runs in RUNS.values():
        saved = pd.read_csv(paths["outputs_dir"] / "runs" / runs[args.fold] / "val.csv", dtype=str)
        assert saved.reset_index(drop=True).equals(fold_keys.reset_index(drop=True)), f"{runs[args.fold]} did not validate on fold {args.fold}"

    # The boxes are the union over the main-stack slices of each series, as in training (before --limit cuts a series)
    boxes = series_boxes(table, CROP, SIZE)
    assert np.isfinite(boxes).all() and (boxes[:, 1] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 2]).all(), "a series has no lung box"
    if args.limit:
        table, boxes = table.iloc[:args.limit], boxes[:args.limit]
    dataset = SliceDataset(processed / STORE, table["series"].to_numpy(dtype="U"), neighbour_images(table, CHANNELS),
                           np.zeros(len(table), np.float32), boxes.astype(int))
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=args.workers, pin_memory=True, worker_init_fn=worker_setup)
    windows = [WINDOWS[name] for name in WINDOW_NAMES]
    models, files = {}, {}
    for name, runs in RUNS.items():
        model = SliceModel(name, in_chans=len(CHANNELS) * len(windows), pretrained=False)
        model.load_state_dict(torch.load(paths["models_dir"] / f"{runs[args.fold]}_e1.pt", map_location="cpu"))
        models[name] = model.cuda().eval().to(memory_format=torch.channels_last)
        (out / name).mkdir(exist_ok=True)
        files[name] = (np.lib.format.open_memmap(out / name / f"f{args.fold}.tmp.npy", mode="w+", dtype=np.float16, shape=(len(table), model.net.num_features)),
                       np.lib.format.open_memmap(out / name / f"logits_f{args.fold}.tmp.npy", mode="w+", dtype=np.float32, shape=(len(table),)))

    start = 0
    with torch.inference_mode():
        for stored, _ in tqdm(loader, desc=f"fold {args.fold}", mininterval=60, ascii=True):
            x = to_windows(stored.cuda(non_blocking=True), windows).contiguous(memory_format=torch.channels_last)
            for name, model in models.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, features = model(x)
                files[name][0][start:start + len(x)] = features.float().cpu().numpy().astype(np.float16)
                files[name][1][start:start + len(x)] = logits.float().cpu().numpy()
            start += len(x)
    assert start == len(table)

    # The fold's validation slices against the logits 04 saved
    position = pd.Series(np.arange(len(table)), index=pd.MultiIndex.from_frame(table[["series", "image"]].astype(str)))
    for name, runs in RUNS.items():
        saved = pd.read_csv(paths["outputs_dir"] / "runs" / runs[args.fold] / "val.csv", dtype=str)
        rows = position.reindex(pd.MultiIndex.from_frame(saved))
        found = rows.notna().to_numpy()
        difference = np.abs(files[name][1][rows[found].to_numpy(dtype=int)] - np.load(paths["outputs_dir"] / "runs" / runs[args.fold] / "val_logits_e1.npy")[found])
        assert args.limit or found.all(), f"{name}: {int((~found).sum())} validation slices missing"
        if difference.size:
            print(f"{name} fold {args.fold}: {found.sum():,} validation slices, logits differ from 04 by {difference.mean():.4f} on average, {difference.max():.4f} at most", flush=True)
            assert difference.max() < 0.25, f"{name}: the logits do not match 04"
    if args.limit:
        return
    for pair in files.values():
        for memmap in pair:
            memmap.flush()
    del files, pair, memmap  # Windows cannot rename a file that is still mapped
    for name in RUNS:
        for stem in (f"f{args.fold}", f"logits_f{args.fold}"):
            os.replace(out / name / f"{stem}.tmp.npy", out / name / f"{stem}.npy")
    print(f"fold {args.fold}: {len(table):,} slices ({train['fold'].eq(args.fold).sum():,} validation series) written to {out}", flush=True)


if __name__ == "__main__":
    main()
