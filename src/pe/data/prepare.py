"""The slice and series tables and the image store, as derived step by step in 02_data_preparation (sections 1-7).

`build_tables` gives one row per slice (train and test) with its stack and its place in it (section 3), and one row per
series with the series labels and the fold (sections 1 and 4). `write_slice` writes one slice to the store (section 6)
and returns its pixel statistics. `scripts/prepare_data.py` runs both; 02, section 7, checks the result.
"""
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pe.data.dicom import HU_MIN, read_hu
from pe.data.formats import FORMATS

IMAGE_LABEL = "pe_present_on_image"
LOCATION = ["rightsided_pe", "leftsided_pe", "central_pe"]
RVLV = ["rv_lv_ratio_gte_1", "rv_lv_ratio_lt_1"]
TYPE = ["chronic_pe", "acute_and_chronic_pe"]
SERIES_LABELS = ["negative_exam_for_pe", "indeterminate"] + LOCATION + RVLV + TYPE  # the nine scored series labels
INFO_LABELS = ["qa_motion", "qa_contrast", "flow_artifact", "true_filling_defect_not_pe"]  # in train.csv, not scored
SYNTAX = {"1.2.840.10008.1.2": "implicit VR", "1.2.840.10008.1.2.1": "explicit VR", "1.2.840.10008.1.2.4.70": "JPEG lossless"}
STACK_KEY = ["SliceThickness", "PixelSpacing", "ConvolutionKernel"]  # a change of any of them starts a new stack (02, 3.1)
SAME_POSITION_MM = 0.01
N_FOLDS, SEED = 5, 42
STORE = FORMATS["png16"]
SLICE_DTYPES = {IMAGE_LABEL: "Int8", "gap_prev": float, "gap_next": float}
SERIES_DTYPES = {**{c: "Int8" for c in SERIES_LABELS + INFO_LABELS}, "fold": "Int8"}


# --- labels and headers (02, sections 1-2) ------------------------------------------------------------------------

def load_labels(csv_path) -> tuple[pd.Series, pd.DataFrame]:
    """The slice label per training image, and one row per training series with its 13 series labels."""
    raw = pd.read_csv(csv_path)
    image_labels = raw.set_index("SOPInstanceUID")[IMAGE_LABEL]
    series_labels = raw.groupby("StudyInstanceUID")[SERIES_LABELS + INFO_LABELS].first()
    return image_labels, series_labels


def slice_table(headers: pd.DataFrame) -> pd.DataFrame:
    """One row per slice from the header table: ids, storage group, what separates stacks, and the position along the normal."""
    def numbers(col, n):
        return headers[col].str.split("\\", expand=True).iloc[:, :n].astype(float).to_numpy()

    slices = pd.DataFrame({"image": headers["SOPInstanceUID"], "split": headers["split"], "series": headers["StudyInstanceUID"],
                           "path": headers["path"], "instance": headers["InstanceNumber"].astype(int)})
    slices["writer"] = headers["ImplementationVersionName"].replace("", "(none)")
    # the storage group of 01, 3.3: writer, transfer syntax, bits stored, signedness, intercept (as a number: "-1024" = "-1024.000000")
    slices["group"] = (slices["writer"] + " | " + headers["TransferSyntaxUID"].map(SYNTAX).fillna(headers["TransferSyntaxUID"])
                       + " | " + headers["BitsStored"] + " bit | " + headers["PixelRepresentation"].map({"0": "unsigned", "1": "signed"})
                       + " | " + headers["RescaleIntercept"].astype(float).astype(int).astype(str))
    slices["thickness"] = headers["SliceThickness"].astype(float)
    slices["pixel_spacing"] = numbers("PixelSpacing", 1)[:, 0]
    position = numbers("ImagePositionPatient", 3)
    orientation = numbers("ImageOrientationPatient", 6)
    normal = np.cross(orientation[:, :3], orientation[:, 3:])
    slices["pos"] = (position * normal).sum(axis=1)
    slices["tilt"] = np.degrees(np.arccos(np.clip(np.abs(normal[:, 2]), 0, 1)))
    slices["reversed"] = (orientation[:, 0] < -0.9) | (orientation[:, 4] < -0.9)  # flipped by `orient` (02, section 5)
    # a slice in another plane is its own stack; + 0.0 turns -0.0 (written "-0.000000" in some files) into 0.0
    plane = pd.Series([f"{a:.1f},{b:.1f},{c:.1f}" for a, b, c in normal.round(1) + 0.0], index=slices.index)
    slices["stack_key"] = headers[STACK_KEY].astype(str).agg(" | ".join, axis=1) + " | " + plane
    return slices.sort_values(["series", "instance", "pos"]).reset_index(drop=True)


# --- stacks and positions (02, section 3) -------------------------------------------------------------------------

def stack_ids(keys, pos) -> np.ndarray:
    """Stacks of one series, its slices in InstanceNumber order: a new stack starts where the stack key changes or the
    position turns back (a second pass over the body). A jump forward in the same direction is a gap, not a new stack."""
    out = np.empty(len(pos), dtype=int)
    k, direction = 0, 0.0
    for i in range(len(pos)):
        if i > 0:
            step = pos[i] - pos[i - 1]
            if keys[i] != keys[i - 1] or (direction and step * direction < -SAME_POSITION_MM):
                k, direction = k + 1, 0.0
            elif not direction and abs(step) > SAME_POSITION_MM:
                direction = np.sign(step)
        out[i] = k
    return out


def add_stacks(slices: pd.DataFrame) -> pd.DataFrame:
    """`stack` (numbered in InstanceNumber order), `n_stacks`, `main` (the first axial stack: the one the labels were
    read on, 02, 3.1), and per stack the slices in body order: `slice` (from the lower end), `n_slices`, the distances
    `gap_prev` and `gap_next` (mm), `local_spacing` (median of the five gaps around the slice), `same_position` and
    `wide_gap` (more than 1.5 x the local spacing below the slice)."""
    slices = slices.copy()
    slices["stack"] = np.concatenate([stack_ids(g["stack_key"].to_numpy(), g["pos"].to_numpy())
                                      for _, g in slices.groupby("series", sort=False)])
    slices["n_stacks"] = slices.groupby("series")["stack"].transform("nunique")
    axial = slices[slices["tilt"] < 45]
    main = axial.groupby("series")["stack"].min()
    slices["main"] = slices["stack"] == slices["series"].map(main)
    slices = slices.sort_values(["series", "stack", "pos", "instance"]).reset_index(drop=True)
    by_stack = slices.groupby(["series", "stack"])
    slices["slice"] = by_stack.cumcount()
    slices["n_slices"] = by_stack["image"].transform("size")
    slices["gap_prev"] = by_stack["pos"].diff()
    slices["gap_next"] = -by_stack["pos"].diff(-1)
    slices["local_spacing"] = slices.groupby(["series", "stack"])["gap_prev"].transform(
        lambda g: g.rolling(5, center=True, min_periods=1).median())
    slices["same_position"] = slices["gap_prev"].lt(SAME_POSITION_MM) | slices["gap_next"].lt(SAME_POSITION_MM)
    slices["wide_gap"] = slices["gap_prev"] > 1.5 * slices["local_spacing"]
    return slices


# --- folds (02, section 4) ----------------------------------------------------------------------------------------

def iterative_folds(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Iterative stratification (Sechidis et al. 2011). y: series x labels (0/1); every row needs at least one 1."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(y), -1)
    want = np.full(k, len(y) / k)
    want_label = np.tile(y.sum(axis=0) / k, (k, 1))
    left = np.ones(len(y), bool)
    while left.any():
        remaining = y[left].sum(axis=0).astype(float)
        remaining[remaining == 0] = np.inf
        label = remaining.argmin()
        for i in rng.permutation(np.flatnonzero(left & (y[:, label] == 1))):
            best = np.flatnonzero(want_label[:, label] == want_label[:, label].max())
            best = best[want[best] == want[best].max()]
            f = rng.choice(best)
            fold[i] = f
            want[f] -= 1
            want_label[f] -= y[i]
            left[i] = False
    return fold


def stratification_targets(series: pd.DataFrame) -> pd.DataFrame:
    """What the folds balance per train series: the series labels, positive, the QA labels and the storage group (one-hot)."""
    y = series[SERIES_LABELS + ["qa_motion", "qa_contrast"]].astype(int)
    y["positive"] = 1 - series["negative_exam_for_pe"] - series["indeterminate"]
    return y.join(pd.get_dummies(series["group"]).astype(int))


# --- the tables ---------------------------------------------------------------------------------------------------

def build_tables(headers: pd.DataFrame, image_labels: pd.Series, series_labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The slice table (one row per slice, train and test, ordered by series, stack and position; train rows with their
    slice label) and the series table (one row per series: split, group, number of slices and stacks; train rows with
    the series labels and the fold)."""
    slices = add_stacks(slice_table(headers))
    slices[IMAGE_LABEL] = slices["image"].map(image_labels).astype("Int8")
    series = slices.groupby("series", sort=False).agg(split=("split", "first"), group=("group", "first"), writer=("writer", "first"),
                                                       n_slices=("image", "size"), n_stacks=("n_stacks", "first"))
    series = series.join(series_labels.astype("Int8"))
    train = series[series["split"] == "train"]
    series["fold"] = pd.Series(iterative_folds(stratification_targets(train).to_numpy(), N_FOLDS, SEED), index=train.index).astype("Int8")
    return slices.drop(columns=["stack_key", "group", "writer"]), series.rename_axis("series").reset_index()


def read_slices(path) -> pd.DataFrame:
    """The slice table as written by `scripts/prepare_data.py`, with its column types."""
    return pd.read_csv(path, dtype=SLICE_DTYPES, float_precision="round_trip")


def read_series(path) -> pd.DataFrame:
    """The series table as written by `scripts/prepare_data.py`, with its column types."""
    return pd.read_csv(path, dtype=SERIES_DTYPES, keep_default_na=False, na_values=[""])


# --- the image store ----------------------------------------------------------------------------------------------

def box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Top, bottom, left, right of the True pixels (bottom and right exclusive), or -1s if there are none."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return -1, -1, -1, -1
    return int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)


def pixel_stats(hu: np.ndarray) -> dict:
    """The body: the largest connected region above -300 HU (skin, muscle, bone, contrast), its holes filled. The lungs:
    the pixels below -400 HU inside the body, regions of at least 0.2% of the image. Box and share of the image of both."""
    n, labels, regions, _ = cv2.connectedComponentsWithStats((hu > -300).astype(np.uint8), connectivity=8)
    if n == 1:
        body = np.zeros(hu.shape, bool)
    else:
        body = (labels == 1 + regions[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8)
        # A body wider than the image splits the background; the 1-pixel frame joins every part that reaches the border
        outside = np.pad(body, 1)
        cv2.floodFill(outside, np.zeros((hu.shape[0] + 4, hu.shape[1] + 4), np.uint8), (0, 0), 1)
        body = (body == 1) | (outside[1:-1, 1:-1] == 0)
    n, labels, regions, _ = cv2.connectedComponentsWithStats((body & (hu < -400)).astype(np.uint8), connectivity=8)
    lung = np.isin(labels, 1 + np.flatnonzero(regions[1:, cv2.CC_STAT_AREA] >= 0.002 * hu.size))
    stats = {"body_share": float(body.mean()), "lung_share": float(lung.mean())}
    stats.update(zip(["body_top", "body_bottom", "body_left", "body_right"], box(body)))
    stats.update(zip(["lung_top", "lung_bottom", "lung_left", "lung_right"], box(lung)))
    return stats


def store_path(store_dir, series, image) -> Path:
    return Path(store_dir) / series / f"{image}{STORE.suffix}"


def write_slice(source, out_path) -> dict:
    """Write one slice to the store (unless it is already there) and return its pixel statistics, computed on the stored HU.

    `source` is the DICOM file or its bytes (`scripts/prepare_data.py` reads the files in one process, in disk order).
    The file is written under a temporary name and renamed, so an interrupted run never leaves a partial file behind.
    """
    out_path = Path(out_path)
    if out_path.exists():
        hu = STORE.decode(out_path.read_bytes()).astype(np.int16) + HU_MIN
    else:
        stored = STORE.to_array(read_hu(io.BytesIO(source) if isinstance(source, bytes) else source))
        out_path.parent.mkdir(exist_ok=True)
        temporary = out_path.with_name(out_path.name + ".tmp")
        temporary.write_bytes(STORE.encode(stored))
        os.replace(temporary, out_path)
        hu = stored.astype(np.int16) + HU_MIN
    return pixel_stats(hu)
