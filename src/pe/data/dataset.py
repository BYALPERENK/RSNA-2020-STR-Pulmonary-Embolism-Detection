"""The slices a slice model reads (04_slice_models).

The Dataset class lives here, not in a notebook: on Windows the loader workers must import it. It returns the stored
values; the window is applied on the GPU (`pe.training`).
"""
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from pe.data.prepare import IMAGE_LABEL, SEED, STORE, store_path

BOX_SIZE = 512  # the boxes of slices.csv are in pixels of the 512 px images


def q_weights(table: pd.DataFrame) -> np.ndarray:
    """One weight per row: the share of positive slices in its series, as in the metric, scaled to a mean of 1."""
    q = table[IMAGE_LABEL].astype(float).groupby(table["series"]).transform("mean").to_numpy()
    return q / q.mean()


def positive_slice_targets(table: pd.DataFrame, series: pd.DataFrame, labels: list[str]) -> np.ndarray:
    """The series `labels` of every row, on its positive slices only; 0 on the others."""
    values = series.set_index("series").loc[table["series"], labels].astype(float).to_numpy()
    return values * table[IMAGE_LABEL].astype(float).to_numpy()[:, None]


def neighbour_images(table: pd.DataFrame, offsets=(-1, 0, 1)) -> np.ndarray:
    """The images at `offsets` slices from every row of a slice table in body order, with the rows of a stack together.
    Neighbours stay inside the stack; past its end the last slice stands in."""
    images = table["image"].to_numpy(dtype="U")
    stack = table.groupby(["series", "stack"], sort=False)
    position, size = stack.cumcount().to_numpy(), stack["image"].transform("size").to_numpy()
    first = np.arange(len(table)) - position
    return np.stack([images[first + np.clip(position + offset, 0, size - 1)] for offset in offsets], axis=1)


def series_boxes(table: pd.DataFrame, name: str, store_size: int, margin: float = 0.05) -> np.ndarray:
    """One box per row (top, bottom, left, right in pixels of the store): the union of the `name` boxes ("lung" or "body")
    of the given slices of its series, with `margin` of each side added around it. Slices without a box have -1s."""
    top, bottom, left, right = (f"{name}_{side}" for side in ("top", "bottom", "left", "right"))
    union = table[table[top] >= 0].groupby("series").agg(top=(top, "min"), bottom=(bottom, "max"), left=(left, "min"), right=(right, "max"))
    height, width = union["bottom"] - union["top"], union["right"] - union["left"]
    grown = pd.DataFrame({"top": union["top"] - margin * height, "bottom": union["bottom"] + margin * height,
                          "left": union["left"] - margin * width, "right": union["right"] + margin * width}).clip(0, BOX_SIZE)
    return (grown * store_size / BOX_SIZE).round().astype(int).reindex(table["series"]).to_numpy()


def screening_series(series: pd.DataFrame, share: float, seed: int = SEED) -> np.ndarray:
    """A fixed share of the given series, drawn inside each storage group and positive or not, so that the mix stays the
    same. One order is drawn per seed and each share takes its beginning, so a smaller share lies inside a larger one."""
    rng = np.random.default_rng(seed)
    positive = 1 - series["negative_exam_for_pe"].astype(int) - series["indeterminate"].astype(int)
    kept = []
    for _, part in series.groupby([series["group"], positive], sort=True):
        order = rng.permutation(part["series"].to_numpy(dtype="U"))
        kept.append(order[:round(share * len(order))])
    return np.concatenate(kept)


class SliceDataset(Dataset):
    """One item per row of `images` (n x channels, as from `neighbour_images`): the stored slices as int16, and the label.
    With `boxes` (n x 4, as from `series_boxes`) each slice is cut to its box and stretched back to the size of the store."""

    def __init__(self, store_dir, series: np.ndarray, images: np.ndarray, labels: np.ndarray, boxes: np.ndarray = None):
        self.store_dir, self.series, self.images, self.labels, self.boxes = store_dir, series, images, labels, boxes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        files = [store_path(self.store_dir, self.series[i], image).read_bytes() for image in self.images[i]]
        slices = [STORE.decode(f) for f in files]
        if self.boxes is not None:
            top, bottom, left, right = self.boxes[i]
            slices = [cv2.resize(s[top:bottom, left:right], s.shape[::-1], interpolation=cv2.INTER_LINEAR) for s in slices]
        return np.stack(slices).astype(np.int16), self.labels[i]
