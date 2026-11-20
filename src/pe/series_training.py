"""Data, loss, prediction and scores of the series model (05_series_models).

A series is the sequence of its main-stack slices in body order, one row of features per slice. The loss is the
competition metric on the batch: the nine series labels with their weights plus every slice with 0.0736 x q_i, summed and
divided by the summed weights. The slices outside the main stack are not predicted; the scores give them `OTHER_STACK`.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit

from pe.consistency import make_consistent, violations
from pe.data.prepare import IMAGE_LABEL, SERIES_LABELS
from pe.metrics import IMAGE_WEIGHT, SERIES_WEIGHTS, image_weights, log_losses, series_losses
from pe.validation import OTHER_STACK, ValidationFold

INPUTS = ("embedding", "logit", "both")


class SliceFeatures:
    """What the slice models of one fold give for every main-stack slice of train and test (written by
    scripts/extract_features.py): `[rows]` reads the rows of index.csv from the files on demand, the models side by side,
    each with its embedding, its logit or both. Embeddings come as float16, logits alone as float32."""

    def __init__(self, directory: Path, fold: int, backbones: list[str], kind: str):
        assert kind in INPUTS, kind
        self.index = pd.read_csv(directory / "index.csv", dtype=str)
        self.files = [np.load(directory / name / f"{stem}{fold}.npy", mmap_mode="r") for name in backbones
                      for stem, wanted in (("f", kind != "logit"), ("logits_f", kind != "embedding")) if wanted]
        assert all(len(f) == len(self.index) for f in self.files), "the feature files do not match index.csv"
        self.widths = [f.shape[1] if f.ndim == 2 else 1 for f in self.files]
        self.dtype = np.float32 if kind == "logit" else np.float16

    def rows_of(self, table: pd.DataFrame) -> np.ndarray:
        """The index rows of the main-stack slices of `table`, in table order."""
        row = pd.Series(np.arange(len(self.index)), index=pd.MultiIndex.from_frame(self.index[["series", "image"]]))
        wanted = row.reindex(pd.MultiIndex.from_frame(table.loc[table["main"], ["series", "image"]].astype(str)))
        assert wanted.notna().all(), f"{int(wanted.isna().sum())} main-stack slices have no features"
        return wanted.to_numpy(dtype=int)

    def __getitem__(self, rows) -> np.ndarray:
        rows = np.asarray(rows)
        out = np.empty((len(rows), sum(self.widths)), self.dtype)
        for start in range(0, len(rows), 100_000):  # in chunks, so that only the output is held whole
            part, column = rows[start:start + 100_000], 0
            for f, width in zip(self.files, self.widths):
                out[start:start + len(part), column:column + width] = np.asarray(f[part]).reshape(len(part), width)
                column += width
        return out


@dataclass
class SeriesData:
    """Per series: `features` (slices x n_in) and `y_image` of its main-stack slices in body order, `rows` (their positions
    in the slice table it was built from), `q` (share of positive slices among all its slices) and `y_series` (9 labels)."""
    names: np.ndarray
    features: list
    y_image: list
    rows: list
    q: np.ndarray
    y_series: np.ndarray


def series_data(table: pd.DataFrame, series: pd.DataFrame, features, feature_rows: np.ndarray | None = None) -> SeriesData:
    """`table`: every slice of `series` (all stacks); `features`: one row per main-stack slice of `table` in table order,
    or with `feature_rows` (the row of each of them) any array or `SliceFeatures`. The rows are read once for all series
    in body order, and the series' features are views of that array."""
    table = table.reset_index(drop=True)
    main = np.flatnonzero(table["main"].to_numpy())
    feature_row = pd.Series(np.arange(len(main)), index=main)
    by_series = table.iloc[main].groupby("series", sort=False).indices
    labels = table[IMAGE_LABEL].astype(float)
    q = labels.groupby(table["series"]).mean()
    names = series["series"].to_numpy(dtype="U")
    slice_number = table["slice"].to_numpy()
    rows = [main[by_series[name]] for name in names]
    rows = [r[np.argsort(slice_number[r], kind="stable")] for r in rows]
    positions = feature_row[np.concatenate(rows)].to_numpy()
    stacked = features[positions if feature_rows is None else feature_rows[positions]]
    ends = np.cumsum([len(r) for r in rows])[:-1]
    return SeriesData(names, np.split(stacked, ends), [labels.to_numpy()[r].astype(np.float32) for r in rows], rows,
                      q.reindex(names).fillna(0).to_numpy(), series[SERIES_LABELS].astype(float).to_numpy())


def with_neighbours(x: torch.Tensor) -> torch.Tensor:
    """[f, f - previous, f - next] along the slices (1st place); at the ends the slice is its own neighbour. Padding gives
    junk differences only at a series' last real slice and beyond, so the lengths must be applied before (see `batch`)."""
    previous = torch.cat([x[:, :1], x[:, :-1]], dim=1)
    following = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
    return torch.cat([x, x - previous, x - following], dim=2)


def resample(x: torch.Tensor, n: int) -> torch.Tensor:
    """The rows of `x` (slices x columns) linearly resampled to n rows; the first and the last row are kept."""
    return F.interpolate(x.T[None], size=n, mode="linear", align_corners=True)[0].T


def batch(data: SeriesData, items, flip: np.ndarray | None = None, neighbours: bool = True, length: int | None = None, device="cuda") -> dict:
    """A padded batch of the given series; `flip` (one bool per item) reverses a series along the body, and `length`
    resamples every series to that many slices (the labels stay on the series' own slices, see `on_slices`)."""
    order = [slice(None, None, -1) if flip is not None and flip[k] else slice(None) for k in range(len(items))]
    features = [torch.from_numpy(np.ascontiguousarray(data.features[i][o])) for i, o in zip(items, order)]  # as stored (float16)
    labels = [data.y_image[i][o] for i, o in zip(items, order)]
    n_slices = torch.tensor([len(f) for f in features])
    if length:
        features = [resample(f.float(), length) for f in features]
    lengths = torch.tensor([len(f) for f in features])
    x = torch.zeros(len(items), int(lengths.max()), features[0].shape[1], dtype=features[0].dtype)
    y = torch.zeros(len(items), int(n_slices.max()))
    for k, (f, l) in enumerate(zip(features, labels)):
        x[k, :len(f)] = f
        if neighbours:  # the last real slice is its own next neighbour, not the padding after it
            x[k, len(f):] = f[-1]
        y[k, :len(l)] = torch.from_numpy(np.ascontiguousarray(l))
    x = x.to(device).float()  # padded on the CPU at the stored precision, the rest on the device
    if neighbours:
        x = with_neighbours(x)
    return {"x": x, "lengths": lengths, "n_slices": n_slices, "resampled": bool(length), "y_image": y.to(device),
            "q": torch.from_numpy(data.q[items]).float().to(device), "y_series": torch.from_numpy(data.y_series[items]).float().to(device)}


def on_slices(slice_logits: torch.Tensor, b: dict) -> torch.Tensor:
    """The slice logits on each series' own slices (resampled back when the batch was resampled), padded."""
    if not b["resampled"]:
        return slice_logits
    out = slice_logits.new_zeros(len(b["n_slices"]), int(b["n_slices"].max()))
    for k, n in enumerate(b["n_slices"].tolist()):
        out[k, :n] = resample(slice_logits[k, :, None], n)[:, 0]
    return out


def slice_q(q: torch.Tensor | np.ndarray, negative_q: float | None):
    """The q that weighs each series' slices in the loss: its own, or `negative_q` for the series without a positive slice."""
    return q if negative_q is None else (q + (q == 0) * negative_q)


def series_loss(slice_logits: torch.Tensor, series_logits: torch.Tensor, b: dict, negative_q: float | None = None,
                weight_per_series: float | None = None) -> torch.Tensor:
    """The competition metric on the batch (module docstring). With `negative_q`, the slices of the series without PE
    weigh 0.0736 x that constant instead of 0, so they train the slice head too. With `weight_per_series` (the mean summed
    weight of a training series), the denominator is that times the batch size, the same for every batch, so that the
    batches add up to the metric over all training series; without it, the batch's own summed weights."""
    slice_logits = on_slices(slice_logits, b)
    mask = torch.arange(slice_logits.shape[1], device=slice_logits.device)[None] < b["n_slices"].to(slice_logits.device)[:, None]
    series_weights = torch.as_tensor(SERIES_WEIGHTS, dtype=torch.float32, device=slice_logits.device)
    series_term = (F.binary_cross_entropy_with_logits(series_logits, b["y_series"], reduction="none") * series_weights).sum()
    slice_weights = IMAGE_WEIGHT * slice_q(b["q"], negative_q)[:, None] * mask
    slice_term = (F.binary_cross_entropy_with_logits(slice_logits, b["y_image"], reduction="none") * slice_weights).sum()
    total = len(series_logits) * (series_weights.sum() + slice_weights.sum() / len(series_logits) if weight_per_series is None else weight_per_series)
    return (series_term + slice_term) / total


@torch.no_grad()
def predict(model, data: SeriesData, batch_size: int = 64, neighbours: bool = True, length: int | None = None) -> tuple[list, np.ndarray]:
    """Slice logits (one array per series, body order) and series logits (series x 9)."""
    model.eval()
    slice_logits, series_logits = [], []
    for start in range(0, len(data.names), batch_size):
        items = np.arange(start, min(start + batch_size, len(data.names)))
        b = batch(data, items, neighbours=neighbours, length=length)
        s, e = model(b["x"], b["lengths"])
        slice_logits += [row[:n].float().cpu().numpy() for row, n in zip(on_slices(s, b), b["n_slices"])]
        series_logits.append(e.float().cpu().numpy())
    return slice_logits, np.concatenate(series_logits)


def fold_predictions(fold: ValidationFold, data: SeriesData, slice_logits: list, series_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Series and slice probabilities of the fold, before the consistency fix; every slice of the fold, in its order (the
    ones outside the main stack get `OTHER_STACK`). `data` must be built from the fold's slice table and series."""
    p_image = np.full(len(fold.y_image), OTHER_STACK)
    for rows, logits in zip(data.rows, slice_logits):
        p_image[rows] = expit(logits)
    return expit(series_logits), p_image


def fixed_losses(fold: ValidationFold, p_series: np.ndarray, p_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loss and weight of every series after the consistency fix, and the fixed slice probabilities."""
    fixed_series, fixed_image = make_consistent(p_series, fold.image_series, p_image, can_raise=fold.main)
    losses, weights = series_losses(fold.y_series, fixed_series, fold.image_series, fold.y_image, fixed_image)
    return losses, weights, fixed_image


def series_scores(fold: ValidationFold, p_series: np.ndarray, p_image: np.ndarray) -> dict:
    """Test-like and full scores after the consistency fix (and before it), the slices' share of the test-like score and
    the number of series that broke a rule before the fix."""
    loss, weight = series_losses(fold.y_series, p_series, fold.image_series, fold.y_image, p_image)
    fixed_loss, _, fixed_image = fixed_losses(fold, p_series, p_image)
    like = fold.test_like
    slice_part = np.bincount(fold.image_series, weights=image_weights(fold.image_series, fold.y_image) * log_losses(fold.y_image, fixed_image), minlength=len(like))
    return {"test-like": float(fixed_loss[like].sum() / weight[like].sum()), "full": float(fixed_loss.sum() / weight.sum()),
            "test-like before the fix": float(loss[like].sum() / weight[like].sum()), "full before the fix": float(loss.sum() / weight.sum()),
            "slices' share": float(slice_part[like].sum() / fixed_loss[like].sum()),
            "violations before the fix": int(violations(p_series, fold.image_series, p_image).sum())}
