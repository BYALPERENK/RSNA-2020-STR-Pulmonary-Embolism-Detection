"""Data, loss, prediction and scores of the series model (05_series_models).

A series is the sequence of its main-stack slices in body order, one row of features per slice. The loss is the
competition metric on the batch: the nine series labels with their weights plus every slice with 0.0736 x q_i, summed and
divided by the summed weights. The slices outside the main stack are not predicted; the scores give them `OTHER_STACK`.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit

from pe.consistency import make_consistent, violations
from pe.data.prepare import IMAGE_LABEL, SERIES_LABELS
from pe.metrics import IMAGE_WEIGHT, SERIES_WEIGHTS, image_weights, log_losses, series_losses
from pe.validation import OTHER_STACK, ValidationFold


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


def series_data(table: pd.DataFrame, series: pd.DataFrame, features: np.ndarray) -> SeriesData:
    """`table`: every slice of `series` (all stacks); `features`: one row per main-stack slice of `table`, in table order."""
    table = table.reset_index(drop=True)
    main = np.flatnonzero(table["main"].to_numpy())
    feature_row = pd.Series(np.arange(len(main)), index=main)
    by_series = table.iloc[main].groupby("series", sort=False).indices
    labels = table[IMAGE_LABEL].astype(float)
    q = labels.groupby(table["series"]).mean()
    out = SeriesData(series["series"].to_numpy(dtype="U"), [], [], [], q.reindex(series["series"]).fillna(0).to_numpy(), series[SERIES_LABELS].astype(float).to_numpy())
    slice_number = table["slice"].to_numpy()
    for name in out.names:
        rows = main[by_series[name]]
        rows = rows[np.argsort(slice_number[rows], kind="stable")]
        out.rows.append(rows)
        out.features.append(features[feature_row[rows].to_numpy()].astype(np.float32))
        out.y_image.append(labels.to_numpy()[rows].astype(np.float32))
    return out


def with_neighbours(x: torch.Tensor) -> torch.Tensor:
    """[f, f - previous, f - next] along the slices (1st place); at the ends the slice is its own neighbour. Padding gives
    junk differences only at a series' last real slice and beyond, so the lengths must be applied before (see `batch`)."""
    previous = torch.cat([x[:, :1], x[:, :-1]], dim=1)
    following = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
    return torch.cat([x, x - previous, x - following], dim=2)


def batch(data: SeriesData, items, flip: np.ndarray | None = None, neighbours: bool = True, device="cuda") -> dict:
    """A padded batch of the given series; `flip` (one bool per item) reverses a series along the body."""
    features = [data.features[i][::-1] if flip is not None and flip[k] else data.features[i] for k, i in enumerate(items)]
    labels = [data.y_image[i][::-1] if flip is not None and flip[k] else data.y_image[i] for k, i in enumerate(items)]
    lengths = torch.tensor([len(f) for f in features])
    x = torch.zeros(len(items), int(lengths.max()), features[0].shape[1])
    y = torch.zeros(len(items), int(lengths.max()))
    for k, (f, l) in enumerate(zip(features, labels)):
        x[k, :len(f)] = torch.from_numpy(np.ascontiguousarray(f))
        if neighbours:  # the last real slice is its own next neighbour, not the padding after it
            x[k, len(f):] = x[k, len(f) - 1]
        y[k, :len(l)] = torch.from_numpy(np.ascontiguousarray(l))
    if neighbours:
        x = with_neighbours(x)
    return {"x": x.to(device), "lengths": lengths, "y_image": y.to(device), "q": torch.from_numpy(data.q[items]).float().to(device),
            "y_series": torch.from_numpy(data.y_series[items]).float().to(device)}


def series_loss(slice_logits: torch.Tensor, series_logits: torch.Tensor, b: dict) -> torch.Tensor:
    """The competition metric on the batch (module docstring)."""
    mask = torch.arange(slice_logits.shape[1], device=slice_logits.device)[None] < b["lengths"].to(slice_logits.device)[:, None]
    series_weights = torch.as_tensor(SERIES_WEIGHTS, dtype=torch.float32, device=slice_logits.device)
    series_term = (F.binary_cross_entropy_with_logits(series_logits, b["y_series"], reduction="none") * series_weights).sum()
    slice_weights = IMAGE_WEIGHT * b["q"][:, None] * mask
    slice_term = (F.binary_cross_entropy_with_logits(slice_logits, b["y_image"], reduction="none") * slice_weights).sum()
    return (series_term + slice_term) / (len(series_logits) * series_weights.sum() + slice_weights.sum())


@torch.no_grad()
def predict(model, data: SeriesData, batch_size: int = 64, neighbours: bool = True) -> tuple[list, np.ndarray]:
    """Slice logits (one array per series, body order) and series logits (series x 9)."""
    model.eval()
    slice_logits, series_logits = [], []
    for start in range(0, len(data.names), batch_size):
        items = np.arange(start, min(start + batch_size, len(data.names)))
        b = batch(data, items, neighbours=neighbours)
        s, e = model(b["x"], b["lengths"])
        slice_logits += [row[:n].float().cpu().numpy() for row, n in zip(s, b["lengths"])]
        series_logits.append(e.float().cpu().numpy())
    return slice_logits, np.concatenate(series_logits)


def series_scores(fold: ValidationFold, data: SeriesData, slice_logits: list, series_logits: np.ndarray) -> dict:
    """Test-like and full scores after the consistency fix (and before it), the slices' share of the test-like score and
    the number of series that broke a rule before the fix. `data` must be built from the fold's slice table and series."""
    p_image = np.full(len(fold.y_image), OTHER_STACK)
    for rows, logits in zip(data.rows, slice_logits):
        p_image[rows] = expit(logits)
    p_series = expit(series_logits)
    args = fold.y_series, p_series, fold.image_series, fold.y_image
    loss, weight = series_losses(*args, p_image)
    fixed_series, fixed_image = make_consistent(p_series, fold.image_series, p_image, can_raise=fold.main)
    fixed_loss, _ = series_losses(fold.y_series, fixed_series, fold.image_series, fold.y_image, fixed_image)
    like = fold.test_like
    slice_part = np.bincount(fold.image_series, weights=image_weights(fold.image_series, fold.y_image) * log_losses(fold.y_image, fixed_image), minlength=len(like))
    return {"test-like": float(fixed_loss[like].sum() / weight[like].sum()), "full": float(fixed_loss.sum() / weight.sum()),
            "test-like before the fix": float(loss[like].sum() / weight[like].sum()), "full before the fix": float(loss.sum() / weight.sum()),
            "slices' share": float(slice_part[like].sum() / fixed_loss[like].sum()),
            "violations before the fix": int(violations(p_series, fold.image_series, p_image).sum())}
