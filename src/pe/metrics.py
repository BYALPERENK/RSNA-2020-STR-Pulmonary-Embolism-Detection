"""The competition metric and the series bootstrap used to compare models (03_metric_and_baselines, sections 1 and 6).

The score is the weighted log loss of the competition: the nine series labels with their weights, plus every slice with
weight 0.0736 x q_i, where q_i is the share of positive slices among all slices of its series. The sum of the losses is
divided by the sum of the weights, N + 0.0736 x (positive slices). Predictions are clipped to [1e-15, 1 - 1e-15]; Kaggle
never published the clip, so this one is an assumption.

`y_series` and `p_series` have shape (n_series, 9) with the columns in the order of `SERIES_LABELS`. `image_series` gives
the row of each slice's series, and `y_image` and `p_image` hold one value per slice, every slice of every series.
"""
import numpy as np

from pe.data.prepare import IMAGE_LABEL, SERIES_LABELS

SERIES_WEIGHTS = np.array([{"negative_exam_for_pe": 12, "indeterminate": 15, "chronic_pe": 17, "acute_and_chronic_pe": 17, "central_pe": 30.6,
                            "leftsided_pe": 10.2, "rightsided_pe": 10.2, "rv_lv_ratio_gte_1": 38.25, "rv_lv_ratio_lt_1": 12.75}[label]
                           for label in SERIES_LABELS]) / 163
IMAGE_WEIGHT = 12 / 163
EPS = 1e-15


def log_losses(y, p) -> np.ndarray:
    """Log loss of every value."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def image_weights(image_series, y_image) -> np.ndarray:
    """The weight 0.0736 x q_i of every slice: 0 in a series without a positive slice."""
    y_image = np.asarray(y_image, dtype=float)
    q = np.bincount(image_series, weights=y_image) / np.bincount(image_series)
    return IMAGE_WEIGHT * q[image_series]


def series_losses(y_series, p_series, image_series, y_image, p_image) -> tuple[np.ndarray, np.ndarray]:
    """Weighted loss and weight of every series, labels and slices together; the score is the ratio of their sums."""
    weights = image_weights(image_series, y_image)
    n = len(y_series)
    image_loss = np.bincount(image_series, weights=weights * log_losses(y_image, p_image), minlength=n)
    return log_losses(y_series, p_series) @ SERIES_WEIGHTS + image_loss, SERIES_WEIGHTS.sum() + np.bincount(image_series, weights=weights, minlength=n)


def score(y_series, p_series, image_series, y_image, p_image) -> float:
    """The competition score."""
    losses, weights = series_losses(y_series, p_series, image_series, y_image, p_image)
    return float(losses.sum() / weights.sum())


def score_by_label(y_series, p_series, image_series, y_image, p_image) -> dict:
    """Mean log loss of each label (slices: weighted by q_i, NaN without a positive slice) and its share of the score;
    the shares add up to the score."""
    weights = image_weights(image_series, y_image)
    total = len(y_series) * SERIES_WEIGHTS.sum() + weights.sum()
    means = log_losses(y_series, p_series).mean(axis=0)
    parts = {label: {"log loss": float(m), "share of the score": float(m * w * len(y_series) / total)}
             for label, m, w in zip(SERIES_LABELS, means, SERIES_WEIGHTS)}
    image_loss = float((weights * log_losses(y_image, p_image)).sum())
    parts[IMAGE_LABEL] = {"log loss": image_loss / weights.sum() if weights.sum() > 0 else float("nan"), "share of the score": image_loss / total}
    return parts


def bootstrap(losses, weights, groups=None, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Ratio of the summed losses and weights with a percentile interval from resampling whole groups with replacement
    (by default every row is its own group).

    Slices of one series are strongly correlated, so resampling slices would give far too narrow an interval.
    """
    losses, weights = np.asarray(losses, dtype=float), np.asarray(weights, dtype=float)
    if groups is not None:
        _, index = np.unique(np.asarray(groups), return_inverse=True)
        losses, weights = np.bincount(index, weights=losses), np.bincount(index, weights=weights)
    rng, uniform = np.random.default_rng(seed), np.full(len(losses), 1 / len(losses))
    batch = max(1, 20_000_000 // len(losses))  # draws per batch, so that memory stays small even with one group per slice
    ratios = []
    for start in range(0, n_boot, batch):
        draws = rng.multinomial(len(losses), uniform, size=min(batch, n_boot - start))
        ratios.append((draws @ losses) / (draws @ weights))
    ratios = np.concatenate(ratios)
    low, high = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(losses.sum() / weights.sum()), "low": float(low), "high": float(high)}


def paired_bootstrap(losses_a, losses_b, weights, groups=None, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Difference of two models' scores (b minus a) on the same series, with an interval from resampling groups."""
    return bootstrap(np.asarray(losses_b, dtype=float) - np.asarray(losses_a, dtype=float), weights, groups, n_boot, seed, alpha)
