"""The label consistency rules of the competition, and the smallest change of the predictions that satisfies them (05_series_models).

A label counts as predicted when its probability is above 0.5, and a series counts as positive when any of its slices is.
- A positive series needs exactly one RV/LV label, at least one location label, at most one of the two chronic labels, and
  neither negative_exam_for_pe nor indeterminate.
- A negative series needs exactly one of negative_exam_for_pe and indeterminate, and none of the other seven labels.

`make_consistent` builds both versions of every series, a consistent positive one and a consistent negative one. It keeps
the one that changes the predictions least, measured as the expected rise of the weighted loss if the original
predictions were the true probabilities (1st place's rule); inside a label group (RV/LV, location, chronic) the same cost
picks which label stays above 0.5. A changed value moves just past 0.5.

`p_series` has shape (n_series, 9) in the order of `SERIES_LABELS`; `image_series` gives the row of each slice's series.
"""
import numpy as np

from pe.data.prepare import LOCATION, RVLV, SERIES_LABELS, TYPE
from pe.metrics import EPS, IMAGE_WEIGHT, SERIES_WEIGHTS

LOW, HIGH = 0.4999, 0.5001
NEGATIVE, INDETERMINATE = SERIES_LABELS.index("negative_exam_for_pe"), SERIES_LABELS.index("indeterminate")
LOC, RV, CHRONIC = ([SERIES_LABELS.index(label) for label in labels] for labels in (LOCATION, RVLV, TYPE))
PE_LABELS = LOC + RV + CHRONIC


def violations(p_series: np.ndarray, image_series: np.ndarray, p_image: np.ndarray) -> np.ndarray:
    """True for every series that breaks a rule."""
    on = np.asarray(p_series) > 0.5
    positive = np.zeros(len(on), bool)
    np.logical_or.at(positive, image_series, np.asarray(p_image) > 0.5)
    good_positive = (on[:, RV].sum(1) == 1) & on[:, LOC].any(1) & (on[:, CHRONIC].sum(1) <= 1) & ~on[:, NEGATIVE] & ~on[:, INDETERMINATE]
    good_negative = (on[:, NEGATIVE] != on[:, INDETERMINATE]) & ~on[:, PE_LABELS].any(1)
    return np.where(positive, ~good_positive, ~good_negative)


def lower(p):
    """Values above 0.5 moved to just below it; the others kept."""
    return np.where(p > 0.5, LOW, p)


def lift(p):
    """Values at 0.5 or below moved to just above it; the others kept."""
    return np.where(p > 0.5, p, HIGH)


def change_cost(p: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Expected rise of the log loss when the prediction p is replaced by `new`, if p is the true probability."""
    p, new = np.clip(p, EPS, 1 - EPS), np.clip(new, EPS, 1 - EPS)
    return p * np.log(p / new) + (1 - p) * np.log((1 - p) / (1 - new))


def keep_cheapest(p: np.ndarray, columns: list[int], chosen, others) -> np.ndarray:
    """The columns of a label group changed into the cheapest of its candidates: candidate c passes column c through
    `chosen` and the other columns through `others`; the cost is weighted by each label's metric weight."""
    group = p[:, columns]
    k = len(columns)
    candidates = np.stack([np.where(np.eye(k, dtype=bool)[c], chosen(group), others(group)) for c in range(k)], axis=1)  # series x candidate x column
    costs = (change_cost(group[:, None, :], candidates) * SERIES_WEIGHTS[columns]).sum(2)
    return candidates[np.arange(len(p)), costs.argmin(1)]


def exactly_one(p, columns):
    """Exactly one label of the group above 0.5."""
    return keep_cheapest(p, columns, lift, lower)


def at_least_one(p, columns):
    """At least one label of the group above 0.5."""
    return keep_cheapest(p, columns, lift, lambda g: g)


def at_most_one(p, columns):
    """At most one label of the group above 0.5."""
    return keep_cheapest(p, columns, lambda g: g, lower)


def make_consistent(p_series: np.ndarray, image_series: np.ndarray, p_image: np.ndarray, can_raise: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Consistent copies of `p_series` and `p_image` (module docstring). Only slices with `can_raise` (by default all; in
    practice the main-stack slices, which the model predicts) may be raised above 0.5."""
    p_series, p_image, image_series = np.asarray(p_series, float), np.asarray(p_image, float), np.asarray(image_series)
    can_raise = np.ones(len(p_image), bool) if can_raise is None else np.asarray(can_raise, bool)
    n = len(p_series)
    assert (np.bincount(image_series[can_raise], minlength=n) > 0).all(), "every series needs a slice that may be raised"
    # The weight of each slice in the metric, with the share of positive slices estimated by the mean prediction
    slice_weight = IMAGE_WEIGHT * (np.bincount(image_series, weights=p_image, minlength=n) / np.bincount(image_series, minlength=n))[image_series]

    positive = p_series.copy()
    positive[:, [NEGATIVE, INDETERMINATE]] = lower(positive[:, [NEGATIVE, INDETERMINATE]])
    positive[:, RV] = exactly_one(p_series, RV)
    positive[:, LOC] = at_least_one(p_series, LOC)
    positive[:, CHRONIC] = at_most_one(p_series, CHRONIC)
    # A positive series whose slices are all at 0.5 or below gets one slice raised: the highest that may be, the cheapest
    order = np.lexsort((np.where(can_raise, p_image, -1.0), image_series))  # by series, then by prediction
    last = np.r_[image_series[order][1:] != image_series[order][:-1], True]
    top = np.full(n, -1)
    top[image_series[order][last]] = order[last]
    raise_top = np.bincount(image_series, weights=p_image > 0.5, minlength=n) == 0
    positive_image_cost = np.where(raise_top, slice_weight[top] * change_cost(p_image[top], HIGH), 0)

    negative = p_series.copy()
    negative[:, PE_LABELS] = lower(negative[:, PE_LABELS])
    negative[:, [NEGATIVE, INDETERMINATE]] = exactly_one(p_series, [NEGATIVE, INDETERMINATE])
    negative_image_cost = np.bincount(image_series, weights=slice_weight * change_cost(p_image, lower(p_image)), minlength=n)

    cost = lambda new: change_cost(p_series, new) @ SERIES_WEIGHTS
    keep_positive = cost(positive) + positive_image_cost <= cost(negative) + negative_image_cost
    fixed_series = np.where(keep_positive[:, None], positive, negative)
    fixed_image = np.where(keep_positive[image_series], p_image, lower(p_image))
    raised = top[keep_positive & raise_top]
    fixed_image[raised] = HIGH
    return fixed_series, fixed_image
