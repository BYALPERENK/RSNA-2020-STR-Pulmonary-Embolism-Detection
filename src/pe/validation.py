"""Scoring a slice model on its validation fold (04_slice_models).

The model predicts the slices only. The series labels get the label rates of the other folds, the same for every model
(03, section 3), and the slices outside the main stacks get `OTHER_STACK` (03). Three numbers come out:

- test-like, after a level correction: the number that ranks the models. One constant added to the logits moves the level
  of the predictions without changing their order. It is fitted on one half of the series and applied to the other, in
  both directions, so the fold scores nothing it was fitted on.
- test-like and full: the same predictions without the correction, on the series without and with the two rare writers (03).
- AUROC of the main-stack slices of the test-like series, which measures the order alone, and their plain BCE.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from pe.data.prepare import IMAGE_LABEL, SEED, SERIES_LABELS
from pe.metrics import image_weights, log_losses, series_losses

OTHER_STACK = 0.001  # the prediction for a slice outside the main stack (03, section 3)
RARE_IN_TEST = ["dcm4che-1.4.35", "dcm4che-1.4.37"]  # 2 of the 650 readable test series (03, section 4)


@dataclass
class ValidationFold:
    """Arrays of one fold: per series `y_series`, `p_series`, `test_like`; per slice `image_series`, `y_image`, `main`."""
    y_series: np.ndarray
    p_series: np.ndarray
    test_like: np.ndarray
    image_series: np.ndarray
    y_image: np.ndarray
    main: np.ndarray


def validation_fold(slices: pd.DataFrame, series: pd.DataFrame, fold: int, val_series=None) -> ValidationFold:
    """The fold's series (or `val_series`, its first rows) with the label rates of the other folds."""
    train = series[series["split"] == "train"]
    val = train[train["fold"] == fold] if val_series is None else train[train["series"].isin(val_series)]
    val = val.reset_index(drop=True)
    table = slices[slices["series"].isin(val["series"])]
    y_series = val[SERIES_LABELS].astype(float).to_numpy()
    rates = train.loc[train["fold"] != fold, SERIES_LABELS].astype(float).to_numpy().mean(axis=0)
    return ValidationFold(y_series, np.tile(rates, (len(val), 1)), ~val["writer"].isin(RARE_IN_TEST).to_numpy(),
                          table["series"].map(pd.Series(val.index, index=val["series"])).to_numpy(),
                          table[IMAGE_LABEL].astype(float).to_numpy(), table["main"].to_numpy())


def fit_level(logits: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """The constant that, added to the logits, gives the smallest weighted log loss."""
    return float(minimize_scalar(lambda c: (weights * log_losses(y, expit(logits + c))).sum(), bounds=(-10, 10), method="bounded").x)


def correct_level(logits: np.ndarray, y: np.ndarray, weights: np.ndarray, groups: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, tuple[float, float]]:
    """The logits with the level fitted on the other half of the groups, and the two levels. Only groups with weight count
    for the fit, so they are the ones divided."""
    order = np.random.default_rng(seed).permutation(np.unique(groups[weights > 0]))
    first = np.isin(groups, order[::2])
    level_first, level_second = (fit_level(logits[part], y[part], weights[part]) for part in (first, ~first))
    return logits + np.where(first, level_second, level_first), (level_first, level_second)


def predictions(fold: ValidationFold, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Slice predictions from the logits of the main-stack slices: as they are, and with the level corrected on the
    test-like series (the other series keep their raw values, and only the test-like score reads the corrected ones)."""
    raw = np.full(len(fold.y_image), OTHER_STACK)
    raw[fold.main] = expit(logits)
    scored = fold.main & fold.test_like[fold.image_series]
    weights = image_weights(fold.image_series, fold.y_image)
    in_main = fold.test_like[fold.image_series][fold.main]
    corrected_logits, levels = correct_level(logits[in_main], fold.y_image[scored], weights[scored], fold.image_series[scored])
    corrected = raw.copy()
    corrected[scored] = expit(corrected_logits)
    return raw, corrected, levels


def slice_scores(fold: ValidationFold, logits: np.ndarray) -> dict:
    """The numbers of the module docstring."""
    raw, corrected, levels = predictions(fold, logits)
    args = fold.y_series, fold.p_series, fold.image_series, fold.y_image
    loss, weight = series_losses(*args, raw)
    corrected_loss, _ = series_losses(*args, corrected)
    scored = fold.main & fold.test_like[fold.image_series]
    return {"test-like corrected": float(corrected_loss[fold.test_like].sum() / weight[fold.test_like].sum()),
            "test-like": float(loss[fold.test_like].sum() / weight[fold.test_like].sum()), "full": float(loss.sum() / weight.sum()),
            "AUROC": float(roc_auc_score(fold.y_image[scored], raw[scored])), "BCE": float(log_losses(fold.y_image[scored], raw[scored]).mean()),
            "level": float(np.mean(levels))}
