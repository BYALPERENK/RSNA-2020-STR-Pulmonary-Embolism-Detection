"""Calibrate the slice models for a submission without a series model: one logistic regression per variant (06).

Usage: python scripts/fit_slice_ensemble.py

The six variants: SE-ResNeXt101 or ConvNeXt Tiny or both, the fold 0 models alone or all five folds (the test gets the
mean of the five logits of a backbone). Each variant's slice logit is a + b . (logits of its backbones), fitted on the
out-of-fold logits of the test-like series with the metric's slice weight (q_i: only the slices of series with PE count),
so one fit gives the level correction and the ensemble weights. The series labels are the label rates of the training
series, and the consistency fix follows (`pe.consistency`).

The score of each variant is measured out of sample: the five-fold variants fit on four folds and score the fifth, the
fold-0 variants fit on one half of fold 0's series and score the other. Beside it: the raw logits (mean of the
backbones) and the same with only a fitted constant. Writes outputs/submission/ensemble.json (the fits on all the data,
the label rates) and prints the scores.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from pe.data.prepare import SERIES_LABELS, read_series, read_slices
from pe.metrics import image_weights, log_losses
from pe.paths import load_paths
from pe.series_training import fixed_losses
from pe.validation import OTHER_STACK, SEED, validation_fold

RUNS = {"seresnext101": [f"s11_seresnext101_f{k}_s1" for k in range(5)],
        "convnext_tiny": ["s8e2_512_convnext_tiny_s1", *[f"s12_convnext_tiny_f{k}_s1" for k in range(1, 5)]]}
VARIANTS = {f"{name}_{folds}": (backbones, folds) for folds in ("f0", "5folds")
            for name, backbones in (("se", ["seresnext101"]), ("convnext", ["convnext_tiny"]), ("both", ["seresnext101", "convnext_tiny"]))}


def fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, slope: bool = True) -> np.ndarray:
    """[a, b...]: the weighted logistic regression y ~ a + x . b (only `a`, with b fixed at 1 / columns, if not `slope`)."""
    design = np.column_stack([np.ones(len(x)), x])
    def loss(theta):
        params = theta if slope else np.r_[theta, np.full(x.shape[1], 1 / x.shape[1])]
        p = expit(design @ params)
        gradient = design.T @ (w * (p - y)) / w.sum()
        return (w * log_losses(y, p)).sum() / w.sum(), gradient if slope else gradient[:1]
    start = np.r_[0.0, np.full(x.shape[1], 1 / x.shape[1])] if slope else np.zeros(1)
    theta = minimize(loss, start, jac=True, method="L-BFGS-B").x
    return theta if slope else np.r_[theta, np.full(x.shape[1], 1 / x.shape[1])]


def main():
    paths = load_paths()
    slices, series = read_slices(paths["processed_data_dir"] / "slices.csv"), read_series(paths["processed_data_dir"] / "series.csv")
    folds = {k: validation_fold(slices, series, k) for k in range(5)}
    # Per fold: the logits of its own models on its main-stack slices (fold order), the slice labels and metric weights there
    data = {}
    for k, fold in folds.items():
        keys = slices.loc[slices["series"].isin(series.loc[(series["split"] == "train") & (series["fold"] == k), "series"]) & slices["main"], ["series", "image"]]
        for name in RUNS:
            saved = pd.read_csv(paths["outputs_dir"] / "runs" / RUNS[name][k] / "val.csv", dtype=str)
            assert saved.reset_index(drop=True).equals(keys.astype(str).reset_index(drop=True)), f"{RUNS[name][k]}: the saved logits are not fold {k}'s main-stack slices in order"
        logits = np.column_stack([np.load(paths["outputs_dir"] / "runs" / RUNS[name][k] / "val_logits_e1.npy") for name in RUNS])
        weight = (image_weights(fold.image_series, fold.y_image) * fold.test_like[fold.image_series])[fold.main]
        data[k] = (logits, fold.y_image[fold.main], weight, fold.image_series[fold.main])

    def score(k, columns, params, rows=None):
        """Loss and weight (test-like series) of fold k with these calibrated slices; `rows`: only those series."""
        fold, logits = folds[k], data[k][0][:, columns]
        p_image = np.full(len(fold.y_image), OTHER_STACK)
        p_image[fold.main] = expit(params[0] + logits @ params[1:])
        losses, weights, _ = fixed_losses(fold, fold.p_series, p_image)
        keep = fold.test_like if rows is None else fold.test_like & rows
        return losses[keep].sum(), weights[keep].sum()

    def stacked(ks, columns, series_rows=None):
        x = np.vstack([data[k][0][:, columns] for k in ks])
        y, w = (np.concatenate([data[k][i] for k in ks]) for i in (1, 2))
        if series_rows is not None:
            w = w * np.concatenate([series_rows[k][data[k][3]] for k in ks])
        return x, y, w

    halves = np.random.default_rng(SEED).permutation(len(folds[0].y_series)) % 2 == 0
    names = list(RUNS)
    results, out = [], {"label_rates": dict(zip(SERIES_LABELS, series.loc[series["split"] == "train", SERIES_LABELS].astype(float).mean().round(6))),
                        "other_stack": OTHER_STACK, "runs": RUNS, "variants": {}}
    for variant, (backbones, which) in VARIANTS.items():
        columns = [names.index(b) for b in backbones]
        row = {"variant": variant}
        for method, kwargs in (("raw", None), ("level only", {"slope": False}), ("regression", {"slope": True})):
            total = np.zeros(2)
            if which == "5folds":
                for k in range(5):
                    params = np.r_[0.0, np.full(len(columns), 1 / len(columns))] if kwargs is None else fit(*stacked([j for j in range(5) if j != k], columns), **kwargs)
                    total += score(k, columns, params)
            else:
                for half in (halves, ~halves):
                    params = np.r_[0.0, np.full(len(columns), 1 / len(columns))] if kwargs is None else fit(*stacked([0], columns, {0: ~half}), **kwargs)
                    total += score(0, columns, params, half)
            row[method] = total[0] / total[1]
        params = fit(*stacked(range(5) if which == "5folds" else [0], columns))
        row["fitted on all"] = " + ".join([f"{params[0]:.3f}"] + [f"{p:.3f} x {b}" for p, b in zip(params[1:], backbones)])
        results.append(row)
        out["variants"][variant] = {"backbones": backbones, "folds": [0] if which == "f0" else list(range(5)),
                                    "intercept": float(params[0]), "weights": [float(p) for p in params[1:]]}
    target = paths["outputs_dir"] / "submission"
    target.mkdir(parents=True, exist_ok=True)
    (target / "ensemble.json").write_text(json.dumps(out, indent=1))
    print(pd.DataFrame(results).set_index("variant").round(4).to_string())


if __name__ == "__main__":
    main()
