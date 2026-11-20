"""Compare two series-model settings over five folds and several seeds (05_series_models).

Usage: python scripts/compare_series_runs.py A B [--folds 0,1,2,3,4] [--seeds 1,2,3]

A and B are run names in outputs/series_runs with {fold} and {seed} in them, e.g. s1_ref_f{fold}_s{seed}; "X+Y" averages
the probabilities of runs X and Y before the consistency fix (two models side by side). For every seed the five folds
give one out-of-fold score after the fix: test-like (the ranking score), full and on the two rare writers. The decision
uses the seed mean, and the difference B - A with a 95% interval from a paired bootstrap over the test-like series, each
series with its loss averaged over the seeds.
"""
import argparse
import json

import numpy as np
import pandas as pd

from pe.data.prepare import read_series, read_slices
from pe.metrics import paired_bootstrap
from pe.paths import load_paths
from pe.series_training import fixed_losses
from pe.validation import validation_fold


def oof_losses(runs_dir, pattern: str, folds: list[int], seed: int, slices: pd.DataFrame, series: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Loss, weight and test-like flag of every validation series of the folds, after the fix."""
    parts = []
    for fold in folds:
        runs = [runs_dir / name.format(fold=fold, seed=seed) for name in pattern.split("+")]
        names = pd.read_csv(runs[0] / "val.csv", dtype=str)["series"]
        for run in runs:
            config = json.loads((run / "config.json").read_text())
            assert config["fold"] == fold, f"{run.name} was trained on fold {config['fold']}"
            assert pd.read_csv(run / "val.csv", dtype=str)["series"].equals(names), f"{pattern}: the runs of fold {fold} validate on different series"
        train = series[series["split"] == "train"]
        fold_series = train.loc[train["fold"] == fold, "series"].reset_index(drop=True)
        assert names.equals(fold_series.iloc[:len(names)]) and (len(names) == len(fold_series) or config.get("limit")), f"{runs[0].name}: not the whole fold {fold} in its order"
        key = (fold, tuple(names))
        if key not in cache:
            cache[key] = validation_fold(slices, series, fold, names)
        val = cache[key]
        p_series = np.mean([np.load(run / "val_p_series.npy") for run in runs], axis=0)
        p_image = np.mean([np.load(run / "val_p_image.npy") for run in runs], axis=0)
        assert p_series.shape == val.y_series.shape and p_image.shape == val.y_image.shape, f"{runs[0].name}: predictions do not match the fold"
        losses, weights, _ = fixed_losses(val, p_series, p_image)
        parts.append(pd.DataFrame({"series": names, "loss": losses, "weight": weights, "test_like": val.test_like}))
    return pd.concat(parts, ignore_index=True)


def scores(table: pd.DataFrame) -> dict:
    ratio = lambda part: part["loss"].sum() / part["weight"].sum()
    return {"test-like": ratio(table[table["test_like"]]), "full": ratio(table), "rare writers": ratio(table[~table["test_like"]])}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="1,2,3")
    args = parser.parse_args()

    paths = load_paths()
    processed, runs_dir = paths["processed_data_dir"], paths["outputs_dir"] / "series_runs"
    slices, series = read_slices(processed / "slices.csv"), read_series(processed / "series.csv")
    folds, seeds = [int(k) for k in args.folds.split(",")], [int(s) for s in args.seeds.split(",")]
    cache, mean_losses = {}, {}
    for side, pattern in (("A", args.a), ("B", args.b)):
        tables = [oof_losses(runs_dir, pattern, folds, seed, slices, series, cache) for seed in seeds]
        assert all(t["series"].equals(tables[0]["series"]) for t in tables), f"{pattern}: the seeds validate on different series"
        rows = pd.DataFrame({seed: scores(t) for seed, t in zip(seeds, tables)}).T.rename_axis("seed")
        print(f"{side} = {pattern}\n{rows.round(4).to_string()}\n"
              f"mean: " + ", ".join(f"{k} {v:.4f}" for k, v in rows.mean().items()) + "\n", flush=True)
        mean_losses[side] = tables[0].assign(loss=np.mean([t["loss"] for t in tables], axis=0))
    a, b = mean_losses["A"], mean_losses["B"]
    assert a["series"].equals(b["series"]), "A and B validate on different series"
    like = a["test_like"].to_numpy()
    difference = paired_bootstrap(a["loss"][like], b["loss"][like], a["weight"][like])
    print(f"B - A, test-like, seed mean: {difference['mean']:+.4f} [{difference['low']:+.4f}, {difference['high']:+.4f}] "
          f"over {like.sum():,} series ({len(folds)} folds x {len(seeds)} seeds)")


if __name__ == "__main__":
    main()
