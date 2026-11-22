"""Predict the test series and write a submission file (06). The same steps as the Kaggle notebook of the submissions.

Usage: python scripts/predict_test.py [--variant final_5folds] [--limit N] [--workers 4] [--out PATH]

Variants: with the series models of 05 (SE-ResNeXt101 embedding [f], SE-ResNeXt101 logit, ConvNeXt Tiny embedding [f])
`final_f0` (fold 0) and `final_5folds`, every model on every GPU; the slice models alone `se_f0`, `convnext_f0`,
`both_f0`, `se_5folds`, `convnext_5folds`, `both_5folds`, the models spread over the GPUs.

Reads raw_data_dir/test/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm, test.csv and sample_submission.csv, the slice
models in models_dir, the series runs in outputs/series_runs/ and outputs/submission/ensemble.json
(scripts/fit_slice_ensemble.py). Series and slices without a prediction (not on disk, beyond --limit, or not readable)
get the label rates and `other_stack` (0.001) before the consistency fix.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from pe.inference import SeriesPredictor, SliceEnsemble, predict_series, predict_slices, submission
from pe.paths import load_paths

SERIES_MODELS = ("s7_noneigh_f{fold}_s1", "s6_selogit_f{fold}_s1", "s7_cnxemb_nn_f{fold}_s1")  # the final of 05
SERIES_VARIANTS = {"final_f0": [0], "final_5folds": [0, 1, 2, 3, 4]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default="final_5folds")
    parser.add_argument("--limit", type=int, help="first N test series only (a timing test)")
    parser.add_argument("--workers", type=int, default=4, help="processes that read the DICOM files")
    parser.add_argument("--out", help="default: outputs/submission/submission_<variant>.csv")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    paths = load_paths()
    raw_dir, outputs = paths["raw_data_dir"], paths["outputs_dir"]
    ensemble = json.loads((outputs / "submission" / "ensemble.json").read_text())
    if args.variant not in SERIES_VARIANTS and args.variant not in ensemble["variants"]:
        parser.error(f"unknown variant {args.variant}: one of {[*SERIES_VARIANTS, *ensemble['variants']]}")
    files = {b: {k: paths["models_dir"] / f"{run}_e1.pt" for k, run in enumerate(runs)} for b, runs in ensemble["runs"].items()}
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if not devices:
        raise SystemExit("no CUDA GPU found")

    studies = sorted(d for d in (raw_dir / "test").iterdir() if d.is_dir())[:args.limit]
    if args.variant in SERIES_VARIANTS:
        runs = [[outputs / "series_runs" / name.format(fold=k) for k in range(5)] for name in SERIES_MODELS]
        predictors = [SeriesPredictor(files, runs, SERIES_VARIANTS[args.variant], device) for device in devices]
        predicted, series_p = predict_series(studies, predictors, workers=args.workers)
    else:
        model = SliceEnsemble(ensemble["variants"][args.variant], files, devices)
        predicted, series_p = predict_slices(studies, model, workers=args.workers), None

    rows = submission(pd.read_csv(raw_dir / "test.csv", dtype=str), predicted, ensemble["label_rates"], ensemble["other_stack"], series_p)
    sample = pd.read_csv(raw_dir / "sample_submission.csv", dtype={"id": str})
    if not (rows["id"].is_unique and set(rows["id"]) == set(sample["id"])):
        raise SystemExit("the rows do not match sample_submission.csv")
    if not np.all((rows["label"] >= 0) & (rows["label"] <= 1)):  # also false for NaN
        raise SystemExit("a label is not a probability")
    out = args.out or outputs / "submission" / f"submission_{args.variant}.csv"
    rows.set_index("id").loc[sample["id"]].reset_index().to_csv(out, index=False)
    found = predicted[predicted["p"].notna()]
    print(f"{len(studies):,} series on disk; predicted: {found['study'].nunique():,} series, {len(found):,} slices; {len(rows):,} rows -> {out}")


if __name__ == "__main__":
    main()
