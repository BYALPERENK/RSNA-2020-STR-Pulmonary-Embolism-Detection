"""Train one series model on one fold and predict its validation fold (05_series_models).

Usage: python scripts/train_series.py --name NAME --features DIR [--fold 0] [--seed 1] [--epochs 25] [--lr 5e-4]
       [--batch 64] [--hidden 256] [--no-neighbours] [--no-flip] [--limit N]

- Features: DIR/index.csv (series, image of every main-stack slice that has features) and DIR/f<fold>.npy (one row per
  index row, from the slice model of that fold), or DIR/all.npy for every fold.
- Training: the series of the other folds in shuffled batches (the last one smaller), the competition metric as loss (`pe.series_training.series_loss`), AdamW with a
  warm-up and cosine schedule, a random reversal along the body for half the series. The last epoch counts.
- Validation: every series of the fold, scored after the consistency fix (`pe.series_training.series_scores`).

Writes outputs/series_runs/NAME/: config.json, log.csv (one row per epoch), val.csv (the validation series),
val_series_logits.npy, val_slice_logits.npz (one array per series, body order) and scores.json.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pe.data.prepare import SERIES_LABELS, read_series, read_slices
from pe.models.series_model import SeriesModel
from pe.paths import load_paths
from pe.series_training import batch, predict, series_data, series_loss, series_scores
from pe.training import learning_rate
from pe.validation import validation_fold


def load_features(directory, fold: int, table: pd.DataFrame) -> np.ndarray:
    """The feature rows of the main-stack slices of `table`, in table order."""
    index = pd.read_csv(directory / "index.csv", dtype=str)
    path = directory / f"f{fold}.npy"
    features = np.load(path if path.exists() else directory / "all.npy", mmap_mode="r")
    row = pd.Series(np.arange(len(index)), index=pd.MultiIndex.from_frame(index[["series", "image"]]))
    main = table[table["main"]]
    wanted = row.reindex(pd.MultiIndex.from_frame(main[["series", "image"]].astype(str)))
    assert wanted.notna().all(), f"{int(wanted.isna().sum())} main-stack slices have no features"
    return np.asarray(features[wanted.to_numpy(dtype=int)], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--features", required=True, help="folder with index.csv and f<fold>.npy or all.npy")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-neighbours", action="store_true", help="the features alone, without the differences to the neighbours")
    parser.add_argument("--no-flip", action="store_true", help="no random reversal along the body")
    parser.add_argument("--limit", type=int, help="keep N series on each side (smoke test)")
    args = parser.parse_args()

    paths = load_paths()
    run_dir = paths["outputs_dir"] / "series_runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    processed = paths["processed_data_dir"]
    slices, series = read_slices(processed / "slices.csv"), read_series(processed / "series.csv")
    train = series[series["split"] == "train"].reset_index(drop=True)
    fit, val = train[train["fold"] != args.fold].reset_index(drop=True), train[train["fold"] == args.fold].reset_index(drop=True)
    if args.limit:
        fit, val = fit.iloc[:args.limit], val.iloc[:args.limit]
    fit_table, val_table = (slices[slices["series"].isin(part["series"])] for part in (fit, val))
    features = Path(args.features)
    fit_data = series_data(fit_table, fit, load_features(features, args.fold, fit_table))
    val_data = series_data(val_table, val, load_features(features, args.fold, val_table))
    fold = validation_fold(slices, series, args.fold, val["series"])
    val[["series"]].to_csv(run_dir / "val.csv", index=False)
    print(f"{args.name}: training {len(fit):,} series | validation {len(val):,} series | {fit_data.features[0].shape[1]} features per slice", flush=True)

    neighbours = not args.no_neighbours
    n_in = fit_data.features[0].shape[1] * (3 if neighbours else 1)
    model = SeriesModel(n_in, args.hidden, len(SERIES_LABELS), args.dropout).cuda()
    rates = fit_data.y_series.mean(axis=0).clip(1e-4, 1 - 1e-4)
    model.series_head.bias.data.copy_(torch.from_numpy(np.log(rates / (1 - rates))).float())  # starts at the label rates
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    batches = -(-len(fit) // args.batch)  # the last, smaller batch too
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate(step, args.epochs * batches))
    rng = np.random.default_rng(args.seed)

    log, start = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order, total = rng.permutation(len(fit)), 0.0
        for k in range(batches):
            items = order[k * args.batch:(k + 1) * args.batch]
            b = batch(fit_data, items, None if args.no_flip else rng.random(len(items)) < 0.5, neighbours)
            loss = series_loss(*model(b["x"], b["lengths"]), b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            total += loss.item()
        log.append({"epoch": epoch, "train loss": total / batches, "minutes": (time.time() - start) / 60})
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)

    slice_logits, series_logits = predict(model, val_data, neighbours=neighbours)
    np.save(run_dir / "val_series_logits.npy", series_logits)
    np.savez_compressed(run_dir / "val_slice_logits.npz", *slice_logits)
    scores = {**log[-1], **series_scores(fold, val_data, slice_logits, series_logits)}
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=1))
    print(scores, flush=True)


if __name__ == "__main__":
    main()
