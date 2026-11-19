"""Train one slice model on one fold and predict its validation fold (04_slice_models).

Usage: python scripts/train_slice.py --name NAME [--fold 0] [--share 1] [--seed 1] [--backbone efficientnet_b0] [--epochs 4] [--lr 4e-4]
       [--channels=-1,0,1] [--windows pe] [--crop lung] [--size 512 --batch 32 --accumulate 4]
       [--positive-series] [--q-weight] [--loss focal] [--series-targets] [--without-rare-writers] ...

- Training: the main-stack slices of the fold's training series, or a fixed `--share` of those series
  (`pe.data.dataset.screening_series`). Binary cross-entropy on pe_present_on_image by default.
- Options: --positive-series and --without-rare-writers drop series from training only; --q-weight weights each slice as the
  metric does; --loss focal uses gamma 2; --series-targets adds the mean cross-entropy of the 7 PE series labels, on positive slices.
- Input: the slices at --channels from each slice, each through every one of --windows, optionally cut to the box of --crop.
- Validation: every main-stack slice of the fold, after every epoch (`pe.validation.slice_scores`). With --stop-when-worse
  the run stops at the first epoch whose test-like score after the level correction is worse than the one before.

Writes outputs/runs/NAME/: config.json, log.csv (one row per epoch, with its scores), val.csv (the validation slices in
prediction order), val_logits_e<k>.npy, inputs.png (the first training batch, for a visual check) and, last of all,
scores.json (the last epoch's scores, and the epoch before the drop if the run stopped). The weights of every epoch go to
models/NAME_e<k>.pt.

The learning rate follows --epochs, also when the run stops early. --limit N keeps N series on each side (smoke test).
"""
import argparse
import json
import random
import time
from itertools import islice

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from pe.data.dataset import SliceDataset, neighbour_images, positive_slice_targets, q_weights, screening_series, series_boxes
from pe.data.prepare import IMAGE_LABEL, LOCATION, RVLV, TYPE, read_series, read_slices
from pe.data.windows import WINDOWS
from pe.models.slice_model import BACKBONES, SliceModel
from pe.paths import load_paths
from pe.training import augment, learning_rate, predict, slice_loss, to_windows
from pe.validation import RARE_IN_TEST, slice_scores, validation_fold

SLICES_STORE = {256: "png16_256", 512: "png16"}
DECIDES = "test-like corrected"  # the score that ranks the models (`pe.validation`)
SERIES_TARGETS = LOCATION + RVLV + TYPE  # the labels that describe a PE, for --series-targets
FOCAL_GAMMA = 2


def worker_setup(_):
    cv2.setNumThreads(1)  # one core per worker


def make_dataset(store_dir, table: pd.DataFrame, labels: np.ndarray, offsets: list[int], crop: str | None, size: int) -> SliceDataset:
    return SliceDataset(store_dir, table["series"].to_numpy(dtype="U"), neighbour_images(table, offsets), labels, series_boxes(table, crop, size) if crop else None)


def save_inputs(x: torch.Tensor, path):
    """The middle channel of the first 16 items, after the window and the augmentation."""
    images = (x[:16, x.shape[1] // 2].float().cpu().numpy() * 255).round().astype(np.uint8)
    cv2.imwrite(str(path), np.vstack([np.hstack(images[r:r + 4]) for r in range(0, 16, 4)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--share", type=float, default=1.0, help="share of the training series")
    parser.add_argument("--seed", type=int, default=1, help="the training randomness; the series of --share do not depend on it")
    parser.add_argument("--backbone", default="efficientnet_b0", choices=BACKBONES)  # the fastest of the 8 candidates (04, section 1)
    parser.add_argument("--size", type=int, default=256, choices=SLICES_STORE)
    parser.add_argument("--batch", type=int, default=128)  # fits every candidate at 256 px (04, section 1)
    parser.add_argument("--accumulate", type=int, default=1, help="batches per optimizer step")
    parser.add_argument("--channels", default="-1,0,1", help="the slices of one input, as distances from its slice; write --channels=-1,0,1")
    parser.add_argument("--windows", default="pe", help="the windows of pe.data.windows every slice goes through, separated by commas")
    parser.add_argument("--crop", choices=["lung", "body"], help="cut every slice to this box of its series")
    parser.add_argument("--positive-series", action="store_true", help="train only on the series with PE")
    parser.add_argument("--without-rare-writers", action="store_true", help="train without the two writers that are rare in the test set")
    parser.add_argument("--q-weight", action="store_true", help="weight each slice by the share of positive slices in its series")
    parser.add_argument("--loss", default="bce", choices=["bce", "focal"])
    parser.add_argument("--series-targets", action="store_true", help="also predict the PE series labels on positive slices")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=12)  # at most 12, always
    parser.add_argument("--stop-when-worse", action="store_true", help="stop at the first epoch whose score is worse than the one before")
    parser.add_argument("--stop-after", type=int, help="stop after this epoch; the learning rate still follows --epochs")
    parser.add_argument("--limit", type=int, help="keep N series on each side (smoke test)")
    args = parser.parse_args()

    paths = load_paths()
    run_dir, models_dir = paths["outputs_dir"] / "runs" / args.name, paths["models_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    processed = paths["processed_data_dir"]
    slices, series = read_slices(processed / "slices.csv"), read_series(processed / "series.csv")
    train_series = series[(series["split"] == "train") & (series["fold"] != args.fold)]
    val_series = series.loc[(series["split"] == "train") & (series["fold"] == args.fold), "series"].to_numpy(dtype="U")
    if args.positive_series:
        train_series = train_series[(train_series["negative_exam_for_pe"] == 0) & (train_series["indeterminate"] == 0)]
    if args.without_rare_writers:
        train_series = train_series[~train_series["writer"].isin(RARE_IN_TEST)]
    fit_series = screening_series(train_series, args.share)
    if args.limit:
        fit_series, val_series = fit_series[:args.limit], val_series[:args.limit]
    main_slices = slices[(slices["split"] == "train") & slices["main"]]
    fit_table, val_table = main_slices[main_slices["series"].isin(fit_series)], main_slices[main_slices["series"].isin(val_series)]
    y_rate = float(fit_table[IMAGE_LABEL].astype(float).mean())
    print(f"{args.name}: training {len(fit_series):,} series, {len(fit_table):,} slices ({y_rate:.1%} positive) | validation {len(val_series):,} series, {len(val_table):,} slices", flush=True)

    store_dir = processed / SLICES_STORE[args.size]
    offsets = [int(offset) for offset in args.channels.split(",")]
    windows = [WINDOWS[name] for name in args.windows.split(",")]
    generator = torch.Generator().manual_seed(args.seed)
    # One row per slice: the label, its weight, then the series targets
    weights = q_weights(fit_table) if args.q_weight else np.ones(len(fit_table))
    targets = positive_slice_targets(fit_table, series, SERIES_TARGETS) if args.series_targets else np.empty((len(fit_table), 0))
    fit_labels = np.column_stack([fit_table[IMAGE_LABEL].astype(float), weights, targets]).astype(np.float32)
    # The workers of one loader end before the other's start, so that never more than --workers exist
    loader = DataLoader(make_dataset(store_dir, fit_table, fit_labels, offsets, args.crop, args.size), batch_size=args.batch, shuffle=True, drop_last=True, generator=generator,
                        num_workers=args.workers, pin_memory=True, worker_init_fn=worker_setup)
    val_loader = DataLoader(make_dataset(store_dir, val_table, val_table[IMAGE_LABEL].to_numpy(dtype=np.float32), offsets, args.crop, args.size), batch_size=2 * args.batch,
                            num_workers=args.workers, pin_memory=True, worker_init_fn=worker_setup)
    fold = validation_fold(slices, series, args.fold, val_series)
    val_table[["series", "image"]].to_csv(run_dir / "val.csv", index=False)
    model = SliceModel(args.backbone, in_chans=len(offsets) * len(windows), dropout=args.dropout, series_targets=targets.shape[1])
    start_rate = np.average(fit_table[IMAGE_LABEL].astype(float), weights=weights)
    torch.nn.init.constant_(model.head.bias, np.log(start_rate / (1 - start_rate)))  # starts at the rate of the positive slices
    if args.series_targets:
        rates = targets.mean(axis=0).clip(1e-4, 1 - 1e-4)
        model.series_head.bias.data.copy_(torch.from_numpy(np.log(rates / (1 - rates))).float())
    model = model.cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    batches = len(loader) // args.accumulate * args.accumulate  # an incomplete last group would move BatchNorm without a step
    steps = args.epochs * batches // args.accumulate
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate(step, steps))
    air = to_windows(torch.zeros(1, len(offsets), 1, 1).cuda(), windows)  # stored 0 is air

    log, start = [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for b, (stored, y) in enumerate(tqdm(islice(loader, batches), total=batches, desc=f"epoch {epoch}", mininterval=60, ascii=True)):
            x = augment(to_windows(stored.cuda(non_blocking=True), windows), air)
            if epoch == 1 and b == 0:
                save_inputs(x, run_dir / "inputs.png")
            x = x.contiguous(memory_format=torch.channels_last)
            y = y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, features = model(x)
                series_logits = model.series_head(model.dropout(features)) if args.series_targets else None
            loss = slice_loss(logits.float(), y[:, 0], y[:, 1], FOCAL_GAMMA if args.loss == "focal" else 0.0)
            if args.series_targets:
                loss = loss + F.binary_cross_entropy_with_logits(series_logits.float(), y[:, 2:])
            (loss / args.accumulate).backward()
            if (b + 1) % args.accumulate == 0:
                optimizer.step()
                schedule.step()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item()
        train_minutes = (time.time() - start) / 60
        torch.save(model.state_dict(), models_dir / f"{args.name}_e{epoch}.pt")
        logits = predict(model, val_loader, windows)
        np.save(run_dir / f"val_logits_e{epoch}.npy", logits)
        log.append({"epoch": epoch, "train loss": total / batches, "train minutes": train_minutes, "minutes": (time.time() - start) / 60, **slice_scores(fold, logits)})
        pd.DataFrame(log).to_csv(run_dir / "log.csv", index=False)
        print(log[-1], flush=True)
        if args.stop_when_worse and epoch > 1 and log[-1][DECIDES] > log[-2][DECIDES]:
            break
        if epoch == args.stop_after:
            break

    worse = args.stop_when_worse and len(log) > 1 and log[-1][DECIDES] > log[-2][DECIDES]
    scores = {**log[-1], "epoch before the drop": log[-2]["epoch"] if worse else None, "peak reserved GB": torch.cuda.max_memory_reserved() / 1e9}
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=1))
    print(scores, flush=True)


if __name__ == "__main__":
    main()
