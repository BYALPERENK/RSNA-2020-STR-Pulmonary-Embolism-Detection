"""Build the slice and series tables and the image store (02_data_preparation, section 7).

Usage: python scripts/prepare_data.py [--workers N] [--tables-only] [--limit N]

Writes into processed_data_dir:
- series.csv: one row per series (`pe.data.prepare.build_tables`): split, storage group, slices and stacks, and for
  train the 13 labels and the fold. Read it with `pe.data.prepare.read_series`;
- slices.csv: one row per slice: its stack and place in it, its label (train), and the pixel statistics of
  `pixel_stats` once the store is written. Read it with `pe.data.prepare.read_slices`;
- png16/<series>/<image>.png: every slice as PNG 16-bit, HU + 1024 (02, section 6). Files already there are kept,
  so an interrupted run continues where it stopped.

The DICOM files are read in one thread in the order they lie on disk (the order of dicom_headers.csv): on a hard disk
this is several times faster than parallel reads. Decoding and writing run in --workers processes. A file that
cannot be decoded is listed with its error at the end; the run then writes slices_failed.csv instead and exits with
an error.

Needs dicom_headers.csv from scripts/extract_dicom_headers.py.
"""
import argparse
import os
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import cv2
import pandas as pd
from tqdm import tqdm

from pe.data.prepare import build_tables, load_labels, store_path, write_slice
from pe.paths import load_paths

IN_FLIGHT = 1024  # slices read but not yet written: bounds the memory to about 0.5 GB


def safe_write(source, out_path):
    try:
        return write_slice(source, out_path)
    except Exception as e:
        return {"error": repr(e)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=os.cpu_count() - 2)
    parser.add_argument("--tables-only", action="store_true", help="write series.csv and slices.csv without the store")
    parser.add_argument("--limit", type=int, help="first N files only, into a separate store (smoke test)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    paths = load_paths()
    raw_dir, out_dir = paths["raw_data_dir"], paths["processed_data_dir"]
    suffix = f"_limit{args.limit}" if args.limit else ""  # a smoke test never mixes with the full run
    headers_path = out_dir / f"dicom_headers{suffix}.csv"  # from extract_dicom_headers.py with the same --limit
    if not headers_path.exists():
        sys.exit(f"{headers_path} not found: run scripts/extract_dicom_headers.py{' --limit ' + str(args.limit) if args.limit else ''} first")
    headers = pd.read_csv(headers_path, dtype=str, keep_default_na=False)
    slices, series = build_tables(headers, *load_labels(raw_dir / "train.csv"))
    series.to_csv(out_dir / f"series{suffix}.csv", index=False)
    print(f"slices: {len(slices):,} | series: {len(series):,} ({series['split'].value_counts().to_dict()})", flush=True)
    if args.tables_only:
        slices.to_csv(out_dir / f"slices{suffix}.csv", index=False)
        return

    store = out_dir / f"png16{suffix}"
    store.mkdir(exist_ok=True)
    order = headers[["path", "StudyInstanceUID", "SOPInstanceUID"]].iloc[:args.limit]  # disk order
    stats, pending = {}, deque()
    with ProcessPoolExecutor(args.workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:  # one core per worker
        for rel, series_id, image in tqdm(order.itertuples(index=False), total=len(order), mininterval=30, ascii=True):
            target = store_path(store, series_id, image)
            source = str(target) if target.exists() else (raw_dir / rel).read_bytes()  # an existing file is only read back
            pending.append((image, pool.submit(safe_write, source, target)))
            while len(pending) >= IN_FLIGHT or (pending and pending[0][1].done()):
                done, future = pending.popleft()
                stats[done] = future.result()
        for done, future in pending:
            stats[done] = future.result()

    stats = pd.DataFrame.from_dict(stats, orient="index")
    slices = slices[slices["image"].isin(stats.index)].join(stats, on="image")
    failed = slices[slices["error"].notna()] if "error" in slices else slices.iloc[:0]
    table = out_dir / f"slices{suffix}{'_failed' if len(failed) else ''}.csv"  # training never reads an incomplete table
    slices.to_csv(table, index=False)
    for row in failed.itertuples():
        print("not decoded:", row.path, row.error)
    size = sum(e.stat().st_size for d in os.scandir(store) for e in os.scandir(d))
    print(f"store: {len(stats) - len(failed):,} files, {size / 1e9:.1f} GB -> {store} | not decoded: {len(failed)}")
    print(f"table: {slices.shape[0]:,} rows, {slices.shape[1]} columns -> {table}")
    if len(failed):
        sys.exit(f"{len(failed)} slices not decoded: the training would stop at them. Fix or replace their files and run again.")


if __name__ == "__main__":
    main()
