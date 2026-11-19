"""Build the 256 px store of the train main stacks from the 512 px store (04_slice_models, section 1).

Usage: python scripts/build_small_store.py [--workers 12] [--limit N]

Writes into processed_data_dir png16_256/<series>/<image>.png: every main-stack slice of train through `pe.data.prepare.halve`,
PNG 16-bit HU + 1024 as in png16/. Files already there are kept, so an interrupted run continues where it stopped. With
--limit, a random sample of N slices goes into its own folder (smoke test and size estimate).

At the end the number of files and the pixels of 2,000 random slices are checked against the 512 px store.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from pe.data.prepare import SEED, STORE, halve, read_slices, store_path
from pe.paths import load_paths

N_CHECK = 2000


def write_small(paths: tuple[str, str]) -> int | str:
    """Write one slice unless it is there; its size in bytes, or the error."""
    source, target = paths
    try:
        if not os.path.exists(target):
            temporary = target + ".tmp"
            with open(source, "rb") as f:
                encoded = STORE.encode(halve(STORE.decode(f.read())))
            with open(temporary, "wb") as f:
                f.write(encoded)
            os.replace(temporary, target)
        return os.path.getsize(target)
    except Exception as e:
        return repr(e)


def same_as_source(paths: tuple[str, str]) -> bool:
    source, target = paths
    with open(source, "rb") as f, open(target, "rb") as g:
        return np.array_equal(halve(STORE.decode(f.read())), STORE.decode(g.read()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, help="a random sample of N slices, into a separate store")
    args = parser.parse_args()

    out_dir = load_paths()["processed_data_dir"]
    table = read_slices(out_dir / "slices.csv")
    table = table[(table["split"] == "train") & table["main"]]
    if args.limit:
        table = table.sample(args.limit, random_state=SEED)
    store = out_dir / (f"png16_256_limit{args.limit}" if args.limit else "png16_256")
    for series in table["series"].unique():
        (store / series).mkdir(parents=True, exist_ok=True)
    pairs = [(str(store_path(out_dir / "png16", s, i)), str(store_path(store, s, i))) for s, i in zip(table["series"], table["image"])]
    print(f"slices: {len(pairs):,} -> {store}", flush=True)

    start = time.time()
    with ProcessPoolExecutor(args.workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:  # one core per worker
        results = list(tqdm(pool.map(write_small, pairs, chunksize=256), total=len(pairs), mininterval=30, ascii=True))
        errors = [(pair, r) for pair, r in zip(pairs, results) if isinstance(r, str)]
        size = sum(r for r in results if not isinstance(r, str))
        print(f"store: {len(results) - len(errors):,} files, {size / 1e9:.2f} GB, {size / max(1, len(results) - len(errors)) / 1e3:.1f} KB per file, "
              f"{(time.time() - start) / 60:.1f} min | not written: {len(errors)}", flush=True)
        for pair, error in errors[:20]:
            print("not written:", pair[0], error)

        files = sum(1 for d in os.scandir(store) for e in os.scandir(d) if e.name.endswith(STORE.suffix))
        temporary = sum(1 for d in os.scandir(store) for e in os.scandir(d) if e.name.endswith(".tmp"))
        rng = np.random.default_rng(SEED)
        check = [pairs[i] for i in rng.choice(len(pairs), min(N_CHECK, len(pairs)), replace=False)]
        same = sum(pool.map(same_as_source, check, chunksize=64))
    print(f"check: {files:,} files for {len(pairs):,} slices | temporary files left: {temporary} | same pixels as halve(512 px): {same:,} of {len(check):,}")


if __name__ == "__main__":
    main()
