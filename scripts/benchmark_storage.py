"""Benchmark of the storage formats compared in 02_data_preparation (section 6).

Writes the same slices (the main stacks of 2 series per storage group) in every format of `pe.data.formats` to a
temporary folder in processed_data_dir, then measures what 02 shows:
- sizes.csv, identical.csv, jpeg_errors.csv: the size per slice and of the whole store, and what each format gives back;
- for the two lossless HU stores that fit on the SSD (MEASURED): png_levels.csv (the PNG compression level), the
  writing time (in sizes.csv), speed.csv and loader.csv (the reading time in one process, and of 2.5D inputs, 3
  neighbouring slices, through a DataLoader with 4, 8 and 12 workers), end_to_end.csv (whether the loader keeps up
  with training at 512 px);
- notes.json: the single numbers printed along the way.
The tables go to outputs/storage_benchmark/. It needs slices.csv from prepare_data.py, a CUDA GPU and about 5 GB free.
It runs in two stages, so that the files can be written while the machine is busy and timed when it is not:
- --stage write: picks the slices (bench.csv), writes them in every format and records what needs no timing
  (sizes, identical, jpeg_errors). The files stay in processed_data_dir/format_benchmark;
- --stage measure: every timing, on those files (it should run alone: the timings share the CPU and disks), then
  deletes them.
Without --stage it runs both. --quick shortens the GPU timings for a test run.

    python scripts/benchmark_storage.py --stage write
    python scripts/benchmark_storage.py --stage measure
"""
import argparse
import gc
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Sampler

from pe.data.dicom import HU_MAX, HU_MIN, read_hu
from pe.data.formats import FORMATS, DecodeDataset, png_encode, read_uncached
from pe.data.prepare import SEED, read_series, read_slices
from pe.data.windows import WINDOWS, window
from pe.paths import load_paths

MEASURED = ["png16", "jp2"]  # the timings: the two lossless HU stores that fit on the SSD (sizes.csv)
END_TO_END = ["efficientnet_b3", "seresnext50_32x4d", "tf_efficientnet_b5"]  # backbones of the top solutions (REFERANS), at 512 px
BATCH = 32


class Indices(Sampler):
    """Yields whatever `indices` holds when an epoch starts, so that one loader (one set of workers) serves every format."""

    def __init__(self):
        self.indices = []

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def write_file(path, blob):
    with open(path, "wb") as out:
        out.write(blob)
        out.flush()
        os.fsync(out.fileno())  # on the disk, so that the cold reads find it there


def ms_per_item(items, fn):
    t = time.perf_counter()
    for item in items:
        fn(item)
    return (time.perf_counter() - t) / len(items) * 1000


def png_levels(hus):
    f, rows = FORMATS["png16"], []
    arrays = [f.to_array(hu) for hu in hus[::6]]
    for level in [1, 6, 9]:
        t = time.perf_counter()
        blobs = [png_encode(a, level) for a in arrays]
        encode = (time.perf_counter() - t) / len(arrays) * 1000
        t = time.perf_counter()
        for b in blobs:
            f.decode(b)
        rows.append({"level": level, "KB per slice": np.mean([len(b) for b in blobs]) / 1024,
                     "encode ms": encode, "decode ms": (time.perf_counter() - t) / len(arrays) * 1000})
    return pd.DataFrame(rows).round(1)


def bench_files(bench, folder):
    """The file of every benchmark slice in every format; DICOM is the source itself."""
    files = {"dicom": list(bench["dicom"])}
    for name, f in FORMATS.items():
        if f.encode is not None:
            files[name] = [folder / name / f"{image}{f.suffix}" for image in bench["image"]]
    return files


def write_formats(bench, hus, folder, slices, paths):
    files = bench_files(bench, folder)
    for name, f in FORMATS.items():
        if f.encode is None:
            continue
        (folder / name).mkdir(parents=True)
        for path, hu in zip(files[name], hus):
            write_file(path, f.encode(f.to_array(hu)))

    # sizes per storage group, weighted by the groups' share of the store
    kb = pd.DataFrame({name: [p.stat().st_size / 1024 for p in ps] for name, ps in files.items()}).groupby(bench["group"].to_numpy()).mean().T
    share = slices["group"].value_counts(normalize=True)[kb.columns]
    sizes = pd.DataFrame({"format": [FORMATS[n].label for n in files], "content": [FORMATS[n].content for n in files]}, index=list(files))
    sizes["KB per slice"] = (kb @ share).round(1)
    sizes["store GB"] = (sizes["KB per slice"] * 1024 * len(slices) / 1e9).round(1)
    sizes["vs DICOM"] = (sizes["KB per slice"] / sizes.loc["dicom", "KB per slice"]).round(3)
    notes = {f"free GB on {name}": int(round(shutil.disk_usage(paths[name]).free / 1e9)) for name in ["processed_data_dir", "raw_data_dir"]}
    return sizes, notes


def write_times(bench, hus, names):
    """ms to write one slice from its DICOM file: reading it with `read_hu` (300 files, from the hard disk) plus encoding."""
    read_ms = ms_per_item(list(bench["dicom"][:300]), read_hu)
    return pd.Series({name: read_ms + ms_per_item(hus, lambda hu: FORMATS[name].encode(FORMATS[name].to_array(hu))) for name in names}).round(1)


def what_is_kept(hus, files):
    rows = []
    for name, f in FORMATS.items():
        if f.encode is None:
            continue
        same = sum(np.array_equal(f.decode(p.read_bytes()), f.to_array(hu)) for p, hu in zip(files[name], hus))
        rows.append({"format": f.label, "content": f.content, "files identical to the stored array": f"{same:,} / {len(hus):,}"})
    return pd.DataFrame(rows)


def jpeg_errors(hus, files):
    """Error of the JPEG files where the stored value lies inside the window (not 0 or 255), in levels and in HU."""
    f, channels = FORMATS["jpg8x3"], list(WINDOWS)
    hist = np.zeros((len(channels), 256), dtype=np.int64)  # pixels by error in levels
    for hu, path in zip(hus, files["jpg8x3"]):
        a, out = f.to_array(hu), f.decode(path.read_bytes())
        for k in range(len(channels)):
            inside = (a[..., k] > 0) & (a[..., k] < 255)
            hist[k] += np.bincount(np.abs(out[..., k][inside].astype(np.int16) - a[..., k][inside]), minlength=256)
    rows = []
    for k, (channel, (_, width)) in enumerate(WINDOWS.items()):
        h, levels = hist[k], np.arange(256)
        mean = (h * levels).sum() / h.sum()
        rows.append({"window": channel, "% pixels changed": (1 - h[0] / h.sum()) * 100, "mean error (levels)": mean,
                     "99.9th percentile": levels[np.searchsorted(np.cumsum(h), 0.999 * h.sum())], "max": levels[h > 0].max(),
                     "mean error (HU)": mean * width / 255})
    return pd.DataFrame(rows).round(2)


def reading_speed(files, sizes):
    """Warm (file cache), cold (past the cache) and disk (16 reads at once, no decoding); median of three passes."""
    warm, cold, disk = ({name: [] for name in files} for _ in range(3))
    for name, ps in files.items():
        ms_per_item(ps, lambda p: FORMATS[name].decode(p.read_bytes()))  # into the file cache
    for _ in range(3):
        for name, ps in files.items():
            decode = FORMATS[name].decode
            warm[name].append(ms_per_item(ps, lambda p: decode(p.read_bytes())))
            cold[name].append(ms_per_item(ps, lambda p: decode(read_uncached(p))))
            with ThreadPoolExecutor(16) as pool:
                t = time.perf_counter()
                list(pool.map(read_uncached, ps))
                disk[name].append(len(ps) / (time.perf_counter() - t))
    speed = sizes[["format", "content", "KB per slice"]].copy()
    speed["warm ms"] = pd.Series({n: np.median(v) for n, v in warm.items()}).round(2)
    speed["cold ms"] = pd.Series({n: np.median(v) for n, v in cold.items()}).round(2)
    speed["disk slices/s (16 at once)"] = pd.Series({n: np.median(v) for n, v in disk.items()}).round(0).astype(int)
    speed["disk MB/s"] = (speed["disk slices/s (16 at once)"] * speed["KB per slice"] * 1024 / 1e6).round(0).astype(int)
    return speed


def triplets(bench):
    """Positions in `bench` of every slice with a neighbour on both sides, and of its two neighbours."""
    inner = bench.index[(bench["slice"] > 0) & (bench["slice"] < bench["n_slices"] - 1)]
    return [(i - 1, i, i + 1) for i in inner]


def loader_speed(files, bench, n_train):
    """2.5D inputs (3 neighbouring slices) per second through a DataLoader, files in the cache, shuffled; median of two runs."""
    names = [n for n in files if n != "dicom"]
    trip = triplets(bench)
    items = [(name, [str(files[name][i]) for i in t]) for name in names for t in trip]
    rng = np.random.default_rng(SEED)
    result = pd.DataFrame(index=names)
    for workers in [4, 8, 12]:
        sampler = Indices()
        loader = DataLoader(DecodeDataset(items), batch_size=32, sampler=sampler, num_workers=workers, persistent_workers=True, timeout=300)
        sampler.indices = list(range(32 * workers))  # files of the first format only, so that every batch has one shape
        for _ in loader:  # starts the workers
            pass
        for k, name in enumerate(names):
            epoch = np.arange(k * len(trip), (k + 1) * len(trip))
            runs = []
            for _ in range(2):
                sampler.indices = rng.permutation(epoch).tolist()
                t = time.perf_counter()
                for _ in loader:
                    pass
                runs.append(time.perf_counter() - t)
            result.loc[name, f"inputs/s, {workers} workers"] = len(epoch) / np.median(runs)
        del loader
    result["epoch min, 12 workers"] = n_train / result["inputs/s, 12 workers"] / 60
    return result.round(1)


def new_model(name):
    model = timm.create_model(name, pretrained=False, num_classes=1).cuda().to(memory_format=torch.channels_last)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-4)


def train_step(model, opt, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def per_second(run, batch, seconds):
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    n, t = 0, time.perf_counter()
    while time.perf_counter() - t < seconds:
        run()
        n += 1
    torch.cuda.synchronize()
    return n * batch / (time.perf_counter() - t)


def end_to_end(files, name, bench, seconds, batches, skip):
    """A store of HU + 1024 (`name`), 3 neighbouring slices per input through 12 workers, the PE window on the GPU, at
    512 px; against the same step on one batch kept on the GPU."""
    lut = torch.from_numpy(window(np.arange(HU_MIN, HU_MAX + 1), *WINDOWS["pe"])).cuda()
    ds = DecodeDataset([(name, [str(files[name][i]) for i in t]) for t in triplets(bench)])
    batch = BATCH

    def step(model, opt, stored_hu, y):
        x = lut[stored_hu.long()]  # HU + 1024 -> the PE window, 3 slices as 3 channels
        train_step(model, opt, x.contiguous(memory_format=torch.channels_last), y)

    def loader_per_second(loader, run):
        # the first `skip` batches are not timed: the workers still fill their queues
        for i, stored_hu in enumerate(loader):
            if i == skip:
                torch.cuda.synchronize()
                t, n = time.perf_counter(), 0
            run(stored_hu)
            if i >= skip:
                n += len(stored_hu)
        torch.cuda.synchronize()
        return n / (time.perf_counter() - t)

    resident = torch.stack([ds[i % len(ds)] for i in range(batch)]).cuda()  # one batch kept on the GPU
    y = torch.rand(batch, 1, device="cuda")
    loader = DataLoader(ds, batch_size=batch, sampler=RandomSampler(ds, replacement=True, num_samples=batch * batches),
                        num_workers=12, pin_memory=True, persistent_workers=True, timeout=300)
    rows = []
    for backbone in END_TO_END:
        model, opt = new_model(backbone)
        on_gpu = per_second(lambda: step(model, opt, resident, y), batch, seconds)
        loaded = loader_per_second(loader, lambda stored_hu: step(model, opt, stored_hu.cuda(non_blocking=True), y))
        rows.append({"store": name, "backbone": backbone, "size": 512, "data on the GPU": on_gpu, "from the loader": loaded,
                     "%": loaded / on_gpu * 100})
        del model, opt
        torch.cuda.empty_cache()
    return pd.DataFrame(rows).round(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-group", type=int, default=2, help="benchmark series per storage group")
    parser.add_argument("--out", type=Path, help="output folder (default: outputs_dir/storage_benchmark)")
    parser.add_argument("--stage", choices=["write", "measure"], help="only this stage (default: both)")
    parser.add_argument("--quick", action="store_true", help="short GPU timings, for a test run")
    args = parser.parse_args()

    paths = load_paths()
    out = args.out or paths["outputs_dir"] / "storage_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    folder = paths["processed_data_dir"] / "format_benchmark"
    seconds, batches, skip = (0.5, 40, 10) if args.quick else (4, 160, 20)

    slices = read_slices(paths["processed_data_dir"] / "slices.csv")
    slices = slices.join(read_series(paths["processed_data_dir"] / "series.csv").set_index("series")["group"], on="series")

    def save(name, df):
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}.csv written ({time.strftime('%H:%M:%S')})", flush=True)

    if args.stage in (None, "write"):
        shutil.rmtree(folder, ignore_errors=True)
        chosen = slices.drop_duplicates("series").groupby("group").sample(args.per_group, random_state=SEED, replace=True)["series"].unique()
        bench = slices[slices["series"].isin(chosen) & slices["main"]].reset_index(drop=True)
        save("bench", bench[["series", "image", "path", "group", "slice", "n_slices"]])
        bench["dicom"] = [paths["raw_data_dir"] / p for p in bench["path"]]
        notes = {"run": time.strftime("%Y-%m-%d"), "store slices": len(slices), "benchmark series": len(chosen), "benchmark slices": len(bench)}
        hus = [read_hu(p) for p in bench["dicom"]]
        sizes, more = write_formats(bench, hus, folder, slices, paths)
        notes.update(more)
        save("sizes", sizes.rename_axis("name").reset_index())
        files = bench_files(bench, folder)
        save("identical", what_is_kept(hus, files))
        save("jpeg_errors", jpeg_errors(hus, files))
        json.dump(notes, open(out / "notes.json", "w", encoding="utf-8"), indent=1)
        del hus

    if args.stage in (None, "measure"):
        torch.backends.cudnn.benchmark = True
        bench = pd.read_csv(out / "bench.csv")
        bench["dicom"] = [paths["raw_data_dir"] / p for p in bench["path"]]
        files = bench_files(bench, folder)
        notes = json.loads((out / "notes.json").read_text(encoding="utf-8"))
        hus = [FORMATS["npy16"].decode(p.read_bytes()).astype(np.float32) for p in files["npy16"]]  # = read_hu, rounded
        save("png_levels", png_levels(hus))
        sizes = pd.read_csv(out / "sizes.csv").set_index("name")
        sizes["write ms"] = write_times(bench, hus, MEASURED)
        save("sizes", sizes.rename_axis("name").reset_index())
        del hus

        files = {name: files[name] for name in MEASURED}
        save("speed", reading_speed(files, sizes.loc[MEASURED]).rename_axis("name").reset_index())
        n_train = int((slices["main"] & (slices["split"] == "train")).sum())
        save("loader", loader_speed(files, bench, n_train).rename_axis("name").reset_index())
        save("end_to_end", pd.concat([end_to_end(files, name, bench, seconds, batches, skip) for name in MEASURED]))
        notes["GPU memory GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

        json.dump(notes, open(out / "notes.json", "w", encoding="utf-8"), indent=1)
        gc.collect()
        shutil.rmtree(folder)
        print("done; test files deleted:", not folder.exists())


if __name__ == "__main__":
    main()
