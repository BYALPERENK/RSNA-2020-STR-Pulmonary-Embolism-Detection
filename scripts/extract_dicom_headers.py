"""Read the header of every train and test DICOM into dicom_headers.csv (one row per file, all fields as text).

Usage: python scripts/extract_dicom_headers.py [--zip PATH] [--workers 4] [--limit N]

Only headers are parsed, not pixels. The table is the input of 01_eda and of the slice table in 02.

- The files are read in the order they lie on disk: the order of the zip they were extracted from when --zip is
  given, else path order. On a hard disk this is several times faster than reading them in any other order.
- With --zip, every file is also read whole and its CRC-32 is checked against the zip (column `crc_ok`).
- The rows are written in parts of 50,000 files to dicom_headers_parts/; a stopped run continues after its last part.
"""
import argparse
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from pe.data.dicom import read_header
from pe.paths import load_paths

PART_SIZE = 50_000


def list_files(raw_dir, zip_path):
    """Relative paths of all .dcm files in disk order, and the CRC-32 the zip stores for each (or None)."""
    if zip_path:
        infos = sorted(zipfile.ZipFile(zip_path).infolist(), key=lambda i: i.header_offset)
        infos = [i for i in infos if i.filename.endswith(".dcm")]
        return [i.filename for i in infos], [i.CRC for i in infos]
    files = []
    for split in ("train", "test"):
        for dirpath, _, names in os.walk(raw_dir / split):
            rel = os.path.relpath(dirpath, raw_dir).replace(os.sep, "/")
            files += [f"{rel}/{n}" for n in names if n.endswith(".dcm")]
    return sorted(files), None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", help="the downloaded competition zip: gives the disk order and the CRC-32 to check")
    parser.add_argument("--workers", type=int, default=4)  # threads: a hard disk is not faster with more
    parser.add_argument("--limit", type=int, help="first N files only (smoke test)")
    args = parser.parse_args()

    paths = load_paths()
    raw_dir, out_dir = paths["raw_data_dir"], paths["processed_data_dir"]
    name = f"dicom_headers_limit{args.limit}" if args.limit else "dicom_headers"  # a smoke test never mixes with the full run
    parts_dir = out_dir / f"{name}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    files, crcs = list_files(raw_dir, args.zip)
    if args.limit:
        files, crcs = files[:args.limit], crcs and crcs[:args.limit]
    check_crc = crcs is not None

    def read(rel):
        return {"path": rel, "split": rel.split("/")[0], **read_header(str(raw_dir / rel), crc=check_crc)}

    n_parts = (len(files) + PART_SIZE - 1) // PART_SIZE
    with ThreadPoolExecutor(args.workers) as pool:
        for k in tqdm(range(n_parts), desc="parts"):
            part = parts_dir / f"part_{k:03d}.csv"
            if part.exists():
                continue
            lo, hi = k * PART_SIZE, min((k + 1) * PART_SIZE, len(files))
            rows = pd.DataFrame(list(pool.map(read, files[lo:hi])))
            if check_crc:
                rows["crc_ok"] = rows.pop("crc32").eq(pd.Series(crcs[lo:hi]))
            tmp = part.with_suffix(".tmp")
            rows.to_csv(tmp, index=False)
            os.replace(tmp, part)

    headers = pd.concat([pd.read_csv(parts_dir / f"part_{k:03d}.csv", dtype=str, keep_default_na=False)
                         for k in range(n_parts)], ignore_index=True)
    out = out_dir / f"{name}.csv"
    headers.to_csv(out, index=False)
    n_errors = (headers["read_error"] != "").sum() if "read_error" in headers else 0
    n_bad_crc = (headers["crc_ok"] != "True").sum() if check_crc else "not checked"
    print(f"files: {len(headers):,} | fields: {headers.shape[1]} | read errors: {n_errors} | "
          f"CRC mismatches: {n_bad_crc} -> {out}")


if __name__ == "__main__":
    main()
