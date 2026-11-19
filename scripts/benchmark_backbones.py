"""Find the largest training batch of each candidate backbone within a GPU memory limit (04_slice_models, section 1).

Usage: python scripts/benchmark_backbones.py [--limit-gb 26] [--size 512] [--channels 3] [--step 4]

Every trial runs in its own process, so that one trial's memory cannot affect the next, and makes real training steps
on random input: bf16 forward, backward, AdamW step. Memory does not depend on the pixel values or on the weights, so
no image and no pretrained weight is read. The process cannot reserve more than the limit, and a trial also fails if
the GPU's used memory rose by more than the limit, which counts what PyTorch does not reserve itself.

Writes outputs/backbone_benchmark/trials_<size>.csv (every trial) and batches_<size>.csv (the largest batch of each backbone).
"""
import argparse
import json
import subprocess
import sys
import time

import pandas as pd

from pe.models.slice_model import BACKBONES
from pe.paths import load_paths

N_STEPS, SKIP_STEPS = 12, 4  # the speed comes from the steps after the first few, which set up cudnn.benchmark


def gpu_used_bytes() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0]) * 2 ** 20


def one_trial(backbone: str, batch: int, size: int, channels: int, limit: float, baseline: int) -> dict:
    import timm
    import torch

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(limit / torch.cuda.get_device_properties(0).total_memory)
    result = {"batch": batch, "ok": False, "reason": ""}
    try:
        net = timm.create_model(backbone, pretrained=False, in_chans=channels, num_classes=0)
        model = torch.nn.Sequential(net, torch.nn.Dropout(0.2), torch.nn.Linear(net.num_features, 1)).cuda().to(memory_format=torch.channels_last).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for step in range(N_STEPS):
            if step == SKIP_STEPS:
                torch.cuda.synchronize()
                start = time.time()
            x = torch.randn(batch, channels, size, size, device="cuda").contiguous(memory_format=torch.channels_last)
            y = torch.randint(0, 2, (batch, 1), device="cuda").float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        result["slices per s"] = round((N_STEPS - SKIP_STEPS) * batch / (time.time() - start), 1)
        result["reserved GB"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
        result["gpu rise GB"] = round((gpu_used_bytes() - baseline) / 1e9, 2)
        result["ok"] = result["gpu rise GB"] <= limit / 1e9
        result["reason"] = "" if result["ok"] else "gpu rise above the limit"
    except torch.cuda.OutOfMemoryError:
        result["reason"] = "out of memory"
    return result


def run_trial(backbone: str, batch: int, args) -> dict:
    baseline = gpu_used_bytes()
    command = [sys.executable, __file__, "--one", backbone, str(batch), str(baseline), "--limit-gb", str(args.limit_gb), "--size", str(args.size), "--channels", str(args.channels)]
    out = subprocess.run(command, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{backbone} batch {batch}:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def search(name: str, backbone: str, args) -> list[dict]:
    """Double the batch until a trial fails, then halve the gap down to `step`."""
    trials, low, high, batch = [], 0, None, 32
    while high is None or high - low > args.step:
        trial = {"name": name, "backbone": backbone, **run_trial(backbone, batch, args)}
        trials.append(trial)
        print(trial, flush=True)
        low, high = (batch, high) if trial["ok"] else (low, batch)
        if high is None:
            batch *= 2
        else:
            batch = (low + high) // 2 // args.step * args.step
            if batch <= low:
                break
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit-gb", type=float, default=26)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--one", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.one:
        backbone, batch, baseline = args.one
        print(json.dumps(one_trial(backbone, int(batch), args.size, args.channels, args.limit_gb * 1e9, int(baseline))))
        return

    out_dir = load_paths()["outputs_dir"] / "backbone_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    for name, backbone in BACKBONES.items():
        trials += search(name, backbone, args)
        pd.DataFrame(trials).to_csv(out_dir / f"trials_{args.size}.csv", index=False)
    table = pd.DataFrame(trials)
    best = table[table["ok"]].sort_values("batch").groupby("name", sort=False).tail(1).set_index("name").loc[list(BACKBONES)]
    best[["backbone", "batch", "slices per s", "reserved GB", "gpu rise GB"]].to_csv(out_dir / f"batches_{args.size}.csv")
    print(best[["batch", "slices per s", "reserved GB", "gpu rise GB"]].to_string())


if __name__ == "__main__":
    main()
