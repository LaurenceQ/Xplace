#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def read_summary(path):
    values = {}
    with path.open("r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(",", 1)
            values[key] = value
    return values


def main():
    parser = argparse.ArgumentParser(description="Profile static DMP gate LUT usage concentration.")
    parser.add_argument("--design", default="blabla", help="Design name passed to run_timer.py.")
    parser.add_argument("--out-dir", default="result/dmp_lut_profile", help="Output directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / args.out_dir / args.design
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / args.design
    log_path = out_dir / f"{args.design}.log"

    env = os.environ.copy()
    env.setdefault("CUDA_MODULE_LOADING", "EAGER")
    env["DMP_PROFILE_LUTS"] = "1"
    env["DMP_LUT_PROFILE_OUT"] = str(prefix)

    cmd = [args.python, "run_timer.py", "--designName", args.design]
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    summary_path = Path(str(prefix) + ".summary.txt")
    top_luts_path = Path(str(prefix) + ".top_luts.csv")
    dims_path = Path(str(prefix) + ".dims.csv")
    top_timings_path = Path(str(prefix) + ".top_timings.csv")

    print(f"log: {log_path}")
    print(f"summary: {summary_path}")
    print(f"top_luts: {top_luts_path}")
    print(f"dims: {dims_path}")
    print(f"top_timings: {top_timings_path}")

    if summary_path.exists():
        summary = read_summary(summary_path)
        keys = [
            "gate_arcs",
            "valid_lanes",
            "invalid_transition_lanes",
            "total_lut_refs",
            "unique_luts",
            "top1_frac",
            "top5_frac",
            "top10_frac",
            "top32_frac",
        ]
        for key in keys:
            if key in summary:
                print(f"{key}: {summary[key]}")

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
