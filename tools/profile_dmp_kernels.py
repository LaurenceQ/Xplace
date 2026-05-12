#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


PROFILE_RE = re.compile(
    r"\[DMP KERNEL PROFILE\]\s+name=(?P<name>\S+)\s+"
    r"launches=(?P<launches>\d+)\s+"
    r"total_ms=(?P<total_ms>[-+0-9.eE]+)\s+"
    r"avg_us=(?P<avg_us>[-+0-9.eE]+)\s+"
    r"max_ms=(?P<max_ms>[-+0-9.eE]+)"
)


def parse_profiles(log_path):
    rows = []
    with log_path.open("r", errors="replace") as f:
        for line in f:
            match = PROFILE_RE.search(line)
            if not match:
                continue
            row = match.groupdict()
            row["launches"] = int(row["launches"])
            row["total_ms"] = float(row["total_ms"])
            row["avg_us"] = float(row["avg_us"])
            row["max_ms"] = float(row["max_ms"])
            rows.append(row)
    rows.sort(key=lambda item: item["total_ms"], reverse=True)
    return rows


def write_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "launches", "total_ms", "avg_us", "max_ms"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_design(design, out_dir, python_exe):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{design}.log"
    csv_path = out_dir / f"{design}.kernels.csv"
    env = os.environ.copy()
    env.setdefault("CUDA_MODULE_LOADING", "EAGER")
    env["DMP_PROFILE_KERNELS"] = "1"
    cmd = [python_exe, "run_timer.py", "--designName", design]
    with log_path.open("w") as log:
        proc = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    rows = parse_profiles(log_path)
    write_csv(rows, csv_path)
    return proc.returncode, log_path, csv_path, rows


def main():
    parser = argparse.ArgumentParser(description="Run DMP SPEF timing and export kernel profile CSV.")
    parser.add_argument("--design", default="blabla", help="Design name passed to run_timer.py.")
    parser.add_argument("--log", default=None, help="Parse an existing run_timer log instead of running.")
    parser.add_argument("--out-dir", default="result/dmp_kernel_profile", help="Output directory.")
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
        rows = parse_profiles(log_path)
        csv_path = Path(args.out_dir) / f"{log_path.stem}.kernels.csv"
        write_csv(rows, csv_path)
        code = 0
    else:
        code, log_path, csv_path, rows = run_design(args.design, Path(args.out_dir), args.python)
    print(f"log: {log_path}")
    print(f"csv: {csv_path}")
    for row in rows[:12]:
        print(f"{row['total_ms']:9.3f} ms  {row['launches']:5d}x  {row['name']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
