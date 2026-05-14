#!/usr/bin/env python3
"""Run Xplace direct route-segment timing against saved OpenROAD references."""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import os
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BENCH = Path("/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks")
PLATFORM = BENCH / "NanGate45"
DEFAULT_OUT = REPO / "result" / "ispd25_direct_route_latest"
REF_RE = re.compile(r"^(tns|wns)\s+max\s+([-+0-9.eE]+)\s*$")
DMP_RE = re.compile(
    r"DMP route-segment RC evaluation:.*?"
    r"wns_late:\s*([-+0-9.eE]+),\s*tns_late:\s*([-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="visible", choices=["visible", "blind"])
    parser.add_argument("--design", action="append", dest="designs")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", default="8")
    parser.add_argument("--timeout-min", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--mark-skipped-large",
        type=float,
        default=0.0,
        metavar="GB",
        help="Do not run cases whose segment file is larger than this many GiB.",
    )
    return parser.parse_args()


def parse_reference(log_path: Path) -> tuple[float, float]:
    tns = None
    wns = None
    with log_path.open("r", errors="replace") as f:
        for line in f:
            match = REF_RE.match(line.strip())
            if not match:
                continue
            value = float(match.group(2))
            if match.group(1) == "tns":
                tns = value
            else:
                wns = value
    if tns is None or wns is None:
        raise RuntimeError(f"Cannot parse tns/wns from {log_path}")
    return wns, tns


def parse_direct(log_path: Path) -> tuple[float, float]:
    text = log_path.read_text(errors="replace")
    matches = list(DMP_RE.finditer(text))
    if not matches:
        raise RuntimeError(f"Cannot parse DMP route-segment WNS/TNS from {log_path}")
    match = matches[-1]
    return float(match.group(1)), float(match.group(2))


def extract_log_error(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(errors="replace")
    for marker in (
        "[DMP INIT] cudaMalloc failed",
        "[DMP INIT] cudaMemset failed",
        "[DMP INIT] slot capacity exceeds int indexing",
        "GPUassert:",
        "out of memory",
    ):
        if marker in text:
            for line in text.splitlines():
                if marker in line:
                    return line.strip()
    if "MemoryError: std::bad_alloc" in text:
        return "MemoryError: std::bad_alloc"
    if "# TIMEOUT after" in text:
        return text[text.rfind("# TIMEOUT after") :].strip().splitlines()[0]
    if "Traceback (most recent call last):" in text:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line:
                return line
    return ""


def pct_diff(actual: float, reference: float) -> float:
    denom = abs(reference)
    if denom < 1e-12:
        if reference >= 0.0 and actual >= 0.0:
            return 0.0
        return 0.0 if abs(actual - reference) < 1e-12 else math.inf
    return abs(actual - reference) / denom * 100.0


def discover_designs(split: str) -> list[str]:
    ref_dir = BENCH / "openroad_gr_logs_skip_fanout300" / split
    return sorted(
        path.name.split(".eval_crpr_off.skip_fanout300.log")[0]
        for path in ref_dir.glob("*.eval_crpr_off.skip_fanout300.log")
    )


def run_case(args: argparse.Namespace, design: str) -> dict[str, str]:
    ref_log = BENCH / "openroad_gr_logs_skip_fanout300" / args.split / (
        f"{design}.eval_crpr_off.skip_fanout300.log"
    )
    segment = BENCH / "openroad_gr_segments_skip_fanout300" / args.split / (
        f"{design}.route_segments"
    )
    case_log_dir = args.out / "logs" / args.split
    case_log_dir.mkdir(parents=True, exist_ok=True)
    direct_log = case_log_dir / f"{design}.direct_route.log"

    ref_wns, ref_tns = parse_reference(ref_log)
    status = "run"
    reason = ""

    segment_size_gib = math.nan
    if segment.exists():
        segment_size_gib = segment.stat().st_size / (1024.0**3)

    if not segment.exists():
        status = "missing_segment"
        reason = f"missing {segment}"
    elif args.mark_skipped_large > 0 and segment_size_gib > args.mark_skipped_large:
        status = "skipped_large"
        reason = f"segment_size_gib={segment_size_gib:.3f} > {args.mark_skipped_large:.3f}"
    elif args.skip_existing and direct_log.exists():
        status = "existing"
    else:
        cmd = [
            sys.executable,
            "run_timer.py",
            "--platformPath",
            str(PLATFORM),
            "--designPath",
            str(BENCH / args.split),
            "--designName",
            design,
            "--route_segments",
            str(segment),
            "--global_placement",
            "False",
            "--legalization",
            "False",
            "--detail_placement",
            "False",
            "--write_placement",
            "False",
            "--num_threads",
            str(args.threads),
            "--result_dir",
            str(args.out / "run_timer_results"),
            "--exp_id",
            f"_direct_latest_{args.split}_{design}",
        ]
        timeout = None if args.timeout_min <= 0 else args.timeout_min * 60.0
        with direct_log.open("w") as log:
            log.write("# " + " ".join(cmd) + "\n")
            log.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                )
                if proc.returncode != 0:
                    status = "failed"
                    reason = extract_log_error(direct_log) or f"returncode={proc.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
                reason = f"timeout_min={args.timeout_min:g}"
                log.write(f"\n# TIMEOUT after {args.timeout_min:g} minutes\n")

    direct_wns = math.nan
    direct_tns = math.nan
    wns_diff = math.inf
    tns_diff = math.inf
    passed = "fail"
    if direct_log.exists() and status not in {"missing_segment"}:
        try:
            direct_wns, direct_tns = parse_direct(direct_log)
            wns_diff = pct_diff(direct_wns, ref_wns)
            tns_diff = pct_diff(direct_tns, ref_tns)
            passed = "pass" if wns_diff <= 1.0 and tns_diff <= 1.0 else "fail"
        except Exception as exc:  # keep the CSV useful after failed runs
            log_error = extract_log_error(direct_log)
            reason = reason or log_error or str(exc)
            if status in {"run", "existing"}:
                status = "failed" if log_error else "parse_failed"

    return {
        "split": args.split,
        "design": design,
        "status": status,
        "pass": passed,
        "ref_wns": f"{ref_wns:.10g}",
        "ref_tns": f"{ref_tns:.10g}",
        "direct_wns": f"{direct_wns:.10g}" if math.isfinite(direct_wns) else "",
        "direct_tns": f"{direct_tns:.10g}" if math.isfinite(direct_tns) else "",
        "wns_pct_diff": f"{wns_diff:.6g}" if math.isfinite(wns_diff) else "inf",
        "tns_pct_diff": f"{tns_diff:.6g}" if math.isfinite(tns_diff) else "inf",
        "segment_size_gib": f"{segment_size_gib:.6g}" if math.isfinite(segment_size_gib) else "",
        "segment": str(segment),
        "direct_log": str(direct_log),
        "ref_log": str(ref_log),
        "reason": reason,
    }


def load_existing_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def merge_rows(existing: list[dict[str, str]], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged = {(row["split"], row["design"]): row for row in existing}
    for row in rows:
        merged[(row["split"], row["design"])] = row
    return [merged[key] for key in sorted(merged)]


def write_summary(out: Path, rows: list[dict[str, str]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_path = out / "summary.csv"
    md_path = out / "README.md"
    fields = [
        "split",
        "design",
        "status",
        "pass",
        "ref_wns",
        "ref_tns",
        "direct_wns",
        "direct_tns",
        "wns_pct_diff",
        "tns_pct_diff",
        "segment_size_gib",
        "segment",
        "direct_log",
        "ref_log",
        "reason",
    ]
    full_rows = merge_rows(load_existing_rows(csv_path), rows)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(full_rows)

    with md_path.open("w") as f:
        f.write("# ISPD2025 Direct Route-Segment Latest OpenROAD Alignment\n\n")
        f.write(f"Updated: {stamp}\n\n")
        f.write("Mainline input is direct `--route_segments`; `--gr_rc` is not used.\n\n")
        f.write("| split | design | status | pass | ref WNS | ref TNS | direct WNS | direct TNS | WNS % | TNS % |\n")
        f.write("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in full_rows:
            f.write(
                f"| {row['split']} | {row['design']} | {row['status']} | "
                f"{row['pass']} | {row['ref_wns']} | {row['ref_tns']} | "
                f"{row['direct_wns']} | {row['direct_tns']} | "
                f"{row['wns_pct_diff']} | {row['tns_pct_diff']} |\n"
            )
        f.write("\nSee `summary.csv` for segment, direct log, reference log, and reason columns.\n")


def main() -> int:
    args = parse_args()
    designs = args.designs or discover_designs(args.split)
    rows = []
    for design in designs:
        row = run_case(args, design)
        rows.append(row)
        write_summary(args.out, rows)
        print(
            f"{row['split']}/{row['design']}: {row['status']} {row['pass']} "
            f"ref=({row['ref_wns']},{row['ref_tns']}) "
            f"direct=({row['direct_wns']},{row['direct_tns']}) "
            f"diff=({row['wns_pct_diff']}%,{row['tns_pct_diff']}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
