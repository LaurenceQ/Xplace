#!/usr/bin/env python3
"""Run and summarize ISPD2025 OpenROAD GR-segment timing comparisons.

This script is intentionally scoped to the current direct route-segment
alignment task.  It does not regenerate route segments.  It reads the existing
skip-fanout300 segment files, times an OpenROAD CRPR-off eval, runs Xplace
direct `--route_segments`, and writes one all-case matrix.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]
BENCH = Path("/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks")
PLATFORM = BENCH / "NanGate45"
OPENROAD = Path("/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad")
XPLACE_PY = Path("/home/qkduan25/.conda/envs/gnn/bin/python")
OUT = REPO / "result" / "ispd25_direct_route_latest"
FAST_OPENROAD_TCL = REPO / "tools" / "openroad_eval_gr_segments_crpr_off_fast.tcl"
FULL_OPENROAD_TCL = BENCH / "openroad_eval_gr_segments_crpr_off.tcl"
DESIGNS = [
    "ariane",
    "bsg_chip",
    "NV_NVDLA_partition_c",
    "mempool_tile_wrap",
    "mempool_group",
    "mempool_cluster",
]
SPLITS = ["visible", "blind"]

REF_RE = re.compile(r"^(tns|wns)\s+max\s+([-+0-9.eE]+)\s*$")
DMP_RE = re.compile(
    r"DMP route-segment RC evaluation:.*?"
    r"wns_late:\s*([-+0-9.eE]+),\s*tns_late:\s*([-+0-9.eE]+)"
)
STAMP_RE = re.compile(r"^\[\s*([-+0-9.]+)\]\s+(.*)$")
ROUTE_PROFILE_RE = re.compile(r"\[ROUTE_SEG_PROFILE\] phase=([A-Za-z0-9_]+) elapsed=([-+0-9.eE]+)")
CUDA_MEM_RE = re.compile(r"(?:cuda_used=|cuda_total=|cuda_free=)([-+0-9.eE]+)\s+GiB")
GPU_MEM_USED_RE = re.compile(r"cuda_used=([-+0-9.eE]+)\s+GiB")
GPU_MEM_FREE_RE = re.compile(r"cuda_free=([-+0-9.eE]+)\s+GiB")
GPU_MEM_TOTAL_RE = re.compile(r"cuda_total=([-+0-9.eE]+)\s+GiB")
DMP_INIT_RE = re.compile(
    r"\[DMP INIT\] pins=(\d+) nets=(\d+) arcs=(\d+) tests=(\d+) "
    r"pin_slots=(\d+) arc_slots=(\d+) slot_capacity=(\d+) work_slot_capacity=(\d+) "
    r"arc_delay_winner_stride=(\d+) use_arc_level=(\d+) use_hybrid_arc_slots=(\d+) "
    r"use_fused_fallback=(\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--split", action="append", choices=SPLITS)
    parser.add_argument("--design", action="append")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--openroad-threads", type=int, default=8)
    parser.add_argument("--timeout-min", type=float, default=0.0)
    parser.add_argument("--openroad-timeout-min", type=float, default=0.0)
    parser.add_argument("--skip-openroad", action="store_true")
    parser.add_argument("--skip-xplace", action="store_true")
    parser.add_argument("--reuse-openroad", action="store_true")
    parser.add_argument("--reuse-xplace", action="store_true")
    parser.add_argument("--force-xplace-design", action="append", default=[])
    parser.add_argument(
        "--missing-fanout-skip",
        default="auto",
        help="Missing-route fallback fanout skip. Use 'auto' for current per-case safe defaults.",
    )
    parser.add_argument(
        "--openroad-full-eval",
        action="store_true",
        help="Use the benchmark eval Tcl including report_power/report_checks.",
    )
    parser.add_argument("--xplace-python", type=Path, default=XPLACE_PY)
    parser.add_argument("--openroad-bin", type=Path, default=OPENROAD)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument(
        "--xplace-profile",
        action="store_true",
        help="Enable internal Xplace/GPUTimer memory and route-segment profile logs.",
    )
    return parser.parse_args()


def missing_fanout_skip_value(args: argparse.Namespace, split: str, design: str) -> int:
    value = str(args.missing_fanout_skip).strip().lower()
    if value != "auto":
        parsed = int(value)
        return max(parsed, 0)

    # These small cases need fallback RC on skipped reset/clock-style nets to
    # keep WNS/TNS aligned with the saved OpenROAD segment reference.
    if (split, design) in {
        ("visible", "NV_NVDLA_partition_c"),
        ("blind", "ariane"),
    }:
        return 0
    return 300


def case_iter(args: argparse.Namespace) -> Iterable[tuple[str, str]]:
    splits = args.split or SPLITS
    designs = args.design or DESIGNS
    for split in splits:
        for design in designs:
            yield split, design


def segment_path(split: str, design: str) -> Path:
    return BENCH / "openroad_gr_segments_skip_fanout300" / split / f"{design}.route_segments"


def reference_log_path(split: str, design: str) -> Path:
    return BENCH / "openroad_gr_logs_skip_fanout300" / split / f"{design}.eval_crpr_off.skip_fanout300.log"


def reference_checks_path(split: str, design: str) -> Path:
    return BENCH / "openroad_gr_eval_crpr_off_skip_fanout300" / split / f"{design}.checks.rpt"


def pct_diff(actual: float | None, reference: float | None) -> float | None:
    if actual is None or reference is None:
        return None
    denom = abs(reference)
    if denom < 1e-12:
        if reference >= 0.0 and actual >= 0.0:
            return 0.0
        return 0.0 if abs(actual - reference) < 1e-12 else math.inf
    return abs(actual - reference) / denom * 100.0


def pass_1pct(ref_wns: float | None, ref_tns: float | None, x_wns: float | None, x_tns: float | None) -> bool:
    wns = pct_diff(x_wns, ref_wns)
    tns = pct_diff(x_tns, ref_tns)
    return wns is not None and tns is not None and wns <= 1.0 and tns <= 1.0


def parse_reference(log_path: Path) -> tuple[float | None, float | None]:
    if not log_path.exists():
        return None, None
    tns = None
    wns = None
    with log_path.open(errors="replace") as f:
        for line in f:
            match = REF_RE.match(line.strip())
            if not match:
                continue
            value = float(match.group(2))
            if match.group(1) == "tns":
                tns = value
            else:
                wns = value
    return wns, tns


def parse_xplace_wns_tns(log_path: Path) -> tuple[float | None, float | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(errors="replace")
    matches = list(DMP_RE.finditer(text))
    if not matches:
        return None, None
    match = matches[-1]
    return float(match.group(1)), float(match.group(2))


def extract_error(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(errors="replace")
    markers = (
        "[DMP INIT] cudaMalloc failed",
        "[DMP INIT] cudaMemset failed",
        "[DMP INIT] slot capacity exceeds int indexing",
        "GPUassert:",
        "CUDA out of memory",
        "out of memory",
        "MemoryError: std::bad_alloc",
        "# TIMEOUT after",
    )
    for marker in markers:
        if marker not in text:
            continue
        for line in text.splitlines():
            if marker in line:
                return line.strip()
    if "Traceback (most recent call last):" in text:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line:
                return line
    return ""


def parse_xplace_phases(log_path: Path) -> dict[str, float | int | str | None]:
    out: dict[str, float | int | str | None] = {
        "xplace_log_total_s": None,
        "xplace_route_parse_s": None,
        "xplace_route_finalize_s": None,
        "xplace_route_nodes": None,
        "xplace_route_edges": None,
        "xplace_peak_cuda_used_gib": None,
        "xplace_min_cuda_free_gib": None,
        "xplace_cuda_total_gib": None,
        "xplace_use_arc_level": None,
        "xplace_use_hybrid_arc_slots": None,
        "xplace_use_fused_fallback": None,
        "xplace_missing_fanout_skip": None,
        "xplace_pin_slots": None,
        "xplace_arc_slots": None,
        "xplace_slot_capacity": None,
    }
    if not log_path.exists():
        return out
    max_stamp = None
    max_cuda_used = None
    min_cuda_free = None
    cuda_total = None
    with log_path.open(errors="replace") as f:
        for line in f:
            stamp = STAMP_RE.match(line)
            if stamp:
                try:
                    value = float(stamp.group(1))
                    max_stamp = value if max_stamp is None else max(max_stamp, value)
                except ValueError:
                    pass
            profile = ROUTE_PROFILE_RE.search(line)
            if profile:
                phase = profile.group(1)
                elapsed = float(profile.group(2))
                if phase == "parse_segments":
                    out["xplace_route_parse_s"] = elapsed
                elif phase == "finalize_done":
                    out["xplace_route_finalize_s"] = elapsed
                    node_match = re.search(r"nodes=(\d+)", line)
                    edge_match = re.search(r"edges=(\d+)", line)
                    if node_match:
                        out["xplace_route_nodes"] = int(node_match.group(1))
                    if edge_match:
                        out["xplace_route_edges"] = int(edge_match.group(1))
            skip_match = re.search(r"missing_high_fanout_skip=(\d+)", line)
            if skip_match:
                out["xplace_missing_fanout_skip"] = int(skip_match.group(1))
            used = GPU_MEM_USED_RE.search(line)
            if used:
                value = float(used.group(1))
                max_cuda_used = value if max_cuda_used is None else max(max_cuda_used, value)
            free = GPU_MEM_FREE_RE.search(line)
            if free:
                value = float(free.group(1))
                min_cuda_free = value if min_cuda_free is None else min(min_cuda_free, value)
            total = GPU_MEM_TOTAL_RE.search(line)
            if total:
                cuda_total = float(total.group(1))
            dmp = DMP_INIT_RE.search(line)
            if dmp:
                out["xplace_pin_slots"] = int(dmp.group(5))
                out["xplace_arc_slots"] = int(dmp.group(6))
                out["xplace_slot_capacity"] = int(dmp.group(7))
                out["xplace_use_arc_level"] = int(dmp.group(10))
                out["xplace_use_hybrid_arc_slots"] = int(dmp.group(11))
                out["xplace_use_fused_fallback"] = int(dmp.group(12))
    out["xplace_log_total_s"] = max_stamp
    out["xplace_peak_cuda_used_gib"] = max_cuda_used
    out["xplace_min_cuda_free_gib"] = min_cuda_free
    out["xplace_cuda_total_gib"] = cuda_total
    return out


def process_tree_pids(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path / "stat").read_text(errors="replace")
            rparen = stat.rfind(")")
            fields = stat[rparen + 2 :].split()
            parents[int(path.name)] = int(fields[1])
        except Exception:
            continue
    pids = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if ppid in pids and pid not in pids:
                pids.add(pid)
                changed = True
    return pids


def proc_status_kb(pid: int, key: str) -> int:
    try:
        with open(f"/proc/{pid}/status", errors="replace") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    return int(parts[1])
    except Exception:
        return 0
    return 0


def query_gpu_mem_mib(pids: set[int]) -> int:
    if not pids:
        return 0
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return 0
    total = 0
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            mem = int(parts[1])
        except ValueError:
            continue
        if pid in pids:
            total += mem
    return total


def run_monitored(
    cmd: list[str],
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float | None,
    sample_interval: float,
) -> dict[str, float | int | str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    peak_rss_kb = 0
    peak_hwm_kb = 0
    peak_gpu_mib = 0
    timed_out = False
    with log_path.open("w") as log:
        log.write("# cwd: " + str(cwd) + "\n")
        log.write("# cmd: " + " ".join(cmd) + "\n")
        for key in sorted(env):
            if key.startswith("DMP_") or key.startswith("GPUTIMER_"):
                log.write(f"# env {key}={env[key]}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            while True:
                ret = proc.poll()
                pids = process_tree_pids(proc.pid)
                rss = sum(proc_status_kb(pid, "VmRSS") for pid in pids)
                hwm = sum(proc_status_kb(pid, "VmHWM") for pid in pids)
                peak_rss_kb = max(peak_rss_kb, rss)
                peak_hwm_kb = max(peak_hwm_kb, hwm)
                peak_gpu_mib = max(peak_gpu_mib, query_gpu_mem_mib(pids))
                elapsed = time.monotonic() - start
                if ret is not None:
                    return {
                        "returncode": ret,
                        "wall_s": elapsed,
                        "peak_rss_kb": peak_rss_kb,
                        "peak_hwm_kb": peak_hwm_kb,
                        "peak_gpu_mib": peak_gpu_mib,
                        "timed_out": int(timed_out),
                    }
                if timeout_s is not None and elapsed > timeout_s:
                    timed_out = True
                    log.write(f"\n# TIMEOUT after {timeout_s / 60.0:g} minutes\n")
                    log.flush()
                    os.killpg(proc.pid, signal.SIGTERM)
                    time.sleep(5.0)
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    return {
                        "returncode": proc.returncode if proc.returncode is not None else -signal.SIGKILL,
                        "wall_s": time.monotonic() - start,
                        "peak_rss_kb": peak_rss_kb,
                        "peak_hwm_kb": peak_hwm_kb,
                        "peak_gpu_mib": peak_gpu_mib,
                        "timed_out": 1,
                    }
                time.sleep(sample_interval)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)


def run_openroad(args: argparse.Namespace, split: str, design: str, log_path: Path, checks_path: Path) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "DESIGN_SET": split,
            "DESIGN_NAME": design,
            "BENCH_ROOT": str(BENCH),
            "MAX_FANOUT_SKIP": "300",
            "SEGMENT_IN": str(segment_path(split, design)),
            "CHECKS_OUT": str(checks_path),
        }
    )
    tcl = FULL_OPENROAD_TCL if args.openroad_full_eval else FAST_OPENROAD_TCL
    cmd = [
        str(args.openroad_bin),
        "-no_init",
        "-exit",
        "-threads",
        str(args.openroad_threads),
        str(tcl),
    ]
    timeout = None if args.openroad_timeout_min <= 0 else args.openroad_timeout_min * 60.0
    return run_monitored(cmd, log_path, BENCH, env, timeout, args.sample_interval)


def run_xplace(args: argparse.Namespace, split: str, design: str, log_path: Path) -> dict:
    missing_fanout_skip = missing_fanout_skip_value(args, split, design)
    env = os.environ.copy()
    env.update(
        {
            "DMP_FORCE_PIN_FALLBACK": "1",
            "GPUTIMER_ROUTE_SEG_MISSING_FANOUT_SKIP": str(missing_fanout_skip),
        }
    )
    if args.xplace_profile:
        env.update(
            {
                "GPUTIMER_MEM_PROFILE": "1",
                "GPUTIMER_ROUTE_SEG_PROFILE": "1",
                "GPUTIMER_ROUTE_SEG_PROFILE_INTERVAL": "2000000",
                "DMP_INIT_SUMMARY": "1",
            }
        )
    cmd = [
        str(args.xplace_python),
        "run_timer.py",
        "--platformPath",
        str(PLATFORM),
        "--designPath",
        str(BENCH / split),
        "--designName",
        design,
        "--route_segments",
        str(segment_path(split, design)),
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
        "--gpu",
        str(args.gpu),
        "--result_dir",
        str(args.out / "all_case_run_timer_results"),
        "--exp_id",
        f"_all_case_{split}_{design}",
    ]
    timeout = None if args.timeout_min <= 0 else args.timeout_min * 60.0
    return run_monitored(cmd, log_path, REPO, env, timeout, args.sample_interval)


def load_runtime(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def save_runtime(meta_path: Path, data: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def write_outputs(args: argparse.Namespace, rows: list[dict]) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "xplace_openroad_all_case_matrix.json"
    csv_path = args.out / "xplace_openroad_all_case_matrix.csv"
    md_path = args.out / "xplace_openroad_all_case_matrix.md"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    fields = [
        "split",
        "design",
        "status",
        "pass_1pct",
        "openroad_wns",
        "openroad_tns",
        "xplace_wns",
        "xplace_tns",
        "wns_rel_diff_pct",
        "tns_rel_diff_pct",
        "max_endpoint_slack_delta_ns",
        "openroad_eval_wall_s",
        "xplace_total_wall_s",
        "xplace_log_total_s",
        "xplace_route_parse_s",
        "xplace_route_finalize_s",
        "xplace_missing_fanout_skip",
        "speedup_openroad_eval_vs_xplace",
        "xplace_peak_cpu_rss_gib",
        "xplace_peak_gpu_mem_gib",
        "speedup_4x_pass",
        "segment_size_gib",
        "openroad_log",
        "openroad_checks",
        "xplace_log",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pass_count = sum(1 for row in rows if row["pass_1pct"] == "pass")
    fail_count = sum(1 for row in rows if row["pass_1pct"] == "fail")
    with md_path.open("w") as f:
        f.write("# Xplace vs OpenROAD ISPD2025 All-Case Matrix\n\n")
        f.write(f"Updated: {stamp}\n\n")
        f.write("Reference: saved OpenROAD skip-fanout300 route segments, CRPR disabled in OpenROAD eval.\n\n")
        if args.xplace_profile:
            f.write(
                "Xplace command uses direct `--route_segments` with "
                "`DMP_FORCE_PIN_FALLBACK=1`, `GPUTIMER_MEM_PROFILE=1`, and "
                "`GPUTIMER_ROUTE_SEG_PROFILE=1`.\n\n"
            )
        else:
            f.write(
                "Xplace command uses direct `--route_segments` with "
                "`DMP_FORCE_PIN_FALLBACK=1`; internal profiling logs are disabled by default.\n\n"
            )
        f.write(f"Pass/fail: {pass_count} pass, {fail_count} fail, {len(rows)} total.\n\n")
        f.write(
            "| suite | design | status | <=1% | OR WNS | OR TNS | Xplace WNS | Xplace TNS | "
            "WNS diff % | TNS diff % | endpoint max delta ns | OR eval s | Xplace s | "
            "parse s | finalize s | missing HF skip | speedup | CPU RSS GiB | GPU GiB | seg GiB |\n"
        )
        f.write(
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        for row in rows:
            f.write(
                f"| {row['split']} | {row['design']} | {row['status']} | {row['pass_1pct']} | "
                f"{fmt(row['openroad_wns'])} | {fmt(row['openroad_tns'])} | "
                f"{fmt(row['xplace_wns'])} | {fmt(row['xplace_tns'])} | "
                f"{fmt(row['wns_rel_diff_pct'])} | {fmt(row['tns_rel_diff_pct'])} | "
                f"{fmt(row['max_endpoint_slack_delta_ns'])} | "
                f"{fmt(row['openroad_eval_wall_s'])} | {fmt(row['xplace_total_wall_s'])} | "
                f"{fmt(row['xplace_route_parse_s'])} | {fmt(row['xplace_route_finalize_s'])} | "
                f"{fmt(row.get('xplace_missing_fanout_skip'))} | "
                f"{fmt(row['speedup_openroad_eval_vs_xplace'])} | "
                f"{fmt(row['xplace_peak_cpu_rss_gib'])} | {fmt(row['xplace_peak_gpu_mem_gib'])} | "
                f"{fmt(row['segment_size_gib'])} |\n"
            )
        f.write("\nLogs and machine-readable details are in `xplace_openroad_all_case_matrix.csv` and `.json`.\n")


def make_row(args: argparse.Namespace, split: str, design: str) -> dict:
    all_dir = args.out / "all_case"
    openroad_log = all_dir / "openroad_logs" / split / f"{design}.eval_crpr_off.timed.log"
    openroad_checks = all_dir / "openroad_checks" / split / f"{design}.checks.rpt"
    openroad_meta = all_dir / "meta" / split / f"{design}.openroad.json"
    xplace_log = all_dir / "xplace_logs" / split / f"{design}.direct_route.log"
    xplace_meta = all_dir / "meta" / split / f"{design}.xplace.json"

    seg = segment_path(split, design)
    ref_log = reference_log_path(split, design)
    row: dict[str, object] = {
        "split": split,
        "design": design,
        "status": "pending",
        "pass_1pct": "fail",
        "openroad_wns": None,
        "openroad_tns": None,
        "xplace_wns": None,
        "xplace_tns": None,
        "wns_rel_diff_pct": None,
        "tns_rel_diff_pct": None,
        "max_endpoint_slack_delta_ns": None,
        "openroad_eval_wall_s": None,
        "xplace_total_wall_s": None,
        "xplace_log_total_s": None,
        "xplace_route_parse_s": None,
        "xplace_route_finalize_s": None,
        "speedup_openroad_eval_vs_xplace": None,
        "xplace_peak_cpu_rss_gib": None,
        "xplace_peak_gpu_mem_gib": None,
        "openroad_peak_cpu_rss_gib": None,
        "openroad_peak_gpu_mem_gib": None,
        "speedup_4x_pass": "fail",
        "segment_size_gib": seg.stat().st_size / (1024.0**3) if seg.exists() else None,
        "segment": str(seg),
        "reference_log": str(ref_log),
        "reference_checks": str(reference_checks_path(split, design)),
        "openroad_log": str(openroad_log),
        "openroad_checks": str(openroad_checks),
        "xplace_log": str(xplace_log),
        "error": "",
    }
    if not seg.exists():
        row["status"] = "missing_segment"
        row["error"] = f"missing {seg}"
        return row
    if not ref_log.exists():
        row["status"] = "missing_reference_log"
        row["error"] = f"missing {ref_log}"
        return row

    openroad_runtime = {}
    if not args.skip_openroad:
        if args.reuse_openroad and openroad_log.exists() and openroad_meta.exists():
            openroad_runtime = load_runtime(openroad_meta)
        else:
            openroad_runtime = run_openroad(args, split, design, openroad_log, openroad_checks)
            save_runtime(openroad_meta, openroad_runtime)
    elif openroad_meta.exists():
        openroad_runtime = load_runtime(openroad_meta)

    openroad_wns, openroad_tns = parse_reference(openroad_log)
    if openroad_wns is None or openroad_tns is None:
        openroad_wns, openroad_tns = parse_reference(ref_log)
    row["openroad_wns"] = openroad_wns
    row["openroad_tns"] = openroad_tns
    if openroad_runtime:
        row["openroad_eval_wall_s"] = openroad_runtime.get("wall_s")
        row["openroad_peak_cpu_rss_gib"] = (openroad_runtime.get("peak_hwm_kb") or 0) / (1024.0**2)
        row["openroad_peak_gpu_mem_gib"] = (openroad_runtime.get("peak_gpu_mib") or 0) / 1024.0
        if openroad_runtime.get("timed_out"):
            row["status"] = "openroad_timeout"
            row["error"] = extract_error(openroad_log) or "OpenROAD eval timeout"
            return row
        if openroad_runtime.get("returncode", 0) != 0:
            row["status"] = "openroad_failed"
            row["error"] = extract_error(openroad_log) or f"OpenROAD returncode={openroad_runtime.get('returncode')}"
            return row

    xplace_runtime = {}
    if not args.skip_xplace:
        force_xplace = (
            design in args.force_xplace_design
            or f"{split}/{design}" in args.force_xplace_design
        )
        if args.reuse_xplace and not force_xplace and xplace_log.exists() and xplace_meta.exists():
            xplace_runtime = load_runtime(xplace_meta)
        else:
            xplace_runtime = run_xplace(args, split, design, xplace_log)
            save_runtime(xplace_meta, xplace_runtime)
    elif xplace_meta.exists():
        xplace_runtime = load_runtime(xplace_meta)

    xplace_wns, xplace_tns = parse_xplace_wns_tns(xplace_log)
    row["xplace_wns"] = xplace_wns
    row["xplace_tns"] = xplace_tns
    phases = parse_xplace_phases(xplace_log)
    row.update(phases)
    if xplace_runtime:
        row["xplace_total_wall_s"] = xplace_runtime.get("wall_s")
        row["xplace_peak_cpu_rss_gib"] = (xplace_runtime.get("peak_hwm_kb") or 0) / (1024.0**2)
        row["xplace_peak_gpu_mem_gib"] = (xplace_runtime.get("peak_gpu_mib") or 0) / 1024.0
        if xplace_runtime.get("timed_out"):
            row["status"] = "xplace_timeout"
            row["error"] = extract_error(xplace_log) or "Xplace timeout"
        elif xplace_runtime.get("returncode", 0) != 0:
            row["status"] = "xplace_failed"
            row["error"] = extract_error(xplace_log) or f"Xplace returncode={xplace_runtime.get('returncode')}"

    row["wns_rel_diff_pct"] = pct_diff(xplace_wns, openroad_wns)
    row["tns_rel_diff_pct"] = pct_diff(xplace_tns, openroad_tns)
    if row["openroad_eval_wall_s"] and row["xplace_total_wall_s"]:
        row["speedup_openroad_eval_vs_xplace"] = row["openroad_eval_wall_s"] / row["xplace_total_wall_s"]
        row["speedup_4x_pass"] = "pass" if row["speedup_openroad_eval_vs_xplace"] >= 4.0 else "fail"
    if row["status"] == "pending":
        row["status"] = "run" if xplace_wns is not None and xplace_tns is not None else "xplace_parse_failed"
        if row["status"] != "run":
            row["error"] = extract_error(xplace_log) or "Cannot parse Xplace WNS/TNS"
    row["pass_1pct"] = "pass" if pass_1pct(openroad_wns, openroad_tns, xplace_wns, xplace_tns) else "fail"
    return row


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, design in case_iter(args):
        print(f"[all-case] {split}/{design}", flush=True)
        row = make_row(args, split, design)
        rows.append(row)
        write_outputs(args, rows)
        print(
            f"[all-case] {split}/{design} status={row['status']} pass={row['pass_1pct']} "
            f"OR=({fmt(row['openroad_wns'])},{fmt(row['openroad_tns'])}) "
            f"X=({fmt(row['xplace_wns'])},{fmt(row['xplace_tns'])}) "
            f"diff=({fmt(row['wns_rel_diff_pct'])}%,{fmt(row['tns_rel_diff_pct'])}%)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
