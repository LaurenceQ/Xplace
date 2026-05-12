#!/usr/bin/env python3
"""Compare Xplace/GPUTimer DMP SPEF timing against OpenROAD CSV pin labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_DIR = Path("/research/d7/ascstd/qkduan25/GNNTimer/csv_graph_sky130")
ATTR_NAMES = ("er", "ef", "lr", "lf")
COMPARE_KINDS = ("at", "slew", "rat", "slack")


def _strip_pin_name(name: str) -> str:
    name = name.strip().strip('"')
    return name


def _unescape_openroad_pin_name(name: str) -> str:
    """Unescape common OpenROAD report/SPEF name escapes without changing hierarchy."""
    name = _strip_pin_name(name)
    out: list[str] = []
    i = 0
    while i < len(name):
        ch = name[i]
        if ch == "\\" and i + 1 < len(name):
            nxt = name[i + 1]
            if nxt in "[]{}()/.:\\ ":
                out.append(nxt)
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _final_slash_to_colon(name: str) -> str:
    if "/" in name:
        head, tail = name.rsplit("/", 1)
        return head + ":" + tail
    return name


def pin_name_aliases(name: str) -> list[str]:
    """Aliases for matching OpenROAD pin names to Xplace `inst:port` names."""
    raw = _strip_pin_name(name)
    unescaped = _unescape_openroad_pin_name(raw)
    aliases = {
        raw,
        unescaped,
        _final_slash_to_colon(raw),
        _final_slash_to_colon(unescaped),
    }
    return [alias for alias in aliases if alias]


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _csv_float(value: str) -> float:
    value = value.strip()
    if not value:
        return float("nan")
    return float(value)


def _csv_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _tensor_ns(tensor: Any, time_to_ns: float) -> list[list[float]]:
    return (tensor.detach().cpu().float() * time_to_ns).tolist()


def _make_alias_lookup(records: dict[str, dict[str, Any]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    alias_to_name: dict[str, str] = {}
    collisions: dict[str, list[str]] = {}
    for name in records:
        for alias in pin_name_aliases(name):
            prev = alias_to_name.get(alias)
            if prev is None:
                alias_to_name[alias] = name
            elif prev != name:
                collisions.setdefault(alias, sorted({prev, name}))
    for alias in collisions:
        alias_to_name.pop(alias, None)
    return alias_to_name, collisions


def _match_pin_records(
    dmp: dict[str, dict[str, Any]],
    opr: dict[str, dict[str, Any]],
) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    dmp_alias, dmp_collisions = _make_alias_lookup(dmp)
    matches: dict[str, tuple[str, str]] = {}
    used_dmp: set[str] = set()
    alias_hit_counts: dict[str, int] = {}

    for opr_name in sorted(opr):
        for alias in pin_name_aliases(opr_name):
            dmp_name = dmp_alias.get(alias)
            if dmp_name is None:
                continue
            matches[opr_name] = (dmp_name, alias)
            used_dmp.add(dmp_name)
            alias_hit_counts[alias] = alias_hit_counts.get(alias, 0) + 1
            break

    missing_opr = sorted(set(opr) - set(matches))
    missing_dmp = sorted(set(dmp) - used_dmp)
    stats = {
        "matched_pins": len(matches),
        "missing_in_dmp": len(missing_opr),
        "missing_in_openroad": len(missing_dmp),
        "missing_in_dmp_examples": [
            {"openroad_pin": name, "aliases": pin_name_aliases(name)}
            for name in missing_opr[:20]
        ],
        "missing_in_openroad_examples": [
            {"xplace_pin": name, "aliases": pin_name_aliases(name)}
            for name in missing_dmp[:20]
        ],
        "dmp_alias_collisions": len(dmp_collisions),
        "dmp_alias_collision_examples": [
            {"alias": alias, "pins": pins}
            for alias, pins in sorted(dmp_collisions.items())[:10]
        ],
    }
    return matches, stats


def _slack_summary(records: dict[str, dict[str, Any]], names: list[str]) -> dict[str, Any]:
    per_attr: dict[str, dict[str, float | int]] = {}
    for attr, attr_name in enumerate(ATTR_NAMES):
        vals = [
            float(records[name]["slack"][attr])
            for name in names
            if name in records and _finite(float(records[name]["slack"][attr]))
        ]
        per_attr[attr_name] = {
            "count": len(vals),
            "wns_ns": min(vals) if vals else float("nan"),
            "tns_ns": sum(v for v in vals if v < 0.0),
        }

    def grouped(label: str, attrs: tuple[int, int]) -> dict[str, float | int]:
        vals: list[float] = []
        for name in names:
            if name not in records:
                continue
            cand = [float(records[name]["slack"][attr]) for attr in attrs]
            cand = [v for v in cand if _finite(v)]
            if cand:
                vals.append(min(cand))
        return {
            "count": len(vals),
            "wns_ns": min(vals) if vals else float("nan"),
            "tns_ns": sum(v for v in vals if v < 0.0),
        }

    return {
        "per_attr": per_attr,
        "early": grouped("early", (0, 1)),
        "late": grouped("late", (2, 3)),
    }


def discover_designs(csv_dir: Path, design_path: Path) -> list[str]:
    csv_designs = {p.name[: -len("_pins.csv")] for p in csv_dir.glob("*_pins.csv")}
    netlist_designs = set()
    for design_dir in design_path.iterdir():
        if not design_dir.is_dir():
            continue
        design = design_dir.name
        if (design_dir / f"20-{design}.spef").exists() and (design_dir / f"{design}.cts_1.sdc").exists():
            netlist_designs.add(design)
    return sorted(csv_designs & netlist_designs)


def load_openroad_pins(path: Path) -> dict[str, dict[str, list[float]]]:
    pins: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pin_name = _strip_pin_name(row["pin_name"])
            pins[pin_name] = {
                "openroad_name": pin_name,
                "is_endpoint": _csv_bool(row.get("is_endpoint", "")),
                "is_clock": _csv_bool(row.get("is_clock", "")),
                "is_ideal_clock": _csv_bool(row.get("is_ideal_clock", "")),
                "at": [
                    _csv_float(row["arrival_default_min_rise_ns"]),
                    _csv_float(row["arrival_default_min_fall_ns"]),
                    _csv_float(row["arrival_default_max_rise_ns"]),
                    _csv_float(row["arrival_default_max_fall_ns"]),
                ],
                "slew": [
                    _csv_float(row["slew_default_min_rise_ns"]),
                    _csv_float(row["slew_default_min_fall_ns"]),
                    _csv_float(row["slew_default_max_rise_ns"]),
                    _csv_float(row["slew_default_max_fall_ns"]),
                ],
                "rat": [
                    _csv_float(row["required_default_min_rise_ns"]),
                    _csv_float(row["required_default_min_fall_ns"]),
                    _csv_float(row["required_default_max_rise_ns"]),
                    _csv_float(row["required_default_max_fall_ns"]),
                ],
                "slack": [
                    _csv_float(row["slack_default_min_rise_ns"]),
                    _csv_float(row["slack_default_min_fall_ns"]),
                    _csv_float(row["slack_default_max_rise_ns"]),
                    _csv_float(row["slack_default_max_fall_ns"]),
                ],
            }
    return pins


def load_dmp_dump(path: Path) -> dict[str, dict[str, list[float]]]:
    pins: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "node":
                continue
            labels = rec["labels"]
            pins[rec["name"]] = {
                "wire": labels[0:4],
                "at": labels[4:8],
                "slew": labels[8:12],
                "is_endpoint": bool(labels[12]) if len(labels) > 12 else False,
                "rat": labels[13:17] if len(labels) >= 17 else [float("nan")] * 4,
                "slack": [
                    labels[4] - labels[13] if len(labels) >= 17 else float("nan"),
                    labels[5] - labels[14] if len(labels) >= 17 else float("nan"),
                    labels[15] - labels[6] if len(labels) >= 17 else float("nan"),
                    labels[16] - labels[7] if len(labels) >= 17 else float("nan"),
                ],
            }
    return pins


def _diff_stats(values: list[float], signed: list[float]) -> dict[str, float | int]:
    return {
        "valid_lanes": len(values),
        "max_abs_ns": max(values) if values else float("nan"),
        "mean_abs_ns": sum(values) / len(values) if values else float("nan"),
        "p95_abs_ns": _percentile(values, 0.95),
        "signed_mean_ns": sum(signed) / len(signed) if signed else float("nan"),
    }


def collect_dmp_pins(gputimer: Any) -> dict[str, dict[str, Any]]:
    time_to_ns = gputimer.timer.time_unit() * 1e9
    at = _tensor_ns(gputimer.timer.report_pin_at(), time_to_ns)
    slew = _tensor_ns(gputimer.timer.report_pin_slew(), time_to_ns)
    rat = _tensor_ns(gputimer.timer.report_pin_rat(), time_to_ns)
    slack = _tensor_ns(gputimer.timer.report_pin_slack(), time_to_ns)
    endpoints = set(int(v) for v in gputimer.timer.endpoints_index().detach().cpu().long().tolist())

    pins: dict[str, dict[str, Any]] = {}
    for pin_id, name in enumerate(gputimer.pin_names):
        pins[name] = {
            "xplace_name": name,
            "pin_id": pin_id,
            "is_endpoint": pin_id in endpoints,
            "at": at[pin_id],
            "slew": slew[pin_id],
            "rat": rat[pin_id],
            "slack": slack[pin_id],
        }
    return pins


def compare_pins(design: str, dmp: dict[str, Any], opr: dict[str, Any], top_n: int) -> dict[str, Any]:
    matches, match_stats = _match_pin_records(dmp, opr)
    matched_opr = sorted(matches)
    summary: dict[str, Any] = {
        "design": design,
        "dmp_pins": len(dmp),
        "openroad_pins": len(opr),
        "common_pins": len(matches),
        **match_stats,
    }

    openroad_endpoint_names = [name for name, rec in opr.items() if rec.get("is_endpoint")]
    matched_openroad_endpoint_names = [name for name in openroad_endpoint_names if name in matches]
    matched_dmp_endpoint_names = [matches[name][0] for name in matched_openroad_endpoint_names]
    dmp_endpoint_names = [name for name, rec in dmp.items() if rec.get("is_endpoint")]
    summary["openroad_endpoint_pins"] = len(openroad_endpoint_names)
    summary["matched_openroad_endpoint_pins"] = len(matched_openroad_endpoint_names)
    summary["dmp_endpoint_pins"] = len(dmp_endpoint_names)

    opr_slack_report = _slack_summary(opr, matched_openroad_endpoint_names)
    dmp_slack_report = _slack_summary(dmp, matched_dmp_endpoint_names)
    summary["openroad_slack_on_matched_endpoints"] = opr_slack_report
    summary["dmp_slack_on_openroad_endpoints"] = dmp_slack_report
    for group in ("early", "late"):
        opr_group = opr_slack_report[group]
        dmp_group = dmp_slack_report[group]
        for metric in ("wns_ns", "tns_ns"):
            opr_value = float(opr_group[metric])
            dmp_value = float(dmp_group[metric])
            summary[f"{group}_{metric}_openroad"] = opr_value
            summary[f"{group}_{metric}_dmp"] = dmp_value
            summary[f"{group}_{metric}_diff"] = dmp_value - opr_value if _finite(opr_value) and _finite(dmp_value) else float("nan")

    for kind in COMPARE_KINDS:
        rows: list[tuple[float, float, str, str, str, str, float, float, bool]] = []
        diffs: list[float] = []
        signed: list[float] = []
        dmp_missing_finite = 0
        openroad_missing_finite = 0
        for opr_name in matched_opr:
            dmp_name, matched_alias = matches[opr_name]
            for attr, attr_name in enumerate(ATTR_NAMES):
                dmp_value = float(dmp[dmp_name][kind][attr])
                opr_value = float(opr[opr_name][kind][attr])
                dmp_finite = _finite(dmp_value)
                opr_finite = _finite(opr_value)
                if opr_finite and not dmp_finite:
                    dmp_missing_finite += 1
                if dmp_finite and not opr_finite:
                    openroad_missing_finite += 1
                if not (dmp_finite and opr_finite):
                    continue
                diff = dmp_value - opr_value
                abs_diff = abs(diff)
                rows.append((
                    abs_diff,
                    diff,
                    dmp_name,
                    opr_name,
                    matched_alias,
                    attr_name,
                    dmp_value,
                    opr_value,
                    bool(opr[opr_name].get("is_endpoint")),
                ))
                diffs.append(abs_diff)
                signed.append(diff)
        rows.sort(reverse=True, key=lambda item: item[0])
        summary[f"{kind}_valid_lanes"] = len(diffs)
        summary[f"{kind}_dmp_missing_finite_lanes"] = dmp_missing_finite
        summary[f"{kind}_openroad_missing_finite_lanes"] = openroad_missing_finite
        summary[f"{kind}_max_abs_ns"] = diffs and max(diffs) or float("nan")
        summary[f"{kind}_mean_abs_ns"] = sum(diffs) / len(diffs) if diffs else float("nan")
        summary[f"{kind}_p95_abs_ns"] = _percentile(diffs, 0.95)
        summary[f"{kind}_signed_mean_ns"] = sum(signed) / len(signed) if signed else float("nan")
        summary[f"top_{kind}"] = [
            {
                "pin": dmp_name,
                "openroad_pin": opr_name,
                "matched_alias": matched_alias,
                "attr": attr_name,
                "is_openroad_endpoint": is_endpoint,
                "abs_diff_ns": abs_diff,
                "diff_ns": diff,
                "dmp_ns": dmp_value,
                "openroad_ns": opr_value,
            }
            for abs_diff, diff, dmp_name, opr_name, matched_alias, attr_name, dmp_value, opr_value, is_endpoint in rows[:top_n]
        ]
        endpoint_rows = [row for row in rows if row[8]]
        summary[f"top_endpoint_{kind}"] = [
            {
                "pin": dmp_name,
                "openroad_pin": opr_name,
                "matched_alias": matched_alias,
                "attr": attr_name,
                "abs_diff_ns": abs_diff,
                "diff_ns": diff,
                "dmp_ns": dmp_value,
                "openroad_ns": opr_value,
            }
            for abs_diff, diff, dmp_name, opr_name, matched_alias, attr_name, dmp_value, opr_value, _ in endpoint_rows[:top_n]
        ]

    rat_groups = {
        "endpoint": lambda rec: bool(rec.get("is_endpoint")),
        "non_clock": lambda rec: not bool(rec.get("is_clock")),
        "clock": lambda rec: bool(rec.get("is_clock")),
        "ideal_clock": lambda rec: bool(rec.get("is_ideal_clock")),
    }
    for group_name, pred in rat_groups.items():
        diffs = []
        signed = []
        dmp_missing_finite = 0
        openroad_missing_finite = 0
        pin_count = 0
        for opr_name in matched_opr:
            opr_rec = opr[opr_name]
            if not pred(opr_rec):
                continue
            pin_count += 1
            dmp_name, _ = matches[opr_name]
            for attr in range(len(ATTR_NAMES)):
                dmp_value = float(dmp[dmp_name]["rat"][attr])
                opr_value = float(opr_rec["rat"][attr])
                dmp_finite = _finite(dmp_value)
                opr_finite = _finite(opr_value)
                if opr_finite and not dmp_finite:
                    dmp_missing_finite += 1
                if dmp_finite and not opr_finite:
                    openroad_missing_finite += 1
                if not (dmp_finite and opr_finite):
                    continue
                diff = dmp_value - opr_value
                diffs.append(abs(diff))
                signed.append(diff)
        prefix = f"rat_{group_name}"
        group_stats = _diff_stats(diffs, signed)
        summary[f"{prefix}_pins"] = pin_count
        summary[f"{prefix}_dmp_missing_finite_lanes"] = dmp_missing_finite
        summary[f"{prefix}_openroad_missing_finite_lanes"] = openroad_missing_finite
        for key, value in group_stats.items():
            summary[f"{prefix}_{key}"] = value
    return summary


def run_worker(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from run_timer import getArgs
    from utils import set_random_seed, setup_logger
    from src import Flute, GPUTimer, load_design
    import torch

    design = args.worker_design
    start = time.time()
    out_dir = args.out_dir.resolve()
    dump_dir = out_dir / "dumps"
    summary_dir = out_dir / "summaries"
    dump_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    timer_argv = [
        "compare_dmp_openroad_csv",
        "--designName",
        design,
        "--platformPath",
        str(args.platform_path),
        "--designPath",
        str(args.design_path),
        "--gpu",
        str(args.gpu),
        "--load_from_raw",
        "True",
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = timer_argv
        Flute.register(8)
        timer_args = getArgs()
        logger = setup_logger(timer_args, timer_argv)
        set_random_seed(timer_args)
        data, rawdb, gpdb, params = load_design(timer_args, logger)
        device = torch.device(f"cuda:{timer_args.gpu}" if torch.cuda.is_available() else "cpu")
        data = data.to(device).preprocess()
        gputimer = GPUTimer(data, rawdb, gpdb, params, timer_args)
        if "spef" not in params or not Path(params["spef"]).exists():
            raise FileNotFoundError(f"SPEF file not found: {params.get('spef')}")
        gputimer.update_timing_dmp_spef()

        dmp = collect_dmp_pins(gputimer)
        opr = load_openroad_pins(args.csv_dir / f"{design}_pins.csv")
        summary = compare_pins(design, dmp, opr, args.top_n)
        summary["status"] = "ok"
        summary["elapsed_s"] = time.time() - start
        summary["dump_path"] = ""
        if args.keep_dump:
            dump_path = dump_dir / f"{design}.dmp.jsonl"
            gputimer.timer.dump_timing_graph(str(dump_path))
            summary["dump_path"] = str(dump_path)
    except Exception as exc:  # noqa: BLE001 - worker records failures per design.
        summary = {
            "design": design,
            "status": "error",
            "error": repr(exc),
            "elapsed_s": time.time() - start,
        }
    finally:
        sys.argv = old_argv

    with (summary_dir / f"{design}.summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "ok" else 1


def write_aggregate(out_dir: Path, summaries: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)

    fields = [
        "design",
        "status",
        "elapsed_s",
        "dmp_pins",
        "openroad_pins",
        "common_pins",
        "matched_pins",
        "missing_in_dmp",
        "missing_in_openroad",
        "openroad_endpoint_pins",
        "matched_openroad_endpoint_pins",
        "dmp_endpoint_pins",
        "at_valid_lanes",
        "at_max_abs_ns",
        "at_mean_abs_ns",
        "at_p95_abs_ns",
        "at_signed_mean_ns",
        "rat_valid_lanes",
        "rat_max_abs_ns",
        "rat_mean_abs_ns",
        "rat_p95_abs_ns",
        "rat_signed_mean_ns",
        "rat_endpoint_pins",
        "rat_endpoint_valid_lanes",
        "rat_endpoint_max_abs_ns",
        "rat_endpoint_mean_abs_ns",
        "rat_endpoint_p95_abs_ns",
        "rat_endpoint_signed_mean_ns",
        "rat_endpoint_dmp_missing_finite_lanes",
        "rat_endpoint_openroad_missing_finite_lanes",
        "rat_non_clock_pins",
        "rat_non_clock_valid_lanes",
        "rat_non_clock_max_abs_ns",
        "rat_non_clock_mean_abs_ns",
        "rat_non_clock_p95_abs_ns",
        "rat_non_clock_signed_mean_ns",
        "rat_non_clock_dmp_missing_finite_lanes",
        "rat_non_clock_openroad_missing_finite_lanes",
        "rat_clock_pins",
        "rat_clock_valid_lanes",
        "rat_clock_max_abs_ns",
        "rat_clock_mean_abs_ns",
        "rat_clock_p95_abs_ns",
        "rat_clock_signed_mean_ns",
        "rat_clock_dmp_missing_finite_lanes",
        "rat_clock_openroad_missing_finite_lanes",
        "rat_ideal_clock_pins",
        "rat_ideal_clock_valid_lanes",
        "rat_ideal_clock_max_abs_ns",
        "rat_ideal_clock_mean_abs_ns",
        "rat_ideal_clock_p95_abs_ns",
        "rat_ideal_clock_signed_mean_ns",
        "rat_ideal_clock_dmp_missing_finite_lanes",
        "rat_ideal_clock_openroad_missing_finite_lanes",
        "slack_valid_lanes",
        "slack_max_abs_ns",
        "slack_mean_abs_ns",
        "slack_p95_abs_ns",
        "slack_signed_mean_ns",
        "slew_valid_lanes",
        "slew_max_abs_ns",
        "slew_mean_abs_ns",
        "slew_p95_abs_ns",
        "slew_signed_mean_ns",
        "early_wns_ns_openroad",
        "early_wns_ns_dmp",
        "early_wns_ns_diff",
        "early_tns_ns_openroad",
        "early_tns_ns_dmp",
        "early_tns_ns_diff",
        "late_wns_ns_openroad",
        "late_wns_ns_dmp",
        "late_wns_ns_diff",
        "late_tns_ns_openroad",
        "late_tns_ns_dmp",
        "late_tns_ns_diff",
        "error",
    ]
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, "") for field in fields})


def run_parent(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.designs:
        designs = args.designs
    else:
        designs = discover_designs(args.csv_dir, args.design_path)
    exclude = set(args.exclude_design)
    designs = [design for design in designs if design not in exclude]
    if not designs:
        raise RuntimeError("No designs selected.")

    print("Selected designs:", " ".join(designs))
    summaries: list[dict[str, Any]] = []
    for index, design in enumerate(designs, start=1):
        print(f"[{index}/{len(designs)}] running {design}", flush=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-design",
            design,
            "--csv-dir",
            str(args.csv_dir),
            "--design-path",
            str(args.design_path),
            "--platform-path",
            str(args.platform_path),
            "--out-dir",
            str(args.out_dir),
            "--gpu",
            str(args.gpu),
            "--top-n",
            str(args.top_n),
        ]
        if args.keep_dump:
            cmd.append("--keep-dump")
        log_path = log_dir / f"{design}.log"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        summary_path = args.out_dir / "summaries" / f"{design}.summary.json"
        if summary_path.exists():
            with summary_path.open() as f:
                summary = json.load(f)
        else:
            summary = {"design": design, "status": "error", "error": f"worker exited {proc.returncode}"}
        summary["log_path"] = str(log_path)
        summaries.append(summary)
        status = summary.get("status")
        at = summary.get("at_max_abs_ns")
        slew = summary.get("slew_max_abs_ns")
        rat = summary.get("rat_max_abs_ns")
        slack = summary.get("slack_max_abs_ns")
        ew = summary.get("early_wns_ns_diff")
        et = summary.get("early_tns_ns_diff")
        print(
            f"[{index}/{len(designs)}] {design} status={status} "
            f"at_max={at} slew_max={slew} rat_max={rat} slack_max={slack} "
            f"early_wns_diff={ew} early_tns_diff={et}",
            flush=True,
        )
        write_aggregate(args.out_dir, summaries)
    return 0 if all(s.get("status") == "ok" for s in summaries) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--design-path", type=Path, default=REPO_ROOT / "netlists")
    parser.add_argument("--platform-path", type=Path, default=REPO_ROOT / "sky130hd")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "result" / "dmp_openroad_csv_diff")
    parser.add_argument("--designs", nargs="*", default=[])
    parser.add_argument("--exclude-design", action="append", default=[])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--keep-dump", action="store_true")
    parser.add_argument("--worker-design", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.csv_dir = args.csv_dir.resolve()
    args.design_path = args.design_path.resolve()
    args.platform_path = args.platform_path.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.worker_design:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
