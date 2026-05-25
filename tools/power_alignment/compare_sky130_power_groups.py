#!/usr/bin/env python3
"""Compare sky130 benchmark power by OpenROAD report_power group/component."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DESIGN_PATH = Path("/research/d7/ascstd/qkduan25/TimingPredict/data/netlists")
DEFAULT_PLATFORM_PATH = REPO_ROOT / "sky130hd"
DEFAULT_GNNTIMER_DIR = Path("/research/d7/ascstd/qkduan25/GNNTimer")
DEFAULT_OPENROAD = Path("/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad")
DEFAULT_OPENROAD_ENV = Path("/research/d7/ascstd/qkduan25/app/openroad-deps/env.sh")
DEFAULT_DESIGN_LIST = REPO_ROOT / "logs/power_benchmark_compare_all_20260512_023957/designs.txt"

COMPONENTS = ("internal", "switching", "leakage", "total")
POWER_GROUPS = ("sequential", "combinational", "clock", "macro", "pad")
REPORT_GROUP_NAMES = {
    "Sequential": "sequential",
    "Combinational": "combinational",
    "Clock": "clock",
    "Macro": "macro",
    "Pad": "pad",
}
DEF_COMPONENT_RE = re.compile(r"^\s*-\s+(\S+)\s+\S+")
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def rel_err(value: float, ref: float) -> float:
    if ref == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return abs(value - ref) / abs(ref)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return str(number)
    if number == 0.0:
        return "0"
    if abs(number) >= 10.0:
        return f"{number:.4f}"
    if abs(number) >= 0.01:
        return f"{number:.6f}"
    return f"{number:.6e}"


def read_designs(args: argparse.Namespace) -> list[str]:
    if args.design:
        return args.design
    if args.design_list and args.design_list.exists():
        with args.design_list.open(errors="replace") as f:
            return [line.split("#", 1)[0].strip() for line in f if line.split("#", 1)[0].strip()]
    return sorted(p.name for p in args.design_path.iterdir() if p.is_dir() and p.name != "techlib")


def design_def_path(design_path: Path, design: str) -> Path:
    exact = design_path / design / f"20-{design}.def"
    if exact.exists():
        return exact
    candidates = sorted((design_path / design).glob("*.def"))
    if not candidates:
        raise FileNotFoundError(f"no DEF found for {design} under {design_path / design}")
    return candidates[0]


def iter_def_component_names(path: Path) -> list[str]:
    names: list[str] = []
    in_components = False
    with path.open(errors="replace") as f:
        for line in f:
            stripped = line.lstrip()
            if not in_components:
                if stripped.startswith("COMPONENTS "):
                    in_components = True
                continue
            if stripped.startswith("END COMPONENTS"):
                break
            match = DEF_COMPONENT_RE.match(line)
            if match:
                names.append(match.group(1))
    return names


def zero_groups() -> dict[str, dict[str, float]]:
    return {group: {component: 0.0 for component in COMPONENTS} for group in POWER_GROUPS}


def parse_openroad_report_power(log_path: Path) -> dict[str, Any]:
    groups = zero_groups()
    total = {component: 0.0 for component in COMPONENTS}
    pattern = re.compile(
        rf"^\s*(Sequential|Combinational|Clock|Macro|Pad|Total)\s+"
        rf"({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})\s+({FLOAT_RE})"
    )
    with log_path.open(errors="replace") as f:
        for line in f:
            match = pattern.match(line)
            if not match:
                continue
            label = match.group(1)
            values = {component: float(match.group(i + 2)) for i, component in enumerate(COMPONENTS)}
            if label == "Total":
                total = values
            else:
                groups[REPORT_GROUP_NAMES[label]] = values
    return {"power": total, "power_groups": groups}


def write_openroad_tcl(path: Path) -> None:
    path.write_text(
        """
set script_dir $::env(GNNTIMER_DIR)
set top_proj_dir $script_dir
set design_name $::env(DESIGN_NAME)
source [file join $script_dir lib_setup.tcl]
source [file join $script_dir design_setup.tcl]
foreach lef_file $lefs { read_lef $lef_file }
foreach lib_file $libbest { read_liberty -min $lib_file }
foreach lib_file $libworst { read_liberty -max $lib_file }
read_def $def_file
read_verilog $verilog_netlist
link_design $design_name
read_sdc $sdc_file
read_spef $spef_file
set_propagated_clock [get_clocks *]
if {[file exists $rc_file]} { source $rc_file }
set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance Ohm -distance um -power mW -digits 8
set_units -power mW
report_power -digits 8
""".lstrip()
    )


def run_openroad(args: argparse.Namespace, design: str, tcl_path: Path, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        "DESIGN_NAME": design,
        "GNNTIMER_DIR": str(args.gnntimer_dir.resolve()),
        "TOP_PROJ_DIR": str(args.gnntimer_dir.resolve()),
        "NETLIST_ROOT": str(args.design_path.resolve()),
        "TECH_ROOT": str(args.platform_path.resolve()),
        "TECHLIB_ROOT": str((args.design_path / "techlib").resolve()),
    }
    exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
    source_env = (
        f"set +u; source {shlex.quote(str(args.openroad_env.resolve()))}; set -u; "
        if args.openroad_env.exists()
        else ""
    )
    cmd = (
        "set -euo pipefail; "
        f"{source_env}{exports} "
        f"{shlex.quote(str(args.openroad_bin.resolve()))} -no_init -exit {shlex.quote(str(tcl_path.resolve()))}"
    )
    start = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=args.gnntimer_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {"returncode": proc.returncode, "wall_s": time.time() - start, **parse_openroad_report_power(log_path)}


def normalize_name(name: Any) -> str:
    if isinstance(name, bytes):
        return name.decode(errors="replace")
    return str(name)


def node_names_from(data: Any, gpdb: Any) -> list[str]:
    candidates: list[Any] = []
    if hasattr(data, "node_id2node_name"):
        candidates.append(getattr(data, "node_id2node_name"))
    if hasattr(gpdb, "node_id2node_name"):
        value = gpdb.node_id2node_name
        candidates.append(value() if callable(value) else value)
    if hasattr(gpdb, "node_names"):
        value = gpdb.node_names
        candidates.append(value() if callable(value) else value)
    for names in candidates:
        result = [normalize_name(name) for name in list(names)]
        if result:
            return result
    raise RuntimeError("node names are empty; cannot align power by instance name")


def summarize_xplace_power_groups(
    node_names: list[str],
    tensors: tuple[Any, ...],
    group_codes: Any,
    def_names: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, int]]:
    cpu_tensors = [tensor.detach().cpu().double() for tensor in tensors[:3]]
    codes_tensor = group_codes.detach().cpu()
    codes = [int(codes_tensor[i].item()) for i in range(int(codes_tensor.numel()))]
    name_to_node: dict[str, int] = {}
    for idx, name in enumerate(node_names):
        name_to_node.setdefault(name, idx)

    groups = zero_groups()
    stats = {"rows": 0, "matched": 0, "unmatched": 0}
    for name in def_names:
        stats["rows"] += 1
        node_id = name_to_node.get(name, -1)
        if node_id < 0 or node_id >= int(cpu_tensors[0].numel()):
            stats["unmatched"] += 1
            continue
        stats["matched"] += 1
        code = codes[node_id] if node_id < len(codes) else -1
        group = POWER_GROUPS[code] if 0 <= code < len(POWER_GROUPS) else "combinational"
        for component, tensor in zip(COMPONENTS[:3], cpu_tensors):
            groups[group][component] += float(tensor[node_id].item())

    power = {component: 0.0 for component in COMPONENTS}
    for group in POWER_GROUPS:
        groups[group]["total"] = groups[group]["internal"] + groups[group]["switching"] + groups[group]["leakage"]
        for component in COMPONENTS:
            power[component] += groups[group][component]
    return power, groups, stats


def run_xplace(args: argparse.Namespace, design: str) -> dict[str, Any]:
    xplace_dir = args.xplace_dir.resolve()
    sys.path.insert(0, str(xplace_dir))

    from run_timer import getArgs
    from src import Flute, GPUTimer, load_design
    from utils import set_random_seed, setup_logger
    import torch

    old_argv = sys.argv[:]
    start = time.time()
    stages: dict[str, float] = {}
    try:
        sys.argv = [
            "compare_sky130_power_groups",
            "--platformPath",
            str(args.platform_path.resolve()),
            "--designPath",
            str(args.design_path.resolve()),
            "--designName",
            design,
            "--load_from_raw",
            "True",
            "--gpu",
            str(args.gpu),
            "--verbose_cpp_log",
            "false",
        ]
        logging.getLogger().handlers.clear()
        Flute.register(8)
        rt_args = getArgs()
        logger = setup_logger(rt_args, sys.argv)
        set_random_seed(rt_args)

        t0 = time.time()
        data, rawdb, gpdb, params = load_design(rt_args, logger)
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        data = data.to(device).preprocess()
        gputimer = GPUTimer(data, rawdb, gpdb, params, rt_args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stages["load_construct"] = time.time() - t0

        if "spef" not in params or not Path(params["spef"]).exists():
            raise FileNotFoundError(f"SPEF file not found: {params.get('spef')}")

        t0 = time.time()
        gputimer.update_timing_dmp_spef()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stages["update_timing_dmp_spef"] = time.time() - t0

        t0 = time.time()
        tensors = gputimer.timer.report_power_total_cuda()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stages["report_power_total_cuda"] = time.time() - t0

        t0 = time.time()
        group_codes = gputimer.timer.report_power_group_codes()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stages["report_power_group_codes"] = time.time() - t0

        t0 = time.time()
        power, groups, stats = summarize_xplace_power_groups(
            node_names_from(data, gpdb),
            tensors,
            group_codes,
            iter_def_component_names(design_def_path(args.design_path, design)),
        )
        stages["summarize_groups"] = time.time() - t0
        return {
            "returncode": 0,
            "wall_s": time.time() - start,
            "stages": stages,
            "power": power,
            "power_groups": groups,
            "stats": stats,
        }
    finally:
        sys.argv = old_argv


def compare_case(design: str, opr: dict[str, Any], xpl: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    worst = ("", -1.0)
    for group in POWER_GROUPS:
        for component in COMPONENTS:
            o = float(opr["power_groups"][group][component])
            x = float(xpl["power_groups"][group][component])
            err = rel_err(x, o)
            row = {
                "design": design,
                "group": group,
                "component": component,
                "openroad": o,
                "xplace": x,
                "diff": x - o,
                "rel_err": err,
                "pass_1pct": err <= 0.01,
            }
            rows.append(row)
            if math.isfinite(err) and err > worst[1]:
                worst = (f"{group}.{component}", err)

    total_err = rel_err(float(xpl["power"]["total"]), float(opr["power"]["total"]))
    summary = {
        "design": design,
        "openroad_returncode": opr.get("returncode"),
        "xplace_returncode": xpl.get("returncode"),
        "openroad_wall_s": opr.get("wall_s"),
        "xplace_wall_s": xpl.get("wall_s"),
        "xplace_load_construct_s": xpl.get("stages", {}).get("load_construct"),
        "xplace_update_timing_dmp_spef_s": xpl.get("stages", {}).get("update_timing_dmp_spef"),
        "xplace_report_power_total_cuda_s": xpl.get("stages", {}).get("report_power_total_cuda"),
        "xplace_report_power_group_codes_s": xpl.get("stages", {}).get("report_power_group_codes"),
        "xplace_summarize_groups_s": xpl.get("stages", {}).get("summarize_groups"),
        "openroad_power_total": opr["power"]["total"],
        "xplace_power_total": xpl["power"]["total"],
        "power_total_rel_err": total_err,
        "worst_group_component": worst[0],
        "worst_group_component_rel_err": worst[1],
        "group_components_pass_1pct": all(row["pass_1pct"] for row in rows),
        "pass_1pct": total_err <= 0.01 and all(row["pass_1pct"] for row in rows),
        "xplace_def_rows": xpl.get("stats", {}).get("rows"),
        "xplace_def_matched": xpl.get("stats", {}).get("matched"),
        "xplace_def_unmatched": xpl.get("stats", {}).get("unmatched"),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out: Path, summaries: list[dict[str, Any]], component_rows: list[dict[str, Any]]) -> None:
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "group_components.csv", component_rows)
    with (out / "summary.json").open("w") as f:
        json.dump({"summary": summaries, "group_components": component_rows}, f, indent=2, sort_keys=True)
    with (out / "SUMMARY.md").open("w") as f:
        ok = sum(1 for row in summaries if row.get("pass_1pct"))
        f.write("# Sky130 Power Group Compare\n\n")
        f.write(f"Cases: {len(summaries)}, pass 1%: {ok}, fail: {len(summaries) - ok}\n\n")
        f.write("| design | pass | OR total | XP total | total err | worst group.component | worst err | matched/unmatched | OR wall | XP wall |\n")
        f.write("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|\n")
        for row in summaries:
            f.write(
                "| {design} | {pass_1pct} | {or_total} | {xp_total} | {total_err} | {worst} | {worst_err} | "
                "{matched}/{unmatched} | {or_wall} | {xp_wall} |\n".format(
                    design=row["design"],
                    pass_1pct=row["pass_1pct"],
                    or_total=fmt(row["openroad_power_total"]),
                    xp_total=fmt(row["xplace_power_total"]),
                    total_err=fmt(row["power_total_rel_err"]),
                    worst=row["worst_group_component"],
                    worst_err=fmt(row["worst_group_component_rel_err"]),
                    matched=row["xplace_def_matched"],
                    unmatched=row["xplace_def_unmatched"],
                    or_wall=fmt(row["openroad_wall_s"]),
                    xp_wall=fmt(row["xplace_wall_s"]),
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", action="append", help="Design name; repeat to run multiple designs.")
    parser.add_argument("--design-list", type=Path, default=DEFAULT_DESIGN_LIST)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("GPU", "0")))
    parser.add_argument("--xplace-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--design-path", type=Path, default=DEFAULT_DESIGN_PATH)
    parser.add_argument("--platform-path", type=Path, default=DEFAULT_PLATFORM_PATH)
    parser.add_argument("--gnntimer-dir", type=Path, default=DEFAULT_GNNTIMER_DIR)
    parser.add_argument("--openroad-bin", type=Path, default=DEFAULT_OPENROAD)
    parser.add_argument("--openroad-env", type=Path, default=DEFAULT_OPENROAD_ENV)
    parser.add_argument("--reuse-openroad", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tcl_path = args.out / "sky130_report_power.tcl"
    write_openroad_tcl(tcl_path)

    summaries: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for design in read_designs(args):
        log_path = args.out / "logs" / "openroad" / f"{design}.log"
        if args.reuse_openroad and log_path.exists():
            opr = {"returncode": 0, "wall_s": 0.0, **parse_openroad_report_power(log_path)}
        else:
            opr = run_openroad(args, design, tcl_path, log_path)
        if opr.get("returncode") != 0:
            raise RuntimeError(f"OpenROAD failed for {design}; see {log_path}")

        xpl = run_xplace(args, design)
        summary, rows = compare_case(design, opr, xpl)
        case_dir = args.out / "summaries"
        case_dir.mkdir(parents=True, exist_ok=True)
        with (case_dir / f"{design}.openroad.json").open("w") as f:
            json.dump(opr, f, indent=2, sort_keys=True)
        with (case_dir / f"{design}.xplace.json").open("w") as f:
            json.dump(xpl, f, indent=2, sort_keys=True)
        summaries.append(summary)
        component_rows.extend(rows)
        print(
            f"{design}: pass={summary['pass_1pct']} total_err={fmt(summary['power_total_rel_err'])} "
            f"worst={summary['worst_group_component']}:{fmt(summary['worst_group_component_rel_err'])} "
            f"matched={summary['xplace_def_matched']} unmatched={summary['xplace_def_unmatched']}",
            flush=True,
        )

    write_outputs(args.out, summaries, component_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
