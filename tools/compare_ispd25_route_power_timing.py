#!/usr/bin/env python3
"""Compare ISPD2025 route-segment timing and power against OpenROAD."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import re
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
BENCH = Path("/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks")
PLATFORM = BENCH / "NanGate45"
OPENROAD = Path("/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad")
XPLACE_PY = Path("/home/qkduan25/.conda/envs/gnn/bin/python")
OUT = REPO / "result" / "ispd25_route_power_timing_latest"
OPENROAD_GOLDEN_CACHE = REPO / "result" / "ispd25_route_power_openroad_golden_cache"
OPENROAD_TCL = REPO / "tools" / "openroad_ispd25_route_power_timed.tcl"
DESIGNS = [
    "ariane",
    "bsg_chip",
    "NV_NVDLA_partition_c",
    "mempool_tile_wrap",
    "mempool_group",
    "mempool_cluster",
]
SPLITS = ["visible", "blind"]
COMPONENTS = ("internal", "switching", "leakage", "total")
POWER_GROUPS = ("sequential", "combinational", "clock", "macro", "pad")

STAGE_RE = re.compile(r"OR_STAGE\s+(\S+)\s+([-+0-9.eE]+)")
TIMING_RE = re.compile(r"^(tns|wns)\s+max\s+([-+0-9.eE]+)\s*$")
POWER_RE = re.compile(
    r"^\s*Total\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+100\.0%\s*$"
)
POWER_GROUP_RE = re.compile(
    r"^\s*(Sequential|Combinational|Clock|Macro|Pad)\s+"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
)
DEF_COMPONENT_RE = re.compile(r"^\s*-\s+(\S+)\s+(\S+)\b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--split", action="append", choices=SPLITS)
    parser.add_argument("--design", action="append")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--openroad-threads", type=int, default=16)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--timeout-min", type=float, default=0.0)
    parser.add_argument("--openroad-timeout-min", type=float, default=0.0)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--xplace-python", type=Path, default=XPLACE_PY)
    parser.add_argument("--openroad-bin", type=Path, default=OPENROAD)
    parser.add_argument("--skip-openroad", action="store_true")
    parser.add_argument("--skip-xplace", action="store_true")
    parser.add_argument("--reuse-openroad", action="store_true")
    parser.add_argument("--reuse-xplace", action="store_true")
    parser.add_argument("--openroad-ref-out", type=Path)
    parser.add_argument("--openroad-golden-cache", type=Path, default=OPENROAD_GOLDEN_CACHE)
    parser.add_argument("--xplace-ref-out", type=Path)
    parser.add_argument("--openroad-root-dump", type=Path)
    parser.add_argument("--xplace-root-dump", type=Path)
    parser.add_argument("--root-probe-pins", type=Path)
    parser.add_argument("--missing-fanout-skip", default="auto")
    parser.add_argument("--force-openroad-golden", action="store_true")
    parser.add_argument("--no-instance-power-csv", action="store_true")
    parser.add_argument("--strict-report-power-order", action="store_true")
    parser.add_argument("--worker", choices=["xplace"], default="")
    parser.add_argument("--worker-split", default="")
    parser.add_argument("--worker-design", default="")
    return parser.parse_args()


def segment_path(split: str, design: str) -> Path:
    return BENCH / "openroad_gr_segments_skip_fanout300" / split / f"{design}.route_segments"


def def_path(split: str, design: str) -> Path:
    return BENCH / split / design / f"{design}.def"


def const_logic_value(text: str) -> str | None:
    value = (text or "").strip().lower()
    if value in {"0", "1'b0", "1'd0", "1'h0"}:
        return "0"
    if value in {"1", "1'b1", "1'd1", "1'h1"}:
        return "1"
    match = re.fullmatch(r"\d+'[bdh]([01]+)", value)
    if match and set(match.group(1)) == {"0"}:
        return "0"
    if match and set(match.group(1)) == {"1"}:
        return "1"
    return None


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def write_xplace_const_port_file(split: str, design: str, out_dir: Path) -> Path | None:
    verilog_path = BENCH / split / design / f"{design}.v.gz"
    if not verilog_path.exists():
        return None
    from tools.power_alignment.trace_activity_divergence import infer_top, norm_name, parse_verilog

    modules = parse_verilog(verilog_path)
    top = infer_top(modules, design if design in modules else None)
    rows: list[dict[str, str]] = []

    def walk(module: Any, prefix: tuple[str, ...], const_nets: dict[str, str]) -> None:
        for inst in module.instances:
            child = modules.get(inst.cell_type)
            child_const_nets: dict[str, str] = {}
            for port, net in inst.conns.items():
                value = const_logic_value(net)
                if value is None:
                    value = const_nets.get(norm_name(net))
                if value is None:
                    continue
                if child:
                    child_const_nets[norm_name(port)] = value
                else:
                    rows.append(
                        {
                            "inst_name": "/".join((*prefix, inst.name)),
                            "port_name": norm_name(port),
                            "value": value,
                        }
                    )
            if child:
                walk(child, (*prefix, inst.name), child_const_nets)

    walk(top, tuple(), {})
    if not rows:
        return None
    const_dir = out_dir / "xplace_const_ports"
    const_dir.mkdir(parents=True, exist_ok=True)
    path = const_dir / f"{split}_{design}.const_ports.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["inst_name", "port_name", "value"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def root_debug_path(base: Path | None, split: str, design: str, leaf: str) -> Path | None:
    if base is None:
        return None
    if base.suffix:
        return base
    return base / f"{split}_{design}" / leaf


def design_dir(split: str, design: str) -> Path:
    return BENCH / split / design


def def_path(split: str, design: str) -> Path:
    return design_dir(split, design) / f"{design}.def"


def sdc_path(split: str, design: str) -> Path:
    return design_dir(split, design) / f"{design}.sdc"


def platform_libs() -> list[Path]:
    libdir = PLATFORM / "lib"
    names = [
        "NangateOpenCellLibrary_typical.lib",
        "fakeram45_256x16.lib",
        "fakeram45_256x32.lib",
        "fakeram45_256x64.lib",
        "fakeram45_32x32.lib",
        "fakeram45_128x256.lib",
        "fakeram45_128x116.lib",
        "fakeram45_128x32.lib",
        "fakeram45_256x48.lib",
        "fakeram45_512x64.lib",
        "fakeram45_64x256.lib",
        "fakeram45_64x62.lib",
        "fakeram45_64x64.lib",
        "fakeram45_64x124.lib",
    ]
    return [libdir / name for name in names]


def file_sig(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def openroad_input_manifest(args: argparse.Namespace, split: str, design: str) -> dict[str, Any]:
    paths = [def_path(split, design), sdc_path(split, design), segment_path(split, design), OPENROAD_TCL]
    paths.extend(platform_libs())
    return {
        "split": split,
        "design": design,
        "openroad_bin": str(args.openroad_bin),
        "inputs": [file_sig(path) for path in paths],
    }


def golden_paths(out: Path, case_id: str) -> dict[str, Path]:
    root = out / "openroad_dump"
    return {
        "csv": root / f"{case_id}_power.csv",
        "pins_csv": root / f"{case_id}_power_pins.csv",
        "arcs_csv": root / f"{case_id}_power_internal_arcs.csv",
        "leakage_csv": root / f"{case_id}_power_leakage.csv",
        "manifest": root / f"{case_id}_manifest.json",
    }


def xplace_power_csv_path(out: Path, case_id: str) -> Path:
    return out / "xplace_dump" / f"{case_id}_power.csv"


def power_compare_csv_path(out: Path, case_id: str) -> Path:
    return out / "compare" / f"{case_id}_power_compare.csv"


def norm_inst_name(name: str) -> str:
    return name.replace(r"\[", "[").replace(r"\]", "]")


def golden_csvs_complete(paths: dict[str, Path]) -> bool:
    for key in ("csv", "pins_csv", "arcs_csv", "leakage_csv"):
        path = paths[key]
        if not path.exists() or path.stat().st_size == 0:
            return False
    return True


def golden_is_valid(args: argparse.Namespace, split: str, design: str, out: Path) -> bool:
    paths = golden_paths(out, f"{split}_{design}")
    if not golden_csvs_complete(paths) or not paths["manifest"].exists():
        return False
    try:
        old = load_json(paths["manifest"])
        new = openroad_input_manifest(args, split, design)
    except Exception:
        return False
    return old == new


def missing_fanout_skip_value(args: argparse.Namespace, split: str, design: str) -> int:
    value = str(args.missing_fanout_skip).strip().lower()
    if value != "auto":
        return max(int(value), 0)
    if (split, design) in {
        ("visible", "NV_NVDLA_partition_c"),
        ("blind", "ariane"),
        ("blind", "bsg_chip"),
        ("blind", "NV_NVDLA_partition_c"),
    }:
        return 0
    return 300


def rel_err(actual: float | None, reference: float | None) -> float | None:
    if actual is None or reference is None:
        return None
    if not math.isfinite(actual) or not math.isfinite(reference):
        return math.inf
    denom = abs(reference)
    if denom < 1e-20:
        return 0.0 if abs(actual - reference) < 1e-20 else math.inf
    return abs(actual - reference) / denom


def pass_rel(actual: float | None, reference: float | None, limit: float = 0.01) -> bool:
    err = rel_err(actual, reference)
    return err is not None and err <= limit


def timing_rel_err(actual: float | None, reference: float | None) -> float | None:
    if actual is None or reference is None:
        return None
    if not math.isfinite(actual) or not math.isfinite(reference):
        return math.inf
    if reference == 0.0 and actual >= 0.0:
        return 0.0
    return rel_err(actual, reference)


def pass_timing(actual: float | None, reference: float | None, limit: float = 0.01) -> bool:
    err = timing_rel_err(actual, reference)
    return err is not None and err <= limit


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
                    return int(line.split()[1])
    except Exception:
        return 0
    return 0


def query_gpu_mem_mib(pids: set[int]) -> int:
    if not pids:
        return 0
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
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
) -> dict[str, Any]:
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
            if key.startswith("DMP_") or key.startswith("GPUTIMER_") or key.startswith("XPLACE_POWER_"):
                log.write(f"# env {key}={env[key]}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
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


def parse_openroad_log(path: Path) -> dict[str, Any]:
    stages: dict[str, float] = {}
    timing = {"wns": None, "tns": None}
    power = {component: None for component in COMPONENTS}
    power_groups = {group: {component: None for component in COMPONENTS} for group in POWER_GROUPS}
    if not path.exists():
        return {"stages": stages, "timing": timing, "power": power}
    with path.open(errors="replace") as f:
        for line in f:
            stage = STAGE_RE.search(line)
            if stage:
                stages[stage.group(1)] = float(stage.group(2))
                continue
            metric = TIMING_RE.match(line.strip())
            if metric:
                timing[metric.group(1)] = float(metric.group(2))
                continue
            total = POWER_RE.match(line)
            if total:
                for component, value in zip(COMPONENTS, total.groups()):
                    power[component] = float(value)
                continue
            group = POWER_GROUP_RE.match(line)
            if group:
                group_name = group.group(1).lower()
                if group_name in power_groups:
                    for component, value in zip(COMPONENTS, group.groups()[1:]):
                        power_groups[group_name][component] = float(value)
    return {"stages": stages, "timing": timing, "power": power, "power_groups": power_groups}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def def_unescape(name: str) -> str:
    return name.replace(r"\[", "[").replace(r"\]", "]")


def read_def_cell_types(def_file: Path) -> dict[str, str]:
    cell_types: dict[str, str] = {}
    in_components = False
    with def_file.open(errors="replace") as f:
        for line in f:
            if not in_components:
                if line.lstrip().startswith("COMPONENTS "):
                    in_components = True
                continue
            if line.lstrip().startswith("END COMPONENTS"):
                break
            match = DEF_COMPONENT_RE.match(line)
            if match:
                cell_types[def_unescape(match.group(1))] = match.group(2)
    return cell_types


def read_power_csv_sums(path: Path) -> dict[str, float]:
    sums = {component: 0.0 for component in COMPONENTS}
    if not path.exists():
        return sums
    with path.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            for component in COMPONENTS:
                sums[component] += float(row.get(component) or 0.0)
    return sums


def read_power_csv_groups(openroad_csv: Path, xplace_csv: Path) -> dict[str, dict[str, float]]:
    groups = {group: {component: 0.0 for component in COMPONENTS} for group in POWER_GROUPS}
    if not openroad_csv.exists() or not xplace_csv.exists():
        return groups
    x_groups: dict[str, str] = {}
    with xplace_csv.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            group = row.get("power_group", "")
            if group in groups:
                x_groups[norm_inst_name(row.get("name", ""))] = group
    if not x_groups:
        return groups
    with openroad_csv.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            group = x_groups.get(norm_inst_name(row.get("name", "")))
            if group not in groups:
                continue
            for component in COMPONENTS:
                groups[group][component] += float(row.get(component) or 0.0)
    return groups


def read_power_csv_groups_by_codes(
    openroad_csv: Path,
    node_names: list[str],
    group_codes: Any,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    groups = {group: {component: 0.0 for component in COMPONENTS} for group in POWER_GROUPS}
    stats = {
        "rows": 0,
        "matched_by_inst_id": 0,
        "matched_by_name": 0,
        "order_mismatches": 0,
        "unmatched": 0,
    }
    if not openroad_csv.exists() or group_codes is None:
        return groups, stats

    if hasattr(group_codes, "detach"):
        codes = group_codes.detach().cpu().to(dtype=__import__("torch").int64).numpy()
    else:
        codes = group_codes
    num_nodes = min(len(node_names), len(codes))
    name_to_code: dict[str, int] | None = None

    def code_to_group(code: int) -> str | None:
        if 0 <= code < len(POWER_GROUPS):
            return POWER_GROUPS[code]
        return None

    def build_name_to_code() -> dict[str, int]:
        return {norm_inst_name(node_names[i]): int(codes[i]) for i in range(num_nodes)}

    with openroad_csv.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            stats["rows"] += 1
            group: str | None = None
            row_name = norm_inst_name(row.get("name", ""))
            inst_id_text = row.get("inst_id", "")
            if inst_id_text:
                try:
                    inst_id = int(inst_id_text)
                except ValueError:
                    inst_id = -1
                if 0 <= inst_id < num_nodes and row_name == norm_inst_name(node_names[inst_id]):
                    group = code_to_group(int(codes[inst_id]))
                    if group is not None:
                        stats["matched_by_inst_id"] += 1
                elif 0 <= inst_id < num_nodes:
                    stats["order_mismatches"] += 1
            if group is None:
                if name_to_code is None:
                    name_to_code = build_name_to_code()
                group = code_to_group(name_to_code.get(row_name, -1))
                if group is not None:
                    stats["matched_by_name"] += 1
            if group is None:
                stats["unmatched"] += 1
                continue
            for component in COMPONENTS:
                groups[group][component] += float(row.get(component) or 0.0)
    return groups, stats


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def power_result_total(values: dict[str, float], use_float32: bool = False) -> float:
    if use_float32:
        return f32(f32(values["internal"]) + f32(values["switching"]) + f32(values["leakage"]))
    return float(values["internal"] + values["switching"] + values["leakage"])


def iter_def_component_names(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as f:
        in_components = False
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
                yield match.group(1)


def iter_power_csv_names(path: Path) -> Any:
    with path.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            yield row.get("name", "")


def summarize_xplace_power_report_order(
    gpdb: Any,
    tensors: tuple[Any, ...],
    group_codes: Any,
    instance_names: Any,
    use_float32: bool = False,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, int]]:
    zero = f32(0.0) if use_float32 else 0.0
    power = {component: zero for component in COMPONENTS}
    groups = {group: {component: zero for component in COMPONENTS} for group in POWER_GROUPS}
    stats = {
        "rows": 0,
        "matched": 0,
        "unmatched": 0,
    }

    try:
        node_names = gpdb.node_id2node_name()
    except Exception:
        node_names = []
    name_to_node = {norm_inst_name(name): idx for idx, name in enumerate(node_names)}
    if hasattr(group_codes, "detach"):
        code_tensor = group_codes.detach().cpu().to(dtype=__import__("torch").int64)
        codes = [int(code_tensor[i].item()) for i in range(int(code_tensor.numel()))]
    else:
        codes = [int(code) for code in group_codes]
    cpu_tensors = [tensor.detach().cpu().float() if use_float32 else tensor.detach().cpu().double() for tensor in tensors]

    for name in instance_names:
        stats["rows"] += 1
        node_id = name_to_node.get(norm_inst_name(name), -1)
        if node_id < 0 or node_id >= int(cpu_tensors[0].numel()):
            stats["unmatched"] += 1
            continue
        code = codes[node_id] if node_id < len(codes) else -1
        group = POWER_GROUPS[code] if 0 <= code < len(POWER_GROUPS) else None
        stats["matched"] += 1
        for component, tensor in zip(COMPONENTS[:3], cpu_tensors[:3]):
            value = f32(float(tensor[node_id].item())) if use_float32 else float(tensor[node_id].item())
            power[component] = f32(power[component] + value) if use_float32 else power[component] + value
            if group is not None:
                groups[group][component] = (
                    f32(groups[group][component] + value) if use_float32 else groups[group][component] + value
                )

    power["total"] = power_result_total(power, use_float32)
    for group in POWER_GROUPS:
        groups[group]["total"] = power_result_total(groups[group], use_float32)
    return power, groups, stats


def power_groups_complete(power_groups: dict[str, dict[str, Any]]) -> bool:
    return all(
        power_groups.get(group, {}).get(component) is not None
        for group in POWER_GROUPS
        for component in COMPONENTS
    )


def power_groups_match_power(
    power_groups: dict[str, dict[str, Any]],
    power: dict[str, Any],
    tolerance: float = 1e-3,
) -> bool:
    if not power_groups_complete(power_groups):
        return False
    for component in COMPONENTS:
        expected = power.get(component)
        if expected is None:
            return True
        grouped = sum(float(power_groups.get(group, {}).get(component) or 0.0) for group in POWER_GROUPS)
        if rel_err(grouped, float(expected)) > tolerance:
            return False
    return True


def power_group_names(group_codes: Any | None, num_nodes: int, cell_types: list[str]) -> list[str]:
    if group_codes is None:
        return [xplace_power_group(cell_types[i]) for i in range(num_nodes)]
    import torch

    codes = group_codes.detach().cpu().to(dtype=torch.int64)
    names: list[str] = []
    for node_id in range(num_nodes):
        code = int(codes[node_id].item()) if node_id < int(codes.numel()) else -1
        if 0 <= code < len(POWER_GROUPS):
            names.append(POWER_GROUPS[code])
        else:
            names.append(xplace_power_group(cell_types[node_id]))
    return names


def write_xplace_power_csv(gpdb: Any, tensors: tuple[Any, ...], path: Path, group_codes: Any | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_types = gpdb.node_id2celltype_name()
    try:
        node_names = gpdb.node_id2node_name()
    except Exception:
        node_names = []
    num_nodes = min(len(cell_types), int(tensors[0].numel()) if tensors else 0)
    cpu_tensors = [tensor.detach().cpu().double()[:num_nodes] for tensor in tensors]
    groups = power_group_names(group_codes, num_nodes, cell_types)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "cell_type", "power_group", *COMPONENTS])
        writer.writeheader()
        for node_id in range(num_nodes):
            row = {
                "name": node_names[node_id] if node_id < len(node_names) else "",
                "cell_type": cell_types[node_id] if node_id < len(cell_types) else "",
                "power_group": groups[node_id],
            }
            for component, tensor in zip(COMPONENTS, cpu_tensors):
                row[component] = float(tensor[node_id].item())
            writer.writerow(row)


def write_xplace_internal_arc_probe_csv(
    gpdb: Any,
    tensors: tuple[Any, Any, Any],
    path: Path,
    inst_filter: set[str] | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        node_names = gpdb.node_id2node_name()
    except Exception:
        node_names = []
    try:
        cell_types = gpdb.node_id2celltype_name()
    except Exception:
        cell_types = []
    pin_names = gpdb.pin_names()
    _, row_power, row_meta = tensors
    power = row_power.detach().cpu().double()
    meta = row_meta.detach().cpu().to(dtype=__import__("torch").int64)
    rows = min(int(power.numel()), int(meta.shape[0]) if len(meta.shape) else 0)

    def pin_port(pin_id: int) -> tuple[str, str]:
        if 0 <= pin_id < len(pin_names):
            pin = pin_names[pin_id]
            if ":" in pin:
                return pin, pin.rsplit(":", 1)[1]
            if "/" in pin:
                return pin, pin.rsplit("/", 1)[1]
            return pin, ""
        return "", ""

    written = 0
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_id",
                "node_id",
                "inst_name",
                "cell_type",
                "kind",
                "to_pin_id",
                "to_pin",
                "to_port",
                "from_pin_id",
                "from_pin",
                "from_port",
                "internal_power_id",
                "duty_mode",
                "row_power",
            ],
        )
        writer.writeheader()
        for row_id in range(rows):
            node_id = int(meta[row_id, 0].item())
            inst_name = node_names[node_id] if 0 <= node_id < len(node_names) else ""
            if inst_filter and inst_name not in inst_filter:
                continue
            to_pin_id = int(meta[row_id, 1].item())
            from_pin_id = int(meta[row_id, 2].item())
            to_pin, to_port = pin_port(to_pin_id)
            from_pin, from_port = pin_port(from_pin_id)
            writer.writerow(
                {
                    "row_id": row_id,
                    "node_id": node_id,
                    "inst_name": inst_name,
                    "cell_type": cell_types[node_id] if 0 <= node_id < len(cell_types) else "",
                    "kind": "output" if int(meta[row_id, 3].item()) == 1 else "input",
                    "to_pin_id": to_pin_id,
                    "to_pin": to_pin,
                    "to_port": to_port,
                    "from_pin_id": from_pin_id,
                    "from_pin": from_pin,
                    "from_port": from_port,
                    "internal_power_id": int(meta[row_id, 4].item()),
                    "duty_mode": int(meta[row_id, 5].item()),
                    "row_power": float(power[row_id].item()),
                }
            )
            written += 1
    return {"path": str(path), "rows": written}


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"max": math.nan, "mean": math.nan, "p50": math.nan, "p90": math.nan, "p99": math.nan}
    vals = sorted(values)
    def q(frac: float) -> float:
        return vals[min(len(vals) - 1, int(round(frac * (len(vals) - 1))))]
    return {
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
    }


def compare_power_csvs(openroad_csv: Path, xplace_csv: Path, compare_csv: Path) -> dict[str, Any]:
    compare_csv.parent.mkdir(parents=True, exist_ok=True)
    x_by_name: dict[str, dict[str, str]] = {}
    with xplace_csv.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            x_by_name[norm_inst_name(row.get("name", ""))] = row

    rows: list[dict[str, Any]] = []
    missing = 0
    xplace_only = 0
    matched_x: set[str] = set()
    worst = {component: {"name": "", "abs_diff": -1.0} for component in COMPONENTS}
    sums = {
        "openroad": {component: 0.0 for component in COMPONENTS},
        "xplace": {component: 0.0 for component in COMPONENTS},
    }
    diffs = {component: [] for component in COMPONENTS}
    with openroad_csv.open(newline="", errors="replace") as f:
        for gt in csv.DictReader(f):
            name = norm_inst_name(gt.get("name", ""))
            x = x_by_name.get(name)
            if x is None:
                missing += 1
                x = {}
            else:
                matched_x.add(name)
            out: dict[str, Any] = {
                "name": name,
                "cell_type": gt.get("cell_type") or x.get("cell_type", ""),
            }
            for component in COMPONENTS:
                o = float(gt.get(component) or 0.0)
                xv = float(x.get(component) or 0.0)
                diff = xv - o
                adiff = abs(diff)
                sums["openroad"][component] += o
                sums["xplace"][component] += xv
                diffs[component].append(adiff)
                if adiff > worst[component]["abs_diff"]:
                    worst[component] = {"name": name, "abs_diff": adiff, "openroad": o, "xplace": xv}
                out[f"openroad_{component}"] = o
                out[f"xplace_{component}"] = xv
                out[f"{component}_diff"] = diff
                out[f"{component}_abs_diff"] = adiff
            rows.append(out)

    for name, x in x_by_name.items():
        if name in matched_x:
            continue
        xplace_only += 1
        out = {"name": name, "cell_type": x.get("cell_type", "")}
        for component in COMPONENTS:
            xv = float(x.get(component) or 0.0)
            adiff = abs(xv)
            sums["xplace"][component] += xv
            diffs[component].append(adiff)
            if adiff > worst[component]["abs_diff"]:
                worst[component] = {"name": name, "abs_diff": adiff, "openroad": 0.0, "xplace": xv}
            out[f"openroad_{component}"] = 0.0
            out[f"xplace_{component}"] = xv
            out[f"{component}_diff"] = xv
            out[f"{component}_abs_diff"] = adiff
        rows.append(out)

    with compare_csv.open("w", newline="") as f:
        fieldnames = ["name", "cell_type"]
        for component in COMPONENTS:
            fieldnames.extend([f"openroad_{component}", f"xplace_{component}", f"{component}_diff", f"{component}_abs_diff"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "openroad_csv": str(openroad_csv),
        "xplace_csv": str(xplace_csv),
        "compare_csv": str(compare_csv),
        "num_openroad_instances": len(rows) - xplace_only,
        "num_compare_rows": len(rows),
        "missing_openroad_names_in_xplace": missing,
        "xplace_only_names_not_in_openroad": xplace_only,
        "worst": worst,
    }
    for component in COMPONENTS:
        o = sums["openroad"][component]
        x = sums["xplace"][component]
        summary[component] = {
            "openroad_sum": o,
            "xplace_sum": x,
            "diff": x - o,
            "abs_diff": abs(x - o),
            "ratio": (x / o) if o else math.nan,
            "rel_err": rel_err(x, o),
            "abs_diff_stats": quantiles(diffs[component]),
        }
    return summary


def run_openroad(args: argparse.Namespace, split: str, design: str, log_path: Path, golden_out: Path) -> dict[str, Any]:
    case_id = f"{split}_{design}"
    paths = golden_paths(golden_out, case_id)
    env = os.environ.copy()
    env.update(
        {
            "DESIGN_SET": split,
            "DESIGN_NAME": design,
            "BENCH_ROOT": str(BENCH),
            "SEGMENT_IN": str(segment_path(split, design)),
        }
    )
    if not args.no_instance_power_csv:
        env["OR_DUMP_POWER_CSV"] = str(paths["csv"])
        env["OR_DUMP_POWER_PINS_CSV"] = str(paths["pins_csv"])
        env["OR_DUMP_POWER_ARCS_CSV"] = str(paths["arcs_csv"])
        env["OR_DUMP_POWER_LEAKAGE_CSV"] = str(paths["leakage_csv"])
    openroad_root_dump = root_debug_path(args.openroad_root_dump, split, design, "openroad_roots.tsv")
    if openroad_root_dump is not None:
        openroad_root_dump.parent.mkdir(parents=True, exist_ok=True)
        env["OR_POWER_DUMP_ROOTS_FILE"] = str(openroad_root_dump)
    if args.root_probe_pins is not None:
        env["OR_POWER_ROOT_PROBE_PINS_FILE"] = str(args.root_probe_pins)
    cmd = [
        str(args.openroad_bin),
        "-no_init",
        "-exit",
        "-threads",
        str(args.openroad_threads),
        str(OPENROAD_TCL),
    ]
    timeout = None if args.openroad_timeout_min <= 0 else args.openroad_timeout_min * 60.0
    return run_monitored(cmd, log_path, BENCH, env, timeout, args.sample_interval)


def run_xplace_parent(args: argparse.Namespace, split: str, design: str, log_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "GPUTIMER_ROUTE_SEG_MISSING_FANOUT_SKIP": str(missing_fanout_skip_value(args, split, design)),
            "GPUTIMER_DISABLE_REF_TIMING_TENSORS": "1",
            "GPUTIMER_DISABLE_STATE_BACKUP_TENSORS": "1",
            "GPUTIMER_EMPTY_CACHE_AFTER_GTDB": "1",
            "DMP_DEFER_TIMING_ALLOC": "1",
        }
    )
    cmd = [
        str(args.xplace_python),
        str(Path(__file__).resolve()),
        "--worker",
        "xplace",
        "--worker-split",
        split,
        "--worker-design",
        design,
        "--out",
        str(args.out),
        "--openroad-golden-cache",
        str(args.openroad_golden_cache),
        "--threads",
        str(args.threads),
        "--gpu",
        "0",
        "--missing-fanout-skip",
        str(args.missing_fanout_skip),
    ]
    if args.no_instance_power_csv:
        cmd.append("--no-instance-power-csv")
    if args.strict_report_power_order:
        cmd.append("--strict-report-power-order")
    xplace_root_dump = root_debug_path(args.xplace_root_dump, split, design, "xplace_roots.tsv")
    if xplace_root_dump is not None:
        xplace_root_dump.parent.mkdir(parents=True, exist_ok=True)
        env["XPLACE_POWER_DUMP_ROOTS_FILE"] = str(xplace_root_dump)
    if args.root_probe_pins is not None:
        env["XPLACE_POWER_ROOT_PROBE_PINS_FILE"] = str(args.root_probe_pins)
        env["XPLACE_POWER_PROBE_PIN_LIST_FILE"] = str(args.root_probe_pins)
    timeout = None if args.timeout_min <= 0 else args.timeout_min * 60.0
    return run_monitored(cmd, log_path, REPO, env, timeout, args.sample_interval)


def sync_cuda(torch_mod: Any) -> None:
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


def time_stage(stages: dict[str, float], name: str, fn: Any, torch_mod: Any | None = None) -> Any:
    if torch_mod is not None:
        sync_cuda(torch_mod)
    t0 = time.monotonic()
    out = fn()
    if torch_mod is not None:
        sync_cuda(torch_mod)
    stages[name] = time.monotonic() - t0
    print(f"XPLACE_STAGE {name} {stages[name]:.6f}", flush=True)
    return out


def tensor_sum(tensor: Any) -> float:
    return float(tensor.detach().double().sum().item())


def summarize_xplace_power_total(tensors: tuple[Any, ...]) -> dict[str, float]:
    power = {component: 0.0 for component in COMPONENTS}
    for component, tensor in zip(COMPONENTS[:3], tensors[:3]):
        power[component] = tensor_sum(tensor)
    power["total"] = power_result_total(power)
    return power


def xplace_power_group(cell_type: str) -> str:
    if not cell_type:
        return "pad"
    base = cell_type.rsplit("/", 1)[-1]
    if not cell_type.startswith("CORE/"):
        return "macro"
    if base.startswith(("DFF", "SDFF", "DLH", "DLL")):
        return "sequential"
    return "combinational"


def summarize_xplace_power_groups(gpdb: Any, tensors: tuple[Any, ...], group_codes: Any | None = None) -> dict[str, dict[str, float]]:
    import torch

    num_nodes = int(tensors[0].numel()) if tensors else 0
    group_to_code = {group: idx for idx, group in enumerate(POWER_GROUPS)}

    def fallback_codes(count: int) -> list[int]:
        cell_types = gpdb.node_id2celltype_name()
        return [
            group_to_code[xplace_power_group(cell_types[i] if i < len(cell_types) else "")]
            for i in range(count)
        ]

    if group_codes is None:
        code_tensor = torch.tensor(fallback_codes(num_nodes), dtype=torch.int64)
    else:
        code_tensor = group_codes.detach().to(dtype=torch.int64)[:num_nodes]
        if int(code_tensor.numel()) < num_nodes:
            fallback_tensor = torch.tensor(
                fallback_codes(num_nodes),
                dtype=torch.int64,
                device=code_tensor.device,
            )
            fallback_tensor[: int(code_tensor.numel())] = code_tensor
            code_tensor = fallback_tensor
    result = {group: {component: 0.0 for component in COMPONENTS} for group in POWER_GROUPS}
    if num_nodes <= 0:
        return result

    code_tensor = code_tensor.to(device=tensors[0].device)
    values = torch.stack([tensor.detach()[:num_nodes].double() for tensor in tensors[:3]], dim=1)
    sums = torch.zeros((len(POWER_GROUPS), values.size(1)), dtype=values.dtype, device=values.device)
    sums.index_add_(0, code_tensor, values)
    sums_cpu = sums.cpu()
    for group, code in group_to_code.items():
        for component_idx, component in enumerate(COMPONENTS[:3]):
            result[group][component] = float(sums_cpu[code, component_idx].item())
    for group in POWER_GROUPS:
        result[group]["total"] = power_result_total(result[group])
    return result


def summarize_xplace_power_types(gpdb: Any, tensors: tuple[Any, ...], limit: int = 40) -> list[dict[str, Any]]:
    cell_types = gpdb.node_id2celltype_name()
    num_nodes = min(len(cell_types), int(tensors[0].numel()) if tensors else 0)
    cpu_tensors = [tensor.detach().cpu().double()[:num_nodes] for tensor in tensors]
    by_type: dict[str, dict[str, float]] = {}
    for node_id in range(num_nodes):
        cell_type = cell_types[node_id] or ""
        row = by_type.setdefault(
            cell_type,
            {"count": 0.0, **{component: 0.0 for component in COMPONENTS}},
        )
        row["count"] += 1.0
        for component, tensor in zip(COMPONENTS, cpu_tensors):
            row[component] += float(tensor[node_id].item())
    rows = [{"cell_type": cell_type, **values} for cell_type, values in by_type.items()]
    rows.sort(key=lambda row: abs(row["total"]), reverse=True)
    for row in rows:
        row["count"] = int(row["count"])
    return rows[:limit]


def summarize_xplace_power_instances(gpdb: Any, tensors: tuple[Any, ...], limit: int = 80) -> list[dict[str, Any]]:
    import torch

    cell_types = gpdb.node_id2celltype_name()
    try:
        node_names = gpdb.node_id2node_name()
    except Exception:
        node_names = []
    num_nodes = min(len(cell_types), int(tensors[0].numel()) if tensors else 0)
    total = tensors[-1].detach().cpu().double()[:num_nodes]
    if num_nodes == 0:
        return []
    top_count = min(limit, num_nodes)
    values, indices = torch.topk(total.abs(), top_count)
    rows: list[dict[str, Any]] = []
    cpu_tensors = [tensor.detach().cpu().double()[:num_nodes] for tensor in tensors]
    for _, node_tensor in zip(values.tolist(), indices.tolist()):
        node_id = int(node_tensor)
        row = {
            "node_id": node_id,
            "inst": node_names[node_id] if node_id < len(node_names) else "",
            "cell_type": cell_types[node_id] if node_id < len(cell_types) else "",
        }
        for component, tensor in zip(COMPONENTS, cpu_tensors):
            row[component] = float(tensor[node_id].item())
        rows.append(row)
    return rows


def summarize_xplace_probe_types(gpdb: Any, tensors: tuple[Any, ...]) -> dict[str, list[dict[str, Any]]]:
    import torch

    type_env = os.environ.get("XPLACE_POWER_PROBE_TYPES", "").strip()
    if not type_env:
        return {}
    wanted = {item.strip() for item in type_env.split(",") if item.strip()}
    if not wanted:
        return {}
    limit = max(1, int(os.environ.get("XPLACE_POWER_PROBE_INSTANCES_PER_TYPE", "20")))
    cell_types = gpdb.node_id2celltype_name()
    try:
        node_names = gpdb.node_id2node_name()
    except Exception:
        node_names = []
    num_nodes = min(len(cell_types), int(tensors[0].numel()) if tensors else 0)
    cpu_tensors = [tensor.detach().cpu().double()[:num_nodes] for tensor in tensors]
    result: dict[str, list[dict[str, Any]]] = {}
    for cell_type in sorted(wanted):
        indices = [i for i in range(num_nodes) if cell_types[i] == cell_type or cell_types[i].endswith("/" + cell_type)]
        if not indices:
            result[cell_type] = []
            continue
        totals = cpu_tensors[-1][indices]
        top_count = min(limit, len(indices))
        _, order = torch.topk(totals.abs(), top_count)
        rows: list[dict[str, Any]] = []
        for pos in order.tolist():
            node_id = indices[int(pos)]
            row = {
                "node_id": node_id,
                "inst": node_names[node_id] if node_id < len(node_names) else "",
                "cell_type": cell_types[node_id] if node_id < len(cell_types) else "",
            }
            for component, tensor in zip(COMPONENTS, cpu_tensors):
                row[component] = float(tensor[node_id].item())
            rows.append(row)
        result[cell_type] = rows
    return result


def summarize_xplace_probe_instances(gpdb: Any, tensors: tuple[Any, ...]) -> list[dict[str, Any]]:
    list_path = os.environ.get("XPLACE_POWER_PROBE_INST_LIST_FILE", "").strip()
    if not list_path:
        return []
    path = Path(list_path)
    if not path.exists():
        return []
    wanted = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    if not wanted:
        return []
    cell_types = gpdb.node_id2celltype_name()
    try:
        node_names = gpdb.node_id2node_name()
    except Exception:
        node_names = []
    name_to_id = {name: idx for idx, name in enumerate(node_names)}
    num_nodes = min(len(cell_types), int(tensors[0].numel()) if tensors else 0)
    cpu_tensors = [tensor.detach().cpu().double()[:num_nodes] for tensor in tensors]
    rows: list[dict[str, Any]] = []
    for inst in wanted:
        node_id = name_to_id.get(inst, -1)
        row = {
            "node_id": node_id,
            "inst": inst,
            "cell_type": cell_types[node_id] if 0 <= node_id < len(cell_types) else "",
            "found": 0 <= node_id < num_nodes,
        }
        if row["found"]:
            for component, tensor in zip(COMPONENTS, cpu_tensors):
                row[component] = float(tensor[node_id].item())
        rows.append(row)
    return rows


def _probe_slew_fields(pin_slew: Any, pin_id: int, time_unit: float) -> dict[str, Any]:
    if pin_slew is None:
        return {}
    slew = pin_slew.detach().cpu()
    if pin_id < 0 or pin_id >= int(slew.shape[0]):
        return {}
    vals = [float(v) for v in slew[pin_id].double().tolist()]
    vals_s = [v * time_unit for v in vals]
    cap_density = None
    for base in range(0, min(len(vals_s), 4), 2):
        rise = vals_s[base]
        fall = vals_s[base + 1]
        if not math.isfinite(rise) or not math.isfinite(fall):
            continue
        avg = 0.5 * (rise + fall)
        if avg <= 0.0:
            continue
        cap = 1.0 / avg
        cap_density = cap if cap_density is None else min(cap_density, cap)
    return {
        "slew_attrs_internal": vals,
        "slew_attrs_s": vals_s,
        "slew_cap_density": cap_density,
    }


def summarize_xplace_probe_instance_pins(gpdb: Any, probe_rows: list[dict[str, Any]], activity: Any, pin_load: Any, pin_slew: Any = None, time_unit: float = 1.0) -> dict[str, list[dict[str, Any]]]:
    if not probe_rows:
        return {}
    pin_names = gpdb.pin_names()
    node2pin = gpdb.node2pin_info_tensor()
    node2pin_list = node2pin[1].detach().cpu().tolist()
    node2pin_end = node2pin[2].detach().cpu().tolist()
    act = activity.detach().cpu()
    load = pin_load.detach().cpu() if pin_load is not None else None
    result: dict[str, list[dict[str, Any]]] = {}
    for row in probe_rows:
        if not row.get("found"):
            continue
        node_id = int(row["node_id"])
        start = 0 if node_id == 0 else int(node2pin_end[node_id - 1])
        end = int(node2pin_end[node_id])
        pins: list[dict[str, Any]] = []
        for pin_id in node2pin_list[start:end]:
            pin_id = int(pin_id)
            item = {
                "pin_id": pin_id,
                "pin": pin_names[pin_id] if 0 <= pin_id < len(pin_names) else "",
                "density": float(act[pin_id, 0].item()),
                "duty": float(act[pin_id, 1].item()),
                "origin": int(float(act[pin_id, 2].item())),
            }
            if load is not None:
                vals = load[pin_id].double().tolist()
                item["load_attrs"] = [float(v) for v in vals]
            item.update(_probe_slew_fields(pin_slew, pin_id, time_unit))
            pins.append(item)
        result[row["inst"]] = pins
    return result


def summarize_xplace_probe_pins(gpdb: Any, activity: Any, pin_load: Any, pin_slew: Any = None, time_unit: float = 1.0) -> list[dict[str, Any]]:
    list_path = os.environ.get("XPLACE_POWER_PROBE_PIN_LIST_FILE", "").strip()
    if not list_path:
        return []
    path = Path(list_path)
    if not path.exists():
        return []
    wanted = [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip()]
    if not wanted:
        return []
    pin_names = gpdb.pin_names()
    name_to_id: dict[str, int] = {}
    for idx, name in enumerate(pin_names):
        name_to_id[name] = idx
        name_to_id[name.replace(":", "/")] = idx
    act = activity.detach().cpu()
    load = pin_load.detach().cpu() if pin_load is not None else None
    rows: list[dict[str, Any]] = []
    for query in wanted:
        pin_id = name_to_id.get(query, -1)
        row: dict[str, Any] = {
            "query": query,
            "pin_id": pin_id,
            "pin": pin_names[pin_id] if 0 <= pin_id < len(pin_names) else "",
            "found": 0 <= pin_id < len(pin_names),
        }
        if row["found"]:
            row["density"] = float(act[pin_id, 0].item())
            row["duty"] = float(act[pin_id, 1].item())
            row["origin"] = int(float(act[pin_id, 2].item()))
            if load is not None:
                row["load_attrs"] = [float(v) for v in load[pin_id].double().tolist()]
            row.update(_probe_slew_fields(pin_slew, pin_id, time_unit))
        rows.append(row)
    return rows


def write_probe_pin_artifacts(out_dir: Path, case_id: str, rows: list[dict[str, Any]], suffix: str = "power_probe_pins") -> dict[str, str]:
    if not rows:
        return {}
    probe_dir = out_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    json_path = probe_dir / f"{case_id}_{suffix}.json"
    csv_path = probe_dir / f"{case_id}_{suffix}.csv"
    write_json(json_path, {"pins": rows})

    fields: list[str] = []
    preferred = ["query", "found", "pin_id", "pin", "density", "duty", "origin", "load_attrs"]
    for key in preferred:
        if any(key in row for row in rows):
            fields.append(key)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(flat)
    return {f"{suffix}_json": str(json_path), f"{suffix}_csv": str(csv_path)}


def write_xplace_pin_activity_csv(gpdb: Any, activity: Any, csv_path: Path) -> dict[str, Any]:
    pin_names = gpdb.pin_names()
    act = activity.detach().cpu()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pin_id",
                "pin_name",
                "pin_name_slash",
                "inst_name",
                "port_name",
                "activity_density",
                "activity_duty",
                "activity_origin",
            ],
        )
        writer.writeheader()
        rows = min(len(pin_names), int(act.shape[0]))
        for pin_id in range(rows):
            pin_name = pin_names[pin_id]
            inst_name = pin_name
            port_name = ""
            if ":" in pin_name:
                inst_name, port_name = pin_name.rsplit(":", 1)
            writer.writerow(
                {
                    "pin_id": pin_id,
                    "pin_name": pin_name,
                    "pin_name_slash": pin_name.replace(":", "/"),
                    "inst_name": inst_name,
                    "port_name": port_name,
                    "activity_density": float(act[pin_id, 0].item()),
                    "activity_duty": float(act[pin_id, 1].item()),
                    "activity_origin": int(float(act[pin_id, 2].item())),
                }
            )
    return {"path": str(csv_path), "rows": rows}


def run_xplace_worker(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO))
    import torch
    from run_timer import getArgs
    from timer_only.flute import Flute
    from timer_only.logger import setup_logger
    from timer_only.read_platform import load_design
    from timer_only.timing_opt import GPUTimer
    from timer_only.tools import set_random_seed

    split = args.worker_split
    design = args.worker_design
    summary_path = args.out / "summaries" / f"{split}_{design}.xplace.json"
    stages: dict[str, float] = {}
    summary: dict[str, Any] = {
        "split": split,
        "design": design,
        "status": "error",
        "stages": stages,
        "timing": {"wns": None, "tns": None, "wns_early": None, "tns_early": None},
        "power": {component: None for component in COMPONENTS},
    }
    old_argv = sys.argv[:]
    try:
        Flute.register(8)
        sys.argv = [
            "compare_ispd25_route_power_timing",
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
            str(args.out / "xplace_run_timer_results"),
            "--exp_id",
            f"_ispd25_route_power_timing_{split}_{design}",
        ]
        timer_args = getArgs()
        logger = setup_logger(timer_args, sys.argv)
        set_random_seed(timer_args)
        data, rawdb, gpdb, params = time_stage(stages, "read_input", lambda: load_design(timer_args, logger))
        device = torch.device(f"cuda:{timer_args.gpu}" if torch.cuda.is_available() else "cpu")
        data = time_stage(stages, "preprocess_timing", lambda: data.to_timing_device(device).preprocess_timing(), torch)
        params["route_segments"] = timer_args.route_segments
        gputimer = time_stage(stages, "build_timing_graph", lambda: GPUTimer(data, rawdb, gpdb, params, timer_args), torch)
        data = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        def build_rc() -> None:
            gputimer.timer.update_states()
            gputimer.timer.init_dmp_rc_route_segments(timer_args.route_segments)

        time_stage(stages, "build_rc", build_rc, torch)

        def run_timer() -> tuple[float, float, float, float]:
            gputimer.timer.update_timing_dmp()
            return gputimer.report_timing_slack()

        wns_early, tns_early, wns_late, tns_late = time_stage(stages, "timer", run_timer, torch)
        keep_dmp_timing_scratch = os.environ.get("XPLACE_POWER_KEEP_DMP_TIMING_SCRATCH", "").strip() not in (
            "",
            "0",
            "false",
            "False",
            "no",
            "off",
        )
        if not keep_dmp_timing_scratch and hasattr(gputimer.timer, "release_dmp_timing_scratch_for_power"):
            time_stage(
                stages,
                "release_dmp_timing_scratch",
                lambda: gputimer.timer.release_dmp_timing_scratch_for_power(),
                torch,
            )
        auto_const_ports = os.environ.get("XPLACE_POWER_AUTO_CONST_PORT_FILE", "").strip() not in (
            "",
            "0",
            "false",
            "False",
            "no",
            "off",
        )
        if auto_const_ports and not os.environ.get("XPLACE_POWER_CONST_PORT_FILE", "").strip():
            const_port_file = write_xplace_const_port_file(split, design, args.out)
            if const_port_file is not None:
                os.environ["XPLACE_POWER_CONST_PORT_FILE"] = str(const_port_file)
        if os.environ.get("XPLACE_POWER_ACTIVITY_SNAPSHOT_CSV", "").strip():
            os.environ.setdefault("XPLACE_POWER_ACTIVITY_SNAPSHOT_SPLIT", split)
            os.environ.setdefault("XPLACE_POWER_ACTIVITY_SNAPSHOT_DESIGN", design)
        path_trace_requested = any(
            os.environ.get(name, "").strip()
            for name in (
                "XPLACE_POWER_TRACE_PATH_OUT",
                "XPLACE_POWER_TRACE_PATH_FILE",
                "XPLACE_POWER_ACTIVITY_PATH_TRACE_FILE",
                "XPLACE_POWER_ACTIVITY_SNAPSHOT_CSV",
            )
        )
        path_trace_only = os.environ.get("XPLACE_POWER_PATH_TRACE_ONLY", "").strip() not in (
            "",
            "0",
            "false",
            "False",
            "no",
        )
        pre_power_cpu_activity = None
        if path_trace_requested:
            pre_power_cpu_activity = time_stage(
                stages,
                "power_activity_cpu_pathtrace",
                lambda: gputimer.timer.report_power_activity_cpu(),
                torch,
            )
            if path_trace_only:
                summary.update(
                    {
                        "status": "ok",
                        "timing": {
                            "wns": float(wns_late),
                            "tns": float(tns_late),
                            "wns_early": float(wns_early),
                            "tns_early": float(tns_early),
                        },
                        "power": {component: None for component in COMPONENTS},
                        "trace_path": os.environ.get("XPLACE_POWER_TRACE_PATH_OUT", ""),
                        "xplace_activity_path_trace": os.environ.get("XPLACE_POWER_ACTIVITY_PATH_TRACE_FILE", ""),
                        "xplace_activity_snapshot_csv": os.environ.get("XPLACE_POWER_ACTIVITY_SNAPSHOT_CSV", ""),
                        "time_unit": float(gputimer.timer.time_unit()),
                    }
                )
                return 0
        activity_only = os.environ.get("XPLACE_POWER_ACTIVITY_ONLY", "").strip() not in (
            "",
            "0",
            "false",
            "False",
            "no",
        )
        if activity_only:
            power_probe_pins = []
            power_probe_pins_cpu = []
            power_probe_pin_paths = {}
            power_probe_pin_cpu_paths = {}
            full_pin_activity_paths: dict[str, Any] = {}
            power_probe_pin_list_file = os.environ.get("XPLACE_POWER_PROBE_PIN_LIST_FILE", "").strip()
            want_pin_activity = os.environ.get("XPLACE_POWER_PROBE_PIN_ACTIVITY", "").strip() not in ("", "0", "false", "False", "no")
            want_cpu_activity = os.environ.get("XPLACE_POWER_PROBE_CPU_ACTIVITY", "").strip() not in ("", "0", "false", "False", "no")
            full_pin_activity_csv = os.environ.get("XPLACE_POWER_FULL_PIN_ACTIVITY_CSV", "").strip()
            full_pin_activity_cpu_csv = os.environ.get("XPLACE_POWER_FULL_PIN_ACTIVITY_CPU_CSV", "").strip()

            cpu_activity = None
            if full_pin_activity_cpu_csv or (want_cpu_activity and power_probe_pin_list_file):
                cpu_activity = pre_power_cpu_activity
                if cpu_activity is None:
                    cpu_activity = time_stage(
                        stages,
                        "power_activity_cpu_only",
                        lambda: gputimer.timer.report_power_activity_cpu(),
                        torch,
                    )
                if full_pin_activity_cpu_csv:
                    full_pin_activity_paths["cpu"] = time_stage(
                        stages,
                        "write_power_activity_cpu_full_pin_csv",
                        lambda: write_xplace_pin_activity_csv(gpdb, cpu_activity, Path(full_pin_activity_cpu_csv)),
                        torch,
                    )

            cuda_activity = None
            if full_pin_activity_csv or (want_pin_activity and power_probe_pin_list_file):
                cuda_activity = time_stage(
                    stages,
                    "power_activity_cuda_only",
                    lambda: gputimer.timer.report_power_activity_cuda(),
                    torch,
                )
                if full_pin_activity_csv:
                    full_pin_activity_paths["cuda"] = time_stage(
                        stages,
                        "write_power_activity_full_pin_csv",
                        lambda: write_xplace_pin_activity_csv(gpdb, cuda_activity, Path(full_pin_activity_csv)),
                        torch,
                    )

            pin_load = None
            if want_pin_activity and power_probe_pin_list_file and cuda_activity is not None:
                pin_load = time_stage(stages, "pin_load_probe", lambda: gputimer.timer.report_pin_load(), torch)
                pin_slew = time_stage(stages, "pin_slew_probe", lambda: gputimer.timer.report_pin_slew(), torch)
                power_probe_pins = summarize_xplace_probe_pins(
                    gpdb, cuda_activity, pin_load, pin_slew, float(gputimer.timer.time_unit())
                )
                power_probe_pin_paths = write_probe_pin_artifacts(args.out, f"{split}_{design}", power_probe_pins)
            if want_cpu_activity and power_probe_pin_list_file and cpu_activity is not None:
                if pin_load is None:
                    pin_load = time_stage(stages, "pin_load_probe", lambda: gputimer.timer.report_pin_load(), torch)
                pin_slew = time_stage(stages, "pin_slew_cpu_probe", lambda: gputimer.timer.report_pin_slew(), torch)
                power_probe_pins_cpu = summarize_xplace_probe_pins(
                    gpdb, cpu_activity, pin_load, pin_slew, float(gputimer.timer.time_unit())
                )
                power_probe_pin_cpu_paths = write_probe_pin_artifacts(args.out, f"{split}_{design}", power_probe_pins_cpu, "power_probe_pins_cpu")

            summary.update(
                {
                    "status": "ok",
                    "timing": {
                        "wns": float(wns_late),
                        "tns": float(tns_late),
                        "wns_early": float(wns_early),
                        "tns_early": float(tns_early),
                    },
                    "power": {component: None for component in COMPONENTS},
                    "power_probe_pins": power_probe_pins,
                    "power_probe_pins_cpu": power_probe_pins_cpu,
                    "power_probe_pin_list_file": power_probe_pin_list_file,
                    "power_probe_pin_paths": power_probe_pin_paths,
                    "power_probe_pin_cpu_paths": power_probe_pin_cpu_paths,
                    "full_pin_activity_paths": full_pin_activity_paths,
                    "time_unit": float(gputimer.timer.time_unit()),
                }
            )
            return 0
        tensors = time_stage(stages, "power", lambda: gputimer.timer.report_power_total_cuda(), torch)
        power = time_stage(stages, "power_total_summary", lambda: summarize_xplace_power_total(tensors), torch)
        group_codes = time_stage(stages, "power_group_codes", lambda: gputimer.timer.report_power_group_codes(), torch)
        power_groups = time_stage(
            stages,
            "power_group_summary",
            lambda: summarize_xplace_power_groups(gpdb, tensors, group_codes),
            torch,
        )
        openroad_power_groups_by_xplace_group: dict[str, Any] = {}
        openroad_power_group_stats: dict[str, int] = {}
        openroad_csv = golden_paths(args.openroad_golden_cache, f"{split}_{design}")["csv"]
        if openroad_csv.exists():
            node_names = gpdb.node_id2node_name()
            openroad_power_groups_by_xplace_group, openroad_power_group_stats = time_stage(
                stages,
                "openroad_power_group_summary",
                lambda: read_power_csv_groups_by_codes(openroad_csv, node_names, group_codes),
                torch,
            )
        report_order_power: dict[str, float] = {}
        report_order_power_groups: dict[str, Any] = {}
        report_order_stats: dict[str, Any] = {}
        report_order_source = "torch_parallel_no_order"
        report_order_stats = {
            "source": report_order_source,
            "rows": int(tensors[0].numel()) if tensors else 0,
            "matched": int(tensors[0].numel()) if tensors else 0,
            "unmatched": 0,
        }
        design_def = def_path(split, design)
        strict_report_order = args.strict_report_power_order or env_bool(
            "XPLACE_POWER_STRICT_REPORT_ORDER",
            False,
        )
        if strict_report_order and design_def.exists():
            report_order_source = "def_components_double"
            report_order_power, report_order_power_groups, report_order_stats = time_stage(
                stages,
                "report_power_order_summary",
                lambda: summarize_xplace_power_report_order(gpdb, tensors, group_codes, iter_def_component_names(design_def)),
                torch,
            )
        elif strict_report_order and openroad_csv.exists():
            report_order_source = "openroad_csv_order_float32"
            report_order_power, report_order_power_groups, report_order_stats = time_stage(
                stages,
                "report_power_order_summary",
                lambda: summarize_xplace_power_report_order(
                    gpdb, tensors, group_codes, iter_power_csv_names(openroad_csv), use_float32=True
                ),
                torch,
            )
        if report_order_stats:
            report_order_stats["source"] = report_order_source
        xplace_power_csv = ""
        if not args.no_instance_power_csv:
            csv_path = xplace_power_csv_path(args.out, f"{split}_{design}")
            time_stage(stages, "write_power_csv", lambda: write_xplace_power_csv(gpdb, tensors, csv_path, group_codes), torch)
            xplace_power_csv = str(csv_path)
        power_type_top = summarize_xplace_power_types(gpdb, tensors)
        power_inst_top = summarize_xplace_power_instances(gpdb, tensors)
        power_probe_type_instances = summarize_xplace_probe_types(gpdb, tensors)
        power_probe_instances = summarize_xplace_probe_instances(gpdb, tensors)
        power_probe_instance_pins = {}
        power_probe_pins = []
        power_probe_pin_paths = {}
        power_probe_pins_cpu = []
        power_probe_pin_cpu_paths = {}
        full_pin_activity_paths: dict[str, Any] = {}
        internal_arc_probe: dict[str, Any] = {}
        power_probe_inst_list_file = os.environ.get("XPLACE_POWER_PROBE_INST_LIST_FILE", "").strip()
        power_probe_pin_list_file = os.environ.get("XPLACE_POWER_PROBE_PIN_LIST_FILE", "").strip()
        want_pin_activity = os.environ.get("XPLACE_POWER_PROBE_PIN_ACTIVITY", "").strip() not in ("", "0", "false", "False", "no")
        want_cpu_activity = os.environ.get("XPLACE_POWER_PROBE_CPU_ACTIVITY", "").strip() not in ("", "0", "false", "False", "no")
        want_probe_pin_list = bool(power_probe_pin_list_file)
        full_pin_activity_csv = os.environ.get("XPLACE_POWER_FULL_PIN_ACTIVITY_CSV", "").strip()
        full_pin_activity_cpu_csv = os.environ.get("XPLACE_POWER_FULL_PIN_ACTIVITY_CPU_CSV", "").strip()
        if full_pin_activity_cpu_csv:
            cpu_activity = pre_power_cpu_activity
            if cpu_activity is None:
                cpu_activity = time_stage(
                    stages,
                    "power_activity_cpu_full_pin",
                    lambda: gputimer.timer.report_power_activity_cpu(),
                    torch,
                )
            full_pin_activity_paths["cpu"] = time_stage(
                stages,
                "write_power_activity_cpu_full_pin_csv",
                lambda: write_xplace_pin_activity_csv(gpdb, cpu_activity, Path(full_pin_activity_cpu_csv)),
                torch,
            )
        if full_pin_activity_csv:
            activity = time_stage(
                stages,
                "power_activity_full_pin",
                lambda: gputimer.timer.report_power_activity_cuda(),
                torch,
            )
            full_pin_activity_paths["cuda"] = time_stage(
                stages,
                "write_power_activity_full_pin_csv",
                lambda: write_xplace_pin_activity_csv(gpdb, activity, Path(full_pin_activity_csv)),
                torch,
            )
        if want_pin_activity and (power_probe_instances or want_probe_pin_list):
            if want_cpu_activity and want_probe_pin_list:
                cpu_activity = pre_power_cpu_activity
                if cpu_activity is None:
                    cpu_activity = time_stage(stages, "power_activity_cpu_probe", lambda: gputimer.timer.report_power_activity_cpu(), torch)
                pin_load_cpu = time_stage(stages, "pin_load_cpu_probe", lambda: gputimer.timer.report_pin_load(), torch)
                pin_slew_cpu = time_stage(stages, "pin_slew_cpu_probe", lambda: gputimer.timer.report_pin_slew(), torch)
                power_probe_pins_cpu = summarize_xplace_probe_pins(
                    gpdb, cpu_activity, pin_load_cpu, pin_slew_cpu, float(gputimer.timer.time_unit())
                )
                power_probe_pin_cpu_paths = write_probe_pin_artifacts(args.out, f"{split}_{design}", power_probe_pins_cpu, "power_probe_pins_cpu")
            activity = time_stage(stages, "power_activity_probe", lambda: gputimer.timer.report_power_activity_cuda(), torch)
            pin_load = time_stage(stages, "pin_load_probe", lambda: gputimer.timer.report_pin_load(), torch)
            pin_slew = time_stage(stages, "pin_slew_probe", lambda: gputimer.timer.report_pin_slew(), torch)
            if power_probe_instances:
                power_probe_instance_pins = summarize_xplace_probe_instance_pins(
                    gpdb, power_probe_instances, activity, pin_load, pin_slew, float(gputimer.timer.time_unit())
                )
            if want_probe_pin_list:
                power_probe_pins = summarize_xplace_probe_pins(
                    gpdb, activity, pin_load, pin_slew, float(gputimer.timer.time_unit())
                )
                power_probe_pin_paths = write_probe_pin_artifacts(args.out, f"{split}_{design}", power_probe_pins)
        internal_arc_probe_csv = os.environ.get("XPLACE_POWER_INTERNAL_ARC_PROBE_CSV", "").strip()
        if internal_arc_probe_csv:
            inst_filter: set[str] | None = None
            inst_list_file = os.environ.get("XPLACE_POWER_INTERNAL_ARC_INST_LIST_FILE", "").strip() or power_probe_inst_list_file
            if inst_list_file:
                with Path(inst_list_file).open(errors="replace") as f:
                    inst_filter = {line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")}
            internal_arc_tensors = time_stage(
                stages,
                "power_internal_arc_probe",
                lambda: gputimer.timer.report_power_internal_arcs_cuda(),
                torch,
            )
            internal_arc_probe = time_stage(
                stages,
                "write_power_internal_arc_probe_csv",
                lambda: write_xplace_internal_arc_probe_csv(
                    gpdb,
                    internal_arc_tensors,
                    Path(internal_arc_probe_csv),
                    inst_filter,
                ),
                torch,
            )
        summary.update(
            {
                "status": "ok",
                "timing": {
                    "wns": float(wns_late),
                    "tns": float(tns_late),
                    "wns_early": float(wns_early),
                    "tns_early": float(tns_early),
                },
                "power": report_order_power or power,
                "power_groups": report_order_power_groups or power_groups,
                "power_double_sum": power,
                "power_groups_double_sum": power_groups,
                "power_tensor_device": str(tensors[0].device) if tensors else "",
                "power_tensor_is_cuda": bool(getattr(tensors[0], "is_cuda", False)) if tensors else False,
                "report_power_order_stats": report_order_stats,
                "openroad_power_groups_by_xplace_group": openroad_power_groups_by_xplace_group,
                "openroad_power_group_stats": openroad_power_group_stats,
                "power_sum_order": report_order_source or "xplace_double",
                "power_activity_engine": "cpu"
                if os.environ.get("XPLACE_POWER_USE_CPU_ACTIVITY_FOR_POWER", "").strip()
                not in ("", "0", "false", "False", "no", "off")
                else "cuda",
                "power_type_top": power_type_top,
                "power_inst_top": power_inst_top,
                "power_probe_type_instances": power_probe_type_instances,
                "power_probe_instances": power_probe_instances,
                "power_probe_instance_pins": power_probe_instance_pins,
                "power_probe_pins": power_probe_pins,
                "power_probe_pins_cpu": power_probe_pins_cpu,
                "power_probe_inst_list_file": power_probe_inst_list_file,
                "power_probe_pin_list_file": power_probe_pin_list_file,
                "power_probe_pin_paths": power_probe_pin_paths,
                "power_probe_pin_cpu_paths": power_probe_pin_cpu_paths,
                "full_pin_activity_paths": full_pin_activity_paths,
                "internal_arc_probe": internal_arc_probe,
                "power_csv": xplace_power_csv,
                "time_unit": float(gputimer.timer.time_unit()),
            }
        )
    except Exception as exc:  # noqa: BLE001 - per-case summary must survive.
        summary["error"] = repr(exc)
        print("XPLACE_ERROR", repr(exc), flush=True)
    finally:
        sys.argv = old_argv
        write_json(summary_path, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary.get("status") == "ok" else 1


def case_iter(args: argparse.Namespace) -> list[tuple[str, str]]:
    splits = args.split or SPLITS
    designs = args.design or DESIGNS
    return [(split, design) for split in splits for design in designs]


def flatten_row(split: str, design: str, opr: dict[str, Any], xpl: dict[str, Any], opr_rt: dict[str, Any], xpl_rt: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split,
        "design": design,
        "openroad_returncode": opr_rt.get("returncode"),
        "xplace_returncode": xpl_rt.get("returncode"),
        "openroad_wall_s": opr_rt.get("wall_s"),
        "xplace_wall_s": xpl_rt.get("wall_s"),
        "openroad_peak_rss_gib": (opr_rt.get("peak_hwm_kb", 0) or 0) / 1024.0 / 1024.0,
        "xplace_peak_rss_gib": (xpl_rt.get("peak_hwm_kb", 0) or 0) / 1024.0 / 1024.0,
        "xplace_peak_gpu_gib": (xpl_rt.get("peak_gpu_mib", 0) or 0) / 1024.0,
    }
    for stage in ("read_input", "read_route_segments", "build_rc", "timer", "power"):
        row[f"openroad_{stage}_s"] = opr.get("stages", {}).get(stage)
    for stage in (
        "read_input",
        "preprocess_timing",
        "build_timing_graph",
        "build_rc",
        "timer",
        "release_dmp_timing_scratch",
        "power",
        "power_total_summary",
        "power_group_codes",
        "power_group_summary",
        "openroad_power_group_summary",
        "report_power_order_summary",
        "write_power_csv",
    ):
        row[f"xplace_{stage}_s"] = xpl.get("stages", {}).get(stage)
    for metric in ("wns", "tns"):
        o = opr.get("timing", {}).get(metric)
        x = xpl.get("timing", {}).get(metric)
        row[f"openroad_{metric}"] = o
        row[f"xplace_{metric}"] = x
        row[f"{metric}_rel_err"] = timing_rel_err(x, o)
        row[f"{metric}_pass_1pct"] = pass_timing(x, o)
    for component in COMPONENTS:
        o = opr.get("power", {}).get(component)
        x = xpl.get("power", {}).get(component)
        row[f"openroad_power_{component}"] = o
        row[f"xplace_power_{component}"] = x
        row[f"power_{component}_rel_err"] = rel_err(x, o)
        row[f"power_{component}_pass_1pct"] = pass_rel(x, o)
    for group in POWER_GROUPS:
        for component in COMPONENTS:
            o = opr.get("power_groups", {}).get(group, {}).get(component)
            x = xpl.get("power_groups", {}).get(group, {}).get(component)
            row[f"openroad_power_{group}_{component}"] = o
            row[f"xplace_power_{group}_{component}"] = x
            row[f"power_{group}_{component}_rel_err"] = rel_err(x, o)
            row[f"power_{group}_{component}_pass_1pct"] = pass_rel(x, o)
    group_errs = [
        (row.get(f"power_{group}_{component}_rel_err"), group, component)
        for group in POWER_GROUPS
        for component in COMPONENTS
        if row.get(f"power_{group}_{component}_rel_err") is not None
    ]
    if group_errs:
        worst_group_err, worst_group, worst_component = max(
            group_errs,
            key=lambda item: item[0] if isinstance(item[0], float) and math.isfinite(item[0]) else math.inf,
        )
        row["worst_power_group_component"] = f"{worst_group}.{worst_component}"
        row["worst_power_group_component_rel_err"] = worst_group_err
    row["timing_pass_1pct"] = bool(row.get("wns_pass_1pct") and row.get("tns_pass_1pct"))
    row["power_total_components_pass_1pct"] = all(
        bool(row.get(f"power_{component}_pass_1pct")) for component in COMPONENTS
    )
    row["power_group_components_pass_1pct"] = all(
        bool(row.get(f"power_{group}_{component}_pass_1pct"))
        for group in POWER_GROUPS
        for component in COMPONENTS
    )
    row["power_pass_1pct"] = bool(
        row["power_total_components_pass_1pct"] and row["power_group_components_pass_1pct"]
    )
    row["pass_1pct"] = bool(row["timing_pass_1pct"] and row["power_pass_1pct"])
    row["openroad_power_csv"] = opr.get("power_csv")
    row["openroad_power_group_source"] = opr.get("power_group_source", "openroad_log_report_power")
    row["xplace_power_sum_order"] = xpl.get("power_sum_order")
    row["xplace_power_tensor_device"] = xpl.get("power_tensor_device")
    row["xplace_power_tensor_is_cuda"] = xpl.get("power_tensor_is_cuda")
    row["xplace_report_power_order_stats"] = xpl.get("report_power_order_stats")
    row["xplace_power_csv"] = xpl.get("power_csv")
    row["golden_reused"] = opr_rt.get("golden_reused")
    return row


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def write_outputs(out: Path, rows: list[dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", {"rows": rows})
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with (out / "SUMMARY.md").open("w") as f:
        ok = sum(1 for row in rows if row.get("pass_1pct"))
        f.write("# ISPD2025 Route Segment Timing/Power Compare\n\n")
        f.write(f"Cases: {len(rows)}, pass 1%: {ok}, fail: {len(rows) - ok}\n\n")
        f.write("| split | design | pass | power groups | golden reused | WNS err | TNS err | Pint err | Psw err | Pleak err | Ptotal err | worst group.component err | worst total inst | x read | x graph | x rc | x timer | x power | x group | x csv | or read | or rc | or timer | or power |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            render = {key: fmt(value) for key, value in row.items()}
            for key in (
                "golden_reused",
                "worst_total_inst",
                "power_compare_csv",
                "worst_power_group_component",
                "worst_power_group_component_rel_err",
            ):
                render.setdefault(key, "")
            f.write(
                "| {split} | {design} | {pass_1pct} | {power_group_components_pass_1pct} | {golden_reused} | {wns_rel_err} | {tns_rel_err} | "
                "{power_internal_rel_err} | {power_switching_rel_err} | {power_leakage_rel_err} | {power_total_rel_err} | "
                "{worst_power_group_component}:{worst_power_group_component_rel_err} | {worst_total_inst} | "
                "{xplace_read_input_s} | {xplace_build_timing_graph_s} | {xplace_build_rc_s} | {xplace_timer_s} | {xplace_power_s} | "
                "{xplace_power_group_summary_s} | {xplace_write_power_csv_s} | "
                "{openroad_read_input_s} | {openroad_build_rc_s} | {openroad_timer_s} | {openroad_power_s} |\n".format(
                    **render
                )
            )
        f.write("\n## Golden And Compare Files\n\n")
        for row in rows:
            f.write(
                f"- `{row.get('split')}/{row.get('design')}`: golden `{row.get('openroad_power_csv')}`, "
                f"Xplace `{row.get('xplace_power_csv')}`, compare `{row.get('power_compare_csv')}`\n"
            )


def main() -> int:
    args = parse_args()
    args.out = args.out.resolve()
    args.openroad_golden_cache = args.openroad_golden_cache.resolve()
    args.openroad_bin = args.openroad_bin.resolve()
    args.xplace_python = args.xplace_python.resolve()
    if args.openroad_ref_out:
        args.openroad_ref_out = args.openroad_ref_out.resolve()
    if args.xplace_ref_out:
        args.xplace_ref_out = args.xplace_ref_out.resolve()
    if args.worker == "xplace":
        return run_xplace_worker(args)

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split, design in case_iter(args):
        if not segment_path(split, design).exists():
            raise FileNotFoundError(segment_path(split, design))
        case_id = f"{split}_{design}"
        golden_out = args.openroad_golden_cache
        openroad_log = golden_out / "logs" / "openroad" / f"{case_id}.log"
        xplace_log = args.out / "logs" / "xplace" / f"{case_id}.log"
        openroad_meta = golden_out / "summaries" / f"{case_id}.openroad_runtime.json"
        xplace_meta = args.out / "summaries" / f"{case_id}.xplace_runtime.json"
        xplace_summary = args.out / "summaries" / f"{case_id}.xplace.json"
        openroad_read_out = args.openroad_ref_out if args.skip_openroad and args.openroad_ref_out else golden_out
        xplace_read_out = args.xplace_ref_out if args.skip_xplace and args.xplace_ref_out else args.out

        if not args.skip_openroad:
            gp = golden_paths(golden_out, case_id)
            if (not args.force_openroad_golden) and (not args.no_instance_power_csv) and golden_is_valid(args, split, design, golden_out):
                opr_rt = load_json(openroad_meta) if openroad_meta.exists() else {"returncode": 0}
                opr_rt["golden_reused"] = 1
                opr_rt["power_csv"] = str(gp["csv"])
            elif args.reuse_openroad and openroad_meta.exists() and openroad_log.exists() and (
                args.no_instance_power_csv or golden_csvs_complete(gp)
            ):
                opr_rt = load_json(openroad_meta)
                opr_rt["golden_reused"] = int(golden_csvs_complete(gp))
            else:
                opr_rt = run_openroad(args, split, design, openroad_log, golden_out)
                opr_rt["golden_reused"] = 0
                if opr_rt.get("returncode") == 0 and not args.no_instance_power_csv:
                    if not golden_csvs_complete(gp):
                        missing = [
                            str(gp[key])
                            for key in ("csv", "pins_csv", "arcs_csv", "leakage_csv")
                            if not gp[key].exists() or gp[key].stat().st_size == 0
                        ]
                        opr_rt["golden_error"] = f"missing OpenROAD power CSVs: {missing}"
                    else:
                        sums = read_power_csv_sums(gp["csv"])
                        write_json(gp["manifest"], openroad_input_manifest(args, split, design))
                        opr_rt["power_csv"] = str(gp["csv"])
                        opr_rt["power_csv_sums"] = sums
                write_json(openroad_meta, opr_rt)
        else:
            opr_rt = load_json(openroad_read_out / "summaries" / f"{case_id}.openroad_runtime.json")
        opr = parse_openroad_log(openroad_read_out / "logs" / "openroad" / f"{case_id}.log")
        gp_read = golden_paths(openroad_read_out, case_id)
        if gp_read["csv"].exists():
            opr["power_csv"] = str(gp_read["csv"])
            opr["power_csv_sums"] = read_power_csv_sums(gp_read["csv"])

        if not args.skip_xplace:
            if args.reuse_xplace and xplace_meta.exists() and xplace_summary.exists():
                xpl_rt = load_json(xplace_meta)
            else:
                xpl_rt = run_xplace_parent(args, split, design, xplace_log)
                write_json(xplace_meta, xpl_rt)
        else:
            xpl_rt = load_json(xplace_read_out / "summaries" / f"{case_id}.xplace_runtime.json")
        xpl = load_json(xplace_read_out / "summaries" / f"{case_id}.xplace.json")
        x_csv = xplace_power_csv_path(xplace_read_out, case_id)
        o_csv = gp_read["csv"]
        if (
            o_csv.exists()
            and x_csv.exists()
            and not power_groups_complete(opr.get("power_groups", {}))
        ):
            opr["power_groups"] = read_power_csv_groups(o_csv, x_csv)
            opr["power_group_source"] = "openroad_csv_by_xplace_csv_group"

        row = flatten_row(split, design, opr, xpl, opr_rt, xpl_rt)
        if o_csv.exists() and x_csv.exists():
            cmp_summary = compare_power_csvs(o_csv, x_csv, power_compare_csv_path(args.out, case_id))
            write_json(args.out / "compare" / f"{case_id}_power_compare_summary.json", cmp_summary)
            row["power_compare_csv"] = cmp_summary["compare_csv"]
            row["missing_openroad_names_in_xplace"] = cmp_summary["missing_openroad_names_in_xplace"]
            for component in COMPONENTS:
                row[f"csv_power_{component}_ratio"] = cmp_summary[component]["ratio"]
                row[f"csv_power_{component}_rel_err"] = cmp_summary[component]["rel_err"]
                row[f"worst_{component}_inst"] = cmp_summary["worst"][component]["name"]
                row[f"worst_{component}_abs_diff"] = cmp_summary["worst"][component]["abs_diff"]
        rows.append(row)
        write_outputs(args.out, rows)
        print(
            f"{case_id}: pass={row['pass_1pct']} "
            f"timing={row['timing_pass_1pct']} power={row['power_pass_1pct']} "
            f"groups={row['power_group_components_pass_1pct']} "
            f"wns_err={fmt(row['wns_rel_err'])} ptotal_err={fmt(row['power_total_rel_err'])} "
            f"worst_group={fmt(row.get('worst_power_group_component'))}:{fmt(row.get('worst_power_group_component_rel_err'))}",
            flush=True,
        )
    write_outputs(args.out, rows)
    return 0 if all(row.get("pass_1pct") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
