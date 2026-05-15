#!/usr/bin/env python3
"""Compare one Xplace CUDA power result against an OpenROAD my_dump_power CSV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DESIGN_PATH = Path("/research/d7/ascstd/qkduan25/TimingPredict/data/netlists")
DEFAULT_PLATFORM_PATH = REPO_ROOT / "sky130hd"
COMPONENTS = ("internal", "switching", "leakage", "total")


def finite(value: float) -> bool:
    return math.isfinite(value)


def safe_ratio(numer: float, denom: float) -> float:
    if denom == 0.0:
        return 1.0 if numer == 0.0 else float("inf")
    return numer / denom


def rel_err(numer: float, denom: float) -> float:
    if denom == 0.0:
        return 0.0 if numer == 0.0 else float("inf")
    return abs(numer - denom) / abs(denom)


def as_float(value: str) -> float:
    value = str(value).strip()
    return float(value) if value else float("nan")


def normalize_name(name: Any) -> str:
    if isinstance(name, bytes):
        return name.decode(errors="replace")
    return str(name)


def load_openroad_power(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [field for field in ("name", *COMPONENTS) if field not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(f"{path} missing required columns: {missing}")
        for row in reader:
            name = row["name"].strip().strip('"')
            rows[name] = {
                "inst_id": row.get("inst_id", ""),
                "cell_type": row.get("cell_type", ""),
                **{component: as_float(row[component]) for component in COMPONENTS},
            }
    if not rows:
        raise RuntimeError(f"empty OpenROAD power CSV: {path}")
    return rows


def tensor_values(tensor: Any) -> list[float]:
    return [float(v) for v in tensor.detach().cpu().double().tolist()]


def node_names_from(data: Any, gpdb: Any) -> list[str]:
    candidates = []
    if hasattr(data, "node_id2node_name"):
        candidates.append(getattr(data, "node_id2node_name"))
    if hasattr(gpdb, "node_id2node_name"):
        value = gpdb.node_id2node_name
        candidates.append(value() if callable(value) else value)
    if hasattr(gpdb, "node_names"):
        value = gpdb.node_names
        candidates.append(value() if callable(value) else value)
    for names in candidates:
        names = [normalize_name(name) for name in list(names)]
        if names:
            return names
    raise RuntimeError("node names are empty; cannot align power by instance name")


def compare_power(
    design: str,
    node_names: list[str],
    tensors: tuple[Any, Any, Any, Any],
    openroad: dict[str, dict[str, Any]],
    compare_csv: Path,
    top_n: int,
) -> dict[str, Any]:
    values = {component: tensor_values(tensor) for component, tensor in zip(COMPONENTS, tensors)}
    if any(len(vals) < len(node_names) for vals in values.values()):
        raise RuntimeError(
            f"power tensor shorter than node list: nodes={len(node_names)}, "
            + ", ".join(f"{key}={len(vals)}" for key, vals in values.items())
        )

    xplace: dict[str, dict[str, float]] = {}
    duplicates: dict[str, int] = {}
    for idx, name in enumerate(node_names):
        if name in xplace:
            duplicates[name] = duplicates.get(name, 1) + 1
            continue
        xplace[name] = {component: values[component][idx] for component in COMPONENTS}

    common = sorted(set(openroad) & set(xplace))
    missing_gt = sorted(set(openroad) - set(xplace))
    missing_x = sorted(set(xplace) - set(openroad))
    compare_csv.parent.mkdir(parents=True, exist_ok=True)

    worst_rows: dict[str, list[dict[str, Any]]] = {component: [] for component in COMPONENTS}
    with compare_csv.open("w", newline="") as f:
        fields = ["name", "inst_id", "cell_type"]
        for component in COMPONENTS:
            fields += [
                f"gt_{component}",
                f"cuda_{component}",
                f"diff_{component}",
                f"abs_diff_{component}",
            ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name in common:
            out: dict[str, Any] = {
                "name": name,
                "inst_id": openroad[name].get("inst_id", ""),
                "cell_type": openroad[name].get("cell_type", ""),
            }
            for component in COMPONENTS:
                gt_value = float(openroad[name][component])
                cuda_value = float(xplace[name][component])
                diff = cuda_value - gt_value
                out[f"gt_{component}"] = gt_value
                out[f"cuda_{component}"] = cuda_value
                out[f"diff_{component}"] = diff
                out[f"abs_diff_{component}"] = abs(diff) if finite(diff) else float("nan")
            writer.writerow(out)
            for component in COMPONENTS:
                value = out[f"abs_diff_{component}"]
                if finite(value):
                    worst_rows[component].append(dict(out))

    summary: dict[str, Any] = {
        "design": design,
        "status": "ok",
        "num_gt_instances": len(openroad),
        "num_x_nodes": len(xplace),
        "num_common_instances": len(common),
        "missing_gt_names_in_xplace": len(missing_gt),
        "missing_xplace_names_in_gt": len(missing_x),
        "missing_gt_examples": missing_gt[:20],
        "missing_xplace_examples": missing_x[:20],
        "duplicate_xplace_node_names": len(duplicates),
        "duplicate_xplace_node_name_examples": sorted(duplicates)[:20],
        "detail_csv": str(compare_csv),
    }
    for component in COMPONENTS:
        gt_sum = sum(float(row[component]) for row in openroad.values())
        cuda_sum = sum(float(xplace[name][component]) for name in common)
        diff = cuda_sum - gt_sum
        rows = sorted(
            worst_rows[component],
            key=lambda row: float(row[f"abs_diff_{component}"]),
            reverse=True,
        )[:top_n]
        summary[component] = {
            "gt_sum": gt_sum,
            "cuda_sum": cuda_sum,
            "diff": diff,
            "abs_diff": abs(diff) if finite(diff) else float("nan"),
            "ratio": safe_ratio(cuda_sum, gt_sum),
            "rel_err": rel_err(cuda_sum, gt_sum),
            "worst_instances": rows,
        }
    return summary


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    xplace_dir = args.xplace_dir.resolve()
    sys.path.insert(0, str(xplace_dir))

    from run_timer import getArgs
    from src import Flute, GPUTimer, load_design
    from utils import set_random_seed, setup_logger
    import torch

    os.environ.pop("XPLACE_POWER_ACTIVITY_FRONTIER", None)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "power_benchmark_compare_one",
            "--platformPath",
            str(args.platform_path.resolve()),
            "--designPath",
            str(args.design_path.resolve()),
            "--designName",
            args.design,
            "--load_from_raw",
            "True",
            "--gpu",
            str(args.gpu),
            "--verbose_cpp_log",
            "false",
        ]
        Flute.register(8)
        rt_args = getArgs()
        logger = setup_logger(rt_args, sys.argv)
        logger.setLevel(logging.INFO)
        set_random_seed(rt_args)

        load_t0 = time.time()
        data, rawdb, gpdb, params = load_design(rt_args, logger)
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        data = data.to(device).preprocess()
        gputimer = GPUTimer(data, rawdb, gpdb, params, rt_args)
        load_s = time.time() - load_t0

        if "spef" not in params or not Path(params["spef"]).exists():
            raise FileNotFoundError(f"SPEF file not found: {params.get('spef')}")

        dmp_t0 = time.time()
        gputimer.update_timing_dmp_spef()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dmp_s = time.time() - dmp_t0

        power_t0 = time.time()
        tensors = gputimer.timer.report_power_total_cuda()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        power_s = time.time() - power_t0

        openroad = load_openroad_power(args.gt_power.resolve())
        summary = compare_power(
            args.design,
            node_names_from(data, gpdb),
            tensors,
            openroad,
            out_dir / f"{args.design}_power_compare.csv",
            args.top_n,
        )
        summary["gt_power_csv"] = str(args.gt_power.resolve())
        summary["timing_s"] = {
            "load_construct": load_s,
            "update_timing_dmp_spef": dmp_s,
            "report_power_total_cuda": power_s,
            "total_script": time.time() - start,
        }
        return summary
    finally:
        sys.argv = old_argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--gt-power", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("GPU", "0")))
    parser.add_argument("--xplace-dir", type=Path, default=Path(os.environ.get("XPLACE_DIR", REPO_ROOT)))
    parser.add_argument(
        "--platform-path",
        type=Path,
        default=Path(os.environ.get("PLATFORM_PATH", str(DEFAULT_PLATFORM_PATH))),
    )
    parser.add_argument(
        "--design-path",
        type=Path,
        default=Path(os.environ.get("DESIGN_PATH", str(DEFAULT_DESIGN_PATH))),
    )
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_compare(args)
        summary_path = args.out_dir.resolve() / f"{args.design}_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("POWER_COMPARE_SUMMARY_JSON", json.dumps(summary, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - command-line tool records traceback.
        traceback.print_exc()
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "design": args.design,
            "status": "error",
            "error": repr(exc),
            "gt_power_csv": str(args.gt_power),
        }
        with (out_dir / f"{args.design}_summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
