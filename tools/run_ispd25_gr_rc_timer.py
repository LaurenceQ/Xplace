#!/usr/bin/env python3
"""Run Xplace/GPUTimer timing on an ISPD2025 design with a GR RC TSV."""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

import run_timer
from utils import setup_logger, set_random_seed
from src import Flute, GPUTimer, load_dataset


BENCH_ROOT = Path("/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks")


def ordered_lefs(nangate_root: Path):
    lefs = sorted(glob.glob(str(nangate_root / "lef" / "*.lef")))
    for idx, path in enumerate(lefs):
        if "tech" in Path(path).name:
            lefs.insert(0, lefs.pop(idx))
            break
    return lefs


def build_params(bench_root: Path, design_set: str, design_name: str):
    design_dir = bench_root / design_set / design_name
    nangate_root = bench_root / "NanGate45"
    return {
        "benchmark": f"ispd2025_{design_set}",
        "design_name": design_name,
        "lefs": ordered_lefs(nangate_root),
        "libs": sorted(glob.glob(str(nangate_root / "lib" / "*.lib"))),
        "def": str(design_dir / f"{design_name}.def"),
        "sdc": str(design_dir / f"{design_name}.sdc"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-set", choices=["visible", "blind"], required=True)
    parser.add_argument("--design-name", default="ariane")
    parser.add_argument("--bench-root", type=Path, default=BENCH_ROOT)
    parser.add_argument("--gr-rc", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=20)
    parser.add_argument("--result-dir", default="result/ispd25_xplace_gr_rc")
    args, unknown = parser.parse_known_args()
    if unknown:
        raise SystemExit(f"unknown arguments: {' '.join(unknown)}")

    saved_argv = sys.argv
    sys.argv = [
        saved_argv[0],
        "--designName",
        args.design_name,
        "--gpu",
        str(args.gpu),
        "--num_threads",
        str(args.num_threads),
        "--result_dir",
        args.result_dir,
        "--gr_rc",
        str(args.gr_rc),
        "--global_placement",
        "False",
        "--legalization",
        "False",
        "--detail_placement",
        "False",
    ]
    xargs = run_timer.getArgs()
    sys.argv = saved_argv
    xargs.exp_id = f"{args.design_set}_{args.design_name}_gr_rc"
    xargs.dataset = f"ispd2025_{args.design_set}"
    xargs.design_name = args.design_name

    logger = setup_logger(xargs, sys.argv)
    set_random_seed(xargs)
    Flute.register(8)

    params = build_params(args.bench_root, args.design_set, args.design_name)
    params["gr_rc"] = str(args.gr_rc)
    for key in ("def", "sdc", "gr_rc"):
        if not Path(params[key]).exists():
            raise FileNotFoundError(f"{key} not found: {params[key]}")

    data, rawdb, gpdb = load_dataset(xargs, logger, params)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    data = data.to(device).preprocess()
    gputimer = GPUTimer(data, rawdb, gpdb, params, xargs)

    gputimer.update_timing_dmp_gr(str(args.gr_rc))
    wns_early, tns_early, wns_late, tns_late = gputimer.report_timing_slack()
    print(
        "XPLACE_GR_RC_RESULT "
        + json.dumps(
            {
                "design_set": args.design_set,
                "design_name": args.design_name,
                "gr_rc": str(args.gr_rc),
                "wns_early": wns_early,
                "tns_early": tns_early,
                "wns_late": wns_late,
                "tns_late": tns_late,
            },
            sort_keys=True,
        )
    )
    gputimer.timer.report_K_path(5, 1, 1, True)


if __name__ == "__main__":
    main()
