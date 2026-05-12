#!/usr/bin/env python3
"""
Batch runner for asap7 designs with R² score extraction and logging.
"""

import subprocess
import sys
import json
import time
import re
from datetime import datetime
from pathlib import Path

TRAINING_DESIGNS = [
    "ac97_top",
    "aes",
    "aes_cipher_top",
    "aes_cipher_top_v2",
    "aes_cipher_top_v3",
    "des3",
    "jpeg_encoder",
    "mc_top",
    "pci_bridge32",
    "uart",
]

TEST_DESIGNS = [
    "des",
    "fpu",
    "gcd",
    "jpeg_encoder_v2",
    "tv80s"  
]

def extract_r2_from_output(output_text):
    patterns = [
        r"Overall\s*(?::\s*)?R²\s*=\s*(-?\d+\.\d+)",
        r"r2_overall['\"]?\s*:\s*(-?\d+\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

def extract_propagate_time(output_text):
    match = re.search(r"\[Timer\] propagate_infer_timing:\s*(\d+\.\d+)s", output_text)
    if match:
        return float(match.group(1))
    return None

def run_design(design_name):
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing: {design_name:25s} ", end='', flush=True)

    try:
        result = subprocess.run(
            ["python", "run_timer.py", "--designName", design_name,
             "--platformPath", "/data/ssd/qkduan25/Xplace/synthetic_data/Platform/ASAP7",
             "--designPath", "/data/ssd/qkduan25/Xplace/synthetic_data/data"],
            capture_output=True, text=True, timeout=600
        )
        elapsed = time.time() - start_time
        combined = result.stdout + result.stderr

        prop_time = extract_propagate_time(combined)
        prop_str = f"  prop={prop_time:.4f}s" if prop_time is not None else ""

        if result.returncode == 0:
            r2 = extract_r2_from_output(combined)
            if r2 is not None:
                print(f"R²={r2:.5f}{prop_str}  ({elapsed:.1f}s)")
                return {"design": design_name, "r2": r2, "propagate_time": prop_time, "time": elapsed, "status": "success"}
            else:
                print(f"no R² found{prop_str}  ({elapsed:.1f}s)")
                return {"design": design_name, "r2": None, "propagate_time": prop_time, "time": elapsed, "status": "failed_no_r2"}
        else:
            print(f"exit={result.returncode}{prop_str}  ({elapsed:.1f}s)")
            return {"design": design_name, "r2": None, "propagate_time": prop_time, "time": elapsed, "status": "failed_runtime"}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"timeout  ({elapsed:.1f}s)")
        return {"design": design_name, "r2": None, "time": elapsed, "status": "timeout"}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"error: {e}")
        return {"design": design_name, "r2": None, "time": elapsed, "status": "error"}

def summarize(label, results):
    r2s = [r["r2"] for r in results if r["r2"] is not None]
    ok  = sum(1 for r in results if r["status"] == "success")
    print(f"  {label}: {ok}/{len(results)} completed", end="")
    if r2s:
        print(f"  |  mean={sum(r2s)/len(r2s):.5f}  min={min(r2s):.5f}  max={max(r2s):.5f}", end="")
    print()

def main():
    Path("./timer_r2_log").mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    txt_file  = f"./timer_r2_log/asap7_r2_{ts}.txt"
    json_file = f"./timer_r2_log/asap7_r2_{ts}.json"

    print("=" * 60)
    print("asap7 Timer R² Batch Runner")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    train_results, test_results = [], []

    print("\n=== Train ===")
    for d in TRAINING_DESIGNS:
        train_results.append(run_design(d))

    print("\n=== Test ===")
    for d in TEST_DESIGNS:
        test_results.append(run_design(d))

    print("\n" + "=" * 60)
    summarize("Train", train_results)
    summarize("Test ", test_results)
    print("=" * 60)

    # Text output
    with open(txt_file, 'w') as f:
        f.write(f"asap7 Timer R² Results\n{datetime.now()}\n{'='*60}\n\n")
        f.write("=== Train ===\n")
        for r in train_results:
            if r["r2"] is not None:
                f.write(f"{r['design']:25s}  r2={r['r2']:.5f}  time={r['time']:.1f}s\n")
            else:
                f.write(f"{r['design']:25s}  {r['status']}\n")
        f.write("\n=== Test ===\n")
        for r in test_results:
            if r["r2"] is not None:
                f.write(f"{r['design']:25s}  r2={r['r2']:.5f}  time={r['time']:.1f}s\n")
            else:
                f.write(f"{r['design']:25s}  {r['status']}\n")

    # JSON output
    with open(json_file, 'w') as f:
        json.dump({"train": train_results, "test": test_results}, f, indent=2)

    print(f"\nSaved: {txt_file}")
    print(f"       {json_file}")

if __name__ == "__main__":
    main()
