#!/usr/bin/env python3
"""
Batch runner for run_timer.py with R² score extraction and logging.
Runs all designs in specified order and records R² scores.

IMPORTANT: Runs SEQUENTIALLY - only 1 run_timer.py process at a time.
Each design completes fully before the next one starts.
"""

import subprocess
import sys
import json
import time
import re
from datetime import datetime
from pathlib import Path

# Training dataset (14 designs)
TRAINING_DESIGNS = [
    "blabla",
    "usb_cdc_core",
    "BM64",
    "salsa20",
    "aes128",
    "wbqspiflash",
    "cic_decimator",
    "aes256",
    "des",
    "aes_cipher",
    "picorv32a",
    "zipdiv",
    "genericfir",
    "usb",
]

# Test dataset (7 designs)
TEST_DESIGNS = [
    "jpeg_encoder",
    "usbf_device",
    "aes192",
    "xtea",
    "spm",
    "y_huff",
    "synth_ram",
]

def extract_r2_from_output(output_text):
    """Extract R² score from run_timer.py output."""
    # Try multiple patterns - Overall line first
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

def run_design(design_name, verbose=False):
    """Run timer for a single design and extract R² score."""
    start_time = time.time()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing: {design_name:20s} ", end='', flush=True)

    try:
        result = subprocess.run(
            ["python", "run_timer.py", "--designName", design_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per design
        )

        elapsed = time.time() - start_time

        combined = result.stdout + result.stderr
        prop_time = extract_propagate_time(combined)
        prop_str = f"  prop={prop_time:.4f}s" if prop_time is not None else ""

        if result.returncode == 0:
            r2 = extract_r2_from_output(combined)

            if r2 is not None:
                print(f"✓ R²={r2:.5f}{prop_str}, time={elapsed:.2f}s")
                return {
                    "design": design_name,
                    "r2": r2,
                    "propagate_time": prop_time,
                    "time": elapsed,
                    "status": "success",
                }
            else:
                print(f"✗ No R² found{prop_str}, time={elapsed:.2f}s")
                return {
                    "design": design_name,
                    "r2": None,
                    "propagate_time": prop_time,
                    "time": elapsed,
                    "status": "failed_no_r2",
                    "output": combined if verbose else None,
                }
        else:
            elapsed = time.time() - start_time
            print(f"✗ Failed (exit code {result.returncode}){prop_str}, time={elapsed:.2f}s")
            return {
                "design": design_name,
                "r2": None,
                "propagate_time": prop_time,
                "time": elapsed,
                "status": "failed_runtime",
                "error": result.stderr if verbose else None,
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ Timeout after {elapsed:.2f}s")
        return {
            "design": design_name,
            "r2": None,
            "time": elapsed,
            "status": "timeout",
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Error: {e}")
        return {
            "design": design_name,
            "r2": None,
            "time": elapsed,
            "status": "error",
            "error_msg": str(e),
        }

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"./timer_r2_log/timer_r2_scores_{timestamp}.txt"
    json_file = f"./timer_r2_log/timer_r2_scores_{timestamp}.json"

    print("=" * 60)
    print("Timer R² Score Batch Runner")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    all_results = []
    training_results = []
    test_results = []

    # Run training dataset
    print("======= Training dataset =======")
    for design in TRAINING_DESIGNS:
        result = run_design(design)
        all_results.append(result)
        training_results.append(result)

    print()

    # Run test dataset
    print("======= Test dataset =======")
    for design in TEST_DESIGNS:
        result = run_design(design)
        all_results.append(result)
        test_results.append(result)

    print()

    # Compute statistics
    train_successful = sum(1 for r in training_results if r["status"] == "success")
    test_successful = sum(1 for r in test_results if r["status"] == "success")
    total_successful = train_successful + test_successful

    train_r2_values = [r["r2"] for r in training_results if r["r2"] is not None]
    test_r2_values = [r["r2"] for r in test_results if r["r2"] is not None]

    print("=" * 60)
    print("Summary:")
    print(f"  Training: {train_successful}/{len(TRAINING_DESIGNS)} completed")
    if train_r2_values:
        print(f"    R² mean: {sum(train_r2_values)/len(train_r2_values):.5f}")
        print(f"    R² min:  {min(train_r2_values):.5f}")
        print(f"    R² max:  {max(train_r2_values):.5f}")
    print(f"  Test: {test_successful}/{len(TEST_DESIGNS)} completed")
    if test_r2_values:
        print(f"    R² mean: {sum(test_r2_values)/len(test_r2_values):.5f}")
        print(f"    R² min:  {min(test_r2_values):.5f}")
        print(f"    R² max:  {max(test_r2_values):.5f}")
    print(f"  Total: {total_successful}/{len(all_results)} completed")
    print("=" * 60)

    # Save results to text file
    with open(results_file, 'w') as f:
        f.write("Timer R² Score Results\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("======= Training dataset =======\n")
        for result in training_results:
            if result["r2"] is not None:
                prop = result.get("propagate_time")
                prop_str = f", prop {prop:.5f}" if prop is not None else ""
                f.write(f"{result['design']:20s} r2 {result['r2']:.5f}, time {result['time']:.5f}{prop_str}\n")
            else:
                f.write(f"{result['design']:20s} {result['status']}\n")

        f.write("\n======= Test dataset =======\n")
        for result in test_results:
            if result["r2"] is not None:
                prop = result.get("propagate_time")
                prop_str = f", prop {prop:.5f}" if prop is not None else ""
                f.write(f"{result['design']:20s} r2 {result['r2']:.5f}, time {result['time']:.5f}{prop_str}\n")
            else:
                f.write(f"{result['design']:20s} {result['status']}\n")

    # Save results to JSON file
    with open(json_file, 'w') as f:
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "training_dataset": training_results,
            "test_dataset": test_results,
            "statistics": {
                "training": {
                    "total": len(TRAINING_DESIGNS),
                    "successful": train_successful,
                    "r2_mean": sum(train_r2_values)/len(train_r2_values) if train_r2_values else None,
                    "r2_min": min(train_r2_values) if train_r2_values else None,
                    "r2_max": max(train_r2_values) if train_r2_values else None,
                },
                "test": {
                    "total": len(TEST_DESIGNS),
                    "successful": test_successful,
                    "r2_mean": sum(test_r2_values)/len(test_r2_values) if test_r2_values else None,
                    "r2_min": min(test_r2_values) if test_r2_values else None,
                    "r2_max": max(test_r2_values) if test_r2_values else None,
                },
            }
        }
        json.dump(json_data, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {json_file}")

if __name__ == "__main__":
    main()
