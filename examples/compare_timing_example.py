#!/usr/bin/env python3
"""
Example: Using the compare_at_spef_vs_infer() function to validate ML timing predictions.

This script demonstrates how to compare SPEF-based (real parasitics) timing values
with ML-predicted timing values, and measure correlation via R² scores.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import logger
from src.core.timing_opt import GPUTimer

def example_compare_timing(gputimer, infer_file, result_dir=None):
    """
    Example usage of compare_at_spef_vs_infer() function.

    Args:
        gputimer: GPUTimer object from timing_opt.py
        infer_file: Path to .infer file for ML predictions
        result_dir: Optional directory to save comparison results JSON
    """
    logger.info("Starting timing comparison: SPEF vs ML Inference")

    # Prepare paths
    save_json = None
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        save_json = os.path.join(result_dir, "timing_comparison_results.json")

    # Run comparison
    comparison_results = gputimer.compare_at_spef_vs_infer(
        infer_file=infer_file,
        logger=logger,
        verbose=False,
        save_json=save_json
    )

    # Access and display detailed results
    logger.info("\n========== Detailed Timing Comparison Results ==========")
    logger.info(f"Overall R² Score: {comparison_results['r2_overall']:.4f}")

    corner_names = ['early-rise', 'early-fall', 'late-rise', 'late-fall']
    for idx, (corner_name, r2) in enumerate(zip(corner_names, comparison_results['r2_per_corner'])):
        logger.info(f"{corner_name:12s} R²: {r2:7.4f}")

    # Access numpy arrays for further analysis
    at_spef = comparison_results['at_spef_ns']     # [num_pins, 4]
    at_infer = comparison_results['at_infer_ns']   # [num_pins, 4]
    residuals = comparison_results['residuals_ns']  # Prediction error per pin

    logger.info(f"\nAT values shape: {at_spef.shape}")
    logger.info(f"Max prediction error (residual): {abs(residuals).max():.6e} ns")
    logger.info(f"Mean prediction error: {abs(residuals).mean():.6e} ns")
    logger.info("=========================================================\n")

    return comparison_results


def example_in_placement_flow(gputimer, design_name, infer_file, result_dir):
    """
    Example: Integration into main placement flow.

    Typical usage in main.py or evaluation script:
    """
    # After placement optimization completes...

    # 1. Evaluate with SPEF timing
    gputimer.update_timing_spef()
    wns_early, tns_early, wns_late, tns_late = gputimer.report_timing_slack()
    logger.info(f"SPEF Eval: WNS_late={wns_late:.3f} ns, TNS_late={tns_late:.3f} ns")

    # 2. Run ML inference and compare
    infer_file = f"./TimingPredict/infer_results/{design_name}.infer"

    if os.path.exists(infer_file):
        save_path = os.path.join(result_dir, f"{design_name}_at_comparison.json")

        comparison_results = gputimer.compare_at_spef_vs_infer(
            infer_file=infer_file,
            logger=logger,
            verbose=False,
            save_json=save_path
        )

        # Log key metrics
        r2_scores = comparison_results['r2_per_corner']
        r2_overall = comparison_results['r2_overall']
        logger.info(f"ML Inference R² (overall): {r2_overall:.4f}")
        logger.info(f"  Per-corner: {[f'{r2:.4f}' for r2 in r2_scores]}")

        # Determine model quality
        if r2_overall > 0.90:
            logger.info("✓ ML model has excellent prediction accuracy (R² > 0.90)")
        elif r2_overall > 0.80:
            logger.info("✓ ML model has good prediction accuracy (R² > 0.80)")
        else:
            logger.warning(f"⚠ ML model prediction accuracy needs improvement (R² = {r2_overall:.4f})")
    else:
        logger.warning(f"Inference file not found: {infer_file}")


# Usage example
if __name__ == "__main__":
    print("This script shows how to use the compare_at_spef_vs_infer() function.")
    print("\nExample 1: Basic comparison")
    print("  comparison_results = gputimer.compare_at_spef_vs_infer(")
    print("      infer_file='path/to/design.infer',")
    print("      logger=logger,")
    print("      save_json='comparison_results.json'")
    print("  )")
    print("\nExample 2: Access results")
    print("  r2_overall = comparison_results['r2_overall']")
    print("  at_spef_ns = comparison_results['at_spef_ns']  # [num_pins, 4]")
    print("  at_infer_ns = comparison_results['at_infer_ns']")
    print("  residuals_ns = comparison_results['residuals_ns']")
    print("\nExample 3: Per-corner analysis")
    print("  r2_per_corner = comparison_results['r2_per_corner']")
    print("  mae_per_corner = comparison_results['mae_per_corner_ns']")
    print("  corr_per_corner = comparison_results['pearson_corr_per_corner']")
