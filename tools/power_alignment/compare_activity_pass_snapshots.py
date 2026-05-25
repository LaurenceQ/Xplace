#!/usr/bin/env python3
"""Compare OpenROAD and Xplace full-pin activity pass snapshots."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterator


TAG_ORDER = {
    "after_seed": 0,
    "after_comb": 1,
    "after_seq_seed": 2,
    "after_pass": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openroad", type=Path, required=True)
    parser.add_argument("--xplace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--density-rel-threshold", type=float, default=0.05)
    parser.add_argument("--density-abs-threshold", type=float, default=0.0)
    parser.add_argument("--duty-threshold", type=float, default=0.05)
    parser.add_argument(
        "--exact-epsilon",
        type=float,
        default=0.0,
        help="Tolerance for exact mismatch status; default records any parsed float difference.",
    )
    return parser.parse_args()


def norm_name(value: str) -> str:
    text = (value or "").strip().strip('"')
    text = text.replace(r"\[", "[").replace(r"\]", "]")
    text = text.replace("\\", "")
    return text.replace(":", "/")


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def rel_err(actual: float, reference: float) -> float:
    if not math.isfinite(actual) or not math.isfinite(reference):
        return math.inf
    denom = abs(reference)
    if denom < 1.0e-30:
        return 0.0 if abs(actual - reference) < 1.0e-30 else math.inf
    return abs(actual - reference) / denom


def snapshot_key(row: dict[str, str]) -> tuple[int, str]:
    return to_int(row.get("pass", "0")), row.get("tag", "")


def key_sort_value(key: tuple[int, str]) -> tuple[int, int, str]:
    return key[0], TAG_ORDER.get(key[1], 99), key[1]


def key_less(left: tuple[int, str], right: tuple[int, str]) -> bool:
    return key_sort_value(left) < key_sort_value(right)


def pin_key(row: dict[str, str]) -> str:
    return norm_name(row.get("pin_name_norm") or row.get("pin_name_slash") or row.get("pin_name", ""))


def grouped_rows(path: Path) -> Iterator[tuple[tuple[int, str], list[dict[str, str]]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        current_key: tuple[int, str] | None = None
        rows: list[dict[str, str]] = []
        for row in reader:
            key = snapshot_key(row)
            if current_key is None:
                current_key = key
            if key != current_key:
                yield current_key, rows
                current_key = key
                rows = []
            rows.append(row)
        if current_key is not None:
            yield current_key, rows


def next_group(
    iterator: Iterator[tuple[tuple[int, str], list[dict[str, str]]]]
) -> tuple[tuple[int, str] | None, list[dict[str, str]]]:
    try:
        return next(iterator)
    except StopIteration:
        return None, []


def row_value(row: dict[str, str] | None, *keys: str) -> str:
    if row is None:
        return ""
    for key in keys:
        value = row.get(key, "")
        if value != "":
            return value
    return ""


def is_exact_mismatch(density_abs: float, duty_abs: float, epsilon: float) -> bool:
    return density_abs > epsilon or duty_abs > epsilon


def is_debug_mismatch(
    density_abs: float,
    density_rel: float,
    duty_abs: float,
    density_abs_threshold: float,
    density_rel_threshold: float,
    duty_threshold: float,
) -> bool:
    density_bad = density_abs > density_abs_threshold and density_rel > density_rel_threshold
    duty_bad = duty_abs > duty_threshold
    return density_bad or duty_bad


def compare_group(
    key: tuple[int, str],
    or_rows: list[dict[str, str]],
    x_rows: list[dict[str, str]],
    compare_writer: csv.DictWriter,
    first_divergence: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    or_by_pin = {pin_key(row): row for row in or_rows if pin_key(row)}
    x_by_pin = {pin_key(row): row for row in x_rows if pin_key(row)}
    pins = sorted(set(or_by_pin) | set(x_by_pin))
    pass_id, tag = key
    summary: dict[str, Any] = {
        "pass": pass_id,
        "tag": tag,
        "openroad_rows": len(or_rows),
        "xplace_rows": len(x_rows),
        "matched_rows": 0,
        "missing_in_openroad": 0,
        "missing_in_xplace": 0,
        "exact_mismatch_count": 0,
        "debug_mismatch_count": 0,
        "max_density_abs": 0.0,
        "max_density_abs_pin": "",
        "max_density_rel": 0.0,
        "max_density_rel_pin": "",
        "max_duty_abs": 0.0,
        "max_duty_abs_pin": "",
    }
    for pin in pins:
        or_row = or_by_pin.get(pin)
        x_row = x_by_pin.get(pin)
        if or_row is None:
            status = "MISSING_IN_OPENROAD"
            debug_status = "MISSING"
            summary["missing_in_openroad"] += 1
            or_density = or_duty = math.nan
            x_density = to_float(x_row.get("density") if x_row else "")
            x_duty = to_float(x_row.get("duty") if x_row else "")
            density_abs = density_rel = duty_abs = math.inf
        elif x_row is None:
            status = "MISSING_IN_XPLACE"
            debug_status = "MISSING"
            summary["missing_in_xplace"] += 1
            or_density = to_float(or_row.get("density"))
            or_duty = to_float(or_row.get("duty"))
            x_density = x_duty = math.nan
            density_abs = density_rel = duty_abs = math.inf
        else:
            summary["matched_rows"] += 1
            or_density = to_float(or_row.get("density"))
            x_density = to_float(x_row.get("density"))
            or_duty = to_float(or_row.get("duty"))
            x_duty = to_float(x_row.get("duty"))
            density_abs = abs(x_density - or_density)
            density_rel = rel_err(x_density, or_density)
            duty_abs = abs(x_duty - or_duty)
            exact_bad = is_exact_mismatch(density_abs, duty_abs, args.exact_epsilon)
            debug_bad = is_debug_mismatch(
                density_abs,
                density_rel,
                duty_abs,
                args.density_abs_threshold,
                args.density_rel_threshold,
                args.duty_threshold,
            )
            status = "MISMATCH" if exact_bad else "MATCH"
            debug_status = "DEBUG_MISMATCH" if debug_bad else "DEBUG_MATCH"
            if exact_bad:
                summary["exact_mismatch_count"] += 1
            if debug_bad:
                summary["debug_mismatch_count"] += 1

        if math.isfinite(density_abs) and density_abs > summary["max_density_abs"]:
            summary["max_density_abs"] = density_abs
            summary["max_density_abs_pin"] = pin
        if math.isfinite(density_rel) and density_rel > summary["max_density_rel"]:
            summary["max_density_rel"] = density_rel
            summary["max_density_rel_pin"] = pin
        if math.isfinite(duty_abs) and duty_abs > summary["max_duty_abs"]:
            summary["max_duty_abs"] = duty_abs
            summary["max_duty_abs_pin"] = pin

        out_row = {
            "pass": pass_id,
            "tag": tag,
            "pin_name_norm": pin,
            "status": status,
            "debug_status": debug_status,
            "or_pin_name": row_value(or_row, "pin_name"),
            "x_pin_name": row_value(x_row, "pin_name"),
            "or_density": or_density,
            "x_density": x_density,
            "density_abs": density_abs,
            "density_rel": density_rel,
            "or_duty": or_duty,
            "x_duty": x_duty,
            "duty_abs": duty_abs,
            "or_origin": row_value(or_row, "origin"),
            "x_origin": row_value(x_row, "origin"),
            "or_inst_name": row_value(or_row, "inst_name"),
            "x_inst_name": row_value(x_row, "inst_name"),
            "or_port_name": row_value(or_row, "port_name"),
            "x_port_name": row_value(x_row, "port_name"),
            "or_is_driver": row_value(or_row, "is_driver"),
            "x_is_driver": row_value(x_row, "is_driver"),
            "x_is_load": row_value(x_row, "is_load"),
            "x_node_pending": row_value(x_row, "node_pending"),
            "x_cell_seq": row_value(x_row, "cell_seq"),
        }
        compare_writer.writerow(out_row)
        if status != "MATCH" and pin not in first_divergence:
            first_divergence[pin] = out_row.copy()
    return summary


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    compare_path = args.out_dir / "activity_snapshot_compare.csv"
    summary_path = args.out_dir / "activity_snapshot_summary.csv"
    first_path = args.out_dir / "activity_first_divergence.csv"

    compare_fields = [
        "pass",
        "tag",
        "pin_name_norm",
        "status",
        "debug_status",
        "or_pin_name",
        "x_pin_name",
        "or_density",
        "x_density",
        "density_abs",
        "density_rel",
        "or_duty",
        "x_duty",
        "duty_abs",
        "or_origin",
        "x_origin",
        "or_inst_name",
        "x_inst_name",
        "or_port_name",
        "x_port_name",
        "or_is_driver",
        "x_is_driver",
        "x_is_load",
        "x_node_pending",
        "x_cell_seq",
    ]
    summary_fields = [
        "pass",
        "tag",
        "openroad_rows",
        "xplace_rows",
        "matched_rows",
        "missing_in_openroad",
        "missing_in_xplace",
        "exact_mismatch_count",
        "debug_mismatch_count",
        "max_density_abs",
        "max_density_abs_pin",
        "max_density_rel",
        "max_density_rel_pin",
        "max_duty_abs",
        "max_duty_abs_pin",
    ]

    first_divergence: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    or_iter = grouped_rows(args.openroad)
    x_iter = grouped_rows(args.xplace)
    or_key, or_rows = next_group(or_iter)
    x_key, x_rows = next_group(x_iter)
    with compare_path.open("w", newline="") as compare_f:
        compare_writer = csv.DictWriter(compare_f, fieldnames=compare_fields)
        compare_writer.writeheader()
        while or_key is not None or x_key is not None:
            if or_key is not None and x_key is not None and or_key == x_key:
                key = or_key
                group_or_rows = or_rows
                group_x_rows = x_rows
                or_key, or_rows = next_group(or_iter)
                x_key, x_rows = next_group(x_iter)
            elif x_key is None or (or_key is not None and key_less(or_key, x_key)):
                key = or_key
                group_or_rows = or_rows
                group_x_rows = []
                or_key, or_rows = next_group(or_iter)
            else:
                key = x_key
                group_or_rows = []
                group_x_rows = x_rows
                x_key, x_rows = next_group(x_iter)
            if key is None:
                break
            summaries.append(
                compare_group(key, group_or_rows, group_x_rows, compare_writer, first_divergence, args)
            )

    with summary_path.open("w", newline="") as summary_f:
        writer = csv.DictWriter(summary_f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    with first_path.open("w", newline="") as first_f:
        writer = csv.DictWriter(first_f, fieldnames=compare_fields)
        writer.writeheader()
        for pin, row in sorted(first_divergence.items(), key=lambda item: key_sort_value((item[1]["pass"], item[1]["tag"])) + (item[0],)):
            writer.writerow(row)

    print(f"wrote {compare_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {first_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
