#!/usr/bin/env python3
"""Diff OpenROAD full pin activity CSV against Xplace full pin activity CSV."""

from __future__ import annotations

import argparse
import csv
import heapq
import itertools
import json
import math
from pathlib import Path
from typing import Any


TOP_COUNTER = itertools.count()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openroad-pins", type=Path, required=True)
    parser.add_argument("--xplace-activity", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--path-tsv", type=Path)
    parser.add_argument("--density-abs-tol", type=float, default=1.0e-6)
    parser.add_argument("--density-rel-tol", type=float, default=1.0e-6)
    parser.add_argument("--duty-abs-tol", type=float, default=1.0e-6)
    parser.add_argument("--top-limit", type=int, default=100)
    return parser.parse_args()


def norm_name(name: str) -> str:
    text = (name or "").strip().strip('"')
    text = text.replace(r"\[", "[").replace(r"\]", "]")
    text = text.replace("\\", "")
    return text.replace(":", "/")


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out


def rel_err(actual: float, reference: float) -> float:
    if not math.isfinite(actual) or not math.isfinite(reference):
        return math.inf
    denom = abs(reference)
    if denom < 1.0e-30:
        return 0.0 if abs(actual - reference) < 1.0e-30 else math.inf
    return abs(actual - reference) / denom


def load_xplace_activity(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pin = norm_name(row.get("pin_name_slash") or row.get("pin_name") or "")
            if not pin:
                continue
            rows[pin] = {
                "pin_id": row.get("pin_id", ""),
                "pin_name": row.get("pin_name", ""),
                "density": to_float(row.get("activity_density")),
                "duty": to_float(row.get("activity_duty")),
                "origin": row.get("activity_origin", ""),
            }
    return rows


def read_path_pins(path: Path | None) -> tuple[list[dict[str, Any]], set[str]]:
    if path is None:
        return [], set()
    rows: list[dict[str, Any]] = []
    pins: set[str] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            for key in ("target_pin", "seed_pin", "from_pin", "to_pin"):
                pin = norm_name(row.get(key, ""))
                if pin:
                    pins.add(pin)
    return rows, pins


def push_top(heap: list[tuple[float, int, dict[str, Any]]], score: float, row: dict[str, Any], limit: int) -> None:
    if not math.isfinite(score):
        score = math.inf
    item = (score, next(TOP_COUNTER), row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def row_diff(or_density: float, x_density: float, or_duty: float, x_duty: float) -> tuple[float, float, float]:
    density_abs = abs(x_density - or_density)
    density_rel = rel_err(x_density, or_density)
    duty_abs = abs(x_duty - or_duty)
    return density_abs, density_rel, duty_abs


def is_mismatch(
    density_abs: float,
    density_rel_value: float,
    duty_abs: float,
    density_abs_tol: float,
    density_rel_tol: float,
    duty_abs_tol: float,
) -> bool:
    density_bad = density_abs > density_abs_tol and density_rel_value > density_rel_tol
    duty_bad = duty_abs > duty_abs_tol
    return density_bad or duty_bad


def build_path_sequences(path_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    for row in path_rows:
        path_id = row.get("path_id", "")
        if row.get("step") == "-1":
            paths.setdefault(path_id, []).append(norm_name(row.get("target_pin", "")))
            continue
        seq = paths.setdefault(path_id, [])
        from_pin = norm_name(row.get("from_pin", ""))
        to_pin = norm_name(row.get("to_pin", ""))
        if from_pin and (not seq or seq[-1] != from_pin):
            seq.append(from_pin)
        if to_pin:
            seq.append(to_pin)
    return paths


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    xrows = load_xplace_activity(args.xplace_activity)
    path_rows, path_pin_set = read_path_pins(args.path_tsv)
    path_or_rows: dict[str, dict[str, Any]] = {}

    diff_path = args.out_dir / "pin_activity_diff.csv"
    summary_path = args.out_dir / "pin_activity_diff_summary.json"
    report_path = args.out_dir / "pin_activity_diff_report.md"
    top_density: list[tuple[float, int, dict[str, Any]]] = []
    top_duty: list[tuple[float, int, dict[str, Any]]] = []
    counts: dict[str, int] = {
        "openroad_rows": 0,
        "matched": 0,
        "missing_xplace": 0,
        "mismatch_rows": 0,
        "density_mismatch_rows": 0,
        "duty_mismatch_rows": 0,
        "or_nonzero_x_zero": 0,
        "x_nonzero_or_zero": 0,
    }

    fields = [
        "pin_name",
        "or_inst_name",
        "or_cell_type",
        "or_port_name",
        "or_direction",
        "or_is_driver",
        "or_net_name",
        "or_density",
        "x_density",
        "density_abs_diff",
        "density_rel_err",
        "or_duty",
        "x_duty",
        "duty_abs_diff",
        "or_origin",
        "x_origin",
        "x_pin_id",
        "x_pin_name",
    ]

    with args.openroad_pins.open(newline="") as rf, diff_path.open("w", newline="") as wf:
        reader = csv.DictReader(rf)
        writer = csv.DictWriter(wf, fieldnames=fields)
        writer.writeheader()
        for orow in reader:
            counts["openroad_rows"] += 1
            pin = norm_name(orow.get("pin_name", ""))
            if pin in path_pin_set:
                path_or_rows[pin] = dict(orow)
            xrow = xrows.get(pin)
            if xrow is None:
                counts["missing_xplace"] += 1
                continue
            counts["matched"] += 1
            or_density = to_float(orow.get("activity_density"))
            or_duty = to_float(orow.get("activity_duty"))
            x_density = float(xrow["density"])
            x_duty = float(xrow["duty"])
            density_abs, density_rel_value, duty_abs = row_diff(or_density, x_density, or_duty, x_duty)
            if or_density != 0.0 and x_density == 0.0:
                counts["or_nonzero_x_zero"] += 1
            if x_density != 0.0 and or_density == 0.0:
                counts["x_nonzero_or_zero"] += 1
            density_bad = density_abs > args.density_abs_tol and density_rel_value > args.density_rel_tol
            duty_bad = duty_abs > args.duty_abs_tol
            if density_bad:
                counts["density_mismatch_rows"] += 1
            if duty_bad:
                counts["duty_mismatch_rows"] += 1
            out = {
                "pin_name": pin,
                "or_inst_name": orow.get("inst_name", ""),
                "or_cell_type": orow.get("cell_type", ""),
                "or_port_name": orow.get("port_name", ""),
                "or_direction": orow.get("direction", ""),
                "or_is_driver": orow.get("is_driver", ""),
                "or_net_name": orow.get("net_name", ""),
                "or_density": or_density,
                "x_density": x_density,
                "density_abs_diff": density_abs,
                "density_rel_err": density_rel_value,
                "or_duty": or_duty,
                "x_duty": x_duty,
                "duty_abs_diff": duty_abs,
                "or_origin": orow.get("activity_origin", ""),
                "x_origin": xrow.get("origin", ""),
                "x_pin_id": xrow.get("pin_id", ""),
                "x_pin_name": xrow.get("pin_name", ""),
            }
            if is_mismatch(
                density_abs,
                density_rel_value,
                duty_abs,
                args.density_abs_tol,
                args.density_rel_tol,
                args.duty_abs_tol,
            ):
                counts["mismatch_rows"] += 1
                writer.writerow(out)
            push_top(top_density, density_abs, out, args.top_limit)
            push_top(top_duty, duty_abs, out, args.top_limit)

    top_density_rows = [row for _, _, row in sorted(top_density, key=lambda item: item[0], reverse=True)]
    top_duty_rows = [row for _, _, row in sorted(top_duty, key=lambda item: item[0], reverse=True)]
    path_outputs: list[dict[str, Any]] = []
    first_path_divergence: dict[str, Any] | None = None
    if path_rows:
        path_activity_path = args.out_dir / "path_activity.tsv"
        with path_activity_path.open("w", newline="") as f:
            fieldnames = [
                "path_id",
                "seq",
                "pin_name",
                "or_density",
                "x_density",
                "density_abs_diff",
                "density_rel_err",
                "or_duty",
                "x_duty",
                "duty_abs_diff",
                "or_origin",
                "x_origin",
                "or_inst_name",
                "or_cell_type",
                "or_direction",
                "or_net_name",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for path_id, pins in build_path_sequences(path_rows).items():
                seen: set[str] = set()
                seq = 0
                for pin in pins:
                    if not pin or pin in seen:
                        continue
                    seen.add(pin)
                    orow = path_or_rows.get(pin, {})
                    xrow = xrows.get(pin, {})
                    or_density = to_float(orow.get("activity_density"))
                    or_duty = to_float(orow.get("activity_duty"))
                    x_density = to_float(xrow.get("density"))
                    x_duty = to_float(xrow.get("duty"))
                    density_abs, density_rel_value, duty_abs = row_diff(or_density, x_density, or_duty, x_duty)
                    out = {
                        "path_id": path_id,
                        "seq": seq,
                        "pin_name": pin,
                        "or_density": or_density,
                        "x_density": x_density,
                        "density_abs_diff": density_abs,
                        "density_rel_err": density_rel_value,
                        "or_duty": or_duty,
                        "x_duty": x_duty,
                        "duty_abs_diff": duty_abs,
                        "or_origin": orow.get("activity_origin", ""),
                        "x_origin": xrow.get("origin", ""),
                        "or_inst_name": orow.get("inst_name", ""),
                        "or_cell_type": orow.get("cell_type", ""),
                        "or_direction": orow.get("direction", ""),
                        "or_net_name": orow.get("net_name", ""),
                    }
                    writer.writerow(out)
                    path_outputs.append(out)
                    if first_path_divergence is None and is_mismatch(
                        density_abs,
                        density_rel_value,
                        duty_abs,
                        args.density_abs_tol,
                        args.density_rel_tol,
                        args.duty_abs_tol,
                    ):
                        first_path_divergence = dict(out)
                    seq += 1

    summary = {
        "openroad_pins": str(args.openroad_pins),
        "xplace_activity": str(args.xplace_activity),
        "diff_csv": str(diff_path),
        "path_tsv": str(args.path_tsv) if args.path_tsv else "",
        "counts": counts,
        "tolerances": {
            "density_abs_tol": args.density_abs_tol,
            "density_rel_tol": args.density_rel_tol,
            "duty_abs_tol": args.duty_abs_tol,
        },
        "top_density": top_density_rows[: args.top_limit],
        "top_duty": top_duty_rows[: args.top_limit],
        "first_path_divergence": first_path_divergence,
        "path_activity_tsv": str(args.out_dir / "path_activity.tsv") if path_rows else "",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Pin Activity Diff",
        "",
        f"- OpenROAD pins: `{args.openroad_pins}`",
        f"- Xplace activity: `{args.xplace_activity}`",
        f"- Diff CSV: `{diff_path}`",
        f"- OpenROAD rows: {counts['openroad_rows']}",
        f"- Matched rows: {counts['matched']}",
        f"- Missing Xplace rows: {counts['missing_xplace']}",
        f"- Mismatch rows: {counts['mismatch_rows']}",
        f"- OR nonzero / X zero: {counts['or_nonzero_x_zero']}",
        f"- X nonzero / OR zero: {counts['x_nonzero_or_zero']}",
        "",
        "## First Path Divergence",
    ]
    if first_path_divergence:
        lines.extend(
            [
                f"- Pin: `{first_path_divergence['pin_name']}`",
                f"- OR density/duty: `{first_path_divergence['or_density']}` / `{first_path_divergence['or_duty']}`",
                f"- X density/duty: `{first_path_divergence['x_density']}` / `{first_path_divergence['x_duty']}`",
                f"- Cell/net: `{first_path_divergence['or_cell_type']}` / `{first_path_divergence['or_net_name']}`",
            ]
        )
    else:
        lines.append("- No path TSV was provided, or no path mismatch crossed tolerance.")
    lines.extend(["", "## Top Density Mismatches"])
    for row in top_density_rows[:10]:
        lines.append(
            f"- `{row['pin_name']}` OR `{row['or_density']}` X `{row['x_density']}` "
            f"abs `{row['density_abs_diff']}` duty OR/X `{row['or_duty']}`/`{row['x_duty']}`"
        )
    report_path.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
