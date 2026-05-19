#!/usr/bin/env python3
"""Trace OR/X activity mismatches upstream using full pin activity CSVs.

This is a diagnostic-only tool. It does not run OpenROAD or Xplace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openroad-pins", type=Path, required=True)
    parser.add_argument("--xplace-activity", type=Path, required=True)
    parser.add_argument("--target-pin", action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=5000)
    parser.add_argument("--density-abs-tol", type=float, default=1.0e-6)
    parser.add_argument("--density-rel-tol", type=float, default=1.0e-6)
    parser.add_argument("--duty-abs-tol", type=float, default=1.0e-6)
    return parser.parse_args()


def norm_name(name: str) -> str:
    text = (name or "").strip().strip('"')
    text = text.replace(r"\[", "[").replace(r"\]", "]")
    text = text.replace("\\", "")
    return text.replace(":", "/")


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


def load_xplace(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pin = norm_name(row.get("pin_name_slash") or row.get("pin_name") or "")
            if not pin:
                continue
            rows[pin] = {
                "density": to_float(row.get("activity_density")),
                "duty": to_float(row.get("activity_duty")),
                "origin": row.get("activity_origin", ""),
                "pin_id": row.get("pin_id", ""),
                "pin_name": row.get("pin_name", ""),
            }
    return rows


def compact_or_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pin_name": norm_name(row.get("pin_name", "")),
        "inst_name": norm_name(row.get("inst_name", "")),
        "cell_type": row.get("cell_type", ""),
        "port_name": row.get("port_name", ""),
        "direction": row.get("direction", ""),
        "is_driver": row.get("is_driver", ""),
        "net_name": row.get("net_name", ""),
        "net_name_norm": norm_name(row.get("net_name", "")),
        "or_density": to_float(row.get("activity_density")),
        "or_duty": to_float(row.get("activity_duty")),
        "or_origin": row.get("activity_origin", ""),
    }


def load_openroad(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    by_pin: dict[str, dict[str, Any]] = {}
    drivers_by_net: dict[str, list[dict[str, Any]]] = {}
    inputs_by_inst: dict[str, list[dict[str, Any]]] = {}
    with path.open(newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = compact_or_row(raw)
            pin = row["pin_name"]
            if not pin:
                continue
            by_pin[pin] = row
            if row["is_driver"] == "1":
                drivers_by_net.setdefault(row["net_name"], []).append(row)
            if row["direction"] == "input":
                inputs_by_inst.setdefault(row["inst_name"], []).append(row)
    return by_pin, drivers_by_net, inputs_by_inst


def annotate(
    row: dict[str, Any],
    xrows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pin = row["pin_name"]
    x = xrows.get(pin, {})
    xd = to_float(x.get("density"))
    xdu = to_float(x.get("duty"))
    od = float(row["or_density"])
    odu = float(row["or_duty"])
    dabs = abs(xd - od) if math.isfinite(xd) and math.isfinite(od) else math.inf
    derr = rel_err(xd, od)
    duty_abs = abs(xdu - odu) if math.isfinite(xdu) and math.isfinite(odu) else math.inf
    density_bad = dabs > args.density_abs_tol and derr > args.density_rel_tol
    duty_bad = duty_abs > args.duty_abs_tol
    out = {
        **row,
        "x_density": xd,
        "x_duty": xdu,
        "x_origin": x.get("origin", ""),
        "x_pin_id": x.get("pin_id", ""),
        "density_abs_diff": dabs,
        "density_rel_err": derr,
        "duty_abs_diff": duty_abs,
        "mismatch": density_bad or duty_bad,
        "or_nonzero_x_zero": od != 0.0 and xd == 0.0,
        "x_missing": pin not in xrows,
    }
    return out


def upstream_sources(
    row: dict[str, Any],
    drivers_by_net: dict[str, list[dict[str, Any]]],
    inputs_by_inst: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    if row["direction"] == "input":
        return [("net_driver", src) for src in drivers_by_net.get(row["net_name"], [])]
    if row["is_driver"] == "1" or row["direction"] == "output":
        inputs = inputs_by_inst.get(row["inst_name"], [])
        return [("driver_input", src) for src in inputs]
    return []


def trace(args: argparse.Namespace) -> dict[str, Any]:
    xrows = load_xplace(args.xplace_activity)
    by_pin, drivers_by_net, inputs_by_inst = load_openroad(args.openroad_pins)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    first_same: list[dict[str, Any]] = []
    queue: deque[tuple[int, str, str]] = deque()
    seen: set[str] = set()

    for target in args.target_pin:
        pin = norm_name(target)
        if pin in by_pin:
            queue.append((0, pin, "target"))
        else:
            first_same.append({"target": pin, "reason": "target_not_in_openroad_pins"})

    while queue and len(nodes) < args.max_nodes:
        depth, pin, relation = queue.popleft()
        if depth > args.max_depth:
            break
        if pin in seen:
            continue
        seen.add(pin)
        row = by_pin.get(pin)
        if row is None:
            continue
        item = annotate(row, xrows, args)
        item["depth"] = depth
        item["relation"] = relation
        nodes.append(item)
        if not item["mismatch"]:
            first_same.append(item)
            continue
        for edge_kind, src in upstream_sources(row, drivers_by_net, inputs_by_inst):
            src_pin = src["pin_name"]
            src_item = annotate(src, xrows, args)
            edges.append(
                {
                    "from_downstream_pin": pin,
                    "to_source_pin": src_pin,
                    "edge_kind": edge_kind,
                    "source_mismatch": src_item["mismatch"],
                    "source_or_density": src_item["or_density"],
                    "source_x_density": src_item["x_density"],
                    "source_or_duty": src_item["or_duty"],
                    "source_x_duty": src_item["x_duty"],
                }
            )
            if src_item["mismatch"]:
                queue.append((depth + 1, src_pin, edge_kind))
            else:
                src_item["depth"] = depth + 1
                src_item["relation"] = edge_kind
                first_same.append(src_item)

    return {
        "openroad_pins": str(args.openroad_pins),
        "xplace_activity": str(args.xplace_activity),
        "targets": [norm_name(pin) for pin in args.target_pin],
        "nodes": nodes,
        "edges": edges,
        "first_same": first_same,
        "limits": {"max_depth": args.max_depth, "max_nodes": args.max_nodes},
    }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "source_trace.json"
    tsv_path = out_dir / "source_trace.tsv"
    md_path = out_dir / "source_trace.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    fields = [
        "depth",
        "relation",
        "pin_name",
        "inst_name",
        "cell_type",
        "port_name",
        "direction",
        "is_driver",
        "net_name_norm",
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
        "mismatch",
        "or_nonzero_x_zero",
    ]
    with tsv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["nodes"])

    lines = [
        "# Source Activity Trace",
        "",
        f"- Targets: `{', '.join(result['targets'])}`",
        f"- Trace nodes: {len(result['nodes'])}",
        f"- Trace edges: {len(result['edges'])}",
        "",
        "## First Same Boundary",
    ]
    if result["first_same"]:
        for row in result["first_same"][:20]:
            if "pin_name" not in row:
                lines.append(f"- `{row.get('target', '')}`: {row.get('reason', '')}")
                continue
            lines.append(
                f"- depth {row['depth']} `{row['pin_name']}` {row['relation']} "
                f"OR `{row['or_density']}`/`{row['or_duty']}` "
                f"X `{row['x_density']}`/`{row['x_duty']}` "
                f"cell `{row['cell_type']}` net `{row['net_name_norm']}`"
            )
    else:
        lines.append("- No same boundary found within limits.")
    lines.extend(["", "## Trace Nodes"])
    for row in result["nodes"][:120]:
        lines.append(
            f"- depth {row['depth']} `{row['pin_name']}` {row['relation']} "
            f"OR `{row['or_density']}`/`{row['or_duty']}` "
            f"X `{row['x_density']}`/`{row['x_duty']}` "
            f"mismatch `{row['mismatch']}`"
        )
    md_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    write_outputs(trace(args), args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
