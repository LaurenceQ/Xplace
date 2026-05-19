#!/usr/bin/env python3
"""Compare seeded pin activity and trace one upstream activity branch.

Diagnostic-only tool. It reads existing OpenROAD/Xplace TSV/CSV artifacts and
does not run either engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openroad-roots", type=Path, required=True)
    parser.add_argument("--xplace-roots", type=Path, required=True)
    parser.add_argument("--openroad-pins", type=Path, required=True)
    parser.add_argument("--xplace-activity", type=Path, required=True)
    parser.add_argument("--target-pin", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument(
        "--branch-mode",
        choices=("or_nonzero_x_zero", "max_density_diff"),
        default="or_nonzero_x_zero",
    )
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


def bool_field(value: Any) -> bool:
    return str(value).strip() in {"1", "true", "True", "yes"}


def load_root_sets(openroad_roots: Path, xplace_roots: Path) -> dict[str, Any]:
    or_seeded: set[str] = set()
    or_roots: set[str] = set()
    or_root_rows: dict[str, dict[str, Any]] = {}
    x_actual: set[str] = set()
    x_candidate: set[str] = set()

    with openroad_roots.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pin = norm_name(row.get("pin_name", ""))
            if not pin:
                continue
            or_root_rows[pin] = {
                "pin_name": pin,
                "inst_name": pin,
                "cell_type": row.get("row_type", "root"),
                "port_name": "",
                "direction": "root",
                "is_driver": "",
                "net_name": pin,
                "net_name_norm": pin,
                "or_density": to_float(row.get("activity_density")),
                "or_duty": to_float(row.get("activity_duty")),
                "or_origin": row.get("origin", ""),
            }
            if bool_field(row.get("in_levelize_roots")):
                or_roots.add(pin)
            if bool_field(row.get("was_seeded")):
                or_seeded.add(pin)

    with xplace_roots.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pin = norm_name(row.get("pin_name", ""))
            if not pin:
                continue
            if bool_field(row.get("in_actual_seed")):
                x_actual.add(pin)
            if bool_field(row.get("in_candidate")):
                x_candidate.add(pin)

    return {
        "or_roots": or_roots,
        "or_seeded": or_seeded,
        "x_actual": x_actual,
        "x_candidate": x_candidate,
        "common_seed": or_seeded & x_actual,
        "union_seed": or_seeded | x_actual,
        "or_root_rows": or_root_rows,
    }


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


def load_openroad(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
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
            }
    return rows


def annotate(
    row: dict[str, Any] | None,
    xrows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if row is None:
        return {
            "pin_name": "",
            "inst_name": "",
            "cell_type": "",
            "port_name": "",
            "direction": "",
            "is_driver": "",
            "net_name_norm": "",
            "or_density": math.nan,
            "or_duty": math.nan,
            "or_origin": "",
            "x_density": math.nan,
            "x_duty": math.nan,
            "x_origin": "",
            "x_pin_id": "",
            "density_abs_diff": math.inf,
            "density_rel_err": math.inf,
            "duty_abs_diff": math.inf,
            "mismatch": True,
            "or_nonzero_x_zero": False,
            "x_missing": True,
        }
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
    return {
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


def resolve_pin(pin_or_suffix: str, by_pin: dict[str, dict[str, Any]]) -> str:
    pin = norm_name(pin_or_suffix)
    if pin in by_pin:
        return pin
    matches = [name for name in by_pin if name.endswith(pin)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(f"target pin not found: {pin}")
    preview = "\n".join(matches[:20])
    raise RuntimeError(f"target pin suffix is ambiguous: {pin}\n{preview}")


def compare_seed_outputs(
    roots: dict[str, Any],
    xrows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_or = 0
    for pin in sorted(roots["common_seed"]):
        row = roots["or_root_rows"].get(pin)
        if row is None:
            missing_or += 1
            continue
        item = annotate(row, xrows, args)
        rows.append(item)
    mismatches = [row for row in rows if row["mismatch"]]
    top = sorted(
        mismatches,
        key=lambda row: (row["density_abs_diff"], row["duty_abs_diff"]),
        reverse=True,
    )[:50]
    return {
        "counts": {
            "or_roots": len(roots["or_roots"]),
            "or_seeded": len(roots["or_seeded"]),
            "x_actual_seed": len(roots["x_actual"]),
            "x_candidate": len(roots["x_candidate"]),
            "common_seed": len(roots["common_seed"]),
            "common_seed_compared": len(rows),
            "common_seed_missing_openroad_pin": missing_or,
            "common_seed_mismatch": len(mismatches),
            "or_seeded_not_x_actual": len(roots["or_seeded"] - roots["x_actual"]),
            "x_actual_not_or_seeded": len(roots["x_actual"] - roots["or_seeded"]),
        },
        "top_mismatches": top,
    }


def select_branch(inputs: list[dict[str, Any]], mode: str) -> tuple[dict[str, Any] | None, str]:
    if inputs and all(not item["mismatch"] for item in inputs):
        return None, "all_inputs_match"
    if mode == "max_density_diff":
        mismatches = [item for item in inputs if item["mismatch"]]
        if mismatches:
            return max(
                mismatches,
                key=lambda item: (item["density_abs_diff"], item["duty_abs_diff"]),
            ), "max_density_diff"
        return None, "all_inputs_match"

    zero_x = [
        item
        for item in inputs
        if math.isfinite(item["or_density"])
        and item["or_density"] != 0.0
        and item["x_density"] == 0.0
    ]
    if zero_x:
        return max(zero_x, key=lambda item: (item["or_density"], -item["duty_abs_diff"])), "or_max_nonzero_x_zero"
    return None, "no_or_nonzero_x_zero_input"


def trace_max_branch(
    target_pin: str,
    by_pin: dict[str, dict[str, Any]],
    drivers_by_net: dict[str, list[dict[str, Any]]],
    inputs_by_inst: dict[str, list[dict[str, Any]]],
    xrows: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    current_pin = resolve_pin(target_pin, by_pin)
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    stop_reason = "max_depth"
    terminal_gate: dict[str, Any] | None = None

    for depth in range(args.max_depth + 1):
        if current_pin in seen:
            stop_reason = "cycle"
            break
        seen.add(current_pin)
        current = annotate(by_pin.get(current_pin), xrows, args)
        if current["direction"] == "input":
            drivers = drivers_by_net.get(by_pin[current_pin]["net_name"], [])
            if not drivers:
                stop_reason = "input_has_no_net_driver"
                steps.append({"depth": depth, "current": current, "driver": None, "inputs": [], "selected": None})
                break
            driver_row = max(drivers, key=lambda row: float(row["or_density"]) if math.isfinite(float(row["or_density"])) else -math.inf)
        else:
            driver_row = by_pin[current_pin]

        driver = annotate(driver_row, xrows, args)
        inputs = [annotate(row, xrows, args) for row in inputs_by_inst.get(driver_row["inst_name"], [])]
        inputs.sort(key=lambda item: item["port_name"])
        all_inputs_match = bool(inputs) and all(not item["mismatch"] for item in inputs)
        selected, select_reason = select_branch(inputs, args.branch_mode)
        step = {
            "depth": depth,
            "current": current,
            "driver": driver,
            "inputs": inputs,
            "all_inputs_match": all_inputs_match,
            "driver_mismatch": driver["mismatch"],
            "selected": selected,
            "select_reason": select_reason,
        }
        steps.append(step)

        if all_inputs_match:
            stop_reason = "all_inputs_match"
            terminal_gate = step
            break
        if selected is None:
            stop_reason = select_reason
            terminal_gate = step
            break
        current_pin = selected["pin_name"]

    return {
            "target_pin": current_pin if not steps else steps[0]["current"]["pin_name"],
        "branch_mode": args.branch_mode,
        "stop_reason": stop_reason,
        "terminal_gate": terminal_gate,
        "steps": steps,
    }


def clean_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    if isinstance(value, dict):
        return {key: clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def write_seed_outputs(out_dir: Path, seed_result: dict[str, Any]) -> None:
    fields = [
        "pin_name",
        "inst_name",
        "cell_type",
        "port_name",
        "direction",
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
    ]
    with (out_dir / "seed_output_compare.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in seed_result["top_mismatches"]:
            writer.writerow(row)


def write_branch_trace(out_dir: Path, trace_result: dict[str, Any]) -> None:
    step_fields = [
        "depth",
        "current_pin",
        "driver_pin",
        "driver_cell",
        "driver_or_density",
        "driver_x_density",
        "driver_or_duty",
        "driver_x_duty",
        "all_inputs_match",
        "select_reason",
        "selected_pin",
        "selected_or_density",
        "selected_x_density",
        "selected_or_duty",
        "selected_x_duty",
    ]
    with (out_dir / "max_branch_trace.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_fields, delimiter="\t")
        writer.writeheader()
        for step in trace_result["steps"]:
            selected = step.get("selected") or {}
            writer.writerow(
                {
                    "depth": step["depth"],
                    "current_pin": step["current"]["pin_name"],
                    "driver_pin": step["driver"]["pin_name"] if step.get("driver") else "",
                    "driver_cell": step["driver"]["cell_type"] if step.get("driver") else "",
                    "driver_or_density": step["driver"]["or_density"] if step.get("driver") else "",
                    "driver_x_density": step["driver"]["x_density"] if step.get("driver") else "",
                    "driver_or_duty": step["driver"]["or_duty"] if step.get("driver") else "",
                    "driver_x_duty": step["driver"]["x_duty"] if step.get("driver") else "",
                    "all_inputs_match": step.get("all_inputs_match", ""),
                    "select_reason": step.get("select_reason", ""),
                    "selected_pin": selected.get("pin_name", ""),
                    "selected_or_density": selected.get("or_density", ""),
                    "selected_x_density": selected.get("x_density", ""),
                    "selected_or_duty": selected.get("or_duty", ""),
                    "selected_x_duty": selected.get("x_duty", ""),
                }
            )

    input_fields = [
        "depth",
        "driver_pin",
        "input_pin",
        "port",
        "or_density",
        "x_density",
        "or_duty",
        "x_duty",
        "mismatch",
        "or_nonzero_x_zero",
    ]
    with (out_dir / "max_branch_cell_inputs.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=input_fields, delimiter="\t")
        writer.writeheader()
        for step in trace_result["steps"]:
            driver_pin = step["driver"]["pin_name"] if step.get("driver") else ""
            for item in step.get("inputs", []):
                writer.writerow(
                    {
                        "depth": step["depth"],
                        "driver_pin": driver_pin,
                        "input_pin": item["pin_name"],
                        "port": item["port_name"],
                        "or_density": item["or_density"],
                        "x_density": item["x_density"],
                        "or_duty": item["or_duty"],
                        "x_duty": item["x_duty"],
                        "mismatch": item["mismatch"],
                        "or_nonzero_x_zero": item["or_nonzero_x_zero"],
                    }
                )


def fmt_activity(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    return f"OR {item.get('or_density')}/{item.get('or_duty')} X {item.get('x_density')}/{item.get('x_duty')}"


def write_report(out_dir: Path, seed_result: dict[str, Any], trace_result: dict[str, Any]) -> None:
    counts = seed_result["counts"]
    lines = [
        "# Seed Output And Max-Activity Branch Trace",
        "",
        "## Seed Output Compare",
        "",
        f"- OR roots: `{counts['or_roots']}`",
        f"- OR seeded: `{counts['or_seeded']}`",
        f"- X actual seed: `{counts['x_actual_seed']}`",
        f"- Common seeded pins: `{counts['common_seed']}`",
        f"- Common seeded pins compared in power pins: `{counts['common_seed_compared']}`",
        f"- Common seed activity mismatches: `{counts['common_seed_mismatch']}`",
        f"- OR seeded but X not actual: `{counts['or_seeded_not_x_actual']}`",
        f"- X actual but OR not seeded: `{counts['x_actual_not_or_seeded']}`",
        "",
        "### Top Common-Seed Mismatches",
        "",
        "| pin | OR density/duty | X density/duty |",
        "|---|---:|---:|",
    ]
    for row in seed_result["top_mismatches"][:20]:
        lines.append(f"| `{row['pin_name']}` | `{row['or_density']}` / `{row['or_duty']}` | `{row['x_density']}` / `{row['x_duty']}` |")

    lines.extend(
        [
            "",
            "## Upstream Branch Trace",
            "",
            f"- Start target: `{trace_result['target_pin']}`",
            f"- Branch mode: `{trace_result['branch_mode']}`",
            f"- Stop reason: `{trace_result['stop_reason']}`",
            "",
            "| depth | driver output | driver activity | chosen upstream input | chosen activity | reason |",
            "|---:|---|---:|---|---:|---|",
        ]
    )
    for step in trace_result["steps"]:
        driver = step.get("driver")
        selected = step.get("selected")
        lines.append(
            f"| {step['depth']} | `{driver['pin_name'] if driver else ''}` | `{fmt_activity(driver)}` | "
            f"`{selected['pin_name'] if selected else ''}` | `{fmt_activity(selected)}` | "
            f"`{step.get('select_reason', '')}` |"
        )

    terminal = trace_result.get("terminal_gate")
    if terminal:
        driver = terminal.get("driver")
        lines.extend(
            [
                "",
                "## Terminal Gate",
                "",
                f"- Driver output: `{driver['pin_name'] if driver else ''}`",
                f"- Cell: `{driver['cell_type'] if driver else ''}`",
                f"- Driver activity: `{fmt_activity(driver)}`",
                f"- All inputs match: `{terminal.get('all_inputs_match')}`",
                "",
                "| input | port | OR density/duty | X density/duty | mismatch |",
                "|---|---|---:|---:|---|",
            ]
        )
        for item in terminal.get("inputs", []):
            lines.append(
                f"| `{item['pin_name']}` | `{item['port_name']}` | `{item['or_density']}` / `{item['or_duty']}` | "
                f"`{item['x_density']}` / `{item['x_duty']}` | `{item['mismatch']}` |"
            )

    (out_dir / "seed_output_and_branch_trace.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    roots = load_root_sets(args.openroad_roots, args.xplace_roots)
    xrows = load_xplace(args.xplace_activity)
    by_pin, drivers_by_net, inputs_by_inst = load_openroad(args.openroad_pins)

    seed_result = compare_seed_outputs(roots, xrows, args)
    trace_result = trace_max_branch(args.target_pin, by_pin, drivers_by_net, inputs_by_inst, xrows, args)

    write_seed_outputs(args.out_dir, seed_result)
    write_branch_trace(args.out_dir, trace_result)
    write_report(args.out_dir, seed_result, trace_result)
    with (args.out_dir / "seed_output_and_branch_trace.json").open("w") as f:
        json.dump(clean_json({"seed_output": seed_result, "trace": trace_result}), f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
