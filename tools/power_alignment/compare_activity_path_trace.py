#!/usr/bin/env python3
"""Compare OpenROAD and Xplace path activity traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def norm(name: Any) -> str:
    text = str(name or "").strip().strip('"').replace("\\", "")
    return text.replace(":", "/")


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def first_update(rows: list[dict[str, str]], to_pin: str, from_pin: str = "") -> dict[str, str] | None:
    to_pin = norm(to_pin)
    from_pin = norm(from_pin)
    best: dict[str, str] | None = None
    for row in rows:
        event = row.get("event", "")
        if event not in {"set_activity", "net_sink", "seq_seed", "update"}:
            continue
        if norm(row.get("to_pin")) != to_pin:
            continue
        if from_pin and norm(row.get("from_pin")) not in {"", from_pin}:
            continue
        if as_float(row.get("density_new")) <= 0.0 and as_float(row.get("density_old")) <= 0.0:
            continue
        if best is None:
            best = row
            continue
        if (as_int(row.get("pass"), 10), row.get("level_tag", "")) < (
            as_int(best.get("pass"), 10),
            best.get("level_tag", ""),
        ):
            best = row
    return best


def first_observed(rows: list[dict[str, str]], to_pin: str, from_pin: str = "") -> dict[str, str] | None:
    to_pin = norm(to_pin)
    from_pin = norm(from_pin)
    activity_best: dict[str, str] | None = None
    queue_best: dict[str, str] | None = None
    for row in rows:
        if norm(row.get("to_pin")) != to_pin:
            continue
        if from_pin and norm(row.get("from_pin")) not in {"", from_pin}:
            continue
        event = row.get("event", "")
        if event not in {"set_activity", "net_sink", "seq_seed", "update", "enqueue", "visit", "seq_pending"}:
            continue
        target = "activity" if event in {"set_activity", "net_sink", "seq_seed", "update"} else "queue"
        current = activity_best if target == "activity" else queue_best
        if current is None or (as_int(row.get("pass"), 10), row.get("level_tag", "")) < (
            as_int(current.get("pass"), 10),
            current.get("level_tag", ""),
        ):
            if target == "activity":
                activity_best = row
            else:
                queue_best = row
    return activity_best or queue_best


def classify_divergence(item: dict[str, Any], x_observed: dict[str, str] | None) -> str:
    edge_kind = item.get("edge_kind", "")
    if x_observed is None:
        if edge_kind.startswith("seq"):
            return "seq_pending_or_q_identification"
        return "xplace_queue_or_enqueue_propagation"
    event = x_observed.get("event", "")
    reason = x_observed.get("reason", "")
    density = as_float(x_observed.get("density_new"))
    if event in {"set_activity", "net_sink", "update", "seq_seed"} and density <= 0.0:
        if edge_kind.startswith("seq") or "seq" in reason:
            return "liberty_sequential_expr_or_seq_seed_eval"
        return "combinational_func_activity_eval"
    if event in {"enqueue", "visit"}:
        return "xplace_queue_reached_but_no_positive_activity"
    return "unclassified_activity_mismatch"


def summarize_row(row: dict[str, str] | None) -> dict[str, Any]:
    if row is None:
        return {}
    return {
        "pass": as_int(row.get("pass")),
        "level_tag": row.get("level_tag", ""),
        "event": row.get("event", ""),
        "from_pin": norm(row.get("from_pin")),
        "to_pin": norm(row.get("to_pin")),
        "density_new": as_float(row.get("density_new")),
        "duty_new": as_float(row.get("duty_new")),
        "reason": row.get("reason", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-path", required=True, type=Path)
    parser.add_argument("--openroad-trace", required=True, type=Path)
    parser.add_argument("--xplace-trace", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    path_rows = read_tsv(args.trace_path)
    or_rows = read_tsv(args.openroad_trace)
    xp_rows = read_tsv(args.xplace_trace)

    real_steps = [row for row in path_rows if as_int(row.get("step")) >= 0]
    first_div: dict[str, Any] | None = None
    checked: list[dict[str, Any]] = []
    for row in real_steps:
        from_pin = norm(row.get("from_pin"))
        to_pin = norm(row.get("to_pin"))
        or_update = first_update(or_rows, to_pin, from_pin)
        xp_update = first_update(xp_rows, to_pin, from_pin)
        xp_observed = first_observed(xp_rows, to_pin, from_pin)
        item = {
            "path_id": as_int(row.get("path_id")),
            "step": as_int(row.get("step")),
            "from_pin": from_pin,
            "to_pin": to_pin,
            "edge_kind": row.get("edge_kind", ""),
            "openroad": summarize_row(or_update),
            "xplace": summarize_row(xp_update),
            "xplace_observed": summarize_row(xp_observed),
        }
        checked.append(item)
        if or_update is not None and xp_update is None:
            first_div = item | {
                "reason": "openroad_updated_xplace_no_positive_update",
                "diagnosis": classify_divergence(item, xp_observed),
            }
            break
        if or_update is not None and xp_update is not None:
            od = as_float(or_update.get("density_new"))
            xd = as_float(xp_update.get("density_new"))
            if od > 0.0 and xd <= 0.0:
                first_div = item | {
                    "reason": "xplace_density_zero",
                    "diagnosis": classify_divergence(item, xp_observed),
                }
                break

    seeds = sorted({norm(row.get("seed_pin")) for row in real_steps if row.get("seed_pin")})
    targets = sorted({norm(row.get("target_pin")) for row in path_rows if row.get("target_pin")})
    report = {
        "trace_path": str(args.trace_path),
        "openroad_trace": str(args.openroad_trace),
        "xplace_trace": str(args.xplace_trace),
        "seed_roots": seeds,
        "targets": targets,
        "first_divergence": first_div,
        "checked_steps": checked[:200],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Activity Path Trace Compare",
        "",
        f"- Trace path: `{args.trace_path}`",
        f"- OpenROAD trace: `{args.openroad_trace}`",
        f"- Xplace trace: `{args.xplace_trace}`",
        f"- Common seed roots on traced paths: {', '.join(seeds) if seeds else '_none_'}",
        f"- Targets: {', '.join(targets) if targets else '_none_'}",
        "",
        "## First Divergence",
    ]
    if first_div is None:
        lines.append("- No OR-positive / X-missing divergence found in traced path rows.")
    else:
        lines.extend(
            [
                f"- Reason: `{first_div['reason']}`",
                f"- Diagnosis: `{first_div.get('diagnosis', 'unclassified')}`",
                f"- Path/step: `{first_div['path_id']}/{first_div['step']}`",
                f"- Arc: `{first_div['from_pin']} -> {first_div['to_pin']}`",
                f"- Edge kind: `{first_div['edge_kind']}`",
                f"- OpenROAD: `{first_div['openroad']}`",
                f"- Xplace: `{first_div['xplace'] or 'missing'}`",
                f"- Xplace observed: `{first_div.get('xplace_observed') or 'missing'}`",
            ]
        )
    lines.extend(["", "## Checked Steps", ""])
    lines.append("| path | step | edge | OR | X |")
    lines.append("|---:|---:|---|---|---|")
    for item in checked[:50]:
        lines.append(
            f"| {item['path_id']} | {item['step']} | `{item['from_pin']} -> {item['to_pin']}` "
            f"| `{item['openroad'] or 'missing'}` | `{item['xplace'] or item['xplace_observed'] or 'missing'}` |"
        )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
