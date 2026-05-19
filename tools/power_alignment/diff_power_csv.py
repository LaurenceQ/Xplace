#!/usr/bin/env python3
"""Diff OpenROAD and Xplace power CSVs by instance name."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

COMPONENTS = ("internal", "switching", "leakage", "total")


def norm_name(name: str) -> str:
    return name.replace(r"\[", "[").replace(r"\]", "]")


def fnum(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except ValueError:
        return 0.0


def rel_err(actual: float, ref: float) -> float:
    return 0.0 if abs(ref) < 1e-30 and abs(actual) < 1e-30 else abs(actual - ref) / max(abs(ref), 1e-30)


def read_xplace(path: Path) -> dict[str, dict[str, float | str]]:
    rows: dict[str, dict[str, float | str]] = {}
    with path.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            name = norm_name(row.get("name", ""))
            if not name:
                continue
            rows[name] = {"cell_type": row.get("cell_type", ""), **{c: fnum(row, c) for c in COMPONENTS}}
    return rows


def zero_row(cell_type: str = "") -> dict[str, float | str]:
    return {"cell_type": cell_type, **{c: 0.0 for c in COMPONENTS}}


def push_top(top: list[dict[str, object]], row: dict[str, object], limit: int) -> None:
    top.append(row)
    top.sort(key=lambda x: float(x["abs_diff"]), reverse=True)
    del top[limit:]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_rows(path: Path | None, names: set[str], inst_key: str, out: Path) -> int:
    if not path or not path.exists():
        return 0
    count = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with path.open(newline="", errors="replace") as src, out.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames or [])
        writer.writeheader()
        for row in reader:
            if norm_name(row.get(inst_key, "")) in names:
                writer.writerow(row)
                count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openroad", type=Path, required=True)
    ap.add_argument("--xplace", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pins", type=Path)
    ap.add_argument("--arcs", type=Path)
    ap.add_argument("--leakage", type=Path)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    xrows = read_xplace(args.xplace)
    sums = {side: {c: 0.0 for c in COMPONENTS} for side in ("openroad", "xplace")}
    missing = 0
    top_by_comp = {c: [] for c in COMPONENTS}
    by_type: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    matched_x: set[str] = set()
    with args.openroad.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            name = norm_name(row.get("name", ""))
            x = xrows.get(name)
            if x is None:
                missing += 1
                x = zero_row(row.get("cell_type", ""))
            else:
                matched_x.add(name)
            cell = str(row.get("cell_type") or x.get("cell_type") or "")
            by_type[cell]["count"] += 1
            matched = name in xrows
            by_type[cell]["missing"] += 0 if matched else 1
            for c in COMPONENTS:
                o = fnum(row, c)
                xv = float(x[c])
                diff = xv - o
                sums["openroad"][c] += o
                sums["xplace"][c] += xv
                by_type[cell][f"openroad_{c}"] += o
                by_type[cell][f"xplace_{c}"] += xv
                push_top(top_by_comp[c], {
                    "name": name,
                    "cell_type": cell,
                    "component": c,
                    "openroad": o,
                    "xplace": xv,
                    "diff": diff,
                    "abs_diff": abs(diff),
                    "rel_err": rel_err(xv, o),
                }, args.top)

    xplace_only = 0
    for name, x in xrows.items():
        if name in matched_x:
            continue
        xplace_only += 1
        cell = str(x.get("cell_type") or "")
        by_type[cell]["count"] += 1
        by_type[cell]["xplace_only"] += 1
        for c in COMPONENTS:
            xv = float(x[c])
            sums["xplace"][c] += xv
            by_type[cell][f"xplace_{c}"] += xv
            push_top(top_by_comp[c], {
                "name": name,
                "cell_type": cell,
                "component": c,
                "openroad": 0.0,
                "xplace": xv,
                "diff": xv,
                "abs_diff": abs(xv),
                "rel_err": rel_err(xv, 0.0),
            }, args.top)

    summary = {"openroad_csv": str(args.openroad), "xplace_csv": str(args.xplace),
               "missing_openroad_names_in_xplace": missing,
               "xplace_only_names_not_in_openroad": xplace_only,
               "components": {}}
    for c in COMPONENTS:
        o = sums["openroad"][c]
        x = sums["xplace"][c]
        summary["components"][c] = {
            "openroad": o,
            "xplace": x,
            "ratio": x / o if o else math.nan,
            "rel_err": rel_err(x, o),
            "top_abs_diff": top_by_comp[c][0] if top_by_comp[c] else {},
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for c in COMPONENTS:
        write_csv(args.out / f"top_{c}_instances.csv", top_by_comp[c])
    type_rows = []
    for cell, vals in by_type.items():
        out = {
            "cell_type": cell,
            "count": int(vals["count"]),
            "missing": int(vals["missing"]),
            "xplace_only": int(vals["xplace_only"]),
        }
        for c in COMPONENTS:
            o = vals[f"openroad_{c}"]
            x = vals[f"xplace_{c}"]
            out[f"openroad_{c}"] = o
            out[f"xplace_{c}"] = x
            out[f"{c}_diff"] = x - o
            out[f"{c}_rel_err"] = rel_err(x, o)
        type_rows.append(out)
    type_rows.sort(key=lambda r: abs(float(r["total_diff"])), reverse=True)
    write_csv(args.out / "cell_type_diff.csv", type_rows[:200])

    worst_names = {str(r["name"]) for rows in top_by_comp.values() for r in rows[: min(10, args.top)]}
    summary["extracted_rows"] = {
        "pins": extract_rows(args.pins, worst_names, "inst_name", args.out / "worst_openroad_pins.csv"),
        "arcs": extract_rows(args.arcs, worst_names, "inst_name", args.out / "worst_openroad_internal_arcs.csv"),
        "leakage": extract_rows(args.leakage, worst_names, "inst_name", args.out / "worst_openroad_leakage.csv"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["components"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
