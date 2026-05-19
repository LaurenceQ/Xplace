#!/usr/bin/env python3
"""Compare OpenROAD power roots against Xplace power seeds/candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


KEY_PINS = (
    "FE_RC_95112_0/ZN",
    "FE_OCPC470995_soc_qvalid/A",
    "FE_OCPC470995_soc_qvalid/Z",
    "g7171/ZN",
)


def norm_name(name: Any) -> str:
    text = str(name or "").strip().strip('"')
    text = text.replace(r"\[", "[").replace(r"\]", "]").replace("\\", "")
    return text.replace(":", "/")


def truth(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def read_probe_pins(path: Path | None) -> list[str]:
    if not path or not path.exists():
        return []
    pins: list[str] = []
    for raw in path.read_text(errors="replace").splitlines():
        raw = raw.split("#", 1)[0].strip()
        if raw:
            name = norm_name(raw)
            if name not in pins:
                pins.append(name)
    return pins


def read_tsv(path: Path, side: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(newline="", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = norm_name(row.get("pin_name") or row.get("pin") or row.get("query"))
            if not name:
                continue
            rec = rows.setdefault(name, {"pin_name": name, "raw_rows": 0})
            rec["raw_rows"] += 1
            if side == "openroad":
                rec["found"] = rec.get("found", False) or truth(row.get("found", "1"))
                rec["in_levelize_roots"] = rec.get("in_levelize_roots", False) or truth(row.get("in_levelize_roots"))
                rec["was_seeded"] = rec.get("was_seeded", False) or truth(row.get("was_seeded"))
                rec["reason"] = rec.get("reason") or row.get("reason", "")
                rec["activity_density"] = row.get("activity_density", rec.get("activity_density", ""))
                rec["activity_duty"] = row.get("activity_duty", rec.get("activity_duty", ""))
                rec["origin"] = row.get("origin", rec.get("origin", ""))
            else:
                rec["in_actual_seed"] = rec.get("in_actual_seed", False) or truth(row.get("in_actual_seed"))
                rec["in_candidate"] = rec.get("in_candidate", False) or truth(row.get("in_candidate"))
                rec["reason"] = rec.get("reason") or row.get("reason", "")
                for key in (
                    "pin_id",
                    "is_primary",
                    "is_clock",
                    "is_driver",
                    "is_load",
                    "power_fanin",
                    "timing_fanin",
                    "power_level",
                    "node_id",
                    "inst_name",
                    "cell_type",
                    "net_id",
                    "net_name",
                ):
                    if key in row and (key not in rec or rec[key] == ""):
                        rec[key] = row[key]
    return rows


def read_activity_csv(path: Path | None, xplace: bool) -> dict[str, dict[str, str]]:
    if not path or not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pin = norm_name(row.get("pin") or row.get("pin_name") or row.get("query"))
            if not pin:
                continue
            if xplace:
                out[pin] = {
                    "density": row.get("density", ""),
                    "duty": row.get("duty", ""),
                    "origin": row.get("origin", ""),
                }
            else:
                out[pin] = {
                    "density": row.get("activity_density", ""),
                    "duty": row.get("activity_duty", ""),
                    "origin": row.get("activity_origin", ""),
                }
    return out


def find_matches(rows: dict[str, dict[str, Any]], pin: str) -> list[str]:
    name = norm_name(pin)
    return sorted(k for k in rows if k == name or k.endswith("/" + name) or k.endswith(name))


def row_bool(row: dict[str, Any] | None, key: str) -> str:
    return "yes" if row and row.get(key) else "no"


def probe_table(
    probes: list[str],
    or_rows: dict[str, dict[str, Any]],
    xp_rows: dict[str, dict[str, Any]],
    or_activity: dict[str, dict[str, str]],
    xp_activity: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for probe in probes:
        or_match = find_matches(or_rows, probe)
        xp_match = find_matches(xp_rows, probe)
        or_name = or_match[0] if or_match else norm_name(probe)
        xp_name = xp_match[0] if xp_match else norm_name(probe)
        or_row = or_rows.get(or_name)
        xp_row = xp_rows.get(xp_name)
        oact = or_activity.get(or_name, {})
        xact = xp_activity.get(xp_name, {})
        table.append(
            {
                "probe": norm_name(probe),
                "openroad_pin": or_name if or_row else "",
                "or_root": row_bool(or_row, "in_levelize_roots"),
                "or_seeded": row_bool(or_row, "was_seeded"),
                "or_reason": (or_row or {}).get("reason", ""),
                "or_density": oact.get("density", (or_row or {}).get("activity_density", "")),
                "or_duty": oact.get("duty", (or_row or {}).get("activity_duty", "")),
                "xplace_pin": xp_name if xp_row else "",
                "x_seed": row_bool(xp_row, "in_actual_seed"),
                "x_candidate": row_bool(xp_row, "in_candidate"),
                "x_reason": (xp_row or {}).get("reason", ""),
                "x_power_fanin": (xp_row or {}).get("power_fanin", ""),
                "x_timing_fanin": (xp_row or {}).get("timing_fanin", ""),
                "x_density": xact.get("density", ""),
                "x_duty": xact.get("duty", ""),
            }
        )
    return table


def md_table(headers: list[str], rows: list[dict[str, Any]], limit: int | None = None) -> list[str]:
    rows = rows if limit is None else rows[:limit]
    if not rows:
        return ["_none_"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openroad-roots", type=Path, required=True)
    ap.add_argument("--xplace-roots", type=Path, required=True)
    ap.add_argument("--probe-pins", type=Path)
    ap.add_argument("--openroad-power-pins", type=Path)
    ap.add_argument("--xplace-probe", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    or_rows = read_tsv(args.openroad_roots, "openroad")
    xp_rows = read_tsv(args.xplace_roots, "xplace")
    probes = read_probe_pins(args.probe_pins)
    for pin in KEY_PINS:
        if pin not in probes:
            probes.append(pin)

    or_activity = read_activity_csv(args.openroad_power_pins, xplace=False)
    xp_activity = read_activity_csv(args.xplace_probe, xplace=True)

    or_seeded_not_x = [
        {
            "pin": name,
            "or_reason": row.get("reason", ""),
            "or_density": row.get("activity_density", ""),
            "or_duty": row.get("activity_duty", ""),
        }
        for name, row in sorted(or_rows.items())
        if row.get("was_seeded") and not xp_rows.get(name, {}).get("in_actual_seed")
    ][:50]
    x_seeded_not_or = [
        {
            "pin": name,
            "x_reason": row.get("reason", ""),
            "x_candidate": "yes" if row.get("in_candidate") else "no",
            "power_fanin": row.get("power_fanin", ""),
            "timing_fanin": row.get("timing_fanin", ""),
        }
        for name, row in sorted(xp_rows.items())
        if row.get("in_actual_seed") and not or_rows.get(name, {}).get("was_seeded")
    ][:50]
    xp_candidate_not_seed = [
        {
            "pin": name,
            "reason": row.get("reason", ""),
            "power_fanin": row.get("power_fanin", ""),
            "timing_fanin": row.get("timing_fanin", ""),
        }
        for name, row in sorted(xp_rows.items())
        if row.get("in_candidate") and not row.get("in_actual_seed")
    ][:50]
    probes_out = probe_table(probes, or_rows, xp_rows, or_activity, xp_activity)

    summary = {
        "openroad_root_count": sum(1 for row in or_rows.values() if row.get("in_levelize_roots")),
        "openroad_seeded_count": sum(1 for row in or_rows.values() if row.get("was_seeded")),
        "xplace_actual_seed_count": sum(1 for row in xp_rows.values() if row.get("in_actual_seed")),
        "xplace_candidate_count": sum(1 for row in xp_rows.values() if row.get("in_candidate")),
        "or_seeded_not_x_seeded_top50": or_seeded_not_x,
        "x_seeded_not_or_seeded_top50": x_seeded_not_or,
        "x_candidate_not_seed_top50": xp_candidate_not_seed,
        "probe_pins": probes_out,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "root_compare.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Power Root Compare",
        "",
        f"- OpenROAD roots: {summary['openroad_root_count']}",
        f"- OpenROAD seeded roots: {summary['openroad_seeded_count']}",
        f"- Xplace actual seeds: {summary['xplace_actual_seed_count']}",
        f"- Xplace candidate roots: {summary['xplace_candidate_count']}",
        "",
        "## Probe Pins",
        "",
    ]
    lines += md_table(
        [
            "probe",
            "or_root",
            "or_seeded",
            "or_reason",
            "or_density",
            "or_duty",
            "x_seed",
            "x_candidate",
            "x_reason",
            "x_power_fanin",
            "x_timing_fanin",
            "x_density",
            "x_duty",
        ],
        probes_out,
    )
    lines += ["", "## OR Seeded But X Not Seeded", ""]
    lines += md_table(["pin", "or_reason", "or_density", "or_duty"], or_seeded_not_x, 50)
    lines += ["", "## X Seeded But OR Not Seeded", ""]
    lines += md_table(["pin", "x_reason", "x_candidate", "power_fanin", "timing_fanin"], x_seeded_not_or, 50)
    lines += ["", "## X Candidate But Not Seed", ""]
    lines += md_table(["pin", "reason", "power_fanin", "timing_fanin"], xp_candidate_not_seed, 50)
    lines += ["", "## Key Pin Conclusion", ""]
    for pin in KEY_PINS:
        matches = [row for row in probes_out if row["probe"] == pin]
        if matches:
            row = matches[0]
            lines.append(
                f"- `{pin}`: OR root={row['or_root']} seeded={row['or_seeded']}; "
                f"X seed={row['x_seed']} candidate={row['x_candidate']} "
                f"reason={row['x_reason'] or 'n/a'}."
            )
    (args.out_dir / "root_compare.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
