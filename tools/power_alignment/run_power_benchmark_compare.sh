#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPLACE_DIR="${XPLACE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

OUT_ROOT="${OUT_ROOT:-$XPLACE_DIR/result/sky130_power_alignment_$(date +%Y%m%d_%H%M%S)}"
DESIGN_LIST="${DESIGN_LIST:-$SCRIPT_DIR/designs_sky130_smoke.txt}"
OPENROAD_BIN="${OPENROAD_BIN:-/research/d7/ascstd/qkduan25/GNNTimer/openroad/build-check/bin/openroad}"
OPENROAD_ENV="${OPENROAD_ENV:-/research/d7/ascstd/qkduan25/app/openroad-deps/env.sh}"
GNNTIMER_DIR="${GNNTIMER_DIR:-/research/d7/ascstd/qkduan25/GNNTimer}"
PYTHON_BIN="${PYTHON_BIN:-/home/qkduan25/.conda/envs/gnn/bin/python}"
POWER_COMPARE_ONE="${POWER_COMPARE_ONE:-$SCRIPT_DIR/power_benchmark_compare_one.py}"
PLATFORM_PATH="${PLATFORM_PATH:-$XPLACE_DIR/sky130hd}"
DESIGN_PATH="${DESIGN_PATH:-/research/d7/ascstd/qkduan25/TimingPredict/data/netlists}"
OPENROAD_DUMP_DIR="${OPENROAD_DUMP_DIR:-$OUT_ROOT/openroad_dump}"
GPU="${GPU:-0}"
TOP_N="${TOP_N:-20}"

mkdir -p "$OUT_ROOT/openroad_logs" "$OUT_ROOT/xplace_logs" "$OUT_ROOT/xplace_compare" "$OPENROAD_DUMP_DIR"
cp "$DESIGN_LIST" "$OUT_ROOT/designs.txt"

SUMMARY_CSV="$OUT_ROOT/summary.csv"
echo "design,status,openroad_status,xplace_status,openroad_csv_source,gt_total,cuda_total,total_ratio,total_rel_err,total_abs_diff,gt_internal,cuda_internal,internal_ratio,gt_switching,cuda_switching,switching_ratio,gt_leakage,cuda_leakage,leakage_ratio,missing_names,openroad_s,xplace_total_s,dmp_s,power_api_s,notes" > "$SUMMARY_CSV"

while IFS= read -r design; do
  design="${design%%#*}"
  design="$(echo "$design" | xargs)"
  [[ -z "$design" ]] && continue

  echo "===== DESIGN $design =====" | tee -a "$OUT_ROOT/run.log"
  dstart=$(date +%s)
  openroad_status="ok"
  xplace_status="ok"
  openroad_csv_source="reused"
  notes=""
  gt_power="$OPENROAD_DUMP_DIR/${design}_power.csv"

  if [[ ! -s "$gt_power" ]]; then
    openroad_csv_source="generated"
    echo "[$design] dumping OpenROAD power..." | tee -a "$OUT_ROOT/run.log"
    if ! (
      set +u
      source "$OPENROAD_ENV"
      set -u
      cd "$GNNTIMER_DIR"
      OPENROAD_BIN="$OPENROAD_BIN" OUT_DIR="$OPENROAD_DUMP_DIR" LOG_DIR="$OUT_ROOT/openroad_logs" \
        PYTHONNOUSERSITE=1 ./eval.sh "$design" power
    ) > "$OUT_ROOT/openroad_logs/${design}_power.stdout.log" 2>&1; then
      openroad_status="fail"
      notes="openroad_failed"
      echo "[$design] OpenROAD failed; see $OUT_ROOT/openroad_logs/${design}_power.stdout.log" | tee -a "$OUT_ROOT/run.log"
      echo "$design,fail,$openroad_status,skip,$openroad_csv_source,,,,,,,,,,,,,,,$(( $(date +%s)-dstart )),,,,${notes}" >> "$SUMMARY_CSV"
      continue
    fi
  else
    echo "[$design] reusing OpenROAD CSV $gt_power" | tee -a "$OUT_ROOT/run.log"
  fi
  openroad_s=$(( $(date +%s)-dstart ))

  echo "[$design] running Xplace CUDA compare..." | tee -a "$OUT_ROOT/run.log"
  if ! PYTHONNOUSERSITE=1 "$PYTHON_BIN" "$POWER_COMPARE_ONE" \
      --design "$design" \
      --gt-power "$gt_power" \
      --out-dir "$OUT_ROOT/xplace_compare" \
      --gpu "$GPU" \
      --xplace-dir "$XPLACE_DIR" \
      --platform-path "$PLATFORM_PATH" \
      --design-path "$DESIGN_PATH" \
      --top-n "$TOP_N" \
      > "$OUT_ROOT/xplace_logs/${design}_compare.log" 2>&1; then
    xplace_status="fail"
    notes="xplace_failed"
    echo "[$design] Xplace compare failed; see $OUT_ROOT/xplace_logs/${design}_compare.log" | tee -a "$OUT_ROOT/run.log"
    echo "$design,fail,$openroad_status,$xplace_status,$openroad_csv_source,,,,,,,,,,,,,,,$openroad_s,,,,${notes}" >> "$SUMMARY_CSV"
    continue
  fi

  "$PYTHON_BIN" - "$OUT_ROOT/xplace_compare/${design}_summary.json" "$openroad_s" "$openroad_csv_source" <<'PY' >> "$SUMMARY_CSV"
import json
import sys

summary_path, openroad_s, openroad_csv_source = sys.argv[1:4]
s = json.load(open(summary_path))

def g(component, key):
    return s[component][key]

print(",".join(map(str, [
    s["design"], "ok", "ok", "ok", openroad_csv_source,
    g("total", "gt_sum"), g("total", "cuda_sum"), g("total", "ratio"),
    g("total", "rel_err"), g("total", "abs_diff"),
    g("internal", "gt_sum"), g("internal", "cuda_sum"), g("internal", "ratio"),
    g("switching", "gt_sum"), g("switching", "cuda_sum"), g("switching", "ratio"),
    g("leakage", "gt_sum"), g("leakage", "cuda_sum"), g("leakage", "ratio"),
    s.get("missing_gt_names_in_xplace", ""),
    openroad_s,
    s["timing_s"]["total_script"],
    s["timing_s"]["update_timing_dmp_spef"],
    s["timing_s"]["report_power_total_cuda"],
    "",
])))
PY
  echo "[$design] done" | tee -a "$OUT_ROOT/run.log"
done < "$DESIGN_LIST"

"$PYTHON_BIN" - "$OUT_ROOT" <<'PY'
import csv
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = list(csv.DictReader(open(root / "summary.csv")))
ok = [row for row in rows if row.get("status") == "ok"]
fail = [row for row in rows if row.get("status") != "ok"]

def fl(value):
    try:
        return float(value)
    except Exception:
        return float("nan")

report = root / "POWER_BENCHMARK_COMPARE_REPORT.md"
with report.open("w") as f:
    f.write("# Power Benchmark Compare Report\n\n")
    f.write(f"Output root: `{root}`\n\n")
    f.write(f"Designs total: {len(rows)}, ok: {len(ok)}, failed/skipped: {len(fail)}\n\n")
    f.write("| design | status | csv | total rel err | total ratio | total abs diff | internal ratio | switching ratio | leakage ratio | notes |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for row in rows:
        f.write(
            "| {design} | {status} | {csv} | {rel:.6e} | {ratio:.12g} | {absdiff:.6e} | "
            "{ir:.12g} | {sr:.12g} | {lr:.12g} | {notes} |\n".format(
                design=row.get("design", ""),
                status=row.get("status", ""),
                csv=row.get("openroad_csv_source", ""),
                rel=fl(row.get("total_rel_err", "")),
                ratio=fl(row.get("total_ratio", "")),
                absdiff=fl(row.get("total_abs_diff", "")),
                ir=fl(row.get("internal_ratio", "")),
                sr=fl(row.get("switching_ratio", "")),
                lr=fl(row.get("leakage_ratio", "")),
                notes=row.get("notes", ""),
            )
        )
    if ok:
        worst = max(ok, key=lambda r: abs(fl(r.get("total_ratio", "nan")) - 1.0) if math.isfinite(fl(r.get("total_ratio", "nan"))) else -1)
        f.write("\n")
        f.write(f"Worst total drift: `{worst['design']}` rel_err={worst.get('total_rel_err')} ratio={worst.get('total_ratio')}\n")

print(report)
PY

echo "DONE $OUT_ROOT"
