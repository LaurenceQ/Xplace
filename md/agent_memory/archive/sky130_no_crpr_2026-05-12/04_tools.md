# Local Timing Alignment Tools

## Sky130 OpenROAD CSV vs DMP Compare

Tool:

```bash
/research/d7/ascstd/qkduan25/Xplace/tools/compare_dmp_openroad_csv.py
```

Purpose:

- Compare Xplace/GPUTimer DMP SPEF timing against OpenROAD sky130 CSV pin data.
- This tool now compares `at`, `slew`, `rat`, and `slack`.
- Use it before changing RAT/slack propagation. It is the primary smoke/regression tool for sky130 RAT/slack alignment.

Important behavior:

- It reads OpenROAD CSV fields:
  - `arrival_default_*_ns`
  - `slew_default_*_ns`
  - `required_default_*_ns`
  - `slack_default_*_ns`
  - `is_endpoint`
- It reads Xplace values directly through pybind:
  - `report_pin_at()`
  - `report_pin_slew()`
  - `report_pin_rat()`
  - `report_pin_slack()`
  - `endpoints_index()`
- It does not rely on `dump_timing_graph()` for RAT/slack comparison because that dump historically replaced NaN RAT with AT, hiding missing-RAT bugs.

Pin matching:

- Matching uses alias normalization instead of exact raw strings.
- It handles OpenROAD final `/port` versus Xplace `:port`.
- It unescapes OpenROAD-style `\[` and `\]` bus brackets.
- Summary JSON includes `missing_in_dmp_examples`, `missing_in_openroad_examples`, and alias collision examples.

Example single-case run:

```bash
cd /research/d7/ascstd/qkduan25/Xplace
source ~/.bashrc
conda activate gnn
python tools/compare_dmp_openroad_csv.py \
  --designs spm \
  --out-dir result/rat_slack_tool_smoke \
  --gpu 0 \
  --top-n 5
```

Example all-case run:

```bash
cd /research/d7/ascstd/qkduan25/Xplace
source ~/.bashrc
conda activate gnn
python tools/compare_dmp_openroad_csv.py \
  --out-dir result/sky130_rat_slack_compare \
  --gpu 0 \
  --top-n 20
```

Example all-case no-CRPR run:

```bash
cd /research/d7/ascstd/qkduan25/Xplace
source ~/.bashrc
conda activate gnn
python tools/compare_dmp_openroad_csv.py \
  --csv-dir /research/d7/ascstd/qkduan25/GNNTimer/csv_graph_sky130_crpr_off \
  --out-dir result/sky130_rat_slack_compare_crpr_off \
  --gpu 0 \
  --top-n 20
```

Output files:

- `summary.json`: full structured summaries for all completed cases.
- `summary.csv`: flat high-level metrics.
- `summaries/<design>.summary.json`: detailed per-design metrics and top diffs.
- `logs/<design>.log`: worker stdout/stderr.

Key metrics:

- `*_max_abs_ns`, `*_mean_abs_ns`, `*_p95_abs_ns` for `at`, `slew`, `rat`, `slack`.
- `early_wns_ns_openroad`, `early_wns_ns_dmp`, `early_wns_ns_diff`.
- `early_tns_ns_openroad`, `early_tns_ns_dmp`, `early_tns_ns_diff`.
- `late_wns_ns_openroad`, `late_wns_ns_dmp`, `late_wns_ns_diff`.
- `late_tns_ns_openroad`, `late_tns_ns_dmp`, `late_tns_ns_diff`.
- `top_endpoint_rat` and `top_endpoint_slack` are usually more actionable than all-pin `top_rat`/`top_slack`, because all-pin diffs can be dominated by clock pins or non-endpoints.

Historical smoke result before RAT/slack fixes:

- Case: `spm`
- AT max diff: about `9.6e-7 ns`
- Slew max diff: about `2.7e-7 ns`
- RAT max diff: about `8.47 ns`
- Slack max diff: about `8.47 ns`
- OpenROAD endpoint early WNS: `0.15011255 ns`
- DMP endpoint early WNS on matched OpenROAD endpoints: `-0.06387684 ns`
- Early WNS diff: about `-0.214 ns`
- Late WNS diff: about `-4.7e-7 ns`

Historical interpretation:

- Sky130 AT/slew could be effectively aligned while RAT/slack was still wrong.
- The first implementation target was early/min RAT/slack semantics, endpoint coverage, and SDC/clock propagation behavior.

## Sky130 OpenROAD CSV Extraction

Use the GNNTimer OpenROAD fork for sky130 CSV extraction because it contains the
custom `my_dump_pins` and `my_dump_graph` commands. Do not regenerate ad hoc Tcl
files in `/tmp` unless intentionally prototyping a new flow.

Important paths:

- OpenROAD binary: `/research/d7/ascstd/qkduan25/GNNTimer/openroad/build/bin/openroad`
- Default CRPR-on wrapper: `/research/d7/ascstd/qkduan25/GNNTimer/eval.sh`
- Default CRPR-on Tcl: `/research/d7/ascstd/qkduan25/GNNTimer/evaluation.tcl`
- Default CRPR-on CSVs: `/research/d7/ascstd/qkduan25/GNNTimer/csv_graph_sky130`
- No-CRPR wrapper: `/research/d7/ascstd/qkduan25/GNNTimer/eval_crpr_off.sh`
- No-CRPR Tcl: `/research/d7/ascstd/qkduan25/GNNTimer/evaluation_crpr_off.tcl`
- No-CRPR CSVs: `/research/d7/ascstd/qkduan25/GNNTimer/csv_graph_sky130_crpr_off`

The no-CRPR Tcl is the normal GNNTimer sky130 extraction flow plus:

```tcl
sta::set_crpr_enabled 0
set ::sta_crpr_enabled 0
```

Run one no-CRPR design:

```bash
cd /research/d7/ascstd/qkduan25/GNNTimer
./eval_crpr_off.sh spm pins
```

Regenerate the 21 stored no-CRPR pin CSVs:

```bash
cd /research/d7/ascstd/qkduan25/GNNTimer
for d in BM64 aes128 aes192 aes256 aes_cipher blabla cic_decimator des genericfir \
  jpeg_encoder picorv32a salsa20 spm synth_ram usb usb_cdc_core usbf_device \
  wbqspiflash xtea y_huff zipdiv; do
  ./eval_crpr_off.sh "$d" pins
done
```

The no-CRPR CSV directory has its own README:

```text
/research/d7/ascstd/qkduan25/GNNTimer/csv_graph_sky130_crpr_off/README.md
```

## Current Sky130 RAT/Slack Validation

Validated on 2026-05-12 after:

- preserving async recovery/removal constraint arcs,
- preventing constraint arcs from back-propagating RAT to related clock pins,
- filtering sky130 `mux2` select arcs that OpenSTA disables when a data input is
  constant-driven or undriven.

No-CRPR full sky130bench result, using
`/research/d7/ascstd/qkduan25/GNNTimer/csv_graph_sky130_crpr_off`:

- `at_max_abs_ns`: `0.0008621186035142614` at `des`
- `slew_max_abs_ns`: `0.0015450762886810576` at `aes192`
- `rat_max_abs_ns`: `0.0005569473413089554` at `aes192`
- `rat_endpoint_max_abs_ns`: `0.0004482284521500901` at `aes256`
- `rat_clock_max_abs_ns`: `0.0003824261254887773` at `aes256`
- `slack_max_abs_ns`: `0.0011262941735843413` at `aes256`

Default CRPR-on full sky130bench result with the same DMP run:

- `at_max_abs_ns`: `0.0008621186035142614` at `des`
- `slew_max_abs_ns`: `0.0015450762886810576` at `aes192`
- `rat_max_abs_ns`: `9.167707440561523` at `jpeg_encoder`
- `slack_max_abs_ns`: `9.167707415507813` at `jpeg_encoder`

Conclusion: AT/slew are aligned, and no-CRPR RAT/slack are aligned within about
`1e-3 ns` across the stored sky130bench cases. Large RAT/slack gaps against the
default CRPR-on CSVs are the expected OpenSTA CRPR/CPPR/tag required semantic
gap for the current GPUTimer implementation.
