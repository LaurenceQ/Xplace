# Timing And RC Pitfalls

## Build And Workspace

- `cpp_to_py/gputimer` in Xplace is a symlink to
  `/research/d7/ascstd/qkduan25/gputimer_merged`.
- `git status` in Xplace may show `cpp_to_py/gputimer` as deleted/untracked
  because of that symlink setup. Check
  `/research/d7/ascstd/qkduan25/gputimer_merged` directly for real GPUTimer
  diffs.
- Build Xplace in `/research/d7/ascstd/qkduan25/Xplace/build` under conda env
  `gnn`.
- Always run `make install` after a successful Xplace/GPUTimer build, otherwise
  Python may load an old installed extension.

## ISPD2025 OpenROAD GR Segment Alignment

- The active target is ISPD2025 no-CRPR timing alignment on OpenROAD
  global-route segment input, not sky130 CSV alignment.
- Use an absolute OpenROAD binary path. `openroad` was not found in the current
  shell PATH during initial inspection.
- Disable CRPR explicitly with `sta::set_crpr_enabled 0` and
  `set ::sta_crpr_enabled 0` before collecting WNS/TNS/slack reference data.
- `ISPD2025_benchmarks/set_design.tcl` currently hardcodes
  `DESIGN_NAME bsg_chip`; do not source it blindly for another design without
  overriding or using a cleaner wrapper.
- `ISPD2025_benchmarks/evaluate.tcl` does not disable CRPR and has
  `set_propagated_clock` commented out. Match OpenROAD semantics deliberately
  instead of assuming this script is already the desired reference.
- `ISPD2025_benchmarks/lib_setup.tcl` and
  `contest25/evaluation/openroad_evaluation.tcl` do not list exactly the same
  RAM Liberty/LEF files. Use one consistent reference script per comparison and
  record it.
- `ariane.route_segments_noclk` likely omits clock nets based on the filename.
  Confirm whether clocks are present before interpreting clock-path slack or
  CRPR-sensitive behavior.
- `contest25/evaluation/openroad_evaluation.tcl` calls
  `read_global_route_segments` on `${OUTPUT_DIR}/${DESIGN_NAME}.route`; the
  contest router output format is intended to be readable as global-route
  segments by OpenROAD.
- The route-file RC graph CSVs under `contest25/rc_graph_dumps` are useful for
  segment-feature inspection, but they are not proof that GPUTimer timing
  matches OpenROAD. Timing must be compared after reading the same segment or
  route file.

## OpenROAD GR RC

- Do not assume GR has four corners. Earlier ariane OpenROAD GR RC alignment
  found the useful GR wire resistance to be scalar; pin cap/library timing can
  still vary by attr.
- For RC alignment claims, compare endpoint pin AT/RAT/slack and at least one
  known critical path pin, not only report-level WNS/TNS.
- If OpenROAD and Xplace differ by ns-level AT, compare RC and Liberty
  semantics before changing AT propagation.
- For missing RAT on RAM pins, check Liberty bus parsing before blaming SDC or
  clocks.

## Pin Name Normalization

- Xplace pin names commonly use `inst:port`.
- OpenROAD reports commonly use hierarchical `/` separators and a final
  `/port`, for example `inst/path/reg/D`.
- Convert only the final pin separator from `/` to `:` when matching OpenROAD
  names to Xplace names. Do not replace all hierarchy `/`.
- OpenROAD may escape bus brackets with backslashes, for example `reg\[3\]/D`,
  while Xplace may use `reg[3]:D`. Normalize escaped `\[` and `\]` before
  matching.
- Do not conclude a pin is missing until aliases have been tried: raw name,
  normalized final `/` to `:`, escaped-bracket removal, and any existing
  `normalized_spef_name` helper.

## Liberty And Timing

- Do not collapse conditional `when` arcs with the same encoded from/to
  transition. OpenSTA keeps multiple conditional arcs as alternatives.
- Do not assume all loaded Liberty files share the first library's capacitance
  or time units. Scale port caps, LUT cap axes, LUT time axes, and timing table
  values.
- Be careful with early/late sign conventions: endpoint slack uses `AT-RAT` for
  early/min checks and `RAT-AT` for late/max checks in current DMP code.
- Default OpenSTA slack can include CRPR/CPPR via path-end required-time logic;
  for the active ISPD2025 target, use CRPR-off OpenROAD reference reports.
