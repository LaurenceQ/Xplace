# Timing And RC Pitfalls

## Build And Workspace

- `cpp_to_py/gputimer` in Xplace is a symlink to `/research/d7/ascstd/qkduan25/gputimer_merged`.
- `git status` in Xplace may show `cpp_to_py/gputimer` as deleted/untracked because of that symlink setup. Check `/research/d7/ascstd/qkduan25/gputimer_merged` directly for real GPUTimer diffs.
- Build in `/research/d7/ascstd/qkduan25/Xplace/build` under conda env `gnn`.
- Always run `make install` after a successful build, otherwise Python may load an old extension.

## OpenROAD GR RC

- Do not assume GR has four corners. The useful GR wire R is scalar; pin cap/library timing can still vary by attr.
- If OpenROAD and Xplace differ by ns-level AT, compare RC and Liberty semantics before changing AT propagation.
- For RC alignment claims, compare endpoint pin AT/RAT/slack and at least one known critical path pin, not only report-level WNS/TNS.
- For missing RAT on RAM pins, check Liberty bus parsing before blaming SDC or clocks.

## Pin Name Normalization

- Xplace pin names commonly use `inst:port`.
- OpenROAD reports/CSV/dumps commonly use hierarchical `/` separators and a final `/port`, for example `inst/path/reg/D`.
- Convert only the final pin separator from `/` to `:` when matching OpenROAD names to Xplace names. Do not replace all hierarchy `/`.
- OpenROAD may escape bus brackets with backslashes, for example `reg\[3\]/D`, while Xplace may use `reg[3]:D`. Normalize escaped `\[` and `\]` before matching.
- Some OpenROAD strings can contain backslash-escaped special characters in hierarchical names. Preserve hierarchy, unescape only syntax escapes needed for matching, then try aliases.
- Do not conclude a pin is missing until aliases have been tried: raw name, normalized final `/` to `:`, escaped-bracket removal, and any existing `normalized_spef_name` helper.
- Pin-name mismatch has caused repeated false leads; make name normalization a first-class debug output in any RAT/slack comparison script.

## Sky130 RAT/Slack

- Existing sky130 CSV comparison summaries mainly measured AT/slew. They are not proof of WNS/TNS or RAT/slack alignment.
- For sky130 no-CRPR validation, use `/research/d7/ascstd/qkduan25/GNNTimer/csv_graph_sky130_crpr_off`. Compare OpenROAD CSV `required_*_ns` directly against Xplace `pinRAT`, and compare no-CRPR CSV `slack_*_ns` directly against Xplace `pin_slack` or endpoint slack.
- The default CSV directory `/research/d7/ascstd/qkduan25/GNNTimer/csv_graph_sky130` is CRPR-on. Compare default CSV `slack_*_ns` directly only when CRPR/CPPR is in scope; otherwise expect large RAT/slack differences from OpenSTA CRPR/CPPR/tag required semantics.
- Use `/research/d7/ascstd/qkduan25/Xplace/tools/compare_dmp_openroad_csv.py` as the standard comparison tool; see `04_tools.md`.
- Use `/research/d7/ascstd/qkduan25/GNNTimer/eval_crpr_off.sh` and `/research/d7/ascstd/qkduan25/GNNTimer/evaluation_crpr_off.tcl` to regenerate no-CRPR CSVs. Do not hand-create temporary OpenROAD Tcl scripts when the stable wrapper is sufficient.
- SDC/clock semantics are likely relevant. Current parser logs unsupported `set_propagated_clock`, `current_design`, and `set_max_fanout`; do not ignore these if RAT/slack is off.
- Be careful with early/late sign conventions: endpoint slack uses `AT-RAT` for early/min checks and `RAT-AT` for late/max checks in current DMP code. Verify against OpenROAD per attr before changing formulas.
- OpenROAD CSV `slack_*_ns` is not necessarily equal to simple `required-arrival` or `arrival-required`; `Sta::slack()` uses path-end slack and can include CRPR/CPPR via `PathEndClkConstrained::requiredTime() -> checkCrpr()`. If AT and `required_*_ns` match but `slack_*_ns` differs by several ns, treat it as a CRPR/CPPR gap unless proven otherwise.
- OpenSTA disables some sky130 `mux2` select-to-output arcs when data inputs are constant-driven or effectively undriven. If only a few designs show RAT on unexpected mux select paths, inspect OpenSTA arc enable/constant propagation before adding broad graph pruning.

## Liberty Parsing

- Do not collapse conditional `when` arcs with the same encoded from/to transition. Some AOI/OAI cells require a later conditional table to match OpenSTA max paths.
- Do not assume all loaded Liberty files share the first library's capacitance/time units. Scale port caps, LUT cap axes, LUT time axes, and timing table values.
- `rise_capacitance_range` and `fall_capacitance_range` affect per-attr pin caps and should be preserved.

## DMP RC

- OpenROAD-style small-driver-resistance checks use ohms, while DMP stores resistance in internal units.
- Explicit RC mode should upload scalar `edge_res` and attr-shaped `node_cap`.
- Pin caps should not be double-counted when the explicit graph already includes pin caps.
