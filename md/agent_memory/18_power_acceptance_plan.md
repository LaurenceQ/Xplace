# Power Acceptance Plan
Scope: component-level power accept and mem/NanGate45 post-fix validation.

## Acceptance Standard
- OpenROAD power CSVs are oracle; compare against OPR/OpenROAD sums.
- Required pass: internal, leakage, and switching each <=1% relative error.
- Total power is still reported, but total-only <=1% is not sufficient.
- Preferred debug target: each pin activity density/duty within 5% of OPR.
- If activity target fails, use it to localize bugs; do not block on every tiny
  zero-near-zero row before component power is understood.

## Current Status
- Recent power/activity code changes exist, but full post-change power diff has
  not been run yet.
- Existing latch-fix probe showed visible/mempool_group Ptotal err `0.00157496`,
  but that probe is not the new component-level acceptance run.
- `tools/compare_ispd25_route_power_timing.py` already reports component
  rel_err columns; its aggregate `power_pass_1pct` may still be total-only.

## Run Plan
- Build in `gnn` from `build`, then `make install` before Python runs.
- Run visible/mempool_group first with cached OR golden and skip-fanout `300`.
- Record Pint/Psw/Pleak/Ptotal rel_err, worst instance/component, and compare CSV.
- If all three components pass <=1%, rerun blind/mempool_group, then mempool_cluster.
- If any component fails: diff instance CSV first, then inspect pin/internal/leakage
  rows for the failing component before changing code.
- For switching/internal activity failures, run full pin activity diff and target
  the largest non-zero activity deltas; use 5% per-pin activity as the debug bar.
