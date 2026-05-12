# Human Feedback And Constraints

## Scope

- The user wants changes in Xplace, not `xplace_dmp` or standalone flows.
- Do not modify placement/access-point code unless the task explicitly requires it. The final goal is OpenROAD global-route segment RC/timing alignment.
- Current immediate goal: align sky130 RAT/slack and WNS/TNS across many cases. Do not stop at AT/slew or RC-only agreement.
- Do not touch unrelated dirty files or revert user changes.
- Do not commit unless explicitly asked.

## OpenROAD Sidecar

- For OpenROAD-side dumps, use `/research/d7/ascstd/qkduan25/GNNTimer/openroad/src/MyDump.cc`.
- Do not change the "pure" OpenROAD flow beyond extra sidecar dump functions.
- Keep one dump command name for GR RC: `my_dump_gr_rc`.

## Communication

- Avoid guessing. When unsure about OpenROAD semantics, inspect OpenROAD code or compare dumped intermediate values.
- The user cares about semantic equivalence first: RC graph, pin caps, AT/RAT/slack, and endpoint-level comparisons.
- Report concrete numeric deltas when claiming alignment.
- When comparing OpenROAD CSV/report pins to Xplace pins, handle name normalization explicitly before drawing conclusions. Pin-name mismatch has been a recurring source of false bugs.
