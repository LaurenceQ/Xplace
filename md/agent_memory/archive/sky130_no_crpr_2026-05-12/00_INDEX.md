# Xplace Agent Memory Index

This directory stores durable notes for long-running Xplace/GPUTimer/OpenROAD alignment work. Read this file after `AGENTS.md`, then open only the files relevant to the current task.

## Files

- `01_human_feedback.md`: user preferences, hard constraints, and repeated corrections.
- `02_openroad_gr_rc_alignment.md`: current target, verified results, semantic decisions, and known-good reference values.
- `03_pitfalls.md`: mistakes already encountered and checks to run before making timing/RC changes.
- `04_tools.md`: local comparison/debug tools, OpenROAD CSV extraction paths,
  no-CRPR CSV regeneration, and known invocation patterns.

## Current Priority

The active sky130 no-CRPR timing target is aligned across the stored benchmark
set as of 2026-05-12: AT/slew and no-CRPR RAT/slack match OpenROAD within about
`1e-3 ns`. Use `/research/d7/ascstd/qkduan25/GNNTimer/csv_graph_sky130_crpr_off`
for RAT/slack validation while GPUTimer does not implement CRPR/CPPR.

The default `/research/d7/ascstd/qkduan25/GNNTimer/csv_graph_sky130` CSVs are
CRPR-on. Large default RAT/slack gaps can be the expected OpenSTA CRPR/CPPR/tag
required semantic gap and should not be treated as a current DMP bug unless the
task is explicitly to implement CRPR/CPPR.

The already-verified ariane OpenROAD GR RC path remains a reference for GR RC
semantics, but sky130 SPEF/CSV timing has its own no-CRPR validation trail in
`02_openroad_gr_rc_alignment.md` and `04_tools.md`. Do not spend effort on
placement access-point logic unless the user explicitly redirects there.
