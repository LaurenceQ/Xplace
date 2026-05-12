# Xplace Agent Memory Index

This directory stores durable notes for long-running
Xplace/GPUTimer/OpenROAD alignment work. Read this file after `AGENTS.md`, then
open only the files relevant to the current task.

## Active Files

- `01_human_feedback.md`: user preferences, hard constraints, and repeated
  corrections.
- `02_openroad_gr_rc_alignment.md`: active ISPD2025 OpenROAD global-route
  segment timing-alignment target, verified facts, scope, and success criteria.
- `03_pitfalls.md`: mistakes and semantic traps to check before changing
  timing, RC, Liberty parsing, or OpenROAD segment flows.
- `04_tools.md`: active ISPD2025 paths, OpenROAD binaries, segment reference
  scripts, and build/run notes.

## Archived Memory

Previous sky130 no-CRPR CSV alignment memory was copied to:

```text
/research/d7/ascstd/qkduan25/Xplace/md/agent_memory/archive/sky130_no_crpr_2026-05-12
```

Use that archive only when returning to sky130 CSV work. It is no longer the
active target.

## Current Priority

The active target is now ISPD2025 OpenROAD global-route segment timing
alignment:

```text
/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks
```

Task definition:

- Use OpenROAD-generated global-route segment files as the reference route/RC
  input.
- Read the same segments into Xplace/GPUTimer.
- Disable OpenROAD CRPR/CPPR for reference timing.
- Compare no-CRPR slack, WNS, and TNS against OpenROAD.
- Start from visible `ariane`, then expand only after the smallest case is
  understood.

Primary OpenROAD reference binary for this task:

```text
/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad
```

OpenROAD CRPR must be disabled explicitly in reference Tcl:

```tcl
sta::set_crpr_enabled 0
set ::sta_crpr_enabled 0
```

Do not claim alignment from RC graph loading alone. Alignment means WNS/TNS and
endpoint slack agree on the same global-route segment input with CRPR disabled.
