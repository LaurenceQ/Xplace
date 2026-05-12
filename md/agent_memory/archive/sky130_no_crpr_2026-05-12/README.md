# Archived Sky130 No-CRPR Alignment Memory

This archive stores the previous active Xplace/GPUTimer/OpenROAD memory before
the target changed to ISPD2025 OpenROAD global-route segment timing alignment.

Archived on 2026-05-12.

Contents copied from the active memory set:

- `00_INDEX.md`
- `01_human_feedback.md`
- `02_openroad_gr_rc_alignment.md`
- `03_pitfalls.md`
- `04_tools.md`

Key archived conclusion:

- Sky130 no-CRPR RAT/slack alignment was validated across the stored 21-case
  sky130bench set within about `1e-3 ns`.
- The default sky130 OpenROAD CSVs remain CRPR-on; large default RAT/slack gaps
  are expected when GPUTimer has no CRPR/CPPR implementation.

Use this archive only when returning to sky130 CSV/no-CRPR alignment work.
