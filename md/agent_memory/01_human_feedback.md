# Human Feedback And Constraints

## Scope

- Work in `/research/d7/ascstd/qkduan25/Xplace`.
- Do not switch to `xplace_dmp` or standalone timer unless explicitly asked.
- Do not revert user changes or unrelated dirty files.
- Do not commit unless explicitly asked.

## Style

- Be concrete. Report exact WNS/TNS, wall time, RSS, GPU memory, and paths.
- Do not guess when memory/logs can answer the question.
- Do not rerun OpenROAD unless the user asks; use saved matrix/reference logs.
- Do not add debug prints or profile logs by default.

## Acceptance

- Timing: Xplace direct `--route_segments` matches CRPR-off OpenROAD WNS/TNS
  within 1%.
- Performance: target is 4x end-to-end speedup over OpenROAD with lower memory.
- Optimize small/mid cases first; large cases should benefit later.

## OpenROAD Reference

- Binary: `/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad`.
- Disable CRPR with `sta::set_crpr_enabled 0` and
  `set ::sta_crpr_enabled 0`.
- Current reference logs are under `openroad_gr_logs_skip_fanout300`.

## Proxy

- Keep Codex/MCP subprocesses on `http://127.0.0.1:7891`.
- Use `cdx` or `/home/qkduan25/bin/codex` for new local sessions.
