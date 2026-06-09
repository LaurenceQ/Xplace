# Xplace Agent Notes

## Read First

- For long-running Xplace/GPUTimer/OpenROAD alignment work, start from
  `/research/d7/ascstd/qkduan25/Xplace/md/agent_memory/00_INDEX.md`.
- For detailed GPUTimer timing-flow alignment notes, start from
  `/research/d7/ascstd/qkduan25/gputimer_merged/md/timing_alignment/00_INDEX.md`.
- Keep this file short. Put detailed feedback, conclusions, pitfalls, tools,
  and command history in `/research/d7/ascstd/qkduan25/Xplace/md/agent_memory/`.

## Codex / MCP Proxy

- For new local Codex sessions, use `cdx` or `/home/qkduan25/bin/codex` so
  Codex/MCP child processes use `http://127.0.0.1:7891`.
- `~/.bash_profile` sources `~/.bashrc`, and `~/.bashrc` keeps `$HOME/bin`
  ahead of nvm so plain `codex` resolves to the proxy wrapper in new login
  shells.
- For diagnostics, use `proxy-test <cmd>`.
- Do not revert the `~/.bashrc` proxy default-preservation logic; it keeps
  injected Codex proxy variables from being overwritten by CUHK defaults.

## CUDA / C++ Build Rules

- Files with `.cpp` are standard C++ translation units.
- Do not add CUDA-only headers such as `cuda_runtime.h` to `.cpp` files.
- Do not put CUDA kernel launches, `cudaDeviceSynchronize`, or
  `cudaGetLastError` checks in `.cpp` files.
- Put CUDA runtime usage, kernel launches, and CUDA-side error/debug checks in
  `.cu` files.
- Expose CUDA functionality to `.cpp` through plain C++ wrapper functions
  defined in `.cu`.
- When debugging GPU issues, add first-failure CUDA error checks in the `.cu`
  launch wrapper that owns the kernel call.

## Timing Alignment Target

- Current goal: align Xplace/GPUTimer timing on ISPD2025 OpenROAD
  global-route segment input with OpenROAD no-CRPR slack/WNS/TNS.
- Benchmark root:
  `/research/d7/ascstd/qkduan25/contest25/ISPD2025_benchmarks`.
- Primary OpenROAD reference binary:
  `/research/d7/ascstd/qkduan25/OpenROAD/build/bin/openroad`.
- Use `md/agent_memory/02_openroad_gr_rc_alignment.md` and
  `md/agent_memory/04_tools.md` for current scope, segment scripts, and
  run/compare notes.

## Build / Run

- Build in conda env `gnn` from `/research/d7/ascstd/qkduan25/Xplace/build`.
- Always run `make install` after a successful build, otherwise Python may load
  an old installed `cpybin`.
- Default timer entry point: `python run_timer.py`.
- Design-specific timer entry point: `python run_timer.py --designName <name>`.
