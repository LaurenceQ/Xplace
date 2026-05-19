# Memory Policy

This directory is the active memory system. Keep it small, current, and useful.

## Hard Limits

- Every active `*.md` file in this directory must be 50 lines or fewer.
- The 50-line cap must never cause skipped updates or lost conclusions.
- If the right file is full, create the next focused numbered file and link it
  from `00_INDEX.md`.
- If `00_INDEX.md` is full, compress navigation text; do not drop active topics.
- `00_INDEX.md` is navigation only: read order, file purpose, and global traps.
- Do not paste long logs, command history, stack traces, or broad transcripts.
- Store artifact paths, exact numeric conclusions, and the decision they imply.

## When To Update

Update memory before final response when a conclusion changes future behavior:

- A case needs a special env/config, e.g. skip-fanout `0` vs `300`.
- A timing miss is explained by root cause, not just observed.
- A DMP branch or kernel schedule is confirmed.
- A performance/memory bottleneck changes optimization priority.
- A previous belief is proven wrong or too broad.

## Add / Delete / Merge / Split

- Add a new focused file for a new topic that does not fit an existing file.
- Delete stale or false conclusions immediately after replacing them.
- Never delete a still-active fact just to satisfy the line cap.
- Merge duplicated facts into the most specific topical file.
- Split any file that would exceed 50 lines.
- Move historical detail to `archive/` only when it is no longer active.

## Write Format

- Prefer exact bullets: case, condition, result, implication.
- Include paths to canonical summaries/logs, not repeated raw output.
- Separate timing correctness from speed/memory status.
- Record negative rules, e.g. "do not debug bsg as skip-fanout timing bug."

## Startup Discipline

- Start from `00_INDEX.md`, then read the one or two relevant topical files.
- Do not grep large result trees until the focused memory is insufficient.
- If grep/logs reveal a durable fact, update memory immediately.
