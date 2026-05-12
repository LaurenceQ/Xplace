# base.h

## Purpose
Global CUDA configuration header for GPUTimer module.

## Key Definitions

| Macro | Value | Purpose |
|-------|-------|---------|
| `BLOCK_SIZE` | 512 | Threads per CUDA block for kernel launches |
| `BLOCK_NUMBER(n)` | `(n + BLOCK_SIZE - 1) / BLOCK_SIZE` | Compute # blocks needed for n threads |
| `NUM_ATTR` | 4 | Timing corners per pin: early-rise, early-fall, late-rise, late-fall |
| `index_type` | `int` | Type for indexing (pin_id, arc_id, etc.) |

## Timing Corners (NUM_ATTR = 4)

- **Index 0**: Early timing, Rise transition (el=0, rf=0)
- **Index 1**: Early timing, Fall transition (el=0, rf=1)
- **Index 2**: Late timing, Rise transition (el=1, rf=0)
- **Index 3**: Late timing, Fall transition (el=1, rf=1)

## Utilities

- `Functors`: Variadic template struct enabling overloaded lambda operations
