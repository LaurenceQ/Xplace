# FLUTE C++ Module

C++ bindings for FLUTE (Fast LookUp Table Rectilinear Steiner Minimal Tree) wirelength computation. Provides efficient rectilinear Steiner tree approximation.

## Purpose

Computes accurate Rectilinear Steiner Minimal Tree (RSMT) wirelength for nets as an alternative to simple HPWL. Used for more accurate wirelength estimation and evaluation.

## Functions

### Single Net RSMT
- **`flute_rsmt_wl(xsvec, ysvec)`**
  - Computes RSMT wirelength for a single net
  - Parameters:
    - `xsvec` - X coordinates of net pins (list of ints)
    - `ysvec` - Y coordinates of net pins (list of ints)
  - Returns: RSMT wirelength as integer

### Multi-Net RSMT (Multithreaded)
- **`flute_rsmt_wl_mt(pos_x, pos_y, hyperedge_list, hyperedge_list_end, num_threads)`**
  - Computes RSMT wirelength for multiple nets in parallel
  - Parameters:
    - `pos_x`, `pos_y` - Pin coordinate lists (int)
    - `hyperedge_list` - Flattened array of pin indices for all nets
    - `hyperedge_list_end` - Cumulative index marking end of each net
    - `num_threads` - Number of threads for parallel computation
  - Returns: List of RSMT wirelengths, one per net

### LUT Initialization
- **`read_lut(powv_filename, post_filename)`**
  - Loads FLUTE lookup tables from files
  - Must be called once before using FLUTE functions
  - Parameters:
    - `powv_filename` - Path to FLUTE POWV9 table (e.g., `thirdparty/flute_mp/lut.ICCAD2015/POWV9.dat`)
    - `post_filename` - Path to FLUTE POST9 table (e.g., `thirdparty/flute_mp/lut.ICCAD2015/POST9.dat`)

## FLUTE Algorithm

FLUTE approximates RSMT using lookup tables (LUTs):
- **Fast**: O(k log k) where k = net degree
- **Accurate**: Usually within 3-5% of optimal RSMT
- **Deterministic**: Always produces same result for same input

## Usage Example

```python
from cpp_to_py import flute_cpp
import torch

# Initialize FLUTE (do once at program start)
flute_cpp.read_lut(
    "thirdparty/flute_mp/lut.ICCAD2015/POWV9.dat",
    "thirdparty/flute_mp/lut.ICCAD2015/POST9.dat"
)

# Single net example
pin_x = [0, 100, 50]
pin_y = [0, 0, 100]
wl = flute_cpp.flute_rsmt_wl(pin_x, pin_y)
print(f"Single net RSMT: {wl}")

# Multiple nets (parallel)
pos_x = [0, 100, 50, 200, 150]
pos_y = [0, 0, 100, 0, 50]

# Net 0: pins 0,1,2
# Net 1: pins 3,4
hyperedge_list = [0, 1, 2, 3, 4]
hyperedge_list_end = [3, 5]  # Net 0 ends at index 3, Net 1 at index 5

wls = flute_cpp.flute_rsmt_wl_mt(
    pos_x, pos_y,
    hyperedge_list, hyperedge_list_end,
    num_threads=20
)
print(f"Multi-net RSMT: {wls}")  # [wl_net0, wl_net1]
```

## RSMT vs HPWL Comparison

| Metric | RSMT | HPWL |
|--------|------|------|
| Accuracy | Higher (3-5% of optimal) | Lower (~10% of optimal) |
| Speed | Fast (~1ms per net) | Faster (~0.1ms per net) |
| Rectilinearity | Yes (Steiner points) | No (bounding box) |
| Complexity | O(k log k) | O(k) |

RSMT ≈ 0.85-0.95 × HPWL for typical designs.

## Multithreading

- Thread-safe via local job distribution
- Linear speedup up to hardware core count
- Automatic fallback to single thread if `num_threads <= 1`
- Recommended: `num_threads = (num_nets // 100)` or system core count

## LUT Files

FLUTE requires lookup table files:
- Location: `thirdparty/flute_mp/lut.ICCAD2015/`
- POWV9.dat: Power routing LUT (~50 MB)
- POST9.dat: Routing pattern LUT (~50 MB)
- Included in repository

## Performance Notes

- First `read_lut()` call takes ~1 second (one-time LUT load)
- Subsequent calls use cached LUT
- Parallel computation efficient for >1000 nets
- Single net faster than parallel overhead for small numbers of nets

## Related Components

- Used in `src/core/flute.py` as Python wrapper
- Alternative to `hpwl_cuda` for more accurate solution evaluation
- Useful for final quality assessment before tape-out
