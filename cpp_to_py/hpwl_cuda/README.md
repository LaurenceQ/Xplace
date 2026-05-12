# HPWL CUDA Module

GPU-accelerated Half-Perimeter Wirelength (HPWL) computation for standard wirelength evaluation in placement.

## Purpose

This module efficiently computes total wirelength across all nets using the half-perimeter wirelength metric on GPU. HPWL is the primary metric in analytical placement optimization.

## Functions

### Wirelength Computation
- **`hpwl(pos, hyperedge_list, hyperedge_list_end)`**
  - Computes total HPWL for all nets
  - Parameters:
    - `pos` - Pin positions tensor (num_pins × 2)
    - `hyperedge_list` - Pin indices for each net (flattened)
    - `hyperedge_list_end` - Cumulative indices marking net boundaries
  - Returns: Total wirelength as scalar tensor

### Pin Position Conversion
- **`node_pos_to_pin_pos(node_pos, pin_id2node_id, pin_rel_cpos)`**
  - Converts cell center positions to pin absolute positions
  - Parameters:
    - `node_pos` - Cell center positions
    - `pin_id2node_id` - Mapping of pin to parent cell
    - `pin_rel_cpos` - Pin offsets relative to cell center
  - Returns: Pin absolute positions tensor

## Algorithm

HPWL for each net = (max_x - min_x) + (max_y - min_y) of all net pins.

Total wirelength = sum of HPWL across all nets.

GPU parallelization: Each thread block processes one or more nets independently.

## Usage Example

```python
from cpp_to_py import hpwl_cuda
import torch

# Setup (typically done once per placement)
num_pins = 50000
pin_id2node_id = torch.tensor([...], device='cuda', dtype=torch.long)
pin_rel_cpos = torch.randn(num_pins, 2, device='cuda')

# During placement optimization
node_pos = torch.randn(num_nodes, 2, device='cuda')

# Convert cell positions to pin positions
pin_pos = hpwl_cuda.node_pos_to_pin_pos(
    node_pos, pin_id2node_id, pin_rel_cpos
)

# Compute total wirelength
total_wl = hpwl_cuda.hpwl(
    pin_pos, hyperedge_list, hyperedge_list_end
)
```

## Parameters Format

**hyperedge_list**: Flattened array of pin indices
```
Example: [pin0, pin1, pin2, pin3, pin4, pin5]
         for nets: Net0=[pin0, pin1], Net1=[pin2, pin3, pin4], Net2=[pin5]
```

**hyperedge_list_end**: Cumulative counts
```
Example: [2, 5, 6] marks end of each net
```

## Performance

- Fast GPU computation: O(total_pins) time
- Used at every placement iteration for evaluation
- Critical path for placement speed

## Related Components

- Used in `src/calculator.py` for solution evaluation
- Compared against `wa_wirelength_hpwl_cuda` for weighted-average metric
- Pin positions computed once per iteration, reused for wirelength and timing calculations
