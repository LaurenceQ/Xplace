# Wirelength Timing CUDA Module

GPU-accelerated combined wirelength and timing metric computation for timing-driven placement optimization.

## Purpose

This module computes a unified metric combining wirelength and timing gradients for timing-aware placement. Used in Xplace's timing optimization mode to minimize delay-critical paths while maintaining reasonable wirelength.

## Functions

### Timing-Weighted Wirelength
- **`merged_wl_loss_grad_timing(node_pos, timing_pin_grad, pin_id2node_id, pin_rel_cpos, node2pin_list, node2pin_list_end, hyperedge_list, hyperedge_list_end, net_mask, net_weight, hpwl_scale, gamma, deterministic)`**
  - Combines wirelength gradient with timing gradient for each pin
  - Timing-critical pins get higher gradient weights
  - Parameters:
    - `timing_pin_grad` - Timing gradient from GPU timer for each pin
    - `net_weight` - Per-net weight combining timing criticality and other factors
    - `gamma` - WA smoothing parameter (as in `wa_wirelength_hpwl_cuda`)
    - `deterministic` - Ensures reproducible computation
  - Returns: `[merged_loss, pin_grad_x, pin_grad_y, scaled_hpwl]`

## Algorithm

**Weighted Gradient Merge**:
```
merged_grad = alpha * wl_grad + beta * timing_grad

where:
  - wl_grad = WA wirelength gradient
  - timing_grad = path delay gradient from static timing analysis
  - alpha, beta = weights from net_weight parameter
```

**Timing Integration**:
- Timing gradients come from GPU timer's sensitivity analysis
- Only delay-critical paths contribute significant timing gradient
- Non-critical paths primarily optimized for wirelength

## Usage Example

```python
from cpp_to_py import wirelength_timing_cuda
import torch

# Timing analysis (from gputimer)
timing_pin_grad = timer.report_pin_slack(...)  # Slack gradients

# Setup placement data
node_pos = torch.randn(num_nodes, 2, device='cuda', requires_grad=True)

# Compute merged objective
merged_loss, pin_grad_x, pin_grad_y, scaled_hpwl = \
    wirelength_timing_cuda.merged_wl_loss_grad_timing(
        node_pos,
        timing_pin_grad,  # From GPU timer
        pin_id2node_id,
        pin_rel_cpos,
        node2pin_list, node2pin_list_end,
        hyperedge_list, hyperedge_list_end,
        net_mask,
        net_weight,  # Combines WL weight and timing criticality
        hpwl_scale,
        gamma=10.0,
        deterministic=True
    )

# Use merged gradients for optimization
cell_grad = compute_cell_grad_from_pin_grad(pin_grad_x, pin_grad_y, ...)
```

## Timing-Driven Placement Parameters

**From main.py for timing optimization**:
- `--timing_opt` - Enable timing-driven placement
- `--timing_freq` - How often to perform timing analysis (every N iterations)
- `--timing_init_weight` - Initial balance between WL and timing (0.0-1.0)
- `--decay_factor` - Reduce timing weight over iterations
- `--decay_boost` - Boost decay when timing becomes non-critical
- `--timing_start_iter` - When to begin timing optimization

## Integration Flow

1. **Compute initial WL gradients**: `wa_wirelength_hpwl_cuda`
2. **Run timing analysis**: `gputimer.update_timing()`
3. **Extract timing gradients**: `gputimer.report_pin_slack()`
4. **Merge gradients**: This module
5. **Apply optimization**: Update node positions based on merged gradient
6. **Iterate**: Repeat steps 2-5

## Advantages

- **Multi-Objective**: Balances timing and wirelength in single pass
- **Adaptive**: Net weights can change per iteration based on timing criticality
- **Efficient**: GPU-native computation avoids CPU-GPU data movement
- **Gradual**: Graceful transition from WL-only to timing-driven optimization

## Related Components

- Works with `gputimer` for timing analysis
- Integrates with `src/core/timing_opt.py` for weighting schemes
- Uses same gradient framework as `wa_wirelength_hpwl_cuda` for consistency
- Gradients feed `src/nesterov_optimizer.py` for position updates
