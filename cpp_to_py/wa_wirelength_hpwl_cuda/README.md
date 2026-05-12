# Weighted-Average Wirelength CUDA Module

GPU-accelerated weighted-average (WA) wirelength computation with gradient calculation for analytical placement optimization.

## Purpose

Computes smooth, differentiable wirelength approximation for gradient-based placement. Uses log-sum-exp weighting to approximate HPWL with smooth gradients suitable for neural network optimization.

## Functions

### WA Wirelength Loss
- **`masked_scale_hpwl_sum(node_pos, pin_id2node_id, pin_rel_cpos, hyperedge_list, hyperedge_list_end, net_mask, hpwl_scale)`**
  - Computes sum of scaled HPWL for masked nets
  - Used to compute overall placement loss
  - Returns: Scalar loss value

### Combined Forward and Backward
- **`merged_forward_backward_with_masked_scale_hpwl(node_pos, pin_id2node_id, pin_rel_cpos, node2pin_list, node2pin_list_end, hyperedge_list, hyperedge_list_end, net_mask, hpwl_scale, gamma, deterministic)`**
  - Computes WA wirelength, pin gradients, and scaled HPWL in one pass
  - Efficient: reduces GPU memory traffic by fusing operations
  - Parameters:
    - `gamma` - Temperature parameter for log-sum-exp smoothing (controls gradient sharpness)
    - `deterministic` - Enable deterministic computation
  - Returns: `[wa_wirelength, pin_grad_x, pin_grad_y, scaled_hpwl]`

## Algorithm

**Weighted-Average Approximation**:
```
WA_length ≈ log(sum(exp(gamma * coord))) / gamma
           where coordinates are pin positions on each axis
```

As gamma → ∞, WA_length → HPWL (max coordinate)
As gamma → 0, WA_length → average coordinate

**Gradient Computation**:
- Analytically computed from log-sum-exp formulation
- Pin gradients w.r.t. positions
- Cell gradients via chain rule using `pin_id2node_id` mapping

## Usage Example

```python
from cpp_to_py import wa_wirelength_hpwl_cuda
import torch

# Setup
node_pos = torch.randn(num_nodes, 2, device='cuda', requires_grad=True)
pin_id2node_id = torch.tensor([...], device='cuda', dtype=torch.long)
pin_rel_cpos = torch.randn(num_pins, 2, device='cuda')
net_mask = torch.ones(num_nets, device='cuda', dtype=torch.bool)
hpwl_scale = torch.ones(num_nets, device='cuda')

# Compute WA wirelength and gradients
wa_wl, pin_grad_x, pin_grad_y, scaled_hpwl = \
    wa_wirelength_hpwl_cuda.merged_forward_backward_with_masked_scale_hpwl(
        node_pos, pin_id2node_id, pin_rel_cpos,
        node2pin_list, node2pin_list_end,
        hyperedge_list, hyperedge_list_end,
        net_mask, hpwl_scale,
        gamma=10.0,  # Smoothing parameter
        deterministic=True
    )

# Compute per-cell gradients from pin gradients
cell_grad = compute_cell_grad_from_pin_grad(pin_grad_x, pin_grad_y, ...)
```

## Parameters

- `gamma` - Smoothing parameter for WA approximation
  - Larger gamma → closer to HPWL but less smooth
  - Smaller gamma → smoother but farther from true HPWL
  - Typical range: 1-50
- `net_mask` - Boolean mask for nets to include (allows selective weighting)
- `hpwl_scale` - Per-net scaling factors (for timing-driven placement)
- `deterministic` - Ensures reproducible results (may be slightly slower)

## Advantages over Direct HPWL

1. **Differentiable**: Smooth gradients everywhere (unlike max/min in HPWL)
2. **Efficient**: Single GPU kernel computes loss and gradients
3. **Flexible**: Per-net masking and scaling for mixed objectives
4. **GPU-Optimized**: Fused forward-backward reduces memory and computation

## Related Components

- Used in `src/core/wa_wirelength_hpwl.py` for placement optimization
- Gradients drive `src/nesterov_optimizer.py` optimization
- Works with timing metrics in `wirelength_timing_cuda` for timing-driven placement
- Compared against `hpwl_cuda` for final solution evaluation
