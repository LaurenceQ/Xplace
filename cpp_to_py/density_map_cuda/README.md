# Density Map CUDA Module

GPU-accelerated density map computation and gradient calculation for cell placement. Computes how cells contribute to bin occupancy and provides gradients for density-driven optimization.

## Purpose

This module computes spatial density distributions of cells across placement bins on GPU. It's core to Xplace's density-based penalty mechanism, which prevents congestion by penalizing high-density regions during placement optimization.

## Functions

### Normalization
- **`pre_normalize(node_pos, node_size, node_weight, expand_ratio, unit_len, normalize_node_info, num_bin_x, num_bin_y, num_nodes)`**
  - Normalizes node information to bin coordinate system
  - Returns normalized node data for forward pass

### Forward Pass
- **`forward(normalize_node_info, sorted_node_map, aux_mat, num_bin_x, num_bin_y, num_nodes, deterministic)`**
  - Computes density map from normalized node info
  - Uses sorted node mapping for efficiency
  - Returns density matrix for each bin

- **`forward_naive(node_pos, node_size, node_weight, unit_len, aux_mat, num_bin_x, num_bin_y, num_nodes, min_node_w, min_node_h, margin, clamp_node, deterministic)`**
  - Direct density computation without pre-normalization
  - Supports node clamping and margin parameters
  - Used when cell position bounds need enforcement

### Backward Pass
- **`backward(normalize_node_info, grad_mat, sorted_node_map, node_grad, grad_weight, num_bin_x, num_bin_y, num_nodes, deterministic)`**
  - Computes gradients of density loss w.r.t. cell positions
  - Used for backpropagation in PyTorch optimization
  - Returns node gradients for position updates

## Parameters

- `num_bin_x`, `num_bin_y` - Grid dimensions for density map
- `num_nodes` - Number of cells
- `node_weight` - Cell weights (for mixed-size designs)
- `expand_ratio` - Inflation ratio for routability-driven placement
- `min_node_w`, `min_node_h` - Minimum cell dimensions
- `margin` - Boundary margin
- `clamp_node` - Whether to enforce position bounds
- `deterministic` - Enable deterministic computation

## Usage Example

```python
from cpp_to_py import density_map_cuda
import torch

num_nodes = 10000
num_bin_x, num_bin_y = 512, 512

# Normalize nodes for fast processing
norm_info = density_map_cuda.pre_normalize(
    node_pos, node_size, node_weight,
    expand_ratio, unit_len,
    norm_info_storage,
    num_bin_x, num_bin_y, num_nodes
)

# Compute density map
density_map = density_map_cuda.forward(
    norm_info, sorted_node_map, aux_mat,
    num_bin_x, num_bin_y, num_nodes,
    deterministic=True
)

# Compute gradients
node_grad = density_map_cuda.backward(
    norm_info, grad_mat, sorted_node_map,
    node_grad_buffer, grad_weight,
    num_bin_x, num_bin_y, num_nodes,
    deterministic=True
)
```

## Algorithms

- **Forward**: Projects cells to bins using soft Gaussian overlap
- **Backward**: Computes analytical gradients for efficient optimization
- **Optimization**: Uses sorted node mapping to parallelize bin updates

## Related Components

- Works with `dct_cuda` for density smoothing via DCT
- Used in `src/core/electronic_density_layer.py` for density loss computation
- Gradients fed to `nesterov_optimizer.py` for position updates
