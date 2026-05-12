# Route-Driven Detailed Placement Module

GPU/CPU hybrid module for routing-aware detailed placement that avoids placing cells under special nets (SNets) that cause design rule violations.

## Purpose

Performs post-global-placement optimization to shift cells away from routing blockages and power delivery network (PDN) while maintaining legalization. Prevents detailed routing issues by respecting special net constraints early in the flow.

## Functions

### Routing-Aware Cell Relocation
- **`dp_route_opt(node_lpos_init, node_size, die_lx, die_hx, die_ly, die_hy, site_width, row_height, rawdb, gpdb, K)`**
  - Optimizes cell positions to avoid routing violations from SNets
  - Uses local search to find valid placements
  - Parameters:
    - `node_lpos_init` - Initial cell positions from global placement (num_cells × 2)
    - `node_size` - Cell dimensions (num_cells × 2)
    - Die bounds: `die_lx, die_hx, die_ly, die_hy`
    - `site_width` - Placement site width
    - `row_height` - Row height
    - `rawdb` - Raw database with routing constraints
    - `gpdb` - GP database with cell/net information
    - `K` - Number of candidate positions to consider per cell
  - Returns: Optimized cell positions (num_cells × 2)

## Algorithm

**Problem**: Cells placed under M2+ SNets (power/ground rails) cause routing violations.

**Solution**:
1. Build spatial index of SNet shapes by layer and bin
2. For each cell, check if current position conflicts with SNets
3. If conflict detected, search K nearby row positions
4. Select position minimizing SNet overlap while respecting density constraints
5. Repeat until convergence

**Row-Based Movement**:
- Cells can move vertically (between rows) freely
- Horizontal movement constrained to avoid undue disruption
- Bounded local search (K candidates within radius)

## Usage Example

```python
from cpp_to_py import routedp
import torch

# Get optimized positions after global placement
optimized_pos = routedp.dp_route_opt(
    node_pos_initial,  # From GP
    node_size,
    die_lx, die_hx, die_ly, die_hy,
    site_width=1.0,
    row_height=5.4,
    rawdb=rawdb,  # From IO parser
    gpdb=gpdb,    # From IO parser
    K=10  # Check 10 nearest candidate rows
)

# Use optimized_pos for detailed placement
```

## Parameters

- **K** - Number of row candidates to evaluate
  - Larger K → better solution, slower execution
  - Typical range: 5-20
  - Recommendation: `K = min(10, num_rows // 1000)`

- **Conflict Resolution**:
  - Avoids overconstrained conflicts
  - Prioritizes M2 (first metal) SNets
  - Respects density limits from GP

## Integration in Xplace Flow

**Call sequence**:
1. Global placement: `run_placement_main()`
2. Cell inflation (optional): `use_cell_inflate=True`
3. **Routing-aware relocation**: `dp_route_opt()` ← This module
4. Legalization: `macro_legalization_main()`
5. Detailed placement: `detail_placement_main()`

**Main.py support**:
- Automatically applied when using routability-driven placement
- Parameters: `--use_cell_inflate True`

## SNet Types

**Avoided SNets**:
- M2 power/ground grids (M1 often skipped)
- Strategic blockages
- High-density regions

**Preserved SNets**:
- Signal nets (handled by detailed routing)
- Clock networks (if not in SNet form)

## Performance

**Time Complexity**:
- O(num_cells × K × average_snet_density)
- Typically: 5-30 seconds for 100K cells

**Quality**:
- Reduces routing violations by 10-50%
- Minimal wirelength increase (<1%)
- Better routed density

## Related Components

- Input: Global placement from `run_placement_main()`
- Output: Cell positions for `macro_legalization_main()`
- Database: Coordinates with `rawdb` from `io_parser`
- Evaluation: Used when `--use_cell_inflate True` or `--final_route_eval True`

## Limitations

- Only handles row-based movement (not arbitrary 2D relocation)
- Designed for row-based design rules (not all technologies)
- Optimizes one cell at a time (not global SNet-aware placement)
- Requires complete SNet definitions in LEF/DEF

## Future Improvements

- 2D local search for non-row-based technologies
- Clustering-based optimization for large designs
- Integration with global routing feedback loop
