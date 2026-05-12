# Draw Placement Module

CPU-based visualization module for rendering placement solutions using Cairo graphics library. Generates PNG/PDF images of chip layouts.

## Purpose

Creates visual representations of placement solutions showing cell positions, density maps, and design features. Useful for debugging, presentation, and analysis of placement results.

## Functions

### Draw Placement
- **`draw(node_pos_x, node_pos_y, node_size_x, node_size_y, node_name, die_info, site_info, bin_size_info, node_types_indices, ele_type_to_rgba_vec, node_special_type, filename, width, height, draw_contents)`**
  - Renders placement to file (PNG/PDF via Cairo)
  - Parameters:
    - `node_pos_x`, `node_pos_y` - Cell center coordinates
    - `node_size_x`, `node_size_y` - Cell dimensions
    - `node_name` - Cell identifiers for labels
    - `die_info` - Die boundaries (lx, hx, ly, hy)
    - `site_info` - Site width and row height
    - `bin_size_info` - Bin dimensions for visualization overlays
    - `node_types_indices` - Cell type groupings for coloring
    - `ele_type_to_rgba_vec` - Color mappings (R, G, B, A) for element types
    - `node_special_type` - Special markers for specific cells (e.g., macros)
    - `filename` - Output file path (.png or .pdf)
    - `width`, `height` - Canvas dimensions in pixels
    - `draw_contents` - List of visualization layers to include
  - Returns: Boolean success status

## Visualization Layers

**draw_contents** parameter can include:
- `"placement"` - Cell rectangles with positions
- `"density_map"` - Heatmap of placement density
- `"congestion"` - Routing congestion overlay
- `"grid"` - Placement bin grid
- `"macro"` - Highlight macro cells
- `"net_path"` - Steiner tree visualizations

## Color Mapping

Element type to RGBA mapping defines visualization colors:
```python
{
    "type_name": (red, green, blue, alpha),
    # Examples:
    "std_cell": (0.8, 0.2, 0.2, 1.0),  # Red for standard cells
    "macro": (0.2, 0.2, 0.8, 1.0),     # Blue for macros
    "filler": (0.5, 0.5, 0.5, 0.5),    # Gray for filler cells
}
```

## Usage Example

```python
from cpp_to_py import draw_placement
import torch

# Convert placement tensors to lists for drawing
node_pos_x = node_pos[:, 0].cpu().numpy().tolist()
node_pos_y = node_pos[:, 1].cpu().numpy().tolist()
node_size_x = node_size[:, 0].cpu().numpy().tolist()
node_size_y = node_size[:, 1].cpu().numpy().tolist()

# Define visualization
ele_type_to_rgba = [
    ("std_cell", 0.8, 0.2, 0.2, 1.0),
    ("macro", 0.2, 0.2, 0.8, 1.0),
]

node_types = [(0, 100, "std_cell"), (100, 150, "macro")]

# Draw placement
success = draw_placement.draw(
    node_pos_x, node_pos_y,
    node_size_x, node_size_y,
    node_names,
    (die_lx, die_hx, die_ly, die_hy),
    (site_width, row_height),
    (bin_width, bin_height),
    node_types,
    ele_type_to_rgba,
    node_special_type,
    "placement_result.png",
    width=2000, height=2000,
    draw_contents=["placement", "grid", "density_map"]
)
```

## Output Formats

Supports output via Cairo:
- **PNG** - Raster format, good for web/reports
- **PDF** - Vector format, scalable quality

Canvas size recommendations:
- Small designs (<10K cells): 1000×1000 pixels
- Medium designs (10K-100K cells): 2000×2000 pixels
- Large designs (>100K cells): 4000×4000 pixels or larger

## Performance

- Single-threaded CPU rendering
- Time scales with canvas size and number of cells
- Can be slow for very large designs with high resolution
- Recommended to render after placement completes, not during optimization

## Dependencies

- **Cairo** - Graphics library for rendering
- **System**: libcairo2 package must be installed

## Related Components

- Used by `src/evaluator.py` for result visualization
- Integrated with `main.py` when `--draw_placement True`
- Output saved to `result/exp_id/eval/` directory
