# IO Parser Module

Comprehensive C++/pybind11 module for parsing VLSI design files (LEF, DEF, Bookshelf, Verilog) and building design databases. Core data loading infrastructure for Xplace.

## Purpose

Parses standard IC design formats and constructs in-memory database representation used throughout placement optimization. Handles multiple design input formats and supports various LEF/DEF variants.

## Main Functions

### Design File Parsing
- **`read(params, verbose_log=False, lite_mode=False, random_place=False, num_threads=20)`**
  - Main entry point for parsing design files
  - Parameters:
    - `params` - Dictionary with file paths and options:
      - `"lef"` or `"lefs"`: LEF file(s) with technology/cell information
      - `"cell_lef"` + `"tech_lef"`: Separate cell and tech LEF files
      - `"def"`: DEF file with physical design information
      - `"bookshelf_variety"` + `"aux"` + `"pl"`: Bookshelf benchmark format
      - `"lib"` or `"libs"`: Liberty timing library files
      - `"verilog"`: Verilog netlist (partial support)
      - `"sdc"`: Synopsys Design Constraints file
      - `"output"`: Output file path
    - `verbose_log` - Enable detailed logging during parsing
    - `lite_mode` - Load minimal routing/timing data for speed
    - `random_place` - Randomly initialize cell positions
    - `num_threads` - Threads for parallel parsing
  - Returns: Tuple of `(rawdb, gpdb)` database objects

### Supported Formats

**LEF (Liberty Exchange Format)**:
- Technology information (layers, vias, spacing rules)
- Cell library definitions (sizes, pins, obstructions)
- Single or multiple files

**DEF (Design Exchange Format)**:
- Physical design data (die size, rows, placement)
- Net connectivity and routing constraints
- Blockages and fences

**Bookshelf**:
- Legacy academic format (ISPD benchmarks)
- AUX file with file references
- PL file with cell placements
- NET file with netlist
- SCL file with standard cell row information

**Liberty**:
- Cell timing information
- Power characteristics
- Essential for timing-driven placement

**Verilog**:
- Netlist connectivity (partial support)
- Used when LEF/DEF connectivity incomplete

## Database Objects

### rawdb (Raw Database)
Low-level database with complete design information:
- Cells with geometry and timing data
- Nets and connectivity
- Layer routing information
- Design rules

### gpdb (GP Database)
High-level database optimized for placement:
- Cell types and groups
- Net hierarchies
- Pin information
- Optimization-friendly data structures

## Usage Example

```python
from utils import IOParser

# Setup parser
parser = IOParser()

# Prepare parameters
params = {
    "benchmark": "ispd2015",
    "lef": "data/raw/mgc_fft_1.lef",
    "def": "data/raw/mgc_fft_1.def",
    "design_name": "mgc_fft_1",
    "lib": "data/raw/saed32nm_27.lib",
}

# Parse design
rawdb, gpdb = parser.read(
    params,
    verbose_log=False,
    lite_mode=False,
    random_place=False,
    num_threads=20
)

# Access design information
num_cells = gpdb.num_cells()
num_nets = gpdb.num_nets()
die_info = gpdb.die_area()  # (lx, hx, ly, hy)
```

## Configuration Options

**From main.py parameters**:
- `--load_from_raw True` - Parse from benchmark files (standard)
- `--custom_path` - Custom LEF/DEF paths
- `--custom_json` - Multi-file configuration
- `--verbose_cpp_log` - Enable detailed parser logging
- `--cpp_log_level` - Set verbosity (0=DEBUG, 1=VERBOSE, 2=INFO)

## Supported Benchmarks

- **ISPD2005** - Classic benchmark (Bookshelf format)
- **ISPD2015** - Modern benchmark (LEF/DEF format)
- **ISPD2018/2019** - Recent benchmarks (LEF/DEF)
- **ICCAD2015/2019** - Contest benchmarks
- **MMS** - Mixed-size designs
- **Custom** - User-defined designs via JSON config

## Performance

**Parsing Time** (on typical 100K cell design):
- LEF parsing: ~2-5 seconds
- DEF parsing: ~5-10 seconds
- Liberty parsing: ~1-2 seconds
- **Lite mode**: 2-3x faster by skipping routing data

**Memory Usage**:
- Standard mode: 2-5 GB depending on design size
- Lite mode: 1-2 GB

**Optimization**:
- Multithreading speeds up independent parsing tasks
- Parallel cell/net processing with `num_threads`
- Use `load_from_raw=False` to load preprocessed `.pt` files for iteration

## Parser Limitations

- Verilog support is partial (use LEF/DEF when possible)
- Fence/blockage support incomplete in some formats
- Routing constraints simplified to grid-based representation

## Related Components

- Used by `src/database.py` to initialize placement database
- Output feeds into `src/initializer.py` for position initialization
- Database used throughout optimization in `src/core/` modules

## File Examples

Config for JSON mode:
```json
{
  "lefs": ["tech.lef", "cell.lef"],
  "def": "design.def",
  "libs": ["early.lib", "late.lib"],
  "sdc": "constraints.sdc",
  "benchmark": "custom"
}
```

## Error Handling

Common parse errors:
- Missing LEF/DEF files → FileNotFoundError
- Mismatched cell names → Warning, cells treated as unknown
- Invalid coordinates → Skip element with warning
- Incomplete timing data → Use default values

Enable `verbose_log=True` to debug parsing issues.
