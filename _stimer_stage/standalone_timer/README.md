# Standalone CUDA/C++ Timer

This project is the new pure C++/CUDA timer target. It intentionally does not
link Python, pybind, Torch, or placement database APIs.

Current phase:

- Build a clean C++/CUDA project skeleton.
- Provide a C++ control program, `stimer_run`, replacing the old Python
  control path.
- Keep CUDA runtime usage inside `.cu` files.
- Keep `.cpp` files free of CUDA runtime headers and kernel launches.
- Use compact C++ structs for future CUDA launch arguments.

Build:

```bash
mkdir -p build
cd build
cmake ..
make -j8
ctest --output-on-failure
```

Run:

```bash
./stimer_run \
  --liberty slow.lib \
  --def design.def \
  --verilog design.v \
  --spef design.spef \
  --sdc design.sdc \
  --mode dmp
```

The first implementation pass only validates the project shape and C++ API.
Parser migration, timing graph construction, SPEF RC construction, DMP, and
full timing propagation will be migrated in later phases.
