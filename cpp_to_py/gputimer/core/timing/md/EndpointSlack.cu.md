# EndpointSlack.cu

## Purpose
CUDA kernels and GPUTimer methods for endpoint slack tensors and endpoint debug
CSV dumps.

## Main Entry Points
- `GPUTimer::update_endpoints()`
- `GPUTimer::debug_dump_endpoint_tests()`

## Data Ownership
Temporary endpoint tensors are owned by Torch. Persistent result tensors are
stored on `GPUTimer` as `endpoint_slacks` and `endpoint_pin_slacks`.

## Invariants
Setup/hold slack sign conventions and endpoint-pin minimum slack reduction must
remain unchanged.

## CUDA/C++ Boundary Notes
Kernel launches and CUDA error clearing are local to this `.cu` file. Public
GPUTimer declarations stay in `GPUTimer.h`.

## Acceptance Tests
- Build/install.
- Timing smoke should preserve WNS/TNS and endpoint slack reporting.
