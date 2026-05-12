# DCT CUDA Module

GPU-accelerated 2D Discrete Cosine Transform (DCT) and Inverse DCT (IDCT) using CUDA for efficient density smoothing in placement optimization.

## Purpose

This module provides high-performance 2D DCT and inverse transforms on GPU, essential for computing smooth density maps in global placement. The transforms use FFT-based algorithms for efficiency.

## Functions

### Forward Transforms
- **`dct2_fft2(x, expkM, expkN, out, buf)`** - 2D DCT using FFT (forward)
- **`idct2_fft2(x, expkM, expkN, out, buf)`** - 2D inverse DCT using FFT (forward)

### Specialized Transforms
- **`idct_idxst(x, expkM, expkN, out, buf)`** - IDCT combined with IDXST (inverse DST)
- **`idxst_idct(x, expkM, expkN, out, buf)`** - IDXST combined with IDCT

## Parameters

- `x` - Input tensor (CUDA)
- `expkM` - Precomputed exponential factors for M dimension
- `expkN` - Precomputed exponential factors for N dimension
- `out` - Output tensor (CUDA)
- `buf` - Temporary buffer for intermediate computations

## Usage Example

```python
from cpp_to_py import dct_cuda
import torch

# Prepare input tensor and precomputed factors
x = torch.randn(batch_size, height, width, device='cuda')
expkM = ...  # Precomputed factors
expkN = ...  # Precomputed factors
out = torch.empty_like(x)
buf = torch.empty(...)

# Apply DCT
dct_cuda.dct2_fft2(x, expkM, expkN, out, buf)

# Apply inverse DCT
dct_cuda.idct2_fft2(out, expkM, expkN, result, buf)
```

## Performance Notes

- FFT-based implementation is O(n log n) where n = height × width
- Precomputed exponential factors (`expkM`, `expkN`) should be reused across multiple transformations
- All tensors must be on CUDA device and contiguous

## Related Components

- Used by `src/core/dct2_fft2.py` for density smoothing in placement optimization
- Works with `density_map_cuda` for density-based force calculations
