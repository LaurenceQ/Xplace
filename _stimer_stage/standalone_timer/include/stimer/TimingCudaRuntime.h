#pragma once

#include <string>

#include "stimer/TimingFlatDB.h"

namespace stimer {

struct CudaStatus {
  bool ok = false;
  float elapsed_ms = 0.0f;
  std::string message;
};

CudaStatus run_cuda_runtime_smoke_test();
TimingEvalResult evaluate_timing_cuda(const TimingFlatDB& flat,
                                      float* elapsed_ms);

}  // namespace stimer
