#include "DmpCudaUtils.cuh"
#include "gputiming.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace gt {

#include "dmp_gate/DmpGateCommon.inc.cuh"
#include "dmp_gate/DmpGateCellModel.inc.cuh"
#include "dmp_gate/DmpGatePropagation.inc.cuh"
#include "dmp_gate/DmpGateDirect.inc.cuh"

} // namespace gt
