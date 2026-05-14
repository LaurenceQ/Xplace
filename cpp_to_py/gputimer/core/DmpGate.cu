#include "DmpGateKernels.cuh"
#include "DmpCudaUtils.cuh"
#include "DmpWaveform.cuh"
#include "gputiming.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

namespace gt {

#include "dmp_gate/DmpGateCommon.inc.cuh"
#include "dmp_gate/DmpGateCellModel.inc.cuh"
#include "dmp_gate/DmpGateRootSolve.inc.cuh"
#include "dmp_gate/DmpGateDrivingCell.inc.cuh"
#include "dmp_gate/DmpGatePropagation.inc.cuh"
#include "dmp_gate/DmpGateFusedFallback.inc.cuh"
#include "dmp_gate/DmpGateHost.inc.cuh"

} // namespace gt
