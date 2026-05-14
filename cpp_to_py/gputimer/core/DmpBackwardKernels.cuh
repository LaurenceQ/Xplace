#pragma once

#include "DmpCeff.h"

namespace gt {

__global__ void propagatePinBack_dmp(dmp_model* dmp_db, int level_start_offset, int num_pins_level);

} // namespace gt
