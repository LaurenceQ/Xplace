#pragma once

#include "gputimer/core/power/common/PowerHostCommon.h"
#include "gputimer/db/GTDatabase.h"

#include <cstdint>
#include <vector>

namespace gt {

void dumpPowerActivityCpuTracePaths(GTDatabase& gtdb,
                                    const char* trace_path_out_env,
                                    int n,
                                    const std::vector<uint8_t>& actual_seed_seen,
                                    const std::vector<int>& pin_level,
                                    const std::vector<int>& pin_to_node,
                                    const std::vector<uint8_t>& is_driver_pin,
                                    const std::vector<std::vector<PowerTraceEdge>>& seq_reverse_edges);

}  // namespace gt
