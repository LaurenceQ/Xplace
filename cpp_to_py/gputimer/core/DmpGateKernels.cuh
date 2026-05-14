#pragma once

#include "DmpCeff.h"

namespace gt {

__global__ void propagatePinTests_dmp(dmp_model* dmp_db, int level_start_offset, int num_pins_level);
__global__ void propagateNetArcSlewDelay_dmp(dmp_model* dmp_db,
                                             const index_type* level_arc_list,
                                             int num_level_arcs);
__global__ void propagateFusedGateNetDelaySlewAndAT_dmp(dmp_model* dmp_db,
                                                        const index_type* level_arc_list,
                                                        int num_level_arcs,
                                                        unsigned long long* debug_counts);
__global__ void finalizePinWinners_dmp(dmp_model* dmp_db,
                                       int level_start_offset,
                                       int num_pins_level);
__global__ void finalizeNetDelayWinnersAndPropagateAT_dmp(dmp_model* dmp_db,
                                                          const index_type* level_arc_list,
                                                          int num_level_arcs);
void reset_dmp_root_profile_cuda();
void print_dmp_root_profile_cuda();

} // namespace gt
