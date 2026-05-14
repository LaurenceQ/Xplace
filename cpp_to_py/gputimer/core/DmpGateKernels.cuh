#pragma once

#include "DmpCeff.h"

namespace gt {

__global__ void propagatePin_dmp(dmp_model* dmp_db, int level_start_offset, int num_pins_level);
__global__ void propagatePinTests_dmp(dmp_model* dmp_db, int level_start_offset, int num_pins_level);
__global__ void propagateArc_dmp(dmp_model* dmp_db, const index_type* level_arc_list, int num_level_arcs);
__global__ void propagateArcDelaySlewAndAT_dmp(dmp_model* dmp_db,
                                               const index_type* level_arc_list,
                                               int num_level_arcs);
__global__ void propagateHybridGateArcDelaySlewAndAT_dmp(dmp_model* dmp_db,
                                                         const index_type* level_arc_list,
                                                         int num_level_arcs);
__global__ void propagateNetArcSlewDelay_dmp(dmp_model* dmp_db,
                                             const index_type* level_arc_list,
                                             int num_level_arcs);
__global__ void propagateHybridNetArcSlewDelayAndAT_dmp(dmp_model* dmp_db,
                                                        const index_type* level_arc_list,
                                                        int num_level_arcs,
                                                        unsigned long long* debug_counts);
__global__ void propagateFusedGateNetDelaySlewAndAT_dmp(dmp_model* dmp_db,
                                                        const index_type* level_arc_list,
                                                        int num_level_arcs,
                                                        unsigned long long* debug_counts);
__global__ void propagateArcAT_dmp(dmp_model* dmp_db,
                                   const index_type* level_arc_list,
                                   int num_level_arcs);
__global__ void finalizeAtWinners_dmp(dmp_model* dmp_db,
                                      int level_start_offset,
                                      int num_pins_level);
__global__ void finalizeSlewWinners_dmp(dmp_model* dmp_db,
                                        int level_start_offset,
                                        int num_pins_level);
__global__ void finalizePinWinners_dmp(dmp_model* dmp_db,
                                       int level_start_offset,
                                       int num_pins_level);
__global__ void finalizeNetDelayWinners_dmp(dmp_model* dmp_db,
                                            const index_type* level_arc_list,
                                            int num_level_arcs);
__global__ void finalizeNetDelayWinnersAndPropagateAT_dmp(dmp_model* dmp_db,
                                                          const index_type* level_arc_list,
                                                          int num_level_arcs);
__global__ void propagateGateNetPair_dmp(dmp_model* dmp_db,
                                         const index_type* gate_arc_list,
                                         const index_type* net_arc_list,
                                         int num_pairs,
                                         unsigned long long* debug_counts);
__global__ void propagateGateNetPairValid_dmp(dmp_model* dmp_db,
                                              const index_type* gate_arc_list,
                                              const index_type* net_arc_list,
                                              const uint8_t* lane_list,
                                              int num_valid_pairs,
                                              unsigned long long* debug_counts);
void reset_dmp_root_profile_cuda();
void print_dmp_root_profile_cuda();

} // namespace gt
