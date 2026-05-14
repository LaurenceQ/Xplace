#pragma once

#include "DmpModel.h"

namespace gt {

__global__ void dmpBackwardKernel(DmpModel* model,
                                  int level_start_offset,
                                  int num_pins_level);
__global__ void dmpTestKernel(DmpModel* model,
                              int level_start_offset,
                              int num_pins_level);
__global__ void dmpDirectNetKernel(DmpModel* model,
                                   const index_type* level_arc_list,
                                   int num_level_arcs);
__global__ void dmpGateKernel(DmpModel* model,
                              const index_type* level_arc_list,
                              int num_level_arcs,
                              unsigned long long* debug_counts);
__global__ void dmpPinWinnerKernel(DmpModel* model,
                                   int level_start_offset,
                                   int num_pins_level);
__global__ void dmpNetWinnerKernel(DmpModel* model,
                                   const index_type* level_arc_list,
                                   int num_level_arcs);

void reset_dmp_root_profile_cuda();
void print_dmp_root_profile_cuda();

void dmp_debug_print_counts(DmpModel* model, const char* label);
void dmp_debug_print_first_level_sample(DmpModel* model,
                                        int level_idx,
                                        index_type level_start_offset);
void dmp_debug_print_parallel_stats(DmpModel* model,
                                    const vector<int>& level_list_end_cpu,
                                    const char* label);

} // namespace gt
