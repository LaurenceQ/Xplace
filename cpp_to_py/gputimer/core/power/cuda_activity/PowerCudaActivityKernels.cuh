#pragma once

#include "PowerCudaActivityDevice.cuh"

namespace gt {

__global__ void power_snapshot_level_active_kernel(const PowerActivityDevice* model,
                                                   PowerActivityPropDevice* scratch,
                                                   int level_start,
                                                   int num_level_pins);

__global__ void power_snapshot_level_active_list_kernel(const PowerActivityDevice* model,
                                                        PowerActivityPropDevice* scratch,
                                                        int level_start,
                                                        int num_level_pins,
                                                        int* active_count,
                                                        int* active_pins);

__global__ void power_seed_pi_kernel(PowerActivityDevice* model,
                                     PowerActivityPropDevice* scratch);

__global__ void power_seed_clock_active_kernel(PowerActivityDevice* model,
                                               PowerActivityPropDevice* scratch);

__global__ void power_seed_case_kernel(PowerActivityDevice* model,
                                       PowerActivityPropDevice* scratch);

__global__ void power_seed_seq_feedback_state_kernel(PowerActivityDevice* model,
                                                     PowerActivityPropDevice* scratch);

__global__ void power_visit_level_kernel(PowerActivityDevice* model,
                                         PowerActivityPropDevice* scratch,
                                         int level_start,
                                         int num_level_pins,
                                         bool defer_pending_seq);

__global__ void power_visit_level_serial_kernel(PowerActivityDevice* model,
                                                PowerActivityPropDevice* scratch,
                                                int level_start,
                                                int num_level_pins,
                                                bool defer_pending_seq);

__global__ void power_visit_active_list_serial_kernel(PowerActivityDevice* model,
                                                      PowerActivityPropDevice* scratch,
                                                      const int* active_pins,
                                                      const int* active_count,
                                                      bool defer_pending_seq);

__global__ void power_mark_pending_seq_changes_kernel(PowerActivityDevice* model,
                                                      PowerActivityPropDevice* scratch);

__global__ void power_seed_seq_kernel(PowerActivityDevice* model,
                                      PowerActivityPropDevice* scratch);

__global__ void power_seed_seq_ordered_kernel(PowerActivityDevice* model,
                                              PowerActivityPropDevice* scratch);

__global__ void power_seed_seq_id_list_ordered_kernel(PowerActivityDevice* model,
                                                      PowerActivityPropDevice* scratch,
                                                      const int* seq_ids,
                                                      int num_seq_ids);

__global__ void power_seed_case_level_queue_kernel(PowerActivityDevice* model,
                                                   PowerActivityPropDevice* scratch,
                                                   PowerActivityLevelQueueDevice* queue);

__global__ void power_seed_pi_level_queue_kernel(PowerActivityDevice* model,
                                                 PowerActivityPropDevice* scratch,
                                                 PowerActivityLevelQueueDevice* queue);

__global__ void power_seed_clock_level_queue_kernel(PowerActivityDevice* model,
                                                    PowerActivityPropDevice* scratch,
                                                    PowerActivityLevelQueueDevice* queue);

__global__ void power_seed_roots_level_queue_ordered_kernel(PowerActivityDevice* model,
                                                            PowerActivityPropDevice* scratch,
                                                            PowerActivityLevelQueueDevice* queue);

__global__ void power_activity_level_queue_persistent_kernel(PowerActivityDevice* model,
                                                             PowerActivityPropDevice* scratch,
                                                             PowerActivityLevelQueueDevice* queue,
                                                             int max_seq_passes);

__global__ void power_activity_level_queue_ordered_kernel(PowerActivityDevice* model,
                                                          PowerActivityPropDevice* scratch,
                                                          PowerActivityLevelQueueDevice* queue,
                                                          int max_seq_passes);

__global__ void power_pack_output_kernel(const PowerActivityDevice* model,
                                         const PowerActivityPropDevice* scratch);

__global__ void power_copy_precomputed_activity_output_kernel(int n,
                                                              const float* activity,
                                                              float* out,
                                                              int out_activity_fields);

__global__ void power_unpack_precomputed_activity_kernel(int n,
                                                         const float* activity,
                                                         float* density,
                                                         float* duty,
                                                         uint8_t* origin);

__global__ void power_unpack_activity_density_duty_kernel(int n,
                                                          const float* activity,
                                                          float* density,
                                                          float* duty);

__global__ void power_switching_kernel(const PowerActivityDevice* model);

__global__ void power_internal_denom_kernel(PowerInternalDenomDevice model,
                                            PowerActivityPropDevice scratch);

__global__ void power_internal_denom_fast_kernel(PowerInternalDenomDevice model,
                                                 PowerActivityPropDevice scratch);

__global__ void power_internal_contrib_kernel(PowerInternalInstDevice model,
                                              PowerActivityPropDevice scratch);

__global__ void power_internal_contrib_fast_kernel(PowerInternalInstDevice model,
                                                   PowerActivityPropDevice scratch);

__global__ void power_leakage_row_kernel(PowerLeakageCondDevice model,
                                         PowerActivityPropDevice scratch);

__global__ void power_leakage_row_fast_kernel(PowerLeakageCondDevice model);

__global__ void power_leakage_summary_kernel(PowerLeakageInstDevice model);

}  // namespace gt
