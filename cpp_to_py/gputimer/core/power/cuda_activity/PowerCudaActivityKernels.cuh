#pragma once

#include "PowerCudaActivityDevice.cuh"

namespace gt {

__global__ void power_snapshot_level_active_kernel(const PowerActivityCudaModel* model,
                                                   PowerActivityScratchView* scratch,
                                                   int level_start,
                                                   int num_level_pins);

__global__ void power_snapshot_level_active_list_kernel(const PowerActivityCudaModel* model,
                                                        PowerActivityScratchView* scratch,
                                                        int level_start,
                                                        int num_level_pins,
                                                        int* active_count,
                                                        int* active_pins);

__global__ void power_seed_pi_kernel(PowerActivityCudaModel* model,
                                     PowerActivityScratchView* scratch);

__global__ void power_seed_clock_active_kernel(PowerActivityCudaModel* model,
                                               PowerActivityScratchView* scratch);

__global__ void power_seed_case_kernel(PowerActivityCudaModel* model,
                                       PowerActivityScratchView* scratch);

__global__ void power_seed_seq_feedback_state_kernel(PowerActivityCudaModel* model,
                                                     PowerActivityScratchView* scratch);

__global__ void power_visit_level_kernel(PowerActivityCudaModel* model,
                                         PowerActivityScratchView* scratch,
                                         int level_start,
                                         int num_level_pins,
                                         bool defer_pending_seq);

__global__ void power_visit_level_serial_kernel(PowerActivityCudaModel* model,
                                                PowerActivityScratchView* scratch,
                                                int level_start,
                                                int num_level_pins,
                                                bool defer_pending_seq);

__global__ void power_visit_active_list_serial_kernel(PowerActivityCudaModel* model,
                                                      PowerActivityScratchView* scratch,
                                                      const int* active_pins,
                                                      const int* active_count,
                                                      bool defer_pending_seq);

__global__ void power_mark_pending_seq_changes_kernel(PowerActivityCudaModel* model,
                                                      PowerActivityScratchView* scratch);

__global__ void power_seed_seq_kernel(PowerActivityCudaModel* model,
                                      PowerActivityScratchView* scratch);

__global__ void power_seed_seq_ordered_kernel(PowerActivityCudaModel* model,
                                              PowerActivityScratchView* scratch);

__global__ void power_seed_seq_id_list_ordered_kernel(PowerActivityCudaModel* model,
                                                      PowerActivityScratchView* scratch,
                                                      const int* seq_ids,
                                                      int num_seq_ids);

__global__ void power_seed_case_level_queue_kernel(PowerActivityCudaModel* model,
                                                   PowerActivityScratchView* scratch,
                                                   PowerActivityQueueView* queue);

__global__ void power_seed_pi_level_queue_kernel(PowerActivityCudaModel* model,
                                                 PowerActivityScratchView* scratch,
                                                 PowerActivityQueueView* queue);

__global__ void power_seed_clock_level_queue_kernel(PowerActivityCudaModel* model,
                                                    PowerActivityScratchView* scratch,
                                                    PowerActivityQueueView* queue);

__global__ void power_seed_roots_level_queue_ordered_kernel(PowerActivityCudaModel* model,
                                                            PowerActivityScratchView* scratch,
                                                            PowerActivityQueueView* queue);

__global__ void power_activity_level_queue_persistent_kernel(PowerActivityCudaModel* model,
                                                             PowerActivityScratchView* scratch,
                                                             PowerActivityQueueView* queue,
                                                             int max_seq_passes);

__global__ void power_activity_level_queue_ordered_kernel(PowerActivityCudaModel* model,
                                                          PowerActivityScratchView* scratch,
                                                          PowerActivityQueueView* queue,
                                                          int max_seq_passes);

__global__ void power_pack_output_kernel(const PowerActivityCudaModel* model,
                                         const PowerActivityScratchView* scratch);

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

__global__ void power_switching_kernel(const PowerActivityCudaModel* model);

__global__ void power_internal_denom_kernel(PowerInternalDenomModel model,
                                            PowerActivityScratchView scratch);

__global__ void power_internal_denom_fast_kernel(PowerInternalDenomModel model,
                                                 PowerActivityScratchView scratch);

__global__ void power_internal_contrib_kernel(PowerInternalContribModel model,
                                              PowerActivityScratchView scratch);

__global__ void power_internal_contrib_fast_kernel(PowerInternalContribModel model,
                                                   PowerActivityScratchView scratch);

__global__ void power_leakage_row_kernel(PowerLeakageRowsModel model,
                                         PowerActivityScratchView scratch);

__global__ void power_leakage_row_fast_kernel(PowerLeakageRowsModel model);

__global__ void power_leakage_summary_kernel(PowerLeakageSummaryModel model);

}  // namespace gt
