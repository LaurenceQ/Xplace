#pragma once

#include "gputimer/core/power/common/PowerCudaModel.h"

#include <cstdint>

namespace gt {

extern __device__ bool g_power_allow_clock_activity_override;
extern __device__ float g_power_min_activity_density;
extern __device__ float g_power_min_activity_duty;
extern __device__ bool g_power_disable_activity_slew_cap;
extern __device__ float g_power_seq_clock_limit_rel_tol;
extern __device__ float g_power_seq_pending_min_density;
extern __device__ float g_power_activity_clock_density_cap;
extern __device__ int g_power_direct_ordered_seq_seed;
extern __device__ int g_power_require_known_seq_data;
extern __device__ int g_power_disable_direct_expr;
extern __device__ int g_power_check_direct_expr;
extern __device__ int g_power_direct_expr_mismatch_count;
extern __device__ int g_power_direct_expr_max_vars;
extern __device__ float g_power_direct_expr_density_rel_tol;
extern __device__ float g_power_direct_expr_duty_abs_tol;

__device__ float power_percent_change(float value, float prev);
__device__ float power_clamp_activity_duty(float duty);
__device__ bool power_should_mark_pending_seq(float density);
__device__ float power_max_activity_density_from_slew(int pin,
                                                      const PowerActivityCudaModel* model);
__device__ bool power_seq_density_exceeds_clock_limit(float in_density, float clk_density);

__device__ bool power_set_activity(int pin,
                                   float new_density,
                                   float new_duty,
                                   int new_origin,
                                   bool force,
                                   const PowerActivityCudaModel* model,
                                   PowerActivityScratchView* scratch);
__device__ void power_enqueue_adjacent(int pin,
                                       const PowerActivityCudaModel* model,
                                       PowerActivityScratchView* scratch);
__device__ void power_activate_pin(int pin,
                                   const PowerActivityCudaModel* model,
                                   PowerActivityScratchView* scratch);
__device__ bool power_set_clock_gate_output(int pin,
                                            const PowerActivityCudaModel* model,
                                            PowerActivityScratchView* scratch);
__device__ void power_enqueue_clock_gate_output(int pin,
                                                const PowerActivityCudaModel* model,
                                                PowerActivityScratchView* scratch);

__device__ bool power_eval_expr_bool(int expr_id,
                                     uint64_t bits,
                                     int force_var,
                                     int force_val,
                                     const int* var_pins,
                                     int var_count,
                                     const GpuPowerExprOpHost* ops,
                                     const int* expr_start,
                                     const int* expr_count,
                                     int8_t& value);
__device__ bool power_eval_expr_activity(int expr_id,
                                         const GpuPowerExprOpHost* ops,
                                         const int* expr_start,
                                         const int* expr_count,
                                         const float* pin_density,
                                         const float* pin_duty,
                                         float& out_density,
                                         float& out_duty,
                                         const int* node_port_pin_start = nullptr,
                                         const int* node_port_pin_list = nullptr,
                                         int node_id = -1);
__device__ bool power_expr_has_known_activity_input(int expr_id,
                                                    const GpuPowerExprOpHost* ops,
                                                    const int* expr_start,
                                                    const int* expr_count,
                                                    const int* origin,
                                                    const int* node_port_pin_start = nullptr,
                                                    const int* node_port_pin_list = nullptr,
                                                    int node_id = -1);
__device__ float power_eval_expr_duty(int expr_id,
                                      const GpuPowerExprOpHost* ops,
                                      const int* expr_start,
                                      const int* expr_count,
                                      const float* pin_density,
                                      const float* pin_duty,
                                      const int* node_port_pin_start = nullptr,
                                      const int* node_port_pin_list = nullptr,
                                      int node_id = -1);
__device__ float power_eval_expr_diff_duty(int expr_id,
                                           int diff_pin,
                                           const GpuPowerExprOpHost* ops,
                                           const int* expr_start,
                                           const int* expr_count,
                                           const float* pin_duty,
                                           const int* node_port_pin_start = nullptr,
                                           const int* node_port_pin_list = nullptr,
                                           int node_id = -1);

}  // namespace gt
