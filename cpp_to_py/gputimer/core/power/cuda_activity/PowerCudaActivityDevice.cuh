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

__device__ __forceinline__ uint32_t power_activity_flag_mask(int index) {
    return 1u << (index & 31);
}

__device__ __forceinline__ uint32_t& power_activity_flag_word(uint32_t* flags, int index) {
    return flags[index >> 5];
}

__device__ __forceinline__ const uint32_t& power_activity_flag_word(const uint32_t* flags, int index) {
    return flags[index >> 5];
}

__device__ __forceinline__ bool power_activity_flag_test(const uint32_t* flags, int index) {
    if (!flags || index < 0) return false;
    return (power_activity_flag_word(flags, index) & power_activity_flag_mask(index)) != 0;
}

__device__ __forceinline__ bool power_activity_flag_atomic_test_and_set(uint32_t* flags, int index) {
    if (!flags || index < 0) return false;
    const uint32_t mask = power_activity_flag_mask(index);
    const uint32_t old = atomicOr(&power_activity_flag_word(flags, index), mask);
    return (old & mask) != 0;
}

__device__ __forceinline__ bool power_activity_flag_atomic_test_and_clear(uint32_t* flags, int index) {
    if (!flags || index < 0) return false;
    const uint32_t mask = power_activity_flag_mask(index);
    const uint32_t old = atomicAnd(&power_activity_flag_word(flags, index), ~mask);
    return (old & mask) != 0;
}

__device__ __forceinline__ void power_activity_flag_clear(uint32_t* flags, int index) {
    if (!flags || index < 0) return;
    power_activity_flag_word(flags, index) &= ~power_activity_flag_mask(index);
}

struct PowerActivityValue {
    float density = 0.0f;
    float duty = 0.0f;

    PowerActivityValue() = default;
    __host__ __device__ __forceinline__ PowerActivityValue(float density_, float duty_)
        : density(density_), duty(duty_) {}
};

struct PowerActivityOps {
    const PowerActivityDevice* model = nullptr;
    PowerActivityPropDevice* scratch = nullptr;

    __device__ __forceinline__ PowerActivityOps(const PowerActivityDevice* model_,
                                                PowerActivityPropDevice* scratch_)
        : model(model_), scratch(scratch_) {}

    __device__ static float percentChange(float value, float prev);
    __device__ static float clampActivityDuty(float duty);
    __device__ static bool shouldMarkPendingSeq(float density);
    __device__ static bool seqDensityExceedsClockLimit(float in_density, float clk_density);
    __device__ float maxActivityDensityFromSlew(int pin) const;
    __device__ bool setActivity(int pin,
                                float new_density,
                                float new_duty,
                                int new_origin,
                                bool force) const;
    __device__ void enqueueAdjacent(int pin) const;
    __device__ void activatePin(int pin) const;
    __device__ bool setClockGateOutput(int pin) const;
    __device__ void enqueueClockGateOutput(int pin) const;
    __device__ bool processLevelPin(int pin, bool defer_pending_seq) const;
    __device__ void seedSeqActivity(int seq_id, bool direct_ordered) const;
};

struct PowerLevelQueueOps : PowerActivityOps {
    PowerActivityLevelQueueDevice* queue = nullptr;

    __device__ __forceinline__ PowerLevelQueueOps(const PowerActivityDevice* model_,
                                                  PowerActivityPropDevice* scratch_,
                                                  PowerActivityLevelQueueDevice* queue_)
        : PowerActivityOps(model_, scratch_), queue(queue_) {}

    __device__ bool processPinFrontier(int pin) const;
    __device__ void enqueuePin(int pin) const;
    __device__ void enqueueAdjacent(int pin) const;
    __device__ void enqueueClockGateOutput(int pin) const;
    __device__ void enqueueMissingFuncOutputs(int pin) const;
    __device__ void seedFrontierSeq(int seq_id) const;
};

struct PowerExprEval {
    const GpuPowerExprOpHost* ops = nullptr;
    const int* expr_start = nullptr;
    const int* expr_count = nullptr;
    const float* pin_density = nullptr;
    const float* pin_duty = nullptr;
    const int* node_port_pin_start = nullptr;
    const int* node_port_pin_list = nullptr;
    int node_id = -1;

    PowerExprEval() = default;
    __host__ __device__ __forceinline__ PowerExprEval(const GpuPowerExprOpHost* ops_,
                                                      const int* expr_start_,
                                                      const int* expr_count_,
                                                      const float* pin_density_,
                                                      const float* pin_duty_,
                                                      const int* node_port_pin_start_,
                                                      const int* node_port_pin_list_,
                                                      int node_id_)
        : ops(ops_),
          expr_start(expr_start_),
          expr_count(expr_count_),
          pin_density(pin_density_),
          pin_duty(pin_duty_),
          node_port_pin_start(node_port_pin_start_),
          node_port_pin_list(node_port_pin_list_),
          node_id(node_id_) {}
    __host__ __device__ __forceinline__ PowerExprEval withDensity(const float* pin_density_) const {
        return PowerExprEval(ops, expr_start, expr_count, pin_density_, pin_duty,
                             node_port_pin_start, node_port_pin_list, node_id);
    }
    __host__ __device__ __forceinline__ PowerExprEval withNode(int node_id_) const {
        return PowerExprEval(ops, expr_start, expr_count, pin_density, pin_duty,
                             node_port_pin_start, node_port_pin_list, node_id_);
    }

    __device__ int resolvePinArg(int arg) const;
    __device__ bool evalBool(int expr_id,
                             uint64_t bits,
                             int force_var,
                             int force_val,
                             const int* var_pins,
                             int var_count,
                             int8_t& value) const;
    __device__ bool activity(int expr_id,
                             float& out_density,
                             float& out_duty) const;
    __device__ bool hasKnownActivityInput(int expr_id, const uint8_t* origin) const;
    __device__ float duty(int expr_id) const;
    __device__ float diffDuty(int expr_id, int diff_pin) const;
};

}  // namespace gt
