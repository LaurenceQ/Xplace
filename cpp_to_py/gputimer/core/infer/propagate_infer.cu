/**
 * propagate_infer.cu - Timing propagation with ML-predicted delays
 *
 * After read_infer() loads ML-predicted delays into arcDelay, this module
 * propagates them through the timing graph to compute:
 * 1. Forward pass (AT): Arrival times + TEST constraint RAT setup
 * 2. Backward pass (RAT): Required arrival times propagated backward
 * 3. Slack: RAT - AT (computed on-demand)
 *
 * ML delays pre-loaded in arcDelay, but we still query LUT for constraint delays
 * to set RAT at timing endpoints (POs and timing sinks).
 *
 * CRITICAL: Net Arc Transition Validity
 * ====================================
 * For net arcs, input and output transitions must MATCH:
 *   - rise → rise (valid, indices 0 and 4)
 *   - fall → fall (valid, indices 3 and 7)
 *   - rise → fall (INVALID, indices 1 and 5)
 *   - fall → rise (INVALID, indices 2 and 6)
 *
 * Invalid transitions are NaN in arcDelay and must be skipped by the kernel.
 * This is enforced by the (irf != orf) check in propagateInferAT().
 */

#include "gputimer/core/gputiming.h"
#include "gputimer/core/utils.cuh"
#include "gputimer/core/timing/TimingPropagationModel.h"
#include <vector>

using std::vector;

namespace gt {

namespace {

__device__ __forceinline__ bool infer_clock_id_valid(const InferTimingModel* model,
                                                     uint16_t clock_id) {
    return clock_id != 65535u && clock_id < static_cast<uint16_t>(model->clock_count);
}

__device__ __forceinline__ float infer_pin_clock_fall_edge(const InferTimingModel* model,
                                                           int pin_id) {
    if (model->pin_clock_ids == nullptr || pin_id < 0 || pin_id >= model->num_pins) {
        return nanf("");
    }
    const uint16_t clock_id = model->pin_clock_ids[pin_id];
    const float override =
        model->pin_clock_latency_overrides != nullptr
            ? model->pin_clock_latency_overrides[pin_id]
            : nanf("");
    if (isfinite(override)) {
        if (infer_clock_id_valid(model, clock_id) &&
            model->clock_waveform_fall_edges != nullptr) {
            const float waveform = model->clock_waveform_fall_edges[clock_id];
            if (isfinite(waveform)) return waveform + override;
        }
        return override;
    }
    if (infer_clock_id_valid(model, clock_id) && model->clock_fall_edges != nullptr) {
        const float edge = model->clock_fall_edges[clock_id];
        if (isfinite(edge)) return edge;
    }
    return nanf("");
}

}  // namespace

__device__ void propagateInferAT(const InferTimingModel* model,
                                 index_type arc_id,
                                 index_type from_pin_id,
                                 index_type to_pin_id) {
    /**
     * Forward pass: Propagate AT using ML-predicted arcDelay
     * AT[to_pin, tel_rf] = min(AT[from_pin, fel_rf] + arcDelay[arc, i])
     *
     * NOTE: For net arcs, arcDelay only has valid values at indices [0,3,4,7]
     * corresponding to [er,ef,lr,lf]. Indices [1,2,5,6] are NaN and should be skipped.
     * Skip entire corner if arcDelay is NaN (delay not provided by ML model).
     * If this is an ideal clock pin, skip propagation to it.
     */
    if (model->pin_is_ideal_clk != nullptr && model->pin_is_ideal_clk[to_pin_id]) return;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;

    int el = i >> 2;
    int fel_rf = i >> 1;
    int tel_rf = ((i & 0b100) >> 1) + (i & 1);
    int irf = fel_rf & 1;
    int orf = tel_rf & 1;
    int timing_id = model->timing_arc_id_map[arc_id * 2 + el];
    float* pinAt = model->pinAT;
    float* arcDelay = model->arcDelay;
    GPULutAllocator* d_allocator = model->d_allocator;
    if (isnan(pinAt[from_pin_id * NUM_ATTR + fel_rf]) || !(d_allocator->is_transition_defined(timing_id, irf, orf))) return;

    // For net arcs, indices where irf != orf (rise→fall, fall→rise) are stored as NaN
    // in arcDelay because those transitions are physically invalid on a wire.
    // This NaN check handles both net arcs (NaN at invalid indices) and cell arcs
    // (all 8 indices populated, NaN only if timing arc is undefined).
    // Do NOT add an explicit irf!=orf check here — it would break cell arcs like
    // inverters that legitimately have rise-to-fall and fall-to-rise transitions.
    float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
    if (isnan(delay)) return;

    float at = pinAt[from_pin_id * NUM_ATTR + fel_rf] + delay;
    if (isnan(pinAt[to_pin_id * NUM_ATTR + tel_rf]) ||
        ((pinAt[to_pin_id * NUM_ATTR + tel_rf] > at) ^ el)) {
        atomicExch(&pinAt[to_pin_id * NUM_ATTR + tel_rf], at);
        model->at_prefix_pin[to_pin_id * NUM_ATTR + tel_rf] = from_pin_id;
        model->at_prefix_arc[to_pin_id * NUM_ATTR + tel_rf] = arc_id;
        model->at_prefix_attr[to_pin_id * NUM_ATTR + tel_rf] = fel_rf;
    }
}

__device__ void propagateInferTest(const InferTimingModel* model,
                                   index_type arc_id,
                                   index_type test_id,
                                   index_type from_pin_id,
                                   index_type to_pin_id) {
    /**
     * Forward pass: Compute RAT at test constraint sinks using library delays
     * Even with ML predictions for cell delays, we need LUT for constraint info
     */
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;

    if (i < NUM_ATTR) {
        const int el = i >> 1;
        const int rf = i & 1;
        int* timing_arc_id_map = model->timing_arc_id_map;
        float* pinSlew = model->pinSlew;
        float* pinAt = model->pinAT;
        GPULutAllocator* d_allocator = model->d_allocator;

        if ((timing_arc_id_map[arc_id * 2 + el] == -1) || isnan(pinSlew[to_pin_id * NUM_ATTR + i])) return;

        int fel = el ^ 1;
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        int frf = d_allocator->timing_is_rising_edge_triggered(timing_id) ? 0 : 1;

        if (frf && !d_allocator->timing_is_falling_edge_triggered(timing_id)) return;

        const int fel_rf = (fel << 1) + frf;
        const bool ideal_clock_pin = model->pin_is_ideal_clk != nullptr && model->pin_is_ideal_clk[from_pin_id];
        if (!ideal_clock_pin &&
            (isnan(pinAt[from_pin_id * NUM_ATTR + fel_rf]) ||
             isnan(pinSlew[from_pin_id * NUM_ATTR + fel_rf]))) return;

        if (el == 0) {
            model->testRelatedAT[test_id * NUM_ATTR + i] =
                ideal_clock_pin ? 0.0f : pinAt[from_pin_id * NUM_ATTR + fel_rf];
        } else {
            model->testRelatedAT[test_id * NUM_ATTR + i] =
                ideal_clock_pin ? model->clock_period : (pinAt[from_pin_id * NUM_ATTR + fel_rf] + model->clock_period);
        }

        float sr = ideal_clock_pin ? 0.0f : pinSlew[from_pin_id * NUM_ATTR + fel_rf];
        float sc = pinSlew[to_pin_id * NUM_ATTR + i];
        model->testConstraint[test_id * NUM_ATTR + i] = d_allocator->query(timing_id, frf, rf, sr, sc, 2);

        if (!isnan(model->testConstraint[test_id * NUM_ATTR + i]) && !isnan(model->testRelatedAT[test_id * NUM_ATTR + i])) {
            float* pinRat = model->pinRAT;
            if (el == 0) {
                pinRat[to_pin_id * NUM_ATTR + i] = model->testRelatedAT[test_id * NUM_ATTR + i] + model->testConstraint[test_id * NUM_ATTR + i];
            } else {
                pinRat[to_pin_id * NUM_ATTR + i] = model->testRelatedAT[test_id * NUM_ATTR + i] - model->testConstraint[test_id * NUM_ATTR + i];
            }
            model->testRAT[test_id * NUM_ATTR + i] = pinRat[to_pin_id * NUM_ATTR + i];
        }
    }
}

__device__ void propagateInferRAT(const InferTimingModel* model,
                                  index_type arc_id,
                                  int arc_type,
                                  index_type from_pin_id,
                                  index_type to_pin_id,
                                  float *from_rats) {
    /**
     * Backward pass: Propagate RAT using ML-predicted arcDelay + library constraints
     * For non-constraint arcs: RAT[from] = RAT[to] - delay
     * For constraint arcs: RAT[from] = AT[from] ± slack (setup: +, hold: -)
     */
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    float* pinRat = model->pinRAT;
    float* arcDelay = model->arcDelay;

    if ((arc_type == 0) && (i < NUM_ATTR)) {
        // Net arcs: simple delay-based RAT
        const int el_rf_rf = (i << 1) + (i & 1);
        const int el = i >> 1;
        if (isnan(pinRat[to_pin_id * NUM_ATTR + i]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf])) return;
        float delay = arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf];
        float rat = pinRat[to_pin_id * NUM_ATTR + i] - delay;
        if (isnan(pinRat[from_pin_id * NUM_ATTR + i]) || ((pinRat[from_pin_id * NUM_ATTR + i] < rat) ^ el)) {
            atomicExch(&pinRat[from_pin_id * NUM_ATTR + i], rat);
        }
    } else if (arc_type == 1) {
        // Cell arcs: check if constraint, use different RAT formula
        int el = i >> 2;
        int tel_rf = ((i & 0b100) >> 1) + (i & 1);
        int* timing_arc_id_map = model->timing_arc_id_map;
        if (timing_arc_id_map[arc_id * 2 + el] == -1) return;
        int timing_id = timing_arc_id_map[arc_id * 2 + el];

        if (model->d_allocator->timing_is_constraint(timing_id)) return;

        // Non-constraint cell arc: RAT[from] = RAT[to] - delay
        if (isnan(pinRat[to_pin_id * NUM_ATTR + tel_rf]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + i])) return;
        float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
        float rat = pinRat[to_pin_id * NUM_ATTR + tel_rf] - delay;
        from_rats[threadIdx.x] = rat;
    }
}

__global__ void initIdealClockPins(InferTimingModel* model) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float* pinAt = model->pinAT;
    float* pinSlew = model->pinSlew;
    const uint8_t* pin_is_ideal_clk = model->pin_is_ideal_clk;
    const int num_pins = model->num_pins;
    const float half_period = model->clock_period / 2.0f;
    if (idx < num_pins && pin_is_ideal_clk[idx]) {
        // Rise edges: AT = 0, Fall edges: AT = T/2
        pinAt[idx * NUM_ATTR + 0] = 0.0f;          // early rise
        pinAt[idx * NUM_ATTR + 1] = half_period;    // early fall
        pinAt[idx * NUM_ATTR + 2] = 0.0f;           // late rise
        pinAt[idx * NUM_ATTR + 3] = half_period;    // late fall
        for (int i = 0; i < NUM_ATTR; i++) {
            pinSlew[idx * NUM_ATTR + i] = 0.0f;
        }
    }
}

__global__ void initClockSourceFallAT(InferTimingModel* model, int num_level0_pins) {
    // For propagated clocks: seed fall-edge AT at clock source pins (level 0 only).
    // Downstream clock pins get their fall AT from propagation (T/2 + accumulated delay).
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_level0_pins) {
        index_type pin_id = model->level_list[idx];
        float* pinAt = model->pinAT;
        const uint8_t* pin_is_clk = model->pin_is_clk;
        const uint8_t* pin_is_ideal_clk = model->pin_is_ideal_clk;
        if (pin_is_clk[pin_id] && (pin_is_ideal_clk == nullptr || !pin_is_ideal_clk[pin_id])) {
            float fall_edge = model->clock_period / 2.0f;
            const float sdc_fall_edge = infer_pin_clock_fall_edge(model, pin_id);
            if (isfinite(sdc_fall_edge)) {
                fall_edge = sdc_fall_edge;
            }
            if (isnan(pinAt[pin_id * NUM_ATTR + 1])) {
                pinAt[pin_id * NUM_ATTR + 1] = fall_edge;  // early fall
            }
            if (isnan(pinAt[pin_id * NUM_ATTR + 3])) {
                pinAt[pin_id * NUM_ATTR + 3] = fall_edge;  // late fall
            }
        }
    }
}

__global__ void propagatePinInferAT(InferTimingModel* model,
                                    index_type level_start_offset,
                                    int num_pins_level) {
    /**
     * Forward pass kernel: Process all pins at current level
     * 8 threads per pin, each thread computes one corner (idx & 0b111)
     * 1. Propagate AT using ML-predicted delays
     * 2. For test constraint arcs, compute RAT at sinks using library constraint info
     */
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_idx = idx >> 3;
    if (pin_idx < num_pins_level) {
        index_type pin_id = model->level_list[level_start_offset + pin_idx];

        // Iterate fanin arcs using CSR format
        for (index_type i = model->pin_backward_arc_list_end[pin_id];
             i < model->pin_backward_arc_list_end[pin_id + 1];
            i++) {
            index_type arc_id = model->pin_backward_arc_list[i];
            index_type from_pin_id = model->timing_arc_from_pin_id[arc_id];

            // Each thread computes one corner via idx & 0b111
            propagateInferAT(model, arc_id, from_pin_id, pin_id);

            // For test constraint arcs, also compute RAT using library constraint info
            int test_id = model->arc_id2test_id[arc_id];
            if (model->clock_period > 0 && test_id != -1) {
                propagateInferTest(model, arc_id, test_id, from_pin_id, pin_id);
            }
        }
    }
}

__global__ void propagatePinInferRAT(InferTimingModel* model,
                                     index_type level_start_offset,
                                     int num_pins_level) {
    /**
     * Backward pass kernel: Process all pins at current level (reversed)
     * Uses shared memory to batch RAT updates with constraint-aware logic
     */
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_idx = idx >> 3;
    extern __shared__ float from_rats[];

    if (pin_idx < num_pins_level) {
        index_type from_pin_id = model->level_list[level_start_offset + pin_idx];
        for (index_type i = model->pin_forward_arc_list_end[from_pin_id]; i < model->pin_forward_arc_list_end[from_pin_id + 1]; i++) {
            index_type arc_id = model->pin_forward_arc_list[i];
            index_type to_pin_id = model->timing_arc_to_pin_id[arc_id];
            int arc_type = model->arc_types[arc_id];

            // Initialize shared memory per thread group
            if ((threadIdx.x % (2 * NUM_ATTR)) == 0) {
                for (int i = threadIdx.x; i < threadIdx.x + 2 * NUM_ATTR; i++) from_rats[i] = nanf("");
            }
            __syncthreads();

            // Fill shared memory with RAT values (net arcs use atomicExch directly)
            propagateInferRAT(model, arc_id, arc_type, from_pin_id, to_pin_id, from_rats);

            __syncthreads();

            // Batch update pinRat from shared memory with constraint-aware conditions
            if ((threadIdx.x % (2 * NUM_ATTR)) == 0) {
                for (int ti = threadIdx.x; ti < threadIdx.x + 2 * NUM_ATTR; ti++) {
                    const int i = ti & 0b111;
                    if (isnan(from_rats[ti])) continue;

                    int el = i >> 2;
                    int fel_rf = i >> 1;
                    float rat = from_rats[ti];

                    if (isnan(model->pinRAT[from_pin_id * NUM_ATTR + fel_rf]) || ((model->pinRAT[from_pin_id * NUM_ATTR + fel_rf] < rat) ^ el)) {
                        atomicExch(&model->pinRAT[from_pin_id * NUM_ATTR + fel_rf], rat);
                    }
                }
            }
        }
    }
}
#define CUDA_CHECK(msg) do { \
    cudaDeviceSynchronize(); \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) \
        printf("[prop_infer] CUDA error at %s (line %d): %s\n", msg, __LINE__, cudaGetErrorString(_e)); \
} while(0)

void propagate_infer_timing_impl(const InferTimingModel& model) {
    const vector<int>& level_list_end_cpu = *model.level_list_end_cpu;
    const int num_pins = model.num_pins;
    /**
     * CUDA-side implementation: Launches kernels for ML-predicted timing propagation
     * Forward pass: AT propagation + test constraint RAT setup
     * Skip level 0 (inputs) and last level (outputs)
     */
    CUDA_CHECK("entry");

    InferTimingModel* d_model = nullptr;
    cudaMalloc(&d_model, sizeof(InferTimingModel));
    cudaMemcpy(d_model, &model, sizeof(InferTimingModel), cudaMemcpyHostToDevice);

    initIdealClockPins<<<BLOCK_NUMBER(num_pins), BLOCK_SIZE>>>(d_model);
    cudaDeviceSynchronize();

    if (model.clock_period > 0) {
        int num_level0 = level_list_end_cpu[1] - level_list_end_cpu[0];
        initClockSourceFallAT<<<BLOCK_NUMBER(num_level0), BLOCK_SIZE>>>(d_model, num_level0);
        cudaDeviceSynchronize();
    }

    for (int i = 1; i < (int)level_list_end_cpu.size() - 1; i++) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        int level_start_offset = level_list_end_cpu[i];

        propagatePinInferAT<<<BLOCK_NUMBER(num_pins_level * 2 * NUM_ATTR), BLOCK_SIZE>>>(
            d_model, level_start_offset, num_pins_level);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    /**
     * Backward pass: RAT propagation
     * Skip last 2 levels (outputs)
     */
    for (int i = (int)level_list_end_cpu.size() - 3; i >= 0; i--) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        int level_start_offset = level_list_end_cpu[i];

        propagatePinInferRAT<<<BLOCK_NUMBER(num_pins_level * 2 * NUM_ATTR), BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d_model, level_start_offset, num_pins_level);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
    cudaFree(d_model);
}

}  // namespace gt
