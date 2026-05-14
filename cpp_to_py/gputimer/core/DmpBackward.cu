#include "DmpKernels.cuh"
#include "DmpCudaUtils.cuh"

#include <cmath>

namespace gt {

__device__ __forceinline__ bool dmpBackwardIsIdealClockTimingArc(const DmpModel* dmp_db,
                                                                 int timing_id,
                                                                 int from_pin_id) {
    return dmp_db->ideal_clock &&
           dmp_db->pin_is_clk != nullptr &&
           dmp_db->pin_is_clk[from_pin_id] &&
           timing_id >= 0;
}

__device__ __forceinline__ float dmpBackwardIdealClockEdgeTime(const DmpModel* dmp_db,
                                                               int timing_id) {
    if (timing_id >= 0 &&
        dmp_db->d_allocator->d_is_falling_edge_triggered[timing_id] &&
        !dmp_db->d_allocator->d_is_rising_edge_triggered[timing_id]) {
        return 0.5f * dmp_db->clock_period;
    }
    return 0.0f;
}

__device__ void DmpModel::updatePinRat(int arc_id, float *from_rats){
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    for (int ti = 0; ti < DMP_PIN_GROUP_SIZE; ti++) {
        const int i = ti & 0b111;
        if (isnan(from_rats[ti])) continue;
        int el = i >> 2;
        int fel_rf = i >> 1;
        float rat = from_rats[ti];
        if (isnan(pinRat[from_pin_id * NUM_ATTR + fel_rf]) || ((pinRat[from_pin_id * NUM_ATTR + fel_rf] < rat) ^ el)) {
            atomicExch(&pinRat[from_pin_id * NUM_ATTR + fel_rf], rat);
        }
    }
}
__device__ void DmpModel::propagateRAT(int arc_id, float *from_rats){
    const int i = threadIdx.x & (DMP_PIN_GROUP_SIZE - 1);
    int arc_type = arc_types[arc_id];
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    if ((arc_type == 0) && (i < NUM_ATTR)) {
        const int el_rf_rf = (i << 1) + (i & 1);
        const int el = i >> 1;
        if (isnan(pinRat[to_pin_id * NUM_ATTR + i]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf])) return;
        float delay = arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf];
        float rat = pinRat[to_pin_id * NUM_ATTR + i] - delay;                                                  // rat_f - delay, at_f - delay.
        if (isnan(pinRat[from_pin_id * NUM_ATTR + i]) || ((pinRat[from_pin_id * NUM_ATTR + i] < rat) ^ el)) {  // early(hold up): max rat
            atomicExch(&pinRat[from_pin_id * NUM_ATTR + i], rat);
        }
    } else if (arc_type == 1) {
        int el = i >> 2;
        int tel_rf = ((i & 0b100) >> 1) + (i & 1);
        if (timing_arc_id_map[arc_id * 2 + el] == -1) return;
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        if (d_allocator->d_is_constraint[timing_id]) return;
        if (isnan(pinRat[to_pin_id * NUM_ATTR + tel_rf]) || isnan(arcDelay[arc_id * 2 * NUM_ATTR + i])) return;
        float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
        float rat = pinRat[to_pin_id * NUM_ATTR + tel_rf] - delay;
        from_rats[i] = rat;
    }    
}
__device__ void DmpModel::propagatePinBack(int level_idx, float *from_rats){
        index_type from_pin_id = level_list[level_idx];
        const int lane = threadIdx.x & (DMP_PIN_GROUP_SIZE - 1);
        const int warp_lane = threadIdx.x & 31;
        const unsigned group_mask = 0xffu << (warp_lane & ~(DMP_PIN_GROUP_SIZE - 1));
        float *group_rats = from_rats + (threadIdx.x & ~(DMP_PIN_GROUP_SIZE - 1));
        for (index_type i = pin_forward_arc_list_end[from_pin_id]; i < pin_forward_arc_list_end[from_pin_id + 1]; i++) {
            index_type arc_id = pin_forward_arc_list[i];
            group_rats[lane] = nanf("");
            __syncwarp(group_mask);

            propagateRAT(arc_id, group_rats);

            __syncwarp(group_mask);
            if (lane == 0) {
                updatePinRat(arc_id, group_rats);

            }
            __syncwarp(group_mask);
        }
}
__global__ void dmpBackwardKernel(DmpModel* dmp_db, int level_start_offset, int num_pins_level){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_idx = idx >> 3;
    extern __shared__ float from_rats[];

    if (pin_idx < num_pins_level) {
        dmp_db -> propagatePinBack(level_start_offset + pin_idx, from_rats);

    }
}


} // namespace gt
