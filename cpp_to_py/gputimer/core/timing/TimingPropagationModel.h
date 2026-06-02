#pragma once

#include "gputimer/base.h"

#include <cstdint>
#include <vector>

namespace gt {

class GPULutAllocator;

struct TimingPropagationModel {
    index_type* level_list = nullptr;
    const std::vector<int>* level_list_end_cpu = nullptr;
    index_type* pin_forward_arc_list_end = nullptr;
    index_type* pin_forward_arc_list = nullptr;
    index_type* timing_arc_to_pin_id = nullptr;
    index_type* pin_backward_arc_list_end = nullptr;
    index_type* pin_backward_arc_list = nullptr;
    index_type* timing_arc_from_pin_id = nullptr;
    const uint8_t* arc_types = nullptr;
    int* arc_id2test_id = nullptr;
    float* pinSlew = nullptr;
    float* pinLoad = nullptr;
    float* pinImpulse = nullptr;
    float* pinRootDelay = nullptr;
    float* pinAT = nullptr;
    float* pinRAT = nullptr;
    float* testRelatedAT = nullptr;
    float* testRAT = nullptr;
    float* testConstraint = nullptr;
    float* arcDelay = nullptr;
    int* timing_arc_id_map = nullptr;
    index_type* at_prefix_pin = nullptr;
    index_type* at_prefix_arc = nullptr;
    index_type* at_prefix_attr = nullptr;
    float clock_period = 0.0f;
    GPULutAllocator* d_allocator = nullptr;
    int num_pins = 0;
    bool deterministic = true;
};

struct InferTimingModel {
    index_type* level_list = nullptr;
    const std::vector<int>* level_list_end_cpu = nullptr;
    index_type* pin_forward_arc_list_end = nullptr;
    index_type* pin_forward_arc_list = nullptr;
    index_type* timing_arc_to_pin_id = nullptr;
    index_type* pin_backward_arc_list_end = nullptr;
    index_type* pin_backward_arc_list = nullptr;
    index_type* timing_arc_from_pin_id = nullptr;
    const uint8_t* arc_types = nullptr;
    int* arc_id2test_id = nullptr;
    float* pinSlew = nullptr;
    float* pinAT = nullptr;
    float* pinRAT = nullptr;
    float* arcDelay = nullptr;
    float* testRelatedAT = nullptr;
    float* testRAT = nullptr;
    float* testConstraint = nullptr;
    int* timing_arc_id_map = nullptr;
    index_type* at_prefix_pin = nullptr;
    index_type* at_prefix_arc = nullptr;
    index_type* at_prefix_attr = nullptr;
    float clock_period = 0.0f;
    GPULutAllocator* d_allocator = nullptr;
    const uint8_t* pin_is_clk = nullptr;
    const uint8_t* pin_is_ideal_clk = nullptr;
    const float* pin_clock_fall_edges = nullptr;
    int num_pins = 0;
};

void update_timing_cuda(const TimingPropagationModel& model);
void propagate_infer_timing_impl(const InferTimingModel& model);

}  // namespace gt
