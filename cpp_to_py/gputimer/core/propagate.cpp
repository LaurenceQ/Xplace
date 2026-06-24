
#include "GPUTimer.h"
#include "DmpModel.h"
#include "timing/TimingPropagationModel.h"

namespace gt {

void GPUTimer::update_timing() {
    TimingPropagationModel model;
    model.level_list = level_list;
    model.level_list_end_cpu = &level_list_end_cpu;
    model.pin_forward_arc_list_end = pin_forward_arc_list_end;
    model.pin_forward_arc_list = pin_forward_arc_list;
    model.timing_arc_to_pin_id = timing_arc_to_pin_id;
    model.pin_backward_arc_list_end = pin_backward_arc_list_end;
    model.pin_backward_arc_list = pin_backward_arc_list;
    model.timing_arc_from_pin_id = timing_arc_from_pin_id;
    model.arc_types = arc_types;
    model.arc_id2test_id = arc_id2test_id;
    model.pinSlew = pinSlew;
    model.pinLoad = pinLoad;
    model.pinImpulse = pinImpulse;
    model.pinRootDelay = pinRootDelay;
    model.pinAT = pinAT;
    model.pinRAT = pinRAT;
    model.testRelatedAT = testRelatedAT;
    model.testRAT = testRAT;
    model.testConstraint = testConstraint;
    model.arcDelay = arcDelay;
    model.timing_arc_id_map = timing_arc_id_map;
    model.at_prefix_pin = at_prefix_pin;
    model.at_prefix_arc = at_prefix_arc;
    model.at_prefix_attr = at_prefix_attr;
    model.clock_period = clock_period;
    model.d_allocator = d_allocator;
    model.num_pins = num_pins;
    model.deterministic = true;
    update_timing_cuda(model);
}

void GPUTimer::propagate_infer_timing() {
    InferTimingModel model;
    model.level_list = level_list;
    model.level_list_end_cpu = &level_list_end_cpu;
    model.pin_forward_arc_list_end = pin_forward_arc_list_end;
    model.pin_forward_arc_list = pin_forward_arc_list;
    model.timing_arc_to_pin_id = timing_arc_to_pin_id;
    model.pin_backward_arc_list_end = pin_backward_arc_list_end;
    model.pin_backward_arc_list = pin_backward_arc_list;
    model.timing_arc_from_pin_id = timing_arc_from_pin_id;
    model.arc_types = arc_types;
    model.arc_id2test_id = arc_id2test_id;
    model.pinSlew = pinSlew;
    model.pinAT = pinAT;
    model.pinRAT = pinRAT;
    model.arcDelay = arcDelay;
    model.testRelatedAT = testRelatedAT;
    model.testRAT = testRAT;
    model.testConstraint = testConstraint;
    model.timing_arc_id_map = timing_arc_id_map;
    model.at_prefix_pin = at_prefix_pin;
    model.at_prefix_arc = at_prefix_arc;
    model.at_prefix_attr = at_prefix_attr;
    model.clock_period = clock_period;
    model.d_allocator = d_allocator;
    model.pin_is_clk = pin_is_clk;
    model.pin_is_ideal_clk = pin_is_ideal_clk;
    if (h_dmp_db != nullptr) {
        model.pin_clock_ids = h_dmp_db->pin_clock_ids;
        model.clock_fall_edges = h_dmp_db->clock_fall_edges;
        model.clock_waveform_fall_edges = h_dmp_db->clock_waveform_fall_edges;
        model.pin_clock_latency_overrides = h_dmp_db->pin_clock_latency_overrides;
        model.clock_count = h_dmp_db->clock_count;
    }
    model.num_pins = num_pins;
    propagate_infer_timing_impl(model);
}

}  // namespace gt
