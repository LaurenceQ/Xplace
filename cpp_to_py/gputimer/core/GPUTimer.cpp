


#include "common/common.h"
#include "common/db/Database.h"
#include "io_parser/gp/GPDatabase.h"
#include "gputimer/db/GTDatabase.h"
#include "GPUTimer.h"

#include <cfloat>

namespace gt {

GPUTimer::GPUTimer(std::shared_ptr<GTDatabase> gtdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_)
    : gtdb(*gtdb_),
      timing_raw_db(*timing_raw_db_),
      x(timing_raw_db.x.data_ptr<float>()),
      y(timing_raw_db.y.data_ptr<float>()),
      init_x(timing_raw_db.init_x.data_ptr<float>()),
      init_y(timing_raw_db.init_y.data_ptr<float>()),
      node_size_x(timing_raw_db.node_size_x.data_ptr<float>()),
      node_size_y(timing_raw_db.node_size_y.data_ptr<float>()),
      pin_offset_x(timing_raw_db.pin_offset_x.data_ptr<float>()),
      pin_offset_y(timing_raw_db.pin_offset_y.data_ptr<float>()),
      // GPU pin attributes array
      pinSlew(timing_raw_db.pinSlew.data_ptr<float>()),
      pinLoad(timing_raw_db.pinLoad.data_ptr<float>()),
      pinRAT(timing_raw_db.pinRAT.data_ptr<float>()),
      pinAT(timing_raw_db.pinAT.data_ptr<float>()),
      pinImpulse(timing_raw_db.pinImpulse.defined() ? timing_raw_db.pinImpulse.data_ptr<float>() : nullptr),
      pinRootDelay(timing_raw_db.pinRootDelay.defined() ? timing_raw_db.pinRootDelay.data_ptr<float>() : nullptr),
      arcDelay(timing_raw_db.arcDelay.data_ptr<float>()),
      // Critical path prefix info
      at_prefix_pin(timing_raw_db.at_prefix_pin.data_ptr<index_type>()),
      at_prefix_arc(timing_raw_db.at_prefix_arc.data_ptr<index_type>()),
      at_prefix_attr(timing_raw_db.at_prefix_attr.data_ptr<index_type>()),
      // Timing graph topology
      pin_forward_arc_list(timing_raw_db.pin_forward_arc_list.data_ptr<index_type>()),
      pin_forward_arc_list_end(timing_raw_db.pin_forward_arc_list_end.data_ptr<index_type>()),
      pin_backward_arc_list(timing_raw_db.pin_backward_arc_list.data_ptr<index_type>()),
      pin_backward_arc_list_end(timing_raw_db.pin_backward_arc_list_end.data_ptr<index_type>()),
      timing_arc_from_pin_id(timing_raw_db.timing_arc_from_pin_id.data_ptr<index_type>()),
      timing_arc_to_pin_id(timing_raw_db.timing_arc_to_pin_id.data_ptr<index_type>()),
      pin_num_fanin(timing_raw_db.pin_num_fanin.data_ptr<int>()),
      pin_fanout_list(timing_raw_db.pin_fanout_list.data_ptr<index_type>()),
      pin_fanout_list_end(timing_raw_db.pin_fanout_list_end.data_ptr<index_type>()),
      // Timer timing liberty variables
      timing_arc_id_map(timing_raw_db.timing_arc_id_map.data_ptr<int>()),
      arc_types(timing_raw_db.arc_types.data_ptr<uint8_t>()),
      arc_id2test_id(timing_raw_db.arc_id2test_id.data_ptr<int>()),
      test_id2_arc_id(timing_raw_db.test_id2_arc_id.data_ptr<int>()),
      test_id2_endpoint_id(timing_raw_db.test_id2_endpoint_id.data_ptr<int>()),
      primary_output2_endpoint_id(timing_raw_db.primary_output2_endpoint_id.data_ptr<int>()),
      // Circuit info
      flat_node2pin_start_map(timing_raw_db.flat_node2pin_start_map.data_ptr<int>()),
      flat_node2pin_map(timing_raw_db.flat_node2pin_map.data_ptr<int>()),
      pin2node_map(timing_raw_db.pin2node_map.data_ptr<int>()),
      flat_net2pin_start_map(timing_raw_db.flat_net2pin_start_map.data_ptr<int>()),
      flat_net2pin_map(timing_raw_db.flat_net2pin_map.data_ptr<int>()),
      pin2net_map(timing_raw_db.pin2net_map.data_ptr<int>()),
      net_mask(timing_raw_db.net_mask.data_ptr<bool>()),
      dmp_input_thresholds(timing_raw_db.dmp_input_thresholds.data_ptr<float>()),
      dmp_output_thresholds(timing_raw_db.dmp_output_thresholds.data_ptr<float>()),
      dmp_slew_lower_thresholds(timing_raw_db.dmp_slew_lower_thresholds.data_ptr<float>()),
      dmp_slew_upper_thresholds(timing_raw_db.dmp_slew_upper_thresholds.data_ptr<float>()),
      dmp_slew_derates(timing_raw_db.dmp_slew_derates.data_ptr<float>()),
      dmp_timing_library_ids(timing_raw_db.dmp_timing_library_ids.data_ptr<int>()),
      dmp_pin_library_ids(timing_raw_db.dmp_pin_library_ids.data_ptr<int>()),
      dmp_library_input_thresholds(timing_raw_db.dmp_library_input_thresholds.data_ptr<float>()),
      dmp_library_output_thresholds(timing_raw_db.dmp_library_output_thresholds.data_ptr<float>()),
      dmp_library_slew_lower_thresholds(timing_raw_db.dmp_library_slew_lower_thresholds.data_ptr<float>()),
      dmp_library_slew_upper_thresholds(timing_raw_db.dmp_library_slew_upper_thresholds.data_ptr<float>()),
      dmp_library_slew_derates(timing_raw_db.dmp_library_slew_derates.data_ptr<float>()),
      num_threads(timing_raw_db.num_threads),
      num_nodes(timing_raw_db.num_nodes),
      num_movable_nodes(timing_raw_db.num_movable_nodes),
      num_nets(timing_raw_db.num_nets),
      num_pins(timing_raw_db.num_pins),
      scale_factor(timing_raw_db.scale_factor) {
    num_arcs = gtdb.num_arcs;
    num_timings = gtdb.num_timings;
    total_num_fanouts = gtdb.total_num_fanouts;
    num_tests = gtdb.num_tests;
    num_POs = gtdb.num_POs;
    num_endpoint_pins = timing_raw_db.endpoint_unique_pin_ids.defined()
                            ? static_cast<int>(timing_raw_db.endpoint_unique_pin_ids.numel())
                            : 0;
    wire_resistance_per_micron = timing_raw_db.wire_resistance_per_micron;
    wire_capacitance_per_micron = timing_raw_db.wire_capacitance_per_micron;
    microns = timing_raw_db.microns;
    res_unit = gtdb.res_unit;
    cap_unit = gtdb.cap_unit;
    test_clock_periods = timing_raw_db.test_clock_periods.defined()
                             ? timing_raw_db.test_clock_periods.data_ptr<float>()
                             : nullptr;
    test_setup_uncertainties = timing_raw_db.test_setup_uncertainties.defined()
                                   ? timing_raw_db.test_setup_uncertainties.data_ptr<float>()
                                   : nullptr;
    test_hold_uncertainties = timing_raw_db.test_hold_uncertainties.defined()
                                  ? timing_raw_db.test_hold_uncertainties.data_ptr<float>()
                                  : nullptr;
    pin_clock_periods = timing_raw_db.pin_clock_periods.defined()
                            ? timing_raw_db.pin_clock_periods.data_ptr<float>()
                            : nullptr;
    pin_clock_rise_edges = timing_raw_db.pin_clock_rise_edges.defined()
                               ? timing_raw_db.pin_clock_rise_edges.data_ptr<float>()
                               : nullptr;
    pin_clock_fall_edges = timing_raw_db.pin_clock_fall_edges.defined()
                               ? timing_raw_db.pin_clock_fall_edges.data_ptr<float>()
                               : nullptr;
    pin_clock_slews = timing_raw_db.pin_clock_slews.defined()
                          ? timing_raw_db.pin_clock_slews.data_ptr<float>()
                          : nullptr;
    if (gtdb.clocks.empty())
        clock_period = 0;
    else
        clock_period = gtdb.clocks.begin()->second.period();
    gtdb_holder = gtdb_;
    timing_raw_db_holder = timing_raw_db_;
}

torch::Tensor GPUTimer::report_pin_at() {return timing_raw_db.pinAT;}
torch::Tensor GPUTimer::report_pin_gt_at() {
    return timing_raw_db.pinGT_AT;
    // return torch::from_blob(host_pinGT_AT.data(), {(long)num_pins, 4}, torch::kFloat32).clone();
}
torch::Tensor GPUTimer::report_pin_rat() { return timing_raw_db.pinRAT; }
torch::Tensor GPUTimer::report_pin_slew() { return timing_raw_db.pinSlew; }
torch::Tensor GPUTimer::report_pin_load() { return timing_raw_db.pinLoad; }
torch::Tensor GPUTimer::report_delay() { return timing_raw_db.arcDelay; }
torch::Tensor GPUTimer::report_endpoint_slack() { return endpoint_slacks; }
torch::Tensor GPUTimer::report_endpoint_pin_slack() {
    if (!endpoint_pin_slacks.defined() || endpoint_pin_slacks.numel() == 0 ||
        !timing_raw_db.endpoint_unique_pin_ids.defined() ||
        timing_raw_db.endpoint_unique_pin_ids.numel() == 0) {
        return endpoint_pin_slacks;
    }
    auto full = torch::full({num_pins, NUM_ATTR},
                            FLT_MAX,
                            torch::dtype(torch::kFloat32).device(endpoint_pin_slacks.device())).contiguous();
    full.index_put_({timing_raw_db.endpoint_unique_pin_ids.to(torch::kLong)}, endpoint_pin_slacks);
    return full;
}
torch::Tensor GPUTimer::endpoints_index(){ return timing_raw_db.endpoints_id;}
float GPUTimer::time_unit() const { return gtdb.time_unit; }

static torch::Tensor endpoint_pin_slacks_for_report(GPUTimer& timer) {
    auto pin_level_slacks = timer.report_pin_slack();
    auto [endpoints_id, tmp] = torch::_unique(timer.timing_raw_db.endpoints_id);
    return torch::nan_to_num(pin_level_slacks.index_select(0, endpoints_id), FLT_MAX);
}

float GPUTimer::report_wns(int el) {
    auto ep_slacks = endpoint_pin_slacks_for_report(*this);
    return torch::min(ep_slacks.index({"...", torch::indexing::Slice(2 * el, 2 * (el + 1))})).item<float>();
}

float GPUTimer::report_tns_elw(int el) {
    auto ep_slacks = endpoint_pin_slacks_for_report(*this);
    auto [slack_elw, order] = torch::min(ep_slacks.index({"...", torch::indexing::Slice(2 * el, 2 * (el + 1))}), 1);
    slack_elw.clamp_max_(0);

    return torch::sum(slack_elw, 0).item<float>();
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> GPUTimer::report_wns_and_tns() {
    auto ep_slacks = endpoint_pin_slacks_for_report(*this);
    auto [slack_e, order_e] = torch::min(ep_slacks.index({"...", torch::indexing::Slice(0, 2)}), 1);
    slack_e.clamp_max_(0);

    auto [slack_l, order_l] = torch::min(ep_slacks.index({"...", torch::indexing::Slice(2, 4)}), 1);
    slack_l.clamp_max_(0);

    return {torch::min(ep_slacks.index({"...", torch::indexing::Slice(0, 2)})),
            torch::sum(slack_e, 0),
            torch::min(ep_slacks.index({"...", torch::indexing::Slice(2, 4)})),
            torch::sum(slack_l, 0)};
}

torch::Tensor GPUTimer::report_pin_slack() {
    pin_slacks = torch::zeros_like(timing_raw_db.pinAT, torch::dtype(torch::kFloat32).device(timing_raw_db.pinAT.device()));
    auto s1 = timing_raw_db.pinAT - timing_raw_db.pinRAT;
    auto s2 = timing_raw_db.pinRAT - timing_raw_db.pinAT;
    pin_slacks.index({"...", torch::indexing::Slice(0, 2)}).data().copy_(s1.index({"...", torch::indexing::Slice(0, 2)}));
    pin_slacks.index({"...", torch::indexing::Slice(2, 4)}).data().copy_(s2.index({"...", torch::indexing::Slice(2, 4)}));

    return pin_slacks.contiguous();
}

// void GPUTimer::update_endpoints() {
//     pin_slacks = torch::zeros_like(timing_raw_db.pinAT, torch::dtype(torch::kFloat32).device(timing_raw_db.pinAT.device()));
//     auto s1 = timing_raw_db.pinAT - timing_raw_db.pinRAT;
//     auto s2 = timing_raw_db.pinRAT - timing_raw_db.pinAT;
//     pin_slacks.index({"...", torch::indexing::Slice(0, 2)}).data().copy_(s1.index({"...", torch::indexing::Slice(0, 2)}));
//     pin_slacks.index({"...", torch::indexing::Slice(2, 4)}).data().copy_(s2.index({"...", torch::indexing::Slice(2, 4)}));
    
//     auto [endpoints_id, tmp1] = torch::_unique(timing_raw_db.endpoints_id);
//     endpoint_slacks = torch::nan_to_num(pin_slacks.index_select(0, endpoints_id));
// }

}  // namespace gt
