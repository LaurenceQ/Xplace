#include "GTDatabase.h"
#include "sdc/SdcUtils.h"

#include "common/common.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Pin.h"
#include "common/lib/Liberty.h"
#include "common/lib/Lut.h"
#include "common/lib/Timing.h"
#include "common/lib/sdc/sdc.h"
#include "io_parser/gp/GPDatabase.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <deque>
#include <iterator>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include <omp.h>

namespace gt {

void gputimer_log_cuda_mem_info(const char* label);
bool gputimer_env_enabled(const char* name);


void GTDatabase::preparePinNameMapForSdc(const sdc::SDC& sdc) {
    pin_name_map_targets.clear();
    build_full_pin_name_map = gputimer_env_enabled("GPUTIMER_BUILD_FULL_PIN_NAME_MAP");
    if (build_full_pin_name_map) {
        return;
    }

    auto collect_object_pin_targets = [this](const sdc::Object& object) {
        std::visit(Functors{[this](const sdc::GetPins& get_pins) {
                                for (const auto& pin_name : get_pins.pins) {
                                    add_pin_name_target_variants(pin_name_map_targets, pin_name);
                                }
                            },
                            [this](const sdc::GetPorts& get_ports) {
                                for (const auto& port_name : get_ports.ports) {
                                    add_pin_name_target_variants(pin_name_map_targets, port_name);
                                }
                            },
                            [](const auto&) {}},
                   object);
    };

    for (const auto& command : sdc.commands) {
        if (const auto* clock_latency = std::get_if<sdc::SetClockLatency>(&command);
            clock_latency != nullptr && clock_latency->object_list) {
            collect_object_pin_targets(*clock_latency->object_list);
        } else if (const auto* propagated_clock = std::get_if<sdc::SetPropagatedClock>(&command);
                   propagated_clock != nullptr && propagated_clock->object_list) {
            collect_object_pin_targets(*propagated_clock->object_list);
        }
    }
}

void GTDatabase::readSdc(sdc::SDC& sdc) {
    driving_cell_sources.clear();
    clock_transitions.clear();
    clock_setup_uncertainty.clear();
    clock_hold_uncertainty.clear();
    propagated_all_clocks = false;
    propagated_clock_names.clear();
    propagated_clock_pins.clear();
    power_disabled_constraint_arc.assign(timing_arc_from_pin_id.size(), 0);
    pin_clock_latency_overrides.clear();
    output_delay_clock_by_pin_attr.clear();
    const size_t pin_state_count = static_cast<size_t>(num_pins) * NUM_ATTR;
    host_pin_slew.assign(pin_state_count, nanf(""));
    host_pin_load.assign(pin_state_count, 0.0f);
    host_pin_rat.assign(pin_state_count, nanf(""));
    host_pin_at.assign(pin_state_count, nanf(""));
    for (auto& command : sdc.commands) {
        std::visit(Functors{[this](auto&& cmd) { _read_sdc(cmd); }}, command);
    }
    num_timings = static_cast<int>(liberty_timing_arcs.size());

    float default_period = 0.0f;
    float default_rise_edge = nanf("");
    float default_fall_edge = nanf("");
    if (!clocks.empty()) {
        default_period = clocks.begin()->second.period();
        default_rise_edge = clocks.begin()->second.rise_edge();
        default_fall_edge = clocks.begin()->second.fall_edge();
    }

    const uint8_t invalid_clock_id = std::numeric_limits<uint8_t>::max();
    clock_periods.clear();
    auto intern_clock_period = [&](float period) -> uint8_t {
        if (!std::isfinite(period) || period <= 0.0f) {
            return invalid_clock_id;
        }
        for (size_t idx = 0; idx < clock_periods.size(); ++idx) {
            if (clock_periods[idx] == period) {
                return static_cast<uint8_t>(idx);
            }
        }
        if (clock_periods.size() >= static_cast<size_t>(invalid_clock_id)) {
            return invalid_clock_id;
        }
        clock_periods.push_back(period);
        return static_cast<uint8_t>(clock_periods.size() - 1);
    };

    const int sdc_threads = std::max(1, timing_raw_db.num_threads);
    const int num_nets = static_cast<int>(gpdb.getNets().size());
    const auto& gp_pins = gpdb.getPins();
    const int gp_pin_count = static_cast<int>(gp_pins.size());
    std::vector<const Clock*> net_clock(num_nets, nullptr);
    std::vector<uint8_t> net_clock_id(num_nets, invalid_clock_id);
    std::vector<float> net_clock_period(num_nets, 0.0f);
    std::vector<float> net_clock_rise_edge(num_nets, nanf(""));
    std::vector<float> net_clock_fall_edge(num_nets, nanf(""));
    std::vector<float> net_clock_setup_uncertainty(num_nets, 0.0f);
    std::vector<float> net_clock_hold_uncertainty(num_nets, 0.0f);
    std::vector<const std::array<float, NUM_ATTR>*> net_clock_transition(num_nets, nullptr);
    const Clock* default_clock = clocks.empty() ? nullptr : &clocks.begin()->second;
    const uint8_t default_clock_id = default_clock ? intern_clock_period(default_clock->period())
                                                   : invalid_clock_id;
    pin_clock_ids.assign(num_pins, default_clock_id);
    pin_clock_periods.assign(num_pins, default_period);
    pin_clock_rise_edges.assign(num_pins, default_rise_edge);
    pin_clock_fall_edges.assign(num_pins, default_fall_edge);
    pin_clock_slews.assign(num_pins * NUM_ATTR, nanf(""));
    net_is_clock.assign(num_nets, 0);
    std::vector<const Clock*> pin_clock_context(num_pins, default_clock);
    std::vector<char> pin_clock_context_is_default(num_pins, default_clock ? 1 : 0);
    auto clock_transition_values = [&](const Clock& clock) -> const std::array<float, NUM_ATTR>* {
        auto transition_iter = clock_transitions.find(clock.name());
        return transition_iter == clock_transitions.end() ? nullptr : &transition_iter->second;
    };
    auto apply_clock_transition_to_pin = [&](int pin_id, const std::array<float, NUM_ATTR>* values) {
        if (values == nullptr) {
            return;
        }
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            const float transition = (*values)[attr];
            if (std::isfinite(transition)) {
                hostPinSlew(pin_id, attr) = transition;
                pin_clock_slews[pin_id * NUM_ATTR + attr] = transition;
            }
        }
    };
    for (auto& [clock_name, clock] : clocks) {
        if (clock.source_id() < 0 ||
            clock.source_id() >= gp_pin_count) {
            continue;
        }
        const uint8_t clock_id = intern_clock_period(clock.period());
        const auto* transition_values = clock_transition_values(clock);
        const int source_pin = clock.source_id();
        const int net_id = gp_pins[source_pin].getParNetId();
        if (net_id >= 0 && net_id < num_nets) {
            net_clock[net_id] = &clock;
            net_clock_id[net_id] = clock_id;
            net_clock_period[net_id] = clock.period();
            net_clock_rise_edge[net_id] = clock.rise_edge();
            net_clock_fall_edge[net_id] = clock.fall_edge();
            if (auto setup_iter = clock_setup_uncertainty.find(clock.name());
                setup_iter != clock_setup_uncertainty.end()) {
                net_clock_setup_uncertainty[net_id] = setup_iter->second;
            }
            if (auto hold_iter = clock_hold_uncertainty.find(clock.name());
                hold_iter != clock_hold_uncertainty.end()) {
                net_clock_hold_uncertainty[net_id] = hold_iter->second;
            }
            net_clock_transition[net_id] = transition_values;
            net_is_clock[net_id] = 1;
        }
        pin_clock_ids[source_pin] = clock_id;
        pin_clock_periods[source_pin] = clock.period();
        pin_clock_rise_edges[source_pin] = clock.rise_edge();
        pin_clock_fall_edges[source_pin] = clock.fall_edge();
        pin_clock_context[source_pin] = &clock;
        pin_clock_context_is_default[source_pin] = 0;
        logger.info("clock: %s, source_pin: %s, period: %.2f, rise_edge: %.3f, fall_edge: %.3f",
                    clock_name.c_str(),
                    gp_pins[source_pin].getName().c_str(),
                    clock.period(),
                    clock.rise_edge(),
                    clock.fall_edge());
    }
#pragma omp parallel for num_threads(sdc_threads) schedule(static)
    for (int pin_index = 0; pin_index < gp_pin_count; ++pin_index) {
        const auto& gppin = gp_pins[pin_index];
        const int net_id = gppin.getParNetId();
        if (net_id < 0 || net_id >= num_nets || net_clock[net_id] == nullptr) {
            continue;
        }
        const int pin_id = gppin.getId();
        const Clock* clock = net_clock[net_id];
        pin_clock_ids[pin_id] = net_clock_id[net_id];
        pin_clock_periods[pin_id] = net_clock_period[net_id];
        pin_clock_rise_edges[pin_id] = net_clock_rise_edge[net_id];
        pin_clock_fall_edges[pin_id] = net_clock_fall_edge[net_id];
        pin_clock_context[pin_id] = clock;
        pin_clock_context_is_default[pin_id] = 0;
        apply_clock_transition_to_pin(pin_id, net_clock_transition[net_id]);
    }

    int pin_clock_latency_override_count = 0;
    int pin_clock_latency_with_clock_context = 0;
    int pin_clock_latency_with_default_context = 0;
    int pin_clock_latency_without_context = 0;
    for (const auto& [pin_id, override] : pin_clock_latency_overrides) {
        if (pin_id < 0 || pin_id >= num_pins || !std::isfinite(override)) {
            continue;
        }
        ++pin_clock_latency_override_count;
        const Clock* clock = pin_clock_context[pin_id];
        if (clock) {
            pin_clock_ids[pin_id] = intern_clock_period(clock->period());
            pin_clock_periods[pin_id] = clock->period();
            // OpenSTA gives pin clock latency precedence over clock-object
            // latency. Apply it to the waveform edge, not to clock.rise_edge()
            // / fall_edge(), which already include clock-object latency.
            pin_clock_rise_edges[pin_id] = clock->waveform_rise_edge() + override;
            pin_clock_fall_edges[pin_id] = clock->waveform_fall_edge() + override;
            if (pin_clock_context_is_default[pin_id]) {
                ++pin_clock_latency_with_default_context;
            } else {
                ++pin_clock_latency_with_clock_context;
            }
        } else {
            pin_clock_ids[pin_id] = invalid_clock_id;
            pin_clock_rise_edges[pin_id] = override;
            pin_clock_fall_edges[pin_id] = override;
            ++pin_clock_latency_without_context;
        }
    }
    if (pin_clock_latency_override_count > 0) {
        logger.warning("Applied %d pin set_clock_latency overrides (%d clock-net/source context, %d default-clock context, %d no-clock context)",
                       pin_clock_latency_override_count,
                       pin_clock_latency_with_clock_context,
                       pin_clock_latency_with_default_context,
                       pin_clock_latency_without_context);
    }
    if (default_clock) {
        const auto* default_transition_values = clock_transition_values(*default_clock);
#pragma omp parallel for num_threads(sdc_threads) schedule(static)
        for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
            if (!pin_is_clk[pin_id]) {
                continue;
            }
            apply_clock_transition_to_pin(pin_id, default_transition_values);
        }
    }

    test_clock_ids.assign(num_tests, default_clock_id);
    test_clock_periods.assign(num_tests, default_period);
    test_setup_uncertainties.assign(num_tests, 0.0f);
    test_hold_uncertainties.assign(num_tests, 0.0f);
    int tests_with_clock = 0;
#pragma omp parallel for num_threads(sdc_threads) schedule(static) reduction(+:tests_with_clock)
    for (int test_id = 0; test_id < static_cast<int>(test_id2_arc_id.size()); ++test_id) {
        const int arc_id = test_id2_arc_id[test_id];
        if (arc_id < 0 || arc_id >= static_cast<int>(timing_arc_from_pin_id.size())) {
            continue;
        }
        const int clock_pin_id = timing_arc_from_pin_id[arc_id];
        if (clock_pin_id < 0 || clock_pin_id >= gp_pin_count) {
            continue;
        }
        const int net_id = gp_pins[clock_pin_id].getParNetId();
        if (net_id < 0 || net_id >= num_nets || net_clock[net_id] == nullptr) {
            continue;
        }
        test_clock_ids[test_id] = net_clock_id[net_id];
        test_clock_periods[test_id] = net_clock_period[net_id];
        test_setup_uncertainties[test_id] = net_clock_setup_uncertainty[net_id];
        test_hold_uncertainties[test_id] = net_clock_hold_uncertainty[net_id];
        ++tests_with_clock;
    }
    logger.info("Mapped %d/%d timing tests to capture clocks", tests_with_clock, num_tests);

    // set nan slew of PIs to half period
#pragma omp parallel for num_threads(sdc_threads) schedule(static)
    for (int pi_idx = 0; pi_idx < static_cast<int>(primary_inputs.size()); ++pi_idx) {
        const int pi = primary_inputs[pi_idx];
        if (std::isnan(hostPinSlew(pi, 0))) hostPinSlew(pi, 0) = 0.0f;
        if (std::isnan(hostPinSlew(pi, 1))) hostPinSlew(pi, 1) = 0.0f;
        if (std::isnan(hostPinSlew(pi, 2))) hostPinSlew(pi, 2) = 0.0f;
        if (std::isnan(hostPinSlew(pi, 3))) hostPinSlew(pi, 3) = 0.0f;
    }

    for (auto& [clock_name, clock] : clocks) {
        if (clock.source_id() == -1) {
            continue;
        }
        int clock_pin_id = clock.source_id();
        pin_is_clk[clock_pin_id] = 1;
        if (std::isnan(hostPinAT(clock_pin_id, 0))) hostPinAT(clock_pin_id, 0) = clock.rise_edge();
        if (std::isnan(hostPinAT(clock_pin_id, 1))) hostPinAT(clock_pin_id, 1) = clock.fall_edge();
        if (std::isnan(hostPinAT(clock_pin_id, 2))) hostPinAT(clock_pin_id, 2) = clock.rise_edge();
        if (std::isnan(hostPinAT(clock_pin_id, 3))) hostPinAT(clock_pin_id, 3) = clock.fall_edge();
    }

    pin_is_ideal_clk.assign(num_pins, 0);
    int ideal_clock_pin_count = 0;
    int propagated_clock_pin_count = 0;
    int direct_propagated_clock_pin_count = 0;
#pragma omp parallel for num_threads(sdc_threads) schedule(static) reduction(+:ideal_clock_pin_count,propagated_clock_pin_count,direct_propagated_clock_pin_count)
    for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
        if (!pin_is_clk[pin_id]) {
            continue;
        }
        const bool direct_propagated = propagated_clock_pins.find(pin_id) != propagated_clock_pins.end();
        const Clock* clock = pin_clock_context[pin_id];
        const bool clock_propagated =
            propagated_all_clocks ||
            (clock != nullptr && propagated_clock_names.find(clock->name()) != propagated_clock_names.end());
        const bool propagated = direct_propagated || clock_propagated;
        pin_is_ideal_clk[pin_id] = propagated ? 0 : 1;
        if (propagated) {
            ++propagated_clock_pin_count;
            if (direct_propagated) {
                ++direct_propagated_clock_pin_count;
            }
        } else {
            ++ideal_clock_pin_count;
        }
    }
#pragma omp parallel for num_threads(sdc_threads) schedule(static)
    for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
        if (!pin_is_ideal_clk[pin_id]) {
            continue;
        }
        const float rise_edge = pin_clock_rise_edges[pin_id];
        const float fall_edge = pin_clock_fall_edges[pin_id];
        if (std::isfinite(rise_edge)) {
            hostPinAT(pin_id, 0) = rise_edge;
            hostPinAT(pin_id, 2) = rise_edge;
        }
        if (std::isfinite(fall_edge)) {
            hostPinAT(pin_id, 1) = fall_edge;
            hostPinAT(pin_id, 3) = fall_edge;
        }
    }
    logger.info("Clock propagation: %d ideal clock pins, %d propagated clock pins (%d direct pin matches)",
                ideal_clock_pin_count,
                propagated_clock_pin_count,
                direct_propagated_clock_pin_count);

    auto device = timing_raw_db.node_size_x.device();
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    auto byte_options = torch::TensorOptions().dtype(torch::kUInt8);
    timing_raw_db.pinSlew = torch::from_blob(host_pin_slew.data(), {num_pins, NUM_ATTR}, float_options).contiguous().to(device);
    timing_raw_db.pinLoad = torch::from_blob(host_pin_load.data(), {num_pins, NUM_ATTR}, float_options).contiguous().to(device);
    timing_raw_db.pinRAT = torch::from_blob(host_pin_rat.data(), {num_pins, NUM_ATTR}, float_options).contiguous().to(device);
    timing_raw_db.pinAT = torch::from_blob(host_pin_at.data(), {num_pins, NUM_ATTR}, float_options).contiguous().to(device);
    vector<float>().swap(host_pin_slew);
    vector<float>().swap(host_pin_load);
    vector<float>().swap(host_pin_rat);
    vector<float>().swap(host_pin_at);
    timing_raw_db.clock_periods = torch::from_blob(clock_periods.data(), {static_cast<int>(clock_periods.size())}, float_options).contiguous().to(device);
    timing_raw_db.pin_clock_ids = torch::from_blob(pin_clock_ids.data(), {static_cast<int>(pin_clock_ids.size())}, byte_options).contiguous().to(device);
    timing_raw_db.test_clock_ids = torch::from_blob(test_clock_ids.data(), {static_cast<int>(test_clock_ids.size())}, byte_options).contiguous().to(device);
    timing_raw_db.test_clock_periods = torch::from_blob(test_clock_periods.data(), {static_cast<int>(test_clock_periods.size())}, float_options).contiguous().to(device);
    timing_raw_db.test_setup_uncertainties = torch::from_blob(test_setup_uncertainties.data(), {static_cast<int>(test_setup_uncertainties.size())}, float_options).contiguous().to(device);
    timing_raw_db.test_hold_uncertainties = torch::from_blob(test_hold_uncertainties.data(), {static_cast<int>(test_hold_uncertainties.size())}, float_options).contiguous().to(device);
    timing_raw_db.pin_clock_periods = torch::from_blob(pin_clock_periods.data(), {static_cast<int>(pin_clock_periods.size())}, float_options).contiguous().to(device);
    timing_raw_db.pin_clock_rise_edges = torch::from_blob(pin_clock_rise_edges.data(), {static_cast<int>(pin_clock_rise_edges.size())}, float_options).contiguous().to(device);
    timing_raw_db.pin_clock_fall_edges = torch::from_blob(pin_clock_fall_edges.data(), {static_cast<int>(pin_clock_fall_edges.size())}, float_options).contiguous().to(device);
    timing_raw_db.pin_clock_slews = torch::from_blob(pin_clock_slews.data(), {static_cast<int>(pin_clock_slews.size())}, float_options).contiguous().to(device);
    gputimer_log_cuda_mem_info("GTDatabase::readSdc after_clock_tensors");
}

// Sets input delay on pins or input ports relative to a clock signal.

}  // namespace gt
