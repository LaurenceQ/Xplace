#include "GTDatabase.h"
#include "sdc/SdcUtils.h"

#include "common/XplaceLog.h"
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
#include <stdexcept>
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
        } else if (const auto* case_analysis = std::get_if<sdc::SetCaseAnalysis>(&command);
                   case_analysis != nullptr && case_analysis->port_pin_list) {
            collect_object_pin_targets(*case_analysis->port_pin_list);
        } else if (const auto* propagated_clock = std::get_if<sdc::SetPropagatedClock>(&command);
                   propagated_clock != nullptr && propagated_clock->object_list) {
            collect_object_pin_targets(*propagated_clock->object_list);
        }
    }
}

void GTDatabase::RunSdcConstantSimulation() {
    // SDC parsing now happens after structural graph construction. Constant
    // propagation and timing-disabled arc masks will be added here.
}

bool GTDatabase::ClockIdValid(uint16_t clock_id) const {
    return clock_id != kInvalidClockId &&
           clock_id < static_cast<uint16_t>(clock_periods.size());
}

float GTDatabase::ClockPeriodForPin(int pin_id) const {
    if (pin_id < 0 || pin_id >= static_cast<int>(pin_clock_ids.size())) {
        return nanf("");
    }
    const uint16_t clock_id = pin_clock_ids[pin_id];
    return ClockIdValid(clock_id) ? clock_periods[clock_id] : nanf("");
}

float GTDatabase::ClockRiseEdgeForPin(int pin_id) const {
    if (pin_id < 0 || pin_id >= static_cast<int>(pin_clock_ids.size())) {
        return nanf("");
    }
    const uint16_t clock_id = pin_clock_ids[pin_id];
    const float override = pin_id < static_cast<int>(pin_clock_latency_overrides.size())
                               ? pin_clock_latency_overrides[pin_id]
                               : nanf("");
    if (std::isfinite(override)) {
        if (ClockIdValid(clock_id)) {
            const float waveform = clock_waveform_rise_edges[clock_id];
            return std::isfinite(waveform) ? waveform + override : override;
        }
        return override;
    }
    return ClockIdValid(clock_id) ? clock_rise_edges[clock_id] : nanf("");
}

float GTDatabase::ClockFallEdgeForPin(int pin_id) const {
    if (pin_id < 0 || pin_id >= static_cast<int>(pin_clock_ids.size())) {
        return nanf("");
    }
    const uint16_t clock_id = pin_clock_ids[pin_id];
    const float override = pin_id < static_cast<int>(pin_clock_latency_overrides.size())
                               ? pin_clock_latency_overrides[pin_id]
                               : nanf("");
    if (std::isfinite(override)) {
        if (ClockIdValid(clock_id)) {
            const float waveform = clock_waveform_fall_edges[clock_id];
            return std::isfinite(waveform) ? waveform + override : override;
        }
        return override;
    }
    return ClockIdValid(clock_id) ? clock_fall_edges[clock_id] : nanf("");
}

float GTDatabase::ClockSlewForPin(int pin_id, int attr) const {
    if (pin_id < 0 || pin_id >= static_cast<int>(pin_clock_ids.size()) ||
        attr < 0 || attr >= NUM_ATTR) {
        return nanf("");
    }
    const uint16_t clock_id = pin_clock_ids[pin_id];
    if (!ClockIdValid(clock_id)) {
        return nanf("");
    }
    const size_t idx = static_cast<size_t>(clock_id) * NUM_ATTR + attr;
    return idx < clock_slews.size() ? clock_slews[idx] : nanf("");
}

float GTDatabase::ClockSetupUncertaintyForTest(int test_id) const {
    if (test_id < 0 || test_id >= static_cast<int>(test_clock_ids.size())) {
        return 0.0f;
    }
    const uint16_t clock_id = test_clock_ids[test_id];
    return ClockIdValid(clock_id) ? clock_setup_uncertainties[clock_id] : 0.0f;
}

float GTDatabase::ClockHoldUncertaintyForTest(int test_id) const {
    if (test_id < 0 || test_id >= static_cast<int>(test_clock_ids.size())) {
        return 0.0f;
    }
    const uint16_t clock_id = test_clock_ids[test_id];
    return ClockIdValid(clock_id) ? clock_hold_uncertainties[clock_id] : 0.0f;
}

void GTDatabase::InitPinClockLatencyOverrides() {
    pin_clock_latency_overrides.assign(num_pins, nanf(""));
}

bool GTDatabase::SetClockLatencyHasUnsupportedMask(const sdc::SetClockLatency& obj) const {
    return obj.rise.has_value() || obj.fall.has_value() ||
           obj.min.has_value() || obj.max.has_value() ||
           obj.source.has_value() || obj.early.has_value() ||
           obj.late.has_value();
}

void GTDatabase::ApplyScalarPinClockLatencyOverride(int pin_id, float delay) {
    if (pin_id >= 0 && pin_id < static_cast<int>(pin_clock_latency_overrides.size()) &&
        std::isfinite(delay)) {
        pin_clock_latency_overrides[pin_id] = delay;
    }
}

uint16_t GTDatabase::BuildClockIdTablesForSdc() {
    const Clock* default_clock = clocks.empty() ? nullptr : &clocks.begin()->second;
    clock_names.clear();
    clock_name2id.clear();
    clock_periods.clear();
    clock_rise_edges.clear();
    clock_fall_edges.clear();
    clock_waveform_rise_edges.clear();
    clock_waveform_fall_edges.clear();
    clock_slews.clear();
    clock_setup_uncertainties.clear();
    clock_hold_uncertainties.clear();

    std::vector<std::string> names;
    names.reserve(clocks.size());
    for (const auto& [clock_name, clock] : clocks) {
        (void)clock;
        names.push_back(clock_name);
    }
    std::sort(names.begin(), names.end());
    for (const std::string& clock_name : names) {
        if (clock_names.size() >= static_cast<size_t>(kInvalidClockId)) {
            throw std::runtime_error("SDC clock count exceeds uint16 clock id range");
        }
        const Clock& clock = clocks.at(clock_name);
        const uint16_t clock_id = static_cast<uint16_t>(clock_names.size());
        clock_names.push_back(clock_name);
        clock_name2id.emplace(clock_name, clock_id);
        clock_periods.push_back(clock.period());
        clock_rise_edges.push_back(clock.rise_edge());
        clock_fall_edges.push_back(clock.fall_edge());
        clock_waveform_rise_edges.push_back(clock.waveform_rise_edge());
        clock_waveform_fall_edges.push_back(clock.waveform_fall_edge());
        auto transition_iter = clock_transitions.find(clock_name);
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            clock_slews.push_back(transition_iter == clock_transitions.end()
                                      ? nanf("")
                                      : transition_iter->second[attr]);
        }
        auto setup_iter = clock_setup_uncertainty.find(clock_name);
        clock_setup_uncertainties.push_back(setup_iter == clock_setup_uncertainty.end()
                                                ? 0.0f
                                                : setup_iter->second);
        auto hold_iter = clock_hold_uncertainty.find(clock_name);
        clock_hold_uncertainties.push_back(hold_iter == clock_hold_uncertainty.end()
                                               ? 0.0f
                                               : hold_iter->second);
    }

    if (default_clock == nullptr) {
        return kInvalidClockId;
    }
    auto default_iter = clock_name2id.find(default_clock->name());
    return default_iter == clock_name2id.end() ? kInvalidClockId : default_iter->second;
}

void GTDatabase::AssignClockIdsToPins(uint16_t default_clock_id,
                                      vector<uint16_t>& net_clock_ids,
                                      int sdc_threads) {
    const int num_nets = static_cast<int>(gpdb.getNets().size());
    const auto& gp_pins = gpdb.getPins();
    const int gp_pin_count = static_cast<int>(gp_pins.size());
    pin_clock_ids.assign(num_pins, default_clock_id);
    pin_clock_is_default_fallback.assign(num_pins, ClockIdValid(default_clock_id) ? 1 : 0);
    net_clock_ids.assign(num_nets, kInvalidClockId);
    net_is_clock.assign(num_nets, 0);

    std::vector<int> clock_net_ids;
    std::vector<uint8_t> clock_net_seen(num_nets, 0);
    for (auto& [clock_name, clock] : clocks) {
        auto clock_id_iter = clock_name2id.find(clock_name);
        if (clock_id_iter == clock_name2id.end()) {
            continue;
        }
        const uint16_t clock_id = clock_id_iter->second;
        const int source_pin = clock.source_id();
        if (source_pin < 0 || source_pin >= gp_pin_count) {
            continue;
        }
        const int net_id = gp_pins[source_pin].getParNetId();
        if (net_id >= 0 && net_id < num_nets) {
            if (net_clock_ids[net_id] != kInvalidClockId &&
                net_clock_ids[net_id] != clock_id) {
                logger.warning("Multiple SDC clocks mapped to net %s; keeping %s and ignoring %s for clock-net propagation",
                               net_names[net_id].c_str(),
                               clock_names[net_clock_ids[net_id]].c_str(),
                               clock_name.c_str());
            } else {
                net_clock_ids[net_id] = clock_id;
                net_is_clock[net_id] = 1;
                if (!clock_net_seen[net_id]) {
                    clock_net_seen[net_id] = 1;
                    clock_net_ids.push_back(net_id);
                }
            }
        }
        if (source_pin >= 0 && source_pin < num_pins) {
            pin_clock_ids[source_pin] = clock_id;
            pin_clock_is_default_fallback[source_pin] = 0;
        }
        XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_CLOCKS",
                      "clock=%s source_pin=%s period=%.2f rise_edge=%.3f fall_edge=%.3f",
                      clock_name.c_str(),
                      gp_pins[source_pin].getName().c_str(),
                      clock.period(),
                      clock.rise_edge(),
                      clock.fall_edge());
    }

#pragma omp parallel for num_threads(sdc_threads) schedule(static)
    for (int idx = 0; idx < static_cast<int>(clock_net_ids.size()); ++idx) {
        const int net_id = clock_net_ids[idx];
        const uint16_t clock_id = net_clock_ids[net_id];
        if (!ClockIdValid(clock_id)) {
            continue;
        }
        for (int pin_id : gpdb.getNets()[net_id].pins()) {
            if (pin_id < 0 || pin_id >= num_pins) {
                continue;
            }
            pin_clock_ids[pin_id] = clock_id;
            pin_clock_is_default_fallback[pin_id] = 0;
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                const float slew = clock_slews[static_cast<size_t>(clock_id) * NUM_ATTR + attr];
                if (std::isfinite(slew)) {
                    hostPinSlew(pin_id, attr) = slew;
                }
            }
        }
    }
}

void GTDatabase::MapTestsToClockIds(const vector<uint16_t>& net_clock_ids,
                                    uint16_t default_clock_id,
                                    int sdc_threads) {
    const auto& gp_pins = gpdb.getPins();
    const int gp_pin_count = static_cast<int>(gp_pins.size());
    test_clock_ids.assign(num_tests, default_clock_id);
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
        if (net_id < 0 || net_id >= static_cast<int>(net_clock_ids.size()) ||
            !ClockIdValid(net_clock_ids[net_id])) {
            continue;
        }
        test_clock_ids[test_id] = net_clock_ids[net_id];
        ++tests_with_clock;
    }
    logger.info("Mapped %d/%d timing tests to capture clocks", tests_with_clock, num_tests);
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
    InitPinClockLatencyOverrides();
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

    const int sdc_threads = std::max(1, timing_raw_db.num_threads);
    const uint16_t default_clock_id = BuildClockIdTablesForSdc();
    std::vector<uint16_t> net_clock_ids;
    AssignClockIdsToPins(default_clock_id, net_clock_ids, sdc_threads);

    int pin_clock_latency_override_count = 0;
    int pin_clock_latency_with_clock_context = 0;
    int pin_clock_latency_with_default_context = 0;
    int pin_clock_latency_without_context = 0;
    for (int pin_id = 0; pin_id < static_cast<int>(pin_clock_latency_overrides.size()); ++pin_id) {
        const float override = pin_clock_latency_overrides[pin_id];
        if (!std::isfinite(override)) {
            continue;
        }
        ++pin_clock_latency_override_count;
        if (pin_id >= static_cast<int>(pin_clock_ids.size()) ||
            !ClockIdValid(pin_clock_ids[pin_id])) {
            ++pin_clock_latency_without_context;
        } else if (pin_id < static_cast<int>(pin_clock_is_default_fallback.size()) &&
                   pin_clock_is_default_fallback[pin_id]) {
            ++pin_clock_latency_with_default_context;
        } else {
            ++pin_clock_latency_with_clock_context;
        }
    }
    if (pin_clock_latency_override_count > 0) {
        logger.warning("Applied %d pin set_clock_latency overrides (%d clock-net/source context, %d default-clock context, %d no-clock context)",
                       pin_clock_latency_override_count,
                       pin_clock_latency_with_clock_context,
                       pin_clock_latency_with_default_context,
                       pin_clock_latency_without_context);
    }
#pragma omp parallel for num_threads(sdc_threads) schedule(static)
    for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
        if (!pin_is_clk[pin_id]) {
            continue;
        }
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            const float slew = ClockSlewForPin(pin_id, attr);
            if (std::isfinite(slew)) {
                hostPinSlew(pin_id, attr) = slew;
            }
        }
    }

    MapTestsToClockIds(net_clock_ids, default_clock_id, sdc_threads);

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
        if (clock.source_id() < 0 || clock.source_id() >= num_pins) {
            continue;
        }
        int clock_pin_id = clock.source_id();
        pin_is_clk[clock_pin_id] = 1;
        const float rise_edge = ClockRiseEdgeForPin(clock_pin_id);
        const float fall_edge = ClockFallEdgeForPin(clock_pin_id);
        if (std::isnan(hostPinAT(clock_pin_id, 0))) hostPinAT(clock_pin_id, 0) = rise_edge;
        if (std::isnan(hostPinAT(clock_pin_id, 1))) hostPinAT(clock_pin_id, 1) = fall_edge;
        if (std::isnan(hostPinAT(clock_pin_id, 2))) hostPinAT(clock_pin_id, 2) = rise_edge;
        if (std::isnan(hostPinAT(clock_pin_id, 3))) hostPinAT(clock_pin_id, 3) = fall_edge;
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
        const uint16_t clock_id = pin_id < static_cast<int>(pin_clock_ids.size())
                                      ? pin_clock_ids[pin_id]
                                      : kInvalidClockId;
        const bool clock_propagated =
            propagated_all_clocks ||
            (ClockIdValid(clock_id) &&
             propagated_clock_names.find(clock_names[clock_id]) != propagated_clock_names.end());
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
        const float rise_edge = ClockRiseEdgeForPin(pin_id);
        const float fall_edge = ClockFallEdgeForPin(pin_id);
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
    timing_raw_db.pinSlew = torch::from_blob(host_pin_slew.data(), {num_pins, NUM_ATTR}, float_options).contiguous().to(device);
    timing_raw_db.pinLoad = torch::from_blob(host_pin_load.data(), {num_pins, NUM_ATTR}, float_options).contiguous().to(device);
    timing_raw_db.pinRAT = torch::from_blob(host_pin_rat.data(), {num_pins, NUM_ATTR}, float_options).contiguous().to(device);
    timing_raw_db.pinAT = torch::from_blob(host_pin_at.data(), {num_pins, NUM_ATTR}, float_options).contiguous().to(device);
    vector<float>().swap(host_pin_slew);
    vector<float>().swap(host_pin_load);
    vector<float>().swap(host_pin_rat);
    vector<float>().swap(host_pin_at);
    gputimer_log_cuda_mem_info("GTDatabase::readSdc after_clock_tensors");
}

// Sets input delay on pins or input ports relative to a clock signal.

}  // namespace gt
