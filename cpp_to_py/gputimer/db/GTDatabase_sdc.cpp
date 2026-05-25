#include "GTDatabase.h"

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
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace gt {

void gputimer_log_cuda_mem_info(const char* label);
bool gputimer_env_enabled(const char* name);

namespace {

bool is_transition_defined_cpu(const TimingArc* timing_arc, int input_rf, int output_rf) {
    if (timing_arc->is_rising_edge_triggered() && input_rf != RISE) return false;
    if (timing_arc->is_falling_edge_triggered() && input_rf != FALL) return false;

    switch (timing_arc->timing_sense_) {
        case TimingSense::positive_unate:
            return input_rf == output_rf;
        case TimingSense::negative_unate:
            return input_rf != output_rf;
        default:
            return true;
    }
}

void warn_missing_sdc_object(const char* command,
                             const char* kind,
                             const std::string& name) {
    if (!gputimer_env_enabled("GPUTIMER_VERBOSE_SDC_WARNINGS")) return;
    std::fprintf(stderr, "%s: %s \"%s\" not found\n", command, kind, name.c_str());
}

void add_pin_name_target_variants(std::unordered_set<std::string>& targets,
                                  const std::string& name) {
    if (name.empty()) {
        return;
    }
    targets.insert(name);
    const auto slash_pos = name.rfind('/');
    if (slash_pos != std::string::npos) {
        std::string gp_pin_name = name;
        gp_pin_name[slash_pos] = ':';
        targets.insert(std::move(gp_pin_name));
    }
    const auto colon_pos = name.rfind(':');
    if (colon_pos != std::string::npos) {
        std::string sdc_pin_name = name;
        sdc_pin_name[colon_pos] = '/';
        targets.insert(std::move(sdc_pin_name));
    }
}

}  // namespace

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
    pin_clock_latency_overrides.assign(num_pins, nanf(""));
    output_delay_clock_by_pin_attr.assign(num_pins, {});
    for (auto& pin_clocks : output_delay_clock_by_pin_attr) {
        pin_clocks.fill("");
    }
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

    std::unordered_map<int, const Clock*> net_to_clock;
    pin_clock_periods.assign(num_pins, default_period);
    pin_clock_rise_edges.assign(num_pins, default_rise_edge);
    pin_clock_fall_edges.assign(num_pins, default_fall_edge);
    pin_clock_slews.assign(num_pins * NUM_ATTR, nanf(""));
    const Clock* default_clock = clocks.empty() ? nullptr : &clocks.begin()->second;
    std::vector<const Clock*> pin_clock_context(num_pins, default_clock);
    std::vector<char> pin_clock_context_is_default(num_pins, default_clock ? 1 : 0);
    auto apply_clock_transition_to_pin = [&](int pin_id, const Clock& clock) {
        if (auto transition_iter = clock_transitions.find(clock.name());
            transition_iter != clock_transitions.end()) {
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                const float transition = transition_iter->second[attr];
                if (std::isfinite(transition)) {
                    timing_raw_db.pinSlew[pin_id][attr] = transition;
                    pin_clock_slews[pin_id * NUM_ATTR + attr] = transition;
                }
            }
        }
    };
    for (auto& [clock_name, clock] : clocks) {
        if (clock.source_id() < 0 ||
            clock.source_id() >= static_cast<int>(gpdb.getPins().size())) {
            continue;
        }
        const int net_id = gpdb.getPins()[clock.source_id()].getParNetId();
        if (net_id >= 0) {
            net_to_clock[net_id] = &clock;
        }
        pin_clock_periods[clock.source_id()] = clock.period();
        pin_clock_rise_edges[clock.source_id()] = clock.rise_edge();
        pin_clock_fall_edges[clock.source_id()] = clock.fall_edge();
        pin_clock_context[clock.source_id()] = &clock;
        pin_clock_context_is_default[clock.source_id()] = 0;
        logger.info("clock: %s, source_pin: %s, period: %.2f, rise_edge: %.3f, fall_edge: %.3f",
                    clock_name.c_str(),
                    gpdb.getPins()[clock.source_id()].getName().c_str(),
                    clock.period(),
                    clock.rise_edge(),
                    clock.fall_edge());
    }
    for (auto& gppin : gpdb.getPins()) {
        const int net_id = gppin.getParNetId();
        auto iter = net_to_clock.find(net_id);
        if (iter != net_to_clock.end()) {
            const int pin_id = gppin.getId();
            const Clock* clock = iter->second;
            pin_clock_periods[pin_id] = clock->period();
            pin_clock_rise_edges[pin_id] = clock->rise_edge();
            pin_clock_fall_edges[pin_id] = clock->fall_edge();
            pin_clock_context[pin_id] = clock;
            pin_clock_context_is_default[pin_id] = 0;
            apply_clock_transition_to_pin(pin_id, *clock);
        }
    }

    int pin_clock_latency_override_count = 0;
    int pin_clock_latency_with_clock_context = 0;
    int pin_clock_latency_with_default_context = 0;
    int pin_clock_latency_without_context = 0;
    for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
        const float override = pin_clock_latency_overrides[pin_id];
        if (!std::isfinite(override)) {
            continue;
        }
        ++pin_clock_latency_override_count;
        const Clock* clock = pin_clock_context[pin_id];
        if (clock) {
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
        for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
            if (!pin_is_clk[pin_id]) {
                continue;
            }
            apply_clock_transition_to_pin(pin_id, *default_clock);
        }
    }

    test_clock_periods.assign(num_tests, default_period);
    test_setup_uncertainties.assign(num_tests, 0.0f);
    test_hold_uncertainties.assign(num_tests, 0.0f);
    int tests_with_clock = 0;
    for (int test_id = 0; test_id < static_cast<int>(test_id2_arc_id.size()); ++test_id) {
        const int arc_id = test_id2_arc_id[test_id];
        if (arc_id < 0 || arc_id >= static_cast<int>(timing_arc_from_pin_id.size())) {
            continue;
        }
        const int clock_pin_id = timing_arc_from_pin_id[arc_id];
        const int net_id = gpdb.getPins()[clock_pin_id].getParNetId();
        auto iter = net_to_clock.find(net_id);
        if (iter == net_to_clock.end()) {
            continue;
        }
        const Clock* clock = iter->second;
        test_clock_periods[test_id] = clock->period();
        if (auto setup_iter = clock_setup_uncertainty.find(clock->name());
            setup_iter != clock_setup_uncertainty.end()) {
            test_setup_uncertainties[test_id] = setup_iter->second;
        }
        if (auto hold_iter = clock_hold_uncertainty.find(clock->name());
            hold_iter != clock_hold_uncertainty.end()) {
            test_hold_uncertainties[test_id] = hold_iter->second;
        }
        ++tests_with_clock;
    }
    logger.info("Mapped %d/%d timing tests to capture clocks", tests_with_clock, num_tests);

    net_is_clock.resize(gpdb.getNets().size(), 0);
    for (auto& [net_id, clock] : net_to_clock) {
        (void) clock;
        if (net_id >= 0 && net_id < static_cast<int>(net_is_clock.size())) {
            net_is_clock[net_id] = 1;
        }
    }

    // set nan slew of PIs to half period
    for (auto& pi : primary_inputs) {
        if (torch::isnan(timing_raw_db.pinSlew[pi][0]).item<bool>()) timing_raw_db.pinSlew[pi][0] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][1]).item<bool>()) timing_raw_db.pinSlew[pi][1] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][2]).item<bool>()) timing_raw_db.pinSlew[pi][2] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][3]).item<bool>()) timing_raw_db.pinSlew[pi][3] = 0.0f;
        // if (torch::isnan(pinAT[pi][0]).item<bool>()) pinAT[pi][0] = 0.0f;
        // if (torch::isnan(pinAT[pi][1]).item<bool>()) pinAT[pi][1] = period / 2.0;
        // if (torch::isnan(pinAT[pi][2]).item<bool>()) pinAT[pi][2] = 0.0f;
        // if (torch::isnan(pinAT[pi][3]).item<bool>()) pinAT[pi][3] = period / 2.0;
    }

    for (auto& [clock_name, clock] : clocks) {
        if (clock.source_id() == -1) {
            continue;
        }
        int clock_pin_id = clock.source_id();
        pin_is_clk[clock_pin_id] = 1;
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][0]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][0] = clock.rise_edge();
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][1]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][1] = clock.fall_edge();
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][2]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][2] = clock.rise_edge();
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][3]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][3] = clock.fall_edge();
    }

    pin_is_ideal_clk.assign(num_pins, 0);
    int ideal_clock_pin_count = 0;
    int propagated_clock_pin_count = 0;
    int direct_propagated_clock_pin_count = 0;
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
    logger.info("Clock propagation: %d ideal clock pins, %d propagated clock pins (%d direct pin matches)",
                ideal_clock_pin_count,
                propagated_clock_pin_count,
                direct_propagated_clock_pin_count);

    auto device = timing_raw_db.node_size_x.device();
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
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
void GTDatabase::_read_sdc(sdc::SetUnits& obj) {
    if (obj.time.has_value()) {
        auto s = *obj.time;
        if (s == "ps") sdc_time_unit = 1e-12;
        if (s == "ns") sdc_time_unit = 1e-9;
        if (s == "us") sdc_time_unit = 1e-6;
        if (s == "ms") sdc_time_unit = 1e-3;
        if (s == "s") sdc_time_unit = 1.0;
    }
    if (obj.capacitance.has_value()) {
        auto s = *obj.capacitance;
        if (s == "fF") sdc_cap_unit = 1e-15;
        if (s == "pF") sdc_cap_unit = 1e-12;
        if (s == "nF") sdc_cap_unit = 1e-9;
        if (s == "uF") sdc_cap_unit = 1e-6;
        if (s == "F") sdc_cap_unit = 1.0;
    }
    if (obj.resistance.has_value()) {
        auto s = *obj.resistance;
        if (s == "Ohm") sdc_res_unit = 1.0;
        if (s == "kOhm") sdc_res_unit = 1e3;
        if (s == "MOhm") sdc_res_unit = 1e6;
    }
    if (gputimer_env_enabled("GPUTIMER_VERBOSE_SDC_UNITS")) {
        if (sdc_time_unit.has_value()) printf("sdc time unit: %.2E\n", *sdc_time_unit);
        if (sdc_cap_unit.has_value()) printf("sdc capacitance unit: %.2E\n", *sdc_cap_unit);
        if (sdc_res_unit.has_value()) printf("sdc resistance unit: %.2E\n", *sdc_res_unit);
    }
}

// Sets input delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetInputDelay& obj) {
    assert(obj.delay_value && obj.port_pin_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);
    float clock_edge = 0.0f;
    if (!obj.clock.empty()) {
        if (auto clock_itr = clocks.find(obj.clock); clock_itr != clocks.end()) {
            clock_edge = obj.clock_fall ? clock_itr->second.fall_edge()
                                        : clock_itr->second.rise_edge();
        } else {
            warn_missing_sdc_object(obj.command, "clock", obj.clock);
        }
    }
    auto input_arrival = [&]() {
        float delay = *obj.delay_value;
        if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
        return clock_edge + delay;
    };

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    timing_raw_db.pinAT[pi][(el << 1) + rf] = input_arrival();
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        timing_raw_db.pinAT[itr->second][(el << 1) + rf] = input_arrival();
                                    }
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", port);
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_pin_list);
}

// Sets input transition on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetInputTransition& obj) {
    assert(obj.transition && obj.port_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float transition = *obj.transition;
                                    if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinSlew[pi][(el << 1) + rf] = transition;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float transition = *obj.transition;
                                        if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinSlew[itr->second][(el << 1) + rf] = transition;
                                    }
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", port);
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_list);
}

// Sets input transition on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetDrivingCell& obj) {
    assert((obj.transitions[0] || obj.transitions[1]) && obj.port_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    auto transition_for_rf = [&](int rf) {
        float transition = obj.transitions[rf].value_or(obj.transitions[rf ^ 1].value_or(0.0f));
        if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
        return transition;
    };

    auto timing_id_for_arc = [&](TimingArc* timing_arc) {
        for (int i = 0; i < static_cast<int>(liberty_timing_arcs.size()); ++i) {
            if (liberty_timing_arcs[i] == timing_arc) return i;
        }
        const int timing_id = static_cast<int>(liberty_timing_arcs.size());
        liberty_timing_arcs.push_back(timing_arc);
        return timing_id;
    };

    auto set_source_lane = [&](int pin_id, int attr, int timing_id, int input_rf, float input_slew) {
        auto iter = std::find_if(driving_cell_sources.begin(), driving_cell_sources.end(),
                                 [pin_id](const DrivingCellSource& source) {
                                     return source.pin_id == pin_id;
                                 });
        if (iter == driving_cell_sources.end()) {
            DrivingCellSource source;
            source.pin_id = pin_id;
            driving_cell_sources.push_back(source);
            iter = std::prev(driving_cell_sources.end());
        }
        iter->timing_ids[attr] = timing_id;
        iter->input_rfs[attr] = input_rf;
        iter->input_slews[attr] = input_slew;
    };

    auto record_driving_cell_source = [&](int pin_id) {
        if (!obj.lib_cell || !obj.pin) {
            return;
        }

        for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
            LibertyCell* liberty_cell = cell_libs_[el]->get_cell(*obj.lib_cell);
            if (!liberty_cell) {
                warn_missing_sdc_object(obj.command, "lib_cell", *obj.lib_cell);
                continue;
            }

            const int port_id = liberty_cell->get_port(*obj.pin);
            if (port_id < 0 || port_id >= static_cast<int>(liberty_cell->ports_.size())) {
                if (gputimer_env_enabled("GPUTIMER_VERBOSE_SDC_WARNINGS")) {
                    std::fprintf(stderr, "%s: pin %s not found in lib_cell %s\n",
                                 obj.command, obj.pin->c_str(), obj.lib_cell->c_str());
                }
                continue;
            }

            LibertyPort* output_port = liberty_cell->ports_[port_id];
            const int output_rf = static_cast<int>(rf);
            const int attr = (static_cast<int>(el) << 1) + output_rf;

            TimingArc* selected_arc = nullptr;
            int selected_input_rf = -1;
            for (TimingArc* timing_arc : output_port->timing_arcs_non_cond_non_bundle_) {
                if (!timing_arc || is_redundant_timing(timing_arc, el) || timing_arc->is_constraint()) {
                    continue;
                }
                if (!timing_arc->transition_[output_rf] || !timing_arc->transition_[output_rf]->set_) {
                    continue;
                }
                for (auto input_rf : TRAN) {
                    const int input_rf_int = static_cast<int>(input_rf);
                    if (is_transition_defined_cpu(timing_arc, input_rf_int, output_rf)) {
                        selected_arc = timing_arc;
                        selected_input_rf = input_rf_int;
                        break;
                    }
                }
                if (selected_arc) {
                    break;
                }
            }

            if (!selected_arc) {
                if (gputimer_env_enabled("GPUTIMER_VERBOSE_SDC_WARNINGS")) {
                    std::fprintf(stderr, "%s: no transition arc found for %s/%s attr %d\n",
                                 obj.command, obj.lib_cell->c_str(), obj.pin->c_str(), attr);
                }
                continue;
            }

            set_source_lane(pin_id,
                            attr,
                            timing_id_for_arc(selected_arc),
                            selected_input_rf,
                            transition_for_rf(selected_input_rf));
        }
    };

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float transition = transition_for_rf(rf);
                                    timing_raw_db.pinSlew[pi][(el << 1) + rf] = transition;
                                }
                                record_driving_cell_source(pi);
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float transition = transition_for_rf(rf);
                                        timing_raw_db.pinSlew[itr->second][(el << 1) + rf] = transition;
                                    }
                                    record_driving_cell_source(itr->second);
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", port);
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_list);
}

// Sets output delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetOutputDelay& obj) {
    assert(obj.delay_value && obj.port_pin_list);

    if (clocks.find(obj.clock) == clocks.end()) {
        warn_missing_sdc_object(obj.command, "clock", obj.clock);
        return;
    }

    auto& clock = clocks.at(obj.clock);
    const float clock_edge = obj.clock_fall ? clock.fall_edge() : clock.rise_edge();
    const float setup_uncertainty =
        clock_setup_uncertainty.count(obj.clock) ? clock_setup_uncertainty.at(obj.clock) : 0.0f;
    const float hold_uncertainty =
        clock_hold_uncertainty.count(obj.clock) ? clock_hold_uncertainty.at(obj.clock) : 0.0f;
    auto output_required = [&](Split el) {
        float delay = *obj.delay_value;
        if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
        return el == MIN ? clock_edge - delay + hold_uncertainty
                         : clock_edge + clock.period() - delay - setup_uncertainty;
    };

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllOutputs&) {
                            for (auto& po : primary_outputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
	                                    const int attr = (el << 1) + rf;
	                                    timing_raw_db.pinRAT[po][attr] = output_required(el);
	                                    output_delay_clock_by_pin_attr[po][attr] = obj.clock;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_output2pin_id.find(port); itr != primary_output2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
	                                        const int attr = (el << 1) + rf;
	                                        timing_raw_db.pinRAT[itr->second][attr] = output_required(el);
	                                        output_delay_clock_by_pin_attr[itr->second][attr] = obj.clock;
                                    }
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", port);
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_pin_list);
}

// Sets the load attribute to a specified value on specified ports and nets.
void GTDatabase::_read_sdc(sdc::SetLoad& obj) {
    assert(obj.value && obj.objects);

    auto mask = sdc::TimingMask(obj.min, obj.max, std::nullopt, std::nullopt);

    std::visit(Functors{[&](sdc::AllOutputs&) {
                            for (auto& po : primary_outputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float load = *obj.value;
                                    if (sdc_cap_unit.has_value()) load = load * *sdc_cap_unit / cap_unit;
                                    timing_raw_db.pinLoad[po][(el << 1) + rf] = load;
                                    pin_capacitance[6 * po + el * 2 + rf] = load;
                                    pin_capacitance[6 * po + 4 + el] = load;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_output2pin_id.find(port); itr != primary_output2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float load = *obj.value;
                                        if (sdc_cap_unit.has_value()) load = load * *sdc_cap_unit / cap_unit;
                                        timing_raw_db.pinLoad[itr->second][(el << 1) + rf] = load;
                                        pin_capacitance[6 * itr->second + el * 2 + rf] = load;
                                        pin_capacitance[6 * itr->second + 4 + el] = load;
                                    }
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", port);
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.objects);
}

void GTDatabase::_read_sdc(sdc::CreateClock& obj) {
    assert(obj.period && !obj.name.empty());

    // create clock from given sources
    if (obj.port_pin_list) {
        std::visit(Functors{[&](sdc::GetPorts& get_ports) {
                                auto& ports = get_ports.ports;
                                assert(ports.size() == 1);
                                if (auto itr = primary_input2pin_id.find(ports.front()); itr != primary_input2pin_id.end()) {
                                    auto [clock_iter, inserted] = clocks.try_emplace(obj.name, obj.name, itr->second, *obj.period);
                                    (void)inserted;
                                    if (obj.waveform) {
                                        clock_iter->second.set_waveform((*obj.waveform)[0], (*obj.waveform)[1]);
                                    }
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", ports.front());
                                }
                            },
                            [](auto&&) { assert(false); }},
                   *obj.port_pin_list);
    }
    // create virtual clock
    else {
        auto [clock_iter, inserted] = clocks.try_emplace(obj.name, obj.name, *obj.period);
        (void)inserted;
        if (obj.waveform) {
            clock_iter->second.set_waveform((*obj.waveform)[0], (*obj.waveform)[1]);
        }
    }
}

void GTDatabase::_read_sdc(sdc::SetClockUncertainty& obj) {
    if (!obj.uncertainty) {
        return;
    }

    float uncertainty = *obj.uncertainty;
    if (sdc_time_unit.has_value()) {
        uncertainty = uncertainty * *sdc_time_unit / time_unit;
    }

    const bool applies_to_hold = obj.hold.has_value();
    const bool applies_to_setup = obj.setup.has_value() || !obj.hold.has_value();

    auto apply_clock = [&](const std::string& clock_name) {
        if (clock_name.empty()) {
            return;
        }
        float setup_delta = 0.0f;
        float hold_delta = 0.0f;
        if (applies_to_setup) {
            const float old = clock_setup_uncertainty.count(clock_name)
                                  ? clock_setup_uncertainty.at(clock_name)
                                  : 0.0f;
            setup_delta = uncertainty - old;
            clock_setup_uncertainty[clock_name] = uncertainty;
        }
        if (applies_to_hold) {
            const float old = clock_hold_uncertainty.count(clock_name)
                                  ? clock_hold_uncertainty.at(clock_name)
                                  : 0.0f;
            hold_delta = uncertainty - old;
            clock_hold_uncertainty[clock_name] = uncertainty;
        }
        if ((applies_to_setup && setup_delta != 0.0f) ||
            (applies_to_hold && hold_delta != 0.0f)) {
            for (int pin_id = 0; pin_id < static_cast<int>(output_delay_clock_by_pin_attr.size()); ++pin_id) {
                for (int attr = 0; attr < NUM_ATTR; ++attr) {
                    if (output_delay_clock_by_pin_attr[pin_id][attr] != clock_name) {
                        continue;
                    }
                    if (torch::isnan(timing_raw_db.pinRAT[pin_id][attr]).item<bool>()) {
                        continue;
                    }
                    if (attr >= 2 && applies_to_setup) {
                        timing_raw_db.pinRAT[pin_id][attr] -= setup_delta;
                    } else if (attr < 2 && applies_to_hold) {
                        timing_raw_db.pinRAT[pin_id][attr] += hold_delta;
                    }
                }
            }
        }
    };

    if (!obj.object_list) {
        for (auto& [clock_name, clock] : clocks) {
            (void) clock;
            apply_clock(clock_name);
        }
        return;
    }

    std::visit(Functors{[&](sdc::AllClocks&) {
                            for (auto& [clock_name, clock] : clocks) {
                                (void) clock;
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetClocks& get_clocks) {
                            for (auto& clock_name : get_clocks.clocks) {
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            // The Tcl SDC bridge currently returns get_clocks as a plain
                            // space-separated object string, so it arrives here as GetPorts.
                            for (auto& clock_name : get_ports.ports) {
                                apply_clock(clock_name);
                            }
                        },
                        [](auto&&) {}},
               *obj.object_list);
}

void GTDatabase::_read_sdc(sdc::SetClockTransition& obj) {
    if (!obj.transition || !obj.clock_list) {
        return;
    }

    float transition = *obj.transition;
    if (sdc_time_unit.has_value()) {
        transition = transition * *sdc_time_unit / time_unit;
    }

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);
    auto apply_clock = [&](const std::string& clock_name) {
        if (clock_name.empty()) {
            return;
        }
        auto [iter, inserted] = clock_transitions.try_emplace(clock_name);
        if (inserted) {
            iter->second.fill(nanf(""));
        }
        auto& values = iter->second;
        for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
            values[(el << 1) + rf] = transition;
        }
    };

    std::visit(Functors{[&](sdc::AllClocks&) {
                            for (auto& [clock_name, clock] : clocks) {
                                (void)clock;
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetClocks& get_clocks) {
                            for (auto& clock_name : get_clocks.clocks) {
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& clock_name : get_ports.ports) {
                                apply_clock(clock_name);
                            }
                        },
                        [](auto&&) {}},
               *obj.clock_list);
}

void GTDatabase::_read_sdc(sdc::SetClockLatency& obj) {
    if (!obj.delay || !obj.object_list) {
        return;
    }

    float delay = *obj.delay;
    if (sdc_time_unit.has_value()) {
        delay = delay * *sdc_time_unit / time_unit;
    }

    auto apply_clock = [&](const std::string& clock_name) {
        if (auto itr = clocks.find(clock_name); itr != clocks.end()) {
            itr->second.set_latency(delay);
        }
    };
    auto apply_pin = [&](const std::string& pin_name) {
        if (auto itr = pin_name2pin_id.find(pin_name); itr != pin_name2pin_id.end()) {
            pin_clock_latency_overrides[itr->second] = delay;
        } else if (auto pi_itr = primary_input2pin_id.find(pin_name); pi_itr != primary_input2pin_id.end()) {
            pin_clock_latency_overrides[pi_itr->second] = delay;
        }
    };

    std::visit(Functors{[&](sdc::AllClocks&) {
                            for (auto& [clock_name, clock] : clocks) {
                                (void)clock;
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetClocks& get_clocks) {
                            for (auto& clock_name : get_clocks.clocks) {
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& name : get_ports.ports) {
                                if (clocks.find(name) != clocks.end()) {
                                    apply_clock(name);
                                } else {
                                    apply_pin(name);
                                }
                            }
                        },
                        [&](sdc::GetPins& get_pins) {
                            for (auto& pin_name : get_pins.pins) {
                                apply_pin(pin_name);
                            }
                        },
                        [](auto&&) {}},
               *obj.object_list);
}

void GTDatabase::_read_sdc(sdc::SetMaxTransition& obj) {
    // set_max_transition is a design-rule constraint in OpenSTA/OpenROAD. It does not
    // change path AT/RAT directly; parse it so benchmark SDCs no longer fall through as
    // unsupported while timing alignment is driven by delay constraints.
    (void)obj;
}

void GTDatabase::_read_sdc(sdc::SetCaseAnalysis& obj) {
    if (!obj.value || !obj.port_pin_list) {
        return;
    }
    const bool is_zero = *obj.value == "0" || *obj.value == "zero";
    const bool is_one = *obj.value == "1" || *obj.value == "one";
    if (!is_zero && !is_one) {
        return;
    }
    const int case_value = is_one ? 1 : 0;
    auto apply_pin = [&](int pin_id) {
        if (pin_id >= 0 && pin_id < static_cast<int>(pin_case_values.size())) {
            pin_case_values[pin_id] = case_value;
        }
    };
    std::visit(Functors{[&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    apply_pin(itr->second);
                                    for (int attr = 0; attr < NUM_ATTR; ++attr) {
                                        timing_raw_db.pinAT[itr->second][attr] = nanf("");
                                    }
                                }
                            }
                        },
                        [&](sdc::GetPins& get_pins) {
                            for (auto& pin_name : get_pins.pins) {
                                if (auto itr = pin_name2pin_id.find(pin_name); itr != pin_name2pin_id.end()) {
                                    apply_pin(itr->second);
                                }
                            }
                        },
                        [](auto&&) {}},
               *obj.port_pin_list);
}

void GTDatabase::_read_sdc(sdc::SetFalsePath& obj) {
    if (!obj.from) {
        return;
    }

    std::vector<std::string> from_clock_names;
    std::vector<std::string> to_clock_names;
    bool from_all_clocks = false;
    bool to_all_clocks = false;
    auto add_clock_name = [&](std::vector<std::string>& names, const std::string& name) {
        if (clocks.find(name) == clocks.end()) return;
        if (std::find(names.begin(), names.end(), name) == names.end())
            names.push_back(name);
    };
    auto collect_clocks = [&](const sdc::Object& object,
                              std::vector<std::string>& names,
                              bool& all_clocks) {
        std::visit(Functors{[&](const sdc::AllClocks&) {
                                all_clocks = true;
                            },
                            [&](const sdc::GetClocks& get_clocks) {
                                for (const auto& clock_name : get_clocks.clocks)
                                    add_clock_name(names, clock_name);
                            },
                            [&](const sdc::GetPorts& get_ports) {
                                // The bundled SDC Tcl bridge returns get_clocks objects
                                // as bare tokens, which parse_port() currently represents
                                // as GetPorts. Treat tokens matching known clock names as
                                // clock names for exception mapping.
                                for (const auto& token : get_ports.ports)
                                    add_clock_name(names, token);
                            },
                            [](auto&&) {}},
                   object);
    };
    collect_clocks(*obj.from, from_clock_names, from_all_clocks);
    if (obj.to) {
        collect_clocks(*obj.to, to_clock_names, to_all_clocks);
    } else {
        to_all_clocks = true;
    }
    if ((from_all_clocks || !from_clock_names.empty()) &&
        (to_all_clocks || !to_clock_names.empty())) {
        // Debug-only power experiment: OpenSTA/OpenROAD set_false_path is a
        // timing path exception, not a default power-activity cut.  Keep this
        // mask inert unless Power.cpp sees XPLACE_POWER_APPLY_FALSE_PATHS=1.
        if (power_disabled_constraint_arc.size() != timing_arc_from_pin_id.size()) {
            power_disabled_constraint_arc.assign(timing_arc_from_pin_id.size(), 0);
        }

        std::unordered_map<std::string, int> clock_name_to_id;
        clock_name_to_id.reserve(clocks.size());
        for (auto& [clock_name, clock] : clocks) {
            (void)clock;
            const int clock_id = static_cast<int>(clock_name_to_id.size());
            clock_name_to_id.emplace(clock_name, clock_id);
        }

        std::vector<int> pin_node_id(num_pins, -1);
        std::vector<int> pin_net_id(num_pins, -1);
        for (const auto& pin : gpdb.getPins()) {
            const int pin_id = static_cast<int>(pin.getId());
            if (pin_id < 0 || pin_id >= num_pins) continue;
            pin_node_id[pin_id] = static_cast<int>(pin.getParNodeId());
            pin_net_id[pin_id] = static_cast<int>(pin.getParNetId());
        }

        std::vector<int> clock_id_by_net(gpdb.getNets().size(), -1);
        std::vector<int> pin_clock_id(num_pins, -1);
        for (auto& [clock_name, clock] : clocks) {
            auto id_itr = clock_name_to_id.find(clock_name);
            if (id_itr == clock_name_to_id.end()) continue;
            const int clock_id = id_itr->second;
            const int source_pin = clock.source_id();
            if (source_pin >= 0 && source_pin < num_pins) {
                pin_clock_id[source_pin] = clock_id;
                const int net_id = pin_net_id[source_pin];
                if (net_id >= 0 && net_id < static_cast<int>(clock_id_by_net.size()))
                    clock_id_by_net[net_id] = clock_id;
            }
        }
        for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
            const int net_id = pin_net_id[pin_id];
            if (net_id >= 0 && net_id < static_cast<int>(clock_id_by_net.size()) &&
                clock_id_by_net[net_id] >= 0) {
                pin_clock_id[pin_id] = clock_id_by_net[net_id];
            }
        }

        std::vector<int> capture_clock_by_pin(num_pins, -1);
        std::vector<int> node_clock_id(gpdb.getNodes().size(), -1);
        for (int test_id = 0; test_id < static_cast<int>(test_id2_arc_id.size()); ++test_id) {
            const int arc_id = test_id2_arc_id[test_id];
            if (arc_id < 0 || arc_id >= static_cast<int>(timing_arc_from_pin_id.size()) ||
                arc_id >= static_cast<int>(timing_arc_to_pin_id.size())) {
                continue;
            }
            const int clock_pin = timing_arc_from_pin_id[arc_id];
            const int data_pin = timing_arc_to_pin_id[arc_id];
            if (clock_pin < 0 || clock_pin >= num_pins || data_pin < 0 || data_pin >= num_pins)
                continue;
            const int clock_id = pin_clock_id[clock_pin];
            if (clock_id < 0) continue;
            capture_clock_by_pin[data_pin] = clock_id;
            const int node_id = pin_node_id[data_pin];
            if (node_id >= 0 && node_id < static_cast<int>(node_clock_id.size())) {
                if (node_clock_id[node_id] < 0) node_clock_id[node_id] = clock_id;
                else if (node_clock_id[node_id] != clock_id) node_clock_id[node_id] = -2;
            }
        }

        auto get_cell = [&](int node_id) -> LibertyCell* {
            if (node_id < 0 || node_id >= static_cast<int>(cell_node_type_map.size())) return nullptr;
            const int cell_type_id = cell_node_type_map[node_id];
            if (cell_type_id < 0 || cell_type_id >= static_cast<int>(rawdb.celltypes.size())) return nullptr;
            db::CellType* cell_type = rawdb.celltypes[cell_type_id];
            return cell_type ? cell_type->liberty_cell : nullptr;
        };
        std::vector<uint8_t> is_seq_output_pin(num_pins, 0);
        std::vector<int> launch_clock_by_pin(num_pins, -1);
        for (const auto& node : gpdb.getNodes()) {
            const int node_id = static_cast<int>(node.getId());
            LibertyCell* cell = get_cell(node_id);
            if (!cell || cell->sequentials_.empty()) continue;
            const int node_clock = node_id >= 0 && node_id < static_cast<int>(node_clock_id.size())
                ? node_clock_id[node_id] : -1;
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= num_pins) continue;
                const int port_offset = pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || port->direction_ != CellPortDirection::output) continue;
                is_seq_output_pin[pin_id] = 1;
                if (node_clock >= 0) launch_clock_by_pin[pin_id] = node_clock;
            }
        }

        auto clock_list_matches = [&](const std::vector<std::string>& names, bool all_clocks, int clock_id) {
            if (clock_id < 0) return false;
            if (all_clocks) return true;
            for (const std::string& name : names) {
                auto iter = clock_name_to_id.find(name);
                if (iter != clock_name_to_id.end() && iter->second == clock_id)
                    return true;
            }
            return false;
        };
        auto power_edge_valid = [&](int arc_id) {
            if (arc_id < 0 || arc_id >= static_cast<int>(timing_arc_to_pin_id.size()) ||
                arc_id >= static_cast<int>(timing_arc_from_pin_id.size())) {
                return false;
            }
            if (arc_id < static_cast<int>(arc_id2test_id.size()) && arc_id2test_id[arc_id] != -1)
                return false;
            const int to_pin = timing_arc_to_pin_id[arc_id];
            if (to_pin < 0 || to_pin >= num_pins) return false;
            if (arc_id < static_cast<int>(arc_types.size()) && arc_types[arc_id] == 1 &&
                is_seq_output_pin[to_pin]) {
                return false;
            }
            return true;
        };

        std::vector<uint8_t> target_pin(num_pins, 0);
        std::deque<int> queue;
        for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
            if (!clock_list_matches(to_clock_names,
                                    to_all_clocks,
                                    capture_clock_by_pin[pin_id])) {
                continue;
            }
            target_pin[pin_id] = 1;
        }

        std::vector<uint8_t> forward_seen(num_pins, 0);
        for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
            if (!clock_list_matches(from_clock_names,
                                    from_all_clocks,
                                    launch_clock_by_pin[pin_id])) {
                continue;
            }
            if (forward_seen[pin_id]) continue;
            forward_seen[pin_id] = 1;
            queue.push_back(pin_id);
        }
        while (!queue.empty()) {
            const int pin_id = queue.front();
            queue.pop_front();
            if (pin_id < 0 || pin_id + 1 >= static_cast<int>(pin_forward_arc_list_end.size()))
                continue;
            for (int idx = pin_forward_arc_list_end[pin_id];
                 idx < pin_forward_arc_list_end[pin_id + 1]; ++idx) {
                const int arc_id = pin_forward_arc_list[idx];
                if (!power_edge_valid(arc_id)) continue;
                const int to_pin = timing_arc_to_pin_id[arc_id];
                if (to_pin < 0 || to_pin >= num_pins)
                    continue;
                if (!forward_seen[to_pin]) {
                    forward_seen[to_pin] = 1;
                    queue.push_back(to_pin);
                }
            }
        }

        int disabled_arc_count = 0;
        int disabled_net_arc_count = 0;
        for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
            if (!target_pin[pin_id] || pin_id + 1 >= static_cast<int>(pin_backward_arc_list_end.size()))
                continue;
            for (int idx = pin_backward_arc_list_end[pin_id];
                 idx < pin_backward_arc_list_end[pin_id + 1]; ++idx) {
                const int arc_id = pin_backward_arc_list[idx];
                if (!power_edge_valid(arc_id)) continue;
                const int from_pin = timing_arc_from_pin_id[arc_id];
                if (from_pin < 0 || from_pin >= num_pins || !forward_seen[from_pin])
                    continue;
                if (!power_disabled_constraint_arc[arc_id]) {
                    power_disabled_constraint_arc[arc_id] = 1;
                    ++disabled_arc_count;
                    if (arc_id < static_cast<int>(arc_types.size()) && arc_types[arc_id] == 0)
                        ++disabled_net_arc_count;
                }
            }
        }
        logger.info("Mapped debug-only SDC clock false-path power mask to %d disabled arcs (%d net arcs); apply with XPLACE_POWER_APPLY_FALSE_PATHS=1",
                    disabled_arc_count,
                    disabled_net_arc_count);
    }

    std::visit(Functors{[&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for (int attr = 0; attr < NUM_ATTR; ++attr) {
                                        timing_raw_db.pinAT[itr->second][attr] = nanf("");
                                    }
                                }
                            }
                        },
                        [](auto&&) {}},
               *obj.from);
}

void GTDatabase::_read_sdc(sdc::SetPropagatedClock& obj) {
    if (!obj.object_list) {
        return;
    }

    auto add_clock_name = [&](const std::string& clock_name) {
        if (clock_name == "*") {
            propagated_all_clocks = true;
            return;
        }
        if (!clock_name.empty()) {
            propagated_clock_names.insert(clock_name);
        }
    };
    auto add_pin = [&](const std::string& pin_name) {
        if (auto itr = pin_name2pin_id.find(pin_name); itr != pin_name2pin_id.end()) {
            propagated_clock_pins.insert(itr->second);
        } else if (auto pi_itr = primary_input2pin_id.find(pin_name); pi_itr != primary_input2pin_id.end()) {
            propagated_clock_pins.insert(pi_itr->second);
        } else {
            warn_missing_sdc_object(obj.command, "pin", pin_name);
        }
    };
    auto add_source_port_clock = [&](const std::string& port_name) {
        auto pin_itr = primary_input2pin_id.find(port_name);
        if (pin_itr == primary_input2pin_id.end()) {
            return;
        }
        const int source_pin_id = pin_itr->second;
        bool matched_clock = false;
        for (auto& [clock_name, clock] : clocks) {
            if (clock.source_id() == source_pin_id) {
                add_clock_name(clock_name);
                matched_clock = true;
            }
        }
        if (!matched_clock) {
            warn_missing_sdc_object(obj.command, "clock source port", port_name);
        }
    };

    std::visit(Functors{[&](sdc::AllClocks&) {
                            propagated_all_clocks = true;
                        },
                        [&](sdc::GetClocks& get_clocks) {
                            for (auto& clock_name : get_clocks.clocks) {
                                add_clock_name(clock_name);
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& token : get_ports.ports) {
                                // The bundled Tcl bridge returns get_clocks as bare
                                // tokens, so keep the token as a possible clock name
                                // and also map it as a clock source port when possible.
                                add_clock_name(token);
                                add_source_port_clock(token);
                            }
                        },
                        [&](sdc::GetPins& get_pins) {
                            for (auto& pin_name : get_pins.pins) {
                                add_pin(pin_name);
                            }
                        },
                        [](auto&&) {}},
               *obj.object_list);
}

void GTDatabase::_read_sdc(sdc::SetIdealNetwork& obj) {
    // OpenSTA/OpenROAD parses set_ideal_network but ignores it.  It is not the
    // command that makes clocks ideal; clocks are ideal by default until
    // set_propagated_clock is applied.
    (void)obj;
}

}  // namespace gt
