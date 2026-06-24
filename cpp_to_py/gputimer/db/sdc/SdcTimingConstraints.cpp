#include "gputimer/db/GTDatabase.h"
#include "gputimer/db/sdc/SdcUtils.h"

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
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace gt {

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
        if (sdc_time_unit.has_value()) {
            XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_UNITS", "time_unit=%.2E", *sdc_time_unit);
        }
        if (sdc_cap_unit.has_value()) {
            XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_UNITS", "capacitance_unit=%.2E", *sdc_cap_unit);
        }
        if (sdc_res_unit.has_value()) {
            XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_UNITS", "resistance_unit=%.2E", *sdc_res_unit);
        }
    }
}

// Sets input delay on pins or input ports relative to a clock signal.
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
                                    hostPinAT(pi, (el << 1) + rf) = input_arrival();
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        hostPinAT(itr->second, (el << 1) + rf) = input_arrival();
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
// Sets input transition on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetInputTransition& obj) {
    assert(obj.transition && obj.port_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float transition = *obj.transition;
                                    if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                    hostPinSlew(pi, (el << 1) + rf) = transition;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float transition = *obj.transition;
                                        if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                        hostPinSlew(itr->second, (el << 1) + rf) = transition;
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
                XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_WARNINGS",
                              "%s: pin %s not found in lib_cell %s",
                              obj.command, obj.pin->c_str(), obj.lib_cell->c_str());
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
                XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_WARNINGS",
                              "%s: no transition arc found for %s/%s attr %d",
                              obj.command, obj.lib_cell->c_str(), obj.pin->c_str(), attr);
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
                                    hostPinSlew(pi, (el << 1) + rf) = transition;
                                }
                                record_driving_cell_source(pi);
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float transition = transition_for_rf(rf);
                                        hostPinSlew(itr->second, (el << 1) + rf) = transition;
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
	                                    hostPinRAT(po, attr) = output_required(el);
	                                    output_delay_clock_by_pin_attr[po][attr] = obj.clock;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_output2pin_id.find(port); itr != primary_output2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
	                                        const int attr = (el << 1) + rf;
	                                        hostPinRAT(itr->second, attr) = output_required(el);
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
// Sets the load attribute to a specified value on specified ports and nets.
void GTDatabase::_read_sdc(sdc::SetLoad& obj) {
    assert(obj.value && obj.objects);

    auto mask = sdc::TimingMask(obj.min, obj.max, std::nullopt, std::nullopt);

    std::visit(Functors{[&](sdc::AllOutputs&) {
                            for (auto& po : primary_outputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float load = *obj.value;
                                    if (sdc_cap_unit.has_value()) load = load * *sdc_cap_unit / cap_unit;
                                    hostPinLoad(po, (el << 1) + rf) = load;
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
                                        hostPinLoad(itr->second, (el << 1) + rf) = load;
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
void GTDatabase::_read_sdc(sdc::SetMaxTransition& obj) {
    // set_max_transition is a design-rule constraint in OpenSTA/OpenROAD. It does not
    // change path AT/RAT directly; parse it so benchmark SDCs no longer fall through as
    // unsupported while timing alignment is driven by delay constraints.
    (void)obj;
}

}  // namespace gt
