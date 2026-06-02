#include "gputimer/db/GTDatabase.h"
#include "gputimer/db/sdc/SdcUtils.h"

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
            for (auto& [pin_id, pin_clocks] : output_delay_clock_by_pin_attr) {
                for (int attr = 0; attr < NUM_ATTR; ++attr) {
                    if (pin_clocks[attr] != clock_name) {
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
