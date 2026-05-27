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

}  // namespace gt
