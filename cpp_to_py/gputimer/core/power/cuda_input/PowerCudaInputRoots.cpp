#include "PowerCudaInputBuildInternal.h"

#include "gputimer/core/power/common/PowerHostCommon.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>

namespace gt {

void dumpPowerCudaInputRoots(GTDatabase& gtdb,
                             int n,
                             const std::vector<int>& primary_inputs,
                             const std::vector<int>& candidate_roots,
                             const std::vector<std::string>& seed_reason,
                             const std::vector<uint8_t>& seed_seen,
                             const std::vector<uint8_t>& is_primary_input,
                             const std::vector<uint8_t>& is_clock_pin,
                             const std::vector<uint8_t>& is_driver_pin,
                             const std::vector<uint8_t>& is_load_pin,
                             const std::vector<int>& power_fanin,
                             const std::vector<int>& pin_to_node,
                             const std::vector<int>& pin_to_net,
                             const std::vector<int>& power_pin_level_cpu) {
    if (const char* root_dump_file = std::getenv("XPLACE_POWER_DUMP_ROOTS_FILE")) {
        if (root_dump_file[0] != '\0') {
            std::vector<uint8_t> h_candidate_seen(n, 0);
            for (int pin_id : candidate_roots) {
                if (pin_id >= 0 && pin_id < n) h_candidate_seen[pin_id] = 1;
            }
            std::vector<int> root_probe_pins =
                resolvePowerTracePins(readPowerRootProbePinQueries(), gtdb.pin_names);
            std::vector<int> dump_pins;
            dump_pins.reserve(primary_inputs.size() + candidate_roots.size() + root_probe_pins.size());
            dump_pins.insert(dump_pins.end(), primary_inputs.begin(), primary_inputs.end());
            dump_pins.insert(dump_pins.end(), candidate_roots.begin(), candidate_roots.end());
            dump_pins.insert(dump_pins.end(), root_probe_pins.begin(), root_probe_pins.end());
            std::sort(dump_pins.begin(), dump_pins.end());
            dump_pins.erase(std::unique(dump_pins.begin(), dump_pins.end()), dump_pins.end());
    
            std::ofstream root_dump(root_dump_file);
            if (root_dump) {
                root_dump
                    << "pin_id\tpin_name\tin_actual_seed\tin_candidate\treason"
                    << "\tis_primary\tis_clock\tis_driver\tis_load\tpower_fanin"
                    << "\ttiming_fanin\tpower_level\tnode_id\tinst_name\tcell_type"
                    << "\tnet_id\tnet_name\n";
                for (int pin_id : dump_pins) {
                    if (pin_id < 0 || pin_id >= n) continue;
                    std::string reason = seed_reason[pin_id];
                    if (reason.empty() && seed_seen[pin_id]) reason = "seed";
                    if (h_candidate_seen[pin_id]) {
                        if (reason.empty()) reason = "power_zero_fanin_candidate";
                        else if (reason.find("power_zero_fanin_candidate") == std::string::npos
                                 && reason.find("power_zero_fanin_seed") == std::string::npos)
                            reason += ";power_zero_fanin_candidate";
                    }
                    if (reason.empty()) reason = "probe_only";
                    const int node_id =
                        (pin_id < static_cast<int>(pin_to_node.size())) ? pin_to_node[pin_id] : -1;
                    const int net_id =
                        (pin_id < static_cast<int>(pin_to_net.size())) ? pin_to_net[pin_id] : -1;
                    std::string inst_name;
                    std::string cell_type;
                    if (node_id >= 0 && node_id < static_cast<int>(gtdb.gpdb.getNodes().size())) {
                        inst_name = gtdb.gpdb.getNodes()[node_id].getName();
                        cell_type = gtdb.gpdb.getNodes()[node_id].getCellTypeName();
                    }
                    std::string net_name;
                    if (net_id >= 0 && net_id < static_cast<int>(gtdb.net_names.size())) {
                        net_name = gtdb.net_names[net_id];
                    } else if (net_id >= 0 && net_id < static_cast<int>(gtdb.gpdb.getNets().size())) {
                        net_name = gtdb.gpdb.getNets()[net_id].getName();
                    }
                    const int timing_fanin =
                        (pin_id < static_cast<int>(gtdb.pin_num_fanin.size())) ? gtdb.pin_num_fanin[pin_id] : -1;
                    const int power_level =
                        (pin_id < static_cast<int>(power_pin_level_cpu.size())) ? power_pin_level_cpu[pin_id] : -1;
                    root_dump << pin_id << '\t' << gtdb.pin_names[pin_id] << '\t'
                              << (seed_seen[pin_id] ? 1 : 0) << '\t'
                              << (h_candidate_seen[pin_id] ? 1 : 0) << '\t'
                              << reason << '\t'
                              << (is_primary_input[pin_id] ? 1 : 0) << '\t'
                              << (is_clock_pin[pin_id] ? 1 : 0) << '\t'
                              << (is_driver_pin[pin_id] ? 1 : 0) << '\t'
                              << (is_load_pin[pin_id] ? 1 : 0) << '\t'
                              << power_fanin[pin_id] << '\t'
                              << timing_fanin << '\t' << power_level << '\t'
                              << node_id << '\t' << inst_name << '\t' << cell_type << '\t'
                              << net_id << '\t' << net_name << '\n';
                }
            }
        }
    }
}

}  // namespace gt
