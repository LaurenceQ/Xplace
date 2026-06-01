#include "PowerActivityCpuDebug.h"

#include "gputimer/core/power/common/PowerActivityHostUtils.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <string>
#include <unordered_set>

namespace gt {

void dumpPowerActivityCpuTracePaths(GTDatabase& gtdb,
                                    const char* trace_path_out_env,
                                    int n,
                                    const std::vector<uint8_t>& actual_seed_seen,
                                    const std::vector<int>& pin_level,
                                    const std::vector<int>& pin_to_node,
                                    const std::vector<uint8_t>& is_driver_pin,
                                    const std::vector<std::vector<PowerTraceEdge>>& seq_reverse_edges) {
        if (!trace_path_out_env || trace_path_out_env[0] == '\0') return;
        std::vector<int> target_pins =
            resolvePowerPathTargetPins(readPowerPathTargetQueries(), gtdb.pin_names);
        std::unordered_set<std::string> or_seed_roots =
            readOpenroadSeedRootNames(std::getenv("XPLACE_POWER_OPENROAD_ROOTS_FILE"));
        std::vector<uint8_t> common_seed(n, 0);
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (!actual_seed_seen[pin_id]) continue;
            const std::string name = normalizePowerPathName(gtdb.pin_names[pin_id]);
            if (or_seed_roots.empty() || or_seed_roots.count(name)) common_seed[pin_id] = 1;
        }
        auto valid_power_arc = [&](int arc_id, int from_pin, int to_pin) -> bool {
            if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) return false;
            if (arc_id < static_cast<int>(gtdb.arc_id2test_id.size()) && gtdb.arc_id2test_id[arc_id] != -1)
                return false;
            if (to_pin < 0 || to_pin >= n || from_pin < 0 || from_pin >= n) return false;
            if (std::getenv("XPLACE_POWER_ACTIVITY_SKIP_BACK_LEVEL_ARCS")
                && arc_id < static_cast<int>(gtdb.arc_types.size()) && gtdb.arc_types[arc_id] == 1
                && pin_level[to_pin] <= pin_level[from_pin])
                return false;
            if (arc_id < static_cast<int>(gtdb.arc_types.size()) && gtdb.arc_types[arc_id] == 1) {
                int to_node = pin_to_node[to_pin];
                LibertyCell* to_cell = powerCellForNode(gtdb, to_node);
                if (to_cell && !to_cell->sequentials_.empty() && is_driver_pin[to_pin])
                    return false;
            }
            return true;
        };
    
        std::ofstream tsv(trace_path_out_env);
        if (!tsv) return;
        tsv << "path_id\tstep\ttarget_pin_id\ttarget_pin\tseed_pin_id\tseed_pin"
            << "\tarc_id\tfrom_pin_id\tfrom_pin\tto_pin_id\tto_pin"
            << "\tedge_kind\tfrom_level\tto_level\n";
        std::string json_path;
        if (const char* json_env = std::getenv("XPLACE_POWER_TRACE_PATH_JSON")) {
            json_path = json_env;
        } else {
            json_path = trace_path_out_env;
            const size_t dot = json_path.find_last_of('.');
            if (dot == std::string::npos) json_path += ".json";
            else json_path.replace(dot, std::string::npos, ".json");
        }
        std::ofstream json(json_path);
        if (json) json << "{\n  \"paths\": [\n";
        bool first_json_path = true;
        int path_id = 0;
        for (int target_pin : target_pins) {
            if (target_pin < 0 || target_pin >= n) continue;
            std::vector<int> pred_pin(n, -2);
            std::vector<int> pred_arc(n, -1);
            std::vector<std::string> pred_reason(n);
            std::queue<int> queue;
            pred_pin[target_pin] = -1;
            queue.push(target_pin);
            int seed_pin = -1;
            while (!queue.empty()) {
                const int pin_id = queue.front();
                queue.pop();
                if (common_seed[pin_id]) {
                    seed_pin = pin_id;
                    break;
                }
                if (pin_id >= 0 && pin_id + 1 < static_cast<int>(gtdb.pin_backward_arc_list_end.size())) {
                    const int start = gtdb.pin_backward_arc_list_end[pin_id];
                    const int end = gtdb.pin_backward_arc_list_end[pin_id + 1];
                    for (int idx = start; idx < end; ++idx) {
                        const int arc_id = gtdb.pin_backward_arc_list[idx];
                        if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.timing_arc_from_pin_id.size())) continue;
                        const int from_pin = gtdb.timing_arc_from_pin_id[arc_id];
                        if (!valid_power_arc(arc_id, from_pin, pin_id)) continue;
                        if (pred_pin[from_pin] != -2) continue;
                        pred_pin[from_pin] = pin_id;
                        pred_arc[from_pin] = arc_id;
                        pred_reason[from_pin] = "power_arc";
                        queue.push(from_pin);
                    }
                }
                for (const PowerTraceEdge& edge : seq_reverse_edges[pin_id]) {
                    if (edge.from_pin < 0 || edge.from_pin >= n || pred_pin[edge.from_pin] != -2)
                        continue;
                    pred_pin[edge.from_pin] = pin_id;
                    pred_arc[edge.from_pin] = edge.arc_id;
                    pred_reason[edge.from_pin] = edge.reason;
                    queue.push(edge.from_pin);
                }
            }
            if (seed_pin < 0) {
                tsv << path_id++ << "\t-1\t" << target_pin << '\t'
                    << gtdb.pin_names[target_pin]
                    << "\t-1\t\t-1\t-1\t\t-1\t\tno_path\t-1\t-1\n";
                continue;
            }
            std::vector<int> path_pins;
            for (int pin_id = seed_pin; pin_id >= 0; pin_id = pred_pin[pin_id]) {
                path_pins.push_back(pin_id);
                if (pin_id == target_pin) break;
            }
            const int current_path = path_id++;
            if (json) {
                if (!first_json_path) json << ",\n";
                first_json_path = false;
                json << "    {\"path_id\": " << current_path
                     << ", \"seed_pin\": \"" << gtdb.pin_names[seed_pin]
                     << "\", \"target_pin\": \"" << gtdb.pin_names[target_pin]
                     << "\", \"steps\": " << (path_pins.size() > 1 ? path_pins.size() - 1 : 0)
                     << "}";
            }
            for (size_t step = 0; step + 1 < path_pins.size(); ++step) {
                const int from_pin = path_pins[step];
                const int to_pin = path_pins[step + 1];
                const int arc_id = pred_arc[from_pin];
                tsv << current_path << '\t' << step << '\t'
                    << target_pin << '\t' << gtdb.pin_names[target_pin] << '\t'
                    << seed_pin << '\t' << gtdb.pin_names[seed_pin] << '\t'
                    << arc_id << '\t'
                    << from_pin << '\t' << gtdb.pin_names[from_pin] << '\t'
                    << to_pin << '\t' << gtdb.pin_names[to_pin] << '\t'
                    << pred_reason[from_pin] << '\t'
                    << (from_pin < static_cast<int>(pin_level.size()) ? pin_level[from_pin] : -1) << '\t'
                    << (to_pin < static_cast<int>(pin_level.size()) ? pin_level[to_pin] : -1) << '\n';
            }
        }
        if (json) json << "\n  ]\n}\n";
        
}

}  // namespace gt
