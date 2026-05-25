#include "GPUTimer.h"

#include "DmpModel.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"
#include "common/lib/Liberty.h"
#include "common/lib/Lut.h"
#include "common/lib/Timing.h"
#include "gputimer/db/GTDatabase.h"
#include "io_parser/gp/GPDatabase.h"

#include <torch/cuda.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gt {

void clear_power_cuda_error();
void check_power_cuda_error(const char* label);

static bool readPowerBoolEnv(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env) return default_value;
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(value.empty() || value == "0" || value == "false" || value == "no");
}

static float readPowerFloatEnv(const char* name, float default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') return default_value;
    char* end = nullptr;
    const float value = std::strtof(env, &end);
    return (end != env && std::isfinite(value)) ? value : default_value;
}

static double canonicalPowerTimeScale(float scale) {
    const double value = static_cast<double>(scale);
    if (!std::isfinite(value) || value <= 0.0) return value;
    constexpr double known_scales[] = {1.0, 1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12, 1.0e-15};
    for (double known : known_scales) {
        if (std::abs(value - known) <= known * 1.0e-5) return known;
    }
    return value;
}

static float powerDensityForPeriod(double transitions, float period, double time_scale) {
    if (!std::isfinite(period) || period <= 0.0f ||
        !std::isfinite(time_scale) || time_scale <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(transitions / (static_cast<double>(period) * time_scale));
}

void run_power_activity_cuda_launcher(int n,
                                      const std::vector<int>& level_list_end_cpu,
                                      index_type* d_level_list,
                                      const int* d_pin_power_level,
                                      index_type* d_pin_forward_arc_list_end,
                                      index_type* d_pin_forward_arc_list,
                                      index_type* d_timing_arc_to_pin_id,
                                      int* d_arc_types,
                                      int* d_arc_id2test_id,
                                      const int* d_pin2net_map,
                                      const int* d_net_driver_pin,
                                      const int* d_flat_net2pin_start_map,
                                      const int* d_flat_net2pin_map,
                                      uint8_t* d_is_load_pin,
                                      uint8_t* d_is_driver_pin,
                                      uint8_t* d_is_cell_pin,
                                      uint8_t* d_is_seq_output_pin,
                                      int* d_clock_gate_out_for_input,
                                      int* d_clock_gate_clock_for_out,
                                      int* d_clock_gate_enable_for_out,
                                      int* d_primary_inputs,
                                      int num_primary_inputs,
                                      int* d_case_values,
                                      int* d_clock_pins,
                                      int num_clock_pins,
                                      const float* d_clock_pin_densities,
                                      const float* d_clock_pin_duties,
                                      const uint8_t* d_clock_pin_enqueue,
                                      GpuPowerExprOpHost* d_expr_ops,
                                      int* d_expr_start,
                                      int* d_expr_count,
                                      int* d_node_port_pin_start,
                                      int* d_node_port_pin_list,
                                      int* d_pin_func_expr_id,
                                      int* d_missing_func_out_start,
                                      int* d_missing_func_out_list,
                                      GpuPowerSeqHost* d_seqs,
                                      int num_seqs,
                                      int* d_pin_seq_list_start,
                                      int* d_pin_seq_list,
                                      int* d_feedback_seed_pins,
                                      int num_feedback_seed_pins,
                                      int* d_feedback_seed_seqs,
                                      int num_feedback_seed_seqs,
                                      float default_density,
                                      float clock_density,
                                      float time_unit,
                                      int max_activity_passes,
                                      int* d_trace_pins,
                                      int num_trace_pins,
                                      const float* d_precomputed_activity,
                                      float* d_out,
                                      int num_nodes,
                                      const int* d_pin2node_map,
                                      const float* d_pinLoad,
                                      const float* d_dmp_C1,
                                      const float* d_dmp_C2,
                                      const float* d_pinSlew,
                                      const float* d_power_clock_slews,
                                      bool allow_clock_activity_override,
                                      float min_activity_density,
                                      GpuPowerInternalHost* d_internal_rows,
                                      int num_internal_rows,
                                      int num_internal_denom_groups,
                                      GPUPowerLutAllocator* d_power_allocator,
                                      float cap_unit,
                                      float voltage,
                                      float* d_inst_switching,
                                      float* d_pin_switching,
                                      float* d_inst_internal,
                                      float* d_internal_row_power,
                                      GpuPowerLeakageRowHost* d_leakage_rows,
                                      int num_leakage_rows,
                                      GpuPowerLeakageGroupHost* d_leakage_groups,
                                      int num_leakage_groups,
                                      float* d_inst_leakage,
                                      float* d_leakage_row_power);

void run_power_internal_denom_chunk_cuda_launcher(int n,
                                                  const float* d_precomputed_activity,
                                                  GpuPowerInternalHost* d_internal_rows,
                                                  int num_internal_rows,
                                                  GpuPowerExprOpHost* d_expr_ops,
                                                  int* d_expr_start,
                                                  int* d_expr_count,
                                                  int* d_node_port_pin_start,
                                                  int* d_node_port_pin_list,
                                                  float* d_denom);

void run_power_internal_contrib_chunk_cuda_launcher(int n,
                                                    int num_nodes,
                                                    const float* d_precomputed_activity,
                                                    GpuPowerInternalHost* d_internal_rows,
                                                    int num_internal_rows,
                                                    GpuPowerExprOpHost* d_expr_ops,
                                                    int* d_expr_start,
                                                    int* d_expr_count,
                                                    int* d_node_port_pin_start,
                                                    int* d_node_port_pin_list,
                                                    const float* d_pinSlew,
                                                    const float* d_power_clock_slews,
                                                    const float* d_dmp_C1,
                                                    const float* d_dmp_C2,
                                                    const float* d_denom,
                                                    GPUPowerLutAllocator* d_power_allocator,
                                                    float cap_unit,
                                                    float* d_inst_internal,
                                                    float* d_internal_row_power);

void run_power_leakage_rows_chunk_cuda_launcher(int n,
                                                const float* d_precomputed_activity,
                                                GpuPowerLeakageRowHost* d_leakage_rows,
                                                int num_leakage_rows,
                                                GpuPowerExprOpHost* d_expr_ops,
                                                int* d_expr_start,
                                                int* d_expr_count,
                                                int* d_node_port_pin_start,
                                                int* d_node_port_pin_list,
                                                float* d_group_cond_leakage,
                                                float* d_group_cond_duty_sum,
                                                int* d_group_cond_count,
                                                float* d_leakage_row_power);

void run_power_leakage_summary_chunk_cuda_launcher(GpuPowerLeakageGroupHost* d_leakage_groups,
                                                   int num_leakage_groups,
                                                   float* d_group_cond_leakage,
                                                   float* d_group_cond_duty_sum,
                                                   int* d_group_cond_count,
                                                   int num_nodes,
                                                   float* d_inst_leakage);

static std::string normalizeTracePinName(std::string name) {
    name.erase(0, name.find_first_not_of(" \t\r\n"));
    size_t end = name.find_last_not_of(" \t\r\n");
    if (end == std::string::npos) return "";
    name.erase(end + 1);
    name.erase(std::remove(name.begin(), name.end(), '\\'), name.end());
    return name;
}

static std::string normalizePowerActivitySnapshotName(std::string name) {
    name.erase(0, name.find_first_not_of(" \t\r\n\""));
    size_t end = name.find_last_not_of(" \t\r\n\"");
    if (end == std::string::npos) return "";
    name.erase(end + 1);
    name.erase(std::remove(name.begin(), name.end(), '\\'), name.end());
    std::replace(name.begin(), name.end(), ':', '/');
    return name;
}

static std::string csvEscapePowerActivitySnapshot(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') escaped += "\"\"";
        else escaped += ch;
    }
    escaped += '"';
    return escaped;
}

static int readPowerActivitySnapshotMaxPass(const char* env_name, int default_value) {
    const char* env = std::getenv(env_name);
    if (!env || env[0] == '\0') return default_value;
    return std::max(0, std::atoi(env));
}

static std::vector<std::string> readPowerTracePinQueries() {
    std::vector<std::string> queries;
    auto add = [&](const std::string& raw) {
        std::string name = normalizeTracePinName(raw);
        if (!name.empty()) queries.push_back(name);
    };
    if (const char* file_name = std::getenv("XPLACE_POWER_ACTIVITY_TRACE_PIN_LIST_FILE")) {
        std::ifstream stream(file_name);
        std::string line;
        while (std::getline(stream, line)) add(line);
    }
    if (const char* trace_name = std::getenv("XPLACE_POWER_ACTIVITY_TRACE_PIN")) add(trace_name);
    if (const char* trace_list = std::getenv("XPLACE_POWER_ACTIVITY_TRACE_PINS")) {
        std::stringstream stream(trace_list);
        std::string item;
        while (std::getline(stream, item, ',')) add(item);
    }
    std::vector<std::string> unique;
    for (const std::string& query : queries) {
        if (std::find(unique.begin(), unique.end(), query) == unique.end()) unique.push_back(query);
    }
    return unique;
}

static std::vector<std::string> readPowerRootProbePinQueries() {
    std::vector<std::string> queries;
    auto add = [&](const std::string& raw) {
        std::string name = normalizeTracePinName(raw);
        if (!name.empty()) queries.push_back(name);
    };
    auto read_file = [&](const char* file_name) {
        if (!file_name || file_name[0] == '\0') return;
        std::ifstream stream(file_name);
        std::string line;
        while (std::getline(stream, line)) add(line);
    };
    read_file(std::getenv("XPLACE_POWER_ROOT_PROBE_PINS_FILE"));
    read_file(std::getenv("XPLACE_POWER_PROBE_PIN_LIST_FILE"));
    read_file(std::getenv("XPLACE_POWER_ACTIVITY_TRACE_PIN_LIST_FILE"));
    if (const char* trace_name = std::getenv("XPLACE_POWER_ROOT_PROBE_PIN")) add(trace_name);
    if (const char* trace_list = std::getenv("XPLACE_POWER_ROOT_PROBE_PINS")) {
        std::stringstream stream(trace_list);
        std::string item;
        while (std::getline(stream, item, ',')) add(item);
    }
    std::vector<std::string> unique;
    for (const std::string& query : queries) {
        if (std::find(unique.begin(), unique.end(), query) == unique.end()) unique.push_back(query);
    }
    return unique;
}

static std::vector<int> resolvePowerTracePins(const std::vector<std::string>& queries,
                                              const std::vector<std::string>& pin_names) {
    std::vector<int> pins;
    for (const std::string& query : queries) {
        int matched = -1;
        for (int i = 0; i < static_cast<int>(pin_names.size()); ++i) {
            std::string slash_name = pin_names[i];
            std::replace(slash_name.begin(), slash_name.end(), ':', '/');
            if (pin_names[i] == query || slash_name == query) {
                matched = i;
                break;
            }
        }
        if (matched >= 0) pins.push_back(matched);
    }
    return pins;
}

namespace {

struct CpuActivity {
    float density = 0.0f;
    float duty = 0.0f;
    int origin = 0;  // 0 unknown, 1 input, 2 clock, 3 propagated, 4 constant.
};

struct PowerTraceEdge {
    int arc_id = -1;
    int from_pin = -1;
    int to_pin = -1;
    std::string reason;
};

struct PowerTracePathState {
    std::unordered_set<int> pins;
    std::unordered_set<int> arcs;
    std::unordered_set<int> nodes;
    std::ofstream out;

    bool enabled() const { return out.good(); }
};

static bool parsePowerBool(const std::string& value) {
    std::string text = value;
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text == "1" || text == "true" || text == "yes" || text == "y";
}

static std::string normalizePowerPathName(std::string name) {
    name.erase(0, name.find_first_not_of(" \t\r\n\""));
    size_t end = name.find_last_not_of(" \t\r\n\"");
    if (end == std::string::npos) return "";
    name.erase(end + 1);
    name.erase(std::remove(name.begin(), name.end(), '\\'), name.end());
    std::replace(name.begin(), name.end(), ':', '/');
    return name;
}

static bool powerPathNameMatches(const std::string& pin_name, const std::string& query) {
    const std::string name = normalizePowerPathName(pin_name);
    const std::string needle = normalizePowerPathName(query);
    if (name.empty() || needle.empty()) return false;
    if (name == needle) return true;
    if (name.size() > needle.size() && name.compare(name.size() - needle.size(), needle.size(), needle) == 0) {
        const size_t prefix = name.size() - needle.size();
        return prefix == 0 || name[prefix - 1] == '/';
    }
    return false;
}

static std::vector<std::string> readPowerPathTargetQueries() {
    std::vector<std::string> queries;
    auto add = [&](const std::string& raw) {
        std::string name = normalizePowerPathName(raw);
        if (!name.empty() && std::find(queries.begin(), queries.end(), name) == queries.end())
            queries.push_back(name);
    };
    if (const char* file_name = std::getenv("XPLACE_POWER_TRACE_PATH_TARGETS_FILE")) {
        std::ifstream stream(file_name);
        std::string line;
        while (std::getline(stream, line)) add(line);
    }
    if (const char* list = std::getenv("XPLACE_POWER_TRACE_PATH_TARGETS")) {
        std::stringstream stream(list);
        std::string item;
        while (std::getline(stream, item, ',')) add(item);
    }
    if (queries.empty() && (std::getenv("XPLACE_POWER_TRACE_PATH_OUT")
                            || std::getenv("XPLACE_POWER_ACTIVITY_PATH_TRACE_FILE"))) {
        add("FE_OCPC470995_soc_qvalid/A");
        add("FE_OCPC470995_soc_qvalid/Z");
        add("FE_RC_95112_0/ZN");
    }
    return queries;
}

static std::unordered_set<std::string> readOpenroadSeedRootNames(const char* file_name) {
    std::unordered_set<std::string> roots;
    if (!file_name || file_name[0] == '\0') return roots;
    std::ifstream stream(file_name);
    if (!stream) return roots;
    std::string header;
    if (!std::getline(stream, header)) return roots;
    std::vector<std::string> cols;
    std::stringstream header_stream(header);
    std::string col;
    while (std::getline(header_stream, col, '\t')) cols.push_back(col);
    auto col_index = [&](const char* name) {
        for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
            if (cols[i] == name) return i;
        }
        return -1;
    };
    const int pin_col = col_index("pin_name");
    const int seeded_col = col_index("was_seeded");
    const int found_col = col_index("found");
    if (pin_col < 0 || seeded_col < 0) return roots;
    std::string line;
    while (std::getline(stream, line)) {
        std::vector<std::string> fields;
        std::stringstream row_stream(line);
        std::string field;
        while (std::getline(row_stream, field, '\t')) fields.push_back(field);
        if (pin_col >= static_cast<int>(fields.size()) || seeded_col >= static_cast<int>(fields.size()))
            continue;
        if (found_col >= 0 && found_col < static_cast<int>(fields.size())
            && !parsePowerBool(fields[found_col]))
            continue;
        if (!parsePowerBool(fields[seeded_col])) continue;
        const std::string name = normalizePowerPathName(fields[pin_col]);
        if (!name.empty()) roots.insert(name);
    }
    return roots;
}

static std::vector<int> resolvePowerPathTargetPins(const std::vector<std::string>& queries,
                                                   const std::vector<std::string>& pin_names) {
    std::vector<int> pins;
    for (const std::string& query : queries) {
        int matched = -1;
        for (int i = 0; i < static_cast<int>(pin_names.size()); ++i) {
            if (powerPathNameMatches(pin_names[i], query)) {
                matched = i;
                break;
            }
        }
        if (matched >= 0 && std::find(pins.begin(), pins.end(), matched) == pins.end())
            pins.push_back(matched);
    }
    return pins;
}

static PowerTracePathState loadPowerTracePathState(const char* trace_path_file,
                                                   const char* out_file,
                                                   const std::vector<int>& pin_to_node) {
    PowerTracePathState state;
    if (!out_file || out_file[0] == '\0') return state;
    if (trace_path_file && trace_path_file[0] != '\0') {
        std::ifstream stream(trace_path_file);
        std::string header;
        if (stream && std::getline(stream, header)) {
            std::vector<std::string> cols;
            std::stringstream hs(header);
            std::string col;
            while (std::getline(hs, col, '\t')) cols.push_back(col);
            auto col_index = [&](const char* name) {
                for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
                    if (cols[i] == name) return i;
                }
                return -1;
            };
            const int arc_col = col_index("arc_id");
            const int from_col = col_index("from_pin_id");
            const int to_col = col_index("to_pin_id");
            std::string line;
            while (std::getline(stream, line)) {
                std::vector<std::string> fields;
                std::stringstream rs(line);
                std::string field;
                while (std::getline(rs, field, '\t')) fields.push_back(field);
                auto read_int = [&](int idx) -> int {
                    if (idx < 0 || idx >= static_cast<int>(fields.size())) return -1;
                    char* end = nullptr;
                    long value = std::strtol(fields[idx].c_str(), &end, 10);
                    return end != fields[idx].c_str() ? static_cast<int>(value) : -1;
                };
                const int arc_id = read_int(arc_col);
                const int from_pin = read_int(from_col);
                const int to_pin = read_int(to_col);
                if (arc_id != -1) state.arcs.insert(arc_id);
                if (from_pin >= 0) state.pins.insert(from_pin);
                if (to_pin >= 0) state.pins.insert(to_pin);
            }
        }
    }
    for (int pin_id : state.pins) {
        if (pin_id >= 0 && pin_id < static_cast<int>(pin_to_node.size()) && pin_to_node[pin_id] >= 0)
            state.nodes.insert(pin_to_node[pin_id]);
    }
    state.out.open(out_file);
    if (state.out) {
        state.out << "engine\tpass\tlevel_tag\tevent\tarc_id\tfrom_pin_id\tfrom_pin"
                  << "\tto_pin_id\tto_pin\tdensity_old\tduty_old\tdensity_new"
                  << "\tduty_new\tchanged\tenqueued\tpending_seq\treason\n";
    }
    return state;
}

static bool evalPowerExprWithPortValues(const PowerExpr& expr,
                                        const std::vector<int8_t>& port_values,
                                        int8_t& value) {
    std::vector<int8_t> stack;
    stack.reserve(expr.ops().size());
    auto pop = [&]() -> int8_t {
        if (stack.empty()) return -1;
        int8_t v = stack.back();
        stack.pop_back();
        return v;
    };

    for (const auto& op : expr.ops()) {
        switch (op.opcode) {
            case PowerExprOpcode::port:
                if (op.port_id >= 0 && op.port_id < static_cast<int>(port_values.size()))
                    stack.push_back(port_values[op.port_id]);
                else
                    stack.push_back(-1);
                break;
            case PowerExprOpcode::const_zero:
                stack.push_back(0);
                break;
            case PowerExprOpcode::const_one:
                stack.push_back(1);
                break;
            case PowerExprOpcode::logical_not: {
                const int8_t a = pop();
                stack.push_back(a < 0 ? -1 : static_cast<int8_t>(!a));
                break;
            }
            case PowerExprOpcode::logical_and: {
                const int8_t b = pop();
                const int8_t a = pop();
                if (a == 0 || b == 0) stack.push_back(0);
                else if (a == 1 && b == 1) stack.push_back(1);
                else stack.push_back(-1);
                break;
            }
            case PowerExprOpcode::logical_or: {
                const int8_t b = pop();
                const int8_t a = pop();
                if (a == 1 || b == 1) stack.push_back(1);
                else if (a == 0 && b == 0) stack.push_back(0);
                else stack.push_back(-1);
                break;
            }
            case PowerExprOpcode::logical_xor: {
                const int8_t b = pop();
                const int8_t a = pop();
                if (a < 0 || b < 0) stack.push_back(-1);
                else stack.push_back(static_cast<int8_t>((a != 0) ^ (b != 0)));
                break;
            }
        }
    }

    if (stack.size() != 1 || stack.back() < 0) return false;
    value = stack.back();
    return true;
}

static bool evalPowerExprActivity(const PowerExpr& expr,
                                  const LibertyCell* cell,
                                  const gp::GPNode& node,
                                  const std::vector<CpuActivity>& pin_activity,
                                  float& density,
                                  float& duty,
                                  const std::unordered_map<int, int>* const_port_values = nullptr,
                                  const std::unordered_set<int>* zero_density_ports = nullptr) {
    std::vector<int8_t> fixed_port_values(cell ? cell->ports_.size() : 0, -1);
    if (const_port_values) {
        for (const auto& [port_id, value] : *const_port_values) {
            if (port_id >= 0 && port_id < static_cast<int>(fixed_port_values.size()))
                fixed_port_values[port_id] = static_cast<int8_t>(value ? 1 : 0);
        }
    }

    struct BddNode {
        int var = -1;
        int low = 0;
        int high = 0;
    };
    struct BddKey {
        int var = -1;
        int low = 0;
        int high = 0;
        bool operator==(const BddKey& other) const {
            return var == other.var && low == other.low && high == other.high;
        }
    };
    struct BddKeyHash {
        size_t operator()(const BddKey& key) const {
            size_t h = std::hash<int>{}(key.var);
            h ^= std::hash<int>{}(key.low + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= std::hash<int>{}(key.high + 0x9e3779b9 + (h << 6) + (h >> 2));
            return h;
        }
    };

    constexpr int one_edge = 0;
    constexpr int zero_edge = 1;
    constexpr int no_var = std::numeric_limits<int>::max();
    std::vector<BddNode> bdd_nodes;
    std::unordered_map<BddKey, int, BddKeyHash> unique_nodes;
    std::unordered_map<int, int> port_to_var;
    std::vector<int> var_ports;
    std::vector<float> var_duties;
    std::vector<float> var_densities;
    std::vector<uint8_t> var_has_pin;

    auto edge_id = [](int edge) { return edge >> 1; };
    auto edge_inv = [](int edge) { return (edge & 1) != 0; };
    auto edge_not = [](int edge) { return edge ^ 1; };

    auto make_node = [&](int var, int low, int high) {
        if (low == high) return low;
        bool result_inv = false;
        // CUDD keeps the then/high edge regular and moves that complement to
        // the returned edge. This preserves OpenROAD's recursive float
        // rounding for Boolean-diff duties.
        if (edge_inv(high)) {
            low = edge_not(low);
            high = edge_not(high);
            result_inv = true;
        }
        BddKey key{var, low, high};
        auto itr = unique_nodes.find(key);
        int id = 0;
        if (itr == unique_nodes.end()) {
            id = static_cast<int>(bdd_nodes.size()) + 1;
            bdd_nodes.push_back(BddNode{var, low, high});
            unique_nodes.emplace(key, id);
        } else {
            id = itr->second;
        }
        int edge = id << 1;
        return result_inv ? edge_not(edge) : edge;
    };

    auto top_var = [&](int edge) {
        const int id = edge_id(edge);
        return id == 0 ? no_var : bdd_nodes[id - 1].var;
    };

    auto cof_top = [&](int edge, int var, bool high_child) {
        const int id = edge_id(edge);
        if (id == 0 || bdd_nodes[id - 1].var != var) return edge;
        int child = high_child ? bdd_nodes[id - 1].high : bdd_nodes[id - 1].low;
        return edge_inv(edge) ? edge_not(child) : child;
    };

    struct ApplyKey {
        int op = 0;
        int left = 0;
        int right = 0;
        bool operator==(const ApplyKey& other) const {
            return op == other.op && left == other.left && right == other.right;
        }
    };
    struct ApplyKeyHash {
        size_t operator()(const ApplyKey& key) const {
            size_t h = std::hash<int>{}(key.op);
            h ^= std::hash<int>{}(key.left + 0x9e3779b9 + (h << 6) + (h >> 2));
            h ^= std::hash<int>{}(key.right + 0x9e3779b9 + (h << 6) + (h >> 2));
            return h;
        }
    };
    std::unordered_map<ApplyKey, int, ApplyKeyHash> apply_cache;
    std::function<int(int, int, int)> apply_bdd = [&](int op, int left, int right) -> int {
        if ((op == 0 || op == 1) && right < left) std::swap(left, right);
        ApplyKey key{op, left, right};
        auto cache_itr = apply_cache.find(key);
        if (cache_itr != apply_cache.end()) return cache_itr->second;
        const int left_id = edge_id(left);
        const int right_id = edge_id(right);
        if (left_id == 0 && right_id == 0) {
            const bool left_value = !edge_inv(left);
            const bool right_value = !edge_inv(right);
            bool value = false;
            if (op == 0) value = left_value && right_value;
            else if (op == 1) value = left_value || right_value;
            else value = left_value != right_value;
            const int result = value ? one_edge : zero_edge;
            apply_cache.emplace(key, result);
            return result;
        }
        const int var = std::min(top_var(left), top_var(right));
        const int low = apply_bdd(op, cof_top(left, var, false), cof_top(right, var, false));
        const int high = apply_bdd(op, cof_top(left, var, true), cof_top(right, var, true));
        const int result = make_node(var, low, high);
        apply_cache.emplace(key, result);
        return result;
    };

    std::unordered_map<long long, int> restrict_cache;
    std::function<int(int, int, bool)> restrict_var = [&](int edge, int target_var, bool high_child) -> int {
        const int id = edge_id(edge);
        if (id == 0) return edge;
        const auto& bdd_node = bdd_nodes[id - 1];
        if (bdd_node.var > target_var) return edge;
        const long long key = (static_cast<long long>(edge) << 32)
            ^ (static_cast<long long>(target_var) << 1)
            ^ static_cast<long long>(high_child ? 1 : 0);
        auto cache_itr = restrict_cache.find(key);
        if (cache_itr != restrict_cache.end()) return cache_itr->second;
        int result = edge;
        if (bdd_node.var == target_var) {
            result = high_child ? bdd_node.high : bdd_node.low;
        } else {
            const int low = restrict_var(bdd_node.low, target_var, high_child);
            const int high = restrict_var(bdd_node.high, target_var, high_child);
            result = make_node(bdd_node.var, low, high);
        }
        if (edge_inv(edge)) result = edge_not(result);
        restrict_cache.emplace(key, result);
        return result;
    };

    auto ensure_var = [&](int port_id, int pin_id) {
        auto itr = port_to_var.find(port_id);
        if (itr != port_to_var.end()) return itr->second;
        const int var = static_cast<int>(var_ports.size());
        port_to_var[port_id] = var;
        var_ports.push_back(port_id);
        const bool has_pin = pin_id >= 0 && pin_id < static_cast<int>(pin_activity.size());
        var_has_pin.push_back(has_pin ? 1 : 0);
        var_duties.push_back(has_pin ? std::clamp(pin_activity[pin_id].duty, 0.0f, 1.0f) : 0.0f);
        const bool zero_density = zero_density_ports && zero_density_ports->count(port_id) != 0;
        var_densities.push_back((has_pin && !zero_density) ? pin_activity[pin_id].density : 0.0f);
        return var;
    };

    std::vector<std::pair<int, int>> expr_vars;
    for (const auto& op : expr.ops()) {
        if (op.opcode != PowerExprOpcode::port || op.port_id < 0
            || op.port_id >= static_cast<int>(cell->ports_.size()))
            continue;
        const std::string& port_name = cell->ports_[op.port_id]->name;
        auto pin_itr = node.portMap.find(port_name);
        if (pin_itr == node.portMap.end()) {
            continue;
        } else {
            const int pin_id = pin_itr->second;
            if (pin_id < 0 || pin_id >= static_cast<int>(pin_activity.size())) return false;
            expr_vars.emplace_back(op.port_id, pin_id);
        }
    }
    std::sort(expr_vars.begin(), expr_vars.end(),
              [](const auto& left, const auto& right) { return left.first < right.first; });
    expr_vars.erase(std::unique(expr_vars.begin(), expr_vars.end(),
                                [](const auto& left, const auto& right) {
                                    return left.first == right.first;
                                }),
                    expr_vars.end());
    for (const auto& [port_id, pin_id] : expr_vars)
        ensure_var(port_id, pin_id);

    std::vector<int> stack;
    for (const auto& op : expr.ops()) {
        switch (op.opcode) {
            case PowerExprOpcode::port: {
                if (!cell || op.port_id < 0 || op.port_id >= static_cast<int>(cell->ports_.size()))
                    return false;
                const std::string& port_name = cell->ports_[op.port_id]->name;
                auto pin_itr = node.portMap.find(port_name);
                if (pin_itr == node.portMap.end()) {
                    if (fixed_port_values[op.port_id] < 0) {
                        stack.push_back(zero_edge);
                        break;
                    }
                    stack.push_back(fixed_port_values[op.port_id] ? one_edge : zero_edge);
                    break;
                }
                const int pin_id = pin_itr->second;
                if (pin_id < 0 || pin_id >= static_cast<int>(pin_activity.size())) return false;
                const int var = ensure_var(op.port_id, pin_id);
                stack.push_back(make_node(var, zero_edge, one_edge));
                break;
            }
            case PowerExprOpcode::const_zero:
                stack.push_back(zero_edge);
                break;
            case PowerExprOpcode::const_one:
                stack.push_back(one_edge);
                break;
            case PowerExprOpcode::logical_not: {
                if (stack.empty()) return false;
                int a = stack.back();
                stack.back() = edge_not(a);
                break;
            }
            case PowerExprOpcode::logical_and: {
                if (stack.size() < 2) return false;
                int right = stack.back();
                stack.pop_back();
                int left = stack.back();
                stack.back() = apply_bdd(0, left, right);
                break;
            }
            case PowerExprOpcode::logical_or: {
                if (stack.size() < 2) return false;
                int right = stack.back();
                stack.pop_back();
                int left = stack.back();
                stack.back() = apply_bdd(1, left, right);
                break;
            }
            case PowerExprOpcode::logical_xor: {
                if (stack.size() < 2) return false;
                int right = stack.back();
                stack.pop_back();
                int left = stack.back();
                stack.back() = apply_bdd(2, left, right);
                break;
            }
        }
    }
    if (stack.size() != 1) return false;
    const int root = stack.back();

    std::function<float(int)> eval_bdd_duty = [&](int edge) -> float {
        const int id = edge_id(edge);
        if (id == 0) return edge_inv(edge) ? 0.0f : 1.0f;
        const auto& bdd_node = bdd_nodes[id - 1];
        if (bdd_node.var >= 0 && bdd_node.var < static_cast<int>(var_has_pin.size())
            && !var_has_pin[bdd_node.var])
            return 0.0f;
        const float duty0 = eval_bdd_duty(bdd_node.low);
        const float duty1 = eval_bdd_duty(bdd_node.high);
        const float var_duty = var_duties[bdd_node.var];
        float result = duty0 * (1.0 - var_duty) + duty1 * var_duty;
        if (edge_inv(edge)) result = 1.0 - result;
        return std::clamp(result, 0.0f, 1.0f);
    };

    duty = eval_bdd_duty(root);
    density = 0.0f;
    std::vector<int> var_order(var_ports.size());
    std::iota(var_order.begin(), var_order.end(), 0);
    std::sort(var_order.begin(), var_order.end(), [&](int left, int right) {
        return var_ports[left] < var_ports[right];
    });
    for (int var : var_order) {
        if (var < 0 || var >= static_cast<int>(var_has_pin.size()) || !var_has_pin[var])
            continue;
        restrict_cache.clear();
        const int low = restrict_var(root, var, false);
        restrict_cache.clear();
        const int high = restrict_var(root, var, true);
        const int diff = apply_bdd(2, low, high);
        const float diff_duty = eval_bdd_duty(diff);
        density += var_densities[var] * diff_duty;
    }

    return std::isfinite(density) && std::isfinite(duty);
}

static const std::string& seqClockExpr(const SequentialPower* seq) {
    static const std::string empty;
    if (!seq) return empty;
    if (seq->is_latch_ && !seq->enable_expr_.empty()) return seq->enable_expr_;
    return seq->clocked_on_expr_;
}

}  // namespace

tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> GPUTimer::report_power_liberty_inventory() {
    int64_t internal_groups = static_cast<int64_t>(gtdb.liberty_internal_powers.size());
    int64_t internal_rise_luts = 0;
    int64_t internal_fall_luts = 0;
    int64_t internal_when_exprs = 0;
    for (auto* internal_power : gtdb.liberty_internal_powers) {
        if (!internal_power) continue;
        if (internal_power->power_[RISE] && internal_power->power_[RISE]->set_) internal_rise_luts++;
        if (internal_power->power_[FALL] && internal_power->power_[FALL]->set_) internal_fall_luts++;
        if (!internal_power->when_expr_.empty()) internal_when_exprs++;
    }

    int64_t leakage_groups = static_cast<int64_t>(gtdb.liberty_leakage_powers.size());
    int64_t leakage_when_exprs = 0;
    for (auto* leakage_power : gtdb.liberty_leakage_powers) {
        if (leakage_power && !leakage_power->when_expr_.empty()) leakage_when_exprs++;
    }

    int64_t output_functions = 0;
    for (uint8_t has_function : gtdb.liberty_port_has_function) {
        if (has_function) output_functions++;
    }

    return {internal_groups, internal_rise_luts, internal_fall_luts, internal_when_exprs, leakage_groups, leakage_when_exprs, output_functions};
}

int64_t GPUTimer::report_power_seq_inventory() {
    int64_t seqs = 0;
    for (const auto* cell_type : gtdb.rawdb.celltypes) {
        if (cell_type && cell_type->liberty_cell)
            seqs += static_cast<int64_t>(cell_type->liberty_cell->sequentials_.size());
    }
    return seqs;
}

torch::Tensor GPUTimer::report_power_group_codes() {
    constexpr int sequential_code = 0;
    constexpr int combinational_code = 1;
    constexpr int clock_code = 2;
    constexpr int macro_code = 3;
    constexpr int pad_code = 4;

    auto out = torch::empty({num_nodes}, torch::dtype(torch::kInt64).device(torch::kCPU));
    auto acc = out.accessor<int64_t, 1>();

    const auto& nodes = gtdb.gpdb.getNodes();
    const int n = static_cast<int>(gtdb.pin_names.size());

    auto get_cell_type = [&](int node_id) -> db::CellType* {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) return nullptr;
        const int cell_type_id = gtdb.cell_node_type_map[node_id];
        if (cell_type_id < 0 || cell_type_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return nullptr;
        return gtdb.rawdb.celltypes[cell_type_id];
    };
    auto get_cell = [&](int node_id) -> LibertyCell* {
        db::CellType* cell_type = get_cell_type(node_id);
        return cell_type ? cell_type->liberty_cell : nullptr;
    };
    auto is_io_node = [](const gp::GPNode& node) {
        const std::string& node_type = node.getNodeType();
        return node_type == "IOPin" || node_type == "FloatIOPin";
    };
    auto is_pad_cell_type = [](const db::CellType* cell_type) {
        if (!cell_type) return false;
        return cell_type->cls.find("PAD") == 0;
    };
    auto is_macro_cell_type = [](const db::CellType* cell_type) {
        if (!cell_type) return false;
        return cell_type->cls != "CORE";
    };

    std::vector<int> pin_to_net(n, -1);
    std::vector<int> pin_to_node(n, -1);
    std::vector<uint8_t> is_driver_pin(n, 0);
    std::vector<uint8_t> is_load_pin(n, 0);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        const int pin_id = static_cast<int>(pin.getId());
        if (pin_id >= 0 && pin_id < n) {
            pin_to_net[pin_id] = static_cast<int>(pin.getParNetId());
            pin_to_node[pin_id] = static_cast<int>(pin.getParNodeId());
            if (pin_id < static_cast<int>(gtdb.pin_id2port_offset_id.size())) {
                LibertyCell* cell = get_cell(pin_to_node[pin_id]);
                const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (cell && port_offset >= 0 && port_offset < static_cast<int>(cell->ports_.size())) {
                    const LibertyPort* port = cell->ports_[port_offset];
                    if (port) {
                        if (port->direction_ == CellPortDirection::input
                            || port->direction_ == CellPortDirection::inout)
                            is_load_pin[pin_id] = 1;
                        if (port->direction_ == CellPortDirection::output
                            || port->direction_ == CellPortDirection::inout)
                            is_driver_pin[pin_id] = 1;
                    }
                }
            }
        }
    }

    const int num_nets = static_cast<int>(gtdb.gpdb.getNets().size());
    std::vector<uint8_t> is_clock_net(num_nets, 0);
    std::vector<uint8_t> forward_clock_net(num_nets, 0);
    std::deque<int> forward_queue;
    auto mark_clock_net = [&](int net_id) -> bool {
        if (net_id < 0 || net_id >= num_nets || is_clock_net[net_id]) return false;
        is_clock_net[net_id] = 1;
        return true;
    };
    auto mark_forward_net = [&](int net_id) {
        if (net_id < 0 || net_id >= num_nets || forward_clock_net[net_id]) return;
        mark_clock_net(net_id);
        forward_clock_net[net_id] = 1;
        forward_queue.push_back(net_id);
    };
    if (gtdb.net_is_clock.size() == static_cast<size_t>(num_nets)) {
        for (int net_id = 0; net_id < num_nets; ++net_id) {
            if (gtdb.net_is_clock[net_id]) mark_forward_net(net_id);
        }
    }
    if (gtdb.pin_is_clk.size() == static_cast<size_t>(n)) {
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (gtdb.pin_is_clk[pin_id]) mark_forward_net(pin_to_net[pin_id]);
        }
    }

    auto is_core_comb_node = [&](int node_id, LibertyCell* cell) {
        if (!cell || !cell->sequentials_.empty()) return false;
        db::CellType* cell_type = get_cell_type(node_id);
        return cell_type && cell_type->cls == "CORE";
    };
    auto is_clock_transparent_from_pin = [&](int node_id, LibertyCell* cell, int in_pin_id) {
        if (!is_core_comb_node(node_id, cell)) return false;
        if (in_pin_id < 0 || in_pin_id >= n ||
            in_pin_id >= static_cast<int>(gtdb.pin_id2port_offset_id.size()))
            return false;
        const int in_port = gtdb.pin_id2port_offset_id[in_pin_id];
        if (in_port < 0 || in_port >= static_cast<int>(cell->ports_.size())) return false;

        bool output_seen = false;
        if (node_id < 0 || node_id >= static_cast<int>(nodes.size())) return false;
        for (int out_pin : nodes[node_id].pins()) {
            if (out_pin < 0 || out_pin >= n || !is_driver_pin[out_pin]) continue;
            if (out_pin >= static_cast<int>(gtdb.pin_id2port_offset_id.size())) return false;
            const int out_port = gtdb.pin_id2port_offset_id[out_pin];
            if (out_port < 0 || out_port >= static_cast<int>(cell->ports_.size())) return false;
            LibertyPort* port = cell->ports_[out_port];
            if (!port || !port->has_function_) return false;
            PowerExpr expr;
            if (!expr.compile(port->function_expr_, cell)) return false;
            const auto& ops = expr.ops();
            const bool direct =
                ops.size() == 1 && ops[0].opcode == PowerExprOpcode::port && ops[0].port_id == in_port;
            const bool inverted =
                ops.size() == 2 && ops[0].opcode == PowerExprOpcode::port && ops[0].port_id == in_port &&
                ops[1].opcode == PowerExprOpcode::logical_not;
            if (!direct && !inverted) return false;
            output_seen = true;
        }
        return output_seen;
    };
    for (size_t queue_pos = 0; queue_pos < forward_queue.size(); ++queue_pos) {
        const int net_id = forward_queue[queue_pos];
        if (net_id < 0 || net_id >= num_nets) continue;
        for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
            if (pin_id < 0 || pin_id >= n || !is_load_pin[pin_id]) continue;
            const int node_id = pin_to_node[pin_id];
            LibertyCell* cell = get_cell(node_id);
            if (!is_clock_transparent_from_pin(node_id, cell, pin_id)) continue;
            for (int out_pin : nodes[node_id].pins()) {
                if (out_pin >= 0 && out_pin < n && is_driver_pin[out_pin])
                    mark_forward_net(pin_to_net[out_pin]);
            }
        }
    }

    std::vector<uint8_t> is_clock_pin(n, 0);
    for (int net_id = 0; net_id < num_nets; ++net_id) {
        if (!is_clock_net[net_id]) continue;
        for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
            if (pin_id >= 0 && pin_id < n) is_clock_pin[pin_id] = 1;
        }
    }
    auto in_clock_network = [&](const gp::GPNode& node, LibertyCell* cell) {
        if (!cell) return false;
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            if (pin_id >= static_cast<int>(gtdb.pin_id2port_offset_id.size())) continue;
            const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            const LibertyPort* port = cell->ports_[port_offset];
            if (!port) continue;
            const bool is_output = port->direction_ == CellPortDirection::output
                || port->direction_ == CellPortDirection::inout;
            if (is_output && !is_clock_pin[pin_id]) return false;
        }
        return true;
    };

    for (int node_id = 0; node_id < num_nodes; ++node_id) {
        int group_code = combinational_code;
        if (node_id >= 0 && node_id < static_cast<int>(nodes.size())) {
            const gp::GPNode& node = nodes[node_id];
            db::CellType* cell_type = get_cell_type(node_id);
            LibertyCell* cell = get_cell(node_id);
            if (is_io_node(node) || is_pad_cell_type(cell_type)) {
                group_code = pad_code;
            } else if (is_macro_cell_type(cell_type)) {
                group_code = macro_code;
            } else if (cell && in_clock_network(node, cell)) {
                group_code = clock_code;
            } else if (cell && !cell->sequentials_.empty()) {
                group_code = sequential_code;
            }
        }
        acc[node_id] = group_code;
    }
    return out;
}

torch::Tensor GPUTimer::report_power_activity_cpu() {
    const int n = static_cast<int>(gtdb.pin_names.size());
    std::vector<CpuActivity> act(n);
    std::vector<CpuActivity> seq_pin_activity(n);
    std::vector<uint8_t> seq_pin_activity_valid(n, 0);
    std::vector<uint8_t> clock_activity_protected(n, 0);

    const double sdc_time_scale =
        canonicalPowerTimeScale(gtdb.sdc_time_unit.has_value() ? *gtdb.sdc_time_unit : gtdb.time_unit);
    double min_period_sec = std::numeric_limits<double>::infinity();
    for (const auto& kv : gtdb.clocks) {
        const float period = kv.second.period();
        const double period_sec = static_cast<double>(period) * sdc_time_scale;
        if (period_sec > 0.0) min_period_sec = std::min(min_period_sec, period_sec);
    }
    if (!std::isfinite(min_period_sec) || min_period_sec <= 0.0) {
        const double fallback_scale = canonicalPowerTimeScale(gtdb.time_unit);
        min_period_sec = fallback_scale > 0.0 ? fallback_scale : 1.0e-9;
    }
    const float default_density = static_cast<float>(0.1 / min_period_sec);
    const float clock_density = static_cast<float>(2.0 / min_period_sec);
    auto clock_activity_for_pin = [&](int pin_id) -> std::pair<float, float> {
        float density = clock_density;
        float duty = 0.5f;
        if (pin_id >= 0 && pin_id < static_cast<int>(gtdb.pin_clock_periods.size())) {
            const float period = gtdb.pin_clock_periods[pin_id];
            if (std::isfinite(period) && period > 0.0f && sdc_time_scale > 0.0) {
                density = powerDensityForPeriod(2.0, period, sdc_time_scale);
                if (pin_id < static_cast<int>(gtdb.pin_clock_rise_edges.size())
                    && pin_id < static_cast<int>(gtdb.pin_clock_fall_edges.size())) {
                    const float rise = gtdb.pin_clock_rise_edges[pin_id];
                    const float fall = gtdb.pin_clock_fall_edges[pin_id];
                    if (std::isfinite(rise) && std::isfinite(fall)) {
                        const float candidate_duty = (fall - rise) / period;
                        if (std::isfinite(candidate_duty) && candidate_duty >= 0.0f && candidate_duty <= 1.0f)
                            duty = candidate_duty;
                    }
                }
            }
        }
        return {density, duty};
    };

    std::vector<int> pin_level(n, 0);
    int max_pin_level = 0;
    if (gtdb.pin_num_fanin.size() == static_cast<size_t>(n)
        && gtdb.pin_fanout_list_end.size() == static_cast<size_t>(n + 1)) {
        std::vector<int> indeg = gtdb.pin_num_fanin;
        std::deque<int> frontier;
        for (int i = 0; i < n; i++) {
            if (indeg[i] == 0) frontier.push_back(i);
        }
        std::vector<uint8_t> seen(n, 0);
        while (!frontier.empty()) {
            int pin_id = frontier.front();
            frontier.pop_front();
            if (pin_id < 0 || pin_id >= n || seen[pin_id]) continue;
            seen[pin_id] = 1;
            max_pin_level = std::max(max_pin_level, pin_level[pin_id]);
            int start = gtdb.pin_fanout_list_end[pin_id];
            int end = gtdb.pin_fanout_list_end[pin_id + 1];
            for (int idx = start; idx < end; idx++) {
                int fanout = gtdb.pin_fanout_list[idx];
                if (fanout < 0 || fanout >= n) continue;
                pin_level[fanout] = std::max(pin_level[fanout], pin_level[pin_id] + 1);
                if (--indeg[fanout] == 0) frontier.push_back(fanout);
            }
        }
    }
    std::vector<std::deque<int>> level_queues(max_pin_level + 2);
    std::set<int> nonempty_queue_levels;
    std::vector<uint8_t> in_queue(n, 0);
    std::vector<uint8_t> force_propagate_on_visit(n, 0);
    std::vector<int> pin_to_node(n, -1);
    std::vector<int> pin_to_net(n, -1);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        int pin_id = static_cast<int>(pin.getId());
        if (pin_id >= 0 && pin_id < n) {
            pin_to_node[pin_id] = static_cast<int>(pin.getParNodeId());
            pin_to_net[pin_id] = static_cast<int>(pin.getParNetId());
        }
    }

    const char* trace_path_file_env = std::getenv("XPLACE_POWER_TRACE_PATH_FILE");
    const char* trace_path_out_env = std::getenv("XPLACE_POWER_TRACE_PATH_OUT");
    const char* activity_path_trace_env = std::getenv("XPLACE_POWER_ACTIVITY_PATH_TRACE_FILE");
    PowerTracePathState path_trace =
        loadPowerTracePathState(trace_path_file_env, activity_path_trace_env, pin_to_node);
    int path_trace_pass = 0;
    std::string path_trace_level_tag = "seed";
    auto path_trace_hit = [&](int pin_id, int arc_id) -> bool {
        return path_trace.enabled()
            && ((pin_id >= 0 && path_trace.pins.count(pin_id))
                || (arc_id != -1 && path_trace.arcs.count(arc_id)));
    };
    auto emit_path_trace = [&](const char* event,
                               int arc_id,
                               int from_pin,
                               int to_pin,
                               float old_density,
                               float old_duty,
                               float new_density,
                               float new_duty,
                               bool changed,
                               bool enqueued,
                               int pending_seq,
                               const char* reason) {
        if (!path_trace.enabled()) return;
        if (!path_trace_hit(from_pin, arc_id) && !path_trace_hit(to_pin, arc_id)) return;
        path_trace.out << "xplace_cpu\t" << path_trace_pass << '\t'
                       << path_trace_level_tag << '\t' << (event ? event : "") << '\t'
                       << arc_id << '\t'
                       << from_pin << '\t'
                       << ((from_pin >= 0 && from_pin < n) ? gtdb.pin_names[from_pin] : "") << '\t'
                       << to_pin << '\t'
                       << ((to_pin >= 0 && to_pin < n) ? gtdb.pin_names[to_pin] : "") << '\t'
                       << old_density << '\t' << old_duty << '\t'
                       << new_density << '\t' << new_duty << '\t'
                       << (changed ? 1 : 0) << '\t'
                       << (enqueued ? 1 : 0) << '\t'
                       << pending_seq << '\t'
                       << (reason ? reason : "") << '\n';
    };

    auto enqueue = [&](int pin_id,
                       bool force_propagate = false,
                       int from_pin = -1,
                       int arc_id = -1,
                       const char* reason = "enqueue") {
        if (pin_id < 0 || pin_id >= n) return;
        if (force_propagate) force_propagate_on_visit[pin_id] = 1;
        if (in_queue[pin_id]) {
            emit_path_trace("enqueue_skip_queued", arc_id, from_pin, pin_id,
                            act[pin_id].density, act[pin_id].duty,
                            act[pin_id].density, act[pin_id].duty,
                            false, false, 0, reason);
            return;
        }
        int level = std::clamp(pin_level[pin_id], 0, max_pin_level + 1);
        if (level_queues[level].empty()) nonempty_queue_levels.insert(level);
        level_queues[level].push_back(pin_id);
        in_queue[pin_id] = 1;
        emit_path_trace("enqueue", arc_id, from_pin, pin_id,
                        act[pin_id].density, act[pin_id].duty,
                        act[pin_id].density, act[pin_id].duty,
                        false, true, 0, reason);
    };

    auto percent_change = [](float value, float prev) -> float {
        if (prev == 0.0f) return value == 0.0f ? 0.0f : 1.0f;
        return std::abs(value - prev) / std::abs(prev);
    };
    const float min_activity_density =
        std::max(0.0f, readPowerFloatEnv("XPLACE_POWER_MIN_ACTIVITY_DENSITY", 1.0e-10f));
    const float min_activity_duty =
        std::max(0.0f, readPowerFloatEnv("XPLACE_POWER_MIN_ACTIVITY_DUTY", 0.0f));
    const bool ignore_scan_enable_density =
        readPowerBoolEnv("XPLACE_POWER_IGNORE_SCAN_ENABLE_DENSITY", false);
    const bool require_known_seq_data =
        readPowerBoolEnv("XPLACE_POWER_REQUIRE_KNOWN_SEQ_DATA", false);
    const bool allow_clock_activity_override =
        readPowerBoolEnv("XPLACE_POWER_ALLOW_CLOCK_ACTIVITY_OVERRIDE", false);
    const bool disable_activity_slew_cap =
        readPowerBoolEnv("XPLACE_POWER_DISABLE_ACTIVITY_SLEW_CAP", false);
    const bool clamp_activity_to_clock_density =
        readPowerBoolEnv("XPLACE_POWER_CLAMP_ACTIVITY_TO_CLOCK_DENSITY", false);
    torch::Tensor pin_slew_cpu;
    const float* pin_slew_host = nullptr;
    if (!disable_activity_slew_cap && timing_raw_db.pinSlew.defined()
        && timing_raw_db.pinSlew.numel() >= static_cast<int64_t>(n) * NUM_ATTR) {
        pin_slew_cpu = timing_raw_db.pinSlew.to(torch::kCPU).contiguous();
        pin_slew_host = pin_slew_cpu.data_ptr<float>();
    }
    auto max_activity_density_for_pin = [&](int pin_id) -> float {
        float max_density = std::numeric_limits<float>::infinity();
        if (pin_slew_host && pin_id >= 0 && pin_id < n
            && gtdb.time_unit > 0.0f) {
            float min_rf_slew = std::numeric_limits<float>::infinity();
            for (int attr = 0; attr + 1 < NUM_ATTR; attr += 2) {
                const float rise = pin_slew_host[pin_id * NUM_ATTR + attr];
                const float fall = pin_slew_host[pin_id * NUM_ATTR + attr + 1];
                if (!std::isfinite(rise) || !std::isfinite(fall)) continue;
                const float avg_slew = 0.5f * (rise + fall) * gtdb.time_unit;
                if (avg_slew > 0.0f && avg_slew < min_rf_slew)
                    min_rf_slew = avg_slew;
            }
            if (std::isfinite(min_rf_slew) && min_rf_slew > 0.0f)
                max_density = 1.0f / min_rf_slew;
        }
        if (clamp_activity_to_clock_density)
            max_density = std::min(max_density, clock_density);
        return max_density;
    };

    std::vector<int> trace_pin_ids = resolvePowerTracePins(readPowerTracePinQueries(), gtdb.pin_names);
    std::vector<uint8_t> trace_first_seen(n, 0);
    auto trace_matches = [&](int pin_id) -> bool {
        return pin_id >= 0 && pin_id < n &&
            std::find(trace_pin_ids.begin(), trace_pin_ids.end(), pin_id) != trace_pin_ids.end();
    };

    auto set_activity = [&](int pin_id, float density, float duty, int origin, bool force, bool enqueue_on_change = true) -> bool {
        if (pin_id < 0 || pin_id >= n) return false;
        if (!force && !allow_clock_activity_override && clock_activity_protected[pin_id]
            && act[pin_id].origin == 2 && origin != 2)
            return false;
        const float prev_density = act[pin_id].density;
        const float prev_duty = act[pin_id].duty;
        const int prev_origin = act[pin_id].origin;
        float duty_clamped = std::clamp(duty, 0.0f, 1.0f);
        if (min_activity_duty > 0.0f) {
            if (duty_clamped < min_activity_duty)
                duty_clamped = 0.0f;
            else if ((1.0f - duty_clamped) < min_activity_duty)
                duty_clamped = 1.0f;
        }
        float density_clamped = std::max(density, 0.0f);
        const float max_density = force ? std::numeric_limits<float>::infinity()
                                        : max_activity_density_for_pin(pin_id);
        if (std::isfinite(max_density))
            density_clamped = std::min(density_clamped, max_density);
        // Match OpenSTA PwrActivity::check() by default; the env override is
        // for diagnosing tiny feedback noise that can be amplified by loops.
        if (std::abs(density_clamped) < min_activity_density) density_clamped = 0.0f;
        const bool value_changed = percent_change(density_clamped, prev_density) > 0.01f
            || percent_change(duty_clamped, prev_duty) > 0.01f;
        const bool changed = value_changed || prev_origin != origin;
        act[pin_id].density = density_clamped;
        act[pin_id].duty = duty_clamped;
        act[pin_id].origin = origin;
        if (trace_matches(pin_id)) {
            std::cerr << "[power_activity_trace_set] pin=" << gtdb.pin_names[pin_id]
                      << " level=" << pin_level[pin_id]
                      << " prev_density=" << prev_density
                      << " prev_duty=" << prev_duty
                      << " density=" << density_clamped
                      << " duty=" << duty_clamped
                      << " origin=" << origin
                      << " changed=" << changed
                      << " enqueue=" << (changed && enqueue_on_change)
                      << std::endl;
        }
        emit_path_trace("set_activity", -1, -1, pin_id,
                        prev_density, prev_duty,
                        density_clamped, duty_clamped,
                        changed, changed && enqueue_on_change, 0,
                        force ? "force" : "activity");
        if (changed && enqueue_on_change) enqueue(pin_id, true, pin_id, -1, "set_activity_revisit");
        return changed;
    };

    auto normalize_expr = [](std::string expr) {
        expr.erase(std::remove_if(expr.begin(), expr.end(), [](unsigned char c) { return std::isspace(c); }), expr.end());
        if (expr.size() >= 2 && expr.front() == '"' && expr.back() == '"')
            expr = expr.substr(1, expr.size() - 2);
        return expr;
    };

    std::vector<uint8_t> pending_reg_flag(gtdb.gpdb.getNodes().size(), 0);
    std::vector<int> pending_regs;
    std::vector<int> pending_reg_trigger_pin(gtdb.gpdb.getNodes().size(), -1);
    auto mark_pending_reg = [&](int node_id, int trigger_pin) {
        if (node_id < 0 || node_id >= static_cast<int>(pending_reg_flag.size())) return;
        if (pending_reg_flag[node_id]) return;
        pending_reg_flag[node_id] = 1;
        pending_reg_trigger_pin[node_id] = trigger_pin;
        pending_regs.push_back(node_id);
        if (path_trace.enabled() && path_trace.nodes.count(node_id)) {
            int node_trace_pin = -1;
            for (int pin_id : path_trace.pins) {
                if (pin_id >= 0 && pin_id < static_cast<int>(pin_to_node.size())
                    && pin_to_node[pin_id] == node_id) {
                    node_trace_pin = pin_id;
                    break;
                }
            }
            emit_path_trace("seq_pending", -1, node_trace_pin, node_trace_pin,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            true, false, static_cast<int>(pending_regs.size()),
                            "load_pin_changed");
        }
    };

    auto get_cell = [&](int node_id) -> LibertyCell* {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) return nullptr;
        int libcell_id = gtdb.cell_node_type_map[node_id];
        if (libcell_id < 0 || libcell_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return nullptr;
        auto* cell_type = gtdb.rawdb.celltypes[libcell_id];
        return cell_type ? cell_type->liberty_cell : nullptr;
    };
    const bool mark_seq_clock_loads =
        readPowerBoolEnv("XPLACE_POWER_MARK_SEQ_CLOCK_LOADS", false);
    auto is_seq_clock_input_pin = [&](int pin_id) {
        if (pin_id < 0 || pin_id >= n) return false;
        const int node_id = pin_to_node[pin_id];
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) return false;
        int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) return false;
        LibertyPort* port = cell->ports_[port_offset];
        return port && port->is_clock_;
    };
    auto parse_const_net_value = [](std::string name) -> int {
        name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char c) { return std::isspace(c); }),
                   name.end());
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (name == "0" || name == "1'b0" || name == "1'd0" || name == "1'h0") return 0;
        if (name == "1" || name == "1'b1" || name == "1'd1" || name == "1'h1") return 1;
        const size_t quote = name.find('\'');
        if (quote != std::string::npos && quote + 2 < name.size()) {
            const std::string digits = name.substr(quote + 2);
            if (!digits.empty() && digits.find_first_not_of("0") == std::string::npos) return 0;
            if (!digits.empty() && digits.find_first_not_of("1") == std::string::npos) return 1;
        }
        return -1;
    };
    std::vector<std::unordered_map<int, int>> node_const_port_values(gtdb.gpdb.getNodes().size());
    std::vector<uint8_t> node_const_port_values_ready(gtdb.gpdb.getNodes().size(), 0);
    std::unordered_map<std::string, int> const_port_file_values;
    bool const_port_file_loaded = false;
    auto load_const_port_file = [&]() {
        if (const_port_file_loaded) return;
        const_port_file_loaded = true;
        const char* file_name = std::getenv("XPLACE_POWER_CONST_PORT_FILE");
        if (!file_name || file_name[0] == '\0') return;
        std::ifstream stream(file_name);
        if (!stream) return;
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string inst;
            std::string port;
            std::string value;
            if (!std::getline(ss, inst, ',')) continue;
            if (!std::getline(ss, port, ',')) continue;
            if (!std::getline(ss, value, ',')) continue;
            if (inst == "inst_name" || inst == "inst") continue;
            const int const_value = parse_const_net_value(value);
            if (const_value < 0) continue;
            const std::string key = normalizePowerActivitySnapshotName(inst) + "/" +
                                    normalizePowerActivitySnapshotName(port);
            const_port_file_values[key] = const_value;
        }
    };
    auto const_port_values_for_node = [&](int node_id, LibertyCell* cell) -> const std::unordered_map<int, int>& {
        static const std::unordered_map<int, int> empty;
        if (!cell || node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return empty;
        if (node_const_port_values_ready[node_id]) return node_const_port_values[node_id];
        node_const_port_values_ready[node_id] = 1;
        const auto& node = gtdb.gpdb.getNodes()[node_id];
        load_const_port_file();
        for (int port_id = 0; port_id < static_cast<int>(cell->ports_.size()); port_id++) {
            LibertyPort* port = cell->ports_[port_id];
            if (!port || port->direction_ == CellPortDirection::output) continue;
            if (node.portMap.find(port->name) != node.portMap.end()) continue;
            const std::string key = normalizePowerActivitySnapshotName(node.getName()) + "/" +
                                    normalizePowerActivitySnapshotName(port->name);
            auto const_itr = const_port_file_values.find(key);
            if (const_itr != const_port_file_values.end())
                node_const_port_values[node_id][port_id] = const_itr->second;
        }
        const int raw_cell_id = static_cast<int>(node.getOriDBId());
        if (raw_cell_id < 0 || raw_cell_id >= static_cast<int>(gtdb.rawdb.cells.size()))
            return node_const_port_values[node_id];
        db::Cell* dbcell = gtdb.rawdb.cells[raw_cell_id];
        if (!dbcell) return node_const_port_values[node_id];
        for (int port_id = 0; port_id < static_cast<int>(cell->ports_.size()); port_id++) {
            LibertyPort* port = cell->ports_[port_id];
            if (!port || port->direction_ == CellPortDirection::output) continue;
            if (node.portMap.find(port->name) != node.portMap.end()) continue;
            db::Pin* dbpin = dbcell->pin(port->name);
            if (!dbpin || !dbpin->net) continue;
            const int value = parse_const_net_value(dbpin->net->name);
            if (value >= 0) node_const_port_values[node_id][port_id] = value;
        }
        return node_const_port_values[node_id];
    };
    auto is_io_node = [&](int node_id) -> bool {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return false;
        const std::string& node_type = gtdb.gpdb.getNodes()[node_id].getNodeType();
        return node_type == "IOPin" || node_type == "FloatIOPin";
    };

    std::vector<uint8_t> is_load_pin(n, 0);
    std::vector<uint8_t> is_driver_pin(n, 0);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        int pin_id = static_cast<int>(pin.getId());
        if (pin_id < 0 || pin_id >= n) continue;
        int node_id = pin_to_node[pin_id];
        if (is_io_node(node_id)
            && std::find(gtdb.primary_inputs.begin(), gtdb.primary_inputs.end(), pin_id) != gtdb.primary_inputs.end()) {
            is_driver_pin[pin_id] = 1;
            continue;
        }
        if (is_io_node(node_id)
            && std::find(gtdb.primary_outputs.begin(), gtdb.primary_outputs.end(), pin_id) != gtdb.primary_outputs.end()) {
            is_load_pin[pin_id] = 1;
            continue;
        }
        LibertyCell* cell = get_cell(node_id);
        if (!cell) continue;
        int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
        LibertyPort* port = cell->ports_[port_offset];
        if (!port) continue;
        if (port->direction_ == CellPortDirection::input) is_load_pin[pin_id] = 1;
        else if (port->direction_ == CellPortDirection::output) is_driver_pin[pin_id] = 1;
    }
    std::vector<uint8_t> is_seq_output_pin(n, 0);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) continue;
        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            const std::string seq_out = normalize_expr(seq->output_name_);
            const std::string seq_out_inv = normalize_expr(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n || !is_driver_pin[pin_id]) continue;
                int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || !port->has_function_) continue;
                const std::string func = normalize_expr(port->function_expr_);
                if ((!seq_out.empty() && func == seq_out) ||
                    (!seq_out_inv.empty() && func == seq_out_inv)) {
                    is_seq_output_pin[pin_id] = 1;
                }
            }
        }
    }

    std::vector<int> net_driver_pin(gtdb.gpdb.getNets().size(), -1);
    for (const auto& net : gtdb.gpdb.getNets()) {
        const int net_id = static_cast<int>(net.getId());
        if (net_id < 0 || net_id >= static_cast<int>(net_driver_pin.size())) continue;
        for (int pin_id : net.pins()) {
            if (pin_id >= 0 && pin_id < n && is_driver_pin[pin_id]) {
                net_driver_pin[net_id] = pin_id;
                break;
            }
        }
    }
    std::vector<int> clock_gate_out_for_input(n, -1);
    std::vector<int> clock_gate_clock_for_out(n, -1);
    std::vector<int> clock_gate_enable_for_out(n, -1);
    std::vector<uint8_t> is_clock_gate_clock_pin(n, 0);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell) continue;
        int clk_pin = -1;
        int enable_pin = -1;
        int out_pin = -1;
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port) continue;
            if (port->is_clock_gate_clock_) clk_pin = pin_id;
            if (port->is_clock_gate_enable_) enable_pin = pin_id;
            if (port->is_clock_gate_out_) out_pin = pin_id;
        }
        if (out_pin >= 0 && clk_pin >= 0 && enable_pin >= 0) {
            clock_gate_clock_for_out[out_pin] = clk_pin;
            clock_gate_enable_for_out[out_pin] = enable_pin;
            clock_gate_out_for_input[clk_pin] = out_pin;
            clock_gate_out_for_input[enable_pin] = out_pin;
            is_clock_gate_clock_pin[clk_pin] = 1;
        }
    }

    auto build_clock_pins = [&]() {
        const int num_nets = static_cast<int>(gtdb.gpdb.getNets().size());
        std::vector<uint8_t> is_clock_net(num_nets, 0);
        auto mark_net = [&](int net_id) -> bool {
            if (net_id < 0 || net_id >= num_nets || is_clock_net[net_id]) return false;
            is_clock_net[net_id] = 1;
            return true;
        };

        if (gtdb.net_is_clock.size() == static_cast<size_t>(num_nets)) {
            for (int net_id = 0; net_id < num_nets; net_id++) {
                if (gtdb.net_is_clock[net_id]) mark_net(net_id);
            }
        }
        for (int pin_id = 0; pin_id < n; pin_id++) {
            if (is_clock_gate_clock_pin[pin_id]) mark_net(pin_to_net[pin_id]);
        }

        std::vector<uint8_t> forward_clock_net(num_nets, 0);
        std::deque<int> forward_queue;
        auto mark_forward_net = [&](int net_id) {
            if (net_id < 0 || net_id >= num_nets || forward_clock_net[net_id]) return;
            forward_clock_net[net_id] = 1;
            forward_queue.push_back(net_id);
        };
        if (gtdb.net_is_clock.size() == static_cast<size_t>(num_nets)) {
            for (int net_id = 0; net_id < num_nets; net_id++) {
                if (gtdb.net_is_clock[net_id]) mark_forward_net(net_id);
            }
        }
        std::vector<int> extra_clock_pins;
        std::vector<uint8_t> extra_clock_pin_seen(n, 0);
        auto add_extra_clock_pin = [&](int pin_id) {
            if (pin_id < 0 || pin_id >= n || extra_clock_pin_seen[pin_id]) return;
            extra_clock_pin_seen[pin_id] = 1;
            extra_clock_pins.push_back(pin_id);
        };
        auto is_core_comb_node = [&](int node_id, LibertyCell* cell) {
            if (!cell || !cell->sequentials_.empty()) return false;
            if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) return false;
            const int cell_type_id = gtdb.cell_node_type_map[node_id];
            if (cell_type_id < 0 || cell_type_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return false;
            db::CellType* cell_type = gtdb.rawdb.celltypes[cell_type_id];
            return cell_type && cell_type->cls == "CORE";
        };
        auto is_clock_transparent_from_pin = [&](int node_id, LibertyCell* cell, int in_pin_id) {
            if (!is_core_comb_node(node_id, cell)) return false;
            if (in_pin_id < 0 || in_pin_id >= n ||
                in_pin_id >= static_cast<int>(gtdb.pin_id2port_offset_id.size()))
                return false;
            const int in_port = gtdb.pin_id2port_offset_id[in_pin_id];
            if (in_port < 0 || in_port >= static_cast<int>(cell->ports_.size())) return false;

            bool output_seen = false;
            for (int out_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
                if (out_pin < 0 || out_pin >= n || !is_driver_pin[out_pin]) continue;
                if (out_pin >= static_cast<int>(gtdb.pin_id2port_offset_id.size())) return false;
                const int out_port = gtdb.pin_id2port_offset_id[out_pin];
                if (out_port < 0 || out_port >= static_cast<int>(cell->ports_.size())) return false;
                LibertyPort* port = cell->ports_[out_port];
                if (!port || !port->has_function_) return false;
                PowerExpr expr;
                if (!expr.compile(port->function_expr_, cell)) return false;
                const auto& ops = expr.ops();
                const bool direct =
                    ops.size() == 1 && ops[0].opcode == PowerExprOpcode::port && ops[0].port_id == in_port;
                const bool inverted =
                    ops.size() == 2 && ops[0].opcode == PowerExprOpcode::port && ops[0].port_id == in_port &&
                    ops[1].opcode == PowerExprOpcode::logical_not;
                if (!direct && !inverted) return false;
                output_seen = true;
            }
            return output_seen;
        };
        for (size_t queue_pos = 0; queue_pos < forward_queue.size(); ++queue_pos) {
            const int net_id = forward_queue[queue_pos];
            if (net_id < 0 || net_id >= num_nets) continue;
            for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
                if (pin_id < 0 || pin_id >= n || !is_load_pin[pin_id]) continue;
                const int node_id = pin_to_node[pin_id];
                LibertyCell* cell = get_cell(node_id);
                if (!is_core_comb_node(node_id, cell)) continue;
                add_extra_clock_pin(pin_id);
                if (!is_clock_transparent_from_pin(node_id, cell, pin_id)) continue;
                for (int out_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
                    if (out_pin >= 0 && out_pin < n && is_driver_pin[out_pin])
                        mark_forward_net(pin_to_net[out_pin]);
                }
            }
        }
        std::vector<int> clock_pins;
        for (int net_id = 0; net_id < num_nets; net_id++) {
            if (!is_clock_net[net_id]) continue;
            for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
                if (pin_id >= 0 && pin_id < n
                    && (is_load_pin[pin_id] || is_io_node(pin_to_node[pin_id])))
                    clock_pins.push_back(pin_id);
            }
        }
        clock_pins.insert(clock_pins.end(), extra_clock_pins.begin(), extra_clock_pins.end());
        std::sort(clock_pins.begin(), clock_pins.end());
        clock_pins.erase(std::unique(clock_pins.begin(), clock_pins.end()), clock_pins.end());
        return clock_pins;
    };

    auto enqueue_adjacent_vertices = [&](int pin_id) {
        if (pin_id < 0 || pin_id >= n) return;
        if (is_driver_pin[pin_id]) {
            const int net_id = pin_to_net[pin_id];
            if (net_id >= 0 && net_id < static_cast<int>(gtdb.gpdb.getNets().size())) {
                for (int sink_pin : gtdb.gpdb.getNets()[net_id].pins()) {
                    if (sink_pin < 0 || sink_pin >= n || sink_pin == pin_id || !is_load_pin[sink_pin])
                        continue;
                    enqueue(sink_pin, false, pin_id, -1, "net_fanout");
                }
            }
        }
        if (gtdb.pin_forward_arc_list_end.size() != static_cast<size_t>(n + 1)) return;
        int start = gtdb.pin_forward_arc_list_end[pin_id];
        int end = gtdb.pin_forward_arc_list_end[pin_id + 1];
        for (int idx = start; idx < end; idx++) {
            int arc_id = gtdb.pin_forward_arc_list[idx];
            if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) continue;
            if (arc_id < static_cast<int>(gtdb.arc_id2test_id.size()) && gtdb.arc_id2test_id[arc_id] != -1) continue;  // timing checks.
            int to_pin = gtdb.timing_arc_to_pin_id[arc_id];
            if (std::getenv("XPLACE_POWER_ACTIVITY_SKIP_BACK_LEVEL_ARCS")
                && arc_id < static_cast<int>(gtdb.arc_types.size()) && gtdb.arc_types[arc_id] == 1
                && to_pin >= 0 && to_pin < n && pin_level[to_pin] <= pin_level[pin_id])
                continue;
            // OpenSTA ActivitySrchPred excludes reg clk->Q/latch D->Q. Xplace does not expose
            // TimingRole here, so conservatively skip cell arcs into sequential outputs; they
            // are propagated only by seed_reg_outputs().
            if (arc_id < static_cast<int>(gtdb.arc_types.size()) && gtdb.arc_types[arc_id] == 1) {
                int to_node = to_pin >= 0 && to_pin < n ? pin_to_node[to_pin] : -1;
                LibertyCell* to_cell = get_cell(to_node);
                if (to_cell && !to_cell->sequentials_.empty() && to_pin >= 0 && to_pin < n && is_driver_pin[to_pin])
                    continue;
            }
            enqueue(to_pin, false, pin_id, arc_id, "adjacent");
        }
    };

    auto expr_has_missing_node_port = [](const PowerExpr& expr,
                                         const LibertyCell* cell,
                                         const gp::GPNode& node) -> bool {
        if (!cell) return false;
        for (const auto& op : expr.ops()) {
            if (op.opcode != PowerExprOpcode::port || op.port_id < 0
                || op.port_id >= static_cast<int>(cell->ports_.size()))
                continue;
            const std::string& port_name = cell->ports_[op.port_id]->name;
            if (node.portMap.find(port_name) == node.portMap.end()) return true;
        }
        return false;
    };

    auto expr_has_known_activity_input = [&](const PowerExpr& expr,
                                             const LibertyCell* cell,
                                             const gp::GPNode& node,
                                             const std::unordered_map<int, int>* const_port_values,
                                             const std::unordered_set<int>* zero_density_ports = nullptr) -> bool {
        if (!cell) return false;
        bool has_port_ref = false;
        for (const auto& op : expr.ops()) {
            if (op.opcode != PowerExprOpcode::port || op.port_id < 0
                || op.port_id >= static_cast<int>(cell->ports_.size()))
                continue;
            has_port_ref = true;
            if (const_port_values && const_port_values->find(op.port_id) != const_port_values->end())
                return true;
            if (zero_density_ports && zero_density_ports->find(op.port_id) != zero_density_ports->end())
                return true;
            const std::string& port_name = cell->ports_[op.port_id]->name;
            auto pin_itr = node.portMap.find(port_name);
            if (pin_itr == node.portMap.end()) continue;
            const int pin_id = pin_itr->second;
            if (pin_id >= 0 && pin_id < n && act[pin_id].origin != 0) return true;
        }
        return !has_port_ref;
    };

    auto eval_cell_outputs = [&](int node_id, bool missing_port_outputs_only = false) {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return;
        const auto& node = gtdb.gpdb.getNodes()[node_id];
        LibertyCell* cell = get_cell(node_id);
        if (!cell) return;
        const auto& const_port_values = const_port_values_for_node(node_id, cell);

        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;

            PowerExpr expr;
            if (!expr.compile(port->function_expr_, cell)) continue;
            const bool has_missing_port = expr_has_missing_node_port(expr, cell, node);
            if (missing_port_outputs_only && !has_missing_port) continue;
            float density = 0.0f;
            float duty = 0.0f;
            if (evalPowerExprActivity(expr, cell, node, act, density, duty, &const_port_values)) {
                set_activity(pin_id, density, duty, 3, false);
            } else if (missing_port_outputs_only && has_missing_port) {
                set_activity(pin_id, act[pin_id].density, act[pin_id].duty, 3, false);
            }
        }
    };

    auto eval_output_pin_activity = [&](int pin_id, bool& changed) -> bool {
        changed = false;
        int node_id = pin_id >= 0 && pin_id < n ? pin_to_node[pin_id] : -1;
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return false;
        const auto& node = gtdb.gpdb.getNodes()[node_id];
        LibertyCell* cell = get_cell(node_id);
        if (!cell) return false;
        const auto& const_port_values = const_port_values_for_node(node_id, cell);
        int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) return false;
        LibertyPort* port = cell->ports_[port_offset];
        if (!port || port->direction_ != CellPortDirection::output) return false;
        bool computed = false;
        if (seq_pin_activity_valid[pin_id]) {
            const CpuActivity& seq_activity = seq_pin_activity[pin_id];
            changed = set_activity(pin_id, seq_activity.density, seq_activity.duty,
                                   seq_activity.origin, false, false);
            computed = true;
        }
        if (port->has_function_) {
            PowerExpr expr;
            if (!computed && expr.compile(port->function_expr_, cell)) {
                float density = 0.0f;
                float duty = 0.0f;
                if (evalPowerExprActivity(expr, cell, node, act, density, duty, &const_port_values)) {
                    changed = set_activity(pin_id, density, duty, 3, false, false);
                    computed = true;
                }
            }
        }
        const int cg_clk = clock_gate_clock_for_out[pin_id];
        const int cg_en = clock_gate_enable_for_out[pin_id];
        if (cg_clk >= 0 && cg_en >= 0 && (act[cg_clk].origin != 0 || act[cg_en].origin != 0)) {
            const float density = act[cg_clk].density * act[cg_en].duty +
                                  act[cg_en].density * act[cg_clk].duty;
            const float duty = act[cg_clk].duty * act[cg_en].duty;
            changed = set_activity(pin_id, density, duty, 3, false, false) || changed;
            computed = true;
        }
        return computed;
    };

    auto seed_reg_outputs = [&](int node_id) {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return;
        const auto& node = gtdb.gpdb.getNodes()[node_id];
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) return;
        const auto& const_port_values = const_port_values_for_node(node_id, cell);

        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            PowerExpr data_expr;
            PowerExpr clk_expr;
            if (!data_expr.compile(seq->next_state_expr_, cell)) continue;
            if (!clk_expr.compile(seqClockExpr(seq), cell)) continue;

            std::unordered_map<int, int> seq_data_const_port_values;
            std::unordered_set<int> seq_data_zero_density_ports;
            for (const auto& op : data_expr.ops()) {
                if (op.opcode != PowerExprOpcode::port || op.port_id < 0
                    || op.port_id >= static_cast<int>(cell->ports_.size()))
                    continue;
                const std::string& port_name = cell->ports_[op.port_id]->name;
                if (node.portMap.find(port_name) == node.portMap.end())
                    seq_data_const_port_values[op.port_id] = 0;
                if (ignore_scan_enable_density && cell->ports_[op.port_id]
                    && cell->ports_[op.port_id]->nextstate_type_ == "scan_enable")
                    seq_data_zero_density_ports.insert(op.port_id);
            }

            float in_density = 0.0f, in_duty = 0.0f;
            float clk_density_eval = 0.0f, clk_duty = 0.5f;
            const auto* zero_density_ports =
                seq_data_zero_density_ports.empty() ? nullptr : &seq_data_zero_density_ports;
            if (require_known_seq_data
                && !expr_has_known_activity_input(data_expr, cell, node, &seq_data_const_port_values,
                                                  zero_density_ports))
                continue;
            if (!evalPowerExprActivity(data_expr, cell, node, act, in_density, in_duty,
                                       &seq_data_const_port_values, zero_density_ports)) continue;
            if (!evalPowerExprActivity(clk_expr, cell, node, act, clk_density_eval, clk_duty, &const_port_values)) {
                clk_density_eval = clock_density;
                clk_duty = 0.5f;
            }

            float out_density = in_density;
            float out_duty = in_duty;
            if (in_density > clk_density_eval / 2.0f) {
                if (!seq->is_latch_)
                    out_density = 2.0f * in_duty * (1.0f - in_duty) * clk_density_eval;
                else
                    out_density = in_density * clk_duty;
            }

            const std::string seq_out = normalize_expr(seq->output_name_);
            const std::string seq_out_inv = normalize_expr(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
                const std::string func = normalize_expr(port->function_expr_);
                if (!seq_out.empty() && func == seq_out) {
                    seq_pin_activity[pin_id] = CpuActivity{out_density, out_duty, 3};
                    seq_pin_activity_valid[pin_id] = 1;
                    emit_path_trace("seq_seed", -1, -1, pin_id,
                                    act[pin_id].density, act[pin_id].duty,
                                    out_density, out_duty,
                                    true, true, static_cast<int>(pending_regs.size()),
                                    "q");
                    enqueue(pin_id, false, pin_id, -1, "seq_seed_output");
                } else if (!seq_out_inv.empty() && func == seq_out_inv) {
                    const float inv_duty = 1.0f - out_duty;
                    seq_pin_activity[pin_id] = CpuActivity{out_density, inv_duty, 3};
                    seq_pin_activity_valid[pin_id] = 1;
                    emit_path_trace("seq_seed", -1, -1, pin_id,
                                    act[pin_id].density, act[pin_id].duty,
                                    out_density, inv_duty,
                                    true, true, static_cast<int>(pending_regs.size()),
                                    "qn");
                    enqueue(pin_id, false, pin_id, -1, "seq_seed_output");
                }
            }
        }
    };

    std::vector<std::vector<PowerTraceEdge>> seq_reverse_edges(n);
    int next_seq_trace_arc = -1000000;
    auto collect_expr_pins = [&](const PowerExpr& expr,
                                 const LibertyCell* cell,
                                 const gp::GPNode& node,
                                 std::vector<int>& pins) {
        if (!cell) return;
        for (const auto& op : expr.ops()) {
            if (op.opcode != PowerExprOpcode::port || op.port_id < 0
                || op.port_id >= static_cast<int>(cell->ports_.size()))
                continue;
            const std::string& port_name = cell->ports_[op.port_id]->name;
            auto pin_itr = node.portMap.find(port_name);
            if (pin_itr == node.portMap.end()) continue;
            const int pin_id = pin_itr->second;
            if (pin_id >= 0 && pin_id < n
                && std::find(pins.begin(), pins.end(), pin_id) == pins.end())
                pins.push_back(pin_id);
        }
    };
    auto add_seq_reverse_edge = [&](int from_pin, int to_pin, const char* reason) {
        if (from_pin < 0 || from_pin >= n || to_pin < 0 || to_pin >= n) return;
        seq_reverse_edges[to_pin].push_back({next_seq_trace_arc--, from_pin, to_pin, reason});
    };
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) continue;
        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            PowerExpr data_expr;
            PowerExpr clk_expr;
            if (!data_expr.compile(seq->next_state_expr_, cell)) continue;
            clk_expr.compile(seqClockExpr(seq), cell);
            std::vector<int> data_pins;
            std::vector<int> clock_pins_expr;
            collect_expr_pins(data_expr, cell, node, data_pins);
            collect_expr_pins(clk_expr, cell, node, clock_pins_expr);
            const std::string seq_out = normalize_expr(seq->output_name_);
            const std::string seq_out_inv = normalize_expr(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
                const std::string func = normalize_expr(port->function_expr_);
                if ((!seq_out.empty() && func == seq_out) ||
                    (!seq_out_inv.empty() && func == seq_out_inv)) {
                    for (int pred_pin : data_pins) add_seq_reverse_edge(pred_pin, pin_id, "seq_data");
                    for (int pred_pin : clock_pins_expr) add_seq_reverse_edge(pred_pin, pin_id, "seq_clock");
                }
            }
        }
    }

    bool level_lifo = true;
    if (const char* order_env = std::getenv("XPLACE_POWER_ACTIVITY_LEVEL_ORDER")) {
        std::string order(order_env);
        std::transform(order.begin(), order.end(), order.begin(), [](unsigned char c) { return std::tolower(c); });
        if (order == "fifo") level_lifo = false;
    }

    auto run_queue = [&](int pass) {
        path_trace_pass = pass;
        while (!nonempty_queue_levels.empty()) {
            const int level = *nonempty_queue_levels.begin();
            path_trace_level_tag = std::string("level:") + std::to_string(level);
            auto& queue = level_queues[level];
            if (queue.empty()) {
                nonempty_queue_levels.erase(level);
                continue;
            }
            int pin_id;
            if (level_lifo) {
                pin_id = queue.back();
                queue.pop_back();
            } else {
                pin_id = queue.front();
                queue.pop_front();
            }
            if (queue.empty()) nonempty_queue_levels.erase(level);
                bool force_visit = force_propagate_on_visit[pin_id] != 0;
                force_propagate_on_visit[pin_id] = 0;
                in_queue[pin_id] = 0;
                emit_path_trace("visit", -1, -1, pin_id,
                                act[pin_id].density, act[pin_id].duty,
                                act[pin_id].density, act[pin_id].duty,
                                force_visit, false, static_cast<int>(pending_regs.size()),
                                force_visit ? "force_visit" : "queue");

                bool changed = false;
                if (is_load_pin[pin_id]) {
                    int net_id = pin_to_net[pin_id];
                    const int driver_pin = (net_id >= 0 && net_id < static_cast<int>(net_driver_pin.size()))
                        ? net_driver_pin[net_id] : -1;
                    if (driver_pin >= 0 && driver_pin < n && driver_pin != pin_id
                        && act[driver_pin].origin != 0) {
                        if (trace_matches(pin_id)) {
                            std::cerr << "[power_activity_trace_net_sink] sink=" << gtdb.pin_names[pin_id]
                                      << " from=" << gtdb.pin_names[driver_pin]
                                      << " driver_level=" << pin_level[driver_pin]
                                      << " sink_level=" << pin_level[pin_id]
                                      << " density=" << act[driver_pin].density
                                      << " duty=" << act[driver_pin].duty
                                      << std::endl;
                        }
                        emit_path_trace("net_sink", -1, driver_pin, pin_id,
                                        act[pin_id].density, act[pin_id].duty,
                                        act[driver_pin].density, act[driver_pin].duty,
                                        false, false, static_cast<int>(pending_regs.size()),
                                        "copy_driver_activity");
                        changed = set_activity(pin_id, act[driver_pin].density, act[driver_pin].duty, 3, false, false);
                    }
                }

                if (is_driver_pin[pin_id]) {
                    bool output_changed = false;
                    bool computed = eval_output_pin_activity(pin_id, output_changed);
                    if (computed)
                        changed = changed || output_changed || force_visit;
                    else
                        changed = changed || force_visit;
                }

                if (changed) {
                    int node_id = pin_id >= 0 && pin_id < n ? pin_to_node[pin_id] : -1;
                    LibertyCell* cell = get_cell(node_id);
                    if (is_load_pin[pin_id] && cell && cell->sequentials_.empty())
                        eval_cell_outputs(node_id, true);
                    if (is_load_pin[pin_id] && cell && !cell->sequentials_.empty()
                        && (mark_seq_clock_loads || !is_seq_clock_input_pin(pin_id)))
                        mark_pending_reg(node_id, pin_id);
                    if (is_load_pin[pin_id] && pin_id >= 0 && pin_id < n &&
                        clock_gate_out_for_input[pin_id] >= 0) {
                        enqueue(clock_gate_out_for_input[pin_id], false, pin_id, -1, "clock_gate");
                    }
                    enqueue_adjacent_vertices(pin_id);
                }
        }
    };

    std::vector<int> clock_pins = build_clock_pins();
    std::vector<uint8_t> is_clock_pin(n, 0);
    for (int pin_id : clock_pins) {
        if (pin_id >= 0 && pin_id < n) is_clock_pin[pin_id] = 1;
        if (pin_id >= 0 && pin_id < n) clock_activity_protected[pin_id] = 1;
    }
    std::vector<uint8_t> is_primary_input(n, 0);
    std::vector<uint8_t> actual_seed_seen(n, 0);
    const bool seed_timing_zero_indeg_roots =
        readPowerBoolEnv("XPLACE_POWER_SEED_TIMING_ZERO_INDEG", true);
    const bool seed_floating_load_roots =
        readPowerBoolEnv("XPLACE_POWER_SEED_FLOATING_LOADS", true);
    auto seed_root_pin = [&](int pin_id, const char* reason) {
        if (pin_id < 0 || pin_id >= n || is_clock_pin[pin_id]) return;
        actual_seed_seen[pin_id] = 1;
        if (set_activity(pin_id, default_density, 0.5f, 1, false, false))
            enqueue_adjacent_vertices(pin_id);
    };
    for (int pin_id : gtdb.primary_inputs) {
        if (pin_id >= 0 && pin_id < n) is_primary_input[pin_id] = 1;
        if (pin_id >= 0 && pin_id < n && is_driver_pin[pin_id] && !is_clock_pin[pin_id]) {
            seed_root_pin(pin_id, "primary_input");
        }
    }

    if (seed_timing_zero_indeg_roots) {
        for (int pin_id : gtdb.pin_frontiers) {
            if (pin_id < 0 || pin_id >= n) continue;
            if (is_primary_input[pin_id] || is_clock_pin[pin_id]) continue;
            seed_root_pin(pin_id, "timing_zero_indeg");
        }
    }

    // OpenSTA power levelization seeds root load pins too. These appear on
    // no-driver input nets and are not always represented in timing frontiers.
    if (seed_floating_load_roots) {
        for (int pin_id = 0; pin_id < n; pin_id++) {
            if (!is_load_pin[pin_id] || is_primary_input[pin_id] || is_clock_pin[pin_id]) continue;
            const int net_id = pin_to_net[pin_id];
            const int driver =
                (net_id >= 0 && net_id < static_cast<int>(net_driver_pin.size())) ? net_driver_pin[net_id] : -1;
            if (driver < 0)
                seed_root_pin(pin_id, "floating_load_input");
        }
    }

    // Constant-generator outputs are roots in OpenSTA's power graph.
    for (int pin_id = 0; pin_id < n; pin_id++) {
        if (!is_driver_pin[pin_id] || is_primary_input[pin_id] || is_clock_pin[pin_id]) continue;
        int node_id = pin_to_node[pin_id];
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
        bool has_input_pin = false;
        for (int node_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
            if (node_pin >= 0 && node_pin < n && is_load_pin[node_pin]) {
                has_input_pin = true;
                break;
            }
        }
        if (!has_input_pin) {
            seed_root_pin(pin_id, "const_output");
        }
    }

    for (int pin_id : clock_pins) {
        if (pin_id >= 0 && pin_id < n) actual_seed_seen[pin_id] = 1;
        auto [pin_density, pin_duty] = clock_activity_for_pin(pin_id);
        const int node_id = pin_id >= 0 && pin_id < n ? pin_to_node[pin_id] : -1;
        LibertyCell* cell = get_cell(node_id);
        const bool enqueue_clock_tree = pin_id >= 0 && pin_id < n && is_load_pin[pin_id]
            && (!cell || cell->sequentials_.empty());
        if (set_activity(pin_id, pin_density, pin_duty, 2, true, false) && enqueue_clock_tree)
            enqueue_adjacent_vertices(pin_id);
    }

    auto dump_trace_paths = [&]() {
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
                LibertyCell* to_cell = get_cell(to_node);
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
    };
    dump_trace_paths();

    int max_activity_passes = 50;
    if (const char* env = std::getenv("XPLACE_POWER_ACTIVITY_MAX_PASSES")) {
        max_activity_passes = std::max(1, std::atoi(env));
    }
    auto trace_pending_regs = [&](int pass) {
        if (!std::getenv("XPLACE_POWER_ACTIVITY_TRACE_REGS")) return;
        for (int node_id : pending_regs) {
            if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
            std::cerr << "[power_activity_reg] pass=" << pass
                      << " node=" << node_id
                      << " inst=" << gtdb.gpdb.getNodes()[node_id].getName()
                      << std::endl;
        }
    };
    auto emit_trace = [&](const char* tag, int pass, size_t pending_count) {
        for (int pin_id : trace_pin_ids) {
            if (pin_id < 0 || pin_id >= n) continue;
            const bool first_nonzero = act[pin_id].density > 0.0f && !trace_first_seen[pin_id];
            if (first_nonzero) trace_first_seen[pin_id] = 1;
            const int node_id = pin_to_node[pin_id];
            const bool node_pending = node_id >= 0 && node_id < static_cast<int>(pending_reg_flag.size()) &&
                                      pending_reg_flag[node_id];
            const bool cell_seq = get_cell(node_id) && !get_cell(node_id)->sequentials_.empty();
            std::cerr << "[power_activity_trace_cpu] tag=" << tag
                      << " pass=" << pass
                      << " pending=" << pending_count
                      << " pin_id=" << pin_id
                      << " pin=" << gtdb.pin_names[pin_id]
                      << " density=" << act[pin_id].density
                      << " duty=" << act[pin_id].duty
                      << " origin=" << act[pin_id].origin
                      << " first_nonzero=" << (first_nonzero ? 1 : 0)
                      << " is_load=" << static_cast<int>(is_load_pin[pin_id])
                      << " is_driver=" << static_cast<int>(is_driver_pin[pin_id])
                      << " node=" << node_id
                      << " node_pending=" << (node_pending ? 1 : 0)
                      << " cell_seq=" << (cell_seq ? 1 : 0);
            if (node_id >= 0 && node_id < static_cast<int>(gtdb.gpdb.getNodes().size())) {
                std::cerr << " inst=" << gtdb.gpdb.getNodes()[node_id].getName();
            }
            std::cerr << std::endl;
        }
    };
    std::ofstream activity_snapshot_csv;
    int activity_snapshot_max_pass = 6;
    if (const char* snapshot_csv_env = std::getenv("XPLACE_POWER_ACTIVITY_SNAPSHOT_CSV")) {
        if (snapshot_csv_env[0] != '\0') {
            activity_snapshot_max_pass =
                readPowerActivitySnapshotMaxPass("XPLACE_POWER_ACTIVITY_SNAPSHOT_MAX_PASS", 6);
            activity_snapshot_csv.open(snapshot_csv_env);
            if (activity_snapshot_csv) {
                activity_snapshot_csv
                    << "engine,split,design,pass,tag,pin_id,pin_name,pin_name_norm,"
                    << "inst_name,port_name,is_load,is_driver,node_id,node_pending,"
                    << "cell_seq,density,duty,origin,pending_count\n";
            }
        }
    }
    const char* snapshot_split_env = std::getenv("XPLACE_POWER_ACTIVITY_SNAPSHOT_SPLIT");
    if (!snapshot_split_env || snapshot_split_env[0] == '\0')
        snapshot_split_env = std::getenv("DESIGN_SET");
    const char* snapshot_design_env = std::getenv("XPLACE_POWER_ACTIVITY_SNAPSHOT_DESIGN");
    if (!snapshot_design_env || snapshot_design_env[0] == '\0')
        snapshot_design_env = std::getenv("DESIGN_NAME");
    const std::string snapshot_split = snapshot_split_env ? snapshot_split_env : "";
    const std::string snapshot_design = snapshot_design_env ? snapshot_design_env : "";
    auto emit_activity_snapshot = [&](const char* tag, int pass, size_t pending_count) {
        if (!activity_snapshot_csv || pass > activity_snapshot_max_pass) return;
        for (int pin_id = 0; pin_id < n; pin_id++) {
            const std::string& pin_name = gtdb.pin_names[pin_id];
            const int node_id = pin_to_node[pin_id];
            std::string inst_name = pin_name;
            std::string port_name;
            const size_t colon = pin_name.rfind(':');
            if (colon != std::string::npos) {
                inst_name = pin_name.substr(0, colon);
                port_name = pin_name.substr(colon + 1);
            } else if (node_id >= 0 && node_id < static_cast<int>(gtdb.gpdb.getNodes().size())) {
                inst_name = gtdb.gpdb.getNodes()[node_id].getName();
            }
            const bool node_pending = node_id >= 0 && node_id < static_cast<int>(pending_reg_flag.size()) &&
                                      pending_reg_flag[node_id];
            const bool cell_seq = get_cell(node_id) && !get_cell(node_id)->sequentials_.empty();
            activity_snapshot_csv
                << "xplace_cpu,"
                << csvEscapePowerActivitySnapshot(snapshot_split) << ','
                << csvEscapePowerActivitySnapshot(snapshot_design) << ','
                << pass << ','
                << csvEscapePowerActivitySnapshot(tag ? tag : "") << ','
                << pin_id << ','
                << csvEscapePowerActivitySnapshot(pin_name) << ','
                << csvEscapePowerActivitySnapshot(normalizePowerActivitySnapshotName(pin_name)) << ','
                << csvEscapePowerActivitySnapshot(inst_name) << ','
                << csvEscapePowerActivitySnapshot(port_name) << ','
                << static_cast<int>(is_load_pin[pin_id]) << ','
                << static_cast<int>(is_driver_pin[pin_id]) << ','
                << node_id << ','
                << (node_pending ? 1 : 0) << ','
                << (cell_seq ? 1 : 0) << ','
                << std::setprecision(10) << act[pin_id].density << ','
                << std::setprecision(10) << act[pin_id].duty << ','
                << act[pin_id].origin << ','
                << pending_count << '\n';
        }
        activity_snapshot_csv.flush();
    };
    const char* pending_seq_dump_file = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_FILE");
    int pending_seq_dump_pass = -1;
    if (const char* env = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_PASS"))
        pending_seq_dump_pass = std::atoi(env);
    std::string pending_seq_dump_tag = "after_pass";
    if (const char* env = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_TAG"))
        pending_seq_dump_tag = env;
    auto dump_cpu_pending_regs = [&](const char* tag, int pass) {
        if (!pending_seq_dump_file || pending_seq_dump_file[0] == '\0') return;
        if (pending_seq_dump_pass >= 0 && pass != pending_seq_dump_pass) return;
        if (pending_seq_dump_tag != (tag ? tag : "")) return;
        std::ofstream out(pending_seq_dump_file, std::ios::app);
        if (!out) return;
        out << "engine,pass,tag,node_id,inst_name,seq_id,q_pin,qn_pin,pin_id,pin_name,"
               "trigger_pin,trigger_pin_name,trigger_port,trigger_density,trigger_duty,trigger_origin\n";
        for (int node_id : pending_regs) {
            if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
            const auto& node = gtdb.gpdb.getNodes()[node_id];
            const int trigger_pin = node_id < static_cast<int>(pending_reg_trigger_pin.size())
                ? pending_reg_trigger_pin[node_id] : -1;
            const char* trigger_name = (trigger_pin >= 0 && trigger_pin < n)
                ? gtdb.pin_names[trigger_pin].c_str() : "";
            const char* trigger_port = (trigger_pin >= 0 && trigger_pin < n)
                ? gtdb.pin_names[trigger_pin].c_str() : "";
            const float trigger_density = (trigger_pin >= 0 && trigger_pin < n)
                ? act[trigger_pin].density : 0.0f;
            const float trigger_duty = (trigger_pin >= 0 && trigger_pin < n)
                ? act[trigger_pin].duty : 0.0f;
            const int trigger_origin = (trigger_pin >= 0 && trigger_pin < n)
                ? act[trigger_pin].origin : 0;
            if (trigger_pin >= 0 && trigger_pin < n) {
                const std::string& full_name = gtdb.pin_names[trigger_pin];
                const size_t slash = full_name.rfind('/');
                if (slash != std::string::npos) trigger_port = full_name.c_str() + slash + 1;
            }
            bool wrote_pin = false;
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n || !is_seq_output_pin[pin_id]) continue;
                out << "xplace_cpu," << pass << ','
                    << csvEscapePowerActivitySnapshot(tag ? tag : "") << ','
                    << node_id << ','
                    << csvEscapePowerActivitySnapshot(node.getName()) << ",-1,"
                    << pin_id << ",-1," << pin_id << ','
                    << csvEscapePowerActivitySnapshot(gtdb.pin_names[pin_id]) << ','
                    << trigger_pin << ','
                    << csvEscapePowerActivitySnapshot(trigger_name) << ','
                    << csvEscapePowerActivitySnapshot(trigger_port) << ','
                    << std::setprecision(10) << trigger_density << ','
                    << std::setprecision(10) << trigger_duty << ','
                    << trigger_origin << '\n';
                wrote_pin = true;
            }
            if (!wrote_pin) {
                out << "xplace_cpu," << pass << ','
                    << csvEscapePowerActivitySnapshot(tag ? tag : "") << ','
                    << node_id << ','
                    << csvEscapePowerActivitySnapshot(node.getName())
                    << ",-1,-1,-1,-1,,"
                    << trigger_pin << ','
                    << csvEscapePowerActivitySnapshot(trigger_name) << ','
                    << csvEscapePowerActivitySnapshot(trigger_port) << ','
                    << std::setprecision(10) << trigger_density << ','
                    << std::setprecision(10) << trigger_duty << ','
                    << trigger_origin << '\n';
            }
        }
    };
    emit_trace("after_seed", 0, pending_regs.size());
    emit_activity_snapshot("after_seed", 0, pending_regs.size());
    dump_cpu_pending_regs("after_seed", 0);
    trace_pending_regs(0);
    // Initial combinational propagation from roots/clock network.
    run_queue(0);
    emit_trace("after_comb", 0, pending_regs.size());
    emit_activity_snapshot("after_comb", 0, pending_regs.size());
    dump_cpu_pending_regs("after_comb", 0);
    for (int pass = 1; !pending_regs.empty() && pass < max_activity_passes; pass++) {
        std::vector<int> regs = std::move(pending_regs);
        pending_regs.clear();
        path_trace_pass = pass;
        path_trace_level_tag = "seq_seed";
        for (int node_id : regs) {
            if (node_id >= 0 && node_id < static_cast<int>(pending_reg_flag.size()))
                pending_reg_flag[node_id] = 0;
            if (node_id >= 0 && node_id < static_cast<int>(pending_reg_trigger_pin.size()))
                pending_reg_trigger_pin[node_id] = -1;
            seed_reg_outputs(node_id);
        }
        emit_trace("after_seq_seed", pass, regs.size());
        emit_activity_snapshot("after_seq_seed", pass, regs.size());
        dump_cpu_pending_regs("after_seq_seed", pass);
        run_queue(pass);
        trace_pending_regs(pass);
        emit_trace("after_pass", pass, pending_regs.size());
        emit_activity_snapshot("after_pass", pass, pending_regs.size());
        dump_cpu_pending_regs("after_pass", pass);
    }

    auto out = torch::empty({n, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto acc = out.accessor<float, 2>();
    for (int i = 0; i < n; i++) {
        acc[i][0] = act[i].density;
        acc[i][1] = act[i].duty;
        acc[i][2] = static_cast<float>(act[i].origin);
    }
    return out;
}

torch::Tensor GPUTimer::compute_power_activity_cuda(torch::Tensor* inst_switching_cpu, torch::Tensor* pin_switching_cpu, torch::Tensor* inst_internal_cpu, torch::Tensor* internal_row_power_cpu, torch::Tensor* internal_row_meta_cpu, torch::Tensor* inst_leakage_cpu, torch::Tensor* leakage_row_power_cpu, torch::Tensor* leakage_row_meta_cpu) {
    const int n = static_cast<int>(gtdb.pin_names.size());
    if (n <= 0) return torch::empty({0, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("report_power_activity_cuda requires CUDA");
    }
    // Some existing init kernels leave a stale CUDA error status that CPU reports ignore.
    // Clear it before allocating/uploading the Plan-A power activity data structures.
    clear_power_cuda_error();

    const double sdc_time_scale =
        canonicalPowerTimeScale(gtdb.sdc_time_unit.has_value() ? *gtdb.sdc_time_unit : gtdb.time_unit);
    double min_period_sec = std::numeric_limits<double>::infinity();
    for (auto& kv : gtdb.clocks) {
        const double period_sec = static_cast<double>(kv.second.period()) * sdc_time_scale;
        if (period_sec > 0.0) min_period_sec = std::min(min_period_sec, period_sec);
    }
    if (!std::isfinite(min_period_sec) || min_period_sec <= 0.0) {
        const double fallback_scale = canonicalPowerTimeScale(gtdb.time_unit);
        min_period_sec = fallback_scale > 0.0 ? fallback_scale : 1.0e-9;
    }
    const float default_density = static_cast<float>(0.1 / min_period_sec);
    const float clock_density = static_cast<float>(2.0 / min_period_sec);
    const bool need_switching_power = inst_switching_cpu || pin_switching_cpu;
    const bool need_internal_power =
        inst_internal_cpu || internal_row_power_cpu || internal_row_meta_cpu;
    const bool need_leakage_power =
        inst_leakage_cpu || leakage_row_power_cpu || leakage_row_meta_cpu;
    const bool want_activity_cpu = !inst_switching_cpu && !pin_switching_cpu &&
        !inst_internal_cpu && !internal_row_power_cpu && !internal_row_meta_cpu &&
        !inst_leakage_cpu && !leakage_row_power_cpu && !leakage_row_meta_cpu;

    std::vector<int> h_pin_to_node(n, -1);
    std::vector<int> h_pin_to_net(n, -1);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        int pin_id = static_cast<int>(pin.getId());
        if (pin_id >= 0 && pin_id < n) {
            h_pin_to_node[pin_id] = static_cast<int>(pin.getParNodeId());
            h_pin_to_net[pin_id] = static_cast<int>(pin.getParNetId());
        }
    }

    auto get_cell = [&](int node_id) -> LibertyCell* {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) return nullptr;
        int libcell_id = gtdb.cell_node_type_map[node_id];
        if (libcell_id < 0 || libcell_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return nullptr;
        auto* cell_type = gtdb.rawdb.celltypes[libcell_id];
        return cell_type ? cell_type->liberty_cell : nullptr;
    };
    auto is_io_node = [&](int node_id) -> bool {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return false;
        const std::string& node_type = gtdb.gpdb.getNodes()[node_id].getNodeType();
        return node_type == "IOPin" || node_type == "FloatIOPin";
    };
    auto normalize_expr = [](std::string expr) {
        expr.erase(std::remove_if(expr.begin(), expr.end(), [](unsigned char c) { return std::isspace(c); }), expr.end());
        if (expr.size() >= 2 && expr.front() == '"' && expr.back() == '"') expr = expr.substr(1, expr.size() - 2);
        return expr;
    };
    auto parse_const_net_value = [](std::string name) -> int {
        name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char c) { return std::isspace(c); }),
                   name.end());
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (name == "0" || name == "1'b0" || name == "1'd0" || name == "1'h0") return 0;
        if (name == "1" || name == "1'b1" || name == "1'd1" || name == "1'h1") return 1;
        const size_t quote = name.find('\'');
        if (quote != std::string::npos && quote + 2 < name.size()) {
            const std::string digits = name.substr(quote + 2);
            if (!digits.empty() && digits.find_first_not_of("0") == std::string::npos) return 0;
            if (!digits.empty() && digits.find_first_not_of("1") == std::string::npos) return 1;
        }
        return -1;
    };
    std::unordered_map<std::string, int> const_port_file_values;
    bool const_port_file_loaded = false;
    auto load_const_port_file = [&]() {
        if (const_port_file_loaded) return;
        const_port_file_loaded = true;
        const char* file_name = std::getenv("XPLACE_POWER_CONST_PORT_FILE");
        if (!file_name || file_name[0] == '\0') return;
        std::ifstream stream(file_name);
        if (!stream) return;
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string inst;
            std::string port;
            std::string value;
            if (!std::getline(ss, inst, ',')) continue;
            if (!std::getline(ss, port, ',')) continue;
            if (!std::getline(ss, value, ',')) continue;
            if (inst == "inst_name" || inst == "inst") continue;
            const int const_value = parse_const_net_value(value);
            if (const_value < 0) continue;
            const std::string key = normalizePowerActivitySnapshotName(inst) + "/" +
                                    normalizePowerActivitySnapshotName(port);
            const_port_file_values[key] = const_value;
        }
    };
    auto const_port_value_for_node = [&](const gp::GPNode& node, const std::string& port_name) -> int {
        load_const_port_file();
        const std::string key = normalizePowerActivitySnapshotName(node.getName()) + "/" +
                                normalizePowerActivitySnapshotName(port_name);
        auto const_itr = const_port_file_values.find(key);
        if (const_itr != const_port_file_values.end()) return const_itr->second;
        const int raw_cell_id = static_cast<int>(node.getOriDBId());
        if (raw_cell_id < 0 || raw_cell_id >= static_cast<int>(gtdb.rawdb.cells.size()))
            return -1;
        db::Cell* dbcell = gtdb.rawdb.cells[raw_cell_id];
        db::Pin* dbpin = dbcell ? dbcell->pin(port_name) : nullptr;
        return (dbpin && dbpin->net) ? parse_const_net_value(dbpin->net->name) : -1;
    };

    std::vector<uint8_t> h_is_load_pin(n, 0), h_is_driver_pin(n, 0), h_is_cell_pin(n, 0), h_is_seq_output_pin(n, 0);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        int pin_id = static_cast<int>(pin.getId());
        if (pin_id < 0 || pin_id >= n) continue;
        const int node_id = h_pin_to_node[pin_id];
        if (!is_io_node(node_id)) h_is_cell_pin[pin_id] = 1;
        if (is_io_node(node_id)
            && std::find(gtdb.primary_inputs.begin(), gtdb.primary_inputs.end(), pin_id) != gtdb.primary_inputs.end()) {
            h_is_driver_pin[pin_id] = 1;
            continue;
        }
        if (is_io_node(node_id)
            && std::find(gtdb.primary_outputs.begin(), gtdb.primary_outputs.end(), pin_id) != gtdb.primary_outputs.end()) {
            h_is_load_pin[pin_id] = 1;
            continue;
        }
        LibertyCell* cell = get_cell(node_id);
        int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        if (!cell || port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
        LibertyPort* port = cell->ports_[port_offset];
        if (!port) continue;
        if (port->direction_ == CellPortDirection::input) h_is_load_pin[pin_id] = 1;
        else if (port->direction_ == CellPortDirection::output) h_is_driver_pin[pin_id] = 1;
    }

    std::vector<int> h_net_driver_pin(gtdb.gpdb.getNets().size(), -1);
    for (const auto& net : gtdb.gpdb.getNets()) {
        const int net_id = static_cast<int>(net.getId());
        if (net_id < 0 || net_id >= static_cast<int>(h_net_driver_pin.size())) continue;
        for (int pin_id : net.pins()) {
            if (pin_id >= 0 && pin_id < n && h_is_driver_pin[pin_id]) {
                h_net_driver_pin[net_id] = pin_id;
                break;
            }
        }
    }
    std::vector<int> h_clock_gate_out_for_input(n, -1);
    std::vector<int> h_clock_gate_clock_for_out(n, -1);
    std::vector<int> h_clock_gate_enable_for_out(n, -1);
    std::vector<uint8_t> h_is_clock_gate_clock_pin(n, 0);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell) continue;
        int clk_pin = -1;
        int enable_pin = -1;
        int out_pin = -1;
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port) continue;
            if (port->is_clock_gate_clock_) clk_pin = pin_id;
            if (port->is_clock_gate_enable_) enable_pin = pin_id;
            if (port->is_clock_gate_out_) out_pin = pin_id;
        }
        if (out_pin >= 0 && clk_pin >= 0 && enable_pin >= 0) {
            h_clock_gate_clock_for_out[out_pin] = clk_pin;
            h_clock_gate_enable_for_out[out_pin] = enable_pin;
            h_clock_gate_out_for_input[clk_pin] = out_pin;
            h_clock_gate_out_for_input[enable_pin] = out_pin;
            h_is_clock_gate_clock_pin[clk_pin] = 1;
        }
    }

    auto build_clock_pins = [&]() {
        const int num_nets = static_cast<int>(gtdb.gpdb.getNets().size());
        std::vector<uint8_t> is_clock_net(num_nets, 0);
        auto mark_net = [&](int net_id) -> bool {
            if (net_id < 0 || net_id >= num_nets || is_clock_net[net_id]) return false;
            is_clock_net[net_id] = 1;
            return true;
        };

        if (gtdb.net_is_clock.size() == static_cast<size_t>(num_nets)) {
            for (int net_id = 0; net_id < num_nets; net_id++) {
                if (gtdb.net_is_clock[net_id]) mark_net(net_id);
            }
        }
        for (int pin_id = 0; pin_id < n; pin_id++) {
            if (h_is_clock_gate_clock_pin[pin_id]) mark_net(h_pin_to_net[pin_id]);
        }

        std::vector<uint8_t> forward_clock_net(num_nets, 0);
        std::deque<int> forward_queue;
        auto mark_forward_net = [&](int net_id) {
            if (net_id < 0 || net_id >= num_nets || forward_clock_net[net_id]) return;
            forward_clock_net[net_id] = 1;
            forward_queue.push_back(net_id);
        };
        if (gtdb.net_is_clock.size() == static_cast<size_t>(num_nets)) {
            for (int net_id = 0; net_id < num_nets; net_id++) {
                if (gtdb.net_is_clock[net_id]) mark_forward_net(net_id);
            }
        }
        std::vector<int> extra_clock_pins;
        std::vector<uint8_t> extra_clock_pin_seen(n, 0);
        auto add_extra_clock_pin = [&](int pin_id) {
            if (pin_id < 0 || pin_id >= n || extra_clock_pin_seen[pin_id]) return;
            extra_clock_pin_seen[pin_id] = 1;
            extra_clock_pins.push_back(pin_id);
        };
        auto is_core_comb_node = [&](int node_id, LibertyCell* cell) {
            if (!cell || !cell->sequentials_.empty()) return false;
            if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) return false;
            const int cell_type_id = gtdb.cell_node_type_map[node_id];
            if (cell_type_id < 0 || cell_type_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return false;
            db::CellType* cell_type = gtdb.rawdb.celltypes[cell_type_id];
            return cell_type && cell_type->cls == "CORE";
        };
        auto is_clock_transparent_from_pin = [&](int node_id, LibertyCell* cell, int in_pin_id) {
            if (!is_core_comb_node(node_id, cell)) return false;
            if (in_pin_id < 0 || in_pin_id >= n ||
                in_pin_id >= static_cast<int>(gtdb.pin_id2port_offset_id.size()))
                return false;
            const int in_port = gtdb.pin_id2port_offset_id[in_pin_id];
            if (in_port < 0 || in_port >= static_cast<int>(cell->ports_.size())) return false;

            bool output_seen = false;
            for (int out_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
                if (out_pin < 0 || out_pin >= n || !h_is_driver_pin[out_pin]) continue;
                if (out_pin >= static_cast<int>(gtdb.pin_id2port_offset_id.size())) return false;
                const int out_port = gtdb.pin_id2port_offset_id[out_pin];
                if (out_port < 0 || out_port >= static_cast<int>(cell->ports_.size())) return false;
                LibertyPort* port = cell->ports_[out_port];
                if (!port || !port->has_function_) return false;
                PowerExpr expr;
                if (!expr.compile(port->function_expr_, cell)) return false;
                const auto& ops = expr.ops();
                const bool direct =
                    ops.size() == 1 && ops[0].opcode == PowerExprOpcode::port && ops[0].port_id == in_port;
                const bool inverted =
                    ops.size() == 2 && ops[0].opcode == PowerExprOpcode::port && ops[0].port_id == in_port &&
                    ops[1].opcode == PowerExprOpcode::logical_not;
                if (!direct && !inverted) return false;
                output_seen = true;
            }
            return output_seen;
        };
        for (size_t queue_pos = 0; queue_pos < forward_queue.size(); ++queue_pos) {
            const int net_id = forward_queue[queue_pos];
            if (net_id < 0 || net_id >= num_nets) continue;
            for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
                if (pin_id < 0 || pin_id >= n || !h_is_load_pin[pin_id]) continue;
                const int node_id = h_pin_to_node[pin_id];
                LibertyCell* cell = get_cell(node_id);
                if (!is_core_comb_node(node_id, cell)) continue;
                add_extra_clock_pin(pin_id);
                if (!is_clock_transparent_from_pin(node_id, cell, pin_id)) continue;
                for (int out_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
                    if (out_pin >= 0 && out_pin < n && h_is_driver_pin[out_pin])
                        mark_forward_net(h_pin_to_net[out_pin]);
                }
            }
        }
        std::vector<int> clock_pins;
        for (int net_id = 0; net_id < num_nets; net_id++) {
            if (!is_clock_net[net_id]) continue;
            for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
                if (pin_id >= 0 && pin_id < n
                    && (h_is_load_pin[pin_id] || is_io_node(h_pin_to_node[pin_id])))
                    clock_pins.push_back(pin_id);
            }
        }
        clock_pins.insert(clock_pins.end(), extra_clock_pins.begin(), extra_clock_pins.end());
        std::sort(clock_pins.begin(), clock_pins.end());
        clock_pins.erase(std::unique(clock_pins.begin(), clock_pins.end()), clock_pins.end());
        return clock_pins;
    };
    std::vector<int> h_clock_pins = build_clock_pins();
    std::vector<float> h_clock_pin_densities;
    std::vector<float> h_clock_pin_duties;
    std::vector<uint8_t> h_clock_pin_enqueue;
    h_clock_pin_densities.reserve(h_clock_pins.size());
    h_clock_pin_duties.reserve(h_clock_pins.size());
    h_clock_pin_enqueue.reserve(h_clock_pins.size());
    auto clock_activity_for_pin = [&](int pin_id) -> std::pair<float, float> {
        float density = clock_density;
        float duty = 0.5f;
        if (pin_id >= 0 && pin_id < static_cast<int>(gtdb.pin_clock_periods.size())) {
            const float period = gtdb.pin_clock_periods[pin_id];
            if (std::isfinite(period) && period > 0.0f && sdc_time_scale > 0.0) {
                density = powerDensityForPeriod(2.0, period, sdc_time_scale);
                if (pin_id < static_cast<int>(gtdb.pin_clock_rise_edges.size())
                    && pin_id < static_cast<int>(gtdb.pin_clock_fall_edges.size())) {
                    const float rise = gtdb.pin_clock_rise_edges[pin_id];
                    const float fall = gtdb.pin_clock_fall_edges[pin_id];
                    if (std::isfinite(rise) && std::isfinite(fall)) {
                        const float candidate_duty = (fall - rise) / period;
                        if (std::isfinite(candidate_duty) && candidate_duty >= 0.0f && candidate_duty <= 1.0f)
                            duty = candidate_duty;
                    }
                }
            }
        }
        return {density, duty};
    };
    for (int pin_id : h_clock_pins) {
        auto [density, duty] = clock_activity_for_pin(pin_id);
        h_clock_pin_densities.push_back(density);
        h_clock_pin_duties.push_back(duty);
        const int node_id = pin_id >= 0 && pin_id < n ? h_pin_to_node[pin_id] : -1;
        LibertyCell* cell = get_cell(node_id);
        const bool enqueue_clock_tree = pin_id >= 0 && pin_id < n && h_is_load_pin[pin_id]
            && (!cell || cell->sequentials_.empty());
        h_clock_pin_enqueue.push_back(enqueue_clock_tree ? 1 : 0);
    }

    std::vector<GpuPowerExprOpHost> h_expr_ops;
    std::vector<int> h_expr_start;
    std::vector<int> h_expr_count;
    const bool ignore_scan_enable_density =
        readPowerBoolEnv("XPLACE_POWER_IGNORE_SCAN_ENABLE_DENSITY", false);
    auto add_expr = [&](const std::string& expr_str, LibertyCell* cell, const gp::GPNode& node,
                        bool* used_missing_const = nullptr,
                        bool zero_scan_enable_density = false) -> int {
        if (used_missing_const) *used_missing_const = false;
        if (!cell) return -1;
        PowerExpr expr;
        if (!expr.compile(expr_str, cell)) return -1;
        std::vector<GpuPowerExprOpHost> local_ops;
        local_ops.reserve(expr.ops().size());
        for (const auto& op : expr.ops()) {
            GpuPowerExprOpHost out;
            switch (op.opcode) {
                case PowerExprOpcode::port: {
                    if (op.port_id < 0 || op.port_id >= static_cast<int>(cell->ports_.size())) return -1;
                    const std::string& port_name = cell->ports_[op.port_id]->name;
                    auto pin_itr = node.portMap.find(port_name);
                    if (pin_itr != node.portMap.end()) {
                        out.op = 0;
                        out.arg = pin_itr->second;
                        out.var_key = op.port_id;
                        if (zero_scan_enable_density && cell->ports_[op.port_id]
                            && cell->ports_[op.port_id]->nextstate_type_ == "scan_enable")
                            out.zero_density = 1;
                    } else {
                        const int const_value = const_port_value_for_node(node, port_name);
                        out.op = const_value > 0 ? 2 : 1;
                        out.arg = -1;
                        out.var_key = op.port_id;
                        if (used_missing_const) *used_missing_const = true;
                    }
                    break;
                }
                case PowerExprOpcode::const_zero: out.op = 1; out.arg = -1; break;
                case PowerExprOpcode::const_one: out.op = 2; out.arg = -1; break;
                case PowerExprOpcode::logical_not: out.op = 3; out.arg = -1; break;
                case PowerExprOpcode::logical_and: out.op = 4; out.arg = -1; break;
                case PowerExprOpcode::logical_or: out.op = 5; out.arg = -1; break;
                case PowerExprOpcode::logical_xor: out.op = 6; out.arg = -1; break;
            }
            local_ops.push_back(out);
        }
        if (local_ops.empty()) return -1;
        int expr_id = static_cast<int>(h_expr_start.size());
        h_expr_start.push_back(static_cast<int>(h_expr_ops.size()));
        h_expr_count.push_back(static_cast<int>(local_ops.size()));
        h_expr_ops.insert(h_expr_ops.end(), local_ops.begin(), local_ops.end());
        return expr_id;
    };

    std::unordered_map<std::string, int> template_expr_cache;
    auto add_template_expr = [&](const std::string& expr_str, LibertyCell* cell) -> int {
        if (!cell) return -1;
        const std::string cache_key = cell->name + "|" + normalize_expr(expr_str);
        auto cache_itr = template_expr_cache.find(cache_key);
        if (cache_itr != template_expr_cache.end()) return cache_itr->second;
        PowerExpr expr;
        if (!expr.compile(expr_str, cell)) return -1;
        std::vector<GpuPowerExprOpHost> local_ops;
        local_ops.reserve(expr.ops().size());
        for (const auto& op : expr.ops()) {
            GpuPowerExprOpHost out;
            switch (op.opcode) {
                case PowerExprOpcode::port:
                    if (op.port_id < 0 || op.port_id >= static_cast<int>(cell->ports_.size())) return -1;
                    out.op = 0;
                    out.arg = -2 - op.port_id;
                    out.var_key = op.port_id;
                    break;
                case PowerExprOpcode::const_zero: out.op = 1; out.arg = -1; break;
                case PowerExprOpcode::const_one: out.op = 2; out.arg = -1; break;
                case PowerExprOpcode::logical_not: out.op = 3; out.arg = -1; break;
                case PowerExprOpcode::logical_and: out.op = 4; out.arg = -1; break;
                case PowerExprOpcode::logical_or: out.op = 5; out.arg = -1; break;
                case PowerExprOpcode::logical_xor: out.op = 6; out.arg = -1; break;
            }
            local_ops.push_back(out);
        }
        if (local_ops.empty()) return -1;
        int expr_id = static_cast<int>(h_expr_start.size());
        h_expr_start.push_back(static_cast<int>(h_expr_ops.size()));
        h_expr_count.push_back(static_cast<int>(local_ops.size()));
        h_expr_ops.insert(h_expr_ops.end(), local_ops.begin(), local_ops.end());
        template_expr_cache.emplace(cache_key, expr_id);
        return expr_id;
    };

    auto expr_contains_pin = [&](int expr_id, int pin_id) -> bool {
        if (expr_id < 0 || expr_id >= static_cast<int>(h_expr_start.size())) return false;
        const int start = h_expr_start[expr_id];
        const int count = h_expr_count[expr_id];
        for (int k = 0; k < count; ++k) {
            if (h_expr_ops[start + k].op == 0 && h_expr_ops[start + k].arg == pin_id) return true;
        }
        return false;
    };

    std::vector<int> h_pin_func_expr_id(n, -1);
    std::vector<uint8_t> h_pin_func_has_missing_const(n, 0);
    const char* debug_expr_node_env = std::getenv("XPLACE_POWER_DEBUG_EXPR_NODE");
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell) continue;
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n || !h_is_driver_pin[pin_id]) continue;
            int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
            bool used_missing_const = false;
            h_pin_func_expr_id[pin_id] = add_expr(port->function_expr_, cell, node, &used_missing_const);
            if (h_pin_func_expr_id[pin_id] >= 0 && used_missing_const) {
                h_pin_func_has_missing_const[pin_id] = 1;
            }
            if (debug_expr_node_env && node.getName().find(debug_expr_node_env) != std::string::npos) {
                std::fprintf(stderr,
                             "[XPLACE_POWER_DEBUG_EXPR] node=%s pin=%s port=%s expr_id=%d missing_const=%d function='%s'\n",
                             node.getName().c_str(),
                             gtdb.pin_names[pin_id].c_str(),
                             port->name.c_str(),
                             h_pin_func_expr_id[pin_id],
                             h_pin_func_has_missing_const[pin_id] ? 1 : 0,
                             port->function_expr_.c_str());
            }
        }
    }

    const bool eval_missing_const_outputs =
        readPowerBoolEnv("XPLACE_POWER_EVAL_MISSING_CONST_OUTPUTS", true);
    std::vector<std::vector<int>> h_missing_func_outputs_by_pin(n);
    if (eval_missing_const_outputs) {
        for (const auto& node : gtdb.gpdb.getNodes()) {
            LibertyCell* cell = get_cell(static_cast<int>(node.getId()));
            if (!cell || !cell->sequentials_.empty()) continue;
            std::vector<int> load_pins;
            std::vector<int> missing_func_out_pins;
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                if (h_is_load_pin[pin_id]) load_pins.push_back(pin_id);
                if (h_is_driver_pin[pin_id] && h_pin_func_has_missing_const[pin_id])
                    missing_func_out_pins.push_back(pin_id);
            }
            if (load_pins.empty() || missing_func_out_pins.empty()) continue;
            for (int load_pin : load_pins) {
                auto& outputs = h_missing_func_outputs_by_pin[load_pin];
                outputs.insert(outputs.end(), missing_func_out_pins.begin(), missing_func_out_pins.end());
            }
        }
    }
    std::vector<int> h_missing_func_out_start(n + 1, 0);
    std::vector<int> h_missing_func_out_list;
    for (int pin_id = 0; pin_id < n; ++pin_id) {
        h_missing_func_out_start[pin_id] = static_cast<int>(h_missing_func_out_list.size());
        auto& outputs = h_missing_func_outputs_by_pin[pin_id];
        std::sort(outputs.begin(), outputs.end());
        outputs.erase(std::unique(outputs.begin(), outputs.end()), outputs.end());
        h_missing_func_out_list.insert(h_missing_func_out_list.end(), outputs.begin(), outputs.end());
    }
    h_missing_func_out_start[n] = static_cast<int>(h_missing_func_out_list.size());

    std::vector<GpuPowerSeqHost> h_seqs;
    std::vector<std::vector<int>> node_seq_ids(gtdb.gpdb.getNodes().size());
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) continue;
        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            GpuPowerSeqHost rec;
            rec.node_id = node_id;
            rec.data_expr_id = add_expr(seq->next_state_expr_, cell, node, nullptr,
                                        ignore_scan_enable_density);
            rec.clk_expr_id = add_expr(seqClockExpr(seq), cell, node);
            rec.is_latch = seq->is_latch_ ? 1 : 0;
            if (rec.data_expr_id < 0) continue;
            const std::string seq_out = normalize_expr(seq->output_name_);
            const std::string seq_out_inv = normalize_expr(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
                const std::string func = normalize_expr(port->function_expr_);
                if (!seq_out.empty() && func == seq_out) rec.q_pin = pin_id;
                else if (!seq_out_inv.empty() && func == seq_out_inv) rec.qn_pin = pin_id;
            }
            if (rec.q_pin < 0 && rec.qn_pin < 0) continue;
            int seq_id = static_cast<int>(h_seqs.size());
            h_seqs.push_back(rec);
            if (rec.q_pin >= 0) h_is_seq_output_pin[rec.q_pin] = 1;
            if (rec.qn_pin >= 0) h_is_seq_output_pin[rec.qn_pin] = 1;
            node_seq_ids[node_id].push_back(seq_id);
        }
    }
    if (const char* seq_map_file = std::getenv("XPLACE_POWER_SEQ_ID_MAP_FILE")) {
        if (seq_map_file[0] != '\0') {
            std::ofstream out(seq_map_file);
            if (out) {
                out << "seq_id,node_id,inst_name,q_pin,q_pin_name,qn_pin,qn_pin_name,is_latch\n";
                for (int seq_id = 0; seq_id < static_cast<int>(h_seqs.size()); ++seq_id) {
                    const auto& seq = h_seqs[seq_id];
                    std::string inst_name;
                    if (seq.node_id >= 0 && seq.node_id < static_cast<int>(gtdb.gpdb.getNodes().size()))
                        inst_name = gtdb.gpdb.getNodes()[seq.node_id].getName();
                    auto pin_name = [&](int pin_id) -> std::string {
                        return (pin_id >= 0 && pin_id < n) ? gtdb.pin_names[pin_id] : "";
                    };
                    out << seq_id << ',' << seq.node_id << ','
                        << csvEscapePowerActivitySnapshot(inst_name) << ','
                        << seq.q_pin << ','
                        << csvEscapePowerActivitySnapshot(pin_name(seq.q_pin)) << ','
                        << seq.qn_pin << ','
                        << csvEscapePowerActivitySnapshot(pin_name(seq.qn_pin)) << ','
                        << static_cast<int>(seq.is_latch) << '\n';
                }
            }
        }
    }
    if (std::getenv("XPLACE_POWER_PRINT_SEQ_DUP_STATS")) {
        std::vector<int> seq_output_write_count(n, 0);
        for (const auto& seq : h_seqs) {
            if (seq.q_pin >= 0 && seq.q_pin < n) seq_output_write_count[seq.q_pin]++;
            if (seq.qn_pin >= 0 && seq.qn_pin < n) seq_output_write_count[seq.qn_pin]++;
        }
        int duplicate_pins = 0;
        int duplicate_writes = 0;
        int max_writes = 0;
        for (int count : seq_output_write_count) {
            if (count > 1) {
                duplicate_pins++;
                duplicate_writes += count;
                max_writes = std::max(max_writes, count);
            }
        }
        std::cerr << "[power_seq_dup_stats] seq_records=" << h_seqs.size()
                  << " duplicate_output_pins=" << duplicate_pins
                  << " duplicate_output_writes=" << duplicate_writes
                  << " max_writes_per_pin=" << max_writes
                  << std::endl;
        int printed = 0;
        for (int pin_id = 0; pin_id < n && printed < 20; ++pin_id) {
            if (seq_output_write_count[pin_id] <= 1) continue;
            std::cerr << "[power_seq_dup_pin] pin_id=" << pin_id
                      << " writes=" << seq_output_write_count[pin_id]
                      << " pin=" << gtdb.pin_names[pin_id]
                      << std::endl;
            printed++;
        }
    }

    const bool mark_seq_clock_loads =
        readPowerBoolEnv("XPLACE_POWER_MARK_SEQ_CLOCK_LOADS", false);
    std::vector<uint8_t> h_is_seq_clock_input_pin(n, 0);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) continue;
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n || !h_is_load_pin[pin_id]) continue;
            int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (port && port->is_clock_) h_is_seq_clock_input_pin[pin_id] = 1;
        }
    }

    std::vector<std::vector<int>> pin_seq_ids(n);
    for (int pin = 0; pin < n; pin++) {
        int node_id = h_pin_to_node[pin];
        if (node_id >= 0 && node_id < static_cast<int>(node_seq_ids.size()) && !node_seq_ids[node_id].empty()) {
            if (h_is_load_pin[pin] && (mark_seq_clock_loads || !h_is_seq_clock_input_pin[pin]))
                pin_seq_ids[pin] = node_seq_ids[node_id];
        }
    }
    std::vector<int> h_pin_seq_list_start(n + 1, 0);
    std::vector<int> h_pin_seq_list;
    for (int pin = 0; pin < n; pin++) {
        h_pin_seq_list_start[pin] = static_cast<int>(h_pin_seq_list.size());
        h_pin_seq_list.insert(h_pin_seq_list.end(), pin_seq_ids[pin].begin(), pin_seq_ids[pin].end());
    }
    h_pin_seq_list_start[n] = static_cast<int>(h_pin_seq_list.size());

    std::vector<uint8_t> h_is_clock_pin(n, 0);
    for (int pin_id : h_clock_pins) if (pin_id >= 0 && pin_id < n) h_is_clock_pin[pin_id] = 1;

    const bool seed_seq_feedback_outputs =
        readPowerBoolEnv("XPLACE_POWER_SEED_SEQ_FEEDBACK_OUTPUTS", false);
    const bool seed_timing_zero_indeg_roots =
        readPowerBoolEnv("XPLACE_POWER_SEED_TIMING_ZERO_INDEG", true);
    const bool seed_floating_load_roots =
        readPowerBoolEnv("XPLACE_POWER_SEED_FLOATING_LOADS", true);
    const bool seed_seq_feedback_d_only =
        std::getenv("XPLACE_POWER_SEED_SEQ_FEEDBACK_D_ONLY") != nullptr;
    const bool init_seq_feedback_state =
        std::getenv("XPLACE_POWER_INIT_SEQ_FEEDBACK_STATE") != nullptr;
    const bool skip_all_seq_output_arcs =
        std::getenv("XPLACE_POWER_SKIP_ALL_SEQ_OUTPUT_ARCS") != nullptr;
    const bool seed_timing_loop_roots =
        std::getenv("XPLACE_POWER_SEED_TIMING_LOOP_ROOTS") != nullptr;
    const bool skip_disabled_loop_arcs =
        readPowerBoolEnv("XPLACE_POWER_SKIP_DISABLED_LOOP_ARCS", false);
    std::vector<int> h_power_arc_types = gtdb.arc_types;
    auto timing_arc_ptr = [&](int arc_id) -> TimingArc* {
        const int base = arc_id * 2;
        int timing_id = -1;
        if (base + static_cast<int>(MAX) < static_cast<int>(gtdb.timing_arc_id_map.size()))
            timing_id = gtdb.timing_arc_id_map[base + static_cast<int>(MAX)];
        if (timing_id < 0 &&
            base + static_cast<int>(MIN) < static_cast<int>(gtdb.timing_arc_id_map.size()))
            timing_id = gtdb.timing_arc_id_map[base + static_cast<int>(MIN)];
        if (timing_id < 0 || timing_id >= static_cast<int>(gtdb.liberty_timing_arcs.size()))
            return nullptr;
        return gtdb.liberty_timing_arcs[timing_id];
    };
    auto skip_seq_output_arc_for_power = [&](int arc_id, int from_pin, int to_pin) -> bool {
        if (to_pin < 0 || to_pin >= n || !h_is_seq_output_pin[to_pin]) return false;
        TimingArc* timing_arc = timing_arc_ptr(arc_id);
        if (timing_arc) {
            const TimingType type = timing_arc->timing_type_;
            if (type == TimingType::rising_edge || type == TimingType::falling_edge)
                return true;
            if (type == TimingType::clear || type == TimingType::preset)
                return false;
        }
        if (from_pin >= 0 && from_pin < static_cast<int>(gtdb.pin_is_clk.size()) && gtdb.pin_is_clk[from_pin])
            return true;
        return false;
    };
    std::vector<uint8_t> h_power_disabled_loop_arc(gtdb.arc_id2test_id.size(), 0);
    std::vector<int> h_timing_loop_roots;
    int root_timing_loop_count = 0;
    int disabled_loop_arc_count = 0;
    if (seed_timing_loop_roots || skip_disabled_loop_arcs) {
        auto timing_level_edge_valid = [&](int arc_id) -> bool {
            if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.arc_id2test_id.size())) return false;
            if (gtdb.arc_id2test_id[arc_id] != -1) return false;
            TimingArc* timing_arc = timing_arc_ptr(arc_id);
            if (timing_arc) {
                const TimingType type = timing_arc->timing_type_;
                if (type == TimingType::clear || type == TimingType::preset) return false;
            }
            return arc_id < static_cast<int>(gtdb.timing_arc_to_pin_id.size());
        };
        auto edge_to_pin = [&](int arc_id) -> int {
            return (arc_id >= 0 && arc_id < static_cast<int>(gtdb.timing_arc_to_pin_id.size()))
                ? gtdb.timing_arc_to_pin_id[arc_id] : -1;
        };
        auto has_valid_in = [&](int pin_id) -> bool {
            if (pin_id < 0 || pin_id + 1 >= static_cast<int>(gtdb.pin_backward_arc_list_end.size()))
                return false;
            for (int idx = gtdb.pin_backward_arc_list_end[pin_id];
                 idx < gtdb.pin_backward_arc_list_end[pin_id + 1]; ++idx) {
                const int arc_id = gtdb.pin_backward_arc_list[idx];
                if (timing_level_edge_valid(arc_id) && !h_power_disabled_loop_arc[arc_id])
                    return true;
            }
            return false;
        };
        auto has_valid_out = [&](int pin_id) -> bool {
            if (pin_id < 0 || pin_id + 1 >= static_cast<int>(gtdb.pin_forward_arc_list_end.size()))
                return false;
            for (int idx = gtdb.pin_forward_arc_list_end[pin_id];
                 idx < gtdb.pin_forward_arc_list_end[pin_id + 1]; ++idx) {
                const int arc_id = gtdb.pin_forward_arc_list[idx];
                if (timing_level_edge_valid(arc_id) && !h_power_disabled_loop_arc[arc_id])
                    return true;
            }
            return false;
        };
        std::vector<uint8_t> visited(n, 0);
        std::vector<uint8_t> on_path(n, 0);
        auto dfs_from = [&](int root_pin) {
            struct Frame {
                int pin = -1;
                int next = 0;
                int end = 0;
            };
            std::vector<Frame> stack;
            if (root_pin < 0 || root_pin >= n || visited[root_pin]) return;
            visited[root_pin] = 1;
            on_path[root_pin] = 1;
            stack.push_back({root_pin, gtdb.pin_forward_arc_list_end[root_pin],
                             gtdb.pin_forward_arc_list_end[root_pin + 1]});
            while (!stack.empty()) {
                Frame& frame = stack.back();
                bool advanced = false;
                while (frame.next < frame.end) {
                    const int arc_id = gtdb.pin_forward_arc_list[frame.next++];
                    if (!timing_level_edge_valid(arc_id) || h_power_disabled_loop_arc[arc_id])
                        continue;
                    const int to_pin = edge_to_pin(arc_id);
                    if (to_pin < 0 || to_pin >= n) continue;
                    if (!visited[to_pin]) {
                        visited[to_pin] = 1;
                        on_path[to_pin] = 1;
                        stack.push_back({to_pin, gtdb.pin_forward_arc_list_end[to_pin],
                                         gtdb.pin_forward_arc_list_end[to_pin + 1]});
                        advanced = true;
                        break;
                    }
                    if (on_path[to_pin]) {
                        h_power_disabled_loop_arc[arc_id] = 1;
                        h_timing_loop_roots.push_back(to_pin);
                        disabled_loop_arc_count++;
                    }
                }
                if (!advanced) {
                    on_path[frame.pin] = 0;
                    stack.pop_back();
                }
            }
        };
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (!has_valid_in(pin_id) && has_valid_out(pin_id)) {
                h_timing_loop_roots.push_back(pin_id);
                dfs_from(pin_id);
            }
        }
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (!visited[pin_id] && has_valid_out(pin_id)) dfs_from(pin_id);
        }
        std::sort(h_timing_loop_roots.begin(), h_timing_loop_roots.end());
        h_timing_loop_roots.erase(std::unique(h_timing_loop_roots.begin(), h_timing_loop_roots.end()),
                                  h_timing_loop_roots.end());
    }
    for (int from_pin = 0; from_pin < n; ++from_pin) {
        if (from_pin + 1 >= static_cast<int>(gtdb.pin_forward_arc_list_end.size())) break;
        const int start = gtdb.pin_forward_arc_list_end[from_pin];
        const int end = gtdb.pin_forward_arc_list_end[from_pin + 1];
        for (int idx = start; idx < end; ++idx) {
            const int arc_id = gtdb.pin_forward_arc_list[idx];
            if (arc_id < 0 || arc_id >= static_cast<int>(h_power_arc_types.size())) continue;
            if (h_power_arc_types[arc_id] != 1) continue;
            if (arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) continue;
            const int to_pin = gtdb.timing_arc_to_pin_id[arc_id];
            if (to_pin < 0 || to_pin >= n || !h_is_seq_output_pin[to_pin]) continue;
            if (!skip_all_seq_output_arcs && !skip_seq_output_arc_for_power(arc_id, from_pin, to_pin))
                h_power_arc_types[arc_id] = 0;
        }
    }

    const bool has_ideal_clock_pins =
        std::find(gtdb.pin_is_ideal_clk.begin(), gtdb.pin_is_ideal_clk.end(), 1) !=
        gtdb.pin_is_ideal_clk.end();
    auto is_ideal_clock_pin = [&](int pin_id) {
        return pin_id >= 0 &&
               pin_id < static_cast<int>(gtdb.pin_is_ideal_clk.size()) &&
               gtdb.pin_is_ideal_clk[pin_id];
    };

    std::vector<float> h_power_clock_slews;
    if (has_ideal_clock_pins && need_internal_power) {
        h_power_clock_slews.assign(n * NUM_ATTR, nanf(""));
        std::array<float, NUM_ATTR> fallback_clock_slews;
        fallback_clock_slews.fill(nanf(""));
        if (!gtdb.clock_transitions.empty()) {
            fallback_clock_slews = gtdb.clock_transitions.begin()->second;
        }
        for (float& slew : fallback_clock_slews) {
            if (!std::isfinite(slew)) slew = 0.0f;
        }
        auto set_power_clock_slew_pin = [&](int pin_id) {
            if (pin_id < 0 || pin_id >= n) return;
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                float slew = nanf("");
                const int idx = pin_id * NUM_ATTR + attr;
                if (idx >= 0 && idx < static_cast<int>(gtdb.pin_clock_slews.size()))
                    slew = gtdb.pin_clock_slews[idx];
                if (!std::isfinite(slew)) slew = fallback_clock_slews[attr];
                h_power_clock_slews[idx] = slew;
            }
        };
        std::vector<uint8_t> h_power_clock_slew_pin(n, 0);
        auto mark_power_clock_slew_pin = [&](int pin_id) {
            if (pin_id >= 0 && pin_id < n) h_power_clock_slew_pin[pin_id] = 1;
        };
        for (int pin_id : h_clock_pins) {
            if (is_ideal_clock_pin(pin_id)) mark_power_clock_slew_pin(pin_id);
        }
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (h_is_seq_clock_input_pin[pin_id] && is_ideal_clock_pin(pin_id))
                mark_power_clock_slew_pin(pin_id);
        }

        const int num_nets = static_cast<int>(gtdb.gpdb.getNets().size());
        std::vector<uint8_t> power_clock_slew_net(num_nets, 0);
        auto mark_power_clock_slew_net = [&](int net_id) {
            if (net_id >= 0 && net_id < num_nets) power_clock_slew_net[net_id] = 1;
        };
        if (gtdb.pin_is_ideal_clk.size() == static_cast<size_t>(n)) {
            for (int pin_id = 0; pin_id < n; ++pin_id) {
                if (gtdb.pin_is_ideal_clk[pin_id]) mark_power_clock_slew_net(h_pin_to_net[pin_id]);
            }
        }
        for (int net_id = 0; net_id < num_nets; ++net_id) {
            if (!power_clock_slew_net[net_id]) continue;
            for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
                mark_power_clock_slew_pin(pin_id);
            }
        }

        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (h_power_clock_slew_pin[pin_id]) set_power_clock_slew_pin(pin_id);
        }
    }

    std::vector<uint8_t> h_is_primary_input(n, 0);
    std::vector<int> h_primary_inputs;
    h_primary_inputs.reserve(gtdb.primary_inputs.size());
    std::vector<std::string> h_seed_reason(n);
    auto add_seed_reason = [&](int pin_id, const char* reason) {
        if (pin_id < 0 || pin_id >= n || !reason || reason[0] == '\0') return;
        std::string& current = h_seed_reason[pin_id];
        const std::string value(reason);
        if (current.empty()) {
            current = value;
        } else if (current.find(value) == std::string::npos) {
            current += ";";
            current += value;
        }
    };
    auto add_seed_pin = [&](int pin_id, const char* reason) {
        h_primary_inputs.push_back(pin_id);
        add_seed_reason(pin_id, reason);
    };
    int root_primary_count = 0;
    int root_zero_indeg_count = 0;
    int root_const_output_count = 0;
    int root_seq_feedback_count = 0;
    int state_seq_feedback_count = 0;
    int root_power_level_count = 0;
    int root_floating_load_count = 0;
    std::vector<int> h_feedback_seed_pins;
    std::vector<int> h_feedback_seed_seqs;
    bool seed_default_inputs = true;
    if (const char* env = std::getenv("XPLACE_POWER_SEED_INPUTS")) {
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        seed_default_inputs = !(value.empty() || value == "0" || value == "false" || value == "no");
    }
    for (auto pin : gtdb.primary_inputs) {
        const int pin_id = static_cast<int>(pin);
        if (pin_id >= 0 && pin_id < n) h_is_primary_input[pin_id] = 1;
        if (seed_default_inputs && pin_id >= 0 && pin_id < n && h_is_driver_pin[pin_id]
            && !h_is_clock_pin[pin_id]) {
            add_seed_pin(pin_id, "primary_input");
            root_primary_count++;
        }
    }
    if (seed_default_inputs) {
        if (seed_timing_zero_indeg_roots) {
            for (int pin_id : gtdb.pin_frontiers) {
                if (pin_id < 0 || pin_id >= n) continue;
                if (h_is_primary_input[pin_id] || h_is_clock_pin[pin_id]) continue;
                add_seed_pin(pin_id, "timing_zero_indeg");
                root_zero_indeg_count++;
            }
        }
        if (seed_floating_load_roots) {
            for (int pin_id = 0; pin_id < n; pin_id++) {
                if (!h_is_load_pin[pin_id] || h_is_primary_input[pin_id] || h_is_clock_pin[pin_id]) continue;
                const int net_id = h_pin_to_net[pin_id];
                const int driver =
                    (net_id >= 0 && net_id < static_cast<int>(h_net_driver_pin.size())) ? h_net_driver_pin[net_id] : -1;
                if (driver >= 0) continue;
                add_seed_pin(pin_id, "floating_load_input");
                root_floating_load_count++;
            }
        }
        if (seed_timing_loop_roots) {
            for (int pin_id : h_timing_loop_roots) {
                if (pin_id < 0 || pin_id >= n) continue;
                if (h_is_primary_input[pin_id] || h_is_clock_pin[pin_id]) continue;
                if (!h_is_load_pin[pin_id] && !h_is_driver_pin[pin_id]) continue;
                add_seed_pin(pin_id, "timing_loop_root");
                root_timing_loop_count++;
            }
        }
    }
    if (seed_default_inputs && (seed_seq_feedback_outputs || init_seq_feedback_state)) {
        std::vector<uint8_t> seed_seen(n, 0);
        std::vector<uint8_t> state_pin_seen(n, 0);
        std::vector<uint8_t> state_seq_seen(h_seqs.size(), 0);
        auto collect_feedback_data_pins = [&](int expr_id, int driver_pin, std::vector<int>* data_pins) -> bool {
            if (expr_id < 0 || driver_pin < 0 || driver_pin >= n) return false;
            const int driver_net = h_pin_to_net[driver_pin];
            if (driver_net < 0 || driver_net >= static_cast<int>(h_net_driver_pin.size())) return false;
            if (h_net_driver_pin[driver_net] != driver_pin) return false;
            bool matched = false;
            const int start = h_expr_start[expr_id];
            const int end = start + h_expr_count[expr_id];
            for (int op_i = start; op_i < end; ++op_i) {
                if (h_expr_ops[op_i].op != 0) continue;
                const int data_pin = h_expr_ops[op_i].arg;
                if (data_pin < 0 || data_pin >= n || h_pin_to_net[data_pin] != driver_net) continue;
                if (seed_seq_feedback_d_only) {
                    const int node_id = h_pin_to_node[data_pin];
                    LibertyCell* cell = get_cell(node_id);
                    const int port_offset = gtdb.pin_id2port_offset_id[data_pin];
                    if (!cell || port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size()))
                        continue;
                    LibertyPort* port = cell->ports_[port_offset];
                    if (!port || port->name != "D") continue;
                }
                if (data_pin >= 0 && data_pin < n) {
                    matched = true;
                    if (data_pins) data_pins->push_back(data_pin);
                }
            }
            return matched;
        };
        for (int seq_id = 0; seq_id < static_cast<int>(h_seqs.size()); ++seq_id) {
            const auto& seq = h_seqs[seq_id];
            std::vector<int> data_pins;
            const bool q_feedback = collect_feedback_data_pins(seq.data_expr_id, seq.q_pin, &data_pins);
            if (q_feedback && seed_seq_feedback_outputs && !seed_seen[seq.q_pin]) {
                add_seed_pin(seq.q_pin, "seq_feedback_q");
                seed_seen[seq.q_pin] = 1;
                root_seq_feedback_count++;
            }
            const bool qn_feedback = collect_feedback_data_pins(seq.data_expr_id, seq.qn_pin, &data_pins);
            if (qn_feedback && seed_seq_feedback_outputs && !seed_seen[seq.qn_pin]) {
                add_seed_pin(seq.qn_pin, "seq_feedback_qn");
                seed_seen[seq.qn_pin] = 1;
                root_seq_feedback_count++;
            }
            if (init_seq_feedback_state && (q_feedback || qn_feedback)) {
                if (!state_seq_seen[seq_id]) {
                    h_feedback_seed_seqs.push_back(seq_id);
                    state_seq_seen[seq_id] = 1;
                    state_seq_feedback_count++;
                }
                for (int data_pin : data_pins) {
                    if (data_pin >= 0 && data_pin < n && !state_pin_seen[data_pin]) {
                        h_feedback_seed_pins.push_back(data_pin);
                        state_pin_seen[data_pin] = 1;
                    }
                }
            }
        }
    }
    // Constant-generator outputs are roots in OpenSTA's power graph.
    for (int pin_id = 0; pin_id < n; pin_id++) {
        if (!h_is_driver_pin[pin_id] || h_is_primary_input[pin_id] || h_is_clock_pin[pin_id]) continue;
        int node_id = h_pin_to_node[pin_id];
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
        bool has_input_pin = false;
        for (int node_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
            if (node_pin >= 0 && node_pin < n && h_is_load_pin[node_pin]) {
                has_input_pin = true;
                break;
            }
        }
        if (seed_default_inputs && !has_input_pin) {
            add_seed_pin(pin_id, "const_output");
            root_const_output_count++;
        }
    }
    {
        std::vector<uint8_t> root_seen(n, 0);
        std::vector<int> ordered_roots;
        ordered_roots.reserve(h_primary_inputs.size());
        for (int pin_id : h_primary_inputs) {
            if (pin_id < 0 || pin_id >= n || root_seen[pin_id]) continue;
            root_seen[pin_id] = 1;
            ordered_roots.push_back(pin_id);
        }
        h_primary_inputs.swap(ordered_roots);
    }
    auto positive_unate_for_power = [](LibertyCell* cell, LibertyPort* from, LibertyPort* to) -> bool {
        if (!cell || !from || !to) return true;
        for (TimingArc* arc : from->timing_arcs_) {
            if (!arc || arc->to_port_ != to) continue;
            return arc->timing_sense_ == TimingSense::positive_unate ||
                   arc->timing_sense_ == TimingSense::non_unate ||
                   arc->timing_sense_ == TimingSense::unknown;
        }
        for (TimingArc* arc : to->timing_arcs_) {
            if (!arc || arc->from_port_ != from) continue;
            return arc->timing_sense_ == TimingSense::positive_unate ||
                   arc->timing_sense_ == TimingSense::non_unate ||
                   arc->timing_sense_ == TimingSense::unknown;
        }
        return true;
    };

    auto compile_when_expr = [&](InternalPower* ip, LibertyCell* cell, const gp::GPNode& node) -> int {
        if (!ip || ip->when_expr_.empty()) return -1;
        return add_template_expr(ip->when_expr_, cell);
    };

    std::vector<GpuPowerInternalHost> h_internal_rows;
    const char* debug_power_node_env = std::getenv("XPLACE_POWER_DEBUG_NODE");
    std::unordered_map<std::string, int> internal_denom_group;
    auto get_denom_group = [&](int to_pin, const std::string& related_pg) -> int {
        std::string key = std::to_string(to_pin) + "|" + related_pg;
        auto it = internal_denom_group.find(key);
        if (it != internal_denom_group.end()) return it->second;
        int id = static_cast<int>(internal_denom_group.size());
        internal_denom_group.emplace(std::move(key), id);
        return id;
    };

    if (need_internal_power) {
        for (const auto& node : gtdb.gpdb.getNodes()) {
            const int node_id = static_cast<int>(node.getId());
            LibertyCell* cell = get_cell(node_id);
            if (!cell || node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) continue;
            const int libcell_id = gtdb.cell_node_type_map[node_id];
            if (libcell_id < 0 || libcell_id + 1 >= static_cast<int>(gtdb.liberty_cell_type2port_list_end.size())) continue;
            const int port_base = gtdb.liberty_cell_type2port_list_end[libcell_id];
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port) continue;
                const int port_global = port_base + port_offset;
                const int range_idx = port_global * 2 + static_cast<int>(MAX);
                if (range_idx + 1 >= static_cast<int>(gtdb.liberty_port2internal_power_list_end.size())) continue;
                const int ip_start = gtdb.liberty_port2internal_power_list_end[range_idx];
                const int ip_end = gtdb.liberty_port2internal_power_list_end[range_idx + 1];
                if (ip_start == ip_end) continue;

                if (h_is_load_pin[pin_id]) {
                    for (int ip_id = ip_start; ip_id < ip_end; ++ip_id) {
                        InternalPower* ip = gtdb.liberty_internal_powers[ip_id];
                        if (!ip) continue;
                        GpuPowerInternalHost row;
                        row.internal_power_id = ip_id;
                        row.node_id = node_id;
                        row.to_pin = pin_id;
                        row.kind = 0;
                        row.energy_unit = ip->energy_unit_;
                        row.duty_mode = 0;
                        int when_expr_id = compile_when_expr(ip, cell, node);
                        if (when_expr_id >= 0) {
                            row.duty_mode = 1;
                            row.duty_expr_id = when_expr_id;
                            for (int op_i = h_expr_start[when_expr_id]; op_i < h_expr_start[when_expr_id] + h_expr_count[when_expr_id]; ++op_i) {
                                int out_pin = h_expr_ops[op_i].op == 0 ? h_expr_ops[op_i].arg : -1;
                                if (out_pin < -1) {
                                    const int port_id = -2 - out_pin;
                                    if (port_id >= 0 && port_id < static_cast<int>(cell->ports_.size())) {
                                        auto pin_itr = node.portMap.find(cell->ports_[port_id]->name);
                                        out_pin = pin_itr == node.portMap.end() ? -1 : pin_itr->second;
                                    }
                                }
                                if (out_pin >= 0 && out_pin < n && h_is_driver_pin[out_pin]) {
                                    const int func_expr_id = h_pin_func_expr_id[out_pin];
                                    if (expr_contains_pin(func_expr_id, pin_id)) {
                                        row.duty_mode = 2;
                                        row.duty_expr_id = func_expr_id;
                                        row.duty_pin = pin_id;
                                        break;
                                    }
                                }
                            }
                        }
                        if (debug_power_node_env && node.getName().find(debug_power_node_env) != std::string::npos) {
                            std::fprintf(stderr,
                                         "[XPLACE_POWER_DEBUG_NODE] node=%s port=%s kind=input ip=%d when='%s' duty_mode=%d duty_expr=%d\n",
                                         node.getName().c_str(),
                                         port->name.c_str(),
                                         ip_id,
                                         ip->when_expr_.c_str(),
                                         row.duty_mode,
                                         row.duty_expr_id);
                        }
                        h_internal_rows.push_back(row);
                    }
                }

                if (h_is_driver_pin[pin_id]) {
                    const int func_expr_id = h_pin_func_expr_id[pin_id];
                    for (int ip_id = ip_start; ip_id < ip_end; ++ip_id) {
                        InternalPower* ip = gtdb.liberty_internal_powers[ip_id];
                        if (!ip) continue;
                        GpuPowerInternalHost row;
                        row.internal_power_id = ip_id;
                        row.node_id = node_id;
                        row.to_pin = pin_id;
                        row.kind = 1;
                        row.energy_unit = ip->energy_unit_;
                        row.duty_mode = 4;
                        LibertyPort* from_port = ip->related_port_;
                        if (from_port && node.portMap.find(from_port->name) != node.portMap.end()) {
                            row.from_pin = node.portMap.at(from_port->name);
                            row.positive_unate = positive_unate_for_power(cell, from_port, port) ? 1 : 0;
                            const int when_expr_id = compile_when_expr(ip, cell, node);
                            if (expr_contains_pin(func_expr_id, row.from_pin)) {
                                row.duty_mode = 2;
                                row.duty_expr_id = func_expr_id;
                                row.duty_pin = row.from_pin;
                            } else if (when_expr_id >= 0) {
                                row.duty_mode = 1;
                                row.duty_expr_id = when_expr_id;
                            } else {
                                row.duty_mode = 3;
                            }
                            const std::string pg = ip->related_pg_pin_ ? ip->related_pg_pin_->name : ip->related_pg_pin_name_;
                            row.denom_group = get_denom_group(pin_id, pg);
                        }
                        if (debug_power_node_env && node.getName().find(debug_power_node_env) != std::string::npos) {
                            std::fprintf(stderr,
                                         "[XPLACE_POWER_DEBUG_NODE] node=%s port=%s kind=output ip=%d related=%s when='%s' duty_mode=%d duty_expr=%d from_pin=%d\n",
                                         node.getName().c_str(),
                                         port->name.c_str(),
                                         ip_id,
                                         ip->related_port_name_.c_str(),
                                         ip->when_expr_.c_str(),
                                         row.duty_mode,
                                         row.duty_expr_id,
                                         row.from_pin);
                        }
                        h_internal_rows.push_back(row);
                    }
                }
            }
        }
    }

    const float max_power_unit = (gtdb.cell_libs_[MAX] && gtdb.cell_libs_[MAX]->power_unit_.has_value())
        ? static_cast<float>(gtdb.cell_libs_[MAX]->power_unit_->value()) : 1.0f;
    std::vector<GpuPowerLeakageRowHost> h_leakage_rows;
    std::vector<GpuPowerLeakageGroupHost> h_leakage_groups;
    std::unordered_map<std::string, int> leakage_group_map;
    auto get_leakage_group = [&](int node_id, const std::string& pg, float cell_leakage_w) -> int {
        std::string key = std::to_string(node_id) + "|" + pg;
        auto it = leakage_group_map.find(key);
        if (it != leakage_group_map.end()) return it->second;
        GpuPowerLeakageGroupHost group;
        group.node_id = node_id;
        group.cell_leakage = cell_leakage_w;
        int id = static_cast<int>(h_leakage_groups.size());
        h_leakage_groups.push_back(group);
        leakage_group_map.emplace(std::move(key), id);
        return id;
    };
    if (need_leakage_power) {
        for (const auto& node : gtdb.gpdb.getNodes()) {
            const int node_id = static_cast<int>(node.getId());
            LibertyCell* cell = get_cell(node_id);
            if (!cell || node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) continue;
            const int libcell_id = gtdb.cell_node_type_map[node_id];
            if (libcell_id < 0 || libcell_id * 2 + static_cast<int>(MAX) + 1 >= static_cast<int>(gtdb.liberty_cell_type2leakage_power_list_end.size())) continue;
            const int leak_range_idx = libcell_id * 2 + static_cast<int>(MAX);
            const int leak_start = gtdb.liberty_cell_type2leakage_power_list_end[leak_range_idx];
            const int leak_end = gtdb.liberty_cell_type2leakage_power_list_end[leak_range_idx + 1];
            // OpenSTA uses scene_cell(max) for leakage_power groups, but the
            // default/cell_leakage fallback comes from the original cell pointer.
            // In this Xplace setup that corresponds to the MIN/early Liberty view.
            LibertyCell* cell_leakage_cell = (gtdb.cell_libs_[MIN] ? gtdb.cell_libs_[MIN]->get_cell(cell->name) : nullptr);
            if (!cell_leakage_cell) cell_leakage_cell = cell;
            LibertyCell* leak_expr_cell = (gtdb.cell_libs_[MAX] ? gtdb.cell_libs_[MAX]->get_cell(cell->name) : nullptr);
            if (!leak_expr_cell) leak_expr_cell = cell;
            const float cell_leakage_w = cell_leakage_cell->leakage_power_.value_or(0.0f) * max_power_unit;
            if (leak_start == leak_end) {
                get_leakage_group(node_id, "", cell_leakage_w);
                continue;
            }
            for (int leak_id = leak_start; leak_id < leak_end; ++leak_id) {
                LeakagePower* lp = gtdb.liberty_leakage_powers[leak_id];
                if (!lp) continue;
                const std::string pg = lp->related_pg_pin_ ? lp->related_pg_pin_->name : lp->related_pg_pin_name_;
                const int group_id = get_leakage_group(node_id, pg, cell_leakage_w);
                GpuPowerLeakageRowHost row;
                row.node_id = node_id;
                row.group_id = group_id;
                row.leakage_power_id = leak_id;
                row.when_expr_id = lp->when_expr_.empty() ? -1 : add_template_expr(lp->when_expr_, leak_expr_cell);
                row.leakage = lp->value_ * max_power_unit;
                h_leakage_rows.push_back(row);
            }
        }
    }

    auto iopt_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto i64opt_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (internal_row_meta_cpu) {
        std::vector<int64_t> meta;
        meta.reserve(h_internal_rows.size() * 6);
        for (const auto& row : h_internal_rows) {
            meta.push_back(row.node_id);
            meta.push_back(row.to_pin);
            meta.push_back(row.from_pin);
            meta.push_back(row.kind);
            meta.push_back(row.internal_power_id);
            meta.push_back(row.duty_mode);
        }
        if (h_internal_rows.empty()) *internal_row_meta_cpu = torch::empty({0, 6}, i64opt_cpu);
        else *internal_row_meta_cpu = torch::from_blob(meta.data(), {(long)h_internal_rows.size(), 6}, i64opt_cpu).clone();
    }
    if (leakage_row_meta_cpu) {
        std::vector<int64_t> meta;
        meta.reserve(h_leakage_rows.size() * 4);
        for (const auto& row : h_leakage_rows) {
            meta.push_back(row.node_id);
            meta.push_back(row.group_id);
            meta.push_back(row.leakage_power_id);
            meta.push_back(row.when_expr_id);
        }
        if (h_leakage_rows.empty()) *leakage_row_meta_cpu = torch::empty({0, 4}, i64opt_cpu);
        else *leakage_row_meta_cpu = torch::from_blob(meta.data(), {(long)h_leakage_rows.size(), 4}, i64opt_cpu).clone();
    }
    auto bopt_cpu = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto fopt_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto iopt_cuda = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto to_cuda_int = [&](const std::vector<int>& v) {
        std::vector<int> tmp = v.empty() ? std::vector<int>{0} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, iopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_index = [&](const std::vector<index_type>& v) {
        std::vector<index_type> tmp = v.empty() ? std::vector<index_type>{0} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, iopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_u8 = [&](const std::vector<uint8_t>& v) {
        std::vector<uint8_t> tmp = v.empty() ? std::vector<uint8_t>{0} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, bopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_float = [&](const std::vector<float>& v) {
        auto fopt_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        std::vector<float> tmp = v.empty() ? std::vector<float>{nanf("")} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, fopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_bytes = [&](const auto& v) {
        using VecT = std::decay_t<decltype(v)>;
        using ElemT = typename VecT::value_type;
        std::vector<ElemT> tmp = v.empty() ? std::vector<ElemT>(1) : v;
        auto cpu = torch::from_blob(reinterpret_cast<uint8_t*>(tmp.data()), {(long)(tmp.size() * sizeof(ElemT))}, bopt_cpu).clone();
        return cpu.to(torch::kCUDA);
    };
    auto to_cuda_bytes_range = [&](const auto& v, size_t begin, size_t count) {
        using VecT = std::decay_t<decltype(v)>;
        using ElemT = typename VecT::value_type;
        if (count == 0) {
            std::vector<ElemT> tmp(1);
            auto cpu = torch::from_blob(reinterpret_cast<uint8_t*>(tmp.data()), {(long)sizeof(ElemT)}, bopt_cpu).clone();
            return cpu.to(torch::kCUDA);
        }
        auto* data = const_cast<ElemT*>(v.data() + begin);
        auto cpu = torch::from_blob(reinterpret_cast<uint8_t*>(data), {(long)(count * sizeof(ElemT))}, bopt_cpu).clone();
        return cpu.to(torch::kCUDA);
    };
    const bool upload_debug = readPowerBoolEnv("XPLACE_POWER_UPLOAD_DEBUG", false);
    const bool upload_sync_debug = readPowerBoolEnv("XPLACE_POWER_UPLOAD_SYNC_DEBUG", false);
    auto power_upload_mark = [&](const char* phase, const char* label, size_t count, size_t elem_size) {
        if (upload_debug) {
            std::fprintf(stderr,
                         "[power_upload] %s %s count=%zu bytes=%zu\n",
                         phase,
                         label ? label : "",
                         count,
                         count * elem_size);
        }
        if (upload_sync_debug) {
            const std::string sync_label =
                std::string("upload ") + (phase ? phase : "") + " " + (label ? label : "");
            check_power_cuda_error(sync_label.c_str());
        }
    };
    auto upload_cuda_int = [&](const char* label, const std::vector<int>& v) {
        power_upload_mark("begin", label, v.size(), sizeof(int));
        auto out = to_cuda_int(v);
        power_upload_mark("end", label, v.size(), sizeof(int));
        return out;
    };
    auto upload_cuda_index = [&](const char* label, const std::vector<index_type>& v) {
        power_upload_mark("begin", label, v.size(), sizeof(index_type));
        auto out = to_cuda_index(v);
        power_upload_mark("end", label, v.size(), sizeof(index_type));
        return out;
    };
    auto upload_cuda_u8 = [&](const char* label, const std::vector<uint8_t>& v) {
        power_upload_mark("begin", label, v.size(), sizeof(uint8_t));
        auto out = to_cuda_u8(v);
        power_upload_mark("end", label, v.size(), sizeof(uint8_t));
        return out;
    };
    auto upload_cuda_float = [&](const char* label, const std::vector<float>& v) {
        power_upload_mark("begin", label, v.size(), sizeof(float));
        auto out = to_cuda_float(v);
        power_upload_mark("end", label, v.size(), sizeof(float));
        return out;
    };
    auto upload_cuda_bytes = [&](const char* label, const auto& v) {
        using VecT = std::decay_t<decltype(v)>;
        using ElemT = typename VecT::value_type;
        power_upload_mark("begin", label, v.size(), sizeof(ElemT));
        auto out = to_cuda_bytes(v);
        power_upload_mark("end", label, v.size(), sizeof(ElemT));
        return out;
    };
    auto read_chunk_bytes = [](const char* env_name, size_t default_value) {
        const char* env = std::getenv(env_name);
        if (!env || env[0] == '\0') return default_value;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end == env || parsed == 0) return default_value;
        return static_cast<size_t>(parsed);
    };
    constexpr size_t default_power_row_chunk_bytes = 512ull * 1024ull * 1024ull;
    const size_t internal_row_bytes = h_internal_rows.size() * sizeof(GpuPowerInternalHost);
    const size_t leakage_row_bytes = h_leakage_rows.size() * sizeof(GpuPowerLeakageRowHost);
    const size_t internal_chunk_bytes = read_chunk_bytes("XPLACE_POWER_INTERNAL_ROW_CHUNK_BYTES",
                                                         read_chunk_bytes("XPLACE_POWER_ROW_CHUNK_BYTES",
                                                                          default_power_row_chunk_bytes));
    const size_t leakage_chunk_bytes = read_chunk_bytes("XPLACE_POWER_LEAKAGE_ROW_CHUNK_BYTES",
                                                        read_chunk_bytes("XPLACE_POWER_ROW_CHUNK_BYTES",
                                                                         default_power_row_chunk_bytes));
    const bool chunk_internal_rows =
        need_internal_power && !h_internal_rows.empty() && internal_row_bytes > internal_chunk_bytes;
    const bool chunk_leakage_rows =
        need_leakage_power && !h_leakage_rows.empty() && leakage_row_bytes > leakage_chunk_bytes;
    if (chunk_internal_rows || chunk_leakage_rows || std::getenv("XPLACE_POWER_PRINT_ROW_STATS")) {
        std::fprintf(stderr,
                     "[power_row_stats] internal_rows=%zu internal_bytes=%zu internal_chunk=%zu chunk_internal=%d "
                     "denom_groups=%zu leakage_rows=%zu leakage_bytes=%zu leakage_chunk=%zu chunk_leakage=%d "
                     "leakage_groups=%zu expr_ops=%zu expr_bytes=%zu expr_cache=%zu\n",
                     h_internal_rows.size(), internal_row_bytes, internal_chunk_bytes,
                     chunk_internal_rows ? 1 : 0, internal_denom_group.size(),
                     h_leakage_rows.size(), leakage_row_bytes, leakage_chunk_bytes,
                     chunk_leakage_rows ? 1 : 0, h_leakage_groups.size(),
                     h_expr_ops.size(), h_expr_ops.size() * sizeof(GpuPowerExprOpHost),
                     template_expr_cache.size());
    }
    std::vector<int> h_node_port_pin_start;
    std::vector<int> h_node_port_pin_list;
    if (need_internal_power || need_leakage_power) {
        const int node_count = static_cast<int>(gtdb.gpdb.getNodes().size());
        h_node_port_pin_start.assign(node_count + 1, 0);
        for (const auto& node : gtdb.gpdb.getNodes()) {
            const int node_id = static_cast<int>(node.getId());
            LibertyCell* cell = get_cell(node_id);
            const int port_count = cell ? static_cast<int>(cell->ports_.size()) : 0;
            if (node_id >= 0 && node_id < node_count)
                h_node_port_pin_start[node_id + 1] = port_count;
        }
        for (int node_id = 0; node_id < node_count; ++node_id)
            h_node_port_pin_start[node_id + 1] += h_node_port_pin_start[node_id];
        h_node_port_pin_list.assign(h_node_port_pin_start.back(), -1);
        for (const auto& node : gtdb.gpdb.getNodes()) {
            const int node_id = static_cast<int>(node.getId());
            LibertyCell* cell = get_cell(node_id);
            if (!cell || node_id < 0 || node_id >= node_count) continue;
            const int start = h_node_port_pin_start[node_id];
            const int end = h_node_port_pin_start[node_id + 1];
            for (int port_id = 0; port_id < static_cast<int>(cell->ports_.size()) && start + port_id < end; ++port_id) {
                auto pin_itr = node.portMap.find(cell->ports_[port_id]->name);
                if (pin_itr != node.portMap.end()) h_node_port_pin_list[start + port_id] = pin_itr->second;
            }
        }
    }

    auto d_pin_forward_arc_list_end = upload_cuda_index("pin_forward_arc_list_end", gtdb.pin_forward_arc_list_end);
    auto d_pin_forward_arc_list = upload_cuda_index("pin_forward_arc_list", gtdb.pin_forward_arc_list);
    auto d_timing_arc_to_pin_id = upload_cuda_index("timing_arc_to_pin_id", gtdb.timing_arc_to_pin_id);
    auto d_arc_types = upload_cuda_int("power_arc_types", h_power_arc_types);
    std::vector<int> h_power_arc_id2test_id = gtdb.arc_id2test_id;
    if (seed_timing_loop_roots || skip_disabled_loop_arcs) {
        const int num_mark_arcs = std::min(static_cast<int>(h_power_arc_id2test_id.size()),
                                           static_cast<int>(h_power_disabled_loop_arc.size()));
        for (int arc_id = 0; arc_id < num_mark_arcs; ++arc_id) {
            if (h_power_disabled_loop_arc[arc_id]) h_power_arc_id2test_id[arc_id] = 0;
        }
    }
    int disabled_constraint_arc_count = 0;
    int disabled_constraint_net_arc_count = 0;
    // Debug-only experiment: set_false_path is a timing exception in
    // OpenSTA/OpenROAD, not a default power activity cut.  Leave this off
    // for normal acceptance unless explicitly probing false-path activity.
    const bool apply_power_false_paths =
        readPowerBoolEnv("XPLACE_POWER_APPLY_FALSE_PATHS", false);
    if (apply_power_false_paths && !gtdb.power_disabled_constraint_arc.empty()) {
        const int num_mark_arcs = std::min(static_cast<int>(h_power_arc_id2test_id.size()),
                                           static_cast<int>(gtdb.power_disabled_constraint_arc.size()));
        for (int arc_id = 0; arc_id < num_mark_arcs; ++arc_id) {
            if (!gtdb.power_disabled_constraint_arc[arc_id]) continue;
            h_power_arc_id2test_id[arc_id] = 0;
            ++disabled_constraint_arc_count;
            if (arc_id < static_cast<int>(h_power_arc_types.size()) && h_power_arc_types[arc_id] == 0)
                ++disabled_constraint_net_arc_count;
        }
        if (disabled_constraint_arc_count > 0) {
            std::fprintf(stderr,
                         "[power_false_path] disabled_arcs=%d disabled_net_arcs=%d\n",
                         disabled_constraint_arc_count,
                         disabled_constraint_net_arc_count);
        }
    } else if (!apply_power_false_paths && !gtdb.power_disabled_constraint_arc.empty()) {
        int mapped_false_path_arcs = 0;
        for (uint8_t mark : gtdb.power_disabled_constraint_arc) {
            if (mark) ++mapped_false_path_arcs;
        }
        if (mapped_false_path_arcs > 0) {
            std::fprintf(stderr,
                         "[power_false_path] mapped_arcs=%d apply=0\n",
                         mapped_false_path_arcs);
        }
    }
    const int* activity_flat_net2pin_start_map = flat_net2pin_start_map;
    const int* activity_flat_net2pin_map = flat_net2pin_map;
    if (disabled_constraint_net_arc_count > 0) {
        // Direct net fanout bypasses arc_id2test_id. When SDC exceptions disable
        // net arcs, rely on the timing graph net arcs so the mask is honored.
        activity_flat_net2pin_start_map = nullptr;
        activity_flat_net2pin_map = nullptr;
    }
    auto d_arc_id2test_id = upload_cuda_int("power_arc_id2test_id", h_power_arc_id2test_id);
    auto d_net_driver_pin = upload_cuda_int("net_driver_pin", h_net_driver_pin);
    auto d_is_load_pin = upload_cuda_u8("is_load_pin", h_is_load_pin);
    auto d_is_driver_pin = upload_cuda_u8("is_driver_pin", h_is_driver_pin);
    auto d_is_cell_pin = upload_cuda_u8("is_cell_pin", h_is_cell_pin);
    auto d_is_seq_output_pin = upload_cuda_u8("is_seq_output_pin", h_is_seq_output_pin);
    auto d_clock_gate_out_for_input = upload_cuda_int("clock_gate_out_for_input", h_clock_gate_out_for_input);
    auto d_clock_gate_clock_for_out = upload_cuda_int("clock_gate_clock_for_out", h_clock_gate_clock_for_out);
    auto d_clock_gate_enable_for_out = upload_cuda_int("clock_gate_enable_for_out", h_clock_gate_enable_for_out);
    auto d_clock_pins = upload_cuda_int("clock_pins", h_clock_pins);
    auto d_clock_pin_densities = upload_cuda_float("clock_pin_densities", h_clock_pin_densities);
    auto d_clock_pin_duties = upload_cuda_float("clock_pin_duties", h_clock_pin_duties);
    auto d_clock_pin_enqueue = upload_cuda_u8("clock_pin_enqueue", h_clock_pin_enqueue);
    torch::Tensor d_power_clock_slews;
    const float* d_power_clock_slews_ptr = nullptr;
    if (need_internal_power && !h_power_clock_slews.empty()) {
        d_power_clock_slews = upload_cuda_float("power_clock_slews", h_power_clock_slews);
        d_power_clock_slews_ptr = d_power_clock_slews.data_ptr<float>();
    }
    auto d_expr_ops = upload_cuda_bytes("expr_ops", h_expr_ops);
    auto d_expr_start = upload_cuda_int("expr_start", h_expr_start);
    auto d_expr_count = upload_cuda_int("expr_count", h_expr_count);
    auto d_node_port_pin_start = upload_cuda_int("node_port_pin_start", h_node_port_pin_start);
    auto d_node_port_pin_list = upload_cuda_int("node_port_pin_list", h_node_port_pin_list);
    auto d_pin_func_expr_id = upload_cuda_int("pin_func_expr_id", h_pin_func_expr_id);
    auto d_missing_func_out_start = upload_cuda_int("missing_func_out_start", h_missing_func_out_start);
    auto d_missing_func_out_list = upload_cuda_int("missing_func_out_list", h_missing_func_out_list);
    auto d_seqs = upload_cuda_bytes("seqs", h_seqs);
    auto d_pin_seq_list_start = upload_cuda_int("pin_seq_list_start", h_pin_seq_list_start);
    auto d_pin_seq_list = upload_cuda_int("pin_seq_list", h_pin_seq_list);
    auto d_feedback_seed_pins = upload_cuda_int("feedback_seed_pins", h_feedback_seed_pins);
    auto d_feedback_seed_seqs = upload_cuda_int("feedback_seed_seqs", h_feedback_seed_seqs);
    auto h_trace_pins = resolvePowerTracePins(readPowerTracePinQueries(), gtdb.pin_names);
    auto d_trace_pins = upload_cuda_int("trace_pins", h_trace_pins);
    torch::Tensor d_internal_rows;
    torch::Tensor d_leakage_rows;
    torch::Tensor d_leakage_groups;
    GpuPowerInternalHost* d_internal_rows_ptr = nullptr;
    GpuPowerLeakageRowHost* d_leakage_rows_ptr = nullptr;
    GpuPowerLeakageGroupHost* d_leakage_groups_ptr = nullptr;
    if (need_internal_power && !chunk_internal_rows && !h_internal_rows.empty()) {
        d_internal_rows = upload_cuda_bytes("internal_rows", h_internal_rows);
        d_internal_rows_ptr = reinterpret_cast<GpuPowerInternalHost*>(d_internal_rows.data_ptr<uint8_t>());
    }
    if (need_leakage_power && !chunk_leakage_rows && !h_leakage_rows.empty()) {
        d_leakage_rows = upload_cuda_bytes("leakage_rows", h_leakage_rows);
        d_leakage_rows_ptr = reinterpret_cast<GpuPowerLeakageRowHost*>(d_leakage_rows.data_ptr<uint8_t>());
    }
    if (need_leakage_power && !h_leakage_groups.empty()) {
        d_leakage_groups = upload_cuda_bytes("leakage_groups", h_leakage_groups);
        d_leakage_groups_ptr = reinterpret_cast<GpuPowerLeakageGroupHost*>(d_leakage_groups.data_ptr<uint8_t>());
    }

    // Power-specific CUDA levelization: use the same propagation edge predicate
    // as power_enqueue_adjacent() (skip constraints/tests and sequential Q/Q_N arcs).
    levelize_power(d_is_seq_output_pin.data_ptr<uint8_t>(),
                   d_arc_types.data_ptr<int>(),
                   d_arc_id2test_id.data_ptr<int>(),
                   d_is_load_pin.data_ptr<uint8_t>(),
                   pin2net_map,
                   d_net_driver_pin.data_ptr<int>(),
                   activity_flat_net2pin_start_map,
                   activity_flat_net2pin_map);
    if (!power_level_list || power_level_list_end_cpu.empty()) {
        throw std::runtime_error("levelize_power failed to build power level list");
    }
    if (seed_default_inputs && std::getenv("XPLACE_POWER_SEED_POWER_LEVEL_ROOTS")) {
        for (int pin_id : power_level_root_pins_cpu) {
            if (pin_id < 0 || pin_id >= n) continue;
            if (h_is_primary_input[pin_id] || h_is_clock_pin[pin_id]) continue;
            if (!h_is_load_pin[pin_id] && !h_is_driver_pin[pin_id]) continue;
            add_seed_pin(pin_id, "power_zero_fanin_seed");
            root_power_level_count++;
        }
    }

    std::vector<uint8_t> h_seed_seen(n, 0);
    std::vector<int> h_seed_inputs;
    h_seed_inputs.reserve(h_primary_inputs.size());
    for (int pin_id : h_primary_inputs) {
        if (pin_id < 0 || pin_id >= n || h_seed_seen[pin_id]) continue;
        h_seed_seen[pin_id] = 1;
        h_seed_inputs.push_back(pin_id);
    }
    h_primary_inputs.swap(h_seed_inputs);
    std::vector<int> h_power_fanin(n, 0);
    if (gtdb.pin_forward_arc_list_end.size() == static_cast<size_t>(n + 1)) {
        for (int from_pin = 0; from_pin < n; ++from_pin) {
            const int start = gtdb.pin_forward_arc_list_end[from_pin];
            const int end = gtdb.pin_forward_arc_list_end[from_pin + 1];
            for (int idx = start; idx < end; ++idx) {
                if (idx < 0 || idx >= static_cast<int>(gtdb.pin_forward_arc_list.size())) continue;
                const int arc_id = gtdb.pin_forward_arc_list[idx];
                if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) continue;
                if (arc_id < static_cast<int>(h_power_arc_id2test_id.size()) && h_power_arc_id2test_id[arc_id] != -1) continue;
                const int to_pin = gtdb.timing_arc_to_pin_id[arc_id];
                if (to_pin < 0 || to_pin >= n) continue;
                if (arc_id < static_cast<int>(h_power_arc_types.size())
                    && h_power_arc_types[arc_id] == 1 && h_is_seq_output_pin[to_pin])
                    continue;
                h_power_fanin[to_pin]++;
            }
        }
    }
    if (const char* root_dump_file = std::getenv("XPLACE_POWER_DUMP_ROOTS_FILE")) {
        if (root_dump_file[0] != '\0') {
            std::vector<uint8_t> h_candidate_seen(n, 0);
            for (int pin_id : power_level_root_pins_cpu) {
                if (pin_id >= 0 && pin_id < n) h_candidate_seen[pin_id] = 1;
            }
            std::vector<int> root_probe_pins =
                resolvePowerTracePins(readPowerRootProbePinQueries(), gtdb.pin_names);
            std::vector<int> dump_pins;
            dump_pins.reserve(h_primary_inputs.size() + power_level_root_pins_cpu.size() + root_probe_pins.size());
            dump_pins.insert(dump_pins.end(), h_primary_inputs.begin(), h_primary_inputs.end());
            dump_pins.insert(dump_pins.end(), power_level_root_pins_cpu.begin(), power_level_root_pins_cpu.end());
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
                    std::string reason = h_seed_reason[pin_id];
                    if (reason.empty() && h_seed_seen[pin_id]) reason = "seed";
                    if (h_candidate_seen[pin_id]) {
                        if (reason.empty()) reason = "power_zero_fanin_candidate";
                        else if (reason.find("power_zero_fanin_candidate") == std::string::npos
                                 && reason.find("power_zero_fanin_seed") == std::string::npos)
                            reason += ";power_zero_fanin_candidate";
                    }
                    if (reason.empty()) reason = "probe_only";
                    const int node_id =
                        (pin_id < static_cast<int>(h_pin_to_node.size())) ? h_pin_to_node[pin_id] : -1;
                    const int net_id =
                        (pin_id < static_cast<int>(h_pin_to_net.size())) ? h_pin_to_net[pin_id] : -1;
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
                              << (h_seed_seen[pin_id] ? 1 : 0) << '\t'
                              << (h_candidate_seen[pin_id] ? 1 : 0) << '\t'
                              << reason << '\t'
                              << (h_is_primary_input[pin_id] ? 1 : 0) << '\t'
                              << (h_is_clock_pin[pin_id] ? 1 : 0) << '\t'
                              << (h_is_driver_pin[pin_id] ? 1 : 0) << '\t'
                              << (h_is_load_pin[pin_id] ? 1 : 0) << '\t'
                              << h_power_fanin[pin_id] << '\t'
                              << timing_fanin << '\t' << power_level << '\t'
                              << node_id << '\t' << inst_name << '\t' << cell_type << '\t'
                              << net_id << '\t' << net_name << '\n';
                }
            }
        }
    }
    if (std::getenv("XPLACE_POWER_PRINT_ROOT_STATS")) {
        std::fprintf(stderr,
                     "[power_activity_roots] seeds=%zu primary=%d timing_roots=%d floating_load_roots=%d timing_loop_roots=%d disabled_loop_arcs=%d power_roots=%d const_outputs=%d\n",
                     h_primary_inputs.size(), root_primary_count, root_zero_indeg_count,
                     root_floating_load_count, root_timing_loop_count, disabled_loop_arc_count,
                     root_power_level_count, root_const_output_count);
        if (seed_seq_feedback_outputs) {
            std::fprintf(stderr,
                         "[power_activity_roots] seq_feedback=%d\n",
                         root_seq_feedback_count);
        }
        if (init_seq_feedback_state) {
            std::fprintf(stderr,
                         "[power_activity_roots] seq_feedback_state=%d pins=%zu\n",
                         state_seq_feedback_count, h_feedback_seed_pins.size());
        }
    }
    auto d_primary_inputs = to_cuda_int(h_primary_inputs);

    auto build_cpu_activity_levels = [&]() {
        std::vector<int> pin_level(n, 0);
        int max_pin_level = 0;
        if (gtdb.pin_num_fanin.size() == static_cast<size_t>(n)
            && gtdb.pin_fanout_list_end.size() == static_cast<size_t>(n + 1)) {
            std::vector<int> indeg = gtdb.pin_num_fanin;
            std::deque<int> frontier;
            for (int pin_id = 0; pin_id < n; ++pin_id) {
                if (indeg[pin_id] == 0) frontier.push_back(pin_id);
            }
            std::vector<uint8_t> seen(n, 0);
            while (!frontier.empty()) {
                const int pin_id = frontier.front();
                frontier.pop_front();
                if (pin_id < 0 || pin_id >= n || seen[pin_id]) continue;
                seen[pin_id] = 1;
                max_pin_level = std::max(max_pin_level, pin_level[pin_id]);
                const int start = gtdb.pin_fanout_list_end[pin_id];
                const int end = gtdb.pin_fanout_list_end[pin_id + 1];
                for (int idx = start; idx < end; ++idx) {
                    const int fanout = gtdb.pin_fanout_list[idx];
                    if (fanout < 0 || fanout >= n) continue;
                    pin_level[fanout] = std::max(pin_level[fanout], pin_level[pin_id] + 1);
                    if (--indeg[fanout] == 0) frontier.push_back(fanout);
                }
            }
        }
        std::vector<std::vector<int>> by_level(std::max(1, max_pin_level + 1));
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            const int level = std::clamp(pin_level[pin_id], 0, max_pin_level);
            by_level[level].push_back(pin_id);
        }
        std::vector<int> flat;
        flat.reserve(n);
        std::vector<int> ends;
        ends.reserve(by_level.size() + 1);
        ends.push_back(0);
        for (const auto& pins : by_level) {
            flat.insert(flat.end(), pins.begin(), pins.end());
            ends.push_back(static_cast<int>(flat.size()));
        }
        return std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>{
            std::move(pin_level), std::move(flat), std::move(ends)};
    };

    const bool use_cpu_activity_levels_for_power =
        std::getenv("XPLACE_POWER_USE_CPU_ACTIVITY_LEVELS") != nullptr;
    const bool use_timing_levels_for_power =
        std::getenv("XPLACE_POWER_USE_TIMING_LEVELS") != nullptr;
    const std::vector<int>* activity_level_list_end_cpu = &power_level_list_end_cpu;
    index_type* activity_level_list = power_level_list;
    std::vector<int> h_pin_power_level;
    torch::Tensor d_cpu_activity_level_list;
    std::vector<int> cpu_activity_level_list_end_cpu;
    if (use_cpu_activity_levels_for_power) {
        auto [cpu_activity_pin_level, cpu_activity_level_list, cpu_activity_level_ends] =
            build_cpu_activity_levels();
        d_cpu_activity_level_list = to_cuda_int(cpu_activity_level_list);
        cpu_activity_level_list_end_cpu = std::move(cpu_activity_level_ends);
        activity_level_list_end_cpu = &cpu_activity_level_list_end_cpu;
        activity_level_list = d_cpu_activity_level_list.data_ptr<int>();
        h_pin_power_level = std::move(cpu_activity_pin_level);
    } else if (use_timing_levels_for_power) {
        if (!level_list || level_list_end_cpu.empty()) {
            throw std::runtime_error("timing level list is unavailable for power activity");
        }
        activity_level_list_end_cpu = &level_list_end_cpu;
        activity_level_list = level_list;
        h_pin_power_level = pin_level_cpu;
        if (static_cast<int>(h_pin_power_level.size()) != n) h_pin_power_level.assign(n, -1);
    } else {
        h_pin_power_level = power_pin_level_cpu;
        if (static_cast<int>(h_pin_power_level.size()) != n) h_pin_power_level.assign(n, -1);
    }
    auto d_pin_power_level = to_cuda_int(h_pin_power_level);

    int max_activity_passes = 50;
    if (const char* env = std::getenv("XPLACE_POWER_ACTIVITY_MAX_PASSES"))
        max_activity_passes = std::max(1, std::atoi(env));
    const float min_activity_density =
        std::max(0.0f, readPowerFloatEnv("XPLACE_POWER_MIN_ACTIVITY_DENSITY", 1.0e-10f));

    torch::Tensor out_gpu;
    float* out_gpu_ptr = nullptr;
    if (need_switching_power || want_activity_cpu || chunk_internal_rows || chunk_leakage_rows) {
        out_gpu = torch::empty({n, 3}, fopt_cuda);
        out_gpu_ptr = out_gpu.data_ptr<float>();
    }
    torch::Tensor inst_switching_gpu;
    torch::Tensor pin_switching_gpu;
    torch::Tensor inst_internal_gpu;
    torch::Tensor internal_row_power_gpu;
    torch::Tensor inst_leakage_gpu;
    torch::Tensor leakage_row_power_gpu;
    float* inst_switching_ptr = nullptr;
    float* pin_switching_ptr = nullptr;
    float* inst_internal_ptr = nullptr;
    float* internal_row_power_ptr = nullptr;
    float* inst_leakage_ptr = nullptr;
    float* leakage_row_power_ptr = nullptr;
    float power_voltage = 1.0f;
    auto env_flag = [](const char* name, bool default_value) {
        const char* env = std::getenv(name);
        if (!env || env[0] == '\0') return default_value;
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return !(value == "0" || value == "false" || value == "no" || value == "off");
    };
    const bool use_cpu_activity_for_power =
        env_flag("XPLACE_POWER_USE_CPU_ACTIVITY_FOR_POWER", false);
    torch::Tensor precomputed_activity_cpu;
    torch::Tensor precomputed_activity_gpu;
    const float* precomputed_activity_ptr = nullptr;
    if (use_cpu_activity_for_power) {
        precomputed_activity_cpu = report_power_activity_cpu();
        if (precomputed_activity_cpu.dim() != 2 || precomputed_activity_cpu.size(0) != n ||
            precomputed_activity_cpu.size(1) != 3) {
            throw std::runtime_error("report_power_activity_cpu returned an unexpected activity tensor shape");
        }
        precomputed_activity_gpu = precomputed_activity_cpu.to(torch::kCUDA);
        precomputed_activity_ptr = precomputed_activity_gpu.data_ptr<float>();
    }
    if (const char* env_voltage = std::getenv("XPLACE_POWER_VOLTAGE")) {
        const float v = std::strtof(env_voltage, nullptr);
        if (std::isfinite(v) && v > 0.0f) power_voltage = v;
    } else {
        // OpenSTA switching power uses the max scene/corner.  Prefer the MAX
        // Liberty operating-condition voltage, then fall back to any available lib.
        auto read_lib_voltage = [](const std::shared_ptr<CellLib>& lib, float& out) -> bool {
            if (!lib) return false;
            auto it = lib->default_values.find("voltage");
            if (it != lib->default_values.end() && it->second.has_value() && *(it->second) > 0.0f) {
                out = *(it->second);
                return true;
            }
            return false;
        };
        if (!read_lib_voltage(gtdb.cell_libs_[MAX], power_voltage)) {
            for (const auto& lib : gtdb.cell_libs_) {
                if (read_lib_voltage(lib, power_voltage)) break;
            }
        }
    }
    if (inst_switching_cpu) {
        inst_switching_gpu = torch::zeros({num_nodes}, fopt_cuda);
        inst_switching_ptr = inst_switching_gpu.data_ptr<float>();
    }
    if (pin_switching_cpu) {
        pin_switching_gpu = torch::zeros({n}, fopt_cuda);
        pin_switching_ptr = pin_switching_gpu.data_ptr<float>();
    }
    if (inst_internal_cpu || internal_row_power_cpu) {
        inst_internal_gpu = torch::zeros({num_nodes}, fopt_cuda);
        inst_internal_ptr = inst_internal_gpu.data_ptr<float>();
    }
    if (internal_row_power_cpu) {
        internal_row_power_gpu = torch::zeros({static_cast<long>(h_internal_rows.size())}, fopt_cuda);
        internal_row_power_ptr = internal_row_power_gpu.data_ptr<float>();
    }
    if (inst_leakage_cpu) {
        inst_leakage_gpu = torch::zeros({num_nodes}, fopt_cuda);
        inst_leakage_ptr = inst_leakage_gpu.data_ptr<float>();
    }
    if (leakage_row_power_cpu) {
        leakage_row_power_gpu = torch::zeros({static_cast<long>(h_leakage_rows.size())}, fopt_cuda);
        leakage_row_power_ptr = leakage_row_power_gpu.data_ptr<float>();
    }

    const float* dmp_C1_ptr = nullptr;
    const float* dmp_C2_ptr = nullptr;
    bool use_dmp_power_load = true;
    if (const char* env = std::getenv("XPLACE_POWER_USE_DMP_LOAD")) {
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        use_dmp_power_load = !(value.empty() || value == "0" || value == "false" || value == "no");
    }
    if (use_dmp_power_load && h_dmp_db && h_dmp_db->C1 && h_dmp_db->C2) {
        dmp_C1_ptr = h_dmp_db->C1;
        dmp_C2_ptr = h_dmp_db->C2;
    }

    GpuPowerInternalHost* launcher_internal_rows_ptr =
        chunk_internal_rows ? nullptr : d_internal_rows_ptr;
    const int launcher_internal_row_count =
        chunk_internal_rows ? 0 : static_cast<int>(h_internal_rows.size());
    float* launcher_inst_internal_ptr =
        chunk_internal_rows ? nullptr : inst_internal_ptr;
    float* launcher_internal_row_power_ptr =
        chunk_internal_rows ? nullptr : internal_row_power_ptr;
    GpuPowerLeakageRowHost* launcher_leakage_rows_ptr =
        chunk_leakage_rows ? nullptr : d_leakage_rows_ptr;
    const int launcher_leakage_row_count =
        chunk_leakage_rows ? 0 : static_cast<int>(h_leakage_rows.size());
    GpuPowerLeakageGroupHost* launcher_leakage_groups_ptr =
        chunk_leakage_rows ? nullptr : d_leakage_groups_ptr;
    const int launcher_leakage_group_count =
        chunk_leakage_rows ? 0 : static_cast<int>(h_leakage_groups.size());
    float* launcher_inst_leakage_ptr =
        chunk_leakage_rows ? nullptr : inst_leakage_ptr;
    float* launcher_leakage_row_power_ptr =
        chunk_leakage_rows ? nullptr : leakage_row_power_ptr;

    run_power_activity_cuda_launcher(
        n, *activity_level_list_end_cpu,
        activity_level_list,
        d_pin_power_level.data_ptr<int>(),
        d_pin_forward_arc_list_end.data_ptr<index_type>(),
        d_pin_forward_arc_list.data_ptr<index_type>(),
        d_timing_arc_to_pin_id.data_ptr<index_type>(),
        d_arc_types.data_ptr<int>(),
        d_arc_id2test_id.data_ptr<int>(),
        pin2net_map,
        d_net_driver_pin.data_ptr<int>(),
        activity_flat_net2pin_start_map,
        activity_flat_net2pin_map,
        d_is_load_pin.data_ptr<uint8_t>(),
        d_is_driver_pin.data_ptr<uint8_t>(),
        d_is_cell_pin.data_ptr<uint8_t>(),
        d_is_seq_output_pin.data_ptr<uint8_t>(),
        d_clock_gate_out_for_input.data_ptr<int>(),
        d_clock_gate_clock_for_out.data_ptr<int>(),
        d_clock_gate_enable_for_out.data_ptr<int>(),
        d_primary_inputs.data_ptr<int>(),
        static_cast<int>(h_primary_inputs.size()),
        nullptr,
        d_clock_pins.data_ptr<int>(),
        static_cast<int>(h_clock_pins.size()),
        d_clock_pin_densities.data_ptr<float>(),
        d_clock_pin_duties.data_ptr<float>(),
        d_clock_pin_enqueue.data_ptr<uint8_t>(),
        reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>()),
        d_expr_start.data_ptr<int>(),
        d_expr_count.data_ptr<int>(),
        d_node_port_pin_start.data_ptr<int>(),
        d_node_port_pin_list.data_ptr<int>(),
        d_pin_func_expr_id.data_ptr<int>(),
        d_missing_func_out_start.data_ptr<int>(),
        d_missing_func_out_list.data_ptr<int>(),
        reinterpret_cast<GpuPowerSeqHost*>(d_seqs.data_ptr<uint8_t>()),
        static_cast<int>(h_seqs.size()),
        d_pin_seq_list_start.data_ptr<int>(),
        d_pin_seq_list.data_ptr<int>(),
        d_feedback_seed_pins.data_ptr<int>(),
        static_cast<int>(h_feedback_seed_pins.size()),
        d_feedback_seed_seqs.data_ptr<int>(),
        static_cast<int>(h_feedback_seed_seqs.size()),
        default_density,
        clock_density,
        gtdb.time_unit,
        max_activity_passes,
        d_trace_pins.data_ptr<int>(),
        static_cast<int>(h_trace_pins.size()),
        precomputed_activity_ptr,
        out_gpu_ptr,
        num_nodes,
        pin2node_map,
        pinLoad,
        dmp_C1_ptr,
        dmp_C2_ptr,
        pinSlew,
        d_power_clock_slews_ptr,
        readPowerBoolEnv("XPLACE_POWER_ALLOW_CLOCK_ACTIVITY_OVERRIDE", false),
        min_activity_density,
        launcher_internal_rows_ptr,
        launcher_internal_row_count,
        static_cast<int>(internal_denom_group.size()),
        d_power_allocator,
        cap_unit,
        power_voltage,
        inst_switching_ptr,
        pin_switching_ptr,
        launcher_inst_internal_ptr,
        launcher_internal_row_power_ptr,
        launcher_leakage_rows_ptr,
        launcher_leakage_row_count,
        launcher_leakage_groups_ptr,
        launcher_leakage_group_count,
        launcher_inst_leakage_ptr,
        launcher_leakage_row_power_ptr);

    const float* chunk_activity_ptr = precomputed_activity_ptr ? precomputed_activity_ptr : out_gpu_ptr;
    if ((chunk_internal_rows || chunk_leakage_rows) && !chunk_activity_ptr) {
        throw std::runtime_error("chunked CUDA power requires a precomputed activity tensor");
    }

    auto rows_per_chunk = [](size_t chunk_bytes, size_t elem_size) {
        return std::max<size_t>(1, chunk_bytes / std::max<size_t>(1, elem_size));
    };

    if (chunk_internal_rows) {
        if (inst_internal_ptr) {
            const size_t denom_count = std::max<size_t>(1, internal_denom_group.size());
            torch::Tensor internal_denom_gpu = torch::zeros({static_cast<long>(denom_count)}, fopt_cuda);
            const size_t chunk_rows =
                rows_per_chunk(internal_chunk_bytes, sizeof(GpuPowerInternalHost));
            std::fprintf(stderr,
                         "[power_row_chunk] component=internal phase=denom rows=%zu chunk_rows=%zu chunks=%zu\n",
                         h_internal_rows.size(), chunk_rows,
                         (h_internal_rows.size() + chunk_rows - 1) / chunk_rows);
            for (size_t begin = 0; begin < h_internal_rows.size(); begin += chunk_rows) {
                const size_t count = std::min(chunk_rows, h_internal_rows.size() - begin);
                auto d_rows_chunk = to_cuda_bytes_range(h_internal_rows, begin, count);
                run_power_internal_denom_chunk_cuda_launcher(
                    n,
                    chunk_activity_ptr,
                    reinterpret_cast<GpuPowerInternalHost*>(d_rows_chunk.data_ptr<uint8_t>()),
                    static_cast<int>(count),
                    reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>()),
                    d_expr_start.data_ptr<int>(),
                    d_expr_count.data_ptr<int>(),
                    d_node_port_pin_start.data_ptr<int>(),
                    d_node_port_pin_list.data_ptr<int>(),
                    internal_denom_gpu.data_ptr<float>());
            }
            std::fprintf(stderr,
                         "[power_row_chunk] component=internal phase=contrib rows=%zu chunk_rows=%zu chunks=%zu\n",
                         h_internal_rows.size(), chunk_rows,
                         (h_internal_rows.size() + chunk_rows - 1) / chunk_rows);
            for (size_t begin = 0; begin < h_internal_rows.size(); begin += chunk_rows) {
                const size_t count = std::min(chunk_rows, h_internal_rows.size() - begin);
                auto d_rows_chunk = to_cuda_bytes_range(h_internal_rows, begin, count);
                float* row_power_ptr =
                    internal_row_power_ptr ? internal_row_power_ptr + begin : nullptr;
                run_power_internal_contrib_chunk_cuda_launcher(
                    n,
                    num_nodes,
                    chunk_activity_ptr,
                    reinterpret_cast<GpuPowerInternalHost*>(d_rows_chunk.data_ptr<uint8_t>()),
                    static_cast<int>(count),
                    reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>()),
                    d_expr_start.data_ptr<int>(),
                    d_expr_count.data_ptr<int>(),
                    d_node_port_pin_start.data_ptr<int>(),
                    d_node_port_pin_list.data_ptr<int>(),
                    pinSlew,
                    d_power_clock_slews_ptr,
                    dmp_C1_ptr,
                    dmp_C2_ptr,
                    internal_denom_gpu.data_ptr<float>(),
                    d_power_allocator,
                    cap_unit,
                    inst_internal_ptr,
                    row_power_ptr);
            }
        }
    }

    if (chunk_leakage_rows) {
        if (inst_leakage_ptr && d_leakage_groups_ptr) {
            const size_t group_count = std::max<size_t>(1, h_leakage_groups.size());
            torch::Tensor group_cond_leakage_gpu =
                torch::zeros({static_cast<long>(group_count)}, fopt_cuda);
            torch::Tensor group_cond_duty_sum_gpu =
                torch::zeros({static_cast<long>(group_count)}, fopt_cuda);
            torch::Tensor group_cond_count_gpu =
                torch::zeros({static_cast<long>(group_count)}, iopt_cuda);
            const size_t chunk_rows =
                rows_per_chunk(leakage_chunk_bytes, sizeof(GpuPowerLeakageRowHost));
            std::fprintf(stderr,
                         "[power_row_chunk] component=leakage phase=rows rows=%zu chunk_rows=%zu chunks=%zu\n",
                         h_leakage_rows.size(), chunk_rows,
                         (h_leakage_rows.size() + chunk_rows - 1) / chunk_rows);
            for (size_t begin = 0; begin < h_leakage_rows.size(); begin += chunk_rows) {
                const size_t count = std::min(chunk_rows, h_leakage_rows.size() - begin);
                auto d_rows_chunk = to_cuda_bytes_range(h_leakage_rows, begin, count);
                float* row_power_ptr =
                    leakage_row_power_ptr ? leakage_row_power_ptr + begin : nullptr;
                run_power_leakage_rows_chunk_cuda_launcher(
                    n,
                    chunk_activity_ptr,
                    reinterpret_cast<GpuPowerLeakageRowHost*>(d_rows_chunk.data_ptr<uint8_t>()),
                    static_cast<int>(count),
                    reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>()),
                    d_expr_start.data_ptr<int>(),
                    d_expr_count.data_ptr<int>(),
                    d_node_port_pin_start.data_ptr<int>(),
                    d_node_port_pin_list.data_ptr<int>(),
                    group_cond_leakage_gpu.data_ptr<float>(),
                    group_cond_duty_sum_gpu.data_ptr<float>(),
                    group_cond_count_gpu.data_ptr<int>(),
                    row_power_ptr);
            }
            run_power_leakage_summary_chunk_cuda_launcher(
                d_leakage_groups_ptr,
                static_cast<int>(h_leakage_groups.size()),
                group_cond_leakage_gpu.data_ptr<float>(),
                group_cond_duty_sum_gpu.data_ptr<float>(),
                group_cond_count_gpu.data_ptr<int>(),
                num_nodes,
                inst_leakage_ptr);
        }
    }

    if (inst_switching_cpu) *inst_switching_cpu = inst_switching_gpu.to(torch::kCPU);
    if (pin_switching_cpu) *pin_switching_cpu = pin_switching_gpu.to(torch::kCPU);
    if (inst_internal_cpu) *inst_internal_cpu = inst_internal_gpu.to(torch::kCPU);
    if (internal_row_power_cpu) *internal_row_power_cpu = internal_row_power_gpu.to(torch::kCPU);
    if (inst_leakage_cpu) *inst_leakage_cpu = inst_leakage_gpu.to(torch::kCPU);
    if (leakage_row_power_cpu) *leakage_row_power_cpu = leakage_row_power_gpu.to(torch::kCPU);
    if (want_activity_cpu) {
        return out_gpu.to(torch::kCPU);
    }
    return torch::empty({0, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
}

torch::Tensor GPUTimer::report_power_activity_cuda() {
    return compute_power_activity_cuda(nullptr, nullptr);
}

tuple<torch::Tensor, torch::Tensor> GPUTimer::report_power_switching_cuda() {
    torch::Tensor inst_switching_cpu;
    torch::Tensor pin_switching_cpu;
    compute_power_activity_cuda(&inst_switching_cpu, &pin_switching_cpu);
    return {inst_switching_cpu, pin_switching_cpu};
}

torch::Tensor GPUTimer::report_power_internal_cuda() {
    torch::Tensor inst_internal_cpu;
    compute_power_activity_cuda(nullptr, nullptr, &inst_internal_cpu);
    return inst_internal_cpu;
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor> GPUTimer::report_power_internal_arcs_cuda() {
    torch::Tensor inst_internal_cpu;
    torch::Tensor internal_row_power_cpu;
    torch::Tensor internal_row_meta_cpu;
    compute_power_activity_cuda(nullptr, nullptr, &inst_internal_cpu, &internal_row_power_cpu, &internal_row_meta_cpu);
    return {inst_internal_cpu, internal_row_power_cpu, internal_row_meta_cpu};
}

torch::Tensor GPUTimer::report_power_leakage_cuda() {
    torch::Tensor inst_leakage_cpu;
    compute_power_activity_cuda(nullptr, nullptr, nullptr, nullptr, nullptr, &inst_leakage_cpu);
    return inst_leakage_cpu;
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor> GPUTimer::report_power_leakage_rows_cuda() {
    torch::Tensor inst_leakage_cpu;
    torch::Tensor leakage_row_power_cpu;
    torch::Tensor leakage_row_meta_cpu;
    compute_power_activity_cuda(nullptr, nullptr, nullptr, nullptr, nullptr, &inst_leakage_cpu, &leakage_row_power_cpu, &leakage_row_meta_cpu);
    return {inst_leakage_cpu, leakage_row_power_cpu, leakage_row_meta_cpu};
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> GPUTimer::report_power_total_cuda() {
    torch::Tensor inst_switching_cpu;
    torch::Tensor inst_internal_cpu;
    torch::Tensor inst_leakage_cpu;
    compute_power_activity_cuda(&inst_switching_cpu, nullptr, &inst_internal_cpu,
                                nullptr, nullptr, &inst_leakage_cpu);
    torch::Tensor inst_total_cpu = inst_internal_cpu + inst_switching_cpu + inst_leakage_cpu;
    return {inst_internal_cpu, inst_switching_cpu, inst_leakage_cpu, inst_total_cpu};
}

}  // namespace gt
