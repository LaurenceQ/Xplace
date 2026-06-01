#include "PowerHostCommon.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>

namespace gt {

namespace {
thread_local double g_power_stage_profile_elapsed = 0.0;
}

bool PowerTracePathState::enabled() const {
    return out.good();
}

bool readPowerBoolEnv(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env) return default_value;
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(value.empty() || value == "0" || value == "false" || value == "no");
}

float readPowerFloatEnv(const char* name, float default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') return default_value;
    char* end = nullptr;
    const float value = std::strtof(env, &end);
    return (end != env && std::isfinite(value)) ? value : default_value;
}

void resetPowerStageProfileElapsed() {
    g_power_stage_profile_elapsed = 0.0;
}

void addPowerStageProfileElapsed(double elapsed) {
    if (std::isfinite(elapsed) && elapsed > 0.0) {
        g_power_stage_profile_elapsed += elapsed;
    }
}

double powerStageProfileElapsed() {
    return g_power_stage_profile_elapsed;
}

double canonicalPowerTimeScale(float scale) {
    const double value = static_cast<double>(scale);
    if (!std::isfinite(value) || value <= 0.0) return value;
    constexpr double known_scales[] = {1.0, 1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12, 1.0e-15};
    for (double known : known_scales) {
        if (std::abs(value - known) <= known * 1.0e-5) return known;
    }
    return value;
}

float powerDensityForPeriod(double transitions, float period, double time_scale) {
    if (!std::isfinite(period) || period <= 0.0f ||
        !std::isfinite(time_scale) || time_scale <= 0.0) {
        return 0.0f;
    }
    return static_cast<float>(transitions / (static_cast<double>(period) * time_scale));
}

std::string normalizeTracePinName(std::string name) {
    name.erase(0, name.find_first_not_of(" \t\r\n"));
    size_t end = name.find_last_not_of(" \t\r\n");
    if (end == std::string::npos) return "";
    name.erase(end + 1);
    name.erase(std::remove(name.begin(), name.end(), '\\'), name.end());
    return name;
}

std::string normalizePowerActivitySnapshotName(std::string name) {
    name.erase(0, name.find_first_not_of(" \t\r\n\""));
    size_t end = name.find_last_not_of(" \t\r\n\"");
    if (end == std::string::npos) return "";
    name.erase(end + 1);
    name.erase(std::remove(name.begin(), name.end(), '\\'), name.end());
    std::replace(name.begin(), name.end(), ':', '/');
    return name;
}

std::string csvEscapePowerActivitySnapshot(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') escaped += "\"\"";
        else escaped += ch;
    }
    escaped += '"';
    return escaped;
}

int readPowerActivitySnapshotMaxPass(const char* env_name, int default_value) {
    const char* env = std::getenv(env_name);
    if (!env || env[0] == '\0') return default_value;
    return std::max(0, std::atoi(env));
}

std::vector<std::string> readPowerTracePinQueries() {
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

std::vector<std::string> readPowerRootProbePinQueries() {
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

std::vector<int> resolvePowerTracePins(const std::vector<std::string>& queries,
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

bool parsePowerBool(const std::string& value) {
    std::string text = value;
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text == "1" || text == "true" || text == "yes" || text == "y";
}

std::string normalizePowerPathName(std::string name) {
    name.erase(0, name.find_first_not_of(" \t\r\n\""));
    size_t end = name.find_last_not_of(" \t\r\n\"");
    if (end == std::string::npos) return "";
    name.erase(end + 1);
    name.erase(std::remove(name.begin(), name.end(), '\\'), name.end());
    std::replace(name.begin(), name.end(), ':', '/');
    return name;
}

bool powerPathNameMatches(const std::string& pin_name, const std::string& query) {
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

std::vector<std::string> readPowerPathTargetQueries() {
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

std::unordered_set<std::string> readOpenroadSeedRootNames(const char* file_name) {
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

std::vector<int> resolvePowerPathTargetPins(const std::vector<std::string>& queries,
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

PowerTracePathState loadPowerTracePathState(const char* trace_path_file,
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

bool evalPowerExprWithPortValues(const PowerExpr& expr,
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

bool evalPowerExprActivity(const PowerExpr& expr,
                                  const LibertyCell* cell,
                                  const gp::GPNode& node,
                                  const std::vector<CpuActivity>& pin_activity,
                                  float& density,
                                  float& duty,
                                  const std::unordered_map<int, int>* const_port_values,
                                  const std::unordered_set<int>* zero_density_ports) {
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

const std::string& seqClockExpr(const SequentialPower* seq) {
    static const std::string empty;
    if (!seq) return empty;
    if (seq->is_latch_ && !seq->enable_expr_.empty()) return seq->enable_expr_;
    return seq->clocked_on_expr_;
}

}  // namespace gt
