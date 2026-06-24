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

bool PowerTracePathWriter::enabled() const {
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

PowerTracePathWriter loadPowerTracePathWriter(const char* trace_path_file,
                                                   const char* out_file,
                                                   const std::vector<int>& pin_to_node) {
    PowerTracePathWriter state;
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

// Small ROBDD evaluator used to mirror OpenROAD/CUDD activity semantics.
//
// Encoding:
//   edge 0 = constant 1
//   edge 1 = constant 0
//   nonterminal edge = (node_id << 1) | complemented_bit
//
// The complemented edge bit lets us represent NOT without creating another
// node. This is also why "true" is edge 0 and "false" is edge 1.
PowerExprBddActivityEvaluator::PowerExprBddActivityEvaluator(
    const LibertyCell* cell,
    const gp::GPNode& node,
    const std::vector<CpuActivity>& pin_activity,
    const std::unordered_map<int, int>* const_port_values,
    const std::unordered_set<int>* zero_density_ports)
    : cell_(cell),
      node_(node),
      pin_activity_(pin_activity),
      zero_density_ports_(zero_density_ports),
      fixed_port_values_(cell ? cell->ports_.size() : 0, -1) {
    if (!const_port_values) return;
    for (const auto& [port_id, value] : *const_port_values) {
        if (port_id >= 0 && port_id < static_cast<int>(fixed_port_values_.size()))
            fixed_port_values_[port_id] = static_cast<int8_t>(value ? 1 : 0);
    }
}

PowerExprBddActivityEvaluator::~PowerExprBddActivityEvaluator() = default;

bool PowerExprBddActivityEvaluator::evaluate(const PowerExpr& expr,
                                             float& density,
                                             float& duty) {
    if (!cell_) return false;

    // Register variables in Liberty port order before building nodes. The BDD
    // shape and recursive float rounding depend on a stable order.
    if (!registerExprVariables(expr)) return false;

    int root = zeroEdge();
    if (!buildBdd(expr, root)) return false;

    duty = evalDuty(root);
    density = evalDensity(root);
    return std::isfinite(density) && std::isfinite(duty);
}

bool PowerExprBddActivityEvaluator::BddKey::operator==(const BddKey& other) const {
    return var == other.var && low == other.low && high == other.high;
}

size_t PowerExprBddActivityEvaluator::BddKeyHash::operator()(const BddKey& key) const {
    size_t h = std::hash<int>{}(key.var);
    h ^= std::hash<int>{}(key.low + 0x9e3779b9 + (h << 6) + (h >> 2));
    h ^= std::hash<int>{}(key.high + 0x9e3779b9 + (h << 6) + (h >> 2));
    return h;
}

bool PowerExprBddActivityEvaluator::ApplyKey::operator==(const ApplyKey& other) const {
    return op == other.op && left == other.left && right == other.right;
}

size_t PowerExprBddActivityEvaluator::ApplyKeyHash::operator()(const ApplyKey& key) const {
    size_t h = std::hash<int>{}(key.op);
    h ^= std::hash<int>{}(key.left + 0x9e3779b9 + (h << 6) + (h >> 2));
    h ^= std::hash<int>{}(key.right + 0x9e3779b9 + (h << 6) + (h >> 2));
    return h;
}

int PowerExprBddActivityEvaluator::oneEdge() { return 0; }

int PowerExprBddActivityEvaluator::zeroEdge() { return 1; }

int PowerExprBddActivityEvaluator::noVar() { return std::numeric_limits<int>::max(); }

int PowerExprBddActivityEvaluator::edgeId(int edge) { return edge >> 1; }

bool PowerExprBddActivityEvaluator::edgeInv(int edge) { return (edge & 1) != 0; }

int PowerExprBddActivityEvaluator::edgeNot(int edge) { return edge ^ 1; }

bool PowerExprBddActivityEvaluator::resolvePortPin(int port_id, int& pin_id) const {
    if (!cell_ || port_id < 0 || port_id >= static_cast<int>(cell_->ports_.size()))
        return false;
    const std::string& port_name = cell_->ports_[port_id]->name;
    auto pin_itr = node_.portMap.find(port_name);
    if (pin_itr == node_.portMap.end()) {
        pin_id = -1;
        return true;
    }
    pin_id = pin_itr->second;
    return pin_id >= 0 && pin_id < static_cast<int>(pin_activity_.size());
}

int PowerExprBddActivityEvaluator::makeNode(int var, int low, int high) {
    if (low == high) return low;

    bool result_inv = false;
    // CUDD stores the then/high edge regular. If a high child is complemented,
    // move that complement onto the returned edge instead. This keeps the same
    // recursive complemented-edge shape as OpenROAD's power code.
    if (edgeInv(high)) {
        low = edgeNot(low);
        high = edgeNot(high);
        result_inv = true;
    }

    BddKey key{var, low, high};
    auto itr = unique_nodes_.find(key);
    int id = 0;
    if (itr == unique_nodes_.end()) {
        id = static_cast<int>(nodes_.size()) + 1;
        nodes_.push_back(BddNode{var, low, high});
        unique_nodes_.emplace(key, id);
    } else {
        id = itr->second;
    }

    const int edge = id << 1;
    return result_inv ? edgeNot(edge) : edge;
}

int PowerExprBddActivityEvaluator::topVar(int edge) const {
    const int id = edgeId(edge);
    return id == 0 ? noVar() : nodes_[id - 1].var;
}

int PowerExprBddActivityEvaluator::cofTop(int edge, int var, bool high_child) const {
    const int id = edgeId(edge);
    if (id == 0 || nodes_[id - 1].var != var) return edge;

    const int child = high_child ? nodes_[id - 1].high : nodes_[id - 1].low;
    return edgeInv(edge) ? edgeNot(child) : child;
}

int PowerExprBddActivityEvaluator::apply(int op, int left, int right) {
    // AND/OR are commutative; normalizing operand order increases cache hits.
    if ((op == 0 || op == 1) && right < left) std::swap(left, right);

    ApplyKey key{op, left, right};
    auto cache_itr = apply_cache_.find(key);
    if (cache_itr != apply_cache_.end()) return cache_itr->second;

    int result = zeroEdge();
    const int left_id = edgeId(left);
    const int right_id = edgeId(right);
    if (left_id == 0 && right_id == 0) {
        const bool left_value = !edgeInv(left);
        const bool right_value = !edgeInv(right);
        bool value = false;
        if (op == 0) value = left_value && right_value;
        else if (op == 1) value = left_value || right_value;
        else value = left_value != right_value;
        result = value ? oneEdge() : zeroEdge();
    } else {
        const int var = std::min(topVar(left), topVar(right));
        const int low = apply(op, cofTop(left, var, false), cofTop(right, var, false));
        const int high = apply(op, cofTop(left, var, true), cofTop(right, var, true));
        result = makeNode(var, low, high);
    }

    apply_cache_.emplace(key, result);
    return result;
}

int PowerExprBddActivityEvaluator::restrictVar(int edge, int target_var, bool high_child) {
    const int id = edgeId(edge);
    if (id == 0) return edge;

    const auto& node = nodes_[id - 1];
    if (node.var > target_var) return edge;

    const long long key = (static_cast<long long>(edge) << 32)
        ^ (static_cast<long long>(target_var) << 1)
        ^ static_cast<long long>(high_child ? 1 : 0);
    auto cache_itr = restrict_cache_.find(key);
    if (cache_itr != restrict_cache_.end()) return cache_itr->second;

    int result = edge;
    if (node.var == target_var) {
        result = high_child ? node.high : node.low;
    } else {
        const int low = restrictVar(node.low, target_var, high_child);
        const int high = restrictVar(node.high, target_var, high_child);
        result = makeNode(node.var, low, high);
    }
    if (edgeInv(edge)) result = edgeNot(result);

    restrict_cache_.emplace(key, result);
    return result;
}

int PowerExprBddActivityEvaluator::ensureVar(int port_id, int pin_id) {
    auto itr = port_to_var_.find(port_id);
    if (itr != port_to_var_.end()) return itr->second;

    const int var = static_cast<int>(var_ports_.size());
    port_to_var_[port_id] = var;
    var_ports_.push_back(port_id);

    const bool has_pin = pin_id >= 0 && pin_id < static_cast<int>(pin_activity_.size());
    var_has_pin_.push_back(has_pin ? 1 : 0);
    var_duties_.push_back(has_pin ? std::clamp(pin_activity_[pin_id].duty, 0.0f, 1.0f) : 0.0f);

    const bool zero_density = zero_density_ports_ && zero_density_ports_->count(port_id) != 0;
    var_densities_.push_back((has_pin && !zero_density) ? pin_activity_[pin_id].density : 0.0f);
    return var;
}

bool PowerExprBddActivityEvaluator::registerExprVariables(const PowerExpr& expr) {
    std::vector<std::pair<int, int>> expr_vars;
    for (const auto& op : expr.ops()) {
        if (op.opcode != PowerExprOpcode::port) continue;
        int pin_id = -1;
        if (!resolvePortPin(op.port_id, pin_id)) return false;
        if (pin_id >= 0) expr_vars.emplace_back(op.port_id, pin_id);
    }

    std::sort(expr_vars.begin(), expr_vars.end(),
              [](const auto& left, const auto& right) { return left.first < right.first; });
    expr_vars.erase(std::unique(expr_vars.begin(), expr_vars.end(),
                                [](const auto& left, const auto& right) {
                                    return left.first == right.first;
                                }),
                    expr_vars.end());

    for (const auto& [port_id, pin_id] : expr_vars)
        ensureVar(port_id, pin_id);
    return true;
}

bool PowerExprBddActivityEvaluator::pushPort(std::vector<int>& stack, int port_id) {
    int pin_id = -1;
    if (!resolvePortPin(port_id, pin_id)) return false;

    if (pin_id < 0) {
        // Missing Liberty ports are treated as a fixed value if supplied by the
        // caller; otherwise OpenROAD-compatible fallback is constant 0.
        const int8_t fixed_value =
            port_id >= 0 && port_id < static_cast<int>(fixed_port_values_.size())
                ? fixed_port_values_[port_id]
                : static_cast<int8_t>(-1);
        stack.push_back(fixed_value > 0 ? oneEdge() : zeroEdge());
        return true;
    }

    const int var = ensureVar(port_id, pin_id);
    stack.push_back(makeNode(var, zeroEdge(), oneEdge()));
    return true;
}

bool PowerExprBddActivityEvaluator::buildBdd(const PowerExpr& expr, int& root) {
    std::vector<int> stack;
    for (const auto& op : expr.ops()) {
        switch (op.opcode) {
            case PowerExprOpcode::port:
                if (!pushPort(stack, op.port_id)) return false;
                break;
            case PowerExprOpcode::const_zero:
                stack.push_back(zeroEdge());
                break;
            case PowerExprOpcode::const_one:
                stack.push_back(oneEdge());
                break;
            case PowerExprOpcode::logical_not:
                if (stack.empty()) return false;
                stack.back() = edgeNot(stack.back());
                break;
            case PowerExprOpcode::logical_and:
                if (!applyBinary(stack, 0)) return false;
                break;
            case PowerExprOpcode::logical_or:
                if (!applyBinary(stack, 1)) return false;
                break;
            case PowerExprOpcode::logical_xor:
                if (!applyBinary(stack, 2)) return false;
                break;
        }
    }

    if (stack.size() != 1) return false;
    root = stack.back();
    return true;
}

bool PowerExprBddActivityEvaluator::applyBinary(std::vector<int>& stack, int op) {
    if (stack.size() < 2) return false;
    const int right = stack.back();
    stack.pop_back();
    const int left = stack.back();
    stack.back() = apply(op, left, right);
    return true;
}

float PowerExprBddActivityEvaluator::evalDuty(int edge) const {
    const int id = edgeId(edge);
    // Terminal edges encode constants: edge 0 is logic 1, edge 1 is logic 0.
    if (id == 0) return edgeInv(edge) ? 0.0f : 1.0f;

    const auto& node = nodes_[id - 1];
    if (node.var >= 0 && node.var < static_cast<int>(var_has_pin_.size())
        && !var_has_pin_[node.var])
        return 0.0f;

    // A BDD node means F = var ? high : low.  The input duty is P(var=1),
    // so the output duty is the weighted probability of the two cofactors:
    //   P(F=1) = P(F_low=1) * P(var=0) + P(F_high=1) * P(var=1).
    const float duty0 = evalDuty(node.low);
    const float duty1 = evalDuty(node.high);
    const float var_duty = var_duties_[node.var];
    float result = duty0 * (1.0f - var_duty) + duty1 * var_duty;

    // A complemented edge represents !F without creating another BDD node.
    if (edgeInv(edge)) result = 1.0f - result;
    return std::clamp(result, 0.0f, 1.0f);
}

float PowerExprBddActivityEvaluator::evalDensity(int root) {
    float density = 0.0f;

    // Output density is the sum of each input toggle density multiplied by the
    // probability that toggling that input changes the expression output.
    // Keep a deterministic Liberty-port order so floating point accumulation is
    // stable and matches the expected OpenROAD-style ordering.
    std::vector<int> var_order(var_ports_.size());
    std::iota(var_order.begin(), var_order.end(), 0);
    std::sort(var_order.begin(), var_order.end(), [&](int left, int right) {
        return var_ports_[left] < var_ports_[right];
    });

    for (int var : var_order) {
        if (var < 0 || var >= static_cast<int>(var_has_pin_.size()) || !var_has_pin_[var])
            continue;

        // Boolean difference for this input:
        //   diff = F(var=0) XOR F(var=1).
        // diff is 1 exactly for the other-input combinations where this input
        // toggle propagates to the output.  Its duty is therefore the
        // propagation probability for var.
        restrict_cache_.clear();
        const int low = restrictVar(root, var, false);
        restrict_cache_.clear();
        const int high = restrictVar(root, var, true);
        const int diff = apply(2, low, high);
        const float diff_duty = evalDuty(diff);

        density += var_densities_[var] * diff_duty;
    }

    return density;
}

bool evalPowerExprActivity(const PowerExpr& expr,
                           const LibertyCell* cell,
                           const gp::GPNode& node,
                           const std::vector<CpuActivity>& pin_activity,
                           float& density,
                           float& duty,
                           const std::unordered_map<int, int>* const_port_values,
                           const std::unordered_set<int>* zero_density_ports) {
    PowerExprBddActivityEvaluator evaluator(cell, node, pin_activity,
                                            const_port_values, zero_density_ports);
    return evaluator.evaluate(expr, density, duty);
}

const std::string& seqClockExpr(const SequentialPower* seq) {
    static const std::string empty;
    if (!seq) return empty;
    if (seq->is_latch_ && !seq->enable_expr_.empty()) return seq->enable_expr_;
    return seq->clocked_on_expr_;
}

}  // namespace gt
