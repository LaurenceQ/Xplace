#include "PowerCudaActivityDevice.cuh"

#include <cmath>
#include <cstdint>

namespace gt {

__device__ bool g_power_allow_clock_activity_override = true;
__device__ float g_power_min_activity_density = 1.0e-10f;
__device__ float g_power_min_activity_duty = 0.0f;
__device__ bool g_power_disable_activity_slew_cap = false;
__device__ float g_power_seq_clock_limit_rel_tol = 0.0f;
__device__ float g_power_seq_pending_min_density = 0.0f;
__device__ float g_power_activity_clock_density_cap = 3.4028234663852886e38f;
__device__ int g_power_direct_ordered_seq_seed = 0;
__device__ int g_power_require_known_seq_data = 0;
__device__ int g_power_disable_direct_expr = 0;
__device__ int g_power_check_direct_expr = 0;
__device__ int g_power_direct_expr_mismatch_count = 0;
__device__ int g_power_direct_expr_max_vars = 2;
__device__ float g_power_direct_expr_density_rel_tol = 1.0e-4f;
__device__ float g_power_direct_expr_duty_abs_tol = 1.0e-5f;

__device__ float PowerActivityOps::percentChange(float value, float prev) {
    if (prev == 0.0f) return value == 0.0f ? 0.0f : 1.0f;
    return fabsf(value - prev) / fabsf(prev);
}

__device__ float PowerActivityOps::clampActivityDuty(float duty) {
    float u = fminf(fmaxf(duty, 0.0f), 1.0f);
    const float eps = fmaxf(g_power_min_activity_duty, 0.0f);
    if (eps > 0.0f) {
        if (u < eps) u = 0.0f;
        else if ((1.0f - u) < eps) u = 1.0f;
    }
    return u;
}

__device__ bool PowerActivityOps::shouldMarkPendingSeq(float density) {
    return density >= fmaxf(g_power_seq_pending_min_density, 0.0f);
}

namespace {

__device__ __forceinline__ bool power_clock_slew_pin_marked(const PowerGraphDevice& graph,
                                                            int pin) {
    const int* pins = graph.power_clock_slew_pins;
    int lo = 0;
    int hi = graph.num_power_clock_slew_pins;
    while (lo < hi) {
        const int mid = lo + ((hi - lo) >> 1);
        const int value = pins[mid];
        if (value < pin) lo = mid + 1;
        else hi = mid;
    }
    return lo < graph.num_power_clock_slew_pins && pins[lo] == pin;
}

__device__ __forceinline__ float power_clock_slew_value(const PowerGraphDevice& graph,
                                                        int pin,
                                                        int attr) {
    if (!graph.power_clock_slew_pins || graph.num_power_clock_slew_pins <= 0 ||
        pin < 0 || attr < 0 || attr >= NUM_ATTR ||
        !power_clock_slew_pin_marked(graph, pin))
        return nanf("");
    if (graph.pin_clock_ids && graph.clock_slews) {
        const uint16_t clock_id = graph.pin_clock_ids[pin];
        if (clock_id != 65535u && clock_id < static_cast<uint16_t>(graph.clock_count)) {
            const float slew = graph.clock_slews[static_cast<int>(clock_id) * NUM_ATTR + attr];
            if (isfinite(slew)) return slew;
        }
    }
    return graph.power_clock_slew_fallback[attr];
}

}  // namespace

__device__ float PowerActivityOps::maxActivityDensityFromSlew(int pin) const {
    if (g_power_disable_activity_slew_cap) return 3.4028234663852886e38f;
    const float* pinSlew = model->graph.pinSlew;
    const bool has_power_clock_slews =
        model->graph.power_clock_slew_pins && model->graph.num_power_clock_slew_pins > 0;
    const float time_unit = model->config.time_unit;
    if ((!pinSlew && !has_power_clock_slews) || pin < 0 || !(time_unit > 0.0f))
        return 3.4028234663852886e38f;
    float min_rf_slew = 3.4028234663852886e38f;
    #pragma unroll
    for (int base = 0; base < NUM_ATTR; base += 2) {
        float rise = pinSlew ? pinSlew[pin * NUM_ATTR + base] : nanf("");
        float fall = pinSlew ? pinSlew[pin * NUM_ATTR + base + 1] : nanf("");
        const bool use_clock_slew_override =
            model->graph.is_seq_clock_input_pin && model->graph.is_seq_clock_input_pin[pin];
        if (use_clock_slew_override && has_power_clock_slews) {
            const float clock_rise = power_clock_slew_value(model->graph, pin, base);
            const float clock_fall = power_clock_slew_value(model->graph, pin, base + 1);
            if (isfinite(clock_rise) && isfinite(clock_fall)) {
                rise = clock_rise;
                fall = clock_fall;
            }
        }
        if (isfinite(rise) && isfinite(fall)) {
            const float avg = 0.5f * (rise + fall) * time_unit;
            if (avg > 0.0f && avg < min_rf_slew) min_rf_slew = avg;
        }
    }
    return (min_rf_slew < 3.4028234663852886e38f) ? (1.0f / min_rf_slew)
                                                  : 3.4028234663852886e38f;
}

__device__ bool PowerActivityOps::seqDensityExceedsClockLimit(float in_density, float clk_density) {
    const float limit = clk_density * 0.5f;
    return in_density > limit * (1.0f + fmaxf(g_power_seq_clock_limit_rel_tol, 0.0f));
}

__device__ bool PowerActivityOps::setActivity(int pin,
                                              float new_density,
                                              float new_duty,
                                              int new_origin,
                                              bool force) const {
    float* density = scratch->density;
    float* duty = scratch->duty;
    uint8_t* origin = scratch->origin;
    if (!force && origin[pin] == 2 && !g_power_allow_clock_activity_override) return false;
    const float prev_density = density[pin];
    const float prev_duty = duty[pin];
    const uint8_t prev_origin = origin[pin];
    const float max_density = force
        ? g_power_activity_clock_density_cap
        : fminf(maxActivityDensityFromSlew(pin),
                g_power_activity_clock_density_cap);
    float d = fminf(fmaxf(new_density, 0.0f), max_density);
    if (fabsf(d) < g_power_min_activity_density) d = 0.0f;
    const float u = clampActivityDuty(new_duty);
    const bool value_changed = percentChange(d, prev_density) > 0.01f
        || percentChange(u, prev_duty) > 0.01f;
    const bool changed = value_changed || prev_origin != new_origin;
    density[pin] = d;
    duty[pin] = u;
    origin[pin] = static_cast<uint8_t>(new_origin);
    return changed;
}

__device__ void PowerActivityOps::enqueueAdjacent(int pin) const {
    const auto& graph = model->graph;
    const uint8_t* is_load_pin = graph.is_load_pin;
    const int* pin2net_map = graph.pin2net_map;
    const int* net_driver_pin = graph.net_driver_pin;
    const int* flat_net2pin_start_map = graph.flat_net2pin_start_map;
    const int* flat_net2pin_map = graph.flat_net2pin_map;
    const index_type* pin_forward_arc_list_end = graph.pin_forward_arc_list_end;
    const index_type* pin_forward_arc_list = graph.pin_forward_arc_list;
    const index_type* timing_arc_to_pin_id = graph.timing_arc_to_pin_id;
    const uint8_t* arc_types = graph.arc_types;
    const uint32_t* seq_output_arc_keep = graph.seq_output_arc_keep;
    const uint8_t* arc_skip = graph.arc_skip;
    const uint8_t* is_seq_output_pin = graph.is_seq_output_pin;
    const int* pin_power_level = graph.pin_power_level;
    uint8_t* active_level = scratch->active_level;
    const int num_power_levels = scratch->num_power_levels;
    uint32_t* active = scratch->active;
    if (is_load_pin && pin2net_map && net_driver_pin && flat_net2pin_start_map && flat_net2pin_map) {
        const int net = pin2net_map[pin];
        if (net >= 0 && net_driver_pin[net] == pin) {
            const int start = flat_net2pin_start_map[net];
            const int end = flat_net2pin_start_map[net + 1];
            for (int pos = start; pos < end; ++pos) {
                const int sink = flat_net2pin_map[pos];
                if (sink < 0 || sink == pin || !is_load_pin[sink]) continue;
                power_activity_flag_atomic_test_and_set(active, sink);
                if (pin_power_level && active_level) {
                    const int level = pin_power_level[sink];
                    if (level >= 0 && level < num_power_levels) active_level[level] = 1;
                }
            }
        }
    }
    for (index_type i = pin_forward_arc_list_end[pin]; i < pin_forward_arc_list_end[pin + 1]; i++) {
        const int arc = pin_forward_arc_list[i];
        if (arc_skip && arc_skip[arc]) continue;
        const int to_pin = timing_arc_to_pin_id[arc];
        if (to_pin < 0) continue;
        if (arc_types && arc_types[arc] == 1 && is_seq_output_pin && is_seq_output_pin[to_pin] &&
            !power_activity_flag_test(seq_output_arc_keep, arc))
            continue;
        power_activity_flag_atomic_test_and_set(active, to_pin);
        if (pin_power_level && active_level) {
            const int level = pin_power_level[to_pin];
            if (level >= 0 && level < num_power_levels) active_level[level] = 1;
        }
    }
}

__device__ void PowerActivityOps::activatePin(int pin) const {
    const int* pin_power_level = model->graph.pin_power_level;
    uint8_t* active_level = scratch->active_level;
    const int num_power_levels = scratch->num_power_levels;
    uint32_t* active = scratch->active;
    if (pin < 0 || !active) return;
    power_activity_flag_atomic_test_and_set(active, pin);
    if (pin_power_level && active_level) {
        const int level = pin_power_level[pin];
        if (level >= 0 && level < num_power_levels) active_level[level] = 1;
    }
}

__device__ bool PowerActivityOps::setClockGateOutput(int pin) const {
    const int* clock_gate_clock_for_out = model->graph.clock_gate_clock_for_out;
    const int* clock_gate_enable_for_out = model->graph.clock_gate_enable_for_out;
    float* density = scratch->density;
    float* duty = scratch->duty;
    uint8_t* origin = scratch->origin;
    if (!clock_gate_clock_for_out || !clock_gate_enable_for_out) return false;
    const int clk = clock_gate_clock_for_out[pin];
    const int en = clock_gate_enable_for_out[pin];
    if (clk < 0 || en < 0) return false;
    if (origin && origin[clk] == 0 && origin[en] == 0) return false;
    const float out_density = density[clk] * duty[en] + density[en] * duty[clk];
    const float out_duty = duty[clk] * duty[en];
    return setActivity(pin, out_density, out_duty, 3, false);
}

__device__ void PowerActivityOps::enqueueClockGateOutput(int pin) const {
    const int* clock_gate_out_for_input = model->graph.clock_gate_out_for_input;
    const int* pin_power_level = model->graph.pin_power_level;
    uint8_t* active_level = scratch->active_level;
    const int num_power_levels = scratch->num_power_levels;
    uint32_t* active = scratch->active;
    if (!clock_gate_out_for_input) return;
    const int out_pin = clock_gate_out_for_input[pin];
    if (out_pin < 0) return;
    power_activity_flag_atomic_test_and_set(active, out_pin);
    if (pin_power_level && active_level) {
        const int level = pin_power_level[out_pin];
        if (level >= 0 && level < num_power_levels) active_level[level] = 1;
    }
}

__device__ bool PowerExprEval::evalBool(int expr_id,
                                        uint64_t bits,
                                        int force_var,
                                        int force_val,
                                        const int* var_pins,
                                        int var_count,
                                        int8_t& value) const {
    if (expr_id < 0) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return false;
    int8_t stack[128];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        switch (op.op) {
            case 0: {
                int var = -1;
                for (int i = 0; i < var_count; i++) {
                    if (var_pins[i] == op.arg) {
                        var = i;
                        break;
                    }
                }
                if (var < 0 || sp >= 128) return false;
                int bit = (bits >> var) & 1ULL;
                if (var == force_var) bit = force_val;
                stack[sp++] = static_cast<int8_t>(bit);
                break;
            }
            case 1:
                if (sp >= 128) return false;
                stack[sp++] = 0;
                break;
            case 2:
                if (sp >= 128) return false;
                stack[sp++] = 1;
                break;
            case 3: {
                if (sp < 1) return false;
                const int8_t a = stack[--sp];
                stack[sp++] = a < 0 ? -1 : static_cast<int8_t>(!a);
                break;
            }
            case 4: {
                if (sp < 2) return false;
                const int8_t b = stack[--sp];
                const int8_t a = stack[--sp];
                if (a == 0 || b == 0) stack[sp++] = 0;
                else if (a == 1 && b == 1) stack[sp++] = 1;
                else stack[sp++] = -1;
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const int8_t b = stack[--sp];
                const int8_t a = stack[--sp];
                if (a == 1 || b == 1) stack[sp++] = 1;
                else if (a == 0 && b == 0) stack[sp++] = 0;
                else stack[sp++] = -1;
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const int8_t b = stack[--sp];
                const int8_t a = stack[--sp];
                if (a < 0 || b < 0) stack[sp++] = -1;
                else stack[sp++] = static_cast<int8_t>((a != 0) ^ (b != 0));
                break;
            }
            case 7:
                if (sp >= 128) return false;
                stack[sp++] = -1;
                break;
            default:
                return false;
        }
    }
    if (sp != 1 || stack[0] < 0) return false;
    value = stack[0];
    return true;
}

namespace {

constexpr int POWER_BDD_MAX_VARS = 32;
constexpr int POWER_BDD_MAX_NODES = 256;
constexpr int POWER_BDD_MAX_APPLY_CACHE = 1024;

struct PowerBddNodeCuda {
    int var = -1;
    int low = 0;
    int high = 0;

    PowerBddNodeCuda() = default;
    __device__ __forceinline__ PowerBddNodeCuda(int var_, int low_, int high_)
        : var(var_), low(low_), high(high_) {}
};

struct PowerBddApplyCacheCuda {
    int op = -1;
    int left = 0;
    int right = 0;
    int result = 0;

    PowerBddApplyCacheCuda() = default;
    __device__ __forceinline__ PowerBddApplyCacheCuda(int op_, int left_, int right_, int result_)
        : op(op_), left(left_), right(right_), result(result_) {}
};

struct PowerBddContextCuda {
    PowerBddNodeCuda nodes[POWER_BDD_MAX_NODES];
    PowerBddApplyCacheCuda apply_cache[POWER_BDD_MAX_APPLY_CACHE];
    int node_count = 0;
    int apply_count = 0;
    int var_pins[POWER_BDD_MAX_VARS];
    int var_keys[POWER_BDD_MAX_VARS];
    uint8_t var_has_pin[POWER_BDD_MAX_VARS];
    float var_duties[POWER_BDD_MAX_VARS];
    float var_densities[POWER_BDD_MAX_VARS];
    int var_count = 0;
    bool ok = true;

    __device__ static int edgeId(int edge);
    __device__ static bool edgeInv(int edge);
    __device__ static int notEdge(int edge);
    __device__ int makeNode(int var, int low, int high);
    __device__ int topVar(int edge) const;
    __device__ int cofTop(int edge, int var, bool high_child) const;
    __device__ int apply(int op, int left, int right);
    __device__ int restrict(int edge, int target_var, bool high_child);
    __device__ float evalDuty(int edge) const;
    __device__ int ensureVar(int var_key,
                             int pin,
                             const float* pin_density,
                             const float* pin_duty,
                             bool zero_density);
    __device__ int findVar(int var_key) const;
};

__device__ int PowerBddContextCuda::edgeId(int edge) { return edge >> 1; }
__device__ bool PowerBddContextCuda::edgeInv(int edge) { return (edge & 1) != 0; }
__device__ int PowerBddContextCuda::notEdge(int edge) { return edge ^ 1; }

__device__ int PowerBddContextCuda::makeNode(int var, int low, int high) {
    if (low == high) return low;
    bool result_inv = false;
    // Mirror CUDD's complemented-edge normalization: the then/high edge is
    // stored regular and a complement is moved onto the returned edge.
    if (edgeInv(high)) {
        low = notEdge(low);
        high = notEdge(high);
        result_inv = true;
    }
    for (int i = 0; i < node_count; i++) {
        const auto& node = nodes[i];
        if (node.var == var && node.low == low && node.high == high) {
            const int edge = (i + 1) << 1;
            return result_inv ? notEdge(edge) : edge;
        }
    }
    if (node_count >= POWER_BDD_MAX_NODES) {
        ok = false;
        return 1;
    }
    const int id = ++node_count;
    nodes[id - 1] = PowerBddNodeCuda{var, low, high};
    const int edge = id << 1;
    return result_inv ? notEdge(edge) : edge;
}

__device__ int PowerBddContextCuda::topVar(int edge) const {
    const int id = edgeId(edge);
    return id == 0 ? 0x3fffffff : nodes[id - 1].var;
}

__device__ int PowerBddContextCuda::cofTop(int edge, int var, bool high_child) const {
    const int id = edgeId(edge);
    if (id == 0 || nodes[id - 1].var != var) return edge;
    const int child = high_child ? nodes[id - 1].high : nodes[id - 1].low;
    return edgeInv(edge) ? notEdge(child) : child;
}

__device__ int PowerBddContextCuda::apply(int op, int left, int right) {
    if (op >= 0 && op <= 2 && right < left) {
        const int tmp = left;
        left = right;
        right = tmp;
    }
    for (int i = 0; i < apply_count; i++) {
        const auto& cache = apply_cache[i];
        if (cache.op == op && cache.left == left && cache.right == right) return cache.result;
    }

    int result = 1;
    const int left_id = edgeId(left);
    const int right_id = edgeId(right);
    if (left_id == 0 && right_id == 0) {
        const bool left_value = !edgeInv(left);
        const bool right_value = !edgeInv(right);
        bool value = false;
        if (op == 0) value = left_value && right_value;
        else if (op == 1) value = left_value || right_value;
        else value = left_value != right_value;
        result = value ? 0 : 1;
    } else {
        const int left_top = topVar(left);
        const int right_top = topVar(right);
        const int var = left_top < right_top ? left_top : right_top;
        const int low = apply(op, cofTop(left, var, false), cofTop(right, var, false));
        const int high = apply(op, cofTop(left, var, true), cofTop(right, var, true));
        result = makeNode(var, low, high);
    }

    if (apply_count < POWER_BDD_MAX_APPLY_CACHE) {
        apply_cache[apply_count++] = PowerBddApplyCacheCuda{op, left, right, result};
    }
    return result;
}

__device__ int PowerBddContextCuda::restrict(int edge, int target_var, bool high_child) {
    const int id = edgeId(edge);
    if (id == 0) return edge;
    const auto node = nodes[id - 1];
    if (node.var > target_var) return edge;
    int result = edge;
    if (node.var == target_var) {
        result = high_child ? node.high : node.low;
    } else {
        const int low = restrict(node.low, target_var, high_child);
        const int high = restrict(node.high, target_var, high_child);
        result = makeNode(node.var, low, high);
    }
    return edgeInv(edge) ? notEdge(result) : result;
}

__device__ float PowerBddContextCuda::evalDuty(int edge) const {
    const int id = edgeId(edge);
    if (id == 0) return edgeInv(edge) ? 0.0f : 1.0f;
    const auto node = nodes[id - 1];
    if (node.var >= 0 && node.var < var_count && !var_has_pin[node.var])
        return 0.0f;
    const float duty0 = evalDuty(node.low);
    const float duty1 = evalDuty(node.high);
    const float var_duty = var_duties[node.var];
    const double result_d =
        static_cast<double>(duty0) * (1.0 - static_cast<double>(var_duty)) +
        static_cast<double>(duty1) * static_cast<double>(var_duty);
    float result = static_cast<float>(result_d);
    if (edgeInv(edge)) result = 1.0f - result;
    return fminf(fmaxf(result, 0.0f), 1.0f);
}

__device__ int PowerBddContextCuda::ensureVar(int var_key,
                                              int pin,
                                              const float* pin_density,
                                              const float* pin_duty,
                                              bool zero_density) {
    int insert_pos = var_count;
    for (int i = 0; i < var_count; i++) {
        if (var_keys[i] == var_key && zero_density)
            var_densities[i] = 0.0f;
        if (var_keys[i] == var_key) return i;
        if (insert_pos == var_count && var_keys[i] > var_key) insert_pos = i;
    }
    if (var_count >= POWER_BDD_MAX_VARS) {
        ok = false;
        return -1;
    }
    for (int i = var_count; i > insert_pos; --i) {
        var_keys[i] = var_keys[i - 1];
        var_pins[i] = var_pins[i - 1];
        var_has_pin[i] = var_has_pin[i - 1];
        var_duties[i] = var_duties[i - 1];
        var_densities[i] = var_densities[i - 1];
    }
    var_count++;
    var_keys[insert_pos] = var_key;
    var_pins[insert_pos] = pin;
    var_has_pin[insert_pos] = pin >= 0 ? 1 : 0;
    var_duties[insert_pos] = pin >= 0 ? PowerActivityOps::clampActivityDuty(pin_duty[pin]) : 0.0f;
    var_densities[insert_pos] = (pin >= 0 && !zero_density && pin_density) ? pin_density[pin] : 0.0f;
    return insert_pos;
}

__device__ int PowerBddContextCuda::findVar(int var_key) const {
    for (int i = 0; i < var_count; i++) {
        if (var_keys[i] == var_key) return i;
    }
    return -1;
}

struct PowerBddExprEval {
    const PowerExprEval* view = nullptr;
    PowerBddContextCuda ctx;
    int root = 1;

    __device__ explicit PowerBddExprEval(const PowerExprEval& view_) : view(&view_) {}
    __device__ bool buildExpr(int expr_id);
    __device__ __noinline__ bool activity(int expr_id,
                                          float& out_density,
                                          float& out_duty,
                                          int& out_var_count);
    __device__ __noinline__ float diffDuty(int expr_id, int diff_pin);
    static __device__ __noinline__ bool activityFallback(const PowerExprEval& view,
                                                         int expr_id,
                                                         float& out_density,
                                                         float& out_duty,
                                                         int& out_var_count);
    static __device__ __noinline__ float diffDutyFallback(const PowerExprEval& view,
                                                          int expr_id,
                                                          int diff_pin);
};

__device__ bool PowerBddExprEval::buildExpr(int expr_id) {
    if (expr_id < 0) return false;
    const int start = view->expr_start[expr_id];
    const int count = view->expr_count[expr_id];
    if (count <= 0 || count > 128) return false;
    for (int k = 0; k < count; k++) {
        const auto op = view->ops[start + k];
        int pin = -1;
        int var_key = -1;
        bool zero_density = op.zero_density != 0;
        if (op.op == 0) {
            pin = view->resolvePinArg(op.arg);
            if (pin < 0 && op.arg > -2) return false;
            if (pin < 0) continue;
            var_key = op.var_key >= 0 ? op.var_key : pin;
        } else if (op.op == 7) {
            var_key = op.var_key;
            zero_density = true;
        } else {
            continue;
        }
        if (var_key < 0) return false;
        if (ctx.ensureVar(var_key, pin, view->pin_density, view->pin_duty,
                          zero_density) < 0 || !ctx.ok)
            return false;
    }
    int stack[128];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = view->ops[start + k];
        switch (op.op) {
            case 0: {
                const int pin = view->resolvePinArg(op.arg);
                if (sp >= 128) return false;
                if (pin < 0 && op.arg > -2) return false;
                if (pin < 0) {
                    stack[sp++] = 1;
                    break;
                }
                const int var_key = op.var_key >= 0 ? op.var_key : pin;
                const int var = ctx.findVar(var_key);
                if (var < 0) return false;
                stack[sp++] = ctx.makeNode(var, 1, 0);
                break;
            }
            case 7: {
                if (sp >= 128) return false;
                const int var_key = op.var_key >= 0 ? op.var_key : -1;
                const int var = ctx.findVar(var_key);
                if (var < 0) return false;
                stack[sp++] = ctx.makeNode(var, 1, 0);
                break;
            }
            case 1:
                if (sp >= 128) return false;
                stack[sp++] = 1;
                break;
            case 2:
                if (sp >= 128) return false;
                stack[sp++] = 0;
                break;
            case 3: {
                if (sp < 1) return false;
                stack[sp - 1] = PowerBddContextCuda::notEdge(stack[sp - 1]);
                break;
            }
            case 4: {
                if (sp < 2) return false;
                const int right = stack[--sp];
                const int left = stack[--sp];
                stack[sp++] = ctx.apply(0, left, right);
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const int right = stack[--sp];
                const int left = stack[--sp];
                stack[sp++] = ctx.apply(1, left, right);
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const int right = stack[--sp];
                const int left = stack[--sp];
                stack[sp++] = ctx.apply(2, left, right);
                break;
            }
            default:
                return false;
        }
        if (!ctx.ok) return false;
    }
    if (sp != 1) return false;
    root = stack[0];
    return ctx.ok;
}

constexpr int POWER_DIRECT_PROB_STACK = 64;

struct PowerDirectProbValue {
    float duty = 0.0f;
    float diff = 0.0f;
    uint8_t has_diff = 0;

    PowerDirectProbValue() = default;
    __device__ __forceinline__ PowerDirectProbValue(float duty_,
                                                    float diff_,
                                                    uint8_t has_diff_)
        : duty(duty_), diff(diff_), has_diff(has_diff_) {}
};

struct PowerDirectActivityValue {
    float density = 0.0f;
    float duty = 0.0f;

    PowerDirectActivityValue() = default;
    __device__ __forceinline__ PowerDirectActivityValue(float density_, float duty_)
        : density(density_), duty(duty_) {}
};

struct PowerDirectExprEval {
    const PowerExprEval* view = nullptr;

    __device__ explicit PowerDirectExprEval(const PowerExprEval& view_) : view(&view_) {}
    __device__ int uniqueVarCount(int expr_id) const;
    __device__ bool isSafe(int expr_id) const;
    __device__ bool dutyPoly(int expr_id, float& out_duty) const;
    __device__ bool duty(int expr_id, float& out_duty) const;
    __device__ bool activityPoly(int expr_id,
                                 float& out_density,
                                 float& out_duty) const;
    __device__ bool activity(int expr_id,
                             float& out_density,
                             float& out_duty) const;
    __device__ bool diffDutyPoly(int expr_id,
                                 int diff_pin,
                                 float& out_duty) const;
    __device__ bool diffDuty(int expr_id,
                             int diff_pin,
                             float& out_duty) const;
};

__device__ int PowerDirectExprEval::uniqueVarCount(int expr_id) const {
    if (expr_id < 0) return -1;
    const int start = view->expr_start[expr_id];
    const int count = view->expr_count[expr_id];
    if (count <= 0 || count > 128) return -1;
    int keys[POWER_DIRECT_PROB_STACK];
    int key_count = 0;
    for (int k = 0; k < count; k++) {
        const auto op = view->ops[start + k];
        int key = -1;
        if (op.op == 0) {
            const int pin = view->resolvePinArg(op.arg);
            if (pin < 0 && op.arg > -2) return -1;
            if (pin < 0) continue;
            key = op.var_key >= 0 ? op.var_key : pin;
        } else if (op.op == 7) {
            return -1;
        } else {
            continue;
        }
        if (key < 0) return -1;
        for (int i = 0; i < key_count; i++) {
            if (keys[i] == key) return -1;
        }
        if (key_count >= POWER_DIRECT_PROB_STACK) return -1;
        keys[key_count++] = key;
    }
    return key_count;
}

__device__ bool PowerDirectExprEval::isSafe(int expr_id) const {
    const int var_count = uniqueVarCount(expr_id);
    const int requested_max_vars = g_power_direct_expr_max_vars < 0 ? 0 : g_power_direct_expr_max_vars;
    const int max_vars = requested_max_vars > POWER_DIRECT_PROB_STACK
        ? POWER_DIRECT_PROB_STACK
        : requested_max_vars;
    return var_count >= 0 && var_count <= max_vars;
}

__device__ bool PowerDirectExprEval::duty(int expr_id, float& out_duty) const {
    return dutyPoly(expr_id, out_duty);
}

__device__ bool PowerDirectExprEval::dutyPoly(int expr_id, float& out_duty) const {
    if (!view->pin_duty || !isSafe(expr_id)) {
        return false;
    }
    const int start = view->expr_start[expr_id];
    const int count = view->expr_count[expr_id];
    float stack[POWER_DIRECT_PROB_STACK];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = view->ops[start + k];
        switch (op.op) {
            case 0: {
                const int pin = view->resolvePinArg(op.arg);
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                if (pin < 0 && op.arg > -2) return false;
                stack[sp++] = pin < 0 ? 0.0f : PowerActivityOps::clampActivityDuty(view->pin_duty[pin]);
                break;
            }
            case 1:
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                stack[sp++] = 0.0f;
                break;
            case 2:
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                stack[sp++] = 1.0f;
                break;
            case 3:
                if (sp < 1) return false;
                stack[sp - 1] = 1.0f - stack[sp - 1];
                break;
            case 4: {
                if (sp < 2) return false;
                const float b = stack[--sp];
                const float a = stack[--sp];
                stack[sp++] = a * b;
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const float b = stack[--sp];
                const float a = stack[--sp];
                stack[sp++] = a + b - a * b;
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const float b = stack[--sp];
                const float a = stack[--sp];
                stack[sp++] = a * (1.0f - b) + (1.0f - a) * b;
                break;
            }
            case 7:
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                stack[sp++] = 0.0f;
                break;
            default:
                return false;
        }
    }
    if (sp != 1 || !isfinite(stack[0])) return false;
    out_duty = fminf(fmaxf(stack[0], 0.0f), 1.0f);
    return true;
}

__device__ bool PowerDirectExprEval::activity(int expr_id,
                                              float& out_density,
                                              float& out_duty) const {
    return activityPoly(expr_id, out_density, out_duty);
}

__device__ bool PowerDirectExprEval::activityPoly(int expr_id,
                                                  float& out_density,
                                                  float& out_duty) const {
    if (!view->pin_density || !view->pin_duty || !isSafe(expr_id)) {
        return false;
    }
    const int start = view->expr_start[expr_id];
    const int count = view->expr_count[expr_id];
    PowerDirectActivityValue stack[POWER_DIRECT_PROB_STACK];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = view->ops[start + k];
        switch (op.op) {
            case 0: {
                const int pin = view->resolvePinArg(op.arg);
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                if (pin < 0 && op.arg > -2) return false;
                const float value_duty =
                    pin < 0 ? 0.0f : PowerActivityOps::clampActivityDuty(view->pin_duty[pin]);
                const float value_density =
                    (pin >= 0 && op.zero_density == 0) ? view->pin_density[pin] : 0.0f;
                stack[sp++] = PowerDirectActivityValue(value_density, value_duty);
                break;
            }
            case 1:
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                stack[sp++] = PowerDirectActivityValue{0.0f, 0.0f};
                break;
            case 2:
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                stack[sp++] = PowerDirectActivityValue{0.0f, 1.0f};
                break;
            case 3:
                if (sp < 1) return false;
                stack[sp - 1] = PowerDirectActivityValue(stack[sp - 1].density,
                                                          1.0f - stack[sp - 1].duty);
                break;
            case 4: {
                if (sp < 2) return false;
                const PowerDirectActivityValue b = stack[--sp];
                const PowerDirectActivityValue a = stack[--sp];
                stack[sp++] = PowerDirectActivityValue(
                    fmaf(a.density, b.duty, b.density * a.duty),
                    a.duty * b.duty);
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const PowerDirectActivityValue b = stack[--sp];
                const PowerDirectActivityValue a = stack[--sp];
                stack[sp++] = PowerDirectActivityValue(
                    fmaf(b.density, 1.0f - a.duty, a.density * (1.0f - b.duty)),
                    fmaf(b.duty, 1.0f - a.duty, a.duty));
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const PowerDirectActivityValue b = stack[--sp];
                const PowerDirectActivityValue a = stack[--sp];
                stack[sp++] = PowerDirectActivityValue(
                    a.density + b.density,
                    fmaf(a.duty, 1.0f - 2.0f * b.duty, b.duty));
                break;
            }
            case 7:
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                stack[sp++] = PowerDirectActivityValue{0.0f, 0.0f};
                break;
            default:
                return false;
        }
    }
    if (sp != 1 || !isfinite(stack[0].density) || !isfinite(stack[0].duty)) return false;
    out_density = fmaxf(stack[0].density, 0.0f);
    out_duty = fminf(fmaxf(stack[0].duty, 0.0f), 1.0f);
    return true;
}

__device__ bool PowerDirectExprEval::diffDuty(int expr_id,
                                              int diff_pin,
                                              float& out_duty) const {
    return diffDutyPoly(expr_id, diff_pin, out_duty);
}

__device__ bool PowerDirectExprEval::diffDutyPoly(int expr_id,
                                                  int diff_pin,
                                                  float& out_duty) const {
    if (!view->pin_duty || diff_pin < 0 || !isSafe(expr_id)) {
        return false;
    }
    const int start = view->expr_start[expr_id];
    const int count = view->expr_count[expr_id];
    PowerDirectProbValue stack[POWER_DIRECT_PROB_STACK];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = view->ops[start + k];
        switch (op.op) {
            case 0: {
                const int pin = view->resolvePinArg(op.arg);
                if (sp >= POWER_DIRECT_PROB_STACK || pin < 0) return false;
                stack[sp++] = PowerDirectProbValue(
                    PowerActivityOps::clampActivityDuty(view->pin_duty[pin]),
                    pin == diff_pin ? 1.0f : 0.0f,
                    pin == diff_pin ? 1 : 0);
                break;
            }
            case 1:
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                stack[sp++] = PowerDirectProbValue{0.0f, 0.0f, 0};
                break;
            case 2:
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                stack[sp++] = PowerDirectProbValue{1.0f, 0.0f, 0};
                break;
            case 3:
                if (sp < 1) return false;
                stack[sp - 1] = PowerDirectProbValue(1.0f - stack[sp - 1].duty,
                                                      stack[sp - 1].diff,
                                                      stack[sp - 1].has_diff);
                break;
            case 4: {
                if (sp < 2) return false;
                const PowerDirectProbValue b = stack[--sp];
                const PowerDirectProbValue a = stack[--sp];
                if (a.has_diff && b.has_diff) return false;
                stack[sp++] = PowerDirectProbValue(a.duty * b.duty,
                                                   a.diff * b.duty + b.diff * a.duty,
                                                   a.has_diff || b.has_diff);
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const PowerDirectProbValue b = stack[--sp];
                const PowerDirectProbValue a = stack[--sp];
                if (a.has_diff && b.has_diff) return false;
                stack[sp++] = PowerDirectProbValue(
                    a.duty + b.duty - a.duty * b.duty,
                    a.diff * (1.0f - b.duty) + b.diff * (1.0f - a.duty),
                    a.has_diff || b.has_diff);
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const PowerDirectProbValue b = stack[--sp];
                const PowerDirectProbValue a = stack[--sp];
                if (a.has_diff && b.has_diff) return false;
                stack[sp++] = PowerDirectProbValue(
                    a.duty * (1.0f - b.duty) + (1.0f - a.duty) * b.duty,
                    a.diff + b.diff,
                    a.has_diff || b.has_diff);
                break;
            }
            case 7:
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                stack[sp++] = PowerDirectProbValue{0.0f, 0.0f, 0};
                break;
            default:
                return false;
        }
    }
    if (sp != 1 || !isfinite(stack[0].diff)) return false;
    out_duty = fminf(fmaxf(stack[0].diff, 0.0f), 1.0f);
    return true;
}

__device__ __noinline__ bool PowerBddExprEval::activity(int expr_id,
                                                        float& out_density,
                                                        float& out_duty,
                                                        int& out_var_count) {
    if (!buildExpr(expr_id)) {
        return false;
    }
    out_duty = ctx.evalDuty(root);
    out_density = 0.0f;
    out_var_count = ctx.var_count;

    int order[POWER_BDD_MAX_VARS];
    for (int i = 0; i < ctx.var_count; i++) order[i] = i;
    for (int i = 1; i < ctx.var_count; i++) {
        const int item = order[i];
        int j = i - 1;
        while (j >= 0 && ctx.var_keys[order[j]] > ctx.var_keys[item]) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = item;
    }
    for (int idx = 0; idx < ctx.var_count; idx++) {
        const int var = order[idx];
        if (!ctx.var_has_pin[var]) continue;
        const int low = ctx.restrict(root, var, false);
        const int high = ctx.restrict(root, var, true);
        const int diff = ctx.apply(2, low, high);
        const float diff_duty = ctx.evalDuty(diff);
        out_density += ctx.var_densities[var] * diff_duty;
    }
    return isfinite(out_density) && isfinite(out_duty);
}

__device__ __noinline__ float PowerBddExprEval::diffDuty(int expr_id, int diff_pin) {
    if (!buildExpr(expr_id)) {
        return 0.0f;
    }
    int diff_var = -1;
    for (int var = 0; var < ctx.var_count; var++) {
        if (ctx.var_pins[var] == diff_pin) {
            diff_var = var;
            break;
        }
    }
    if (diff_var < 0) return 0.0f;
    const int low = ctx.restrict(root, diff_var, false);
    const int high = ctx.restrict(root, diff_var, true);
    const int diff = ctx.apply(2, low, high);
    return ctx.evalDuty(diff);
}

__device__ __noinline__ bool PowerBddExprEval::activityFallback(const PowerExprEval& view,
                                                                int expr_id,
                                                                float& out_density,
                                                                float& out_duty,
                                                                int& out_var_count) {
    PowerBddExprEval eval(view);
    return eval.activity(expr_id, out_density, out_duty, out_var_count);
}

__device__ __noinline__ float PowerBddExprEval::diffDutyFallback(const PowerExprEval& view,
                                                                 int expr_id,
                                                                 int diff_pin) {
    PowerBddExprEval eval(view);
    return eval.diffDuty(expr_id, diff_pin);
}

}  // namespace

__device__ int PowerExprEval::resolvePinArg(int arg) const {
    if (arg >= 0) return arg;
    if (arg == -1 || !node_port_pin_start || !node_port_pin_list || node_id < 0) return -1;
    const int port_id = -2 - arg;
    const int start = node_port_pin_start[node_id];
    const int end = node_port_pin_start[node_id + 1];
    if (port_id < 0 || start + port_id < start || start + port_id >= end) return -1;
    return node_port_pin_list[start + port_id];
}

__device__ bool PowerExprEval::activity(int expr_id,
                                        float& out_density,
                                        float& out_duty) const {
    const bool direct_allowed = !g_power_disable_direct_expr;
    float direct_density = 0.0f;
    float direct_duty = 0.0f;
    bool direct_ok = false;
    if (direct_allowed) {
        const PowerDirectExprEval direct_eval(*this);
        direct_ok = direct_eval.activity(expr_id, direct_density, direct_duty);
    }
    if (direct_ok && !g_power_check_direct_expr) {
        out_density = direct_density;
        out_duty = direct_duty;
        return true;
    }
    float bdd_density = 0.0f;
    float bdd_duty = 0.0f;
    int bdd_var_count = 0;
    if (!PowerBddExprEval::activityFallback(*this, expr_id, bdd_density,
                                            bdd_duty, bdd_var_count)) {
        return false;
    }
    if (direct_ok && g_power_check_direct_expr) {
        const float density_scale = fmaxf(fmaxf(fabsf(bdd_density), fabsf(direct_density)), 1.0f);
        const float density_rel = fabsf(bdd_density - direct_density) / density_scale;
        const float duty_abs = fabsf(bdd_duty - direct_duty);
        if (density_rel > g_power_direct_expr_density_rel_tol ||
            duty_abs > g_power_direct_expr_duty_abs_tol) {
            const int pos = atomicAdd(&g_power_direct_expr_mismatch_count, 1);
            if (pos < 64) {
                printf("[power_direct_expr_mismatch] kind=activity expr=%d node=%d direct_density=%.9e bdd_density=%.9e direct_duty=%.9e bdd_duty=%.9e density_rel=%.9e duty_abs=%.9e vars=%d\n",
                       expr_id, node_id, direct_density, bdd_density,
                       direct_duty, bdd_duty, density_rel, duty_abs,
                       bdd_var_count);
            }
        }
        out_density = direct_density;
        out_duty = direct_duty;
    } else {
        out_density = bdd_density;
        out_duty = bdd_duty;
    }
    return isfinite(out_density) && isfinite(out_duty);
}

__device__ bool PowerExprEval::hasKnownActivityInput(int expr_id, const uint8_t* origin) const {
    if (expr_id < 0) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return false;
    bool has_pin_arg = false;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        if (op.op != 0) continue;
        const int pin = resolvePinArg(op.arg);
        if (pin < 0) continue;
        has_pin_arg = true;
        if (!origin || origin[pin] != 0 || op.zero_density != 0) return true;
    }
    return !has_pin_arg;
}

__device__ float PowerExprEval::duty(int expr_id) const {
    float direct_duty = 0.0f;
    if (!g_power_disable_direct_expr) {
        const PowerDirectExprEval direct_eval(*this);
        if (direct_eval.duty(expr_id, direct_duty)) {
            return direct_duty;
        }
    }
    float density = 0.0f;
    float activity_duty = 0.0f;
    if (!activity(expr_id, density, activity_duty)) {
        return 0.0f;
    }
    return fminf(fmaxf(activity_duty, 0.0f), 1.0f);
}

__device__ float PowerExprEval::diffDuty(int expr_id, int diff_pin) const {
    if (expr_id < 0 || diff_pin < 0) return 0.0f;
    float direct_duty = 0.0f;
    if (!g_power_disable_direct_expr) {
        const PowerDirectExprEval direct_eval(*this);
        if (direct_eval.diffDuty(expr_id, diff_pin, direct_duty)) {
            return direct_duty;
        }
    }
    const PowerExprEval bdd_view = withDensity(nullptr);
    return PowerBddExprEval::diffDutyFallback(bdd_view, expr_id, diff_pin);
}

}  // namespace gt
