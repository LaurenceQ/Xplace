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

__device__ float power_percent_change(float value, float prev) {
    if (prev == 0.0f) return value == 0.0f ? 0.0f : 1.0f;
    return fabsf(value - prev) / fabsf(prev);
}

__device__ float power_clamp_activity_duty(float duty) {
    float u = fminf(fmaxf(duty, 0.0f), 1.0f);
    const float eps = fmaxf(g_power_min_activity_duty, 0.0f);
    if (eps > 0.0f) {
        if (u < eps) u = 0.0f;
        else if ((1.0f - u) < eps) u = 1.0f;
    }
    return u;
}

__device__ bool power_should_mark_pending_seq(float density) {
    return density >= fmaxf(g_power_seq_pending_min_density, 0.0f);
}

__device__ float power_max_activity_density_from_slew(int pin,
                                                      const PowerActivityCudaModel* model) {
    if (g_power_disable_activity_slew_cap) return 3.4028234663852886e38f;
    const float* pinSlew = model->graph.pinSlew;
    const float* powerClockSlews = model->graph.power_clock_slews;
    const float time_unit = model->config.time_unit;
    if ((!pinSlew && !powerClockSlews) || pin < 0 || !(time_unit > 0.0f))
        return 3.4028234663852886e38f;
    float min_rf_slew = 3.4028234663852886e38f;
    #pragma unroll
    for (int base = 0; base < NUM_ATTR; base += 2) {
        float rise = pinSlew ? pinSlew[pin * NUM_ATTR + base] : nanf("");
        float fall = pinSlew ? pinSlew[pin * NUM_ATTR + base + 1] : nanf("");
        const bool use_clock_slew_override =
            model->graph.is_seq_clock_input_pin && model->graph.is_seq_clock_input_pin[pin];
        if (use_clock_slew_override && powerClockSlews) {
            const float clock_rise = powerClockSlews[pin * NUM_ATTR + base];
            const float clock_fall = powerClockSlews[pin * NUM_ATTR + base + 1];
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

__device__ bool power_seq_density_exceeds_clock_limit(float in_density, float clk_density) {
    const float limit = clk_density * 0.5f;
    return in_density > limit * (1.0f + fmaxf(g_power_seq_clock_limit_rel_tol, 0.0f));
}

__device__ bool power_set_activity(int pin,
                                   float new_density,
                                   float new_duty,
                                   int new_origin,
                                   bool force,
                                   const PowerActivityCudaModel* model,
                                   PowerActivityScratchView* scratch) {
    float* density = scratch->density;
    float* duty = scratch->duty;
    int* origin = scratch->origin;
    if (!force && origin[pin] == 2 && !g_power_allow_clock_activity_override) return false;
    const float prev_density = density[pin];
    const float prev_duty = duty[pin];
    const int prev_origin = origin[pin];
    const float max_density = force
        ? g_power_activity_clock_density_cap
        : fminf(power_max_activity_density_from_slew(pin, model),
                g_power_activity_clock_density_cap);
    float d = fminf(fmaxf(new_density, 0.0f), max_density);
    if (fabsf(d) < g_power_min_activity_density) d = 0.0f;
    const float u = power_clamp_activity_duty(new_duty);
    const bool value_changed = power_percent_change(d, prev_density) > 0.01f
        || power_percent_change(u, prev_duty) > 0.01f;
    const bool changed = value_changed || prev_origin != new_origin;
    density[pin] = d;
    duty[pin] = u;
    origin[pin] = new_origin;
    return changed;
}

__device__ void power_enqueue_adjacent(int pin,
                                       const PowerActivityCudaModel* model,
                                       PowerActivityScratchView* scratch) {
    const auto& graph = model->graph;
    const uint8_t* is_load_pin = graph.is_load_pin;
    const int* pin2net_map = graph.pin2net_map;
    const int* net_driver_pin = graph.net_driver_pin;
    const int* flat_net2pin_start_map = graph.flat_net2pin_start_map;
    const int* flat_net2pin_map = graph.flat_net2pin_map;
    const index_type* pin_forward_arc_list_end = graph.pin_forward_arc_list_end;
    const index_type* pin_forward_arc_list = graph.pin_forward_arc_list;
    const index_type* timing_arc_to_pin_id = graph.timing_arc_to_pin_id;
    const int* arc_types = graph.arc_types;
    const int* arc_id2test_id = graph.arc_id2test_id;
    const uint8_t* is_seq_output_pin = graph.is_seq_output_pin;
    const int* pin_power_level = graph.pin_power_level;
    uint8_t* active_level = scratch->active_level;
    const int num_power_levels = scratch->num_power_levels;
    int* active = scratch->active;
    if (is_load_pin && pin2net_map && net_driver_pin && flat_net2pin_start_map && flat_net2pin_map) {
        const int net = pin2net_map[pin];
        if (net >= 0 && net_driver_pin[net] == pin) {
            const int start = flat_net2pin_start_map[net];
            const int end = flat_net2pin_start_map[net + 1];
            for (int pos = start; pos < end; ++pos) {
                const int sink = flat_net2pin_map[pos];
                if (sink < 0 || sink == pin || !is_load_pin[sink]) continue;
                atomicExch(&active[sink], 1);
                if (pin_power_level && active_level) {
                    const int level = pin_power_level[sink];
                    if (level >= 0 && level < num_power_levels) active_level[level] = 1;
                }
            }
        }
    }
    for (index_type i = pin_forward_arc_list_end[pin]; i < pin_forward_arc_list_end[pin + 1]; i++) {
        const int arc = pin_forward_arc_list[i];
        if (arc_id2test_id && arc_id2test_id[arc] != -1) continue;
        const int to_pin = timing_arc_to_pin_id[arc];
        if (to_pin < 0) continue;
        if (arc_types && arc_types[arc] == 1 && is_seq_output_pin && is_seq_output_pin[to_pin]) continue;
        atomicExch(&active[to_pin], 1);
        if (pin_power_level && active_level) {
            const int level = pin_power_level[to_pin];
            if (level >= 0 && level < num_power_levels) active_level[level] = 1;
        }
    }
}

__device__ void power_activate_pin(int pin,
                                   const PowerActivityCudaModel* model,
                                   PowerActivityScratchView* scratch) {
    const int* pin_power_level = model->graph.pin_power_level;
    uint8_t* active_level = scratch->active_level;
    const int num_power_levels = scratch->num_power_levels;
    int* active = scratch->active;
    if (pin < 0 || !active) return;
    atomicExch(&active[pin], 1);
    if (pin_power_level && active_level) {
        const int level = pin_power_level[pin];
        if (level >= 0 && level < num_power_levels) active_level[level] = 1;
    }
}

__device__ bool power_set_clock_gate_output(int pin,
                                            const PowerActivityCudaModel* model,
                                            PowerActivityScratchView* scratch) {
    const int* clock_gate_clock_for_out = model->graph.clock_gate_clock_for_out;
    const int* clock_gate_enable_for_out = model->graph.clock_gate_enable_for_out;
    float* density = scratch->density;
    float* duty = scratch->duty;
    int* origin = scratch->origin;
    if (!clock_gate_clock_for_out || !clock_gate_enable_for_out) return false;
    const int clk = clock_gate_clock_for_out[pin];
    const int en = clock_gate_enable_for_out[pin];
    if (clk < 0 || en < 0) return false;
    if (origin && origin[clk] == 0 && origin[en] == 0) return false;
    const float out_density = density[clk] * duty[en] + density[en] * duty[clk];
    const float out_duty = duty[clk] * duty[en];
    return power_set_activity(pin, out_density, out_duty, 3, false, model, scratch);
}

__device__ void power_enqueue_clock_gate_output(int pin,
                                                const PowerActivityCudaModel* model,
                                                PowerActivityScratchView* scratch) {
    const int* clock_gate_out_for_input = model->graph.clock_gate_out_for_input;
    const int* pin_power_level = model->graph.pin_power_level;
    uint8_t* active_level = scratch->active_level;
    const int num_power_levels = scratch->num_power_levels;
    int* active = scratch->active;
    if (!clock_gate_out_for_input) return;
    const int out_pin = clock_gate_out_for_input[pin];
    if (out_pin < 0) return;
    atomicExch(&active[out_pin], 1);
    if (pin_power_level && active_level) {
        const int level = pin_power_level[out_pin];
        if (level >= 0 && level < num_power_levels) active_level[level] = 1;
    }
}

__device__ bool power_eval_expr_bool(int expr_id,
                                     uint64_t bits,
                                     int force_var,
                                     int force_val,
                                     const int* var_pins,
                                     int var_count,
                                     const GpuPowerExprOpHost* ops,
                                     const int* expr_start,
                                     const int* expr_count,
                                     int8_t& value) {
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
};

struct PowerBddApplyCacheCuda {
    int op = -1;
    int left = 0;
    int right = 0;
    int result = 0;
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
};

__device__ int power_bdd_edge_id(int edge) { return edge >> 1; }
__device__ bool power_bdd_edge_inv(int edge) { return (edge & 1) != 0; }
__device__ int power_bdd_not(int edge) { return edge ^ 1; }

__device__ int power_bdd_make_node(PowerBddContextCuda& ctx, int var, int low, int high) {
    if (low == high) return low;
    bool result_inv = false;
    // Mirror CUDD's complemented-edge normalization: the then/high edge is
    // stored regular and a complement is moved onto the returned edge.
    if (power_bdd_edge_inv(high)) {
        low = power_bdd_not(low);
        high = power_bdd_not(high);
        result_inv = true;
    }
    for (int i = 0; i < ctx.node_count; i++) {
        const auto& node = ctx.nodes[i];
        if (node.var == var && node.low == low && node.high == high) {
            const int edge = (i + 1) << 1;
            return result_inv ? power_bdd_not(edge) : edge;
        }
    }
    if (ctx.node_count >= POWER_BDD_MAX_NODES) {
        ctx.ok = false;
        return 1;
    }
    const int id = ++ctx.node_count;
    ctx.nodes[id - 1] = PowerBddNodeCuda{var, low, high};
    const int edge = id << 1;
    return result_inv ? power_bdd_not(edge) : edge;
}

__device__ int power_bdd_top_var(const PowerBddContextCuda& ctx, int edge) {
    const int id = power_bdd_edge_id(edge);
    return id == 0 ? 0x3fffffff : ctx.nodes[id - 1].var;
}

__device__ int power_bdd_cof_top(const PowerBddContextCuda& ctx, int edge, int var, bool high_child) {
    const int id = power_bdd_edge_id(edge);
    if (id == 0 || ctx.nodes[id - 1].var != var) return edge;
    const int child = high_child ? ctx.nodes[id - 1].high : ctx.nodes[id - 1].low;
    return power_bdd_edge_inv(edge) ? power_bdd_not(child) : child;
}

__device__ int power_bdd_apply(PowerBddContextCuda& ctx, int op, int left, int right) {
    if (op >= 0 && op <= 2 && right < left) {
        const int tmp = left;
        left = right;
        right = tmp;
    }
    for (int i = 0; i < ctx.apply_count; i++) {
        const auto& cache = ctx.apply_cache[i];
        if (cache.op == op && cache.left == left && cache.right == right) return cache.result;
    }

    int result = 1;
    const int left_id = power_bdd_edge_id(left);
    const int right_id = power_bdd_edge_id(right);
    if (left_id == 0 && right_id == 0) {
        const bool left_value = !power_bdd_edge_inv(left);
        const bool right_value = !power_bdd_edge_inv(right);
        bool value = false;
        if (op == 0) value = left_value && right_value;
        else if (op == 1) value = left_value || right_value;
        else value = left_value != right_value;
        result = value ? 0 : 1;
    } else {
        const int left_top = power_bdd_top_var(ctx, left);
        const int right_top = power_bdd_top_var(ctx, right);
        const int var = left_top < right_top ? left_top : right_top;
        const int low = power_bdd_apply(ctx, op,
                                        power_bdd_cof_top(ctx, left, var, false),
                                        power_bdd_cof_top(ctx, right, var, false));
        const int high = power_bdd_apply(ctx, op,
                                         power_bdd_cof_top(ctx, left, var, true),
                                         power_bdd_cof_top(ctx, right, var, true));
        result = power_bdd_make_node(ctx, var, low, high);
    }

    if (ctx.apply_count < POWER_BDD_MAX_APPLY_CACHE) {
        ctx.apply_cache[ctx.apply_count++] = PowerBddApplyCacheCuda{op, left, right, result};
    }
    return result;
}

__device__ int power_bdd_restrict(PowerBddContextCuda& ctx, int edge, int target_var, bool high_child) {
    const int id = power_bdd_edge_id(edge);
    if (id == 0) return edge;
    const auto node = ctx.nodes[id - 1];
    if (node.var > target_var) return edge;
    int result = edge;
    if (node.var == target_var) {
        result = high_child ? node.high : node.low;
    } else {
        const int low = power_bdd_restrict(ctx, node.low, target_var, high_child);
        const int high = power_bdd_restrict(ctx, node.high, target_var, high_child);
        result = power_bdd_make_node(ctx, node.var, low, high);
    }
    return power_bdd_edge_inv(edge) ? power_bdd_not(result) : result;
}

__device__ float power_bdd_eval_duty(const PowerBddContextCuda& ctx, int edge) {
    const int id = power_bdd_edge_id(edge);
    if (id == 0) return power_bdd_edge_inv(edge) ? 0.0f : 1.0f;
    const auto node = ctx.nodes[id - 1];
    if (node.var >= 0 && node.var < ctx.var_count && !ctx.var_has_pin[node.var])
        return 0.0f;
    const float duty0 = power_bdd_eval_duty(ctx, node.low);
    const float duty1 = power_bdd_eval_duty(ctx, node.high);
    const float var_duty = ctx.var_duties[node.var];
    const double result_d =
        static_cast<double>(duty0) * (1.0 - static_cast<double>(var_duty)) +
        static_cast<double>(duty1) * static_cast<double>(var_duty);
    float result = static_cast<float>(result_d);
    if (power_bdd_edge_inv(edge)) result = 1.0f - result;
    return fminf(fmaxf(result, 0.0f), 1.0f);
}

__device__ int power_bdd_ensure_var(PowerBddContextCuda& ctx,
                                    int var_key,
                                    int pin,
                                    const float* pin_density,
                                    const float* pin_duty,
                                    bool zero_density) {
    for (int i = 0; i < ctx.var_count; i++) {
        if (ctx.var_keys[i] == var_key && zero_density)
            ctx.var_densities[i] = 0.0f;
        if (ctx.var_keys[i] == var_key) return i;
    }
    if (ctx.var_count >= POWER_BDD_MAX_VARS) {
        ctx.ok = false;
        return -1;
    }
    const int var = ctx.var_count++;
    ctx.var_keys[var] = var_key;
    ctx.var_pins[var] = pin;
    ctx.var_has_pin[var] = pin >= 0 ? 1 : 0;
    ctx.var_duties[var] = pin >= 0 ? power_clamp_activity_duty(pin_duty[pin]) : 0.0f;
    ctx.var_densities[var] = (pin >= 0 && !zero_density && pin_density) ? pin_density[pin] : 0.0f;
    return var;
}

__device__ int power_expr_resolve_pin_arg(int arg,
                                          const int* node_port_pin_start,
                                          const int* node_port_pin_list,
                                          int node_id) {
    if (arg >= 0) return arg;
    if (arg == -1 || !node_port_pin_start || !node_port_pin_list || node_id < 0) return -1;
    const int port_id = -2 - arg;
    const int start = node_port_pin_start[node_id];
    const int end = node_port_pin_start[node_id + 1];
    if (port_id < 0 || start + port_id < start || start + port_id >= end) return -1;
    return node_port_pin_list[start + port_id];
}

__device__ bool power_bdd_build_expr(int expr_id,
                                     const GpuPowerExprOpHost* ops,
                                     const int* expr_start,
                                     const int* expr_count,
                                     const float* pin_density,
                                     const float* pin_duty,
                                     PowerBddContextCuda& ctx,
                                     int& root,
                                     const int* node_port_pin_start,
                                     const int* node_port_pin_list,
                                     int node_id) {
    if (expr_id < 0) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return false;
    int pre_keys[POWER_BDD_MAX_VARS];
    int pre_pins[POWER_BDD_MAX_VARS];
    uint8_t pre_zero_density[POWER_BDD_MAX_VARS];
    int pre_count = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        int pin = -1;
        int var_key = -1;
        bool zero_density = op.zero_density != 0;
        if (op.op == 0) {
            pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                             node_port_pin_list, node_id);
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
        int pos = -1;
        for (int i = 0; i < pre_count; i++) {
            if (pre_keys[i] == var_key) {
                pos = i;
                break;
            }
        }
        if (pos < 0) {
            if (pre_count >= POWER_BDD_MAX_VARS) return false;
            pos = pre_count++;
            pre_keys[pos] = var_key;
            pre_pins[pos] = pin;
            pre_zero_density[pos] = zero_density ? 1 : 0;
        } else if (zero_density) {
            pre_zero_density[pos] = 1;
        }
    }
    for (int i = 1; i < pre_count; i++) {
        const int key = pre_keys[i];
        const int pin = pre_pins[i];
        const uint8_t zero_density = pre_zero_density[i];
        int j = i - 1;
        while (j >= 0 && pre_keys[j] > key) {
            pre_keys[j + 1] = pre_keys[j];
            pre_pins[j + 1] = pre_pins[j];
            pre_zero_density[j + 1] = pre_zero_density[j];
            j--;
        }
        pre_keys[j + 1] = key;
        pre_pins[j + 1] = pin;
        pre_zero_density[j + 1] = zero_density;
    }
    for (int i = 0; i < pre_count; i++) {
        if (power_bdd_ensure_var(ctx, pre_keys[i], pre_pins[i], pin_density, pin_duty,
                                 pre_zero_density[i] != 0) < 0 || !ctx.ok)
            return false;
    }
    int stack[128];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        switch (op.op) {
            case 0: {
                const int pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                                           node_port_pin_list, node_id);
                if (sp >= 128) return false;
                if (pin < 0 && op.arg > -2) return false;
                if (pin < 0) {
                    stack[sp++] = 1;
                    break;
                }
                const int var_key = op.var_key >= 0 ? op.var_key : pin;
                const int var = power_bdd_ensure_var(ctx, var_key, pin, pin_density, pin_duty,
                                                     op.zero_density != 0);
                if (var < 0 || !ctx.ok) return false;
                stack[sp++] = power_bdd_make_node(ctx, var, 1, 0);
                break;
            }
            case 7: {
                if (sp >= 128) return false;
                const int var_key = op.var_key >= 0 ? op.var_key : -1;
                const int var = power_bdd_ensure_var(ctx, var_key, -1, pin_density, pin_duty,
                                                     true);
                if (var < 0 || !ctx.ok) return false;
                stack[sp++] = power_bdd_make_node(ctx, var, 1, 0);
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
                stack[sp - 1] = power_bdd_not(stack[sp - 1]);
                break;
            }
            case 4: {
                if (sp < 2) return false;
                const int right = stack[--sp];
                const int left = stack[--sp];
                stack[sp++] = power_bdd_apply(ctx, 0, left, right);
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const int right = stack[--sp];
                const int left = stack[--sp];
                stack[sp++] = power_bdd_apply(ctx, 1, left, right);
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const int right = stack[--sp];
                const int left = stack[--sp];
                stack[sp++] = power_bdd_apply(ctx, 2, left, right);
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
};

struct PowerDirectActivityValue {
    float density = 0.0f;
    float duty = 0.0f;
};

__device__ int power_direct_expr_unique_var_count(int expr_id,
                                                  const GpuPowerExprOpHost* ops,
                                                  const int* expr_start,
                                                  const int* expr_count,
                                                  const int* node_port_pin_start,
                                                  const int* node_port_pin_list,
                                                  int node_id) {
    if (expr_id < 0) return -1;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return -1;
    int keys[POWER_DIRECT_PROB_STACK];
    int key_count = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        int key = -1;
        if (op.op == 0) {
            const int pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                                       node_port_pin_list, node_id);
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

__device__ bool power_direct_expr_is_safe(int expr_id,
                                          const GpuPowerExprOpHost* ops,
                                          const int* expr_start,
                                          const int* expr_count,
                                          const int* node_port_pin_start,
                                          const int* node_port_pin_list,
                                          int node_id) {
    const int var_count = power_direct_expr_unique_var_count(expr_id, ops, expr_start,
                                                            expr_count, node_port_pin_start,
                                                            node_port_pin_list, node_id);
    const int requested_max_vars = g_power_direct_expr_max_vars < 0 ? 0 : g_power_direct_expr_max_vars;
    const int max_vars = requested_max_vars > POWER_DIRECT_PROB_STACK
        ? POWER_DIRECT_PROB_STACK
        : requested_max_vars;
    return var_count >= 0 && var_count <= max_vars;
}

struct PowerDirectVarValue {
    int key = -1;
    int pin = -1;
    float duty = 0.0f;
    float density = 0.0f;
};

__device__ bool power_direct_expr_collect_vars(int expr_id,
                                               const GpuPowerExprOpHost* ops,
                                               const int* expr_start,
                                               const int* expr_count,
                                               const float* pin_density,
                                               const float* pin_duty,
                                               const int* node_port_pin_start,
                                               const int* node_port_pin_list,
                                               int node_id,
                                               PowerDirectVarValue* vars,
                                               int& var_count) {
    if (expr_id < 0 || !pin_duty) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return false;
    var_count = 0;
    const int max_vars = g_power_direct_expr_max_vars < 0 ? 0 : g_power_direct_expr_max_vars;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        if (op.op == 7) return false;
        if (op.op != 0) continue;
        const int pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                                   node_port_pin_list, node_id);
        if (pin < 0 && op.arg > -2) return false;
        if (pin < 0) continue;
        const int key = op.var_key >= 0 ? op.var_key : pin;
        if (key < 0) return false;
        int pos = -1;
        for (int i = 0; i < var_count; i++) {
            if (vars[i].key == key) {
                pos = i;
                break;
            }
        }
        if (pos < 0) {
            if (var_count >= max_vars || var_count >= POWER_DIRECT_PROB_STACK)
                return false;
            pos = var_count;
            while (pos > 0 && vars[pos - 1].key > key) {
                vars[pos] = vars[pos - 1];
                pos--;
            }
            var_count++;
            vars[pos].key = key;
            vars[pos].pin = pin;
            vars[pos].duty = power_clamp_activity_duty(pin_duty[pin]);
            vars[pos].density = (pin_density && op.zero_density == 0) ? pin_density[pin] : 0.0f;
        } else if (op.zero_density != 0) {
            vars[pos].density = 0.0f;
        }
    }
    return true;
}

__device__ int power_direct_expr_find_var_key(const PowerDirectVarValue* vars,
                                              int var_count,
                                              int key) {
    for (int i = 0; i < var_count; i++) {
        if (vars[i].key == key) return i;
    }
    return -1;
}

__device__ bool power_direct_expr_truth_table(int expr_id,
                                              const PowerDirectVarValue* vars,
                                              int var_count,
                                              const GpuPowerExprOpHost* ops,
                                              const int* expr_start,
                                              const int* expr_count,
                                              const int* node_port_pin_start,
                                              const int* node_port_pin_list,
                                              int node_id,
                                              uint8_t& truth) {
    if (expr_id < 0 || var_count < 0 || var_count > 2) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return false;
    const int states = 1 << var_count;
    const uint8_t all_bits = static_cast<uint8_t>((1 << states) - 1);
    uint8_t stack[POWER_DIRECT_PROB_STACK];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        switch (op.op) {
            case 0: {
                const int pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                                           node_port_pin_list, node_id);
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                if (pin < 0 && op.arg > -2) return false;
                if (pin < 0) {
                    stack[sp++] = 0;
                    break;
                }
                const int key = op.var_key >= 0 ? op.var_key : pin;
                const int var = power_direct_expr_find_var_key(vars, var_count, key);
                if (var < 0) return false;
                uint8_t bits = 0;
                for (int mask = 0; mask < states; mask++) {
                    if ((mask >> var) & 1) bits |= static_cast<uint8_t>(1 << mask);
                }
                stack[sp++] = bits;
                break;
            }
            case 1:
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                stack[sp++] = 0;
                break;
            case 2:
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                stack[sp++] = all_bits;
                break;
            case 3:
                if (sp < 1) return false;
                stack[sp - 1] = static_cast<uint8_t>((~stack[sp - 1]) & all_bits);
                break;
            case 4: {
                if (sp < 2) return false;
                const uint8_t b = stack[--sp];
                const uint8_t a = stack[--sp];
                stack[sp++] = static_cast<uint8_t>(a & b);
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const uint8_t b = stack[--sp];
                const uint8_t a = stack[--sp];
                stack[sp++] = static_cast<uint8_t>(a | b);
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const uint8_t b = stack[--sp];
                const uint8_t a = stack[--sp];
                stack[sp++] = static_cast<uint8_t>((a ^ b) & all_bits);
                break;
            }
            default:
                return false;
        }
    }
    if (sp != 1) return false;
    truth = static_cast<uint8_t>(stack[0] & all_bits);
    return true;
}

__device__ bool power_direct_expr_eval_mask(int expr_id,
                                            uint64_t bits,
                                            const PowerDirectVarValue* vars,
                                            int var_count,
                                            const GpuPowerExprOpHost* ops,
                                            const int* expr_start,
                                            const int* expr_count,
                                            const int* node_port_pin_start,
                                            const int* node_port_pin_list,
                                            int node_id,
                                            int8_t& value) {
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
                const int pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                                           node_port_pin_list, node_id);
                if (sp >= 128) return false;
                if (pin < 0 && op.arg > -2) return false;
                if (pin < 0) {
                    stack[sp++] = 0;
                    break;
                }
                const int key = op.var_key >= 0 ? op.var_key : pin;
                const int var = power_direct_expr_find_var_key(vars, var_count, key);
                if (var < 0) return false;
                stack[sp++] = static_cast<int8_t>((bits >> var) & 1ULL);
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
            case 3:
                if (sp < 1) return false;
                stack[sp - 1] = static_cast<int8_t>(!stack[sp - 1]);
                break;
            case 4: {
                if (sp < 2) return false;
                const int8_t b = stack[--sp];
                const int8_t a = stack[--sp];
                stack[sp++] = static_cast<int8_t>((a != 0) && (b != 0));
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const int8_t b = stack[--sp];
                const int8_t a = stack[--sp];
                stack[sp++] = static_cast<int8_t>((a != 0) || (b != 0));
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const int8_t b = stack[--sp];
                const int8_t a = stack[--sp];
                stack[sp++] = static_cast<int8_t>((a != 0) ^ (b != 0));
                break;
            }
            default:
                return false;
        }
    }
    if (sp != 1) return false;
    value = stack[0] ? 1 : 0;
    return true;
}

__device__ bool power_direct_expr_small_duty(int expr_id,
                                             const GpuPowerExprOpHost* ops,
                                             const int* expr_start,
                                             const int* expr_count,
                                             const float* pin_duty,
                                             const int* node_port_pin_start,
                                             const int* node_port_pin_list,
                                             int node_id,
                                             float& out_duty) {
    PowerDirectVarValue vars[2];
    int var_count = 0;
    if (!power_direct_expr_collect_vars(expr_id, ops, expr_start, expr_count,
                                        nullptr, pin_duty, node_port_pin_start,
                                        node_port_pin_list, node_id, vars, var_count)) {
        return false;
    }
    uint8_t truth = 0;
    if (!power_direct_expr_truth_table(expr_id, vars, var_count, ops, expr_start,
                                       expr_count, node_port_pin_start,
                                       node_port_pin_list, node_id, truth)) {
        return false;
    }
    const int states = 1 << var_count;
    double duty = 0.0;
    for (int mask = 0; mask < states; mask++) {
        double prob = 1.0;
        for (int var = 0; var < var_count; var++) {
            const double p = static_cast<double>(vars[var].duty);
            prob *= ((mask >> var) & 1) ? p : (1.0 - p);
        }
        if ((truth >> mask) & 1) duty += prob;
    }
    out_duty = fminf(fmaxf(static_cast<float>(duty), 0.0f), 1.0f);
    return true;
}

__device__ bool power_direct_expr_small_diff_duty(int expr_id,
                                                  int diff_var,
                                                  const PowerDirectVarValue* vars,
                                                  int var_count,
                                                  const GpuPowerExprOpHost* ops,
                                                  const int* expr_start,
                                                  const int* expr_count,
                                                  const int* node_port_pin_start,
                                                  const int* node_port_pin_list,
                                                  int node_id,
                                                  float& out_duty) {
    if (diff_var < 0 || diff_var >= var_count) return false;
    uint8_t truth = 0;
    if (!power_direct_expr_truth_table(expr_id, vars, var_count, ops, expr_start,
                                       expr_count, node_port_pin_start,
                                       node_port_pin_list, node_id, truth)) {
        return false;
    }
    const int other_states = 1 << (var_count - 1);
    double duty = 0.0;
    for (int mask = 0; mask < other_states; mask++) {
        int low_mask = 0;
        int high_mask = 0;
        int src_bit = 0;
        double prob = 1.0;
        for (int var = 0; var < var_count; var++) {
            if (var == diff_var) {
                high_mask |= (1 << var);
                continue;
            }
            const int bit = (mask >> src_bit) & 1;
            src_bit++;
            if (bit) {
                low_mask |= (1 << var);
                high_mask |= (1 << var);
            }
            const double p = static_cast<double>(vars[var].duty);
            prob *= bit ? p : (1.0 - p);
        }
        const int low_value = (truth >> low_mask) & 1;
        const int high_value = (truth >> high_mask) & 1;
        if (low_value != high_value) duty += prob;
    }
    out_duty = fminf(fmaxf(static_cast<float>(duty), 0.0f), 1.0f);
    return true;
}

__device__ bool power_direct_expr_small_activity(int expr_id,
                                                 const GpuPowerExprOpHost* ops,
                                                 const int* expr_start,
                                                 const int* expr_count,
                                                 const float* pin_density,
                                                 const float* pin_duty,
                                                 const int* node_port_pin_start,
                                                 const int* node_port_pin_list,
                                                 int node_id,
                                                 float& out_density,
                                                 float& out_duty) {
    PowerDirectVarValue vars[2];
    int var_count = 0;
    if (!power_direct_expr_collect_vars(expr_id, ops, expr_start, expr_count,
                                        pin_density, pin_duty, node_port_pin_start,
                                        node_port_pin_list, node_id, vars, var_count)) {
        return false;
    }
    if (!power_direct_expr_small_duty(expr_id, ops, expr_start, expr_count, pin_duty,
                                      node_port_pin_start, node_port_pin_list,
                                      node_id, out_duty)) {
        return false;
    }
    double density = 0.0;
    for (int var = 0; var < var_count; var++) {
        if (vars[var].density == 0.0f) continue;
        float diff_duty = 0.0f;
        if (!power_direct_expr_small_diff_duty(expr_id, var, vars, var_count, ops,
                                               expr_start, expr_count,
                                               node_port_pin_start, node_port_pin_list,
                                               node_id, diff_duty)) {
            return false;
        }
        density += static_cast<double>(vars[var].density) * static_cast<double>(diff_duty);
    }
    out_density = fmaxf(static_cast<float>(density), 0.0f);
    return isfinite(out_density) && isfinite(out_duty);
}

__device__ float power_direct_expr_lerp01(float low, float high, float p) {
    return fmaf(p, high - low, low);
}

__device__ bool power_direct_expr_small_activity_floatbdd(int expr_id,
                                                          const GpuPowerExprOpHost* ops,
                                                          const int* expr_start,
                                                          const int* expr_count,
                                                          const float* pin_density,
                                                          const float* pin_duty,
                                                          const int* node_port_pin_start,
                                                          const int* node_port_pin_list,
                                                          int node_id,
                                                          float& out_density,
                                                          float& out_duty) {
    PowerDirectVarValue vars[2];
    int var_count = 0;
    if (!power_direct_expr_collect_vars(expr_id, ops, expr_start, expr_count,
                                        pin_density, pin_duty, node_port_pin_start,
                                        node_port_pin_list, node_id, vars, var_count)) {
        return false;
    }
    uint8_t truth = 0;
    if (!power_direct_expr_truth_table(expr_id, vars, var_count, ops, expr_start,
                                       expr_count, node_port_pin_start,
                                       node_port_pin_list, node_id, truth)) {
        return false;
    }
    if (var_count == 0) {
        out_density = 0.0f;
        out_duty = (truth & 1) ? 1.0f : 0.0f;
        return true;
    }
    if (var_count == 1) {
        const float p0 = vars[0].duty;
        const float f0 = (truth & 1) ? 1.0f : 0.0f;
        const float f1 = (truth & 2) ? 1.0f : 0.0f;
        const float diff0 = f0 != f1 ? 1.0f : 0.0f;
        out_duty = power_clamp_activity_duty(power_direct_expr_lerp01(f0, f1, p0));
        out_density = fmaxf(vars[0].density * diff0, 0.0f);
        return isfinite(out_density) && isfinite(out_duty);
    }
    if (var_count != 2) return false;
    const float p0 = vars[0].duty;
    const float p1 = vars[1].duty;
    const float f00 = (truth & 1) ? 1.0f : 0.0f;
    const float f10 = (truth & 2) ? 1.0f : 0.0f;
    const float f01 = (truth & 4) ? 1.0f : 0.0f;
    const float f11 = (truth & 8) ? 1.0f : 0.0f;
    const float low0 = power_direct_expr_lerp01(f00, f01, p1);
    const float high0 = power_direct_expr_lerp01(f10, f11, p1);
    out_duty = power_clamp_activity_duty(power_direct_expr_lerp01(low0, high0, p0));

    const float d0_low = f00 != f10 ? 1.0f : 0.0f;
    const float d0_high = f01 != f11 ? 1.0f : 0.0f;
    const float diff0 = power_direct_expr_lerp01(d0_low, d0_high, p1);
    const float d1_low = f00 != f01 ? 1.0f : 0.0f;
    const float d1_high = f10 != f11 ? 1.0f : 0.0f;
    const float diff1 = power_direct_expr_lerp01(d1_low, d1_high, p0);
    out_density = fmaxf(fmaf(vars[1].density, diff1, vars[0].density * diff0), 0.0f);
    return isfinite(out_density) && isfinite(out_duty);
}

__device__ bool power_eval_expr_duty_direct_poly(int expr_id,
                                                 const GpuPowerExprOpHost* ops,
                                                 const int* expr_start,
                                                 const int* expr_count,
                                                 const float* pin_duty,
                                                 const int* node_port_pin_start,
                                                 const int* node_port_pin_list,
                                                 int node_id,
                                                 float& out_duty);

__device__ bool power_eval_expr_duty_direct(int expr_id,
                                            const GpuPowerExprOpHost* ops,
                                            const int* expr_start,
                                            const int* expr_count,
                                            const float* pin_duty,
                                            const int* node_port_pin_start,
                                            const int* node_port_pin_list,
                                            int node_id,
                                            float& out_duty) {
    return power_eval_expr_duty_direct_poly(expr_id, ops, expr_start, expr_count,
                                            pin_duty, node_port_pin_start,
                                            node_port_pin_list, node_id, out_duty);
}

__device__ bool power_eval_expr_duty_direct_poly(int expr_id,
                                                 const GpuPowerExprOpHost* ops,
                                                 const int* expr_start,
                                                 const int* expr_count,
                                                 const float* pin_duty,
                                                 const int* node_port_pin_start,
                                                 const int* node_port_pin_list,
                                                 int node_id,
                                                 float& out_duty) {
    if (!pin_duty ||
        !power_direct_expr_is_safe(expr_id, ops, expr_start, expr_count,
                                   node_port_pin_start, node_port_pin_list,
                                   node_id)) {
        return false;
    }
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    float stack[POWER_DIRECT_PROB_STACK];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        switch (op.op) {
            case 0: {
                const int pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                                           node_port_pin_list, node_id);
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                if (pin < 0 && op.arg > -2) return false;
                stack[sp++] = pin < 0 ? 0.0f : power_clamp_activity_duty(pin_duty[pin]);
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

__device__ bool power_eval_expr_activity_direct_poly(int expr_id,
                                                     const GpuPowerExprOpHost* ops,
                                                     const int* expr_start,
                                                     const int* expr_count,
                                                     const float* pin_density,
                                                     const float* pin_duty,
                                                     const int* node_port_pin_start,
                                                     const int* node_port_pin_list,
                                                     int node_id,
                                                     float& out_density,
                                                     float& out_duty);

__device__ bool power_eval_expr_activity_direct(int expr_id,
                                                const GpuPowerExprOpHost* ops,
                                                const int* expr_start,
                                                const int* expr_count,
                                                const float* pin_density,
                                                const float* pin_duty,
                                                const int* node_port_pin_start,
                                                 const int* node_port_pin_list,
                                                 int node_id,
                                                 float& out_density,
                                                 float& out_duty) {
    return power_eval_expr_activity_direct_poly(expr_id, ops, expr_start, expr_count,
                                                pin_density, pin_duty,
                                                node_port_pin_start, node_port_pin_list,
                                                node_id, out_density, out_duty);
}

__device__ bool power_eval_expr_activity_direct_poly(int expr_id,
                                                     const GpuPowerExprOpHost* ops,
                                                     const int* expr_start,
                                                     const int* expr_count,
                                                     const float* pin_density,
                                                     const float* pin_duty,
                                                     const int* node_port_pin_start,
                                                     const int* node_port_pin_list,
                                                     int node_id,
                                                     float& out_density,
                                                     float& out_duty) {
    if (!pin_density || !pin_duty ||
        !power_direct_expr_is_safe(expr_id, ops, expr_start, expr_count,
                                   node_port_pin_start, node_port_pin_list,
                                   node_id)) {
        return false;
    }
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    PowerDirectActivityValue stack[POWER_DIRECT_PROB_STACK];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        switch (op.op) {
            case 0: {
                const int pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                                           node_port_pin_list, node_id);
                if (sp >= POWER_DIRECT_PROB_STACK) return false;
                if (pin < 0 && op.arg > -2) return false;
                PowerDirectActivityValue value;
                value.duty = pin < 0 ? 0.0f : power_clamp_activity_duty(pin_duty[pin]);
                value.density = (pin >= 0 && op.zero_density == 0) ? pin_density[pin] : 0.0f;
                stack[sp++] = value;
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
                stack[sp - 1].duty = 1.0f - stack[sp - 1].duty;
                break;
            case 4: {
                if (sp < 2) return false;
                const PowerDirectActivityValue b = stack[--sp];
                const PowerDirectActivityValue a = stack[--sp];
                PowerDirectActivityValue value;
                value.duty = a.duty * b.duty;
                value.density = fmaf(a.density, b.duty, b.density * a.duty);
                stack[sp++] = value;
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const PowerDirectActivityValue b = stack[--sp];
                const PowerDirectActivityValue a = stack[--sp];
                PowerDirectActivityValue value;
                value.duty = fmaf(b.duty, 1.0f - a.duty, a.duty);
                value.density = fmaf(b.density, 1.0f - a.duty,
                                     a.density * (1.0f - b.duty));
                stack[sp++] = value;
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const PowerDirectActivityValue b = stack[--sp];
                const PowerDirectActivityValue a = stack[--sp];
                PowerDirectActivityValue value;
                value.duty = fmaf(a.duty, 1.0f - 2.0f * b.duty, b.duty);
                value.density = a.density + b.density;
                stack[sp++] = value;
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

__device__ bool power_eval_expr_diff_duty_direct_poly(int expr_id,
                                                      int diff_pin,
                                                      const GpuPowerExprOpHost* ops,
                                                      const int* expr_start,
                                                      const int* expr_count,
                                                      const float* pin_duty,
                                                      const int* node_port_pin_start,
                                                      const int* node_port_pin_list,
                                                      int node_id,
                                                      float& out_duty);

__device__ bool power_eval_expr_diff_duty_direct(int expr_id,
                                                 int diff_pin,
                                                 const GpuPowerExprOpHost* ops,
                                                 const int* expr_start,
                                                 const int* expr_count,
                                                 const float* pin_duty,
                                                 const int* node_port_pin_start,
                                                 const int* node_port_pin_list,
                                                 int node_id,
                                                 float& out_duty) {
     return power_eval_expr_diff_duty_direct_poly(expr_id, diff_pin, ops, expr_start,
                                                  expr_count, pin_duty,
                                                  node_port_pin_start,
                                                  node_port_pin_list, node_id,
                                                  out_duty);
}

__device__ bool power_eval_expr_diff_duty_direct_poly(int expr_id,
                                                      int diff_pin,
                                                      const GpuPowerExprOpHost* ops,
                                                      const int* expr_start,
                                                      const int* expr_count,
                                                      const float* pin_duty,
                                                      const int* node_port_pin_start,
                                                      const int* node_port_pin_list,
                                                      int node_id,
                                                      float& out_duty) {
     if (!pin_duty || diff_pin < 0 ||
          !power_direct_expr_is_safe(expr_id, ops, expr_start, expr_count,
                                     node_port_pin_start, node_port_pin_list,
                                     node_id)) {
          return false;
      }
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    PowerDirectProbValue stack[POWER_DIRECT_PROB_STACK];
    int sp = 0;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
          switch (op.op) {
              case 0: {
                  const int pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                                             node_port_pin_list, node_id);
                  if (sp >= POWER_DIRECT_PROB_STACK || pin < 0) return false;
                  PowerDirectProbValue value;
                  value.duty = power_clamp_activity_duty(pin_duty[pin]);
                value.diff = pin == diff_pin ? 1.0f : 0.0f;
                value.has_diff = pin == diff_pin ? 1 : 0;
                stack[sp++] = value;
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
                stack[sp - 1].duty = 1.0f - stack[sp - 1].duty;
                break;
            case 4: {
                if (sp < 2) return false;
                const PowerDirectProbValue b = stack[--sp];
                const PowerDirectProbValue a = stack[--sp];
                if (a.has_diff && b.has_diff) return false;
                PowerDirectProbValue value;
                value.duty = a.duty * b.duty;
                value.diff = a.diff * b.duty + b.diff * a.duty;
                value.has_diff = a.has_diff || b.has_diff;
                stack[sp++] = value;
                break;
            }
            case 5: {
                if (sp < 2) return false;
                const PowerDirectProbValue b = stack[--sp];
                const PowerDirectProbValue a = stack[--sp];
                if (a.has_diff && b.has_diff) return false;
                PowerDirectProbValue value;
                value.duty = a.duty + b.duty - a.duty * b.duty;
                value.diff = a.diff * (1.0f - b.duty) + b.diff * (1.0f - a.duty);
                value.has_diff = a.has_diff || b.has_diff;
                stack[sp++] = value;
                break;
            }
            case 6: {
                if (sp < 2) return false;
                const PowerDirectProbValue b = stack[--sp];
                const PowerDirectProbValue a = stack[--sp];
                if (a.has_diff && b.has_diff) return false;
                PowerDirectProbValue value;
                value.duty = a.duty * (1.0f - b.duty) + (1.0f - a.duty) * b.duty;
                value.diff = a.diff + b.diff;
                value.has_diff = a.has_diff || b.has_diff;
                stack[sp++] = value;
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

}  // namespace

__device__ __noinline__ bool power_eval_expr_activity_bdd(int expr_id,
                                                          const GpuPowerExprOpHost* ops,
                                                          const int* expr_start,
                                                          const int* expr_count,
                                                          const float* pin_density,
                                                          const float* pin_duty,
                                                          float& out_density,
                                                          float& out_duty,
                                                          const int* node_port_pin_start,
                                                          const int* node_port_pin_list,
                                                          int node_id,
                                                          int& out_var_count) {
    PowerBddContextCuda ctx;
    int root = 1;
    if (!power_bdd_build_expr(expr_id, ops, expr_start, expr_count,
                              pin_density, pin_duty, ctx, root,
                              node_port_pin_start, node_port_pin_list, node_id)) {
        return false;
    }
    out_duty = power_bdd_eval_duty(ctx, root);
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
        const int low = power_bdd_restrict(ctx, root, var, false);
        const int high = power_bdd_restrict(ctx, root, var, true);
        const int diff = power_bdd_apply(ctx, 2, low, high);
        const float diff_duty = power_bdd_eval_duty(ctx, diff);
        out_density += ctx.var_densities[var] * diff_duty;
    }
    return isfinite(out_density) && isfinite(out_duty);
}

__device__ bool power_eval_expr_activity(int expr_id,
                                         const GpuPowerExprOpHost* ops,
                                         const int* expr_start,
                                         const int* expr_count,
                                         const float* pin_density,
                                         const float* pin_duty,
                                         float& out_density,
                                         float& out_duty,
                                         const int* node_port_pin_start,
                                         const int* node_port_pin_list,
                                         int node_id) {
    const bool direct_allowed = !g_power_disable_direct_expr;
    float direct_density = 0.0f;
    float direct_duty = 0.0f;
    const bool direct_ok = direct_allowed &&
        power_eval_expr_activity_direct(expr_id, ops, expr_start, expr_count,
                                        pin_density, pin_duty,
                                        node_port_pin_start, node_port_pin_list,
                                        node_id, direct_density, direct_duty);
    if (direct_ok && !g_power_check_direct_expr) {
        out_density = direct_density;
        out_duty = direct_duty;
        return true;
    }
    float bdd_density = 0.0f;
    float bdd_duty = 0.0f;
    int bdd_var_count = 0;
    if (!power_eval_expr_activity_bdd(expr_id, ops, expr_start, expr_count,
                                      pin_density, pin_duty, bdd_density,
                                      bdd_duty, node_port_pin_start,
                                      node_port_pin_list, node_id,
                                      bdd_var_count)) {
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

__device__ bool power_expr_has_known_activity_input(int expr_id,
                                                    const GpuPowerExprOpHost* ops,
                                                    const int* expr_start,
                                                    const int* expr_count,
                                                    const int* origin,
                                                    const int* node_port_pin_start,
                                                    const int* node_port_pin_list,
                                                    int node_id) {
    if (expr_id < 0) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    if (count <= 0 || count > 128) return false;
    bool has_pin_arg = false;
    for (int k = 0; k < count; k++) {
        const auto op = ops[start + k];
        if (op.op != 0) continue;
        const int pin = power_expr_resolve_pin_arg(op.arg, node_port_pin_start,
                                                   node_port_pin_list, node_id);
        if (pin < 0) continue;
        has_pin_arg = true;
        if (!origin || origin[pin] != 0 || op.zero_density != 0) return true;
    }
    return !has_pin_arg;
}

__device__ float power_eval_expr_duty(int expr_id,
                                      const GpuPowerExprOpHost* ops,
                                      const int* expr_start,
                                      const int* expr_count,
                                      const float* pin_density,
                                      const float* pin_duty,
                                      const int* node_port_pin_start,
                                      const int* node_port_pin_list,
                                      int node_id) {
    float direct_duty = 0.0f;
    if (!g_power_disable_direct_expr &&
        power_eval_expr_duty_direct(expr_id, ops, expr_start, expr_count,
                                    pin_duty, node_port_pin_start,
                                    node_port_pin_list, node_id,
                                    direct_duty)) {
        return direct_duty;
    }
    float density = 0.0f;
    float duty = 0.0f;
    if (!power_eval_expr_activity(expr_id, ops, expr_start, expr_count, pin_density, pin_duty,
                                  density, duty, node_port_pin_start, node_port_pin_list, node_id)) {
        return 0.0f;
    }
    return fminf(fmaxf(duty, 0.0f), 1.0f);
}

__device__ __noinline__ float power_eval_expr_diff_duty_bdd(int expr_id,
                                                            int diff_pin,
                                                            const GpuPowerExprOpHost* ops,
                                                            const int* expr_start,
                                                            const int* expr_count,
                                                            const float* pin_duty,
                                                            const int* node_port_pin_start,
                                                            const int* node_port_pin_list,
                                                            int node_id) {
      PowerBddContextCuda ctx;
      int root = 1;
      if (!power_bdd_build_expr(expr_id, ops, expr_start, expr_count,
                                nullptr, pin_duty, ctx, root,
                                node_port_pin_start, node_port_pin_list,
                                node_id)) {
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
    const int low = power_bdd_restrict(ctx, root, diff_var, false);
    const int high = power_bdd_restrict(ctx, root, diff_var, true);
    const int diff = power_bdd_apply(ctx, 2, low, high);
    return power_bdd_eval_duty(ctx, diff);
}

__device__ float power_eval_expr_diff_duty(int expr_id,
                                           int diff_pin,
                                           const GpuPowerExprOpHost* ops,
                                           const int* expr_start,
                                           const int* expr_count,
                                           const float* pin_duty,
                                           const int* node_port_pin_start,
                                           const int* node_port_pin_list,
                                           int node_id) {
     if (expr_id < 0 || diff_pin < 0) return 0.0f;
     float direct_duty = 0.0f;
     if (!g_power_disable_direct_expr &&
         power_eval_expr_diff_duty_direct(expr_id, diff_pin, ops, expr_start,
                                          expr_count, pin_duty,
                                          node_port_pin_start, node_port_pin_list,
                                          node_id, direct_duty)) {
          return direct_duty;
      }
      return power_eval_expr_diff_duty_bdd(expr_id, diff_pin, ops, expr_start,
                                           expr_count, pin_duty,
                                           node_port_pin_start,
                                           node_port_pin_list, node_id);
}

}  // namespace gt
