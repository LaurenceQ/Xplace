static bool read_power_bool_env_host(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env) return default_value;
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(value.empty() || value == "0" || value == "false" || value == "no");
}

namespace {
__device__ bool g_power_allow_clock_activity_override = true;
__device__ float g_power_min_activity_density = 1.0e-10f;
__device__ float g_power_min_activity_duty = 0.0f;
__device__ bool g_power_disable_activity_slew_cap = false;
__device__ float g_power_seq_clock_limit_rel_tol = 0.0f;
__device__ float g_power_seq_pending_min_density = 0.0f;
__device__ float g_power_activity_clock_density_cap = 3.4028234663852886e38f;
__device__ int g_power_direct_ordered_seq_seed = 0;
__device__ int g_power_require_known_seq_data = 0;

__device__ __forceinline__ float power_percent_change(float value, float prev) {
    if (prev == 0.0f) return value == 0.0f ? 0.0f : 1.0f;
    return fabsf(value - prev) / fabsf(prev);
}

__device__ __forceinline__ float power_clamp_activity_duty(float duty) {
    float u = fminf(fmaxf(duty, 0.0f), 1.0f);
    const float eps = fmaxf(g_power_min_activity_duty, 0.0f);
    if (eps > 0.0f) {
        if (u < eps) u = 0.0f;
        else if ((1.0f - u) < eps) u = 1.0f;
    }
    return u;
}

__device__ __forceinline__ bool power_should_mark_pending_seq(float density) {
    return density >= fmaxf(g_power_seq_pending_min_density, 0.0f);
}

__device__ __forceinline__ float power_max_activity_density_from_slew(int pin,
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

__device__ __forceinline__ bool power_seq_density_exceeds_clock_limit(float in_density,
                                                                      float clk_density) {
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

__global__ void power_snapshot_level_active_kernel(const PowerActivityCudaModel* model,
                                                   PowerActivityScratchView* scratch,
                                                   int level_start,
                                                   int num_level_pins) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = model->graph.level_list[level_start + pos];
    if (pin < 0) return;
    scratch->visit_active[pin] = static_cast<uint8_t>(atomicExch(&scratch->active[pin], 0) != 0);
}

__global__ void power_snapshot_level_active_list_kernel(const PowerActivityCudaModel* model,
                                                        PowerActivityScratchView* scratch,
                                                        int level_start,
                                                        int num_level_pins,
                                                        int* active_count,
                                                        int* active_pins) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_level_pins) return;
    const int pin = model->graph.level_list[level_start + pos];
    if (pin < 0) return;
    const bool is_active = atomicExch(&scratch->active[pin], 0) != 0;
    if (scratch->visit_active) scratch->visit_active[pin] = static_cast<uint8_t>(is_active);
    if (!is_active || !active_count || !active_pins) return;
    const int out_pos = atomicAdd(active_count, 1);
    active_pins[out_pos] = pin;
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
                    if (var_pins[i] == op.arg) { var = i; break; }
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

__device__ __forceinline__ int power_bdd_edge_id(int edge) { return edge >> 1; }
__device__ __forceinline__ bool power_bdd_edge_inv(int edge) { return (edge & 1) != 0; }
__device__ __forceinline__ int power_bdd_not(int edge) { return edge ^ 1; }

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
        const int var = min(power_bdd_top_var(ctx, left), power_bdd_top_var(ctx, right));
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
                                    bool zero_density = false) {
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
                                     const int* node_port_pin_start = nullptr,
                                     const int* node_port_pin_list = nullptr,
                                     int node_id = -1) {
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

__device__ bool power_eval_expr_activity(int expr_id,
                                         const GpuPowerExprOpHost* ops,
                                         const int* expr_start,
                                         const int* expr_count,
                                         const float* pin_density,
                                         const float* pin_duty,
                                         float& out_density,
                                         float& out_duty,
                                         const int* node_port_pin_start = nullptr,
                                         const int* node_port_pin_list = nullptr,
                                         int node_id = -1) {
    PowerBddContextCuda ctx;
    int root = 1;
    if (!power_bdd_build_expr(expr_id, ops, expr_start, expr_count,
                              pin_density, pin_duty, ctx, root,
                              node_port_pin_start, node_port_pin_list, node_id)) {
        return false;
    }
    out_duty = power_bdd_eval_duty(ctx, root);
    out_density = 0.0f;

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

__device__ bool power_expr_has_known_activity_input(int expr_id,
                                                    const GpuPowerExprOpHost* ops,
                                                    const int* expr_start,
                                                    const int* expr_count,
                                                    const int* origin,
                                                    const int* node_port_pin_start = nullptr,
                                                    const int* node_port_pin_list = nullptr,
                                                    int node_id = -1) {
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
