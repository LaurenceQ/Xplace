#include "PowerCudaActivityKernels.cuh"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gt {

static bool read_power_bool_env_host(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env) return default_value;
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return !(value.empty() || value == "0" || value == "false" || value == "no");
}

static void power_cuda_call(cudaError_t err, const char* label) {
    if (err == cudaSuccess) return;
    std::string where = label ? label : "unknown";
    throw std::runtime_error("[power] CUDA call failed at " + where + ": " +
                             cudaGetErrorString(err));
}

static int power_activity_flag_word_count(int n) {
    return std::max(1, (std::max(0, n) + 31) / 32);
}

void run_power_activity_cuda_launcher(const PowerActivityCudaModel& model) {

    /*
     * 1. Unpack the host-side model into local aliases.
     *
     * This is the host launcher.  The per-pin activity math runs in kernels;
     * this function prepares device pointers, scratch buffers, runtime flags,
     * launch policy, optional debug dumps, and component power kernels.
     */
    int n = model.n;
    const std::vector<int>& level_list_end_cpu = *model.level_list_end_cpu;
    const int num_primary_inputs = model.state.num_primary_inputs;
    int* d_case_values = model.state.case_values;
    const int num_clock_pins = model.state.num_clock_pins;
    GpuPowerExprOpHost* d_expr_ops = model.expr.expr_ops;
    int* d_expr_start = model.expr.expr_start;
    int* d_expr_count = model.expr.expr_count;
    int* d_node_port_pin_start = model.expr.node_port_pin_start;
    int* d_node_port_pin_list = model.expr.node_port_pin_list;
    GpuPowerSeqHost* d_seqs = model.state.seqs;
    int num_seqs = model.state.num_seqs;
    int* d_pin_seq_list_start = model.state.pin_seq_list_start;
    int* d_pin_seq_list = model.state.pin_seq_list;
    const int num_feedback_seed_pins = model.state.num_feedback_seed_pins;
    const int num_feedback_seed_seqs = model.state.num_feedback_seed_seqs;
    float clock_density = model.config.clock_density;
    int max_activity_passes = model.config.max_activity_passes;
    int* d_trace_pins = model.config.trace_pins;
    const int num_trace_pins = model.config.num_trace_pins;
    const float* d_precomputed_activity = model.config.precomputed_activity;
    float* d_out = model.out;
    const int num_nodes = model.graph.num_nodes;
    const int* d_pin2node_map = model.graph.pin2node_map;
    const float* d_pinLoad = model.graph.pinLoad;
    const float* d_dmp_C1 = model.graph.dmp_C1;
    const float* d_dmp_C2 = model.graph.dmp_C2;
    const float* d_pinSlew = model.graph.pinSlew;
    const float* d_pin_clock_slews = model.graph.pin_clock_slews;
    const int* d_power_clock_slew_pins = model.graph.power_clock_slew_pins;
    const int num_power_clock_slew_pins = model.graph.num_power_clock_slew_pins;
    const bool allow_clock_activity_override = model.config.allow_clock_activity_override;
    const float min_activity_density = model.config.min_activity_density;
    GpuPowerInternalHost* d_internal_rows = model.components.internal_rows;
    const int num_internal_rows = model.components.num_internal_rows;
    const int num_internal_denom_groups = model.components.num_internal_denom_groups;
    GPUPowerLutAllocator* d_power_allocator = model.components.power_allocator;
    const float cap_unit = model.components.cap_unit;
    float* d_inst_switching = model.components.inst_switching;
    float* d_pin_switching = model.components.pin_switching;
    float* d_inst_internal = model.components.inst_internal;
    float* d_internal_row_power = model.components.internal_row_power;
    GpuPowerLeakageRowHost* d_leakage_rows = model.components.leakage_rows;
    const int num_leakage_rows = model.components.num_leakage_rows;
    GpuPowerLeakageGroupHost* d_leakage_groups = model.components.leakage_groups;
    const int num_leakage_groups = model.components.num_leakage_groups;
    float* d_inst_leakage = model.components.inst_leakage;
    float* d_leakage_row_power = model.components.leakage_row_power;
    const int out_activity_fields = model.out_activity_fields;
    float* d_out_density = (d_out && out_activity_fields > 0) ? d_out : nullptr;
    float* d_out_duty = (d_out && out_activity_fields > 1) ? d_out + n : nullptr;
    float* d_density = nullptr;
    float* d_duty = nullptr;
    bool owns_density = false;
    bool owns_duty = false;
    float* d_prev_density = nullptr;
    float* d_prev_duty = nullptr;
    float* d_seq_pin_density = nullptr;
    float* d_seq_pin_duty = nullptr;
    uint8_t* d_origin = nullptr;
    uint8_t* d_prev_origin = nullptr;
    uint32_t* d_active = nullptr;
    uint8_t* d_active_level = nullptr;
    uint8_t* d_visit_active = nullptr;
    uint8_t* d_seq_pin_valid = nullptr;
    int* d_pending_seq = nullptr;
    int* d_pending_seq_count = nullptr;
    /*
     * 2. Decide which scratch state is needed.
     *
     * Some callers only need precomputed activity copied out; others need full
     * propagation, inline internal/leakage power, tracing, or final dumps.
     * These booleans keep allocations and kernels scoped to the requested work.
     */
    const int num_power_levels = std::max(0, static_cast<int>(level_list_end_cpu.size()) - 1);
    const int activity_flag_words = power_activity_flag_word_count(n);
    const char* final_dump_env = std::getenv("XPLACE_POWER_ACTIVITY_FINAL_DUMP");
    const bool needs_final_activity_dump = final_dump_env && final_dump_env[0] != '\0';
    const bool needs_trace_activity = num_trace_pins > 0 && d_trace_pins;
    const bool needs_activity_propagation = !d_precomputed_activity;
    const bool needs_inline_internal =
        (d_inst_internal || d_internal_row_power) && d_internal_rows &&
        num_internal_rows > 0 && d_power_allocator;
    const bool needs_inline_leakage_activity =
        (d_inst_leakage || d_leakage_row_power) && d_leakage_rows &&
        num_leakage_rows > 0 && d_leakage_groups && num_leakage_groups > 0;
    const bool needs_density_duty =
        needs_activity_propagation || needs_inline_internal ||
        needs_inline_leakage_activity || needs_final_activity_dump || needs_trace_activity;
    const bool needs_origin =
        needs_activity_propagation || needs_final_activity_dump || needs_trace_activity;
    const bool needs_seq_activity_state = needs_activity_propagation && num_seqs > 0;
    bool defer_pending_seq = false;
    if (needs_activity_propagation) {
        if (const char* env_defer_pending = std::getenv("XPLACE_POWER_DEFER_PENDING_SEQ"))
            defer_pending_seq = std::atoi(env_defer_pending) != 0;
    }
    /*
     * 3. Allocate device scratch arrays.
     *
     * d_density/d_duty may alias the caller output buffer.  The other arrays
     * are temporary propagation state: previous pass values, sequential pending
     * flags, active-level queues, and activity origins.
     */
    if (needs_density_duty) {
        if (d_out_density && d_out_duty) {
            d_density = d_out_density;
            d_duty = d_out_duty;
        } else {
            power_cuda_call(cudaMalloc(&d_density, sizeof(float) * n), "activity malloc density");
            power_cuda_call(cudaMalloc(&d_duty, sizeof(float) * n), "activity malloc duty");
            owns_density = true;
            owns_duty = true;
        }
    }
    if (defer_pending_seq) {
        power_cuda_call(cudaMalloc(&d_prev_density, sizeof(float) * n), "activity malloc prev_density");
        power_cuda_call(cudaMalloc(&d_prev_duty, sizeof(float) * n), "activity malloc prev_duty");
    }
    if (needs_seq_activity_state) {
        power_cuda_call(cudaMalloc(&d_seq_pin_density, sizeof(float) * n), "activity malloc seq_pin_density");
        power_cuda_call(cudaMalloc(&d_seq_pin_duty, sizeof(float) * n), "activity malloc seq_pin_duty");
        power_cuda_call(cudaMalloc(&d_seq_pin_valid, sizeof(uint8_t) * n), "activity malloc seq_pin_valid");
        power_cuda_call(cudaMalloc(&d_pending_seq, sizeof(int) * num_seqs), "activity malloc pending_seq");
    }
    if (needs_activity_propagation) {
        power_cuda_call(cudaMalloc(&d_active, sizeof(uint32_t) * activity_flag_words), "activity malloc active");
        power_cuda_call(cudaMalloc(&d_active_level, sizeof(uint8_t) * std::max(1, num_power_levels)), "activity malloc active_level");
        power_cuda_call(cudaMalloc(&d_visit_active, sizeof(uint8_t) * n), "activity malloc visit_active");
        power_cuda_call(cudaMalloc(&d_pending_seq_count, sizeof(int)), "activity malloc pending_seq_count");
    }
    if (needs_origin) {
        power_cuda_call(cudaMalloc(&d_origin, sizeof(uint8_t) * n), "activity malloc origin");
    }
    if (defer_pending_seq) {
        power_cuda_call(cudaMalloc(&d_prev_origin, sizeof(uint8_t) * n), "activity malloc prev_origin");
    }
    /*
     * 4. Initialize scratch state before launching kernels.
     *
     * Activity starts at zero unless precomputed data is unpacked later.
     * Active/pending flags also start cleared before root seeding.
     */
    if (needs_density_duty) {
        power_cuda_call(cudaMemset(d_density, 0, sizeof(float) * n), "activity memset density");
        power_cuda_call(cudaMemset(d_duty, 0, sizeof(float) * n), "activity memset duty");
    }
    if (defer_pending_seq) {
        power_cuda_call(cudaMemset(d_prev_density, 0, sizeof(float) * n), "activity memset prev_density");
        power_cuda_call(cudaMemset(d_prev_duty, 0, sizeof(float) * n), "activity memset prev_duty");
    }
    if (needs_seq_activity_state) {
        power_cuda_call(cudaMemset(d_seq_pin_density, 0, sizeof(float) * n), "activity memset seq_pin_density");
        power_cuda_call(cudaMemset(d_seq_pin_duty, 0, sizeof(float) * n), "activity memset seq_pin_duty");
        power_cuda_call(cudaMemset(d_seq_pin_valid, 0, sizeof(uint8_t) * n), "activity memset seq_pin_valid");
        power_cuda_call(cudaMemset(d_pending_seq, 0, sizeof(int) * num_seqs), "activity memset pending_seq");
    }
    if (needs_activity_propagation) {
        power_cuda_call(cudaMemset(d_active, 0, sizeof(uint32_t) * activity_flag_words), "activity memset active");
        power_cuda_call(cudaMemset(d_active_level, 0, sizeof(uint8_t) * std::max(1, num_power_levels)), "activity memset active_level");
        power_cuda_call(cudaMemset(d_visit_active, 0, sizeof(uint8_t) * n), "activity memset visit_active");
        power_cuda_call(cudaMemset(d_pending_seq_count, 0, sizeof(int)), "activity memset pending_seq_count");
    }
    if (needs_origin) {
        power_cuda_call(cudaMemset(d_origin, 0, sizeof(uint8_t) * n), "activity memset origin");
    }
    if (defer_pending_seq) {
        power_cuda_call(cudaMemset(d_prev_origin, 0, sizeof(uint8_t) * n), "activity memset prev_origin");
    }
    /*
     * 5. Package model and scratch views for device kernels.
     *
     * Kernels receive a pointer to PowerActivityCudaModel plus a pointer to the
     * scratch view.  The model owns static graph/table data; scratch owns
     * mutable activity propagation state.
     */
    PowerActivityScratchView activity_scratch(d_density,
                                               d_duty,
                                               d_prev_density,
                                               d_prev_duty,
                                               d_seq_pin_density,
                                               d_seq_pin_duty,
                                               d_origin,
                                               d_prev_origin,
                                               d_active,
                                               d_active_level,
                                               d_visit_active,
                                               d_seq_pin_valid,
                                               d_pending_seq,
                                               d_pending_seq_count,
                                               num_power_levels);
    PowerActivityCudaModel* d_activity_model = nullptr;
    PowerActivityScratchView* d_activity_scratch = nullptr;
    power_cuda_call(cudaMalloc(&d_activity_model, sizeof(PowerActivityCudaModel)), "activity malloc model");
    power_cuda_call(cudaMalloc(&d_activity_scratch, sizeof(PowerActivityScratchView)), "activity malloc scratch view");
    power_cuda_call(cudaMemcpy(d_activity_model, &model, sizeof(PowerActivityCudaModel), cudaMemcpyHostToDevice),
                    "activity copy model");
    power_cuda_call(cudaMemcpy(d_activity_scratch, &activity_scratch, sizeof(PowerActivityScratchView), cudaMemcpyHostToDevice),
                    "activity copy scratch view");
    /*
     * 6. Copy runtime knobs into CUDA constants.
     *
     * Environment-controlled flags tune clamp behavior, sequential feedback,
     * and direct-expression-vs-BDD checking inside device code without changing
     * kernel signatures.
     */
    size_t power_stack_size = 32768;
    if (const char* env_stack = std::getenv("XPLACE_POWER_CUDA_STACK_SIZE")) {
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(env_stack, &end, 10);
        if (end != env_stack && parsed > 0) power_stack_size = parsed;
    }
    power_cuda_call(cudaDeviceSetLimit(cudaLimitStackSize, power_stack_size), "activity set stack size");
    power_cuda_call(cudaMemcpyToSymbol(g_power_allow_clock_activity_override,
                                       &allow_clock_activity_override,
                                       sizeof(bool)),
                    "activity copy allow_clock_activity_override");
    power_cuda_call(cudaMemcpyToSymbol(g_power_min_activity_density,
                                       &min_activity_density,
                                       sizeof(float)),
                    "activity copy min_activity_density");
    float min_activity_duty = 0.0f;
    if (const char* env = std::getenv("XPLACE_POWER_MIN_ACTIVITY_DUTY")) {
        char* end = nullptr;
        const float value = std::strtof(env, &end);
        if (end != env && value >= 0.0f) min_activity_duty = value;
    }
    power_cuda_call(cudaMemcpyToSymbol(g_power_min_activity_duty,
                                       &min_activity_duty,
                                       sizeof(float)),
                    "activity copy min_activity_duty");
    const bool disable_activity_slew_cap =
        std::getenv("XPLACE_POWER_DISABLE_ACTIVITY_SLEW_CAP") != nullptr;
    power_cuda_call(cudaMemcpyToSymbol(g_power_disable_activity_slew_cap,
                                       &disable_activity_slew_cap,
                                       sizeof(bool)),
                    "activity copy disable_activity_slew_cap");
    float seq_clock_limit_rel_tol = 0.0f;
    if (const char* env = std::getenv("XPLACE_POWER_SEQ_CLOCK_LIMIT_REL_TOL")) {
        char* end = nullptr;
        const float value = std::strtof(env, &end);
        if (end != env && value >= 0.0f) seq_clock_limit_rel_tol = value;
    }
    power_cuda_call(cudaMemcpyToSymbol(g_power_seq_clock_limit_rel_tol,
                                       &seq_clock_limit_rel_tol,
                                       sizeof(float)),
                    "activity copy seq_clock_limit_rel_tol");
    float seq_pending_min_density = 0.0f;
    if (const char* env = std::getenv("XPLACE_POWER_SEQ_PENDING_MIN_DENSITY")) {
        char* end = nullptr;
        const float value = std::strtof(env, &end);
        if (end != env && value >= 0.0f) seq_pending_min_density = value;
    }
    power_cuda_call(cudaMemcpyToSymbol(g_power_seq_pending_min_density,
                                       &seq_pending_min_density,
                                       sizeof(float)),
                    "activity copy seq_pending_min_density");
    int seq_clock_limit_rel_tol_start_pass = 1;
    if (const char* env = std::getenv("XPLACE_POWER_SEQ_CLOCK_LIMIT_REL_TOL_START_PASS"))
        seq_clock_limit_rel_tol_start_pass = std::max(1, std::atoi(env));
    const bool clamp_activity_to_clock_density =
        std::getenv("XPLACE_POWER_CLAMP_ACTIVITY_TO_CLOCK_DENSITY") != nullptr;
    const float activity_clock_density_cap = clamp_activity_to_clock_density
        ? clock_density
        : 3.4028234663852886e38f;
    power_cuda_call(cudaMemcpyToSymbol(g_power_activity_clock_density_cap,
                                       &activity_clock_density_cap,
                                       sizeof(float)),
                    "activity copy activity_clock_density_cap");
    const int require_known_seq_data =
        read_power_bool_env_host("XPLACE_POWER_REQUIRE_KNOWN_SEQ_DATA", false) ? 1 : 0;
    power_cuda_call(cudaMemcpyToSymbol(g_power_require_known_seq_data,
                                       &require_known_seq_data,
                                       sizeof(int)),
                    "activity copy require_known_seq_data");
    const int disable_direct_expr =
        read_power_bool_env_host("XPLACE_POWER_DISABLE_DIRECT_EXPR", false) ? 1 : 0;
    power_cuda_call(cudaMemcpyToSymbol(g_power_disable_direct_expr,
                                       &disable_direct_expr,
                                       sizeof(int)),
                    "activity copy disable_direct_expr");
    const int check_direct_expr =
        read_power_bool_env_host("XPLACE_POWER_CHECK_DIRECT_EXPR", false) ? 1 : 0;
    power_cuda_call(cudaMemcpyToSymbol(g_power_check_direct_expr,
                                       &check_direct_expr,
                                       sizeof(int)),
                    "activity copy check_direct_expr");
    int direct_expr_max_vars = 8;
    if (const char* env = std::getenv("XPLACE_POWER_DIRECT_EXPR_MAX_VARS")) {
        direct_expr_max_vars = std::max(0, std::atoi(env));
    }
    power_cuda_call(cudaMemcpyToSymbol(g_power_direct_expr_max_vars,
                                       &direct_expr_max_vars,
                                       sizeof(int)),
                    "activity copy direct_expr_max_vars");
    float direct_expr_density_rel_tol = 1.0e-4f;
    if (const char* env = std::getenv("XPLACE_POWER_CHECK_DIRECT_DENSITY_REL_TOL")) {
        char* end = nullptr;
        const float value = std::strtof(env, &end);
        if (end != env && value >= 0.0f) direct_expr_density_rel_tol = value;
    }
    power_cuda_call(cudaMemcpyToSymbol(g_power_direct_expr_density_rel_tol,
                                       &direct_expr_density_rel_tol,
                                       sizeof(float)),
                    "activity copy direct_expr_density_rel_tol");
    float direct_expr_duty_abs_tol = 1.0e-5f;
    if (const char* env = std::getenv("XPLACE_POWER_CHECK_DIRECT_DUTY_ABS_TOL")) {
        char* end = nullptr;
        const float value = std::strtof(env, &end);
        if (end != env && value >= 0.0f) direct_expr_duty_abs_tol = value;
    }
    power_cuda_call(cudaMemcpyToSymbol(g_power_direct_expr_duty_abs_tol,
                                       &direct_expr_duty_abs_tol,
                                       sizeof(float)),
                    "activity copy direct_expr_duty_abs_tol");
    const int direct_expr_mismatch_count = 0;
    power_cuda_call(cudaMemcpyToSymbol(g_power_direct_expr_mismatch_count,
                                       &direct_expr_mismatch_count,
                                       sizeof(int)),
                    "activity reset direct_expr_mismatch_count");
    /*
     * 7. Optional host-side tracing helpers.
     *
     * These lambdas copy selected pin or pending-sequential state back to the
     * host for debugging.  Normal runs skip this unless trace env vars are set.
     */
    std::vector<int> h_trace_pins(std::max(0, num_trace_pins));
    if (num_trace_pins > 0 && d_trace_pins) {
        cudaMemcpy(h_trace_pins.data(), d_trace_pins, sizeof(int) * num_trace_pins, cudaMemcpyDeviceToHost);
    }
    std::vector<uint8_t> h_trace_first_seen(std::max(1, num_trace_pins), 0);
    auto trace_cuda = [&](const char* tag, int pass, int pending_count) {
        for (int idx = 0; idx < num_trace_pins; ++idx) {
            const int pin = h_trace_pins[idx];
            if (pin < 0 || pin >= n) continue;
            float pin_density = 0.0f;
            float pin_duty = 0.0f;
            uint8_t pin_origin = 0;
            cudaMemcpy(&pin_density, d_density + pin, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pin_duty, d_duty + pin, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pin_origin, d_origin + pin, sizeof(uint8_t), cudaMemcpyDeviceToHost);
            int seq_count = 0;
            int seq_pending = 0;
            if (d_pin_seq_list_start && d_pin_seq_list && d_pending_seq) {
                int start = 0;
                int end = 0;
                cudaMemcpy(&start, d_pin_seq_list_start + pin, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&end, d_pin_seq_list_start + pin + 1, sizeof(int), cudaMemcpyDeviceToHost);
                seq_count = std::max(0, end - start);
                for (int pos = start; pos < end; ++pos) {
                    int seq_id = -1;
                    int pending = 0;
                    cudaMemcpy(&seq_id, d_pin_seq_list + pos, sizeof(int), cudaMemcpyDeviceToHost);
                    if (seq_id >= 0 && seq_id < num_seqs) {
                        cudaMemcpy(&pending, d_pending_seq + seq_id, sizeof(int), cudaMemcpyDeviceToHost);
                        if (pending) seq_pending++;
                    }
                }
            }
            const bool first_nonzero = pin_density > 0.0f && !h_trace_first_seen[idx];
            if (first_nonzero) h_trace_first_seen[idx] = 1;
            fprintf(stderr,
                    "[power_activity_trace_cuda] tag=%s pass=%d pending=%d pin_id=%d density=%.10e duty=%.10g origin=%d first_nonzero=%d seq_count=%d seq_pending=%d\n",
                    tag, pass, pending_count, pin, pin_density, pin_duty,
                    static_cast<int>(pin_origin),
                    first_nonzero ? 1 : 0, seq_count, seq_pending);
        }
    };
    const char* pending_seq_dump_file = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_FILE");
    int pending_seq_dump_pass = -1;
    if (const char* env = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_PASS"))
        pending_seq_dump_pass = std::atoi(env);
    std::string pending_seq_dump_tag = "after_pass";
    if (const char* env = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_TAG"))
        pending_seq_dump_tag = env;
    auto dump_cuda_pending_seq = [&](const char* tag, int pass) {
        if (!pending_seq_dump_file || pending_seq_dump_file[0] == '\0') return;
        if (pending_seq_dump_pass >= 0 && pass != pending_seq_dump_pass) return;
        if (pending_seq_dump_tag != (tag ? tag : "")) return;
        if (!d_pending_seq || !d_seqs || num_seqs <= 0) return;
        std::vector<int> h_pending(std::max(1, num_seqs), 0);
        std::vector<GpuPowerSeqHost> h_seq_dump(std::max(1, num_seqs));
        cudaMemcpy(h_pending.data(), d_pending_seq, sizeof(int) * num_seqs, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_seq_dump.data(), d_seqs, sizeof(GpuPowerSeqHost) * num_seqs, cudaMemcpyDeviceToHost);
        std::ofstream out(pending_seq_dump_file, std::ios::app);
        if (!out) return;
        out << "engine,pass,tag,node_id,inst_name,seq_id,q_pin,qn_pin,pin_id,pin_name\n";
        for (int seq_id = 0; seq_id < num_seqs; ++seq_id) {
            if (!h_pending[seq_id]) continue;
            const auto seq = h_seq_dump[seq_id];
            const int pin_id = seq.q_pin >= 0 ? seq.q_pin : seq.qn_pin;
            out << "xplace_cuda," << pass << ',' << (tag ? tag : "")
                << ",-1,," << seq_id << ','
                << seq.q_pin << ',' << seq.qn_pin << ','
                << pin_id << ",\n";
        }
    };

    /*
     * 8. Activity source selection.
     *
     * If another engine already produced activity, just copy/unpack it.  If
     * not, seed roots and run CUDA propagation through combinational levels and
     * sequential feedback passes.
     */
    if (d_precomputed_activity) {
        if (d_out && n > 0 && out_activity_fields > 0) {
            power_copy_precomputed_activity_output_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
                n, d_precomputed_activity, d_out, out_activity_fields);
            check_power_cuda_error("activity copy precomputed output");
        }
        const bool density_duty_alias_output = d_density == d_out_density && d_duty == d_out_duty;
        if (d_density && d_duty && d_origin) {
            power_unpack_precomputed_activity_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
                n, d_precomputed_activity, d_density, d_duty, d_origin);
            check_power_cuda_error("activity unpack_precomputed");
        } else if (d_density && d_duty && !density_duty_alias_output) {
            power_unpack_activity_density_duty_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
                n, d_precomputed_activity, d_density, d_duty);
            check_power_cuda_error("activity unpack_precomputed_density_duty");
        }
        trace_cuda("precomputed", 0, 0);
    } else {
    /*
     * 9. Choose propagation scheduler.
     *
     * Frontier mode uses a device-side level queue.  The default path scans
     * active levels from the host and launches per-level kernels.
     */
    bool use_frontier = false;
    if (const char* env_frontier = std::getenv("XPLACE_POWER_ACTIVITY_FRONTIER"))
        use_frontier = std::atoi(env_frontier) != 0;
    bool use_ordered_frontier = false;
    if (const char* env_ordered = std::getenv("XPLACE_POWER_ACTIVITY_ORDERED_QUEUE"))
        use_ordered_frontier = std::atoi(env_ordered) != 0;
    if (use_ordered_frontier) use_frontier = true;
    cudaDeviceProp prop{};
    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);
    if (use_frontier && !use_ordered_frontier && !prop.cooperativeLaunch) {
        fprintf(stderr, "[power_frontier] cooperative launch unsupported; falling back to level scan\n");
        use_frontier = false;
    }
    int max_comb_sweeps = 1000;
    if (const char* env_comb_sweeps = std::getenv("XPLACE_POWER_ACTIVITY_MAX_COMB_SWEEPS"))
        max_comb_sweeps = std::max(1, std::atoi(env_comb_sweeps));

    const int num_feedback_seed_items = std::max(num_feedback_seed_pins, num_feedback_seed_seqs);
    if (num_feedback_seed_items > 0) {
        power_seed_seq_feedback_state_kernel<<<BLOCK_NUMBER(num_feedback_seed_items), BLOCK_SIZE>>>(
            d_activity_model, d_activity_scratch);
        check_power_cuda_error("activity seed_seq_feedback_state");
    }

    /*
     * 9a. Frontier/queue propagation path.
     *
     * Roots are queued by level and a persistent/cooperative kernel drains the
     * queue on the device.  Ordered mode uses a single ordered kernel for
     * deterministic debugging.
     */
    if (use_frontier) {
        int *d_level_offsets = nullptr, *d_level_queue = nullptr, *d_level_counts = nullptr;
        uint32_t *d_queued = nullptr;
        int *d_overflow = nullptr;
        int *d_frontier_pending_seq_list = nullptr, *d_frontier_pending_seq_list_count = nullptr;
        std::vector<int> frontier_level_offsets = level_list_end_cpu;
        int frontier_queue_size = std::max(1, n);
        if (use_ordered_frontier) {
            int cap_mult = 4;
            if (const char* env_cap = std::getenv("XPLACE_POWER_ORDERED_QUEUE_CAP_MULT"))
                cap_mult = std::max(1, std::atoi(env_cap));
            frontier_level_offsets.assign(std::max(1, num_power_levels + 1), 0);
            for (int level = 0; level < num_power_levels; ++level) {
                const int level_count = level_list_end_cpu[level + 1] - level_list_end_cpu[level];
                frontier_level_offsets[level + 1] =
                    frontier_level_offsets[level] + std::max(1, level_count * cap_mult);
            }
            frontier_queue_size = std::max(1, frontier_level_offsets.back());
            cudaMalloc(&d_frontier_pending_seq_list, sizeof(int) * std::max(1, num_seqs));
            cudaMalloc(&d_frontier_pending_seq_list_count, sizeof(int));
            cudaMemset(d_frontier_pending_seq_list_count, 0, sizeof(int));
        }
        cudaMalloc(&d_level_offsets, sizeof(int) * std::max(1, num_power_levels + 1));
        cudaMalloc(&d_level_queue, sizeof(int) * frontier_queue_size);
        cudaMalloc(&d_level_counts, sizeof(int) * std::max(1, num_power_levels));
        cudaMalloc(&d_queued, sizeof(uint32_t) * activity_flag_words);
        cudaMalloc(&d_overflow, sizeof(int));
        if (num_power_levels + 1 > 0) {
            cudaMemcpy(d_level_offsets, frontier_level_offsets.data(), sizeof(int) * (num_power_levels + 1), cudaMemcpyHostToDevice);
        }
        cudaMemset(d_level_queue, 0, sizeof(int) * frontier_queue_size);
        cudaMemset(d_level_counts, 0, sizeof(int) * std::max(1, num_power_levels));
        cudaMemset(d_queued, 0, sizeof(uint32_t) * activity_flag_words);
        cudaMemset(d_overflow, 0, sizeof(int));
        PowerActivityQueueView activity_queue(d_level_offsets,
                                              d_level_queue,
                                              d_level_counts,
                                              d_queued,
                                              d_overflow,
                                              d_frontier_pending_seq_list,
                                              d_frontier_pending_seq_list_count);
        PowerActivityQueueView* d_activity_queue = nullptr;
        cudaMalloc(&d_activity_queue, sizeof(PowerActivityQueueView));
        cudaMemcpy(d_activity_queue, &activity_queue, sizeof(PowerActivityQueueView), cudaMemcpyHostToDevice);

        const bool ordered_root_seed =
            std::getenv("XPLACE_POWER_ORDERED_ROOT_SEED") != nullptr;
        if (ordered_root_seed) {
            power_seed_roots_level_queue_ordered_kernel<<<1, 1>>>(
                d_activity_model, d_activity_scratch, d_activity_queue);
        } else {
        if (d_case_values) {
            power_seed_case_level_queue_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
                d_activity_model, d_activity_scratch, d_activity_queue);
        }
        if (num_primary_inputs > 0) {
            power_seed_pi_level_queue_kernel<<<BLOCK_NUMBER(num_primary_inputs), BLOCK_SIZE>>>(
                d_activity_model, d_activity_scratch, d_activity_queue);
        }
        if (num_clock_pins > 0) {
            power_seed_clock_level_queue_kernel<<<BLOCK_NUMBER(num_clock_pins), BLOCK_SIZE>>>(
                d_activity_model, d_activity_scratch, d_activity_queue);
        }
        }
        check_power_cuda_error("activity frontier seed");
        if (num_trace_pins > 0) {
            fprintf(stderr, "[power_activity_trace_cuda] frontier_trace=unsupported\n");
        }

        if (use_ordered_frontier) {
            power_activity_level_queue_ordered_kernel<<<1, 1>>>(
                d_activity_model, d_activity_scratch, d_activity_queue, max_activity_passes);
        } else {
            int blocks_per_sm = 1;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, power_activity_level_queue_persistent_kernel, BLOCK_SIZE, 0);
            int coop_blocks = std::max(1, prop.multiProcessorCount * std::max(1, blocks_per_sm));
            void* args[] = {
                &d_activity_model,
                &d_activity_scratch,
                &d_activity_queue,
                &max_activity_passes

            };
            cudaError_t launch_err = cudaLaunchCooperativeKernel(
                reinterpret_cast<void*>(power_activity_level_queue_persistent_kernel),
                dim3(coop_blocks), dim3(BLOCK_SIZE), args, 0, nullptr);
            if (launch_err != cudaSuccess) {
                fprintf(stderr, "[power_frontier] cooperative level-queue launch failed: %s\n", cudaGetErrorString(launch_err));
            }
        }
        check_power_cuda_error("activity frontier propagate");
        int overflow = 0;
        cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost);
        if (overflow) fprintf(stderr, "[power_frontier] level queue overflow detected; results may be incomplete\n");
        cudaFree(d_activity_queue);
        cudaFree(d_level_offsets);
        cudaFree(d_level_queue);
        cudaFree(d_level_counts);
        cudaFree(d_queued);
        cudaFree(d_overflow);
        if (d_frontier_pending_seq_list) cudaFree(d_frontier_pending_seq_list);
        if (d_frontier_pending_seq_list_count) cudaFree(d_frontier_pending_seq_list_count);
    } else {
        /*
         * 9b. Host-driven level-scan propagation path.
         *
         * Seed case/PI/clock roots, then repeatedly visit active combinational
         * levels.  Sequential outputs can mark more levels active in later passes.
         */
        if (d_case_values) {
            power_seed_case_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
                d_activity_model, d_activity_scratch);
        }
        if (num_primary_inputs > 0) {
            power_seed_pi_kernel<<<BLOCK_NUMBER(num_primary_inputs), BLOCK_SIZE>>>(
                d_activity_model, d_activity_scratch);
        }
        if (num_clock_pins > 0) {
            power_seed_clock_active_kernel<<<BLOCK_NUMBER(num_clock_pins), BLOCK_SIZE>>>(
                d_activity_model, d_activity_scratch);
        }
        check_power_cuda_error("activity seed roots");
        int pending_count = 0;
        cudaMemcpy(&pending_count, d_pending_seq_count, sizeof(int), cudaMemcpyDeviceToHost);
        trace_cuda("after_seed", 0, pending_count);
        dump_cuda_pending_seq("after_seed", 0);

        const bool trace_level_progress =
            std::getenv("XPLACE_POWER_TRACE_LEVEL_PROGRESS") != nullptr;
        int trace_level_progress_start_pass = 0;
        if (const char* env_trace_start = std::getenv("XPLACE_POWER_TRACE_LEVEL_PROGRESS_START_PASS"))
            trace_level_progress_start_pass = std::max(0, std::atoi(env_trace_start));
        int trace_level_progress_end_pass = max_activity_passes;
        if (const char* env_trace_end = std::getenv("XPLACE_POWER_TRACE_LEVEL_PROGRESS_END_PASS"))
            trace_level_progress_end_pass = std::max(0, std::atoi(env_trace_end));
        int trace_level_progress_pass = 0;
        const bool ascending_level_scan =
            std::getenv("XPLACE_POWER_ACTIVITY_ASCENDING_SCAN") != nullptr;
        bool use_serial_level = false;
        if (const char* env_serial_level = std::getenv("XPLACE_POWER_ACTIVITY_SERIAL_LEVEL"))
            use_serial_level = std::atoi(env_serial_level) != 0;
        int serial_level_max_active = 0x3fffffff;
        if (const char* env_serial_max = std::getenv("XPLACE_POWER_ACTIVITY_SERIAL_LEVEL_MAX_ACTIVE"))
            serial_level_max_active = std::max(0, std::atoi(env_serial_max));
        int serial_level_max_count = -1;
        if (const char* env_serial_count = std::getenv("XPLACE_POWER_ACTIVITY_SERIAL_LEVEL_MAX_COUNT"))
            serial_level_max_count = std::max(0, std::atoi(env_serial_count));
        std::vector<uint8_t> serial_level_selected(std::max(1, num_power_levels), 0);
        if (const char* env_serial_levels = std::getenv("XPLACE_POWER_ACTIVITY_SERIAL_LEVELS")) {
            std::stringstream stream(env_serial_levels);
            std::string item;
            while (std::getline(stream, item, ',')) {
                const int level = std::atoi(item.c_str());
                if (level >= 0 && level < num_power_levels) serial_level_selected[level] = 1;
            }
        }
        int* d_serial_active_pins = nullptr;
        int* d_serial_active_count = nullptr;
        if (use_serial_level) {
            cudaMalloc(&d_serial_active_pins, sizeof(int) * std::max(1, n));
            cudaMalloc(&d_serial_active_count, sizeof(int));
        }
        std::vector<uint8_t> active_levels(std::max(1, num_power_levels));
        constexpr int POWER_ACTIVITY_VISIT_BLOCK_SIZE = 128;
        auto power_activity_visit_blocks = [](int work_items) {
            return (work_items + POWER_ACTIVITY_VISIT_BLOCK_SIZE - 1) / POWER_ACTIVITY_VISIT_BLOCK_SIZE;
        };
        /*
         * Visit one combinational level.
         *
         * Depending on debug knobs and level size, this either runs a serial
         * kernel for deterministic inspection or the normal parallel visitor.
         */
        auto run_level = [&](int level) {
            if (level < 0 || level >= num_power_levels || level + 1 >= static_cast<int>(level_list_end_cpu.size()))
                return;
            const int start = level_list_end_cpu[level];
            const int count = level_list_end_cpu[level + 1] - start;
            if (d_active_level) cudaMemsetAsync(d_active_level + level, 0, sizeof(uint8_t));
            if (count <= 0) {
                check_power_cuda_error("activity empty level");
                return;
            }
            if ((level < static_cast<int>(serial_level_selected.size()) && serial_level_selected[level])
                || (serial_level_max_count >= 0 && count <= serial_level_max_count)) {
                power_visit_level_serial_kernel<<<1, 1>>>(
                    d_activity_model, d_activity_scratch, start, count, defer_pending_seq);
            } else if (use_serial_level) {
                cudaMemset(d_serial_active_count, 0, sizeof(int));
                power_snapshot_level_active_list_kernel<<<BLOCK_NUMBER(count), BLOCK_SIZE>>>(
                    d_activity_model, d_activity_scratch, start, count,
                    d_serial_active_count, d_serial_active_pins);
                check_power_cuda_error("activity snapshot level active list");
                int serial_active_count = 0;
                cudaMemcpy(&serial_active_count, d_serial_active_count, sizeof(int), cudaMemcpyDeviceToHost);
                if (serial_active_count <= serial_level_max_active) {
                    power_visit_active_list_serial_kernel<<<1, 1>>>(
                        d_activity_model, d_activity_scratch,
                        d_serial_active_pins, d_serial_active_count, defer_pending_seq);
                } else {
                    power_visit_level_kernel<<<power_activity_visit_blocks(count),
                                               POWER_ACTIVITY_VISIT_BLOCK_SIZE>>>(
                        d_activity_model, d_activity_scratch, start, count, defer_pending_seq);
                }
            } else {
                power_snapshot_level_active_kernel<<<BLOCK_NUMBER(count), BLOCK_SIZE>>>(
                    d_activity_model, d_activity_scratch, start, count);
                check_power_cuda_error("activity snapshot level active");
                power_visit_level_kernel<<<power_activity_visit_blocks(count),
                                           POWER_ACTIVITY_VISIT_BLOCK_SIZE>>>(
                    d_activity_model, d_activity_scratch, start, count, defer_pending_seq);
            }
            check_power_cuda_error("activity visit level");
        };

        const bool print_pass_stats = std::getenv("XPLACE_POWER_PRINT_PASS_STATS") != nullptr;
        int total_comb_sweeps = 0;
        /*
         * Drain currently active combinational levels.
         *
         * This is one BFS-like combinational settle phase.  The sequential loop
         * below calls it after root seeding and after each sequential seed pass.
         */
        auto drain_bfs = [&]() {
            if (defer_pending_seq) {
                cudaMemcpy(d_prev_density, d_density, sizeof(float) * n, cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_prev_duty, d_duty, sizeof(float) * n, cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_prev_origin, d_origin, sizeof(uint8_t) * n, cudaMemcpyDeviceToDevice);
            }
            int level_visits = 0;
            const int max_level_visits = max_comb_sweeps;
            if (ascending_level_scan) {
                for (int level = 0; level < num_power_levels && level_visits < max_level_visits; ++level) {
                    uint8_t level_active = 0;
                    cudaMemcpy(&level_active, d_active_level + level, sizeof(uint8_t),
                               cudaMemcpyDeviceToHost);
                    if (!level_active) continue;
                    run_level(level);
                    ++level_visits;
                    if (trace_level_progress
                        && trace_level_progress_pass >= trace_level_progress_start_pass
                        && trace_level_progress_pass <= trace_level_progress_end_pass) {
                        char tag[64];
                        snprintf(tag, sizeof(tag), "after_level:%d", level);
                        trace_cuda(tag, trace_level_progress_pass, -1);
                    }
                }
            } else {
            for (; level_visits < max_level_visits; ++level_visits) {
                if (num_power_levels <= 0) break;
                cudaMemcpy(active_levels.data(), d_active_level, sizeof(uint8_t) * num_power_levels,
                           cudaMemcpyDeviceToHost);
                int next_level = -1;
                for (int level = 0; level < num_power_levels; ++level) {
                    if (active_levels[level]) {
                        next_level = level;
                        break;
                    }
                }
                if (next_level < 0) break;
                run_level(next_level);
                if (trace_level_progress
                    && trace_level_progress_pass >= trace_level_progress_start_pass
                    && trace_level_progress_pass <= trace_level_progress_end_pass) {
                    char tag[64];
                    snprintf(tag, sizeof(tag), "after_level:%d", next_level);
                    trace_cuda(tag, trace_level_progress_pass, -1);
                }
            }
            }
            total_comb_sweeps += level_visits;
            if (defer_pending_seq) {
                power_mark_pending_seq_changes_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
                    d_activity_model, d_activity_scratch);
                check_power_cuda_error("activity mark pending seq changes");
            }
            return level_visits;
        };

        trace_level_progress_pass = 0;
        drain_bfs();
        cudaMemcpy(&pending_count, d_pending_seq_count, sizeof(int), cudaMemcpyDeviceToHost);
        trace_cuda("after_comb", 0, pending_count);
        dump_cuda_pending_seq("after_comb", 0);
        /*
         * 9c. Sequential feedback fixed-point loop.
         *
         * Pending sequential elements are reseeded, then combinational activity
         * is drained again.  The loop stops when no sequential state remains
         * pending or max_activity_passes is reached.
         */
        int seq_passes = 0;
        const bool direct_ordered_seq_seed =
            std::getenv("XPLACE_POWER_DIRECT_ORDERED_SEQ_SEED") != nullptr;
        const bool ordered_seq_seed =
            direct_ordered_seq_seed || std::getenv("XPLACE_POWER_ORDERED_SEQ_SEED") != nullptr;
        const int direct_ordered_seq_seed_device = direct_ordered_seq_seed ? 1 : 0;
        cudaMemcpyToSymbol(g_power_direct_ordered_seq_seed,
                           &direct_ordered_seq_seed_device,
                           sizeof(int));
        int* d_ordered_pending_seq_ids = nullptr;
        std::vector<int> h_ordered_pending_flags;
        std::vector<int> h_ordered_pending_seq_ids;
        if (ordered_seq_seed && num_seqs > 0) {
            cudaMalloc(&d_ordered_pending_seq_ids, sizeof(int) * num_seqs);
            h_ordered_pending_flags.resize(num_seqs);
            h_ordered_pending_seq_ids.reserve(num_seqs);
        }
        for (int pass = 1; pending_count > 0 && pass < max_activity_passes; pass++) {
            seq_passes = pass;
            const int pending_before_seed = pending_count;
            if (num_seqs > 0) {
                if (seq_clock_limit_rel_tol > 0.0f && seq_clock_limit_rel_tol_start_pass > 1) {
                    const float active_tol =
                        pass >= seq_clock_limit_rel_tol_start_pass ? seq_clock_limit_rel_tol : 0.0f;
                    cudaMemcpyToSymbol(g_power_seq_clock_limit_rel_tol,
                                       &active_tol,
                                       sizeof(float));
                }
                if (ordered_seq_seed) {
                    h_ordered_pending_seq_ids.clear();
                    cudaMemcpy(h_ordered_pending_flags.data(), d_pending_seq,
                               sizeof(int) * num_seqs, cudaMemcpyDeviceToHost);
                    for (int seq_id = 0; seq_id < num_seqs; ++seq_id) {
                        if (h_ordered_pending_flags[seq_id])
                            h_ordered_pending_seq_ids.push_back(seq_id);
                    }
                    if (std::getenv("XPLACE_POWER_REVERSE_ORDERED_SEQ_SEED") != nullptr) {
                        std::reverse(h_ordered_pending_seq_ids.begin(),
                                     h_ordered_pending_seq_ids.end());
                    }
                    if (!h_ordered_pending_seq_ids.empty()) {
                        cudaMemcpy(d_ordered_pending_seq_ids,
                                   h_ordered_pending_seq_ids.data(),
                                   sizeof(int) * h_ordered_pending_seq_ids.size(),
                                   cudaMemcpyHostToDevice);
                        const int ordered_pending_count =
                            static_cast<int>(h_ordered_pending_seq_ids.size());
                        power_seed_seq_id_list_ordered_kernel<<<1, 1>>>(
                            d_activity_model, d_activity_scratch,
                            d_ordered_pending_seq_ids, ordered_pending_count);
                    } else {
                        cudaMemset(d_pending_seq_count, 0, sizeof(int));
                    }
                } else {
                    power_seed_seq_kernel<<<power_activity_visit_blocks(num_seqs),
                                            POWER_ACTIVITY_VISIT_BLOCK_SIZE>>>(
                        d_activity_model, d_activity_scratch);
                }
                check_power_cuda_error("activity seed seq");
            }
            trace_cuda("after_seq_seed", pass, pending_before_seed);
            dump_cuda_pending_seq("after_seq_seed", pass);
            trace_level_progress_pass = pass;
            const int comb_sweeps = drain_bfs();
            cudaMemcpy(&pending_count, d_pending_seq_count, sizeof(int), cudaMemcpyDeviceToHost);
            trace_cuda("after_pass", pass, pending_count);
            dump_cuda_pending_seq("after_pass", pass);
            if (print_pass_stats && std::getenv("XPLACE_POWER_PRINT_PASS_STATS_VERBOSE")) {
                fprintf(stderr,
                        "[power_activity_pass] pass=%d pending=%d comb_sweeps=%d\n",
                        pass, pending_count, comb_sweeps);
            }
        }
        if (d_ordered_pending_seq_ids) cudaFree(d_ordered_pending_seq_ids);
        if (print_pass_stats) {
            fprintf(stderr,
                    "[power_activity_passes] seq_passes=%d final_pending=%d total_comb_sweeps=%d max_seq_passes=%d max_comb_sweeps=%d\n",
                    seq_passes, pending_count, total_comb_sweeps, max_activity_passes, max_comb_sweeps);
        }
        if (d_serial_active_pins) cudaFree(d_serial_active_pins);
        if (d_serial_active_count) cudaFree(d_serial_active_count);
    }

    }

    /*
     * 10. Release propagation-only scratch that is no longer needed.
     *
     * Density/duty may still be needed for output packing or component power,
     * so those buffers are handled separately below.
     */
    auto free_device_ptr = [](auto*& ptr) {
        if (!ptr) return;
        cudaFree(ptr);
        ptr = nullptr;
    };
    auto free_owned_density_duty = [&]() {
        if (owns_density && d_density) cudaFree(d_density);
        if (owns_duty && d_duty) cudaFree(d_duty);
        d_density = nullptr;
        d_duty = nullptr;
        owns_density = false;
        owns_duty = false;
    };
    if (needs_activity_propagation) {
        free_device_ptr(d_prev_density);
        free_device_ptr(d_prev_duty);
        free_device_ptr(d_seq_pin_density);
        free_device_ptr(d_seq_pin_duty);
        free_device_ptr(d_prev_origin);
        free_device_ptr(d_active);
        free_device_ptr(d_active_level);
        free_device_ptr(d_visit_active);
        free_device_ptr(d_seq_pin_valid);
        free_device_ptr(d_pending_seq);
        free_device_ptr(d_pending_seq_count);
    }

    /*
     * 11. Pack activity output and optional final activity dump.
     *
     * Pack writes the caller-visible density/duty array.  The CSV dump is a
     * debug path that snapshots density, duty, origin, and slew caps.
     */
    if (d_out && !d_precomputed_activity) {
        power_pack_output_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(d_activity_model, d_activity_scratch);
        check_power_cuda_error("activity pack output");
    }
    if (needs_final_activity_dump) {
        const char* final_dump = final_dump_env;
        std::vector<float> h_density(std::max(0, n));
        std::vector<float> h_duty(std::max(0, n));
        std::vector<uint8_t> h_origin(std::max(0, n));
        std::vector<float> h_slew(std::max(0, n * NUM_ATTR), nanf(""));
        if (n > 0) {
            cudaMemcpy(h_density.data(), d_density, sizeof(float) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_duty.data(), d_duty, sizeof(float) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_origin.data(), d_origin, sizeof(uint8_t) * n, cudaMemcpyDeviceToHost);
            if (d_pinSlew)
                cudaMemcpy(h_slew.data(), d_pinSlew, sizeof(float) * n * NUM_ATTR, cudaMemcpyDeviceToHost);
        }
        std::ofstream dump(final_dump);
        if (dump) {
            dump << "pin_id,density,duty,origin,slew0,slew1,slew2,slew3,slew_cap\n";
            for (int pin = 0; pin < n; ++pin) {
                float min_rf_slew = 3.4028234663852886e38f;
                for (int attr = 0; attr + 1 < NUM_ATTR; attr += 2) {
                    const float rise = h_slew[pin * NUM_ATTR + attr];
                    const float fall = h_slew[pin * NUM_ATTR + attr + 1];
                    if (!std::isfinite(rise) || !std::isfinite(fall)) continue;
                    const float avg = 0.5f * (rise + fall) * model.config.time_unit;
                    if (avg > 0.0f && avg < min_rf_slew) min_rf_slew = avg;
                }
                const float slew_cap = min_rf_slew < 3.4028234663852886e38f
                    ? 1.0f / min_rf_slew
                    : 3.4028234663852886e38f;
                dump << pin << ',' << h_density[pin] << ',' << h_duty[pin]
                     << ',' << static_cast<int>(h_origin[pin])
                     << ',' << h_slew[pin * NUM_ATTR + 0]
                     << ',' << h_slew[pin * NUM_ATTR + 1]
                     << ',' << h_slew[pin * NUM_ATTR + 2]
                     << ',' << h_slew[pin * NUM_ATTR + 3]
                     << ',' << slew_cap << '\n';
            }
        }
    }
    free_device_ptr(d_origin);
    if (!needs_inline_internal && !needs_inline_leakage_activity) {
        free_owned_density_duty();
    }
    /*
     * 12. Component power kernels.
     *
     * Once activity is available, compute switching power, internal power
     * contributions, and leakage rows/summaries requested by model outputs.
     */
    if ((d_inst_switching || d_pin_switching) && d_out && d_pin2node_map && d_pinLoad) {
        if (d_inst_switching) cudaMemset(d_inst_switching, 0, sizeof(float) * num_nodes);
        if (d_pin_switching) cudaMemset(d_pin_switching, 0, sizeof(float) * n);
        power_switching_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(d_activity_model);
        check_power_cuda_error("activity switching");
    }
    if ((d_inst_internal || d_internal_row_power) && d_internal_rows && num_internal_rows > 0 && d_power_allocator) {
        if (d_inst_internal) cudaMemset(d_inst_internal, 0, sizeof(float) * num_nodes);
        if (d_internal_row_power) cudaMemset(d_internal_row_power, 0, sizeof(float) * num_internal_rows);
        float* d_denom = nullptr;
        cudaMalloc(&d_denom, sizeof(float) * std::max(1, num_internal_denom_groups));
        cudaMemset(d_denom, 0, sizeof(float) * std::max(1, num_internal_denom_groups));
        PowerInternalDenomModel denom_model(n,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            d_internal_rows,
                                            num_internal_rows,
                                            d_expr_ops,
                                            d_expr_start,
                                            d_expr_count,
                                            d_node_port_pin_start,
                                            d_node_port_pin_list,
                                            d_denom);
        power_internal_denom_fast_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            denom_model, activity_scratch);
        check_power_cuda_error("activity internal denom fast");
        constexpr int POWER_COMPONENT_EXPR_BLOCK_SIZE = 128;
        const int component_expr_blocks =
            (num_internal_rows + POWER_COMPONENT_EXPR_BLOCK_SIZE - 1) / POWER_COMPONENT_EXPR_BLOCK_SIZE;
        power_internal_denom_kernel<<<component_expr_blocks,
                                      POWER_COMPONENT_EXPR_BLOCK_SIZE>>>(
            denom_model, activity_scratch);
        check_power_cuda_error("activity internal denom");
        PowerInternalContribModel contrib_model(n,
                                                num_nodes,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                d_internal_rows,
                                                num_internal_rows,
                                                d_expr_ops,
                                                d_expr_start,
                                                d_expr_count,
                                                d_node_port_pin_start,
                                                d_node_port_pin_list,
                                                d_pinSlew,
                                                d_pin_clock_slews,
                                                d_power_clock_slew_pins,
                                                num_power_clock_slew_pins,
                                                model.graph.power_clock_slew_fallback,
                                                d_dmp_C1,
                                                d_dmp_C2,
                                                d_denom,
                                                d_power_allocator,
                                                cap_unit,
                                                d_inst_internal,
                                                d_internal_row_power);
        power_internal_contrib_fast_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            contrib_model, activity_scratch);
        check_power_cuda_error("activity internal contrib fast");
        power_internal_contrib_kernel<<<component_expr_blocks,
                                        POWER_COMPONENT_EXPR_BLOCK_SIZE>>>(
            contrib_model, activity_scratch);
        check_power_cuda_error("activity internal contrib");
        cudaFree(d_denom);
    }
    if ((d_inst_leakage || d_leakage_row_power) && d_leakage_groups && num_leakage_groups > 0) {
        if (d_inst_leakage) cudaMemset(d_inst_leakage, 0, sizeof(float) * num_nodes);
        if (d_leakage_row_power && num_leakage_rows > 0) cudaMemset(d_leakage_row_power, 0, sizeof(float) * num_leakage_rows);
        float* d_group_cond_leakage = nullptr;
        float* d_group_cond_duty_sum = nullptr;
        int* d_group_cond_count = nullptr;
        cudaMalloc(&d_group_cond_leakage, sizeof(float) * num_leakage_groups);
        cudaMalloc(&d_group_cond_duty_sum, sizeof(float) * num_leakage_groups);
        cudaMalloc(&d_group_cond_count, sizeof(int) * num_leakage_groups);
        cudaMemset(d_group_cond_leakage, 0, sizeof(float) * num_leakage_groups);
        cudaMemset(d_group_cond_duty_sum, 0, sizeof(float) * num_leakage_groups);
        cudaMemset(d_group_cond_count, 0, sizeof(int) * num_leakage_groups);
        if (d_leakage_rows && num_leakage_rows > 0) {
            PowerLeakageRowsModel rows_model(n,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             d_leakage_rows,
                                             num_leakage_rows,
                                             d_expr_ops,
                                             d_expr_start,
                                             d_expr_count,
                                             d_node_port_pin_start,
                                             d_node_port_pin_list,
                                             d_group_cond_leakage,
                                             d_group_cond_duty_sum,
                                             d_group_cond_count,
                                             d_leakage_row_power);
            power_leakage_row_fast_kernel<<<BLOCK_NUMBER(num_leakage_rows), BLOCK_SIZE>>>(
                rows_model);
            check_power_cuda_error("activity leakage rows fast");
            constexpr int POWER_LEAKAGE_EXPR_BLOCK_SIZE = 128;
            const int leakage_expr_blocks =
                (num_leakage_rows + POWER_LEAKAGE_EXPR_BLOCK_SIZE - 1) / POWER_LEAKAGE_EXPR_BLOCK_SIZE;
            power_leakage_row_kernel<<<leakage_expr_blocks,
                                       POWER_LEAKAGE_EXPR_BLOCK_SIZE>>>(
                rows_model, activity_scratch);
            check_power_cuda_error("activity leakage rows");
        }
        if (d_inst_leakage) {
            PowerLeakageSummaryModel summary_model(d_leakage_groups,
                                                   num_leakage_groups,
                                                   d_group_cond_leakage,
                                                   d_group_cond_duty_sum,
                                                   d_group_cond_count,
                                                   num_nodes,
                                                   d_inst_leakage);
            power_leakage_summary_kernel<<<BLOCK_NUMBER(num_leakage_groups), BLOCK_SIZE>>>(summary_model);
            check_power_cuda_error("activity leakage summary");
        }
        cudaFree(d_group_cond_leakage);
        cudaFree(d_group_cond_duty_sum);
        cudaFree(d_group_cond_count);
    }

    /*
     * 13. Final cleanup for device-side model/scratch wrappers and any buffers
     * retained for component power.
     */
    cudaFree(d_activity_model);
    cudaFree(d_activity_scratch);
    free_owned_density_duty();
    if (d_prev_density) cudaFree(d_prev_density);
    if (d_prev_duty) cudaFree(d_prev_duty);
    cudaFree(d_seq_pin_density);
    cudaFree(d_seq_pin_duty);
    cudaFree(d_origin);
    if (d_prev_origin) cudaFree(d_prev_origin);
    cudaFree(d_active);
    cudaFree(d_active_level);
    cudaFree(d_visit_active);
    cudaFree(d_seq_pin_valid);
    cudaFree(d_pending_seq);
    cudaFree(d_pending_seq_count);
}

}  // namespace gt
