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
    const float* d_power_clock_slews = model.graph.power_clock_slews;
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
    float* d_density = nullptr;
    float* d_duty = nullptr;
    float* d_prev_density = nullptr;
    float* d_prev_duty = nullptr;
    float* d_seq_pin_density = nullptr;
    float* d_seq_pin_duty = nullptr;
    int* d_origin = nullptr;
    int* d_prev_origin = nullptr;
    int* d_active = nullptr;
    uint8_t* d_active_level = nullptr;
    uint8_t* d_visit_active = nullptr;
    uint8_t* d_seq_pin_valid = nullptr;
    int* d_pending_seq = nullptr;
    int* d_pending_seq_count = nullptr;
    const int num_power_levels = std::max(0, static_cast<int>(level_list_end_cpu.size()) - 1);
    power_cuda_call(cudaMalloc(&d_density, sizeof(float) * n), "activity malloc density");
    power_cuda_call(cudaMalloc(&d_duty, sizeof(float) * n), "activity malloc duty");
    power_cuda_call(cudaMalloc(&d_prev_density, sizeof(float) * n), "activity malloc prev_density");
    power_cuda_call(cudaMalloc(&d_prev_duty, sizeof(float) * n), "activity malloc prev_duty");
    power_cuda_call(cudaMalloc(&d_seq_pin_density, sizeof(float) * n), "activity malloc seq_pin_density");
    power_cuda_call(cudaMalloc(&d_seq_pin_duty, sizeof(float) * n), "activity malloc seq_pin_duty");
    power_cuda_call(cudaMalloc(&d_origin, sizeof(int) * n), "activity malloc origin");
    power_cuda_call(cudaMalloc(&d_prev_origin, sizeof(int) * n), "activity malloc prev_origin");
    power_cuda_call(cudaMalloc(&d_active, sizeof(int) * n), "activity malloc active");
    power_cuda_call(cudaMalloc(&d_active_level, sizeof(uint8_t) * std::max(1, num_power_levels)), "activity malloc active_level");
    power_cuda_call(cudaMalloc(&d_visit_active, sizeof(uint8_t) * n), "activity malloc visit_active");
    power_cuda_call(cudaMalloc(&d_seq_pin_valid, sizeof(uint8_t) * n), "activity malloc seq_pin_valid");
    power_cuda_call(cudaMalloc(&d_pending_seq, sizeof(int) * std::max(1, num_seqs)), "activity malloc pending_seq");
    power_cuda_call(cudaMalloc(&d_pending_seq_count, sizeof(int)), "activity malloc pending_seq_count");
    power_cuda_call(cudaMemset(d_density, 0, sizeof(float) * n), "activity memset density");
    power_cuda_call(cudaMemset(d_duty, 0, sizeof(float) * n), "activity memset duty");
    power_cuda_call(cudaMemset(d_prev_density, 0, sizeof(float) * n), "activity memset prev_density");
    power_cuda_call(cudaMemset(d_prev_duty, 0, sizeof(float) * n), "activity memset prev_duty");
    power_cuda_call(cudaMemset(d_seq_pin_density, 0, sizeof(float) * n), "activity memset seq_pin_density");
    power_cuda_call(cudaMemset(d_seq_pin_duty, 0, sizeof(float) * n), "activity memset seq_pin_duty");
    power_cuda_call(cudaMemset(d_origin, 0, sizeof(int) * n), "activity memset origin");
    power_cuda_call(cudaMemset(d_prev_origin, 0, sizeof(int) * n), "activity memset prev_origin");
    power_cuda_call(cudaMemset(d_active, 0, sizeof(int) * n), "activity memset active");
    power_cuda_call(cudaMemset(d_active_level, 0, sizeof(uint8_t) * std::max(1, num_power_levels)), "activity memset active_level");
    power_cuda_call(cudaMemset(d_visit_active, 0, sizeof(uint8_t) * n), "activity memset visit_active");
    power_cuda_call(cudaMemset(d_seq_pin_valid, 0, sizeof(uint8_t) * n), "activity memset seq_pin_valid");
    power_cuda_call(cudaMemset(d_pending_seq, 0, sizeof(int) * std::max(1, num_seqs)), "activity memset pending_seq");
    power_cuda_call(cudaMemset(d_pending_seq_count, 0, sizeof(int)), "activity memset pending_seq_count");
    PowerActivityScratchView activity_scratch;
    activity_scratch.density = d_density;
    activity_scratch.duty = d_duty;
    activity_scratch.prev_density = d_prev_density;
    activity_scratch.prev_duty = d_prev_duty;
    activity_scratch.seq_pin_density = d_seq_pin_density;
    activity_scratch.seq_pin_duty = d_seq_pin_duty;
    activity_scratch.origin = d_origin;
    activity_scratch.prev_origin = d_prev_origin;
    activity_scratch.active = d_active;
    activity_scratch.active_level = d_active_level;
    activity_scratch.visit_active = d_visit_active;
    activity_scratch.seq_pin_valid = d_seq_pin_valid;
    activity_scratch.pending_seq = d_pending_seq;
    activity_scratch.pending_seq_count = d_pending_seq_count;
    activity_scratch.num_power_levels = num_power_levels;
    PowerActivityCudaModel* d_activity_model = nullptr;
    PowerActivityScratchView* d_activity_scratch = nullptr;
    power_cuda_call(cudaMalloc(&d_activity_model, sizeof(PowerActivityCudaModel)), "activity malloc model");
    power_cuda_call(cudaMalloc(&d_activity_scratch, sizeof(PowerActivityScratchView)), "activity malloc scratch view");
    power_cuda_call(cudaMemcpy(d_activity_model, &model, sizeof(PowerActivityCudaModel), cudaMemcpyHostToDevice),
                    "activity copy model");
    power_cuda_call(cudaMemcpy(d_activity_scratch, &activity_scratch, sizeof(PowerActivityScratchView), cudaMemcpyHostToDevice),
                    "activity copy scratch view");
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
            int pin_origin = 0;
            cudaMemcpy(&pin_density, d_density + pin, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pin_duty, d_duty + pin, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&pin_origin, d_origin + pin, sizeof(int), cudaMemcpyDeviceToHost);
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
                    tag, pass, pending_count, pin, pin_density, pin_duty, pin_origin,
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

    if (d_precomputed_activity) {
        power_unpack_precomputed_activity_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
            n, d_precomputed_activity, d_density, d_duty, d_origin);
        check_power_cuda_error("activity unpack_precomputed");
        trace_cuda("precomputed", 0, 0);
    } else {
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

    if (use_frontier) {
        int *d_level_offsets = nullptr, *d_level_queue = nullptr, *d_level_counts = nullptr;
        int *d_queued = nullptr, *d_overflow = nullptr;
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
        cudaMalloc(&d_queued, sizeof(int) * std::max(1, n));
        cudaMalloc(&d_overflow, sizeof(int));
        if (num_power_levels + 1 > 0) {
            cudaMemcpy(d_level_offsets, frontier_level_offsets.data(), sizeof(int) * (num_power_levels + 1), cudaMemcpyHostToDevice);
        }
        cudaMemset(d_level_queue, 0, sizeof(int) * frontier_queue_size);
        cudaMemset(d_level_counts, 0, sizeof(int) * std::max(1, num_power_levels));
        cudaMemset(d_queued, 0, sizeof(int) * std::max(1, n));
        cudaMemset(d_overflow, 0, sizeof(int));
        PowerActivityQueueView activity_queue;
        activity_queue.level_offsets = d_level_offsets;
        activity_queue.level_queue = d_level_queue;
        activity_queue.level_counts = d_level_counts;
        activity_queue.queued = d_queued;
        activity_queue.overflow = d_overflow;
        activity_queue.pending_seq_list = d_frontier_pending_seq_list;
        activity_queue.pending_seq_list_count = d_frontier_pending_seq_list_count;
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
