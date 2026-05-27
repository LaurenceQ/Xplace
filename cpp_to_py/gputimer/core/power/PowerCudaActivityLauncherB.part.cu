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

        bool defer_pending_seq = false;
        if (const char* env_defer_pending = std::getenv("XPLACE_POWER_DEFER_PENDING_SEQ"))
            defer_pending_seq = std::atoi(env_defer_pending) != 0;
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
                    power_visit_level_kernel<<<BLOCK_NUMBER(count), BLOCK_SIZE>>>(
                        d_activity_model, d_activity_scratch, start, count, defer_pending_seq);
                }
            } else {
                power_snapshot_level_active_kernel<<<BLOCK_NUMBER(count), BLOCK_SIZE>>>(
                    d_activity_model, d_activity_scratch, start, count);
                check_power_cuda_error("activity snapshot level active");
                power_visit_level_kernel<<<BLOCK_NUMBER(count), BLOCK_SIZE>>>(
                    d_activity_model, d_activity_scratch, start, count, defer_pending_seq);
            }
            check_power_cuda_error("activity visit level");
        };

        const bool print_pass_stats = std::getenv("XPLACE_POWER_PRINT_PASS_STATS") != nullptr;
        int total_comb_sweeps = 0;
        auto drain_bfs = [&]() {
            if (defer_pending_seq) {
                cudaMemcpy(d_prev_density, d_density, sizeof(float) * n, cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_prev_duty, d_duty, sizeof(float) * n, cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_prev_origin, d_origin, sizeof(int) * n, cudaMemcpyDeviceToDevice);
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
                    power_seed_seq_kernel<<<BLOCK_NUMBER(num_seqs), BLOCK_SIZE>>>(
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

    if (d_out) {
        power_pack_output_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(d_activity_model, d_activity_scratch);
        check_power_cuda_error("activity pack output");
    }
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
        PowerInternalDenomModel denom_model;
        denom_model.n = n;
        denom_model.internal_rows = d_internal_rows;
        denom_model.num_internal_rows = num_internal_rows;
        denom_model.expr_ops = d_expr_ops;
        denom_model.expr_start = d_expr_start;
        denom_model.expr_count = d_expr_count;
        denom_model.node_port_pin_start = d_node_port_pin_start;
        denom_model.node_port_pin_list = d_node_port_pin_list;
        denom_model.denom = d_denom;
        PowerInternalDenomModel* d_denom_model = nullptr;
        cudaMalloc(&d_denom_model, sizeof(PowerInternalDenomModel));
        cudaMemcpy(d_denom_model, &denom_model, sizeof(PowerInternalDenomModel), cudaMemcpyHostToDevice);
        power_internal_denom_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            d_denom_model, d_activity_scratch);
        check_power_cuda_error("activity internal denom");
        PowerInternalContribModel contrib_model;
        contrib_model.n = n;
        contrib_model.num_nodes = num_nodes;
        contrib_model.internal_rows = d_internal_rows;
        contrib_model.num_internal_rows = num_internal_rows;
        contrib_model.expr_ops = d_expr_ops;
        contrib_model.expr_start = d_expr_start;
        contrib_model.expr_count = d_expr_count;
        contrib_model.node_port_pin_start = d_node_port_pin_start;
        contrib_model.node_port_pin_list = d_node_port_pin_list;
        contrib_model.pinSlew = d_pinSlew;
        contrib_model.power_clock_slews = d_power_clock_slews;
        contrib_model.dmp_C1 = d_dmp_C1;
        contrib_model.dmp_C2 = d_dmp_C2;
        contrib_model.denom = d_denom;
        contrib_model.power_allocator = d_power_allocator;
        contrib_model.cap_unit = cap_unit;
        contrib_model.inst_internal = d_inst_internal;
        contrib_model.internal_row_power = d_internal_row_power;
        PowerInternalContribModel* d_contrib_model = nullptr;
        cudaMalloc(&d_contrib_model, sizeof(PowerInternalContribModel));
        cudaMemcpy(d_contrib_model, &contrib_model, sizeof(PowerInternalContribModel), cudaMemcpyHostToDevice);
        power_internal_contrib_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            d_contrib_model, d_activity_scratch);
        check_power_cuda_error("activity internal contrib");
        cudaFree(d_denom_model);
        cudaFree(d_contrib_model);
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
            PowerLeakageRowsModel rows_model;
            rows_model.n = n;
            rows_model.leakage_rows = d_leakage_rows;
            rows_model.num_leakage_rows = num_leakage_rows;
            rows_model.expr_ops = d_expr_ops;
            rows_model.expr_start = d_expr_start;
            rows_model.expr_count = d_expr_count;
            rows_model.node_port_pin_start = d_node_port_pin_start;
            rows_model.node_port_pin_list = d_node_port_pin_list;
            rows_model.group_cond_leakage = d_group_cond_leakage;
            rows_model.group_cond_duty_sum = d_group_cond_duty_sum;
            rows_model.group_cond_count = d_group_cond_count;
            rows_model.leakage_row_power = d_leakage_row_power;
            PowerLeakageRowsModel* d_rows_model = nullptr;
            cudaMalloc(&d_rows_model, sizeof(PowerLeakageRowsModel));
            cudaMemcpy(d_rows_model, &rows_model, sizeof(PowerLeakageRowsModel), cudaMemcpyHostToDevice);
            power_leakage_row_kernel<<<BLOCK_NUMBER(num_leakage_rows), BLOCK_SIZE>>>(
                d_rows_model, d_activity_scratch);
            check_power_cuda_error("activity leakage rows");
            cudaFree(d_rows_model);
        }
        if (d_inst_leakage) {
            PowerLeakageSummaryModel summary_model;
            summary_model.leakage_groups = d_leakage_groups;
            summary_model.num_leakage_groups = num_leakage_groups;
            summary_model.group_cond_leakage = d_group_cond_leakage;
            summary_model.group_cond_duty_sum = d_group_cond_duty_sum;
            summary_model.group_cond_count = d_group_cond_count;
            summary_model.num_nodes = num_nodes;
            summary_model.inst_leakage = d_inst_leakage;
            PowerLeakageSummaryModel* d_summary_model = nullptr;
            cudaMalloc(&d_summary_model, sizeof(PowerLeakageSummaryModel));
            cudaMemcpy(d_summary_model, &summary_model, sizeof(PowerLeakageSummaryModel), cudaMemcpyHostToDevice);
            power_leakage_summary_kernel<<<BLOCK_NUMBER(num_leakage_groups), BLOCK_SIZE>>>(d_summary_model);
            check_power_cuda_error("activity leakage summary");
            cudaFree(d_summary_model);
        }
        cudaFree(d_group_cond_leakage);
        cudaFree(d_group_cond_duty_sum);
        cudaFree(d_group_cond_count);
    }

    cudaFree(d_activity_model);
    cudaFree(d_activity_scratch);
    cudaFree(d_density);
    cudaFree(d_duty);
    cudaFree(d_prev_density);
    cudaFree(d_prev_duty);
    cudaFree(d_seq_pin_density);
    cudaFree(d_seq_pin_duty);
    cudaFree(d_origin);
    cudaFree(d_prev_origin);
    cudaFree(d_active);
    cudaFree(d_active_level);
    cudaFree(d_visit_active);
    cudaFree(d_seq_pin_valid);
    cudaFree(d_pending_seq);
    cudaFree(d_pending_seq_count);
}
