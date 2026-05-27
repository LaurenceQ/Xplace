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
    auto d_is_seq_clock_input_pin = upload_cuda_u8("is_seq_clock_input_pin", h_is_seq_clock_input_pin);
    auto d_clock_gate_out_for_input = upload_cuda_int("clock_gate_out_for_input", h_clock_gate_out_for_input);
    auto d_clock_gate_clock_for_out = upload_cuda_int("clock_gate_clock_for_out", h_clock_gate_clock_for_out);
    auto d_clock_gate_enable_for_out = upload_cuda_int("clock_gate_enable_for_out", h_clock_gate_enable_for_out);
    auto d_clock_pins = upload_cuda_int("clock_pins", h_clock_pins);
    auto d_clock_pin_densities = upload_cuda_float("clock_pin_densities", h_clock_pin_densities);
    auto d_clock_pin_duties = upload_cuda_float("clock_pin_duties", h_clock_pin_duties);
    auto d_clock_pin_enqueue = upload_cuda_u8("clock_pin_enqueue", h_clock_pin_enqueue);
    torch::Tensor d_power_clock_slews;
    const float* d_power_clock_slews_ptr = nullptr;
    if (!h_power_clock_slews.empty()) {
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

    PowerActivityCudaModel activity_model;
    activity_model.n = n;
    activity_model.level_list_end_cpu = activity_level_list_end_cpu;
    activity_model.graph.level_list = activity_level_list;
    activity_model.graph.pin_power_level = d_pin_power_level.data_ptr<int>();
    activity_model.graph.pin_forward_arc_list_end = d_pin_forward_arc_list_end.data_ptr<index_type>();
    activity_model.graph.pin_forward_arc_list = d_pin_forward_arc_list.data_ptr<index_type>();
    activity_model.graph.timing_arc_to_pin_id = d_timing_arc_to_pin_id.data_ptr<index_type>();
    activity_model.graph.arc_types = d_arc_types.data_ptr<int>();
    activity_model.graph.arc_id2test_id = d_arc_id2test_id.data_ptr<int>();
    activity_model.graph.pin2net_map = pin2net_map;
    activity_model.graph.net_driver_pin = d_net_driver_pin.data_ptr<int>();
    activity_model.graph.flat_net2pin_start_map = activity_flat_net2pin_start_map;
    activity_model.graph.flat_net2pin_map = activity_flat_net2pin_map;
    activity_model.graph.is_load_pin = d_is_load_pin.data_ptr<uint8_t>();
    activity_model.graph.is_driver_pin = d_is_driver_pin.data_ptr<uint8_t>();
    activity_model.graph.is_cell_pin = d_is_cell_pin.data_ptr<uint8_t>();
    activity_model.graph.is_seq_output_pin = d_is_seq_output_pin.data_ptr<uint8_t>();
    activity_model.graph.is_seq_clock_input_pin = d_is_seq_clock_input_pin.data_ptr<uint8_t>();
    activity_model.graph.clock_gate_out_for_input = d_clock_gate_out_for_input.data_ptr<int>();
    activity_model.graph.clock_gate_clock_for_out = d_clock_gate_clock_for_out.data_ptr<int>();
    activity_model.graph.clock_gate_enable_for_out = d_clock_gate_enable_for_out.data_ptr<int>();
    activity_model.graph.pin2node_map = pin2node_map;
    activity_model.graph.pinLoad = pinLoad;
    activity_model.graph.dmp_C1 = dmp_C1_ptr;
    activity_model.graph.dmp_C2 = dmp_C2_ptr;
    activity_model.graph.pinSlew = pinSlew;
    activity_model.graph.power_clock_slews = d_power_clock_slews_ptr;
    activity_model.graph.num_nodes = num_nodes;
    activity_model.state.primary_inputs = d_primary_inputs.data_ptr<int>();
    activity_model.state.num_primary_inputs = static_cast<int>(h_primary_inputs.size());
    activity_model.state.case_values = nullptr;
    activity_model.state.clock_pins = d_clock_pins.data_ptr<int>();
    activity_model.state.num_clock_pins = static_cast<int>(h_clock_pins.size());
    activity_model.state.clock_pin_densities = d_clock_pin_densities.data_ptr<float>();
    activity_model.state.clock_pin_duties = d_clock_pin_duties.data_ptr<float>();
    activity_model.state.clock_pin_enqueue = d_clock_pin_enqueue.data_ptr<uint8_t>();
    activity_model.expr.expr_ops = reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>());
    activity_model.expr.expr_start = d_expr_start.data_ptr<int>();
    activity_model.expr.expr_count = d_expr_count.data_ptr<int>();
    activity_model.expr.node_port_pin_start = d_node_port_pin_start.data_ptr<int>();
    activity_model.expr.node_port_pin_list = d_node_port_pin_list.data_ptr<int>();
    activity_model.expr.pin_func_expr_id = d_pin_func_expr_id.data_ptr<int>();
    activity_model.expr.missing_func_out_start = d_missing_func_out_start.data_ptr<int>();
    activity_model.expr.missing_func_out_list = d_missing_func_out_list.data_ptr<int>();
    activity_model.state.seqs = reinterpret_cast<GpuPowerSeqHost*>(d_seqs.data_ptr<uint8_t>());
    activity_model.state.num_seqs = static_cast<int>(h_seqs.size());
    activity_model.state.pin_seq_list_start = d_pin_seq_list_start.data_ptr<int>();
    activity_model.state.pin_seq_list = d_pin_seq_list.data_ptr<int>();
    activity_model.state.feedback_seed_pins = d_feedback_seed_pins.data_ptr<int>();
    activity_model.state.num_feedback_seed_pins = static_cast<int>(h_feedback_seed_pins.size());
    activity_model.state.feedback_seed_seqs = d_feedback_seed_seqs.data_ptr<int>();
    activity_model.state.num_feedback_seed_seqs = static_cast<int>(h_feedback_seed_seqs.size());
    activity_model.config.default_density = default_density;
    activity_model.config.clock_density = clock_density;
    activity_model.config.time_unit = gtdb.time_unit;
    activity_model.config.max_activity_passes = max_activity_passes;
    activity_model.config.trace_pins = d_trace_pins.data_ptr<int>();
    activity_model.config.num_trace_pins = static_cast<int>(h_trace_pins.size());
    activity_model.config.precomputed_activity = precomputed_activity_ptr;
    activity_model.config.allow_clock_activity_override =
        readPowerBoolEnv("XPLACE_POWER_ALLOW_CLOCK_ACTIVITY_OVERRIDE", false);
    activity_model.config.min_activity_density = min_activity_density;
    activity_model.components.internal_rows = launcher_internal_rows_ptr;
    activity_model.components.num_internal_rows = launcher_internal_row_count;
    activity_model.components.num_internal_denom_groups = static_cast<int>(internal_denom_group.size());
    activity_model.components.power_allocator = d_power_allocator;
    activity_model.components.cap_unit = cap_unit;
    activity_model.components.voltage = power_voltage;
    activity_model.components.inst_switching = inst_switching_ptr;
    activity_model.components.pin_switching = pin_switching_ptr;
    activity_model.components.inst_internal = launcher_inst_internal_ptr;
    activity_model.components.internal_row_power = launcher_internal_row_power_ptr;
    activity_model.components.leakage_rows = launcher_leakage_rows_ptr;
    activity_model.components.num_leakage_rows = launcher_leakage_row_count;
    activity_model.components.leakage_groups = launcher_leakage_groups_ptr;
    activity_model.components.num_leakage_groups = launcher_leakage_group_count;
    activity_model.components.inst_leakage = launcher_inst_leakage_ptr;
    activity_model.components.leakage_row_power = launcher_leakage_row_power_ptr;
    activity_model.out = out_gpu_ptr;
    run_power_activity_cuda_launcher(activity_model);

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
                PowerInternalDenomModel denom_model;
                denom_model.n = n;
                denom_model.precomputed_activity = chunk_activity_ptr;
                denom_model.internal_rows =
                    reinterpret_cast<GpuPowerInternalHost*>(d_rows_chunk.data_ptr<uint8_t>());
                denom_model.num_internal_rows = static_cast<int>(count);
                denom_model.expr_ops =
                    reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>());
                denom_model.expr_start = d_expr_start.data_ptr<int>();
                denom_model.expr_count = d_expr_count.data_ptr<int>();
                denom_model.node_port_pin_start = d_node_port_pin_start.data_ptr<int>();
                denom_model.node_port_pin_list = d_node_port_pin_list.data_ptr<int>();
                denom_model.denom = internal_denom_gpu.data_ptr<float>();
                run_power_internal_denom_chunk_cuda_launcher(denom_model);
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
                PowerInternalContribModel contrib_model;
                contrib_model.n = n;
                contrib_model.num_nodes = num_nodes;
                contrib_model.precomputed_activity = chunk_activity_ptr;
                contrib_model.internal_rows =
                    reinterpret_cast<GpuPowerInternalHost*>(d_rows_chunk.data_ptr<uint8_t>());
                contrib_model.num_internal_rows = static_cast<int>(count);
                contrib_model.expr_ops =
                    reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>());
                contrib_model.expr_start = d_expr_start.data_ptr<int>();
                contrib_model.expr_count = d_expr_count.data_ptr<int>();
                contrib_model.node_port_pin_start = d_node_port_pin_start.data_ptr<int>();
                contrib_model.node_port_pin_list = d_node_port_pin_list.data_ptr<int>();
                contrib_model.pinSlew = pinSlew;
                contrib_model.power_clock_slews = d_power_clock_slews_ptr;
                contrib_model.dmp_C1 = dmp_C1_ptr;
                contrib_model.dmp_C2 = dmp_C2_ptr;
                contrib_model.denom = internal_denom_gpu.data_ptr<float>();
                contrib_model.power_allocator = d_power_allocator;
                contrib_model.cap_unit = cap_unit;
                contrib_model.inst_internal = inst_internal_ptr;
                contrib_model.internal_row_power = row_power_ptr;
                run_power_internal_contrib_chunk_cuda_launcher(contrib_model);
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
                PowerLeakageRowsModel rows_model;
                rows_model.n = n;
                rows_model.precomputed_activity = chunk_activity_ptr;
                rows_model.leakage_rows =
                    reinterpret_cast<GpuPowerLeakageRowHost*>(d_rows_chunk.data_ptr<uint8_t>());
                rows_model.num_leakage_rows = static_cast<int>(count);
                rows_model.expr_ops =
                    reinterpret_cast<GpuPowerExprOpHost*>(d_expr_ops.data_ptr<uint8_t>());
                rows_model.expr_start = d_expr_start.data_ptr<int>();
                rows_model.expr_count = d_expr_count.data_ptr<int>();
                rows_model.node_port_pin_start = d_node_port_pin_start.data_ptr<int>();
                rows_model.node_port_pin_list = d_node_port_pin_list.data_ptr<int>();
                rows_model.group_cond_leakage = group_cond_leakage_gpu.data_ptr<float>();
                rows_model.group_cond_duty_sum = group_cond_duty_sum_gpu.data_ptr<float>();
                rows_model.group_cond_count = group_cond_count_gpu.data_ptr<int>();
                rows_model.leakage_row_power = row_power_ptr;
                run_power_leakage_rows_chunk_cuda_launcher(rows_model);
            }
            PowerLeakageSummaryModel summary_model;
            summary_model.leakage_groups = d_leakage_groups_ptr;
            summary_model.num_leakage_groups = static_cast<int>(h_leakage_groups.size());
            summary_model.group_cond_leakage = group_cond_leakage_gpu.data_ptr<float>();
            summary_model.group_cond_duty_sum = group_cond_duty_sum_gpu.data_ptr<float>();
            summary_model.group_cond_count = group_cond_count_gpu.data_ptr<int>();
            summary_model.num_nodes = num_nodes;
            summary_model.inst_leakage = inst_leakage_ptr;
            run_power_leakage_summary_chunk_cuda_launcher(summary_model);
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
