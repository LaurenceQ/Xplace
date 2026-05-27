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
