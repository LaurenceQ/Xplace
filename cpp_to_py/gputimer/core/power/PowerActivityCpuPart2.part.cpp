        if (port->has_function_) {
            PowerExpr expr;
            if (!computed && expr.compile(port->function_expr_, cell)) {
                float density = 0.0f;
                float duty = 0.0f;
                if (evalPowerExprActivity(expr, cell, node, act, density, duty, &const_port_values)) {
                    changed = set_activity(pin_id, density, duty, 3, false, false);
                    computed = true;
                }
            }
        }
        const int cg_clk = clock_gate_clock_for_out[pin_id];
        const int cg_en = clock_gate_enable_for_out[pin_id];
        if (cg_clk >= 0 && cg_en >= 0 && (act[cg_clk].origin != 0 || act[cg_en].origin != 0)) {
            const float density = act[cg_clk].density * act[cg_en].duty +
                                  act[cg_en].density * act[cg_clk].duty;
            const float duty = act[cg_clk].duty * act[cg_en].duty;
            changed = set_activity(pin_id, density, duty, 3, false, false) || changed;
            computed = true;
        }
        return computed;
    };

    auto seed_reg_outputs = [&](int node_id) {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return;
        const auto& node = gtdb.gpdb.getNodes()[node_id];
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) return;
        const auto& const_port_values = const_port_values_for_node(node_id, cell);

        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            PowerExpr data_expr;
            PowerExpr clk_expr;
            if (!data_expr.compile(seq->next_state_expr_, cell)) continue;
            if (!clk_expr.compile(seqClockExpr(seq), cell)) continue;

            std::unordered_map<int, int> seq_data_const_port_values;
            std::unordered_set<int> seq_data_zero_density_ports;
            for (const auto& op : data_expr.ops()) {
                if (op.opcode != PowerExprOpcode::port || op.port_id < 0
                    || op.port_id >= static_cast<int>(cell->ports_.size()))
                    continue;
                const std::string& port_name = cell->ports_[op.port_id]->name;
                if (node.portMap.find(port_name) == node.portMap.end())
                    seq_data_const_port_values[op.port_id] = 0;
                if (ignore_scan_enable_density && cell->ports_[op.port_id]
                    && cell->ports_[op.port_id]->nextstate_type_ == "scan_enable")
                    seq_data_zero_density_ports.insert(op.port_id);
            }

            float in_density = 0.0f, in_duty = 0.0f;
            float clk_density_eval = 0.0f, clk_duty = 0.5f;
            const auto* zero_density_ports =
                seq_data_zero_density_ports.empty() ? nullptr : &seq_data_zero_density_ports;
            if (require_known_seq_data
                && !expr_has_known_activity_input(data_expr, cell, node, &seq_data_const_port_values,
                                                  zero_density_ports))
                continue;
            if (!evalPowerExprActivity(data_expr, cell, node, act, in_density, in_duty,
                                       &seq_data_const_port_values, zero_density_ports)) continue;
            if (!evalPowerExprActivity(clk_expr, cell, node, act, clk_density_eval, clk_duty, &const_port_values)) {
                clk_density_eval = clock_density;
                clk_duty = 0.5f;
            }

            float out_density = in_density;
            float out_duty = in_duty;
            if (in_density > clk_density_eval / 2.0f) {
                if (!seq->is_latch_)
                    out_density = 2.0f * in_duty * (1.0f - in_duty) * clk_density_eval;
                else
                    out_density = in_density * clk_duty;
            }

            const std::string seq_out = normalize_expr(seq->output_name_);
            const std::string seq_out_inv = normalize_expr(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
                const std::string func = normalize_expr(port->function_expr_);
                if (!seq_out.empty() && func == seq_out) {
                    seq_pin_activity[pin_id] = CpuActivity{out_density, out_duty, 3};
                    seq_pin_activity_valid[pin_id] = 1;
                    emit_path_trace("seq_seed", -1, -1, pin_id,
                                    act[pin_id].density, act[pin_id].duty,
                                    out_density, out_duty,
                                    true, true, static_cast<int>(pending_regs.size()),
                                    "q");
                    enqueue(pin_id, false, pin_id, -1, "seq_seed_output");
                } else if (!seq_out_inv.empty() && func == seq_out_inv) {
                    const float inv_duty = 1.0f - out_duty;
                    seq_pin_activity[pin_id] = CpuActivity{out_density, inv_duty, 3};
                    seq_pin_activity_valid[pin_id] = 1;
                    emit_path_trace("seq_seed", -1, -1, pin_id,
                                    act[pin_id].density, act[pin_id].duty,
                                    out_density, inv_duty,
                                    true, true, static_cast<int>(pending_regs.size()),
                                    "qn");
                    enqueue(pin_id, false, pin_id, -1, "seq_seed_output");
                }
            }
        }
    };

    std::vector<std::vector<PowerTraceEdge>> seq_reverse_edges(n);
    int next_seq_trace_arc = -1000000;
    auto collect_expr_pins = [&](const PowerExpr& expr,
                                 const LibertyCell* cell,
                                 const gp::GPNode& node,
                                 std::vector<int>& pins) {
        if (!cell) return;
        for (const auto& op : expr.ops()) {
            if (op.opcode != PowerExprOpcode::port || op.port_id < 0
                || op.port_id >= static_cast<int>(cell->ports_.size()))
                continue;
            const std::string& port_name = cell->ports_[op.port_id]->name;
            auto pin_itr = node.portMap.find(port_name);
            if (pin_itr == node.portMap.end()) continue;
            const int pin_id = pin_itr->second;
            if (pin_id >= 0 && pin_id < n
                && std::find(pins.begin(), pins.end(), pin_id) == pins.end())
                pins.push_back(pin_id);
        }
    };
    auto add_seq_reverse_edge = [&](int from_pin, int to_pin, const char* reason) {
        if (from_pin < 0 || from_pin >= n || to_pin < 0 || to_pin >= n) return;
        seq_reverse_edges[to_pin].push_back({next_seq_trace_arc--, from_pin, to_pin, reason});
    };
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) continue;
        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            PowerExpr data_expr;
            PowerExpr clk_expr;
            if (!data_expr.compile(seq->next_state_expr_, cell)) continue;
            clk_expr.compile(seqClockExpr(seq), cell);
            std::vector<int> data_pins;
            std::vector<int> clock_pins_expr;
            collect_expr_pins(data_expr, cell, node, data_pins);
            collect_expr_pins(clk_expr, cell, node, clock_pins_expr);
            const std::string seq_out = normalize_expr(seq->output_name_);
            const std::string seq_out_inv = normalize_expr(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
                const std::string func = normalize_expr(port->function_expr_);
                if ((!seq_out.empty() && func == seq_out) ||
                    (!seq_out_inv.empty() && func == seq_out_inv)) {
                    for (int pred_pin : data_pins) add_seq_reverse_edge(pred_pin, pin_id, "seq_data");
                    for (int pred_pin : clock_pins_expr) add_seq_reverse_edge(pred_pin, pin_id, "seq_clock");
                }
            }
        }
    }

    bool level_lifo = true;
    if (const char* order_env = std::getenv("XPLACE_POWER_ACTIVITY_LEVEL_ORDER")) {
        std::string order(order_env);
        std::transform(order.begin(), order.end(), order.begin(), [](unsigned char c) { return std::tolower(c); });
        if (order == "fifo") level_lifo = false;
    }

    auto run_queue = [&](int pass) {
        path_trace_pass = pass;
        while (!nonempty_queue_levels.empty()) {
            const int level = *nonempty_queue_levels.begin();
            path_trace_level_tag = std::string("level:") + std::to_string(level);
            auto& queue = level_queues[level];
            if (queue.empty()) {
                nonempty_queue_levels.erase(level);
                continue;
            }
            int pin_id;
            if (level_lifo) {
                pin_id = queue.back();
                queue.pop_back();
            } else {
                pin_id = queue.front();
                queue.pop_front();
            }
            if (queue.empty()) nonempty_queue_levels.erase(level);
                bool force_visit = force_propagate_on_visit[pin_id] != 0;
                force_propagate_on_visit[pin_id] = 0;
                in_queue[pin_id] = 0;
                emit_path_trace("visit", -1, -1, pin_id,
                                act[pin_id].density, act[pin_id].duty,
                                act[pin_id].density, act[pin_id].duty,
                                force_visit, false, static_cast<int>(pending_regs.size()),
                                force_visit ? "force_visit" : "queue");

                bool changed = false;
                if (is_load_pin[pin_id]) {
                    int net_id = pin_to_net[pin_id];
                    const int driver_pin = (net_id >= 0 && net_id < static_cast<int>(net_driver_pin.size()))
                        ? net_driver_pin[net_id] : -1;
                    if (driver_pin >= 0 && driver_pin < n && driver_pin != pin_id
                        && act[driver_pin].origin != 0) {
                        if (trace_matches(pin_id)) {
                            std::cerr << "[power_activity_trace_net_sink] sink=" << gtdb.pin_names[pin_id]
                                      << " from=" << gtdb.pin_names[driver_pin]
                                      << " driver_level=" << pin_level[driver_pin]
                                      << " sink_level=" << pin_level[pin_id]
                                      << " density=" << act[driver_pin].density
                                      << " duty=" << act[driver_pin].duty
                                      << std::endl;
                        }
                        emit_path_trace("net_sink", -1, driver_pin, pin_id,
                                        act[pin_id].density, act[pin_id].duty,
                                        act[driver_pin].density, act[driver_pin].duty,
                                        false, false, static_cast<int>(pending_regs.size()),
                                        "copy_driver_activity");
                        changed = set_activity(pin_id, act[driver_pin].density, act[driver_pin].duty, 3, false, false);
                    }
                }

                if (is_driver_pin[pin_id]) {
                    bool output_changed = false;
                    bool computed = eval_output_pin_activity(pin_id, output_changed);
                    if (computed)
                        changed = changed || output_changed || force_visit;
                    else
                        changed = changed || force_visit;
                }

                if (changed) {
                    int node_id = pin_id >= 0 && pin_id < n ? pin_to_node[pin_id] : -1;
                    LibertyCell* cell = get_cell(node_id);
                    if (is_load_pin[pin_id] && cell && cell->sequentials_.empty())
                        eval_cell_outputs(node_id, true);
                    if (is_load_pin[pin_id] && cell && !cell->sequentials_.empty()
                        && (mark_seq_clock_loads || !is_seq_clock_input_pin(pin_id)))
                        mark_pending_reg(node_id, pin_id);
                    if (is_load_pin[pin_id] && pin_id >= 0 && pin_id < n &&
                        clock_gate_out_for_input[pin_id] >= 0) {
                        enqueue(clock_gate_out_for_input[pin_id], false, pin_id, -1, "clock_gate");
                    }
                    enqueue_adjacent_vertices(pin_id);
                }
        }
    };

    std::vector<int> clock_pins = build_clock_pins();
    std::vector<uint8_t> is_clock_pin(n, 0);
    for (int pin_id : clock_pins) {
        if (pin_id >= 0 && pin_id < n) is_clock_pin[pin_id] = 1;
        if (pin_id >= 0 && pin_id < n) clock_activity_protected[pin_id] = 1;
    }
    std::vector<uint8_t> is_primary_input(n, 0);
    std::vector<uint8_t> actual_seed_seen(n, 0);
    const bool seed_timing_zero_indeg_roots =
        readPowerBoolEnv("XPLACE_POWER_SEED_TIMING_ZERO_INDEG", true);
    const bool seed_floating_load_roots =
        readPowerBoolEnv("XPLACE_POWER_SEED_FLOATING_LOADS", true);
    auto seed_root_pin = [&](int pin_id, const char* reason) {
        if (pin_id < 0 || pin_id >= n || is_clock_pin[pin_id]) return;
        actual_seed_seen[pin_id] = 1;
        if (set_activity(pin_id, default_density, 0.5f, 1, false, false))
            enqueue_adjacent_vertices(pin_id);
    };
    for (int pin_id : gtdb.primary_inputs) {
        if (pin_id >= 0 && pin_id < n) is_primary_input[pin_id] = 1;
        if (pin_id >= 0 && pin_id < n && is_driver_pin[pin_id] && !is_clock_pin[pin_id]) {
            seed_root_pin(pin_id, "primary_input");
        }
    }

    if (seed_timing_zero_indeg_roots) {
        for (int pin_id : gtdb.pin_frontiers) {
            if (pin_id < 0 || pin_id >= n) continue;
            if (is_primary_input[pin_id] || is_clock_pin[pin_id]) continue;
            seed_root_pin(pin_id, "timing_zero_indeg");
        }
    }

    // OpenSTA power levelization seeds root load pins too. These appear on
    // no-driver input nets and are not always represented in timing frontiers.
    if (seed_floating_load_roots) {
        for (int pin_id = 0; pin_id < n; pin_id++) {
            if (!is_load_pin[pin_id] || is_primary_input[pin_id] || is_clock_pin[pin_id]) continue;
            const int net_id = pin_to_net[pin_id];
            const int driver =
                (net_id >= 0 && net_id < static_cast<int>(net_driver_pin.size())) ? net_driver_pin[net_id] : -1;
            if (driver < 0)
                seed_root_pin(pin_id, "floating_load_input");
        }
    }

    // Constant-generator outputs are roots in OpenSTA's power graph.
    for (int pin_id = 0; pin_id < n; pin_id++) {
        if (!is_driver_pin[pin_id] || is_primary_input[pin_id] || is_clock_pin[pin_id]) continue;
        int node_id = pin_to_node[pin_id];
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
        bool has_input_pin = false;
        for (int node_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
            if (node_pin >= 0 && node_pin < n && is_load_pin[node_pin]) {
                has_input_pin = true;
                break;
            }
        }
        if (!has_input_pin) {
            seed_root_pin(pin_id, "const_output");
        }
    }

    for (int pin_id : clock_pins) {
        if (pin_id >= 0 && pin_id < n) actual_seed_seen[pin_id] = 1;
        auto [pin_density, pin_duty] = clock_activity_for_pin(pin_id);
        const int node_id = pin_id >= 0 && pin_id < n ? pin_to_node[pin_id] : -1;
        LibertyCell* cell = get_cell(node_id);
        const bool enqueue_clock_tree = pin_id >= 0 && pin_id < n && is_load_pin[pin_id]
            && (!cell || cell->sequentials_.empty());
        if (set_activity(pin_id, pin_density, pin_duty, 2, true, false) && enqueue_clock_tree)
            enqueue_adjacent_vertices(pin_id);
    }

    auto dump_trace_paths = [&]() {
        if (!trace_path_out_env || trace_path_out_env[0] == '\0') return;
        std::vector<int> target_pins =
            resolvePowerPathTargetPins(readPowerPathTargetQueries(), gtdb.pin_names);
        std::unordered_set<std::string> or_seed_roots =
            readOpenroadSeedRootNames(std::getenv("XPLACE_POWER_OPENROAD_ROOTS_FILE"));
        std::vector<uint8_t> common_seed(n, 0);
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (!actual_seed_seen[pin_id]) continue;
            const std::string name = normalizePowerPathName(gtdb.pin_names[pin_id]);
            if (or_seed_roots.empty() || or_seed_roots.count(name)) common_seed[pin_id] = 1;
        }
        auto valid_power_arc = [&](int arc_id, int from_pin, int to_pin) -> bool {
            if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) return false;
            if (arc_id < static_cast<int>(gtdb.arc_id2test_id.size()) && gtdb.arc_id2test_id[arc_id] != -1)
                return false;
            if (to_pin < 0 || to_pin >= n || from_pin < 0 || from_pin >= n) return false;
            if (std::getenv("XPLACE_POWER_ACTIVITY_SKIP_BACK_LEVEL_ARCS")
                && arc_id < static_cast<int>(gtdb.arc_types.size()) && gtdb.arc_types[arc_id] == 1
                && pin_level[to_pin] <= pin_level[from_pin])
                return false;
            if (arc_id < static_cast<int>(gtdb.arc_types.size()) && gtdb.arc_types[arc_id] == 1) {
                int to_node = pin_to_node[to_pin];
                LibertyCell* to_cell = get_cell(to_node);
                if (to_cell && !to_cell->sequentials_.empty() && is_driver_pin[to_pin])
                    return false;
            }
            return true;
        };

        std::ofstream tsv(trace_path_out_env);
        if (!tsv) return;
        tsv << "path_id\tstep\ttarget_pin_id\ttarget_pin\tseed_pin_id\tseed_pin"
            << "\tarc_id\tfrom_pin_id\tfrom_pin\tto_pin_id\tto_pin"
            << "\tedge_kind\tfrom_level\tto_level\n";
        std::string json_path;
        if (const char* json_env = std::getenv("XPLACE_POWER_TRACE_PATH_JSON")) {
            json_path = json_env;
        } else {
            json_path = trace_path_out_env;
            const size_t dot = json_path.find_last_of('.');
            if (dot == std::string::npos) json_path += ".json";
            else json_path.replace(dot, std::string::npos, ".json");
        }
        std::ofstream json(json_path);
        if (json) json << "{\n  \"paths\": [\n";
        bool first_json_path = true;
        int path_id = 0;
        for (int target_pin : target_pins) {
            if (target_pin < 0 || target_pin >= n) continue;
            std::vector<int> pred_pin(n, -2);
            std::vector<int> pred_arc(n, -1);
            std::vector<std::string> pred_reason(n);
            std::queue<int> queue;
            pred_pin[target_pin] = -1;
            queue.push(target_pin);
            int seed_pin = -1;
            while (!queue.empty()) {
                const int pin_id = queue.front();
                queue.pop();
                if (common_seed[pin_id]) {
                    seed_pin = pin_id;
                    break;
                }
                if (pin_id >= 0 && pin_id + 1 < static_cast<int>(gtdb.pin_backward_arc_list_end.size())) {
                    const int start = gtdb.pin_backward_arc_list_end[pin_id];
                    const int end = gtdb.pin_backward_arc_list_end[pin_id + 1];
                    for (int idx = start; idx < end; ++idx) {
                        const int arc_id = gtdb.pin_backward_arc_list[idx];
                        if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.timing_arc_from_pin_id.size())) continue;
                        const int from_pin = gtdb.timing_arc_from_pin_id[arc_id];
                        if (!valid_power_arc(arc_id, from_pin, pin_id)) continue;
                        if (pred_pin[from_pin] != -2) continue;
                        pred_pin[from_pin] = pin_id;
                        pred_arc[from_pin] = arc_id;
                        pred_reason[from_pin] = "power_arc";
                        queue.push(from_pin);
                    }
                }
                for (const PowerTraceEdge& edge : seq_reverse_edges[pin_id]) {
                    if (edge.from_pin < 0 || edge.from_pin >= n || pred_pin[edge.from_pin] != -2)
                        continue;
                    pred_pin[edge.from_pin] = pin_id;
                    pred_arc[edge.from_pin] = edge.arc_id;
                    pred_reason[edge.from_pin] = edge.reason;
                    queue.push(edge.from_pin);
                }
            }
            if (seed_pin < 0) {
                tsv << path_id++ << "\t-1\t" << target_pin << '\t'
                    << gtdb.pin_names[target_pin]
                    << "\t-1\t\t-1\t-1\t\t-1\t\tno_path\t-1\t-1\n";
                continue;
            }
            std::vector<int> path_pins;
            for (int pin_id = seed_pin; pin_id >= 0; pin_id = pred_pin[pin_id]) {
                path_pins.push_back(pin_id);
                if (pin_id == target_pin) break;
            }
            const int current_path = path_id++;
            if (json) {
                if (!first_json_path) json << ",\n";
                first_json_path = false;
                json << "    {\"path_id\": " << current_path
                     << ", \"seed_pin\": \"" << gtdb.pin_names[seed_pin]
                     << "\", \"target_pin\": \"" << gtdb.pin_names[target_pin]
                     << "\", \"steps\": " << (path_pins.size() > 1 ? path_pins.size() - 1 : 0)
                     << "}";
            }
            for (size_t step = 0; step + 1 < path_pins.size(); ++step) {
                const int from_pin = path_pins[step];
                const int to_pin = path_pins[step + 1];
                const int arc_id = pred_arc[from_pin];
                tsv << current_path << '\t' << step << '\t'
                    << target_pin << '\t' << gtdb.pin_names[target_pin] << '\t'
                    << seed_pin << '\t' << gtdb.pin_names[seed_pin] << '\t'
                    << arc_id << '\t'
                    << from_pin << '\t' << gtdb.pin_names[from_pin] << '\t'
                    << to_pin << '\t' << gtdb.pin_names[to_pin] << '\t'
                    << pred_reason[from_pin] << '\t'
                    << (from_pin < static_cast<int>(pin_level.size()) ? pin_level[from_pin] : -1) << '\t'
                    << (to_pin < static_cast<int>(pin_level.size()) ? pin_level[to_pin] : -1) << '\n';
            }
        }
        if (json) json << "\n  ]\n}\n";
    };
    dump_trace_paths();

    int max_activity_passes = 50;
    if (const char* env = std::getenv("XPLACE_POWER_ACTIVITY_MAX_PASSES")) {
        max_activity_passes = std::max(1, std::atoi(env));
    }
    auto trace_pending_regs = [&](int pass) {
        if (!std::getenv("XPLACE_POWER_ACTIVITY_TRACE_REGS")) return;
        for (int node_id : pending_regs) {
            if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
            std::cerr << "[power_activity_reg] pass=" << pass
                      << " node=" << node_id
                      << " inst=" << gtdb.gpdb.getNodes()[node_id].getName()
                      << std::endl;
        }
    };
    auto emit_trace = [&](const char* tag, int pass, size_t pending_count) {
        for (int pin_id : trace_pin_ids) {
            if (pin_id < 0 || pin_id >= n) continue;
            const bool first_nonzero = act[pin_id].density > 0.0f && !trace_first_seen[pin_id];
            if (first_nonzero) trace_first_seen[pin_id] = 1;
            const int node_id = pin_to_node[pin_id];
            const bool node_pending = node_id >= 0 && node_id < static_cast<int>(pending_reg_flag.size()) &&
                                      pending_reg_flag[node_id];
            const bool cell_seq = get_cell(node_id) && !get_cell(node_id)->sequentials_.empty();
            std::cerr << "[power_activity_trace_cpu] tag=" << tag
                      << " pass=" << pass
                      << " pending=" << pending_count
                      << " pin_id=" << pin_id
                      << " pin=" << gtdb.pin_names[pin_id]
                      << " density=" << act[pin_id].density
                      << " duty=" << act[pin_id].duty
                      << " origin=" << act[pin_id].origin
                      << " first_nonzero=" << (first_nonzero ? 1 : 0)
                      << " is_load=" << static_cast<int>(is_load_pin[pin_id])
                      << " is_driver=" << static_cast<int>(is_driver_pin[pin_id])
                      << " node=" << node_id
                      << " node_pending=" << (node_pending ? 1 : 0)
                      << " cell_seq=" << (cell_seq ? 1 : 0);
            if (node_id >= 0 && node_id < static_cast<int>(gtdb.gpdb.getNodes().size())) {
                std::cerr << " inst=" << gtdb.gpdb.getNodes()[node_id].getName();
            }
            std::cerr << std::endl;
        }
    };
    std::ofstream activity_snapshot_csv;
    int activity_snapshot_max_pass = 6;
    if (const char* snapshot_csv_env = std::getenv("XPLACE_POWER_ACTIVITY_SNAPSHOT_CSV")) {
        if (snapshot_csv_env[0] != '\0') {
            activity_snapshot_max_pass =
                readPowerActivitySnapshotMaxPass("XPLACE_POWER_ACTIVITY_SNAPSHOT_MAX_PASS", 6);
            activity_snapshot_csv.open(snapshot_csv_env);
            if (activity_snapshot_csv) {
                activity_snapshot_csv
                    << "engine,split,design,pass,tag,pin_id,pin_name,pin_name_norm,"
                    << "inst_name,port_name,is_load,is_driver,node_id,node_pending,"
                    << "cell_seq,density,duty,origin,pending_count\n";
            }
        }
    }
    const char* snapshot_split_env = std::getenv("XPLACE_POWER_ACTIVITY_SNAPSHOT_SPLIT");
    if (!snapshot_split_env || snapshot_split_env[0] == '\0')
        snapshot_split_env = std::getenv("DESIGN_SET");
    const char* snapshot_design_env = std::getenv("XPLACE_POWER_ACTIVITY_SNAPSHOT_DESIGN");
    if (!snapshot_design_env || snapshot_design_env[0] == '\0')
        snapshot_design_env = std::getenv("DESIGN_NAME");
    const std::string snapshot_split = snapshot_split_env ? snapshot_split_env : "";
    const std::string snapshot_design = snapshot_design_env ? snapshot_design_env : "";
    auto emit_activity_snapshot = [&](const char* tag, int pass, size_t pending_count) {
        if (!activity_snapshot_csv || pass > activity_snapshot_max_pass) return;
        for (int pin_id = 0; pin_id < n; pin_id++) {
            const std::string& pin_name = gtdb.pin_names[pin_id];
            const int node_id = pin_to_node[pin_id];
            std::string inst_name = pin_name;
            std::string port_name;
            const size_t colon = pin_name.rfind(':');
            if (colon != std::string::npos) {
                inst_name = pin_name.substr(0, colon);
                port_name = pin_name.substr(colon + 1);
            } else if (node_id >= 0 && node_id < static_cast<int>(gtdb.gpdb.getNodes().size())) {
                inst_name = gtdb.gpdb.getNodes()[node_id].getName();
            }
            const bool node_pending = node_id >= 0 && node_id < static_cast<int>(pending_reg_flag.size()) &&
                                      pending_reg_flag[node_id];
            const bool cell_seq = get_cell(node_id) && !get_cell(node_id)->sequentials_.empty();
            activity_snapshot_csv
                << "xplace_cpu,"
                << csvEscapePowerActivitySnapshot(snapshot_split) << ','
                << csvEscapePowerActivitySnapshot(snapshot_design) << ','
                << pass << ','
                << csvEscapePowerActivitySnapshot(tag ? tag : "") << ','
                << pin_id << ','
                << csvEscapePowerActivitySnapshot(pin_name) << ','
                << csvEscapePowerActivitySnapshot(normalizePowerActivitySnapshotName(pin_name)) << ','
                << csvEscapePowerActivitySnapshot(inst_name) << ','
                << csvEscapePowerActivitySnapshot(port_name) << ','
                << static_cast<int>(is_load_pin[pin_id]) << ','
                << static_cast<int>(is_driver_pin[pin_id]) << ','
                << node_id << ','
                << (node_pending ? 1 : 0) << ','
                << (cell_seq ? 1 : 0) << ','
                << std::setprecision(10) << act[pin_id].density << ','
                << std::setprecision(10) << act[pin_id].duty << ','
                << act[pin_id].origin << ','
                << pending_count << '\n';
        }
        activity_snapshot_csv.flush();
    };
    const char* pending_seq_dump_file = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_FILE");
    int pending_seq_dump_pass = -1;
    if (const char* env = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_PASS"))
        pending_seq_dump_pass = std::atoi(env);
    std::string pending_seq_dump_tag = "after_pass";
    if (const char* env = std::getenv("XPLACE_POWER_PENDING_SEQ_DUMP_TAG"))
        pending_seq_dump_tag = env;
    auto dump_cpu_pending_regs = [&](const char* tag, int pass) {
        if (!pending_seq_dump_file || pending_seq_dump_file[0] == '\0') return;
        if (pending_seq_dump_pass >= 0 && pass != pending_seq_dump_pass) return;
        if (pending_seq_dump_tag != (tag ? tag : "")) return;
        std::ofstream out(pending_seq_dump_file, std::ios::app);
        if (!out) return;
        out << "engine,pass,tag,node_id,inst_name,seq_id,q_pin,qn_pin,pin_id,pin_name,"
               "trigger_pin,trigger_pin_name,trigger_port,trigger_density,trigger_duty,trigger_origin\n";
        for (int node_id : pending_regs) {
            if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
            const auto& node = gtdb.gpdb.getNodes()[node_id];
            const int trigger_pin = node_id < static_cast<int>(pending_reg_trigger_pin.size())
                ? pending_reg_trigger_pin[node_id] : -1;
            const char* trigger_name = (trigger_pin >= 0 && trigger_pin < n)
                ? gtdb.pin_names[trigger_pin].c_str() : "";
            const char* trigger_port = (trigger_pin >= 0 && trigger_pin < n)
                ? gtdb.pin_names[trigger_pin].c_str() : "";
            const float trigger_density = (trigger_pin >= 0 && trigger_pin < n)
                ? act[trigger_pin].density : 0.0f;
            const float trigger_duty = (trigger_pin >= 0 && trigger_pin < n)
                ? act[trigger_pin].duty : 0.0f;
            const int trigger_origin = (trigger_pin >= 0 && trigger_pin < n)
                ? act[trigger_pin].origin : 0;
            if (trigger_pin >= 0 && trigger_pin < n) {
                const std::string& full_name = gtdb.pin_names[trigger_pin];
                const size_t slash = full_name.rfind('/');
                if (slash != std::string::npos) trigger_port = full_name.c_str() + slash + 1;
            }
            bool wrote_pin = false;
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n || !is_seq_output_pin[pin_id]) continue;
                out << "xplace_cpu," << pass << ','
                    << csvEscapePowerActivitySnapshot(tag ? tag : "") << ','
                    << node_id << ','
                    << csvEscapePowerActivitySnapshot(node.getName()) << ",-1,"
                    << pin_id << ",-1," << pin_id << ','
                    << csvEscapePowerActivitySnapshot(gtdb.pin_names[pin_id]) << ','
                    << trigger_pin << ','
                    << csvEscapePowerActivitySnapshot(trigger_name) << ','
                    << csvEscapePowerActivitySnapshot(trigger_port) << ','
                    << std::setprecision(10) << trigger_density << ','
                    << std::setprecision(10) << trigger_duty << ','
                    << trigger_origin << '\n';
                wrote_pin = true;
            }
            if (!wrote_pin) {
                out << "xplace_cpu," << pass << ','
                    << csvEscapePowerActivitySnapshot(tag ? tag : "") << ','
                    << node_id << ','
                    << csvEscapePowerActivitySnapshot(node.getName())
                    << ",-1,-1,-1,-1,,"
                    << trigger_pin << ','
                    << csvEscapePowerActivitySnapshot(trigger_name) << ','
                    << csvEscapePowerActivitySnapshot(trigger_port) << ','
                    << std::setprecision(10) << trigger_density << ','
                    << std::setprecision(10) << trigger_duty << ','
                    << trigger_origin << '\n';
            }
        }
    };
    emit_trace("after_seed", 0, pending_regs.size());
    emit_activity_snapshot("after_seed", 0, pending_regs.size());
    dump_cpu_pending_regs("after_seed", 0);
    trace_pending_regs(0);
    // Initial combinational propagation from roots/clock network.
    run_queue(0);
    emit_trace("after_comb", 0, pending_regs.size());
    emit_activity_snapshot("after_comb", 0, pending_regs.size());
    dump_cpu_pending_regs("after_comb", 0);
    for (int pass = 1; !pending_regs.empty() && pass < max_activity_passes; pass++) {
        std::vector<int> regs = std::move(pending_regs);
        pending_regs.clear();
        path_trace_pass = pass;
        path_trace_level_tag = "seq_seed";
        for (int node_id : regs) {
            if (node_id >= 0 && node_id < static_cast<int>(pending_reg_flag.size()))
                pending_reg_flag[node_id] = 0;
            if (node_id >= 0 && node_id < static_cast<int>(pending_reg_trigger_pin.size()))
                pending_reg_trigger_pin[node_id] = -1;
            seed_reg_outputs(node_id);
        }
        emit_trace("after_seq_seed", pass, regs.size());
        emit_activity_snapshot("after_seq_seed", pass, regs.size());
        dump_cpu_pending_regs("after_seq_seed", pass);
        run_queue(pass);
        trace_pending_regs(pass);
        emit_trace("after_pass", pass, pending_regs.size());
        emit_activity_snapshot("after_pass", pass, pending_regs.size());
        dump_cpu_pending_regs("after_pass", pass);
    }

    auto out = torch::empty({n, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    auto acc = out.accessor<float, 2>();
    for (int i = 0; i < n; i++) {
        acc[i][0] = act[i].density;
        acc[i][1] = act[i].duty;
        acc[i][2] = static_cast<float>(act[i].origin);
    }
    return out;
}
