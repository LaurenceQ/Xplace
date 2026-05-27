                const int arc_id = gtdb.pin_forward_arc_list[idx];
                if (timing_level_edge_valid(arc_id) && !h_power_disabled_loop_arc[arc_id])
                    return true;
            }
            return false;
        };
        std::vector<uint8_t> visited(n, 0);
        std::vector<uint8_t> on_path(n, 0);
        auto dfs_from = [&](int root_pin) {
            struct Frame {
                int pin = -1;
                int next = 0;
                int end = 0;
            };
            std::vector<Frame> stack;
            if (root_pin < 0 || root_pin >= n || visited[root_pin]) return;
            visited[root_pin] = 1;
            on_path[root_pin] = 1;
            stack.push_back({root_pin, gtdb.pin_forward_arc_list_end[root_pin],
                             gtdb.pin_forward_arc_list_end[root_pin + 1]});
            while (!stack.empty()) {
                Frame& frame = stack.back();
                bool advanced = false;
                while (frame.next < frame.end) {
                    const int arc_id = gtdb.pin_forward_arc_list[frame.next++];
                    if (!timing_level_edge_valid(arc_id) || h_power_disabled_loop_arc[arc_id])
                        continue;
                    const int to_pin = edge_to_pin(arc_id);
                    if (to_pin < 0 || to_pin >= n) continue;
                    if (!visited[to_pin]) {
                        visited[to_pin] = 1;
                        on_path[to_pin] = 1;
                        stack.push_back({to_pin, gtdb.pin_forward_arc_list_end[to_pin],
                                         gtdb.pin_forward_arc_list_end[to_pin + 1]});
                        advanced = true;
                        break;
                    }
                    if (on_path[to_pin]) {
                        h_power_disabled_loop_arc[arc_id] = 1;
                        h_timing_loop_roots.push_back(to_pin);
                        disabled_loop_arc_count++;
                    }
                }
                if (!advanced) {
                    on_path[frame.pin] = 0;
                    stack.pop_back();
                }
            }
        };
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (!has_valid_in(pin_id) && has_valid_out(pin_id)) {
                h_timing_loop_roots.push_back(pin_id);
                dfs_from(pin_id);
            }
        }
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (!visited[pin_id] && has_valid_out(pin_id)) dfs_from(pin_id);
        }
        std::sort(h_timing_loop_roots.begin(), h_timing_loop_roots.end());
        h_timing_loop_roots.erase(std::unique(h_timing_loop_roots.begin(), h_timing_loop_roots.end()),
                                  h_timing_loop_roots.end());
    }
    for (int from_pin = 0; from_pin < n; ++from_pin) {
        if (from_pin + 1 >= static_cast<int>(gtdb.pin_forward_arc_list_end.size())) break;
        const int start = gtdb.pin_forward_arc_list_end[from_pin];
        const int end = gtdb.pin_forward_arc_list_end[from_pin + 1];
        for (int idx = start; idx < end; ++idx) {
            const int arc_id = gtdb.pin_forward_arc_list[idx];
            if (arc_id < 0 || arc_id >= static_cast<int>(h_power_arc_types.size())) continue;
            if (h_power_arc_types[arc_id] != 1) continue;
            if (arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) continue;
            const int to_pin = gtdb.timing_arc_to_pin_id[arc_id];
            if (to_pin < 0 || to_pin >= n || !h_is_seq_output_pin[to_pin]) continue;
            if (!skip_all_seq_output_arcs && !skip_seq_output_arc_for_power(arc_id, from_pin, to_pin))
                h_power_arc_types[arc_id] = 0;
        }
    }

    auto is_power_clock_slew_pin = [&](int pin_id) {
        if (pin_id < 0 || pin_id >= n) return false;
        if (pin_id < static_cast<int>(gtdb.pin_is_ideal_clk.size()) &&
            gtdb.pin_is_ideal_clk[pin_id]) {
            return true;
        }
        if (pin_id >= static_cast<int>(gtdb.pin_is_clk.size()) ||
            !gtdb.pin_is_clk[pin_id]) {
            return false;
        }
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            const int idx = pin_id * NUM_ATTR + attr;
            if (idx >= 0 && idx < static_cast<int>(gtdb.pin_clock_slews.size()) &&
                std::isfinite(gtdb.pin_clock_slews[idx])) {
                return true;
            }
        }
        return false;
    };
    bool has_ideal_clock_pins = false;
    for (int pin_id = 0; pin_id < n; ++pin_id) {
        if (is_power_clock_slew_pin(pin_id)) {
            has_ideal_clock_pins = true;
            break;
        }
    }

    std::vector<float> h_power_clock_slews;
    if (has_ideal_clock_pins && need_internal_power) {
        h_power_clock_slews.assign(n * NUM_ATTR, nanf(""));
        std::array<float, NUM_ATTR> fallback_clock_slews;
        fallback_clock_slews.fill(nanf(""));
        if (!gtdb.clock_transitions.empty()) {
            fallback_clock_slews = gtdb.clock_transitions.begin()->second;
        }
        for (float& slew : fallback_clock_slews) {
            if (!std::isfinite(slew)) slew = 0.0f;
        }
        auto set_power_clock_slew_pin = [&](int pin_id) {
            if (pin_id < 0 || pin_id >= n) return;
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                float slew = nanf("");
                const int idx = pin_id * NUM_ATTR + attr;
                if (idx >= 0 && idx < static_cast<int>(gtdb.pin_clock_slews.size()))
                    slew = gtdb.pin_clock_slews[idx];
                if (!std::isfinite(slew)) slew = fallback_clock_slews[attr];
                h_power_clock_slews[idx] = slew;
            }
        };
        std::vector<uint8_t> h_power_clock_slew_pin(n, 0);
        auto mark_power_clock_slew_pin = [&](int pin_id) {
            if (pin_id >= 0 && pin_id < n) h_power_clock_slew_pin[pin_id] = 1;
        };
        for (int pin_id : h_clock_pins) {
            if (is_power_clock_slew_pin(pin_id)) mark_power_clock_slew_pin(pin_id);
        }
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (h_is_seq_clock_input_pin[pin_id] && is_power_clock_slew_pin(pin_id))
                mark_power_clock_slew_pin(pin_id);
        }

        const int num_nets = static_cast<int>(gtdb.gpdb.getNets().size());
        std::vector<uint8_t> power_clock_slew_net(num_nets, 0);
        auto mark_power_clock_slew_net = [&](int net_id) {
            if (net_id >= 0 && net_id < num_nets) power_clock_slew_net[net_id] = 1;
        };
        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (is_power_clock_slew_pin(pin_id)) mark_power_clock_slew_net(h_pin_to_net[pin_id]);
        }
        for (int net_id = 0; net_id < num_nets; ++net_id) {
            if (!power_clock_slew_net[net_id]) continue;
            for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
                mark_power_clock_slew_pin(pin_id);
            }
        }

        for (int pin_id = 0; pin_id < n; ++pin_id) {
            if (h_power_clock_slew_pin[pin_id]) set_power_clock_slew_pin(pin_id);
        }
    }

    std::vector<uint8_t> h_is_primary_input(n, 0);
    std::vector<int> h_primary_inputs;
    h_primary_inputs.reserve(gtdb.primary_inputs.size());
    std::vector<std::string> h_seed_reason(n);
    auto add_seed_reason = [&](int pin_id, const char* reason) {
        if (pin_id < 0 || pin_id >= n || !reason || reason[0] == '\0') return;
        std::string& current = h_seed_reason[pin_id];
        const std::string value(reason);
        if (current.empty()) {
            current = value;
        } else if (current.find(value) == std::string::npos) {
            current += ";";
            current += value;
        }
    };
    auto add_seed_pin = [&](int pin_id, const char* reason) {
        h_primary_inputs.push_back(pin_id);
        add_seed_reason(pin_id, reason);
    };
    int root_primary_count = 0;
    int root_zero_indeg_count = 0;
    int root_const_output_count = 0;
    int root_seq_feedback_count = 0;
    int state_seq_feedback_count = 0;
    int root_power_level_count = 0;
    int root_floating_load_count = 0;
    std::vector<int> h_feedback_seed_pins;
    std::vector<int> h_feedback_seed_seqs;
    bool seed_default_inputs = true;
    if (const char* env = std::getenv("XPLACE_POWER_SEED_INPUTS")) {
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        seed_default_inputs = !(value.empty() || value == "0" || value == "false" || value == "no");
    }
    for (auto pin : gtdb.primary_inputs) {
        const int pin_id = static_cast<int>(pin);
        if (pin_id >= 0 && pin_id < n) h_is_primary_input[pin_id] = 1;
        if (seed_default_inputs && pin_id >= 0 && pin_id < n && h_is_driver_pin[pin_id]
            && !h_is_clock_pin[pin_id]) {
            add_seed_pin(pin_id, "primary_input");
            root_primary_count++;
        }
    }
    if (seed_default_inputs) {
        if (seed_timing_zero_indeg_roots) {
            for (int pin_id : gtdb.pin_frontiers) {
                if (pin_id < 0 || pin_id >= n) continue;
                if (h_is_primary_input[pin_id] || h_is_clock_pin[pin_id]) continue;
                add_seed_pin(pin_id, "timing_zero_indeg");
                root_zero_indeg_count++;
            }
        }
        if (seed_floating_load_roots) {
            for (int pin_id = 0; pin_id < n; pin_id++) {
                if (!h_is_load_pin[pin_id] || h_is_primary_input[pin_id] || h_is_clock_pin[pin_id]) continue;
                const int net_id = h_pin_to_net[pin_id];
                const int driver =
                    (net_id >= 0 && net_id < static_cast<int>(h_net_driver_pin.size())) ? h_net_driver_pin[net_id] : -1;
                if (driver >= 0) continue;
                add_seed_pin(pin_id, "floating_load_input");
                root_floating_load_count++;
            }
        }
        if (seed_timing_loop_roots) {
            for (int pin_id : h_timing_loop_roots) {
                if (pin_id < 0 || pin_id >= n) continue;
                if (h_is_primary_input[pin_id] || h_is_clock_pin[pin_id]) continue;
                if (!h_is_load_pin[pin_id] && !h_is_driver_pin[pin_id]) continue;
                add_seed_pin(pin_id, "timing_loop_root");
                root_timing_loop_count++;
            }
        }
    }
    if (seed_default_inputs && (seed_seq_feedback_outputs || init_seq_feedback_state)) {
        std::vector<uint8_t> seed_seen(n, 0);
        std::vector<uint8_t> state_pin_seen(n, 0);
        std::vector<uint8_t> state_seq_seen(h_seqs.size(), 0);
        auto collect_feedback_data_pins = [&](int expr_id, int driver_pin, std::vector<int>* data_pins) -> bool {
            if (expr_id < 0 || driver_pin < 0 || driver_pin >= n) return false;
            const int driver_net = h_pin_to_net[driver_pin];
            if (driver_net < 0 || driver_net >= static_cast<int>(h_net_driver_pin.size())) return false;
            if (h_net_driver_pin[driver_net] != driver_pin) return false;
            bool matched = false;
            const int start = h_expr_start[expr_id];
            const int end = start + h_expr_count[expr_id];
            for (int op_i = start; op_i < end; ++op_i) {
                if (h_expr_ops[op_i].op != 0) continue;
                const int data_pin = h_expr_ops[op_i].arg;
                if (data_pin < 0 || data_pin >= n || h_pin_to_net[data_pin] != driver_net) continue;
                if (seed_seq_feedback_d_only) {
                    const int node_id = h_pin_to_node[data_pin];
                    LibertyCell* cell = get_cell(node_id);
                    const int port_offset = gtdb.pin_id2port_offset_id[data_pin];
                    if (!cell || port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size()))
                        continue;
                    LibertyPort* port = cell->ports_[port_offset];
                    if (!port || port->name != "D") continue;
                }
                if (data_pin >= 0 && data_pin < n) {
                    matched = true;
                    if (data_pins) data_pins->push_back(data_pin);
                }
            }
            return matched;
        };
        for (int seq_id = 0; seq_id < static_cast<int>(h_seqs.size()); ++seq_id) {
            const auto& seq = h_seqs[seq_id];
            std::vector<int> data_pins;
            const bool q_feedback = collect_feedback_data_pins(seq.data_expr_id, seq.q_pin, &data_pins);
            if (q_feedback && seed_seq_feedback_outputs && !seed_seen[seq.q_pin]) {
                add_seed_pin(seq.q_pin, "seq_feedback_q");
                seed_seen[seq.q_pin] = 1;
                root_seq_feedback_count++;
            }
            const bool qn_feedback = collect_feedback_data_pins(seq.data_expr_id, seq.qn_pin, &data_pins);
            if (qn_feedback && seed_seq_feedback_outputs && !seed_seen[seq.qn_pin]) {
                add_seed_pin(seq.qn_pin, "seq_feedback_qn");
                seed_seen[seq.qn_pin] = 1;
                root_seq_feedback_count++;
            }
            if (init_seq_feedback_state && (q_feedback || qn_feedback)) {
                if (!state_seq_seen[seq_id]) {
                    h_feedback_seed_seqs.push_back(seq_id);
                    state_seq_seen[seq_id] = 1;
                    state_seq_feedback_count++;
                }
                for (int data_pin : data_pins) {
                    if (data_pin >= 0 && data_pin < n && !state_pin_seen[data_pin]) {
                        h_feedback_seed_pins.push_back(data_pin);
                        state_pin_seen[data_pin] = 1;
                    }
                }
            }
        }
    }
    // Constant-generator outputs are roots in OpenSTA's power graph.
    for (int pin_id = 0; pin_id < n; pin_id++) {
        if (!h_is_driver_pin[pin_id] || h_is_primary_input[pin_id] || h_is_clock_pin[pin_id]) continue;
        int node_id = h_pin_to_node[pin_id];
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
        bool has_input_pin = false;
        for (int node_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
            if (node_pin >= 0 && node_pin < n && h_is_load_pin[node_pin]) {
                has_input_pin = true;
                break;
            }
        }
        if (seed_default_inputs && !has_input_pin) {
            add_seed_pin(pin_id, "const_output");
            root_const_output_count++;
        }
    }
    {
        std::vector<uint8_t> root_seen(n, 0);
        std::vector<int> ordered_roots;
        ordered_roots.reserve(h_primary_inputs.size());
        for (int pin_id : h_primary_inputs) {
            if (pin_id < 0 || pin_id >= n || root_seen[pin_id]) continue;
            root_seen[pin_id] = 1;
            ordered_roots.push_back(pin_id);
        }
        h_primary_inputs.swap(ordered_roots);
    }
    auto positive_unate_for_power = [](LibertyCell* cell, LibertyPort* from, LibertyPort* to) -> bool {
        if (!cell || !from || !to) return true;
        for (TimingArc* arc : from->timing_arcs_) {
            if (!arc || arc->to_port_ != to) continue;
            return arc->timing_sense_ == TimingSense::positive_unate ||
                   arc->timing_sense_ == TimingSense::non_unate ||
                   arc->timing_sense_ == TimingSense::unknown;
        }
        for (TimingArc* arc : to->timing_arcs_) {
            if (!arc || arc->from_port_ != from) continue;
            return arc->timing_sense_ == TimingSense::positive_unate ||
                   arc->timing_sense_ == TimingSense::non_unate ||
                   arc->timing_sense_ == TimingSense::unknown;
        }
        return true;
    };

    auto compile_when_expr = [&](InternalPower* ip, LibertyCell* cell, const gp::GPNode& node) -> int {
        if (!ip || ip->when_expr_.empty()) return -1;
        return add_template_expr(ip->when_expr_, cell);
    };

    std::vector<GpuPowerInternalHost> h_internal_rows;
    const char* debug_power_node_env = std::getenv("XPLACE_POWER_DEBUG_NODE");
    std::unordered_map<std::string, int> internal_denom_group;
    auto get_denom_group = [&](int to_pin, const std::string& related_pg) -> int {
        std::string key = std::to_string(to_pin) + "|" + related_pg;
        auto it = internal_denom_group.find(key);
        if (it != internal_denom_group.end()) return it->second;
        int id = static_cast<int>(internal_denom_group.size());
        internal_denom_group.emplace(std::move(key), id);
        return id;
    };

    if (need_internal_power) {
        for (const auto& node : gtdb.gpdb.getNodes()) {
            const int node_id = static_cast<int>(node.getId());
            LibertyCell* cell = get_cell(node_id);
            if (!cell || node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) continue;
            const int libcell_id = gtdb.cell_node_type_map[node_id];
            if (libcell_id < 0 || libcell_id + 1 >= static_cast<int>(gtdb.liberty_cell_type2port_list_end.size())) continue;
            const int port_base = gtdb.liberty_cell_type2port_list_end[libcell_id];
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port) continue;
                const int port_global = port_base + port_offset;
                const int range_idx = port_global * 2 + static_cast<int>(MAX);
                if (range_idx + 1 >= static_cast<int>(gtdb.liberty_port2internal_power_list_end.size())) continue;
                const int ip_start = gtdb.liberty_port2internal_power_list_end[range_idx];
                const int ip_end = gtdb.liberty_port2internal_power_list_end[range_idx + 1];
                if (ip_start == ip_end) continue;

                if (h_is_load_pin[pin_id]) {
                    for (int ip_id = ip_start; ip_id < ip_end; ++ip_id) {
                        InternalPower* ip = gtdb.liberty_internal_powers[ip_id];
                        if (!ip) continue;
                        GpuPowerInternalHost row;
                        row.internal_power_id = ip_id;
                        row.node_id = node_id;
                        row.to_pin = pin_id;
                        row.kind = 0;
                        row.energy_unit = ip->energy_unit_;
                        row.duty_mode = 0;
                        int when_expr_id = compile_when_expr(ip, cell, node);
                        if (when_expr_id >= 0) {
                            row.duty_mode = 1;
                            row.duty_expr_id = when_expr_id;
                            for (int op_i = h_expr_start[when_expr_id]; op_i < h_expr_start[when_expr_id] + h_expr_count[when_expr_id]; ++op_i) {
                                int out_pin = h_expr_ops[op_i].op == 0 ? h_expr_ops[op_i].arg : -1;
                                if (out_pin < -1) {
                                    const int port_id = -2 - out_pin;
                                    if (port_id >= 0 && port_id < static_cast<int>(cell->ports_.size())) {
                                        auto pin_itr = node.portMap.find(cell->ports_[port_id]->name);
                                        out_pin = pin_itr == node.portMap.end() ? -1 : pin_itr->second;
                                    }
                                }
                                if (out_pin >= 0 && out_pin < n && h_is_driver_pin[out_pin]) {
                                    const int func_expr_id = h_pin_func_expr_id[out_pin];
                                    if (expr_contains_pin(func_expr_id, pin_id)) {
                                        row.duty_mode = 2;
                                        row.duty_expr_id = func_expr_id;
                                        row.duty_pin = pin_id;
                                        break;
                                    }
                                }
                            }
                        }
                        if (debug_power_node_env && node.getName().find(debug_power_node_env) != std::string::npos) {
                            std::fprintf(stderr,
                                         "[XPLACE_POWER_DEBUG_NODE] node=%s port=%s kind=input ip=%d when='%s' duty_mode=%d duty_expr=%d\n",
                                         node.getName().c_str(),
                                         port->name.c_str(),
                                         ip_id,
                                         ip->when_expr_.c_str(),
                                         row.duty_mode,
                                         row.duty_expr_id);
                        }
                        h_internal_rows.push_back(row);
                    }
                }

                if (h_is_driver_pin[pin_id]) {
                    const int func_expr_id = h_pin_func_expr_id[pin_id];
                    for (int ip_id = ip_start; ip_id < ip_end; ++ip_id) {
                        InternalPower* ip = gtdb.liberty_internal_powers[ip_id];
                        if (!ip) continue;
                        GpuPowerInternalHost row;
                        row.internal_power_id = ip_id;
                        row.node_id = node_id;
                        row.to_pin = pin_id;
                        row.kind = 1;
                        row.energy_unit = ip->energy_unit_;
                        row.duty_mode = 4;
                        LibertyPort* from_port = ip->related_port_;
                        if (from_port && node.portMap.find(from_port->name) != node.portMap.end()) {
                            row.from_pin = node.portMap.at(from_port->name);
                            row.positive_unate = positive_unate_for_power(cell, from_port, port) ? 1 : 0;
                            const int when_expr_id = compile_when_expr(ip, cell, node);
                            if (expr_contains_pin(func_expr_id, row.from_pin)) {
                                row.duty_mode = 2;
                                row.duty_expr_id = func_expr_id;
                                row.duty_pin = row.from_pin;
                            } else if (when_expr_id >= 0) {
                                row.duty_mode = 1;
                                row.duty_expr_id = when_expr_id;
                            } else {
                                row.duty_mode = 3;
                            }
                            const std::string pg = ip->related_pg_pin_ ? ip->related_pg_pin_->name : ip->related_pg_pin_name_;
                            row.denom_group = get_denom_group(pin_id, pg);
                        }
                        if (debug_power_node_env && node.getName().find(debug_power_node_env) != std::string::npos) {
                            std::fprintf(stderr,
                                         "[XPLACE_POWER_DEBUG_NODE] node=%s port=%s kind=output ip=%d related=%s when='%s' duty_mode=%d duty_expr=%d from_pin=%d\n",
                                         node.getName().c_str(),
                                         port->name.c_str(),
                                         ip_id,
                                         ip->related_port_name_.c_str(),
                                         ip->when_expr_.c_str(),
                                         row.duty_mode,
                                         row.duty_expr_id,
                                         row.from_pin);
                        }
                        h_internal_rows.push_back(row);
                    }
                }
            }
        }
    }

    const float max_power_unit = (gtdb.cell_libs_[MAX] && gtdb.cell_libs_[MAX]->power_unit_.has_value())
        ? static_cast<float>(gtdb.cell_libs_[MAX]->power_unit_->value()) : 1.0f;
    std::vector<GpuPowerLeakageRowHost> h_leakage_rows;
    std::vector<GpuPowerLeakageGroupHost> h_leakage_groups;
    std::unordered_map<std::string, int> leakage_group_map;
    auto get_leakage_group = [&](int node_id, const std::string& pg, float cell_leakage_w) -> int {
        std::string key = std::to_string(node_id) + "|" + pg;
        auto it = leakage_group_map.find(key);
        if (it != leakage_group_map.end()) return it->second;
        GpuPowerLeakageGroupHost group;
        group.node_id = node_id;
        group.cell_leakage = cell_leakage_w;
        int id = static_cast<int>(h_leakage_groups.size());
        h_leakage_groups.push_back(group);
        leakage_group_map.emplace(std::move(key), id);
        return id;
    };
    if (need_leakage_power) {
        for (const auto& node : gtdb.gpdb.getNodes()) {
            const int node_id = static_cast<int>(node.getId());
            LibertyCell* cell = get_cell(node_id);
            if (!cell || node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) continue;
            const int libcell_id = gtdb.cell_node_type_map[node_id];
            if (libcell_id < 0 || libcell_id * 2 + static_cast<int>(MAX) + 1 >= static_cast<int>(gtdb.liberty_cell_type2leakage_power_list_end.size())) continue;
            const int leak_range_idx = libcell_id * 2 + static_cast<int>(MAX);
            const int leak_start = gtdb.liberty_cell_type2leakage_power_list_end[leak_range_idx];
            const int leak_end = gtdb.liberty_cell_type2leakage_power_list_end[leak_range_idx + 1];
            // OpenSTA uses scene_cell(max) for leakage_power groups, but the
            // default/cell_leakage fallback comes from the original cell pointer.
            // In this Xplace setup that corresponds to the MIN/early Liberty view.
            LibertyCell* cell_leakage_cell = (gtdb.cell_libs_[MIN] ? gtdb.cell_libs_[MIN]->get_cell(cell->name) : nullptr);
            if (!cell_leakage_cell) cell_leakage_cell = cell;
            LibertyCell* leak_expr_cell = (gtdb.cell_libs_[MAX] ? gtdb.cell_libs_[MAX]->get_cell(cell->name) : nullptr);
            if (!leak_expr_cell) leak_expr_cell = cell;
            const float cell_leakage_w = cell_leakage_cell->leakage_power_.value_or(0.0f) * max_power_unit;
            if (leak_start == leak_end) {
                get_leakage_group(node_id, "", cell_leakage_w);
                continue;
            }
            for (int leak_id = leak_start; leak_id < leak_end; ++leak_id) {
                LeakagePower* lp = gtdb.liberty_leakage_powers[leak_id];
                if (!lp) continue;
                const std::string pg = lp->related_pg_pin_ ? lp->related_pg_pin_->name : lp->related_pg_pin_name_;
                const int group_id = get_leakage_group(node_id, pg, cell_leakage_w);
                GpuPowerLeakageRowHost row;
                row.node_id = node_id;
                row.group_id = group_id;
                row.leakage_power_id = leak_id;
                row.when_expr_id = lp->when_expr_.empty() ? -1 : add_template_expr(lp->when_expr_, leak_expr_cell);
                row.leakage = lp->value_ * max_power_unit;
                h_leakage_rows.push_back(row);
            }
        }
    }

    auto iopt_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto i64opt_cpu = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    if (internal_row_meta_cpu) {
        std::vector<int64_t> meta;
        meta.reserve(h_internal_rows.size() * 6);
        for (const auto& row : h_internal_rows) {
            meta.push_back(row.node_id);
            meta.push_back(row.to_pin);
            meta.push_back(row.from_pin);
            meta.push_back(row.kind);
            meta.push_back(row.internal_power_id);
            meta.push_back(row.duty_mode);
        }
        if (h_internal_rows.empty()) *internal_row_meta_cpu = torch::empty({0, 6}, i64opt_cpu);
        else *internal_row_meta_cpu = torch::from_blob(meta.data(), {(long)h_internal_rows.size(), 6}, i64opt_cpu).clone();
    }
    if (leakage_row_meta_cpu) {
        std::vector<int64_t> meta;
        meta.reserve(h_leakage_rows.size() * 4);
        for (const auto& row : h_leakage_rows) {
            meta.push_back(row.node_id);
            meta.push_back(row.group_id);
            meta.push_back(row.leakage_power_id);
            meta.push_back(row.when_expr_id);
        }
        if (h_leakage_rows.empty()) *leakage_row_meta_cpu = torch::empty({0, 4}, i64opt_cpu);
        else *leakage_row_meta_cpu = torch::from_blob(meta.data(), {(long)h_leakage_rows.size(), 4}, i64opt_cpu).clone();
    }
    auto bopt_cpu = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto fopt_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto iopt_cuda = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto to_cuda_int = [&](const std::vector<int>& v) {
        std::vector<int> tmp = v.empty() ? std::vector<int>{0} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, iopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_index = [&](const std::vector<index_type>& v) {
        std::vector<index_type> tmp = v.empty() ? std::vector<index_type>{0} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, iopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_u8 = [&](const std::vector<uint8_t>& v) {
        std::vector<uint8_t> tmp = v.empty() ? std::vector<uint8_t>{0} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, bopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_float = [&](const std::vector<float>& v) {
        auto fopt_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        std::vector<float> tmp = v.empty() ? std::vector<float>{nanf("")} : v;
        return torch::from_blob(tmp.data(), {(long)tmp.size()}, fopt_cpu).clone().to(torch::kCUDA);
    };
    auto to_cuda_bytes = [&](const auto& v) {
        using VecT = std::decay_t<decltype(v)>;
        using ElemT = typename VecT::value_type;
        std::vector<ElemT> tmp = v.empty() ? std::vector<ElemT>(1) : v;
        auto cpu = torch::from_blob(reinterpret_cast<uint8_t*>(tmp.data()), {(long)(tmp.size() * sizeof(ElemT))}, bopt_cpu).clone();
        return cpu.to(torch::kCUDA);
    };
    auto to_cuda_bytes_range = [&](const auto& v, size_t begin, size_t count) {
        using VecT = std::decay_t<decltype(v)>;
        using ElemT = typename VecT::value_type;
        if (count == 0) {
            std::vector<ElemT> tmp(1);
            auto cpu = torch::from_blob(reinterpret_cast<uint8_t*>(tmp.data()), {(long)sizeof(ElemT)}, bopt_cpu).clone();
            return cpu.to(torch::kCUDA);
        }
        auto* data = const_cast<ElemT*>(v.data() + begin);
        auto cpu = torch::from_blob(reinterpret_cast<uint8_t*>(data), {(long)(count * sizeof(ElemT))}, bopt_cpu).clone();
        return cpu.to(torch::kCUDA);
    };
    const bool upload_debug = readPowerBoolEnv("XPLACE_POWER_UPLOAD_DEBUG", false);
    const bool upload_sync_debug = readPowerBoolEnv("XPLACE_POWER_UPLOAD_SYNC_DEBUG", false);
    auto power_upload_mark = [&](const char* phase, const char* label, size_t count, size_t elem_size) {
        if (upload_debug) {
            std::fprintf(stderr,
                         "[power_upload] %s %s count=%zu bytes=%zu\n",
                         phase,
                         label ? label : "",
                         count,
                         count * elem_size);
        }
        if (upload_sync_debug) {
            const std::string sync_label =
                std::string("upload ") + (phase ? phase : "") + " " + (label ? label : "");
            check_power_cuda_error(sync_label.c_str());
        }
    };
    auto upload_cuda_int = [&](const char* label, const std::vector<int>& v) {
        power_upload_mark("begin", label, v.size(), sizeof(int));
        auto out = to_cuda_int(v);
        power_upload_mark("end", label, v.size(), sizeof(int));
        return out;
    };
    auto upload_cuda_index = [&](const char* label, const std::vector<index_type>& v) {
        power_upload_mark("begin", label, v.size(), sizeof(index_type));
        auto out = to_cuda_index(v);
        power_upload_mark("end", label, v.size(), sizeof(index_type));
        return out;
    };
    auto upload_cuda_u8 = [&](const char* label, const std::vector<uint8_t>& v) {
        power_upload_mark("begin", label, v.size(), sizeof(uint8_t));
        auto out = to_cuda_u8(v);
        power_upload_mark("end", label, v.size(), sizeof(uint8_t));
        return out;
    };
    auto upload_cuda_float = [&](const char* label, const std::vector<float>& v) {
        power_upload_mark("begin", label, v.size(), sizeof(float));
        auto out = to_cuda_float(v);
        power_upload_mark("end", label, v.size(), sizeof(float));
        return out;
    };
    auto upload_cuda_bytes = [&](const char* label, const auto& v) {
        using VecT = std::decay_t<decltype(v)>;
        using ElemT = typename VecT::value_type;
        power_upload_mark("begin", label, v.size(), sizeof(ElemT));
        auto out = to_cuda_bytes(v);
        power_upload_mark("end", label, v.size(), sizeof(ElemT));
        return out;
    };
    auto read_chunk_bytes = [](const char* env_name, size_t default_value) {
        const char* env = std::getenv(env_name);
        if (!env || env[0] == '\0') return default_value;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end == env || parsed == 0) return default_value;
        return static_cast<size_t>(parsed);
    };
    constexpr size_t default_power_row_chunk_bytes = 512ull * 1024ull * 1024ull;
    const size_t internal_row_bytes = h_internal_rows.size() * sizeof(GpuPowerInternalHost);
    const size_t leakage_row_bytes = h_leakage_rows.size() * sizeof(GpuPowerLeakageRowHost);
    const size_t internal_chunk_bytes = read_chunk_bytes("XPLACE_POWER_INTERNAL_ROW_CHUNK_BYTES",
                                                         read_chunk_bytes("XPLACE_POWER_ROW_CHUNK_BYTES",
                                                                          default_power_row_chunk_bytes));
    const size_t leakage_chunk_bytes = read_chunk_bytes("XPLACE_POWER_LEAKAGE_ROW_CHUNK_BYTES",
                                                        read_chunk_bytes("XPLACE_POWER_ROW_CHUNK_BYTES",
                                                                         default_power_row_chunk_bytes));
    const bool chunk_internal_rows =
        need_internal_power && !h_internal_rows.empty() && internal_row_bytes > internal_chunk_bytes;
    const bool chunk_leakage_rows =
        need_leakage_power && !h_leakage_rows.empty() && leakage_row_bytes > leakage_chunk_bytes;
    if (chunk_internal_rows || chunk_leakage_rows || std::getenv("XPLACE_POWER_PRINT_ROW_STATS")) {
        std::fprintf(stderr,
                     "[power_row_stats] internal_rows=%zu internal_bytes=%zu internal_chunk=%zu chunk_internal=%d "
                     "denom_groups=%zu leakage_rows=%zu leakage_bytes=%zu leakage_chunk=%zu chunk_leakage=%d "
                     "leakage_groups=%zu expr_ops=%zu expr_bytes=%zu expr_cache=%zu\n",
                     h_internal_rows.size(), internal_row_bytes, internal_chunk_bytes,
                     chunk_internal_rows ? 1 : 0, internal_denom_group.size(),
                     h_leakage_rows.size(), leakage_row_bytes, leakage_chunk_bytes,
                     chunk_leakage_rows ? 1 : 0, h_leakage_groups.size(),
                     h_expr_ops.size(), h_expr_ops.size() * sizeof(GpuPowerExprOpHost),
                     template_expr_cache.size());
    }
    std::vector<int> h_node_port_pin_start;
    std::vector<int> h_node_port_pin_list;
    if (need_internal_power || need_leakage_power) {
        const int node_count = static_cast<int>(gtdb.gpdb.getNodes().size());
        h_node_port_pin_start.assign(node_count + 1, 0);
        for (const auto& node : gtdb.gpdb.getNodes()) {
            const int node_id = static_cast<int>(node.getId());
            LibertyCell* cell = get_cell(node_id);
            const int port_count = cell ? static_cast<int>(cell->ports_.size()) : 0;
            if (node_id >= 0 && node_id < node_count)
                h_node_port_pin_start[node_id + 1] = port_count;
        }
        for (int node_id = 0; node_id < node_count; ++node_id)
            h_node_port_pin_start[node_id + 1] += h_node_port_pin_start[node_id];
        h_node_port_pin_list.assign(h_node_port_pin_start.back(), -1);
        for (const auto& node : gtdb.gpdb.getNodes()) {
            const int node_id = static_cast<int>(node.getId());
            LibertyCell* cell = get_cell(node_id);
            if (!cell || node_id < 0 || node_id >= node_count) continue;
            const int start = h_node_port_pin_start[node_id];
            const int end = h_node_port_pin_start[node_id + 1];
            for (int port_id = 0; port_id < static_cast<int>(cell->ports_.size()) && start + port_id < end; ++port_id) {
                auto pin_itr = node.portMap.find(cell->ports_[port_id]->name);
                if (pin_itr != node.portMap.end()) h_node_port_pin_list[start + port_id] = pin_itr->second;
            }
        }
