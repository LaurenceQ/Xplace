torch::Tensor GPUTimer::report_power_activity_cpu() {
    const int n = static_cast<int>(gtdb.pin_names.size());
    std::vector<CpuActivity> act(n);
    std::vector<CpuActivity> seq_pin_activity(n);
    std::vector<uint8_t> seq_pin_activity_valid(n, 0);
    std::vector<uint8_t> clock_activity_protected(n, 0);

    const double sdc_time_scale =
        canonicalPowerTimeScale(gtdb.sdc_time_unit.has_value() ? *gtdb.sdc_time_unit : gtdb.time_unit);
    double min_period_sec = std::numeric_limits<double>::infinity();
    for (const auto& kv : gtdb.clocks) {
        const float period = kv.second.period();
        const double period_sec = static_cast<double>(period) * sdc_time_scale;
        if (period_sec > 0.0) min_period_sec = std::min(min_period_sec, period_sec);
    }
    if (!std::isfinite(min_period_sec) || min_period_sec <= 0.0) {
        const double fallback_scale = canonicalPowerTimeScale(gtdb.time_unit);
        min_period_sec = fallback_scale > 0.0 ? fallback_scale : 1.0e-9;
    }
    const float default_density = static_cast<float>(0.1 / min_period_sec);
    const float clock_density = static_cast<float>(2.0 / min_period_sec);
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

    std::vector<int> pin_level(n, 0);
    int max_pin_level = 0;
    if (gtdb.pin_num_fanin.size() == static_cast<size_t>(n)
        && gtdb.pin_fanout_list_end.size() == static_cast<size_t>(n + 1)) {
        std::vector<int> indeg = gtdb.pin_num_fanin;
        std::deque<int> frontier;
        for (int i = 0; i < n; i++) {
            if (indeg[i] == 0) frontier.push_back(i);
        }
        std::vector<uint8_t> seen(n, 0);
        while (!frontier.empty()) {
            int pin_id = frontier.front();
            frontier.pop_front();
            if (pin_id < 0 || pin_id >= n || seen[pin_id]) continue;
            seen[pin_id] = 1;
            max_pin_level = std::max(max_pin_level, pin_level[pin_id]);
            int start = gtdb.pin_fanout_list_end[pin_id];
            int end = gtdb.pin_fanout_list_end[pin_id + 1];
            for (int idx = start; idx < end; idx++) {
                int fanout = gtdb.pin_fanout_list[idx];
                if (fanout < 0 || fanout >= n) continue;
                pin_level[fanout] = std::max(pin_level[fanout], pin_level[pin_id] + 1);
                if (--indeg[fanout] == 0) frontier.push_back(fanout);
            }
        }
    }
    std::vector<std::deque<int>> level_queues(max_pin_level + 2);
    std::set<int> nonempty_queue_levels;
    std::vector<uint8_t> in_queue(n, 0);
    std::vector<uint8_t> force_propagate_on_visit(n, 0);
    std::vector<int> pin_to_node(n, -1);
    std::vector<int> pin_to_net(n, -1);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        int pin_id = static_cast<int>(pin.getId());
        if (pin_id >= 0 && pin_id < n) {
            pin_to_node[pin_id] = static_cast<int>(pin.getParNodeId());
            pin_to_net[pin_id] = static_cast<int>(pin.getParNetId());
        }
    }

    const char* trace_path_file_env = std::getenv("XPLACE_POWER_TRACE_PATH_FILE");
    const char* trace_path_out_env = std::getenv("XPLACE_POWER_TRACE_PATH_OUT");
    const char* activity_path_trace_env = std::getenv("XPLACE_POWER_ACTIVITY_PATH_TRACE_FILE");
    PowerTracePathState path_trace =
        loadPowerTracePathState(trace_path_file_env, activity_path_trace_env, pin_to_node);
    int path_trace_pass = 0;
    std::string path_trace_level_tag = "seed";
    auto path_trace_hit = [&](int pin_id, int arc_id) -> bool {
        return path_trace.enabled()
            && ((pin_id >= 0 && path_trace.pins.count(pin_id))
                || (arc_id != -1 && path_trace.arcs.count(arc_id)));
    };
    auto emit_path_trace = [&](const char* event,
                               int arc_id,
                               int from_pin,
                               int to_pin,
                               float old_density,
                               float old_duty,
                               float new_density,
                               float new_duty,
                               bool changed,
                               bool enqueued,
                               int pending_seq,
                               const char* reason) {
        if (!path_trace.enabled()) return;
        if (!path_trace_hit(from_pin, arc_id) && !path_trace_hit(to_pin, arc_id)) return;
        path_trace.out << "xplace_cpu\t" << path_trace_pass << '\t'
                       << path_trace_level_tag << '\t' << (event ? event : "") << '\t'
                       << arc_id << '\t'
                       << from_pin << '\t'
                       << ((from_pin >= 0 && from_pin < n) ? gtdb.pin_names[from_pin] : "") << '\t'
                       << to_pin << '\t'
                       << ((to_pin >= 0 && to_pin < n) ? gtdb.pin_names[to_pin] : "") << '\t'
                       << old_density << '\t' << old_duty << '\t'
                       << new_density << '\t' << new_duty << '\t'
                       << (changed ? 1 : 0) << '\t'
                       << (enqueued ? 1 : 0) << '\t'
                       << pending_seq << '\t'
                       << (reason ? reason : "") << '\n';
    };

    auto enqueue = [&](int pin_id,
                       bool force_propagate = false,
                       int from_pin = -1,
                       int arc_id = -1,
                       const char* reason = "enqueue") {
        if (pin_id < 0 || pin_id >= n) return;
        if (force_propagate) force_propagate_on_visit[pin_id] = 1;
        if (in_queue[pin_id]) {
            emit_path_trace("enqueue_skip_queued", arc_id, from_pin, pin_id,
                            act[pin_id].density, act[pin_id].duty,
                            act[pin_id].density, act[pin_id].duty,
                            false, false, 0, reason);
            return;
        }
        int level = std::clamp(pin_level[pin_id], 0, max_pin_level + 1);
        if (level_queues[level].empty()) nonempty_queue_levels.insert(level);
        level_queues[level].push_back(pin_id);
        in_queue[pin_id] = 1;
        emit_path_trace("enqueue", arc_id, from_pin, pin_id,
                        act[pin_id].density, act[pin_id].duty,
                        act[pin_id].density, act[pin_id].duty,
                        false, true, 0, reason);
    };

    auto percent_change = [](float value, float prev) -> float {
        if (prev == 0.0f) return value == 0.0f ? 0.0f : 1.0f;
        return std::abs(value - prev) / std::abs(prev);
    };
    const float min_activity_density =
        std::max(0.0f, readPowerFloatEnv("XPLACE_POWER_MIN_ACTIVITY_DENSITY", 1.0e-10f));
    const float min_activity_duty =
        std::max(0.0f, readPowerFloatEnv("XPLACE_POWER_MIN_ACTIVITY_DUTY", 0.0f));
    const bool ignore_scan_enable_density =
        readPowerBoolEnv("XPLACE_POWER_IGNORE_SCAN_ENABLE_DENSITY", false);
    const bool require_known_seq_data =
        readPowerBoolEnv("XPLACE_POWER_REQUIRE_KNOWN_SEQ_DATA", false);
    const bool allow_clock_activity_override =
        readPowerBoolEnv("XPLACE_POWER_ALLOW_CLOCK_ACTIVITY_OVERRIDE", false);
    const bool disable_activity_slew_cap =
        readPowerBoolEnv("XPLACE_POWER_DISABLE_ACTIVITY_SLEW_CAP", false);
    const bool clamp_activity_to_clock_density =
        readPowerBoolEnv("XPLACE_POWER_CLAMP_ACTIVITY_TO_CLOCK_DENSITY", false);
    torch::Tensor pin_slew_cpu;
    const float* pin_slew_host = nullptr;
    if (!disable_activity_slew_cap && timing_raw_db.pinSlew.defined()
        && timing_raw_db.pinSlew.numel() >= static_cast<int64_t>(n) * NUM_ATTR) {
        pin_slew_cpu = timing_raw_db.pinSlew.to(torch::kCPU).contiguous();
        pin_slew_host = pin_slew_cpu.data_ptr<float>();
    }
    auto clock_slew_override = [&](int pin_id, int attr) -> float {
        if (pin_id < 0 || pin_id >= n) return nanf("");
        const bool is_ideal_clock =
            pin_id < static_cast<int>(gtdb.pin_is_ideal_clk.size()) &&
            gtdb.pin_is_ideal_clk[pin_id];
        const bool is_clock_pin =
            pin_id < static_cast<int>(gtdb.pin_is_clk.size()) &&
            gtdb.pin_is_clk[pin_id];
        if (!is_ideal_clock && !is_clock_pin) return nanf("");
        const int node_id = pin_to_node[pin_id];
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size()))
            return nanf("");
        const int libcell_id = gtdb.cell_node_type_map[node_id];
        if (libcell_id < 0 || libcell_id >= static_cast<int>(gtdb.rawdb.celltypes.size()))
            return nanf("");
        db::CellType* cell_type = gtdb.rawdb.celltypes[libcell_id];
        LibertyCell* cell = cell_type ? cell_type->liberty_cell : nullptr;
        if (!cell || cell->sequentials_.empty()) return nanf("");
        const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size()))
            return nanf("");
        LibertyPort* port = cell->ports_[port_offset];
        if (!port || !port->is_clock_) return nanf("");
        const int idx = pin_id * NUM_ATTR + attr;
        if (idx < 0 || idx >= static_cast<int>(gtdb.pin_clock_slews.size()))
            return nanf("");
        return gtdb.pin_clock_slews[idx];
    };
    auto max_activity_density_for_pin = [&](int pin_id) -> float {
        float max_density = std::numeric_limits<float>::infinity();
        if ((pin_slew_host || !gtdb.pin_clock_slews.empty()) && pin_id >= 0 && pin_id < n
            && gtdb.time_unit > 0.0f) {
            float min_rf_slew = std::numeric_limits<float>::infinity();
            for (int attr = 0; attr + 1 < NUM_ATTR; attr += 2) {
                float rise = pin_slew_host ? pin_slew_host[pin_id * NUM_ATTR + attr] : nanf("");
                float fall = pin_slew_host ? pin_slew_host[pin_id * NUM_ATTR + attr + 1] : nanf("");
                const float clock_rise = clock_slew_override(pin_id, attr);
                const float clock_fall = clock_slew_override(pin_id, attr + 1);
                if (std::isfinite(clock_rise) && std::isfinite(clock_fall)) {
                    rise = clock_rise;
                    fall = clock_fall;
                }
                if (!std::isfinite(rise) || !std::isfinite(fall)) continue;
                const float avg_slew = 0.5f * (rise + fall) * gtdb.time_unit;
                if (avg_slew > 0.0f && avg_slew < min_rf_slew)
                    min_rf_slew = avg_slew;
            }
            if (std::isfinite(min_rf_slew) && min_rf_slew > 0.0f)
                max_density = 1.0f / min_rf_slew;
        }
        if (clamp_activity_to_clock_density)
            max_density = std::min(max_density, clock_density);
        return max_density;
    };

    std::vector<int> trace_pin_ids = resolvePowerTracePins(readPowerTracePinQueries(), gtdb.pin_names);
    std::vector<uint8_t> trace_first_seen(n, 0);
    auto trace_matches = [&](int pin_id) -> bool {
        return pin_id >= 0 && pin_id < n &&
            std::find(trace_pin_ids.begin(), trace_pin_ids.end(), pin_id) != trace_pin_ids.end();
    };

    auto set_activity = [&](int pin_id, float density, float duty, int origin, bool force, bool enqueue_on_change = true) -> bool {
        if (pin_id < 0 || pin_id >= n) return false;
        if (!force && !allow_clock_activity_override && clock_activity_protected[pin_id]
            && act[pin_id].origin == 2 && origin != 2)
            return false;
        const float prev_density = act[pin_id].density;
        const float prev_duty = act[pin_id].duty;
        const int prev_origin = act[pin_id].origin;
        float duty_clamped = std::clamp(duty, 0.0f, 1.0f);
        if (min_activity_duty > 0.0f) {
            if (duty_clamped < min_activity_duty)
                duty_clamped = 0.0f;
            else if ((1.0f - duty_clamped) < min_activity_duty)
                duty_clamped = 1.0f;
        }
        float density_clamped = std::max(density, 0.0f);
        const float max_density = force ? std::numeric_limits<float>::infinity()
                                        : max_activity_density_for_pin(pin_id);
        if (std::isfinite(max_density))
            density_clamped = std::min(density_clamped, max_density);
        // Match OpenSTA PwrActivity::check() by default; the env override is
        // for diagnosing tiny feedback noise that can be amplified by loops.
        if (std::abs(density_clamped) < min_activity_density) density_clamped = 0.0f;
        const bool value_changed = percent_change(density_clamped, prev_density) > 0.01f
            || percent_change(duty_clamped, prev_duty) > 0.01f;
        const bool changed = value_changed || prev_origin != origin;
        act[pin_id].density = density_clamped;
        act[pin_id].duty = duty_clamped;
        act[pin_id].origin = origin;
        if (trace_matches(pin_id)) {
            std::cerr << "[power_activity_trace_set] pin=" << gtdb.pin_names[pin_id]
                      << " level=" << pin_level[pin_id]
                      << " prev_density=" << prev_density
                      << " prev_duty=" << prev_duty
                      << " density=" << density_clamped
                      << " duty=" << duty_clamped
                      << " origin=" << origin
                      << " changed=" << changed
                      << " enqueue=" << (changed && enqueue_on_change)
                      << std::endl;
        }
        emit_path_trace("set_activity", -1, -1, pin_id,
                        prev_density, prev_duty,
                        density_clamped, duty_clamped,
                        changed, changed && enqueue_on_change, 0,
                        force ? "force" : "activity");
        if (changed && enqueue_on_change) enqueue(pin_id, true, pin_id, -1, "set_activity_revisit");
        return changed;
    };

    auto normalize_expr = [](std::string expr) {
        expr.erase(std::remove_if(expr.begin(), expr.end(), [](unsigned char c) { return std::isspace(c); }), expr.end());
        if (expr.size() >= 2 && expr.front() == '"' && expr.back() == '"')
            expr = expr.substr(1, expr.size() - 2);
        return expr;
    };

    std::vector<uint8_t> pending_reg_flag(gtdb.gpdb.getNodes().size(), 0);
    std::vector<int> pending_regs;
    std::vector<int> pending_reg_trigger_pin(gtdb.gpdb.getNodes().size(), -1);
    auto mark_pending_reg = [&](int node_id, int trigger_pin) {
        if (node_id < 0 || node_id >= static_cast<int>(pending_reg_flag.size())) return;
        if (pending_reg_flag[node_id]) return;
        pending_reg_flag[node_id] = 1;
        pending_reg_trigger_pin[node_id] = trigger_pin;
        pending_regs.push_back(node_id);
        if (path_trace.enabled() && path_trace.nodes.count(node_id)) {
            int node_trace_pin = -1;
            for (int pin_id : path_trace.pins) {
                if (pin_id >= 0 && pin_id < static_cast<int>(pin_to_node.size())
                    && pin_to_node[pin_id] == node_id) {
                    node_trace_pin = pin_id;
                    break;
                }
            }
            emit_path_trace("seq_pending", -1, node_trace_pin, node_trace_pin,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            true, false, static_cast<int>(pending_regs.size()),
                            "load_pin_changed");
        }
    };

    auto get_cell = [&](int node_id) -> LibertyCell* {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) return nullptr;
        int libcell_id = gtdb.cell_node_type_map[node_id];
        if (libcell_id < 0 || libcell_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return nullptr;
        auto* cell_type = gtdb.rawdb.celltypes[libcell_id];
        return cell_type ? cell_type->liberty_cell : nullptr;
    };
    const bool mark_seq_clock_loads =
        readPowerBoolEnv("XPLACE_POWER_MARK_SEQ_CLOCK_LOADS", false);
    auto is_seq_clock_input_pin = [&](int pin_id) {
        if (pin_id < 0 || pin_id >= n) return false;
        const int node_id = pin_to_node[pin_id];
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) return false;
        int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) return false;
        LibertyPort* port = cell->ports_[port_offset];
        return port && port->is_clock_;
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
    std::vector<std::unordered_map<int, int>> node_const_port_values(gtdb.gpdb.getNodes().size());
    std::vector<uint8_t> node_const_port_values_ready(gtdb.gpdb.getNodes().size(), 0);
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
    auto const_port_values_for_node = [&](int node_id, LibertyCell* cell) -> const std::unordered_map<int, int>& {
        static const std::unordered_map<int, int> empty;
        if (!cell || node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return empty;
        if (node_const_port_values_ready[node_id]) return node_const_port_values[node_id];
        node_const_port_values_ready[node_id] = 1;
        const auto& node = gtdb.gpdb.getNodes()[node_id];
        load_const_port_file();
        for (int port_id = 0; port_id < static_cast<int>(cell->ports_.size()); port_id++) {
            LibertyPort* port = cell->ports_[port_id];
            if (!port || port->direction_ == CellPortDirection::output) continue;
            if (node.portMap.find(port->name) != node.portMap.end()) continue;
            const std::string key = normalizePowerActivitySnapshotName(node.getName()) + "/" +
                                    normalizePowerActivitySnapshotName(port->name);
            auto const_itr = const_port_file_values.find(key);
            if (const_itr != const_port_file_values.end())
                node_const_port_values[node_id][port_id] = const_itr->second;
        }
        const int raw_cell_id = static_cast<int>(node.getOriDBId());
        if (raw_cell_id < 0 || raw_cell_id >= static_cast<int>(gtdb.rawdb.cells.size()))
            return node_const_port_values[node_id];
        db::Cell* dbcell = gtdb.rawdb.cells[raw_cell_id];
        if (!dbcell) return node_const_port_values[node_id];
        for (int port_id = 0; port_id < static_cast<int>(cell->ports_.size()); port_id++) {
            LibertyPort* port = cell->ports_[port_id];
            if (!port || port->direction_ == CellPortDirection::output) continue;
            if (node.portMap.find(port->name) != node.portMap.end()) continue;
            db::Pin* dbpin = dbcell->pin(port->name);
            if (!dbpin || !dbpin->net) continue;
            const int value = parse_const_net_value(dbpin->net->name);
            if (value >= 0) node_const_port_values[node_id][port_id] = value;
        }
        return node_const_port_values[node_id];
    };
    auto is_io_node = [&](int node_id) -> bool {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return false;
        const std::string& node_type = gtdb.gpdb.getNodes()[node_id].getNodeType();
        return node_type == "IOPin" || node_type == "FloatIOPin";
    };

    std::vector<uint8_t> is_load_pin(n, 0);
    std::vector<uint8_t> is_driver_pin(n, 0);
    for (const auto& pin : gtdb.gpdb.getPins()) {
        int pin_id = static_cast<int>(pin.getId());
        if (pin_id < 0 || pin_id >= n) continue;
        int node_id = pin_to_node[pin_id];
        if (is_io_node(node_id)
            && std::find(gtdb.primary_inputs.begin(), gtdb.primary_inputs.end(), pin_id) != gtdb.primary_inputs.end()) {
            is_driver_pin[pin_id] = 1;
            continue;
        }
        if (is_io_node(node_id)
            && std::find(gtdb.primary_outputs.begin(), gtdb.primary_outputs.end(), pin_id) != gtdb.primary_outputs.end()) {
            is_load_pin[pin_id] = 1;
            continue;
        }
        LibertyCell* cell = get_cell(node_id);
        if (!cell) continue;
        int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
        LibertyPort* port = cell->ports_[port_offset];
        if (!port) continue;
        if (port->direction_ == CellPortDirection::input) is_load_pin[pin_id] = 1;
        else if (port->direction_ == CellPortDirection::output) is_driver_pin[pin_id] = 1;
    }
    std::vector<uint8_t> is_seq_output_pin(n, 0);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell || cell->sequentials_.empty()) continue;
        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            const std::string seq_out = normalize_expr(seq->output_name_);
            const std::string seq_out_inv = normalize_expr(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n || !is_driver_pin[pin_id]) continue;
                int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || !port->has_function_) continue;
                const std::string func = normalize_expr(port->function_expr_);
                if ((!seq_out.empty() && func == seq_out) ||
                    (!seq_out_inv.empty() && func == seq_out_inv)) {
                    is_seq_output_pin[pin_id] = 1;
                }
            }
        }
    }

    std::vector<int> net_driver_pin(gtdb.gpdb.getNets().size(), -1);
    for (const auto& net : gtdb.gpdb.getNets()) {
        const int net_id = static_cast<int>(net.getId());
        if (net_id < 0 || net_id >= static_cast<int>(net_driver_pin.size())) continue;
        for (int pin_id : net.pins()) {
            if (pin_id >= 0 && pin_id < n && is_driver_pin[pin_id]) {
                net_driver_pin[net_id] = pin_id;
                break;
            }
        }
    }
    std::vector<int> clock_gate_out_for_input(n, -1);
    std::vector<int> clock_gate_clock_for_out(n, -1);
    std::vector<int> clock_gate_enable_for_out(n, -1);
    std::vector<uint8_t> is_clock_gate_clock_pin(n, 0);
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
            clock_gate_clock_for_out[out_pin] = clk_pin;
            clock_gate_enable_for_out[out_pin] = enable_pin;
            clock_gate_out_for_input[clk_pin] = out_pin;
            clock_gate_out_for_input[enable_pin] = out_pin;
            is_clock_gate_clock_pin[clk_pin] = 1;
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
            if (is_clock_gate_clock_pin[pin_id]) mark_net(pin_to_net[pin_id]);
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
                if (out_pin < 0 || out_pin >= n || !is_driver_pin[out_pin]) continue;
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
                if (pin_id < 0 || pin_id >= n || !is_load_pin[pin_id]) continue;
                const int node_id = pin_to_node[pin_id];
                LibertyCell* cell = get_cell(node_id);
                if (!is_core_comb_node(node_id, cell)) continue;
                add_extra_clock_pin(pin_id);
                if (!is_clock_transparent_from_pin(node_id, cell, pin_id)) continue;
                for (int out_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
                    if (out_pin >= 0 && out_pin < n && is_driver_pin[out_pin])
                        mark_forward_net(pin_to_net[out_pin]);
                }
            }
        }
        std::vector<int> clock_pins;
        for (int net_id = 0; net_id < num_nets; net_id++) {
            if (!is_clock_net[net_id]) continue;
            for (int pin_id : gtdb.gpdb.getNets()[net_id].pins()) {
                if (pin_id >= 0 && pin_id < n
                    && (is_load_pin[pin_id] || is_io_node(pin_to_node[pin_id])))
                    clock_pins.push_back(pin_id);
            }
        }
        clock_pins.insert(clock_pins.end(), extra_clock_pins.begin(), extra_clock_pins.end());
        std::sort(clock_pins.begin(), clock_pins.end());
        clock_pins.erase(std::unique(clock_pins.begin(), clock_pins.end()), clock_pins.end());
        return clock_pins;
    };

    auto enqueue_adjacent_vertices = [&](int pin_id) {
        if (pin_id < 0 || pin_id >= n) return;
        if (is_driver_pin[pin_id]) {
            const int net_id = pin_to_net[pin_id];
            if (net_id >= 0 && net_id < static_cast<int>(gtdb.gpdb.getNets().size())) {
                for (int sink_pin : gtdb.gpdb.getNets()[net_id].pins()) {
                    if (sink_pin < 0 || sink_pin >= n || sink_pin == pin_id || !is_load_pin[sink_pin])
                        continue;
                    enqueue(sink_pin, false, pin_id, -1, "net_fanout");
                }
            }
        }
        if (gtdb.pin_forward_arc_list_end.size() != static_cast<size_t>(n + 1)) return;
        int start = gtdb.pin_forward_arc_list_end[pin_id];
        int end = gtdb.pin_forward_arc_list_end[pin_id + 1];
        for (int idx = start; idx < end; idx++) {
            int arc_id = gtdb.pin_forward_arc_list[idx];
            if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) continue;
            if (arc_id < static_cast<int>(gtdb.arc_id2test_id.size()) && gtdb.arc_id2test_id[arc_id] != -1) continue;  // timing checks.
            int to_pin = gtdb.timing_arc_to_pin_id[arc_id];
            if (std::getenv("XPLACE_POWER_ACTIVITY_SKIP_BACK_LEVEL_ARCS")
                && arc_id < static_cast<int>(gtdb.arc_types.size()) && gtdb.arc_types[arc_id] == 1
                && to_pin >= 0 && to_pin < n && pin_level[to_pin] <= pin_level[pin_id])
                continue;
            // OpenSTA ActivitySrchPred excludes reg clk->Q/latch D->Q. Xplace does not expose
            // TimingRole here, so conservatively skip cell arcs into sequential outputs; they
            // are propagated only by seed_reg_outputs().
            if (arc_id < static_cast<int>(gtdb.arc_types.size()) && gtdb.arc_types[arc_id] == 1) {
                int to_node = to_pin >= 0 && to_pin < n ? pin_to_node[to_pin] : -1;
                LibertyCell* to_cell = get_cell(to_node);
                if (to_cell && !to_cell->sequentials_.empty() && to_pin >= 0 && to_pin < n && is_driver_pin[to_pin])
                    continue;
            }
            enqueue(to_pin, false, pin_id, arc_id, "adjacent");
        }
    };

    auto expr_has_missing_node_port = [](const PowerExpr& expr,
                                         const LibertyCell* cell,
                                         const gp::GPNode& node) -> bool {
        if (!cell) return false;
        for (const auto& op : expr.ops()) {
            if (op.opcode != PowerExprOpcode::port || op.port_id < 0
                || op.port_id >= static_cast<int>(cell->ports_.size()))
                continue;
            const std::string& port_name = cell->ports_[op.port_id]->name;
            if (node.portMap.find(port_name) == node.portMap.end()) return true;
        }
        return false;
    };

    auto expr_has_known_activity_input = [&](const PowerExpr& expr,
                                             const LibertyCell* cell,
                                             const gp::GPNode& node,
                                             const std::unordered_map<int, int>* const_port_values,
                                             const std::unordered_set<int>* zero_density_ports = nullptr) -> bool {
        if (!cell) return false;
        bool has_port_ref = false;
        for (const auto& op : expr.ops()) {
            if (op.opcode != PowerExprOpcode::port || op.port_id < 0
                || op.port_id >= static_cast<int>(cell->ports_.size()))
                continue;
            has_port_ref = true;
            if (const_port_values && const_port_values->find(op.port_id) != const_port_values->end())
                return true;
            if (zero_density_ports && zero_density_ports->find(op.port_id) != zero_density_ports->end())
                return true;
            const std::string& port_name = cell->ports_[op.port_id]->name;
            auto pin_itr = node.portMap.find(port_name);
            if (pin_itr == node.portMap.end()) continue;
            const int pin_id = pin_itr->second;
            if (pin_id >= 0 && pin_id < n && act[pin_id].origin != 0) return true;
        }
        return !has_port_ref;
    };

    auto eval_cell_outputs = [&](int node_id, bool missing_port_outputs_only = false) {
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return;
        const auto& node = gtdb.gpdb.getNodes()[node_id];
        LibertyCell* cell = get_cell(node_id);
        if (!cell) return;
        const auto& const_port_values = const_port_values_for_node(node_id, cell);

        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;

            PowerExpr expr;
            if (!expr.compile(port->function_expr_, cell)) continue;
            const bool has_missing_port = expr_has_missing_node_port(expr, cell, node);
            if (missing_port_outputs_only && !has_missing_port) continue;
            float density = 0.0f;
            float duty = 0.0f;
            if (evalPowerExprActivity(expr, cell, node, act, density, duty, &const_port_values)) {
                set_activity(pin_id, density, duty, 3, false);
            } else if (missing_port_outputs_only && has_missing_port) {
                set_activity(pin_id, act[pin_id].density, act[pin_id].duty, 3, false);
            }
        }
    };

    auto eval_output_pin_activity = [&](int pin_id, bool& changed) -> bool {
        changed = false;
        int node_id = pin_id >= 0 && pin_id < n ? pin_to_node[pin_id] : -1;
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) return false;
        const auto& node = gtdb.gpdb.getNodes()[node_id];
        LibertyCell* cell = get_cell(node_id);
        if (!cell) return false;
        const auto& const_port_values = const_port_values_for_node(node_id, cell);
        int port_offset = gtdb.pin_id2port_offset_id[pin_id];
        if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) return false;
        LibertyPort* port = cell->ports_[port_offset];
        if (!port || port->direction_ != CellPortDirection::output) return false;
        bool computed = false;
        if (seq_pin_activity_valid[pin_id]) {
            const CpuActivity& seq_activity = seq_pin_activity[pin_id];
            changed = set_activity(pin_id, seq_activity.density, seq_activity.duty,
                                   seq_activity.origin, false, false);
            computed = true;
        }
