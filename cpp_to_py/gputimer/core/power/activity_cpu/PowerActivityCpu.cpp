#include "gputimer/core/GPUTimer.h"
#include "gputimer/core/DmpModel.h"
#include "gputimer/core/power/common/PowerCudaModel.h"
#include "gputimer/core/power/common/PowerHostCommon.h"
#include "gputimer/core/power/common/PowerActivityHostUtils.h"
#include "gputimer/core/power/activity_cpu/PowerActivityCpuDebug.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"
#include "common/lib/Liberty.h"
#include "common/lib/Lut.h"
#include "common/lib/Timing.h"
#include "gputimer/db/GTDatabase.h"
#include "io_parser/gp/GPDatabase.h"

#include <torch/cuda.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gt {

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

    PowerCpuActivityLevels cpu_activity_levels = buildPowerCpuActivityLevels(gtdb, n);
    std::vector<int> pin_level = std::move(cpu_activity_levels.pin_level);
    int max_pin_level = cpu_activity_levels.max_level;
    std::vector<std::deque<int>> level_queues(max_pin_level + 2);
    std::set<int> nonempty_queue_levels;
    std::vector<uint8_t> in_queue(n, 0);
    std::vector<uint8_t> force_propagate_on_visit(n, 0);
    std::vector<int> pin_to_node;
    std::vector<int> pin_to_net;
    buildPowerPinNodeNetMaps(gtdb, n, pin_to_node, pin_to_net);

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

    auto normalize_expr = [](std::string expr) { return normalizePowerExprString(expr); };

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

    auto get_cell = [&](int node_id) -> LibertyCell* { return powerCellForNode(gtdb, node_id); };
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
    auto parse_const_net_value = [](std::string name) -> int { return parsePowerConstNetValue(name); };
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
    auto is_io_node = [&](int node_id) -> bool { return powerIsIoNode(gtdb, node_id); };

    std::vector<uint8_t> is_load_pin;
    std::vector<uint8_t> is_driver_pin;
    classifyPowerPins(gtdb, n, pin_to_node, is_load_pin, is_driver_pin);

    std::vector<uint8_t> is_seq_output_pin;
    markPowerSeqOutputPins(gtdb, n, pin_to_node, is_driver_pin, is_seq_output_pin);

    std::vector<int> net_driver_pin;
    buildPowerNetDriverPins(gtdb, n, is_driver_pin, net_driver_pin);

    std::vector<int> clock_gate_out_for_input;
    std::vector<int> clock_gate_clock_for_out;
    std::vector<int> clock_gate_enable_for_out;
    std::vector<uint8_t> is_clock_gate_clock_pin;
    buildPowerClockGateMaps(gtdb, n, pin_to_node, clock_gate_out_for_input,
                             clock_gate_clock_for_out, clock_gate_enable_for_out,
                             is_clock_gate_clock_pin);

    auto build_clock_pins = [&]() {
        return buildPowerClockPins(gtdb, n, pin_to_node, pin_to_net,
                                   is_load_pin, is_driver_pin,
                                   is_clock_gate_clock_pin);
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
        auto [pin_density, pin_duty] = powerClockActivityForPin(gtdb, pin_id, sdc_time_scale, clock_density);
        const int node_id = pin_id >= 0 && pin_id < n ? pin_to_node[pin_id] : -1;
        LibertyCell* cell = get_cell(node_id);
        const bool enqueue_clock_tree = pin_id >= 0 && pin_id < n && is_load_pin[pin_id]
            && (!cell || cell->sequentials_.empty());
        if (set_activity(pin_id, pin_density, pin_duty, 2, true, false) && enqueue_clock_tree)
            enqueue_adjacent_vertices(pin_id);
    }

    dumpPowerActivityCpuTracePaths(gtdb, trace_path_out_env, n, actual_seed_seen,
                                    pin_level, pin_to_node, is_driver_pin,
                                    seq_reverse_edges);

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

}  // namespace gt
