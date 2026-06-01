#include "gputimer/core/GPUTimer.h"
#include "gputimer/core/DmpModel.h"
#include "gputimer/core/power/common/PowerCudaModel.h"
#include "gputimer/core/power/common/PowerHostCommon.h"
#include "gputimer/core/power/common/PowerActivityHostUtils.h"
#include "gputimer/core/power/cuda_input/PowerCudaInputBuildInternal.h"
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
#include <array>
#include <chrono>
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

torch::Tensor GPUTimer::compute_power_activity_cuda(torch::Tensor* inst_switching_cpu,
                                                    torch::Tensor* pin_switching_cpu,
                                                    torch::Tensor* inst_internal_cpu,
                                                    torch::Tensor* internal_row_power_cpu,
                                                    torch::Tensor* internal_row_meta_cpu,
                                                    torch::Tensor* inst_leakage_cpu,
                                                    torch::Tensor* leakage_row_power_cpu,
                                                    torch::Tensor* leakage_row_meta_cpu,
                                                    bool output_power_tensors_cuda) {
    const int n = static_cast<int>(gtdb.pin_names.size());
    if (n <= 0) return torch::empty({0, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("report_power_activity_cuda requires CUDA");
    }
    // Some existing init kernels leave a stale CUDA error status that CPU reports ignore.
    // Clear it before allocating/uploading the Plan-A power activity data structures.
    clear_power_cuda_error();
    const bool profile_power_stages = readPowerBoolEnv("XPLACE_POWER_PROFILE_STAGES", false);
    if (profile_power_stages) resetPowerStageProfileElapsed();
    auto profile_last = std::chrono::steady_clock::now();
    auto profile_mark = [&](const char* label) {
        if (!profile_power_stages) return;
        const auto now = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - profile_last).count();
        addPowerStageProfileElapsed(elapsed);
        std::fprintf(stderr, "[power_stage_profile] %s %.6f\n", label, elapsed);
        profile_last = now;
    };

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

    if (const char* pin_name_dump = std::getenv("XPLACE_POWER_PIN_NAME_DUMP")) {
        if (pin_name_dump[0] != '\0') {
            std::ofstream out(pin_name_dump);
            if (out) {
                out << "pin_id,pin_name\n";
                for (int pin_id = 0; pin_id < n; ++pin_id) {
                    out << pin_id << ',' << csvEscapePowerActivitySnapshot(gtdb.pin_names[pin_id]) << '\n';
                }
            }
        }
    }

    std::vector<int> h_pin_to_node;
    std::vector<int> h_pin_to_net;
    buildPowerPinNodeNetMaps(gtdb, n, h_pin_to_node, h_pin_to_net);

    auto get_cell = [&](int node_id) -> LibertyCell* { return powerCellForNode(gtdb, node_id); };
    auto is_io_node = [&](int node_id) -> bool { return powerIsIoNode(gtdb, node_id); };
    auto normalize_expr = [](std::string expr) { return normalizePowerExprString(expr); };
    auto parse_const_net_value = [](std::string name) -> int { return parsePowerConstNetValue(name); };
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

    std::vector<uint8_t> h_is_load_pin;
    std::vector<uint8_t> h_is_driver_pin;
    std::vector<uint8_t> h_is_cell_pin;
    std::vector<uint8_t> h_is_seq_output_pin(n, 0);
    classifyPowerPins(gtdb, n, h_pin_to_node, h_is_load_pin, h_is_driver_pin, &h_is_cell_pin);

    std::vector<int> h_net_driver_pin;
    buildPowerNetDriverPins(gtdb, n, h_is_driver_pin, h_net_driver_pin);

    std::vector<int> h_clock_gate_out_for_input;
    std::vector<int> h_clock_gate_clock_for_out;
    std::vector<int> h_clock_gate_enable_for_out;
    std::vector<uint8_t> h_is_clock_gate_clock_pin;
    buildPowerClockGateMaps(gtdb, n, h_pin_to_node, h_clock_gate_out_for_input,
                             h_clock_gate_clock_for_out, h_clock_gate_enable_for_out,
                             h_is_clock_gate_clock_pin);
    profile_mark("pin_maps");

    auto build_clock_pins = [&]() {
        return buildPowerClockPins(gtdb, n, h_pin_to_node, h_pin_to_net,
                                   h_is_load_pin, h_is_driver_pin,
                                   h_is_clock_gate_clock_pin);
    };
    std::vector<int> h_clock_pins = build_clock_pins();
    std::vector<float> h_clock_pin_densities;
    std::vector<float> h_clock_pin_duties;
    std::vector<uint8_t> h_clock_pin_enqueue;
    h_clock_pin_densities.reserve(h_clock_pins.size());
    h_clock_pin_duties.reserve(h_clock_pins.size());
    h_clock_pin_enqueue.reserve(h_clock_pins.size());
    for (int pin_id : h_clock_pins) {
        auto [density, duty] = powerClockActivityForPin(gtdb, pin_id, sdc_time_scale, clock_density);
        h_clock_pin_densities.push_back(density);
        h_clock_pin_duties.push_back(duty);
        const int node_id = pin_id >= 0 && pin_id < n ? h_pin_to_node[pin_id] : -1;
        LibertyCell* cell = get_cell(node_id);
        const bool enqueue_clock_tree = pin_id >= 0 && pin_id < n && h_is_load_pin[pin_id]
            && (!cell || cell->sequentials_.empty());
        h_clock_pin_enqueue.push_back(enqueue_clock_tree ? 1 : 0);
    }
    profile_mark("clock_pins");

    std::vector<GpuPowerExprOpHost> h_expr_ops;
    std::vector<int> h_expr_start;
    std::vector<int> h_expr_count;
    const bool ignore_scan_enable_density =
        readPowerBoolEnv("XPLACE_POWER_IGNORE_SCAN_ENABLE_DENSITY", false);
    auto add_expr = [&](const std::string& expr_str, LibertyCell* cell, const gp::GPNode& node,
                        bool* used_missing_const = nullptr,
                        bool zero_scan_enable_density = false) -> int {
        return addPowerCudaExpr(expr_str, cell, node, h_expr_ops, h_expr_start, h_expr_count,
                                const_port_value_for_node, used_missing_const,
                                zero_scan_enable_density);
    };

    std::unordered_map<std::string, int> template_expr_cache;
    auto add_template_expr = [&](const std::string& expr_str, LibertyCell* cell) -> int {
        return addPowerCudaTemplateExpr(expr_str, cell, h_expr_ops, h_expr_start,
                                        h_expr_count, template_expr_cache);
    };

    auto expr_contains_pin = [&](int expr_id, int pin_id) -> bool {
        const int port_offset = pin_id >= 0 && pin_id < static_cast<int>(gtdb.pin_id2port_offset_id.size())
            ? gtdb.pin_id2port_offset_id[pin_id]
            : -1;
        return powerCudaExprContainsPin(expr_id, pin_id, port_offset,
                                        h_expr_ops, h_expr_start, h_expr_count);
    };

    std::vector<int> h_pin_func_expr_id(n, -1);
    std::vector<uint8_t> h_pin_func_has_missing_const(n, 0);
    const char* debug_expr_node_env = std::getenv("XPLACE_POWER_DEBUG_EXPR_NODE");
    std::vector<int> port_pin_by_offset;
    for (const auto& node : gtdb.gpdb.getNodes()) {
        int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = get_cell(node_id);
        if (!cell) continue;
        port_pin_by_offset.assign(cell->ports_.size(), -1);
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset >= 0 && port_offset < static_cast<int>(port_pin_by_offset.size()))
                port_pin_by_offset[port_offset] = pin_id;
        }
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n || !h_is_driver_pin[pin_id]) continue;
            int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
            bool used_missing_const = false;
            const int template_expr_id = add_template_expr(port->function_expr_, cell);
            if (template_expr_id >= 0 &&
                powerCudaTemplateExprPortsPresent(template_expr_id, port_pin_by_offset,
                                                  h_expr_ops, h_expr_start, h_expr_count)) {
                h_pin_func_expr_id[pin_id] = template_expr_id;
            } else {
                h_pin_func_expr_id[pin_id] = add_expr(port->function_expr_, cell, node,
                                                      &used_missing_const);
            }
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
    profile_mark("function_exprs");

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
        port_pin_by_offset.assign(cell->ports_.size(), -1);
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset >= 0 && port_offset < static_cast<int>(port_pin_by_offset.size()))
                port_pin_by_offset[port_offset] = pin_id;
        }
        for (SequentialPower* seq : cell->sequentials_) {
            if (!seq) continue;
            GpuPowerSeqHost rec;
            rec.node_id = node_id;
            const int data_template_expr_id =
                addPowerCudaTemplateExpr(seq->next_state_expr_, cell, h_expr_ops,
                                         h_expr_start, h_expr_count, template_expr_cache,
                                         ignore_scan_enable_density);
            rec.data_expr_id =
                (data_template_expr_id >= 0 &&
                 powerCudaTemplateExprPortsPresent(data_template_expr_id, port_pin_by_offset,
                                                   h_expr_ops, h_expr_start, h_expr_count))
                ? data_template_expr_id
                : add_expr(seq->next_state_expr_, cell, node, nullptr,
                           ignore_scan_enable_density);
            const std::string clk_expr = seqClockExpr(seq);
            const int clk_template_expr_id = add_template_expr(clk_expr, cell);
            rec.clk_expr_id =
                (clk_template_expr_id >= 0 &&
                 powerCudaTemplateExprPortsPresent(clk_template_expr_id, port_pin_by_offset,
                                                   h_expr_ops, h_expr_start, h_expr_count))
                ? clk_template_expr_id
                : add_expr(clk_expr, cell, node);
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
    profile_mark("sequentials");

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
         auto pin_for_node_port = [&](int node_id, int port_offset) -> int {
             if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size()) ||
                 port_offset < 0)
                 return -1;
             const auto& node = gtdb.gpdb.getNodes()[node_id];
             for (int pin_id : node.pins()) {
                 if (pin_id < 0 || pin_id >= n) continue;
                 if (pin_id < static_cast<int>(gtdb.pin_id2port_offset_id.size()) &&
                     gtdb.pin_id2port_offset_id[pin_id] == port_offset)
                     return pin_id;
             }
             return -1;
         };
         auto collect_feedback_data_pins = [&](int expr_id, int seq_node_id, int driver_pin,
                                               std::vector<int>* data_pins) -> bool {
             if (expr_id < 0 || driver_pin < 0 || driver_pin >= n) return false;
             const int driver_net = h_pin_to_net[driver_pin];
             if (driver_net < 0 || driver_net >= static_cast<int>(h_net_driver_pin.size())) return false;
            if (h_net_driver_pin[driver_net] != driver_pin) return false;
            bool matched = false;
            const int start = h_expr_start[expr_id];
             const int end = start + h_expr_count[expr_id];
             for (int op_i = start; op_i < end; ++op_i) {
                 if (h_expr_ops[op_i].op != 0) continue;
                 int data_pin = h_expr_ops[op_i].arg;
                 if (data_pin < -1) data_pin = pin_for_node_port(seq_node_id, -2 - data_pin);
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
             const bool q_feedback = collect_feedback_data_pins(seq.data_expr_id, seq.node_id,
                                                                seq.q_pin, &data_pins);
             if (q_feedback && seed_seq_feedback_outputs && !seed_seen[seq.q_pin]) {
                 add_seed_pin(seq.q_pin, "seq_feedback_q");
                 seed_seen[seq.q_pin] = 1;
                 root_seq_feedback_count++;
             }
             const bool qn_feedback = collect_feedback_data_pins(seq.data_expr_id, seq.node_id,
                                                                 seq.qn_pin, &data_pins);
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
    profile_mark("roots");
    std::vector<GpuPowerInternalHost> h_internal_rows;
    std::unordered_map<uint64_t, int> internal_denom_group;
    buildPowerCudaInternalRows(gtdb, n, need_internal_power, h_is_load_pin,
                               h_is_driver_pin, h_pin_func_expr_id, h_expr_ops,
                               h_expr_start, h_expr_count, add_template_expr,
                               expr_contains_pin, h_internal_rows,
                               internal_denom_group);

    std::vector<GpuPowerLeakageRowHost> h_leakage_rows;
    std::vector<GpuPowerLeakageGroupHost> h_leakage_groups;
    buildPowerCudaLeakageRows(gtdb, need_leakage_power, add_template_expr,
                              h_leakage_rows, h_leakage_groups);
    profile_mark("component_rows");

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
        if (v.empty()) return torch::zeros({1}, iopt_cpu).to(torch::kCUDA);
        return torch::from_blob(const_cast<int*>(v.data()), {(long)v.size()}, iopt_cpu).to(torch::kCUDA);
    };
    auto to_cuda_index = [&](const std::vector<index_type>& v) {
        if (v.empty()) return torch::zeros({1}, iopt_cpu).to(torch::kCUDA);
        return torch::from_blob(const_cast<index_type*>(v.data()), {(long)v.size()}, iopt_cpu).to(torch::kCUDA);
    };
    auto to_cuda_u8 = [&](const std::vector<uint8_t>& v) {
        if (v.empty()) return torch::zeros({1}, bopt_cpu).to(torch::kCUDA);
        return torch::from_blob(const_cast<uint8_t*>(v.data()), {(long)v.size()}, bopt_cpu).to(torch::kCUDA);
    };
    auto to_cuda_float = [&](const std::vector<float>& v) {
        auto fopt_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        if (v.empty()) return torch::full({1}, nanf(""), fopt_cpu).to(torch::kCUDA);
        return torch::from_blob(const_cast<float*>(v.data()), {(long)v.size()}, fopt_cpu).to(torch::kCUDA);
    };
    auto to_cuda_bytes = [&](const auto& v) {
        using VecT = std::decay_t<decltype(v)>;
        using ElemT = typename VecT::value_type;
        if (v.empty()) return torch::zeros({(long)sizeof(ElemT)}, bopt_cpu).to(torch::kCUDA);
        auto* data = reinterpret_cast<uint8_t*>(const_cast<ElemT*>(v.data()));
        return torch::from_blob(data, {(long)(v.size() * sizeof(ElemT))}, bopt_cpu).to(torch::kCUDA);
    };
    auto to_cuda_bytes_range = [&](const auto& v, size_t begin, size_t count) {
        using VecT = std::decay_t<decltype(v)>;
        using ElemT = typename VecT::value_type;
        if (count == 0) {
            return torch::zeros({(long)sizeof(ElemT)}, bopt_cpu).to(torch::kCUDA);
        }
        auto* data = const_cast<ElemT*>(v.data() + begin);
        return torch::from_blob(reinterpret_cast<uint8_t*>(data), {(long)(count * sizeof(ElemT))}, bopt_cpu).to(torch::kCUDA);
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
    constexpr size_t default_power_row_chunk_bytes = 8ull * 1024ull * 1024ull * 1024ull;
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
    if (std::getenv("XPLACE_POWER_PRINT_ROW_STATS")) {
        std::array<size_t, 5> internal_duty_modes{};
        size_t internal_input_rows = 0;
        size_t internal_output_rows = 0;
        size_t internal_fast_duty_rows = 0;
        size_t internal_expr_duty_rows = 0;
        for (const auto& row : h_internal_rows) {
            if (row.kind == 0) ++internal_input_rows;
            if (row.kind == 1) ++internal_output_rows;
            if (row.duty_mode >= 0 && row.duty_mode < static_cast<int>(internal_duty_modes.size()))
                ++internal_duty_modes[row.duty_mode];
            if (row.duty_mode == 1 || row.duty_mode == 2) ++internal_expr_duty_rows;
            else ++internal_fast_duty_rows;
        }
        size_t leakage_when_rows = 0;
        size_t leakage_no_when_rows = 0;
        for (const auto& row : h_leakage_rows) {
            if (row.when_expr_id >= 0) ++leakage_when_rows;
            else ++leakage_no_when_rows;
        }
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
        std::fprintf(stderr,
                     "[power_row_stats] internal_kind input=%zu output=%zu duty0=%zu duty1_expr=%zu "
                     "duty2_diff=%zu duty3_half=%zu duty4_zero=%zu fast_duty=%zu expr_duty=%zu "
                     "leakage_no_when=%zu leakage_when=%zu\n",
                     internal_input_rows, internal_output_rows,
                     internal_duty_modes[0], internal_duty_modes[1],
                     internal_duty_modes[2], internal_duty_modes[3],
                     internal_duty_modes[4], internal_fast_duty_rows,
                     internal_expr_duty_rows, leakage_no_when_rows,
                     leakage_when_rows);
    }
    std::vector<int> h_node_port_pin_start;
    std::vector<int> h_node_port_pin_list;
    if (need_internal_power || need_leakage_power) {
        buildPowerNodePortPinMap(gtdb, h_node_port_pin_start, h_node_port_pin_list);
    }
    profile_mark("node_port_pin_map");

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
    profile_mark("uploads");

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
    profile_mark("levelize_power");
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
    dumpPowerCudaInputRoots(gtdb, n, h_primary_inputs, power_level_root_pins_cpu,
                            h_seed_reason, h_seed_seen, h_is_primary_input,
                            h_is_clock_pin, h_is_driver_pin, h_is_load_pin,
                            h_power_fanin, h_pin_to_node, h_pin_to_net,
                            power_pin_level_cpu);
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
    profile_mark("root_upload");


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
        PowerCpuActivityLevels cpu_activity_levels = buildPowerCpuActivityLevels(gtdb, n);
        d_cpu_activity_level_list = to_cuda_int(cpu_activity_levels.level_list);
        cpu_activity_level_list_end_cpu = std::move(cpu_activity_levels.level_list_end);
        activity_level_list_end_cpu = &cpu_activity_level_list_end_cpu;
        activity_level_list = d_cpu_activity_level_list.data_ptr<int>();
        h_pin_power_level = std::move(cpu_activity_levels.pin_level);
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
    profile_mark("activity_levels");

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
    auto env_int64 = [](const char* name, int64_t default_value) {
        const char* env = std::getenv(name);
        if (!env || env[0] == '\0') return default_value;
        char* end = nullptr;
        const long long parsed = std::strtoll(env, &end, 10);
        return end != env ? static_cast<int64_t>(parsed) : default_value;
    };
    constexpr int64_t default_cpu_activity_pin_limit = 4000000;
    const int64_t auto_cpu_activity_pin_limit =
        env_int64("XPLACE_POWER_AUTO_CPU_ACTIVITY_PIN_LIMIT", default_cpu_activity_pin_limit);
    const bool force_cpu_activity_for_power =
        env_flag("XPLACE_POWER_USE_CPU_ACTIVITY_FOR_POWER", false);
    const bool use_cpu_activity_for_power =
        force_cpu_activity_for_power || (auto_cpu_activity_pin_limit > 0 && n <= auto_cpu_activity_pin_limit);
    torch::Tensor precomputed_activity_cpu;
    torch::Tensor precomputed_activity_gpu;
    const float* precomputed_activity_ptr = nullptr;
    if (use_cpu_activity_for_power) {
        precomputed_activity_cpu = report_power_activity_cpu();
        profile_mark(force_cpu_activity_for_power ? "cpu_activity_for_power"
                                                  : "auto_cpu_activity_for_power");
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
    profile_mark("launcher_prepare");
    run_power_activity_cuda_launcher(activity_model);
    profile_mark("launcher");

    const float* chunk_activity_ptr = precomputed_activity_ptr ? precomputed_activity_ptr : out_gpu_ptr;
    if ((chunk_internal_rows || chunk_leakage_rows) && !chunk_activity_ptr) {
        throw std::runtime_error("chunked CUDA power requires a precomputed activity tensor");
    }
    PowerChunkActivityStorage chunk_activity_storage;
    if (chunk_internal_rows || chunk_leakage_rows) {
        init_power_chunk_activity_storage(n, chunk_activity_ptr, &chunk_activity_storage);
        if (!chunk_activity_storage.density || !chunk_activity_storage.duty) {
            throw std::runtime_error("chunked CUDA power failed to prepare activity density/duty");
        }
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
                denom_model.activity_density = chunk_activity_storage.density;
                denom_model.activity_duty = chunk_activity_storage.duty;
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
                contrib_model.activity_density = chunk_activity_storage.density;
                contrib_model.activity_duty = chunk_activity_storage.duty;
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
                rows_model.activity_density = chunk_activity_storage.density;
                rows_model.activity_duty = chunk_activity_storage.duty;
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
    free_power_chunk_activity_storage(&chunk_activity_storage);
    profile_mark("chunk_components");

    auto output_power_tensor = [&](const torch::Tensor& tensor) {
        return output_power_tensors_cuda ? tensor : tensor.to(torch::kCPU);
    };
    if (inst_switching_cpu) *inst_switching_cpu = output_power_tensor(inst_switching_gpu);
    if (pin_switching_cpu) *pin_switching_cpu = output_power_tensor(pin_switching_gpu);
    if (inst_internal_cpu) *inst_internal_cpu = output_power_tensor(inst_internal_gpu);
    if (internal_row_power_cpu) *internal_row_power_cpu = output_power_tensor(internal_row_power_gpu);
    if (inst_leakage_cpu) *inst_leakage_cpu = output_power_tensor(inst_leakage_gpu);
    if (leakage_row_power_cpu) *leakage_row_power_cpu = output_power_tensor(leakage_row_power_gpu);
    profile_mark("downloads");
    if (want_activity_cpu) {
        return out_gpu.to(torch::kCPU);
    }
    return torch::empty({0, 3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
}


}  // namespace gt
