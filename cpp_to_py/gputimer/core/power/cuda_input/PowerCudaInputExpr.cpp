#include "PowerCudaInputBuildInternal.h"

#include "gputimer/core/power/common/PowerActivityHostUtils.h"
#include "gputimer/core/power/common/PowerHostCommon.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

namespace gt {

namespace {

bool powerExprPortIdFitsVarKey(int port_id) {
    return port_id >= 0 && port_id <= static_cast<int>(std::numeric_limits<int16_t>::max());
}

}  // namespace

class PowerCudaConstPortResolverImpl {
public:
    explicit PowerCudaConstPortResolverImpl(GTDatabase& gtdb) : gtdb_(gtdb) {}

    int operator()(const gp::GPNode& node, const std::string& port_name) {
        loadConstPortFile();
        const std::string key = normalizePowerActivitySnapshotName(node.getName()) + "/" +
                                normalizePowerActivitySnapshotName(port_name);
        auto const_itr = const_port_file_values_.find(key);
        if (const_itr != const_port_file_values_.end()) return const_itr->second;
        const int raw_cell_id = static_cast<int>(node.getOriDBId());
        if (raw_cell_id < 0 || raw_cell_id >= static_cast<int>(gtdb_.rawdb.cells.size()))
            return -1;
        db::Cell* dbcell = gtdb_.rawdb.cells[raw_cell_id];
        db::Pin* dbpin = dbcell ? dbcell->pin(port_name) : nullptr;
        return (dbpin && dbpin->net) ? parsePowerConstNetValue(dbpin->net->name) : -1;
    }

private:
    void loadConstPortFile() {
        if (const_port_file_loaded_) return;
        const_port_file_loaded_ = true;
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
            const int const_value = parsePowerConstNetValue(value);
            if (const_value < 0) continue;
            const std::string key = normalizePowerActivitySnapshotName(inst) + "/" +
                                    normalizePowerActivitySnapshotName(port);
            const_port_file_values_[key] = const_value;
        }
    }

    GTDatabase& gtdb_;
    std::unordered_map<std::string, int> const_port_file_values_;
    bool const_port_file_loaded_ = false;
};

PowerCudaExprInputs::PowerCudaExprInputs() = default;

PowerCudaExprInputs::PowerCudaExprInputs(int n)
    : pin_func_expr_id(n, -1),
      missing_func_out_start(n + 1, 0) {}

PowerCudaSeqInputs::PowerCudaSeqInputs() = default;

PowerCudaSeqInputs::PowerCudaSeqInputs(int n)
    : is_seq_output_pin(n, 0),
      is_seq_clock_input_pin(n, 0),
      pin_seq_list_start(n + 1, 0) {}

PowerCudaSeqInputs::PowerCudaSeqInputs(std::vector<GpuPowerSeqHost> seqs_,
                                       std::vector<uint8_t> is_seq_output_pin_,
                                       std::vector<uint8_t> is_seq_clock_input_pin_,
                                       std::vector<int> pin_seq_list_start_,
                                       std::vector<int> pin_seq_list_)
    : seqs(std::move(seqs_)),
      is_seq_output_pin(std::move(is_seq_output_pin_)),
      is_seq_clock_input_pin(std::move(is_seq_clock_input_pin_)),
      pin_seq_list_start(std::move(pin_seq_list_start_)),
      pin_seq_list(std::move(pin_seq_list_)) {}

int PowerCudaExprInputs::addExpr(const std::string& expr_str,
                                 LibertyCell* cell,
                                 const gp::GPNode& node,
                                 const PowerConstPortResolver& const_port_value_for_node,
                                 bool* used_missing_const,
                                 bool zero_scan_enable_density) {
    return addPowerCudaExpr(expr_str, cell, node, ops, start, count,
                            const_port_value_for_node, used_missing_const,
                            zero_scan_enable_density);
}

int PowerCudaExprInputs::addTemplateExpr(const std::string& expr_str,
                                         LibertyCell* cell,
                                         bool zero_scan_enable_density) {
    return addPowerCudaTemplateExpr(expr_str, cell, ops, start, count,
                                    template_expr_cache, zero_scan_enable_density);
}

bool PowerCudaExprInputs::containsPin(const GTDatabase& gtdb, int expr_id, int pin_id) const {
    const int port_offset = pin_id >= 0 && pin_id < static_cast<int>(gtdb.pin_id2port_offset_id.size())
        ? gtdb.pin_id2port_offset_id[pin_id]
        : -1;
    return powerCudaExprContainsPin(expr_id, pin_id, port_offset, ops, start, count);
}

bool PowerCudaExprInputs::templatePortsPresent(int expr_id,
                                               const std::vector<int>& port_pin_by_offset) const {
    return powerCudaTemplateExprPortsPresent(expr_id, port_pin_by_offset, ops, start, count);
}

PowerCudaExprInputs buildPowerCudaExprInputs(GTDatabase& gtdb,
                                             int n,
                                             const std::vector<uint8_t>& is_load_pin,
                                             const std::vector<uint8_t>& is_driver_pin) {
    PowerCudaExprInputs expr_inputs(n);
    PowerCudaConstPortResolverImpl const_port_resolver(gtdb);
    PowerConstPortResolver const_port_value_for_node =
        [&](const gp::GPNode& node, const std::string& port_name) {
            return const_port_resolver(node, port_name);
        };

    std::vector<uint8_t> pin_func_has_missing_const(n, 0);
    const char* debug_expr_node_env = std::getenv("XPLACE_POWER_DEBUG_EXPR_NODE");
    std::vector<int> port_pin_by_offset;
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = powerCellForNode(gtdb, node_id);
        if (!cell) continue;
        port_pin_by_offset.assign(cell->ports_.size(), -1);
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset >= 0 && port_offset < static_cast<int>(port_pin_by_offset.size()))
                port_pin_by_offset[port_offset] = pin_id;
        }
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n || !is_driver_pin[pin_id]) continue;
            const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
            bool used_missing_const = false;
            const int template_expr_id = expr_inputs.addTemplateExpr(port->function_expr_, cell);
            if (template_expr_id >= 0 && expr_inputs.templatePortsPresent(template_expr_id, port_pin_by_offset)) {
                expr_inputs.pin_func_expr_id[pin_id] = template_expr_id;
            } else {
                expr_inputs.pin_func_expr_id[pin_id] =
                    expr_inputs.addExpr(port->function_expr_, cell, node,
                                        const_port_value_for_node, &used_missing_const);
            }
            if (expr_inputs.pin_func_expr_id[pin_id] >= 0 && used_missing_const) {
                pin_func_has_missing_const[pin_id] = 1;
            }
            if (debug_expr_node_env && node.getName().find(debug_expr_node_env) != std::string::npos) {
                std::fprintf(stderr,
                             "[XPLACE_POWER_DEBUG_EXPR] node=%s pin=%s port=%s expr_id=%d missing_const=%d function='%s'\n",
                             node.getName().c_str(),
                             gtdb.pin_names[pin_id].c_str(),
                             port->name.c_str(),
                             expr_inputs.pin_func_expr_id[pin_id],
                             pin_func_has_missing_const[pin_id] ? 1 : 0,
                             port->function_expr_.c_str());
            }
        }
    }

    const bool eval_missing_const_outputs =
        readPowerBoolEnv("XPLACE_POWER_EVAL_MISSING_CONST_OUTPUTS", true);
    std::vector<std::vector<int>> missing_func_outputs_by_pin(n);
    if (eval_missing_const_outputs) {
        for (const auto& node : gtdb.gpdb.getNodes()) {
            LibertyCell* cell = powerCellForNode(gtdb, static_cast<int>(node.getId()));
            if (!cell || !cell->sequentials_.empty()) continue;
            std::vector<int> load_pins;
            std::vector<int> missing_func_out_pins;
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                if (is_load_pin[pin_id]) load_pins.push_back(pin_id);
                if (is_driver_pin[pin_id] && pin_func_has_missing_const[pin_id])
                    missing_func_out_pins.push_back(pin_id);
            }
            if (load_pins.empty() || missing_func_out_pins.empty()) continue;
            for (int load_pin : load_pins) {
                auto& outputs = missing_func_outputs_by_pin[load_pin];
                outputs.insert(outputs.end(), missing_func_out_pins.begin(), missing_func_out_pins.end());
            }
        }
    }
    expr_inputs.missing_func_out_list.clear();
    for (int pin_id = 0; pin_id < n; ++pin_id) {
        expr_inputs.missing_func_out_start[pin_id] =
            static_cast<int>(expr_inputs.missing_func_out_list.size());
        auto& outputs = missing_func_outputs_by_pin[pin_id];
        std::sort(outputs.begin(), outputs.end());
        outputs.erase(std::unique(outputs.begin(), outputs.end()), outputs.end());
        expr_inputs.missing_func_out_list.insert(expr_inputs.missing_func_out_list.end(),
                                                 outputs.begin(), outputs.end());
    }
    expr_inputs.missing_func_out_start[n] = static_cast<int>(expr_inputs.missing_func_out_list.size());
    return expr_inputs;
}

PowerCudaSeqInputs buildPowerCudaSeqInputs(GTDatabase& gtdb,
                                           int n,
                                           const std::vector<int>& pin_to_node,
                                           const std::vector<uint8_t>& is_load_pin,
                                           PowerCudaExprInputs& expr_inputs) {
    PowerCudaSeqInputs seq_inputs(n);
    PowerCudaConstPortResolverImpl const_port_resolver(gtdb);
    PowerConstPortResolver const_port_value_for_node =
        [&](const gp::GPNode& node, const std::string& port_name) {
            return const_port_resolver(node, port_name);
        };
    const bool ignore_scan_enable_density =
        readPowerBoolEnv("XPLACE_POWER_IGNORE_SCAN_ENABLE_DENSITY", false);

    std::vector<std::vector<int>> node_seq_ids(gtdb.gpdb.getNodes().size());
    std::vector<int> port_pin_by_offset;
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = powerCellForNode(gtdb, node_id);
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
            const int data_template_expr_id =
                expr_inputs.addTemplateExpr(seq->next_state_expr_, cell, ignore_scan_enable_density);
            const int data_expr_id =
                (data_template_expr_id >= 0 &&
                 expr_inputs.templatePortsPresent(data_template_expr_id, port_pin_by_offset))
                ? data_template_expr_id
                : expr_inputs.addExpr(seq->next_state_expr_, cell, node,
                                      const_port_value_for_node, nullptr,
                                      ignore_scan_enable_density);
            const std::string clk_expr = seqClockExpr(seq);
            const int clk_template_expr_id = expr_inputs.addTemplateExpr(clk_expr, cell);
            const int clk_expr_id =
                (clk_template_expr_id >= 0 &&
                 expr_inputs.templatePortsPresent(clk_template_expr_id, port_pin_by_offset))
                ? clk_template_expr_id
                : expr_inputs.addExpr(clk_expr, cell, node, const_port_value_for_node);
            if (data_expr_id < 0) continue;
            int q_pin = -1;
            int qn_pin = -1;
            const std::string seq_out = normalizePowerExprString(seq->output_name_);
            const std::string seq_out_inv = normalizePowerExprString(seq->output_inv_name_);
            for (int pin_id : node.pins()) {
                if (pin_id < 0 || pin_id >= n) continue;
                const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
                if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
                LibertyPort* port = cell->ports_[port_offset];
                if (!port || port->direction_ != CellPortDirection::output || !port->has_function_) continue;
                const std::string func = normalizePowerExprString(port->function_expr_);
                if (!seq_out.empty() && func == seq_out) q_pin = pin_id;
                else if (!seq_out_inv.empty() && func == seq_out_inv) qn_pin = pin_id;
            }
            if (q_pin < 0 && qn_pin < 0) continue;
            const GpuPowerSeqHost rec(data_expr_id,
                                      clk_expr_id,
                                      node_id,
                                      q_pin,
                                      qn_pin,
                                      seq->is_latch_ ? 1 : 0);
            const int seq_id = static_cast<int>(seq_inputs.seqs.size());
            seq_inputs.seqs.push_back(rec);
            if (rec.q_pin >= 0) seq_inputs.is_seq_output_pin[rec.q_pin] = 1;
            if (rec.qn_pin >= 0) seq_inputs.is_seq_output_pin[rec.qn_pin] = 1;
            if (node_id >= 0 && node_id < static_cast<int>(node_seq_ids.size()))
                node_seq_ids[node_id].push_back(seq_id);
        }
    }
    dumpPowerSeqIdMapIfRequested(gtdb, seq_inputs.seqs, n);
    printPowerSeqDupStatsIfRequested(gtdb, seq_inputs.seqs, n);

    const bool mark_seq_clock_loads =
        readPowerBoolEnv("XPLACE_POWER_MARK_SEQ_CLOCK_LOADS", false);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        LibertyCell* cell = powerCellForNode(gtdb, node_id);
        if (!cell || cell->sequentials_.empty()) continue;
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n || !is_load_pin[pin_id]) continue;
            const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size())) continue;
            LibertyPort* port = cell->ports_[port_offset];
            if (port && port->is_clock_) seq_inputs.is_seq_clock_input_pin[pin_id] = 1;
        }
    }

    std::vector<std::vector<int>> pin_seq_ids(n);
    for (int pin = 0; pin < n; pin++) {
        const int node_id = pin_to_node[pin];
        if (node_id >= 0 && node_id < static_cast<int>(node_seq_ids.size()) && !node_seq_ids[node_id].empty()) {
            if (is_load_pin[pin] && (mark_seq_clock_loads || !seq_inputs.is_seq_clock_input_pin[pin]))
                pin_seq_ids[pin] = node_seq_ids[node_id];
        }
    }
    seq_inputs.pin_seq_list.clear();
    for (int pin = 0; pin < n; pin++) {
        seq_inputs.pin_seq_list_start[pin] = static_cast<int>(seq_inputs.pin_seq_list.size());
        seq_inputs.pin_seq_list.insert(seq_inputs.pin_seq_list.end(),
                                       pin_seq_ids[pin].begin(), pin_seq_ids[pin].end());
    }
    seq_inputs.pin_seq_list_start[n] = static_cast<int>(seq_inputs.pin_seq_list.size());
    return seq_inputs;
}

int addPowerCudaExpr(const std::string& expr_str,
                     LibertyCell* cell,
                     const gp::GPNode& node,
                     std::vector<GpuPowerExprOpHost>& expr_ops,
                     std::vector<int>& expr_start,
                     std::vector<int>& expr_count,
                     const PowerConstPortResolver& const_port_value_for_node,
                     bool* used_missing_const,
                     bool zero_scan_enable_density) {
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
                if (!powerExprPortIdFitsVarKey(op.port_id)) return -1;
                const std::string& port_name = cell->ports_[op.port_id]->name;
                auto pin_itr = node.portMap.find(port_name);
                if (pin_itr != node.portMap.end()) {
                    const uint8_t zero_density =
                        (zero_scan_enable_density && cell->ports_[op.port_id] &&
                         cell->ports_[op.port_id]->nextstate_type_ == "scan_enable")
                            ? 1
                            : 0;
                    out = GpuPowerExprOpHost(pin_itr->second,
                                             static_cast<int16_t>(op.port_id),
                                             0,
                                             zero_density);
                } else {
                    const int const_value = const_port_value_for_node(node, port_name);
                    out = GpuPowerExprOpHost(-1,
                                             static_cast<int16_t>(op.port_id),
                                             const_value > 0 ? 2 : 1);
                    if (used_missing_const) *used_missing_const = true;
                }
                break;
            }
            case PowerExprOpcode::const_zero: out = GpuPowerExprOpHost(-1, -1, 1); break;
            case PowerExprOpcode::const_one: out = GpuPowerExprOpHost(-1, -1, 2); break;
            case PowerExprOpcode::logical_not: out = GpuPowerExprOpHost(-1, -1, 3); break;
            case PowerExprOpcode::logical_and: out = GpuPowerExprOpHost(-1, -1, 4); break;
            case PowerExprOpcode::logical_or: out = GpuPowerExprOpHost(-1, -1, 5); break;
            case PowerExprOpcode::logical_xor: out = GpuPowerExprOpHost(-1, -1, 6); break;
        }
        local_ops.push_back(out);
    }
    if (local_ops.empty()) return -1;
    const int expr_id = static_cast<int>(expr_start.size());
    expr_start.push_back(static_cast<int>(expr_ops.size()));
    expr_count.push_back(static_cast<int>(local_ops.size()));
    expr_ops.insert(expr_ops.end(), local_ops.begin(), local_ops.end());
    return expr_id;
}

int addPowerCudaTemplateExpr(const std::string& expr_str,
                             LibertyCell* cell,
                             std::vector<GpuPowerExprOpHost>& expr_ops,
                             std::vector<int>& expr_start,
                             std::vector<int>& expr_count,
                             std::unordered_map<std::string, int>& template_expr_cache,
                             bool zero_scan_enable_density) {
    if (!cell) return -1;
    const std::string cache_key = cell->name + "|" + normalizePowerExprString(expr_str) +
                                  "|" + (zero_scan_enable_density ? "zd1" : "zd0");
    auto cache_itr = template_expr_cache.find(cache_key);
    if (cache_itr != template_expr_cache.end()) return cache_itr->second;
    PowerExpr expr;
    if (!expr.compile(expr_str, cell)) return -1;
    std::vector<GpuPowerExprOpHost> local_ops;
    local_ops.reserve(expr.ops().size());
    for (const auto& op : expr.ops()) {
        GpuPowerExprOpHost out;
        switch (op.opcode) {
            case PowerExprOpcode::port: {
                if (op.port_id < 0 || op.port_id >= static_cast<int>(cell->ports_.size())) return -1;
                if (!powerExprPortIdFitsVarKey(op.port_id)) return -1;
                const uint8_t zero_density =
                    (zero_scan_enable_density && cell->ports_[op.port_id] &&
                     cell->ports_[op.port_id]->nextstate_type_ == "scan_enable")
                        ? 1
                        : 0;
                out = GpuPowerExprOpHost(-2 - op.port_id,
                                         static_cast<int16_t>(op.port_id),
                                         0,
                                         zero_density);
                break;
            }
            case PowerExprOpcode::const_zero: out = GpuPowerExprOpHost(-1, -1, 1); break;
            case PowerExprOpcode::const_one: out = GpuPowerExprOpHost(-1, -1, 2); break;
            case PowerExprOpcode::logical_not: out = GpuPowerExprOpHost(-1, -1, 3); break;
            case PowerExprOpcode::logical_and: out = GpuPowerExprOpHost(-1, -1, 4); break;
            case PowerExprOpcode::logical_or: out = GpuPowerExprOpHost(-1, -1, 5); break;
            case PowerExprOpcode::logical_xor: out = GpuPowerExprOpHost(-1, -1, 6); break;
        }
        local_ops.push_back(out);
    }
    if (local_ops.empty()) return -1;
    const int expr_id = static_cast<int>(expr_start.size());
    expr_start.push_back(static_cast<int>(expr_ops.size()));
    expr_count.push_back(static_cast<int>(local_ops.size()));
    expr_ops.insert(expr_ops.end(), local_ops.begin(), local_ops.end());
    template_expr_cache.emplace(cache_key, expr_id);
    return expr_id;
}

bool powerCudaExprContainsPin(int expr_id,
                              int pin_id,
                              int port_offset,
                              const std::vector<GpuPowerExprOpHost>& expr_ops,
                              const std::vector<int>& expr_start,
                              const std::vector<int>& expr_count) {
    if (expr_id < 0 || expr_id >= static_cast<int>(expr_start.size())) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    for (int k = 0; k < count; ++k) {
        if (expr_ops[start + k].op != 0) continue;
        const int arg = expr_ops[start + k].arg;
        if (arg == pin_id) return true;
        if (arg < -1 && port_offset >= 0 && -2 - arg == port_offset) return true;
    }
    return false;
}

bool powerCudaTemplateExprPortsPresent(int expr_id,
                                       const std::vector<int>& port_pin_by_offset,
                                       const std::vector<GpuPowerExprOpHost>& expr_ops,
                                       const std::vector<int>& expr_start,
                                       const std::vector<int>& expr_count) {
    if (expr_id < 0 || expr_id >= static_cast<int>(expr_start.size())) return false;
    const int start = expr_start[expr_id];
    const int count = expr_count[expr_id];
    for (int k = 0; k < count; ++k) {
        const auto& op = expr_ops[start + k];
        if (op.op != 0 || op.arg >= -1) continue;
        const int port_id = -2 - op.arg;
        if (port_id < 0 || port_id >= static_cast<int>(port_pin_by_offset.size()) ||
            port_pin_by_offset[port_id] < 0)
            return false;
    }
    return true;
}

bool positiveUnateForPower(LibertyCell* cell, LibertyPort* from, LibertyPort* to) {
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
}

}  // namespace gt
