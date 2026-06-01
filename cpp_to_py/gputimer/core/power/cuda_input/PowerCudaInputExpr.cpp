#include "PowerCudaInputBuildInternal.h"

#include "gputimer/core/power/common/PowerActivityHostUtils.h"

namespace gt {

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
            case PowerExprOpcode::port:
                if (op.port_id < 0 || op.port_id >= static_cast<int>(cell->ports_.size())) return -1;
                out.op = 0;
                out.arg = -2 - op.port_id;
                out.var_key = op.port_id;
                if (zero_scan_enable_density && cell->ports_[op.port_id]
                    && cell->ports_[op.port_id]->nextstate_type_ == "scan_enable")
                    out.zero_density = 1;
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
