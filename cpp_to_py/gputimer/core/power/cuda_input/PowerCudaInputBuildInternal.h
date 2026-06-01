#pragma once

#include "gputimer/core/power/common/PowerCudaModel.h"
#include "gputimer/db/GTDatabase.h"
#include "common/lib/Liberty.h"
#include "common/lib/Timing.h"
#include "io_parser/gp/GPDatabase.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace gt {

using PowerConstPortResolver = std::function<int(const gp::GPNode&, const std::string&)>;

int addPowerCudaExpr(const std::string& expr_str,
                     LibertyCell* cell,
                     const gp::GPNode& node,
                     std::vector<GpuPowerExprOpHost>& expr_ops,
                     std::vector<int>& expr_start,
                     std::vector<int>& expr_count,
                     const PowerConstPortResolver& const_port_value_for_node,
                     bool* used_missing_const = nullptr,
                     bool zero_scan_enable_density = false);

int addPowerCudaTemplateExpr(const std::string& expr_str,
                             LibertyCell* cell,
                             std::vector<GpuPowerExprOpHost>& expr_ops,
                             std::vector<int>& expr_start,
                             std::vector<int>& expr_count,
                             std::unordered_map<std::string, int>& template_expr_cache,
                             bool zero_scan_enable_density = false);

bool powerCudaExprContainsPin(int expr_id,
                              int pin_id,
                              int port_offset,
                              const std::vector<GpuPowerExprOpHost>& expr_ops,
                              const std::vector<int>& expr_start,
                              const std::vector<int>& expr_count);
bool powerCudaTemplateExprPortsPresent(int expr_id,
                                       const std::vector<int>& port_pin_by_offset,
                                       const std::vector<GpuPowerExprOpHost>& expr_ops,
                                       const std::vector<int>& expr_start,
                                       const std::vector<int>& expr_count);

bool positiveUnateForPower(LibertyCell* cell, LibertyPort* from, LibertyPort* to);

void dumpPowerCudaInputRoots(GTDatabase& gtdb,
                             int n,
                             const std::vector<int>& primary_inputs,
                             const std::vector<int>& candidate_roots,
                             const std::vector<std::string>& seed_reason,
                             const std::vector<uint8_t>& seed_seen,
                             const std::vector<uint8_t>& is_primary_input,
                             const std::vector<uint8_t>& is_clock_pin,
                             const std::vector<uint8_t>& is_driver_pin,
                             const std::vector<uint8_t>& is_load_pin,
                             const std::vector<int>& power_fanin,
                             const std::vector<int>& pin_to_node,
                             const std::vector<int>& pin_to_net,
                             const std::vector<int>& power_pin_level_cpu);

using PowerTemplateExprAdder = std::function<int(const std::string&, LibertyCell*)>;
using PowerExprPinPredicate = std::function<bool(int, int)>;

void buildPowerCudaInternalRows(GTDatabase& gtdb,
                                int n,
                                bool need_internal_power,
                                const std::vector<uint8_t>& is_load_pin,
                                const std::vector<uint8_t>& is_driver_pin,
                                const std::vector<int>& pin_func_expr_id,
                                const std::vector<GpuPowerExprOpHost>& expr_ops,
                                const std::vector<int>& expr_start,
                                const std::vector<int>& expr_count,
                                const PowerTemplateExprAdder& add_template_expr,
                                const PowerExprPinPredicate& expr_contains_pin,
                                std::vector<GpuPowerInternalHost>& internal_rows,
                                std::unordered_map<uint64_t, int>& internal_denom_group);

void buildPowerCudaLeakageRows(GTDatabase& gtdb,
                               bool need_leakage_power,
                               const PowerTemplateExprAdder& add_template_expr,
                               std::vector<GpuPowerLeakageRowHost>& leakage_rows,
                               std::vector<GpuPowerLeakageGroupHost>& leakage_groups);

}  // namespace gt
