#include "PowerCudaInputBuildInternal.h"

#include "common/XplaceLog.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "gputimer/core/power/common/PowerActivityHostUtils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace gt {

namespace {
uint64_t packPowerGroupKey(int object_id, int pg_id) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(object_id)) << 32) |
           static_cast<uint32_t>(pg_id);
}

int getPowerPgId(std::unordered_map<std::string, int>& pg_ids,
                 const std::string& pg) {
    auto it = pg_ids.find(pg);
    if (it != pg_ids.end()) return it->second;
    const int id = static_cast<int>(pg_ids.size());
    pg_ids.emplace(pg, id);
    return id;
}

int addInternalDenomGroup(std::unordered_map<uint64_t, int>& groups,
                          int to_pin,
                          int related_pg_id) {
    const int id = static_cast<int>(groups.size());
    groups.emplace(packPowerGroupKey(to_pin, related_pg_id), id);
    return id;
}

int addLeakageGroup(std::vector<GpuPowerLeakageGroupHost>& groups,
                    int node_id,
                    float cell_leakage_w) {
    const int id = static_cast<int>(groups.size());
    groups.emplace_back(node_id, cell_leakage_w);
    return id;
}

int libertyPortOffset(LibertyCell* cell, LibertyPort* port, const std::string& port_name) {
    if (!cell) return -1;
    for (int i = 0; i < static_cast<int>(cell->ports_.size()); ++i) {
        LibertyPort* candidate = cell->ports_[i];
        if (!candidate) continue;
        if ((port && candidate == port) || (!port_name.empty() && candidate->name == port_name))
            return i;
    }
    return -1;
}

std::vector<size_t> countPowerNodesByLibcell(GTDatabase& gtdb, int num_libcell_slots) {
    std::vector<size_t> counts(std::max(1, num_libcell_slots), 0);
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) continue;
        const int libcell_id = gtdb.cell_node_type_map[node_id];
        if (libcell_id >= 0 && libcell_id < num_libcell_slots) ++counts[libcell_id];
    }
    return counts;
}

LibertyCell* libertyCellForLibcell(GTDatabase& gtdb, int libcell_id) {
    if (libcell_id < 0 || libcell_id >= static_cast<int>(gtdb.rawdb.celltypes.size())) return nullptr;
    auto* cell_type = gtdb.rawdb.celltypes[libcell_id];
    return cell_type ? cell_type->liberty_cell : nullptr;
}
}  // namespace


size_t readPowerChunkBytes(const char* env_name, size_t default_value) {
    const char* env = std::getenv(env_name);
    if (!env || env[0] == '\0') return default_value;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || parsed == 0) return default_value;
    return static_cast<size_t>(parsed);
}

size_t powerRowsPerChunk(size_t chunk_bytes, size_t elem_size) {
    return std::max<size_t>(1, chunk_bytes / std::max<size_t>(1, elem_size));
}

void printPowerRowStatsIfRequested(const std::vector<GpuPowerInternalHost>& h_internal_rows,
                                   const std::vector<GpuPowerLeakageRowHost>& h_leakage_rows,
                                   size_t internal_row_bytes,
                                   size_t leakage_row_bytes,
                                   size_t internal_chunk_bytes,
                                   size_t leakage_chunk_bytes,
                                   bool chunk_internal_rows,
                                   bool chunk_leakage_rows,
                                   size_t denom_group_count,
                                   size_t leakage_group_count,
                                   size_t expr_ops_count,
                                   size_t expr_bytes,
                                   size_t expr_cache_count) {
    if (!std::getenv("XPLACE_POWER_PRINT_ROW_STATS")) return;
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
    XPLACE_DEBUGF("XPLACE_POWER_ROW_STATS",
                  "internal_rows=%zu internal_bytes=%zu internal_chunk=%zu chunk_internal=%d denom_groups=%zu leakage_rows=%zu leakage_bytes=%zu leakage_chunk=%zu chunk_leakage=%d leakage_groups=%zu expr_ops=%zu expr_bytes=%zu expr_cache=%zu",
                  h_internal_rows.size(), internal_row_bytes, internal_chunk_bytes,
                  chunk_internal_rows ? 1 : 0, denom_group_count,
                  h_leakage_rows.size(), leakage_row_bytes, leakage_chunk_bytes,
                  chunk_leakage_rows ? 1 : 0, leakage_group_count,
                  expr_ops_count, expr_bytes, expr_cache_count);
    XPLACE_DEBUGF("XPLACE_POWER_ROW_STATS",
                  "internal_kind input=%zu output=%zu duty0=%zu duty1_expr=%zu duty2_diff=%zu duty3_half=%zu duty4_zero=%zu fast_duty=%zu expr_duty=%zu leakage_no_when=%zu leakage_when=%zu",
                  internal_input_rows, internal_output_rows,
                  internal_duty_modes[0], internal_duty_modes[1], internal_duty_modes[2],
                  internal_duty_modes[3], internal_duty_modes[4], internal_fast_duty_rows,
                  internal_expr_duty_rows, leakage_no_when_rows, leakage_when_rows);
}

void buildPowerCudaInternalRows(GTDatabase& gtdb,
                                int n,
                                bool need_internal_power,
                                const std::vector<uint8_t>& is_load_pin,
                                const std::vector<uint8_t>& is_driver_pin,
                                PowerCudaExprInputs& expr_inputs,
                                std::vector<GpuPowerInternalHost>& internal_rows,
                                std::unordered_map<uint64_t, int>& internal_denom_group) {
     internal_rows.clear();
     internal_denom_group.clear();
     const char* debug_power_node_env = std::getenv("XPLACE_POWER_DEBUG_NODE");
     if (!need_internal_power) return;
     size_t reserve_rows = 0;
     size_t reserve_denom_groups = 0;
     const int num_libcell_slots =
         std::max(0, static_cast<int>(gtdb.liberty_cell_type2port_list_end.size()) - 1);
     const std::vector<size_t> node_count_by_libcell =
         countPowerNodesByLibcell(gtdb, num_libcell_slots);
     for (int libcell_id = 0; libcell_id < num_libcell_slots; ++libcell_id) {
         const size_t node_count = node_count_by_libcell[libcell_id];
         if (node_count == 0) continue;
         const int port_start = gtdb.liberty_cell_type2port_list_end[libcell_id];
         const int port_end = gtdb.liberty_cell_type2port_list_end[libcell_id + 1];
         size_t rows_per_cell = 0;
         size_t denom_groups_per_cell = 0;
         for (int port_global = port_start; port_global < port_end; ++port_global) {
             const int range_idx = port_global * 2 + static_cast<int>(MAX);
             if (range_idx + 1 >= static_cast<int>(gtdb.liberty_port2internal_power_list_end.size())) continue;
             const int ip_start = gtdb.liberty_port2internal_power_list_end[range_idx];
             const int ip_end = gtdb.liberty_port2internal_power_list_end[range_idx + 1];
             if (ip_end <= ip_start) continue;
             rows_per_cell += static_cast<size_t>(ip_end - ip_start);
             ++denom_groups_per_cell;
         }
         reserve_rows += node_count * rows_per_cell;
         reserve_denom_groups += node_count * denom_groups_per_cell;
     }
     internal_rows.reserve(reserve_rows);
     internal_denom_group.reserve(reserve_denom_groups);
     std::unordered_map<std::string, int> pg_ids;
    const int num_internal_power_defs = static_cast<int>(gtdb.liberty_internal_powers.size());
    std::vector<int> ip_when_expr_id(num_internal_power_defs, -2);
    std::vector<int> ip_related_port_offset(num_internal_power_defs, -2);
    std::vector<int> ip_positive_unate(num_internal_power_defs, -1);
    std::vector<int> ip_pg_id(num_internal_power_defs, -1);
    auto cached_when_expr_id = [&](int ip_id, InternalPower* ip, LibertyCell* cell) {
        if (ip_id < 0 || ip_id >= num_internal_power_defs || !ip) return -1;
        int& cached = ip_when_expr_id[ip_id];
        if (cached != -2) return cached;
        cached = ip->when_expr_.empty() ? -1 : expr_inputs.addTemplateExpr(ip->when_expr_, cell);
        return cached;
    };
    auto cached_related_offset = [&](int ip_id, InternalPower* ip, LibertyCell* cell) {
        if (ip_id < 0 || ip_id >= num_internal_power_defs || !ip) return -1;
        int& cached = ip_related_port_offset[ip_id];
        if (cached != -2) return cached;
        cached = libertyPortOffset(cell, ip->related_port_, ip->related_port_name_);
        return cached;
    };
    auto cached_positive_unate = [&](int ip_id, LibertyCell* cell, LibertyPort* from, LibertyPort* to) {
        if (ip_id < 0 || ip_id >= num_internal_power_defs) return 1;
        int& cached = ip_positive_unate[ip_id];
        if (cached >= 0) return cached;
        cached = positiveUnateForPower(cell, from, to) ? 1 : 0;
        return cached;
    };
    auto cached_pg_id = [&](int ip_id, InternalPower* ip) {
        if (ip_id < 0 || ip_id >= num_internal_power_defs || !ip) return getPowerPgId(pg_ids, "");
        int& cached = ip_pg_id[ip_id];
        if (cached >= 0) return cached;
        const std::string pg = ip->related_pg_pin_ ? ip->related_pg_pin_->name : ip->related_pg_pin_name_;
        cached = getPowerPgId(pg_ids, pg);
        return cached;
    };
    std::vector<int> port_pin_by_offset;
    std::vector<std::pair<int, int>> denom_group_by_pg;

    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) continue;
        const int libcell_id = gtdb.cell_node_type_map[node_id];
        LibertyCell* cell = libertyCellForLibcell(gtdb, libcell_id);
        if (!cell) continue;
        if (libcell_id < 0 || libcell_id + 1 >= static_cast<int>(gtdb.liberty_cell_type2port_list_end.size())) continue;
        const int port_base = gtdb.liberty_cell_type2port_list_end[libcell_id];
        port_pin_by_offset.assign(cell->ports_.size(), -1);
        for (int pin_id : node.pins()) {
            if (pin_id < 0 || pin_id >= n) continue;
            const int port_offset = gtdb.pin_id2port_offset_id[pin_id];
            if (port_offset >= 0 && port_offset < static_cast<int>(port_pin_by_offset.size()))
                port_pin_by_offset[port_offset] = pin_id;
        }
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

            if (is_load_pin[pin_id]) {
                for (int ip_id = ip_start; ip_id < ip_end; ++ip_id) {
                    InternalPower* ip = gtdb.liberty_internal_powers[ip_id];
                    if (!ip) continue;
                    uint8_t duty_mode = 0;
                    int duty_expr_id = -1;
                    int duty_pin = -1;
                    const int when_expr_id = cached_when_expr_id(ip_id, ip, cell);
                    if (when_expr_id >= 0) {
                        duty_mode = 1;
                        duty_expr_id = when_expr_id;
                        for (int op_i = expr_inputs.start[when_expr_id]; op_i < expr_inputs.start[when_expr_id] + expr_inputs.count[when_expr_id]; ++op_i) {
                            int out_pin = expr_inputs.ops[op_i].op == 0 ? expr_inputs.ops[op_i].arg : -1;
                            if (out_pin < -1) {
                                const int port_id = -2 - out_pin;
                                if (port_id >= 0 && port_id < static_cast<int>(port_pin_by_offset.size()))
                                    out_pin = port_pin_by_offset[port_id];
                            }
                            if (out_pin >= 0 && out_pin < n && is_driver_pin[out_pin]) {
                                const int func_expr_id = expr_inputs.pin_expr_id[out_pin];
                                if (expr_inputs.containsPin(gtdb, func_expr_id, pin_id)) {
                                    duty_mode = 2;
                                    duty_expr_id = func_expr_id;
                                    duty_pin = pin_id;
                                    break;
                                }
                            }
                        }
                    }
                    GpuPowerInternalHost row(ip_id,
                                             node_id,
                                             pin_id,
                                             -1,
                                             duty_expr_id,
                                             duty_pin,
                                             -1,
                                             ip->energy_unit_,
                                             0,
                                             duty_mode,
                                             1);
                    if (debug_power_node_env && node.getName().find(debug_power_node_env) != std::string::npos) {
                        XPLACE_DEBUGF("XPLACE_POWER_DEBUG_NODE",
                                      "node=%s port=%s kind=input ip=%d when='%s' duty_mode=%d duty_expr=%d",
                                      node.getName().c_str(), port->name.c_str(), ip_id,
                                      ip->when_expr_.c_str(), row.duty_mode, row.duty_expr_id);
                    }
                    internal_rows.push_back(row);
                }
            }

            if (is_driver_pin[pin_id]) {
                const int func_expr_id = expr_inputs.pin_expr_id[pin_id];
                denom_group_by_pg.clear();
                for (int ip_id = ip_start; ip_id < ip_end; ++ip_id) {
                    InternalPower* ip = gtdb.liberty_internal_powers[ip_id];
                    if (!ip) continue;
                    int from_pin = -1;
                    int duty_expr_id = -1;
                    int duty_pin = -1;
                    int denom_group = -1;
                    uint8_t duty_mode = 4;
                    uint8_t positive_unate = 1;
                    LibertyPort* from_port = ip->related_port_;
                    const int from_offset = cached_related_offset(ip_id, ip, cell);
                    if (!from_port && from_offset >= 0 && from_offset < static_cast<int>(cell->ports_.size()))
                        from_port = cell->ports_[from_offset];
                    if (from_offset >= 0 && from_offset < static_cast<int>(port_pin_by_offset.size()) &&
                        port_pin_by_offset[from_offset] >= 0) {
                        from_pin = port_pin_by_offset[from_offset];
                        positive_unate = cached_positive_unate(ip_id, cell, from_port, port);
                        const int when_expr_id = cached_when_expr_id(ip_id, ip, cell);
                        if (expr_inputs.containsPin(gtdb, func_expr_id, from_pin)) {
                            duty_mode = 2;
                            duty_expr_id = func_expr_id;
                            duty_pin = from_pin;
                        } else if (when_expr_id >= 0) {
                            duty_mode = 1;
                            duty_expr_id = when_expr_id;
                        } else {
                            duty_mode = 3;
                        }
                        const int pg_id = cached_pg_id(ip_id, ip);
                        int denom_group_id = -1;
                        for (const auto& entry : denom_group_by_pg) {
                            if (entry.first == pg_id) {
                                denom_group_id = entry.second;
                                break;
                            }
                        }
                        if (denom_group_id < 0) {
                            denom_group_id = addInternalDenomGroup(internal_denom_group, pin_id, pg_id);
                            denom_group_by_pg.emplace_back(pg_id, denom_group_id);
                        }
                        denom_group = denom_group_id;
                    }
                    GpuPowerInternalHost row(ip_id,
                                             node_id,
                                             pin_id,
                                             from_pin,
                                             duty_expr_id,
                                             duty_pin,
                                             denom_group,
                                             ip->energy_unit_,
                                             1,
                                             duty_mode,
                                             positive_unate);
                    if (debug_power_node_env && node.getName().find(debug_power_node_env) != std::string::npos) {
                        XPLACE_DEBUGF("XPLACE_POWER_DEBUG_NODE",
                                      "node=%s port=%s kind=output ip=%d related=%s when='%s' duty_mode=%d duty_expr=%d from_pin=%d",
                                      node.getName().c_str(), port->name.c_str(), ip_id,
                                      ip->related_port_name_.c_str(), ip->when_expr_.c_str(),
                                      row.duty_mode, row.duty_expr_id, row.from_pin);
                    }
                    internal_rows.push_back(row);
                }
            }
        }
    }
}

void buildPowerCudaLeakageRows(GTDatabase& gtdb,
                               bool need_leakage_power,
                               PowerCudaExprInputs& expr_inputs,
                               std::vector<GpuPowerLeakageRowHost>& leakage_rows,
                               std::vector<GpuPowerLeakageGroupHost>& leakage_groups) {
     leakage_rows.clear();
     leakage_groups.clear();
     if (!need_leakage_power) return;
     size_t reserve_rows = 0;
     size_t reserve_groups = 0;
     const int num_libcell_slots =
         static_cast<int>(gtdb.liberty_cell_type2leakage_power_list_end.size() / 2) + 1;
     const std::vector<size_t> node_count_by_libcell =
         countPowerNodesByLibcell(gtdb, num_libcell_slots);
     for (int libcell_id = 0; libcell_id < num_libcell_slots; ++libcell_id) {
         const size_t node_count = node_count_by_libcell[libcell_id];
         if (node_count == 0) continue;
         if (libcell_id * 2 + static_cast<int>(MAX) + 1 >=
             static_cast<int>(gtdb.liberty_cell_type2leakage_power_list_end.size()))
             continue;
         const int leak_range_idx = libcell_id * 2 + static_cast<int>(MAX);
         const int leak_start = gtdb.liberty_cell_type2leakage_power_list_end[leak_range_idx];
         const int leak_end = gtdb.liberty_cell_type2leakage_power_list_end[leak_range_idx + 1];
         if (leak_end > leak_start) {
             const size_t count = static_cast<size_t>(leak_end - leak_start);
             reserve_rows += node_count * count;
             reserve_groups += node_count * count;
         } else {
             reserve_groups += node_count;
         }
     }
     leakage_rows.reserve(reserve_rows);
     leakage_groups.reserve(std::min(reserve_groups, gtdb.gpdb.getNodes().size()));
     const float max_power_unit = (gtdb.cell_libs_[MAX] && gtdb.cell_libs_[MAX]->power_unit_.has_value())
        ? static_cast<float>(gtdb.cell_libs_[MAX]->power_unit_->value()) : 1.0f;
    std::unordered_map<std::string, int> pg_ids;
    const int num_leakage_power_defs = static_cast<int>(gtdb.liberty_leakage_powers.size());
    std::vector<int> leakage_when_expr_id(num_leakage_power_defs, -2);
    std::vector<int> leakage_pg_id(num_leakage_power_defs, -1);
    auto cached_leakage_when_expr_id = [&](int leak_id, LeakagePower* lp, LibertyCell* expr_cell) {
        if (leak_id < 0 || leak_id >= num_leakage_power_defs || !lp) return -1;
        int& cached = leakage_when_expr_id[leak_id];
        if (cached != -2) return cached;
        cached = lp->when_expr_.empty() ? -1 : expr_inputs.addTemplateExpr(lp->when_expr_, expr_cell);
        return cached;
    };
    auto cached_leakage_pg_id = [&](int leak_id, LeakagePower* lp) {
        if (leak_id < 0 || leak_id >= num_leakage_power_defs || !lp) return getPowerPgId(pg_ids, "");
        int& cached = leakage_pg_id[leak_id];
        if (cached >= 0) return cached;
        const std::string pg = lp->related_pg_pin_ ? lp->related_pg_pin_->name : lp->related_pg_pin_name_;
        cached = getPowerPgId(pg_ids, pg);
        return cached;
    };
    std::vector<uint8_t> leakage_cell_cache_valid(std::max(1, num_libcell_slots), 0);
    std::vector<float> cell_leakage_w_cache(std::max(1, num_libcell_slots), 0.0f);
    std::vector<LibertyCell*> leak_expr_cell_cache(std::max(1, num_libcell_slots), nullptr);
    std::vector<std::pair<int, int>> leakage_group_by_pg;
    for (const auto& node : gtdb.gpdb.getNodes()) {
        const int node_id = static_cast<int>(node.getId());
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.cell_node_type_map.size())) continue;
        const int libcell_id = gtdb.cell_node_type_map[node_id];
        LibertyCell* cell = libertyCellForLibcell(gtdb, libcell_id);
        if (!cell) continue;
        if (libcell_id < 0 || libcell_id * 2 + static_cast<int>(MAX) + 1 >= static_cast<int>(gtdb.liberty_cell_type2leakage_power_list_end.size())) continue;
        const int leak_range_idx = libcell_id * 2 + static_cast<int>(MAX);
        const int leak_start = gtdb.liberty_cell_type2leakage_power_list_end[leak_range_idx];
        const int leak_end = gtdb.liberty_cell_type2leakage_power_list_end[leak_range_idx + 1];
        if (libcell_id >= static_cast<int>(leakage_cell_cache_valid.size())) continue;
        if (!leakage_cell_cache_valid[libcell_id]) {
            LibertyCell* cell_leakage_cell = (gtdb.cell_libs_[MIN] ? gtdb.cell_libs_[MIN]->get_cell(cell->name) : nullptr);
            if (!cell_leakage_cell) cell_leakage_cell = cell;
            LibertyCell* leak_expr_cell = (gtdb.cell_libs_[MAX] ? gtdb.cell_libs_[MAX]->get_cell(cell->name) : nullptr);
            if (!leak_expr_cell) leak_expr_cell = cell;
            cell_leakage_w_cache[libcell_id] = cell_leakage_cell->leakage_power_.value_or(0.0f) * max_power_unit;
            leak_expr_cell_cache[libcell_id] = leak_expr_cell;
            leakage_cell_cache_valid[libcell_id] = 1;
        }
        const float cell_leakage_w = cell_leakage_w_cache[libcell_id];
        LibertyCell* leak_expr_cell = leak_expr_cell_cache[libcell_id] ? leak_expr_cell_cache[libcell_id] : cell;
        if (leak_start == leak_end) {
            addLeakageGroup(leakage_groups, node_id, cell_leakage_w);
            continue;
        }
        leakage_group_by_pg.clear();
        for (int leak_id = leak_start; leak_id < leak_end; ++leak_id) {
            LeakagePower* lp = gtdb.liberty_leakage_powers[leak_id];
            if (!lp) continue;
            const int pg_id = cached_leakage_pg_id(leak_id, lp);
            int group_id = -1;
            for (const auto& entry : leakage_group_by_pg) {
                if (entry.first == pg_id) {
                    group_id = entry.second;
                    break;
                }
            }
            if (group_id < 0) {
                group_id = addLeakageGroup(leakage_groups, node_id, cell_leakage_w);
                leakage_group_by_pg.emplace_back(pg_id, group_id);
            }
            leakage_rows.emplace_back(node_id,
                                      group_id,
                                      leak_id,
                                      cached_leakage_when_expr_id(leak_id, lp, leak_expr_cell),
                                      lp->value_ * max_power_unit);
        }
    }
}

}  // namespace gt
