

#include "GTDatabase.h"

#include "common/common.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Pin.h"
#include "common/lib/Liberty.h"
#include "common/lib/Lut.h"
#include "common/lib/Timing.h"
#include "common/lib/sdc/sdc.h"
#include "io_parser/gp/GPDatabase.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <unordered_map>

namespace gt {

void gputimer_log_cuda_mem_info(const char* label);
void gputimer_empty_cuda_cache(const char* label);
bool gputimer_env_enabled(const char* name);

namespace {

bool is_transition_defined_cpu(const TimingArc* timing_arc, int input_rf, int output_rf) {
    if (timing_arc->is_rising_edge_triggered() && input_rf != RISE) return false;
    if (timing_arc->is_falling_edge_triggered() && input_rf != FALL) return false;

    switch (timing_arc->timing_sense_) {
        case TimingSense::positive_unate:
            return input_rf == output_rf;
        case TimingSense::negative_unate:
            return input_rf != output_rf;
        default:
            return true;
    }
}

bool is_clock_gating_check(const TimingArc* timing_arc) {
    if (timing_arc == nullptr || !timing_arc->is_constraint() ||
        timing_arc->liberty_port_ == nullptr || timing_arc->liberty_port_->cell_ == nullptr) {
        return false;
    }
    const std::string& cell_name = timing_arc->liberty_port_->cell_->name;
    return cell_name.find("CLKGATE") != std::string::npos ||
           cell_name.rfind("DLH_", 0) == 0 ||
           cell_name.rfind("DLL_", 0) == 0 ||
           cell_name.rfind("TLAT_", 0) == 0;
}

bool is_level_sensitive_latch_cell(const TimingArc* timing_arc) {
    if (timing_arc == nullptr ||
        timing_arc->liberty_port_ == nullptr || timing_arc->liberty_port_->cell_ == nullptr) {
        return false;
    }
    const std::string& cell_name = timing_arc->liberty_port_->cell_->name;
    return cell_name.rfind("DLH_", 0) == 0 ||
           cell_name.rfind("DLL_", 0) == 0 ||
           cell_name.rfind("TLAT_", 0) == 0;
}

bool is_latch_enable_to_q_arc(const TimingArc* timing_arc) {
    if (!is_level_sensitive_latch_cell(timing_arc) ||
        timing_arc->is_constraint() ||
        timing_arc->from_port_ == nullptr ||
        timing_arc->to_port_ == nullptr) {
        return false;
    }
    const bool edge_arc = timing_arc->timing_type_ == TimingType::rising_edge ||
                          timing_arc->timing_type_ == TimingType::falling_edge;
    if (!edge_arc) {
        return false;
    }
    const std::string& from = timing_arc->from_port_->name;
    const std::string& to = timing_arc->to_port_->name;
    return (from == "G" || from == "GN") && (to == "Q" || to == "QN");
}

bool extract_profile_enabled()
{
    const char* value = std::getenv("XPLACE_TIMER_PROFILE");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

void warn_missing_sdc_object(const char* command,
                             const char* kind,
                             const std::string& name) {
    if (!gputimer_env_enabled("GPUTIMER_VERBOSE_SDC_WARNINGS")) return;
    std::fprintf(stderr, "%s: %s \"%s\" not found\n", command, kind, name.c_str());
}

class ExtractProfileTimer {
public:
    explicit ExtractProfileTimer(bool enabled)
        : enabled_(enabled),
          start_(std::chrono::steady_clock::now()),
          last_(start_) {}

    void log(const char* phase)
    {
        if (!enabled_) {
            return;
        }
        auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_).count();
        const double total = std::chrono::duration<double>(now - start_).count();
        std::fprintf(stdout, "[XPLACE_EXTRACT_PROFILE] phase=%s elapsed=%.3f total=%.3f\n",
                     phase, elapsed, total);
        std::fflush(stdout);
        last_ = now;
    }

private:
    bool enabled_ = false;
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point last_;
};

void add_pin_name_target_variants(std::unordered_set<std::string>& targets,
                                  const std::string& name) {
    if (name.empty()) {
        return;
    }
    targets.insert(name);
    const auto slash_pos = name.rfind('/');
    if (slash_pos != std::string::npos) {
        std::string gp_pin_name = name;
        gp_pin_name[slash_pos] = ':';
        targets.insert(std::move(gp_pin_name));
    }
    const auto colon_pos = name.rfind(':');
    if (colon_pos != std::string::npos) {
        std::string sdc_pin_name = name;
        sdc_pin_name[colon_pos] = '/';
        targets.insert(std::move(sdc_pin_name));
    }
}

}  // namespace

bool GTDatabase::is_redundant_timing(const TimingArc* timing_arc, Split el) {
    if (timing_arc->from_port_->name == timing_arc->to_port_->name) return true;
    if (timing_arc->related_port_name_.empty()) return true;
    if (timing_arc->timing_type_ == TimingType::non_seq_setup_rising || timing_arc->timing_type_ == TimingType::non_seq_setup_falling || timing_arc->timing_type_ == TimingType::non_seq_hold_rising ||
        timing_arc->timing_type_ == TimingType::non_seq_hold_falling || timing_arc->timing_type_ == TimingType::clear || timing_arc->timing_type_ == TimingType::preset)
        return true;
    switch (el) {
        case MIN:
            if (timing_arc->is_max_constraint()) {
                return true;
            }
            break;
        case MAX:
            if (timing_arc->is_min_constraint()) {
                return true;
            }
            break;
    }
    return false;
}

void GTDatabase::preparePinNameMapForSdc(const sdc::SDC& sdc) {
    pin_name_map_targets.clear();
    build_full_pin_name_map = gputimer_env_enabled("GPUTIMER_BUILD_FULL_PIN_NAME_MAP");
    if (build_full_pin_name_map) {
        return;
    }

    for (const auto& command : sdc.commands) {
        const auto* clock_latency = std::get_if<sdc::SetClockLatency>(&command);
        if (clock_latency == nullptr || !clock_latency->object_list) {
            continue;
        }
        std::visit(Functors{[this](const sdc::GetPins& get_pins) {
                                for (const auto& pin_name : get_pins.pins) {
                                    add_pin_name_target_variants(pin_name_map_targets, pin_name);
                                }
                            },
                            [this](const sdc::GetPorts& get_ports) {
                                for (const auto& port_name : get_ports.ports) {
                                    add_pin_name_target_variants(pin_name_map_targets, port_name);
                                }
                            },
                            [](const auto&) {}},
                   *clock_latency->object_list);
    }
}

GTDatabase::GTDatabase(shared_ptr<db::Database> rawdb_, shared_ptr<gp::GPDatabase> gpdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_) : rawdb(*rawdb_), gpdb(*gpdb_), timing_raw_db(*timing_raw_db_) {
    cell_libs_[MIN] = rawdb.cell_libs_[MIN];
    cell_libs_[MAX] = rawdb.cell_libs_[MAX];
}


void GTDatabase::ExtractTimingGraph() {
    ExtractProfileTimer extract_profile(extract_profile_enabled());
    res_unit = cell_libs_[MIN]->resistance_unit_->value();
    cap_unit = cell_libs_[MIN]->capacitance_unit_->value();
    time_unit = cell_libs_[MIN]->time_unit_->value();
    pin_names = gpdb.getPinNames();
    net_names = gpdb.getNetNames();

    dmp_input_thresholds.assign(NUM_ATTR, 0.5f);
    dmp_output_thresholds.assign(NUM_ATTR, 0.5f);
    dmp_slew_lower_thresholds.assign(NUM_ATTR, 0.2f);
    dmp_slew_upper_thresholds.assign(NUM_ATTR, 0.8f);
    dmp_slew_derates.assign(NUM_ATTR, 1.0f);
    for_each_el_rf_if(el, rf, true) {
        const int attr = (static_cast<int>(el) << 1) + static_cast<int>(rf);
        CellLib* lib = cell_libs_[el].get();
        if (!lib) {
            continue;
        }
        dmp_input_thresholds[attr] = lib->default_input_threshold_pct[rf];
        dmp_output_thresholds[attr] = lib->default_output_threshold_pct[rf];
        dmp_slew_lower_thresholds[attr] = lib->default_slew_lower_threshold_pct[rf];
        dmp_slew_upper_thresholds[attr] = lib->default_slew_upper_threshold_pct[rf];
        dmp_slew_derates[attr] = lib->default_slew_derate_from_library;
    }
    logger.info("DMP library thresholds: in=%.3f/%.3f/%.3f/%.3f out=%.3f/%.3f/%.3f/%.3f slew=%.3f-%.3f/%.3f-%.3f/%.3f-%.3f/%.3f-%.3f derate=%.3f/%.3f/%.3f/%.3f",
                dmp_input_thresholds[0],
                dmp_input_thresholds[1],
                dmp_input_thresholds[2],
                dmp_input_thresholds[3],
                dmp_output_thresholds[0],
                dmp_output_thresholds[1],
                dmp_output_thresholds[2],
                dmp_output_thresholds[3],
                dmp_slew_lower_thresholds[0],
                dmp_slew_upper_thresholds[0],
                dmp_slew_lower_thresholds[1],
                dmp_slew_upper_thresholds[1],
                dmp_slew_lower_thresholds[2],
                dmp_slew_upper_thresholds[2],
                dmp_slew_lower_thresholds[3],
                dmp_slew_upper_thresholds[3],
                dmp_slew_derates[0],
                dmp_slew_derates[1],
                dmp_slew_derates[2],
                dmp_slew_derates[3]);
    extract_profile.log("thresholds");

    //  Flatten Liberty Cell Timing
    std::map<std::array<float, 9>, int> dmp_library_id_by_thresholds;
    std::unordered_map<const LibertyCell*, int> dmp_library_id_by_cell;
    auto register_dmp_library = [&](const LibertyCell* liberty_cell) -> int {
        if (liberty_cell == nullptr) {
            return -1;
        }
        auto cached_cell = dmp_library_id_by_cell.find(liberty_cell);
        if (cached_cell != dmp_library_id_by_cell.end()) {
            return cached_cell->second;
        }
        std::array<float, 9> threshold_key = {
            liberty_cell->input_threshold_pct[RISE],
            liberty_cell->input_threshold_pct[FALL],
            liberty_cell->output_threshold_pct[RISE],
            liberty_cell->output_threshold_pct[FALL],
            liberty_cell->slew_lower_threshold_pct[RISE],
            liberty_cell->slew_lower_threshold_pct[FALL],
            liberty_cell->slew_upper_threshold_pct[RISE],
            liberty_cell->slew_upper_threshold_pct[FALL],
            liberty_cell->slew_derate_from_library,
        };
        auto cached_thresholds = dmp_library_id_by_thresholds.find(threshold_key);
        if (cached_thresholds != dmp_library_id_by_thresholds.end()) {
            dmp_library_id_by_cell.emplace(liberty_cell, cached_thresholds->second);
            return cached_thresholds->second;
        }
        const int lib_id = static_cast<int>(dmp_library_input_thresholds.size() / MAX_TRAN);
        dmp_library_id_by_thresholds.emplace(threshold_key, lib_id);
        dmp_library_id_by_cell.emplace(liberty_cell, lib_id);
        for (auto rf : TRAN) {
            dmp_library_input_thresholds.push_back(liberty_cell->input_threshold_pct[rf]);
            dmp_library_output_thresholds.push_back(liberty_cell->output_threshold_pct[rf]);
            dmp_library_slew_lower_thresholds.push_back(liberty_cell->slew_lower_threshold_pct[rf]);
            dmp_library_slew_upper_thresholds.push_back(liberty_cell->slew_upper_threshold_pct[rf]);
            dmp_library_slew_derates.push_back(liberty_cell->slew_derate_from_library);
        }
        return lib_id;
    };
    for (db::CellType* cell_type : rawdb.celltypes) {
        string cell_type_name = cell_type->name;
        array<LibertyCell*, 2> liberty_cell_view = {cell_libs_[MIN]->get_cell(cell_type_name), cell_libs_[MAX]->get_cell(cell_type_name)};
        for_each_el(el) {
            register_dmp_library(liberty_cell_view[el]);
        }
        if (!liberty_cell_view[MIN] || !liberty_cell_view[MAX]) {
            liberty_cell_type2port_list_end.push_back(liberty_cell_type2port_list_end.back());
            for_each_el(el) {
                liberty_cell_type2leakage_power_list_end.push_back(
                    liberty_cell_type2leakage_power_list_end.back());
            }
            continue;
        }
        for_each_el(el) {
            for (auto* leakage_power : liberty_cell_view[el]->leakage_power_groups_) {
                liberty_leakage_powers.push_back(leakage_power);
            }
            liberty_cell_type2leakage_power_list_end.push_back(
                liberty_cell_type2leakage_power_list_end.back() +
                liberty_cell_view[el]->leakage_power_groups_.size());
        }
        liberty_cell_type2port_list_end.push_back(liberty_cell_type2port_list_end.back() + liberty_cell_view[MIN]->ports_.size());
        for (int i = 0; i < liberty_cell_view[MIN]->ports_.size(); i++) {
            array<LibertyPort*, 2> liberty_port_view = {liberty_cell_view[MIN]->ports_[i], liberty_cell_view[MAX]->ports_[i]};
            liberty_port_function_exprs.push_back(liberty_port_view[MAX]->function_expr_);
            liberty_port_has_function.push_back(liberty_port_view[MAX]->has_function_ ? 1 : 0);
            for_each_el(el) {
                for (auto rf : TRAN) {
                    float lib_pin_cap = liberty_port_view[el]->port_capacitances_[rf][el].value_or(nanf(""));
                    if (isnan(lib_pin_cap)) {
                        lib_pin_cap = liberty_port_view[el ^ 1]->port_capacitances_[rf][el].value_or(nanf(""));
                    }
                    if (isnan(lib_pin_cap)) {
                        lib_pin_cap = liberty_port_view[el]->port_capacitance_[rf].value_or(
                            liberty_port_view[el]->port_capacitance_[2].value_or(0.0f));
                    }
                    liberty_port_capacitance.push_back(lib_pin_cap);
                }
                liberty_port_capacitance.push_back(liberty_port_view[el]->port_capacitance_[2].value_or(0.0f));
            }

            for_each_el(el) {
                liberty_port2internal_power_list_end.push_back(
                    liberty_port2internal_power_list_end.back() +
                    liberty_port_view[el]->internal_powers_.size());
                for (auto* internal_power : liberty_port_view[el]->internal_powers_) {
                    liberty_internal_powers.push_back(internal_power);
                }
            }

            for_each_el(el) {
                liberty_port2timing_list_end.push_back(liberty_port2timing_list_end.back() + liberty_port_view[el]->timing_arcs_non_cond_non_bundle_.size());
                for (int j = 0; j < liberty_port_view[el]->timing_arcs_non_cond_non_bundle_.size(); j++) {
                    liberty_timing_arcs.push_back(liberty_port_view[el]->timing_arcs_non_cond_non_bundle_[j]);
                }
            }
        }
    }
    extract_profile.log("flatten_liberty");
    dmp_timing_library_ids.resize(liberty_timing_arcs.size(), -1);
    for (size_t timing_id = 0; timing_id < liberty_timing_arcs.size(); ++timing_id) {
        TimingArc* timing_arc = liberty_timing_arcs[timing_id];
        LibertyCell* liberty_cell = timing_arc && timing_arc->liberty_port_
                                        ? timing_arc->liberty_port_->cell_
                                        : nullptr;
        dmp_timing_library_ids[timing_id] = register_dmp_library(liberty_cell);
    }
    extract_profile.log("liberty_threshold_vectors");

    //  Traverse Circuit Pins
    //
    num_pins = gpdb.getPins().size();
    pin_names = gpdb.getPinNames();
    net_names = gpdb.getNetNames();
    pin_name2pin_id.clear();
    if (build_full_pin_name_map) {
        pin_name2pin_id.reserve(pin_names.size() * 2);
        for (int pin_id = 0; pin_id < static_cast<int>(pin_names.size()); ++pin_id) {
            const std::string& pin_name = pin_names[pin_id];
            pin_name2pin_id[pin_name] = pin_id;
            const auto pin_delim_pos = pin_name.rfind(':');
            if (pin_delim_pos != std::string::npos) {
                std::string sdc_pin_name = pin_name;
                sdc_pin_name[pin_delim_pos] = '/';
                pin_name2pin_id.emplace(std::move(sdc_pin_name), pin_id);
            }
        }
    } else if (!pin_name_map_targets.empty()) {
        pin_name2pin_id.reserve(pin_name_map_targets.size());
        for (int pin_id = 0; pin_id < static_cast<int>(pin_names.size()); ++pin_id) {
            const std::string& pin_name = pin_names[pin_id];
            if (pin_name_map_targets.find(pin_name) == pin_name_map_targets.end()) {
                continue;
            }
            pin_name2pin_id[pin_name] = pin_id;
            const auto pin_delim_pos = pin_name.rfind(':');
            if (pin_delim_pos != std::string::npos) {
                std::string sdc_pin_name = pin_name;
                sdc_pin_name[pin_delim_pos] = '/';
                pin_name2pin_id.emplace(std::move(sdc_pin_name), pin_id);
            }
        }
    }
    pin_name_map_targets.clear();
    extract_profile.log("pin_name_map");
    pin_id2cell_type_id.resize(num_pins);
    pin_id2port_offset_id.resize(num_pins);
    dmp_pin_library_ids.assign(num_pins * MAX_SPLIT, -1);
    pin_is_clk.assign(num_pins, 0);
    pin_case_values.assign(num_pins, -1);
    STA_pins.resize(num_pins, nullptr);
    pin_capacitance.resize(2 * 3 * num_pins, 0.0f);
    for (auto& gppin : gpdb.getPins()) {
        int pin_id = gppin.getId();
        string pin_name = gppin.getName();
        string pin_macro_name = gppin.getMacroName();
        STA_pins[pin_id] = new STAPin();
        auto [ori_node_id, ori_node_pin_id, ori_net_id] = gppin.getOriDBInfo();
        if (ori_node_pin_id == -1) {
            auto dbiopin = rawdb.iopins[ori_node_id];
            pin_id2cell_type_id[pin_id] = -1;
            if (dbiopin->type->direction() == 'i') {
                primary_outputs.push_back(pin_id);
                endpoints_id.push_back(pin_id);
                primary_output2pin_id[pin_name] = pin_id;
            } else if (dbiopin->type->direction() == 'o') {
                primary_inputs.push_back(pin_id);
                primary_input2pin_id[pin_name] = pin_id;
            }
        } else {
            auto& dbcell = rawdb.cells[ori_node_id];
            LibertyCell* liberty_cell = dbcell->ctype()->liberty_cell;
            pin_id2cell_type_id[pin_id] = dbcell->ctype()->libcell();
            for_each_el(el) {
                LibertyCell* corner_liberty_cell = cell_libs_[el]->get_cell(dbcell->ctype()->name);
                dmp_pin_library_ids[pin_id * MAX_SPLIT + static_cast<int>(el)] =
                    register_dmp_library(corner_liberty_cell);
            }
            if (!liberty_cell) {
                pin_id2port_offset_id[pin_id] = 0;
                continue;
            }
            pin_id2port_offset_id[pin_id] = liberty_cell->ports_map_[pin_macro_name];
            int pin_port_offset = pin_id2port_offset_id[pin_id];
            if (pin_port_offset >= 0 &&
                pin_port_offset < static_cast<int>(liberty_cell->ports_.size()) &&
                liberty_cell->ports_[pin_port_offset]->is_clock_) {
                pin_is_clk[pin_id] = 1;
            }

            int liberty_port_id = liberty_cell_type2port_list_end[pin_id2cell_type_id[pin_id]] + pin_id2port_offset_id[pin_id];

            for_each_el(el) {
                pin_capacitance[6 * pin_id + el * 2 + 0] = liberty_port_capacitance[6 * liberty_port_id + el * 3 + 0];
                pin_capacitance[6 * pin_id + el * 2 + 1] = liberty_port_capacitance[6 * liberty_port_id + el * 3 + 1];
                pin_capacitance[6 * pin_id + 4 + el] = liberty_port_capacitance[6 * liberty_port_id + el * 3 + 2];
            }
        }
    }
    num_POs = primary_outputs.size();
    extract_profile.log("traverse_pins");


    //  Map Pin to Liberty Timing
    //
    auto connect_from_to_pin = [&](int from_pin_id, int to_pin_id) -> pair<STAPin*, STAPin*> {
        STAPin* from_pin = STA_pins[from_pin_id];
        STAPin* to_pin = STA_pins[to_pin_id];
        from_pin->fanout_pin_ids.push_back(to_pin_id);
        to_pin->fanin_pin_ids.push_back(from_pin_id);
        timing_arc_from_pin_id.push_back(from_pin_id);
        timing_arc_to_pin_id.push_back(to_pin_id);
        from_pin->timing_arc_out.push_back(num_arcs);
        to_pin->timing_arc_in.push_back(num_arcs);
        return {from_pin, to_pin};
    };

    for (auto& gpnet : gpdb.getNets()) {
        int driver_pin_id = gpnet.pins()[0];
        for (index_type i = 1; i < static_cast<index_type>(gpnet.pins().size()); i++) {
            int sink_pin_id = gpnet.pins()[i];
            auto [from_pin, to_pin] = connect_from_to_pin(driver_pin_id, sink_pin_id);
            timing_arc_id_map.push_back(-1);
            timing_arc_id_map.push_back(-1);
            arc_types.push_back(0);
            arc_id2test_id.push_back(-1);
            num_arcs++;
        }
    }
    extract_profile.log("net_arcs");

    auto is_primary_input_pin = [&](int pin_id) -> bool {
        return std::find(primary_inputs.begin(), primary_inputs.end(), pin_id) != primary_inputs.end();
    };
    auto is_undriven_non_pi_pin = [&](int pin_id) -> bool {
        return pin_id >= 0 &&
               !is_primary_input_pin(pin_id) &&
               STA_pins[pin_id] != nullptr &&
               STA_pins[pin_id]->fanin_pin_ids.empty();
    };
    auto is_constant_driver_pin = [&](int pin_id) -> bool {
        if (pin_id < 0) return false;
        auto [ori_node_id, ori_node_pin_id, ori_net_id] = gpdb.getPins()[pin_id].getOriDBInfo();
        (void) ori_net_id;
        if (ori_node_pin_id == -1 || ori_node_id < 0 || ori_node_id >= static_cast<int>(rawdb.cells.size())) {
            return false;
        }
        auto& cell = rawdb.cells[ori_node_id];
        return cell && cell->ctype() && cell->ctype()->name.find("__conb_") != std::string::npos;
    };
    auto constant_driver_value = [&](int pin_id) -> int {
        if (!is_constant_driver_pin(pin_id)) {
            return -1;
        }
        const std::string& name = pin_names[pin_id];
        if (name.size() >= 3 && name.compare(name.size() - 3, 3, ":LO") == 0) {
            return 0;
        }
        if (name.size() >= 3 && name.compare(name.size() - 3, 3, ":HI") == 0) {
            return 1;
        }
        return -1;
    };
    auto constant_driven_pin_value = [&](int pin_id) -> int {
        if (pin_id < 0 || STA_pins[pin_id] == nullptr || STA_pins[pin_id]->fanin_pin_ids.empty()) {
            return -1;
        }
        int const_value = -1;
        for (int src_pin_id : STA_pins[pin_id]->fanin_pin_ids) {
            int src_value = constant_driver_value(src_pin_id);
            if (src_value < 0) {
                return -1;
            }
            if (const_value >= 0 && const_value != src_value) {
                return -1;
            }
            const_value = src_value;
        }
        return const_value;
    };

    cell_node_type_map.resize(gpdb.getNodes().size(), -1);
    for (auto& dbcell : rawdb.cells) {
        int gpdb_id = dbcell->gpdb_id;
        int libcell_id = dbcell->ctype()->libcell();
        cell_node_type_map[gpdb_id] = libcell_id;
        if (libcell_id < 0 || !dbcell->ctype()->liberty_cell) continue;
        for_each_el(el) {
            for (int pin_id : gpdb.getNodes()[gpdb_id].pins()) {
                int pin_id2port_start = liberty_cell_type2port_list_end[libcell_id];
                int pin_id2port_offset = pin_id2port_offset_id[pin_id];
                int port_id = pin_id2port_start + pin_id2port_offset;
                int start = liberty_port2timing_list_end[2 * port_id + el];
                int end = liberty_port2timing_list_end[2 * port_id + el + 1];
                for (int i = start; i < end; i++) {
                    TimingArc* timing_arc = liberty_timing_arcs[i];
                    if (is_redundant_timing(timing_arc, el)) {
                        continue;
                    }
                    array<int, 2> timing_view = {-1, -1};
                    timing_view[el] = i;

                    int from_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName(timing_arc->from_port_->name);;
                    int to_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName(timing_arc->to_port_->name);
                    if (from_pin_id < 0 || to_pin_id < 0) {
                        continue;
                    }
                    if (timing_arc->from_port_->name == "S" &&
                        timing_arc->to_port_->name == "X" &&
                        dbcell->ctype()->liberty_cell->name.find("__mux2_") != std::string::npos) {
                        const int a0_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName("A0");
                        const int a1_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName("A1");
                        const int a0_const = constant_driven_pin_value(a0_pin_id);
                        const int a1_const = constant_driven_pin_value(a1_pin_id);
                        if (is_undriven_non_pi_pin(a0_pin_id) ||
                            is_undriven_non_pi_pin(a1_pin_id) ||
                            (timing_arc->timing_sense_ == TimingSense::positive_unate &&
                             (a0_const == 1 || a1_const == 0)) ||
                            (timing_arc->timing_sense_ == TimingSense::negative_unate &&
                             (a0_const == 0 || a1_const == 1))) {
                            continue;
                        }
                    }
                    auto [from_pin, to_pin] = connect_from_to_pin(from_pin_id, to_pin_id);
                    timing_arc_id_map.push_back(timing_view[MIN]);
                    timing_arc_id_map.push_back(timing_view[MAX]);
                    arc_types.push_back(1);
                    num_arcs++;

                    if (timing_arc->is_constraint() && !is_clock_gating_check(timing_arc)) {
                        arc_id2test_id.push_back(num_tests++);
                        test_id2_arc_id.push_back(num_arcs - 1);
                        endpoints_id.push_back(to_pin_id);
                    } else {
                        arc_id2test_id.push_back(-1);
                    }
                }
            }
        }
    }
    extract_profile.log("cell_arcs");

    //  Construct Connectivity Graph
    //
    for (int i = 0; i < num_pins; i++) total_num_fanouts += STA_pins[i]->fanout_pin_ids.size();

    pin_fanout_list_end.resize(num_pins + 1);
    pin_fanout_list_end[0] = 0;
    pin_num_fanin.resize(num_pins);
    pin_fanout_list.resize(total_num_fanouts);

    index_type ptr = 0;
    index_type last_idx = 0;
    for (index_type i = 0; i < static_cast<index_type>(num_pins); i++) {
        for (auto fanout_pin_id : STA_pins[i]->fanout_pin_ids) pin_fanout_list[ptr++] = fanout_pin_id;
        last_idx += STA_pins[i]->fanout_pin_ids.size();
        pin_fanout_list_end[i + 1] = last_idx;
        pin_num_fanin[i] = STA_pins[i]->fanin_pin_ids.size();
    }
    for (int i = 0; i < num_pins; i++) {
        if (pin_num_fanin[i] == 0) pin_frontiers.push_back(i);
    }
    extract_profile.log("pin_fanout_lists");

    pin_forward_arc_list_end.push_back(0);
    pin_backward_arc_list_end.push_back(0);
    for (index_type i = 0; i < static_cast<index_type>(num_pins); i++) {
        for (auto fanout_arc : STA_pins[i]->timing_arc_out) {
            pin_forward_arc_list.push_back(fanout_arc);
        }
        pin_forward_arc_list_end.push_back(pin_forward_arc_list.size());
        for (auto fanin_arc : STA_pins[i]->timing_arc_in) {
            pin_backward_arc_list.push_back(fanin_arc);
        }
        pin_backward_arc_list_end.push_back(pin_backward_arc_list.size());
    }
    extract_profile.log("pin_arc_lists");

    // Build pin_is_clk: mark register clock pins (from_pin of test/constraint arcs)
    for (int i = 0; i < num_arcs; i++) {
        if (arc_id2test_id[i] != -1) {
            pin_is_clk[timing_arc_from_pin_id[i]] = 1;
        }
    }
    int num_clk_pins = 0;
    for (int i = 0; i < num_pins; i++) num_clk_pins += pin_is_clk[i];
    logger.info("Identified %d register clock pins for ideal_clock", num_clk_pins);

    std::unordered_map<int, int> endpoint_pin_to_compact;
    endpoint_pin_to_compact.reserve(endpoints_id.size());
    endpoint_unique_pin_ids.clear();
    auto compact_endpoint_id = [&](int pin_id) {
        auto [iter, inserted] = endpoint_pin_to_compact.emplace(pin_id, static_cast<int>(endpoint_unique_pin_ids.size()));
        if (inserted) {
            endpoint_unique_pin_ids.push_back(pin_id);
        }
        return iter->second;
    };
    primary_output2_endpoint_id.clear();
    primary_output2_endpoint_id.reserve(primary_outputs.size());
    for (int pin_id : primary_outputs) {
        primary_output2_endpoint_id.push_back(compact_endpoint_id(pin_id));
    }
    test_id2_endpoint_id.assign(test_id2_arc_id.size(), -1);
    for (int test_id = 0; test_id < static_cast<int>(test_id2_arc_id.size()); ++test_id) {
        const int arc_id = test_id2_arc_id[test_id];
        if (arc_id >= 0 && arc_id < static_cast<int>(timing_arc_to_pin_id.size())) {
            test_id2_endpoint_id[test_id] = compact_endpoint_id(timing_arc_to_pin_id[arc_id]);
        }
    }
    extract_profile.log("endpoint_compaction");

    // gputimer arrays
    auto device = timing_raw_db.node_size_x.device();
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    gputimer_log_cuda_mem_info("GTDatabase::ExtractTimingGraph before_topology_tensors");
    // Timer graph topology variables
    timing_raw_db.pin_forward_arc_list = torch::from_blob(pin_forward_arc_list.data(), {static_cast<index_type>(pin_forward_arc_list.size())}, options).contiguous().to(device);
    timing_raw_db.pin_forward_arc_list_end = torch::from_blob(pin_forward_arc_list_end.data(), {static_cast<index_type>(pin_forward_arc_list_end.size())}, options).contiguous().to(device);
    timing_raw_db.pin_backward_arc_list = torch::from_blob(pin_backward_arc_list.data(), {static_cast<index_type>(pin_backward_arc_list.size())}, options).contiguous().to(device);
    timing_raw_db.pin_backward_arc_list_end = torch::from_blob(pin_backward_arc_list_end.data(), {static_cast<index_type>(pin_backward_arc_list_end.size())}, options).contiguous().to(device);
    timing_raw_db.timing_arc_from_pin_id = torch::from_blob(timing_arc_from_pin_id.data(), {static_cast<index_type>(timing_arc_from_pin_id.size())}, options).contiguous().to(device);
    timing_raw_db.timing_arc_to_pin_id = torch::from_blob(timing_arc_to_pin_id.data(), {static_cast<index_type>(timing_arc_to_pin_id.size())}, options).contiguous().to(device);
    timing_raw_db.pin_num_fanin = torch::from_blob(pin_num_fanin.data(), {static_cast<index_type>(pin_num_fanin.size())}, options).contiguous().to(device);
    timing_raw_db.pin_fanout_list = torch::from_blob(pin_fanout_list.data(), {static_cast<index_type>(pin_fanout_list.size())}, options).contiguous().to(device);
    timing_raw_db.pin_fanout_list_end = torch::from_blob(pin_fanout_list_end.data(), {static_cast<index_type>(pin_fanout_list_end.size())}, options).contiguous().to(device);
    gputimer_log_cuda_mem_info("GTDatabase::ExtractTimingGraph after_topology_tensors");
    extract_profile.log("topology_tensors");

    // Timer timing liberty variables
    timing_raw_db.arc_types = torch::from_blob(arc_types.data(), {static_cast<int>(arc_types.size())}, options).contiguous().to(device);
    timing_raw_db.timing_arc_id_map = torch::from_blob(timing_arc_id_map.data(), {static_cast<int>(timing_arc_id_map.size())}, options).contiguous().to(device);
    timing_raw_db.arc_id2test_id = torch::from_blob(arc_id2test_id.data(), {static_cast<int>(arc_id2test_id.size())}, options).contiguous().to(device);
    timing_raw_db.test_id2_arc_id = torch::from_blob(test_id2_arc_id.data(), {static_cast<int>(test_id2_arc_id.size())}, options).contiguous().to(device);
    timing_raw_db.endpoints_id = torch::from_blob(endpoints_id.data(), {static_cast<index_type>(endpoints_id.size())}, options).contiguous().to(device);
    timing_raw_db.endpoint_unique_pin_ids = torch::from_blob(endpoint_unique_pin_ids.data(), {static_cast<index_type>(endpoint_unique_pin_ids.size())}, options).contiguous().to(device);
    timing_raw_db.test_id2_endpoint_id = torch::from_blob(test_id2_endpoint_id.data(), {static_cast<int>(test_id2_endpoint_id.size())}, options).contiguous().to(device);
    timing_raw_db.primary_output2_endpoint_id = torch::from_blob(primary_output2_endpoint_id.data(), {static_cast<int>(primary_output2_endpoint_id.size())}, options).contiguous().to(device);
    timing_raw_db.dmp_input_thresholds = torch::from_blob(dmp_input_thresholds.data(), {NUM_ATTR}, float_options).contiguous().to(device);
    timing_raw_db.dmp_output_thresholds = torch::from_blob(dmp_output_thresholds.data(), {NUM_ATTR}, float_options).contiguous().to(device);
    timing_raw_db.dmp_slew_lower_thresholds = torch::from_blob(dmp_slew_lower_thresholds.data(), {NUM_ATTR}, float_options).contiguous().to(device);
    timing_raw_db.dmp_slew_upper_thresholds = torch::from_blob(dmp_slew_upper_thresholds.data(), {NUM_ATTR}, float_options).contiguous().to(device);
    timing_raw_db.dmp_slew_derates = torch::from_blob(dmp_slew_derates.data(), {NUM_ATTR}, float_options).contiguous().to(device);
    timing_raw_db.dmp_timing_library_ids = torch::from_blob(dmp_timing_library_ids.data(), {static_cast<int>(dmp_timing_library_ids.size())}, options).contiguous().to(device);
    timing_raw_db.dmp_pin_library_ids = torch::from_blob(dmp_pin_library_ids.data(), {static_cast<int>(dmp_pin_library_ids.size())}, options).contiguous().to(device);
    timing_raw_db.dmp_library_input_thresholds = torch::from_blob(dmp_library_input_thresholds.data(), {static_cast<int>(dmp_library_input_thresholds.size())}, float_options).contiguous().to(device);
    timing_raw_db.dmp_library_output_thresholds = torch::from_blob(dmp_library_output_thresholds.data(), {static_cast<int>(dmp_library_output_thresholds.size())}, float_options).contiguous().to(device);
    timing_raw_db.dmp_library_slew_lower_thresholds = torch::from_blob(dmp_library_slew_lower_thresholds.data(), {static_cast<int>(dmp_library_slew_lower_thresholds.size())}, float_options).contiguous().to(device);
    timing_raw_db.dmp_library_slew_upper_thresholds = torch::from_blob(dmp_library_slew_upper_thresholds.data(), {static_cast<int>(dmp_library_slew_upper_thresholds.size())}, float_options).contiguous().to(device);
    timing_raw_db.dmp_library_slew_derates = torch::from_blob(dmp_library_slew_derates.data(), {static_cast<int>(dmp_library_slew_derates.size())}, float_options).contiguous().to(device);
    gputimer_log_cuda_mem_info("GTDatabase::ExtractTimingGraph after_liberty_tensors");
    extract_profile.log("liberty_tensors");

    timing_raw_db.pinSlew = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinLoad = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRAT = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinAT = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinImpulse = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.pinRootDelay = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    timing_raw_db.at_prefix_pin = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(device))).contiguous();
    timing_raw_db.at_prefix_arc = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(device))).contiguous();
    timing_raw_db.at_prefix_attr = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(device))).contiguous();
    torch::fill_(timing_raw_db.pinSlew, nanf(""));
    torch::fill_(timing_raw_db.pinRAT, nanf(""));
    torch::fill_(timing_raw_db.pinAT, nanf(""));
    torch::fill_(timing_raw_db.pinImpulse, nanf(""));
    torch::fill_(timing_raw_db.pinRootDelay, nanf(""));

    timing_raw_db.arcDelay = torch::zeros({num_arcs, 2 * NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    gputimer_log_cuda_mem_info("GTDatabase::ExtractTimingGraph after_state_tensors");
    extract_profile.log("state_tensors");
    if (!gputimer_env_enabled("GPUTIMER_DISABLE_REF_TIMING_TENSORS")) {
        timing_raw_db.pinImpulse_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
        timing_raw_db.pinLoad_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
        timing_raw_db.pinLoad_ratio = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
        timing_raw_db.pinRootDelay_ref = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
        timing_raw_db.pinRootDelay_ratio = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
        timing_raw_db.pinRootDelay_compensation = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
        gputimer_log_cuda_mem_info("GTDatabase::ExtractTimingGraph after_ref_ratio_tensors");
    } else {
        logger.info("Skipping reference/ratio timing tensors for direct route-segment eval");
        gputimer_log_cuda_mem_info("GTDatabase::ExtractTimingGraph skipped_ref_ratio_tensors");
    }
    gputimer_empty_cuda_cache("GTDatabase::ExtractTimingGraph end");

    logger.info("Design info: %d pins, %d arcs, %d tests", num_pins, num_arcs, num_tests);
}

void GTDatabase::readSdc(sdc::SDC& sdc) {
    driving_cell_sources.clear();
    clock_transitions.clear();
    clock_setup_uncertainty.clear();
    clock_hold_uncertainty.clear();
    pin_clock_latency_overrides.assign(num_pins, nanf(""));
    output_delay_clock_by_pin_attr.assign(num_pins, {});
    for (auto& pin_clocks : output_delay_clock_by_pin_attr) {
        pin_clocks.fill("");
    }
    for (auto& command : sdc.commands) {
        std::visit(Functors{[this](auto&& cmd) { _read_sdc(cmd); }}, command);
    }
    num_timings = static_cast<int>(liberty_timing_arcs.size());

    float default_period = 0.0f;
    float default_rise_edge = nanf("");
    float default_fall_edge = nanf("");
    if (!clocks.empty()) {
        default_period = clocks.begin()->second.period();
        default_rise_edge = clocks.begin()->second.rise_edge();
        default_fall_edge = clocks.begin()->second.fall_edge();
    }

    std::unordered_map<int, const Clock*> net_to_clock;
    pin_clock_periods.assign(num_pins, default_period);
    pin_clock_rise_edges.assign(num_pins, default_rise_edge);
    pin_clock_fall_edges.assign(num_pins, default_fall_edge);
    pin_clock_slews.assign(num_pins * NUM_ATTR, nanf(""));
    const Clock* default_clock = clocks.empty() ? nullptr : &clocks.begin()->second;
    std::vector<const Clock*> pin_clock_context(num_pins, default_clock);
    std::vector<char> pin_clock_context_is_default(num_pins, default_clock ? 1 : 0);
    auto apply_clock_transition_to_pin = [&](int pin_id, const Clock& clock) {
        if (auto transition_iter = clock_transitions.find(clock.name());
            transition_iter != clock_transitions.end()) {
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                const float transition = transition_iter->second[attr];
                if (std::isfinite(transition)) {
                    timing_raw_db.pinSlew[pin_id][attr] = transition;
                    pin_clock_slews[pin_id * NUM_ATTR + attr] = transition;
                }
            }
        }
    };
    for (auto& [clock_name, clock] : clocks) {
        if (clock.source_id() < 0 ||
            clock.source_id() >= static_cast<int>(gpdb.getPins().size())) {
            continue;
        }
        const int net_id = gpdb.getPins()[clock.source_id()].getParNetId();
        if (net_id >= 0) {
            net_to_clock[net_id] = &clock;
        }
        pin_clock_periods[clock.source_id()] = clock.period();
        pin_clock_rise_edges[clock.source_id()] = clock.rise_edge();
        pin_clock_fall_edges[clock.source_id()] = clock.fall_edge();
        pin_clock_context[clock.source_id()] = &clock;
        pin_clock_context_is_default[clock.source_id()] = 0;
        logger.info("clock: %s, source_pin: %s, period: %.2f, rise_edge: %.3f, fall_edge: %.3f",
                    clock_name.c_str(),
                    gpdb.getPins()[clock.source_id()].getName().c_str(),
                    clock.period(),
                    clock.rise_edge(),
                    clock.fall_edge());
    }
    for (auto& gppin : gpdb.getPins()) {
        const int net_id = gppin.getParNetId();
        auto iter = net_to_clock.find(net_id);
        if (iter != net_to_clock.end()) {
            const int pin_id = gppin.getId();
            const Clock* clock = iter->second;
            pin_clock_periods[pin_id] = clock->period();
            pin_clock_rise_edges[pin_id] = clock->rise_edge();
            pin_clock_fall_edges[pin_id] = clock->fall_edge();
            pin_clock_context[pin_id] = clock;
            pin_clock_context_is_default[pin_id] = 0;
            apply_clock_transition_to_pin(pin_id, *clock);
        }
    }
    int pin_clock_latency_override_count = 0;
    int pin_clock_latency_with_clock_context = 0;
    int pin_clock_latency_with_default_context = 0;
    int pin_clock_latency_without_context = 0;
    for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
        const float override = pin_clock_latency_overrides[pin_id];
        if (!std::isfinite(override)) {
            continue;
        }
        ++pin_clock_latency_override_count;
        const Clock* clock = pin_clock_context[pin_id];
        if (clock) {
            pin_clock_periods[pin_id] = clock->period();
            // OpenSTA gives pin clock latency precedence over clock-object
            // latency. Apply it to the waveform edge, not to clock.rise_edge()
            // / fall_edge(), which already include clock-object latency.
            pin_clock_rise_edges[pin_id] = clock->waveform_rise_edge() + override;
            pin_clock_fall_edges[pin_id] = clock->waveform_fall_edge() + override;
            if (pin_clock_context_is_default[pin_id]) {
                ++pin_clock_latency_with_default_context;
            } else {
                ++pin_clock_latency_with_clock_context;
            }
        } else {
            pin_clock_rise_edges[pin_id] = override;
            pin_clock_fall_edges[pin_id] = override;
            ++pin_clock_latency_without_context;
        }
    }
    if (pin_clock_latency_override_count > 0) {
        logger.warning("Applied %d pin set_clock_latency overrides (%d clock-net/source context, %d default-clock context, %d no-clock context)",
                       pin_clock_latency_override_count,
                       pin_clock_latency_with_clock_context,
                       pin_clock_latency_with_default_context,
                       pin_clock_latency_without_context);
    }
    if (default_clock) {
        for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
            if (!pin_is_clk[pin_id]) {
                continue;
            }
            apply_clock_transition_to_pin(pin_id, *default_clock);
        }
    }

    test_clock_periods.assign(num_tests, default_period);
    test_setup_uncertainties.assign(num_tests, 0.0f);
    test_hold_uncertainties.assign(num_tests, 0.0f);
    int tests_with_clock = 0;
    for (int test_id = 0; test_id < static_cast<int>(test_id2_arc_id.size()); ++test_id) {
        const int arc_id = test_id2_arc_id[test_id];
        if (arc_id < 0 || arc_id >= static_cast<int>(timing_arc_from_pin_id.size())) {
            continue;
        }
        const int clock_pin_id = timing_arc_from_pin_id[arc_id];
        const int net_id = gpdb.getPins()[clock_pin_id].getParNetId();
        auto iter = net_to_clock.find(net_id);
        if (iter == net_to_clock.end()) {
            continue;
        }
        const Clock* clock = iter->second;
        test_clock_periods[test_id] = clock->period();
        if (auto setup_iter = clock_setup_uncertainty.find(clock->name());
            setup_iter != clock_setup_uncertainty.end()) {
            test_setup_uncertainties[test_id] = setup_iter->second;
        }
        if (auto hold_iter = clock_hold_uncertainty.find(clock->name());
            hold_iter != clock_hold_uncertainty.end()) {
            test_hold_uncertainties[test_id] = hold_iter->second;
        }
        ++tests_with_clock;
    }
    logger.info("Mapped %d/%d timing tests to capture clocks", tests_with_clock, num_tests);

    net_is_clock.resize(gpdb.getNets().size(), 0);
    for (auto& [net_id, clock] : net_to_clock) {
        (void) clock;
        if (net_id >= 0 && net_id < static_cast<int>(net_is_clock.size())) {
            net_is_clock[net_id] = 1;
        }
    }

    // set nan slew of PIs to half period
    for (auto& pi : primary_inputs) {
        if (torch::isnan(timing_raw_db.pinSlew[pi][0]).item<bool>()) timing_raw_db.pinSlew[pi][0] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][1]).item<bool>()) timing_raw_db.pinSlew[pi][1] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][2]).item<bool>()) timing_raw_db.pinSlew[pi][2] = 0.0f;
        if (torch::isnan(timing_raw_db.pinSlew[pi][3]).item<bool>()) timing_raw_db.pinSlew[pi][3] = 0.0f;
        // if (torch::isnan(pinAT[pi][0]).item<bool>()) pinAT[pi][0] = 0.0f;
        // if (torch::isnan(pinAT[pi][1]).item<bool>()) pinAT[pi][1] = period / 2.0;
        // if (torch::isnan(pinAT[pi][2]).item<bool>()) pinAT[pi][2] = 0.0f;
        // if (torch::isnan(pinAT[pi][3]).item<bool>()) pinAT[pi][3] = period / 2.0;
    }

    for (auto& [clock_name, clock] : clocks) {
        if (clock.source_id() == -1) {
            continue;
        }
        int clock_pin_id = clock.source_id();
        pin_is_clk[clock_pin_id] = 1;
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][0]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][0] = clock.rise_edge();
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][1]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][1] = clock.fall_edge();
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][2]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][2] = clock.rise_edge();
        if (torch::isnan(timing_raw_db.pinAT[clock_pin_id][3]).item<bool>()) timing_raw_db.pinAT[clock_pin_id][3] = clock.fall_edge();
    }

    auto device = timing_raw_db.node_size_x.device();
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
    timing_raw_db.test_clock_periods = torch::from_blob(test_clock_periods.data(), {static_cast<int>(test_clock_periods.size())}, float_options).contiguous().to(device);
    timing_raw_db.test_setup_uncertainties = torch::from_blob(test_setup_uncertainties.data(), {static_cast<int>(test_setup_uncertainties.size())}, float_options).contiguous().to(device);
    timing_raw_db.test_hold_uncertainties = torch::from_blob(test_hold_uncertainties.data(), {static_cast<int>(test_hold_uncertainties.size())}, float_options).contiguous().to(device);
    timing_raw_db.pin_clock_periods = torch::from_blob(pin_clock_periods.data(), {static_cast<int>(pin_clock_periods.size())}, float_options).contiguous().to(device);
    timing_raw_db.pin_clock_rise_edges = torch::from_blob(pin_clock_rise_edges.data(), {static_cast<int>(pin_clock_rise_edges.size())}, float_options).contiguous().to(device);
    timing_raw_db.pin_clock_fall_edges = torch::from_blob(pin_clock_fall_edges.data(), {static_cast<int>(pin_clock_fall_edges.size())}, float_options).contiguous().to(device);
    timing_raw_db.pin_clock_slews = torch::from_blob(pin_clock_slews.data(), {static_cast<int>(pin_clock_slews.size())}, float_options).contiguous().to(device);
    gputimer_log_cuda_mem_info("GTDatabase::readSdc after_clock_tensors");
}

// Sets input delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetUnits& obj) {
    if (obj.time.has_value()) {
        auto s = *obj.time;
        if (s == "ps") sdc_time_unit = 1e-12;
        if (s == "ns") sdc_time_unit = 1e-9;
        if (s == "us") sdc_time_unit = 1e-6;
        if (s == "ms") sdc_time_unit = 1e-3;
        if (s == "s") sdc_time_unit = 1.0;
    }
    if (obj.capacitance.has_value()) {
        auto s = *obj.capacitance;
        if (s == "fF") sdc_cap_unit = 1e-15;
        if (s == "pF") sdc_cap_unit = 1e-12;
        if (s == "nF") sdc_cap_unit = 1e-9;
        if (s == "uF") sdc_cap_unit = 1e-6;
        if (s == "F") sdc_cap_unit = 1.0;
    }
    if (obj.resistance.has_value()) {
        auto s = *obj.resistance;
        if (s == "Ohm") sdc_res_unit = 1.0;
        if (s == "kOhm") sdc_res_unit = 1e3;
        if (s == "MOhm") sdc_res_unit = 1e6;
    }
    if (gputimer_env_enabled("GPUTIMER_VERBOSE_SDC_UNITS")) {
        if (sdc_time_unit.has_value()) printf("sdc time unit: %.2E\n", *sdc_time_unit);
        if (sdc_cap_unit.has_value()) printf("sdc capacitance unit: %.2E\n", *sdc_cap_unit);
        if (sdc_res_unit.has_value()) printf("sdc resistance unit: %.2E\n", *sdc_res_unit);
    }
}

// Sets input delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetInputDelay& obj) {
    assert(obj.delay_value && obj.port_pin_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);
    float clock_edge = 0.0f;
    if (!obj.clock.empty()) {
        if (auto clock_itr = clocks.find(obj.clock); clock_itr != clocks.end()) {
            clock_edge = obj.clock_fall ? clock_itr->second.fall_edge()
                                        : clock_itr->second.rise_edge();
        } else {
            warn_missing_sdc_object(obj.command, "clock", obj.clock);
        }
    }
    auto input_arrival = [&]() {
        float delay = *obj.delay_value;
        if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
        return clock_edge + delay;
    };

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    timing_raw_db.pinAT[pi][(el << 1) + rf] = input_arrival();
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        timing_raw_db.pinAT[itr->second][(el << 1) + rf] = input_arrival();
                                    }
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", port);
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_pin_list);
}

// Sets input transition on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetInputTransition& obj) {
    assert(obj.transition && obj.port_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float transition = *obj.transition;
                                    if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                    timing_raw_db.pinSlew[pi][(el << 1) + rf] = transition;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float transition = *obj.transition;
                                        if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
                                        timing_raw_db.pinSlew[itr->second][(el << 1) + rf] = transition;
                                    }
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", port);
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_list);
}

// Sets input transition on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetDrivingCell& obj) {
    assert((obj.transitions[0] || obj.transitions[1]) && obj.port_list);

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    auto transition_for_rf = [&](int rf) {
        float transition = obj.transitions[rf].value_or(obj.transitions[rf ^ 1].value_or(0.0f));
        if (sdc_time_unit.has_value()) transition = transition * *sdc_time_unit / time_unit;
        return transition;
    };

    auto timing_id_for_arc = [&](TimingArc* timing_arc) {
        for (int i = 0; i < static_cast<int>(liberty_timing_arcs.size()); ++i) {
            if (liberty_timing_arcs[i] == timing_arc) return i;
        }
        const int timing_id = static_cast<int>(liberty_timing_arcs.size());
        liberty_timing_arcs.push_back(timing_arc);
        return timing_id;
    };

    auto set_source_lane = [&](int pin_id, int attr, int timing_id, int input_rf, float input_slew) {
        auto iter = std::find_if(driving_cell_sources.begin(), driving_cell_sources.end(),
                                 [pin_id](const DrivingCellSource& source) {
                                     return source.pin_id == pin_id;
                                 });
        if (iter == driving_cell_sources.end()) {
            DrivingCellSource source;
            source.pin_id = pin_id;
            driving_cell_sources.push_back(source);
            iter = std::prev(driving_cell_sources.end());
        }
        iter->timing_ids[attr] = timing_id;
        iter->input_rfs[attr] = input_rf;
        iter->input_slews[attr] = input_slew;
    };

    auto record_driving_cell_source = [&](int pin_id) {
        if (!obj.lib_cell || !obj.pin) {
            return;
        }

        for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
            LibertyCell* liberty_cell = cell_libs_[el]->get_cell(*obj.lib_cell);
            if (!liberty_cell) {
                warn_missing_sdc_object(obj.command, "lib_cell", *obj.lib_cell);
                continue;
            }

            const int port_id = liberty_cell->get_port(*obj.pin);
            if (port_id < 0 || port_id >= static_cast<int>(liberty_cell->ports_.size())) {
                if (gputimer_env_enabled("GPUTIMER_VERBOSE_SDC_WARNINGS")) {
                    std::fprintf(stderr, "%s: pin %s not found in lib_cell %s\n",
                                 obj.command, obj.pin->c_str(), obj.lib_cell->c_str());
                }
                continue;
            }

            LibertyPort* output_port = liberty_cell->ports_[port_id];
            const int output_rf = static_cast<int>(rf);
            const int attr = (static_cast<int>(el) << 1) + output_rf;

            TimingArc* selected_arc = nullptr;
            int selected_input_rf = -1;
            for (TimingArc* timing_arc : output_port->timing_arcs_non_cond_non_bundle_) {
                if (!timing_arc || is_redundant_timing(timing_arc, el) || timing_arc->is_constraint()) {
                    continue;
                }
                if (!timing_arc->transition_[output_rf] || !timing_arc->transition_[output_rf]->set_) {
                    continue;
                }
                for (auto input_rf : TRAN) {
                    const int input_rf_int = static_cast<int>(input_rf);
                    if (is_transition_defined_cpu(timing_arc, input_rf_int, output_rf)) {
                        selected_arc = timing_arc;
                        selected_input_rf = input_rf_int;
                        break;
                    }
                }
                if (selected_arc) {
                    break;
                }
            }

            if (!selected_arc) {
                if (gputimer_env_enabled("GPUTIMER_VERBOSE_SDC_WARNINGS")) {
                    std::fprintf(stderr, "%s: no transition arc found for %s/%s attr %d\n",
                                 obj.command, obj.lib_cell->c_str(), obj.pin->c_str(), attr);
                }
                continue;
            }

            set_source_lane(pin_id,
                            attr,
                            timing_id_for_arc(selected_arc),
                            selected_input_rf,
                            transition_for_rf(selected_input_rf));
        }
    };

    std::visit(Functors{[&](sdc::AllInputs&) {
                            for (auto& pi : primary_inputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float transition = transition_for_rf(rf);
                                    timing_raw_db.pinSlew[pi][(el << 1) + rf] = transition;
                                }
                                record_driving_cell_source(pi);
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float transition = transition_for_rf(rf);
                                        timing_raw_db.pinSlew[itr->second][(el << 1) + rf] = transition;
                                    }
                                    record_driving_cell_source(itr->second);
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", port);
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_list);
}

// Sets output delay on pins or input ports relative to a clock signal.
void GTDatabase::_read_sdc(sdc::SetOutputDelay& obj) {
    assert(obj.delay_value && obj.port_pin_list);

    if (clocks.find(obj.clock) == clocks.end()) {
        warn_missing_sdc_object(obj.command, "clock", obj.clock);
        return;
    }

    auto& clock = clocks.at(obj.clock);
    const float clock_edge = obj.clock_fall ? clock.fall_edge() : clock.rise_edge();
    const float setup_uncertainty =
        clock_setup_uncertainty.count(obj.clock) ? clock_setup_uncertainty.at(obj.clock) : 0.0f;
    const float hold_uncertainty =
        clock_hold_uncertainty.count(obj.clock) ? clock_hold_uncertainty.at(obj.clock) : 0.0f;
    auto output_required = [&](Split el) {
        float delay = *obj.delay_value;
        if (sdc_time_unit.has_value()) delay = delay * *sdc_time_unit / time_unit;
        return el == MIN ? clock_edge - delay + hold_uncertainty
                         : clock_edge + clock.period() - delay - setup_uncertainty;
    };

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);

    std::visit(Functors{[&](sdc::AllOutputs&) {
                            for (auto& po : primary_outputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
	                                    const int attr = (el << 1) + rf;
	                                    timing_raw_db.pinRAT[po][attr] = output_required(el);
	                                    output_delay_clock_by_pin_attr[po][attr] = obj.clock;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_output2pin_id.find(port); itr != primary_output2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
	                                        const int attr = (el << 1) + rf;
	                                        timing_raw_db.pinRAT[itr->second][attr] = output_required(el);
	                                        output_delay_clock_by_pin_attr[itr->second][attr] = obj.clock;
                                    }
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", port);
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.port_pin_list);
}

// Sets the load attribute to a specified value on specified ports and nets.
void GTDatabase::_read_sdc(sdc::SetLoad& obj) {
    assert(obj.value && obj.objects);

    auto mask = sdc::TimingMask(obj.min, obj.max, std::nullopt, std::nullopt);

    std::visit(Functors{[&](sdc::AllOutputs&) {
                            for (auto& po : primary_outputs) {
                                for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                    float load = *obj.value;
                                    if (sdc_cap_unit.has_value()) load = load * *sdc_cap_unit / cap_unit;
                                    timing_raw_db.pinLoad[po][(el << 1) + rf] = load;
                                    pin_capacitance[6 * po + el * 2 + rf] = load;
                                    pin_capacitance[6 * po + 4 + el] = load;
                                }
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_output2pin_id.find(port); itr != primary_output2pin_id.end()) {
                                    for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
                                        float load = *obj.value;
                                        if (sdc_cap_unit.has_value()) load = load * *sdc_cap_unit / cap_unit;
                                        timing_raw_db.pinLoad[itr->second][(el << 1) + rf] = load;
                                        pin_capacitance[6 * itr->second + el * 2 + rf] = load;
                                        pin_capacitance[6 * itr->second + 4 + el] = load;
                                    }
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", port);
                                }
                            }
                        },
                        [](auto&&) { assert(false); }},
               *obj.objects);
}

void GTDatabase::_read_sdc(sdc::CreateClock& obj) {
    assert(obj.period && !obj.name.empty());

    // create clock from given sources
    if (obj.port_pin_list) {
        std::visit(Functors{[&](sdc::GetPorts& get_ports) {
                                auto& ports = get_ports.ports;
                                assert(ports.size() == 1);
                                if (auto itr = primary_input2pin_id.find(ports.front()); itr != primary_input2pin_id.end()) {
                                    auto [clock_iter, inserted] = clocks.try_emplace(obj.name, obj.name, itr->second, *obj.period);
                                    (void)inserted;
                                    if (obj.waveform) {
                                        clock_iter->second.set_waveform((*obj.waveform)[0], (*obj.waveform)[1]);
                                    }
                                } else {
                                    warn_missing_sdc_object(obj.command, "port", ports.front());
                                }
                            },
                            [](auto&&) { assert(false); }},
                   *obj.port_pin_list);
    }
    // create virtual clock
    else {
        auto [clock_iter, inserted] = clocks.try_emplace(obj.name, obj.name, *obj.period);
        (void)inserted;
        if (obj.waveform) {
            clock_iter->second.set_waveform((*obj.waveform)[0], (*obj.waveform)[1]);
        }
    }
}

void GTDatabase::_read_sdc(sdc::SetClockUncertainty& obj) {
    if (!obj.uncertainty) {
        return;
    }

    float uncertainty = *obj.uncertainty;
    if (sdc_time_unit.has_value()) {
        uncertainty = uncertainty * *sdc_time_unit / time_unit;
    }

    const bool applies_to_hold = obj.hold.has_value();
    const bool applies_to_setup = obj.setup.has_value() || !obj.hold.has_value();

    auto apply_clock = [&](const std::string& clock_name) {
        if (clock_name.empty()) {
            return;
        }
        float setup_delta = 0.0f;
        float hold_delta = 0.0f;
        if (applies_to_setup) {
            const float old = clock_setup_uncertainty.count(clock_name)
                                  ? clock_setup_uncertainty.at(clock_name)
                                  : 0.0f;
            setup_delta = uncertainty - old;
            clock_setup_uncertainty[clock_name] = uncertainty;
        }
        if (applies_to_hold) {
            const float old = clock_hold_uncertainty.count(clock_name)
                                  ? clock_hold_uncertainty.at(clock_name)
                                  : 0.0f;
            hold_delta = uncertainty - old;
            clock_hold_uncertainty[clock_name] = uncertainty;
        }
        if ((applies_to_setup && setup_delta != 0.0f) ||
            (applies_to_hold && hold_delta != 0.0f)) {
            for (int pin_id = 0; pin_id < static_cast<int>(output_delay_clock_by_pin_attr.size()); ++pin_id) {
                for (int attr = 0; attr < NUM_ATTR; ++attr) {
                    if (output_delay_clock_by_pin_attr[pin_id][attr] != clock_name) {
                        continue;
                    }
                    if (torch::isnan(timing_raw_db.pinRAT[pin_id][attr]).item<bool>()) {
                        continue;
                    }
                    if (attr >= 2 && applies_to_setup) {
                        timing_raw_db.pinRAT[pin_id][attr] -= setup_delta;
                    } else if (attr < 2 && applies_to_hold) {
                        timing_raw_db.pinRAT[pin_id][attr] += hold_delta;
                    }
                }
            }
        }
    };

    if (!obj.object_list) {
        for (auto& [clock_name, clock] : clocks) {
            (void) clock;
            apply_clock(clock_name);
        }
        return;
    }

    std::visit(Functors{[&](sdc::AllClocks&) {
                            for (auto& [clock_name, clock] : clocks) {
                                (void) clock;
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetClocks& get_clocks) {
                            for (auto& clock_name : get_clocks.clocks) {
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            // The Tcl SDC bridge currently returns get_clocks as a plain
                            // space-separated object string, so it arrives here as GetPorts.
                            for (auto& clock_name : get_ports.ports) {
                                apply_clock(clock_name);
                            }
                        },
                        [](auto&&) {}},
               *obj.object_list);
}

void GTDatabase::_read_sdc(sdc::SetClockTransition& obj) {
    if (!obj.transition || !obj.clock_list) {
        return;
    }

    float transition = *obj.transition;
    if (sdc_time_unit.has_value()) {
        transition = transition * *sdc_time_unit / time_unit;
    }

    auto mask = sdc::TimingMask(obj.min, obj.max, obj.rise, obj.fall);
    auto apply_clock = [&](const std::string& clock_name) {
        if (clock_name.empty()) {
            return;
        }
        auto [iter, inserted] = clock_transitions.try_emplace(clock_name);
        if (inserted) {
            iter->second.fill(nanf(""));
        }
        auto& values = iter->second;
        for_each_el_rf_if(el, rf, (mask | el) && (mask | rf)) {
            values[(el << 1) + rf] = transition;
        }
    };

    std::visit(Functors{[&](sdc::AllClocks&) {
                            for (auto& [clock_name, clock] : clocks) {
                                (void)clock;
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetClocks& get_clocks) {
                            for (auto& clock_name : get_clocks.clocks) {
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& clock_name : get_ports.ports) {
                                apply_clock(clock_name);
                            }
                        },
                        [](auto&&) {}},
               *obj.clock_list);
}

void GTDatabase::_read_sdc(sdc::SetClockLatency& obj) {
    if (!obj.delay || !obj.object_list) {
        return;
    }

    float delay = *obj.delay;
    if (sdc_time_unit.has_value()) {
        delay = delay * *sdc_time_unit / time_unit;
    }

    auto apply_clock = [&](const std::string& clock_name) {
        if (auto itr = clocks.find(clock_name); itr != clocks.end()) {
            itr->second.set_latency(delay);
        }
    };
    auto apply_pin = [&](const std::string& pin_name) {
        if (auto itr = pin_name2pin_id.find(pin_name); itr != pin_name2pin_id.end()) {
            pin_clock_latency_overrides[itr->second] = delay;
        } else if (auto pi_itr = primary_input2pin_id.find(pin_name); pi_itr != primary_input2pin_id.end()) {
            pin_clock_latency_overrides[pi_itr->second] = delay;
        }
    };

    std::visit(Functors{[&](sdc::AllClocks&) {
                            for (auto& [clock_name, clock] : clocks) {
                                (void)clock;
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetClocks& get_clocks) {
                            for (auto& clock_name : get_clocks.clocks) {
                                apply_clock(clock_name);
                            }
                        },
                        [&](sdc::GetPorts& get_ports) {
                            for (auto& name : get_ports.ports) {
                                if (clocks.find(name) != clocks.end()) {
                                    apply_clock(name);
                                } else {
                                    apply_pin(name);
                                }
                            }
                        },
                        [&](sdc::GetPins& get_pins) {
                            for (auto& pin_name : get_pins.pins) {
                                apply_pin(pin_name);
                            }
                        },
                        [](auto&&) {}},
               *obj.object_list);
}

void GTDatabase::_read_sdc(sdc::SetMaxTransition& obj) {
    // set_max_transition is a design-rule constraint in OpenSTA/OpenROAD. It does not
    // change path AT/RAT directly; parse it so benchmark SDCs no longer fall through as
    // unsupported while timing alignment is driven by delay constraints.
    (void)obj;
}

void GTDatabase::_read_sdc(sdc::SetCaseAnalysis& obj) {
    if (!obj.value || !obj.port_pin_list) {
        return;
    }
    const bool is_zero = *obj.value == "0" || *obj.value == "zero";
    const bool is_one = *obj.value == "1" || *obj.value == "one";
    if (!is_zero && !is_one) {
        return;
    }
    const int case_value = is_one ? 1 : 0;
    auto apply_pin = [&](int pin_id) {
        if (pin_id >= 0 && pin_id < static_cast<int>(pin_case_values.size())) {
            pin_case_values[pin_id] = case_value;
        }
    };
    std::visit(Functors{[&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    apply_pin(itr->second);
                                    for (int attr = 0; attr < NUM_ATTR; ++attr) {
                                        timing_raw_db.pinAT[itr->second][attr] = nanf("");
                                    }
                                }
                            }
                        },
                        [&](sdc::GetPins& get_pins) {
                            for (auto& pin_name : get_pins.pins) {
                                if (auto itr = pin_name2pin_id.find(pin_name); itr != pin_name2pin_id.end()) {
                                    apply_pin(itr->second);
                                }
                            }
                        },
                        [](auto&&) {}},
               *obj.port_pin_list);
}

void GTDatabase::_read_sdc(sdc::SetFalsePath& obj) {
    if (!obj.from) {
        return;
    }

    std::visit(Functors{[&](sdc::GetPorts& get_ports) {
                            for (auto& port : get_ports.ports) {
                                if (auto itr = primary_input2pin_id.find(port); itr != primary_input2pin_id.end()) {
                                    for (int attr = 0; attr < NUM_ATTR; ++attr) {
                                        timing_raw_db.pinAT[itr->second][attr] = nanf("");
                                    }
                                }
                            }
                        },
                        [](auto&&) {}},
               *obj.from);
}

void GTDatabase::_read_sdc(sdc::SetIdealNetwork& obj) {
    // OpenSTA ideal-network propagation changes clock/reset network delay modelling.
    // Xplace direct route_segments already uses explicit RC for signal nets; no timing
    // AT/RAT adjustment is applied here without a matched ideal-net propagation model.
    (void)obj;
}

TimingTorchRawDB::TimingTorchRawDB(torch::Tensor node_lpos_init_,
                                   torch::Tensor node_size_,
                                   torch::Tensor pin_rel_lpos_,
                                   torch::Tensor pin_id2node_id_,
                                   torch::Tensor pin_id2net_id_,
                                   torch::Tensor node2pin_list_,
                                   torch::Tensor node2pin_list_end_,
                                   torch::Tensor hyperedge_list_,
                                   torch::Tensor hyperedge_list_end_,
                                   torch::Tensor net_mask_,
                                   int num_movable_nodes_,
                                   float scale_factor_,
                                   int microns_,
                                   float wire_resistance_per_micron_,
                                   float wire_capacitance_per_micron_) {
    const auto device = node_size_.device();
    node_lpos_init = torch::Tensor();
    node_size = torch::Tensor();
    pin_rel_lpos = torch::Tensor();

    node_size_x = node_size_.index({"...", 0}).clone().contiguous();
    node_size_y = node_size_.index({"...", 1}).clone().contiguous();
    init_x = node_lpos_init_.index({"...", 0}).clone().contiguous();
    init_y = node_lpos_init_.index({"...", 1}).clone().contiguous();
    pin_offset_x = pin_rel_lpos_.index({"...", 0}).clone().contiguous();
    pin_offset_y = pin_rel_lpos_.index({"...", 1}).clone().contiguous();
    x = init_x.clone().contiguous();
    y = init_y.clone().contiguous();

    num_nodes = node_size_.size(0);
    num_pins = pin_id2node_id_.size(0);
    num_nets = hyperedge_list_end_.size(0);
    num_movable_nodes = num_movable_nodes_;
    net_mask = net_mask_.to(torch::kBool).contiguous();

    flat_node2pin_start_map = torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(device)), node2pin_list_end_}, 0).to(torch::kInt32).contiguous();
    flat_node2pin_map = node2pin_list_.to(torch::kInt32);
    pin2node_map = pin_id2node_id_.to(torch::kInt32);

    flat_net2pin_start_map = torch::cat({torch::zeros({1}, torch::dtype(torch::kInt32).device(device)), hyperedge_list_end_}, 0).to(torch::kInt32).contiguous();
    flat_net2pin_map = hyperedge_list_.to(torch::kInt32);
    pin2net_map = pin_id2net_id_.to(torch::kInt32);

    num_threads = std::max(6, 1);
    scale_factor = scale_factor_;
    microns = microns_;
    wire_resistance_per_micron = wire_resistance_per_micron_;
    wire_capacitance_per_micron = wire_capacitance_per_micron_;
}

void TimingTorchRawDB::commit_from(torch::Tensor x_, torch::Tensor y_) {
    // commit external pos to original pos
    init_x.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(x_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    init_y.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(y_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    x.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(x_.index({torch::indexing::Slice(0, num_movable_nodes)}));
    y.index({torch::indexing::Slice(0, num_movable_nodes)}).data().copy_(y_.index({torch::indexing::Slice(0, num_movable_nodes)}));
}

torch::Tensor TimingTorchRawDB::get_curr_cposx() { return x + node_size_x / 2; }
torch::Tensor TimingTorchRawDB::get_curr_cposy() { return y + node_size_y / 2; }
torch::Tensor TimingTorchRawDB::get_curr_lposx() { return x; }
torch::Tensor TimingTorchRawDB::get_curr_lposy() { return y; }

}  // namespace gt
