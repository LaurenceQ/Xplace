

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
#include <deque>
#include <iterator>
#include <map>
#include <unordered_map>

namespace gt {

void gputimer_log_cuda_mem_info(const char* label);
void gputimer_empty_cuda_cache(const char* label);
bool gputimer_env_enabled(const char* name);

namespace {


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


Clock::Clock(const std::string& name, float period)
    : _name(name), _period(period), _fall_edge(period * 0.5f), _source_id(-1) {}

Clock::Clock(const std::string& name, int source_id, float period)
    : _name(name), _period(period), _fall_edge(period * 0.5f), _source_id(source_id) {}

const std::string& Clock::name() const { return _name; }
float Clock::period() const { return _period; }
int Clock::source_id() const { return _source_id; }
float Clock::rise_edge() const { return _rise_edge + _latency; }
float Clock::fall_edge() const { return _fall_edge + _latency; }
float Clock::waveform_rise_edge() const { return _rise_edge; }
float Clock::waveform_fall_edge() const { return _fall_edge; }
float Clock::latency() const { return _latency; }

void Clock::set_waveform(float rise_edge, float fall_edge) {
    _rise_edge = rise_edge;
    _fall_edge = fall_edge;
}

void Clock::set_latency(float latency) { _latency = latency; }

GTDatabase::GTDatabase(shared_ptr<db::Database> rawdb_, shared_ptr<gp::GPDatabase> gpdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_) : rawdb(*rawdb_), gpdb(*gpdb_), timing_raw_db(*timing_raw_db_) {
    cell_libs_[MIN] = rawdb.cell_libs_[MIN];
    cell_libs_[MAX] = rawdb.cell_libs_[MAX];
}

GTDatabase::~GTDatabase() { logger.info("destruct gtdb"); }


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
    std::map<std::array<float, 18>, int> dmp_library_id_by_thresholds;
    std::unordered_map<const LibertyCell*, int> dmp_library_id_by_cell;
    auto register_dmp_library = [&](const LibertyCell* min_cell,
                                    const LibertyCell* max_cell) -> int {
        if (min_cell == nullptr || max_cell == nullptr) {
            return -1;
        }
        auto cached_min_cell = dmp_library_id_by_cell.find(min_cell);
        if (cached_min_cell != dmp_library_id_by_cell.end()) {
            return cached_min_cell->second;
        }
        auto cached_max_cell = dmp_library_id_by_cell.find(max_cell);
        if (cached_max_cell != dmp_library_id_by_cell.end()) {
            return cached_max_cell->second;
        }
        std::array<float, 18> threshold_key = {
            min_cell->input_threshold_pct[RISE],
            min_cell->input_threshold_pct[FALL],
            min_cell->output_threshold_pct[RISE],
            min_cell->output_threshold_pct[FALL],
            min_cell->slew_lower_threshold_pct[RISE],
            min_cell->slew_lower_threshold_pct[FALL],
            min_cell->slew_upper_threshold_pct[RISE],
            min_cell->slew_upper_threshold_pct[FALL],
            min_cell->slew_derate_from_library,
            max_cell->input_threshold_pct[RISE],
            max_cell->input_threshold_pct[FALL],
            max_cell->output_threshold_pct[RISE],
            max_cell->output_threshold_pct[FALL],
            max_cell->slew_lower_threshold_pct[RISE],
            max_cell->slew_lower_threshold_pct[FALL],
            max_cell->slew_upper_threshold_pct[RISE],
            max_cell->slew_upper_threshold_pct[FALL],
            max_cell->slew_derate_from_library,
        };
        auto cached_thresholds = dmp_library_id_by_thresholds.find(threshold_key);
        if (cached_thresholds != dmp_library_id_by_thresholds.end()) {
            dmp_library_id_by_cell.emplace(min_cell, cached_thresholds->second);
            dmp_library_id_by_cell.emplace(max_cell, cached_thresholds->second);
            return cached_thresholds->second;
        }
        const int lib_id = static_cast<int>(dmp_library_input_thresholds.size() / NUM_ATTR);
        dmp_library_id_by_thresholds.emplace(threshold_key, lib_id);
        dmp_library_id_by_cell.emplace(min_cell, lib_id);
        dmp_library_id_by_cell.emplace(max_cell, lib_id);
        const std::array<const LibertyCell*, MAX_SPLIT> cells = {min_cell, max_cell};
        for_each_el_rf_if(el, rf, true) {
            const LibertyCell* liberty_cell = cells[el];
            dmp_library_input_thresholds.push_back(liberty_cell->input_threshold_pct[rf]);
            dmp_library_output_thresholds.push_back(liberty_cell->output_threshold_pct[rf]);
            dmp_library_slew_lower_thresholds.push_back(liberty_cell->slew_lower_threshold_pct[rf]);
            dmp_library_slew_upper_thresholds.push_back(liberty_cell->slew_upper_threshold_pct[rf]);
            dmp_library_slew_derates.push_back(liberty_cell->slew_derate_from_library);
        }
        return lib_id;
    };
    auto dmp_library_id_for_cell = [&](const LibertyCell* liberty_cell) -> int {
        auto cached_cell = dmp_library_id_by_cell.find(liberty_cell);
        return cached_cell != dmp_library_id_by_cell.end() ? cached_cell->second : -1;
    };
    for (db::CellType* cell_type : rawdb.celltypes) {
        string cell_type_name = cell_type->name;
        array<LibertyCell*, 2> liberty_cell_view = {cell_libs_[MIN]->get_cell(cell_type_name), cell_libs_[MAX]->get_cell(cell_type_name)};
        register_dmp_library(liberty_cell_view[MIN], liberty_cell_view[MAX]);
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
        dmp_timing_library_ids[timing_id] = dmp_library_id_for_cell(liberty_cell);
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
    dmp_pin_library_ids.assign(num_pins, -1);
    pin_is_clk.assign(num_pins, 0);
    pin_is_ideal_clk.assign(num_pins, 0);
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
            dmp_pin_library_ids[pin_id] = dmp_library_id_for_cell(liberty_cell);
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
    logger.info("Identified %d register clock pins", num_clk_pins);

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
    if (!skip_legacy_rc_tensors) {
        timing_raw_db.pinImpulse = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
        timing_raw_db.pinRootDelay = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    }
    timing_raw_db.at_prefix_pin = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(device))).contiguous();
    timing_raw_db.at_prefix_arc = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(device))).contiguous();
    timing_raw_db.at_prefix_attr = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(device))).contiguous();
    torch::fill_(timing_raw_db.pinSlew, nanf(""));
    torch::fill_(timing_raw_db.pinRAT, nanf(""));
    torch::fill_(timing_raw_db.pinAT, nanf(""));
    if (!skip_legacy_rc_tensors) {
        torch::fill_(timing_raw_db.pinImpulse, nanf(""));
        torch::fill_(timing_raw_db.pinRootDelay, nanf(""));
    } else {
        logger.info("Skipping legacy RC timing tensors pinImpulse/pinRootDelay for direct route-segment eval");
    }

    timing_raw_db.arcDelay = torch::zeros({num_arcs, 2 * NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    gputimer_log_cuda_mem_info("GTDatabase::ExtractTimingGraph after_state_tensors");
    extract_profile.log("state_tensors");
    if (!skip_legacy_rc_tensors && !gputimer_env_enabled("GPUTIMER_DISABLE_REF_TIMING_TENSORS")) {
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
