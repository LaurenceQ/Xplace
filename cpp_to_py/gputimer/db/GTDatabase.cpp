

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
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <omp.h>

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

void release_sta_pin_storage(vector<STAPin*>& sta_pins) {
    for (STAPin* pin : sta_pins) {
        delete pin;
    }
    vector<STAPin*>().swap(sta_pins);
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

int positive_env_int(const char* name, int fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

int graph_thread_count(int fallback)
{
    const int env_threads = positive_env_int("XPLACE_TIMER_GRAPH_THREADS", fallback);
    return std::max(1, env_threads);
}

int prefix_sum_counts(vector<int>& starts, const char* label)
{
    long long total = 0;
    for (size_t i = 0; i + 1 < starts.size(); ++i) {
        const int count = starts[i];
        starts[i] = static_cast<int>(total);
        total += count;
        if (total > std::numeric_limits<int>::max()) {
            throw std::runtime_error(std::string("Timing graph ") + label + " exceeds int index range");
        }
    }
    if (!starts.empty()) {
        starts.back() = static_cast<int>(total);
    }
    return static_cast<int>(total);
}

template <typename T>
void release_vector_storage(vector<T>& values)
{
    vector<T>().swap(values);
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

GTDatabase::GTDatabase(shared_ptr<db::Database> rawdb_, shared_ptr<gp::GPDatabase> gpdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_) : rawdb(*rawdb_), gpdb(*gpdb_), timing_raw_db(*timing_raw_db_), pin_names(gpdb.getPinNames()), net_names(gpdb.getNetNames()) {
    cell_libs_[MIN] = rawdb.cell_libs_[MIN];
    cell_libs_[MAX] = rawdb.cell_libs_[MAX];
}

GTDatabase::~GTDatabase() {
    release_sta_pin_storage(STA_pins);
    logger.info("destruct gtdb");
}


void GTDatabase::ExtractTimingGraph() {
    ExtractProfileTimer extract_profile(extract_profile_enabled());
    res_unit = cell_libs_[MIN]->resistance_unit_->value();
    cap_unit = cell_libs_[MIN]->capacitance_unit_->value();
    time_unit = cell_libs_[MIN]->time_unit_->value();

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
    const int graph_threads = graph_thread_count(timing_raw_db.num_threads);
    logger.info("Timing graph extraction threads: %d", graph_threads);
    pin_name2pin_id.clear();
    struct PinNameMapEntry {
        std::string name;
        int pin_id = -1;
        bool emplace_only = false;
    };
    auto build_pin_name_map_entries = [&](bool full_map) {
        const int pin_name_count = static_cast<int>(pin_names.size());
        std::vector<std::vector<PinNameMapEntry>> local_entries(graph_threads);
#pragma omp parallel num_threads(graph_threads)
        {
            const int tid = omp_get_thread_num();
            const int start = (pin_name_count * tid) / graph_threads;
            const int end = (pin_name_count * (tid + 1)) / graph_threads;
            auto& entries = local_entries[tid];
            if (full_map) {
                entries.reserve(static_cast<size_t>(end - start) * 2u);
            }
            for (int pin_id = start; pin_id < end; ++pin_id) {
                const std::string& pin_name = pin_names[pin_id];
                if (!full_map && pin_name_map_targets.find(pin_name) == pin_name_map_targets.end()) {
                    continue;
                }
                entries.push_back(PinNameMapEntry{pin_name, pin_id, false});
                const auto pin_delim_pos = pin_name.rfind(':');
                if (pin_delim_pos != std::string::npos) {
                    std::string sdc_pin_name = pin_name;
                    sdc_pin_name[pin_delim_pos] = '/';
                    entries.push_back(PinNameMapEntry{std::move(sdc_pin_name), pin_id, true});
                }
            }
        }
        for (auto& entries : local_entries) {
            for (auto& entry : entries) {
                if (entry.emplace_only) {
                    pin_name2pin_id.emplace(std::move(entry.name), entry.pin_id);
                } else {
                    pin_name2pin_id[entry.name] = entry.pin_id;
                }
            }
        }
    };
    if (build_full_pin_name_map) {
        pin_name2pin_id.reserve(pin_names.size() * 2);
        build_pin_name_map_entries(true);
    } else if (!pin_name_map_targets.empty()) {
        pin_name2pin_id.reserve(pin_name_map_targets.size() * 2);
        build_pin_name_map_entries(false);
    }
    pin_name_map_targets.clear();
    extract_profile.log("pin_name_map");
    pin_id2cell_type_id.resize(num_pins);
    pin_id2port_offset_id.resize(num_pins);
    dmp_pin_library_ids.assign(num_pins, -1);
    pin_is_clk.assign(num_pins, 0);
    pin_is_ideal_clk.assign(num_pins, 0);
    pin_case_values.assign(num_pins, -1);
    pin_capacitance.resize(2 * 3 * num_pins, 0.0f);
    vector<uint8_t> primary_input_pin(num_pins, 0);
    vector<uint8_t> primary_output_pin(num_pins, 0);
    const auto& gp_pins = gpdb.getPins();
    const int gp_pin_count = static_cast<int>(gp_pins.size());
    const auto& dmp_library_id_by_cell_const = dmp_library_id_by_cell;
    auto dmp_library_id_for_cell_const = [&](const LibertyCell* liberty_cell) -> int {
        auto cached_cell = dmp_library_id_by_cell_const.find(liberty_cell);
        return cached_cell != dmp_library_id_by_cell_const.end() ? cached_cell->second : -1;
    };
#pragma omp parallel for num_threads(graph_threads) schedule(static)
    for (int pin_index = 0; pin_index < gp_pin_count; ++pin_index) {
        const auto& gppin = gp_pins[pin_index];
        int pin_id = static_cast<int>(gppin.getId());
        string pin_macro_name = gppin.getMacroName();
        auto [ori_node_id, ori_node_pin_id, ori_net_id] = gppin.getOriDBInfo();
        (void) ori_net_id;
        if (ori_node_pin_id == -1) {
            auto dbiopin = rawdb.iopins[ori_node_id];
            pin_id2cell_type_id[pin_id] = -1;
            if (dbiopin->type->direction() == 'i') {
                primary_output_pin[pin_id] = 1;
            } else if (dbiopin->type->direction() == 'o') {
                primary_input_pin[pin_id] = 1;
            }
        } else {
            auto& dbcell = rawdb.cells[ori_node_id];
            LibertyCell* liberty_cell = dbcell->ctype()->liberty_cell;
            pin_id2cell_type_id[pin_id] = dbcell->ctype()->libcell();
            dmp_pin_library_ids[pin_id] = dmp_library_id_for_cell_const(liberty_cell);
            if (!liberty_cell) {
                pin_id2port_offset_id[pin_id] = 0;
                continue;
            }
            auto port_iter = liberty_cell->ports_map_.find(pin_macro_name);
            pin_id2port_offset_id[pin_id] = port_iter == liberty_cell->ports_map_.end() ? 0 : port_iter->second;
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
    for (int pin_index = 0; pin_index < gp_pin_count; ++pin_index) {
        const auto& gppin = gp_pins[pin_index];
        const int pin_id = static_cast<int>(gppin.getId());
        if (primary_output_pin[pin_id]) {
            primary_outputs.push_back(pin_id);
            endpoints_id.push_back(pin_id);
            primary_output2pin_id[gppin.getName()] = pin_id;
        } else if (primary_input_pin[pin_id]) {
            primary_inputs.push_back(pin_id);
            primary_input2pin_id[gppin.getName()] = pin_id;
        }
    }
    num_POs = primary_outputs.size();
    extract_profile.log("traverse_pins");

    const int num_nets = static_cast<int>(gpdb.getNets().size());
    const int num_cells = static_cast<int>(rawdb.cells.size());
    vector<uint8_t> primary_input_mask(num_pins, 0);
    for (int pin_id : primary_inputs) {
        if (pin_id >= 0 && pin_id < num_pins) primary_input_mask[pin_id] = 1;
    }

    vector<int> net_arc_start(num_nets + 1, 0);
    vector<int> net_driver_pin(num_nets, -1);
#pragma omp parallel for num_threads(graph_threads) schedule(static)
    for (int net_id = 0; net_id < num_nets; ++net_id) {
        const auto& pins = gpdb.getNets()[net_id].pins();
        if (!pins.empty()) {
            net_driver_pin[net_id] = static_cast<int>(pins[0]);
            net_arc_start[net_id] = static_cast<int>(pins.size()) - 1;
        }
    }
    const int num_net_arcs = prefix_sum_counts(net_arc_start, "net arcs");

    auto is_primary_input_pin = [&](int pin_id) -> bool {
        return pin_id >= 0 && pin_id < num_pins && primary_input_mask[pin_id] != 0;
    };
    auto pin_has_net_fanin = [&](int pin_id) -> bool {
        if (pin_id < 0 || pin_id >= num_pins) return false;
        const int net_id = static_cast<int>(gpdb.getPins()[pin_id].getParNetId());
        return net_id >= 0 && net_id < num_nets &&
               net_driver_pin[net_id] >= 0 && net_driver_pin[net_id] != pin_id;
    };
    auto is_undriven_non_pi_pin = [&](int pin_id) -> bool {
        return pin_id >= 0 && !is_primary_input_pin(pin_id) && !pin_has_net_fanin(pin_id);
    };
    auto is_constant_driver_pin = [&](int pin_id) -> bool {
        if (pin_id < 0 || pin_id >= num_pins) return false;
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
        if (!pin_has_net_fanin(pin_id)) {
            return -1;
        }
        const int net_id = static_cast<int>(gpdb.getPins()[pin_id].getParNetId());
        return constant_driver_value(net_driver_pin[net_id]);
    };
    auto valid_cell_timing_arc = [&](db::Cell* dbcell,
                                     int gpdb_id,
                                     int libcell_id,
                                     Split el,
                                     int timing_id,
                                     int& from_pin_id,
                                     int& to_pin_id,
                                     bool& is_test) -> bool {
        TimingArc* timing_arc = liberty_timing_arcs[timing_id];
        if (is_redundant_timing(timing_arc, el)) {
            return false;
        }
        from_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName(timing_arc->from_port_->name);
        to_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName(timing_arc->to_port_->name);
        if (from_pin_id < 0 || to_pin_id < 0) {
            return false;
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
                return false;
            }
        }
        is_test = timing_arc->is_constraint() && !is_clock_gating_check(timing_arc);
        (void) libcell_id;
        return true;
    };

    cell_node_type_map.assign(gpdb.getNodes().size(), -1);
#pragma omp parallel for num_threads(graph_threads) schedule(static)
    for (int cell_idx = 0; cell_idx < num_cells; ++cell_idx) {
        db::Cell* dbcell = rawdb.cells[cell_idx];
        if (dbcell == nullptr || dbcell->ctype() == nullptr) continue;
        const int gpdb_id = dbcell->gpdb_id;
        if (gpdb_id >= 0 && gpdb_id < static_cast<int>(cell_node_type_map.size())) {
            cell_node_type_map[gpdb_id] = dbcell->ctype()->libcell();
        }
    }
    vector<int> cell_arc_start(num_cells + 1, 0);
    vector<int> cell_test_start(num_cells + 1, 0);
#pragma omp parallel for num_threads(graph_threads) schedule(dynamic, 256)
    for (int cell_idx = 0; cell_idx < num_cells; ++cell_idx) {
        db::Cell* dbcell = rawdb.cells[cell_idx];
        if (dbcell == nullptr || dbcell->ctype() == nullptr) continue;
        const int gpdb_id = dbcell->gpdb_id;
        const int libcell_id = dbcell->ctype()->libcell();
        if (libcell_id < 0 || !dbcell->ctype()->liberty_cell ||
            gpdb_id < 0 || gpdb_id >= static_cast<int>(gpdb.getNodes().size())) {
            continue;
        }
        int arc_count = 0;
        int test_count = 0;
        for_each_el(el) {
            for (int pin_id : gpdb.getNodes()[gpdb_id].pins()) {
                int pin_id2port_start = liberty_cell_type2port_list_end[libcell_id];
                int pin_id2port_offset = pin_id2port_offset_id[pin_id];
                int port_id = pin_id2port_start + pin_id2port_offset;
                int start = liberty_port2timing_list_end[2 * port_id + el];
                int end = liberty_port2timing_list_end[2 * port_id + el + 1];
                for (int timing_id = start; timing_id < end; ++timing_id) {
                    int from_pin_id = -1;
                    int to_pin_id = -1;
                    bool is_test = false;
                    if (!valid_cell_timing_arc(dbcell, gpdb_id, libcell_id, el, timing_id,
                                               from_pin_id, to_pin_id, is_test)) {
                        continue;
                    }
                    ++arc_count;
                    if (is_test) ++test_count;
                }
            }
        }
        cell_arc_start[cell_idx] = arc_count;
        cell_test_start[cell_idx] = test_count;
    }
    const int num_cell_arcs = prefix_sum_counts(cell_arc_start, "cell arcs");
    num_tests = prefix_sum_counts(cell_test_start, "test arcs");
    num_arcs = num_net_arcs + num_cell_arcs;

    timing_arc_from_pin_id.assign(num_arcs, -1);
    timing_arc_to_pin_id.assign(num_arcs, -1);
    timing_arc_id_map.assign(static_cast<size_t>(num_arcs) * 2u, -1);
    arc_types.assign(num_arcs, 0);
    arc_id2test_id.assign(num_arcs, -1);
    test_id2_arc_id.assign(num_tests, -1);

#pragma omp parallel for num_threads(graph_threads) schedule(static)
    for (int net_id = 0; net_id < num_nets; ++net_id) {
        const auto& pins = gpdb.getNets()[net_id].pins();
        if (pins.size() <= 1) continue;
        const int driver_pin_id = static_cast<int>(pins[0]);
        int arc_id = net_arc_start[net_id];
        for (index_type i = 1; i < static_cast<index_type>(pins.size()); ++i, ++arc_id) {
            timing_arc_from_pin_id[arc_id] = driver_pin_id;
            timing_arc_to_pin_id[arc_id] = static_cast<int>(pins[i]);
        }
    }
    extract_profile.log("net_arcs");

#pragma omp parallel for num_threads(graph_threads) schedule(dynamic, 256)
    for (int cell_idx = 0; cell_idx < num_cells; ++cell_idx) {
        db::Cell* dbcell = rawdb.cells[cell_idx];
        if (dbcell == nullptr || dbcell->ctype() == nullptr) continue;
        const int gpdb_id = dbcell->gpdb_id;
        const int libcell_id = dbcell->ctype()->libcell();
        if (libcell_id < 0 || !dbcell->ctype()->liberty_cell ||
            gpdb_id < 0 || gpdb_id >= static_cast<int>(gpdb.getNodes().size())) {
            continue;
        }
        int local_arc = 0;
        int local_test = 0;
        for_each_el(el) {
            for (int pin_id : gpdb.getNodes()[gpdb_id].pins()) {
                int pin_id2port_start = liberty_cell_type2port_list_end[libcell_id];
                int pin_id2port_offset = pin_id2port_offset_id[pin_id];
                int port_id = pin_id2port_start + pin_id2port_offset;
                int start = liberty_port2timing_list_end[2 * port_id + el];
                int end = liberty_port2timing_list_end[2 * port_id + el + 1];
                for (int timing_id = start; timing_id < end; ++timing_id) {
                    int from_pin_id = -1;
                    int to_pin_id = -1;
                    bool is_test = false;
                    if (!valid_cell_timing_arc(dbcell, gpdb_id, libcell_id, el, timing_id,
                                               from_pin_id, to_pin_id, is_test)) {
                        continue;
                    }
                    const int arc_id = num_net_arcs + cell_arc_start[cell_idx] + local_arc++;
                    timing_arc_from_pin_id[arc_id] = from_pin_id;
                    timing_arc_to_pin_id[arc_id] = to_pin_id;
                    timing_arc_id_map[arc_id * 2 + static_cast<int>(el)] = timing_id;
                    arc_types[arc_id] = 1;
                    if (is_test) {
                        const int test_id = cell_test_start[cell_idx] + local_test++;
                        arc_id2test_id[arc_id] = test_id;
                        test_id2_arc_id[test_id] = arc_id;
                    }
                }
            }
        }
    }
    for (int test_id = 0; test_id < num_tests; ++test_id) {
        const int arc_id = test_id2_arc_id[test_id];
        if (arc_id >= 0 && arc_id < num_arcs) {
            endpoints_id.push_back(timing_arc_to_pin_id[arc_id]);
        }
    }
    extract_profile.log("cell_arcs");
    release_vector_storage(primary_input_mask);
    release_vector_storage(net_arc_start);
    release_vector_storage(net_driver_pin);
    release_vector_storage(cell_arc_start);
    release_vector_storage(cell_test_start);
    extract_profile.log("release_arc_build_temps");

    // Construct connectivity CSR from final arc arrays.  Scatter is serial to
    // keep each pin's arc order identical to increasing arc_id order.
    vector<int> pin_fanout_count(num_pins, 0);
    pin_num_fanin.assign(num_pins, 0);
#pragma omp parallel for num_threads(graph_threads) schedule(static)
    for (int arc_id = 0; arc_id < num_arcs; ++arc_id) {
        const int from_pin = timing_arc_from_pin_id[arc_id];
        const int to_pin = timing_arc_to_pin_id[arc_id];
        if (from_pin >= 0 && from_pin < num_pins) {
#pragma omp atomic update
            pin_fanout_count[from_pin]++;
        }
        if (to_pin >= 0 && to_pin < num_pins) {
#pragma omp atomic update
            pin_num_fanin[to_pin]++;
        }
    }
    pin_fanout_list_end.resize(num_pins + 1);
    pin_forward_arc_list_end.resize(num_pins + 1);
    pin_backward_arc_list_end.resize(num_pins + 1);
    int fanout_total = 0;
    int fanin_total = 0;
    for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
        pin_fanout_list_end[pin_id] = fanout_total;
        pin_forward_arc_list_end[pin_id] = fanout_total;
        pin_backward_arc_list_end[pin_id] = fanin_total;
        fanout_total += pin_fanout_count[pin_id];
        fanin_total += pin_num_fanin[pin_id];
    }
    pin_fanout_list_end[num_pins] = fanout_total;
    pin_forward_arc_list_end[num_pins] = fanout_total;
    pin_backward_arc_list_end[num_pins] = fanin_total;
    total_num_fanouts = fanout_total;
    pin_fanout_list.resize(fanout_total);
    pin_forward_arc_list.resize(fanout_total);
    pin_backward_arc_list.resize(fanin_total);

    std::fill(pin_fanout_count.begin(), pin_fanout_count.end(), 0);
    vector<int> pin_backward_cursor(num_pins, 0);
    for (int arc_id = 0; arc_id < num_arcs; ++arc_id) {
        const int from_pin = timing_arc_from_pin_id[arc_id];
        const int to_pin = timing_arc_to_pin_id[arc_id];
        if (from_pin >= 0 && from_pin < num_pins) {
            const int pos = pin_forward_arc_list_end[from_pin] + pin_fanout_count[from_pin]++;
            pin_forward_arc_list[pos] = arc_id;
            pin_fanout_list[pos] = to_pin;
        }
        if (to_pin >= 0 && to_pin < num_pins) {
            const int pos = pin_backward_arc_list_end[to_pin] + pin_backward_cursor[to_pin]++;
            pin_backward_arc_list[pos] = arc_id;
        }
    }
    pin_frontiers.clear();
    std::vector<std::vector<index_type>> local_frontiers(graph_threads);
#pragma omp parallel num_threads(graph_threads)
    {
        const int tid = omp_get_thread_num();
        const int start = (num_pins * tid) / graph_threads;
        const int end = (num_pins * (tid + 1)) / graph_threads;
        auto& frontiers = local_frontiers[tid];
        for (int pin_id = start; pin_id < end; ++pin_id) {
            pin_num_fanin[pin_id] = pin_backward_arc_list_end[pin_id + 1] - pin_backward_arc_list_end[pin_id];
            if (pin_num_fanin[pin_id] == 0) frontiers.push_back(pin_id);
        }
    }
    for (auto& frontiers : local_frontiers) {
        pin_frontiers.insert(pin_frontiers.end(), frontiers.begin(), frontiers.end());
    }
    release_vector_storage(pin_fanout_count);
    release_vector_storage(pin_backward_cursor);
    extract_profile.log("pin_fanout_lists");
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
    auto byte_options = torch::TensorOptions().dtype(torch::kUInt8);
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
    timing_raw_db.arc_types = torch::from_blob(arc_types.data(), {static_cast<int>(arc_types.size())}, byte_options).contiguous().to(device);
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

    if (!skip_legacy_rc_tensors) {
        timing_raw_db.pinImpulse = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
        timing_raw_db.pinRootDelay = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::Device(device))).contiguous();
    }
    timing_raw_db.at_prefix_pin = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(device))).contiguous();
    timing_raw_db.at_prefix_arc = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(device))).contiguous();
    timing_raw_db.at_prefix_attr = torch::zeros({num_pins, NUM_ATTR}, torch::dtype(torch::kInt32).device(torch::Device(device))).contiguous();
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
