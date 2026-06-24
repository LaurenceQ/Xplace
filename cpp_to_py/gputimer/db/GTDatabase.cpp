

#include "GTDatabase.h"
#include "sdc/SdcUtils.h"

#include "common/common.h"
#include "common/StageProfiler.h"
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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp.h>

namespace gt {

void gputimer_log_cuda_mem_info(const char* label);
void gputimer_empty_cuda_cache(const char* label);
bool gputimer_env_enabled(const char* name);

namespace {

struct CellTimingArc {
    int from_pin_id = -1;
    int to_pin_id = -1;
    int timing_id = -1;
    bool el = false;
    uint8_t is_test = 0;
};

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
    return xplace_env_enabled("XPLACE_TIMER_PROFILE");
}

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

int prefix_sum_counts(vector<int>& starts, const char* label, int num_threads)
{
    if (starts.empty()) {
        return 0;
    }
    const size_t count_size = starts.size() - 1;
    if (count_size == 0) {
        starts.back() = 0;
        return 0;
    }
    auto overflow_error = [&]() {
        throw std::runtime_error(std::string("Timing graph ") + label + " exceeds int index range");
    };
    auto serial_scan = [&]() -> int {
        long long total = 0;
        for (size_t i = 0; i < count_size; ++i) {
            const int count = starts[i];
            starts[i] = static_cast<int>(total);
            total += count;
            if (total > std::numeric_limits<int>::max()) {
                overflow_error();
            }
        }
        starts.back() = static_cast<int>(total);
        return static_cast<int>(total);
    };

    if (num_threads <= 2 || count_size < 4096) {
        return serial_scan();
    }
    const int workers = num_threads;
    vector<long long> block_offsets(workers, 0);

#pragma omp parallel num_threads(workers)
    {
        const int tid = omp_get_thread_num();
        const size_t start = (count_size * static_cast<size_t>(tid)) / workers;
        const size_t end = (count_size * static_cast<size_t>(tid + 1)) / workers;
        long long block_total = 0;
        for (size_t i = start; i < end; ++i) {
            block_total += starts[i];
        }
        block_offsets[tid] = block_total;
    }

    long long total = 0;
    for (int tid = 0; tid < workers; ++tid) {
        const long long block_total = block_offsets[tid];
        block_offsets[tid] = total;
        total += block_total;
        if (total > std::numeric_limits<int>::max()) {
            overflow_error();
        }
    }

#pragma omp parallel num_threads(workers)
    {
        const int tid = omp_get_thread_num();
        const size_t start = (count_size * static_cast<size_t>(tid)) / workers;
        const size_t end = (count_size * static_cast<size_t>(tid + 1)) / workers;
        long long running = block_offsets[tid];
        for (size_t i = start; i < end; ++i) {
            const int count = starts[i];
            starts[i] = static_cast<int>(running);
            running += count;
        }
    }

    starts.back() = static_cast<int>(total);
    return static_cast<int>(total);
}

template <typename T>
void release_vector_storage(vector<T>& values)
{
    vector<T>().swap(values);
}

std::array<float, 18> dmp_library_threshold_key(const LibertyCell* min_cell,
                                                const LibertyCell* max_cell)
{
    return {
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
}

std::pair<int, int> BuildNetCellArcAndTest(
    GTDatabase& db,
    int graph_threads,
    vector<int>& net_arc_start,
    vector<vector<CellTimingArc>>& local_cell_timing_arcs,
    vector<int>& thread_cell_arc_start,
    vector<int>& thread_cell_test_start)
{
    auto& rawdb = db.rawdb;
    auto& gpdb = db.gpdb;
    auto& pin_forward_arc_list_end = db.pin_forward_arc_list_end;
    auto& pin_backward_arc_list_end = db.pin_backward_arc_list_end;
    auto& cell_node_type_map = db.cell_node_type_map;
    auto& liberty_timing_arcs = db.liberty_timing_arcs;
    auto& liberty_cell_type2port_list_end = db.liberty_cell_type2port_list_end;
    auto& pin_id2port_offset_id = db.pin_id2port_offset_id;
    auto& liberty_port2timing_list_end = db.liberty_port2timing_list_end;
    const int num_pins = db.num_pins;
    const int num_nets = static_cast<int>(gpdb.getNets().size());
    const int num_cells = static_cast<int>(rawdb.cells.size());

    pin_forward_arc_list_end.assign(num_pins + 1, 0);
    pin_backward_arc_list_end.assign(num_pins + 1, 0);
    net_arc_start.assign(num_nets + 1, 0);
#pragma omp parallel for num_threads(graph_threads) schedule(static)
    for (int net_id = 0; net_id < num_nets; ++net_id) {
        const auto& pins = gpdb.getNets()[net_id].pins();
        if (pins.size() > 1) {
            const int sink_count = static_cast<int>(pins.size()) - 1;
            net_arc_start[net_id] = sink_count;
            const int driver_pin_id = static_cast<int>(pins[0]);
            if (driver_pin_id >= 0 && driver_pin_id < num_pins) {
#pragma omp atomic update
                pin_forward_arc_list_end[driver_pin_id] += sink_count;
            }
            for (index_type i = 1; i < static_cast<index_type>(pins.size()); ++i) {
                const int sink_pin_id = static_cast<int>(pins[i]);
                if (sink_pin_id >= 0 && sink_pin_id < num_pins) {
#pragma omp atomic update
                    pin_backward_arc_list_end[sink_pin_id]++;
                }
            }
        }
    }
    const int num_net_arcs = prefix_sum_counts(net_arc_start, "net arcs", graph_threads);

    auto valid_cell_timing_arc = [&](int gpdb_id,
                                     Split el,
                                     int timing_id,
                                     int& from_pin_id,
                                     int& to_pin_id,
                                     bool& is_test) -> bool {
        TimingArc* timing_arc = liberty_timing_arcs[timing_id];
        if (db.is_redundant_timing(timing_arc, el)) {
            return false;
        }
        from_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName(timing_arc->from_port_->name);
        to_pin_id = gpdb.getNodes()[gpdb_id].getPinbyPortName(timing_arc->to_port_->name);
        if (from_pin_id < 0 || to_pin_id < 0) {
            return false;
        }
        is_test = timing_arc->is_constraint() && !is_clock_gating_check(timing_arc);
        return true;
    };

    cell_node_type_map.assign(gpdb.getNodes().size(), -1);
    local_cell_timing_arcs.clear();
    local_cell_timing_arcs.resize(graph_threads);
    vector<int> local_cell_test_counts(graph_threads, 0);
#pragma omp parallel num_threads(graph_threads)
    {
        const int tid = omp_get_thread_num();
        const int start = (num_cells * tid) / graph_threads;
        const int end = (num_cells * (tid + 1)) / graph_threads;
        auto& thread_cell_timing_arcs = local_cell_timing_arcs[tid];
        int thread_cell_test_count = 0;
        for (int cell_idx = start; cell_idx < end; ++cell_idx) {
            db::Cell* dbcell = rawdb.cells[cell_idx];
            if (dbcell == nullptr || dbcell->ctype() == nullptr) continue;
            const int gpdb_id = dbcell->gpdb_id;
            const int libcell_id = dbcell->ctype()->libcell();
            if (libcell_id < 0 || !dbcell->ctype()->liberty_cell ||
                gpdb_id < 0 || gpdb_id >= static_cast<int>(gpdb.getNodes().size())) {
                continue;
            }
            cell_node_type_map[gpdb_id] = libcell_id;
            for_each_el(el) {
                for (int pin_id : gpdb.getNodes()[gpdb_id].pins()) {
                    int pin_id2port_start = liberty_cell_type2port_list_end[libcell_id];
                    int pin_id2port_offset = pin_id2port_offset_id[pin_id];
                    int port_id = pin_id2port_start + pin_id2port_offset;
                    const int timing_start = liberty_port2timing_list_end[2 * port_id + el];
                    const int timing_end = liberty_port2timing_list_end[2 * port_id + el + 1];
                    for (int timing_id = timing_start; timing_id < timing_end; ++timing_id) {
                        int from_pin_id = -1;
                        int to_pin_id = -1;
                        bool is_test = false;
                        if (!valid_cell_timing_arc(gpdb_id, el, timing_id,
                                                   from_pin_id, to_pin_id, is_test)) {
                            continue;
                        }
                        thread_cell_timing_arcs.push_back(CellTimingArc{
                            from_pin_id,
                            to_pin_id,
                            timing_id,
                            el == MAX,
                            static_cast<uint8_t>(is_test ? 1 : 0)});
                        if (from_pin_id >= 0 && from_pin_id < num_pins) {
#pragma omp atomic update
                            pin_forward_arc_list_end[from_pin_id]++;
                        }
                        if (to_pin_id >= 0 && to_pin_id < num_pins) {
#pragma omp atomic update
                            pin_backward_arc_list_end[to_pin_id]++;
                        }
                        if (is_test) {
                            ++thread_cell_test_count;
                        }
                    }
                }
            }
        }
        local_cell_test_counts[tid] = thread_cell_test_count;
    }
    thread_cell_arc_start.assign(graph_threads + 1, 0);
    thread_cell_test_start.assign(graph_threads + 1, 0);
    for (int tid = 0; tid < graph_threads; ++tid) {
        if (local_cell_timing_arcs[tid].size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("Timing graph cell arcs exceed int index range");
        }
        thread_cell_arc_start[tid] = static_cast<int>(local_cell_timing_arcs[tid].size());
        thread_cell_test_start[tid] = local_cell_test_counts[tid];
    }
    const int num_cell_arcs = prefix_sum_counts(thread_cell_arc_start, "cell arcs", graph_threads);
    const int num_tests = prefix_sum_counts(thread_cell_test_start, "test arcs", graph_threads);
    return {num_net_arcs + num_cell_arcs, num_tests};
}

}  // namespace

void GTDatabase::AllocatePinArcListStorage(
    int graph_threads,
    vector<index_type>& pin_forward_arc_cursor,
    vector<index_type>& pin_backward_arc_cursor)
{
    timing_arc_from_pin_id.assign(num_arcs, -1);
    timing_arc_to_pin_id.assign(num_arcs, -1);
    timing_arc_id_map.assign(static_cast<size_t>(num_arcs) * 2u, -1);
    arc_types.assign(num_arcs, 0);
    arc_id2test_id.assign(num_arcs, -1);
    test_id2_arc_id.assign(num_tests, -1);

    const int fanout_total = prefix_sum_counts(pin_forward_arc_list_end, "pin forward arcs", graph_threads);
    const int fanin_total = prefix_sum_counts(pin_backward_arc_list_end, "pin backward arcs", graph_threads);
    pin_fanout_list_end = pin_forward_arc_list_end;
    total_num_fanouts = fanout_total;
    pin_fanout_list.resize(fanout_total);
    pin_forward_arc_list.resize(fanout_total);
    pin_backward_arc_list.resize(fanin_total);

    pin_forward_arc_cursor = pin_forward_arc_list_end;
    pin_backward_arc_cursor = pin_backward_arc_list_end;
}

void GTDatabase::WriteNetArcList(
    int graph_threads,
    vector<int>& net_arc_start,
    vector<index_type>& pin_forward_arc_cursor,
    vector<index_type>& pin_backward_arc_cursor)
{
    const int num_nets = static_cast<int>(gpdb.getNets().size());
#pragma omp parallel for num_threads(graph_threads) schedule(static)
    for (int net_id = 0; net_id < num_nets; ++net_id) {
        const auto& pins = gpdb.getNets()[net_id].pins();
        if (pins.size() <= 1) {
            continue;
        }
        const int driver_pin_id = static_cast<int>(pins[0]);
        int arc_id = net_arc_start[net_id];
        for (index_type i = 1; i < static_cast<index_type>(pins.size()); ++i, ++arc_id) {
            const int sink_pin_id = static_cast<int>(pins[i]);
            timing_arc_from_pin_id[arc_id] = driver_pin_id;
            timing_arc_to_pin_id[arc_id] = sink_pin_id;
            if (driver_pin_id >= 0 && driver_pin_id < num_pins) {
                int pos = -1;
#pragma omp atomic capture
                {
                    pos = pin_forward_arc_cursor[driver_pin_id];
                    pin_forward_arc_cursor[driver_pin_id]++;
                }
                pin_forward_arc_list[pos] = arc_id;
                pin_fanout_list[pos] = sink_pin_id;
            }
            if (sink_pin_id >= 0 && sink_pin_id < num_pins) {
                int pos = -1;
#pragma omp atomic capture
                {
                    pos = pin_backward_arc_cursor[sink_pin_id];
                    pin_backward_arc_cursor[sink_pin_id]++;
                }
                pin_backward_arc_list[pos] = arc_id;
            }
        }
    }
    release_vector_storage(net_arc_start);
}

namespace {

void WriteCellArcListAndTest(
    GTDatabase& db,
    int graph_threads,
    int num_net_arcs,
    vector<vector<CellTimingArc>>& local_cell_timing_arcs,
    vector<int>& thread_cell_arc_start,
    vector<int>& thread_cell_test_start,
    vector<index_type>& pin_forward_arc_cursor,
    vector<index_type>& pin_backward_arc_cursor)
{
    auto& timing_arc_from_pin_id = db.timing_arc_from_pin_id;
    auto& timing_arc_to_pin_id = db.timing_arc_to_pin_id;
    auto& timing_arc_id_map = db.timing_arc_id_map;
    auto& arc_types = db.arc_types;
    auto& arc_id2test_id = db.arc_id2test_id;
    auto& test_id2_arc_id = db.test_id2_arc_id;
    auto& pin_forward_arc_list = db.pin_forward_arc_list;
    auto& pin_fanout_list = db.pin_fanout_list;
    auto& pin_backward_arc_list = db.pin_backward_arc_list;
    auto& pin_is_clk = db.pin_is_clk;
    const int num_pins = db.num_pins;
#pragma omp parallel num_threads(graph_threads)
    {
        const int tid = omp_get_thread_num();
        const auto& thread_cell_timing_arcs = local_cell_timing_arcs[tid];
        const int arc_base = num_net_arcs + thread_cell_arc_start[tid];
        const int test_base = thread_cell_test_start[tid];
        int thread_test_offset = 0;
        for (int local_arc = 0; local_arc < static_cast<int>(thread_cell_timing_arcs.size()); ++local_arc) {
            const CellTimingArc& entry = thread_cell_timing_arcs[local_arc];
            const int arc_id = arc_base + local_arc;
            timing_arc_from_pin_id[arc_id] = entry.from_pin_id;
            timing_arc_to_pin_id[arc_id] = entry.to_pin_id;
            timing_arc_id_map[arc_id * 2 + static_cast<int>(entry.el)] = entry.timing_id;
            arc_types[arc_id] = 1;
            if (entry.from_pin_id >= 0 && entry.from_pin_id < num_pins) {
                int pos = -1;
#pragma omp atomic capture
                {
                    pos = pin_forward_arc_cursor[entry.from_pin_id];
                    pin_forward_arc_cursor[entry.from_pin_id]++;
                }
                pin_forward_arc_list[pos] = arc_id;
                pin_fanout_list[pos] = entry.to_pin_id;
            }
            if (entry.to_pin_id >= 0 && entry.to_pin_id < num_pins) {
                int pos = -1;
#pragma omp atomic capture
                {
                    pos = pin_backward_arc_cursor[entry.to_pin_id];
                    pin_backward_arc_cursor[entry.to_pin_id]++;
                }
                pin_backward_arc_list[pos] = arc_id;
            }
            if (entry.is_test != 0) {
                const int test_id = test_base + thread_test_offset++;
                arc_id2test_id[arc_id] = test_id;
                test_id2_arc_id[test_id] = arc_id;
                if (entry.from_pin_id >= 0 && entry.from_pin_id < num_pins) {
#pragma omp atomic write
                    pin_is_clk[entry.from_pin_id] = 1;
                }
            }
        }
    }
    release_vector_storage(local_cell_timing_arcs);
    release_vector_storage(thread_cell_arc_start);
    release_vector_storage(thread_cell_test_start);
    release_vector_storage(pin_forward_arc_cursor);
    release_vector_storage(pin_backward_arc_cursor);
}

}  // namespace

void GTDatabase::AppendTestEndpoints()
{
    for (int test_id = 0; test_id < num_tests; ++test_id) {
        const int arc_id = test_id2_arc_id[test_id];
        if (arc_id >= 0 && arc_id < num_arcs) {
            endpoints_id.push_back(timing_arc_to_pin_id[arc_id]);
        }
    }
}

void GTDatabase::BuildPinFrontiers(int graph_threads)
{
    pin_frontiers.clear();
    pin_num_fanin.assign(num_pins, 0);
    std::vector<std::vector<index_type>> local_frontiers(graph_threads);
#pragma omp parallel num_threads(graph_threads)
    {
        const int tid = omp_get_thread_num();
        const int start = (num_pins * tid) / graph_threads;
        const int end = (num_pins * (tid + 1)) / graph_threads;
        auto& frontiers = local_frontiers[tid];
        for (int pin_id = start; pin_id < end; ++pin_id) {
            pin_num_fanin[pin_id] =
                pin_backward_arc_list_end[pin_id + 1] - pin_backward_arc_list_end[pin_id];
            if (pin_num_fanin[pin_id] == 0) frontiers.push_back(pin_id);
        }
    }
    for (auto& frontiers : local_frontiers) {
        pin_frontiers.insert(pin_frontiers.end(), frontiers.begin(), frontiers.end());
    }
}

int GTDatabase::CountRegisterClockPins(int graph_threads) const
{
    int num_clk_pins = 0;
#pragma omp parallel for num_threads(graph_threads) schedule(static) reduction(+:num_clk_pins)
    for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
        num_clk_pins += pin_is_clk[pin_id] != 0 ? 1 : 0;
    }
    return num_clk_pins;
}

void GTDatabase::CompactEndpointPins()
{
    std::unordered_map<int, int> endpoint_pin_to_compact;
    endpoint_pin_to_compact.reserve(endpoints_id.size());
    endpoint_unique_pin_ids.clear();
    auto compact_endpoint_id = [&](int pin_id) {
        auto [iter, inserted] = endpoint_pin_to_compact.emplace(
            pin_id,
            static_cast<int>(endpoint_unique_pin_ids.size()));
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
}

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

void GTDatabase::MarkExtractProfile(const char* phase)
{
    if (extract_profile) {
        extract_profile->mark(phase);
    }
}

void GTDatabase::SetupThresholdAndFlattenLib() {
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
    MarkExtractProfile("thresholds");

    //  Flatten Liberty Cell Timing
    std::map<std::array<float, 18>, int> dmp_library_id_by_thresholds;
    auto register_dmp_library = [&](const LibertyCell* min_cell,
                                    const LibertyCell* max_cell) -> int {
        if (min_cell == nullptr || max_cell == nullptr) {
            return -1;
        }
        std::array<float, 18> threshold_key = dmp_library_threshold_key(min_cell, max_cell);
        auto cached_thresholds = dmp_library_id_by_thresholds.find(threshold_key);
        if (cached_thresholds != dmp_library_id_by_thresholds.end()) {
            return cached_thresholds->second;
        }
        const int lib_id = static_cast<int>(dmp_library_input_thresholds.size() / NUM_ATTR);
        dmp_library_id_by_thresholds.emplace(threshold_key, lib_id);
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
    MarkExtractProfile("flatten_liberty");
}

vector<uint8_t> GTDatabase::SetPinMapAndTag(int graph_threads) {
    std::map<std::array<float, 18>, int> dmp_library_id_by_thresholds;
    std::unordered_map<const LibertyCell*, int> dmp_library_id_by_cell;
    dmp_library_id_by_cell.reserve(rawdb.celltypes.size() * 2u);
    auto register_dmp_library_id = [&](const LibertyCell* min_cell,
                                       const LibertyCell* max_cell) -> int {
        if (min_cell == nullptr || max_cell == nullptr) {
            return -1;
        }
        std::array<float, 18> threshold_key = dmp_library_threshold_key(min_cell, max_cell);
        auto cached_thresholds = dmp_library_id_by_thresholds.find(threshold_key);
        if (cached_thresholds != dmp_library_id_by_thresholds.end()) {
            dmp_library_id_by_cell.emplace(min_cell, cached_thresholds->second);
            dmp_library_id_by_cell.emplace(max_cell, cached_thresholds->second);
            return cached_thresholds->second;
        }
        const int lib_id = static_cast<int>(dmp_library_id_by_thresholds.size());
        dmp_library_id_by_thresholds.emplace(threshold_key, lib_id);
        dmp_library_id_by_cell.emplace(min_cell, lib_id);
        dmp_library_id_by_cell.emplace(max_cell, lib_id);
        return lib_id;
    };
    for (db::CellType* cell_type : rawdb.celltypes) {
        string cell_type_name = cell_type->name;
        array<LibertyCell*, 2> liberty_cell_view = {
            cell_libs_[MIN]->get_cell(cell_type_name),
            cell_libs_[MAX]->get_cell(cell_type_name)};
        register_dmp_library_id(liberty_cell_view[MIN], liberty_cell_view[MAX]);
    }
    dmp_timing_library_ids.resize(liberty_timing_arcs.size(), -1);
    for (size_t timing_id = 0; timing_id < liberty_timing_arcs.size(); ++timing_id) {
        TimingArc* timing_arc = liberty_timing_arcs[timing_id];
        LibertyCell* liberty_cell = timing_arc && timing_arc->liberty_port_
                                        ? timing_arc->liberty_port_->cell_
                                        : nullptr;
        auto cached_cell = dmp_library_id_by_cell.find(liberty_cell);
        dmp_timing_library_ids[timing_id] =
            cached_cell != dmp_library_id_by_cell.end() ? cached_cell->second : -1;
    }
    MarkExtractProfile("liberty_threshold_vectors");

    //  Traverse Circuit Pins
    //
    num_pins = gpdb.getPins().size();
    pin_name2pin_id.clear();
    if (build_full_pin_name_map) {
        pin_name2pin_id.reserve(pin_names.size());
    } else if (!pin_name_map_targets.empty()) {
        pin_name2pin_id.reserve(pin_name_map_targets.size());
    }
    pin_id2cell_type_id.resize(num_pins);
    pin_id2port_offset_id.resize(num_pins);
    dmp_pin_library_ids.assign(num_pins, -1);
    pin_is_clk.assign(num_pins, 0);
    pin_is_ideal_clk.assign(num_pins, 0);
    pin_case_values.assign(num_pins, -1);
    pin_capacitance.resize(2 * 3 * num_pins, 0.0f);
    const auto& gp_pins = gpdb.getPins();
    const int gp_pin_count = static_cast<int>(gp_pins.size());
    const bool build_pin_name_map = build_full_pin_name_map || !pin_name_map_targets.empty();
    const int pin_name_count = static_cast<int>(pin_names.size());
    vector<vector<std::pair<std::string, int>>> local_name_entries(graph_threads);
    vector<vector<int>> local_primary_inputs(graph_threads);
    vector<vector<int>> local_primary_outputs(graph_threads);
    const auto& pin_name_map_targets_const = pin_name_map_targets;
    const auto& dmp_library_id_by_cell_const = dmp_library_id_by_cell;
#pragma omp parallel num_threads(graph_threads)
    {
        const int tid = omp_get_thread_num();
        const int start = (gp_pin_count * tid) / graph_threads;
        const int end = (gp_pin_count * (tid + 1)) / graph_threads;
        auto& thread_name_entries = local_name_entries[tid];
        auto& thread_primary_inputs = local_primary_inputs[tid];
        auto& thread_primary_outputs = local_primary_outputs[tid];
        if (build_full_pin_name_map) {
            thread_name_entries.reserve(static_cast<size_t>(end - start));
        }
        for (int pin_index = start; pin_index < end; ++pin_index) {
            const auto& gppin = gp_pins[pin_index];
            int pin_id = static_cast<int>(gppin.getId());
            if (build_pin_name_map &&
                pin_id >= 0 &&
                pin_id < pin_name_count) {
                const std::string& pin_name = pin_names[pin_id];
                if (build_full_pin_name_map || pin_name_map_targets_const.find(pin_name) != pin_name_map_targets_const.end()) {
                    thread_name_entries.emplace_back(pin_name_colon_to_slash(pin_name), pin_id);
                }
            }
            const std::string& pin_macro_name = gppin.getMacroName();
            auto [ori_node_id, ori_node_pin_id, ori_net_id] = gppin.getOriDBInfo();
            (void) ori_net_id;
            if (ori_node_pin_id == -1) {
                auto dbiopin = rawdb.iopins[ori_node_id];
                pin_id2cell_type_id[pin_id] = -1;
                if (dbiopin->type->direction() == 'i') {
                    thread_primary_outputs.push_back(pin_id);
                } else if (dbiopin->type->direction() == 'o') {
                    thread_primary_inputs.push_back(pin_id);
                }
            } else {
                auto& dbcell = rawdb.cells[ori_node_id];
                LibertyCell* liberty_cell = dbcell->ctype()->liberty_cell;
                pin_id2cell_type_id[pin_id] = dbcell->ctype()->libcell();
                const auto& dmp_library_iter = dmp_library_id_by_cell_const.find(liberty_cell);
                dmp_pin_library_ids[pin_id] =
                    dmp_library_iter == dmp_library_id_by_cell_const.end() ? -1 : dmp_library_iter->second;
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
    }
    for (auto& entries : local_name_entries) {
        for (auto& entry : entries) {
            pin_name2pin_id.emplace(std::move(entry.first), entry.second);
        }
    }
    pin_name_map_targets.clear();
    MarkExtractProfile("pin_name_map");
    size_t primary_input_count = 0;
    size_t primary_output_count = 0;
    for (int tid = 0; tid < graph_threads; ++tid) {
        primary_input_count += local_primary_inputs[tid].size();
        primary_output_count += local_primary_outputs[tid].size();
    }
    primary_inputs.reserve(primary_input_count);
    primary_outputs.reserve(primary_output_count);
    endpoints_id.reserve(endpoints_id.size() + primary_output_count);
    for (int tid = 0; tid < graph_threads; ++tid) {
        primary_inputs.insert(primary_inputs.end(),
                              local_primary_inputs[tid].begin(),
                              local_primary_inputs[tid].end());
        primary_outputs.insert(primary_outputs.end(),
                               local_primary_outputs[tid].begin(),
                               local_primary_outputs[tid].end());
    }
    for (int pin_id : primary_outputs) {
        endpoints_id.push_back(pin_id);
        primary_output2pin_id[pin_names[pin_id]] = pin_id;
    }
    for (int pin_id : primary_inputs) {
        primary_input2pin_id[pin_names[pin_id]] = pin_id;
    }
    vector<uint8_t> primary_input_mask(num_pins, 0);
    for (int pin_id : primary_inputs) {
        if (pin_id >= 0 && pin_id < num_pins) {
            primary_input_mask[pin_id] = 1;
        }
    }
    num_POs = primary_outputs.size();
    MarkExtractProfile("set_pin_map_and_tag");
    return primary_input_mask;
}

void GTDatabase::ExtractTimingGraph() {
    extract_profile = std::make_unique<StageProfiler>("XPLACE_EXTRACT_PROFILE", extract_profile_enabled(), stdout);
    const int graph_threads = graph_thread_count(timing_raw_db.num_threads);
    logger.info("Timing graph extraction threads: %d", graph_threads);
    SetupThresholdAndFlattenLib();
    SetPinMapAndTag(graph_threads);

    vector<int> net_arc_start;
    vector<vector<CellTimingArc>> local_cell_timing_arcs;
    vector<int> thread_cell_arc_start;
    vector<int> thread_cell_test_start;
    auto [built_num_arcs, built_num_tests] = BuildNetCellArcAndTest(
        *this,
        graph_threads,
        net_arc_start,
        local_cell_timing_arcs,
        thread_cell_arc_start,
        thread_cell_test_start);
    num_arcs = built_num_arcs;
    num_tests = built_num_tests;
    const int num_net_arcs = net_arc_start.empty() ? 0 : net_arc_start.back();

    vector<index_type> pin_forward_arc_cursor;
    vector<index_type> pin_backward_arc_cursor;
    AllocatePinArcListStorage(
        graph_threads,
        pin_forward_arc_cursor,
        pin_backward_arc_cursor);

    WriteNetArcList(
        graph_threads,
        net_arc_start,
        pin_forward_arc_cursor,
        pin_backward_arc_cursor);
    MarkExtractProfile("net_arcs");

    WriteCellArcListAndTest(
        *this,
        graph_threads,
        num_net_arcs,
        local_cell_timing_arcs,
        thread_cell_arc_start,
        thread_cell_test_start,
        pin_forward_arc_cursor,
        pin_backward_arc_cursor);
    MarkExtractProfile("cell_arcs");

    AppendTestEndpoints();
    MarkExtractProfile("release_arc_build_temps");

    BuildPinFrontiers(graph_threads);
    MarkExtractProfile("pin_fanout_lists");
    MarkExtractProfile("pin_arc_lists");

    const int num_clk_pins = CountRegisterClockPins(graph_threads);
    logger.info("Identified %d register clock pins", num_clk_pins);

    CompactEndpointPins();
    MarkExtractProfile("endpoint_compaction");

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
    MarkExtractProfile("topology_tensors");

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
    MarkExtractProfile("liberty_tensors");

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
    MarkExtractProfile("state_tensors");
    extract_profile.reset();
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
