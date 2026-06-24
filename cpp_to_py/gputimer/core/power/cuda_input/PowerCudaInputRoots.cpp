#include "PowerCudaInputBuildInternal.h"

#include "common/XplaceLog.h"
#include "gputimer/core/power/common/PowerHostCommon.h"
#include "gputimer/core/power/common/PowerActivityHostUtils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

namespace gt {


namespace {

char powerLowerChar(char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

bool isPowerFalseEnvValue(const std::string& value) {
    return value.empty() || value == "0" || value == "false" || value == "no" || value == "off";
}

bool readPowerLibVoltage(const std::shared_ptr<CellLib>& lib, float& out) {
    if (!lib) return false;
    auto it = lib->default_values.find("voltage");
    if (it != lib->default_values.end() && it->second.has_value() && *(it->second) > 0.0f) {
        out = *(it->second);
        return true;
    }
    return false;
}

std::string powerPinNameOrEmpty(GTDatabase& gtdb, int n, int pin_id) {
    return (pin_id >= 0 && pin_id < n) ? gtdb.pin_names[pin_id] : "";
}

TimingArc* powerTimingArc(GTDatabase& gtdb, int arc_id) {
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
}

bool powerTimingLevelEdgeValid(GTDatabase& gtdb, int arc_id) {
    if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.arc_id2test_id.size())) return false;
    if (gtdb.arc_id2test_id[arc_id] != -1) return false;
    TimingArc* timing_arc = powerTimingArc(gtdb, arc_id);
    if (timing_arc) {
        const TimingType type = timing_arc->timing_type_;
        if (type == TimingType::clear || type == TimingType::preset) return false;
    }
    return arc_id < static_cast<int>(gtdb.timing_arc_to_pin_id.size());
}

int powerTimingEdgeToPin(GTDatabase& gtdb, int arc_id) {
    return (arc_id >= 0 && arc_id < static_cast<int>(gtdb.timing_arc_to_pin_id.size()))
        ? gtdb.timing_arc_to_pin_id[arc_id]
        : -1;
}

bool powerTimingArcDisabled(const std::vector<uint8_t>& disabled_loop_arc, int arc_id) {
    return arc_id >= 0 && arc_id < static_cast<int>(disabled_loop_arc.size()) && disabled_loop_arc[arc_id];
}

bool powerHasValidTimingIn(GTDatabase& gtdb,
                           int pin_id,
                           const std::vector<uint8_t>& disabled_loop_arc) {
    if (pin_id < 0 || pin_id + 1 >= static_cast<int>(gtdb.pin_backward_arc_list_end.size()))
        return false;
    for (int idx = gtdb.pin_backward_arc_list_end[pin_id];
         idx < gtdb.pin_backward_arc_list_end[pin_id + 1]; ++idx) {
        const int arc_id = gtdb.pin_backward_arc_list[idx];
        if (powerTimingLevelEdgeValid(gtdb, arc_id) && !powerTimingArcDisabled(disabled_loop_arc, arc_id))
            return true;
    }
    return false;
}

bool powerHasValidTimingOut(GTDatabase& gtdb,
                            int pin_id,
                            const std::vector<uint8_t>& disabled_loop_arc) {
    if (pin_id < 0 || pin_id + 1 >= static_cast<int>(gtdb.pin_forward_arc_list_end.size()))
        return false;
    for (int idx = gtdb.pin_forward_arc_list_end[pin_id];
         idx < gtdb.pin_forward_arc_list_end[pin_id + 1]; ++idx) {
        const int arc_id = gtdb.pin_forward_arc_list[idx];
        if (powerTimingLevelEdgeValid(gtdb, arc_id) && !powerTimingArcDisabled(disabled_loop_arc, arc_id))
            return true;
    }
    return false;
}

void markPowerTimingLoopsFromRoot(GTDatabase& gtdb,
                                  int n,
                                  int root_pin,
                                  std::vector<uint8_t>& visited,
                                  std::vector<uint8_t>& on_path,
                                  PowerTimingLoopInfo& loop_info) {
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
            if (!powerTimingLevelEdgeValid(gtdb, arc_id) ||
                powerTimingArcDisabled(loop_info.disabled_loop_arc, arc_id))
                continue;
            const int to_pin = powerTimingEdgeToPin(gtdb, arc_id);
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
                loop_info.disabled_loop_arc[arc_id] = 1;
                loop_info.roots.push_back(to_pin);
                loop_info.disabled_loop_arc_count++;
            }
        }
        if (!advanced) {
            on_path[frame.pin] = 0;
            stack.pop_back();
        }
    }
}

bool powerIsClockSlewPin(GTDatabase& gtdb, int n, int pin_id) {
    if (pin_id < 0 || pin_id >= n) return false;
    if (pin_id < static_cast<int>(gtdb.pin_is_ideal_clk.size()) && gtdb.pin_is_ideal_clk[pin_id])
        return true;
    if (pin_id >= static_cast<int>(gtdb.pin_is_clk.size()) || !gtdb.pin_is_clk[pin_id])
        return false;
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        if (std::isfinite(gtdb.ClockSlewForPin(pin_id, attr)))
            return true;
    }
    return false;
}

void markPowerClockSlewPin(std::vector<uint8_t>& h_power_clock_slew_pin, int n, int pin_id) {
    if (pin_id >= 0 && pin_id < n) h_power_clock_slew_pin[pin_id] = 1;
}

}  // namespace

PowerStageProfiler::PowerStageProfiler(bool enabled)
    : profiler_("power_stage_profile", enabled, stderr) {}

void PowerStageProfiler::mark(const char* label) {
    const double elapsed = profiler_.markSeconds(label);
    if (elapsed <= 0.0) return;
    addPowerStageProfileElapsed(elapsed);
}

PowerClockPinActivity::PowerClockPinActivity() = default;

PowerClockPinActivity::PowerClockPinActivity(std::vector<int> pins_,
                                             std::vector<float> densities_,
                                             std::vector<float> duties_,
                                             std::vector<uint8_t> enqueue_)
    : pins(std::move(pins_)),
      densities(std::move(densities_)),
      duties(std::move(duties_)),
      enqueue(std::move(enqueue_)) {}

PowerTimingLoopInfo::PowerTimingLoopInfo() = default;

PowerTimingLoopInfo::PowerTimingLoopInfo(size_t arc_count)
    : disabled_loop_arc(arc_count, 0) {}

PowerTimingLoopInfo::PowerTimingLoopInfo(std::vector<uint8_t> disabled_loop_arc_,
                                         std::vector<int> roots_,
                                         int disabled_loop_arc_count_)
    : disabled_loop_arc(std::move(disabled_loop_arc_)),
      roots(std::move(roots_)),
      disabled_loop_arc_count(disabled_loop_arc_count_) {}

PowerCudaRootInputs::PowerCudaRootInputs() = default;

PowerCudaRootInputs::PowerCudaRootInputs(int n, size_t arc_count)
    : is_clock_pin(n, 0),
      is_primary_input(n, 0),
      seed_reason(n),
      seq_output_arc_keep((arc_count + 31) / 32, 0) {}

PowerCudaArcSkipInputs::PowerCudaArcSkipInputs() = default;

PowerCudaArcSkipInputs::PowerCudaArcSkipInputs(std::vector<uint8_t> arc_skip_,
                                               const int* flat_net2pin_start_map_,
                                               const int* flat_net2pin_map_)
    : arc_skip(std::move(arc_skip_)),
      flat_net2pin_start_map(flat_net2pin_start_map_),
      flat_net2pin_map(flat_net2pin_map_) {}

void dumpPowerPinNamesIfRequested(GTDatabase& gtdb, int n) {
    const char* pin_name_dump = std::getenv("XPLACE_POWER_PIN_NAME_DUMP");
    if (!pin_name_dump || pin_name_dump[0] == '\0') return;
    std::ofstream out(pin_name_dump);
    if (!out) return;
    out << "pin_id,pin_name\n";
    for (int pin_id = 0; pin_id < n; ++pin_id) {
        out << pin_id << ',' << csvEscapePowerActivitySnapshot(gtdb.pin_names[pin_id]) << '\n';
    }
}

bool readPowerEnvFlag(const char* name, bool default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') return default_value;
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(), powerLowerChar);
    return !isPowerFalseEnvValue(value);
}

int64_t readPowerEnvInt64(const char* name, int64_t default_value) {
    const char* env = std::getenv(name);
    if (!env || env[0] == '\0') return default_value;
    char* end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    return end != env ? static_cast<int64_t>(parsed) : default_value;
}

float powerVoltageForReport(GTDatabase& gtdb) {
    if (const char* env_voltage = std::getenv("XPLACE_POWER_VOLTAGE")) {
        const float v = std::strtof(env_voltage, nullptr);
        if (std::isfinite(v) && v > 0.0f) return v;
    }

    float power_voltage = 1.0f;
    if (!readPowerLibVoltage(gtdb.cell_libs_[MAX], power_voltage)) {
        for (const auto& lib : gtdb.cell_libs_) {
            if (readPowerLibVoltage(lib, power_voltage)) break;
        }
    }
    return power_voltage;
}

PowerClockPinActivity buildPowerClockPinActivity(
    GTDatabase& gtdb,
    int n,
    const std::vector<int>& h_pin_to_node,
    const std::vector<int>& h_pin_to_net,
    const std::vector<uint8_t>& h_is_load_pin,
    const std::vector<uint8_t>& h_is_driver_pin,
    const std::vector<uint8_t>& h_is_clock_gate_clock_pin,
    double sdc_time_scale,
    float clock_density) {
    std::vector<int> pins = buildPowerClockPins(gtdb, n, h_pin_to_node, h_pin_to_net,
                                                h_is_load_pin, h_is_driver_pin,
                                                h_is_clock_gate_clock_pin);
    std::vector<float> densities;
    std::vector<float> duties;
    std::vector<uint8_t> enqueue;
    densities.reserve(pins.size());
    duties.reserve(pins.size());
    enqueue.reserve(pins.size());
    for (int pin_id : pins) {
        auto [density, duty] = powerClockActivityForPin(gtdb, pin_id, sdc_time_scale, clock_density);
        densities.push_back(density);
        duties.push_back(duty);
        const int node_id = pin_id >= 0 && pin_id < n ? h_pin_to_node[pin_id] : -1;
        LibertyCell* cell = powerCellForNode(gtdb, node_id);
        const bool enqueue_clock_tree = pin_id >= 0 && pin_id < n && h_is_load_pin[pin_id]
            && (!cell || cell->sequentials_.empty());
        enqueue.push_back(enqueue_clock_tree ? 1 : 0);
    }
    return PowerClockPinActivity(std::move(pins),
                                 std::move(densities),
                                 std::move(duties),
                                 std::move(enqueue));
}

void dumpPowerSeqIdMapIfRequested(GTDatabase& gtdb,
                                  const std::vector<GpuPowerSeqHost>& h_seqs,
                                  int n) {
    const char* seq_map_file = std::getenv("XPLACE_POWER_SEQ_ID_MAP_FILE");
    if (!seq_map_file || seq_map_file[0] == '\0') return;
    std::ofstream out(seq_map_file);
    if (!out) return;
    out << "seq_id,node_id,inst_name,q_pin,q_pin_name,qn_pin,qn_pin_name,is_latch\n";
    for (int seq_id = 0; seq_id < static_cast<int>(h_seqs.size()); ++seq_id) {
        const auto& seq = h_seqs[seq_id];
        std::string inst_name;
        if (seq.node_id >= 0 && seq.node_id < static_cast<int>(gtdb.gpdb.getNodes().size()))
            inst_name = gtdb.gpdb.getNodes()[seq.node_id].getName();
        out << seq_id << ',' << seq.node_id << ','
            << csvEscapePowerActivitySnapshot(inst_name) << ','
            << seq.q_pin << ','
            << csvEscapePowerActivitySnapshot(powerPinNameOrEmpty(gtdb, n, seq.q_pin)) << ','
            << seq.qn_pin << ','
            << csvEscapePowerActivitySnapshot(powerPinNameOrEmpty(gtdb, n, seq.qn_pin)) << ','
            << static_cast<int>(seq.is_latch) << '\n';
    }
}

void printPowerSeqDupStatsIfRequested(GTDatabase& gtdb,
                                      const std::vector<GpuPowerSeqHost>& h_seqs,
                                      int n) {
    if (!std::getenv("XPLACE_POWER_PRINT_SEQ_DUP_STATS")) return;
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
    XPLACE_DEBUGF("XPLACE_POWER_PRINT_SEQ_DUP_STATS",
                  "seq_records=%zu duplicate_output_pins=%d duplicate_output_writes=%d max_writes_per_pin=%d",
                  h_seqs.size(), duplicate_pins, duplicate_writes, max_writes);
    int printed = 0;
    for (int pin_id = 0; pin_id < n && printed < 20; ++pin_id) {
        if (seq_output_write_count[pin_id] <= 1) continue;
        XPLACE_DEBUGF("XPLACE_POWER_PRINT_SEQ_DUP_STATS",
                      "seq_dup_pin pin_id=%d writes=%d pin=%s",
                      pin_id, seq_output_write_count[pin_id], gtdb.pin_names[pin_id].c_str());
        printed++;
    }
}

PowerTimingLoopInfo buildPowerTimingLoopInfo(GTDatabase& gtdb, int n, bool enabled) {
    PowerTimingLoopInfo loop_info(gtdb.arc_id2test_id.size());
    if (!enabled) return loop_info;
    std::vector<uint8_t> visited(n, 0);
    std::vector<uint8_t> on_path(n, 0);
    for (int pin_id = 0; pin_id < n; ++pin_id) {
        if (!powerHasValidTimingIn(gtdb, pin_id, loop_info.disabled_loop_arc) &&
            powerHasValidTimingOut(gtdb, pin_id, loop_info.disabled_loop_arc)) {
            loop_info.roots.push_back(pin_id);
            markPowerTimingLoopsFromRoot(gtdb, n, pin_id, visited, on_path, loop_info);
        }
    }
    for (int pin_id = 0; pin_id < n; ++pin_id) {
        if (!visited[pin_id] && powerHasValidTimingOut(gtdb, pin_id, loop_info.disabled_loop_arc))
            markPowerTimingLoopsFromRoot(gtdb, n, pin_id, visited, on_path, loop_info);
    }
    std::sort(loop_info.roots.begin(), loop_info.roots.end());
    loop_info.roots.erase(std::unique(loop_info.roots.begin(), loop_info.roots.end()),
                          loop_info.roots.end());
    return loop_info;
}

bool shouldSkipSeqOutputArcForPower(GTDatabase& gtdb,
                                    int n,
                                    const std::vector<uint8_t>& h_is_seq_output_pin,
                                    int arc_id,
                                    int from_pin,
                                    int to_pin) {
    if (to_pin < 0 || to_pin >= n || !h_is_seq_output_pin[to_pin]) return false;
    TimingArc* timing_arc = powerTimingArc(gtdb, arc_id);
    if (timing_arc) {
        const TimingType type = timing_arc->timing_type_;
        if (type == TimingType::rising_edge || type == TimingType::falling_edge) return true;
        if (type == TimingType::clear || type == TimingType::preset) return false;
    }
    if (from_pin >= 0 && from_pin < static_cast<int>(gtdb.pin_is_clk.size()) && gtdb.pin_is_clk[from_pin])
        return true;
    return false;
}

void PowerCudaRootInputs::addSeedPin(int pin_id, const char* reason) {
    if (pin_id < 0 || pin_id >= static_cast<int>(seed_reason.size())) return;
    primary_inputs.push_back(pin_id);
    if (!reason || reason[0] == '\0') return;
    std::string& current = seed_reason[pin_id];
    const std::string value(reason);
    if (current.empty()) {
        current = value;
    } else if (current.find(value) == std::string::npos) {
        current += ";";
        current += value;
    }
}

void PowerCudaRootInputs::markSeqOutputArcKeep(int arc_id) {
    if (arc_id < 0) return;
    const size_t word = static_cast<size_t>(arc_id) >> 5;
    if (word >= seq_output_arc_keep.size()) return;
    seq_output_arc_keep[word] |= (1u << (static_cast<unsigned>(arc_id) & 31u));
}

bool PowerCudaRootInputs::seqOutputArcKept(int arc_id) const {
    if (arc_id < 0) return false;
    const size_t word = static_cast<size_t>(arc_id) >> 5;
    if (word >= seq_output_arc_keep.size()) return false;
    return (seq_output_arc_keep[word] & (1u << (static_cast<unsigned>(arc_id) & 31u))) != 0;
}

PowerCudaRootInputs buildPowerCudaRootInputs(GTDatabase& gtdb,
                                             int n,
                                             const std::vector<int>& clock_pins,
                                             const std::vector<int>& pin_to_node,
                                             const std::vector<int>& pin_to_net,
                                             const std::vector<int>& net_driver_pin,
                                             const std::vector<uint8_t>& is_load_pin,
                                             const std::vector<uint8_t>& is_driver_pin,
                                             const PowerCudaSeqInputs& seq_inputs,
                                             const PowerCudaExprInputs& expr_inputs) {
    PowerCudaRootInputs roots(n, gtdb.arc_types.size());
    for (int pin_id : clock_pins) {
        if (pin_id >= 0 && pin_id < n) roots.is_clock_pin[pin_id] = 1;
    }

    roots.seed_seq_feedback_outputs =
        readPowerBoolEnv("XPLACE_POWER_SEED_SEQ_FEEDBACK_OUTPUTS", false);
    const bool seed_timing_zero_indeg_roots =
        readPowerBoolEnv("XPLACE_POWER_SEED_TIMING_ZERO_INDEG", true);
    const bool seed_floating_load_roots =
        readPowerBoolEnv("XPLACE_POWER_SEED_FLOATING_LOADS", true);
    const bool seed_seq_feedback_d_only =
        std::getenv("XPLACE_POWER_SEED_SEQ_FEEDBACK_D_ONLY") != nullptr;
    roots.init_seq_feedback_state =
        std::getenv("XPLACE_POWER_INIT_SEQ_FEEDBACK_STATE") != nullptr;
    const bool skip_all_seq_output_arcs =
        std::getenv("XPLACE_POWER_SKIP_ALL_SEQ_OUTPUT_ARCS") != nullptr;
    roots.seed_timing_loop_roots =
        std::getenv("XPLACE_POWER_SEED_TIMING_LOOP_ROOTS") != nullptr;
    roots.skip_disabled_loop_arcs =
        readPowerBoolEnv("XPLACE_POWER_SKIP_DISABLED_LOOP_ARCS", false);
    roots.seed_default_inputs = readPowerEnvFlag("XPLACE_POWER_SEED_INPUTS", true);

    PowerTimingLoopInfo timing_loop_info =
        buildPowerTimingLoopInfo(gtdb, n, roots.seed_timing_loop_roots || roots.skip_disabled_loop_arcs);
    roots.disabled_loop_arc = std::move(timing_loop_info.disabled_loop_arc);
    std::vector<int> timing_loop_roots = std::move(timing_loop_info.roots);
    roots.disabled_loop_arc_count = timing_loop_info.disabled_loop_arc_count;

    for (int from_pin = 0; from_pin < n; ++from_pin) {
        if (from_pin + 1 >= static_cast<int>(gtdb.pin_forward_arc_list_end.size())) break;
        const int start = gtdb.pin_forward_arc_list_end[from_pin];
        const int end = gtdb.pin_forward_arc_list_end[from_pin + 1];
        for (int idx = start; idx < end; ++idx) {
            const int arc_id = gtdb.pin_forward_arc_list[idx];
            if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.arc_types.size())) continue;
            if (gtdb.arc_types[arc_id] != 1) continue;
            if (arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) continue;
            const int to_pin = gtdb.timing_arc_to_pin_id[arc_id];
            if (to_pin < 0 || to_pin >= n || !seq_inputs.is_seq_output_pin[to_pin]) continue;
            if (!skip_all_seq_output_arcs &&
                !shouldSkipSeqOutputArcForPower(gtdb, n, seq_inputs.is_seq_output_pin,
                                                arc_id, from_pin, to_pin))
                roots.markSeqOutputArcKeep(arc_id);
        }
    }

    roots.primary_inputs.reserve(gtdb.primary_inputs.size());
    for (auto pin : gtdb.primary_inputs) {
        const int pin_id = static_cast<int>(pin);
        if (pin_id >= 0 && pin_id < n) roots.is_primary_input[pin_id] = 1;
        if (roots.seed_default_inputs && pin_id >= 0 && pin_id < n && is_driver_pin[pin_id]
            && !roots.is_clock_pin[pin_id]) {
            roots.addSeedPin(pin_id, "primary_input");
            roots.primary_count++;
        }
    }

    if (roots.seed_default_inputs) {
        if (seed_timing_zero_indeg_roots) {
            for (int pin_id : gtdb.pin_frontiers) {
                if (pin_id < 0 || pin_id >= n) continue;
                if (roots.is_primary_input[pin_id] || roots.is_clock_pin[pin_id]) continue;
                roots.addSeedPin(pin_id, "timing_zero_indeg");
                roots.zero_indeg_count++;
            }
        }
        if (seed_floating_load_roots) {
            for (int pin_id = 0; pin_id < n; pin_id++) {
                if (!is_load_pin[pin_id] || roots.is_primary_input[pin_id] || roots.is_clock_pin[pin_id]) continue;
                const int net_id = pin_to_net[pin_id];
                const int driver =
                    (net_id >= 0 && net_id < static_cast<int>(net_driver_pin.size())) ? net_driver_pin[net_id] : -1;
                if (driver >= 0) continue;
                roots.addSeedPin(pin_id, "floating_load_input");
                roots.floating_load_count++;
            }
        }
        if (roots.seed_timing_loop_roots) {
            for (int pin_id : timing_loop_roots) {
                if (pin_id < 0 || pin_id >= n) continue;
                if (roots.is_primary_input[pin_id] || roots.is_clock_pin[pin_id]) continue;
                if (!is_load_pin[pin_id] && !is_driver_pin[pin_id]) continue;
                roots.addSeedPin(pin_id, "timing_loop_root");
                roots.timing_loop_count++;
            }
        }
    }

    if (roots.seed_default_inputs && (roots.seed_seq_feedback_outputs || roots.init_seq_feedback_state)) {
        std::vector<uint8_t> seed_seen(n, 0);
        std::vector<uint8_t> state_pin_seen(n, 0);
        std::vector<uint8_t> state_seq_seen(seq_inputs.seqs.size(), 0);
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
            if (expr_id < 0 || expr_id >= static_cast<int>(expr_inputs.start.size()) ||
                driver_pin < 0 || driver_pin >= n)
                return false;
            const int driver_net = pin_to_net[driver_pin];
            if (driver_net < 0 || driver_net >= static_cast<int>(net_driver_pin.size())) return false;
            if (net_driver_pin[driver_net] != driver_pin) return false;
            bool matched = false;
            const int start = expr_inputs.start[expr_id];
            const int end = start + expr_inputs.count[expr_id];
            for (int op_i = start; op_i < end; ++op_i) {
                if (expr_inputs.ops[op_i].op != 0) continue;
                int data_pin = expr_inputs.ops[op_i].arg;
                if (data_pin < -1) data_pin = pin_for_node_port(seq_node_id, -2 - data_pin);
                if (data_pin < 0 || data_pin >= n || pin_to_net[data_pin] != driver_net) continue;
                if (seed_seq_feedback_d_only) {
                    const int node_id = pin_to_node[data_pin];
                    LibertyCell* cell = powerCellForNode(gtdb, node_id);
                    const int port_offset = gtdb.pin_id2port_offset_id[data_pin];
                    if (!cell || port_offset < 0 || port_offset >= static_cast<int>(cell->ports_.size()))
                        continue;
                    LibertyPort* port = cell->ports_[port_offset];
                    if (!port || port->name != "D") continue;
                }
                matched = true;
                if (data_pins) data_pins->push_back(data_pin);
            }
            return matched;
        };
        for (int seq_id = 0; seq_id < static_cast<int>(seq_inputs.seqs.size()); ++seq_id) {
            const auto& seq = seq_inputs.seqs[seq_id];
            std::vector<int> data_pins;
            const bool q_feedback = collect_feedback_data_pins(seq.data_expr_id, seq.node_id,
                                                               seq.q_pin, &data_pins);
            if (q_feedback && roots.seed_seq_feedback_outputs && !seed_seen[seq.q_pin]) {
                roots.addSeedPin(seq.q_pin, "seq_feedback_q");
                seed_seen[seq.q_pin] = 1;
                roots.seq_feedback_count++;
            }
            const bool qn_feedback = collect_feedback_data_pins(seq.data_expr_id, seq.node_id,
                                                                seq.qn_pin, &data_pins);
            if (qn_feedback && roots.seed_seq_feedback_outputs && !seed_seen[seq.qn_pin]) {
                roots.addSeedPin(seq.qn_pin, "seq_feedback_qn");
                seed_seen[seq.qn_pin] = 1;
                roots.seq_feedback_count++;
            }
            if (roots.init_seq_feedback_state && (q_feedback || qn_feedback)) {
                if (!state_seq_seen[seq_id]) {
                    roots.feedback_seed_seqs.push_back(seq_id);
                    state_seq_seen[seq_id] = 1;
                    roots.state_seq_feedback_count++;
                }
                for (int data_pin : data_pins) {
                    if (data_pin >= 0 && data_pin < n && !state_pin_seen[data_pin]) {
                        roots.feedback_seed_pins.push_back(data_pin);
                        state_pin_seen[data_pin] = 1;
                    }
                }
            }
        }
    }

    // Constant-generator outputs are roots in OpenSTA's power graph.
    for (int pin_id = 0; pin_id < n; pin_id++) {
        if (!is_driver_pin[pin_id] || roots.is_primary_input[pin_id] || roots.is_clock_pin[pin_id]) continue;
        const int node_id = pin_to_node[pin_id];
        if (node_id < 0 || node_id >= static_cast<int>(gtdb.gpdb.getNodes().size())) continue;
        bool has_input_pin = false;
        for (int node_pin : gtdb.gpdb.getNodes()[node_id].pins()) {
            if (node_pin >= 0 && node_pin < n && is_load_pin[node_pin]) {
                has_input_pin = true;
                break;
            }
        }
        if (roots.seed_default_inputs && !has_input_pin) {
            roots.addSeedPin(pin_id, "const_output");
            roots.const_output_count++;
        }
    }

    std::vector<uint8_t> root_seen(n, 0);
    std::vector<int> ordered_roots;
    ordered_roots.reserve(roots.primary_inputs.size());
    for (int pin_id : roots.primary_inputs) {
        if (pin_id < 0 || pin_id >= n || root_seen[pin_id]) continue;
        root_seen[pin_id] = 1;
        ordered_roots.push_back(pin_id);
    }
    roots.primary_inputs.swap(ordered_roots);
    return roots;
}

PowerCudaArcSkipInputs buildPowerCudaArcSkipInputs(GTDatabase& gtdb,
                                                   const PowerCudaRootInputs& roots,
                                                   const int* default_flat_net2pin_start_map,
                                                   const int* default_flat_net2pin_map) {
    std::vector<uint8_t> arc_skip(gtdb.arc_id2test_id.size(), 0);
    for (size_t arc_id = 0; arc_id < gtdb.arc_id2test_id.size(); ++arc_id) {
        arc_skip[arc_id] = static_cast<uint8_t>(gtdb.arc_id2test_id[arc_id] != -1);
    }
    if (roots.seed_timing_loop_roots || roots.skip_disabled_loop_arcs) {
        const int num_mark_arcs = std::min(static_cast<int>(arc_skip.size()),
                                           static_cast<int>(roots.disabled_loop_arc.size()));
        for (int arc_id = 0; arc_id < num_mark_arcs; ++arc_id) {
            if (roots.disabled_loop_arc[arc_id]) arc_skip[arc_id] = 1;
        }
    }

    int disabled_constraint_arc_count = 0;
    int disabled_constraint_net_arc_count = 0;
    // Debug-only experiment: set_false_path is a timing exception in
    // OpenSTA/OpenROAD, not a default power activity cut. Leave this off
    // for normal acceptance unless explicitly probing false-path activity.
    const bool apply_power_false_paths =
        readPowerBoolEnv("XPLACE_POWER_APPLY_FALSE_PATHS", false);
    if (apply_power_false_paths && !gtdb.power_disabled_constraint_arc.empty()) {
        const int num_mark_arcs = std::min(static_cast<int>(arc_skip.size()),
                                           static_cast<int>(gtdb.power_disabled_constraint_arc.size()));
        for (int arc_id = 0; arc_id < num_mark_arcs; ++arc_id) {
            if (!gtdb.power_disabled_constraint_arc[arc_id]) continue;
            arc_skip[arc_id] = 1;
            ++disabled_constraint_arc_count;
            if (arc_id < static_cast<int>(gtdb.arc_types.size()) && gtdb.arc_types[arc_id] == 0)
                ++disabled_constraint_net_arc_count;
        }
        if (disabled_constraint_arc_count > 0) {
            XPLACE_DEBUGF("XPLACE_POWER_APPLY_FALSE_PATHS",
                          "power_false_path disabled_arcs=%d disabled_net_arcs=%d",
                          disabled_constraint_arc_count,
                          disabled_constraint_net_arc_count);
        }
    } else if (!apply_power_false_paths && !gtdb.power_disabled_constraint_arc.empty()) {
        int mapped_false_path_arcs = 0;
        for (uint8_t mark : gtdb.power_disabled_constraint_arc) {
            if (mark) ++mapped_false_path_arcs;
        }
        if (mapped_false_path_arcs > 0) {
            XPLACE_DEBUGF("XPLACE_POWER_DEBUG",
                          "power_false_path mapped_arcs=%d apply=0",
                          mapped_false_path_arcs);
        }
    }

    const int* activity_flat_net2pin_start_map = default_flat_net2pin_start_map;
    const int* activity_flat_net2pin_map = default_flat_net2pin_map;
    if (disabled_constraint_net_arc_count > 0) {
        // Direct net fanout bypasses arc_id2test_id. When SDC exceptions disable
        // net arcs, rely on the timing graph net arcs so the mask is honored.
        activity_flat_net2pin_start_map = nullptr;
        activity_flat_net2pin_map = nullptr;
    }
    return PowerCudaArcSkipInputs(std::move(arc_skip),
                                  activity_flat_net2pin_start_map,
                                  activity_flat_net2pin_map);
}

void finalizePowerCudaRootInputs(GTDatabase& gtdb,
                                 int n,
                                 PowerCudaRootInputs& roots,
                                 const PowerCudaSeqInputs& seq_inputs,
                                 const std::vector<uint8_t>& is_load_pin,
                                 const std::vector<uint8_t>& is_driver_pin,
                                 const std::vector<int>& pin_to_node,
                                 const std::vector<int>& pin_to_net,
                                 const std::vector<int>& power_level_root_pins_cpu,
                                 const std::vector<int>& power_pin_level_cpu,
                                 const std::vector<uint8_t>& arc_skip) {
    if (roots.seed_default_inputs && std::getenv("XPLACE_POWER_SEED_POWER_LEVEL_ROOTS")) {
        for (int pin_id : power_level_root_pins_cpu) {
            if (pin_id < 0 || pin_id >= n) continue;
            if (roots.is_primary_input[pin_id] || roots.is_clock_pin[pin_id]) continue;
            if (!is_load_pin[pin_id] && !is_driver_pin[pin_id]) continue;
            roots.addSeedPin(pin_id, "power_zero_fanin_seed");
            roots.power_level_count++;
        }
    }

    std::vector<uint8_t> seed_seen(n, 0);
    std::vector<int> seed_inputs;
    seed_inputs.reserve(roots.primary_inputs.size());
    for (int pin_id : roots.primary_inputs) {
        if (pin_id < 0 || pin_id >= n || seed_seen[pin_id]) continue;
        seed_seen[pin_id] = 1;
        seed_inputs.push_back(pin_id);
    }
    roots.primary_inputs.swap(seed_inputs);

    std::vector<int> power_fanin(n, 0);
    if (gtdb.pin_forward_arc_list_end.size() == static_cast<size_t>(n + 1)) {
        for (int from_pin = 0; from_pin < n; ++from_pin) {
            const int start = gtdb.pin_forward_arc_list_end[from_pin];
            const int end = gtdb.pin_forward_arc_list_end[from_pin + 1];
            for (int idx = start; idx < end; ++idx) {
                if (idx < 0 || idx >= static_cast<int>(gtdb.pin_forward_arc_list.size())) continue;
                const int arc_id = gtdb.pin_forward_arc_list[idx];
                if (arc_id < 0 || arc_id >= static_cast<int>(gtdb.timing_arc_to_pin_id.size())) continue;
                if (arc_id < static_cast<int>(arc_skip.size()) && arc_skip[arc_id]) continue;
                const int to_pin = gtdb.timing_arc_to_pin_id[arc_id];
                if (to_pin < 0 || to_pin >= n) continue;
                if (arc_id < static_cast<int>(gtdb.arc_types.size()) &&
                    gtdb.arc_types[arc_id] == 1 && seq_inputs.is_seq_output_pin[to_pin] &&
                    !roots.seqOutputArcKept(arc_id))
                    continue;
                power_fanin[to_pin]++;
            }
        }
    }
    dumpPowerCudaInputRoots(gtdb, n, roots.primary_inputs, power_level_root_pins_cpu,
                            roots.seed_reason, seed_seen, roots.is_primary_input,
                            roots.is_clock_pin, is_driver_pin, is_load_pin,
                            power_fanin, pin_to_node, pin_to_net,
                            power_pin_level_cpu);
    if (std::getenv("XPLACE_POWER_PRINT_ROOT_STATS")) {
        XPLACE_DEBUGF("XPLACE_POWER_PRINT_ROOT_STATS",
                      "power_activity_roots seeds=%zu primary=%d timing_roots=%d floating_load_roots=%d timing_loop_roots=%d disabled_loop_arcs=%d power_roots=%d const_outputs=%d",
                      roots.primary_inputs.size(), roots.primary_count, roots.zero_indeg_count,
                      roots.floating_load_count, roots.timing_loop_count, roots.disabled_loop_arc_count,
                      roots.power_level_count, roots.const_output_count);
        if (roots.seed_seq_feedback_outputs) {
            XPLACE_DEBUGF("XPLACE_POWER_PRINT_ROOT_STATS",
                          "power_activity_roots seq_feedback=%d",
                          roots.seq_feedback_count);
        }
        if (roots.init_seq_feedback_state) {
            XPLACE_DEBUGF("XPLACE_POWER_PRINT_ROOT_STATS",
                          "power_activity_roots seq_feedback_state=%d pins=%zu",
                          roots.state_seq_feedback_count, roots.feedback_seed_pins.size());
        }
    }
}

PowerClockSlewSparse buildPowerClockSlews(GTDatabase& gtdb,
                                          int n,
                                          const std::vector<int>& h_clock_pins,
                                          const std::vector<uint8_t>& h_is_seq_clock_input_pin,
                                          const std::vector<int>& h_pin_to_net,
                                          bool need_internal_power) {
    PowerClockSlewSparse result;
    if (!need_internal_power) return result;
    bool has_clock_slew_pins = false;
    for (int pin_id = 0; pin_id < n; ++pin_id) {
        if (powerIsClockSlewPin(gtdb, n, pin_id)) {
            has_clock_slew_pins = true;
            break;
        }
    }
    if (!has_clock_slew_pins) return result;

    result.fallback.fill(nanf(""));
    for (int clock_id = 0; clock_id < static_cast<int>(gtdb.clock_periods.size()); ++clock_id) {
        bool has_finite = false;
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            const size_t idx = static_cast<size_t>(clock_id) * NUM_ATTR + attr;
            if (idx < gtdb.clock_slews.size() && std::isfinite(gtdb.clock_slews[idx])) {
                result.fallback[attr] = gtdb.clock_slews[idx];
                has_finite = true;
            }
        }
        if (has_finite) break;
    }
    for (float& slew : result.fallback) {
        if (!std::isfinite(slew)) slew = 0.0f;
    }

    std::vector<uint8_t> h_power_clock_slew_pin(n, 0);
    for (int pin_id : h_clock_pins) {
        if (powerIsClockSlewPin(gtdb, n, pin_id)) markPowerClockSlewPin(h_power_clock_slew_pin, n, pin_id);
    }
    for (int pin_id = 0; pin_id < n; ++pin_id) {
        if (h_is_seq_clock_input_pin[pin_id] && powerIsClockSlewPin(gtdb, n, pin_id))
            markPowerClockSlewPin(h_power_clock_slew_pin, n, pin_id);
    }

    const int num_nets = static_cast<int>(gtdb.gpdb.getNets().size());
    std::vector<uint8_t> power_clock_slew_net(num_nets, 0);
    for (int pin_id = 0; pin_id < n; ++pin_id) {
        if (!powerIsClockSlewPin(gtdb, n, pin_id)) continue;
        const int net_id = h_pin_to_net[pin_id];
        if (net_id >= 0 && net_id < num_nets) power_clock_slew_net[net_id] = 1;
    }
    for (int net_id = 0; net_id < num_nets; ++net_id) {
        if (!power_clock_slew_net[net_id]) continue;
        for (int pin_id : gtdb.gpdb.getNets()[net_id].pins())
            markPowerClockSlewPin(h_power_clock_slew_pin, n, pin_id);
    }

    for (int pin_id = 0; pin_id < n; ++pin_id) {
        if (!h_power_clock_slew_pin[pin_id]) continue;
        result.pins.push_back(pin_id);
    }
    return result;
}

torch::Tensor powerCudaIntTensor(const std::vector<int>& v) {
    auto iopt_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    if (v.empty()) return torch::zeros({1}, iopt_cpu).to(torch::kCUDA);
    return torch::from_blob(const_cast<int*>(v.data()), {(long)v.size()}, iopt_cpu).to(torch::kCUDA);
}

torch::Tensor powerCudaIndexTensor(const std::vector<index_type>& v) {
    auto iopt_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    if (v.empty()) return torch::zeros({1}, iopt_cpu).to(torch::kCUDA);
    return torch::from_blob(const_cast<index_type*>(v.data()), {(long)v.size()}, iopt_cpu).to(torch::kCUDA);
}

torch::Tensor powerCudaU8Tensor(const std::vector<uint8_t>& v) {
    auto bopt_cpu = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    if (v.empty()) return torch::zeros({1}, bopt_cpu).to(torch::kCUDA);
    return torch::from_blob(const_cast<uint8_t*>(v.data()), {(long)v.size()}, bopt_cpu).to(torch::kCUDA);
}

torch::Tensor powerCudaFloatTensor(const std::vector<float>& v) {
    auto fopt_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (v.empty()) return torch::full({1}, nanf(""), fopt_cpu).to(torch::kCUDA);
    return torch::from_blob(const_cast<float*>(v.data()), {(long)v.size()}, fopt_cpu).to(torch::kCUDA);
}

PowerCudaUploader::PowerCudaUploader(bool debug, bool sync_debug)
    : debug_(debug), sync_debug_(sync_debug) {}

torch::Tensor PowerCudaUploader::uploadInt(const char* label, const std::vector<int>& v) const {
    mark("begin", label, v.size(), sizeof(int));
    auto out = powerCudaIntTensor(v);
    mark("end", label, v.size(), sizeof(int));
    return out;
}

torch::Tensor PowerCudaUploader::uploadIndex(const char* label, const std::vector<index_type>& v) const {
    mark("begin", label, v.size(), sizeof(index_type));
    auto out = powerCudaIndexTensor(v);
    mark("end", label, v.size(), sizeof(index_type));
    return out;
}

torch::Tensor PowerCudaUploader::uploadU8(const char* label, const std::vector<uint8_t>& v) const {
    mark("begin", label, v.size(), sizeof(uint8_t));
    auto out = powerCudaU8Tensor(v);
    mark("end", label, v.size(), sizeof(uint8_t));
    return out;
}

torch::Tensor PowerCudaUploader::uploadFloat(const char* label, const std::vector<float>& v) const {
    mark("begin", label, v.size(), sizeof(float));
    auto out = powerCudaFloatTensor(v);
    mark("end", label, v.size(), sizeof(float));
    return out;
}

void PowerCudaUploader::mark(const char* phase, const char* label, size_t count, size_t elem_size) const {
    if (debug_) {
        XPLACE_ERRORF("power_upload",
                      "phase=%s label=%s count=%zu bytes=%zu",
                      phase, label ? label : "", count, count * elem_size);
    }
    if (sync_debug_) {
        const std::string sync_label =
            std::string("upload ") + (phase ? phase : "") + " " + (label ? label : "");
        check_power_cuda_error(sync_label.c_str());
    }
}

torch::Tensor outputPowerTensorForRequest(const torch::Tensor& tensor, bool output_power_tensors_cuda) {
    return output_power_tensors_cuda ? tensor : tensor.to(torch::kCPU);
}

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
                             const std::vector<int>& power_pin_level_cpu) {
    if (const char* root_dump_file = std::getenv("XPLACE_POWER_DUMP_ROOTS_FILE")) {
        if (root_dump_file[0] != '\0') {
            std::vector<uint8_t> h_candidate_seen(n, 0);
            for (int pin_id : candidate_roots) {
                if (pin_id >= 0 && pin_id < n) h_candidate_seen[pin_id] = 1;
            }
            std::vector<int> root_probe_pins =
                resolvePowerTracePins(readPowerRootProbePinQueries(), gtdb.pin_names);
            std::vector<int> dump_pins;
            dump_pins.reserve(primary_inputs.size() + candidate_roots.size() + root_probe_pins.size());
            dump_pins.insert(dump_pins.end(), primary_inputs.begin(), primary_inputs.end());
            dump_pins.insert(dump_pins.end(), candidate_roots.begin(), candidate_roots.end());
            dump_pins.insert(dump_pins.end(), root_probe_pins.begin(), root_probe_pins.end());
            std::sort(dump_pins.begin(), dump_pins.end());
            dump_pins.erase(std::unique(dump_pins.begin(), dump_pins.end()), dump_pins.end());
    
            std::ofstream root_dump(root_dump_file);
            if (root_dump) {
                root_dump
                    << "pin_id\tpin_name\tin_actual_seed\tin_candidate\treason"
                    << "\tis_primary\tis_clock\tis_driver\tis_load\tpower_fanin"
                    << "\ttiming_fanin\tpower_level\tnode_id\tinst_name\tcell_type"
                    << "\tnet_id\tnet_name\n";
                for (int pin_id : dump_pins) {
                    if (pin_id < 0 || pin_id >= n) continue;
                    std::string reason = seed_reason[pin_id];
                    if (reason.empty() && seed_seen[pin_id]) reason = "seed";
                    if (h_candidate_seen[pin_id]) {
                        if (reason.empty()) reason = "power_zero_fanin_candidate";
                        else if (reason.find("power_zero_fanin_candidate") == std::string::npos
                                 && reason.find("power_zero_fanin_seed") == std::string::npos)
                            reason += ";power_zero_fanin_candidate";
                    }
                    if (reason.empty()) reason = "probe_only";
                    const int node_id =
                        (pin_id < static_cast<int>(pin_to_node.size())) ? pin_to_node[pin_id] : -1;
                    const int net_id =
                        (pin_id < static_cast<int>(pin_to_net.size())) ? pin_to_net[pin_id] : -1;
                    std::string inst_name;
                    std::string cell_type;
                    if (node_id >= 0 && node_id < static_cast<int>(gtdb.gpdb.getNodes().size())) {
                        inst_name = gtdb.gpdb.getNodes()[node_id].getName();
                        cell_type = gtdb.gpdb.getNodes()[node_id].getCellTypeName();
                    }
                    std::string net_name;
                    if (net_id >= 0 && net_id < static_cast<int>(gtdb.net_names.size())) {
                        net_name = gtdb.net_names[net_id];
                    } else if (net_id >= 0 && net_id < static_cast<int>(gtdb.gpdb.getNets().size())) {
                        net_name = gtdb.gpdb.getNets()[net_id].getName();
                    }
                    const int timing_fanin =
                        (pin_id < static_cast<int>(gtdb.pin_num_fanin.size())) ? gtdb.pin_num_fanin[pin_id] : -1;
                    const int power_level =
                        (pin_id < static_cast<int>(power_pin_level_cpu.size())) ? power_pin_level_cpu[pin_id] : -1;
                    root_dump << pin_id << '\t' << gtdb.pin_names[pin_id] << '\t'
                              << (seed_seen[pin_id] ? 1 : 0) << '\t'
                              << (h_candidate_seen[pin_id] ? 1 : 0) << '\t'
                              << reason << '\t'
                              << (is_primary_input[pin_id] ? 1 : 0) << '\t'
                              << (is_clock_pin[pin_id] ? 1 : 0) << '\t'
                              << (is_driver_pin[pin_id] ? 1 : 0) << '\t'
                              << (is_load_pin[pin_id] ? 1 : 0) << '\t'
                              << power_fanin[pin_id] << '\t'
                              << timing_fanin << '\t' << power_level << '\t'
                              << node_id << '\t' << inst_name << '\t' << cell_type << '\t'
                              << net_id << '\t' << net_name << '\n';
                }
            }
        }
    }
}

}  // namespace gt
