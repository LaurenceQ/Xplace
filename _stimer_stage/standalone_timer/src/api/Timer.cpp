#include "stimer/Timer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "stimer/DefReader.h"
#include "stimer/LibertyReader.h"
#include "stimer/SpefReader.h"
#include "stimer/TimingCudaRuntime.h"
#include "stimer/VerilogReader.h"

namespace stimer {

namespace {

int has_probed_file(const InputProbeResult& probe, InputKind kind) {
  for (const auto& file : probe.files) {
    if (file.kind == kind && file.error.empty()) {
      return 1;
    }
  }
  return 0;
}

const LibertyCell* find_liberty_cell(const LibertyDB& liberty,
                                     const std::string& cell_name,
                                     int* library_index,
                                     int* cell_index) {
  for (std::size_t lib_id = 0; lib_id < liberty.libraries.size(); ++lib_id) {
    const auto& library = liberty.libraries[lib_id];
    for (std::size_t cell_id = 0; cell_id < library.cells.size(); ++cell_id) {
      if (library.cells[cell_id].name == cell_name) {
        *library_index = static_cast<int>(lib_id);
        *cell_index = static_cast<int>(cell_id);
        return &library.cells[cell_id];
      }
    }
  }
  *library_index = -1;
  *cell_index = -1;
  return nullptr;
}

float timing_result_max_abs_error(const TimingEvalResult& lhs,
                                  const TimingEvalResult& rhs) {
  float error = 0.0f;
  error = std::max(error, std::fabs(lhs.max_delay - rhs.max_delay));
  error = std::max(error,
                   std::fabs(lhs.max_output_slew - rhs.max_output_slew));
  error = std::max(error, std::fabs(lhs.max_arrival - rhs.max_arrival));
  return error;
}

template <typename Fn>
void log_api_stage(const char* name, Fn&& fn) {
  const auto start = std::chrono::steady_clock::now();
  std::cerr << "[stage] " << name << " start\n" << std::flush;
  fn();
  const auto end = std::chrono::steady_clock::now();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  std::cerr << "[stage] " << name << " done_ms=" << elapsed_ms << '\n'
            << std::flush;
}

}  // namespace

Timer::Timer(TimerConfig config) : config_(std::move(config)) {}

void Timer::read_inputs() {
  input_probe_ = probe_input_files(config_);
  if (!input_probe_.ok()) {
    result_.status = "invalid_inputs";
    throw std::runtime_error(format_input_errors(input_probe_));
  }

  liberty_ = load_liberty_files(config_.liberty_files);
  result_.num_liberty_files = liberty_.num_files;
  result_.has_def = has_probed_file(input_probe_, InputKind::kDef);
  result_.has_verilog = has_probed_file(input_probe_, InputKind::kVerilog);
  result_.has_spef = has_probed_file(input_probe_, InputKind::kSpef);
  result_.has_sdc = has_probed_file(input_probe_, InputKind::kSdc);
  result_.num_input_files = static_cast<int>(input_probe_.files.size());
  result_.total_input_bytes = input_probe_.total_size_bytes;
  result_.num_liberty_cells = liberty_.num_cells;
  result_.num_liberty_pins = liberty_.num_pins;
  result_.num_liberty_pin_caps = liberty_.num_pin_caps;
  result_.num_liberty_lut_templates = liberty_.num_lut_templates;
  result_.num_liberty_timing_luts = liberty_.num_timing_luts;
  result_.num_timing_arcs = liberty_.num_timing_arcs;
  result_.status = "inputs_recorded";
}

void Timer::build_design() {
  design_ = DesignDB{};
  bool loaded_def = false;
  if (!config_.def_file.empty()) {
    log_api_stage("read_def_design", [&] {
      design_ = read_def_design(config_.def_file);
    });
    loaded_def = true;
  }
  const bool def_has_design =
      loaded_def && !design_.instances.empty() && !design_.nets.empty();
  if (!config_.verilog_file.empty()) {
    if (!def_has_design || config_.read_verilog_after_def) {
      log_api_stage("read_verilog_design", [&] {
        read_verilog_design_into(config_.verilog_file, &design_);
      });
    } else {
      std::cerr << "[stage] read_verilog_design skipped reason=def_has_instances_and_nets\n"
                << std::flush;
    }
  }
  log_api_stage("design_update_counts", [&] {
    design_.update_counts_preserve_connection_keys();
  });
  result_.num_instances = design_.num_instances;
  result_.num_nets = design_.num_nets;
  result_.num_pins = design_.num_pins;
  result_.status = "design_built";
}

void Timer::build_timing_graph() {
  graph_ = TimingGraph{};
  graph_.num_pins = design_.num_pins;

  for (const auto& instance : design_.instances) {
    int library_index = -1;
    int cell_index = -1;
    const LibertyCell* liberty_cell = find_liberty_cell(
        liberty_, instance.cell_name, &library_index, &cell_index);
    if (liberty_cell == nullptr) {
      ++graph_.num_unmatched_instances;
      continue;
    }

    for (std::size_t pin_id = 0; pin_id < liberty_cell->pins.size(); ++pin_id) {
      const auto& pin = liberty_cell->pins[pin_id];
      for (std::size_t arc_id = 0; arc_id < pin.timing_arcs.size();
           ++arc_id) {
        const auto& liberty_arc = pin.timing_arcs[arc_id];
        TimingGraphArc graph_arc;
        graph_arc.instance_name = instance.name;
        graph_arc.cell_name = instance.cell_name;
        graph_arc.from_pin = liberty_arc.related_pin;
        graph_arc.to_pin = liberty_arc.output_pin;
        graph_arc.timing_sense = liberty_arc.timing_sense;
        graph_arc.timing_type = liberty_arc.timing_type;
        graph_arc.library_index = library_index;
        graph_arc.cell_index = cell_index;
        graph_arc.output_pin_index = static_cast<int>(pin_id);
        graph_arc.timing_arc_index = static_cast<int>(arc_id);
        graph_.arcs.push_back(std::move(graph_arc));
      }
    }
  }

  graph_.num_arcs = static_cast<int>(graph_.arcs.size());
  graph_.num_levels = 0;
  result_.num_timing_arcs = graph_.num_arcs;
  result_.status = "timing_graph_seed_built";
}

void Timer::build_rc_graph() {
  SpefParseOptions spef_options;
  spef_options.keep_coupling_caps = config_.keep_coupling_caps;
  spef_options.coupling_reduction_factor =
      config_.coupling_reduction_factor;
  rc_ = load_spef_rc_graph(config_.spef_file, spef_options);
  result_.num_rc_nets = rc_.num_nets;
  result_.num_rc_nodes = rc_.num_nodes;
  result_.num_rc_resistors = rc_.num_resistors;
  result_.num_rc_capacitors = rc_.num_capacitors;
  result_.num_rc_coupling_capacitors = rc_.num_coupling_capacitors;
  result_.num_rc_ground_cap_nodes = rc_.num_ground_cap_nodes;
  result_.num_rc_connections = rc_.num_connections;
  result_.rc_includes_pin_caps = rc_.includes_pin_caps ? 1 : 0;
  result_.status = "rc_graph_built";
}

void Timer::update_timing() {
  flat_ = build_timing_flat_db(graph_, liberty_);
  result_.num_flat_arcs = flat_.num_arcs;
  result_.num_flat_luts = flat_.num_luts;
  result_.num_flat_values = flat_.num_values;

  cpu_timing_ = evaluate_timing_cpu(&flat_);
  if (!cpu_timing_.ok) {
    result_.status = "cpu_timing_error: " + cpu_timing_.message;
    throw std::runtime_error(result_.status);
  }
  result_.max_cell_delay = cpu_timing_.max_delay;
  result_.max_output_slew = cpu_timing_.max_output_slew;
  result_.max_arrival = cpu_timing_.max_arrival;

  if (config_.use_cuda) {
    const CudaStatus status = run_cuda_runtime_smoke_test();
    result_.cuda_smoke_ok = status.ok;
    result_.cuda_smoke_ms = status.elapsed_ms;
    if (!status.ok) {
      result_.status = "cuda_error: " + status.message;
      throw std::runtime_error(result_.status);
    }

    cuda_timing_ms_ = 0.0f;
    cuda_timing_ = evaluate_timing_cuda(flat_, &cuda_timing_ms_);
    result_.cuda_timing_ok = cuda_timing_.ok;
    result_.cuda_timing_ms = cuda_timing_ms_;
    result_.cuda_max_cell_delay = cuda_timing_.max_delay;
    result_.cuda_max_output_slew = cuda_timing_.max_output_slew;
    result_.cuda_max_arrival = cuda_timing_.max_arrival;
    result_.cpu_cuda_max_abs_error =
        timing_result_max_abs_error(cpu_timing_, cuda_timing_);
    if (!cuda_timing_.ok) {
      result_.status = "cuda_timing_error: " + cuda_timing_.message;
      throw std::runtime_error(result_.status);
    }
  }

  result_.wns = 0.0;
  result_.tns = 0.0;
  result_.status = "single_level_timing_completed";
}

const TimerConfig& Timer::config() const {
  return config_;
}

const TimerResult& Timer::result() const {
  return result_;
}

std::string Timer::input_summary() const {
  std::ostringstream os;
  os << "mode: " << timing_mode_name(config_.mode) << '\n';
  os << "use_cuda: " << (config_.use_cuda ? "true" : "false") << '\n';
  os << "read_verilog_after_def: "
     << (config_.read_verilog_after_def ? "true" : "false") << '\n';
  os << "keep_coupling_caps: "
     << (config_.keep_coupling_caps ? "true" : "false") << '\n';
  os << "coupling_reduction_factor: "
     << config_.coupling_reduction_factor << '\n';
  os << "liberty_files: " << config_.liberty_files.size() << '\n';
  for (const auto& liberty_file : config_.liberty_files) {
    os << "  - " << liberty_file << '\n';
  }
  os << "def: " << (config_.def_file.empty() ? "<none>" : config_.def_file)
     << '\n';
  os << "verilog: "
     << (config_.verilog_file.empty() ? "<none>" : config_.verilog_file)
     << '\n';
  os << "spef: " << (config_.spef_file.empty() ? "<none>" : config_.spef_file)
     << '\n';
  os << "sdc: " << (config_.sdc_file.empty() ? "<none>" : config_.sdc_file)
     << '\n';
  os << "report: "
     << (config_.report_file.empty() ? "<none>" : config_.report_file)
     << '\n';
  os << "dump_net: "
     << (config_.dump_net.empty() ? "<none>" : config_.dump_net) << '\n';
  return os.str();
}

std::string Timer::input_probe_summary() const {
  return format_input_probe(input_probe_);
}

std::string Timer::design_summary() const {
  return format_design_summary(design_);
}

std::string Timer::timing_graph_summary() const {
  return format_timing_graph_summary(graph_);
}

std::string Timer::timing_flat_summary() const {
  return format_timing_flat_summary(flat_);
}

std::string Timer::cpu_timing_summary() const {
  return format_timing_eval_summary(cpu_timing_, "cpu");
}

std::string Timer::cuda_timing_summary() const {
  std::ostringstream os;
  os << format_timing_eval_summary(cuda_timing_, "cuda");
  os << "cuda_timing_ms: " << cuda_timing_ms_ << '\n';
  os << "cpu_cuda_max_abs_error: " << result_.cpu_cuda_max_abs_error << '\n';
  return os.str();
}

std::string Timer::liberty_summary() const {
  return format_liberty_summary(liberty_);
}

std::string Timer::rc_summary() const {
  return format_rc_summary(rc_);
}

std::string Timer::result_summary() const {
  std::ostringstream os;
  os << "status: " << result_.status << '\n';
  os << "inputs: liberty=" << result_.num_liberty_files
     << " def=" << result_.has_def
     << " verilog=" << result_.has_verilog
     << " spef=" << result_.has_spef
     << " sdc=" << result_.has_sdc << '\n';
  os << "input_files: " << result_.num_input_files << '\n';
  os << "total_input_bytes: " << result_.total_input_bytes << '\n';
  os << "liberty_cells: " << result_.num_liberty_cells << '\n';
  os << "liberty_pins: " << result_.num_liberty_pins << '\n';
  os << "liberty_pin_caps: " << result_.num_liberty_pin_caps << '\n';
  os << "liberty_lut_templates: " << result_.num_liberty_lut_templates
     << '\n';
  os << "liberty_timing_luts: " << result_.num_liberty_timing_luts << '\n';
  os << "flat_arcs: " << result_.num_flat_arcs << '\n';
  os << "flat_luts: " << result_.num_flat_luts << '\n';
  os << "flat_values: " << result_.num_flat_values << '\n';
  os << "rc_nets: " << result_.num_rc_nets << '\n';
  os << "rc_nodes: " << result_.num_rc_nodes << '\n';
  os << "rc_connections: " << result_.num_rc_connections << '\n';
  os << "rc_resistors: " << result_.num_rc_resistors << '\n';
  os << "rc_capacitors: " << result_.num_rc_capacitors << '\n';
  os << "rc_coupling_capacitors: " << result_.num_rc_coupling_capacitors
     << '\n';
  os << "rc_ground_cap_nodes: " << result_.num_rc_ground_cap_nodes << '\n';
  os << "rc_includes_pin_caps: " << result_.rc_includes_pin_caps << '\n';
  os << "design: instances=" << result_.num_instances
     << " nets=" << result_.num_nets
     << " pins=" << result_.num_pins << '\n';
  os << "timing_arcs: " << result_.num_timing_arcs << '\n';
  os << "max_cell_delay: " << result_.max_cell_delay << '\n';
  os << "max_output_slew: " << result_.max_output_slew << '\n';
  os << "max_arrival: " << result_.max_arrival << '\n';
  os << "wns: " << result_.wns << '\n';
  os << "tns: " << result_.tns << '\n';
  if (config_.use_cuda) {
    os << "cuda_smoke_ok: " << (result_.cuda_smoke_ok ? "true" : "false")
       << '\n';
    os << "cuda_smoke_ms: " << result_.cuda_smoke_ms << '\n';
    os << "cuda_timing_ok: " << (result_.cuda_timing_ok ? "true" : "false")
       << '\n';
    os << "cuda_timing_ms: " << result_.cuda_timing_ms << '\n';
    os << "cuda_max_cell_delay: " << result_.cuda_max_cell_delay << '\n';
    os << "cuda_max_output_slew: " << result_.cuda_max_output_slew << '\n';
    os << "cuda_max_arrival: " << result_.cuda_max_arrival << '\n';
    os << "cpu_cuda_max_abs_error: " << result_.cpu_cuda_max_abs_error
       << '\n';
  }
  return os.str();
}

void Timer::write_report(const std::string& path) const {
  if (path.empty()) {
    return;
  }

  std::ofstream report(path);
  if (!report) {
    throw std::runtime_error("failed to open report file: " + path);
  }
  report << result_summary();
}

void Timer::dump_net_debug(const std::string& net_name,
                           const std::string& path) const {
  if (net_name.empty() || path.empty()) {
    return;
  }

  std::ofstream dump(path);
  if (!dump) {
    throw std::runtime_error("failed to open net debug dump: " + path);
  }
  dump << "net: " << net_name << '\n';
  for (const auto& net : rc_.nets) {
    if (net.name == net_name) {
      dump << "matched: true\n";
      dump << "total_cap: " << net.total_capacitance << '\n';
      dump << "nodes: " << net.nodes.size() << '\n';
      dump << "connections: " << net.connections.size() << '\n';
      dump << "capacitors: " << net.capacitors.size() << '\n';
      dump << "ground_cap_nodes: "
           << std::count_if(net.nodes.begin(), net.nodes.end(), [](const RcNode& node) {
                return node.ground_capacitance > 0.0;
              })
           << '\n';
      dump << "resistors: " << net.resistors.size() << '\n';
      return;
    }
  }
  dump << "matched: false\n";
  dump << "rc_nodes: " << rc_.num_nodes << '\n';
  dump << "rc_resistors: " << rc_.num_resistors << '\n';
  dump << "rc_capacitors: " << rc_.num_capacitors << '\n';
  dump << "rc_ground_cap_nodes: " << rc_.num_ground_cap_nodes << '\n';
  dump << "includes_pin_caps: " << (rc_.includes_pin_caps ? "true" : "false")
       << '\n';
}

}  // namespace stimer
