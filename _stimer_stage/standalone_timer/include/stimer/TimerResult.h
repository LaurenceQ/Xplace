#pragma once

#include <cstdint>
#include <limits>
#include <string>

namespace stimer {

struct TimerResult {
  int num_liberty_files = 0;
  int has_def = 0;
  int has_verilog = 0;
  int has_spef = 0;
  int has_sdc = 0;
  int num_input_files = 0;
  std::uintmax_t total_input_bytes = 0;
  int num_instances = 0;
  int num_nets = 0;
  int num_pins = 0;
  int num_liberty_cells = 0;
  int num_liberty_pins = 0;
  int num_liberty_pin_caps = 0;
  int num_liberty_lut_templates = 0;
  int num_liberty_timing_luts = 0;
  int num_timing_arcs = 0;
  int num_flat_arcs = 0;
  int num_flat_luts = 0;
  int num_flat_values = 0;
  int num_rc_nets = 0;
  int num_rc_nodes = 0;
  int num_rc_resistors = 0;
  int num_rc_capacitors = 0;
  int num_rc_coupling_capacitors = 0;
  int num_rc_ground_cap_nodes = 0;
  int num_rc_connections = 0;
  int rc_includes_pin_caps = 0;
  double wns = std::numeric_limits<double>::quiet_NaN();
  double tns = std::numeric_limits<double>::quiet_NaN();
  float cuda_smoke_ms = 0.0f;
  bool cuda_smoke_ok = false;
  float max_cell_delay = 0.0f;
  float max_output_slew = 0.0f;
  float max_arrival = 0.0f;
  float cuda_timing_ms = 0.0f;
  float cuda_max_cell_delay = 0.0f;
  float cuda_max_output_slew = 0.0f;
  float cuda_max_arrival = 0.0f;
  float cpu_cuda_max_abs_error = 0.0f;
  bool cuda_timing_ok = false;
  std::string status = "not_started";
};

}  // namespace stimer
