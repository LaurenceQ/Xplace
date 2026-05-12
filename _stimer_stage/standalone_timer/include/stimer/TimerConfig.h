#pragma once

#include <string>
#include <vector>

namespace stimer {

enum class TimingMode {
  kElmore,
  kDmp,
};

struct TimerConfig {
  std::vector<std::string> liberty_files;
  std::string def_file;
  std::string verilog_file;
  std::string spef_file;
  std::string sdc_file;
  std::string report_file;
  std::string dump_net;
  TimingMode mode = TimingMode::kDmp;
  bool use_cuda = true;
  bool dump_debug = false;
  bool read_verilog_after_def = false;
  bool keep_coupling_caps = false;
  double coupling_reduction_factor = 1.0;
};

std::string timing_mode_name(TimingMode mode);

}  // namespace stimer
