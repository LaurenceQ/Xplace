#pragma once

#include <string>

#include "stimer/DesignDB.h"
#include "stimer/InputFiles.h"
#include "stimer/LibertyDB.h"
#include "stimer/RcGraph.h"
#include "stimer/TimerConfig.h"
#include "stimer/TimerResult.h"
#include "stimer/TimingFlatDB.h"
#include "stimer/TimingGraph.h"

namespace stimer {

class Timer {
 public:
  explicit Timer(TimerConfig config);

  void read_inputs();
  void build_design();
  void build_timing_graph();
  void build_rc_graph();
  void update_timing();

  const TimerConfig& config() const;
  const TimerResult& result() const;

  std::string input_summary() const;
  std::string input_probe_summary() const;
  std::string design_summary() const;
  std::string timing_graph_summary() const;
  std::string timing_flat_summary() const;
  std::string cpu_timing_summary() const;
  std::string cuda_timing_summary() const;
  std::string liberty_summary() const;
  std::string rc_summary() const;
  std::string result_summary() const;
  void write_report(const std::string& path) const;
  void dump_net_debug(const std::string& net_name,
                      const std::string& path) const;

 private:
  TimerConfig config_;
  DesignDB design_;
  LibertyDB liberty_;
  InputProbeResult input_probe_;
  TimingGraph graph_;
  TimingFlatDB flat_;
  TimingEvalResult cpu_timing_;
  TimingEvalResult cuda_timing_;
  RcGraph rc_;
  TimerResult result_;
  float cuda_timing_ms_ = 0.0f;
};

}  // namespace stimer
