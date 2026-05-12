#pragma once

#include <string>
#include <vector>

#include "stimer/LibertyDB.h"
#include "stimer/TimingGraph.h"

namespace stimer {

struct FlatLut {
  int index1_begin = 0;
  int index1_size = 0;
  int index2_begin = 0;
  int index2_size = 0;
  int values_begin = 0;
  int value_size = 0;
  int variable1 = static_cast<int>(LibertyLutVariable::kUnknown);
  int variable2 = static_cast<int>(LibertyLutVariable::kUnknown);
  bool valid = false;
};

struct FlatTimingArc {
  int delay_rise_lut = -1;
  int delay_fall_lut = -1;
  int slew_rise_lut = -1;
  int slew_fall_lut = -1;
};

struct TimingFlatDB {
  int num_arcs = 0;
  int num_luts = 0;
  int num_values = 0;
  int num_missing_delay_luts = 0;
  int num_missing_slew_luts = 0;

  std::vector<FlatTimingArc> arcs;
  std::vector<FlatLut> luts;
  std::vector<float> indices1;
  std::vector<float> indices2;
  std::vector<float> values;

  std::vector<float> arc_input_slew;
  std::vector<float> arc_load_cap;
  std::vector<float> arc_delay;
  std::vector<float> arc_output_slew;
  std::vector<float> arc_arrival;
};

struct TimingEvalResult {
  int num_arcs = 0;
  int num_luts = 0;
  int num_values = 0;
  float max_delay = 0.0f;
  float max_output_slew = 0.0f;
  float max_arrival = 0.0f;
  bool ok = true;
  std::string message = "ok";
};

TimingFlatDB build_timing_flat_db(const TimingGraph& graph,
                                  const LibertyDB& liberty);
TimingEvalResult evaluate_timing_cpu(TimingFlatDB* flat);
std::string format_timing_flat_summary(const TimingFlatDB& flat);
std::string format_timing_eval_summary(const TimingEvalResult& result,
                                       const char* label);

}  // namespace stimer
