#pragma once

#include <string>
#include <vector>

namespace stimer {

struct TimingGraphArc {
  std::string instance_name;
  std::string cell_name;
  std::string from_pin;
  std::string to_pin;
  std::string timing_sense;
  std::string timing_type;
  int library_index = -1;
  int cell_index = -1;
  int output_pin_index = -1;
  int timing_arc_index = -1;
};

struct TimingGraph {
  int num_pins = 0;
  int num_arcs = 0;
  int num_levels = 0;
  int num_unmatched_instances = 0;
  std::vector<TimingGraphArc> arcs;
};

std::string format_timing_graph_summary(const TimingGraph& graph);

}  // namespace stimer
