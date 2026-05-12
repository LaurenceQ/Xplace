#include "stimer/TimingGraph.h"

#include <algorithm>
#include <sstream>

namespace stimer {

std::string format_timing_graph_summary(const TimingGraph& graph) {
  std::ostringstream os;
  os << "timing_graph_pins: " << graph.num_pins << '\n';
  os << "timing_graph_arcs: " << graph.num_arcs << '\n';
  os << "timing_graph_levels: " << graph.num_levels << '\n';
  os << "timing_graph_unmatched_instances: "
     << graph.num_unmatched_instances << '\n';

  const std::size_t arc_limit = std::min<std::size_t>(graph.arcs.size(), 5);
  for (std::size_t i = 0; i < arc_limit; ++i) {
    const auto& arc = graph.arcs[i];
    os << "  - arc=" << arc.instance_name << '/' << arc.from_pin << "->"
       << arc.to_pin << " cell=" << arc.cell_name
       << " sense=" << arc.timing_sense
       << " type=" << arc.timing_type << '\n';
  }
  return os.str();
}

}  // namespace stimer
