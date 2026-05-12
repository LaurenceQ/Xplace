#include "stimer/TimingFlatDB.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace stimer {
namespace {

constexpr int kRise = static_cast<int>(CapTransition::kRise);
constexpr int kFall = static_cast<int>(CapTransition::kFall);

const LibertyTimingArc* find_liberty_arc(const TimingGraphArc& graph_arc,
                                         const LibertyDB& liberty) {
  if (graph_arc.library_index < 0 || graph_arc.cell_index < 0 ||
      graph_arc.output_pin_index < 0 || graph_arc.timing_arc_index < 0) {
    return nullptr;
  }

  const auto lib_id = static_cast<std::size_t>(graph_arc.library_index);
  const auto cell_id = static_cast<std::size_t>(graph_arc.cell_index);
  const auto pin_id = static_cast<std::size_t>(graph_arc.output_pin_index);
  const auto arc_id = static_cast<std::size_t>(graph_arc.timing_arc_index);
  if (lib_id >= liberty.libraries.size()) {
    return nullptr;
  }
  const auto& library = liberty.libraries[lib_id];
  if (cell_id >= library.cells.size()) {
    return nullptr;
  }
  const auto& cell = library.cells[cell_id];
  if (pin_id >= cell.pins.size()) {
    return nullptr;
  }
  const auto& pin = cell.pins[pin_id];
  if (arc_id >= pin.timing_arcs.size()) {
    return nullptr;
  }
  return &pin.timing_arcs[arc_id];
}

int append_lut(const LibertyLut& lut, TimingFlatDB* flat) {
  if (!lut.valid || lut.values.empty()) {
    return -1;
  }

  FlatLut flat_lut;
  flat_lut.index1_begin = static_cast<int>(flat->indices1.size());
  flat_lut.index1_size = static_cast<int>(lut.indices1.size());
  flat_lut.index2_begin = static_cast<int>(flat->indices2.size());
  flat_lut.index2_size = static_cast<int>(lut.indices2.size());
  flat_lut.values_begin = static_cast<int>(flat->values.size());
  flat_lut.value_size = static_cast<int>(lut.values.size());
  flat_lut.variable1 = static_cast<int>(lut.variable1);
  flat_lut.variable2 = static_cast<int>(lut.variable2);
  flat_lut.valid = true;

  flat->indices1.insert(flat->indices1.end(), lut.indices1.begin(),
                        lut.indices1.end());
  flat->indices2.insert(flat->indices2.end(), lut.indices2.begin(),
                        lut.indices2.end());
  flat->values.insert(flat->values.end(), lut.values.begin(), lut.values.end());

  const int id = static_cast<int>(flat->luts.size());
  flat->luts.push_back(flat_lut);
  return id;
}

bool variable_is_slew(int variable) {
  return variable == static_cast<int>(LibertyLutVariable::kInputNetTransition) ||
         variable ==
             static_cast<int>(LibertyLutVariable::kRelatedPinTransition) ||
         variable ==
             static_cast<int>(LibertyLutVariable::kConstrainedPinTransition) ||
         variable ==
             static_cast<int>(LibertyLutVariable::kInputTransitionTime);
}

bool variable_is_cap(int variable) {
  return variable ==
         static_cast<int>(LibertyLutVariable::kTotalOutputNetCapacitance);
}

float first_index_value(const FlatLut& lut,
                        const std::vector<float>& indices1,
                        const std::vector<float>& indices2,
                        int variable) {
  if (variable == lut.variable1 && lut.index1_size > 0) {
    return indices1[static_cast<std::size_t>(lut.index1_begin)];
  }
  if (variable == lut.variable2 && lut.index2_size > 0) {
    return indices2[static_cast<std::size_t>(lut.index2_begin)];
  }
  return 0.0f;
}

float select_probe_value(const TimingFlatDB& flat,
                         const FlatTimingArc& arc,
                         bool want_slew) {
  const int lut_id = arc.delay_rise_lut >= 0 ? arc.delay_rise_lut
                                             : arc.delay_fall_lut;
  if (lut_id < 0 || static_cast<std::size_t>(lut_id) >= flat.luts.size()) {
    return 0.0f;
  }
  const FlatLut& lut = flat.luts[static_cast<std::size_t>(lut_id)];
  if (want_slew) {
    if (variable_is_slew(lut.variable1)) {
      return first_index_value(lut, flat.indices1, flat.indices2,
                               lut.variable1);
    }
    if (variable_is_slew(lut.variable2)) {
      return first_index_value(lut, flat.indices1, flat.indices2,
                               lut.variable2);
    }
  } else {
    if (variable_is_cap(lut.variable1)) {
      return first_index_value(lut, flat.indices1, flat.indices2,
                               lut.variable1);
    }
    if (variable_is_cap(lut.variable2)) {
      return first_index_value(lut, flat.indices1, flat.indices2,
                               lut.variable2);
    }
  }
  return 0.0f;
}

int lower_index(const std::vector<float>& data, int begin, int size, float x) {
  if (size <= 1) {
    return 0;
  }
  if (x <= data[static_cast<std::size_t>(begin)]) {
    return 0;
  }
  for (int i = 0; i + 1 < size; ++i) {
    const float hi = data[static_cast<std::size_t>(begin + i + 1)];
    if (x <= hi) {
      return i;
    }
  }
  return size - 2;
}

float axis_value(int variable, float input_slew, float load_cap) {
  if (variable_is_slew(variable)) {
    return input_slew;
  }
  if (variable_is_cap(variable)) {
    return load_cap;
  }
  return 0.0f;
}

float interpolate_lut(const TimingFlatDB& flat,
                      int lut_id,
                      float input_slew,
                      float load_cap) {
  if (lut_id < 0 || static_cast<std::size_t>(lut_id) >= flat.luts.size()) {
    return 0.0f;
  }
  const FlatLut& lut = flat.luts[static_cast<std::size_t>(lut_id)];
  if (!lut.valid || lut.value_size == 0) {
    return 0.0f;
  }

  const int size1 = std::max(lut.index1_size, 1);
  const int size2 = std::max(lut.index2_size, 1);
  if (size1 == 1 && size2 == 1) {
    return flat.values[static_cast<std::size_t>(lut.values_begin)];
  }

  const float x = axis_value(lut.variable1, input_slew, load_cap);
  const float y = axis_value(lut.variable2, input_slew, load_cap);
  const int x0_id = lower_index(flat.indices1, lut.index1_begin,
                                lut.index1_size, x);
  const int y0_id = lower_index(flat.indices2, lut.index2_begin,
                                lut.index2_size, y);
  const int x1_id = lut.index1_size <= 1 ? x0_id : x0_id + 1;
  const int y1_id = lut.index2_size <= 1 ? y0_id : y0_id + 1;

  const float x0 =
      lut.index1_size <= 1
          ? x
          : flat.indices1[static_cast<std::size_t>(lut.index1_begin + x0_id)];
  const float x1 =
      lut.index1_size <= 1
          ? x
          : flat.indices1[static_cast<std::size_t>(lut.index1_begin + x1_id)];
  const float y0 =
      lut.index2_size <= 1
          ? y
          : flat.indices2[static_cast<std::size_t>(lut.index2_begin + y0_id)];
  const float y1 =
      lut.index2_size <= 1
          ? y
          : flat.indices2[static_cast<std::size_t>(lut.index2_begin + y1_id)];

  const auto value_at = [&](int i, int j) {
    const int offset = lut.values_begin + i * size2 + j;
    if (offset < lut.values_begin || offset >= lut.values_begin + lut.value_size) {
      return 0.0f;
    }
    return flat.values[static_cast<std::size_t>(offset)];
  };

  const float q00 = value_at(x0_id, y0_id);
  const float q01 = value_at(x0_id, y1_id);
  const float q10 = value_at(x1_id, y0_id);
  const float q11 = value_at(x1_id, y1_id);
  const float tx = std::fabs(x1 - x0) < 1e-12f ? 0.0f : (x - x0) / (x1 - x0);
  const float ty = std::fabs(y1 - y0) < 1e-12f ? 0.0f : (y - y0) / (y1 - y0);
  const float a = q00 * (1.0f - tx) + q10 * tx;
  const float b = q01 * (1.0f - tx) + q11 * tx;
  return a * (1.0f - ty) + b * ty;
}

}  // namespace

TimingFlatDB build_timing_flat_db(const TimingGraph& graph,
                                  const LibertyDB& liberty) {
  TimingFlatDB flat;
  flat.arcs.reserve(graph.arcs.size());

  for (const auto& graph_arc : graph.arcs) {
    const LibertyTimingArc* liberty_arc = find_liberty_arc(graph_arc, liberty);
    if (liberty_arc == nullptr) {
      continue;
    }

    FlatTimingArc flat_arc;
    flat_arc.delay_rise_lut = append_lut(liberty_arc->cell_delay[kRise], &flat);
    flat_arc.delay_fall_lut = append_lut(liberty_arc->cell_delay[kFall], &flat);
    flat_arc.slew_rise_lut = append_lut(liberty_arc->transition[kRise], &flat);
    flat_arc.slew_fall_lut = append_lut(liberty_arc->transition[kFall], &flat);
    if (flat_arc.delay_rise_lut < 0 && flat_arc.delay_fall_lut < 0) {
      ++flat.num_missing_delay_luts;
    }
    if (flat_arc.slew_rise_lut < 0 && flat_arc.slew_fall_lut < 0) {
      ++flat.num_missing_slew_luts;
    }
    flat.arcs.push_back(flat_arc);
  }

  flat.num_arcs = static_cast<int>(flat.arcs.size());
  flat.num_luts = static_cast<int>(flat.luts.size());
  flat.num_values = static_cast<int>(flat.values.size());
  flat.arc_input_slew.resize(flat.arcs.size(), 0.0f);
  flat.arc_load_cap.resize(flat.arcs.size(), 0.0f);
  flat.arc_delay.resize(flat.arcs.size(), 0.0f);
  flat.arc_output_slew.resize(flat.arcs.size(), 0.0f);
  flat.arc_arrival.resize(flat.arcs.size(), 0.0f);

  for (std::size_t i = 0; i < flat.arcs.size(); ++i) {
    flat.arc_input_slew[i] = select_probe_value(flat, flat.arcs[i], true);
    flat.arc_load_cap[i] = select_probe_value(flat, flat.arcs[i], false);
  }
  return flat;
}

TimingEvalResult evaluate_timing_cpu(TimingFlatDB* flat) {
  TimingEvalResult result;
  if (flat == nullptr) {
    result.ok = false;
    result.message = "null TimingFlatDB";
    return result;
  }

  result.num_arcs = flat->num_arcs;
  result.num_luts = flat->num_luts;
  result.num_values = flat->num_values;
  for (std::size_t i = 0; i < flat->arcs.size(); ++i) {
    const FlatTimingArc& arc = flat->arcs[i];
    const float input_slew = flat->arc_input_slew[i];
    const float load_cap = flat->arc_load_cap[i];
    const int delay_lut =
        arc.delay_rise_lut >= 0 ? arc.delay_rise_lut : arc.delay_fall_lut;
    const int slew_lut =
        arc.slew_rise_lut >= 0 ? arc.slew_rise_lut : arc.slew_fall_lut;

    const float delay = interpolate_lut(*flat, delay_lut, input_slew, load_cap);
    const float output_slew =
        interpolate_lut(*flat, slew_lut, input_slew, load_cap);
    flat->arc_delay[i] = delay;
    flat->arc_output_slew[i] = output_slew;
    flat->arc_arrival[i] = delay;
    result.max_delay = std::max(result.max_delay, delay);
    result.max_output_slew = std::max(result.max_output_slew, output_slew);
    result.max_arrival = std::max(result.max_arrival, flat->arc_arrival[i]);
  }
  return result;
}

std::string format_timing_flat_summary(const TimingFlatDB& flat) {
  std::ostringstream os;
  os << "flat_arcs: " << flat.num_arcs << '\n';
  os << "flat_luts: " << flat.num_luts << '\n';
  os << "flat_values: " << flat.num_values << '\n';
  os << "flat_missing_delay_luts: " << flat.num_missing_delay_luts << '\n';
  os << "flat_missing_slew_luts: " << flat.num_missing_slew_luts << '\n';
  const std::size_t arc_limit = std::min<std::size_t>(flat.arcs.size(), 5);
  for (std::size_t i = 0; i < arc_limit; ++i) {
    os << "  - flat_arc=" << i
       << " in_slew=" << flat.arc_input_slew[i]
       << " load_cap=" << flat.arc_load_cap[i]
       << " delay_lut=" << flat.arcs[i].delay_rise_lut
       << " slew_lut=" << flat.arcs[i].slew_rise_lut << '\n';
  }
  return os.str();
}

std::string format_timing_eval_summary(const TimingEvalResult& result,
                                       const char* label) {
  std::ostringstream os;
  os << label << "_timing_ok: " << (result.ok ? "true" : "false") << '\n';
  os << label << "_timing_message: " << result.message << '\n';
  os << label << "_timing_arcs: " << result.num_arcs << '\n';
  os << label << "_timing_luts: " << result.num_luts << '\n';
  os << label << "_timing_values: " << result.num_values << '\n';
  os << label << "_max_delay: " << result.max_delay << '\n';
  os << label << "_max_output_slew: " << result.max_output_slew << '\n';
  os << label << "_max_arrival: " << result.max_arrival << '\n';
  return os.str();
}

}  // namespace stimer
