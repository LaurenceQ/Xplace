#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace stimer {

enum class CapTransition {
  kRise = 0,
  kFall = 1,
};

enum class CapRange {
  kMin = 0,
  kMax = 1,
};

enum class LibertyLutVariable {
  kUnknown,
  kTotalOutputNetCapacitance,
  kInputNetTransition,
  kConstrainedPinTransition,
  kRelatedPinTransition,
  kInputTransitionTime,
};

struct LibertyLutTemplate {
  std::string name;
  LibertyLutVariable variable1 = LibertyLutVariable::kUnknown;
  LibertyLutVariable variable2 = LibertyLutVariable::kUnknown;
  std::vector<float> indices1;
  std::vector<float> indices2;
};

struct LibertyLut {
  std::string template_name;
  LibertyLutVariable variable1 = LibertyLutVariable::kUnknown;
  LibertyLutVariable variable2 = LibertyLutVariable::kUnknown;
  std::vector<float> indices1;
  std::vector<float> indices2;
  std::vector<float> values;
  bool valid = false;
};

struct LibertyPinCap {
  std::array<std::array<float, 2>, 2> value{};
  std::array<std::array<bool, 2>, 2> valid{};

  bool complete() const;
};

struct LibertyTimingArc {
  std::string related_pin;
  std::string output_pin;
  std::string timing_sense = "unknown";
  std::string timing_type = "unknown";
  std::string sdf_cond;
  bool is_cond = false;

  std::array<LibertyLut, 2> cell_delay;
  std::array<LibertyLut, 2> transition;
  std::array<LibertyLut, 2> constraint;
};

struct LibertyPin {
  std::string name;
  std::string direction;
  bool is_clock = false;
  LibertyPinCap cap;
  std::vector<LibertyTimingArc> timing_arcs;
};

struct LibertyCell {
  std::string name;
  std::vector<LibertyPin> pins;
  int num_timing_groups = 0;
};

struct LibertyLibrary {
  std::string path;
  std::string name;
  std::string capacitive_load_unit;
  float default_input_pin_cap = 0.0f;
  float default_output_pin_cap = 0.0f;
  float default_inout_pin_cap = 0.0f;
  bool has_default_input_pin_cap = false;
  bool has_default_output_pin_cap = false;
  bool has_default_inout_pin_cap = false;
  std::vector<LibertyLutTemplate> lut_templates;
  std::vector<LibertyCell> cells;
};

struct LibertyDB {
  int num_files = 0;
  int num_cells = 0;
  int num_pins = 0;
  int num_pin_caps = 0;
  int num_timing_arcs = 0;
  int num_lut_templates = 0;
  int num_timing_luts = 0;
  std::vector<LibertyLibrary> libraries;
};

const char* liberty_lut_variable_name(LibertyLutVariable variable);
std::string format_liberty_summary(const LibertyDB& db);

}  // namespace stimer
