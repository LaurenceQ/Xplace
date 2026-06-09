#pragma once

#include "common/lib/Timing.h"

#include <string>
#include <unordered_set>

namespace gt {

bool gputimer_env_enabled(const char* name);
bool is_transition_defined_cpu(const TimingArc* timing_arc, int input_rf, int output_rf);
void warn_missing_sdc_object(const char* command, const char* kind, const std::string& name);
std::string pin_name_colon_to_slash(const std::string& name);
void add_pin_name_target_variants(std::unordered_set<std::string>& targets, const std::string& name);

}  // namespace gt
