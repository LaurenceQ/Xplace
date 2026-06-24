#include "gputimer/db/sdc/SdcUtils.h"

#include "common/XplaceLog.h"

#include <utility>

namespace gt {

bool is_transition_defined_cpu(const TimingArc* timing_arc, int input_rf, int output_rf) {
    if (timing_arc->is_rising_edge_triggered() && input_rf != RISE) return false;
    if (timing_arc->is_falling_edge_triggered() && input_rf != FALL) return false;

    switch (timing_arc->timing_sense_) {
        case TimingSense::positive_unate:
            return input_rf == output_rf;
        case TimingSense::negative_unate:
            return input_rf != output_rf;
        default:
            return true;
    }
}

void warn_missing_sdc_object(const char* command, const char* kind, const std::string& name) {
    XPLACE_DEBUGF("GPUTIMER_VERBOSE_SDC_WARNINGS",
                  "%s: %s \"%s\" not found",
                  command, kind, name.c_str());
}

std::string pin_name_colon_to_slash(const std::string& name) {
    std::string key = name;
    const auto colon_pos = key.rfind(':');
    if (colon_pos != std::string::npos) {
        key[colon_pos] = '/';
    }
    return key;
}

void add_pin_name_target_variants(std::unordered_set<std::string>& targets, const std::string& name) {
    if (name.empty()) {
        return;
    }
    targets.insert(name);
    const auto slash_pos = name.rfind('/');
    if (slash_pos != std::string::npos) {
        std::string gp_pin_name = name;
        gp_pin_name[slash_pos] = ':';
        targets.insert(std::move(gp_pin_name));
    }
}

}  // namespace gt
