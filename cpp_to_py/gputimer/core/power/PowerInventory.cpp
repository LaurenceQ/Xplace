#include "gputimer/core/GPUTimer.h"
#include "gputimer/core/DmpModel.h"
#include "gputimer/core/power/PowerCudaModel.h"
#include "gputimer/core/power/PowerHostCommon.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"
#include "common/lib/Liberty.h"
#include "common/lib/Lut.h"
#include "common/lib/Timing.h"
#include "gputimer/db/GTDatabase.h"
#include "io_parser/gp/GPDatabase.h"

#include <torch/cuda.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gt {

tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> GPUTimer::report_power_liberty_inventory() {
    int64_t internal_groups = static_cast<int64_t>(gtdb.liberty_internal_powers.size());
    int64_t internal_rise_luts = 0;
    int64_t internal_fall_luts = 0;
    int64_t internal_when_exprs = 0;
    for (auto* internal_power : gtdb.liberty_internal_powers) {
        if (!internal_power) continue;
        if (internal_power->power_[RISE] && internal_power->power_[RISE]->set_) internal_rise_luts++;
        if (internal_power->power_[FALL] && internal_power->power_[FALL]->set_) internal_fall_luts++;
        if (!internal_power->when_expr_.empty()) internal_when_exprs++;
    }

    int64_t leakage_groups = static_cast<int64_t>(gtdb.liberty_leakage_powers.size());
    int64_t leakage_when_exprs = 0;
    for (auto* leakage_power : gtdb.liberty_leakage_powers) {
        if (leakage_power && !leakage_power->when_expr_.empty()) leakage_when_exprs++;
    }

    int64_t output_functions = 0;
    for (uint8_t has_function : gtdb.liberty_port_has_function) {
        if (has_function) output_functions++;
    }

    return {internal_groups, internal_rise_luts, internal_fall_luts, internal_when_exprs, leakage_groups, leakage_when_exprs, output_functions};
}

int64_t GPUTimer::report_power_seq_inventory() {
    int64_t seqs = 0;
    for (const auto* cell_type : gtdb.rawdb.celltypes) {
        if (cell_type && cell_type->liberty_cell)
            seqs += static_cast<int64_t>(cell_type->liberty_cell->sequentials_.size());
    }
    return seqs;
}


}  // namespace gt
