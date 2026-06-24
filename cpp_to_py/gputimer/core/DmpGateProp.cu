#include "DmpGateModel.cuh"

namespace gt {

__device__ __forceinline__ double dmpFinitePositiveOr(double value, double fallback) {
    return (isfinite(value) && value > 0.0) ? value : fallback;
}

__device__ __forceinline__ double dmpThresholdArrayValue(const float* values,
                                                         int attr,
                                                         double fallback) {
    return values ? dmpFinitePositiveOr(values[attr], fallback) : fallback;
}

__device__ __forceinline__ double dmpLibraryThresholdArrayValue(const float* values,
                                                                int library_idx,
                                                                double fallback) {
    return (values && library_idx >= 0) ? dmpFinitePositiveOr(values[library_idx], fallback)
                                        : fallback;
}

__device__ int DmpModel::timingLibraryId(int timing_id) const {
    return (dmp_timing_library_ids && timing_id >= 0)
               ? dmp_timing_library_ids[timing_id]
               : -1;
}

__device__ int DmpModel::pinLibraryId(int pin_id) const {
    return (dmp_pin_library_ids && pin_id >= 0)
               ? dmp_pin_library_ids[pin_id]
               : -1;
}

__device__ __forceinline__ void dmpSanitizeThresholds(const DmpModel& dmp_db,
                                                      double& vl,
                                                      double& vh) {
    if (vh <= vl) {
        vl = dmp_db.vl_;
        vh = dmp_db.vh_;
    }
}

__device__ inline void DmpModel::loadPinThresholds(int pin_id,
                                                     int attr,
                                                     double& vth,
                                                     double& vl,
                                                     double& vh,
                                                     double& slew_derate) const {
    const int library_id = pinLibraryId(pin_id);
    const int library_idx = library_id >= 0 ? library_id * NUM_ATTR + attr : -1;
    vth = dmpLibraryThresholdArrayValue(
        dmp_library_input_thresholds,
        library_idx,
        dmpThresholdArrayValue(dmp_input_thresholds, attr, vth_));
    vl = dmpLibraryThresholdArrayValue(
        dmp_library_slew_lower_thresholds,
        library_idx,
        dmpThresholdArrayValue(dmp_slew_lower_thresholds, attr, vl_));
    vh = dmpLibraryThresholdArrayValue(
        dmp_library_slew_upper_thresholds,
        library_idx,
        dmpThresholdArrayValue(dmp_slew_upper_thresholds, attr, vh_));
    slew_derate = dmpLibraryThresholdArrayValue(
        dmp_library_slew_derates,
        library_idx,
        dmpThresholdArrayValue(dmp_slew_derates, attr, slew_derate_));
    dmpSanitizeThresholds(*this, vl, vh);
}

__device__ void DmpModel::driverLibraryThresholds(int library_id,
                                                          int attr,
                                                          double& vth,
                                                          double& vl,
                                                          double& vh,
                                                          double& slew_derate) const {
    const int library_idx = library_id >= 0 ? library_id * NUM_ATTR + attr : -1;
    vth = dmpLibraryThresholdArrayValue(
        dmp_library_output_thresholds,
        library_idx,
        dmpThresholdArrayValue(dmp_output_thresholds, attr, vth_));
    vl = dmpLibraryThresholdArrayValue(
        dmp_library_slew_lower_thresholds,
        library_idx,
        dmpThresholdArrayValue(dmp_slew_lower_thresholds, attr, vl_));
    vh = dmpLibraryThresholdArrayValue(
        dmp_library_slew_upper_thresholds,
        library_idx,
        dmpThresholdArrayValue(dmp_slew_upper_thresholds, attr, vh_));
    slew_derate = dmpLibraryThresholdArrayValue(
        dmp_library_slew_derates,
        library_idx,
        dmpThresholdArrayValue(dmp_slew_derates, attr, slew_derate_));
    dmpSanitizeThresholds(*this, vl, vh);
}

__device__ __forceinline__ float dmpDecodeWinnerFloat(unsigned int cmp_key,
                                                      bool pick_max) {
    const unsigned int ordered = pick_max ? cmp_key : ~cmp_key;
    const unsigned int raw = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u)
                                                     : ~ordered;
    return __uint_as_float(raw);
}

__device__ unsigned long long dmpPackWinner(float value,
                                                            unsigned int payload,
                                                            bool pick_max) {
    unsigned int ordered = __float_as_uint(value);
    ordered = (ordered & 0x80000000u) ? ~ordered : (ordered ^ 0x80000000u);
    const unsigned int cmp_key = pick_max ? ordered : ~ordered;
    return (static_cast<unsigned long long>(cmp_key) << 32)
           | static_cast<unsigned long long>(payload);
}

__device__ bool DmpModel::updateAtWinner(int to_slot,
                                          float at,
                                          int arc_id,
                                          int from_attr) {
    if (!isfinite(at)) {
        return false;
    }
    const bool pick_max = (to_slot & 0b10) != 0;
    const unsigned int payload = (static_cast<unsigned int>(arc_id) << 2)
                                 | static_cast<unsigned int>(from_attr & 0x3);
    const unsigned long long packed = dmpPackWinner(at, payload, pick_max);
    const unsigned long long old = atomicMax(&pin_at_winner[to_slot], packed);
    return packed > old;
}

__device__ void DmpModel::thresholdAdjust(int load_pin_id,
                                                  int load_attr,
                                                       float driver_vth,
                                                       float driver_vl,
                                                       float driver_vh,
                                                       float driver_derate,
                                                       int driver_library_id,
                                                       double& wire_delay,
                                                       double& load_slew) const {
    if (!isfinite(wire_delay) || !isfinite(load_slew)) {
        return;
    }
    const int load_library_id = pinLibraryId(load_pin_id);
    if (driver_library_id >= 0 && driver_library_id == load_library_id) {
        return;
    }
    double load_vth, load_vl, load_vh, load_derate;
    loadPinThresholds(load_pin_id,
                         load_attr,
                         load_vth,
                         load_vl,
                         load_vh,
                         load_derate);
    const double driver_vth_d = static_cast<double>(driver_vth);
    const double driver_vl_d = static_cast<double>(driver_vl);
    const double driver_vh_d = static_cast<double>(driver_vh);
    const double driver_derate_d = static_cast<double>(driver_derate);
    const double driver_delta = driver_vh_d - driver_vl_d;
    const double load_delta = load_vh - load_vl;
    if (!isfinite(driver_delta) || !isfinite(load_delta) ||
        !isfinite(driver_derate_d) || !isfinite(load_derate) ||
        driver_delta <= 0.0 || load_delta <= 0.0 ||
        driver_derate_d <= 0.0 || load_derate <= 0.0) {
        return;
    }
    const double delay_delta = load_slew * ((load_vth - driver_vth_d) / driver_delta);
    wire_delay += ((load_attr & 1) == 0) ? delay_delta : -delay_delta;
    load_slew *= ((load_delta / load_derate) / (driver_delta / driver_derate_d));
}


__device__ void DmpModel::inputPortDelaySlew(int load_pin_id,
                                                     int load_attr,
                                                          double source_slew,
                                                          double elmore,
                                                          double& wire_delay,
                                                          double& load_slew) const {
    double load_vth, load_vl, load_vh, load_derate;
    loadPinThresholds(load_pin_id,
                         load_attr,
                         load_vth,
                         load_vl,
                         load_vh,
                         load_derate);
    if (!isfinite(source_slew) || !isfinite(elmore) || elmore < 0.0 ||
        !isfinite(load_vth) || !isfinite(load_vl) || !isfinite(load_vh) ||
        !isfinite(load_derate) || load_vth <= 0.0 || load_vth >= 1.0 ||
        load_vl <= 0.0 || load_vh >= 1.0 || load_vh <= load_vl ||
        load_derate <= 0.0) {
        wire_delay = nanf("");
        load_slew = nanf("");
        return;
    }

    wire_delay = -elmore * log(1.0 - load_vth);
    load_slew = source_slew + elmore * log((1.0 - load_vl) / (1.0 - load_vh)) / load_derate;

    const double driver_vth = dmpThresholdArrayValue(dmp_output_thresholds, load_attr, vth_);
    const double driver_vl = dmpThresholdArrayValue(dmp_slew_lower_thresholds, load_attr, vl_);
    const double driver_vh = dmpThresholdArrayValue(dmp_slew_upper_thresholds, load_attr, vh_);
    const double driver_derate = dmpThresholdArrayValue(dmp_slew_derates, load_attr, slew_derate_);
    thresholdAdjust(load_pin_id,
                           load_attr,
                           driver_vth,
                           driver_vl,
                           driver_vh,
                           driver_derate,
                           -1,
                           wire_delay,
                           load_slew);
}


static constexpr unsigned int DMP_LUT_META_SCALAR = 1u << 0;
static constexpr unsigned int DMP_LUT_META_VAR0_IS_SLEW = 1u << 1;
static constexpr unsigned int DMP_LUT_META_VALID = 1u << 31;


__device__ __forceinline__ int dmpLowerBound7(const float* arr, float val) {
    if (val <= arr[3]) {
        if (val <= arr[1]) {
            return (val <= arr[0]) ? 0 : 1;
        }
        return (val <= arr[2]) ? 2 : 3;
    }
    if (val <= arr[5]) {
        return (val <= arr[4]) ? 4 : 5;
    }
    return 6;
}

__device__ __forceinline__ int dmpGateLowerBound(const float* arr, int size, float val) {
    if (size == 7) {
        return dmpLowerBound7(arr, val);
    }
    int l = 0;
    int r = size - 1;
    while (l < r) {
        const int m = (l + r) >> 1;
        if (arr[m] < val) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

__device__ DmpGateLutMeta GPULutAllocator::makeGateLutMeta(int lut_id) {
    DmpGateLutMeta meta{};
    meta.dims = 0;
    meta.flags = 0;
    meta.x_offset = 0;
    meta.y_offset = 0;
    meta.table_offset = 0;
    if (lut_id < 0 || !lut_allocated(lut_id)) {
        return meta;
    }
    const int num_x = d_num_x[lut_id];
    const int num_y = d_num_y[lut_id];
    const int num_table = d_num_table[lut_id];
    const int var0 = d_lut_template_var[lut_id * 2];
    if (num_x < 1 || num_y < 1 || num_x > 65535 || num_y > 65535 ||
        num_table < 1 || (var0 != 0 && var0 != 1)) {
        return meta;
    }
    meta.dims = static_cast<unsigned int>(num_x) |
                (static_cast<unsigned int>(num_y) << 16);
    meta.flags = DMP_LUT_META_VALID |
                 (num_table == 1 ? DMP_LUT_META_SCALAR : 0u) |
                 (var0 == 1 ? DMP_LUT_META_VAR0_IS_SLEW : 0u);
    meta.x_offset = static_cast<int>(d_x_offset[lut_id]);
    meta.y_offset = static_cast<int>(d_y_offset[lut_id]);
    meta.table_offset = static_cast<int>(d_table_offset[lut_id]);
    return meta;
}

__device__ inline float GPULutAllocator::gateLutWithMeta(const DmpGateLutMeta& meta,
                                                                  float input_slew,
                                                                  float load) {
    if (!meta.valid()) {
        return nanf("");
    }
    float x = 0.0f;
    float y = 0.0f;
    if ((meta.flags & DMP_LUT_META_VAR0_IS_SLEW) != 0u) {
        x = input_slew;
        y = load;
    } else {
        x = load;
        y = input_slew;
    }
    if ((meta.flags & DMP_LUT_META_SCALAR) != 0u) {
        return d_table_array[meta.table_offset];
    }

    const int num_x = meta.numX();
    const int num_y = meta.numY();
    const float* x_array = d_x_array + meta.x_offset;
    const float* y_array = d_y_array + meta.y_offset;
    const float* table_array = d_table_array + meta.table_offset;
    int x_idx1 = dmpGateLowerBound(x_array, num_x, x);
    int y_idx1 = dmpGateLowerBound(y_array, num_y, y);
    x_idx1 = max(1, min(num_x - 1, x_idx1));
    y_idx1 = max(1, min(num_y - 1, y_idx1));
    const int x_idx0 = x_idx1 - 1;
    const int y_idx0 = y_idx1 - 1;
    if (num_x == 1) {
        x_idx1 = 0;
    }
    if (num_y == 1) {
        y_idx1 = 0;
    }

    const int table_stride = num_y;
    const float x0 = x_array[x_idx0];
    const float x1 = x_array[x_idx1];
    const float y0 = y_array[y_idx0];
    const float y1 = y_array[y_idx1];
    const float v00 = table_array[x_idx0 * table_stride + y_idx0];
    const float v10 = table_array[x_idx1 * table_stride + y_idx0];
    const float v01 = table_array[x_idx0 * table_stride + y_idx1];
    const float v11 = table_array[x_idx1 * table_stride + y_idx1];
    const float numeric0 = interpolate<float>(x0, x1, v00, v10, x);
    const float numeric1 = interpolate<float>(x0, x1, v01, v11, x);
    return interpolate<float>(y0, y1, numeric0, numeric1, y);
}

__device__ inline float GPULutAllocator::queryGateLutNoTransitionCheck(int timing_id,
                                                                                int output_rf,
                                                                                float input_slew,
                                                                                float load,
                                                                                int type) {
    const int lut_id = num_luts_in_timing * timing_id + output_rf + type * 2;
    const DmpGateLutMeta meta = makeGateLutMeta(lut_id);
    return gateLutWithMeta(meta, input_slew, load);
}

__device__ void DmpModel::gateCapDelaySlew(
    int timing_id,
                                                          int input_rf,
                                                          int output_rf,
                                                          float input_slew,
                                                          double load_cap,
                                                          double& delay,
                                                          double& slew) {
    delay = nanf("");
    slew = nanf("");
    if (d_allocator == nullptr ||
        timing_id < 0 || input_rf < 0 || output_rf < 0 ||
        !isfinite(input_slew) || !isfinite(load_cap) || load_cap < 0.0) {
        return;
    }
    if (!d_allocator->is_transition_defined(timing_id, input_rf, output_rf)) {
        return;
    }
    const float load = static_cast<float>(load_cap);
    delay = d_allocator->queryGateLutNoTransitionCheck(
                                             timing_id,
                                             output_rf,
                                             input_slew,
                                             load,
                                             0);
    slew = d_allocator->queryGateLutNoTransitionCheck(
                                            timing_id,
                                            output_rf,
                                            input_slew,
                                            load,
                                            1);
}

__device__ inline DmpGateArcMeta DmpModel::makeGateArcMeta(int pin_id,
                                                                     int attr,
                                                                     int timing_id,
                                                                     int input_rf,
                                                                     int output_rf,
                                                                     float input_slew,
                                                                     DmpDriverThresholds& thresholds) {
    DmpGateArcMeta gate_arc_meta{};
    gate_arc_meta.allocator = d_allocator;
    gate_arc_meta.delay_lut = {};
    gate_arc_meta.slew_lut = {};
    gate_arc_meta.input_slew = input_slew;
    gate_arc_meta.valid = false;

    thresholds = {};
    const int library_id = pinLibraryId(pin_id);
    double driver_vth = nan("");
    double driver_vl = nan("");
    double driver_vh = nan("");
    double driver_derate = nan("");
    driverLibraryThresholds(library_id,
                            attr,
                            driver_vth,
                            driver_vl,
                            driver_vh,
                            driver_derate);
    thresholds.set(driver_vth, driver_vl, driver_vh, driver_derate, library_id);

    if (gate_arc_meta.allocator == nullptr ||
        timing_id < 0 || input_rf < 0 || output_rf < 0 ||
        !isfinite(input_slew) || !thresholds.valid() ||
        !gate_arc_meta.allocator->is_transition_defined(timing_id, input_rf, output_rf)) {
        return gate_arc_meta;
    }
    const int base_lut_id = gate_arc_meta.allocator->num_luts_in_timing * timing_id + output_rf;
    gate_arc_meta.delay_lut = gate_arc_meta.allocator->makeGateLutMeta(base_lut_id);
    gate_arc_meta.slew_lut = gate_arc_meta.allocator->makeGateLutMeta(base_lut_id + 2);
    gate_arc_meta.valid = gate_arc_meta.hasValidLuts();
    return gate_arc_meta;
}

__device__ void DmpGateArcMeta::capDelaySlew(double load_cap,
                                                             double& delay,
                                                             double& slew) const {
    delay = nanf("");
    slew = nanf("");
    if (!valid || !isfinite(load_cap) || load_cap < 0.0) {
        return;
    }
    const float load = static_cast<float>(load_cap);
    const float delay_f = allocator->gateLutWithMeta(delay_lut, input_slew, load);
    const float slew_f = allocator->gateLutWithMeta(slew_lut, input_slew, load);
    delay = static_cast<double>(delay_f);
    slew = static_cast<double>(slew_f);
}

__device__ void DmpGateArcMeta::capDelaySlew(float load,
                                                             float& delay,
                                                             float& slew) const {
    delay = nanf("");
    slew = nanf("");
    if (!valid || !isfinite(load) || load < 0.0f) {
        return;
    }
    delay = allocator->gateLutWithMeta(delay_lut, input_slew, load);
    slew = allocator->gateLutWithMeta(slew_lut, input_slew, load);
}

__device__ void DmpGateArcMeta::gateDelays(const DmpDriverThresholds& thresholds,
                                                           double ceff,
                                                           double& t_vth,
                                               double& t_vl,
                                               double& slew) const {
    t_vth = nanf("");
    t_vl = nanf("");
    slew = nanf("");
    if (!thresholds.valid()) {
        return;
    }
    double table_slew = nanf("");
    capDelaySlew(ceff, t_vth, table_slew);
    if (!isfinite(t_vth) || !isfinite(table_slew)) {
        return;
    }
    const double driver_delta = static_cast<double>(thresholds.vh - thresholds.vl);
    slew = table_slew * static_cast<double>(thresholds.derate);
    t_vl = t_vth - slew * static_cast<double>(thresholds.vth - thresholds.vl) / driver_delta;
}

__device__ void DmpGateArcMeta::gateDelays(const DmpDriverThresholds& thresholds,
                                                           float ceff,
                                                           float& t_vth,
                                               float& t_vl,
                                               float& slew) const {
    t_vth = nanf("");
    t_vl = nanf("");
    slew = nanf("");
    if (!thresholds.valid()) {
        return;
    }
    float table_slew = nanf("");
    capDelaySlew(ceff, t_vth, table_slew);
    if (!isfinite(t_vth) || !isfinite(table_slew)) {
        return;
    }
    slew = table_slew * thresholds.derate;
    t_vl = t_vth - slew * (thresholds.vth - thresholds.vl) / (thresholds.vh - thresholds.vl);
}

__device__ bool DmpModel::isIdealClockTimingArc(int timing_id,
                                                                int from_pin_id) const {
    if (timing_id < 0) {
        return false;
    }
    if (hasPinFlag(from_pin_id, DMP_PIN_IDEAL_CLK)) {
        return true;
    }
    // ISPD25 SDC clocks are ideal unless explicitly propagated. Treat clock
    // pins with a parsed SDC waveform as ideal even if the packed flag is not
    // present after DMP RC setup.
    return hasPinFlag(from_pin_id, DMP_PIN_CLK) && clockIdValid(pinClockId(from_pin_id));
}

__device__ __forceinline__ bool DmpModel::clockIdValid(uint16_t clock_id) const {
    return clock_id != DMP_INVALID_CLOCK_ID &&
           clock_id < static_cast<uint16_t>(clock_count);
}

__device__ __forceinline__ uint16_t DmpModel::pinClockId(int pin_id) const {
    if (pin_clock_ids == nullptr || pin_id < 0 || pin_id >= num_pins) {
        return DMP_INVALID_CLOCK_ID;
    }
    return pin_clock_ids[pin_id];
}

__device__ __forceinline__ uint16_t DmpModel::testClockId(int test_id) const {
    if (test_clock_ids == nullptr || test_id < 0 || test_id >= num_tests) {
        return DMP_INVALID_CLOCK_ID;
    }
    return test_clock_ids[test_id];
}

__device__ __forceinline__ float DmpModel::clockPeriodForTest(int test_id) const {
    if (clock_periods != nullptr) {
        const uint16_t clock_id = testClockId(test_id);
        if (clockIdValid(clock_id)) {
            const float period = clock_periods[clock_id];
            if (isfinite(period) && period > 0.0f) return period;
        }
    }
    return clock_period;
}

__device__ __forceinline__ float DmpModel::pinClockEdge(int pin_id, bool fall) const {
    const uint16_t clock_id = pinClockId(pin_id);
    const float override =
        (pin_clock_latency_overrides != nullptr && pin_id >= 0 && pin_id < num_pins)
            ? pin_clock_latency_overrides[pin_id]
            : nanf("");
    if (isfinite(override)) {
        if (clockIdValid(clock_id)) {
            const float* waveform_edges = fall ? clock_waveform_fall_edges
                                               : clock_waveform_rise_edges;
            if (waveform_edges != nullptr) {
                const float waveform = waveform_edges[clock_id];
                if (isfinite(waveform)) return waveform + override;
            }
        }
        return override;
    }
    if (clockIdValid(clock_id)) {
        const float* edges = fall ? clock_fall_edges : clock_rise_edges;
        if (edges != nullptr) {
            const float edge = edges[clock_id];
            if (isfinite(edge)) return edge;
        }
    }
    return nanf("");
}

__device__ float DmpModel::idealClockEdgeTime(int timing_id,
                                                              int from_pin_id) const {
    const bool latch_clock_arc = timing_id >= 0 &&
                                 d_allocator->timing_is_latch_clock_arc(timing_id);
    const bool falling_triggered = timing_id >= 0 &&
                                   d_allocator->timing_is_falling_edge_triggered(timing_id) &&
                                   !d_allocator->timing_is_rising_edge_triggered(timing_id);
    const bool use_fall_edge = latch_clock_arc ? !falling_triggered : falling_triggered;
    if (from_pin_id >= 0) {
        const float clock_edge = pinClockEdge(from_pin_id, use_fall_edge);
        if (isfinite(clock_edge)) {
            return clock_edge;
        }
        if (pinAt != nullptr) {
            const int rf = use_fall_edge ? 1 : 0;
            const int early_slot = from_pin_id * NUM_ATTR + rf;
            const float early_edge = pinAt[early_slot];
            if (isfinite(early_edge)) {
                return early_edge;
            }
            const float late_edge = pinAt[early_slot + 2];
            if (isfinite(late_edge)) {
                return late_edge;
            }
        }
    }
    return nanf("");
}

__device__ float DmpModel::idealClockSlew(int from_pin_id,
                                          int attr) const {
    if (clock_slews != nullptr && from_pin_id >= 0 && attr >= 0 && attr < NUM_ATTR) {
        const uint16_t clock_id = pinClockId(from_pin_id);
        if (clockIdValid(clock_id)) {
            const float slew = clock_slews[static_cast<int>(clock_id) * NUM_ATTR + attr];
            if (isfinite(slew)) {
                return slew;
            }
        }
    }
    return 0.0f;
}

__device__ __forceinline__ float DmpModel::setupUncertaintyForTest(int test_id) const {
    if (clock_setup_uncertainties != nullptr) {
        const uint16_t clock_id = testClockId(test_id);
        if (clockIdValid(clock_id)) {
            const float uncertainty = clock_setup_uncertainties[clock_id];
            return isfinite(uncertainty) ? uncertainty : 0.0f;
        }
    }
    return 0.0f;
}

__device__ __forceinline__ float DmpModel::holdUncertaintyForTest(int test_id) const {
    if (clock_hold_uncertainties != nullptr) {
        const uint16_t clock_id = testClockId(test_id);
        if (clockIdValid(clock_id)) {
            const float uncertainty = clock_hold_uncertainties[clock_id];
            return isfinite(uncertainty) ? uncertainty : 0.0f;
        }
    }
    return 0.0f;
}

__device__ inline void DmpModel::propagateTest(int test_id,
                                                                 int from_pin_id,
                                                                 int attr,
                                                                 int el,
                                                                 int rf,
                                                                 int timing_id,
                                                                 int to_slot){
    if(test_id == -1)return ;
    if (attr < NUM_ATTR) {
        if ((timing_id == -1) || (isnan(pinSlew[to_slot]))) return;
        int fel = el ^ 1;  // clock -> data. clock late -> data early (hold)
        int frf = d_allocator->timing_is_rising_edge_triggered(timing_id) ? 0 : 1;
        if (frf && !d_allocator->timing_is_falling_edge_triggered(timing_id)) {
            return;
        }
        const int fel_rf = (fel << 1) + frf;
        const int from_related_slot = from_pin_id * NUM_ATTR + fel_rf;
        const bool ideal_clock_arc = isIdealClockTimingArc(timing_id, from_pin_id);
        if (!ideal_clock_arc &&
            (isnan(pinAt[from_related_slot]) ||
             isnan(pinSlew[from_related_slot]))) return;

        float related_at = ideal_clock_arc
                               ? idealClockEdgeTime(timing_id, from_pin_id)
                               : pinAt[from_related_slot];
        if (ideal_clock_arc && isnan(related_at)) {
            related_at = pinAt[from_related_slot];
        }
        if (el == 0) {
            testRelatedAT[test_id * NUM_ATTR + attr] = related_at;
        } else {
            const float test_period = clockPeriodForTest(test_id);
            testRelatedAT[test_id * NUM_ATTR + attr] = related_at + (frf ? 0.5f * test_period : test_period);  // setup is checked at next cycle (first cycle for triggering 1st FF)
        }

        float sr = pinSlew[from_related_slot];
        if (ideal_clock_arc) {
            sr = idealClockSlew(from_pin_id, fel_rf);
        }
        float sc = pinSlew[to_slot];
            testConstraint[test_id * NUM_ATTR + attr] = d_allocator->query(timing_id, frf, rf, sr, sc, 2);
        if (!isnan(testConstraint[test_id * NUM_ATTR + attr]) && !isnan(testRelatedAT[test_id * NUM_ATTR + attr])) {
            if (el == 0) {
                const float hold_uncertainty = holdUncertaintyForTest(test_id);
                pinRat[to_slot] = testRelatedAT[test_id * NUM_ATTR + attr] + testConstraint[test_id * NUM_ATTR + attr] + hold_uncertainty;  // hold clocks needs data stay late, rat = at_clk + T_hold
            } else {
                const float setup_uncertainty = setupUncertaintyForTest(test_id);
                pinRat[to_slot] = testRelatedAT[test_id * NUM_ATTR + attr] - testConstraint[test_id * NUM_ATTR + attr] - setup_uncertainty;  // setup clock needs data come early, rat = at_clk - T_setup
            }
            testRAT[test_id * NUM_ATTR + attr] = pinRat[to_slot];
        }
    }
}
__device__ inline void DmpModel::propagatePinTests(int to_pin_idx){
    if (clock_period <= 0) {
        return;
    }
    int to_pin = level_list[to_pin_idx];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int attr = idx & 0b111;
    const int el = attr >> 1;
    const int rf = attr & 1;
    for (index_type i = pin_backward_arc_list_end[to_pin]; i < pin_backward_arc_list_end[to_pin + 1]; i++) {
        const int arc_id = pin_backward_arc_list[i];
        const int to_pin_id = timing_arc_to_pin_id[arc_id];
        const int to_slot = to_pin_id * NUM_ATTR + attr;
        const int timing_id = timing_arc_id_map[arc_id * 2 + el];
        propagateTest(arc_id2test_id[arc_id],
                      timing_arc_from_pin_id[arc_id],
                      attr,
                      el,
                      rf,
                      timing_id,
                      to_slot);
    }
}

__global__ void dmpTestKernel(DmpModel* dmp_db, int level_start_offset, int num_pins_level){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_id = idx >> 3;
    if(pin_id < num_pins_level){
        dmp_db -> propagatePinTests(level_start_offset + pin_id);
    }
}

__global__ void dmpResetForwardTargetsKernel(DmpModel* dmp_db) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_slots = dmp_db->dmp_pin_slot_count;
    if (idx < pin_slots) {
        const int pin_id = idx / NUM_ATTR;
        const bool source_clock_pin =
            (dmp_db->hasPinFlag(pin_id, DMP_PIN_CLK) ||
             dmp_db->hasPinFlag(pin_id, DMP_PIN_IDEAL_CLK)) &&
            dmp_db->pin_backward_arc_list_end != nullptr &&
            pin_id + 1 <= dmp_db->num_pins &&
            dmp_db->pin_backward_arc_list_end[pin_id] ==
                dmp_db->pin_backward_arc_list_end[pin_id + 1];
        // Preserve real source slews. For set_driving_cell slots, pinSlew holds
        // the virtual cell input transition and prefix arrays hold its arc tag.
        const bool preserve_source_slew =
            dmp_db->hasPinFlag(pin_id, DMP_PIN_PRIMARY_INPUT) ||
            source_clock_pin ||
            (dmp_db->at_prefix_attr != nullptr &&
             dmp_db->at_prefix_attr[idx] == DMP_DRIVING_CELL_PREFIX_ATTR);
        if (!preserve_source_slew) {
            reinterpret_cast<unsigned int*>(dmp_db->pinSlew)[idx] = 0u;
        }
    }

    const int arc_idx = idx - pin_slots;
    const int arc_slots = dmp_db->num_arcs * 2 * NUM_ATTR;
    if (arc_idx >= 0 && arc_idx < arc_slots) {
        const int arc_id = arc_idx / (2 * NUM_ATTR);
        if (dmp_db->arc_types != nullptr && dmp_db->arc_types[arc_id] == 1) {
            dmp_db->arcDelay[arc_idx] = nanf("");
        } else {
            reinterpret_cast<unsigned int*>(dmp_db->arcDelay)[arc_idx] = 0u;
        }
    }
}

__global__ void dmpDirectNetKernel(DmpModel* dmp_db,
                                             const index_type* level_arc_list,
                                             int num_level_arcs){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_pos = idx >> 3;
    if (arc_pos < num_level_arcs) {
        const int arc_id = level_arc_list[arc_pos];
        if (dmp_db->arc_types[arc_id] == 0) {
            const int attr = idx & 0b11;
            dmp_db->propagateLoadSlewDelay(arc_id, attr);
        }
    }
}

__global__ void dmpPinWinnerKernel(DmpModel* dmp_db,
                                       int level_start_offset,
                                       int num_pins_level) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_pos = idx >> 2;
    if (pin_pos >= num_pins_level) {
        return;
    }
    const int attr = idx & 0b11;
    const int pin_id = dmp_db->level_list[level_start_offset + pin_pos];
    const int to_slot = pin_id * NUM_ATTR + attr;
    const bool pick_max = (attr >> 1) != 0;

    const unsigned long long packed_at = dmp_db->pin_at_winner[to_slot];
    const bool has_at_winner = packed_at != 0ULL;
    if (packed_at != 0ULL) {
        const unsigned int cmp_key = static_cast<unsigned int>(packed_at >> 32);
        const unsigned int payload = static_cast<unsigned int>(packed_at & 0xffffffffULL);
        const int arc_id = static_cast<int>(payload >> 2);
        const int from_attr = static_cast<int>(payload & 0x3);
        const float at = dmpDecodeWinnerFloat(cmp_key, pick_max);
        if (arc_id >= 0 && arc_id < dmp_db->num_arcs && isfinite(at)) {
            dmp_db->pinAt[to_slot] = at;
            dmp_db->at_prefix_pin[to_slot] = dmp_db->timing_arc_from_pin_id[arc_id];
            dmp_db->at_prefix_arc[to_slot] = arc_id;
            dmp_db->at_prefix_attr[to_slot] = from_attr;
        }
        dmp_db->pin_at_winner[to_slot] = 0ULL;
    }

    // The forward loop normally skips root/source pins. Keep this guard for
    // any source-tagged slot that reaches finalize: source pinSlew is a raw
    // slew/transition value, while non-source pinSlew holds an encoded winner key.
    const bool source_clock_pin =
        !has_at_winner &&
        (dmp_db->hasPinFlag(pin_id, DMP_PIN_CLK) ||
         dmp_db->hasPinFlag(pin_id, DMP_PIN_IDEAL_CLK)) &&
        dmp_db->pin_backward_arc_list_end != nullptr &&
        pin_id + 1 <= dmp_db->num_pins &&
        dmp_db->pin_backward_arc_list_end[pin_id] ==
            dmp_db->pin_backward_arc_list_end[pin_id + 1];
    const bool preserve_source_slew =
        dmp_db->hasPinFlag(pin_id, DMP_PIN_PRIMARY_INPUT) ||
        source_clock_pin ||
        (dmp_db->at_prefix_attr != nullptr &&
         dmp_db->at_prefix_attr[to_slot] == DMP_DRIVING_CELL_PREFIX_ATTR);
    if (!preserve_source_slew) {
        const unsigned int slew_key = __float_as_uint(dmp_db->pinSlew[to_slot]);
        if (slew_key != 0u) {
            const float slew = dmpDecodeWinnerFloat(slew_key, pick_max);
            dmp_db->pinSlew[to_slot] = isfinite(slew) ? slew : nanf("");
        } else {
            dmp_db->pinSlew[to_slot] = nanf("");
        }
    }
}

__global__ void dmpNetWinnerKernel(DmpModel* dmp_db,
                                                          const index_type* level_arc_list,
                                                          int num_level_arcs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_pos = idx >> 2;
    if (arc_pos >= num_level_arcs) {
        return;
    }
    const int attr = idx & 0b11;
    const int arc_id = level_arc_list[arc_pos];
    const int delay_idx = (attr << 1) + (attr & 1);
    const int delay_slot = arc_id * 2 * NUM_ATTR + delay_idx;
    const bool pick_max = (attr >> 1) != 0;

    const unsigned int delay_key = __float_as_uint(dmp_db->arcDelay[delay_slot]);
    if (delay_key == 0u) {
        dmp_db->arcDelay[delay_slot] = nanf("");
        return;
    }
    const float decoded_delay = dmpDecodeWinnerFloat(delay_key, pick_max);
    if (!isfinite(decoded_delay)) {
        dmp_db->arcDelay[delay_slot] = nanf("");
        return;
    }
    dmp_db->arcDelay[delay_slot] = decoded_delay;

    const int from_pin_id = dmp_db->timing_arc_from_pin_id[arc_id];
    const int to_pin_id = dmp_db->timing_arc_to_pin_id[arc_id];
    const float from_at = dmp_db->pinAt[from_pin_id * NUM_ATTR + attr];
    if (isnan(from_at)) {
        return;
    }
    const float at = from_at + decoded_delay;
    dmp_db->updateAtWinner(to_pin_id * NUM_ATTR + attr,
                           at,
                           arc_id,
                           attr);
}

} // namespace gt
