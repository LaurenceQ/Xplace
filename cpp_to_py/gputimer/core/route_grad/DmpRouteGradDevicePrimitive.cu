#include "gputimer/core/route_grad/DmpRouteGradDevice.cuh"
#include "gputimer/core/route_grad/DmpRouteGradDeviceInternal.cuh"

#include "gputimer/core/DmpGateModel.cuh"
#include "gputimer/core/DmpModel.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace gt {

// Primitive threshold/LUT helpers and forward candidate reconstruction.
// Threshold, LUT, and root-driver resistance helpers for primitive derivatives.
// Timing-library and threshold helpers. This block mirrors the forward DMP
// table lookup/threshold-adjust behavior closely enough for branch-local
// derivatives to match the current forward branch.
__device__ double RouteGradNetPrimitiveReverse::thresholdArrayValue(const float* values,
                                                                    int attr,
                                                                    double fallback) const
{
    if (values == nullptr || attr < 0 || attr >= NUM_ATTR) {
        return fallback;
    }
    return routeGradFinitePositiveOr(static_cast<double>(values[attr]), fallback);
}

__device__ double RouteGradNetPrimitiveReverse::libraryThresholdArrayValue(
    const float* values,
    int library_idx,
    double fallback) const
{
    if (values == nullptr || library_idx < 0) {
        return fallback;
    }
    return routeGradFinitePositiveOr(static_cast<double>(values[library_idx]), fallback);
}

__device__ int RouteGradNetPrimitiveReverse::timingLibraryId(int timing_id) const
{
    return (model != nullptr && model->dmp_timing_library_ids != nullptr && timing_id >= 0)
               ? model->dmp_timing_library_ids[timing_id]
               : -1;
}

__device__ int RouteGradNetPrimitiveReverse::pinLibraryId(int pin_id) const
{
    return (model != nullptr && model->dmp_pin_library_ids != nullptr && pin_id >= 0)
               ? model->dmp_pin_library_ids[pin_id]
               : -1;
}

__device__ void RouteGradNetPrimitiveReverse::loadPinThresholds(
    int pin_id,
    int attr,
    double& vth,
    double& vl,
    double& vh,
    double& slew_derate) const
{
    const int library_id = pinLibraryId(pin_id);
    const int library_idx = library_id >= 0 ? library_id * NUM_ATTR + attr : -1;
    vth = libraryThresholdArrayValue(model ? model->dmp_library_input_thresholds : nullptr,
                                     library_idx,
                                     thresholdArrayValue(model ? model->dmp_input_thresholds : nullptr,
                                                         attr,
                                                         model ? model->vth_ : 0.5));
    vl = libraryThresholdArrayValue(model ? model->dmp_library_slew_lower_thresholds : nullptr,
                                    library_idx,
                                    thresholdArrayValue(model ? model->dmp_slew_lower_thresholds : nullptr,
                                                        attr,
                                                        model ? model->vl_ : 0.2));
    vh = libraryThresholdArrayValue(model ? model->dmp_library_slew_upper_thresholds : nullptr,
                                    library_idx,
                                    thresholdArrayValue(model ? model->dmp_slew_upper_thresholds : nullptr,
                                                        attr,
                                                        model ? model->vh_ : 0.8));
    slew_derate = libraryThresholdArrayValue(model ? model->dmp_library_slew_derates : nullptr,
                                             library_idx,
                                             thresholdArrayValue(model ? model->dmp_slew_derates : nullptr,
                                                                 attr,
                                                                 model ? model->slew_derate_ : 1.0));
    if (vh <= vl && model != nullptr) {
        vl = model->vl_;
        vh = model->vh_;
    }
}

__device__ void RouteGradNetPrimitiveReverse::driverLibraryThresholds(
    int library_id,
    int attr,
    double& vth,
    double& vl,
    double& vh,
    double& slew_derate) const
{
    const int library_idx = library_id >= 0 ? library_id * NUM_ATTR + attr : -1;
    vth = libraryThresholdArrayValue(model ? model->dmp_library_output_thresholds : nullptr,
                                     library_idx,
                                     thresholdArrayValue(model ? model->dmp_output_thresholds : nullptr,
                                                         attr,
                                                         model ? model->vth_ : 0.5));
    vl = libraryThresholdArrayValue(model ? model->dmp_library_slew_lower_thresholds : nullptr,
                                    library_idx,
                                    thresholdArrayValue(model ? model->dmp_slew_lower_thresholds : nullptr,
                                                        attr,
                                                        model ? model->vl_ : 0.2));
    vh = libraryThresholdArrayValue(model ? model->dmp_library_slew_upper_thresholds : nullptr,
                                    library_idx,
                                    thresholdArrayValue(model ? model->dmp_slew_upper_thresholds : nullptr,
                                                        attr,
                                                        model ? model->vh_ : 0.8));
    slew_derate = libraryThresholdArrayValue(model ? model->dmp_library_slew_derates : nullptr,
                                             library_idx,
                                             thresholdArrayValue(model ? model->dmp_slew_derates : nullptr,
                                                                 attr,
                                                                 model ? model->slew_derate_ : 1.0));
    if (vh <= vl && model != nullptr) {
        vl = model->vl_;
        vh = model->vh_;
    }
}

__device__ DmpGateArcMeta RouteGradNetPrimitiveReverse::makeGateArcMetaForTiming(
    int timing_id,
    int input_rf,
    int to_attr,
    float input_slew,
    DmpDriverThresholds& thresholds) const
{
    DmpGateArcMeta gate_arc_meta{};
    gate_arc_meta.allocator = model ? model->d_allocator : nullptr;
    gate_arc_meta.delay_lut = {};
    gate_arc_meta.slew_lut = {};
    gate_arc_meta.input_slew = input_slew;
    gate_arc_meta.valid = false;

    thresholds = {};
    const int library_id = timingLibraryId(timing_id);
    double driver_vth = nanf("");
    double driver_vl = nanf("");
    double driver_vh = nanf("");
    double driver_derate = nanf("");
    driverLibraryThresholds(library_id,
                            to_attr,
                            driver_vth,
                            driver_vl,
                            driver_vh,
                            driver_derate);
    thresholds.set(driver_vth, driver_vl, driver_vh, driver_derate, library_id);

    const int output_rf = to_attr & 1;
    if (gate_arc_meta.allocator == nullptr || timing_id < 0 || input_rf < 0 ||
        to_attr < 0 || !isfinite(input_slew) || !thresholds.valid() ||
        !gate_arc_meta.allocator->is_transition_defined(timing_id, input_rf, output_rf)) {
        return gate_arc_meta;
    }
    const int base_lut_id = gate_arc_meta.allocator->num_luts_in_timing * timing_id + output_rf;
    gate_arc_meta.delay_lut = gate_arc_meta.allocator->makeGateLutMeta(base_lut_id);
    gate_arc_meta.slew_lut = gate_arc_meta.allocator->makeGateLutMeta(base_lut_id + 2);
    gate_arc_meta.valid = gate_arc_meta.hasValidLuts();
    return gate_arc_meta;
}

__device__ void RouteGradNetPrimitiveReverse::thresholdAdjustedSlopes(
    int load_pin_id,
    int load_attr,
    float driver_vth,
    float driver_vl,
    float driver_vh,
    float driver_derate,
    int driver_library_id,
    double raw_delay_slope,
    double raw_slew_slope,
    double& delay_slope,
    double& slew_slope) const
{
    delay_slope = raw_delay_slope;
    slew_slope = raw_slew_slope;
    if (model == nullptr || !isfinite(delay_slope) || !isfinite(slew_slope)) {
        return;
    }

    const int load_library_id = pinLibraryId(load_pin_id);
    if (driver_library_id >= 0 && driver_library_id == load_library_id) {
        return;
    }

    double load_vth = nanf("");
    double load_vl = nanf("");
    double load_vh = nanf("");
    double load_derate = nanf("");
    loadPinThresholds(load_pin_id, load_attr, load_vth, load_vl, load_vh, load_derate);

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

    const double beta = (load_vth - driver_vth_d) / driver_delta;
    const double sign = ((load_attr & 1) == 0) ? 1.0 : -1.0;
    const double gamma = (load_delta / load_derate) / (driver_delta / driver_derate_d);
    delay_slope += sign * beta * raw_slew_slope;
    slew_slope *= gamma;
}

__device__ void RouteGradNetPrimitiveReverse::thresholdAdjustedWaveSlopes(
    int load_pin_id,
    int load_attr,
    const DmpDriverThresholds& thresholds,
    const RouteGradWaveParamSlopes& raw_delay,
    const RouteGradWaveParamSlopes& raw_slew,
    RouteGradDelaySlewWaveSlopes& final_slopes) const
{
    final_slopes.delay = raw_delay;
    final_slopes.slew = raw_slew;
    if (model == nullptr) {
        return;
    }

    const int load_library_id = pinLibraryId(load_pin_id);
    if (thresholds.library_id >= 0 && thresholds.library_id == load_library_id) {
        return;
    }

    double load_vth = nanf("");
    double load_vl = nanf("");
    double load_vh = nanf("");
    double load_derate = nanf("");
    loadPinThresholds(load_pin_id, load_attr, load_vth, load_vl, load_vh, load_derate);

    const double driver_vth = static_cast<double>(thresholds.vth);
    const double driver_vl = static_cast<double>(thresholds.vl);
    const double driver_vh = static_cast<double>(thresholds.vh);
    const double driver_derate = static_cast<double>(thresholds.derate);
    const double driver_delta = driver_vh - driver_vl;
    const double load_delta = load_vh - load_vl;
    if (!isfinite(driver_delta) || !isfinite(load_delta) ||
        !isfinite(driver_derate) || !isfinite(load_derate) ||
        driver_delta <= 0.0 || load_delta <= 0.0 ||
        driver_derate <= 0.0 || load_derate <= 0.0) {
        return;
    }

    const double beta = (load_vth - driver_vth) / driver_delta;
    const double sign = ((load_attr & 1) == 0) ? 1.0 : -1.0;
    const double gamma = (load_delta / load_derate) / (driver_delta / driver_derate);
#define ROUTE_GRAD_APPLY_WAVE_FIELD(field) \
    final_slopes.delay.field += sign * beta * raw_slew.field; \
    final_slopes.slew.field *= gamma
    ROUTE_GRAD_APPLY_WAVE_FIELD(t0);
    ROUTE_GRAD_APPLY_WAVE_FIELD(dt);
    ROUTE_GRAD_APPLY_WAVE_FIELD(k0);
    ROUTE_GRAD_APPLY_WAVE_FIELD(k1);
    ROUTE_GRAD_APPLY_WAVE_FIELD(k2);
    ROUTE_GRAD_APPLY_WAVE_FIELD(k3);
    ROUTE_GRAD_APPLY_WAVE_FIELD(k4);
    ROUTE_GRAD_APPLY_WAVE_FIELD(p1);
    ROUTE_GRAD_APPLY_WAVE_FIELD(p2);
#undef ROUTE_GRAD_APPLY_WAVE_FIELD
}

__device__ bool RouteGradNetPrimitiveReverse::gateLutValueSlopes(
    GPULutAllocator* allocator,
    const DmpGateLutMeta& meta,
    float input_slew,
    float load,
    RouteGradLutSlopes& slopes) const
{
    slopes = {};
    if (allocator == nullptr || !meta.valid() || !isfinite(input_slew) || !isfinite(load)) {
        return false;
    }
    float x = 0.0f;
    float y = 0.0f;
    const bool var0_is_slew = (meta.flags & kRouteGradLutMetaVar0IsSlew) != 0u;
    if (var0_is_slew) {
        x = input_slew;
        y = load;
    } else {
        x = load;
        y = input_slew;
    }
    if ((meta.flags & kRouteGradLutMetaScalar) != 0u) {
        slopes.value = static_cast<double>(allocator->d_table_array[meta.table_offset]);
        return isfinite(slopes.value);
    }

    const int num_x = meta.numX();
    const int num_y = meta.numY();
    if (num_x < 1 || num_y < 1) {
        return false;
    }
    const float* x_array = allocator->d_x_array + meta.x_offset;
    const float* y_array = allocator->d_y_array + meta.y_offset;
    const float* table_array = allocator->d_table_array + meta.table_offset;
    int x_idx1 = routeGradLowerBoundFloat(x_array, num_x, x);
    int y_idx1 = routeGradLowerBoundFloat(y_array, num_y, y);
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

    const int stride = num_y;
    const double x0 = static_cast<double>(x_array[x_idx0]);
    const double x1 = static_cast<double>(x_array[x_idx1]);
    const double y0 = static_cast<double>(y_array[y_idx0]);
    const double y1 = static_cast<double>(y_array[y_idx1]);
    const double v00 = static_cast<double>(table_array[x_idx0 * stride + y_idx0]);
    const double v10 = static_cast<double>(table_array[x_idx1 * stride + y_idx0]);
    const double v01 = static_cast<double>(table_array[x_idx0 * stride + y_idx1]);
    const double v11 = static_cast<double>(table_array[x_idx1 * stride + y_idx1]);
    const double xd = static_cast<double>(x);
    const double yd = static_cast<double>(y);

    const double numeric0 = (x0 == x1) ? v00 : v00 + (v10 - v00) * (xd - x0) / (x1 - x0);
    const double numeric1 = (x0 == x1) ? v01 : v01 + (v11 - v01) * (xd - x0) / (x1 - x0);
    const double d_numeric0_dx = (x0 == x1) ? 0.0 : (v10 - v00) / (x1 - x0);
    const double d_numeric1_dx = (x0 == x1) ? 0.0 : (v11 - v01) / (x1 - x0);
    const double y_weight = (y0 == y1) ? 0.0 : (yd - y0) / (y1 - y0);
    const double value = (y0 == y1) ? numeric0 : numeric0 + (numeric1 - numeric0) * y_weight;
    const double value_x = (y0 == y1)
                               ? d_numeric0_dx
                               : d_numeric0_dx + (d_numeric1_dx - d_numeric0_dx) * y_weight;
    const double value_y = (y0 == y1) ? 0.0 : (numeric1 - numeric0) / (y1 - y0);

    slopes.value = value;
    if (var0_is_slew) {
        slopes.input_slew_slope = value_x;
        slopes.load_slope = value_y;
    } else {
        slopes.load_slope = value_x;
        slopes.input_slew_slope = value_y;
    }
    return isfinite(slopes.value) && isfinite(slopes.input_slew_slope) &&
           isfinite(slopes.load_slope);
}

__device__ bool RouteGradNetPrimitiveReverse::gateArcCapDelaySlewSlopes(
    const DmpGateArcMeta& gate_arc_meta,
    double load_cap,
    RouteGradLutSlopes& delay,
    RouteGradLutSlopes& slew) const
{
    delay = {};
    slew = {};
    if (!gate_arc_meta.valid || gate_arc_meta.allocator == nullptr ||
        !isfinite(load_cap) || load_cap < 0.0) {
        return false;
    }
    const float load = static_cast<float>(load_cap);
    const bool delay_ok = gateLutValueSlopes(gate_arc_meta.allocator,
                                             gate_arc_meta.delay_lut,
                                             gate_arc_meta.input_slew,
                                             load,
                                             delay);
    const bool slew_ok = gateLutValueSlopes(gate_arc_meta.allocator,
                                            gate_arc_meta.slew_lut,
                                            gate_arc_meta.input_slew,
                                            load,
                                            slew);
    return delay_ok && slew_ok;
}

__device__ bool RouteGradNetPrimitiveReverse::estimateRdWithSlopes(
    const DmpGateArcMeta& gate_arc_meta,
    const DmpDriverThresholds& thresholds,
    double c1,
    double c2,
    double& rd,
    double& rd_c1,
    double& rd_c2,
    double& rd_input_slew) const
{
    rd = nanf("");
    rd_c1 = 0.0;
    rd_c2 = 0.0;
    rd_input_slew = 0.0;
    if (model == nullptr || !thresholds.valid()) {
        return false;
    }
    const double ctot = c1 + c2;
    const float cap1_f = static_cast<float>(ctot);
    const float cap_delta_f = static_cast<float>(1.0e-15 / static_cast<double>(model->cap_unit));
    const float cap2_f = cap1_f + cap_delta_f;
    if (!isfinite(ctot) || !isfinite(cap1_f) || !isfinite(cap_delta_f) ||
        !isfinite(cap2_f) || cap_delta_f <= 0.0f || cap2_f == cap1_f) {
        return false;
    }
    RouteGradLutSlopes d1;
    RouteGradLutSlopes s1;
    RouteGradLutSlopes d2;
    RouteGradLutSlopes s2;
    if (!gateArcCapDelaySlewSlopes(gate_arc_meta, static_cast<double>(cap1_f), d1, s1) ||
        !gateArcCapDelaySlewSlopes(gate_arc_meta, static_cast<double>(cap2_f), d2, s2)) {
        return false;
    }
    const float d1_f = static_cast<float>(d1.value);
    const float d2_f = static_cast<float>(d2.value);
    const double diff = static_cast<double>(d1_f - d2_f);
    const double denom = static_cast<double>(cap2_f - cap1_f);
    const double log_factor = -log(static_cast<double>(thresholds.vth));
    if (!isfinite(diff) || !isfinite(denom) || !isfinite(log_factor) ||
        diff == 0.0 || denom == 0.0) {
        return false;
    }
    const double sign = diff > 0.0 ? 1.0 : -1.0;
    const double factor = log_factor * sign / denom;
    rd = log_factor * fabs(diff) / denom;
    rd_c1 = factor * (d1.load_slope - d2.load_slope);
    rd_c2 = rd_c1;
    rd_input_slew = factor * (d1.input_slew_slope - d2.input_slew_slope);
    return isfinite(rd) && rd > 0.0 && isfinite(rd_c1) && isfinite(rd_input_slew);
}


// Forward reconstruction and finite-difference helpers for local primitive candidates.
// Forward reconstruction helpers for the local primitives. They rebuild the
// same direct-net and gate-net candidate values that DMP used, then the analytic
// slope code differentiates the selected branch.
__device__ bool RouteGradNetPrimitiveReverse::directNetCandidate(
    int arc_id,
    int attr,
    double& wire_delay,
    double& load_slew,
    double& delay_slope,
    double& slew_slope) const
{
    wire_delay = nanf("");
    load_slew = nanf("");
    delay_slope = nanf("");
    slew_slope = nanf("");
    if (model == nullptr || arc_id < 0 || arc_id >= model->num_arcs ||
        attr < 0 || attr >= NUM_ATTR) {
        return false;
    }

    const int from_pin_id = model->timing_arc_from_pin_id[arc_id];
    const int to_pin_id = model->timing_arc_to_pin_id[arc_id];
    if (from_pin_id < 0 || from_pin_id >= model->num_pins ||
        to_pin_id < 0 || to_pin_id >= model->num_pins) {
        return false;
    }
    const int from_slot = from_pin_id * NUM_ATTR + attr;
    const int to_slot = to_pin_id * NUM_ATTR + attr;
    if (from_slot < 0 || to_slot < 0 ||
        from_slot >= model->dmp_pin_slot_count ||
        to_slot >= model->dmp_pin_slot_count) {
        return false;
    }

    const double elmore = model->elmore_delay[to_slot];
    float source_slew = model->pinSlew[from_slot];
    const bool has_driving_cell =
        model->at_prefix_arc != nullptr &&
        model->at_prefix_attr != nullptr &&
        model->at_prefix_attr[from_slot] == DMP_DRIVING_CELL_PREFIX_ATTR;
    if (!isfinite(source_slew) || !isfinite(elmore)) {
        return false;
    }

    if (has_driving_cell) {
        const int driving_tag = model->at_prefix_arc[from_slot];
        const int timing_id = driving_tag >> 1;
        const int input_rf = driving_tag & 1;
        const int output_rf = attr & 1;
        DmpDriverWave wave;
        DmpDriverThresholds thresholds{};
        float gate_delay = nanf("");
        const DmpGateArcMeta gate_arc_meta =
            makeGateArcMetaForTiming(timing_id, input_rf, attr, source_slew, thresholds);
        if (!model->computeGateDriverWaveForSlot(gate_arc_meta,
                                                 thresholds,
                                                 from_slot,
                                                 wave,
                                                 gate_delay)) {
            return false;
        }
        if (!delaySlewSlopeForDriverWave(wave,
                                         thresholds,
                                         to_pin_id,
                                         attr,
                                         elmore,
                                         wire_delay,
                                         load_slew,
                                         delay_slope,
                                         slew_slope)) {
            return false;
        }
        double intrinsic_delay = nanf("");
        double intrinsic_slew = nanf("");
        model->gateCapDelaySlew(timing_id,
                                input_rf,
                                output_rf,
                                source_slew,
                                0.0,
                                intrinsic_delay,
                                intrinsic_slew);
        if (isfinite(intrinsic_delay) && isfinite(gate_delay)) {
            wire_delay += static_cast<double>(gate_delay) - intrinsic_delay;
        }
        return isfinite(wire_delay) && isfinite(load_slew) &&
               isfinite(delay_slope) && isfinite(slew_slope);
    }

    if (model->hasPinFlag(from_pin_id, DMP_PIN_PRIMARY_INPUT)) {
        model->inputPortDelaySlew(to_pin_id,
                                  attr,
                                  static_cast<double>(source_slew),
                                  elmore,
                                  wire_delay,
                                  load_slew);
        delay_slope = inputPortDelayElmoreSlope(to_pin_id, attr, slew_slope);
        return isfinite(wire_delay) && isfinite(load_slew) &&
               isfinite(delay_slope) && isfinite(slew_slope);
    }

    double driver_vth = nanf("");
    double driver_vl = nanf("");
    double driver_vh = nanf("");
    double driver_derate = nanf("");
    const int driver_library_id = pinLibraryId(from_pin_id);
    driverLibraryThresholds(driver_library_id,
                                   attr,
                                   driver_vth,
                                   driver_vl,
                                   driver_vh,
                                   driver_derate);
    wire_delay = elmore;
    load_slew = static_cast<double>(source_slew);
    model->thresholdAdjust(to_pin_id,
                           attr,
                           static_cast<float>(driver_vth),
                           static_cast<float>(driver_vl),
                           static_cast<float>(driver_vh),
                           static_cast<float>(driver_derate),
                           driver_library_id,
                           wire_delay,
                           load_slew);
    thresholdAdjustedSlopes(to_pin_id,
                            attr,
                            static_cast<float>(driver_vth),
                            static_cast<float>(driver_vl),
                            static_cast<float>(driver_vh),
                            static_cast<float>(driver_derate),
                            driver_library_id,
                            1.0,
                            0.0,
                            delay_slope,
                            slew_slope);
    return isfinite(wire_delay) && isfinite(load_slew) &&
           isfinite(delay_slope) && isfinite(slew_slope);
}

__device__ bool RouteGradNetPrimitiveReverse::computeDriverWaveForRc(
    const DmpGateArcMeta& gate_arc_meta,
    const DmpDriverThresholds& thresholds,
    double c1,
    double c2,
    double rpi,
    DmpDriverWave& driver_wave,
    float& gate_delay) const
{
    gate_delay = nanf("");
    driver_wave = {};
    driver_wave.alg = DMP_ALG_CAP;
    driver_wave.t0 = nanf("");
    driver_wave.dt = nanf("");
    driver_wave.vo_delay = nanf("");
    driver_wave.vo_slew = nanf("");
    driver_wave.vo_upper_time = nanf("");
    if (model == nullptr || !gate_arc_meta.valid || !thresholds.valid()) {
        return false;
    }

    DmpRcParams rc{};
    rc.c1 = c1;
    rc.c2 = c2;
    rc.rpi = rpi;
    rc.rd = nanf("");

    double table_delay = nanf("");
    double table_slew = nanf("");
    gate_arc_meta.capDelaySlew(rc.c1 + rc.c2, table_delay, table_slew);
    if (!isfinite(table_delay) || !isfinite(table_slew)) {
        return false;
    }

    const bool rc_can_model = isfinite(rc.c1) && isfinite(rc.c2) &&
                              isfinite(rc.rpi) && isfinite(rc.c1 + rc.c2) &&
                              rc.c1 > 0.0 && rc.c2 >= 0.0 && rc.rpi > 0.0;
    bool dmp_ok = false;
    if (rc_can_model && gate_arc_meta.estimateRd(thresholds, model->cap_unit, rc, table_delay, rc.rd)) {
        const int alg = rc.selectAlg(static_cast<double>(model->res_unit));
        if (alg == DMP_ALG_ZERO_C2) {
            dmp_ok = model->computeZeroC2DriverWave(gate_arc_meta,
                                                    thresholds,
                                                    rc,
                                                    driver_wave,
                                                    gate_delay);
        } else if (alg == DMP_ALG_PI) {
            dmp_ok = model->computePiDriverWave(gate_arc_meta,
                                                thresholds,
                                                rc,
                                                driver_wave,
                                                gate_delay);
        }
    }

    if (!dmp_ok) {
        driver_wave.alg = DMP_ALG_CAP;
        driver_wave.t0 = nanf("");
        driver_wave.dt = nanf("");
        driver_wave.vo_delay = nanf("");
        driver_wave.vo_slew = static_cast<float>(table_slew);
        driver_wave.vo_upper_time = nanf("");
        gate_delay = static_cast<float>(table_delay);
    }
    return isfinite(gate_delay) && isfinite(driver_wave.vo_slew);
}

__device__ bool RouteGradGatePrimitiveSlopes::hasFiniteValue() const
{
    return isfinite(delay_c1) || isfinite(delay_c2) || isfinite(delay_rpi) ||
           isfinite(delay_input_slew) || isfinite(slew_c1) || isfinite(slew_c2) ||
           isfinite(slew_rpi) || isfinite(slew_input_slew);
}

__device__ bool RouteGradNetPrimitiveReverse::gateDelaySlewWithRootRc(
    int gate_arc_id,
    int from_attr,
    int to_attr,
    double c1,
    double c2,
    double rpi,
    double input_slew,
    double& gate_delay,
    double& gate_slew) const
{
    gate_delay = nanf("");
    gate_slew = nanf("");
    if (model == nullptr || gate_arc_id < 0 || gate_arc_id >= model->num_arcs ||
        from_attr < 0 || from_attr >= NUM_ATTR || to_attr < 0 || to_attr >= NUM_ATTR ||
        model->arc_types == nullptr || model->arc_types[gate_arc_id] != 1 ||
        model->d_allocator == nullptr || !isfinite(input_slew)) {
        return false;
    }
    const int el = to_attr >> 1;
    if ((from_attr >> 1) != el) {
        return false;
    }
    const int from_pin = model->timing_arc_from_pin_id[gate_arc_id];
    const int to_pin = model->timing_arc_to_pin_id[gate_arc_id];
    const int from_slot = from_pin * NUM_ATTR + from_attr;
    const int to_slot = to_pin * NUM_ATTR + to_attr;
    const int timing_id = model->timing_arc_id_map[gate_arc_id * 2 + el];
    if (timing_id < 0 || from_pin < 0 || from_pin >= model->num_pins ||
        to_pin < 0 || to_pin >= model->num_pins ||
        from_slot < 0 || from_slot >= model->dmp_pin_slot_count ||
        to_slot < 0 || to_slot >= model->dmp_pin_slot_count) {
        return false;
    }

    DmpDriverThresholds thresholds{};
    const int input_rf = from_attr & 1;
    const DmpGateArcMeta gate_arc_meta =
        makeGateArcMetaForTiming(timing_id,
                                 input_rf,
                                 to_attr,
                                 static_cast<float>(input_slew),
                                 thresholds);
    DmpDriverWave driver_wave;
    float gate_delay_f = nanf("");
    if (!computeDriverWaveForRc(gate_arc_meta,
                                thresholds,
                                c1,
                                c2,
                                rpi,
                                driver_wave,
                                gate_delay_f)) {
        return false;
    }
    gate_delay = static_cast<double>(gate_delay_f);
    gate_slew = static_cast<double>(driver_wave.vo_slew);
    return isfinite(gate_delay) && isfinite(gate_slew);
}

// Standalone primitive finite-difference diagnostics. The main gradient path no
// longer calls these helpers; analytic failure is counted as a fail so FD remains
// a validator/debug tool rather than part of the derivative chain.
__device__ bool RouteGradNetPrimitiveReverse::gatePrimitiveFiniteDiff(
    int gate_arc_id,
    int from_attr,
    int to_attr,
    int root_slot,
    RouteGradGatePrimitiveSlopes& slopes) const
{
    slopes = {};
    if (model == nullptr || root_slot < 0 || root_slot >= model->dmp_pin_slot_count ||
        model->C1 == nullptr || model->C2 == nullptr || model->r_pi == nullptr ||
        model->pinSlew == nullptr || model->arc_types == nullptr ||
        gate_arc_id < 0 || gate_arc_id >= model->num_arcs ||
        model->arc_types[gate_arc_id] != 1 ||
        from_attr < 0 || from_attr >= NUM_ATTR || to_attr < 0 || to_attr >= NUM_ATTR) {
        return false;
    }

    const int el = to_attr >> 1;
    if ((from_attr >> 1) != el) {
        return false;
    }
    const int from_pin = model->timing_arc_from_pin_id[gate_arc_id];
    const int from_slot = from_pin * NUM_ATTR + from_attr;
    const int timing_id = model->timing_arc_id_map[gate_arc_id * 2 + el];
    if (timing_id < 0 || from_pin < 0 || from_pin >= model->num_pins ||
        from_slot < 0 || from_slot >= model->dmp_pin_slot_count ||
        model->d_allocator == nullptr) {
        return false;
    }

    const double c1 = static_cast<double>(model->C1[root_slot]);
    const double c2 = static_cast<double>(model->C2[root_slot]);
    const double rpi = static_cast<double>(model->r_pi[root_slot]);
    if (!isfinite(c1) || !isfinite(c2) || !isfinite(rpi) ||
        c1 < 0.0 || c2 < 0.0 || rpi < 0.0) {
        return false;
    }

    const bool ideal_clock_arc =
        model->isIdealClockTimingArc(timing_id, from_pin) &&
        !model->d_allocator->timing_is_constraint(timing_id);
    const double input_slew = ideal_clock_arc
                                  ? static_cast<double>(model->idealClockSlew(from_pin, from_attr))
                                  : static_cast<double>(model->pinSlew[from_slot]);
    if (!isfinite(input_slew)) {
        return false;
    }

    double base_delay = nanf("");
    double base_slew = nanf("");
    if (!gateDelaySlewWithRootRc(gate_arc_id,
                                 from_attr,
                                 to_attr,
                                 c1,
                                 c2,
                                 rpi,
                                 input_slew,
                                 base_delay,
                                 base_slew)) {
        return false;
    }

    auto slope_from_samples = [] __device__ (double base,
                                             double plus,
                                             bool plus_ok,
                                             double minus,
                                             bool minus_ok,
                                             double eps) -> double {
        if (plus_ok && minus_ok && isfinite(plus) && isfinite(minus)) {
            return (plus - minus) / (2.0 * eps);
        }
        if (plus_ok && isfinite(plus) && isfinite(base)) {
            return (plus - base) / eps;
        }
        return 0.0;
    };

    auto sample_param = [&] __device__ (int param,
                                        double value,
                                        double lower_bound,
                                        double& delay_slope,
                                        double& slew_slope) {
        const double eps = routeGradRootParamStep(value);
        double plus_delay = nanf("");
        double plus_slew = nanf("");
        double minus_delay = nanf("");
        double minus_slew = nanf("");
        double pc1 = c1;
        double pc2 = c2;
        double prpi = rpi;
        double pslew = input_slew;
        if (param == 0) pc1 = value + eps;
        if (param == 1) pc2 = value + eps;
        if (param == 2) prpi = value + eps;
        if (param == 3) pslew = value + eps;
        const bool plus_ok = gateDelaySlewWithRootRc(gate_arc_id,
                                                     from_attr,
                                                     to_attr,
                                                     pc1,
                                                     pc2,
                                                     prpi,
                                                     pslew,
                                                     plus_delay,
                                                     plus_slew);
        bool minus_ok = false;
        if (value - eps > lower_bound) {
            pc1 = c1;
            pc2 = c2;
            prpi = rpi;
            pslew = input_slew;
            if (param == 0) pc1 = value - eps;
            if (param == 1) pc2 = value - eps;
            if (param == 2) prpi = value - eps;
            if (param == 3) pslew = value - eps;
            minus_ok = gateDelaySlewWithRootRc(gate_arc_id,
                                               from_attr,
                                               to_attr,
                                               pc1,
                                               pc2,
                                               prpi,
                                               pslew,
                                               minus_delay,
                                               minus_slew);
        }
        delay_slope = slope_from_samples(base_delay,
                                         plus_delay,
                                         plus_ok,
                                         minus_delay,
                                         minus_ok,
                                         eps);
        slew_slope = slope_from_samples(base_slew,
                                        plus_slew,
                                        plus_ok,
                                        minus_slew,
                                        minus_ok,
                                        eps);
    };

    sample_param(0, c1, 0.0, slopes.delay_c1, slopes.slew_c1);
    sample_param(1, c2, -1.0e-30, slopes.delay_c2, slopes.slew_c2);
    sample_param(2, rpi, 0.0, slopes.delay_rpi, slopes.slew_rpi);
    sample_param(3, input_slew, 0.0, slopes.delay_input_slew, slopes.slew_input_slew);
    return slopes.hasFiniteValue();
}

__device__ bool RouteGradNetPrimitiveReverse::gateNetCandidate(
    int gate_arc_id,
    int net_arc_id,
    int attr,
    int input_rf,
    double& wire_delay,
    double& load_slew,
    double& delay_slope,
    double& slew_slope) const
{
    wire_delay = nanf("");
    load_slew = nanf("");
    delay_slope = nanf("");
    slew_slope = nanf("");
    if (model == nullptr || gate_arc_id < 0 || gate_arc_id >= model->num_arcs ||
        net_arc_id < 0 || net_arc_id >= model->num_arcs ||
        attr < 0 || attr >= NUM_ATTR || input_rf < 0 || input_rf > 1) {
        return false;
    }
    if (model->arc_types[gate_arc_id] != 1 || model->arc_types[net_arc_id] != 0) {
        return false;
    }

    const int gate_from_pin = model->timing_arc_from_pin_id[gate_arc_id];
    const int gate_to_pin = model->timing_arc_to_pin_id[gate_arc_id];
    const int net_from_pin = model->timing_arc_from_pin_id[net_arc_id];
    const int load_pin = model->timing_arc_to_pin_id[net_arc_id];
    if (gate_from_pin < 0 || gate_from_pin >= model->num_pins ||
        gate_to_pin != net_from_pin ||
        net_from_pin < 0 || net_from_pin >= model->num_pins ||
        load_pin < 0 || load_pin >= model->num_pins) {
        return false;
    }

    const int el = attr >> 1;
    const int from_attr = (el << 1) | input_rf;
    const int timing_id = model->timing_arc_id_map[gate_arc_id * 2 + el];
    const int gate_to_slot = gate_to_pin * NUM_ATTR + attr;
    const int gate_from_slot = gate_from_pin * NUM_ATTR + from_attr;
    const int load_slot = load_pin * NUM_ATTR + attr;
    if (timing_id < 0 || gate_to_slot < 0 || gate_from_slot < 0 || load_slot < 0 ||
        gate_to_slot >= model->dmp_pin_slot_count ||
        gate_from_slot >= model->dmp_pin_slot_count ||
        load_slot >= model->dmp_pin_slot_count || model->d_allocator == nullptr) {
        return false;
    }

    const bool ideal_clock_arc =
        model->isIdealClockTimingArc(timing_id, gate_from_pin) &&
        !model->d_allocator->timing_is_constraint(timing_id);
    const float nominal_input_slew = model->pinSlew[gate_from_slot];
    const float input_slew = ideal_clock_arc
                                 ? model->idealClockSlew(gate_from_pin, from_attr)
                                 : nominal_input_slew;
    if (!isfinite(input_slew)) {
        return false;
    }

    DmpDriverWave wave;
    DmpDriverThresholds thresholds{};
    float gate_delay = nanf("");
    const DmpGateArcMeta gate_arc_meta =
        makeGateArcMetaForTiming(timing_id, input_rf, attr, input_slew, thresholds);
    if (!model->computeGateDriverWaveForSlot(gate_arc_meta,
                                             thresholds,
                                             gate_to_slot,
                                             wave,
                                             gate_delay)) {
        return false;
    }

    const double elmore = model->elmore_delay[load_slot];
    return delaySlewSlopeForDriverWave(wave,
                                       thresholds,
                                       load_pin,
                                       attr,
                                       elmore,
                                       wire_delay,
                                       load_slew,
                                       delay_slope,
                                       slew_slope);
}

__device__ bool RouteGradNetPrimitiveReverse::makeDirectNetDriverSlopeKey(
    int arc_id,
    int attr,
    RouteGradNetDriverSlopeKey& key,
    double& delay_input_slew_slope,
    double& slew_input_slew_slope) const
{
    key = {};
    delay_input_slew_slope = 0.0;
    slew_input_slew_slope = 0.0;
    if (model == nullptr || arc_id < 0 || arc_id >= model->num_arcs ||
        attr < 0 || attr >= NUM_ATTR || model->arc_types == nullptr ||
        model->arc_types[arc_id] != 0) {
        return false;
    }
    const int from_pin = model->timing_arc_from_pin_id[arc_id];
    const int to_pin = model->timing_arc_to_pin_id[arc_id];
    if (from_pin < 0 || from_pin >= model->num_pins ||
        to_pin < 0 || to_pin >= model->num_pins) {
        return false;
    }
    const int from_slot = from_pin * NUM_ATTR + attr;
    if (from_slot < 0 || from_slot >= model->dmp_pin_slot_count ||
        model->pinSlew == nullptr || !isfinite(model->pinSlew[from_slot])) {
        return false;
    }

    const bool has_driving_cell =
        model->at_prefix_arc != nullptr && model->at_prefix_attr != nullptr &&
        model->at_prefix_attr[from_slot] == DMP_DRIVING_CELL_PREFIX_ATTR;
    key.net_arc_id = arc_id;
    key.attr = attr;
    key.input_slew_slot = from_slot;
    if (has_driving_cell) {
        key.kind = kRouteGradNetKeyDirectDrivingCell;
        key.root_slot = from_slot;
        return true;
    }

    double driver_vth = nanf("");
    double driver_vl = nanf("");
    double driver_vh = nanf("");
    double driver_derate = nanf("");
    int driver_library_id = -1;
    if (model->hasPinFlag(from_pin, DMP_PIN_PRIMARY_INPUT)) {
        driver_vth = thresholdArrayValue(model->dmp_output_thresholds, attr, model->vth_);
        driver_vl = thresholdArrayValue(model->dmp_slew_lower_thresholds, attr, model->vl_);
        driver_vh = thresholdArrayValue(model->dmp_slew_upper_thresholds, attr, model->vh_);
        driver_derate = thresholdArrayValue(model->dmp_slew_derates, attr, model->slew_derate_);
    } else {
        driver_library_id = pinLibraryId(from_pin);
        driverLibraryThresholds(driver_library_id,
                                attr,
                                driver_vth,
                                driver_vl,
                                driver_vh,
                                driver_derate);
    }
    thresholdAdjustedSlopes(to_pin,
                            attr,
                            static_cast<float>(driver_vth),
                            static_cast<float>(driver_vl),
                            static_cast<float>(driver_vh),
                            static_cast<float>(driver_derate),
                            driver_library_id,
                            0.0,
                            1.0,
                            delay_input_slew_slope,
                            slew_input_slew_slope);
    return isfinite(delay_input_slew_slope) && isfinite(slew_input_slew_slope);
}

__device__ bool RouteGradNetPrimitiveReverse::makeGateNetDriverSlopeKey(
    int gate_arc_id,
    int net_arc_id,
    int attr,
    int input_rf,
    RouteGradNetDriverSlopeKey& key) const
{
    key = {};
    if (model == nullptr || gate_arc_id < 0 || gate_arc_id >= model->num_arcs ||
        net_arc_id < 0 || net_arc_id >= model->num_arcs ||
        attr < 0 || attr >= NUM_ATTR || input_rf < 0 || input_rf > 1 ||
        model->arc_types == nullptr || model->arc_types[gate_arc_id] != 1 ||
        model->arc_types[net_arc_id] != 0 || model->d_allocator == nullptr) {
        return false;
    }
    const int gate_from_pin = model->timing_arc_from_pin_id[gate_arc_id];
    const int gate_to_pin = model->timing_arc_to_pin_id[gate_arc_id];
    const int net_from_pin = model->timing_arc_from_pin_id[net_arc_id];
    if (gate_from_pin < 0 || gate_from_pin >= model->num_pins ||
        gate_to_pin < 0 || gate_to_pin >= model->num_pins ||
        gate_to_pin != net_from_pin) {
        return false;
    }
    const int el = attr >> 1;
    const int from_attr = (el << 1) | input_rf;
    const int timing_id = model->timing_arc_id_map[gate_arc_id * 2 + el];
    const int root_slot = gate_to_pin * NUM_ATTR + attr;
    const int input_slot = gate_from_pin * NUM_ATTR + from_attr;
    if (timing_id < 0 || root_slot < 0 || root_slot >= model->dmp_pin_slot_count ||
        input_slot < 0 || input_slot >= model->dmp_pin_slot_count) {
        return false;
    }

    key.kind = kRouteGradNetKeyGateNetPair;
    key.net_arc_id = net_arc_id;
    key.gate_arc_id = gate_arc_id;
    key.attr = attr;
    key.input_rf = input_rf;
    key.root_slot = root_slot;
    const bool ideal_clock_arc =
        model->isIdealClockTimingArc(timing_id, gate_from_pin) &&
        !model->d_allocator->timing_is_constraint(timing_id);
    key.input_slew_slot = ideal_clock_arc ? -1 : input_slot;
    return true;
}

__device__ bool RouteGradNetPrimitiveReverse::netDriverWaveForKey(
    const RouteGradNetDriverSlopeKey& key,
    double c1,
    double c2,
    double rpi,
    double input_slew,
    RouteGradNetDriverWaveEval& eval) const
{
    eval = {};
    eval.wave.alg = DMP_ALG_CAP;
    if (model == nullptr || key.net_arc_id < 0 || key.net_arc_id >= model->num_arcs ||
        key.attr < 0 || key.attr >= NUM_ATTR || !isfinite(input_slew)) {
        return false;
    }
    const int load_pin = model->timing_arc_to_pin_id[key.net_arc_id];
    const int load_slot = load_pin * NUM_ATTR + key.attr;
    if (load_pin < 0 || load_pin >= model->num_pins ||
        load_slot < 0 || load_slot >= model->dmp_pin_slot_count ||
        model->elmore_delay == nullptr) {
        return false;
    }
    const double elmore = model->elmore_delay[load_slot];
    if (!isfinite(elmore)) {
        return false;
    }

    int timing_id = -1;
    int input_rf = -1;
    if (key.kind == kRouteGradNetKeyDirectDrivingCell) {
        if (model->at_prefix_arc == nullptr || model->at_prefix_attr == nullptr ||
            key.root_slot < 0 || key.root_slot >= model->dmp_pin_slot_count ||
            model->at_prefix_attr[key.root_slot] != DMP_DRIVING_CELL_PREFIX_ATTR) {
            return false;
        }
        const int driving_tag = model->at_prefix_arc[key.root_slot];
        timing_id = driving_tag >> 1;
        input_rf = driving_tag & 1;
    } else if (key.kind == kRouteGradNetKeyGateNetPair) {
        if (key.gate_arc_id < 0 || key.gate_arc_id >= model->num_arcs ||
            key.input_rf < 0 || key.input_rf > 1) {
            return false;
        }
        timing_id = model->timing_arc_id_map[key.gate_arc_id * 2 + (key.attr >> 1)];
        input_rf = key.input_rf;
    } else {
        return false;
    }
    if (timing_id < 0 || input_rf < 0) {
        return false;
    }

    DmpDriverThresholds thresholds{};
    const DmpGateArcMeta gate_arc_meta =
        makeGateArcMetaForTiming(timing_id,
                                 input_rf,
                                 key.attr,
                                 static_cast<float>(input_slew),
                                 thresholds);
    DmpDriverWave wave;
    float gate_delay = nanf("");
    if (!computeDriverWaveForRc(gate_arc_meta,
                                thresholds,
                                c1,
                                c2,
                                rpi,
                                wave,
                                gate_delay)) {
        return false;
    }

    eval.wave = wave;
    eval.thresholds = thresholds;
    eval.elmore = elmore;
    eval.gate_delay = static_cast<double>(gate_delay);
    eval.intrinsic_delay = 0.0;
    eval.timing_id = timing_id;
    eval.input_rf = input_rf;
    eval.load_pin = load_pin;
    eval.has_extra_delay = false;
    if (key.kind == kRouteGradNetKeyDirectDrivingCell) {
        double intrinsic_delay = nanf("");
        double intrinsic_slew = nanf("");
        model->gateCapDelaySlew(timing_id,
                                input_rf,
                                key.attr & 1,
                                static_cast<float>(input_slew),
                                0.0,
                                intrinsic_delay,
                                intrinsic_slew);
        if (isfinite(intrinsic_delay) && isfinite(eval.gate_delay)) {
            eval.intrinsic_delay = intrinsic_delay;
            eval.has_extra_delay = true;
        }
    }
    return true;
}

__device__ bool RouteGradNetPrimitiveReverse::netDriverDelaySlewForKey(
    const RouteGradNetDriverSlopeKey& key,
    double c1,
    double c2,
    double rpi,
    double input_slew,
    double& wire_delay,
    double& load_slew) const
{
    wire_delay = nanf("");
    load_slew = nanf("");
    if (model == nullptr || key.net_arc_id < 0 || key.net_arc_id >= model->num_arcs ||
        key.attr < 0 || key.attr >= NUM_ATTR || !isfinite(input_slew)) {
        return false;
    }
    const int load_pin = model->timing_arc_to_pin_id[key.net_arc_id];
    const int load_slot = load_pin * NUM_ATTR + key.attr;
    if (load_pin < 0 || load_pin >= model->num_pins ||
        load_slot < 0 || load_slot >= model->dmp_pin_slot_count ||
        model->elmore_delay == nullptr) {
        return false;
    }
    const double elmore = model->elmore_delay[load_slot];
    if (!isfinite(elmore)) {
        return false;
    }

    int timing_id = -1;
    int input_rf = -1;
    if (key.kind == kRouteGradNetKeyDirectDrivingCell) {
        if (model->at_prefix_arc == nullptr || model->at_prefix_attr == nullptr ||
            key.root_slot < 0 || key.root_slot >= model->dmp_pin_slot_count ||
            model->at_prefix_attr[key.root_slot] != DMP_DRIVING_CELL_PREFIX_ATTR) {
            return false;
        }
        const int driving_tag = model->at_prefix_arc[key.root_slot];
        timing_id = driving_tag >> 1;
        input_rf = driving_tag & 1;
    } else if (key.kind == kRouteGradNetKeyGateNetPair) {
        if (key.gate_arc_id < 0 || key.gate_arc_id >= model->num_arcs ||
            key.input_rf < 0 || key.input_rf > 1) {
            return false;
        }
        timing_id = model->timing_arc_id_map[key.gate_arc_id * 2 + (key.attr >> 1)];
        input_rf = key.input_rf;
    } else {
        return false;
    }
    if (timing_id < 0 || input_rf < 0) {
        return false;
    }

    DmpDriverWave wave;
    DmpDriverThresholds thresholds{};
    float gate_delay = nanf("");
    const DmpGateArcMeta gate_arc_meta =
        makeGateArcMetaForTiming(timing_id,
                                 input_rf,
                                 key.attr,
                                 static_cast<float>(input_slew),
                                 thresholds);
    if (!computeDriverWaveForRc(gate_arc_meta,
                                thresholds,
                                c1,
                                c2,
                                rpi,
                                wave,
                                gate_delay)) {
        return false;
    }

    double delay_elmore_slope = nanf("");
    double slew_elmore_slope = nanf("");
    if (!delaySlewSlopeForDriverWave(wave,
                                     thresholds,
                                     load_pin,
                                     key.attr,
                                     elmore,
                                     wire_delay,
                                     load_slew,
                                     delay_elmore_slope,
                                     slew_elmore_slope)) {
        return false;
    }

    if (key.kind == kRouteGradNetKeyDirectDrivingCell) {
        double intrinsic_delay = nanf("");
        double intrinsic_slew = nanf("");
        model->gateCapDelaySlew(timing_id,
                                input_rf,
                                key.attr & 1,
                                static_cast<float>(input_slew),
                                0.0,
                                intrinsic_delay,
                                intrinsic_slew);
        if (isfinite(intrinsic_delay) && isfinite(gate_delay)) {
            wire_delay += static_cast<double>(gate_delay) - intrinsic_delay;
        }
    }
    return isfinite(wire_delay) && isfinite(load_slew);
}

}  // namespace gt
