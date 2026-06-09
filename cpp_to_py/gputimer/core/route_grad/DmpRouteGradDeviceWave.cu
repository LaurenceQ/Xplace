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

// Waveform partial derivatives and gate-primitive chain rules.
// Waveform partial derivatives and threshold-crossing derivatives. For a
// crossing V(T, q) = alpha, the local derivative is dT/dq = -V_q / V_T. Load
// crossings also carry the explicit Elmore derivative path.
__device__ bool RouteGradNetPrimitiveReverse::driverBaseWavePartials(
    const DmpDriverWave& wave,
    double t,
    RouteGradWaveParamSlopes& partials,
    double& value,
    double& value_dt) const
{
    partials.clear();
    value = nanf("");
    value_dt = nanf("");
    if (wave.alg == DMP_ALG_CAP || !isfinite(t) || t < 0.0 ||
        !isfinite(wave.coeffs.k0) || !isfinite(wave.coeffs.p1)) {
        return false;
    }

    const double exp_p1 = exp2(-wave.coeffs.p1 * t);
    const double exp_p2 = wave.alg == DMP_ALG_PI ? exp2(-wave.coeffs.p2 * t) : 0.0;
    const double pi_k4 = wave.alg == DMP_ALG_PI ? wave.coeffs.k4 : 0.0;
    const double inner = wave.coeffs.k1 + wave.coeffs.k2 * t +
                         wave.coeffs.k3 * exp_p1 + pi_k4 * exp_p2;
    value = wave.coeffs.k0 * inner;
    value_dt = wave.coeffs.k0 * (wave.coeffs.k2 -
                                 wave.coeffs.k3 * wave.coeffs.p1 * exp_p1 -
                                 pi_k4 * wave.coeffs.p2 * exp_p2);
    partials.k0 = inner;
    partials.k1 = wave.coeffs.k0;
    partials.k2 = wave.coeffs.k0 * t;
    partials.k3 = wave.coeffs.k0 * exp_p1;
    partials.p1 = -wave.coeffs.k0 * wave.coeffs.k3 * t * exp_p1;
    if (wave.alg == DMP_ALG_PI) {
        partials.k4 = wave.coeffs.k0 * exp_p2;
        partials.p2 = -wave.coeffs.k0 * wave.coeffs.k4 * t * exp_p2;
    }
    return isfinite(value) && isfinite(value_dt);
}

__device__ bool RouteGradNetPrimitiveReverse::loadBaseWavePartials(
    const DmpDriverWave& wave,
    double elmore,
    double t,
    RouteGradWaveParamSlopes& partials,
    double& value,
    double& value_dt) const
{
    partials.clear();
    value = nanf("");
    value_dt = nanf("");
    if (wave.alg == DMP_ALG_CAP || !isfinite(elmore) || elmore <= 0.0 ||
        !isfinite(t) || t < 0.0 || !isfinite(wave.coeffs.k0)) {
        return false;
    }

    const double p3 = 1.0 / elmore;
    const double p1_gap = wave.coeffs.p1 - p3;
    if (!isfinite(p1_gap) || p1_gap == 0.0) {
        return false;
    }
    const double exp_p1 = exp2(-wave.coeffs.p1 * t);
    const double exp_p3 = exp2(-p3 * t);
    const double d1 = wave.coeffs.k0 * (wave.coeffs.k1 - wave.coeffs.k2 / p3);
    const double d3 = -p3 * wave.coeffs.k0 * wave.coeffs.k3 / p1_gap;
    double d4 = 0.0;
    double d5 = wave.coeffs.k0 * (wave.coeffs.k2 / p3 - wave.coeffs.k1 +
                                  p3 * wave.coeffs.k3 / p1_gap);

    double exp_p2 = 0.0;
    if (wave.alg == DMP_ALG_PI) {
        const double p2_gap = wave.coeffs.p2 - p3;
        if (!isfinite(p2_gap) || p2_gap == 0.0) {
            return false;
        }
        exp_p2 = exp2(-wave.coeffs.p2 * t);
        d4 = -p3 * wave.coeffs.k0 * wave.coeffs.k4 / p2_gap;
        d5 += wave.coeffs.k0 * p3 * wave.coeffs.k4 / p2_gap;
    }

    value = d1 + t + d3 * exp_p1 + d4 * exp_p2 + d5 * exp_p3;
    value_dt = 1.0 - d3 * wave.coeffs.p1 * exp_p1 -
               d4 * wave.coeffs.p2 * exp_p2 - d5 * p3 * exp_p3;

    const double d1_k0 = wave.coeffs.k1 - wave.coeffs.k2 / p3;
    const double d3_k0 = -p3 * wave.coeffs.k3 / p1_gap;
    double d4_k0 = 0.0;
    double d5_k0 = wave.coeffs.k2 / p3 - wave.coeffs.k1 +
                   p3 * wave.coeffs.k3 / p1_gap;
    partials.k1 = wave.coeffs.k0 * (1.0 - exp_p3);
    partials.k2 = wave.coeffs.k0 * (exp_p3 - 1.0) / p3;
    partials.k3 = wave.coeffs.k0 * p3 * (exp_p3 - exp_p1) / p1_gap;
    const double d3_p1 = p3 * wave.coeffs.k0 * wave.coeffs.k3 / (p1_gap * p1_gap);
    const double d5_p1 = -d3_p1;
    partials.p1 = d3_p1 * exp_p1 - d3 * t * exp_p1 + d5_p1 * exp_p3;

    if (wave.alg == DMP_ALG_PI) {
        const double p2_gap = wave.coeffs.p2 - p3;
        d4_k0 = -p3 * wave.coeffs.k4 / p2_gap;
        d5_k0 += p3 * wave.coeffs.k4 / p2_gap;
        partials.k4 = wave.coeffs.k0 * p3 * (exp_p3 - exp_p2) / p2_gap;
        const double d4_p2 = p3 * wave.coeffs.k0 * wave.coeffs.k4 / (p2_gap * p2_gap);
        const double d5_p2 = -d4_p2;
        partials.p2 = d4_p2 * exp_p2 - d4 * t * exp_p2 + d5_p2 * exp_p3;
    }
    partials.k0 = d1_k0 + d3_k0 * exp_p1 + d4_k0 * exp_p2 + d5_k0 * exp_p3;
    return isfinite(value) && isfinite(value_dt);
}

__device__ bool RouteGradNetPrimitiveReverse::driverWaveValuePartials(
    const DmpDriverWave& wave,
    double t,
    RouteGradWaveParamSlopes& partials,
    double& value_dt) const
{
    partials.clear();
    value_dt = nanf("");
    const double t1 = t - wave.t0;
    if (t1 <= 0.0 || !isfinite(wave.dt) || wave.dt <= 0.0) {
        return false;
    }

    RouteGradWaveParamSlopes p_now;
    double v_now = nanf("");
    double dv_now = nanf("");
    if (!driverBaseWavePartials(wave, t1, p_now, v_now, dv_now)) {
        return false;
    }
    const double inv_dt = 1.0 / wave.dt;
    const double inv_dt2 = inv_dt * inv_dt;
    if (t1 <= wave.dt) {
        partials = p_now;
        partials.scale(inv_dt);
        partials.t0 = -dv_now * inv_dt;
        partials.dt = -v_now * inv_dt2;
        value_dt = dv_now * inv_dt;
        return isfinite(value_dt);
    }

    RouteGradWaveParamSlopes p_prev;
    double v_prev = nanf("");
    double dv_prev = nanf("");
    if (!driverBaseWavePartials(wave, t1 - wave.dt, p_prev, v_prev, dv_prev)) {
        return false;
    }
#define ROUTE_GRAD_DIFF_FIELD(field) partials.field = (p_now.field - p_prev.field) * inv_dt
    ROUTE_GRAD_DIFF_FIELD(k0);
    ROUTE_GRAD_DIFF_FIELD(k1);
    ROUTE_GRAD_DIFF_FIELD(k2);
    ROUTE_GRAD_DIFF_FIELD(k3);
    ROUTE_GRAD_DIFF_FIELD(k4);
    ROUTE_GRAD_DIFF_FIELD(p1);
    ROUTE_GRAD_DIFF_FIELD(p2);
#undef ROUTE_GRAD_DIFF_FIELD
    partials.t0 = (-dv_now + dv_prev) * inv_dt;
    partials.dt = dv_prev * inv_dt - (v_now - v_prev) * inv_dt2;
    value_dt = (dv_now - dv_prev) * inv_dt;
    return isfinite(value_dt);
}

__device__ bool RouteGradNetPrimitiveReverse::loadWaveValuePartials(
    const DmpDriverWave& wave,
    double elmore,
    double t,
    RouteGradWaveParamSlopes& partials,
    double& value_dt,
    double& value_elmore) const
{
    partials.clear();
    value_dt = nanf("");
    value_elmore = nanf("");
    const double t1 = t - wave.t0;
    if (t1 <= 0.0 || !isfinite(wave.dt) || wave.dt <= 0.0) {
        return false;
    }

    RouteGradWaveParamSlopes p_now;
    double v_now = nanf("");
    double dv_now = nanf("");
    if (!loadBaseWavePartials(wave, elmore, t1, p_now, v_now, dv_now)) {
        return false;
    }
    const double inv_dt = 1.0 / wave.dt;
    const double inv_dt2 = inv_dt * inv_dt;
    if (t1 <= wave.dt) {
        partials = p_now;
        partials.scale(inv_dt);
        partials.t0 = -dv_now * inv_dt;
        partials.dt = -v_now * inv_dt2;
        value_dt = dv_now * inv_dt;
        value_elmore = loadWave0ElmoreDerivative(wave, elmore, t1) * inv_dt;
        return isfinite(value_dt) && isfinite(value_elmore);
    }

    RouteGradWaveParamSlopes p_prev;
    double v_prev = nanf("");
    double dv_prev = nanf("");
    if (!loadBaseWavePartials(wave, elmore, t1 - wave.dt, p_prev, v_prev, dv_prev)) {
        return false;
    }
#define ROUTE_GRAD_DIFF_FIELD(field) partials.field = (p_now.field - p_prev.field) * inv_dt
    ROUTE_GRAD_DIFF_FIELD(k0);
    ROUTE_GRAD_DIFF_FIELD(k1);
    ROUTE_GRAD_DIFF_FIELD(k2);
    ROUTE_GRAD_DIFF_FIELD(k3);
    ROUTE_GRAD_DIFF_FIELD(k4);
    ROUTE_GRAD_DIFF_FIELD(p1);
    ROUTE_GRAD_DIFF_FIELD(p2);
#undef ROUTE_GRAD_DIFF_FIELD
    partials.t0 = (-dv_now + dv_prev) * inv_dt;
    partials.dt = dv_prev * inv_dt - (v_now - v_prev) * inv_dt2;
    value_dt = (dv_now - dv_prev) * inv_dt;
    value_elmore = (loadWave0ElmoreDerivative(wave, elmore, t1) -
                    loadWave0ElmoreDerivative(wave, elmore, t1 - wave.dt)) * inv_dt;
    return isfinite(value_dt) && isfinite(value_elmore);
}

__device__ bool RouteGradNetPrimitiveReverse::driverCrossingWaveSlopes(
    const DmpDriverWave& wave,
    double crossing,
    RouteGradWaveParamSlopes& slopes) const
{
    slopes.clear();
    RouteGradWaveParamSlopes value_partials;
    double value_dt = nanf("");
    if (!driverWaveValuePartials(wave, crossing, value_partials, value_dt) ||
        !isfinite(value_dt) || value_dt == 0.0) {
        return false;
    }
    slopes = value_partials;
    slopes.scale(-1.0 / value_dt);
    return true;
}

__device__ bool RouteGradNetPrimitiveReverse::loadCrossingWaveSlopes(
    const DmpDriverWave& wave,
    double elmore,
    double crossing,
    RouteGradWaveParamSlopes& slopes,
    double& elmore_slope) const
{
    slopes.clear();
    elmore_slope = nanf("");
    RouteGradWaveParamSlopes value_partials;
    double value_dt = nanf("");
    double value_elmore = nanf("");
    if (!loadWaveValuePartials(wave, elmore, crossing, value_partials, value_dt, value_elmore) ||
        !isfinite(value_dt) || value_dt == 0.0 || !isfinite(value_elmore)) {
        return false;
    }
    slopes = value_partials;
    slopes.scale(-1.0 / value_dt);
    elmore_slope = -value_elmore / value_dt;
    return isfinite(elmore_slope);
}

__device__ double RouteGradNetPrimitiveReverse::inputPortDelayElmoreSlope(
    int load_pin_id,
    int load_attr,
    double& slew_slope) const
{
    slew_slope = nanf("");
    if (model == nullptr) {
        return nanf("");
    }

    double load_vth = nanf("");
    double load_vl = nanf("");
    double load_vh = nanf("");
    double load_derate = nanf("");
    loadPinThresholds(load_pin_id, load_attr, load_vth, load_vl, load_vh, load_derate);
    if (!isfinite(load_vth) || !isfinite(load_vl) || !isfinite(load_vh) ||
        !isfinite(load_derate) || load_vth <= 0.0 || load_vth >= 1.0 ||
        load_vl <= 0.0 || load_vh >= 1.0 || load_vh <= load_vl ||
        load_derate <= 0.0) {
        return nanf("");
    }

    const double raw_delay_slope = -log(1.0 - load_vth);
    const double raw_slew_slope = log((1.0 - load_vl) / (1.0 - load_vh)) / load_derate;
    const double driver_vth = thresholdArrayValue(model->dmp_output_thresholds,
                                                  load_attr,
                                                  model->vth_);
    const double driver_vl = thresholdArrayValue(model->dmp_slew_lower_thresholds,
                                                 load_attr,
                                                 model->vl_);
    const double driver_vh = thresholdArrayValue(model->dmp_slew_upper_thresholds,
                                                 load_attr,
                                                 model->vh_);
    const double driver_derate = thresholdArrayValue(model->dmp_slew_derates,
                                                     load_attr,
                                                     model->slew_derate_);
    double delay_slope = nanf("");
    thresholdAdjustedSlopes(load_pin_id,
                            load_attr,
                            static_cast<float>(driver_vth),
                            static_cast<float>(driver_vl),
                            static_cast<float>(driver_vh),
                            static_cast<float>(driver_derate),
                            -1,
                            raw_delay_slope,
                            raw_slew_slope,
                            delay_slope,
                            slew_slope);
    return delay_slope;
}

__device__ double RouteGradNetPrimitiveReverse::loadWave0ElmoreDerivative(
    const DmpDriverWave& wave,
    double elmore,
    double t) const
{
    if (wave.alg == DMP_ALG_CAP || !isfinite(elmore) || elmore <= 0.0 ||
        !isfinite(t) || t < 0.0) {
        return 0.0;
    }

    const double p3 = 1.0 / elmore;
    const double p3_sq = p3 * p3;
    const double p1_gap = wave.coeffs.p1 - p3;
    if (!isfinite(p1_gap) || p1_gap == 0.0) {
        return nanf("");
    }

    const double dd1 = -wave.coeffs.k0 * wave.coeffs.k2;
    const double dd3 = wave.coeffs.k0 * wave.coeffs.k3 * wave.coeffs.p1 *
                       p3_sq / (p1_gap * p1_gap);
    double dd4 = 0.0;
    double dd5 = wave.coeffs.k0 *
                 (wave.coeffs.k2 - wave.coeffs.k3 * wave.coeffs.p1 *
                                       p3_sq / (p1_gap * p1_gap));

    double d5 = wave.coeffs.k0 * (wave.coeffs.k2 / p3 - wave.coeffs.k1 +
                                  p3 * wave.coeffs.k3 / p1_gap);
    if (wave.alg == DMP_ALG_PI) {
        const double p2_gap = wave.coeffs.p2 - p3;
        if (!isfinite(p2_gap) || p2_gap == 0.0) {
            return nanf("");
        }
        dd4 = wave.coeffs.k0 * wave.coeffs.k4 * wave.coeffs.p2 *
              p3_sq / (p2_gap * p2_gap);
        dd5 -= wave.coeffs.k0 * wave.coeffs.k4 * wave.coeffs.p2 *
               p3_sq / (p2_gap * p2_gap);
        d5 += wave.coeffs.k0 * p3 * wave.coeffs.k4 / p2_gap;
    }

    const double exp_p1 = exp2(-wave.coeffs.p1 * t);
    const double exp_p2 = wave.alg == DMP_ALG_PI ? exp2(-wave.coeffs.p2 * t) : 0.0;
    const double exp_p3 = exp2(-p3 * t);
    return dd1 + dd3 * exp_p1 + dd4 * exp_p2 +
           dd5 * exp_p3 + d5 * exp_p3 * t * p3_sq;
}

__device__ double RouteGradNetPrimitiveReverse::loadWaveElmoreDerivative(
    const DmpDriverWave& wave,
    double elmore,
    double t) const
{
    const double t1 = t - wave.t0;
    if (t1 <= 0.0) {
        return 0.0;
    }
    if (!isfinite(wave.dt) || wave.dt <= 0.0) {
        return nanf("");
    }
    const double inv_dt = 1.0 / wave.dt;
    if (t1 <= wave.dt) {
        return loadWave0ElmoreDerivative(wave, elmore, t1) * inv_dt;
    }
    return (loadWave0ElmoreDerivative(wave, elmore, t1) -
            loadWave0ElmoreDerivative(wave, elmore, t1 - wave.dt)) * inv_dt;
}

__device__ bool RouteGradNetPrimitiveReverse::loadCrossingElmoreSlope(
    const DmpDriverWave& wave,
    double elmore,
    double crossing,
    double& slope) const
{
    slope = nanf("");
    if (!isfinite(crossing)) {
        return false;
    }
    double vl = nanf("");
    double dvl_dt = nanf("");
    wave.loadWave(elmore, crossing, vl, dvl_dt);
    const double dvl_de = loadWaveElmoreDerivative(wave, elmore, crossing);
    if (!isfinite(dvl_dt) || !isfinite(dvl_de) || dvl_dt == 0.0) {
        return false;
    }
    slope = -dvl_de / dvl_dt;
    return isfinite(slope);
}

__device__ bool RouteGradNetPrimitiveReverse::delaySlewSlopeForDriverWave(
    const DmpDriverWave& wave,
    const DmpDriverThresholds& thresholds,
    int load_pin_id,
    int load_attr,
    double elmore,
    double& wire_delay,
    double& load_slew,
    double& delay_slope,
    double& slew_slope) const
{
    wire_delay = nanf("");
    load_slew = nanf("");
    delay_slope = 1.0;
    slew_slope = 0.0;
    if (model == nullptr) {
        return false;
    }

    model->loadDelaySlewFromDriverWave(wave,
                                       thresholds,
                                       load_pin_id,
                                       load_attr,
                                       elmore,
                                       wire_delay,
                                       load_slew);
    if (!isfinite(wire_delay) || !isfinite(load_slew)) {
        return false;
    }

    if (!thresholds.valid() || !isfinite(wave.vo_slew) || !isfinite(elmore)) {
        return true;
    }
    if (!wave.hasValidDriver() || elmore == 0.0 ||
        elmore < static_cast<double>(wave.vo_slew) * 1e-3) {
        thresholdAdjustedSlopes(load_pin_id,
                                load_attr,
                                thresholds.vth,
                                thresholds.vl,
                                thresholds.vh,
                                thresholds.derate,
                                thresholds.library_id,
                                1.0,
                                0.0,
                                delay_slope,
                                slew_slope);
        return isfinite(delay_slope) && isfinite(slew_slope);
    }

    const double t_lower = wave.t0;
    const double t_upper = wave.vo_upper_time + elmore * 2.0;
#if DMP_LOAD_CROSSING_BISECTION
    const double t_vth = wave.findLoadCrossingBisection(elmore,
                                                        thresholds.vth,
                                                        t_lower,
                                                        t_upper,
                                                        model->MAX_ITER,
                                                        model->x_tol);
    const double t_vl = wave.findLoadCrossingBisection(elmore,
                                                       thresholds.vl,
                                                       t_lower,
                                                       t_vth,
                                                       model->MAX_ITER,
                                                       model->x_tol);
    const double t_vh = wave.findLoadCrossingBisection(elmore,
                                                       thresholds.vh,
                                                       t_vth,
                                                       t_upper,
                                                       model->MAX_ITER,
                                                       model->x_tol);
#else
    const double t_vth = wave.findLoadCrossing(elmore,
                                               thresholds.vth,
                                               t_lower,
                                               t_upper,
                                               model->MAX_ITER,
                                               model->x_tol);
    const double t_vl = wave.findLoadCrossing(elmore,
                                              thresholds.vl,
                                              t_lower,
                                              t_vth,
                                              model->MAX_ITER,
                                              model->x_tol);
    const double t_vh = wave.findLoadCrossing(elmore,
                                              thresholds.vh,
                                              t_vth,
                                              t_upper,
                                              model->MAX_ITER,
                                              model->x_tol);
#endif
    double raw_delay = t_vth - wave.vo_delay;
    double raw_slew = (t_vh - t_vl) / static_cast<double>(thresholds.derate);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(t_vh) ||
        !isfinite(raw_delay) || !isfinite(raw_slew)) {
        return true;
    }

    double raw_delay_slope = nanf("");
    double t_vl_slope = nanf("");
    double t_vh_slope = nanf("");
    if (!loadCrossingElmoreSlope(wave, elmore, t_vth, raw_delay_slope) ||
        !loadCrossingElmoreSlope(wave, elmore, t_vl, t_vl_slope) ||
        !loadCrossingElmoreSlope(wave, elmore, t_vh, t_vh_slope)) {
        return true;
    }
    double raw_slew_slope = (t_vh_slope - t_vl_slope) /
                            static_cast<double>(thresholds.derate);

    if (raw_delay < 0.0) {
        if (-raw_delay > model->vth_time_tol * wave.vo_delay) {
            delay_slope = 1.0;
            slew_slope = 0.0;
            return true;
        }
        raw_delay_slope = 1.0;
    }
    if (raw_slew < static_cast<double>(wave.vo_slew)) {
        if ((static_cast<double>(wave.vo_slew) - raw_slew) >
            model->vth_time_tol * static_cast<double>(wave.vo_slew)) {
            delay_slope = 1.0;
            slew_slope = 0.0;
            return true;
        }
        raw_slew_slope = 0.0;
    }

    thresholdAdjustedSlopes(load_pin_id,
                            load_attr,
                            thresholds.vth,
                            thresholds.vl,
                            thresholds.vh,
                            thresholds.derate,
                            thresholds.library_id,
                            raw_delay_slope,
                            raw_slew_slope,
                            delay_slope,
                            slew_slope);
    return isfinite(delay_slope) && isfinite(slew_slope);
}


// Waveform crossing derivatives and gate-primitive chain rules.
// Convert waveform/crossing partials into net delay and net slew slopes for one
// reconstructed driver/sink primitive, including threshold adjustment and the
// explicit sink-Elmore contribution.
__device__ bool RouteGradNetPrimitiveReverse::delaySlewWaveParamSlopes(
    const DmpDriverWave& wave,
    const DmpDriverThresholds& thresholds,
    int load_pin_id,
    int load_attr,
    double elmore,
    RouteGradDelaySlewWaveSlopes& slopes) const
{
    slopes.delay.clear();
    slopes.slew.clear();
    if (model == nullptr || !thresholds.valid() || !wave.hasValidDriver() ||
        !isfinite(wave.vo_slew) || !isfinite(elmore) || elmore <= 0.0 ||
        elmore < static_cast<double>(wave.vo_slew) * 1.0e-3) {
        return false;
    }

    const double t_lower = wave.t0;
    const double t_upper = wave.vo_upper_time + elmore * 2.0;
#if DMP_LOAD_CROSSING_BISECTION
    const double load_vth = wave.findLoadCrossingBisection(elmore,
                                                           thresholds.vth,
                                                           t_lower,
                                                           t_upper,
                                                           model->MAX_ITER,
                                                           model->x_tol);
    const double load_vl = wave.findLoadCrossingBisection(elmore,
                                                          thresholds.vl,
                                                          t_lower,
                                                          load_vth,
                                                          model->MAX_ITER,
                                                          model->x_tol);
    const double load_vh = wave.findLoadCrossingBisection(elmore,
                                                          thresholds.vh,
                                                          load_vth,
                                                          t_upper,
                                                          model->MAX_ITER,
                                                          model->x_tol);
#else
    const double load_vth = wave.findLoadCrossing(elmore,
                                                  thresholds.vth,
                                                  t_lower,
                                                  t_upper,
                                                  model->MAX_ITER,
                                                  model->x_tol);
    const double load_vl = wave.findLoadCrossing(elmore,
                                                 thresholds.vl,
                                                 t_lower,
                                                 load_vth,
                                                 model->MAX_ITER,
                                                 model->x_tol);
    const double load_vh = wave.findLoadCrossing(elmore,
                                                 thresholds.vh,
                                                 load_vth,
                                                 t_upper,
                                                 model->MAX_ITER,
                                                 model->x_tol);
#endif
    const double driver_vth = wave.vo_delay;
    if (!isfinite(load_vth) || !isfinite(load_vl) || !isfinite(load_vh) ||
        !isfinite(driver_vth)) {
        return false;
    }

    RouteGradWaveParamSlopes load_vth_slope;
    RouteGradWaveParamSlopes load_vl_slope;
    RouteGradWaveParamSlopes load_vh_slope;
    RouteGradWaveParamSlopes driver_vth_slope;
    double load_vth_elmore = nanf("");
    double load_vl_elmore = nanf("");
    double load_vh_elmore = nanf("");
    if (!loadCrossingWaveSlopes(wave, elmore, load_vth, load_vth_slope, load_vth_elmore) ||
        !loadCrossingWaveSlopes(wave, elmore, load_vl, load_vl_slope, load_vl_elmore) ||
        !loadCrossingWaveSlopes(wave, elmore, load_vh, load_vh_slope, load_vh_elmore) ||
        !driverCrossingWaveSlopes(wave, driver_vth, driver_vth_slope)) {
        return false;
    }

    RouteGradWaveParamSlopes raw_delay;
    RouteGradWaveParamSlopes raw_slew;
#define ROUTE_GRAD_RAW_DELAY_FIELD(field) raw_delay.field = load_vth_slope.field - driver_vth_slope.field
    ROUTE_GRAD_RAW_DELAY_FIELD(t0);
    ROUTE_GRAD_RAW_DELAY_FIELD(dt);
    ROUTE_GRAD_RAW_DELAY_FIELD(k0);
    ROUTE_GRAD_RAW_DELAY_FIELD(k1);
    ROUTE_GRAD_RAW_DELAY_FIELD(k2);
    ROUTE_GRAD_RAW_DELAY_FIELD(k3);
    ROUTE_GRAD_RAW_DELAY_FIELD(k4);
    ROUTE_GRAD_RAW_DELAY_FIELD(p1);
    ROUTE_GRAD_RAW_DELAY_FIELD(p2);
#undef ROUTE_GRAD_RAW_DELAY_FIELD
#define ROUTE_GRAD_RAW_SLEW_FIELD(field) \
    raw_slew.field = (load_vh_slope.field - load_vl_slope.field) / static_cast<double>(thresholds.derate)
    ROUTE_GRAD_RAW_SLEW_FIELD(t0);
    ROUTE_GRAD_RAW_SLEW_FIELD(dt);
    ROUTE_GRAD_RAW_SLEW_FIELD(k0);
    ROUTE_GRAD_RAW_SLEW_FIELD(k1);
    ROUTE_GRAD_RAW_SLEW_FIELD(k2);
    ROUTE_GRAD_RAW_SLEW_FIELD(k3);
    ROUTE_GRAD_RAW_SLEW_FIELD(k4);
    ROUTE_GRAD_RAW_SLEW_FIELD(p1);
    ROUTE_GRAD_RAW_SLEW_FIELD(p2);
#undef ROUTE_GRAD_RAW_SLEW_FIELD

    const double raw_delay_value = load_vth - driver_vth;
    const double raw_slew_value = (load_vh - load_vl) / static_cast<double>(thresholds.derate);
    if (!isfinite(raw_delay_value) || !isfinite(raw_slew_value)) {
        return false;
    }
    if (raw_delay_value < 0.0) {
        if (-raw_delay_value > model->vth_time_tol * wave.vo_delay) {
            return false;
        }
        raw_delay.clear();
    }
    if (raw_slew_value < static_cast<double>(wave.vo_slew)) {
        if ((static_cast<double>(wave.vo_slew) - raw_slew_value) >
            model->vth_time_tol * static_cast<double>(wave.vo_slew)) {
            return false;
        }
        const double driver_vl = wave.findDriverCrossing(thresholds.vl,
                                                         wave.t0,
                                                         wave.vo_delay,
                                                         model->MAX_ITER,
                                                         model->x_tol);
        const double driver_vh = wave.findDriverCrossing(thresholds.vh,
                                                         wave.vo_delay,
                                                         wave.vo_upper_time,
                                                         model->MAX_ITER,
                                                         model->x_tol);
        RouteGradWaveParamSlopes driver_vl_slope;
        RouteGradWaveParamSlopes driver_vh_slope;
        if (!isfinite(driver_vl) || !isfinite(driver_vh) ||
            !driverCrossingWaveSlopes(wave, driver_vl, driver_vl_slope) ||
            !driverCrossingWaveSlopes(wave, driver_vh, driver_vh_slope)) {
            return false;
        }
#define ROUTE_GRAD_DRV_SLEW_FIELD(field) \
        raw_slew.field = (driver_vh_slope.field - driver_vl_slope.field) / static_cast<double>(thresholds.derate)
        ROUTE_GRAD_DRV_SLEW_FIELD(t0);
        ROUTE_GRAD_DRV_SLEW_FIELD(dt);
        ROUTE_GRAD_DRV_SLEW_FIELD(k0);
        ROUTE_GRAD_DRV_SLEW_FIELD(k1);
        ROUTE_GRAD_DRV_SLEW_FIELD(k2);
        ROUTE_GRAD_DRV_SLEW_FIELD(k3);
        ROUTE_GRAD_DRV_SLEW_FIELD(k4);
        ROUTE_GRAD_DRV_SLEW_FIELD(p1);
        ROUTE_GRAD_DRV_SLEW_FIELD(p2);
#undef ROUTE_GRAD_DRV_SLEW_FIELD
    }

    thresholdAdjustedWaveSlopes(load_pin_id,
                                load_attr,
                                thresholds,
                                raw_delay,
                                raw_slew,
                                slopes);
    return true;
}

__device__ bool RouteGradNetPrimitiveReverse::driverOutputSlewWaveParamSlopes(
    const DmpDriverWave& wave,
    const DmpDriverThresholds& thresholds,
    RouteGradWaveParamSlopes& slopes) const
{
    slopes.clear();
    if (model == nullptr || !thresholds.valid() || !wave.hasValidDriver() ||
        !isfinite(wave.vo_delay) || !isfinite(wave.vo_upper_time)) {
        return false;
    }
    const double driver_vl = wave.findDriverCrossing(thresholds.vl,
                                                     wave.t0,
                                                     wave.vo_delay,
                                                     model->MAX_ITER,
                                                     model->x_tol);
    const double driver_vh = wave.findDriverCrossing(thresholds.vh,
                                                     wave.vo_delay,
                                                     wave.vo_upper_time,
                                                     model->MAX_ITER,
                                                     model->x_tol);
    RouteGradWaveParamSlopes vl_slope;
    RouteGradWaveParamSlopes vh_slope;
    if (!isfinite(driver_vl) || !isfinite(driver_vh) ||
        !driverCrossingWaveSlopes(wave, driver_vl, vl_slope) ||
        !driverCrossingWaveSlopes(wave, driver_vh, vh_slope)) {
        return false;
    }
    const double inv_derate = 1.0 / static_cast<double>(thresholds.derate);
#define ROUTE_GRAD_DRIVER_SLEW_FIELD(field) \
    slopes.field = (vh_slope.field - vl_slope.field) * inv_derate
    ROUTE_GRAD_DRIVER_SLEW_FIELD(t0);
    ROUTE_GRAD_DRIVER_SLEW_FIELD(dt);
    ROUTE_GRAD_DRIVER_SLEW_FIELD(k0);
    ROUTE_GRAD_DRIVER_SLEW_FIELD(k1);
    ROUTE_GRAD_DRIVER_SLEW_FIELD(k2);
    ROUTE_GRAD_DRIVER_SLEW_FIELD(k3);
    ROUTE_GRAD_DRIVER_SLEW_FIELD(k4);
    ROUTE_GRAD_DRIVER_SLEW_FIELD(p1);
    ROUTE_GRAD_DRIVER_SLEW_FIELD(p2);
#undef ROUTE_GRAD_DRIVER_SLEW_FIELD
    return isfinite(slopes.t0) && isfinite(slopes.dt) && isfinite(slopes.k0) &&
           isfinite(slopes.k1) && isfinite(slopes.k2) && isfinite(slopes.k3) &&
           isfinite(slopes.k4) && isfinite(slopes.p1) && isfinite(slopes.p2);
}

// Analytic gate primitive slopes: output load PI parameters and input slew to
// gate delay/output slew. Covers CAP, ZERO_C2, and PI branches.
__device__ bool RouteGradNetPrimitiveReverse::gatePrimitiveWaveChainSlopes(
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
        from_attr < 0 || from_attr >= NUM_ATTR || to_attr < 0 || to_attr >= NUM_ATTR ||
        model->d_allocator == nullptr) {
        return false;
    }
    const int el = to_attr >> 1;
    if ((from_attr >> 1) != el) {
        return false;
    }
    const int from_pin = model->timing_arc_from_pin_id[gate_arc_id];
    const int timing_id = model->timing_arc_id_map[gate_arc_id * 2 + el];
    const int from_slot = from_pin * NUM_ATTR + from_attr;
    if (timing_id < 0 || from_pin < 0 || from_pin >= model->num_pins ||
        from_slot < 0 || from_slot >= model->dmp_pin_slot_count) {
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

    DmpDriverThresholds thresholds{};
    const int input_rf = from_attr & 1;
    const DmpGateArcMeta gate_arc_meta =
        makeGateArcMetaForTiming(timing_id,
                                 input_rf,
                                 to_attr,
                                 static_cast<float>(input_slew),
                                 thresholds);
    if (!gate_arc_meta.valid || !thresholds.valid()) {
        return false;
    }

    DmpDriverWave base_wave;
    float base_gate_delay = nanf("");
    if (!computeDriverWaveForRc(gate_arc_meta,
                                thresholds,
                                c1,
                                c2,
                                rpi,
                                base_wave,
                                base_gate_delay)) {
        return false;
    }
    if (base_wave.alg == DMP_ALG_CAP) {
        RouteGradLutSlopes cap_delay;
        RouteGradLutSlopes cap_slew;
        if (!gateArcCapDelaySlewSlopes(gate_arc_meta, c1 + c2, cap_delay, cap_slew)) {
            return false;
        }
        slopes.delay_c1 = cap_delay.load_slope;
        slopes.delay_c2 = cap_delay.load_slope;
        slopes.delay_rpi = 0.0;
        slopes.delay_input_slew = cap_delay.input_slew_slope;
        slopes.slew_c1 = cap_slew.load_slope;
        slopes.slew_c2 = cap_slew.load_slope;
        slopes.slew_rpi = 0.0;
        slopes.slew_input_slew = cap_slew.input_slew_slope;
        return slopes.hasFiniteValue();
    }
    if (!base_wave.hasValidDriver() ||
        (base_wave.alg != DMP_ALG_PI && base_wave.alg != DMP_ALG_ZERO_C2)) {
        return false;
    }

    double rd = nanf("");
    double rd_c1 = 0.0;
    double rd_c2 = 0.0;
    double rd_input_slew = 0.0;
    if (!estimateRdWithSlopes(gate_arc_meta,
                              thresholds,
                              c1,
                              c2,
                              rd,
                              rd_c1,
                              rd_c2,
                              rd_input_slew)) {
        if (base_wave.alg == DMP_ALG_PI) {
            routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailRd);
        }
        return false;
    }
    DmpRcParams rc{};
    rc.c1 = c1;
    rc.c2 = c2;
    rc.rpi = rpi;
    rc.rd = rd;
    if (base_wave.alg == DMP_ALG_ZERO_C2) {
        DmpWaveCoeffs coeffs{};
        if (!rc.initZeroC2(coeffs)) {
            return false;
        }
        RouteGradWaveParamSlopes gate_delay_wave_slopes;
        RouteGradWaveParamSlopes output_slew_slopes;
        if (!driverCrossingWaveSlopes(base_wave, base_wave.vo_delay, gate_delay_wave_slopes) ||
            !driverOutputSlewWaveParamSlopes(base_wave, thresholds, output_slew_slopes)) {
            return false;
        }
        auto eval_zero_direction = [&] __device__ (double dc1,
                                                   double drpi,
                                                   double drd,
                                                   double dinput_slew,
                                                   double& delay_slope,
                                                   double& slew_slope) -> bool {
            RouteGradWaveParamSlopes coeff_dir;
            if (!zeroC2CoeffDirectionSlopes(rc, coeffs, dc1, drpi, drd, coeff_dir)) {
                return false;
            }
            RouteGradOnePoleSolveSlopes solve_dir;
            if (!onePoleImplicitSolveDirectionSlopes(gate_arc_meta,
                                                     thresholds,
                                                     rc,
                                                     base_wave.t0,
                                                     base_wave.dt,
                                                     dc1,
                                                     drd,
                                                     dinput_slew,
                                                     coeff_dir,
                                                     solve_dir)) {
                return false;
            }
            delay_slope = gate_delay_wave_slopes.dot(solve_dir.wave);
            slew_slope = output_slew_slopes.dot(solve_dir.wave);
            return isfinite(delay_slope) && isfinite(slew_slope);
        };
        bool ok = eval_zero_direction(1.0,
                                      0.0,
                                      rd_c1,
                                      0.0,
                                      slopes.delay_c1,
                                      slopes.slew_c1);
        ok = eval_zero_direction(0.0,
                                 0.0,
                                 rd_c2,
                                 0.0,
                                 slopes.delay_c2,
                                 slopes.slew_c2) && ok;
        ok = eval_zero_direction(0.0,
                                 1.0,
                                 0.0,
                                 0.0,
                                 slopes.delay_rpi,
                                 slopes.slew_rpi) && ok;
        ok = eval_zero_direction(0.0,
                                 0.0,
                                 rd_input_slew,
                                 1.0,
                                 slopes.delay_input_slew,
                                 slopes.slew_input_slew) && ok;
        return ok && slopes.hasFiniteValue();
    }
    DmpWaveCoeffs coeffs{};
    double current_a = nanf("");
    double current_b = nanf("");
    double current_d = nanf("");
    if (!rc.initPi(coeffs, current_a, current_b, current_d)) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailInit);
        return false;
    }

    double solve_t0 = nanf("");
    double solve_dt = nanf("");
    double solve_ceff = nanf("");
    bool solve_ok = model->findDriverParamsLocalPi(gate_arc_meta,
                                                   thresholds,
                                                   rc,
                                                   coeffs,
                                                   current_a,
                                                   current_b,
                                                   current_d,
                                                   false,
                                                   solve_t0,
                                                   solve_dt,
                                                   solve_ceff);
    if (!solve_ok && c2 > 0.0) {
        solve_ok = model->findDriverParamsLocalPi(gate_arc_meta,
                                                  thresholds,
                                                  rc,
                                                  coeffs,
                                                  current_a,
                                                  current_b,
                                                  current_d,
                                                  true,
                                                  solve_t0,
                                                  solve_dt,
                                                  solve_ceff);
    }
    if (!solve_ok || !isfinite(solve_ceff)) {
        solve_ok = routeGradRecoverCeffFromGateDelay(gate_arc_meta,
                                                     static_cast<double>(base_gate_delay),
                                                     c1 + c2,
                                                     solve_ceff);
        if (solve_ok) {
            routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiRecoveredCeffFromDelay);
        }
    }
    if (!solve_ok || !isfinite(solve_ceff)) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailForwardSolve);
        return false;
    }

    RouteGradWaveParamSlopes output_slew_slopes;
    if (!driverOutputSlewWaveParamSlopes(base_wave, thresholds, output_slew_slopes)) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailWaveSlope);
        return false;
    }

    auto eval_direction = [&] __device__ (double dc1,
                                          double dc2,
                                          double drpi,
                                          double drd,
                                          double dinput_slew,
                                          double& delay_slope,
                                          double& slew_slope) -> bool {
        RouteGradPiCoeffSlopes coeff_dir;
        if (!piCoeffDirectionSlopes(rc,
                                    coeffs,
                                    current_a,
                                    current_b,
                                    current_d,
                                    dc1,
                                    dc2,
                                    drpi,
                                    drd,
                                    coeff_dir)) {
            return false;
        }
        RouteGradPiSolveSlopes solve_dir;
        if (!piImplicitSolveDirectionSlopes(gate_arc_meta,
                                            thresholds,
                                            rc,
                                            coeffs,
                                            current_a,
                                            current_b,
                                            current_d,
                                            base_wave.t0,
                                            base_wave.dt,
                                            solve_ceff,
                                            dinput_slew,
                                            drd,
                                            coeff_dir,
                                            solve_dir)) {
            return false;
        }
        delay_slope = solve_dir.gate_delay;
        slew_slope = output_slew_slopes.dot(solve_dir.wave);
        return isfinite(delay_slope) && isfinite(slew_slope);
    };

    bool ok = eval_direction(1.0,
                             0.0,
                             0.0,
                             rd_c1,
                             0.0,
                             slopes.delay_c1,
                             slopes.slew_c1);
    ok = eval_direction(0.0,
                        1.0,
                        0.0,
                        rd_c2,
                        0.0,
                        slopes.delay_c2,
                        slopes.slew_c2) && ok;
    ok = eval_direction(0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        slopes.delay_rpi,
                        slopes.slew_rpi) && ok;
    ok = eval_direction(0.0,
                        0.0,
                        0.0,
                        rd_input_slew,
                        1.0,
                        slopes.delay_input_slew,
                        slopes.slew_input_slew) && ok;
    return ok && slopes.hasFiniteValue();
}

__device__ bool RouteGradNetPrimitiveReverse::netDriverPrimitiveCapTableSlopes(
    const RouteGradNetDriverSlopeKey& key,
    const RouteGradNetDriverWaveEval& eval,
    const DmpGateArcMeta& gate_arc_meta,
    double c1,
    double c2,
    RouteGradGatePrimitiveSlopes& slopes) const
{
    slopes = {};
    if (model == nullptr || !eval.thresholds.valid() || !gate_arc_meta.valid ||
        eval.load_pin < 0 || eval.load_pin >= model->num_pins ||
        key.attr < 0 || key.attr >= NUM_ATTR || !isfinite(c1) || !isfinite(c2)) {
        return false;
    }

    RouteGradLutSlopes cap_delay;
    RouteGradLutSlopes cap_slew;
    if (!gateArcCapDelaySlewSlopes(gate_arc_meta, c1 + c2, cap_delay, cap_slew)) {
        return false;
    }

    RouteGradLutSlopes intrinsic_delay;
    RouteGradLutSlopes intrinsic_slew;
    if (eval.has_extra_delay &&
        !gateArcCapDelaySlewSlopes(gate_arc_meta, 0.0, intrinsic_delay, intrinsic_slew)) {
        return false;
    }

    auto eval_direction = [&] __device__ (double dload,
                                          double dinput_slew,
                                          double& delay_slope,
                                          double& slew_slope) -> bool {
        const double raw_slew_slope = cap_slew.load_slope * dload +
                                      cap_slew.input_slew_slope * dinput_slew;
        double adjusted_delay = nanf("");
        double adjusted_slew = nanf("");
        thresholdAdjustedSlopes(eval.load_pin,
                                key.attr,
                                eval.thresholds.vth,
                                eval.thresholds.vl,
                                eval.thresholds.vh,
                                eval.thresholds.derate,
                                eval.thresholds.library_id,
                                0.0,
                                raw_slew_slope,
                                adjusted_delay,
                                adjusted_slew);
        double extra_delay_slope = 0.0;
        if (eval.has_extra_delay) {
            extra_delay_slope = cap_delay.load_slope * dload +
                                (cap_delay.input_slew_slope -
                                 intrinsic_delay.input_slew_slope) * dinput_slew;
        }
        delay_slope = adjusted_delay + extra_delay_slope;
        slew_slope = adjusted_slew;
        return isfinite(delay_slope) && isfinite(slew_slope);
    };

    bool ok = eval_direction(1.0, 0.0, slopes.delay_c1, slopes.slew_c1);
    ok = eval_direction(1.0, 0.0, slopes.delay_c2, slopes.slew_c2) && ok;
    slopes.delay_rpi = 0.0;
    slopes.slew_rpi = 0.0;
    if (key.input_slew_slot >= 0) {
        ok = eval_direction(0.0,
                            1.0,
                            slopes.delay_input_slew,
                            slopes.slew_input_slew) && ok;
    }
    return ok && slopes.hasFiniteValue();
}

__device__ int RouteGradNetPrimitiveReverse::classifyGatePrimitiveAlg(
    int gate_arc_id,
    int from_attr,
    int to_attr,
    int root_slot) const
{
    if (model == nullptr || root_slot < 0 || root_slot >= model->dmp_pin_slot_count ||
        model->C1 == nullptr || model->C2 == nullptr || model->r_pi == nullptr ||
        model->pinSlew == nullptr || model->arc_types == nullptr ||
        gate_arc_id < 0 || gate_arc_id >= model->num_arcs ||
        model->arc_types[gate_arc_id] != 1 ||
        from_attr < 0 || from_attr >= NUM_ATTR || to_attr < 0 || to_attr >= NUM_ATTR ||
        model->d_allocator == nullptr) {
        return -1;
    }
    const int el = to_attr >> 1;
    if ((from_attr >> 1) != el) {
        return -1;
    }
    const int from_pin = model->timing_arc_from_pin_id[gate_arc_id];
    const int timing_id = model->timing_arc_id_map[gate_arc_id * 2 + el];
    const int from_slot = from_pin * NUM_ATTR + from_attr;
    if (timing_id < 0 || from_pin < 0 || from_pin >= model->num_pins ||
        from_slot < 0 || from_slot >= model->dmp_pin_slot_count) {
        return -1;
    }
    const double c1 = static_cast<double>(model->C1[root_slot]);
    const double c2 = static_cast<double>(model->C2[root_slot]);
    const double rpi = static_cast<double>(model->r_pi[root_slot]);
    if (!isfinite(c1) || !isfinite(c2) || !isfinite(rpi) ||
        c1 < 0.0 || c2 < 0.0 || rpi < 0.0) {
        return -1;
    }
    const bool ideal_clock_arc =
        model->isIdealClockTimingArc(timing_id, from_pin) &&
        !model->d_allocator->timing_is_constraint(timing_id);
    const double input_slew = ideal_clock_arc
                                  ? static_cast<double>(model->idealClockSlew(from_pin, from_attr))
                                  : static_cast<double>(model->pinSlew[from_slot]);
    if (!isfinite(input_slew)) {
        return -1;
    }
    DmpDriverThresholds thresholds{};
    const DmpGateArcMeta gate_arc_meta =
        makeGateArcMetaForTiming(timing_id,
                                 from_attr & 1,
                                 to_attr,
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
        return -1;
    }
    return wave.alg;
}

}  // namespace gt
