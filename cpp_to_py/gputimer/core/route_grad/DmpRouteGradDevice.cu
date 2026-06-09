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

// Shared device helpers plus PI/zero-C2 implicit solve derivatives.
__device__ bool routeGradRecoverCeffFromGateDelay(const DmpGateArcMeta& gate_arc_meta,
                                                  double target_delay,
                                                  double max_load,
                                                  double& ceff)
{
    ceff = nanf("");
    if (!gate_arc_meta.valid || !isfinite(target_delay) || !isfinite(max_load) ||
        max_load < 0.0) {
        return false;
    }
    double lo_delay = nanf("");
    double lo_slew = nanf("");
    double hi_delay = nanf("");
    double hi_slew = nanf("");
    gate_arc_meta.capDelaySlew(0.0, lo_delay, lo_slew);
    gate_arc_meta.capDelaySlew(max_load, hi_delay, hi_slew);
    if (!isfinite(lo_delay) || !isfinite(hi_delay)) {
        return false;
    }
    const double scale = fmax(1.0, fmax(fabs(lo_delay), fmax(fabs(hi_delay), fabs(target_delay))));
    const double tol = 1.0e-7 * scale;
    if (fabs(target_delay - lo_delay) <= tol) {
        ceff = 0.0;
        return true;
    }
    if (fabs(target_delay - hi_delay) <= tol) {
        ceff = max_load;
        return true;
    }
    const bool increasing = hi_delay >= lo_delay;
    const double min_delay = increasing ? lo_delay : hi_delay;
    const double max_delay = increasing ? hi_delay : lo_delay;
    if (target_delay < min_delay - tol || target_delay > max_delay + tol) {
        return false;
    }
    double lo = 0.0;
    double hi = max_load;
    for (int iter = 0; iter < 40; ++iter) {
        const double mid = 0.5 * (lo + hi);
        double mid_delay = nanf("");
        double mid_slew = nanf("");
        gate_arc_meta.capDelaySlew(mid, mid_delay, mid_slew);
        if (!isfinite(mid_delay)) {
            return false;
        }
        if ((increasing && mid_delay < target_delay) ||
            (!increasing && mid_delay > target_delay)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    ceff = 0.5 * (lo + hi);
    return isfinite(ceff) && ceff >= 0.0 && ceff <= max_load;
}

__device__ void routeGradEvalCapYDyRd(double t,
                                             double t0,
                                             double dt,
                                             double rd,
                                             double cl,
                                             double& y,
                                             double& dydt0,
                                             double& dyddt,
                                             double& dydcl,
                                             double& dydrd)
{
    const double t1 = t - t0;
    if (t1 <= 0.0) {
        y = 0.0;
        dydt0 = 0.0;
        dyddt = 0.0;
        dydcl = 0.0;
        dydrd = 0.0;
        return;
    }

    const double rd_cl = rd * cl;
    const double inv_rd_cl = 1.0 / rd_cl;
    const double inv_dt = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;
    const double exp_t1 = exp2(-t1 * inv_rd_cl);
    const double y0_t1 = t1 - rd_cl * (1.0 - exp_t1);
    const double y0dt_t1 = 1.0 - exp_t1;
    const double y0dcl_t1 = rd * ((1.0 + t1 * inv_rd_cl) * exp_t1 - 1.0);
    const double y0drd_t1 = cl * ((1.0 + t1 * inv_rd_cl) * exp_t1 - 1.0);

    if (t1 <= dt) {
        y = y0_t1 * inv_dt;
        dydt0 = -y0dt_t1 * inv_dt;
        dyddt = -y0_t1 * inv_dt2;
        dydcl = y0dcl_t1 * inv_dt;
        dydrd = y0drd_t1 * inv_dt;
        return;
    }

    const double t1_dt = t1 - dt;
    const double exp_t1_dt = exp2(-t1_dt * inv_rd_cl);
    const double y0_t1_dt = t1_dt - rd_cl * (1.0 - exp_t1_dt);
    const double y0dt_t1_dt = 1.0 - exp_t1_dt;
    const double y0dcl_t1_dt = rd * ((1.0 + t1_dt * inv_rd_cl) * exp_t1_dt - 1.0);
    const double y0drd_t1_dt = cl * ((1.0 + t1_dt * inv_rd_cl) * exp_t1_dt - 1.0);
    y = (y0_t1 - y0_t1_dt) * inv_dt;
    dydt0 = -(y0dt_t1 - y0dt_t1_dt) * inv_dt;
    dyddt = -(y0_t1 + y0_t1_dt) * inv_dt2 + y0dt_t1_dt * inv_dt;
    dydcl = (y0dcl_t1 - y0dcl_t1_dt) * inv_dt;
    dydrd = (y0drd_t1 - y0drd_t1_dt) * inv_dt;
}

__device__ double routeGradIpiMinusIceffDirectionSlope(double dt,
                                                       double ceff_time,
                                                       double ceff,
                                                       double rd,
                                                       const DmpWaveCoeffs& coeffs,
                                                       double current_a,
                                                       double current_b,
                                                       double current_d,
                                                       const RouteGradPiCoeffSlopes& coeff_slopes,
                                                       double ddt,
                                                       double dceff_time,
                                                       double dceff,
                                                       double drd)
{
    const double t = ceff_time;
    if (!isfinite(dt) || !isfinite(t) || !isfinite(ceff) || !isfinite(rd) ||
        dt <= 0.0 || t <= 0.0 || ceff <= 0.0 || rd <= 0.0 ||
        coeffs.p1 == 0.0 || coeffs.p2 == 0.0) {
        return nanf("");
    }

    const double p1 = coeffs.p1;
    const double p2 = coeffs.p2;
    const double exp_p1_t = exp2(-p1 * t);
    const double exp_p2_t = exp2(-p2 * t);
    const double ipi_num = current_a * t +
                           (current_b / p1) * (1.0 - exp_p1_t) +
                           (current_d / p2) * (1.0 - exp_p2_t);
    const double ipi_den = rd * t * dt;
    const double ipi = ipi_num / ipi_den;

    const double dp1 = coeff_slopes.wave.p1;
    const double dp2 = coeff_slopes.wave.p2;
    const double done_minus_exp1 = exp_p1_t * (t * dp1 + p1 * dceff_time);
    const double done_minus_exp2 = exp_p2_t * (t * dp2 + p2 * dceff_time);
    const double d_b_over_p1 = coeff_slopes.current_b / p1 - current_b * dp1 / (p1 * p1);
    const double d_d_over_p2 = coeff_slopes.current_d / p2 - current_d * dp2 / (p2 * p2);
    const double dipi_num = coeff_slopes.current_a * t + current_a * dceff_time +
                            d_b_over_p1 * (1.0 - exp_p1_t) +
                            (current_b / p1) * done_minus_exp1 +
                            d_d_over_p2 * (1.0 - exp_p2_t) +
                            (current_d / p2) * done_minus_exp2;
    const double dipi_den = drd * t * dt + rd * dceff_time * dt + rd * t * ddt;
    const double dipi = dipi_num / ipi_den - ipi * dipi_den / ipi_den;

    const double tau = rd * ceff;
    const double dtau = drd * ceff + rd * dceff;
    const double exp_tau = exp2(-t / tau);
    const double one_minus_exp_tau = 1.0 - exp_tau;
    const double iceff_num = tau * t - tau * tau * one_minus_exp_tau;
    const double iceff_den = rd * t * dt;
    const double iceff = iceff_num / iceff_den;
    const double done_minus_exp_tau = exp_tau * (dceff_time / tau - t * dtau / (tau * tau));
    const double diceff_num = dtau * t + tau * dceff_time -
                              2.0 * tau * dtau * one_minus_exp_tau -
                              tau * tau * done_minus_exp_tau;
    const double diceff_den = drd * t * dt + rd * dceff_time * dt + rd * t * ddt;
    const double diceff = diceff_num / iceff_den - iceff * diceff_den / iceff_den;
    return dipi - diceff;
}

// Compact device-side derivative containers. They are deliberately split by
// purpose so large kernels do not keep one oversized live variable set.
__device__ void RouteGradWaveParamSlopes::clear()
{
    t0 = 0.0;
    dt = 0.0;
    k0 = 0.0;
    k1 = 0.0;
    k2 = 0.0;
    k3 = 0.0;
    k4 = 0.0;
    p1 = 0.0;
    p2 = 0.0;
}

__device__ void RouteGradWaveParamSlopes::scale(double factor)
{
    t0 *= factor;
    dt *= factor;
    k0 *= factor;
    k1 *= factor;
    k2 *= factor;
    k3 *= factor;
    k4 *= factor;
    p1 *= factor;
    p2 *= factor;
}

__device__ double RouteGradWaveParamSlopes::dot(const RouteGradWaveParamSlopes& rhs) const
{
    return t0 * rhs.t0 + dt * rhs.dt + k0 * rhs.k0 + k1 * rhs.k1 +
           k2 * rhs.k2 + k3 * rhs.k3 + k4 * rhs.k4 + p1 * rhs.p1 +
           p2 * rhs.p2;
}

__device__ void RouteGradPiCoeffSlopes::clear()
{
    wave.clear();
    current_a = 0.0;
    current_b = 0.0;
    current_d = 0.0;
}

__device__ void RouteGradPiSolveSlopes::clear()
{
    wave.clear();
    ceff = 0.0;
    gate_delay = 0.0;
}

__device__ void RouteGradOnePoleSolveSlopes::clear()
{
    wave.clear();
}

static constexpr unsigned int kRouteGradLutMetaScalar = 1u << 0;
static constexpr unsigned int kRouteGradLutMetaVar0IsSlew = 1u << 1;

static constexpr int kRouteGradNetKeyDirectDrivingCell = 1;
static constexpr int kRouteGradNetKeyGateNetPair = 2;


// Analytic branch-local solve derivatives for DMP driver models. ZERO_C2 uses a
// one-pole implicit solve; PI uses coefficient derivatives plus the final
// implicit system J_x dx = -F_p dp for (t0, dt, ceff).
__device__ bool RouteGradNetPrimitiveReverse::zeroC2CoeffDirectionSlopes(
    const DmpRcParams& rc,
    const DmpWaveCoeffs& coeffs,
    double dc1,
    double drpi,
    double drd,
    RouteGradWaveParamSlopes& slopes) const
{
    slopes.clear();
    const double c1 = rc.c1;
    const double rpi = rc.rpi;
    const double rd = rc.rd;
    const double rd_rpi = rd + rpi;
    const double z1 = 1.0 / (rpi * c1);
    if (!isfinite(c1) || !isfinite(rpi) || !isfinite(rd) ||
        !isfinite(rd_rpi) || !isfinite(z1) || c1 <= 0.0 ||
        rpi <= 0.0 || rd <= 0.0 || rd_rpi == 0.0 ||
        coeffs.p1 == 0.0 || coeffs.k0 == 0.0) {
        return false;
    }
    const double dz1 = -z1 * (drpi / rpi + dc1 / c1);
    const double dp1 = -coeffs.p1 * (dc1 / c1 + (drd + drpi) / rd_rpi);
    const double dk0 = dp1 / z1 - coeffs.k0 * dz1 / z1;
    const double dk2 = -coeffs.k2 * dk0 / coeffs.k0;
    const double k1_den = coeffs.p1 * coeffs.p1;
    const double dk1_num = dp1 - dz1;
    const double dk1_den = 2.0 * coeffs.p1 * dp1;
    const double dk1 = dk1_num / k1_den - coeffs.k1 * dk1_den / k1_den;
    slopes.k0 = dk0;
    slopes.k1 = dk1;
    slopes.k2 = dk2;
    slopes.k3 = -dk1;
    slopes.k4 = 0.0;
    slopes.p1 = dp1;
    slopes.p2 = 0.0;
    return isfinite(slopes.k0) && isfinite(slopes.k1) &&
           isfinite(slopes.k2) && isfinite(slopes.k3) && isfinite(slopes.p1);
}

__device__ bool RouteGradNetPrimitiveReverse::onePoleImplicitSolveDirectionSlopes(
    const DmpGateArcMeta& gate_arc_meta,
    const DmpDriverThresholds& thresholds,
    const DmpRcParams& rc,
    double t0,
    double dt,
    double dc1,
    double drd,
    double dinput_slew,
    const RouteGradWaveParamSlopes& coeff_slopes,
    RouteGradOnePoleSolveSlopes& solve_slopes) const
{
    solve_slopes.clear();
    if (!gate_arc_meta.valid || !thresholds.valid() || !isfinite(t0) ||
        !isfinite(dt) || !isfinite(rc.c1) || !isfinite(rc.rd) ||
        dt <= 0.0 || rc.c1 <= 0.0 || rc.rd <= 0.0) {
        return false;
    }

    RouteGradLutSlopes delay_lut;
    RouteGradLutSlopes slew_lut;
    if (!gateArcCapDelaySlewSlopes(gate_arc_meta, rc.c1, delay_lut, slew_lut)) {
        return false;
    }
    const double driver_vth = static_cast<double>(thresholds.vth);
    const double driver_vl = static_cast<double>(thresholds.vl);
    const double driver_vh = static_cast<double>(thresholds.vh);
    const double driver_delta = driver_vh - driver_vl;
    if (!isfinite(driver_delta) || driver_delta <= 0.0) {
        return false;
    }
    const double t_vth = delay_lut.value;
    const double measured_slew = slew_lut.value * static_cast<double>(thresholds.derate);
    if (!isfinite(t_vth) || !isfinite(measured_slew) || measured_slew <= 0.0) {
        return false;
    }
    const double lower_beta = (driver_vth - driver_vl) / driver_delta;
    const double t_vl = t_vth - measured_slew * lower_beta;
    const double d_t_vth = delay_lut.input_slew_slope * dinput_slew +
                           delay_lut.load_slope * dc1;
    const double d_measured_slew = (slew_lut.input_slew_slope * dinput_slew +
                                    slew_lut.load_slope * dc1) *
                                   static_cast<double>(thresholds.derate);
    const double d_t_vl = d_t_vth - d_measured_slew * lower_beta;

    double y_vl = nanf("");
    double y_vth = nanf("");
    double a00 = nanf("");
    double a01 = nanf("");
    double a02 = nanf("");
    double a10 = nanf("");
    double a11 = nanf("");
    double a12 = nanf("");
    double yrd_vl = nanf("");
    double yrd_vth = nanf("");
    routeGradEvalCapYDyRd(t_vl, t0, dt, rc.rd, rc.c1, y_vl, a00, a01, a02, yrd_vl);
    routeGradEvalCapYDyRd(t_vth, t0, dt, rc.rd, rc.c1, y_vth, a10, a11, a12, yrd_vth);
    (void)y_vl;
    (void)y_vth;
    const double f0_param = -a00 * d_t_vl + a02 * dc1 + yrd_vl * drd;
    const double f1_param = -a10 * d_t_vth + a12 * dc1 + yrd_vth * drd;
    if (!isfinite(a00) || !isfinite(a01) || !isfinite(a10) || !isfinite(a11) ||
        !isfinite(f0_param) || !isfinite(f1_param)) {
        return false;
    }

    double dt0 = nanf("");
    double ddt = nanf("");
    if (!dmpSolve2x2(a00, a01, a10, a11, -f0_param, -f1_param, dt0, ddt)) {
        return false;
    }
    solve_slopes.wave = coeff_slopes;
    solve_slopes.wave.t0 = dt0;
    solve_slopes.wave.dt = ddt;
    return isfinite(solve_slopes.wave.t0) && isfinite(solve_slopes.wave.dt);
}

__device__ bool RouteGradNetPrimitiveReverse::piCoeffDirectionSlopes(
    const DmpRcParams& rc,
    const DmpWaveCoeffs& coeffs,
    double current_a,
    double current_b,
    double current_d,
    double dc1,
    double dc2,
    double drpi,
    double drd,
    RouteGradPiCoeffSlopes& slopes) const
{
    slopes.clear();
    const double c1 = rc.c1;
    const double c2 = rc.c2;
    const double rpi = rc.rpi;
    const double rd = rc.rd;
    const double z1 = 1.0 / (rpi * c1);
    const double k0 = coeffs.k0;
    const double a = rpi * rd * c1 * c2;
    const double b = rd * (c1 + c2) + rpi * c1;
    const double disc = b * b - 4.0 * a;
    if (!isfinite(c1) || !isfinite(c2) || !isfinite(rpi) || !isfinite(rd) ||
        !isfinite(z1) || !isfinite(k0) || !isfinite(a) || !isfinite(b) ||
        !isfinite(disc) || c1 <= 0.0 || c2 <= 0.0 || rpi <= 0.0 ||
        rd <= 0.0 || a == 0.0 || disc <= 0.0) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailCoeff);
        return false;
    }
    const double sqrt_disc = sqrt(disc);
    const double p1 = coeffs.p1;
    const double p2 = coeffs.p2;
    const double p1p2 = p1 * p2;
    const double ctot = c1 + c2;
    if (!isfinite(p1p2) || !isfinite(ctot) || p1p2 == 0.0 || p1 == p2 ||
        ctot == 0.0) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailCoeff);
        return false;
    }

    const double dz1 = -z1 * (drpi / rpi + dc1 / c1);
    const double dk0 = -k0 * (drd / rd + dc2 / c2);
    const double da = a * (drpi / rpi + drd / rd + dc1 / c1 + dc2 / c2);
    const double db = drd * (c1 + c2) + rd * (dc1 + dc2) +
                      drpi * c1 + rpi * dc1;
    const double ddisc = 2.0 * b * db - 4.0 * da;
    const double dsqrt = ddisc / (2.0 * sqrt_disc);
    const double dp1 = (db + dsqrt) / (2.0 * a) - p1 * da / a;
    const double dp2 = (db - dsqrt) / (2.0 * a) - p2 * da / a;
    const double dp1p2 = dp1 * p2 + p1 * dp2;
    const double dk2 = dz1 / p1p2 - coeffs.k2 * dp1p2 / p1p2;
    const double psum = p1 + p2;
    const double dpsum = dp1 + dp2;
    const double dk1_num = -dk2 * psum - coeffs.k2 * dpsum;
    const double dk1 = dk1_num / p1p2 - coeffs.k1 * dp1p2 / p1p2;
    const double dk4_num = dk1 * p1 + coeffs.k1 * dp1 + dk2;
    const double k4_den = p2 - p1;
    const double dk4_den = dp2 - dp1;
    const double dk4 = dk4_num / k4_den - coeffs.k4 * dk4_den / k4_den;
    const double dk3 = -dk1 - dk4;

    const double z = ctot / (rpi * c1 * c2);
    const double dz = z * ((dc1 + dc2) / ctot - drpi / rpi - dc1 / c1 - dc2 / c2);
    const double dcurrent_a = dz / p1p2 - current_a * dp1p2 / p1p2;
    const double b_den = p1 * (p1 - p2);
    const double d_den_b = dp1 * (p1 - p2) + p1 * (dp1 - dp2);
    const double dcurrent_b = (dz - dp1) / b_den - current_b * d_den_b / b_den;
    const double d_den_d = dp2 * (p2 - p1) + p2 * (dp2 - dp1);
    const double d_den = p2 * (p2 - p1);
    const double dcurrent_d = (dz - dp2) / d_den - current_d * d_den_d / d_den;

    slopes.wave.k0 = dk0;
    slopes.wave.k1 = dk1;
    slopes.wave.k2 = dk2;
    slopes.wave.k3 = dk3;
    slopes.wave.k4 = dk4;
    slopes.wave.p1 = dp1;
    slopes.wave.p2 = dp2;
    slopes.current_a = dcurrent_a;
    slopes.current_b = dcurrent_b;
    slopes.current_d = dcurrent_d;
    const bool ok = isfinite(slopes.wave.k0) && isfinite(slopes.wave.k1) &&
                    isfinite(slopes.wave.k2) && isfinite(slopes.wave.k3) &&
                    isfinite(slopes.wave.k4) && isfinite(slopes.wave.p1) &&
                    isfinite(slopes.wave.p2) && isfinite(slopes.current_a) &&
                    isfinite(slopes.current_b) && isfinite(slopes.current_d);
    if (!ok) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailCoeff);
    }
    return ok;
}

__device__ bool RouteGradNetPrimitiveReverse::piImplicitSolveDirectionSlopes(
    const DmpGateArcMeta& gate_arc_meta,
    const DmpDriverThresholds& thresholds,
    const DmpRcParams& rc,
    const DmpWaveCoeffs& coeffs,
    double current_a,
    double current_b,
    double current_d,
    double t0,
    double dt,
    double ceff,
    double dinput_slew,
    double drd,
    const RouteGradPiCoeffSlopes& coeff_slopes,
    RouteGradPiSolveSlopes& solve_slopes) const
{
    solve_slopes.clear();
    if (!gate_arc_meta.valid || !thresholds.valid() || !isfinite(t0) ||
        !isfinite(dt) || !isfinite(ceff) || !isfinite(rc.rd) ||
        dt <= 0.0 || ceff <= 0.0 || rc.rd <= 0.0) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailImplicitSetup);
        return false;
    }

    RouteGradLutSlopes delay_lut;
    RouteGradLutSlopes slew_lut;
    if (!gateArcCapDelaySlewSlopes(gate_arc_meta, ceff, delay_lut, slew_lut)) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailImplicitLut);
        return false;
    }
    const double driver_vth = static_cast<double>(thresholds.vth);
    const double driver_vl = static_cast<double>(thresholds.vl);
    const double driver_vh = static_cast<double>(thresholds.vh);
    const double driver_delta = driver_vh - driver_vl;
    if (!isfinite(driver_delta) || driver_delta <= 0.0) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailImplicitSetup);
        return false;
    }

    const double t_vth = delay_lut.value;
    const double measured_slew = slew_lut.value * static_cast<double>(thresholds.derate);
    if (!isfinite(t_vth) || !isfinite(measured_slew) || measured_slew <= 0.0) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailImplicitSetup);
        return false;
    }
    const double lower_beta = (driver_vth - driver_vl) / driver_delta;
    const double t_vl = t_vth - measured_slew * lower_beta;
    const double d_t_vth = delay_lut.input_slew_slope * dinput_slew;
    const double d_measured_slew = slew_lut.input_slew_slope *
                                   dinput_slew * static_cast<double>(thresholds.derate);
    const double d_t_vl = d_t_vth - d_measured_slew * lower_beta;
    const double dtvth_dceff = delay_lut.load_slope;
    const double dmeasured_dceff = slew_lut.load_slope * static_cast<double>(thresholds.derate);
    const double dtvl_dceff = dtvth_dceff - dmeasured_dceff * lower_beta;

    double ceff_time = measured_slew / driver_delta;
    double d_ceff_time = d_measured_slew / driver_delta;
    double dceff_time_ddt = 0.0;
    double dceff_time_dceff = dmeasured_dceff / driver_delta;
    if (ceff_time > 1.4 * dt) {
        ceff_time = 1.4 * dt;
        d_ceff_time = 0.0;
        dceff_time_ddt = 1.4;
        dceff_time_dceff = 0.0;
    }

    double y_vth = nanf("");
    double y_vl = nanf("");
    double a10 = nanf("");
    double a11 = nanf("");
    double a12 = nanf("");
    double a20 = nanf("");
    double a21 = nanf("");
    double a22 = nanf("");
    double yrd_vth = nanf("");
    double yrd_vl = nanf("");
    routeGradEvalCapYDyRd(t_vth, t0, dt, rc.rd, ceff, y_vth, a10, a11, a12, yrd_vth);
    routeGradEvalCapYDyRd(t_vl, t0, dt, rc.rd, ceff, y_vl, a20, a21, a22, yrd_vl);
    (void)y_vth;
    (void)y_vl;

    RouteGradPiCoeffSlopes zero_coeff_slopes;
    zero_coeff_slopes.clear();
    const double a01 = routeGradIpiMinusIceffDirectionSlope(dt,
                                                            ceff_time,
                                                            ceff,
                                                            rc.rd,
                                                            coeffs,
                                                            current_a,
                                                            current_b,
                                                            current_d,
                                                            zero_coeff_slopes,
                                                            1.0,
                                                            dceff_time_ddt,
                                                            0.0,
                                                            0.0);
    const double a02 = routeGradIpiMinusIceffDirectionSlope(dt,
                                                            ceff_time,
                                                            ceff,
                                                            rc.rd,
                                                            coeffs,
                                                            current_a,
                                                            current_b,
                                                            current_d,
                                                            zero_coeff_slopes,
                                                            0.0,
                                                            dceff_time_dceff,
                                                            1.0,
                                                            0.0);
    const double f0_param = routeGradIpiMinusIceffDirectionSlope(dt,
                                                                 ceff_time,
                                                                 ceff,
                                                                 rc.rd,
                                                                 coeffs,
                                                                 current_a,
                                                                 current_b,
                                                                 current_d,
                                                                 coeff_slopes,
                                                                 0.0,
                                                                 d_ceff_time,
                                                                 0.0,
                                                                 drd);
    const double f1_param = -a10 * d_t_vth + yrd_vth * drd;

    const double f2_param = -a20 * d_t_vl + yrd_vl * drd;
    const double a12_total = a12 - a10 * dtvth_dceff;
    const double a22_total = a22 - a20 * dtvl_dceff;
    if (!isfinite(a01) || !isfinite(a02) || !isfinite(a10) || !isfinite(a11) ||
        !isfinite(a12_total) || !isfinite(a20) || !isfinite(a21) ||
        !isfinite(a22_total) || !isfinite(f0_param) || !isfinite(f1_param) ||
        !isfinite(f2_param)) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailImplicitEquation);
        return false;
    }

    double dt0 = nanf("");
    double ddt = nanf("");
    double dceff = nanf("");
    if (!dmpSolve3x3A00Zero(a01,
                            a02,
                            a10,
                            a11,
                            a12_total,
                            a20,
                            a21,
                            a22_total,
                            -f0_param,
                            -f1_param,
                            -f2_param,
                            dt0,
                            ddt,
                            dceff)) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailImplicitSolve);
        return false;
    }

    solve_slopes.wave = coeff_slopes.wave;
    solve_slopes.wave.t0 = dt0;
    solve_slopes.wave.dt = ddt;
    solve_slopes.ceff = dceff;
    solve_slopes.gate_delay = delay_lut.input_slew_slope * dinput_slew +
                              delay_lut.load_slope * dceff;
    const bool ok = isfinite(solve_slopes.wave.t0) && isfinite(solve_slopes.wave.dt) &&
                    isfinite(solve_slopes.ceff) && isfinite(solve_slopes.gate_delay);
    if (!ok) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailImplicitOutput);
    }
    return ok;
}

__device__ bool RouteGradNetPrimitiveReverse::piCoeffRootParamSlopes(
    const DmpGateArcMeta& gate_arc_meta,
    const DmpDriverThresholds& thresholds,
    double c1,
    double c2,
    double rpi,
    RouteGradWaveParamSlopes& c1_slopes,
    RouteGradWaveParamSlopes& c2_slopes,
    RouteGradWaveParamSlopes& rpi_slopes,
    RouteGradWaveParamSlopes& input_slew_slopes) const
{
    c1_slopes.clear();
    c2_slopes.clear();
    rpi_slopes.clear();
    input_slew_slopes.clear();
    double rd = nanf("");
    double rd_c1 = 0.0;
    double rd_c2 = 0.0;
    double rd_input = 0.0;
    if (!estimateRdWithSlopes(gate_arc_meta,
                              thresholds,
                              c1,
                              c2,
                              rd,
                              rd_c1,
                              rd_c2,
                              rd_input)) {
        return false;
    }
    DmpRcParams rc{};
    rc.c1 = c1;
    rc.c2 = c2;
    rc.rpi = rpi;
    rc.rd = rd;
    DmpWaveCoeffs coeffs{};
    double A = nanf("");
    double B = nanf("");
    double D = nanf("");
    if (!rc.initPi(coeffs, A, B, D)) {
        return false;
    }

    auto fill_one = [&] __device__ (double dc1,
                                    double dc2,
                                    double drpi,
                                    double drd,
                                    RouteGradWaveParamSlopes& out) -> bool {
        out.clear();
        const double z1 = 1.0 / (rpi * c1);
        const double k0 = coeffs.k0;
        const double a = rpi * rd * c1 * c2;
        const double b = rd * (c1 + c2) + rpi * c1;
        const double disc = b * b - 4.0 * a;
        if (!isfinite(z1) || !isfinite(k0) || !isfinite(a) || !isfinite(b) ||
            !isfinite(disc) || a == 0.0 || disc <= 0.0 || c1 == 0.0 ||
            c2 == 0.0 || rpi == 0.0 || rd == 0.0) {
            return false;
        }
        const double sqrt_disc = sqrt(disc);
        const double p1 = coeffs.p1;
        const double p2 = coeffs.p2;
        const double p1p2 = p1 * p2;
        if (!isfinite(p1p2) || p1p2 == 0.0 || p1 == p2) {
            return false;
        }
        const double dz1 = -z1 * (drpi / rpi + dc1 / c1);
        const double dk0 = -k0 * (drd / rd + dc2 / c2);
        const double da = a * (drpi / rpi + drd / rd + dc1 / c1 + dc2 / c2);
        const double db = drd * (c1 + c2) + rd * (dc1 + dc2) +
                          drpi * c1 + rpi * dc1;
        const double ddisc = 2.0 * b * db - 4.0 * da;
        const double dsqrt = ddisc / (2.0 * sqrt_disc);
        const double dp1 = (db + dsqrt) / (2.0 * a) - p1 * da / a;
        const double dp2 = (db - dsqrt) / (2.0 * a) - p2 * da / a;
        const double dp1p2 = dp1 * p2 + p1 * dp2;
        const double dk2 = dz1 / p1p2 - coeffs.k2 * dp1p2 / p1p2;
        const double psum = p1 + p2;
        const double dpsum = dp1 + dp2;
        const double dk1_num = -dk2 * psum - coeffs.k2 * dpsum;
        const double dk1 = dk1_num / p1p2 - coeffs.k1 * dp1p2 / p1p2;
        const double dk4_num = dk1 * p1 + coeffs.k1 * dp1 + dk2;
        const double k4_den = p2 - p1;
        const double dk4_den = dp2 - dp1;
        const double dk4 = dk4_num / k4_den - coeffs.k4 * dk4_den / k4_den;
        const double dk3 = -dk1 - dk4;
        out.k0 = dk0;
        out.k1 = dk1;
        out.k2 = dk2;
        out.k3 = dk3;
        out.k4 = dk4;
        out.p1 = dp1;
        out.p2 = dp2;
        return isfinite(out.k0) && isfinite(out.k1) && isfinite(out.k2) &&
               isfinite(out.k3) && isfinite(out.k4) && isfinite(out.p1) &&
               isfinite(out.p2);
    };

    return fill_one(1.0, 0.0, 0.0, rd_c1, c1_slopes) &&
           fill_one(0.0, 1.0, 0.0, rd_c2, c2_slopes) &&
           fill_one(0.0, 0.0, 1.0, 0.0, rpi_slopes) &&
           fill_one(0.0, 0.0, 0.0, rd_input, input_slew_slopes);
}

}  // namespace gt
