#include "DmpGateModel.cuh"

namespace gt {

__device__ __forceinline__ unsigned int dmpFloatWinnerKey(float value,
                                                         bool pick_max) {
    unsigned int ordered = __float_as_uint(value);
    ordered = (ordered & 0x80000000u) ? ~ordered : (ordered ^ 0x80000000u);
    return pick_max ? ordered : ~ordered;
}

__device__ __forceinline__ void dmpAtomicSelectFloatKey(float* addr,
                                                        float value,
                                                        bool pick_max) {
    if (!isfinite(value)) {
        return;
    }
    atomicMax(reinterpret_cast<unsigned int*>(addr), dmpFloatWinnerKey(value, pick_max));
}

__device__ __forceinline__ int dmpPackDrivingCellTag(int timing_id, int input_rf) {
    return (timing_id << 1) | (input_rf & 1);
}

__device__ __forceinline__ int dmpDrivingCellTagTimingId(int tag) {
    return tag >> 1;
}

__device__ __forceinline__ int dmpDrivingCellTagInputRf(int tag) {
    return tag & 1;
}


__device__ inline void dmpEvalCapYDy(double t,
                                      double t0,
                                      double dt,
                                      double rd,
                                      double cl,
                                      double& y,
                                      double& dydt0,
                                      double& dyddt,
                                      double& dydcl) {
    // Compute the finite-ramp RC response y(t) and its Jacobian
    // with respect to t0, dt, and cl for the local Newton solve.
    const double t1 = t - t0;                                      // Time elapsed since the input ramp starts.
    if (t1 <= 0.0) {                                               // Sample point is before the input ramp.
        y = dydt0 = dyddt = dydcl = 0.0;                            // Response and all partial derivatives are zero before the ramp.
        return;
    }

    const double rd_cl = rd * cl;                                  // RC time constant tau = rd * cl.
    const double inv_rd_cl = 1.0 / rd_cl;                          // Reciprocal time constant 1 / tau for exp(-t / tau).
    const double inv_dt = 1.0 / dt;                                // Reciprocal ramp duration 1 / dt.
    const double inv_dt2 = inv_dt * inv_dt;                        // Squared reciprocal ramp duration 1 / dt^2.
    const double exp_t1 = exp2(-t1 * inv_rd_cl);                   // Fast approximation of exp(-t1 / tau).
    const double y0_t1 = t1 - rd_cl * (1.0 - exp_t1);              // RC ramp primitive y0(t1).
    const double y0dt_t1 = 1.0 - exp_t1;                           // Partial d y0(t1) / d t1.
    const double y0dcl_t1 = rd * ((1.0 + t1 * inv_rd_cl) * exp_t1 - 1.0); // Partial d y0(t1) / d cl.

    if (t1 <= dt) {                                                // Sample point is inside the input ramp.
        y = y0_t1 * inv_dt;                                       // Ramp response y = y0(t1) / dt.
        dydt0 = -y0dt_t1 * inv_dt;                                // Partial d y / d t0 = -(d y0 / d t1) / dt.
        dyddt = -y0_t1 * inv_dt2;                                 // Partial d y / d dt = -y0(t1) / dt^2.
        dydcl = y0dcl_t1 * inv_dt;                                // Partial d y / d cl = (d y0 / d cl) / dt.
        return;
    }

    const double t1_dt = t1 - dt;                                  // Time since the input ramp ended.
    const double exp_t1_dt = exp2(-t1_dt * inv_rd_cl);             // Fast approximation of exp(-(t1 - dt) / tau).
    const double y0_t1_dt = t1_dt - rd_cl * (1.0 - exp_t1_dt);     // RC ramp primitive y0(t1 - dt).
    const double y0dt_t1_dt = 1.0 - exp_t1_dt;                     // Partial d y0(t1 - dt) / d (t1 - dt).
    const double y0dcl_t1_dt = rd * ((1.0 + t1_dt * inv_rd_cl) * exp_t1_dt - 1.0); // Partial d y0(t1 - dt) / d cl.
    y = (y0_t1 - y0_t1_dt) * inv_dt;                              // Post-ramp response y = (y0(t1) - y0(t1 - dt)) / dt.
    dydt0 = -(y0dt_t1 - y0dt_t1_dt) * inv_dt;                     // Partial d y / d t0 from both y0 endpoints.
    dyddt = -(y0_t1 + y0_t1_dt) * inv_dt2 + y0dt_t1_dt * inv_dt;  // Partial d y / d dt matching the old dy() post-ramp branch.
    dydcl = (y0dcl_t1 - y0dcl_t1_dt) * inv_dt;                    // Partial d y / d cl from both y0 endpoint cl-derivatives.
}

__device__ inline double dmpIpiMinusIceff(double dt,
                                             double ceff_time,
                                             double ceff,
                                             double rd,
                                             const DmpWaveCoeffs& coeffs,
                                             double A,
                                             double B,
                                             double D)
{
    const double exp_p1_dt = exp2(-coeffs.p1 * ceff_time);
    const double exp_p2_dt = exp2(-coeffs.p2 * ceff_time);
    const double exp_dt_rd_ceff = exp2(-ceff_time / (rd * ceff));
    const double ipi = (A * ceff_time + (B / coeffs.p1) * (1.0 - exp_p1_dt) +
                        (D / coeffs.p2) * (1.0 - exp_p2_dt)) /
                       (rd * ceff_time * dt);
    const double rd_ceff = rd * ceff;
    const double iceff = (rd_ceff * ceff_time -
                          rd_ceff * rd_ceff * (1.0 - exp_dt_rd_ceff)) /
                         (rd * ceff_time * dt);
    return ipi - iceff;
}

__device__ double DmpDriverWave::findLoadCrossing(double elmore,
                                                      float vth,
                                                      double x1,
                                                      double x2,
                                                      int max_iter,
                                                      double x_tol) const {
    double y1, y2, dy;
    loadRootFunc(elmore, vth, x1, y1, dy);
    loadRootFunc(elmore, vth, x2, y2, dy);
    if (y1 * y2 > 0.0) {
        return nanf("");
    }
    if (y1 == 0.0) {
        return x1;
    }
    if (y2 == 0.0) {
        return x2;
    }
    if (y1 > 0.0) {
        const double tmp = x1;
        x1 = x2;
        x2 = tmp;
    }
    double root = (x1 + x2) / 2.0;
    double dx_prev = fabs(x2 - x1);
    double dx = dx_prev;
    double y;
    loadRootFunc(elmore, vth, root, y, dy);
    for (int iter = 0; iter < max_iter; ++iter) {
        if ((((x2 - root) * dy + y) * ((x1 - root) * dy + y) > 0.0) ||
            (fabs(2.0 * y) > fabs(dx_prev * dy))) {
            dx_prev = dx;
            dx = (x2 - x1) * 0.5;
            root = x1 + dx;
        } else {
            dx_prev = dx;
            dx = y / dy;
            root -= dx;
        }
        if (fabs(dx) <= x_tol * fabs(root)) {
            return static_cast<double>(root);
        }

        loadRootFunc(elmore, vth, root, y, dy);
        if (y < 0.0) {
            x1 = root;
        } else {
            x2 = root;
        }
    }
    return nanf("");
}

__device__ double DmpDriverWave::findLoadCrossingBisection(double elmore,
                                                               float vth,
                                                               double x1,
                                                               double x2,
                                                               int max_iter,
                                                               double x_tol) const {
    DmpDriverWave::LoadWaveValueEval wave_eval;
    wave_eval.init(*this, elmore);
    float fx1 = static_cast<float>(x1);
    float fx2 = static_cast<float>(x2);
    float y1 = wave_eval.rootValue(*this, vth, fx1);
    float y2 = wave_eval.rootValue(*this, vth, fx2);
    if (y1 * y2 > 0.0) {
        return nanf("");
    }
    if (y1 == 0.0) {
        return x1;
    }
    if (y2 == 0.0) {
        return x2;
    }
    if (y1 > 0.0) {
        const float tmp_x = fx1;
        fx1 = fx2;
        fx2 = tmp_x;
    }

#if DMP_LOAD_BISECTION_ITERS > 0
#pragma unroll 1
    for (int iter = 0; iter < DMP_LOAD_BISECTION_ITERS; ++iter) {
#else
#pragma unroll 1
    for (int iter = 0; iter < max_iter; ++iter) {
#endif
        const float mid = 0.5f * (fx1 + fx2);
        const float y = wave_eval.rootValue(*this, vth, mid);
        if (y < 0.0f) {
            fx1 = mid;
        } else {
            fx2 = mid;
        }
    }
    const float root = 0.5f * (fx1 + fx2);
    if (!isfinite(root) || fabsf(fx2 - fx1) > static_cast<float>(x_tol) * fabsf(root) * 2.0f) {
        return nanf("");
    }
    return root;
}

__device__ __noinline__ double DmpDriverWave::findDriverCrossing(float vth,
                                                        double x1,
                                                        double x2,
                                                        int max_iter,
                                                        double x_tol) const {
    dmpRootProfileVoCall();
    double y1, y2, dy;
    driverRootFunc(vth, x1, y1, dy);
    driverRootFunc(vth, x2, y2, dy);
    if (y1 * y2 > 0.0) {
        dmpRootProfileVoBracketFail();
        return nanf("");
    }
    if (y1 == 0.0) {
        dmpRootProfileVoSuccess();
        dmpRootProfileVoEndpointHit();
        return x1;
    }
    if (y2 == 0.0) {
        dmpRootProfileVoSuccess();
        dmpRootProfileVoEndpointHit();
        return x2;
    }
    if (y1 > 0.0) {
        const double tmp = x1;
        x1 = x2;
        x2 = tmp;
    }
    double root = (x1 + x2) / 2.0;
    double dx_prev = fabs(x2 - x1);
    double dx = dx_prev;
    double y;
    driverRootFunc(vth, root, y, dy);
    for (int iter = 0; iter < max_iter; ++iter) {
        if ((((x2 - root) * dy + y) * ((x1 - root) * dy + y) > 0.0) ||
            (fabs(2.0 * y) > fabs(dx_prev * dy))) {
            dx_prev = dx;
            dx = (x2 - x1) * 0.5;
            root = x1 + dx;
        } else {
            dx_prev = dx;
            dx = y / dy;
            root -= dx;
        }
        if (fabs(dx) <= x_tol * fabs(root)) {
            dmpRootProfileVoSuccess();
            dmpRootProfileVoIters(static_cast<unsigned long long>(iter + 1));
            return root;
        }

        driverRootFunc(vth, root, y, dy);
        if (y < 0.0) {
            x1 = root;
        } else {
            x2 = root;
        }
    }
    dmpRootProfileVoMaxIterFail();
    return nanf("");
}

__device__ __noinline__ bool DmpDriverWave::findDriverDelaySlew(const DmpDriverThresholds& thresholds,
                                                       int max_iter,
                                                       double x_tol,
                                                       double& delay,
                                                       double& slew) const {
    delay = nanf("");
    slew = nanf("");
    if (!hasValidDriver() || !thresholds.valid()) {
        return false;
    }
    const double t_upper = vo_upper_time;
    delay = findDriverCrossing(thresholds.vth,
                               t0,
                               t_upper,
                               max_iter,
                               x_tol);
    if (!isfinite(delay)) {
        delay = slew = nanf("");
        return false;
    }
    const double tl = findDriverCrossing(thresholds.vl,
                                         t0,
                                         delay,
                                         max_iter,
                                         x_tol);
    const double th = findDriverCrossing(thresholds.vh,
                                         delay,
                                         t_upper,
                                         max_iter,
                                         x_tol);
    if (!isfinite(tl) || !isfinite(th)) {
        delay = slew = nanf("");
        return false;
    }
    slew = (th - tl) / static_cast<double>(thresholds.derate);
    return isfinite(slew);
}

__device__ inline DmpGateArcMeta DmpModel::makeGateArcMetaForTiming(int timing_id,
                                                                           int input_rf,
                                                                           int to_attr,
                                                                           float input_slew,
                                                                           DmpDriverThresholds& thresholds) {
    DmpGateArcMeta gate_arc_meta{};
    gate_arc_meta.allocator = d_allocator;
    gate_arc_meta.delay_lut = {};
    gate_arc_meta.slew_lut = {};
    gate_arc_meta.input_slew = input_slew;
    gate_arc_meta.valid = false;

    thresholds = {};
    const int library_id = timingLibraryId(timing_id);
    double driver_vth = nan("");
    double driver_vl = nan("");
    double driver_vh = nan("");
    double driver_derate = nan("");
    driverLibraryThresholds(library_id,
                            to_attr,
                            driver_vth,
                            driver_vl,
                            driver_vh,
                            driver_derate);
    thresholds.set(driver_vth, driver_vl, driver_vh, driver_derate, library_id);

    const int output_rf = to_attr & 1;
    if (gate_arc_meta.allocator == nullptr ||
        timing_id < 0 || input_rf < 0 || to_attr < 0 ||
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


__device__ __noinline__ bool DmpModel::findDriverParamsLocalOnePole(const DmpGateArcMeta& gate_arc_meta,
                                                const DmpDriverThresholds& thresholds,
                                                const DmpRcParams& rc,
                                                double& t0,
                                                double& dt) {
    t0 = nanf("");
    dt = nanf("");
    const double fixed_ceff = rc.c1;
    double t_vth = nanf("");
    double t_vl = nanf("");
    double measured_slew = nanf("");
    gate_arc_meta.gateDelays(thresholds, fixed_ceff, t_vth, t_vl, measured_slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(fixed_ceff) || !isfinite(rc.rd) || !thresholds.valid() ||
        measured_slew <= 0.0 || fixed_ceff <= 0.0) {
        return false;
    }
    const double driver_vth = static_cast<double>(thresholds.vth);
    const double driver_vl = static_cast<double>(thresholds.vl);
    const double driver_vh = static_cast<double>(thresholds.vh);
    double x_dt = measured_slew / (driver_vh - driver_vl);
    double x_t0 = t_vth + log(1.0 - driver_vth) * rc.rd * fixed_ceff -
                  driver_vth * x_dt;
    if (!isfinite(x_t0) || !isfinite(x_dt)) {
        return false;
    }
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        if (x_dt <= 0.0) {
            x_dt = (t_vl - t_vth) / 100.0;
        }
        double y_vl, y_vth;
        double a00, a01, a02_ignore;
        double a10, a11, a12_ignore;
        dmpEvalCapYDy(t_vl, x_t0, x_dt, rc.rd, fixed_ceff,
                      y_vl, a00, a01, a02_ignore);
        dmpEvalCapYDy(t_vth, x_t0, x_dt, rc.rd, fixed_ceff,
                      y_vth, a10, a11, a12_ignore);
        const double f0 = y_vl - driver_vl;
        const double f1 = y_vth - driver_vth;
        double p0, p1;
        if (!dmpSolve2x2(a00, a01, a10, a11, -f0, -f1, p0, p1)) {
            return false;
        }
        const bool converged = fabs(p0) <= fabs(x_t0) * x_tol &&
                               fabs(p1) <= fabs(x_dt) * x_tol;
        x_t0 += p0;
        x_dt += p1;
        if (converged) {
            if (!isfinite(x_t0) || !isfinite(x_dt) || x_dt <= 0.0) {
                return false;
            }
            t0 = x_t0;
            dt = x_dt;
            return true;
        }
    }
    return false;
}

__device__ __noinline__ bool DmpModel::findDriverParamsLocalPi(const DmpGateArcMeta& gate_arc_meta,
                                           const DmpDriverThresholds& thresholds,
                                           const DmpRcParams& rc,
                                           const DmpWaveCoeffs& coeffs,
                                           double A,
                                           double B,
                                           double D,
                                           bool use_c2_initial_ceff,
                                           double& t0,
                                           double& dt,
                                           double& ceff) {
    t0 = nanf("");
    dt = nanf("");
    ceff = nanf("");
    double x_ceff = use_c2_initial_ceff ? rc.c2 : rc.c1 + rc.c2;
    double t_vth = nanf("");
    double t_vl = nanf("");
    double measured_slew = nanf("");
    gate_arc_meta.gateDelays(thresholds, x_ceff, t_vth, t_vl, measured_slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(x_ceff) || !isfinite(rc.rd) || !thresholds.valid() ||
        measured_slew <= 0.0 || x_ceff < 0.0) {
        return false;
    }
    const double driver_vth = static_cast<double>(thresholds.vth);
    const double driver_vl = static_cast<double>(thresholds.vl);
    const double driver_vh = static_cast<double>(thresholds.vh);
    double x_dt = measured_slew / (driver_vh - driver_vl);
    double x_t0 = t_vth + log(1.0 - driver_vth) * rc.rd * x_ceff -
                  driver_vth * x_dt;
    if (!isfinite(x_t0) || !isfinite(x_dt) || x_dt <= 0.0) {
        return false;
    }
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        if (x_ceff < 0.0 || x_ceff > rc.c1 + rc.c2 || x_dt <= 0.0) {
            return false;
        }
        if (iter > 0) {
            gate_arc_meta.gateDelays(thresholds, x_ceff, t_vth, t_vl, measured_slew);
            if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
                measured_slew <= 0.0) {
                return false;
            }
        }
        double ceff_time = measured_slew / (driver_vh - driver_vl);
        if (ceff_time > 1.4 * x_dt) {
            ceff_time = 1.4 * x_dt;
        }

        const double exp_p1_dt = exp2(-coeffs.p1 * x_dt);
        const double exp_p2_dt = exp2(-coeffs.p2 * x_dt);
        const double exp_dt_rd_ceff = exp2(-x_dt / (rc.rd * x_ceff));
        double y_vth, y_vl;
        double a10, a11, a12;
        double a20, a21, a22;
        dmpEvalCapYDy(t_vth, x_t0, x_dt, rc.rd, x_ceff,
                      y_vth, a10, a11, a12);
        dmpEvalCapYDy(t_vl, x_t0, x_dt, rc.rd, x_ceff,
                      y_vl, a20, a21, a22);
        const double f0 = dmpIpiMinusIceff(x_dt, ceff_time, x_ceff, rc.rd, coeffs, A, B, D);
        const double f1 = y_vth - driver_vth;
        const double f2 = y_vl - driver_vl;

        const double a01 =
            (-A * x_dt + B * x_dt * exp_p1_dt -
             (2 * B / coeffs.p1) * (1.0 - exp_p1_dt) +
             D * x_dt * exp_p2_dt -
             (2 * D / coeffs.p2) * (1.0 - exp_p2_dt) +
             rc.rd * x_ceff *
                 (x_dt + x_dt * exp_dt_rd_ceff -
                  2 * rc.rd * x_ceff * (1.0 - exp_dt_rd_ceff))) /
            (rc.rd * x_dt * x_dt * x_dt);
        const double a02 =
            (2 * rc.rd * x_ceff - x_dt -
             (2 * rc.rd * x_ceff + x_dt) * exp_dt_rd_ceff) /
            (x_dt * x_dt);

        double p0, p1, p2;
        if (!dmpSolve3x3A00Zero(a01,
                                  a02,
                                  a10,
                                  a11,
                                  a12,
                                  a20,
                                  a21,
                                  a22,
                                  -f0,
                                  -f1,
                                  -f2,
                                  p0,
                                  p1,
                                  p2)) {
            return false;
        }
        const bool converged = fabs(p0) <= fabs(x_t0) * x_tol &&
                               fabs(p1) <= fabs(x_dt) * x_tol &&
                               fabs(p2) <= fabs(x_ceff) * x_tol;
        x_t0 += p0;
        x_dt += p1;
        x_ceff += p2;
        if (converged) {
            if (!isfinite(x_t0) || !isfinite(x_dt) || !isfinite(x_ceff) ||
                x_dt <= 0.0 || x_ceff < 0.0 || x_ceff > rc.c1 + rc.c2) {
                return false;
            }
            t0 = x_t0;
            dt = x_dt;
            ceff = x_ceff;
            return true;
        }
    }
    return false;
}

__device__ inline void DmpModel::initDriverWave(DmpDriverWave& driver_wave) {
    driver_wave = {};
    driver_wave.alg = DMP_ALG_CAP;
    driver_wave.t0 = nanf("");
    driver_wave.dt = nanf("");
    driver_wave.vo_delay = nanf("");
    driver_wave.vo_slew = nanf("");
    driver_wave.vo_upper_time = nanf("");
}

__device__ __noinline__ bool DmpModel::computeZeroC2DriverWave(const DmpGateArcMeta& gate_arc_meta,
                                                                const DmpDriverThresholds& thresholds,
                                                                const DmpRcParams& rc,
                                                                DmpDriverWave& driver_wave,
                                                                float& gate_delay) {
    double c1_delay = nanf("");
    double c1_slew = nanf("");
    gate_arc_meta.capDelaySlew(rc.c1, c1_delay, c1_slew);
    DmpWaveCoeffs coeffs{};
    double t0 = nanf("");
    double dt = nanf("");
    if (!isfinite(c1_delay) || !isfinite(c1_slew) ||
        !rc.initZeroC2(coeffs) ||
        !findDriverParamsLocalOnePole(gate_arc_meta, thresholds, rc, t0, dt)) {
        return false;
    }

    driver_wave.alg = DMP_ALG_ZERO_C2;
    driver_wave.coeffs = coeffs;
    driver_wave.t0 = t0;
    driver_wave.dt = dt;
    driver_wave.vo_upper_time = rc.voUpperTime(DMP_ALG_ZERO_C2, t0, dt);
    double vo_delay = nanf("");
    double vo_slew = nanf("");
    if (!driver_wave.findDriverDelaySlew(thresholds,
                                         MAX_ITER,
                                         x_tol,
                                         vo_delay,
                                         vo_slew)) {
        return false;
    }
    gate_delay = static_cast<float>(vo_delay);
    driver_wave.vo_delay = vo_delay;
    driver_wave.vo_slew = static_cast<float>(vo_slew);
    return isfinite(gate_delay) && isfinite(driver_wave.vo_slew);
}

__device__ __noinline__ bool DmpModel::computePiDriverWave(const DmpGateArcMeta& gate_arc_meta,
                                                                const DmpDriverThresholds& thresholds,
                                                                const DmpRcParams& rc,
                                                                DmpDriverWave& driver_wave,
                                                                float& gate_delay) {
    DmpWaveCoeffs coeffs{};
    double A = nanf("");
    double B = nanf("");
    double D = nanf("");
    if (!rc.initPi(coeffs, A, B, D)) {
        return false;
    }

    double t0 = nanf("");
    double dt = nanf("");
    double ceff = nanf("");
    bool params_ok = findDriverParamsLocalPi(gate_arc_meta,
                                             thresholds,
                                             rc,
                                             coeffs,
                                             A,
                                             B,
                                             D,
                                             false,
                                             t0,
                                             dt,
                                             ceff);
    if (!params_ok && rc.c2 > 0.0) {
        params_ok = findDriverParamsLocalPi(gate_arc_meta,
                                            thresholds,
                                            rc,
                                            coeffs,
                                            A,
                                            B,
                                            D,
                                            true,
                                            t0,
                                            dt,
                                            ceff);
    }
    if (!params_ok) {
        return false;
    }

    double ceff_delay = nanf("");
    double ceff_slew = nanf("");
    gate_arc_meta.capDelaySlew(ceff, ceff_delay, ceff_slew);
    if (!isfinite(ceff_delay) || !isfinite(ceff_slew)) {
        return false;
    }

    driver_wave.alg = DMP_ALG_PI;
    driver_wave.coeffs = coeffs;
    driver_wave.t0 = t0;
    driver_wave.dt = dt;
    driver_wave.vo_upper_time = rc.voUpperTime(DMP_ALG_PI, t0, dt);
    double vo_delay = nanf("");
    double vo_slew = nanf("");
    if (!driver_wave.findDriverDelaySlew(thresholds,
                                         MAX_ITER,
                                         x_tol,
                                         vo_delay,
                                         vo_slew)) {
        return false;
    }
    gate_delay = static_cast<float>(ceff_delay);
    driver_wave.vo_delay = vo_delay;
    driver_wave.vo_slew = static_cast<float>(vo_slew);
    return isfinite(gate_delay) && isfinite(driver_wave.vo_slew);
}

__device__ __noinline__ bool DmpModel::computeGateDriverWaveForSlot(const DmpGateArcMeta& gate_arc_meta,
                                                                const DmpDriverThresholds& thresholds,
                                                                int rc_slot,
                                                                DmpDriverWave& driver_wave,
                                                                float& gate_delay) {
    gate_delay = nanf("");
    if (!gate_arc_meta.valid || !thresholds.valid() || rc_slot < 0 || rc_slot >= dmp_pin_slot_count) {
        return false;
    }

    DmpRcParams rc{};
    rc.c1 = C1[rc_slot];
    rc.c2 = C2[rc_slot];
    rc.rpi = r_pi[rc_slot];
    rc.rd = nanf("");

    double table_delay = nanf("");
    double table_slew = nanf("");
    gate_arc_meta.capDelaySlew(rc.c1 + rc.c2, table_delay, table_slew);
    if (!isfinite(table_delay) || !isfinite(table_slew)) {
        return false;
    }

    const bool rc_can_model = isfinite(rc.c1) && isfinite(rc.c2) &&
                              isfinite(rc.rpi) && isfinite(rc.c1 + rc.c2) &&
                              rc.c1 > 0.0 && rc.c2 >= 0.0 &&
                              rc.rpi > 0.0;
    bool dmp_ok = false;
    if (rc_can_model && gate_arc_meta.estimateRd(thresholds, cap_unit, rc, table_delay, rc.rd)) {
        const int alg = rc.selectAlg(static_cast<double>(res_unit));
        if (alg == DMP_ALG_ZERO_C2) {
            dmp_ok = computeZeroC2DriverWave(gate_arc_meta,
                                             thresholds,
                                             rc,
                                             driver_wave,
                                             gate_delay);
        } else if (alg == DMP_ALG_PI) {
            dmp_ok = computePiDriverWave(gate_arc_meta,
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

__device__ inline bool DmpModel::computeGateArcDriverWave(int to_attr,
                                                         int input_rf,
                                                         int output_rf,
                                                         int to_slot,
                                                         int timing_id,
                                                         float input_slew,
                                                         DmpDriverWave& driver_wave,
                                                         float& gate_delay) {
    if (timing_id < 0 || to_slot < 0 || d_allocator == nullptr || !isfinite(input_slew)) {
        initDriverWave(driver_wave);
        gate_delay = nanf("");
        return false;
    }

    DmpDriverThresholds thresholds{};
    const DmpGateArcMeta gate_arc_meta =
        makeGateArcMetaForTiming(timing_id, input_rf, to_attr, input_slew, thresholds);
    return computeGateDriverWaveForSlot(gate_arc_meta, thresholds, to_slot, driver_wave, gate_delay);
}

__device__ inline bool DmpModel::computeDrivingCellDriverWave(int pin_slot,
                                                               int attr,
                                                               int timing_id,
                                                               int input_rf,
                                                               float input_slew,
                                                               DmpDriverWave& driver_wave,
                                                               float& gate_delay) {
    if (d_allocator == nullptr || pin_slot < 0 ||
        pin_slot >= dmp_pin_slot_count ||
        timing_id < 0 || input_rf < 0 || attr < 0 ||
        !isfinite(input_slew)) {
        initDriverWave(driver_wave);
        gate_delay = nanf("");
        return false;
    }
    DmpDriverThresholds thresholds{};
    const DmpGateArcMeta gate_arc_meta =
        makeGateArcMetaForTiming(timing_id, input_rf, attr, input_slew, thresholds);
    return computeGateDriverWaveForSlot(gate_arc_meta, thresholds, pin_slot, driver_wave, gate_delay);
}

__device__ __noinline__ void DmpModel::loadDelaySlewFromDriverWave(const DmpDriverWave& driver_wave,
                                                               const DmpDriverThresholds& thresholds,
                                                               int load_pin_id,
                                                               int load_attr,
                                                               double elmore,
                                                               double& wire_delay,
                                                               double& load_slew) {
    const float drvr_slew = driver_wave.vo_slew;
    wire_delay = elmore;
    load_slew = static_cast<double>(drvr_slew);
    if (!isfinite(drvr_slew) || !isfinite(elmore)) {
        wire_delay = nanf("");
        load_slew = nanf("");
        return;
    }
    if (!thresholds.valid()) {
        wire_delay = nanf("");
        load_slew = nanf("");
        return;
    }
    if (!driver_wave.hasValidDriver() || elmore == 0.0 || elmore < static_cast<double>(drvr_slew) * 1e-3) {
        thresholdAdjust(load_pin_id, load_attr,
                               thresholds.vth, thresholds.vl, thresholds.vh, thresholds.derate,
                               thresholds.library_id,
                               wire_delay, load_slew);
        return;
    }

    const int max_iter = MAX_ITER;
    const double solve_x_tol = x_tol;
    const double t_lower = driver_wave.t0;
    const double t_upper = driver_wave.vo_upper_time + elmore * 2.0;
#if DMP_LOAD_CROSSING_BISECTION
    const double load_delay =
        driver_wave.findLoadCrossingBisection(
                            elmore,
                            thresholds.vth,
                            t_lower,
                            t_upper,
                            max_iter,
                            solve_x_tol);
    const double tl =
        driver_wave.findLoadCrossingBisection(
                            elmore,
                            thresholds.vl,
                            t_lower,
                            load_delay,
                            max_iter,
                            solve_x_tol);
    const double th =
        driver_wave.findLoadCrossingBisection(
                            elmore,
                            thresholds.vh,
                            load_delay,
                            t_upper,
                            max_iter,
                            solve_x_tol);
#else
    const double load_delay =
        driver_wave.findLoadCrossing(
                            elmore,
                            thresholds.vth,
                            t_lower,
                            t_upper,
                            max_iter,
                            solve_x_tol);
    const double tl =
        driver_wave.findLoadCrossing(
                            elmore,
                            thresholds.vl,
                            t_lower,
                            load_delay,
                            max_iter,
                            solve_x_tol);
    const double th =
        driver_wave.findLoadCrossing(
                            elmore,
                            thresholds.vh,
                            load_delay,
                            t_upper,
                            max_iter,
                            solve_x_tol);
#endif
    double delay1 = load_delay - driver_wave.vo_delay;
    double slew1 = (th - tl) / thresholds.derate;
    if (!isfinite(load_delay) || !isfinite(tl) || !isfinite(th) ||
        !isfinite(slew1) || !isfinite(delay1)) {
        return;
    }
    if (delay1 < 0.0) {
        if (-delay1 > vth_time_tol * driver_wave.vo_delay) {
            return;
        }
        delay1 = elmore;
    }
    if (slew1 < static_cast<double>(drvr_slew)) {
        if ((static_cast<double>(drvr_slew) - slew1) > vth_time_tol * static_cast<double>(drvr_slew)) {
            return;
        }
        slew1 = static_cast<double>(drvr_slew);
    }
    wire_delay = delay1;
    load_slew = slew1;
    thresholdAdjust(load_pin_id, load_attr,
                           thresholds.vth, thresholds.vl, thresholds.vh, thresholds.derate,
                           thresholds.library_id,
                           wire_delay, load_slew);
}

__device__ void DmpModel::propagateLoadSlewDelay(int arc_id,
                                                                         int attr) {
    const int from_pin_id = timing_arc_from_pin_id[arc_id];
    const int to_pin_id = timing_arc_to_pin_id[arc_id];
    const int from_slot = from_pin_id * NUM_ATTR + attr;
    const int to_slot = to_pin_id * NUM_ATTR + attr;
    const double elmore = elmore_delay[to_slot];

    float source_slew = pinSlew[from_slot];
    const bool has_driving_cell =
        at_prefix_arc != nullptr &&
        at_prefix_attr != nullptr &&
        at_prefix_attr[from_slot] == DMP_DRIVING_CELL_PREFIX_ATTR;
    if (!isfinite(source_slew)) {
        return;
    }

    double final_delay = nanf("");
    double final_slew = nanf("");
    bool used_driving_cell = false;
    double debug_extra_delay = nanf("");
#if defined(DMP_DIRECT_CLOCK_DEBUG_PRINT) && DMP_DIRECT_CLOCK_DEBUG_PRINT
    bool used_dmp_load = false;
    double debug_vo_delay = nanf("");
    int debug_alg = DMP_ALG_CAP;
#endif

    if (has_driving_cell) {
        // set_driving_cell source slots reuse the source prefix metadata:
        // at_prefix_arc stores (timing_id << 1) | input_rf, and pinSlew stores
        // the SDC input transition used to rebuild the virtual driver waveform.
        const int driving_tag = at_prefix_arc[from_slot];
        const int timing_id = dmpDrivingCellTagTimingId(driving_tag);
        const int input_rf = dmpDrivingCellTagInputRf(driving_tag);
        const int output_rf = attr & 1;
        const float input_slew = source_slew;
        DmpDriverWave driver_wave;
        DmpDriverThresholds thresholds{};
        float gate_delay = nanf("");
        {
            const DmpGateArcMeta gate_arc_meta =
                makeGateArcMetaForTiming(timing_id, input_rf, attr, input_slew, thresholds);
            if (computeGateDriverWaveForSlot(gate_arc_meta, thresholds, from_slot, driver_wave, gate_delay)) {
                double wire_delay = nanf("");
                double load_slew = nanf("");
                loadDelaySlewFromDriverWave(driver_wave,
                                            thresholds,
                                            to_pin_id,
                                            attr,
                                            elmore,
                                            wire_delay,
                                            load_slew);

                double intrinsic_delay = nanf("");
                double intrinsic_slew = nanf("");
                gateCapDelaySlew(
                                          timing_id,
                                          input_rf,
                                          output_rf,
                                          input_slew,
                                          0.0,
                                          intrinsic_delay,
                                          intrinsic_slew);
                if (isfinite(intrinsic_delay) && isfinite(gate_delay)) {
                    debug_extra_delay = static_cast<double>(gate_delay) - intrinsic_delay;
                    wire_delay += debug_extra_delay;
                }
                if (isfinite(wire_delay) && isfinite(load_slew)) {
                    final_delay = wire_delay;
                    final_slew = load_slew;
                    source_slew = static_cast<float>(driver_wave.vo_slew);
                    used_driving_cell = true;
#if defined(DMP_DIRECT_CLOCK_DEBUG_PRINT) && DMP_DIRECT_CLOCK_DEBUG_PRINT
                    used_dmp_load = driver_wave.hasValidDriver();
                    debug_vo_delay = driver_wave.vo_delay;
                    debug_alg = driver_wave.alg;
#endif
                }
            }
        }
    }

    if (!used_driving_cell) {
        if (has_driving_cell) {
            return;
        }
        double driver_vth, driver_vl, driver_vh, driver_derate;
        const int driver_library_id = pinLibraryId(from_pin_id);
        driverLibraryThresholds(
                                   driver_library_id,
                                   attr,
                                   driver_vth,
                                   driver_vl,
                                   driver_vh,
                                   driver_derate);
        final_delay = elmore;
        final_slew = source_slew;
        if (hasPinFlag(from_pin_id, DMP_PIN_PRIMARY_INPUT)) {
            inputPortDelaySlew(
                                      to_pin_id,
                                      attr,
                                      source_slew,
                                      elmore,
                                      final_delay,
                                      final_slew);
        } else {
            thresholdAdjust(
                                   to_pin_id,
                                   attr,
                                   driver_vth,
                                   driver_vl,
                                   driver_vh,
                                   driver_derate,
                                   driver_library_id,
                                   final_delay,
                                   final_slew);
        }
    }

    if (!isfinite(final_delay) || !isfinite(final_slew)) {
        return;
    }
    const int delay_idx = (attr << 1) + (attr & 1);
    const bool pick_max = (attr >> 1) != 0;
    dmpAtomicSelectFloatKey(&pinSlew[to_slot],
                         static_cast<float>(final_slew),
                         pick_max);
    dmpAtomicSelectFloatKey(&arcDelay[arc_id * 2 * NUM_ATTR + delay_idx],
                         static_cast<float>(final_delay),
                         pick_max);

#if defined(DMP_DIRECT_CLOCK_DEBUG_PRINT) && DMP_DIRECT_CLOCK_DEBUG_PRINT
    dmpDebugPrintDirectClock(this,
                             from_pin_id,
                             to_pin_id,
                             from_slot,
                             attr,
                             source_slew,
                             final_slew,
                             elmore,
                             debug_extra_delay,
                             debug_vo_delay,
                             final_delay,
                             debug_alg,
                             used_dmp_load,
                             used_driving_cell);
#endif
}

__global__ void dmpGateKernel(DmpModel* dmp_db,
                                                        const index_type* level_arc_list,
                                                        int num_level_arcs,
                                                        unsigned long long* debug_counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_pos = idx >> 3;
    if (arc_pos >= num_level_arcs) {
        return;
    }
    const int lane = idx & 0b111;
    const int gate_arc_id = level_arc_list[arc_pos];
    if (gate_arc_id < 0 || gate_arc_id >= dmp_db->num_arcs ||
        dmp_db->arc_types[gate_arc_id] != 1) {
        return;
    }

    const int el = lane >> 2;
    const int from_attr = lane >> 1;
    const int to_attr = ((lane & 0b100) >> 1) + (lane & 1);
    const int input_rf = from_attr & 1;
    const int output_rf = to_attr & 1;
    const int from_pin_id = dmp_db->timing_arc_from_pin_id[gate_arc_id];
    const int to_pin_id = dmp_db->timing_arc_to_pin_id[gate_arc_id];
    const int timing_id = dmp_db->timing_arc_id_map[gate_arc_id * 2 + el];
    const int to_slot = to_pin_id * NUM_ATTR + to_attr;

    if (dmp_db->d_allocator == nullptr ||
        timing_id < 0 ||
        from_pin_id < 0 || from_pin_id >= dmp_db->num_pins ||
        to_pin_id < 0 || to_pin_id >= dmp_db->num_pins ||
        to_slot < 0 || to_slot >= dmp_db->dmp_pin_slot_count) {
        dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_SCRATCH_SKIPS);
        return;
    }

    const bool ideal_clock_arc = dmp_db->isIdealClockTimingArc(timing_id, from_pin_id) &&
                                 !dmp_db->d_allocator->d_is_constraint[timing_id];
    const float nominal_input_slew = dmp_db->pinSlew[from_pin_id * NUM_ATTR + from_attr];
    const float input_slew = ideal_clock_arc
                                 ? dmp_db->idealClockSlew(from_pin_id, from_attr)
                                 : nominal_input_slew;

    DmpDriverWave driver_wave;
    DmpDriverThresholds thresholds{};
    float gate_delay = nanf("");
    {
        const DmpGateArcMeta gate_arc_meta =
            dmp_db->makeGateArcMetaForTiming(timing_id, input_rf, to_attr, input_slew, thresholds);
        if (!dmp_db->computeGateDriverWaveForSlot(gate_arc_meta, thresholds, to_slot, driver_wave, gate_delay)) {
            dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_SCRATCH_SKIPS);
            return;
        }
    }
    dmp_db->arcDelay[gate_arc_id * 2 * NUM_ATTR + lane] = gate_delay;
    dmpAtomicSelectFloatKey(&dmp_db->pinSlew[to_slot],
                         static_cast<float>(driver_wave.vo_slew),
                         el != 0);

    float from_at = ideal_clock_arc
                        ? dmp_db->idealClockEdgeTime(timing_id, from_pin_id)
                        : dmp_db->pinAt[from_pin_id * NUM_ATTR + from_attr];
    if (ideal_clock_arc && isnan(from_at)) {
        from_at = dmp_db->pinAt[from_pin_id * NUM_ATTR + from_attr];
    }
    if (!isnan(from_at)) {
        dmp_db->updateAtWinner(to_slot,
                               from_at + gate_delay,
                               gate_arc_id,
                               from_attr);
    }

    const int load_attr = (el << 1) | output_rf;
    for (index_type fanout_pos = dmp_db->pin_forward_arc_list_end[to_pin_id];
         fanout_pos < dmp_db->pin_forward_arc_list_end[to_pin_id + 1];
         ++fanout_pos) {
        const int net_arc_id = dmp_db->pin_forward_arc_list[fanout_pos];
        if (net_arc_id < 0 || net_arc_id >= dmp_db->num_arcs ||
            dmp_db->arc_types[net_arc_id] != 0) {
            continue;
        }
        dmpGateNetPairCount(debug_counts, DMP_GNP_TOTAL_CANDIDATES);
        const int load_to_pin_id = dmp_db->timing_arc_to_pin_id[net_arc_id];
        const int load_to_slot = load_to_pin_id * NUM_ATTR + load_attr;
        if (load_to_pin_id < 0 || load_to_pin_id >= dmp_db->num_pins
        || load_to_slot < 0 || load_to_slot >= dmp_db->dmp_pin_slot_count) {
            dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_SCRATCH_SKIPS);
            continue;
        }
        const double load_elmore = dmp_db->elmore_delay[load_to_slot];
        double wire_delay = nanf("");
        double load_slew = nanf("");
        dmp_db->loadDelaySlewFromDriverWave(driver_wave,
                                          thresholds,
                                          load_to_pin_id,
                                          load_attr,
                                          load_elmore,
                                          wire_delay,
                                          load_slew);
        if (!isfinite(wire_delay) || !isfinite(load_slew)) {
            dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_SCRATCH_SKIPS);
            continue;
        }
        dmpGateNetPairCount(debug_counts, DMP_GNP_FINITE_CANDIDATES);
        const int delay_idx = (load_attr << 1) + (load_attr & 1);
        const int delay_slot = net_arc_id * 2 * NUM_ATTR + delay_idx;
        const bool pick_max = (load_attr >> 1) != 0;
        dmpAtomicSelectFloatKey(&dmp_db->pinSlew[load_to_slot],
                             static_cast<float>(load_slew),
                             pick_max);
        dmpAtomicSelectFloatKey(&dmp_db->arcDelay[delay_slot],
                             static_cast<float>(wire_delay),
                             pick_max);
    }
}

__global__ void applyDrivingCellSourceSlewKernel(DmpModel* dmp_db,
                                                 const int* pin_ids,
                                                 const int* timing_ids,
                                                 const int* input_rfs,
                                                 const float* input_slews,
                                                 int num_sources,
                                                 unsigned long long* counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_sources * NUM_ATTR;
    if (idx >= total) {
        return;
    }

    const int source_idx = idx / NUM_ATTR;
    const int attr = idx % NUM_ATTR;
    const int timing_id = timing_ids[idx];
    const int input_rf = input_rfs[idx];
    const int pin_id = pin_ids[source_idx];
    if (pin_id < 0 || pin_id >= dmp_db->num_pins ||
        timing_id < 0 || input_rf < 0 || !isfinite(input_slews[idx])) {
        dmpDrivingCellCount(counts, DMP_DRIVING_CELL_SKIPPED);
        return;
    }

    const int pin_slot = pin_id * NUM_ATTR + attr;
    const int output_rf = attr & 1;
    DmpDriverWave driver_wave;
    float gate_delay = nanf("");
    if (!dmp_db->computeDrivingCellDriverWave(pin_slot,
                                         attr,
                                         timing_id,
                                         input_rf,
                                         input_slews[idx],
                                         driver_wave,
                                         gate_delay)) {
        dmpDrivingCellCount(counts, DMP_DRIVING_CELL_SKIPPED);
        return;
    }

    const float intrinsic_delay = dmp_db->d_allocator->query(timing_id,
                                                            input_rf,
                                                            output_rf,
                                                            input_slews[idx],
                                                            0.0f,
                                                            0);
    const double extra_delay = isfinite(intrinsic_delay)
                                   ? static_cast<double>(gate_delay) - static_cast<double>(intrinsic_delay)
                                   : nanf("");

    // Source pins do not need a timing predecessor. Reuse their traceback slot
    // for the sparse set_driving_cell tag instead of allocating a dense array.
    // pinSlew intentionally keeps the virtual cell input transition here.
    dmp_db->at_prefix_pin[pin_slot] = -1;
    dmp_db->at_prefix_arc[pin_slot] = dmpPackDrivingCellTag(timing_id, input_rf);
    dmp_db->at_prefix_attr[pin_slot] = DMP_DRIVING_CELL_PREFIX_ATTR;
    dmp_db->pinSlew[pin_slot] = input_slews[idx];

    dmpDrivingCellCount(counts, DMP_DRIVING_CELL_APPLIED);
    if (driver_wave.alg == DMP_ALG_ZERO_C2) {
        dmpDrivingCellCount(counts, DMP_DRIVING_CELL_ZERO_C2);
    } else if (driver_wave.alg == DMP_ALG_PI) {
        dmpDrivingCellCount(counts, DMP_DRIVING_CELL_PI);
    } else {
        dmpDrivingCellCount(counts, DMP_DRIVING_CELL_CAP);
    }
    if (driver_wave.hasValidDriver()) {
        dmpDrivingCellCount(counts, DMP_DRIVING_CELL_DMP_VALID);
    } else {
        dmpDrivingCellCount(counts, DMP_DRIVING_CELL_FALLBACK);
    }

#if defined(DMP_DRIVING_CELL_DEBUG_PRINT) && DMP_DRIVING_CELL_DEBUG_PRINT
    dmpDebugPrintDrivingCell(dmp_db,
                             pin_id,
                             attr,
                             input_slews[idx],
                             driver_wave,
                             gate_delay,
                             intrinsic_delay,
                             extra_delay);
#endif
}

void apply_dmp_driving_cell_source_slew_cuda(DmpModel* dmp_db,
                                             const std::vector<int>& pin_ids,
                                             const std::vector<int>& timing_ids,
                                             const std::vector<int>& input_rfs,
                                             const std::vector<float>& input_slews) {
    const int num_sources = static_cast<int>(pin_ids.size());
    const int total = num_sources * NUM_ATTR;
    const bool profile_kernels = dmp_kernel_profile_enabled();
    const bool collect_counts = profile_kernels || dmp_timing_debug_enabled();
    if (num_sources == 0 || total == 0) {
        if (collect_counts) {
            dmp_debug_print_driving_cell_counts(0, 0, nullptr);
            fflush(stdout);
        }
        return;
    }

    int* d_pin_ids = nullptr;
    int* d_timing_ids = nullptr;
    int* d_input_rfs = nullptr;
    float* d_input_slews = nullptr;
    unsigned long long* d_counts = nullptr;
    unsigned long long h_counts[DMP_DRIVING_CELL_COUNTER_COUNT] = {0};

    gpuErrchk(cudaMalloc(&d_pin_ids, sizeof(int) * num_sources));
    gpuErrchk(cudaMalloc(&d_timing_ids, sizeof(int) * total));
    gpuErrchk(cudaMalloc(&d_input_rfs, sizeof(int) * total));
    gpuErrchk(cudaMalloc(&d_input_slews, sizeof(float) * total));
    if (collect_counts) {
        gpuErrchk(cudaMalloc(&d_counts, sizeof(h_counts)));
    }
    gpuErrchk(cudaMemcpy(d_pin_ids, pin_ids.data(), sizeof(int) * num_sources, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_timing_ids, timing_ids.data(), sizeof(int) * total, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_input_rfs, input_rfs.data(), sizeof(int) * total, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_input_slews, input_slews.data(), sizeof(float) * total, cudaMemcpyHostToDevice));
    if (collect_counts) {
        gpuErrchk(cudaMemset(d_counts, 0, sizeof(h_counts)));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (profile_kernels) {
        dmp_event_create(&start, &stop);
        gpuErrchk(cudaEventRecord(start));
    }
    applyDrivingCellSourceSlewKernel<<<DMP_TIMING_BLOCK_NUMBER(total), DMP_TIMING_BLOCK_SIZE>>>(dmp_db,
                                                                                               d_pin_ids,
                                                                                               d_timing_ids,
                                                                                               d_input_rfs,
                                                                                               d_input_slews,
                                                                                               num_sources,
                                                                                               d_counts);
    gpuErrchk(cudaPeekAtLastError());
    float elapsed_ms = 0.0f;
    if (profile_kernels) {
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&elapsed_ms, start, stop));
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));
        cudaGetLastError();
    }
    if (collect_counts) {
        gpuErrchk(cudaMemcpy(h_counts, d_counts, sizeof(h_counts), cudaMemcpyDeviceToHost));

        dmp_debug_print_driving_cell_counts(num_sources, total, h_counts);
    }
    if (profile_kernels) {
        dmp_debug_print_driving_cell_kernel_profile(elapsed_ms, total);
    }
    if (collect_counts || profile_kernels) {
        fflush(stdout);
    }

    cudaFree(d_pin_ids);
    cudaFree(d_timing_ids);
    cudaFree(d_input_rfs);
    cudaFree(d_input_slews);
    if (d_counts != nullptr) {
        cudaFree(d_counts);
    }
}

} // namespace gt
