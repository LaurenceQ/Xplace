__device__ __forceinline__ bool dmpSolve2x2(double a00,
                                            double a01,
                                            double a10,
                                            double a11,
                                            double b0,
                                            double b1,
                                            double& x0,
                                            double& x1) {
    const double det = a00 * a11 - a01 * a10;
    const double scale = fabs(a00 * a11) + fabs(a01 * a10) + 1e-300;
    if (!isfinite(det) || fabs(det) <= 1e-14 * scale) {
        return false;
    }
    x0 = (b0 * a11 - a01 * b1) / det;
    x1 = (a00 * b1 - b0 * a10) / det;
    return isfinite(x0) && isfinite(x1);
}

__device__ __forceinline__ bool dmpSolve3x3(double a00,
                                            double a01,
                                            double a02,
                                            double a10,
                                            double a11,
                                            double a12,
                                            double a20,
                                            double a21,
                                            double a22,
                                            double b0,
                                            double b1,
                                            double b2,
                                            double& x0,
                                            double& x1,
                                            double& x2) {
    const double c00 = a11 * a22 - a12 * a21;
    const double c01 = a10 * a22 - a12 * a20;
    const double c02 = a10 * a21 - a11 * a20;
    const double det = a00 * c00 - a01 * c01 + a02 * c02;
    const double scale = fabs(a00 * c00) + fabs(a01 * c01) + fabs(a02 * c02) + 1e-300;
    if (!isfinite(det) || fabs(det) <= 1e-14 * scale) {
        return false;
    }
    const double det0 = b0 * c00 - a01 * (b1 * a22 - a12 * b2) +
                        a02 * (b1 * a21 - a11 * b2);
    const double det1 = a00 * (b1 * a22 - a12 * b2) -
                        b0 * c01 +
                        a02 * (a10 * b2 - b1 * a20);
    const double det2 = a00 * (a11 * b2 - b1 * a21) -
                        a01 * (a10 * b2 - b1 * a20) +
                        b0 * c02;
    x0 = det0 / det;
    x1 = det1 / det;
    x2 = det2 / det;
    return isfinite(x0) && isfinite(x1) && isfinite(x2);
}

__device__ __forceinline__ float dmpFastExp2f(float x) {
    if (x < -12.0f) {
        return 0.0f;
    }
    float y = 1.0f + x / 4096.0f;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    y *= y;
    return y;
}

__device__ __forceinline__ float dmpY0f(float t, float rd, float cl) {
    return t - rd * cl * (1.0f - dmpFastExp2f(-t / (rd * cl)));
}

__device__ __forceinline__ float dmpYf(float t,
                                       float t0,
                                       float dt,
                                       float rd,
                                       float cl) {
    const float t1 = t - t0;
    if (t1 <= 0.0f) {
        return 0.0f;
    }
    if (t1 <= dt) {
        return dmpY0f(t1, rd, cl) / dt;
    }
    return (dmpY0f(t1, rd, cl) - dmpY0f(t1 - dt, rd, cl)) / dt;
}

__device__ __forceinline__ float dmpY0dtf(float t, float rd, float cl) {
    return 1.0f - dmpFastExp2f(-t / (rd * cl));
}

__device__ __forceinline__ float dmpY0dclf(float t, float rd, float cl) {
    return rd * ((1.0f + t / (rd * cl)) * dmpFastExp2f(-t / (rd * cl)) - 1.0f);
}

__device__ __forceinline__ void dmpDyf(float t,
                                       float t0,
                                       float dt,
                                       float rd,
                                       float cl,
                                       float& dydt0,
                                       float& dyddt,
                                       float& dydcl) {
    const float t1 = t - t0;
    if (t1 <= 0.0f) {
        dydt0 = 0.0f;
        dyddt = 0.0f;
        dydcl = 0.0f;
    } else if (t1 <= dt) {
        dydt0 = -dmpY0dtf(t1, rd, cl) / dt;
        dyddt = -dmpY0f(t1, rd, cl) / (dt * dt);
        dydcl = dmpY0dclf(t1, rd, cl) / dt;
    } else {
        dydt0 = -(dmpY0dtf(t1, rd, cl) - dmpY0dtf(t1 - dt, rd, cl)) / dt;
        dyddt = -(dmpY0f(t1, rd, cl) + dmpY0f(t1 - dt, rd, cl)) / (dt * dt) +
                 dmpY0dtf(t1 - dt, rd, cl) / dt;
        dydcl = (dmpY0dclf(t1, rd, cl) - dmpY0dclf(t1 - dt, rd, cl)) / dt;
    }
}

__device__ __forceinline__ bool dmpSolve2x2f(float a00,
                                             float a01,
                                             float a10,
                                             float a11,
                                             float b0,
                                             float b1,
                                             float& x0,
                                             float& x1) {
    const float det = a00 * a11 - a01 * a10;
    const float scale = fabsf(a00 * a11) + fabsf(a01 * a10) + 1e-30f;
    if (!isfinite(det) || fabsf(det) <= 1e-6f * scale) {
        return false;
    }
    x0 = (b0 * a11 - a01 * b1) / det;
    x1 = (a00 * b1 - b0 * a10) / det;
    return isfinite(x0) && isfinite(x1);
}

__device__ __forceinline__ bool dmpSolve3x3f(float a00,
                                             float a01,
                                             float a02,
                                             float a10,
                                             float a11,
                                             float a12,
                                             float a20,
                                             float a21,
                                             float a22,
                                             float b0,
                                             float b1,
                                             float b2,
                                             float& x0,
                                             float& x1,
                                             float& x2) {
    const float c00 = a11 * a22 - a12 * a21;
    const float c01 = a10 * a22 - a12 * a20;
    const float c02 = a10 * a21 - a11 * a20;
    const float det = a00 * c00 - a01 * c01 + a02 * c02;
    const float scale = fabsf(a00 * c00) + fabsf(a01 * c01) + fabsf(a02 * c02) + 1e-30f;
    if (!isfinite(det) || fabsf(det) <= 1e-6f * scale) {
        return false;
    }
    const float det0 = b0 * c00 - a01 * (b1 * a22 - a12 * b2) +
                       a02 * (b1 * a21 - a11 * b2);
    const float det1 = a00 * (b1 * a22 - a12 * b2) -
                       b0 * c01 +
                       a02 * (a10 * b2 - b1 * a20);
    const float det2 = a00 * (a11 * b2 - b1 * a21) -
                       a01 * (a10 * b2 - b1 * a20) +
                       b0 * c02;
    x0 = det0 / det;
    x1 = det1 / det;
    x2 = det2 / det;
    return isfinite(x0) && isfinite(x1) && isfinite(x2);
}

__device__ bool dmp_model::findDriverParamsOnePoleScalar(double delay, double slew, double fixed_ceff) {
    int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(this, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    double t_vth, t_vl, measured_slew;
    gateDelays(fixed_ceff, t_vth, t_vl, measured_slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(fixed_ceff) || measured_slew <= 0.0 || fixed_ceff <= 0.0) {
        return false;
    }
    (void)delay;
    (void)slew;
    double x_t0 = t_vth + log(1.0 - driver_vth) * rd_[pin_idx] * fixed_ceff -
                  driver_vth * (measured_slew / (driver_vh - driver_vl));
    double x_dt = measured_slew / (driver_vh - driver_vl);
    if (!isfinite(x_t0) || !isfinite(x_dt)) {
        return false;
    }
    for (int iter = 0; iter < 100; ++iter) {
        if (x_dt <= 0.0) {
            x_dt = (t_vl - t_vth) / 100.0;
        }
        double ignore = 0.0;
        const double y_vl = y(t_vl, x_t0, x_dt, rd_[pin_idx], fixed_ceff);
        const double y_vth = y(t_vth, x_t0, x_dt, rd_[pin_idx], fixed_ceff);
        const double f0 = y_vl - driver_vl;
        const double f1 = y_vth - driver_vth;
        double a00, a01, a10, a11;
        dy(t_vl, x_t0, x_dt, rd_[pin_idx], fixed_ceff, a00, a01, ignore);
        dy(t_vth, x_t0, x_dt, rd_[pin_idx], fixed_ceff, a10, a11, ignore);
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
            t0[pin_idx] = x_t0;
            dt[pin_idx] = x_dt;
            ceff[pin_idx] = fixed_ceff;
            return true;
        }
    }
    return false;
}

__device__ bool dmp_model::findDriverParamsScalar(double delay, double slew, double initial_ceff) {
    int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(this, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    double t_vth, t_vl, measured_slew;
    gateDelays(initial_ceff, t_vth, t_vl, measured_slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(initial_ceff) || measured_slew <= 0.0 || initial_ceff < 0.0) {
        return false;
    }
    (void)delay;
    (void)slew;
    double x_ceff = initial_ceff;
    double x_dt = measured_slew / (driver_vh - driver_vl);
    double x_t0 = t_vth + log(1.0 - driver_vth) * rd_[pin_idx] * initial_ceff -
                  driver_vth * x_dt;
    if (!isfinite(x_t0) || !isfinite(x_dt) || x_dt <= 0.0) {
        return false;
    }
    for (int iter = 0; iter < 100; ++iter) {
        if (x_ceff < 0.0 || x_ceff > (C1[pin_idx] + C2[pin_idx]) || x_dt <= 0.0) {
            return false;
        }
        if (iter > 0) {
            gateDelays(x_ceff, t_vth, t_vl, measured_slew);
            if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
                measured_slew <= 0.0) {
                return false;
            }
        }
        double ceff_time = measured_slew / (driver_vh - driver_vl);
        if (ceff_time > 1.4 * x_dt) {
            ceff_time = 1.4 * x_dt;
        }

        const double exp_p1_dt = exp2(-p1_[pin_idx] * x_dt);
        const double exp_p2_dt = exp2(-p2_[pin_idx] * x_dt);
        const double exp_dt_rd_ceff = exp2(-x_dt / (rd_[pin_idx] * x_ceff));
        const double y_vth = y(t_vth, x_t0, x_dt, rd_[pin_idx], x_ceff);
        const double y_vl = y(t_vl, x_t0, x_dt, rd_[pin_idx], x_ceff);
        const double f0 = ipiIceff(pin_idx, x_dt, ceff_time, x_ceff);
        const double f1 = y_vth - driver_vth;
        const double f2 = y_vl - driver_vl;

        const double a00 = 0.0;
        const double a01 =
            (-A_[pin_idx] * x_dt + B_[pin_idx] * x_dt * exp_p1_dt -
             (2 * B_[pin_idx] / p1_[pin_idx]) * (1.0 - exp_p1_dt) +
             D_[pin_idx] * x_dt * exp_p2_dt -
             (2 * D_[pin_idx] / p2_[pin_idx]) * (1.0 - exp_p2_dt) +
             rd_[pin_idx] * x_ceff *
                 (x_dt + x_dt * exp_dt_rd_ceff -
                  2 * rd_[pin_idx] * x_ceff * (1.0 - exp_dt_rd_ceff))) /
            (rd_[pin_idx] * x_dt * x_dt * x_dt);
        const double a02 =
            (2 * rd_[pin_idx] * x_ceff - x_dt -
             (2 * rd_[pin_idx] * x_ceff + x_dt) *
                 exp2(-x_dt / (rd_[pin_idx] * x_ceff))) /
            (x_dt * x_dt);

        double a10, a11, a12;
        double a20, a21, a22;
        dy(t_vth, x_t0, x_dt, rd_[pin_idx], x_ceff, a10, a11, a12);
        dy(t_vl, x_t0, x_dt, rd_[pin_idx], x_ceff, a20, a21, a22);
        double p0, p1, p2;
        if (!dmpSolve3x3(a00, a01, a02,
                         a10, a11, a12,
                         a20, a21, a22,
                         -f0, -f1, -f2,
                         p0, p1, p2)) {
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
                x_dt <= 0.0 || x_ceff < 0.0 || x_ceff > C1[pin_idx] + C2[pin_idx]) {
                return false;
            }
            t0[pin_idx] = x_t0;
            dt[pin_idx] = x_dt;
            ceff[pin_idx] = x_ceff;
            return true;
        }
    }
    return false;
}

__device__ bool dmpFindDriverParamsOnePoleScalarWithCtx(dmp_model* dmp_db,
                                                        int pin_idx,
                                                        const DmpGateLaneContext& ctx,
                                                        double fixed_ceff) {
    dmpRootProfileAdd(DMP_ROOT_ONEPOLE_CALLS, 1ULL);
    double t_vth, t_vl, measured_slew;
    dmpGateDelaysWithCtx(ctx, fixed_ceff, t_vth, t_vl, measured_slew);
    const double rd = dmp_db->rd_[pin_idx];
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(fixed_ceff) || !isfinite(rd) ||
        measured_slew <= 0.0 || fixed_ceff <= 0.0) {
        dmpRootProfileAdd(DMP_ROOT_ONEPOLE_FAIL, 1ULL);
        return false;
    }
    double x_dt = measured_slew / (ctx.driver_vh - ctx.driver_vl);
    double x_t0 = t_vth + log(1.0 - ctx.driver_vth) * rd * fixed_ceff -
                  ctx.driver_vth * x_dt;
    if (!isfinite(x_t0) || !isfinite(x_dt)) {
        dmpRootProfileAdd(DMP_ROOT_ONEPOLE_FAIL, 1ULL);
        return false;
    }
    for (int iter = 0; iter < 100; ++iter) {
        if (x_dt <= 0.0) {
            x_dt = (t_vl - t_vth) / 100.0;
        }
        double ignore = 0.0;
        const double y_vl = dmp_db->y(t_vl, x_t0, x_dt, rd, fixed_ceff);
        const double y_vth = dmp_db->y(t_vth, x_t0, x_dt, rd, fixed_ceff);
        const double f0 = y_vl - ctx.driver_vl;
        const double f1 = y_vth - ctx.driver_vth;
        double a00, a01, a10, a11;
        dmp_db->dy(t_vl, x_t0, x_dt, rd, fixed_ceff, a00, a01, ignore);
        dmp_db->dy(t_vth, x_t0, x_dt, rd, fixed_ceff, a10, a11, ignore);
        double p0, p1;
        if (!dmpSolve2x2(a00, a01, a10, a11, -f0, -f1, p0, p1)) {
            dmpRootProfileAdd(DMP_ROOT_ONEPOLE_FAIL, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_ONEPOLE_ITERS, static_cast<unsigned long long>(iter + 1));
            return false;
        }
        const bool converged = fabs(p0) <= fabs(x_t0) * dmp_db->x_tol &&
                               fabs(p1) <= fabs(x_dt) * dmp_db->x_tol;
        x_t0 += p0;
        x_dt += p1;
        if (converged) {
            if (!isfinite(x_t0) || !isfinite(x_dt) || x_dt <= 0.0) {
                dmpRootProfileAdd(DMP_ROOT_ONEPOLE_FAIL, 1ULL);
                dmpRootProfileAdd(DMP_ROOT_ONEPOLE_ITERS, static_cast<unsigned long long>(iter + 1));
                return false;
            }
            dmp_db->t0[pin_idx] = x_t0;
            dmp_db->dt[pin_idx] = x_dt;
            dmp_db->ceff[pin_idx] = fixed_ceff;
            dmpRootProfileAdd(DMP_ROOT_ONEPOLE_SUCCESS, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_ONEPOLE_ITERS, static_cast<unsigned long long>(iter + 1));
            return true;
        }
    }
    dmpRootProfileAdd(DMP_ROOT_ONEPOLE_FAIL, 1ULL);
    dmpRootProfileAdd(DMP_ROOT_ONEPOLE_ITERS, 100ULL);
    return false;
}

__device__ __forceinline__ double dmpIpiIceffCached(double a,
                                                    double b,
                                                    double d,
                                                    double p1,
                                                    double p2,
                                                    double rd,
                                                    double dt,
                                                    double ceff_time,
                                                    double ceff) {
    const double exp_p1_dt = exp2(-p1 * ceff_time);
    const double exp_p2_dt = exp2(-p2 * ceff_time);
    const double exp_dt_rd_ceff = exp2(-ceff_time / (rd * ceff));
    const double ipi = (a * ceff_time + (b / p1) * (1.0 - exp_p1_dt) +
                        (d / p2) * (1.0 - exp_p2_dt)) /
                       (rd * ceff_time * dt);
    const double iceff = (rd * ceff * ceff_time -
                          (rd * ceff) * (rd * ceff) * (1.0 - exp_dt_rd_ceff)) /
                         (rd * ceff_time * dt);
    return ipi - iceff;
}

__device__ bool dmpFindDriverParamsScalarWithCtx(dmp_model* dmp_db,
                                                 int pin_idx,
                                                 const DmpGateLaneContext& ctx,
                                                 double initial_ceff) {
    dmpRootProfileAdd(DMP_ROOT_PI_CALLS, 1ULL);
    double t_vth, t_vl, measured_slew;
    dmpGateDelaysWithCtx(ctx, initial_ceff, t_vth, t_vl, measured_slew);
    const double rd = dmp_db->rd_[pin_idx];
    const double c1 = dmp_db->C1[pin_idx];
    const double c2 = dmp_db->C2[pin_idx];
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(initial_ceff) || !isfinite(rd) || !isfinite(c1) || !isfinite(c2) ||
        measured_slew <= 0.0 || initial_ceff < 0.0) {
        dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
        return false;
    }
    const double cap_sum = c1 + c2;
    const double a = dmp_db->A_[pin_idx];
    const double b = dmp_db->B_[pin_idx];
    const double d = dmp_db->D_[pin_idx];
    const double p1_wave = dmp_db->p1_[pin_idx];
    const double p2_wave = dmp_db->p2_[pin_idx];
    double x_ceff = initial_ceff;
    double x_dt = measured_slew / (ctx.driver_vh - ctx.driver_vl);
    double x_t0 = t_vth + log(1.0 - ctx.driver_vth) * rd * initial_ceff -
                  ctx.driver_vth * x_dt;
    if (!isfinite(x_t0) || !isfinite(x_dt) || x_dt <= 0.0) {
        dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
        return false;
    }
    for (int iter = 0; iter < 100; ++iter) {
        if (x_ceff < 0.0 || x_ceff > cap_sum || x_dt <= 0.0) {
            dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_PI_ITERS, static_cast<unsigned long long>(iter + 1));
            return false;
        }
        if (iter > 0) {
            dmpGateDelaysWithCtx(ctx, x_ceff, t_vth, t_vl, measured_slew);
            if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
                measured_slew <= 0.0) {
                dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
                dmpRootProfileAdd(DMP_ROOT_PI_ITERS, static_cast<unsigned long long>(iter + 1));
                return false;
            }
        }
        double ceff_time = measured_slew / (ctx.driver_vh - ctx.driver_vl);
        if (ceff_time > 1.4 * x_dt) {
            ceff_time = 1.4 * x_dt;
        }

        const double exp_p1_dt = exp2(-p1_wave * x_dt);
        const double exp_p2_dt = exp2(-p2_wave * x_dt);
        const double exp_dt_rd_ceff = exp2(-x_dt / (rd * x_ceff));
        const double y_vth = dmp_db->y(t_vth, x_t0, x_dt, rd, x_ceff);
        const double y_vl = dmp_db->y(t_vl, x_t0, x_dt, rd, x_ceff);
        const double f0 = dmpIpiIceffCached(a, b, d, p1_wave, p2_wave, rd,
                                            x_dt, ceff_time, x_ceff);
        const double f1 = y_vth - ctx.driver_vth;
        const double f2 = y_vl - ctx.driver_vl;

        const double a00 = 0.0;
        const double a01 =
            (-a * x_dt + b * x_dt * exp_p1_dt -
             (2 * b / p1_wave) * (1.0 - exp_p1_dt) +
             d * x_dt * exp_p2_dt -
             (2 * d / p2_wave) * (1.0 - exp_p2_dt) +
             rd * x_ceff *
                 (x_dt + x_dt * exp_dt_rd_ceff -
                  2 * rd * x_ceff * (1.0 - exp_dt_rd_ceff))) /
            (rd * x_dt * x_dt * x_dt);
        const double a02 =
            (2 * rd * x_ceff - x_dt -
             (2 * rd * x_ceff + x_dt) *
                 exp2(-x_dt / (rd * x_ceff))) /
            (x_dt * x_dt);

        double a10, a11, a12;
        double a20, a21, a22;
        dmp_db->dy(t_vth, x_t0, x_dt, rd, x_ceff, a10, a11, a12);
        dmp_db->dy(t_vl, x_t0, x_dt, rd, x_ceff, a20, a21, a22);
        double p0, p1, p2;
        if (!dmpSolve3x3(a00, a01, a02,
                         a10, a11, a12,
                         a20, a21, a22,
                         -f0, -f1, -f2,
                         p0, p1, p2)) {
            dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_PI_ITERS, static_cast<unsigned long long>(iter + 1));
            return false;
        }
        const bool converged = fabs(p0) <= fabs(x_t0) * dmp_db->x_tol &&
                               fabs(p1) <= fabs(x_dt) * dmp_db->x_tol &&
                               fabs(p2) <= fabs(x_ceff) * dmp_db->x_tol;
        x_t0 += p0;
        x_dt += p1;
        x_ceff += p2;
        if (converged) {
            if (!isfinite(x_t0) || !isfinite(x_dt) || !isfinite(x_ceff) ||
                x_dt <= 0.0 || x_ceff < 0.0 || x_ceff > cap_sum) {
                dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
                dmpRootProfileAdd(DMP_ROOT_PI_ITERS, static_cast<unsigned long long>(iter + 1));
                return false;
            }
            dmp_db->t0[pin_idx] = x_t0;
            dmp_db->dt[pin_idx] = x_dt;
            dmp_db->ceff[pin_idx] = x_ceff;
            dmpRootProfileAdd(DMP_ROOT_PI_SUCCESS, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_PI_ITERS, static_cast<unsigned long long>(iter + 1));
            return true;
        }
    }
    dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
    dmpRootProfileAdd(DMP_ROOT_PI_ITERS, 100ULL);
    return false;
}

__device__ __forceinline__ double dmpVoUpperBoundCached(int alg,
                                                        double t0_value,
                                                        double dt_value,
                                                        double c1,
                                                        double c2,
                                                        double rpi,
                                                        double rd) {
    if (alg == DMP_ALG_ZERO_C2) {
        return t0_value + dt_value + c1 * (rd + rpi) * 2.0;
    }
    if (alg == DMP_ALG_CAP) {
        return 0.0;
    }
    return t0_value + dt_value + (c1 + c2) * (rd + rpi) * 2.0;
}

__device__ __forceinline__ void dmpV0Cached(int alg,
                                            double k0,
                                            double k1,
                                            double k2,
                                            double k3,
                                            double k4,
                                            double p1,
                                            double p2,
                                            double t,
                                            double& vo,
                                            double& dvo_dt) {
    if (alg == DMP_ALG_CAP) {
        vo = 0.0;
        dvo_dt = 0.0;
        return;
    }
    const double exp_p1 = exp2(-p1 * t);
    if (alg == DMP_ALG_ZERO_C2) {
        vo = k0 * (k1 + k2 * t + k3 * exp_p1);
        dvo_dt = k0 * (k2 - k3 * p1 * exp_p1);
        return;
    }
    const double exp_p2 = exp2(-p2 * t);
    vo = k0 * (k1 + k2 * t + k3 * exp_p1 + k4 * exp_p2);
    dvo_dt = k0 * (k2 - k3 * p1 * exp_p1 - k4 * p2 * exp_p2);
}

__device__ __forceinline__ void dmpVoCached(int alg,
                                            double k0,
                                            double k1,
                                            double k2,
                                            double k3,
                                            double k4,
                                            double p1,
                                            double p2,
                                            double t0_value,
                                            double dt_value,
                                            double t,
                                            double& vo,
                                            double& dvo_dt) {
    const double t1 = t - t0_value;
    if (t1 <= 0.0) {
        vo = 0.0;
        dvo_dt = 0.0;
    } else if (t1 <= dt_value) {
        double v0, dv0_dt;
        dmpV0Cached(alg, k0, k1, k2, k3, k4, p1, p2, t1, v0, dv0_dt);
        vo = v0 / dt_value;
        dvo_dt = dv0_dt / dt_value;
    } else {
        double v0, dv0_dt;
        double v0_dt, dv0_dt_dt;
        dmpV0Cached(alg, k0, k1, k2, k3, k4, p1, p2, t1, v0, dv0_dt);
        dmpV0Cached(alg, k0, k1, k2, k3, k4, p1, p2, t1 - dt_value, v0_dt, dv0_dt_dt);
        vo = (v0 - v0_dt) / dt_value;
        dvo_dt = (dv0_dt - dv0_dt_dt) / dt_value;
    }
}

__device__ __forceinline__ void dmpVoFuncCached(int alg,
                                                double k0,
                                                double k1,
                                                double k2,
                                                double k3,
                                                double k4,
                                                double p1,
                                                double p2,
                                                double t0_value,
                                                double dt_value,
                                                double vth,
                                                double t,
                                                double& y,
                                                double& dy) {
    double vo, vo_dt;
    dmpVoCached(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, t, vo, vo_dt);
    y = vo - vth;
    dy = vo_dt;
}

__device__ double dmpFindRootVoCached(const dmp_model* dmp_db,
                                      int alg,
                                      double k0,
                                      double k1,
                                      double k2,
                                      double k3,
                                      double k4,
                                      double p1,
                                      double p2,
                                      double t0_value,
                                      double dt_value,
                                      double vth,
                                      double x1,
                                      double x2) {
    dmpRootProfileAdd(DMP_ROOT_VO_CALLS, 1ULL);
    double y1, y2, dy;
    dmpVoFuncCached(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, vth, x1, y1, dy);
    dmpVoFuncCached(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, vth, x2, y2, dy);
    if (y1 * y2 > 0.0) {
        dmpRootProfileAdd(DMP_ROOT_VO_BRACKET_FAIL, 1ULL);
        return nanf("");
    }
    if (y1 == 0.0) {
        dmpRootProfileAdd(DMP_ROOT_VO_SUCCESS, 1ULL);
        dmpRootProfileAdd(DMP_ROOT_VO_ENDPOINT_HIT, 1ULL);
        return x1;
    }
    if (y2 == 0.0) {
        dmpRootProfileAdd(DMP_ROOT_VO_SUCCESS, 1ULL);
        dmpRootProfileAdd(DMP_ROOT_VO_ENDPOINT_HIT, 1ULL);
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
    dmpVoFuncCached(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, vth, root, y, dy);
    for (int iter = 0; iter < dmp_db->MAX_ITER; ++iter) {
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
        if (fabs(dx) <= dmp_db->x_tol * fabs(root)) {
            dmpRootProfileAdd(DMP_ROOT_VO_SUCCESS, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_VO_ITERS, static_cast<unsigned long long>(iter + 1));
            return root;
        }

        dmpVoFuncCached(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, vth, root, y, dy);
        if (y < 0.0) {
            x1 = root;
        } else {
            x2 = root;
        }
    }
    dmpRootProfileAdd(DMP_ROOT_VO_MAXITER_FAIL, 1ULL);
    return nanf("");
}

__device__ __forceinline__ bool dmpFindDriverDelaySlewCached(const dmp_model* dmp_db,
                                                             int alg,
                                                             double k0,
                                                             double k1,
                                                             double k2,
                                                             double k3,
                                                             double k4,
                                                             double p1,
                                                             double p2,
                                                             double t0_value,
                                                             double dt_value,
                                                             double c1,
                                                             double c2,
                                                             double rpi,
                                                             double rd,
                                                             double driver_vth,
                                                             double driver_vl,
                                                             double driver_vh,
                                                             double driver_derate,
                                                             double& delay,
                                                             double& slew) {
    delay = nanf("");
    slew = nanf("");
    if (alg == DMP_ALG_CAP || !isfinite(t0_value) || !isfinite(dt_value) || dt_value <= 0.0 ||
        !isfinite(rd) || rd <= 0.0 || !isfinite(driver_derate) || driver_derate <= 0.0) {
        return false;
    }
    const double t_upper = dmpVoUpperBoundCached(alg, t0_value, dt_value, c1, c2, rpi, rd);
    delay = dmpFindRootVoCached(dmp_db, alg, k0, k1, k2, k3, k4, p1, p2,
                                t0_value, dt_value, driver_vth, t0_value, t_upper);
    if (!isfinite(delay)) {
        delay = slew = nanf("");
        return false;
    }
    const double tl = dmpFindRootVoCached(dmp_db, alg, k0, k1, k2, k3, k4, p1, p2,
                                          t0_value, dt_value, driver_vl, t0_value, delay);
    const double th = dmpFindRootVoCached(dmp_db, alg, k0, k1, k2, k3, k4, p1, p2,
                                          t0_value, dt_value, driver_vh, delay, t_upper);
    if (!isfinite(tl) || !isfinite(th)) {
        delay = slew = nanf("");
        return false;
    }
    slew = (th - tl) / driver_derate;
    return isfinite(slew);
}

__device__ __forceinline__ float dmpIpiIceffCachedFloat(float a,
                                                        float b,
                                                        float d,
                                                        float p1,
                                                        float p2,
                                                        float rd,
                                                        float dt,
                                                        float ceff_time,
                                                        float ceff) {
    const float exp_p1_dt = dmpFastExp2f(-p1 * ceff_time);
    const float exp_p2_dt = dmpFastExp2f(-p2 * ceff_time);
    const float exp_dt_rd_ceff = dmpFastExp2f(-ceff_time / (rd * ceff));
    const float ipi = (a * ceff_time + (b / p1) * (1.0f - exp_p1_dt) +
                       (d / p2) * (1.0f - exp_p2_dt)) /
                      (rd * ceff_time * dt);
    const float rd_ceff = rd * ceff;
    const float iceff = (rd_ceff * ceff_time -
                         rd_ceff * rd_ceff * (1.0f - exp_dt_rd_ceff)) /
                        (rd * ceff_time * dt);
    return ipi - iceff;
}

__device__ bool dmpFindDriverParamsOnePoleFloatWithCtx(dmp_model* dmp_db,
                                                       int pin_idx,
                                                       const DmpGateLaneContext& ctx,
                                                       double fixed_ceff) {
    dmpRootProfileAdd(DMP_ROOT_ONEPOLE_CALLS, 1ULL);
    const float fixed_ceff_f = static_cast<float>(fixed_ceff);
    float t_vth, t_vl, measured_slew;
    dmpGateDelaysWithCtxFloat(ctx, fixed_ceff_f, t_vth, t_vl, measured_slew);
    const float rd = static_cast<float>(dmp_db->rd_[pin_idx]);
    const float driver_vth = static_cast<float>(ctx.driver_vth);
    const float driver_vl = static_cast<float>(ctx.driver_vl);
    const float driver_vh = static_cast<float>(ctx.driver_vh);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(fixed_ceff_f) || !isfinite(rd) ||
        measured_slew <= 0.0f || fixed_ceff_f <= 0.0f) {
        dmpRootProfileAdd(DMP_ROOT_ONEPOLE_FAIL, 1ULL);
        return false;
    }
    float x_dt = measured_slew / (driver_vh - driver_vl);
    float x_t0 = t_vth + logf(1.0f - driver_vth) * rd * fixed_ceff_f -
                 driver_vth * x_dt;
    if (!isfinite(x_t0) || !isfinite(x_dt)) {
        dmpRootProfileAdd(DMP_ROOT_ONEPOLE_FAIL, 1ULL);
        return false;
    }
    const float x_tol = static_cast<float>(dmp_db->x_tol);
    for (int iter = 0; iter < 100; ++iter) {
        if (x_dt <= 0.0f) {
            x_dt = (t_vl - t_vth) / 100.0f;
        }
        float ignore = 0.0f;
        const float y_vl = dmpYf(t_vl, x_t0, x_dt, rd, fixed_ceff_f);
        const float y_vth = dmpYf(t_vth, x_t0, x_dt, rd, fixed_ceff_f);
        const float f0 = y_vl - driver_vl;
        const float f1 = y_vth - driver_vth;
        float a00, a01, a10, a11;
        dmpDyf(t_vl, x_t0, x_dt, rd, fixed_ceff_f, a00, a01, ignore);
        dmpDyf(t_vth, x_t0, x_dt, rd, fixed_ceff_f, a10, a11, ignore);
        float p0, p1;
        if (!dmpSolve2x2f(a00, a01, a10, a11, -f0, -f1, p0, p1)) {
            dmpRootProfileAdd(DMP_ROOT_ONEPOLE_FAIL, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_ONEPOLE_ITERS, static_cast<unsigned long long>(iter + 1));
            return false;
        }
        const bool converged = fabsf(p0) <= fabsf(x_t0) * x_tol &&
                               fabsf(p1) <= fabsf(x_dt) * x_tol;
        x_t0 += p0;
        x_dt += p1;
        if (converged) {
            if (!isfinite(x_t0) || !isfinite(x_dt) || x_dt <= 0.0f) {
                dmpRootProfileAdd(DMP_ROOT_ONEPOLE_FAIL, 1ULL);
                dmpRootProfileAdd(DMP_ROOT_ONEPOLE_ITERS, static_cast<unsigned long long>(iter + 1));
                return false;
            }
            dmp_db->t0[pin_idx] = static_cast<double>(x_t0);
            dmp_db->dt[pin_idx] = static_cast<double>(x_dt);
            dmp_db->ceff[pin_idx] = fixed_ceff;
            dmpRootProfileAdd(DMP_ROOT_ONEPOLE_SUCCESS, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_ONEPOLE_ITERS, static_cast<unsigned long long>(iter + 1));
            return true;
        }
    }
    dmpRootProfileAdd(DMP_ROOT_ONEPOLE_FAIL, 1ULL);
    dmpRootProfileAdd(DMP_ROOT_ONEPOLE_ITERS, 100ULL);
    return false;
}

__device__ bool dmpFindDriverParamsFloatWithCtx(dmp_model* dmp_db,
                                                int pin_idx,
                                                const DmpGateLaneContext& ctx,
                                                double initial_ceff) {
    dmpRootProfileAdd(DMP_ROOT_PI_CALLS, 1ULL);
    float t_vth, t_vl, measured_slew;
    const float initial_ceff_f = static_cast<float>(initial_ceff);
    dmpGateDelaysWithCtxFloat(ctx, initial_ceff_f, t_vth, t_vl, measured_slew);
    const float rd = static_cast<float>(dmp_db->rd_[pin_idx]);
    const float c1 = static_cast<float>(dmp_db->C1[pin_idx]);
    const float c2 = static_cast<float>(dmp_db->C2[pin_idx]);
    const float driver_vth = static_cast<float>(ctx.driver_vth);
    const float driver_vl = static_cast<float>(ctx.driver_vl);
    const float driver_vh = static_cast<float>(ctx.driver_vh);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(initial_ceff_f) || !isfinite(rd) || !isfinite(c1) || !isfinite(c2) ||
        measured_slew <= 0.0f || initial_ceff_f < 0.0f) {
        dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
        return false;
    }
    const float cap_sum = c1 + c2;
    const float a = static_cast<float>(dmp_db->A_[pin_idx]);
    const float b = static_cast<float>(dmp_db->B_[pin_idx]);
    const float d = static_cast<float>(dmp_db->D_[pin_idx]);
    const float p1_wave = static_cast<float>(dmp_db->p1_[pin_idx]);
    const float p2_wave = static_cast<float>(dmp_db->p2_[pin_idx]);
    float x_ceff = initial_ceff_f;
    float x_dt = measured_slew / (driver_vh - driver_vl);
    float x_t0 = t_vth + logf(1.0f - driver_vth) * rd * initial_ceff_f -
                 driver_vth * x_dt;
    if (!isfinite(x_t0) || !isfinite(x_dt) || x_dt <= 0.0f) {
        dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
        return false;
    }
    const float x_tol = static_cast<float>(dmp_db->x_tol);
    for (int iter = 0; iter < 100; ++iter) {
        if (x_ceff < 0.0f || x_ceff > cap_sum || x_dt <= 0.0f) {
            dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_PI_ITERS, static_cast<unsigned long long>(iter + 1));
            return false;
        }
        if (iter > 0) {
            dmpGateDelaysWithCtxFloat(ctx, x_ceff, t_vth, t_vl, measured_slew);
            if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
                measured_slew <= 0.0f) {
                dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
                dmpRootProfileAdd(DMP_ROOT_PI_ITERS, static_cast<unsigned long long>(iter + 1));
                return false;
            }
        }
        float ceff_time = measured_slew / (driver_vh - driver_vl);
        if (ceff_time > 1.4f * x_dt) {
            ceff_time = 1.4f * x_dt;
        }

        const float exp_p1_dt = dmpFastExp2f(-p1_wave * x_dt);
        const float exp_p2_dt = dmpFastExp2f(-p2_wave * x_dt);
        const float exp_dt_rd_ceff = dmpFastExp2f(-x_dt / (rd * x_ceff));
        const float y_vth = dmpYf(t_vth, x_t0, x_dt, rd, x_ceff);
        const float y_vl = dmpYf(t_vl, x_t0, x_dt, rd, x_ceff);
        const float f0 = dmpIpiIceffCachedFloat(a, b, d, p1_wave, p2_wave, rd,
                                                x_dt, ceff_time, x_ceff);
        const float f1 = y_vth - driver_vth;
        const float f2 = y_vl - driver_vl;

        const float a00 = 0.0f;
        const float a01 =
            (-a * x_dt + b * x_dt * exp_p1_dt -
             (2.0f * b / p1_wave) * (1.0f - exp_p1_dt) +
             d * x_dt * exp_p2_dt -
             (2.0f * d / p2_wave) * (1.0f - exp_p2_dt) +
             rd * x_ceff *
                 (x_dt + x_dt * exp_dt_rd_ceff -
                  2.0f * rd * x_ceff * (1.0f - exp_dt_rd_ceff))) /
            (rd * x_dt * x_dt * x_dt);
        const float a02 =
            (2.0f * rd * x_ceff - x_dt -
             (2.0f * rd * x_ceff + x_dt) *
                 dmpFastExp2f(-x_dt / (rd * x_ceff))) /
            (x_dt * x_dt);

        float a10, a11, a12;
        float a20, a21, a22;
        dmpDyf(t_vth, x_t0, x_dt, rd, x_ceff, a10, a11, a12);
        dmpDyf(t_vl, x_t0, x_dt, rd, x_ceff, a20, a21, a22);
        float p0, p1, p2;
        if (!dmpSolve3x3f(a00, a01, a02,
                          a10, a11, a12,
                          a20, a21, a22,
                          -f0, -f1, -f2,
                          p0, p1, p2)) {
            dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_PI_ITERS, static_cast<unsigned long long>(iter + 1));
            return false;
        }
        const bool converged = fabsf(p0) <= fabsf(x_t0) * x_tol &&
                               fabsf(p1) <= fabsf(x_dt) * x_tol &&
                               fabsf(p2) <= fabsf(x_ceff) * x_tol;
        x_t0 += p0;
        x_dt += p1;
        x_ceff += p2;
        if (converged) {
            if (!isfinite(x_t0) || !isfinite(x_dt) || !isfinite(x_ceff) ||
                x_dt <= 0.0f || x_ceff < 0.0f || x_ceff > cap_sum) {
                dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
                dmpRootProfileAdd(DMP_ROOT_PI_ITERS, static_cast<unsigned long long>(iter + 1));
                return false;
            }
            dmp_db->t0[pin_idx] = static_cast<double>(x_t0);
            dmp_db->dt[pin_idx] = static_cast<double>(x_dt);
            dmp_db->ceff[pin_idx] = static_cast<double>(x_ceff);
            dmpRootProfileAdd(DMP_ROOT_PI_SUCCESS, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_PI_ITERS, static_cast<unsigned long long>(iter + 1));
            return true;
        }
    }
    dmpRootProfileAdd(DMP_ROOT_PI_FAIL, 1ULL);
    dmpRootProfileAdd(DMP_ROOT_PI_ITERS, 100ULL);
    return false;
}

__device__ __forceinline__ float dmpVoUpperBoundCachedFloat(int alg,
                                                            float t0_value,
                                                            float dt_value,
                                                            float c1,
                                                            float c2,
                                                            float rpi,
                                                            float rd) {
    if (alg == DMP_ALG_ZERO_C2) {
        return t0_value + dt_value + c1 * (rd + rpi) * 2.0f;
    }
    if (alg == DMP_ALG_CAP) {
        return 0.0f;
    }
    return t0_value + dt_value + (c1 + c2) * (rd + rpi) * 2.0f;
}

__device__ __forceinline__ void dmpV0CachedFloat(int alg,
                                                 float k0,
                                                 float k1,
                                                 float k2,
                                                 float k3,
                                                 float k4,
                                                 float p1,
                                                 float p2,
                                                 float t,
                                                 float& vo,
                                                 float& dvo_dt) {
    if (alg == DMP_ALG_CAP) {
        vo = 0.0f;
        dvo_dt = 0.0f;
        return;
    }
    const float exp_p1 = dmpFastExp2f(-p1 * t);
    if (alg == DMP_ALG_ZERO_C2) {
        vo = k0 * (k1 + k2 * t + k3 * exp_p1);
        dvo_dt = k0 * (k2 - k3 * p1 * exp_p1);
        return;
    }
    const float exp_p2 = dmpFastExp2f(-p2 * t);
    vo = k0 * (k1 + k2 * t + k3 * exp_p1 + k4 * exp_p2);
    dvo_dt = k0 * (k2 - k3 * p1 * exp_p1 - k4 * p2 * exp_p2);
}

__device__ __forceinline__ void dmpVoCachedFloat(int alg,
                                                 float k0,
                                                 float k1,
                                                 float k2,
                                                 float k3,
                                                 float k4,
                                                 float p1,
                                                 float p2,
                                                 float t0_value,
                                                 float dt_value,
                                                 float t,
                                                 float& vo,
                                                 float& dvo_dt) {
    const float t1 = t - t0_value;
    if (t1 <= 0.0f) {
        vo = 0.0f;
        dvo_dt = 0.0f;
    } else if (t1 <= dt_value) {
        float v0, dv0_dt;
        dmpV0CachedFloat(alg, k0, k1, k2, k3, k4, p1, p2, t1, v0, dv0_dt);
        vo = v0 / dt_value;
        dvo_dt = dv0_dt / dt_value;
    } else {
        float v0, dv0_dt;
        float v0_dt, dv0_dt_dt;
        dmpV0CachedFloat(alg, k0, k1, k2, k3, k4, p1, p2, t1, v0, dv0_dt);
        dmpV0CachedFloat(alg, k0, k1, k2, k3, k4, p1, p2, t1 - dt_value, v0_dt, dv0_dt_dt);
        vo = (v0 - v0_dt) / dt_value;
        dvo_dt = (dv0_dt - dv0_dt_dt) / dt_value;
    }
}

__device__ __forceinline__ void dmpVoFuncCachedFloat(int alg,
                                                     float k0,
                                                     float k1,
                                                     float k2,
                                                     float k3,
                                                     float k4,
                                                     float p1,
                                                     float p2,
                                                     float t0_value,
                                                     float dt_value,
                                                     float vth,
                                                     float t,
                                                     float& y,
                                                     float& dy) {
    float vo, vo_dt;
    dmpVoCachedFloat(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, t, vo, vo_dt);
    y = vo - vth;
    dy = vo_dt;
}

__device__ float dmpFindRootVoFloatCached(const dmp_model* dmp_db,
                                          int alg,
                                          float k0,
                                          float k1,
                                          float k2,
                                          float k3,
                                          float k4,
                                          float p1,
                                          float p2,
                                          float t0_value,
                                          float dt_value,
                                          float vth,
                                          float x1,
                                          float x2) {
    dmpRootProfileAdd(DMP_ROOT_VO_CALLS, 1ULL);
    float y1, y2, dy;
    dmpVoFuncCachedFloat(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, vth, x1, y1, dy);
    dmpVoFuncCachedFloat(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, vth, x2, y2, dy);
    if (y1 * y2 > 0.0f) {
        dmpRootProfileAdd(DMP_ROOT_VO_BRACKET_FAIL, 1ULL);
        return nanf("");
    }
    if (y1 == 0.0f) {
        dmpRootProfileAdd(DMP_ROOT_VO_SUCCESS, 1ULL);
        dmpRootProfileAdd(DMP_ROOT_VO_ENDPOINT_HIT, 1ULL);
        return x1;
    }
    if (y2 == 0.0f) {
        dmpRootProfileAdd(DMP_ROOT_VO_SUCCESS, 1ULL);
        dmpRootProfileAdd(DMP_ROOT_VO_ENDPOINT_HIT, 1ULL);
        return x2;
    }
    if (y1 > 0.0f) {
        const float tmp = x1;
        x1 = x2;
        x2 = tmp;
    }
    float root = (x1 + x2) * 0.5f;
    float dx_prev = fabsf(x2 - x1);
    float dx = dx_prev;
    float y;
    dmpVoFuncCachedFloat(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, vth, root, y, dy);
    const float x_tol = static_cast<float>(dmp_db->x_tol);
    for (int iter = 0; iter < dmp_db->MAX_ITER; ++iter) {
        if ((((x2 - root) * dy + y) * ((x1 - root) * dy + y) > 0.0f) ||
            (fabsf(2.0f * y) > fabsf(dx_prev * dy))) {
            dx_prev = dx;
            dx = (x2 - x1) * 0.5f;
            root = x1 + dx;
        } else {
            dx_prev = dx;
            dx = y / dy;
            root -= dx;
        }
        if (fabsf(dx) <= x_tol * fabsf(root)) {
            dmpRootProfileAdd(DMP_ROOT_VO_SUCCESS, 1ULL);
            dmpRootProfileAdd(DMP_ROOT_VO_ITERS, static_cast<unsigned long long>(iter + 1));
            return root;
        }

        dmpVoFuncCachedFloat(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, vth, root, y, dy);
        if (y < 0.0f) {
            x1 = root;
        } else {
            x2 = root;
        }
    }
    dmpRootProfileAdd(DMP_ROOT_VO_MAXITER_FAIL, 1ULL);
    return nanf("");
}

__device__ __forceinline__ bool dmpFindDriverDelaySlewFloatCached(const dmp_model* dmp_db,
                                                                  int alg,
                                                                  float k0,
                                                                  float k1,
                                                                  float k2,
                                                                  float k3,
                                                                  float k4,
                                                                  float p1,
                                                                  float p2,
                                                                  float t0_value,
                                                                  float dt_value,
                                                                  float c1,
                                                                  float c2,
                                                                  float rpi,
                                                                  float rd,
                                                                  float driver_vth,
                                                                  float driver_vl,
                                                                  float driver_vh,
                                                                  float driver_derate,
                                                                  double& delay,
                                                                  double& slew) {
    delay = nanf("");
    slew = nanf("");
    if (alg == DMP_ALG_CAP || !isfinite(t0_value) || !isfinite(dt_value) || dt_value <= 0.0f ||
        !isfinite(rd) || rd <= 0.0f || !isfinite(driver_derate) || driver_derate <= 0.0f) {
        return false;
    }
    const float t_upper = dmpVoUpperBoundCachedFloat(alg, t0_value, dt_value, c1, c2, rpi, rd);
    const float delay_f = dmpFindRootVoFloatCached(dmp_db, alg, k0, k1, k2, k3, k4, p1, p2,
                                                   t0_value, dt_value, driver_vth, t0_value, t_upper);
    if (!isfinite(delay_f)) {
        delay = slew = nanf("");
        return false;
    }
    const float tl = dmpFindRootVoFloatCached(dmp_db, alg, k0, k1, k2, k3, k4, p1, p2,
                                              t0_value, dt_value, driver_vl, t0_value, delay_f);
    const float th = dmpFindRootVoFloatCached(dmp_db, alg, k0, k1, k2, k3, k4, p1, p2,
                                              t0_value, dt_value, driver_vh, delay_f, t_upper);
    if (!isfinite(tl) || !isfinite(th)) {
        delay = slew = nanf("");
        return false;
    }
    const float slew_f = (th - tl) / driver_derate;
    delay = static_cast<double>(delay_f);
    slew = static_cast<double>(slew_f);
    return isfinite(slew_f);
}

