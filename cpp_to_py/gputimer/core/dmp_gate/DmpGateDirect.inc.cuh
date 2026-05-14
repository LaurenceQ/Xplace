__device__ __forceinline__ double dmpSlotVoUpperBoundCached(int alg,
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

__device__ __forceinline__ void dmpVl0Cached(int alg,
                                             double k0,
                                             double k1,
                                             double k2,
                                             double k3,
                                             double k4,
                                             double p1,
                                             double p2,
                                             double elmore,
                                             double t,
                                             double& vl,
                                             double& dvl_dt) {
    if (alg == DMP_ALG_CAP || !isfinite(elmore) || elmore <= 0.0) {
        vl = 0.0;
        dvl_dt = 0.0;
        return;
    }
    const double p3 = 1.0 / elmore;
    double d1 = k0 * (k1 - k2 / p3);
    double d3 = -p3 * k0 * k3 / (p1 - p3);
    double d4 = 0.0;
    double d5 = k0 * (k2 / p3 - k1 + p3 * k3 / (p1 - p3));
    if (alg == DMP_ALG_PI) {
        d4 = -p3 * k0 * k4 / (p2 - p3);
        d5 += k0 * p3 * k4 / (p2 - p3);
    }
    const double exp_p1 = exp2(-p1 * t);
    const double exp_p2 = alg == DMP_ALG_PI ? exp2(-p2 * t) : 0.0;
    const double exp_p3 = exp2(-p3 * t);
    vl = d1 + t + d3 * exp_p1 + d4 * exp_p2 + d5 * exp_p3;
    dvl_dt = 1.0 - d3 * p1 * exp_p1 - d4 * p2 * exp_p2 - d5 * p3 * exp_p3;
}

__device__ __forceinline__ void dmpVlCached(int alg,
                                            double k0,
                                            double k1,
                                            double k2,
                                            double k3,
                                            double k4,
                                            double p1,
                                            double p2,
                                            double t0_value,
                                            double dt_value,
                                            double elmore,
                                            double t,
                                            double& vl,
                                            double& dvl_dt) {
    const double t1 = t - t0_value;
    if (t1 <= 0.0) {
        vl = 0.0;
        dvl_dt = 0.0;
    } else if (t1 <= dt_value) {
        double vl0, dvl0_dt;
        dmpVl0Cached(alg, k0, k1, k2, k3, k4, p1, p2, elmore, t1, vl0, dvl0_dt);
        vl = vl0 / dt_value;
        dvl_dt = dvl0_dt / dt_value;
    } else {
        double vl0, dvl0_dt;
        double vl0_dt, dvl0_dt_dt;
        dmpVl0Cached(alg, k0, k1, k2, k3, k4, p1, p2, elmore, t1, vl0, dvl0_dt);
        dmpVl0Cached(alg, k0, k1, k2, k3, k4, p1, p2, elmore, t1 - dt_value, vl0_dt, dvl0_dt_dt);
        vl = (vl0 - vl0_dt) / dt_value;
        dvl_dt = (dvl0_dt - dvl0_dt_dt) / dt_value;
    }
}

__device__ __forceinline__ void dmpVlFuncCached(int alg,
                                                double k0,
                                                double k1,
                                                double k2,
                                                double k3,
                                                double k4,
                                                double p1,
                                                double p2,
                                                double t0_value,
                                                double dt_value,
                                                double elmore,
                                                double vth,
                                                double t,
                                                double& y,
                                                double& dy) {
    double vl, vl_dt;
    dmpVlCached(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, elmore, t, vl, vl_dt);
    y = vl - vth;
    dy = vl_dt;
}

__device__ double dmpFindRootVlCached(const DmpModel* dmp_db,
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
                                      double elmore,
                                      double vth,
                                      double x1,
                                      double x2) {
    double y1, y2, dy;
    dmpVlFuncCached(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, elmore, vth, x1, y1, dy);
    dmpVlFuncCached(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, elmore, vth, x2, y2, dy);
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
    dmpVlFuncCached(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, elmore, vth, root, y, dy);
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
            return root;
        }

        dmpVlFuncCached(alg, k0, k1, k2, k3, k4, p1, p2, t0_value, dt_value, elmore, vth, root, y, dy);
        if (y < 0.0) {
            x1 = root;
        } else {
            x2 = root;
        }
    }
    return nanf("");
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

__device__ double dmpFindRootVoCached(const DmpModel* dmp_db,
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

__device__ __forceinline__ bool dmpFindDriverDelaySlewCached(const DmpModel* dmp_db,
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
    const double t_upper =
        dmpSlotVoUpperBoundCached(alg, t0_value, dt_value, c1, c2, rpi, rd);
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

struct DmpLocalGateState {
    int alg;
    double k0;
    double k1;
    double k2;
    double k3;
    double k4;
    double p1;
    double p2;
    double z1;
    double A;
    double B;
    double D;
    double c1;
    double c2;
    double rpi;
    double rd;
    double t0_value;
    double dt_value;
    double ceff_value;
    double vo_delay;
    double vo_slew;
    double gate_delay;
    double driver_vth;
    double driver_vl;
    double driver_vh;
    double driver_derate;
    int driver_library_id;
    bool dmp_valid;
};

__device__ __forceinline__ DmpGateLaneContext dmpMakeGateLaneContextDirect(DmpModel* dmp_db,
                                                                           int timing_id,
                                                                           int input_rf,
                                                                           int to_attr,
                                                                           int output_rf,
                                                                           float input_slew) {
    DmpGateLaneContext ctx{};
    ctx.allocator = dmp_db->d_allocator;
    ctx.delay_lut = {};
    ctx.slew_lut = {};
    ctx.input_slew = input_slew;
    ctx.valid = false;
    ctx.driver_library_id = dmpTimingLibraryId(dmp_db, timing_id);
    dmpDriverLibraryThresholds(dmp_db,
                               ctx.driver_library_id,
                               to_attr,
                               ctx.driver_vth,
                               ctx.driver_vl,
                               ctx.driver_vh,
                               ctx.driver_derate);

    if (ctx.allocator == nullptr ||
        timing_id < 0 || input_rf < 0 || output_rf < 0 ||
        !isfinite(input_slew) ||
        !ctx.allocator->is_transition_defined(timing_id, input_rf, output_rf)) {
        return ctx;
    }
    const int base_lut_id = ctx.allocator->num_luts_in_timing * timing_id + output_rf;
    ctx.delay_lut = dmpMakeGateLutMeta(ctx.allocator, base_lut_id);
    ctx.slew_lut = dmpMakeGateLutMeta(ctx.allocator, base_lut_id + 2);
    const double driver_delta = ctx.driver_vh - ctx.driver_vl;
    ctx.valid = dmpLutMetaValid(ctx.delay_lut) &&
                dmpLutMetaValid(ctx.slew_lut) &&
                isfinite(ctx.driver_vth) &&
                isfinite(ctx.driver_vl) &&
                isfinite(ctx.driver_vh) &&
                isfinite(ctx.driver_derate) &&
                isfinite(driver_delta) &&
                driver_delta > 0.0 &&
                ctx.driver_derate > 0.0;
    return ctx;
}

__device__ __forceinline__ bool dmpLocalGateModelRd(DmpModel* dmp_db,
                                                    const DmpGateLaneContext& ctx,
                                                    double cap1,
                                                    double d1,
                                                    double& rd) {
    rd = nanf("");
    const float cap1_f = static_cast<float>(cap1);
    const float cap_delta_f = static_cast<float>(1e-15 / dmp_db->cap_unit);
    const float cap2_f = cap1_f + cap_delta_f;
    const float d1_f = static_cast<float>(d1);
    float d2_f = nanf("");
    float s2_f = nanf("");
    if (!isfinite(cap1_f) || !isfinite(cap2_f) || !isfinite(d1_f) ||
        !isfinite(cap_delta_f) || cap_delta_f <= 0.0f || cap2_f == cap1_f) {
        return false;
    }
    dmpGateCapDelaySlewWithCtxFloat(ctx, cap2_f, d2_f, s2_f);
    if (!isfinite(d2_f)) {
        return false;
    }
    rd = static_cast<double>(
        -log(ctx.driver_vth) * fabsf(d1_f - d2_f) / (cap2_f - cap1_f));
    return isfinite(rd) && rd > 0.0;
}

__device__ __forceinline__ int dmpSelectLocalAlg(const DmpModel* dmp_db,
                                                 const DmpLocalGateState& state) {
    if (!isfinite(state.rd) || !isfinite(state.c1) || !isfinite(state.c2) ||
        !isfinite(state.rpi) || state.rd <= 0.0 || state.c1 <= 0.0 ||
        state.c2 < 0.0 || state.rpi <= 0.0) {
        return DMP_ALG_CAP;
    }
    const double min_rd = (isfinite(dmp_db->res_unit) && dmp_db->res_unit > 0.0f)
                              ? (1e-2 / static_cast<double>(dmp_db->res_unit))
                              : 1e-2;
    if (state.rd < min_rd || state.rpi < state.rd * 1e-3 ||
        state.c1 < state.c2 * 1e-3) {
        return DMP_ALG_CAP;
    }
    if (state.c2 < state.c1 * 1e-3) {
        return DMP_ALG_ZERO_C2;
    }
    return DMP_ALG_PI;
}

__device__ __forceinline__ bool dmpLocalInitZeroC2(DmpLocalGateState& state) {
    const double denom_z1 = state.rpi * state.c1;
    const double denom_p1 = state.c1 * (state.rd + state.rpi);
    if (!isfinite(denom_z1) || !isfinite(denom_p1) ||
        denom_z1 == 0.0 || denom_p1 == 0.0) {
        return false;
    }
    state.ceff_value = state.c1;
    state.z1 = 1.0 / (state.rpi * state.c1);
    state.p1 = 1.0 / (state.c1 * (state.rd + state.rpi));
    state.k0 = state.p1 / state.z1;
    if (!isfinite(state.k0) || state.k0 == 0.0 || state.p1 == 0.0) {
        return false;
    }
    state.k2 = 1.0 / state.k0;
    state.k1 = (state.p1 - state.z1) / (state.p1 * state.p1);
    state.k3 = -state.k1;
    state.p2 = 0.0;
    state.k4 = 0.0;
    state.A = 0.0;
    state.B = 0.0;
    state.D = 0.0;
    return isfinite(state.z1) && isfinite(state.k0) &&
           isfinite(state.k1) && isfinite(state.k2) &&
           isfinite(state.k3) && isfinite(state.p1);
}

__device__ __forceinline__ bool dmpLocalInitPi(DmpLocalGateState& state) {
    const double denom_z1 = state.rpi * state.c1;
    const double denom_k0 = state.rd * state.c2;
    if (!isfinite(denom_z1) || !isfinite(denom_k0) ||
        denom_z1 == 0.0 || denom_k0 == 0.0) {
        return false;
    }
    state.z1 = 1.0 / (state.rpi * state.c1);
    state.k0 = 1.0 / (state.rd * state.c2);
    const double a = state.rpi * state.rd * state.c1 * state.c2;
    const double b = state.rd * (state.c1 + state.c2) + state.rpi * state.c1;
    const double disc = b * b - 4 * a;
    if (!isfinite(a) || !isfinite(b) || !isfinite(disc) || a == 0.0 || disc < 0.0) {
        return false;
    }
    const double sqrt_disc = sqrt(disc);
    state.p1 = (b + sqrt_disc) / (2 * a);
    state.p2 = (b - sqrt_disc) / (2 * a);
    const double p1p2 = state.p1 * state.p2;
    if (!isfinite(p1p2) || p1p2 == 0.0 || state.p1 == state.p2) {
        return false;
    }
    state.k2 = state.z1 / p1p2;
    state.k1 = (1.0 - state.k2 * (state.p1 + state.p2)) / p1p2;
    state.k4 = (state.k1 * state.p1 + state.k2) / (state.p2 - state.p1);
    state.k3 = -state.k1 - state.k4;
    const double z = (state.c1 + state.c2) / (state.rpi * state.c1 * state.c2);
    state.A = z / p1p2;
    state.B = (z - state.p1) / (state.p1 * (state.p1 - state.p2));
    state.D = (z - state.p2) / (state.p2 * (state.p2 - state.p1));
    return isfinite(state.z1) && isfinite(state.k0) &&
           isfinite(state.k1) && isfinite(state.k2) &&
           isfinite(state.k3) && isfinite(state.k4) &&
           isfinite(state.p1) && isfinite(state.p2) &&
           isfinite(state.A) && isfinite(state.B) && isfinite(state.D);
}

__device__ bool dmpFindDriverParamsLocalOnePole(DmpModel* dmp_db,
                                                const DmpGateLaneContext& ctx,
                                                DmpLocalGateState& state,
                                                double fixed_ceff) {
    double t_vth = nanf("");
    double t_vl = nanf("");
    double measured_slew = nanf("");
    dmpGateDelaysWithCtx(ctx, fixed_ceff, t_vth, t_vl, measured_slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(fixed_ceff) || !isfinite(state.rd) ||
        measured_slew <= 0.0 || fixed_ceff <= 0.0) {
        return false;
    }
    double x_dt = measured_slew / (ctx.driver_vh - ctx.driver_vl);
    double x_t0 = t_vth + log(1.0 - ctx.driver_vth) * state.rd * fixed_ceff -
                  ctx.driver_vth * x_dt;
    if (!isfinite(x_t0) || !isfinite(x_dt)) {
        return false;
    }
    for (int iter = 0; iter < 100; ++iter) {
        if (x_dt <= 0.0) {
            x_dt = (t_vl - t_vth) / 100.0;
        }
        double ignore = 0.0;
        const double y_vl = dmp_db->y(t_vl, x_t0, x_dt, state.rd, fixed_ceff);
        const double y_vth = dmp_db->y(t_vth, x_t0, x_dt, state.rd, fixed_ceff);
        const double f0 = y_vl - ctx.driver_vl;
        const double f1 = y_vth - ctx.driver_vth;
        double a00, a01, a10, a11;
        dmp_db->dy(t_vl, x_t0, x_dt, state.rd, fixed_ceff, a00, a01, ignore);
        dmp_db->dy(t_vth, x_t0, x_dt, state.rd, fixed_ceff, a10, a11, ignore);
        double p0, p1;
        if (!dmpSolve2x2(a00, a01, a10, a11, -f0, -f1, p0, p1)) {
            return false;
        }
        const bool converged = fabs(p0) <= fabs(x_t0) * dmp_db->x_tol &&
                               fabs(p1) <= fabs(x_dt) * dmp_db->x_tol;
        x_t0 += p0;
        x_dt += p1;
        if (converged) {
            if (!isfinite(x_t0) || !isfinite(x_dt) || x_dt <= 0.0) {
                return false;
            }
            state.t0_value = x_t0;
            state.dt_value = x_dt;
            state.ceff_value = fixed_ceff;
            return true;
        }
    }
    return false;
}

__device__ bool dmpFindDriverParamsLocalPi(DmpModel* dmp_db,
                                           const DmpGateLaneContext& ctx,
                                           DmpLocalGateState& state,
                                           double initial_ceff) {
    double t_vth = nanf("");
    double t_vl = nanf("");
    double measured_slew = nanf("");
    dmpGateDelaysWithCtx(ctx, initial_ceff, t_vth, t_vl, measured_slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(initial_ceff) || !isfinite(state.rd) ||
        measured_slew <= 0.0 || initial_ceff < 0.0) {
        return false;
    }
    const double cap_sum = state.c1 + state.c2;
    double x_ceff = initial_ceff;
    double x_dt = measured_slew / (ctx.driver_vh - ctx.driver_vl);
    double x_t0 = t_vth + log(1.0 - ctx.driver_vth) * state.rd * initial_ceff -
                  ctx.driver_vth * x_dt;
    if (!isfinite(x_t0) || !isfinite(x_dt) || x_dt <= 0.0) {
        return false;
    }
    for (int iter = 0; iter < 100; ++iter) {
        if (x_ceff < 0.0 || x_ceff > cap_sum || x_dt <= 0.0) {
            return false;
        }
        if (iter > 0) {
            dmpGateDelaysWithCtx(ctx, x_ceff, t_vth, t_vl, measured_slew);
            if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
                measured_slew <= 0.0) {
                return false;
            }
        }
        double ceff_time = measured_slew / (ctx.driver_vh - ctx.driver_vl);
        if (ceff_time > 1.4 * x_dt) {
            ceff_time = 1.4 * x_dt;
        }

        const double exp_p1_dt = exp2(-state.p1 * x_dt);
        const double exp_p2_dt = exp2(-state.p2 * x_dt);
        const double exp_dt_rd_ceff = exp2(-x_dt / (state.rd * x_ceff));
        const double y_vth = dmp_db->y(t_vth, x_t0, x_dt, state.rd, x_ceff);
        const double y_vl = dmp_db->y(t_vl, x_t0, x_dt, state.rd, x_ceff);
        const double f0 = dmpIpiIceffCached(state.A,
                                            state.B,
                                            state.D,
                                            state.p1,
                                            state.p2,
                                            state.rd,
                                            x_dt,
                                            ceff_time,
                                            x_ceff);
        const double f1 = y_vth - ctx.driver_vth;
        const double f2 = y_vl - ctx.driver_vl;

        const double a00 = 0.0;
        const double a01 =
            (-state.A * x_dt + state.B * x_dt * exp_p1_dt -
             (2 * state.B / state.p1) * (1.0 - exp_p1_dt) +
             state.D * x_dt * exp_p2_dt -
             (2 * state.D / state.p2) * (1.0 - exp_p2_dt) +
             state.rd * x_ceff *
                 (x_dt + x_dt * exp_dt_rd_ceff -
                  2 * state.rd * x_ceff * (1.0 - exp_dt_rd_ceff))) /
            (state.rd * x_dt * x_dt * x_dt);
        const double a02 =
            (2 * state.rd * x_ceff - x_dt -
             (2 * state.rd * x_ceff + x_dt) * exp2(-x_dt / (state.rd * x_ceff))) /
            (x_dt * x_dt);

        double a10, a11, a12;
        double a20, a21, a22;
        dmp_db->dy(t_vth, x_t0, x_dt, state.rd, x_ceff, a10, a11, a12);
        dmp_db->dy(t_vl, x_t0, x_dt, state.rd, x_ceff, a20, a21, a22);
        double p0, p1, p2;
        if (!dmpSolve3x3(a00, a01, a02,
                         a10, a11, a12,
                         a20, a21, a22,
                         -f0, -f1, -f2,
                         p0, p1, p2)) {
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
                return false;
            }
            state.t0_value = x_t0;
            state.dt_value = x_dt;
            state.ceff_value = x_ceff;
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ void dmpInitLocalGateState(DmpModel* dmp_db,
                                                      DmpLocalGateState& state) {
    state = {};
    state.alg = DMP_ALG_CAP;
    state.rd = nanf("");
    state.t0_value = nanf("");
    state.dt_value = nanf("");
    state.ceff_value = nanf("");
    state.vo_delay = nanf("");
    state.vo_slew = nanf("");
    state.gate_delay = nanf("");
    state.driver_vth = dmp_db->vth_;
    state.driver_vl = dmp_db->vl_;
    state.driver_vh = dmp_db->vh_;
    state.driver_derate = dmp_db->slew_derate_;
    state.driver_library_id = -1;
    state.dmp_valid = false;
}

__device__ __forceinline__ bool dmpComputeLocalGateStateForSlot(DmpModel* dmp_db,
                                                                const DmpGateLaneContext& ctx,
                                                                int rc_slot,
                                                                DmpLocalGateState& state) {
    dmpInitLocalGateState(dmp_db, state);
    if (!ctx.valid || rc_slot < 0 || rc_slot >= dmp_db->dmp_pin_slot_count) {
        return false;
    }

    state.driver_vth = ctx.driver_vth;
    state.driver_vl = ctx.driver_vl;
    state.driver_vh = ctx.driver_vh;
    state.driver_derate = ctx.driver_derate;
    state.driver_library_id = ctx.driver_library_id;
    state.c1 = dmp_db->C1[rc_slot];
    state.c2 = dmp_db->C2[rc_slot];
    state.rpi = dmp_db->r_pi[rc_slot];
    double table_ceff = state.c1 + state.c2;
    double table_delay = nanf("");
    double table_slew = nanf("");
    dmpGateCapDelaySlewWithCtx(ctx, table_ceff, table_delay, table_slew);
    if (!isfinite(table_delay) || !isfinite(table_slew)) {
        return false;
    }
    state.gate_delay = table_delay;
    state.vo_slew = table_slew;
    state.ceff_value = table_ceff;

    const bool rc_can_model = isfinite(state.c1) && isfinite(state.c2) &&
                              isfinite(state.rpi) && isfinite(table_ceff) &&
                              state.c1 > 0.0 && state.c2 >= 0.0 &&
                              state.rpi > 0.0;
    if (rc_can_model && dmpLocalGateModelRd(dmp_db, ctx, table_ceff, table_delay, state.rd)) {
        state.alg = dmpSelectLocalAlg(dmp_db, state);
        if (state.alg == DMP_ALG_ZERO_C2) {
            double c1_delay = nanf("");
            double c1_slew = nanf("");
            dmpGateCapDelaySlewWithCtx(ctx, state.c1, c1_delay, c1_slew);
            if (isfinite(c1_delay) && isfinite(c1_slew) &&
                dmpLocalInitZeroC2(state) &&
                dmpFindDriverParamsLocalOnePole(dmp_db, ctx, state, state.c1)) {
                double vo_delay = nanf("");
                double vo_slew = nanf("");
                if (dmpFindDriverDelaySlewCached(dmp_db,
                                                 state.alg,
                                                 state.k0,
                                                 state.k1,
                                                 state.k2,
                                                 state.k3,
                                                 state.k4,
                                                 state.p1,
                                                 state.p2,
                                                 state.t0_value,
                                                 state.dt_value,
                                                 state.c1,
                                                 state.c2,
                                                 state.rpi,
                                                 state.rd,
                                                 ctx.driver_vth,
                                                 ctx.driver_vl,
                                                 ctx.driver_vh,
                                                 ctx.driver_derate,
                                                 vo_delay,
                                                 vo_slew)) {
                    state.gate_delay = vo_delay;
                    state.vo_delay = vo_delay;
                    state.vo_slew = vo_slew;
                    state.dmp_valid = true;
                }
            }
        } else if (state.alg == DMP_ALG_PI) {
            if (dmpLocalInitPi(state)) {
                bool params_ok = dmpFindDriverParamsLocalPi(dmp_db, ctx, state, table_ceff);
                if (!params_ok && state.c2 > 0.0) {
                    params_ok = dmpFindDriverParamsLocalPi(dmp_db, ctx, state, state.c2);
                }
                if (params_ok) {
                    double ceff_delay = nanf("");
                    double ceff_slew = nanf("");
                    dmpGateCapDelaySlewWithCtx(ctx, state.ceff_value, ceff_delay, ceff_slew);
                    if (isfinite(ceff_delay) && isfinite(ceff_slew)) {
                        double vo_delay = nanf("");
                        double vo_slew = nanf("");
                        if (dmpFindDriverDelaySlewCached(dmp_db,
                                                         state.alg,
                                                         state.k0,
                                                         state.k1,
                                                         state.k2,
                                                         state.k3,
                                                         state.k4,
                                                         state.p1,
                                                         state.p2,
                                                         state.t0_value,
                                                         state.dt_value,
                                                         state.c1,
                                                         state.c2,
                                                         state.rpi,
                                                         state.rd,
                                                         ctx.driver_vth,
                                                         ctx.driver_vl,
                                                         ctx.driver_vh,
                                                         ctx.driver_derate,
                                                         vo_delay,
                                                         vo_slew)) {
                            state.gate_delay = ceff_delay;
                            state.vo_delay = vo_delay;
                            state.vo_slew = vo_slew;
                            state.dmp_valid = true;
                        }
                    }
                }
            }
        }
    }

    if (!state.dmp_valid) {
        state.alg = DMP_ALG_CAP;
        state.rd = nanf("");
        state.t0_value = nanf("");
        state.dt_value = nanf("");
        state.ceff_value = table_ceff;
        state.vo_delay = nanf("");
        state.vo_slew = table_slew;
        state.gate_delay = table_delay;
    }
    return isfinite(state.gate_delay) && isfinite(state.vo_slew);
}

__device__ __forceinline__ bool dmpComputeLocalGateState(DmpModel* dmp_db,
                                                         int arc_id,
                                                         int lane,
                                                         DmpLocalGateState& state) {
    const int el = lane >> 2;
    const int from_attr = lane >> 1;
    const int to_attr = ((lane & 0b100) >> 1) + (lane & 1);
    const int input_rf = from_attr & 1;
    const int output_rf = to_attr & 1;
    const int from_pin_id = dmp_db->timing_arc_from_pin_id[arc_id];
    const int to_pin_id = dmp_db->timing_arc_to_pin_id[arc_id];
    const int to_slot = to_pin_id * NUM_ATTR + to_attr;
    const int timing_id = dmp_db->timing_arc_id_map[arc_id * 2 + el];
    if (timing_id < 0 || from_pin_id < 0 || to_pin_id < 0 ||
        dmp_db->d_allocator == nullptr) {
        dmpInitLocalGateState(dmp_db, state);
        return false;
    }

    float input_slew = dmp_db->pinSlew[from_pin_id * NUM_ATTR + from_attr];
    if (dmpIsIdealClockTimingArc(dmp_db, timing_id, from_pin_id) &&
        !dmp_db->d_allocator->d_is_constraint[timing_id]) {
        input_slew = dmpIdealClockSlew(dmp_db, from_pin_id, from_attr);
    }
    const DmpGateLaneContext ctx =
        dmpMakeGateLaneContextDirect(dmp_db, timing_id, input_rf, to_attr, output_rf, input_slew);
    return dmpComputeLocalGateStateForSlot(dmp_db, ctx, to_slot, state);
}

__device__ __forceinline__ bool dmpComputeDrivingCellLocalState(DmpModel* dmp_db,
                                                               int pin_slot,
                                                               int timing_id,
                                                               int input_rf,
                                                               int output_rf,
                                                               float input_slew,
                                                               DmpLocalGateState& state) {
    const int attr = pin_slot & (NUM_ATTR - 1);
    if (dmp_db->d_allocator == nullptr || pin_slot < 0 ||
        pin_slot >= dmp_db->dmp_pin_slot_count ||
        timing_id < 0 || input_rf < 0 || output_rf < 0 ||
        !isfinite(input_slew)) {
        dmpInitLocalGateState(dmp_db, state);
        return false;
    }
    const DmpGateLaneContext ctx =
        dmpMakeGateLaneContextDirect(dmp_db, timing_id, input_rf, attr, output_rf, input_slew);
    return dmpComputeLocalGateStateForSlot(dmp_db, ctx, pin_slot, state);
}

__device__ __forceinline__ bool dmpUpdateSlewWinnerValue(DmpModel* dmp_db,
                                                         int to_slot,
                                                         float slew,
                                                         bool pick_max) {
    if (!isfinite(slew)) {
        return false;
    }
    const unsigned long long packed = dmpPackWinner(slew, 0u, pick_max);
    const unsigned long long old = atomicMax(&dmp_db->pin_slew_winner[to_slot], packed);
    return packed > old;
}

__device__ __forceinline__ void dmpLoadDelaySlewFromLocalState(DmpModel* dmp_db,
                                                               const DmpLocalGateState& state,
                                                               int net_arc_id,
                                                               int load_attr,
                                                               double& wire_delay,
                                                               double& load_slew) {
    const int to_pin_id = dmp_db->timing_arc_to_pin_id[net_arc_id];
    const double elmore = dmp_db->elmore_delay[to_pin_id * NUM_ATTR + load_attr];
    const double drvr_slew = state.vo_slew;
    wire_delay = elmore;
    load_slew = drvr_slew;
    if (!isfinite(drvr_slew) || !isfinite(elmore)) {
        wire_delay = nanf("");
        load_slew = nanf("");
        return;
    }
    const bool driver_valid = state.alg != DMP_ALG_CAP &&
                              isfinite(state.rd) && state.rd > 0.0 &&
                              isfinite(state.t0_value) &&
                              isfinite(state.dt_value) && state.dt_value > 0.0 &&
                              isfinite(state.vo_delay);
    const double driver_vth = state.driver_vth;
    const double driver_vl = state.driver_vl;
    const double driver_vh = state.driver_vh;
    const double driver_derate = state.driver_derate;
    if (!isfinite(driver_vth) || !isfinite(driver_vl) || !isfinite(driver_vh) ||
        !isfinite(driver_derate) || driver_derate <= 0.0) {
        wire_delay = nanf("");
        load_slew = nanf("");
        return;
    }
    if (!driver_valid || elmore == 0.0 || elmore < drvr_slew * 1e-3) {
        dmpThresholdAdjustCuda(dmp_db, to_pin_id, load_attr,
                               driver_vth, driver_vl, driver_vh, driver_derate,
                               state.driver_library_id,
                               wire_delay, load_slew);
        return;
    }

    const double t_lower = state.t0_value;
    const double t_upper =
        dmpSlotVoUpperBoundCached(state.alg,
                                  state.t0_value,
                                  state.dt_value,
                                  state.c1,
                                  state.c2,
                                  state.rpi,
                                  state.rd) +
        elmore * 2.0;
    const double load_delay =
        dmpFindRootVlCached(dmp_db,
                            state.alg,
                            state.k0,
                            state.k1,
                            state.k2,
                            state.k3,
                            state.k4,
                            state.p1,
                            state.p2,
                            state.t0_value,
                            state.dt_value,
                            elmore,
                            driver_vth,
                            t_lower,
                            t_upper);
    const double tl =
        dmpFindRootVlCached(dmp_db,
                            state.alg,
                            state.k0,
                            state.k1,
                            state.k2,
                            state.k3,
                            state.k4,
                            state.p1,
                            state.p2,
                            state.t0_value,
                            state.dt_value,
                            elmore,
                            driver_vl,
                            t_lower,
                            load_delay);
    const double th =
        dmpFindRootVlCached(dmp_db,
                            state.alg,
                            state.k0,
                            state.k1,
                            state.k2,
                            state.k3,
                            state.k4,
                            state.p1,
                            state.p2,
                            state.t0_value,
                            state.dt_value,
                            elmore,
                            driver_vh,
                            load_delay,
                            t_upper);
    double delay1 = load_delay - state.vo_delay;
    double slew1 = (th - tl) / driver_derate;
    if (!isfinite(load_delay) || !isfinite(tl) || !isfinite(th) ||
        !isfinite(slew1) || !isfinite(delay1)) {
        return;
    }
    if (delay1 < 0.0) {
        if (-delay1 > dmp_db->vth_time_tol * state.vo_delay) {
            return;
        }
        delay1 = elmore;
    }
    if (slew1 < drvr_slew) {
        if ((drvr_slew - slew1) > dmp_db->vth_time_tol * drvr_slew) {
            return;
        }
        slew1 = drvr_slew;
    }
    wire_delay = delay1;
    load_slew = slew1;
    dmpThresholdAdjustCuda(dmp_db, to_pin_id, load_attr,
                           driver_vth, driver_vl, driver_vh, driver_derate,
                           state.driver_library_id,
                           wire_delay, load_slew);
}

__device__ void DmpModel::propagateLoadSlewDelay() {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int attr = idx & 0b11;
    const int arc_id = arc_ids[idx];
    const int from_pin_id = timing_arc_from_pin_id[arc_id];
    const int to_pin_id = timing_arc_to_pin_id[arc_id];
    const int from_slot = from_pin_id * NUM_ATTR + attr;
    const int to_slot = to_pin_id * NUM_ATTR + attr;
    const double elmore = elmore_delay[to_slot];

    float source_slew = pinSlew[from_slot];
    const bool has_driving_cell =
        driving_cell_timing_id != nullptr &&
        driving_cell_input_rf != nullptr &&
        driving_cell_input_slew != nullptr &&
        driving_cell_timing_id[from_slot] >= 0 &&
        driving_cell_input_rf[from_slot] >= 0 &&
        isfinite(driving_cell_input_slew[from_slot]);
    if (isnan(source_slew) && !has_driving_cell) {
        return;
    }

    double final_delay = nanf("");
    double final_slew = nanf("");
    bool used_driving_cell = false;
    bool used_dmp_load = false;
    double debug_extra_delay = nanf("");
    double debug_vo_delay = nanf("");
    int debug_alg = DMP_ALG_CAP;

    if (has_driving_cell) {
        const int timing_id = driving_cell_timing_id[from_slot];
        const int input_rf = driving_cell_input_rf[from_slot];
        const int output_rf = attr & 1;
        const float input_slew = driving_cell_input_slew[from_slot];
        DmpLocalGateState state;
        if (dmpComputeDrivingCellLocalState(this,
                                            from_slot,
                                            timing_id,
                                            input_rf,
                                            output_rf,
                                            input_slew,
                                            state)) {
            double wire_delay = nanf("");
            double load_slew = nanf("");
            dmpLoadDelaySlewFromLocalState(this, state, arc_id, attr, wire_delay, load_slew);

            double intrinsic_delay = nanf("");
            double intrinsic_slew = nanf("");
            dmpGateCapDelaySlewCached(this,
                                      timing_id,
                                      input_rf,
                                      output_rf,
                                      input_slew,
                                      0.0,
                                      intrinsic_delay,
                                      intrinsic_slew);
            if (isfinite(intrinsic_delay) && isfinite(state.gate_delay)) {
                debug_extra_delay = state.gate_delay - intrinsic_delay;
                wire_delay += debug_extra_delay;
            }
            if (isfinite(wire_delay) && isfinite(load_slew)) {
                final_delay = wire_delay;
                final_slew = load_slew;
                source_slew = static_cast<float>(state.vo_slew);
                used_driving_cell = true;
                used_dmp_load = state.dmp_valid;
                debug_vo_delay = state.vo_delay;
                debug_alg = state.alg;
            }
        }
    }

    if (!used_driving_cell) {
        if (isnan(source_slew)) {
            return;
        }
        double driver_vth, driver_vl, driver_vh, driver_derate;
        const int driver_library_id = dmpPinLibraryId(this, from_pin_id, attr);
        dmpDriverLibraryThresholds(this,
                                   driver_library_id,
                                   attr,
                                   driver_vth,
                                   driver_vl,
                                   driver_vh,
                                   driver_derate);
        final_delay = elmore;
        final_slew = source_slew;
        if (pin_is_primary_input != nullptr && pin_is_primary_input[from_pin_id]) {
            dmpInputPortDelaySlewCuda(this,
                                      to_pin_id,
                                      attr,
                                      source_slew,
                                      elmore,
                                      final_delay,
                                      final_slew);
        } else {
            dmpThresholdAdjustCuda(this,
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
    pinSlew[to_slot] = static_cast<float>(final_slew);
    arcDelay[arc_id * 2 * NUM_ATTR + delay_idx] = static_cast<float>(final_delay);

    if (DMP_DIRECT_CLOCK_DEBUG_PRINT &&
        pin_names != nullptr &&
        dmpStringEquals(pin_names[from_pin_id], "clk") &&
        dmpStringEquals(pin_names[to_pin_id], "clkbuf_0_clk:A")) {
        const float src_at = pinAt[from_slot];
        const double cand_at = isfinite(src_at) && isfinite(final_delay)
                                   ? static_cast<double>(src_at) + final_delay
                                   : nan("");
        printf("[DMP DIRECT CLOCK] attr=%d source_slew=%.9f load_slew=%.9f elmore=%.9f extra_delay=%.9f vo_delay=%.9f wire_delay=%.9f at=%.9f alg=%d dmp_load=%d driving_cell=%d\n",
               attr,
               static_cast<double>(source_slew),
               final_slew,
               elmore,
               debug_extra_delay,
               debug_vo_delay,
               final_delay,
               cand_at,
               debug_alg,
               used_dmp_load ? 1 : 0,
               used_driving_cell ? 1 : 0);
    }
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

    DmpLocalGateState state;
    if (!dmpComputeLocalGateState(dmp_db, gate_arc_id, lane, state)) {
        dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_SCRATCH_SKIPS);
        return;
    }

    const int el = lane >> 2;
    const int from_attr = lane >> 1;
    const int to_attr = ((lane & 0b100) >> 1) + (lane & 1);
    const int output_rf = to_attr & 1;
    const int from_pin_id = dmp_db->timing_arc_from_pin_id[gate_arc_id];
    const int to_pin_id = dmp_db->timing_arc_to_pin_id[gate_arc_id];
    const int to_slot = to_pin_id * NUM_ATTR + to_attr;
    dmp_db->arcDelay[gate_arc_id * 2 * NUM_ATTR + lane] = static_cast<float>(state.gate_delay);
    dmpUpdateSlewWinnerValue(dmp_db, to_slot, static_cast<float>(state.vo_slew), el != 0);

    const int timing_id = dmp_db->timing_arc_id_map[gate_arc_id * 2 + el];
    const bool ideal_clock_arc = dmpIsIdealClockTimingArc(dmp_db, timing_id, from_pin_id) &&
                                 !dmp_db->d_allocator->d_is_constraint[timing_id];
    float from_at = ideal_clock_arc
                        ? dmpIdealClockEdgeTime(dmp_db, timing_id, from_pin_id)
                        : dmp_db->pinAt[from_pin_id * NUM_ATTR + from_attr];
    if (ideal_clock_arc && isnan(from_at)) {
        from_at = dmp_db->pinAt[from_pin_id * NUM_ATTR + from_attr];
    }
    if (!isnan(from_at)) {
        dmp_db->updateAtWinner(to_slot,
                               from_at + static_cast<float>(state.gate_delay),
                               el != 0,
                               from_pin_id,
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
        double wire_delay = nanf("");
        double load_slew = nanf("");
        dmpLoadDelaySlewFromLocalState(dmp_db,
                                       state,
                                       net_arc_id,
                                       load_attr,
                                       wire_delay,
                                       load_slew);
        if (!isfinite(wire_delay) || !isfinite(load_slew)) {
            dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_SCRATCH_SKIPS);
            continue;
        }
        dmpGateNetPairCount(debug_counts, DMP_GNP_FINITE_CANDIDATES);
        dmp_db->updateLoadWinner(net_arc_id,
                                 load_attr,
                                 static_cast<float>(wire_delay),
                                 static_cast<float>(load_slew));
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
        if (counts != nullptr) {
            atomicAdd(&counts[DMP_DRIVING_CELL_SKIPPED], 1ULL);
        }
        return;
    }

    const int pin_slot = pin_id * NUM_ATTR + attr;
    const int output_rf = attr & 1;
    DmpLocalGateState state;
    if (!dmpComputeDrivingCellLocalState(dmp_db,
                                         pin_slot,
                                         timing_id,
                                         input_rf,
                                         output_rf,
                                         input_slews[idx],
                                         state)) {
        if (counts != nullptr) {
            atomicAdd(&counts[DMP_DRIVING_CELL_SKIPPED], 1ULL);
        }
        return;
    }

    const float intrinsic_delay = dmp_db->d_allocator->query(timing_id,
                                                            input_rf,
                                                            output_rf,
                                                            input_slews[idx],
                                                            0.0f,
                                                            0);
    const double extra_delay = isfinite(intrinsic_delay)
                                   ? state.gate_delay - static_cast<double>(intrinsic_delay)
                                   : nanf("");

    dmp_db->driving_cell_timing_id[pin_slot] = timing_id;
    dmp_db->driving_cell_input_rf[pin_slot] = input_rf;
    dmp_db->driving_cell_input_slew[pin_slot] = input_slews[idx];
    dmp_db->pinSlew[pin_slot] = static_cast<float>(state.vo_slew);

    if (counts != nullptr) {
        atomicAdd(&counts[DMP_DRIVING_CELL_APPLIED], 1ULL);
        if (state.alg == DMP_ALG_ZERO_C2) {
            atomicAdd(&counts[DMP_DRIVING_CELL_ZERO_C2], 1ULL);
        } else if (state.alg == DMP_ALG_PI) {
            atomicAdd(&counts[DMP_DRIVING_CELL_PI], 1ULL);
        } else {
            atomicAdd(&counts[DMP_DRIVING_CELL_CAP], 1ULL);
        }
        if (state.dmp_valid) {
            atomicAdd(&counts[DMP_DRIVING_CELL_DMP_VALID], 1ULL);
        } else {
            atomicAdd(&counts[DMP_DRIVING_CELL_FALLBACK], 1ULL);
        }
    }

    if (DMP_DRIVING_CELL_DEBUG_PRINT &&
        dmp_db->pin_names != nullptr &&
        dmpStringEquals(dmp_db->pin_names[pin_id], "clk")) {
        printf("[DMP DRIVING CELL DBG] pin=clk attr=%d alg=%d dmp_valid=%d input_slew=%.9f source_slew=%.9f gate_delay=%.9f intrinsic=%.9f extra=%.9f C1=%.9e C2=%.9e rpi=%.9e rd=%.9e t0=%.9e dt=%.9e ceff=%.9e\n",
               attr,
               state.alg,
               state.dmp_valid ? 1 : 0,
               static_cast<double>(input_slews[idx]),
               state.vo_slew,
               state.gate_delay,
               static_cast<double>(intrinsic_delay),
               extra_delay,
               state.c1,
               state.c2,
               state.rpi,
               state.rd,
               state.t0_value,
               state.dt_value,
               state.ceff_value);
    }
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
            printf("[DMP DRIVING CELL] sources=0 lanes=0 applied=0 skipped=0 cap=0 zero_c2=0 pi=0 dmp_valid=0 fallback=0\n");
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
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaEventElapsedTime(&elapsed_ms, start, stop));
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));
        cudaGetLastError();
    }
    if (collect_counts) {
        gpuErrchk(cudaMemcpy(h_counts, d_counts, sizeof(h_counts), cudaMemcpyDeviceToHost));

        printf("[DMP DRIVING CELL] sources=%d lanes=%d applied=%llu skipped=%llu cap=%llu zero_c2=%llu pi=%llu dmp_valid=%llu fallback=%llu\n",
               num_sources,
               total,
               h_counts[DMP_DRIVING_CELL_APPLIED],
               h_counts[DMP_DRIVING_CELL_SKIPPED],
               h_counts[DMP_DRIVING_CELL_CAP],
               h_counts[DMP_DRIVING_CELL_ZERO_C2],
               h_counts[DMP_DRIVING_CELL_PI],
               h_counts[DMP_DRIVING_CELL_DMP_VALID],
               h_counts[DMP_DRIVING_CELL_FALLBACK]);
    }
    if (profile_kernels) {
        printf("[DMP KERNEL PROFILE] name=applyDrivingCellSourceSlewKernel launches=1 total_ms=%.3f avg_us=%.3f max_ms=%.3f work_items=%d blocks=%d block=(%d,1) work_per_ms=%.1f\n",
               elapsed_ms,
               static_cast<double>(elapsed_ms) * 1000.0,
               elapsed_ms,
               total,
               DMP_TIMING_BLOCK_NUMBER(total),
               DMP_TIMING_BLOCK_SIZE,
               elapsed_ms > 0.0f ? static_cast<double>(total) / static_cast<double>(elapsed_ms) : 0.0);
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
