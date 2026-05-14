__device__ void
dmp_model::findDriverDelaySlew(int pin_idx, double &delay, double &slew){
  double t_upper = voCrossingUpperBound(pin_idx);
  double driver_vth, driver_vl, driver_vh, driver_derate;
  dmpLoadSlotThresholds(this, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
  delay = findVoCrossing(driver_vth, t0[pin_idx], t_upper);
  if(isnan(delay)){
    delay = slew = nanf("");
    return ;
  }
  double tl = findVoCrossing(driver_vl, t0[pin_idx], delay);
  double th = findVoCrossing(driver_vh, delay, t_upper);
  if(isnan(tl) || isnan(th)){
    delay = slew = nanf("");
    return ;
  }
  // Convert measured slew to table slew.
  slew = (th - tl) / driver_derate;
}

enum DmpDrivingCellCounter {
    DMP_DRIVING_CELL_APPLIED = 0,
    DMP_DRIVING_CELL_SKIPPED = 1,
    DMP_DRIVING_CELL_CAP = 2,
    DMP_DRIVING_CELL_ZERO_C2 = 3,
    DMP_DRIVING_CELL_PI = 4,
    DMP_DRIVING_CELL_DMP_VALID = 5,
    DMP_DRIVING_CELL_FALLBACK = 6,
    DMP_DRIVING_CELL_COUNTER_COUNT = 7
};

__device__ __forceinline__ void dmpVirtualGateCapDelaySlew(dmp_model* dmp_db,
                                                           int timing_id,
                                                           int input_rf,
                                                           int output_rf,
                                                           float input_slew,
                                                           double lc,
                                                           double& delay,
                                                           double& slew) {
    delay = nanf("");
    slew = nanf("");
    if (dmp_db->d_allocator == nullptr ||
        timing_id < 0 || input_rf < 0 || output_rf < 0 ||
        !isfinite(input_slew) || !isfinite(lc) || lc < 0.0) {
        return;
    }
    const float load = static_cast<float>(lc);
    delay = dmp_db->d_allocator->query(timing_id, input_rf, output_rf, input_slew, load, 0);
    slew = dmp_db->d_allocator->query(timing_id, input_rf, output_rf, input_slew, load, 1);
}

__device__ __forceinline__ void dmpVirtualGateDelays(dmp_model* dmp_db,
                                                     int pin_idx,
                                                     int timing_id,
                                                     int input_rf,
                                                     int output_rf,
                                                     float input_slew,
                                                     double lc,
                                                     double& t_vth,
                                                     double& t_vl,
                                                     double& slew) {
    t_vth = nanf("");
    t_vl = nanf("");
    slew = nanf("");
    double table_slew = nanf("");
    dmpVirtualGateCapDelaySlew(dmp_db, timing_id, input_rf, output_rf, input_slew, lc, t_vth, table_slew);
    if (!isfinite(t_vth) || !isfinite(table_slew)) {
        return;
    }
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(dmp_db, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    const double driver_delta = driver_vh - driver_vl;
    if (!isfinite(driver_delta) || driver_delta <= 0.0 || !isfinite(driver_derate)) {
        return;
    }
    slew = table_slew * driver_derate;
    t_vl = t_vth - slew * (driver_vth - driver_vl) / driver_delta;
}

__device__ __forceinline__ void dmpVirtualGateModelRd(dmp_model* dmp_db,
                                                      int pin_idx,
                                                      int timing_id,
                                                      int input_rf,
                                                      int output_rf,
                                                      float input_slew,
                                                      double d1) {
    double cap1 = dmp_db->C1[pin_idx] + dmp_db->C2[pin_idx];
    const double kGateModelRdCapDelta = 1e-15 / dmp_db->cap_unit;
    double cap2 = cap1 + kGateModelRdCapDelta;
    double d2 = nanf("");
    double s2 = nanf("");
    if (!isfinite(cap1) || !isfinite(cap2) || !isfinite(d1) ||
        !isfinite(kGateModelRdCapDelta) || kGateModelRdCapDelta <= 0.0 ||
        cap2 == cap1) {
        dmp_db->rd_[pin_idx] = nanf("");
        return;
    }
    dmpVirtualGateCapDelaySlew(dmp_db, timing_id, input_rf, output_rf, input_slew, cap2, d2, s2);
    if (!isfinite(d2)) {
        dmp_db->rd_[pin_idx] = nanf("");
        return;
    }
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(dmp_db, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    dmp_db->rd_[pin_idx] = -log(driver_vth) * fabs(d1 - d2) / (cap2 - cap1);
    if (!isfinite(dmp_db->rd_[pin_idx]) || dmp_db->rd_[pin_idx] <= 0.0) {
        dmp_db->rd_[pin_idx] = nanf("");
    }
}

__device__ bool dmpVirtualEvalDmpEqns(dmp_model* dmp_db,
                                      int pin_idx,
                                      int timing_id,
                                      int input_rf,
                                      int output_rf,
                                      float input_slew,
                                      double* x_,
                                      double (*fjac_)[3],
                                      double* fvec_,
                                      int size) {
    double t0 = x_[DmpParam::t0];
    double dt = x_[DmpParam::dt];
    double ceff = x_[DmpParam::ceff];
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(dmp_db, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    if (ceff < 0.0 || ceff > (dmp_db->C1[pin_idx] + dmp_db->C2[pin_idx])) {
        return false;
    }

    double t_vth, t_vl, slew;
    dmpVirtualGateDelays(dmp_db, pin_idx, timing_id, input_rf, output_rf, input_slew, ceff, t_vth, t_vl, slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(slew) || slew == 0.0) {
        return false;
    }

    double ceff_time = slew / (driver_vh - driver_vl);
    if (ceff_time > 1.4 * dt) {
        ceff_time = 1.4 * dt;
    }

    if (size == 2) {
        if (dt <= 0.0) {
            dt = x_[DmpParam::dt] = (t_vl - t_vth) / 100.0;
        }
        double ignore;
        double y50 = dmp_db->y(t_vth, t0, dt, dmp_db->rd_[pin_idx], ceff);
        double y20 = dmp_db->y(t_vl, t0, dt, dmp_db->rd_[pin_idx], ceff);
        fvec_[DmpFunc::y50] = y50 - driver_vth;
        fvec_[DmpFunc::y20] = y20 - driver_vl;
        dmp_db->dy(t_vl, t0, dt, dmp_db->rd_[pin_idx], ceff, fjac_[DmpFunc::y20][DmpParam::t0], fjac_[DmpFunc::y20][DmpParam::dt], ignore);
        dmp_db->dy(t_vth, t0, dt, dmp_db->rd_[pin_idx], ceff, fjac_[DmpFunc::y50][DmpParam::t0], fjac_[DmpFunc::y50][DmpParam::dt], ignore);
        return true;
    }
    if (dt <= 0.0) {
        return false;
    }
    double exp_p1_dt = exp2(-dmp_db->p1_[pin_idx] * dt);
    double exp_p2_dt = exp2(-dmp_db->p2_[pin_idx] * dt);
    double exp_dt_rd_ceff = exp2(-dt / (dmp_db->rd_[pin_idx] * ceff));

    double y50 = dmp_db->y(t_vth, t0, dt, dmp_db->rd_[pin_idx], ceff);
    double y20 = dmp_db->y(t_vl, t0, dt, dmp_db->rd_[pin_idx], ceff);
    fvec_[DmpFunc::ipi] = dmp_db->ipiIceff(pin_idx, dt, ceff_time, ceff);
    fvec_[DmpFunc::y50] = y50 - driver_vth;
    fvec_[DmpFunc::y20] = y20 - driver_vl;
    fjac_[DmpFunc::ipi][DmpParam::t0] = 0.0;
    fjac_[DmpFunc::ipi][DmpParam::dt] =
        (-dmp_db->A_[pin_idx] * dt + dmp_db->B_[pin_idx] * dt * exp_p1_dt - (2 * dmp_db->B_[pin_idx] / dmp_db->p1_[pin_idx]) * (1.0 - exp_p1_dt) + dmp_db->D_[pin_idx] * dt * exp_p2_dt - (2 * dmp_db->D_[pin_idx] / dmp_db->p2_[pin_idx]) * (1.0 - exp_p2_dt) + dmp_db->rd_[pin_idx] * ceff * (dt + dt * exp_dt_rd_ceff - 2 * dmp_db->rd_[pin_idx] * ceff * (1.0 - exp_dt_rd_ceff))) / (dmp_db->rd_[pin_idx] * dt * dt * dt);
    fjac_[DmpFunc::ipi][DmpParam::ceff] =
        (2 * dmp_db->rd_[pin_idx] * ceff - dt - (2 * dmp_db->rd_[pin_idx] * ceff + dt) * exp2(-dt / (dmp_db->rd_[pin_idx] * ceff))) / (dt * dt);

    dmp_db->dy(t_vl, t0, dt, dmp_db->rd_[pin_idx], ceff, fjac_[DmpFunc::y20][DmpParam::t0], fjac_[DmpFunc::y20][DmpParam::dt], fjac_[DmpFunc::y20][DmpParam::ceff]);
    dmp_db->dy(t_vth, t0, dt, dmp_db->rd_[pin_idx], ceff, fjac_[DmpFunc::y50][DmpParam::t0], fjac_[DmpFunc::y50][DmpParam::dt], fjac_[DmpFunc::y50][DmpParam::ceff]);

    return true;
}

__device__ bool dmpVirtualNewtonRaphson(dmp_model* dmp_db,
                                        int pin_idx,
                                        int timing_id,
                                        int input_rf,
                                        int output_rf,
                                        float input_slew,
                                        int max_iter,
                                        int size,
                                        double* x,
                                        double (*fjac)[3],
                                        double* fvec,
                                        int* index,
                                        double* p,
                                        double* scale) {
    for (int k = 0; k < max_iter; k++) {
        if (!dmpVirtualEvalDmpEqns(dmp_db, pin_idx, timing_id, input_rf, output_rf, input_slew, x, fjac, fvec, size)) {
            return false;
        }
        for (int i = 0; i < size; i++) {
            p[i] = -fvec[i];
        }
        if (!luDecomp(fjac, size, index, scale)) {
            return false;
        }
        luSolve(fjac, size, index, p);

        bool all_under_x_tol = true;
        for (int i = 0; i < size; i++) {
            if (abs(p[i]) > abs(x[i]) * dmp_db->x_tol) {
                all_under_x_tol = false;
            }
            x[i] += p[i];
        }
        if (all_under_x_tol) {
            return dmpVirtualEvalDmpEqns(dmp_db, pin_idx, timing_id, input_rf, output_rf, input_slew, x, fjac, fvec, size);
        }
    }
    return false;
}

__device__ bool dmpVirtualFindDriverParams(dmp_model* dmp_db,
                                           int pin_idx,
                                           int timing_id,
                                           int input_rf,
                                           int output_rf,
                                           float input_slew,
                                           double initial_ceff) {
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(dmp_db, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    double t_vth, t_vl, measured_slew;
    dmpVirtualGateDelays(dmp_db, pin_idx, timing_id, input_rf, output_rf, input_slew, initial_ceff, t_vth, t_vl, measured_slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(initial_ceff) || measured_slew <= 0.0 || initial_ceff < 0.0) {
        return false;
    }
    double init_dt = measured_slew / (driver_vh - driver_vl);
    double init_t0 = t_vth + log(1.0 - driver_vth) * dmp_db->rd_[pin_idx] * initial_ceff - driver_vth * init_dt;
    if (!isfinite(init_t0) || !isfinite(init_dt) || init_dt <= 0.0) {
        return false;
    }
    double x_[3];
    double fjac_[3][3];
    double fvec_[3];
    int index[3];
    double p[3];
    double scale[3] = {1.0, 1.0, 1.0};
    x_[DmpParam::t0] = init_t0;
    x_[DmpParam::dt] = init_dt;
    x_[DmpParam::ceff] = initial_ceff;
    if (!dmpVirtualNewtonRaphson(dmp_db, pin_idx, timing_id, input_rf, output_rf, input_slew, 100, 3, x_, fjac_, fvec_, index, p, scale)) {
        return false;
    }
    if (!isfinite(x_[DmpParam::t0]) || !isfinite(x_[DmpParam::dt]) ||
        !isfinite(x_[DmpParam::ceff]) || x_[DmpParam::dt] <= 0.0 ||
        x_[DmpParam::ceff] < 0.0 || x_[DmpParam::ceff] > dmp_db->C1[pin_idx] + dmp_db->C2[pin_idx]) {
        return false;
    }
    dmp_db->t0[pin_idx] = x_[DmpParam::t0];
    dmp_db->dt[pin_idx] = x_[DmpParam::dt];
    dmp_db->ceff[pin_idx] = x_[DmpParam::ceff];
    return true;
}

__device__ bool dmpVirtualFindDriverParamsOnePole(dmp_model* dmp_db,
                                                  int pin_idx,
                                                  int timing_id,
                                                  int input_rf,
                                                  int output_rf,
                                                  float input_slew,
                                                  double fixed_ceff) {
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(dmp_db, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    double t_vth, t_vl, measured_slew;
    dmpVirtualGateDelays(dmp_db, pin_idx, timing_id, input_rf, output_rf, input_slew, fixed_ceff, t_vth, t_vl, measured_slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(fixed_ceff) || measured_slew <= 0.0 || fixed_ceff <= 0.0) {
        return false;
    }
    double init_dt = measured_slew / (driver_vh - driver_vl);
    double init_t0 = t_vth + log(1.0 - driver_vth) * dmp_db->rd_[pin_idx] * fixed_ceff - driver_vth * init_dt;
    if (!isfinite(init_t0) || !isfinite(init_dt)) {
        return false;
    }
    double x_[3] = {0.0, 0.0, fixed_ceff};
    double fjac_[3][3];
    double fvec_[3];
    int index[3];
    double p[3];
    double scale[3] = {1.0, 1.0, 1.0};
    x_[DmpParam::t0] = init_t0;
    x_[DmpParam::dt] = init_dt;
    x_[DmpParam::ceff] = fixed_ceff;
    if (!dmpVirtualNewtonRaphson(dmp_db, pin_idx, timing_id, input_rf, output_rf, input_slew, 100, 2, x_, fjac_, fvec_, index, p, scale)) {
        return false;
    }
    if (!isfinite(x_[DmpParam::t0]) || !isfinite(x_[DmpParam::dt]) ||
        x_[DmpParam::dt] <= 0.0) {
        return false;
    }
    dmp_db->t0[pin_idx] = x_[DmpParam::t0];
    dmp_db->dt[pin_idx] = x_[DmpParam::dt];
    dmp_db->ceff[pin_idx] = fixed_ceff;
    return true;
}

__device__ __forceinline__ void dmpV0Explicit(dmp_model* dmp_db,
                                              int pin_idx,
                                              double t,
                                              double& vo,
                                              double& dvo_dt) {
    if (dmp_db->dmp_alg_kind[pin_idx] == DMP_ALG_CAP) {
        vo = 0.0;
        dvo_dt = 0.0;
        return;
    }
    double exp_p1 = exp2(-dmp_db->p1_[pin_idx] * t);
    if (dmp_db->dmp_alg_kind[pin_idx] == DMP_ALG_ZERO_C2) {
        vo = dmp_db->k0_[pin_idx] * (dmp_db->k1_[pin_idx] + dmp_db->k2_[pin_idx] * t + dmp_db->k3_[pin_idx] * exp_p1);
        dvo_dt = dmp_db->k0_[pin_idx] * (dmp_db->k2_[pin_idx] - dmp_db->k3_[pin_idx] * dmp_db->p1_[pin_idx] * exp_p1);
        return;
    }
    double exp_p2 = exp2(-dmp_db->p2_[pin_idx] * t);
    vo = dmp_db->k0_[pin_idx] * (dmp_db->k1_[pin_idx] + dmp_db->k2_[pin_idx] * t + dmp_db->k3_[pin_idx] * exp_p1 + dmp_db->k4_[pin_idx] * exp_p2);
    dvo_dt = dmp_db->k0_[pin_idx] * (dmp_db->k2_[pin_idx] - dmp_db->k3_[pin_idx] * dmp_db->p1_[pin_idx] * exp_p1 - dmp_db->k4_[pin_idx] * dmp_db->p2_[pin_idx] * exp_p2);
}

__device__ __forceinline__ void dmpVoExplicit(dmp_model* dmp_db,
                                             int pin_idx,
                                             double t,
                                             double& vo,
                                             double& dvo_dt) {
    double t1 = t - dmp_db->t0[pin_idx];
    if (t1 <= 0.0) {
        vo = 0.0;
        dvo_dt = 0.0;
    } else if (t1 <= dmp_db->dt[pin_idx]) {
        double v0, dv0_dt;
        dmpV0Explicit(dmp_db, pin_idx, t1, v0, dv0_dt);
        vo = v0 / dmp_db->dt[pin_idx];
        dvo_dt = dv0_dt / dmp_db->dt[pin_idx];
    } else {
        double v0, dv0_dt;
        dmpV0Explicit(dmp_db, pin_idx, t1, v0, dv0_dt);
        double v0_dt, dv0_dt_dt;
        dmpV0Explicit(dmp_db, pin_idx, t1 - dmp_db->dt[pin_idx], v0_dt, dv0_dt_dt);
        vo = (v0 - v0_dt) / dmp_db->dt[pin_idx];
        dvo_dt = (dv0_dt - dv0_dt_dt) / dmp_db->dt[pin_idx];
    }
}

__device__ __forceinline__ void dmpVoFuncExplicit(dmp_model* dmp_db,
                                                 int pin_idx,
                                                 double vth,
                                                 double t,
                                                 double& y,
                                                 double& dy) {
    double vo, vo_dt;
    dmpVoExplicit(dmp_db, pin_idx, t, vo, vo_dt);
    y = vo - vth;
    dy = vo_dt;
}

__device__ double dmpFindRootVoExplicit(dmp_model* dmp_db,
                                        int pin_idx,
                                        double vth,
                                        double x1,
                                        double x2) {
    double y1, y2, dy;
    dmpVoFuncExplicit(dmp_db, pin_idx, vth, x1, y1, dy);
    dmpVoFuncExplicit(dmp_db, pin_idx, vth, x2, y2, dy);
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
        double xtemp = x1;
        x1 = x2;
        x2 = xtemp;
    }
    double root = (x1 + x2) / 2.0;
    double dx_prev = abs(x2 - x1);
    double dx = dx_prev;
    double y;
    dmpVoFuncExplicit(dmp_db, pin_idx, vth, root, y, dy);
    for (int iter = 0; iter < dmp_db->MAX_ITER; iter++) {
        if ((((x2 - root) * dy + y) * ((x1 - root) * dy + y) > 0.0) ||
            (abs(2.0 * y) > abs(dx_prev * dy))) {
            dx_prev = dx;
            dx = (x2 - x1) * 0.5;
            root = x1 + dx;
        } else {
            dx_prev = dx;
            dx = y / dy;
            root -= dx;
        }
        if (abs(dx) <= dmp_db->x_tol * abs(root)) {
            return root;
        }

        dmpVoFuncExplicit(dmp_db, pin_idx, vth, root, y, dy);
        if (y < 0.0) {
            x1 = root;
        } else {
            x2 = root;
        }
    }
    return nanf("");
}

__device__ bool dmpVirtualFindDriverDelaySlew(dmp_model* dmp_db,
                                              int pin_idx,
                                              double& delay,
                                              double& slew) {
    double t_upper = dmp_db->voCrossingUpperBound(pin_idx);
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(dmp_db, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    delay = dmpFindRootVoExplicit(dmp_db, pin_idx, driver_vth, dmp_db->t0[pin_idx], t_upper);
    if (!isfinite(delay)) {
        delay = slew = nanf("");
        return false;
    }
    double tl = dmpFindRootVoExplicit(dmp_db, pin_idx, driver_vl, dmp_db->t0[pin_idx], delay);
    double th = dmpFindRootVoExplicit(dmp_db, pin_idx, driver_vh, delay, t_upper);
    if (!isfinite(tl) || !isfinite(th)) {
        delay = slew = nanf("");
        return false;
    }
    slew = (th - tl) / driver_derate;
    return isfinite(slew);
}

__device__ bool dmpApplyVirtualDrivingCellSource(dmp_model* dmp_db,
                                                 int pin_idx,
                                                 int timing_id,
                                                 int input_rf,
                                                 int output_rf,
                                                 float input_slew,
                                                 double& gate_delay_parasitic,
                                                 double& source_slew,
                                                 bool& dmp_valid) {
    dmp_valid = false;
    gate_delay_parasitic = nanf("");
    source_slew = nanf("");
    const double c1 = dmp_db->C1[pin_idx];
    const double c2 = dmp_db->C2[pin_idx];
    const double rpi = dmp_db->r_pi[pin_idx];
    if (!isfinite(c1) || !isfinite(c2) || !isfinite(rpi)) {
        return false;
    }

    double table_ceff = c1 + c2;
    double table_delay = nanf("");
    double table_slew = nanf("");
    dmpVirtualGateCapDelaySlew(dmp_db, timing_id, input_rf, output_rf, input_slew, table_ceff, table_delay, table_slew);
    if (!isfinite(table_delay) || !isfinite(table_slew)) {
        return false;
    }

    gate_delay_parasitic = table_delay;
    source_slew = table_slew;
    dmp_db->ceff[pin_idx] = table_ceff;
    dmpVirtualGateModelRd(dmp_db, pin_idx, timing_id, input_rf, output_rf, input_slew, table_delay);
    int alg = dmp_db->selectDmpAlg(pin_idx);
    dmp_db->dmp_alg_kind[pin_idx] = alg;

    if (alg == DMP_ALG_ZERO_C2) {
        double c1_delay = nanf("");
        double c1_slew = nanf("");
        dmpVirtualGateCapDelaySlew(dmp_db, timing_id, input_rf, output_rf, input_slew, c1, c1_delay, c1_slew);
        bool ok = isfinite(c1_delay) && isfinite(c1_slew) &&
                  dmp_db->init_zero_c2_factors(pin_idx) &&
                  dmpVirtualFindDriverParamsOnePole(dmp_db, pin_idx, timing_id, input_rf, output_rf, input_slew, c1);
        if (ok) {
            double vo_delay = nanf("");
            double vo_slew = nanf("");
            if (dmpVirtualFindDriverDelaySlew(dmp_db, pin_idx, vo_delay, vo_slew)) {
                gate_delay_parasitic = vo_delay;
                source_slew = vo_slew;
                dmp_db->vo_delay_[pin_idx] = vo_delay;
                dmp_db->vo_slew_[pin_idx] = vo_slew;
                dmp_valid = true;
            }
        }
    } else if (alg == DMP_ALG_PI) {
        bool factors_ok = dmp_db->init_dmp_factors(pin_idx);
        bool params_ok = factors_ok && dmpVirtualFindDriverParams(dmp_db, pin_idx, timing_id, input_rf, output_rf, input_slew, table_ceff);
        if (factors_ok && !params_ok && c2 > 0.0) {
            params_ok = dmpVirtualFindDriverParams(dmp_db, pin_idx, timing_id, input_rf, output_rf, input_slew, c2);
        }
        if (params_ok) {
            double ceff_delay = nanf("");
            double ceff_slew = nanf("");
            dmpVirtualGateCapDelaySlew(dmp_db, timing_id, input_rf, output_rf, input_slew, dmp_db->ceff[pin_idx], ceff_delay, ceff_slew);
            if (isfinite(ceff_delay) && isfinite(ceff_slew)) {
                double vo_delay = nanf("");
                double vo_slew = nanf("");
                if (dmpVirtualFindDriverDelaySlew(dmp_db, pin_idx, vo_delay, vo_slew)) {
                    gate_delay_parasitic = ceff_delay;
                    source_slew = vo_slew;
                    dmp_db->vo_delay_[pin_idx] = vo_delay;
                    dmp_db->vo_slew_[pin_idx] = vo_slew;
                    dmp_valid = true;
                }
            }
        }
    }

    if (!dmp_valid) {
        dmp_db->ceff[pin_idx] = table_ceff;
        dmp_db->rd_[pin_idx] = nanf("");
        dmp_db->t0[pin_idx] = nanf("");
        dmp_db->dt[pin_idx] = nanf("");
        dmp_db->vo_delay_[pin_idx] = nanf("");
        dmp_db->vo_slew_[pin_idx] = table_slew;
        dmp_db->dmp_alg_kind[pin_idx] = DMP_ALG_CAP;
        gate_delay_parasitic = table_delay;
        source_slew = table_slew;
    }
    return isfinite(gate_delay_parasitic) && isfinite(source_slew);
}

__device__ bool dmp_model::updateGateWinner(int to_slot,
                                            int src_slot,
                                            float slew,
                                            bool pick_max,
                                            bool dmp_valid,
                                            double table_ceff) {
    (void)dmp_valid;
    (void)table_ceff;
    if (!isfinite(slew)) {
        return false;
    }
    if ((use_arc_level || use_hybrid_arc_slots) && pin_slew_winner != nullptr) {
        const unsigned int payload = 0x80000000u | static_cast<unsigned int>(src_slot);
        const unsigned long long packed = dmpPackWinner(slew, payload, pick_max);
        const unsigned long long old = atomicMax(&pin_slew_winner[to_slot], packed);
        return packed > old;
    }
    while (atomicCAS(&pin_slew_update_lock[to_slot], 0, 1) != 0) {
    }
    float old_slew = pinSlew[to_slot];
    bool wins = isnan(old_slew) || (pick_max ? (slew > old_slew) : (slew < old_slew));
    if (wins) {
        pinSlew[to_slot] = slew;
        if (dmp_valid) {
            k0_[to_slot] = k0_[src_slot];
            k1_[to_slot] = k1_[src_slot];
            k2_[to_slot] = k2_[src_slot];
            k3_[to_slot] = k3_[src_slot];
            k4_[to_slot] = k4_[src_slot];
            p1_[to_slot] = p1_[src_slot];
            p2_[to_slot] = p2_[src_slot];
            p3_[to_slot] = p3_[src_slot];
            z1_[to_slot] = z1_[src_slot];
            A_[to_slot] = A_[src_slot];
            B_[to_slot] = B_[src_slot];
            D_[to_slot] = D_[src_slot];
            rd_[to_slot] = rd_[src_slot];
            t0[to_slot] = t0[src_slot];
            dt[to_slot] = dt[src_slot];
            ceff[to_slot] = ceff[src_slot];
            vo_delay_[to_slot] = vo_delay_[src_slot];
            vo_slew_[to_slot] = vo_slew_[src_slot];
            driving_cell_extra_delay_[to_slot] = driving_cell_extra_delay_[src_slot];
            dmp_alg_kind[to_slot] = dmp_alg_kind[src_slot];
            if (slot_vth != nullptr) {
                slot_vth[to_slot] = slot_vth[src_slot];
                slot_vl[to_slot] = slot_vl[src_slot];
                slot_vh[to_slot] = slot_vh[src_slot];
                slot_slew_derate[to_slot] = slot_slew_derate[src_slot];
            }
        } else {
            ceff[to_slot] = table_ceff;
            rd_[to_slot] = nanf("");
            t0[to_slot] = nanf("");
            dt[to_slot] = nanf("");
            vo_delay_[to_slot] = nanf("");
            vo_slew_[to_slot] = nanf("");
            driving_cell_extra_delay_[to_slot] = nanf("");
            dmp_alg_kind[to_slot] = DMP_ALG_CAP;
            if (slot_vth != nullptr) {
                slot_vth[to_slot] = slot_vth[src_slot];
                slot_vl[to_slot] = slot_vl[src_slot];
                slot_vh[to_slot] = slot_vh[src_slot];
                slot_slew_derate[to_slot] = slot_slew_derate[src_slot];
            }
        }
    }
    atomicExch(&pin_slew_update_lock[to_slot], 0);
    return wins;
}

__device__ bool dmp_model::updateLoadWinner(int net_arc_id,
                                            int load_attr,
                                            float wire_delay,
                                            float load_slew) {
    if (!isfinite(wire_delay) || !isfinite(load_slew)) {
        return false;
    }
    const int to_pin_id = timing_arc_to_pin_id[net_arc_id];
    const int to_slot = to_pin_id * NUM_ATTR + load_attr;
    const int delay_idx = (load_attr << 1) + (load_attr & 1);
    const bool pick_max = (load_attr >> 1) != 0;

    if ((use_arc_level || use_hybrid_arc_slots || use_fused_fallback) &&
        pin_slew_winner != nullptr &&
        arc_delay_winner != nullptr) {
        const unsigned int slew_payload = static_cast<unsigned int>(net_arc_id);
        const unsigned long long packed_slew = dmpPackWinner(load_slew, slew_payload, pick_max);
        const unsigned long long old_slew = atomicMax(&pin_slew_winner[to_slot], packed_slew);

        const int delay_slot = arcDelayWinnerSlot(net_arc_id, load_attr);
        const unsigned int delay_payload = static_cast<unsigned int>(to_slot);
        const unsigned long long packed_delay = dmpPackWinner(wire_delay, delay_payload, pick_max);
        const unsigned long long old_delay = atomicMax(&arc_delay_winner[delay_slot], packed_delay);
        return packed_slew > old_slew || packed_delay > old_delay;
    }

    while (atomicCAS(&pin_slew_update_lock[to_slot], 0, 1) != 0) {
    }

    bool changed = false;
    float old_slew = pinSlew[to_slot];
    bool slew_wins = isnan(old_slew) || (pick_max ? (load_slew > old_slew) : (load_slew < old_slew));
    if (slew_wins) {
        pinSlew[to_slot] = load_slew;
        changed = true;
    }

    float old_delay = arcDelay[net_arc_id * 2 * NUM_ATTR + delay_idx];
    bool delay_wins = isnan(old_delay) || (pick_max ? (wire_delay > old_delay) : (wire_delay < old_delay));
    if (delay_wins) {
        arcDelay[net_arc_id * 2 * NUM_ATTR + delay_idx] = wire_delay;
        changed = true;
    }

    atomicExch(&pin_slew_update_lock[to_slot], 0);
    return changed;
}

__device__ bool dmp_model::updateAtWinner(int to_slot,
                                          float at,
                                          bool pick_max,
                                          int from_pin_id,
                                          int arc_id,
                                          int from_attr) {
    if (!isfinite(at)) {
        return false;
    }
    if ((use_arc_level || use_hybrid_arc_slots || use_fused_fallback) && pin_at_winner != nullptr) {
        const unsigned int payload = (static_cast<unsigned int>(arc_id) << 2)
                                     | static_cast<unsigned int>(from_attr & 0x3);
        const unsigned long long packed = dmpPackWinner(at, payload, pick_max);
        const unsigned long long old = atomicMax(&pin_at_winner[to_slot], packed);
        return packed > old;
    }
    while (atomicCAS(&pin_at_update_lock[to_slot], 0, 1) != 0) {
    }
    float old_at = pinAt[to_slot];
    bool wins = isnan(old_at) || (pick_max ? (at > old_at) : (at < old_at));
    if (wins) {
        pinAt[to_slot] = at;
        at_prefix_pin[to_slot] = from_pin_id;
        at_prefix_arc[to_slot] = arc_id;
        at_prefix_attr[to_slot] = from_attr;
    }
    atomicExch(&pin_at_update_lock[to_slot], 0);
    return wins;
}
