#pragma once

#include "DmpCeff.h"

namespace gt {

__device__ __forceinline__ double
dmp_model::voCrossingUpperBound(int pin_idx){
    if (dmp_alg_kind[pin_idx] == DMP_ALG_ZERO_C2) {
        return t0[pin_idx] + dt[pin_idx] + C1[pin_idx] * (rd_[pin_idx] + r_pi[pin_idx]) * 2.0;
    }
    if (dmp_alg_kind[pin_idx] == DMP_ALG_CAP) {
        return 0.0;
    }
    return t0[pin_idx] + dt[pin_idx] + (C1[pin_idx] + C2[pin_idx]) * (rd_[pin_idx] + r_pi[pin_idx]) * 2.0;
}

__device__ __forceinline__ double exp2(double x){
    if (x < -12.0)
        // exp(-12) = 6.1e-6
        return 0.0;
    else {
        double y = 1.0 + x / 4096.0;
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
}
__device__ __forceinline__ double
dmp_model::y0(double t, double rd, 
           double cl)
{
  return t - rd * cl * (1.0 - exp2(-t / (rd * cl)));
}

__device__ __forceinline__ double
dmp_model::y(double t,
          double t0,
          double dt,
          double rd, 
          double cl)
{
  double t1 = t - t0;
  if (t1 <= 0.0)
    return 0.0;
  else if (t1 <= dt)
    return y0(t1, rd, cl) / dt;
  else
    return (y0(t1, rd, cl) - y0(t1 - dt, rd, cl)) / dt;
}

__device__ __forceinline__ double
dmp_model::y0dt(double t,
            double rd,
            double cl)
{
  return 1.0 - exp2(-t / (rd * cl));
}

__device__ __forceinline__ double
dmp_model::y0dcl(double t,
                double rd,
                double cl)
{
  return rd * ((1.0 + t / (rd * cl)) * exp2(-t / (rd * cl)) - 1);
}
__device__ __forceinline__ void
dmp_model::dy(double t,
           double t0,
           double dt,
           double rd,
           double cl,
           // Return values.
           double &dydt0,
           double &dyddt,
           double &dydcl)
{
  double t1 = t - t0;
  if (t1 <= 0.0)
    dydt0 = dyddt = dydcl = 0.0;
  else if (t1 <= dt) {
    dydt0 = -y0dt(t1, rd, cl) / dt;
    dyddt = -y0(t1, rd, cl) / (dt * dt);
    dydcl = y0dcl(t1, rd, cl) / dt;
  }
  else {
    dydt0 = -(y0dt(t1, rd, cl) - y0dt(t1 - dt, rd, cl)) / dt;
    dyddt = -(y0(t1, rd, cl) + y0(t1 - dt, rd, cl)) / (dt * dt) + y0dt(t1 - dt, rd, cl) / dt;
    dydcl = (y0dcl(t1, rd, cl) - y0dcl(t1 - dt, rd, cl)) / dt;
  }
}





__device__ __forceinline__ void dmp_model::Vl0(int pin_idx, double t, double &vl, double &dvl_dt){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx % NUM_ATTR;
    int arc_id = arc_ids[idx];
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    if (dmp_alg_kind[pin_idx] == DMP_ALG_CAP) {
        vl = 0.0;
        dvl_dt = 0.0;
        return;
    }
    double p3_ = 1.0 / elmore_delay[to_pin_id * NUM_ATTR + i];
    double D1 = k0_[pin_idx] * (k1_[pin_idx] - k2_[pin_idx] / p3_);
    double D3 = -p3_ * k0_[pin_idx] * k3_[pin_idx] / (p1_[pin_idx] - p3_);
    double D4 = 0.0;
    double D5 = k0_[pin_idx] * (k2_[pin_idx] / p3_ - k1_[pin_idx] + p3_ * k3_[pin_idx] / (p1_[pin_idx] - p3_));
    if (dmp_alg_kind[pin_idx] == DMP_ALG_PI) {
        D4 = -p3_ * k0_[pin_idx] * k4_[pin_idx] / (p2_[pin_idx] - p3_);
        D5 += k0_[pin_idx] * p3_ * k4_[pin_idx] / (p2_[pin_idx] - p3_);
    }
    double exp_p1 = exp2(-p1_[pin_idx] * t);
    double exp_p2 = dmp_alg_kind[pin_idx] == DMP_ALG_PI ? exp2(-p2_[pin_idx] * t) : 0.0;
    double exp_p3 = exp2(-p3_ * t);
    vl = D1 + t + D3 * exp_p1 + D4 * exp_p2 + D5 * exp_p3;
    dvl_dt = 1.0 - D3 * p1_[pin_idx] * exp_p1 - D4 * p2_[pin_idx] * exp_p2 - D5 * p3_ * exp_p3;
}

__device__ __forceinline__ void dmp_model::Vl0Explicit(int pin_idx,
                                       double elmore,
                                       double t,
                                       double &vl,
                                       double &dvl_dt){
    if (dmp_alg_kind[pin_idx] == DMP_ALG_CAP || !isfinite(elmore) || elmore <= 0.0) {
        vl = 0.0;
        dvl_dt = 0.0;
        return;
    }
    double p3_local = 1.0 / elmore;
    double D1 = k0_[pin_idx] * (k1_[pin_idx] - k2_[pin_idx] / p3_local);
    double D3 = -p3_local * k0_[pin_idx] * k3_[pin_idx] / (p1_[pin_idx] - p3_local);
    double D4 = 0.0;
    double D5 = k0_[pin_idx] * (k2_[pin_idx] / p3_local - k1_[pin_idx] + p3_local * k3_[pin_idx] / (p1_[pin_idx] - p3_local));
    if (dmp_alg_kind[pin_idx] == DMP_ALG_PI) {
        D4 = -p3_local * k0_[pin_idx] * k4_[pin_idx] / (p2_[pin_idx] - p3_local);
        D5 += k0_[pin_idx] * p3_local * k4_[pin_idx] / (p2_[pin_idx] - p3_local);
    }
    double exp_p1 = exp2(-p1_[pin_idx] * t);
    double exp_p2 = dmp_alg_kind[pin_idx] == DMP_ALG_PI ? exp2(-p2_[pin_idx] * t) : 0.0;
    double exp_p3 = exp2(-p3_local * t);
    vl = D1 + t + D3 * exp_p1 + D4 * exp_p2 + D5 * exp_p3;
    dvl_dt = 1.0 - D3 * p1_[pin_idx] * exp_p1 - D4 * p2_[pin_idx] * exp_p2 - D5 * p3_local * exp_p3;
}

__device__ __forceinline__ void
dmp_model::V0(
          int pin_idx,
          double t,
          // Return values.
          double &vo,
          double &dvo_dt)
{
    if (dmp_alg_kind[pin_idx] == DMP_ALG_CAP) {
        vo = 0.0;
        dvo_dt = 0.0;
        return;
    }
    double exp_p1 = exp2(-p1_[pin_idx] * t);
    if (dmp_alg_kind[pin_idx] == DMP_ALG_ZERO_C2) {
        vo = k0_[pin_idx] * (k1_[pin_idx] + k2_[pin_idx] * t + k3_[pin_idx] * exp_p1);
        dvo_dt = k0_[pin_idx] * (k2_[pin_idx] - k3_[pin_idx] * p1_[pin_idx] * exp_p1);
        return;
    }
    double exp_p2 = exp2(-p2_[pin_idx] * t);
    vo = k0_[pin_idx] * (k1_[pin_idx] + k2_[pin_idx] * t + k3_[pin_idx] * exp_p1 + k4_[pin_idx] * exp_p2);
    dvo_dt = k0_[pin_idx] * (k2_[pin_idx] - k3_[pin_idx] * p1_[pin_idx] * exp_p1 - k4_[pin_idx] * p2_[pin_idx] * exp_p2);
}

__device__ __forceinline__ void dmp_model::Vl(double t, double &vl, double &dvl_dt){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pin_idx = pin_ids[idx];
    double t1 = t - t0[pin_idx];
    if(t1 <= 0.0){
        vl = 0.0;
        dvl_dt = 0.0;
    }
    else if(t1 <= dt[pin_idx]){
        double vl0, dvl0_dt;
        Vl0(pin_idx, t1, vl0, dvl0_dt);
        vl = vl0 / dt[pin_idx];
        dvl_dt = dvl0_dt / dt[pin_idx];
    }
    else{
        double vl0, dvl0_dt;
        Vl0(pin_idx, t1, vl0, dvl0_dt);
        double vl0_dt, dvl0_dt_dt;
        Vl0(pin_idx, t1 - dt[pin_idx], vl0_dt, dvl0_dt_dt);
        vl = (vl0 - vl0_dt) / dt[pin_idx];
        dvl_dt = (dvl0_dt - dvl0_dt_dt) / dt[pin_idx];
    }
}

__device__ __forceinline__ void dmp_model::VlExplicit(int pin_idx,
                                      double elmore,
                                      double t,
                                      double &vl,
                                      double &dvl_dt){
    double t1 = t - t0[pin_idx];
    if(t1 <= 0.0){
        vl = 0.0;
        dvl_dt = 0.0;
    }
    else if(t1 <= dt[pin_idx]){
        double vl0, dvl0_dt;
        Vl0Explicit(pin_idx, elmore, t1, vl0, dvl0_dt);
        vl = vl0 / dt[pin_idx];
        dvl_dt = dvl0_dt / dt[pin_idx];
    }
    else{
        double vl0, dvl0_dt;
        Vl0Explicit(pin_idx, elmore, t1, vl0, dvl0_dt);
        double vl0_dt, dvl0_dt_dt;
        Vl0Explicit(pin_idx, elmore, t1 - dt[pin_idx], vl0_dt, dvl0_dt_dt);
        vl = (vl0 - vl0_dt) / dt[pin_idx];
        dvl_dt = (dvl0_dt - dvl0_dt_dt) / dt[pin_idx];
    }
}

__device__ __forceinline__ void
dmp_model::Vo(double t,
           // Return values.
           double &vo,
           double &dvo_dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pin_idx = pin_ids[idx];
    double t1 = t - t0[pin_idx];
    if (t1 <= 0.0) {
        vo = 0.0;
        dvo_dt = 0.0;
    }
    else if (t1 <= dt[pin_idx]) {
        double v0, dv0_dt;
        V0(pin_idx, t1, v0, dv0_dt);

        vo = v0 / dt[pin_idx];
        dvo_dt = dv0_dt / dt[pin_idx];
    }
    else {
        double v0, dv0_dt;
        V0(pin_idx, t1, v0, dv0_dt);

        double v0_dt, dv0_dt_dt;
        V0(pin_idx, t1 - dt[pin_idx], v0_dt, dv0_dt_dt);

        vo = (v0 - v0_dt) / dt[pin_idx];
        dvo_dt = (dv0_dt - dv0_dt_dt) / dt[pin_idx];
    }
}
__device__ __forceinline__ void dmp_model::vl_func(double vth, double t, double &y, double &dy){
    double vl, vl_dt;
    Vl(t, vl, vl_dt);
    y = vl - vth; // goal: y = 0, y = vl(t) - vth
    dy = vl_dt;
}
__device__ __forceinline__ void dmp_model::vlFuncExplicit(int pin_idx,
                                          double elmore,
                                          double vth,
                                          double t,
                                          double &y,
                                          double &dy){
    double vl, vl_dt;
    VlExplicit(pin_idx, elmore, t, vl, vl_dt);
    y = vl - vth;
    dy = vl_dt;
}
__device__ __forceinline__ void dmp_model::vo_func(double vth, double t, double &y, double &dy){
    double vo, vo_dt;
    Vo(t, vo, vo_dt);
    y = vo - vth;
    dy = vo_dt;
}
__device__ inline double dmp_model::findRoot_vo(double vth, double x1, double x2){
    double y1, y2, dy;
    vo_func(vth, x1, y1, dy);
    vo_func(vth, x2, y2, dy);
    if(y1 * y2 > 0.0) return nanf(""); // cannot find root
    if(y1 == 0.0) return x1;
    if(y2 == 0.0) return x2;
    if(y1 > 0.0){
        double xtemp = x1; x1 = x2; x2 = xtemp;
    }
    double root = (x1 + x2) / 2.0;
    double dx_prev = abs(x2 - x1);
    double dx = dx_prev;
    double y;
    vo_func(vth, root, y, dy);
    for(int iter = 0; iter < MAX_ITER; iter++){
        // Newton/raphson out of range.
        if ((((x2 - root) * dy + y) * ((x1 - root) * dy + y) > 0.0)
        // Not decreasing fast enough.
        || (abs(2.0 * y) > abs(dx_prev * dy))) { // step too large
        // Bisect x1/x2 interval.
            dx_prev = dx;
            dx = (x2 - x1) * 0.5;
            root = x1 + dx;
        }
        else {
            dx_prev = dx;
            dx = y / dy;
            root -= dx;
        }
        if (abs(dx) <= x_tol * abs(root)) {
            // Converged.
            return root;
        }

        vo_func(vth, root, y, dy);
        if (y < 0.0)
            x1 = root;
        else
            x2 = root;
    }
    return nanf("");
}
__device__ inline double dmp_model::findRoot_vl(double vth, double x1, double x2){ // TODO: solve non-deterministic
    double y1, y2, dy;
    vl_func(vth, x1, y1, dy);
    vl_func(vth, x2, y2, dy);
    if(y1 * y2 > 0.0) return nanf(""); // cannot find root
    if(y1 == 0.0) return x1;
    if(y2 == 0.0) return x2;
    if(y1 > 0.0){
        double xtemp = x1; x1 = x2; x2 = xtemp;
    }
    double root = (x1 + x2) / 2.0;
    double dx_prev = abs(x2 - x1);
    double dx = dx_prev;
    double y;
    vl_func(vth, root, y, dy);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int arc_id = arc_ids[idx];
    int to_pin_id = timing_arc_to_pin_id[arc_id];

    // if(to_pin_id == 4706){
    //     int pin_idx = pin_ids[idx];
        // printf("Rd = %.4f t0 = %.4f dt = %.4f\n        "
        //     "k0 = %.4f k1 = %.4f k3 = %.4f k4 = %.4f p1 = %.4f p2 = %.4f p3 = %.4f\n        ",
        //     rd_[pin_idx], t0[pin_idx], dt[pin_idx], 
        //     k0_[pin_idx], k1_[pin_idx], k3_[pin_idx], k4_[pin_idx], p1_[pin_idx], p2_[pin_idx], p3_[pin_idx]         
        // );
    // }
    for(int iter = 0; iter < MAX_ITER; iter++){
        // if(to_pin_id == 4706)printf("iter:%d root:%.4f y:%.4f dy:%.4f x1:%.4f x2:%.4f dx:%.4f dx_prev:%.4f\n", iter, root, y, dy, x1, x2, dx, dx_prev);

        // Newton/raphson out of range.
        if ((((x2 - root) * dy + y) * ((x1 - root) * dy + y) > 0.0)
        // Not decreasing fast enough.
        || (abs(2.0 * y) > abs(dx_prev * dy))) { // step too large
        // Bisect x1/x2 interval.
            dx_prev = dx;
            dx = (x2 - x1) * 0.5;
            root = x1 + dx;
        }
        else {
            dx_prev = dx;
            dx = y / dy;
            root -= dx;
        }
        if (abs(dx) <= x_tol * abs(root)) {
            // Converged.
            return root;
        }

        vl_func(vth, root, y, dy);
        if (y < 0.0)
            x1 = root;
        else
            x2 = root;
    }
    return nanf("");
}
__device__ inline double dmp_model::findRootVlExplicit(int pin_idx,
                                                double elmore,
                                                double vth,
                                                double x1,
                                                double x2){
    double y1, y2, dy;
    vlFuncExplicit(pin_idx, elmore, vth, x1, y1, dy);
    vlFuncExplicit(pin_idx, elmore, vth, x2, y2, dy);
    if(y1 * y2 > 0.0) return nanf("");
    if(y1 == 0.0) return x1;
    if(y2 == 0.0) return x2;
    if(y1 > 0.0){
        double xtemp = x1; x1 = x2; x2 = xtemp;
    }
    double root = (x1 + x2) / 2.0;
    double dx_prev = abs(x2 - x1);
    double dx = dx_prev;
    double y;
    vlFuncExplicit(pin_idx, elmore, vth, root, y, dy);
    for(int iter = 0; iter < MAX_ITER; iter++){
        if ((((x2 - root) * dy + y) * ((x1 - root) * dy + y) > 0.0)
        || (abs(2.0 * y) > abs(dx_prev * dy))) {
            dx_prev = dx;
            dx = (x2 - x1) * 0.5;
            root = x1 + dx;
        }
        else {
            dx_prev = dx;
            dx = y / dy;
            root -= dx;
        }
        if (abs(dx) <= x_tol * abs(root)) {
            return root;
        }

        vlFuncExplicit(pin_idx, elmore, vth, root, y, dy);
        if (y < 0.0)
            x1 = root;
        else
            x2 = root;
    }
    return nanf("");
}
// __device__ __forceinline__ void DmpAlg::showVl()
// {
//   report_->reportLine("  t    vl(t)");
//   double ub = vlCrossingUpperBound();
//   for (double t = t0_; t < t0_ + ub * 2.0; t += ub / 10.0) {
//     double vl, dvl_dt;
//     Vl(t, vl, dvl_dt);
//     report_->reportLine(" %g %g", t, vl);
//   }
// }

__device__ inline double dmp_model::findVlCrossing(double vth, double t_lower, double t_upper){
    double t_vth = findRoot_vl(vth, t_lower, t_upper);
    if(isnan(t_vth)){
        int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
        int arc_id = arc_ids[blockIdx.x * blockDim.x + threadIdx.x];
        int from_pin_id = timing_arc_from_pin_id[arc_id];
        int to_pin_id = timing_arc_to_pin_id[arc_id];
        double y1, y2, dy; 
        vl_func(vth, t_lower, y1, dy);
        vl_func(vth, t_upper, y2, dy);
        if(debug_on)printf("Error: cannot find Vl crossing point from:%s to:%s vth:%.3f t_vth:%e x1:%e x2:%e y1:%.4f y2:%.4f rd:%.4f, r_pi:%.4f cap1:%.4f cap2:%.4f t0:%.4f, dt:%.4f ceff:%.4f\n", pin_names[from_pin_id], pin_names[to_pin_id], vth, t_vth, t_lower, t_upper, y1, y2, rd_[pin_idx], r_pi[pin_idx], C1[pin_idx], C2[pin_idx], t0[pin_idx], dt[pin_idx], ceff[pin_idx]);
        return nanf("");
    }
    else return t_vth;
}
__device__ inline double dmp_model::findVlCrossingExplicit(int pin_idx,
                                                    double elmore,
                                                    double vth,
                                                    double t_lower,
                                                    double t_upper){
    double t_vth = findRootVlExplicit(pin_idx, elmore, vth, t_lower, t_upper);
    if(isnan(t_vth)){
        if(debug_on) {
            double y1, y2, dy;
            vlFuncExplicit(pin_idx, elmore, vth, t_lower, y1, dy);
            vlFuncExplicit(pin_idx, elmore, vth, t_upper, y2, dy);
            printf("Error: cannot find explicit Vl crossing vth:%.3f x1:%e x2:%e y1:%.4f y2:%.4f slot:%d rd:%.4f r_pi:%.4f cap1:%.4f cap2:%.4f t0:%.4f dt:%.4f ceff:%.4f elmore:%.4f\n",
                   vth, t_lower, t_upper, y1, y2, pin_idx, rd_[pin_idx],
                   r_pi[pin_idx], C1[pin_idx], C2[pin_idx], t0[pin_idx],
                   dt[pin_idx], ceff[pin_idx], elmore);
        }
        return nanf("");
    }
    return t_vth;
}
__device__ inline double 
dmp_model::findVoCrossing(double vth,
                       double t_lower,
                       double t_upper){
    double t_vth = findRoot_vo(vth, t_lower, t_upper);
    if(isnan(t_vth)){
        int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
        int arc_id = arc_ids[blockIdx.x * blockDim.x + threadIdx.x];
        int from_pin_id = timing_arc_from_pin_id[arc_id];
        int to_pin_id = timing_arc_to_pin_id[arc_id];
        if(debug_on)printf("Error: cannot find Vo crossing point from:%s to:%s rd:%.4f, r_pi:%.4f cap1:%.4f cap2:%.4f t0:%.4f, dt:%.4f ceff:%.4f\n", pin_names[from_pin_id], pin_names[to_pin_id], rd_[pin_idx], r_pi[pin_idx], C1[pin_idx], C2[pin_idx], t0[pin_idx], dt[pin_idx], ceff[pin_idx]);
        return nanf("");
    }
    return t_vth;
}


} // namespace gt
