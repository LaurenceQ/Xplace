#pragma once

#include "DmpModel.h"

namespace gt {

__device__ __forceinline__ double exp2(double x) {
    if (x < -12.0) {
        return 0.0;
    }
    double y = 1.0 + x / 4096.0;
    y *= y; y *= y; y *= y; y *= y;
    y *= y; y *= y; y *= y; y *= y;
    y *= y; y *= y; y *= y; y *= y;
    return y;
}

__device__ __forceinline__ double DmpModel::y0(double t, double rd, double cl) {
    return t - rd * cl * (1.0 - exp2(-t / (rd * cl)));
}

__device__ __forceinline__ double DmpModel::y(double t,
                                              double t0,
                                              double dt,
                                              double rd,
                                              double cl) {
    double t1 = t - t0;
    if (t1 <= 0.0) {
        return 0.0;
    }
    if (t1 <= dt) {
        return y0(t1, rd, cl) / dt;
    }
    return (y0(t1, rd, cl) - y0(t1 - dt, rd, cl)) / dt;
}

__device__ __forceinline__ double DmpModel::y0dt(double t, double rd, double cl) {
    return 1.0 - exp2(-t / (rd * cl));
}

__device__ __forceinline__ double DmpModel::y0dcl(double t, double rd, double cl) {
    return rd * ((1.0 + t / (rd * cl)) * exp2(-t / (rd * cl)) - 1);
}

__device__ __forceinline__ void DmpModel::dy(double t,
                                             double t0,
                                             double dt,
                                             double rd,
                                             double cl,
                                             double &dydt0,
                                             double &dyddt,
                                             double &dydcl) {
    double t1 = t - t0;
    if (t1 <= 0.0) {
        dydt0 = dyddt = dydcl = 0.0;
    } else if (t1 <= dt) {
        dydt0 = -y0dt(t1, rd, cl) / dt;
        dyddt = -y0(t1, rd, cl) / (dt * dt);
        dydcl = y0dcl(t1, rd, cl) / dt;
    } else {
        dydt0 = -(y0dt(t1, rd, cl) - y0dt(t1 - dt, rd, cl)) / dt;
        dyddt = -(y0(t1, rd, cl) + y0(t1 - dt, rd, cl)) / (dt * dt) +
                 y0dt(t1 - dt, rd, cl) / dt;
        dydcl = (y0dcl(t1, rd, cl) - y0dcl(t1 - dt, rd, cl)) / dt;
    }
}

} // namespace gt
