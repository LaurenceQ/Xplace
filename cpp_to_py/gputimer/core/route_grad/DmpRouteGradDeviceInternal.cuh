#pragma once

#include "gputimer/core/route_grad/DmpRouteGradDevice.cuh"

#include <cmath>

namespace gt {

__device__ bool routeGradRecoverCeffFromGateDelay(const DmpGateArcMeta& gate_arc_meta,
                                                  double target_delay,
                                                  double max_load,
                                                  double& ceff);

__device__ void routeGradEvalCapYDyRd(double t,
                                      double t0,
                                      double dt,
                                      double rd,
                                      double cl,
                                      double& y,
                                      double& dydt0,
                                      double& dyddt,
                                      double& dydcl,
                                      double& dydrd);

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
                                                       double drd);

__device__ __forceinline__ double routeGradFinitePositiveOr(double value, double fallback)
{
    return (isfinite(value) && value > 0.0) ? value : fallback;
}

__device__ __forceinline__ bool routeGradBetterCandidate(double value,
                                                         double best_value,
                                                         bool has_best,
                                                         bool pick_max)
{
    if (!isfinite(value)) {
        return false;
    }
    if (!has_best) {
        return true;
    }
    return pick_max ? (value > best_value) : (value < best_value);
}

__device__ __forceinline__ double routeGradRootParamStep(double value)
{
    return fmax(fabs(value) * 1.0e-3, 1.0e-6);
}

__device__ __forceinline__ void routeGradPrimitiveStatInc(unsigned long long* stats, int index)
{
    if (stats != nullptr) {
        atomicAdd(stats + index, 1ULL);
    }
}

__device__ __forceinline__ int routeGradLowerBoundFloat(const float* arr, int size, float value)
{
    int left = 0;
    int right = size;
    while (left < right) {
        const int mid = (left + right) >> 1;
        if (arr[mid] < value) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__device__ __forceinline__ double routeGradSlopeFromSamples(double base,
                                                           double plus,
                                                           bool plus_ok,
                                                           double minus,
                                                           bool minus_ok,
                                                           double eps)
{
    if (plus_ok && minus_ok && isfinite(plus) && isfinite(minus)) {
        return (plus - minus) / (2.0 * eps);
    }
    if (plus_ok && isfinite(plus) && isfinite(base)) {
        return (plus - base) / eps;
    }
    if (minus_ok && isfinite(minus) && isfinite(base)) {
        return (base - minus) / eps;
    }
    return 0.0;
}

}  // namespace gt
