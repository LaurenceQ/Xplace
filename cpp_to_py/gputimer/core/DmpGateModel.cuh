#pragma once

#include "DmpCudaUtils.cuh"

#ifndef DMP_LOAD_CROSSING_BISECTION
#define DMP_LOAD_CROSSING_BISECTION 0
#endif

#ifndef DMP_LOAD_BISECTION_ITERS
#define DMP_LOAD_BISECTION_ITERS 12
#endif

namespace gt {

struct DmpGateLutMeta {
    unsigned int dims;
    unsigned int flags;
    int x_offset;
    int y_offset;
    int table_offset;

    __device__ __forceinline__ int numX() const {
        return static_cast<int>(dims & 0xffffu);
    }

    __device__ __forceinline__ int numY() const {
        return static_cast<int>((dims >> 16) & 0xffffu);
    }

    __device__ __forceinline__ bool valid() const {
        return (flags & (1u << 31)) != 0u;
    }
};

struct DmpDriverThresholds {
    float vth;
    float vl;
    float vh;
    float derate;
    int library_id;

    __device__ __forceinline__ void set(double driver_vth,
                                        double driver_vl,
                                        double driver_vh,
                                        double driver_derate,
                                        int driver_library_id)
    {
        vth = static_cast<float>(driver_vth);
        vl = static_cast<float>(driver_vl);
        vh = static_cast<float>(driver_vh);
        derate = static_cast<float>(driver_derate);
        library_id = driver_library_id;
    }

    __device__ __forceinline__ bool valid() const
    {
        const float delta = vh - vl;
        return isfinite(vth) &&
               isfinite(vl) &&
               isfinite(vh) &&
               isfinite(derate) &&
               isfinite(delta) &&
               delta > 0.0f &&
               derate > 0.0f;
    }
};

struct DmpRcParams;

struct DmpGateArcMeta {
    GPULutAllocator* allocator;
    DmpGateLutMeta delay_lut;
    DmpGateLutMeta slew_lut;
    float input_slew;
    bool valid;

    __device__ __forceinline__ bool hasValidLuts() const;
    __device__ void capDelaySlew(double load_cap,
                                double& delay,
                                double& slew) const;
    __device__ void capDelaySlew(float load,
                                float& delay,
                                float& slew) const;
    __device__ void gateDelays(const DmpDriverThresholds& thresholds,
                               double ceff,
                               double& t_vth,
                               double& t_vl,
                               double& slew) const;
    __device__ void gateDelays(const DmpDriverThresholds& thresholds,
                               float ceff,
                               float& t_vth,
                               float& t_vl,
                               float& slew) const;
    __device__ inline bool estimateRd(const DmpDriverThresholds& thresholds,
                                      double cap_unit,
                                      const DmpRcParams& rc,
                                      double d1,
                                      double& rd) const;
};

struct DmpWaveCoeffs {
    double k0;
    double k1;
    double k2;
    double k3;
    double k4;
    double p1;
    double p2;
};

struct DmpRcParams {
    double c1;
    double c2;
    double rpi;
    double rd;
    __device__ __forceinline__ int selectAlg(double res_unit) const
    {
        if (!isfinite(rd) || !isfinite(c1) || !isfinite(c2) ||
            !isfinite(rpi) || rd <= 0.0 || c1 <= 0.0 ||
            c2 < 0.0 || rpi <= 0.0) {
            return DMP_ALG_CAP;
        }
        const double min_rd = (isfinite(res_unit) && res_unit > 0.0)
                                  ? (1e-2 / res_unit)
                                  : 1e-2;
        if (rd < min_rd || rpi < rd * 1e-3 || c1 < c2 * 1e-3) {
            return DMP_ALG_CAP;
        }
        if (c2 < c1 * 1e-3) {
            return DMP_ALG_ZERO_C2;
        }
        return DMP_ALG_PI;
    }

    __device__ inline double voUpperTime(int wave_alg,
                                         double t0,
                                         double dt) const
    {
        if (wave_alg == DMP_ALG_ZERO_C2) {
            return t0 + dt + c1 * (rd + rpi) * 2.0;
        }
        if (wave_alg == DMP_ALG_CAP) {
            return 0.0;
        }
        return t0 + dt + (c1 + c2) * (rd + rpi) * 2.0;
    }

    __device__ inline bool initZeroC2(DmpWaveCoeffs& coeffs) const
    {
        const double denom_z1 = rpi * c1;
        const double denom_p1 = c1 * (rd + rpi);
        if (!isfinite(denom_z1) || !isfinite(denom_p1) ||
            denom_z1 == 0.0 || denom_p1 == 0.0) {
            return false;
        }
        const double z1 = 1.0 / denom_z1;
        coeffs.p1 = 1.0 / denom_p1;
        coeffs.k0 = coeffs.p1 / z1;
        if (!isfinite(coeffs.k0) || coeffs.k0 == 0.0 || coeffs.p1 == 0.0) {
            return false;
        }
        coeffs.k2 = 1.0 / coeffs.k0;
        coeffs.k1 = (coeffs.p1 - z1) / (coeffs.p1 * coeffs.p1);
        coeffs.k3 = -coeffs.k1;
        coeffs.p2 = 0.0;
        coeffs.k4 = 0.0;
        return isfinite(z1) && isfinite(coeffs.k0) &&
               isfinite(coeffs.k1) && isfinite(coeffs.k2) &&
               isfinite(coeffs.k3) && isfinite(coeffs.p1);
    }

    __device__ inline bool initPi(DmpWaveCoeffs& coeffs,
                                  double& A,
                                  double& B,
                                  double& D) const
    {
        A = B = D = nanf("");
        const double denom_z1 = rpi * c1;
        const double denom_k0 = rd * c2;
        if (!isfinite(denom_z1) || !isfinite(denom_k0) ||
            denom_z1 == 0.0 || denom_k0 == 0.0) {
            return false;
        }
        const double z1 = 1.0 / denom_z1;
        coeffs.k0 = 1.0 / denom_k0;
        const double a = rpi * rd * c1 * c2;
        const double b = rd * (c1 + c2) + rpi * c1;
        const double disc = b * b - 4 * a;
        if (!isfinite(a) || !isfinite(b) || !isfinite(disc) || a == 0.0 || disc < 0.0) {
            return false;
        }
        const double sqrt_disc = sqrt(disc);
        coeffs.p1 = (b + sqrt_disc) / (2 * a);
        coeffs.p2 = (b - sqrt_disc) / (2 * a);
        const double p1p2 = coeffs.p1 * coeffs.p2;
        if (!isfinite(p1p2) || p1p2 == 0.0 || coeffs.p1 == coeffs.p2) {
            return false;
        }
        coeffs.k2 = z1 / p1p2;
        coeffs.k1 = (1.0 - coeffs.k2 * (coeffs.p1 + coeffs.p2)) / p1p2;
        coeffs.k4 = (coeffs.k1 * coeffs.p1 + coeffs.k2) / (coeffs.p2 - coeffs.p1);
        coeffs.k3 = -coeffs.k1 - coeffs.k4;
        const double z = (c1 + c2) / (rpi * c1 * c2);
        A = z / p1p2;
        B = (z - coeffs.p1) / (coeffs.p1 * (coeffs.p1 - coeffs.p2));
        D = (z - coeffs.p2) / (coeffs.p2 * (coeffs.p2 - coeffs.p1));
        return isfinite(z1) && isfinite(coeffs.k0) &&
               isfinite(coeffs.k1) && isfinite(coeffs.k2) &&
               isfinite(coeffs.k3) && isfinite(coeffs.k4) &&
               isfinite(coeffs.p1) && isfinite(coeffs.p2) &&
               isfinite(A) && isfinite(B) && isfinite(D);
    }
};

struct DmpDriverWave {
    DmpWaveCoeffs coeffs;
    double t0;
    double dt;
    double vo_delay;
    double vo_upper_time;
    float vo_slew;
    int alg;

    __device__ double findLoadCrossing(double elmore,
                                       float vth,
                                       double x1,
                                       double x2,
                                       int max_iter,
                                       double x_tol) const;
    __device__ double findLoadCrossingBisection(double elmore,
                                                float vth,
                                                double x1,
                                                double x2,
                                                int max_iter,
                                                double x_tol) const;
    __device__ __noinline__ double findDriverCrossing(float vth,
                                         double x1,
                                         double x2,
                                         int max_iter,
                                         double x_tol) const;
    __device__ __noinline__ bool findDriverDelaySlew(const DmpDriverThresholds& thresholds,
                                               int max_iter,
                                               double x_tol,
                                               double& delay,
                                               double& slew) const;

    __device__ __forceinline__ bool hasValidDriver() const
    {
        return alg != DMP_ALG_CAP &&
               isfinite(t0) &&
               isfinite(dt) && dt > 0.0 &&
               isfinite(vo_upper_time);
    }


    struct LoadWaveValueEval {
        float inv_dt;
        float p3;
        float d1;
        float d3;
        float d4;
        float d5;
        float one_minus_exp_p1_dt;
        float one_minus_exp_p2_dt;
        float one_minus_exp_p3_dt;
        bool valid;

        __device__ inline void init(const DmpDriverWave& wave,
                                    double elmore)
        {
            valid = false;
            inv_dt = 1.0f / static_cast<float>(wave.dt);
            if (wave.alg == DMP_ALG_CAP || !isfinite(elmore) || elmore <= 0.0) {
                return;
            }

            const float k0 = static_cast<float>(wave.coeffs.k0);
            const float k1 = static_cast<float>(wave.coeffs.k1);
            const float k2 = static_cast<float>(wave.coeffs.k2);
            const float k3 = static_cast<float>(wave.coeffs.k3);
            const float k4 = static_cast<float>(wave.coeffs.k4);
            const float p1 = static_cast<float>(wave.coeffs.p1);
            const float p2 = static_cast<float>(wave.coeffs.p2);
            const float dt = static_cast<float>(wave.dt);
            p3 = 1.0f / static_cast<float>(elmore);
            d1 = k0 * (k1 - k2 / p3);
            d3 = -p3 * k0 * k3 / (p1 - p3);
            d4 = 0.0f;
            d5 = k0 * (k2 / p3 - k1 + p3 * k3 / (p1 - p3));
            if (wave.alg == DMP_ALG_PI) {
                d4 = -p3 * k0 * k4 / (p2 - p3);
                d5 += k0 * p3 * k4 / (p2 - p3);
            }
            one_minus_exp_p1_dt = 1.0f - exp2(-p1 * dt);
            one_minus_exp_p2_dt = wave.alg == DMP_ALG_PI ? 1.0f - exp2(-p2 * dt) : 0.0f;
            one_minus_exp_p3_dt = 1.0f - exp2(-p3 * dt);
            valid = true;
        }

        __device__ inline float value(const DmpDriverWave& wave,
                                      float t) const
        {
            if (!valid) {
                return 0.0f;
            }
            const float t1 = t - static_cast<float>(wave.t0);
            if (t1 <= 0.0f) {
                return 0.0f;
            }
            const float dt = static_cast<float>(wave.dt);
            if (t1 <= dt) {
                const float exp_p1 = exp2(-static_cast<float>(wave.coeffs.p1) * t1);
                const float exp_p2 = wave.alg == DMP_ALG_PI
                                         ? exp2(-static_cast<float>(wave.coeffs.p2) * t1)
                                         : 0.0f;
                const float exp_p3 = exp2(-p3 * t1);
                return (d1 + t1 + d3 * exp_p1 + d4 * exp_p2 + d5 * exp_p3) * inv_dt;
            }

            const float t_prev = t1 - dt;
            const float exp_p1_prev = exp2(-static_cast<float>(wave.coeffs.p1) * t_prev);
            const float exp_p2_prev = wave.alg == DMP_ALG_PI
                                          ? exp2(-static_cast<float>(wave.coeffs.p2) * t_prev)
                                          : 0.0f;
            const float exp_p3_prev = exp2(-p3 * t_prev);
            return 1.0f -
                   (d3 * exp_p1_prev * one_minus_exp_p1_dt +
                    d4 * exp_p2_prev * one_minus_exp_p2_dt +
                    d5 * exp_p3_prev * one_minus_exp_p3_dt) * inv_dt;
        }

        __device__ inline float rootValue(const DmpDriverWave& wave,
                                          float vth,
                                          float t) const
        {
            return value(wave, t) - vth;
        }
    };

    __device__ inline void loadWave0(double elmore,
                                     double t,
                                     double& vl,
                                     double& dvl_dt) const
    {
        if (alg == DMP_ALG_CAP || !isfinite(elmore) || elmore <= 0.0) {
            vl = 0.0;
            dvl_dt = 0.0;
            return;
        }
        const double p3 = 1.0 / elmore;
        double d1 = coeffs.k0 * (coeffs.k1 - coeffs.k2 / p3);
        double d3 = -p3 * coeffs.k0 * coeffs.k3 / (coeffs.p1 - p3);
        double d4 = 0.0;
        double d5 = coeffs.k0 * (coeffs.k2 / p3 - coeffs.k1 +
                                 p3 * coeffs.k3 / (coeffs.p1 - p3));
        if (alg == DMP_ALG_PI) {
            d4 = -p3 * coeffs.k0 * coeffs.k4 / (coeffs.p2 - p3);
            d5 += coeffs.k0 * p3 * coeffs.k4 / (coeffs.p2 - p3);
        }
        const double exp_p1 = exp2(-coeffs.p1 * t);
        const double exp_p2 = alg == DMP_ALG_PI ? exp2(-coeffs.p2 * t) : 0.0;
        const double exp_p3 = exp2(-p3 * t);
        vl = d1 + t + d3 * exp_p1 + d4 * exp_p2 + d5 * exp_p3;
        dvl_dt = 1.0 - d3 * coeffs.p1 * exp_p1 -
                 d4 * coeffs.p2 * exp_p2 - d5 * p3 * exp_p3;
    }

    __device__ inline double loadWave0Value(double elmore,
                                            double t) const
    {
        if (alg == DMP_ALG_CAP || !isfinite(elmore) || elmore <= 0.0) {
            return 0.0;
        }
        const double p3 = 1.0 / elmore;
        const double d1 = coeffs.k0 * (coeffs.k1 - coeffs.k2 / p3);
        const double d3 = -p3 * coeffs.k0 * coeffs.k3 / (coeffs.p1 - p3);
        double d4 = 0.0;
        double d5 = coeffs.k0 * (coeffs.k2 / p3 - coeffs.k1 +
                                 p3 * coeffs.k3 / (coeffs.p1 - p3));
        if (alg == DMP_ALG_PI) {
            d4 = -p3 * coeffs.k0 * coeffs.k4 / (coeffs.p2 - p3);
            d5 += coeffs.k0 * p3 * coeffs.k4 / (coeffs.p2 - p3);
        }
        const double exp_p1 = exp2(-coeffs.p1 * t);
        const double exp_p2 = alg == DMP_ALG_PI ? exp2(-coeffs.p2 * t) : 0.0;
        const double exp_p3 = exp2(-p3 * t);
        return d1 + t + d3 * exp_p1 + d4 * exp_p2 + d5 * exp_p3;
    }

    __device__ inline double loadWaveValue(double elmore,
                                           double t) const
    {
        const double t1 = t - t0;
        if (t1 <= 0.0) {
            return 0.0;
        }
        const double inv_dt = 1.0 / dt;
        if (t1 <= dt) {
            return loadWave0Value(elmore, t1) * inv_dt;
        }
        return (loadWave0Value(elmore, t1) -
                loadWave0Value(elmore, t1 - dt)) * inv_dt;
    }

    __device__ inline double loadRootValue(double elmore,
                                           float vth,
                                           double t) const
    {
        return loadWaveValue(elmore, t) - static_cast<double>(vth);
    }

    __device__ inline void loadWave(double elmore,
                                    double t,
                                    double& vl,
                                    double& dvl_dt) const
    {
        const double t1 = t - t0;
        if (t1 <= 0.0) {
            vl = 0.0;
            dvl_dt = 0.0;
        } else if (t1 <= dt) {
            double vl0, dvl0_dt;
            loadWave0(elmore, t1, vl0, dvl0_dt);
            vl = vl0 / dt;
            dvl_dt = dvl0_dt / dt;
        } else {
            double vl0, dvl0_dt;
            double vl0_dt, dvl0_dt_dt;
            loadWave0(elmore, t1, vl0, dvl0_dt);
            loadWave0(elmore, t1 - dt, vl0_dt, dvl0_dt_dt);
            vl = (vl0 - vl0_dt) / dt;
            dvl_dt = (dvl0_dt - dvl0_dt_dt) / dt;
        }
    }

    __device__ inline void loadRootFunc(double elmore,
                                        float vth,
                                        double t,
                                        double& y,
                                        double& dy) const
    {
        double vl, vl_dt;
        loadWave(elmore, t, vl, vl_dt);
        y = vl - static_cast<double>(vth);
        dy = vl_dt;
    }

    __device__ inline void driverWave0(double t,
                                       double& vo,
                                       double& dvo_dt) const
    {
        if (alg == DMP_ALG_CAP) {
            vo = 0.0;
            dvo_dt = 0.0;
            return;
        }
        const double exp_p1 = exp2(-coeffs.p1 * t);
        if (alg == DMP_ALG_ZERO_C2) {
            vo = coeffs.k0 * (coeffs.k1 + coeffs.k2 * t + coeffs.k3 * exp_p1);
            dvo_dt = coeffs.k0 * (coeffs.k2 - coeffs.k3 * coeffs.p1 * exp_p1);
            return;
        }
        const double exp_p2 = exp2(-coeffs.p2 * t);
        vo = coeffs.k0 * (coeffs.k1 + coeffs.k2 * t +
                          coeffs.k3 * exp_p1 + coeffs.k4 * exp_p2);
        dvo_dt = coeffs.k0 * (coeffs.k2 - coeffs.k3 * coeffs.p1 * exp_p1 -
                              coeffs.k4 * coeffs.p2 * exp_p2);
    }

    __device__ inline void driverWave(double t,
                                      double& vo,
                                      double& dvo_dt) const
    {
        const double t1 = t - t0;
        if (t1 <= 0.0) {
            vo = 0.0;
            dvo_dt = 0.0;
        } else if (t1 <= dt) {
            double v0, dv0_dt;
            driverWave0(t1, v0, dv0_dt);
            vo = v0 / dt;
            dvo_dt = dv0_dt / dt;
        } else {
            double v0, dv0_dt;
            double v0_dt, dv0_dt_dt;
            driverWave0(t1, v0, dv0_dt);
            driverWave0(t1 - dt, v0_dt, dv0_dt_dt);
            vo = (v0 - v0_dt) / dt;
            dvo_dt = (dv0_dt - dv0_dt_dt) / dt;
        }
    }

    __device__ inline void driverRootFunc(float vth,
                                          double t,
                                          double& y,
                                          double& dy) const
    {
        double vo, vo_dt;
        driverWave(t, vo, vo_dt);
        y = vo - static_cast<double>(vth);
        dy = vo_dt;
    }
};

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

enum DmpGateNetPairDebugCounter {
    DMP_GNP_TOTAL_CANDIDATES = 0,
    DMP_GNP_INVALID_TRANSITION_SKIPS = 1,
    DMP_GNP_INVALID_SCRATCH_SKIPS = 2,
    DMP_GNP_FINITE_CANDIDATES = 3
};

__device__ void dmpRootProfileVoCall();
__device__ void dmpRootProfileVoSuccess();
__device__ void dmpRootProfileVoIters(unsigned long long value);
__device__ void dmpRootProfileVoBracketFail();
__device__ void dmpRootProfileVoEndpointHit();
__device__ void dmpRootProfileVoMaxIterFail();
__device__ void dmpGateNetPairCount(unsigned long long* counts, int counter);
__device__ void dmpDrivingCellCount(unsigned long long* counts, int counter);
__device__ void dmpDebugPrintDirectClock(const DmpModel* dmp_db,
                                         int from_pin_id,
                                         int to_pin_id,
                                         int from_slot,
                                         int attr,
                                         float source_slew,
                                         double final_slew,
                                         double elmore,
                                         double extra_delay,
                                         double vo_delay,
                                         double final_delay,
                                         int alg,
                                         bool used_dmp_load,
                                         bool used_driving_cell);
__device__ void dmpDebugPrintDrivingCell(const DmpModel* dmp_db,
                                         int pin_id,
                                         int attr,
                                         float input_slew,
                                         const DmpDriverWave& driver_wave,
                                         float gate_delay,
                                         float intrinsic_delay,
                                         double extra_delay);
__device__ unsigned long long dmpPackWinner(float value, unsigned int payload, bool pick_max);
__device__ __forceinline__ bool DmpGateArcMeta::hasValidLuts() const
{
    return delay_lut.valid() && slew_lut.valid();
}

__device__ inline bool DmpGateArcMeta::estimateRd(const DmpDriverThresholds& thresholds,
                                                               double cap_unit,
                                                               const DmpRcParams& rc,
                                                               double d1,
                                                               double& rd) const
{
    rd = nanf("");
    const float cap1_f = static_cast<float>(rc.c1 + rc.c2);
    const float cap_delta_f = static_cast<float>(1e-15 / cap_unit);
    const float cap2_f = cap1_f + cap_delta_f;
    const float d1_f = static_cast<float>(d1);
    float d2_f = nanf("");
    float s2_f = nanf("");
    if (!isfinite(cap1_f) || !isfinite(cap2_f) || !isfinite(d1_f) ||
        !isfinite(cap_delta_f) || cap_delta_f <= 0.0f || cap2_f == cap1_f) {
        return false;
    }
    capDelaySlew(cap2_f, d2_f, s2_f);
    if (!isfinite(d2_f)) {
        return false;
    }
    if (!thresholds.valid()) {
        return false;
    }
    rd = static_cast<double>(
        -log(static_cast<double>(thresholds.vth)) * fabsf(d1_f - d2_f) / (cap2_f - cap1_f));
    return isfinite(rd) && rd > 0.0;
}

} // namespace gt
