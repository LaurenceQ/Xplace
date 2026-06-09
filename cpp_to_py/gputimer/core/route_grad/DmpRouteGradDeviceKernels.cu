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

// Net-driver primitive chain rules and slope writer kernels.
// Net-driver primitive chain rules from driver-root RC to downstream net delay/slew.
__device__ int RouteGradNetPrimitiveReverse::classifyNetDriverPrimitiveAlg(
    const RouteGradNetDriverSlopeKey& key) const
{
    if (model == nullptr || key.root_slot < 0 || key.root_slot >= model->dmp_pin_slot_count ||
        model->C1 == nullptr || model->C2 == nullptr || model->r_pi == nullptr ||
        model->pinSlew == nullptr) {
        return -1;
    }
    const double c1 = static_cast<double>(model->C1[key.root_slot]);
    const double c2 = static_cast<double>(model->C2[key.root_slot]);
    const double rpi = static_cast<double>(model->r_pi[key.root_slot]);
    if (!isfinite(c1) || !isfinite(c2) || !isfinite(rpi) ||
        c1 < 0.0 || c2 < 0.0 || rpi < 0.0) {
        return -1;
    }
    double input_slew = nanf("");
    if (key.input_slew_slot >= 0 && key.input_slew_slot < model->dmp_pin_slot_count) {
        input_slew = static_cast<double>(model->pinSlew[key.input_slew_slot]);
    } else if (key.kind == kRouteGradNetKeyGateNetPair && key.gate_arc_id >= 0 &&
               model->d_allocator != nullptr) {
        const int gate_from_pin = model->timing_arc_from_pin_id[key.gate_arc_id];
        const int from_attr = ((key.attr >> 1) << 1) | key.input_rf;
        const int timing_id = model->timing_arc_id_map[key.gate_arc_id * 2 + (key.attr >> 1)];
        if (gate_from_pin >= 0 && gate_from_pin < model->num_pins && timing_id >= 0) {
            input_slew = static_cast<double>(model->idealClockSlew(gate_from_pin, from_attr));
        }
    }
    if (!isfinite(input_slew)) {
        return -1;
    }
    RouteGradNetDriverWaveEval eval{};
    if (!netDriverWaveForKey(key, c1, c2, rpi, input_slew, eval)) {
        return -1;
    }
    return eval.wave.alg;
}

// Analytic net-driver primitive slopes: driver-root PI parameters and input
// slew to downstream net delay/slew through driver waveform, load waveform, and
// sink Elmore.
__device__ bool RouteGradNetPrimitiveReverse::netDriverPrimitiveWaveChainSlopes(
    const RouteGradNetDriverSlopeKey& key,
    RouteGradGatePrimitiveSlopes& slopes) const
{
    slopes = {};
    if (model == nullptr || key.root_slot < 0 || key.root_slot >= model->dmp_pin_slot_count ||
        model->C1 == nullptr || model->C2 == nullptr || model->r_pi == nullptr ||
        model->pinSlew == nullptr) {
        return false;
    }
    const double c1 = static_cast<double>(model->C1[key.root_slot]);
    const double c2 = static_cast<double>(model->C2[key.root_slot]);
    const double rpi = static_cast<double>(model->r_pi[key.root_slot]);
    if (!isfinite(c1) || !isfinite(c2) || !isfinite(rpi) ||
        c1 < 0.0 || c2 < 0.0 || rpi < 0.0) {
        return false;
    }

    double input_slew = nanf("");
    if (key.input_slew_slot >= 0 && key.input_slew_slot < model->dmp_pin_slot_count) {
        input_slew = static_cast<double>(model->pinSlew[key.input_slew_slot]);
    } else if (key.kind == kRouteGradNetKeyGateNetPair && key.gate_arc_id >= 0 &&
               model->d_allocator != nullptr) {
        const int gate_from_pin = model->timing_arc_from_pin_id[key.gate_arc_id];
        const int from_attr = ((key.attr >> 1) << 1) | key.input_rf;
        const int timing_id = model->timing_arc_id_map[key.gate_arc_id * 2 + (key.attr >> 1)];
        if (gate_from_pin >= 0 && gate_from_pin < model->num_pins && timing_id >= 0) {
            input_slew = static_cast<double>(model->idealClockSlew(gate_from_pin, from_attr));
        }
    }
    if (!isfinite(input_slew)) {
        return false;
    }

    RouteGradNetDriverWaveEval base_eval{};
    if (!netDriverWaveForKey(key, c1, c2, rpi, input_slew, base_eval)) {
        return false;
    }
    DmpDriverThresholds coeff_thresholds{};
    const DmpGateArcMeta coeff_gate_arc_meta =
        makeGateArcMetaForTiming(base_eval.timing_id,
                                 base_eval.input_rf,
                                 key.attr,
                                 static_cast<float>(input_slew),
                                 coeff_thresholds);
    if (!coeff_gate_arc_meta.valid || !coeff_thresholds.valid()) {
        return false;
    }
    if (base_eval.wave.alg == DMP_ALG_CAP) {
        return netDriverPrimitiveCapTableSlopes(key,
                                                base_eval,
                                                coeff_gate_arc_meta,
                                                c1,
                                                c2,
                                                slopes);
    }
    if (!base_eval.wave.hasValidDriver() ||
        (base_eval.wave.alg != DMP_ALG_PI && base_eval.wave.alg != DMP_ALG_ZERO_C2)) {
        return false;
    }

    double rd = nanf("");
    double rd_c1 = 0.0;
    double rd_c2 = 0.0;
    double rd_input_slew = 0.0;
    if (!estimateRdWithSlopes(coeff_gate_arc_meta,
                              coeff_thresholds,
                              c1,
                              c2,
                              rd,
                              rd_c1,
                              rd_c2,
                              rd_input_slew)) {
        if (base_eval.wave.alg == DMP_ALG_PI) {
            routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailRd);
        }
        return false;
    }
    DmpRcParams rc{};
    rc.c1 = c1;
    rc.c2 = c2;
    rc.rpi = rpi;
    rc.rd = rd;
    if (base_eval.wave.alg == DMP_ALG_ZERO_C2) {
        DmpWaveCoeffs coeffs{};
        if (!rc.initZeroC2(coeffs)) {
            return false;
        }
        RouteGradDelaySlewWaveSlopes wave_slopes;
        if (!delaySlewWaveParamSlopes(base_eval.wave,
                                      base_eval.thresholds,
                                      base_eval.load_pin,
                                      key.attr,
                                      base_eval.elmore,
                                      wave_slopes)) {
            return false;
        }
        RouteGradLutSlopes intrinsic_delay_lut;
        RouteGradLutSlopes intrinsic_slew_lut;
        if (base_eval.has_extra_delay &&
            !gateArcCapDelaySlewSlopes(coeff_gate_arc_meta,
                                       0.0,
                                       intrinsic_delay_lut,
                                       intrinsic_slew_lut)) {
            return false;
        }
        RouteGradWaveParamSlopes gate_delay_wave_slopes;
        if (base_eval.has_extra_delay &&
            !driverCrossingWaveSlopes(base_eval.wave,
                                      base_eval.wave.vo_delay,
                                      gate_delay_wave_slopes)) {
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
            if (!onePoleImplicitSolveDirectionSlopes(coeff_gate_arc_meta,
                                                     coeff_thresholds,
                                                     rc,
                                                     base_eval.wave.t0,
                                                     base_eval.wave.dt,
                                                     dc1,
                                                     drd,
                                                     dinput_slew,
                                                     coeff_dir,
                                                     solve_dir)) {
                return false;
            }
            double extra_deriv = 0.0;
            if (base_eval.has_extra_delay) {
                extra_deriv = gate_delay_wave_slopes.dot(solve_dir.wave) -
                              intrinsic_delay_lut.input_slew_slope * dinput_slew;
            }
            delay_slope = wave_slopes.delay.dot(solve_dir.wave) + extra_deriv;
            slew_slope = wave_slopes.slew.dot(solve_dir.wave);
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
        if (key.input_slew_slot >= 0) {
            ok = eval_zero_direction(0.0,
                                     0.0,
                                     rd_input_slew,
                                     1.0,
                                     slopes.delay_input_slew,
                                     slopes.slew_input_slew) && ok;
        }
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
    bool solve_ok = model->findDriverParamsLocalPi(coeff_gate_arc_meta,
                                                   coeff_thresholds,
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
        solve_ok = model->findDriverParamsLocalPi(coeff_gate_arc_meta,
                                                  coeff_thresholds,
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
        solve_ok = routeGradRecoverCeffFromGateDelay(coeff_gate_arc_meta,
                                                     base_eval.gate_delay,
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

    RouteGradDelaySlewWaveSlopes wave_slopes;
    if (!delaySlewWaveParamSlopes(base_eval.wave,
                                  base_eval.thresholds,
                                  base_eval.load_pin,
                                  key.attr,
                                  base_eval.elmore,
                                  wave_slopes)) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailWaveSlope);
        return false;
    }

    RouteGradLutSlopes intrinsic_delay_lut;
    RouteGradLutSlopes intrinsic_slew_lut;
    if (base_eval.has_extra_delay &&
        !gateArcCapDelaySlewSlopes(coeff_gate_arc_meta,
                                   0.0,
                                   intrinsic_delay_lut,
                                   intrinsic_slew_lut)) {
        routeGradPrimitiveStatInc(primitive_stats, kRouteGradStatPiFailExtraLut);
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
        if (!piImplicitSolveDirectionSlopes(coeff_gate_arc_meta,
                                            coeff_thresholds,
                                            rc,
                                            coeffs,
                                            current_a,
                                            current_b,
                                            current_d,
                                            base_eval.wave.t0,
                                            base_eval.wave.dt,
                                            solve_ceff,
                                            dinput_slew,
                                            drd,
                                            coeff_dir,
                                            solve_dir)) {
            return false;
        }
        double extra_deriv = 0.0;
        if (base_eval.has_extra_delay) {
            extra_deriv = solve_dir.gate_delay - intrinsic_delay_lut.input_slew_slope * dinput_slew;
        }
        delay_slope = wave_slopes.delay.dot(solve_dir.wave) + extra_deriv;
        slew_slope = wave_slopes.slew.dot(solve_dir.wave);
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
    if (key.input_slew_slot >= 0) {
        ok = eval_direction(0.0,
                            0.0,
                            0.0,
                            rd_input_slew,
                            1.0,
                            slopes.delay_input_slew,
                            slopes.slew_input_slew) && ok;
    }
    return ok && slopes.hasFiniteValue();
}

__device__ bool RouteGradNetPrimitiveReverse::netDriverPrimitiveFiniteDiff(
    const RouteGradNetDriverSlopeKey& key,
    RouteGradGatePrimitiveSlopes& slopes) const
{
    slopes = {};
    if (model == nullptr || key.root_slot < 0 || key.root_slot >= model->dmp_pin_slot_count ||
        model->C1 == nullptr || model->C2 == nullptr || model->r_pi == nullptr ||
        model->pinSlew == nullptr) {
        return false;
    }
    const double c1 = static_cast<double>(model->C1[key.root_slot]);
    const double c2 = static_cast<double>(model->C2[key.root_slot]);
    const double rpi = static_cast<double>(model->r_pi[key.root_slot]);
    if (!isfinite(c1) || !isfinite(c2) || !isfinite(rpi) ||
        c1 < 0.0 || c2 < 0.0 || rpi < 0.0) {
        return false;
    }

    double input_slew = nanf("");
    if (key.input_slew_slot >= 0 && key.input_slew_slot < model->dmp_pin_slot_count) {
        input_slew = static_cast<double>(model->pinSlew[key.input_slew_slot]);
    } else if (key.kind == kRouteGradNetKeyGateNetPair && key.gate_arc_id >= 0 &&
               model->d_allocator != nullptr) {
        const int gate_from_pin = model->timing_arc_from_pin_id[key.gate_arc_id];
        const int from_attr = ((key.attr >> 1) << 1) | key.input_rf;
        const int timing_id = model->timing_arc_id_map[key.gate_arc_id * 2 + (key.attr >> 1)];
        if (gate_from_pin >= 0 && gate_from_pin < model->num_pins && timing_id >= 0) {
            input_slew = static_cast<double>(model->idealClockSlew(gate_from_pin, from_attr));
        }
    }
    if (!isfinite(input_slew)) {
        return false;
    }

    double base_delay = nanf("");
    double base_slew = nanf("");
    if (!netDriverDelaySlewForKey(key,
                                  c1,
                                  c2,
                                  rpi,
                                  input_slew,
                                  base_delay,
                                  base_slew)) {
        return false;
    }

    auto slope_from_samples = [] __device__ (double base,
                                             double plus,
                                             bool plus_ok,
                                             double minus,
                                             bool minus_ok,
                                             double eps) -> double {
        if (plus_ok && minus_ok && isfinite(plus) && isfinite(minus)) {
            return (plus - minus) / (2.0 * eps);
        }
        if (plus_ok && isfinite(plus) && isfinite(base)) {
            return (plus - base) / eps;
        }
        return 0.0;
    };

    auto sample_param = [&] __device__ (int param,
                                        double value,
                                        double lower_bound,
                                        double& delay_slope,
                                        double& slew_slope) {
        const double eps = routeGradRootParamStep(value);
        double plus_delay = nanf("");
        double plus_slew = nanf("");
        double minus_delay = nanf("");
        double minus_slew = nanf("");
        double pc1 = c1;
        double pc2 = c2;
        double prpi = rpi;
        double pslew = input_slew;
        if (param == 0) pc1 = value + eps;
        if (param == 1) pc2 = value + eps;
        if (param == 2) prpi = value + eps;
        if (param == 3) pslew = value + eps;
        const bool plus_ok = netDriverDelaySlewForKey(key,
                                                      pc1,
                                                      pc2,
                                                      prpi,
                                                      pslew,
                                                      plus_delay,
                                                      plus_slew);
        bool minus_ok = false;
        if (value - eps > lower_bound) {
            pc1 = c1;
            pc2 = c2;
            prpi = rpi;
            pslew = input_slew;
            if (param == 0) pc1 = value - eps;
            if (param == 1) pc2 = value - eps;
            if (param == 2) prpi = value - eps;
            if (param == 3) pslew = value - eps;
            minus_ok = netDriverDelaySlewForKey(key,
                                                pc1,
                                                pc2,
                                                prpi,
                                                pslew,
                                                minus_delay,
                                                minus_slew);
        }
        delay_slope = slope_from_samples(base_delay,
                                         plus_delay,
                                         plus_ok,
                                         minus_delay,
                                         minus_ok,
                                         eps);
        slew_slope = slope_from_samples(base_slew,
                                        plus_slew,
                                        plus_ok,
                                        minus_slew,
                                        minus_ok,
                                        eps);
    };

    sample_param(0, c1, 0.0, slopes.delay_c1, slopes.slew_c1);
    sample_param(1, c2, -1.0e-30, slopes.delay_c2, slopes.slew_c2);
    sample_param(2, rpi, 0.0, slopes.delay_rpi, slopes.slew_rpi);
    if (key.input_slew_slot >= 0) {
        sample_param(3, input_slew, 0.0, slopes.delay_input_slew, slopes.slew_input_slew);
    }
    return slopes.hasFiniteValue();
}


// Per-net-arc slope writer. This is the device entry for net delay/slew slope
// preparation; it stores compact per-pin-slot slopes that the host reverse pass
// later consumes.
__device__ void RouteGradNetPrimitiveReverse::writeSlopeForNetArc(int arc_id,
                                                                  int attr) const
{
    if (model == nullptr || delay_elmore_slope == nullptr || slew_elmore_slope == nullptr ||
        arc_id < 0 || arc_id >= model->num_arcs || attr < 0 || attr >= NUM_ATTR ||
        model->arc_types == nullptr || model->arc_types[arc_id] != 0) {
        return;
    }

    const int from_pin = model->timing_arc_from_pin_id[arc_id];
    const int to_pin = model->timing_arc_to_pin_id[arc_id];
    if (from_pin < 0 || from_pin >= model->num_pins ||
        to_pin < 0 || to_pin >= model->num_pins) {
        return;
    }
    const int to_slot = to_pin * NUM_ATTR + attr;
    if (to_slot < 0 || to_slot >= model->dmp_pin_slot_count) {
        return;
    }

    const bool pick_max = (attr >> 1) != 0;
    bool has_delay = false;
    bool has_slew = false;
    double best_delay = pick_max ? -INFINITY : INFINITY;
    double best_slew = pick_max ? -INFINITY : INFINITY;
    double delay_elmore = 0.0;
    double slew_elmore = 0.0;
    double delay_input_slew = 0.0;
    double slew_input_slew = 0.0;
    RouteGradNetDriverSlopeKey delay_key;
    RouteGradNetDriverSlopeKey slew_key;

    auto consider_candidate = [&] __device__ (double wire_delay,
                                              double load_slew,
                                              double delay_slope,
                                              double slew_slope,
                                              const RouteGradNetDriverSlopeKey& key,
                                              double candidate_delay_input_slew,
                                              double candidate_slew_input_slew) {
        if (routeGradBetterCandidate(wire_delay, best_delay, has_delay, pick_max)) {
            best_delay = wire_delay;
            delay_elmore = isfinite(delay_slope) ? delay_slope : 0.0;
            delay_input_slew = isfinite(candidate_delay_input_slew)
                                   ? candidate_delay_input_slew
                                   : 0.0;
            delay_key = key;
            has_delay = true;
        }
        if (routeGradBetterCandidate(load_slew, best_slew, has_slew, pick_max)) {
            best_slew = load_slew;
            slew_elmore = isfinite(slew_slope) ? slew_slope : 0.0;
            slew_input_slew = isfinite(candidate_slew_input_slew)
                                  ? candidate_slew_input_slew
                                  : 0.0;
            slew_key = key;
            has_slew = true;
        }
    };

    double wire_delay = nanf("");
    double load_slew = nanf("");
    double delay_slope = nanf("");
    double slew_slope = nanf("");
    if (directNetCandidate(arc_id,
                           attr,
                           wire_delay,
                           load_slew,
                           delay_slope,
                           slew_slope)) {
        RouteGradNetDriverSlopeKey key;
        double candidate_delay_input_slew = 0.0;
        double candidate_slew_input_slew = 0.0;
        makeDirectNetDriverSlopeKey(arc_id,
                                    attr,
                                    key,
                                    candidate_delay_input_slew,
                                    candidate_slew_input_slew);
        consider_candidate(wire_delay,
                           load_slew,
                           delay_slope,
                           slew_slope,
                           key,
                           candidate_delay_input_slew,
                           candidate_slew_input_slew);
    }

    if (model->pin_backward_arc_list_end != nullptr &&
        model->pin_backward_arc_list != nullptr &&
        from_pin + 1 <= model->num_pins) {
        const index_type begin = model->pin_backward_arc_list_end[from_pin];
        const index_type end = model->pin_backward_arc_list_end[from_pin + 1];
        for (index_type pos = begin; pos < end; ++pos) {
            const int gate_arc_id = model->pin_backward_arc_list[pos];
            if (gate_arc_id < 0 || gate_arc_id >= model->num_arcs ||
                model->arc_types[gate_arc_id] != 1) {
                continue;
            }
            for (int input_rf = 0; input_rf < 2; ++input_rf) {
                if (!gateNetCandidate(gate_arc_id,
                                      arc_id,
                                      attr,
                                      input_rf,
                                      wire_delay,
                                      load_slew,
                                      delay_slope,
                                      slew_slope)) {
                    continue;
                }
                RouteGradNetDriverSlopeKey key;
                makeGateNetDriverSlopeKey(gate_arc_id,
                                          arc_id,
                                          attr,
                                          input_rf,
                                          key);
                consider_candidate(wire_delay,
                                   load_slew,
                                   delay_slope,
                                   slew_slope,
                                   key,
                                   0.0,
                                   0.0);
            }
        }
    }

    const int delay_idx = (attr << 1) + (attr & 1);
    const int delay_slot = arc_id * 2 * NUM_ATTR + delay_idx;
    if (!has_delay && model->arcDelay != nullptr && delay_slot >= 0 &&
        delay_slot < model->num_arcs * 2 * NUM_ATTR &&
        isfinite(model->arcDelay[delay_slot])) {
        delay_elmore = 1.0;
        delay_key = {};
        has_delay = true;
    }
    if (!has_delay && !has_slew) {
        return;
    }

    if (has_delay) {
        delay_elmore_slope[to_slot] = isfinite(delay_elmore)
                                          ? static_cast<float>(delay_elmore)
                                          : 0.0f;
        if (delay_driver_root_slot != nullptr) {
            delay_driver_root_slot[to_slot] = delay_key.root_slot;
        }
        if (delay_driver_input_slew_slot != nullptr) {
            delay_driver_input_slew_slot[to_slot] = delay_key.input_slew_slot;
        }
        if (delay_input_slew_slope != nullptr) {
            delay_input_slew_slope[to_slot] = isfinite(delay_input_slew)
                                                  ? static_cast<float>(delay_input_slew)
                                                  : 0.0f;
        }
        if (delay_key.kind != 0) {
            RouteGradGatePrimitiveSlopes driver_slopes;
            const bool analytic_ok = netDriverPrimitiveWaveChainSlopes(delay_key, driver_slopes);
            routeGradPrimitiveStatInc(primitive_stats,
                                      analytic_ok ? kRouteGradStatNetDelayAnalytic
                                                  : kRouteGradStatNetDelayFail);
            if (analytic_ok) {
                if (delay_driver_root_slot != nullptr) {
                    delay_driver_root_slot[to_slot] = delay_key.root_slot;
                }
                if (delay_driver_input_slew_slot != nullptr) {
                    delay_driver_input_slew_slot[to_slot] = delay_key.input_slew_slot;
                }
                if (delay_c1_slope != nullptr) {
                    delay_c1_slope[to_slot] = isfinite(driver_slopes.delay_c1)
                                                  ? static_cast<float>(driver_slopes.delay_c1)
                                                  : 0.0f;
                }
                if (delay_c2_slope != nullptr) {
                    delay_c2_slope[to_slot] = isfinite(driver_slopes.delay_c2)
                                                  ? static_cast<float>(driver_slopes.delay_c2)
                                                  : 0.0f;
                }
                if (delay_rpi_slope != nullptr) {
                    delay_rpi_slope[to_slot] = isfinite(driver_slopes.delay_rpi)
                                                   ? static_cast<float>(driver_slopes.delay_rpi)
                                                   : 0.0f;
                }
                if (delay_input_slew_slope != nullptr) {
                    delay_input_slew_slope[to_slot] = isfinite(driver_slopes.delay_input_slew)
                                                          ? static_cast<float>(driver_slopes.delay_input_slew)
                                                          : 0.0f;
                }
            }
        }
    }

    if (has_slew) {
        slew_elmore_slope[to_slot] = isfinite(slew_elmore)
                                         ? static_cast<float>(slew_elmore)
                                         : 0.0f;
        if (slew_driver_root_slot != nullptr) {
            slew_driver_root_slot[to_slot] = slew_key.root_slot;
        }
        if (slew_driver_input_slew_slot != nullptr) {
            slew_driver_input_slew_slot[to_slot] = slew_key.input_slew_slot;
        }
        if (slew_input_slew_slope != nullptr) {
            slew_input_slew_slope[to_slot] = isfinite(slew_input_slew)
                                                 ? static_cast<float>(slew_input_slew)
                                                 : 0.0f;
        }
        if (slew_key.kind != 0) {
            RouteGradGatePrimitiveSlopes driver_slopes;
            const bool analytic_ok = netDriverPrimitiveWaveChainSlopes(slew_key, driver_slopes);
            routeGradPrimitiveStatInc(primitive_stats,
                                      analytic_ok ? kRouteGradStatNetSlewAnalytic
                                                  : kRouteGradStatNetSlewFail);
            if (analytic_ok) {
                if (slew_driver_root_slot != nullptr) {
                    slew_driver_root_slot[to_slot] = slew_key.root_slot;
                }
                if (slew_driver_input_slew_slot != nullptr) {
                    slew_driver_input_slew_slot[to_slot] = slew_key.input_slew_slot;
                }
                if (slew_c1_slope != nullptr) {
                    slew_c1_slope[to_slot] = isfinite(driver_slopes.slew_c1)
                                                 ? static_cast<float>(driver_slopes.slew_c1)
                                                 : 0.0f;
                }
                if (slew_c2_slope != nullptr) {
                    slew_c2_slope[to_slot] = isfinite(driver_slopes.slew_c2)
                                                 ? static_cast<float>(driver_slopes.slew_c2)
                                                 : 0.0f;
                }
                if (slew_rpi_slope != nullptr) {
                    slew_rpi_slope[to_slot] = isfinite(driver_slopes.slew_rpi)
                                                  ? static_cast<float>(driver_slopes.slew_rpi)
                                                  : 0.0f;
                }
                if (slew_input_slew_slope != nullptr) {
                    slew_input_slew_slope[to_slot] = isfinite(driver_slopes.slew_input_slew)
                                                         ? static_cast<float>(driver_slopes.slew_input_slew)
                                                         : 0.0f;
                }
            }
        }
    }
}

__global__ void routeGradNetElmoreSlopeKernel(RouteGradNetPrimitiveReverse op)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = op.model ? op.model->num_arcs * NUM_ATTR : 0;
    if (idx >= total) {
        return;
    }
    op.writeSlopeForNetArc(idx / NUM_ATTR, idx & 0x3);
}

__device__ void RouteGradActiveGatePrimitiveSlope::writeActiveGateSlope(int to_slot) const
{
    if (model == nullptr || root_slot == nullptr ||
        delay_c1_slope == nullptr || delay_c2_slope == nullptr ||
        delay_rpi_slope == nullptr || delay_input_slew_slope == nullptr ||
        slew_c1_slope == nullptr || slew_c2_slope == nullptr ||
        slew_rpi_slope == nullptr || slew_input_slew_slope == nullptr ||
        to_slot < 0 || to_slot >= model->dmp_pin_slot_count ||
        model->at_prefix_arc == nullptr || model->at_prefix_attr == nullptr ||
        model->arc_types == nullptr) {
        return;
    }
    const int gate_arc_id = model->at_prefix_arc[to_slot];
    const int from_attr = model->at_prefix_attr[to_slot];
    const int to_attr = to_slot & 0x3;
    if (gate_arc_id < 0 || gate_arc_id >= model->num_arcs ||
        from_attr < 0 || from_attr >= NUM_ATTR ||
        model->arc_types[gate_arc_id] != 1) {
        return;
    }

    RouteGradNetPrimitiveReverse helper;
    helper.model = model;
    helper.primitive_stats = primitive_stats;
    RouteGradGatePrimitiveSlopes slopes;
    const bool analytic_ok = helper.gatePrimitiveWaveChainSlopes(gate_arc_id,
                                                                 from_attr,
                                                                 to_attr,
                                                                 to_slot,
                                                                 slopes);
    routeGradPrimitiveStatInc(primitive_stats,
                              analytic_ok ? kRouteGradStatActiveGateAnalytic
                                          : kRouteGradStatActiveGateFail);
    if (!analytic_ok) {
        return;
    }
    root_slot[to_slot] = to_slot;
    delay_c1_slope[to_slot] = isfinite(slopes.delay_c1)
                                  ? static_cast<float>(slopes.delay_c1)
                                  : 0.0f;
    delay_c2_slope[to_slot] = isfinite(slopes.delay_c2)
                                  ? static_cast<float>(slopes.delay_c2)
                                  : 0.0f;
    delay_rpi_slope[to_slot] = isfinite(slopes.delay_rpi)
                                   ? static_cast<float>(slopes.delay_rpi)
                                   : 0.0f;
    delay_input_slew_slope[to_slot] = isfinite(slopes.delay_input_slew)
                                          ? static_cast<float>(slopes.delay_input_slew)
                                          : 0.0f;
    slew_c1_slope[to_slot] = isfinite(slopes.slew_c1)
                                 ? static_cast<float>(slopes.slew_c1)
                                 : 0.0f;
    slew_c2_slope[to_slot] = isfinite(slopes.slew_c2)
                                 ? static_cast<float>(slopes.slew_c2)
                                 : 0.0f;
    slew_rpi_slope[to_slot] = isfinite(slopes.slew_rpi)
                                  ? static_cast<float>(slopes.slew_rpi)
                                  : 0.0f;
    slew_input_slew_slope[to_slot] = isfinite(slopes.slew_input_slew)
                                         ? static_cast<float>(slopes.slew_input_slew)
                                         : 0.0f;
}

__global__ void routeGradActiveGatePrimitiveSlopeKernel(RouteGradActiveGatePrimitiveSlope op)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = op.model ? op.model->dmp_pin_slot_count : 0;
    if (idx >= total) {
        return;
    }
    op.writeActiveGateSlope(idx);
}

__device__ void RouteGradActiveGateSlewWinnerSlope::writeGateSlewWinnerSlope(int to_slot) const
{
    if (model == nullptr || root_slot == nullptr || input_slew_slot == nullptr ||
        slew_c1_slope == nullptr || slew_c2_slope == nullptr ||
        slew_rpi_slope == nullptr || slew_input_slew_slope == nullptr ||
        to_slot < 0 || to_slot >= model->dmp_pin_slot_count ||
        model->pin_backward_arc_list_end == nullptr ||
        model->pin_backward_arc_list == nullptr || model->arc_types == nullptr ||
        model->pinSlew == nullptr || model->d_allocator == nullptr) {
        return;
    }

    const int to_pin = to_slot / NUM_ATTR;
    const int to_attr = to_slot & 0x3;
    if (to_pin < 0 || to_pin >= model->num_pins || to_pin + 1 > model->num_pins) {
        return;
    }
    const int el = to_attr >> 1;
    const bool pick_max = el != 0;
    bool has_best = false;
    double best_slew = pick_max ? -INFINITY : INFINITY;
    int best_arc = -1;
    int best_from_attr = -1;
    int best_input_slot = -1;

    RouteGradNetPrimitiveReverse helper;
    helper.model = model;
    helper.primitive_stats = primitive_stats;
    const index_type begin = model->pin_backward_arc_list_end[to_pin];
    const index_type end = model->pin_backward_arc_list_end[to_pin + 1];
    for (index_type pos = begin; pos < end; ++pos) {
        const int gate_arc_id = model->pin_backward_arc_list[pos];
        if (gate_arc_id < 0 || gate_arc_id >= model->num_arcs ||
            model->arc_types[gate_arc_id] != 1 ||
            model->timing_arc_to_pin_id[gate_arc_id] != to_pin) {
            continue;
        }
        const int from_pin = model->timing_arc_from_pin_id[gate_arc_id];
        const int timing_id = model->timing_arc_id_map[gate_arc_id * 2 + el];
        if (timing_id < 0 || from_pin < 0 || from_pin >= model->num_pins) {
            continue;
        }
        const bool ideal_clock_arc =
            model->isIdealClockTimingArc(timing_id, from_pin) &&
            !model->d_allocator->timing_is_constraint(timing_id);
        for (int input_rf = 0; input_rf < 2; ++input_rf) {
            const int from_attr = (el << 1) | input_rf;
            const int from_slot = from_pin * NUM_ATTR + from_attr;
            if (from_slot < 0 || from_slot >= model->dmp_pin_slot_count) {
                continue;
            }
            const float input_slew = ideal_clock_arc
                                         ? model->idealClockSlew(from_pin, from_attr)
                                         : model->pinSlew[from_slot];
            if (!isfinite(input_slew)) {
                continue;
            }
            DmpDriverThresholds thresholds{};
            DmpDriverWave wave;
            float gate_delay = nanf("");
            const DmpGateArcMeta gate_arc_meta =
                helper.makeGateArcMetaForTiming(timing_id,
                                                input_rf,
                                                to_attr,
                                                input_slew,
                                                thresholds);
            if (!model->computeGateDriverWaveForSlot(gate_arc_meta,
                                                     thresholds,
                                                     to_slot,
                                                     wave,
                                                     gate_delay) ||
                !isfinite(wave.vo_slew)) {
                continue;
            }
            const double candidate = static_cast<double>(wave.vo_slew);
            if (routeGradBetterCandidate(candidate, best_slew, has_best, pick_max)) {
                best_slew = candidate;
                best_arc = gate_arc_id;
                best_from_attr = from_attr;
                best_input_slot = ideal_clock_arc ? -1 : from_slot;
                has_best = true;
            }
        }
    }
    if (!has_best || best_arc < 0 || best_from_attr < 0) {
        return;
    }

    RouteGradGatePrimitiveSlopes slopes;
    const bool analytic_ok = helper.gatePrimitiveWaveChainSlopes(best_arc,
                                                                 best_from_attr,
                                                                 to_attr,
                                                                 to_slot,
                                                                 slopes);
    routeGradPrimitiveStatInc(primitive_stats,
                              analytic_ok ? kRouteGradStatGateSlewAnalytic
                                          : kRouteGradStatGateSlewFail);
    if (!analytic_ok) {
        return;
    }
    root_slot[to_slot] = to_slot;
    input_slew_slot[to_slot] = best_input_slot;
    slew_c1_slope[to_slot] = isfinite(slopes.slew_c1)
                                 ? static_cast<float>(slopes.slew_c1)
                                 : 0.0f;
    slew_c2_slope[to_slot] = isfinite(slopes.slew_c2)
                                 ? static_cast<float>(slopes.slew_c2)
                                 : 0.0f;
    slew_rpi_slope[to_slot] = isfinite(slopes.slew_rpi)
                                  ? static_cast<float>(slopes.slew_rpi)
                                  : 0.0f;
    slew_input_slew_slope[to_slot] = isfinite(slopes.slew_input_slew)
                                         ? static_cast<float>(slopes.slew_input_slew)
                                         : 0.0f;
}

__global__ void routeGradActiveGateSlewWinnerSlopeKernel(RouteGradActiveGateSlewWinnerSlope op)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = op.model ? op.model->dmp_pin_slot_count : 0;
    if (idx >= total) {
        return;
    }
    op.writeGateSlewWinnerSlope(idx);
}

}  // namespace gt
