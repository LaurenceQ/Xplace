#pragma once

#include "gputimer/core/DmpGateModel.cuh"
#include "gputimer/core/DmpModel.h"

namespace gt {

inline constexpr int kRouteGradStatNetDelayAnalytic = 0;
inline constexpr int kRouteGradStatNetDelayFail = 2;
inline constexpr int kRouteGradStatNetSlewAnalytic = 3;
inline constexpr int kRouteGradStatNetSlewFail = 5;
inline constexpr int kRouteGradStatActiveGateAnalytic = 6;
inline constexpr int kRouteGradStatActiveGateFail = 8;
inline constexpr int kRouteGradStatGateSlewAnalytic = 9;
inline constexpr int kRouteGradStatGateSlewFail = 11;
inline constexpr int kRouteGradStatPiFailCoeff = 28;
inline constexpr int kRouteGradStatPiFailImplicitSetup = 29;
inline constexpr int kRouteGradStatPiFailImplicitLut = 30;
inline constexpr int kRouteGradStatPiFailImplicitEquation = 31;
inline constexpr int kRouteGradStatPiFailImplicitSolve = 32;
inline constexpr int kRouteGradStatPiFailImplicitOutput = 33;
inline constexpr int kRouteGradStatPiFailRd = 34;
inline constexpr int kRouteGradStatPiFailInit = 35;
inline constexpr int kRouteGradStatPiFailForwardSolve = 36;
inline constexpr int kRouteGradStatPiFailWaveSlope = 37;
inline constexpr int kRouteGradStatPiFailExtraLut = 38;
inline constexpr int kRouteGradStatPiRecoveredCeffFromDelay = 39;
inline constexpr int kRouteGradPrimitiveStatCount = 40;

inline constexpr unsigned int kRouteGradLutMetaScalar = 1u << 0;
inline constexpr unsigned int kRouteGradLutMetaVar0IsSlew = 1u << 1;

inline constexpr int kRouteGradNetKeyDirectDrivingCell = 1;
inline constexpr int kRouteGradNetKeyGateNetPair = 2;

struct RouteGradObjectiveInit {
    const float* pin_at = nullptr;
    const float* pin_rat = nullptr;
    const int64_t* endpoint_ids = nullptr;
    double* bar_pin_at = nullptr;
    double tau_ns = 0.02;
    double time_to_ns = 1.0;
    int endpoint_count = 0;
};

struct RouteGradPinWinnerReverse {
    const index_type* at_prefix_pin = nullptr;
    const index_type* at_prefix_arc = nullptr;
    const index_type* at_prefix_attr = nullptr;
    const uint8_t* arc_types = nullptr;
    double* bar_pin_at = nullptr;
    double* bar_elmore = nullptr;
    int num_pins = 0;
    int num_arcs = 0;
};

struct RouteGradRcTreeReverse {
    const int* edge_from = nullptr;
    const int* edge_to = nullptr;
    const float* edge_res = nullptr;
    const float* node_cap = nullptr;
    const int* node2pin = nullptr;
    const int* net2node_start = nullptr;
    const int* net2edge_start = nullptr;
    const uint8_t* includes_pin_caps = nullptr;
    const float* pin_cap = nullptr;
    const double* bar_elmore = nullptr;
    double* edge_res_grad = nullptr;
    double* node_cap_grad = nullptr;
    double rc_time_factor = 1.0;
    int num_nets = 0;
    int num_nodes = 0;
    int num_edges = 0;
};

struct RouteGradGatePrimitiveReverse {
    // Reserved for the full PI/cell-load adjoint. Kept separate so gate
    // primitive temporaries never extend RC-tree reverse live ranges.
};




struct RouteGradLutSlopes {
    double value = 0.0;
    double input_slew_slope = 0.0;
    double load_slope = 0.0;
};

struct RouteGradWaveParamSlopes {
    double t0 = 0.0;
    double dt = 0.0;
    double k0 = 0.0;
    double k1 = 0.0;
    double k2 = 0.0;
    double k3 = 0.0;
    double k4 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;

    __device__ void clear();
    __device__ void scale(double factor);
    __device__ double dot(const RouteGradWaveParamSlopes& rhs) const;
};

struct RouteGradPiCoeffSlopes {
    RouteGradWaveParamSlopes wave;
    double current_a = 0.0;
    double current_b = 0.0;
    double current_d = 0.0;

    __device__ void clear();
};

struct RouteGradPiSolveSlopes {
    RouteGradWaveParamSlopes wave;
    double ceff = 0.0;
    double gate_delay = 0.0;

    __device__ void clear();
};

struct RouteGradOnePoleSolveSlopes {
    RouteGradWaveParamSlopes wave;

    __device__ void clear();
};

struct RouteGradDelaySlewWaveSlopes {
    RouteGradWaveParamSlopes delay;
    RouteGradWaveParamSlopes slew;
};

struct RouteGradNetDriverWaveEval {
    DmpDriverWave wave;
    DmpDriverThresholds thresholds;
    double elmore = 0.0;
    double gate_delay = 0.0;
    double intrinsic_delay = 0.0;
    int timing_id = -1;
    int input_rf = -1;
    int load_pin = -1;
    bool has_extra_delay = false;
};

struct RouteGradNetDriverSlopeKey {
    int kind = 0;
    int net_arc_id = -1;
    int gate_arc_id = -1;
    int attr = -1;
    int input_rf = -1;
    int root_slot = -1;
    int input_slew_slot = -1;
};

struct RouteGradGatePrimitiveSlopes {
    double delay_c1 = 0.0;
    double delay_c2 = 0.0;
    double delay_rpi = 0.0;
    double delay_input_slew = 0.0;
    double slew_c1 = 0.0;
    double slew_c2 = 0.0;
    double slew_rpi = 0.0;
    double slew_input_slew = 0.0;

    __device__ bool hasFiniteValue() const;
};

struct RouteGradNetPrimitiveReverse {
    DmpModel* model = nullptr;
    float* delay_elmore_slope = nullptr;
    float* slew_elmore_slope = nullptr;
    int* delay_driver_root_slot = nullptr;
    int* delay_driver_input_slew_slot = nullptr;
    int* slew_driver_root_slot = nullptr;
    int* slew_driver_input_slew_slot = nullptr;
    float* delay_c1_slope = nullptr;
    float* delay_c2_slope = nullptr;
    float* delay_rpi_slope = nullptr;
    float* delay_input_slew_slope = nullptr;
    float* slew_c1_slope = nullptr;
    float* slew_c2_slope = nullptr;
    float* slew_rpi_slope = nullptr;
    float* slew_input_slew_slope = nullptr;
    unsigned long long* primitive_stats = nullptr;

    __device__ double thresholdArrayValue(const float* values,
                                          int attr,
                                          double fallback) const;
    __device__ double libraryThresholdArrayValue(const float* values,
                                                 int library_idx,
                                                 double fallback) const;
    __device__ int timingLibraryId(int timing_id) const;
    __device__ int pinLibraryId(int pin_id) const;
    __device__ void loadPinThresholds(int pin_id,
                                      int attr,
                                      double& vth,
                                      double& vl,
                                      double& vh,
                                      double& slew_derate) const;
    __device__ void driverLibraryThresholds(int library_id,
                                            int attr,
                                            double& vth,
                                            double& vl,
                                            double& vh,
                                            double& slew_derate) const;
    __device__ DmpGateArcMeta makeGateArcMetaForTiming(int timing_id,
                                                       int input_rf,
                                                       int to_attr,
                                                       float input_slew,
                                                       DmpDriverThresholds& thresholds) const;
    __device__ void thresholdAdjustedSlopes(int load_pin_id,
                                            int load_attr,
                                            float driver_vth,
                                            float driver_vl,
                                            float driver_vh,
                                            float driver_derate,
                                            int driver_library_id,
                                            double raw_delay_slope,
                                            double raw_slew_slope,
                                            double& delay_slope,
                                            double& slew_slope) const;
    __device__ void thresholdAdjustedWaveSlopes(int load_pin_id,
                                                int load_attr,
                                                const DmpDriverThresholds& thresholds,
                                                const RouteGradWaveParamSlopes& raw_delay,
                                                const RouteGradWaveParamSlopes& raw_slew,
                                                RouteGradDelaySlewWaveSlopes& final_slopes) const;
    __device__ bool gateLutValueSlopes(GPULutAllocator* allocator,
                                       const DmpGateLutMeta& meta,
                                       float input_slew,
                                       float load,
                                       RouteGradLutSlopes& slopes) const;
    __device__ bool gateArcCapDelaySlewSlopes(const DmpGateArcMeta& gate_arc_meta,
                                              double load_cap,
                                              RouteGradLutSlopes& delay,
                                              RouteGradLutSlopes& slew) const;
    __device__ bool estimateRdWithSlopes(const DmpGateArcMeta& gate_arc_meta,
                                         const DmpDriverThresholds& thresholds,
                                         double c1,
                                         double c2,
                                         double& rd,
                                         double& rd_c1,
                                         double& rd_c2,
                                         double& rd_input_slew) const;
    __device__ bool piCoeffRootParamSlopes(const DmpGateArcMeta& gate_arc_meta,
                                           const DmpDriverThresholds& thresholds,
                                           double c1,
                                           double c2,
                                           double rpi,
                                           RouteGradWaveParamSlopes& c1_slopes,
                                           RouteGradWaveParamSlopes& c2_slopes,
                                           RouteGradWaveParamSlopes& rpi_slopes,
                                           RouteGradWaveParamSlopes& input_slew_slopes) const;
    __device__ bool zeroC2CoeffDirectionSlopes(const DmpRcParams& rc,
                                               const DmpWaveCoeffs& coeffs,
                                               double dc1,
                                               double drpi,
                                               double drd,
                                               RouteGradWaveParamSlopes& slopes) const;
    __device__ bool onePoleImplicitSolveDirectionSlopes(const DmpGateArcMeta& gate_arc_meta,
                                                        const DmpDriverThresholds& thresholds,
                                                        const DmpRcParams& rc,
                                                        double t0,
                                                        double dt,
                                                        double dc1,
                                                        double drd,
                                                        double dinput_slew,
                                                        const RouteGradWaveParamSlopes& coeff_slopes,
                                                        RouteGradOnePoleSolveSlopes& solve_slopes) const;
    __device__ bool piCoeffDirectionSlopes(const DmpRcParams& rc,
                                           const DmpWaveCoeffs& coeffs,
                                           double current_a,
                                           double current_b,
                                           double current_d,
                                           double dc1,
                                           double dc2,
                                           double drpi,
                                           double drd,
                                           RouteGradPiCoeffSlopes& slopes) const;
    __device__ bool piImplicitSolveDirectionSlopes(const DmpGateArcMeta& gate_arc_meta,
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
                                                   RouteGradPiSolveSlopes& solve_slopes) const;
    __device__ bool driverBaseWavePartials(const DmpDriverWave& wave,
                                           double t,
                                           RouteGradWaveParamSlopes& partials,
                                           double& value,
                                           double& value_dt) const;
    __device__ bool loadBaseWavePartials(const DmpDriverWave& wave,
                                         double elmore,
                                         double t,
                                         RouteGradWaveParamSlopes& partials,
                                         double& value,
                                         double& value_dt) const;
    __device__ bool driverWaveValuePartials(const DmpDriverWave& wave,
                                            double t,
                                            RouteGradWaveParamSlopes& partials,
                                            double& value_dt) const;
    __device__ bool loadWaveValuePartials(const DmpDriverWave& wave,
                                          double elmore,
                                          double t,
                                          RouteGradWaveParamSlopes& partials,
                                          double& value_dt,
                                          double& value_elmore) const;
    __device__ bool driverCrossingWaveSlopes(const DmpDriverWave& wave,
                                             double crossing,
                                             RouteGradWaveParamSlopes& slopes) const;
    __device__ bool loadCrossingWaveSlopes(const DmpDriverWave& wave,
                                           double elmore,
                                           double crossing,
                                           RouteGradWaveParamSlopes& slopes,
                                           double& elmore_slope) const;
    __device__ double inputPortDelayElmoreSlope(int load_pin_id,
                                                int load_attr,
                                                double& slew_slope) const;
    __device__ double loadWave0ElmoreDerivative(const DmpDriverWave& wave,
                                                double elmore,
                                                double t) const;
    __device__ double loadWaveElmoreDerivative(const DmpDriverWave& wave,
                                               double elmore,
                                               double t) const;
    __device__ bool loadCrossingElmoreSlope(const DmpDriverWave& wave,
                                            double elmore,
                                            double crossing,
                                            double& slope) const;
    __device__ bool delaySlewSlopeForDriverWave(const DmpDriverWave& wave,
                                                const DmpDriverThresholds& thresholds,
                                                int load_pin_id,
                                                int load_attr,
                                                double elmore,
                                                double& wire_delay,
                                                double& load_slew,
                                                double& delay_slope,
                                                double& slew_slope) const;
    __device__ bool directNetCandidate(int arc_id,
                                       int attr,
                                       double& wire_delay,
                                       double& load_slew,
                                       double& delay_slope,
                                       double& slew_slope) const;
    __device__ bool computeDriverWaveForRc(const DmpGateArcMeta& gate_arc_meta,
                                           const DmpDriverThresholds& thresholds,
                                           double c1,
                                           double c2,
                                           double rpi,
                                           DmpDriverWave& driver_wave,
                                           float& gate_delay) const;
    __device__ bool gateDelaySlewWithRootRc(int gate_arc_id,
                                            int from_attr,
                                            int to_attr,
                                            double c1,
                                            double c2,
                                            double rpi,
                                            double input_slew,
                                            double& gate_delay,
                                            double& gate_slew) const;
    __device__ bool gatePrimitiveFiniteDiff(int gate_arc_id,
                                            int from_attr,
                                            int to_attr,
                                            int root_slot,
                                            RouteGradGatePrimitiveSlopes& slopes) const;
    __device__ bool gateNetCandidate(int gate_arc_id,
                                     int net_arc_id,
                                     int attr,
                                     int input_rf,
                                     double& wire_delay,
                                     double& load_slew,
                                     double& delay_slope,
                                     double& slew_slope) const;
    __device__ bool makeDirectNetDriverSlopeKey(int arc_id,
                                                int attr,
                                                RouteGradNetDriverSlopeKey& key,
                                                double& delay_input_slew_slope,
                                                double& slew_input_slew_slope) const;
    __device__ bool makeGateNetDriverSlopeKey(int gate_arc_id,
                                              int net_arc_id,
                                              int attr,
                                              int input_rf,
                                              RouteGradNetDriverSlopeKey& key) const;
    __device__ bool netDriverWaveForKey(const RouteGradNetDriverSlopeKey& key,
                                       double c1,
                                       double c2,
                                       double rpi,
                                       double input_slew,
                                       RouteGradNetDriverWaveEval& eval) const;
    __device__ bool netDriverDelaySlewForKey(const RouteGradNetDriverSlopeKey& key,
                                             double c1,
                                             double c2,
                                             double rpi,
                                             double input_slew,
                                             double& wire_delay,
                                             double& load_slew) const;
    __device__ bool delaySlewWaveParamSlopes(const DmpDriverWave& wave,
                                             const DmpDriverThresholds& thresholds,
                                             int load_pin_id,
                                             int load_attr,
                                             double elmore,
                                             RouteGradDelaySlewWaveSlopes& slopes) const;
    __device__ bool driverOutputSlewWaveParamSlopes(const DmpDriverWave& wave,
                                                    const DmpDriverThresholds& thresholds,
                                                    RouteGradWaveParamSlopes& slopes) const;
    __device__ bool gatePrimitiveWaveChainSlopes(int gate_arc_id,
                                                 int from_attr,
                                                 int to_attr,
                                                 int root_slot,
                                                 RouteGradGatePrimitiveSlopes& slopes) const;
    __device__ bool netDriverPrimitiveCapTableSlopes(const RouteGradNetDriverSlopeKey& key,
                                                     const RouteGradNetDriverWaveEval& eval,
                                                     const DmpGateArcMeta& gate_arc_meta,
                                                     double c1,
                                                     double c2,
                                                     RouteGradGatePrimitiveSlopes& slopes) const;
    __device__ int classifyGatePrimitiveAlg(int gate_arc_id,
                                            int from_attr,
                                            int to_attr,
                                            int root_slot) const;
    __device__ int classifyNetDriverPrimitiveAlg(const RouteGradNetDriverSlopeKey& key) const;
    __device__ bool netDriverPrimitiveWaveChainSlopes(const RouteGradNetDriverSlopeKey& key,
                                                      RouteGradGatePrimitiveSlopes& slopes) const;
    __device__ bool netDriverPrimitiveFiniteDiff(const RouteGradNetDriverSlopeKey& key,
                                                 RouteGradGatePrimitiveSlopes& slopes) const;
    __device__ void writeSlopeForNetArc(int arc_id,
                                        int attr) const;
};


struct RouteGradActiveGateSlewWinnerSlope {
    DmpModel* model = nullptr;
    int* root_slot = nullptr;
    int* input_slew_slot = nullptr;
    float* slew_c1_slope = nullptr;
    float* slew_c2_slope = nullptr;
    float* slew_rpi_slope = nullptr;
    float* slew_input_slew_slope = nullptr;
    unsigned long long* primitive_stats = nullptr;

    __device__ void writeGateSlewWinnerSlope(int to_slot) const;
};

struct RouteGradActiveGatePrimitiveSlope {
    DmpModel* model = nullptr;
    int* root_slot = nullptr;
    float* delay_c1_slope = nullptr;
    float* delay_c2_slope = nullptr;
    float* delay_rpi_slope = nullptr;
    float* delay_input_slew_slope = nullptr;
    float* slew_c1_slope = nullptr;
    float* slew_c2_slope = nullptr;
    float* slew_rpi_slope = nullptr;
    float* slew_input_slew_slope = nullptr;
    unsigned long long* primitive_stats = nullptr;

    __device__ void writeActiveGateSlope(int to_slot) const;
};

__global__ void routeGradNetElmoreSlopeKernel(RouteGradNetPrimitiveReverse op);
__global__ void routeGradActiveGatePrimitiveSlopeKernel(RouteGradActiveGatePrimitiveSlope op);
__global__ void routeGradActiveGateSlewWinnerSlopeKernel(RouteGradActiveGateSlewWinnerSlope op);

}  // namespace gt
