#pragma once

#include "GPUTimer.h"
// #include "gputiming.h"
// #include "utils.cuh"
#include <cstdint>
#include <vector>
#ifdef __CUDACC__
#define CUDA_DEV __device__
#define CUDA_DEV_INLINE __device__ inline
#define CUDA_DEV_NOINLINE __device__ __noinline__
#else
#define CUDA_DEV
#define CUDA_DEV_INLINE
#define CUDA_DEV_NOINLINE
#endif



namespace gt {
class GPUTimer;
class GPULutAllocator;
struct DmpGateArcMeta;
struct DmpDriverThresholds;
struct DmpRcParams;
struct DmpWaveCoeffs;
struct DmpDriverWave;

enum DmpAlgKind { DMP_ALG_CAP = 0, DMP_ALG_ZERO_C2 = 1, DMP_ALG_PI = 2 };
// Source pins terminate path tracing, so their prefix slots are otherwise unused.
// For set_driving_cell sources, at_prefix_arc stores (timing_id << 1) | input_rf,
// at_prefix_attr is this sentinel, and pinSlew stores the SDC input transition.
static constexpr int DMP_DRIVING_CELL_PREFIX_ATTR = -2;
static constexpr uint8_t DMP_PIN_PRIMARY_INPUT = 1u << 0;
static constexpr uint8_t DMP_PIN_CLK = 1u << 1;
static constexpr uint8_t DMP_PIN_IDEAL_CLK = 1u << 2;

struct DmpModel {
    const char **pin_names;
    const char **net_names;
    int num_pins, num_nets, num_arcs, num_tests;
    const int *flat_net2pin_start_map, *flat_net2pin_map, *pin2net_map;
    float *C1, *C2, *r_pi;
    float *timer_ceff;
    int dmp_pin_slot_count, dmp_slot_capacity, dmp_work_slot_capacity;
    bool owns_allocations;
    unsigned long long *pin_at_winner;
    const float *dmp_input_thresholds;
    const float *dmp_output_thresholds;
    const float *dmp_slew_lower_thresholds;
    const float *dmp_slew_upper_thresholds;
    const float *dmp_slew_derates;
    const int *dmp_timing_library_ids;
    const int *dmp_pin_library_ids;
    const float *dmp_library_input_thresholds;
    const float *dmp_library_output_thresholds;
    const float *dmp_library_slew_lower_thresholds;
    const float *dmp_library_slew_upper_thresholds;
    const float *dmp_library_slew_derates;
    uint8_t *pin_flags;
    index_type *level_list;
    index_type *pin_forward_arc_list_end;
    index_type *pin_forward_arc_list;
    index_type *timing_arc_to_pin_id;
    index_type *pin_backward_arc_list_end;
    index_type *pin_backward_arc_list;
    index_type *timing_arc_from_pin_id;
    index_type *at_prefix_pin;
    index_type *at_prefix_arc;
    index_type *at_prefix_attr;
    const uint8_t *arc_types;
    int *arc_id2test_id;
    float *pinSlew;
    float *elmore_delay; // elmore delay
    float *pinAt;
    float *pinRat;
    float *testRelatedAT;
    float *testRAT;
    float *testConstraint;
    uint8_t *test_clock_ids;
    float *clock_periods;
    int clock_count;
    const float *pin_clock_rise_edges;
    const float *pin_clock_fall_edges;
    const float *pin_clock_slews;
    const float *test_setup_uncertainties;
    const float *test_hold_uncertainties;
    float *arcDelay;
    int *timing_arc_id_map;
    float clock_period;
    GPULutAllocator *d_allocator;
    float res_unit;
    float cap_unit;
    const double vth_time_tol = .01;// vars here are usally the case
    const double x_tol = .01;
    const double slew_derate_ = 1.0; 
    const double vth_ = 0.5;
    const double vl_ = 0.2;
    const double vh_ = 0.8;
    const int MAX_ITER = 20;
    DmpModel() : num_pins(0), num_nets(0), num_arcs(0), num_tests(0), pin_names(nullptr),
                  flat_net2pin_start_map(nullptr), flat_net2pin_map(nullptr), pin2net_map(nullptr),
                  C1(nullptr), C2(nullptr), r_pi(nullptr), timer_ceff(nullptr),
                  dmp_pin_slot_count(0), dmp_slot_capacity(0), dmp_work_slot_capacity(0),
                  owns_allocations(false),
                  level_list(nullptr),
                  pin_forward_arc_list_end(nullptr), pin_forward_arc_list(nullptr),
                  timing_arc_to_pin_id(nullptr),
                  pin_backward_arc_list_end(nullptr), pin_backward_arc_list(nullptr),
                  timing_arc_from_pin_id(nullptr),
                  arc_types(nullptr), arc_id2test_id(nullptr),
                  pin_at_winner(nullptr),
                  dmp_input_thresholds(nullptr), dmp_output_thresholds(nullptr),
                  dmp_slew_lower_thresholds(nullptr), dmp_slew_upper_thresholds(nullptr),
                  dmp_slew_derates(nullptr), dmp_timing_library_ids(nullptr),
                  dmp_pin_library_ids(nullptr),
                  dmp_library_input_thresholds(nullptr),
                  dmp_library_output_thresholds(nullptr),
                  dmp_library_slew_lower_thresholds(nullptr),
                  dmp_library_slew_upper_thresholds(nullptr),
                  dmp_library_slew_derates(nullptr), pin_flags(nullptr),
                  pinSlew(nullptr), elmore_delay(nullptr), pinAt(nullptr), pinRat(nullptr),
                  testRelatedAT(nullptr), testRAT(nullptr), testConstraint(nullptr),
                  test_clock_ids(nullptr), clock_periods(nullptr), clock_count(0),
                  pin_clock_rise_edges(nullptr), pin_clock_fall_edges(nullptr),
                  pin_clock_slews(nullptr),
                  test_setup_uncertainties(nullptr), test_hold_uncertainties(nullptr),
                  arcDelay(nullptr),
                  timing_arc_id_map(nullptr),
                  at_prefix_pin(nullptr), at_prefix_arc(nullptr), at_prefix_attr(nullptr),
                  edge_wl(nullptr), edge_res(nullptr), includes_pin_caps(nullptr),
                  explicit_rc(false),
                  clock_period(0.0), d_allocator(nullptr), res_unit(1.0f), cap_unit(1.0f) {}
    
    ~DmpModel();

    // CUDA_DEV void compute_pi_model(int net_id, int el_rf); 
    CUDA_DEV_INLINE bool hasPinFlag(int pin_id, uint8_t flag) const {
        return pin_flags != nullptr && pin_id >= 0 && pin_id < num_pins &&
               (pin_flags[pin_id] & flag) != 0u;
    }
    CUDA_DEV int timingLibraryId(int timing_id) const;
    CUDA_DEV int pinLibraryId(int pin_id) const;
    CUDA_DEV_INLINE void loadPinThresholds(int pin_id,
                                    int attr,
                                    double& vth,
                                    double& vl,
                                    double& vh,
                                    double& slew_derate) const;
    CUDA_DEV void driverLibraryThresholds(int library_id,
                                          int attr,
                                          double& vth,
                                          double& vl,
                                          double& vh,
                                          double& slew_derate) const;
    CUDA_DEV void thresholdAdjust(int load_pin_id,
                                  int load_attr,
                                  float driver_vth,
                                  float driver_vl,
                                  float driver_vh,
                                  float driver_derate,
                                  int driver_library_id,
                                  double& wire_delay,
                                  double& load_slew) const;
    CUDA_DEV void inputPortDelaySlew(int load_pin_id,
                                     int load_attr,
                                     double source_slew,
                                     double elmore,
                                     double& wire_delay,
                                     double& load_slew) const;
    CUDA_DEV void gateCapDelaySlew(int timing_id,
                                   int input_rf,
                                   int output_rf,
                                   float input_slew,
                                   double load_cap,
                                   double& delay,
                                   double& slew);
    CUDA_DEV_INLINE DmpGateArcMeta makeGateArcMeta(int pin_id,
                                            int attr,
                                            int timing_id,
                                            int input_rf,
                                            int output_rf,
                                            float input_slew,
                                            DmpDriverThresholds& thresholds);
    CUDA_DEV_INLINE DmpGateArcMeta makeGateArcMetaForTiming(int timing_id,
                                                          int input_rf,
                                                          int to_attr,
                                                          float input_slew,
                                                          DmpDriverThresholds& thresholds);
    CUDA_DEV_NOINLINE bool findDriverParamsLocalOnePole(const DmpGateArcMeta& gate_arc_meta,
                                               const DmpDriverThresholds& thresholds,
                                               const DmpRcParams& rc,
                                               double& t0,
                                               double& dt);
    CUDA_DEV_NOINLINE bool findDriverParamsLocalPi(const DmpGateArcMeta& gate_arc_meta,
                                          const DmpDriverThresholds& thresholds,
                                          const DmpRcParams& rc,
                                          const DmpWaveCoeffs& coeffs,
                                          double A,
                                          double B,
                                          double D,
                                          bool use_c2_initial_ceff,
                                          double& t0,
                                          double& dt,
                                          double& ceff);
    CUDA_DEV_INLINE void initDriverWave(DmpDriverWave& driver_wave);
    CUDA_DEV_NOINLINE bool computeGateDriverWaveForSlot(const DmpGateArcMeta& gate_arc_meta,
                                               const DmpDriverThresholds& thresholds,
                                               int rc_slot,
                                               DmpDriverWave& driver_wave,
                                               float& gate_delay);
    CUDA_DEV_NOINLINE bool computeZeroC2DriverWave(const DmpGateArcMeta& gate_arc_meta,
                                               const DmpDriverThresholds& thresholds,
                                               const DmpRcParams& rc,
                                               DmpDriverWave& driver_wave,
                                               float& gate_delay);
    CUDA_DEV_NOINLINE bool computePiDriverWave(const DmpGateArcMeta& gate_arc_meta,
                                               const DmpDriverThresholds& thresholds,
                                               const DmpRcParams& rc,
                                               DmpDriverWave& driver_wave,
                                               float& gate_delay);
    CUDA_DEV_INLINE bool computeGateArcDriverWave(int to_attr,
                                         int input_rf,
                                         int output_rf,
                                         int to_slot,
                                         int timing_id,
                                         float input_slew,
                                         DmpDriverWave& driver_wave,
                                         float& gate_delay);
    CUDA_DEV_INLINE bool computeDrivingCellDriverWave(int pin_slot,
                                               int attr,
                                               int timing_id,
                                               int input_rf,
                                               float input_slew,
                                               DmpDriverWave& driver_wave,
                                               float& gate_delay);
    CUDA_DEV_NOINLINE void loadDelaySlewFromDriverWave(const DmpDriverWave& driver_wave,
                                            const DmpDriverThresholds& thresholds,
                                            int load_pin_id,
                                            int load_attr,
                                            double elmore,
                                            double& wire_delay,
                                            double& load_slew);
    CUDA_DEV bool isIdealClockTimingArc(int timing_id, int from_pin_id) const;
    CUDA_DEV_INLINE float clockPeriodForTest(int test_id) const;
    CUDA_DEV float idealClockEdgeTime(int timing_id, int from_pin_id) const;
    CUDA_DEV float idealClockSlew(int from_pin_id, int attr) const;
    CUDA_DEV void propagateLoadSlewDelay(int arc_id, int attr);
    CUDA_DEV bool updateAtWinner(int to_slot, float at, int arc_id, int from_attr);
    CUDA_DEV_INLINE void propagateTest(int test_id, int from_pin_id, int attr, int el, int rf, int timing_id, int to_slot);
    CUDA_DEV_INLINE void propagatePinTests(int to_pin_idx);
    CUDA_DEV void updatePinRat(int arc_id, float *from_rats);
    CUDA_DEV void propagateRAT(int arc_id, float *from_rats);
    CUDA_DEV void propagatePinBack(int level_idx, float *from_rats);

    int *edge_from;
    int *edge_to;
    int *flat_net2node_start_map;
    int *flat_net2edge_start_map;
    int *node2pin_map;
    float *pinCap;
    float *edge_wl;
    float *edge_res;
    uint8_t *includes_pin_caps;
    bool explicit_rc;
    // int num_nets;
    int num_edges;
    int num_nodes;
    float unit_to_micron;
    float rf;
    float cf;
    int *root_dist;
    int *cnts;
    int *node_order;
    int *parent_node;
    float *res_parent;
    float *node_cap;
    float *node_delay;
    float *y1;
    float *y2;
    float *y3;
    float *down_cap;
    // float *elmore_delay;
    // float *C1;
    // float *C2;
    // float *r_pi;
    DmpModel(GPUTimer* timer);
    void allocate_timing_scratch();
    void release_rc_transient();
    void release_after_timing();
    void initialize_rc(
               const std::vector<int>& host_edge_from,
               const std::vector<int>& host_edge_to,
               const std::vector<int>& host_flat_net2node_start_map,
               const std::vector<int>& host_flat_net2edge_start_map,
               const std::vector<int>& host_node2pin_map,
               const std::vector<float>& host_edge_wl,
               float *pinCap,
               int num_nets,
               int num_nodes,
               int num_edges,
               float unit_to_micron,
               float rf,
               float cf);
    void initialize_rc_explicit(
               const std::vector<int>& host_edge_from,
               const std::vector<int>& host_edge_to,
               const std::vector<int>& host_flat_net2node_start_map,
               const std::vector<int>& host_flat_net2edge_start_map,
               const std::vector<int>& host_node2pin_map,
               const std::vector<float>& host_edge_res,
               const std::vector<float>& host_node_cap,
               const std::vector<uint8_t>& host_includes_pin_caps,
               float *pinCap,
               int num_nets,
               int num_nodes,
               int num_edges);
    CUDA_DEV void calc_dmp_rc();
    CUDA_DEV void propagate_dmp_rc();

    bool debug_on = false;
};


// struct dmp_rc{
//     int *edge_from;
//     int *edge_to;
//     int *flat_net2node_start_map;
//     int *flat_net2edge_start_map;
//     int *node2pin_map;
//     float *pinCap;
//     float *edge_wl;
//     int num_nets;
//     int num_edges;
//     int num_nodes;
//     float unit_to_micron;
//     float rf;
//     float cf;
//     int *root_dist;
//     int *cnts;
//     int *node_order;
//     int *parent_node;
//     float *res_parent;
//     float *node_cap;
//     float *node_delay;
//     float *y1;
//     float *y2;
//     float *y3;
//     float *down_cap;
//     float *elmore_delay;
//     float *C1;
//     float *C2;
//     float *r_pi;
//     dmp_rc(std::vector<int> host_edge_from,
//                std::vector<int> host_edge_to,
//                std::vector<int> host_flat_net2node_start_map,
//                std::vector<int> host_flat_net2edge_start_map,
//                std::vector<int> host_node2pin_map,
//                std::vector<float> host_edge_wl,
//                float *pinCap,
//                int num_nets,
//                int num_edges,
//                int num_nodes,
//                float unit_to_micron,
//                float rf,
//                float cf);
//     CUDA_DEV void calc_dmp_rc();
//     CUDA_DEV void propagate_dmp_rc();
// };

} // namespace gt
