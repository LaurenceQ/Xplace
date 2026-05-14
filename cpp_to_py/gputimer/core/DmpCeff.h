#pragma once

#include "GPUTimer.h"
// #include "gputiming.h"
// #include "utils.cuh"
#include <cstdint>
#include <vector>
#ifdef __CUDACC__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif



namespace gt {
class GPUTimer;
class GPULutAllocator;

enum DmpAlgKind { DMP_ALG_CAP = 0, DMP_ALG_ZERO_C2 = 1, DMP_ALG_PI = 2 };

struct dmp_model {
    const char **pin_names;
    const char **net_names;
    int num_pins, num_nets, num_arcs, num_tests;
    const int *flat_net2pin_start_map, *flat_net2pin_map, *pin2net_map;
    double *k0_, *k1_, *k2_, *k3_, *k4_;
    double *p1_, *p2_, *p3_;
    double *z1_;
    double *A_, *B_, *D_;
    double *C1, *C2, *r_pi;
    double *rd_, *t0, *dt;
    float *ceff;
    float *timer_ceff;
    int dmp_pin_slot_count, dmp_slot_capacity, dmp_work_slot_capacity;
    int dmp_arc_delay_winner_stride;
    bool owns_allocations;
    int *dmp_alg_kind;
    unsigned long long *pin_at_winner;
    unsigned long long *pin_slew_winner;
    unsigned long long *arc_delay_winner;
    const float *dmp_input_thresholds;
    const float *dmp_output_thresholds;
    const float *dmp_slew_lower_thresholds;
    const float *dmp_slew_upper_thresholds;
    const float *dmp_slew_derates;
    const float *dmp_timing_output_thresholds;
    const float *dmp_timing_slew_lower_thresholds;
    const float *dmp_timing_slew_upper_thresholds;
    const float *dmp_timing_slew_derates;
    const float *dmp_pin_input_thresholds;
    const float *dmp_pin_slew_lower_thresholds;
    const float *dmp_pin_slew_upper_thresholds;
    const float *dmp_pin_slew_derates;
    int *pin_is_primary_input;
    int *pin_is_clk;
    bool ideal_clock;
    float *slot_vth;
    float *slot_vl;
    float *slot_vh;
    float *slot_slew_derate;
    int *pin_ids, *arc_ids;
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
    int *arc_types;
    int *arc_id2test_id;
    float *pinSlew;
    float *elmore_delay; // elmore delay
    float *pinAt;
    float *pinRat;
    float *testRelatedAT;
    float *testRAT;
    float *testConstraint;
    const float *test_clock_periods;
    const float *test_setup_uncertainties;
    const float *test_hold_uncertainties;
    const float *pin_clock_periods;
    const float *pin_clock_rise_edges;
    const float *pin_clock_fall_edges;
    const float *pin_clock_slews;
    float *arcDelay;
    double *vo_delay_;
    double *vo_slew_;
    double *driving_cell_extra_delay_;
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
    dmp_model() : num_pins(0), num_nets(0), num_arcs(0), num_tests(0), pin_names(nullptr),
                  flat_net2pin_start_map(nullptr), flat_net2pin_map(nullptr), pin2net_map(nullptr),
                  C1(nullptr), C2(nullptr), r_pi(nullptr), ceff(nullptr), timer_ceff(nullptr),
                  dmp_pin_slot_count(0), dmp_slot_capacity(0), dmp_work_slot_capacity(0),
                  dmp_arc_delay_winner_stride(NUM_ATTR),
                  owns_allocations(false),
                  level_list(nullptr),
                  pin_forward_arc_list_end(nullptr), pin_forward_arc_list(nullptr),
                  timing_arc_to_pin_id(nullptr),
                  pin_backward_arc_list_end(nullptr), pin_backward_arc_list(nullptr),
                  timing_arc_from_pin_id(nullptr),
                  arc_types(nullptr), arc_id2test_id(nullptr),
                  dmp_alg_kind(nullptr),
                  pin_at_winner(nullptr),
                  pin_slew_winner(nullptr), arc_delay_winner(nullptr),
                  dmp_input_thresholds(nullptr), dmp_output_thresholds(nullptr),
                  dmp_slew_lower_thresholds(nullptr), dmp_slew_upper_thresholds(nullptr),
                  dmp_slew_derates(nullptr), dmp_timing_output_thresholds(nullptr),
                  dmp_timing_slew_lower_thresholds(nullptr),
                  dmp_timing_slew_upper_thresholds(nullptr),
                  dmp_timing_slew_derates(nullptr),
                  dmp_pin_input_thresholds(nullptr),
                  dmp_pin_slew_lower_thresholds(nullptr),
                  dmp_pin_slew_upper_thresholds(nullptr),
                  dmp_pin_slew_derates(nullptr), pin_is_primary_input(nullptr),
                  pin_is_clk(nullptr), ideal_clock(false),
                  slot_vth(nullptr), slot_vl(nullptr),
                  slot_vh(nullptr), slot_slew_derate(nullptr),
                  pinSlew(nullptr), elmore_delay(nullptr), pinAt(nullptr), pinRat(nullptr),
                  testRelatedAT(nullptr), testRAT(nullptr), testConstraint(nullptr),
                  test_clock_periods(nullptr), test_setup_uncertainties(nullptr),
                  test_hold_uncertainties(nullptr), pin_clock_periods(nullptr),
                  pin_clock_rise_edges(nullptr), pin_clock_fall_edges(nullptr),
                  pin_clock_slews(nullptr),
                  arcDelay(nullptr), vo_delay_(nullptr), vo_slew_(nullptr),
                  driving_cell_extra_delay_(nullptr),
                  timing_arc_id_map(nullptr),
                  at_prefix_pin(nullptr), at_prefix_arc(nullptr), at_prefix_attr(nullptr),
                  edge_wl(nullptr), edge_res(nullptr), includes_pin_caps(nullptr),
                  explicit_rc(false),
                  clock_period(0.0), d_allocator(nullptr), res_unit(1.0f), cap_unit(1.0f) {}
    
    ~dmp_model();

    // CUDA_DEV void compute_pi_model(int net_id, int el_rf); 
    CUDA_DEV double voCrossingUpperBound(int net_idx);
    CUDA_DEV double y0(double t, double rd, double cl);
    CUDA_DEV double y(double t, double t0, double dt, double rd, double cl);
    CUDA_DEV double y0dt(double t, double rd, double cl);
    CUDA_DEV double y0dcl(double t, double rd, double cl);
    CUDA_DEV void dy(double t, double t0, double dt, double rd, double cl,
                     double &dydt0, double &dyddt, double &dydcl);
    CUDA_DEV void Vl0(int net_idx, double t, double &vl, double &dvl_dt);
    CUDA_DEV void Vl0Explicit(int net_idx, double elmore, double t, double &vl, double &dvl_dt);
    CUDA_DEV void V0(int net_idx, double t, double &vo, double &dvo_dt);
    CUDA_DEV void Vl(double t, double &vl, double &dvl_dt);
    CUDA_DEV void VlExplicit(int net_idx, double elmore, double t, double &vl, double &dvl_dt);
    CUDA_DEV void Vo(double t, double &vo, double &dvo_dt);
    CUDA_DEV void vl_func(double vth, double t, double &y, double &dy);
    CUDA_DEV void vlFuncExplicit(int net_idx, double elmore, double vth, double t, double &y, double &dy);
    CUDA_DEV void vo_func(double vth, double t, double &y, double &dy);
    CUDA_DEV double findRoot_vo(double vth, double x1, double x2);
    CUDA_DEV double findRoot_vl(double vth, double x1, double x2);
    CUDA_DEV double findRootVlExplicit(int net_idx, double elmore, double vth, double x1, double x2);
    CUDA_DEV double findVlCrossing(double vth, double t_lower, double t_upper);
    CUDA_DEV double findVlCrossingExplicit(int net_idx, double elmore, double vth, double t_lower, double t_upper);
    CUDA_DEV double findVoCrossing(double vth, double t_lower, double t_upper);
    CUDA_DEV void propagateLoadSlewDelay();
    CUDA_DEV void gateCapDelaySlew(double lc, double &delay, double &slew);
    CUDA_DEV void gateDelays(double ceff, double &t_vth, double &t_vl, double &slew);
    CUDA_DEV void gateModelRd(int net_idx, double d1, double s1);
    CUDA_DEV int selectDmpAlg(int net_idx);
    CUDA_DEV bool init_dmp_factors(int net_idx);
    CUDA_DEV bool init_zero_c2_factors(int net_idx);
    CUDA_DEV double ipiIceff(int net_idx, double dt, double ceff_time, double ceff);
    CUDA_DEV bool evalDmpEqns(double *x_, double *fvec_, double (*fjac_)[3], int size);
    CUDA_DEV bool newtonRaphson(int max_iter, int size, double *x, double (*fjac)[3], double *fvec, int *index, double *p, double *scale);
    CUDA_DEV bool findDriverParams(double delay, double slew, double initial_ceff);
    CUDA_DEV bool findDriverParamsOnePole(double delay, double slew, double fixed_ceff);
    CUDA_DEV bool findDriverParamsScalar(double delay, double slew, double initial_ceff);
    CUDA_DEV bool findDriverParamsOnePoleScalar(double delay, double slew, double fixed_ceff);
    CUDA_DEV void findDriverDelaySlew(int net_idx, double &delay, double &slew);
    CUDA_DEV bool updateLoadWinner(int net_arc_id, int load_attr, float wire_delay, float load_slew);
    CUDA_DEV int arcDelayWinnerSlot(int arc_id, int attr) const { return arc_id * NUM_ATTR + attr; }
    CUDA_DEV bool updateAtWinner(int to_slot, float at, bool pick_max, int from_pin_id, int arc_id, int from_attr);
    CUDA_DEV void propagateTest();
    CUDA_DEV void propagatePinTests(int to_pin_idx);
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
    dmp_model(GPUTimer* timer);
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
