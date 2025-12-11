#include "GPUTimer.h"
// #include "gputiming.h"
// #include "utils.cuh"
#include <vector>
#ifdef __CUDACC__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif



namespace gt {
class GPUTimer;
class GPULutAllocator;

struct dmp_model {
    const char **pin_names;
    const char **net_names;
    int num_pins, num_nets;
    const int *flat_net2pin_start_map, *flat_net2pin_map, *pin2net_map;
    float *k0_, *k1_, *k2_, *k3_, *k4_;
    float *p1_, *p2_, *p3_;
    float *z1_;
    float *A_, *B_, *D_;
    float *C1, *C2, *r_pi;
    float *rd_, *t0, *dt, *ceff;
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
    float *arcDelay;
    float *vo_delay_;
    float *vo_slew_;
    int *timing_arc_id_map;
    float clock_period;
    GPULutAllocator *d_allocator;
    const double vth_time_tol = .01;// vars here are usally the case
    const double x_tol = .01;
    const double slew_derate_ = 1.0; 
    const double vth_ = 0.5;
    const double vl_ = 0.1;
    const double vh_ = 0.9; 
    const int MAX_ITER = 20;
    dmp_model() : num_pins(0), num_nets(0), pin_names(nullptr),
                  flat_net2pin_start_map(nullptr), flat_net2pin_map(nullptr), pin2net_map(nullptr),
                  C1(nullptr), C2(nullptr), r_pi(nullptr), level_list(nullptr),
                  pin_forward_arc_list_end(nullptr), pin_forward_arc_list(nullptr),
                  timing_arc_to_pin_id(nullptr),
                  pin_backward_arc_list_end(nullptr), pin_backward_arc_list(nullptr),
                  timing_arc_from_pin_id(nullptr),
                  arc_types(nullptr), arc_id2test_id(nullptr),
                  pinSlew(nullptr), elmore_delay(nullptr), pinAt(nullptr), pinRat(nullptr),
                  testRelatedAT(nullptr), testRAT(nullptr), testConstraint(nullptr),
                  arcDelay(nullptr), vo_delay_(nullptr),
                  timing_arc_id_map(nullptr),
                  at_prefix_pin(nullptr), at_prefix_arc(nullptr), at_prefix_attr(nullptr),
                  clock_period(0.0), d_allocator(nullptr) {}
    
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
    CUDA_DEV void V0(int net_idx, double t, double &vo, double &dvo_dt);
    CUDA_DEV void Vl(double t, double &vl, double &dvl_dt);
    CUDA_DEV void Vo(double t, double &vo, double &dvo_dt);
    CUDA_DEV void vl_func(double vth, double t, double &y, double &dy);
    CUDA_DEV void vo_func(double vth, double t, double &y, double &dy);
    CUDA_DEV double findRoot_vo(double vth, double x1, double x2);
    CUDA_DEV double findRoot_vl(double vth, double x1, double x2);
    CUDA_DEV double findVlCrossing(double vth, double t_lower, double t_upper);
    CUDA_DEV double findVoCrossing(double vth, double t_lower, double t_upper);
    CUDA_DEV void propagateLoadSlewDelay();
    CUDA_DEV void gateCapDelaySlew(double lc, double &delay, double &slew);
    CUDA_DEV void gateDelays(double ceff, double &t_vth, double &t_vl, double &slew);
    CUDA_DEV void gateModelRd(int net_idx, double d1, double s1);
    CUDA_DEV void init_dmp_factors(int net_idx);
    CUDA_DEV double ipiIceff(int net_idx, double dt, double ceff_time, double ceff);
    CUDA_DEV bool evalDmpEqns(double *x_, double *fvec_, double (*fjac_)[3]);
    CUDA_DEV bool newtonRaphson(int max_iter, int size, double *x, double (*fjac)[3], double *fvec, int *index, double *p, double *scale);
    CUDA_DEV void findDriverParams(double delay, double slew, bool &error_flag);
    CUDA_DEV void findDriverDelaySlew(int net_idx, double &delay, double &slew);
    CUDA_DEV void propagateGateSlewDelay();
    CUDA_DEV void propagateSlewDelay();
    CUDA_DEV void propagateAT();
    CUDA_DEV void propagateTest();
    CUDA_DEV void updatePinRat(int arc_id, float *from_rats);
    CUDA_DEV void propagateRAT(int arc_id, float *from_rats);
    CUDA_DEV void propagatePin(int to_pin_idx);
    CUDA_DEV void propagatePinBack(int level_idx, float *from_rats);

    int *edge_from;
    int *edge_to;
    int *flat_net2node_start_map;
    int *flat_net2edge_start_map;
    int *node2pin_map;
    float *pinCap;
    float *edge_wl;
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
    void initialize_rc(
               std::vector<int> host_edge_from,
               std::vector<int> host_edge_to,
               std::vector<int> host_flat_net2node_start_map,
               std::vector<int> host_flat_net2edge_start_map,
               std::vector<int> host_node2pin_map,
               std::vector<float> host_edge_wl,
               float *pinCap,
               int num_nets,
               int num_edges,
               int num_nodes,
               float unit_to_micron,
               float rf,
               float cf);
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