#pragma once

#include <torch/extension.h>
#include "common/common.h"
#include "common/lib/spef/parser-spef.hpp"
#include "gputimer/base.h"
#include <cstdint>
#include <string>

using std::tuple;
using std::vector;
using std::shared_ptr;

namespace gt {

class TimingArc;
class TimingTorchRawDB;
class GTDatabase;
class GPULutAllocator;
class GPUPowerLutAllocator;
class DmpModel;

struct GpuPowerExprOpHost {
    uint8_t op = 0;   // 0=pin, 1=const0, 2=const1, 3=not, 4=and, 5=or, 6=xor
    int arg = -1;     // physical pin id for op=pin
};

struct GpuPowerSeqHost {
    int data_expr_id = -1;
    int clk_expr_id = -1;
    int q_pin = -1;
    int qn_pin = -1;
    uint8_t is_latch = 0;
};

struct GpuPowerInternalHost {
    int internal_power_id = -1;
    int node_id = -1;
    int to_pin = -1;
    int from_pin = -1;
    int kind = 0;          // 0=input internal_power, 1=output internal_power
    int duty_mode = 0;     // 0=const1, 1=expr duty, 2=diff duty, 3=const0.5, 4=const0
    int duty_expr_id = -1;
    int duty_pin = -1;
    int denom_group = -1;
    int positive_unate = 1;
    float energy_unit = 1.0f;  // Liberty internal_power energy unit in joules.
};

struct GpuPowerLeakageRowHost {
    int node_id = -1;
    int group_id = -1;
    int leakage_power_id = -1;
    int when_expr_id = -1;
    float leakage = 0.0f;  // Watts
};

struct GpuPowerLeakageGroupHost {
    int node_id = -1;
    float cell_leakage = 0.0f;  // Watts
};

struct HostRcGraph {
    std::vector<int> edge_from;
    std::vector<int> edge_to;
    std::vector<float> edge_res;
    std::vector<float> node_cap;
    std::vector<int> net2node_start;
    std::vector<int> net2edge_start;
    std::vector<int> node2pin;
    std::vector<std::string> node_names;
    std::vector<uint8_t> includes_pin_caps;
    int skipped_loop_edges = 0;
    int repaired_edges = 0;
    int num_nodes = 0;
    int num_edges = 0;
};

class GPUTimer {
public:
    GPULutAllocator *allocator = nullptr;
    GPULutAllocator *d_allocator = nullptr;
    GPUPowerLutAllocator *power_allocator = nullptr;
    GPUPowerLutAllocator *d_power_allocator = nullptr;
    GTDatabase& gtdb;
    TimingTorchRawDB& timing_raw_db;
    shared_ptr<GTDatabase> gtdb_holder;
    shared_ptr<TimingTorchRawDB> timing_raw_db_holder;
    GPUTimer(shared_ptr<GTDatabase> gtdb_, shared_ptr<TimingTorchRawDB> timing_raw_db_);
    ~GPUTimer();
    spef::Spef spef;
    void read_spef(const std::string& file);

    // === functions ===
    void initialize();
    void levelize();
    void levelize_power(const uint8_t* d_is_seq_output_pin);
    void update_rc_timing(torch::Tensor node_lpos, bool record = false, bool load = false, bool conpensation = false);
    void update_rc_timing_flute(torch::Tensor node_lpos, bool record = false);
    void update_rc_timing_spef();
    HostRcGraph build_spef_rc();
    HostRcGraph build_openroad_gr_rc(const std::string& file);
    HostRcGraph build_openroad_route_segments_rc(const std::string& file);
    void debug_dump_spef_rc_net(const std::string& net_name);
    void debug_dump_openroad_gr_rc_net(const std::string& file, const std::string& net_name);
    void debug_dump_openroad_route_segments_rc_net(const std::string& file,
                                                   const std::string& net_name);
    void debug_compare_openroad_route_segments_rc(const std::string& gr_rc_file,
                                                  const std::string& route_segments_file,
                                                  int top_n);
    void update_states();
    void update_timing();
    void update_endpoints();
    
    float report_wns(int el);
    float report_tns_elw(int el);
    tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> report_wns_and_tns();
    torch::Tensor report_pin_slack();
    torch::Tensor endpoints_index();
    torch::Tensor report_endpoint_slack();
    torch::Tensor report_endpoint_pin_slack();
    torch::Tensor report_pin_at();
    torch::Tensor report_pin_rat();
    torch::Tensor report_pin_slew();
    torch::Tensor report_pin_load();
    torch::Tensor report_delay();
    tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> report_power_liberty_inventory();
    int64_t report_power_seq_inventory();
    torch::Tensor report_power_internal_lut_cuda_probe();
    torch::Tensor report_power_activity_cpu();
    torch::Tensor report_power_activity_cuda();
    tuple<torch::Tensor, torch::Tensor> report_power_switching_cuda();
    torch::Tensor report_power_internal_cuda();
    tuple<torch::Tensor, torch::Tensor, torch::Tensor> report_power_internal_arcs_cuda();
    torch::Tensor report_power_leakage_cuda();
    tuple<torch::Tensor, torch::Tensor, torch::Tensor> report_power_leakage_rows_cuda();
    tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> report_power_total_cuda();
    torch::Tensor compute_power_activity_cuda(torch::Tensor* inst_switching_cpu,
                                              torch::Tensor* pin_switching_cpu,
                                              torch::Tensor* inst_internal_cpu = nullptr,
                                              torch::Tensor* internal_row_power_cpu = nullptr,
                                              torch::Tensor* internal_row_meta_cpu = nullptr,
                                              torch::Tensor* inst_leakage_cpu = nullptr,
                                              torch::Tensor* leakage_row_power_cpu = nullptr,
                                              torch::Tensor* leakage_row_meta_cpu = nullptr);
    void debug_dump_endpoint_tests(const std::string& outfile,
                                   const vector<std::string>& endpoint_pin_names);

    tuple<vector<int64_t>, vector<float>, vector<float>> report_path(int ep_idx = -1, int el = -1, int rf = -1, bool verbose = false);
    vector<vector<int64_t>> report_K_path(int K, int el = -1, int rf = -1, bool verbose = false);
    tuple<torch::Tensor, torch::Tensor> report_criticality(int K, bool verbose = false, bool deterministic = true);
    tuple<torch::Tensor, torch::Tensor> report_criticality_threshold(float thrs, bool verbose = false, bool deterministic = true);

    /**
     * Dump the full timing graph to a JSONL file.
     * Each line is one JSON record — node, net_arc, or cell_arc.
     * Pin IDs in the output correspond 1-to-1 to gtdb.pin_names indices.
     * See dump.cpp for the full feature / label schema.
     */
    void dump_timing_graph(const std::string& outfile);

    /**
     * Load TimingPredict inference results from CSV and update GPU timing arrays.
     * Updates arcDelay with ML-predicted net delays and cell edge delays.
     * Input file format: .infer CSV with node predictions (net delays) and edge predictions (cell delays).
     * All times in file assumed to be in nanoseconds.
     */
    void read_infer(const std::string& infile);

    /**
     * Load OpenROAD ground-truth .infer CSV and update GPU timing arrays.
     * Node format: node_id,pin_name,gt_at_er,gt_at_ef,gt_at_lr,gt_at_lf,
     *              slew_er,slew_ef,slew_lr,slew_lf,net_delay_er,ef,lr,lf
     * Edge format: edge_id,from_opr_node,to_opr_node,cell_delay_er,ef,lr,lf
     * OPR node IDs are mapped to GPUTimer pin IDs via pin_name lookup.
     * GT AT values (ns) are stored in host_pinGT_AT for later R² comparison.
     */
    void read_opr_gt_infer(const std::string& infile);

    /** Return ground-truth AT values loaded by read_opr_gt_infer(). [num_pins, 4] internal units, NaN=missing. */
    torch::Tensor report_pin_gt_at();

    /** Host-side GT AT values populated by read_opr_gt_infer(). [num_pins*4], internal units. */
    std::vector<float> host_pinGT_AT;

    /**
     * Propagate ML-predicted delays through timing graph.
     * Performs forward pass (AT), backward pass (RAT), and slack computation.
     * Must call read_infer() first to load arcDelay with ML predictions.
     */
    void propagate_infer_timing();

    /** Shared GPU update helper: D2H → update slew/net-delay/cell-delay → H2D.
     *  Called by both read_infer() and read_opr_gt_infer() after parsing. */
    void apply_infer_data(
        const std::vector<std::pair<int, std::array<float, 4>>>& slews,
        const std::vector<std::pair<int, std::array<float, 4>>>& net_delays,
        const std::vector<std::tuple<int, int, std::array<float, 4>>>& cell_delays,
        float time_to_internal);

public:
    float time_unit() const;

public:
    int num_pins, num_arcs, num_timings, num_tests, num_POs, total_num_fanouts;
    int num_endpoint_pins;
    float *pinSlew, *pinLoad, *pinRAT, *pinAT;
    float *pinImpulse, *pinRootDelay, *pinRootRes;
    float *arcDelay, *arcSlew;
    float *pinCap, *pinWireCap;
    float *testRelatedAT, *testConstraint, *testRAT;
    float *test_clock_periods, *test_setup_uncertainties, *test_hold_uncertainties;
    float *pin_clock_periods;
    float *pin_clock_rise_edges, *pin_clock_fall_edges;
    float *pin_clock_slews;

    float *__pinSlew__, *__pinLoad__, *__pinRAT__, *__pinAT__;
    float *pinImpulse_ref, *pinLoad_ref, *pinRootDelay_ref;
    float *pinLoad_ratio, *pinRootDelay_ratio;

    int* pin_num_fanin;
    index_type *pin_fanout_list_end, *pin_fanout_list;
    index_type *pin_forward_arc_list_end, *pin_forward_arc_list;
    index_type *pin_backward_arc_list_end, *pin_backward_arc_list;
    index_type *timing_arc_from_pin_id, *timing_arc_to_pin_id;
    int *arc_types, *timing_arc_id_map, *arc_id2test_id;
    int* test_id2_arc_id;
    int* test_id2_endpoint_id;
    int* primary_output2_endpoint_id;

    index_type* primary_outputs;
    TimingArc* liberty_timing_arcs;
    index_type *level_list_end = nullptr, *level_list = nullptr;
    vector<int> level_list_end_cpu;
    vector<int> pin_level_cpu;
    index_type *power_level_list_end = nullptr, *power_level_list = nullptr;
    vector<int> power_level_list_end_cpu;
    vector<int> power_pin_level_cpu;
    vector<int> power_level_root_pins_cpu;
    int* net_is_clock;
    int* pin_is_clk;  // GPU array: 1 if register clock pin, 0 otherwise

    float clock_period;
    bool ideal_clock = false;

    void set_ideal_clock(bool v) { ideal_clock = v; }

public:
    float* x;
    float* y;
    const float* init_x;
    const float* init_y;
    const float* node_size_x;
    const float* node_size_y;

    const float* pin_offset_x;
    const float* pin_offset_y;
    index_type *at_prefix_pin;
    index_type *at_prefix_arc;
    index_type *at_prefix_attr;

    const int* flat_node2pin_start_map;
    const int* flat_node2pin_map;
    const int* pin2node_map;

    const int* flat_net2pin_start_map;
    const int* flat_net2pin_map;
    const int* pin2net_map;
    const bool* net_mask;

    const float* dmp_input_thresholds;
    const float* dmp_output_thresholds;
    const float* dmp_slew_lower_thresholds;
    const float* dmp_slew_upper_thresholds;
    const float* dmp_slew_derates;
    const int* dmp_timing_library_ids;
    const int* dmp_pin_library_ids;
    const float* dmp_library_input_thresholds;
    const float* dmp_library_output_thresholds;
    const float* dmp_library_slew_lower_thresholds;
    const float* dmp_library_slew_upper_thresholds;
    const float* dmp_library_slew_derates;

    /* row info */
    int num_nets;
    int num_movable_nodes;
    int num_nodes;

    int num_threads;

    float wire_resistance_per_micron;
    float wire_capacitance_per_micron;
    int microns;
    float scale_factor;
    float res_unit;
    float cap_unit;

    torch::Tensor pin_slacks;
    torch::Tensor endpoint_slacks;
    torch::Tensor endpoint_pin_slacks;
//DMP model
public:

    void initialize_dmp_model();
    void initialize_dmp_rc(
                  const std::vector<int>& host_edge_from,
                  const std::vector<int>& host_edge_to,
                  const std::vector<int>& host_flat_net2node_start_map,
                  const std::vector<int>& host_flat_net2edge_start_map,
                  const std::vector<int>& host_node2pin_map,
                  const std::vector<float>& host_edge_wl,
                  int num_nets,
                  int num_nodes,
                  int num_edges,
                  float unit_to_micron,
                  float rf,
                  float cf);
    void initialize_dmp_rc_explicit(
                  const std::vector<int>& host_edge_from,
                  const std::vector<int>& host_edge_to,
                  const std::vector<int>& host_flat_net2node_start_map,
                  const std::vector<int>& host_flat_net2edge_start_map,
                  const std::vector<int>& host_node2pin_map,
                  std::vector<float>& host_edge_res,
                  const std::vector<float>& host_node_cap,
                  const std::vector<uint8_t>& host_includes_pin_caps,
                  int num_nets,
                  int num_nodes,
                  int num_edges);
    void update_timing_dmp();
    void print_pin_id_name();
    void get_units();
    void update_rc_timing_flute_dmp(torch::Tensor node_lpos, bool record = false);
    void init_dmp_rc_spef();
    void init_dmp_rc_gr(const std::string& file);
    void init_dmp_rc_route_segments(const std::string& file);
    void debug_dump_dmp_rc_net(const std::string& net_name);
    void print_pinLoad();
    DmpModel* dmp_db = nullptr;
    DmpModel* h_dmp_db = nullptr;
    void* dmp_forward_schedule = nullptr;
    bool dmp_debug_on = false;
    void set_dmp_debug_flag(bool flag){dmp_debug_on = flag;}

};

}  // namespace gt
