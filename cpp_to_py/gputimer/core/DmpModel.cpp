// #include "DmpModel.h"
#include "GPUTimer.h"
#include "common/XplaceLog.h"
#include "common/utils/utils.h"
#include "common/db/Database.h"
#include "gputimer/db/GTDatabase.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <flute.hpp>
using namespace Flute;

namespace gt {

static bool dmp_cpp_env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

static bool dmp_progress_enabled()
{
    return dmp_cpp_env_enabled("XPLACE_TIMER_VERBOSE") ||
           dmp_cpp_env_enabled("DMP_PROGRESS") ||
           dmp_cpp_env_enabled("DMP_DEBUG_TIMING") ||
           dmp_cpp_env_enabled("DMP_RC_PROFILE");
}

#define DMP_PROGRESS_PRINT(...)           \
    do {                                  \
        if (dmp_progress_enabled()) {     \
            XPLACE_LOGF("DMP_PROGRESS", __VA_ARGS__); \
        }                                 \
    } while (false)

template <typename T>
static void release_vector_storage(std::vector<T>& values)
{
    std::vector<T>().swap(values);
}

void HostRcGraph::release_storage()
{
    release_vector_storage(edge_from);
    release_vector_storage(edge_to);
    release_vector_storage(edge_res);
    release_vector_storage(node_cap);
    release_vector_storage(net2node_start);
    release_vector_storage(net2edge_start);
    release_vector_storage(node2pin);
    release_vector_storage(node_names);
    release_vector_storage(includes_pin_caps);
}

void compute_pi_model_cuda(DmpModel* dmp_db, int num_nets);
// void GPUTimer::compute_pi_model(){
//     assert(dmp_db != nullptr);
//     compute_pi_model_cuda(dmp_db, num_nets);
// }

void update_timing_dmp_cuda(GPUTimer* timer);
void print_pinLoad_cuda(DmpModel* dmp_db, vector<int> level_list_end_cpu, vector<string> pin_names);
static void apply_dmp_driving_cell_source_slew(GPUTimer& timer);
void GPUTimer::update_timing_dmp(){
    DMP_PROGRESS_PRINT("Update timing DMP started.");
    apply_dmp_driving_cell_source_slew(*this);
    update_timing_dmp_cuda(this);
}
void GPUTimer::print_pinLoad(){
    print_pinLoad_cuda(dmp_db, level_list_end_cpu, gtdb.pin_names);
}

void GPUTimer::print_pin_id_name(){
    for (size_t pin_id = 0; pin_id < gtdb.pin_names.size(); ++pin_id) {
        XPLACE_LOGF("DMP_PIN_ID", "%zu %s", pin_id, gtdb.pin_names[pin_id].c_str());
    }
}


void calc_res_cap_dmp(DmpModel* dmp_db, int num_nets);
void propagate_rc_tree_dmp(DmpModel* dmp_db, int num_nets);
void dmp_prepare_timing_after_rc(DmpModel* h_dmp_db, DmpModel* dmp_db);
void apply_dmp_driving_cell_source_slew_cuda(DmpModel* dmp_db,
                                             const std::vector<int>& pin_ids,
                                             const std::vector<int>& timing_ids,
                                             const std::vector<int>& input_rfs,
                                             const std::vector<float>& input_slews);
void debug_dump_dmp_rc_net_cuda(DmpModel* h_dmp_db,
                                int net_id,
                                const std::vector<std::string>& net_names,
                                const std::vector<std::string>& pin_names);

static void apply_dmp_driving_cell_source_slew(GPUTimer& timer) {
    if (timer.gtdb.driving_cell_sources.empty()) {
        return;
    }

    std::vector<int> pin_ids;
    std::vector<int> timing_ids;
    std::vector<int> input_rfs;
    std::vector<float> input_slews;
    pin_ids.reserve(timer.gtdb.driving_cell_sources.size());
    timing_ids.reserve(timer.gtdb.driving_cell_sources.size() * NUM_ATTR);
    input_rfs.reserve(timer.gtdb.driving_cell_sources.size() * NUM_ATTR);
    input_slews.reserve(timer.gtdb.driving_cell_sources.size() * NUM_ATTR);

    int finite_lanes = 0;
    for (const auto& source : timer.gtdb.driving_cell_sources) {
        pin_ids.push_back(source.pin_id);
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            timing_ids.push_back(source.timing_ids[attr]);
            input_rfs.push_back(source.input_rfs[attr]);
            input_slews.push_back(source.input_slews[attr]);
            if (source.timing_ids[attr] >= 0 && source.input_rfs[attr] >= 0) {
                ++finite_lanes;
            }
        }
    }

    DMP_PROGRESS_PRINT("[DMP DRIVING CELL] prepared sources=%zu lanes=%d",
                       timer.gtdb.driving_cell_sources.size(),
                       finite_lanes);
    apply_dmp_driving_cell_source_slew_cuda(timer.dmp_db, pin_ids, timing_ids, input_rfs, input_slews);
}
void GPUTimer::get_units(){
    XPLACE_LOGF("DMP_UNITS", "time_unit=%E", time_unit());
    XPLACE_LOGF("DMP_UNITS", "res_unit=%E", res_unit);
    XPLACE_LOGF("DMP_UNITS", "cap_unit=%E", cap_unit);
    XPLACE_LOGF("DMP_UNITS", "micron=%d", microns);
    XPLACE_LOGF("DMP_UNITS", "scale_factor=%E", scale_factor);
}
auto& retrieve_pins_from_pos_dmp(std::map<utils::PointT<int>, std::set<int>>& pos2pins_map, const utils::PointT<int>& point, int& index) {
    if (pos2pins_map.find(point) != pos2pins_map.end()) return pos2pins_map[point];
    pos2pins_map.emplace(point, std::set<int>{index++});
    return pos2pins_map[point];
}
tuple<vector<int>, vector<int>, vector<float>, vector<int>, vector<int>, vector<int>, int, int> FluteRCTreeDMP(TimingTorchRawDB& timing_raw_db,
                                                                                                            float rf,
                                                                                                            float cf,
                                                                                                            const vector<string>& net_names,
                                                                                                            const vector<string>& pin_names,
                                                                                                            bool dmp_debug_on) {
    torch::Tensor flat_node2pin_start_map_at = timing_raw_db.flat_node2pin_start_map.clone().cpu().contiguous();
    torch::Tensor flat_node2pin_map_at = timing_raw_db.flat_node2pin_map.clone().cpu().contiguous();
    torch::Tensor pin2node_map_at = timing_raw_db.pin2node_map.clone().cpu().contiguous();
    torch::Tensor flat_net2pin_start_map_at = timing_raw_db.flat_net2pin_start_map.clone().cpu().contiguous();
    torch::Tensor flat_net2pin_map_at = timing_raw_db.flat_net2pin_map.clone().cpu().contiguous();
    torch::Tensor pin2net_map_at = timing_raw_db.pin2net_map.clone().cpu().contiguous();
    torch::Tensor x_at = timing_raw_db.x.clone().cpu().contiguous();
    torch::Tensor y_at = timing_raw_db.y.clone().cpu().contiguous();
    torch::Tensor pin_offset_x_at = timing_raw_db.pin_offset_x.clone().cpu().contiguous();
    torch::Tensor pin_offset_y_at = timing_raw_db.pin_offset_y.clone().cpu().contiguous();

    const int* flat_node2pin_start_map = flat_node2pin_start_map_at.data_ptr<int>();
    const int* flat_node2pin_map = flat_node2pin_map_at.data_ptr<int>();
    const int* pin2node_map = pin2node_map_at.data_ptr<int>();
    const int* flat_net2pin_start_map = flat_net2pin_start_map_at.data_ptr<int>();
    const int* flat_net2pin_map = flat_net2pin_map_at.data_ptr<int>();
    const int* pin2net_map = pin2net_map_at.data_ptr<int>();
    const float* x = x_at.data_ptr<float>();
    const float* y = y_at.data_ptr<float>();
    const float* pin_offset_x = pin_offset_x_at.data_ptr<float>();
    const float* pin_offset_y = pin_offset_y_at.data_ptr<float>();
    int& num_nets = timing_raw_db.num_nets;

    constexpr const int scale = 1000;  // flute only supports integers.
    using Point2i = utils::PointT<int>;

    vector<int> edge_from;
    vector<int> edge_to;
    vector<float> edge_wl;
    vector<int> flat_net2node_start_map;
    vector<int> flat_net2edge_start_map;
    vector<int> node2pin_map;
    int node_count = 0;
    int edge_count = 0;
    flat_net2node_start_map.emplace_back(0);
    flat_net2edge_start_map.emplace_back(0);

    vector<vector<int>> net_id2edge_from(num_nets);
    vector<vector<int>> net_id2edge_to(num_nets);
    vector<vector<float>> net_id2edge_wl(num_nets);
    vector<vector<int>> net_id2node2pin_map(num_nets);
    omp_lock_t lock;
    omp_init_lock(&lock);
// #pragma omp parallel for
    for (int i = 0; i < num_nets; ++i) {
        const int degree = flat_net2pin_start_map[i + 1] - flat_net2pin_start_map[i];
        const int root = flat_net2pin_map[flat_net2pin_start_map[i]];
        std::map<Point2i, std::set<int>> pos2pins_map;
        std::vector<int> vx, vy;
        vx.reserve(degree);
        vy.reserve(degree);
        std::map<int, int> global2inner_map;
        if(dmp_debug_on)
            XPLACE_ERRORF("DMP_FLUTE_DEBUG", "net=%s", net_names[i].c_str());
        for (int j = 0; j < degree; ++j) {
            int pin = flat_net2pin_map[j + flat_net2pin_start_map[i]];
            int node = pin2node_map[pin];
            float offset_x = pin_offset_x[pin], offset_y = pin_offset_y[pin];
            // Find the correct pin locations given cell locations.
            auto x_ = static_cast<int>((x[node] + offset_x) * scale);
            auto y_ = static_cast<int>((y[node] + offset_y) * scale);
            if(dmp_debug_on)
                XPLACE_ERRORF("DMP_FLUTE_DEBUG",
                              "pin=%s node=%d loc=(%.4f,%.4f) offset=(%.4f,%.4f) final=(%d,%d)",
                              pin_names[pin].c_str(), node, x[node], y[node], offset_x, offset_y, x_, y_);
            global2inner_map[pin] = j;

            if (pos2pins_map.find(Point2i(x_, y_)) != pos2pins_map.end())
                pos2pins_map[Point2i(x_, y_)].insert(j);
            else {
                pos2pins_map.emplace(Point2i(x_, y_), std::set<int>{j});
                vx.emplace_back(x_);
                vy.emplace_back(y_);
            }
        }
        const int valid_size = static_cast<int>(vx.size());
        int num_pins = degree;
        std::set<Point2i> multipin_pos;
        std::map<Point2i, Point2i> pos2neighbor_map;
        if(dmp_debug_on)
            XPLACE_ERRORF("DMP_FLUTE_DEBUG", "net=%s degree=%d valid_size=%d", net_names[i].c_str(), degree, valid_size);
        if (valid_size > 1) {
            Tree flutetree = flute(valid_size, vx.data(), vy.data(), 8);
            for (int bid = 0; bid < 2 * valid_size - 2; ++bid) {
                Branch& branch1 = flutetree.branch[bid];
                Branch& branch2 = flutetree.branch[branch1.n];
                if(dmp_debug_on)
                    XPLACE_ERRORF("DMP_FLUTE_DEBUG",
                                  "branch=%d from=(%d,%d) to=(%d,%d) dist=%d",
                                  bid, branch1.x, branch1.y, branch2.x, branch2.y,
                                  utils::Dist(Point2i(branch1.x, branch1.y), Point2i(branch2.x, branch2.y)));
                Point2i p1(branch1.x, branch1.y), p2(branch2.x, branch2.y);

                if (p1 == p2) continue;

                pos2neighbor_map.emplace(p2, p1);
                auto& id1 = retrieve_pins_from_pos_dmp(pos2pins_map, p1, num_pins);
                auto& id2 = retrieve_pins_from_pos_dmp(pos2pins_map, p2, num_pins);

                auto distance = utils::Dist(p1, p2);
                float wl = static_cast<float>(distance * 1.0) / scale;

                if (!id1.empty() && !id2.empty()) {
                    auto base1 = id1.begin(), base2 = id2.begin();
                    if (*base1 != *base2) {
                        net_id2edge_from[i].emplace_back(*base1);
                        net_id2edge_to[i].emplace_back(*base2);
                        net_id2edge_wl[i].emplace_back(wl);
                    }
                    if (id1.size() > 1) multipin_pos.insert(p1);
                    if (id2.size() > 1) multipin_pos.insert(p2);
                }
            }
            free(flutetree.branch);
        } else if (valid_size == 1 && degree > 1) {
            multipin_pos.emplace(vx[0], vy[0]);
        }
        for (const auto& pos : multipin_pos) {
            const auto& pins = pos2pins_map[pos];
            int adj_pin = global2inner_map[root];
            const auto& _ppos = pos2neighbor_map[pos];
            if (auto itr = pos2pins_map.find(_ppos); itr != pos2pins_map.end()) {
                adj_pin = *itr->second.cbegin();
            }
            auto distance = utils::Dist(pos, _ppos);
            float wl = static_cast<float>(distance * 1.0) / scale;
            for (auto it = std::next(pins.cbegin()); it != pins.cend(); ++it) {
                net_id2edge_from[i].emplace_back(adj_pin);
                net_id2edge_to[i].emplace_back(*it);
                net_id2edge_wl[i].emplace_back(0);
            }
        }

        for (int j = 0; j < num_pins; ++j) {
            if (j < degree)
                net_id2node2pin_map[i].emplace_back(flat_net2pin_map[j + flat_net2pin_start_map[i]]);
            else
                net_id2node2pin_map[i].emplace_back(-1);
        }
    }
    omp_destroy_lock(&lock);

    for (int i = 0; i < num_nets; ++i) {
        for (int j = 0; j < net_id2edge_from[i].size(); ++j) {
            edge_from.emplace_back(node_count + net_id2edge_from[i][j]);
            edge_to.emplace_back(node_count + net_id2edge_to[i][j]);
            edge_wl.emplace_back(net_id2edge_wl[i][j]);
            edge_count++;
        }
        node_count += net_id2node2pin_map[i].size();
        for (int j = 0; j < net_id2node2pin_map[i].size(); ++j) {
            node2pin_map.emplace_back(net_id2node2pin_map[i][j]);
        }
        flat_net2node_start_map.emplace_back(node_count);
        flat_net2edge_start_map.emplace_back(edge_count);
    }

    return {edge_from, edge_to, edge_wl, flat_net2node_start_map, flat_net2edge_start_map, node2pin_map, node_count, edge_count};
}
void GPUTimer::update_rc_timing_flute_dmp(torch::Tensor node_lpos, bool record) {
    timing_raw_db.commit_from(node_lpos.index({"...", 0}).contiguous(), node_lpos.index({"...", 1}).contiguous());

    float unit_to_micron = scale_factor * microns;
    float rf = wire_resistance_per_micron / res_unit; // ohm/um -> kohm/um
    float cf = wire_capacitance_per_micron / cap_unit; // F/um -> fF/um

    auto [edge_from, edge_to, edge_wl, flat_net2node_start_map, flat_net2edge_start_map, node2pin_map, num_nodes, num_edges] =
        FluteRCTreeDMP(timing_raw_db, rf, cf, gtdb.net_names, gtdb.pin_names, dmp_debug_on);
    // auto device = timing_raw_db.node_size.device();
    // torch::Tensor node_order = torch::zeros({num_nodes}, torch::dtype(torch::kInt32).device(device)).contiguous();
    // torch::Tensor edge_order = torch::zeros({num_edges}, torch::dtype(torch::kInt32).device(device)).contiguous();
    // torch::Tensor parent_node = -torch::ones({num_nodes}, torch::dtype(torch::kInt32).device(device)).contiguous();
    // torch::Tensor res_parent = torch::zeros({num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32).device(device)).contiguous();
    // torch::Tensor node_cap = torch::zeros({num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32).device(device)).contiguous();
    // torch::Tensor edge_res = torch::zeros({num_edges}, torch::dtype(torch::kFloat32).device(device)).contiguous();
    initialize_dmp_rc(edge_from, edge_to, flat_net2node_start_map, flat_net2edge_start_map, node2pin_map, edge_wl, num_nets, num_nodes, num_edges, unit_to_micron, rf, cf);
    release_vector_storage(edge_from);
    release_vector_storage(edge_to);
    release_vector_storage(flat_net2node_start_map);
    release_vector_storage(flat_net2edge_start_map);
    release_vector_storage(node2pin_map);
    release_vector_storage(edge_wl);
    DMP_PROGRESS_PRINT("DMP RC calculation starting, num_nets: %d num_nodes: %d num_edges: %d",
                       num_nets, num_nodes, num_edges);
    calc_res_cap_dmp(dmp_db, num_nets);
    DMP_PROGRESS_PRINT("DMP RC calculation done.");
    propagate_rc_tree_dmp(dmp_db, num_nets);
    DMP_PROGRESS_PRINT("DMP RC propagation done.");
    apply_dmp_driving_cell_source_slew(*this);

    if (record) {
        timing_raw_db.pinImpulse_ref.data().copy_(timing_raw_db.pinImpulse.data());
        timing_raw_db.pinLoad_ref.data().copy_(timing_raw_db.pinLoad.data());
        timing_raw_db.pinRootDelay_ref.data().copy_(timing_raw_db.pinRootDelay.data());
    }
}

void GPUTimer::run_dmp_rc(int num_nets, bool update_timing_after_rc) {
    calc_res_cap_dmp(dmp_db, num_nets);
    propagate_rc_tree_dmp(dmp_db, num_nets);
    if (update_timing_after_rc) {
        dmp_prepare_timing_after_rc(h_dmp_db, dmp_db);
    }
    apply_dmp_driving_cell_source_slew(*this);
}

void GPUTimer::init_dmp_rc_spef() {
    HostRcGraph graph = build_spef_rc();
    const int graph_num_nets = graph.num_nets;
    const int graph_num_nodes = graph.num_nodes;
    const int graph_num_edges = graph.num_edges;
    initialize_dmp_rc_explicit(graph);
    graph.release_storage();
    DMP_PROGRESS_PRINT("DMP SPEF RC calculation starting, num_nets: %d num_nodes: %d num_edges: %d",
                       graph_num_nets, graph_num_nodes, graph_num_edges);
    run_dmp_rc(graph_num_nets, false);
    DMP_PROGRESS_PRINT("DMP SPEF RC propagation done.");
}

void GPUTimer::init_dmp_rc_gr(const std::string& file) {
    HostRcGraph graph = build_openroad_gr_rc(file);
    const int graph_num_nets = graph.num_nets;
    const int graph_num_nodes = graph.num_nodes;
    const int graph_num_edges = graph.num_edges;
    initialize_dmp_rc_explicit(graph);
    graph.release_storage();
    DMP_PROGRESS_PRINT("DMP GR RC calculation starting, num_nets: %d num_nodes: %d num_edges: %d",
                       graph_num_nets, graph_num_nodes, graph_num_edges);
    run_dmp_rc(graph_num_nets, true);
    DMP_PROGRESS_PRINT("DMP GR RC propagation done.");
}

void GPUTimer::init_dmp_rc_route_segments(const std::string& file) {
    HostRcGraph graph = build_openroad_route_segments_rc(file);
    const int graph_num_nets = graph.num_nets;
    const int graph_num_nodes = graph.num_nodes;
    const int graph_num_edges = graph.num_edges;
    initialize_dmp_rc_explicit(graph);
    graph.release_storage();
    DMP_PROGRESS_PRINT("DMP route-segment RC calculation starting, num_nets: %d num_nodes: %d num_edges: %d",
                       graph_num_nets, graph_num_nodes, graph_num_edges);
    run_dmp_rc(graph_num_nets, true);
    DMP_PROGRESS_PRINT("DMP route-segment RC propagation done.");
}

void GPUTimer::debug_dump_dmp_rc_net(const std::string& net_name) {
    std::string normalized_net_name = net_name;
    validate_token(normalized_net_name);
    int net_idx = -1;
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        std::string candidate = gtdb.net_names[i];
        validate_token(candidate);
        if (gtdb.net_names[i] == net_name || candidate == normalized_net_name) {
            net_idx = i;
            break;
        }
    }
    if (net_idx < 0) {
        XPLACE_ERRORF("DMP_RC_DUMP", "net %s not found", net_name.c_str());
        return;
    }
    debug_dump_dmp_rc_net_cuda(h_dmp_db, net_idx, gtdb.net_names, gtdb.pin_names);
}


}
