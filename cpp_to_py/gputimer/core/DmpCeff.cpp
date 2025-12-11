// #include "DmpCeff.h"
#include "GPUTimer.h"
#include "rctree.h"
#include "common/utils/utils.h"
#include "common/db/Database.h"
#include "gputimer/db/GTDatabase.h"
#include <flute.hpp>
#include <pd.h>
using namespace flt;

namespace gt {

void compute_pi_model_cuda(dmp_model* dmp_db, int num_nets);
// void GPUTimer::compute_pi_model(){
//     assert(dmp_db != nullptr);
//     compute_pi_model_cuda(dmp_db, num_nets);
// }

void update_timing_dmp_cuda(dmp_model* dmp_db, vector<int> level_list_end_cpu);
void print_pinLoad_cuda(dmp_model* dmp_db, vector<int> level_list_end_cpu, vector<string> pin_names);
void GPUTimer::update_timing_dmp(){
    printf("Update timing DMP started.\n");
    fflush(stdout);
    update_timing_dmp_cuda(dmp_db, level_list_end_cpu);
}
void GPUTimer::print_pinLoad(){
    print_pinLoad_cuda(dmp_db, level_list_end_cpu, gtdb.pin_names);
}


void calc_res_cap_dmp(dmp_model* dmp_db, int num_nets);
void propagate_rc_tree_dmp(dmp_model* dmp_db, int num_nets);
void GPUTimer::get_units(){
    printf("time unit: %E\n", time_unit());
    printf("res unit: %E\n", res_unit);
    printf("cap unit: %E\n", cap_unit);
    printf("micron: %d\n", microns);
    printf("scale factor: %E\n", scale_factor);
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

    constexpr const int scale = 1;  // flute only supports integers.
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
    Flute Fobj;     
    Fobj.readLUT();                                                                                                       
    omp_lock_t lock;
    omp_init_lock(&lock);
// #pragma omp parallel for
    for (int i = 0; i < num_nets; ++i) {
        const int degree = flat_net2pin_start_map[i + 1] - flat_net2pin_start_map[i];
        const int root = flat_net2pin_map[flat_net2pin_start_map[i]];
        std::map<Point2i, std::set<int>> pos2pins_map;
        std::vector<int> vx, vy;
        std::vector<std::pair<int, int>> pin_locs;
        vx.reserve(degree);
        vy.reserve(degree);
        pin_locs.reserve(degree);
        std::map<int, int> global2inner_map;
        if(dmp_debug_on)
            printf("net:%s\n", net_names[i].c_str());
        int root_x = 0, root_y = 0;
        for (int j = 0; j < degree; ++j) {
            int pin = flat_net2pin_map[j + flat_net2pin_start_map[i]];
            int node = pin2node_map[pin];
            float offset_x = pin_offset_x[pin], offset_y = pin_offset_y[pin];
            // Find the correct pin locations given cell locations.
            auto x_ = static_cast<int>((x[node] + offset_x) * scale);
            auto y_ = static_cast<int>((y[node] + offset_y) * scale);
            if(j == 0){
                root_x = x_;
                root_y = y_;
            }
            if(dmp_debug_on)
                printf("pin:%s node:%d loc:(%.4f, %.4f) offset:(%.4f, %.4f) final:(%d, %d)\n", pin_names[pin].c_str(), node, x[node], y[node], offset_x, offset_y, x_, y_);
            global2inner_map[pin] = j;

            if (pos2pins_map.find(Point2i(x_, y_)) != pos2pins_map.end())
                pos2pins_map[Point2i(x_, y_)].insert(j);
            else {
                pos2pins_map.emplace(Point2i(x_, y_), std::set<int>{j});
                pin_locs.emplace_back(x_, y_);
            }
        }
        const int valid_size = static_cast<int>(pin_locs.size());
        sort(pin_locs.begin(), pin_locs.end());
        int root_idx = 0;
        for (int i = 0; i < pin_locs.size(); ++i) {
            auto& loc = pin_locs[i];
            if(loc.first == root_x && loc.second == root_y){
                root_idx = i;
            }
            if(dmp_debug_on)
                printf("  loc:(%d, %d)\n", loc.first, loc.second);
            vx.emplace_back(loc.first);
            vy.emplace_back(loc.second);
        }
        int num_pins = degree;
        std::set<Point2i> multipin_pos;
        std::map<Point2i, Point2i> pos2neighbor_map;
        if(dmp_debug_on)
            printf("net:%s degree:%d valid_size:%d\n", net_names[i].c_str(), degree, valid_size);
        if (valid_size > 1) {
            // Tree flutetree = Fobj.flute(vx, vy, 3);
            Tree flutetree = pdr::primDijkstra(vx, vy, root_idx, 0.3);
            // if(i == 0)printf("flutetree branch count:%d\n", flutetree.branchCount());
            for (int bid = 0; bid < flutetree.branchCount(); ++bid) {
                Branch& branch1 = flutetree.branch[bid];
                Branch& branch2 = flutetree.branch[branch1.n];
                if(dmp_debug_on)
                    printf("flutetree branch %d : %d %d -> %d %d dist %d\n", bid, branch1.x, branch1.y, branch2.x, branch2.y, Dist(Point2i(branch1.x, branch1.y), Point2i(branch2.x, branch2.y)));
                Point2i p1(branch1.x, branch1.y), p2(branch2.x, branch2.y);

                if (p1 == p2) continue;

                pos2neighbor_map.emplace(p2, p1);
                auto& id1 = retrieve_pins_from_pos_dmp(pos2pins_map, p1, num_pins);
                auto& id2 = retrieve_pins_from_pos_dmp(pos2pins_map, p2, num_pins);

                auto distance = Dist(p1, p2);
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
            vector<Branch>().swap(flutetree.branch);
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
            auto distance = Dist(pos, _ppos);
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
    printf("DMP RC calculation starting, num_nets: %d num_nodes: %d num_edges: %d\n", num_nets, num_nodes, num_edges);
    fflush(stdout);
    calc_res_cap_dmp(dmp_db, num_nets);
    printf("DMP RC calculation done.\n");
    fflush(stdout);
    propagate_rc_tree_dmp(dmp_db, num_nets);
    printf("DMP RC propagation done.\n");
    fflush(stdout);

    if (record) {
        timing_raw_db.pinImpulse_ref.data().copy_(timing_raw_db.pinImpulse.data());
        timing_raw_db.pinLoad_ref.data().copy_(timing_raw_db.pinLoad.data());
        timing_raw_db.pinRootDelay_ref.data().copy_(timing_raw_db.pinRootDelay.data());
    }
}


}