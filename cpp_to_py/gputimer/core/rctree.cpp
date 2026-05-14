
#include "GPUTimer.h"
#include "common/utils/utils.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "common/db/Geometry.h"
#include "common/db/Layer.h"
#include "common/db/Net.h"
#include "common/db/Pin.h"
#include "gputimer/db/GTDatabase.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <flute.hpp>
using namespace Flute;

namespace gt {

void update_rc_timing_cuda(float* x,
                           float* y,
                           const float* pin_offset_x,
                           const float* pin_offset_y,
                           const int* pin2node_map,
                           const int* flat_net2pin_start_map,
                           const int* flat_net2pin_map,
                           float* pinLoad,
                           float* pinImpulse,
                           float* pinCap,
                           float* pinWireCap,
                           float* pinRootDelay,
                           float* pinRootRes,
                           int num_nets,
                           int num_pins,
                           float unit_to_micron,
                           int* net_is_clock,
                           float cf,
                           float rf);

void GPUTimer::update_rc_timing(torch::Tensor node_lpos, bool record, bool load, bool conpensation) {
    timing_raw_db.commit_from(node_lpos.index({"...", 0}).contiguous(), node_lpos.index({"...", 1}).contiguous());
    float unit_to_micron = scale_factor * microns;
    float rf = wire_resistance_per_micron / res_unit;
    float cf = wire_capacitance_per_micron / cap_unit;
    update_rc_timing_cuda(x,
                          y,
                          pin_offset_x,
                          pin_offset_y,
                          pin2node_map,
                          flat_net2pin_start_map,
                          flat_net2pin_map,
                          pinLoad,
                          pinImpulse,
                          pinCap,
                          pinWireCap,
                          pinRootDelay,
                          pinRootRes,
                          num_nets,
                          num_pins,
                          unit_to_micron,
                          net_is_clock,
                          cf,
                          rf);
    if (record) {
        auto ratio_load = torch::nan_to_num(timing_raw_db.pinLoad_ref / timing_raw_db.pinLoad, 1.0);
        timing_raw_db.pinLoad_ratio.data().copy_(ratio_load.contiguous().data());
        auto ratio_delay = torch::sqrt(torch::nan_to_num(timing_raw_db.pinRootDelay_ref / timing_raw_db.pinRootDelay, 1.0));
        timing_raw_db.pinRootDelay_ratio.data().copy_(ratio_delay.contiguous().data());

        auto delay_comp = (torch::nan_to_num(timing_raw_db.pinRootDelay_ref - timing_raw_db.pinRootDelay, 0)).clamp(0.0);
        timing_raw_db.pinRootDelay_compensation.data().copy_(delay_comp.contiguous().data());
    }
    if (load) {
        timing_raw_db.pinImpulse.data().copy_(timing_raw_db.pinImpulse_ref.data());
        timing_raw_db.pinLoad *= timing_raw_db.pinLoad_ratio;
        if (conpensation)
            timing_raw_db.pinRootDelay += timing_raw_db.pinRootDelay_compensation;
        else
            timing_raw_db.pinRootDelay *= timing_raw_db.pinRootDelay_ratio;
    }
}

// ------------------------------------------------------------------------------------------------------------------------
//

auto& retrieve_pins_from_pos(std::map<utils::PointT<int>, std::set<int>>& pos2pins_map, const utils::PointT<int>& point, int& index) {
    if (pos2pins_map.find(point) != pos2pins_map.end()) return pos2pins_map[point];
    pos2pins_map.emplace(point, std::set<int>{index++});
    return pos2pins_map[point];
}

tuple<vector<int>, vector<int>, vector<float>, vector<int>, vector<int>, vector<int>, int, int> FluteRCTree(TimingTorchRawDB& timing_raw_db,
                                                                                                            float rf,
                                                                                                            float cf) {
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
    flat_net2node_start_map.push_back(0);
    flat_net2edge_start_map.push_back(0);

    vector<vector<int>> net_id2edge_from(num_nets);
    vector<vector<int>> net_id2edge_to(num_nets);
    vector<vector<float>> net_id2edge_wl(num_nets);
    vector<vector<int>> net_id2node2pin_map(num_nets);

    omp_lock_t lock;
    omp_init_lock(&lock);
#pragma omp parallel for
    for (int i = 0; i < num_nets; ++i) {
        const int degree = flat_net2pin_start_map[i + 1] - flat_net2pin_start_map[i];
        const int root = flat_net2pin_map[flat_net2pin_start_map[i]];
        std::map<Point2i, std::set<int>> pos2pins_map;
        std::vector<int> vx, vy;
        vx.reserve(degree);
        vy.reserve(degree);

        std::map<int, int> global2inner_map;

        for (int j = 0; j < degree; ++j) {
            int pin = flat_net2pin_map[j + flat_net2pin_start_map[i]];
            int node = pin2node_map[pin];
            float offset_x = pin_offset_x[pin], offset_y = pin_offset_y[pin];
            // Find the correct pin locations given cell locations.
            auto x_ = static_cast<int>((x[node] + offset_x) * scale);
            auto y_ = static_cast<int>((y[node] + offset_y) * scale);
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

        if (valid_size > 1) {
            Tree flutetree = flute(valid_size, vx.data(), vy.data(), 8);

            for (int bid = 0; bid < 2 * valid_size - 2; ++bid) {
                Branch& branch1 = flutetree.branch[bid];
                Branch& branch2 = flutetree.branch[branch1.n];

                Point2i p1(branch1.x, branch1.y), p2(branch2.x, branch2.y);

                if (p1 == p2) continue;

                pos2neighbor_map.emplace(p2, p1);
                auto& id1 = retrieve_pins_from_pos(pos2pins_map, p1, num_pins);
                auto& id2 = retrieve_pins_from_pos(pos2pins_map, p2, num_pins);

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
                net_id2node2pin_map[i].push_back(flat_net2pin_map[j + flat_net2pin_start_map[i]]);
            else
                net_id2node2pin_map[i].push_back(-1);
        }
    }
    omp_destroy_lock(&lock);

    for (int i = 0; i < num_nets; ++i) {
        for (int j = 0; j < net_id2edge_from[i].size(); ++j) {
            edge_from.push_back(node_count + net_id2edge_from[i][j]);
            edge_to.push_back(node_count + net_id2edge_to[i][j]);
            edge_wl.push_back(net_id2edge_wl[i][j]);
            edge_count++;
        }
        node_count += net_id2node2pin_map[i].size();
        for (int j = 0; j < net_id2node2pin_map[i].size(); ++j) {
            node2pin_map.push_back(net_id2node2pin_map[i][j]);
        }
        flat_net2node_start_map.push_back(node_count);
        flat_net2edge_start_map.push_back(edge_count);
    }

    return {edge_from, edge_to, edge_wl, flat_net2node_start_map, flat_net2edge_start_map, node2pin_map, node_count, edge_count};
}

void flatten_rc_tree(std::vector<int> host_edge_from,
                     std::vector<int> host_edge_to,
                     float* edge_res,
                     float* node_cap,
                     std::vector<int> host_flat_net2node_start_map,
                     std::vector<int> host_flat_net2edge_start_map,
                     std::vector<int> host_node2pin_map,
                     int* node_order,
                     int* edge_order,
                     int* parent_node,
                     float* res_parent,
                     float* pinLoad,
                     float* pinImpulse,
                     float* pinCap,
                     float* pinWireCap,
                     float* pinRootDelay,
                     float* pinRootRes,
                     int num_nets,
                     int num_pins,
                     int num_nodes,
                     int num_edges);

void propagate_rc_tree(std::vector<int> host_edge_from,
                       std::vector<int> host_edge_to,
                       float* edge_res,
                       float* node_cap,
                       std::vector<int> host_flat_net2node_start_map,
                       std::vector<int> host_flat_net2edge_start_map,
                       std::vector<int> host_node2pin_map,
                       std::vector<uint8_t> host_includes_pin_caps,
                       int* node_order,
                       int* parent_node,
                       float* res_parent,
                       float* pinLoad,
                       float* pinImpulse,
                       float* pinCap,
                       float* pinWireCap,
                       float* pinRootDelay,
                       float* pinRootRes,
                       int num_nets,
                       int num_pins,
                       int num_nodes,
                       int num_edges);


void calc_res_cap(std::vector<int> host_edge_from,
                  std::vector<int> host_edge_to,
                  int* edge_order,
                  float* edge_res,
                  float* node_cap,
                  std::vector<int> host_flat_net2node_start_map,
                  std::vector<int> host_flat_net2edge_start_map,
                  std::vector<int> host_node2pin_map,
                  std::vector<float> host_edge_wl,
                  int num_nets,
                  int num_edges,
                  int num_nodes,
                  int* net_is_clock,
                  float unit_to_micron,
                  float rf,
                  float cf);

void GPUTimer::update_rc_timing_flute(torch::Tensor node_lpos, bool record) {
    timing_raw_db.commit_from(node_lpos.index({"...", 0}).contiguous(), node_lpos.index({"...", 1}).contiguous());

    float unit_to_micron = scale_factor * microns;
    float rf = wire_resistance_per_micron / res_unit;
    float cf = wire_capacitance_per_micron / cap_unit;

    auto [edge_from, edge_to, edge_wl, flat_net2node_start_map, flat_net2edge_start_map, node2pin_map, num_nodes, num_edges] =
        FluteRCTree(timing_raw_db, rf, cf);
    auto device = timing_raw_db.node_size_x.device();
    torch::Tensor node_order = torch::zeros({num_nodes}, torch::dtype(torch::kInt32).device(device)).contiguous();
    torch::Tensor edge_order = torch::zeros({num_edges}, torch::dtype(torch::kInt32).device(device)).contiguous();
    torch::Tensor parent_node = -torch::ones({num_nodes}, torch::dtype(torch::kInt32).device(device)).contiguous();
    torch::Tensor res_parent = torch::zeros({num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32).device(device)).contiguous();
    torch::Tensor node_cap = torch::zeros({num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32).device(device)).contiguous();
    torch::Tensor edge_res = torch::zeros({num_edges}, torch::dtype(torch::kFloat32).device(device)).contiguous();

    calc_res_cap(edge_from,
                 edge_to,
                 edge_order.data_ptr<int>(),
                 edge_res.data_ptr<float>(),
                 node_cap.data_ptr<float>(),
                 flat_net2node_start_map,
                 flat_net2edge_start_map,
                 node2pin_map,
                 edge_wl,
                 num_nets,
                 num_edges,
                 num_nodes,
                 net_is_clock,
                 unit_to_micron,
                 rf,
                 cf);

    flatten_rc_tree(edge_from,
                    edge_to,
                    edge_res.data_ptr<float>(),
                    node_cap.data_ptr<float>(),
                    flat_net2node_start_map,
                    flat_net2edge_start_map,
                    node2pin_map,
                    node_order.data_ptr<int>(),
                    edge_order.data_ptr<int>(),
                    parent_node.data_ptr<int>(),
                    res_parent.data_ptr<float>(),
                    pinLoad,
                    pinImpulse,
                    pinCap,
                    pinWireCap,
                    pinRootDelay,
                    pinRootRes,
                    num_nets,
                    num_pins,
                    num_nodes,
                    num_edges);

    propagate_rc_tree(edge_from,
                      edge_to,
                      edge_res.data_ptr<float>(),
                      node_cap.data_ptr<float>(),
                      flat_net2node_start_map,
                      flat_net2edge_start_map,
                      node2pin_map,
                      std::vector<uint8_t>(),
                      node_order.data_ptr<int>(),
                      parent_node.data_ptr<int>(),
                      res_parent.data_ptr<float>(),
                      pinLoad,
                      pinImpulse,
                      pinCap,
                      pinWireCap,
                      pinRootDelay,
                      pinRootRes,
                      num_nets,
                      num_pins,
                      num_nodes,
                      num_edges);

    if (record) {
        timing_raw_db.pinImpulse_ref.data().copy_(timing_raw_db.pinImpulse.data());
        timing_raw_db.pinLoad_ref.data().copy_(timing_raw_db.pinLoad.data());
        timing_raw_db.pinRootDelay_ref.data().copy_(timing_raw_db.pinRootDelay.data());
    }
}

namespace {

struct SpefRcBuildStats {
    int parsed_nets = 0;
    int missing_nets = 0;
    int missing_pin_nodes = 0;
    int unresolved_cap_nodes = 0;
    int unresolved_res_nodes = 0;
    int ground_caps = 0;
    int coupling_caps = 0;
    int folded_coupling_terms = 0;
    int resistors = 0;
    int skipped_self_resistors = 0;
    int skipped_loop_edges = 0;
    int repaired_edges = 0;
    int fallback_nets = 0;
};

struct LocalSpefNetRc {
    std::vector<int> edge_from;
    std::vector<int> edge_to;
    std::vector<float> edge_res;
    std::vector<float> node_cap;
    std::vector<int> node2pin;
    std::vector<std::string> node_names;
    std::unordered_map<std::string, int> node_name2id;
    std::unordered_map<std::string, int> pin_name2id;
};

static std::string normalized_spef_name(std::string name)
{
    validate_token(name);
    return name;
}

static bool spef_digits_only(const std::string& value)
{
    if (value.empty()) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isdigit(c) != 0;
    });
}

static void add_name_alias(std::unordered_map<std::string, int>& map,
                           const std::string& name,
                           int value)
{
    if (name.empty()) {
        return;
    }
    map.emplace(name, value);
    map.emplace(normalized_spef_name(name), value);
}

static void add_attr_cap(std::vector<float>& node_cap, int node, float cap)
{
    for (int attr = 0; attr < NUM_ATTR; ++attr) {
        node_cap[node * NUM_ATTR + attr] += cap;
    }
}

static std::string spef_upper(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

static bool spef_includes_pin_caps_from_design_flow(const std::string& design_flow)
{
    std::string flow = spef_upper(design_flow);
    size_t pin_cap_pos = flow.find("PIN_CAP");
    if (pin_cap_pos == std::string::npos) {
        return false;
    }
    std::string pin_cap_clause = flow.substr(pin_cap_pos);
    return pin_cap_clause.find("NONE") == std::string::npos;
}

static int count_tree_edges_from_root(const LocalSpefNetRc& local)
{
    if (local.node2pin.empty()) {
        return 0;
    }
    std::vector<uint8_t> seen(local.node2pin.size(), 0);
    std::vector<int> stack;
    int tree_edges = 0;
    seen[0] = 1;
    stack.emplace_back(0);
    for (size_t cursor = 0; cursor < stack.size(); ++cursor) {
        int node = stack[cursor];
        for (size_t edge = 0; edge < local.edge_from.size(); ++edge) {
            int from = local.edge_from[edge];
            int to = local.edge_to[edge];
            int next = -1;
            if (from == node) next = to;
            if (to == node) next = from;
            if (next >= 0 && next < static_cast<int>(seen.size()) && !seen[next]) {
                seen[next] = 1;
                stack.emplace_back(next);
                tree_edges++;
            }
        }
    }
    return tree_edges;
}


}  // namespace

HostRcGraph GPUTimer::build_spef_rc() {
    if (!gtdb.spef_res_unit.has_value() || !gtdb.spef_cap_unit.has_value()) {
        throw std::runtime_error("build_spef_rc requires read_spef() before building the SPEF RC graph.");
    }

    torch::Tensor flat_net2pin_start_map_at = timing_raw_db.flat_net2pin_start_map.clone().cpu().contiguous();
    torch::Tensor flat_net2pin_map_at = timing_raw_db.flat_net2pin_map.clone().cpu().contiguous();
    const int* flat_net2pin_start_map = flat_net2pin_start_map_at.data_ptr<int>();
    const int* flat_net2pin_map = flat_net2pin_map_at.data_ptr<int>();

    float spef_res_ratio = *gtdb.spef_res_unit / gtdb.res_unit;
    float spef_cap_ratio = *gtdb.spef_cap_unit / gtdb.cap_unit;
    float spef_time_ratio = gtdb.spef_time_unit.has_value() ? *gtdb.spef_time_unit / gtdb.time_unit : 1.0f;
    logger.info("spef lib ratios: res %.5E cap %.5E time %.5E", spef_res_ratio, spef_cap_ratio, spef_time_ratio);

    std::unordered_map<std::string, int> net_name_to_index;
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        add_name_alias(net_name_to_index, gtdb.net_names[i], i);
    }

    std::unordered_map<std::string, int> global_pin_name_to_id;
    for (int i = 0; i < static_cast<int>(gtdb.pin_names.size()); ++i) {
        add_name_alias(global_pin_name_to_id, gtdb.pin_names[i], i);
    }

    std::vector<LocalSpefNetRc> local_nets(num_nets);
    std::vector<uint8_t> parsed_net(num_nets, 0);
    SpefRcBuildStats stats;
    const std::string delimiter = spef.delimiter.empty() ? ":" : spef.delimiter;
    const bool spef_includes_pin_caps = spef_includes_pin_caps_from_design_flow(spef.design_flow);

    auto init_net = [&](int net_idx) {
        auto& local = local_nets[net_idx];
        if (!local.node2pin.empty()) {
            return;
        }
        int start = flat_net2pin_start_map[net_idx];
        int end = flat_net2pin_start_map[net_idx + 1];
        local.node2pin.reserve(end - start);
        local.node_cap.reserve((end - start) * NUM_ATTR);
        for (int j = start; j < end; ++j) {
            int pin_id = flat_net2pin_map[j];
            int local_id = static_cast<int>(local.node2pin.size());
            local.node2pin.emplace_back(pin_id);
            local.node_names.emplace_back(gtdb.pin_names[pin_id]);
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                local.node_cap.emplace_back(0.0f);
            }
            add_name_alias(local.node_name2id, gtdb.pin_names[pin_id], local_id);
            add_name_alias(local.pin_name2id, gtdb.pin_names[pin_id], local_id);
        }
    };

    auto add_internal_node = [&](int net_idx, const std::string& node_name) {
        auto& local = local_nets[net_idx];
        int node_id = static_cast<int>(local.node2pin.size());
        local.node2pin.emplace_back(-1);
        local.node_names.emplace_back(node_name);
        for (int attr = 0; attr < NUM_ATTR; ++attr) {
            local.node_cap.emplace_back(0.0f);
        }
        add_name_alias(local.node_name2id, node_name, node_id);
        return node_id;
    };

    auto resolve_node = [&](const std::string& raw_name, int net_idx, bool create) {
        init_net(net_idx);
        auto& local = local_nets[net_idx];
        std::string node_name = normalized_spef_name(raw_name);
        if (auto it = local.node_name2id.find(node_name); it != local.node_name2id.end()) {
            return it->second;
        }

        size_t delim_pos = delimiter.empty() ? std::string::npos : node_name.rfind(delimiter);
        if (delim_pos != std::string::npos) {
            std::string name1 = normalized_spef_name(node_name.substr(0, delim_pos));
            std::string name2 = normalized_spef_name(node_name.substr(delim_pos + delimiter.size()));
            std::string pin_candidate = name1 + ":" + name2;
            if (auto it = local.pin_name2id.find(pin_candidate); it != local.pin_name2id.end()) {
                add_name_alias(local.node_name2id, node_name, it->second);
                return it->second;
            }
            if (auto it = local.pin_name2id.find(node_name); it != local.pin_name2id.end()) {
                return it->second;
            }
            if (auto pin_it = global_pin_name_to_id.find(pin_candidate); pin_it != global_pin_name_to_id.end()) {
                stats.missing_pin_nodes++;
                return -1;
            }

            auto net_it = net_name_to_index.find(name1);
            if (net_it != net_name_to_index.end()) {
                if (net_it->second == net_idx && spef_digits_only(name2)) {
                    return create ? add_internal_node(net_idx, node_name) : -1;
                }
                return -1;
            }

            stats.missing_pin_nodes++;
            return -1;
        }

        if (auto it = local.pin_name2id.find(node_name); it != local.pin_name2id.end()) {
            return it->second;
        }
        if (global_pin_name_to_id.find(node_name) != global_pin_name_to_id.end()) {
            stats.missing_pin_nodes++;
        }
        return -1;
    };

    for (const auto& n : spef.nets) {
        string net_name = normalized_spef_name(n.name);
        auto net_itr = net_name_to_index.find(net_name);
        if (net_itr == net_name_to_index.end()) continue;

        int net_idx = net_itr->second;
        if (!parsed_net[net_idx]) {
            parsed_net[net_idx] = 1;
            stats.parsed_nets++;
        }
        init_net(net_idx);
        auto& local = local_nets[net_idx];

        for (const auto& [node1, node2, cap] : n.caps) {
            const float cap_internal = cap * spef_cap_ratio;
            if (node2.empty()) {
                stats.ground_caps++;
                int node = resolve_node(node1, net_idx, true);
                if (node >= 0) {
                    add_attr_cap(local.node_cap, node, cap_internal);
                } else {
                    stats.unresolved_cap_nodes++;
                }
            } else {
                stats.coupling_caps++;
                int node_a = resolve_node(node1, net_idx, true);
                int node_b = resolve_node(node2, net_idx, true);
                if (node_a >= 0) {
                    add_attr_cap(local.node_cap, node_a, cap_internal);
                    stats.folded_coupling_terms++;
                }
                if (node_b >= 0) {
                    add_attr_cap(local.node_cap, node_b, cap_internal);
                    stats.folded_coupling_terms++;
                }
                if (node_a < 0 && node_b < 0) {
                    stats.unresolved_cap_nodes++;
                }
            }
        }

        for (const auto& [node1, node2, res] : n.ress) {
            stats.resistors++;
            int from = resolve_node(node1, net_idx, true);
            int to = resolve_node(node2, net_idx, true);
            if (from < 0 || to < 0) {
                stats.unresolved_res_nodes++;
                continue;
            }
            if (from == to) {
                stats.skipped_self_resistors++;
                continue;
            }
            local.edge_from.emplace_back(from);
            local.edge_to.emplace_back(to);
            local.edge_res.emplace_back(res * spef_res_ratio);
        }
    }

    HostRcGraph graph;
    graph.includes_pin_caps.assign(num_nets, spef_includes_pin_caps ? 1 : 0);
    graph.net2node_start.emplace_back(0);
    graph.net2edge_start.emplace_back(0);

    for (int i = 0; i < num_nets; ++i) {
        if (!parsed_net[i]) {
            stats.missing_nets++;
            stats.fallback_nets++;
            init_net(i);
            auto& local = local_nets[i];
            for (int node = 1; node < static_cast<int>(local.node2pin.size()); ++node) {
                local.edge_from.emplace_back(0);
                local.edge_to.emplace_back(node);
                local.edge_res.emplace_back(0.0f);
            }
        }

        auto& local = local_nets[i];
        if (!local.node2pin.empty()) {
            std::vector<uint8_t> seen(local.node2pin.size(), 0);
            std::vector<int> stack;
            seen[0] = 1;
            stack.emplace_back(0);
            for (size_t cursor = 0; cursor < stack.size(); ++cursor) {
                int node = stack[cursor];
                for (size_t edge = 0; edge < local.edge_from.size(); ++edge) {
                    int from = local.edge_from[edge];
                    int to = local.edge_to[edge];
                    int next = -1;
                    if (from == node) next = to;
                    if (to == node) next = from;
                    if (next >= 0 && next < static_cast<int>(seen.size()) && !seen[next]) {
                        seen[next] = 1;
                        stack.emplace_back(next);
                    }
                }
            }
            for (int node = 1; node < static_cast<int>(seen.size()); ++node) {
                if (!seen[node]) {
                    local.edge_from.emplace_back(0);
                    local.edge_to.emplace_back(node);
                    local.edge_res.emplace_back(0.0f);
                    stats.repaired_edges++;
                }
            }
        }
        int tree_edges = count_tree_edges_from_root(local);
        int skipped_loop_edges = static_cast<int>(local.edge_from.size()) - tree_edges;
        if (skipped_loop_edges > 0) {
            stats.skipped_loop_edges += skipped_loop_edges;
        }

        for (size_t edge = 0; edge < local.edge_from.size(); ++edge) {
            graph.edge_from.emplace_back(graph.num_nodes + local.edge_from[edge]);
            graph.edge_to.emplace_back(graph.num_nodes + local.edge_to[edge]);
            graph.edge_res.emplace_back(local.edge_res[edge]);
            graph.num_edges++;
        }
        for (int node = 0; node < static_cast<int>(local.node2pin.size()); ++node) {
            graph.node2pin.emplace_back(local.node2pin[node]);
            graph.node_names.emplace_back(local.node_names[node]);
            for (int attr = 0; attr < NUM_ATTR; ++attr) {
                graph.node_cap.emplace_back(local.node_cap[node * NUM_ATTR + attr]);
            }
        }
        graph.num_nodes += static_cast<int>(local.node2pin.size());
        graph.net2node_start.emplace_back(graph.num_nodes);
        graph.net2edge_start.emplace_back(graph.num_edges);
    }
    graph.skipped_loop_edges = stats.skipped_loop_edges;
    graph.repaired_edges = stats.repaired_edges;

    logger.info("SPEF RC graph: parsed_nets=%d missing_nets=%d nodes=%d edges=%d includes_pin_caps=%d",
                stats.parsed_nets, stats.missing_nets, graph.num_nodes, graph.num_edges,
                spef_includes_pin_caps ? 1 : 0);
    logger.info("SPEF RC details: ground_caps=%d coupling_caps=%d folded_coupling_terms=%d resistors=%d self_res=%d loop_edges=%d missing_pin_nodes=%d unresolved_cap_nodes=%d unresolved_res_nodes=%d repaired_edges=%d fallback_nets=%d",
                stats.ground_caps, stats.coupling_caps, stats.folded_coupling_terms,
                stats.resistors, stats.skipped_self_resistors, stats.skipped_loop_edges,
                stats.missing_pin_nodes, stats.unresolved_cap_nodes, stats.unresolved_res_nodes,
                stats.repaired_edges, stats.fallback_nets);
    return graph;
}

void GPUTimer::debug_dump_spef_rc_net(const std::string& net_name) {
    HostRcGraph graph = build_spef_rc();
    std::string normalized_net_name = normalized_spef_name(net_name);
    int net_idx = -1;
    for (int i = 0; i < static_cast<int>(gtdb.net_names.size()); ++i) {
        if (gtdb.net_names[i] == net_name || normalized_spef_name(gtdb.net_names[i]) == normalized_net_name) {
            net_idx = i;
            break;
        }
    }
    if (net_idx < 0) {
        logger.warning("SPEF RC dump: net %s not found", net_name.c_str());
        return;
    }
    int nst = graph.net2node_start[net_idx];
    int nend = graph.net2node_start[net_idx + 1];
    int est = graph.net2edge_start[net_idx];
    int eend = graph.net2edge_start[net_idx + 1];
    printf("[SPEF RC DUMP] net=%s id=%d nodes=%d edges=%d includes_pin_caps=%d repaired_edges_total=%d skipped_loop_edges_total=%d\n",
           gtdb.net_names[net_idx].c_str(), net_idx, nend - nst, eend - est,
           graph.includes_pin_caps[net_idx] ? 1 : 0, graph.repaired_edges,
           graph.skipped_loop_edges);
    for (int node = nst; node < nend; ++node) {
        int local_node = node - nst;
        int pin = graph.node2pin[node];
        printf("[SPEF RC DUMP] node local=%d global=%d pin=%d name=%s cap=(%.9e,%.9e,%.9e,%.9e)\n",
               local_node, node, pin, graph.node_names[node].c_str(),
               graph.node_cap[node * NUM_ATTR + 0],
               graph.node_cap[node * NUM_ATTR + 1],
               graph.node_cap[node * NUM_ATTR + 2],
               graph.node_cap[node * NUM_ATTR + 3]);
    }
    for (int edge = est; edge < eend; ++edge) {
        printf("[SPEF RC DUMP] edge local=%d global=%d from=%d to=%d res=%.9e\n",
               edge - est, edge, graph.edge_from[edge] - nst,
               graph.edge_to[edge] - nst, graph.edge_res[edge]);
    }
    fflush(stdout);
}

void GPUTimer::update_rc_timing_spef() {
    HostRcGraph graph = build_spef_rc();
    auto device = timing_raw_db.node_size_x.device();
    torch::Tensor edge_res = torch::from_blob(graph.edge_res.data(), {graph.num_edges}, torch::dtype(torch::kFloat32)).contiguous().to(device);
    torch::Tensor node_cap = torch::from_blob(graph.node_cap.data(), {graph.num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32)).contiguous().to(device);
    torch::Tensor node_order = torch::zeros({graph.num_nodes}, torch::kInt32).contiguous().to(device);
    torch::Tensor edge_order = torch::zeros({graph.num_edges}, torch::kInt32).contiguous().to(device);
    torch::Tensor parent_node = -torch::ones({graph.num_nodes}, torch::dtype(torch::kInt32).device(device));
    torch::Tensor res_parent = torch::zeros({graph.num_nodes * NUM_ATTR}, torch::dtype(torch::kFloat32).device(device));

    flatten_rc_tree(graph.edge_from,
                    graph.edge_to,
                    edge_res.data_ptr<float>(),
                    node_cap.data_ptr<float>(),
                    graph.net2node_start,
                    graph.net2edge_start,
                    graph.node2pin,
                    node_order.data_ptr<int>(),
                    edge_order.data_ptr<int>(),
                    parent_node.data_ptr<int>(),
                    res_parent.data_ptr<float>(),
                    pinLoad,
                    pinImpulse,
                    pinCap,
                    pinWireCap,
                    pinRootDelay,
                    pinRootRes,
                    num_nets,
                    num_pins,
                    graph.num_nodes,
                    graph.num_edges);

    propagate_rc_tree(graph.edge_from,
                      graph.edge_to,
                      edge_res.data_ptr<float>(),
                      node_cap.data_ptr<float>(),
                      graph.net2node_start,
                      graph.net2edge_start,
                      graph.node2pin,
                      graph.includes_pin_caps,
                      node_order.data_ptr<int>(),
                      parent_node.data_ptr<int>(),
                      res_parent.data_ptr<float>(),
                      pinLoad,
                      pinImpulse,
                      pinCap,
                      pinWireCap,
                      pinRootDelay,
                      pinRootRes,
                      num_nets,
                      num_pins,
                      graph.num_nodes,
                      graph.num_edges);
}
}  // namespace gt
