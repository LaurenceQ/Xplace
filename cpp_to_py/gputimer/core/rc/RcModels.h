#pragma once

#include <cstdint>
#include <vector>

namespace gt {

struct RcStarModel {
    float* x = nullptr;
    float* y = nullptr;
    const float* pin_offset_x = nullptr;
    const float* pin_offset_y = nullptr;
    const int* pin2node_map = nullptr;
    const int* flat_net2pin_start_map = nullptr;
    const int* flat_net2pin_map = nullptr;
    float* pinLoad = nullptr;
    float* pinImpulse = nullptr;
    float* pinCap = nullptr;
    float* pinWireCap = nullptr;
    float* pinRootDelay = nullptr;
    float* pinRootRes = nullptr;
    int num_nets = 0;
    int num_pins = 0;
    float unit_to_micron = 1.0f;
    const uint8_t* net_is_clock = nullptr;
    float cf = 0.0f;
    float rf = 0.0f;
};

struct RcGraphDeviceView {
    int* edge_from = nullptr;
    int* edge_to = nullptr;
    int* flat_net2node_start_map = nullptr;
    int* flat_net2edge_start_map = nullptr;
    int* node2pin_map = nullptr;
    unsigned char* includes_pin_caps = nullptr;
    int num_nets = 0;
    int num_nodes = 0;
    int num_edges = 0;
};

struct RcPropagateScratch {
    float* node_load = nullptr;
    float* node_delay = nullptr;
    float* node_ldelay = nullptr;
    float* node_impulse = nullptr;
    float* node_beta = nullptr;
};

struct RcExplicitTreeModel {
    const std::vector<int>* edge_from = nullptr;
    const std::vector<int>* edge_to = nullptr;
    const std::vector<int>* flat_net2node_start_map = nullptr;
    const std::vector<int>* flat_net2edge_start_map = nullptr;
    const std::vector<int>* node2pin_map = nullptr;
    const std::vector<float>* edge_wl = nullptr;
    const std::vector<uint8_t>* includes_pin_caps = nullptr;
    int* node_order = nullptr;
    int* edge_order = nullptr;
    int* parent_node = nullptr;
    float* edge_res = nullptr;
    float* node_cap = nullptr;
    float* res_parent = nullptr;
    float* pinLoad = nullptr;
    float* pinImpulse = nullptr;
    float* pinCap = nullptr;
    float* pinWireCap = nullptr;
    float* pinRootDelay = nullptr;
    float* pinRootRes = nullptr;
    const uint8_t* net_is_clock = nullptr;
    int num_nets = 0;
    int num_pins = 0;
    int num_nodes = 0;
    int num_edges = 0;
    float unit_to_micron = 1.0f;
    float rf = 0.0f;
    float cf = 0.0f;
    RcExplicitTreeModel() = default;
    RcExplicitTreeModel(
        const std::vector<int>* edge_from_,
        const std::vector<int>* edge_to_,
        const std::vector<int>* flat_net2node_start_map_,
        const std::vector<int>* flat_net2edge_start_map_,
        const std::vector<int>* node2pin_map_,
        const std::vector<uint8_t>* includes_pin_caps_,
        int* node_order_,
        int* edge_order_,
        int* parent_node_,
        float* edge_res_,
        float* node_cap_,
        float* res_parent_,
        float* pinLoad_,
        float* pinImpulse_,
        float* pinCap_,
        float* pinWireCap_,
        float* pinRootDelay_,
        float* pinRootRes_,
        int num_nets_,
        int num_pins_,
        int num_nodes_,
        int num_edges_
    ) : edge_from(edge_from_),
        edge_to(edge_to_),
        flat_net2node_start_map(flat_net2node_start_map_),
        flat_net2edge_start_map(flat_net2edge_start_map_),
        node2pin_map(node2pin_map_),
        includes_pin_caps(includes_pin_caps_),
        node_order(node_order_),
        edge_order(edge_order_),
        parent_node(parent_node_),
        edge_res(edge_res_),
        node_cap(node_cap_),
        res_parent(res_parent_),
        pinLoad(pinLoad_),
        pinImpulse(pinImpulse_),
        pinCap(pinCap_),
        pinWireCap(pinWireCap_),
        pinRootDelay(pinRootDelay_),
        pinRootRes(pinRootRes_),
        num_nets(num_nets_),
        num_pins(num_pins_),
        num_nodes(num_nodes_),
        num_edges(num_edges_) {}
};

struct RcExplicitDeviceModel {
    RcGraphDeviceView graph;
    RcPropagateScratch scratch;
    int* root_dist = nullptr;
    int* cnts = nullptr;
    int* edge_cnts = nullptr;
    int* node_order = nullptr;
    int* edge_order = nullptr;
    int* parent_node = nullptr;
    float* edge_res = nullptr;
    float* node_cap = nullptr;
    float* edge_wl = nullptr;
    float* res_parent = nullptr;
    float* pinLoad = nullptr;
    float* pinImpulse = nullptr;
    float* pinCap = nullptr;
    float* pinWireCap = nullptr;
    float* pinRootDelay = nullptr;
    float* pinRootRes = nullptr;
    const uint8_t* net_is_clock = nullptr;
    float unit_to_micron = 1.0f;
    float rf = 0.0f;
    float cf = 0.0f;
};

void update_rc_timing_cuda(const RcStarModel& model);
void calc_res_cap(const RcExplicitTreeModel& model);
void flatten_rc_tree(const RcExplicitTreeModel& model);
void propagate_rc_tree(const RcExplicitTreeModel& model);

}  // namespace gt
