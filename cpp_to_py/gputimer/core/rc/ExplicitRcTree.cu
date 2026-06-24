#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "gputimer/core/rc/RcModels.h"
#include "gputimer/core/utils.cuh"

namespace gt {

__global__ void flatten_rc_kernel(RcTreeDevice *rc_device_ptr) {
    const int idx = blockIdx.x;
    RcTreeDevice& m = *rc_device_ptr;
    const RcTreeDeviceGraph& g = m.graph;
    if (idx < g.num_nets) {
        int nst = g.flat_net2node_start_map[idx];
        int nend = g.flat_net2node_start_map[idx + 1];
        int root = nst;

        int est = g.flat_net2edge_start_map[idx];
        int eend = g.flat_net2edge_start_map[idx + 1];

        if (threadIdx.x == 0) {
            m.parent_node[root] = -1;
            m.root_dist[root] = 0;
        }

        __syncthreads();

        for (int d = 0; d < nend - nst; d++) {
            for (int i = est + threadIdx.x; i < eend; i += blockDim.x) {
                int from = g.edge_from[i];
                int to = g.edge_to[i];
                float res = m.edge_res[i];
                if ((m.root_dist[from] == d) && (m.root_dist[to] == -1)) {
                    m.parent_node[to] = from;
                    m.root_dist[to] = d + 1;
                    atomicAdd(&m.cnts[d + nst], 1);
                    for (int j = 0; j < NUM_ATTR; j++) {
                        atomicAdd(&m.res_parent[to * NUM_ATTR + j], res);
                    }
                } else if ((m.root_dist[to] == d) && (m.root_dist[from] == -1)) {
                    m.parent_node[from] = to;
                    m.root_dist[from] = d + 1;
                    atomicAdd(&m.cnts[d + nst], 1);
                    for (int j = 0; j < NUM_ATTR; j++) {
                        atomicAdd(&m.res_parent[from * NUM_ATTR + j], res);
                    }
                }
            }
            __syncthreads();
            if (m.cnts[d + nst] == 0) break;
        }

        if (threadIdx.x == 0) {
            const int num_edges_local = eend - est;

            // calculate accumulation
            int edge_count = 0;
            for (int i = 0; i < num_edges_local; i++) {
                edge_count += m.cnts[i + nst];  // FIXME:
                m.cnts[i + nst] = edge_count;
            }

            // calculate order
            for (int i = 0; i < num_edges_local; i++) {
                int from = g.edge_from[i + est];
                int to = g.edge_to[i + est];
                int min_d = min(m.root_dist[from], m.root_dist[to]);

                int start = min_d == 0 ? 0 : m.cnts[min_d - 1 + nst];
                m.edge_order[est + start + m.edge_cnts[min_d + est]] = i + est;
                atomicAdd(&m.edge_cnts[min_d + est], 1);
            }
        }

        __syncthreads();

        // sort according to dist and cnts
        __shared__ int offset;
        if (threadIdx.x == 0) {
            offset = 0;
        }
        __syncthreads();
        for (int d = 0; d < nend - nst; d++) {
            // if (threadIdx.x == 0) {
            //     offset += cnts[d + nst];
            // }
            __syncthreads();
            for (int i = nst + threadIdx.x; i < nend; i += blockDim.x) {
                if (m.root_dist[i] == d) {
                    int pos = atomicAdd(&m.cnts[d + nst], -1);
                    int off = atomicAdd(&offset, 1);
                    // order[nst + offset - pos] = i;
                    m.node_order[nst + off] = i;
                }
            }
            __syncthreads();
        }
    }
}

__global__ void propagate_rc_kernel(RcTreeDevice *rc_device_ptr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cond = threadIdx.y;
    RcTreeDevice& m = *rc_device_ptr;
    const RcTreeDeviceGraph& g = m.graph;
    RcTreePropagation& s = m.propagation;
    if (idx < g.num_nets) {
        int nst = g.flat_net2node_start_map[idx];
        int nend = g.flat_net2node_start_map[idx + 1];

        for (int i = nend - 1; i >= nst; i--) {
            int node = m.node_order[i];
            int pnode = m.parent_node[node];
            int pin = g.node2pin_map[node];
            float wire_cap = m.node_cap[node * NUM_ATTR + cond];
            if (pin != -1) {
                float pin_cap_lib = (!g.includes_pin_caps || g.includes_pin_caps[idx] == 0)
                                        ? pin_cap_attr(m.pinCap, pin, cond)
                                        : 0.0f;
                float pin_load = m.pinLoad[pin * NUM_ATTR + cond];
                wire_cap = wire_cap + pin_cap_lib + pin_load;
            }
            atomicAdd(&s.node_load[node * NUM_ATTR + cond], wire_cap);
            if (pnode != -1) atomicAdd(&s.node_load[pnode * NUM_ATTR + cond], s.node_load[node * NUM_ATTR + cond]);
        }
        for (int i = nst + 1; i < nend; i++) {
            int node = m.node_order[i];
            int pnode = m.parent_node[node];
            float t = s.node_load[node * NUM_ATTR + cond] * m.res_parent[node * NUM_ATTR + cond];
            s.node_delay[node * NUM_ATTR + cond] = s.node_delay[pnode * NUM_ATTR + cond] + t;
        }
        for (int i = nend - 1; i >= nst; i--) {
            int node = m.node_order[i];
            int pnode = m.parent_node[node];
            int pin = g.node2pin_map[node];
            float wire_cap = m.node_cap[node * NUM_ATTR + cond];
            if (pin != -1) {
                float pin_cap_lib = (!g.includes_pin_caps || g.includes_pin_caps[idx] == 0)
                                        ? pin_cap_attr(m.pinCap, pin, cond)
                                        : 0.0f;
                float pin_load = m.pinLoad[pin * NUM_ATTR + cond];
                wire_cap = wire_cap + pin_cap_lib + pin_load;
            }
            float l = wire_cap * s.node_delay[node * NUM_ATTR + cond];
            atomicAdd(&s.node_ldelay[node * NUM_ATTR + cond], l);
            if (pnode != -1) atomicAdd(&s.node_ldelay[pnode * NUM_ATTR + cond], s.node_ldelay[node * NUM_ATTR + cond]);
        }
        for (int i = nst + 1; i < nend; i++) {
            int node = m.node_order[i];
            int pnode = m.parent_node[node];
            float t = s.node_ldelay[node * NUM_ATTR + cond] * m.res_parent[node * NUM_ATTR + cond];
            s.node_beta[node * NUM_ATTR + cond] = s.node_beta[pnode * NUM_ATTR + cond] + t;
            s.node_impulse[node * NUM_ATTR + cond] =
                sqrt(2 * s.node_beta[node * NUM_ATTR + cond] - s.node_delay[node * NUM_ATTR + cond] * s.node_delay[node * NUM_ATTR + cond]);
        }
    }
}

__global__ void move_to_timing_graph(RcTreeDevice *rc_device_ptr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cond = threadIdx.y;
    RcTreeDevice& m = *rc_device_ptr;
    const RcTreeDeviceGraph& g = m.graph;
    const RcTreePropagation& s = m.propagation;
    if (idx < g.num_nets) {
        int nst = g.flat_net2node_start_map[idx];
        int nend = g.flat_net2node_start_map[idx + 1];
        for (int i = nst; i < nend; i++) {
            int node = i;
            int pin = g.node2pin_map[node];
            if (pin != -1) {
                m.pinLoad[pin * NUM_ATTR + cond] = s.node_load[node * NUM_ATTR + cond];
                m.pinRootDelay[pin * NUM_ATTR + cond] = s.node_delay[node * NUM_ATTR + cond];
                m.pinImpulse[pin * NUM_ATTR + cond] = s.node_impulse[node * NUM_ATTR + cond];
            }
        }
    }
}

static void copy_rc_tree_graph_to_device(const RcTreeHost& rc_host,
                                         RcTreeDeviceGraph& graph,
                                         float** edge_wl) {
    graph.num_nets = rc_host.num_nets;
    graph.num_nodes = rc_host.num_nodes;
    graph.num_edges = rc_host.num_edges;
    cudaMalloc(&graph.edge_from, rc_host.edge_from->size() * sizeof(int));
    cudaMalloc(&graph.edge_to, rc_host.edge_to->size() * sizeof(int));
    cudaMalloc(&graph.flat_net2node_start_map, rc_host.flat_net2node_start_map->size() * sizeof(int));
    cudaMalloc(&graph.flat_net2edge_start_map, rc_host.flat_net2edge_start_map->size() * sizeof(int));
    cudaMalloc(&graph.node2pin_map, rc_host.node2pin_map->size() * sizeof(int));
    cudaMemcpy(graph.edge_from, rc_host.edge_from->data(), rc_host.edge_from->size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph.edge_to, rc_host.edge_to->data(), rc_host.edge_to->size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph.flat_net2node_start_map, rc_host.flat_net2node_start_map->data(), rc_host.flat_net2node_start_map->size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph.flat_net2edge_start_map, rc_host.flat_net2edge_start_map->data(), rc_host.flat_net2edge_start_map->size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(graph.node2pin_map, rc_host.node2pin_map->data(), rc_host.node2pin_map->size() * sizeof(int), cudaMemcpyHostToDevice);
    if (rc_host.includes_pin_caps && !rc_host.includes_pin_caps->empty()) {
        cudaMalloc(&graph.includes_pin_caps, rc_host.includes_pin_caps->size() * sizeof(uint8_t));
        cudaMemcpy(graph.includes_pin_caps, rc_host.includes_pin_caps->data(), rc_host.includes_pin_caps->size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }
    if (edge_wl && rc_host.edge_wl) {
        cudaMalloc(edge_wl, rc_host.edge_wl->size() * sizeof(float));
        cudaMemcpy(*edge_wl, rc_host.edge_wl->data(), rc_host.edge_wl->size() * sizeof(float), cudaMemcpyHostToDevice);
    }
}

static void free_rc_tree_device_graph(RcTreeDeviceGraph& graph) {
    cudaFree(graph.edge_from);
    cudaFree(graph.edge_to);
    cudaFree(graph.flat_net2node_start_map);
    cudaFree(graph.flat_net2edge_start_map);
    cudaFree(graph.node2pin_map);
    if (graph.includes_pin_caps) cudaFree(graph.includes_pin_caps);
}

static RcTreeDevice* copy_rc_tree_device_to_gpu(const RcTreeDevice& rc_device) {
    RcTreeDevice* d_rc_device = nullptr;
    cudaMalloc(&d_rc_device, sizeof(RcTreeDevice));
    cudaMemcpy(d_rc_device, &rc_device, sizeof(RcTreeDevice), cudaMemcpyHostToDevice);
    return d_rc_device;
}

void flatten_rc_tree(const RcTreeHost& rc_host) {
    int *root_dist, *cnts;
    int *edge_cnts;
    cudaMalloc(&root_dist, rc_host.num_nodes * sizeof(int));
    cudaMalloc(&cnts, rc_host.num_nodes * sizeof(int));
    cudaMalloc(&edge_cnts, rc_host.num_edges * sizeof(int));

    reset_val<int><<<BLOCK_NUMBER(rc_host.num_nodes), BLOCK_SIZE>>>(root_dist, rc_host.num_nodes);
    cudaMemset(cnts, 0, rc_host.num_nodes * sizeof(int));
    cudaMemset(edge_cnts, 0, rc_host.num_edges * sizeof(int));

    RcTreeDevice rc_device;
    copy_rc_tree_graph_to_device(rc_host, rc_device.graph, nullptr);
    rc_device.root_dist = root_dist;
    rc_device.cnts = cnts;
    rc_device.edge_cnts = edge_cnts;
    rc_device.node_order = rc_host.node_order;
    rc_device.edge_order = rc_host.edge_order;
    rc_device.parent_node = rc_host.parent_node;
    rc_device.edge_res = rc_host.edge_res;
    rc_device.res_parent = rc_host.res_parent;
    RcTreeDevice* d_rc_device = copy_rc_tree_device_to_gpu(rc_device);

    int thread_count = 64;
    int numBlocks = rc_host.num_nets;
    flatten_rc_kernel<<<numBlocks, thread_count>>>(d_rc_device);
    cudaFree(d_rc_device);
    free_rc_tree_device_graph(rc_device.graph);
    cudaFree(root_dist);
    cudaFree(cnts);
    cudaFree(edge_cnts);

    // device sync
    cudaDeviceSynchronize();
}

void propagate_rc_tree(const RcTreeHost& rc_host) {
    RcTreeDevice rc_device;
    copy_rc_tree_graph_to_device(rc_host, rc_device.graph, nullptr);

    float *node_load, *node_delay, *node_ldelay, *node_impulse, *node_beta;

    cudaMalloc(&node_load, rc_host.num_nodes * NUM_ATTR * sizeof(float));
    cudaMalloc(&node_delay, rc_host.num_nodes * NUM_ATTR * sizeof(float));
    cudaMalloc(&node_ldelay, rc_host.num_nodes * NUM_ATTR * sizeof(float));
    cudaMalloc(&node_impulse, rc_host.num_nodes * NUM_ATTR * sizeof(float));
    cudaMalloc(&node_beta, rc_host.num_nodes * NUM_ATTR * sizeof(float));

    cudaMemset(node_load, 0, rc_host.num_nodes * NUM_ATTR * sizeof(float));
    cudaMemset(node_delay, 0, rc_host.num_nodes * NUM_ATTR * sizeof(float));
    cudaMemset(node_ldelay, 0, rc_host.num_nodes * NUM_ATTR * sizeof(float));
    cudaMemset(node_impulse, 0, rc_host.num_nodes * NUM_ATTR * sizeof(float));
    cudaMemset(node_beta, 0, rc_host.num_nodes * NUM_ATTR * sizeof(float));

    rc_device.propagation.node_load = node_load;
    rc_device.propagation.node_delay = node_delay;
    rc_device.propagation.node_ldelay = node_ldelay;
    rc_device.propagation.node_impulse = node_impulse;
    rc_device.propagation.node_beta = node_beta;
    rc_device.node_order = rc_host.node_order;
    rc_device.parent_node = rc_host.parent_node;
    rc_device.res_parent = rc_host.res_parent;
    rc_device.node_cap = rc_host.node_cap;
    rc_device.pinLoad = rc_host.pinLoad;
    rc_device.pinImpulse = rc_host.pinImpulse;
    rc_device.pinCap = rc_host.pinCap;
    rc_device.pinWireCap = rc_host.pinWireCap;
    rc_device.pinRootDelay = rc_host.pinRootDelay;
    rc_device.pinRootRes = rc_host.pinRootRes;
    RcTreeDevice* d_rc_device = copy_rc_tree_device_to_gpu(rc_device);

    int thread_count2 = 64;
    dim3 block_size(thread_count2, NUM_ATTR);
    int numBlocks2 = rc_host.num_nets - 1 + thread_count2 / thread_count2;

    propagate_rc_kernel<<<numBlocks2, block_size>>>(d_rc_device);
    move_to_timing_graph<<<numBlocks2, block_size>>>(d_rc_device);

    cudaFree(d_rc_device);
    free_rc_tree_device_graph(rc_device.graph);
    cudaFree(node_load);
    cudaFree(node_delay);
    cudaFree(node_ldelay);
    cudaFree(node_impulse);
    cudaFree(node_beta);
    cudaDeviceSynchronize();
}

__global__ void calc_rc_kernel(RcTreeDevice *rc_device_ptr) {
    const int idx = blockIdx.x;
    RcTreeDevice& m = *rc_device_ptr;
    const RcTreeDeviceGraph& g = m.graph;
    if (idx < g.num_nets) {
        int nst = g.flat_net2node_start_map[idx];
        int nend = g.flat_net2node_start_map[idx + 1];
        int root = nst;

        int est = g.flat_net2edge_start_map[idx];
        int eend = g.flat_net2edge_start_map[idx + 1];

        if (threadIdx.x == 0) {
            m.root_dist[root] = 0;
        }
        __syncthreads();

        for (int d = 0; d < nend - nst; d++) {
            for (int i = est + threadIdx.x; i < eend; i += blockDim.x) {
                int from = g.edge_from[i];
                int to = g.edge_to[i];
                float wl = m.edge_wl[i];
                if (m.net_is_clock[idx] == 1) wl = 0;
                float cap = wl * m.cf * 0.5 / m.unit_to_micron;
                float res = wl * m.rf / m.unit_to_micron;
                if ((m.root_dist[from] == d) && (m.root_dist[to] == -1)) {
                    m.root_dist[to] = d + 1;
                    atomicAdd(&m.cnts[d + nst], 1);
                    atomicAdd(&m.edge_res[i], res);
                    for (int j = 0; j < NUM_ATTR; j++) {
                        atomicAdd(&m.node_cap[to * NUM_ATTR + j], cap);
                        atomicAdd(&m.node_cap[from * NUM_ATTR + j], cap);
                    }
                } else if ((m.root_dist[to] == d) && (m.root_dist[from] == -1)) {
                    m.root_dist[from] = d + 1;
                    atomicAdd(&m.cnts[d + nst], 1);
                    atomicAdd(&m.edge_res[i], res);
                    for (int j = 0; j < NUM_ATTR; j++) {
                        atomicAdd(&m.node_cap[to * NUM_ATTR + j], cap);
                        atomicAdd(&m.node_cap[from * NUM_ATTR + j], cap);
                    }
                }
            }
            __syncthreads();
            if (m.cnts[d + nst] == 0) break;
        }
    }
}

void calc_res_cap(const RcTreeHost& rc_host) {
    int *root_dist, *cnts;
    cudaMalloc(&root_dist, rc_host.num_nodes * sizeof(int));
    cudaMalloc(&cnts, rc_host.num_nodes * sizeof(int));

    reset_val<int><<<BLOCK_NUMBER(rc_host.num_nodes), BLOCK_SIZE>>>(root_dist, rc_host.num_nodes);
    cudaMemset(cnts, 0, rc_host.num_nodes * sizeof(int));

    RcTreeDevice rc_device;
    copy_rc_tree_graph_to_device(rc_host, rc_device.graph, &rc_device.edge_wl);
    rc_device.root_dist = root_dist;
    rc_device.cnts = cnts;
    rc_device.edge_order = rc_host.edge_order;
    rc_device.edge_res = rc_host.edge_res;
    rc_device.node_cap = rc_host.node_cap;
    rc_device.net_is_clock = rc_host.net_is_clock;
    rc_device.unit_to_micron = rc_host.unit_to_micron;
    rc_device.rf = rc_host.rf;
    rc_device.cf = rc_host.cf;
    RcTreeDevice* d_rc_device = copy_rc_tree_device_to_gpu(rc_device);

    int thread_count = 64;
    int numBlocks = rc_host.num_nets;
    calc_rc_kernel<<<numBlocks, thread_count>>>(d_rc_device);

    cudaFree(d_rc_device);
    free_rc_tree_device_graph(rc_device.graph);
    cudaFree(root_dist);
    cudaFree(cnts);
    cudaFree(rc_device.edge_wl);
}

}  // namespace gt
