#include "DmpCeff.h"
#include "GPUTimer.h"


namespace gt{
__host__ void dmp_model::initialize_rc(std::vector<int> host_edge_from,
               std::vector<int> host_edge_to,
               std::vector<int> host_flat_net2node_start_map,
               std::vector<int> host_flat_net2edge_start_map,
               std::vector<int> host_node2pin_map,
               std::vector<float> host_edge_wl,
               float *pinCap_,
               int num_nets_,
               int num_nodes_,
               int num_edges_,
               float unit_to_micron_,
               float rf_,
               float cf_){
        pinCap = pinCap_;
        num_nets = num_nets_;
        num_nodes = num_nodes_;
        num_edges = num_edges_;
        unit_to_micron = unit_to_micron_;
        rf = rf_;
        cf = cf_;
        cudaMalloc(&edge_from, host_edge_from.size() * sizeof(int));
        cudaMalloc(&edge_to, host_edge_to.size() * sizeof(int));
        cudaMalloc(&flat_net2node_start_map, host_flat_net2node_start_map.size() * sizeof(int));
        cudaMalloc(&flat_net2edge_start_map, host_flat_net2edge_start_map.size() * sizeof(int));
        cudaMalloc(&node2pin_map, host_node2pin_map.size() * sizeof(int));
        cudaMalloc(&edge_wl, host_edge_wl.size() * sizeof(float));    

        cudaMemcpy(edge_from, host_edge_from.data(), host_edge_from.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(edge_to, host_edge_to.data(), host_edge_to.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(flat_net2node_start_map, host_flat_net2node_start_map.data(), host_flat_net2node_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(flat_net2edge_start_map, host_flat_net2edge_start_map.data(), host_flat_net2edge_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(node2pin_map, host_node2pin_map.data(), host_node2pin_map.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(edge_wl, host_edge_wl.data(), host_edge_wl.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaMalloc(&root_dist, num_nodes * sizeof(int));
        cudaMalloc(&cnts, num_nodes * sizeof(int));
        cudaMalloc(&node_order, num_nodes * sizeof(int));
        cudaMalloc(&parent_node, num_nodes * sizeof(int));
        cudaMalloc(&res_parent, num_nodes * sizeof(float));
        cudaMalloc(&node_cap, num_nodes * sizeof(float));
        cudaMalloc(&node_delay, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&y1, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&y2, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&y3, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&down_cap, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&elmore_delay, num_pins * NUM_ATTR * sizeof(float));
        cudaMalloc(&C1, num_pins * NUM_ATTR * sizeof(double));
        cudaMalloc(&C2, num_pins * NUM_ATTR * sizeof(double));
        cudaMalloc(&r_pi, num_pins * NUM_ATTR * sizeof(double));

        cudaMemset(cnts, 0, num_nodes * sizeof(int));
        cudaMemset(node_order, 0, num_nodes * sizeof(int));
        cudaMemset(parent_node, -1, num_nodes * sizeof(int));
        cudaMemset(res_parent, 0, num_nodes * sizeof(float));
        cudaMemset(node_cap, 0, num_nodes * sizeof(float));
        cudaMemset(node_delay, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(y1, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(y2, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(y3, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(down_cap, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(elmore_delay, 0, num_pins * NUM_ATTR * sizeof(float));
        cudaMemset(root_dist, -1, num_nodes * sizeof(int));
      }
void GPUTimer::initialize_dmp_rc(
                  std::vector<int> host_edge_from,
                  std::vector<int> host_edge_to,
                  std::vector<int> host_flat_net2node_start_map,
                  std::vector<int> host_flat_net2edge_start_map,
                  std::vector<int> host_node2pin_map,
                  std::vector<float> host_edge_wl,
                  int num_nets,
                  int num_nodes,
                  int num_edges,
                  float unit_to_micron,
                  float rf,
                  float cf){
    h_dmp_db = new dmp_model(this);
    h_dmp_db -> initialize_rc(host_edge_from,
                              host_edge_to,
                              host_flat_net2node_start_map,
                              host_flat_net2edge_start_map,
                              host_node2pin_map,
                              host_edge_wl,
                              pinCap,
                              num_nets,
                              num_nodes,
                              num_edges,
                              unit_to_micron,
                              rf,
                              cf);
    cudaMalloc(&dmp_db, sizeof(dmp_model));
    cudaMemcpy(dmp_db, h_dmp_db, sizeof(dmp_model), cudaMemcpyHostToDevice);
   
}

__device__ __forceinline__ void dmp_model::calc_dmp_rc(){
    const int idx = blockIdx.x;
    if (idx < num_nets) {
        int nst = flat_net2node_start_map[idx];
        int nend = flat_net2node_start_map[idx + 1];
        int root = nst;

        int est = flat_net2edge_start_map[idx];
        int eend = flat_net2edge_start_map[idx + 1];

        if (threadIdx.x == 0) {
            parent_node[root] = -1;
            root_dist[root] = 0;
            int root_pin = node2pin_map[root];
            if(debug_on)printf("reduce parasitics: net %s id:%d driv_pin=%s root=%d\n", net_names[idx], idx, pin_names[root_pin], root);
        }
        __syncthreads();
        int offset = 1;
        node_order[nst] = root;

        for (int d = 0; d < nend - nst; d++) {
            for (int i = est + threadIdx.x; i < eend; i += blockDim.x) {
                int from = edge_from[i];
                int to = edge_to[i];
                float wl = edge_wl[i];
                float cap = wl * cf * 0.5 / unit_to_micron; // unit * fF/um * um/unit = fF
                float res = wl * rf / unit_to_micron; // unit * kohm/um * um/unit = kohm
                if ((root_dist[from] == d) && (root_dist[to] == -1)) {
                    root_dist[to] = d + 1;
                    parent_node[to] = from;
                    res_parent[to] = res;
                    node_cap[to] = cap;
                    int order = atomicAdd(&cnts[d + nst], 1);
                    if(debug_on)printf("net:%s nst:%d offset:%d order:%d from:%d to:%d res:%.4f, cap:%.4f, wl:%.6f\n", net_names[idx], nst, offset, order, from, to, res, cap, wl);
                    node_order[nst + offset + order] = to;
                    atomicAdd(&node_cap[from], cap);

                } else if ((root_dist[to] == d) && (root_dist[from] == -1)) {
                    parent_node[from] = to;
                    root_dist[from] = d + 1;
                    res_parent[from] = res;
                    node_cap[from] = cap;
                    int order = atomicAdd(&cnts[d + nst], 1);
                    if(debug_on)printf("net:%s nst:%d offset:%d order:%d from:%d to:%d res:%.4f, cap:%.4f, wl:%.6f\n", net_names[idx], nst, offset, order, to, from, res, cap, wl);
                    node_order[nst + offset + order] = from;
                    atomicAdd(&node_cap[to], cap);
                }
            }
            __syncthreads();
            if (cnts[d + nst] == 0) break;
            offset += cnts[d + nst];
        }
    }
}

__global__ void calc_dmp_rc(dmp_model* dmp_rc_){
    dmp_rc_->calc_dmp_rc();
}

__device__ __forceinline__ void dmp_model::propagate_dmp_rc(){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cond = threadIdx.y;
    if (idx < num_nets) {
        int nst = flat_net2node_start_map[idx];
        int nend = flat_net2node_start_map[idx + 1];
        // printf("Propagate RC for net %d from node %d to %d\n", idx, nst, nend);
       for (int i = nend - 1; i >= nst; i--) {
            
            int node = node_order[i];
            int pnode = parent_node[node];
            int pin = node2pin_map[node];
            float wire_cap = node_cap[node];
            // printf("  Node %d, Parent %d, Pin %d, Wire cap %.6f\n", node, pnode, pin, wire_cap);
            if (pin != -1) {
                float pin_cap_lib =
                    isnan(pinCap[pin * (NUM_ATTR + 2) + cond]) ? pinCap[pin * (NUM_ATTR + 2) + 4 + (cond >> 1)] : pinCap[pin * (NUM_ATTR + 2) + cond];
                wire_cap = wire_cap + pin_cap_lib;
            }
            y1[node * NUM_ATTR + cond] += wire_cap;
            down_cap[node * NUM_ATTR + cond] += wire_cap;
            if (pnode != -1){
                down_cap[pnode * NUM_ATTR + cond] += down_cap[node * NUM_ATTR + cond];
                y1[pnode * NUM_ATTR + cond] += y1[node * NUM_ATTR + cond];
                y2[pnode * NUM_ATTR + cond] += y2[node * NUM_ATTR + cond] - res_parent[node] * y1[node * NUM_ATTR + cond] * y1[node * NUM_ATTR + cond];
                y3[pnode * NUM_ATTR + cond] += y3[node * NUM_ATTR + cond] - 2 * res_parent[node] * y1[node * NUM_ATTR + cond] * y2[node * NUM_ATTR + cond]
                            + res_parent[node] * res_parent[node] * y1[node * NUM_ATTR + cond] * y1[node * NUM_ATTR + cond] * y1[node * NUM_ATTR + cond];
            }
        }
        int root = node_order[nst];
        int root_pin = node2pin_map[node_order[nst]];

        if(debug_on)printf("reduce parasitics: net %s id:%d driv_pin=%s root=%d\n", net_names[idx], idx, pin_names[root_pin], root);

        if(y2[root * NUM_ATTR + cond] < 1e-10 && y3[root * NUM_ATTR + cond] < 1e-10){
            C1[root_pin * NUM_ATTR + cond] = y1[root * NUM_ATTR + cond];
            C2[root_pin * NUM_ATTR + cond] = 0.0;
            r_pi[root_pin * NUM_ATTR + cond] = 0.0;
        }
        else{
            C1[root_pin * NUM_ATTR + cond] = y2[root * NUM_ATTR + cond] * y2[root * NUM_ATTR + cond] / y3[root * NUM_ATTR + cond];
            C2[root_pin * NUM_ATTR + cond] = y1[root * NUM_ATTR + cond] - y2[root * NUM_ATTR + cond] * y2[root * NUM_ATTR + cond] / y3[root * NUM_ATTR + cond];
            r_pi[root_pin * NUM_ATTR + cond] = -y3[root * NUM_ATTR + cond] * y3[root * NUM_ATTR + cond] / (y2[root * NUM_ATTR + cond] * y2[root * NUM_ATTR + cond] * y2[root * NUM_ATTR + cond]);
        }
        for (int i = nst + 1; i < nend; i++) {
            int node = node_order[i];
            int pnode = parent_node[node];
            int pin = node2pin_map[node];
            float t = down_cap[node * NUM_ATTR + cond] * res_parent[node];
            if(pnode != -1)node_delay[node * NUM_ATTR + cond] = node_delay[pnode * NUM_ATTR + cond] + t;
            if(debug_on)printf("net %s node %d parent %d res_parent:%.4f downcap:%E t:%.4f delay:%.4f\n", net_names[idx], node, pnode, res_parent[node], down_cap[node * NUM_ATTR + cond], t, node_delay[node * NUM_ATTR + cond]);
            if(pin != -1){
                if(debug_on)printf("net %s pin %s id:%d elmore:%.4f t:%.4f root=%d node=%d downcap:%E res:%E\n", net_names[idx], pin_names[pin], pin, node_delay[node * NUM_ATTR + cond], t, root, node, down_cap[node * NUM_ATTR + cond], res_parent[node]);
                elmore_delay[pin * NUM_ATTR + cond] = node_delay[node * NUM_ATTR + cond];
            }
        }
    }        
}

__global__ void propagate_rc_dmp(dmp_model* dmp_rc_){
    dmp_rc_->propagate_dmp_rc();
}

void calc_res_cap_dmp(dmp_model* dmp_rc_, int num_nets){
    // Implementation of the function using dmp_rc_
    int thread_count = 64;
    calc_dmp_rc<<<num_nets, thread_count>>>(dmp_rc_);
}

// void flatten_rc_tree_dmp(dmp_model* dmp_rc_){
//     // Implementation of the function using dmp_rc_
//     int thread_count = 64;
//     flatten_rc_dmp<<<dmp_rc_ -> num_nets, thread_count>>>(dmp_rc_);
// }


void propagate_rc_tree_dmp(dmp_model* dmp_rc_, int num_nets){
    // Implementation of the function using dmp_rc_
    int thread_count = 64;
    dim3 block_size(thread_count, NUM_ATTR);
    int num_blocks = (num_nets + thread_count - 1) / thread_count;
    propagate_rc_dmp<<<num_blocks, block_size>>>(dmp_rc_);

}

}