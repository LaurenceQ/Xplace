#include "DmpCeff.h"
#include "GPUTimer.h"
#include "utils.cuh"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s,\nat %s, line %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static void clear_stale_cuda_error(const char* label)
{
    cudaError_t stale_error = cudaGetLastError();
    if (stale_error != cudaSuccess) {
        fprintf(stderr, "[DMP CUDA] cleared stale CUDA error before %s: %s\n",
                label, cudaGetErrorString(stale_error));
    }
}

namespace gt{
static bool dmp_rc_env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

static bool dmp_rc_kernel_profile_enabled()
{
    return dmp_rc_env_enabled("DMP_PROFILE_KERNELS");
}

static bool dmp_rc_profile_enabled()
{
    return dmp_rc_kernel_profile_enabled() ||
           dmp_rc_env_enabled("DMP_RC_PROFILE") ||
           dmp_rc_env_enabled("DMP_DEBUG_TIMING");
}

static int dmp_rc_hist_bucket(int degree)
{
    if (degree <= 0) return 0;
    if (degree == 1) return 1;
    if (degree == 2) return 2;
    if (degree <= 4) return 3;
    if (degree <= 8) return 4;
    if (degree <= 16) return 5;
    return 6;
}

static void print_dmp_rc_parallel_stats(const std::vector<int>& net2node_start,
                                        const std::vector<int>& net2edge_start,
                                        int num_nets,
                                        int num_nodes,
                                        int num_edges)
{
    unsigned long long node_hist[7] = {0};
    unsigned long long edge_hist[7] = {0};
    int max_nodes = 0;
    int max_nodes_net = -1;
    int max_edges = 0;
    int max_edges_net = -1;
    unsigned long long node_work = 0;
    unsigned long long edge_work = 0;
    for (int net = 0; net < num_nets; ++net) {
        const int nodes = net2node_start[net + 1] - net2node_start[net];
        const int edges = net2edge_start[net + 1] - net2edge_start[net];
        node_hist[dmp_rc_hist_bucket(nodes)]++;
        edge_hist[dmp_rc_hist_bucket(edges)]++;
        node_work += static_cast<unsigned long long>(nodes) * NUM_ATTR;
        edge_work += static_cast<unsigned long long>(edges) * NUM_ATTR;
        if (nodes > max_nodes) {
            max_nodes = nodes;
            max_nodes_net = net;
        }
        if (edges > max_edges) {
            max_edges = edges;
            max_edges_net = net;
        }
    }
    printf("[DMP RC PARALLEL STATS] nets=%d nodes=%d edges=%d node_attr_work=%llu edge_attr_work=%llu\n",
           num_nets, num_nodes, num_edges, node_work, edge_work);
    printf("[DMP RC PARALLEL STATS] nodes_per_net[0,1,2,3-4,5-8,9-16,17+]=%llu,%llu,%llu,%llu,%llu,%llu,%llu max_nodes=%d@net%d edges_per_net[0,1,2,3-4,5-8,9-16,17+]=%llu,%llu,%llu,%llu,%llu,%llu,%llu max_edges=%d@net%d\n",
           node_hist[0], node_hist[1], node_hist[2], node_hist[3],
           node_hist[4], node_hist[5], node_hist[6], max_nodes, max_nodes_net,
           edge_hist[0], edge_hist[1], edge_hist[2], edge_hist[3],
           edge_hist[4], edge_hist[5], edge_hist[6], max_edges, max_edges_net);
}

static void print_dmp_rc_kernel_profile(const char* name,
                                        int launches,
                                        int launch_blocks,
                                        int block_x,
                                        int block_y,
                                        long long work_items,
                                        int num_nets,
                                        int num_nodes,
                                        int num_edges,
                                        float elapsed_ms)
{
    const double avg_us = launches > 0
                              ? static_cast<double>(elapsed_ms) * 1000.0 /
                                    static_cast<double>(launches)
                              : 0.0;
    const double work_per_ms = elapsed_ms > 0.0f
                                   ? static_cast<double>(work_items) /
                                         static_cast<double>(elapsed_ms)
                                   : 0.0;
    printf("[DMP KERNEL PROFILE] name=%s launches=%d total_ms=%.3f avg_us=%.3f max_ms=%.3f work_items=%lld blocks=%d block=(%d,%d) work_per_ms=%.1f rc_nets=%d rc_nodes=%d rc_edges=%d\n",
           name,
           launches,
           elapsed_ms,
           avg_us,
           elapsed_ms,
           work_items,
           launch_blocks,
           block_x,
           block_y,
           work_per_ms,
           num_nets,
           num_nodes,
           num_edges);
    fflush(stdout);
}

__host__ void dmp_model::initialize_rc(const std::vector<int>& host_edge_from,
               const std::vector<int>& host_edge_to,
               const std::vector<int>& host_flat_net2node_start_map,
               const std::vector<int>& host_flat_net2edge_start_map,
               const std::vector<int>& host_node2pin_map,
               const std::vector<float>& host_edge_wl,
               float *pinCap_,
               int num_nets_,
               int num_nodes_,
               int num_edges_,
               float unit_to_micron_,
               float rf_,
               float cf_){
        if (dmp_rc_profile_enabled()) {
            print_dmp_rc_parallel_stats(host_flat_net2node_start_map,
                                        host_flat_net2edge_start_map,
                                        num_nets_,
                                        num_nodes_,
                                        num_edges_);
        }
        pinCap = pinCap_;
        num_nets = num_nets_;
        num_nodes = num_nodes_;
        num_edges = num_edges_;
        unit_to_micron = unit_to_micron_;
        rf = rf_;
        cf = cf_;
        explicit_rc = false;
        cudaMalloc(&edge_from, host_edge_from.size() * sizeof(int));
        cudaMalloc(&edge_to, host_edge_to.size() * sizeof(int));
        cudaMalloc(&flat_net2node_start_map, host_flat_net2node_start_map.size() * sizeof(int));
        cudaMalloc(&flat_net2edge_start_map, host_flat_net2edge_start_map.size() * sizeof(int));
        cudaMalloc(&node2pin_map, host_node2pin_map.size() * sizeof(int));
        cudaMalloc(&edge_wl, host_edge_wl.size() * sizeof(float));    
        cudaMalloc(&edge_res, num_edges * sizeof(float));
        cudaMalloc(&includes_pin_caps, num_nets * sizeof(uint8_t));

        cudaMemcpy(edge_from, host_edge_from.data(), host_edge_from.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(edge_to, host_edge_to.data(), host_edge_to.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(flat_net2node_start_map, host_flat_net2node_start_map.data(), host_flat_net2node_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(flat_net2edge_start_map, host_flat_net2edge_start_map.data(), host_flat_net2edge_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(node2pin_map, host_node2pin_map.data(), host_node2pin_map.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(edge_wl, host_edge_wl.data(), host_edge_wl.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(edge_res, 0, num_edges * sizeof(float));
        cudaMemset(includes_pin_caps, 0, num_nets * sizeof(uint8_t));
        
        cudaMalloc(&root_dist, num_nodes * sizeof(int));
        cudaMalloc(&cnts, num_nodes * sizeof(int));
        cudaMalloc(&node_order, num_nodes * sizeof(int));
        cudaMalloc(&parent_node, num_nodes * sizeof(int));
        cudaMalloc(&res_parent, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&node_cap, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&node_delay, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&y1, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&y2, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&y3, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&down_cap, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&elmore_delay, num_pins * NUM_ATTR * sizeof(float));
        cudaMalloc(&C1, dmp_slot_capacity * sizeof(double));
        cudaMalloc(&C2, dmp_slot_capacity * sizeof(double));
        cudaMalloc(&r_pi, dmp_slot_capacity * sizeof(double));

        cudaMemset(cnts, 0, num_nodes * sizeof(int));
        cudaMemset(node_order, 0, num_nodes * sizeof(int));
        cudaMemset(parent_node, -1, num_nodes * sizeof(int));
        cudaMemset(res_parent, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(node_cap, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(node_delay, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(y1, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(y2, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(y3, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(down_cap, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(elmore_delay, 0, num_pins * NUM_ATTR * sizeof(float));
        cudaMemset(C1, 0, num_pins * NUM_ATTR * sizeof(double));
        cudaMemset(C2, 0, num_pins * NUM_ATTR * sizeof(double));
        cudaMemset(r_pi, 0, num_pins * NUM_ATTR * sizeof(double));
        cudaMemset(root_dist, -1, num_nodes * sizeof(int));
      }

__host__ void dmp_model::initialize_rc_explicit(
               const std::vector<int>& host_edge_from,
               const std::vector<int>& host_edge_to,
               const std::vector<int>& host_flat_net2node_start_map,
               const std::vector<int>& host_flat_net2edge_start_map,
               const std::vector<int>& host_node2pin_map,
               const std::vector<float>& host_edge_res,
               const std::vector<float>& host_node_cap,
               const std::vector<uint8_t>& host_includes_pin_caps,
               float *pinCap_,
               int num_nets_,
               int num_nodes_,
               int num_edges_){
        if (dmp_rc_profile_enabled()) {
            print_dmp_rc_parallel_stats(host_flat_net2node_start_map,
                                        host_flat_net2edge_start_map,
                                        num_nets_,
                                        num_nodes_,
                                        num_edges_);
        }
        pinCap = pinCap_;
        num_nets = num_nets_;
        num_nodes = num_nodes_;
        num_edges = num_edges_;
        unit_to_micron = 1.0f;
        rf = 0.0f;
        cf = 0.0f;
        explicit_rc = true;
        if (host_edge_res.size() != static_cast<size_t>(num_edges)) {
            throw std::runtime_error("initialize_rc_explicit edge_res must have num_edges values.");
        }
        cudaMalloc(&edge_from, host_edge_from.size() * sizeof(int));
        cudaMalloc(&edge_to, host_edge_to.size() * sizeof(int));
        cudaMalloc(&flat_net2node_start_map, host_flat_net2node_start_map.size() * sizeof(int));
        cudaMalloc(&flat_net2edge_start_map, host_flat_net2edge_start_map.size() * sizeof(int));
        cudaMalloc(&node2pin_map, host_node2pin_map.size() * sizeof(int));
        cudaMalloc(&edge_res, host_edge_res.size() * sizeof(float));
        cudaMalloc(&node_cap, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&includes_pin_caps, num_nets * sizeof(uint8_t));
        edge_wl = nullptr;

        cudaMemcpy(edge_from, host_edge_from.data(), host_edge_from.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(edge_to, host_edge_to.data(), host_edge_to.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(flat_net2node_start_map, host_flat_net2node_start_map.data(), host_flat_net2node_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(flat_net2edge_start_map, host_flat_net2edge_start_map.data(), host_flat_net2edge_start_map.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(node2pin_map, host_node2pin_map.data(), host_node2pin_map.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(edge_res, host_edge_res.data(), host_edge_res.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(node_cap, host_node_cap.data(), num_nodes * NUM_ATTR * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(includes_pin_caps, host_includes_pin_caps.data(), num_nets * sizeof(uint8_t), cudaMemcpyHostToDevice);

        cudaMalloc(&root_dist, num_nodes * sizeof(int));
        cudaMalloc(&cnts, num_nodes * sizeof(int));
        cudaMalloc(&node_order, num_nodes * sizeof(int));
        cudaMalloc(&parent_node, num_nodes * sizeof(int));
        cudaMalloc(&res_parent, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&node_delay, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&y1, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&y2, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&y3, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&down_cap, num_nodes * NUM_ATTR * sizeof(float));
        cudaMalloc(&elmore_delay, num_pins * NUM_ATTR * sizeof(float));
        cudaMalloc(&C1, dmp_slot_capacity * sizeof(double));
        cudaMalloc(&C2, dmp_slot_capacity * sizeof(double));
        cudaMalloc(&r_pi, dmp_slot_capacity * sizeof(double));

        cudaMemset(cnts, 0, num_nodes * sizeof(int));
        cudaMemset(node_order, 0, num_nodes * sizeof(int));
        cudaMemset(parent_node, -1, num_nodes * sizeof(int));
        cudaMemset(res_parent, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(node_delay, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(y1, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(y2, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(y3, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(down_cap, 0, num_nodes * NUM_ATTR * sizeof(float));
        cudaMemset(elmore_delay, 0, num_pins * NUM_ATTR * sizeof(float));
        cudaMemset(C1, 0, num_pins * NUM_ATTR * sizeof(double));
        cudaMemset(C2, 0, num_pins * NUM_ATTR * sizeof(double));
        cudaMemset(r_pi, 0, num_pins * NUM_ATTR * sizeof(double));
        cudaMemset(root_dist, -1, num_nodes * sizeof(int));
      }

void GPUTimer::initialize_dmp_rc(
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

void GPUTimer::initialize_dmp_rc_explicit(
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
                  int num_edges){
    const float rc_time_factor = (res_unit * cap_unit) / time_unit();
    for (float& res : host_edge_res) {
        res *= rc_time_factor;
    }
    logger.info("DMP explicit RC time scale: res_unit=%.5E cap_unit=%.5E time_unit=%.5E factor=%.5E",
                res_unit, cap_unit, time_unit(), rc_time_factor);
    h_dmp_db = new dmp_model(this);
    h_dmp_db -> initialize_rc_explicit(host_edge_from,
                                       host_edge_to,
                                       host_flat_net2node_start_map,
                                       host_flat_net2edge_start_map,
                                       host_node2pin_map,
                                       host_edge_res,
                                       host_node_cap,
                                       host_includes_pin_caps,
                                       pinCap,
                                       num_nets,
                                       num_nodes,
                                       num_edges);
    cudaMalloc(&dmp_db, sizeof(dmp_model));
    cudaMemcpy(dmp_db, h_dmp_db, sizeof(dmp_model), cudaMemcpyHostToDevice);
   
}

__device__ __forceinline__ void dmp_model::calc_dmp_rc(){
    const int idx = blockIdx.x;
    if (idx < num_nets) {
        int nst = flat_net2node_start_map[idx];
        int nend = flat_net2node_start_map[idx + 1];
        if (nst >= nend) {
            return;
        }
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
                if (from == to) {
                    continue;
                }
                float cap = 0.0f;
                float res = edge_res[i];
                if (!explicit_rc) {
                    float wl = edge_wl[i];
                    cap = wl * cf * 0.5f / unit_to_micron; // unit * fF/um * um/unit = fF
                    res = wl * rf / unit_to_micron; // unit * kohm/um * um/unit = kohm
                    edge_res[i] = res;
                }
                if ((root_dist[from] == d) && (root_dist[to] == -1)) {
                    root_dist[to] = d + 1;
                    parent_node[to] = from;
                    for (int attr = 0; attr < NUM_ATTR; ++attr) {
                        res_parent[to * NUM_ATTR + attr] = res;
                    }
                    int order = atomicAdd(&cnts[d + nst], 1);
                    if(debug_on)printf("net:%s nst:%d offset:%d order:%d from:%d to:%d res:%.4f, cap:%.4f\n", net_names[idx], nst, offset, order, from, to, res, cap);
                    node_order[nst + offset + order] = to;
                    if (!explicit_rc) {
                        for (int attr = 0; attr < NUM_ATTR; ++attr) {
                            atomicAdd(&node_cap[to * NUM_ATTR + attr], cap);
                            atomicAdd(&node_cap[from * NUM_ATTR + attr], cap);
                        }
                    }

                } else if ((root_dist[to] == d) && (root_dist[from] == -1)) {
                    parent_node[from] = to;
                    root_dist[from] = d + 1;
                    for (int attr = 0; attr < NUM_ATTR; ++attr) {
                        res_parent[from * NUM_ATTR + attr] = res;
                    }
                    int order = atomicAdd(&cnts[d + nst], 1);
                    if(debug_on)printf("net:%s nst:%d offset:%d order:%d from:%d to:%d res:%.4f, cap:%.4f\n", net_names[idx], nst, offset, order, to, from, res, cap);
                    node_order[nst + offset + order] = from;
                    if (!explicit_rc) {
                        for (int attr = 0; attr < NUM_ATTR; ++attr) {
                            atomicAdd(&node_cap[to * NUM_ATTR + attr], cap);
                            atomicAdd(&node_cap[from * NUM_ATTR + attr], cap);
                        }
                    }
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
        if (nst >= nend) {
            return;
        }
        // printf("Propagate RC for net %d from node %d to %d\n", idx, nst, nend);
       for (int i = nend - 1; i >= nst; i--) {
            
            int node = node_order[i];
            int pnode = parent_node[node];
            int pin = node2pin_map[node];
            float wire_cap = node_cap[node * NUM_ATTR + cond];
            // printf("  Node %d, Parent %d, Pin %d, Wire cap %.6f\n", node, pnode, pin, wire_cap);
            if (pin != -1 && (!includes_pin_caps || includes_pin_caps[idx] == 0)) {
                float pin_cap_lib = pin_cap_attr(pinCap, pin, cond);
                wire_cap = wire_cap + pin_cap_lib;
            }
            y1[node * NUM_ATTR + cond] += wire_cap;
            down_cap[node * NUM_ATTR + cond] += wire_cap;
            if (pnode != -1){
                const float parent_res = res_parent[node * NUM_ATTR + cond];
                down_cap[pnode * NUM_ATTR + cond] += down_cap[node * NUM_ATTR + cond];
                y1[pnode * NUM_ATTR + cond] += y1[node * NUM_ATTR + cond];
                y2[pnode * NUM_ATTR + cond] += y2[node * NUM_ATTR + cond] - parent_res * y1[node * NUM_ATTR + cond] * y1[node * NUM_ATTR + cond];
                y3[pnode * NUM_ATTR + cond] += y3[node * NUM_ATTR + cond] - 2 * parent_res * y1[node * NUM_ATTR + cond] * y2[node * NUM_ATTR + cond]
                            + parent_res * parent_res * y1[node * NUM_ATTR + cond] * y1[node * NUM_ATTR + cond] * y1[node * NUM_ATTR + cond];
            }
        }
        int root = node_order[nst];
        int root_pin = node2pin_map[node_order[nst]];

        if(debug_on)printf("reduce parasitics: net %s id:%d driv_pin=%s root=%d\n", net_names[idx], idx, pin_names[root_pin], root);

        double y1_root = y1[root * NUM_ATTR + cond];
        double y2_root = y2[root * NUM_ATTR + cond];
        double y3_root = y3[root * NUM_ATTR + cond];
        if(!isfinite(y1_root) || !isfinite(y2_root) || !isfinite(y3_root) ||
           (fabs(y2_root) < 1e-20 && fabs(y3_root) < 1e-20)){
            C1[root_pin * NUM_ATTR + cond] = isfinite(y1_root) && y1_root > 0.0 ? y1_root : 0.0;
            C2[root_pin * NUM_ATTR + cond] = 0.0;
            r_pi[root_pin * NUM_ATTR + cond] = 0.0;
        }
        else{
            double c1 = y2_root * y2_root / y3_root;
            double c2 = y1_root - c1;
            double rpi = -y3_root * y3_root / (y2_root * y2_root * y2_root);
            if (!isfinite(c1) || !isfinite(c2) || !isfinite(rpi) ||
                c1 <= 0.0 || c2 < 0.0 || rpi <= 0.0) {
                C1[root_pin * NUM_ATTR + cond] = isfinite(y1_root) && y1_root > 0.0 ? y1_root : 0.0;
                C2[root_pin * NUM_ATTR + cond] = 0.0;
                r_pi[root_pin * NUM_ATTR + cond] = 0.0;
            }
            else {
                C1[root_pin * NUM_ATTR + cond] = c1;
                C2[root_pin * NUM_ATTR + cond] = c2;
                r_pi[root_pin * NUM_ATTR + cond] = rpi;
            }
        }
        for (int i = nst + 1; i < nend; i++) {
            int node = node_order[i];
            int pnode = parent_node[node];
            int pin = node2pin_map[node];
            const float parent_res = res_parent[node * NUM_ATTR + cond];
            float t = down_cap[node * NUM_ATTR + cond] * parent_res;
            if(pnode != -1)node_delay[node * NUM_ATTR + cond] = node_delay[pnode * NUM_ATTR + cond] + t;
            if(debug_on)printf("net %s node %d parent %d res_parent:%.4f downcap:%E t:%.4f delay:%.4f\n", net_names[idx], node, pnode, parent_res, down_cap[node * NUM_ATTR + cond], t, node_delay[node * NUM_ATTR + cond]);
            if(pin != -1){
                if(debug_on)printf("net %s pin %s id:%d elmore:%.4f t:%.4f root=%d node=%d downcap:%E res:%E\n", net_names[idx], pin_names[pin], pin, node_delay[node * NUM_ATTR + cond], t, root, node, down_cap[node * NUM_ATTR + cond], parent_res);
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
    clear_stale_cuda_error("calc_res_cap_dmp");
    const bool profile_kernels = dmp_rc_kernel_profile_enabled();
    dmp_model h_dmp;
    if (profile_kernels) {
        gpuErrchk(cudaMemcpy(&h_dmp, dmp_rc_, sizeof(dmp_model), cudaMemcpyDeviceToHost));
        h_dmp.owns_allocations = false;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (profile_kernels) {
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));
        gpuErrchk(cudaEventRecord(start));
    }
    calc_dmp_rc<<<num_nets, thread_count>>>(dmp_rc_);
    gpuErrchk(cudaPeekAtLastError());
    if (profile_kernels) {
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaDeviceSynchronize());
        float elapsed_ms = 0.0f;
        gpuErrchk(cudaEventElapsedTime(&elapsed_ms, start, stop));
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));
        cudaGetLastError();
        print_dmp_rc_kernel_profile("calc_dmp_rc",
                                    1,
                                    num_nets,
                                    thread_count,
                                    1,
                                    static_cast<long long>(h_dmp.num_edges),
                                    h_dmp.num_nets,
                                    h_dmp.num_nodes,
                                    h_dmp.num_edges,
                                    elapsed_ms);
    }
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
    clear_stale_cuda_error("propagate_rc_tree_dmp");
    const bool profile_kernels = dmp_rc_kernel_profile_enabled();
    dmp_model h_dmp;
    if (profile_kernels) {
        gpuErrchk(cudaMemcpy(&h_dmp, dmp_rc_, sizeof(dmp_model), cudaMemcpyDeviceToHost));
        h_dmp.owns_allocations = false;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (profile_kernels) {
        gpuErrchk(cudaEventCreate(&start));
        gpuErrchk(cudaEventCreate(&stop));
        gpuErrchk(cudaEventRecord(start));
    }
    propagate_rc_dmp<<<num_blocks, block_size>>>(dmp_rc_);
    gpuErrchk(cudaPeekAtLastError());
    if (profile_kernels) {
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaDeviceSynchronize());
        float elapsed_ms = 0.0f;
        gpuErrchk(cudaEventElapsedTime(&elapsed_ms, start, stop));
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));
        cudaGetLastError();
        print_dmp_rc_kernel_profile("propagate_rc_dmp",
                                    1,
                                    num_blocks,
                                    thread_count,
                                    NUM_ATTR,
                                    static_cast<long long>(h_dmp.num_nodes) * NUM_ATTR,
                                    h_dmp.num_nets,
                                    h_dmp.num_nodes,
                                    h_dmp.num_edges,
                                    elapsed_ms);
    }

}

void debug_dump_dmp_rc_net_cuda(dmp_model* h_dmp_db,
                                int net_id,
                                const std::vector<std::string>& net_names,
                                const std::vector<std::string>& pin_names){
    if (h_dmp_db == nullptr || net_id < 0 || net_id >= h_dmp_db->num_nets) {
        printf("[DMP RC DUMP] invalid net_id=%d\n", net_id);
        fflush(stdout);
        return;
    }
    clear_stale_cuda_error("debug_dump_dmp_rc_net_cuda");
    int nst = 0, nend = 0, est = 0, eend = 0;
    gpuErrchk(cudaMemcpy(&nst, h_dmp_db->flat_net2node_start_map + net_id, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&nend, h_dmp_db->flat_net2node_start_map + net_id + 1, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&est, h_dmp_db->flat_net2edge_start_map + net_id, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&eend, h_dmp_db->flat_net2edge_start_map + net_id + 1, sizeof(int), cudaMemcpyDeviceToHost));
    int node_count = nend - nst;
    int edge_count = eend - est;
    if (node_count <= 0) {
        printf("[DMP RC DUMP] net=%s id=%d empty\n", net_names[net_id].c_str(), net_id);
        fflush(stdout);
        return;
    }

    std::vector<int> node_order_host(node_count);
    std::vector<int> parent_host(node_count);
    std::vector<int> node2pin_host(node_count);
    std::vector<float> res_parent_host(node_count * NUM_ATTR);
    std::vector<float> node_cap_host(node_count * NUM_ATTR);
    std::vector<float> down_cap_host(node_count * NUM_ATTR);
    std::vector<float> y1_host(node_count * NUM_ATTR);
    std::vector<float> y2_host(node_count * NUM_ATTR);
    std::vector<float> y3_host(node_count * NUM_ATTR);
    std::vector<float> node_delay_host(node_count * NUM_ATTR);
    std::vector<int> edge_from_host(edge_count);
    std::vector<int> edge_to_host(edge_count);
    std::vector<float> edge_res_host(edge_count);

    gpuErrchk(cudaMemcpy(node_order_host.data(), h_dmp_db->node_order + nst, node_count * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(parent_host.data(), h_dmp_db->parent_node + nst, node_count * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(node2pin_host.data(), h_dmp_db->node2pin_map + nst, node_count * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(res_parent_host.data(), h_dmp_db->res_parent + nst * NUM_ATTR, node_count * NUM_ATTR * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(node_cap_host.data(), h_dmp_db->node_cap + nst * NUM_ATTR, node_count * NUM_ATTR * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(down_cap_host.data(), h_dmp_db->down_cap + nst * NUM_ATTR, node_count * NUM_ATTR * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(y1_host.data(), h_dmp_db->y1 + nst * NUM_ATTR, node_count * NUM_ATTR * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(y2_host.data(), h_dmp_db->y2 + nst * NUM_ATTR, node_count * NUM_ATTR * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(y3_host.data(), h_dmp_db->y3 + nst * NUM_ATTR, node_count * NUM_ATTR * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(node_delay_host.data(), h_dmp_db->node_delay + nst * NUM_ATTR, node_count * NUM_ATTR * sizeof(float), cudaMemcpyDeviceToHost));
    if (edge_count > 0) {
        gpuErrchk(cudaMemcpy(edge_from_host.data(), h_dmp_db->edge_from + est, edge_count * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(edge_to_host.data(), h_dmp_db->edge_to + est, edge_count * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(edge_res_host.data(),
                             h_dmp_db->edge_res + est,
                             edge_res_host.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
    }

    int root_pin = node2pin_host[0];
    double c1[NUM_ATTR] = {0.0}, c2[NUM_ATTR] = {0.0}, rpi[NUM_ATTR] = {0.0};
    if (root_pin >= 0) {
        gpuErrchk(cudaMemcpy(c1, h_dmp_db->C1 + root_pin * NUM_ATTR, NUM_ATTR * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(c2, h_dmp_db->C2 + root_pin * NUM_ATTR, NUM_ATTR * sizeof(double), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(rpi, h_dmp_db->r_pi + root_pin * NUM_ATTR, NUM_ATTR * sizeof(double), cudaMemcpyDeviceToHost));
    }
    uint8_t includes_pin_caps_flag = 0;
    if (h_dmp_db->includes_pin_caps) {
        gpuErrchk(cudaMemcpy(&includes_pin_caps_flag, h_dmp_db->includes_pin_caps + net_id, sizeof(uint8_t), cudaMemcpyDeviceToHost));
    }

    printf("[DMP RC DUMP] net=%s id=%d nodes=%d edges=%d root_pin=%d root_pin_name=%s explicit_rc=%d includes_pin_caps=%d\n",
           net_names[net_id].c_str(), net_id, node_count, edge_count, root_pin,
           root_pin >= 0 ? pin_names[root_pin].c_str() : "<none>",
           h_dmp_db->explicit_rc ? 1 : 0,
           h_dmp_db->includes_pin_caps ? static_cast<int>(includes_pin_caps_flag) : -1);
    printf("[DMP RC DUMP] root_pi C1=(%.9e,%.9e,%.9e,%.9e) C2=(%.9e,%.9e,%.9e,%.9e) rpi=(%.9e,%.9e,%.9e,%.9e)\n",
           c1[0], c1[1], c1[2], c1[3],
           c2[0], c2[1], c2[2], c2[3],
           rpi[0], rpi[1], rpi[2], rpi[3]);
    for (int edge = 0; edge < edge_count; ++edge) {
        printf("[DMP RC DUMP] edge local=%d from=%d to=%d res=%.9e\n",
               edge, edge_from_host[edge] - nst, edge_to_host[edge] - nst,
               edge_res_host[edge]);
    }
    for (int pos = 0; pos < node_count; ++pos) {
        int node = node_order_host[pos];
        int local = node - nst;
        if (local < 0 || local >= node_count) {
            continue;
        }
        int pin = node2pin_host[local];
        float elmore[NUM_ATTR] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (pin >= 0) {
            gpuErrchk(cudaMemcpy(elmore, h_dmp_db->elmore_delay + pin * NUM_ATTR, NUM_ATTR * sizeof(float), cudaMemcpyDeviceToHost));
        }
        printf("[DMP RC DUMP] node order=%d local=%d global=%d parent=%d pin=%d pin_name=%s res_parent=%.9e cap=(%.9e,%.9e,%.9e,%.9e) down=(%.9e,%.9e,%.9e,%.9e) delay=(%.9e,%.9e,%.9e,%.9e) elmore=(%.9e,%.9e,%.9e,%.9e) y1=(%.9e,%.9e,%.9e,%.9e) y2=(%.9e,%.9e,%.9e,%.9e) y3=(%.9e,%.9e,%.9e,%.9e)\n",
               pos, local, node,
               parent_host[local] >= 0 ? parent_host[local] - nst : -1,
               pin, pin >= 0 ? pin_names[pin].c_str() : "<none>",
               res_parent_host[local * NUM_ATTR + 0],
               node_cap_host[local * NUM_ATTR + 0], node_cap_host[local * NUM_ATTR + 1],
               node_cap_host[local * NUM_ATTR + 2], node_cap_host[local * NUM_ATTR + 3],
               down_cap_host[local * NUM_ATTR + 0], down_cap_host[local * NUM_ATTR + 1],
               down_cap_host[local * NUM_ATTR + 2], down_cap_host[local * NUM_ATTR + 3],
               node_delay_host[local * NUM_ATTR + 0], node_delay_host[local * NUM_ATTR + 1],
               node_delay_host[local * NUM_ATTR + 2], node_delay_host[local * NUM_ATTR + 3],
               elmore[0], elmore[1], elmore[2], elmore[3],
               y1_host[local * NUM_ATTR + 0], y1_host[local * NUM_ATTR + 1],
               y1_host[local * NUM_ATTR + 2], y1_host[local * NUM_ATTR + 3],
               y2_host[local * NUM_ATTR + 0], y2_host[local * NUM_ATTR + 1],
               y2_host[local * NUM_ATTR + 2], y2_host[local * NUM_ATTR + 3],
               y3_host[local * NUM_ATTR + 0], y3_host[local * NUM_ATTR + 1],
               y3_host[local * NUM_ATTR + 2], y3_host[local * NUM_ATTR + 3]);
    }
    gpuErrchk(cudaDeviceSynchronize());
    fflush(stdout);
}

}
