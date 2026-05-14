#include "DmpCudaUtils.cuh"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace gt {
__global__ void dmp_debug_count_kernel(DmpModel* dmp_db, unsigned long long* counts, int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int pin_total = dmp_db->num_pins * NUM_ATTR;
    const int arc_total = dmp_db->num_arcs * 2 * NUM_ATTR;
    const int test_total = dmp_db->num_tests * NUM_ATTR;
    if (idx < pin_total) {
        if (isfinite(dmp_db->pinSlew[idx])) atomicAdd(&counts[0], 1ULL);
        if (isfinite(dmp_db->pinAt[idx])) atomicAdd(&counts[1], 1ULL);
        if (isfinite(dmp_db->pinRat[idx])) atomicAdd(&counts[2], 1ULL);
        if (isfinite(dmp_db->elmore_delay[idx])) atomicAdd(&counts[3], 1ULL);
        if (isfinite(dmp_db->C1[idx])) atomicAdd(&counts[4], 1ULL);
        if (isfinite(dmp_db->C2[idx])) atomicAdd(&counts[5], 1ULL);
        if (isfinite(dmp_db->r_pi[idx])) atomicAdd(&counts[6], 1ULL);
    }
    if (idx < arc_total && isfinite(dmp_db->arcDelay[idx])) {
        atomicAdd(&counts[7], 1ULL);
    }
    if (idx < test_total) {
        if (isfinite(dmp_db->testRAT[idx])) atomicAdd(&counts[8], 1ULL);
        if (isfinite(dmp_db->testConstraint[idx])) atomicAdd(&counts[9], 1ULL);
    }
}

void dmp_debug_print_counts(DmpModel* dmp_db, const char* label) {
    DmpModel h_dmp;
    gpuErrchk(cudaMemcpy(&h_dmp, dmp_db, sizeof(DmpModel), cudaMemcpyDeviceToHost));
    h_dmp.owns_allocations = false;
    const int pin_total = h_dmp.num_pins * NUM_ATTR;
    const int arc_total = h_dmp.num_arcs * 2 * NUM_ATTR;
    const int test_total = h_dmp.num_tests * NUM_ATTR;
    const int total = std::max(pin_total, std::max(arc_total, test_total));
    const int blocks = DMP_DEBUG_BLOCK_NUMBER(total);
    unsigned long long* d_counts = nullptr;
    unsigned long long h_counts[10] = {0};
    if (pin_total <= 0 || arc_total <= 0 || test_total < 0 || total <= 0 || blocks <= 0) {
        fprintf(stderr, "[DMP DEBUG COUNTS] skip %s due to invalid dimensions\n", label);
        return;
    }
    gpuErrchk(cudaMalloc(&d_counts, sizeof(h_counts)));
    gpuErrchk(cudaMemset(d_counts, 0, sizeof(h_counts)));
    dmp_debug_count_kernel<<<blocks, DMP_DEBUG_BLOCK_SIZE>>>(dmp_db, d_counts, total);
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        fprintf(stderr, "[DMP DEBUG COUNTS] %s launch failed and was cleared: %s\n",
                label, cudaGetErrorString(launch_error));
        cudaFree(d_counts);
        cudaGetLastError();
        return;
    }
    cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error != cudaSuccess) {
        fprintf(stderr, "[DMP DEBUG COUNTS] %s sync failed and was cleared: %s\n",
                label, cudaGetErrorString(sync_error));
        cudaGetLastError();
        cudaFree(d_counts);
        cudaGetLastError();
        return;
    }
    gpuErrchk(cudaMemcpy(h_counts, d_counts, sizeof(h_counts), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_counts));
    cudaError_t post_count_error = cudaGetLastError();
    if (post_count_error != cudaSuccess) {
        fprintf(stderr, "[DMP DEBUG COUNTS] cleared post-count stale error at %s: %s\n",
                label, cudaGetErrorString(post_count_error));
    }
    printf("[DMP DEBUG COUNTS] %s pinSlew=%llu/%d pinAT=%llu/%d pinRAT=%llu/%d elmore=%llu/%d C1=%llu/%d C2=%llu/%d rpi=%llu/%d arcDelay=%llu/%d testRAT=%llu/%d testConstraint=%llu/%d\n",
           label,
           h_counts[0], pin_total,
           h_counts[1], pin_total,
           h_counts[2], pin_total,
           h_counts[3], pin_total,
           h_counts[4], pin_total,
           h_counts[5], pin_total,
           h_counts[6], pin_total,
           h_counts[7], arc_total,
           h_counts[8], test_total,
           h_counts[9], test_total);
}

void dmp_debug_print_first_level_sample(DmpModel* dmp_db, int level_idx, index_type level_start_offset) {
    DmpModel h_dmp;
    gpuErrchk(cudaMemcpy(&h_dmp, dmp_db, sizeof(DmpModel), cudaMemcpyDeviceToHost));
    h_dmp.owns_allocations = false;
    index_type to_pin = -1;
    index_type arc_start = -1;
    index_type arc_end = -1;
    index_type arc_id = -1;
    index_type from_pin = -1;
    int arc_type = -1;
    gpuErrchk(cudaMemcpy(&to_pin, h_dmp.level_list + level_start_offset, sizeof(index_type), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&arc_start, h_dmp.pin_backward_arc_list_end + to_pin, sizeof(index_type), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&arc_end, h_dmp.pin_backward_arc_list_end + to_pin + 1, sizeof(index_type), cudaMemcpyDeviceToHost));
    if (arc_start < arc_end) {
        gpuErrchk(cudaMemcpy(&arc_id, h_dmp.pin_backward_arc_list + arc_start, sizeof(index_type), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&from_pin, h_dmp.timing_arc_from_pin_id + arc_id, sizeof(index_type), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&arc_type, h_dmp.arc_types + arc_id, sizeof(int), cudaMemcpyDeviceToHost));
    }
    float from_slew[NUM_ATTR];
    float to_slew[NUM_ATTR];
    float to_at[NUM_ATTR];
    float elmore[NUM_ATTR];
    float delay[2 * NUM_ATTR];
    for (int a = 0; a < NUM_ATTR; ++a) {
        from_slew[a] = to_slew[a] = to_at[a] = elmore[a] = nanf("");
    }
    for (int a = 0; a < 2 * NUM_ATTR; ++a) {
        delay[a] = nanf("");
    }
    if (from_pin >= 0) {
        gpuErrchk(cudaMemcpy(from_slew, h_dmp.pinSlew + from_pin * NUM_ATTR, sizeof(float) * NUM_ATTR, cudaMemcpyDeviceToHost));
    }
    if (to_pin >= 0) {
        gpuErrchk(cudaMemcpy(to_slew, h_dmp.pinSlew + to_pin * NUM_ATTR, sizeof(float) * NUM_ATTR, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(to_at, h_dmp.pinAt + to_pin * NUM_ATTR, sizeof(float) * NUM_ATTR, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(elmore, h_dmp.elmore_delay + to_pin * NUM_ATTR, sizeof(float) * NUM_ATTR, cudaMemcpyDeviceToHost));
    }
    if (arc_id >= 0) {
        gpuErrchk(cudaMemcpy(delay, h_dmp.arcDelay + arc_id * 2 * NUM_ATTR, sizeof(float) * 2 * NUM_ATTR, cudaMemcpyDeviceToHost));
    }
    printf("[DMP DEBUG SAMPLE] level=%d to_pin=%d arc_range=[%d,%d) arc=%d type=%d from=%d\n",
           level_idx, (int)to_pin, (int)arc_start, (int)arc_end, (int)arc_id, arc_type, (int)from_pin);
    printf("[DMP DEBUG SAMPLE] from_slew=(%e,%e,%e,%e) to_slew=(%e,%e,%e,%e) to_at=(%e,%e,%e,%e) elmore=(%e,%e,%e,%e)\n",
           from_slew[0], from_slew[1], from_slew[2], from_slew[3],
           to_slew[0], to_slew[1], to_slew[2], to_slew[3],
           to_at[0], to_at[1], to_at[2], to_at[3],
           elmore[0], elmore[1], elmore[2], elmore[3]);
    printf("[DMP DEBUG SAMPLE] arcDelay=(%e,%e,%e,%e,%e,%e,%e,%e)\n",
           delay[0], delay[1], delay[2], delay[3], delay[4], delay[5], delay[6], delay[7]);
    cudaError_t post_sample_error = cudaGetLastError();
    if (post_sample_error != cudaSuccess) {
        fprintf(stderr, "[DMP DEBUG SAMPLE] cleared post-sample stale error: %s\n",
                cudaGetErrorString(post_sample_error));
    }
}

static int dmp_hist_bucket(int degree) {
    if (degree <= 0) return 0;
    if (degree == 1) return 1;
    if (degree == 2) return 2;
    if (degree <= 4) return 3;
    if (degree <= 8) return 4;
    if (degree <= 16) return 5;
    return 6;
}

void dmp_debug_print_parallel_stats(DmpModel* dmp_db,
                                    const vector<int>& level_list_end_cpu,
                                    const char* label) {
    DmpModel h_dmp;
    gpuErrchk(cudaMemcpy(&h_dmp, dmp_db, sizeof(DmpModel), cudaMemcpyDeviceToHost));
    h_dmp.owns_allocations = false;
    if (h_dmp.num_pins <= 0 || h_dmp.num_arcs < 0 || level_list_end_cpu.empty()) {
        printf("[DMP PARALLEL STATS] %s skip invalid dimensions pins=%d arcs=%d levels=%zu\n",
               label, h_dmp.num_pins, h_dmp.num_arcs, level_list_end_cpu.size());
        return;
    }

    vector<index_type> level_list(level_list_end_cpu.back());
    vector<index_type> fanin_end(h_dmp.num_pins + 1);
    vector<index_type> fanout_end(h_dmp.num_pins + 1);
    vector<int> arc_types(h_dmp.num_arcs);
    vector<int> arc_id2test_id(h_dmp.num_arcs);
    if (!level_list.empty()) {
        gpuErrchk(cudaMemcpy(level_list.data(), h_dmp.level_list,
                             sizeof(index_type) * level_list.size(),
                             cudaMemcpyDeviceToHost));
    }
    gpuErrchk(cudaMemcpy(fanin_end.data(), h_dmp.pin_backward_arc_list_end,
                         sizeof(index_type) * fanin_end.size(),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(fanout_end.data(), h_dmp.pin_forward_arc_list_end,
                         sizeof(index_type) * fanout_end.size(),
                         cudaMemcpyDeviceToHost));
    if (h_dmp.num_arcs > 0) {
        gpuErrchk(cudaMemcpy(arc_types.data(), h_dmp.arc_types,
                             sizeof(int) * arc_types.size(),
                             cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(arc_id2test_id.data(), h_dmp.arc_id2test_id,
                             sizeof(int) * arc_id2test_id.size(),
                             cudaMemcpyDeviceToHost));
    }

    unsigned long long fanin_hist[7] = {0};
    unsigned long long fanout_hist[7] = {0};
    int max_fanin = 0;
    int max_fanin_pin = -1;
    int max_fanout = 0;
    int max_fanout_pin = -1;
    for (int pin = 0; pin < h_dmp.num_pins; ++pin) {
        const int fanin = fanin_end[pin + 1] - fanin_end[pin];
        const int fanout = fanout_end[pin + 1] - fanout_end[pin];
        fanin_hist[dmp_hist_bucket(fanin)]++;
        fanout_hist[dmp_hist_bucket(fanout)]++;
        if (fanin > max_fanin) {
            max_fanin = fanin;
            max_fanin_pin = pin;
        }
        if (fanout > max_fanout) {
            max_fanout = fanout;
            max_fanout_pin = pin;
        }
    }

    unsigned long long forward_pin_work = 0;
    unsigned long long backward_pin_work = 0;
    unsigned long long forward_arc_work = 0;
    unsigned long long backward_arc_work = 0;
    int nonempty_forward_levels = 0;
    int nonempty_backward_levels = 0;
    int max_forward_pins = 0;
    int max_forward_pin_level = -1;
    int max_backward_pins = 0;
    int max_backward_pin_level = -1;
    unsigned long long max_forward_arc_work = 0;
    int max_forward_arc_level = -1;
    unsigned long long max_backward_arc_work = 0;
    int max_backward_arc_level = -1;
    const int num_levels = static_cast<int>(level_list_end_cpu.size()) - 1;
    for (int level = 0; level < num_levels; ++level) {
        const int start = level_list_end_cpu[level];
        const int end = level_list_end_cpu[level + 1];
        const int pins = end - start;
        if (pins <= 0) continue;
        unsigned long long level_fanin_arcs = 0;
        unsigned long long level_fanout_arcs = 0;
        for (int pos = start; pos < end; ++pos) {
            const int pin = level_list[pos];
            level_fanin_arcs += fanin_end[pin + 1] - fanin_end[pin];
            level_fanout_arcs += fanout_end[pin + 1] - fanout_end[pin];
        }
        if (level >= 1 && level < num_levels - 1) {
            nonempty_forward_levels++;
            forward_pin_work += static_cast<unsigned long long>(pins) * DMP_PIN_GROUP_SIZE;
            forward_arc_work += level_fanin_arcs * DMP_PIN_GROUP_SIZE;
            if (pins > max_forward_pins) {
                max_forward_pins = pins;
                max_forward_pin_level = level;
            }
            if (level_fanin_arcs > max_forward_arc_work) {
                max_forward_arc_work = level_fanin_arcs;
                max_forward_arc_level = level;
            }
        }
        if (level <= num_levels - 3) {
            nonempty_backward_levels++;
            backward_pin_work += static_cast<unsigned long long>(pins) * DMP_PIN_GROUP_SIZE;
            backward_arc_work += level_fanout_arcs * DMP_PIN_GROUP_SIZE;
            if (pins > max_backward_pins) {
                max_backward_pins = pins;
                max_backward_pin_level = level;
            }
            if (level_fanout_arcs > max_backward_arc_work) {
                max_backward_arc_work = level_fanout_arcs;
                max_backward_arc_level = level;
            }
        }
    }

    unsigned long long net_arcs = 0;
    unsigned long long gate_arcs = 0;
    unsigned long long other_arcs = 0;
    unsigned long long test_arcs = 0;
    for (int arc = 0; arc < h_dmp.num_arcs; ++arc) {
        if (arc_types[arc] == 0) net_arcs++;
        else if (arc_types[arc] == 1) gate_arcs++;
        else other_arcs++;
        if (arc_id2test_id[arc] != -1) test_arcs++;
    }

    unsigned long long alg_cap = 0;
    unsigned long long alg_zero_c2 = 0;
    unsigned long long alg_pi = 0;
    unsigned long long alg_other = 0;

    printf("[DMP PARALLEL STATS] %s levels=%d pins=%d arcs=%d tests=%d block=%d group=%d nonempty_fwd=%d nonempty_bwd=%d\n",
           label, num_levels, h_dmp.num_pins, h_dmp.num_arcs, h_dmp.num_tests,
           DMP_TIMING_BLOCK_SIZE, DMP_PIN_GROUP_SIZE,
           nonempty_forward_levels, nonempty_backward_levels);
    printf("[DMP PARALLEL STATS] %s fwd_pin_work=%llu fwd_arc_work=%llu max_fwd_pins=%d@L%d max_fwd_arcs=%llu@L%d bwd_pin_work=%llu bwd_arc_work=%llu max_bwd_pins=%d@L%d max_bwd_arcs=%llu@L%d\n",
           label, forward_pin_work, forward_arc_work,
           max_forward_pins, max_forward_pin_level,
           max_forward_arc_work, max_forward_arc_level,
           backward_pin_work, backward_arc_work,
           max_backward_pins, max_backward_pin_level,
           max_backward_arc_work, max_backward_arc_level);
    printf("[DMP PARALLEL STATS] %s fanin_hist[0,1,2,3-4,5-8,9-16,17+]=%llu,%llu,%llu,%llu,%llu,%llu,%llu max_fanin=%d@pin%d fanout_hist[0,1,2,3-4,5-8,9-16,17+]=%llu,%llu,%llu,%llu,%llu,%llu,%llu max_fanout=%d@pin%d\n",
           label,
           fanin_hist[0], fanin_hist[1], fanin_hist[2], fanin_hist[3],
           fanin_hist[4], fanin_hist[5], fanin_hist[6],
           max_fanin, max_fanin_pin,
           fanout_hist[0], fanout_hist[1], fanout_hist[2], fanout_hist[3],
           fanout_hist[4], fanout_hist[5], fanout_hist[6],
           max_fanout, max_fanout_pin);
    printf("[DMP PARALLEL STATS] %s arc_types net=%llu gate=%llu other=%llu test=%llu dmp_alg_available=%d CAP=%llu ZERO_C2=%llu PI=%llu other=%llu\n",
           label, net_arcs, gate_arcs, other_arcs, test_arcs,
           0, alg_cap, alg_zero_c2, alg_pi, alg_other);
    dmp_clear_stale_cuda_error("parallel stats");
}


} // namespace gt
