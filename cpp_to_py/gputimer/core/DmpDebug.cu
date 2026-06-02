#include "DmpGateModel.cuh"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace gt {

#ifndef DMP_DIRECT_CLOCK_DEBUG_PRINT
#define DMP_DIRECT_CLOCK_DEBUG_PRINT 0
#endif

#ifndef DMP_DRIVING_CELL_DEBUG_PRINT
#define DMP_DRIVING_CELL_DEBUG_PRINT 0
#endif

#ifndef DMP_ROOT_SOLVE_PROFILE
#define DMP_ROOT_SOLVE_PROFILE 0
#endif

enum DmpRootProfileCounter {
    DMP_ROOT_VO_CALLS = 0,
    DMP_ROOT_VO_SUCCESS = 1,
    DMP_ROOT_VO_ITERS = 2,
    DMP_ROOT_VO_BRACKET_FAIL = 3,
    DMP_ROOT_VO_ENDPOINT_HIT = 4,
    DMP_ROOT_VO_MAXITER_FAIL = 5,
    DMP_ROOT_ONEPOLE_CALLS = 6,
    DMP_ROOT_ONEPOLE_SUCCESS = 7,
    DMP_ROOT_ONEPOLE_ITERS = 8,
    DMP_ROOT_ONEPOLE_FAIL = 9,
    DMP_ROOT_PI_CALLS = 10,
    DMP_ROOT_PI_SUCCESS = 11,
    DMP_ROOT_PI_ITERS = 12,
    DMP_ROOT_PI_FAIL = 13,
    DMP_ROOT_PROFILE_COUNT = 14
};

#if DMP_ROOT_SOLVE_PROFILE
__device__ unsigned long long dmp_root_profile_counts[DMP_ROOT_PROFILE_COUNT];

__global__ void resetDmpRootProfileKernel()
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < DMP_ROOT_PROFILE_COUNT) {
        dmp_root_profile_counts[idx] = 0ULL;
    }
}

__global__ void copyDmpRootProfileKernel(unsigned long long* out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < DMP_ROOT_PROFILE_COUNT) {
        out[idx] = dmp_root_profile_counts[idx];
    }
}
#endif

__device__ void dmpRootProfileAddCounter(int counter, unsigned long long value) {
#if DMP_ROOT_SOLVE_PROFILE
    atomicAdd(&dmp_root_profile_counts[counter], value);
#else
    (void)counter;
    (void)value;
#endif
}

__device__ void dmpRootProfileVoCall() {
    dmpRootProfileAddCounter(DMP_ROOT_VO_CALLS, 1ULL);
}

__device__ void dmpRootProfileVoSuccess() {
    dmpRootProfileAddCounter(DMP_ROOT_VO_SUCCESS, 1ULL);
}

__device__ void dmpRootProfileVoIters(unsigned long long value) {
    dmpRootProfileAddCounter(DMP_ROOT_VO_ITERS, value);
}

__device__ void dmpRootProfileVoBracketFail() {
    dmpRootProfileAddCounter(DMP_ROOT_VO_BRACKET_FAIL, 1ULL);
}

__device__ void dmpRootProfileVoEndpointHit() {
    dmpRootProfileAddCounter(DMP_ROOT_VO_ENDPOINT_HIT, 1ULL);
}

__device__ void dmpRootProfileVoMaxIterFail() {
    dmpRootProfileAddCounter(DMP_ROOT_VO_MAXITER_FAIL, 1ULL);
}

void reset_dmp_root_profile_cuda()
{
#if DMP_ROOT_SOLVE_PROFILE
    resetDmpRootProfileKernel<<<1, 32>>>();
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif
}

void print_dmp_root_profile_cuda()
{
#if DMP_ROOT_SOLVE_PROFILE
    unsigned long long counts[DMP_ROOT_PROFILE_COUNT] = {};
    unsigned long long* d_counts = nullptr;
    gpuErrchk(cudaMalloc(&d_counts, sizeof(unsigned long long) * DMP_ROOT_PROFILE_COUNT));
    copyDmpRootProfileKernel<<<1, 32>>>(d_counts);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(counts,
                         d_counts,
                         sizeof(unsigned long long) * DMP_ROOT_PROFILE_COUNT,
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_counts));
    const auto avg = [](unsigned long long total, unsigned long long calls) {
        return calls == 0ULL ? 0.0 : static_cast<double>(total) / static_cast<double>(calls);
    };
    printf("[DMP ROOT PROFILE] vo calls=%llu success=%llu avg_iter_success=%.3f bracket_fail=%llu endpoint=%llu maxiter_fail=%llu\n",
           counts[DMP_ROOT_VO_CALLS],
           counts[DMP_ROOT_VO_SUCCESS],
           avg(counts[DMP_ROOT_VO_ITERS], counts[DMP_ROOT_VO_SUCCESS]),
           counts[DMP_ROOT_VO_BRACKET_FAIL],
           counts[DMP_ROOT_VO_ENDPOINT_HIT],
           counts[DMP_ROOT_VO_MAXITER_FAIL]);
    printf("[DMP ROOT PROFILE] onepole calls=%llu success=%llu avg_iter_success=%.3f fail=%llu\n",
           counts[DMP_ROOT_ONEPOLE_CALLS],
           counts[DMP_ROOT_ONEPOLE_SUCCESS],
           avg(counts[DMP_ROOT_ONEPOLE_ITERS], counts[DMP_ROOT_ONEPOLE_SUCCESS]),
           counts[DMP_ROOT_ONEPOLE_FAIL]);
    printf("[DMP ROOT PROFILE] pi calls=%llu success=%llu avg_iter_success=%.3f fail=%llu\n",
           counts[DMP_ROOT_PI_CALLS],
           counts[DMP_ROOT_PI_SUCCESS],
           avg(counts[DMP_ROOT_PI_ITERS], counts[DMP_ROOT_PI_SUCCESS]),
           counts[DMP_ROOT_PI_FAIL]);
#else
    printf("[DMP ROOT PROFILE] disabled at compile time; set DMP_ROOT_SOLVE_PROFILE=1 for a profiling build.\n");
#endif
    fflush(stdout);
}

__device__ void dmpGateNetPairCount(unsigned long long* counts, int counter) {
    if (counts != nullptr) {
        atomicAdd(&counts[counter], 1ULL);
    }
}

__device__ void dmpDrivingCellCount(unsigned long long* counts, int counter) {
    if (counts != nullptr) {
        atomicAdd(&counts[counter], 1ULL);
    }
}

__device__ __forceinline__ bool dmpDebugStringEquals(const char* lhs, const char* rhs) {
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    int idx = 0;
    while (lhs[idx] != '\0' && rhs[idx] != '\0') {
        if (lhs[idx] != rhs[idx]) {
            return false;
        }
        ++idx;
    }
    return lhs[idx] == rhs[idx];
}

__device__ void dmpDebugPrintDirectClock(const DmpModel* dmp_db,
                                         int from_pin_id,
                                         int to_pin_id,
                                         int from_slot,
                                         int attr,
                                         float source_slew,
                                         double final_slew,
                                         double elmore,
                                         double extra_delay,
                                         double vo_delay,
                                         double final_delay,
                                         int alg,
                                         bool used_dmp_load,
                                         bool used_driving_cell) {
    if (!DMP_DIRECT_CLOCK_DEBUG_PRINT ||
        dmp_db == nullptr ||
        dmp_db->pin_names == nullptr ||
        from_pin_id < 0 ||
        to_pin_id < 0 ||
        !dmpDebugStringEquals(dmp_db->pin_names[from_pin_id], "clk") ||
        !dmpDebugStringEquals(dmp_db->pin_names[to_pin_id], "clkbuf_0_clk:A")) {
        return;
    }
    const float src_at = dmp_db->pinAt != nullptr ? dmp_db->pinAt[from_slot] : nanf("");
    const double cand_at = isfinite(src_at) && isfinite(final_delay)
                               ? static_cast<double>(src_at) + final_delay
                               : nan("");
    printf("[DMP DIRECT CLOCK] attr=%d source_slew=%.9f load_slew=%.9f elmore=%.9f extra_delay=%.9f vo_delay=%.9f wire_delay=%.9f at=%.9f alg=%d dmp_load=%d driving_cell=%d\n",
           attr,
           static_cast<double>(source_slew),
           final_slew,
           elmore,
           extra_delay,
           vo_delay,
           final_delay,
           cand_at,
           alg,
           used_dmp_load ? 1 : 0,
           used_driving_cell ? 1 : 0);
}

__device__ void dmpDebugPrintDrivingCell(const DmpModel* dmp_db,
                                         int pin_id,
                                         int attr,
                                         float input_slew,
                                         const DmpDriverWave& driver_wave,
                                         float gate_delay,
                                         float intrinsic_delay,
                                         double extra_delay) {
    if (!DMP_DRIVING_CELL_DEBUG_PRINT ||
        dmp_db == nullptr ||
        dmp_db->pin_names == nullptr ||
        pin_id < 0 ||
        !dmpDebugStringEquals(dmp_db->pin_names[pin_id], "clk")) {
        return;
    }
    printf("[DMP DRIVING CELL DBG] pin=clk attr=%d alg=%d dmp_valid=%d input_slew=%.9f source_slew=%.9f gate_delay=%.9f intrinsic=%.9f extra=%.9f t0=%.9e dt=%.9e vo_upper=%.9e\n",
           attr,
           driver_wave.alg,
           driver_wave.hasValidDriver() ? 1 : 0,
           static_cast<double>(input_slew),
           driver_wave.vo_slew,
           static_cast<double>(gate_delay),
           static_cast<double>(intrinsic_delay),
           extra_delay,
           driver_wave.t0,
           driver_wave.dt,
           driver_wave.vo_upper_time);
}


void dmp_debug_print_driving_cell_counts(int num_sources,
                                         int total,
                                         const unsigned long long* counts) {
    const unsigned long long zeros[DMP_DRIVING_CELL_COUNTER_COUNT] = {};
    const unsigned long long* c = counts != nullptr ? counts : zeros;
    printf("[DMP DRIVING CELL] sources=%d lanes=%d applied=%llu skipped=%llu cap=%llu zero_c2=%llu pi=%llu dmp_valid=%llu fallback=%llu\n",
           num_sources,
           total,
           c[DMP_DRIVING_CELL_APPLIED],
           c[DMP_DRIVING_CELL_SKIPPED],
           c[DMP_DRIVING_CELL_CAP],
           c[DMP_DRIVING_CELL_ZERO_C2],
           c[DMP_DRIVING_CELL_PI],
           c[DMP_DRIVING_CELL_DMP_VALID],
           c[DMP_DRIVING_CELL_FALLBACK]);
}

void dmp_debug_print_driving_cell_kernel_profile(float elapsed_ms, int total) {
    printf("[DMP KERNEL PROFILE] name=applyDrivingCellSourceSlewKernel launches=1 total_ms=%.3f avg_us=%.3f max_ms=%.3f work_items=%d blocks=%d block=(%d,1) work_per_ms=%.1f\n",
           elapsed_ms,
           static_cast<double>(elapsed_ms) * 1000.0,
           elapsed_ms,
           total,
           DMP_TIMING_BLOCK_NUMBER(total),
           DMP_TIMING_BLOCK_SIZE,
           elapsed_ms > 0.0f ? static_cast<double>(total) / static_cast<double>(elapsed_ms) : 0.0);
}

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
    uint8_t arc_type_u8 = 0xffu;
    gpuErrchk(cudaMemcpy(&to_pin, h_dmp.level_list + level_start_offset, sizeof(index_type), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&arc_start, h_dmp.pin_backward_arc_list_end + to_pin, sizeof(index_type), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&arc_end, h_dmp.pin_backward_arc_list_end + to_pin + 1, sizeof(index_type), cudaMemcpyDeviceToHost));
    if (arc_start < arc_end) {
        gpuErrchk(cudaMemcpy(&arc_id, h_dmp.pin_backward_arc_list + arc_start, sizeof(index_type), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&from_pin, h_dmp.timing_arc_from_pin_id + arc_id, sizeof(index_type), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&arc_type_u8, h_dmp.arc_types + arc_id, sizeof(uint8_t), cudaMemcpyDeviceToHost));
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
           level_idx, (int)to_pin, (int)arc_start, (int)arc_end, (int)arc_id,
           arc_type_u8 == 0xffu ? -1 : static_cast<int>(arc_type_u8), (int)from_pin);
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
    vector<uint8_t> arc_types(h_dmp.num_arcs);
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
                             sizeof(uint8_t) * arc_types.size(),
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
