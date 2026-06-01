#include "DmpModel.h"
#include "DmpCudaUtils.cuh"
#include "gputiming.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gt {

struct DmpForwardArcLevels {
    vector<index_type> gate_arc_list;
    vector<int> gate_arc_end;
    vector<index_type> net_arc_list;
    vector<int> net_arc_end;
    vector<index_type> direct_net_arc_list;
    vector<int> direct_net_arc_end;
    int max_level_gate_arcs = 0;
    int max_gate_level = -1;
    int max_level_net_arcs = 0;
    int max_net_level = -1;
    int max_level_direct_net_arcs = 0;
    int max_direct_net_level = -1;
    long long gate_net_pairs = 0;
    long long pair_lanes = 0;
    long long valid_pair_lanes = 0;
    long long invalid_pair_lanes = 0;
    long long max_level_gate_net_pairs = 0;
    int max_gate_net_pair_level = -1;
    long long max_level_valid_pair_lanes = 0;
    int max_valid_pair_lane_level = -1;
    int scratch_capacity_items = 0;
    int num_pins = 0;
    int num_arcs = 0;
    int level_list_size = 0;
    index_type* d_forward_gate_arc_list = nullptr;
    index_type* d_forward_net_arc_list = nullptr;
    index_type* d_forward_direct_net_arc_list = nullptr;
    bool uploaded = false;
};

static constexpr int DMP_GNP_DEBUG_COUNTERS = 4;

struct DmpKernelProfile {
    const char* name = nullptr;
    int launches = 0;
    long long work_items = 0;
    long long blocks = 0;
    int max_work_items = 0;
    int max_blocks = 0;
    float total_ms = 0.0f;
    float max_ms = 0.0f;
    int max_level = -1;
};

enum DmpKernelProfileId {
    DMP_PROFILE_GATE_DELAY_SLEW = 0,
    DMP_PROFILE_DIRECT_NET,
    DMP_PROFILE_NET_DELAY_FINALIZE,
    DMP_PROFILE_AT_FINALIZE,
    DMP_PROFILE_ARC_TEST,
    DMP_PROFILE_BACKWARD,
    DMP_PROFILE_COUNT
};

static void dmp_init_kernel_profiles(DmpKernelProfile profiles[DMP_PROFILE_COUNT])
{
    profiles[DMP_PROFILE_GATE_DELAY_SLEW].name = "dmpGateKernel";
    profiles[DMP_PROFILE_DIRECT_NET].name = "dmpDirectNetKernel";
    profiles[DMP_PROFILE_NET_DELAY_FINALIZE].name = "dmpNetWinnerKernel";
    profiles[DMP_PROFILE_AT_FINALIZE].name = "dmpPinWinnerKernel";
    profiles[DMP_PROFILE_ARC_TEST].name = "dmpTestKernel";
    profiles[DMP_PROFILE_BACKWARD].name = "dmpBackwardKernel";
}

static bool dmp_root_profile_enabled()
{
    const char* env = std::getenv("DMP_PROFILE_ROOTS");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool dmp_level_arc_profile_enabled()
{
    const char* env = std::getenv("DMP_PROFILE_LEVEL_ARCS");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static void dmp_record_kernel_profile(DmpKernelProfile& profile,
                                      int level,
                                      int work_items,
                                      int blocks,
                                      float elapsed_ms)
{
    ++profile.launches;
    profile.work_items += work_items;
    profile.blocks += blocks;
    if (work_items > profile.max_work_items) {
        profile.max_work_items = work_items;
    }
    if (blocks > profile.max_blocks) {
        profile.max_blocks = blocks;
    }
    profile.total_ms += elapsed_ms;
    if (elapsed_ms > profile.max_ms) {
        profile.max_ms = elapsed_ms;
        profile.max_level = level;
    }
}

static void dmp_print_kernel_profiles(const DmpKernelProfile profiles[DMP_PROFILE_COUNT])
{
    float total_ms = 0.0f;
    int total_launches = 0;
    for (int idx = 0; idx < DMP_PROFILE_COUNT; ++idx) {
        total_ms += profiles[idx].total_ms;
        total_launches += profiles[idx].launches;
    }
    printf("[DMP KERNEL PROFILE] total_kernel_ms=%.3f launches=%d block_size=%d\n",
           total_ms, total_launches, DMP_TIMING_BLOCK_SIZE);
    for (int idx = 0; idx < DMP_PROFILE_COUNT; ++idx) {
        const DmpKernelProfile& profile = profiles[idx];
        if (profile.launches == 0) {
            continue;
        }
        const double avg_us = static_cast<double>(profile.total_ms) * 1000.0 /
                              static_cast<double>(profile.launches);
        const double avg_work = static_cast<double>(profile.work_items) /
                                static_cast<double>(profile.launches);
        const double avg_blocks = static_cast<double>(profile.blocks) /
                                  static_cast<double>(profile.launches);
        const double work_per_ms = profile.total_ms > 0.0f
                                       ? static_cast<double>(profile.work_items) /
                                             static_cast<double>(profile.total_ms)
                                       : 0.0;
        printf("[DMP KERNEL PROFILE] name=%s launches=%d total_ms=%.3f avg_us=%.3f max_ms=%.3f@L%d work_items=%lld avg_work=%.1f max_work=%d blocks=%lld avg_blocks=%.1f max_blocks=%d work_per_ms=%.1f\n",
               profile.name,
               profile.launches,
               profile.total_ms,
               avg_us,
               profile.max_ms,
               profile.max_level,
               profile.work_items,
               avg_work,
               profile.max_work_items,
               profile.blocks,
               avg_blocks,
               profile.max_blocks,
               work_per_ms);
    }
    fflush(stdout);
}

static DmpForwardArcLevels build_forward_arc_levels(DmpModel* dmp_db,
                                                    const vector<int>& level_list_end_cpu) {
    DmpForwardArcLevels result;
    result.gate_arc_end.reserve(level_list_end_cpu.size());
    result.net_arc_end.reserve(level_list_end_cpu.size());
    result.direct_net_arc_end.reserve(level_list_end_cpu.size());
    result.gate_arc_end.push_back(0);
    result.net_arc_end.push_back(0);
    result.direct_net_arc_end.push_back(0);

    DmpModel h_dmp;
    gpuErrchk(cudaMemcpy(&h_dmp, dmp_db, sizeof(DmpModel), cudaMemcpyDeviceToHost));
    h_dmp.owns_allocations = false;
    const bool log_schedule = dmp_kernel_profile_enabled() || dmp_timing_debug_enabled();
    const bool log_level_arcs = dmp_level_arc_profile_enabled();
    if (h_dmp.num_pins <= 0 || h_dmp.num_arcs < 0 || level_list_end_cpu.empty()) {
        if (log_schedule) {
            printf("[DMP FORWARD SCHEDULE] skip build invalid dimensions pins=%d arcs=%d levels=%zu\n",
                   h_dmp.num_pins, h_dmp.num_arcs, level_list_end_cpu.size());
        }
        return result;
    }
    result.scratch_capacity_items = h_dmp.dmp_work_slot_capacity;
    result.num_pins = h_dmp.num_pins;
    result.num_arcs = h_dmp.num_arcs;
    result.level_list_size = level_list_end_cpu.empty() ? 0 : level_list_end_cpu.back();

    vector<index_type> level_list(level_list_end_cpu.back());
    vector<index_type> fanin_end(h_dmp.num_pins + 1);
    vector<index_type> fanin_arcs(h_dmp.num_arcs);
    vector<index_type> fanout_end(h_dmp.num_pins + 1);
    vector<index_type> fanout_arcs(h_dmp.num_arcs);
    vector<index_type> timing_from(h_dmp.num_arcs);
    vector<index_type> timing_to(h_dmp.num_arcs);
    vector<int> timing_id_map(static_cast<size_t>(h_dmp.num_arcs) * 2u);
    vector<int> arc_types(h_dmp.num_arcs);
    if (!level_list.empty()) {
        gpuErrchk(cudaMemcpy(level_list.data(), h_dmp.level_list,
                             sizeof(index_type) * level_list.size(),
                             cudaMemcpyDeviceToHost));
    }
    gpuErrchk(cudaMemcpy(fanin_end.data(), h_dmp.pin_backward_arc_list_end,
                         sizeof(index_type) * fanin_end.size(),
                         cudaMemcpyDeviceToHost));
    if (h_dmp.pin_forward_arc_list_end != nullptr) {
        gpuErrchk(cudaMemcpy(fanout_end.data(), h_dmp.pin_forward_arc_list_end,
                             sizeof(index_type) * fanout_end.size(),
                             cudaMemcpyDeviceToHost));
    }
    if (!fanin_arcs.empty()) {
        gpuErrchk(cudaMemcpy(fanin_arcs.data(), h_dmp.pin_backward_arc_list,
                             sizeof(index_type) * fanin_arcs.size(),
                             cudaMemcpyDeviceToHost));
        if (h_dmp.pin_forward_arc_list != nullptr) {
            gpuErrchk(cudaMemcpy(fanout_arcs.data(), h_dmp.pin_forward_arc_list,
                                 sizeof(index_type) * fanout_arcs.size(),
                                 cudaMemcpyDeviceToHost));
        }
        gpuErrchk(cudaMemcpy(timing_from.data(), h_dmp.timing_arc_from_pin_id,
                             sizeof(index_type) * timing_from.size(),
                             cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(timing_to.data(), h_dmp.timing_arc_to_pin_id,
                             sizeof(index_type) * timing_to.size(),
                             cudaMemcpyDeviceToHost));
        if (h_dmp.timing_arc_id_map != nullptr) {
            gpuErrchk(cudaMemcpy(timing_id_map.data(), h_dmp.timing_arc_id_map,
                                 sizeof(int) * timing_id_map.size(),
                                 cudaMemcpyDeviceToHost));
        }
        gpuErrchk(cudaMemcpy(arc_types.data(), h_dmp.arc_types,
                             sizeof(int) * arc_types.size(),
                             cudaMemcpyDeviceToHost));
    }

    for (int level = 0; level + 1 < static_cast<int>(level_list_end_cpu.size()); ++level) {
        const int start = level_list_end_cpu[level];
        const int end = level_list_end_cpu[level + 1];
        const int gate_before = static_cast<int>(result.gate_arc_list.size());
        const int net_before = static_cast<int>(result.net_arc_list.size());
        const int direct_before = static_cast<int>(result.direct_net_arc_list.size());
        long long level_gate_net_pairs = 0;
        long long level_pair_lanes = 0;
        long long level_valid_pair_lanes = 0;
        for (int pos = start; pos < end; ++pos) {
            const int pin = level_list[pos];
            for (index_type arc_pos = fanin_end[pin]; arc_pos < fanin_end[pin + 1]; ++arc_pos) {
                const int arc_id = fanin_arcs[arc_pos];
                if (arc_id < 0 || arc_id >= h_dmp.num_arcs) {
                    continue;
                }
                if (arc_types[arc_id] == 1) {
                    result.gate_arc_list.push_back(arc_id);
                    const int to_pin = timing_to[arc_id];
                    long long sink_net_arcs = 0;
                    if (to_pin >= 0 && to_pin < h_dmp.num_pins &&
                        h_dmp.pin_forward_arc_list_end != nullptr &&
                        h_dmp.pin_forward_arc_list != nullptr) {
                        for (index_type fanout_pos = fanout_end[to_pin];
                             fanout_pos < fanout_end[to_pin + 1];
                             ++fanout_pos) {
                            const int net_arc = fanout_arcs[fanout_pos];
                            if (net_arc >= 0 && net_arc < h_dmp.num_arcs &&
                                arc_types[net_arc] == 0) {
                                ++sink_net_arcs;
                            }
                        }
                    }
                    int valid_lanes = DMP_PIN_GROUP_SIZE;
                    if (h_dmp.timing_arc_id_map != nullptr) {
                        valid_lanes = 0;
                        for (int lane = 0; lane < DMP_PIN_GROUP_SIZE; ++lane) {
                            const int el = lane >> 2;
                            const int timing_id = timing_id_map[arc_id * 2 + el];
                            if (timing_id >= 0) {
                                ++valid_lanes;
                            }
                        }
                    }
                    level_gate_net_pairs += sink_net_arcs;
                    level_pair_lanes += sink_net_arcs * DMP_PIN_GROUP_SIZE;
                    level_valid_pair_lanes += sink_net_arcs * valid_lanes;
                } else if (arc_types[arc_id] == 0) {
                    result.net_arc_list.push_back(arc_id);
                    const int from_pin = timing_from[arc_id];
                    bool has_gate_driver = false;
                    if (from_pin >= 0 && from_pin < h_dmp.num_pins) {
                        for (index_type src_pos = fanin_end[from_pin];
                             src_pos < fanin_end[from_pin + 1];
                             ++src_pos) {
                            const int gate_arc = fanin_arcs[src_pos];
                            if (gate_arc >= 0 && gate_arc < h_dmp.num_arcs &&
                                arc_types[gate_arc] == 1) {
                                has_gate_driver = true;
                            }
                        }
                    }
                    if (!has_gate_driver) {
                        result.direct_net_arc_list.push_back(arc_id);
                    }
                }
            }
        }
        const int level_gate_arcs = static_cast<int>(result.gate_arc_list.size()) - gate_before;
        const int level_net_arcs = static_cast<int>(result.net_arc_list.size()) - net_before;
        const int level_direct_net_arcs = static_cast<int>(result.direct_net_arc_list.size()) - direct_before;
        const long long level_invalid_pair_lanes = level_pair_lanes - level_valid_pair_lanes;
        result.gate_net_pairs += level_gate_net_pairs;
        result.pair_lanes += level_pair_lanes;
        result.valid_pair_lanes += level_valid_pair_lanes;
        result.invalid_pair_lanes += level_invalid_pair_lanes;
        if (log_level_arcs) {
            const int level_pins = end - start;
            const int level_arcs = level_gate_arcs + level_net_arcs;
            const long long gate_lanes = static_cast<long long>(level_gate_arcs) * DMP_PIN_GROUP_SIZE;
            printf("[DMP LEVEL ARCS] L%d pins=%d arcs=%d gate_arcs=%d net_arcs=%d direct_net_arcs=%d gate_lanes=%lld gate_net_pairs=%lld pair_lanes=%lld valid_pair_lanes=%lld invalid_pair_lanes=%lld\n",
                   level,
                   level_pins,
                   level_arcs,
                   level_gate_arcs,
                   level_net_arcs,
                   level_direct_net_arcs,
                   gate_lanes,
                   level_gate_net_pairs,
                   level_pair_lanes,
                   level_valid_pair_lanes,
                   level_invalid_pair_lanes);
        }
        if (level_gate_arcs > result.max_level_gate_arcs) {
            result.max_level_gate_arcs = level_gate_arcs;
            result.max_gate_level = level;
        }
        if (level_net_arcs > result.max_level_net_arcs) {
            result.max_level_net_arcs = level_net_arcs;
            result.max_net_level = level;
        }
        if (level_direct_net_arcs > result.max_level_direct_net_arcs) {
            result.max_level_direct_net_arcs = level_direct_net_arcs;
            result.max_direct_net_level = level;
        }
        if (level_gate_net_pairs > result.max_level_gate_net_pairs) {
            result.max_level_gate_net_pairs = level_gate_net_pairs;
            result.max_gate_net_pair_level = level;
        }
        if (level_valid_pair_lanes > result.max_level_valid_pair_lanes) {
            result.max_level_valid_pair_lanes = level_valid_pair_lanes;
            result.max_valid_pair_lane_level = level;
        }
        result.gate_arc_end.push_back(static_cast<int>(result.gate_arc_list.size()));
        result.net_arc_end.push_back(static_cast<int>(result.net_arc_list.size()));
        result.direct_net_arc_end.push_back(static_cast<int>(result.direct_net_arc_list.size()));
    }

    if (log_schedule) {
        printf("[DMP FORWARD SCHEDULE] built levels=%zu gate_arcs=%zu net_arcs=%zu direct_net_arcs=%zu gate_net_pairs=%lld valid_pair_lanes=%lld invalid_pair_lanes=%lld max_gate=%d@L%d max_net=%d@L%d max_direct_net=%d@L%d max_pairs=%lld@L%d max_valid_pair_lanes=%lld@L%d scratch_capacity_items=%d mode=direct\n",
               result.gate_arc_end.empty() ? 0 : result.gate_arc_end.size() - 1,
               result.gate_arc_list.size(),
               result.net_arc_list.size(),
               result.direct_net_arc_list.size(),
               result.gate_net_pairs,
               result.valid_pair_lanes,
               result.invalid_pair_lanes,
               result.max_level_gate_arcs,
               result.max_gate_level,
               result.max_level_net_arcs,
               result.max_net_level,
               result.max_level_direct_net_arcs,
               result.max_direct_net_level,
               result.max_level_gate_net_pairs,
               result.max_gate_net_pair_level,
               result.max_level_valid_pair_lanes,
               result.max_valid_pair_lane_level,
               result.scratch_capacity_items);
    }
    dmp_clear_stale_cuda_error("build forward arc levels");
    return result;
}

template <typename T>
static void dmp_upload_vector(const vector<T>& values, T** device_ptr)
{
    *device_ptr = nullptr;
    if (values.empty()) {
        return;
    }
    gpuErrchk(cudaMalloc(device_ptr, sizeof(T) * values.size()));
    gpuErrchk(cudaMemcpy(*device_ptr,
                         values.data(),
                         sizeof(T) * values.size(),
                         cudaMemcpyHostToDevice));
}

static void dmp_upload_forward_schedule(DmpForwardArcLevels& schedule)
{
    if (schedule.uploaded) {
        return;
    }
    dmp_upload_vector(schedule.gate_arc_list, &schedule.d_forward_gate_arc_list);
    dmp_upload_vector(schedule.net_arc_list, &schedule.d_forward_net_arc_list);
    dmp_upload_vector(schedule.direct_net_arc_list, &schedule.d_forward_direct_net_arc_list);
    schedule.uploaded = true;
}

void release_dmp_forward_schedule_cuda(void* schedule_ptr)
{
    auto* schedule = reinterpret_cast<DmpForwardArcLevels*>(schedule_ptr);
    if (schedule == nullptr) {
        return;
    }
    if (schedule->d_forward_gate_arc_list != nullptr) {
        gpuErrchk(cudaFree(schedule->d_forward_gate_arc_list));
    }
    if (schedule->d_forward_net_arc_list != nullptr) {
        gpuErrchk(cudaFree(schedule->d_forward_net_arc_list));
    }
    if (schedule->d_forward_direct_net_arc_list != nullptr) {
        gpuErrchk(cudaFree(schedule->d_forward_direct_net_arc_list));
    }
    delete schedule;
}

static DmpForwardArcLevels& dmp_get_forward_schedule(GPUTimer* timer)
{
    auto* schedule = reinterpret_cast<DmpForwardArcLevels*>(timer->dmp_forward_schedule);
    const int level_list_size = timer->level_list_end_cpu.empty() ? 0 : timer->level_list_end_cpu.back();
    if (schedule != nullptr &&
        schedule->num_pins == timer->num_pins &&
        schedule->num_arcs == timer->num_arcs &&
        schedule->level_list_size == level_list_size) {
        return *schedule;
    }
    if (schedule != nullptr) {
        release_dmp_forward_schedule_cuda(schedule);
        timer->dmp_forward_schedule = nullptr;
    }
    schedule = new DmpForwardArcLevels(
        build_forward_arc_levels(timer->dmp_db, timer->level_list_end_cpu));
    dmp_upload_forward_schedule(*schedule);
    timer->dmp_forward_schedule = schedule;
    return *schedule;
}

void update_timing_dmp_cuda(GPUTimer* timer){

    DmpModel* dmp_db = timer->dmp_db;
    const vector<int>& level_list_end_cpu = timer->level_list_end_cpu;

    dmp_clear_stale_cuda_error("DMP timing entry");
    const bool profile_kernels = dmp_kernel_profile_enabled();
    const bool profile_roots = dmp_root_profile_enabled();
    const bool debug_timing = dmp_timing_debug_enabled();
    DmpModel h_entry_dmp;
    gpuErrchk(cudaMemcpy(&h_entry_dmp, dmp_db, sizeof(DmpModel), cudaMemcpyDeviceToHost));
    h_entry_dmp.owns_allocations = false;
    if (h_entry_dmp.pin_at_winner != nullptr && h_entry_dmp.dmp_pin_slot_count > 0) {
        gpuErrchk(cudaMemset(h_entry_dmp.pin_at_winner, 0,
                             sizeof(unsigned long long) * h_entry_dmp.dmp_pin_slot_count));
    }
    if (debug_timing) {
        dmp_debug_print_counts(dmp_db, "entry");
        dmp_debug_print_parallel_stats(dmp_db, level_list_end_cpu, "entry");
    }
    unsigned long long* d_gate_net_pair_debug_counts = nullptr;
    unsigned long long gate_net_pair_debug_counts[DMP_GNP_DEBUG_COUNTERS] = {0ULL, 0ULL, 0ULL, 0ULL};
    DmpForwardArcLevels* forward_arc_levels = &dmp_get_forward_schedule(timer);
    const int reset_work_items = h_entry_dmp.dmp_pin_slot_count + h_entry_dmp.num_arcs * 2 * NUM_ATTR;
    if (reset_work_items > 0) {
        const int reset_blocks = DMP_TIMING_BLOCK_NUMBER(reset_work_items);
        dmpResetForwardTargetsKernel<<<reset_blocks, DMP_TIMING_BLOCK_SIZE>>>(dmp_db);
        gpuErrchk(cudaGetLastError());
    }
    if (debug_timing) {
        gpuErrchk(cudaMalloc(&d_gate_net_pair_debug_counts,
                             sizeof(unsigned long long) * DMP_GNP_DEBUG_COUNTERS));
        gpuErrchk(cudaMemset(d_gate_net_pair_debug_counts, 0,
                             sizeof(unsigned long long) * DMP_GNP_DEBUG_COUNTERS));
    }
    if (debug_timing) {
        printf("[DMP DEBUG LAUNCH] timing_block_size=%d\n", DMP_TIMING_BLOCK_SIZE);
    }
    if (profile_roots) {
        reset_dmp_root_profile_cuda();
    }
    bool traced_first_forward_level = false;
    float forward_total_ms = 0.0f;
    float forward_max_level_ms = 0.0f;
    int forward_max_level = -1;
    int forward_launches = 0;
    int forward_gate_launches = 0;
    int forward_direct_net_launches = 0;
    int forward_at_finalize_launches = 0;
    int forward_net_delay_finalize_launches = 0;
    int forward_arc_test_launches = 0;
    DmpKernelProfile kernel_profiles[DMP_PROFILE_COUNT];
    dmp_init_kernel_profiles(kernel_profiles);
    for (int i = 1; i < level_list_end_cpu.size() - 1; i++) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        if (num_pins_level <= 0) {
            continue;
        }
        index_type level_start_offset = level_list_end_cpu[i];
        // printf("==== level %d ======= %d \n", i, num_pins_level);
        bool debug_level = debug_timing && !traced_first_forward_level && num_pins_level > 0;
        if (debug_level) {
            printf("[DMP TRACE host] first non-empty forward level=%d start=%d count=%d\n", i, (int)level_start_offset, num_pins_level);
            traced_first_forward_level = true;
        }
        const int level_gate_start = (forward_arc_levels != nullptr &&
                                      i + 1 < static_cast<int>(forward_arc_levels->gate_arc_end.size()))
                                         ? forward_arc_levels->gate_arc_end[i]
                                         : 0;
        const int level_gate_end = (forward_arc_levels != nullptr &&
                                    i + 1 < static_cast<int>(forward_arc_levels->gate_arc_end.size()))
                                       ? forward_arc_levels->gate_arc_end[i + 1]
                                       : level_gate_start;
        const int level_net_start = (forward_arc_levels != nullptr &&
                                     i + 1 < static_cast<int>(forward_arc_levels->net_arc_end.size()))
                                        ? forward_arc_levels->net_arc_end[i]
                                        : 0;
        const int level_net_end = (forward_arc_levels != nullptr &&
                                   i + 1 < static_cast<int>(forward_arc_levels->net_arc_end.size()))
                                      ? forward_arc_levels->net_arc_end[i + 1]
                                      : level_net_start;
        const int level_direct_net_start = (forward_arc_levels != nullptr &&
                                            i + 1 < static_cast<int>(forward_arc_levels->direct_net_arc_end.size()))
                                               ? forward_arc_levels->direct_net_arc_end[i]
                                               : 0;
        const int level_direct_net_end = (forward_arc_levels != nullptr &&
                                          i + 1 < static_cast<int>(forward_arc_levels->direct_net_arc_end.size()))
                                             ? forward_arc_levels->direct_net_arc_end[i + 1]
                                             : level_direct_net_start;
        const int num_gate_arcs_level = level_gate_end - level_gate_start;
        const int num_net_arcs_level = level_net_end - level_net_start;
        const int num_direct_net_arcs_level = level_direct_net_end - level_direct_net_start;
        const int pin_work_items = num_pins_level * DMP_PIN_GROUP_SIZE;
        const int gate_work_items = num_gate_arcs_level * DMP_PIN_GROUP_SIZE;
        const int direct_net_work_items = num_direct_net_arcs_level * DMP_PIN_GROUP_SIZE;
        const int pin_blocks = DMP_TIMING_BLOCK_NUMBER(pin_work_items);
        if (profile_kernels) {
            dmp_clear_stale_cuda_error("before forward launch");
        }
        cudaEvent_t level_start = nullptr;
        cudaEvent_t level_stop = nullptr;
        if (profile_kernels) {
            dmp_event_create(&level_start, &level_stop);
            gpuErrchk(cudaEventRecord(level_start));
        }
        auto start_kernel_profile = [&](cudaEvent_t* kernel_start, cudaEvent_t* kernel_stop) {
            if (!profile_kernels) {
                *kernel_start = nullptr;
                *kernel_stop = nullptr;
                return;
            }
            dmp_event_create(kernel_start, kernel_stop);
            gpuErrchk(cudaEventRecord(*kernel_start));
        };
        auto finish_forward_cuda = [&](DmpKernelProfileId profile_id,
                                       const char* stage,
                                       int stage_work_items,
                                       int stage_blocks,
                                       cudaEvent_t kernel_start,
                                       cudaEvent_t kernel_stop) {
            cudaError_t launch_error = cudaPeekAtLastError();
            if (launch_error != cudaSuccess) {
                fprintf(stderr,
                        "[DMP CUDA] forward %s launch failed level=%d pin_start=%d pin_count=%d gate=%d net=%d direct_net=%d work=%d blocks=%d block=%d: %s\n",
                        stage, i, (int)level_start_offset, num_pins_level,
                        num_gate_arcs_level, num_net_arcs_level,
                        num_direct_net_arcs_level, stage_work_items, stage_blocks,
                        DMP_TIMING_BLOCK_SIZE, cudaGetErrorString(launch_error));
                exit(launch_error);
            }
            if (!profile_kernels) {
                return;
            }
            gpuErrchk(cudaEventRecord(kernel_stop));
            cudaError_t sync_error = cudaDeviceSynchronize();
            if (sync_error != cudaSuccess) {
                fprintf(stderr,
                        "[DMP CUDA] forward %s sync failed level=%d pin_start=%d pin_count=%d gate=%d net=%d direct_net=%d work=%d blocks=%d block=%d: %s\n",
                        stage, i, (int)level_start_offset, num_pins_level,
                        num_gate_arcs_level, num_net_arcs_level,
                        num_direct_net_arcs_level, stage_work_items, stage_blocks,
                        DMP_TIMING_BLOCK_SIZE, cudaGetErrorString(sync_error));
                cudaGetLastError();
                exit(sync_error);
            }
            float elapsed_ms = 0.0f;
            gpuErrchk(cudaEventElapsedTime(&elapsed_ms, kernel_start, kernel_stop));
            gpuErrchk(cudaEventDestroy(kernel_start));
            gpuErrchk(cudaEventDestroy(kernel_stop));
            dmp_record_kernel_profile(kernel_profiles[profile_id],
                                      i,
                                      stage_work_items,
                                      stage_blocks,
                                      elapsed_ms);
            cudaGetLastError();
        };
        if (num_gate_arcs_level > 0) {
            const int gate_blocks = DMP_TIMING_BLOCK_NUMBER(gate_work_items);
            cudaEvent_t kernel_start, kernel_stop;
            start_kernel_profile(&kernel_start, &kernel_stop);
            dmpGateKernel<<<gate_blocks, DMP_TIMING_BLOCK_SIZE>>>(dmp_db,
                                                                                            forward_arc_levels->d_forward_gate_arc_list + level_gate_start,
                                                                                            num_gate_arcs_level,
                                                                                            d_gate_net_pair_debug_counts);
            finish_forward_cuda(DMP_PROFILE_GATE_DELAY_SLEW,
                                "direct-gate-net-delay-slew-at",
                                gate_work_items,
                                gate_blocks,
                                kernel_start,
                                kernel_stop);
            forward_gate_launches++;
        }
        if (num_direct_net_arcs_level > 0) {
            const int direct_net_blocks = DMP_TIMING_BLOCK_NUMBER(direct_net_work_items);
            cudaEvent_t kernel_start, kernel_stop;
            start_kernel_profile(&kernel_start, &kernel_stop);
            dmpDirectNetKernel<<<direct_net_blocks, DMP_TIMING_BLOCK_SIZE>>>(dmp_db,
                                                                                       forward_arc_levels->d_forward_direct_net_arc_list + level_direct_net_start,
                                                                                       num_direct_net_arcs_level);
            finish_forward_cuda(DMP_PROFILE_DIRECT_NET,
                                "direct-net",
                                direct_net_work_items,
                                direct_net_blocks,
                                kernel_start,
                                kernel_stop);
            forward_direct_net_launches++;
        }
        if (num_net_arcs_level > 0) {
            const int net_delay_finalize_work_items = num_net_arcs_level * NUM_ATTR;
            const int net_delay_finalize_blocks = DMP_TIMING_BLOCK_NUMBER(net_delay_finalize_work_items);
            cudaEvent_t kernel_start, kernel_stop;
            start_kernel_profile(&kernel_start, &kernel_stop);
            dmpNetWinnerKernel<<<net_delay_finalize_blocks, DMP_TIMING_BLOCK_SIZE>>>(dmp_db,
                                                                                                            forward_arc_levels->d_forward_net_arc_list + level_net_start,
                                                                                                            num_net_arcs_level);
            finish_forward_cuda(DMP_PROFILE_NET_DELAY_FINALIZE,
                                "net-delay-finalize-at",
                                net_delay_finalize_work_items,
                                net_delay_finalize_blocks,
                                kernel_start,
                                kernel_stop);
            forward_net_delay_finalize_launches++;
        }
        const int at_finalize_work_items = num_pins_level * NUM_ATTR;
        const int at_finalize_blocks = DMP_TIMING_BLOCK_NUMBER(at_finalize_work_items);
        cudaEvent_t kernel_start, kernel_stop;
        start_kernel_profile(&kernel_start, &kernel_stop);
        dmpPinWinnerKernel<<<at_finalize_blocks, DMP_TIMING_BLOCK_SIZE>>>(dmp_db,
                                                                              level_start_offset,
                                                                              num_pins_level);
        finish_forward_cuda(DMP_PROFILE_AT_FINALIZE,
                            "pin-winner-finalize",
                            at_finalize_work_items,
                            at_finalize_blocks,
                            kernel_start,
                            kernel_stop);
        forward_at_finalize_launches++;
        start_kernel_profile(&kernel_start, &kernel_stop);
        dmpTestKernel<<<pin_blocks, DMP_TIMING_BLOCK_SIZE>>>(dmp_db, level_start_offset, num_pins_level);
        finish_forward_cuda(DMP_PROFILE_ARC_TEST,
                            "arc-test",
                            pin_work_items,
                            pin_blocks,
                            kernel_start,
                            kernel_stop);
        forward_arc_test_launches++;
        float level_ms = 0.0f;
        if (profile_kernels) {
            gpuErrchk(cudaEventRecord(level_stop));
            gpuErrchk(cudaDeviceSynchronize());
            gpuErrchk(cudaEventElapsedTime(&level_ms, level_start, level_stop));
            gpuErrchk(cudaEventDestroy(level_start));
            gpuErrchk(cudaEventDestroy(level_stop));
            forward_total_ms += level_ms;
        }
        forward_launches++;
        if (profile_kernels && level_ms > forward_max_level_ms) {
            forward_max_level_ms = level_ms;
            forward_max_level = i;
        }
        if (debug_level) {
            dmp_debug_print_first_level_sample(dmp_db, i, level_start_offset);
        }
        if (profile_kernels) {
            dmp_clear_stale_cuda_error("forward level profile");
        }
        // if(i == 2)break;
    }
    gpuErrchk( cudaDeviceSynchronize() );
    if (d_gate_net_pair_debug_counts != nullptr) {
        gpuErrchk(cudaMemcpy(gate_net_pair_debug_counts,
                             d_gate_net_pair_debug_counts,
                             sizeof(unsigned long long) * DMP_GNP_DEBUG_COUNTERS,
                             cudaMemcpyDeviceToHost));
    }
    if (profile_kernels) {
        printf("[DMP TIMING PROFILE] forward levels=%d mode=direct gate=%d direct_net=%d net_delay_finalize=%d at_finalize=%d arc_test=%d total_ms=%.3f max_level_ms=%.3f@L%d\n",
               forward_launches,
               forward_gate_launches,
               forward_direct_net_launches,
               forward_net_delay_finalize_launches,
               forward_at_finalize_launches,
               forward_arc_test_launches,
               forward_total_ms,
               forward_max_level_ms,
               forward_max_level);
    }
    if (debug_timing) {
        printf("[DMP TIMING PROFILE] gate_net_pair total=%llu invalid_transition_skip=%llu invalid_scratch_skip=%llu finite_submitted=%llu\n",
               gate_net_pair_debug_counts[0],
               gate_net_pair_debug_counts[1],
               gate_net_pair_debug_counts[2],
               gate_net_pair_debug_counts[3]);
        dmp_debug_print_counts(dmp_db, "after forward");
        dmp_debug_print_parallel_stats(dmp_db, level_list_end_cpu, "after forward");
    }
    dmp_clear_stale_cuda_error("after forward stats");
    float backward_total_ms = 0.0f;
    float backward_max_level_ms = 0.0f;
    int backward_max_level = -1;
    int backward_launches = 0;
    for (int i = level_list_end_cpu.size() - 3; i >= 0; i--) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        if (num_pins_level <= 0) {
            continue;
        }
        index_type level_start_offset = level_list_end_cpu[i];
        // printf("==== level %d ======= %d \n", i, num_pins_level);
        const int work_items = num_pins_level * 2 * NUM_ATTR;
        const int blocks = DMP_TIMING_BLOCK_NUMBER(work_items);
        if (profile_kernels) {
            dmp_clear_stale_cuda_error("before backward launch");
        }
        cudaEvent_t level_start = nullptr;
        cudaEvent_t level_stop = nullptr;
        if (profile_kernels) {
            dmp_event_create(&level_start, &level_stop);
            gpuErrchk(cudaEventRecord(level_start));
        }
        dmpBackwardKernel<<<blocks, DMP_TIMING_BLOCK_SIZE, DMP_TIMING_BLOCK_SIZE * sizeof(float)>>>(dmp_db, level_start_offset, num_pins_level);

        cudaError_t launch_error = cudaPeekAtLastError();
        if (launch_error != cudaSuccess) {
            fprintf(stderr,
                    "[DMP CUDA] backward launch failed level=%d start=%d count=%d work=%d blocks=%d block=%d: %s\n",
                    i, (int)level_start_offset, num_pins_level, work_items, blocks,
                    DMP_TIMING_BLOCK_SIZE, cudaGetErrorString(launch_error));
            exit(launch_error);
        }
        float level_ms = 0.0f;
        if (profile_kernels) {
            gpuErrchk(cudaEventRecord(level_stop));
            cudaError_t sync_error = cudaDeviceSynchronize();
            if (sync_error != cudaSuccess) {
                fprintf(stderr,
                        "[DMP CUDA] backward sync failed level=%d start=%d count=%d work=%d blocks=%d block=%d: %s\n",
                        i, (int)level_start_offset, num_pins_level, work_items, blocks,
                        DMP_TIMING_BLOCK_SIZE, cudaGetErrorString(sync_error));
                cudaGetLastError();
                exit(sync_error);
            }
            gpuErrchk(cudaEventElapsedTime(&level_ms, level_start, level_stop));
            gpuErrchk(cudaEventDestroy(level_start));
            gpuErrchk(cudaEventDestroy(level_stop));
            backward_total_ms += level_ms;
        }
        backward_launches++;
        if (profile_kernels && level_ms > backward_max_level_ms) {
            backward_max_level_ms = level_ms;
            backward_max_level = i;
        }
        if (profile_kernels) {
            dmp_record_kernel_profile(kernel_profiles[DMP_PROFILE_BACKWARD],
                                      i,
                                      work_items,
                                      blocks,
                                      level_ms);
            dmp_clear_stale_cuda_error("backward level profile");
        }
    }
    gpuErrchk( cudaDeviceSynchronize() );   
    if (profile_kernels) {
        printf("[DMP TIMING PROFILE] backward launches=%d total_ms=%.3f max_level_ms=%.3f@L%d\n",
               backward_launches, backward_total_ms, backward_max_level_ms, backward_max_level);
        dmp_print_kernel_profiles(kernel_profiles);
    }
    if (profile_roots) {
        print_dmp_root_profile_cuda();
    }
    if (debug_timing) {
        dmp_debug_print_counts(dmp_db, "after backward");
    }
    dmp_clear_stale_cuda_error("DMP timing exit");
    if (d_gate_net_pair_debug_counts != nullptr) {
        gpuErrchk(cudaFree(d_gate_net_pair_debug_counts));
    }

}

__device__ void DmpModel::updatePinRat(int arc_id, float* from_rats)
{
    const int from_pin_id = timing_arc_from_pin_id[arc_id];
    for (int ti = 0; ti < DMP_PIN_GROUP_SIZE; ++ti) {
        const int i = ti & 0b111;
        if (isnan(from_rats[ti])) {
            continue;
        }
        const int el = i >> 2;
        const int fel_rf = i >> 1;
        const float rat = from_rats[ti];
        if (isnan(pinRat[from_pin_id * NUM_ATTR + fel_rf]) ||
            ((pinRat[from_pin_id * NUM_ATTR + fel_rf] < rat) ^ el)) {
            atomicExch(&pinRat[from_pin_id * NUM_ATTR + fel_rf], rat);
        }
    }
}

__device__ void DmpModel::propagateRAT(int arc_id, float* from_rats)
{
    const int i = threadIdx.x & (DMP_PIN_GROUP_SIZE - 1);
    const int arc_type = arc_types[arc_id];
    const int from_pin_id = timing_arc_from_pin_id[arc_id];
    const int to_pin_id = timing_arc_to_pin_id[arc_id];
    if ((arc_type == 0) && (i < NUM_ATTR)) {
        const int el_rf_rf = (i << 1) + (i & 1);
        const int el = i >> 1;
        if (isnan(pinRat[to_pin_id * NUM_ATTR + i]) ||
            isnan(arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf])) {
            return;
        }
        const float delay = arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf];
        const float rat = pinRat[to_pin_id * NUM_ATTR + i] - delay;
        if (isnan(pinRat[from_pin_id * NUM_ATTR + i]) ||
            ((pinRat[from_pin_id * NUM_ATTR + i] < rat) ^ el)) {
            atomicExch(&pinRat[from_pin_id * NUM_ATTR + i], rat);
        }
    } else if (arc_type == 1) {
        const int el = i >> 2;
        const int tel_rf = ((i & 0b100) >> 1) + (i & 1);
        if (timing_arc_id_map[arc_id * 2 + el] == -1) {
            return;
        }
        const int timing_id = timing_arc_id_map[arc_id * 2 + el];
        if (d_allocator->d_is_constraint[timing_id]) {
            return;
        }
        if (isnan(pinRat[to_pin_id * NUM_ATTR + tel_rf]) ||
            isnan(arcDelay[arc_id * 2 * NUM_ATTR + i])) {
            return;
        }
        const float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
        from_rats[i] = pinRat[to_pin_id * NUM_ATTR + tel_rf] - delay;
    }
}

__device__ void DmpModel::propagatePinBack(int level_idx, float* from_rats) // can be arc level parallesim 
{
    const index_type from_pin_id = level_list[level_idx];
    const int lane = threadIdx.x & (DMP_PIN_GROUP_SIZE - 1);
    const int warp_lane = threadIdx.x & 31;
    const unsigned group_mask = 0xffu << (warp_lane & ~(DMP_PIN_GROUP_SIZE - 1));
    float* group_rats = from_rats + (threadIdx.x & ~(DMP_PIN_GROUP_SIZE - 1));
    for (index_type i = pin_forward_arc_list_end[from_pin_id];
         i < pin_forward_arc_list_end[from_pin_id + 1];
         ++i) {
        const index_type arc_id = pin_forward_arc_list[i];
        group_rats[lane] = nanf("");
        __syncwarp(group_mask);

        propagateRAT(arc_id, group_rats);

        __syncwarp(group_mask);
        if (lane == 0) {
            updatePinRat(arc_id, group_rats);
        }
        __syncwarp(group_mask);
    }
}

__global__ void dmpBackwardKernel(DmpModel* dmp_db,
                                  int level_start_offset,
                                  int num_pins_level)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_idx = idx >> 3;
    extern __shared__ float from_rats[];

    if (pin_idx < num_pins_level) {
        dmp_db->propagatePinBack(level_start_offset + pin_idx, from_rats);
    }
}

} // namespace gt
