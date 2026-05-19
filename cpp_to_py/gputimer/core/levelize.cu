
#include "GPUTimer.h"
#include "utils.cuh"
#include "gputimer/db/GTDatabase.h"

namespace gt {

__global__ void advanceLevel(index_type *frontiers,
                             index_type *next_frontiers,
                             index_type *level_list,
                             index_type *pin_fanout_list_end,
                             index_type *pin_fanout_list,
                             int *pin_num_fanin,
                             int num_frontiers,
                             int *next_num_frontiers,
                             int *last_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_frontiers) {
        index_type pin_id = frontiers[idx];
        index_type ptr = atomicAdd(last_idx, 1);
        level_list[ptr] = pin_id;
        for (index_type i = pin_fanout_list_end[pin_id]; i < pin_fanout_list_end[pin_id + 1]; i++) {
            index_type fo_pin_id = pin_fanout_list[i];
            int prev_num = atomicAdd(&pin_num_fanin[fo_pin_id], -1);
            if (prev_num == 1) {
                index_type end = atomicAdd(next_num_frontiers, 1);
                next_frontiers[end] = fo_pin_id;
            }
        }
    }
}

__device__ bool power_levelize_valid_edge(index_type arc,
                                          const index_type* timing_arc_to_pin_id,
                                          const int* arc_types,
                                          const int* arc_id2test_id,
                                          const uint8_t* is_seq_output_pin,
                                          int num_pins,
                                          int& to_pin) {
    if (arc < 0) return false;
    if (arc_id2test_id && arc_id2test_id[arc] != -1) return false;
    to_pin = timing_arc_to_pin_id[arc];
    if (to_pin < 0 || to_pin >= num_pins) return false;
    // OpenSTA power activity seeds sequential Q/Q_N separately; do not levelize
    // ordinary D/CLK -> Q timing arcs as activity propagation edges.
    if (arc_types && arc_types[arc] == 1 && is_seq_output_pin && is_seq_output_pin[to_pin]) return false;
    return true;
}

__global__ void countPowerFanin(index_type* pin_forward_arc_list_end,
                                index_type* pin_forward_arc_list,
                                index_type* timing_arc_to_pin_id,
                                int* arc_types,
                                int* arc_id2test_id,
                                const uint8_t* is_seq_output_pin,
                                int* power_num_fanin,
                                int num_pins) {
    int pin = blockIdx.x * blockDim.x + threadIdx.x;
    if (pin >= num_pins) return;
    for (index_type i = pin_forward_arc_list_end[pin]; i < pin_forward_arc_list_end[pin + 1]; i++) {
        int to_pin = -1;
        if (power_levelize_valid_edge(pin_forward_arc_list[i], timing_arc_to_pin_id, arc_types, arc_id2test_id,
                                      is_seq_output_pin, num_pins, to_pin)) {
            atomicAdd(&power_num_fanin[to_pin], 1);
        }
    }
}

__global__ void seedPowerFrontiers(const int* power_num_fanin,
                                   index_type* frontiers,
                                   int* num_frontiers,
                                   int num_pins) {
    int pin = blockIdx.x * blockDim.x + threadIdx.x;
    if (pin >= num_pins) return;
    if (power_num_fanin[pin] == 0) {
        int pos = atomicAdd(num_frontiers, 1);
        frontiers[pos] = pin;
    }
}

__global__ void advancePowerLevel(index_type *frontiers,
                                  index_type *next_frontiers,
                                  index_type *level_list,
                                  index_type *pin_forward_arc_list_end,
                                  index_type *pin_forward_arc_list,
                                  index_type *timing_arc_to_pin_id,
                                  int *arc_types,
                                  int *arc_id2test_id,
                                  const uint8_t* is_seq_output_pin,
                                  int *power_num_fanin,
                                  uint8_t *emitted,
                                  int num_frontiers,
                                  int *next_num_frontiers,
                                  int *last_idx,
                                  int num_pins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frontiers) return;
    index_type pin_id = frontiers[idx];
    if (pin_id < 0 || pin_id >= num_pins) return;
    index_type ptr = atomicAdd(last_idx, 1);
    level_list[ptr] = pin_id;
    emitted[pin_id] = 1;
    for (index_type i = pin_forward_arc_list_end[pin_id]; i < pin_forward_arc_list_end[pin_id + 1]; i++) {
        int to_pin = -1;
        if (!power_levelize_valid_edge(pin_forward_arc_list[i], timing_arc_to_pin_id, arc_types, arc_id2test_id,
                                       is_seq_output_pin, num_pins, to_pin)) {
            continue;
        }
        int prev_num = atomicAdd(&power_num_fanin[to_pin], -1);
        if (prev_num == 1) {
            index_type end = atomicAdd(next_num_frontiers, 1);
            next_frontiers[end] = to_pin;
        }
    }
}

__global__ void appendUnemittedPowerPins(index_type* level_list,
                                         const uint8_t* emitted,
                                         int* last_idx,
                                         int num_pins) {
    int pin = blockIdx.x * blockDim.x + threadIdx.x;
    if (pin >= num_pins || emitted[pin]) return;
    int pos = atomicAdd(last_idx, 1);
    level_list[pos] = pin;
}

void checkTimingGraph(index_type* level_list_cpu, int num_pins, vector<std::string> pin_names) {
    // check which pins are not in timing graph
    std::set<index_type> pins;
    for (int i = 0; i < num_pins; i++) {
        pins.insert(i);
    }
    for (int i = 0; i < num_pins; i++) {
        index_type pin_id = level_list_cpu[i];
        pins.erase(pin_id);
    }
    for (auto pin_id : pins) {
        printf("Unconnected pin_id: %d, name: %s\n", pin_id, pin_names[pin_id].c_str());
    }
}


#define CUDA_CHECK(msg) do { \
    cudaDeviceSynchronize(); \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) \
        printf("[levelize] CUDA error at %s (line %d): %s\n", msg, __LINE__, cudaGetErrorString(_e)); \
} while(0)

void GPUTimer::levelize() {
    CUDA_CHECK("entry");
    index_type *frontiers, *next_frontiers;
    int *next_num_frontiers, *last_idx;
    int num_frontiers = gtdb.pin_frontiers.size();
    cudaMalloc(&frontiers, num_pins * sizeof(index_type));
    cudaMalloc(&next_frontiers, num_pins * sizeof(index_type));
    cudaMalloc(&next_num_frontiers, sizeof(int));
    cudaMalloc(&last_idx, sizeof(int));
    cudaMemset(next_num_frontiers, 0, sizeof(int));
    cudaMemset(last_idx, 0, sizeof(int));
    cudaMemcpy(frontiers, gtdb.pin_frontiers.data(), num_frontiers * sizeof(index_type), cudaMemcpyHostToDevice);

    level_list_end_cpu.clear();
    level_list_end_cpu.push_back(0);
    int total_num_frontiers = 0;
    while (num_frontiers) {
        total_num_frontiers += num_frontiers;
        level_list_end_cpu.push_back(total_num_frontiers);
        advanceLevel<<<BLOCK_NUMBER(num_pins), BLOCK_SIZE>>>(
            frontiers, next_frontiers, level_list, pin_fanout_list_end, pin_fanout_list, pin_num_fanin, num_frontiers, next_num_frontiers, last_idx);
        CUDA_CHECK("advanceLevel");
        cudaMemcpy(&num_frontiers, next_num_frontiers, sizeof(int), cudaMemcpyDeviceToHost);
        device_copy<index_type><<<1, 1>>>(next_frontiers, frontiers, num_frontiers);
        cudaMemset(next_num_frontiers, 0, sizeof(int));
        // debugPrint<int><<<1, 1>>>(next_num_frontiers, 1);
    }
    cudaMalloc(&level_list_end, level_list_end_cpu.size() * sizeof(index_type));
    cudaMemcpy(level_list_end, level_list_end_cpu.data(), level_list_end_cpu.size() * sizeof(index_type), cudaMemcpyHostToDevice);
    index_type *level_list_cpu = new index_type[total_num_frontiers];
    cudaMemcpy(level_list_cpu, level_list, total_num_frontiers * sizeof(index_type), cudaMemcpyDeviceToHost);
    pin_level_cpu.assign(num_pins, -1);
    for (int level = 0; level + 1 < static_cast<int>(level_list_end_cpu.size()); ++level) {
        const int start = level_list_end_cpu[level];
        const int end = level_list_end_cpu[level + 1];
        for (int i = start; i < end && i < total_num_frontiers; ++i) {
            const int pin = static_cast<int>(level_list_cpu[i]);
            if (pin >= 0 && pin < num_pins) pin_level_cpu[pin] = level;
        }
    }
    delete[] level_list_cpu;
    CUDA_CHECK("exit");
    // checkTimingGraph(level_list_cpu, num_pins, gtdb.pin_names);
}


void GPUTimer::levelize_power(const uint8_t* d_is_seq_output_pin) {
    CUDA_CHECK("power entry");
    if (num_pins <= 0) {
        power_level_list_end_cpu.clear();
        power_level_root_pins_cpu.clear();
        power_level_list_end_cpu.push_back(0);
        return;
    }

    if (power_level_list) {
        cudaFree(power_level_list);
        power_level_list = nullptr;
    }
    if (power_level_list_end) {
        cudaFree(power_level_list_end);
        power_level_list_end = nullptr;
    }
    cudaMalloc(&power_level_list, num_pins * sizeof(index_type));

    index_type *frontiers = nullptr, *next_frontiers = nullptr;
    int *num_frontiers_dev = nullptr, *next_num_frontiers = nullptr, *last_idx = nullptr;
    int *power_num_fanin = nullptr;
    uint8_t *emitted = nullptr;
    cudaMalloc(&frontiers, num_pins * sizeof(index_type));
    cudaMalloc(&next_frontiers, num_pins * sizeof(index_type));
    cudaMalloc(&num_frontiers_dev, sizeof(int));
    cudaMalloc(&next_num_frontiers, sizeof(int));
    cudaMalloc(&last_idx, sizeof(int));
    cudaMalloc(&power_num_fanin, num_pins * sizeof(int));
    cudaMalloc(&emitted, num_pins * sizeof(uint8_t));
    cudaMemset(num_frontiers_dev, 0, sizeof(int));
    cudaMemset(next_num_frontiers, 0, sizeof(int));
    cudaMemset(last_idx, 0, sizeof(int));
    cudaMemset(power_num_fanin, 0, num_pins * sizeof(int));
    cudaMemset(emitted, 0, num_pins * sizeof(uint8_t));

    countPowerFanin<<<BLOCK_NUMBER(num_pins), BLOCK_SIZE>>>(
        pin_forward_arc_list_end, pin_forward_arc_list, timing_arc_to_pin_id, arc_types, arc_id2test_id,
        d_is_seq_output_pin, power_num_fanin, num_pins);
    CUDA_CHECK("countPowerFanin");
    seedPowerFrontiers<<<BLOCK_NUMBER(num_pins), BLOCK_SIZE>>>(power_num_fanin, frontiers, num_frontiers_dev, num_pins);
    CUDA_CHECK("seedPowerFrontiers");

    int num_frontiers = 0;
    cudaMemcpy(&num_frontiers, num_frontiers_dev, sizeof(int), cudaMemcpyDeviceToHost);

    power_level_list_end_cpu.clear();
    power_level_list_end_cpu.push_back(0);
    int total_num_frontiers = 0;
    while (num_frontiers > 0) {
        total_num_frontiers += num_frontiers;
        power_level_list_end_cpu.push_back(total_num_frontiers);
        advancePowerLevel<<<BLOCK_NUMBER(num_frontiers), BLOCK_SIZE>>>(
            frontiers, next_frontiers, power_level_list, pin_forward_arc_list_end, pin_forward_arc_list,
            timing_arc_to_pin_id, arc_types, arc_id2test_id, d_is_seq_output_pin, power_num_fanin, emitted,
            num_frontiers, next_num_frontiers, last_idx, num_pins);
        CUDA_CHECK("advancePowerLevel");
        cudaMemcpy(&num_frontiers, next_num_frontiers, sizeof(int), cudaMemcpyDeviceToHost);
        device_copy<index_type><<<1, 1>>>(next_frontiers, frontiers, num_frontiers);
        cudaMemset(next_num_frontiers, 0, sizeof(int));
    }

    int emitted_count = 0;
    cudaMemcpy(&emitted_count, last_idx, sizeof(int), cudaMemcpyDeviceToHost);
    if (emitted_count < num_pins) {
        appendUnemittedPowerPins<<<BLOCK_NUMBER(num_pins), BLOCK_SIZE>>>(power_level_list, emitted, last_idx, num_pins);
        CUDA_CHECK("appendUnemittedPowerPins");
        cudaMemcpy(&emitted_count, last_idx, sizeof(int), cudaMemcpyDeviceToHost);
        power_level_list_end_cpu.push_back(emitted_count);
    }

    cudaMalloc(&power_level_list_end, power_level_list_end_cpu.size() * sizeof(index_type));
    cudaMemcpy(power_level_list_end, power_level_list_end_cpu.data(),
               power_level_list_end_cpu.size() * sizeof(index_type), cudaMemcpyHostToDevice);

    power_pin_level_cpu.assign(num_pins, -1);
    power_level_root_pins_cpu.clear();
    if (emitted_count > 0) {
        std::vector<index_type> power_level_list_cpu(emitted_count);
        cudaMemcpy(power_level_list_cpu.data(), power_level_list,
                   emitted_count * sizeof(index_type), cudaMemcpyDeviceToHost);
        if (power_level_list_end_cpu.size() > 1) {
            const int root_end = std::min(power_level_list_end_cpu[1], emitted_count);
            power_level_root_pins_cpu.reserve(root_end);
            for (int i = 0; i < root_end; ++i) {
                const int pin = static_cast<int>(power_level_list_cpu[i]);
                if (pin >= 0 && pin < num_pins) power_level_root_pins_cpu.push_back(pin);
            }
        }
        for (int level = 0; level + 1 < static_cast<int>(power_level_list_end_cpu.size()); ++level) {
            const int start = power_level_list_end_cpu[level];
            const int end = power_level_list_end_cpu[level + 1];
            for (int i = start; i < end && i < emitted_count; ++i) {
                const int pin = static_cast<int>(power_level_list_cpu[i]);
                if (pin >= 0 && pin < num_pins) power_pin_level_cpu[pin] = level;
            }
        }
    }

    cudaFree(frontiers);
    cudaFree(next_frontiers);
    cudaFree(num_frontiers_dev);
    cudaFree(next_num_frontiers);
    cudaFree(last_idx);
    cudaFree(power_num_fanin);
    cudaFree(emitted);
    CUDA_CHECK("power exit");
}

} // namespace gt
