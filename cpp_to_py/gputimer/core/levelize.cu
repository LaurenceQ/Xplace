
#include "GPUTimer.h"
#include "utils.cuh"
#include "gputimer/db/GTDatabase.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace gt {

struct TimingLevelizeModel {
    index_type *frontiers = nullptr;
    index_type *next_frontiers = nullptr;
    index_type *level_list = nullptr;
    index_type *pin_fanout_list_end = nullptr;
    index_type *pin_fanout_list = nullptr;
    int *pin_num_fanin = nullptr;
    int *next_num_frontiers = nullptr;
    int *last_idx = nullptr;
};

struct PowerLevelizeModel {
    index_type *frontiers = nullptr;
    index_type *next_frontiers = nullptr;
    index_type *level_list = nullptr;
    index_type *pin_forward_arc_list_end = nullptr;
    index_type *pin_forward_arc_list = nullptr;
    index_type *timing_arc_to_pin_id = nullptr;
    const int *arc_types = nullptr;
    const int *arc_id2test_id = nullptr;
    const uint8_t* is_seq_output_pin = nullptr;
    const uint8_t* is_load_pin = nullptr;
    const int* pin2net_map = nullptr;
    const int* net_driver_pin = nullptr;
    const int* flat_net2pin_start_map = nullptr;
    const int* flat_net2pin_map = nullptr;
    int *power_num_fanin = nullptr;
    uint8_t *emitted = nullptr;
    int *num_frontiers = nullptr;
    int *next_num_frontiers = nullptr;
    int *last_idx = nullptr;
    int num_nets = 0;
    int num_arcs = 0;
    int num_pins = 0;
};

__global__ void advanceLevel(TimingLevelizeModel *model, int num_frontiers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_frontiers) {
        index_type pin_id = model->frontiers[idx];
        index_type ptr = atomicAdd(model->last_idx, 1);
        model->level_list[ptr] = pin_id;
        for (index_type i = model->pin_fanout_list_end[pin_id]; i < model->pin_fanout_list_end[pin_id + 1]; i++) {
            index_type fo_pin_id = model->pin_fanout_list[i];
            int prev_num = atomicAdd(&model->pin_num_fanin[fo_pin_id], -1);
            if (prev_num == 1) {
                index_type end = atomicAdd(model->next_num_frontiers, 1);
                model->next_frontiers[end] = fo_pin_id;
            }
        }
    }
}

__device__ bool power_levelize_valid_edge(index_type arc,
                                          const PowerLevelizeModel *model,
                                          int& to_pin) {
    if (arc < 0 || arc >= model->num_arcs) return false;
    if (model->arc_id2test_id && model->arc_id2test_id[arc] != -1) return false;
    to_pin = model->timing_arc_to_pin_id[arc];
    if (to_pin < 0 || to_pin >= model->num_pins) return false;
    // OpenSTA power activity seeds sequential Q/Q_N separately; do not levelize
    // ordinary D/CLK -> Q timing arcs as activity propagation edges.
    if (model->arc_types && model->arc_types[arc] == 1 && model->is_seq_output_pin && model->is_seq_output_pin[to_pin]) return false;
    return true;
}

__global__ void countPowerFanin(PowerLevelizeModel *model) {
    int pin = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_pins = model->num_pins;
    if (pin >= num_pins) return;
    if (model->is_load_pin && model->pin2net_map && model->net_driver_pin && model->flat_net2pin_start_map && model->flat_net2pin_map) {
        const int net = model->pin2net_map[pin];
        if (net >= 0 && net < model->num_nets && model->net_driver_pin[net] == pin) {
            const int start = model->flat_net2pin_start_map[net];
            const int end = model->flat_net2pin_start_map[net + 1];
            if (start >= 0 && end >= start && end <= num_pins) {
                for (int pos = start; pos < end; ++pos) {
                    const int sink = model->flat_net2pin_map[pos];
                    if (sink < 0 || sink >= num_pins || sink == pin || !model->is_load_pin[sink]) continue;
                    atomicAdd(&model->power_num_fanin[sink], 1);
                }
            }
        }
    }
    const index_type arc_start = model->pin_forward_arc_list_end[pin];
    const index_type arc_end = model->pin_forward_arc_list_end[pin + 1];
    if (arc_start < 0 || arc_end < arc_start || arc_end > model->num_arcs) return;
    for (index_type i = arc_start; i < arc_end; i++) {
        int to_pin = -1;
        if (power_levelize_valid_edge(model->pin_forward_arc_list[i], model, to_pin)) {
            atomicAdd(&model->power_num_fanin[to_pin], 1);
        }
    }
}

__global__ void seedPowerFrontiers(PowerLevelizeModel *model) {
    int pin = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_pins = model->num_pins;
    if (pin >= num_pins) return;
    if (model->power_num_fanin[pin] == 0) {
        int pos = atomicAdd(model->num_frontiers, 1);
        model->frontiers[pos] = pin;
    }
}

__global__ void advancePowerLevel(PowerLevelizeModel *model, int num_frontiers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_pins = model->num_pins;
    if (idx >= num_frontiers) return;
    index_type pin_id = model->frontiers[idx];
    if (pin_id < 0 || pin_id >= num_pins) return;
    index_type ptr = atomicAdd(model->last_idx, 1);
    if (ptr >= 0 && ptr < num_pins) {
        model->level_list[ptr] = pin_id;
    }
    model->emitted[pin_id] = 1;
    if (model->is_load_pin && model->pin2net_map && model->net_driver_pin && model->flat_net2pin_start_map && model->flat_net2pin_map) {
        const int net = model->pin2net_map[pin_id];
        if (net >= 0 && net < model->num_nets && model->net_driver_pin[net] == pin_id) {
            const int start = model->flat_net2pin_start_map[net];
            const int end = model->flat_net2pin_start_map[net + 1];
            if (start >= 0 && end >= start && end <= num_pins) {
                for (int pos = start; pos < end; ++pos) {
                    const int sink = model->flat_net2pin_map[pos];
                    if (sink < 0 || sink >= num_pins || sink == pin_id || !model->is_load_pin[sink]) continue;
                    int prev_num = atomicAdd(&model->power_num_fanin[sink], -1);
                    if (prev_num == 1) {
                        index_type end_pos = atomicAdd(model->next_num_frontiers, 1);
                        if (end_pos >= 0 && end_pos < num_pins) {
                            model->next_frontiers[end_pos] = sink;
                        }
                    }
                }
            }
        }
    }
    const index_type arc_start = model->pin_forward_arc_list_end[pin_id];
    const index_type arc_end = model->pin_forward_arc_list_end[pin_id + 1];
    if (arc_start < 0 || arc_end < arc_start || arc_end > model->num_arcs) return;
    for (index_type i = arc_start; i < arc_end; i++) {
        int to_pin = -1;
        if (!power_levelize_valid_edge(model->pin_forward_arc_list[i], model, to_pin)) {
            continue;
        }
        int prev_num = atomicAdd(&model->power_num_fanin[to_pin], -1);
        if (prev_num == 1) {
            index_type end = atomicAdd(model->next_num_frontiers, 1);
            if (end >= 0 && end < num_pins) {
                model->next_frontiers[end] = to_pin;
            }
        }
    }
}

__global__ void appendUnemittedPowerPins(PowerLevelizeModel *model) {
    int pin = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_pins = model->num_pins;
    if (pin >= num_pins || model->emitted[pin]) return;
    int pos = atomicAdd(model->last_idx, 1);
    if (pos >= 0 && pos < num_pins) {
        model->level_list[pos] = pin;
    }
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


static void levelizeThrowCuda(cudaError_t err, const char* msg, int line) {
    if (err == cudaSuccess) return;
    throw std::runtime_error(std::string("[levelize] CUDA error at ") + msg +
                             " (line " + std::to_string(line) + "): " +
                             cudaGetErrorString(err));
}

#define CUDA_CALL(call, msg) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) levelizeThrowCuda(_e, msg, __LINE__); \
} while(0)

#define CUDA_CHECK(msg) do { \
    cudaError_t _sync = cudaDeviceSynchronize(); \
    if (_sync != cudaSuccess) levelizeThrowCuda(_sync, msg, __LINE__); \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) levelizeThrowCuda(_e, msg, __LINE__); \
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
    TimingLevelizeModel level_model;
    level_model.frontiers = frontiers;
    level_model.next_frontiers = next_frontiers;
    level_model.level_list = level_list;
    level_model.pin_fanout_list_end = pin_fanout_list_end;
    level_model.pin_fanout_list = pin_fanout_list;
    level_model.pin_num_fanin = pin_num_fanin;
    level_model.next_num_frontiers = next_num_frontiers;
    level_model.last_idx = last_idx;
    TimingLevelizeModel* d_level_model = nullptr;
    cudaMalloc(&d_level_model, sizeof(TimingLevelizeModel));
    cudaMemcpy(d_level_model, &level_model, sizeof(TimingLevelizeModel), cudaMemcpyHostToDevice);

    level_list_end_cpu.clear();
    level_list_end_cpu.push_back(0);
    int total_num_frontiers = 0;
    while (num_frontiers) {
        total_num_frontiers += num_frontiers;
        level_list_end_cpu.push_back(total_num_frontiers);
        advanceLevel<<<BLOCK_NUMBER(num_pins), BLOCK_SIZE>>>(d_level_model, num_frontiers);
        CUDA_CHECK("advanceLevel");
        cudaMemcpy(&num_frontiers, next_num_frontiers, sizeof(int), cudaMemcpyDeviceToHost);
        device_copy<index_type><<<1, 1>>>(next_frontiers, frontiers, num_frontiers);
        cudaMemset(next_num_frontiers, 0, sizeof(int));
        // debugPrint<int><<<1, 1>>>(next_num_frontiers, 1);
    }
    cudaFree(d_level_model);
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


void GPUTimer::levelize_power(const uint8_t* d_is_seq_output_pin,
                              const int* d_power_arc_types,
                              const int* d_power_arc_id2test_id,
                              const uint8_t* d_is_load_pin,
                              const int* d_pin2net_map,
                              const int* d_net_driver_pin,
                              const int* d_flat_net2pin_start_map,
                              const int* d_flat_net2pin_map) {
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
    CUDA_CALL(cudaMalloc(&power_level_list, num_pins * sizeof(index_type)), "malloc power_level_list");

    index_type *frontiers = nullptr, *next_frontiers = nullptr;
    int *num_frontiers_dev = nullptr, *next_num_frontiers = nullptr, *last_idx = nullptr;
    int *power_num_fanin = nullptr;
    uint8_t *emitted = nullptr;
    CUDA_CALL(cudaMalloc(&frontiers, num_pins * sizeof(index_type)), "malloc frontiers");
    CUDA_CALL(cudaMalloc(&next_frontiers, num_pins * sizeof(index_type)), "malloc next_frontiers");
    CUDA_CALL(cudaMalloc(&num_frontiers_dev, sizeof(int)), "malloc num_frontiers_dev");
    CUDA_CALL(cudaMalloc(&next_num_frontiers, sizeof(int)), "malloc next_num_frontiers");
    CUDA_CALL(cudaMalloc(&last_idx, sizeof(int)), "malloc last_idx");
    CUDA_CALL(cudaMalloc(&power_num_fanin, num_pins * sizeof(int)), "malloc power_num_fanin");
    CUDA_CALL(cudaMalloc(&emitted, num_pins * sizeof(uint8_t)), "malloc emitted");
    CUDA_CALL(cudaMemset(num_frontiers_dev, 0, sizeof(int)), "memset num_frontiers_dev");
    CUDA_CALL(cudaMemset(next_num_frontiers, 0, sizeof(int)), "memset next_num_frontiers");
    CUDA_CALL(cudaMemset(last_idx, 0, sizeof(int)), "memset last_idx");
    CUDA_CALL(cudaMemset(power_num_fanin, 0, num_pins * sizeof(int)), "memset power_num_fanin");
    CUDA_CALL(cudaMemset(emitted, 0, num_pins * sizeof(uint8_t)), "memset emitted");

    PowerLevelizeModel power_model;
    power_model.frontiers = frontiers;
    power_model.next_frontiers = next_frontiers;
    power_model.level_list = power_level_list;
    power_model.pin_forward_arc_list_end = pin_forward_arc_list_end;
    power_model.pin_forward_arc_list = pin_forward_arc_list;
    power_model.timing_arc_to_pin_id = timing_arc_to_pin_id;
    power_model.arc_types = d_power_arc_types ? d_power_arc_types : arc_types;
    power_model.arc_id2test_id = d_power_arc_id2test_id ? d_power_arc_id2test_id : arc_id2test_id;
    power_model.is_seq_output_pin = d_is_seq_output_pin;
    power_model.is_load_pin = d_is_load_pin;
    power_model.pin2net_map = d_pin2net_map;
    power_model.net_driver_pin = d_net_driver_pin;
    power_model.flat_net2pin_start_map = d_flat_net2pin_start_map;
    power_model.flat_net2pin_map = d_flat_net2pin_map;
    power_model.power_num_fanin = power_num_fanin;
    power_model.emitted = emitted;
    power_model.num_frontiers = num_frontiers_dev;
    power_model.next_num_frontiers = next_num_frontiers;
    power_model.last_idx = last_idx;
    power_model.num_nets = num_nets;
    power_model.num_arcs = num_arcs;
    power_model.num_pins = num_pins;
    PowerLevelizeModel* d_power_model = nullptr;
    CUDA_CALL(cudaMalloc(&d_power_model, sizeof(PowerLevelizeModel)), "malloc power levelize model");
    CUDA_CALL(cudaMemcpy(d_power_model, &power_model, sizeof(PowerLevelizeModel), cudaMemcpyHostToDevice),
              "copy power levelize model");

    countPowerFanin<<<BLOCK_NUMBER(num_pins), BLOCK_SIZE>>>(d_power_model);
    CUDA_CHECK("countPowerFanin");
    seedPowerFrontiers<<<BLOCK_NUMBER(num_pins), BLOCK_SIZE>>>(d_power_model);
    CUDA_CHECK("seedPowerFrontiers");

    int num_frontiers = 0;
    cudaMemcpy(&num_frontiers, num_frontiers_dev, sizeof(int), cudaMemcpyDeviceToHost);
    if (num_frontiers < 0) num_frontiers = 0;
    if (num_frontiers > num_pins) num_frontiers = num_pins;

    power_level_list_end_cpu.clear();
    power_level_list_end_cpu.push_back(0);
    int total_num_frontiers = 0;
    while (num_frontiers > 0) {
        total_num_frontiers = std::min(total_num_frontiers + num_frontiers, num_pins);
        power_level_list_end_cpu.push_back(total_num_frontiers);
        advancePowerLevel<<<BLOCK_NUMBER(num_frontiers), BLOCK_SIZE>>>(d_power_model, num_frontiers);
        CUDA_CHECK("advancePowerLevel");
        cudaMemcpy(&num_frontiers, next_num_frontiers, sizeof(int), cudaMemcpyDeviceToHost);
        if (num_frontiers < 0) num_frontiers = 0;
        if (num_frontiers > num_pins) num_frontiers = num_pins;
        device_copy<index_type><<<1, 1>>>(next_frontiers, frontiers, num_frontiers);
        cudaMemset(next_num_frontiers, 0, sizeof(int));
    }

    int emitted_count = 0;
    cudaMemcpy(&emitted_count, last_idx, sizeof(int), cudaMemcpyDeviceToHost);
    if (emitted_count < 0) emitted_count = 0;
    if (emitted_count > num_pins) emitted_count = num_pins;
    if (emitted_count < num_pins) {
        const int unemitted_start = emitted_count;
        appendUnemittedPowerPins<<<BLOCK_NUMBER(num_pins), BLOCK_SIZE>>>(d_power_model);
        CUDA_CHECK("appendUnemittedPowerPins");
        cudaMemcpy(&emitted_count, last_idx, sizeof(int), cudaMemcpyDeviceToHost);
        if (emitted_count < 0) emitted_count = 0;
        if (emitted_count > num_pins) emitted_count = num_pins;
        if (std::getenv("XPLACE_POWER_SORT_UNEMITTED_DESC") != nullptr
            && emitted_count > unemitted_start) {
            std::vector<index_type> tail(emitted_count - unemitted_start);
            cudaMemcpy(tail.data(), power_level_list + unemitted_start,
                       sizeof(index_type) * tail.size(), cudaMemcpyDeviceToHost);
            std::sort(tail.begin(), tail.end(), std::greater<index_type>());
            cudaMemcpy(power_level_list + unemitted_start, tail.data(),
                       sizeof(index_type) * tail.size(), cudaMemcpyHostToDevice);
        }
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
    cudaFree(d_power_model);
    CUDA_CHECK("power exit");
}

} // namespace gt
