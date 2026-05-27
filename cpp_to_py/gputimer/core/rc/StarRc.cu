#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "gputimer/core/rc/RcModels.h"
#include "gputimer/core/utils.cuh"

namespace gt {

__global__ void RCTreeNet(RcStarModel* model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float* x = model->x;
    float* y = model->y;
    const float* pin_offset_x = model->pin_offset_x;
    const float* pin_offset_y = model->pin_offset_y;
    const int* pin2node_map = model->pin2node_map;
    const int* flat_net2pin_start_map = model->flat_net2pin_start_map;
    const int* flat_net2pin_map = model->flat_net2pin_map;
    float* pinLoad = model->pinLoad;
    float* pinImpulse = model->pinImpulse;
    float* pinCap = model->pinCap;
    float* pinRootDelay = model->pinRootDelay;
    float* pinRootRes = model->pinRootRes;
    const int num_nets = model->num_nets;
    const float unit_to_micron = model->unit_to_micron;
    int* net_is_clock = model->net_is_clock;
    const float cf = model->cf;
    const float rf = model->rf;
    if (idx < num_nets) {
        int start_idx = flat_net2pin_start_map[idx];
        int end_idx = flat_net2pin_start_map[idx + 1];
        int root = flat_net2pin_map[start_idx];
        float x_root = x[pin2node_map[root]] + pin_offset_x[root];
        float y_root = y[pin2node_map[root]] + pin_offset_y[root];
        float root_cap = 0;

        // Load
        for (int i = start_idx + 1; i < end_idx; i++) {
            int pin_id = flat_net2pin_map[i];
            float x_pin = x[pin2node_map[pin_id]] + pin_offset_x[pin_id];
            float y_pin = y[pin2node_map[pin_id]] + pin_offset_y[pin_id];
            float dist = abs(x_pin - x_root) + abs(y_pin - y_root);
            float wl = dist / unit_to_micron;
            if (net_is_clock[idx]) wl = 0;
            float pin_cap = cf * wl * 0.5;
            float pin_res = rf * wl;
            root_cap += pin_cap;

            for (int j = 0; j < NUM_ATTR; j++) {
                float pin_cap_lib = pin_cap_attr(pinCap, pin_id, j);
                float load = pinLoad[pin_id * NUM_ATTR + j];

                pinLoad[pin_id * NUM_ATTR + j] = isnan(load) ? pin_cap + pin_cap_lib : load + pin_cap + pin_cap_lib;
                pinRootRes[pin_id * NUM_ATTR + j] = pin_res;
                pinLoad[root * NUM_ATTR + j] = isnan(pinLoad[root * NUM_ATTR + j]) ? pinLoad[pin_id * NUM_ATTR + j]
                                                                                   : pinLoad[root * NUM_ATTR + j] + pinLoad[pin_id * NUM_ATTR + j];
            }
        }
        // Root
        for (int j = 0; j < NUM_ATTR; j++) {
            float pin_cap_lib = pin_cap_attr(pinCap, root, j);
            float load = pinLoad[root * NUM_ATTR + j];
            pinLoad[root * NUM_ATTR + j] = isnan(load) ? root_cap + pin_cap_lib : load + root_cap + pin_cap_lib;
        }
        // Delay
        for (int i = start_idx + 1; i < end_idx; i++) {
            int pin_id = flat_net2pin_map[i];
            for (int j = 0; j < NUM_ATTR; j++) {
                pinRootDelay[pin_id * NUM_ATTR + j] = pinRootRes[pin_id * NUM_ATTR + j] * pinLoad[pin_id * NUM_ATTR + j];
                pinImpulse[pin_id * NUM_ATTR + j] = 0;
            }
        }
        // Impulse
        for (int i = start_idx + 1; i < end_idx; i++) {
            int pin_id = flat_net2pin_map[i];
            for (int j = 0; j < NUM_ATTR; j++) {
                float pin_cap_lib = pin_cap_attr(pinCap, pin_id, j);
                float res = pinRootRes[pin_id * NUM_ATTR + j];
                float cap = pinLoad[pin_id * NUM_ATTR + j];
                float delay = pinRootDelay[pin_id * NUM_ATTR + j];
                pinImpulse[pin_id * NUM_ATTR + j] = sqrt(2 * res * cap * delay - delay * delay);
            }
        }
        if (end_idx - start_idx == 1) {
            for (int j = 0; j < NUM_ATTR; j++) {
                pinRootDelay[root * NUM_ATTR + j] = 0;
                pinImpulse[root * NUM_ATTR + j] = 0;
            }
        }
    }
}

void update_rc_timing_cuda(const RcStarModel& model) {
    const int num_nets = model.num_nets;
    (void)model.num_pins;
    RcStarModel* d_model = nullptr;
    cudaMalloc(&d_model, sizeof(RcStarModel));
    cudaMemcpy(d_model, &model, sizeof(RcStarModel), cudaMemcpyHostToDevice);
    RCTreeNet<<<BLOCK_NUMBER(num_nets), BLOCK_SIZE>>>(d_model);
    cudaFree(d_model);
}


}  // namespace gt
