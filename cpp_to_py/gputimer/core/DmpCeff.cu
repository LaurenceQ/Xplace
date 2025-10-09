#include "DmpCeff.h"
#include "GPUTimer.h"

namespace gt {
    
__host__ dmp_model::dmp_model(GPUTimer* timer)
        : flat_net2pin_start_map(timer -> flat_net2pin_start_map), 
        flat_net2pin_map(timer -> flat_net2pin_map), 
        pinLoad(timer -> pinLoad), 
        pinRootRes(timer -> pinRootRes), 
        num_pins(timer -> num_pins), 
        num_nets(timer -> num_nets) {
    cudaMalloc(&C1, sizeof(float) * num_nets * NUM_ATTR);
    cudaMalloc(&C2, sizeof(float) * num_nets * NUM_ATTR);
    cudaMalloc(&r_pi, sizeof(float) * num_nets * NUM_ATTR);
}
__host__ dmp_model::~dmp_model(){
    if(C1){
        cudaFree(C1);
        cudaFree(C2);
        cudaFree(r_pi);
        C1 = C2 = r_pi = nullptr;
    } 
}
__device__ void dmp_model::compute_pi_model(int net_id, int el_rf){
    int start_id = flat_net2pin_start_map[net_id];
    int end_id = flat_net2pin_start_map[net_id + 1];
    int root = flat_net2pin_map[start_id];
    double y1 = pinLoad[root * NUM_ATTR + el_rf];
    double y2 = 0;
    double y3 = 0;
    for(int i = start_id + 1; i < end_id; i++){
        int pin_id = flat_net2pin_map[i];
        double cap = pinLoad[pin_id * NUM_ATTR + el_rf];
        double res = pinRootRes[pin_id * NUM_ATTR + el_rf];
        double y2_ = cap * cap * res;
        y2 += -y2_;
        y3 += y2_ * res * cap;
    }
    if(y3 <= 1e-10){
        r_pi[net_id * NUM_ATTR + el_rf] = 0;
        C1[net_id * NUM_ATTR + el_rf] = 0;
        C2[net_id * NUM_ATTR + el_rf] = 0;
    }
    else{
        C1[net_id * NUM_ATTR + el_rf] = static_cast<float>(y2 * y2 / y3); // 远端电容
        C2[net_id * NUM_ATTR + el_rf] = static_cast<float>(y1 - y2 * y2 / y3); // 近端电容
        if (C2[net_id * NUM_ATTR + el_rf] < 0.0)
          C2[net_id * NUM_ATTR + el_rf] = 0.0;
        r_pi[net_id * NUM_ATTR + el_rf] = static_cast<float>(-y3 * y3 / (y2 * y2 * y2));
    }
    // printf("net_id:%d el_rf:%d root:%d rpi:%.4f C1:%.4f C2:%.4f rootLoad:%.4f\n", net_id, el_rf, root, r_pi[net_id*NUM_ATTR+el_rf], C1[net_id*NUM_ATTR+el_rf], C2[net_id*NUM_ATTR+el_rf], pinLoad[root * NUM_ATTR+el_rf]);

}

__global__ void compute_pi_model_kernel(dmp_model *dmp_db){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int net_id = idx >> 2;
    int el_rf = idx & (NUM_ATTR - 1);
    if(net_id < dmp_db -> num_nets){
        dmp_db -> compute_pi_model(net_id, el_rf);
    }
}
void compute_pi_model_cuda(dmp_model *dmp_db, int num_nets){
    
    compute_pi_model_kernel<<<BLOCK_NUMBER(num_nets * NUM_ATTR), BLOCK_SIZE>>>(dmp_db);
 
}
void GPUTimer::initialize_dmp_model(){
    h_dmp_db = new dmp_model(this);
    cudaMalloc(&dmp_db, sizeof(dmp_model));
    cudaMemcpy(dmp_db, h_dmp_db, sizeof(dmp_model), cudaMemcpyHostToDevice);
    
}


}