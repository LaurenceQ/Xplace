

#include "GPUTimer.h"
#include "gputimer/db/GTDatabase.h"
#include "gputiming.h"
#include "utils.cuh"
namespace gt {

void GPUTimer::initialize() {
    cudaMalloc(&pinCap, num_pins * (NUM_ATTR + 2) * sizeof(float));
    cudaMalloc(&pinWireCap, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&testRelatedAT, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&testRAT, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&testConstraint, num_tests * NUM_ATTR * sizeof(float));
    cudaMalloc(&pinRootRes, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&arcSlew, num_arcs * 2 * NUM_ATTR * sizeof(float));

    cudaMalloc(&net_is_clock, num_nets * sizeof(int));
    cudaMalloc(&level_list, num_pins * sizeof(int));
    cudaMalloc(&primary_outputs, num_POs * sizeof(index_type));

    cudaMemcpy(pinCap, gtdb.pin_capacitance.data(), num_pins * (NUM_ATTR + 2) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(net_is_clock, gtdb.net_is_clock.data(), num_nets * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(primary_outputs, gtdb.endpoints_id.data(), gtdb.endpoints_id.size() * sizeof(index_type), cudaMemcpyHostToDevice);


    allocator = new GPULutAllocator();
    // std::cout<<"gtdb has " << gtdb.liberty_timing_arcs.size() << " timing arcs" << std::endl;
    allocator->AllocateBatch(gtdb.liberty_timing_arcs);
    allocator->CopyToGPU();
    cudaMalloc((void **)&d_allocator, sizeof(GPULutAllocator));
    cudaMemcpy(d_allocator, allocator, sizeof(GPULutAllocator), cudaMemcpyHostToDevice);
    allocator->CopyToGPU(d_allocator);

    logger.info("GPUTimer initialized");

    cudaMalloc(&__pinSlew__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinLoad__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinRAT__, num_pins * NUM_ATTR * sizeof(float));
    cudaMalloc(&__pinAT__, num_pins * NUM_ATTR * sizeof(float));

    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinSlew, __pinSlew__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinLoad, __pinLoad__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinRAT, __pinRAT__, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(pinAT, __pinAT__, num_pins * NUM_ATTR);
}

GPUTimer::~GPUTimer() {
    logger.info("destruct GPUTimer");

    cudaFree(pinCap);
    cudaFree(pinWireCap);
    cudaFree(testRelatedAT);
    cudaFree(testRAT);
    cudaFree(testConstraint);
    cudaFree(pinRootRes);
    cudaFree(arcSlew);

    cudaFree(net_is_clock);
    cudaFree(level_list);
    cudaFree(primary_outputs);

    cudaFree(__pinSlew__);
    cudaFree(__pinLoad__);
    cudaFree(__pinRAT__);
    cudaFree(__pinAT__);

    allocator->~GPULutAllocator();
    cudaFree(d_allocator);

    cudaFree(pin_id2equivalent_cell_id);
    cudaFree(liberty_cell_type2port_list_end);
    cudaFree(pin_id2port_offset_id);
    cudaFree(liberty_port2timing_list_end);
    cudaFree(timing_arc_in_port_id);
    cudaFree(pin_id2timing_arc_list_start);
    cudaFree(pin_id2timing_arc_list_end);
    cudaFree(liberty_port_capacitance);
    cudaFree(arcLambda);
    cudaFree(POLambda);
    cudaFree(sizing_level_list);
    cudaFree(level_pinSlew);
    cudaFree(level_cost);
    cudaFree(candidate_id2pin_id);
    cudaFree(candidate_id2lib_cell_id);
    cudaFree(topo_order2candidate_list_end);
    for(int i = 0;i < level_equivalent_cell_num_pin.size(); i++) cudaFree(level_equivalent_cell_num_pin[i]);
}

void GPUTimer::update_states() {
    cudaMemset(pinImpulse, 0, num_pins * NUM_ATTR * sizeof(float));
    cudaMemset(pinRootRes, 0, num_pins * NUM_ATTR * sizeof(float));
    cudaMemset(pinRootDelay, 0, num_pins * NUM_ATTR * sizeof(float));
    cudaMemset(pinWireCap, 0, num_pins * NUM_ATTR * sizeof(float));

    reset_val<float><<<BLOCK_NUMBER(2 * num_arcs * NUM_ATTR), BLOCK_SIZE>>>(arcDelay, 2 * num_arcs * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(2 * num_arcs * NUM_ATTR), BLOCK_SIZE>>>(arcSlew, 2 * num_arcs * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testRelatedAT, num_tests * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testRAT, num_tests * NUM_ATTR);
    reset_val<float><<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(testConstraint, num_tests * NUM_ATTR);

    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_pin, num_pins * NUM_ATTR);
    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_arc, num_pins * NUM_ATTR);
    reset_val<index_type><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(at_prefix_attr, num_pins * NUM_ATTR);

    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinSlew__, pinSlew, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinLoad__, pinLoad, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinRAT__, pinRAT, num_pins * NUM_ATTR);
    device_copy_batch<float><<<BLOCK_NUMBER(num_pins * NUM_ATTR), BLOCK_SIZE>>>(__pinAT__, pinAT, num_pins * NUM_ATTR);
    cudaDeviceSynchronize();
}

__global__ void update_endpoints_kernel0(float *pinAT, float *testRAT, int *test_id2_arc_id, index_type *timing_arc_from_pin_id, index_type *timing_arc_to_pin_id, float *endpoints0, int num_tests) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int test_idx = idx >> 2;
    const int i = idx & 0b11;
    const int el = i >> 1;
    const int rf = i & 1;
    if (test_idx < num_tests) {
        const int arc_id = test_id2_arc_id[test_idx];
        const int from_pin_id = timing_arc_from_pin_id[arc_id];
        const int to_pin_id = timing_arc_to_pin_id[arc_id];
        if (isnan(pinAT[to_pin_id * NUM_ATTR + i]) || isnan(testRAT[test_idx * NUM_ATTR + i])) return;
        if (el == 0) {
            endpoints0[test_idx * NUM_ATTR + i] = pinAT[to_pin_id * NUM_ATTR + i] - testRAT[test_idx * NUM_ATTR + i];
        } else {
            endpoints0[test_idx * NUM_ATTR + i] = testRAT[test_idx * NUM_ATTR + i] - pinAT[to_pin_id * NUM_ATTR + i];
        }
    }
}

__global__ void update_endpoints_kernel1(float *pinAT, float *pinRAT, index_type *primary_outputs, float *endpoints1, int num_POs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int po_idx = idx >> 2;
    const int i = idx & 0b11;
    const int el = i >> 1;
    if (po_idx < num_POs) {
        const int pin_idx = primary_outputs[po_idx];
        if (isnan(pinAT[pin_idx * NUM_ATTR + i]) || isnan(pinRAT[pin_idx * NUM_ATTR + i])) return;
        if (el == 0) {
            endpoints1[po_idx * NUM_ATTR + i] = pinAT[pin_idx * NUM_ATTR + i] - pinRAT[pin_idx * NUM_ATTR + i];
        } else {
            endpoints1[po_idx * NUM_ATTR + i] = pinRAT[pin_idx * NUM_ATTR + i] - pinAT[pin_idx * NUM_ATTR + i];
        }
    }
}

void GPUTimer::update_endpoints() {
    torch::Tensor endpoints0 = torch::zeros({num_tests, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::kCUDA)).contiguous();
    torch::Tensor endpoints1 = torch::zeros({num_POs, NUM_ATTR}, torch::dtype(torch::kFloat32).device(torch::kCUDA)).contiguous();
    torch::fill_(endpoints0, nanf(""));
    torch::fill_(endpoints1, nanf(""));

    update_endpoints_kernel0<<<BLOCK_NUMBER(num_tests * NUM_ATTR), BLOCK_SIZE>>>(pinAT, testRAT, test_id2_arc_id, timing_arc_from_pin_id, timing_arc_to_pin_id, endpoints0.data_ptr<float>(), num_tests);
    update_endpoints_kernel1<<<BLOCK_NUMBER(num_POs * NUM_ATTR), BLOCK_SIZE>>>(pinAT, pinRAT, primary_outputs, endpoints1.data_ptr<float>(), num_POs);

    endpoint_slacks = torch::cat({endpoints0, endpoints1}, 0).contiguous();
}
void GPUTimer::change_db_sizing(){
  vector<int> cell_type(sizing_level_list_cpu.size());
  cudaMemcpy(cell_type.data(), level_candidate, cell_type.size() * sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 0;i < sizing_level_list_cpu.size(); i++){
    if(cell_type[i] != -1)
        gtdb.change_cell_size(sizing_level_list_cpu[i], cell_type[i]);
        // printf("pin:%d new cell type:%d\n", sizing_level_list_cpu[i], cell_type[i]);
  }
}

void GPUTimer::init_sizing(){
    gtdb.gate_sizing_init();
    sizing_level_list_end_cpu.clear();
    sizing_level_list_end_cpu.push_back(0);
    for(int i = 0; i < level_list_end_cpu.size() - 1; i++){
        for(int j = level_list_end_cpu[i];j < level_list_end_cpu[i + 1]; j++){
            index_type pin_id = level_list_cpu[j];
            if (gtdb.pin_id2equivalent_cell_id[pin_id] == -1 || gtdb.pin_IO_direction[pin_id] != 'o' || gtdb.is_FF_Q[pin_id]) continue; // skip PI and PO
            sizing_level_list_cpu.emplace_back(pin_id);
        }
        // if(sizing_level_list_cpu.size() == sizing_level_list_end_cpu.back()) continue; // no pin in this level
        sizing_level_list_end_cpu.emplace_back(sizing_level_list_cpu.size());
    }
    // for(int i = 0; i < level_list_end_cpu.back(); i++){
    //     index_type pin_id = level_list_cpu[i];
    //     if (gtdb.pin_id2equivalent_cell_id[pin_id] == -1 || gtdb.pin_IO_direction[pin_id] != 'o') continue; // skip PI and PO
    //     sizing_level_list_cpu.emplace_back(pin_id);
    //     // if(sizing_level_list_cpu.size() == sizing_level_list_end_cpu.back()) continue; // no pin in this level
    //     sizing_level_list_end_cpu.emplace_back(sizing_level_list_cpu.size());
    // }
    vector<index_type> candidate_id2pin_id_;
    vector<index_type> candidate_id2lib_cell_id_;
    vector<index_type> topo_order2candidate_list_end_ = {0};
    vector<vector<int>> level_equivalent_cell_num_pin_(sizing_level_list_end_cpu.size());
    level_equivalent_cell_num_pin.resize(sizing_level_list_end_cpu.size());

    // int last_size = 0;
    int max_level_pin_num = 0;
    int max_level_cell_num = 0;
    for(int i = 0;i < sizing_level_list_end_cpu.size() - 1; i++){
        int equivalent_cell_num_pin = 0;
        int num_equivalent_cell = 0;
        level_equivalent_cell_num_pin_[i].clear();
        level_equivalent_cell_num_pin_[i].push_back(0);
        for(int j = sizing_level_list_end_cpu[i];j < sizing_level_list_end_cpu[i+1]; j++){
            index_type pin_id = sizing_level_list_cpu[j];
            index_type cell_id = gtdb.pin_id2equivalent_cell_id[pin_id];
            int equivalen_cell_id = gtdb.lib_cell2equivalen_cell_map[cell_id];
            bool down_size = false;            
            if(!down_size){
                topo_order2candidate_list_end_.emplace_back(topo_order2candidate_list_end_.back() + gtdb.equivalent_cell_list_end[cell_id + 1] - gtdb.equivalent_cell_list_end[cell_id]);
                num_equivalent_cell += gtdb.equivalent_cell_list_end[cell_id + 1] - gtdb.equivalent_cell_list_end[cell_id];
                for(int k = gtdb.equivalent_cell_list_end[cell_id];k < gtdb.equivalent_cell_list_end[cell_id + 1]; k++){
                    candidate_id2pin_id_.emplace_back(pin_id);
                    candidate_id2lib_cell_id_.emplace_back(gtdb.equivalent_cell_list[k]);
                    equivalent_cell_num_pin += gtdb.equivalent_cell_num_pin[cell_id];
                    level_equivalent_cell_num_pin_[i].emplace_back(equivalent_cell_num_pin);
                }
            }
            else{
                int ori_lib_cell_id = gtdb.pin_id2cell_type_id[pin_id];
                // printf("pin:%d ori_cell:%d ori_area:%.4f\n", pin_id, ori_lib_cell_id, gtdb.cell_area[ori_lib_cell_id]);
                int k;
                for(k = gtdb.equivalent_cell_list_end[cell_id];k < gtdb.equivalent_cell_list_end[cell_id + 1]; k++){
                    int eq_id = gtdb.equivalent_cell_list[k];
                    if(gtdb.cell_area[eq_id] >= gtdb.cell_area[ori_lib_cell_id]){
                        break;
                    }
                    candidate_id2pin_id_.emplace_back(pin_id);
                    candidate_id2lib_cell_id_.emplace_back(eq_id);
                    equivalent_cell_num_pin += gtdb.equivalent_cell_num_pin[cell_id];
                    level_equivalent_cell_num_pin_[i].emplace_back(equivalent_cell_num_pin);
                    // printf("cell_type:%d cell_area:%.4f\n", eq_id, gtdb.cell_area[eq_id]);
                }
                topo_order2candidate_list_end_.emplace_back(topo_order2candidate_list_end_.back() + k - gtdb.equivalent_cell_list_end[cell_id]);
                num_equivalent_cell += k - gtdb.equivalent_cell_list_end[cell_id];

            }
        }
        max_level_pin_num = max(max_level_pin_num, equivalent_cell_num_pin);
        max_level_cell_num = max(max_level_cell_num, num_equivalent_cell);
        // std::cout<<"level "<< i << " equivalent pin cnt: "<< equivalent_cell_num_pin << " candidate_id2pin_id size: "<<candidate_id2pin_id_.size() - last_size << std::endl;
        // last_size = candidate_id2pin_id_.size();
        cudaMalloc(&level_equivalent_cell_num_pin[i], level_equivalent_cell_num_pin_[i].size() * sizeof(int));
        cudaMemcpy(level_equivalent_cell_num_pin[i], level_equivalent_cell_num_pin_[i].data(), level_equivalent_cell_num_pin_[i].size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&level_candidate, sizing_level_list_cpu.size() * sizeof(int));
    cudaMalloc(&pin_id2equivalent_cell_id, gtdb.pin_id2equivalent_cell_id.size() * sizeof(int));
    cudaMalloc(&pin_id2cell_type_id, gtdb.pin_id2cell_type_id.size() * sizeof(int));
    cudaMalloc(&liberty_cell_type2port_list_end, gtdb.liberty_cell_type2port_list_end.size() * sizeof(int));
    cudaMalloc(&pin_id2port_offset_id, gtdb.pin_id2port_offset_id.size() * sizeof(int));
    cudaMalloc(&liberty_port2timing_list_end, gtdb.liberty_port2timing_list_end.size() * sizeof(int));
    cudaMalloc(&timing_arc_in_port_id, gtdb.timing_arc_in_port_id.size() * sizeof(int));
    cudaMalloc(&pin_id2timing_arc_list_start, gtdb.pin_id2timing_arc_list_start.size() * sizeof(int));
    cudaMalloc(&pin_id2timing_arc_list_end, gtdb.pin_id2timing_arc_list_end.size() * sizeof(int));
    cudaMalloc(&liberty_port_capacitance, gtdb.liberty_port_capacitance.size() * sizeof(float));
    cudaMalloc(&arcLambda, gtdb.arc_lambda.size() * sizeof(float));
    cudaMalloc(&POLambda, gtdb.po_lambda.size() * sizeof(float));
    cudaMalloc(&sizing_level_list, sizing_level_list_cpu.size() * sizeof(int));
    cudaMalloc(&level_pinSlew, max_level_pin_num * 2 * sizeof(float));
    cudaMalloc(&level_cost, max_level_cell_num * sizeof(float));
    cudaMalloc(&candidate_id2pin_id, candidate_id2pin_id_.size() * sizeof(index_type));
    cudaMalloc(&candidate_id2lib_cell_id, candidate_id2lib_cell_id_.size() * sizeof(index_type));
    cudaMalloc(&topo_order2candidate_list_end, topo_order2candidate_list_end_.size() * sizeof(index_type));

    cudaMemset(level_pinSlew, 0, max_level_pin_num * 2 * sizeof(float));
    cudaMemset(level_cost, 0, max_level_cell_num * sizeof(float));
    cudaMemset(level_candidate, -1, sizing_level_list_cpu.size() * sizeof(int));
    cudaMemcpy(pin_id2cell_type_id, gtdb.pin_id2cell_type_id.data(), gtdb.pin_id2cell_type_id.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pin_id2equivalent_cell_id, gtdb.pin_id2equivalent_cell_id.data(), gtdb.pin_id2equivalent_cell_id.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(liberty_cell_type2port_list_end, gtdb.liberty_cell_type2port_list_end.data(), gtdb.liberty_cell_type2port_list_end.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pin_id2port_offset_id, gtdb.pin_id2port_offset_id.data(), gtdb.pin_id2port_offset_id.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(liberty_port2timing_list_end, gtdb.liberty_port2timing_list_end.data(), gtdb.liberty_port2timing_list_end.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(timing_arc_in_port_id, gtdb.timing_arc_in_port_id.data(), gtdb.timing_arc_in_port_id.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pin_id2timing_arc_list_start, gtdb.pin_id2timing_arc_list_start.data(), gtdb.pin_id2timing_arc_list_start.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pin_id2timing_arc_list_end, gtdb.pin_id2timing_arc_list_end.data(), gtdb.pin_id2timing_arc_list_end.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(liberty_port_capacitance, gtdb.liberty_port_capacitance.data(), gtdb.liberty_port_capacitance.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(arcLambda, gtdb.arc_lambda.data(), gtdb.arc_lambda.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(POLambda, gtdb.po_lambda.data(), gtdb.po_lambda.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sizing_level_list, sizing_level_list_cpu.data(), sizing_level_list_cpu.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(candidate_id2pin_id, candidate_id2pin_id_.data(), candidate_id2pin_id_.size() * sizeof(index_type), cudaMemcpyHostToDevice);
    cudaMemcpy(candidate_id2lib_cell_id, candidate_id2lib_cell_id_.data(), candidate_id2lib_cell_id_.size() * sizeof(index_type), cudaMemcpyHostToDevice);
    cudaMemcpy(topo_order2candidate_list_end, topo_order2candidate_list_end_.data(), topo_order2candidate_list_end_.size() * sizeof(index_type), cudaMemcpyHostToDevice);
}
}  // namespace gt
