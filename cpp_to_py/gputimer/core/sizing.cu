#include "gputiming.h"
#include "utils.cuh"
// #include "GPUTimer.h"
// #include "gputimer/db/GTDatabase.h"

namespace gt {

__device__ float getDriveCost(index_type output_pin_id,
                              index_type equivalent_liberty_cell_id,
                              index_type *liberty_cell_type2port_list_end,
                              index_type *pin_id2port_offset_id,
                              index_type *pin_backward_arc_list_end,
                              index_type *pin_backward_arc_list,
                            //   index_type *pin_forward_arc_list_end,
                              index_type *timing_arc_from_pin_id,
                              index_type *timing_arc_id_map,
                              float *liberty_port_capacitance,
                              float *pinCap,
                              float *pinRootRes,
                              float *pinRootDelay,
                              float *pinLoad,
                              float *pinSlew,
                              float *arcLambda,
                              float *level_pinSlew,
                              int candidate_pin_offset,
                              GPULutAllocator *d_allocator) {
    const int el = 1;  // setup
    float cost = 0;
    int last_pin_id = -1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // bool pflag = output_pin_id == 23296;
    for (int i = pin_backward_arc_list_end[output_pin_id]; i < pin_backward_arc_list_end[output_pin_id + 1]; i++) {
        index_type arc_id = pin_backward_arc_list[i];
        index_type input_pin_id = timing_arc_from_pin_id[arc_id];
        if (input_pin_id == last_pin_id) continue;  // input pin no repeat
        last_pin_id = input_pin_id;
        index_type drive_arc_id = pin_backward_arc_list[pin_backward_arc_list_end[input_pin_id]];
        index_type drive_pin_id = timing_arc_from_pin_id[drive_arc_id];
        // int net_connect = pin_forward_arc_list_end[drive_pin_id + 1] - pin_forward_arc_list_end[drive_pin_id];
        // pflag = output_pin_id == 7026;
        // if(pflag)printf("i:%d input_id:%d output_pin_id:%d timing_arc:%d #input_back_edge:%d drive_pin_id:%d\n", i, input_pin_id, output_pin_id, timing_arc_id_map[arc_id * 2 + el], pin_backward_arc_list_end[input_pin_id + 1] - pin_backward_arc_list_end[input_pin_id], drive_pin_id);
        index_type pin_id2port_start = liberty_cell_type2port_list_end[equivalent_liberty_cell_id];
        index_type pin_id2port_offset = pin_id2port_offset_id[input_pin_id];
        index_type port_id = pin_id2port_start + pin_id2port_offset;
        float lc[2];
        if(pin_backward_arc_list_end[drive_pin_id+1] == pin_backward_arc_list_end[drive_pin_id]){
            level_pinSlew[(candidate_pin_offset + pin_id2port_offset) * 2] = pinSlew[drive_pin_id * NUM_ATTR + (el << 1)];
            level_pinSlew[(candidate_pin_offset + pin_id2port_offset) * 2 + 1] = pinSlew[drive_pin_id * NUM_ATTR + (el << 1) + 1];
        }
        else{
            lc[0] = pinLoad[drive_pin_id * NUM_ATTR + (el << 1)] +
                    (isnan(liberty_port_capacitance[6 * port_id + (el << 1)]) ? liberty_port_capacitance[6 * port_id + 4 + el] : liberty_port_capacitance[6 * port_id + (el << 1)]) -
                    (isnan(pinCap[input_pin_id * (NUM_ATTR + 2) + (el << 1)]) ? pinCap[input_pin_id * (NUM_ATTR + 2) + 4 + el] : pinCap[input_pin_id * (NUM_ATTR + 2) + (el << 1)]);

            lc[1] = pinLoad[drive_pin_id * NUM_ATTR + (el << 1) + 1] +
                    (isnan(liberty_port_capacitance[6 * port_id + (el << 1) + 1]) ? liberty_port_capacitance[6 * port_id + 4 + el] : liberty_port_capacitance[6 * port_id + (el << 1) + 1]) -
                    (isnan(pinCap[input_pin_id * (NUM_ATTR + 2) + (el << 1) + 1]) ? pinCap[input_pin_id * (NUM_ATTR + 2) + 4 + el] : pinCap[input_pin_id * (NUM_ATTR + 2) + (el << 1) + 1]);

            level_pinSlew[(candidate_pin_offset + pin_id2port_offset) * 2] = -FLT_MAX;
            level_pinSlew[(candidate_pin_offset + pin_id2port_offset) * 2 + 1] = -FLT_MAX;
            // if(pflag)printf("i:%d input_id:%d output_pin_id:%d timing_arc:%d #input_back_edge:%d drive_pin_id:%d #driver_back_edge:%d lc:%.10f\n", i, input_pin_id, output_pin_id, timing_arc_id_map[arc_id * 2 + el], pin_backward_arc_list_end[input_pin_id + 1] - pin_backward_arc_list_end[input_pin_id], drive_pin_id, pin_backward_arc_list_end[drive_pin_id + 1] - pin_backward_arc_list_end[drive_pin_id], lc[0]);
            for (int j = pin_backward_arc_list_end[drive_pin_id]; j < pin_backward_arc_list_end[drive_pin_id + 1]; j++) {
                index_type driver_input_arc_id = pin_backward_arc_list[j];
                index_type driver_input_pin_id = timing_arc_from_pin_id[driver_input_arc_id];
                int timing_id = timing_arc_id_map[driver_input_arc_id * 2 + el];
                if (timing_id == -1 ) continue;
                // if(pflag)printf("j:%d input_id:%d output_pin_id:%d driver_pin:%d driver_arc:%d driver_input:%d timing_arc:%d arc_lambda_r:%.10f arc_lambda_f:%.10f\n", j, input_pin_id, output_pin_id, drive_pin_id, driver_input_arc_id, driver_input_pin_id, timing_id, arcLambda[driver_input_arc_id * 2], arcLambda[driver_input_arc_id * 2 + 1]);
                for (int orf = 0; orf < 2; orf++) {
                    if(abs(arcLambda[driver_input_arc_id * 2 + orf]) <= 1e-8)continue;
                    float max_delay = -FLT_MAX;
                    for (int irf = 0; irf < 2; irf++) {
                        if(!(d_allocator -> is_transition_defined(timing_id, irf, orf)))continue;
                        float si = pinSlew[driver_input_pin_id * NUM_ATTR + (el << 1) + irf];
                        // if(pflag)printf("idx:%d timing_id:%d outpin:%d input_pin:%d drivepin:%d drive_intput_pin:%d irf:%d orf:%d si:%.10f lc:%.10f\n", idx, timing_id, output_pin_id,  input_pin_id, drive_pin_id, driver_input_pin_id, irf, orf, si, lc[orf]);

                        if (isnan(si)) continue;
                        float so = d_allocator->query(timing_id, irf, orf, si, lc[orf], 1);  // slew output = LUT(slew input, load capacitance)
                        // if(pflag)printf("idx:%d timing_id:%d outpin:%d input_pin:%d drivepin:%d drive_intput_pin:%d irf:%d orf:%d si:%.10f so:%.10f lc:%.10f\n", idx, timing_id, output_pin_id,  input_pin_id, drive_pin_id, driver_input_pin_id, irf, orf, si, so, lc[orf]);
                        if (isnan(so)) continue;                                     
                        level_pinSlew[(candidate_pin_offset + pin_id2port_offset) * 2 + orf] = fmaxf(level_pinSlew[(candidate_pin_offset + pin_id2port_offset) * 2 + orf], so);
                        float delay = d_allocator->query(timing_id, irf, orf, si, lc[orf], 0);
                        // if(flag)printf("delay:%.10f\n", delay);
                        if (isnan(delay)) continue;
                        max_delay = fmaxf(max_delay, delay);
                        // if(pflag)printf("idx:%d cell_type:%d input_pin_id:%d si:%.10f so:%.10f max_delay:%.10f lambda:%.10f\n", idx, equivalent_liberty_cell_id, input_pin_id, si, so, max_delay, arcLambda[driver_input_arc_id * 2 + orf]);
                    }
                    if(max_delay < 0)continue; // MAX_FLT != nan
                    // if(pflag)printf("driveridx:%d cell_type:%d drive_pin_id:%d drive_input_pin_id:%d max_delay:%.5f lambda:%.5f\n", idx, equivalent_liberty_cell_id, drive_pin_id,  driver_input_pin_id, max_delay, arcLambda[driver_input_arc_id * 2 + orf]);
                    cost += max_delay * arcLambda[driver_input_arc_id * 2 + orf];  // TODO: 0.01 should be lambda[orf]
                    
                    // printf("idx:%d pin_id:%d max_delay:%.10f lambda:%.10f\n", idx, output_pin_id, max_delay, arcLambda[driver_input_arc_id * 2 + orf]);
                

                }
            }
        }
        lc[0] = pinLoad[input_pin_id * NUM_ATTR + (el << 1)] +
                   (isnan(liberty_port_capacitance[6 * port_id + (el << 1)]) ? liberty_port_capacitance[6 * port_id + 4 + el] : liberty_port_capacitance[6 * port_id + (el << 1)]) -
                   (isnan(pinCap[input_pin_id * (NUM_ATTR + 2) + (el << 1)]) ? pinCap[input_pin_id * (NUM_ATTR + 2) + 4 + el] : pinCap[input_pin_id * (NUM_ATTR + 2) + (el << 1)]);

        lc[1] = pinLoad[input_pin_id * NUM_ATTR + (el << 1) + 1] +
                   (isnan(liberty_port_capacitance[6 * port_id + (el << 1) + 1]) ? liberty_port_capacitance[6 * port_id + 4 + el] : liberty_port_capacitance[6 * port_id + (el << 1) + 1]) -
                   (isnan(pinCap[input_pin_id * (NUM_ATTR + 2) + (el << 1) + 1]) ? pinCap[input_pin_id * (NUM_ATTR + 2) + 4 + el] : pinCap[input_pin_id * (NUM_ATTR + 2) + (el << 1) + 1]);


        for (int rf = 0; rf <= 1; rf++) {
            int el_rf = (el << 1) + rf;
            float si = level_pinSlew[(candidate_pin_offset + pin_id2port_offset) * 2 + rf];
            float t_delay = lc[rf] * pinRootRes[input_pin_id * NUM_ATTR + el_rf] + pinRootDelay[drive_pin_id * NUM_ATTR + el_rf];
            float res = pinRootRes[input_pin_id * NUM_ATTR + el_rf];
            float imp = sqrt(2 * res * lc[rf] * t_delay - t_delay * t_delay);
            float so = si < 0.0 ? -sqrt(si * si + imp * imp) : sqrt(si * si + imp * imp);
            level_pinSlew[(candidate_pin_offset + pin_id2port_offset) * 2 + rf] = so;
        }
    }
    return cost;
}
__device__ float getGateCost(index_type output_pin_id,
                             index_type equivalent_liberty_cell_id,
                             index_type *liberty_cell_type2port_list_end,
                             index_type *liberty_port2timing_list_end,
                             index_type *pin_id2port_offset_id,
                             index_type *pin_backward_arc_list_end,
                             index_type *pin_backward_arc_list,
                             index_type *timing_arc_from_pin_id,
                             index_type *timing_arc_id_map,
                             index_type *timing_arc_in_port_id,
                             float *liberty_port_capacitance,
                             float *pinCap,
                             float *pinLoad,
                             float *pinSlew,
                             float *arcLambda,
                             float *level_pinSlew,
                             int candidate_pin_offset,
                             GPULutAllocator *d_allocator) {
    const int el = 1;  // setup
    float cost = 0;
    index_type port_start = liberty_cell_type2port_list_end[equivalent_liberty_cell_id];
    index_type output_pin_id2port_offset = pin_id2port_offset_id[output_pin_id];
    int port_id = port_start + output_pin_id2port_offset; // dircted edge, only care about edge to output port. 
    int start = liberty_port2timing_list_end[2 * port_id + el];

    level_pinSlew[(candidate_pin_offset + output_pin_id2port_offset) * 2] = -FLT_MAX;
    level_pinSlew[(candidate_pin_offset + output_pin_id2port_offset) * 2 + 1] = -FLT_MAX;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // bool pflag = output_pin_id == 7026;
    // bool pflag = output_pin_id == 23296;
    // bool pflag = false;
    float lc[2];
    for(int orf = 0; orf <= 1; orf++)
        lc[orf] = pinLoad[output_pin_id * NUM_ATTR + (el << 1) + orf] + (isnan(liberty_port_capacitance[6 * port_id + (el << 1) + orf]) ? liberty_port_capacitance[6 * port_id + 4 + el] : liberty_port_capacitance[6 * port_id + (el << 1) + orf]) 
                    - (isnan(pinCap[output_pin_id * (NUM_ATTR + 2) + (el << 1) + orf]) ? pinCap[output_pin_id * (NUM_ATTR + 2) + 4 + el + orf] : pinCap[output_pin_id * (NUM_ATTR + 2) + (el << 1) + orf]);
    for (int i = pin_backward_arc_list_end[output_pin_id]; i < pin_backward_arc_list_end[output_pin_id + 1]; i++) {
        index_type arc_id = pin_backward_arc_list[i];
        if(timing_arc_id_map[arc_id * 2 + el] == -1)continue;
        index_type input_pin_id = timing_arc_from_pin_id[arc_id];
        index_type input_pin_id2port_offset = pin_id2port_offset_id[input_pin_id];
        int in_port_id = timing_arc_in_port_id[arc_id];
        int timing_id = start + in_port_id;
        for (int orf = 0; orf < 2; orf++) {
            if(abs(arcLambda[arc_id * 2 + orf]) <= 1e-8)continue;
            float max_delay = -FLT_MAX;

            for (int irf = 0; irf < 2; irf++) {
                if(!(d_allocator -> is_transition_defined(timing_id, irf, orf)))continue;
                float si = level_pinSlew[(candidate_pin_offset + input_pin_id2port_offset) * 2 + irf];
                // if(pflag)printf("gateidx:%d timing_id:%d outpin:%d input_pin:%d  irf:%d orf:%d si:%.10f lc:%.10f\n", idx, timing_id, output_pin_id,  input_pin_id, irf, orf, si, lc[orf]);
                if (isnan(si)) continue;
                float so = d_allocator->query(timing_id, irf, orf, si, lc[orf], 1);  // slew output = LUT(slew input, load capacitance)
                if (isnan(so)) continue;
                level_pinSlew[(candidate_pin_offset + output_pin_id2port_offset) * 2 + orf] = fmaxf(level_pinSlew[(candidate_pin_offset + output_pin_id2port_offset) * 2 + orf], so);
                float delay = d_allocator->query(timing_id, irf, orf, si, lc[orf], 0);
                if (isnan(delay)) continue;
                // if(pflag)printf("gateidx:%d cell_type:%d timing_id:%d outpin:%d input_pin:%d irf:%d orf:%d si:%.10f so:%.10f lc:%.10f delay:%.10f\n", idx, equivalent_liberty_cell_id, timing_id, output_pin_id,  input_pin_id, irf, orf, si, so, lc[orf], delay);
                max_delay = fmaxf(max_delay, delay);
            }
            if(max_delay < 0)continue;
            // if(pflag)printf("gateidx:%d cell_type:%d outpin:%d input_pin:%d max_delay:%.5f lambda:%.5f\n", idx, equivalent_liberty_cell_id, output_pin_id,  input_pin_id, max_delay, arcLambda[arc_id * 2 + orf]);
            cost += max_delay * arcLambda[arc_id * 2 + orf];  // TODO: 0.01 should be lambda[orf]
        }
    }
    // if(cost < 1e-8)printf("strange gate cost!!! gateIdx:%d output_pin_id:%d\n", idx, output_pin_id);
    return cost;
}
__device__ float getSinkCost(index_type output_pin_id,
                             index_type *pin_forward_arc_list_end,
                             index_type *pin_forward_arc_list,
                             index_type *timing_arc_to_pin_id,
                             index_type *timing_arc_id_map,
                             index_type *pin_id2port_offset_id,
                             float *pinRootRes,
                             float *pinRootDelay,
                             float *pinLoad,
                             float *pinSlew,
                             float *arcLambda,
                             float *level_pinSlew,
                             int candidate_pin_offset,
                             GPULutAllocator *d_allocator) {
    const int el = 1;  // setup
    float cost = 0;
    index_type output_pin_id2port_offset = pin_id2port_offset_id[output_pin_id];
    // bool flag = output_pin_id == 1498;
    // bool flag = false;
    index_type sink_output_pin = -1;
    int timing_id = -1;
    index_type sink_input_pin = -1;
    float so[2] = {-FLT_MAX, -FLT_MAX};
    float max_delay = -FLT_MAX;
    index_type sink_arc_id = -1;
    float si[2] = {level_pinSlew[(candidate_pin_offset + output_pin_id2port_offset) * 2], level_pinSlew[(candidate_pin_offset + output_pin_id2port_offset) * 2 + 1]};
    for (int i = pin_forward_arc_list_end[output_pin_id]; i < pin_forward_arc_list_end[output_pin_id + 1]; i++) {
        index_type arc_id = pin_forward_arc_list[i];
        sink_input_pin = timing_arc_to_pin_id[arc_id];
        
        for (int rf = 0; rf <= 1; rf++) {
            int el_rf = (el << 1) + rf;
            float t_delay = pinLoad[sink_input_pin * NUM_ATTR + el_rf] * pinRootRes[sink_input_pin * NUM_ATTR + el_rf] + pinRootDelay[output_pin_id * NUM_ATTR + el_rf];
            float res = pinRootRes[sink_input_pin * NUM_ATTR + el_rf];
            float imp = sqrt(2 * res * pinLoad[sink_input_pin * NUM_ATTR + el_rf] * t_delay - t_delay * t_delay);            
            so[rf] = si[rf] < 0.0 ? -sqrt(si[rf] * si[rf] + imp * imp) : sqrt(si[rf] * si[rf] + imp * imp);
        }
        // if(flag)printf("sink_input_pin_id:%d output_pin_id:%d #forward_edge:%d si:%.10f\n",  sink_input_pin, output_pin_id, pin_forward_arc_list_end[sink_input_pin + 1] - pin_forward_arc_list_end[sink_input_pin], so[0]);

        for (int j = pin_forward_arc_list_end[sink_input_pin]; j < pin_forward_arc_list_end[sink_input_pin + 1]; j++) {
            sink_arc_id = pin_forward_arc_list[j];
            sink_output_pin = timing_arc_to_pin_id[sink_arc_id];

            timing_id = timing_arc_id_map[sink_arc_id * 2 + el];

            if (timing_id == -1) continue;

            for (int orf = 0; orf < 2; orf++) {
                if(arcLambda[sink_arc_id * 2 + orf] <= 1e-8)continue;
                max_delay = -FLT_MAX;
                for (int irf = 0; irf < 2; irf++) {
                    if(!(d_allocator -> is_transition_defined(timing_id, irf, orf)))continue;
                    
                    float lc = pinLoad[sink_output_pin * NUM_ATTR + (el << 1) + orf];
                    if (isnan(so[irf])) continue;
                    float delay = d_allocator->query(timing_id, irf, orf, so[irf], lc, 0);
                    // if(flag)printf("sink_input_pin_id:%d output_pin_id:%d sink_output_pin:%d timing_arc:%d si:%.10f lc:%.10f delay:%.10f\n",  sink_input_pin, output_pin_id, sink_output_pin, timing_arc_id_map[sink_arc_id * 2 + el], so[irf], lc, delay);

                    if (isnan(delay)) continue;
                    max_delay = fmaxf(max_delay, delay);
                }
                if(max_delay < 0)continue;
                cost += max_delay * arcLambda[sink_arc_id * 2 + orf];  // TODO: 0.01 should be lambda[orf]
            }
        }
    }

    // if(cost <= 1e-8 && pin_forward_arc_list_end[sink_input_pin] != pin_forward_arc_list_end[sink_input_pin + 1])printf("strange 0 sink cost!!! sink_input_pin_id:%d output_pin_id:%d sink_output_pin:%d timing_arc:%d si:%.10f lc:%.10f delay:%.10f\n",  sink_input_pin, output_pin_id, sink_output_pin, timing_arc_id_map[sink_arc_id * 2 + el], so[0], pinLoad[sink_output_pin * NUM_ATTR + (el << 1)], max_delay);

    return cost;
}
__global__ void evaluateEquivalentCell(index_type *level_list,
                                       index_type *candidate_id2pin_id,
                                       index_type *candidate_id2lib_cell_id,
                                       index_type *pin_backward_arc_list_end,
                                       index_type *pin_backward_arc_list,
                                       index_type *pin_forward_arc_list_end,
                                       index_type *pin_forward_arc_list,
                                       index_type *timing_arc_from_pin_id,
                                       index_type *timing_arc_to_pin_id,
                                       index_type *pin_id2equivalent_cell_id,
                                       index_type *equivalent_cell_num_pin,
                                       index_type *liberty_cell_type2port_list_end,
                                       index_type *liberty_port2timing_list_end,
                                       index_type *timing_arc_in_port_id,
                                       index_type *pin_id2port_offset_id,
                                       float *liberty_port_capacitance,
                                       float *pinRootRes,
                                       float *pinRootDelay,
                                       float *pinCap,
                                       float *pinSlew,
                                       float *pinLoad,
                                       float *arcLambda,
                                       float *level_pinSlew,
                                       float *level_cost,
                                       int *timing_arc_id_map,
                                       GPULutAllocator *d_allocator,
                                       int level_start_offset,
                                       int num_candidate,
                                       int candidate_start_offset
                                       ) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidate) return;
    const int pin_id = candidate_id2pin_id[idx + candidate_start_offset];
    index_type cell_type_id = candidate_id2lib_cell_id[idx + candidate_start_offset];
    if(cell_type_id == -1){
        level_cost[idx] = FLT_MAX;
        return ;
    }
    index_type candidate_pin_offset = equivalent_cell_num_pin[idx];
    float drive_cost = getDriveCost(pin_id,
                                   cell_type_id,
                                   liberty_cell_type2port_list_end,
                                   pin_id2port_offset_id,
                                   pin_backward_arc_list_end,
                                   pin_backward_arc_list,
                                //    pin_forward_arc_list_end,
                                   timing_arc_from_pin_id,
                                   timing_arc_id_map,
                                   liberty_port_capacitance,
                                   pinCap,
                                   pinRootRes,
                                   pinRootDelay,
                                   pinLoad,
                                   pinSlew,
                                   arcLambda,
                                   level_pinSlew,
                                   candidate_pin_offset,
                                   d_allocator); 
    float gate_cost = getGateCost(pin_id,
                                  cell_type_id,
                                  liberty_cell_type2port_list_end,
                                  liberty_port2timing_list_end,
                                  pin_id2port_offset_id,
                                  pin_backward_arc_list_end,
                                  pin_backward_arc_list,
                                  timing_arc_from_pin_id,
                                  timing_arc_id_map,
                                  timing_arc_in_port_id,
                                  liberty_port_capacitance,
                                  pinCap,
                                  pinLoad,
                                  pinSlew,
                                  arcLambda,
                                  level_pinSlew,
                                  candidate_pin_offset,
                                  d_allocator);
    float sink_cost = getSinkCost(pin_id,
                                  pin_forward_arc_list_end,
                                  pin_forward_arc_list,
                                  timing_arc_to_pin_id,
                                  timing_arc_id_map,
                                  pin_id2port_offset_id,
                                  pinRootRes,
                                  pinRootDelay,
                                  pinLoad,
                                  pinSlew,
                                  arcLambda,
                                  level_pinSlew,
                                  candidate_pin_offset,
                                  d_allocator);

    level_cost[idx] = drive_cost + gate_cost + sink_cost;
    // if(pin_id == 23296)
    // printf("idx:%d level_start_offset:%d pin_id:%d cell_type:%d drive:%.10f gate.%.10f sink:%.10f cost:%.10f \n", idx, level_start_offset, pin_id, cell_type_id, drive_cost, gate_cost, sink_cost, level_cost[idx]);
}
__global__ void evaluateLevelPin(index_type *level_list,
                                 index_type *topo_order2candidate_list_end,
                                 index_type *candidate_id2lib_cell_id,
                                 index_type *pin_backward_arc_list_end,
                                 index_type *pin_backward_arc_list,
                                 index_type *pin_forward_arc_list_end,
                                 index_type *pin_forward_arc_list,
                                 index_type *timing_arc_from_pin_id,
                                 index_type *timing_arc_to_pin_id,
                                 index_type *pin_id2equivalent_cell_id,
                                 int *pin_id2cell_type_id,
                                 index_type *equivalent_cell_num_pin,
                                 index_type *liberty_cell_type2port_list_end,
                                 index_type *liberty_port2timing_list_end,
                                 index_type *timing_arc_in_port_id,
                                 index_type *pin_id2port_offset_id,
                                 float *liberty_port_capacitance,
                                 float *pinCap,
                                 float *pinSlew,
                                 float *pinLoad,
                                 float *level_pinSlew,
                                 float *level_cost,
                                 int *level_candidate,
                                 int *timing_arc_id_map,
                                 GPULutAllocator *d_allocator,
                                 int level_start_offset,
                                 int num_pins_level
                                 ) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pins_level) {
        const int el = 1;
        index_type pin_id = level_list[level_start_offset + idx];
        float best_cost = FLT_MAX;
        float ori_cost = -FLT_MAX;
        index_type best_candidate_id = -1;
        for (int i = topo_order2candidate_list_end[level_start_offset + idx]; i < topo_order2candidate_list_end[level_start_offset + idx + 1]; i++) {
            index_type cell_type_id = candidate_id2lib_cell_id[i];
            if (cell_type_id == -1) continue;
            float cost = level_cost[i-topo_order2candidate_list_end[level_start_offset]]; //better unify level system, level_cost is with equivalent_cell_num_pin
            if(pin_id2cell_type_id[pin_id] == cell_type_id) ori_cost = cost;
            if (cost < best_cost) {
                best_cost = cost;
                best_candidate_id = i;
            }
        }
        if (best_candidate_id == -1 || best_cost >= 0.8 * ori_cost) return;
        int best_lib_cell_id = candidate_id2lib_cell_id[best_candidate_id];
        level_candidate[level_start_offset + idx] = best_lib_cell_id;
        best_candidate_id -= topo_order2candidate_list_end[level_start_offset];
        index_type last_pin_id = -1;
        int candidate_pin_offset = equivalent_cell_num_pin[best_candidate_id];
        int pin_id2port_start = liberty_cell_type2port_list_end[best_lib_cell_id];
        int output_pin_id2port_offset = pin_id2port_offset_id[pin_id];
        int output_port_id = pin_id2port_start + output_pin_id2port_offset;
        int start = liberty_port2timing_list_end[2 * output_port_id + el];
        float old_cap = -1;
        for (int i = pin_backward_arc_list_end[pin_id]; i < pin_backward_arc_list_end[pin_id + 1]; i++) {
            index_type arc_id = pin_backward_arc_list[i];
            if (timing_arc_id_map[arc_id * 2 + el] == -1) continue;
            index_type input_pin_id = timing_arc_from_pin_id[arc_id];
            int input_pin_id2port_offset = pin_id2port_offset_id[input_pin_id];
            int input_port_id = pin_id2port_start + input_pin_id2port_offset;
            int in_port_id = timing_arc_in_port_id[arc_id];
            timing_arc_id_map[arc_id * 2 + el] = start + in_port_id;
            if (input_pin_id == last_pin_id) continue;
            last_pin_id = input_pin_id;
            index_type drive_arc_id = pin_backward_arc_list[pin_backward_arc_list_end[input_pin_id]];
            index_type drive_pin_id = timing_arc_from_pin_id[drive_arc_id];
            old_cap = pinCap[6 * input_pin_id + 4 + el];
            pinCap[6 * input_pin_id + 4 + el] = liberty_port_capacitance[6 * input_port_id + el * 3 + 2];
            for (int rf = 0; rf <= 1; rf++){
                int el_rf = (el << 1) + rf;
                pinSlew[input_pin_id * NUM_ATTR + el_rf] = level_pinSlew[(candidate_pin_offset + input_pin_id2port_offset) * 2 + rf];
                atomicAdd(&pinLoad[input_pin_id * NUM_ATTR + el_rf],   -(isnan(pinCap[6 * input_pin_id + el_rf]) ? old_cap : pinCap[6 * input_pin_id + el_rf]));
                atomicAdd(&pinLoad[drive_pin_id * NUM_ATTR + el_rf],   -(isnan(pinCap[6 * input_pin_id + el_rf]) ? old_cap : pinCap[6 * input_pin_id + el_rf]));
                pinCap[6 * input_pin_id + el_rf] = liberty_port_capacitance[6 * input_port_id + el * 3 + rf];
                atomicAdd(&pinLoad[input_pin_id * NUM_ATTR + el_rf],   (isnan(pinCap[6 * input_pin_id + el_rf]) ? pinCap[6 * input_pin_id + 4 + el] : pinCap[6 * input_pin_id + el_rf]));
                atomicAdd(&pinLoad[drive_pin_id * NUM_ATTR + el_rf],   (isnan(pinCap[6 * input_pin_id + el_rf]) ? pinCap[6 * input_pin_id + 4 + el] : pinCap[6 * input_pin_id + el_rf]));
            }
        }
        old_cap = pinCap[6 * pin_id + 4 + el];
        pinCap[6 * pin_id + 4 + el] = liberty_port_capacitance[6 * output_port_id + el * 3 + 2];
        for (int rf = 0; rf <= 1; rf++){
            int el_rf = (el << 1) + rf;
            pinSlew[pin_id * NUM_ATTR + el_rf] = level_pinSlew[(candidate_pin_offset + output_pin_id2port_offset) * 2 + rf];
            atomicAdd(&pinLoad[pin_id * NUM_ATTR + el_rf],   -(isnan(pinCap[6 * pin_id + el_rf]) ? old_cap : pinCap[6 * pin_id + el_rf]));
            pinCap[6 * pin_id + el_rf] = liberty_port_capacitance[6 * output_port_id + el * 3 + rf];
            atomicAdd(&pinLoad[pin_id * NUM_ATTR + el_rf],   (isnan(pinCap[6 * pin_id + el_rf]) ? pinCap[6 * pin_id + 4 + el] : pinCap[6 * pin_id + el_rf]));

        }

    }
}
// __global__ void cachePinLoad(
//                             int *level_list,
//                             index_type *pin_backward_arc_list_end,
//                             index_type *pin_backward_arc_list,
//                             index_type *timing_arc_from_pin_id,
//                             index_type *timing_arc_id_map,
//                             float *pinCap,
//                             float *pinLoad,
//                             int level_start_offset, 
//                             int num_pins_level,
//                             int flag
//                         ) {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int pin_idx = idx >> 1;
//     if (pin_idx >= num_pins_level) return ;
//     const int el = 1;  // setup
//     const int rf = idx & 1;
//     const int el_rf = (el << 1) + rf;
//     int last_pin_id = -1;
//     index_type output_pin_id = level_list[level_start_offset + pin_idx];
//     for (int i = pin_backward_arc_list_end[output_pin_id]; i < pin_backward_arc_list_end[output_pin_id + 1]; i++) {
//         index_type arc_id = pin_backward_arc_list[i];
//         if (timing_arc_id_map[arc_id * 2 + el] == -1) continue;
//         index_type input_pin_id = timing_arc_from_pin_id[arc_id];
//         if (input_pin_id == last_pin_id) continue;
//         last_pin_id = input_pin_id;
//         index_type drive_arc_id = pin_backward_arc_list[pin_backward_arc_list_end[input_pin_id]];
//         index_type drive_pin_id = timing_arc_from_pin_id[drive_arc_id];
//         atomicAdd(&pinLoad[input_pin_id * NUM_ATTR + el_rf],  flag * (isnan(pinCap[6 * input_pin_id + el_rf]) ? pinCap[6 * input_pin_id + 4 + el] : pinCap[6 * input_pin_id + el_rf]));
//         atomicAdd(&pinLoad[drive_pin_id * NUM_ATTR + el_rf],  flag * (isnan(pinCap[6 * input_pin_id + el_rf]) ? pinCap[6 * input_pin_id + 4 + el] : pinCap[6 * input_pin_id + el_rf]));
//     }
//     atomicAdd(&pinLoad[output_pin_id * NUM_ATTR + el_rf],  flag * (isnan(pinCap[6 * output_pin_id + el_rf]) ? pinCap[6 * output_pin_id + 4 + el] : pinCap[6 * output_pin_id + el_rf]));
// }

// __global__ void updatePinImpulse(   int *level_list,
//                                     index_type *pin_backward_arc_list_end,
//                                     index_type *pin_backward_arc_list,
//                                     index_type *timing_arc_from_pin_id,
//                                     float *pinCap,
//                                     float *pinLoad,
//                                     float *pinImpulse,
//                                     float *pinRootDelay,
//                                     float *pinRootRes,
//                                     int level_start_offset,
//                                     int num_pins){
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int pin_idx = idx >> 1;
//     const int el = 1;
//     const int rf = idx & 1;
//     const int el_rf = (el << 1) + rf;
//     if(pin_idx < num_pins){
//         int pin_id = level_list[pin_idx + level_start_offset];
//         int root_pin = timing_arc_from_pin_id[pin_backward_arc_list[pin_backward_arc_list_end[pin_id]]];
//         float t_delay = pinLoad[pin_id * NUM_ATTR + el_rf] * pinRootRes[pin_id * NUM_ATTR + el_rf];
//         pinRootDelay[pin_id] = pinRootDelay[root_pin * NUM_ATTR + el_rf] + t_delay;
//         float res = pinRootRes[pin_id * NUM_ATTR + el_rf];
//         float cap = pinLoad[pin_id * NUM_ATTR + el_rf];
//         float delay = pinRootDelay[pin_id * NUM_ATTR + el_rf];
//         pinImpulse[pin_id * NUM_ATTR + el_rf] = sqrt(2 * res * cap * delay - delay * delay);
//     }
// }
__global__ void updateArcLambda(index_type *timing_arc_from_pin_id,
                                index_type *timing_arc_to_pin_id,
                                int *arc_types,
                                int *timing_arc_id_map,
                                float *pinAt,
                                float *pinRat,
                                float *arcDelay,
                                float *arcLambda,
                                float *fanout_sum,
                                float clock_period,
                                GPULutAllocator *d_allocator
                            ){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_id = idx; 
    const int arc_type = arc_types[arc_id];
    if (arc_type == 1) {
        int el = 1;
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        if (timing_id == -1){
            arcLambda[arc_id * 2] = 0;
            arcLambda[arc_id * 2 + 1] = 0;
            return ;
        }

        if(!d_allocator->d_is_constraint[timing_id]){
            int from_pin_id = timing_arc_from_pin_id[arc_id];
            int to_pin_id = timing_arc_to_pin_id[arc_id];
            for(int orf = 0; orf <= 1; orf++){
                float arc_to_at = -FLT_MAX;
                for(int irf = 0; irf <= 1; irf++){
                    int fel_rf = (el << 1) | irf;
                    int el_iorf = (el << 2) + (irf << 1) + orf;
                    if(d_allocator -> is_transition_defined(timing_id, irf, orf)){
                        arc_to_at = max(arc_to_at, pinAt[from_pin_id * NUM_ATTR + fel_rf] + arcDelay[arc_id * 2 * NUM_ATTR + el_iorf]);
                    }
                }
                int tel_rf = (el << 1) | orf;
                // float slack = pinRat[to_pin_id * NUM_ATTR + tel_rf] - pinAt[to_pin_id * NUM_ATTR + tel_rf];
                // if(from_pin_id == 16388 || from_pin_id == 14844)printf("unconstraint from_pin:%d to_pin:%d at:%.10f rat:%.10f slack:%.10f\n", from_pin_id, to_pin_id, pinAt[to_pin_id * NUM_ATTR + tel_rf], pinRat[to_pin_id * NUM_ATTR + tel_rf], slack);

                if(isnan(arc_to_at) || isnan(pinAt[to_pin_id * NUM_ATTR + tel_rf])) arcLambda[arc_id * 2 + orf] = 0;
                else {
                    arcLambda[arc_id * 2 + orf] *= arc_to_at / pinAt[to_pin_id * NUM_ATTR + tel_rf];
                }
                // if(isnan(arcLambda[arc_id * 2 + orf]))printf("nan arclambda arc_id:%d lambda:%.10f arc to at:%.10f pinAT:%.10f\n", arc_id, arcLambda[arc_id * 2 + orf], arc_to_at, pinAt[to_pin_id * NUM_ATTR + tel_rf]);
                if(arcLambda[arc_id * 2 + orf] < 0)arcLambda[arc_id * 2 + orf] = 0;
            }
        }
        // else{
            
        //     for(int irf = 0; irf <= 1; irf++){
        //         int orf = d_allocator -> is_transition_defined(timing_id, irf, irf) ? irf : irf ^ 1; 
        //         const int tel_rf = 2 + orf;
        //         float slack = pinRat[to_pin_id * NUM_ATTR + tel_rf] - pinAt[to_pin_id * NUM_ATTR + tel_rf];
        //         if(from_pin_id == 16388 || from_pin_id == 14844)printf("from_pin:%d to_pin:%d at:%.10f rat:%.10f slack:%.10f\n", from_pin_id, to_pin_id, pinAt[to_pin_id * NUM_ATTR + tel_rf], pinRat[to_pin_id * NUM_ATTR + tel_rf], slack);
        //         if(isnan(slack)){
        //             arcLambda[arc_id * 2 + irf] = 0;
        //             printf("arc %d fr_p %d to_p %d nan slack rat %.10f at %.10f\n", arc_id, from_pin_id, to_pin_id, pinRat[to_pin_id * NUM_ATTR + tel_rf], pinAt[to_pin_id * NUM_ATTR + tel_rf]);
        //             continue;
        //         }
        //         if(slack < 0){
        //             arcLambda[arc_id * 2 + irf] *= sqrt(1.0 - slack / clock_period);
        //         }
        //         else arcLambda[arc_id * 2 + irf] *= (1.0 / (1.0 + slack / clock_period)) * (1.0 / (1.0 + slack / clock_period));
        //         if(arcLambda[arc_id * 2 + irf] < 0) arcLambda[arc_id * 2 + irf] = 0;
        //         fanout_sum[from_pin_id * 2 + irf] = arcLambda[arc_id * 2 + irf];
        //     }
        // }
    }
}
__global__ void updatePOLambda( int *primary_outputs,
                                float *pinAt, 
                                float *pinRat,
                                float *POLambda,
                                float *fanout_sum,
                                float clock_period,
                                int num_PO){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int PO_id = idx >> 1;
    if(PO_id >= num_PO)return;
    int pin_id = primary_outputs[PO_id];
    int el = 1;
    int rf = idx & 1;
    int el_rf = (el << 1) + rf;
    float slack = pinRat[pin_id * NUM_ATTR + el_rf] - pinAt[pin_id * NUM_ATTR + el_rf];
    if(isnan(slack)){
        POLambda[idx] = 0;
        // printf("PO %d pin %d nan slack\n", PO_id, pin_id);
        return ;
    }
    if(slack < 0){
        POLambda[idx] *= sqrt(1.0 - slack / clock_period);
    }
    else POLambda[idx] *= (1.0 / (1.0 + slack / clock_period)) * (1.0 / (1.0 + slack / clock_period));
    if(POLambda[idx] < 0) POLambda[idx] = 0;
    fanout_sum[pin_id * 2 + rf] = POLambda[idx];
    // if(isnan(POLambda[idx]))printf("PO_id:%d pin_id:%d lambda:%.10f slack:%.10f clk_period:%.10f\n", PO_id, pin_id, POLambda[idx], slack, clock_period);
}
// __global__ void KKTProjectionPO(int *primary_outputs,
//                                 int *pin_backward_arc_list,
//                                 int *pin_backward_arc_list_end,
//                                 int *timing_arc_from_pin_id,
//                                 int *timing_arc_id_map,
//                                 int *arc_types,
//                                 float *pinAt, 
//                                 float *pinRat,
//                                 float *POLambda,
//                                 float *arcLambda,
//                                 int num_PO){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int PO_id = idx >> 1;
//     if(PO_id >= num_PO)return;
//     int pin_id = primary_outputs[PO_id];
//     int el = 1;
//     int rf = idx & 1;
//     int el_rf = (el << 1) + rf;
//     float fanin_sum = 0;
//     for(int i = pin_backward_arc_list_end[pin_id]; i < pin_backward_arc_list_end[pin_id + 1]; i++){
//         int arc_id = pin_backward_arc_list[i];
//         fanin_sum += arcLambda[arc_id * 2 + rf];
//     }
//     float scale = POLambda[idx] / fanin_sum;
//     if(pin_id == 1451){
//         printf("POlambda:%.10f fanin_sum:%.10f\n", POLambda[idx], fanin_sum);
//     }
//     if (abs(scale) <= 1e-8) scale = 0;
//     for(int i = pin_backward_arc_list_end[pin_id]; i < pin_backward_arc_list_end[pin_id + 1]; i++){
//         int arc_id = pin_backward_arc_list[i];
//         // if(timing_arc_id_map[arc_id * 2 + el] == -1)continue; // PO is connected by nets edge
//         arcLambda[arc_id * 2 + rf] *= scale;
//         if(arcLambda[arc_id * 2 + rf] < 0)arcLambda[arc_id * 2 + rf] = 0;

//         // if(arcLambda[arc_id * 2 + rf] <= 1e-8 || arc_types[arc_id] == 0){
//         //     printf("PO0 leads to 0arcs arc_id:%d lambda:%.10f scale:%.10f POlamb:%.10f fanin_sum:%.10f\n", arc_id, arcLambda[arc_id * 2 + rf], scale, POLambda[idx], fanin_sum);
//         // }
//     }
// }
__global__ void KKTProjectionGate(
                                int *level_list,
                                int *pin_forward_arc_list_end,
                                int *pin_forward_arc_list,
                                int *pin_backward_arc_list,
                                int *pin_backward_arc_list_end,
                                int *timing_arc_from_pin_id,
                                int *timing_arc_to_pin_id,
                                int *timing_arc_id_map,
                                int *arc_types,
                                float *arcDelay,
                                float *arcLambda,
                                float *fanout_sum,
                                int level_start_offset,
                                int num_pins,
                                int level,
                                GPULutAllocator *d_allocator
                            ){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pin_idx = idx >> 1;
    if(pin_idx >= num_pins)return;
    int pin_id = level_list[pin_idx + level_start_offset];
    if(pin_backward_arc_list_end[pin_id] == pin_backward_arc_list_end[pin_id + 1])return ;
    int el = 1;
    int rf = idx & 1;
    // int el_rf = (el << 1) + rf;
    int arc_type = -1;
    bool is_FF_D = false;
    // bool flag = pin_id == 2508;
    for(int i = pin_forward_arc_list_end[pin_id]; i < pin_forward_arc_list_end[pin_id + 1]; i++){
        int arc_id = pin_forward_arc_list[i];
        arc_type = arc_types[arc_id];
        int out_pin_id = timing_arc_to_pin_id[arc_id];
        if(arc_type == 1){
            int timing_id = timing_arc_id_map[arc_id * 2 + el];
            if(timing_id == -1)continue;
            // printf("level:%d arc_id:%d fr:%d to:%d arctype:%d arcLamb:%.10f\n", level, arc_id, pin_id, out_pin_id, arc_type, arcLambda[arc_id * 2 + rf]);

            if(!(d_allocator -> is_transition_defined(timing_id, rf, rf)))
                fanout_sum[pin_id * 2 + rf] += arcLambda[arc_id * 2 + (rf ^ 1)];
            else 
                fanout_sum[pin_id * 2 + rf] += arcLambda[arc_id * 2 + rf];
            is_FF_D |= d_allocator->d_is_constraint[timing_id];
        } 
        else {
            fanout_sum[pin_id * 2 + rf] += fanout_sum[out_pin_id * 2 + rf];
            // printf("level:%d arc_id:%d fr:%d to:%d arctype:%d to_fanout_sum:%.10f\n", level, arc_id, pin_id, out_pin_id, arc_type, fanout_sum[out_pin_id * 2 + rf]);
        }
    }
    if(arc_type == 1)return ;
    float fanin_sum = 0;
    // bool is_FF_Q = false;
    for(int i = pin_backward_arc_list_end[pin_id]; i < pin_backward_arc_list_end[pin_id + 1]; i++){
        int arc_id = pin_backward_arc_list[i];
        if(timing_arc_id_map[arc_id * 2 + el] == -1)continue;
        // is_FF_Q |= d_allocator->d_is_constraint[timing_arc_id_map[arc_id * 2 + el]];
        fanin_sum += arcLambda[arc_id * 2 + rf];
    }
    // if(is_FF_Q)return ;
    float scale = fanout_sum[pin_id * 2 + rf] / fanin_sum;
    if (scale <= 1e-8) scale = 0;
    for(int i = pin_backward_arc_list_end[pin_id]; i < pin_backward_arc_list_end[pin_id + 1]; i++){
        int arc_id = pin_backward_arc_list[i];
        if(timing_arc_id_map[arc_id * 2 + el] == -1)continue;
        arcLambda[arc_id * 2 + rf] *= scale;
        if(arcLambda[arc_id * 2 + rf] < 0)arcLambda[arc_id * 2 + rf] = 0;
        // if(flag || arc_id == 13784 || fanout_sum[pin_id * 2 + rf] < 1e-8){
        //     int from_pin = timing_arc_from_pin_id[arc_id];
        //     int to_pin = timing_arc_to_pin_id[arc_id];
        //     printf("level:%d fr_pin:%d to_pin:%d arc_id:%d lambda:%.10f scale:%.10f fanout_sum:%.10f fanin_sum:%.10f isFFD:%d isFFQ:%d\n", level, from_pin, to_pin, arc_id, arcLambda[arc_id * 2 + rf], scale, fanout_sum[pin_id * 2 + rf], fanin_sum, (int)is_FF_D, (int)is_FF_Q);
        // }
    }

}

void KKTProjection(int *level_list, 
                    vector<int> level_list_end_cpu,
                    int *primary_outputs,
                    index_type *pin_forward_arc_list_end,
                    index_type *pin_forward_arc_list,
                    index_type *pin_backward_arc_list_end,
                    index_type *pin_backward_arc_list,
                    index_type *timing_arc_from_pin_id,
                    index_type *timing_arc_to_pin_id,
                    index_type *timing_arc_id_map,
                    int *arc_types,
                    float *arcDelay,
                    float *pinAt,
                    float *pinRat,
                    float *POLambda,
                    float *arcLambda,
                    float *fanout_sum,
                    int num_PO,
                    GPULutAllocator *d_allocator
                ){
    // KKTProjectionPO<<<BLOCK_NUMBER(num_PO * 2), BLOCK_SIZE>>>(primary_outputs, pin_backward_arc_list, pin_backward_arc_list_end, timing_arc_from_pin_id, timing_arc_id_map, arc_types, pinAt, pinRat, POLambda, arcLambda, num_PO);

    // cudaDeviceSynchronize();
    for(int i = level_list_end_cpu.size() - 2;i >= 0; i--){
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        // std::cout<<"KKT Projection for level " << i <<" with " << num_pins_level << " Pins" << std::endl;
        index_type level_start_offset = level_list_end_cpu[i];
        KKTProjectionGate<<<BLOCK_NUMBER(num_pins_level * 2), BLOCK_SIZE>>>(level_list, 
                                                                            pin_forward_arc_list_end, 
                                                                            pin_forward_arc_list,
                                                                            pin_backward_arc_list, 
                                                                            pin_backward_arc_list_end, 
                                                                            timing_arc_from_pin_id, 
                                                                            timing_arc_to_pin_id, 
                                                                            timing_arc_id_map, 
                                                                            arc_types,
                                                                            arcDelay, 
                                                                            arcLambda, 
                                                                            fanout_sum,
                                                                            level_start_offset, 
                                                                            num_pins_level,
                                                                            i, 
                                                                            d_allocator
                                                                        );
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
}
__global__ void check_lambda(float *arcLambda, int *arc_types, int *timing_arc_id_map, int* timing_arc_from_pin_id, int* timing_arc_to_pin_id, float* pinAt, float* pinRat, int num_arcs){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int arc_id = idx >> 1;
    if(arc_id >= num_arcs)return ;
    if(arc_types[arc_id] == 0 || timing_arc_id_map[arc_id * 2 + 1] == -1)return ;
    int from = timing_arc_from_pin_id[arc_id];
    int to = timing_arc_to_pin_id[arc_id];
    // if(abs(arcLambda[idx]) <= 1e-8)
    //     printf("arc:%d lambda:%.10f fr_p:%d to_p:%d fr_at:%.10f fr_rat:%.10f  to_at:%.10f to_rat:%.10f\n", arc_id, arcLambda[idx], from, to, pinAt[from], pinRat[from], pinAt[to], pinRat[to]);
}
void evaluate_sizing_cuda(index_type *level_gate_list,
                          vector<int> level_gate_list_end_cpu,
                          index_type *level_list,
                          vector<int> level_list_end_cpu,
                          index_type *primary_outputs,
                          index_type *topo_order2candidate_list_end,
                          index_type *candidate_id2pin_id,
                          index_type *candidate_id2lib_cell_id,
                          index_type *pin_forward_arc_list_end,
                          index_type *pin_forward_arc_list,
                          index_type *pin_backward_arc_list_end,
                          index_type *pin_backward_arc_list,
                          index_type *timing_arc_to_pin_id,
                          index_type *timing_arc_from_pin_id,
                          index_type *arc_types,
                          index_type *pin_id2equivalent_cell_id,
                          int *pin_id2cell_type_id,
                          vector<index_type*> level_equivalent_cell_num_pin,
                          index_type *liberty_cell_type2port_list_end,
                          index_type *liberty_port2timing_list_end,
                          index_type *timing_arc_in_port_id,
                          index_type *pin_id2port_offset_id,
                          float *pinCap,
                          float *pinSlew,
                          float *pinLoad,
                          float *pinRootRes,
                          float *pinRootDelay,
                          float *arcDelay,
                          float *pinAt,
                          float *pinRat,
                          float *POLambda,
                          float *arcLambda,
                          float *level_pinSlew,
                          float *liberty_port_capacitance,
                          float *level_cost,
                          int *level_candidate,
                          int *timing_arc_id_map,
                          GPULutAllocator *d_allocator,
                          float clock_period,
                          int num_pins,
                          int num_arcs,
                          int i,
                          int num_PO) {
    if(i >= level_gate_list_end_cpu.size() - 1)return ;
    if(i == 0){
        float *fanout_sum;
        cudaMalloc(&fanout_sum, num_arcs * 2 * sizeof(float));
        cudaMemset(fanout_sum, 0, num_arcs * 2 * sizeof(float));
        updateArcLambda<<<BLOCK_NUMBER(num_arcs), BLOCK_SIZE>>>(
            timing_arc_from_pin_id,
            timing_arc_to_pin_id,
            arc_types,
            timing_arc_id_map,
            pinAt,
            pinRat,
            arcDelay,
            arcLambda,
            fanout_sum,
            clock_period,
            d_allocator
        );
        cudaDeviceSynchronize();
        updatePOLambda<<<BLOCK_NUMBER(num_PO * 2), BLOCK_SIZE>>>(
            primary_outputs,
            pinAt,
            pinRat,
            POLambda,
            fanout_sum,
            clock_period,
            num_PO
        );
        cudaDeviceSynchronize();
        KKTProjection(
            level_list,
            level_list_end_cpu,
            primary_outputs,
            pin_forward_arc_list_end,
            pin_forward_arc_list,
            pin_backward_arc_list_end,
            pin_backward_arc_list,
            timing_arc_from_pin_id,
            timing_arc_to_pin_id,
            timing_arc_id_map,
            arc_types,
            arcDelay,
            pinAt,
            pinRat,
            POLambda,
            arcLambda,
            fanout_sum,
            num_PO,
            d_allocator
        );
        cudaFree(fanout_sum);
        cudaDeviceSynchronize();
        // check_lambda<<<BLOCK_NUMBER(num_arcs * 2), BLOCK_SIZE>>>(arcLambda, arc_types, timing_arc_id_map, timing_arc_from_pin_id, timing_arc_to_pin_id, pinAt, pinRat, num_arcs);
        // cudaDeviceSynchronize();
    }
    int candidate_start = 0;
    int candidate_end;
    // for (int i = 0; i < level_gate_list_end_cpu.size() - 1; i++) {
        int num_pins_level = level_gate_list_end_cpu[i + 1] - level_gate_list_end_cpu[i];
        // if(num_pins_level == 0)continue;
        index_type level_start_offset = level_gate_list_end_cpu[i];
        cudaMemcpy(&candidate_end, &topo_order2candidate_list_end[level_gate_list_end_cpu[i + 1]], sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&candidate_start, &topo_order2candidate_list_end[level_gate_list_end_cpu[i]], sizeof(int), cudaMemcpyDeviceToHost);
        int num_candidate = candidate_end - candidate_start;
        // std::cout<<"Calc "<< num_candidate << " candidates in level " << i << std::endl;
        evaluateEquivalentCell<<<BLOCK_NUMBER(num_candidate), BLOCK_SIZE>>>(level_gate_list,
                                                                        candidate_id2pin_id,
                                                                        candidate_id2lib_cell_id,
                                                                        pin_backward_arc_list_end,
                                                                        pin_backward_arc_list,
                                                                        pin_forward_arc_list_end,
                                                                        pin_forward_arc_list,
                                                                        timing_arc_from_pin_id,
                                                                        timing_arc_to_pin_id,
                                                                        pin_id2equivalent_cell_id,
                                                                        level_equivalent_cell_num_pin[i],
                                                                        liberty_cell_type2port_list_end,
                                                                        liberty_port2timing_list_end,
                                                                        timing_arc_in_port_id,
                                                                        pin_id2port_offset_id,
                                                                        liberty_port_capacitance,
                                                                        pinRootRes,
                                                                        pinRootDelay,
                                                                        pinCap,
                                                                        pinSlew,
                                                                        pinLoad,
                                                                        arcLambda,
                                                                        level_pinSlew,
                                                                        level_cost,
                                                                        timing_arc_id_map,
                                                                        d_allocator,
                                                                        level_start_offset,
                                                                        num_candidate,
                                                                        candidate_start
                                                                        );
        cudaDeviceSynchronize();
        // cachePinLoad<<<BLOCK_NUMBER(num_pins_level * 2), BLOCK_SIZE>>>(level_gate_list,
        //                                                             pin_backward_arc_list_end,
        //                                                             pin_backward_arc_list,
        //                                                             timing_arc_from_pin_id,
        //                                                             timing_arc_id_map,
        //                                                             pinCap,
        //                                                             pinLoad,
        //                                                             level_start_offset,
        //                                                             num_pins_level,
        //                                                             -1
        //                                                             );

        // cudaDeviceSynchronize();
        evaluateLevelPin<<<BLOCK_NUMBER(num_pins_level), BLOCK_SIZE>>>(level_gate_list,
                                                                    topo_order2candidate_list_end,
                                                                    candidate_id2lib_cell_id,
                                                                    pin_backward_arc_list_end,
                                                                    pin_backward_arc_list,
                                                                    pin_forward_arc_list_end,
                                                                    pin_forward_arc_list,
                                                                    timing_arc_from_pin_id,
                                                                    timing_arc_to_pin_id,
                                                                    pin_id2equivalent_cell_id,
                                                                    pin_id2cell_type_id,
                                                                    level_equivalent_cell_num_pin[i],
                                                                    liberty_cell_type2port_list_end,
                                                                    liberty_port2timing_list_end,
                                                                    timing_arc_in_port_id,
                                                                    pin_id2port_offset_id,
                                                                    liberty_port_capacitance,
                                                                    pinCap,
                                                                    pinSlew,
                                                                    pinLoad,
                                                                    level_pinSlew,
                                                                    level_cost,
                                                                    level_candidate,
                                                                    timing_arc_id_map,
                                                                    d_allocator,
                                                                    level_start_offset,
                                                                    num_pins_level
                                                                    );
        cudaDeviceSynchronize();
        // cachePinLoad<<<BLOCK_NUMBER(num_pins_level * 2), BLOCK_SIZE>>>(level_gate_list,
        //                                                             pin_backward_arc_list_end,
        //                                                             pin_backward_arc_list,
        //                                                             timing_arc_from_pin_id,
        //                                                             timing_arc_id_map,
        //                                                             pinCap,
        //                                                             pinLoad,
        //                                                             level_start_offset,
        //                                                             num_pins_level,
        //                                                             1
        //                                                             );
        // printf("==== level %d ======= %d \n", i, num_pins_level);
        // cudaDeviceSynchronize();
        // int level_input_start_offset = level_list_end_cpu[i-1];
        // int num_input_pins = level_list_end_cpu[i] - level_list_end_cpu[i-1];
        // updatePinImpulse<<<BLOCK_NUMBER(num_input_pins * 2), BLOCK_SIZE>>>(level_list, 
        //                                                                 pin_backward_arc_list_end,
        //                                                                 pin_backward_arc_list,
        //                                                                 timing_arc_from_pin_id, 
        //                                                                 pinCap,
        //                                                                 pinLoad,
        //                                                                 pinImpulse,
        //                                                                 pinRootDelay,
        //                                                                 pinRootRes,
        //                                                                 level_input_start_offset,
        //                                                                 num_input_pins
        //                                                                 );
        candidate_start = candidate_end;
    // }
    cudaDeviceSynchronize();
}


}  // namespace gt