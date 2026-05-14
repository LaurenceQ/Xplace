static constexpr bool DMP_DIRECT_CLOCK_DEBUG_PRINT = false;
static constexpr bool DMP_DRIVING_CELL_DEBUG_PRINT = false;

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

__device__ __forceinline__ void dmpRootProfileAdd(int counter,
                                                  unsigned long long value) {
#if DMP_ROOT_SOLVE_PROFILE
    atomicAdd(&dmp_root_profile_counts[counter], value);
#else
    (void)counter;
    (void)value;
#endif
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

enum DmpParam { t0,
                dt,
                ceff };
enum DmpFunc { y20,
               y50,
               ipi };

__device__ __forceinline__ double dmpFinitePositiveOr(double value, double fallback) {
    return (isfinite(value) && value > 0.0) ? value : fallback;
}

__device__ __forceinline__ bool dmpStringEquals(const char* lhs, const char* rhs) {
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

__device__ __forceinline__ double dmpThresholdArrayValue(const float* values,
                                                         int attr,
                                                         double fallback) {
    return values ? dmpFinitePositiveOr(values[attr], fallback) : fallback;
}

__device__ __forceinline__ double dmpPinThresholdArrayValue(const float* values,
                                                            int pin_id,
                                                            int rf,
                                                            double fallback) {
    return (values && pin_id >= 0) ? dmpFinitePositiveOr(values[pin_id * 2 + rf], fallback)
                                   : fallback;
}

__device__ __forceinline__ double dmpTimingThresholdArrayValue(const float* values,
                                                               int timing_id,
                                                               int rf,
                                                               double fallback) {
    return (values && timing_id >= 0) ? dmpFinitePositiveOr(values[timing_id * 2 + rf], fallback)
                                      : fallback;
}

__device__ __forceinline__ void dmpLoadSlotThresholds(const dmp_model* dmp_db,
                                                      int slot,
                                                      double& vth,
                                                      double& vl,
                                                      double& vh,
                                                      double& slew_derate) {
    int attr = slot & (NUM_ATTR - 1);
    int timing_id = -1;
    int output_rf = attr & 1;
    vth = dmpTimingThresholdArrayValue(
        dmp_db->dmp_timing_output_thresholds,
        timing_id,
        output_rf,
        dmpThresholdArrayValue(dmp_db->dmp_output_thresholds, attr, dmp_db->vth_));
    vl = dmpTimingThresholdArrayValue(
        dmp_db->dmp_timing_slew_lower_thresholds,
        timing_id,
        output_rf,
        dmpThresholdArrayValue(dmp_db->dmp_slew_lower_thresholds, attr, dmp_db->vl_));
    vh = dmpTimingThresholdArrayValue(
        dmp_db->dmp_timing_slew_upper_thresholds,
        timing_id,
        output_rf,
        dmpThresholdArrayValue(dmp_db->dmp_slew_upper_thresholds, attr, dmp_db->vh_));
    slew_derate = dmpTimingThresholdArrayValue(
        dmp_db->dmp_timing_slew_derates,
        timing_id,
        output_rf,
        dmpThresholdArrayValue(dmp_db->dmp_slew_derates, attr, dmp_db->slew_derate_));
    if (vh <= vl) {
        vl = dmp_db->vl_;
        vh = dmp_db->vh_;
    }
}

__device__ __forceinline__ float dmpDecodeWinnerFloat(unsigned int cmp_key,
                                                      bool pick_max) {
    const unsigned int ordered = pick_max ? cmp_key : ~cmp_key;
    const unsigned int raw = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u)
                                                     : ~ordered;
    return __uint_as_float(raw);
}

__device__ __forceinline__ unsigned long long dmpPackWinner(float value,
                                                            unsigned int payload,
                                                            bool pick_max) {
    unsigned int ordered = __float_as_uint(value);
    ordered = (ordered & 0x80000000u) ? ~ordered : (ordered ^ 0x80000000u);
    const unsigned int cmp_key = pick_max ? ordered : ~ordered;
    return (static_cast<unsigned long long>(cmp_key) << 32)
           | static_cast<unsigned long long>(payload);
}

enum DmpGateNetPairDebugCounter {
    DMP_GNP_TOTAL_CANDIDATES = 0,
    DMP_GNP_INVALID_TRANSITION_SKIPS = 1,
    DMP_GNP_INVALID_SCRATCH_SKIPS = 2,
    DMP_GNP_FINITE_CANDIDATES = 3
};

__device__ __forceinline__ void dmpGateNetPairCount(unsigned long long* counts,
                                                    int counter) {
    if (counts != nullptr) {
        atomicAdd(&counts[counter], 1ULL);
    }
}

__device__ __forceinline__ void dmpThresholdAdjustCuda(const dmp_model* dmp_db,
                                                       int load_pin_id,
                                                       int load_attr,
                                                       double driver_vth,
                                                       double driver_vl,
                                                       double driver_vh,
                                                       double driver_derate,
                                                       double& wire_delay,
                                                       double& load_slew) {
    if (!isfinite(wire_delay) || !isfinite(load_slew)) {
        return;
    }
    const int load_rf = load_attr & 1;
    const double load_vth = dmpPinThresholdArrayValue(dmp_db->dmp_pin_input_thresholds, load_pin_id, load_rf,
                                                      dmpThresholdArrayValue(dmp_db->dmp_input_thresholds, load_attr, dmp_db->vth_));
    const double load_vl = dmpPinThresholdArrayValue(dmp_db->dmp_pin_slew_lower_thresholds, load_pin_id, load_rf,
                                                     dmpThresholdArrayValue(dmp_db->dmp_slew_lower_thresholds, load_attr, dmp_db->vl_));
    const double load_vh = dmpPinThresholdArrayValue(dmp_db->dmp_pin_slew_upper_thresholds, load_pin_id, load_rf,
                                                     dmpThresholdArrayValue(dmp_db->dmp_slew_upper_thresholds, load_attr, dmp_db->vh_));
    const double load_derate = dmpPinThresholdArrayValue(dmp_db->dmp_pin_slew_derates, load_pin_id, load_rf,
                                                         dmpThresholdArrayValue(dmp_db->dmp_slew_derates, load_attr, dmp_db->slew_derate_));
    const double driver_delta = driver_vh - driver_vl;
    const double load_delta = load_vh - load_vl;
    if (!isfinite(driver_delta) || !isfinite(load_delta) ||
        !isfinite(driver_derate) || !isfinite(load_derate) ||
        driver_delta <= 0.0 || load_delta <= 0.0 ||
        driver_derate <= 0.0 || load_derate <= 0.0) {
        return;
    }
    const double delay_delta = load_slew * ((load_vth - driver_vth) / driver_delta);
    wire_delay += ((load_attr & 1) == 0) ? delay_delta : -delay_delta;
    load_slew *= ((load_delta / load_derate) / (driver_delta / driver_derate));
}

__device__ __forceinline__ void dmpInputPortDelaySlewCuda(const dmp_model* dmp_db,
                                                          int load_pin_id,
                                                          int load_attr,
                                                          double source_slew,
                                                          double elmore,
                                                          double& wire_delay,
                                                          double& load_slew) {
    const int load_rf = load_attr & 1;
    const double load_vth = dmpPinThresholdArrayValue(dmp_db->dmp_pin_input_thresholds, load_pin_id, load_rf,
                                                      dmpThresholdArrayValue(dmp_db->dmp_input_thresholds, load_attr, dmp_db->vth_));
    const double load_vl = dmpPinThresholdArrayValue(dmp_db->dmp_pin_slew_lower_thresholds, load_pin_id, load_rf,
                                                     dmpThresholdArrayValue(dmp_db->dmp_slew_lower_thresholds, load_attr, dmp_db->vl_));
    const double load_vh = dmpPinThresholdArrayValue(dmp_db->dmp_pin_slew_upper_thresholds, load_pin_id, load_rf,
                                                     dmpThresholdArrayValue(dmp_db->dmp_slew_upper_thresholds, load_attr, dmp_db->vh_));
    const double load_derate = dmpPinThresholdArrayValue(dmp_db->dmp_pin_slew_derates, load_pin_id, load_rf,
                                                         dmpThresholdArrayValue(dmp_db->dmp_slew_derates, load_attr, dmp_db->slew_derate_));
    if (!isfinite(source_slew) || !isfinite(elmore) || elmore < 0.0 ||
        !isfinite(load_vth) || !isfinite(load_vl) || !isfinite(load_vh) ||
        !isfinite(load_derate) || load_vth <= 0.0 || load_vth >= 1.0 ||
        load_vl <= 0.0 || load_vh >= 1.0 || load_vh <= load_vl ||
        load_derate <= 0.0) {
        wire_delay = nanf("");
        load_slew = nanf("");
        return;
    }

    wire_delay = -elmore * log(1.0 - load_vth);
    load_slew = source_slew + elmore * log((1.0 - load_vl) / (1.0 - load_vh)) / load_derate;

    const double driver_vth = dmpThresholdArrayValue(dmp_db->dmp_output_thresholds, load_attr, dmp_db->vth_);
    const double driver_vl = dmpThresholdArrayValue(dmp_db->dmp_slew_lower_thresholds, load_attr, dmp_db->vl_);
    const double driver_vh = dmpThresholdArrayValue(dmp_db->dmp_slew_upper_thresholds, load_attr, dmp_db->vh_);
    const double driver_derate = dmpThresholdArrayValue(dmp_db->dmp_slew_derates, load_attr, dmp_db->slew_derate_);
    dmpThresholdAdjustCuda(dmp_db, load_pin_id, load_attr, driver_vth, driver_vl, driver_vh, driver_derate, wire_delay, load_slew);
}

__device__ void dmp_model::propagateLoadSlewDelay(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx & 0b11;
    int arc_id = arc_ids[idx];
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    double elmore = elmore_delay[to_pin_id * NUM_ATTR + i];
    pin_ids[idx] = from_pin_id * NUM_ATTR + i;
    int pin_idx = pin_ids[idx];
    float si = pinSlew[pin_idx];
    if (isnan(si)) return;
    // float imp = pinImpulse[to_pin_id * NUM_ATTR + i];
    // float so = si < 0.0 ? -sqrt(si * si + imp * imp) : sqrt(si * si + imp * imp); // todo: how to calc slew?
    int el_rf_rf = (i << 1) + (i & 1);  // same rise/fall for two pins in net connections
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(this, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    double fallback_delay = elmore;
    double fallback_slew = si;
    if (pin_is_primary_input != nullptr && pin_is_primary_input[from_pin_id]) {
        dmpInputPortDelaySlewCuda(this, to_pin_id, i, si, elmore, fallback_delay, fallback_slew);
    }
    else {
        dmpThresholdAdjustCuda(this, to_pin_id, i, driver_vth, driver_vl, driver_vh, driver_derate, fallback_delay, fallback_slew);
    }
    const double driving_cell_extra_delay = driving_cell_extra_delay_[pin_idx];
    if (isfinite(driving_cell_extra_delay)) {
        fallback_delay += driving_cell_extra_delay;
    }
    double final_delay = fallback_delay;
    double final_slew = fallback_slew;
    bool used_dmp_load = false;

    bool driver_valid = isfinite(rd_[pin_idx]) && rd_[pin_idx] > 0.0 &&
                        isfinite(t0[pin_idx]) && isfinite(dt[pin_idx]) && dt[pin_idx] > 0.0 &&
                        isfinite(vo_delay_[pin_idx]);
    if (driver_valid && isfinite(elmore) && elmore != 0.0 && elmore >= si * 1e-3) {
        bool error_flag = false;
        double t_lower = t0[pin_idx];
        double t_upper = voCrossingUpperBound(pin_idx) + elmore * 2.0;
        double load_delay = findVlCrossing(driver_vth, t_lower, t_upper); // time point voltage reach middle
        double tl = findVlCrossing(driver_vl, t_lower, load_delay); // time point voltage reach low
        double th = findVlCrossing(driver_vh, load_delay, t_upper); // time point voltage reach high
        double delay1 = load_delay - vo_delay_[pin_idx];
        double slew1 = (th - tl) / driver_derate;

        if(!isfinite(load_delay) || !isfinite(tl) || !isfinite(th) ||
           !isfinite(slew1) || !isfinite(delay1)){
            error_flag = true;
            if(debug_on)printf("Error: load slew or delay is nan net_id:%d rd:%.4f, r_pi:%.4f cap1:%.4f cap2:%.4f t0:%.4f, dt:%.4f ceff:%.4f\n", pin_idx, rd_[pin_idx], r_pi[pin_idx], C1[pin_idx], C2[pin_idx], t0[pin_idx], dt[pin_idx], ceff[pin_idx]);
        }
        else{
            if(delay1 < 0.0){
                if(-delay1 > vth_time_tol * vo_delay_[pin_idx]){
                    error_flag = true;
                    if(debug_on)printf("Error: load delay less than 0 net_id:%d rd:%.4f, r_pi:%.4f cap1:%.4f cap2:%.4f t0:%.4f, dt:%.4f ceff:%.4f\n", pin_idx, rd_[pin_idx], r_pi[pin_idx], C1[pin_idx], C2[pin_idx], t0[pin_idx], dt[pin_idx], ceff[pin_idx]);
                }
                else delay1 = elmore;
            }
            if(slew1 < si){
                if((si - slew1) > vth_time_tol * si){
                    error_flag = true;
                    if(debug_on)printf("Error: load slew less than driver slew net_id:%d rd:%.4f, r_pi:%.4f cap1:%.4f cap2:%.4f t0:%.4f, dt:%.4f ceff:%.4f\n", pin_idx, rd_[pin_idx], r_pi[pin_idx], C1[pin_idx], C2[pin_idx], t0[pin_idx], dt[pin_idx], ceff[pin_idx]);
                }
                else slew1 = pinSlew[from_pin_id * NUM_ATTR + i];
            }
        }
        if(!error_flag) {
            dmpThresholdAdjustCuda(this, to_pin_id, i, driver_vth, driver_vl, driver_vh, driver_derate, delay1, slew1);
            if (isfinite(driving_cell_extra_delay)) {
                delay1 += driving_cell_extra_delay;
            }
            final_delay = delay1;
            final_slew = slew1;
            used_dmp_load = true;
        }
    }

    pinSlew[to_pin_id * NUM_ATTR + i] = final_slew;
    arcDelay[arc_id * 2 * NUM_ATTR + el_rf_rf] = final_delay;

    if (DMP_DIRECT_CLOCK_DEBUG_PRINT &&
        pin_names != nullptr &&
        dmpStringEquals(pin_names[from_pin_id], "clk") &&
        dmpStringEquals(pin_names[to_pin_id], "clkbuf_0_clk:A")) {
        const float src_at = pinAt[pin_idx];
        const double cand_at = isfinite(src_at) && isfinite(final_delay)
                                   ? static_cast<double>(src_at) + final_delay
                                   : nan("");
        printf("[DMP DIRECT CLOCK] attr=%d source_slew=%.9f load_slew=%.9f elmore=%.9f extra_delay=%.9f vo_delay=%.9f wire_delay=%.9f at=%.9f alg=%d dmp_load=%d\n",
               i,
               static_cast<double>(si),
               final_slew,
               elmore,
               driving_cell_extra_delay,
               vo_delay_[pin_idx],
               final_delay,
               cand_at,
               dmp_alg_kind[pin_idx],
               used_dmp_load ? 1 : 0);
    }
}
