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

__device__ __forceinline__ bool dmpSolve2x2(double a00,
                                            double a01,
                                            double a10,
                                            double a11,
                                            double b0,
                                            double b1,
                                            double& x0,
                                            double& x1) {
    const double det = a00 * a11 - a01 * a10;
    const double scale = fabs(a00 * a11) + fabs(a01 * a10) + 1e-300;
    if (!isfinite(det) || fabs(det) <= 1e-14 * scale) {
        return false;
    }
    x0 = (b0 * a11 - a01 * b1) / det;
    x1 = (a00 * b1 - b0 * a10) / det;
    return isfinite(x0) && isfinite(x1);
}

__device__ __forceinline__ bool dmpSolve3x3(double a00,
                                            double a01,
                                            double a02,
                                            double a10,
                                            double a11,
                                            double a12,
                                            double a20,
                                            double a21,
                                            double a22,
                                            double b0,
                                            double b1,
                                            double b2,
                                            double& x0,
                                            double& x1,
                                            double& x2) {
    const double c00 = a11 * a22 - a12 * a21;
    const double c01 = a10 * a22 - a12 * a20;
    const double c02 = a10 * a21 - a11 * a20;
    const double det = a00 * c00 - a01 * c01 + a02 * c02;
    const double scale = fabs(a00 * c00) + fabs(a01 * c01) + fabs(a02 * c02) + 1e-300;
    if (!isfinite(det) || fabs(det) <= 1e-14 * scale) {
        return false;
    }
    const double det0 = b0 * c00 - a01 * (b1 * a22 - a12 * b2) +
                        a02 * (b1 * a21 - a11 * b2);
    const double det1 = a00 * (b1 * a22 - a12 * b2) -
                        b0 * c01 +
                        a02 * (a10 * b2 - b1 * a20);
    const double det2 = a00 * (a11 * b2 - b1 * a21) -
                        a01 * (a10 * b2 - b1 * a20) +
                        b0 * c02;
    x0 = det0 / det;
    x1 = det1 / det;
    x2 = det2 / det;
    return isfinite(x0) && isfinite(x1) && isfinite(x2);
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

__device__ __forceinline__ double dmpLibraryThresholdArrayValue(const float* values,
                                                                int library_id,
                                                                int rf,
                                                                double fallback) {
    return (values && library_id >= 0) ? dmpFinitePositiveOr(values[library_id * MAX_TRAN + rf], fallback)
                                       : fallback;
}

__device__ __forceinline__ int dmpTimingLibraryId(const dmp_model* dmp_db,
                                                  int timing_id) {
    return (dmp_db->dmp_timing_library_ids && timing_id >= 0)
               ? dmp_db->dmp_timing_library_ids[timing_id]
               : -1;
}

__device__ __forceinline__ int dmpPinLibraryId(const dmp_model* dmp_db,
                                               int pin_id,
                                               int attr) {
    const int el = attr >> 1;
    return (dmp_db->dmp_pin_library_ids && pin_id >= 0)
               ? dmp_db->dmp_pin_library_ids[pin_id * MAX_SPLIT + el]
               : -1;
}

__device__ __forceinline__ void dmpSanitizeThresholds(const dmp_model* dmp_db,
                                                      double& vl,
                                                      double& vh) {
    if (vh <= vl) {
        vl = dmp_db->vl_;
        vh = dmp_db->vh_;
    }
}

__device__ __forceinline__ void dmpLoadPinThresholds(const dmp_model* dmp_db,
                                                     int pin_id,
                                                     int attr,
                                                     double& vth,
                                                     double& vl,
                                                     double& vh,
                                                     double& slew_derate) {
    const int rf = attr & 1;
    const int library_id = dmpPinLibraryId(dmp_db, pin_id, attr);
    vth = dmpLibraryThresholdArrayValue(
        dmp_db->dmp_library_input_thresholds,
        library_id,
        rf,
        dmpThresholdArrayValue(dmp_db->dmp_input_thresholds, attr, dmp_db->vth_));
    vl = dmpLibraryThresholdArrayValue(
        dmp_db->dmp_library_slew_lower_thresholds,
        library_id,
        rf,
        dmpThresholdArrayValue(dmp_db->dmp_slew_lower_thresholds, attr, dmp_db->vl_));
    vh = dmpLibraryThresholdArrayValue(
        dmp_db->dmp_library_slew_upper_thresholds,
        library_id,
        rf,
        dmpThresholdArrayValue(dmp_db->dmp_slew_upper_thresholds, attr, dmp_db->vh_));
    slew_derate = dmpLibraryThresholdArrayValue(
        dmp_db->dmp_library_slew_derates,
        library_id,
        rf,
        dmpThresholdArrayValue(dmp_db->dmp_slew_derates, attr, dmp_db->slew_derate_));
    dmpSanitizeThresholds(dmp_db, vl, vh);
}

__device__ __forceinline__ void dmpDriverLibraryThresholds(const dmp_model* dmp_db,
                                                          int library_id,
                                                          int attr,
                                                          double& vth,
                                                          double& vl,
                                                          double& vh,
                                                          double& slew_derate) {
    const int rf = attr & 1;
    vth = dmpLibraryThresholdArrayValue(
        dmp_db->dmp_library_output_thresholds,
        library_id,
        rf,
        dmpThresholdArrayValue(dmp_db->dmp_output_thresholds, attr, dmp_db->vth_));
    vl = dmpLibraryThresholdArrayValue(
        dmp_db->dmp_library_slew_lower_thresholds,
        library_id,
        rf,
        dmpThresholdArrayValue(dmp_db->dmp_slew_lower_thresholds, attr, dmp_db->vl_));
    vh = dmpLibraryThresholdArrayValue(
        dmp_db->dmp_library_slew_upper_thresholds,
        library_id,
        rf,
        dmpThresholdArrayValue(dmp_db->dmp_slew_upper_thresholds, attr, dmp_db->vh_));
    slew_derate = dmpLibraryThresholdArrayValue(
        dmp_db->dmp_library_slew_derates,
        library_id,
        rf,
        dmpThresholdArrayValue(dmp_db->dmp_slew_derates, attr, dmp_db->slew_derate_));
    dmpSanitizeThresholds(dmp_db, vl, vh);
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

enum DmpDrivingCellCounter {
    DMP_DRIVING_CELL_APPLIED = 0,
    DMP_DRIVING_CELL_SKIPPED = 1,
    DMP_DRIVING_CELL_CAP = 2,
    DMP_DRIVING_CELL_ZERO_C2 = 3,
    DMP_DRIVING_CELL_PI = 4,
    DMP_DRIVING_CELL_DMP_VALID = 5,
    DMP_DRIVING_CELL_FALLBACK = 6,
    DMP_DRIVING_CELL_COUNTER_COUNT = 7
};

__device__ bool dmp_model::updateLoadWinner(int net_arc_id,
                                            int load_attr,
                                            float wire_delay,
                                            float load_slew) {
    if (!isfinite(wire_delay) || !isfinite(load_slew)) {
        return false;
    }
    const int to_pin_id = timing_arc_to_pin_id[net_arc_id];
    const int to_slot = to_pin_id * NUM_ATTR + load_attr;
    const bool pick_max = (load_attr >> 1) != 0;

    const unsigned int slew_payload = static_cast<unsigned int>(net_arc_id);
    const unsigned long long packed_slew = dmpPackWinner(load_slew, slew_payload, pick_max);
    const unsigned long long old_slew = atomicMax(&pin_slew_winner[to_slot], packed_slew);

    const int delay_slot = arcDelayWinnerSlot(net_arc_id, load_attr);
    const unsigned int delay_payload = static_cast<unsigned int>(to_slot);
    const unsigned long long packed_delay = dmpPackWinner(wire_delay, delay_payload, pick_max);
    const unsigned long long old_delay = atomicMax(&arc_delay_winner[delay_slot], packed_delay);
    return packed_slew > old_slew || packed_delay > old_delay;
}

__device__ bool dmp_model::updateAtWinner(int to_slot,
                                          float at,
                                          bool pick_max,
                                          int from_pin_id,
                                          int arc_id,
                                          int from_attr) {
    if (!isfinite(at)) {
        return false;
    }
    (void)from_pin_id;
    const unsigned int payload = (static_cast<unsigned int>(arc_id) << 2)
                                 | static_cast<unsigned int>(from_attr & 0x3);
    const unsigned long long packed = dmpPackWinner(at, payload, pick_max);
    const unsigned long long old = atomicMax(&pin_at_winner[to_slot], packed);
    return packed > old;
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
                                                       int driver_library_id,
                                                       double& wire_delay,
                                                       double& load_slew) {
    if (!isfinite(wire_delay) || !isfinite(load_slew)) {
        return;
    }
    const int load_library_id = dmpPinLibraryId(dmp_db, load_pin_id, load_attr);
    if (driver_library_id >= 0 && driver_library_id == load_library_id) {
        return;
    }
    double load_vth, load_vl, load_vh, load_derate;
    dmpLoadPinThresholds(dmp_db,
                         load_pin_id,
                         load_attr,
                         load_vth,
                         load_vl,
                         load_vh,
                         load_derate);
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
    double load_vth, load_vl, load_vh, load_derate;
    dmpLoadPinThresholds(dmp_db,
                         load_pin_id,
                         load_attr,
                         load_vth,
                         load_vl,
                         load_vh,
                         load_derate);
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
    dmpThresholdAdjustCuda(dmp_db,
                           load_pin_id,
                           load_attr,
                           driver_vth,
                           driver_vl,
                           driver_vh,
                           driver_derate,
                           -1,
                           wire_delay,
                           load_slew);
}
