
struct DmpGateLutMeta {
    unsigned int dims;
    unsigned int flags;
    int x_offset;
    int y_offset;
    int table_offset;
};

static constexpr unsigned int DMP_LUT_META_SCALAR = 1u << 0;
static constexpr unsigned int DMP_LUT_META_VAR0_IS_SLEW = 1u << 1;
static constexpr unsigned int DMP_LUT_META_VALID = 1u << 31;

__device__ __forceinline__ int dmpLutMetaNumX(const DmpGateLutMeta& meta) {
    return static_cast<int>(meta.dims & 0xffffu);
}

__device__ __forceinline__ int dmpLutMetaNumY(const DmpGateLutMeta& meta) {
    return static_cast<int>((meta.dims >> 16) & 0xffffu);
}

__device__ __forceinline__ bool dmpLutMetaValid(const DmpGateLutMeta& meta) {
    return (meta.flags & DMP_LUT_META_VALID) != 0u;
}

__device__ __forceinline__ int dmpLowerBound7(const float* arr, float val) {
    if (val <= arr[3]) {
        if (val <= arr[1]) {
            return (val <= arr[0]) ? 0 : 1;
        }
        return (val <= arr[2]) ? 2 : 3;
    }
    if (val <= arr[5]) {
        return (val <= arr[4]) ? 4 : 5;
    }
    return 6;
}

__device__ __forceinline__ int dmpGateLowerBound(const float* arr, int size, float val) {
    if (size == 7) {
        return dmpLowerBound7(arr, val);
    }
    int l = 0;
    int r = size - 1;
    while (l < r) {
        const int m = (l + r) >> 1;
        if (arr[m] < val) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}

__device__ __forceinline__ DmpGateLutMeta dmpMakeGateLutMeta(GPULutAllocator* allocator,
                                                             int lut_id) {
    DmpGateLutMeta meta{};
    meta.dims = 0;
    meta.flags = 0;
    meta.x_offset = 0;
    meta.y_offset = 0;
    meta.table_offset = 0;
    if (allocator == nullptr || lut_id < 0 || !allocator->d_allocated[lut_id]) {
        return meta;
    }
    const int num_x = allocator->d_num_x[lut_id];
    const int num_y = allocator->d_num_y[lut_id];
    const int num_table = allocator->d_num_table[lut_id];
    const int var0 = allocator->d_lut_template_var[lut_id * 2];
    if (num_x < 1 || num_y < 1 || num_x > 65535 || num_y > 65535 ||
        num_table < 1 || (var0 != 0 && var0 != 1)) {
        return meta;
    }
    meta.dims = static_cast<unsigned int>(num_x) |
                (static_cast<unsigned int>(num_y) << 16);
    meta.flags = DMP_LUT_META_VALID |
                 (num_table == 1 ? DMP_LUT_META_SCALAR : 0u) |
                 (var0 == 1 ? DMP_LUT_META_VAR0_IS_SLEW : 0u);
    meta.x_offset = static_cast<int>(allocator->d_x_offset[lut_id]);
    meta.y_offset = static_cast<int>(allocator->d_y_offset[lut_id]);
    meta.table_offset = static_cast<int>(allocator->d_table_offset[lut_id]);
    return meta;
}

__device__ __forceinline__ float dmpGateLutWithMeta(GPULutAllocator* allocator,
                                                    const DmpGateLutMeta& meta,
                                                    float input_slew,
                                                    float load) {
    if (allocator == nullptr || !dmpLutMetaValid(meta)) {
        return nanf("");
    }
    float x = 0.0f;
    float y = 0.0f;
    if ((meta.flags & DMP_LUT_META_VAR0_IS_SLEW) != 0u) {
        x = input_slew;
        y = load;
    } else {
        x = load;
        y = input_slew;
    }
    if ((meta.flags & DMP_LUT_META_SCALAR) != 0u) {
        return allocator->d_table_array[meta.table_offset];
    }

    const int num_x = dmpLutMetaNumX(meta);
    const int num_y = dmpLutMetaNumY(meta);
    const float* x_array = allocator->d_x_array + meta.x_offset;
    const float* y_array = allocator->d_y_array + meta.y_offset;
    const float* table_array = allocator->d_table_array + meta.table_offset;
    int x_idx1 = dmpGateLowerBound(x_array, num_x, x);
    int y_idx1 = dmpGateLowerBound(y_array, num_y, y);
    x_idx1 = max(1, min(num_x - 1, x_idx1));
    y_idx1 = max(1, min(num_y - 1, y_idx1));
    const int x_idx0 = x_idx1 - 1;
    const int y_idx0 = y_idx1 - 1;
    if (num_x == 1) {
        x_idx1 = 0;
    }
    if (num_y == 1) {
        y_idx1 = 0;
    }

    const int table_stride = num_y;
    const float x0 = x_array[x_idx0];
    const float x1 = x_array[x_idx1];
    const float y0 = y_array[y_idx0];
    const float y1 = y_array[y_idx1];
    const float v00 = table_array[x_idx0 * table_stride + y_idx0];
    const float v10 = table_array[x_idx1 * table_stride + y_idx0];
    const float v01 = table_array[x_idx0 * table_stride + y_idx1];
    const float v11 = table_array[x_idx1 * table_stride + y_idx1];
    const float numeric0 = interpolate<float>(x0, x1, v00, v10, x);
    const float numeric1 = interpolate<float>(x0, x1, v01, v11, x);
    return interpolate<float>(y0, y1, numeric0, numeric1, y);
}

__device__ __forceinline__ float dmpQueryGateLutNoTransitionCheck(GPULutAllocator* allocator,
                                                                  int timing_id,
                                                                  int output_rf,
                                                                  float input_slew,
                                                                  float load,
                                                                  int type) {
    const int lut_id = allocator->num_luts_in_timing * timing_id + output_rf + type * 2;
    const DmpGateLutMeta meta = dmpMakeGateLutMeta(allocator, lut_id);
    return dmpGateLutWithMeta(allocator, meta, input_slew, load);
}

__device__ __forceinline__ void dmpGateCapDelaySlewCached(dmp_model* dmp_db,
                                                          int timing_id,
                                                          int input_rf,
                                                          int output_rf,
                                                          float input_slew,
                                                          double load_cap,
                                                          double& delay,
                                                          double& slew) {
    delay = nanf("");
    slew = nanf("");
    if (dmp_db->d_allocator == nullptr ||
        timing_id < 0 || input_rf < 0 || output_rf < 0 ||
        !isfinite(input_slew) || !isfinite(load_cap) || load_cap < 0.0) {
        return;
    }
    if (!dmp_db->d_allocator->is_transition_defined(timing_id, input_rf, output_rf)) {
        return;
    }
    const float load = static_cast<float>(load_cap);
    delay = dmpQueryGateLutNoTransitionCheck(dmp_db->d_allocator,
                                             timing_id,
                                             output_rf,
                                             input_slew,
                                             load,
                                             0);
    slew = dmpQueryGateLutNoTransitionCheck(dmp_db->d_allocator,
                                            timing_id,
                                            output_rf,
                                            input_slew,
                                            load,
                                            1);
}

struct DmpGateLaneContext {
    GPULutAllocator* allocator;
    DmpGateLutMeta delay_lut;
    DmpGateLutMeta slew_lut;
    float input_slew;
    double driver_vth;
    double driver_vl;
    double driver_vh;
    double driver_derate;
    bool valid;
};

__device__ __forceinline__ DmpGateLaneContext dmpMakeGateLaneContext(dmp_model* dmp_db,
                                                                     int pin_idx,
                                                                     int timing_id,
                                                                     int input_rf,
                                                                     int output_rf,
                                                                     float input_slew) {
    DmpGateLaneContext ctx{};
    ctx.allocator = dmp_db->d_allocator;
    ctx.delay_lut = {};
    ctx.slew_lut = {};
    ctx.input_slew = input_slew;
    ctx.valid = false;
    dmpLoadSlotThresholds(dmp_db, pin_idx, ctx.driver_vth, ctx.driver_vl, ctx.driver_vh, ctx.driver_derate);

    if (ctx.allocator == nullptr ||
        timing_id < 0 || input_rf < 0 || output_rf < 0 ||
        !isfinite(input_slew) ||
        !ctx.allocator->is_transition_defined(timing_id, input_rf, output_rf)) {
        return ctx;
    }
    const int base_lut_id = ctx.allocator->num_luts_in_timing * timing_id + output_rf;
    const int delay_lut_id = base_lut_id;
    const int slew_lut_id = base_lut_id + 2;
    ctx.delay_lut = dmpMakeGateLutMeta(ctx.allocator, delay_lut_id);
    ctx.slew_lut = dmpMakeGateLutMeta(ctx.allocator, slew_lut_id);
    const double driver_delta = ctx.driver_vh - ctx.driver_vl;
    ctx.valid = dmpLutMetaValid(ctx.delay_lut) &&
                dmpLutMetaValid(ctx.slew_lut) &&
                isfinite(ctx.driver_vth) &&
                isfinite(ctx.driver_vl) &&
                isfinite(ctx.driver_vh) &&
                isfinite(ctx.driver_derate) &&
                isfinite(driver_delta) &&
                driver_delta > 0.0 &&
                ctx.driver_derate > 0.0;
    return ctx;
}

__device__ __forceinline__ void dmpGateCapDelaySlewWithCtx(const DmpGateLaneContext& ctx,
                                                           double load_cap,
                                                           double& delay,
                                                           double& slew) {
    delay = nanf("");
    slew = nanf("");
    if (!ctx.valid || !isfinite(load_cap) || load_cap < 0.0) {
        return;
    }
    const float load = static_cast<float>(load_cap);
    const float delay_f = dmpGateLutWithMeta(ctx.allocator, ctx.delay_lut, ctx.input_slew, load);
    const float slew_f = dmpGateLutWithMeta(ctx.allocator, ctx.slew_lut, ctx.input_slew, load);
    delay = static_cast<double>(delay_f);
    slew = static_cast<double>(slew_f);
}

__device__ __forceinline__ void dmpGateCapDelaySlewWithCtxFloat(const DmpGateLaneContext& ctx,
                                                                float load,
                                                                float& delay,
                                                                float& slew) {
    delay = nanf("");
    slew = nanf("");
    if (!ctx.valid || !isfinite(load) || load < 0.0f) {
        return;
    }
    delay = dmpGateLutWithMeta(ctx.allocator, ctx.delay_lut, ctx.input_slew, load);
    slew = dmpGateLutWithMeta(ctx.allocator, ctx.slew_lut, ctx.input_slew, load);
}

__device__ __forceinline__ void dmpGateDelaysWithCtx(const DmpGateLaneContext& ctx,
                                                     double ceff,
                                                     double& t_vth,
                                                     double& t_vl,
                                                     double& slew) {
    t_vth = nanf("");
    t_vl = nanf("");
    slew = nanf("");
    double table_slew = nanf("");
    dmpGateCapDelaySlewWithCtx(ctx, ceff, t_vth, table_slew);
    if (!isfinite(t_vth) || !isfinite(table_slew)) {
        return;
    }
    slew = table_slew * ctx.driver_derate;
    t_vl = t_vth - slew * (ctx.driver_vth - ctx.driver_vl) / (ctx.driver_vh - ctx.driver_vl);
}

__device__ __forceinline__ void dmpGateDelaysWithCtxFloat(const DmpGateLaneContext& ctx,
                                                          float ceff,
                                                          float& t_vth,
                                                          float& t_vl,
                                                          float& slew) {
    t_vth = nanf("");
    t_vl = nanf("");
    slew = nanf("");
    float table_slew = nanf("");
    dmpGateCapDelaySlewWithCtxFloat(ctx, ceff, t_vth, table_slew);
    if (!isfinite(t_vth) || !isfinite(table_slew)) {
        return;
    }
    const float driver_vth = static_cast<float>(ctx.driver_vth);
    const float driver_vl = static_cast<float>(ctx.driver_vl);
    const float driver_vh = static_cast<float>(ctx.driver_vh);
    const float driver_derate = static_cast<float>(ctx.driver_derate);
    slew = table_slew * driver_derate;
    t_vl = t_vth - slew * (driver_vth - driver_vl) / (driver_vh - driver_vl);
}

__device__ __forceinline__ void dmpGateModelRdWithCtx(dmp_model* dmp_db,
                                                      int pin_idx,
                                                      const DmpGateLaneContext& ctx,
                                                      double cap1,
                                                      double d1) {
    const float cap1_f = static_cast<float>(cap1);
    const float cap_delta_f = static_cast<float>(1e-15 / dmp_db->cap_unit);
    const float cap2_f = cap1_f + cap_delta_f;
    const float d1_f = static_cast<float>(d1);
    float d2_f = nanf("");
    float s2_f = nanf("");
    if (!isfinite(cap1_f) || !isfinite(cap2_f) || !isfinite(d1_f) ||
        !isfinite(cap_delta_f) || cap_delta_f <= 0.0f || cap2_f == cap1_f) {
        dmp_db->rd_[pin_idx] = nanf("");
        return;
    }
    dmpGateCapDelaySlewWithCtxFloat(ctx, cap2_f, d2_f, s2_f);
    if (!isfinite(d2_f)) {
        dmp_db->rd_[pin_idx] = nanf("");
        return;
    }
    const float rd_f = static_cast<float>(
        -log(ctx.driver_vth) * fabsf(d1_f - d2_f) / (cap2_f - cap1_f));
    dmp_db->rd_[pin_idx] = static_cast<double>(rd_f);
    if (!isfinite(dmp_db->rd_[pin_idx]) || dmp_db->rd_[pin_idx] <= 0.0) {
        dmp_db->rd_[pin_idx] = nanf("");
    }
}

__device__ void
dmp_model::gateCapDelaySlew(double lc, double &delay, double &slew){ // maybe there is some better way to do this
    delay = nanf("");
    slew = nanf("");
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    int el = i >> 2;                            // early late
    int fel_rf = i >> 1;                        // from early/late rise/fall
    int tel_rf = ((i & 0b100) >> 1) + (i & 1);  // to early/late rise/fall
    int irf = fel_rf & 1;                       // input rise/fall
    int orf = tel_rf & 1;                       // output rise/fall
    int arc_id = arc_ids[idx];
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    // int to_pin_id = timing_arc_to_pin_id[arc_id];

    if ((timing_arc_id_map[arc_id * 2 + el] == -1) || isnan(pinSlew[from_pin_id * NUM_ATTR + fel_rf])) return;
    float si = pinSlew[from_pin_id * NUM_ATTR + fel_rf];
    int timing_id = timing_arc_id_map[arc_id * 2 + el];
    dmpGateCapDelaySlewCached(this, timing_id, irf, orf, si, lc, delay, slew);

}

__device__ void
dmp_model::gateDelays(double ceff, double &t_vth, double &t_vl, double &slew){
    t_vth = nanf("");
    t_vl = nanf("");
    slew = nanf("");
    double table_slew = nanf("");
    gateCapDelaySlew(ceff, t_vth, table_slew);
    if (!isfinite(t_vth) || !isfinite(table_slew)) return;
    int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(this, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    slew = table_slew * driver_derate;
    t_vl = t_vth - slew * (driver_vth - driver_vl) / (driver_vh - driver_vl);
}




__device__ void dmp_model::gateModelRd(int pin_idx, double d1, double s1){
    double cap1 = C1[pin_idx] + C2[pin_idx];
    const double kGateModelRdCapDelta = 1e-15 / cap_unit;
    double cap2 = cap1 + kGateModelRdCapDelta;
    double d2 = nanf("");
    double s2 = nanf("");
    if (!isfinite(cap1) || !isfinite(cap2) || !isfinite(d1) ||
        !isfinite(kGateModelRdCapDelta) || kGateModelRdCapDelta <= 0.0 ||
        cap2 == cap1) {
        rd_[pin_idx] = nanf("");
        return;
    }
    gateCapDelaySlew(cap2, d2, s2);
    if (!isfinite(d2)) {
        rd_[pin_idx] = nanf("");
        return;
    }
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(this, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    rd_[pin_idx] = -log(driver_vth) * fabs(d1 - d2) / (cap2 - cap1); // dak : strange, should be 1/log(vth), seems like rd is not independent of inslew
    if (!isfinite(rd_[pin_idx]) || rd_[pin_idx] <= 0.0) {
        rd_[pin_idx] = nanf("");
    }
    // double cap = C1[pin_idx] + C2[pin_idx];
    // rd_[pin_idx] = s1 / cap / log(vh_ / vl_);
    // if(rd_[pin_idx] > 0.1)
        // printf("pin_idx:%d rd:%.4f cap1:%.4f cap2:%.4f d1:%.4f d2:%.4f s1:%.4f s2:%.4f\n", pin_idx, rd_[pin_idx], C1[pin_idx], C2[pin_idx], d1, d2, s1, s2);
    // printf("pin_idx:%d rd:%.4f cap:%.4f slew:%.4f\n", pin_idx, rd_[pin_idx], cap, s1);
}
__device__ int dmp_model::selectDmpAlg(int pin_idx) {
    double rd = rd_[pin_idx];
    double c1 = C1[pin_idx];
    double c2 = C2[pin_idx];
    double rpi = r_pi[pin_idx];
    if (!isfinite(rd) || !isfinite(c1) || !isfinite(c2) || !isfinite(rpi) ||
        rd <= 0.0 || c1 <= 0.0 || c2 < 0.0 || rpi <= 0.0) {
        return DMP_ALG_CAP;
    }
    const double min_rd = (isfinite(res_unit) && res_unit > 0.0f)
                              ? (1e-2 / static_cast<double>(res_unit))
                              : 1e-2;
    if (rd < min_rd || rpi < rd * 1e-3 || c1 < c2 * 1e-3) {
        return DMP_ALG_CAP;
    }
    if (c2 < c1 * 1e-3) {
        return DMP_ALG_ZERO_C2;
    }
    return DMP_ALG_PI;
}

__device__ bool dmp_model::init_dmp_factors(int pin_idx){
    double denom_z1 = r_pi[pin_idx] * C1[pin_idx];
    double denom_k0 = rd_[pin_idx] * C2[pin_idx];
    if (!isfinite(denom_z1) || !isfinite(denom_k0) ||
        denom_z1 == 0.0 || denom_k0 == 0.0) {
        return false;
    }
    z1_[pin_idx] = 1.0 / (r_pi[pin_idx] * C1[pin_idx]);
    k0_[pin_idx] = 1.0 / (rd_[pin_idx] * C2[pin_idx]);
    double a = r_pi[pin_idx] * rd_[pin_idx] * C1[pin_idx] * C2[pin_idx];
    double b = rd_[pin_idx] * (C1[pin_idx] + C2[pin_idx]) + r_pi[pin_idx] * C1[pin_idx];
    double disc = b * b - 4 * a;
    if (!isfinite(a) || !isfinite(b) || !isfinite(disc) || a == 0.0 || disc < 0.0) {
        return false;
    }
    double sqrt_ = sqrt(disc);
    p1_[pin_idx] = (b + sqrt_) / (2 * a);
    p2_[pin_idx] = (b - sqrt_) / (2 * a);

    double p1p2 = (p1_[pin_idx] * p2_[pin_idx]);
    if (!isfinite(p1p2) || p1p2 == 0.0 || p1_[pin_idx] == p2_[pin_idx]) {
        return false;
    }
    k2_[pin_idx] = z1_[pin_idx] / p1p2;
    k1_[pin_idx] = (1.0 - k2_[pin_idx] * (p1_[pin_idx] + p2_[pin_idx])) / p1p2;
    k4_[pin_idx] = (k1_[pin_idx] * p1_[pin_idx] + k2_[pin_idx]) / (p2_[pin_idx] - p1_[pin_idx]);
    k3_[pin_idx] = -k1_[pin_idx] - k4_[pin_idx];

    double z_ = (C1[pin_idx] + C2[pin_idx]) / (r_pi[pin_idx] * C1[pin_idx] * C2[pin_idx]);
    A_[pin_idx] = z_ / p1p2;
    B_[pin_idx] = (z_ - p1_[pin_idx]) / (p1_[pin_idx] * (p1_[pin_idx] - p2_[pin_idx]));
    D_[pin_idx] = (z_ - p2_[pin_idx]) / (p2_[pin_idx] * (p2_[pin_idx] - p1_[pin_idx]));
    return isfinite(z1_[pin_idx]) && isfinite(k0_[pin_idx]) &&
           isfinite(k1_[pin_idx]) && isfinite(k2_[pin_idx]) &&
           isfinite(k3_[pin_idx]) && isfinite(k4_[pin_idx]) &&
           isfinite(p1_[pin_idx]) && isfinite(p2_[pin_idx]) &&
           isfinite(A_[pin_idx]) && isfinite(B_[pin_idx]) &&
           isfinite(D_[pin_idx]);
}

__device__ bool dmp_model::init_zero_c2_factors(int pin_idx){
    double denom_z1 = r_pi[pin_idx] * C1[pin_idx];
    double denom_p1 = C1[pin_idx] * (rd_[pin_idx] + r_pi[pin_idx]);
    if (!isfinite(denom_z1) || !isfinite(denom_p1) ||
        denom_z1 == 0.0 || denom_p1 == 0.0) {
        return false;
    }
    ceff[pin_idx] = C1[pin_idx];
    z1_[pin_idx] = 1.0 / (r_pi[pin_idx] * C1[pin_idx]);
    p1_[pin_idx] = 1.0 / (C1[pin_idx] * (rd_[pin_idx] + r_pi[pin_idx]));
    k0_[pin_idx] = p1_[pin_idx] / z1_[pin_idx];
    if (!isfinite(k0_[pin_idx]) || k0_[pin_idx] == 0.0 || p1_[pin_idx] == 0.0) {
        return false;
    }
    k2_[pin_idx] = 1.0 / k0_[pin_idx];
    k1_[pin_idx] = (p1_[pin_idx] - z1_[pin_idx]) / (p1_[pin_idx] * p1_[pin_idx]);
    k3_[pin_idx] = -k1_[pin_idx];
    p2_[pin_idx] = 0.0;
    k4_[pin_idx] = 0.0;
    A_[pin_idx] = 0.0;
    B_[pin_idx] = 0.0;
    D_[pin_idx] = 0.0;
    return isfinite(z1_[pin_idx]) && isfinite(k0_[pin_idx]) &&
           isfinite(k1_[pin_idx]) && isfinite(k2_[pin_idx]) &&
           isfinite(k3_[pin_idx]) && isfinite(p1_[pin_idx]);
}
// Eqn 13, Eqn 14.
__device__ double
dmp_model::ipiIceff(int pin_idx, double dt, double ceff_time, double ceff)
{
  double exp_p1_dt = exp2(-p1_[pin_idx] * ceff_time);
  double exp_p2_dt = exp2(-p2_[pin_idx] * ceff_time);
  double exp_dt_rd_ceff = exp2(-ceff_time / (rd_[pin_idx] * ceff));
  double ipi = (A_[pin_idx] * ceff_time + (B_[pin_idx] / p1_[pin_idx]) * (1.0 - exp_p1_dt) + (D_[pin_idx] / p2_[pin_idx]) * (1.0 - exp_p2_dt)) / (rd_[pin_idx] * ceff_time * dt);
  double iceff = (rd_[pin_idx] * ceff * ceff_time - (rd_[pin_idx] * ceff) * (rd_[pin_idx] * ceff) * (1.0 - exp_dt_rd_ceff)) / (rd_[pin_idx] * ceff_time * dt);
  return ipi - iceff;
}
__device__ bool
dmp_model::evalDmpEqns(double *x_, double *fvec_, double (*fjac_)[3], int size){
    double t0 = x_[DmpParam::t0];
    double dt = x_[DmpParam::dt];
    double ceff = x_[DmpParam::ceff];
    int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(this, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    if (ceff < 0.0){
        if(debug_on)printf("Error: eqn eval failed: ceff < 0\n");
        return false;
    }
    if (ceff > (C1[pin_idx] + C2[pin_idx])){
        if(debug_on)printf("Error: eqn eval failed: ceff > c2 + c1\n");
        return false;
    }

    double t_vth, t_vl, slew;
    gateDelays(ceff, t_vth, t_vl, slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(slew) || slew == 0.0){
        if(debug_on)printf("Error: eqn eval failed: slew = 0\n");
        return false;
    }

    double ceff_time = slew / (driver_vh - driver_vl);
    if (ceff_time > 1.4 * dt)
        ceff_time = 1.4 * dt;

    if (size == 2) {
        if (dt <= 0.0) {
            dt = x_[DmpParam::dt] = (t_vl - t_vth) / 100.0;
        }
        double ignore;
        double y50 = y(t_vth, t0, dt, rd_[pin_idx], ceff);
        double y20 = y(t_vl, t0, dt, rd_[pin_idx], ceff);
        fvec_[DmpFunc::y50] = y50 - driver_vth;
        fvec_[DmpFunc::y20] = y20 - driver_vl;
        dy(t_vl, t0, dt, rd_[pin_idx], ceff, fjac_[DmpFunc::y20][DmpParam::t0], fjac_[DmpFunc::y20][DmpParam::dt], ignore);
        dy(t_vth, t0, dt, rd_[pin_idx], ceff, fjac_[DmpFunc::y50][DmpParam::t0], fjac_[DmpFunc::y50][DmpParam::dt], ignore);
        return true;
    }
    if (dt <= 0.0){
        if(debug_on)printf("Error: eqn eval failed: dt < 0\n");
        return false;
    }
    double exp_p1_dt = exp2(-p1_[pin_idx] * dt);
    double exp_p2_dt = exp2(-p2_[pin_idx] * dt);
    double exp_dt_rd_ceff = exp2(-dt / (rd_[pin_idx] * ceff));

    double y50 = y(t_vth, t0, dt, rd_[pin_idx], ceff);
    // Match Vl.
    double y20 = y(t_vl, t0, dt, rd_[pin_idx], ceff);
    fvec_[DmpFunc::ipi] = ipiIceff(pin_idx, dt, ceff_time, ceff);
    fvec_[DmpFunc::y50] = y50 - driver_vth;
    fvec_[DmpFunc::y20] = y20 - driver_vl;
    fjac_[DmpFunc::ipi][DmpParam::t0] = 0.0;
    fjac_[DmpFunc::ipi][DmpParam::dt] =
        (-A_[pin_idx] * dt + B_[pin_idx] * dt * exp_p1_dt - (2 * B_[pin_idx] / p1_[pin_idx]) * (1.0 - exp_p1_dt) + D_[pin_idx] * dt * exp_p2_dt - (2 * D_[pin_idx] / p2_[pin_idx]) * (1.0 - exp_p2_dt) + rd_[pin_idx] * ceff * (dt + dt * exp_dt_rd_ceff - 2 * rd_[pin_idx] * ceff * (1.0 - exp_dt_rd_ceff))) / (rd_[pin_idx] * dt * dt * dt);
    fjac_[DmpFunc::ipi][DmpParam::ceff] =
        (2 * rd_[pin_idx] * ceff - dt - (2 * rd_[pin_idx] * ceff + dt) * exp2(-dt / (rd_[pin_idx] * ceff))) / (dt * dt);

    dy(t_vl, t0, dt, rd_[pin_idx], ceff, fjac_[DmpFunc::y20][DmpParam::t0], fjac_[DmpFunc::y20][DmpParam::dt], fjac_[DmpFunc::y20][DmpParam::ceff]);

    dy(t_vth, t0, dt, rd_[pin_idx], ceff, fjac_[DmpFunc::y50][DmpParam::t0], fjac_[DmpFunc::y50][DmpParam::dt], fjac_[DmpFunc::y50][DmpParam::ceff]);

    return true;
}
// luDecomp, luSolve based on MatClass from C. R. Birchenhall,
// University of Manchester
// ftp://ftp.mcc.ac.uk/pub/matclass/libmat.tar.Z

// Crout's Method of LU decomposition of square matrix, with implicit
// partial pivoting.  A is overwritten. U is explicit in the upper
// triangle and L is in multiplier form in the subdiagionals i.e. subdiag
// a[i,j] is the multiplier used to eliminate the [i,j] term.
//
// Replaces a[0..size-1][0..size-1] by the LU decomposition.
// index[0..size-1] is an output vector of the row permutations.
// Return error msg on failure.
__device__ bool
luDecomp(double (*a)[3], const int size, int *index,
         // Temporary supplied by caller.
         // scale stores the implicit scaling of each row.
         double *scale){
  // Find implicit scaling factors.
    for (int i = 0; i < size; i++) {
        double big = 0.0;
        for (int j = 0; j < size; j++) {
        double temp = abs(a[i][j]);
        if (temp > big)
            big = temp;
        }
        if (big == 0.0){
            // Caller falls back to table-cap delay when DMP Newton/LU fails.
            return false;
        }
        scale[i] = 1.0 / big;
    }
    int size_1 = size - 1;
    for (int j = 0; j < size; j++) {
        // Run down jth column from top to diag, to form the elements of U.
        for (int i = 0; i < j; i++) {
            double sum = a[i][j];
            for (int k = 0; k < i; k++)
                sum -= a[i][k] * a[k][j];
            a[i][j] = sum;
        }
        // Run down jth subdiag to form the residuals after the elimination
        // of the first j-1 subdiags.  These residuals diviyded by the
        // appropriate diagonal term will become the multipliers in the
        // elimination of the jth. subdiag. Find index of largest scaled
        // term in imax.
        double big = 0.0;
        int imax = 0;
        for (int i = j; i < size; i++) {
            double sum = a[i][j];
            for (int k = 0; k < j; k++)
                sum -= a[i][k] * a[k][j];
            a[i][j] = sum;
            double dum = scale[i] * abs(sum);
            if (dum >= big) {
                big = dum;
                imax = i;
            }
        }
        // Permute current row with imax.
        if (j != imax) {
        // Yes, do so...
            for (int k = 0; k < size; k++) {
                double dum = a[imax][k];
                a[imax][k] = a[j][k];
                a[j][k] = dum;
            }
            scale[imax] = scale[j];
        }
        index[j] = imax;
        // If diag term is not zero divide subdiag to form multipliers.
        if (a[j][j] == 0.0)
            a[j][j] = 1e-12;
        if (j != size_1) {
            double pivot = 1.0 / a[j][j];
            for (int i = j + 1; i < size; i++)
                a[i][j] *= pivot;
        }
    }
    return true;
}

// Solves the set of size linear equations a*x=b, assuming A is LU form
// but assume b has not been transformed.
//  a[0..size-1] is LU decomposition
// Returns the solution vector x in b.
// a and index are not modified.
__device__ void luSolve(double (*a)[3], const int size, const int *index, double b[]){
// Transform b allowing for leading zeros.
    int non_zero = -1;
    for (int i = 0; i < size; i++) {
        int iperm = index[i];
        double sum = b[iperm];
        b[iperm] = b[i];
        if (non_zero != -1) {
            for (int j = non_zero; j <= i - 1; j++)
                sum -= a[i][j] * b[j];
        }
        else {
            if (sum != 0.0)
                non_zero = i;
        }
        b[i] = sum;
    }
    // Backsubstitution.
    for (int i = size - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < size; j++)
            sum -= a[i][j] * b[j];
        b[i] = sum / a[i][i];
    }
}
__device__ bool dmp_model::newtonRaphson(int max_iter, int size, double *x, double (*fjac)[3], double *fvec, int *index, double *p, double *scale){
    for (int k = 0; k < max_iter; k++) {
        bool error = !evalDmpEqns(x, fvec, fjac, size);
        if(debug_on){
            int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
            int arc_id = arc_ids[blockIdx.x * blockDim.x + threadIdx.x];
            int from_pin_id = timing_arc_from_pin_id[arc_id];
            int to_pin_id = timing_arc_to_pin_id[arc_id];
            double t0 = x[DmpParam::t0];
            double dt = x[DmpParam::dt];
            double ceff = x[DmpParam::ceff];
            printf("k=%d from:%s to:%s ceff:%e dt:%e t0:%e fvec:ipi:%e y50:%e y20:%e\n", k, pin_names[from_pin_id], pin_names[to_pin_id], ceff, dt, t0, fvec[DmpFunc::ipi], fvec[DmpFunc::y50], fvec[DmpFunc::y20]);
        }
        if(error)return false;
        for (int i = 0; i < size; i++)
        // Right-hand side of linear equations.
            p[i] = -fvec[i];
        error |= !luDecomp(fjac, size, index, scale);
        if(error)return false;
        luSolve(fjac, size, index, p);

        bool all_under_x_tol = true;
        for (int i = 0; i < size; i++) {
            if (abs(p[i]) > abs(x[i]) * x_tol)
                all_under_x_tol = false;
            x[i] += p[i];
        }
        if (all_under_x_tol){
            // printf("NR converged in %d iterations\n", k);
            return evalDmpEqns(x, fvec, fjac, size);
        }
    }
    if(debug_on)printf("Error: Newton-Raphson exceeded maximum iterations\n");
    return false;
}

__device__ bool dmp_model::findDriverParams(double delay, double slew, double initial_ceff){
    int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(this, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    double t_vth, t_vl, measured_slew;
    gateDelays(initial_ceff, t_vth, t_vl, measured_slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(initial_ceff) || measured_slew <= 0.0 || initial_ceff < 0.0) {
        return false;
    }
    (void)delay;
    (void)slew;
    double init_dt = measured_slew / (driver_vh - driver_vl);
    double init_t0 = t_vth + log(1.0 - driver_vth) * rd_[pin_idx] * initial_ceff - driver_vth * init_dt;
    if (!isfinite(init_t0) || !isfinite(init_dt) || init_dt <= 0.0) {
        return false;
    }
    double x_[3]; // hard code pi_model dimension is 3
    double fjac_[3][3];
    double fvec_[3];
    int index[3];
    double p[3];
    double scale[3] = {1.0, 1.0, 1.0};
    x_[DmpParam::t0] = init_t0;
    x_[DmpParam::dt] = init_dt;
    x_[DmpParam::ceff] = initial_ceff;
    if (!newtonRaphson(100, 3, x_, fjac_, fvec_, index, p, scale)) {
        return false;
    }
    if (!isfinite(x_[DmpParam::t0]) || !isfinite(x_[DmpParam::dt]) ||
        !isfinite(x_[DmpParam::ceff]) || x_[DmpParam::dt] <= 0.0 ||
        x_[DmpParam::ceff] < 0.0 || x_[DmpParam::ceff] > C1[pin_idx] + C2[pin_idx]) {
        return false;
    }
    t0[pin_idx] = x_[DmpParam::t0];
    dt[pin_idx] = x_[DmpParam::dt];
    ceff[pin_idx] = x_[DmpParam::ceff];
    return true;
}

__device__ bool dmp_model::findDriverParamsOnePole(double delay, double slew, double fixed_ceff){
    int pin_idx = pin_ids[blockIdx.x * blockDim.x + threadIdx.x];
    double driver_vth, driver_vl, driver_vh, driver_derate;
    dmpLoadSlotThresholds(this, pin_idx, driver_vth, driver_vl, driver_vh, driver_derate);
    double t_vth, t_vl, measured_slew;
    gateDelays(fixed_ceff, t_vth, t_vl, measured_slew);
    if (!isfinite(t_vth) || !isfinite(t_vl) || !isfinite(measured_slew) ||
        !isfinite(fixed_ceff) || measured_slew <= 0.0 || fixed_ceff <= 0.0) {
        return false;
    }
    (void)delay;
    (void)slew;
    double init_dt = measured_slew / (driver_vh - driver_vl);
    double init_t0 = t_vth + log(1.0 - driver_vth) * rd_[pin_idx] * fixed_ceff - driver_vth * init_dt;
    if (!isfinite(init_t0) || !isfinite(init_dt)) {
        return false;
    }
    double x_[3] = {0.0, 0.0, fixed_ceff};
    double fjac_[3][3];
    double fvec_[3];
    int index[3];
    double p[3];
    double scale[3] = {1.0, 1.0, 1.0};
    x_[DmpParam::t0] = init_t0;
    x_[DmpParam::dt] = init_dt;
    x_[DmpParam::ceff] = fixed_ceff;
    if (!newtonRaphson(100, 2, x_, fjac_, fvec_, index, p, scale)) {
        return false;
    }
    if (!isfinite(x_[DmpParam::t0]) || !isfinite(x_[DmpParam::dt]) ||
        x_[DmpParam::dt] <= 0.0) {
        return false;
    }
    t0[pin_idx] = x_[DmpParam::t0];
    dt[pin_idx] = x_[DmpParam::dt];
    ceff[pin_idx] = fixed_ceff;
    return true;
}

