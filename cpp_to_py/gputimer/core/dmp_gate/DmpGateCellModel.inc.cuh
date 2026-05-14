
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

__device__ __forceinline__ void dmpGateCapDelaySlewCached(DmpModel* dmp_db,
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
    int driver_library_id;
    bool valid;
};

__device__ __forceinline__ DmpGateLaneContext dmpMakeGateLaneContext(DmpModel* dmp_db,
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
    const int attr = pin_idx & (NUM_ATTR - 1);
    const int pin_id = pin_idx / NUM_ATTR;
    ctx.driver_library_id = dmpPinLibraryId(dmp_db, pin_id, attr);
    dmpDriverLibraryThresholds(dmp_db,
                               ctx.driver_library_id,
                               attr,
                               ctx.driver_vth,
                               ctx.driver_vl,
                               ctx.driver_vh,
                               ctx.driver_derate);

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
