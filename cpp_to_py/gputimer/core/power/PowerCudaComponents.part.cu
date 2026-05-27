__global__ void power_pack_output_kernel(const PowerActivityCudaModel* model,
                                         const PowerActivityScratchView* scratch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->n) return;
    model->out[idx * 3 + 0] = scratch->density[idx];
    model->out[idx * 3 + 1] = scratch->duty[idx];
    model->out[idx * 3 + 2] = static_cast<float>(scratch->origin[idx]);
}

__global__ void power_unpack_precomputed_activity_kernel(int n,
                                                         const float* activity,
                                                         float* density,
                                                         float* duty,
                                                         int* origin) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    density[idx] = activity[idx * 3 + 0];
    duty[idx] = activity[idx * 3 + 1];
    origin[idx] = static_cast<int>(activity[idx * 3 + 2]);
}

__global__ void power_unpack_activity_density_duty_kernel(int n,
                                                          const float* activity,
                                                          float* density,
                                                          float* duty) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    density[idx] = activity[idx * 3 + 0];
    duty[idx] = activity[idx * 3 + 1];
}

__global__ void power_switching_kernel(const PowerActivityCudaModel* model) {
    const int pin = blockIdx.x * blockDim.x + threadIdx.x;
    const auto& graph = model->graph;
    const auto& components = model->components;
    const int n = model->n;
    if (pin >= n) return;
    float sw = 0.0f;
    if (graph.is_driver_pin[pin] && (!graph.is_cell_pin || graph.is_cell_pin[pin])) {
        float load_internal = 0.0f;
        // OpenSTA Power::findSwitchingPower asks GraphDelayCalc::loadCap(..., MinMax::max())
        // and then takes max over rise/fall.  In the flattened attr layout, max/late
        // rise/fall are attrs 2 and 3 (early/min are attrs 0 and 1).
        for (int i = 2; i < NUM_ATTR; i++) {
            float v = graph.pinLoad ? graph.pinLoad[pin * NUM_ATTR + i] : 0.0f;
            if (graph.dmp_C1 && graph.dmp_C2) {
                const double dv = graph.dmp_C1[pin * NUM_ATTR + i] + graph.dmp_C2[pin * NUM_ATTR + i];
                v = isfinite(dv) && dv > 0.0 ? static_cast<float>(dv) : 0.0f;
            }
            if (isfinite(v)) load_internal = fmaxf(load_internal, v);
        }
        const float density = model->out[pin * 3 + 0];
        if (load_internal > 0.0f && density > 0.0f && components.voltage > 0.0f) {
            const float load_f = load_internal * components.cap_unit;
            sw = 0.5f * load_f * components.voltage * components.voltage * density;
        }
        const int node = graph.pin2node_map[pin];
        if (components.inst_switching && node >= 0 && node < graph.num_nodes && sw != 0.0f) atomicAdd(&components.inst_switching[node], sw);
    }
    if (components.pin_switching) components.pin_switching[pin] = sw;
}

__device__ float power_expr_duty_cuda(int expr_id,
                                      const GpuPowerExprOpHost* ops,
                                      const int* expr_start,
                                      const int* expr_count,
                                      const float* pin_density,
                                      const float* pin_duty,
                                      const int* node_port_pin_start = nullptr,
                                      const int* node_port_pin_list = nullptr,
                                      int node_id = -1) {
    float d = 0.0f, u = 0.0f;
    if (!power_eval_expr_activity(expr_id, ops, expr_start, expr_count, pin_density, pin_duty, d, u,
                                  node_port_pin_start, node_port_pin_list, node_id)) return 0.0f;
    return fminf(fmaxf(u, 0.0f), 1.0f);
}

__device__ float power_expr_diff_duty_cuda(int expr_id,
                                           int diff_pin,
                                           const GpuPowerExprOpHost* ops,
                                           const int* expr_start,
                                           const int* expr_count,
                                           const float* pin_duty) {
    if (expr_id < 0 || diff_pin < 0) return 0.0f;
    PowerBddContextCuda ctx;
    int root = 1;
    if (!power_bdd_build_expr(expr_id, ops, expr_start, expr_count,
                              nullptr, pin_duty, ctx, root)) {
        return 0.0f;
    }
    int diff_var = -1;
    for (int var = 0; var < ctx.var_count; var++) {
        if (ctx.var_pins[var] == diff_pin) {
            diff_var = var;
            break;
        }
    }
    if (diff_var < 0) return 0.0f;
    const int low = power_bdd_restrict(ctx, root, diff_var, false);
    const int high = power_bdd_restrict(ctx, root, diff_var, true);
    const int diff = power_bdd_apply(ctx, 2, low, high);
    return power_bdd_eval_duty(ctx, diff);
}

__device__ float power_internal_row_duty(const GpuPowerInternalHost& row,
                                         const GpuPowerExprOpHost* expr_ops,
                                         const int* expr_start,
                                         const int* expr_count,
                                         const float* density,
                                         const float* duty,
                                         const int* node_port_pin_start,
                                         const int* node_port_pin_list) {
    if (row.duty_mode == 0) return 1.0f;
    if (row.duty_mode == 1) return power_expr_duty_cuda(row.duty_expr_id, expr_ops, expr_start, expr_count,
                                                        density, duty, node_port_pin_start,
                                                        node_port_pin_list, row.node_id);
    if (row.duty_mode == 2) return power_expr_diff_duty_cuda(row.duty_expr_id, row.duty_pin, expr_ops, expr_start, expr_count, duty);
    if (row.duty_mode == 3) return 0.5f;
    return 0.0f;
}

__global__ void power_internal_denom_kernel(const PowerInternalDenomModel* model,
                                            const PowerActivityScratchView* scratch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->num_internal_rows) return;
    const auto row = model->internal_rows[idx];
    if (row.kind != 1 || row.denom_group < 0 || row.from_pin < 0) return;
    const float d = power_internal_row_duty(row, model->expr_ops, model->expr_start, model->expr_count,
                                            scratch->density, scratch->duty,
                                            model->node_port_pin_start, model->node_port_pin_list);
    const float numer = scratch->density[row.from_pin] * d;
    if (isfinite(numer) && numer != 0.0f) atomicAdd(&model->denom[row.denom_group], numer);
}

__global__ void power_internal_contrib_kernel(const PowerInternalContribModel* model,
                                              const PowerActivityScratchView* scratch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->num_internal_rows) return;
    if (model->internal_row_power) model->internal_row_power[idx] = 0.0f;
    const auto row = model->internal_rows[idx];
    if (row.node_id < 0 || row.node_id >= model->num_nodes || row.to_pin < 0 || !model->power_allocator) return;
    const float row_duty = power_internal_row_duty(row, model->expr_ops, model->expr_start, model->expr_count,
                                                   scratch->density, scratch->duty,
                                                   model->node_port_pin_start, model->node_port_pin_list);
    float weight = (row.kind == 0) ? row_duty : 1.0f;
    if (row.kind == 1) {
        if (row.from_pin < 0 || row.denom_group < 0) return;
        const float numer = scratch->density[row.from_pin] * row_duty;
        const float den = model->denom[row.denom_group];
        if (!(den != 0.0f) || !isfinite(den) || !isfinite(numer)) return;
        weight = numer / den;
    }
    float load_internal = 0.0f;
    if (row.kind == 1 && model->dmp_C1 && model->dmp_C2) {
        for (int attr = 2; attr < NUM_ATTR; ++attr) {
            const double dv = model->dmp_C1[row.to_pin * NUM_ATTR + attr] + model->dmp_C2[row.to_pin * NUM_ATTR + attr];
            if (isfinite(dv) && dv > 0.0) load_internal = fmaxf(load_internal, static_cast<float>(dv));
        }
    }
    const int slew_pin = row.kind == 0 ? row.to_pin : row.from_pin;
    float slew_r = 0.0f, slew_f = 0.0f;
    if (slew_pin >= 0 && model->pinSlew) {
        auto slew_value = [&](int attr) {
            float value = model->pinSlew[slew_pin * NUM_ATTR + attr];
            if (model->power_clock_slews) {
                const float clock_slew = model->power_clock_slews[slew_pin * NUM_ATTR + attr];
                if (isfinite(clock_slew)) value = clock_slew;
            }
            return value;
        };
        if (row.kind == 0 || row.positive_unate) {
            slew_r = slew_value(2);
            slew_f = slew_value(3);
        } else {
            slew_r = slew_value(3);
            slew_f = slew_value(2);
        }
    }
    float energy = 0.0f;
    int rf_count = 0;
    if (isfinite(slew_r)) {
        const float e = model->power_allocator->query_internal_power(row.internal_power_id, 0, slew_r, load_internal);
        if (isfinite(e)) { energy += e; rf_count++; }
    }
    if (isfinite(slew_f)) {
        const float e = model->power_allocator->query_internal_power(row.internal_power_id, 1, slew_f, load_internal);
        if (isfinite(e)) { energy += e; rf_count++; }
    }
    if (rf_count == 0) return;
    energy /= static_cast<float>(rf_count);
    const float energy_unit = (isfinite(row.energy_unit) && row.energy_unit > 0.0f)
        ? row.energy_unit
        : model->cap_unit;
    const float p = weight * energy * energy_unit * scratch->density[row.to_pin];
    if (isfinite(p)) {
        if (model->internal_row_power) model->internal_row_power[idx] = p;
        if (p != 0.0f && model->inst_internal) atomicAdd(&model->inst_internal[row.node_id], p);
    }
}

__global__ void power_leakage_row_kernel(const PowerLeakageRowsModel* model,
                                         const PowerActivityScratchView* scratch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->num_leakage_rows) return;
    if (model->leakage_row_power) model->leakage_row_power[idx] = 0.0f;
    const auto row = model->leakage_rows[idx];
    if (row.group_id < 0) return;
    float cond_duty = 1.0f;
    if (row.when_expr_id >= 0) {
        cond_duty = power_expr_duty_cuda(row.when_expr_id, model->expr_ops, model->expr_start,
                                         model->expr_count, scratch->density, scratch->duty,
                                         model->node_port_pin_start, model->node_port_pin_list,
                                         row.node_id);
    }
    const float weighted = row.leakage * cond_duty;
    if (isfinite(weighted)) {
        atomicAdd(&model->group_cond_leakage[row.group_id], weighted);
        if (row.leakage > 0.0f) atomicAdd(&model->group_cond_duty_sum[row.group_id], cond_duty);
        atomicAdd(&model->group_cond_count[row.group_id], 1);
        if (model->leakage_row_power) model->leakage_row_power[idx] = weighted;
    }
}

__global__ void power_leakage_summary_kernel(const PowerLeakageSummaryModel* model) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= model->num_leakage_groups) return;
    const auto group = model->leakage_groups[idx];
    if (group.node_id < 0 || group.node_id >= model->num_nodes) return;
    float leakage = 0.0f;
    if (model->group_cond_count[idx] > 0) {
        float fallback_duty = 1.0f - model->group_cond_duty_sum[idx];
        leakage = model->group_cond_leakage[idx] + group.cell_leakage * fallback_duty;
    } else {
        leakage = group.cell_leakage;
    }
    if (isfinite(leakage) && leakage != 0.0f) atomicAdd(&model->inst_leakage[group.node_id], leakage);
}

}  // namespace
void clear_power_cuda_error() { cudaGetLastError(); }

void check_power_cuda_error(const char* label) {
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::string where = label ? label : "unknown";
        cudaGetLastError();
        throw std::runtime_error("[power] CUDA error at " + where + ": " +
                                 cudaGetErrorString(sync_err));
    }
    cudaError_t last_err = cudaGetLastError();
    if (last_err != cudaSuccess) {
        std::string where = label ? label : "unknown";
        throw std::runtime_error("[power] CUDA error at " + where + ": " +
                                 cudaGetErrorString(last_err));
    }
}

static void power_cuda_call(cudaError_t err, const char* label) {
    if (err == cudaSuccess) return;
    std::string where = label ? label : "unknown";
    throw std::runtime_error("[power] CUDA call failed at " + where + ": " +
                             cudaGetErrorString(err));
}

static void unpack_power_activity_density_duty(int n,
                                               const float* d_precomputed_activity,
                                               float** d_density,
                                               float** d_duty) {
    *d_density = nullptr;
    *d_duty = nullptr;
    if (n <= 0 || !d_precomputed_activity) return;
    cudaMalloc(d_density, sizeof(float) * n);
    cudaMalloc(d_duty, sizeof(float) * n);
    power_unpack_activity_density_duty_kernel<<<BLOCK_NUMBER(n), BLOCK_SIZE>>>(
        n, d_precomputed_activity, *d_density, *d_duty);
    cudaDeviceSynchronize();
}

void run_power_internal_denom_chunk_cuda_launcher(const PowerInternalDenomModel& model) {
    int n = model.n;
    const float* d_precomputed_activity = model.precomputed_activity;
    GpuPowerInternalHost* d_internal_rows = model.internal_rows;
    const int num_internal_rows = model.num_internal_rows;
    float* d_denom = model.denom;
    if (n <= 0 || !d_precomputed_activity || !d_internal_rows || num_internal_rows <= 0 || !d_denom)
        return;
    float* d_density = nullptr;
    float* d_duty = nullptr;
    unpack_power_activity_density_duty(n, d_precomputed_activity, &d_density, &d_duty);
    if (d_density && d_duty) {
        PowerActivityScratchView scratch;
        scratch.density = d_density;
        scratch.duty = d_duty;
        PowerInternalDenomModel* d_model = nullptr;
        PowerActivityScratchView* d_scratch = nullptr;
        cudaMalloc(&d_model, sizeof(PowerInternalDenomModel));
        cudaMalloc(&d_scratch, sizeof(PowerActivityScratchView));
        cudaMemcpy(d_model, &model, sizeof(PowerInternalDenomModel), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scratch, &scratch, sizeof(PowerActivityScratchView), cudaMemcpyHostToDevice);
        power_internal_denom_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            d_model, d_scratch);
        cudaDeviceSynchronize();
        cudaFree(d_model);
        cudaFree(d_scratch);
    }
    cudaFree(d_density);
    cudaFree(d_duty);
}

void run_power_internal_contrib_chunk_cuda_launcher(const PowerInternalContribModel& model) {
    int n = model.n;
    const float* d_precomputed_activity = model.precomputed_activity;
    GpuPowerInternalHost* d_internal_rows = model.internal_rows;
    const int num_internal_rows = model.num_internal_rows;
    const float* d_denom = model.denom;
    GPUPowerLutAllocator* d_power_allocator = model.power_allocator;
    if (n <= 0 || !d_precomputed_activity || !d_internal_rows || num_internal_rows <= 0 ||
        !d_denom || !d_power_allocator)
        return;
    float* d_density = nullptr;
    float* d_duty = nullptr;
    unpack_power_activity_density_duty(n, d_precomputed_activity, &d_density, &d_duty);
    if (d_density && d_duty) {
        PowerActivityScratchView scratch;
        scratch.density = d_density;
        scratch.duty = d_duty;
        PowerInternalContribModel* d_model = nullptr;
        PowerActivityScratchView* d_scratch = nullptr;
        cudaMalloc(&d_model, sizeof(PowerInternalContribModel));
        cudaMalloc(&d_scratch, sizeof(PowerActivityScratchView));
        cudaMemcpy(d_model, &model, sizeof(PowerInternalContribModel), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scratch, &scratch, sizeof(PowerActivityScratchView), cudaMemcpyHostToDevice);
        power_internal_contrib_kernel<<<BLOCK_NUMBER(num_internal_rows), BLOCK_SIZE>>>(
            d_model, d_scratch);
        cudaDeviceSynchronize();
        cudaFree(d_model);
        cudaFree(d_scratch);
    }
    cudaFree(d_density);
    cudaFree(d_duty);
}

void run_power_leakage_rows_chunk_cuda_launcher(const PowerLeakageRowsModel& model) {
    const int n = model.n;
    const float* d_precomputed_activity = model.precomputed_activity;
    GpuPowerLeakageRowHost* d_leakage_rows = model.leakage_rows;
    const int num_leakage_rows = model.num_leakage_rows;
    float* d_group_cond_leakage = model.group_cond_leakage;
    float* d_group_cond_duty_sum = model.group_cond_duty_sum;
    int* d_group_cond_count = model.group_cond_count;
    if (n <= 0 || !d_precomputed_activity || !d_leakage_rows || num_leakage_rows <= 0 ||
        !d_group_cond_leakage || !d_group_cond_duty_sum || !d_group_cond_count)
        return;
    float* d_density = nullptr;
    float* d_duty = nullptr;
    unpack_power_activity_density_duty(n, d_precomputed_activity, &d_density, &d_duty);
    if (d_density && d_duty) {
        PowerActivityScratchView scratch;
        scratch.density = d_density;
        scratch.duty = d_duty;
        PowerLeakageRowsModel* d_model = nullptr;
        PowerActivityScratchView* d_scratch = nullptr;
        cudaMalloc(&d_model, sizeof(PowerLeakageRowsModel));
        cudaMalloc(&d_scratch, sizeof(PowerActivityScratchView));
        cudaMemcpy(d_model, &model, sizeof(PowerLeakageRowsModel), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scratch, &scratch, sizeof(PowerActivityScratchView), cudaMemcpyHostToDevice);
        power_leakage_row_kernel<<<BLOCK_NUMBER(num_leakage_rows), BLOCK_SIZE>>>(
            d_model, d_scratch);
        cudaDeviceSynchronize();
        cudaFree(d_model);
        cudaFree(d_scratch);
    }
    cudaFree(d_density);
    cudaFree(d_duty);
}

void run_power_leakage_summary_chunk_cuda_launcher(const PowerLeakageSummaryModel& model) {
    GpuPowerLeakageGroupHost* d_leakage_groups = model.leakage_groups;
    const int num_leakage_groups = model.num_leakage_groups;
    float* d_group_cond_leakage = model.group_cond_leakage;
    float* d_group_cond_duty_sum = model.group_cond_duty_sum;
    int* d_group_cond_count = model.group_cond_count;
    float* d_inst_leakage = model.inst_leakage;
    if (!d_leakage_groups || num_leakage_groups <= 0 || !d_group_cond_leakage ||
        !d_group_cond_duty_sum || !d_group_cond_count || !d_inst_leakage)
        return;
    PowerLeakageSummaryModel* d_model = nullptr;
    cudaMalloc(&d_model, sizeof(PowerLeakageSummaryModel));
    cudaMemcpy(d_model, &model, sizeof(PowerLeakageSummaryModel), cudaMemcpyHostToDevice);
    power_leakage_summary_kernel<<<BLOCK_NUMBER(num_leakage_groups), BLOCK_SIZE>>>(d_model);
    check_power_cuda_error("leakage summary chunk");
    cudaFree(d_model);
}
void run_power_activity_cuda_launcher(const PowerActivityCudaModel& model) {
