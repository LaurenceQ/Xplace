void apply_dmp_driving_cell_source_slew_cuda(dmp_model* dmp_db,
                                             const std::vector<int>& pin_ids,
                                             const std::vector<int>& timing_ids,
                                             const std::vector<int>& input_rfs,
                                             const std::vector<float>& input_slews) {
    const int num_sources = static_cast<int>(pin_ids.size());
    const int total = num_sources * NUM_ATTR;
    const bool profile_kernels = dmp_kernel_profile_enabled();
    const bool collect_counts = profile_kernels || dmp_timing_debug_enabled();
    if (num_sources == 0 || total == 0) {
        if (collect_counts) {
            printf("[DMP DRIVING CELL] sources=0 lanes=0 applied=0 skipped=0 cap=0 zero_c2=0 pi=0 dmp_valid=0 fallback=0\n");
            fflush(stdout);
        }
        return;
    }

    int* d_pin_ids = nullptr;
    int* d_timing_ids = nullptr;
    int* d_input_rfs = nullptr;
    float* d_input_slews = nullptr;
    unsigned long long* d_counts = nullptr;
    unsigned long long h_counts[DMP_DRIVING_CELL_COUNTER_COUNT] = {0};

    gpuErrchk(cudaMalloc(&d_pin_ids, sizeof(int) * num_sources));
    gpuErrchk(cudaMalloc(&d_timing_ids, sizeof(int) * total));
    gpuErrchk(cudaMalloc(&d_input_rfs, sizeof(int) * total));
    gpuErrchk(cudaMalloc(&d_input_slews, sizeof(float) * total));
    if (collect_counts) {
        gpuErrchk(cudaMalloc(&d_counts, sizeof(h_counts)));
    }
    gpuErrchk(cudaMemcpy(d_pin_ids, pin_ids.data(), sizeof(int) * num_sources, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_timing_ids, timing_ids.data(), sizeof(int) * total, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_input_rfs, input_rfs.data(), sizeof(int) * total, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_input_slews, input_slews.data(), sizeof(float) * total, cudaMemcpyHostToDevice));
    if (collect_counts) {
        gpuErrchk(cudaMemset(d_counts, 0, sizeof(h_counts)));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (profile_kernels) {
        dmp_event_create(&start, &stop);
        gpuErrchk(cudaEventRecord(start));
    }
    applyDrivingCellSourceSlewKernel<<<DMP_TIMING_BLOCK_NUMBER(total), DMP_TIMING_BLOCK_SIZE>>>(dmp_db,
                                                                                               d_pin_ids,
                                                                                               d_timing_ids,
                                                                                               d_input_rfs,
                                                                                               d_input_slews,
                                                                                               num_sources,
                                                                                               d_counts);
    gpuErrchk(cudaPeekAtLastError());
    float elapsed_ms = 0.0f;
    if (profile_kernels) {
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaEventElapsedTime(&elapsed_ms, start, stop));
        gpuErrchk(cudaEventDestroy(start));
        gpuErrchk(cudaEventDestroy(stop));
        cudaGetLastError();
    }
    if (collect_counts) {
        gpuErrchk(cudaMemcpy(h_counts, d_counts, sizeof(h_counts), cudaMemcpyDeviceToHost));

        printf("[DMP DRIVING CELL] sources=%d lanes=%d applied=%llu skipped=%llu cap=%llu zero_c2=%llu pi=%llu dmp_valid=%llu fallback=%llu\n",
               num_sources,
               total,
               h_counts[DMP_DRIVING_CELL_APPLIED],
               h_counts[DMP_DRIVING_CELL_SKIPPED],
               h_counts[DMP_DRIVING_CELL_CAP],
               h_counts[DMP_DRIVING_CELL_ZERO_C2],
               h_counts[DMP_DRIVING_CELL_PI],
               h_counts[DMP_DRIVING_CELL_DMP_VALID],
               h_counts[DMP_DRIVING_CELL_FALLBACK]);
    }
    if (profile_kernels) {
        printf("[DMP KERNEL PROFILE] name=applyDrivingCellSourceSlewKernel launches=1 total_ms=%.3f avg_us=%.3f max_ms=%.3f work_items=%d blocks=%d block=(%d,1) work_per_ms=%.1f\n",
               elapsed_ms,
               static_cast<double>(elapsed_ms) * 1000.0,
               elapsed_ms,
               total,
               DMP_TIMING_BLOCK_NUMBER(total),
               DMP_TIMING_BLOCK_SIZE,
               elapsed_ms > 0.0f ? static_cast<double>(total) / static_cast<double>(elapsed_ms) : 0.0);
    }
    if (collect_counts || profile_kernels) {
        fflush(stdout);
    }

    cudaFree(d_pin_ids);
    cudaFree(d_timing_ids);
    cudaFree(d_input_rfs);
    cudaFree(d_input_slews);
    if (d_counts != nullptr) {
        cudaFree(d_counts);
    }
}
