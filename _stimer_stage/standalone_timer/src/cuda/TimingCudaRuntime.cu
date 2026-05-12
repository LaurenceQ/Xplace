#include "stimer/TimingCudaRuntime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <string>
#include <vector>

namespace stimer {

namespace {

__global__ void smoke_kernel(int* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = 7;
  }
}

struct TimingEvalKernelArgs {
  int num_arcs = 0;
  const FlatTimingArc* arcs = nullptr;
  const FlatLut* luts = nullptr;
  const float* indices1 = nullptr;
  const float* indices2 = nullptr;
  const float* values = nullptr;
  const float* arc_input_slew = nullptr;
  const float* arc_load_cap = nullptr;
  float* arc_delay = nullptr;
  float* arc_output_slew = nullptr;
  float* arc_arrival = nullptr;
};

CudaStatus cuda_failure(cudaError_t error, const char* step) {
  CudaStatus status;
  status.ok = false;
  status.message = std::string(step) + ": " + cudaGetErrorString(error);
  status.elapsed_ms = 0.0f;
  cudaGetLastError();
  return status;
}

TimingEvalResult cuda_eval_failure(cudaError_t error, const char* step) {
  TimingEvalResult result;
  result.ok = false;
  result.message = std::string(step) + ": " + cudaGetErrorString(error);
  cudaGetLastError();
  return result;
}

__device__ bool device_variable_is_slew(int variable) {
  return variable ==
             static_cast<int>(LibertyLutVariable::kInputNetTransition) ||
         variable ==
             static_cast<int>(LibertyLutVariable::kRelatedPinTransition) ||
         variable ==
             static_cast<int>(LibertyLutVariable::kConstrainedPinTransition) ||
         variable ==
             static_cast<int>(LibertyLutVariable::kInputTransitionTime);
}

__device__ bool device_variable_is_cap(int variable) {
  return variable ==
         static_cast<int>(LibertyLutVariable::kTotalOutputNetCapacitance);
}

__device__ float device_axis_value(int variable,
                                   float input_slew,
                                   float load_cap) {
  if (device_variable_is_slew(variable)) {
    return input_slew;
  }
  if (device_variable_is_cap(variable)) {
    return load_cap;
  }
  return 0.0f;
}

__device__ int device_lower_index(const float* data,
                                  int begin,
                                  int size,
                                  float x) {
  if (size <= 1) {
    return 0;
  }
  if (x <= data[begin]) {
    return 0;
  }
  for (int i = 0; i + 1 < size; ++i) {
    if (x <= data[begin + i + 1]) {
      return i;
    }
  }
  return size - 2;
}

__device__ float device_lut_value(const TimingEvalKernelArgs& args,
                                  int lut_id,
                                  float input_slew,
                                  float load_cap) {
  if (lut_id < 0) {
    return 0.0f;
  }
  const FlatLut lut = args.luts[lut_id];
  if (!lut.valid || lut.value_size == 0) {
    return 0.0f;
  }

  const int size1 = lut.index1_size > 0 ? lut.index1_size : 1;
  const int size2 = lut.index2_size > 0 ? lut.index2_size : 1;
  if (size1 == 1 && size2 == 1) {
    return args.values[lut.values_begin];
  }

  const float x = device_axis_value(lut.variable1, input_slew, load_cap);
  const float y = device_axis_value(lut.variable2, input_slew, load_cap);
  const int x0_id =
      device_lower_index(args.indices1, lut.index1_begin, lut.index1_size, x);
  const int y0_id =
      device_lower_index(args.indices2, lut.index2_begin, lut.index2_size, y);
  const int x1_id = lut.index1_size <= 1 ? x0_id : x0_id + 1;
  const int y1_id = lut.index2_size <= 1 ? y0_id : y0_id + 1;

  const float x0 = lut.index1_size <= 1 ? x : args.indices1[lut.index1_begin + x0_id];
  const float x1 = lut.index1_size <= 1 ? x : args.indices1[lut.index1_begin + x1_id];
  const float y0 = lut.index2_size <= 1 ? y : args.indices2[lut.index2_begin + y0_id];
  const float y1 = lut.index2_size <= 1 ? y : args.indices2[lut.index2_begin + y1_id];

  const int q00_offset = lut.values_begin + x0_id * size2 + y0_id;
  const int q01_offset = lut.values_begin + x0_id * size2 + y1_id;
  const int q10_offset = lut.values_begin + x1_id * size2 + y0_id;
  const int q11_offset = lut.values_begin + x1_id * size2 + y1_id;
  const int value_end = lut.values_begin + lut.value_size;
  const float q00 =
      (q00_offset >= lut.values_begin && q00_offset < value_end)
          ? args.values[q00_offset]
          : 0.0f;
  const float q01 =
      (q01_offset >= lut.values_begin && q01_offset < value_end)
          ? args.values[q01_offset]
          : 0.0f;
  const float q10 =
      (q10_offset >= lut.values_begin && q10_offset < value_end)
          ? args.values[q10_offset]
          : 0.0f;
  const float q11 =
      (q11_offset >= lut.values_begin && q11_offset < value_end)
          ? args.values[q11_offset]
          : 0.0f;
  const float dx = x1 - x0;
  const float dy = y1 - y0;
  const float tx = fabsf(dx) < 1e-12f ? 0.0f : (x - x0) / dx;
  const float ty = fabsf(dy) < 1e-12f ? 0.0f : (y - y0) / dy;
  const float a = q00 * (1.0f - tx) + q10 * tx;
  const float b = q01 * (1.0f - tx) + q11 * tx;
  return a * (1.0f - ty) + b * ty;
}

__global__ void timing_eval_kernel(TimingEvalKernelArgs args) {
  const int arc_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (arc_id >= args.num_arcs) {
    return;
  }

  const FlatTimingArc arc = args.arcs[arc_id];
  const float input_slew = args.arc_input_slew[arc_id];
  const float load_cap = args.arc_load_cap[arc_id];
  const int delay_lut =
      arc.delay_rise_lut >= 0 ? arc.delay_rise_lut : arc.delay_fall_lut;
  const int slew_lut =
      arc.slew_rise_lut >= 0 ? arc.slew_rise_lut : arc.slew_fall_lut;

  const float delay =
      device_lut_value(args, delay_lut, input_slew, load_cap);
  const float output_slew =
      device_lut_value(args, slew_lut, input_slew, load_cap);
  args.arc_delay[arc_id] = delay;
  args.arc_output_slew[arc_id] = output_slew;
  args.arc_arrival[arc_id] = delay;
}

template <typename T>
cudaError_t copy_vector_to_device(const std::vector<T>& host, T** device) {
  *device = nullptr;
  if (host.empty()) {
    return cudaSuccess;
  }
  cudaError_t error = cudaMalloc(device, host.size() * sizeof(T));
  if (error != cudaSuccess) {
    return error;
  }
  return cudaMemcpy(*device, host.data(), host.size() * sizeof(T),
                    cudaMemcpyHostToDevice);
}

}  // namespace

CudaStatus run_cuda_runtime_smoke_test() {
  cudaGetLastError();

  int* device_value = nullptr;
  cudaError_t error = cudaMalloc(&device_value, sizeof(int));
  if (error != cudaSuccess) {
    return cuda_failure(error, "cudaMalloc");
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  error = cudaEventCreate(&start);
  if (error != cudaSuccess) {
    cudaFree(device_value);
    return cuda_failure(error, "cudaEventCreate(start)");
  }
  error = cudaEventCreate(&stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(start);
    cudaFree(device_value);
    return cuda_failure(error, "cudaEventCreate(stop)");
  }

  error = cudaEventRecord(start);
  if (error != cudaSuccess) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(device_value);
    return cuda_failure(error, "cudaEventRecord(start)");
  }

  smoke_kernel<<<1, 32>>>(device_value);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(device_value);
    return cuda_failure(error, "smoke_kernel launch");
  }

  error = cudaEventRecord(stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(device_value);
    return cuda_failure(error, "cudaEventRecord(stop)");
  }

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(device_value);
    return cuda_failure(error, "cudaDeviceSynchronize");
  }

  float elapsed_ms = 0.0f;
  error = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (error != cudaSuccess) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(device_value);
    return cuda_failure(error, "cudaEventElapsedTime");
  }

  int host_value = 0;
  error = cudaMemcpy(&host_value, device_value, sizeof(int),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(device_value);
    return cuda_failure(error, "cudaMemcpy");
  }

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cudaFree(device_value);
  cudaGetLastError();

  CudaStatus status;
  status.ok = (host_value == 7);
  status.elapsed_ms = elapsed_ms;
  status.message = status.ok ? "ok" : "unexpected smoke kernel value";
  return status;
}

TimingEvalResult evaluate_timing_cuda(const TimingFlatDB& flat,
                                      float* elapsed_ms) {
  if (elapsed_ms != nullptr) {
    *elapsed_ms = 0.0f;
  }

  TimingEvalResult result;
  result.num_arcs = flat.num_arcs;
  result.num_luts = flat.num_luts;
  result.num_values = flat.num_values;
  if (flat.num_arcs == 0) {
    return result;
  }

  cudaGetLastError();

  TimingEvalKernelArgs args;
  args.num_arcs = flat.num_arcs;
  FlatTimingArc* d_arcs = nullptr;
  FlatLut* d_luts = nullptr;
  float* d_indices1 = nullptr;
  float* d_indices2 = nullptr;
  float* d_values = nullptr;
  float* d_arc_input_slew = nullptr;
  float* d_arc_load_cap = nullptr;
  float* d_arc_delay = nullptr;
  float* d_arc_output_slew = nullptr;
  float* d_arc_arrival = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t error = cudaSuccess;

  const auto cleanup = [&]() {
    if (stop != nullptr) cudaEventDestroy(stop);
    if (start != nullptr) cudaEventDestroy(start);
    cudaFree(d_arc_arrival);
    cudaFree(d_arc_output_slew);
    cudaFree(d_arc_delay);
    cudaFree(d_arc_load_cap);
    cudaFree(d_arc_input_slew);
    cudaFree(d_values);
    cudaFree(d_indices2);
    cudaFree(d_indices1);
    cudaFree(d_luts);
    cudaFree(d_arcs);
  };

  const auto fail = [&](cudaError_t cuda_error, const char* step) {
    cleanup();
    return cuda_eval_failure(cuda_error, step);
  };

  error = copy_vector_to_device(flat.arcs, &d_arcs);
  if (error != cudaSuccess) return fail(error, "cudaMalloc/copy arcs");
  args.arcs = d_arcs;
  error = copy_vector_to_device(flat.luts, &d_luts);
  if (error != cudaSuccess) return fail(error, "cudaMalloc/copy luts");
  args.luts = d_luts;
  error = copy_vector_to_device(flat.indices1, &d_indices1);
  if (error != cudaSuccess) return fail(error, "cudaMalloc/copy indices1");
  args.indices1 = d_indices1;
  error = copy_vector_to_device(flat.indices2, &d_indices2);
  if (error != cudaSuccess) return fail(error, "cudaMalloc/copy indices2");
  args.indices2 = d_indices2;
  error = copy_vector_to_device(flat.values, &d_values);
  if (error != cudaSuccess) return fail(error, "cudaMalloc/copy values");
  args.values = d_values;
  error = copy_vector_to_device(flat.arc_input_slew, &d_arc_input_slew);
  if (error != cudaSuccess) return fail(error, "cudaMalloc/copy input slew");
  args.arc_input_slew = d_arc_input_slew;
  error = copy_vector_to_device(flat.arc_load_cap, &d_arc_load_cap);
  if (error != cudaSuccess) return fail(error, "cudaMalloc/copy load cap");
  args.arc_load_cap = d_arc_load_cap;

  error = cudaMalloc(&d_arc_delay, flat.arcs.size() * sizeof(float));
  if (error != cudaSuccess) return fail(error, "cudaMalloc arc_delay");
  error = cudaMalloc(&d_arc_output_slew, flat.arcs.size() * sizeof(float));
  if (error != cudaSuccess) return fail(error, "cudaMalloc arc_output_slew");
  error = cudaMalloc(&d_arc_arrival, flat.arcs.size() * sizeof(float));
  if (error != cudaSuccess) return fail(error, "cudaMalloc arc_arrival");
  args.arc_delay = d_arc_delay;
  args.arc_output_slew = d_arc_output_slew;
  args.arc_arrival = d_arc_arrival;

  error = cudaEventCreate(&start);
  if (error != cudaSuccess) return fail(error, "cudaEventCreate(start)");
  error = cudaEventCreate(&stop);
  if (error != cudaSuccess) return fail(error, "cudaEventCreate(stop)");
  error = cudaEventRecord(start);
  if (error != cudaSuccess) return fail(error, "cudaEventRecord(start)");

  constexpr int block_size = 128;
  const int grid_size = (flat.num_arcs + block_size - 1) / block_size;
  timing_eval_kernel<<<grid_size, block_size>>>(args);
  error = cudaGetLastError();
  if (error != cudaSuccess) return fail(error, "timing_eval_kernel launch");

  error = cudaEventRecord(stop);
  if (error != cudaSuccess) return fail(error, "cudaEventRecord(stop)");
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) return fail(error, "cudaDeviceSynchronize");

  if (elapsed_ms != nullptr) {
    error = cudaEventElapsedTime(elapsed_ms, start, stop);
    if (error != cudaSuccess) return fail(error, "cudaEventElapsedTime");
  }

  std::vector<float> arc_delay(flat.arcs.size(), 0.0f);
  std::vector<float> arc_output_slew(flat.arcs.size(), 0.0f);
  std::vector<float> arc_arrival(flat.arcs.size(), 0.0f);
  error = cudaMemcpy(arc_delay.data(), d_arc_delay,
                     arc_delay.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail(error, "cudaMemcpy arc_delay");
  error = cudaMemcpy(arc_output_slew.data(), d_arc_output_slew,
                     arc_output_slew.size() * sizeof(float),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail(error, "cudaMemcpy arc_output_slew");
  error = cudaMemcpy(arc_arrival.data(), d_arc_arrival,
                     arc_arrival.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) return fail(error, "cudaMemcpy arc_arrival");

  for (std::size_t i = 0; i < arc_delay.size(); ++i) {
    result.max_delay = std::max(result.max_delay, arc_delay[i]);
    result.max_output_slew =
        std::max(result.max_output_slew, arc_output_slew[i]);
    result.max_arrival = std::max(result.max_arrival, arc_arrival[i]);
  }

  cleanup();
  cudaGetLastError();

  return result;
}

}  // namespace stimer
