#pragma once

#include "common/StageProfiler.h"
#include "gputimer/core/power/common/PowerCudaModel.h"
#include "gputimer/db/GTDatabase.h"
#include "common/lib/Liberty.h"
#include "common/lib/Timing.h"
#include "io_parser/gp/GPDatabase.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gt {

struct PowerClockSlewSparse {
    std::vector<int> pins;
    std::array<float, NUM_ATTR> fallback{};
};

class PowerStageProfiler {
public:
    explicit PowerStageProfiler(bool enabled);
    void mark(const char* label);

private:
    StageProfiler profiler_;
};

struct PowerActivityLevelSelection {
    torch::Tensor level_list_gpu;
    std::vector<int> owned_level_list_end;
    std::vector<int> pin_power_level;
    const std::vector<int>* external_level_list_end = nullptr;
    index_type* level_list = nullptr;
    bool use_owned_level_list_end = false;

    PowerActivityLevelSelection();
    PowerActivityLevelSelection(torch::Tensor level_list_gpu_,
                                std::vector<int> owned_level_list_end_,
                                std::vector<int> pin_power_level_,
                                const std::vector<int>* external_level_list_end_,
                                index_type* level_list_,
                                bool use_owned_level_list_end_);

    const std::vector<int>* levelListEnd() const;
};

struct PowerCudaRunBuffers {
    torch::Tensor out_gpu;
    torch::Tensor inst_switching_gpu;
    torch::Tensor pin_switching_gpu;
    torch::Tensor inst_internal_gpu;
    torch::Tensor internal_row_power_gpu;
    torch::Tensor inst_leakage_gpu;
    torch::Tensor leakage_row_power_gpu;
    torch::Tensor precomputed_activity_cpu;
    torch::Tensor precomputed_activity_gpu;
    float* out_gpu_ptr = nullptr;
    float* activity_density_ptr = nullptr;
    float* activity_duty_ptr = nullptr;
    float* inst_switching_ptr = nullptr;
    float* pin_switching_ptr = nullptr;
    float* inst_internal_ptr = nullptr;
    float* internal_row_power_ptr = nullptr;
    float* inst_leakage_ptr = nullptr;
    float* leakage_row_power_ptr = nullptr;
    const float* precomputed_activity_ptr = nullptr;
    int out_activity_fields = 0;

    PowerCudaRunBuffers();
    PowerCudaRunBuffers(torch::Tensor out_gpu_,
                        torch::Tensor inst_switching_gpu_,
                        torch::Tensor pin_switching_gpu_,
                        torch::Tensor inst_internal_gpu_,
                        torch::Tensor internal_row_power_gpu_,
                        torch::Tensor inst_leakage_gpu_,
                        torch::Tensor leakage_row_power_gpu_,
                        torch::Tensor precomputed_activity_cpu_,
                        torch::Tensor precomputed_activity_gpu_,
                        float* out_gpu_ptr_,
                        float* activity_density_ptr_,
                        float* activity_duty_ptr_,
                        float* inst_switching_ptr_,
                        float* pin_switching_ptr_,
                        float* inst_internal_ptr_,
                        float* internal_row_power_ptr_,
                        float* inst_leakage_ptr_,
                        float* leakage_row_power_ptr_,
                        const float* precomputed_activity_ptr_,
                        int out_activity_fields_);
};

struct PowerDmpLoadPointers {
    const float* C1 = nullptr;
    const float* C2 = nullptr;

    PowerDmpLoadPointers();
    PowerDmpLoadPointers(const float* C1_, const float* C2_);
};

void dumpPowerPinNamesIfRequested(GTDatabase& gtdb, int n);

bool readPowerEnvFlag(const char* name, bool default_value);
int64_t readPowerEnvInt64(const char* name, int64_t default_value);
size_t readPowerChunkBytes(const char* env_name, size_t default_value);
size_t powerRowsPerChunk(size_t chunk_bytes, size_t elem_size);
float powerVoltageForReport(GTDatabase& gtdb);

struct PowerClockPinActivity {
    std::vector<int> pins;
    std::vector<float> densities;
    std::vector<float> duties;
    std::vector<uint8_t> enqueue;

    PowerClockPinActivity();
    PowerClockPinActivity(std::vector<int> pins_,
                          std::vector<float> densities_,
                          std::vector<float> duties_,
                          std::vector<uint8_t> enqueue_);
};

PowerClockPinActivity buildPowerClockPinActivity(
    GTDatabase& gtdb,
    int n,
    const std::vector<int>& pin_to_node,
    const std::vector<int>& pin_to_net,
    const std::vector<uint8_t>& is_load_pin,
    const std::vector<uint8_t>& is_driver_pin,
    const std::vector<uint8_t>& is_clock_gate_clock_pin,
    double sdc_time_scale,
    float clock_density);

void dumpPowerSeqIdMapIfRequested(GTDatabase& gtdb,
                                  const std::vector<GpuPowerSeqHost>& seqs,
                                  int n);
void printPowerSeqDupStatsIfRequested(GTDatabase& gtdb,
                                      const std::vector<GpuPowerSeqHost>& seqs,
                                      int n);

struct PowerTimingLoopInfo {
    std::vector<uint8_t> disabled_loop_arc;
    std::vector<int> roots;
    int disabled_loop_arc_count = 0;

    PowerTimingLoopInfo();
    explicit PowerTimingLoopInfo(size_t arc_count);
    PowerTimingLoopInfo(std::vector<uint8_t> disabled_loop_arc_,
                        std::vector<int> roots_,
                        int disabled_loop_arc_count_);
};

PowerTimingLoopInfo buildPowerTimingLoopInfo(GTDatabase& gtdb, int n, bool enabled);
bool shouldSkipSeqOutputArcForPower(GTDatabase& gtdb,
                                    int n,
                                    const std::vector<uint8_t>& is_seq_output_pin,
                                    int arc_id,
                                    int from_pin,
                                    int to_pin);

struct PowerCudaExprInputs;
struct PowerCudaSeqInputs;

struct PowerCudaRootInputs {
    std::vector<uint8_t> is_clock_pin;
    std::vector<uint8_t> is_primary_input;
    std::vector<int> primary_inputs;
    std::vector<std::string> seed_reason;
    std::vector<int> feedback_seed_pins;
    std::vector<int> feedback_seed_seqs;
    std::vector<uint32_t> seq_output_arc_keep;
    std::vector<uint8_t> disabled_loop_arc;
    bool seed_default_inputs = false;
    bool seed_seq_feedback_outputs = false;
    bool init_seq_feedback_state = false;
    bool seed_timing_loop_roots = false;
    bool skip_disabled_loop_arcs = false;
    int primary_count = 0;
    int zero_indeg_count = 0;
    int const_output_count = 0;
    int seq_feedback_count = 0;
    int state_seq_feedback_count = 0;
    int power_level_count = 0;
    int floating_load_count = 0;
    int timing_loop_count = 0;
    int disabled_loop_arc_count = 0;

    PowerCudaRootInputs();
    PowerCudaRootInputs(int n, size_t arc_count);

    void addSeedPin(int pin_id, const char* reason);
    void markSeqOutputArcKeep(int arc_id);
    bool seqOutputArcKept(int arc_id) const;
};

struct PowerCudaArcSkipInputs {
    std::vector<uint8_t> arc_skip;
    const int* flat_net2pin_start_map = nullptr;
    const int* flat_net2pin_map = nullptr;

    PowerCudaArcSkipInputs();
    PowerCudaArcSkipInputs(std::vector<uint8_t> arc_skip_,
                           const int* flat_net2pin_start_map_,
                           const int* flat_net2pin_map_);
};

PowerCudaRootInputs buildPowerCudaRootInputs(GTDatabase& gtdb,
                                             int n,
                                             const std::vector<int>& clock_pins,
                                             const std::vector<int>& pin_to_node,
                                             const std::vector<int>& pin_to_net,
                                             const std::vector<int>& net_driver_pin,
                                             const std::vector<uint8_t>& is_load_pin,
                                             const std::vector<uint8_t>& is_driver_pin,
                                             const PowerCudaSeqInputs& seq_inputs,
                                             const PowerCudaExprInputs& expr_inputs);

PowerCudaArcSkipInputs buildPowerCudaArcSkipInputs(GTDatabase& gtdb,
                                                   const PowerCudaRootInputs& roots,
                                                   const int* default_flat_net2pin_start_map,
                                                   const int* default_flat_net2pin_map);

void finalizePowerCudaRootInputs(GTDatabase& gtdb,
                                 int n,
                                 PowerCudaRootInputs& roots,
                                 const PowerCudaSeqInputs& seq_inputs,
                                 const std::vector<uint8_t>& is_load_pin,
                                 const std::vector<uint8_t>& is_driver_pin,
                                 const std::vector<int>& pin_to_node,
                                 const std::vector<int>& pin_to_net,
                                 const std::vector<int>& power_level_root_pins_cpu,
                                 const std::vector<int>& power_pin_level_cpu,
                                 const std::vector<uint8_t>& arc_skip);

PowerClockSlewSparse buildPowerClockSlews(GTDatabase& gtdb,
                                          int n,
                                          const std::vector<int>& clock_pins,
                                          const std::vector<uint8_t>& is_seq_clock_input_pin,
                                          const std::vector<int>& pin_to_net,
                                          bool need_internal_power);

void printPowerRowStatsIfRequested(const std::vector<GpuPowerInternalHost>& internal_rows,
                                   const std::vector<GpuPowerLeakageRowHost>& leakage_rows,
                                   size_t internal_row_bytes,
                                   size_t leakage_row_bytes,
                                   size_t internal_chunk_bytes,
                                   size_t leakage_chunk_bytes,
                                   bool chunk_internal_rows,
                                   bool chunk_leakage_rows,
                                   size_t denom_group_count,
                                   size_t leakage_group_count,
                                   size_t expr_ops_count,
                                   size_t expr_bytes,
                                   size_t expr_cache_count);

torch::Tensor powerCudaIntTensor(const std::vector<int>& v);
torch::Tensor powerCudaIndexTensor(const std::vector<index_type>& v);
torch::Tensor powerCudaU8Tensor(const std::vector<uint8_t>& v);
torch::Tensor powerCudaFloatTensor(const std::vector<float>& v);

template <typename VecT>
torch::Tensor powerCudaBytesTensor(const VecT& v) {
    using ElemT = typename VecT::value_type;
    auto bopt_cpu = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    if (v.empty()) return torch::zeros({(long)sizeof(ElemT)}, bopt_cpu).to(torch::kCUDA);
    auto* data = reinterpret_cast<uint8_t*>(const_cast<ElemT*>(v.data()));
    return torch::from_blob(data, {(long)(v.size() * sizeof(ElemT))}, bopt_cpu).to(torch::kCUDA);
}

template <typename VecT>
torch::Tensor powerCudaBytesTensorRange(const VecT& v, size_t begin, size_t count) {
    using ElemT = typename VecT::value_type;
    auto bopt_cpu = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    if (count == 0) return torch::zeros({(long)sizeof(ElemT)}, bopt_cpu).to(torch::kCUDA);
    auto* data = const_cast<ElemT*>(v.data() + begin);
    return torch::from_blob(reinterpret_cast<uint8_t*>(data),
                            {(long)(count * sizeof(ElemT))}, bopt_cpu).to(torch::kCUDA);
}

class PowerCudaUploader {
public:
    PowerCudaUploader(bool debug, bool sync_debug);

    torch::Tensor uploadInt(const char* label, const std::vector<int>& v) const;
    torch::Tensor uploadIndex(const char* label, const std::vector<index_type>& v) const;
    torch::Tensor uploadU8(const char* label, const std::vector<uint8_t>& v) const;
    torch::Tensor uploadFloat(const char* label, const std::vector<float>& v) const;

    template <typename VecT>
    torch::Tensor uploadBytes(const char* label, const VecT& v) const {
        using ElemT = typename VecT::value_type;
        mark("begin", label, v.size(), sizeof(ElemT));
        auto out = powerCudaBytesTensor(v);
        mark("end", label, v.size(), sizeof(ElemT));
        return out;
    }

private:
    void mark(const char* phase, const char* label, size_t count, size_t elem_size) const;

    bool debug_ = false;
    bool sync_debug_ = false;
};

torch::Tensor outputPowerTensorForRequest(const torch::Tensor& tensor,
                                          bool output_power_tensors_cuda);

using PowerConstPortResolver = std::function<int(const gp::GPNode&, const std::string&)>;

struct PowerCudaExprInputs {
    std::vector<GpuPowerExprOpHost> ops;
    std::vector<int> start;
    std::vector<int> count;
    std::vector<int> pin_expr_id;
    std::vector<int> missing_func_out_start;
    std::vector<int> missing_func_out_list;
    std::unordered_map<std::string, int> template_expr_cache;

    PowerCudaExprInputs();
    explicit PowerCudaExprInputs(int n);

    int addExpr(const std::string& expr_str,
                LibertyCell* cell,
                const gp::GPNode& node,
                const PowerConstPortResolver& const_port_value_for_node,
                bool* used_missing_const = nullptr,
                bool zero_scan_enable_density = false);
    int addTemplateExpr(const std::string& expr_str,
                        LibertyCell* cell,
                        bool zero_scan_enable_density = false);
    bool containsPin(const GTDatabase& gtdb, int expr_id, int pin_id) const;
    bool templatePortsPresent(int expr_id,
                              const std::vector<int>& port_pin_by_offset) const;
};

struct PowerCudaSeqInputs {
    std::vector<GpuPowerSeqHost> seqs;
    std::vector<uint8_t> is_seq_output_pin;
    std::vector<uint8_t> is_seq_clock_input_pin;
    std::vector<int> pin_seq_list_start;
    std::vector<int> pin_seq_list;

    PowerCudaSeqInputs();
    explicit PowerCudaSeqInputs(int n);
    PowerCudaSeqInputs(std::vector<GpuPowerSeqHost> seqs_,
                       std::vector<uint8_t> is_seq_output_pin_,
                       std::vector<uint8_t> is_seq_clock_input_pin_,
                       std::vector<int> pin_seq_list_start_,
                       std::vector<int> pin_seq_list_);
};

PowerCudaExprInputs buildPowerCudaExprInputs(GTDatabase& gtdb,
                                             int n,
                                             const std::vector<uint8_t>& is_load_pin,
                                             const std::vector<uint8_t>& is_driver_pin);

PowerCudaSeqInputs buildPowerCudaSeqInputs(GTDatabase& gtdb,
                                           int n,
                                           const std::vector<int>& pin_to_node,
                                           const std::vector<uint8_t>& is_load_pin,
                                           PowerCudaExprInputs& expr_inputs);

int addPowerCudaExpr(const std::string& expr_str,
                     LibertyCell* cell,
                     const gp::GPNode& node,
                     std::vector<GpuPowerExprOpHost>& expr_ops,
                     std::vector<int>& expr_start,
                     std::vector<int>& expr_count,
                     const PowerConstPortResolver& const_port_value_for_node,
                     bool* used_missing_const = nullptr,
                     bool zero_scan_enable_density = false);

int addPowerCudaTemplateExpr(const std::string& expr_str,
                             LibertyCell* cell,
                             std::vector<GpuPowerExprOpHost>& expr_ops,
                             std::vector<int>& expr_start,
                             std::vector<int>& expr_count,
                             std::unordered_map<std::string, int>& template_expr_cache,
                             bool zero_scan_enable_density = false);

bool powerCudaExprContainsPin(int expr_id,
                              int pin_id,
                              int port_offset,
                              const std::vector<GpuPowerExprOpHost>& expr_ops,
                              const std::vector<int>& expr_start,
                              const std::vector<int>& expr_count);
bool powerCudaTemplateExprPortsPresent(int expr_id,
                                       const std::vector<int>& port_pin_by_offset,
                                       const std::vector<GpuPowerExprOpHost>& expr_ops,
                                       const std::vector<int>& expr_start,
                                       const std::vector<int>& expr_count);

bool positiveUnateForPower(LibertyCell* cell, LibertyPort* from, LibertyPort* to);

void dumpPowerCudaInputRoots(GTDatabase& gtdb,
                             int n,
                             const std::vector<int>& primary_inputs,
                             const std::vector<int>& candidate_roots,
                             const std::vector<std::string>& seed_reason,
                             const std::vector<uint8_t>& seed_seen,
                             const std::vector<uint8_t>& is_primary_input,
                             const std::vector<uint8_t>& is_clock_pin,
                             const std::vector<uint8_t>& is_driver_pin,
                             const std::vector<uint8_t>& is_load_pin,
                             const std::vector<int>& power_fanin,
                             const std::vector<int>& pin_to_node,
                             const std::vector<int>& pin_to_net,
                             const std::vector<int>& power_pin_level_cpu);

using PowerTemplateExprAdder = std::function<int(const std::string&, LibertyCell*)>;
using PowerExprPinPredicate = std::function<bool(int, int)>;

void buildPowerCudaInternalRows(GTDatabase& gtdb,
                                int n,
                                bool need_internal_power,
                                const std::vector<uint8_t>& is_load_pin,
                                const std::vector<uint8_t>& is_driver_pin,
                                PowerCudaExprInputs& expr_inputs,
                                std::vector<GpuPowerInternalHost>& internal_rows,
                                std::unordered_map<uint64_t, int>& internal_denom_group);

void buildPowerCudaLeakageRows(GTDatabase& gtdb,
                               bool need_leakage_power,
                               PowerCudaExprInputs& expr_inputs,
                               std::vector<GpuPowerLeakageRowHost>& leakage_rows,
                               std::vector<GpuPowerLeakageGroupHost>& leakage_groups);

}  // namespace gt
