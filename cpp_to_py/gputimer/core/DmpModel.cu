#include "DmpModel.h"
#include "DmpCudaUtils.cuh"
#include "GPUTimer.h"
#include "gputiming.h"
#include "gputimer/db/GTDatabase.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
// #include "utils.cuh"
namespace gt {

struct DmpInitAllocRecord {
    const char* name = "";
    size_t count = 0;
    size_t elem_size = 0;
    size_t bytes = 0;
};

static std::vector<DmpInitAllocRecord>& dmpInitAllocRecords()
{
    static std::vector<DmpInitAllocRecord> records;
    return records;
}

static size_t& dmpInitAllocatedBytes()
{
    static size_t bytes = 0;
    return bytes;
}

static bool dmpEnvEnabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

static bool dmpInitMemProfileEnabled()
{
    return dmpEnvEnabled("DMP_INIT_MEM_PROFILE");
}

static bool dmpInitSummaryEnabled()
{
    return dmpInitMemProfileEnabled() ||
           dmpEnvEnabled("DMP_INIT_SUMMARY") ||
           dmpEnvEnabled("DMP_DEBUG_TIMING") ||
           dmpEnvEnabled("XPLACE_TIMER_VERBOSE");
}

static double dmpMiB(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static double dmpGiB(size_t bytes)
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

static void dmpResetInitAllocRecords()
{
    dmpInitAllocRecords().clear();
    dmpInitAllocatedBytes() = 0;
}

static void dmpPrintCudaMemInfo(const char* label)
{
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err == cudaSuccess) {
        std::fprintf(stderr,
                     "[DMP INIT MEM] %s cuda_free=%.3f GiB cuda_total=%.3f GiB\n",
                     label,
                     dmpGiB(free_bytes),
                     dmpGiB(total_bytes));
    } else {
        std::fprintf(stderr,
                     "[DMP INIT MEM] %s cudaMemGetInfo failed: %s\n",
                     label,
                     cudaGetErrorString(err));
        cudaGetLastError();
    }
}

static void dmpRecordInitAlloc(const char* name, size_t count, size_t elem_size, size_t bytes)
{
    dmpInitAllocRecords().push_back(DmpInitAllocRecord{name, count, elem_size, bytes});
    dmpInitAllocatedBytes() += bytes;
    if (dmpInitMemProfileEnabled()) {
        std::fprintf(stderr,
                     "[DMP INIT MEM] alloc name=%s count=%zu elem=%zu bytes=%zu %.3f MiB cumulative=%.3f GiB\n",
                     name,
                     count,
                     elem_size,
                     bytes,
                     dmpMiB(bytes),
                     dmpGiB(dmpInitAllocatedBytes()));
    }
}

static void dmpDumpInitAllocFailure(const char* failed_name,
                                    size_t failed_count,
                                    size_t failed_elem_size,
                                    size_t failed_bytes,
                                    cudaError_t err)
{
    std::fprintf(stderr,
                 "[DMP INIT MEM] cudaMalloc failed name=%s count=%zu elem=%zu bytes=%zu %.3f MiB error=%s\n",
                 failed_name,
                 failed_count,
                 failed_elem_size,
                 failed_bytes,
                 dmpMiB(failed_bytes),
                 cudaGetErrorString(err));
    std::fprintf(stderr,
                 "[DMP INIT MEM] already_allocated=%zu bytes %.3f GiB records=%zu\n",
                 dmpInitAllocatedBytes(),
                 dmpGiB(dmpInitAllocatedBytes()),
                 dmpInitAllocRecords().size());
    dmpPrintCudaMemInfo("after_cudaMalloc_failure");
    for (const DmpInitAllocRecord& rec : dmpInitAllocRecords()) {
        std::fprintf(stderr,
                     "[DMP INIT MEM] previous name=%s count=%zu elem=%zu bytes=%zu %.3f MiB\n",
                     rec.name,
                     rec.count,
                     rec.elem_size,
                     rec.bytes,
                     dmpMiB(rec.bytes));
    }
    std::fflush(stderr);
}

template <typename T>
static void dmpCudaMallocChecked(T** ptr, size_t count, const char* name) {
    const size_t bytes = sizeof(T) * count;
    if (count == 0) {
        *ptr = nullptr;
        return;
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(ptr), bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "[DMP INIT] cudaMalloc failed for %s: count=%zu bytes=%zu error=%s\n",
                     name,
                     count,
                     bytes,
                     cudaGetErrorString(err));
        dmpDumpInitAllocFailure(name, count, sizeof(T), bytes, err);
        std::exit(err);
    }
    dmpRecordInitAlloc(name, count, sizeof(T), bytes);
}

template <typename T>
static void dmpCudaMemsetChecked(T* ptr, int value, size_t count, const char* name) {
    const size_t bytes = sizeof(T) * count;
    cudaError_t err = cudaMemset(ptr, value, bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "[DMP INIT] cudaMemset failed for %s: count=%zu bytes=%zu error=%s\n",
                     name,
                     count,
                     bytes,
                     cudaGetErrorString(err));
        std::exit(err);
    }
}

static bool dmpDeviceNamesEnabled()
{
    const char* value = std::getenv("GPUTIMER_DMP_DEVICE_NAMES");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

static constexpr bool DMP_UPLOAD_NAMES_FOR_DIRECT_CLOCK_DEBUG = false;
static constexpr bool DMP_UPLOAD_NAMES_FOR_DRIVING_CELL_DEBUG = false;

static bool dmpShouldUploadDeviceNames(bool debug_on)
{
    return debug_on ||
           DMP_UPLOAD_NAMES_FOR_DIRECT_CLOCK_DEBUG ||
           DMP_UPLOAD_NAMES_FOR_DRIVING_CELL_DEBUG ||
           dmpDeviceNamesEnabled();
}

static bool dmpDeferTimingAlloc()
{
    const char* value = std::getenv("DMP_DEFER_TIMING_ALLOC");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return !(value[0] == '0' ||
             value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}
__host__ DmpModel::DmpModel(GPUTimer* timer)
        : flat_net2pin_start_map(timer -> flat_net2pin_start_map),
        flat_net2pin_map(timer -> flat_net2pin_map),
        pin2net_map(timer -> pin2net_map),
        num_pins(timer -> num_pins),
        num_nets(timer -> num_nets),
        num_arcs(timer -> num_arcs),
        num_tests(timer -> num_tests),
        level_list(timer -> level_list),
        pin_forward_arc_list_end(timer -> pin_forward_arc_list_end),
        pin_forward_arc_list(timer -> pin_forward_arc_list),
        timing_arc_to_pin_id(timer -> timing_arc_to_pin_id),
        pin_backward_arc_list_end(timer -> pin_backward_arc_list_end),
        pin_backward_arc_list(timer -> pin_backward_arc_list),
        timing_arc_from_pin_id(timer -> timing_arc_from_pin_id),
        arc_types(timer -> arc_types),
        arc_id2test_id(timer -> arc_id2test_id),
        // C1(timer -> h_dmp_rc_ -> C1),
        // C2(timer -> h_dmp_rc_ -> C2),
        // r_pi(timer -> h_dmp_rc_ -> r_pi),
        // elmore_delay(timer -> h_dmp_rc_ -> elmore_delay),
        pinSlew(timer -> pinSlew),
        pinAt(timer -> pinAT),
        pinRat(timer -> pinRAT),
        timer_ceff(timer -> pinLoad),
        testRelatedAT(timer -> testRelatedAT),
        testRAT(timer -> testRAT),
        testConstraint(timer -> testConstraint),
        test_clock_ids(nullptr),
        clock_periods(nullptr),
        clock_count(0),
        pin_clock_rise_edges(timer -> pin_clock_rise_edges),
        pin_clock_fall_edges(timer -> pin_clock_fall_edges),
        pin_clock_slews(timer -> pin_clock_slews),
        test_setup_uncertainties(timer -> test_setup_uncertainties),
        test_hold_uncertainties(timer -> test_hold_uncertainties),
        arcDelay(timer -> arcDelay),
        timing_arc_id_map(timer -> timing_arc_id_map),
        at_prefix_pin(timer -> at_prefix_pin),
        at_prefix_arc(timer -> at_prefix_arc),
        at_prefix_attr(timer -> at_prefix_attr),
        dmp_input_thresholds(timer -> dmp_input_thresholds),
        dmp_output_thresholds(timer -> dmp_output_thresholds),
        dmp_slew_lower_thresholds(timer -> dmp_slew_lower_thresholds),
        dmp_slew_upper_thresholds(timer -> dmp_slew_upper_thresholds),
        dmp_slew_derates(timer -> dmp_slew_derates),
        dmp_timing_library_ids(timer -> dmp_timing_library_ids),
        dmp_pin_library_ids(timer -> dmp_pin_library_ids),
        dmp_library_input_thresholds(timer -> dmp_library_input_thresholds),
        dmp_library_output_thresholds(timer -> dmp_library_output_thresholds),
        dmp_library_slew_lower_thresholds(timer -> dmp_library_slew_lower_thresholds),
        dmp_library_slew_upper_thresholds(timer -> dmp_library_slew_upper_thresholds),
        dmp_library_slew_derates(timer -> dmp_library_slew_derates),
        clock_period(timer -> clock_period),
        d_allocator(timer -> d_allocator),
        res_unit(timer -> res_unit),
        cap_unit(timer -> cap_unit),
        debug_on(timer -> dmp_debug_on)
        {
    owns_allocations = true;
    dmpResetInitAllocRecords();
    if (dmpInitMemProfileEnabled()) {
        dmpPrintCudaMemInfo("before_dmp_model_alloc");
    }
    const long long pin_slot_count_ll = static_cast<long long>(num_pins) * NUM_ATTR;
    const long long slot_capacity_ll = pin_slot_count_ll;
    const long long work_slot_capacity_ll = pin_slot_count_ll * 2;
    if (pin_slot_count_ll > std::numeric_limits<int>::max() ||
        slot_capacity_ll > std::numeric_limits<int>::max() ||
        work_slot_capacity_ll > std::numeric_limits<int>::max()) {
        std::fprintf(stderr,
                     "[DMP INIT] slot capacity exceeds int indexing: pins=%d arcs=%d pin_slots=%lld slot_capacity=%lld work_slot_capacity=%lld\n",
                     num_pins,
                     num_arcs,
                     pin_slot_count_ll,
                     slot_capacity_ll,
                     work_slot_capacity_ll);
        std::exit(cudaErrorInvalidValue);
    }
    dmp_pin_slot_count = static_cast<int>(pin_slot_count_ll);
    dmp_slot_capacity = static_cast<int>(slot_capacity_ll);
    dmp_work_slot_capacity = static_cast<int>(work_slot_capacity_ll);
    if (dmpInitSummaryEnabled()) {
        std::fprintf(stderr,
                     "[DMP INIT] pins=%d nets=%d arcs=%d tests=%d pin_slots=%d slot_capacity=%d work_slot_capacity=%d mode=direct\n",
                     num_pins,
                     num_nets,
                     num_arcs,
                     num_tests,
                     dmp_pin_slot_count,
                     dmp_slot_capacity,
                     dmp_work_slot_capacity);
    }
    C1 = C2 = r_pi = nullptr;
    pin_at_winner = nullptr;
    pin_names = nullptr;
    net_names = nullptr;
    edge_from = nullptr;
    edge_to = nullptr;
    flat_net2node_start_map = nullptr;
    flat_net2edge_start_map = nullptr;
    node2pin_map = nullptr;
    edge_wl = nullptr;
    edge_res = nullptr;
    includes_pin_caps = nullptr;
    root_dist = nullptr;
    cnts = nullptr;
    node_order = nullptr;
    parent_node = nullptr;
    res_parent = nullptr;
    node_cap = nullptr;
    node_delay = nullptr;
    y1 = nullptr;
    y2 = nullptr;
    y3 = nullptr;
    down_cap = nullptr;
    elmore_delay = nullptr;
    dmpCudaMallocChecked(&pin_flags, num_pins, "pin_flags");
    std::vector<uint8_t> host_pin_flags(num_pins, 0);
    for (int pin_id : timer->gtdb.primary_inputs) {
        if (pin_id >= 0 && pin_id < num_pins) {
            host_pin_flags[pin_id] |= DMP_PIN_PRIMARY_INPUT;
        }
    }
    for (int pin_id = 0; pin_id < num_pins; ++pin_id) {
        if (pin_id < static_cast<int>(timer->gtdb.pin_is_clk.size()) &&
            timer->gtdb.pin_is_clk[pin_id]) {
            host_pin_flags[pin_id] |= DMP_PIN_CLK;
        }
        if (pin_id < static_cast<int>(timer->gtdb.pin_is_ideal_clk.size()) &&
            timer->gtdb.pin_is_ideal_clk[pin_id]) {
            host_pin_flags[pin_id] |= DMP_PIN_IDEAL_CLK;
        }
    }
    cudaMemcpy(pin_flags,
               host_pin_flags.data(),
               sizeof(uint8_t) * num_pins,
               cudaMemcpyHostToDevice);
    const std::vector<float>& host_clock_periods = timer->gtdb.clock_periods;
    std::vector<uint8_t> host_test_clock_ids(num_tests, std::numeric_limits<uint8_t>::max());
    const int host_test_clock_id_count = static_cast<int>(timer->gtdb.test_clock_ids.size());
    for (int test_id = 0; test_id < num_tests; ++test_id) {
        if (test_id < host_test_clock_id_count) {
            host_test_clock_ids[test_id] = timer->gtdb.test_clock_ids[test_id];
        }
    }
    clock_count = static_cast<int>(host_clock_periods.size());
    dmpCudaMallocChecked(&clock_periods, host_clock_periods.size(), "clock_periods");
    if (!host_clock_periods.empty()) {
        cudaMemcpy(clock_periods,
                   host_clock_periods.data(),
                   sizeof(float) * host_clock_periods.size(),
                   cudaMemcpyHostToDevice);
    }
    dmpCudaMallocChecked(&test_clock_ids, host_test_clock_ids.size(), "test_clock_ids");
    if (!host_test_clock_ids.empty()) {
        cudaMemcpy(test_clock_ids,
                   host_test_clock_ids.data(),
                   sizeof(uint8_t) * host_test_clock_ids.size(),
                   cudaMemcpyHostToDevice);
    }
    if (dmpDeferTimingAlloc()) {
        if (dmpInitSummaryEnabled()) {
            std::fprintf(stderr, "[DMP INIT] timing scratch allocation deferred until after RC propagation\n");
        }
        return;
    }
    allocate_timing_scratch();
    if (!dmpShouldUploadDeviceNames(debug_on)) {
        return;
    }
    // host-side
    const char** host_pin_ptrs = new const char*[num_pins];
    for (int i = 0; i < num_pins; i++) {
        size_t len = (timer->gtdb).pin_names[i].length() + 1;
        const char* dev_str;
        cudaMalloc(&dev_str, len);
        cudaMemcpy((void *)dev_str, (timer->gtdb).pin_names[i].c_str(), len, cudaMemcpyHostToDevice);
        host_pin_ptrs[i] = dev_str; // 保存 device 地址
    }
    cudaMalloc(&pin_names, sizeof(const char*) * num_pins);
    cudaMemcpy(pin_names, host_pin_ptrs, sizeof(const char*) * num_pins, cudaMemcpyHostToDevice);
    delete[] host_pin_ptrs;
    // host-side
    const char** host_net_ptrs = new const char*[num_nets];
    for (int i = 0; i < num_nets; i++) {
        size_t len = (timer->gtdb).net_names[i].length() + 1;
        const char* dev_str;
        cudaMalloc(&dev_str, len);
        cudaMemcpy((void *)dev_str, (timer->gtdb).net_names[i].c_str(), len, cudaMemcpyHostToDevice);
        host_net_ptrs[i] = dev_str; // 保存 device 地址
    }
    cudaMalloc(&net_names, sizeof(const char*) * num_nets);
    cudaMemcpy(net_names, host_net_ptrs, sizeof(const char*) * num_nets, cudaMemcpyHostToDevice);
    delete[] host_net_ptrs;
}

void DmpModel::allocate_timing_scratch()
{
    if (pin_at_winner != nullptr) {
        return;
    }
    if (dmpInitSummaryEnabled()) {
        std::fprintf(stderr,
                     "[DMP INIT] allocating timing scratch pin_slots=%d slot_capacity=%d work_slot_capacity=%d\n",
                     dmp_pin_slot_count,
                     dmp_slot_capacity,
                     dmp_work_slot_capacity);
    }
    dmpCudaMallocChecked(&pin_at_winner, dmp_pin_slot_count, "pin_at_winner");
    dmpCudaMemsetChecked(pin_at_winner, 0, dmp_pin_slot_count, "pin_at_winner");
    if (dmpInitMemProfileEnabled()) {
        dmpPrintCudaMemInfo("after_timing_scratch_alloc");
    }
}

void DmpModel::release_rc_transient()
{
    cudaFree(edge_from);
    cudaFree(edge_to);
    cudaFree(flat_net2node_start_map);
    cudaFree(flat_net2edge_start_map);
    cudaFree(node2pin_map);
    cudaFree(edge_wl);
    cudaFree(edge_res);
    cudaFree(node_cap);
    cudaFree(includes_pin_caps);
    cudaFree(root_dist);
    cudaFree(cnts);
    cudaFree(node_order);
    cudaFree(parent_node);
    cudaFree(res_parent);
    cudaFree(node_delay);
    cudaFree(y1);
    cudaFree(y2);
    cudaFree(y3);
    cudaFree(down_cap);
    edge_from = nullptr;
    edge_to = nullptr;
    flat_net2node_start_map = nullptr;
    flat_net2edge_start_map = nullptr;
    node2pin_map = nullptr;
    edge_wl = nullptr;
    edge_res = nullptr;
    node_cap = nullptr;
    includes_pin_caps = nullptr;
    root_dist = nullptr;
    cnts = nullptr;
    node_order = nullptr;
    parent_node = nullptr;
    res_parent = nullptr;
    node_delay = nullptr;
    y1 = nullptr;
    y2 = nullptr;
    y3 = nullptr;
    down_cap = nullptr;
    if (dmpInitMemProfileEnabled()) {
        dmpPrintCudaMemInfo("after_rc_transient_free");
    }
}

void DmpModel::release_after_timing()
{
    cudaFree(pin_at_winner);
    cudaFree(pin_flags);
    cudaFree(test_clock_ids);
    cudaFree(clock_periods);
    cudaFree(r_pi);
    cudaFree(elmore_delay);
    pin_at_winner = nullptr;
    pin_flags = nullptr;
    test_clock_ids = nullptr;
    clock_periods = nullptr;
    clock_count = 0;
    r_pi = nullptr;
    elmore_delay = nullptr;
    if (dmpInitMemProfileEnabled()) {
        dmpPrintCudaMemInfo("after_dmp_release_after_timing");
    }
}

void dmp_prepare_timing_after_rc(DmpModel* h_dmp_db, DmpModel* dmp_db)
{
    if (h_dmp_db == nullptr || dmp_db == nullptr) {
        return;
    }
    DmpModel device_state;
    gpuErrchk(cudaMemcpy(&device_state, dmp_db, sizeof(DmpModel), cudaMemcpyDeviceToHost));
    device_state.owns_allocations = false;
    const bool owns_allocations = h_dmp_db->owns_allocations;
    std::memcpy(h_dmp_db, &device_state, sizeof(DmpModel));
    h_dmp_db->owns_allocations = owns_allocations;
    h_dmp_db->release_rc_transient();
    h_dmp_db->allocate_timing_scratch();
    gpuErrchk(cudaMemcpy(dmp_db, h_dmp_db, sizeof(DmpModel), cudaMemcpyHostToDevice));
}

void dmp_release_after_timing(DmpModel* h_dmp_db, DmpModel* dmp_db)
{
    if (!dmpDeferTimingAlloc() || h_dmp_db == nullptr || dmp_db == nullptr) {
        return;
    }
    h_dmp_db->release_after_timing();
    gpuErrchk(cudaMemcpy(dmp_db, h_dmp_db, sizeof(DmpModel), cudaMemcpyHostToDevice));
}
__host__ DmpModel::~DmpModel(){
    if(owns_allocations){
        if (C1) cudaFree(C1);
        if (C2) cudaFree(C2);
        if (r_pi) cudaFree(r_pi);
        if (elmore_delay) cudaFree(elmore_delay);
        cudaFree(pin_at_winner);
        cudaFree(pin_flags);
        cudaFree(test_clock_ids);
        cudaFree(clock_periods);
            }
}

void GPUTimer::release_dmp_timing_scratch_for_power()
{
    dmp_release_after_timing(h_dmp_db, dmp_db);
}

void GPUTimer::initialize_dmp_model(){
    // cudaMemcpy(h_dmp_rc_, dmp_rc_, sizeof(dmp_rc), cudaMemcpyDeviceToHost);
    h_dmp_db = new DmpModel(this);
    cudaMalloc(&dmp_db, sizeof(DmpModel));
    cudaMemcpy(dmp_db, h_dmp_db, sizeof(DmpModel), cudaMemcpyHostToDevice);

}
// __device__ void DmpModel::compute_pi_model(int net_id, int el_rf){
//     int start_id = flat_net2pin_start_map[net_id];
//     int end_id = flat_net2pin_start_map[net_id + 1];
//     int root = flat_net2pin_map[start_id];
//     double y1 = pinLoad[root * NUM_ATTR + el_rf];
//     double y2 = 0;
//     double y3 = 0;
//     for(int i = start_id + 1; i < end_id; i++){
//         int pin_id = flat_net2pin_map[i];
//         double cap = pinLoad[pin_id * NUM_ATTR + el_rf];
//         double res = pinRootRes[pin_id * NUM_ATTR + el_rf];
//         double y2_ = cap * cap * res;
//         y2 += -y2_;
//         y3 += y2_ * res * cap;
//     }
//     if(y3 <= 1e-10){
//         r_pi[net_id * NUM_ATTR + el_rf] = 0;
//         C1[net_id * NUM_ATTR + el_rf] = 0;
//         C2[net_id * NUM_ATTR + el_rf] = 0;
//     }
//     else{
//         C1[net_id * NUM_ATTR + el_rf] = static_cast<float>(y2 * y2 / y3); // 远端电容
//         C2[net_id * NUM_ATTR + el_rf] = static_cast<float>(y1 - y2 * y2 / y3); // 近端电容
//         if (C2[net_id * NUM_ATTR + el_rf] < 0.0)
//           C2[net_id * NUM_ATTR + el_rf] = 0.0;
//         r_pi[net_id * NUM_ATTR + el_rf] = static_cast<float>(-y3 * y3 / (y2 * y2 * y2));
//     }
//     // printf("net_id:%d el_rf:%d root:%d rpi:%.4f C1:%.4f C2:%.4f rootLoad:%.4f\n", net_id, el_rf, root, r_pi[net_id*NUM_ATTR+el_rf], C1[net_id*NUM_ATTR+el_rf], C2[net_id*NUM_ATTR+el_rf], pinLoad[root * NUM_ATTR+el_rf]);

// }

// __global__ void compute_pi_model_kernel(DmpModel *dmp_db){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int net_id = idx >> 2;
//     int el_rf = idx & (NUM_ATTR - 1);
//     if(net_id < dmp_db -> num_nets){
//         dmp_db -> compute_pi_model(net_id, el_rf);
//     }
// }
// void compute_pi_model_cuda(DmpModel *dmp_db, int num_nets){
//     compute_pi_model_kernel<<<BLOCK_NUMBER(num_nets * NUM_ATTR), BLOCK_SIZE>>>(dmp_db);
// }


void print_pinLoad_cuda(DmpModel* dmp_db, vector<int> level_list_end_cpu, vector<std::string> pin_names){
    int total_num_pins = level_list_end_cpu.back();
    assert(total_num_pins == (int)pin_names.size());
    float* pinLoad_host = new float[total_num_pins * NUM_ATTR];
    float* C1_host = new float[total_num_pins * NUM_ATTR];
    float* C2_host = new float[total_num_pins * NUM_ATTR];
    float* pinslew_host = new float[total_num_pins * NUM_ATTR];
    float* pinAt_host = new float[total_num_pins * NUM_ATTR];
    float* pinRat_host = new float[total_num_pins * NUM_ATTR];
    int* level_pin_list_host = new int[total_num_pins];
    int* backward_arc_list_end_host = new int[total_num_pins + 1];
    DmpModel* dmp_db_host = new DmpModel();
    cudaMemcpy(dmp_db_host, dmp_db, sizeof(DmpModel), cudaMemcpyDeviceToHost);
    dmp_db_host->owns_allocations = false;
    cudaMemcpy(backward_arc_list_end_host, dmp_db_host->pin_backward_arc_list_end, sizeof(int) * (total_num_pins + 1), cudaMemcpyDeviceToHost);
    int num_arcs = backward_arc_list_end_host[total_num_pins];
    printf("num_arcs = %d\n", num_arcs);
    int* backward_arc_list_host = new int[num_arcs];
    int* timing_arc_from_pin_id_host = new int[num_arcs];
    float* arcDelay_host = new float[num_arcs * 2 * NUM_ATTR];
    cudaMemcpy(timing_arc_from_pin_id_host, dmp_db_host->timing_arc_from_pin_id, sizeof(int) * num_arcs, cudaMemcpyDeviceToHost);
    cudaMemcpy(backward_arc_list_host, dmp_db_host->pin_backward_arc_list, sizeof(int) * num_arcs, cudaMemcpyDeviceToHost);
    cudaMemcpy(arcDelay_host, dmp_db_host->arcDelay, sizeof(float) * num_arcs * 2 * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(level_pin_list_host, dmp_db_host->level_list, sizeof(int) * total_num_pins, cudaMemcpyDeviceToHost);
    cudaMemcpy(pinLoad_host, dmp_db_host->timer_ceff, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(C1_host, dmp_db_host->C1, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(C2_host, dmp_db_host->C2, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(pinslew_host, dmp_db_host->pinSlew, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(pinAt_host, dmp_db_host->pinAt, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    cudaMemcpy(pinRat_host, dmp_db_host->pinRat, sizeof(float) * total_num_pins * NUM_ATTR, cudaMemcpyDeviceToHost);
    for (int i = 0; i < level_list_end_cpu.size() - 1; i++) {
        int num_pins_level = level_list_end_cpu[i + 1] - level_list_end_cpu[i];
        index_type level_start_offset = level_list_end_cpu[i];
        printf("==== level %d ======= %d \n", i, num_pins_level);
        for(int j = 0; j < num_pins_level; j++){
            for(int attr = 0; attr < NUM_ATTR; attr++){
                int pin = level_pin_list_host[level_start_offset + j];
                if(!isnan(pinLoad_host[pin * NUM_ATTR + attr])){
                    printf("pin %s attr %d pin_at = %E pin_rat = %E load = %E C1 = %E C2 = %E pinslew = %E\n", pin_names[pin].c_str(), attr, pinAt_host[pin * NUM_ATTR + attr], pinRat_host[pin * NUM_ATTR + attr], pinLoad_host[pin * NUM_ATTR + attr], C1_host[pin * NUM_ATTR + attr], C2_host[pin * NUM_ATTR + attr], pinslew_host[pin * NUM_ATTR + attr]);
                }
                if(j > 0){
                    for(int k = backward_arc_list_end_host[pin] ; k < backward_arc_list_end_host[pin + 1]; k++){
                        int arc_id = backward_arc_list_host[k];
                        int from_pin = timing_arc_from_pin_id_host[arc_id];
                        assert(k < num_arcs);
                        assert(arc_id < num_arcs);
                        assert(from_pin < total_num_pins);
                        for(int frf = 0; frf < 2; frf++){
                            int arc_idx = arc_id * 2 * NUM_ATTR + ((attr & (1 << 1)) << 1) + (frf << 1) + (attr & 1);
                            assert(arc_idx < num_arcs * 2 * NUM_ATTR);
                            float delay = arcDelay_host[arc_idx];
                            printf("    from pin %s %c  arc_id = %d delay = %E\n", pin_names[from_pin].c_str(), frf == 0 ? '^' : 'v',   arc_id, delay);
                        }
                    }
                }
            }
        }
        gpuErrchk( cudaDeviceSynchronize() );
        // if(i == 2)break;
    }

}

}
