#include "gputimer/core/gputiming.h"

#include "common/common.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace gt {

void GPULutAllocator::AllocateBatch(vector<TimingArc*> timings) {
    auto check_lut = [&](Lut* lut) {
        if (!lut) return;
        if (lut->set_) {
            x_size += lut->indices1.size();
            y_size += lut->indices2.size();
            table_size += lut->table.size();
        }
    };
    num_timings = timings.size();
    is_rising_edge_triggered = new bool[num_timings];
    is_falling_edge_triggered = new bool[num_timings];
    is_constraint = new bool[num_timings];
    is_latch_clock_arc = new bool[num_timings];
    timing_sense = new int[num_timings];
    num_timings = 0;
    for (auto timing_ptr : timings) {
        auto& timing = *timing_ptr;
        check_lut(timing.cell_delay_[0]);
        check_lut(timing.cell_delay_[1]);
        check_lut(timing.transition_[0]);
        check_lut(timing.transition_[1]);
        check_lut(timing.constraint_[0]);
        check_lut(timing.constraint_[1]);
        is_rising_edge_triggered[num_timings] = timing.is_rising_edge_triggered();
        is_falling_edge_triggered[num_timings] = timing.is_falling_edge_triggered();
        is_constraint[num_timings] = timing.is_constraint();
        is_latch_clock_arc[num_timings] = false;
        if (!timing.is_constraint() &&
            (timing.is_rising_edge_triggered() || timing.is_falling_edge_triggered())) {
            is_latch_clock_arc[num_timings] =
                timing.related_port_name_ == "G" || timing.related_port_name_ == "GN";
        }
        if (timing.timing_sense_ != TimingSense::unknown)
            timing_sense[num_timings] = static_cast<int>(timing.timing_sense_);
        else
            timing_sense[num_timings] = -1;
        num_timings++;
    }

    num_luts = num_luts_in_timing * timings.size();
    x_array = new float[x_size];
    y_array = new float[y_size];
    table_array = new float[table_size];
    num_x = new int[num_luts];
    num_y = new int[num_luts];
    num_table = new int[num_luts];
    x_offset = new size_t[num_luts + 1];
    y_offset = new size_t[num_luts + 1];
    table_offset = new size_t[num_luts + 1];
    lut_template_var = new int[num_luts * 2];
    allocated = new bool[num_luts];
    x_offset[0] = 0;
    y_offset[0] = 0;
    table_offset[0] = 0;

    num_luts = 0;
    auto insert_lut = [&](Lut* lut) {
        if (lut->set_) {
            num_x[num_luts] = lut->indices1.size();
            num_y[num_luts] = lut->indices2.size();
            num_table[num_luts] = lut->table.size();
            x_offset[num_luts + 1] = x_offset[num_luts] + num_x[num_luts];
            y_offset[num_luts + 1] = y_offset[num_luts] + num_y[num_luts];
            table_offset[num_luts + 1] = table_offset[num_luts] + num_table[num_luts];
            memcpy(x_array + x_offset[num_luts], lut->indices1.data(), lut->indices1.size() * sizeof(float));
            memcpy(y_array + y_offset[num_luts], lut->indices2.data(), lut->indices2.size() * sizeof(float));
            memcpy(table_array + table_offset[num_luts], lut->table.data(), lut->table.size() * sizeof(float));

            if (lut->lut_template) {
                if (lut->lut_template->variable1)
                    lut_template_var[num_luts * 2] = static_cast<int>(lut->lut_template->variable1.value());
                else
                    lut_template_var[num_luts * 2] = -1;
                if (lut->lut_template->variable2)
                    lut_template_var[num_luts * 2 + 1] = static_cast<int>(lut->lut_template->variable2.value());
                else
                    lut_template_var[num_luts * 2 + 1] = -1;
            } else {
                lut_template_var[num_luts * 2] = -1;
                lut_template_var[num_luts * 2 + 1] = -1;
            }
            allocated[num_luts] = true;
        } else {
            num_x[num_luts] = 0;
            num_y[num_luts] = 0;
            num_table[num_luts] = 0;
            x_offset[num_luts + 1] = x_offset[num_luts];
            y_offset[num_luts + 1] = y_offset[num_luts];
            table_offset[num_luts + 1] = table_offset[num_luts];
            lut_template_var[num_luts * 2] = -1;
            lut_template_var[num_luts * 2 + 1] = -1;
            allocated[num_luts] = false;
        }
        num_luts++;
    };
    for (auto timing_ptr : timings) {
        auto& timing = *timing_ptr;
        insert_lut(timing.cell_delay_[0]);
        insert_lut(timing.cell_delay_[1]);
        insert_lut(timing.transition_[0]);
        insert_lut(timing.transition_[1]);
        insert_lut(timing.constraint_[0]);
        insert_lut(timing.constraint_[1]);
    }
}

void GPULutAllocator::CopyToGPU() {
    cudaMalloc(&d_x_array, x_size * sizeof(float));
    cudaMalloc(&d_y_array, y_size * sizeof(float));
    cudaMalloc(&d_table_array, table_size * sizeof(float));
    cudaMalloc(&d_num_x, num_luts * sizeof(int));
    cudaMalloc(&d_num_y, num_luts * sizeof(int));
    cudaMalloc(&d_num_table, num_luts * sizeof(int));
    cudaMalloc(&d_x_offset, (num_luts + 1) * sizeof(size_t));
    cudaMalloc(&d_y_offset, (num_luts + 1) * sizeof(size_t));
    cudaMalloc(&d_table_offset, (num_luts + 1) * sizeof(size_t));
    const int allocated_word_count = std::max(1, (num_luts + 31) / 32);
    cudaMalloc(&d_allocated_bits, allocated_word_count * sizeof(uint32_t));
    cudaMalloc(&d_timing_flags, num_timings * sizeof(uint8_t));
    cudaMalloc(&d_timing_sense, num_timings * sizeof(int8_t));
    cudaMalloc(&d_lut_template_var, 2 * num_luts * sizeof(int8_t));

    cudaMemcpy(d_x_array, x_array, x_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_array, y_array, y_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table_array, table_array, table_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_x, num_x, num_luts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_y, num_y, num_luts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_table, num_table, num_luts * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_offset, x_offset, (num_luts + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_offset, y_offset, (num_luts + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table_offset, table_offset, (num_luts + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    std::vector<uint32_t> allocated_bits(allocated_word_count, 0);
    for (int i = 0; i < num_luts; ++i) {
        if (allocated[i]) allocated_bits[i >> 5] |= (1u << (i & 31));
    }
    cudaMemcpy(d_allocated_bits, allocated_bits.data(), allocated_word_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    std::vector<uint8_t> timing_flags(num_timings, 0);
    for (int i = 0; i < num_timings; ++i) {
        if (is_rising_edge_triggered[i]) timing_flags[i] |= GT_TIMING_FLAG_RISING_EDGE;
        if (is_falling_edge_triggered[i]) timing_flags[i] |= GT_TIMING_FLAG_FALLING_EDGE;
        if (is_constraint[i]) timing_flags[i] |= GT_TIMING_FLAG_CONSTRAINT;
        if (is_latch_clock_arc[i]) timing_flags[i] |= GT_TIMING_FLAG_LATCH_CLOCK_ARC;
    }
    cudaMemcpy(d_timing_flags, timing_flags.data(), num_timings * sizeof(uint8_t), cudaMemcpyHostToDevice);
    std::vector<int8_t> timing_sense_i8(num_timings, -1);
    for (int i = 0; i < num_timings; ++i)
        timing_sense_i8[i] = static_cast<int8_t>(timing_sense[i]);
    std::vector<int8_t> lut_template_var_i8(2 * num_luts, -1);
    for (int i = 0; i < 2 * num_luts; ++i)
        lut_template_var_i8[i] = static_cast<int8_t>(lut_template_var[i]);
    cudaMemcpy(d_timing_sense, timing_sense_i8.data(), num_timings * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut_template_var, lut_template_var_i8.data(), 2 * num_luts * sizeof(int8_t), cudaMemcpyHostToDevice);
}

void GPULutAllocator::CopyToGPU(GPULutAllocator* d_gpuluts) {
    cudaMemcpy(&(d_gpuluts->d_num_x), &d_num_x, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_num_y), &d_num_y, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_num_table), &d_num_table, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_x_array), &d_x_array, sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_y_array), &d_y_array, sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_table_array), &d_table_array, sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_x_offset), &d_x_offset, sizeof(size_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_y_offset), &d_y_offset, sizeof(size_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_table_offset), &d_table_offset, sizeof(size_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_allocated_bits), &d_allocated_bits, sizeof(uint32_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_timing_flags), &d_timing_flags, sizeof(uint8_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_timing_sense), &d_timing_sense, sizeof(int8_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpuluts->d_lut_template_var), &d_lut_template_var, sizeof(int8_t*), cudaMemcpyHostToDevice);
}

void GPULutAllocator::freeMem() {
    if (allocated) {
        logger.info("destruct gputiming");
        delete[] num_x;
        delete[] num_y;
        delete[] num_table;
        delete[] x_array;
        delete[] y_array;
        delete[] table_array;
        delete[] x_offset;
        delete[] y_offset;
        delete[] table_offset;
        delete[] allocated;
        delete[] is_rising_edge_triggered;
        delete[] is_falling_edge_triggered;
        delete[] is_constraint;
        delete[] is_latch_clock_arc;
        delete[] timing_sense;
        cudaFree(d_num_x);
        cudaFree(d_num_y);
        cudaFree(d_num_table);
        cudaFree(d_x_array);
        cudaFree(d_y_array);
        cudaFree(d_table_array);
        cudaFree(d_x_offset);
        cudaFree(d_y_offset);
        cudaFree(d_table_offset);
        cudaFree(d_allocated_bits);
        cudaFree(d_timing_flags);
        cudaFree(d_timing_sense);
        cudaFree(d_lut_template_var);
    }
}

GPULutAllocator::~GPULutAllocator() { freeMem(); }

void GPUPowerLutAllocator::AllocateBatch(const vector<InternalPower*>& internal_powers) {
    auto check_lut = [&](Lut* lut) {
        if (!lut) return;
        if (lut->set_) {
            x_size += lut->indices1.size();
            y_size += lut->indices2.size();
            table_size += lut->table.size();
        }
    };
    num_internal_powers = internal_powers.size();
    for (auto* ip : internal_powers) {
        if (!ip) continue;
        check_lut(ip->power_[0]);
        check_lut(ip->power_[1]);
    }
    num_luts = num_luts_in_internal_power * num_internal_powers;
    x_array = new float[std::max(1, x_size)];
    y_array = new float[std::max(1, y_size)];
    table_array = new float[std::max(1, table_size)];
    num_x = new int[std::max(1, num_luts)];
    num_y = new int[std::max(1, num_luts)];
    num_table = new int[std::max(1, num_luts)];
    x_offset = new size_t[std::max(1, num_luts + 1)];
    y_offset = new size_t[std::max(1, num_luts + 1)];
    table_offset = new size_t[std::max(1, num_luts + 1)];
    allocated = new bool[std::max(1, num_luts)];
    lut_template_var = new int[std::max(1, num_luts * 2)];
    x_offset[0] = y_offset[0] = table_offset[0] = 0;
    int lut_idx = 0;
    auto insert_lut = [&](Lut* lut) {
        if (lut && lut->set_) {
            num_x[lut_idx] = lut->indices1.size();
            num_y[lut_idx] = lut->indices2.size();
            num_table[lut_idx] = lut->table.size();
            x_offset[lut_idx + 1] = x_offset[lut_idx] + num_x[lut_idx];
            y_offset[lut_idx + 1] = y_offset[lut_idx] + num_y[lut_idx];
            table_offset[lut_idx + 1] = table_offset[lut_idx] + num_table[lut_idx];
            memcpy(x_array + x_offset[lut_idx], lut->indices1.data(), lut->indices1.size() * sizeof(float));
            memcpy(y_array + y_offset[lut_idx], lut->indices2.data(), lut->indices2.size() * sizeof(float));
            memcpy(table_array + table_offset[lut_idx], lut->table.data(), lut->table.size() * sizeof(float));
            if (lut->lut_template) {
                lut_template_var[lut_idx * 2] =
                    lut->lut_template->variable1 ? static_cast<int>(lut->lut_template->variable1.value()) : -1;
                lut_template_var[lut_idx * 2 + 1] =
                    lut->lut_template->variable2 ? static_cast<int>(lut->lut_template->variable2.value()) : -1;
            } else {
                lut_template_var[lut_idx * 2] = -1;
                lut_template_var[lut_idx * 2 + 1] = -1;
            }
            allocated[lut_idx] = true;
        } else {
            num_x[lut_idx] = num_y[lut_idx] = num_table[lut_idx] = 0;
            x_offset[lut_idx + 1] = x_offset[lut_idx];
            y_offset[lut_idx + 1] = y_offset[lut_idx];
            table_offset[lut_idx + 1] = table_offset[lut_idx];
            lut_template_var[lut_idx * 2] = -1;
            lut_template_var[lut_idx * 2 + 1] = -1;
            allocated[lut_idx] = false;
        }
        lut_idx++;
    };
    for (auto* ip : internal_powers) {
        insert_lut(ip ? ip->power_[0] : nullptr);
        insert_lut(ip ? ip->power_[1] : nullptr);
    }
}

void GPUPowerLutAllocator::CopyToGPU() {
    cudaMalloc(&d_x_array, std::max(1, x_size) * sizeof(float));
    cudaMalloc(&d_y_array, std::max(1, y_size) * sizeof(float));
    cudaMalloc(&d_table_array, std::max(1, table_size) * sizeof(float));
    cudaMalloc(&d_num_x, std::max(1, num_luts) * sizeof(int));
    cudaMalloc(&d_num_y, std::max(1, num_luts) * sizeof(int));
    cudaMalloc(&d_num_table, std::max(1, num_luts) * sizeof(int));
    cudaMalloc(&d_x_offset, std::max(1, num_luts + 1) * sizeof(size_t));
    cudaMalloc(&d_y_offset, std::max(1, num_luts + 1) * sizeof(size_t));
    cudaMalloc(&d_table_offset, std::max(1, num_luts + 1) * sizeof(size_t));
    const int allocated_word_count = std::max(1, (num_luts + 31) / 32);
    cudaMalloc(&d_allocated_bits, allocated_word_count * sizeof(uint32_t));
    cudaMalloc(&d_lut_template_var, std::max(1, num_luts * 2) * sizeof(int8_t));
    if (x_size) cudaMemcpy(d_x_array, x_array, x_size * sizeof(float), cudaMemcpyHostToDevice);
    if (y_size) cudaMemcpy(d_y_array, y_array, y_size * sizeof(float), cudaMemcpyHostToDevice);
    if (table_size) cudaMemcpy(d_table_array, table_array, table_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_x, num_x, std::max(1, num_luts) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_y, num_y, std::max(1, num_luts) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_table, num_table, std::max(1, num_luts) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_offset, x_offset, std::max(1, num_luts + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_offset, y_offset, std::max(1, num_luts + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_table_offset, table_offset, std::max(1, num_luts + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    std::vector<uint32_t> allocated_bits(allocated_word_count, 0);
    for (int i = 0; i < num_luts; ++i) {
        if (allocated[i]) allocated_bits[i >> 5] |= (1u << (i & 31));
    }
    cudaMemcpy(d_allocated_bits, allocated_bits.data(), allocated_word_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    const int power_lut_template_count = std::max(1, num_luts * 2);
    std::vector<int8_t> lut_template_var_i8(power_lut_template_count, -1);
    for (int i = 0; i < num_luts * 2; ++i)
        lut_template_var_i8[i] = static_cast<int8_t>(lut_template_var[i]);
    cudaMemcpy(d_lut_template_var, lut_template_var_i8.data(), power_lut_template_count * sizeof(int8_t), cudaMemcpyHostToDevice);
}

void GPUPowerLutAllocator::CopyToGPU(GPUPowerLutAllocator* d_gpu_luts) {
    cudaMemcpy(&(d_gpu_luts->d_num_x), &d_num_x, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpu_luts->d_num_y), &d_num_y, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpu_luts->d_num_table), &d_num_table, sizeof(int*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpu_luts->d_x_array), &d_x_array, sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpu_luts->d_y_array), &d_y_array, sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpu_luts->d_table_array), &d_table_array, sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpu_luts->d_x_offset), &d_x_offset, sizeof(size_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpu_luts->d_y_offset), &d_y_offset, sizeof(size_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpu_luts->d_table_offset), &d_table_offset, sizeof(size_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpu_luts->d_allocated_bits), &d_allocated_bits, sizeof(uint32_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_gpu_luts->d_lut_template_var), &d_lut_template_var, sizeof(int8_t*), cudaMemcpyHostToDevice);
}

void GPUPowerLutAllocator::freeMem() {
    delete[] num_x; delete[] num_y; delete[] num_table;
    delete[] x_array; delete[] y_array; delete[] table_array;
    delete[] x_offset; delete[] y_offset; delete[] table_offset;
    delete[] allocated; delete[] lut_template_var;
    if (d_num_x) cudaFree(d_num_x); if (d_num_y) cudaFree(d_num_y); if (d_num_table) cudaFree(d_num_table);
    if (d_x_array) cudaFree(d_x_array); if (d_y_array) cudaFree(d_y_array); if (d_table_array) cudaFree(d_table_array);
    if (d_x_offset) cudaFree(d_x_offset); if (d_y_offset) cudaFree(d_y_offset); if (d_table_offset) cudaFree(d_table_offset);
    if (d_allocated_bits) cudaFree(d_allocated_bits); if (d_lut_template_var) cudaFree(d_lut_template_var);
    num_x = num_y = num_table = nullptr;
    x_array = y_array = table_array = nullptr;
    x_offset = y_offset = table_offset = nullptr;
    allocated = nullptr; lut_template_var = nullptr;
    d_num_x = d_num_y = d_num_table = nullptr;
    d_x_array = d_y_array = d_table_array = nullptr;
    d_x_offset = d_y_offset = d_table_offset = nullptr;
    d_allocated_bits = nullptr; d_lut_template_var = nullptr;
}

GPUPowerLutAllocator::~GPUPowerLutAllocator() { freeMem(); }

}  // namespace gt
