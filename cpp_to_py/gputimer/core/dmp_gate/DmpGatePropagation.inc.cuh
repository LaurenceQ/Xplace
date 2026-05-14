__device__ __forceinline__ bool dmpIsIdealClockTimingArc(const DmpModel* dmp_db,
                                                         int timing_id,
                                                         int from_pin_id) {
    return dmp_db->ideal_clock &&
           dmp_db->pin_is_clk != nullptr &&
           dmp_db->pin_is_clk[from_pin_id] &&
           timing_id >= 0;
}

__device__ __forceinline__ float dmpClockPeriodForPin(const DmpModel* dmp_db,
                                                      int pin_id) {
    if (dmp_db->pin_clock_periods != nullptr && pin_id >= 0) {
        const float period = dmp_db->pin_clock_periods[pin_id];
        if (isfinite(period) && period > 0.0f) {
            return period;
        }
    }
    return dmp_db->clock_period;
}

__device__ __forceinline__ float dmpClockPeriodForTest(const DmpModel* dmp_db,
                                                       int test_id) {
    if (dmp_db->test_clock_periods != nullptr && test_id >= 0) {
        const float period = dmp_db->test_clock_periods[test_id];
        if (isfinite(period) && period > 0.0f) {
            return period;
        }
    }
    return dmp_db->clock_period;
}

__device__ __forceinline__ float dmpIdealClockEdgeTime(const DmpModel* dmp_db,
                                                       int timing_id,
                                                       int from_pin_id) {
    const bool latch_clock_arc = timing_id >= 0 &&
                                 dmp_db->d_allocator->d_is_latch_clock_arc != nullptr &&
                                 dmp_db->d_allocator->d_is_latch_clock_arc[timing_id];
    const bool falling_triggered = timing_id >= 0 &&
                                   dmp_db->d_allocator->d_is_falling_edge_triggered[timing_id] &&
                                   !dmp_db->d_allocator->d_is_rising_edge_triggered[timing_id];
    const bool use_fall_edge = latch_clock_arc ? !falling_triggered : falling_triggered;
    if (use_fall_edge) {
        if (dmp_db->pin_clock_fall_edges != nullptr && from_pin_id >= 0) {
            const float edge = dmp_db->pin_clock_fall_edges[from_pin_id];
            if (isfinite(edge)) {
                return edge;
            }
        }
        return nanf("");
    }
    if (dmp_db->pin_clock_rise_edges != nullptr && from_pin_id >= 0) {
        const float edge = dmp_db->pin_clock_rise_edges[from_pin_id];
        if (isfinite(edge)) {
            return edge;
        }
    }
    return nanf("");
}

__device__ __forceinline__ float dmpIdealClockSlew(const DmpModel* dmp_db,
                                                   int from_pin_id,
                                                   int attr) {
    if (dmp_db->pin_clock_slews != nullptr && from_pin_id >= 0 && attr >= 0) {
        const float slew = dmp_db->pin_clock_slews[from_pin_id * NUM_ATTR + attr];
        if (isfinite(slew)) {
            return slew;
        }
    }
    return 0.0f;
}

__device__ void DmpModel::propagateTest(){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    int arc_id = arc_ids[idx];
    int test_id = arc_id2test_id[arc_id];
    if(test_id == -1)return ;
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    if (i < NUM_ATTR) {
        const int el = i >> 1;
        const int rf = i & 1;
        if ((timing_arc_id_map[arc_id * 2 + el] == -1) || (isnan(pinSlew[to_pin_id * NUM_ATTR + i]))) return;
        int fel = el ^ 1;  // clock -> data. clock late -> data early (hold)
        int timing_id = timing_arc_id_map[arc_id * 2 + el];
        int frf = d_allocator->d_is_rising_edge_triggered[timing_id] ? 0 : 1;
        if (frf && !d_allocator->d_is_falling_edge_triggered[timing_id]) {
            return;
        }
        const int fel_rf = (fel << 1) + frf;
        const bool ideal_clock_arc = dmpIsIdealClockTimingArc(this, timing_id, from_pin_id);
        if (!ideal_clock_arc &&
            (isnan(pinAt[from_pin_id * NUM_ATTR + fel_rf]) ||
             isnan(pinSlew[from_pin_id * NUM_ATTR + fel_rf]))) return;

        float related_at = ideal_clock_arc
                               ? dmpIdealClockEdgeTime(this, timing_id, from_pin_id)
                               : pinAt[from_pin_id * NUM_ATTR + fel_rf];
        if (ideal_clock_arc && isnan(related_at)) {
            related_at = pinAt[from_pin_id * NUM_ATTR + fel_rf];
        }
        if (el == 0) {
            testRelatedAT[test_id * NUM_ATTR + i] = related_at;
        } else {
            const float test_period = dmpClockPeriodForTest(this, test_id);
            testRelatedAT[test_id * NUM_ATTR + i] = related_at + (frf ? 0.5f * test_period : test_period);  // setup is checked at next cycle (first cycle for triggering 1st FF)
        }

        float sr = pinSlew[from_pin_id * NUM_ATTR + fel_rf];
        if (ideal_clock_arc) {
            sr = dmpIdealClockSlew(this, from_pin_id, fel_rf);
        }
        float sc = pinSlew[to_pin_id * NUM_ATTR + i];
        testConstraint[test_id * NUM_ATTR + i] = d_allocator->query(timing_id, frf, rf, sr, sc, 2);
        if (!isnan(testConstraint[test_id * NUM_ATTR + i]) && !isnan(testRelatedAT[test_id * NUM_ATTR + i])) {
            if (el == 0) {
                const float hold_uncertainty = test_hold_uncertainties ? test_hold_uncertainties[test_id] : 0.0f;
                pinRat[to_pin_id * NUM_ATTR + i] = testRelatedAT[test_id * NUM_ATTR + i] + testConstraint[test_id * NUM_ATTR + i] + hold_uncertainty;  // hold clocks needs data stay late, rat = at_clk + T_hold
            } else {
                const float setup_uncertainty = test_setup_uncertainties ? test_setup_uncertainties[test_id] : 0.0f;
                pinRat[to_pin_id * NUM_ATTR + i] = testRelatedAT[test_id * NUM_ATTR + i] - testConstraint[test_id * NUM_ATTR + i] - setup_uncertainty;  // setup clock needs data come early, rat = at_clk - T_setup
            }
            testRAT[test_id * NUM_ATTR + i] = pinRat[to_pin_id * NUM_ATTR + i];
        }
    }
}
__device__ void DmpModel::propagatePinTests(int to_pin_idx){
    if (clock_period <= 0) {
        return;
    }
    int to_pin = level_list[to_pin_idx];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (index_type i = pin_backward_arc_list_end[to_pin]; i < pin_backward_arc_list_end[to_pin + 1]; i++) {
        arc_ids[idx] = pin_backward_arc_list[i];
        propagateTest();
    }
}

__global__ void dmpTestKernel(DmpModel* dmp_db, int level_start_offset, int num_pins_level){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_id = idx >> 3;
    if(pin_id < num_pins_level){
        dmp_db -> propagatePinTests(level_start_offset + pin_id);
    }
}

__global__ void dmpDirectNetKernel(DmpModel* dmp_db,
                                             const index_type* level_arc_list,
                                             int num_level_arcs){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_pos = idx >> 3;
    if (arc_pos < num_level_arcs) {
        const int arc_id = level_arc_list[arc_pos];
        if (dmp_db->arc_types[arc_id] == 0) {
            dmp_db->arc_ids[idx] = arc_id;
            dmp_db->propagateLoadSlewDelay();
        }
    }
}

__global__ void dmpPinWinnerKernel(DmpModel* dmp_db,
                                       int level_start_offset,
                                       int num_pins_level) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_pos = idx >> 2;
    if (pin_pos >= num_pins_level) {
        return;
    }
    const int attr = idx & 0b11;
    const int pin_id = dmp_db->level_list[level_start_offset + pin_pos];
    const int to_slot = pin_id * NUM_ATTR + attr;
    const bool pick_max = (attr >> 1) != 0;

    const unsigned long long packed_at = dmp_db->pin_at_winner[to_slot];
    if (packed_at != 0ULL) {
        const unsigned int cmp_key = static_cast<unsigned int>(packed_at >> 32);
        const unsigned int payload = static_cast<unsigned int>(packed_at & 0xffffffffULL);
        const int arc_id = static_cast<int>(payload >> 2);
        const int from_attr = static_cast<int>(payload & 0x3);
        const float at = dmpDecodeWinnerFloat(cmp_key, pick_max);
        if (arc_id >= 0 && arc_id < dmp_db->num_arcs && isfinite(at)) {
            dmp_db->pinAt[to_slot] = at;
            dmp_db->at_prefix_pin[to_slot] = dmp_db->timing_arc_from_pin_id[arc_id];
            dmp_db->at_prefix_arc[to_slot] = arc_id;
            dmp_db->at_prefix_attr[to_slot] = from_attr;
        }
        dmp_db->pin_at_winner[to_slot] = 0ULL;
    }

    const unsigned long long packed_slew = dmp_db->pin_slew_winner[to_slot];
    if (packed_slew == 0ULL) {
        return;
    }
    const unsigned int cmp_key = static_cast<unsigned int>(packed_slew >> 32);
    const float slew = dmpDecodeWinnerFloat(cmp_key, pick_max);
    if (isfinite(slew)) {
        dmp_db->pinSlew[to_slot] = slew;
    }
    dmp_db->pin_slew_winner[to_slot] = 0ULL;
}

__global__ void dmpNetWinnerKernel(DmpModel* dmp_db,
                                                          const index_type* level_arc_list,
                                                          int num_level_arcs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_pos = idx >> 2;
    if (arc_pos >= num_level_arcs) {
        return;
    }
    const int attr = idx & 0b11;
    const int arc_id = level_arc_list[arc_pos];
    const int delay_idx = (attr << 1) + (attr & 1);
    const int delay_slot = arc_id * 2 * NUM_ATTR + delay_idx;
    const int winner_slot = arc_id * NUM_ATTR + attr;
    const bool pick_max = (attr >> 1) != 0;

    const unsigned long long packed = dmp_db->arc_delay_winner[winner_slot];
    if (packed != 0ULL) {
        const unsigned int cmp_key = static_cast<unsigned int>(packed >> 32);
        const float delay = dmpDecodeWinnerFloat(cmp_key, pick_max);
        if (isfinite(delay)) {
            dmp_db->arcDelay[delay_slot] = delay;
        }
        dmp_db->arc_delay_winner[winner_slot] = 0ULL;
    }

    const int from_pin_id = dmp_db->timing_arc_from_pin_id[arc_id];
    const int to_pin_id = dmp_db->timing_arc_to_pin_id[arc_id];
    const float from_at = dmp_db->pinAt[from_pin_id * NUM_ATTR + attr];
    const float delay = dmp_db->arcDelay[delay_slot];
    if (isnan(from_at) || isnan(delay)) {
        return;
    }
    const float at = from_at + delay;
    dmp_db->updateAtWinner(to_pin_id * NUM_ATTR + attr,
                           at,
                           pick_max,
                           arc_id,
                           attr);
}
