__device__ __forceinline__ void dmpSetGateScratchFallback(dmp_model* dmp_db,
                                                          int slot,
                                                          double table_ceff,
                                                          double table_slew) {
    dmp_db->ceff[slot] = table_ceff;
    dmp_db->rd_[slot] = nanf("");
    dmp_db->t0[slot] = nanf("");
    dmp_db->dt[slot] = nanf("");
    dmp_db->vo_delay_[slot] = nanf("");
    dmp_db->vo_slew_[slot] = table_slew;
    dmp_db->driving_cell_extra_delay_[slot] = nanf("");
    dmp_db->dmp_alg_kind[slot] = DMP_ALG_CAP;
}

__device__ __forceinline__ bool dmpIsIdealClockTimingArc(const dmp_model* dmp_db,
                                                         int timing_id,
                                                         int from_pin_id) {
    return dmp_db->ideal_clock &&
           dmp_db->pin_is_clk != nullptr &&
           dmp_db->pin_is_clk[from_pin_id] &&
           timing_id >= 0;
}

__device__ __forceinline__ float dmpClockPeriodForPin(const dmp_model* dmp_db,
                                                      int pin_id) {
    if (dmp_db->pin_clock_periods != nullptr && pin_id >= 0) {
        const float period = dmp_db->pin_clock_periods[pin_id];
        if (isfinite(period) && period > 0.0f) {
            return period;
        }
    }
    return dmp_db->clock_period;
}

__device__ __forceinline__ float dmpClockPeriodForTest(const dmp_model* dmp_db,
                                                       int test_id) {
    if (dmp_db->test_clock_periods != nullptr && test_id >= 0) {
        const float period = dmp_db->test_clock_periods[test_id];
        if (isfinite(period) && period > 0.0f) {
            return period;
        }
    }
    return dmp_db->clock_period;
}

__device__ __forceinline__ float dmpIdealClockEdgeTime(const dmp_model* dmp_db,
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

__device__ __forceinline__ float dmpIdealClockSlew(const dmp_model* dmp_db,
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

__device__ void dmp_model::propagateGateSlewDelayForArc(int arc_id, int i, bool lock_scratch){
    int el = i >> 2;                            // early late
    int fel_rf = i >> 1;                        // from early/late rise/fall
    int tel_rf = ((i & 0b100) >> 1) + (i & 1);  // to early/late rise/fall
    int irf = fel_rf & 1;                       // input rise/fall
    int orf = tel_rf & 1;                       // output rise/fall
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    int to_slot = to_pin_id * NUM_ATTR + tel_rf;
    int pin_idx = lock_scratch ? (dmp_arc_slot_base + arc_id * DMP_PIN_GROUP_SIZE + i)
                               : (from_pin_id * NUM_ATTR + tel_rf);
    const int timing_id = timing_arc_id_map[arc_id * 2 + el];

    C1[pin_idx] = C1[to_slot];
    C2[pin_idx] = C2[to_slot];
    r_pi[pin_idx] = r_pi[to_slot];
    float input_slew = pinSlew[from_pin_id * NUM_ATTR + fel_rf];
    if (dmpIsIdealClockTimingArc(this, timing_id, from_pin_id) &&
        !d_allocator->d_is_constraint[timing_id]) {
        input_slew = dmpIdealClockSlew(this, from_pin_id, fel_rf);
    }
    const DmpGateLaneContext gate_ctx =
        dmpMakeGateLaneContext(this, pin_idx, timing_id, irf, orf, input_slew);
    double table_ceff = C1[pin_idx] + C2[pin_idx];
    double table_delay = nanf("");
    double table_slew = nanf("");
    dmpGateCapDelaySlewWithCtx(gate_ctx, table_ceff, table_delay, table_slew);
    if (!isfinite(table_delay) || !isfinite(table_slew)) {
        return;
    }

    double delay = table_delay;
    double so = table_slew;
    bool dmp_valid = false;
    const bool rc_can_model = isfinite(C1[pin_idx]) && isfinite(C2[pin_idx]) &&
                              isfinite(r_pi[pin_idx]) && isfinite(table_ceff) &&
                              C1[pin_idx] > 0.0 && C2[pin_idx] >= 0.0 &&
                              r_pi[pin_idx] > 0.0;
    if (!rc_can_model) {
        dmpSetGateScratchFallback(this, pin_idx, table_ceff, so);
        arcDelay[arc_id * 2 * NUM_ATTR + i] = delay;
        updateGateWinner(to_slot, pin_idx, static_cast<float>(so), el != 0, false, table_ceff);
        return;
    }
    dmpGateModelRdWithCtx(this, pin_idx, gate_ctx, table_ceff, table_delay);
    int alg = selectDmpAlg(pin_idx);
    dmp_alg_kind[pin_idx] = alg;

    if (alg == DMP_ALG_ZERO_C2) {
        double c1_delay = nanf("");
        double c1_slew = nanf("");
        dmpGateCapDelaySlewWithCtx(gate_ctx, C1[pin_idx], c1_delay, c1_slew);
        bool ok = isfinite(c1_delay) && isfinite(c1_slew) &&
                  init_zero_c2_factors(pin_idx) &&
                  dmpFindDriverParamsOnePoleScalarWithCtx(this, pin_idx, gate_ctx, C1[pin_idx]);
        if (ok) {
            double vo_delay = nanf("");
            double vo_slew = nanf("");
            dmpFindDriverDelaySlewCached(this,
                                         alg,
                                         k0_[pin_idx],
                                         k1_[pin_idx],
                                         k2_[pin_idx],
                                         k3_[pin_idx],
                                         k4_[pin_idx],
                                         p1_[pin_idx],
                                         p2_[pin_idx],
                                         t0[pin_idx],
                                         dt[pin_idx],
                                         C1[pin_idx],
                                         C2[pin_idx],
                                         r_pi[pin_idx],
                                         rd_[pin_idx],
                                         gate_ctx.driver_vth,
                                         gate_ctx.driver_vl,
                                         gate_ctx.driver_vh,
                                         gate_ctx.driver_derate,
                                         vo_delay,
                                         vo_slew);
            if (isfinite(vo_delay) && isfinite(vo_slew)) {
                delay = vo_delay;
                so = vo_slew;
                vo_delay_[pin_idx] = vo_delay;
                vo_slew_[pin_idx] = vo_slew;
                dmp_valid = true;
            }
        }
    } else if (alg == DMP_ALG_PI) {
        bool factors_ok = init_dmp_factors(pin_idx);
        bool params_ok = factors_ok &&
                         dmpFindDriverParamsScalarWithCtx(this, pin_idx, gate_ctx, table_ceff);
        if (factors_ok && !params_ok && C2[pin_idx] > 0.0) {
            params_ok = dmpFindDriverParamsScalarWithCtx(this, pin_idx, gate_ctx, C2[pin_idx]);
        }
        if (params_ok) {
            double ceff_delay = nanf("");
            double ceff_slew = nanf("");
            dmpGateCapDelaySlewWithCtx(gate_ctx, ceff[pin_idx], ceff_delay, ceff_slew);
            if (isfinite(ceff_delay) && isfinite(ceff_slew)) {
                double vo_delay = nanf("");
                double vo_slew = nanf("");
                dmpFindDriverDelaySlewCached(this,
                                             alg,
                                             k0_[pin_idx],
                                             k1_[pin_idx],
                                             k2_[pin_idx],
                                             k3_[pin_idx],
                                             k4_[pin_idx],
                                             p1_[pin_idx],
                                             p2_[pin_idx],
                                             t0[pin_idx],
                                             dt[pin_idx],
                                             C1[pin_idx],
                                             C2[pin_idx],
                                             r_pi[pin_idx],
                                             rd_[pin_idx],
                                             gate_ctx.driver_vth,
                                             gate_ctx.driver_vl,
                                             gate_ctx.driver_vh,
                                             gate_ctx.driver_derate,
                                             vo_delay,
                                             vo_slew);
                if (isfinite(vo_delay) && isfinite(vo_slew)) {
                    delay = ceff_delay;
                    so = vo_slew;
                    vo_delay_[pin_idx] = vo_delay;
                    vo_slew_[pin_idx] = vo_slew;
                    dmp_valid = true;
                }
            }
        }
    }

    if (!dmp_valid) {
        delay = table_delay;
        so = table_slew;
        dmpSetGateScratchFallback(this, pin_idx, table_ceff, so);
    }
    vo_slew_[pin_idx] = so;
    driving_cell_extra_delay_[pin_idx] = nanf("");

    arcDelay[arc_id * 2 * NUM_ATTR + i] = delay;
    updateGateWinner(to_slot, pin_idx, static_cast<float>(so), el != 0, dmp_valid, table_ceff);

}

__device__ void dmp_model::propagateGateSlewDelay(bool lock_scratch){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    propagateGateSlewDelayForArc(arc_ids[idx], idx & 0b111, lock_scratch);
}

__device__ void dmp_model::propagateSlewDelay(bool lock_scratch){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx & 0b111;
    int arc_id = arc_ids[idx];
    int arc_type = arc_types[arc_id];
    if ((arc_type == 0) && (i < NUM_ATTR)) {  // 0 for net arc
        propagateLoadSlewDelay();
        // int from_pin_id = timing_arc_from_pin_id[arc_id];
        // int to_pin_id = timing_arc_to_pin_id[arc_id];
        // float si = pinSlew[from_pin_id * NUM_ATTR + (i >> 1)];
        // float so = pinSlew[to_pin_id * NUM_ATTR + (i & (0b11))];
        // float delay = arcDelay[arc_id * 2 * NUM_ATTR + i];
        // if((arc_type == 0) && (i < NUM_ATTR))delay = arcDelay[arc_id * 2 * NUM_ATTR + ((i << 1) + (i & 1))]; // net arc delay use same rise/fall
        // printf("arc_id:%d from:%s to:%s #in_arc:%d arc_type:%d i:%d si:%.4f so:%.4f delay:%.4f\n", arc_id, pin_names[from_pin_id], pin_names[to_pin_id], pin_backward_arc_list_end[from_pin_id+1] - pin_backward_arc_list_end[from_pin_id], arc_type, i, si, so, delay);

    } else if (arc_type == 1) {                     // 1 for gate arc
        propagateGateSlewDelay(lock_scratch);
    }
    else return ;
}
__device__ void dmp_model::propagateAT(){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int arc_id = arc_ids[idx];
    int arc_type = arc_types[arc_id];
    int i = idx & 0b111;
    int from_pin_id = timing_arc_from_pin_id[arc_id];
    int to_pin_id = timing_arc_to_pin_id[arc_id];
    int el = 0;
    int fel_rf = 0;
    int tel_rf = 0;
    int delay_idx = i;
    if (arc_type == 0) {
        if (i >= NUM_ATTR) {
            return;
        }
        el = i >> 1;
        fel_rf = i;
        tel_rf = i;
        delay_idx = (i << 1) + (i & 1);
    } else {
        el = i >> 2;
        fel_rf = i >> 1;
        tel_rf = ((i & 0b100) >> 1) + (i & 1);
    }
    const int timing_id = (arc_type == 1) ? timing_arc_id_map[arc_id * 2 + el] : -1;
    const bool ideal_clock_arc = (arc_type == 1) &&
                                 dmpIsIdealClockTimingArc(this, timing_id, from_pin_id) &&
                                 !d_allocator->d_is_constraint[timing_id];
    if ((!ideal_clock_arc && isnan(pinAt[from_pin_id * NUM_ATTR + fel_rf])) ||
        isnan(arcDelay[arc_id * 2 * NUM_ATTR + delay_idx])) return;
    float delay = arcDelay[arc_id * 2 * NUM_ATTR + delay_idx];
    float from_at = ideal_clock_arc
                        ? dmpIdealClockEdgeTime(this, timing_id, from_pin_id)
                        : pinAt[from_pin_id * NUM_ATTR + fel_rf];
    if (ideal_clock_arc && isnan(from_at)) {
        from_at = pinAt[from_pin_id * NUM_ATTR + fel_rf];
    }
    float at = from_at + delay;
    updateAtWinner(to_pin_id * NUM_ATTR + tel_rf, at, el != 0, from_pin_id, arc_id, fel_rf);
}

__device__ __forceinline__ void dmpPropagateGateATForArc(dmp_model* dmp_db,
                                                         int arc_id,
                                                         int lane) {
    const int from_pin_id = dmp_db->timing_arc_from_pin_id[arc_id];
    const int to_pin_id = dmp_db->timing_arc_to_pin_id[arc_id];
    const int el = lane >> 2;
    const int from_attr = lane >> 1;
    const int to_attr = ((lane & 0b100) >> 1) + (lane & 1);
    const int timing_id = dmp_db->timing_arc_id_map[arc_id * 2 + el];
    const bool ideal_clock_arc = dmpIsIdealClockTimingArc(dmp_db, timing_id, from_pin_id) &&
                                 !dmp_db->d_allocator->d_is_constraint[timing_id];
    float from_at = ideal_clock_arc
                        ? dmpIdealClockEdgeTime(dmp_db, timing_id, from_pin_id)
                        : dmp_db->pinAt[from_pin_id * NUM_ATTR + from_attr];
    if (ideal_clock_arc && isnan(from_at)) {
        from_at = dmp_db->pinAt[from_pin_id * NUM_ATTR + from_attr];
    }
    const float delay = dmp_db->arcDelay[arc_id * 2 * NUM_ATTR + lane];
    if (isnan(from_at) || isnan(delay)) {
        return;
    }
    dmp_db->updateAtWinner(to_pin_id * NUM_ATTR + to_attr,
                           from_at + delay,
                           el != 0,
                           from_pin_id,
                           arc_id,
                           from_attr);
}

__device__ void dmp_model::propagateTest(){
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
        const int el_rf_rf = (i << 1) + (i & 1);
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
__device__ void dmp_model::propagatePin(int to_pin_idx){
    int to_pin = level_list[to_pin_idx];
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (index_type i = pin_backward_arc_list_end[to_pin]; i < pin_backward_arc_list_end[to_pin + 1]; i++) {
        arc_ids[idx] = pin_backward_arc_list[i];
        propagateSlewDelay();
        propagateAT();
        if (clock_period > 0) {
            propagateTest();
        }
    }
}

__device__ void dmp_model::propagatePinTests(int to_pin_idx){
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

__global__ void propagatePin_dmp(dmp_model* dmp_db, int level_start_offset, int num_pins_level){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_id = idx >> 3;
    if(pin_id < num_pins_level){
        dmp_db -> propagatePin(level_start_offset + pin_id);
    }
}

__global__ void propagatePinTests_dmp(dmp_model* dmp_db, int level_start_offset, int num_pins_level){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pin_id = idx >> 3;
    if(pin_id < num_pins_level){
        dmp_db -> propagatePinTests(level_start_offset + pin_id);
    }
}

__global__ void propagateArc_dmp(dmp_model* dmp_db, const index_type* level_arc_list, int num_level_arcs){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_pos = idx >> 3;
    if (arc_pos < num_level_arcs) {
        dmp_db->arc_ids[idx] = level_arc_list[arc_pos];
        dmp_db->propagateSlewDelay(true);
    }
}

__global__ void propagateArcDelaySlewAndAT_dmp(dmp_model* dmp_db,
                                               const index_type* level_arc_list,
                                               int num_level_arcs){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_pos = idx >> 3;
    if (arc_pos < num_level_arcs) {
        const int lane = idx & 0b111;
        const int arc_id = level_arc_list[arc_pos];
        dmp_db->propagateGateSlewDelayForArc(arc_id, lane, true);
        dmpPropagateGateATForArc(dmp_db, arc_id, lane);
    }
}

__global__ void propagateHybridGateArcDelaySlewAndAT_dmp(dmp_model* dmp_db,
                                                         const index_type* level_arc_list,
                                                         int num_level_arcs){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_pos = idx >> 3;
    if (arc_pos < num_level_arcs) {
        const int lane = idx & 0b111;
        const int arc_id = level_arc_list[arc_pos];
        dmp_db->propagateGateSlewDelayForArc(arc_id, lane, dmp_db->use_hybrid_arc_slots);
        dmpPropagateGateATForArc(dmp_db, arc_id, lane);
    }
}

__global__ void propagateNetArcSlewDelay_dmp(dmp_model* dmp_db,
                                             const index_type* level_arc_list,
                                             int num_level_arcs){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_pos = idx >> 3;
    if (arc_pos < num_level_arcs) {
        const int arc_id = level_arc_list[arc_pos];
        if (dmp_db->arc_types[arc_id] == 0) {
            dmp_db->arc_ids[idx] = arc_id;
            dmp_db->propagateSlewDelay(false);
        }
    }
}

__device__ __forceinline__ void dmpLoadDelaySlewFromSlotCached(dmp_model* dmp_db,
                                                               int src_slot,
                                                               int net_arc_id,
                                                               int load_attr,
                                                               double& wire_delay,
                                                               double& load_slew);

__global__ void propagateHybridNetArcSlewDelayAndAT_dmp(dmp_model* dmp_db,
                                                        const index_type* level_arc_list,
                                                        int num_level_arcs,
                                                        unsigned long long* debug_counts){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_pos = idx >> 3;
    if (arc_pos >= num_level_arcs) {
        return;
    }
    const int lane = idx & 0b111;
    if (lane >= NUM_ATTR) {
        return;
    }
    const int arc_id = level_arc_list[arc_pos];
    if (dmp_db->arc_types[arc_id] != 0) {
        return;
    }

    if (dmp_db->use_hybrid_arc_slots) {
        const int from_pin_id = dmp_db->timing_arc_from_pin_id[arc_id];
        const int el = lane >> 1;
        const int output_rf = lane & 1;
        bool saw_gate_driver = false;

        if (from_pin_id >= 0 && from_pin_id < dmp_db->num_pins) {
            for (index_type src_pos = dmp_db->pin_backward_arc_list_end[from_pin_id];
                 src_pos < dmp_db->pin_backward_arc_list_end[from_pin_id + 1];
                 ++src_pos) {
                const int gate_arc_id = dmp_db->pin_backward_arc_list[src_pos];
                if (gate_arc_id < 0 || gate_arc_id >= dmp_db->num_arcs ||
                    dmp_db->arc_types[gate_arc_id] != 1) {
                    continue;
                }
                saw_gate_driver = true;
                const int timing_id = dmp_db->timing_arc_id_map[gate_arc_id * 2 + el];
                if (timing_id < 0 || dmp_db->d_allocator == nullptr) {
                    continue;
                }
                for (int input_rf = 0; input_rf < 2; ++input_rf) {
                    if (!dmp_db->d_allocator->is_transition_defined(timing_id, input_rf, output_rf)) {
                        dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_TRANSITION_SKIPS);
                        continue;
                    }
                    dmpGateNetPairCount(debug_counts, DMP_GNP_TOTAL_CANDIDATES);
                    const int gate_lane = (el << 2) | (input_rf << 1) | output_rf;
                    const int src_slot = dmp_db->dmp_arc_slot_base +
                                         gate_arc_id * DMP_PIN_GROUP_SIZE +
                                         gate_lane;
                    if (src_slot < dmp_db->dmp_arc_slot_base ||
                        src_slot >= dmp_db->dmp_slot_capacity) {
                        dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_SCRATCH_SKIPS);
                        continue;
                    }
                    double wire_delay = nanf("");
                    double load_slew = nanf("");
                    dmpLoadDelaySlewFromSlotCached(dmp_db,
                                                   src_slot,
                                                   arc_id,
                                                   lane,
                                                   wire_delay,
                                                   load_slew);
                    if (!isfinite(wire_delay) || !isfinite(load_slew)) {
                        dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_SCRATCH_SKIPS);
                        continue;
                    }
                    dmpGateNetPairCount(debug_counts, DMP_GNP_FINITE_CANDIDATES);
                    dmp_db->updateLoadWinner(arc_id,
                                             lane,
                                             static_cast<float>(wire_delay),
                                             static_cast<float>(load_slew));
                }
            }
        }

        if (saw_gate_driver) {
            return;
        }

        dmp_db->arc_ids[idx] = arc_id;
        dmp_db->propagateSlewDelay(false);
        return;
    }

    dmp_db->arc_ids[idx] = arc_id;
    dmp_db->propagateSlewDelay(false);
    dmp_db->propagateAT();
}

__global__ void propagateArcAT_dmp(dmp_model* dmp_db,
                                   const index_type* level_arc_list,
                                   int num_level_arcs){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int arc_pos = idx >> 3;
    if (arc_pos < num_level_arcs) {
        dmp_db->arc_ids[idx] = level_arc_list[arc_pos];
        dmp_db->propagateAT();
    }
}

__global__ void finalizeAtWinners_dmp(dmp_model* dmp_db,
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
    const unsigned long long packed = dmp_db->pin_at_winner[to_slot];
    if (packed == 0ULL) {
        return;
    }

    const bool pick_max = (attr >> 1) != 0;
    const unsigned int cmp_key = static_cast<unsigned int>(packed >> 32);
    const unsigned int payload = static_cast<unsigned int>(packed & 0xffffffffULL);
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

__global__ void finalizeSlewWinners_dmp(dmp_model* dmp_db,
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
    const unsigned long long packed = dmp_db->pin_slew_winner[to_slot];
    if (packed == 0ULL) {
        return;
    }

    const bool pick_max = (attr >> 1) != 0;
    const unsigned int cmp_key = static_cast<unsigned int>(packed >> 32);
    const unsigned int payload = static_cast<unsigned int>(packed & 0xffffffffULL);
    const float slew = dmpDecodeWinnerFloat(cmp_key, pick_max);
    if (isfinite(slew)) {
        dmp_db->pinSlew[to_slot] = slew;
        if ((payload & 0x80000000u) != 0u) {
            const int src_slot = static_cast<int>(payload & 0x7fffffffu);
            if (src_slot >= 0 && src_slot < dmp_db->dmp_slot_capacity) {
                dmp_db->k0_[to_slot] = dmp_db->k0_[src_slot];
                dmp_db->k1_[to_slot] = dmp_db->k1_[src_slot];
                dmp_db->k2_[to_slot] = dmp_db->k2_[src_slot];
                dmp_db->k3_[to_slot] = dmp_db->k3_[src_slot];
                dmp_db->k4_[to_slot] = dmp_db->k4_[src_slot];
                dmp_db->p1_[to_slot] = dmp_db->p1_[src_slot];
                dmp_db->p2_[to_slot] = dmp_db->p2_[src_slot];
                dmp_db->p3_[to_slot] = dmp_db->p3_[src_slot];
                dmp_db->z1_[to_slot] = dmp_db->z1_[src_slot];
                dmp_db->A_[to_slot] = dmp_db->A_[src_slot];
                dmp_db->B_[to_slot] = dmp_db->B_[src_slot];
                dmp_db->D_[to_slot] = dmp_db->D_[src_slot];
                dmp_db->rd_[to_slot] = dmp_db->rd_[src_slot];
                dmp_db->t0[to_slot] = dmp_db->t0[src_slot];
                dmp_db->dt[to_slot] = dmp_db->dt[src_slot];
                dmp_db->ceff[to_slot] = dmp_db->ceff[src_slot];
                dmp_db->vo_delay_[to_slot] = dmp_db->vo_delay_[src_slot];
                dmp_db->vo_slew_[to_slot] = dmp_db->vo_slew_[src_slot];
                dmp_db->driving_cell_extra_delay_[to_slot] = dmp_db->driving_cell_extra_delay_[src_slot];
                dmp_db->dmp_alg_kind[to_slot] = dmp_db->dmp_alg_kind[src_slot];
                if (dmp_db->slot_vth != nullptr) {
                    dmp_db->slot_vth[to_slot] = dmp_db->slot_vth[src_slot];
                    dmp_db->slot_vl[to_slot] = dmp_db->slot_vl[src_slot];
                    dmp_db->slot_vh[to_slot] = dmp_db->slot_vh[src_slot];
                    dmp_db->slot_slew_derate[to_slot] = dmp_db->slot_slew_derate[src_slot];
                }
            }
        }
    }
    dmp_db->pin_slew_winner[to_slot] = 0ULL;
}

__global__ void finalizePinWinners_dmp(dmp_model* dmp_db,
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
    const unsigned int payload = static_cast<unsigned int>(packed_slew & 0xffffffffULL);
    const float slew = dmpDecodeWinnerFloat(cmp_key, pick_max);
    if (isfinite(slew)) {
        dmp_db->pinSlew[to_slot] = slew;
        if ((payload & 0x80000000u) != 0u) {
            const int src_slot = static_cast<int>(payload & 0x7fffffffu);
            if (src_slot >= 0 && src_slot < dmp_db->dmp_slot_capacity) {
                dmp_db->k0_[to_slot] = dmp_db->k0_[src_slot];
                dmp_db->k1_[to_slot] = dmp_db->k1_[src_slot];
                dmp_db->k2_[to_slot] = dmp_db->k2_[src_slot];
                dmp_db->k3_[to_slot] = dmp_db->k3_[src_slot];
                dmp_db->k4_[to_slot] = dmp_db->k4_[src_slot];
                dmp_db->p1_[to_slot] = dmp_db->p1_[src_slot];
                dmp_db->p2_[to_slot] = dmp_db->p2_[src_slot];
                dmp_db->p3_[to_slot] = dmp_db->p3_[src_slot];
                dmp_db->z1_[to_slot] = dmp_db->z1_[src_slot];
                dmp_db->A_[to_slot] = dmp_db->A_[src_slot];
                dmp_db->B_[to_slot] = dmp_db->B_[src_slot];
                dmp_db->D_[to_slot] = dmp_db->D_[src_slot];
                dmp_db->rd_[to_slot] = dmp_db->rd_[src_slot];
                dmp_db->t0[to_slot] = dmp_db->t0[src_slot];
                dmp_db->dt[to_slot] = dmp_db->dt[src_slot];
                dmp_db->ceff[to_slot] = dmp_db->ceff[src_slot];
                dmp_db->vo_delay_[to_slot] = dmp_db->vo_delay_[src_slot];
                dmp_db->vo_slew_[to_slot] = dmp_db->vo_slew_[src_slot];
                dmp_db->driving_cell_extra_delay_[to_slot] = dmp_db->driving_cell_extra_delay_[src_slot];
                dmp_db->dmp_alg_kind[to_slot] = dmp_db->dmp_alg_kind[src_slot];
                if (dmp_db->slot_vth != nullptr) {
                    dmp_db->slot_vth[to_slot] = dmp_db->slot_vth[src_slot];
                    dmp_db->slot_vl[to_slot] = dmp_db->slot_vl[src_slot];
                    dmp_db->slot_vh[to_slot] = dmp_db->slot_vh[src_slot];
                    dmp_db->slot_slew_derate[to_slot] = dmp_db->slot_slew_derate[src_slot];
                }
            }
        }
    }
    dmp_db->pin_slew_winner[to_slot] = 0ULL;
}

__global__ void finalizeNetDelayWinners_dmp(dmp_model* dmp_db,
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
    const int winner_slot = dmp_db->arcDelayWinnerSlot(arc_id, attr);
    const unsigned long long packed = dmp_db->arc_delay_winner[winner_slot];
    if (packed == 0ULL) {
        return;
    }

    const bool pick_max = (attr >> 1) != 0;
    const unsigned int cmp_key = static_cast<unsigned int>(packed >> 32);
    const float delay = dmpDecodeWinnerFloat(cmp_key, pick_max);
    if (isfinite(delay)) {
        dmp_db->arcDelay[delay_slot] = delay;
    }
    dmp_db->arc_delay_winner[winner_slot] = 0ULL;
}

__global__ void finalizeNetDelayWinnersAndPropagateAT_dmp(dmp_model* dmp_db,
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
    const int winner_slot = dmp_db->arcDelayWinnerSlot(arc_id, attr);
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
                           from_pin_id,
                           arc_id,
                           attr);
}

__global__ void propagateGateNetPair_dmp(dmp_model* dmp_db,
                                         const index_type* gate_arc_list,
                                         const index_type* net_arc_list,
                                         int num_pairs,
                                         unsigned long long* debug_counts){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair_pos = idx >> 3;
    if (pair_pos >= num_pairs) {
        return;
    }
    const int gate_arc_id = gate_arc_list[pair_pos];
    const int net_arc_id = net_arc_list[pair_pos];
    if (dmp_db->arc_types[gate_arc_id] != 1 || dmp_db->arc_types[net_arc_id] != 0) {
        return;
    }
    const int lane = idx & 0b111;
    dmpGateNetPairCount(debug_counts, DMP_GNP_TOTAL_CANDIDATES);
    const int el = lane >> 2;
    const int input_rf = (lane >> 1) & 1;
    const int output_rf = lane & 1;
    const int timing_id = dmp_db->timing_arc_id_map[gate_arc_id * 2 + el];
    if (timing_id == -1 ||
        dmp_db->d_allocator == nullptr ||
        !dmp_db->d_allocator->is_transition_defined(timing_id, input_rf, output_rf)) {
        dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_TRANSITION_SKIPS);
        return;
    }
    const int load_attr = (el << 1) | output_rf;
    const int src_slot = dmp_db->dmp_arc_slot_base + gate_arc_id * DMP_PIN_GROUP_SIZE + lane;
    if (src_slot >= dmp_db->dmp_slot_capacity) {
        dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_SCRATCH_SKIPS);
        return;
    }
    double wire_delay = nanf("");
    double load_slew = nanf("");
    dmp_db->loadDelaySlewFromSlot(src_slot, net_arc_id, load_attr, wire_delay, load_slew);
    if (!isfinite(wire_delay) || !isfinite(load_slew)) {
        dmpGateNetPairCount(debug_counts, DMP_GNP_INVALID_SCRATCH_SKIPS);
        return;
    }
    dmpGateNetPairCount(debug_counts, DMP_GNP_FINITE_CANDIDATES);
    dmp_db->updateLoadWinner(net_arc_id,
                             load_attr,
                             static_cast<float>(wire_delay),
                             static_cast<float>(load_slew));
}
