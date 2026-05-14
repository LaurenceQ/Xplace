enum DmpDrivingCellCounter {
    DMP_DRIVING_CELL_APPLIED = 0,
    DMP_DRIVING_CELL_SKIPPED = 1,
    DMP_DRIVING_CELL_CAP = 2,
    DMP_DRIVING_CELL_ZERO_C2 = 3,
    DMP_DRIVING_CELL_PI = 4,
    DMP_DRIVING_CELL_DMP_VALID = 5,
    DMP_DRIVING_CELL_FALLBACK = 6,
    DMP_DRIVING_CELL_COUNTER_COUNT = 7
};

__device__ bool dmp_model::updateLoadWinner(int net_arc_id,
                                            int load_attr,
                                            float wire_delay,
                                            float load_slew) {
    if (!isfinite(wire_delay) || !isfinite(load_slew)) {
        return false;
    }
    const int to_pin_id = timing_arc_to_pin_id[net_arc_id];
    const int to_slot = to_pin_id * NUM_ATTR + load_attr;
    const bool pick_max = (load_attr >> 1) != 0;

    const unsigned int slew_payload = static_cast<unsigned int>(net_arc_id);
    const unsigned long long packed_slew = dmpPackWinner(load_slew, slew_payload, pick_max);
    const unsigned long long old_slew = atomicMax(&pin_slew_winner[to_slot], packed_slew);

    const int delay_slot = arcDelayWinnerSlot(net_arc_id, load_attr);
    const unsigned int delay_payload = static_cast<unsigned int>(to_slot);
    const unsigned long long packed_delay = dmpPackWinner(wire_delay, delay_payload, pick_max);
    const unsigned long long old_delay = atomicMax(&arc_delay_winner[delay_slot], packed_delay);
    return packed_slew > old_slew || packed_delay > old_delay;
}

__device__ bool dmp_model::updateAtWinner(int to_slot,
                                          float at,
                                          bool pick_max,
                                          int from_pin_id,
                                          int arc_id,
                                          int from_attr) {
    if (!isfinite(at)) {
        return false;
    }
    (void)from_pin_id;
    const unsigned int payload = (static_cast<unsigned int>(arc_id) << 2)
                                 | static_cast<unsigned int>(from_attr & 0x3);
    const unsigned long long packed = dmpPackWinner(at, payload, pick_max);
    const unsigned long long old = atomicMax(&pin_at_winner[to_slot], packed);
    return packed > old;
}
