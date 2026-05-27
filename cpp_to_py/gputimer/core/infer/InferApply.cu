#include "gputimer/core/GPUTimer.h"
#include "gputimer/db/GTDatabase.h"
#include "common/utils/log.h"

#include <array>
#include <tuple>
#include <vector>

namespace gt {

void GPUTimer::apply_infer_data(
    const std::vector<std::pair<int, std::array<float, 4>>>& slews,
    const std::vector<std::pair<int, std::array<float, 4>>>& net_delays,
    const std::vector<std::tuple<int, int, std::array<float, 4>>>& cell_delays,
    float time_to_internal)
{
    int num_arcs = static_cast<int>(gtdb.timing_arc_from_pin_id.size());
    logger.info("[apply_infer_data] num_arcs=%d\n", num_arcs);

    // Ensure any previous GPU kernels have completed before copying
    cudaDeviceSynchronize();
    logger.info("[apply_infer_data] GPU synchronized before D2H copy\n");

    // Copy arcDelay to host for modification
    float* h_arcDelay = new float[num_arcs * 8];
    cudaMemcpy(h_arcDelay, arcDelay, num_arcs * 8 * sizeof(float), cudaMemcpyDeviceToHost);
    logger.info("[apply_infer_data] Copied arcDelay D2H: %d arcs * 8 * 4 bytes\n", num_arcs);

    // Copy pinSlew to host for modification (pinSlew is pinSlew GPU array)
    // pinSlew layout: pinSlew[pin_id * NUM_ATTR + corner] where NUM_ATTR = 4
    int num_pins = static_cast<int>(gtdb.pin_names.size());
    float* h_pinSlew = new float[num_pins * 4];
    cudaMemcpy(h_pinSlew, pinSlew, num_pins * 4 * sizeof(float), cudaMemcpyDeviceToHost);
    logger.info("[apply_infer_data] Copied pinSlew D2H: %d pins * 4 corners * 4 bytes\n", num_pins);

    // Ensure D2H copy is complete before modifying on host
    cudaDeviceSynchronize();
    logger.info("[apply_infer_data] GPU synchronized after D2H copy\n");

    // Update slew values for pins
    int slew_count = 0;
    for (const auto& [pin_id, slew] : slews) {
        if (pin_id >= 0 && pin_id < num_pins) {
            const char* pin_name = (pin_id < (int)gtdb.pin_names.size()) ? gtdb.pin_names[pin_id].c_str() : "UNKNOWN";
            for (int corner = 0; corner < 4; corner++) {
                float slew_internal = slew[corner] * time_to_internal;
                h_pinSlew[pin_id * 4 + corner] = slew_internal;
                if (slew_count < 3) {
                    logger.info("[apply_infer_data]   pin[%d] '%s' slew[%d] = %.6e ns -> %.6e internal\n",
                        pin_id, pin_name, corner, slew[corner], slew_internal);
                }
            }
            slew_count++;
        }
    }
    logger.info("[apply_infer_data] Updated %d pins with slew values\n", slew_count);

    // Update net delays
    // For each net delay prediction (indexed by pin_id), find fanin net arcs using CSR lookup
    int net_delay_updates = 0;
    int net_delay_missing = 0;
    for (const auto& [pin_id, delays] : net_delays) {
        // Use pin_backward_arc_list with CSR format for efficient fanin lookup
        int arc_start = gtdb.pin_backward_arc_list_end[pin_id];
        int arc_end = (pin_id + 1 < (int)gtdb.pin_backward_arc_list_end.size())
                      ? gtdb.pin_backward_arc_list_end[pin_id + 1]
                      : (int)gtdb.pin_backward_arc_list.size();

        bool found = false;
        for (int i = arc_start; i < arc_end; i++) {
            int arc_id = gtdb.pin_backward_arc_list[i];
            if (gtdb.arc_types[arc_id] != 0) continue;  // Skip non-net arcs

            found = true;
            // Found net arc driving this pin - update only valid indices {0, 3, 4, 7}
            // For net arcs, input and output transitions must match (rise→rise or fall→fall)
            // Invalid transitions (rise→fall, fall→rise) are indices {1,2,5,6} - left as NaN
            float delay0 = delays[0] * time_to_internal;  // er (rise)
            float delay1 = delays[1] * time_to_internal;  // ef (fall)
            float delay2 = delays[2] * time_to_internal;  // lr (rise)
            float delay3 = delays[3] * time_to_internal;  // lf (fall)
            h_arcDelay[arc_id * 8 + 0] = delay0;  // i=0: rise-to-rise (valid)
            h_arcDelay[arc_id * 8 + 3] = delay1;  // i=3: fall-to-fall (valid)
            h_arcDelay[arc_id * 8 + 4] = delay2;  // i=4: rise-to-rise (valid)
            h_arcDelay[arc_id * 8 + 7] = delay3;  // i=7: fall-to-fall (valid)
            // Indices [1,2,5,6] remain NaN (invalid transitions for net arcs)

            if (net_delay_updates < 3) {
                const char* pin_name = (pin_id < (int)gtdb.pin_names.size()) ? gtdb.pin_names[pin_id].c_str() : "UNKNOWN";
                logger.info("[apply_infer_data] Net delay pin[%d] '%s' -> arc %d: [%.6e,%.6e,%.6e,%.6e] internal\n",
                    pin_id, pin_name, arc_id, delay0, delay1, delay2, delay3);
            }
            net_delay_updates++;
        }
        if (!found) net_delay_missing++;
    }
    logger.info("[apply_infer_data] Updated %d net arcs, %d net delays with no fanin arc\n", net_delay_updates, net_delay_missing);

    // Update cell delays
    // For each cell delay prediction, find the arc by traversing forward arcs
    int cell_delay_updates = 0;
    int cell_delay_missing = 0;
    for (const auto& [from_pin_id, to_pin_id, delays] : cell_delays) {
        // Iterate through forward arcs of from_pin_id using CSR format
        int arc_start = gtdb.pin_forward_arc_list_end[from_pin_id];
        int arc_end = (from_pin_id + 1 < (int)gtdb.pin_forward_arc_list_end.size())
                      ? gtdb.pin_forward_arc_list_end[from_pin_id + 1]
                      : (int)gtdb.pin_forward_arc_list.size();

        bool found = false;
        int num_fanout = arc_end - arc_start;

        for (int i = arc_start; i < arc_end; i++) {
            int arc_id = gtdb.pin_forward_arc_list[i];

            // Check if this arc connects to to_pin_id and is a cell arc
            if (gtdb.timing_arc_to_pin_id[arc_id] != to_pin_id) continue;
            if (gtdb.arc_types[arc_id] != 1) continue;  // Must be cell arc

            found = true;

            // CSV corners: [er, ef, lr, lf] = [(el=0,orf=0), (el=0,orf=1), (el=1,orf=0), (el=1,orf=1)]
            std::array<std::pair<int, int>, 4> el_orf = {{{0, 0}, {0, 1}, {1, 0}, {1, 1}}};

            int corners_updated = 0;
            for (int corner = 0; corner < 4; corner++) {
                int el = el_orf[corner].first;
                int orf = el_orf[corner].second;

                // Check if this timing is valid
                if (gtdb.timing_arc_id_map[arc_id * 2 + el] == -1) continue;

                float corner_value = delays[corner] * time_to_internal;

                // Update all indices i (0-7) where (i >> 2) == el && (((i & 4) >> 1) + (i & 1)) & 1 == orf
                for (int i = 0; i < 8; i++) {
                    int i_el = i >> 2;
                    int i_tel_rf = ((i & 0b100) >> 1) + (i & 1);
                    int i_orf = i_tel_rf & 1;

                    if (i_el == el && i_orf == orf) {
                        h_arcDelay[arc_id * 8 + i] = corner_value;
                        corners_updated++;
                    }
                }
            }

            if (cell_delay_updates < 3) {
                const char* from_name = (from_pin_id < (int)gtdb.pin_names.size()) ? gtdb.pin_names[from_pin_id].c_str() : "UNKNOWN";
                const char* to_name = (to_pin_id < (int)gtdb.pin_names.size()) ? gtdb.pin_names[to_pin_id].c_str() : "UNKNOWN";
                logger.info("[apply_infer_data] Cell delay [%d]'%s' -> [%d]'%s' arc_id=%d: delays=[%.6e,%.6e,%.6e,%.6e] internal, updated %d corners\n",
                    from_pin_id, from_name, to_pin_id, to_name, arc_id,
                    delays[0]*time_to_internal, delays[1]*time_to_internal,
                    delays[2]*time_to_internal, delays[3]*time_to_internal, corners_updated);
            }
            cell_delay_updates++;
            // Do NOT break — GTDatabase creates TWO arcs per Liberty timing arc:
            // Arc_A: timing_arc_id_map[arc*2+0]=valid, [arc*2+1]=-1  (early corner only)
            // Arc_B: timing_arc_id_map[arc*2+0]=-1,    [arc*2+1]=valid (late corner only)
            // Both have the same (from_pin_id, to_pin_id). We must process both so that
            // Arc_A gets early delays (indices 0-3) and Arc_B gets late delays (indices 4-7).
        }

        if (!found) {
            if (cell_delay_missing < 10) {  // Only log first 10 missing
                const char* from_name = (from_pin_id < (int)gtdb.pin_names.size()) ? gtdb.pin_names[from_pin_id].c_str() : "UNKNOWN";
                const char* to_name = (to_pin_id < (int)gtdb.pin_names.size()) ? gtdb.pin_names[to_pin_id].c_str() : "UNKNOWN";
                logger.warning("[apply_infer_data] Could not find cell arc [%d]'%s' -> [%d]'%s' (searched %d fanout arcs)\n",
                    from_pin_id, from_name, to_pin_id, to_name, num_fanout);
            }
            cell_delay_missing++;
        }
    }
    logger.info("[apply_infer_data] Updated %d cell arcs, %d cell delays with no matching arc\n", cell_delay_updates, cell_delay_missing);

    // Copy back to GPU
    cudaMemcpy(arcDelay, h_arcDelay, num_arcs * 8 * sizeof(float), cudaMemcpyHostToDevice);
    logger.info("[apply_infer_data] Copied arcDelay H2D: %d arcs * 8 * 4 bytes\n", num_arcs);

    cudaMemcpy(pinSlew, h_pinSlew, num_pins * 4 * sizeof(float), cudaMemcpyHostToDevice);
    logger.info("[apply_infer_data] Copied pinSlew H2D: %d pins * 4 corners * 4 bytes\n", num_pins);

    // Ensure H2D copy is complete and GPU sees updated data before kernels access it
    cudaDeviceSynchronize();
    logger.info("[apply_infer_data] GPU synchronized after H2D copy - arcDelay and pinSlew ready for kernels\n");

    // Sample verification: read back some values to confirm
    float sample_arcDelay[8];
    if (num_arcs > 0) {
        cudaMemcpy(sample_arcDelay, arcDelay, 8 * sizeof(float), cudaMemcpyDeviceToHost);
        logger.info("[apply_infer_data] Arc 0 after H2D: [%.6e, %.6e, %.6e, %.6e, %.6e, %.6e, %.6e, %.6e]\n",
            sample_arcDelay[0], sample_arcDelay[1], sample_arcDelay[2], sample_arcDelay[3],
            sample_arcDelay[4], sample_arcDelay[5], sample_arcDelay[6], sample_arcDelay[7]);
    }

    float sample_pinSlew[4];
    if (num_pins > 0) {
        cudaMemcpy(sample_pinSlew, pinSlew, 4 * sizeof(float), cudaMemcpyDeviceToHost);
        logger.info("[apply_infer_data] Pin 0 slew after H2D: [%.6e, %.6e, %.6e, %.6e]\n",
            sample_pinSlew[0], sample_pinSlew[1], sample_pinSlew[2], sample_pinSlew[3]);
    }

    // Final sync before returning
    cudaDeviceSynchronize();
    logger.info("[apply_infer_data] GPU synchronized before returning\n");

    delete[] h_arcDelay;
    delete[] h_pinSlew;
}

// ─────────────────────────────────────────────────────────────────────────────
// read_infer: ML inference CSV where node_id == GPUTimer pin_id directly.
// ─────────────────────────────────────────────────────────────────────────────

}  // namespace gt
