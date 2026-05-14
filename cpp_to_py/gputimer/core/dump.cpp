
/*
 * dump.cpp  — Dump the Xplace gputimer timing graph to a JSONL file.
 *
 * Each line in the output file is one JSON record:
 *
 *   Node record
 *   -----------
 *   {
 *     "type":     "node",
 *     "id":       <int>          // pin_id — matches pin_names 1:1
 *     "name":     "<string>",    // full pin name  (cell/port or IO)
 *     "features": [<10 floats>], // see below
 *     "labels":   [<17 floats>]  // see below
 *   }
 *
 *   Net-arc records (two records emitted per physical net connection)
 *   ----------------------------------------------------------------
 *   net_out  (driver → sink):
 *   {
 *     "type":     "net_out",
 *     "from":     <int>,         // driver pin_id
 *     "to":       <int>,         // sink   pin_id
 *     "features": [<2 floats>],  // [dx_um, dy_um]  (to − from)
 *     "labels":   [<12 floats>]  // OpenROAD CSV-compatible: all zeros
 *   }
 *   net_in   (sink → driver, reversed):
 *   {
 *     "type":     "net_in",
 *     "from":     <int>,         // sink   pin_id
 *     "to":       <int>,         // driver pin_id
 *     "features": [<2 floats>],  // [-dx_um, -dy_um]  (from − to, negated)
 *     "labels":   [<12 floats>]  // OpenROAD CSV-compatible: all zeros
 *   }
 *
 *   Cell-arc record (to output pin)
 *   --------------------------------
 *   {
 *     "type":     "cell_out",
 *     "from":     <int>,                    // input pin_id
 *     "to":       <int>,                    // output pin_id
 *     "sdf_cond": "<string>",               // SDF condition (e.g., "")
 *     "labels":   [<12 floats>],            // delay[0..3], dst slew[4..7], reserved[8..11]
 *     "features": {                         // LUT objects packed for ML
 *       "lut_cell_delay_rise": <lut>,       // LUT for cell delay rise
 *       "lut_cell_delay_fall": <lut>,       // LUT for cell delay fall
 *       "lut_trans_rise":      <lut>,       // LUT for transition time rise
 *       "lut_trans_fall":      <lut>        // LUT for transition time fall
 *     }
 *   }
 *
 *   labels layout [12]:
 *     [0]  early-rise   — min delay across input transitions (el=0, orf=0), or NaN if both undefined
 *     [1]  early-fall   — min delay across input transitions (el=0, orf=1)
 *     [2]  late-rise    — max delay across input transitions (el=1, orf=0)
 *     [3]  late-fall    — max delay across input transitions (el=1, orf=1)
 *     [4..7] destination pin slew [early-rise, early-fall, late-rise, late-fall]
 *     [8..11] reserved zeros
 *
 *   LUT object layout (each in features object):
 *     {
 *       "allocated": true/false,    // whether this LUT is allocated in library
 *       "var": [<int>, <int>],      // variable indices (timing dimension indices)
 *       "x": [<7 floats>],          // x-axis breakpoints (7 values)
 *       "y": [<7 floats>],          // y-axis breakpoints (7 values)
 *       "table": [<49 floats>]      // 7×7 lookup table flattened row-major
 *     }
 *
 * Node feature layout [10]:
 *   [0]   is_PI_PO   — 1.0 if primary input or primary output, else 0.0
 *   [1]   is_driver  — 1.0 if pin drives a net (source of a net arc), else 0.0
 *   [2]   pin_x      — pin X position in µm
 *   [3]   pin_y      — pin Y position in µm
 *   [4]   right_margin — (die_hx − pin_x) in µm
 *   [5]   top_margin  — (die_hy − pin_y) in µm
 *   [6]   load_er    — total load cap, early-rise, in pF
 *   [7]   load_ef    — total load cap, early-fall, in pF
 *   [8]   load_lr    — total load cap, late-rise,  in pF
 *   [9]   load_lf    — total load cap, late-fall,  in pF
 *
 * Node label layout [17]:
 *   [0..3]  incoming_wire_delay — net-arc delay into this pin [er,ef,lr,lf] ns
 *                                 (0 for driver pins / PIs with no fanin net arc)
 *   [4..7]  at    — arrival time                        [er,ef,lr,lf] ns
 *   [8..11] slew  — transition time                     [er,ef,lr,lf] ns
 *   [12]    is_endpoint — 1.0 if timing endpoint, else 0.0
 *   [13..16] rat  — required arrival time               [er,ef,lr,lf] ns
 *
 * Index order for 4-corner (NUM_ATTR = 4):
 *   0 → early-rise  (el=0, rf=0)
 *   1 → early-fall  (el=0, rf=1)
 *   2 → late-rise   (el=1, rf=0)
 *   3 → late-fall   (el=1, rf=1)
 */

#include "GPUTimer.h"
#include "common/db/Database.h"
#include "common/lib/Lut.h"
#include "common/lib/Timing.h"
#include "gputimer/db/GTDatabase.h"

#include <fstream>
#include <sstream>
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <cstdio>

namespace gt {

// arcDelay layout for net arcs:
//   arcDelay[arc_id * 2*NUM_ATTR + NET_DELAY_IDX[k]]  (k = 0..3)
// Derivation from propagate.cu:
//   i=0 (er): el_rf_rf = (0<<1)+(0&1) = 0
//   i=1 (ef): el_rf_rf = (1<<1)+(1&1) = 3
//   i=2 (lr): el_rf_rf = (2<<1)+(2&1) = 4
//   i=3 (lf): el_rf_rf = (3<<1)+(3&1) = 7
static constexpr int NET_DELAY_IDX[4] = {0, 3, 4, 7};

void GPUTimer::dump_timing_graph(const std::string& outfile) {

    // ----------------------------------------------------------------
    // 1.  Transfer GPU tensors to CPU
    // ----------------------------------------------------------------
    auto h_AT    = timing_raw_db.pinAT.cpu().contiguous();
    auto h_RAT   = timing_raw_db.pinRAT.cpu().contiguous();
    auto h_Slew  = timing_raw_db.pinSlew.cpu().contiguous();
    auto h_Load  = timing_raw_db.pinLoad.cpu().contiguous();
    auto h_AD    = timing_raw_db.arcDelay.cpu().contiguous();
    auto h_x     = timing_raw_db.x.cpu().contiguous();
    auto h_y     = timing_raw_db.y.cpu().contiguous();
    auto h_pox   = timing_raw_db.pin_offset_x.cpu().contiguous();
    auto h_poy   = timing_raw_db.pin_offset_y.cpu().contiguous();
    auto h_p2n   = timing_raw_db.pin2node_map.cpu().contiguous();

    const float* at   = h_AT.data_ptr<float>();
    const float* rat  = h_RAT.data_ptr<float>();
    const float* slew = h_Slew.data_ptr<float>();
    const float* load = h_Load.data_ptr<float>();
    const float* ad   = h_AD.data_ptr<float>();
    const float* cx   = h_x.data_ptr<float>();
    const float* cy   = h_y.data_ptr<float>();
    const float* pox  = h_pox.data_ptr<float>();
    const float* poy  = h_poy.data_ptr<float>();
    const int*   p2n  = h_p2n.data_ptr<int>();

    // ----------------------------------------------------------------
    // 2.  Unit conversions
    // ----------------------------------------------------------------
    // gtdb.time_unit / cap_unit are in SI base units (seconds, farads).
    const float time_to_ns = static_cast<float>(gtdb.time_unit * 1e9);
    const float cap_to_pF  = static_cast<float>(gtdb.cap_unit  * 1e12);
    // x, y, pin_offset_x, pin_offset_y are in placement coordinates (normalized by site_width).
    // scale_factor = 1/site_width, so to convert to microns:
    // pos * (1/scale_factor) gives database units, then * (1/DBU_Micron) gives microns.
    const float pos_to_um  = (1.0f / scale_factor) / static_cast<float>(gtdb.rawdb.DBU_Micron);

    // Die upper bounds are in database units, convert to microns.
    const float die_hx_um = gtdb.rawdb.dieHX / static_cast<float>(gtdb.rawdb.DBU_Micron);
    const float die_hy_um = gtdb.rawdb.dieHY / static_cast<float>(gtdb.rawdb.DBU_Micron);

    // ----------------------------------------------------------------
    // 3.  Build classification sets (all CPU vectors from gtdb)
    // ----------------------------------------------------------------
    std::unordered_set<int> pi_set(gtdb.primary_inputs.begin(),  gtdb.primary_inputs.end());
    std::unordered_set<int> po_set(gtdb.primary_outputs.begin(), gtdb.primary_outputs.end());
    std::unordered_set<int> ep_set(gtdb.endpoints_id.begin(),    gtdb.endpoints_id.end());

    // ----------------------------------------------------------------
    // 4.  Pre-pass: is_driver flag + incoming wire delay per sink pin
    //     Uses gtdb CPU vectors for arc topology (arc_types etc.)
    // ----------------------------------------------------------------
    std::vector<bool>                 is_drv(num_pins, false);
    std::vector<std::array<float, 4>> wire_in(num_pins, {0.f, 0.f, 0.f, 0.f});

    for (int arc_id = 0; arc_id < num_arcs; ++arc_id) {
        if (gtdb.arc_types[arc_id] != 0) continue;  // only net arcs
        int from = gtdb.timing_arc_from_pin_id[arc_id];
        int to   = gtdb.timing_arc_to_pin_id[arc_id];
        is_drv[from] = true;
        for (int k = 0; k < 4; ++k) {
            wire_in[to][k] =
                ad[arc_id * 2 * NUM_ATTR + NET_DELAY_IDX[k]] * time_to_ns;
        }
    }

    // ----------------------------------------------------------------
    // 5.  Open output file (create parent directories if needed)
    // ----------------------------------------------------------------
    std::filesystem::path outpath(outfile);
    std::filesystem::create_directories(outpath.parent_path());

    std::ofstream fout(outfile);
    if (!fout.is_open()) {
        logger.error("dump_timing_graph: cannot open %s\n", outfile.c_str());
        return;
    }
    // Use larger buffer for faster I/O
    char iobuf[65536];
    fout.rdbuf()->pubsetbuf(iobuf, sizeof(iobuf));

    // Helper to format float with 6 decimals
    auto fmt_float = [](float f) -> std::string {
        if (!std::isfinite(f)) f = 0.0f;
        char buf[32];
        snprintf(buf, sizeof(buf), "%.6f", f);
        return std::string(buf);
    };
    const std::string zero_edge_labels =
        "0.000000,0.000000,0.000000,0.000000,"
        "0.000000,0.000000,0.000000,0.000000,"
        "0.000000,0.000000,0.000000,0.000000";

    // ----------------------------------------------------------------
    // 6.  Write node records
    // ----------------------------------------------------------------
    for (int pid = 0; pid < num_pins; ++pid) {
        int   nid   = p2n[pid];
        float px_um = (cx[nid] + pox[pid]) * pos_to_um;
        float py_um = (cy[nid] + poy[pid]) * pos_to_um;

        bool pipo = pi_set.count(pid) || po_set.count(pid);
        bool drv  = is_drv[pid];
        bool ep   = ep_set.count(pid) > 0;

        std::string rec = "{\"type\":\"node\",\"id\":" + std::to_string(pid)
             + ",\"name\":\"" + gtdb.pin_names[pid] + "\""
             + ",\"features\":["
             + std::to_string(pipo ? 1 : 0) + ","
             + std::to_string(drv  ? 1 : 0) + ","
             + fmt_float(px_um) + ","
             + fmt_float(py_um) + ","
             + fmt_float(die_hx_um - px_um) + ","
             + fmt_float(die_hy_um - py_um);
        for (int k = 0; k < NUM_ATTR; ++k) {
            float load_val = load[pid * NUM_ATTR + k] * cap_to_pF;
            if (std::isnan(load_val) || drv) load_val = 0.0f;
            rec += "," + fmt_float(load_val);
        }
        rec += "],\"labels\":[";

        for (int k = 0; k < 4; ++k)
            rec += (k ? "," : "") + fmt_float(wire_in[pid][k]);
        for (int k = 0; k < NUM_ATTR; ++k)
            rec += "," + fmt_float(at[pid * NUM_ATTR + k] * time_to_ns);
        for (int k = 0; k < NUM_ATTR; ++k)
            rec += "," + fmt_float(slew[pid * NUM_ATTR + k] * time_to_ns);
        rec += "," + std::to_string(ep ? 1 : 0);
        for (int k = 0; k < NUM_ATTR; ++k) {
            float rat_val = rat[pid * NUM_ATTR + k] * time_to_ns;
            // If RAT is NaN, set it equal to AT (slack = 0)
            if (std::isnan(rat_val)) {
                rat_val = at[pid * NUM_ATTR + k] * time_to_ns;
            }
            rec += "," + fmt_float(rat_val);
        }

        rec += "]}\n";
        fout << rec;
    }

    // ----------------------------------------------------------------
    // 7.  Collect cell_out edges keyed by the logical arc signature.
    //     Min/max Liberty arcs are distinct objects, so use encode_str_ +
    //     sdf_cond to merge them into one OpenROAD-compatible edge.
    // ----------------------------------------------------------------
    auto nan_safe_min = [](float a, float b) -> float {
        if (std::isnan(a) && std::isnan(b)) return std::numeric_limits<float>::quiet_NaN();
        return std::isnan(a) ? b : (std::isnan(b) ? a : std::min(a, b));
    };
    auto nan_safe_max = [](float a, float b) -> float {
        if (std::isnan(a) && std::isnan(b)) return std::numeric_limits<float>::quiet_NaN();
        return std::isnan(a) ? b : (std::isnan(b) ? a : std::max(a, b));
    };

    struct CellOutKey {
        int from, to;
        std::string arc_sig;
        std::string sdf_cond;

        bool operator==(const CellOutKey& o) const {
            return from == o.from
                && to == o.to
                && arc_sig == o.arc_sig
                && sdf_cond == o.sdf_cond;
        }
    };
    struct CellOutKeyHash {
        size_t operator()(const CellOutKey& k) const {
            size_t h1 = std::hash<int>()(k.from);
            size_t h2 = std::hash<int>()(k.to);
            size_t h3 = std::hash<std::string>()(k.arc_sig);
            size_t h4 = std::hash<std::string>()(k.sdf_cond);
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
        }
    };
    struct CellOutEdge {
        std::array<float, 4> delays;  // [er, ef, lr, lf]
        TimingArc* arc;
        CellOutEdge() : delays({std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::quiet_NaN()}),
                        arc(nullptr) {}
    };

    std::unordered_map<CellOutKey, CellOutEdge, CellOutKeyHash> cell_out_map;

    for (int arc_id = 0; arc_id < num_arcs; ++arc_id) {
        int from  = gtdb.timing_arc_from_pin_id[arc_id];
        int to    = gtdb.timing_arc_to_pin_id[arc_id];
        int atype = gtdb.arc_types[arc_id];

        if (atype == 0) {
            // ---- net arc: emit net_out (driver→sink) + net_in (sink→driver) ----
            float fx = (cx[p2n[from]] + pox[from]) * pos_to_um;
            float fy = (cy[p2n[from]] + poy[from]) * pos_to_um;
            float tx = (cx[p2n[to]]   + pox[to])   * pos_to_um;
            float ty = (cy[p2n[to]]   + poy[to])   * pos_to_um;

            // net_out: driver → sink,  features = [dx, dy]
            std::string rec_out = "{\"type\":\"net_out\",\"from\":" + std::to_string(from)
                 + ",\"to\":" + std::to_string(to)
                 + ",\"features\":["
                 + fmt_float(tx - fx) + "," + fmt_float(ty - fy)
                 + "],\"labels\":[" + zero_edge_labels + "]}\n";
            fout << rec_out;

            // net_in:  sink → driver,  features = [-dx, -dy]
            std::string rec_in = "{\"type\":\"net_in\",\"from\":" + std::to_string(to)
                 + ",\"to\":" + std::to_string(from)
                 + ",\"features\":["
                 + fmt_float(fx - tx) + "," + fmt_float(fy - ty)
                 + "],\"labels\":[" + zero_edge_labels + "]}\n";
            fout << rec_in;

        } else if (atype == 1) {
            // ---- cell arc: collect per logical TimingArc signature for merging ----
            // timing_arc_id_map[arc_id*2+el]: liberty timing id (-1 if this el is absent).
            // For each valid el, reduce over input rise/fall (irf):
            //   el=0 (early): min over irf for each orf  → arcDelay[0,2] (orf=0), [1,3] (orf=1)
            //   el=1 (late):  max over irf for each orf  → arcDelay[4,6] (orf=0), [5,7] (orf=1)
            // Key merges tid_e/tid_l from separate min/max Liberty libraries.
            int tid_e = gtdb.timing_arc_id_map[arc_id * 2 + 0];
            int tid_l = gtdb.timing_arc_id_map[arc_id * 2 + 1];

            if (tid_e != -1) {
                TimingArc* arc_ptr = gtdb.liberty_timing_arcs[tid_e];
                if (arc_ptr && !arc_ptr->is_constraint()) {
                    CellOutKey key{from, to, arc_ptr->encode_str_, arc_ptr->sdf_cond_};
                    auto& edge = cell_out_map[key];
                    if (!edge.arc) edge.arc = arc_ptr;
                    float er = nan_safe_min(ad[arc_id * 2 * NUM_ATTR + 0], ad[arc_id * 2 * NUM_ATTR + 2]);
                    float ef = nan_safe_min(ad[arc_id * 2 * NUM_ATTR + 1], ad[arc_id * 2 * NUM_ATTR + 3]);
                    edge.delays[0] = nan_safe_min(edge.delays[0], er);
                    edge.delays[1] = nan_safe_min(edge.delays[1], ef);
                }
            }
            if (tid_l != -1) {
                TimingArc* arc_ptr = gtdb.liberty_timing_arcs[tid_l];
                if (arc_ptr && !arc_ptr->is_constraint()) {
                    CellOutKey key{from, to, arc_ptr->encode_str_, arc_ptr->sdf_cond_};
                    auto& edge = cell_out_map[key];
                    if (!edge.arc) edge.arc = arc_ptr;
                    float lr = nan_safe_max(ad[arc_id * 2 * NUM_ATTR + 4], ad[arc_id * 2 * NUM_ATTR + 6]);
                    float lf = nan_safe_max(ad[arc_id * 2 * NUM_ATTR + 5], ad[arc_id * 2 * NUM_ATTR + 7]);
                    edge.delays[2] = nan_safe_max(edge.delays[2], lr);
                    edge.delays[3] = nan_safe_max(edge.delays[3], lf);
                }
            }
        }
    }

    // ---- Helper to serialize a Lut to string ----
    auto emit_lut = [&](const Lut* lut) -> std::string {
        std::string res;
        bool alloc = lut && lut->set_;
        res += "{\"allocated\":" + std::string(alloc ? "true" : "false");
        if (alloc) {
            int v1 = -1, v2 = -1;
            if (lut->lut_template) {
                if (lut->lut_template->variable1.has_value())
                    v1 = static_cast<int>(lut->lut_template->variable1.value());
                if (lut->lut_template->variable2.has_value())
                    v2 = static_cast<int>(lut->lut_template->variable2.value());
            }
            float sx = (v1 == 0) ? cap_to_pF : time_to_ns;
            float sy = (v2 == 0) ? cap_to_pF : time_to_ns;

            res += ",\"var\":[" + std::to_string(v1) + "," + std::to_string(v2) + "]";
            res += ",\"x\":[";
            for (size_t i = 0; i < lut->indices1.size(); ++i) {
                if (i) res += ',';
                res += fmt_float(lut->indices1[i] * sx);
            }
            res += "],\"y\":[";
            for (size_t i = 0; i < lut->indices2.size(); ++i) {
                if (i) res += ',';
                res += fmt_float(lut->indices2[i] * sy);
            }
            res += "],\"table\":[";
            for (size_t i = 0; i < lut->table.size(); ++i) {
                if (i) res += ',';
                res += fmt_float(lut->table[i] * time_to_ns);
            }
            res += "]";
        }
        res += "}";
        return res;
    };

    // ---- Emit merged cell_out records ----
    // Each unique logical arc signature produces one edge, merging min/max
    // Liberty TimingArc objects into the same OpenROAD-compatible cell_out row.
    for (const auto& [key, edge] : cell_out_map) {
        std::string rec = "{\"type\":\"cell_out\",\"from\":" + std::to_string(key.from)
             + ",\"to\":" + std::to_string(key.to)
             + ",\"sdf_cond\":\"" + key.sdf_cond + "\""
             + ",\"labels\":["
             + fmt_float(edge.delays[0] * time_to_ns) + ","
             + fmt_float(edge.delays[1] * time_to_ns) + ","
             + fmt_float(edge.delays[2] * time_to_ns) + ","
             + fmt_float(edge.delays[3] * time_to_ns) + ","
             + fmt_float(slew[key.to * NUM_ATTR + 0] * time_to_ns) + ","
             + fmt_float(slew[key.to * NUM_ATTR + 1] * time_to_ns) + ","
             + fmt_float(slew[key.to * NUM_ATTR + 2] * time_to_ns) + ","
             + fmt_float(slew[key.to * NUM_ATTR + 3] * time_to_ns) + ","
             + "0.000000,0.000000,0.000000,0.000000"
             + "],\"features\":{\"lut_cell_delay_rise\":"
             + emit_lut(edge.arc ? edge.arc->cell_delay_[0] : nullptr)
             + ",\"lut_cell_delay_fall\":"
             + emit_lut(edge.arc ? edge.arc->cell_delay_[1] : nullptr)
             + ",\"lut_trans_rise\":"
             + emit_lut(edge.arc ? edge.arc->transition_[0] : nullptr)
             + ",\"lut_trans_fall\":"
             + emit_lut(edge.arc ? edge.arc->transition_[1] : nullptr)
             + "}}\n";
        fout << rec;
    }

    fout.close();
    logger.info("dump_timing_graph: wrote %d pins, %d arcs (net arcs doubled as net_in/net_out) → %s\n",
                num_pins, num_arcs, outfile.c_str());
}

}  // namespace gt
/*
            vector<float> delay_table_values = flattenTableModel(model->delayModel());
            vector<float> slew_table_values = flattenTableModel(model->slewModel());
            vector<float> delay_axis_values = extractAxes(model->delayModel());
            vector<float> slew_axis_values = extractAxes(model->slewModel());
            size_t elrf = dcalc_ap->slewMinMax()->index()*2 + arc->fromEdge()->asRiseFall()->index();
            size_t offset_delay_axis = elrf * delay_axis_values.size() + 8;
            size_t offset_slew_axis = elrf * slew_axis_values.size() + delay_axis_values.size() * 4 + 8;
            size_t offset_delay = elrf * delay_table_values.size() + (delay_axis_values.size() + slew_axis_values.size()) * 4 + 8;
            size_t offset_slew = elrf * slew_table_values.size() + (delay_table_values.size() + slew_axis_values.size() + delay_axis_values.size()) * 4 + 8;
            // printf("offset_delay: %zu, offset_slew: %zu, offset_delay_axis: %zu, offset_slew_axis: %zu\n", offset_delay, offset_slew, offset_delay_axis, offset_slew_axis);
            edge.features[elrf] = edge.features[elrf + 4] = 1.0;
            for(int i = 0; i < delay_table_values.size(); i++) {
              assert(offset_delay + i < edge.features.size());
              edge.features[offset_delay + i] = delay_table_values[i] * 1e12;
            }
            for(int i = 0; i < slew_table_values.size(); i++) {
              assert(offset_slew + i < edge.features.size());
              edge.features[offset_slew + i] = slew_table_values[i] * 1e12;
            }
            for(int i = 0; i < delay_axis_values.size(); i++) {
              assert(offset_delay_axis + i < edge.features.size());
              if(i < 7)
                edge.features[offset_delay_axis + i] = delay_axis_values[i] * 1e12;
              else
                edge.features[offset_delay_axis + i] = delay_axis_values[i] * 1e15;
            }
            for(int i = 0; i < slew_axis_values.size(); i++) {
              assert(offset_slew_axis + i < edge.features.size());
              if(i < 7)
                edge.features[offset_slew_axis + i] = slew_axis_values[i] * 1e12;
              else
                edge.features[offset_slew_axis + i] = slew_axis_values[i] * 1e15;
            }   


*/
