#include "gputimer/core/GPUTimer.h"
#include "gputimer/db/GTDatabase.h"
#include "common/XplaceLog.h"
#include "common/utils/log.h"

#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace gt {

// Helper: trim whitespace from string
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

// Helper: parse CSV line into float vector
static bool parse_csv_line(const std::string& line, std::vector<float>& values) {
    std::istringstream iss(line);
    std::string token;
    values.clear();

    while (std::getline(iss, token, ',')) {
        token = trim(token);
        if (token.empty()) continue;
        try {
            values.push_back(std::stof(token));
        } catch (...) {
            return false;  // Parse error
        }
    }
    return !values.empty();
}

// Helper: split CSV line into string tokens (used when a field is a string, not a float)
static std::vector<std::string> split_csv_str(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string token;
    while (std::getline(iss, token, ','))
        tokens.push_back(trim(token));
    return tokens;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared GPU update: D2H → apply slew / net-delay / cell-delay updates → H2D
// Called by both read_infer() and read_opr_gt_infer() after their respective
// parsers have built the three data vectors (all indexed by GPUTimer pin IDs).
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
void GPUTimer::read_opr_gt_infer(const std::string& infile) {
    std::ifstream fin(infile);
    if (!fin.is_open()) {
        logger.error("Cannot open OPR GT infer file: %s", infile.c_str());
        return;
    }
    logger.info("[read_opr_gt_infer] opened file: %s", infile.c_str());

    // Build reverse map: pin_name → GPUTimer pin_id
    std::unordered_map<std::string, int> name_to_id;
    name_to_id.reserve(gtdb.pin_names.size());
    for (int i = 0; i < (int)gtdb.pin_names.size(); i++)
        name_to_id[gtdb.pin_names[i]] = i;

    // OPR node_id → GPUTimer pin_id (-1 = unmapped)
    std::unordered_map<int, int> opr_to_gt;

    int num_pins_gt = static_cast<int>(gtdb.pin_names.size());
    host_pinGT_AT.assign(num_pins_gt * 4, std::numeric_limits<float>::quiet_NaN());

    float time_to_internal = 1.0f / (gtdb.time_unit * 1e9f);
    XPLACE_DEBUGF("XPLACE_INFER_DEBUG",
                  "[read_opr_gt_infer] time_unit=%.6e time_to_internal=%.6e",
                  gtdb.time_unit, time_to_internal);

    std::vector<std::pair<int, std::array<float, 4>>> slews, net_delays;
    std::vector<std::tuple<int, int, std::array<float, 4>>> cell_delays;

    std::string line;
    int section = 0;
    int line_count = 0, name_miss = 0;

    while (std::getline(fin, line)) {
        line = trim(line);
        line_count++;
        if (line.empty()) continue;
        if (line.find("# Node predictions") == 0) {
            section = 1;
            XPLACE_DEBUGF("XPLACE_INFER_DEBUG", "[read_opr_gt_infer] section=node line=%d", line_count);
            continue;
        }
        if (line.find("# Cell edge predictions") == 0) {
            section = 2;
            XPLACE_DEBUGF("XPLACE_INFER_DEBUG", "[read_opr_gt_infer] section=edge line=%d", line_count);
            continue;
        }
        if (line[0] == '#') continue;

        if (section == 1) {
            // node_id, pin_name, gt_at_er, gt_at_ef, gt_at_lr, gt_at_lf,
            // slew_er, slew_ef, slew_lr, slew_lf,
            // net_delay_er, net_delay_ef, net_delay_lr, net_delay_lf  (14 cols)
            auto tokens = split_csv_str(line);
            if (tokens.size() < 14) continue;
            try {
                int opr_id = std::stoi(tokens[0]);
                // OPR uses "hierarchy/inst/pin" separators; GTDatabase uses "hierarchy/inst:pin"
                // Replace only the LAST '/' (which separates instance path from port name).
                std::string pin_name = tokens[1];
                auto slash = pin_name.rfind('/');
                if (slash != std::string::npos) pin_name[slash] = ':';
                auto it = name_to_id.find(pin_name);
                if (it == name_to_id.end()) {
                    opr_to_gt[opr_id] = -1;
                    if (++name_miss <= 5) {
                        logger.warning("[read_opr_gt_infer] pin_name '%s' not found in GPUTimer", pin_name.c_str());
                    }
                    continue;
                }
                int gt_id = it->second;
                opr_to_gt[opr_id] = gt_id;

                // GT AT values (cols 2-5), stored in internal units
                for (int c = 0; c < 4; c++)
                    host_pinGT_AT[gt_id * 4 + c] = std::stof(tokens[2 + c]) * time_to_internal;

                // Slew (cols 6-9) and net delay (cols 10-13)
                std::array<float, 4> slew   = {{std::stof(tokens[6]),  std::stof(tokens[7]),  std::stof(tokens[8]),  std::stof(tokens[9])}};
                std::array<float, 4> delays = {{std::stof(tokens[10]), std::stof(tokens[11]), std::stof(tokens[12]), std::stof(tokens[13])}};
                slews.push_back({gt_id, slew});
                net_delays.push_back({gt_id, delays});

                if (slews.size() <= 3 || slews.size() % 10000 == 0)
                    XPLACE_DEBUGF("XPLACE_INFER_DEBUG",
                                  "[read_opr_gt_infer] node opr=%d gt=%d name=%s gt_at0=%.6e slew0=%.6e total=%zu",
                                  opr_id, gt_id, pin_name.c_str(), std::stof(tokens[2]), slew[0], slews.size());
            } catch (...) { continue; }

        } else if (section == 2) {
            // edge_id, from_opr_node, to_opr_node, cell_delay_er, ef, lr, lf  (7 cols)
            std::vector<float> values;
            if (!parse_csv_line(line, values) || values.size() < 7) continue;
            int opr_from = static_cast<int>(values[1]);
            int opr_to   = static_cast<int>(values[2]);
            auto it1 = opr_to_gt.find(opr_from), it2 = opr_to_gt.find(opr_to);
            if (it1 == opr_to_gt.end() || it2 == opr_to_gt.end() || it1->second < 0 || it2->second < 0) continue;
            std::array<float, 4> delays = {{values[3], values[4], values[5], values[6]}};
            cell_delays.push_back({it1->second, it2->second, delays});
            if (cell_delays.size() <= 3 || cell_delays.size() % 5000 == 0)
                XPLACE_DEBUGF("XPLACE_INFER_DEBUG",
                              "[read_opr_gt_infer] edge=%d opr=%d->%d gt=%d->%d delay0=%.6e total=%zu",
                              (int)values[0], opr_from, opr_to,
                              it1->second, it2->second, delays[0], cell_delays.size());
        }
    }
    auto device = timing_raw_db.node_size_x.device();
    timing_raw_db.pinGT_AT = torch::from_blob(host_pinGT_AT.data(), {(long)num_pins, 4}, torch::kFloat32).contiguous().to(device);
    fin.close();

    apply_infer_data(slews, net_delays, cell_delays, time_to_internal);

    logger.info("[read_opr_gt_infer] loaded slews=%zu net_delays=%zu cell_delays=%zu name_misses=%d file=%s",
                slews.size(), net_delays.size(), cell_delays.size(), name_miss, infile.c_str());
}

}  // namespace gt
