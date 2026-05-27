#include "gputimer/core/GPUTimer.h"
#include "gputimer/db/GTDatabase.h"
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
void GPUTimer::read_infer(const std::string& infile) {
    std::ifstream fin(infile);
    if (!fin.is_open()) {
        logger.error("Cannot open inference file: %s\n", infile.c_str());
        return;
    }
    logger.info("[read_infer] Opened file: %s\n", infile.c_str());

    std::vector<std::pair<int, std::array<float, 4>>> slews, net_delays;
    std::vector<std::tuple<int, int, std::array<float, 4>>> cell_delays;
    float time_to_internal = 1.0f / (gtdb.time_unit * 1e9f);
    logger.info("[read_infer] time_unit=%.6e, time_to_internal=%.6e\n", gtdb.time_unit, time_to_internal);

    std::string line;
    int section = 0;  // 0=none, 1=node, 2=edge
    int line_count = 0;

    while (std::getline(fin, line)) {
        line = trim(line);
        line_count++;
        if (line.empty()) continue;
        if (line.find("# Node predictions") == 0) { section = 1; logger.info("[read_infer] Entering section 1: Node predictions (line %d)\n", line_count); continue; }
        if (line.find("# Cell edge predictions") == 0) { section = 2; logger.info("[read_infer] Entering section 2: Cell edge predictions (line %d)\n", line_count); continue; }
        if (line[0] == '#') continue;

        std::vector<float> values;
        if (section == 1) {
            // node_id, slew_er, slew_ef, slew_lr, slew_lf, net_delay_er, ef, lr, lf
            if (!parse_csv_line(line, values) || values.size() < 9) continue;
            int node_id = static_cast<int>(values[0]);
            std::array<float, 4> slew   = {{values[1], values[2], values[3], values[4]}};
            std::array<float, 4> delays = {{values[5], values[6], values[7], values[8]}};
            slews.push_back({node_id, slew});
            net_delays.push_back({node_id, delays});
            if (slews.size() <= 3 || slews.size() % 10000 == 0)
                logger.info("[read_infer] Node %d: slew=[%.6e,%.6e,%.6e,%.6e] ns, net_delay=[%.6e,%.6e,%.6e,%.6e] ns (total: %zu)\n",
                    node_id, slew[0], slew[1], slew[2], slew[3], delays[0], delays[1], delays[2], delays[3], slews.size());
        } else if (section == 2) {
            // edge_id, from_pin_id, to_pin_id, cell_delay_er, ef, lr, lf
            if (!parse_csv_line(line, values) || values.size() < 7) continue;
            int from_pin_id = static_cast<int>(values[1]);
            int to_pin_id   = static_cast<int>(values[2]);
            std::array<float, 4> delays = {{values[3], values[4], values[5], values[6]}};
            cell_delays.push_back({from_pin_id, to_pin_id, delays});
            if (cell_delays.size() <= 3 || cell_delays.size() % 5000 == 0)
                logger.info("[read_infer] Edge %d: %d->%d, delays=[%.6e,%.6e,%.6e,%.6e] ns (total: %zu)\n",
                    (int)values[0], from_pin_id, to_pin_id, delays[0], delays[1], delays[2], delays[3], cell_delays.size());
        }
    }
    fin.close();
    logger.info("[read_infer] File parsing complete. Parsed %zu slews, %zu net delays, %zu cell delays\n",
                slews.size(), net_delays.size(), cell_delays.size());

    apply_infer_data(slews, net_delays, cell_delays, time_to_internal);

    logger.info("[read_infer] ========== COMPLETE ==========\n");
    logger.info("[read_infer] Loaded inference: %zu slews, %zu net delays, %zu cell delays from %s\n",
                slews.size(), net_delays.size(), cell_delays.size(), infile.c_str());
    logger.info("[read_infer] ================================\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// read_opr_gt_infer: OpenROAD ground-truth CSV where node_id is an OPR-internal
// ID, not a GPUTimer pin ID. Mapping is done via pin_name lookup.
// Also stores GT AT values in host_pinGT_AT for later R² comparison.
// ─────────────────────────────────────────────────────────────────────────────

}  // namespace gt
