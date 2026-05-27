#pragma once

#include "common/lib/Liberty.h"
#include "io_parser/gp/GPDatabase.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gt {

struct CpuActivity {
    float density = 0.0f;
    float duty = 0.0f;
    int origin = 0;
};

struct PowerTraceEdge {
    int arc_id = -1;
    int from_pin = -1;
    int to_pin = -1;
    std::string reason;
};

struct PowerTracePathState {
    std::unordered_set<int> pins;
    std::unordered_set<int> arcs;
    std::unordered_set<int> nodes;
    std::ofstream out;

    bool enabled() const { return out.good(); }
};

bool readPowerBoolEnv(const char* name, bool default_value);
float readPowerFloatEnv(const char* name, float default_value);
double canonicalPowerTimeScale(float scale);
float powerDensityForPeriod(double transitions, float period, double time_scale);
std::string normalizeTracePinName(std::string name);
std::string normalizePowerActivitySnapshotName(std::string name);
std::string csvEscapePowerActivitySnapshot(const std::string& value);
int readPowerActivitySnapshotMaxPass(const char* env_name, int default_value);
std::vector<std::string> readPowerTracePinQueries();
std::vector<std::string> readPowerRootProbePinQueries();
std::vector<int> resolvePowerTracePins(const std::vector<std::string>& queries,
                                       const std::vector<std::string>& pin_names);
bool parsePowerBool(const std::string& value);
std::string normalizePowerPathName(std::string name);
bool powerPathNameMatches(const std::string& pin_name, const std::string& query);
std::vector<std::string> readPowerPathTargetQueries();
std::unordered_set<std::string> readOpenroadSeedRootNames(const char* file_name);
std::vector<int> resolvePowerPathTargetPins(const std::vector<std::string>& queries,
                                            const std::vector<std::string>& pin_names);
PowerTracePathState loadPowerTracePathState(const char* trace_path_file,
                                            const char* out_file,
                                            const std::vector<int>& pin_to_node);
bool evalPowerExprWithPortValues(const PowerExpr& expr,
                                 const std::vector<int8_t>& port_values,
                                 int8_t& value);
bool evalPowerExprActivity(const PowerExpr& expr,
                           const LibertyCell* cell,
                           const gp::GPNode& node,
                           const std::vector<CpuActivity>& pin_activity,
                           float& density,
                           float& duty,
                           const std::unordered_map<int, int>* const_port_values = nullptr,
                           const std::unordered_set<int>* zero_density_ports = nullptr);
const std::string& seqClockExpr(const SequentialPower* seq);

}  // namespace gt
