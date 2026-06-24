#pragma once

#include "common/lib/Liberty.h"
#include "io_parser/gp/GPDatabase.h"

#include <cstddef>
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

struct PowerTracePathWriter {
    std::unordered_set<int> pins;
    std::unordered_set<int> arcs;
    std::unordered_set<int> nodes;
    std::ofstream out;

    bool enabled() const;
};

class PowerExprBddActivityEvaluator {
public:
    PowerExprBddActivityEvaluator(const LibertyCell* cell,
                                  const gp::GPNode& node,
                                  const std::vector<CpuActivity>& pin_activity,
                                  const std::unordered_map<int, int>* const_port_values,
                                  const std::unordered_set<int>* zero_density_ports);
    ~PowerExprBddActivityEvaluator();

    PowerExprBddActivityEvaluator(const PowerExprBddActivityEvaluator&) = delete;
    PowerExprBddActivityEvaluator& operator=(const PowerExprBddActivityEvaluator&) = delete;

    bool evaluate(const PowerExpr& expr, float& density, float& duty);

private:
    /**
     * A BDD node is one decision in the Boolean function:
     *
     *     F = (var == 0) ? low : high
     *
     * var is the evaluator's compact BDD variable id.  It maps back to the
     * Liberty port through var_ports_[var], and its activity data is stored in
     * var_duties_[var] and var_densities_[var].
     *
     * low and high are not variable ids.  They are BDD edges: integer handles
     * for the sub-expression reached when this node's var is 0 or 1.
     *
     * Edge encoding:
     *
     *     0 = constant true
     *     1 = constant false
     *     node edge = node_id << 1
     *     inverted edge = edge ^ 1
     *
     * Example 1: one AND gate, F = A & B.
     * Variable order is A -> var 0, B -> var 1.
     *
     * Boolean reasoning:
     *
     *     if A = 0: F = 0
     *     if A = 1: F = B
     *
     * BDD shape:
     *
     *             A(var 0)
     *            /        \
     *       low /          \ high
     *          v            v
     *        false        B(var 1)
     *                    /        \
     *               low /          \ high
     *                  v            v
     *                false        true
     *
     * Nodes stored by the manager:
     *
     *     B node: var = 1, low = false edge, high = true edge
     *     A node: var = 0, low = false edge, high = edge of B node
     *
     * Example 2: two gates, Y = (A & B) | C.
     * This is a gate graph first:
     *
     *     A ----\
     *            AND ----\
     *     B ----/        OR ---- Y
     *     C ------------/
     *
     * BDD does not keep the gate boundaries.  It represents the final Boolean
     * function Y directly. With variable order A, B, C:
     *
     *     if A = 0: Y = C
     *     if A = 1: Y = B | C
     *
     * BDD shape:
     *
     *             A
     *            / \
     *           /   \
     *          C     B
     *         / \   / \
     *        0   1 C   1
     *             / \
     *            0   1
     *
     * Shared subgraphs are reused: the C node under A=0 and the C node under
     * B=0 are the same logical node if their (var, low, high) are identical.
     *
     * Example 3: three gates, Y = (A & B) | (C & D).
     * Gate graph:
     *
     *     A ----\
     *            AND ----\
     *     B ----/        \
     *                     OR ---- Y
     *     C ----\        /
     *            AND ----/
     *     D ----/
     *
     * With variable order A, B, C, D:
     *
     *     if A = 0: Y = C & D
     *     if A = 1: Y = B | (C & D)
     *
     * The C&D subgraph is:
     *
     *             C
     *            / \
     *           0   D
     *              / \
     *             0   1
     *
     * The full BDD reuses that same C&D subgraph twice:
     *
     *             A
     *            / \
     *           /   \
     *        C&D     B
     *       subgraph / \
     *              C&D  1
     *             subgraph
     *
     * This is why a BDD is a graph, not a tree: identical sub-functions point
     * to the same node instead of being duplicated.
     */
    struct BddNode {
        int var = -1;
        int low = 0;
        int high = 0;
    };

    struct BddKey {
        int var = -1;
        int low = 0;
        int high = 0;

        bool operator==(const BddKey& other) const;
    };

    struct BddKeyHash {
        size_t operator()(const BddKey& key) const;
    };

    struct ApplyKey {
        int op = 0;  // 0=and, 1=or, 2=xor
        int left = 0;
        int right = 0;

        bool operator==(const ApplyKey& other) const;
    };

    struct ApplyKeyHash {
        size_t operator()(const ApplyKey& key) const;
    };

    static int oneEdge();
    static int zeroEdge();
    static int noVar();
    static int edgeId(int edge);
    static bool edgeInv(int edge);
    static int edgeNot(int edge);

    bool resolvePortPin(int port_id, int& pin_id) const;
    int makeNode(int var, int low, int high);
    int topVar(int edge) const;
    int cofTop(int edge, int var, bool high_child) const;
    int apply(int op, int left, int right);
    int restrictVar(int edge, int target_var, bool high_child);
    int ensureVar(int port_id, int pin_id);
    bool registerExprVariables(const PowerExpr& expr);
    bool pushPort(std::vector<int>& stack, int port_id);
    bool buildBdd(const PowerExpr& expr, int& root);
    bool applyBinary(std::vector<int>& stack, int op);
    float evalDuty(int edge) const;
    float evalDensity(int root);

    const LibertyCell* cell_ = nullptr;
    const gp::GPNode& node_;
    const std::vector<CpuActivity>& pin_activity_;
    const std::unordered_set<int>* zero_density_ports_ = nullptr;
    std::vector<int8_t> fixed_port_values_;

    std::vector<BddNode> nodes_;
    std::unordered_map<BddKey, int, BddKeyHash> unique_nodes_;
    std::unordered_map<int, int> port_to_var_;
    std::vector<int> var_ports_;
    std::vector<float> var_duties_;
    std::vector<float> var_densities_;
    std::vector<uint8_t> var_has_pin_;

    std::unordered_map<ApplyKey, int, ApplyKeyHash> apply_cache_;
    std::unordered_map<long long, int> restrict_cache_;
};

bool readPowerBoolEnv(const char* name, bool default_value);
float readPowerFloatEnv(const char* name, float default_value);
void resetPowerStageProfileElapsed();
void addPowerStageProfileElapsed(double elapsed);
double powerStageProfileElapsed();
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
PowerTracePathWriter loadPowerTracePathWriter(const char* trace_path_file,
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
