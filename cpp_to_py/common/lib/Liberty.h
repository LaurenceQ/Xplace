

#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "EnumNameMap.h"
#include "Helper.h"
#include "tokenizer.h"
#include "unit.h"

using std::optional;
using std::string;
using std::unordered_map;
using std::variant;
using std::vector;

namespace db {
class Database;
class CellType;
};  // namespace db

namespace gt {

class CellLib;
class LibertyCell;
class LibertyPort;
class TimingArc;
struct LutTemplate;
class Lut;
class InternalPower;
class LeakagePower;
class SequentialPower;

enum class DelayModel { generic_cmos, table_lookup, cmos2, piecewise_cmos, dcm, polynomial, unknown };
enum class CellPortDirection { input, output, inout, internal, unknown };
DelayModel findDelayModel(const std::string model_name);
CellPortDirection findPortDirection(const std::string dir_name);

enum class PowerExprOpcode : uint8_t { port, const_zero, const_one, logical_not, logical_and, logical_or, logical_xor };

struct PowerExprOp {
    PowerExprOpcode opcode = PowerExprOpcode::const_zero;
    int port_id = -1;
};

class PowerExpr {
public:
    bool compile(const string& expr, const LibertyCell* cell);
    int8_t eval(const vector<int8_t>& port_values) const;
    bool valid() const { return valid_; }
    const string& source() const { return source_; }
    const vector<PowerExprOp>& ops() const { return ops_; }

private:
    string source_;
    vector<PowerExprOp> ops_;
    bool valid_ = false;
};

class CellLib {
public:
    ~CellLib() { logger.info("Destruct celllib"); }
    CellLib() = default;
    CellLib(db::Database *rawdb_) : rawdb(rawdb_) {}
    db::Database *rawdb = nullptr;
    db::CellType *cell_type_ = nullptr;

    using token_iterator = std::vector<std::string_view>::iterator;
    struct BusType {
        int bit_from = 0;
        int bit_to = -1;
        int bit_width = 0;
        bool valid = false;
    };
    DelayModel delay_model;
    string name;

    optional<second_t> time_unit_;
    optional<watt_t> power_unit_;
    optional<ohm_t> resistance_unit_;
    optional<farad_t> capacitance_unit_;
    optional<ampere_t> current_unit_;
    optional<volt_t> voltage_unit_;

    unordered_map<string, optional<float>> default_values = {
        {"default_cell_leakage_power", optional<float>{}},
        {"default_inout_pin_cap", optional<float>{}},
        {"default_input_pin_cap", optional<float>{}},
        {"default_output_pin_cap", optional<float>{}},
        {"default_fanout_load", optional<float>{}},
        {"default_max_fanout", optional<float>{}},
        {"default_max_transition", optional<float>{}},
        {"voltage", optional<float>{}},
    };

    unordered_map<string, float> scale_factors = {
        {"time", 1.0},
        {"resistance", 1.0},
        {"power", 1.0},
        {"capacitance", 1.0},
        {"current", 1.0},
        {"voltage", 1.0},
    };

    float input_threshold_pct[MAX_TRAN] = {0.5f, 0.5f};
    float output_threshold_pct[MAX_TRAN] = {0.5f, 0.5f};
    float slew_lower_threshold_pct[MAX_TRAN] = {0.2f, 0.2f};
    float slew_upper_threshold_pct[MAX_TRAN] = {0.8f, 0.8f};
    float slew_derate_from_library = 1.0f;
    bool default_thresholds_initialized = false;
    float default_input_threshold_pct[MAX_TRAN] = {0.5f, 0.5f};
    float default_output_threshold_pct[MAX_TRAN] = {0.5f, 0.5f};
    float default_slew_lower_threshold_pct[MAX_TRAN] = {0.2f, 0.2f};
    float default_slew_upper_threshold_pct[MAX_TRAN] = {0.8f, 0.8f};
    float default_slew_derate_from_library = 1.0f;

    unordered_map<string, LutTemplate *> lut_templates_;
    unordered_map<string, LibertyCell *> lib_cells_;
    unordered_map<string, BusType> bus_types_;

    LutTemplate *get_lut_template(const string &);
    LibertyCell *get_cell(const std::string &name);
    void read(const string &file);
    void finish_read();
    void finish_port_read(LibertyPort *liberty_port);

public:
    LibertyCell *extractLibertyCell(token_iterator &, const token_iterator);
    LibertyPort *extractLibertyPort(token_iterator &, const token_iterator, LibertyCell *);
    vector<LibertyPort *> extractLibertyBus(token_iterator &, const token_iterator, LibertyCell *);
    TimingArc *extractTimingArc(token_iterator &, const token_iterator, LibertyPort *);
    InternalPower *extractInternalPower(token_iterator &, const token_iterator, LibertyPort *);
    SequentialPower *extractSequential(token_iterator &, const token_iterator, LibertyCell *, bool is_latch);
    std::optional<float> extract_operating_conditions(token_iterator &itr, const token_iterator end);
    BusType extract_bus_type(token_iterator &, const token_iterator);
    LutTemplate *extract_lut_template(token_iterator &, const token_iterator);
    Lut *extract_lut(token_iterator &, const token_iterator);

    void apply_default_values();
    void uncomment(std::vector<char> &);
    void tokenize(const std::vector<char> &, std::vector<std::string_view> &);
};

class LeakagePower {
public:
    float value_ = 0.0f;
    string when_expr_;
    string related_pg_pin_name_;
    LibertyPort* related_pg_pin_ = nullptr;
};

class SequentialPower {
public:
    bool is_latch_ = false;
    string output_name_;
    string output_inv_name_;
    string clocked_on_expr_;
    string next_state_expr_;
    string enable_expr_;
    string clear_expr_;
    string preset_expr_;
};

class InternalPower {
public:
    LibertyPort* liberty_port_ = nullptr;  // owning/to port.
    string related_port_name_;
    LibertyPort* related_port_ = nullptr;
    string related_pg_pin_name_;
    LibertyPort* related_pg_pin_ = nullptr;
    string when_expr_;
    float energy_unit_ = 1.0f;
    Lut* power_[MAX_TRAN] = {nullptr, nullptr};  // rise/fall internal_power tables.
};

class LibertyCell {
public:
    LibertyCell() = default;
    string name;
    db:: CellType *cell_type_ = nullptr;
    vector<LibertyPort *> ports_;
    // map<string, LibertyPort *> ports_map_;
    map<string, int> ports_map_;

    vector<float> leakage_powers_;
    vector<LeakagePower*> leakage_power_groups_;
    vector<SequentialPower*> sequentials_;
    optional<float> leakage_power_;
    optional<float> area_;

    bool is_seq_ = false;
    int num_bits_ = 0;
    float input_threshold_pct[MAX_TRAN] = {0.5f, 0.5f};
    float output_threshold_pct[MAX_TRAN] = {0.5f, 0.5f};
    float slew_lower_threshold_pct[MAX_TRAN] = {0.2f, 0.2f};
    float slew_upper_threshold_pct[MAX_TRAN] = {0.8f, 0.8f};
    float slew_derate_from_library = 1.0f;

public:
    int get_port(const std::string &name) const;
};

class LibertyPort {
public:
    LibertyPort() = default;

public:
    string name;
    LibertyCell *cell_;
    CellPortDirection direction_;
    optional<float> port_capacitance_[3];
    optional<float> port_capacitances_[MAX_TRAN][MAX_SPLIT];

    bool is_clock_ = false;
    bool is_clock_gate_clock_ = false;
    bool is_clock_gate_enable_ = false;
    bool is_clock_gate_test_ = false;
    bool is_clock_gate_out_ = false;
    bool is_bundle_ = false;
    vector<LibertyPort *> member_ports_;

    vector<TimingArc *> timing_arcs_;
    map<string, TimingArc *> timing_arcs_map_;
    vector<TimingArc *> timing_arcs_non_cond_non_bundle_;
    vector<InternalPower *> internal_powers_;

    string function_expr_;
    bool has_function_ = false;
    string related_power_pin_name_;
    string related_ground_pin_name_;




    // optional<float> capacitance;
    // optional<float> fall_capacitance;
    // optional<float> rise_capacitance;

    optional<float> fanout_load;
    optional<float> max_fanout;
    optional<float> min_fanout;
    optional<float> max_capacitance;
    optional<float> min_capacitance;
    optional<float> max_transition;
    optional<float> min_transition;
};

};  // namespace gt
