

#pragma once
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

enum class DelayModel { generic_cmos, table_lookup, cmos2, piecewise_cmos, dcm, polynomial, unknown };
enum class CellPortDirection { input, output, inout, internal, unknown };
DelayModel findDelayModel(const std::string model_name);
CellPortDirection findPortDirection(const std::string dir_name);

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
    std::optional<float> extract_operating_conditions(token_iterator &itr, const token_iterator end);
    BusType extract_bus_type(token_iterator &, const token_iterator);
    LutTemplate *extract_lut_template(token_iterator &, const token_iterator);
    Lut *extract_lut(token_iterator &, const token_iterator);

    void apply_default_values();
    void uncomment(std::vector<char> &);
    void tokenize(const std::vector<char> &, std::vector<std::string_view> &);
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
    int get_port(const std::string &name);
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
    bool is_bundle_ = false;
    vector<LibertyPort *> member_ports_;

    vector<TimingArc *> timing_arcs_;
    map<string, TimingArc *> timing_arcs_map_;
    vector<TimingArc *> timing_arcs_non_cond_non_bundle_;




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
