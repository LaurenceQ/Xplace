#include <zlib.h>
#include <algorithm>
#include <iostream>
#include <unordered_set>

#include "Liberty.h"
#include "Timing.h"
#include "common/db/Cell.h"
#include "common/db/Database.h"
#include "Lut.h"

using std::ifstream;

namespace gt {

static std::string liberty_next_string(CellLib::token_iterator& itr,
                                       const CellLib::token_iterator end) {
    if (++itr == end) {
        return "";
    }
    return std::string(*itr);
}

static bool liberty_expr_stop_token(const std::string& token) {
    static const std::unordered_set<std::string> stop_tokens = {
        "}",
        "area", "bundle", "cell", "cell_leakage_power", "clock", "direction",
        "fall_capacitance", "fall_capacitance_range", "fall_power", "fanout_load",
        "function", "internal_power", "leakage_power", "max_capacitance",
        "max_fanout", "max_transition", "min_capacitance", "min_fanout",
        "min_transition", "pin", "related_ground_pin", "related_pg_pin",
        "related_pin", "related_power_pin", "rise_capacitance",
        "rise_capacitance_range", "rise_power", "timing", "value", "when",
        "clocked_on", "next_state", "clear", "preset", "enable", "data_in",
        "clear_preset_var1", "clear_preset_var2",
        "clock_gate_clock_pin", "clock_gate_enable_pin",
        "clock_gate_test_pin", "clock_gate_out_pin"
    };
    return stop_tokens.find(token) != stop_tokens.end();
}

static std::string liberty_next_expr(CellLib::token_iterator& itr,
                                     const CellLib::token_iterator end) {
    auto cur = itr;
    if (++cur == end) {
        return "";
    }

    std::string expr;
    int paren_depth = 0;
    for (; cur != end; ++cur) {
        const std::string token(*cur);
        if (paren_depth == 0 && liberty_expr_stop_token(token)) {
            break;
        }
        if (token == "(") {
            paren_depth++;
        } else if (token == ")") {
            paren_depth = std::max(0, paren_depth - 1);
        }
        if (!expr.empty()) expr.push_back(' ');
        expr += token;
    }

    itr = cur;
    --itr;
    return expr;
}

static float liberty_port_default_cap(const CellLib* lib, CellPortDirection direction) {
    const char* key = "default_inout_pin_cap";
    if (direction == CellPortDirection::input) {
        key = "default_input_pin_cap";
    } else if (direction == CellPortDirection::output) {
        key = "default_output_pin_cap";
    }

    auto default_itr = lib->default_values.find(key);
    auto scale_itr = lib->scale_factors.find("capacitance");
    const float scale = scale_itr == lib->scale_factors.end() ? 1.0f : scale_itr->second;
    if (default_itr != lib->default_values.end() && default_itr->second.has_value()) {
        return default_itr->second.value() * scale;
    }
    return 0.0f;
}

static void apply_liberty_port_default_caps(const CellLib* lib, LibertyPort* cell_port) {
    const float default_cap =
        cell_port->port_capacitance_[2].value_or(liberty_port_default_cap(lib, cell_port->direction_));
    if (!cell_port->port_capacitance_[2].has_value()) {
        cell_port->port_capacitance_[2] = default_cap;
    }
    for_each_el(el) {
        for (auto rf : TRAN) {
            if (!cell_port->port_capacitances_[rf][el].has_value()) {
                cell_port->port_capacitances_[rf][el] =
                    cell_port->port_capacitance_[rf].value_or(default_cap);
            }
        }
    }
}

static void copy_liberty_port_attributes(const LibertyPort* from, LibertyPort* to) {
    to->direction_ = from->direction_;
    for (int i = 0; i < 3; ++i) {
        to->port_capacitance_[i] = from->port_capacitance_[i];
    }
    for (auto rf : TRAN) {
        for_each_el(el) {
            to->port_capacitances_[rf][el] = from->port_capacitances_[rf][el];
        }
    }
    to->is_clock_ = from->is_clock_;
    to->is_clock_gate_clock_ = from->is_clock_gate_clock_;
    to->is_clock_gate_enable_ = from->is_clock_gate_enable_;
    to->is_clock_gate_test_ = from->is_clock_gate_test_;
    to->is_clock_gate_out_ = from->is_clock_gate_out_;
    to->fanout_load = from->fanout_load;
    to->max_fanout = from->max_fanout;
    to->min_fanout = from->min_fanout;
    to->max_capacitance = from->max_capacitance;
    to->min_capacitance = from->min_capacitance;
    to->max_transition = from->max_transition;
    to->min_transition = from->min_transition;
}

static float liberty_lut_var_scale(const CellLib* lib, const std::optional<LutVar>& var) {
    if (!var.has_value()) {
        return 1.0f;
    }
    if (is_capacitance_lut_var(var.value())) {
        auto itr = lib->scale_factors.find("capacitance");
        return itr == lib->scale_factors.end() ? 1.0f : itr->second;
    }
    if (is_time_lut_var(var.value())) {
        auto itr = lib->scale_factors.find("time");
        return itr == lib->scale_factors.end() ? 1.0f : itr->second;
    }
    return 1.0f;
}

LutTemplate* CellLib::get_lut_template(const std::string& name) {
    if (auto itr = lut_templates_.find(name); itr == lut_templates_.end()) {
        return nullptr;
    } else {
        return itr->second;
    }
}

std::optional<float> CellLib::extract_operating_conditions(token_iterator& itr, const token_iterator end) {
    std::optional<float> voltage;
    std::string operating_condition_name;
    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { operating_condition_name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }
    // Extract the lut template group
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find lut template group brace '{'");
    }

    int stack = 1;
    while (stack && ++itr != end) {
        // variable 1
        if (*itr == "voltage") {  // Read the variable.

            if (++itr == end) {
                logger.info("volate error in operating_conditions template %s", operating_condition_name);
            }

            voltage = std::strtof(std::string(*itr).c_str(), nullptr);
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find operating_conditions template group brace '}'");
    }

    return voltage;
}

CellLib::BusType CellLib::extract_bus_type(token_iterator& itr, const token_iterator end) {
    std::string type_name;
    BusType bus_type;
    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { type_name = name; }); itr == end) {
        logger.info("can't find bus type name");
    }
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find bus type group brace '{'");
    }

    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "bit_from") {
            logger.infoif(++itr == end, "can't get bit_from in type %s", type_name.c_str());
            bus_type.bit_from = std::atoi(itr->data());
        } else if (*itr == "bit_to") {
            logger.infoif(++itr == end, "can't get bit_to in type %s", type_name.c_str());
            bus_type.bit_to = std::atoi(itr->data());
        } else if (*itr == "bit_width") {
            logger.infoif(++itr == end, "can't get bit_width in type %s", type_name.c_str());
            bus_type.bit_width = std::atoi(itr->data());
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find bus type group brace '}'");
    }

    bus_type.valid = bus_type.bit_width > 0 || bus_type.bit_from != bus_type.bit_to;
    if (!type_name.empty()) {
        bus_types_[type_name] = bus_type;
    }
    return bus_type;
}

LutTemplate* CellLib::extract_lut_template(token_iterator& itr, const token_iterator end) {
    LutTemplate* lt = new LutTemplate();

    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { lt->name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }

    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find lut template group brace '{'");
    }

    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "variable_1") {
            if (++itr == end) {
                logger.info("variable_1 error in lut template %s", lt->name.c_str());
            }

            if (auto vitr = lut_vars.find(*itr); vitr != lut_vars.end()) {
                lt->variable1 = vitr->second;
            } else {
                logger.warning(
                    "unexpected lut template variable %.*s", static_cast<int>((*itr).length()), (*itr).data());
            }
        } else if (*itr == "variable_2") {
            if (++itr == end) {
                logger.info("variable_2 error in lut template %s", lt->name.c_str());
            }
            if (auto vitr = lut_vars.find(*itr); vitr != lut_vars.end()) {
                lt->variable2 = vitr->second;
            } else {
                logger.warning(
                    "unexpected lut template variable %.*s", static_cast<int>((*itr).length()), (*itr).data());
            }
        } else if (*itr == "index_1") {
            itr = on_next_parentheses(
                itr, end, [&](auto& str) { lt->indices1.push_back(std::strtof(str.data(), nullptr)); });
        } else if (*itr == "index_2") {
            itr = on_next_parentheses(
                itr, end, [&](auto& str) { lt->indices2.push_back(std::strtof(str.data(), nullptr)); });
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find lut template brace '}'");
    }

    lut_templates_[lt->name] = lt;

    return lt;
}

Lut* CellLib::extract_lut(token_iterator& itr, const token_iterator end) {
    Lut* lut = new Lut();

    if (itr = on_next_parentheses(itr, end, [&](auto& name) mutable { lut->name = name; }); itr == end) {
        logger.info("can't find lut template name");
    }

    lut->lut_template = get_lut_template(lut->name);
    const float index1_scale = lut->lut_template ? liberty_lut_var_scale(this, lut->lut_template->variable1) : 1.0f;
    const float index2_scale = lut->lut_template ? liberty_lut_var_scale(this, lut->lut_template->variable2) : 1.0f;
    const float table_scale = scale_factors["time"];

    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("group brace '{' error in lut ", lut->name);
    }

    int stack = 1;
    size_t size1 = 1;
    size_t size2 = 1;
    while (stack && ++itr != end) {
        if (*itr == "index_1") {
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut->indices1.push_back(std::strtof(v.data(), nullptr) * index1_scale); });

            if (lut->indices1.size() == 0) {
                logger.info("syntax error in %s index_1", lut->name);
            }

            size1 = lut->indices1.size();
        } else if (*itr == "index_2") {
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut->indices2.push_back(std::strtof(v.data(), nullptr) * index2_scale); });

            if (lut->indices2.size() == 0) {
                logger.info("syntax error in %s index_2", lut->name);
            }

            size2 = lut->indices2.size();
        } else if (*itr == "values") {
            if (lut->indices1.empty()) {
                if (size1 != 1) {
                    logger.info("empty indices1 in non-scalar lut %s", lut->name);
                }
                lut->indices1.resize(size1);
            }

            if (lut->indices2.empty()) {
                if (size2 != 1) {
                    logger.info("empty indices2 in non-scalar lut %s", lut->name);
                }
                lut->indices2.resize(size2);
            }

            lut->table.resize(size1 * size2);

            int id{0};
            itr = on_next_parentheses(
                itr, end, [&](auto& v) mutable { lut->table[id++] = std::strtof(v.data(), nullptr) * table_scale; });
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    lut->set_ = true;

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in lut ");
    }

    return lut;
}

InternalPower* CellLib::extractInternalPower(token_iterator& itr, const token_iterator end, LibertyPort* cell_port) {
    InternalPower* internal_power = new InternalPower();
    internal_power->liberty_port_ = cell_port;
    const float cap_unit = (capacitance_unit_.has_value() ? static_cast<float>(capacitance_unit_->value()) : 1.0f) *
                           scale_factors["capacitance"];
    const float voltage_unit = (voltage_unit_.has_value() ? static_cast<float>(voltage_unit_->value()) : 1.0f) *
                               scale_factors["voltage"];
    internal_power->energy_unit_ = cap_unit * voltage_unit * voltage_unit;
    cell_port->internal_powers_.push_back(internal_power);

    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in internal_power");
    }

    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "fall_power") {
            internal_power->power_[FALL] = extract_lut(itr, end);
        } else if (*itr == "rise_power") {
            internal_power->power_[RISE] = extract_lut(itr, end);
        } else if (*itr == "related_pin") {
            internal_power->related_port_name_ = liberty_next_string(itr, end);
        } else if (*itr == "related_pg_pin") {
            internal_power->related_pg_pin_name_ = liberty_next_string(itr, end);
        } else if (*itr == "when") {
            internal_power->when_expr_ = liberty_next_expr(itr, end);
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in internal_power");
    }

    return internal_power;
}

SequentialPower* CellLib::extractSequential(token_iterator& itr, const token_iterator end, LibertyCell* liberty_cell, bool is_latch) {
    SequentialPower* seq = new SequentialPower();
    seq->is_latch_ = is_latch;
    std::vector<std::string> names;
    on_next_parentheses(itr, end, [&](auto& name) mutable { names.emplace_back(name); });
    if (!names.empty()) seq->output_name_ = names[0];
    if (names.size() > 1) seq->output_inv_name_ = names[1];
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in sequential");
    }
    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "clocked_on") {
            seq->clocked_on_expr_ = liberty_next_expr(itr, end);
        } else if (*itr == "next_state") {
            seq->next_state_expr_ = liberty_next_expr(itr, end);
        } else if (*itr == "data_in") {
            seq->next_state_expr_ = liberty_next_expr(itr, end);
        } else if (*itr == "enable") {
            seq->enable_expr_ = liberty_next_expr(itr, end);
        } else if (*itr == "clear") {
            seq->clear_expr_ = liberty_next_expr(itr, end);
        } else if (*itr == "preset") {
            seq->preset_expr_ = liberty_next_expr(itr, end);
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }
    liberty_cell->sequentials_.push_back(seq);
    liberty_cell->is_seq_ = true;
    liberty_cell->num_bits_ = std::max(liberty_cell->num_bits_, 1);
    return seq;
}

TimingArc* CellLib::extractTimingArc(token_iterator& itr, const token_iterator end, LibertyPort* cell_port) {
    TimingArc* timing_arc = new TimingArc();
    timing_arc->liberty_port_ = cell_port;
    cell_port->timing_arcs_.push_back(timing_arc);

    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in timing");
    }
    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "cell_fall") {
            timing_arc->cell_delay_[1] = extract_lut(itr, end);
        } else if (*itr == "cell_rise") {
            timing_arc->cell_delay_[0] = extract_lut(itr, end);
        } else if (*itr == "fall_transition") {
            timing_arc->transition_[1] = extract_lut(itr, end);
        } else if (*itr == "rise_transition") {
            timing_arc->transition_[0] = extract_lut(itr, end);
        } else if (*itr == "fall_constraint") {
            timing_arc->constraint_[1] = extract_lut(itr, end);
        } else if (*itr == "rise_constraint") {
            timing_arc->constraint_[0] = extract_lut(itr, end);
        } else if (*itr == "timing_sense") {
            logger.infoif(++itr == end, "can't get the timing_sense in cellpin ");
            timing_arc->timing_sense_ = findTimingSense(string(*itr));
        } else if (*itr == "timing_type") {
            logger.infoif(++itr == end, "can't get the timing_type in cellpin ");
            timing_arc->timing_type_ = findTimingType(string(*itr));
        } else if (*itr == "sdf_cond") {
            logger.infoif(++itr == end, "can't get the sdf_cond in cellpin ");
            timing_arc->sdf_cond_ = *itr;
            timing_arc->is_cond_ = true;
        } else if (*itr == "related_pin") {
            logger.infoif(++itr == end, "can't get the related port ");
            timing_arc->related_port_name_ = *itr;
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in cell timing ");
    }

    return timing_arc;
}

LibertyPort* CellLib::extractLibertyPort(token_iterator& itr, const token_iterator end, LibertyCell* liberty_cell) {
    LibertyPort* cell_port = new LibertyPort();
    cell_port->cell_ = liberty_cell;

    on_next_parentheses(itr, end, [&](auto& name) mutable { cell_port->name = name; });
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in port");
    }

    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "direction") {
            logger.infoif(++itr == end, "can't get direction in cell ", cell_port->name);
            cell_port->direction_ = findPortDirection(string(*itr));
        } else if (*itr == "capacitance") {
            logger.infoif(++itr == end, "can't get the capacitance in cellpin");
            cell_port->port_capacitance_[2] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
        } else if (*itr == "fall_capacitance") {
            logger.infoif(++itr == end, "can't get fall_capacitance in cellpin");
            cell_port->port_capacitance_[FALL] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
        } else if (*itr == "rise_capacitance") {
            logger.infoif(++itr == end, "can't get rise_capacitance in cellpin");
            cell_port->port_capacitance_[RISE] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
        } else if (*itr == "rise_capacitance_range") {
            logger.infoif(++itr == end, "can't get rise_capacitance_range in cellpin");
            ++itr;
            cell_port->port_capacitances_[RISE][MIN] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
            ++itr;
            cell_port->port_capacitances_[RISE][MAX] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
            ++itr;
        } else if (*itr == "fall_capacitance_range") {
            logger.infoif(++itr == end, "can't get fall_capacitance_range in cellpin");
            ++itr;
            cell_port->port_capacitances_[FALL][MIN] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
            ++itr;
            cell_port->port_capacitances_[FALL][MAX] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
            ++itr;
        } else if (*itr == "max_capacitance") {
            logger.infoif(++itr == end, "can't get the max_capacitance in cellpin");
            cell_port->max_capacitance = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
        } else if (*itr == "min_capacitance") {
            logger.infoif(++itr == end, "can't get the min_capacitance in cellpin");
            cell_port->min_capacitance = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
        } else if (*itr == "max_transition") {
            logger.infoif(++itr == end, "can't get the max_transition in cellpin");
            cell_port->max_transition = std::strtof(itr->data(), nullptr) * scale_factors["time"];
        } else if (*itr == "min_transition") {
            logger.infoif(++itr == end, "can't get the min_transition in cellpin");
            cell_port->min_transition = std::strtof(itr->data(), nullptr) * scale_factors["time"];
        } else if (*itr == "fanout_load") {
            logger.infoif(++itr == end, "can't get fanout_load in cellpin");
            cell_port->fanout_load = std::strtof(itr->data(), nullptr);
        } else if (*itr == "max_fanout") {
            logger.infoif(++itr == end, "can't get max_fanout in cellpin");
            cell_port->max_fanout = std::strtof(itr->data(), nullptr);
        } else if (*itr == "min_fanout") {
            logger.infoif(++itr == end, "can't get min_fanout in cellpin");
            cell_port->min_fanout = std::strtof(itr->data(), nullptr);
        } else if (*itr == "clock") {
            logger.infoif(++itr == end, "can't get the clock status in cellpin");
            cell_port->is_clock_ = (*itr == "true") ? true : false;
        } else if (*itr == "clock_gate_clock_pin") {
            logger.infoif(++itr == end, "can't get the clock gate clock status in cellpin");
            cell_port->is_clock_gate_clock_ = (*itr == "true") ? true : false;
        } else if (*itr == "clock_gate_enable_pin") {
            logger.infoif(++itr == end, "can't get the clock gate enable status in cellpin");
            cell_port->is_clock_gate_enable_ = (*itr == "true") ? true : false;
        } else if (*itr == "clock_gate_test_pin") {
            logger.infoif(++itr == end, "can't get the clock gate test status in cellpin");
            cell_port->is_clock_gate_test_ = (*itr == "true") ? true : false;
        } else if (*itr == "clock_gate_out_pin") {
            logger.infoif(++itr == end, "can't get the clock gate output status in cellpin");
            cell_port->is_clock_gate_out_ = (*itr == "true") ? true : false;
        } else if (*itr == "function") {
            cell_port->function_expr_ = liberty_next_expr(itr, end);
            cell_port->has_function_ = true;
        } else if (*itr == "related_power_pin") {
            cell_port->related_power_pin_name_ = liberty_next_string(itr, end);
        } else if (*itr == "related_ground_pin") {
            cell_port->related_ground_pin_name_ = liberty_next_string(itr, end);
        } else if (*itr == "internal_power") {
            InternalPower* internal_power_ = extractInternalPower(itr, end, cell_port);
        } else if (*itr == "timing") {
            TimingArc* timing_arc_ = extractTimingArc(itr, end, cell_port);
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in cell port");
    }

    apply_liberty_port_default_caps(this, cell_port);

    return cell_port;
}

vector<LibertyPort*> CellLib::extractLibertyBus(token_iterator& itr, const token_iterator end, LibertyCell* liberty_cell) {
    LibertyPort bus_port;
    bus_port.cell_ = liberty_cell;
    bus_port.is_bundle_ = true;
    std::string bus_type_name;

    on_next_parentheses(itr, end, [&](auto& name) mutable { bus_port.name = name; });
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in bus");
    }

    int stack = 1;
    while (stack && ++itr != end) {
        if (*itr == "bus_type") {
            logger.infoif(++itr == end, "can't get bus_type in bus %s", bus_port.name.c_str());
            bus_type_name = string(*itr);
        } else if (*itr == "direction") {
            logger.infoif(++itr == end, "can't get direction in bus %s", bus_port.name.c_str());
            bus_port.direction_ = findPortDirection(string(*itr));
        } else if (*itr == "capacitance") {
            logger.infoif(++itr == end, "can't get the capacitance in bus");
            bus_port.port_capacitance_[2] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
        } else if (*itr == "fall_capacitance") {
            logger.infoif(++itr == end, "can't get fall_capacitance in bus");
            bus_port.port_capacitance_[FALL] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
        } else if (*itr == "rise_capacitance") {
            logger.infoif(++itr == end, "can't get rise_capacitance in bus");
            bus_port.port_capacitance_[RISE] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
        } else if (*itr == "rise_capacitance_range") {
            logger.infoif(++itr == end, "can't get rise_capacitance_range in bus");
            ++itr;
            bus_port.port_capacitances_[RISE][MIN] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
            ++itr;
            bus_port.port_capacitances_[RISE][MAX] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
            ++itr;
        } else if (*itr == "fall_capacitance_range") {
            logger.infoif(++itr == end, "can't get fall_capacitance_range in bus");
            ++itr;
            bus_port.port_capacitances_[FALL][MIN] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
            ++itr;
            bus_port.port_capacitances_[FALL][MAX] = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
            ++itr;
        } else if (*itr == "max_capacitance") {
            logger.infoif(++itr == end, "can't get the max_capacitance in bus");
            bus_port.max_capacitance = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
        } else if (*itr == "min_capacitance") {
            logger.infoif(++itr == end, "can't get the min_capacitance in bus");
            bus_port.min_capacitance = std::strtof(itr->data(), nullptr) * scale_factors["capacitance"];
        } else if (*itr == "max_transition") {
            logger.infoif(++itr == end, "can't get the max_transition in bus");
            bus_port.max_transition = std::strtof(itr->data(), nullptr) * scale_factors["time"];
        } else if (*itr == "min_transition") {
            logger.infoif(++itr == end, "can't get the min_transition in bus");
            bus_port.min_transition = std::strtof(itr->data(), nullptr) * scale_factors["time"];
        } else if (*itr == "fanout_load") {
            logger.infoif(++itr == end, "can't get fanout_load in bus");
            bus_port.fanout_load = std::strtof(itr->data(), nullptr);
        } else if (*itr == "max_fanout") {
            logger.infoif(++itr == end, "can't get max_fanout in bus");
            bus_port.max_fanout = std::strtof(itr->data(), nullptr);
        } else if (*itr == "min_fanout") {
            logger.infoif(++itr == end, "can't get min_fanout in bus");
            bus_port.min_fanout = std::strtof(itr->data(), nullptr);
        } else if (*itr == "clock") {
            logger.infoif(++itr == end, "can't get the clock status in bus");
            bus_port.is_clock_ = (*itr == "true") ? true : false;
        } else if (*itr == "clock_gate_clock_pin") {
            logger.infoif(++itr == end, "can't get the clock gate clock status in bus");
            bus_port.is_clock_gate_clock_ = (*itr == "true") ? true : false;
        } else if (*itr == "clock_gate_enable_pin") {
            logger.infoif(++itr == end, "can't get the clock gate enable status in bus");
            bus_port.is_clock_gate_enable_ = (*itr == "true") ? true : false;
        } else if (*itr == "clock_gate_test_pin") {
            logger.infoif(++itr == end, "can't get the clock gate test status in bus");
            bus_port.is_clock_gate_test_ = (*itr == "true") ? true : false;
        } else if (*itr == "clock_gate_out_pin") {
            logger.infoif(++itr == end, "can't get the clock gate output status in bus");
            bus_port.is_clock_gate_out_ = (*itr == "true") ? true : false;
        } else if (*itr == "timing") {
            extractTimingArc(itr, end, &bus_port);
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in bus");
    }

    apply_liberty_port_default_caps(this, &bus_port);

    auto type_itr = bus_types_.find(bus_type_name);
    if (type_itr == bus_types_.end() || !type_itr->second.valid) {
        logger.warning("bus %s has unknown or invalid bus_type %s",
                       bus_port.name.c_str(),
                       bus_type_name.c_str());
        return {};
    }

    const BusType& bus_type = type_itr->second;
    const int step = bus_type.bit_from >= bus_type.bit_to ? -1 : 1;
    vector<LibertyPort*> member_ports;
    for (int bit = bus_type.bit_from;; bit += step) {
        LibertyPort* member = new LibertyPort();
        member->name = bus_port.name + "[" + std::to_string(bit) + "]";
        member->cell_ = liberty_cell;
        copy_liberty_port_attributes(&bus_port, member);
        for (TimingArc* bus_arc : bus_port.timing_arcs_) {
            TimingArc* timing_arc = new TimingArc(*bus_arc);
            timing_arc->liberty_port_ = member;
            timing_arc->from_port_ = nullptr;
            timing_arc->to_port_ = nullptr;
            timing_arc->encode_str_.clear();
            member->timing_arcs_.push_back(timing_arc);
        }
        member_ports.push_back(member);
        if (bit == bus_type.bit_to) {
            break;
        }
    }

    return member_ports;
}

LibertyCell* CellLib::extractLibertyCell(token_iterator& itr, const token_iterator end) {
    LibertyCell* liberty_cell = new LibertyCell();
    for (auto rf : TRAN) {
        liberty_cell->input_threshold_pct[rf] = input_threshold_pct[rf];
        liberty_cell->output_threshold_pct[rf] = output_threshold_pct[rf];
        liberty_cell->slew_lower_threshold_pct[rf] = slew_lower_threshold_pct[rf];
        liberty_cell->slew_upper_threshold_pct[rf] = slew_upper_threshold_pct[rf];
    }
    liberty_cell->slew_derate_from_library = slew_derate_from_library;

    on_next_parentheses(itr, end, [&](auto& name) mutable { liberty_cell->name = name; });
    if (itr = std::find(itr, end, "{"); itr == end) {
        logger.info("can't find group brace '{' in cell %s", liberty_cell->name);
    }

    int stack = 1;
    int stage = -1;
    while (stack && ++itr != end) {
        if (*itr == "cell_leakage_power") {
            logger.infoif(++itr == end, "can't get the cell_leakage_power ");
            liberty_cell->leakage_power_ = scale_factors["power"] * std::strtof(itr->data(), nullptr);
        }
        if (*itr == "leakage_power") {
            LeakagePower* leakage_power = new LeakagePower();
            itr = std::find(itr, end, "{");
            int stack_1 = 1;
            while (stack_1 && ++itr != end) {
                if (*itr == "value") {
                    logger.infoif(++itr == end, "can't get value in cell %s", liberty_cell->name);
                    leakage_power->value_ = scale_factors["power"] * std::strtof(itr->data(), nullptr);
                    liberty_cell->leakage_powers_.push_back(leakage_power->value_);
                } else if (*itr == "when") {
                    leakage_power->when_expr_ = liberty_next_expr(itr, end);
                } else if (*itr == "related_pg_pin") {
                    leakage_power->related_pg_pin_name_ = liberty_next_string(itr, end);
                } else if (*itr == "}")
                    stack_1--;
                else if (*itr == "{")
                    stack_1++;
            }
            liberty_cell->leakage_power_groups_.push_back(leakage_power);
        } else if (*itr == "area") {
            logger.infoif(++itr == end, "can't get area in cell %s", liberty_cell->name);
            liberty_cell->area_ = std::strtof(itr->data(), nullptr);
        } else if (*itr == "ff") {
            extractSequential(itr, end, liberty_cell, false);
        } else if (*itr == "latch") {
            extractSequential(itr, end, liberty_cell, true);
        } else if (*itr == "pin") {
            logger.infoif(++itr == end, "can't get port in cell %s", liberty_cell->name);
            LibertyPort* cell_port_ = extractLibertyPort(itr, end, liberty_cell);
            liberty_cell->ports_.push_back(cell_port_);
        } else if (*itr == "bus") {
            vector<LibertyPort*> bus_ports = extractLibertyBus(itr, end, liberty_cell);
            liberty_cell->ports_.insert(liberty_cell->ports_.end(), bus_ports.begin(), bus_ports.end());
        } else if (*itr == "bundle") {
            LibertyPort* cell_port_bundle = new LibertyPort();
            liberty_cell->ports_.push_back(cell_port_bundle);
            cell_port_bundle->cell_ = liberty_cell;
            cell_port_bundle->is_bundle_ = true;

            on_next_parentheses(itr, end, [&](auto& name) mutable { cell_port_bundle->name = name; });
            itr = std::find(itr, end, "{");

            int stack_1 = 1;
            while (stack_1 && ++itr != end) {
                if (*itr == "direction") {
                    logger.infoif(++itr == end, "can't get direction in cell %s", liberty_cell->name);
                    cell_port_bundle->direction_ = findPortDirection(string(*itr));
                } else if (*itr == "pin") {
                    LibertyPort* cell_port_ = extractLibertyPort(itr, end, liberty_cell);
                    cell_port_bundle->member_ports_.push_back(cell_port_);
                } else if (*itr == "}")
                    stack_1--;
                else if (*itr == "{")
                    stack_1++;
            }
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }

    if (stack != 0 || *itr != "}") {
        logger.info("can't find group brace '}' in cellpin ");
    }
    return liberty_cell;
}

void CellLib::read(const std::string& file) {
    // process .gz file with zlib
    std::vector<char> buffer;
    if (file.substr(file.find_last_of(".") + 1) == "gz") {
        logger.info("reading gzip celllib %s ...", file.c_str());
        gzFile fs = gzopen(file.c_str(), "rb");
        if (!fs) {
            logger.error("cannot open verilog file: %s", file.c_str());
        }
        char buf[1024];
        int len = 0;
        while ((len = gzread(fs, buf, 1024)) > 0) {
            buffer.insert(buffer.end(), buf, buf + len);
        }
        gzclose(fs);
        buffer.push_back(0);
    } else {
        ifstream fs(file.c_str(), std::ios::ate);
        if (!fs.good()) {
            logger.error("cannot open liberty file: %s", file.c_str());
        }
        logger.info("reading celllib %s ...", file.c_str());

        size_t fsize = fs.tellg();
        fs.seekg(0, std::ios::beg);
        buffer.resize(fsize + 1);
        fs.read(buffer.data(), fsize);
        buffer[fsize] = 0;
    }

    // get tokens
    std::vector<std::string_view> tokens;
    tokens.reserve(buffer.size() / sizeof(std::string));

    uncomment(buffer);
    tokenize(buffer, tokens);

    // Set up the iterator
    auto itr = tokens.begin();
    auto end = tokens.end();

    // Read the library name.
    if (itr = std::find(itr, end, "library"); itr == end) {
        logger.error("can't find keyword %s", "library");
    }

    if (itr = on_next_parentheses(itr, end, [&](auto& str) mutable { name = str; }); itr == end) {
        logger.info("can't find library name");
    }

    if (itr = std::find(itr, tokens.end(), "{"); itr == tokens.end()) {
        logger.info("can't find library group symbol '{'");
    }

    int stack = 1;

    while (stack && ++itr != end) {
        if (*itr == "lu_table_template") {
            auto lut = extract_lut_template(itr, end);
        } else if (*itr == "power_lut_template") {
            auto lut = extract_lut_template(itr, end);
        } else if (*itr == "type") {
            extract_bus_type(itr, end);
        } else if (*itr == "delay_model") {
            logger.infoif(++itr == end, "syntax error in delay_model");
            delay_model = findDelayModel(string(*itr));
        } else if (*itr == "input_threshold_pct_fall") {
            logger.infoif(++itr == end, "syntax error in input_threshold_pct_fall");
            input_threshold_pct[FALL] = std::strtof(itr->data(), nullptr) / 100.0f;
        } else if (*itr == "input_threshold_pct_rise") {
            logger.infoif(++itr == end, "syntax error in input_threshold_pct_rise");
            input_threshold_pct[RISE] = std::strtof(itr->data(), nullptr) / 100.0f;
        } else if (*itr == "output_threshold_pct_fall") {
            logger.infoif(++itr == end, "syntax error in output_threshold_pct_fall");
            output_threshold_pct[FALL] = std::strtof(itr->data(), nullptr) / 100.0f;
        } else if (*itr == "output_threshold_pct_rise") {
            logger.infoif(++itr == end, "syntax error in output_threshold_pct_rise");
            output_threshold_pct[RISE] = std::strtof(itr->data(), nullptr) / 100.0f;
        } else if (*itr == "slew_lower_threshold_pct_fall") {
            logger.infoif(++itr == end, "syntax error in slew_lower_threshold_pct_fall");
            slew_lower_threshold_pct[FALL] = std::strtof(itr->data(), nullptr) / 100.0f;
        } else if (*itr == "slew_lower_threshold_pct_rise") {
            logger.infoif(++itr == end, "syntax error in slew_lower_threshold_pct_rise");
            slew_lower_threshold_pct[RISE] = std::strtof(itr->data(), nullptr) / 100.0f;
        } else if (*itr == "slew_upper_threshold_pct_fall") {
            logger.infoif(++itr == end, "syntax error in slew_upper_threshold_pct_fall");
            slew_upper_threshold_pct[FALL] = std::strtof(itr->data(), nullptr) / 100.0f;
        } else if (*itr == "slew_upper_threshold_pct_rise") {
            logger.infoif(++itr == end, "syntax error in slew_upper_threshold_pct_rise");
            slew_upper_threshold_pct[RISE] = std::strtof(itr->data(), nullptr) / 100.0f;
        } else if (*itr == "slew_derate_from_library") {
            logger.infoif(++itr == end, "syntax error in slew_derate_from_library");
            slew_derate_from_library = std::strtof(itr->data(), nullptr);
        } else if (*itr == "default_cell_leakage_power" || *itr == "default_inout_pin_cap" ||
                   *itr == "default_input_pin_cap" || *itr == "default_output_pin_cap" ||
                   *itr == "default_fanout_load" || *itr == "default_max_fanout" || *itr == "default_max_transition") {
            std::string attr_name(*itr);
            logger.infoif(++itr == end, "syntax error");
            default_values[attr_name] = std::strtof(itr->data(), nullptr);
        } else if (*itr == "operating_conditions") {
            logger.infoif(++itr == end, "syntax error");
            default_values["voltage"] = extract_operating_conditions(itr, end);
        } else if (*itr == "time_unit") {
            logger.infoif(++itr == end, "syntax error");
            auto current_time_unit = make_time_unit(*itr);
            if (!time_unit_) time_unit_ = current_time_unit;
            scale_factors["time"] = *current_time_unit / *time_unit_;
        } else if (*itr == "voltage_unit") {
            logger.infoif(++itr == end, "syntax error");
            auto current_voltage_unit = make_voltage_unit(*itr);
            if (!voltage_unit_) voltage_unit_ = current_voltage_unit;
            scale_factors["voltage"] = *current_voltage_unit / *voltage_unit_;
        } else if (*itr == "current_unit") {
            logger.infoif(++itr == end, "syntax error");
            auto current_current_unit = make_current_unit(*itr);
            if (!current_unit_) current_unit_ = current_current_unit;
            scale_factors["current"] = *current_current_unit / *current_unit_;
        } else if (*itr == "pulling_resistance_unit") {
            logger.infoif(++itr == end, "syntax error");
            auto current_resistance_unit = make_resistance_unit(*itr);
            if (!resistance_unit_) resistance_unit_ = current_resistance_unit;
            scale_factors["resistance"] = *current_resistance_unit / *resistance_unit_;
        } else if (*itr == "capacitive_load_unit") {
            string unit;
            on_next_parentheses(itr, end, [&](auto& str) mutable { unit += str; });
            auto current_capacitance_unit = make_capacitance_unit(unit);
            if (!capacitance_unit_) capacitance_unit_ = current_capacitance_unit;
            scale_factors["capacitance"] = *current_capacitance_unit / *capacitance_unit_;
        } else if (*itr == "leakage_power_unit") {
            logger.infoif(++itr == end, "syntax error");
            auto current_power_unit_ = make_power_unit(*itr);
            if (!power_unit_) power_unit_ = current_power_unit_;
            scale_factors["power"] = *current_power_unit_ / *power_unit_;
        } else if (*itr == "cell") {
            LibertyCell* libterty_cell = extractLibertyCell(itr, end);
            lib_cells_[libterty_cell->name] = libterty_cell;
        } else if (*itr == "}") {
            stack--;
        } else if (*itr == "{") {
            stack++;
        } else {
            // undefined token TODO:
        }
    }
    logger.info("Liberty thresholds %s: in(r/f)=%.3f/%.3f out(r/f)=%.3f/%.3f slew(r/f)=%.3f-%.3f/%.3f-%.3f derate=%.3f",
                name.c_str(),
                input_threshold_pct[RISE],
                input_threshold_pct[FALL],
                output_threshold_pct[RISE],
                output_threshold_pct[FALL],
                slew_lower_threshold_pct[RISE],
                slew_upper_threshold_pct[RISE],
                slew_lower_threshold_pct[FALL],
                slew_upper_threshold_pct[FALL],
                slew_derate_from_library);
    if (!default_thresholds_initialized) {
        for (auto rf : TRAN) {
            default_input_threshold_pct[rf] = input_threshold_pct[rf];
            default_output_threshold_pct[rf] = output_threshold_pct[rf];
            default_slew_lower_threshold_pct[rf] = slew_lower_threshold_pct[rf];
            default_slew_upper_threshold_pct[rf] = slew_upper_threshold_pct[rf];
        }
        default_slew_derate_from_library = slew_derate_from_library;
        default_thresholds_initialized = true;
    }
}

void CellLib::finish_port_read(LibertyPort* liberty_port) {
    for (InternalPower* internal_power : liberty_port->internal_powers_) {
        if (!internal_power->related_port_name_.empty()) {
            int related_port = liberty_port->cell_->get_port(internal_power->related_port_name_);
            if (related_port == -1) {
                logger.warning("internal_power %s.%s has no related pin %s",
                               liberty_port->cell_->name.c_str(),
                               liberty_port->name.c_str(),
                               internal_power->related_port_name_.c_str());
            } else {
                internal_power->related_port_ = liberty_port->cell_->ports_[related_port];
            }
        }
        if (!internal_power->related_pg_pin_name_.empty()) {
            int related_pg_pin = liberty_port->cell_->get_port(internal_power->related_pg_pin_name_);
            if (related_pg_pin != -1) {
                internal_power->related_pg_pin_ = liberty_port->cell_->ports_[related_pg_pin];
            }
        }
    }

    for (TimingArc* timing_arc : liberty_port->timing_arcs_) {
        if (timing_arc->related_port_name_.empty()) {
            logger.warning("timing arc %s.%s.%s has no related pin",
                           liberty_port->cell_->name.c_str(),
                           liberty_port->name.c_str(),
                           timing_arc->timing_type_);
            continue;
        }
        if (auto related_port = liberty_port->cell_->get_port(timing_arc->related_port_name_);
            related_port == -1) {
            logger.warning("timing arc %s.%s.%s has no related pin",
                           liberty_port->cell_->name.c_str(),
                           liberty_port->name.c_str(),
                           timing_arc->timing_type_);
        } else {
            timing_arc->from_port_ = liberty_port->cell_->ports_[related_port];;
        }
        timing_arc->to_port_ = timing_arc->liberty_port_;
    }

    for (TimingArc* timing_arc : liberty_port->timing_arcs_) {
        timing_arc->encode_str_ = timing_arc->encode_arc();
        if (liberty_port->timing_arcs_map_.find(timing_arc->encode_str_) != liberty_port->timing_arcs_map_.end()) {
            TimingArc* old_timing_arc = liberty_port->timing_arcs_map_[timing_arc->encode_str_];
            if (!timing_arc->is_cond_) {
                liberty_port->timing_arcs_map_[timing_arc->encode_str_] = timing_arc;
            }
        } else
            liberty_port->timing_arcs_map_[timing_arc->encode_str_] = timing_arc;
    }
}

void CellLib::finish_read() {
    for (auto [name, liberty_cell] : lib_cells_) {
        db::CellType* lef_cell_type = rawdb->getCellType(name);
        if (lef_cell_type == nullptr) {
            logger.warning("cell %s not found in lef", name.c_str());
            continue;
        } else {
            lef_cell_type->liberty_cell = liberty_cell;
            liberty_cell->cell_type_ = lef_cell_type;
        }
        // sort port by name
        std::sort(liberty_cell->ports_.begin(),
                  liberty_cell->ports_.end(),
                  [](const LibertyPort* a, const LibertyPort* b) { return a->name < b->name; });
        
        for (int i = 0; i < liberty_cell->ports_.size(); i++) {
            liberty_cell->ports_map_[liberty_cell->ports_[i]->name] = i;
        }

        for (auto leakage_power : liberty_cell->leakage_power_groups_) {
            if (!leakage_power->related_pg_pin_name_.empty()) {
                int related_pg_pin = liberty_cell->get_port(leakage_power->related_pg_pin_name_);
                if (related_pg_pin != -1) {
                    leakage_power->related_pg_pin_ = liberty_cell->ports_[related_pg_pin];
                }
            }
        }

        for (auto port : liberty_cell->ports_) {
            if (port->is_clock_) liberty_cell->is_seq_ = true;
            if (port->is_bundle_) {
                for (auto member_port : port->member_ports_) {
                    finish_port_read(member_port);
                }
            } else
                finish_port_read(port);
        }

        for (auto port : liberty_cell->ports_) {
            LibertyPort* non_bundle_port;
            if (port->is_bundle_) {
                non_bundle_port = port->member_ports_[0];
            } else
                non_bundle_port = port;
            // OpenSTA keeps conditional timing arcs as separate alternatives.  They
            // must all participate in min/max propagation; collapsing by
            // encode_arc() can pick the wrong Liberty table for paths like
            // AOI/OAI cells with multiple "when" arcs on the same input.
            for (TimingArc* timing_arc : non_bundle_port->timing_arcs_) {
                port->timing_arcs_non_cond_non_bundle_.push_back(timing_arc);
            }
        }
    }
}

};  // namespace gt
