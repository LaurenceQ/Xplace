

#include "Liberty.h"

#include <cctype>

namespace gt {

namespace {

class PowerExprParser {
public:
    PowerExprParser(const std::string& expr, const LibertyCell* cell, std::vector<PowerExprOp>& ops)
        : expr_(expr), cell_(cell), ops_(ops) {}

    bool parse() {
        skip_spaces();
        if (!parse_or()) return false;
        skip_spaces();
        return pos_ == expr_.size() && ok_;
    }

private:
    bool parse_or() {
        if (!parse_xor()) return false;
        while (true) {
            skip_spaces();
            if (!consume('|') && !consume('+')) break;
            if (!parse_xor()) return false;
            ops_.push_back({PowerExprOpcode::logical_or, -1});
        }
        return true;
    }

    bool parse_xor() {
        if (!parse_and()) return false;
        while (true) {
            skip_spaces();
            if (!consume('^')) break;
            if (!parse_and()) return false;
            ops_.push_back({PowerExprOpcode::logical_xor, -1});
        }
        return true;
    }

    bool parse_and() {
        if (!parse_unary()) return false;
        while (true) {
            skip_spaces();
            bool explicit_and = consume('&') || consume('*');
            if (!explicit_and && !starts_unary()) break;
            if (!parse_unary()) return false;
            ops_.push_back({PowerExprOpcode::logical_and, -1});
        }
        return true;
    }

    bool parse_unary() {
        skip_spaces();
        if (consume('!')) {
            if (!parse_unary()) return false;
            ops_.push_back({PowerExprOpcode::logical_not, -1});
            return true;
        }
        return parse_primary();
    }

    bool parse_primary() {
        skip_spaces();
        if (consume('(')) {
            if (!parse_or()) return false;
            skip_spaces();
            return consume(')');
        }

        const std::string ident = parse_identifier();
        if (ident.empty()) return false;
        if (ident == "0" || ident == "1'b0") {
            ops_.push_back({PowerExprOpcode::const_zero, -1});
            return true;
        }
        if (ident == "1" || ident == "1'b1") {
            ops_.push_back({PowerExprOpcode::const_one, -1});
            return true;
        }

        int port_id = cell_ ? cell_->get_port(ident) : -1;
        // Sequential Liberty functions may reference state/internal symbols that are
        // not cell ports (for example IQ). Keep the bytecode valid and let eval()
        // return unknown for unresolved symbols until Sim/state handling is added.
        ops_.push_back({PowerExprOpcode::port, port_id});
        return true;
    }

    std::string parse_identifier() {
        skip_spaces();
        const size_t start = pos_;
        while (pos_ < expr_.size()) {
            const char ch = expr_[pos_];
            if (std::isspace(static_cast<unsigned char>(ch)) || ch == '!' || ch == '&' || ch == '|' ||
                ch == '^' || ch == '+' || ch == '*' || ch == '(' || ch == ')') {
                break;
            }
            pos_++;
        }
        return expr_.substr(start, pos_ - start);
    }

    void skip_spaces() {
        while (pos_ < expr_.size() && std::isspace(static_cast<unsigned char>(expr_[pos_]))) pos_++;
    }

    bool starts_unary() const {
        if (pos_ >= expr_.size()) return false;
        const char ch = expr_[pos_];
        return ch == '!' || ch == '(' || is_identifier_char(ch);
    }

    static bool is_identifier_char(char ch) {
        return !(std::isspace(static_cast<unsigned char>(ch)) || ch == '!' || ch == '&' || ch == '|' ||
                 ch == '^' || ch == '+' || ch == '*' || ch == '(' || ch == ')');
    }

    bool consume(char ch) {
        if (pos_ < expr_.size() && expr_[pos_] == ch) {
            pos_++;
            return true;
        }
        return false;
    }

    const std::string& expr_;
    const LibertyCell* cell_ = nullptr;
    std::vector<PowerExprOp>& ops_;
    size_t pos_ = 0;
    bool ok_ = true;
};

}  // namespace

void CellLib::uncomment(std::vector<char>& buffer) {
    auto fsize = buffer.size() > 0 ? buffer.size() - 1 : 0;

    for (size_t i = 0; i < fsize; ++i) {
        // Block comment
        if (buffer[i] == '/' && buffer[i + 1] == '*') {
            buffer[i] = buffer[i + 1] = ' ';
            for (i = i + 2; i < fsize; buffer[i++] = ' ') {
                if (buffer[i] == '*' && buffer[i + 1] == '/') {
                    buffer[i] = buffer[i + 1] = ' ';
                    i = i + 1;
                    break;
                }
            }
        }

        // Line comment
        if (buffer[i] == '/' && buffer[i + 1] == '/') {
            buffer[i] = buffer[i + 1] = ' ';
            for (i = i + 2; i < fsize; ++i) {
                if (buffer[i] == '\n' || buffer[i] == '\r') {
                    break;
                } else
                    buffer[i] = ' ';
            }
        }

        // Pond comment
        if (buffer[i] == '#') {
            buffer[i] = ' ';
            for (i = i + 1; i < fsize; ++i) {
                if (buffer[i] == '\n' || buffer[i] == '\r') {
                    break;
                } else
                    buffer[i] = ' ';
            }
        }
    }
}

void CellLib::tokenize(const std::vector<char>& buf, std::vector<std::string_view>& tokens) {
    static std::string_view dels = "(),:;/#[]{}\"\\";
    const char* beg = buf.data();
    const char* end = buf.data() + buf.size();

    const char* token{nullptr};
    size_t len{0};

    tokens.clear();

    for (const char* itr = beg; itr != end && *itr != 0; ++itr) {
        if (std::isspace(*itr) || (dels.find(*itr) != std::string_view::npos)) {
            if (len > 0) {  // Add the current token.
                tokens.push_back({token, len});
                token = nullptr;
                len = 0;
            }
            if (*itr == '(' || *itr == ')' || *itr == '{' || *itr == '}') {
                tokens.push_back({itr, 1});
            }
        } else {
            if (len == 0) {
                token = itr;
            }
            ++len;
        }
    }

    if (len > 0) {
        tokens.push_back({token, len});
    }
}

LibertyCell* CellLib::get_cell(const std::string& name) {
    if (auto itr = lib_cells_.find(name); itr == lib_cells_.end()) {
        return nullptr;
    } else {
        return itr->second;
    }
}

int LibertyCell::get_port(const std::string& name) const {
    if (auto itr = ports_map_.find(name); itr == ports_map_.end()) {
        return -1;
    } else {
        return itr->second;
    }
}

bool PowerExpr::compile(const string& expr, const LibertyCell* cell) {
    source_ = expr;
    ops_.clear();
    PowerExprParser parser(expr, cell, ops_);
    valid_ = parser.parse();
    if (!valid_) {
        ops_.clear();
    }
    return valid_;
}

int8_t PowerExpr::eval(const vector<int8_t>& port_values) const {
    vector<int8_t> stack;
    stack.reserve(ops_.size());
    auto pop = [&]() -> int8_t {
        if (stack.empty()) return -1;
        int8_t v = stack.back();
        stack.pop_back();
        return v;
    };

    for (const auto& op : ops_) {
        switch (op.opcode) {
            case PowerExprOpcode::port:
                if (op.port_id >= 0 && op.port_id < static_cast<int>(port_values.size())) {
                    stack.push_back(port_values[op.port_id]);
                } else {
                    stack.push_back(-1);
                }
                break;
            case PowerExprOpcode::const_zero:
                stack.push_back(0);
                break;
            case PowerExprOpcode::const_one:
                stack.push_back(1);
                break;
            case PowerExprOpcode::logical_not: {
                const int8_t a = pop();
                stack.push_back(a < 0 ? -1 : static_cast<int8_t>(!a));
                break;
            }
            case PowerExprOpcode::logical_and: {
                const int8_t b = pop();
                const int8_t a = pop();
                if (a == 0 || b == 0) stack.push_back(0);
                else if (a == 1 && b == 1) stack.push_back(1);
                else stack.push_back(-1);
                break;
            }
            case PowerExprOpcode::logical_or: {
                const int8_t b = pop();
                const int8_t a = pop();
                if (a == 1 || b == 1) stack.push_back(1);
                else if (a == 0 && b == 0) stack.push_back(0);
                else stack.push_back(-1);
                break;
            }
            case PowerExprOpcode::logical_xor: {
                const int8_t b = pop();
                const int8_t a = pop();
                if (a < 0 || b < 0) stack.push_back(-1);
                else stack.push_back(static_cast<int8_t>((a != 0) ^ (b != 0)));
                break;
            }
        }
    }

    return stack.size() == 1 ? stack.back() : -1;
}

EnumNameMap<DelayModel> delay_model_name_map = {{DelayModel::generic_cmos, "generic_cmos"},
                                                {DelayModel::table_lookup, "table_lookup"},
                                                {DelayModel::cmos2, "cmos2"},
                                                {DelayModel::piecewise_cmos, "piecewise_cmos"},
                                                {DelayModel::dcm, "dcm"},
                                                {DelayModel::polynomial, "polynomial"},
                                                {DelayModel::unknown, "unknown"}};
DelayModel findDelayModel(const std::string model_name) {
    return delay_model_name_map.find(model_name, DelayModel::unknown);
}

EnumNameMap<CellPortDirection> port_direction_name_map = {{CellPortDirection::input, "input"},
                                                          {CellPortDirection::output, "output"},
                                                          {CellPortDirection::inout, "inout"},
                                                          {CellPortDirection::internal, "internal"},
                                                          {CellPortDirection::unknown, "unknown"}};

CellPortDirection findPortDirection(const std::string dir_name) {
    return port_direction_name_map.find(dir_name, CellPortDirection::unknown);
}

}  // namespace gt
