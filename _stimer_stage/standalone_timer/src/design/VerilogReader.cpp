#include "stimer/VerilogReader.h"

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stimer {
namespace {

std::string validate_token(std::string name) {
  // Same normalization shape as Xplace common/db/Database.cpp::validate_token.
  std::string::size_type pos = 0;
  while ((pos = name.find('\\', pos)) != std::string::npos) {
    name.erase(pos, 1);
  }
  pos = 0;
  while ((pos = name.find(' ', pos)) != std::string::npos) {
    name.erase(pos, 1);
  }
  return name;
}

bool is_verilog_symbol(unsigned char c) {
  // Copied from Xplace common/io/file_verilog.cpp and kept intentionally
  // simple for gate-level timing netlists.
  static char symbols[256] = {0};
  static bool initialized = false;
  if (!initialized) {
    symbols[static_cast<int>('(')] = 1;
    symbols[static_cast<int>(')')] = 1;
    symbols[static_cast<int>(',')] = 1;
    symbols[static_cast<int>('.')] = 1;
    symbols[static_cast<int>(':')] = 1;
    symbols[static_cast<int>(';')] = 1;
    symbols[static_cast<int>('#')] = 1;
    symbols[static_cast<int>('[')] = 1;
    symbols[static_cast<int>(']')] = 1;
    symbols[static_cast<int>('{')] = 1;
    symbols[static_cast<int>('}')] = 1;
    symbols[static_cast<int>('*')] = 1;
    symbols[static_cast<int>('\"')] = 1;
    symbols[static_cast<int>('\\')] = 1;
    symbols[static_cast<int>(' ')] = 2;
    symbols[static_cast<int>('\t')] = 2;
    symbols[static_cast<int>('\n')] = 2;
    symbols[static_cast<int>('\r')] = 2;
    initialized = true;
  }
  return symbols[static_cast<int>(c)] != 0;
}

void tokenize_verilog_line(const std::string& raw_line,
                           std::vector<std::string>* tokens) {
  std::string line = raw_line;
  const std::size_t comment = line.find("//");
  if (comment != std::string::npos) {
    line.resize(comment);
  }

  std::string token;
  for (const unsigned char c : line) {
    if (is_verilog_symbol(c)) {
      if (!token.empty()) {
        tokens->push_back(validate_token(std::move(token)));
        token.clear();
      }
      if (c == ';') {
        tokens->push_back(";");
      }
    } else {
      token.push_back(static_cast<char>(c));
    }
  }
  if (!token.empty()) {
    tokens->push_back(validate_token(std::move(token)));
  }
}

bool read_verilog_statement(std::istream& input,
                            std::vector<std::string>* statement) {
  statement->clear();
  std::string line;
  while (std::getline(input, line)) {
    tokenize_verilog_line(line, statement);
    if (!statement->empty() && statement->back() == ";") {
      statement->pop_back();
      return true;
    }
    if (!statement->empty() && statement->front() == "endmodule") {
      return true;
    }
  }
  return !statement->empty();
}

bool is_number_token(const std::string& token) {
  if (token.empty()) {
    return false;
  }
  for (const unsigned char c : token) {
    if (!std::isdigit(c)) {
      return false;
    }
  }
  return true;
}

bool is_constant_net(const std::string& token) {
  return token.find('\'') != std::string::npos || token == "1'b0" ||
         token == "1'b1" || token == "1" || token == "0";
}

void parse_port_declaration(const std::vector<std::string>& tokens,
                            DesignPinDirection direction,
                            DesignDB* design) {
  for (std::size_t i = 1; i < tokens.size(); ++i) {
    const std::string& name = tokens[i];
    if (name == "wire" || name == "reg" || name == "signed" ||
        is_number_token(name)) {
      continue;
    }
    design->get_or_add_port(name, name, direction);
    DesignConnection connection;
    connection.pin_name = name;
    connection.is_port = true;
    design->add_connection_to_net(name, std::move(connection));
  }
}

void parse_net_declaration(const std::vector<std::string>& tokens,
                           DesignDB* design) {
  for (std::size_t i = 1; i < tokens.size(); ++i) {
    const std::string& name = tokens[i];
    if (name == "wire" || name == "reg" || name == "signed" ||
        is_number_token(name)) {
      continue;
    }
    design->get_or_add_net(name);
  }
}

void parse_gate_instance(const std::vector<std::string>& tokens,
                         DesignDB* design) {
  if (tokens.size() < 2 || tokens[0] == "assign") {
    return;
  }

  const std::string cell_name = tokens[0];
  const std::string instance_name = tokens[1];
  design->get_or_add_instance(instance_name, cell_name);

  for (std::size_t i = 2; i + 1 < tokens.size(); i += 2) {
    const std::string& pin_name = tokens[i];
    const std::string& net_name = tokens[i + 1];
    if (pin_name.empty() || net_name.empty() || is_constant_net(net_name)) {
      continue;
    }
    DesignConnection connection;
    connection.instance_name = instance_name;
    connection.pin_name = pin_name;
    design->add_connection_to_net(net_name, std::move(connection));
  }
}

}  // namespace

DesignDB read_verilog_design(const std::string& path) {
  DesignDB design;
  read_verilog_design_into(path, &design);
  return design;
}

void read_verilog_design_into(const std::string& path, DesignDB* design) {
  if (design == nullptr) {
    throw std::runtime_error("read_verilog_design_into got null design");
  }

  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open Verilog file: " + path);
  }

  std::vector<std::string> tokens;
  while (read_verilog_statement(input, &tokens)) {
    if (tokens.empty()) {
      continue;
    }
    if (tokens[0] == "module") {
      if (tokens.size() > 1 && design->design_name.empty()) {
        design->design_name = tokens[1];
      }
    } else if (tokens[0] == "input") {
      parse_port_declaration(tokens, DesignPinDirection::kInput, design);
    } else if (tokens[0] == "output") {
      parse_port_declaration(tokens, DesignPinDirection::kOutput, design);
    } else if (tokens[0] == "inout") {
      parse_port_declaration(tokens, DesignPinDirection::kInout, design);
    } else if (tokens[0] == "wire" || tokens[0] == "reg") {
      parse_net_declaration(tokens, design);
    } else if (tokens[0] == "endmodule") {
      break;
    } else {
      parse_gate_instance(tokens, design);
    }
  }

  design->update_counts();
}

}  // namespace stimer
