#include "stimer/LibertyReader.h"

#include <zlib.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace stimer {

namespace {

using Token = std::string_view;
using TokenList = std::vector<Token>;
using TokenIterator = TokenList::iterator;

void uncomment(std::vector<char>* buffer) {
  const std::size_t fsize = buffer->empty() ? 0 : buffer->size() - 1;
  for (std::size_t i = 0; i < fsize; ++i) {
    if ((*buffer)[i] == '/' && (*buffer)[i + 1] == '*') {
      (*buffer)[i] = (*buffer)[i + 1] = ' ';
      for (i += 2; i < fsize; (*buffer)[i++] = ' ') {
        if ((*buffer)[i] == '*' && (*buffer)[i + 1] == '/') {
          (*buffer)[i] = (*buffer)[i + 1] = ' ';
          ++i;
          break;
        }
      }
    }

    if ((*buffer)[i] == '/' && (*buffer)[i + 1] == '/') {
      (*buffer)[i] = (*buffer)[i + 1] = ' ';
      for (i += 2; i < fsize; ++i) {
        if ((*buffer)[i] == '\n' || (*buffer)[i] == '\r') {
          break;
        }
        (*buffer)[i] = ' ';
      }
    }

    if ((*buffer)[i] == '#') {
      (*buffer)[i] = ' ';
      for (++i; i < fsize; ++i) {
        if ((*buffer)[i] == '\n' || (*buffer)[i] == '\r') {
          break;
        }
        (*buffer)[i] = ' ';
      }
    }
  }
}

void tokenize(const std::vector<char>& buffer, TokenList* tokens) {
  static constexpr std::string_view delimiters = "(),:;/#[]{}*\"\\";
  const char* token = nullptr;
  std::size_t len = 0;
  tokens->clear();

  for (const char* itr = buffer.data(); *itr != 0; ++itr) {
    if (std::isspace(static_cast<unsigned char>(*itr)) ||
        delimiters.find(*itr) != std::string_view::npos) {
      if (len > 0) {
        tokens->push_back({token, len});
        token = nullptr;
        len = 0;
      }
      if (*itr == '(' || *itr == ')' || *itr == '{' || *itr == '}') {
        tokens->push_back({itr, 1});
      }
    } else {
      if (len == 0) {
        token = itr;
      }
      ++len;
    }
  }

  if (len > 0) {
    tokens->push_back({token, len});
  }
}

template <typename Callback>
TokenIterator on_next_parentheses(TokenIterator begin,
                                  TokenIterator end,
                                  Callback&& callback) {
  auto left = std::find(begin, end, "(");
  auto right = left;
  int stack = 0;
  while (right != end) {
    if (*right == "(") {
      ++stack;
    } else if (*right == ")") {
      --stack;
    }
    if (stack == 0) {
      break;
    }
    ++right;
  }

  if (left == end || right == end) {
    return end;
  }
  for (++left; left != right; ++left) {
    callback(*left);
  }
  return right;
}

std::string token_string(Token token) {
  return std::string(token.data(), token.size());
}

bool token_is(Token token, std::string_view value) {
  return token == value;
}

float parse_float(Token token) {
  return std::strtof(token.data(), nullptr);
}

LibertyLutVariable lut_variable_from_token(Token token) {
  if (token == "total_output_net_capacitance") {
    return LibertyLutVariable::kTotalOutputNetCapacitance;
  }
  if (token == "input_net_transition") {
    return LibertyLutVariable::kInputNetTransition;
  }
  if (token == "constrained_pin_transition") {
    return LibertyLutVariable::kConstrainedPinTransition;
  }
  if (token == "related_pin_transition") {
    return LibertyLutVariable::kRelatedPinTransition;
  }
  if (token == "input_transition_timing" ||
      token == "input_transition_time") {
    return LibertyLutVariable::kInputTransitionTime;
  }
  return LibertyLutVariable::kUnknown;
}

bool has_suffix(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

std::vector<char> read_liberty_file(const std::string& path) {
  std::vector<char> buffer;
  if (has_suffix(path, ".gz")) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (!file) {
      throw std::runtime_error("cannot open gzip liberty file: " + path);
    }
    char chunk[4096];
    int len = 0;
    while ((len = gzread(file, chunk, sizeof(chunk))) > 0) {
      buffer.insert(buffer.end(), chunk, chunk + len);
    }
    gzclose(file);
    buffer.push_back(0);
    return buffer;
  }

  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file.good()) {
    throw std::runtime_error("cannot open liberty file: " + path);
  }
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  buffer.resize(static_cast<std::size_t>(size) + 1);
  file.read(buffer.data(), size);
  buffer[static_cast<std::size_t>(size)] = 0;
  return buffer;
}

std::string direction_name(Token token) {
  if (token == "input" || token == "output" || token == "inout" ||
      token == "internal") {
    return token_string(token);
  }
  return "unknown";
}

void set_cap(LibertyPinCap* cap,
             CapTransition transition,
             CapRange range,
             float value) {
  const int transition_index = static_cast<int>(transition);
  const int range_index = static_cast<int>(range);
  cap->value[transition_index][range_index] = value;
  cap->valid[transition_index][range_index] = true;
}

float default_pin_cap(const LibertyLibrary& library,
                      const std::string& direction) {
  if (direction == "input" && library.has_default_input_pin_cap) {
    return library.default_input_pin_cap;
  }
  if (direction == "output" && library.has_default_output_pin_cap) {
    return library.default_output_pin_cap;
  }
  if (direction == "inout" && library.has_default_inout_pin_cap) {
    return library.default_inout_pin_cap;
  }
  if (library.has_default_input_pin_cap) {
    return library.default_input_pin_cap;
  }
  if (library.has_default_inout_pin_cap) {
    return library.default_inout_pin_cap;
  }
  if (library.has_default_output_pin_cap) {
    return library.default_output_pin_cap;
  }
  return 0.0f;
}

void fill_missing_pin_caps(const LibertyLibrary& library,
                           bool has_capacitance,
                           float capacitance,
                           bool has_rise_capacitance,
                           float rise_capacitance,
                           bool has_fall_capacitance,
                           float fall_capacitance,
                           LibertyPin* pin) {
  const float base =
      has_capacitance ? capacitance : default_pin_cap(library, pin->direction);
  const float rise = has_rise_capacitance ? rise_capacitance : base;
  const float fall = has_fall_capacitance ? fall_capacitance : base;

  for (int range = 0; range < 2; ++range) {
    if (!pin->cap.valid[static_cast<int>(CapTransition::kRise)][range]) {
      pin->cap.value[static_cast<int>(CapTransition::kRise)][range] = rise;
      pin->cap.valid[static_cast<int>(CapTransition::kRise)][range] = true;
    }
    if (!pin->cap.valid[static_cast<int>(CapTransition::kFall)][range]) {
      pin->cap.value[static_cast<int>(CapTransition::kFall)][range] = fall;
      pin->cap.valid[static_cast<int>(CapTransition::kFall)][range] = true;
    }
  }
}

void skip_group_body(TokenIterator* iterator, TokenIterator end) {
  auto itr = std::find(*iterator, end, "{");
  if (itr == end) {
    *iterator = end;
    return;
  }
  int stack = 1;
  while (stack && ++itr != end) {
    if (*itr == "{") {
      ++stack;
    } else if (*itr == "}") {
      --stack;
    }
  }
  *iterator = itr;
}

const LibertyLutTemplate* find_lut_template(const LibertyLibrary& library,
                                            const std::string& name) {
  for (const auto& lut_template : library.lut_templates) {
    if (lut_template.name == name) {
      return &lut_template;
    }
  }
  return nullptr;
}

int valid_lut_count(const LibertyTimingArc& arc) {
  int count = 0;
  for (const auto& lut : arc.cell_delay) {
    count += lut.valid ? 1 : 0;
  }
  for (const auto& lut : arc.transition) {
    count += lut.valid ? 1 : 0;
  }
  for (const auto& lut : arc.constraint) {
    count += lut.valid ? 1 : 0;
  }
  return count;
}

class LibertyReader {
 public:
  LibertyLibrary read(const std::string& path) {
    buffer_ = read_liberty_file(path);
    uncomment(&buffer_);
    tokens_.reserve(buffer_.size() / sizeof(std::string));
    tokenize(buffer_, &tokens_);

    auto itr = std::find(tokens_.begin(), tokens_.end(), "library");
    if (itr == tokens_.end()) {
      throw std::runtime_error("can't find keyword library in " + path);
    }

    LibertyLibrary library;
    library.path = path;
    itr = on_next_parentheses(itr, tokens_.end(), [&](Token name) {
      library.name = token_string(name);
    });
    itr = std::find(itr, tokens_.end(), "{");
    if (itr == tokens_.end()) {
      throw std::runtime_error("can't find library group brace in " + path);
    }

    int stack = 1;
    while (stack && ++itr != tokens_.end()) {
      if (*itr == "default_input_pin_cap") {
        if (++itr != tokens_.end()) {
          library.default_input_pin_cap = parse_float(*itr);
          library.has_default_input_pin_cap = true;
        }
      } else if (*itr == "default_output_pin_cap") {
        if (++itr != tokens_.end()) {
          library.default_output_pin_cap = parse_float(*itr);
          library.has_default_output_pin_cap = true;
        }
      } else if (*itr == "default_inout_pin_cap") {
        if (++itr != tokens_.end()) {
          library.default_inout_pin_cap = parse_float(*itr);
          library.has_default_inout_pin_cap = true;
        }
      } else if (*itr == "capacitive_load_unit") {
        std::ostringstream unit;
        bool first = true;
        itr = on_next_parentheses(itr, tokens_.end(), [&](Token value) {
          if (!first) {
            unit << ' ';
          }
          unit << value;
          first = false;
        });
        library.capacitive_load_unit = unit.str();
      } else if (*itr == "lu_table_template" ||
                 *itr == "power_lut_template") {
        library.lut_templates.push_back(
            extract_lut_template(&itr, tokens_.end()));
      } else if (*itr == "cell") {
        library.cells.push_back(extract_cell(&itr, tokens_.end(), library));
      } else if (*itr == "}") {
        --stack;
      } else if (*itr == "{") {
        ++stack;
      }
    }

    return library;
  }

 private:
  LibertyLutTemplate extract_lut_template(TokenIterator* iterator,
                                          TokenIterator end) {
    LibertyLutTemplate lut_template;
    auto itr = *iterator;
    itr = on_next_parentheses(itr, end, [&](Token name) {
      lut_template.name = token_string(name);
    });
    itr = std::find(itr, end, "{");
    if (itr == end) {
      *iterator = end;
      return lut_template;
    }

    int stack = 1;
    while (stack && ++itr != end) {
      if (*itr == "variable_1") {
        if (++itr != end) {
          lut_template.variable1 = lut_variable_from_token(*itr);
        }
      } else if (*itr == "variable_2") {
        if (++itr != end) {
          lut_template.variable2 = lut_variable_from_token(*itr);
        }
      } else if (*itr == "index_1") {
        lut_template.indices1.clear();
        itr = on_next_parentheses(itr, end, [&](Token value) {
          lut_template.indices1.push_back(parse_float(value));
        });
      } else if (*itr == "index_2") {
        lut_template.indices2.clear();
        itr = on_next_parentheses(itr, end, [&](Token value) {
          lut_template.indices2.push_back(parse_float(value));
        });
      } else if (*itr == "}") {
        --stack;
      } else if (*itr == "{") {
        ++stack;
      }
    }

    *iterator = itr;
    return lut_template;
  }

  LibertyLut extract_lut(TokenIterator* iterator,
                         TokenIterator end,
                         const LibertyLibrary& library) {
    LibertyLut lut;
    auto itr = *iterator;
    itr = on_next_parentheses(itr, end, [&](Token name) {
      lut.template_name = token_string(name);
    });

    if (const auto* lut_template =
            find_lut_template(library, lut.template_name)) {
      lut.variable1 = lut_template->variable1;
      lut.variable2 = lut_template->variable2;
      lut.indices1 = lut_template->indices1;
      lut.indices2 = lut_template->indices2;
    }

    itr = std::find(itr, end, "{");
    if (itr == end) {
      *iterator = end;
      return lut;
    }

    int stack = 1;
    while (stack && ++itr != end) {
      if (*itr == "index_1") {
        lut.indices1.clear();
        itr = on_next_parentheses(itr, end, [&](Token value) {
          lut.indices1.push_back(parse_float(value));
        });
      } else if (*itr == "index_2") {
        lut.indices2.clear();
        itr = on_next_parentheses(itr, end, [&](Token value) {
          lut.indices2.push_back(parse_float(value));
        });
      } else if (*itr == "values") {
        lut.values.clear();
        itr = on_next_parentheses(itr, end, [&](Token value) {
          lut.values.push_back(parse_float(value));
        });
      } else if (*itr == "}") {
        --stack;
      } else if (*itr == "{") {
        ++stack;
      }
    }

    lut.valid = !lut.values.empty();
    *iterator = itr;
    return lut;
  }

  LibertyTimingArc extract_timing_arc(TokenIterator* iterator,
                                      TokenIterator end,
                                      const LibertyLibrary& library,
                                      const std::string& output_pin) {
    LibertyTimingArc arc;
    arc.output_pin = output_pin;
    auto itr = *iterator;
    itr = std::find(itr, end, "{");
    if (itr == end) {
      *iterator = end;
      return arc;
    }

    int stack = 1;
    while (stack && ++itr != end) {
      if (*itr == "cell_rise") {
        arc.cell_delay[static_cast<int>(CapTransition::kRise)] =
            extract_lut(&itr, end, library);
      } else if (*itr == "cell_fall") {
        arc.cell_delay[static_cast<int>(CapTransition::kFall)] =
            extract_lut(&itr, end, library);
      } else if (*itr == "rise_transition") {
        arc.transition[static_cast<int>(CapTransition::kRise)] =
            extract_lut(&itr, end, library);
      } else if (*itr == "fall_transition") {
        arc.transition[static_cast<int>(CapTransition::kFall)] =
            extract_lut(&itr, end, library);
      } else if (*itr == "rise_constraint") {
        arc.constraint[static_cast<int>(CapTransition::kRise)] =
            extract_lut(&itr, end, library);
      } else if (*itr == "fall_constraint") {
        arc.constraint[static_cast<int>(CapTransition::kFall)] =
            extract_lut(&itr, end, library);
      } else if (*itr == "timing_sense") {
        if (++itr != end) {
          arc.timing_sense = token_string(*itr);
        }
      } else if (*itr == "timing_type") {
        if (++itr != end) {
          arc.timing_type = token_string(*itr);
        }
      } else if (*itr == "sdf_cond") {
        if (++itr != end) {
          arc.sdf_cond = token_string(*itr);
          arc.is_cond = true;
        }
      } else if (*itr == "related_pin") {
        if (++itr != end) {
          arc.related_pin = token_string(*itr);
        }
      } else if (*itr == "}") {
        --stack;
      } else if (*itr == "{") {
        ++stack;
      }
    }

    *iterator = itr;
    return arc;
  }

  LibertyCell extract_cell(TokenIterator* iterator,
                           TokenIterator end,
                           const LibertyLibrary& library) {
    LibertyCell cell;
    auto itr = *iterator;
    itr = on_next_parentheses(itr, end, [&](Token name) {
      cell.name = token_string(name);
    });
    itr = std::find(itr, end, "{");
    if (itr == end) {
      *iterator = end;
      return cell;
    }

    int stack = 1;
    while (stack && ++itr != end) {
      if (*itr == "pin") {
        cell.pins.push_back(extract_pin(&itr, end, library, &cell));
      } else if (*itr == "}") {
        --stack;
      } else if (*itr == "{") {
        ++stack;
      }
    }
    *iterator = itr;
    return cell;
  }

  LibertyPin extract_pin(TokenIterator* iterator,
                         TokenIterator end,
                         const LibertyLibrary& library,
                         LibertyCell* cell) {
    LibertyPin pin;
    auto itr = *iterator;
    itr = on_next_parentheses(itr, end, [&](Token name) {
      pin.name = token_string(name);
    });
    itr = std::find(itr, end, "{");
    if (itr == end) {
      *iterator = end;
      return pin;
    }

    bool has_capacitance = false;
    bool has_rise_capacitance = false;
    bool has_fall_capacitance = false;
    float capacitance = 0.0f;
    float rise_capacitance = 0.0f;
    float fall_capacitance = 0.0f;

    int stack = 1;
    while (stack && ++itr != end) {
      if (*itr == "direction") {
        if (++itr != end) {
          pin.direction = direction_name(*itr);
        }
      } else if (*itr == "clock") {
        if (++itr != end) {
          pin.is_clock = token_is(*itr, "true");
        }
      } else if (*itr == "capacitance") {
        if (++itr != end) {
          capacitance = parse_float(*itr);
          has_capacitance = true;
        }
      } else if (*itr == "rise_capacitance") {
        if (++itr != end) {
          rise_capacitance = parse_float(*itr);
          has_rise_capacitance = true;
        }
      } else if (*itr == "fall_capacitance") {
        if (++itr != end) {
          fall_capacitance = parse_float(*itr);
          has_fall_capacitance = true;
        }
      } else if (*itr == "rise_capacitance_range") {
        int index = 0;
        itr = on_next_parentheses(itr, end, [&](Token value) {
          if (index == 0) {
            set_cap(&pin.cap, CapTransition::kRise, CapRange::kMin,
                    parse_float(value));
          } else if (index == 1) {
            set_cap(&pin.cap, CapTransition::kRise, CapRange::kMax,
                    parse_float(value));
          }
          ++index;
        });
      } else if (*itr == "fall_capacitance_range") {
        int index = 0;
        itr = on_next_parentheses(itr, end, [&](Token value) {
          if (index == 0) {
            set_cap(&pin.cap, CapTransition::kFall, CapRange::kMin,
                    parse_float(value));
          } else if (index == 1) {
            set_cap(&pin.cap, CapTransition::kFall, CapRange::kMax,
                    parse_float(value));
          }
          ++index;
        });
      } else if (*itr == "timing") {
        pin.timing_arcs.push_back(
            extract_timing_arc(&itr, end, library, pin.name));
        ++cell->num_timing_groups;
      } else if (*itr == "}") {
        --stack;
      } else if (*itr == "{") {
        ++stack;
      }
    }

    fill_missing_pin_caps(library, has_capacitance, capacitance,
                          has_rise_capacitance, rise_capacitance,
                          has_fall_capacitance, fall_capacitance, &pin);
    *iterator = itr;
    return pin;
  }

  std::vector<char> buffer_;
  TokenList tokens_;
};

void accumulate_library(const LibertyLibrary& library, LibertyDB* db) {
  ++db->num_files;
  db->num_cells += static_cast<int>(library.cells.size());
  db->num_lut_templates += static_cast<int>(library.lut_templates.size());
  for (const auto& cell : library.cells) {
    db->num_pins += static_cast<int>(cell.pins.size());
    for (const auto& pin : cell.pins) {
      db->num_timing_arcs += static_cast<int>(pin.timing_arcs.size());
      for (const auto& arc : pin.timing_arcs) {
        db->num_timing_luts += valid_lut_count(arc);
      }
      if (pin.cap.complete()) {
        ++db->num_pin_caps;
      }
    }
  }
}

}  // namespace

bool LibertyPinCap::complete() const {
  for (const auto& transition : valid) {
    for (const bool has_value : transition) {
      if (!has_value) {
        return false;
      }
    }
  }
  return true;
}

const char* liberty_lut_variable_name(LibertyLutVariable variable) {
  switch (variable) {
    case LibertyLutVariable::kTotalOutputNetCapacitance:
      return "total_output_net_capacitance";
    case LibertyLutVariable::kInputNetTransition:
      return "input_net_transition";
    case LibertyLutVariable::kConstrainedPinTransition:
      return "constrained_pin_transition";
    case LibertyLutVariable::kRelatedPinTransition:
      return "related_pin_transition";
    case LibertyLutVariable::kInputTransitionTime:
      return "input_transition_time";
    case LibertyLutVariable::kUnknown:
      return "unknown";
  }
  return "unknown";
}

LibertyDB load_liberty_files(const std::vector<std::string>& paths) {
  LibertyDB db;
  for (std::size_t i = 0; i < paths.size(); ++i) {
    const auto& path = paths[i];
    const auto start = std::chrono::steady_clock::now();
    std::cerr << "[stage] liberty_file " << (i + 1) << "/" << paths.size()
              << " start " << path << '\n'
              << std::flush;
    LibertyReader reader;
    LibertyLibrary library = reader.read(path);
    accumulate_library(library, &db);
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::cerr << "[stage] liberty_file " << (i + 1) << "/" << paths.size()
              << " done_ms=" << elapsed_ms
              << " cells=" << library.cells.size()
              << " timing_luts_total=" << db.num_timing_luts << '\n'
              << std::flush;
    db.libraries.push_back(std::move(library));
  }
  return db;
}

std::string format_liberty_summary(const LibertyDB& db) {
  std::ostringstream os;
  os << "liberty_libraries: " << db.num_files << '\n';
  os << "liberty_cells: " << db.num_cells << '\n';
  os << "liberty_pins: " << db.num_pins << '\n';
  os << "liberty_pin_caps: " << db.num_pin_caps << '\n';
  os << "liberty_timing_groups: " << db.num_timing_arcs << '\n';
  os << "liberty_lut_templates: " << db.num_lut_templates << '\n';
  os << "liberty_timing_luts: " << db.num_timing_luts << '\n';
  for (const auto& library : db.libraries) {
    os << "  - library=" << library.name << " path=" << library.path
       << " cells=" << library.cells.size()
       << " lut_templates=" << library.lut_templates.size()
       << " cap_unit="
       << (library.capacitive_load_unit.empty() ? "<unknown>"
                                               : library.capacitive_load_unit)
       << '\n';
    for (const auto& lut_template : library.lut_templates) {
      os << "    template=" << lut_template.name
         << " var1=" << liberty_lut_variable_name(lut_template.variable1)
         << " var2=" << liberty_lut_variable_name(lut_template.variable2)
         << " size=" << lut_template.indices1.size() << "x"
         << lut_template.indices2.size() << '\n';
    }
  }
  return os.str();
}

}  // namespace stimer
