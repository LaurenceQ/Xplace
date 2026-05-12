#include "stimer/SpefReader.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "parser-spef.hpp"

namespace stimer {

namespace {

namespace pegtl = tao::TAO_PEGTL_NAMESPACE;

struct SpefBuilder {
  RcGraph graph;
  SpefParseOptions options;
  std::vector<std::string> name_map;
  std::vector<std::string_view> tokens;
  std::unordered_map<std::string_view, int> current_raw_node_index;
  RcNet* current_net = nullptr;
};

struct LineTokens {
  std::array<std::string_view, 8> values{};
  std::size_t size = 0;

  bool empty() const { return size == 0; }
  std::string_view operator[](std::size_t index) const {
    return values[index];
  }
};

std::string file_to_memory(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    return "";
  }

  std::ifstream file(path);
  file.seekg(0, std::ios::end);
  std::string buffer;
  buffer.resize(file.tellg());
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  return buffer;
}

void strip_line_comments(std::string* buffer) {
  for (std::size_t i = 0; i < buffer->size(); ++i) {
    if ((*buffer)[i] == '/' && i + 1 < buffer->size() &&
        (*buffer)[i + 1] == '/') {
      (*buffer)[i] = ' ';
      (*buffer)[i + 1] = ' ';
      for (i += 2; i < buffer->size(); ++i) {
        if ((*buffer)[i] == '\n' || (*buffer)[i] == '\r') {
          break;
        }
        (*buffer)[i] = ' ';
      }
    }
  }
}

std::string string_from_view(std::string_view view) {
  return std::string(view.data(), view.size());
}

bool parse_unsigned_prefix(std::string_view view,
                           std::size_t* value,
                           std::size_t* digits) {
  std::size_t result = 0;
  std::size_t count = 0;
  for (const unsigned char c : view) {
    if (!std::isdigit(c)) {
      break;
    }
    ++count;
    result = result * 10 + static_cast<std::size_t>(c - '0');
  }
  if (count == 0) {
    return false;
  }
  *value = result;
  *digits = count;
  return true;
}

bool parse_unsigned(std::string_view view, std::size_t* value) {
  std::size_t digits = 0;
  return parse_unsigned_prefix(view, value, &digits);
}

double parse_double(std::string_view view) {
  double value = 0.0;
  const char* begin = view.data();
  const char* end = view.data() + view.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec == std::errc()) {
    return value;
  }

  return std::strtod(string_from_view(view).c_str(), nullptr);
}

std::string resolve_name(const SpefBuilder& builder, std::string_view raw) {
  if (raw.size() > 1 && raw.front() == '*') {
    std::size_t key = 0;
    std::size_t digits = 0;
    if (parse_unsigned_prefix(raw.substr(1), &key, &digits)) {
      if (key < builder.name_map.size() && !builder.name_map[key].empty()) {
        const std::string& mapped = builder.name_map[key];
        const std::string_view suffix = raw.substr(1 + digits);
        if (suffix.empty()) {
          return mapped;
        }
        std::string expanded = mapped;
        expanded.append(suffix.data(), suffix.size());
        return expanded;
      }
    }
  }
  return string_from_view(raw);
}

int node_index(SpefBuilder* builder, std::string_view raw_name) {
  auto raw_found = builder->current_raw_node_index.find(raw_name);
  if (raw_found != builder->current_raw_node_index.end()) {
    return raw_found->second;
  }

  RcNet* net = builder->current_net;
  const int index = static_cast<int>(net->nodes.size());
  RcNode node;
  node.name = resolve_name(*builder, raw_name);
  net->nodes.push_back(std::move(node));
  builder->current_raw_node_index.emplace(raw_name, index);
  return index;
}

bool numeric_base_matches_current_net(const SpefBuilder& builder,
                                      std::string_view raw) {
  if (builder.current_net == nullptr || raw.size() <= 1 ||
      raw.front() != '*') {
    return false;
  }

  std::size_t key = 0;
  std::size_t digits = 0;
  if (!parse_unsigned_prefix(raw.substr(1), &key, &digits)) {
    return false;
  }
  if (key >= builder.name_map.size() || builder.name_map[key].empty()) {
    return false;
  }
  const std::string& mapped = builder.name_map[key];
  if (mapped != builder.current_net->name) {
    return false;
  }
  const std::string_view suffix = raw.substr(1 + digits);
  return suffix.empty() || suffix.front() == ':' || suffix.front() == '.';
}

bool plain_base_matches_current_net(const SpefBuilder& builder,
                                    std::string_view raw) {
  if (builder.current_net == nullptr || raw.empty() || raw.front() == '*') {
    return false;
  }
  const std::string& net_name = builder.current_net->name;
  if (raw.size() < net_name.size() ||
      raw.substr(0, net_name.size()) != net_name) {
    return false;
  }
  if (raw.size() == net_name.size()) {
    return true;
  }
  const char suffix = raw[net_name.size()];
  return suffix == ':' || suffix == '.';
}

bool raw_node_is_local_to_current_net(const SpefBuilder& builder,
                                      std::string_view raw) {
  if (builder.current_raw_node_index.find(raw) !=
      builder.current_raw_node_index.end()) {
    return true;
  }
  return numeric_base_matches_current_net(builder, raw) ||
         plain_base_matches_current_net(builder, raw);
}

void add_ground_cap(SpefBuilder* builder,
                    std::string_view raw_name,
                    double capacitance) {
  if (capacitance <= 0.0) {
    return;
  }

  const int index = node_index(builder, raw_name);
  builder->current_net->nodes[index].ground_capacitance += capacitance;
}

RcDirection parse_direction(char value) {
  switch (value) {
    case 'I':
      return RcDirection::kInput;
    case 'O':
      return RcDirection::kOutput;
    case 'B':
      return RcDirection::kInout;
    default:
      return RcDirection::kUnknown;
  }
}

std::string upper(std::string value) {
  for (char& ch : value) {
    ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  }
  return value;
}

bool includes_pin_caps_from_design_flow(const std::string& design_flow) {
  const std::string flow = upper(design_flow);
  const std::size_t pin_cap_pos = flow.find("PIN_CAP");
  if (pin_cap_pos == std::string::npos) {
    return false;
  }
  return flow.substr(pin_cap_pos).find("NONE") == std::string::npos;
}

bool same_token(std::string_view token, const char* value) {
  const std::string_view expected(value);
  return token == expected;
}

bool is_spef_control_token(std::string_view token) {
  return token.size() > 1 && token.front() == '*' &&
         !std::isdigit(static_cast<unsigned char>(token[1]));
}

std::string_view trim(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front()))) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back()))) {
    value.remove_suffix(1);
  }
  return value;
}

std::string header_value(std::string_view line, std::string_view key) {
  if (line.size() <= key.size()) {
    return "";
  }
  return string_from_view(trim(line.substr(key.size())));
}

void split_ws(std::string_view line, std::vector<std::string_view>* tokens) {
  tokens->clear();
  std::size_t pos = 0;
  while (pos < line.size()) {
    while (pos < line.size() &&
           std::isspace(static_cast<unsigned char>(line[pos]))) {
      ++pos;
    }
    const std::size_t begin = pos;
    while (pos < line.size() &&
           !std::isspace(static_cast<unsigned char>(line[pos]))) {
      ++pos;
    }
    if (begin < pos) {
      tokens->push_back(line.substr(begin, pos - begin));
    }
  }
}

void split_ws_fast(std::string_view line, LineTokens* tokens) {
  tokens->size = 0;
  std::size_t pos = 0;
  while (pos < line.size()) {
    while (pos < line.size() &&
           std::isspace(static_cast<unsigned char>(line[pos]))) {
      ++pos;
    }
    const std::size_t begin = pos;
    while (pos < line.size() &&
           !std::isspace(static_cast<unsigned char>(line[pos]))) {
      ++pos;
    }
    if (begin < pos && tokens->size < tokens->values.size()) {
      tokens->values[tokens->size++] = line.substr(begin, pos - begin);
    }
  }
}

enum class SpefSection {
  kNone,
  kNameMap,
  kConn,
  kCap,
  kRes,
};

void parse_name_map_tokens(const LineTokens& tokens, SpefBuilder* builder) {
  if (tokens.size != 2 || tokens[0].size() <= 1 ||
      tokens[0].front() != '*') {
    return;
  }
  std::size_t key = 0;
  if (parse_unsigned(tokens[0].substr(1), &key)) {
    if (key >= builder->name_map.size()) {
      builder->name_map.resize(key + 1);
    }
    builder->name_map[key] = string_from_view(tokens[1]);
  }
}

void begin_net_from_tokens(const LineTokens& tokens, SpefBuilder* builder) {
  RcNet net;
  net.name = tokens.size > 1 ? resolve_name(*builder, tokens[1])
                             : std::string();
  if (tokens.size > 2) {
    net.total_capacitance = parse_double(tokens[2]);
  }
  builder->graph.nets.push_back(std::move(net));
  builder->current_net = &builder->graph.nets.back();
  builder->current_raw_node_index.clear();
}

void parse_conn_tokens(const LineTokens& tokens, SpefBuilder* builder) {
  if (builder->current_net == nullptr || tokens.size < 3) {
    return;
  }

  RcConnection connection;
  connection.kind = tokens[0].size() > 1 && tokens[0][1] == 'P'
                        ? RcConnectionKind::kPort
                        : RcConnectionKind::kInstancePin;
  connection.node = node_index(builder, tokens[1]);
  connection.direction = parse_direction(tokens[2][0]);
  builder->current_net->connections.push_back(connection);
}

void parse_cap_tokens(const LineTokens& tokens, SpefBuilder* builder) {
  if (builder->current_net == nullptr || tokens.size < 3) {
    return;
  }

  if (tokens.size >= 4) {
    const double capacitance = parse_double(tokens[3]);
    if (builder->options.keep_coupling_caps) {
      RcCapacitor capacitor;
      capacitor.node1 = node_index(builder, tokens[1]);
      capacitor.node2 = node_index(builder, tokens[2]);
      capacitor.capacitance = capacitance;
      builder->current_net->capacitors.push_back(capacitor);
    } else {
      const double scaled_capacitance =
          capacitance * builder->options.coupling_reduction_factor;
      add_ground_cap(builder, tokens[1], scaled_capacitance);
      if (raw_node_is_local_to_current_net(*builder, tokens[2])) {
        add_ground_cap(builder, tokens[2], scaled_capacitance);
      }
    }
    return;
  }

  add_ground_cap(builder, tokens[1], parse_double(tokens[2]));
}

void parse_res_tokens(const LineTokens& tokens, SpefBuilder* builder) {
  if (builder->current_net == nullptr || tokens.size < 4) {
    return;
  }

  RcResistor resistor;
  resistor.node1 = node_index(builder, tokens[1]);
  resistor.node2 = node_index(builder, tokens[2]);
  resistor.resistance = parse_double(tokens[3]);
  builder->current_net->resistors.push_back(resistor);
}

void parse_spef_lines(std::string_view buffer, SpefBuilder* builder) {
  SpefSection section = SpefSection::kNone;
  LineTokens tokens;
  std::size_t line_begin = 0;
  while (line_begin <= buffer.size()) {
    std::size_t line_end = buffer.find_first_of("\r\n", line_begin);
    if (line_end == std::string_view::npos) {
      line_end = buffer.size();
    }

    std::string_view line(buffer.data() + line_begin, line_end - line_begin);
    const std::size_t comment = line.find("//");
    if (comment != std::string_view::npos) {
      line = line.substr(0, comment);
    }
    line = trim(line);
    split_ws_fast(line, &tokens);
    if (!tokens.empty()) {
      const std::string_view first = tokens[0];
      if (same_token(first, "*DESIGN")) {
        builder->graph.design_name = header_value(line, "*DESIGN");
      } else if (same_token(first, "*DESIGN_FLOW")) {
        builder->graph.includes_pin_caps =
            includes_pin_caps_from_design_flow(header_value(line,
                                                            "*DESIGN_FLOW"));
      } else if (same_token(first, "*NAME_MAP")) {
        section = SpefSection::kNameMap;
      } else if (same_token(first, "*D_NET")) {
        begin_net_from_tokens(tokens, builder);
        section = SpefSection::kNone;
      } else if (same_token(first, "*CONN")) {
        section = SpefSection::kConn;
      } else if (same_token(first, "*CAP")) {
        section = SpefSection::kCap;
      } else if (same_token(first, "*RES")) {
        section = SpefSection::kRes;
      } else if (same_token(first, "*END")) {
        builder->current_net = nullptr;
        builder->current_raw_node_index.clear();
        section = SpefSection::kNone;
      } else if (section == SpefSection::kNameMap &&
                 is_spef_control_token(first)) {
        section = SpefSection::kNone;
      } else if (section == SpefSection::kNameMap) {
        parse_name_map_tokens(tokens, builder);
      } else if (section == SpefSection::kConn) {
        parse_conn_tokens(tokens, builder);
      } else if (section == SpefSection::kCap) {
        parse_cap_tokens(tokens, builder);
      } else if (section == SpefSection::kRes) {
        parse_res_tokens(tokens, builder);
      }
    }

    if (line_end == buffer.size()) {
      break;
    }
    line_begin = line_end + 1;
    if (line_begin < buffer.size() && buffer[line_end] == '\r' &&
        buffer[line_begin] == '\n') {
      ++line_begin;
    }
  }
}

void recompute_graph_counts(RcGraph* graph) {
  graph->num_nets = static_cast<int>(graph->nets.size());
  graph->num_nodes = 0;
  graph->num_connections = 0;
  graph->num_resistors = 0;
  graph->num_capacitors = 0;
  graph->num_coupling_capacitors = 0;
  graph->num_ground_cap_nodes = 0;
  for (const auto& net : graph->nets) {
    graph->num_nodes += static_cast<int>(net.nodes.size());
    graph->num_connections += static_cast<int>(net.connections.size());
    graph->num_resistors += static_cast<int>(net.resistors.size());
    int ground_cap_nodes = 0;
    for (const auto& node : net.nodes) {
      if (node.ground_capacitance > 0.0) {
        ++ground_cap_nodes;
      }
    }
    graph->num_ground_cap_nodes += ground_cap_nodes;
    for (const auto& capacitor : net.capacitors) {
      if (capacitor.is_coupling()) {
        ++graph->num_coupling_capacitors;
      }
    }
    graph->num_capacitors +=
        ground_cap_nodes + static_cast<int>(net.capacitors.size());
  }
}

const char* yes_no(bool value) {
  return value ? "true" : "false";
}

template <typename Rule>
struct SpefAction : pegtl::nothing<Rule> {};

template <>
struct SpefAction<spef::RuleDesign> {
  template <typename Input>
  static void apply(const Input& in, SpefBuilder& builder) {
    builder.graph.design_name = spef::RemoveHeaderKey(in, sizeof("*DESIGN"));
  }
};

template <>
struct SpefAction<spef::RuleDesignFlow> {
  template <typename Input>
  static void apply(const Input& in, SpefBuilder& builder) {
    builder.graph.includes_pin_caps = includes_pin_caps_from_design_flow(
        spef::RemoveHeaderKey(in, sizeof("*DESIGN_FLOW")));
  }
};

template <>
struct SpefAction<spef::RuleNameMap> {
  template <typename Input>
  static void apply(const Input& in, SpefBuilder& builder) {
    spef::split_on_space(in.begin(), in.end(), builder.tokens);
    if (builder.tokens.size() < 2 || builder.tokens[0].size() <= 1) {
      return;
    }
    std::size_t key = 0;
    if (parse_unsigned(builder.tokens[0].substr(1), &key)) {
      if (key >= builder.name_map.size()) {
        builder.name_map.resize(key + 1);
      }
      builder.name_map[key] = string_from_view(builder.tokens[1]);
    }
  }
};

template <>
struct SpefAction<spef::RuleNetBeg> {
  template <typename Input>
  static void apply(const Input& in, SpefBuilder& builder) {
    spef::split_on_space(in.begin(), in.end(), builder.tokens);
    builder.current_raw_node_index.clear();
    RcNet net;
    net.name = builder.tokens.size() > 1 ? resolve_name(builder, builder.tokens[1])
                                         : std::string();
    if (builder.tokens.size() > 2) {
      net.total_capacitance = parse_double(builder.tokens[2]);
    }
    builder.graph.nets.push_back(std::move(net));
    builder.current_net = &builder.graph.nets.back();
  }
};

template <>
struct SpefAction<spef::RuleConn> {
  template <typename Input>
  static void apply(const Input& in, SpefBuilder& builder) {
    if (builder.current_net == nullptr) {
      throw pegtl::parse_error("CONN entry without current net", in);
    }
    spef::split_on_space(in.begin(), in.end(), builder.tokens);
    if (builder.tokens.size() < 3) {
      return;
    }

    RcConnection connection;
    connection.kind = builder.tokens[0].size() > 1 && builder.tokens[0][1] == 'P'
                          ? RcConnectionKind::kPort
                          : RcConnectionKind::kInstancePin;
    connection.node = node_index(&builder, builder.tokens[1]);
    connection.direction = parse_direction(builder.tokens[2][0]);
    builder.current_net->connections.push_back(connection);
  }
};

template <>
struct SpefAction<spef::RuleCapGround> {
  template <typename Input>
  static void apply(const Input& in, SpefBuilder& builder) {
    if (builder.current_net == nullptr) {
      throw pegtl::parse_error("CAP entry without current net", in);
    }
    spef::split_on_space(in.begin(), in.end(), builder.tokens);
    if (builder.tokens.size() < 3) {
      return;
    }

    add_ground_cap(&builder, builder.tokens[1], parse_double(builder.tokens[2]));
  }
};

template <>
struct SpefAction<spef::RuleCapCouple> {
  template <typename Input>
  static void apply(const Input& in, SpefBuilder& builder) {
    if (builder.current_net == nullptr) {
      throw pegtl::parse_error("CAP entry without current net", in);
    }
    spef::split_on_space(in.begin(), in.end(), builder.tokens);
    if (builder.tokens.size() < 4) {
      return;
    }

    const double capacitance = parse_double(builder.tokens[3]);
    if (builder.options.keep_coupling_caps) {
      RcCapacitor capacitor;
      capacitor.node1 = node_index(&builder, builder.tokens[1]);
      capacitor.node2 = node_index(&builder, builder.tokens[2]);
      capacitor.capacitance = capacitance;
      builder.current_net->capacitors.push_back(capacitor);
    } else {
      const double scaled_capacitance =
          capacitance * builder.options.coupling_reduction_factor;
      add_ground_cap(&builder, builder.tokens[1], scaled_capacitance);
      if (raw_node_is_local_to_current_net(builder, builder.tokens[2])) {
        add_ground_cap(&builder, builder.tokens[2], scaled_capacitance);
      }
    }
  }
};

template <>
struct SpefAction<spef::RuleRes> {
  template <typename Input>
  static void apply(const Input& in, SpefBuilder& builder) {
    if (builder.current_net == nullptr) {
      throw pegtl::parse_error("RES entry without current net", in);
    }
    spef::split_on_space(in.begin(), in.end(), builder.tokens);
    if (builder.tokens.size() < 4) {
      return;
    }

    RcResistor resistor;
    resistor.node1 = node_index(&builder, builder.tokens[1]);
    resistor.node2 = node_index(&builder, builder.tokens[2]);
    resistor.resistance = parse_double(builder.tokens[3]);
    builder.current_net->resistors.push_back(resistor);
  }
};

template <>
struct SpefAction<spef::RuleNetEnd> {
  template <typename Input>
  static void apply(const Input&, SpefBuilder& builder) {
    builder.current_net = nullptr;
    builder.current_raw_node_index.clear();
  }
};

template <>
struct SpefAction<spef::RuleInputEnd> {
  template <typename Input>
  static void apply(const Input& in, SpefBuilder&) {
    if (in.size() != 0) {
      throw pegtl::parse_error("Unrecognized token", in);
    }
  }
};

}  // namespace

RcGraph load_spef_rc_graph(const std::string& path,
                           const SpefParseOptions& options) {
  const auto total_start = std::chrono::steady_clock::now();
  auto stage_start = total_start;
  std::string buffer = file_to_memory(path);
  if (buffer.empty()) {
    throw std::runtime_error("failed to open or read SPEF file: " + path);
  }
  auto stage_end = std::chrono::steady_clock::now();
  std::cerr << "[stage] spef_read_file done_ms="
            << std::chrono::duration<double, std::milli>(stage_end -
                                                          stage_start)
                   .count()
            << " bytes=" << buffer.size() << '\n'
            << std::flush;

  SpefBuilder builder;
  builder.options = options;
  builder.name_map.reserve(std::max<std::size_t>(1024, buffer.size() / 180));
  builder.graph.nets.reserve(std::max<std::size_t>(1024, buffer.size() / 2200));
  builder.current_raw_node_index.reserve(128);
  stage_start = std::chrono::steady_clock::now();
  parse_spef_lines(buffer, &builder);
  stage_end = std::chrono::steady_clock::now();
  std::cerr << "[stage] spef_parse_lines done_ms="
            << std::chrono::duration<double, std::milli>(stage_end -
                                                          stage_start)
                   .count()
            << " nets=" << builder.graph.nets.size() << '\n'
            << std::flush;
  if (builder.graph.nets.empty()) {
    throw std::runtime_error("failed to parse SPEF file: no *D_NET entries in " +
                             path);
  }

  stage_start = std::chrono::steady_clock::now();
  recompute_graph_counts(&builder.graph);
  stage_end = std::chrono::steady_clock::now();
  std::cerr << "[stage] spef_recompute_counts done_ms="
            << std::chrono::duration<double, std::milli>(stage_end -
                                                          stage_start)
                   .count()
            << '\n';
  std::cerr << "[stage] spef_total done_ms="
            << std::chrono::duration<double, std::milli>(stage_end -
                                                          total_start)
                   .count()
            << '\n'
            << std::flush;
  return builder.graph;
}

std::string format_rc_summary(const RcGraph& graph) {
  std::ostringstream os;
  os << "spef_design: "
     << (graph.design_name.empty() ? "<unknown>" : graph.design_name) << '\n';
  os << "rc_nets: " << graph.num_nets << '\n';
  os << "rc_nodes: " << graph.num_nodes << '\n';
  os << "rc_connections: " << graph.num_connections << '\n';
  os << "rc_resistors: " << graph.num_resistors << '\n';
  os << "rc_capacitors: " << graph.num_capacitors << '\n';
  os << "rc_coupling_capacitors: " << graph.num_coupling_capacitors << '\n';
  os << "rc_ground_cap_nodes: " << graph.num_ground_cap_nodes << '\n';
  os << "includes_pin_caps: " << yes_no(graph.includes_pin_caps) << '\n';
  const std::size_t max_summary_nets = 5;
  for (std::size_t i = 0; i < graph.nets.size() && i < max_summary_nets; ++i) {
    const auto& net = graph.nets[i];
    os << "  - net=" << net.name << " total_cap=" << net.total_capacitance
       << " nodes=" << net.nodes.size()
       << " conns=" << net.connections.size()
       << " ground_cap_nodes="
       << std::count_if(net.nodes.begin(), net.nodes.end(), [](const RcNode& node) {
            return node.ground_capacitance > 0.0;
          })
       << " coupling_caps=" << net.capacitors.size()
       << " res=" << net.resistors.size() << '\n';
  }
  if (graph.nets.size() > max_summary_nets) {
    os << "  ... " << (graph.nets.size() - max_summary_nets)
       << " more nets\n";
  }
  return os.str();
}

}  // namespace stimer
