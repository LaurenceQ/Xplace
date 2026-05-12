#pragma once

#include <string>
#include <vector>

namespace stimer {

enum class RcConnectionKind {
  kInternal,
  kPort,
  kInstancePin,
};

enum class RcDirection {
  kUnknown,
  kInput,
  kOutput,
  kInout,
};

struct RcNode {
  std::string name;
  double ground_capacitance = 0.0;
};

struct RcConnection {
  int node = -1;
  RcConnectionKind kind = RcConnectionKind::kInternal;
  RcDirection direction = RcDirection::kUnknown;
};

struct RcCapacitor {
  int node1 = -1;
  int node2 = -1;
  double capacitance = 0.0;

  bool is_coupling() const { return node2 >= 0; }
};

struct RcResistor {
  int node1 = -1;
  int node2 = -1;
  double resistance = 0.0;
};

struct RcNet {
  std::string name;
  double total_capacitance = 0.0;
  std::vector<RcNode> nodes;
  std::vector<RcConnection> connections;
  std::vector<RcCapacitor> capacitors;
  std::vector<RcResistor> resistors;
};

struct RcGraph {
  int num_nets = 0;
  int num_nodes = 0;
  int num_resistors = 0;
  int num_capacitors = 0;
  int num_coupling_capacitors = 0;
  int num_ground_cap_nodes = 0;
  int num_connections = 0;
  bool includes_pin_caps = false;
  std::string design_name;
  std::vector<RcNet> nets;
};

std::string format_rc_summary(const RcGraph& graph);

}  // namespace stimer
