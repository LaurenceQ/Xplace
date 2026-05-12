#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace stimer {

enum class DesignPinDirection {
  kUnknown,
  kInput,
  kOutput,
  kInout,
};

struct DesignConnection {
  std::string instance_name;
  std::string pin_name;
  bool is_port = false;
};

struct DesignInstance {
  std::string name;
  std::string cell_name;
  int x = 0;
  int y = 0;
  int orient = -1;
  bool placed = false;
  bool fixed = false;
};

struct DesignPort {
  std::string name;
  std::string net_name;
  DesignPinDirection direction = DesignPinDirection::kUnknown;
};

struct DesignNet {
  std::string name;
  std::string use;
  std::vector<DesignConnection> connections;
  std::unordered_set<std::string> connection_keys;
};

struct DesignDB {
  std::string design_name;
  int db_units_per_micron = 0;

  std::vector<DesignInstance> instances;
  std::vector<DesignPort> ports;
  std::vector<DesignNet> nets;

  std::unordered_map<std::string, std::size_t> instance_index;
  std::unordered_map<std::string, std::size_t> port_index;
  std::unordered_map<std::string, std::size_t> net_index;

  int num_instances = 0;
  int num_nets = 0;
  int num_pins = 0;

  void rebuild_indices();
  void rebuild_name_indices();
  void update_counts();
  void update_counts_preserve_connection_keys();

  DesignInstance& get_or_add_instance(const std::string& name,
                                      const std::string& cell_name);
  DesignPort& get_or_add_port(const std::string& name,
                              const std::string& net_name,
                              DesignPinDirection direction);
  DesignNet& get_or_add_net(const std::string& name);
  void reserve_net_connections(DesignNet* net, std::size_t count);
  void add_connection_to_net(const std::string& net_name,
                             DesignConnection connection);
};

const char* design_pin_direction_name(DesignPinDirection direction);
std::string format_design_summary(const DesignDB& design);

}  // namespace stimer
