#include "stimer/DesignDB.h"

#include <algorithm>
#include <sstream>

namespace stimer {

namespace {

std::string connection_key(const DesignConnection& connection) {
  std::string key;
  key.reserve(connection.instance_name.size() + connection.pin_name.size() + 4);
  key.push_back(connection.is_port ? 'P' : 'I');
  key.push_back('\x1f');
  key.append(connection.instance_name);
  key.push_back('\x1f');
  key.append(connection.pin_name);
  return key;
}

}  // namespace

void DesignDB::rebuild_indices() {
  rebuild_name_indices();

  for (auto& net : nets) {
    net.connection_keys.clear();
    net.connection_keys.reserve(net.connections.size());
    for (const auto& connection : net.connections) {
      net.connection_keys.insert(connection_key(connection));
    }
  }
}

void DesignDB::rebuild_name_indices() {
  instance_index.clear();
  port_index.clear();
  net_index.clear();

  for (std::size_t i = 0; i < instances.size(); ++i) {
    instance_index[instances[i].name] = i;
  }
  for (std::size_t i = 0; i < ports.size(); ++i) {
    port_index[ports[i].name] = i;
  }
  for (std::size_t i = 0; i < nets.size(); ++i) {
    net_index[nets[i].name] = i;
  }
}

void DesignDB::update_counts() {
  rebuild_indices();
  update_counts_preserve_connection_keys();
}

void DesignDB::update_counts_preserve_connection_keys() {
  rebuild_name_indices();
  num_instances = static_cast<int>(instances.size());
  num_nets = static_cast<int>(nets.size());

  int instance_pin_connections = 0;
  for (const auto& net : nets) {
    for (const auto& connection : net.connections) {
      if (!connection.is_port) {
        ++instance_pin_connections;
      }
    }
  }
  num_pins = static_cast<int>(ports.size()) + instance_pin_connections;
}

DesignInstance& DesignDB::get_or_add_instance(const std::string& name,
                                              const std::string& cell_name) {
  auto iter = instance_index.find(name);
  if (iter != instance_index.end()) {
    DesignInstance& instance = instances[iter->second];
    if (instance.cell_name.empty() && !cell_name.empty()) {
      instance.cell_name = cell_name;
    }
    return instance;
  }

  const std::size_t id = instances.size();
  instance_index[name] = id;
  DesignInstance instance;
  instance.name = name;
  instance.cell_name = cell_name;
  instances.push_back(std::move(instance));
  return instances.back();
}

DesignPort& DesignDB::get_or_add_port(const std::string& name,
                                      const std::string& net_name,
                                      DesignPinDirection direction) {
  auto iter = port_index.find(name);
  if (iter != port_index.end()) {
    DesignPort& port = ports[iter->second];
    if (port.net_name.empty() && !net_name.empty()) {
      port.net_name = net_name;
    }
    if (port.direction == DesignPinDirection::kUnknown &&
        direction != DesignPinDirection::kUnknown) {
      port.direction = direction;
    }
    return port;
  }

  const std::size_t id = ports.size();
  port_index[name] = id;
  DesignPort port;
  port.name = name;
  port.net_name = net_name;
  port.direction = direction;
  ports.push_back(std::move(port));
  return ports.back();
}

DesignNet& DesignDB::get_or_add_net(const std::string& name) {
  auto iter = net_index.find(name);
  if (iter != net_index.end()) {
    return nets[iter->second];
  }

  const std::size_t id = nets.size();
  net_index[name] = id;
  DesignNet net;
  net.name = name;
  nets.push_back(std::move(net));
  return nets.back();
}

void DesignDB::reserve_net_connections(DesignNet* net, std::size_t count) {
  if (net == nullptr || count == 0) {
    return;
  }
  const std::size_t target = net->connections.size() + count;
  if (net->connections.capacity() < target) {
    net->connections.reserve(target);
  }
  if (net->connection_keys.bucket_count() < target) {
    net->connection_keys.reserve(target);
  }
}

void DesignDB::add_connection_to_net(const std::string& net_name,
                                     DesignConnection connection) {
  DesignNet& net = get_or_add_net(net_name);
  const std::string key = connection_key(connection);
  if (net.connection_keys.insert(key).second) {
    net.connections.push_back(std::move(connection));
  }
}

const char* design_pin_direction_name(DesignPinDirection direction) {
  switch (direction) {
    case DesignPinDirection::kInput:
      return "input";
    case DesignPinDirection::kOutput:
      return "output";
    case DesignPinDirection::kInout:
      return "inout";
    case DesignPinDirection::kUnknown:
      return "unknown";
  }
  return "unknown";
}

std::string format_design_summary(const DesignDB& design) {
  std::ostringstream os;
  os << "design_name: "
     << (design.design_name.empty() ? "<none>" : design.design_name) << '\n';
  os << "db_units_per_micron: " << design.db_units_per_micron << '\n';
  os << "instances: " << design.num_instances << '\n';
  os << "ports: " << design.ports.size() << '\n';
  os << "nets: " << design.num_nets << '\n';
  os << "pins: " << design.num_pins << '\n';

  const std::size_t instance_limit = std::min<std::size_t>(design.instances.size(), 3);
  for (std::size_t i = 0; i < instance_limit; ++i) {
    const auto& instance = design.instances[i];
    os << "  - inst=" << instance.name
       << " cell=" << (instance.cell_name.empty() ? "<unknown>" : instance.cell_name)
       << '\n';
  }

  const std::size_t net_limit = std::min<std::size_t>(design.nets.size(), 3);
  for (std::size_t i = 0; i < net_limit; ++i) {
    const auto& net = design.nets[i];
    os << "  - net=" << net.name
       << " conns=" << net.connections.size();
    if (!net.use.empty()) {
      os << " use=" << net.use;
    }
    os << '\n';
  }
  return os.str();
}

}  // namespace stimer
