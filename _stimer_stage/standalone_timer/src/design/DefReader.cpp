#include "stimer/DefReader.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

// tcl has another definition of EXTERN in some build environments.
#undef EXTERN

#include "def58/include/defiComponent.hpp"
#include "def58/include/defiNet.hpp"
#include "def58/include/defiPinCap.hpp"
#include "def58/include/defrReader.hpp"

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

DesignPinDirection direction_from_def(const char* direction) {
  if (direction == nullptr) {
    return DesignPinDirection::kUnknown;
  }
  if (std::strcmp(direction, "INPUT") == 0) {
    return DesignPinDirection::kInput;
  }
  if (std::strcmp(direction, "OUTPUT") == 0) {
    return DesignPinDirection::kOutput;
  }
  if (std::strcmp(direction, "INOUT") == 0) {
    return DesignPinDirection::kInout;
  }
  return DesignPinDirection::kUnknown;
}

struct DefBuildContext {
  DesignDB design;
};

int on_def_design(defrCallbackType_e, const char* name, defiUserData user_data) {
  auto* context = reinterpret_cast<DefBuildContext*>(user_data);
  context->design.design_name = name == nullptr ? std::string{} : validate_token(name);
  return 0;
}

int on_def_units(defrCallbackType_e, double units, defiUserData user_data) {
  auto* context = reinterpret_cast<DefBuildContext*>(user_data);
  context->design.db_units_per_micron = static_cast<int>(units);
  return 0;
}

int on_def_component_start(defrCallbackType_e, int count, defiUserData user_data) {
  auto* context = reinterpret_cast<DefBuildContext*>(user_data);
  if (count > 0) {
    context->design.instances.reserve(static_cast<std::size_t>(count));
  }
  return 0;
}

int on_def_component(defrCallbackType_e, defiComponent* component,
                     defiUserData user_data) {
  auto* context = reinterpret_cast<DefBuildContext*>(user_data);
  const std::string instance_name = validate_token(component->id());
  const std::string cell_name = component->name() == nullptr
                                    ? std::string{}
                                    : validate_token(component->name());
  DesignInstance& instance =
      context->design.get_or_add_instance(instance_name, cell_name);

  if (component->isPlaced() || component->isFixed()) {
    instance.x = component->placementX();
    instance.y = component->placementY();
    instance.orient = component->placementOrient();
    instance.placed = true;
    instance.fixed = component->isFixed();
  } else if (component->isUnplaced()) {
    instance.placed = false;
    instance.fixed = false;
  }
  return 0;
}

int on_def_pin(defrCallbackType_e, defiPin* pin, defiUserData user_data) {
  auto* context = reinterpret_cast<DefBuildContext*>(user_data);
  const std::string pin_name = validate_token(pin->pinName());
  const std::string net_name =
      pin->netName() == nullptr ? std::string{} : validate_token(pin->netName());
  context->design.get_or_add_port(pin_name, net_name,
                                  direction_from_def(pin->direction()));
  if (!net_name.empty()) {
    context->design.get_or_add_net(net_name);
  }
  return 0;
}

int on_def_net_start(defrCallbackType_e, int count, defiUserData user_data) {
  auto* context = reinterpret_cast<DefBuildContext*>(user_data);
  if (count > 0) {
    context->design.nets.reserve(static_cast<std::size_t>(count));
  }
  return 0;
}

int on_def_net(defrCallbackType_e, defiNet* def_net, defiUserData user_data) {
  auto* context = reinterpret_cast<DefBuildContext*>(user_data);
  if (def_net->numConnections() == 0) {
    return 0;
  }

  const std::string net_name = validate_token(def_net->name());
  if (net_name == "VDD" || net_name == "VSS") {
    return 0;
  }

  DesignNet& net = context->design.get_or_add_net(net_name);
  if (def_net->hasUse()) {
    net.use = def_net->use();
  }
  context->design.reserve_net_connections(
      &net, static_cast<std::size_t>(def_net->numConnections()));

  for (int i = 0; i < def_net->numConnections(); ++i) {
    const char* instance = def_net->instance(i);
    const char* pin = def_net->pin(i);
    if (instance == nullptr || pin == nullptr) {
      continue;
    }

    DesignConnection connection;
    connection.pin_name = validate_token(pin);
    if (std::strcmp(instance, "PIN") == 0) {
      connection.is_port = true;
      context->design.get_or_add_port(connection.pin_name, net_name,
                                      DesignPinDirection::kUnknown);
    } else {
      connection.instance_name = validate_token(instance);
      context->design.get_or_add_instance(connection.instance_name, "");
    }
    context->design.add_connection_to_net(net_name, std::move(connection));
  }
  return 0;
}

void set_def_callbacks() {
  defrSetDesignCbk(on_def_design);
  defrSetUnitsCbk(on_def_units);
  defrSetComponentStartCbk(on_def_component_start);
  defrSetComponentCbk(on_def_component);
  defrSetPinCbk(on_def_pin);
  defrSetNetStartCbk(on_def_net_start);
  defrSetNetCbk(on_def_net);
}

}  // namespace

DesignDB read_def_design(const std::string& path) {
  FILE* file = std::fopen(path.c_str(), "r");
  if (file == nullptr) {
    throw std::runtime_error("failed to open DEF file: " + path);
  }

  DefBuildContext context;
  set_def_callbacks();
  defrInit();
  defrReset();
  const int result = defrRead(file, path.c_str(), &context, 1);
  defrReleaseNResetMemory();
  defrUnsetCallbacks();
  std::fclose(file);

  if (result != 0) {
    throw std::runtime_error("failed to parse DEF file: " + path);
  }

  context.design.update_counts_preserve_connection_keys();
  return context.design;
}

}  // namespace stimer
