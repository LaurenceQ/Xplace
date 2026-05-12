#pragma once

#include <string>

#include "stimer/DesignDB.h"

namespace stimer {

DesignDB read_verilog_design(const std::string& path);
void read_verilog_design_into(const std::string& path, DesignDB* design);

}  // namespace stimer
