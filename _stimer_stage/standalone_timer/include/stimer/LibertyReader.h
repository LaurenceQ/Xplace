#pragma once

#include <string>
#include <vector>

#include "stimer/LibertyDB.h"

namespace stimer {

LibertyDB load_liberty_files(const std::vector<std::string>& paths);

}  // namespace stimer
