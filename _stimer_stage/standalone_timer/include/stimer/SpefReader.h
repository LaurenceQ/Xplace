#pragma once

#include <string>

#include "stimer/RcGraph.h"

namespace stimer {

struct SpefParseOptions {
  bool keep_coupling_caps = false;
  double coupling_reduction_factor = 1.0;
};

RcGraph load_spef_rc_graph(const std::string& path,
                           const SpefParseOptions& options = {});

}  // namespace stimer
