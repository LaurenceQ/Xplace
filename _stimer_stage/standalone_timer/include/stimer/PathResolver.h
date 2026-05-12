#pragma once

#include <string>
#include <vector>

#include "stimer/TimerConfig.h"

namespace stimer {

struct PathRequest {
  std::string platform_path;
  std::string design_root;
  std::string design_dir;
  std::string design_name;
  bool include_ram_libs = false;
};

struct PathResult {
  std::vector<std::string> liberty_files;
  std::string def_file;
  std::string verilog_file;
  std::string spef_file;
  std::string sdc_file;
  std::string design_dir;
  std::string design_name;
  std::vector<std::string> notes;
};

PathResult resolve_xplace_paths(const PathRequest& request);
void apply_path_result(const PathResult& paths, TimerConfig* config);
std::string format_path_result(const PathResult& paths);

}  // namespace stimer
