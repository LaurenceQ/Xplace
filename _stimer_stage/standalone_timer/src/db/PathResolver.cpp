#include "stimer/PathResolver.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace stimer {

namespace {

std::string to_lower(std::string value) {
  for (char& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

bool exists_regular_file(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::exists(path, error) &&
         std::filesystem::is_regular_file(path, error);
}

bool exists_directory(const std::filesystem::path& path) {
  std::error_code error;
  return std::filesystem::exists(path, error) &&
         std::filesystem::is_directory(path, error);
}

bool has_suffix(const std::string& value, const std::string& suffix) {
  if (suffix.size() > value.size()) {
    return false;
  }
  return value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
         0;
}

bool is_liberty_file(const std::filesystem::path& path) {
  const std::string name = to_lower(path.filename().string());
  return has_suffix(name, ".lib") || has_suffix(name, ".lib.gz");
}

bool should_skip_liberty(const std::filesystem::path& path,
                         bool include_ram_libs) {
  if (include_ram_libs) {
    return false;
  }
  return to_lower(path.filename().string()).find("ram") != std::string::npos;
}

std::vector<std::filesystem::path> platform_lib_dirs(
    const std::filesystem::path& platform) {
  std::vector<std::filesystem::path> dirs;
  dirs.push_back(platform / "LIB");
  dirs.push_back(platform / "lib");
  dirs.push_back(platform);

  std::vector<std::filesystem::path> unique_dirs;
  for (const auto& dir : dirs) {
    const auto normalized = dir.lexically_normal();
    if (std::find(unique_dirs.begin(), unique_dirs.end(), normalized) ==
        unique_dirs.end()) {
      unique_dirs.push_back(normalized);
    }
  }
  return unique_dirs;
}

std::vector<std::string> find_liberty_files(
    const std::filesystem::path& platform,
    bool include_ram_libs,
    std::vector<std::string>* notes) {
  std::vector<std::string> files;
  for (const auto& dir : platform_lib_dirs(platform)) {
    if (!exists_directory(dir)) {
      continue;
    }

    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(dir, error)) {
      if (error) {
        notes->push_back("failed to scan liberty dir: " + dir.string() +
                         ": " + error.message());
        break;
      }
      if (!entry.is_regular_file(error) || error) {
        continue;
      }
      const auto& path = entry.path();
      if (!is_liberty_file(path) || should_skip_liberty(path, include_ram_libs)) {
        continue;
      }
      files.push_back(path.string());
    }
  }

  std::sort(files.begin(), files.end());
  files.erase(std::unique(files.begin(), files.end()), files.end());
  return files;
}

std::string pick_existing(const std::vector<std::filesystem::path>& paths) {
  for (const auto& path : paths) {
    if (exists_regular_file(path)) {
      return path.string();
    }
  }
  return "";
}

std::vector<std::filesystem::path> sorted_files_with_extension(
    const std::filesystem::path& dir,
    const std::string& extension) {
  std::vector<std::filesystem::path> files;
  if (!exists_directory(dir)) {
    return files;
  }

  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(dir, error)) {
    if (error) {
      break;
    }
    if (!entry.is_regular_file(error) || error) {
      continue;
    }
    if (to_lower(entry.path().extension().string()) == extension) {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

bool is_secondary_def(const std::filesystem::path& path) {
  const std::string name = to_lower(path.filename().string());
  return has_suffix(name, ".cells.def") || has_suffix(name, ".fp.def") ||
         has_suffix(name, ".ref.def");
}

std::string pick_def_file(const std::filesystem::path& design_dir,
                          const std::string& design_name) {
  std::vector<std::filesystem::path> candidates = {
      design_dir / (design_name + ".def"),
      design_dir / ("20-" + design_name + ".def"),
  };
  const std::string direct = pick_existing(candidates);
  if (!direct.empty()) {
    return direct;
  }

  for (const auto& path : sorted_files_with_extension(design_dir, ".def")) {
    if (!is_secondary_def(path)) {
      return path.string();
    }
  }
  return "";
}

std::string pick_first_by_extension(const std::filesystem::path& design_dir,
                                    const std::string& extension) {
  const auto files = sorted_files_with_extension(design_dir, extension);
  if (files.empty()) {
    return "";
  }
  return files.front().string();
}

std::string infer_design_name(const PathRequest& request,
                              const std::filesystem::path& design_dir) {
  if (!request.design_name.empty()) {
    return request.design_name;
  }
  if (!design_dir.empty()) {
    return design_dir.filename().string();
  }
  throw std::runtime_error("missing design name: pass --design or --design-dir");
}

std::filesystem::path resolve_design_dir(const PathRequest& request) {
  if (!request.design_dir.empty()) {
    return std::filesystem::path(request.design_dir);
  }
  if (!request.design_root.empty() && !request.design_name.empty()) {
    return std::filesystem::path(request.design_root) / request.design_name;
  }
  if (!request.design_root.empty()) {
    return std::filesystem::path(request.design_root);
  }
  throw std::runtime_error(
      "missing design path: pass --design-dir or --design-root");
}

}  // namespace

PathResult resolve_xplace_paths(const PathRequest& request) {
  if (request.platform_path.empty()) {
    throw std::runtime_error("missing platform path: pass --platform");
  }

  PathResult result;
  const std::filesystem::path platform(request.platform_path);
  const std::filesystem::path design_dir = resolve_design_dir(request);
  const std::string design_name = infer_design_name(request, design_dir);

  result.design_dir = design_dir.string();
  result.design_name = design_name;

  result.liberty_files =
      find_liberty_files(platform, request.include_ram_libs, &result.notes);
  if (result.liberty_files.empty()) {
    result.notes.push_back("no Liberty files found under platform path");
  }

  result.def_file = pick_def_file(design_dir, design_name);
  result.verilog_file =
      pick_existing({design_dir / (design_name + ".v")});
  if (result.verilog_file.empty()) {
    result.verilog_file = pick_first_by_extension(design_dir, ".v");
  }
  result.spef_file =
      pick_existing({design_dir / (design_name + ".spef"),
                     design_dir / ("20-" + design_name + ".spef")});
  if (result.spef_file.empty()) {
    result.spef_file = pick_first_by_extension(design_dir, ".spef");
  }
  result.sdc_file =
      pick_existing({design_dir / (design_name + ".sdc"),
                     design_dir / (design_name + ".cts_1.sdc")});
  if (result.sdc_file.empty()) {
    result.sdc_file = pick_first_by_extension(design_dir, ".sdc");
  }

  return result;
}

void apply_path_result(const PathResult& paths, TimerConfig* config) {
  if (config == nullptr) {
    throw std::runtime_error("apply_path_result got null config");
  }
  if (config->liberty_files.empty()) {
    config->liberty_files = paths.liberty_files;
  }
  if (config->def_file.empty()) {
    config->def_file = paths.def_file;
  }
  if (config->verilog_file.empty()) {
    config->verilog_file = paths.verilog_file;
  }
  if (config->spef_file.empty()) {
    config->spef_file = paths.spef_file;
  }
  if (config->sdc_file.empty()) {
    config->sdc_file = paths.sdc_file;
  }
}

std::string format_path_result(const PathResult& paths) {
  std::ostringstream os;
  os << "resolved_design: "
     << (paths.design_name.empty() ? "<none>" : paths.design_name) << '\n';
  os << "resolved_design_dir: "
     << (paths.design_dir.empty() ? "<none>" : paths.design_dir) << '\n';
  os << "resolved_liberty_files: " << paths.liberty_files.size() << '\n';
  for (const auto& liberty_file : paths.liberty_files) {
    os << "  - " << liberty_file << '\n';
  }
  os << "resolved_def: "
     << (paths.def_file.empty() ? "<none>" : paths.def_file) << '\n';
  os << "resolved_verilog: "
     << (paths.verilog_file.empty() ? "<none>" : paths.verilog_file) << '\n';
  os << "resolved_spef: "
     << (paths.spef_file.empty() ? "<none>" : paths.spef_file) << '\n';
  os << "resolved_sdc: "
     << (paths.sdc_file.empty() ? "<none>" : paths.sdc_file) << '\n';
  if (!paths.notes.empty()) {
    os << "path_notes: " << paths.notes.size() << '\n';
    for (const auto& note : paths.notes) {
      os << "  - " << note << '\n';
    }
  }
  return os.str();
}

}  // namespace stimer
