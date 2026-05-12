#include "stimer/InputFiles.h"

#include <filesystem>
#include <sstream>
#include <system_error>
#include <utility>

namespace stimer {

namespace {

InputFileRecord probe_one(InputKind kind,
                          const std::string& path,
                          bool required) {
  InputFileRecord record;
  record.kind = kind;
  record.path = path;
  record.required = required;

  if (path.empty()) {
    record.error = required ? "missing required path" : "";
    return record;
  }

  const std::filesystem::path fs_path(path);
  std::error_code error;
  record.exists = std::filesystem::exists(fs_path, error);
  if (error) {
    record.error = error.message();
    return record;
  }
  if (!record.exists) {
    record.error = "file does not exist";
    return record;
  }

  record.is_regular_file = std::filesystem::is_regular_file(fs_path, error);
  if (error) {
    record.error = error.message();
    return record;
  }
  if (!record.is_regular_file) {
    record.error = "path is not a regular file";
    return record;
  }

  record.size_bytes = std::filesystem::file_size(fs_path, error);
  if (error) {
    record.error = error.message();
    record.size_bytes = 0;
    return record;
  }

  return record;
}

bool has_any_file(const InputProbeResult& probe, InputKind kind) {
  for (const auto& file : probe.files) {
    if (file.kind == kind && file.error.empty()) {
      return true;
    }
  }
  return false;
}

void append_probe(InputProbeResult* probe, InputFileRecord record) {
  if (!record.error.empty()) {
    std::ostringstream error;
    error << input_kind_name(record.kind) << " `" << record.path
          << "`: " << record.error;
    probe->errors.push_back(error.str());
  } else if (!record.path.empty()) {
    probe->total_size_bytes += record.size_bytes;
  }
  probe->files.push_back(std::move(record));
}

}  // namespace

bool InputProbeResult::ok() const {
  return errors.empty();
}

const char* input_kind_name(InputKind kind) {
  switch (kind) {
    case InputKind::kLiberty:
      return "liberty";
    case InputKind::kDef:
      return "def";
    case InputKind::kVerilog:
      return "verilog";
    case InputKind::kSpef:
      return "spef";
    case InputKind::kSdc:
      return "sdc";
  }
  return "unknown";
}

InputProbeResult probe_input_files(const TimerConfig& config) {
  InputProbeResult probe;

  if (config.liberty_files.empty()) {
    probe.errors.push_back("missing required input: --liberty <file>");
  }
  for (const auto& liberty_file : config.liberty_files) {
    append_probe(&probe, probe_one(InputKind::kLiberty, liberty_file, true));
  }

  if (config.def_file.empty() && config.verilog_file.empty()) {
    probe.errors.push_back("missing required input: --def or --verilog");
  }
  if (!config.def_file.empty()) {
    append_probe(&probe, probe_one(InputKind::kDef, config.def_file, true));
  }
  if (!config.verilog_file.empty()) {
    append_probe(&probe,
                 probe_one(InputKind::kVerilog, config.verilog_file, true));
  }

  if (config.spef_file.empty()) {
    probe.errors.push_back("missing required input: --spef <file>");
  } else {
    append_probe(&probe, probe_one(InputKind::kSpef, config.spef_file, true));
  }

  if (config.sdc_file.empty()) {
    probe.errors.push_back("missing required input: --sdc <file>");
  } else {
    append_probe(&probe, probe_one(InputKind::kSdc, config.sdc_file, true));
  }

  if (!config.liberty_files.empty() &&
      !has_any_file(probe, InputKind::kLiberty)) {
    probe.errors.push_back("no usable Liberty file after probing");
  }
  if ((!config.def_file.empty() || !config.verilog_file.empty()) &&
      !has_any_file(probe, InputKind::kDef) &&
      !has_any_file(probe, InputKind::kVerilog)) {
    probe.errors.push_back("no usable DEF/Verilog file after probing");
  }

  return probe;
}

std::string format_input_probe(const InputProbeResult& probe) {
  std::ostringstream os;
  os << "probed_files: " << probe.files.size() << '\n';
  os << "total_input_bytes: " << probe.total_size_bytes << '\n';
  for (const auto& file : probe.files) {
    os << "  - kind=" << input_kind_name(file.kind)
       << " path=" << file.path
       << " exists=" << (file.exists ? "true" : "false")
       << " regular=" << (file.is_regular_file ? "true" : "false")
       << " bytes=" << file.size_bytes;
    if (!file.error.empty()) {
      os << " error=\"" << file.error << "\"";
    }
    os << '\n';
  }
  if (!probe.errors.empty()) {
    os << "input_errors: " << probe.errors.size() << '\n';
    for (const auto& error : probe.errors) {
      os << "  - " << error << '\n';
    }
  }
  return os.str();
}

std::string format_input_errors(const InputProbeResult& probe) {
  std::ostringstream os;
  os << "invalid timing inputs";
  for (const auto& error : probe.errors) {
    os << "\n  - " << error;
  }
  return os.str();
}

}  // namespace stimer
