#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "stimer/TimerConfig.h"

namespace stimer {

enum class InputKind {
  kLiberty,
  kDef,
  kVerilog,
  kSpef,
  kSdc,
};

struct InputFileRecord {
  InputKind kind = InputKind::kLiberty;
  std::string path;
  bool required = true;
  bool exists = false;
  bool is_regular_file = false;
  std::uintmax_t size_bytes = 0;
  std::string error;
};

struct InputProbeResult {
  std::vector<InputFileRecord> files;
  std::vector<std::string> errors;
  std::uintmax_t total_size_bytes = 0;

  bool ok() const;
};

const char* input_kind_name(InputKind kind);
InputProbeResult probe_input_files(const TimerConfig& config);
std::string format_input_probe(const InputProbeResult& probe);
std::string format_input_errors(const InputProbeResult& probe);

}  // namespace stimer
