#include "stimer/TimerConfig.h"

namespace stimer {

std::string timing_mode_name(TimingMode mode) {
  switch (mode) {
    case TimingMode::kElmore:
      return "elmore";
    case TimingMode::kDmp:
      return "dmp";
  }
  return "unknown";
}

}  // namespace stimer
