#include <cstdlib>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include "stimer/PathResolver.h"
#include "stimer/Timer.h"

namespace {

void print_usage(const char* program) {
  std::cout
      << "usage: " << program << " [options]\n"
      << "\n"
      << "options:\n"
      << "  --liberty <file>    Liberty file, may be repeated\n"
      << "  --def <file>        DEF file\n"
      << "  --verilog <file>    Verilog netlist\n"
      << "  --spef <file>       SPEF parasitics\n"
      << "  --sdc <file>        SDC constraints\n"
      << "  --platform <path>   Xplace/Xplace_dmp platform path\n"
      << "  --design-dir <path> Design directory path\n"
      << "  --design-root <path> Parent directory containing designs\n"
      << "  --design <name>     Design name for path discovery\n"
      << "  --include-ram-lib   Include RAM Liberty files during discovery\n"
      << "  --force-verilog     Read Verilog even when DEF has instances/nets\n"
      << "  --keep-coupling-caps Keep original SPEF coupling capacitors\n"
      << "  --coupling-reduction-factor <value> Coupling cap ground factor, default 1.0\n"
      << "  --mode <dmp|elmore> Timing RC mode, default dmp\n"
      << "  --report <file>     Write summary report\n"
      << "  --dump-net <name>   Net name for debug dump\n"
      << "  --dump-file <file>  Debug dump path for --dump-net\n"
      << "  --cpu               Disable CUDA for this run\n"
      << "  --dump-debug        Enable debug dumps in later phases\n"
      << "  --help              Show this help\n";
}

std::string require_value(int argc, char** argv, int* index) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + argv[*index]);
  }
  ++(*index);
  return argv[*index];
}

stimer::TimingMode parse_mode(const std::string& value) {
  if (value == "dmp") {
    return stimer::TimingMode::kDmp;
  }
  if (value == "elmore") {
    return stimer::TimingMode::kElmore;
  }
  throw std::runtime_error("unknown mode: " + value);
}

template <typename Fn>
void run_stage(const char* name, Fn&& fn) {
  const auto start = std::chrono::steady_clock::now();
  std::cout << "\n[stage] " << name << " start\n" << std::flush;
  fn();
  const auto end = std::chrono::steady_clock::now();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  std::cout << "[stage] " << name << " done_ms=" << elapsed_ms << "\n"
            << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
  stimer::TimerConfig config;
  stimer::PathRequest path_request;
  bool use_path_request = false;
  std::string dump_file;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
      }
      if (arg == "--liberty") {
        config.liberty_files.push_back(require_value(argc, argv, &i));
      } else if (arg == "--def") {
        config.def_file = require_value(argc, argv, &i);
      } else if (arg == "--verilog") {
        config.verilog_file = require_value(argc, argv, &i);
      } else if (arg == "--spef") {
        config.spef_file = require_value(argc, argv, &i);
      } else if (arg == "--sdc") {
        config.sdc_file = require_value(argc, argv, &i);
      } else if (arg == "--platform" || arg == "--xplace-platform") {
        path_request.platform_path = require_value(argc, argv, &i);
        use_path_request = true;
      } else if (arg == "--design-dir") {
        path_request.design_dir = require_value(argc, argv, &i);
        use_path_request = true;
      } else if (arg == "--design-root" || arg == "--xplace-design-path") {
        path_request.design_root = require_value(argc, argv, &i);
        use_path_request = true;
      } else if (arg == "--design" || arg == "--xplace-design") {
        path_request.design_name = require_value(argc, argv, &i);
        use_path_request = true;
      } else if (arg == "--include-ram-lib") {
        path_request.include_ram_libs = true;
        use_path_request = true;
      } else if (arg == "--force-verilog") {
        config.read_verilog_after_def = true;
      } else if (arg == "--keep-coupling-caps") {
        config.keep_coupling_caps = true;
      } else if (arg == "--coupling-reduction-factor") {
        config.coupling_reduction_factor =
            std::stod(require_value(argc, argv, &i));
      } else if (arg == "--mode") {
        config.mode = parse_mode(require_value(argc, argv, &i));
      } else if (arg == "--report") {
        config.report_file = require_value(argc, argv, &i);
      } else if (arg == "--dump-net") {
        config.dump_net = require_value(argc, argv, &i);
      } else if (arg == "--dump-file") {
        dump_file = require_value(argc, argv, &i);
      } else if (arg == "--cpu") {
        config.use_cuda = false;
      } else if (arg == "--dump-debug") {
        config.dump_debug = true;
      } else {
        throw std::runtime_error("unknown argument: " + arg);
      }
    }

    if (argc == 1) {
      print_usage(argv[0]);
      return EXIT_SUCCESS;
    }

    if (use_path_request) {
      const stimer::PathResult paths = stimer::resolve_xplace_paths(
          path_request);
      stimer::apply_path_result(paths, &config);
      std::cout << "Resolved Xplace-style paths\n";
      std::cout << stimer::format_path_result(paths) << '\n' << std::flush;
    }

    stimer::Timer timer(config);
    std::cout << "Standalone timer input summary\n";
    std::cout << timer.input_summary() << std::flush;

    run_stage("read_inputs", [&] { timer.read_inputs(); });
    std::cout << "\nInput file probe\n";
    std::cout << timer.input_probe_summary();
    std::cout << "\nLiberty summary\n";
    std::cout << timer.liberty_summary() << std::flush;

    run_stage("build_design", [&] { timer.build_design(); });
    std::cout << "\nDesign summary\n";
    std::cout << timer.design_summary() << std::flush;
    run_stage("build_timing_graph", [&] { timer.build_timing_graph(); });
    std::cout << "\nTiming graph summary\n";
    std::cout << timer.timing_graph_summary() << std::flush;
    run_stage("build_rc_graph", [&] { timer.build_rc_graph(); });
    std::cout << "\nRC summary\n";
    std::cout << timer.rc_summary() << std::flush;

    run_stage("update_timing", [&] { timer.update_timing(); });
    std::cout << "\nTiming flat summary\n";
    std::cout << timer.timing_flat_summary();
    std::cout << "\nCPU timing summary\n";
    std::cout << timer.cpu_timing_summary();
    if (config.use_cuda) {
      std::cout << "\nCUDA timing summary\n";
      std::cout << timer.cuda_timing_summary();
    }

    std::cout << "\nStandalone timer result summary\n";
    std::cout << timer.result_summary() << std::flush;

    timer.write_report(config.report_file);
    if (!config.dump_net.empty() && !dump_file.empty()) {
      timer.dump_net_debug(config.dump_net, dump_file);
    }
  } catch (const std::exception& error) {
    std::cerr << "stimer_run error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
