#include "common/common.h"

#include "common/StageProfiler.h"
#include "common/db/Database.h"
#include "io_parser/gp/GPDatabase.h"
#include "gputimer/db/GTDatabase.h"
#include "gputimer/core/GPUTimer.h"

#include <pybind11/stl.h>

#include <flute.hpp>
using namespace Flute;

namespace Xplace {

static bool pybind_timer_profile_enabled()
{
    return xplace_env_enabled("XPLACE_TIMER_PROFILE");
}

std::shared_ptr<gt::GPUTimer> create_gputimer(const py::dict& kwargs,
                                              std::shared_ptr<db::Database> rawdb,
                                              std::shared_ptr<gp::GPDatabase> gpdb,
                                              std::shared_ptr<gt::TimingTorchRawDB> timing_raw_db) {

    StageProfiler profile("XPLACE_TIMER_PROFILE_CPP", pybind_timer_profile_enabled(), stdout);

    if (!rawdb->liberty_read) {
        throw std::invalid_argument("Liberty file not found. Please check!");
    }
    if (kwargs.contains("num_threads")) {
        const int requested_threads = kwargs["num_threads"].cast<int>();
        if (requested_threads > 0) {
            timing_raw_db->num_threads = requested_threads;
        }
    }
    std::shared_ptr<gt::GTDatabase> gtdb = std::make_shared<gt::GTDatabase>(rawdb, gpdb, timing_raw_db);
    const bool direct_rc_mode = kwargs.contains("route_segments") || kwargs.contains("gr_rc");
    gtdb->skip_legacy_rc_tensors = direct_rc_mode;
    profile.mark("construct_gtdb");
    auto sdc = std::make_shared<gt::sdc::SDC>();

    try {
        if (kwargs.contains("sdc")) sdc->read(kwargs["sdc"].cast<std::string>());
    } catch (std::exception& e) {
        logger.error("%s\n", e.what());
    }
    profile.mark("read_sdc_json");

    gtdb->preparePinNameMapForSdc(*sdc);
    profile.mark("prepare_pin_name_map_targets");
    gtdb->ExtractTimingGraph();
    profile.mark("extract_timing_graph");
    gtdb->readSdc(*sdc);
    profile.mark("read_sdc_into_gtdb");
    gtdb->RunSdcConstantSimulation();
    profile.mark("sdc_constant_simulation");

    std::shared_ptr<gt::GPUTimer> gputimer = std::make_shared<gt::GPUTimer>(gtdb, timing_raw_db);
    profile.mark("construct_gputimer");

    if (!direct_rc_mode) {
        readLUT("thirdparty/flute_mp/lut.ICCAD2015/POWV9.dat", "thirdparty/flute_mp/lut.ICCAD2015/POST9.dat");
        profile.mark("read_flute_lut");
    }

    return gputimer;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<gt::GPUTimer, std::shared_ptr<gt::GPUTimer>>(m, "GPUTimer")
        .def(pybind11::init<std::shared_ptr<gt::GTDatabase>, std::shared_ptr<gt::TimingTorchRawDB>>())
        .def("time_unit", &gt::GPUTimer::time_unit)
        .def("read_spef", &gt::GPUTimer::read_spef)
        .def("init", &gt::GPUTimer::initialize)
        .def("levelize", &gt::GPUTimer::levelize)
        .def("update_rc", &gt::GPUTimer::update_rc_timing)
        .def("update_rc_flute", &gt::GPUTimer::update_rc_timing_flute)
        .def("update_rc_spef", &gt::GPUTimer::update_rc_timing_spef)
        .def("debug_dump_spef_rc_net", &gt::GPUTimer::debug_dump_spef_rc_net)
        .def("debug_dump_openroad_gr_rc_net",
             &gt::GPUTimer::debug_dump_openroad_gr_rc_net,
             py::arg("file"),
             py::arg("net_name"))
        .def("debug_dump_openroad_route_segments_rc_net",
             &gt::GPUTimer::debug_dump_openroad_route_segments_rc_net,
             py::arg("file"),
             py::arg("net_name"))
        .def("debug_compare_openroad_route_segments_rc",
             &gt::GPUTimer::debug_compare_openroad_route_segments_rc,
             py::arg("gr_rc_file"),
             py::arg("route_segments_file"),
             py::arg("top_n") = 20)
        .def("update_states", &gt::GPUTimer::update_states)
        .def("update_timing", &gt::GPUTimer::update_timing)
        .def("update_endpoints", &gt::GPUTimer::update_endpoints)
        .def("report_wns", &gt::GPUTimer::report_wns)
        .def("report_tns_elw", &gt::GPUTimer::report_tns_elw)
        .def("report_wns_and_tns", &gt::GPUTimer::report_wns_and_tns)
        .def("report_pin_slack", &gt::GPUTimer::report_pin_slack, py::return_value_policy::move)
        .def("report_pin_at", &gt::GPUTimer::report_pin_at, py::return_value_policy::move)
        .def("report_pin_rat", &gt::GPUTimer::report_pin_rat, py::return_value_policy::move)
        .def("report_pin_slew", &gt::GPUTimer::report_pin_slew, py::return_value_policy::move)
        .def("report_pin_load", &gt::GPUTimer::report_pin_load, py::return_value_policy::move)
        .def("report_delay", &gt::GPUTimer::report_delay, py::return_value_policy::move)
        .def("report_power_liberty_inventory", &gt::GPUTimer::report_power_liberty_inventory)
        .def("report_power_activity_cpu", &gt::GPUTimer::report_power_activity_cpu, py::return_value_policy::move)
        .def("report_power_activity_cuda", &gt::GPUTimer::report_power_activity_cuda, py::return_value_policy::move)
        .def("report_power_switching_cuda", &gt::GPUTimer::report_power_switching_cuda, py::return_value_policy::move)
        .def("report_power_internal_cuda", &gt::GPUTimer::report_power_internal_cuda, py::return_value_policy::move)
        .def("report_power_internal_arcs_cuda", &gt::GPUTimer::report_power_internal_arcs_cuda, py::return_value_policy::move)
        .def("report_power_leakage_cuda", &gt::GPUTimer::report_power_leakage_cuda, py::return_value_policy::move)
        .def("report_power_leakage_rows_cuda", &gt::GPUTimer::report_power_leakage_rows_cuda, py::return_value_policy::move)
        .def("report_power_total_cuda", &gt::GPUTimer::report_power_total_cuda, py::return_value_policy::move)
        .def("report_power_seq_inventory", &gt::GPUTimer::report_power_seq_inventory)
        .def("report_power_group_codes", &gt::GPUTimer::report_power_group_codes, py::return_value_policy::move)
        .def("report_power_internal_lut_cuda_probe", &gt::GPUTimer::report_power_internal_lut_cuda_probe, py::return_value_policy::move)
        .def("debug_dump_endpoint_tests",
             &gt::GPUTimer::debug_dump_endpoint_tests,
             py::arg("outfile"),
             py::arg("endpoint_pin_names"))
        .def("report_endpoint_slack", &gt::GPUTimer::report_endpoint_slack, py::return_value_policy::move)
        .def("report_endpoint_pin_slack", &gt::GPUTimer::report_endpoint_pin_slack, py::return_value_policy::move)
        .def("endpoints_index", &gt::GPUTimer::endpoints_index, py::return_value_policy::copy)
        .def("report_path", &gt::GPUTimer::report_path, py::return_value_policy::copy)
        .def("report_K_path", &gt::GPUTimer::report_K_path, py::return_value_policy::copy)
        .def("report_criticality", &gt::GPUTimer::report_criticality, py::return_value_policy::copy)
        .def("report_criticality_threshold", &gt::GPUTimer::report_criticality_threshold, py::return_value_policy::copy)
        .def("dump_timing_graph", &gt::GPUTimer::dump_timing_graph,
             py::arg("outfile"),
             "Dump timing graph to JSONL file. Each line: node / net_arc / cell_arc record.\n"
             "Pin IDs correspond 1:1 to internal pin_names.\n"
             "Call after update_timing() for valid AT/RAT/slew labels.")
        .def("read_infer", &gt::GPUTimer::read_infer,
             py::arg("infile"),
             "Load TimingPredict inference results from CSV and update GPU timing arrays.\n"
             "Updates arcDelay with ML-predicted net delays and cell edge delays.\n"
             "Input file format: .infer CSV with node/edge predictions in nanoseconds.")
        .def("propagate_infer_timing", &gt::GPUTimer::propagate_infer_timing,
             "Propagate ML-predicted delays through timing graph.\n"
             "Performs forward pass (AT), backward pass (RAT), and slack computation.\n"
             "Must call read_infer() first to load arcDelay with ML predictions.")
        .def("read_opr_gt_infer", &gt::GPUTimer::read_opr_gt_infer,
             py::arg("infile"),
             "Load OpenROAD ground-truth .infer CSV. Maps OPR node IDs to GPUTimer pin IDs\n"
             "via pin_name. Stores GT AT values in host_pinGT_AT (accessible via report_pin_gt_at()).\n"
             "Also loads slew/net_delay/cell_delay into GPU arrays for propagation.")
        .def("report_pin_gt_at", &gt::GPUTimer::report_pin_gt_at, py::return_value_policy::move,
             "Return GT AT values loaded by read_opr_gt_infer(). Shape [num_pins, 4], internal units, NaN=missing.")
        .def("update_rc_flute_dmp", &gt::GPUTimer::update_rc_timing_flute_dmp)
        .def("init_dmp_rc_spef", &gt::GPUTimer::init_dmp_rc_spef)
        .def("init_dmp_rc_gr", &gt::GPUTimer::init_dmp_rc_gr,
             py::arg("file"))
        .def("init_dmp_rc_route_segments", &gt::GPUTimer::init_dmp_rc_route_segments,
             py::arg("file"))
        .def("debug_dmp_route_segment_fd_grad",
             &gt::GPUTimer::debug_dmp_route_segment_fd_grad,
             py::arg("route_segments_file"),
             py::arg("kind") = "edge_res",
             py::arg("ids") = std::vector<int64_t>{},
             py::arg("sample_count") = 0,
             py::arg("seed") = 1,
             py::arg("eps_rel") = 1.0e-3,
             py::arg("eps_abs") = 0.0,
             py::arg("tau_ns") = 0.02)
        .def("compute_dmp_route_segment_soft_timing_grad",
             &gt::GPUTimer::compute_dmp_route_segment_soft_timing_grad,
             py::arg("route_segments_file"),
             py::arg("tau_ns") = 0.02)
        .def("debug_dmp_route_segment_grad_fd_validate",
             &gt::GPUTimer::debug_dmp_route_segment_grad_fd_validate,
             py::arg("route_segments_file"),
             py::arg("sample_net_count") = 10000,
             py::arg("seed") = 1,
             py::arg("eps_rel") = 1.0e-3,
             py::arg("eps_abs_edge") = 0.0,
             py::arg("eps_abs_node") = 1.0e-4,
             py::arg("tau_ns") = 0.02)
        .def("debug_dmp_route_segment_primitive_slope_stats",
             &gt::GPUTimer::debug_dmp_route_segment_primitive_slope_stats,
             py::arg("route_segments_file"))
        .def("debug_dmp_route_segment_rc_tree_gradcheck",
             &gt::GPUTimer::debug_dmp_route_segment_rc_tree_gradcheck,
             py::arg("route_segments_file"),
             py::arg("sample_net_count") = 10000,
             py::arg("seed") = 1,
             py::arg("eps_rel") = 1.0e-4,
             py::arg("eps_abs_edge") = 1.0e-6,
             py::arg("eps_abs_node") = 1.0e-8)
        .def("debug_dump_dmp_rc_net", &gt::GPUTimer::debug_dump_dmp_rc_net)
        .def("initialize_dmp_model", &gt::GPUTimer::initialize_dmp_model)
        .def("release_dmp_timing_scratch_for_power", &gt::GPUTimer::release_dmp_timing_scratch_for_power)
        .def("update_timing_dmp", &gt::GPUTimer::update_timing_dmp)
        .def("print_pin_id_name", &gt::GPUTimer::print_pin_id_name)
        .def("get_units", &gt::GPUTimer::get_units)
        .def("print_pinLoad", &gt::GPUTimer::print_pinLoad)
        .def("set_dmp_debug_flag", &gt::GPUTimer::set_dmp_debug_flag)
        ;
    pybind11::class_<gt::TimingTorchRawDB, std::shared_ptr<gt::TimingTorchRawDB>>(m, "TimingTorchRawDB")
        .def(pybind11::init<torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            torch::Tensor,
                            int,
                            float,
                            int,
                            float,
                            float>())
        .def("commit_from", &gt::TimingTorchRawDB::commit_from)
        .def("get_curr_cposx", &gt::TimingTorchRawDB::get_curr_cposx, py::return_value_policy::move)
        .def("get_curr_cposy", &gt::TimingTorchRawDB::get_curr_cposy, py::return_value_policy::move)
        .def("get_curr_lposx", &gt::TimingTorchRawDB::get_curr_lposx, py::return_value_policy::move)
        .def("get_curr_lposy", &gt::TimingTorchRawDB::get_curr_lposy, py::return_value_policy::move);

    pybind11::class_<gt::GTDatabase, std::shared_ptr<gt::GTDatabase>>(m, "GTDatabase")
        .def(pybind11::init<std::shared_ptr<db::Database>, std::shared_ptr<gp::GPDatabase>, std::shared_ptr<gt::TimingTorchRawDB>>());

    m.def("create_gputimer", &create_gputimer, "Create gputimer object");
    m.def("create_timing_rawdb",
          [](torch::Tensor node_lpos_init_,
             torch::Tensor node_size_,
             torch::Tensor pin_rel_lpos_,
             torch::Tensor pin_id2node_id_,
             torch::Tensor pin_id2net_id_,
             torch::Tensor node2pin_list_,
             torch::Tensor node2pin_list_end_,
             torch::Tensor hyperedge_list_,
             torch::Tensor hyperedge_list_end_,
             torch::Tensor net_mask_,
             int num_movable_nodes_,
             float scale_factor_,
             int microns_,
             float wire_resistance_per_micron_,
             float wire_capacitance_per_micron_) {
              return std::make_shared<gt::TimingTorchRawDB>(node_lpos_init_,
                                                            node_size_,
                                                            pin_rel_lpos_,
                                                            pin_id2node_id_,
                                                            pin_id2net_id_,
                                                            node2pin_list_,
                                                            node2pin_list_end_,
                                                            hyperedge_list_,
                                                            hyperedge_list_end_,
                                                            net_mask_,
                                                            num_movable_nodes_,
                                                            scale_factor_,
                                                            microns_,
                                                            wire_resistance_per_micron_,
                                                            wire_capacitance_per_micron_);
          });
}

}  // namespace Xplace
