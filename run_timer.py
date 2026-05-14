import time
import os
import sys
import argparse
import datetime
import torch
from timer_only.logger import setup_logger
from timer_only.tools import set_random_seed, str2bool
from timer_only.flute import Flute
from timer_only.timing_opt import GPUTimer
from timer_only.read_platform import load_design

def getArgs():
    parser = argparse.ArgumentParser('sizer')
    # general setting
    parser.add_argument('--platformPath', type=str, default='/research/d7/ascstd/qkduan25/Xplace/sky130hd', help='folder of platform files including lef and lib')
    parser.add_argument('--designName', type=str, default='blabla', help='design name')
    parser.add_argument('--designPath', type=str, default='./netlists', help='parent folder of designs')
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--outputPath', type=str, default='./output', help='output path')
    parser.add_argument('--load_from_raw', type=str2bool, default=True, help='If True, parse and load from benchmark files. If False, load from pt')
    parser.add_argument('--num_threads', type=int, default=20, help='threads')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')


     # logging and saver
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--verbose_cpp_log', type=str2bool, default=False, help='verbose cpp log for debugging')
    parser.add_argument('--cpp_log_level', type=int, default=2, help='0: DEBUG, 1: VERBOSE, 2:INFO')
    parser.add_argument('--result_dir', type=str, default='result', help='log/model root directory')
    parser.add_argument('--exp_id', type=str, default='', help='experiment id')
    parser.add_argument('--log_dir', type=str, default='log', help='log directory')
    parser.add_argument('--log_name', type=str, default='test.log', help='log file name')
    parser.add_argument('--eval_dir', type=str, default='eval', help='visualization directory')

    # global placement params
    parser.add_argument('--global_placement', type=str2bool, default=True, help='perform gp')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--inner_iter', type=int, default=10000, help='#inner iters')
    parser.add_argument('--wa_coeff', type=float, default=4.0, help='wa coeff')
    parser.add_argument('--num_bin_x', type=int, default=512, help='#binX for density function')
    parser.add_argument('--num_bin_y', type=int, default=512, help='#binY for density function')
    parser.add_argument('--density_weight', type=float, default=8e-5, help='the weight of density loss')
    parser.add_argument('--density_weight_coef', type=float, default=1.05, help='the ratio of density_weight')
    parser.add_argument('--target_density', type=float, default=1.0, help='placement target density')
    parser.add_argument('--use_filler', type=str2bool, default=True, help='placement filler')
    parser.add_argument('--noise_ratio', type=float, default=0.025, help='noise ratio for initialization')
    parser.add_argument('--ignore_net_degree', type=int, default=100, help='threshold of net degree to ignore in wirelength calculation')
    parser.add_argument('--use_eplace_nesterov', type=str2bool, default=True, help='enable eplace nesterov optimizer')
    parser.add_argument('--clamp_node', type=str2bool, default=True, help='enable eplace node clamp trick')
    parser.add_argument('--use_precond', type=str2bool, default=True, help='apply precond')
    parser.add_argument('--stop_overflow', type=float, default=0.07, help='stop overflow in scheduler')
    parser.add_argument('--enable_skip_update', type=str2bool, default=True, help='enable skip update')
    parser.add_argument('--enable_sample_force', type=str2bool, default=True, help='enable sample force')
    parser.add_argument("--mixed_size", type=str2bool, default=False, help="enable mixed size placement")

    # global routing params
    parser.add_argument('--use_cell_inflate', type=str2bool, default=False, help='use cell inflation')
    parser.add_argument('--min_area_inc', type=float, default=0.01, help='threshold of terminating inflation')
    parser.add_argument('--use_route_force', type=str2bool, default=False, help='use routing force')
    parser.add_argument('--route_freq', type=int, default=1000, help='routing freq')
    parser.add_argument('--num_route_iter', type=int, default=400, help='number of routing iters')
    parser.add_argument('--route_weight', type=float, default=0, help='the weight of route')
    parser.add_argument('--congest_weight', type=float, default=0, help='the weight of congested force')
    parser.add_argument('--pseudo_weight', type=float, default=0, help='the weight of pseudo net')
    parser.add_argument('--visualize_cgmap', type=str2bool, default=False, help='visualize congestion map')

    # timing opt params
    parser.add_argument('--timing_opt', type=str2bool, default=False, help='perform timing optimization')
    parser.add_argument('--timing_freq', type=int, default=1, help='timing freq')
    parser.add_argument('--calibration', type=str2bool, default=True, help='perform timer calibration')
    parser.add_argument('--calibration_step', type=float, default=0.1, help='timing calibration step')
    parser.add_argument('--timing_start_iter', type=int, default=100, help='start iteration of timing optimization')
    parser.add_argument('--timing_init_weight', type=float, default=0.05, help='initial timing wirelength weight')
    parser.add_argument('--decay_factor', type=float, default=0.3, help='decay factor of timing weight')
    parser.add_argument('--decay_boost', type=float, default=3, help='dynamic decay boost factor')
    parser.add_argument('--gr_rc', type=str, default='', help='OpenROAD my_dump_gr_rc output for DMP RC')
    parser.add_argument('--route_segments', type=str, default='', help='OpenROAD write_global_route_segments output for direct DMP RC')
    parser.add_argument('--debug_dump_rc_net', type=str, default='', help='dump one net RC after loading --gr_rc and/or --route_segments, then exit')
    parser.add_argument('--debug_pin_timing', action='append', default=[], help='print AT/RAT/slack/slew/load for one pin after timing')
    parser.add_argument('--debug_report_path_pin', action='append', default=[], help='print report_path for one endpoint pin after timing')
    parser.add_argument('--debug_dump_endpoint_tests', type=str, default='', help='write per-test timing details for --debug_endpoint_test_pin endpoint pins')
    parser.add_argument('--debug_endpoint_test_pin', action='append', default=[], help='endpoint pin to include in --debug_dump_endpoint_tests')
    parser.add_argument('--debug_dump_endpoint_slacks', type=str, default='', help='write unique endpoint pin slack CSV after timing')
    parser.add_argument('--debug_report_k_path', type=int, default=0, help='debug only: print top-K late-fall timing paths after timing')
    parser.add_argument('--timer_verbose', type=str2bool, default=False, help='print verbose timer progress messages')
    parser.add_argument('--fast_exit_after_timing', type=str2bool, default=False, help='direct timing only: opt-in fast process exit after reporting results')
    parser.add_argument('--wire_resistance_per_micron', type=float, default=2.4222e-02 * 1e3, help='unit wire resistance ohm/um, normalized across all layers. setup.sh kohm/um (follows last lib read by openroad)')
    parser.add_argument('--wire_capacitance_per_micron', type=float, default=1.2918e-01 * 1e-15, help='unit wire capacitance F/um, normalized across all layers. setup.sh fF/um (follows last lib read by openroad)')
    # parser.add_argument('--wire_resistance_per_micron', type=float, default=5.235, help='unit wire resistance, normalized across all layers')
    # parser.add_argument('--wire_capacitance_per_micron', type=float, default=0.131e-15, help='unit wire capacitance, normalized across all layers')

    # detailed placement and evaluation
    parser.add_argument('--legalization', type=str2bool, default=True, help='perform lg')
    parser.add_argument('--detail_placement', type=str2bool, default=True, help='perform dp')
    parser.add_argument('--dp_engine', type=str, default="default", help='choose dp engine')
    parser.add_argument('--eval_by_external', type=str2bool, default=False, help='eval dp sol by external binary')
    parser.add_argument('--eval_engine', type=str, default="ntuplace4dr", help='choose eval engine')
    parser.add_argument('--final_route_eval', type=str2bool, default=False, help='eval placement solution by GR')

    # placement output
    parser.add_argument('--draw_placement', type=str2bool, default=False, help='draw placement')
    parser.add_argument('--write_placement', type=str2bool, default=True, help='write placement result')
    parser.add_argument('--write_global_placement', type=str2bool, default=False, help='write global placement result')
    parser.add_argument('--output_dir', type=str, default="output", help='output directory')
    parser.add_argument('--output_prefix', type=str, default="placement", help='prefix of placement output file')
    parser.add_argument('--deterministic', type=str2bool, default=True, help='use deterministic mode')

    # todo: add alpha, beta, gamma
    # parser.add_argument('--customPath', type=str, default='', help='custom design path, set it as token1:path1,token2:path2 e.g. lef:data/test.lef,def:data/test.def,design_name:mydesign,benchmark:mybenchmark')
    # parser.add_argument('--customJson', type=str, default='', help='custom json path,
    args = parser.parse_args()


    args.exp_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + args.exp_id
    args.exp_id = "{}_{}".format(args.exp_id, args.designName)
    return args


CORNERS = ['early-rise', 'early-fall', 'late-rise', 'late-fall']

def _r2_report(at_true_ns, at_pred_ns, label, logger):
    """Compute and log R² + MAE per corner and overall. at_* are [num_pins, 4] tensors in ns."""
    valid = torch.isfinite(at_true_ns) & torch.isfinite(at_pred_ns)
    r2_list = []
    logger.info(f"\n========== {label} ==========")
    for c, name in enumerate(CORNERS):
        m = valid[:, c]
        if m.sum() < 2:
            logger.info(f"  {name:12s}: R²=  NaN (insufficient valid pins: {m.sum().item()})")
            r2_list.append(float('nan'))
            continue
        y, yhat = at_true_ns[m, c], at_pred_ns[m, c]
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2  = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
        mae = (y - yhat).abs().mean().item()
        r2_list.append(r2)
        logger.info(f"  {name:12s}: R²={r2:7.4f}, MAE={mae:.4e} ns, Valid={m.sum().item()}")
    y_all, yhat_all = at_true_ns[valid], at_pred_ns[valid]
    ss_res = ((y_all - yhat_all) ** 2).sum()
    ss_tot = ((y_all - y_all.mean()) ** 2).sum()
    r2_all = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
    logger.info(f"  {'Overall':12s}: R²={r2_all:7.4f}, Valid={valid.sum().item()}")
    logger.info("=" * (len(label) + 22) + "\n")
    return r2_list, r2_all


def compare_spef_vs_ml(gputimer, infer_file, logger):
    """Compare SPEF-based AT (ground truth) vs ML inference AT. Returns (r2_per_corner, r2_overall)."""
    t = gputimer.timer.time_unit() * 1e9
    gputimer.update_timing_spef()
    wns_early, tns_early, wns_late, tns_late = gputimer.report_timing_slack()
    logger.info("SPEF Evaluation wns_early: %.3f, tns_early: %.3f, wns_late: %.3f, tns_late: %.3f" % (wns_early, tns_early, wns_late, tns_late))
    at_spef = gputimer.timer.report_pin_at() * t
    gputimer.update_timing_infer(infer_file)
    at_ml   = gputimer.timer.report_pin_at() * t
    return _r2_report(at_spef, at_ml, "SPEF vs ML Inference AT", logger)


def compare_opr_vs_ml(gputimer, infer_file, logger):
    """Compare OpenROAD GT AT vs ML inference AT. Returns (r2_per_corner, r2_overall)."""
    gputimer.update_timing_opr_infer(infer_file)
    wns_early, tns_early, wns_late, tns_late = gputimer.report_timing_slack()
    logger.info("OPR Evaluation wns_early: %.3f, tns_early: %.3f, wns_late: %.3f, tns_late: %.3f" % (wns_early, tns_early, wns_late, tns_late))
    gputimer.timer.report_K_path(10, 1, 1, True)
    # gputimer.report_path("_81438_:D", 1, 0, True)
    # gputimer.report_path("qnt_cnt[3]", 1, 1, True)
    t = gputimer.timer.time_unit() * 1e9
    at_ml = gputimer.timer.report_pin_at()    * t
    at_gt = gputimer.timer.report_pin_gt_at().to(at_ml.device) * t
    # idx = gputimer.pin_names.index("i43/i163:QN")
    # print(f"at:{at_ml[idx]} gt
    # :{at_gt[idx]}")
    # Top-20 pins per corner by squared error
    valid = torch.isfinite(at_gt) & torch.isfinite(at_ml)
    se = torch.where(valid, (at_gt - at_ml) ** 2, torch.zeros_like(at_gt))
    for c, cname in enumerate(CORNERS):
        corner_valid_count = valid[:, c].sum().item()
        topk = min(20, corner_valid_count)
        top_vals, top_idx = torch.topk(se[:, c], topk)
        logger.info(f"\n===== Top-{topk} pins by squared error [{cname}] (OPR GT vs ML) =====")
        logger.info(f"  {'Rank':>4}  {'Pin Name':<50}  {'SE (ns²)':>14}  {'GT (ns)':>12}  {'ML (ns)':>12}")
        for rank, (idx, sq) in enumerate(zip(top_idx.tolist(), top_vals.tolist()), 1):
            name = gputimer.pin_names[idx] if idx < len(gputimer.pin_names) else f"pin_{idx}"
            gt_val = at_gt[idx, c].item()
            ml_val = at_ml[idx, c].item()
            logger.info(f"  {rank:4d}  {name:<50}  {sq:14.4e}  {gt_val:12.4e}  {ml_val:12.4e}")
        logger.info("=" * 60)

    return _r2_report(at_gt, at_ml, "OPR GT vs ML Inference AT", logger)


def main():
    Flute.register(8)
    args = getArgs()
    if args.route_segments or args.gr_rc:
        os.environ.setdefault("GPUTIMER_DISABLE_REF_TIMING_TENSORS", "1")
        os.environ.setdefault("GPUTIMER_DISABLE_STATE_BACKUP_TENSORS", "1")
        os.environ.setdefault("GPUTIMER_EMPTY_CACHE_AFTER_GTDB", "1")
        os.environ.setdefault("DMP_DEFER_TIMING_ALLOC", "1")
    logger = setup_logger(args, sys.argv)

    set_random_seed(args)
    data, rawdb, gpdb, params = load_design(args, logger)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    if args.route_segments or args.gr_rc:
        data = data.to_timing_device(device)
        data = data.preprocess_timing()
    else:
        data = data.to(device)
        data = data.preprocess()
    if args.route_segments:
        params["route_segments"] = args.route_segments
    if args.gr_rc:
        params["gr_rc"] = args.gr_rc
    gputimer = GPUTimer(data, rawdb, gpdb, params, args)
    if args.route_segments or args.gr_rc:
        data = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # gputimer.timer.set_ideal_clock(True)
    if args.debug_dump_rc_net:
        if args.gr_rc:
            gputimer.debug_dump_openroad_gr_rc_net(args.gr_rc, args.debug_dump_rc_net)
        if args.route_segments:
            gputimer.debug_dump_openroad_route_segments_rc_net(args.route_segments, args.debug_dump_rc_net)
        return
    if args.route_segments:
        if not os.path.exists(args.route_segments):
            raise FileNotFoundError(f"OpenROAD route segment file not found: {args.route_segments}")
        if args.gr_rc:
            if not os.path.exists(args.gr_rc):
                raise FileNotFoundError(f"OpenROAD GR RC dump not found: {args.gr_rc}")
            logger.info(f"Comparing raw route-segment RC against OpenROAD GR RC TSV: {args.gr_rc}")
            gputimer.debug_compare_openroad_route_segments_rc(args.gr_rc, args.route_segments)
        logger.info(f"Running DMP timing with OpenROAD route segments: {args.route_segments}")
        gputimer.update_timing_dmp_route_segments(args.route_segments)
        label = "DMP route-segment RC evaluation"
    elif args.gr_rc:
        if not os.path.exists(args.gr_rc):
            raise FileNotFoundError(f"OpenROAD GR RC dump not found: {args.gr_rc}")
        logger.info(f"Running DMP timing with OpenROAD GR RC: {args.gr_rc}")
        gputimer.update_timing_dmp_gr(args.gr_rc)
        label = "DMP GR RC evaluation"
    elif "spef" not in params or not os.path.exists(params["spef"]):
        raise FileNotFoundError(f"SPEF file not found: {params.get('spef')}")
    else:
        logger.info(f"Running DMP timing with SPEF RC: {params['spef']}")
        gputimer.update_timing_dmp_spef()
        label = "DMP SPEF evaluation"
    wns_early, tns_early, wns_late, tns_late = gputimer.report_timing_slack()
    logger.info("%s: wns_early: %.3f, tns_early: %.3f, wns_late: %.3f, tns_late: %.3f" % (label, wns_early, tns_early, wns_late, tns_late))
    time_to_ns = gputimer.timer.time_unit() * 1e9
    for pin_name in args.debug_pin_timing:
        try:
            idx = gputimer.pin_names.index(pin_name)
            at = (gputimer.timer.report_pin_at()[idx] * time_to_ns).detach().cpu().tolist()
            rat = (gputimer.timer.report_pin_rat()[idx] * time_to_ns).detach().cpu().tolist()
            slew = (gputimer.timer.report_pin_slew()[idx] * time_to_ns).detach().cpu().tolist()
            load = gputimer.timer.report_pin_load()[idx].detach().cpu().tolist()
            slack = [r - a for a, r in zip(at, rat)]
            logger.info(
                f"DEBUG_PIN_TIMING {pin_name} idx={idx} AT(ns)={at} "
                f"RAT(ns)={rat} slack(ns)={slack} slew(ns)={slew} load={load}"
            )
        except ValueError:
            logger.info(f"DEBUG_PIN_TIMING {pin_name} not found")
    for pin_name in args.debug_report_path_pin:
        try:
            logger.info(f"DEBUG_REPORT_PATH {pin_name} late-fall")
            gputimer.report_path(pin_name, 1, 1, True)
        except ValueError:
            logger.info(f"DEBUG_REPORT_PATH {pin_name} not found")
    if args.debug_dump_endpoint_tests:
        gputimer.timer.debug_dump_endpoint_tests(args.debug_dump_endpoint_tests, args.debug_endpoint_test_pin)
    if args.debug_dump_endpoint_slacks:
        import csv
        endpoint_pin_slack = None
        if hasattr(gputimer.timer, "report_endpoint_pin_slack"):
            endpoint_pin_slack = (gputimer.timer.report_endpoint_pin_slack() * time_to_ns).detach().cpu()
        pin_slack = (gputimer.timer.report_pin_slack() * time_to_ns).detach().cpu()
        pin_at = (gputimer.timer.report_pin_at() * time_to_ns).detach().cpu()
        pin_rat = (gputimer.timer.report_pin_rat() * time_to_ns).detach().cpu()
        rows = []
        if endpoint_pin_slack is not None and endpoint_pin_slack.numel() > 0:
            endpoint_ids = []
            for pin_id in range(endpoint_pin_slack.shape[0]):
                slacks = endpoint_pin_slack[pin_id]
                valid = torch.isfinite(slacks) & (slacks < 1.0e20)
                if torch.any(valid).item():
                    endpoint_ids.append(pin_id)
        else:
            endpoint_ids = torch.unique(gputimer.timer.endpoints_index()).detach().cpu().tolist()
        for pin_id in endpoint_ids:
            slacks = (endpoint_pin_slack[pin_id] if endpoint_pin_slack is not None and endpoint_pin_slack.numel() > 0
                      else pin_slack[pin_id]).tolist()
            pin_slacks = pin_slack[pin_id].tolist()
            late_slack = min(slacks[2], slacks[3])
            rows.append((late_slack, pin_id, slacks, pin_slacks, pin_at[pin_id].tolist(), pin_rat[pin_id].tolist()))
        rows.sort(key=lambda x: x[0])
        with open(args.debug_dump_endpoint_slacks, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pin_id", "pin_name", "endpoint_slack_er", "endpoint_slack_ef",
                             "endpoint_slack_lr", "endpoint_slack_lf",
                             "pin_slack_er", "pin_slack_ef", "pin_slack_lr", "pin_slack_lf",
                             "at_er", "at_ef", "at_lr", "at_lf",
                             "rat_er", "rat_ef", "rat_lr", "rat_lf"])
            for _, pin_id, slacks, pin_slacks, at, rat in rows:
                writer.writerow([pin_id, gputimer.pin_names[pin_id], *slacks, *pin_slacks, *at, *rat])
        logger.info(f"DEBUG_DUMP_ENDPOINT_SLACKS wrote {len(rows)} rows to {args.debug_dump_endpoint_slacks}")
    if args.debug_report_k_path > 0:
        gputimer.timer.report_K_path(args.debug_report_k_path, 1, 1, True)
    if (args.route_segments or args.gr_rc) and args.fast_exit_after_timing:
        for handler in logger.handlers:
            handler.flush()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    # spef_infer = f"./TimingPredict/infer_results/02_gpu_order/{args.designName}.infer"
    # opr_infer  = f"./synthetic_data/infer_results/05_netconv_sage_16400/{args.designName}.infer"
    # opr_infer  = f"./synthetic_data/infer_results/00_gcn_sky130/{args.designName}.infer"
    # opr_infer  = f"./synthetic_data/infer_results/03_scell_netconv_27400/{args.designName}.infer"
    # compare_spef_vs_ml(gputimer, spef_infer, logger)
    # compare_opr_vs_ml(gputimer, opr_infer, logger)

if __name__ == "__main__":
    main()
