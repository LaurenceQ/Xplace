import time
import os
import torch
from utils import *
from src import Flute, GPUTimer
from src import load_design, ParamScheduler
from src.detail_placement import run_lg, detail_placement_main
from pdb import set_trace as bp

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
    logger = setup_logger(args, sys.argv)

    set_random_seed(args)
    data, rawdb, gpdb, params = load_design(args, logger)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    data = data.to(device)
    data = data.preprocess()
    if args.gr_rc:
        params["gr_rc"] = args.gr_rc
    gputimer = GPUTimer(data, rawdb, gpdb, params, args)
    # gputimer.timer.set_ideal_clock(True)
    if args.gr_rc:
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
    gputimer.timer.report_K_path(2, 1, 1, True)
    # spef_infer = f"./TimingPredict/infer_results/02_gpu_order/{args.designName}.infer"
    # opr_infer  = f"./synthetic_data/infer_results/05_netconv_sage_16400/{args.designName}.infer"
    # opr_infer  = f"./synthetic_data/infer_results/00_gcn_sky130/{args.designName}.infer"
    # opr_infer  = f"./synthetic_data/infer_results/03_scell_netconv_27400/{args.designName}.infer"
    # compare_spef_vs_ml(gputimer, spef_infer, logger)
    # compare_opr_vs_ml(gputimer, opr_infer, logger)

if __name__ == "__main__":
    main()
