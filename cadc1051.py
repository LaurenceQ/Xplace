import time
from utils import *
from src import Flute, GPUTimer
from src import load_design, ParamScheduler
from src.detail_placement import run_lg, detail_placement_main
from src.run_placement_nesterov import run_placement_main_nesterov_and_sizing
from pdb import set_trace as bp
def getArgs():
    parser = argparse.ArgumentParser('sizer')
    # general setting
    parser.add_argument('--platformPath', type=str, default='./platform', help='parent folder of ASAP7')
    parser.add_argument('--designName', type=str, default='aes_cipher_top', help='design name')
    parser.add_argument('--designPath', type=str, default='./design', help='parent folder of designs')
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--outputPath', type=str, default='', help='output path')
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
    # parser.add_argument('--wire_resistance_per_micron', type=float, default=2.4222e-05, help='unit wire resistance, normalized across all layers')
    # parser.add_argument('--wire_capacitance_per_micron', type=float, default=1.2918e-22, help='unit wire capacitance, normalized across all layers')
    parser.add_argument('--wire_resistance_per_micron', type=float, default=5.235, help='unit wire resistance, normalized across all layers')
    parser.add_argument('--wire_capacitance_per_micron', type=float, default=0.131e-15, help='unit wire capacitance, normalized across all layers')

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
def main():
    Flute.register(8)
    args = getArgs()
    logger = setup_logger(args, sys.argv)

    set_random_seed(args)
    # Flute.register(args.num_threads)
    data, rawdb, gpdb = load_design(args, logger)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    data = data.to(device)
    data = data.preprocess()
    # run_placement_main_nesterov_and_sizing(args, logger, data, rawdb, gpdb)
    # # use bangqi's db? or construct sizing db?
    params = {
        "design_name": f"{args.designName}",
        "sdc": f"design/{args.designName}/{args.designName}.sdc",
        "spef": f"design/{args.designName}/{args.designName}.spef",
    }
    # ps = ParamScheduler(data, args, logger)


    # node_pos = data.node_pos
    # node_pos, dp_hpwl, top5overflow, lg_time, dp_time = detail_placement_main(
    #     node_pos, gpdb, rawdb, ps, data, args, logger
    # )
    # gpdb.write_placement(f"./output/{args.designName}")
    # logger.info("Data preprocessing finished")
    gputimer = GPUTimer(data, rawdb, gpdb, params, args)
    
    # # timing analysis for extracted RC network

    
    gputimer.update_timing_eval(data.node_pos)
    wns_early, tns_early, wns_late, tns_late = gputimer.report_timing_slack()
    logger.info("FLUTE evaluation: wns_early: %.3f, tns_early: %.3f, wns_late: %.3f, tns_late: %.3f" % ( wns_early, tns_early, wns_late, tns_late))

    # gputimer.timer.init_sizing()
    # for T in range(1):
    #     gputimer.timer.evaluate_sizing(-1)
    #     gputimer.timer.change_db_sizing()
    #     data.node_pos = run_lg(data.node_pos, data, args, logger)
    #     gputimer.update_timing_eval(data.node_pos)
    #     wns_early, tns_early, wns_late, tns_late = gputimer.report_timing_slack()
    #     logger.info("FLUTE evaluation: wns_early: %.3f, tns_early: %.3f, wns_late: %.3f, tns_late: %.3f" % ( wns_early, tns_early, wns_late, tns_late))
    # # bp()
    
    
    # gputimer.timer.report_K_path(10, 1, True)    
    # # # gputimer.report_path(ep_name = "i_cache_subsystem_i_nbdcache_i_miss_handler_evict_cl_q_reg[data][102]:SETN", el = 1, verbose = True)
    # gpdb.write_placement(f"./output/{args.designName}")
    # bp()
               

if __name__ == "__main__":
    main()