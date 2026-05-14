import glob
import os
from .database import load_dataset


def load_design(args,logger):
    # params = find_design_params(args, logger)
    # todo: 改一个新的get_custom_json_params
    # params = get_custom_json_params(args, logger)
    arg_dict = vars(args)
    # with open(args.custom_json, 'r') as f:
    #     params = json.load(f)
    platformPath = arg_dict.get('platformPath', '')
    designPath = arg_dict.get('designPath', '')
    designName = arg_dict.get('designName', '')

    # 自动读取platformPath下的lef和lib文件夹中的所有文件
    lef_path = os.path.join(platformPath, "lef")
    lib_path = os.path.join(platformPath, "lib")

    lefs = []
    libs = []

    if os.path.isdir(lef_path):
        lefs = sorted(glob.glob(os.path.join(lef_path, "*lef")))
    else:
        logger.warning(f"LEF path {lef_path} does not exist. No LEF files will be loaded.")
    if os.path.isdir(lib_path):
        libs = sorted(glob.glob(os.path.join(lib_path, "*lib")))
        if not arg_dict.get('gr_rc', '') and not arg_dict.get('route_segments', ''):
            libs = [lib for lib in libs if 'ram' not in lib.lower()]

    # platformPath += '/ASAP7' if platformPath != '' else ''
    direct_rc_mode = bool(arg_dict.get('gr_rc', '') or arg_dict.get('route_segments', ''))

    if "ASAP7" in platformPath:
        params = {
            "benchmark": "gzz",
            "design_name": designName,
            "lefs": lefs,
            "libs": libs,
            "def": designPath + "/" + designName + "/" + designName + ".def",
            "sdc": designPath + "/" + designName + "/" + designName + ".sdc",
        }
        if not direct_rc_mode:
            params["spef"] = designPath + "/" + designName + "/" + designName + ".spef"
    else :
        target_techlib_path = "/research/d7/ascstd/qkduan25/TimingPredict/data/netlists/techlib"
        target_lef = os.path.join(target_techlib_path, "merged_unpadded.lef")
        target_early_lib = os.path.join(target_techlib_path, "sky130_fd_sc_hd__ff_n40C_1v95.lib")
        target_late_lib = os.path.join(target_techlib_path, "sky130_fd_sc_hd__ss_100C_1v60.lib")
        use_target_techlib = all(os.path.exists(path) for path in [target_lef, target_early_lib, target_late_lib])
        if use_target_techlib:
            lefs = [target_lef]
            libs = []
        params = {
            "benchmark": "gzz",
            "design_name": designName,
            "lefs": lefs,
            "def": designPath + "/" + designName + "/20-" + designName + ".def",
            "sdc": designPath + "/" + designName + "/" + designName + ".cts_1.sdc",
        }
        if not direct_rc_mode:
            params["spef"] = designPath + "/" + designName + "/20-" + designName + ".spef"
        if use_target_techlib:
            params["early_lib"] = target_early_lib
            params["late_lib"] = target_late_lib
        else:
            params["libs"] = libs
    #? sdc file?

    if "benchmark" not in params.keys():
        raise ValueError("Cannot find 'benchmark' in args.custom_path")
    if "design_name" not in params.keys():
        raise ValueError("Cannot find 'design_name' in args.custom_path")
    args.dataset = params["benchmark"]
    args.design_name = params["design_name"]
    if "lefs" in params.keys():
        logger.info("Detect json LEF/DEF mode. Please make sure that tech_lef are included first.")
        lefs = params["lefs"]
        for i in range(len(lefs)):
            # Simple heuristic to find tech_lef (Only for ASAP7, Nangate45, Sky130, GF180)
            if i == 0:
                continue
            if "tech" in lefs[i] or ".tlef" in lefs[i]:
                lefs[i], lefs[0] = lefs[0], lefs[i]
                break
        params["lefs"] = lefs
    if "output_path" in params.keys():
        args.output_path = params["output_path"]
    for key in params.keys():
        if key in arg_dict.keys():
            arg_dict[key] = params[key]
    #! use original functions here after

    # print info
    content = "Design Info:\n"
    num_items = 0
    if "benchmark" in params.keys():
        content += f"benchmark: {params['benchmark']}\n"
        num_items += 1
    if "design_name" in params.keys():
        content += f"design_name: {params['design_name']}\n"
        num_items += 1
    for key, value in params.items():
        if key == "design_name" or key == "benchmark":
            continue
        content += f"{key}: {value}"
        if num_items < len(params) - 1:
            content += "\n"
        num_items += 1
    logger.info(content)

    data, rawdb, gpdb = load_dataset(args, logger, params)
    logger.info("Design loaded successfully.")
    return data, rawdb, gpdb, params
