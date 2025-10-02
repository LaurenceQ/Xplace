from utils import *
from src import *
def load_design(args,logger):
    # params = find_design_params(args, logger)
    # todo: 改一个新的get_custom_json_params
    # params = get_custom_json_params(args, logger)
    import json
    arg_dict = vars(args)
    # with open(args.custom_json, 'r') as f:
    #     params = json.load(f)
    platformPath = arg_dict.get('platformPath', '')
    designPath = arg_dict.get('designPath', '')
    designName = arg_dict.get('designName', '')
    # platformPath += '/ASAP7' if platformPath != '' else ''
    params = {
        "benchmark": "iccad2025",
        "design_name": designName,
        "lefs": [platformPath + "/ASAP7/techlef/asap7_tech_1x_201209.lef",
                 platformPath + "/ASAP7/LEF/sram_asap7_64x256_1rw.lef",
                 platformPath + "/ASAP7/LEF/asap7sc7p5t_28_L_1x_220121a.lef",
                 platformPath + "/ASAP7/LEF/sram_asap7_16x256_1rw.lef",
                 platformPath + "/ASAP7/LEF/sram_asap7_64x64_1rw.lef",
                 platformPath + "/ASAP7/LEF/asap7sc7p5t_28_R_1x_220121a.lef",
                 platformPath + "/ASAP7/LEF/sram_asap7_32x256_1rw.lef",
                 platformPath + "/ASAP7/LEF/asap7sc7p5t_28_SL_1x_220121a.lef",
                 platformPath + "/ASAP7/LEF/asap7sc7p5t_28_SRAM_1x_220121a.lef"
        ],
        "libs": [platformPath + "/ASAP7/LIB/asap7sc7p5t_AO_LVT_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_AO_RVT_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_AO_SLVT_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_AO_SRAM_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_INVBUF_LVT_TT_nldm_220122.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_INVBUF_RVT_TT_nldm_220122.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_INVBUF_SLVT_TT_nldm_220122.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_INVBUF_SRAM_TT_nldm_220122.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_OA_LVT_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_OA_RVT_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_OA_SLVT_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_OA_SRAM_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_SEQ_LVT_TT_nldm_220123.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_SEQ_SLVT_TT_nldm_220123.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_SEQ_SRAM_TT_nldm_220123.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_SIMPLE_LVT_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_SIMPLE_RVT_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_SIMPLE_SLVT_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/asap7sc7p5t_SIMPLE_SRAM_TT_nldm_211120.lib",
                 platformPath + "/ASAP7/LIB/sram_asap7_16x256_1rw.lib",
                 platformPath + "/ASAP7/LIB/sram_asap7_32x256_1rw.lib",
                 platformPath + "/ASAP7/LIB/sram_asap7_64x256_1rw.lib",
        ],
        "def": designPath + "/" + designName + "/" + designName + ".def"
        # "def": "output/" + designName + ".def"
        
    }
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
    return data, rawdb, gpdb