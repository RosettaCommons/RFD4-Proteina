import argparse
import itertools
import os
import shutil

import hydra
from modelhub.utils.ddp import RankedLogger
from omegaconf import OmegaConf

logger = RankedLogger(__name__, rank_zero_only=True)

"""
Given a base config file for inference, this does an exhaustive sweep over parameters as indicated in the sweeper 
dictionary. Then, for each resulting combination, it creates a separate config file.

If some parameter is not specified in the sweeper, then the default value in the base config file is kept.

If you don't want to do a sweep, just use the sweeper dictionary given by

sweeper = {
    "seed": [1],
}

Note: This is mostly used for the cluster. If we want to evaluate multiple checkpoints, with and without self-conditioning,
parallelize over seeds for more efficient sampling and evaluation, etc. The script to submit the inference job to the cluster
calls this file, and submits a separate job for each of the config files produced.
"""


# # Define varibles to sweep over
# # For now, always have to be from first level stuff, need to specify dictionaries if
# # want to sweep over not 1st level variables

# Complexa paper PPI benchmark targets
PPI_TARGETS = [
    "01_PD1",
    "02_PDL1",
    "03_PDL1_AAV",
    "04_IFNAR2",
    "05_CD45",
    "06_CD45",
    "07_CD45",
    "08_CD45",
    "09_CD45",
    "10_CD45",
    "11_CD45",
    "12_Claudin1",
    "13_BBF14",
    "14_CrSAS6",
    "15_DerF7",
    "16_DerF7",
    "17_DerF7",
    "18_DerF21",
    "19_DerF21",
    "20_DerF21",
    "21_DerF21",
    "22_DerF21",
    "23_BetV1",
    "24_SpCas9",
    "25_CbAgo",
    "26_CbAgo",
    "27_HER2_AAV",
    "28_HER2_AAV",
    "29_BHRF1",
    "30_SC2RBD",
    "31_IL7RA",
    "32_PDL1_ALPHA",
    "33_TrkA",
    "34_Insulin",
    "35_H1",
    "36_VEGFA",
    "37_IL17A",
    "38_TNFalpha",
]

sweeper = {
    "generation.args.nsteps": [400],
    "ckpt_path": [
        "/path/to/shared/storage"
    ],
    "ckpt_name": ["chk_epoch=00000169_step=000000390000.ckpt"],
    # Use OmegaConf interpolation to get absolute path via PROJECT_ROOT env var (set by rootutils)
    "inputs": ["${oc.env:PROJECT_ROOT}/projects/proteinfoundation/benchmarks/complexa_paper_ppi.yaml"],
    "num_replicates": [200],
    "batch_size": [20],
    "gen_njobs": [10],
    "generation.dataset.subset_to_examples": PPI_TARGETS,
}


"""
You can specify the nested dictionary keys by using dot keys.
If one argument is specified twice, the one with large depth win.
E.g.: if you specify guidance_weight with
"generation.args.guidance_w": 20
"generation.args": {"guidance_w": 10}
You will get the output {"generation": {"args": {"guidance_w": 20}}}.
To avoid this confusion, please never specify one key twice!!!!!
"""


def dot_key_dict_to_nested_dict(input_dict):
    """This function recursively change dictonary with dot keys to nested dictionary.
    input_dict = {
        "A.B.1": False,
        "A.B.2": 100,
        "A.B.3": 1.5,
        "A.C.1": {"a": "vf", "b": 1.0, "c": 0.0},
    }
    output_dict = {
        "A": {
            "B": {
                "1": False,
                "2": 100,
                "3": 1.5,
            },
            "C": {
                "1": {"a": "vf", "b": 1.0, "c": 0.0}
            }
        }
    }
    """
    output_dict = {}
    child_keys = set()
    for key, value in input_dict.items():
        if "." not in key:
            # No child
            output_dict[key] = value
        else:
            child_keys.add(key.split(".")[0])

    for child_key in child_keys:
        child_input_dict = {}
        for key, value in input_dict.items():
            if key.startswith(child_key + "."):
                child_input_dict[".".join(key.split(".")[1:])] = value
        child_output_dict = dot_key_dict_to_nested_dict(child_input_dict)

        if child_key in output_dict:
            # If the child_key itself is already in output_dict, it should be a dict and be override by child_output_dict
            assert isinstance(
                output_dict[child_key], dict
            ), f"Key {child_key} is supposed to map to a dict, but get {type(output_dict[child_key])}"
            output_dict[child_key].update(child_output_dict)
        else:
            output_dict[child_key] = child_output_dict

    return output_dict


def apply_sweeper_n_save_configs(cfg_base, sweeper, path_new_configs, last_idx_used=0):
    # Create all possible combinations of variables in the sweeper
    keys = sweeper.keys()
    values = sweeper.values()
    combinations = itertools.product(*values)
    combinations = [dict(zip(keys, combination, strict=False)) for combination in combinations]

    # Generate and save all generated configs
    for i, comb in enumerate(combinations):
        comb = dot_key_dict_to_nested_dict(comb)  # Transform to nested dict
        override_cfg = OmegaConf.create(comb)  # Create sub-config
        cfg_o = OmegaConf.merge(cfg_base, override_cfg)  # Override original
        name = f"inf_{last_idx_used + i + 1}.yaml"
        OmegaConf.save(config=cfg_o, f=os.path.join(path_new_configs, name))  # Save new config
    return i + 1


parser = argparse.ArgumentParser(description="Job info")
parser.add_argument(
    "--dir_cfgs",
    type=str,
    default="./configs/inference_configs",
    help="Directory used to store the generated config files.",
)
parser.add_argument(
    "--no_standard_metrics",
    action="store_true",
    help="Do not generate configs for standard metrics (designability, diversity, novelty). If included will generate standard metrics stuff, otherwise not)",
)
parser.add_argument(
    "--no_fid_metrics",
    action="store_true",
    help="Do not generate configs for fid metrics (if included will not generate fid stuff, otherwise yes).",
)
parser.add_argument(
    "--fid_full_backbone",
    action="store_true",
    help="Whether to compute fid metrics for full backbone (if included will use full backbone, otherwise not).",
)
args = parser.parse_args()


# Create/clean directory where new configs will be stores and save all configs, one per combination
path_new_configs = args.dir_cfgs
if os.path.exists(path_new_configs):
    shutil.rmtree(path_new_configs)  # Remove old configs
os.makedirs(path_new_configs, exist_ok=False)

base_config_path = "../configs"
last_idx_used = -1

if not args.no_standard_metrics:
    with hydra.initialize(base_config_path, version_base=hydra.__version__):
        cfg = hydra.compose(config_name="inference")
        logger.info(f"Loaded base inference config {cfg}")
    last_idx_used = apply_sweeper_n_save_configs(cfg, sweeper, path_new_configs, last_idx_used)

if not args.no_fid_metrics:
    config_name = "inference_fid_ff" if args.fid_full_backbone else "inference_fid_ca"
    with hydra.initialize(base_config_path, version_base=hydra.__version__):
        cfg_fid = hydra.compose(config_name=config_name)
        logger.info(f"Loaded fid inference config {cfg_fid}")
    sweeper_fid = {k: sweeper[k] for k in sweeper if k != "seed"}  # Remove seed as just one single process
    last_idx_used = apply_sweeper_n_save_configs(cfg_fid, sweeper_fid, path_new_configs, last_idx_used)
