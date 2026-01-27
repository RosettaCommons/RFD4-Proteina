#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../.ipd/shebang/latent_exec.sh" "$0" "$@"'

import json
import os
import sys
import time
from os import PathLike
from pathlib import Path

import hydra
import lightning as L
import rootutils
import torch
import yaml
from atomworks.common import exists
from atomworks.ml.transforms.base import Transform, Identity
from atomworks.ml.utils.condition import save_atom_array_with_conditions_to_cif
from foundry.utils.env import load_dotenv
from hydra.core.hydra_config import HydraConfig
from foundry.utils.ddp import RankedLogger
from foundry.utils.logging import print_config_tree
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader

from proteinfoundation.datapipes.collate import atom37_collate_fn
from proteinfoundation.datapipes.pipeline import build_design_validation_pipeline
from proteinfoundation.inference_utils.datasets import ContigJsonDataset, DesignProblemDataset
from proteinfoundation.inference_utils.input_parsing import InputSpecification
from proteinfoundation.proteina import Proteina

# Setup root dir and environment variables (more info: https://github.com/ashleve/rootutils)
# NOTE: Sets the `PROJECT_ROOT` environment variable to the root directory of the project (where `.project-root` is located)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logger = RankedLogger(__name__, rank_zero_only=True)

# If the user has set `PROJECT_PATH`, use it to build the config path; otherwise, fall back to `PROJECT_ROOT`
_config_path = os.path.join(os.environ["PROJECT_ROOT"], "projects/proteinfoundation/configs")

def _get_mutually_exclusive_config(cfg: DictConfig, key1: str, key2: str, default_value=None):
    """Get exactly one of two mutually exclusive config keys.

    Primarily used for supporting RFD3 launch syntax, to ease the transition for designers.

    Args:
        cfg: Configuration DictConfig.
        key1: First config key.
        key2: Second config key.
        default_value: Value to return if neither key is provided.

    Returns:
        The value of the specified config key.

    Raises:
        ValueError: If neither or both keys are provided.
    """
    has_key1 = exists(cfg.get(key1, None))
    has_key2 = exists(cfg.get(key2, None))

    if has_key1 and has_key2:
        raise ValueError(f"Please provide only one of `{key1}` or `{key2}` in the config.")
    elif not has_key1 and not has_key2:
        return default_value
    elif has_key1:
        return cfg[key1]
    else:
        return cfg[key2]

def _assemble_dataset_inputs_from_dict(
    data: dict,
    input_file_path: Path,
    subset_to_examples: list[str] | str | None,
    specification_overrides: dict = {},
):
    # ... Apply any global args for this input file
    if "global_args" in data:
        global_args = data.pop("global_args")
        for example in data:
            data[example].update(global_args)

    # ... Subset to keys
    if subset_to_examples is not None:
        subset_to_examples = (
            subset_to_examples.split(",") if isinstance(subset_to_examples, str) else subset_to_examples
        )
        data = {example: data[example] for example in subset_to_examples if example in data}

    # ... Extract
    all_example_specifications = {}
    for example_name, args in data.items():
        # Validate structure file input, if any
        if exists(args.get("input", None)):
            input_structure_file = Path(args["input"])

            # Allow paths relative to the original yaml/json file
            if not input_structure_file.is_absolute():
                input_structure_file_absolute = input_file_path.parent / input_structure_file
            else:
                input_structure_file_absolute = input_structure_file

            if not input_structure_file_absolute.exists():
                raise FileNotFoundError(f"Input structure file {input_structure_file_absolute} does not exist.")

            args["input"] = str(input_structure_file_absolute)

        # ... Set json/yaml key as the example name
        args["extra"] = args.get("extra", {}) | {"example": example_name}

        # Validate input specification
        logger.info(f"Prevalidating design specification for example: {example_name}")
        InputSpecification.safe_init(**args)

        all_example_specifications[example_name] = args | specification_overrides

    return all_example_specifications


def _extract_input_data(inputs: PathLike | list[PathLike]):
    if isinstance(inputs, list):
        inputs = [str(p) for p in inputs]
        if not all([(".cif" in p or ".bcif" in p) for p in inputs]):
            raise ValueError(
                "If passing a list of structure files, they must all be in CIF format. For json/yaml inputs, please pass "
                "only a single file containing entries for each of your design problems."
            )
        use_design_problem_dataset = True
    else:
        inputs = str(inputs)
        if not os.path.exists(inputs):
            raise FileNotFoundError(f"Input path {inputs} does not exist.")
        if Path(inputs).is_dir():
            use_design_problem_dataset = True
        elif ".cif" in inputs or ".bcif" in inputs:
            inputs = [inputs]
            use_design_problem_dataset = True
        elif inputs.endswith(".json"):
            data = json.load(open(inputs))
            use_design_problem_dataset = exists(data.get("cif_creation_strategy", None))
        elif inputs.endswith(".yaml"):
            data = yaml.safe_load(open(inputs))
            use_design_problem_dataset = exists(data.get("cif_creation_strategy", None))
        else:
            raise ValueError(f"Input must be a path to a directory or a .cif/.bcif/.json/.yaml file, got {inputs}.")

    if use_design_problem_dataset:
        return inputs, use_design_problem_dataset
    else:
        return data, use_design_problem_dataset


def split_replicates_by_job(num_replicates: int, job_id: int, njobs: int) -> int:
    """Split total replicates across multiple jobs.

    Args:
        num_replicates: Total number of replicates per example.
        job_id: Current job ID (0-indexed).
        njobs: Total number of parallel jobs.

    Returns:
        Number of replicates for this job_id.
    """
    replicates_per_split = (num_replicates - 1) // njobs + 1
    if replicates_per_split * job_id >= num_replicates:
        logger.info(f"Job id {job_id} gets 0 replicates. Finishing job...")
        sys.exit(0)
    return min(replicates_per_split, num_replicates - replicates_per_split * job_id)


def get_dataset_from_cfg(cfg: DictConfig, job_id: int) -> torch.utils.data.Dataset:
    # validate inputs
    cfg_gen = cfg.generation
    check_cfg_validity(cfg_gen.args)

    # Get the desired number of replicates and adjust for job splitting
    njobs = cfg.get("gen_njobs", 1)
    num_replicates = cfg.get("num_replicates", 1)
    num_replicates = split_replicates_by_job(num_replicates, job_id, njobs)

    # If a design Dataset was passed, use it directly
    cfg_gen.dataset.num_replicates = 1  # Temporarily set to 1 to get the number of examples
    instantiated_dset = hydra.utils.instantiate(cfg_gen.dataset)
    if isinstance(instantiated_dset, (DesignProblemDataset | ContigJsonDataset)):
        cfg_gen.dataset.num_replicates = num_replicates
        instantiated_dset = hydra.utils.instantiate(cfg_gen.dataset)  # Re-instantiate with correct num_replicates
        return instantiated_dset

    # Otherwise, we will create a Dataset based on the user specifications
    assert exists(cfg.inputs), "Must provide `inputs` config if not using a pre-instantiated Dataset."
    subset_to_examples = _get_mutually_exclusive_config(cfg, "subset_to_examples", "json_keys_subset", default_value=None)
    specification_overrides = dict(cfg_gen.dataset.specification) if "specification" in cfg_gen.dataset else {}

    metrics_tags = cfg_gen.dataset.transform.get("metrics_tags") if "transform" in cfg_gen.dataset else None
    inference_pipeline = build_design_validation_pipeline(metrics_tags=metrics_tags)

    dataset = get_contig_or_design_problem_dataset(
        cfg.inputs,
        num_replicates=num_replicates,
        transform=inference_pipeline,
        subset_to_examples=subset_to_examples,
        specification_overrides=specification_overrides,
    )

    return dataset


def get_contig_or_design_problem_dataset(
    json_or_design_problem_inputs: str | list[str],
    num_replicates: int | None = None,
    transform: Transform | None = None,
    subset_to_examples: list[str] | str | None = None,
    specification_overrides: dict = {},
) -> torch.utils.data.Dataset:
    """Create either a ContigJsonDataset or DesignProblemDataset based on the input type.

    Args:
        json_or_design_problem_inputs: A PFM-style json/yaml file, or a list/directory of Condition-annotated CIF files.
        num_replicates: Number of times to replicate each example. If None, defaults to 1 replicate per example.
        transform: Transform apply to loaded dataset examples. If None, an identity transform is used.
        subset_to_examples: If provided, only use these example keys from the json/yaml input.
        specification_overrides: Dictionary of input specification overrides (only applies to ContigJsonDataset).

    Returns:
        A torch Dataset instance.
    """
    # Determine which dataset class to create
    data, use_design_problem_dataset = _extract_input_data(json_or_design_problem_inputs)

    # Default to 1 replicate per example
    if not exists(num_replicates):
        num_replicates = 1

    # Create the appropriate Dataset
    if use_design_problem_dataset:
        return DesignProblemDataset(
            dir_path_or_file_paths=data,
            transform=transform if exists(transform) else Identity(),
            filter_fn=lambda x: str(x).endswith((".cif", ".json", ".yaml")),
            num_replicates=num_replicates,
        )

    data = _assemble_dataset_inputs_from_dict(
        data,
        input_file_path=Path(json_or_design_problem_inputs),
        subset_to_examples=subset_to_examples,
        specification_overrides=specification_overrides,
    )

    return ContigJsonDataset(
        data=data,
        transform=transform if exists(transform) else Identity(),
        name="inference-dataset",
        num_replicates=num_replicates,
    )


def infer_root_path(create_root: bool = True, config_name: str = ".") -> str:
    """
    Sets the right seed and creates the root directory where the run will store things.

    Returns:
        Path of the root directory (string)
    """
    logger.info(" ".join(sys.argv))
    # Set root path for this inference run
    root_path = f"./inference/{config_name}"
    if create_root:
        os.makedirs(root_path, exist_ok=True)
    else:
        if not os.path.exists(root_path):
            raise ValueError("Results path %s does not exist" % root_path)

    return root_path


def check_cfg_validity(cfg_sample_args: dict) -> None:
    """
    Checks if guidance arguments (CFG and AG) are valid.
    """
    # Logging CFG
    if cfg_sample_args.guidance_w != 1.0:
        logger.info(
            f"Guidance is turned on with guidance weight {cfg_sample_args.guidance_w} and autoguidance ratio {cfg_sample_args.ag_ratio}."
        )
        assert (
            cfg_sample_args.ag_ratio >= 0.0 and cfg_sample_args.ag_ratio <= 1.0
        ), f"Autoguidance ratio should be between 0 and 1, but now is {cfg_sample_args.ag_ratio}."
        assert (cfg_sample_args.ag_ratio == 0.0) or (
            cfg_sample_args.ag_ckpt_path is not None
        ), "Autoguidance checkpoint path should be provided"
    else:
        logger.info("Guidance is turned off.")


def load_ag_ckpt(cfg: dict) -> None | torch.nn.Module:
    """
    Loads the neural network for the "bad" checkpoint in autoguidance, if requested.

    Returns:
        A nn module, if autogudance enabled.
    """
    nn_ag = None
    if cfg.ag_ratio > 0 and cfg.guidance_w != 1.0:
        logger.info(
            f"Using autoguidance with guidance weight {cfg.guidance_w} and autoguidance ratio {cfg.ag_ratio} based on the checkpoint {cfg.ag_ckpt_path}"
        )
        ckpt_ag_file = cfg.ag_ckpt_path
        assert os.path.exists(ckpt_ag_file), f"Not a valid checkpoint {ckpt_ag_file}"
        model_ag = Proteina.load_from_checkpoint(ckpt_ag_file, strict=False)

        # OPTIMIZATION: Remove encoder from autoguidance model autoencoder during generation (only decoder needed)
        if model_ag.autoencoder is not None:
            logger.info("Removing autoencoder encoder from autoguidance model during generation to save memory")
            del model_ag.autoencoder.encoder
            model_ag.autoencoder.encoder = None

        nn_ag = model_ag.nn
    return nn_ag


def configure_search_strategy_from_cfg(search_cfg: DictConfig):
    """Configure search strategy from config.

    Args:
        search_cfg: Search configuration with strategy definition

    Returns:
        SearchStrategy instance
    """
    if not hasattr(search_cfg, "strategy") or search_cfg.strategy is None:
        raise ValueError(
            "Search config requires 'strategy:' field with _target_ definition. "
            "Example: search.strategy._target_: proteinfoundation.search.beam_search.BeamSearchStrategy"
        )

    search_strategy = hydra.utils.instantiate(search_cfg.strategy)
    logger.info(f"Configured search strategy: {type(search_strategy).__name__}")
    return search_strategy


def load_ckpt_n_configure_inference(cfg: dict) -> Proteina:
    """Load model checkpoint for inference."""
    ckpt_path = cfg.ckpt_path
    ckpt_file = os.path.join(ckpt_path, cfg.ckpt_name)

    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    logger.info(f"Loading checkpoint: {ckpt_file}")

    model = Proteina.load_from_checkpoint(
        ckpt_file,
        strict=False,
        autoencoder_ckpt_path=cfg.get("autoencoder_ckpt_path"),
        weights_only=False,
        map_location="cpu",
    )

    # Remove val sets used in training, and do not compute loss on inference examples, since they start with NaN coords
    model.cfg_exp.datasets.val = OmegaConf.create(
        {"inference_dataset": {"dataset": cfg.generation.dataset, "compute_loss": False}}
    )

    # Set inference variables and potentially load autoguidance
    nn_ag = load_ag_ckpt(cfg.generation.args)

    metrics = cfg.get("metrics")
    metrics_tags = (
        cfg.generation.dataset.transform.get("metrics_tags") if "transform" in cfg.generation.dataset else None
    )
    if exists(metrics) != exists(metrics_tags):
        logger.warning(
            f"Metrics is {"not" if exists(metrics) else ""} None but metrics_tags is {"not" if exists(metrics_tags) else ""} None. "
            "No metrics will be computed."
        )

    model.configure_inference(
        cfg.generation,
        nn_ag=nn_ag,
        store_dir=cfg.get("out_dir"),
        metrics_cfg=metrics,
        offline_validation=cfg.get("offline_validation", False),
        job_id=cfg.get("job_id"),
    )

    # Configure search if specified
    if hasattr(cfg, "search") and cfg.search is not None:
        search_strategy = configure_search_strategy_from_cfg(cfg.search)

        # Initialize search reward model from nested search.rewards (can be non-differentiable)
        search_reward_model = None
        if hasattr(cfg.search, "rewards") and cfg.search.rewards is not None:
            from foundry.metrics.metric import MetricManager

            from proteinfoundation.rewards import RewardModel

            metric_manager = MetricManager.instantiate_from_hydra(cfg.search.rewards.metrics)
            search_reward_model = RewardModel(
                metrics=metric_manager,
                weights=dict(cfg.search.rewards.weights),
                differentiable=cfg.search.rewards.get("differentiable", False),
            )
            logger.info(
                f"Search reward model: {len(metric_manager.metrics)} metrics, "
                f"differentiable={search_reward_model.differentiable}"
            )

        model.configure_search(search_strategy, search_reward_model)

    # Configure refinement if specified
    if hasattr(cfg, "refinement") and cfg.refinement is not None:
        from foundry.metrics.metric import MetricManager

        from proteinfoundation.rewards import RewardModel

        # Require nested refinement.rewards config (MUST be differentiable)
        if not hasattr(cfg.refinement, "rewards") or cfg.refinement.rewards is None:
            raise ValueError(
                "Refinement requires 'refinement.rewards:' config section. "
                "Add rewards under refinement with differentiable=true."
            )

        # Create refinement reward model
        metric_manager = MetricManager.instantiate_from_hydra(cfg.refinement.rewards.metrics)
        refinement_reward_model = RewardModel(
            metrics=metric_manager,
            weights=dict(cfg.refinement.rewards.weights),
            differentiable=cfg.refinement.rewards.get("differentiable", False),
        )

        if not refinement_reward_model.differentiable:
            raise ValueError("Refinement requires differentiable=true in refinement.rewards config")

        logger.info(
            f"Refinement reward model: {len(metric_manager.metrics)} metrics, "
            f"differentiable={refinement_reward_model.differentiable}"
        )

        # Configure refinement
        refinement_strategy = hydra.utils.instantiate(cfg.refinement.strategy)
        model.configure_refinement(refinement_strategy, refinement_reward_model)

    return model




def save_predictions(
    root_path: str,
    predictions: list[list],
    job_id: int = 0,
) -> None:
    """
    Saves generated samples.

    Args:
        root_path: root directory where samples will be stored (within subdirectories)/
        predictions: List of lists where each element is either:
            - tuple (coors [n, 37, 3], aatype [n]) - legacy format
            - dict with keys 'coors', 'residue_type', 'mask', 'atom_array' - new format
        job_id: job number, used to store files.
    """
    predictions = [sample for sublist in predictions for sample in sublist]

    for pred in predictions:
        atom_array = pred["generated_atom_array"]
        out_name = f"{pred["example_id"]}_job_{job_id}.cif"

        # Save generated structure (will be saved as CIF format)
        file_path = os.path.join(root_path, out_name)
        extra_annotations_to_save = [
            "mapped_to_coordinate_conditioned_region",
            "mapped_to_coordinate_conditioned_region_token_lvl",
        ]

        save_atom_array_with_conditions_to_cif(atom_array, file_path, extra_annotations=extra_annotations_to_save)


def run_generation(cfg: DictConfig, job_id: int = 0, config_name: str | None = None) -> None:
    """
    Core generation function that runs inference given a Hydra config.

    Args:
        cfg: Hydra DictConfig with all generation parameters
        job_id: Job ID for parallel job splitting
        config_name: Name of the config (used for output directory naming)
    """

    # Determine config name for output paths
    if config_name is None:
        config_name = cfg.get("run_name_", "inference")

    root_path = cfg.get("out_dir", None)
    if not exists(root_path):
        root_path = infer_root_path(create_root=True, config_name=config_name)

    # Set seed
    if exists(cfg.seed):
        base_seed = cfg.seed
        logger.info(f"Using provided base seed: {base_seed}")
    else:
        base_seed = int(time.time() * 1000) % (2**31)
        logger.info(f"Using generated base seed: {base_seed}")

    cfg.seed = base_seed + job_id  # Different seeds for different splits ids
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    # Exit if results from analysis already exist
    csv_filename = f"results_{config_name}_{job_id}.csv"
    csv_path = os.path.join(root_path, "..", csv_filename)
    if os.path.exists(csv_path):
        logger.info(f"Results already exist at {csv_path}. Exiting generate.py.")
        sys.exit(0)

    # Load model
    model = load_ckpt_n_configure_inference(cfg)

    dataset = get_dataset_from_cfg(cfg, job_id)

    # Support either batch_size argument for backwards compatibility with rfd3 syntax
    batch_size = _get_mutually_exclusive_config(cfg, "batch_size", "diffusion_batch_size", default_value=1)

    # Create dataloader
    num_workers = cfg.num_workers
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=atom37_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"✓ Dataloader created: {len(dataset)} samples, batch_size={batch_size}")

    # Sample model
    logger.info(f"Running inference on {len(dataloader)} batches...")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.hardware.ngpus_per_node_,  # This is number of gpus per node, not total
        num_nodes=cfg.hardware.nnodes_,
        default_root_dir=root_path,
        logger=False,
    )

    if model.metrics is not None:
        trainer.validate(model, dataloader)  # validation_step automatically saves outputs
    else:
        predictions = trainer.predict(model, dataloader)  # list of lists of dicts,
        # first list is for each batch, second list is for each sample in the batch
        # Save predictions to cif files
        save_predictions(
            root_path,
            predictions,
            job_id=job_id,
        )

    logger.info(f"✓ Generation complete! Results saved to {root_path}")


@hydra.main(version_base="1.3", config_path=_config_path, config_name="inference")
def main_hydra(cfg: DictConfig) -> None:
    """
    Hydra-wrapped main function for generation with command-line config overrides.

    Usage:
        # Use default config
        python generate.py

        # Override config file
        python generate.py --config-name=inference_ucond_tri

        # Override specific parameters
        python generate.py generation.args.nsteps=200 seed=42

    Args:
        cfg: Hydra DictConfig automatically loaded from config files with CLI overrides applied
    """
    load_dotenv()
    job_id = cfg.get("job_id", 0)
    if not exists(job_id):
        job_id = 0

    if exists(cfg.get("data_path", None)):
        os.environ["DATA_PATH"] = cfg.data_path

    # Get config name from hydra runtime
    config_name = HydraConfig.get().job.config_name

    logger.info(f"Starting generation with config: {config_name}")
    print_config_tree(cfg, resolve=True)

    run_generation(cfg, job_id=job_id, config_name=config_name)


if __name__ == "__main__":
    main_hydra()
