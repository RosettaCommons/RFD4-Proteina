#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../.ipd/shebang/latent_inference_exec.sh" "$0" "$@"'

import os
import sys
import time
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
import torch
from atomworks.common import exists
from atomworks.ml.transforms.base import Identity, Transform
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset

from rfproteina.datapipes.collate import collate_batch
from rfproteina.datapipes.pipeline import build_design_validation_pipeline
from rfproteina.guidance.guidance_model import GuidanceModel, build_guidance_terms
from rfproteina.inference_utils.condition_spec import is_legacy_dialect, load_spec_file
from rfproteina.inference_utils.datasets import (
    ConditionSpecDataset,
    DesignProblemDataset,
)
from rfproteina.metrics.metric import MetricManager
from rfproteina.proteina import Proteina
from rfproteina.search.base_strategy import SearchStrategy
from rfproteina.utils.common import is_structure_file
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.env import load_dotenv
from rfproteina.utils.logging import print_config_tree
from rfproteina.utils.simple_ema import SimpleEMA
from rfproteina.utils.torch import apply_torch_deterministic
from rfproteina.utils.torch_compile import load_compile_cache, save_compile_cache

# Setup root dir and environment variables (more info: https://github.com/ashleve/rootutils)
# NOTE: Sets the `PROJECT_ROOT` environment variable to the root directory of the project
# (where `.project-root` is located)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

logger = RankedLogger(__name__, rank_zero_only=True)

# If the user has set `PROJECT_PATH`, use it to build the config path; otherwise, fall back to `PROJECT_ROOT`
_config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")


def _get_mutually_exclusive_config(cfg: DictConfig, key1: str, key2: str, default_value: Any = None) -> Any:
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


def _extract_input_data(inputs: PathLike | list[PathLike]) -> tuple[Any, str]:
    """Resolve `inputs=` to a `DesignProblemDataset` (CIF) or `ConditionSpecDataset` source.

    A `.json`/`.yaml` file is read as the condition_spec format (`docs/condition_spec_yaml.md`).
    The legacy contig-JSON dialect has been removed: such a file is detected (see
    :py:func:`~rfproteina.inference_utils.condition_spec.is_legacy_dialect`) and rejected with a
    migration error rather than misinterpreted as condition_spec.

    Returns:
      A ``(data, kind)`` pair, where ``kind`` is one of ``"cif"`` or ``"condition_spec"``.

    Raises:
      NotImplementedError: If the input is a legacy contig-JSON file.
    """
    if isinstance(inputs, list | ListConfig):
        inputs = [str(p) for p in inputs]
        if not all(is_structure_file(p) for p in inputs):
            raise ValueError(
                "If passing a list of structure files, they must all be in CIF format. "
                "For json/yaml inputs, please pass only a single file containing entries "
                "for each of your design problems."
            )
        return inputs, "cif"

    inputs = str(inputs)
    if not os.path.exists(inputs):
        raise FileNotFoundError(f"Input path {inputs} does not exist.")
    if Path(inputs).is_dir():
        return inputs, "cif"
    if is_structure_file(inputs):
        return [inputs], "cif"
    if inputs.endswith((".json", ".yaml", ".yml")):
        data = load_spec_file(Path(inputs))
        if is_legacy_dialect(data):
            raise NotImplementedError(
                f"{inputs}: this file uses the legacy contig-JSON (`InputSpecification`) dialect, which "
                "has been removed. Convert it to Condition-annotated CIFs (scripts/json2cif/build_cif.py) "
                "or rewrite it in the condition_spec format (docs/condition_spec_yaml.md)."
            )
        return data, "condition_spec"
    raise ValueError(f"Input must be a path to a directory or a .cif/.bcif/.json/.yaml file, got {inputs}.")


def split_replicates_by_job(num_replicates: int, job_id: int, njobs: int) -> int:
    """Split total replicates as evenly as possible across multiple jobs.

    The remainder is spread one-per-job over the lowest job ids, so no job is starved when
    ``njobs <= num_replicates`` (e.g. 10 replicates over 8 jobs gives jobs 0-1 two each and
    jobs 2-7 one each).

    Args:
        num_replicates: Total number of replicates per example.
        job_id: Current job ID (0-indexed).
        njobs: Total number of parallel jobs.

    Returns:
        Number of replicates for this job_id.
    """
    replicates_per_split, remainder = divmod(num_replicates, njobs)
    return replicates_per_split + int(job_id < remainder)


def plan_job_split(num_replicates: int, job_id: int, njobs: int) -> tuple[int, int | None, int | None]:
    """Decide which axis ``gen_njobs`` divides for this job.

    Replicates are divided while there are enough to go round (at least one per job). Beyond that
    the surplus jobs would get zero replicates, so the examples are divided instead and every job
    keeps the full replicate count.

    Args:
        num_replicates: Total number of replicates per example.
        job_id: Current job ID (0-indexed).
        njobs: Total number of parallel jobs.

    Returns:
        ``(num_replicates, shard_index, num_shards)``, where the shard values are ``None`` when
        replicates were divided and ``num_replicates`` is this job's share.
    """
    if njobs <= num_replicates:
        return split_replicates_by_job(num_replicates, job_id, njobs), None, None
    return num_replicates, job_id, njobs


def get_dataset_from_cfg(cfg: DictConfig, job_id: int) -> torch.utils.data.Dataset:
    # validate inputs
    cfg_gen = cfg.generation
    check_cfg_validity(cfg_gen.args)

    # Divide the work across parallel jobs; see `plan_job_split`.
    njobs = cfg.get("gen_njobs", 1)
    num_replicates = cfg.get("num_replicates", 1)
    num_replicates, shard_index, num_shards = plan_job_split(num_replicates, job_id, njobs)

    # If a design Dataset was passed, use it directly
    cfg_gen.dataset.num_replicates = num_replicates
    if exists(num_shards):
        # A pre-instantiated dataset must be sharded too, or every surplus job repeats the whole set.
        with open_dict(cfg_gen.dataset):
            cfg_gen.dataset.shard_index = shard_index
            cfg_gen.dataset.num_shards = num_shards
    instantiated_dset = hydra.utils.instantiate(cfg_gen.dataset)
    if isinstance(instantiated_dset, (DesignProblemDataset | ConditionSpecDataset)):
        if cfg_gen.val_set_name_override is None:
            if "data" in cfg_gen.dataset:
                cfg_gen.val_set_name_override = cfg_gen.dataset.data
            elif "dir_path_or_file_paths" in cfg_gen.dataset:
                if isinstance(cfg_gen.dataset.dir_path_or_file_paths, list | ListConfig):
                    cfg_gen.val_set_name_override = ",".join(cfg_gen.dataset.dir_path_or_file_paths)
                else:
                    cfg_gen.val_set_name_override = cfg_gen.dataset.dir_path_or_file_paths
        return instantiated_dset

    # Otherwise, we will create a Dataset based on the user specifications
    assert exists(cfg.inputs), "Must provide `inputs` config if not using a pre-instantiated Dataset."
    subset_to_examples = _get_mutually_exclusive_config(
        cfg, "subset_to_examples", "json_keys_subset", default_value=None
    )
    # Infer partial-diffusion runs from the generation args so the partial_diffusion metrics tag
    # is added below (inverse folding, or an explicit partial-t / trajectory-fraction argument).
    cfg_args = OmegaConf.to_container(cfg_gen.args, resolve=True) if hasattr(cfg_gen, "args") else {}
    is_partial_diffusion = cfg_args.get("inverse_folding", False) or any(
        cfg_args.get(k) is not None
        for k in ("partial_t_backbone", "partial_t_latent", "partial_trajectory_fraction_steps")
    )

    transform_overrides = (
        OmegaConf.to_container(cfg_gen.dataset.transform, resolve=True) if "transform" in cfg_gen.dataset else {}
    )
    metrics_tags = transform_overrides.pop("metrics_tags", None)
    if isinstance(metrics_tags, str):
        metrics_tags = metrics_tags.split(",")
    if is_partial_diffusion:
        if metrics_tags is None:
            metrics_tags = []
        if "partial_diffusion" not in metrics_tags:
            metrics_tags.append("partial_diffusion")
    inference_pipeline = build_design_validation_pipeline(metrics_tags=metrics_tags, **transform_overrides)

    sampling_transforms = (
        hydra.utils.instantiate(cfg_gen.dataset.sampling_transforms)
        if "sampling_transforms" in cfg_gen.dataset
        else None
    )

    dataset = get_contig_or_design_problem_dataset(
        cfg.inputs,
        num_replicates=num_replicates,
        transform=inference_pipeline,
        sampling_transforms=sampling_transforms,
        subset_to_examples=subset_to_examples,
        global_prefix=cfg.get("global_prefix"),
        prevalidate=cfg.get("prevalidate_inputs", False),
        shard_index=shard_index,
        num_shards=num_shards,
    )

    if cfg_gen.val_set_name_override is None:
        cfg_gen.val_set_name_override = str(cfg.inputs)

    return dataset


def get_contig_or_design_problem_dataset(
    json_or_design_problem_inputs: str | list[str],
    num_replicates: int | None = None,
    transform: Transform | None = None,
    sampling_transforms: list[Transform] | None = None,
    subset_to_examples: list[str] | str | None = None,
    global_prefix: str | None = None,
    prevalidate: bool = False,
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> torch.utils.data.Dataset:
    """Create a DesignProblemDataset (CIF) or ConditionSpecDataset based on the input type.

    Args:
        json_or_design_problem_inputs: A condition_spec json/yaml file, or a list/directory of
            Condition-annotated CIF files.
        num_replicates: Number of times to replicate each example. If None, defaults to 1 replicate per example.
        transform: Transform apply to loaded dataset examples. If None, an identity transform is used.
        sampling_transforms: Per-replicate condition-sampling transforms.
        subset_to_examples: If provided, only use these example keys from the json/yaml input.
        global_prefix: Optional global prefix for output example ids.
        prevalidate: If True, build + validate each unique input once up front (fail fast).
        shard_index: Which shard of the replicated examples this job serves, in ``[0, num_shards)``.
        num_shards: Number of shards to divide the replicated examples into; see
            :py:func:`~rfproteina.inference_utils.datasets.take_shard`.

    Returns:
        A torch Dataset instance.
    """
    # Determine which dataset class to create
    data, kind = _extract_input_data(json_or_design_problem_inputs)

    # Default to 1 replicate per example
    if not exists(num_replicates):
        num_replicates = 1

    if kind == "cif":
        return DesignProblemDataset(
            dir_path_or_file_paths=data,
            transform=transform if exists(transform) else Identity(),
            sampling_transforms=sampling_transforms,
            num_replicates=num_replicates,
            global_prefix=global_prefix,
            prevalidate=prevalidate,
            shard_index=shard_index,
            num_shards=num_shards,
        )

    # kind == "condition_spec"
    source_path = str(Path(json_or_design_problem_inputs).resolve())
    if subset_to_examples is not None:
        subset_to_keys = (
            subset_to_examples.split(",") if isinstance(subset_to_examples, str) else list(subset_to_examples)
        )
    else:
        subset_to_keys = None
    return ConditionSpecDataset(
        data=data,
        transform=transform if exists(transform) else Identity(),
        name="inference-dataset",
        sampling_transforms=sampling_transforms,
        subset_to_keys=subset_to_keys,
        num_replicates=num_replicates,
        global_prefix=global_prefix,
        source_path=source_path,
        prevalidate=prevalidate,
        shard_index=shard_index,
        num_shards=num_shards,
    )


def infer_root_path(create_root: bool = True, config_name: str = ".") -> str:
    """
    Creates the root directory where inference outputs will be stored.

    Returns:
        Path of the root directory (string)
    """
    logger.info(" ".join(sys.argv))
    root_path = f"./runs/{config_name}"
    if create_root:
        os.makedirs(root_path, exist_ok=True)
    else:
        if not os.path.exists(root_path):
            raise ValueError(f"Results path {root_path} does not exist")

    return root_path


def check_cfg_validity(cfg_sample_args: dict) -> None:
    """
    Checks if guidance arguments (CFG and AG) are valid.
    """
    # Logging CFG
    if cfg_sample_args.guidance_w != 1.0:
        logger.info(
            f"Guidance is turned on with guidance weight {cfg_sample_args.guidance_w} "
            f"and autoguidance ratio {cfg_sample_args.ag_ratio}."
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
            f"Using autoguidance with guidance weight {cfg.guidance_w} and autoguidance ratio {cfg.ag_ratio} "
            f"based on the checkpoint {cfg.ag_ckpt_path}"
        )
        ckpt_ag_file = cfg.ag_ckpt_path
        assert os.path.exists(ckpt_ag_file), f"Not a valid checkpoint {ckpt_ag_file}"
        model_ag = Proteina.load_from_checkpoint(ckpt_ag_file, strict=True)

        # OPTIMIZATION: Remove encoder from autoguidance model autoencoder during generation (only decoder needed)
        if model_ag.autoencoder is not None:
            logger.info("Removing autoencoder encoder from autoguidance model during generation to save memory")
            del model_ag.autoencoder.encoder
            model_ag.autoencoder.encoder = None

        nn_ag = model_ag.nn
    return nn_ag


def configure_search_strategy_from_cfg(search_cfg: DictConfig) -> SearchStrategy:
    """Configure search strategy from config.

    Args:
        search_cfg: Search configuration with strategy definition

    Returns:
        SearchStrategy instance
    """
    if not hasattr(search_cfg, "strategy") or search_cfg.strategy is None:
        raise ValueError(
            "Search config requires 'strategy:' field with _target_ definition. "
            "Example: search.strategy._target_: rfproteina.search.beam_search.BeamSearchStrategy"
        )

    search_strategy = hydra.utils.instantiate(search_cfg.strategy)
    logger.info(f"Configured search strategy: {type(search_strategy).__name__}")
    return search_strategy


def _load_ema_from_checkpoint(ckpt_file: str) -> SimpleEMA | None:
    """Load SimpleEMA callback state from a checkpoint if present.

    Args:
        ckpt_file: Path to the checkpoint file.

    Returns:
        Configured SimpleEMA ready for inference (EMA weights will be swapped in on
        validation_start / loaded directly before predict), or None if no EMA state found.
    """
    raw_ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    callbacks_state = raw_ckpt.get("callbacks", {})

    ema_state = next((v for k, v in callbacks_state.items() if "SimpleEMA" in k), None)
    if ema_state is None or not ema_state.get("ema_state_dict"):
        logger.info("No EMA state found in checkpoint; inference will use training weights.")
        return None

    decay = ema_state.get("decay", 0.999)
    update_every_n = ema_state.get("update_every_n", 1)
    ema_callback = SimpleEMA(decay=decay, update_every_n=update_every_n)
    ema_callback.load_state_dict(ema_state)
    logger.info(f"Found EMA state in checkpoint (decay={decay}). Inference will use EMA weights.")
    return ema_callback


def load_ckpt_n_configure_inference(cfg: dict) -> tuple[Proteina, SimpleEMA | None]:
    """Load model checkpoint for inference."""
    ckpt_path = cfg.ckpt_path
    ckpt_file = os.path.join(ckpt_path, cfg.ckpt_name)

    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    logger.info(f"Loading checkpoint: {ckpt_file}")

    # strict=True: the architecture is rebuilt from the checkpoint's own saved hparams, so any
    # missing/unexpected key means the code has drifted from what produced these weights. Loading
    # a partially-initialized model would silently produce garbage designs.
    model = Proteina.load_from_checkpoint(
        ckpt_file,
        strict=True,
        autoencoder_ckpt_path=cfg.get("autoencoder_ckpt_path"),
        weights_only=False,
        map_location="cpu",
    )

    # An "-ema" checkpoint's weights are already EMA (loaded above), so re-applying the EMA
    # callback is redundant and can mismatch under torch.compile. Skip it; non-ema still gets EMA.
    if cfg.ckpt_name.endswith("-ema.ckpt"):
        logger.info(f"'{cfg.ckpt_name}' is an EMA checkpoint; weights are already EMA, skipping EMA re-application.")
        ema_callback = None
    else:
        # The checkpoint is typically warm in OS page cache here, so this second torch.load is fast.
        ema_callback = _load_ema_from_checkpoint(ckpt_file)

    # Remove val sets used in training, and do not compute loss on inference examples, since they start with NaN coords
    # Resolve cfg.generation.dataset here, while it still has access to the full cfg context.
    resolved_dataset = OmegaConf.to_container(cfg.generation.dataset, resolve=True, throw_on_missing=False)
    model.cfg_exp.datasets.val = OmegaConf.create(
        {"inference_dataset": {"dataset": resolved_dataset, "compute_loss": False}}
    )

    # Set inference variables and potentially load autoguidance
    nn_ag = load_ag_ckpt(cfg.generation.args)

    metrics = cfg.get("metrics")

    model.configure_inference(
        cfg.generation,
        nn_ag=nn_ag,
        store_dir=cfg.get("out_dir"),
        metrics_cfg=metrics,
        is_inference=True,
        global_prefix=cfg.get("global_prefix"),
        job_id=cfg.get("job_id"),
        ckpt_path=ckpt_path,
        ckpt_name=cfg.ckpt_name,
        save_chain_of_interest=cfg.get("save_chain_of_interest", False),
    )

    # Configure search if specified
    if hasattr(cfg, "search") and cfg.search is not None:
        search_strategy = configure_search_strategy_from_cfg(cfg.search)

        # Build search reward model from search.rewards config.
        # Metrics come from the model's configured managers (design, oracle, refolding,
        # pass_criteria); reward_metrics_tags controls which fire during scoring.
        search_reward_model = None
        if hasattr(cfg.search, "rewards") and cfg.search.rewards is not None:
            from rfproteina.rewards import RewardModel

            rewards_cfg = cfg.search.rewards
            clip_values_cfg = rewards_cfg.get("clip_values")
            overrides_cfg = rewards_cfg.get("checkpoint_overrides")
            tags_cfg = rewards_cfg.get("reward_metrics_tags")
            search_reward_model = RewardModel(
                weights=dict(rewards_cfg.weights),
                reward_metrics_tags=OmegaConf.to_container(tags_cfg, resolve=True)
                if OmegaConf.is_config(tags_cfg)
                else tags_cfg,
                clip_values=OmegaConf.to_container(clip_values_cfg, resolve=True) if clip_values_cfg else None,
                checkpoint_overrides=OmegaConf.to_container(overrides_cfg, resolve=True) if overrides_cfg else None,
            )
            logger.info(
                f"Search reward model: reward_metrics_tags={search_reward_model.reward_metrics_tags}, "
                f"weights={dict(search_reward_model.weights)}, clip_values={search_reward_model.clip_values}"
            )

        model.configure_search(search_strategy, search_reward_model)

    # Configure differentiable-metric gradient guidance if specified
    if hasattr(cfg, "guidance") and cfg.guidance is not None:
        guidance_metric_manager = MetricManager.instantiate_from_hydra(cfg.guidance.metrics)
        guidance_terms = build_guidance_terms(cfg.guidance.terms)
        guidance_model = GuidanceModel(
            guidance_metric_manager,
            guidance_terms,
            decode_fn=model.decode_for_guidance,
            insertion_fn=model.insertion_mapping_for_guidance,
        )
        model.configure_guidance(guidance_model)

    return model, ema_callback


def _get_example_ids_from_dataset(dataset: torch.utils.data.Dataset) -> list[str]:
    """Extract example IDs from a dataset without loading data.

    Args:
        dataset: A ConditionSpecDataset or DesignProblemDataset instance.

    Returns:
        List of example ID strings, one per dataset index.
    """
    if isinstance(dataset, ConditionSpecDataset):
        return dataset.names
    elif isinstance(dataset, DesignProblemDataset):
        ids = []
        for idx in range(len(dataset)):
            base_id = dataset.base_ids[idx]
            if dataset.num_replicates > 1:
                path = dataset.file_paths[idx]
                previous_occurrences = dataset.file_paths[:idx].count(path)
                ids.append(f"{base_id}_rep{previous_occurrences}")
            else:
                ids.append(base_id)
        return ids
    else:
        raise ValueError(f"Unexpected Dataset type: {type(dataset).__name__}")


def _filter_existing_outputs(
    dataset: torch.utils.data.Dataset,
    model: Proteina,
    root_path: str,
    job_id: int,
) -> torch.utils.data.Dataset:
    """Filter a dataset to exclude examples that already have output structure files.

    Args:
        dataset: The inference dataset.
        model: The loaded Proteina model (used to determine output paths).
        root_path: Root output path for this inference run.
        job_id: Job ID for parallel job splitting.

    Returns:
        A Subset of the original dataset excluding already-completed examples,
        or the original dataset if no outputs exist yet.
    """
    store_dir = model.store_dir
    output_dir = Path(store_dir) / "inference_samples"

    # Collect design IDs that already have outputs. Sibling jobs' designs land here too but
    # carry their own job suffix, so they never match.
    existing_ids = {p.name.removesuffix("_generated.cif") for p in output_dir.glob("*/*_generated.cif")}
    if not existing_ids:
        return dataset

    # Determine which indices to keep (compare against the on-disk ids validation_step assigns).
    all_ids = _get_example_ids_from_dataset(dataset)
    # proteina.py names outputs from model.job_id (set via cfg), not the run_generation job_id.
    job_suffix = f"_job{model.job_id}" if model.job_id is not None else ""
    keep_indices = [i for i, eid in enumerate(all_ids) if f"{eid}{job_suffix}" not in existing_ids]
    n_skipped = len(all_ids) - len(keep_indices)

    if n_skipped > 0:
        logger.info(
            f"skip_existing: Found {len(existing_ids)} existing outputs in {output_dir}. "
            f"Skipping {n_skipped}/{len(all_ids)} examples."
        )
        return torch.utils.data.Subset(dataset, keep_indices)

    return dataset


def _sort_dataset_by_length(dataset: torch.utils.data.Dataset) -> torch.utils.data.Dataset:
    """Group similar-length examples so each batch pads up to a smaller bucket.

    Lengths come from each spec's ``length`` field (upper bound of a range), so they are
    a hint: a no-op for datasets without specs, and never load-bearing for correctness.
    """
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = list(dataset.indices) if isinstance(dataset, Subset) else list(range(len(dataset)))

    specs = getattr(base, "data", None)
    if not isinstance(specs, dict) or not hasattr(base, "idx_to_id"):
        logger.info("Dataset exposes no length specs; skipping length sorting.")
        return dataset

    def approx_length(idx: int) -> int:
        bounds = str(specs[base.idx_to_id(idx)].get("length", "")).split("-")
        return max((int(b) for b in bounds if b.isdigit()), default=0)

    return Subset(base, sorted(indices, key=approx_length))


def run_generation(cfg: DictConfig, job_id: int = 0, config_name: str | None = None) -> None:
    """
    Core generation function that runs inference given a Hydra config.

    Args:
        cfg: Hydra DictConfig with all generation parameters
        job_id: Job ID for parallel job splitting
        config_name: Name of the config (used for output directory naming)
    """
    precision = cfg.precision
    torch.set_float32_matmul_precision("high" if precision in ("32", "32-true") else "medium")

    # Determine config name for output paths
    if config_name is None:
        config_name = cfg.get("run_name_", "inference")

    if not exists(cfg.get("out_dir", None)):
        cfg.out_dir = infer_root_path(create_root=True, config_name=config_name)

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
    csv_path = os.path.join(cfg.out_dir, "..", csv_filename)
    if os.path.exists(csv_path):
        logger.info(f"Results already exist at {csv_path}. Exiting generate.py.")
        sys.exit(0)

    # Load model and EMA callback
    model, ema_callback = load_ckpt_n_configure_inference(cfg)
    model.inference_seed = cfg.seed

    # Fully deterministic mode — MUST come after model init, because model.__init__
    # may call _configure_flash_attention_backends() which re-enables Flash SDP.
    if cfg.get("torch_deterministic", False):
        apply_torch_deterministic(logger)

    dataset = get_dataset_from_cfg(cfg, job_id)

    # Optionally skip examples that already have outputs on disk
    if cfg.get("skip_existing", False):
        dataset = _filter_existing_outputs(
            dataset,
            model=model,
            root_path=cfg.out_dir,
            job_id=job_id,
        )

    # Support either batch_size argument for backwards compatibility with rfd3 syntax
    batch_size = _get_mutually_exclusive_config(cfg, "batch_size", "diffusion_batch_size", default_value=1)

    # Pad every batch up to one of a few token counts so the network sees a handful of
    # static shapes and can be compiled (variable lengths otherwise recompile forever).
    compile_lengths = cfg.get("compile_lengths", None)
    fixed_dimensions = None
    if compile_lengths:
        compile_lengths = sorted(int(n) for n in compile_lengths)
        assert all(
            n % 16 == 0 for n in compile_lengths
        ), f"compile_lengths must be 16-aligned for torch.compile, got {compile_lengths}"
        fixed_dimensions = {"I": compile_lengths}
        dataset = _sort_dataset_by_length(dataset)
        # Match training so inference gets the same coordinate-descent-tuned kernels.
        torch._inductor.config.coordinate_descent_tuning = True
        # Inference compiles different graphs than training (no backward, different
        # shapes), so keep them in a sibling blob: saving here must never overwrite
        # the training cache that $TORCH_COMPILE_CACHE_BLOB points at.
        blob = os.environ.get("TORCH_COMPILE_CACHE_BLOB")
        if blob:
            os.environ["TORCH_COMPILE_CACHE_BLOB"] = f"{blob}.inference"
        load_compile_cache()
        model.nn.enable_inference_compile(compile_lengths)
        logger.info(f"Inference torch.compile enabled with length buckets {compile_lengths}")

    # Create dataloader
    num_workers = cfg.num_workers
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_batch,
            schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
            fixed_dimensions=fixed_dimensions,
        ),
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
        default_root_dir=cfg.out_dir,
        logger=False,
        callbacks=[ema_callback] if ema_callback is not None else [],
        precision=precision,
    )
    trainer.validate(model, dataloader)  # validation_step automatically saves outputs

    # Persist the graphs compiled for each bucket so the next run starts warm
    if compile_lengths and trainer.is_global_zero:
        save_compile_cache()

    logger.info(f"✓ Generation complete! Results saved to {cfg.out_dir}")


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

    from rfproteina.utils.logging import silence_warnings

    silence_warnings()

    job_id = cfg.get("job_id", 0)
    if not exists(job_id):
        job_id = 0

    if exists(cfg.get("data_path", None)):
        os.environ["DATA_PATH"] = cfg.data_path

    logger.info("Starting inference with config:")
    print_config_tree(cfg, resolve=True)

    run_generation(cfg, job_id=job_id)


if __name__ == "__main__":
    main_hydra()
