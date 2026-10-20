"""Full BioIR computation with native RF input/output, rewards and metrics."""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import groupby
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from bionemo_ir.models.rfproteina import RFProteina
    from bionemo_ir.models.rfproteina.config import (
        RFProteinaBeamSearchConfig,
        RFProteinaPartialSimulationConfig,
        RFProteinaSamplingConfig,
    )


def _copy_loaded_states(native: Any) -> dict[str, dict[str, torch.Tensor]]:
    """Copy the weights selected by native loading, including callback EMA."""
    return {
        "flow": {f"nn.{key}": value.detach().to("cpu", copy=True) for key, value in native.nn.state_dict().items()},
        "autoencoder": {
            key: value.detach().to("cpu", copy=True) for key, value in native.autoencoder.state_dict().items()
        },
    }


#: Model entry points the full workflow calls on the installed BioIR runtime.
REQUIRED_MODEL_API = ("from_reference_state_dicts", "configure_graphs", "graph_status", "reset_graphs")


def require_bioir() -> None:
    """Reject an installed BioIR that cannot run the whole workflow, naming what to install."""
    try:
        from bionemo_ir.models.rfproteina import RFProteina
    except (ImportError, OSError) as error:
        raise ImportError(
            "execution.inference.*=bioir requires a BioIR runtime with the full RFProteína model. "
            "PyPI bionemo-ir 0.1.1 does not contain it. Install a compatible runtime when available, "
            "or select inference_backend=torch_compile or eager."
        ) from error
    missing = sorted(name for name in REQUIRED_MODEL_API if not callable(getattr(RFProteina, name, None)))
    if missing:
        raise RuntimeError(
            f"The installed BioIR RFProteína is missing {missing}; install a compatible BioIR runtime, "
            "or select inference_backend=torch_compile or eager."
        )


def load_full_model(
    native: Any, *, sampling_config: Any, device: torch.device, options: Mapping[str, Any]
) -> RFProteina:
    """Strictly convert active native weights without opening checkpoints again."""
    from bionemo_ir.models.rfproteina import RFProteina

    require_bioir()
    if options.get("dtype", "bfloat16") != "bfloat16":
        raise ValueError("Full BioIR currently requires execution.bioir.dtype=bfloat16")
    states = _copy_loaded_states(native)
    return RFProteina.from_reference_state_dicts(
        states["flow"],
        states["autoencoder"],
        sampling_config=sampling_config,
        device=device,
        attention_backend=options.get("attention_backend"),
    )


def sampling_from_native(native: Any) -> RFProteinaSamplingConfig:
    """Copy resolved inference policy, including checkpoint reference-noise laws."""
    from bionemo_ir.models.rfproteina.config import (
        RFProteinaModalitySamplingConfig,
        RFProteinaSamplingConfig,
    )

    args = native.inf_cfg.args
    if args.get("guidance_w", 1.0) != 1.0 or getattr(native, "guidance_model", None) is not None:
        raise NotImplementedError("Full BioIR gradient/CFG guidance is not qualified")
    if args.get("save_trajectory_every", 0) or args.get("dump_noisy_trajectories", False):
        raise NotImplementedError("Native trajectory export is not yet bridged")
    modalities = {}
    for mode, field in (("bb_ca", "backbone"), ("local_latents", "local_latents")):
        source = native.inf_cfg.model[mode]
        step, schedule, noise = source["simulation_step_params"], source["schedule"], source["gt"]
        reference = native.cfg_exp.product_flowmatcher[mode]
        values = {
            "schedule": {
                "mode": schedule["mode"],
                "power": schedule["p"],
                "tail_steps": schedule.get("tail_steps", 0),
                "tail_step_multiplier": schedule.get("tail_step_multiplier", 1),
            },
            "noise_schedule": {"mode": noise["mode"], "power": noise["p"], "clamp": noise.get("clamp_val")},
            "step": {
                "sampling_mode": step["sampling_mode"],
                "score_scale": step["sc_scale_score"],
                "noise_scale": step["sc_scale_noise"],
                "ode_upper": step["t_lim_ode"],
                "ode_lower": step["t_lim_ode_below"],
                "center_every_step": step["center_every_step"],
                "tsr_k": step["tsr_k"],
                "tsr_sigma": step["tsr_sigma"],
            },
            "reference": {
                key: reference[key]
                for key in (
                    "zero_com_noise",
                    "stochastic_centering_mode",
                    "stochastic_centering_scale",
                    "stochastic_centering_scope",
                    "stochastic_centering_group_keys",
                )
                if key in reference
            },
        }
        modalities[field] = RFProteinaModalitySamplingConfig(**values)
        # Older Pydantic models ignore unknown keys: never mistake that for parity.
        for key in ("tail_steps", "tail_step_multiplier"):
            if getattr(modalities[field].schedule, key, None) != values["schedule"][key]:
                raise RuntimeError("BioIR runtime lacks native tail-grid compatibility")
    nsteps = int(args.nsteps)
    bands = [(len(list(depths)), int(depth)) for depth, depths in groupby(native.recycle_depths(nsteps))]
    condition_sequence = bool(args.get("condition_decoder_on_sequence_prediction", False))
    config = RFProteinaSamplingConfig(
        num_steps=nsteps,
        n_recycle_schedule=bands,
        state_dtype="float32",
        self_condition=False,
        condition_decoder_on_sequence_prediction=condition_sequence,
        **modalities,
    )
    if getattr(config, "condition_decoder_on_sequence_prediction", None) != condition_sequence:
        raise RuntimeError("BioIR runtime lacks native decoder sequence-conditioning compatibility")
    return config


def _features(batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    from rfproteina.model_masks import compute_ae_masks, compute_flow_emit_masks

    features = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    attention, emit = compute_ae_masks(dict(batch))
    features["coords_nm"] = batch["coords"] / 10.0
    features["can_be_encoded"] = emit.clone()
    features["ae_attention_mask"] = attention.clone()
    features["flow_backbone_mask"] = compute_flow_emit_masks(dict(batch))["bb_ca"].clone()
    features["_native_row_index"] = torch.arange(len(emit), device=emit.device)
    return features


def _batch_aliases(batch: Any) -> tuple[str, ...]:
    """Alias keys a collated batch resolves but never enumerates from ``items()``."""
    mappings = getattr(batch, "_build_key_mappings", None)
    return tuple(mappings()) if callable(mappings) else ()


def _rows(batch: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    count = batch["padding_mask"].shape[0]
    ids = indices.tolist()
    rows = {
        key: (
            value.index_select(0, indices.to(value.device))
            if torch.is_tensor(value) and value.ndim and value.shape[0] == count
            else [value[index] for index in ids]
            if isinstance(value, list) and len(value) == count
            else value
        )
        for key, value in batch.items()
    }
    # A collated batch resolves ``<key>_batch`` aliases that ``items()`` never
    # enumerates, and structure export reads them to tell a batch of structures
    # from a single one. Selecting rows into a plain mapping drops the
    # resolution, so carry each alias the source batch answers.
    for alias in _batch_aliases(batch):
        if alias in rows:
            continue
        try:
            value = batch[alias]
        except (KeyError, AttributeError, TypeError):
            continue
        if isinstance(value, list) and len(value) == count:
            rows[alias] = [value[index] for index in ids]
    return rows


def format_prediction(
    native: Any,
    prediction: Any,
    features: Mapping[str, torch.Tensor],
    source_batch: Mapping[str, Any],
    example_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Keep native insertion/chemical output assembly; do not decode a second time."""
    batch = _rows(source_batch, features["_native_row_index"])
    # Use raw decoder tensors so native formatting alone applies insertion and
    # conditioning. BioIR's final_* properties already materialize insertion.
    coords = prediction.coordinates
    if prediction.partial_redesign is not None:
        coords = prediction.partial_redesign.coordinates
    residue_type = prediction.residue_types
    residue_type = torch.where(residue_type >= 0, residue_type + prediction.residue_type_offset, residue_type)
    sample = {
        "coors": coords.float() * 10.0,
        "residue_type": residue_type,
        "padding_mask": features["padding_mask"],
        "atom_mask": prediction.atom_mask,
    }
    results = native.format_sample_prots_to_atom_arrays(
        sample, batch, example_ids=example_ids, insertion_logits=prediction.insertion_logits
    )
    if len(results) != len(coords):
        raise RuntimeError("Native structure formatting dropped BioIR output rows")
    for result, index in zip(results, features["_native_row_index"].tolist(), strict=True):
        result["orig_batch_idx"] = index
    return results


class NativeMetricEvaluator:
    """Use the existing full-structure native metric/oracle managers during search."""

    def __init__(self, native: Any, batch: Mapping[str, Any]):
        self.native, self.batch = native, batch

    def evaluate(self, prediction: Any, features: Mapping[str, torch.Tensor], context: Any) -> list[dict[str, Any]]:
        from rfproteina.metrics.utils.metrics_utils import (
            run_metrics_for_result,
            run_oracle_refolding_and_pass_criteria,
        )
        from rfproteina.transforms.unindexing import annotate_unindexed_insertion

        batch = _rows(self.batch, features["_native_row_index"])
        entries = format_prediction(self.native, prediction, features, self.batch, list(context.request_ids))
        metrics = []
        for index, entry in enumerate(entries):
            source = entry.get("full_array_with_unindexed")
            if source is None:
                source = entry["generated_atom_array"]
            entry["generated_atom_array"] = annotate_unindexed_insertion(
                source, entry.get("predicted_insertion_mapping") or {}
            )
            row = {}
            if self.native.design_metric_manager is not None:
                row = run_metrics_for_result(
                    entry,
                    batch,
                    index,
                    self.native.design_metric_manager,
                    output_base_dir=None,
                    metrics_tags_override=list(context.tags),
                )
            row.update(
                run_oracle_refolding_and_pass_criteria(
                    atom_array=entry["generated_atom_array"],
                    example_id=entry["example_id"],
                    output_dir=None,
                    oracle_mm=self.native.oracle_metric_manager,
                    refold_mm=self.native.refolding_metric_manager,
                    pass_mm=self.native.pass_criteria_metric_manager,
                    metrics_tags=list(context.tags),
                    prior_metrics=row,
                )
            )
            # Preserve native/custom reward validation (including NIM's finite
            # weighted-value/float32-overflow checks) before BioIR ranks the row.
            reward = self.native.search_reward_model
            if context.checkpoint is not None:
                reward = reward.at_step(context.checkpoint)
            reward.aggregate_metrics_output(row)
            metrics.append(row)
        return metrics


@contextmanager
def native_random_streams(batch: Mapping[str, torch.Tensor]) -> Iterator[None]:
    """Continue post-featurization row streams, never reseed at this seam."""
    from bionemo_ir._torch.sampling.random import normal_draws

    from rfproteina.utils.sample_rng import (
        sample_randn,
        sample_rng_active,
        sample_rng_states,
    )

    if sample_rng_active():
        if len(sample_rng_states()) != len(batch["padding_mask"]):
            raise ValueError("Native RNG row count does not match the BioIR batch")
        with normal_draws(sample_randn):
            yield
        return
    device = batch["padding_mask"].device
    initial = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
    generator = torch.Generator(device=device)
    generator.set_state(initial)

    def draw(shape: Sequence[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    # The CLI may use a native global batched stream, unlike NIM's independent
    # rows. Commit after BioIR's internal fork exits; never reseed or discard the
    # draws already consumed while featurizing the input.
    try:
        with normal_draws(draw):
            yield
    finally:
        final = generator.get_state()
        if device.type == "cuda":
            torch.cuda.set_rng_state(final, device)
        else:
            torch.set_rng_state(final)


def partial_from_native(native: Any) -> RFProteinaPartialSimulationConfig | None:
    from bionemo_ir.models.rfproteina.config import RFProteinaPartialSimulationConfig

    args = native.inf_cfg.args
    fields = (
        "partial_t_backbone",
        "partial_t_latent",
        "couple_partial_t",
        "partial_trajectory_fraction_steps",
        "inverse_folding",
        "inverse_folding_method",
    )
    values = {key: args[key] for key in fields if key in args and args[key] is not None}
    active = values.get("inverse_folding", False) or any(
        values.get(key) is not None
        for key in ("partial_t_backbone", "partial_t_latent", "partial_trajectory_fraction_steps")
    )
    return RFProteinaPartialSimulationConfig(**values) if active else None


def search_from_native(native: Any, *, scoring_chunk_size: int | None = None) -> RFProteinaBeamSearchConfig | None:
    from bionemo_ir.models.rfproteina.config import RFProteinaBeamSearchConfig

    strategy = native.search_strategy
    if strategy is None:
        return None
    from rfproteina.search.beam_search import BeamSearchStrategy

    if not isinstance(strategy, BeamSearchStrategy):
        raise NotImplementedError("Full BioIR only bridges native beam search")
    if scoring_chunk_size is None:
        scoring_chunk_size = getattr(strategy, "scoring_chunk_size", None)
    reward = native.search_reward_model
    if reward is None:
        raise ValueError("Native beam search has no reward model")
    criterion = strategy.intermediate_sample_save_criterion
    threshold = None
    if criterion is not None:
        thresholds = criterion.metric_thresholds
        if set(thresholds) != {"search.reward"} or set(thresholds["search.reward"]) != {"min"}:
            raise NotImplementedError("Full BioIR requires a scalar intermediate reward minimum")
        threshold = thresholds["search.reward"]["min"]
    return RFProteinaBeamSearchConfig(
        step_checkpoints=list(strategy.step_checkpoints),
        n_branch=strategy.n_branch,
        beam_width=strategy.beam_width,
        scoring_chunk_size=scoring_chunk_size,
        intermediate_save_min_reward=threshold,
        rewards={
            "weights": reward.weights,
            "clip_values": reward.clip_values,
            "reward_metrics_tags": reward.reward_metrics_tags,
            "checkpoint_overrides": reward.checkpoint_overrides,
        },
    )


class FullBioIRAdapter:
    """Replace generation computation while retaining native service boundaries."""

    def __init__(self, native: Any, model: Any):
        self.native, self.model = weakref.proxy(native), model
        self.options = {}
        self.graph_capacity = 0
        self.rng_mode = None
        self.completed_calls = 0
        self.last_policy = None
        self.used_native_row_rng = False

    def predict_step(
        self,
        batch: Mapping[str, Any],
        *,
        seed: int | None = None,
        partial_config: Any = None,
        search_config: Any = None,
    ) -> list[dict[str, Any]]:
        from rfproteina.utils.sample_rng import sample_rng_active

        config = sampling_from_native(self.native)
        self.configure_graphs(config)
        self.rng_mode = "independent_rows" if sample_rng_active() else "global_batched_stream"
        self.model.config.sampling = self.model.sampler.config = config
        self.last_policy = config.model_dump()
        features = _features(batch)
        if partial_config is None:
            partial_config = partial_from_native(self.native)
        if search_config is not None:
            self.rng_mode = "derived_search"
            # Search owns deterministic derived child streams; it must not borrow
            # the parent's unbranched row callback. This is not a promise of exact
            # native beam-search random-number identity.
            output = self.model.search(
                features,
                search_config,
                seed=seed,
                partial_config=partial_config,
                evaluators=(NativeMetricEvaluator(self.native, batch),),
            )
            results = []
            for candidate in output.candidates:
                index = candidate.request_index
                one = {key: value[index : index + 1] for key, value in features.items()}
                entries = format_prediction(self.native, candidate.prediction, one, batch, [candidate.candidate_id])
                entries[0].update(
                    search_metrics=candidate.metrics,
                    reward=candidate.reward,
                    search_checkpoint=candidate.checkpoint,
                    search_seed=candidate.seed,
                    is_intermediate=candidate.is_intermediate,
                )
                results.extend(entries)
            self.completed_calls += 1
            self.used_native_row_rng = False
            return results
        with native_random_streams(features):
            if partial_config is not None:
                prediction = self.model.redesign(features, partial_config, seed=seed)
            else:
                prediction = self.model(features, seed=seed)
        results = format_prediction(self.native, prediction, features, batch)
        self.completed_calls += 1
        self.used_native_row_rng = self.rng_mode == "independent_rows"
        return results

    def configure_graphs(self, sampling_config: RFProteinaSamplingConfig) -> None:
        """Size at the request boundary, retaining upstream routing/static keys."""
        if not self.options.get("cuda_graph", True):
            return
        required = 2 * len({depth for _, depth in sampling_config.n_recycle_schedule})
        requested = self.options.get("max_graphs")
        if requested is not None and requested < required:
            raise ValueError(
                f"Full BioIR needs at least {required} graph slots for this recycling schedule "
                "and a singleton tail; increase execution.bioir.max_graphs or leave it null for automatic sizing"
            )
        capacity = required if requested is None else requested
        if capacity <= self.graph_capacity:
            return
        self.model.configure_graphs(capacity=capacity, verify_capture=True)
        self.graph_capacity = capacity


def initialize_full_bioir(native: Any, device: str | torch.device | None = None) -> FullBioIRAdapter | None:
    """Initialize after final device placement; keep compute out of state_dict()."""
    options = getattr(native, "_full_bioir_options", None)
    if options is None:
        return None
    adapter = getattr(native, "_full_bioir", None)
    if adapter is not None:
        return adapter
    device = torch.device(native.device if device is None else device)
    if device.type != "cuda":
        raise ValueError("Initialize full BioIR after moving the native model to its final CUDA device")
    model = load_full_model(native, sampling_config=sampling_from_native(native), device=device, options=options)
    adapter = FullBioIRAdapter(native, model)
    adapter.options = dict(options)
    # Metadata/formatting stays native. Only the converted model retains CUDA weights.
    native.nn.cpu()
    native.autoencoder.cpu()
    native._full_bioir = adapter  # Plain object: never registers converted weights or a parent cycle.
    return adapter


def predict_full_bioir(native: Any, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    adapter = initialize_full_bioir(native, batch["padding_mask"].device)
    if adapter is None:
        raise RuntimeError("Full BioIR was not selected")
    device = batch["padding_mask"].device
    if next(adapter.model.parameters()).device != device:
        raise RuntimeError("Full BioIR was initialized on a different device; reconfigure after moving the model")
    search = search_from_native(native)
    seed = None
    if search is not None:
        with torch.cuda.device(device):
            seed = torch.cuda.initial_seed()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        return adapter.predict_step(batch, search_config=search, seed=seed)


def full_bioir_runtime(native: Any) -> dict[str, Any] | None:
    """Report cache state and kernel availability, never fabricated dispatch counts."""
    adapter = getattr(native, "_full_bioir", None)
    if adapter is None:
        return None
    from bionemo_ir._torch.layers.triangle_nodes import TriangleMultiplicationNode

    graphs = adapter.model.graph_status()
    nodes = [module for module in adapter.model.modules() if isinstance(module, TriangleMultiplicationNode)]
    return {
        "workflow": "full",
        "graph_scope": "velocity_model",
        "graph_capacity": graphs["capacity"],
        "graph_cache": {key: graphs[key] for key in ("active", "captured", "verified", "fallback")},
        "completed_calls": adapter.completed_calls,
        "rng_mode": adapter.rng_mode,
        "triangle_modules": len(nodes),
        "kf_available_modules": sum(bool(node.kf_available) for node in nodes),
        "precision": "FP32 features/AE heads and BF16 transformer weights/compute",
        "attention_backend": str(adapter.model.config.trunk.pairwise_attention_backend),
        "native_cuda_parameter_elements": sum(p.numel() for p in native.parameters() if p.is_cuda),
        "conversion_tensor_counts": {
            f"convert_rfproteina_{component}_torch": len(getattr(adapter.model, report).loaded_keys)
            for component, report in (
                ("trunk", "conversion_report"),
                ("representation_builder", "representation_conversion_report"),
                ("decoder", "decoder_conversion_report"),
                ("encoder", "encoder_conversion_report"),
            )
        },
    }
