"""Full BioIR computation with native RF input/output, rewards and metrics."""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from bionemo_ir.models.rfproteina import RFProteina
    from bionemo_ir.models.rfproteina.config import (
        RFProteinaBeamSearchConfig,
        RFProteinaPartialSimulationConfig,
        RFProteinaSamplingConfig,
    )


def _flow_state_with_merged_lora(denoiser: torch.nn.Module) -> dict[str, torch.Tensor]:
    """The denoiser's state with every PEFT LoRA adapter folded into its base weight, on CPU.

    BioIR converts plain ``weight``/``bias`` entries, so an adapted layer becomes ``W + scaling * B @ A``
    under its original name, which is the function the adapter computes at inference.
    """
    state = {key: value.detach().to("cpu", copy=True) for key, value in denoiser.state_dict().items()}
    for name, module in denoiser.named_modules():
        if not (hasattr(module, "base_layer") and hasattr(module, "lora_A")):
            continue
        adapters = list(module.lora_A)
        if adapters != ["default"]:
            raise ValueError(f"BioIR merges one LoRA adapter named 'default'; {name!r} has {adapters}")
        base = module.base_layer
        delta = module.get_delta_weight("default").detach().to("cpu", torch.float32)
        stale = (f"{name}.lora_", f"{name}.base_layer.")
        for key in [key for key in state if key.startswith(stale)]:
            del state[key]
        state[f"{name}.weight"] = (base.weight.detach().to("cpu", torch.float32) + delta).to(base.weight.dtype)
        if base.bias is not None:
            state[f"{name}.bias"] = base.bias.detach().to("cpu", copy=True)
    leftover = [key for key in state if ".lora_" in key]
    if leftover:
        raise ValueError(f"LoRA weights outside an adapted layer cannot be merged for BioIR: {leftover[:3]}")
    return state


def _copy_loaded_states(native: Any) -> dict[str, dict[str, torch.Tensor]]:
    """Copy the weights selected by native loading, including callback EMA and merged LoRA adapters."""
    return {
        "flow": {f"nn.{key}": value for key, value in _flow_state_with_merged_lora(native.nn).items()},
        "autoencoder": {
            key: value.detach().to("cpu", copy=True) for key, value in native.autoencoder.state_dict().items()
        },
    }


#: Model entry points the full workflow calls on the installed BioIR runtime.
REQUIRED_MODEL_API = ("from_reference_state_dicts", "configure_graphs", "graph_status", "reset_graphs")


def require_bioir(native: Any = None) -> None:
    """Reject an installed BioIR that cannot run the whole workflow, naming what to install.

    Given the native model, also probe that the runtime converts its checkpoint's vocabulary,
    recycle conditioning, and noise alignment.
    """
    try:
        from bionemo_ir._torch.sampling import random as bioir_random
        from bionemo_ir.models.rfproteina import RFProteina
        from bionemo_ir.models.rfproteina.config import (
            RFProteinaDecoderConfig,
            RFProteinaRepresentationConfig,
            RFProteinaSamplingConfig,
            RFProteinaScheduleConfig,
            RFProteinaVFStepConfig,
        )
    except (ImportError, OSError) as error:
        raise ImportError(
            "inference_backend=bioir runs the whole workflow through BioIR's RFProteína model, which is not "
            "available in this environment. Install a bionemo-ir runtime that provides it, or select "
            "inference_backend=native."
        ) from error
    missing = [name for name in REQUIRED_MODEL_API if not callable(getattr(RFProteina, name, None))]
    if "max_tokens" not in inspect.signature(RFProteina.configure_graphs).parameters:
        missing.append("configure_graphs(max_tokens=)")
    # Older runtimes silently drop config fields they do not know, so check the fields exist.
    # ``model_copy(update=)`` does not validate: an unknown key lands in ``__dict__`` and reads
    # back through ``getattr``, so only ``model_fields`` distinguishes a runtime that declares
    # the field from one that will never consult it.
    for owner, field in (
        (RFProteinaScheduleConfig, "tail_steps"),
        (RFProteinaScheduleConfig, "tail_step_multiplier"),
        (RFProteinaSamplingConfig, "condition_decoder_on_sequence_prediction"),
        (RFProteinaSamplingConfig, "match_reference_rng"),
        (RFProteinaVFStepConfig, "tsr_scaled_score_drift"),
    ):
        if field not in owner.model_fields:
            missing.append(f"{owner.__name__}.{field}")
    # The stream seams are imported lazily on the first batch, after conversion and graph capture.
    for name in ("normal_draws", "permutations"):
        if not callable(getattr(bioir_random, name, None)):
            missing.append(f"bionemo_ir._torch.sampling.random.{name}")
    if missing:
        raise RuntimeError(
            f"The installed BioIR RFProteína is missing {missing}; install the pinned BioIR runtime, "
            "or select inference_backend=native."
        )
    residue_types = _checkpoint_residue_types(native)
    if residue_types is not None:
        # Ask the runtime's own validator rather than keeping a list of vocabularies it supports.
        try:
            RFProteinaDecoderConfig(num_residue_types=residue_types)
        except ValueError as error:
            raise RuntimeError(
                f"The installed BioIR RFProteína cannot convert this checkpoint's {residue_types}-class "
                "residue vocabulary; install the pinned BioIR runtime, or select inference_backend=native."
            ) from error
    # An undeclared field is ignored rather than rejected, so declaring it is the capability.
    if _recycle_wider_than_depth(native) and "recycle_conditioning" not in RFProteinaRepresentationConfig.model_fields:
        raise RuntimeError(
            "The installed BioIR RFProteína cannot convert this checkpoint's recycle conditioning, which is "
            "wider than its gradient depth; install the pinned BioIR runtime, or select inference_backend=native."
        )
    alignment = _noise_alignment(native)
    if alignment["noise_align_internal_coords"] or alignment["noise_align_internal_coords_translate"]:
        raise RuntimeError(
            "BioIR does not align internal-coordinate conditioning to the noisy state, which this checkpoint "
            "was trained with; select inference_backend=native."
        )
    if (
        alignment["noise_align_chirality"] or alignment["noise_align_chirality_translate"]
    ) and "noise_align_chirality" not in RFProteinaRepresentationConfig.model_fields:
        raise RuntimeError(
            "The installed BioIR RFProteína cannot align chirality to the noisy state, which this checkpoint was "
            "trained with; install the pinned BioIR runtime, or select inference_backend=native."
        )


def _noise_alignment(native: Any) -> dict[str, bool]:
    """Which conditioning the checkpoint was trained to see in the noisy state's frame.

    A transformation with no parameters leaves nothing in the weights, so the checkpoint's config is
    the only record.
    """
    cfg_exp = getattr(native, "cfg_exp", None)
    nn = {} if cfg_exp is None else cfg_exp.nn
    return {
        key: bool(nn.get(key, False))
        for key in (
            "noise_align_chirality",
            "noise_align_chirality_translate",
            "noise_align_internal_coords",
            "noise_align_internal_coords_translate",
        )
    }


def _recycle_wider_than_depth(native: Any) -> bool:
    """Whether the checkpoint conditions on recycling more widely than it backpropagates through it."""
    cfg_exp = getattr(native, "cfg_exp", None)
    if cfg_exp is None:
        return False
    recycle = cfg_exp.nn.get("recycle", {}) or {}
    width = recycle.get("conditioning_dim")
    depth = {"last": 1, "last2": 2}.get(recycle.get("grad", "last"))
    return width is not None and depth is not None and width != depth


def _checkpoint_residue_types(native: Any) -> int | None:
    """The autoencoder's residue vocabulary size, or None when the model carries no autoencoder."""
    autoencoder = getattr(native, "autoencoder", None)
    if autoencoder is None:
        return None
    return int(autoencoder.cfg_ae.nn_ae.get("n_residue_types", 33))


def load_full_model(
    native: Any, *, sampling_config: Any, device: torch.device, options: Mapping[str, Any]
) -> RFProteina:
    """Strictly convert active native weights without opening checkpoints again."""
    require_bioir(native)  # first, so a missing runtime reports what to install
    from bionemo_ir.models.rfproteina import RFProteina

    if options.get("dtype", "bfloat16") != "bfloat16":
        raise ValueError("Full BioIR currently requires bioir.dtype=bfloat16")
    states = _copy_loaded_states(native)
    alignment = _noise_alignment(native)
    return RFProteina.from_reference_state_dicts(
        states["flow"],
        states["autoencoder"],
        sampling_config=sampling_config,
        device=device,
        attention_backend=options.get("attention_backend"),
        # Only a checkpoint trained with it names it, so a runtime without the option still loads the rest.
        **{key: True for key in ("noise_align_chirality", "noise_align_chirality_translate") if alignment[key]},
    )


def sampling_from_native(native: Any) -> RFProteinaSamplingConfig:
    """Copy resolved inference policy, including checkpoint reference-noise laws."""
    from bionemo_ir.models.rfproteina.config import (
        RFProteinaModalitySamplingConfig,
        RFProteinaSamplingConfig,
    )

    args = native.inf_cfg.args
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
        # Native's stochastic VF-TSR update drives the Langevin term with the TSR-scaled score.
        if step["sampling_mode"] == "vf_tsr":
            values["step"]["tsr_scaled_score_drift"] = True
        modalities[field] = RFProteinaModalitySamplingConfig(**values)
    nsteps = int(args.nsteps)
    bands = [(len(list(depths)), int(depth)) for depth, depths in groupby(native.recycle_depths(nsteps))]
    config = RFProteinaSamplingConfig(
        num_steps=nsteps,
        n_recycle_schedule=bands,
        state_dtype="float32",
        self_condition=False,
        condition_decoder_on_sequence_prediction=bool(args.get("condition_decoder_on_sequence_prediction", False)),
        **modalities,
    )
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
    decoder_presence: list[torch.Tensor] | None = None,
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
    if decoder_presence is not None:
        from rfproteina.nn.bioir_decoder import constrained_sample

        sample = constrained_sample(native, prediction, features, decoder_presence)
    results = native.format_sample_prots_to_atom_arrays(
        sample, batch, example_ids=example_ids, insertion_logits=prediction.insertion_logits
    )
    if len(results) != len(coords):
        raise RuntimeError("Native structure formatting dropped BioIR output rows")
    for result, index in zip(results, features["_native_row_index"].tolist(), strict=True):
        result["orig_batch_idx"] = index
    return results


class NativeMetricEvaluator:
    """Score BioIR candidates with the native metric and oracle managers, as native beam search does.

    BioIR ranks candidates itself from the reward weights; this only supplies the raw metrics the
    checkpoint lists under ``metrics``. Oracle/refolding results are cached under ``output_dir``.
    """

    def __init__(self, native: Any, batch: Mapping[str, Any], output_dir: Path):
        self.native, self.batch, self.output_dir = native, batch, output_dir

    def evaluate(self, prediction: Any, features: Mapping[str, torch.Tensor], context: Any) -> list[dict[str, Any]]:
        from rfproteina.metrics.utils.metrics_utils import run_metrics_for_result, run_oracle_and_refolding_metrics
        from rfproteina.transforms.unindexing import annotate_unindexed_insertion

        strategy = self.native.search_strategy
        names = list(strategy.checkpoints[context.checkpoint].get("metrics", []))
        reward_model = strategy.reward_models[context.checkpoint]
        batch = _rows(self.batch, features["_native_row_index"])
        entries = format_prediction(self.native, prediction, features, self.batch, list(context.request_ids))
        metrics = []
        for index, entry in enumerate(entries):
            array = entry.get("full_array_with_unindexed") or entry["generated_atom_array"]
            entry["generated_atom_array"] = annotate_unindexed_insertion(
                array, entry.get("predicted_insertion_mapping") or {}
            )
            example_id = entry.get("example_id") or context.request_ids[index]
            row = {}
            if self.native.design_metric_manager is not None:
                row = run_metrics_for_result(
                    entry,
                    batch,
                    index,
                    self.native.design_metric_manager,
                    output_base_dir=self.output_dir,
                    example_id=example_id,
                    metric_names=names,
                )
            row.update(
                run_oracle_and_refolding_metrics(
                    atom_array=entry["generated_atom_array"],
                    example_id=example_id,
                    output_dir=self.output_dir,
                    oracle_mm=self.native.oracle_metric_manager,
                    refold_mm=self.native.refolding_metric_manager,
                    metric_names=names,
                )
            )
            reward_model.aggregate_metrics_output(row)  # warns on a missing reward key, as native search does
            metrics.append(row)
        return metrics


@contextmanager
def native_random_streams(batch: Mapping[str, torch.Tensor]) -> Iterator[None]:
    """Continue post-featurization row streams, never reseed at this seam."""
    from bionemo_ir._torch.sampling.random import normal_draws, permutations

    device = batch["padding_mask"].device
    initial = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
    generator = torch.Generator(device=device)
    generator.set_state(initial)

    def draw(shape: Sequence[int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    def permute(n: int, *, device: torch.device) -> torch.Tensor:
        return torch.randperm(n, device=device, generator=generator)

    # Commit after BioIR's internal fork exits; never reseed or discard the draws
    # already consumed while featurizing the input. Coordinate-group permutation
    # shares the stream: native takes it from the same generator as its noise and
    # permutes once per trunk pass, so a stream carrying only the noise falls
    # behind at the first input that has a group to permute.
    try:
        with normal_draws(draw), permutations(permute):
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


def search_from_native(native: Any) -> RFProteinaBeamSearchConfig | None:
    """Map native beam search onto BioIR's, rejecting options BioIR does not reproduce."""
    from bionemo_ir.models.rfproteina.config import RFProteinaBeamSearchConfig

    from rfproteina.search.beam_search import BeamSearchStrategy

    strategy = native.search_strategy
    if strategy is None:
        return None
    if not isinstance(strategy, BeamSearchStrategy):
        raise NotImplementedError("Full BioIR only bridges native beam search")
    policy = getattr(strategy, "continuation_policy", "stop")
    if policy != "stop":
        raise NotImplementedError(
            f"Full BioIR does not bridge search.continuation_policy={policy!r}; it would stop failing "
            "originals instead. Use inference_backend=native."
        )
    if strategy.save_states or strategy.resume_from is not None:
        raise NotImplementedError("Full BioIR search does not save or resume candidate states")
    if strategy.step_checkpoints[-1] != int(native.inf_cfg.args.nsteps):
        raise NotImplementedError("Full BioIR search needs its last checkpoint at the end of the schedule")
    for step, config in strategy.checkpoints.items():
        if (
            config.get("use_clean_prediction")
            or "continuation_criteria" in config
            or config.get("n_branch", strategy.n_branch) != strategy.n_branch
        ):
            raise NotImplementedError(
                f"Full BioIR search supports neither clean-prediction scoring, continuation criteria nor "
                f"per-checkpoint branching (checkpoint {step})"
            )
    threshold = None
    if criterion := strategy.intermediate_sample_save_criterion:
        thresholds = criterion.metric_thresholds
        if set(thresholds) != {"search.reward"} or set(thresholds["search.reward"]) != {"min"}:
            raise NotImplementedError("Full BioIR requires a scalar intermediate reward minimum")
        threshold = thresholds["search.reward"]["min"]
    # Native checkpoints each own a reward model; BioIR takes one base objective plus per-step replacements.
    models = strategy.reward_models
    base = models[strategy.step_checkpoints[1]]
    overrides = {
        step: {"weights": model.weights, "clip_values": model.clip_values}
        for step, model in models.items()
        if (model.weights, model.clip_values) != (base.weights, base.clip_values)
    }
    return RFProteinaBeamSearchConfig(
        step_checkpoints=list(strategy.step_checkpoints),
        n_branch=strategy.n_branch,
        beam_width=strategy.beam_width,
        intermediate_save_min_reward=threshold,
        rewards={"weights": base.weights, "clip_values": base.clip_values, "checkpoint_overrides": overrides},
    )


def _fresh_seed() -> int:
    """A seed drawn from the global generator: reproducible under the run seed, different per batch."""
    return int(torch.randint(0, 2**31 - 1, (1,)).item())


class FullBioIRAdapter:
    """Replace generation computation while retaining native service boundaries."""

    def __init__(self, native: Any, model: Any, options: Mapping[str, Any], device: torch.device):
        self.native, self.model, self.options, self.device = weakref.proxy(native), model, dict(options), device
        self.completed_calls = 0
        # The inference config is fixed for a run, so translate it once.
        self.sampling = sampling_from_native(native).model_copy(
            update={"match_reference_rng": bool(self.options.get("match_reference_rng", False))}
        )
        self.partial = partial_from_native(native)
        self.search = search_from_native(native)
        model.config.sampling = model.sampler.config = self.sampling
        self.graph_capacity = 0
        self.configure_graphs()

    def predict_step(self, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
        features = _features(batch)
        if self.search is not None:
            # Search derives its own child streams from a root seed.
            output = self.model.search(
                features,
                self.search,
                seed=_fresh_seed(),
                partial_config=self.partial,
                evaluators=(NativeMetricEvaluator(self.native, batch, Path(self.native.store_dir) / "bioir_search"),),
            )
            results = []
            example_ids = batch.get("example_id_batch") or [f"sample{k}" for k in range(len(features["padding_mask"]))]
            for candidate in output.candidates:
                index = candidate.request_index
                one = {key: features[key][index : index + 1] for key in ("padding_mask", "_native_row_index")}
                # BioIR ids restart at req0 in every batch; prefix the input's own id so outputs never collide.
                candidate_id = f"{example_ids[index]}_{candidate.candidate_id}"
                entries = format_prediction(self.native, candidate.prediction, one, batch, [candidate_id])
                entries[0].update(
                    search_metrics=candidate.metrics,
                    reward=candidate.reward,
                    search_checkpoint=candidate.checkpoint,
                    search_seed=candidate.seed,
                    is_intermediate=candidate.is_intermediate,
                )
                results.extend(entries)
        else:
            decoder_presence = None
            with native_random_streams(features):
                if self.partial is not None:
                    # Redesign encodes the clean structure with its own generator: give it a seed so its noise
                    # is independent of the sampler's global-stream draws.
                    prediction = self.model.redesign(features, self.partial, seed=_fresh_seed())
                else:
                    from rfproteina.nn.bioir_decoder import capture_presence

                    # Match native AE output constraints using the decoder's actual
                    # predicted slots, not coordinates grafted from the source CIF.
                    with capture_presence(self.model.decoder) as decoder_presence:
                        prediction = self.model(features)
            results = format_prediction(self.native, prediction, features, batch, decoder_presence=decoder_presence)
        self.completed_calls += 1
        return results

    def configure_graphs(self) -> None:
        """Size the trunk's graph cache at the request boundary.

        Requests longer than ``bioir.cuda_graph_max_tokens`` run the trunk eagerly.
        """
        if not self.options.get("cuda_graph", True):
            return
        # BioIR captures the trunk's recycle step, so a slot covers one ``(batch, tokens)`` shape
        # whatever depths the schedule asks for: a request needs its full batch and the singleton
        # tail a replica count can leave. Serving several lengths at once wants more.
        required = 2
        requested = self.options.get("max_graphs")
        if requested is not None and requested < required:
            raise ValueError(
                f"Full BioIR needs at least {required} graph slots, for a full batch and its "
                "singleton tail; raise bioir.max_graphs or leave it null for automatic sizing"
            )
        capacity = required if requested is None else requested
        if capacity <= self.graph_capacity:
            return
        max_tokens = self.options.get("cuda_graph_max_tokens")
        # Capture verification replays each new graph against an extra eager pass to compare them.
        # That checks the capture machinery, which the test suite covers; here it only adds
        # 0.5-0.9 s to the request that first meets a shape.
        self.model.configure_graphs(capacity=capacity, verify_capture=False, max_tokens=max_tokens)
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
    adapter = FullBioIRAdapter(native, model, options, device)
    # Metadata/formatting stays native. Only the converted model retains CUDA weights.
    native.nn.cpu()
    native.autoencoder.cpu()
    native._full_bioir = adapter  # Plain object: never registers converted weights or a parent cycle.
    return adapter


def release_full_bioir(native: Any) -> None:
    """Drop any BioIR selection and put the native weights back, so the model runs natively again."""
    adapter = getattr(native, "_full_bioir", None)
    native._full_bioir, native._full_bioir_options = None, None
    if adapter is not None:
        adapter.model.reset_graphs()
        device = adapter.device
        del adapter  # Free the converted GPU weights before the native ones return.
        native.nn.to(device)
        native.autoencoder.to(device)


def predict_full_bioir(native: Any, batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    adapter = initialize_full_bioir(native, batch["padding_mask"].device)
    if adapter is None:
        raise RuntimeError("Full BioIR was not selected")
    device = batch["padding_mask"].device
    if next(adapter.model.parameters()).device != device:
        raise RuntimeError("Full BioIR was initialized on a different device; reconfigure after moving the model")
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        return adapter.predict_step(batch)


def full_bioir_runtime(native: Any) -> dict[str, Any] | None:
    """Report the graph cache state and call count of the converted model."""
    adapter = getattr(native, "_full_bioir", None)
    if adapter is None:
        return None
    graphs = adapter.model.graph_status()
    return {
        "graph_capacity": graphs["capacity"],
        "graph_max_tokens": adapter.options.get("cuda_graph_max_tokens"),
        "graph_cache": {key: graphs[key] for key in ("active", "captured", "verified", "fallback")},
        "completed_calls": adapter.completed_calls,
    }
