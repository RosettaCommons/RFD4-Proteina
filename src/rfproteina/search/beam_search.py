"""Beam search strategy for protein design.

Beam search maintains a fixed-size set of top-N partial trajectories (the beam),
advancing them iteratively while pruning based on reward scores. This balances
exploration (branching) with exploitation (keeping only top candidates).

Algorithm:

1. Initialize beam with N noise samples at t=0

2. While t < 1:

   - For each state in beam, generate L branches by running K denoising steps
   - Roll out each branch to completion to compute rewards R(x, s)
   - Select top-N branches by reward to form new beam
   - Update timestep t += K

3. Return all samples that meet success criteria

Completed rollouts can be retained as designs. Selected nonfinal checkpoints
can instead score predicted clean structures, without retaining predictions
or calling refolding models.
"""

import copy
import math
import shutil
from pathlib import Path
from typing import Any

import torch

from rfproteina.flow_matching.product_space_flow_matcher import extract_groups
from rfproteina.metrics.oracles.pass_criterion_metric import PassCriterionMetric
from rfproteina.metrics.utils.metrics_utils import run_metrics_for_result, run_oracle_and_refolding_metrics
from rfproteina.model_masks import compute_flow_emit_masks
from rfproteina.rewards.reward_model import SEARCH_REWARD_KEY, RewardModel
from rfproteina.search.base_strategy import SearchStrategy
from rfproteina.search.checkpoints import parse_checkpoints
from rfproteina.search.vanilla import build_partial_schedule, resolve_partial_t
from rfproteina.transforms.unindexing import annotate_unindexed_insertion
from rfproteina.utils.common import design_output_dir
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.torch import concat_dict_tensors

logger = RankedLogger(__name__, rank_zero_only=True)


def _select_batch_examples(batch: dict, indices: list[int], batch_size: int) -> dict:
    """Select collated example axes, including schema aliases, without changing padding."""

    def select(value: Any) -> Any:
        if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size:
            return value[indices]
        if isinstance(value, dict):
            return {key: select(item) for key, item in value.items()}
        if isinstance(value, list | tuple) and len(value) == batch_size:
            return type(value)(value[index] for index in indices)
        return value

    fields = dict(batch.items())
    for key in (
        "example_id_batch",
        "dataset_name_batch",
        "atom_array_batch",
        "task_batch",
        "metrics_tags_batch",
        "specification_batch",
    ):
        if key in batch:
            fields[key] = batch[key]
    return {key: select(value) for key, value in fields.items()}


def _continuation_passes(criterion: PassCriterionMetric, metrics: list[dict], step: int) -> list[bool]:
    """Fail missing/nonfinite candidate values; reject metrics absent from the entire pool."""
    required = criterion.required_metric_keys
    absent = required - {key for candidate in metrics for key in candidate}
    if absent:
        raise ValueError(f"continuation_criteria at step {step} requests metrics never produced: {sorted(absent)}")
    return [
        all(key in candidate and candidate[key] is not None and math.isfinite(candidate[key]) for key in required)
        and bool(criterion.compute(candidate)["pass"])
        for candidate in metrics
    ]


class BeamSearchStrategy(SearchStrategy):
    """Branch and prune at self-contained scoring checkpoints.

    Args:
        checkpoints: Positive step keys with metrics, rewards, use_clean_prediction, and continuation_criteria.
            Zero is implicit. Without a checkpoint at the schedule end, survivors finish without selection.
        n_branch: Default children per state; checkpoints may override it for their incoming interval.
        beam_width: Candidates retained per input after pruning.
        save_rollouts: Keep all completed scoring rollouts as additional outputs.
        save_criteria: Optional raw-metric thresholds for saving completed rollouts.
        save_states: Save selected candidates at interior checkpoints for later branching.
        resume_from: Path to a saved candidate batch; start a new beam at its saved step.
        resume_seed: Seed for the new branches, independent of input preprocessing.
    """

    def __init__(
        self,
        checkpoints: dict[int, dict[str, Any]],
        n_branch: int = 4,
        beam_width: int = 4,
        save_rollouts: bool = False,
        save_criteria: dict[str, dict[str, Any]] | None = None,
        save_states: bool = False,
        resume_from: str | None = None,
        resume_seed: int | None = None,
    ) -> None:
        self.checkpoints = parse_checkpoints(checkpoints)
        self.step_checkpoints = [0, *self.checkpoints]
        for name, value in (("n_branch", n_branch), ("beam_width", beam_width)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(save_rollouts, bool):
            raise ValueError("save_rollouts must be a bool.")
        if save_criteria is not None and not save_rollouts:
            raise ValueError("save_criteria requires save_rollouts: true.")
        if (resume_from is None) != (resume_seed is None):
            raise ValueError("Set resume_from and resume_seed together.")
        if not isinstance(save_states, bool):
            raise ValueError("save_states must be a bool.")
        self.save_states = save_states
        self.resume_from = resume_from
        self.resume_seed = resume_seed
        self.n_branch = n_branch
        self.beam_width = beam_width
        self.intermediate_sample_save_criterion = PassCriterionMetric(save_criteria or {}) if save_rollouts else None
        self.clean_prediction_steps = frozenset(
            step for step, config in self.checkpoints.items() if config.get("use_clean_prediction", False)
        )
        self.reward_models = {step: RewardModel(config.get("rewards", {})) for step, config in self.checkpoints.items()}

        self.continuation_criteria = {}
        for step, config in self.checkpoints.items():
            if "continuation_criteria" in config:
                if not isinstance(config["continuation_criteria"], dict):
                    raise ValueError("continuation_criteria must map metric keys to threshold specifications.")
                self.continuation_criteria[step] = PassCriterionMetric(config["continuation_criteria"])

    @property
    def metric_names(self) -> set[str]:
        """Explicit search metrics to initialize for the validation epoch."""
        return {name for config in self.checkpoints.values() for name in config.get("metrics", [])}

    def configure_metrics(self, model: "Proteina") -> None:
        """Validate explicitly selected metrics without inferring dependencies."""
        providers = {}
        for attr in ("design_metric_manager", "oracle_metric_manager", "refolding_metric_manager"):
            manager = getattr(model, attr, None)
            for name, metric in manager.metrics.items() if manager is not None else []:
                if name in providers:
                    raise ValueError(f"Search metric name {name!r} is ambiguous across metrics groups.")
                providers[name] = (attr, metric)
        for step, config in self.checkpoints.items():
            names = set(config.get("metrics", []))
            if unknown := names - providers.keys():
                raise ValueError(f"Checkpoint {step} requests unknown search metrics: {sorted(unknown)}.")
            if any(isinstance(providers[name][1], PassCriterionMetric) for name in names):
                raise ValueError("PassCriterionMetric is not supported as a search metric; use raw metric outputs.")
            groups = {providers[name][0] for name in names}
            if step in self.clean_prediction_steps and groups - {"design_metric_manager"}:
                raise ValueError("A use_clean_prediction checkpoint can only select design metrics.")
            if "refolding_metric_manager" in groups and "oracle_metric_manager" not in groups:
                raise ValueError(f"Checkpoint {step} must explicitly list an oracle in metrics for refolding.")
            missing = {
                key
                for key, weight in self.reward_models[step].weights.items()
                if weight and key.split(".")[0] not in names
            }
            if missing:
                raise ValueError(f"Checkpoint {step} rewards {sorted(missing)} require explicitly listed metrics.")

    def generate(
        self,
        model: "Proteina",
        batch: dict,
    ) -> list[dict[str, Any]]:
        """Generate structures using beam search.

        Implements beam search with reward-guided pruning.

        Args:
            model: Proteina model instance
            batch: Input batch from dataloader

        Returns:
            List of dictionaries, each containing:
                - generated_atom_array: Final protein structure
                - reward: Reward score, if scored at a checkpoint
                - (optionally) intermediate_samples

        Raises:
            ValueError: If a checkpoint's reward or save criterion is unevaluable.
        """
        self.configure_metrics(model)

        logger.info(
            f"Running beam search with n_branch={self.n_branch}, "
            f"beam_width={self.beam_width}, checkpoints={self.step_checkpoints}"
        )

        # Preprocess batch from atom37 collate format
        batch = model.preprocess_batch_for_generation(batch)

        # Store batch in model for reward computation
        model._current_batch = batch

        # Extract parameters from model's inference config
        nsteps = model.inf_cfg.args.nsteps
        guidance_w = model.inf_cfg.args.get("guidance_w", 1.0)

        # Extract batch dimensions
        nsamples = batch["coords_nm"].shape[0]  # batch size
        original_batch_indices = list(range(nsamples))
        n = batch["coords_nm"].shape[1]  # number of tokens
        device = model.device

        padding_mask = batch["padding_mask"]
        if padding_mask.dtype != torch.bool or padding_mask.shape != (nsamples, n):
            raise ValueError(
                f"padding_mask must be bool with shape {(nsamples, n)}, got {padding_mask.dtype} "
                f"and {tuple(padding_mask.shape)}"
            )

        simulation_step_params = {
            data_mode: model.inf_cfg.model[data_mode]["simulation_step_params"] for data_mode in model.fm.data_modes
        }

        partial_t_dict = resolve_partial_t(model.inf_cfg.args, model.fm, model.inf_cfg.model)

        # partial_trajectory_fraction_steps affects the number of steps to actually run
        frac = model.inf_cfg.args.get("partial_trajectory_fraction_steps", None)
        actual_nsteps = round(frac * nsteps) if frac is not None else nsteps
        recycle_depths = model.recycle_depths(actual_nsteps, full_nsteps=nsteps)

        if self.step_checkpoints[-1] > actual_nsteps:
            frac_hint = f" (nsteps={nsteps} x fraction={frac} = {actual_nsteps})" if frac is not None else ""
            raise ValueError(
                f"Last checkpoint {self.step_checkpoints[-1]} exceeds actual_nsteps={actual_nsteps}{frac_hint}."
            )
        if actual_nsteps in self.continuation_criteria:
            raise ValueError("continuation_criteria must target checkpoints before the schedule end.")
        if actual_nsteps in self.clean_prediction_steps:
            raise ValueError(
                "A checkpoint at the schedule end cannot use_clean_prediction; it emits completed designs."
            )

        # Build schedule: partial (covering [partial_t, 1]) or full (covering [0, 1])
        if partial_t_dict:
            ts, gt = build_partial_schedule(
                partial_t_dict=partial_t_dict,
                nsteps=actual_nsteps,
                fm=model.fm,
                sampling_model_args=model.inf_cfg.model,
                full_nsteps=nsteps if frac is not None else None,
            )
        else:
            ts, gt = model.fm.sample_schedule(
                nsteps=actual_nsteps,
                sampling_model_args=model.inf_cfg.model,
            )

        start_step = 0

        if self.resume_from is not None:
            state = torch.load(self.resume_from, map_location=device, weights_only=True)
            if not isinstance(state, dict) or "step" not in state or "x" not in state:
                raise ValueError("Saved candidate must contain a step and modality tensors in x.")
            start_step, candidate = state["step"], state["x"]
            if type(start_step) is not int or not 0 <= start_step < actual_nsteps:
                raise ValueError(f"Saved candidate step must be an integer in [0, {actual_nsteps}).")
            if not isinstance(candidate, dict) or set(candidate) != set(model.fm.data_modes):
                raise ValueError(f"Saved candidate modalities must match {model.fm.data_modes}.")
            for key, value in candidate.items():
                expected_shape = (nsamples, n, model.fm.base_flow_matchers[key].dim)
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != expected_shape
                    or not value.is_floating_point()
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(
                        f"Saved candidate {key} must be a finite floating tensor of shape {expected_shape}."
                    )
            xt = {key: value.repeat(self.beam_width, 1, 1) for key, value in candidate.items()}
            torch.manual_seed(self.resume_seed)
            model.inference_seed = self.resume_seed
        else:
            # Initialize beam: nsamples * beam_width samples
            # Interleaved format: [sample0, sample1, sample2, sample0, sample1, sample2, ...]
            logger.info(
                f"Initializing beam: {nsamples} samples x {self.beam_width} width = "
                f"{nsamples * self.beam_width} total states"
            )
            tiled_padding_mask = padding_mask.repeat(self.beam_width, 1)
            beam_emit_mask_batch = {
                "supports_local_latent": batch["supports_local_latent"].repeat(self.beam_width, 1),
                "is_in_context_conditioning": batch["is_in_context_conditioning"].repeat(self.beam_width, 1),
                "padding_mask": tiled_padding_mask,
            }
            coord_cond_mask = batch.get("coord_cond_mask")
            tiled_coord_cond_mask = coord_cond_mask.repeat(self.beam_width, 1) if coord_cond_mask is not None else None
            tiled_groups = {
                key: value.repeat(self.beam_width, 1) if value is not None else None
                for key, value in extract_groups(batch).items()
            }

            x_0 = model.fm.sample_noise(
                n,
                shape=(nsamples * self.beam_width,),
                device=device,
                emit_masks=compute_flow_emit_masks(beam_emit_mask_batch),
                training=False,
                coord_cond_mask=tiled_coord_cond_mask,
                groups=tiled_groups,
            )

            if partial_t_dict:
                batch = model.add_clean_samples(batch)
                x_1 = batch["x_1"]  # {dm: [nsamples, n_tokens, ...]}
                x_1_tiled = {dm: torch.cat([x_1[dm]] * self.beam_width, dim=0) for dm in partial_t_dict}
                xt = {dm: (1.0 - pt_val) * x_0[dm] + pt_val * x_1_tiled[dm] for dm, pt_val in partial_t_dict.items()}
                logger.info(
                    f"Partial diffusion: bb_ca t={partial_t_dict.get('bb_ca', 0):.4f}, "
                    f"local_latents t={partial_t_dict.get('local_latents', 0):.4f}"
                )
            else:
                xt = x_0

            # Mask the initial beam state in its interleaved, beam-width-tiled layout.
            xt = model.fm.apply_flow_emit_masks(xt, beam_emit_mask_batch)

        all_intermediate_results: list[dict] = []

        batch_ids: list[str] = batch.get("example_id_batch", [f"sample{k}" for k in range(nsamples)])

        # Oracle results are written during search under the same design id proteina.py
        # will later assign the design CIF, so the epoch-end pass reuses them as-is.
        output_base_dir = getattr(model, "_current_output_base_dir", None)
        output_suffix: str = getattr(model, "_current_output_suffix", "")

        if self.resume_from is not None:
            batch_ids = [f"{name}_r{self.resume_seed}" for name in batch_ids]

        # Keep original checkpoint numbers and absolute schedule indices when continuing.
        end_step = start_step
        for i, checkpoint_step in enumerate(self.step_checkpoints[1:]):
            if checkpoint_step <= start_step:
                continue
            end_step = checkpoint_step
            n_branch = self.checkpoints[end_step].get("n_branch", self.n_branch)
            logger.info(f"Checkpoint {i + 1}/{len(self.step_checkpoints) - 1}: steps {start_step} → {end_step}")

            score_clean_prediction = end_step in self.clean_prediction_steps
            all_clean_predictions = []
            all_xt_branches = []
            all_insertion_logits_branches = []
            all_sequence_logits_branches = []

            # Process each replica separately due to batch size constraints
            for replica_idx in range(self.beam_width):
                # Extract samples for this replica (interleaved format)
                start = replica_idx * nsamples
                end = (replica_idx + 1) * nsamples
                replica_xt = {key: value[start:end] for key, value in xt.items()}
                ts_segment = {k: v[start_step:] for k, v in ts.items()}
                gt_segment = {k: v[start_step:] for k, v in gt.items()}

                # Generate n_branch branches from this replica
                for _ in range(n_branch):
                    branch_xt, clean_prediction, branch_extra_info = model.fm.partial_simulation(
                        batch=copy.deepcopy(batch),
                        x=replica_xt,
                        predict_for_sampling=model.predict_for_sampling,
                        recycle_depths=recycle_depths[start_step:end_step],
                        nsteps=end_step - start_step,
                        ts=ts_segment,
                        gt=gt_segment,
                        simulation_step_params=simulation_step_params,
                        device=model.device,
                        guidance_w=guidance_w,
                    )

                    all_xt_branches.append(branch_xt)
                    if score_clean_prediction:
                        all_clean_predictions.append(clean_prediction)
                    all_insertion_logits_branches.append(branch_extra_info.get("insertion_logits"))
                    all_sequence_logits_branches.append(branch_extra_info.get("sequence_logits"))

            # Concatenate all replicas and branches
            xt = concat_dict_tensors(all_xt_branches, dim=0)

            insertion_logits = None
            if all(logits is not None for logits in all_insertion_logits_branches):
                insertion_logits = torch.cat(all_insertion_logits_branches, dim=0)
            sequence_logits = None
            if all(logits is not None for logits in all_sequence_logits_branches):
                sequence_logits = torch.cat(all_sequence_logits_branches, dim=0)

            # After each checkpoint transition, evaluate rewards and select top samples.
            # Example ids for each of the n_branch * beam_width * nsamples evaluated
            # samples at this checkpoint.  Interleaved format: entry j maps to batch
            # item j % nsamples and intra-checkpoint position j // nsamples.
            # Survivors of the last checkpoint keep these ids as their final design ids, so the
            # oracle results written here are the ones the epoch-end pass finds cached.
            eval_batch_size = n_branch * self.beam_width * nsamples
            checkpoint_example_ids = [
                f"{batch_ids[j % nsamples]}_bsc{i}_{j // nsamples}" for j in range(eval_batch_size)
            ]
            checkpoint_design_ids = [f"{eid}{output_suffix}" for eid in checkpoint_example_ids]

            checkpoint_reward_model = self.reward_models[end_step]

            rewards, atom_array_results, candidate_metrics = self._simulate_and_compute_reward(
                model,
                batch,
                concat_dict_tensors(all_clean_predictions, dim=0) if score_clean_prediction else xt,
                end_step,
                checkpoint_reward_model,
                ts=ts,
                gt=gt,
                recycle_depths=recycle_depths,
                example_ids=checkpoint_example_ids,
                output_base_dir=output_base_dir,
                design_ids=checkpoint_design_ids,
                insertion_logits=insertion_logits,
                sequence_logits=sequence_logits,
                rollout=not score_clean_prediction,
                metric_names=self.checkpoints[end_step].get("metrics", []),
            )
            if score_clean_prediction:
                rewards = rewards.masked_fill(~torch.isfinite(rewards), -torch.inf)
            for candidate, result in enumerate(atom_array_results):
                result["orig_batch_idx"] = original_batch_indices[candidate % nsamples]

            # Reshape and select top beam_width candidates
            # Current shape: [n_branch * beam_width * nsamples] in interleaved format
            # Reshape to: [n_branch, beam_width, nsamples]
            # Then permute to: [nsamples, n_branch * beam_width]
            rewards_reshaped = rewards.view(n_branch, self.beam_width, nsamples)
            rewards_for_selection = rewards_reshaped.permute(2, 0, 1).reshape(nsamples, n_branch * self.beam_width)

            # Top-k selection for beam search.
            # stable=True guarantees deterministic tie-breaking (lower index wins)
            top_k_indices = torch.argsort(rewards_for_selection, dim=1, descending=True, stable=True)[
                :, : self.beam_width
            ]

            # Convert flat indices to replica and branch indices
            replica_indices = top_k_indices % self.beam_width  # [nsamples, beam_width]
            branch_indices = top_k_indices // self.beam_width  # [nsamples, beam_width]

            logger.debug(f"Selected top {self.beam_width} candidates per sample")

            # Create sample indices for broadcasting
            sample_indices = torch.arange(nsamples, device=device).unsqueeze(1).expand(-1, self.beam_width)

            # Calculate global indices using vectorized operations
            global_indices = branch_indices * nsamples * self.beam_width + replica_indices * nsamples + sample_indices

            # Transpose and flatten to get interleaved format [1,2,3,1,2,3]
            select_indices = global_indices.T.flatten()

            continuing_inputs = torch.ones(nsamples, dtype=torch.bool, device=device)
            if criterion := self.continuation_criteria.get(end_step):
                passes = _continuation_passes(criterion, candidate_metrics, end_step)
                selected_passes = torch.tensor(passes, dtype=torch.bool, device=device)[select_indices]
                continuing_inputs = selected_passes.view(self.beam_width, nsamples).any(dim=0)
            # A weighted key that none of the selected metrics produces scores every
            # candidate -inf, which would silently reduce the prune below to index order.
            if torch.isinf(rewards).all() and continuing_inputs.any():
                raise ValueError(
                    f"Every candidate at step {end_step} scored -inf, so the prune would rank nothing: "
                    f"a weighted key in {sorted(checkpoint_reward_model.weights)} is not produced by "
                    f"the checkpoint's explicit metrics list. The warnings above name the missing key(s)."
                )

            # Only survivors scored at the schedule end are returned directly as final designs.
            survivor_mask = torch.zeros_like(rewards, dtype=torch.bool)
            if end_step == actual_nsteps:
                survivor_mask[select_indices] = True

            # Keep the intermediate samples that satisfy the configured save criterion.
            criterion = None if score_clean_prediction else self.intermediate_sample_save_criterion
            criterion_outputs = [criterion.compute(m) for m in candidate_metrics] if criterion is not None else []
            if criterion_outputs and not any(out["all_metrics_present"] for out in criterion_outputs):
                raise ValueError(
                    f"intermediate_sample_save_criterion thresholds {sorted(criterion.metric_thresholds)} were "
                    f"unevaluable for every candidate at step {end_step}, so no intermediate sample can ever be "
                    f"kept there: explicitly list their providers in the checkpoint's metrics."
                )
            keep_mask = torch.tensor(
                [bool(out["pass"]) for out in criterion_outputs] or [False] * len(candidate_metrics),
                dtype=torch.bool,
                device=rewards.device,
            )
            # A final-beam member is returned once, as a final sample -- never also as an intermediate.
            keep_mask &= ~survivor_mask
            if keep_mask.any():
                for k_idx in keep_mask.nonzero(as_tuple=False).squeeze(1).tolist():
                    result = dict(atom_array_results[k_idx])
                    result["reward"] = rewards[k_idx].item()
                    all_intermediate_results.append(result)
                logger.info(f"Checkpoint {i}: kept {keep_mask.sum().item()}/{keep_mask.numel()} intermediate samples")
            elif criterion is not None:
                logger.info(
                    f"Checkpoint {i}: no intermediate samples met the save criterion "
                    f"(max reward={rewards.max().item():.3f})"
                )

            # Clean up oracle output dirs for scored candidates we are not keeping as outputs.
            # These were written during reward computation. Each design owns its directory,
            # so removing it cannot disturb a concurrent per-GPU process.
            if output_base_dir is not None and not score_clean_prediction:
                for k_idx in (~(keep_mask | survivor_mask)).nonzero(as_tuple=False).squeeze(1).tolist():
                    design_id = checkpoint_design_ids[k_idx]
                    try:
                        shutil.rmtree(design_output_dir(output_base_dir, design_id), ignore_errors=True)
                    except ValueError as e:
                        logger.error(f"beam search: skipping cleanup for design {design_id!r} -- {e}")

            if not continuing_inputs.all():
                retained = continuing_inputs.nonzero(as_tuple=True)[0].tolist()
                logger.info(f"Checkpoint {end_step}: continuing {len(retained)}/{nsamples} inputs")
                if not retained:
                    return all_intermediate_results
                select_indices = select_indices.view(self.beam_width, nsamples)[:, retained].flatten()
                batch = _select_batch_examples(batch, retained, nsamples)
                batch_ids = [batch_ids[index] for index in retained]
                original_batch_indices = [original_batch_indices[index] for index in retained]
                nsamples = len(retained)
                model._current_batch = batch

            xt = {key: value[select_indices] for key, value in xt.items()}
            if self.save_states and end_step < actual_nsteps:
                state_dir = Path(output_base_dir).parent / "sampling_states"
                state_dir.mkdir(parents=True, exist_ok=True)
                for rank in range(self.beam_width):
                    path = state_dir / f"{batch_ids[0]}{output_suffix}_step{end_step}_beam{rank}.pt"
                    candidate = {
                        key: value[rank * nsamples : (rank + 1) * nsamples].detach().cpu().clone()
                        for key, value in xt.items()
                    }
                    torch.save({"x": candidate, "step": end_step}, path)
                    logger.info(f"Saved candidate: {path}")
            start_step = end_step

        if end_step == actual_nsteps:
            final_results = [atom_array_results[k] for k in select_indices.tolist()]
            final_rewards = rewards[select_indices]
            for result, reward in zip(final_results, final_rewards.tolist(), strict=True):
                result["reward"] = reward
            logger.info(
                f"Final beam reward stats: mean={final_rewards.mean().item():.4f}, std={final_rewards.std().item():.4f}"
            )
        else:
            _, final_results, _ = self._simulate_and_compute_reward(
                model,
                batch,
                xt,
                end_step,
                reward_model=None,
                ts=ts,
                gt=gt,
                recycle_depths=recycle_depths,
                example_ids=[f"{name}_bsfinal_{replica}" for replica in range(self.beam_width) for name in batch_ids],
            )
            for candidate, result in enumerate(final_results):
                result["orig_batch_idx"] = original_batch_indices[candidate % nsamples]

        logger.info(
            f"Beam search complete. Generated {len(all_intermediate_results) + len(final_results)} samples "
            f"({len(all_intermediate_results)} intermediate + {len(final_results)} final)"
        )
        return all_intermediate_results + final_results

    def _simulate_and_compute_reward(
        self,
        model: "Proteina",
        batch: dict,
        xt: dict[str, torch.Tensor],
        current_step: int,
        reward_model: RewardModel | None,
        ts: dict[str, torch.Tensor],
        gt: dict[str, torch.Tensor],
        recycle_depths: tuple[int, ...],
        example_ids: list[str] | None = None,
        output_base_dir: "Path | None" = None,
        design_ids: list[str] | None = None,
        insertion_logits: torch.Tensor | None = None,
        sequence_logits: torch.Tensor | None = None,
        rollout: bool = True,
        metric_names: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, list[dict], list[dict]]:
        """Decode and score candidates, optionally rolling out to completion first.

        With rollout=False, xt is the clean prediction from the preceding step;
        only design metrics run, without writing scoring outputs.

        Args:
            model: Proteina model instance
            batch: Original batch
            xt: Intermediate states, or clean predictions when rollout=False
            current_step: Current timestep in diffusion process
            reward_model: Reward model for scoring; None skips checkpoint metrics and rewards.
            recycle_depths: Recycle depth for each complete-trajectory step.
            example_ids: Per-sample base IDs carried on the returned atom-array results.
            output_base_dir: Directory for oracle saving (enables caching).
            design_ids: Per-sample on-disk design IDs to save oracle results under.
            insertion_logits: Fallback ``[batch, n, n]`` logits for *xt*, kept when the rollout
                produces none.
            sequence_logits: Fallback ``[batch, n, classes]`` logits for *xt*, kept when the
                rollout produces none.
            rollout: Complete the trajectory before decoding and allow oracle scoring.
            metric_names: Explicit metric providers to run; omitted means no metrics.

        Returns:
            Tuple of (rewards tensor, atom_array_results list, per-sample metrics list).
            Without a reward model, rewards is None and metrics is empty; output evaluation runs separately.
            With scoring enabled, both lists contain one entry per sample in interleaved batch-item order.
        """
        # Derive actual schedule length from the passed schedule (partial or full).
        actual_nsteps = next(iter(ts.values())).shape[0] - 1

        # Get batch parameters
        batch_size = next(iter(xt.values())).shape[0]
        segment_size = batch["padding_mask"].shape[0]

        final_xt = xt
        if not rollout:
            output_base_dir = None
        else:
            # Roll out from current_step to end of schedule to get clean samples.
            logger.debug(f"Rolling out {batch_size} samples: step {current_step} → {actual_nsteps}")

            # Slice schedule once — same segment used for every batch chunk.
            ts_segment = {k: v[current_step:] for k, v in ts.items()}
            gt_segment = {k: v[current_step:] for k, v in gt.items()}

            final_xt_batches = []
            insertion_logits_batches = []
            sequence_logits_batches = []

            # Process in segments to respect batch size constraints
            for i in range(0, batch_size, segment_size):
                end_idx = min(i + segment_size, batch_size)
                batch_xt = {key: value[i:end_idx] for key, value in xt.items()}

                batch_final_xt, _, batch_extra_info = model.fm.partial_simulation(
                    batch=copy.deepcopy(batch),
                    x=batch_xt,
                    predict_for_sampling=model.predict_for_sampling,
                    recycle_depths=recycle_depths[current_step:],
                    nsteps=actual_nsteps - current_step,
                    ts=ts_segment,
                    gt=gt_segment,
                    simulation_step_params={
                        data_mode: model.inf_cfg.model[data_mode]["simulation_step_params"]
                        for data_mode in model.fm.data_modes
                    },
                    device=model.device,
                    guidance_w=model.inf_cfg.args.get("guidance_w", 1.0),
                )
                final_xt_batches.append(batch_final_xt)
                insertion_logits_batches.append(batch_extra_info.get("insertion_logits"))
                sequence_logits_batches.append(batch_extra_info.get("sequence_logits"))

            # Concatenate batches
            final_xt = concat_dict_tensors(final_xt_batches, dim=0)

            # A zero-step rollout (last checkpoint) runs no forward pass, so the caller's logits stand:
            # they come from the step that produced this state, exactly as the vanilla path uses them.
            if all(logits is not None for logits in insertion_logits_batches):
                insertion_logits = torch.cat(insertion_logits_batches, dim=0)
            if all(logits is not None for logits in sequence_logits_batches):
                sequence_logits = torch.cat(sequence_logits_batches, dim=0)

        # Expand the batch across the branched sample axis.
        num_repeats = (batch_size + segment_size - 1) // segment_size
        expanded_batch = {
            k: (
                v.repeat(num_repeats, *([1] * (v.dim() - 1)))[:batch_size]
                if torch.is_tensor(v) and v.shape[0] == segment_size
                else v
            )
            for k, v in batch.items()
        }
        expanded_batch["sequence_logits"] = sequence_logits
        # Format to protein structures (decode to structure-sequence pair)
        final_sample_prots = model.sample_formatting(
            x=final_xt,
            ret_mode="coors37_n_aatype",
            batch=expanded_batch,
        )

        # Format to atom arrays, then compute rewards R(x, s)
        atom_array_results = model.format_sample_prots_to_atom_arrays(
            final_sample_prots, batch, batch_idx=0, example_ids=example_ids, insertion_logits=insertion_logits
        )
        if reward_model is None:
            return None, atom_array_results, []
        rewards, candidate_metrics = self._compute_rewards(
            model,
            atom_array_results,
            reward_model,
            output_base_dir=output_base_dir,
            design_ids=design_ids,
            run_oracles=rollout,
            metric_names=metric_names,
        )

        logger.debug(
            f"Computed rewards: mean={rewards.mean().item():.4f}, "
            f"min={rewards.min().item():.4f}, max={rewards.max().item():.4f}"
        )

        return rewards, atom_array_results, candidate_metrics

    def _compute_rewards(
        self,
        model: "Proteina",
        atom_array_results: list[dict],
        reward_model: RewardModel,
        output_base_dir: "Path | None" = None,
        design_ids: list[str] | None = None,
        run_oracles: bool = True,
        metric_names: list[str] | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Compute rewards for pre-formatted atom array results.

        Runs the explicitly named design, oracle, and refolding metrics, then
        aggregates their outputs with the checkpoint's reward weights.

        Args:
            model: Proteina model instance
            atom_array_results: Pre-formatted list of result dicts from
                :py:meth:`~Proteina.format_sample_prots_to_atom_arrays`.
            reward_model: Reward model for scoring
            output_base_dir: Directory for oracle CIF / confidence JSON saving.
                Pass ``None`` to skip saving (oracle results won't be cached).
            run_oracles: False for prediction scoring; evaluate design metrics only.
            metric_names: Explicit metric providers to run; omitted means no metrics.
            design_ids: On-disk design IDs to save oracle results under. Must match the ids
                ``validation_step`` later assigns, so on_validation_epoch_end skips re-folding.

        Returns:
            Tuple of (rewards tensor, per-sample metrics dicts), both in the same order as
            *atom_array_results*. Each metrics dict carries the sample's own reward under
            :py:data:`~rfproteina.rewards.reward_model.SEARCH_REWARD_KEY` alongside
            its raw (unclipped) metric values, so pass criteria can threshold on either.
        """
        device = model.device
        batch = model._current_batch

        metric_names = metric_names or []
        rewards = []
        all_metrics: list[dict] = []
        for i, result in enumerate(atom_array_results):
            orig_batch_idx = i % max(len(batch.get("atom_array_batch", [batch.get("atom_array")])), 1)
            example_id: str | None = design_ids[i] if design_ids is not None else None

            # Annotate unindexed insertion before scoring, mirroring _stage_generated_structure.
            array_to_annotate = result.get("full_array_with_unindexed") or result["generated_atom_array"]
            result["generated_atom_array"] = annotate_unindexed_insertion(
                array_to_annotate, result.get("predicted_insertion_mapping") or {}
            )

            # Phase 1: explicitly selected design metrics.
            metrics_output = {}
            design_manager = model.design_metric_manager
            if design_manager is not None:
                metrics_output = run_metrics_for_result(
                    result,
                    batch,
                    orig_batch_idx,
                    design_manager,
                    output_base_dir=output_base_dir,
                    example_id=example_id,
                    metric_names=metric_names,
                )

            # Phase 2: explicitly selected oracle and refolding metrics.
            oracle_output = (
                run_oracle_and_refolding_metrics(
                    atom_array=result["generated_atom_array"],
                    example_id=example_id or result.get("example_id", "sample"),
                    output_dir=output_base_dir,
                    oracle_mm=model.oracle_metric_manager,
                    refold_mm=model.refolding_metric_manager,
                    metric_names=metric_names,
                )
                if run_oracles
                else {}
            )

            merged = {**metrics_output, **oracle_output}
            reward = reward_model.aggregate_metrics_output(merged)
            merged[SEARCH_REWARD_KEY] = reward
            rewards.append(reward)
            all_metrics.append(merged)

        return torch.tensor(rewards, device=device, dtype=torch.float32), all_metrics
