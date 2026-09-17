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

Key insight: We roll out to clean samples for reward evaluation instead of
using Tweedie's formula, as intermediate states are too noisy for folding models.
Unlike standard beam search, we keep the clean samples generated during reward
estimation if they meet success criteria, significantly improving total successes.
"""

import copy
import shutil
from pathlib import Path
from typing import Any

import torch

from rfproteina.metrics.oracles.pass_criterion_metric import PassCriterionMetric
from rfproteina.metrics.utils.metrics_utils import run_metrics_for_result, run_oracle_refolding_and_pass_criteria
from rfproteina.model_masks import compute_flow_emit_masks
from rfproteina.rewards.reward_model import SEARCH_REWARD_KEY
from rfproteina.search.base_strategy import SearchStrategy
from rfproteina.search.vanilla import build_partial_schedule, resolve_partial_t
from rfproteina.transforms.unindexing import annotate_unindexed_insertion
from rfproteina.utils.common import design_output_dir
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.torch import concat_dict_tensors

logger = RankedLogger(__name__, rank_zero_only=True)


class BeamSearchStrategy(SearchStrategy):
    """Beam search with reward-guided pruning.

    Explores multiple generation paths (beam), evaluating and pruning
    at specified checkpoints. Maintains top-k candidates based on rewards.

    Args:
        step_checkpoints: Diffusion steps at which to evaluate and prune (e.g., [0, 25, 50])
        n_branch: Number of candidates to generate at each branch point (typically 4)
        beam_width: Number of candidates to keep after each prune (typically 4)
        intermediate_sample_save_criterion: PassCriterionMetric deciding which of the rolled-out
            candidates scored at each checkpoint are kept as extra outputs. Final-beam members
            are always returned, and never also counted as intermediates.

    Example:
        >>> strategy = BeamSearchStrategy(
        ...     step_checkpoints=[0, 25, 50],
        ...     n_branch=4,
        ...     beam_width=4,
        ...     intermediate_sample_save_criterion=PassCriterionMetric(
        ...         metric_thresholds={"search.reward": {"min": 0.0}}
        ...     ),
        ... )
        >>> results = strategy.generate(model, batch, reward_model)
    """

    def __init__(
        self,
        step_checkpoints: list[int],
        n_branch: int = 4,
        beam_width: int = 4,
        intermediate_sample_save_criterion: PassCriterionMetric | None = None,
    ):
        """Initialize beam search strategy.

        Args:
            step_checkpoints: Diffusion steps to evaluate at (K steps between each)
            n_branch: Candidates to generate per beam state
            beam_width: Candidates to keep after pruning
            intermediate_sample_save_criterion: Criterion for keeping intermediate samples.
                Defaults to ``None`` (keep none).
        """
        self.step_checkpoints = sorted(step_checkpoints)
        self.n_branch = n_branch
        self.beam_width = beam_width
        self.intermediate_sample_save_criterion = intermediate_sample_save_criterion

        if len(self.step_checkpoints) < 2:
            raise ValueError(
                f"step_checkpoints must have at least 2 entries (start and end), " f"got {step_checkpoints}."
            )
        if self.step_checkpoints[0] != 0:
            raise ValueError(
                f"step_checkpoints must begin at 0 (beam is initialized at the start of the "
                f"schedule), got {self.step_checkpoints[0]}."
            )
        diffs = [b - a for a, b in zip(self.step_checkpoints, self.step_checkpoints[1:], strict=False)]
        if any(d <= 0 for d in diffs):
            raise ValueError(
                f"step_checkpoints must be strictly increasing "
                f"(duplicate or decreasing values produce zero- or negative-length intervals); "
                f"got {step_checkpoints}."
            )

    def generate(
        self,
        model: "Proteina",
        batch: dict,
        reward_model: "RewardModel | None" = None,
    ) -> list[dict[str, Any]]:
        """Generate structures using beam search.

        Implements beam search with reward-guided pruning.

        Args:
            model: Proteina model instance
            batch: Input batch from dataloader
            reward_model: Reward model for scoring candidates (required)

        Returns:
            List of dictionaries, each containing:
                - generated_atom_array: Final protein structure
                - reward: Final reward score
                - (optionally) intermediate_samples

        Raises:
            ValueError: If reward_model is None, if an override targets an unscored step, or if
                a checkpoint's reward or save criterion turns out to be unevaluable there.
        """
        if reward_model is None:
            raise ValueError("Beam search requires a reward model")

        # Scoring happens after advancing to each checkpoint past the first, so those are
        # the steps a per-checkpoint reward override may target.
        reward_model.validate_checkpoint_steps(self.step_checkpoints[1:])

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

        if self.step_checkpoints[-1] != actual_nsteps:
            frac_hint = f" (nsteps={nsteps} x fraction={frac} = {actual_nsteps})" if frac is not None else ""
            raise ValueError(
                f"step_checkpoints[-1]={self.step_checkpoints[-1]} must equal"
                f" actual_nsteps={actual_nsteps}{frac_hint}. "
                f"The last checkpoint must coincide with the end of the schedule so the surviving "
                f"beam states are fully denoised (t=1) before final formatting."
            )

        # Build schedule: partial (covering [partial_t, 1]) or full (covering [0, 1])
        if partial_t_dict:
            ts, gt = build_partial_schedule(
                partial_t_dict=partial_t_dict,
                nsteps=actual_nsteps,
                fm=model.fm,
                sampling_model_args=model.inf_cfg.model,
            )
        else:
            ts, gt = model.fm.sample_schedule(
                nsteps=actual_nsteps,
                sampling_model_args=model.inf_cfg.model,
            )

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

        x_0 = model.fm.sample_noise(
            n,
            shape=(nsamples * self.beam_width,),
            device=device,
            emit_masks=compute_flow_emit_masks(beam_emit_mask_batch),
            training=False,
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

        # Beam search loop over checkpoints
        for i in range(len(self.step_checkpoints) - 1):
            start_step = self.step_checkpoints[i]
            end_step = self.step_checkpoints[i + 1]
            logger.info(f"Checkpoint {i + 1}/{len(self.step_checkpoints) - 1}: steps {start_step} → {end_step}")

            all_xt_branches = []
            all_insertion_logits_branches = []

            # Process each replica separately due to batch size constraints
            for replica_idx in range(self.beam_width):
                # Extract samples for this replica (interleaved format)
                start = replica_idx * nsamples
                end = (replica_idx + 1) * nsamples
                replica_xt = {key: value[start:end] for key, value in xt.items()}
                ts_segment = {k: v[start_step:] for k, v in ts.items()}
                gt_segment = {k: v[start_step:] for k, v in gt.items()}

                # Generate n_branch branches from this replica
                for _ in range(self.n_branch):
                    branch_xt, _, branch_extra_info = model.fm.partial_simulation(
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
                    all_insertion_logits_branches.append(branch_extra_info.get("insertion_logits"))

            # Concatenate all replicas and branches
            xt = concat_dict_tensors(all_xt_branches, dim=0)

            insertion_logits = None
            if all(logits is not None for logits in all_insertion_logits_branches):
                insertion_logits = torch.cat(all_insertion_logits_branches, dim=0)

            # After each checkpoint transition, evaluate rewards and select top samples.
            # Example ids for each of the n_branch * beam_width * nsamples evaluated
            # samples at this checkpoint.  Interleaved format: entry j maps to batch
            # item j % nsamples and intra-checkpoint position j // nsamples.
            # Survivors of the last checkpoint keep these ids as their final design ids, so the
            # oracle results written here are the ones the epoch-end pass finds cached.
            eval_batch_size = self.n_branch * self.beam_width * nsamples
            checkpoint_example_ids = [
                f"{batch_ids[j % nsamples]}_bsc{i}_{j // nsamples}" for j in range(eval_batch_size)
            ]
            checkpoint_design_ids = [f"{eid}{output_suffix}" for eid in checkpoint_example_ids]

            # Reward used at this checkpoint: the base reward unless the config overrides it here.
            checkpoint_reward_model = reward_model.at_step(end_step)

            # Compute rewards by rolling out to completion.
            # atom_array_results is in interleaved format (i % nsamples maps to batch item i).
            rewards, atom_array_results, candidate_metrics = self._simulate_and_compute_reward(
                model,
                batch,
                xt,
                end_step,
                checkpoint_reward_model,
                ts=ts,
                gt=gt,
                recycle_depths=recycle_depths,
                example_ids=checkpoint_example_ids,
                output_base_dir=output_base_dir,
                design_ids=checkpoint_design_ids,
                insertion_logits=insertion_logits,
            )

            # A weighted key that no metric produces under this checkpoint's tags scores every
            # candidate -inf, which would silently reduce the prune below to index order.
            if torch.isinf(rewards).all():
                raise ValueError(
                    f"Every candidate at step {end_step} scored -inf, so the prune would rank nothing: "
                    f"a weighted key in {sorted(checkpoint_reward_model.weights)} is not produced under "
                    f"reward_metrics_tags={checkpoint_reward_model.reward_metrics_tags}. The warnings above "
                    f"name the missing key(s)."
                )

            # Reshape and select top beam_width candidates
            # Current shape: [n_branch * beam_width * nsamples] in interleaved format
            # Reshape to: [n_branch, beam_width, nsamples]
            # Then permute to: [nsamples, n_branch * beam_width]
            rewards_reshaped = rewards.view(self.n_branch, self.beam_width, nsamples)
            rewards_for_selection = rewards_reshaped.permute(2, 0, 1).reshape(nsamples, self.n_branch * self.beam_width)

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

            # Survivors of the last checkpoint are the final beam. They are already fully denoised
            # and already scored here, so they are returned as-is instead of being decoded and
            # scored a second time under a different design id (which the oracle cache would miss).
            survivor_mask = torch.zeros_like(rewards, dtype=torch.bool)
            if i == len(self.step_checkpoints) - 2:
                survivor_mask[select_indices] = True

            # Keep the intermediate samples that satisfy the configured save criterion.
            criterion = self.intermediate_sample_save_criterion
            criterion_outputs = [criterion.compute(m) for m in candidate_metrics] if criterion is not None else []
            if criterion_outputs and not any(out["all_metrics_present"] for out in criterion_outputs):
                raise ValueError(
                    f"intermediate_sample_save_criterion thresholds {sorted(criterion.metric_thresholds)} were "
                    f"unevaluable for every candidate at step {end_step}, so no intermediate sample can ever be "
                    f"kept there: those metrics do not run under "
                    f"reward_metrics_tags={checkpoint_reward_model.reward_metrics_tags}."
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
            if output_base_dir is not None:
                for k_idx in (~(keep_mask | survivor_mask)).nonzero(as_tuple=False).squeeze(1).tolist():
                    design_id = checkpoint_design_ids[k_idx]
                    try:
                        shutil.rmtree(design_output_dir(output_base_dir, design_id), ignore_errors=True)
                    except ValueError as e:
                        logger.error(f"beam search: skipping cleanup for design {design_id!r} -- {e}")

            # Select samples
            xt = {key: value[select_indices] for key, value in xt.items()}

        # The final beam is the last checkpoint's survivors, already decoded and scored above.
        final_results = [atom_array_results[k] for k in select_indices.tolist()]
        final_rewards = rewards[select_indices]
        for result, reward in zip(final_results, final_rewards.tolist(), strict=True):
            result["reward"] = reward

        logger.info(
            f"Beam search complete. Generated {len(all_intermediate_results) + len(final_results)} samples "
            f"({len(all_intermediate_results)} intermediate + {len(final_results)} final)"
        )
        logger.info(
            f"Final beam reward stats: mean={final_rewards.mean().item():.4f}, std={final_rewards.std().item():.4f}"
        )

        return all_intermediate_results + final_results

    def _simulate_and_compute_reward(
        self,
        model: "Proteina",
        batch: dict,
        xt: dict[str, torch.Tensor],
        current_step: int,
        reward_model: "RewardModel",
        ts: dict[str, torch.Tensor],
        gt: dict[str, torch.Tensor],
        recycle_depths: tuple[int, ...],
        example_ids: list[str] | None = None,
        output_base_dir: "Path | None" = None,
        design_ids: list[str] | None = None,
        insertion_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict], list[dict]]:
        """Roll out samples to completion and compute rewards.

        This implements the "roll out to get clean sample" step.
        Each intermediate state is completed to t=1, decoded to structure-sequence
        pair (x, s), and evaluated with reward function R(x, s).

        Args:
            model: Proteina model instance
            batch: Original batch
            xt: Current intermediate states at timestep current_step
            current_step: Current timestep in diffusion process
            reward_model: Reward model for scoring
            recycle_depths: Recycle depth for each complete-trajectory step.
            example_ids: Per-sample base IDs carried on the returned atom-array results.
            output_base_dir: Directory for oracle saving (enables caching).
            design_ids: Per-sample on-disk design IDs to save oracle results under.
            insertion_logits: Fallback ``[batch, n, n]`` logits for *xt*, kept when the rollout
                produces none.

        Returns:
            Tuple of (rewards tensor, atom_array_results list, per-sample metrics list).
            The two lists contain one entry per sample, in interleaved batch-item order —
            safe to index directly by position.
        """
        # Derive actual schedule length from the passed schedule (partial or full).
        actual_nsteps = next(iter(ts.values())).shape[0] - 1

        # Get batch parameters
        batch_size = next(iter(xt.values())).shape[0]
        segment_size = batch["padding_mask"].shape[0]

        # Roll out from current_step to end of schedule to get clean samples.
        logger.debug(f"Rolling out {batch_size} samples: step {current_step} → {actual_nsteps}")

        # Slice schedule once — same segment used for every batch chunk.
        ts_segment = {k: v[current_step:] for k, v in ts.items()}
        gt_segment = {k: v[current_step:] for k, v in gt.items()}

        final_xt_batches = []
        insertion_logits_batches = []

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

        # Concatenate batches
        final_xt = concat_dict_tensors(final_xt_batches, dim=0)

        # A zero-step rollout (last checkpoint) runs no forward pass, so the caller's logits stand:
        # they come from the step that produced this state, exactly as the vanilla path uses them.
        if all(logits is not None for logits in insertion_logits_batches):
            insertion_logits = torch.cat(insertion_logits_batches, dim=0)

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
        rewards, candidate_metrics = self._compute_rewards(
            model,
            atom_array_results,
            reward_model,
            output_base_dir=output_base_dir,
            design_ids=design_ids,
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
        reward_model: "RewardModel",
        output_base_dir: "Path | None" = None,
        design_ids: list[str] | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Compute rewards for pre-formatted atom array results.

        Uses the model's configured metric managers (design, oracle, refolding,
        pass_criteria) with :py:attr:`~RewardModel.reward_metrics_tags` to control
        which metrics fire, then aggregates with the reward weights.

        Args:
            model: Proteina model instance
            atom_array_results: Pre-formatted list of result dicts from
                :py:meth:`~Proteina.format_sample_prots_to_atom_arrays`.
            reward_model: Reward model for scoring
            output_base_dir: Directory for oracle CIF / confidence JSON saving.
                Pass ``None`` to skip saving (oracle results won't be cached).
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

        # Compute rewards using Proteina's configured metric managers.
        # reward_metrics_tags controls which metrics fire
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

            # Phase 1: design metrics (fast; tag-unconditional metrics always run)
            metrics_output = {}
            if model.design_metric_manager is not None:
                metrics_output = run_metrics_for_result(
                    result,
                    batch,
                    orig_batch_idx,
                    model.design_metric_manager,
                    output_base_dir=output_base_dir,
                    example_id=example_id,
                    metrics_tags_override=reward_model.reward_metrics_tags,
                )

            # Phase 2: oracle + refolding + pass_criteria
            # reward_metrics_tags determines which oracle fires.
            oracle_output = run_oracle_refolding_and_pass_criteria(
                atom_array=result["generated_atom_array"],
                example_id=example_id or result.get("example_id", "sample"),
                output_dir=output_base_dir,
                oracle_mm=model.oracle_metric_manager,
                refold_mm=model.refolding_metric_manager,
                pass_mm=model.pass_criteria_metric_manager,
                metrics_tags=reward_model.reward_metrics_tags,
                prior_metrics=metrics_output,
            )

            merged = {**metrics_output, **oracle_output}
            reward = reward_model.aggregate_metrics_output(merged)
            merged[SEARCH_REWARD_KEY] = reward
            rewards.append(reward)
            all_metrics.append(merged)

        return torch.tensor(rewards, device=device, dtype=torch.float32), all_metrics
