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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from foundry.utils.ddp import RankedLogger

if TYPE_CHECKING:
    pass

from proteinfoundation.metrics.oracles.oracle_metric import example_output_dir
from proteinfoundation.metrics.utils.metrics_utils import run_metrics_for_result, run_oracle_refolding_and_pass_criteria
from proteinfoundation.search.base_strategy import SearchStrategy
from proteinfoundation.search.vanilla import build_partial_schedule, resolve_partial_t
from proteinfoundation.transforms.unindexing import annotate_unindexed_insertion
from proteinfoundation.utils.tensor_utils import concat_dict_tensors

logger = RankedLogger(__name__, rank_zero_only=True)


class BeamSearchStrategy(SearchStrategy):
    """Beam search with reward-guided pruning.

    Explores multiple generation paths (beam), evaluating and pruning
    at specified checkpoints. Maintains top-k candidates based on rewards.

    Args:
        step_checkpoints: Diffusion steps at which to evaluate and prune (e.g., [0, 100, 200, 300, 400])
        n_branch: Number of candidates to generate at each branch point (typically 4)
        beam_width: Number of candidates to keep after each prune (typically 4)
        keep_intermediate_samples: Whether to save intermediate structures that meet criteria
        intermediate_sample_reward_threshold: When ``keep_intermediate_samples=True``, only
            save intermediate structures scoring above this value. Defaults to ``-inf`` (save all).

    Example:
        >>> strategy = BeamSearchStrategy(
        ...     step_checkpoints=[0, 100, 200, 300, 400], n_branch=4, beam_width=4
        ... )
        >>> results = strategy.generate(model, batch, reward_model)
    """

    def __init__(
        self,
        step_checkpoints: list[int],
        n_branch: int = 4,
        beam_width: int = 4,
        keep_intermediate_samples: bool = False,
        intermediate_sample_reward_threshold: float = float("-inf"),
    ):
        """Initialize beam search strategy.

        Args:
            step_checkpoints: Diffusion steps to evaluate at (K steps between each)
            n_branch: Candidates to generate per beam state
            beam_width: Candidates to keep after pruning
            keep_intermediate_samples: Save intermediate results
            intermediate_sample_reward_threshold: Minimum reward for saving intermediate
                samples. Only used when ``keep_intermediate_samples=True``.
        """
        self.step_checkpoints = sorted(step_checkpoints)
        self.n_branch = n_branch
        self.beam_width = beam_width
        self.keep_intermediate_samples = keep_intermediate_samples
        self.intermediate_sample_reward_threshold = intermediate_sample_reward_threshold

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
            ValueError: If reward_model is None
        """
        if reward_model is None:
            raise ValueError("Beam search requires a reward model")

        logger.info(
            f"Running beam search with n_branch={self.n_branch}, "
            f"beam_width={self.beam_width}, checkpoints={self.step_checkpoints}"
        )

        # Preprocess batch from atom37 collate format
        batch = model.preprocess_batch_for_generation(batch)

        # Store batch in model for reward computation
        model._current_batch = batch

        # Extract parameters from model's inference config
        self_cond = model.inf_cfg.args.self_cond
        nsteps = model.inf_cfg.args.nsteps
        guidance_w = model.inf_cfg.args.get("guidance_w", 1.0)
        ag_ratio = model.inf_cfg.args.get("ag_ratio", 0.0)

        # Prepare prediction function
        fn_predict_for_sampling = partial(model.predict_for_sampling, n_recycle=model.inf_cfg.get("n_recycle", 0))

        # Extract batch dimensions
        nsamples = batch["coords_nm"].shape[0]  # batch size
        n = batch["coords_nm"].shape[1]  # number of tokens
        device = model.device

        # Initialize mask (keep batch dimension, no squeezing needed)
        if "padding_mask" in batch:
            init_mask = batch["padding_mask"]
        else:
            init_mask = torch.ones(nsamples, n).long().bool().to(device)
        assert init_mask.shape == (nsamples, n), f"Expected mask shape {(nsamples, n)}, got {init_mask.shape}"

        simulation_step_params = {
            data_mode: model.inf_cfg.model[data_mode]["simulation_step_params"] for data_mode in model.fm.data_modes
        }

        partial_t_dict = resolve_partial_t(model.inf_cfg.args, model.fm, model.inf_cfg.model)

        # partial_trajectory_fraction_steps affects the number of steps to actually run
        frac = model.inf_cfg.args.get("partial_trajectory_fraction_steps", None)
        actual_nsteps = round(frac * nsteps) if frac is not None else nsteps

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
        mask = init_mask.repeat(self.beam_width, 1)

        x_0 = model.fm.sample_noise(
            n,
            shape=(nsamples * self.beam_width,),
            device=device,
            mask=mask,
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

        x_1_pred = None
        insertion_logits = None

        all_intermediate_results: list[dict] = []
        all_example_ids: list[str] = []

        # Use the canonical dataset_name_hash prefix so oracle outputs land in the
        # same content-addressed directory that proteina.py will assign the design CIF to.
        _raw_batch_ids: list[str] = batch.get("example_id_batch", [f"sample{k}" for k in range(nsamples)])
        _dataset_name_hash: str = getattr(model, "_current_dataset_name_hash", "")
        batch_ids: list[str] = (
            [f"{_dataset_name_hash}__{rid}" for rid in _raw_batch_ids] if _dataset_name_hash else _raw_batch_ids
        )

        output_base_dir = getattr(model, "_current_output_base_dir", None)
        output_suffix: str = getattr(model, "_current_output_suffix", "")

        # Beam search loop over checkpoints
        for i in range(len(self.step_checkpoints) - 1):
            start_step = self.step_checkpoints[i]
            end_step = self.step_checkpoints[i + 1]
            logger.info(f"Checkpoint {i + 1}/{len(self.step_checkpoints) - 1}: steps {start_step} → {end_step}")

            all_xt_branches = []
            all_x_1_pred_branches = []
            all_insertion_logits_branches = []

            # Process each replica separately due to batch size constraints
            for replica_idx in range(self.beam_width):
                # Extract samples for this replica (interleaved format)
                start = replica_idx * nsamples
                end = (replica_idx + 1) * nsamples
                replica_xt = {key: value[start:end] for key, value in xt.items()}
                replica_x_1_pred = (
                    {key: value[start:end] for key, value in x_1_pred.items()} if x_1_pred is not None else None
                )
                ts_segment = {k: v[start_step:] for k, v in ts.items()}
                gt_segment = {k: v[start_step:] for k, v in gt.items()}

                # Generate n_branch branches from this replica
                for _ in range(self.n_branch):
                    branch_xt, branch_x_1_pred, branch_extra_info = model.fm.partial_simulation(
                        batch=copy.deepcopy(batch),
                        x=replica_xt,
                        x_1_pred=replica_x_1_pred,
                        mask=init_mask,
                        predict_for_sampling=fn_predict_for_sampling,
                        nsteps=end_step - start_step,
                        ts=ts_segment,
                        gt=gt_segment,
                        self_cond=self_cond,
                        simulation_step_params=simulation_step_params,
                        device=model.device,
                        guidance_w=guidance_w,
                        ag_ratio=ag_ratio,
                    )

                    all_xt_branches.append(branch_xt)
                    all_x_1_pred_branches.append(branch_x_1_pred)
                    all_insertion_logits_branches.append(branch_extra_info.get("insertion_logits"))

            # Concatenate all replicas and branches
            xt = concat_dict_tensors(all_xt_branches, dim=0)

            x_1_pred = None
            if all_x_1_pred_branches[0] is not None:
                x_1_pred = concat_dict_tensors(all_x_1_pred_branches, dim=0)

            insertion_logits = None
            if all(logits is not None for logits in all_insertion_logits_branches):
                insertion_logits = torch.cat(all_insertion_logits_branches, dim=0)

            # After each checkpoint transition, evaluate rewards and select top samples.
            # Example ids for each of the n_branch * beam_width * nsamples evaluated
            # samples at this checkpoint.  Interleaved format: entry j maps to batch
            # item j % nsamples and intra-checkpoint position j // nsamples.
            eval_batch_size = self.n_branch * self.beam_width * nsamples
            checkpoint_example_ids = [
                f"{batch_ids[j % nsamples]}_bsc{i}_{j // nsamples}" for j in range(eval_batch_size)
            ]

            # Compute rewards by rolling out to completion.
            # atom_array_results is in interleaved format (i % nsamples maps to batch item i).
            _, rewards, atom_array_results = self._simulate_and_compute_reward(
                model,
                batch,
                xt,
                end_step,
                init_mask,
                reward_model,
                ts=ts,
                gt=gt,
                example_ids=checkpoint_example_ids if self.keep_intermediate_samples else None,
                output_base_dir=output_base_dir if self.keep_intermediate_samples else None,
                output_suffix=output_suffix if self.keep_intermediate_samples else "",
            )

            # Save intermediate samples if requested
            if self.keep_intermediate_samples:
                reward_threshold = self.intermediate_sample_reward_threshold
                reward_mask = rewards > reward_threshold
                if reward_mask.any():
                    kept_indices = reward_mask.nonzero(as_tuple=False).squeeze(1).tolist()
                    for k_idx in kept_indices:
                        result = dict(atom_array_results[k_idx])
                        result["reward"] = rewards[k_idx].item()
                        all_intermediate_results.append(result)
                        all_example_ids.append(checkpoint_example_ids[k_idx])
                    logger.info(
                        f"Checkpoint {i}: saved {reward_mask.sum().item()} intermediate samples "
                        f"above threshold {reward_threshold}"
                    )
                else:
                    logger.info(
                        f"Checkpoint {i}: no intermediate samples above threshold {reward_threshold} "
                        f"(max reward={rewards.max().item():.3f})"
                    )

                # Clean up oracle output dirs for samples that didn't pass the threshold.
                # These were written during reward computation but won't be kept as outputs.
                if output_base_dir is not None:
                    failing_indices = (~reward_mask).nonzero(as_tuple=False).squeeze(1).tolist()
                    for k_idx in failing_indices:
                        eid = checkpoint_example_ids[k_idx]
                        full_eid = f"{eid}{output_suffix}" if output_suffix else eid
                        target = example_output_dir(output_base_dir, full_eid)
                        if target.exists():
                            shutil.rmtree(target, ignore_errors=True)
                        # Remove empty shard parent dirs up to (but not including) output_base_dir.
                        shard_parent = target.parent
                        while shard_parent != output_base_dir and shard_parent.exists():
                            if next(shard_parent.iterdir(), None) is not None:
                                break
                            shard_parent.rmdir()
                            shard_parent = shard_parent.parent

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

            # Select samples
            xt = {key: value[select_indices] for key, value in xt.items()}
            if x_1_pred is not None:
                x_1_pred = {key: value[select_indices] for key, value in x_1_pred.items()}
            if insertion_logits is not None:
                insertion_logits = insertion_logits[select_indices]

        # Format final beam samples
        # mask here is already the expanded beam mask [nsamples * beam_width, n] in interleaved format
        # Need to expand can_be_encoded to match (interleaved: [1,2,3,1,2,3])
        can_be_encoded = batch.get("can_be_encoded")
        if can_be_encoded is not None:
            # Expand from [nsamples, n] to [nsamples * beam_width, n] in interleaved format
            expanded_can_be_encoded = can_be_encoded.repeat(self.beam_width, 1)
        else:
            expanded_can_be_encoded = None

        final_extra_info = {
            "padding_mask": mask,
            "can_be_encoded": expanded_can_be_encoded,
        }
        sample_prots = model.sample_formatting(
            x=xt,
            extra_info=final_extra_info,
            ret_mode="coors37_n_aatype",
        )

        # Example ids for the final beam: entry k maps to batch item k % nsamples
        # and beam position k // nsamples.  Always saved so oracle results are cached.
        final_example_ids = [
            f"{batch_ids[k % nsamples]}_bsfinal_{k // nsamples}" for k in range(nsamples * self.beam_width)
        ]
        all_example_ids.extend(final_example_ids)

        # Format and score the final beam in one pass
        final_results = model.format_sample_prots_to_atom_arrays(
            sample_prots, batch, batch_idx=0, example_ids=final_example_ids, insertion_logits=insertion_logits
        )
        final_rewards = self._compute_rewards(
            model,
            final_results,
            reward_model,
            example_ids=final_example_ids,
            output_base_dir=output_base_dir,
            output_suffix=output_suffix,
        )

        expected_sample_count = nsamples * self.beam_width
        actual_count = sample_prots["coors"].shape[0]
        assert (
            actual_count == expected_sample_count
        ), f"Expected {expected_sample_count} final beam samples, got {actual_count}"

        total_count = len(all_intermediate_results) + actual_count
        logger.info(
            f"Beam search complete. Generated {total_count} samples "
            f"({len(all_intermediate_results)} intermediate + {actual_count} final)"
        )
        logger.info(
            f"Final beam reward stats: mean={final_rewards.mean().item():.4f}, " f"std={final_rewards.std().item():.4f}"
        )

        for i, result in enumerate(final_results):
            result["reward"] = final_rewards[i].item()

        return all_intermediate_results + final_results

    def _simulate_and_compute_reward(
        self,
        model: "Proteina",
        batch: dict,
        xt: dict[str, torch.Tensor],
        current_step: int,
        mask: torch.Tensor,
        reward_model: "RewardModel",
        ts: dict[str, torch.Tensor],
        gt: dict[str, torch.Tensor],
        example_ids: list[str] | None = None,
        output_base_dir: "Path | None" = None,
        output_suffix: str = "",
    ) -> tuple[dict, torch.Tensor, list[dict]]:
        """Roll out samples to completion and compute rewards.

        This implements the "roll out to get clean sample" step.
        Each intermediate state is completed to t=1, decoded to structure-sequence
        pair (x, s), and evaluated with reward function R(x, s).

        Args:
            model: Proteina model instance
            batch: Original batch
            xt: Current intermediate states at timestep current_step
            current_step: Current timestep in diffusion process
            mask: Residue mask [nsamples, n_residues]
            reward_model: Reward model for scoring
            example_ids: Per-sample IDs forwarded to oracle metrics for saving.
            output_base_dir: Directory for oracle saving (enables caching).
            output_suffix: Rank/job suffix appended to each id before saving.

        Returns:
            Tuple of (sample_prots dict, rewards tensor, atom_array_results list).
            atom_array_results contains one formatted result dict per sample, in
            interleaved batch-item order — safe to index directly by position.
        """
        # Derive actual schedule length from the passed schedule (partial or full).
        actual_nsteps = next(iter(ts.values())).shape[0] - 1

        # Get batch parameters
        batch_size = next(iter(xt.values())).shape[0]
        segment_size = mask.shape[0]

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
                x_1_pred=None,
                mask=mask,
                predict_for_sampling=partial(model.predict_for_sampling, n_recycle=model.inf_cfg.get("n_recycle", 0)),
                nsteps=actual_nsteps - current_step,
                ts=ts_segment,
                gt=gt_segment,
                self_cond=model.inf_cfg.args.self_cond,
                simulation_step_params={
                    data_mode: model.inf_cfg.model[data_mode]["simulation_step_params"]
                    for data_mode in model.fm.data_modes
                },
                device=model.device,
                guidance_w=model.inf_cfg.args.get("guidance_w", 1.0),
                ag_ratio=model.inf_cfg.args.get("ag_ratio", 0.0),
            )
            final_xt_batches.append(batch_final_xt)
            insertion_logits_batches.append(batch_extra_info.get("insertion_logits"))

        # Concatenate batches
        final_xt = concat_dict_tensors(final_xt_batches, dim=0)

        insertion_logits = None
        if all(logits is not None for logits in insertion_logits_batches):
            insertion_logits = torch.cat(insertion_logits_batches, dim=0)

        # Construct extra_info with properly sized masks
        # Both mask and can_be_encoded need to expand from [nsamples, n] to [batch_size, n]
        # where batch_size = nsamples * n_branch * beam_width from all the branching
        num_repeats = (batch_size + segment_size - 1) // segment_size
        expanded_mask = mask.repeat(num_repeats, 1)[:batch_size]

        # Expand can_be_encoded similarly (if it exists)
        can_be_encoded = batch.get("can_be_encoded")
        if can_be_encoded is not None:
            expanded_can_be_encoded = can_be_encoded.repeat(num_repeats, 1)[:batch_size]
        else:
            expanded_can_be_encoded = None

        final_extra_info = {
            "padding_mask": expanded_mask,
            "can_be_encoded": expanded_can_be_encoded,
        }

        # Format to protein structures (decode to structure-sequence pair)
        final_sample_prots = model.sample_formatting(
            x=final_xt,
            extra_info=final_extra_info,
            ret_mode="coors37_n_aatype",
        )

        # Format to atom arrays, then compute rewards R(x, s)
        atom_array_results = model.format_sample_prots_to_atom_arrays(
            final_sample_prots, batch, batch_idx=0, example_ids=example_ids, insertion_logits=insertion_logits
        )
        rewards = self._compute_rewards(
            model,
            atom_array_results,
            reward_model,
            example_ids=example_ids,
            output_base_dir=output_base_dir,
            output_suffix=output_suffix,
        )

        logger.debug(
            f"Computed rewards: mean={rewards.mean().item():.4f}, "
            f"min={rewards.min().item():.4f}, max={rewards.max().item():.4f}"
        )

        return final_sample_prots, rewards, atom_array_results

    def _compute_rewards(
        self,
        model: "Proteina",
        atom_array_results: list[dict],
        reward_model: "RewardModel",
        example_ids: list[str] | None = None,
        output_base_dir: "Path | None" = None,
        output_suffix: str = "",
    ) -> torch.Tensor:
        """Compute rewards for pre-formatted atom array results.

        Uses the model's configured metric managers (design, oracle, refolding,
        pass_criteria) with :py:attr:`~RewardModel.reward_metrics_tags` to control
        which metrics fire, then aggregates with the reward weights.

        Args:
            model: Proteina model instance
            atom_array_results: Pre-formatted list of result dicts from
                :py:meth:`~Proteina.format_sample_prots_to_atom_arrays`.
            reward_model: Reward model for scoring
            example_ids: Per-sample IDs (without output_suffix). When provided, oracle
                metrics save results keyed by ``f"{example_ids[i]}{output_suffix}"``,
                enabling on_validation_epoch_end to skip re-folding.
            output_base_dir: Directory for oracle CIF / confidence JSON saving.
                Pass ``None`` to skip saving (oracle results won't be cached).
            output_suffix: Rank/job suffix appended to each example_id before saving.

        Returns:
            Rewards tensor in the same order as atom_array_results.
        """
        device = model.device
        batch = model._current_batch

        # Compute rewards using Proteina's configured metric managers.
        # reward_metrics_tags controls which metrics fire
        rewards = []
        for i, result in enumerate(atom_array_results):
            orig_batch_idx = i % max(len(batch.get("atom_array_batch", [batch.get("atom_array")])), 1)
            example_id: str | None = None
            if example_ids is not None:
                example_id = f"{example_ids[i]}{output_suffix}" if output_suffix else example_ids[i]

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
            rewards.append(reward)

        return torch.tensor(rewards, device=device, dtype=torch.float32)
