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

from functools import partial
from pathlib import Path
from typing import Any

import torch
from foundry.utils.ddp import RankedLogger

from proteinfoundation.metrics.metrics_utils import run_metrics_for_result
from proteinfoundation.search.base_strategy import SearchStrategy
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
    ):
        """Initialize beam search strategy.

        Args:
            step_checkpoints: Diffusion steps to evaluate at (K steps between each)
            n_branch: Candidates to generate per beam state
            beam_width: Candidates to keep after pruning
            keep_intermediate_samples: Save intermediate results
        """
        self.step_checkpoints = sorted(step_checkpoints)
        self.n_branch = n_branch
        self.beam_width = beam_width
        self.keep_intermediate_samples = keep_intermediate_samples

    def generate(self, model, batch: dict, reward_model=None) -> list[dict[str, Any]]:
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

        # Get simulation schedule
        ts, gt = model.fm.sample_schedule(
            nsteps=nsteps,
            sampling_model_args=model.inf_cfg.model,
        )
        simulation_step_params = {
            data_mode: model.inf_cfg.model[data_mode]["simulation_step_params"] for data_mode in model.fm.data_modes
        }

        # Initialize beam: nsamples * beam_width samples
        # Interleaved format: [sample0, sample1, sample2, sample0, sample1, sample2, ...]
        logger.info(
            f"Initializing beam: {nsamples} samples × {self.beam_width} width = {nsamples * self.beam_width} total states"
        )
        mask = init_mask.repeat(self.beam_width, 1)
        xt = model.fm.sample_noise(
            n,
            shape=(nsamples * self.beam_width,),
            device=device,
            mask=mask,
            training=False,
        )
        x_1_pred = None

        all_sample_prots = []

        # Beam search loop over checkpoints
        for i in range(len(self.step_checkpoints) - 1):
            start_step = self.step_checkpoints[i]
            end_step = self.step_checkpoints[i + 1]
            logger.info(f"Checkpoint {i+1}/{len(self.step_checkpoints)-1}: steps {start_step} → {end_step}")

            all_xt_branches = []
            all_x_1_pred_branches = []

            # Process each replica separately due to batch size constraints
            for replica_idx in range(self.beam_width):
                # Extract samples for this replica (interleaved format)
                start = replica_idx * nsamples
                end = (replica_idx + 1) * nsamples
                replica_xt = {key: value[start:end] for key, value in xt.items()}
                replica_x_1_pred = (
                    {key: value[start:end] for key, value in x_1_pred.items()} if x_1_pred is not None else None
                )

                # Generate n_branch branches from this replica
                for _ in range(self.n_branch):
                    # Run partial simulation for K steps
                    branch_xt, branch_x_1_pred, _ = model.fm.partial_simulation(
                        batch=batch,
                        x=replica_xt,
                        x_1_pred=replica_x_1_pred,
                        mask=init_mask,
                        predict_for_sampling=fn_predict_for_sampling,
                        nsteps= end_step - start_step,
                        ts=ts,
                        gt=gt,
                        self_cond=self_cond,
                        simulation_step_params=simulation_step_params,
                        device=model.device,
                        guidance_w=guidance_w,
                        ag_ratio=ag_ratio,
                    )

                    all_xt_branches.append(branch_xt)
                    all_x_1_pred_branches.append(branch_x_1_pred)
                    # Note: We discard extra_info here since we'll recreate it when formatting

            # Concatenate all replicas and branches
            xt = concat_dict_tensors(all_xt_branches, dim=0)

            x_1_pred = None
            if all_x_1_pred_branches[0] is not None:
                x_1_pred = concat_dict_tensors(all_x_1_pred_branches, dim=0)

            # After each step (except the last), evaluate rewards and select top samples
            if i < len(self.step_checkpoints) - 1:
                # Compute rewards by rolling out to completion
                final_sample_prots, rewards = self._simulate_and_compute_reward(
                    model, batch, xt, end_step, init_mask, reward_model
                )

                # Save intermediate samples if requested
                if self.keep_intermediate_samples:
                    reward_threshold = model.inf_cfg.search.get("reward_threshold", float("-inf"))
                    reward_mask = rewards > reward_threshold
                    if reward_mask.any():
                        returned_sample_prots = {k: v[reward_mask] for k, v in final_sample_prots.items()}
                        returned_sample_prots["rewards"] = rewards[reward_mask]
                        all_sample_prots.append(returned_sample_prots)
                        logger.info(
                            f"Saved {reward_mask.sum().item()} intermediate samples above threshold {reward_threshold}"
                        )

                # Reshape and select top beam_width candidates
                # Current shape: [n_branch * beam_width * nsamples] in interleaved format
                # Reshape to: [n_branch, beam_width, nsamples]
                # Then permute to: [nsamples, n_branch * beam_width]
                rewards_reshaped = rewards.view(self.n_branch, self.beam_width, nsamples)
                rewards_for_selection = rewards_reshaped.permute(2, 0, 1).reshape(
                    nsamples, self.n_branch * self.beam_width
                )

                # Top-k selection for beam search
                top_k_indices = torch.topk(rewards_for_selection, k=self.beam_width, dim=1)[1]

                # Convert flat indices to replica and branch indices
                replica_indices = top_k_indices % self.beam_width  # [nsamples, beam_width]
                branch_indices = top_k_indices // self.beam_width  # [nsamples, beam_width]

                logger.debug(f"Selected top {self.beam_width} candidates per sample")

                # Create sample indices for broadcasting
                sample_indices = torch.arange(nsamples, device=device).unsqueeze(1).expand(-1, self.beam_width)

                # Calculate global indices using vectorized operations
                global_indices = (
                    branch_indices * nsamples * self.beam_width + replica_indices * nsamples + sample_indices
                )

                # Transpose and flatten to get interleaved format [1,2,3,1,2,3]
                select_indices = global_indices.T.flatten()

                # Select samples
                xt = {key: value[select_indices] for key, value in xt.items()}
                if x_1_pred is not None:
                    x_1_pred = {key: value[select_indices] for key, value in x_1_pred.items()}

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

        # Compute final rewards
        final_rewards = self._compute_rewards_from_sample_prots(model, sample_prots, reward_model)
        sample_prots["rewards"] = final_rewards
        all_sample_prots.append(sample_prots)

        # Concatenate all samples (intermediate + final)
        all_sample_prots = concat_dict_tensors(all_sample_prots, dim=0)

        # Verify output shape
        if not self.keep_intermediate_samples:
            expected_sample_count = nsamples * self.beam_width
            actual_count = all_sample_prots["coors"].shape[0]
            assert (
                actual_count == expected_sample_count
            ), f"Expected {expected_sample_count} samples, got {actual_count}"

        logger.info(f"Beam search complete. Generated {all_sample_prots['coors'].shape[0]} samples")
        logger.info(
            f"Reward stats: mean={all_sample_prots['rewards'].mean().item():.4f}, "
            f"std={all_sample_prots['rewards'].std().item():.4f}"
        )

        # Convert to list format expected by generate.py using Proteina's helper method
        results = model.format_sample_prots_to_atom_arrays(all_sample_prots, batch, batch_idx=0)

        # Add rewards to results if available
        if "rewards" in all_sample_prots:
            for i, result in enumerate(results):
                result["reward"] = all_sample_prots["rewards"][i].item()

        return results

    def _simulate_and_compute_reward(
        self, model, batch: dict, xt: dict[str, torch.Tensor], current_step: int, mask: torch.Tensor, reward_model
    ) -> tuple[dict, torch.Tensor]:
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

        Returns:
            Tuple of (sample_prots dict, rewards tensor)
        """
        # Get final step from checkpoints
        final_step = self.step_checkpoints[-1]

        # Get batch parameters
        batch_size = next(iter(xt.values())).shape[0]
        segment_size = mask.shape[0]

        # Roll out from current_step to final_step to get clean samples
        logger.debug(f"Rolling out {batch_size} samples: step {current_step} → {final_step}")

        final_xt_batches = []

        # Process in segments to respect batch size constraints
        for i in range(0, batch_size, segment_size):
            end_idx = min(i + segment_size, batch_size)
            batch_xt = {key: value[i:end_idx] for key, value in xt.items()}

            # Get simulation schedule
            ts_sim, gt_sim = model.fm.sample_schedule(
                nsteps=model.inf_cfg.args.nsteps,
                sampling_model_args=model.inf_cfg.model,
            )

            batch_final_xt, _, _ = model.fm.partial_simulation(
                batch=batch,
                x=batch_xt,
                x_1_pred=None,
                mask=mask,
                predict_for_sampling=partial(model.predict_for_sampling, n_recycle=model.inf_cfg.get("n_recycle", 0)),
                nsteps = final_step - current_step,
                ts=ts_sim,
                gt=gt_sim,
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

        # Concatenate batches
        final_xt = concat_dict_tensors(final_xt_batches, dim=0)

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

        # Compute rewards R(x, s)
        rewards = self._compute_rewards_from_sample_prots(model, final_sample_prots, reward_model)

        logger.debug(
            f"Computed rewards: mean={rewards.mean().item():.4f}, "
            f"min={rewards.min().item():.4f}, max={rewards.max().item():.4f}"
        )

        return final_sample_prots, rewards

    def _compute_rewards_from_sample_prots(self, model, sample_prots: dict, reward_model) -> torch.Tensor:
        """Compute rewards for formatted sample proteins.

        Uses the model's format_sample_prots_to_atom_arrays helper to properly
        convert sample_prots to AtomArrays, then evaluates with reward model.

        Args:
            model: Proteina model instance
            sample_prots: Dictionary with keys 'coors', 'residue_type', 'padding_mask'
            reward_model: Reward model for scoring

        Returns:
            Tensor of reward values for each sample
        """
        batch_size = sample_prots["coors"].shape[0]
        device = sample_prots["coors"].device

        # set up all metrics
        for metric in reward_model.metric_manager.metrics.values():
            metric.setup()

        # Convert sample_prots to proper AtomArrays using Proteina's helper
        # NOTE: Expects interleaved format [1,2,3,1,2,3]!!!
        atom_array_results = model.format_sample_prots_to_atom_arrays(sample_prots, model._current_batch, batch_idx=0)

        batch = model._current_batch

        # Compute rewards for each atom array
        rewards = []
        for i, result in enumerate(atom_array_results):
            orig_batch_idx = i % max(len(batch.get("atom_array_batch", [batch.get("atom_array")])), 1)
            metrics_output = run_metrics_for_result(result, batch, orig_batch_idx, reward_model.metric_manager, output_base_dir=None)
            reward = reward_model.aggregate_metrics_output(metrics_output)
            rewards.append(reward)

        # tear down all metrics
        for metric in reward_model.metric_manager.metrics.values():
            metric.teardown()

        return torch.tensor(rewards, device=device, dtype=torch.float32)
