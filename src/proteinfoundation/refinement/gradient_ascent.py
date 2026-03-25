"""Gradient ascent refinement in latent space with line search."""

import os
import shutil
import tempfile

import numpy as np
import torch
import torch.nn as nn
from biotite.structure import AtomArray
from jaxtyping import Bool, Float
from torch import Tensor

from foundry.utils.ddp import RankedLogger
from proteinfoundation.refinement.base_refinement import RefinementStrategy
from proteinfoundation.utils.coors_utils import nm_to_ang
from proteinfoundation.utils.pdb_utils import get_chain_ids_from_pdb, write_prot_to_pdb

logger = RankedLogger(__name__, rank_zero_only=True)


class GradientAscentRefinement(RefinementStrategy):
    """Gradient ascent refinement in latent space with adaptive line search.

    Optimizes protein designs by gradient ascent on latents (bb_ca + local_latents)
    using a differentiable reward model. Uses line search to adaptively select
    step sizes, which is more robust than fixed learning rates.

    Args:
        n_iterations: Number of gradient ascent iterations
        line_search_steps: Step sizes to test in line search
        update_bb_ca: Whether to update bb_ca coordinates (default: False, latents only)
        normalize_logits: Whether to normalize sequence logits before computing reward

    Example:
        >>> refinement = GradientAscentRefinement(
        ...     n_iterations=50,
        ...     line_search_steps=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2],
        ...     update_bb_ca=False,
        ... )
        >>> refined = refinement.refine(sample_prots, reward_model, autoencoder, batch)
    """

    def __init__(
        self,
        n_iterations: int = 50,
        line_search_steps: list[float] | None = None,
        update_bb_ca: bool = False,
        normalize_logits: bool = True,
    ):
        """Initialize gradient ascent refinement.

        Args:
            n_iterations: Number of optimization iterations
            line_search_steps: Step sizes for line search
            update_bb_ca: If True, optimize bb_ca; if False, only optimize local_latents
            normalize_logits: Apply normalization to sequence logits
        """
        self.n_iterations = n_iterations
        self.line_search_steps = line_search_steps or [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
        self.update_bb_ca = update_bb_ca
        self.normalize_logits = normalize_logits

    def refine_atom_arrays(
        self,
        atom_arrays: list[AtomArray],
        reward_model: "RewardModel",
        autoencoder: nn.Module | None = None,
    ) -> list[AtomArray]:
        """Refine AtomArrays using gradient ascent in latent space (STANDALONE interface).

        This is the primary interface for standalone usage on arbitrary structures.

        Args:
            atom_arrays: List of AtomArray structures to refine
            reward_model: Differentiable AF2-based reward model
            autoencoder: Frozen autoencoder (REQUIRED for gradient ascent)

        Returns:
            List of refined AtomArray structures

        Raises:
            ValueError: If reward model is not differentiable or autoencoder is None
        """
        self._check_differentiable(reward_model)

        if autoencoder is None:
            raise ValueError("Gradient ascent refinement requires autoencoder")

        logger.info(
            f"Starting gradient ascent refinement on {len(atom_arrays)} structures: "
            f"n_iterations={self.n_iterations}, update_bb_ca={self.update_bb_ca}"
        )

        # Convert AtomArrays to tensors for batch processing
        bb_ca, local_latents, mask, chain_info = self._encode_atom_arrays(atom_arrays, autoencoder)

        # Store original sequence for comparison
        with torch.no_grad():
            decoded_output = autoencoder.decode(
                z_latent=local_latents,
                ca_coors_nm=bb_ca,
                mask=mask,
            )
            orig_seq = decoded_output["residue_type"]

        # Enable gradients
        bb_ca.requires_grad_(True)
        local_latents.requires_grad_(True)

        _batch_size = bb_ca.shape[0]
        temp_dir = tempfile.mkdtemp()

        try:
            for iteration in range(self.n_iterations):
                if iteration % 10 == 0:
                    logger.info(f"Refinement iteration {iteration + 1}/{self.n_iterations}")

                # Compute gradients
                bb_ca_grads, local_latents_grads = self._compute_gradients(
                    bb_ca, local_latents, mask, reward_model, autoencoder, chain_info, temp_dir, iteration, orig_seq
                )

                # Line search to find best step size
                best_bb_ca, best_local_latents = self._line_search(
                    bb_ca,
                    local_latents,
                    bb_ca_grads,
                    local_latents_grads,
                    mask,
                    reward_model,
                    autoencoder,
                    chain_info,
                    temp_dir,
                    iteration,
                )

                # Update latents
                with torch.no_grad():
                    bb_ca.copy_(best_bb_ca)
                    local_latents.copy_(best_local_latents)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Convert refined latents back to AtomArrays
        refined_atom_arrays = self._decode_to_atom_arrays(
            bb_ca.detach(), local_latents.detach(), mask, autoencoder, atom_arrays
        )

        logger.info("Gradient ascent refinement complete")
        return refined_atom_arrays

    def _encode_atom_arrays(
        self,
        atom_arrays: list[AtomArray],
        autoencoder: nn.Module,
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Encode AtomArrays to latents using atomworks encoding.

        Returns:
            Tuple of (bb_ca, local_latents, mask, chain_info dict)
        """
        from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
        from atomworks.ml.transforms.encoding import atom_array_to_encoding

        batch_coors = []
        batch_residue_types = []
        batch_masks = []
        chain_infos = []

        # Find max length for padding
        max_len = 0
        for aa in atom_arrays:
            encoding = atom_array_to_encoding(aa, UNIFIED_ATOM37_ENCODING)
            max_len = max(max_len, encoding["xyz"].shape[0])

        for atom_array in atom_arrays:
            # Use atomworks encoding
            encoding = atom_array_to_encoding(atom_array, UNIFIED_ATOM37_ENCODING)

            coors = encoding["xyz"]  # [n_tokens, 37, 3]
            residue_types = encoding["seq"]  # [n_tokens]
            _coord_mask = encoding["mask"]  # [n_tokens, 37]

            n_tokens = coors.shape[0]

            # Pad to max_len
            coors_padded = np.zeros((max_len, 37, 3), dtype=np.float32)
            residue_types_padded = np.zeros(max_len, dtype=np.int64)
            mask_padded = np.zeros(max_len, dtype=bool)

            coors_padded[:n_tokens] = coors
            residue_types_padded[:n_tokens] = residue_types
            mask_padded[:n_tokens] = True

            # Extract chain info
            chain_ids = []
            seen_res = set()
            for res_id, chain_id in zip(atom_array.res_id, atom_array.chain_id, strict=False):
                if (res_id, chain_id) not in seen_res:
                    seen_res.add((res_id, chain_id))
                    chain_ids.append(chain_id)

            chain_info = {
                "target_chain": chain_ids[0] if chain_ids else "A",
                "binder_chain": chain_ids[-1] if len(chain_ids) > 1 else "B",
                "chain_ids": chain_ids,
            }

            batch_coors.append(torch.tensor(coors_padded, dtype=torch.float32))
            batch_residue_types.append(torch.tensor(residue_types_padded, dtype=torch.long))
            batch_masks.append(torch.tensor(mask_padded, dtype=torch.bool))
            chain_infos.append(chain_info)

        # Stack to batches
        coors_batch = torch.stack(batch_coors)  # [batch, max_len, 37, 3]
        residue_type_batch = torch.stack(batch_residue_types)
        mask_batch = torch.stack(batch_masks)

        # Move to device
        device = next(autoencoder.parameters()).device
        coors_batch = coors_batch.to(device)
        residue_type_batch = residue_type_batch.to(device)
        mask_batch = mask_batch.to(device)

        # Convert Angstroms to nm and extract CA
        coors_nm = coors_batch / 10.0
        bb_ca = coors_nm[:, :, 1, :].clone()  # [batch, n, 3]

        # Encode to latents
        encoded_batch = {
            "coords_nm": coors_nm,
            "coord_mask": torch.ones_like(coors_nm[..., 0]).bool(),
            "residue_type": residue_type_batch,
            "mask": mask_batch,
        }

        with torch.no_grad():
            encoded_output = autoencoder.encode(encoded_batch)
            local_latents = encoded_output["z_latent"].clone()

        # Combine chain info
        combined_chain_info = {
            "chain_infos": chain_infos,
            "target_chain": chain_infos[0]["target_chain"],
            "binder_chain": chain_infos[0]["binder_chain"],
        }

        return bb_ca, local_latents, mask_batch, combined_chain_info

    def _decode_to_atom_arrays(
        self,
        bb_ca: Tensor,
        local_latents: Tensor,
        mask: Tensor,
        autoencoder: nn.Module,
        original_atom_arrays: list[AtomArray],
    ) -> list[AtomArray]:
        """Decode latents back to AtomArrays using atomworks utilities.

        Returns:
            List of refined AtomArray structures
        """
        from proteinfoundation.utils.pdb_utils import atom37_to_atom_array

        # Decode latents
        decoded_output = autoencoder.decode(
            z_latent=local_latents,
            ca_coors_nm=bb_ca,
            mask=mask,
        )

        # Convert nm to Angstroms
        coors_atom37 = (decoded_output["coors_nm"] * 10.0).detach().cpu().numpy()  # [batch, n, 37, 3]
        residue_types = decoded_output["residue_type"].detach().cpu().numpy()  # [batch, n]

        # Reconstruct AtomArrays
        refined_arrays = []
        for i, original_array in enumerate(original_atom_arrays):
            # Extract chain info from original
            chain_ids = []
            seen_res = set()
            for res_id, chain_id in zip(original_array.res_id, original_array.chain_id, strict=False):
                if (res_id, chain_id) not in seen_res:
                    seen_res.add((res_id, chain_id))
                    chain_ids.append(ord(chain_id) - ord("A"))

            # Get valid (non-padded) region
            n_valid = mask[i].sum().item()

            # Convert to AtomArray
            refined_array = atom37_to_atom_array(
                coords_atom37=coors_atom37[i, :n_valid],
                residue_type=residue_types[i, :n_valid],
                chain_index=np.array(chain_ids[:n_valid]) if chain_ids else None,
            )

            refined_arrays.append(refined_array)

        return refined_arrays

    @torch.enable_grad()
    def _compute_gradients(
        self,
        bb_ca: Float[Tensor, "batch n 3"],
        local_latents: Float[Tensor, "batch n latent_dim"],
        mask: Bool[Tensor, "batch n"],
        reward_model: "RewardModel",
        autoencoder: nn.Module,
        chain_info: dict,
        temp_dir: str,
        iteration: int,
        orig_seq: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute gradients of reward w.r.t. latents.

        Returns:
            Tuple of (bb_ca_grads, local_latents_grads)
        """
        # Decode latents to protein
        decoded_output = autoencoder.decode(
            z_latent=local_latents,
            ca_coors_nm=bb_ca,
            mask=mask,
        )

        # Get sequence logits and structures
        seq_logits = decoded_output["seq_logits"]  # [batch, n, 20]
        local_seq = decoded_output["residue_type"]
        coors = nm_to_ang(decoded_output["coors_nm"])

        batch_size = bb_ca.shape[0]
        all_seq_grads = []

        # Compute reward with gradients for each sample
        for batch_idx in range(batch_size):
            # Log sequence changes
            diff_count = (local_seq[batch_idx] != orig_seq[batch_idx]).sum()
            if iteration % 10 == 0:
                logger.debug(
                    f"Sample {batch_idx}: {diff_count}/{local_seq[batch_idx].shape[0]} "
                    f"sequence differences from original"
                )

            # Write PDB
            pdb_filename = os.path.join(temp_dir, f"refinement_iter_{iteration}_batch_{batch_idx}.pdb")

            # Determine chain indices
            _target_chain = chain_info.get("target_chain", "A")
            _binder_chain = chain_info.get("binder_chain", "B")
            n = mask[batch_idx].sum().item()

            # Simple chain assignment (first half=target, second half=binder)
            # TODO: Use actual chain info from atom_arrays
            chain_index = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))

            write_prot_to_pdb(
                prot_pos=coors[batch_idx].detach().cpu().numpy(),
                file_path=pdb_filename,
                aatype=local_seq[batch_idx].detach().cpu().numpy(),
                chain_index=chain_index if len(set(chain_index)) > 1 else None,
                overwrite=True,
                no_indexing=True,
            )

            # Get chain IDs from PDB
            target_chain, binder_chain = get_chain_ids_from_pdb(pdb_filename)

            # Prepare sequence logits for binder only
            sample_mask = mask[batch_idx].bool()
            valid_seq_logits = seq_logits[batch_idx][sample_mask]  # [valid_len, 20]

            # Identify binder region (simple heuristic for now)
            binder_mask = (
                chain_index == binder_chain if chain_index is not None else np.ones(len(chain_index), dtype=bool)
            )
            binder_seq_logits = valid_seq_logits[binder_mask[: valid_seq_logits.shape[0]]]

            # Normalize logits
            if self.normalize_logits:
                binder_seq_logits = self._normalize_logits(binder_seq_logits, iteration)

            # Compute temperature annealing (soft → hard transition)
            soft = 1.0
            hard = 0.0
            temp = 0.01 + (1.0 - 0.01) * (1 - (iteration + 1) / (self.n_iterations + 1)) ** 2

            # Compute weighted reward + gradients using RewardModel
            # This uses the configured weights to aggregate both metrics and gradients
            reward_dict = reward_model.compute_with_gradients(
                pdb_path=pdb_filename,
                target_chain=target_chain,
                binder_chain=binder_chain,
                binder_sequence_logits=binder_seq_logits.detach().cpu().numpy(),
                soft=soft,
                hard=hard,
                temp=temp,
            )

            reward = reward_dict["total_reward"]
            if iteration % 10 == 0:
                logger.info(f"Sample {batch_idx} iteration {iteration + 1}: reward={reward:.4f}")

            # Extract weighted gradients
            if "grad" in reward_dict and "sequence" in reward_dict["grad"]:
                binder_grad = torch.as_tensor(
                    reward_dict["grad"]["sequence"], device=seq_logits.device
                )  # [n_binder, 20]

                # Pad to full sequence length
                valid_seq_grad = torch.zeros_like(valid_seq_logits)
                valid_seq_grad[binder_mask[: valid_seq_logits.shape[0]]] = binder_grad

                padded_seq_grad = torch.zeros_like(seq_logits[batch_idx])
                padded_seq_grad[sample_mask] = valid_seq_grad
                all_seq_grads.append(padded_seq_grad)
            else:
                all_seq_grads.append(torch.zeros_like(seq_logits[batch_idx]))

            # Cleanup PDB
            if os.path.exists(pdb_filename):
                os.remove(pdb_filename)

        # Stack gradients
        batch_seq_grads = torch.stack(all_seq_grads, dim=0)  # [batch, n, 20]

        # Backpropagate through decoder
        bb_ca_grads, local_latents_grads = torch.autograd.grad(
            outputs=seq_logits,
            inputs=[bb_ca, local_latents],
            grad_outputs=batch_seq_grads,
            retain_graph=False,
            create_graph=False,
        )

        return bb_ca_grads, local_latents_grads

    def _normalize_logits(
        self,
        logits: Float[Tensor, "n 20"],
        iteration: int,
    ) -> Float[Tensor, "n 20"]:
        """Normalize sequence logits.

        Uses mean-max normalization to keep logits in reasonable range.
        """
        mean = logits.mean(dim=-1, keepdim=True)
        maxx = logits.max(dim=-1, keepdim=True).values
        normalized = (logits - mean) / (maxx - mean + 1e-8)
        return normalized

    def _line_search(
        self,
        bb_ca: Tensor,
        local_latents: Tensor,
        bb_ca_grads: Tensor,
        local_latents_grads: Tensor,
        mask: Tensor,
        reward_model: "RewardModel",
        autoencoder: nn.Module,
        chain_info: dict,
        temp_dir: str,
        iteration: int,
    ) -> tuple[Tensor, Tensor]:
        """Perform line search to find optimal step size.

        Tests multiple step sizes and selects the one with highest reward.

        Returns:
            Tuple of (best_bb_ca, best_local_latents)
        """
        batch_size = bb_ca.shape[0]

        with torch.no_grad():
            # Generate candidate latents for each step size
            bb_ca_candidates = []
            local_latents_candidates = []

            for step_size in self.line_search_steps:
                if self.update_bb_ca:
                    bb_ca_candidates.append(bb_ca + step_size * bb_ca_grads)
                else:
                    bb_ca_candidates.append(bb_ca.clone())  # Don't update bb_ca

                local_latents_candidates.append(local_latents + step_size * local_latents_grads)

            # Evaluate reward for each candidate
            rewards_per_step = []
            for step_idx in range(len(self.line_search_steps)):
                rewards = self._evaluate_candidates(
                    bb_ca_candidates[step_idx],
                    local_latents_candidates[step_idx],
                    mask,
                    reward_model,
                    autoencoder,
                    chain_info,
                    temp_dir,
                    iteration,
                )
                rewards_per_step.append(rewards)

            # Select best step size per sample
            best_bb_ca_list = []
            best_local_latents_list = []

            for sample_idx in range(batch_size):
                # Extract rewards for this sample across all step sizes
                sample_rewards = [rewards[sample_idx] for rewards in rewards_per_step]

                # Find best step (maximize reward)
                best_step_idx = sample_rewards.index(max(sample_rewards))

                if iteration % 10 == 0:
                    logger.info(
                        f"Sample {sample_idx}: "
                        f"rewards={[round(r, 4) for r in sample_rewards]}, "
                        f"best_step={self.line_search_steps[best_step_idx]}, "
                        f"best_reward={sample_rewards[best_step_idx]:.4f}"
                    )

                best_bb_ca_list.append(bb_ca_candidates[best_step_idx][sample_idx])
                best_local_latents_list.append(local_latents_candidates[best_step_idx][sample_idx])

            # Stack back to batch
            best_bb_ca = torch.stack(best_bb_ca_list, dim=0)
            best_local_latents = torch.stack(best_local_latents_list, dim=0)

        return best_bb_ca, best_local_latents

    def _evaluate_candidates(
        self,
        bb_ca: Tensor,
        local_latents: Tensor,
        mask: Tensor,
        reward_model: "RewardModel",
        autoencoder: nn.Module,
        chain_info: dict,
        temp_dir: str,
        iteration: int,
    ) -> list[float]:
        """Evaluate reward for candidate latents.

        Returns:
            List of rewards (floats, one per sample in batch)
        """
        batch_size = bb_ca.shape[0]

        # Decode latents
        decoded_output = autoencoder.decode(
            z_latent=local_latents,
            ca_coors_nm=bb_ca,
            mask=mask,
        )

        coors = nm_to_ang(decoded_output["coors_nm"])
        residue_types = decoded_output["residue_type"]

        rewards = []
        for sample_idx in range(batch_size):
            pdb_filename = os.path.join(temp_dir, f"line_search_iter_{iteration}_sample_{sample_idx}.pdb")

            # Get chain info for this sample
            _target_chain = chain_info.get("target_chain", "A")
            _binder_chain = chain_info.get("binder_chain", "B")

            n = mask[sample_idx].sum().item()
            chain_index = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2))

            write_prot_to_pdb(
                prot_pos=coors[sample_idx].detach().cpu().numpy(),
                file_path=pdb_filename,
                aatype=residue_types[sample_idx].detach().cpu().numpy(),
                chain_index=chain_index if len(set(chain_index)) > 1 else None,
                overwrite=True,
                no_indexing=True,
            )

            # Compute reward (without gradients for line search - faster)
            from biotite.structure.io import load_structure

            atom_array = load_structure(pdb_filename)

            # Use standard non-differentiable compute (much faster)
            total_reward = reward_model.aggregate_metrics_output(
                reward_model.metric_manager(predicted_atom_array=atom_array)
            )
            rewards.append(total_reward)

            # Cleanup
            if os.path.exists(pdb_filename):
                os.remove(pdb_filename)

        return rewards
