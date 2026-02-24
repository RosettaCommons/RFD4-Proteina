"""BindCraft refinement protocol for sequence optimization."""

import os
import shutil
import tempfile
from typing import Literal

import numpy as np
import torch
from biotite.structure import AtomArray
from foundry.utils.ddp import RankedLogger

from proteinfoundation.refinement.base_refinement import RefinementStrategy
from proteinfoundation.utils.pdb_utils import get_chain_ids_from_pdb, write_prot_to_pdb

logger = RankedLogger(__name__, rank_zero_only=True)


class BindCraftRefinement(RefinementStrategy):
    """BindCraft-style sequence optimization with fixed structure.

    Optimizes binder sequence while keeping structure fixed, using AF2 gradients
    on sequence logits. Supports two operational modes:

    - **full**: Soft optimization (temperature annealing) + hard optimization + greedy
    - **hard_only**: Hard optimization + greedy (faster)

    Args:
        mode: Optimization mode ('full' or 'hard_only')
        n_soft_iters: Soft optimization iterations (temperature annealing)
        n_hard_iters: Hard optimization iterations (discrete mutations)
        n_greedy_iters: Greedy search iterations (local mutations)
        n_recycles: AF2 recycles per iteration

    Example:
        >>> # Full BindCraft protocol
        >>> refinement = BindCraftRefinement(mode="full", n_soft_iters=45, n_hard_iters=5)
        >>> refined = refinement.refine(sample_prots, reward_model, autoencoder, batch)

        >>> # Fast hard-only mode
        >>> refinement = BindCraftRefinement(mode="hard_only", n_hard_iters=10)
        >>> refined = refinement.refine(sample_prots, reward_model, autoencoder, batch)

    Reference:
        BindCraft: https://github.com/martinpacesa/BindCraft
    """

    def __init__(
        self,
        mode: Literal["full", "hard_only"] = "full",
        n_soft_iters: int = 45,
        n_hard_iters: int = 5,
        n_greedy_iters: int = 15,
        n_recycles: int = 3,
    ):
        """Initialize BindCraft refinement.

        Args:
            mode: 'full' (all phases) or 'hard_only' (skip soft optimization)
            n_soft_iters: Soft optimization iterations with temperature annealing
            n_hard_iters: Hard optimization iterations with discrete sampling
            n_greedy_iters: Greedy local search iterations
            n_recycles: Number of AF2 recycles per evaluation
        """
        self.mode = mode
        self.n_soft_iters = n_soft_iters
        self.n_hard_iters = n_hard_iters
        self.n_greedy_iters = n_greedy_iters
        self.n_recycles = n_recycles

    def refine_atom_arrays(
        self,
        atom_arrays: list[AtomArray],
        reward_model,
        autoencoder=None,
    ) -> list[AtomArray]:
        """Refine AtomArrays using BindCraft protocol (STANDALONE interface).

        Optimizes sequence while keeping structure fixed. This is the primary
        interface for standalone usage on arbitrary PDB/CIF files.

        Args:
            atom_arrays: List of AtomArray structures to refine
            reward_model: Differentiable AF2-based reward model
            autoencoder: Not used (BindCraft doesn't require autoencoder)

        Returns:
            List of refined AtomArray structures with optimized sequences

        Raises:
            ValueError: If reward model is not differentiable
        """
        self._check_differentiable(reward_model)

        logger.info(
            f"Starting BindCraft refinement on {len(atom_arrays)} structures: "
            f"mode={self.mode}, n_soft={self.n_soft_iters}, "
            f"n_hard={self.n_hard_iters}, n_greedy={self.n_greedy_iters}"
        )

        refined_arrays = []
        temp_dir = tempfile.mkdtemp()

        try:
            # Refine each structure independently
            for idx, atom_array in enumerate(atom_arrays):
                logger.info(f"Refining structure {idx + 1}/{len(atom_arrays)}")
                refined = self._refine_single_structure(atom_array, reward_model, temp_dir, idx)
                refined_arrays.append(refined)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        logger.info("BindCraft refinement complete")
        return refined_arrays

    def _refine_single_structure(
        self,
        atom_array: AtomArray,
        reward_model,
        temp_dir: str,
        structure_idx: int,
    ) -> AtomArray:
        """Refine a single structure using BindCraft protocol with ColabDesign.

        Args:
            atom_array: Structure to refine
            reward_model: Differentiable reward model (must have AF2Metric)
            temp_dir: Temporary directory for PDB files
            structure_idx: Index for naming temp files

        Returns:
            Refined AtomArray
        """
        import jax
        from biotite.structure.io import load_structure
        from colabdesign import mk_afdesign_model

        # Write initial PDB
        pdb_path = os.path.join(temp_dir, f"structure_{structure_idx}.pdb")
        self._write_atom_array_to_pdb(atom_array, pdb_path)

        # Get chain IDs
        target_chain, binder_chain = get_chain_ids_from_pdb(pdb_path)

        # Initialize ColabDesign model
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        device = jax.devices("gpu")[device_id] if torch.cuda.is_available() else jax.devices("cpu")[0]

        # Get AF2 data dir from reward model's AF2Metric
        af2_metric = reward_model.metric_manager.metrics.get("af2")
        if af2_metric is None:
            raise ValueError("BindCraft requires AF2Metric in reward model")

        data_dir = af2_metric.data_dir or os.path.join(os.environ.get("DATA_PATH", "."), "tools/AF2")

        af_model = mk_afdesign_model(
            protocol="binder",
            debug=False,
            data_dir=data_dir,
            use_multimer=True,
            num_recycles=self.n_recycles,
            use_initial_guess=False,
            use_initial_atom_pos=False,
            best_metric="loss",
            device=device,
        )

        # Prepare inputs
        af_model.prep_inputs(
            pdb_filename=pdb_path,
            target_chain=target_chain,
            binder_chain=binder_chain,
            mode="wildtype",
            rm_target=False,
            rm_target_seq=False,
            rm_target_sc=False,
            hotspot=None,
            use_binder_template=True,
            rm_template_ic=True,
        )

        # Set loss weights (use reward model weights if available, else defaults)
        af_model.opt["weights"].update(
            {
                "pae": 0.4,
                "plddt": 0.1,
                "i_pae": 0.1,
                "con": 1.0,
                "i_con": 1.0,
                "dgram_cce": 0.0,
            }
        )

        # Run BindCraft phases based on mode
        if self.mode == "full":
            # Stage 2: Softmax optimization with temperature annealing
            logger.info(f"  Stage 2: Softmax optimization ({self.n_soft_iters} iters)")
            af_model.design_soft(
                self.n_soft_iters,
                e_temp=1e-2,
                models=[0],
                num_models=1,
                sample_models=False,
                ramp_recycles=False,
            )

            # Stage 3: One-hot optimization (straight-through estimator)
            logger.info(f"  Stage 3: One-hot optimization ({self.n_hard_iters} iters)")
            af_model.design_hard(
                self.n_hard_iters,
                temp=1e-2,
                models=[0],
                num_models=1,
                sample_models=False,
                dropout=False,
                ramp_recycles=False,
            )

        # Stage 4: PSSM semigreedy optimization (discrete)
        logger.info(f"  Stage 4: PSSM semigreedy optimization ({self.n_greedy_iters} iters)")

        # Count number of residues
        n_res = len(set(zip(atom_array.res_id, atom_array.chain_id, strict=False)))
        greedy_tries = max(1, int(n_res * 0.01))  # 1% of residues per iteration

        af_model.design_pssm_semigreedy(
            soft_iters=0,
            hard_iters=self.n_greedy_iters,
            tries=greedy_tries,
            models=[0],
            num_models=1,
            sample_models=False,
            ramp_models=False,
            save_best=True,
        )

        # Save refined structure
        refined_pdb_path = pdb_path.replace(".pdb", "_refined.pdb")
        af_model.save_pdb(refined_pdb_path)

        # Load refined structure
        refined_atom_array = load_structure(refined_pdb_path)

        # Cleanup
        if os.path.exists(refined_pdb_path):
            os.remove(refined_pdb_path)

        # Clear JAX memory
        jax.clear_backends()

        return refined_atom_array

    def _write_atom_array_to_pdb(self, atom_array: AtomArray, pdb_path: str):
        """Write AtomArray to PDB file using atomworks encoding.

        Args:
            atom_array: Structure to write
            pdb_path: Output PDB path
        """
        from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
        from atomworks.ml.transforms.encoding import atom_array_to_encoding

        # Convert to atom37 encoding
        encoding = atom_array_to_encoding(atom_array, UNIFIED_ATOM37_ENCODING)

        coords_atom37 = encoding["xyz"]  # [n_tokens, 37, 3]
        residue_types = encoding["seq"]  # [n_tokens]

        # Extract chain IDs
        chain_ids = []
        seen_res = set()
        for res_id, chain_id in zip(atom_array.res_id, atom_array.chain_id, strict=False):
            if (res_id, chain_id) not in seen_res:
                seen_res.add((res_id, chain_id))
                chain_ids.append(chain_id)

        write_prot_to_pdb(
            prot_pos=coords_atom37,
            file_path=pdb_path,
            aatype=residue_types,
            chain_index=np.array(chain_ids) if chain_ids else None,
            overwrite=True,
            no_indexing=True,
        )
