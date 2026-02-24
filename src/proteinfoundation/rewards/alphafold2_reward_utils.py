"""Utility functions for AF2 reward model using ColabDesign."""

import os
import warnings

import jax.numpy as jnp
import numpy as np
from biotite.structure import AtomArray
from foundry.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*divide by zero.*")
warnings.filterwarnings("ignore", message=".*invalid value.*")


def get_af2_model_runner(
    model_name: str = "model_1_ptm",
    num_recycles: int = 3,
    use_templates: bool = False,
    data_dir: str | None = None,
    protocol: str = "hallucination",
):
    """Initialize AF2 model runner using ColabDesign.

    Args:
        model_name: AF2 model name. Defaults to ``"model_1_ptm"``.
        num_recycles: Number of recycling iterations. Defaults to ``3``.
        use_templates: Whether to use templates. Defaults to ``False``.
        data_dir: Path to AF2 parameters directory. If None, uses DATA_PATH env var.
        protocol: Protocol to use ("hallucination" or "binder"). Defaults to ``"hallucination"``.

    Returns:
        ColabDesign AF model object
    """
    try:
        from colabdesign import mk_afdesign_model
    except ImportError:
        raise ImportError("ColabDesign not installed. Please install it to use AF2 reward model.")

    # Get AF2 parameters directory
    if data_dir is None:
        data_path = os.environ.get("DATA_PATH", ".")
        af_params_dir = os.path.join(data_path, "tools/AF2")
    else:
        af_params_dir = data_dir

    if not os.path.exists(af_params_dir):
        raise ValueError(f"AF2 parameters directory not found: {af_params_dir}")

    # Initialize model using mk_afdesign_model (same as public codebase)
    af_model = mk_afdesign_model(
        protocol=protocol,
        num_recycles=num_recycles,
        data_dir=af_params_dir,
        use_templates=use_templates,
    )

    return af_model


def atom_array_to_sequence(atom_array: AtomArray) -> str:
    """Extract amino acid sequence from AtomArray.

    Args:
        atom_array: Input structure

    Returns:
        Single-letter amino acid sequence
    """

    # Get unique residues in order
    residues = []
    seen_res = set()

    for res_id, res_name in zip(atom_array.res_id, atom_array.res_name, strict=False):
        key = (res_id, res_name)
        if key not in seen_res:
            seen_res.add(key)
            residues.append(res_name)

    # Convert 3-letter codes to 1-letter
    sequence = ""
    for res in residues:
        # Simple mapping - can be extended
        mapping = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
            "UNK": "X",
        }
        sequence += mapping.get(res, "X")

    return sequence


def get_atom37_coords_from_atom_array(atom_array: AtomArray) -> np.ndarray:
    """Convert AtomArray to atom37 coordinate representation.

    Args:
        atom_array: Input structure

    Returns:
        Coordinates in atom37 format (N_res, 37, 3)
    """
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
    from atomworks.ml.transforms.encoding import atom_array_to_encoding

    # Convert to encoding
    encoding = atom_array_to_encoding(atom_array, UNIFIED_ATOM37_ENCODING)

    # Extract coordinates
    coords = encoding["coordinates"]  # Shape: (N_res, 37, 3)

    return np.array(coords)


def setup_af2_model_for_sequence(
    af_model,
    sequence: str,
    chain_breaks: list[int] | None = None,
):
    """Set up AF2 model with a specific sequence.

    Args:
        af_model: ColabDesign AF model object
        sequence: Amino acid sequence
        chain_breaks: List of residue indices where chain breaks occur

    Returns:
        Prepared model ready for prediction
    """
    # Prepare sequence (ColabDesign format)
    if chain_breaks:
        # Insert chain break markers
        seq_list = list(sequence)
        for idx in sorted(chain_breaks, reverse=True):
            seq_list.insert(idx, ":")
        sequence = "".join(seq_list)

    # Set sequence
    af_model.prep_inputs(sequence=sequence)

    return af_model


def extract_af2_confidence_metrics(result: dict) -> dict[str, float]:
    """Extract confidence metrics from AF2 prediction result.

    Args:
        result: AF2 prediction result dictionary

    Returns:
        Dictionary of confidence metrics
    """
    metrics = {}

    # pLDDT (0-100 scale)
    if "plddt" in result:
        metrics["plddt"] = float(jnp.mean(result["plddt"]))

    # pTM (0-1 scale)
    if "ptm" in result:
        metrics["ptm"] = float(result["ptm"])

    # PAE-based metrics
    if "pae" in result:
        pae = result["pae"]
        metrics["pae"] = float(jnp.mean(pae))
        metrics["max_pae"] = float(jnp.max(pae))

    # Interface PAE (i_pae) - requires chain information
    # This needs to be computed separately based on chain indices

    return metrics


def compute_interface_pae(pae: jnp.ndarray, chain_indices: list[int]) -> float:
    """Compute interface PAE between chains.

    Args:
        pae: PAE matrix (N_res, N_res)
        chain_indices: List indicating chain ID for each residue

    Returns:
        Mean interface PAE
    """
    chain_indices = jnp.array(chain_indices)
    n_chains = int(jnp.max(chain_indices)) + 1

    if n_chains < 2:
        return float(jnp.mean(pae))

    # Compute inter-chain PAE
    interface_values = []
    for i in range(n_chains):
        for j in range(i + 1, n_chains):
            mask_i = chain_indices == i
            mask_j = chain_indices == j

            # PAE from chain i to chain j and vice versa
            pae_ij = pae[mask_i][:, mask_j]
            pae_ji = pae[mask_j][:, mask_i]

            interface_values.append(jnp.mean(pae_ij))
            interface_values.append(jnp.mean(pae_ji))

    return float(jnp.mean(jnp.array(interface_values)))


def run_af2_with_gradients(
    af_model,
    pdb_path: str,
    target_chain: str,
    binder_chain: str,
    binder_sequence_logits: jnp.ndarray | None = None,
    soft: float = 1.0,
    hard: float = 0.0,
    temp: float = 1.0,
    num_recycles: int = 3,
) -> dict:
    """Run AF2 prediction with sequence gradients for refinement.

    Supports both soft (logits-based) and hard (discrete) optimization modes.

    Args:
        af_model: ColabDesign AF model instance
        pdb_path: Path to PDB file with structure
        target_chain: Target chain ID (e.g., 'A')
        binder_chain: Binder chain ID (e.g., 'B')
        binder_sequence_logits: Sequence logits for binder [n_binder, 20]
        soft: Weight for soft optimization (logits-based)
        hard: Weight for hard optimization (discrete)
        temp: Temperature for soft optimization
        num_recycles: Number of AF2 recycles

    Returns:
        Dictionary containing:
            - 'reward': Aggregated scalar reward
            - 'total_reward': Same as reward
            - 'grad': Dict with 'sequence' gradients [n_binder, 20]
            - Individual metrics: 'plddt', 'ptm', 'pae', 'ipae', etc.
    """
    from biotite.structure.io import load_structure

    # Load structure
    atom_array = load_structure(pdb_path)

    # Extract full sequence
    sequence = atom_array_to_sequence(atom_array)

    # Identify target and binder residues
    chain_ids_full = []
    seen_res = set()
    for res_id, chain_id in zip(atom_array.res_id, atom_array.chain_id, strict=False):
        key = (res_id, chain_id)
        if key not in seen_res:
            seen_res.add(key)
            chain_ids_full.append(chain_id)

    # Find binder region
    binder_mask = jnp.array([cid == binder_chain for cid in chain_ids_full])
    binder_indices = jnp.where(binder_mask)[0]

    # Prepare inputs
    af_model.prep_inputs(
        length=len(sequence),
        seed=0,
    )

    # Set sequence with gradient support
    if soft > 0 and binder_sequence_logits is not None:
        # Soft mode: Use sequence logits for binder
        # Build full sequence logits (one-hot for target, provided for binder)
        full_seq_logits = jnp.zeros((len(sequence), 20))

        # One-hot encode target
        for i, (cid, aa) in enumerate(zip(chain_ids_full, sequence, strict=False)):
            if cid != binder_chain:
                aa_idx = "ARNDCQEGHILKMFPSTWYV".find(aa)
                if aa_idx >= 0:
                    full_seq_logits = full_seq_logits.at[i, aa_idx].set(1.0)

        # Set binder logits
        full_seq_logits = full_seq_logits.at[binder_indices].set(binder_sequence_logits)

        # Normalize with temperature
        full_seq_logits = full_seq_logits / temp

        af_model.set_seq(seq=full_seq_logits, mode="soft")
    else:
        # Hard mode: Use discrete sequence
        af_model.set_seq(sequence)

    # Mix soft and hard if both specified
    if 0 < soft < 1 and 0 < hard < 1:
        # Blend soft and hard modes
        af_model._params["soft"] = soft
        af_model._params["hard"] = hard

    # Run with backprop enabled
    af_model.run(
        num_recycles=num_recycles,
        sample_models=False,
        models=[0],
        backprop=True,  # Enable gradients
    )

    # Extract results
    aux = af_model.aux

    # Compute metrics
    metrics = extract_af2_confidence_metrics(aux)

    # Compute interface PAE if multi-chain
    chain_ids_numeric = [ord(cid) - ord("A") for cid in chain_ids_full]
    if len(set(chain_ids_numeric)) > 1 and "pae" in aux:
        pae = aux["pae"]
        metrics["ipae"] = compute_interface_pae(pae, chain_ids_numeric)

    # Extract gradients if in soft mode
    grad_dict = {}
    if soft > 0 and "grad" in aux:
        # Extract binder sequence gradients
        full_grad = aux["grad"]["seq"]  # [n_total, 20]
        binder_grad = full_grad[binder_indices]  # [n_binder, 20]
        grad_dict["sequence"] = binder_grad

    # Return metrics and gradients (NO hardcoded reward aggregation)
    # Reward aggregation should be done by RewardModel using configured weights
    result = {
        "metrics": metrics,
        "grad": grad_dict,
    }

    # Clean up
    if hasattr(af_model, "_inputs"):
        af_model._inputs.clear()
    if hasattr(af_model, "aux"):
        af_model.aux.clear()

    return result
