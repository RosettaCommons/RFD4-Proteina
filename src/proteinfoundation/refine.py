"""Standalone refinement script for optimizing protein structures.

Usage:
    # Refine single file with BindCraft
    python proteinfoundation/refine.py \
        --config-name refine_bindcraft \
        inputs=my_binder.pdb

    # Refine directory with Gradient Ascent  
    python proteinfoundation/refine.py \
        --config-name refine_gradient_ascent \
        inputs=outputs/generated_structures/

    # Override parameters
    python proteinfoundation/refine.py \
        --config-name refine_bindcraft \
        inputs=my_binder.pdb \
        refinement.strategy.n_soft_iters=30
"""

from pathlib import Path

import hydra
from biotite.structure import AtomArray
from biotite.structure.io import load_structure, save_structure
from foundryutils.ddp import RankedLogger
from omegaconf import DictConfig

logger = RankedLogger(__name__, rank_zero_only=True)


def load_structures_from_path(input_path: str | Path) -> list[tuple[AtomArray, str]]:
    """Load structures from file or directory.

    Args:
        input_path: Path to PDB/CIF file or directory

    Returns:
        List of (atom_array, filename) tuples
    """
    input_path = Path(input_path)
    structures = []

    if input_path.is_file():
        # Single file
        atom_array = load_structure(str(input_path))
        structures.append((atom_array, input_path.name))

    elif input_path.is_dir():
        # Directory - load all PDB/CIF files
        for file_path in sorted(input_path.glob("*.pdb")):
            try:
                atom_array = load_structure(str(file_path))
                structures.append((atom_array, file_path.name))
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        for file_path in sorted(input_path.glob("*.cif")):
            try:
                atom_array = load_structure(str(file_path))
                structures.append((atom_array, file_path.name))
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    logger.info(f"Loaded {len(structures)} structures from {input_path}")
    return structures


def save_refined_structures(
    refined_atom_arrays: list[AtomArray],
    original_filenames: list[str],
    output_dir: str | Path,
):
    """Save refined structures with _refined suffix.

    Args:
        refined_atom_arrays: List of refined AtomArrays
        original_filenames: Original filenames
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for atom_array, filename in zip(refined_atom_arrays, original_filenames, strict=False):
        # Add _refined suffix
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        output_filename = f"{stem}_refined{suffix}"
        output_path = output_dir / output_filename

        save_structure(str(output_path), atom_array)
        logger.info(f"Saved refined structure: {output_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="refine_bindcraft")
def main(cfg: DictConfig):
    """Main refinement entry point."""

    logger.info("=" * 80)
    logger.info("STANDALONE PROTEIN REFINEMENT")
    logger.info("=" * 80)

    # Load structures
    structures_with_names = load_structures_from_path(cfg.inputs)
    atom_arrays = [s[0] for s in structures_with_names]
    filenames = [s[1] for s in structures_with_names]

    # Instantiate refinement strategy
    refinement_strategy = hydra.utils.instantiate(cfg.refinement.strategy)
    logger.info(f"Refinement strategy: {type(refinement_strategy).__name__}")

    # Instantiate reward model
    from foundrymetrics.metric import MetricManager

    from proteinfoundation.rewards import RewardModel

    metric_manager = MetricManager.instantiate_from_hydra(cfg.rewards.metrics)
    reward_model = RewardModel(
        metrics=metric_manager,
        weights=dict(cfg.rewards.weights),
        differentiable=cfg.rewards.get("differentiable", False),
    )

    logger.info(f"Reward model: differentiable={reward_model.differentiable}")

    # Load autoencoder if needed
    autoencoder = None
    if hasattr(cfg, "autoencoder_ckpt_path") and cfg.autoencoder_ckpt_path is not None:
        from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder

        logger.info(f"Loading autoencoder from {cfg.autoencoder_ckpt_path}")
        autoencoder = AutoEncoder.load_from_checkpoint(cfg.autoencoder_ckpt_path)
        autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False
        logger.info("Autoencoder loaded and frozen")

    # Run refinement
    logger.info(f"Refining {len(atom_arrays)} structures...")
    refined_atom_arrays = refinement_strategy.refine_atom_arrays(
        atom_arrays=atom_arrays,
        reward_model=reward_model,
        autoencoder=autoencoder,
    )

    # Save refined structures
    output_dir = cfg.get("output_dir", "refined_structures")
    save_refined_structures(refined_atom_arrays, filenames, output_dir)

    logger.info("=" * 80)
    logger.info(f"✓ Refinement complete! Saved {len(refined_atom_arrays)} structures to {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
