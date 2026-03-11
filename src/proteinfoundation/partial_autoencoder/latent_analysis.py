"""Latent space analysis utilities for autoencoder.

This module provides tools for analyzing and visualizing the latent space
of the sidechain autoencoder using dimensionality reduction techniques.
"""

import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from foundry.utils.env import load_dotenv
from foundry.utils.ddp import RankedLogger

from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.partial_autoencoder.inference import load_dataloader, parse_args_and_cfg, setup
from proteinfoundation.partial_autoencoder.latent_viz_utils import (
    get_residue_type_labels,
    plot_latent_projection_with_labels,
    project_latents_2d,
)

logger = RankedLogger(__name__, rank_zero_only=True)

# Note: Color palettes, labels, projection, and plotting functions are now
# imported from latent_viz_utils to ensure consistency between training and inference


def extract_latents_from_predictions(
    predictions: list[tuple[dict, dict]], max_samples: int | None = None
) -> dict[str, np.ndarray]:
    """Extract latent vectors and metadata from prediction tuples.

    Args:
      predictions: List of (batch_input, batch_output) tuples from trainer.predict().
      max_samples: Maximum number of tokens to extract (subsample if needed).

    Returns:
      Dictionary with keys:
        - 'latents': [n_samples, latent_dim] latent vectors
        - 'residue_types': [n_samples] residue type indices
        - 'masks': [n_samples] boolean mask for valid tokens
        - 'n_residue_types': int, number of residue types (20 or 33)
    """
    logger.info(f"Extracting latents from {len(predictions)} batches")

    all_latents = []
    all_residue_types = []
    all_masks = []
    n_residue_types = None

    for batch_input, batch_output in predictions:
        # Extract latent vectors [batch, n_tokens, latent_dim]
        z_latent = batch_output["z_latent"].cpu().detach().numpy()

        # Extract residue types [batch, n_tokens]
        residue_type = batch_input["residue_type"].cpu().detach().numpy()

        # Extract mask [batch, n_tokens] - True = valid token
        mask = batch_input["padding_mask"].cpu().detach().numpy()

        # Detect n_residue_types from logits shape
        if n_residue_types is None and "seq_logits" in batch_output:
            n_residue_types = batch_output["seq_logits"].shape[-1]
            logger.info(f"Detected n_residue_types={n_residue_types} from logits shape")

        # Flatten batch dimension
        batch_size, n_tokens, latent_dim = z_latent.shape
        z_latent_flat = z_latent.reshape(-1, latent_dim)
        residue_type_flat = residue_type.reshape(-1)
        mask_flat = mask.reshape(-1)

        all_latents.append(z_latent_flat)
        all_residue_types.append(residue_type_flat)
        all_masks.append(mask_flat)

    # Stack all batches
    latents = np.vstack(all_latents)
    residue_types = np.concatenate(all_residue_types)
    masks = np.concatenate(all_masks)

    # Filter by mask (keep only valid tokens)
    valid_indices = masks > 0
    latents = latents[valid_indices]
    residue_types = residue_types[valid_indices]

    logger.info(f"Extracted {len(latents)} valid tokens with latent_dim={latents.shape[1]}")

    # Subsample if needed
    if max_samples is not None and len(latents) > max_samples:
        logger.info(f"Subsampling from {len(latents)} to {max_samples} tokens")
        indices = np.random.choice(len(latents), size=max_samples, replace=False)
        latents = latents[indices]
        residue_types = residue_types[indices]

    # Default to 20 if we couldn't detect
    if n_residue_types is None:
        n_residue_types = 20
        logger.warning("Could not detect n_residue_types from logits, defaulting to 20")

    return {
        "latents": latents,
        "residue_types": residue_types,
        "n_residue_types": n_residue_types,
    }


# Note: project_latents_2d now imported from latent_viz_utils


def plot_latent_projection(
    projection: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "Latent Space Projection",
    filter_residues: list[int] | None = None,
    n_residue_types: int = 20,
    method: str = "t-SNE",
) -> None:
    """Create scatter plot of 2D projection colored by residue type.

    Uses shared visualization utilities for consistency.

    Args:
      projection: [n_samples, 2] projected coordinates.
      labels: [n_samples] residue type indices.
      output_path: Path to save the plot.
      title: Plot title.
      filter_residues: List of residue type indices to include (None = all).
      n_residue_types: Number of residue types (20 or 33).
      method: Name of projection method for axis labels.
    """
    # Filter if requested
    if filter_residues is not None:
        mask = np.isin(labels, filter_residues)
        projection = projection[mask]
        labels = labels[mask]
        logger.info(f"Filtered to {len(projection)} tokens from residue types {filter_residues}")

    # Create plot using shared utility (with cluster labels!)
    fig = plot_latent_projection_with_labels(
        projection=projection,
        labels=labels,
        n_residue_types=n_residue_types,
        method=method,
        title=title,
        figsize=(14, 12),
        add_cluster_labels=True,
    )

    # Save to file
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot to {output_path}")


def run_latent_analysis(
    predictions: list[tuple[dict, dict]],
    output_dir: str,
    method: str = "tsne",
    n_samples: int = 10000,
    filter_residues: list[int] | None = None,
    **kwargs,
) -> None:
    """Run complete latent space analysis pipeline.

    Args:
      predictions: List of (batch_input, batch_output) tuples.
      output_dir: Directory to save results.
      method: Projection method ('tsne', 'pca', or 'lda').
      n_samples: Maximum number of tokens to analyze.
      filter_residues: Residue type indices to include (None = all).
      **kwargs: Additional arguments for projection method.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract latents
    data = extract_latents_from_predictions(predictions, max_samples=n_samples)
    latents = data["latents"]
    residue_types = data["residue_types"]
    n_residue_types = data["n_residue_types"]

    logger.info("Residue type distribution:")
    unique, counts = np.unique(residue_types, return_counts=True)
    labels = get_residue_type_labels(n_residue_types)
    for idx, count in zip(unique, counts, strict=False):
        if idx < len(labels):
            logger.info(f"  {labels[int(idx)]}: {count}")

    # Project to 2D using shared utility
    perplexity = kwargs.get("perplexity", 30)
    max_iter = kwargs.get("max_iter", kwargs.get("n_iter", 1000))  # Support both param names

    projection = project_latents_2d(
        latents=latents,
        labels=residue_types,
        method=method,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=42,
    )

    # Plot
    output_path = os.path.join(output_dir, f"latent_projection_{method}.png")
    plot_latent_projection(
        projection=projection,
        labels=residue_types,
        output_path=output_path,
        title=f"Latent Space Projection ({method.upper()})",
        filter_residues=filter_residues,
        n_residue_types=n_residue_types,
        method=method.upper(),
    )

    logger.info(f"Latent analysis complete. Results saved to {output_dir}")


def main() -> None:
    """Main entry point for latent analysis script."""
    load_dotenv()

    # Parse args and config
    args, cfg, config_name = parse_args_and_cfg()

    # Setup
    root_path = setup(cfg, create_root=True, config_name=config_name)

    # Get dataloader
    dataloader = load_dataloader(cfg)

    # Load model
    model = AutoEncoder.load_from_checkpoint(cfg.ckpt_file)

    # Make predictions
    trainer = L.Trainer(accelerator="gpu", devices=1, limit_predict_batches=int(cfg.n_structs / cfg.bs))
    predictions = trainer.predict(model, dataloader)

    # Run latent analysis
    latent_cfg = cfg.get("latent_analysis", {})
    if latent_cfg.get("enabled", False):
        output_dir = latent_cfg.get("output_dir", "./latent_analysis")
        method = latent_cfg.get("projection_method", "tsne")
        n_samples = latent_cfg.get("n_samples", 10000)
        filter_residues = latent_cfg.get("filter_residues", None)

        # Get method-specific kwargs
        method_kwargs = latent_cfg.get(method, {})

        run_latent_analysis(
            predictions=predictions,
            output_dir=output_dir,
            method=method,
            n_samples=n_samples,
            filter_residues=filter_residues,
            **method_kwargs,
        )
    else:
        logger.info("Latent analysis disabled in config")


if __name__ == "__main__":
    main()
