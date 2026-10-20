"""Shared utilities for latent space visualization.

This module provides reusable functions for visualizing latent spaces with
dimensionality reduction techniques, used by both training and inference.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from rfproteina.encoding import UNIFIED_ATOM37_ENCODING
from rfproteina.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def get_residue_type_labels() -> list[str]:
    """Return residue names in model vocabulary index order."""
    return UNIFIED_ATOM37_ENCODING.idx_to_token.tolist()


def get_extended_color_palette() -> list[str]:
    """Return one deterministic color per model vocabulary entry."""
    colors = plt.get_cmap("hsv")(np.arange(UNIFIED_ATOM37_ENCODING.n_tokens) / UNIFIED_ATOM37_ENCODING.n_tokens)
    return [to_hex(color) for color in colors]


def project_latents_2d(
    latents: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Project high-dimensional latents to 2D using specified method.

    Args:
      latents: [n_samples, latent_dim] latent vectors.
      labels: [n_samples] class labels (residue types).
      method: Projection method ('tsne', 'pca', or 'lda').
      perplexity: Perplexity parameter for t-SNE.
      max_iter: Number of iterations for t-SNE.
      random_state: Random seed for reproducibility.

    Returns:
      [n_samples, 2] projected coordinates.
    """
    if method == "tsne":
        perplexity = min(perplexity, len(latents) // 5)  # Adjust for small datasets
        projector = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=random_state)
        return projector.fit_transform(latents)

    elif method == "pca":
        projector = PCA(n_components=2)
        projection = projector.fit_transform(latents)
        logger.info(f"PCA explained variance: {projector.explained_variance_ratio_}")
        return projection

    elif method == "lda":
        n_components = min(2, len(np.unique(labels)) - 1)
        if n_components < 1:
            logger.warning("LDA requires at least 2 classes, falling back to PCA")
            return project_latents_2d(latents, labels, method="pca")

        projector = LinearDiscriminantAnalysis(n_components=n_components)
        projection = projector.fit_transform(latents, labels)

        # Pad to 2D if needed
        if projection.shape[1] == 1:
            projection = np.hstack([projection, np.zeros((len(projection), 1))])
        return projection

    else:
        raise ValueError(f"Unknown projection method: {method}")


def plot_latent_projection_with_labels(
    projection: np.ndarray,
    labels: np.ndarray,
    method: str = "t-SNE",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 8),
    add_cluster_labels: bool = True,
) -> plt.Figure:
    """Create scatter plot of 2D projection colored by residue type.

    Args:
      projection: [n_samples, 2] projected coordinates.
      labels: [n_samples] residue type indices.
      method: Name of projection method for axis labels.
      title: Plot title (optional).
      figsize: Figure size in inches.
      add_cluster_labels: Whether to add text labels at cluster centers.

    Returns:
      Matplotlib figure object.
    """
    colors = get_extended_color_palette()
    residue_labels = get_residue_type_labels()

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    centroids = {}

    # Plot each residue type
    for label_idx in unique_labels:
        mask = labels == label_idx
        label_idx_int = int(label_idx)

        if label_idx_int < len(colors) and label_idx_int < len(residue_labels):
            points = projection[mask]

            ax.scatter(
                points[:, 0],
                points[:, 1],
                c=colors[label_idx_int],
                label=residue_labels[label_idx_int],
                s=10 if len(projection) > 1000 else 15,
                alpha=0.6,
                edgecolors="none",
            )

            # Calculate centroid for text annotation
            if add_cluster_labels:
                centroids[label_idx_int] = points.mean(axis=0)

    # Add text labels at cluster centers
    if add_cluster_labels:
        for label_idx, centroid in centroids.items():
            if label_idx < len(residue_labels):
                ax.text(
                    centroid[0],
                    centroid[1],
                    residue_labels[label_idx],
                    fontsize=12 if len(unique_labels) > 20 else 14,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox={
                        "boxstyle": "round,pad=0.4",
                        "facecolor": "white",
                        "edgecolor": "black",
                        "linewidth": 1.5,
                        "alpha": 0.9,
                    },
                    zorder=1000,
                )

    # Formatting
    ax.set_xlabel(f"{method} axis 1", fontsize=12)
    ax.set_ylabel(f"{method} axis 2", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    # Compact legend
    if len(unique_labels) <= 10:
        ncol = 1
    elif len(unique_labels) <= 20:
        ncol = 2
    else:
        ncol = 3

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=ncol,
        fontsize=8 if len(unique_labels) > 20 else 9,
        framealpha=0.9,
        markerscale=2,
        title="Residue Type",
        title_fontsize=10,
    )

    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    plt.tight_layout()

    return fig
