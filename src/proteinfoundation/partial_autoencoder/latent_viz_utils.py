"""Shared utilities for latent space visualization.

This module provides reusable functions for visualizing latent spaces with
dimensionality reduction techniques, used by both training and inference.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from proteinfoundation.utils.ddp import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


# Color palette for unified encoding (n_residue_types=33)
# Index 0: <M> (mask token)
# Indices 1-20: Standard amino acids
# Index 21: UNK
# Indices 22-26: RNA (rA, rC, rG, rU, rN)
# Indices 27-31: DNA (dA, dC, dG, dT, dN)
# Index 32: <A> (atomised token)

COLORS_UNIFIED_33 = [
    # Index 0: Mask token
    "#2F2F2F",  # Dark Gray (<M>)
    # Indices 1-20: Standard amino acids (distinct, saturated colors)
    "#E41A1C",  # Bright Red (ALA)
    "#377EB8",  # Bright Blue (ARG)
    "#4DAF4A",  # Bright Green (ASN)
    "#984EA3",  # Bright Purple (ASP)
    "#FF7F00",  # Bright Orange (CYS)
    "#FFFF33",  # Bright Yellow (GLN)
    "#A65628",  # Brown (GLU)
    "#F781BF",  # Pink (GLY)
    "#999999",  # Gray (HIS)
    "#66C2A5",  # Teal (ILE)
    "#FC8D62",  # Salmon (LEU)
    "#8DA0CB",  # Light Blue (LYS)
    "#E78AC3",  # Light Pink (MET)
    "#A6D854",  # Lime Green (PHE)
    "#FFD92F",  # Gold (PRO)
    "#E5C494",  # Tan (SER)
    "#B3B3B3",  # Light Gray (THR)
    "#1B9E77",  # Dark Teal (TRP)
    "#D95F02",  # Dark Orange (TYR)
    "#7570B3",  # Lavender (VAL)
    # Index 21: UNK
    "#B0B0B0",  # Gray (UNK)
    # Indices 22-26: RNA - warm, saturated colors
    "#FF6B6B",  # Bright Red (RNA A)
    "#FF8C42",  # Bright Orange (RNA C)
    "#FFD93D",  # Bright Yellow (RNA G)
    "#FF1493",  # Deep Pink (RNA U)
    "#FFA07A",  # Light Salmon (RNA N)
    # Indices 27-31: DNA - cool, saturated colors
    "#4ECDC4",  # Bright Teal (DNA DA)
    "#5B5FFF",  # Bright Blue-Purple (DNA DC)
    "#00CED1",  # Dark Turquoise (DNA DG)
    "#1E90FF",  # Dodger Blue (DNA DT)
    "#87CEEB",  # Sky Blue (DNA DN)
    # Index 32: Atomised token
    "#696969",  # DimGray (<A>)
]


def get_residue_type_labels(n_residue_types: int) -> list[str]:
    """Get human-readable labels for residue types.

    Args:
      n_residue_types: Number of residue types (20 for legacy, 33 for unified).

    Returns:
      List of labels for each residue type index.
    """
    # Standard 20 amino acids (3-letter codes)
    aa_labels = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]

    if n_residue_types == 20:
        return [*aa_labels, "UNK"]
    elif n_residue_types == 33:
        return ["MSK", *aa_labels, "UNK", "rA", "rC", "rG", "rU", "rN", "dA", "dC", "dG", "dT", "dN", "ATM"]
    else:
        # Fallback for unknown encodings
        return [f"RT{i}" for i in range(n_residue_types + 1)]


def get_extended_color_palette(n_colors: int) -> list[str]:
    """Get extended color palette for visualization.

    Args:
      n_colors: Number of colors needed (20 for legacy, 33 for unified encoding).

    Returns:
      List of hex color codes, correctly indexed to match residue type indices.
    """
    if n_colors == 33:
        return COLORS_UNIFIED_33
    elif n_colors == 20:
        # Legacy 20 AA + UNK (21 colors total)
        return COLORS_UNIFIED_33[1:21] + ["#B0B0B0"]
    elif n_colors <= len(COLORS_UNIFIED_33):
        return COLORS_UNIFIED_33[:n_colors]
    else:
        # Generate additional colors if needed
        logger.warning(f"Requested {n_colors} colors, only {len(COLORS_UNIFIED_33)} predefined.")
        cmap = plt.cm.get_cmap("tab20c")
        extra_colors = [cmap(i / (n_colors - len(COLORS_UNIFIED_33))) for i in range(n_colors - len(COLORS_UNIFIED_33))]
        extra_hex = [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b, _ in extra_colors]
        return list(COLORS_UNIFIED_33) + extra_hex


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
    n_residue_types: int,
    method: str = "t-SNE",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 8),
    add_cluster_labels: bool = True,
) -> plt.Figure:
    """Create scatter plot of 2D projection colored by residue type.

    Args:
      projection: [n_samples, 2] projected coordinates.
      labels: [n_samples] residue type indices.
      n_residue_types: Number of residue types (20 or 33).
      method: Name of projection method for axis labels.
      title: Plot title (optional).
      figsize: Figure size in inches.
      add_cluster_labels: Whether to add text labels at cluster centers.

    Returns:
      Matplotlib figure object.
    """
    colors = get_extended_color_palette(n_residue_types)
    residue_labels = get_residue_type_labels(n_residue_types)

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
