"""Featurization of stereogenic centers and their neighbors."""

from collections.abc import Generator

import numpy as np
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.geometry import get_random_rots
from biotite.structure import AtomArray
from scipy.spatial.transform import Rotation

from rfproteina.utils.chirals import (
    STEREOISOMER_TEMPLATES,
    classify_stereoisomer,
    identify_stereogenic_centers,
)

MAX_OVERLAPPING_CENTERS: int = 6
"""Max disambiguable overlapping chiral centers per atom.

Sized to match the largest supported coordination number (octahedral, 6
neighbors); see :py:data:`rfproteina.utils.chirals.IDEAL_POLYHEDRA`.
"""


def _idealize_coords(centered: np.ndarray, geometry: str) -> np.ndarray:
    """Replace neighbor coords with the best-matching stereoisomeric ideal, rotated."""
    idx = classify_stereoisomer(centered, geometry)
    if idx is None:
        return centered.astype(np.float32)
    template = STEREOISOMER_TEMPLATES[(geometry, len(centered))][idx]
    unit = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    rot, _ = Rotation.align_vectors(unit, template)
    return rot.apply(template).astype(np.float32)


def _iter_idealized_chiral_features(
    atom_array: AtomArray,
    is_inference: bool,
) -> Generator[tuple[int, np.ndarray, np.ndarray], None, None]:
    """Yield ``(center_idx, bonded_sorted, rotated_ideal_coords)`` per chiral center.

    At inference, no random rotation is applied so the feature is deterministic.
    """
    has_atomize = "atomize" in atom_array.get_annotation_categories()
    for center in identify_stereogenic_centers(atom_array):
        cidx, bonded = center["center_idx"], center["bonded_idxs"]
        if has_atomize and not atom_array.atomize[cidx]:
            continue
        centered = atom_array.coord[bonded] - atom_array.coord[cidx]
        idealized = _idealize_coords(centered, center["geometry"])
        rot = np.eye(3) if is_inference else get_random_rots(1, mode="numpy")[0]
        yield cidx, bonded, (idealized @ rot.T)


# ── Public API ──


def build_chirality_block_diagonal(
    atom_array: AtomArray,
    embed_dim: int = 18,
    is_inference: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Per-atom chirality via non-overlapping 3D blocks with cycling fill.

    Uses idealized polyhedron vertices so the feature encodes only chirality.

    Args:
      atom_array: Input structure.
      embed_dim: Output feature dimension; must be a multiple of 3.
      is_inference: If True, output is deterministic (no random rotation, no shuffle).
      rng: Generator used for block-order and cycle shuffles during training.
        Defaults to a fresh :py:func:`numpy.random.default_rng`. Ignored when
        ``is_inference=True``.

    Returns:
      Float ``(n_atoms, embed_dim)``.
    """
    assert embed_dim % 3 == 0, f"embed_dim must be multiple of 3, got {embed_dim}"
    if rng is None:
        # Seed from the global (seed_everything) stream; an unseeded default_rng() ignores the seed.
        rng = np.random.default_rng(np.random.randint(0, 2**32))
    n = atom_array.array_length()
    output = np.zeros((n, embed_dim), dtype=np.float32)
    n_blocks = embed_dim // 3

    atom_block_data: dict[int, dict[int, np.ndarray]] = {}

    for _cidx, bonded, rotated in _iter_idealized_chiral_features(atom_array, is_inference):
        block_order = np.arange(n_blocks)
        if not is_inference:
            rng.shuffle(block_order)

        for block in block_order:
            offset = int(block) * 3
            if not np.any(output[bonded, offset : offset + 3]):
                for j, b in enumerate(bonded):
                    output[b, offset : offset + 3] = rotated[j]
                    atom_block_data.setdefault(b, {})[int(block)] = rotated[j]
                break

    for atom_idx, block_data in atom_block_data.items():
        occupied = sorted(block_data.keys())
        empty = [b for b in range(n_blocks) if b not in block_data]
        if not empty:
            continue
        cycle = list(occupied)
        if not is_inference:
            rng.shuffle(cycle)
        for i, block in enumerate(empty):
            offset = block * 3
            output[atom_idx, offset : offset + 3] = block_data[cycle[i % len(cycle)]]

    return output


def build_is_chiral_center_feature(atom_array: AtomArray) -> np.ndarray:
    """Per-atom boolean: True for chiral center atoms in atomized residues."""
    ensure_annotations(atom_array, "chiral_type")
    mask = atom_array.chiral_type != 0
    if "atomize" in atom_array.get_annotation_categories():
        mask = mask & atom_array.atomize
    return mask


# ── Transforms ──


class AddChiralityFeature(Transform):
    """Add per-atom idealized chirality feature."""

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        is_inference = data.get("is_inference", False)
        feats = data.get("feats", {})

        feats["chirality"] = build_chirality_block_diagonal(
            atom_array,
            embed_dim=MAX_OVERLAPPING_CENTERS * 3,
            is_inference=is_inference,
        )

        data["feats"] = feats
        return data
