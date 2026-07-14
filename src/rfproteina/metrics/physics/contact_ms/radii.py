"""Contact Molecular Surface radii lookup.

Uses a two-tier dict for fast O(1) lookup: exact ``(residue, atom)`` match
first, then prefix-based fallback for wildcard patterns. No string parsing
at query time.

Supports both protein-protein and protein-small molecule interfaces.
Standard amino-acid radii follow Lawrence & Colman (1993).  Ligand
residues (names starting with ``LG``) have their own lookup table.
Common drug-like atoms (Cl, Br, F, P, S, …) are handled via the
universal prefix fallbacks.

Reference: Lawrence & Colman (1993), Shape Complementarity at Protein/Protein Interfaces.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from jaxtyping import Float

# ── Exact radii: (residue, atom_name) -> radius ──────────────────────────────
# Residue-specific atoms that have an exact match (no wildcards needed).

_EXACT_RADII: dict[tuple[str, str], float] = {
    ("ALA", "CB"): 1.95,
    ("ARG", "CZ"): 1.80,
    ("ARG", "NE"): 1.65,
    ("ARG", "CD"): 1.90,
    ("ARG", "CG"): 1.90,
    ("ASN", "ND2"): 1.70,
    ("ASN", "OD1"): 1.60,
    ("ASN", "CG"): 1.80,
    ("ASP", "CG"): 1.80,
    ("GLN", "NE2"): 1.70,
    ("GLN", "OE1"): 1.60,
    ("GLN", "CD"): 1.80,
    ("GLN", "CG"): 1.90,
    ("GLU", "CD"): 1.80,
    ("GLU", "CG"): 1.90,
    ("GLY", "CA"): 1.90,
    ("HIS", "CD2"): 1.90,
    ("HIS", "NE2"): 1.65,
    ("HIS", "CE1"): 1.90,
    ("HIS", "ND1"): 1.65,
    ("HIS", "CG"): 1.80,
    ("ILE", "CD1"): 1.95,
    ("ILE", "CG1"): 1.90,
    ("ILE", "CB"): 1.85,
    ("ILE", "CG2"): 1.95,
    ("LEU", "CG"): 1.85,
    ("LYS", "NZ"): 1.75,
    ("LYS", "CE"): 1.90,
    ("LYS", "CD"): 1.90,
    ("LYS", "CG"): 1.90,
    ("MET", "CE"): 1.95,
    ("MET", "CG"): 1.90,
    ("PHE", "CZ"): 1.90,
    ("PHE", "CG"): 1.80,
    ("PRO", "CD"): 1.90,
    ("PRO", "CG"): 1.90,
    ("SER", "OG"): 1.70,
    ("SUL", "S"): 1.90,
    ("THR", "CG2"): 1.95,
    ("THR", "OG1"): 1.70,
    ("THR", "CB"): 1.85,
    ("TRP", "CE2"): 1.80,
    ("TRP", "CE3"): 1.90,
    ("TRP", "CD1"): 1.90,
    ("TRP", "CD2"): 1.80,
    ("TRP", "CH2"): 1.90,
    ("TRP", "NE1"): 1.65,
    ("TRP", "CG"): 1.80,
    ("TYR", "OH"): 1.70,
    ("TYR", "CZ"): 1.80,
    ("TYR", "CG"): 1.80,
    ("VAL", "CB"): 1.85,
    ("WAT", "O"): 1.70,
}

# ── Prefix radii: (residue, atom_prefix, radius) ─────────────────────────────
# Entries where the original table had trailing "*" wildcards.
# Checked by ``atom_name.startswith(prefix)``.
# Order matters: more specific (longer prefix) entries should come first
# within the same residue, but since we iterate per-residue, the order
# within a residue group is what matters.

_PREFIX_RADII: list[tuple[str, str, float]] = [
    ("ARG", "NH", 1.70),
    ("ASP", "OD", 1.60),
    ("GLU", "OE", 1.60),
    ("HOH", "O", 1.70),
    ("LEU", "CD", 1.95),
    ("PHE", "CD", 1.90),
    ("PHE", "CE", 1.90),
    ("SUL", "O", 1.65),
    ("TRP", "CZ", 1.90),
    ("TYR", "CD", 1.90),
    ("TYR", "CE", 1.90),
    ("VAL", "CG", 1.95),
    ("WAT", "O", 1.70),
]

# ── Universal fallbacks (residue="*"): atom_name or atom_prefix -> radius ────
# Checked when no residue-specific match is found.

_UNIVERSAL_EXACT: dict[str, float] = {
    "C": 1.80,
    "CA": 1.85,
    "CB": 1.90,
    "N": 1.65,
    "O": 1.60,
    "OXT": 1.60,
    "H": 0.50,
}

_UNIVERSAL_PREFIX: list[tuple[str, float]] = [
    ("OT", 1.60),
    ("H", 0.50),
    ("1H", 0.50),
    ("2H", 0.50),
    ("3H", 0.50),
    ("C", 1.85),
    ("N", 1.65),
    ("O", 1.60),
    ("S", 1.90),
    ("P", 2.15),
    ("F", 1.50),
    ("Cl", 1.80),
    ("Br", 1.95),
]

# Pre-group prefix radii by residue for fast lookup
_PREFIX_BY_RESIDUE: dict[str, list[tuple[str, float]]] = {}
for _res, _prefix, _rad in _PREFIX_RADII:
    _PREFIX_BY_RESIDUE.setdefault(_res, []).append((_prefix, _rad))

# Ligand prefix patterns (residue starts with "LG")
_LIGAND_EXACT: dict[str, float] = {
    "HARO": 1.00,
    "AROC": 2.00,
    "ARO": 2.00,
    "NHIS": 1.75,
    "NHI": 1.75,
    "COO": 1.55,
    "OCBB": 1.55,
    "OCB": 1.55,
    "OOC": 1.55,
    "NLYS": 1.75,
}


@lru_cache(maxsize=4096)
def _lookup_radius(res_name: str, atom_name: str) -> float:
    """Look up radius for a single (residue, atom) pair. Cached."""
    # 1. Exact (residue, atom) match
    r = _EXACT_RADII.get((res_name, atom_name))
    if r is not None:
        return r

    # 2. Residue-specific prefix match
    prefixes = _PREFIX_BY_RESIDUE.get(res_name)
    if prefixes is not None:
        for prefix, radius in prefixes:
            if atom_name.startswith(prefix):
                return radius

    # 3. Ligand residues (LG*)
    if res_name.startswith("LG"):
        r = _LIGAND_EXACT.get(atom_name)
        if r is not None:
            return r

    # 4. Universal exact match
    r = _UNIVERSAL_EXACT.get(atom_name)
    if r is not None:
        return r

    # 5. Universal prefix match
    for prefix, radius in _UNIVERSAL_PREFIX:
        if atom_name.startswith(prefix):
            return radius

    return 0.0


def get_radii_from_names(
    res_names: list[str],
    atom_names: list[str],
) -> Float[np.ndarray, N]:
    """Look up CMS radii for parallel lists of residue and atom names.

    Names should be stripped (e.g. ``"N"`` not ``" N  "``).

    Args:
        res_names: Residue 3-letter codes, length *N*.
        atom_names: Atom names, length *N*.

    Returns:
        Radii array of shape ``(N,)``. Unmatched atoms get ``0.0``.
    """
    radii = np.empty(len(res_names), dtype=np.float64)
    for i, (res, atom) in enumerate(zip(res_names, atom_names, strict=False)):
        radii[i] = _lookup_radius(res, atom)
    return radii
