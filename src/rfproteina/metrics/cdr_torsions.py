"""Chothia CDR phi/psi naturalness against native bivariate von Mises KDEs."""

import gzip
import json
from functools import cached_property, lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from biotite.structure import (
    AtomArray,
    dihedral_backbone,
    filter_amino_acids,
    get_residue_starts,
)
from scipy.integrate import quad
from scipy.special import i0e, logsumexp

from rfproteina.metrics.antibody_metrics import CDR_RANGES, annotate_antibody_numbering
from rfproteina.metrics.metric import Metric


@lru_cache(maxsize=128)
def log_normalization(kappa1: float, kappa2: float, lam: float) -> float:
    """Log normalizer of the sine-model BVM; integrate out psi analytically."""
    offset = abs(kappa1) + np.hypot(kappa2, lam)

    def integrand(phi: float) -> float:
        radius = np.hypot(kappa2, lam * np.sin(phi))
        return float(2 * np.pi * np.exp(kappa1 * np.cos(phi) + radius - offset) * i0e(radius))

    integral, _ = quad(integrand, -np.pi, np.pi, epsabs=1e-12, epsrel=1e-10)
    return float(np.log(integral) + offset)


def log_likelihood(angles: np.ndarray, native: np.ndarray, parameters: dict[str, float]) -> float:
    """Evaluate one phi/psi pair (radians) against an equally weighted native KDE."""
    if not np.isfinite(angles).all():
        return float("nan")
    native = native[np.isfinite(native).all(axis=1)]
    if not len(native):
        return float("nan")
    k1, k2, lam = (parameters[key] for key in ("kappa1", "kappa2", "lambda"))
    delta = angles - native
    kernels = k1 * np.cos(delta[:, 0]) + k2 * np.cos(delta[:, 1])
    kernels += lam * np.sin(delta[:, 0]) * np.sin(delta[:, 1])
    return float(logsumexp(kernels) - np.log(len(native)) - log_normalization(k1, k2, lam))


def backbone_phi_psi(chain: AtomArray) -> np.ndarray:
    """Return residue-ordered radians, marking missing atoms and peptide breaks NaN."""
    phi, psi, _ = dihedral_backbone(chain)
    starts = get_residue_starts(chain, add_exclusive_stop=True)
    n = np.full((len(starts) - 1, 3), np.nan)
    c = np.full_like(n, np.nan)
    for i, (start, stop) in enumerate(pairwise(starts)):
        residue = chain[start:stop]
        for name, coords in (("N", n), ("C", c)):
            mask = residue.atom_name == name
            if mask.sum() == 1:
                coords[i] = residue.coord[mask][0]
    distance = np.linalg.norm(c[:-1] - n[1:], axis=1)
    connected = np.isfinite(distance) & (distance > 0) & (distance <= 2.0)
    phi[1:][~connected] = np.nan
    psi[:-1][~connected] = np.nan
    return np.column_stack((phi, psi))


class CDRTorsionNaturalness(Metric):
    """Sum native-KDE threshold deviations for each Chothia CDR, scoring phi/psi only.

    Shared Chothia annotations identify variable domains across all protein chains.
    Native distributions default to VH/VL when a light domain is present and VHH
    otherwise. This is a composition heuristic, not a VHH sequence classification;
    set ``reference_set="vhvl"`` when scoring an isolated conventional VH domain.
    Score complexes containing multiple domains of the same chain type separately.

    Scores retain the reference convention: zero passes, more negative is worse.
    Unsupported lengths, incomplete backbones, and failed numbering yield NaN;
    per-CDR lengths and ``n_scored_cdrs`` make incomplete coverage visible.
    """

    def __init__(
        self,
        reference_set: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if reference_set not in {None, "vhh", "vhvl"}:
            raise ValueError("reference_set must be None, 'vhh', or 'vhvl'.")
        self.reference_set = reference_set

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"predicted_atom_array": ("predicted_atom_array",)}

    @cached_property
    def parameters(self) -> dict[str, Any]:
        result = {}
        for reference in ("vhh", "vhvl"):
            path = Path(__file__).with_name("cdr_torsion_data") / f"{reference}_parameters.json.gz"
            with gzip.open(path, "rt") as file:
                result[reference] = json.load(file)
        return result

    @cached_property
    def native_samples(self) -> dict[str, dict[str, np.ndarray]]:
        root = Path(__file__).with_name("cdr_torsion_data")
        result = {}
        for reference in ("vhh", "vhvl"):
            path = root / f"{reference}_torsions.npz"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing CDR native samples: {path}. Run scripts/prepare_cdr_torsion_data.py "
                    "on the supplied torsion pickles."
                )
            with np.load(path, allow_pickle=False) as data:
                result[reference] = {key: data[key] for key in data.files}
        return result

    def setup(self) -> None:
        """Check reference data before scoring designs."""
        _ = self.parameters, self.native_samples

    def teardown(self) -> None:
        """Release native samples at the end of validation."""
        self.__dict__.pop("native_samples", None)

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Score a generated antibody without relying on input residue numbering."""
        array = annotate_antibody_numbering(predicted_atom_array)
        selected = filter_amino_acids(array)
        has_light = np.isin(array.chothia_chain_type, ["K", "L"]).any()
        reference = self.reference_set or ("vhvl" if has_light else "vhh")
        if reference == "vhh" and has_light:
            raise ValueError("VHH reference distributions do not support light domains; use reference_set='vhvl'.")
        chain_attr = "chain_iid" if "chain_iid" in array.get_annotation_categories() else "chain_id"
        chain_ids = array.get_annotation(chain_attr)
        result = {cdr: float("nan") for cdr in CDR_RANGES}
        result.update({f"{cdr}_length": float("nan") for cdr in CDR_RANGES})
        result["n_scored_cdrs"] = 0.0
        seen = set()
        for domain_id in np.unique(array.chothia_domain_id[array.chothia_domain_id >= 0]):
            domain = array.chothia_domain_id == domain_id
            chain_id = chain_ids[domain][0]
            chain = array[selected & (chain_ids == chain_id)]
            starts = get_residue_starts(chain)
            numbers = chain.chothia_numbering[starts]
            in_domain = chain.chothia_domain_id[starts] == domain_id
            prefix = "H" if array.chothia_chain_type[domain][0] == "H" else "L"
            loops = {}
            for cdr, (lo, hi) in CDR_RANGES.items():
                if cdr.startswith(prefix) and lo in numbers[in_domain] and hi in numbers[in_domain]:
                    loops[cdr] = np.flatnonzero(in_domain & chain.is_cdr[starts] & (numbers >= lo) & (numbers <= hi))
            if seen.intersection(loops):
                raise ValueError(
                    "Multiple domains of the same antibody chain type found; score each antibody separately."
                )
            seen.update(loops)
            angles = backbone_phi_psi(chain)
            for cdr, indices in loops.items():
                length = len(indices)
                result[f"{cdr}_length"] = float(length)
                params = self.parameters[reference].get(cdr, {}).get(str(length))
                if params is None:
                    continue
                native = self.native_samples[reference][f"{cdr}_{length}"]
                if native.ndim != 3 or native.shape[1:] != (length, 2):
                    raise ValueError(f"Invalid native sample shape for {cdr}, length {length}: {native.shape}.")
                scores = []
                for i, index in enumerate(indices):
                    p = params[str(i + 1)]
                    threshold = p["mean_stdev_threshold"]
                    if not np.isfinite(threshold) or threshold >= 0:
                        raise ValueError("CDR threshold deviations require finite, negative reference thresholds.")
                    ll = log_likelihood(angles[index], native[:, i], p)
                    score = min(0.0, (threshold - ll) / threshold) if np.isfinite(ll) else float("nan")
                    scores.append(score)
                result[cdr] = float(np.sum(scores))
        result["n_scored_cdrs"] = float(sum(np.isfinite(result[cdr]) for cdr in CDR_RANGES))
        return result
