"""Protected, residue-scoped surface engineering primitives.

Probability conditions are learned preferences, not guaranteed output frequencies.
Logit biases operate at the sequence decision and must precede decoder C_SEQ.
"""

import random
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from atomworks.ml.conditions import C_SEQ
from biotite.structure import AtomArray
from numpy.typing import NDArray

from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS

AA_NAMES = (
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
)
ResidueKey = tuple[str, int, str]
XTR_BIASES = {"ASP": 1.0, "GLU": 0.5, "ASN": -1.0, "GLN": -1.0, "GLY": -1.0, "SER": -1.0}
NOTION_BIASES = XTR_BIASES | {"ALA": 0.5, "THR": 0.5}

# Positive magnitudes of side-chain T*dS loss, kcal/mol at 20 C relative to Ala.
# Pickett/Sternberg values reproduced in GEMS Table 1:
# https://pmc.ncbi.nlm.nih.gov/articles/PMC2373623/
# Backbone entropy is excluded: this scale cannot assess Gly->Ala rigidity.
SIDECHAIN_TDS = dict(
    zip(
        AA_NAMES,
        (
            0.0,
            2.03,
            1.57,
            1.25,
            0.55,
            2.11,
            1.81,
            0.0,
            0.96,
            0.89,
            0.78,
            1.94,
            1.61,
            0.58,
            0.0,
            1.71,
            1.63,
            0.97,
            0.98,
            0.51,
        ),
        strict=True,
    )
)


def logit_bias_override(bias: Mapping[str, float]) -> str:
    """Serialize a numeric AA map in Hydra syntax (JSON-quoted keys are invalid)."""
    if not bias or set(bias) - set(AA_NAMES) or not all(np.isfinite(v) for v in bias.values()):
        raise ValueError("Logit biases must contain finite canonical-AA values")
    return (
        "generation.args.sequence_logit_bias={" + ",".join(f"{aa}:{float(value)}" for aa, value in bias.items()) + "}"
    )


def atom_residue_keys(atoms: AtomArray) -> list[ResidueKey]:
    """Return exact chain, PDB residue number and insertion code per atom."""
    insertion = atoms.ins_code if "ins_code" in atoms.get_annotation_categories() else [""] * len(atoms)
    return [(str(c), int(r), str(i)) for c, r, i in zip(atoms.chain_id, atoms.res_id, insertion, strict=True)]


def residue_keys(atoms: AtomArray) -> list[ResidueKey]:
    """Return ordered residue identities, rejecting discontiguous duplicate IDs."""
    keys: list[ResidueKey] = []
    seen: set[ResidueKey] = set()
    for key in atom_residue_keys(atoms):
        if keys and key == keys[-1]:
            continue
        if key in seen:
            raise ValueError(f"Discontiguous duplicate residue identity: {key}")
        keys.append(key)
        seen.add(key)
    return keys


def validate_probabilities(probabilities: Mapping[str, float]) -> None:
    """Validate a full or partial canonical-AA probability specification."""
    if not probabilities or set(probabilities) - set(AA_NAMES):
        raise ValueError("Probabilities must name canonical three-letter amino acids")
    values = np.asarray(list(probabilities.values()), dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    mass = float(values.sum())
    if mass > 1.0 + 1e-7 or (len(probabilities) == len(AA_NAMES) and not np.isclose(mass, 1.0)):
        raise ValueError("Specified probability mass exceeds one or full distribution is not normalized")


def complete_probability_profile(profile: Mapping[str, float], background: Mapping[str, float]) -> dict[str, float]:
    """Resolve only unspecified mass using a declared background distribution.

    Original specified entries, including exact zero, remain unchanged. This is
    an explicit remainder policy for an experiment, not a claim about the model's
    internal completion of omitted conditioning values.
    """
    validate_probabilities(profile)
    validate_probabilities(background)
    if set(background) != set(AA_NAMES):
        raise ValueError("Completion needs a full background distribution")
    missing = [name for name in AA_NAMES if name not in profile]
    if not missing:
        return dict(profile)
    remainder = max(0.0, 1.0 - sum(profile.values()))
    mass = sum(background[name] for name in missing)
    if mass == 0 and remainder > 0:
        raise ValueError("The background cannot allocate the unspecified probability mass")
    result = dict(profile) | {name: remainder * background[name] / mass if mass else 0.0 for name in missing}
    validate_probabilities(result)
    return result


def reweight_probabilities(base: Mapping[str, float], biases: Mapping[str, float], strength: float) -> dict[str, float]:
    """Exponentially tilt a full reference distribution, preserving true zeros.

    This translates bias preferences into probability inputs; it does not make
    the model's learned conditioning equivalent to biased sampling logits.
    """
    validate_probabilities(base)
    if set(base) != set(AA_NAMES):
        raise ValueError("Reweighting needs a full reference distribution")
    if not np.isfinite(strength) or strength < 0 or set(biases) - set(AA_NAMES):
        raise ValueError("Strength must be finite/nonnegative and biases must name canonical AAs")
    delta = np.array([biases.get(aa, 0.0) for aa in AA_NAMES], dtype=float)
    if not np.isfinite(delta).all():
        raise ValueError("Biases must be finite")
    if strength == 0:
        return dict(base)
    raw = np.array([base[aa] for aa in AA_NAMES])
    supported = raw > 0
    log_weights = np.full_like(raw, -np.inf)
    log_weights[supported] = np.log(raw[supported]) + strength * delta[supported]
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()
    return dict(zip(AA_NAMES, map(float, weights), strict=True))


def selection_mask(atoms: AtomArray, selected: Sequence[ResidueKey]) -> NDArray[np.bool_]:
    """Resolve residue keys without silently dropping absent or duplicated IDs."""
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Selection must contain unique, nonempty residue keys")
    missing = set(selected) - set(residue_keys(atoms))
    if missing:
        raise ValueError(f"Selected residues are missing from the structure: {sorted(missing)}")
    selected_set = set(selected)
    return np.array([key in selected_set for key in atom_residue_keys(atoms)], dtype=bool)


def annotate_probabilities(
    atoms: AtomArray, selected: Sequence[ResidueKey], probabilities: Mapping[str, float]
) -> None:
    """Set only selected whole residues, retaining NaN for unspecified identities.

    Fail before writing if a requested site is sequence-fixed or noncanonical.
    Existing values on every unselected atom are preserved.
    """
    validate_probabilities(probabilities)
    mask = selection_mask(atoms, selected)
    if (mask & np.asarray(C_SEQ.mask(atoms, default="generate"), dtype=bool)).any():
        raise ValueError("Surface probability selection includes sequence-fixed residues")
    if not np.isin(atoms.res_name[mask], AA_NAMES).all():
        raise ValueError("Surface probability selection includes noncanonical residues")
    for condition in SEQUENCE_PROBABILITY_CONDITIONS:
        values = np.asarray(condition.annotation(atoms, default="generate"), dtype=np.float32).copy()
        values[mask] = probabilities.get(condition.token_name, float("nan"))
        condition.set_annotation(atoms, values)


def bias_sequence_logits(
    logits: torch.Tensor,
    bias: torch.Tensor,
    selected: torch.Tensor,
    fixed: torch.Tensor,
    legal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bias mutable selected tokens and exclude illegal token identities."""
    if selected.shape != logits.shape[:-1] or fixed.shape != selected.shape or bias.shape != logits.shape[-1:]:
        raise ValueError("Logit bias and selection shapes do not match")
    if not torch.isfinite(bias).all():
        raise ValueError("Sequence biases must be finite")
    active = selected.bool() & ~fixed.bool()
    result = logits + active[..., None].to(logits.dtype) * bias.to(device=logits.device, dtype=logits.dtype)
    if legal is not None:
        if legal.shape != logits.shape or not legal.any(dim=-1).all():
            raise ValueError("Every token needs a correctly shaped nonempty legal vocabulary")
        result = result.masked_fill(~legal, -torch.inf)
    return result


def surface_metrics(
    reference: Mapping[ResidueKey, str],
    generated: Mapping[ResidueKey, str],
    eligible: Sequence[ResidueKey],
    protected: Sequence[ResidueKey],
) -> dict[str, float | int]:
    """Measure a fixed reference selection; reject missing maps and protected edits."""
    if not eligible or len(set(eligible)) != len(eligible) or set(eligible) & set(protected):
        raise ValueError("Eligible selection must be nonempty, unique and disjoint from protected sites")
    if set(reference) != set(generated) or not (set(eligible) | set(protected)) <= reference.keys():
        raise ValueError("Generated and reference residue mappings differ")
    changed_protected = [key for key in protected if reference[key] != generated[key]]
    if changed_protected:
        raise ValueError(f"Changed protected identities: {changed_protected}")
    if any(generated[key] not in AA_NAMES for key in eligible):
        raise ValueError("Noncanonical generated identity on the eligible surface")
    result: dict[str, float | int] = {
        "n_surface": len(eligible),
        "n_mutations": sum(reference[k] != generated[k] for k in reference),
        "n_surface_mutations": sum(reference[k] != generated[k] for k in eligible),
        "n_protected_violations": 0,
    }
    result.update(
        {f"surface_fraction_{aa}": sum(generated[k] == aa for k in eligible) / len(eligible) for aa in AA_NAMES}
    )
    result["surface_fraction_DETA"] = sum(generated[k] in {"ASP", "GLU", "THR", "ALA"} for k in eligible) / len(
        eligible
    )
    result["surface_fraction_NQGS"] = sum(generated[k] in {"ASN", "GLN", "GLY", "SER"} for k in eligible) / len(
        eligible
    )
    result["surface_sidechain_TDS_mean_kcal_mol"] = sum(SIDECHAIN_TDS[generated[k]] for k in eligible) / len(eligible)
    result["surface_sidechain_TDS_delta_kcal_mol"] = sum(
        SIDECHAIN_TDS[generated[k]] - SIDECHAIN_TDS[reference[k]] for k in eligible
    ) / len(eligible)
    return result


def select_surface_pool(scores: Mapping[str, float], count: int, *, random_seed: int | None = None) -> list[str]:
    """Select from the same finite pool by descending preference or seeded random.

    Quality/protection eligibility must be established before constructing this
    pool; its rejected members and compute still belong in the caller's ledger.
    Stable IDs make random controls independent of score values and input order.
    """
    if not scores or not 1 <= count <= len(scores) or any(not np.isfinite(value) for value in scores.values()):
        raise ValueError("Selection needs a nonempty finite pool and a valid count")
    if random_seed is not None:
        return random.Random(random_seed).sample(sorted(scores), count)
    return sorted(scores, key=lambda key: (-scores[key], key))[:count]


def audit_guidance_delta(delta: Mapping[str, torch.Tensor], mutable: torch.Tensor) -> dict[str, float]:
    """Audit real guidance forces without modifying or retaining their tensors."""
    norm, outside = 0.0, 0.0
    for value in delta.values():
        if value.ndim < 2 or value.shape[:2] != mutable.shape or not torch.isfinite(value).all():
            raise ValueError("Guidance delta must have the expected token shape and finite values")
        flat = value.detach().reshape(*mutable.shape, -1)
        unselected = flat[~mutable.bool()]
        if unselected.numel():
            outside = max(outside, float(unselected.abs().max()))
        norm += float(flat.abs().sum())
    if outside != 0:
        raise ValueError("Guidance force leaked outside mutable selected tokens")
    return {"gradient_l1": norm, "outside_max": outside}
