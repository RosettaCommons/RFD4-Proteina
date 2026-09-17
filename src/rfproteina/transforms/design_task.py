import itertools
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum
from functools import partial
from typing import ClassVar, Literal

import biotite.structure as struc
import numpy as np
from atomworks.common import exists
from atomworks.enums import ChainType, ChainTypeInfo
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array import apply_and_spread
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.scatter import (
    apply_segment_wise,
    spread_segment_wise,
)
from atomworks.io.utils.standard_annotations import S_ATM
from atomworks.ml.conditions import (
    C_CRD,
    C_IDX,
    C_SEQ,
)
from atomworks.ml.transforms import mask_generator as mg
from atomworks.ml.transforms.crop import resize_crop_info_if_too_many_atoms
from atomworks.ml.transforms.design_task import DesignTaskABC
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from atomworks.ml.utils.numpy import unique_by_first_occurrence
from atomworks.ml.utils.token import apply_and_spread_token_wise, get_token_count, get_token_starts
from biotite.structure import AtomArray
from scipy.spatial import ConvexHull
from scipy.special import softmax

from rfproteina.conditions import C_CRDGRP, C_HOT, C_IDXGRP, C_INTCRD
from rfproteina.transforms.design_task_modifiers import (
    add_cath_condition,
    add_hbond_condition,
    add_hydrophobic_condition,
    add_metal_condition,
    add_not_cysteine_condition,
    add_pication_condition,
    add_pistack_condition,
    add_rasa_condition,
    add_salt_condition,
    add_secondary_structure_condition,
)
from rfproteina.transforms.na_geom import apply_training_na_ss_condition
from rfproteina.transforms.utils.conditioning_utils import (
    convert_to_internal_coords,
    sample_unindexed_groups,
)
from rfproteina.transforms.utils.ppi_utils import (
    crop_spatial_keep_full_binder,
    sample_hotspots,
)
from rfproteina.utils.common import true_with_prob
from rfproteina.utils.ddp import RankedLogger


# Default potential functions for TipAtomPartialLigandTask (module-level for picklability)
def _default_ligand_potential(x: np.ndarray, scaling_factor: float = 5.0) -> np.ndarray:
    """Default ligand sampling potential: inverse distance with offset."""
    return 1.0 / (scaling_factor + x)


def _default_tip_atom_potential(x: np.ndarray, scaling_factor: float = 8.0) -> np.ndarray:
    """Default tip atom sampling potential: exponential decay."""
    return np.exp(-x / scaling_factor)


ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def _get_pn_unit_chain_type(atom_array: AtomArray, pn_unit_iid: str) -> ChainType | None:
    """Return the pn_unit chain's :py:class:`ChainType`, or ``None`` if absent or mixed."""
    mask = atom_array.pn_unit_iid == pn_unit_iid
    if not mask.any():
        return None
    chain_types = np.unique(atom_array.chain_type[mask])
    if len(chain_types) != 1:
        return None
    return ChainType(int(chain_types[0]))


RFD3_DESIGN_TASK_DEFAULTS = {
    "rasa_kwargs": {
        "p_add_rasa_condition": 0.0,  # 0.6 in RFD3
        "probe_radius": 1.4,
        "atom_radii": "element",
        "point_number": 100,
        "p_include_non_ligand": 0.1,
        "rasa_max": 0.2,
    },
    "hbond_kwargs": {
        "p_add_hbond_condition": 0.0,  # 0.2 in RFD3
        "p_hbond_subsample": 0.5,
        "hbond_subsample_frac": 0.5,
    },
    "pistack_kwargs": {
        "p_add_pistack_condition": 0.0,
        "p_pistack_subsample": 0.5,
        "pistack_subsample_frac": 0.5,
    },
    "pication_kwargs": {
        "p_add_pication_condition": 0.0,
        "p_pication_subsample": 0.5,
        "pication_subsample_frac": 0.5,
    },
    "salt_kwargs": {
        "p_add_salt_condition": 0.0,
        "p_salt_subsample": 0.5,
        "salt_subsample_frac": 0.5,
    },
    "hydrophobic_kwargs": {
        "p_add_hydrophobic_condition": 0.0,
        "p_hydrophobic_subsample": 0.5,
        "hydrophobic_subsample_frac": 0.5,
    },
    "metal_kwargs": {
        "p_add_metal_condition": 0.0,
        "p_metal_subsample": 0.5,
        "metal_subsample_frac": 0.5,
    },
    "secondary_structure_kwargs": {
        "p_add_secondary_structure_condition": 0.0,  # 0.1 in RFD3
        "max_secondary_structure_frac_to_provide": 0.4,
        "min_ss_island_len": 1,
        # Might want to expand later, this is only average. Done for now to avoid over-conditioning.
        "max_ss_island_len": 10,
        "n_islands_max": 3,
    },
    "cath_kwargs": {
        "p_add_cath_c_condition": 0.0,
        "p_add_cath_ca_condition": 0.0,
        "p_add_cath_cat_condition": 0.0,
        "query_pn_unit_iid": None,
    },
    "not_cysteine_kwargs": {
        "p_add_not_cysteine_condition": 0.0,
        "max_subsample_frac": 0.25,
    },
}


class RFD3DesignTask(DesignTaskABC):
    """Wrapper class for DesignTasks used to train RFD3

    The concept behind this class is to allow for common functions to be called by each
    design task without having to specify all kwargs in each design task separately.
    All kwargs and default values are specified in the RFD3_DESIGN_TASK_DEFAULTS constant.

    Keyword Args:
        See ``RFD3_DESIGN_TASK_DEFAULTS`` for supported options and defaults.
    """

    # Overrides for the RFD3_DESIGN_TASK_DEFAULTS. Subclasses can modify their defaults or set to {}
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: dict[str, dict]

    def __init_subclass__(cls):
        super().__init_subclass__()
        if not hasattr(cls, "RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES"):
            raise TypeError(
                f"{cls.__name__} must define class variable `RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES` (even if empty)."
            )

    @staticmethod
    def _deep_merge_dicts(base: dict, override: dict) -> dict:
        """Deep merge two dictionaries, with override values taking precedence.

        Args:
            base: The base dictionary
            override: The dictionary with override values

        Returns:
            A new dictionary with merged values
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = RFD3DesignTask._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def __init__(self, global_design_task_kwargs: dict | None = None, **kwargs):
        super().__init__()
        # Set default values for all design task modifiers with proper deep merge
        design_task_defaults = self._deep_merge_dicts(
            RFD3_DESIGN_TASK_DEFAULTS, self.RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES
        )

        # Handle any global args that were passed, with task-specific kwargs taking precedence
        if not exists(global_design_task_kwargs):
            global_design_task_kwargs = {}
        kwargs = self._deep_merge_dicts(global_design_task_kwargs, kwargs)

        for key, value in design_task_defaults.items():
            params_to_use = value
            if key in kwargs:
                # Deep merge with any provided kwargs
                params_to_use = self._deep_merge_dicts(params_to_use, kwargs[key])
            setattr(self, key, params_to_use)

    def apply_design_task_modifiers(self, data: dict, overrides: dict = {}) -> dict:
        # Make local copies to avoid mutating instance state
        rasa_kwargs = self.rasa_kwargs.copy() | overrides.get("rasa_kwargs", {})
        hbond_kwargs = self.hbond_kwargs.copy() | overrides.get("hbond_kwargs", {})
        pistack_kwargs = self.pistack_kwargs.copy() | overrides.get("pistack_kwargs", {})
        pication_kwargs = self.pication_kwargs.copy() | overrides.get("pication_kwargs", {})
        salt_kwargs = self.salt_kwargs.copy() | overrides.get("salt_kwargs", {})
        hydrophobic_kwargs = self.hydrophobic_kwargs.copy() | overrides.get("hydrophobic_kwargs", {})
        metal_kwargs = self.metal_kwargs.copy() | overrides.get("metal_kwargs", {})
        secondary_structure_kwargs = self.secondary_structure_kwargs.copy() | overrides.get(
            "secondary_structure_kwargs", {}
        )
        cath_kwargs = self.cath_kwargs.copy() | overrides.get("cath_kwargs", {})
        not_cysteine_kwargs = self.not_cysteine_kwargs.copy() | overrides.get("not_cysteine_kwargs", {})

        # Set defaults for task dictionary
        data["task"] |= {
            "added_rasa_condition": False,
            "added_hbond_condition": False,
            "added_pistack_condition": False,
            "added_pication_condition": False,
            "added_salt_condition": False,
            "added_hydrophobic_condition": False,
            "added_metal_condition": False,
            "added_secondary_structure_condition": False,
            "added_cath_c_condition": False,
            "added_cath_ca_condition": False,
            "added_cath_cat_condition": False,
            "added_not_cysteine_condition": False,
        }

        # Possibly add Conditions corresponding to design task modifiers
        data = add_rasa_condition(data, **rasa_kwargs)
        data = add_secondary_structure_condition(data, **secondary_structure_kwargs)
        data = add_not_cysteine_condition(data, **not_cysteine_kwargs)
        data = add_hbond_condition(data, **hbond_kwargs)
        data = add_pistack_condition(data, **pistack_kwargs)
        data = add_pication_condition(data, **pication_kwargs)
        data = add_salt_condition(data, **salt_kwargs)
        data = add_hydrophobic_condition(data, **hydrophobic_kwargs)
        data = add_metal_condition(data, **metal_kwargs)

        # Stochasticity and task annotations handled internally for CATH conditioning
        data = add_cath_condition(data, **cath_kwargs)

        return data

    def ensure_atomized_tokens_are_sequence_conditioned(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        if "atomize" in atom_array.get_annotation_categories():
            seq_cond_mask = C_SEQ.mask(atom_array, default="generate")
            C_SEQ.set_mask(atom_array, seq_cond_mask | atom_array.atomize)
        data["atom_array"] = atom_array
        return data


class NoMotifABC(RFD3DesignTask):
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.1},
        "cath_kwargs": {
            "p_add_cath_c_condition": 0.1,
            "p_add_cath_ca_condition": 0.15,
            "p_add_cath_cat_condition": 0.25,
        },
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.2},
    }

    def can_apply(self, data: dict) -> bool:
        return True

    def set_task_info(self, data: dict) -> dict:
        # ... NOTE: `total_mask` is all zeros, because no atom/token will be allowed to get a condition
        atom_array = as_atom_array_plus(data["atom_array"])

        # ... set the conditions for the task
        n_atoms = atom_array.array_length()

        # ... update the now condition-annotated atom array & task info
        data["atom_array"] = atom_array
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            # required fields
            "class": self.__class__.__name__,
            "mask": np.zeros(n_atoms, dtype=bool),
        }
        return data


class UnconditionalTask(NoMotifABC):
    """Sample a protein backbone & sequence unconditionally."""

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.1},
        "cath_kwargs": {
            "p_add_cath_c_condition": 0.1,
            "p_add_cath_ca_condition": 0.15,
            "p_add_cath_cat_condition": 0.25,
        },
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.2},
    }

    def forward(self, data: dict) -> dict:
        data = super().set_task_info(data)
        data = self.ensure_atomized_tokens_are_sequence_conditioned(data)
        data = self.apply_design_task_modifiers(data)
        return data


class StructurePredictionTask(NoMotifABC):
    """Sample a protein backbone, given the sequence information (= structure prediction)."""

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.0},
        "hbond_kwargs": {"p_add_hbond_condition": 0.0},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.0},
        "cath_kwargs": {"p_add_cath_c_condition": 0.0, "p_add_cath_ca_condition": 0.0, "p_add_cath_cat_condition": 0.0},
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.0},
    }

    def forward(self, data: dict) -> dict:
        data = super().set_task_info(data)

        # ... show sequence to the model
        C_SEQ.set_mask(data["atom_array"], np.ones(data["atom_array"].array_length(), dtype=bool))

        data = self.apply_design_task_modifiers(data)  # This won't do anything because probs are defualted to 0.0

        return data


class InverseFoldingTask(NoMotifABC):
    """Design sequence and sidechains/bases from selected polymers' backbone coordinates.

    Other molecules stay fully conditioned; at least one designable atom is required.

    Args:
        polymer_types: Polymer types to design: "protein" (default), "rna", or "dna".
    """

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.0},
        "hbond_kwargs": {"p_add_hbond_condition": 0.0},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.0},
        "cath_kwargs": {"p_add_cath_c_condition": 0.0, "p_add_cath_ca_condition": 0.0, "p_add_cath_cat_condition": 0.0},
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.2},
    }

    # polymer type -> (per-residue membership annotation, backbone-atom annotation)
    _POLYMER_MASK_ANNOT: ClassVar[dict[str, tuple[str, str]]] = {
        "protein": ("is_standard_aa", "is_protein_backbone"),
        "rna": ("is_rna", "is_rna_backbone"),
        "dna": ("is_dna", "is_dna_backbone"),
    }

    def __init__(self, polymer_types: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        polymer_types = polymer_types or ["protein"]
        unknown = set(polymer_types) - set(self._POLYMER_MASK_ANNOT)
        assert not unknown, f"unknown polymer_types {unknown}; choose from {list(self._POLYMER_MASK_ANNOT)}"
        self.polymer_types = list(polymer_types)

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]
        res_annots = [self._POLYMER_MASK_ANNOT[p][0] for p in self.polymer_types]
        ensure_annotations(atom_array, *res_annots)
        return any(getattr(atom_array, annotation).any() for annotation in res_annots)

    def forward(self, data: dict) -> dict:
        data = super().set_task_info(data)

        atom_array = data["atom_array"]
        res_annots = [self._POLYMER_MASK_ANNOT[p][0] for p in self.polymer_types]
        bb_annots = [self._POLYMER_MASK_ANNOT[p][1] for p in self.polymer_types]
        ensure_annotations(atom_array, *res_annots, *bb_annots)

        # A token is "designed" if it belongs to one of the requested polymer types;
        # its backbone atoms are still shown as coordinate context.
        designed = np.zeros(atom_array.array_length(), dtype=bool)
        backbone = np.zeros(atom_array.array_length(), dtype=bool)
        for res_annot, bb_annot in zip(res_annots, bb_annots, strict=True):
            designed |= getattr(atom_array, res_annot).astype(bool)
            backbone |= getattr(atom_array, bb_annot).astype(bool)

        # provide coordinate info for every atom except designed non-backbone atoms
        hidden_coordinates = designed & ~backbone
        C_CRD.set_mask(atom_array, ~hidden_coordinates)

        # provide sequence info for every atom that is not part of a designed polymer
        C_SEQ.set_mask(atom_array, ~designed)

        data["atom_array"] = atom_array

        data = self.apply_design_task_modifiers(data)
        return data


class ProteinFromDNAContextABC(RFD3DesignTask):
    """Base for DNA-binder design tasks: the whole protein is generated from
    noise while a DNA chain is provided as context.

    The DNA is supplied through the ``C_CRD`` / ``C_SEQ`` condition annotations
    rather than as a protein motif, so there is no motif region — hence the
    all-zeros motif mask stamped by :py:meth:`set_task_info`. This mirrors the
    no-motif tasks, but subclassing :py:class:`RFD3DesignTask` directly keeps
    this a distinct setting. Subclasses decide whether the DNA *structure* is
    given as context or co-generated.

    Design-task modifiers (RASA / H-bond / SSE / CATH conditioning) are left at
    the experiment's global rates (``datasets.global_design_task_kwargs``); the
    DNA tasks intentionally participate in the same conditioning as the protein
    tasks.
    """

    # Empty: take the base defaults so the experiment's global conditioning rates
    # flow through unchanged. (A per-class 0.0 override could not force
    # conditioning off anyway — global kwargs win in the RFD3DesignTask merge.)
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {}

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_dna")
        return bool(atom_array.is_dna.any())

    def set_task_info(self, data: dict) -> dict:
        """Stamp task bookkeeping with an all-zeros motif mask (no protein motif)."""
        atom_array = as_atom_array_plus(data["atom_array"])
        n_atoms = atom_array.array_length()
        data["atom_array"] = atom_array
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": np.zeros(n_atoms, dtype=bool),
        }
        return data


class ProteinFromDNAStructureContextTask(ProteinFromDNAContextABC):
    """Generate the entire protein from noise; DNA structure and sequence are
    both provided as context.
    """

    def forward(self, data: dict) -> dict:
        data = super().set_task_info(data)

        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_dna")
        is_dna = atom_array.is_dna.astype(bool)

        # Provide DNA coordinates (skipping NaN coords) and DNA sequence.
        nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
        C_CRD.set_mask(atom_array, is_dna & ~nan_coord_mask)
        C_SEQ.set_mask(atom_array, is_dna)
        data["atom_array"] = atom_array

        data = self.ensure_atomized_tokens_are_sequence_conditioned(data)
        data = self.apply_design_task_modifiers(data)
        return data


class ProteinFromDNASequenceContextTask(ProteinFromDNAContextABC):
    """Generate the entire protein from noise; only the DNA *sequence* is
    provided as context — the DNA structure is co-generated.
    """

    def forward(self, data: dict) -> dict:
        data = super().set_task_info(data)

        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_dna")
        is_dna = atom_array.is_dna.astype(bool)

        # Provide DNA sequence only; no coordinate conditioning.
        C_SEQ.set_mask(atom_array, is_dna)
        data["atom_array"] = atom_array

        data = self.ensure_atomized_tokens_are_sequence_conditioned(data)
        data = self.apply_design_task_modifiers(data)
        return data


class PartialLigandTask(RFD3DesignTask):
    """
    Intended only for OMOL examples where we have ligands and maybe some tip atoms.
    In this case, we always coordinate condition the non-ligands and sequence condition everything.

    """

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.0},
        "hbond_kwargs": {"p_add_hbond_condition": 0.0},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.0},
        "cath_kwargs": {"p_add_cath_c_condition": 0.0, "p_add_cath_ca_condition": 0.0, "p_add_cath_cat_condition": 0.0},
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.0},
    }

    def __init__(
        self,
        *,
        min_coverage_fraction: float = 0.0,
        max_coverage_fraction: float = 1.0,
        p_fix_ligand_surroundings: float = 1.0,  # on omol we always fix things that aren't ligands
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.min_coverage_fraction = min_coverage_fraction
        self.max_coverage_fraction = max_coverage_fraction
        self.p_fix_ligand_surroundings = p_fix_ligand_surroundings

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_ligand", "is_res_start", "res_min_occupancy", "pn_unit_iid")
        n_eligible_ligands = np.sum(atom_array.mask("is_ligand & is_res_start & (res_min_occupancy > 0)"))
        return n_eligible_ligands >= 1

    def sample_ligand_mask(self, atom_array: AtomArray, ligand_mask: np.ndarray) -> np.ndarray:
        if not ligand_mask.any():
            return np.zeros(atom_array.array_length(), dtype=bool)

        sample_seed = mg.SampleUniformly(is_eligible=ligand_mask)

        grow_seed = mg.GrowByHoppingAlongBondGraph(
            n_hops_expected=1,  # Hop one at a time, and rely on check budget to stop correctly
            require_same_annotation=("pn_unit_iid",),
            atom_array=atom_array,
        )

        # Check budget based on sampled coverage fraction
        n_ligand_atoms = ligand_mask.sum()
        coverage_fraction = np.random.uniform(self.min_coverage_fraction, self.max_coverage_fraction)

        check_budget = mg.CheckAtomBudget(
            n_min_atoms=int(n_ligand_atoms * coverage_fraction),  # rounds down
            n_max_atoms=n_ligand_atoms,
        )
        sampled_ligand_mask, _ = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=sample_seed,
            fn_grow_seed=grow_seed,
            fn_check_budget=check_budget,
        )
        return sampled_ligand_mask

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_ligand", "pn_unit_iid", "res_min_occupancy", "is_res_start")

        # Loop through available ligands and sample a mask for each
        coord_conditioned_mask = np.zeros(atom_array.array_length(), dtype=bool)
        ligand_pn_unit_iids = np.unique(atom_array.pn_unit_iid[atom_array.is_ligand])

        for ligand_pn_unit_iid in ligand_pn_unit_iids:
            ligand_mask = atom_array.pn_unit_iid == ligand_pn_unit_iid
            sampled_ligand_mask = self.sample_ligand_mask(atom_array, ligand_mask)
            coord_conditioned_mask |= sampled_ligand_mask

        sequence_condition_mask = atom_array.is_ligand | atom_array.atomize

        # Coord condition everything that is not a ligand
        if true_with_prob(self.p_fix_ligand_surroundings):
            coord_conditioned_mask |= ~atom_array.is_ligand
            sequence_condition_mask |= ~atom_array.is_ligand

        # Do not coord condition nan coordinates
        nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
        coord_condition_mask = coord_conditioned_mask & ~nan_coord_mask
        C_CRD.set_mask(atom_array, coord_condition_mask)

        C_SEQ.set_mask(atom_array, sequence_condition_mask)

        _n_atoms = atom_array.array_length()
        C_IDX.set_mask(atom_array, np.ones(atom_array.array_length(), dtype=bool))

        data["atom_array"] = atom_array

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": coord_conditioned_mask,
        }

        data = self.apply_design_task_modifiers(data)
        return data


class TipAtomPartialLigandTask(RFD3DesignTask):
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.1},
        "cath_kwargs": {
            "p_add_cath_c_condition": 0.1,
            "p_add_cath_ca_condition": 0.15,
            "p_add_cath_cat_condition": 0.25,
        },
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.2},
    }

    def __init__(
        self,
        *,
        n_max_ligands: int = 3,
        n_hops_expected_ligand: int = 5,
        n_max_residues: int = 10,
        n_hops_expected_tip_atom: int = 1,
        ligand_scaling_factor: float = 5.0,
        tip_atom_scaling_factor: float = 1.5,
        p_seq_conditioned: float = 1.0,  # For now we always sequence condition things with distance conditioning!
        p_knockout_seq: float = 0.0,
        p_unindex_all: float = 0.9,
        p_merge_unindexed_groups: float = 0.1,
        p_split_unindexed_groups: float = 0.01,
        p_increase_group_id_gap: float = 0.1,
        p_internal_coords: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.n_max_ligands = n_max_ligands
        self.n_hops_expected_ligand = n_hops_expected_ligand
        self.ligand_scaling_factor = ligand_scaling_factor

        self.n_max_residues = n_max_residues
        self.n_hops_expected_tip_atom = n_hops_expected_tip_atom
        self.tip_atom_scaling_factor = tip_atom_scaling_factor

        self.p_seq_conditioned = p_seq_conditioned
        self.p_knockout_seq = p_knockout_seq
        self.p_unindex_all = p_unindex_all
        self.p_merge_unindexed_groups = p_merge_unindexed_groups
        self.p_split_unindexed_groups = p_split_unindexed_groups
        self.p_increase_group_id_gap = p_increase_group_id_gap
        self.p_internal_coords = p_internal_coords

        if p_seq_conditioned < 1.0 or p_knockout_seq > 0.0:
            raise ValueError(
                "p_seq_conditioned < 1.0 or p_knockout_seq > 0.0 not currently supported. "
                "Model must see sequence conditioning if given a distance or coordinate condition"
            )

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]

        ensure_annotations(
            atom_array,
            "is_protein",
            "is_standard_aa",
            "is_res_start",
            "res_min_occupancy",
            "is_ligand",
            "res_has_tip_atom",
        )

        n_eligible_residues = np.sum(
            atom_array.mask("is_standard_aa & is_protein & is_res_start & (res_min_occupancy > 0)")
        )
        n_eligible_ligands = np.sum(atom_array.mask("is_ligand & is_res_start & (res_min_occupancy > 0)"))
        return n_eligible_residues >= 1 and n_eligible_ligands >= 1

    def sample_mask(self, atom_array: AtomArray) -> tuple[np.ndarray, list[np.ndarray]]:
        n_eligible_ligands = atom_array.mask("is_ligand & is_res_start & (res_min_occupancy > 0)").sum()
        n_max_ligands = min(n_eligible_ligands, self.n_max_ligands)
        n_ligands_to_sample = np.random.randint(1, n_max_ligands + 1)

        n_eligible_res = atom_array.mask("res_has_tip_atom & is_res_start & (res_min_occupancy > 0)").sum()
        n_max_res = min(n_eligible_res, self.n_max_residues)
        n_res_to_sample = np.random.randint(1, n_max_res + 1)

        _same_ligand = ("pn_unit_iid",)
        fn_sample_seed = mg.SampleWithPotential(
            is_eligible="is_ligand & (res_min_occupancy > 0)",
            avoid_same=_same_ligand,
            potential=partial(_default_ligand_potential, scaling_factor=self.ligand_scaling_factor),
        )
        fn_grow_seed = mg.GrowByHoppingAlongBondGraph(
            n_hops_expected=self.n_hops_expected_ligand,
            require_same_annotation=_same_ligand,
            atom_array=atom_array,
        )
        fn_check_budget = mg.CheckResidueBudget(
            n_min_residues=n_ligands_to_sample,
            n_max_residues=n_ligands_to_sample,
            reduce=np.any,
        )

        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=fn_sample_seed,
            fn_grow_seed=fn_grow_seed,
            fn_check_budget=fn_check_budget,
        )

        _same_residue = ("chain_iid", "res_name", "res_id", "transformation_id")
        fn_sample_seed = mg.SampleTipAtomWithPotential(
            is_eligible="is_tip_atom & (res_min_occupancy > 0)",
            avoid_same=_same_residue,
            potential=partial(_default_tip_atom_potential, scaling_factor=self.tip_atom_scaling_factor),
        )
        fn_grow_seed = mg.GrowByHoppingAlongBondGraph(
            n_hops_expected=self.n_hops_expected_tip_atom,
            require_same_annotation=_same_residue,
            atom_array=atom_array,
        )
        fn_check_budget = mg.CheckResidueBudget(
            n_min_residues=n_ligands_to_sample + n_res_to_sample,
            n_max_residues=n_ligands_to_sample + n_res_to_sample + 1,
            reduce=np.any,
        )
        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            all_masks=all_masks,
            fn_sample_seed=fn_sample_seed,
            fn_grow_seed=fn_grow_seed,
            fn_check_budget=fn_check_budget,
        )

        return total_mask, all_masks

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_protein_backbone", "is_nucleic_acid_backbone")

        total_mask, all_masks = self.sample_mask(atom_array)

        n_atoms = atom_array.array_length()
        _true = np.ones(n_atoms, dtype=bool)

        # Mark the protein-side conditioning region (tip-atom seed + bond-hop growth).
        atom_array.set_annotation("is_sampled_tip_atom", total_mask & atom_array.is_protein)

        nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
        coord_condition_mask = total_mask & ~nan_coord_mask
        C_CRD.set_mask(atom_array, coord_condition_mask)

        # Possibly convert some absolute coordinate conditioning to internal coordinate conditioning.
        # Sample the number of masks geometrically based on p_internal_coords.
        if self.p_internal_coords > 0 and len(all_masks) > 0:
            if self.p_internal_coords >= 1.0:
                n_to_convert = len(all_masks)
            else:
                n_to_convert = min(
                    np.random.geometric(1 - self.p_internal_coords),
                    len(all_masks),
                )
            selected = np.random.choice(len(all_masks), size=n_to_convert, replace=False)
            for idx in selected:
                atom_array = convert_to_internal_coords(atom_array, all_masks[idx])

        is_seq_conditioned = C_SEQ.mask(atom_array, default="generate")

        if true_with_prob(self.p_seq_conditioned):
            res_segments = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
            is_res_seq_conditioned = apply_segment_wise(res_segments, total_mask, np.any)
            n_eligible_res = is_res_seq_conditioned.sum()
            show_sequence = true_with_prob([1.0 - self.p_knockout_seq] * n_eligible_res)
            is_res_seq_conditioned[is_res_seq_conditioned] = show_sequence
            is_seq_conditioned = spread_segment_wise(res_segments, is_res_seq_conditioned)
            is_seq_conditioned |= atom_array.mask("~is_standard_aa")
            C_SEQ.set_mask(atom_array, is_seq_conditioned)

        # Sample the regions to be unindexed
        show_idx = _true
        if true_with_prob(self.p_unindex_all):
            show_idx[total_mask & atom_array.is_protein & ~atom_array.atomize] = False
            is_unindexed = ~show_idx
            is_unindexed = apply_and_spread_token_wise(atom_array, is_unindexed, np.any)
            show_idx = ~is_unindexed
        show_idx[S_ATM.mask(atom_array, default="generate")] = True  # Never unindex atomized regions
        C_IDX.set_mask(atom_array, show_idx)

        # Sample the unindexed groups
        all_index_group_ids = sample_unindexed_groups(
            atom_array,
            p_merge_groups=self.p_merge_unindexed_groups,
            p_split_group=self.p_split_unindexed_groups,
            p_increase_group_id_gap=self.p_increase_group_id_gap,
        )
        C_IDXGRP.set_annotation(atom_array, all_index_group_ids)

        data["atom_array"] = atom_array

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "all_masks": all_masks,
        }

        data = self.apply_design_task_modifiers(data)
        data = self.ensure_atomized_tokens_are_sequence_conditioned(data)

        return data


class TipAtomActiveSiteTask(RFD3DesignTask):
    """
    Designed to sample tip atoms around a protein pocket when no ligand is present.
    This is done by calculating the SASA of all atoms, and calculating the distance they lie from
    the convex hull of the protein chain. Atoms above a SASA threshold are sampled at rates
    proportional to their distance from the convex hull. In effect this samples tip atoms around protein
    pocket.

    Args:
        tip_atom_rasa_threshold: rASA threshold to consider tip atoms for sampling
        min_tip_atoms: minimum number of tip atoms to sample
        max_tip_atoms: maximum number of tip atoms to sample
        p_seq_conditioned: probability of providing sequence conditioning to atoms with tip-atoms sampled
            NOTE: In proteina, providing tip atoms inherently gives away sequence information because of Atom37!
        min_protein_length: minimum length of protein chain to sample
        p_unindex_all: probability of unindexing all atoms in the protein chain
        p_unindex_any: probability of unindexing individual sampled atom. Will only do this if
            p_index_all was sampled to False
        p_merge_unindexed_groups: probability of merging unindexed groups
        p_split_unindexed_groups: probability of splitting unindexed groups
        p_increase_group_id_gap: probability of increasing the gap between unindexed group ids
    """

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.1},
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.2},
    }

    def __init__(
        self,
        tip_atom_rasa_threshold: float = 0.2,
        min_tip_atoms: int = 2,
        max_tip_atoms: int = 15,
        p_seq_conditioned: float = 1.0,
        min_protein_length: int = 50,  # Avoid peptides
        # Unindexing
        p_unindex_all: float = 0.5,
        p_unindex_any: float = 0.05,
        p_merge_unindexed_groups: float = 0.1,
        p_split_unindexed_groups: float = 0.01,  # Should be rare
        p_increase_group_id_gap: float = 0.1,
        p_internal_coords: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tip_atom_rasa_threshold = tip_atom_rasa_threshold
        self.min_tip_atoms = min_tip_atoms
        self.max_tip_atoms = max_tip_atoms
        self.p_seq_conditioned = p_seq_conditioned
        self.min_protein_length = min_protein_length

        # Unindexing
        self.p_unindex_all = p_unindex_all
        self.p_unindex_any = p_unindex_any
        self.p_merge_unindexed_groups = p_merge_unindexed_groups
        self.p_split_unindexed_groups = p_split_unindexed_groups
        self.p_increase_group_id_gap = p_increase_group_id_gap

        # Internal coords
        self.p_internal_coords = p_internal_coords

        self.sampled_chain = None
        self.sampled_tip_atom_ind = None

    def annotate_for_task_selection(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_tip_atom", "is_protein", "token_min_occupancy")

        data["atom_array"] = atom_array
        return data

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]

        eligible_mask = atom_array.mask("is_protein & (token_min_occupancy > 0)")
        if "at_crop_boundary" in atom_array.get_annotation_categories():
            eligible_mask &= ~atom_array.mask("at_crop_boundary")

        # randomly choose one of the protein chains that is long enough to be our sampled protein chain
        possible_chains = np.unique(atom_array[eligible_mask].chain_id)
        chain_lengths = {
            str(chain_id): len(np.unique(atom_array[atom_array.chain_id == chain_id].res_id))
            for chain_id in possible_chains
        }
        eligible_chains = [chain_id for chain_id, length in chain_lengths.items() if length >= self.min_protein_length]

        if len(eligible_chains) == 0:
            return False

        sampled_chain = np.random.choice(eligible_chains)
        self.sampled_chain = sampled_chain

        # calculate rasa for all atoms
        rasa = calculate_atomwise_rasa(
            atom_array,
            probe_radius=1.4,
            atom_radii="ProtOr",
            point_number=100,
            atom_filter=eligible_mask & (atom_array.chain_id == self.sampled_chain) & atom_array.is_tip_atom,
        )

        atom_array.set_annotation("tip_atom_rasa", rasa)

        return not (rasa > self.tip_atom_rasa_threshold).sum() < self.min_tip_atoms

    def sample_protein_pocket(self, atom_array: AtomArray, eligible_tip_atom_mask: np.ndarray) -> int:
        """
        Calculates convex hull of the protein chain (self.sampled_chain) and samples a tip atom
        based on distance from the convex hull boundary and the rasa (stored in tip_atom_rasa
        annotation). Returns the index of the sampled tip atom.
        """
        # Get chain coordinates, filtering out NaN/Inf values and unresolved atoms
        chain_mask = atom_array.chain_id == self.sampled_chain
        # Also filter by occupancy if available to exclude unresolved atoms
        if "token_min_occupancy" in atom_array.get_annotation_categories():
            chain_mask = chain_mask & (atom_array.token_min_occupancy > 0)

        chain_coords = atom_array[chain_mask].coord
        # Use np.isfinite to catch both NaN and Inf values
        valid_coord_mask = np.isfinite(chain_coords).all(axis=1)

        if valid_coord_mask.sum() < 4:
            raise ValueError(f"Not enough valid coordinates for convex hull (need >= 4, got {valid_coord_mask.sum()})")

        # Make explicit copy to avoid any view issues
        valid_chain_coords = chain_coords[valid_coord_mask].copy()
        convex_hull = ConvexHull(valid_chain_coords)

        eligible_tip_atom_inds = np.where(eligible_tip_atom_mask)[0]
        tip_atom_coords = atom_array.coord[eligible_tip_atom_inds]

        # Filter out tip atoms with NaN/Inf coordinates
        valid_tip_mask = np.isfinite(tip_atom_coords).all(axis=1)
        if valid_tip_mask.sum() == 0:
            raise ValueError("No valid tip atom coordinates (all contain NaN/Inf)")

        eligible_tip_atom_inds = eligible_tip_atom_inds[valid_tip_mask]
        tip_atom_coords = tip_atom_coords[valid_tip_mask].copy()

        distances_to_facets = (
            tip_atom_coords @ convex_hull.equations[:, :-1].T + convex_hull.equations[:, -1]
        )  # (n_tip_atoms, n_facets)
        distances_to_hull_boundary = np.min(distances_to_facets, axis=1)

        probabilities = softmax(distances_to_hull_boundary)
        sampled_tip_atom_ind = np.random.choice(eligible_tip_atom_inds, p=probabilities)
        return sampled_tip_atom_ind

    def sample_mask(self, atom_array: AtomArray) -> np.ndarray:
        is_eligible_query = (
            f"is_tip_atom & (chain_id == '{self.sampled_chain}') & (token_min_occupancy > 0)"
            f" & (tip_atom_rasa > {self.tip_atom_rasa_threshold})"
        )
        if "at_crop_boundary" in atom_array.get_annotation_categories():
            is_eligible_query += " & (at_crop_boundary == False)"

        eligible_tip_atom_mask = atom_array.mask(is_eligible_query)

        sampled_tip_atom_ind = self.sample_protein_pocket(atom_array, eligible_tip_atom_mask)
        sampled_tip_atom_mask = np.zeros(atom_array.array_length(), dtype=bool)
        sampled_tip_atom_mask[sampled_tip_atom_ind] = True

        target_num_tip_atoms = np.random.randint(
            self.min_tip_atoms, min(self.max_tip_atoms, eligible_tip_atom_mask.sum()) + 1
        )

        fn_sample_seed = mg.SampleWithPotential(
            is_eligible=is_eligible_query,
            avoid_same=("atom_id"),  # Effectively no token collision avoidance
            potential=partial(_default_tip_atom_potential, scaling_factor=2.0),
        )

        # remember with seed grow merge, start with the sampled tip atom already as the first
        # all_masks! Then it will sample from there
        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            all_masks=[
                sampled_tip_atom_mask
            ],  # start with sampled tip atom as the first mask. Potential will grow around it
            fn_sample_seed=fn_sample_seed,
            fn_check_budget=mg.CheckAtomBudget(
                n_min_atoms=target_num_tip_atoms,
                n_max_atoms=target_num_tip_atoms,
            ),
        )
        return total_mask, all_masks

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_tip_atom", "tip_atom_rasa")
        assert self.sampled_chain is not None, "sampled_chain was not set"

        total_mask, all_masks = self.sample_mask(atom_array)

        # Mark the conditioning region
        atom_array.set_annotation("is_sampled_tip_atom", total_mask & atom_array.is_protein)

        # set coordinate conditions for sampled atoms
        C_CRD.set_mask(atom_array, total_mask)

        # Possibly convert some absolute coordinate conditioning to internal coordinate conditioning.

        if self.p_internal_coords > 0 and len(all_masks) > 0:
            # In this DesignTask, the individual masks are single atoms, but we will set those
            # in the same residue to have the same internal coordinate group.
            _same_residue = ("chain_iid", "res_name", "res_id", "transformation_id")

            all_masks_residue_level = []
            atom_masks_per_residue = defaultdict(list)
            for mask in all_masks:
                masked_atom_array = atom_array[mask]
                if masked_atom_array.array_length() > 1:
                    all_masks_residue_level.append(mask)
                else:
                    residue_data = tuple(masked_atom_array.get_annotation(category)[0] for category in _same_residue)
                    atom_masks_per_residue[residue_data].append(mask)
            for _, atom_masks in atom_masks_per_residue.items():
                combined_mask = np.zeros(atom_array.array_length(), dtype=bool)
                for m in atom_masks:
                    combined_mask |= m
                all_masks_residue_level.append(combined_mask)

            # Sample the number of masks geometrically based on p_internal_coords.
            if self.p_internal_coords >= 1.0:
                n_to_convert = len(all_masks_residue_level)
            else:
                n_to_convert = min(
                    np.random.geometric(1 - self.p_internal_coords),
                    len(all_masks_residue_level),
                )
            selected = np.random.choice(len(all_masks_residue_level), size=n_to_convert, replace=False)
            for idx in selected:
                atom_array = convert_to_internal_coords(atom_array, all_masks_residue_level[idx])

        # also sequence condition coordinates with sequence conditioning probability
        if true_with_prob(self.p_seq_conditioned):
            seq_cond_mask = apply_and_spread_token_wise(atom_array, total_mask, np.any)
            C_SEQ.set_mask(atom_array, seq_cond_mask)

        # ... unindex atoms appropriately
        show_idx = np.ones(atom_array.array_length(), dtype=bool)
        unindexed_any = False
        if true_with_prob(self.p_unindex_all):
            show_idx[total_mask] = False
            unindexed_any = True
        elif true_with_prob(self.p_unindex_any):
            show_idx[total_mask] = np.random.rand(total_mask.sum()) > self.p_unindex_any
            unindexed_any = True
        if unindexed_any:
            # no partially unindexed tokens!
            is_unindexed = ~show_idx
            is_unindexed = apply_and_spread_token_wise(atom_array, is_unindexed, np.any)
            show_idx = ~is_unindexed
        show_idx[S_ATM.mask(atom_array, default="generate")] = True  # Never unindex atomized regions
        C_IDX.set_mask(atom_array, show_idx)

        # ... sample unindexed groups
        all_index_group_ids = sample_unindexed_groups(
            atom_array,
            p_merge_groups=self.p_merge_unindexed_groups,
            p_split_group=self.p_split_unindexed_groups,
            p_increase_group_id_gap=self.p_increase_group_id_gap,
        )
        C_IDXGRP.set_annotation(atom_array, all_index_group_ids)

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "total_mask": total_mask,
            "target_num_tip_atoms": total_mask.sum(),
        }

        data["atom_array"] = atom_array

        data = self.apply_design_task_modifiers(data)
        data = self.ensure_atomized_tokens_are_sequence_conditioned(data)

        return data


class TipAtomTask(RFD3DesignTask):
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.1},
        "cath_kwargs": {
            "p_add_cath_c_condition": 0.1,
            "p_add_cath_ca_condition": 0.15,
            "p_add_cath_cat_condition": 0.25,
        },
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.2},
    }

    def __init__(
        self,
        min_residues: int = 2,
        max_residues: int = 25,
        p_tip_atom: float = 0.8,
        n_hops_expected: int = 3,
        p_unindex_all: float = 0.9,
        p_merge_unindexed_groups: float = 0.1,
        p_split_unindexed_groups: float = 0.01,
        p_increase_group_id_gap: float = 0.1,
        p_internal_coords: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_residues = min_residues
        self.max_residues = max_residues
        self.n_hops_expected = n_hops_expected
        self.p_tip_atom = p_tip_atom
        self.p_unindex_all = p_unindex_all
        self.p_merge_unindexed_groups = p_merge_unindexed_groups
        self.p_split_unindexed_groups = p_split_unindexed_groups
        self.p_increase_group_id_gap = p_increase_group_id_gap
        self.p_internal_coords = p_internal_coords

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]

        ensure_annotations(
            atom_array,
            "is_protein",
            "is_standard_aa",
            "is_res_start",
            "res_min_occupancy",
            "is_tip_atom",
            "res_has_tip_atom",
        )

        n_eligible_residues = np.sum(
            atom_array.mask("is_standard_aa & is_protein & is_res_start & (res_min_occupancy > 0)")
        )
        return n_eligible_residues >= self.min_residues

    def sample_mask(self, atom_array: AtomArray) -> tuple[np.ndarray, list[np.ndarray], int]:
        n_eligible_residues = np.sum(
            atom_array.mask("is_standard_aa & is_protein & is_res_start & (res_min_occupancy > 0)")
        )

        _same_residue = ("chain_iid", "res_name", "res_id", "transformation_id")

        def fn_sample_seed(
            atom_array: AtomArray,
            total_mask: np.ndarray,
            all_masks: list,
        ) -> np.ndarray:
            if true_with_prob(self.p_tip_atom):
                return mg.SampleTipAtomUniformly(
                    is_eligible="is_tip_atom & (res_min_occupancy > 0)",
                    avoid_same=_same_residue,
                )(atom_array, total_mask, all_masks)

            return mg.SampleUniformly(
                is_eligible="is_standard_aa & is_protein & (res_min_occupancy > 0)",
                avoid_same=_same_residue,
            )(atom_array, total_mask, all_masks)

        fn_grow_seed = mg.GrowByHoppingAlongBondGraph(
            n_hops_expected=self.n_hops_expected,
            require_same_annotation=_same_residue,
            atom_array=atom_array,
        )

        max_residues = min(n_eligible_residues, self.max_residues)
        target_n_residues = np.random.randint(self.min_residues, max_residues + 1)
        fn_check_budget = mg.CheckResidueBudget(
            n_min_residues=target_n_residues,
            n_max_residues=target_n_residues,
            reduce=np.any,
        )

        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=fn_sample_seed,
            fn_grow_seed=fn_grow_seed,
            fn_check_budget=fn_check_budget,
        )

        return total_mask, all_masks, target_n_residues

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        total_mask, all_masks, target_n_residues = self.sample_mask(atom_array)

        n_atoms = atom_array.array_length()
        _true = np.ones(n_atoms, dtype=bool)

        # Mark the conditioning region
        atom_array.set_annotation("is_sampled_tip_atom", total_mask & atom_array.is_protein)

        nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
        coord_condition_mask = total_mask & ~nan_coord_mask
        C_CRD.set_mask(atom_array, coord_condition_mask)

        # Possibly convert some absolute coordinate conditioning to internal coordinate conditioning.
        # Sample the number of masks geometrically based on p_internal_coords.
        if self.p_internal_coords > 0 and len(all_masks) > 0:
            if self.p_internal_coords >= 1.0:
                n_to_convert = len(all_masks)
            else:
                n_to_convert = min(
                    np.random.geometric(1 - self.p_internal_coords),
                    len(all_masks),
                )
            selected = np.random.choice(len(all_masks), size=n_to_convert, replace=False)
            for idx in selected:
                atom_array = convert_to_internal_coords(atom_array, all_masks[idx])

        res_segments = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        is_seq_conditioned = apply_and_spread(res_segments, total_mask, np.any)
        is_seq_conditioned |= atom_array.mask("~is_standard_aa")
        C_SEQ.set_mask(atom_array, is_seq_conditioned)

        # Sample the regions to be unindexed
        show_idx = _true
        if true_with_prob(self.p_unindex_all):
            show_idx[total_mask & atom_array.is_protein & ~atom_array.atomize] = False
            is_unindexed = ~show_idx
            is_unindexed = apply_and_spread_token_wise(atom_array, is_unindexed, np.any)
            show_idx = ~is_unindexed
        show_idx[S_ATM.mask(atom_array, default="generate")] = True  # Never unindex atomized regions
        C_IDX.set_mask(atom_array, show_idx)

        # Sample the unindexed groups
        all_index_group_ids = sample_unindexed_groups(
            atom_array,
            p_merge_groups=self.p_merge_unindexed_groups,
            p_split_group=self.p_split_unindexed_groups,
            p_increase_group_id_gap=self.p_increase_group_id_gap,
        )
        C_IDXGRP.set_annotation(atom_array, all_index_group_ids)

        data["atom_array"] = atom_array

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "target_n_residues": target_n_residues,
            "all_masks": all_masks,
        }

        data = self.apply_design_task_modifiers(data)

        return data


class MotifScaffoldTask(RFD3DesignTask):
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.1},
        "cath_kwargs": {
            "p_add_cath_c_condition": 0.1,
            "p_add_cath_ca_condition": 0.15,
            "p_add_cath_cat_condition": 0.25,
        },
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.2},
    }

    def __init__(
        self,
        min_tokens_per_motif: int = 1,
        max_tokens_per_motif: int = 50,
        max_unindexed_tokens: int = 50,
        min_coverage_fraction: float = 0.05,
        max_coverage_fraction: float = 1.0,
        p_unindex_any: float = 0.5,
        p_merge_unindexed_groups: float = 0.1,
        p_split_unindexed_groups: float = 0.01,
        p_increase_group_id_gap: float = 0.1,
        p_internal_coords: float = 0.0,
        p_backbone_only: float = 0.0,
        p_bb_only_seq_conditioned: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_tokens_per_motif = min_tokens_per_motif
        self.max_tokens_per_motif = max_tokens_per_motif
        self.max_unindexed_tokens = max_unindexed_tokens
        self.min_coverage_fraction = min_coverage_fraction
        self.max_coverage_fraction = max_coverage_fraction
        self.p_unindex_any = p_unindex_any
        self.p_merge_unindexed_groups = p_merge_unindexed_groups
        self.p_split_unindexed_groups = p_split_unindexed_groups
        self.p_increase_group_id_gap = p_increase_group_id_gap
        self.p_internal_coords = p_internal_coords
        self.p_backbone_only = p_backbone_only
        self.p_bb_only_seq_conditioned = p_bb_only_seq_conditioned

    def annotate_for_task_selection(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_token_start", "token_min_occupancy", "is_polymer", "token_id", "is_protein")
        return data

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]

        polymer_chain_iids = np.unique(atom_array.chain_iid[atom_array.is_polymer])
        for chain_iid in polymer_chain_iids:
            chain_length = atom_array.mask(f"is_token_start & (chain_iid == '{chain_iid}')").sum()
            if chain_length >= self.min_tokens_per_motif:
                return True

        return False

    def sample_mask(self, atom_array: AtomArray) -> tuple[np.ndarray, list[np.ndarray]]:
        # Sample coverage fraction per-example (not per-instance) for reproducibility across restarts
        self.coverage_fraction_goal = np.random.uniform(self.min_coverage_fraction, self.max_coverage_fraction)

        fn_sample_seed = mg.SampleUniformly(is_eligible="is_polymer & (token_min_occupancy > 0) & ~has_nan_coord()")
        fn_grow_seed = mg.GrowToTokenIsland(
            island_min_size=self.min_tokens_per_motif,
            island_max_size=self.max_tokens_per_motif,
        )

        total_eligible_tokens = atom_array.mask("is_polymer & is_token_start").sum()
        min_total_tokens = int((self.coverage_fraction_goal - 0.05) * total_eligible_tokens)
        max_total_tokens = int((self.coverage_fraction_goal + 0.05) * total_eligible_tokens)
        fn_check_budget = mg.CheckTokenBudget(
            n_min_tokens=min_total_tokens,
            n_max_tokens=max_total_tokens,
        )
        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=fn_sample_seed,
            fn_grow_seed=fn_grow_seed,
            fn_check_budget=fn_check_budget,
        )

        return total_mask, all_masks

    def sample_unindexed_mask(
        self,
        all_masks: list[np.ndarray],
        token_starts: np.ndarray,
    ) -> np.ndarray:
        """Samples a mask of tokens to be unindexed, ensuring only full coordinate-conditioned
        islands are unindexed.

        NOTE: This function assumes that the coordinate-conditioned islands do not contain partial
        tokens, which is always true for the MotifScaffoldTask.

        Args:
            all_masks: List of boolean masks for each coordinate-conditioned island (shape of each mask is (n_atoms,))
            token_starts: Boolean array indicating the start of each token (shape (n_atoms,))

        Returns:
            unindexed_mask: Boolean array of shape (n_atoms,) where True indicates the token is unindexed
        """

        unindexed_mask = np.zeros_like(all_masks[0], dtype=bool)
        unindexed_mask_indices = []
        target_num_islands_to_unindex = np.random.randint(1, len(all_masks) + 1)

        for _ in range(target_num_islands_to_unindex):
            valid_mask_indices = [
                i
                for i, mask in enumerate(all_masks)
                if i not in unindexed_mask_indices
                and (mask | unindexed_mask)[token_starts].sum() <= self.max_unindexed_tokens
            ]
            if len(valid_mask_indices) == 0:
                break
            mask_index_to_add = np.random.choice(valid_mask_indices)
            unindexed_mask_indices.append(mask_index_to_add)
            unindexed_mask |= all_masks[mask_index_to_add]

        return unindexed_mask

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        total_mask, all_masks = self.sample_mask(atom_array)

        nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
        coord_condition_mask = total_mask & ~nan_coord_mask
        C_CRD.set_mask(atom_array, coord_condition_mask)

        res_segments = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        is_seq_conditioned = apply_and_spread(res_segments, total_mask, np.any)
        C_SEQ.set_mask(atom_array, is_seq_conditioned)

        # Per-residue probability of reducing coord conditioning to backbone atoms only.
        # We do not give sequence conditioning in such cases
        if self.p_backbone_only > 0.0:
            ensure_annotations(atom_array, "is_protein_backbone", "is_rna_backbone", "is_dna_backbone")
            is_backbone = atom_array.is_protein_backbone | atom_array.is_rna_backbone | atom_array.is_dna_backbone
            for start, stop in itertools.pairwise(res_segments):
                if coord_condition_mask[start:stop].any() and true_with_prob(self.p_backbone_only):
                    coord_condition_mask[start:stop] &= is_backbone[start:stop]
                    if true_with_prob(self.p_bb_only_seq_conditioned):
                        is_seq_conditioned[start:stop] = True
                    else:
                        is_seq_conditioned[start:stop] = False
            C_CRD.set_mask(atom_array, coord_condition_mask)
            C_SEQ.set_mask(atom_array, is_seq_conditioned)

        # Sample the regions to be unindexed using an island mask
        # NOTE: Since absolute coords aren't converted to internal coords until later, this may
        # also unindex regions with internal coord conditioning. This is fine for training but the
        # unindexed mapping is not supported at inference
        if true_with_prob(self.p_unindex_any):
            unindexed_mask = self.sample_unindexed_mask(all_masks, get_token_starts(atom_array))
            if "atomize" in atom_array.get_annotation_categories():
                unindexed_mask &= ~atom_array.atomize
            C_IDX.set_mask(atom_array, ~unindexed_mask)
        else:
            C_IDX.set_mask(atom_array, C_IDX.default_mask(atom_array))

        # Sample the unindexed groups
        all_index_group_ids = sample_unindexed_groups(
            atom_array,
            p_merge_groups=self.p_merge_unindexed_groups,
            p_split_group=self.p_split_unindexed_groups,
            p_increase_group_id_gap=self.p_increase_group_id_gap,
        )
        C_IDXGRP.set_annotation(atom_array, all_index_group_ids)

        # By default, coord-conditioned regions all have absolute coords
        C_CRDGRP.set_annotation(atom_array, C_CRDGRP.default_annotation(atom_array))
        C_INTCRD.set_mask(atom_array, C_INTCRD.default_mask(atom_array))
        C_INTCRD.set_annotation(atom_array, C_INTCRD.default_annotation(atom_array))

        # Sample the internal coordinate groups, keeping full coord-conditioned islands together
        if true_with_prob(self.p_internal_coords):
            # permute indices to avoid biases
            mask_indices = np.random.permutation(len(all_masks))

            # We have already decided to give the first group internal coords
            atom_array = convert_to_internal_coords(atom_array, all_masks[mask_indices[0]])

            # Additional internal coord groups are iteratively added with geometrically-decreasing probability
            current_prob_geom = self.p_internal_coords
            masks_added = 1
            while True:
                current_prob_geom *= self.p_internal_coords
                if true_with_prob(current_prob_geom) and masks_added < len(all_masks):
                    atom_array = convert_to_internal_coords(atom_array, all_masks[mask_indices[masks_added]])
                    masks_added += 1
                else:
                    break

        data["atom_array"] = atom_array

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "coverage_fraction_goal": self.coverage_fraction_goal,
            "all_masks": all_masks,
        }

        data = self.apply_design_task_modifiers(data)
        data = self.ensure_atomized_tokens_are_sequence_conditioned(data)

        return data


class MotifScaffoldProteinaTask(RFD3DesignTask):
    """Motif scaffolding task following la proteina logic.

    Supports:
    - residue_selection_mode: "relative_fraction" or "absolute_number"
    - atom_selection_mode: "ca_only", "ame", "tip_atoms", "backbone", "all"
    - Automatic ligand-proximity selection when ligands present (in absolute_number mode)

    Based on MotifMaskTransform from protein-foundation-models public repo.
    """

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.1},
        "cath_kwargs": {
            "p_add_cath_c_condition": 0.1,
            "p_add_cath_ca_condition": 0.15,
            "p_add_cath_cat_condition": 0.25,
        },
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.2},
    }

    def __init__(
        self,
        atom_selection_mode: str = "ca_only",  # "ca_only", "ame", "tip_atoms", "backbone", "all"
        residue_selection_mode: str = "relative_fraction",  # "relative_fraction", "absolute_number"
        motif_prob: float = 1.0,
        # Used if residue_selection_mode == "relative_fraction":
        motif_min_pct_res: float = 0.05,
        motif_max_pct_res: float = 0.5,
        motif_min_n_seg: int = 1,
        motif_max_n_seg: int = 4,
        # Used if residue_selection_mode == "absolute_number":
        motif_min_n_res: int = 1,
        motif_max_n_res: int = 8,
        # Ligand proximity (used in absolute_number mode with ligands)
        ligand_distance_cutoff: float = 8.0,
        # Unindexing - simple mode like la proteina
        unindex_mode: str = "simple",  # "simple", "none", "complex"
        p_unindex_motif: float = 0.5,  # Probability to unindex motif (simple mode only)
        # Complex unindexing parameters (only used if unindex_mode="complex")
        max_unindexed_tokens: int = 50,
        p_merge_unindexed_groups: float = 0.1,
        p_split_unindexed_groups: float = 0.01,
        p_increase_group_id_gap: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.atom_selection_mode = atom_selection_mode
        self.residue_selection_mode = residue_selection_mode
        self.motif_prob = motif_prob
        self.motif_min_pct_res = motif_min_pct_res
        self.motif_max_pct_res = motif_max_pct_res
        self.motif_min_n_seg = motif_min_n_seg
        self.motif_max_n_seg = motif_max_n_seg
        self.motif_min_n_res = motif_min_n_res
        self.motif_max_n_res = motif_max_n_res
        self.ligand_distance_cutoff = ligand_distance_cutoff
        self.unindex_mode = unindex_mode
        self.p_unindex_motif = p_unindex_motif
        self.max_unindexed_tokens = max_unindexed_tokens
        self.p_merge_unindexed_groups = p_merge_unindexed_groups
        self.p_split_unindexed_groups = p_split_unindexed_groups
        self.p_increase_group_id_gap = p_increase_group_id_gap

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]
        ensure_annotations(
            atom_array, "is_protein", "is_standard_aa", "is_token_start", "token_min_occupancy", "token_id"
        )

        n_eligible_residues = atom_array.mask(
            "is_standard_aa & is_protein & is_token_start & (token_min_occupancy > 0)"
        ).sum()
        return n_eligible_residues >= self.motif_min_n_res

    def _select_atoms_by_mode(
        self,
        atom_array: AtomArray,
        residue_mask: np.ndarray,
        res_token_id: int | None = None,
    ) -> np.ndarray:
        """Select atoms within residue based on atom_selection_mode.

        Args:
            atom_array: Full atom array.
            residue_mask: Boolean mask for atoms in this single residue.
            res_token_id: Token ID of the residue (used for some modes).

        Returns:
            Boolean mask for selected atoms within this residue.
        """
        available_atoms = np.where(residue_mask)[0]
        if len(available_atoms) == 0:
            return residue_mask

        if self.atom_selection_mode == "ca_only":
            return residue_mask & (atom_array.atom_name == "CA")

        elif self.atom_selection_mode == "backbone":
            backbone_atoms = {"N", "CA", "C", "O"}
            return residue_mask & np.isin(atom_array.atom_name, list(backbone_atoms))

        elif self.atom_selection_mode == "all":
            return residue_mask

        elif self.atom_selection_mode == "tip_atoms":
            from rfproteina.utils.constants import SIDECHAIN_TIP_ATOMS

            res_name = atom_array.res_name[available_atoms[0]]
            tip_atom_names = SIDECHAIN_TIP_ATOMS.get(res_name, [])

            tip_mask = np.zeros(len(atom_array), dtype=bool)
            for atom_name in tip_atom_names:
                tip_mask |= residue_mask & (atom_array.atom_name == atom_name)

            return tip_mask if tip_mask.any() else (residue_mask & (atom_array.atom_name == "CA"))

        elif self.atom_selection_mode == "ame":
            from rfproteina.utils.constants import AME_ATOMS

            res_name = atom_array.res_name[available_atoms[0]]
            atom_configs = AME_ATOMS.get(res_name, [["CA"]])

            # Randomly select one configuration
            selected_config = atom_configs[np.random.randint(len(atom_configs))]

            ame_mask = np.zeros(len(atom_array), dtype=bool)
            for atom_name in selected_config:
                ame_mask |= residue_mask & (atom_array.atom_name == atom_name)

            return ame_mask if ame_mask.any() else (residue_mask & (atom_array.atom_name == "CA"))

        else:
            raise ValueError(f"Unknown atom_selection_mode: {self.atom_selection_mode}")

    def _select_residues_near_ligand(self, atom_array: AtomArray, motif_n_res: int) -> list[int]:
        """Select residue token IDs near ligands (for absolute_number mode with ligands).

        Args:
            atom_array: Full atom array.
            motif_n_res: Number of residues to select.

        Returns:
            List of token IDs for selected residues.
        """
        if "atomize" not in atom_array.get_annotation_categories():
            return []

        is_ligand = atom_array.atomize
        if not is_ligand.any():
            return []

        # Get binder chain (non-ligand protein)
        is_binder = atom_array.is_protein & ~is_ligand & (atom_array.occupancy > 0)
        if not is_binder.any():
            return []

        from scipy.spatial.distance import cdist

        binder_coords = atom_array.coord[is_binder]
        ligand_coords = atom_array.coord[is_ligand]

        # Compute distances
        dists = cdist(binder_coords, ligand_coords)
        min_dists = dists.min(axis=1)  # min distance to any ligand atom

        # Find binder atoms within cutoff
        binder_atom_indices = np.where(is_binder)[0]
        close_atom_indices = binder_atom_indices[min_dists < self.ligand_distance_cutoff]

        if len(close_atom_indices) == 0:
            return []

        # Group by token (residue)
        close_token_ids = atom_array.token_id[close_atom_indices]
        unique_tokens, counts = np.unique(close_token_ids, return_counts=True)

        # Sort by count (tokens with most atoms near ligand)
        sorted_indices = np.argsort(-counts)

        # Select top motif_n_res tokens, but also try to expand to nearby tokens
        selected_tokens = list(unique_tokens[sorted_indices[:motif_n_res]])

        # If we need more tokens, expand to nearby residues (within 5 positions in sequence)
        if len(selected_tokens) < motif_n_res:
            all_binder_tokens = np.unique(atom_array.token_id[is_binder & atom_array.is_token_start])
            for selected_tok in selected_tokens.copy():
                neighbors = [t for t in all_binder_tokens if abs(t - selected_tok) <= 5 and t not in selected_tokens]
                for neighbor in neighbors:
                    if len(selected_tokens) >= motif_n_res:
                        break
                    selected_tokens.append(neighbor)

        return selected_tokens[:motif_n_res]

    def _select_residues_by_mode(self, atom_array: AtomArray) -> list[int]:
        """Select residue token IDs based on residue_selection_mode.

        Returns:
            List of token IDs for selected residues.
        """
        # Get eligible tokens (protein residues with decent occupancy)
        is_eligible = atom_array.mask("is_standard_aa & is_protein & is_token_start & (token_min_occupancy > 0)")
        eligible_token_ids = np.unique(atom_array.token_id[is_eligible])
        num_eligible = len(eligible_token_ids)

        if num_eligible == 0:
            return []

        if self.residue_selection_mode == "relative_fraction":
            # Select a fraction of residues in contiguous segments
            motif_n_res = int(np.random.uniform(self.motif_min_pct_res, self.motif_max_pct_res) * num_eligible)
            motif_n_res = max(1, min(motif_n_res, num_eligible))

            # Sample number of segments
            motif_n_seg = np.random.randint(self.motif_min_n_seg, min(motif_n_res, self.motif_max_n_seg) + 1)

            # Create segments
            if motif_n_seg > 1:
                split_indices = np.sort(np.random.choice(motif_n_res - 1, motif_n_seg - 1, replace=False)) + 1
                split_indices = np.concatenate([[0], split_indices, [motif_n_res]])
            else:
                split_indices = np.array([0, motif_n_res])

            # Build segment pattern
            segments = []
            for i in range(len(split_indices) - 1):
                segment_len = split_indices[i + 1] - split_indices[i]
                segments.append([True] * segment_len)

            # Add non-motif residues
            segments.extend([[False]] * (num_eligible - motif_n_res))

            # Shuffle segments
            np.random.shuffle(segments)
            motif_sequence_mask = np.array([val for segment in segments for val in segment])

            selected_token_ids = eligible_token_ids[motif_sequence_mask]

        elif self.residue_selection_mode == "absolute_number":
            # Check if ligands are present
            has_ligands = "atomize" in atom_array.get_annotation_categories() and atom_array.atomize.any()

            motif_n_res = np.random.randint(self.motif_min_n_res, self.motif_max_n_res + 1)
            motif_n_res = min(motif_n_res, num_eligible)

            if has_ligands:
                # Use ligand-proximity selection
                selected_token_ids = self._select_residues_near_ligand(atom_array, motif_n_res)
                if len(selected_token_ids) == 0:
                    # Fallback to random if no close residues
                    selected_token_ids = np.random.choice(eligible_token_ids, motif_n_res, replace=False)
            else:
                # Random selection
                selected_token_ids = np.random.choice(eligible_token_ids, motif_n_res, replace=False)

        else:
            raise ValueError(f"Unknown residue_selection_mode: {self.residue_selection_mode}")

        return selected_token_ids

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        # Maybe skip motif creation based on probability
        if not true_with_prob(self.motif_prob):
            # No motif - unconditional generation
            data = self.apply_design_task_modifiers(data)
            data = self.ensure_atomized_tokens_are_sequence_conditioned(data)
            if "task" not in data:
                data["task"] = {"name": self.__class__.__name__}
            data["task"] |= {
                "class": self.__class__.__name__,
                "mask": np.zeros(atom_array.array_length(), dtype=bool),
                "motif_created": False,
            }
            return data

        # Select residues for motif
        selected_token_ids = self._select_residues_by_mode(atom_array)

        if len(selected_token_ids) == 0:
            # Couldn't select any residues
            data = self.apply_design_task_modifiers(data)
            data = self.ensure_atomized_tokens_are_sequence_conditioned(data)
            if "task" not in data:
                data["task"] = {"name": self.__class__.__name__}
            data["task"] |= {
                "class": self.__class__.__name__,
                "mask": np.zeros(atom_array.array_length(), dtype=bool),
                "motif_created": False,
            }
            return data

        # Select atoms within each selected residue
        total_mask = np.zeros(atom_array.array_length(), dtype=bool)
        for token_id in selected_token_ids:
            # Get all atoms in this token
            residue_mask = atom_array.token_id == token_id

            # Select specific atoms based on mode
            atom_mask = self._select_atoms_by_mode(atom_array, residue_mask, token_id)
            total_mask |= atom_mask

        # Set coordinate conditioning for selected atoms
        nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
        coord_condition_mask = total_mask & ~nan_coord_mask
        C_CRD.set_mask(atom_array, coord_condition_mask)

        # Set sequence conditioning for motif residues (spread to entire residue)
        res_segments = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        is_seq_conditioned = apply_and_spread(res_segments, total_mask, np.any)
        is_seq_conditioned |= atom_array.mask("~is_standard_aa")  # Always condition on non-standard AAs
        C_SEQ.set_mask(atom_array, is_seq_conditioned)

        # Handle unindexing based on mode
        if self.unindex_mode == "none":
            # Everything is indexed (no unindexing)
            C_IDX.set_mask(atom_array, C_IDX.default_mask(atom_array))
            C_IDXGRP.set_annotation(atom_array, C_IDXGRP.default_annotation(atom_array))

        elif self.unindex_mode == "simple":
            # Simple mode: with probability p_unindex_motif, unindex all motif residues
            if true_with_prob(self.p_unindex_motif):
                # Unindex all motif residues (spread coord condition to full residues)
                motif_residues_mask = apply_and_spread(res_segments, total_mask, np.any)
                # Only unindex protein residues, not atomized/ligands
                motif_to_unindex = motif_residues_mask & atom_array.is_protein & ~atom_array.atomize
                C_IDX.set_mask(atom_array, ~motif_to_unindex)
            else:
                # Keep everything indexed
                C_IDX.set_mask(atom_array, C_IDX.default_mask(atom_array))

            unindexed_group_ids = sample_unindexed_groups(
                atom_array  # just use default args because we are in simple mode
            )
            C_IDXGRP.set_annotation(atom_array, unindexed_group_ids)

        elif self.unindex_mode == "complex":
            # Complex mode: use island sampling with merge/split logic
            # Create unindexed mask using island sampling
            fn_sample_seed = mg.SampleUniformly(
                is_eligible=f"{C_CRD.mask_name} & is_protein & ~atomize",
            )
            fn_grow_seed = mg.GrowToSegmentIsland(
                island_min_size=1,
                island_max_size=self.max_unindexed_tokens,
                segment_has_same_annotation=("chain_iid", "res_name", "res_id", C_CRD.get_mask_name()),
                require_same_annotation=("chain_iid", C_CRD.get_mask_name(), "atomize"),
            )

            target_fraction = np.random.uniform(0.0, 1.0)
            total_eligible = atom_array.mask(f"is_protein & is_token_start & {C_CRD.get_mask_name()}").sum()
            min_tokens = max(0, int((target_fraction - 0.05) * total_eligible))
            max_tokens = min(int((target_fraction + 0.05) * total_eligible), self.max_unindexed_tokens)

            fn_check_budget = mg.CheckTokenBudget(n_min_tokens=min_tokens, n_max_tokens=max_tokens)

            unindexed_mask, _ = mg.sample_mask_via_seed_grow_merge(
                atom_array,
                fn_sample_seed=fn_sample_seed,
                fn_grow_seed=fn_grow_seed,
                fn_check_budget=fn_check_budget,
            )
            C_IDX.set_mask(atom_array, ~unindexed_mask)

        else:
            raise ValueError(f"Unknown unindex_mode: {self.unindex_mode}. Must be 'none', 'simple', or 'complex'.")

        # Sample the unindexed groups (applies to all modes)
        # This properly assigns group IDs based on contiguous unindexed regions
        all_index_group_ids = sample_unindexed_groups(
            atom_array,
            p_merge_groups=self.p_merge_unindexed_groups,
            p_split_group=self.p_split_unindexed_groups,
            p_increase_group_id_gap=self.p_increase_group_id_gap,
        )
        C_IDXGRP.set_annotation(atom_array, all_index_group_ids)

        data["atom_array"] = atom_array

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "motif_created": True,
            "n_motif_residues": len(selected_token_ids),
            "atom_selection_mode": self.atom_selection_mode,
            "residue_selection_mode": self.residue_selection_mode,
        }

        data = self.apply_design_task_modifiers(data)
        data = self.ensure_atomized_tokens_are_sequence_conditioned(data)

        return data


class BinderHotspotWith3DTarget(RFD3DesignTask):
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.1},
        "cath_kwargs": {
            "p_add_cath_c_condition": 0.1,
            "p_add_cath_ca_condition": 0.15,
            "p_add_cath_cat_condition": 0.25,
        },
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.2},
    }

    def __init__(
        self,
        hotspot_count_distribution: Literal["uniform", "geometric"] = "uniform",
        hotspot_geometric_p: float | None = None,
        max_hotspots_frac_to_provide: float = 0.1,
        hotspot_max_distance: float = 7.0,
        p_no_hotspots: float = 0.05,
        p_sample_ppi_crop: float = 0.75,
        ppi_crop_binder_fraction: float = 0.667,
        p_bb_only_coord_conditioning: float = 0.0,
        target_chain_types: Sequence[ChainType] = (ChainType.POLYPEPTIDE_L, ChainType.POLYPEPTIDE_D),
        **kwargs,
    ):
        """
        This task samples a set of hotspots (atom level) and provides as 1D conditioning to the
        model. Logic and implementation is based on Rafi's code here:
        https://github.com/private-organization/modelhub/blob/aa_design/ppi-v2025_06_24/projects/aa_design/transforms/ppi_transforms.py

        The logic is as follows:
        1. Decide which chain is the binder. We do this by randomly choosing one of the
           query_pn_unit_iids since we assume these chains form a valid interface.
        2. Find all possible hotspots based on the target chain based on whether the atom is within
           hotspot_max_distance to any atom on the target chain.
        3. Sample a set of hotspots from the possible hotspots.

        Args:
            hotspot_count_distribution: Distribution for sampling the number of hotspots. Must be
                ``"geometric"`` or ``"uniform"``.
            hotspot_geometric_p: Success probability for the geometric distribution; mean hotspots
                = 1 / hotspot_geometric_p. Required when ``hotspot_count_distribution`` is
                ``"geometric"``; must be ``None`` when it is ``"uniform"``.
            max_hotspots_frac_to_provide: Hard cap on the fraction of available hotspot atoms to
                provide. The sampled count is capped at this fraction of the total hotspot pool.
            hotspot_max_distance: Maximum distance between a binder atom and a target atom for
                the binder atom to be considered a hotspot candidate.
            p_no_hotspots: Probability of providing no hotspot conditioning at all.
            p_sample_ppi_crop: probability of using the PPI crop, which keeps the entirety of the
                chain designated as the binder if it's smaller than
                ``int(crop_size * ppi_crop_binder_fraction)`` residues. Only consulted when an
                upstream crop has run; otherwise the PPI re-crop is skipped entirely.
            ppi_crop_binder_fraction: Max fraction of the upstream crop budget that can be spent
                on the binder; the rest is reserved for target context. Defaults to ``0.667``.
            p_bb_only_coord_conditioning: Probability of applying backbone-only target conditioning:
                only the backbone atoms of the target chain receive coordinate conditioning (sidechain
                atoms can still be hotspots, but their precise coordinates are not given). Set to 1.0
                for always-on, 0.0 to disable.
            target_chain_types: :py:class:`~atomworks.enums.ChainType` values allowed for the 3D
                target chain. The designed binder is always a protein. Defaults to protein-only
                (classic protein-protein interfaces); add ``ChainType.DNA`` / ``ChainType.RNA`` to
                design binders against nucleic-acid targets.

        Note:
            Crop parameters (``crop_size``, ``crop_center_cutoff_distance``, ``max_atoms_in_crop``)
            are read from ``data["crop_info"]`` at forward() time — populated by the upstream
            crop transform (:py:class:`CropSpatialLikeAF3` / :py:class:`CropContiguousLikeAF3`).
            If no upstream crop ran, the PPI re-crop is skipped and the task operates on
            ``data["atom_array"]`` as-is.
        """
        super().__init__(**kwargs)
        self.hotspot_count_distribution = hotspot_count_distribution
        self.hotspot_geometric_p = hotspot_geometric_p
        self.max_hotspots_frac_to_provide = max_hotspots_frac_to_provide
        self.hotspot_max_distance = hotspot_max_distance
        self.p_no_hotspots = p_no_hotspots
        self.p_sample_ppi_crop = p_sample_ppi_crop
        self.ppi_crop_binder_fraction = ppi_crop_binder_fraction
        self.p_bb_only_coord_conditioning = p_bb_only_coord_conditioning
        # Accept ChainType members directly; coerce str names (e.g. from YAML configs) for convenience.
        self.target_chain_types = tuple(
            ChainType[ct.upper()] if isinstance(ct, str) else ChainType(ct) for ct in target_chain_types
        )

    def annotate_for_task_selection(self, data: dict) -> dict:
        if "crop_info" in data:
            if "atom_array" not in data["crop_info"]:
                raise ValueError("Must preserve uncropped AtomArray in crop_info before applying this design task.")
            atom_array = data["crop_info"]["atom_array"]
        else:
            atom_array = data["atom_array"]

        ensure_annotations(atom_array, "chain_type")
        return data

    def _valid_binder_pn_units(self, atom_array: AtomArray, query_pn_unit_iids: Sequence[str]) -> list[str]:
        """Return the query pn_units that can serve as the (protein) binder.

        A pn_unit is a valid binder if it is a protein chain and every *other* query chain's
        type is in :py:attr:`target_chain_types`. The remaining query chains form the 3D target (
        which may span multiple chains -- e.g. a double-stranded DNA target).

        For a protein-protein interface both chains are valid binders; for a protein-nucleic
        interface only the protein is a valid binder.
        """
        chain_types = {pn: _get_pn_unit_chain_type(atom_array, pn) for pn in query_pn_unit_iids}
        valid_binders: list[str] = []
        for binder in query_pn_unit_iids:
            if chain_types[binder] not in ChainTypeInfo.PROTEINS:
                continue
            other_types = [chain_types[pn] for pn in query_pn_unit_iids if pn != binder]
            if all(ct in self.target_chain_types for ct in other_types):
                valid_binders.append(binder)
        return valid_binders

    def can_apply(self, data: dict) -> bool:
        """
        Can be applied if query_pn_unit_iids has at least two distinct chains that form a valid
        binder/target interface: a protein binder and one or more target chains whose types are in
        :py:attr:`target_chain_types` (protein-only by default; add DNA/RNA for nucleic-acid targets).
        The target may span multiple chains, e.g. a double-stranded DNA target (two nucleic chains).
        Based on Rafi's code in PPICondition.is_valid_for_example() here: https://github.dev/private-organization/modelhub/blob/aa_design/ppi-v2025_06_24/projects/aa_design/transforms/ppi_transforms.py
        """
        query_pn_unit_iids = data.get("query_pn_unit_iids")
        if "crop_info" in data and "atom_array" in data["crop_info"]:
            atom_array = data["crop_info"]["atom_array"]
        else:
            atom_array = data["atom_array"]
        if query_pn_unit_iids is None or len(query_pn_unit_iids) < 2:
            return False
        if len(np.unique(query_pn_unit_iids)) != len(query_pn_unit_iids):
            return False

        for pn_unit_iid in query_pn_unit_iids:
            # make sure pn_unit_iid is actually in the atom array
            if atom_array.mask(f"pn_unit_iid == '{pn_unit_iid}'").sum() == 0:
                return False

        return len(self._valid_binder_pn_units(atom_array, query_pn_unit_iids)) > 0

    def forward(self, data: dict) -> dict:
        query_pn_unit_iids = data["query_pn_unit_iids"]  # existence guaranteed via can_apply()

        # Select the binder chain (always protein); every other query chain is the 3D target.
        # The target may span multiple chains (e.g. double-stranded DNA). Uses the uncropped
        # array (if present) so chain types match can_apply().
        selection_atom_array = (
            data["crop_info"]["atom_array"]
            if "crop_info" in data and "atom_array" in data["crop_info"]
            else data["atom_array"]
        )
        valid_binders = self._valid_binder_pn_units(selection_atom_array, query_pn_unit_iids)
        if len(query_pn_unit_iids) == 2 and len(valid_binders) == 2:
            # Legacy protein-protein path: keep the exact RNG draw (permutation of the two
            # chains) so training trajectories and regression baselines are unchanged.
            binder_pn_unit_iid = np.random.permutation(query_pn_unit_iids)[0]
        else:
            binder_pn_unit_iid = valid_binders[np.random.randint(len(valid_binders))]
        target_pn_unit_iids = [pn for pn in query_pn_unit_iids if pn != binder_pn_unit_iid]

        # Possibly apply the PPI crop instead of the previously-sampled crop
        pre_crop_info = data.get("crop_info")
        perform_ppi_crop = pre_crop_info is not None and true_with_prob(self.p_sample_ppi_crop)
        if perform_ppi_crop:
            crop_size = pre_crop_info["crop_size"]
            max_atoms_in_crop = pre_crop_info["max_atoms_in_crop"]
            crop_center_cutoff_distance = pre_crop_info.get("crop_center_cutoff_distance", 15.0)
            ppi_crop_max_binder_length = int(crop_size * self.ppi_crop_binder_fraction)

            pre_crop_atom_array = pre_crop_info["atom_array"]
            pre_crop_atom_array.set_annotation(
                "is_binder_pn_unit",
                pre_crop_atom_array.pn_unit_iid == binder_pn_unit_iid,
            )

            if (
                get_token_count(pre_crop_atom_array[pre_crop_atom_array.is_binder_pn_unit])
                <= ppi_crop_max_binder_length
            ):
                # Apply the PPI crop
                crop_info = crop_spatial_keep_full_binder(
                    atom_array=pre_crop_atom_array,
                    query_pn_unit_iids=query_pn_unit_iids,
                    binder_pn_unit_iid=binder_pn_unit_iid,
                    crop_size=crop_size,
                    crop_center_cutoff_distance=crop_center_cutoff_distance,
                    force_crop=False,
                )

                # Resize if there are too many atoms
                crop_info = resize_crop_info_if_too_many_atoms(
                    crop_info=crop_info,
                    atom_array=pre_crop_atom_array,
                    max_atoms=max_atoms_in_crop,
                )

                # Overwrite the old crop info with the info for the newly-applied PPI crop.
                # Carry forward crop_size/max_atoms_in_crop/crop_center_cutoff_distance from
                # pre_crop_info -- crop_spatial_keep_full_binder()'s return dict doesn't include
                # them, but EnforceCropBudget (and a subsequent re-crop) still needs them.
                data["crop_info"] = {
                    "type": "PPIFullBinderCropSpatial",
                    "crop_size": crop_size,
                    "max_atoms_in_crop": max_atoms_in_crop,
                    "crop_center_cutoff_distance": crop_center_cutoff_distance,
                } | crop_info
                data["crop_info"]["atom_array"] = pre_crop_atom_array

                # Update data with the newly-cropped atom array
                data["atom_array"] = pre_crop_atom_array[crop_info["crop_atom_idxs"]]

        atom_array = data["atom_array"]

        # sample_hotspot
        if true_with_prob(self.p_no_hotspots):
            hotspot_atom_mask = np.zeros(atom_array.array_length(), dtype=bool)
        else:
            hotspot_atom_mask = sample_hotspots(
                atom_array,
                binder_pn_unit_iid,
                hotspot_max_distance=self.hotspot_max_distance,
                max_hotspots_frac_to_provide=self.max_hotspots_frac_to_provide,
                hotspot_count_distribution=self.hotspot_count_distribution,
                hotspot_geometric_p=self.hotspot_geometric_p,
            )
            if hotspot_atom_mask.sum() == 0:
                ranked_logger.warning(
                    f"No hotspots found for binder chain {binder_pn_unit_iid} on example_id {data['example_id']}"
                )

        # ... set the conditions for the task
        n_atoms = atom_array.array_length()
        _true = np.ones(n_atoms, dtype=bool)
        _false = np.zeros(n_atoms, dtype=bool)
        is_target_chain = np.isin(atom_array.pn_unit_iid, target_pn_unit_iids)
        nan_coord_mask = atom_array.mask("has_nan_coord()")

        C_HOT.set_annotation(atom_array=atom_array, array=hotspot_atom_mask)
        C_HOT.set_mask(atom_array, hotspot_atom_mask, 1, "atom")

        if true_with_prob(self.p_bb_only_coord_conditioning):
            ensure_annotations(atom_array, "is_protein_backbone", "is_nucleic_acid_backbone")
            is_backbone = atom_array.is_protein_backbone | atom_array.is_nucleic_acid_backbone
            C_CRD.set_mask(atom_array, is_target_chain & ~nan_coord_mask & is_backbone)
        else:
            C_CRD.set_mask(atom_array, is_target_chain & ~nan_coord_mask)

        C_SEQ.set_mask(atom_array, is_target_chain)
        # TODO: enable `unindexed` condition
        C_IDX.set_mask(atom_array, _true, 1, "atom")

        # ... update the now condition-annotated atom array
        data["atom_array"] = atom_array

        # ... add the task mask
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            # required fields
            "class": self.__class__.__name__,
            "mask": hotspot_atom_mask,
            # optional fields
            "binder_pn_unit_iid": binder_pn_unit_iid,
            "num_hotspots_to_sample": hotspot_atom_mask.sum(),
        }

        # If we did the ppi crop, we will add CATH labels to the binder chain
        cath_query_pn_unit_iid = binder_pn_unit_iid if perform_ppi_crop else None
        data = self.apply_design_task_modifiers(
            data, overrides={"cath_kwargs": {"query_pn_unit_iid": cath_query_pn_unit_iid}}
        )

        data = self.ensure_atomized_tokens_are_sequence_conditioned(data)

        return data


class NucleicSecondaryStructureTask(RFD3DesignTask):
    """Apply NA secondary-structure conditioning, with optional small motif scaffolds.

    When this task is selected, it computes ``bp_partners`` from the current
    structure and either reveals the full annotation or a partial island mask.
    It can also optionally add a motif-scaffold overlay with coordinate/sequence
    conditioning on a small fraction of polymer tokens.
    """

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.0},
        "hbond_kwargs": {"p_add_hbond_condition": 0.0},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.0},
        "cath_kwargs": {"p_add_cath_c_condition": 0.0, "p_add_cath_ca_condition": 0.0, "p_add_cath_cat_condition": 0.0},
        "na_ss_kwargs": {"p_add_na_ss_condition": 1.0},
        "not_cysteine_kwargs": {"p_add_not_cysteine_condition": 0.0},
    }

    def __init__(
        self,
        min_nucleic_tokens: int = 2,
        p_show_partial_feats: float = 0.7,
        p_add_motif_scaffold_condition: float = 0.0,
        min_tokens_per_motif: int = 1,
        max_tokens_per_motif: int = 12,
        min_motif_coverage_fraction: float = 0.02,
        max_motif_coverage_fraction: float = 0.10,
        max_unindexed_tokens: int = 0,
        p_unindex_any: float = 0.0,
        p_merge_unindexed_groups: float = 0.0,
        p_split_unindexed_groups: float = 0.0,
        p_increase_group_id_gap: float = 0.0,
        p_internal_coords: float = 0.0,
        nucleic_ss_min_shown: float = 0.2,
        nucleic_ss_max_shown: float = 1.0,
        n_islands_min: int = 1,
        n_islands_max: int = 6,
        p_canonical_bp_filter: float = 0.0,
        na_only: bool = False,
        planar_only: bool = True,
        **kwargs,
    ):
        """Initialize NA-SS reveal, optional motif overlay, and annotation filters.

        Control knobs for conditioning control are:
        - ``p_show_partial_feats`` controls full vs. partial ``bp_partners`` reveal.
        - ``nucleic_ss_min_shown`` / ``nucleic_ss_max_shown`` and
          ``n_islands_min`` / ``n_islands_max`` shape the partial reveal.
        - ``p_add_motif_scaffold_condition`` controls whether a motif scaffold is
          additionally sampled.
        - motif sizing/unindexing is controlled by the motif-specific arguments.
        - ``p_canonical_bp_filter``, ``na_only``, and ``planar_only`` control
          which interactions are eligible for annotation.

        Control knobs for motif scaffold sampling are:
         - ``p_add_motif_scaffold_condition`` controls whether a motif scaffold
           is sampled in addition to the NA-SS conditioning.
         - ``min_tokens_per_motif`` and ``max_tokens_per_motif`` control motif size.
         - ``min_motif_coverage_fraction`` and ``max_motif_coverage_fraction``
           control motif scaffold coverage as a fraction of eligible tokens.

        Control knobs for unindexing behavior are:
        - ``max_unindexed_tokens``  maximum number of tokens that can be unindexed in total.
        - ``p_unindex_any`` probability of unindexing any motif at all.
        - ``p_merge_unindexed_groups`` probability of merging two unindexed motif groups.
        - ``p_split_unindexed_groups`` probability of splitting any unindexed motif group.
        - ``p_increase_group_id_gap`` prob of increasing the gap in group IDs between indexed and unindexed groups

        """
        super().__init__(**kwargs)
        self.min_nucleic_tokens = min_nucleic_tokens
        self.p_show_partial_feats = p_show_partial_feats
        self.p_add_motif_scaffold_condition = p_add_motif_scaffold_condition
        self.min_tokens_per_motif = min_tokens_per_motif
        self.max_tokens_per_motif = max_tokens_per_motif
        self.min_motif_coverage_fraction = min_motif_coverage_fraction
        self.max_motif_coverage_fraction = max_motif_coverage_fraction
        self.max_unindexed_tokens = max_unindexed_tokens
        self.p_unindex_any = p_unindex_any
        self.p_merge_unindexed_groups = p_merge_unindexed_groups
        self.p_split_unindexed_groups = p_split_unindexed_groups
        self.p_increase_group_id_gap = p_increase_group_id_gap
        self.p_internal_coords = p_internal_coords
        self.nucleic_ss_min_shown = nucleic_ss_min_shown
        self.nucleic_ss_max_shown = nucleic_ss_max_shown
        self.n_islands_min = n_islands_min
        self.n_islands_max = n_islands_max
        self.p_canonical_bp_filter = p_canonical_bp_filter
        self.na_only = na_only
        self.planar_only = planar_only

    def can_apply(self, data: dict) -> bool:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_nucleic_acid", "is_polymer")
        token_starts = get_token_starts(atom_array)
        if len(token_starts) == 0:
            return False
        if self.na_only:
            n_nucleic_tokens = int(atom_array.is_nucleic_acid[token_starts].sum())
            return n_nucleic_tokens >= self.min_nucleic_tokens
        n_polymer_tokens = int(atom_array.is_polymer[token_starts].sum())
        return n_polymer_tokens >= 2

    def sample_motif_mask(self, atom_array: AtomArray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Sample small polymer token islands to retain as motif scaffolds."""
        coverage_fraction_goal = np.random.uniform(
            self.min_motif_coverage_fraction,
            self.max_motif_coverage_fraction,
        )

        fn_sample_seed = mg.SampleUniformly(is_eligible="is_polymer & (token_min_occupancy > 0) & ~has_nan_coord()")
        fn_grow_seed = mg.GrowToTokenIsland(
            island_min_size=self.min_tokens_per_motif,
            island_max_size=self.max_tokens_per_motif,
        )

        total_eligible_tokens = atom_array.mask("is_polymer & is_token_start").sum()
        min_total_tokens = int(max(1, (coverage_fraction_goal - 0.02) * total_eligible_tokens))
        max_total_tokens = int(max(1, (coverage_fraction_goal + 0.02) * total_eligible_tokens))
        fn_check_budget = mg.CheckTokenBudget(
            n_min_tokens=min_total_tokens,
            n_max_tokens=max_total_tokens,
        )
        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=fn_sample_seed,
            fn_grow_seed=fn_grow_seed,
            fn_check_budget=fn_check_budget,
        )
        self.motif_coverage_fraction_goal = coverage_fraction_goal
        return total_mask, all_masks

    def sample_unindexed_mask(
        self,
        all_masks: list[np.ndarray],
        token_starts: np.ndarray,
    ) -> np.ndarray:
        """Sample motif islands to unindex while preserving whole conditioned islands."""
        unindexed_mask = np.zeros_like(all_masks[0], dtype=bool)
        unindexed_mask_indices = []
        target_num_islands_to_unindex = np.random.randint(1, len(all_masks) + 1)

        for _ in range(target_num_islands_to_unindex):
            valid_mask_indices = [
                i
                for i, mask in enumerate(all_masks)
                if i not in unindexed_mask_indices
                and (mask | unindexed_mask)[token_starts].sum() <= self.max_unindexed_tokens
            ]
            if len(valid_mask_indices) == 0:
                break
            mask_index_to_add = np.random.choice(valid_mask_indices)
            unindexed_mask_indices.append(mask_index_to_add)
            unindexed_mask |= all_masks[mask_index_to_add]

        return unindexed_mask

    def forward(self, data: dict) -> dict:
        atom_array = as_atom_array_plus(data["atom_array"])
        ensure_annotations(atom_array, "is_nucleic_acid", "is_polymer", "is_token_start", "token_min_occupancy")

        nucleic_mask = atom_array.is_nucleic_acid.astype(bool)
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": nucleic_mask,
        }

        atom_array = apply_training_na_ss_condition(
            atom_array,
            p_show_partial_feats=self.p_show_partial_feats,
            nucleic_ss_min_shown=self.nucleic_ss_min_shown,
            nucleic_ss_max_shown=self.nucleic_ss_max_shown,
            n_islands_min=self.n_islands_min,
            n_islands_max=self.n_islands_max,
            p_canonical_bp_filter=self.p_canonical_bp_filter,
            NA_only=self.na_only,
            planar_only=self.planar_only,
        )

        if true_with_prob(self.p_add_motif_scaffold_condition):
            total_mask, all_masks = self.sample_motif_mask(atom_array)

            nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
            coord_condition_mask = total_mask & ~nan_coord_mask
            C_CRD.set_mask(atom_array, coord_condition_mask)

            res_segments = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
            is_seq_conditioned = apply_and_spread(res_segments, total_mask, np.any)
            C_SEQ.set_mask(atom_array, is_seq_conditioned)

            if true_with_prob(self.p_unindex_any):
                unindexed_mask = self.sample_unindexed_mask(all_masks, get_token_starts(atom_array))
                if "atomize" in atom_array.get_annotation_categories():
                    unindexed_mask &= ~atom_array.atomize
                C_IDX.set_mask(atom_array, ~unindexed_mask)
            else:
                C_IDX.set_mask(atom_array, C_IDX.default_mask(atom_array))

            all_index_group_ids = sample_unindexed_groups(
                atom_array,
                p_merge_groups=self.p_merge_unindexed_groups,
                p_split_group=self.p_split_unindexed_groups,
                p_increase_group_id_gap=self.p_increase_group_id_gap,
            )
            C_IDXGRP.set_annotation(atom_array, all_index_group_ids)
            C_CRDGRP.set_annotation(atom_array, C_CRDGRP.default_annotation(atom_array))
            C_INTCRD.set_mask(atom_array, C_INTCRD.default_mask(atom_array))
            C_INTCRD.set_annotation(atom_array, C_INTCRD.default_annotation(atom_array))

            if true_with_prob(self.p_internal_coords) and len(all_masks) > 0:
                mask_indices = np.random.permutation(len(all_masks))
                atom_array = convert_to_internal_coords(atom_array, all_masks[mask_indices[0]])

                current_prob_geom = self.p_internal_coords
                masks_added = 1
                while True:
                    current_prob_geom *= self.p_internal_coords
                    if true_with_prob(current_prob_geom) and masks_added < len(all_masks):
                        atom_array = convert_to_internal_coords(atom_array, all_masks[mask_indices[masks_added]])
                        masks_added += 1
                    else:
                        break

            data["task"] |= {
                "motif_mask": total_mask,
                "motif_all_masks": all_masks,
                "motif_coverage_fraction_goal": getattr(self, "motif_coverage_fraction_goal", None),
            }

        data["atom_array"] = atom_array
        data = self.apply_design_task_modifiers(data)
        data = self.ensure_atomized_tokens_are_sequence_conditioned(data)
        return data


# ---------------------------------------------------------------------------
# Helpers for IgFrameworkConditioningTask
# ---------------------------------------------------------------------------


class IgRegion(StrEnum):
    """Allowed values of the per-atom ``ig_region`` CIF annotation."""

    NOT_IG = ""
    FRAMEWORK = "framework"
    CDR1 = "cdr1"
    CDR2 = "cdr2"
    CDR3 = "cdr3"
    CDR4 = "cdr4"


IG_CDR_REGIONS = (IgRegion.CDR1, IgRegion.CDR2, IgRegion.CDR3, IgRegion.CDR4)


def _expand_cdr_mask(
    atom_array: AtomArray,
    cdr_mask: np.ndarray,
    boundary_mask: np.ndarray,
    expand_n_term: int,
    expand_c_term: int,
) -> np.ndarray:
    """Expand (>0) or contract (<0) a CDR atom mask independently at each terminus.

    Each terminus is grown/shrunk on its own, so e.g. ``expand_n_term=+1`` with
    ``expand_c_term=-1`` shifts the CDR mask by one residue toward the C-terminus
    rather than resizing it. Expansion is clamped to boundary_mask (the Ig chain);
    contraction erodes boundary residues from the chosen terminus.

    Args:
        atom_array: The full atom array.
        cdr_mask: Bool atom mask of the CDR region to expand/contract.
        boundary_mask: Bool atom mask bounding the expansion (Ig chain).
        expand_n_term: Residues to add (>0) or remove (<0) at the N-terminus
            (lower residue indices).
        expand_c_term: Residues to add (>0) or remove (<0) at the C-terminus
            (higher residue indices).

    Returns:
        New bool atom mask after expansion/contraction.
    """
    if expand_n_term == 0 and expand_c_term == 0:
        return cdr_mask

    res_starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    n_res = len(res_starts) - 1
    n_atoms = atom_array.array_length()

    # Build residue-level masks
    res_is_cdr = np.array([cdr_mask[res_starts[i] : res_starts[i + 1]].any() for i in range(n_res)])
    res_in_boundary = np.array([boundary_mask[res_starts[i] : res_starts[i + 1]].any() for i in range(n_res)])

    new_res = res_is_cdr.copy()
    # N-terminus = lower residue indices ("left"), C-terminus = higher ("right").
    for amount, toward_n_term in ((expand_n_term, True), (expand_c_term, False)):
        for _ in range(amount):  # grow toward this terminus
            grown = np.zeros(n_res, dtype=bool)
            if toward_n_term:
                grown[:-1] = new_res[1:]
            else:
                grown[1:] = new_res[:-1]
            new_res = (new_res | grown) & res_in_boundary
        for _ in range(-amount):  # shrink from this terminus
            neighbor = np.zeros(n_res, dtype=bool)
            if toward_n_term:
                neighbor[1:] = new_res[:-1]  # keep only residues with an N-side neighbor
            else:
                neighbor[:-1] = new_res[1:]  # keep only residues with a C-side neighbor
            new_res = new_res & neighbor

    new_atom_mask = np.zeros(n_atoms, dtype=bool)
    for i in range(n_res):
        if new_res[i]:
            new_atom_mask[res_starts[i] : res_starts[i + 1]] = True
    return new_atom_mask


class IgFrameworkConditioningTask(RFD3DesignTask):
    """Framework-conditioning task for antibodies and TCRs.

    Conditions on the Ig framework and (optionally) the target/antigen while
    leaving CDR loops free for design. It:

    - reads the native ``ig_region`` structure annotation,
    - replaces 2D distance conditioning with internal-coordinate conditioning,
    - supports variable CDR boundary expansion/contraction,
    - supports per-CDR fixation (internal coords + sequence conditioning), and
    - supports separate rigid-body coordinate groups for multi-chain Ig molecules
      (e.g. scFv heavy + light).

    Works for both antibodies and TCRs.

    Args:
        cdr_expand_min: Minimum residue expansion per CDR side (negative = shrink).
        cdr_expand_max: Maximum residue expansion per CDR side.
        p_fix_all_but_cdr3: Probability of fixing every CDR **except** CDR3,
            forcing CDR3 to be designed. When False the per-CDR probability
            ``p_cdr_fixed`` is used instead. CDR4 is excluded from this logic;
            see ``p_design_cdr4``.
        p_cdr_fixed: Per-CDR probability of fixing (internal coords + seq
            conditioning). Applied independently to each CDR (except CDR4)
            when ``p_fix_all_but_cdr3`` is not triggered.
        p_design_cdr4: Probability of including CDR4 in the designed (free)
            region. Sampled once per example and applied to CDR4 on every Ig
            chain; otherwise CDR4 follows the framework conditioning policy.
        p_framework_seq_cond: Probability of providing sequence conditioning
            on the framework in addition to internal-coord conditioning.
        p_framework_bb_only: Probability of restricting framework internal-coord
            conditioning to backbone atoms only (CA/N/C/O) rather than all atoms.
        p_flexible_target: Probability of giving the target chain **no**
            coordinate conditioning at all (free antigen).
        p_target_intcoord: Probability of using internal-coord conditioning for
            the target (given it is conditioned). Otherwise absolute coords.
        p_target_bb_only: Probability of restricting target conditioning to
            backbone atoms only.
        p_separate_ig_coord_groups: When multiple Ig chains are present
            (e.g. heavy + light), probability of assigning each its own rigid-body
            coordinate group. Otherwise all Ig chains are in one group.
        p_no_hotspots: Probability of providing no hotspot conditioning.
        hotspot_max_distance: Å threshold for hotspot eligibility.
        max_hotspots_frac_to_provide: Hard cap on the fraction of interface
            atoms to label as hotspots.
        hotspot_count_distribution: ``"uniform"`` or ``"geometric"``.
        hotspot_geometric_p: Geometric distribution parameter (required when
            ``hotspot_count_distribution="geometric"``).
    """

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.6},
        "hbond_kwargs": {"p_add_hbond_condition": 0.2},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.1},
        "cath_kwargs": {
            "p_add_cath_c_condition": 0.1,
            "p_add_cath_ca_condition": 0.15,
            "p_add_cath_cat_condition": 0.25,
        },
    }

    def __init__(
        self,
        cdr_expand_min: int = -2,
        cdr_expand_max: int = 5,
        p_fix_all_but_cdr3: float = 0.3,
        p_cdr_fixed: float = 0.2,
        p_design_cdr4: float = 0.1,
        p_framework_seq_cond: float = 0.8,
        p_framework_bb_only: float = 0.3,
        p_flexible_target: float = 0.1,
        p_target_intcoord: float = 0.5,
        p_target_bb_only: float = 0.3,
        p_separate_ig_coord_groups: float = 0.5,
        p_no_hotspots: float = 0.05,
        hotspot_max_distance: float = 7.0,
        max_hotspots_frac_to_provide: float = 0.1,
        hotspot_count_distribution: Literal["uniform", "geometric"] = "uniform",
        hotspot_geometric_p: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cdr_expand_min = cdr_expand_min
        self.cdr_expand_max = cdr_expand_max
        self.p_fix_all_but_cdr3 = p_fix_all_but_cdr3
        self.p_cdr_fixed = p_cdr_fixed
        self.p_design_cdr4 = p_design_cdr4
        self.p_framework_seq_cond = p_framework_seq_cond
        self.p_framework_bb_only = p_framework_bb_only
        self.p_flexible_target = p_flexible_target
        self.p_target_intcoord = p_target_intcoord
        self.p_target_bb_only = p_target_bb_only
        self.p_separate_ig_coord_groups = p_separate_ig_coord_groups
        self.p_no_hotspots = p_no_hotspots
        self.hotspot_max_distance = hotspot_max_distance
        self.max_hotspots_frac_to_provide = max_hotspots_frac_to_provide
        self.hotspot_count_distribution = hotspot_count_distribution
        self.hotspot_geometric_p = hotspot_geometric_p

    # ------------------------------------------------------------------
    # Pipeline hooks
    # ------------------------------------------------------------------

    def annotate_for_task_selection(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        assert "ig_region" in atom_array.get_annotation_categories(), (
            "IgFrameworkConditioningTask requires the per-atom 'ig_region' annotation. Generate it with "
            "atomworks.ml.utils.antibody.IgAnnotation.get_masks(), attach it with "
            "AtomArray.set_annotation('ig_region', values), save it with AtomWorks "
            "to_cif_file(..., extra_fields=['ig_region']), and load it with "
            "ParseConfig(extra_fields=['ig_region']) or equivalent parser_args."
        )
        ensure_annotations(atom_array, "is_protein")
        regions = atom_array.get_annotation("ig_region")
        invalid_regions = sorted(
            str(region) for region in set(np.unique(regions)) - {region.value for region in IgRegion}
        )
        if invalid_regions:
            raise ValueError(f"Invalid ig_region value(s): {invalid_regions}")

        per_chain_cdr_masks: dict[str, dict[IgRegion, np.ndarray]] = {}
        per_chain_ig_masks: dict[str, np.ndarray] = {}

        ig_mask = regions != IgRegion.NOT_IG.value
        for pn_unit_iid in unique_by_first_occurrence(atom_array.pn_unit_iid[ig_mask]):
            pn_unit_mask = atom_array.pn_unit_iid == pn_unit_iid
            cdr_masks = {
                region: pn_unit_mask & (regions == region.value)
                for region in IG_CDR_REGIONS
                if np.any(pn_unit_mask & (regions == region.value))
            }
            per_chain_cdr_masks[pn_unit_iid] = cdr_masks
            per_chain_ig_masks[pn_unit_iid] = pn_unit_mask & ig_mask

        data["_ig_masks"] = {
            "per_chain_cdr_masks": per_chain_cdr_masks,
            "per_chain_ig_masks": per_chain_ig_masks,
        }
        return data

    def can_apply(self, data: dict) -> bool:
        ig_masks = data.get("_ig_masks")
        if not ig_masks:
            return False

        per_chain_cdr_masks = ig_masks["per_chain_cdr_masks"]
        per_chain_ig_masks = ig_masks["per_chain_ig_masks"]

        if not per_chain_ig_masks:
            return False

        # Need at least one valid CDR on at least one Ig chain
        any_cdr = any(any(m.any() for m in chain_cdrs.values()) for chain_cdrs in per_chain_cdr_masks.values())
        if not any_cdr:
            return False

        # All Ig chains must have at least one CDR
        for pn_unit_iid in per_chain_ig_masks:
            chain_cdrs = per_chain_cdr_masks.get(pn_unit_iid, {})
            if not any(m.any() for m in chain_cdrs.values()):
                return False

        # There must be target atoms (not Ig chain)
        atom_array = data["atom_array"]
        combined_ig_mask = np.zeros(atom_array.array_length(), dtype=bool)
        for ig_mask in per_chain_ig_masks.values():
            combined_ig_mask |= ig_mask

        target_mask = atom_array.is_protein & ~combined_ig_mask
        return target_mask.any()

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ig_masks = data["_ig_masks"]

        per_chain_cdr_masks = ig_masks["per_chain_cdr_masks"]
        per_chain_ig_masks = ig_masks["per_chain_ig_masks"]

        # Combined Ig mask across all Ig chains
        combined_ig_mask = np.zeros(atom_array.array_length(), dtype=bool)
        for ig_mask in per_chain_ig_masks.values():
            combined_ig_mask |= ig_mask

        # Target mask: everything that isn't an Ig PN unit
        target_mask = ~np.isin(atom_array.pn_unit_iid, np.unique(atom_array.pn_unit_iid[combined_ig_mask]))

        # ------------------------------------------------------------------
        # Step 1: CDR expansion — sample each terminus independently, apply to all CDRs.
        # Sampling the two ends separately lets the mask shift (e.g. +1 / -1) rather
        # than only grow or shrink symmetrically about the CDR center.
        # ------------------------------------------------------------------
        expand_n_term = int(np.random.randint(self.cdr_expand_min, self.cdr_expand_max + 1))
        expand_c_term = int(np.random.randint(self.cdr_expand_min, self.cdr_expand_max + 1))

        expanded_cdr_masks: dict[str, dict[IgRegion, np.ndarray]] = {}
        for pn_unit_iid, chain_cdrs in per_chain_cdr_masks.items():
            ig_mask = per_chain_ig_masks[pn_unit_iid]
            expanded_cdr_masks[pn_unit_iid] = {}
            for cdr_region, cdr_mask in chain_cdrs.items():
                expanded_cdr_masks[pn_unit_iid][cdr_region] = _expand_cdr_mask(
                    atom_array, cdr_mask, ig_mask, expand_n_term, expand_c_term
                )

        # ------------------------------------------------------------------
        # Step 2: Decide which CDRs are fixed (internal coord + seq cond)
        # ------------------------------------------------------------------
        fixed_cdr_mask = np.zeros(atom_array.array_length(), dtype=bool)
        free_cdr_mask = np.zeros(atom_array.array_length(), dtype=bool)

        if true_with_prob(self.p_fix_all_but_cdr3):
            # Fix all CDRs except CDR3; leave CDR3 free for design. CDR4 is
            # excluded here and resolved separately below via p_design_cdr4.
            for chain_cdrs in expanded_cdr_masks.values():
                for cdr_region, cdr_mask in chain_cdrs.items():
                    if cdr_region is IgRegion.CDR4:
                        continue
                    if cdr_region is IgRegion.CDR3:
                        free_cdr_mask |= cdr_mask
                    else:
                        fixed_cdr_mask |= cdr_mask
        else:
            # Per-CDR independent fixation probability (CDR4 excluded, see below).
            for chain_cdrs in expanded_cdr_masks.values():
                for cdr_region, cdr_mask in chain_cdrs.items():
                    if cdr_region is IgRegion.CDR4:
                        continue
                    if true_with_prob(self.p_cdr_fixed):
                        fixed_cdr_mask |= cdr_mask
                    else:
                        free_cdr_mask |= cdr_mask

        # CDR4 is designed with one draw shared across all Ig chains; otherwise
        # it follows the framework conditioning policy.
        design_cdr4 = true_with_prob(self.p_design_cdr4)
        for chain_cdrs in expanded_cdr_masks.values():
            cdr4_mask = chain_cdrs.get(IgRegion.CDR4)
            if cdr4_mask is None:
                continue
            if design_cdr4:  # otherwise treated like a framework region
                free_cdr_mask |= cdr4_mask

        # Framework = Ig chain minus ALL CDRs (both fixed and free)
        all_cdr_mask = fixed_cdr_mask | free_cdr_mask
        framework_mask = combined_ig_mask & ~all_cdr_mask

        # ------------------------------------------------------------------
        # Step 3: Framework internal-coord conditioning (mark to be conditioned)
        # ------------------------------------------------------------------
        ensure_annotations(atom_array, "is_protein_backbone")
        fw_conditioning_mask = (
            framework_mask & atom_array.is_protein_backbone
            if true_with_prob(self.p_framework_bb_only)
            else framework_mask
        )
        nan_coord_mask = np.any(np.isnan(atom_array.coord), axis=1)
        fw_coordinate_conditioning_mask = fw_conditioning_mask & ~nan_coord_mask

        # ------------------------------------------------------------------
        # Step 4: Fixed CDR internal-coord conditioning + seq conditioning
        # ------------------------------------------------------------------
        fixed_cdr_conditioning_mask = fixed_cdr_mask & ~nan_coord_mask
        if fixed_cdr_mask.any():
            C_SEQ.set_mask(atom_array, C_SEQ.mask(atom_array, default="generate") | fixed_cdr_mask)

        # ------------------------------------------------------------------
        # Step 5: Framework sequence conditioning (optional)
        # ------------------------------------------------------------------
        if true_with_prob(self.p_framework_seq_cond):
            C_SEQ.set_mask(atom_array, C_SEQ.mask(atom_array, default="generate") | framework_mask)
        else:  # need to remove any type of coordinate conditioning from framework sidechains (keep backbone coords)
            fw_coordinate_conditioning_mask = fw_coordinate_conditioning_mask & atom_array.is_protein_backbone

        # ------------------------------------------------------------------
        # Step 5.5 Apply internal coord conditioning for Ig chains
        # ------------------------------------------------------------------
        atom_array = convert_to_internal_coords(
            atom_array, fw_coordinate_conditioning_mask | fixed_cdr_conditioning_mask
        )

        # ------------------------------------------------------------------
        # Step 6: Multiple Ig chains — coord groups
        # ------------------------------------------------------------------
        ig_chain_ids = list(per_chain_ig_masks.keys())
        if len(ig_chain_ids) > 1 and true_with_prob(self.p_separate_ig_coord_groups):
            # Each Ig chain gets its own coord group (can translate/rotate independently)
            for pn_unit_iid in ig_chain_ids:
                ig_mask = per_chain_ig_masks[pn_unit_iid]
                # Only re-group atoms that already received internal coord conditioning
                intcrd_mask = C_INTCRD.mask(atom_array, default="generate")
                chain_intcrd = ig_mask & intcrd_mask
                if chain_intcrd.any():
                    # Reassign to a fresh coord group by calling convert again
                    # (it always picks max+1, so each call creates a new group)
                    # We only update the CRDGRP annotation, not re-run the full conditioning
                    coord_groups = C_CRDGRP.annotation(atom_array, default="generate")
                    coord_groups[chain_intcrd] = coord_groups.max() + 1
                    C_CRDGRP.set_annotation(atom_array, coord_groups)
        # (else: all Ig chains are already in the same group from convert_to_internal_coords calls)

        # ------------------------------------------------------------------
        # Step 7: Target conditioning
        # ------------------------------------------------------------------
        target_nan_mask = np.any(np.isnan(atom_array.coord), axis=1)

        if true_with_prob(self.p_flexible_target):
            # No coordinate conditioning for target
            coord_groups = C_CRDGRP.annotation(atom_array, default="generate")
            coord_groups[target_mask] = C_CRDGRP.UNKNOWN_COORDS
            C_CRDGRP.set_annotation(atom_array, coord_groups)
        elif true_with_prob(self.p_target_intcoord):
            # Internal-coord conditioning for target
            target_cond_mask = (
                target_mask & atom_array.is_protein_backbone if true_with_prob(self.p_target_bb_only) else target_mask
            )
            atom_array = convert_to_internal_coords(atom_array, target_cond_mask & ~target_nan_mask)
        else:
            # Absolute-coord conditioning for target
            target_cond_mask = (
                target_mask & atom_array.is_protein_backbone if true_with_prob(self.p_target_bb_only) else target_mask
            )
            C_CRD.set_mask(
                atom_array, C_CRD.mask(atom_array, default="generate") | (target_cond_mask & ~target_nan_mask)
            )

        # Target always gets sequence conditioning
        C_SEQ.set_mask(atom_array, C_SEQ.mask(atom_array, default="generate") | target_mask)

        # ------------------------------------------------------------------
        # Step 8: Hotspots
        # ------------------------------------------------------------------
        if true_with_prob(self.p_no_hotspots):
            hotspot_mask = np.zeros(atom_array.array_length(), dtype=bool)
        else:
            hotspot_mask = sample_hotspots(
                atom_array,
                ig_chain_ids,
                hotspot_max_distance=self.hotspot_max_distance,
                max_hotspots_frac_to_provide=self.max_hotspots_frac_to_provide,
                hotspot_count_distribution=self.hotspot_count_distribution,
                hotspot_geometric_p=self.hotspot_geometric_p,
            )

        C_HOT.set_annotation(atom_array=atom_array, array=hotspot_mask)
        C_HOT.set_mask(atom_array, hotspot_mask)

        # ------------------------------------------------------------------
        # Step 9: Index everything
        # ------------------------------------------------------------------
        C_IDX.set_mask(atom_array, np.ones(atom_array.array_length(), dtype=bool))

        data["atom_array"] = atom_array
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": hotspot_mask,
            "ig_chain_ids": ig_chain_ids,
            "target_mask": target_mask,
            "num_hotspots": hotspot_mask.sum(),
            "cdr_expand_n_term": expand_n_term,
            "cdr_expand_c_term": expand_c_term,
            "design_cdr4": design_cdr4,
        }

        data = self.apply_design_task_modifiers(data)
        data = self.ensure_atomized_tokens_are_sequence_conditioned(data)
        return data
