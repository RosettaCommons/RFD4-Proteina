from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import ClassVar, Literal

import biotite.structure as struc
import numpy as np
from atomworks.common import exists
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array import apply_and_spread
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.scatter import (
    apply_segment_wise,
    spread_segment_wise,
)
from atomworks.ml.conditions import (
    C_CRD,
    C_DIS,
    C_IDX,
    C_SEQ,
)
from atomworks.ml.transforms import mask_generator as mg
from atomworks.ml.transforms.crop import resize_crop_info_if_too_many_atoms
from atomworks.ml.transforms.design_task import DesignTaskABC
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from atomworks.ml.utils.token import apply_and_spread_token_wise, get_token_count, get_token_starts
from biotite.structure import AtomArray
from scipy.spatial import ConvexHull
from scipy.special import softmax

from foundry.common import true_with_prob
from foundry.utils.ddp import RankedLogger
from proteinfoundation.conditions import C_ABH, C_ABL, C_CDR, C_CRDGRP, C_HOT, C_IDXGRP, C_INTCRD
from proteinfoundation.transforms.design_task_modifiers import (
    add_cath_condition,
    add_hbond_condition,
    add_rasa_condition,
    add_secondary_structure_condition,
)
from proteinfoundation.transforms.utils.conditioning_utils import (
    convert_to_internal_coords,
    sample_unindexed_groups,
)
from proteinfoundation.transforms.utils.distance_condition import (
    sample_distance_condition,
)
from proteinfoundation.transforms.utils.ppi_utils import (
    crop_spatial_keep_full_binder,
    sample_hotspots,
)


# Default potential functions for TipAtomPartialLigandTask (module-level for picklability)
def _default_ligand_potential(x: np.ndarray, scaling_factor: float = 5.0) -> np.ndarray:
    """Default ligand sampling potential: inverse distance with offset."""
    return 1.0 / (scaling_factor + x)


def _default_tip_atom_potential(x: np.ndarray, scaling_factor: float = 8.0) -> np.ndarray:
    """Default tip atom sampling potential: exponential decay."""
    return np.exp(-x / scaling_factor)


ranked_logger = RankedLogger(__name__, rank_zero_only=True)


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
        "cutoff_HA_dist": 2.2,
        "cutoff_DA_distance": 3.2,
        "p_hbond_subsample": 0.5,
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
}


class RFD3DesignTask(DesignTaskABC):
    """Wrapper class for DesignTasks used to train RFD3

    The concept behind this class is to allow for common functions to be called by each
    design task without having to specify all kwargs in each design task separately.
    All kwargs and default values are specified in the RFD3_DESIGN_TASK_DEFAULTS constant.

    **kwargs: See RFD3_DESIGN_TASK_DEFAULTS for details.
    """

    # Overrides for the RFD3_DESIGN_TASK_DEFAULTS. Subclasses can modify their defaults or set to {}
    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: dict[str, dict]

    def __init_subclass__(cls):
        super().__init_subclass__()
        if not hasattr(cls, "RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES"):
            raise TypeError(
                f"{cls.__name__} must define class variable " "`RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES` (even if empty)."
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
        secondary_structure_kwargs = self.secondary_structure_kwargs.copy() | overrides.get(
            "secondary_structure_kwargs", {}
        )
        cath_kwargs = self.cath_kwargs.copy() | overrides.get("cath_kwargs", {})

        # Set defaults for task dictionary
        data["task"] |= {
            "added_rasa_condition": False,
            "added_hbond_condition": False,
            "added_secondary_structure_condition": False,
            "added_cath_c_condition": False,
            "added_cath_ca_condition": False,
            "added_cath_cat_condition": False,
        }

        # Possibly add Conditions corresponding to design task modifiers
        if true_with_prob(rasa_kwargs.pop("p_add_rasa_condition")):
            data = add_rasa_condition(data, **rasa_kwargs)
            data["task"]["added_rasa_condition"] = True
        if true_with_prob(hbond_kwargs.pop("p_add_hbond_condition")):
            data = add_hbond_condition(data, **hbond_kwargs)
            data["task"]["added_hbond_condition"] = True
        if true_with_prob(secondary_structure_kwargs.pop("p_add_secondary_structure_condition")):
            data = add_secondary_structure_condition(data, **secondary_structure_kwargs)
            data["task"]["added_secondary_structure_condition"] = True

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
    }

    def forward(self, data: dict) -> dict:
        data = super().set_task_info(data)

        # ... show sequence to the model
        C_SEQ.set_mask(data["atom_array"], np.ones(data["atom_array"].array_length(), dtype=bool))

        data = self.apply_design_task_modifiers(data)  # This won't do anything because probs are defualted to 0.0

        return data


class InverseFoldingTask(NoMotifABC):
    """
    Inverse folding task. For non-protein molecules, we provide all coordinates and sequence information.
    For proteins, we provide backbone coordinates only.

    TODO: Extend to DNA/RNA sequence design as well once the model can generate those sequences.
    """

    RFD3_DESIGN_TASK_DEFAULTS_OVERRIDES: ClassVar[dict[str, dict]] = {
        "rasa_kwargs": {"p_add_rasa_condition": 0.0},
        "hbond_kwargs": {"p_add_hbond_condition": 0.0},
        "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.0},
        "cath_kwargs": {"p_add_cath_c_condition": 0.0, "p_add_cath_ca_condition": 0.0, "p_add_cath_cat_condition": 0.0},
    }

    def forward(self, data: dict) -> dict:
        data = super().set_task_info(data)

        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_standard_aa", "is_protein_backbone")

        # provide coordinate info for every atom that is not a standard amino acid side chain
        hidden_coordinates = atom_array.is_standard_aa & ~atom_array.is_protein_backbone
        C_CRD.set_mask(atom_array, ~hidden_coordinates)

        # provide sequence info for every atom that is not a standard amino acid
        C_SEQ.set_mask(atom_array, ~atom_array.is_standard_aa)

        data["atom_array"] = atom_array

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
            n_max_residues=n_ligands_to_sample + n_res_to_sample,
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


class TipAtomPartialLigandDistanceTask(TipAtomPartialLigandTask):
    def __init__(
        self,
        *,
        p_knockout_dist: float = 0.95,
        knockout_dist_min_fraction: float = 0.1,
        knockout_dist_max_fraction: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p_knockout_dist = p_knockout_dist
        self.knockout_dist_min_fraction = knockout_dist_min_fraction
        self.knockout_dist_max_fraction = knockout_dist_max_fraction

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_protein_backbone", "is_nucleic_acid_backbone")

        total_mask, all_masks = self.sample_mask(atom_array)

        n_atoms = atom_array.array_length()
        _true = np.ones(n_atoms, dtype=bool)

        atom_array = sample_distance_condition(
            atom_array,
            can_have_condition=total_mask,
            knockout_p=self.p_knockout_dist,
            dropout_min_fraction=self.knockout_dist_min_fraction,
            dropout_max_fraction=self.knockout_dist_max_fraction,
        )
        total_mask = C_DIS.mask(atom_array).as_dense_array(default=False).any(axis=0)

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
        if true_with_prob(self.p_unindex_all):
            show_idx[total_mask] = False
        elif true_with_prob(self.p_unindex_any):
            show_idx[total_mask] = np.random.rand(total_mask.sum()) > self.p_unindex_any
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


class RandomAtomDistanceTask(RFD3DesignTask):
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
        min_atoms: int = 2,
        max_atoms: int = 35,
        p_seq_conditioned: float = 1.0,
        p_knockout_seq: float = 0.0,
        p_knockout_dist: float = 0.8,
        knockout_dist_min_fraction: float = 0.1,
        knockout_dist_max_fraction: float = 0.9,
        potential: Callable[[np.ndarray], np.ndarray] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.p_seq_conditioned = p_seq_conditioned
        self.p_knockout_seq = p_knockout_seq
        self.p_knockout_dist = p_knockout_dist
        self.knockout_dist_min_fraction = knockout_dist_min_fraction
        self.knockout_dist_max_fraction = knockout_dist_max_fraction
        self.potential = potential

        if p_seq_conditioned < 1.0 or p_knockout_seq > 0.0:
            raise ValueError(
                "p_seq_conditioned < 1.0 or p_knockout_seq > 0.0 not currently supported. "
                "Model must see sequence conditioning if given a distance or coordinate condition"
            )

    def can_apply(self, data: dict) -> bool:
        ensure_annotations(
            data["atom_array"],
            "token_min_occupancy",
        )
        return data["atom_array"].mask("token_min_occupancy > 0").sum() > self.min_atoms

    def sample_mask(self, atom_array: AtomArray) -> np.ndarray:
        n_eligible_atoms = atom_array.mask("token_min_occupancy > 0").sum()
        n_max_atoms = min(n_eligible_atoms, self.max_atoms)
        n_atoms_to_sample = np.random.randint(self.min_atoms, n_max_atoms + 1)

        if self.potential is None:
            fn_sample_seed = mg.SampleUniformly(
                is_eligible="token_min_occupancy > 0",
            )
        else:
            fn_sample_seed = mg.SampleWithPotential(
                is_eligible="token_min_occupancy > 0",
                potential=self.potential,
            )

        total_mask, all_masks = mg.sample_mask_via_seed_grow_merge(
            atom_array,
            fn_sample_seed=fn_sample_seed,
            fn_check_budget=mg.CheckAtomBudget(
                n_min_atoms=n_atoms_to_sample,
                n_max_atoms=n_atoms_to_sample,
            ),
        )
        return total_mask, all_masks

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(atom_array, "is_standard_aa", "is_protein_backbone", "is_nucleic_acid_backbone")

        # ... sample mask
        total_mask, all_masks = self.sample_mask(atom_array)

        # ... annotate with the help of the mask
        n_atoms = atom_array.array_length()
        _true = np.ones(n_atoms, dtype=bool)
        _false = np.zeros(n_atoms, dtype=bool)

        # ... distance condition
        # (NOTE: This sets `C_DIS.set_annotation`)
        atom_array = sample_distance_condition(
            atom_array,
            can_have_condition=total_mask,
            knockout_p=self.p_knockout_dist,
            dropout_min_fraction=self.knockout_dist_min_fraction,
            dropout_max_fraction=self.knockout_dist_max_fraction,
        )
        # ... filter down the total mask to only include the atoms that
        #  have a distance condition
        total_mask = C_DIS.mask(atom_array).as_dense_array(default=False).any(axis=0)

        # ... sequence condition
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

        # ... index condition (TODO: Allow 'unindexed' condition)
        C_IDX.set_mask(atom_array, _true, 1, "atom")

        # ... update the now condition-annotated atom array
        data["atom_array"] = atom_array

        # ... add the task mask
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            # required fields
            "class": self.__class__.__name__,
            "mask": total_mask,
            # optional fields
            "all_masks": all_masks,
        }

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


class TipAtomDistanceTask(TipAtomTask):
    def __init__(
        self,
        knockout_p: float = 0.8,
        dropout_min_fraction: float = 0.1,
        dropout_max_fraction: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.knockout_p = knockout_p
        self.dropout_min_fraction = dropout_min_fraction
        self.dropout_max_fraction = dropout_max_fraction

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        total_mask, all_masks, target_n_residues = self.sample_mask(atom_array)

        n_atoms = atom_array.array_length()
        _true = np.ones(n_atoms, dtype=bool)

        atom_array = sample_distance_condition(
            atom_array,
            can_have_condition=total_mask,
            knockout_p=self.knockout_p,
            dropout_min_fraction=self.dropout_min_fraction,
            dropout_max_fraction=self.dropout_max_fraction,
        )
        total_mask = C_DIS.mask(atom_array).as_dense_array(default=False).any(axis=0)

        res_segments = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        is_seq_conditioned = apply_and_spread(res_segments, total_mask, np.any)
        is_seq_conditioned |= atom_array.mask("~is_standard_aa")
        C_SEQ.set_mask(atom_array, is_seq_conditioned)

        # Sample the regions to be unindexed
        show_idx = _true
        if true_with_prob(self.p_unindex_all):
            show_idx[total_mask & atom_array.is_protein & ~atom_array.atomize] = False
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
            from proteinfoundation.utils.constants import SIDECHAIN_TIP_ATOMS

            res_name = atom_array.res_name[available_atoms[0]]
            tip_atom_names = SIDECHAIN_TIP_ATOMS.get(res_name, [])

            tip_mask = np.zeros(len(atom_array), dtype=bool)
            for atom_name in tip_atom_names:
                tip_mask |= residue_mask & (atom_array.atom_name == atom_name)

            return tip_mask if tip_mask.any() else (residue_mask & (atom_array.atom_name == "CA"))

        elif self.atom_selection_mode == "ame":
            from proteinfoundation.utils.constants import AME_ATOMS

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


class MotifScaffold2DDistTask(MotifScaffoldTask):
    def __init__(
        self,
        inter_motif_distance_p: float = 0.2,
        knockout_p: float = 0.8,
        dropout_min_fraction: float = 0.1,
        dropout_max_fraction: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inter_motif_distance_p = inter_motif_distance_p
        self.knockout_p = knockout_p
        self.dropout_min_fraction = dropout_min_fraction
        self.dropout_max_fraction = dropout_max_fraction

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        total_mask, all_masks = self.sample_mask(atom_array)

        if true_with_prob(self.inter_motif_distance_p):
            atom_array = sample_distance_condition(
                atom_array,
                can_have_condition=total_mask,
                knockout_p=self.knockout_p,
                dropout_min_fraction=self.dropout_min_fraction,
                dropout_max_fraction=self.dropout_max_fraction,
                additive=False,
            )
            added_inter_motif_distances = True
        else:
            total_mask_so_far = np.zeros(atom_array.array_length(), dtype=bool)
            for motif_mask in all_masks:
                motif_mask = motif_mask & ~total_mask_so_far
                total_mask_so_far |= motif_mask
                atom_array = sample_distance_condition(
                    atom_array,
                    can_have_condition=motif_mask,
                    knockout_p=self.knockout_p,
                    dropout_min_fraction=self.dropout_min_fraction,
                    dropout_max_fraction=self.dropout_max_fraction,
                    additive=True,
                )
            added_inter_motif_distances = False

        total_mask = C_DIS.mask(atom_array).as_dense_array(default=False).any(axis=0)

        # Don't do coordinate conditioning when we use 2D distance conditioning
        if C_CRD.full_name in atom_array.get_annotation_categories():
            atom_array.del_annotation(C_CRD.full_name)
        if C_CRD.mask_name in atom_array.get_annotation_categories():
            atom_array.del_annotation(C_CRD.mask_name)
        C_CRD.set_annotation(atom_array, C_CRD.default_annotation(atom_array))
        C_CRD.set_mask(atom_array, C_CRD.default_mask(atom_array))

        res_segments = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        is_seq_conditioned = apply_and_spread(res_segments, total_mask, np.any)
        C_SEQ.set_mask(atom_array, is_seq_conditioned)

        C_DIS.set_mask(atom_array, total_mask, n_body=1)

        # Sample the regions to be unindexed using an island mask
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

        data["atom_array"] = atom_array

        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": total_mask,
            "coverage_fraction_goal": self.coverage_fraction_goal,
            "all_masks": all_masks,
            "added_inter_motif_distances": added_inter_motif_distances,
        }

        data = self.apply_design_task_modifiers(data)
        data = self.ensure_atomized_tokens_are_sequence_conditioned(data)

        return data


class PPIHotspotWith3DTarget(RFD3DesignTask):
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
        hotspot_count_distribution: Literal["uniform", "geometric"] = "uniform",
        hotspot_geometric_p: float | None = None,
        max_hotspots_frac_to_provide: float = 0.1,
        hotspot_max_distance: float = 7.0,
        p_no_hotspots: float = 0.05,
        p_sample_ppi_crop: float = 0.75,
        ppi_crop_binder_fraction: float = 0.667,
        bb_only_coord_conditioning: bool = False,
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
            bb_only_coord_conditioning: If True, only the backbone atoms of the target chain
                receive coordinate conditioning. NOTE: Sidechain atoms can still be hotspots, but
                their precise coordinates are not given.

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
        self.bb_only_coord_conditioning = bb_only_coord_conditioning

    def annotate_for_task_selection(self, data: dict) -> dict:
        if "crop_info" in data:
            if "atom_array" not in data["crop_info"]:
                raise ValueError("Must preserve uncropped AtomArray in crop_info before applying this design task.")
            atom_array = data["crop_info"]["atom_array"]
        else:
            atom_array = data["atom_array"]

        ensure_annotations(atom_array, "is_protein")
        return data

    def can_apply(self, data: dict) -> bool:
        """
        Can be applied if there are exactly two different chains in query_pn_unit_iids that are both protein chains.
        Based on Rafi's code in PPICondition.is_valid_for_example() here: https://github.dev/private-organization/modelhub/blob/aa_design/ppi-v2025_06_24/projects/aa_design/transforms/ppi_transforms.py
        """
        query_pn_unit_iids = data.get("query_pn_unit_iids")
        if "crop_info" in data and "atom_array" in data["crop_info"]:
            atom_array = data["crop_info"]["atom_array"]
        else:
            atom_array = data["atom_array"]
        if query_pn_unit_iids is None or len(query_pn_unit_iids) != 2 or len(np.unique(query_pn_unit_iids)) != 2:
            return False

        for pn_unit_iid in query_pn_unit_iids:
            # make sure pn_unit_iid is actually in the atom array
            if atom_array.mask(f"pn_unit_iid == '{pn_unit_iid}'").sum() == 0:
                return False
            if not atom_array[atom_array.mask(f"pn_unit_iid == '{pn_unit_iid}'")].mask("is_protein").all():
                return False

        return True

    def forward(self, data: dict) -> dict:
        query_pn_unit_iids = data["query_pn_unit_iids"]  # existence guaranteed via can_apply()

        # randomly select one of the chains as the binder chain
        binder_pn_unit_iid, target_pn_unit_iid = np.random.permutation(query_pn_unit_iids)

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

                # Overwrite the old crop info with the info for the newly-applied PPI crop
                data["crop_info"] = {"type": "PPIFullBinderCropSpatial"} | crop_info
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
        is_target_chain = atom_array.mask(f"pn_unit_iid == '{target_pn_unit_iid}'")
        nan_coord_mask = atom_array.mask("has_nan_coord()")

        C_HOT.set_annotation(atom_array=atom_array, array=hotspot_atom_mask)
        C_HOT.set_mask(atom_array, hotspot_atom_mask, 1, "atom")

        if self.bb_only_coord_conditioning:
            ensure_annotations(atom_array, "is_protein_backbone")
            C_CRD.set_mask(atom_array, is_target_chain & ~nan_coord_mask & atom_array.is_protein_backbone)
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


class AntibodyFrameworkConditioningTask3DTarget(RFD3DesignTask):
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

    """
    Antibody-specific design task that conditions on framework regions and the target while leaving
    CDRs free for design.

    This task:
    1. Maps CDR regions from sequence positions to atom positions
    2. Applies 2D distance conditioning to framework regions only
    3. Provides sequence conditioning to framework regions and target chains
    4. Leaves CDR regions completely unconditioned for free design
    5. Specifies hotspots on the target chain

    Args:
        hotspot_geometric_p: geometric distribution parameter controlling the number of hotspot
            residues sampled (lower values → more residues on average)
        max_hotspots_frac_to_provide: maximum fraction of interface residues to sample as hotspots
        hotspot_max_distance: maximum distance (Å) between an antibody atom and a target atom for
            a residue to be eligible as a hotspot
        p_no_hotspots: probability of providing no hotspots at all (empty hotspot mask)
        p_flexible_target: probability of providing the target chain as a 2D distance motif
        motif_distance_knockout_p: probability of doing dropout of 2D distance condition at all
            (otherwise it is a fully connected graph of participating atoms)
        min_dropout_2d_dist: minimum fraction of distances to keep after dropout during 2D
            distance conditioning
        max_dropout_2d_dist: maximum fraction of distances to keep after dropout during 2D
            distance conditioning
    """

    def __init__(
        self,
        hotspot_count_distribution: Literal["uniform", "geometric"] = "uniform",
        hotspot_geometric_p: float | None = None,
        max_hotspots_frac_to_provide: float = 0.1,
        hotspot_max_distance: float = 7.0,
        p_no_hotspots: float = 0.05,
        p_flexible_target: float = 0.1,
        motif_distance_knockout_p: float = 0.8,
        min_dropout_2d_dist: float = 0.1,
        max_dropout_2d_dist: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hotspot_count_distribution = hotspot_count_distribution
        self.hotspot_geometric_p = hotspot_geometric_p
        self.max_hotspots_frac_to_provide = max_hotspots_frac_to_provide
        self.hotspot_max_distance = hotspot_max_distance
        self.p_no_hotspots = p_no_hotspots
        self.motif_distance_knockout_p = motif_distance_knockout_p
        self.min_dropout_2d_dist = min_dropout_2d_dist
        self.max_dropout_2d_dist = max_dropout_2d_dist

    def _parse_one_chain(
        self,
        atom_array: AtomArray,
        chain_id: str,
        chain_type: Literal["L", "H"],
        seq_start_index: int,
        seq_end_index: int,
        cdr_1_len: int,
        cdr_2_len: int,
        cdr_3_len: int,
        cdr_1_start_pos: int,
        cdr_2_start_pos: int,
        cdr_3_start_pos: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse one heavy or light chain and return CDR and chain type masks.
        Args:
            atom_array: AtomArray or AtomArrayPlus
            chain_id: Chain ID
            chain_type: Chain type (L or H)
            seq_start_index: Sequence start index
            seq_end_index: Sequence end index
            cdr_1_len: CDR 1 length
            cdr_2_len: CDR 2 length
            cdr_3_len: CDR 3 length
            cdr_1_start_pos: CDR 1 start position
            cdr_2_start_pos: CDR 2 start position
            cdr_3_start_pos: CDR 3 start position
        Returns:
            - cdr_mask: mask of the CDR regions (of length atom_array.array_length())
            - heavy_chain_mask: mask of heavy chain (if scFv, this will be a part of the chain)
            - light_chain_mask: mask of light chain
        """

        cdr_mask = np.zeros(atom_array.array_length(), dtype=bool)
        heavy_chain_mask = np.zeros(atom_array.array_length(), dtype=bool)
        light_chain_mask = np.zeros(atom_array.array_length(), dtype=bool)
        chain_mask = atom_array.mask(f"chain_id == '{chain_id}'")

        cdr_inds = [
            seq_start_index + i + cdr_start_pos
            for cdr_len, cdr_start_pos in zip(
                [cdr_1_len, cdr_2_len, cdr_3_len],
                [cdr_1_start_pos, cdr_2_start_pos, cdr_3_start_pos],
                strict=False,
            )
            for i in range(cdr_len)
        ]

        cdr_mask = np.isin(atom_array.res_id, cdr_inds) & chain_mask
        heavy_chain_mask = (
            np.isin(atom_array.res_id, list(range(seq_start_index, seq_end_index + 1)))
            & chain_mask
            & (chain_type == "H")
        )
        light_chain_mask = (
            np.isin(atom_array.res_id, list(range(seq_start_index, seq_end_index + 1)))
            & chain_mask
            & (chain_type == "L")
        )

        return cdr_mask, heavy_chain_mask, light_chain_mask

    def _get_cdr_and_chain_mask_from_dict(self, atom_array: AtomArray, all_ig_chains: dict) -> np.ndarray:
        """
        Parses an input dictionary from 'extra_info/all_ig_chains' to get the CDR mask and chain type masks.
        Input dict needs to have keys:
            - 'seqstart_index'
            - 'seqend_index'
            - 'vdj_cdrX_start_pos'  <- 'X' can be 1, 2, or 3. All must be present.
            - 'vdj_cdrX'
            - 'vj_cdrX_start_pos'
            - 'vj_cdrX'
            - 'chain_type'  <- 'K', 'H', 'L', or a combo in the case of scFvs ('L-H')
        Returns:
            - cdr_mask: mask of the CDR regions (of length atom_array.array_length())
            - heavy_chain_mask: mask of heavy chain (if scFv, this will be a part of the chain)
            - light_chain_mask: mask of light chain
        """

        cdr_mask = np.zeros(atom_array.array_length(), dtype=bool)
        heavy_chain_mask = np.zeros(atom_array.array_length(), dtype=bool)
        light_chain_mask = np.zeros(atom_array.array_length(), dtype=bool)

        for chain_id, cdr_info in all_ig_chains.items():
            if "-" in cdr_info["chain_type"]:
                chain_types = cdr_info["chain_type"].split("-")
                seq_start_positions = [int(i) for i in cdr_info["seqstart_index"].split("-")]
                seq_end_positions = [int(i) for i in cdr_info["seqend_index"].split("-")]
            else:
                chain_types = [cdr_info["chain_type"]]
                seq_start_positions = [int(cdr_info["seqstart_index"])]
                seq_end_positions = [int(cdr_info["seqend_index"])]

            for i in range(len(chain_types)):
                chain_type = "L" if chain_types[i] == "K" else chain_types[i]
                gene_name = "vdj" if chain_type == "H" else "vj"
                _cdr_mask, _heavy_chain_mask, _light_chain_mask = self._parse_one_chain(
                    atom_array,
                    chain_id,
                    chain_type,
                    seq_start_positions[i],
                    seq_end_positions[i],
                    len(cdr_info[f"{gene_name}_cdr1"]),
                    len(cdr_info[f"{gene_name}_cdr2"]),
                    len(cdr_info[f"{gene_name}_cdr3"]),
                    int(cdr_info[f"{gene_name}_cdr1_start_pos"]),
                    int(cdr_info[f"{gene_name}_cdr2_start_pos"]),
                    int(cdr_info[f"{gene_name}_cdr3_start_pos"]),
                )
                cdr_mask |= _cdr_mask
                heavy_chain_mask |= _heavy_chain_mask
                light_chain_mask |= _light_chain_mask

        return cdr_mask, heavy_chain_mask, light_chain_mask

    def annotate_for_task_selection(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        ensure_annotations(
            atom_array,
            "is_protein",
        )
        cdr_mask, heavy_chain_mask, light_chain_mask = self._get_cdr_and_chain_mask_from_dict(
            atom_array,
            eval(data["extra_info"]["all_ig_chains"].replace("null", "None")),
        )
        C_CDR.set_mask(atom_array, cdr_mask)
        C_ABH.set_mask(atom_array, heavy_chain_mask)
        C_ABL.set_mask(atom_array, light_chain_mask)

        data["atom_array"] = atom_array
        return data

    def can_apply(self, data: dict) -> bool:
        """
        Can be applied if:
            - sum of CDR mask is greater than 0
            - there is a heavy or light chain annotated
            - the heavy and/or light chain have cdr atoms annotated
            - there is at least one chain (target chain) that is not annotated with CDRs or heavy/light chains
        """
        atom_array = data["atom_array"]
        cdr_mask = C_CDR.mask(atom_array)
        heavy_chain_mask = C_ABH.mask(atom_array)
        light_chain_mask = C_ABL.mask(atom_array)

        ig_chains = np.unique(atom_array.chain_iid[cdr_mask | heavy_chain_mask | light_chain_mask])
        target_chains = list(set(atom_array.chain_iid) - set(ig_chains))

        if cdr_mask.sum() == 0:
            return False

        if heavy_chain_mask.sum() == 0 and light_chain_mask.sum() == 0:
            return False

        # heavy/light chains must have cdrs
        if heavy_chain_mask.sum() > 0 and cdr_mask[heavy_chain_mask].sum() == 0:
            return False
        if light_chain_mask.sum() > 0 and cdr_mask[light_chain_mask].sum() == 0:
            return False

        # CDRs must only be on heavy or light chains
        if cdr_mask[~heavy_chain_mask & ~light_chain_mask].sum() > 0:
            return False

        # there must be at least one chain that is not annotated with CDRs or heavy/light chains
        return len(target_chains) != 0

    def forward(self, data: dict) -> dict:
        """
        Logic:
            1. Sample hotspots
            2. Apply 2D distance conditioning to framework
            3. Maybe apply 2D distance conditioning to target chain
            4. Provide sequence conditioning to framework regions and target chains
            5. Set hotspot conditioning
        """
        atom_array = data["atom_array"]
        cdr_mask = C_CDR.mask(atom_array)

        ab_chains = np.unique(atom_array[cdr_mask].chain_iid)
        target_chains = [chain for chain in np.unique(atom_array.chain_iid) if chain not in ab_chains]

        target_chain_mask = np.isin(atom_array.chain_iid, target_chains)
        framework_mask = ~cdr_mask & np.isin(atom_array.chain_iid, ab_chains)

        # ... sample hotspots
        if true_with_prob(self.p_no_hotspots):
            hotspot_mask = np.zeros(atom_array.array_length(), dtype=bool)
        else:
            hotspot_mask = sample_hotspots(
                atom_array,
                ab_chains,
                hotspot_max_distance=self.hotspot_max_distance,
                max_hotspots_frac_to_provide=self.max_hotspots_frac_to_provide,
                hotspot_count_distribution=self.hotspot_count_distribution,
                hotspot_geometric_p=self.hotspot_geometric_p,
            )
        C_HOT.set_annotation(atom_array=atom_array, array=hotspot_mask)
        C_HOT.set_mask(atom_array, target_chain_mask)

        # ... apply absolute coordinate conditioning to target chain
        C_CRD.set_mask(atom_array, target_chain_mask)

        # ... apply internal coordinate conditioning to framework regions
        atom_array = convert_to_internal_coords(atom_array, framework_mask)

        # ... provide sequence conditioning to framework regions and target chains
        C_SEQ.set_mask(atom_array, target_chain_mask | framework_mask)
        C_IDX.set_mask(atom_array, np.ones(atom_array.array_length(), dtype=bool))  # everything is indexed

        # ... update data dict
        data["atom_array"] = atom_array
        if "task" not in data:
            data["task"] = {"name": self.__class__.__name__}
        data["task"] |= {
            "class": self.__class__.__name__,
            "mask": hotspot_mask,
            "binder_chain_ids": ab_chains,
            "target_chain_ids": target_chains,
            "num_hotspots_to_sample": hotspot_mask.sum(),
        }

        data = self.apply_design_task_modifiers(data)

        return data


class AntibodyFrameworkConditioningTask(AntibodyFrameworkConditioningTask3DTarget):
    def __init__(self, *args, p_flexible_target: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_flexible_target = p_flexible_target

    def forward(self, data: dict) -> dict:
        data = super().forward(data)

        atom_array = data["atom_array"]
        target_chains = data["task"]["target_chain_ids"]
        target_chain_mask = np.isin(atom_array.chain_iid, target_chains)

        if true_with_prob(self.p_flexible_target):
            # flexible target, no coordinate conditioning at all
            new_coord_cond_mask = C_CRD.mask(atom_array, default="generate") & ~target_chain_mask
            C_CRD.set_mask(atom_array, new_coord_cond_mask)
            coord_group_annot = C_CRDGRP.annotation(atom_array, default="generate")
            coord_group_annot[target_chain_mask] = C_CRDGRP.UNKNOWN_COORDS
            C_CRDGRP.set_annotation(atom_array, coord_group_annot)
        else:
            # rigid target with internal coordinate conditioning
            atom_array = convert_to_internal_coords(atom_array, target_chain_mask)

        data["atom_array"] = atom_array
        data["task"]["name"] = self.__class__.__name__
        data["task"]["class"] = self.__class__.__name__
        return data


class PPIHotspotWithInternalCoordTargetMotifTask(PPIHotspotWith3DTarget):
    def __init__(
        self,
        *args,
        p_flexible_target: float = 0.9,
        motif_distance_knockout_p: float = 0.8,
        min_dropout_2d_dist: float = 0.1,
        max_dropout_2d_dist: float = 0.9,
        **kwargs,
    ):
        """
        Same as PPIHotspotWith3DTarget, but with internal instead of absolute conditioning for
        the target chain.
        Args:
            p_flexible_target: Probability of making the target completely flexible (no coordinate
                conditioning given)
            motif_distance_knockout_p: probability of doing dropout of 2D distance condition at all
                (otherwise it is a fully connected graph of participating atoms)
            min_dropout_2d_dist: minimum fraction of distances to keep after dropout during 2D
                distance conditioning
            max_dropout_2d_dist: maximum fraction of distances to keep after dropout during 2D
                distance conditioning
        """
        super().__init__(*args, **kwargs)
        self.p_flexible_target = p_flexible_target
        self.motif_distance_knockout_p = motif_distance_knockout_p
        self.min_dropout_2d_dist = min_dropout_2d_dist
        self.max_dropout_2d_dist = max_dropout_2d_dist

    def forward(self, data: dict) -> dict:
        data = super().forward(data)
        atom_array = data["atom_array"]
        target_mask = C_CRD.mask(atom_array, default="generate")

        if true_with_prob(self.p_flexible_target):
            # flexible target, no coordinate conditioning at all
            new_coord_cond_mask = C_CRD.mask(atom_array, default="generate") & ~target_mask
            C_CRD.set_mask(atom_array, new_coord_cond_mask)
            coord_group_annot = C_CRDGRP.annotation(atom_array, default="generate")
            coord_group_annot[target_mask] = C_CRDGRP.UNKNOWN_COORDS
            C_CRDGRP.set_annotation(atom_array, coord_group_annot)
        else:
            # rigid target with internal coordinate conditioning
            atom_array = convert_to_internal_coords(atom_array, target_mask)

        data["atom_array"] = atom_array

        data["task"]["name"] = self.__class__.__name__
        data["task"]["class"] = self.__class__.__name__
        return data
