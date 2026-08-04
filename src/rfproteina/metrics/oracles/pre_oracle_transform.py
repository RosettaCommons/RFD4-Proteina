"""Pre-oracle AtomArray transforms applied before structure prediction.

A transform rewrites the generated AtomArray just before it is folded, and may
fan one design out into several outputs. Each output carries a transform-chosen
``save_id``; :py:class:`~rfproteina.metrics.oracles.oracle_metric.OracleMetric`
folds each one separately, saves it into the design's own output directory under
a filename carrying that save_id, and prefixes its metrics with it — so refolds
that differ only by the transform stay directly comparable.

Subclasses that load heavy resources override :py:meth:`setup` /
:py:meth:`teardown`; the oracle calls these once per validation epoch.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.transforms.atom_array import (
    add_chain_iid_annotation,
    add_id_and_entity_annotations,
    add_molecule_iid_annotation,
    add_pn_unit_iid_annotation,
)
from atomworks.io.utils.annotator import ensure_annotations, remove_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus, concatenate_atom_array_plus
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.io.utils.selection import get_annotation_categories
from biotite.structure import AtomArray

from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.parse_presets import ANNOTATED_CIF

logger = RankedLogger(__name__, rank_zero_only=True)

# Off-targets are loaded like any other annotated CIF, so whatever annotations
# the file carries are the ones it contributes (see
# :py:func:`_align_annotations`). The two departures are atoms the oracle cannot
# round-trip: hydrogens and waters have no counterpart in its heavy-atom output,
# and an input atom with no counterpart desyncs the transfer of the predicted
# coordinates back onto the input array.
OFF_TARGET_PARSE_CONFIG: ParseConfig = ANNOTATED_CIF.replace(
    hydrogen_policy="remove",
    remove_waters=True,
)


@dataclass(frozen=True, slots=True)
class PreOracleOutput:
    """One named output of a :py:class:`PreOracleTransform`.

    Args:
        save_id: Id for this output, unique among the outputs of one
            :py:meth:`PreOracleTransform.forward` call. It ends up in filenames,
            so it must be non-empty and free of path separators.
        atom_array: The transformed AtomArray to fold.
        chain_of_interest: Chain identifier of the designed chain *in*
            ``atom_array``. A transform knows which chain it kept; re-deriving it
            downstream would run the heuristic over a complex whose other chains
            came off disk carrying only default conditioning annotations, where it
            is ambiguous or plain wrong. ``None`` falls back to that heuristic.
    """

    save_id: str
    atom_array: AtomArray
    chain_of_interest: str | None = None


class PreOracleTransform(ABC):
    """Abstract base class for pre-oracle AtomArray transforms."""

    def setup(self) -> None:
        """Load resources once (called from OracleMetric.setup)."""
        return None

    def teardown(self) -> None:
        """Release loaded resources."""
        return None

    @abstractmethod
    def expected_save_ids(self) -> list[str]:
        """The save_ids :py:meth:`forward` emits, for every input.

        This is what the oracle looks for on disk — it is how a cache hit finds
        a design's refolds without re-running the transform — so it must match
        what ``forward`` returns.
        """

    @abstractmethod
    def forward(self, atom_array: AtomArray) -> list[PreOracleOutput]:
        """Transform *atom_array* into one or more named outputs to fold.

        Args:
            atom_array: Generated AtomArray from the model.

        Returns:
            A non-empty list of :py:class:`PreOracleOutput` whose save_ids are
            :py:meth:`expected_save_ids`. The common case is a single-element list.
        """


def _coi_mask(atom_array: AtomArray) -> tuple[np.ndarray, str]:
    """Boolean mask selecting the design's chain of interest, and its identifier.

    Returns:
        Tuple of (atom mask selecting the COI, its ``chain_iid`` when the input
        carries that annotation, else its ``chain_id``).

    Raises:
        ValueError: If the input has no chain that could be the COI.
    """
    chain_of_interest, unambiguous = get_chain_of_interest(atom_array)
    if chain_of_interest is None:
        raise ValueError("Could not determine a chain of interest for the input structure (no protein chains).")
    if not unambiguous:
        logger.warning(f"Chain of interest {chain_of_interest!r} was determined ambiguously.")
    category = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
    return atom_array.get_annotation(category) == chain_of_interest, str(chain_of_interest)


def _align_annotations(design: AtomArray, off_target: AtomArray) -> tuple[AtomArray, AtomArray]:
    """Give each array whichever of the other's annotations it can default.

    Concatenation keeps only the annotations *common* to its inputs, so without
    this the design would silently lose every conditioning annotation the
    off-target file does not have.
    """
    design, off_target = as_atom_array_plus(design.copy()), as_atom_array_plus(off_target.copy())
    for target, reference in ((off_target, design), (design, off_target)):
        missing = set(get_annotation_categories(reference, n_body="all")) - set(
            get_annotation_categories(target, n_body="all")
        )
        for name in sorted(missing):
            # No generator registered -> no defensible default; let it be dropped.
            with contextlib.suppress(ValueError):
                ensure_annotations(target, name)
    return design, off_target


def _rename_chains(atom_array: AtomArray, unavailable_chain_ids: set[str]) -> AtomArray:
    """Reassign every chain in ``atom_array`` to a chain_id not in ``unavailable_chain_ids``.

    Chains are renamed in sorted order so the mapping is deterministic, as
    :py:meth:`PreOracleTransform.forward` requires. Only ``chain_id`` is set here;
    every id derived from it is rebuilt by :py:func:`_reindex_identities` once the
    two structures are one array.
    """
    atom_array = atom_array.copy()
    chain_id_generator = create_chain_id_generator(unavailable_chain_ids=unavailable_chain_ids)

    original_chain_ids = atom_array.chain_id.copy()
    # Widen the dtype first: assigning into a fixed-width '<U1' array would
    # truncate multi-character ids like "AA" that the generator falls back to.
    new_chain_id = np.empty(atom_array.array_length(), dtype="<U8")
    for chain_id in sorted(set(original_chain_ids.tolist())):
        mask = original_chain_ids == chain_id
        new_chain_id[mask] = next(chain_id_generator)

    atom_array.set_annotation("chain_id", new_chain_id)
    return atom_array


def _reindex_identities(atom_array: AtomArray) -> AtomArray:
    """Rebuild the chain / pn_unit / molecule entity and instance ids of a combined array."""
    # Stale: written straight to the CIF if kept, and re-derived from chain_entity if not.
    remove_annotations(atom_array, "label_entity_id")
    atom_array = add_id_and_entity_annotations(atom_array)
    atom_array = add_chain_iid_annotation(atom_array)
    atom_array = add_pn_unit_iid_annotation(atom_array)
    return as_atom_array_plus(add_molecule_iid_annotation(atom_array))


class ApoOnlyTransform(PreOracleTransform):
    """Fold the design's chain of interest on its own, dropping everything else.

    Pair it with an untransformed oracle to ask whether a binder folds the same
    alone as it does in its complex: the two refolds differ only by this
    transform, so a downstream
    :py:class:`~rfproteina.metrics.gt_structure_comparison.StructureComparisonMetric`
    can compare them directly.

    Args:
        save_id: save_id for the single emitted output.
    """

    def __init__(self, save_id: str = "apo") -> None:
        self.save_id = save_id

    def expected_save_ids(self) -> list[str]:
        return [self.save_id]

    def forward(self, atom_array: AtomArray) -> list[PreOracleOutput]:
        """Subset ``atom_array`` to its chain of interest.

        Raises:
            ValueError: If no chain of interest could be determined.
        """
        coi_mask, coi = _coi_mask(atom_array)
        design_chain = as_atom_array_plus(atom_array[coi_mask])
        logger.info(f"ApoOnly: kept chain {coi!r} ({design_chain.array_length()}/{atom_array.array_length()} atoms)")
        return [PreOracleOutput(save_id=self.save_id, atom_array=design_chain, chain_of_interest=coi)]


class OffTargetSubstitutionTransform(PreOracleTransform):
    """Re-fold the designed chain against other (off-target) structures.

    A specificity assay: keep only the design's chain of interest and swap
    everything it was generated against for an off-target loaded from disk, once
    per entry in ``off_target_paths``. The on-target baseline is deliberately
    *not* produced here -- configure a second, untransformed oracle for it, and
    compare its refold against these.

    Handled edge cases:

    - **Chain-ID clashes**: each off-target's chains are renamed to ids that
      appear nowhere in the design — not merely nowhere in the kept chain — so a
      design and an off-target that both call their chain "A" stay
      distinguishable, and an off-target chain can never inherit the identity
      (or the MSA) of the partner it replaced. Every id derived from ``chain_id``
      is then rebuilt over the pair (see :py:func:`_reindex_identities`).
    - **Annotation mismatch**: an off-target parsed off disk carries none of the
      design's conditioning annotations; those that know their own default are
      filled in rather than dropped from the design (see
      :py:func:`_align_annotations`).

    The new boolean annotation ``is_loaded_target`` is ``True`` on the off-target's atoms.
    This may be useful for templating. StandardAnnotations saved in the CIF are preserved.

    Args:
        off_target_paths: ``{save_id: structure file}``. One refold per entry.
        parse_config: Parse preset for the off-target files; see
            :py:data:`OFF_TARGET_PARSE_CONFIG` for what the default drops and why.

    Examples:
        >>> transform = OffTargetSubstitutionTransform({"her2": "her2.cif", "egfr": "egfr.cif"})
        >>> transform.setup()
        >>> [output.save_id for output in transform.forward(design)]
        ['her2', 'egfr']
    """

    def __init__(
        self,
        off_target_paths: dict[str, str | Path],
        parse_config: ParseConfig = OFF_TARGET_PARSE_CONFIG,
    ) -> None:
        if not off_target_paths:
            raise ValueError("off_target_paths must contain at least one {save_id: path} entry")
        self.off_target_paths = {save_id: Path(path) for save_id, path in off_target_paths.items()}
        self.parse_config = parse_config
        self._off_targets: dict[str, AtomArray] = {}

    def expected_save_ids(self) -> list[str]:
        return list(self.off_target_paths)

    def setup(self) -> None:
        """Parse and cache every off-target structure."""
        self._off_targets = {
            save_id: parse(path, config=self.parse_config)["assemblies"]["1"][0]
            for save_id, path in self.off_target_paths.items()
        }

    def teardown(self) -> None:
        """Release the cached off-target structures."""
        self._off_targets = {}

    def forward(self, atom_array: AtomArray) -> list[PreOracleOutput]:
        """Build one substituted complex per off-target.

        Raises:
            ValueError: If no chain of interest could be determined.
        """
        if not self._off_targets:
            self.setup()

        coi_mask, coi = _coi_mask(atom_array)
        design_chain = as_atom_array_plus(atom_array[coi_mask])
        # Every chain_id of the *whole* design is off-limits, not just the kept
        # one: reusing a dropped partner's id would hand the off-target that
        # partner's identity downstream (its MSA, its per-chain metrics).
        unavailable_chain_ids = set(atom_array.chain_id.tolist())

        outputs = []
        for save_id, parsed in self._off_targets.items():
            design, off_target = _align_annotations(design_chain, parsed)
            # Handle for templating the off-target and not the design. Set on both
            # halves, since concatenation keeps only the annotations they share.
            design.set_annotation("is_loaded_target", np.zeros(design.array_length(), dtype=bool))
            off_target.set_annotation("is_loaded_target", np.ones(off_target.array_length(), dtype=bool))
            off_target = _rename_chains(off_target, unavailable_chain_ids=unavailable_chain_ids)
            substituted = _reindex_identities(concatenate_atom_array_plus([design, off_target]))
            outputs.append(PreOracleOutput(save_id=save_id, atom_array=substituted, chain_of_interest=coi))
            logger.info(
                f"OffTargetSubstitution [{save_id}]: kept chain {coi!r} ({design_chain.array_length()} atoms), "
                f"substituted {off_target.array_length()} atoms from {self.off_target_paths[save_id].name}"
            )
        return outputs
