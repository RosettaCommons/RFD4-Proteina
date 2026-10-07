"""Exhaustive matching of identical molecular subunits before atom symmetry."""

from collections import defaultdict

import numpy as np
from atomworks.io.transforms.atom_array import add_id_and_entity_annotations, ensure_atom_array_stack
from atomworks.io.utils.scatter import get_segments
from atomworks.ml.transforms.symmetry import get_isomorphisms_from_symmetry_groups, instance_to_token_lvl_isomorphisms
from biotite.structure import (
    AtomArray,
    AtomArrayStack,
    BondList,
    filter_amino_acids,
    filter_nucleotides,
    superimpose,
)

from rfproteina.utils.atom_permutation import _observed_mask


def _subunit_candidates(array: AtomArray, classes: np.ndarray, limit: int) -> np.ndarray:
    """Enumerate entity-equivalent swaps with identical chain/residue boundaries and ordered atoms.

    Returns destination-to-source atom maps, with identity first. Wholly inactive
    molecules stay fixed; incompatible chain/residue boundaries or atom order raises ValueError.
    """
    identity = np.arange(len(array))
    if not len(array):
        return identity[None]

    # Identify physical chain and residue boundaries, including assembly copies.
    categories = array.get_annotation_categories()
    chains = array.get_annotation("chain_iid" if "chain_iid" in categories else "chain_id")
    if "chain_iid" not in categories and "transformation_id" in categories:
        _, chains = np.unique(np.column_stack((chains, array.transformation_id)), axis=0, return_inverse=True)
    starts = get_segments(chains, array.res_id, add_exclusive_stop=True)
    residues = np.repeat(np.arange(len(starts) - 1), np.diff(starts))

    # Reuse molecule annotations; assign missing ones on a private copy.
    annotated = array
    if not {"molecule_iid", "molecule_entity"}.issubset(categories):
        annotated = array.copy()
        polymer = (
            array.is_polymer if "is_polymer" in categories else filter_amino_acids(array) | filter_nucleotides(array)
        )
        annotated.set_annotation("is_polymer", polymer)
        _, chain_ids = np.unique(chains, return_inverse=True)
        annotated.chain_id = np.where(polymer, chain_ids, chain_ids.max() + 1 + residues).astype(str)
        if annotated.bonds is None:
            annotated.bonds = BondList(array.array_length())
        add_id_and_entity_annotations(annotated)
    units = array.molecule_iid if "molecule_iid" in categories else annotated.molecule_id
    entities = array.molecule_entity if "molecule_entity" in categories else annotated.molecule_entity

    # Validate existing order without sorting or renumbering equivalent molecules.
    signatures = {}
    groups = defaultdict(list)
    instance_atoms = []
    for unit in dict.fromkeys(units):
        indices = np.flatnonzero(units == unit)
        if not classes[indices].any():
            continue
        chain_starts = get_segments(chains[indices])
        residue_starts = get_segments(residues[indices])
        fields = (array.res_name, array.atom_name, array.element)
        signature = (chain_starts.tobytes(), residue_starts.tobytes(), *(f[indices].tobytes() for f in fields))
        entity = entities[indices[0]]
        if signatures.setdefault(entity, signature) != signature:
            raise ValueError(
                f"Molecule entity {entity} requires identical chain boundaries and ordered atoms "
                "with matching residue boundaries; use resolve_symmetry=[atom] for incompatible atom order or coverage."
            )
        groups[(entity, classes[indices].tobytes())].append(len(instance_atoms))
        instance_atoms.append(indices)

    # Bound the Cartesian product of within-group permutations before allocating it.
    count = 1
    for group in groups.values():
        for factor in range(2, len(group) + 1):
            count *= factor
            if count > limit:
                raise NotImplementedError(
                    f"Subunit symmetry above {limit} permutations is not supported; use resolve_symmetry=[atom]."
                )
    if count == 1:
        return identity[None]

    # Expand whole-molecule assignments into destination-to-source atom index maps.
    assignments = get_isomorphisms_from_symmetry_groups(dict(enumerate(groups.values())), max_isomorphisms=limit)
    ordered_atoms = [instance_atoms[i] for i in assignments[0]]
    candidates = np.tile(identity, (count, 1))
    candidates[:, np.concatenate(ordered_atoms)] = instance_to_token_lvl_isomorphisms(assignments, ordered_atoms)
    return candidates


def permute_gt_subunits(
    predicted: AtomArray | AtomArrayStack,
    ground_truth: AtomArray | AtomArrayStack,
    *,
    frame_mask: np.ndarray | None = None,
    score_mask: np.ndarray | None = None,
    align: bool = True,
    max_permutations: int = 720,
) -> AtomArrayStack:
    """Match ground-truth molecule copies to each prediction by minimum coordinate error.

    Assignment is greedy: current atom labels determine the subunit match, before
    any subsequent atom-symmetry resolution; joint optimality is not guaranteed.
    Equivalent active molecules must have identical chain/residue boundaries and ordered
    atoms. Absolute residue numbers may differ between copies.

    Args:
        predicted: One structure or a stack, in atom correspondence with ground_truth.
        ground_truth: Reference structure(s); one model may broadcast to prediction depth.
        frame_mask: Boolean (n_atoms,) mask used for rigid alignment; defaults to all atoms.
        score_mask: Boolean (n_atoms,) mask used to score assignments; defaults to all atoms.
        align: Fit the prediction on frame_mask before scoring; otherwise compare coordinates directly.
        max_permutations: Search limit; larger searches raise NotImplementedError before scoring.

    Returns:
        A copied AtomArrayStack with coordinates shaped (prediction_depth, n_atoms, 3).
        Each model's GT coordinates are reordered by its best subunit assignment.
        Coordinates remain in the original GT frame: alignment is only used for scoring.
        Annotations and bonds retain their original atom order; neither input is mutated.
    """
    # Validate stack compatibility and normalize the alignment/scoring selections.
    if not isinstance(max_permutations, int) or max_permutations < 1:
        raise ValueError("max_subunit_permutations must be a positive integer")
    pred, gt = ensure_atom_array_stack(predicted), ensure_atom_array_stack(ground_truth)
    n = gt.array_length()
    if pred.array_length() != n or gt.stack_depth() not in (1, pred.stack_depth()):
        raise ValueError("Subunit matching requires equal atom counts and GT depth one or matching prediction depth")
    frame, score = [
        np.ones(n, dtype=bool) if m is None else np.asarray(m, dtype=bool) for m in (frame_mask, score_mask)
    ]
    if frame.shape != (n,) or score.shape != (n,):
        raise ValueError(f"Subunit masks must have shape ({n},)")

    # Encode both mask memberships as four classes so swaps preserve each selection.
    classes = (frame.astype(int) if align else 0) + 2 * score
    candidates = _subunit_candidates(gt[0], classes, max_permutations)

    # Prepare one independent output model per prediction, broadcasting GT if needed.
    resolved = gt.copy()
    if gt.stack_depth() != pred.stack_depth():
        resolved = resolved[np.zeros(pred.stack_depth(), dtype=int)]
    if len(candidates) == 1:
        return resolved

    # Choose an assignment separately for each model, using only observed coordinates.
    for model, valid in enumerate(_observed_mask(resolved)):
        xp, xg = pred.coord[model], resolved.coord[model]
        observed = valid & np.isfinite(xp).all(axis=-1)
        fit, scored = frame & observed, score & observed
        if not scored.any() or (align and not fit.any()):
            continue
        best, best_error = candidates[0], float("inf")
        # Score bounded batches; reject swaps that change which GT atoms are observed.
        for start in range(0, len(candidates), 32):
            chunk = candidates[start : start + 32]
            chunk = chunk[(valid[chunk] == valid).all(axis=1)]
            if not len(chunk):
                continue
            target, fitted = xg[chunk], xp[scored]
            if align:
                # Fit each candidate on frame atoms, then transform only the scored atoms.
                mobile = np.broadcast_to(xp, target.shape)
                _, transform = superimpose(target[:, fit], mobile[:, fit])
                fitted = transform.apply(mobile[:, scored])

            # Minimize mean squared distance on score atoms; keep the first assignment on ties.
            errors = np.square(fitted - target[:, scored]).sum(axis=(1, 2), dtype=float) / scored.sum()
            index = errors.argmin()
            if errors[index] < best_error:
                best, best_error = chunk[index], errors[index]

        # Apply only the chosen permutation, not the scoring alignment.
        resolved.coord[model] = resolved.coord[model][best]
    return resolved
