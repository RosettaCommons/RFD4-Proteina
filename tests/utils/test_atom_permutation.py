"""Tests for ground-truth atom permutation."""

import biotite.structure as struc
import numpy as np
import pytest
import torch
from atomworks.io import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.transforms.symmetry import build_automorphism_units, find_automorphisms_with_networkx
from biotite.structure import AtomArray

from rfproteina.datapipes.collate import collate_batch
from rfproteina.datapipes.pipeline import (
    ConvertToAtom37Format,
    PrecomputeAtomPermutation,
    TokenType,
    build_atom37_slots,
)
from rfproteina.datapipes.schema.proteina import ProteinaDataSample
from rfproteina.metrics.gt_structure_comparison import StructureComparisonMetric
from rfproteina.metrics.lddt_metric import AllAtomLDDT
from rfproteina.partial_autoencoder.autoencoder import AutoEncoder
from rfproteina.utils.atom_permutation import atom_permutation, permute_gt_atom37
from rfproteina.utils.parse_presets import RCSB_WITH_METALS, STANDARD_ANNOTATIONS_ONLY
from tests.conftest import TEST_DATA_DIR

_STRUCTURE = TEST_DATA_DIR / "pdb" / "6wtf.cif"


def test_unknown_amino_acid_retains_observed_atom37_coordinates() -> None:
    atoms = as_atom_array_plus(AtomArray(9))
    atoms.res_name[:] = "UNK"
    atoms.atom_name = np.array(["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"])
    atoms.coord = np.arange(27, dtype=np.float32).reshape(9, 3)
    atoms.set_annotation("token_type", np.full(9, TokenType.PROTEIN))
    atoms.set_annotation("token_id", np.zeros(9, dtype=int))
    feats = ConvertToAtom37Format().forward({"atom_array": atoms, "feats": {"coords": atoms.coord}})["feats"]
    assert feats["atom37_mask"].sum() == 9
    np.testing.assert_array_equal(feats["coords"][0, [0, 1, 2, 4, 3, 5, 11, 26, 27]], atoms.coord)
    atoms.atom_name[-1] = "ZZ"
    with pytest.raises(ValueError, match="UNK:ZZ"):
        ConvertToAtom37Format().forward({"atom_array": atoms, "feats": {"coords": atoms.coord}})


# ...(name, pairs of selection queries whose atoms are interchangeable)
SWAPPABLE = [
    ("tyrosine ring flip", [("TYR", "CD1", "CD2"), ("TYR", "CE1", "CE2")]),
    ("aspartate carboxylate", [("ASP", "OD1", "OD2")]),
    ("glutamate carboxylate", [("GLU", "OE1", "OE2")]),
    ("arginine guanidinium", [("ARG", "NH1", "NH2")]),
    ("iron-sulfur cubane", [("SF4", "FE3", "FE4"), ("SF4", "S3", "S4")]),
]


@pytest.fixture(scope="module")
def ground_truth() -> AtomArray:
    """6wtf chain A plus its ligands, heavy atoms with resolved coordinates."""
    structure = parse(str(_STRUCTURE))["assemblies"]["1"][0]
    structure = structure[structure.element != "H"]
    return structure[np.isfinite(structure.coord).all(axis=-1)]


def swap(atom_array: AtomArray, pairs: list[tuple[str, str, str]]) -> AtomArray:
    """Return a copy with each named atom pair exchanged, per residue."""
    swapped = atom_array.copy()
    selectable = as_atom_array_plus(atom_array)
    for res_name, first, second in pairs:
        for residue_id in np.unique(atom_array.res_id[atom_array.res_name == res_name]):
            where = f"res_name == '{res_name}' and res_id == {residue_id} and atom_name == "
            a = np.flatnonzero(selectable.mask(where + f"'{first}'"))
            b = np.flatnonzero(selectable.mask(where + f"'{second}'"))
            if len(a) == 1 and len(b) == 1:
                swapped.coord[[a[0], b[0]]] = swapped.coord[[b[0], a[0]]]
    assert not np.allclose(swapped.coord, atom_array.coord), "the swap must actually move something"
    return swapped


@pytest.mark.parametrize(("name", "pairs"), SWAPPABLE, ids=[case[0] for case in SWAPPABLE])
def test_metrics_ignore_a_swap_of_indistinguishable_atoms(
    name: str, pairs: list[tuple[str, str, str]], ground_truth: AtomArray
) -> None:
    """Perfect with resolution; the last line keeps that honest by requiring a penalty without."""
    predicted = swap(ground_truth, pairs)

    assert AllAtomLDDT().compute(predicted, ground_truth)["all_atom_lddt"] == pytest.approx(1.0)
    assert StructureComparisonMetric(metric_type="rmsd").compute(ground_truth, predicted)["rmsd"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert AllAtomLDDT(resolve_symmetry=()).compute(predicted, ground_truth)["all_atom_lddt"] < 1.0


def test_a_real_difference_is_still_scored(ground_truth: AtomArray) -> None:
    """Resolution must not launder a genuine error: moving a backbone atom stays penalized."""
    predicted = ground_truth.copy()
    ca = np.flatnonzero(as_atom_array_plus(ground_truth).mask("atom_name == 'CA'"))
    predicted.coord[ca[:20]] += 1.0

    assert AllAtomLDDT().compute(predicted, ground_truth)["all_atom_lddt"] < 1.0
    assert StructureComparisonMetric(metric_type="rmsd").compute(ground_truth, predicted)["rmsd"] > 0.1


def test_metrics_do_not_swap_across_a_subset_boundary(ground_truth: AtomArray) -> None:
    """`subset_mask` makes atoms outside the subset a separate class, which blocks the swap."""
    predicted = swap(ground_truth, [("ASP", "OD1", "OD2")])

    result = StructureComparisonMetric(subset_mask="atom_name == 'OD1'", metric_type="rmsd").compute(
        ground_truth, predicted
    )

    assert result["rmsd"] > 0.0


# +------------------- real model output, checked in with its conditions -------------------+

_GENERATED = TEST_DATA_DIR / "atom_permutation"

# ...design_motif_allatom_rmsd, verbatim: the ground truth is the prediction with the motif's
# conditioning coordinates substituted in, so only one structure is needed per case
MOTIF_METRIC = {
    "align_mask": "sa_mask('SA_CRDSRC')",
    "subset_mask": "sa_mask('SA_CRDSRC')",
    "subdivide_by_annotation": "condition_CoordinateGroup_1_atom",
    "gt_coord_annotation_name": "annotation_coords-with-insertion_1_atom",
    "metric_type": "rmsd",
}


def read(path):
    """Parse keeping conditions; the default config drops the columns even when present."""
    return parse(str(path), config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]


def motif_rmsd(structure, *, permute: bool) -> float:
    """Motif RMSD with atom permutation on or off."""
    kwargs = MOTIF_METRIC if permute else {**MOTIF_METRIC, "resolve_symmetry": ()}
    result = StructureComparisonMetric(**kwargs).compute(structure, structure)
    return float(result[next(iter(result))])


@pytest.mark.parametrize(
    ("case", "improves"),
    [("internal_coords", True), ("absolute_coords", False)],
    ids=["internal-coords-improves", "absolute-coords-already-optimal"],
)
def test_permutation_on_a_conditioned_design(case: str, improves: bool) -> None:
    """Permutation recovers a swapped labelling, unless the conditioning already pins it."""
    structure = read(_GENERATED / f"{case}_generated.cif.gz")

    unpermuted = motif_rmsd(structure, permute=False)
    permuted = motif_rmsd(structure, permute=True)

    assert np.isfinite(unpermuted) and np.isfinite(permuted)
    if improves:
        assert permuted < unpermuted - 1e-6, "resolution should recover the swapped labelling"
    else:
        assert permuted == pytest.approx(unpermuted, abs=1e-9), "nothing to recover here"


# +------------------- the autoencoder path: atom37 gather / scatter -------------------+


@pytest.mark.parametrize(
    ("carried", "expect_raise"),
    [((True, True), False), ((True, False), True), ((False, False), False)],
    ids=["all-samples", "mixed-datasets", "no-samples"],
)
def test_a_batch_where_only_some_datasets_precomputed_permutations(
    carried: tuple[bool, ...], expect_raise: bool
) -> None:
    """A sample carrying nothing must fail loudly, not be scored against unpermuted truth."""
    n_atoms = 4
    samples = [
        {
            "feats": {"coords": torch.zeros(2, 37, 3), "atom37_mask": torch.zeros(2, 37, dtype=torch.bool)},
            "example_id": f"s{i}",
            "extra_info": {
                "atom_automorphisms": [np.arange(n_atoms).reshape(1, n_atoms)],
                "atom37_layout": {
                    "token_ids": np.zeros(n_atoms, dtype=np.int64),
                    "slots": np.arange(n_atoms, dtype=np.int32),
                },
            }
            if has
            else {},
        }
        for i, has in enumerate(carried)
    ]
    batch = collate_batch(samples, schema=ProteinaDataSample, fixed_dimensions={"I": 4})

    extra = batch["extra_info"] or {}
    automorphisms, layout = extra.get("atom_automorphisms"), extra.get("atom37_layout")
    shape = torch.zeros(len(carried), 2, 37, 3)
    mask = torch.zeros(len(carried), 2, 37, dtype=torch.bool)

    if expect_raise:
        with pytest.raises(ValueError, match="carry no"):
            permute_gt_atom37(shape, shape.clone(), mask, automorphisms, layout)
    else:
        assert permute_gt_atom37(shape, shape.clone(), mask, automorphisms, layout).shape == shape.shape


def as_atom37(atom_array: AtomArray) -> tuple[AtomArray, torch.Tensor, torch.Tensor]:
    """Project onto the autoencoder's atom37 layout, in nanometres.

    `parse` already supplies `pn_unit_iid` and `is_polymer`; `token_id` and `token_type` are
    pipeline products no CIF carries, so they are all that is added here.
    """
    annotated = atom_array.copy()
    n_atoms = annotated.array_length()
    polymer = bool(annotated.is_polymer[0])
    annotated.set_annotation(
        "token_id", np.zeros(n_atoms, dtype=np.int32) if polymer else np.arange(n_atoms, dtype=np.int32)
    )
    token_type = TokenType.PROTEIN if polymer else TokenType.ATOMIZED
    annotated.set_annotation("token_type", np.full(n_atoms, token_type, dtype=np.int16))

    slots = build_atom37_slots(annotated)
    valid = slots >= 0
    tokens, slot = annotated.token_id[valid], slots[valid]
    coords = torch.zeros((int(annotated.token_id.max()) + 1, 37, 3), dtype=torch.float32)
    atom_mask = torch.zeros(coords.shape[:2], dtype=torch.bool)
    coords[tokens, slot] = torch.as_tensor(annotated.coord[valid] / 10.0, dtype=torch.float32)
    atom_mask[tokens, slot] = True
    return annotated, coords, atom_mask


def precomputed_for(annotated: AtomArray) -> dict:
    """What the dataloader workers hand the loss, shaped as collation delivers it (batch of one)."""
    # ... `forward` rather than `__call__`: the ordering contract needs a pipeline history
    extra = PrecomputeAtomPermutation().forward({"atom_array": annotated})["extra_info"]
    return {
        "atom_automorphisms": [extra["atom_automorphisms"]],
        "atom37_layout": {key: [value] for key, value in extra["atom37_layout"].items()},
    }


def autoencoder_loss(
    annotated: AtomArray,
    gt: torch.Tensor,
    pred: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    atom_permutation: bool = True,  # ...off by default in the autoencoder; these tests are about it
    **kwargs,
):
    """Run the autoencoder's structural reconstruction loss over one sample."""
    all_tokens = torch.ones((1, pred.shape[0]), dtype=torch.bool)
    return AutoEncoder.compute_struct_rec_loss(
        output_dec={"coors_nm": pred.unsqueeze(0), "emit_mask": all_tokens},
        batch={"coords_nm": gt.unsqueeze(0), "atom37_mask": atom_mask.unsqueeze(0), **kwargs},
        loss_mask=all_tokens,  # these tests are about permutation, so supervise everything
        loss_ty="l12",
        atom_permutation=atom_permutation,
    )


def as_ligand(ccd_code: str) -> AtomArray:
    """A heavy-atom CCD entry annotated as its own non-polymer unit."""
    ligand = struc.info.residue(ccd_code)
    ligand = ligand[ligand.element != "H"]
    n_atoms = ligand.array_length()
    ligand.set_annotation("pn_unit_iid", np.full(n_atoms, "L0"))
    ligand.set_annotation("is_polymer", np.zeros(n_atoms, dtype=bool))
    return ligand


def relabelled(atom_array: AtomArray) -> AtomArray:
    """A copy whose coordinates are relabelled by a real automorphism of the molecule.

    Applying the symmetry itself, rather than a hand-picked atom pair, is what lets a case be
    an arbitrarily hard molecule: no atom names to look up and none to get wrong.
    """
    groups = [np.asarray(group) for group in find_automorphisms_with_networkx(atom_array) if len(group) > 1]
    assert groups, "case has no symmetry, so it would pass with resolution deleted"
    moved = atom_array.copy()
    for group in groups:
        moved.coord[group[0]] = atom_array.coord[group[1]]  # ...first non-identity candidate
    return moved


# ...residues out of 6wtf reach the named atom37 slots; CCD ligands reach the atomized slot
# and carry the hardest automorphism groups in atomworks' reference set (up to 26 movable atoms)
AE_POLYMER_CASES = ["TYR", "ASP", "GLU", "ARG", "PHE", "SF4"]
AE_LIGAND_CASES = ["ATP", "GDP", "HEM", "FAD", "BVE", "ZW7", "H9H", "XQR", "MPM", "1M9", "9P6"]


def assert_permutation_absorbs_relabelling(source: AtomArray) -> None:
    """The relabelled copy must cost nothing with permutation, and something without it."""
    annotated, gt, atom_mask = as_atom37(source)
    _, predicted, _ = as_atom37(relabelled(source))
    predicted = predicted.detach().requires_grad_(True)

    losses = autoencoder_loss(annotated, gt, predicted, atom_mask, extra_info=precomputed_for(annotated))

    torch.testing.assert_close(losses["struct_l2"], torch.zeros(1), atol=1e-7, rtol=0)
    torch.testing.assert_close(losses["struct_l1"], torch.zeros(1), atol=1e-7, rtol=0)
    # ... permutation happens under no_grad, so gradients must still reach the prediction
    (losses["struct_l1"] + losses["struct_l2"]).sum().backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()

    # ... without permutation the relabelling costs, so none of the above passes vacuously
    unpermuted = autoencoder_loss(annotated, gt, predicted.detach(), atom_mask, atom_permutation=False)
    assert float(unpermuted["struct_l2"]) > 0.0

    # ... and asked to permute with nothing precomputed, the atom37 path refuses rather than
    # quietly scoring an unpermuted loss -- it holds tensors, so it cannot derive them itself
    with pytest.raises(ValueError, match="precompute_atom_permutation"):
        autoencoder_loss(annotated, gt, predicted.detach(), atom_mask)


@pytest.mark.parametrize("res_name", AE_POLYMER_CASES)
def test_autoencoder_loss_ignores_a_relabelling_of_a_residue(res_name: str, ground_truth: AtomArray) -> None:
    """Gather into atom37, permute, scatter back, for a residue of a real structure."""
    res_id = np.unique(ground_truth.res_id[ground_truth.res_name == res_name])[0]
    assert_permutation_absorbs_relabelling(ground_truth[ground_truth.res_id == res_id])


@pytest.mark.parametrize("ccd_code", AE_LIGAND_CASES)
def test_autoencoder_loss_ignores_a_relabelling_of_a_hard_ligand(ccd_code: str) -> None:
    """The same, for ligands whose symmetry groups are far larger than any amino acid's."""
    assert_permutation_absorbs_relabelling(as_ligand(ccd_code))


# +------------------- whole plinder systems, checked in -------------------+

# Every case above is one residue or an isolated CCD ligand in its own pn_unit, so none has a
# group spanning residues. 1gya does: a covalent modification reassigns a residue onto the
# ligand's pn_unit, which is then searched whole. 1h0a carries the widest search, 216 candidates.
PLINDER_CASES = [
    # (stem, min residues spanned by one unit, min candidate permutations in one group)
    ("plinder_1gya_covalent_modification", 9, 2),
    ("plinder_1h0a_wide_automorphism", 1, 216),
]

_HEAVY_ATOMS = RCSB_WITH_METALS.replace(hydrogen_policy="remove")


@pytest.fixture(scope="module")
def plinder_system():
    """Parse a checked-in plinder system, heavy atoms with resolved coordinates."""

    def load(stem: str) -> AtomArray:
        structure = parse(str(_GENERATED / f"{stem}.cif.gz"), config=_HEAVY_ATOMS)["assemblies"]["1"][0]
        return structure[np.isfinite(structure.coord).all(axis=-1)]

    return load


@pytest.mark.parametrize(
    ("stem", "min_span_residues", "min_candidates"),
    PLINDER_CASES,
    ids=[case[0] for case in PLINDER_CASES],
)
def test_plinder_system_keeps_the_symmetry_it_was_chosen_for(
    stem: str, min_span_residues: int, min_candidates: int, plinder_system
) -> None:
    """Guard the fixtures: each must still carry the topology it is here to exercise."""
    structure = plinder_system(stem)

    # ... res_id alone repeats across chains, so pair it with the chain to key a residue
    res_key = structure.res_id + 1000 * np.unique(structure.chain_id, return_inverse=True)[1]
    spans = [len(np.unique(res_key[np.asarray(unit)])) for unit in build_automorphism_units(structure)]
    assert max(spans) >= min_span_residues

    groups = [np.asarray(g) for g in find_automorphisms_with_networkx(structure) if len(g) > 1]
    assert max(int(group.shape[0]) for group in groups) >= min_candidates


@pytest.mark.parametrize("stem", [case[0] for case in PLINDER_CASES])
def test_metrics_ignore_a_relabelling_of_a_whole_plinder_system(stem: str, plinder_system) -> None:
    """Relabelling by the system's own automorphisms must cost nothing -- and something without."""
    ground_truth = plinder_system(stem)
    predicted = relabelled(ground_truth)
    assert not np.allclose(predicted.coord, ground_truth.coord), "the relabelling must move something"

    assert AllAtomLDDT().compute(predicted, ground_truth)["all_atom_lddt"] == pytest.approx(1.0)
    assert StructureComparisonMetric(metric_type="rmsd").compute(ground_truth, predicted)["rmsd"] == pytest.approx(
        0.0, abs=1e-6
    )
    # ... without resolution the same relabelling is penalized, so the above is not vacuous
    assert AllAtomLDDT(resolve_symmetry=()).compute(predicted, ground_truth)["all_atom_lddt"] < 1.0


def reference_permutation(x_pred, x_gt, gt_mask, automorphisms):
    """The per-group loop `atom_permutation` replaced, kept as the thing it must match."""
    permutation = torch.arange(x_gt.shape[0])
    for automorphism in automorphisms:
        group = torch.as_tensor(np.asarray(automorphism), dtype=torch.long)
        if group.ndim != 2 or group.shape[0] <= 1 or group.shape[1] == 0:
            continue
        movable = torch.any(group != group[0], dim=0)
        destinations, sources = group[0][movable], group[:, movable]
        active = gt_mask[destinations]
        if not bool(active.any()):
            continue
        destinations, sources = destinations[active], sources[:, active]
        legal = torch.all(gt_mask[sources], dim=-1)
        error = torch.square(x_pred[destinations] - x_gt[sources]).sum(dim=-1).mean(dim=-1)
        error = error.nan_to_num(nan=torch.inf, posinf=torch.inf, neginf=torch.inf).masked_fill(~legal, torch.inf)
        permutation[destinations] = permutation[sources[int(error.argmin())]]
    return permutation


def test_batched_search_matches_a_per_group_loop(plinder_system) -> None:
    """Batching the groups must pick the identical permutation the per-group loop picked."""
    structure = plinder_system("plinder_1h0a_wide_automorphism")
    groups = [np.asarray(g) for g in find_automorphisms_with_networkx(structure) if len(g) > 1]
    x_gt = torch.tensor(structure.coord, dtype=torch.float32)
    x_pred = x_gt + torch.tensor(np.random.default_rng(0).normal(scale=1.5, size=x_gt.shape), dtype=torch.float32)
    gt_mask = torch.ones(x_gt.shape[0], dtype=torch.bool)

    batched = atom_permutation(x_pred, x_gt, gt_mask, groups, align=False)
    looped = reference_permutation(x_pred, x_gt, gt_mask, groups)

    assert not torch.equal(batched, torch.arange(x_gt.shape[0])), "noise this large must move something"
    assert torch.equal(batched, looped)
