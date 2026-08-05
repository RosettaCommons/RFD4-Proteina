"""Refold pre-cutoff crystals (PTM, HEM, N-glycan, custom CCD, protein-DNA, RNA) and check RMSD, bonds and clashes."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import biotite.structure as struc
import numpy as np
import pytest
from atomworks.constants import PDB_MIRROR_PATH
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.io.utils.query import mask as query_mask
from atomworks.ml.conditions import C_CRD
from atomworks.ml.transforms.filters import remove_nucleic_acid_terminal_oxygen, remove_protein_terminal_oxygen
from biotite.structure import AtomArray

from rfproteina.metrics.design_metrics import BondLengthMetric, ClashMetric
from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput
from rfproteina.utils.atom_permutation import permute_gt_atoms
from tests.conftest import TEST_DATA_DIR
from tests.e2e.helpers.gpu_e2e import require_preval_oracle

pytestmark = pytest.mark.gpu

_CUSTOM_HEM_CODE = "L:0"  # same shape as the re-code in tests/test_oracles.py::modified_1a1e
# a3m for 2w3o chain A, sliced to the observed residues.
_PROTEIN_MSA_A3M = TEST_DATA_DIR / "msas/2w3o_A.a3m.gz"
# a3m for the 1y26 riboswitch, from AF3's own nhmmer search against Rfam.
_RNA_MSA_A3M = TEST_DATA_DIR / "msas/1y26_A.a3m.gz"


# The lane's AF3 daemon set, captured before either fixture rewrites the variable.
_AF3_SOCKET_TEMPLATE = os.environ.get("AF3_LAUNCHER_SOCKET_TEMPLATE", "")

# How closely the user-CCD heme refold must reproduce the built-in-CCD one.
_CUSTOM_CCD_PROTEIN_RMSD = 0.02
_CUSTOM_CCD_LIGAND_RMSD = 0.03


def _protein_ca(atom_array: AtomArray) -> np.ndarray:
    """Mask of protein CA atoms (``is_protein`` also excludes calcium ions, which are named ``CA``)."""
    ensure_annotations(atom_array, "is_protein")
    return atom_array.is_protein & (atom_array.atom_name == "CA")


def _bonded_atoms(atom_array: AtomArray) -> np.ndarray:
    """Every atom in a multi-atom residue, i.e. everything but lone ions.

    Neither runtime anchors a lone ion: on 1arx's two Ca and one K, AF3 is 7.3 A out and OF3 22.4.
    """
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    sizes = np.diff(starts)
    return np.repeat(sizes, sizes) > 1


def _nucleic(atom_array: AtomArray) -> np.ndarray:
    """Mask of nucleic-acid atoms (both DNA strands; the RNA chain)."""
    ensure_annotations(atom_array, "is_nucleic_acid")
    return atom_array.is_nucleic_acid


class Example(NamedTuple):
    """One refold case: which crystal, whether it is templated, and how to score it."""

    name: str
    pdb: str
    template: str | None
    allowable_af3: float
    align: Callable[[AtomArray], np.ndarray]
    measure: Callable[[AtomArray], np.ndarray]
    bonded_ligand: bool = False  # has a covalent ligand-polymer bond AF3 must reproduce
    allowable_of3: float = 0.0  # ported weights, same path; ~15% over measured


_EXAMPLES: tuple[Example, ...] = (
    Example("2w3o", "2w3o", "A", 0.25, _protein_ca, _protein_ca, allowable_of3=0.25),
    Example("2w3o_msa", "2w3o", None, 0.30, _protein_ca, _protein_ca, allowable_of3=0.30),
    # OF3's residual here is its N-glycan (0.91 vs AF3's 0.45); heme and its Fe are placed
    # better than AF3 (0.21 vs 0.31, 0.13 vs 0.24).
    Example("1arx", "1arx", "A", 0.60, _protein_ca, _bonded_atoms, True, allowable_of3=0.65),
    Example(
        "1arx_custom_hem",
        "1arx",
        "A",
        0.60,
        _protein_ca,
        _bonded_atoms,
        True,
        allowable_of3=0.65,
    ),
    Example("7psx", "7psx", None, 1.35, _nucleic, _nucleic, allowable_of3=1.50),
    Example("1y26", "1y26", None, 0.75, _nucleic, _nucleic, allowable_of3=0.45),
    Example("1y26_msa", "1y26", None, 0.65, _nucleic, _nucleic, allowable_of3=0.45),
)


class SubsetRMSD(NamedTuple):
    """RMSD of part of one refold, for regions whole-structure RMSD would dilute.

    ``select`` and ``align`` are atomworks query expressions.
    """

    pdb: str
    label: str
    select: str
    align: str
    allowable_af3: float
    allowable_of3: float

    def score(self, refolds: dict[str, AtomArray]) -> float:
        crystal = refolds[f"crystal_{self.pdb}"]
        return _rmsd(crystal, refolds[self.pdb], _query(crystal, self.align), _query(crystal, self.select))


_SUBSET_RMSDS: tuple[SubsetRMSD, ...] = (
    # 28 glycan atoms among 2507 vanish into the whole-structure number. OF3 trails AF3 here,
    # but the linkage is a real bond: ASN-NAG 1.41 A against a 1.45 A crystal.
    SubsetRMSD(
        "1arx",
        "N-glycan",
        "res_name == 'NAG'",
        "is_protein & (atom_name == 'CA')",
        allowable_af3=0.60,  # measured 0.446
        allowable_of3=1.10,  # measured 0.914
    ),
)


class BondGeometry(NamedTuple):
    """Worst deviation, in Angstrom, of a bond's length from the crystal's.

    Every ``select_a`` x ``select_b`` pair within a residue is measured.
    """

    pdb: str
    label: str
    select_a: str
    select_b: str
    allowable_af3: float
    allowable_of3: float

    def score(self, refolds: dict[str, AtomArray]) -> float:
        crystal, predicted = refolds[f"crystal_{self.pdb}"], refolds[self.pdb]
        a_idx = np.where(_query(crystal, self.select_a))[0]
        b_idx = np.where(_query(crystal, self.select_b))[0]
        pairs = [(i, j) for i in a_idx for j in b_idx if crystal.res_id[i] == crystal.res_id[j]]
        return max(
            abs(
                float(np.linalg.norm(predicted.coord[j] - predicted.coord[i]))
                - float(np.linalg.norm(crystal.coord[j] - crystal.coord[i]))
            )
            for i, j in pairs
        )


_BOND_GEOMETRIES: tuple[BondGeometry, ...] = (
    # Heme's iron is the one metal both runtimes are told about -- the CCD types these four
    # SINGLE -- unlike a lone ion, whose metalc coordination clean_structure discards.
    BondGeometry(
        "1arx",
        "heme Fe-N(pyrrole)",
        "(res_name == 'HEM') & (atom_name == 'FE')",
        "(res_name == 'HEM') & (atom_name in ['NA', 'NB', 'NC', 'ND'])",
        allowable_af3=0.15,  # measured 0.077, against a 1.94-1.97 A crystal
        allowable_of3=0.15,  # measured 0.071
    ),
)


def _load_assembly(path: Path) -> AtomArray:
    """Load ``path``'s first biological assembly, shaped like a design output."""
    config = ParseConfig.from_preset("rcsb", hydrogen_policy="remove", return_atom_array_plus=True)
    atom_array = parse(path, config=config)["assemblies"]["1"][0]

    # OpenFold3 strips these from every polymer residue, so its weights never learned to place
    # them; left in, they land on a neighbour (OXT 0.83 A from its own C) and read as clashes.
    atom_array = remove_nucleic_acid_terminal_oxygen(remove_protein_terminal_oxygen(atom_array))
    return atom_array[~np.isnan(atom_array.coord).any(axis=-1)]


def _templated(atom_array: AtomArray, chain_id: str | None) -> AtomArray:
    """Copy of ``atom_array`` with ``chain_id`` flagged for templating; nothing when ``chain_id`` is None."""
    marked = atom_array.copy()
    C_CRD.set_mask(marked, marked.chain_id == (chain_id or ""))  # no chain is named "" -> templates nothing
    return marked


def _rename_residue(atom_array: AtomArray, frm: str, to: str) -> AtomArray:
    """Copy of ``atom_array`` with residue code ``frm`` re-coded to ``to``."""
    renamed = atom_array.copy()
    renamed.res_name[renamed.res_name == frm] = to
    return renamed


_PERMUTATION_ANNOTATIONS = ("pn_unit_iid", "is_polymer")


def _borrow_annotations(target: AtomArray, source: AtomArray) -> AtomArray:
    """Copy the annotations :py:func:`permute_gt_atoms` needs from ``source`` onto ``target``.

    Predictions are parsed minimally and so lack them; identical ordering makes a copy valid.
    """
    out = target.copy()
    present = set(source.get_annotation_categories())
    for name in _PERMUTATION_ANNOTATIONS:
        if name not in out.get_annotation_categories() and name in present:
            out.set_annotation(name, source.get_annotation(name))
    return out


def _rmsd(
    reference: AtomArray,
    subject: AtomArray,
    align: np.ndarray,
    measure: np.ndarray | None = None,
    symmetric: bool = True,
) -> float:
    """Superimpose ``subject`` onto ``reference`` over ``align``; return RMSD over ``measure`` (default ``align``).

    Symmetric ground-truth atoms are permuted first, in the alignment frame, so an
    equally-correct placement of a symmetric group is not charged as error.
    """
    if symmetric:
        missing = [n for n in _PERMUTATION_ANNOTATIONS if n not in reference.get_annotation_categories()]
        # Never skip quietly: a missing annotation would over-report symmetric groups.
        assert not missing, f"reference lacks {missing}; symmetry resolution would be skipped"
        reference = permute_gt_atoms(_borrow_annotations(subject, reference), reference, frame_mask=align)[0]
    _, transform = struc.superimpose(reference[align], subject[align])
    fitted = transform.apply(subject)
    scored = align if measure is None else measure
    return float(struc.rmsd(reference[scored], fitted[scored]))


def _query(atom_array: AtomArray, expr: str) -> np.ndarray:
    """Boolean mask for an atomworks query expression; an empty match would silently pass."""
    ensure_annotations(atom_array, "is_protein", "is_nucleic_acid")
    selected = query_mask(atom_array, expr)
    assert selected.any(), f"selection matched no atoms: {expr!r}"
    return selected


def _fold_all_examples(out_dir: Path) -> dict[str, AtomArray]:
    """Fold every example through one warm worker; returns ``{name: prediction}`` plus ``crystal_<pdb>``.

    Whichever weights the ambient ``AF3_LAUNCHER_SOCKET_TEMPLATE`` daemon holds are the ones used,
    so the OF3 fixture differs from the AF3 one only in that variable.
    """
    mirror = Path(PDB_MIRROR_PATH)
    crystals = {ex.pdb: _load_assembly(mirror / ex.pdb[1:3] / f"{ex.pdb}.cif.gz") for ex in _EXAMPLES}

    # The MSA travels the way inference supplies it: through the `_msa_paths_by_chain_id` CIF category.
    annotated_cif = out_dir / "2w3o_msa.cif"
    to_cif_file(
        crystals["2w3o"], annotated_cif, extra_categories={"msa_paths_by_chain_id": {"A": str(_PROTEIN_MSA_A3M)}}
    )
    msa_paths = {"A": parse(annotated_cif, config=ParseConfig.from_preset("minimal"))["chain_info"]["A"]["msa_path"]}
    msa_paths_by_name = {"2w3o_msa": msa_paths, "1y26_msa": {"A": _RNA_MSA_A3M}}

    designs: dict[str, AtomArray] = {}
    for ex in _EXAMPLES:
        design = crystals[ex.pdb]
        if ex.name == "1arx_custom_hem":
            design = _rename_residue(design, "HEM", _CUSTOM_HEM_CODE)
        designs[ex.name] = _templated(design, ex.template)

    # seed=1 pins the diffusion seed so these RMSDs are reproducible; production oracles omit it.
    results: dict[str, AtomArray] = {f"crystal_{k}": v for k, v in crystals.items()}
    oracle = AF3Oracle(oracle_name="af3", template_query="mask_coordinate_1_atom", seed=1)
    oracle.setup()
    try:
        jobs = [
            OracleJobInput(aa, name, out_dir, msa_paths_by_chain_id=msa_paths_by_name.get(name))
            for name, aa in designs.items()
        ]
        n_ok = oracle.fold_batch(jobs)
        assert n_ok == len(jobs), f"AF3 folded {n_ok}/{len(jobs)} designs"
        for name in designs:
            cif = oracle.refolded_cif_path(out_dir, name)
            assert cif.is_file(), f"no refolded CIF for {name} at {cif}"
            results[name] = as_atom_array_plus(parse(cif, config=ParseConfig.from_preset("minimal"))["asym_unit"][0])
    finally:
        oracle.teardown()

    # The oracle writes the refold back onto the input array, so RMSD masks need no re-indexing.
    for ex in _EXAMPLES:
        assert np.array_equal(results[ex.name].atom_name, crystals[ex.pdb].atom_name), f"{ex.name}: ordering drifted"
    return results


@pytest.fixture(scope="module")
def refolds(tmp_path_factory, monkeypatch_module) -> dict[str, AtomArray]:
    """Every example folded with AlphaFold3 weights."""
    require_preval_oracle("af3")
    if not PDB_MIRROR_PATH or not Path(PDB_MIRROR_PATH).is_dir():
        pytest.skip("PDB_MIRROR_PATH is unset")
    monkeypatch_module.setenv("AF3_LAUNCHER_SOCKET_TEMPLATE", _AF3_SOCKET_TEMPLATE)
    return _fold_all_examples(tmp_path_factory.mktemp("af3_structure_quality"))


@pytest.fixture(scope="module")
def of3_refolds(tmp_path_factory, monkeypatch_module) -> dict[str, AtomArray]:
    """Every example folded with converted OpenFold3 weights, via the lane's second daemon set."""
    require_preval_oracle("af3")
    if not PDB_MIRROR_PATH or not Path(PDB_MIRROR_PATH).is_dir():
        pytest.skip("PDB_MIRROR_PATH is unset")

    of3_socket = os.environ.get("OF3_LAUNCHER_SOCKET_TEMPLATE")
    if not of3_socket:
        message = "OF3_LAUNCHER_SOCKET_TEMPLATE is unset; set OF3_MODEL_DIR in .ipd/.env.<cluster>"
        pytest.fail(message) if os.environ.get("REQUIRE_AF3") == "1" else pytest.skip(message)

    # Defence in depth: if the lane ever hands us the AF3 template, this arm would fold with
    # AF3 weights and pass, which is worse than not running at all.
    if of3_socket == _AF3_SOCKET_TEMPLATE:
        pytest.fail(
            "OF3_LAUNCHER_SOCKET_TEMPLATE equals the AF3 one; the OF3 arm would fold with "
            "AF3 weights. Check the OF3 daemon launch in run_gpu_tests.sh."
        )

    monkeypatch_module.setenv("AF3_LAUNCHER_SOCKET_TEMPLATE", of3_socket)
    # The oracle reads AF3_MODEL_DIR to decide whether to atomize covalently modified
    # residues; redirecting only the socket would leave it looking at the AF3 weights.
    of3_model_dir = os.environ.get("OF3_MODEL_DIR")
    if of3_model_dir:
        monkeypatch_module.setenv("AF3_MODEL_DIR", of3_model_dir)
    return _fold_all_examples(tmp_path_factory.mktemp("of3_structure_quality"))


@pytest.fixture(scope="module")
def monkeypatch_module():
    """Module-scoped ``monkeypatch`` (the built-in fixture is function-scoped)."""
    patcher = pytest.MonkeyPatch()
    yield patcher
    patcher.undo()


def _quality_failures(refolds: dict[str, AtomArray], arm: str) -> list[str]:
    """Every bound ``arm`` violates: per-example RMSD, bond lengths and clashes, then the subset checks."""
    failures: list[str] = []
    for example in _EXAMPLES:
        crystal, predicted = refolds[f"crystal_{example.pdb}"], refolds[example.name]
        bound = getattr(example, f"allowable_{arm}")

        rmsd = _rmsd(crystal, predicted, example.align(crystal), example.measure(crystal))
        if rmsd >= bound:
            failures.append(f"{example.name}: RMSD {rmsd:.3f} >= {bound}")

        bonds = BondLengthMetric(verbose=True).compute(predicted)
        if bonds["bond_length.frac_unreasonable"] != 0:
            failures.append(f"{example.name}: unreasonable bonds {bonds['bond_length.frac_unreasonable']}")
        # NaN (not 0) when the ligand-polymer bond was dropped entirely -- the regression to catch.
        key = "bond_length.frac_unreasonable.between_chain.nonpolymer_to_polymer"
        if example.bonded_ligand and bonds[key] != 0:
            failures.append(f"{example.name}: {key} = {bonds[key]}")

        clashes = ClashMetric(threshold=1.5).compute(predicted[_bonded_atoms(predicted)])
        if clashes["frac_atoms_clashing_1p5A"] != 0:
            failures.append(f"{example.name}: clashing {clashes['frac_atoms_clashing_1p5A']}")

    for check in (*_SUBSET_RMSDS, *_BOND_GEOMETRIES):
        value, bound = check.score(refolds), getattr(check, f"allowable_{arm}")
        if value >= bound:
            failures.append(f"{check.pdb} {check.label}: {value:.3f} >= {bound}")
    return failures


def test_af3_refold_structure_quality(refolds):
    """Each refold reproduces the deposited fold, protein-ligand bonds and clash-free packing."""
    failures = _quality_failures(refolds, "af3")

    # symmetric=False: both sides are predictions, so neither carries the permutation annotations.
    crystal = refolds["crystal_1arx"]
    ca, heme = _protein_ca(crystal), crystal.res_name == "HEM"
    protein = _rmsd(refolds["1arx"], refolds["1arx_custom_hem"], ca, symmetric=False)
    ligand = _rmsd(refolds["1arx"], refolds["1arx_custom_hem"], align=ca, measure=heme, symmetric=False)
    if protein >= _CUSTOM_CCD_PROTEIN_RMSD or ligand >= _CUSTOM_CCD_LIGAND_RMSD:
        failures.append(f"user-CCD heme refold drifted: protein {protein:.3f}, ligand {ligand:.3f}")
    assert not failures, "\n".join(failures)


def test_of3_refold_structure_quality(of3_refolds):
    """Converted OpenFold3 weights reproduce the deposited folds through the same path as AF3.

    A converter regression like the missing ``msa_activations`` permutation put 1y26 at 24.9 A.
    """
    assert not (failures := _quality_failures(of3_refolds, "of3")), "\n".join(failures)
