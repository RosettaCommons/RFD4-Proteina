"""Refold pre-cutoff crystals (PTM, HEM, N-glycan, custom CCD, protein-DNA, RNA) and check RMSD, bonds and clashes."""

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
from atomworks.ml.conditions import C_CRD
from biotite.structure import AtomArray

from rfproteina.metrics.design_metrics import BondLengthMetric, ClashMetric
from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput
from tests.conftest import TEST_DATA_DIR
from tests.e2e.helpers.gpu_e2e import require_preval_oracle

pytestmark = pytest.mark.gpu

_CUSTOM_HEM_CODE = "L:0"  # same shape as the re-code in tests/test_oracles.py::modified_1a1e
# a3m for 2w3o chain A, sliced to the observed residues.
_PROTEIN_MSA_A3M = TEST_DATA_DIR / "msas/2w3o_A.a3m.gz"
# a3m for the 1y26 riboswitch, from AF3's own nhmmer search against Rfam.
_RNA_MSA_A3M = TEST_DATA_DIR / "msas/1y26_A.a3m.gz"

# How closely the user-CCD heme refold must reproduce the built-in-CCD one.
_CUSTOM_CCD_PROTEIN_RMSD = 0.02
_CUSTOM_CCD_LIGAND_RMSD = 0.03


def _protein_ca(atom_array: AtomArray) -> np.ndarray:
    """Mask of protein CA atoms (``is_protein`` also excludes calcium ions, which are named ``CA``)."""
    ensure_annotations(atom_array, "is_protein")
    return atom_array.is_protein & (atom_array.atom_name == "CA")


def _all_atoms(atom_array: AtomArray) -> np.ndarray:
    """Mask of every atom (hydrogens are already removed on load)."""
    return np.ones(atom_array.array_length(), dtype=bool)


def _nucleic(atom_array: AtomArray) -> np.ndarray:
    """Mask of nucleic-acid atoms (both DNA strands; the RNA chain)."""
    ensure_annotations(atom_array, "is_nucleic_acid")
    return atom_array.is_nucleic_acid


class Example(NamedTuple):
    """One refold case: which crystal, whether it is templated, and how to score it."""

    name: str
    pdb: str
    template: str | None
    allowable_rmsd: float
    align: Callable[[AtomArray], np.ndarray]
    measure: Callable[[AtomArray], np.ndarray]
    bonded_ligand: bool = False  # has a covalent ligand-polymer bond AF3 must reproduce


_EXAMPLES: tuple[Example, ...] = (
    Example("2w3o", "2w3o", "A", 0.25, _protein_ca, _protein_ca),
    Example("2w3o_msa", "2w3o", None, 0.30, _protein_ca, _protein_ca),
    Example("1arx", "1arx", "A", 0.70, _protein_ca, _all_atoms, True),
    Example("1arx_custom_hem", "1arx", "A", 0.70, _protein_ca, _all_atoms, True),
    Example("7psx", "7psx", None, 1.35, _nucleic, _nucleic),
    Example("1y26", "1y26", None, 0.75, _nucleic, _nucleic),
    Example("1y26_msa", "1y26", None, 0.65, _nucleic, _nucleic),
)


def _load_assembly(path: Path) -> AtomArray:
    """Load ``path``'s first biological assembly, shaped like a design output."""
    config = ParseConfig.from_preset("rcsb", hydrogen_policy="remove", return_atom_array_plus=True)
    atom_array = parse(path, config=config)["assemblies"]["1"][0]

    atom_array = atom_array[atom_array.atom_name != "OXT"]
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


def _rmsd(reference: AtomArray, subject: AtomArray, align: np.ndarray, measure: np.ndarray | None = None) -> float:
    """Superimpose ``subject`` onto ``reference`` over ``align``; return RMSD over ``measure`` (default ``align``)."""
    _, transform = struc.superimpose(reference[align], subject[align])
    fitted = transform.apply(subject)
    scored = align if measure is None else measure
    return float(struc.rmsd(reference[scored], fitted[scored]))


@pytest.fixture(scope="module")
def refolds(tmp_path_factory) -> dict[str, AtomArray]:
    """Fold every example through one warm AF3 worker; returns ``{name: prediction}`` plus ``crystal_<pdb>``."""
    require_preval_oracle("af3")
    if not PDB_MIRROR_PATH or not Path(PDB_MIRROR_PATH).is_dir():
        pytest.skip("PDB_MIRROR_PATH is unset")

    out_dir = tmp_path_factory.mktemp("af3_structure_quality")
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


def test_af3_refold_structure_quality(refolds):
    """Each refold reproduces the deposited fold, protein-ligand bonds and clash-free packing."""
    # One batched fold serves every example, so check them all and report every failure at once
    # rather than parametrising (which would refold the batch on each xdist worker).
    failures: list[str] = []
    for example in _EXAMPLES:
        crystal = refolds[f"crystal_{example.pdb}"]
        predicted = refolds[example.name]

        rmsd = _rmsd(crystal, predicted, example.align(crystal), example.measure(crystal))
        if rmsd >= example.allowable_rmsd:
            failures.append(f"{example.name}: RMSD {rmsd:.3f} >= {example.allowable_rmsd}")

        bonds = BondLengthMetric(verbose=True).compute(predicted)
        if bonds["bond_length.frac_unreasonable"] != 0:
            failures.append(f"{example.name}: unreasonable bonds {bonds['bond_length.frac_unreasonable']}")
        # NaN (not 0) when AF3 dropped the ligand-polymer bond entirely — the regression to catch.
        key = "bond_length.frac_unreasonable.between_chain.nonpolymer_to_polymer"
        if example.bonded_ligand and bonds[key] != 0:
            failures.append(f"{example.name}: {key} = {bonds[key]}")

        clashes = ClashMetric(threshold=1.5).compute(predicted)
        if clashes["frac_atoms_clashing_1p5A"] != 0:
            failures.append(f"{example.name}: clashing {clashes['frac_atoms_clashing_1p5A']}")
    assert not failures, "\n".join(failures)

    # Self-validating: the user-CCD heme must land where the built-in-CCD refold put it.
    crystal_1arx = refolds["crystal_1arx"]
    ca, heme = _protein_ca(crystal_1arx), crystal_1arx.res_name == "HEM"
    assert _rmsd(refolds["1arx"], refolds["1arx_custom_hem"], ca) < _CUSTOM_CCD_PROTEIN_RMSD
    assert _rmsd(refolds["1arx"], refolds["1arx_custom_hem"], align=ca, measure=heme) < _CUSTOM_CCD_LIGAND_RMSD
