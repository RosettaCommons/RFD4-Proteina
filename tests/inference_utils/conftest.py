"""Fixtures for condition-spec / contig-spec tests.

Uses real (small) PDB entries via `cached_parse` rather than hand-rolled
AtomArrays, matching `tests/transforms/conftest.py`'s convention -- hand-built
arrays are missing the standard-annotation scaffolding (`chain_type`,
`transformation_id`, `chain_iid`, ...) that `insert_expandable_segment` relies
on, which real parsed structures always carry.
"""

import pytest
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.utils.testing import cached_parse


@pytest.fixture
def lysozyme():
    """Small single-chain protein (6lyz): chain A, res_id 1-129."""
    return cached_parse("6lyz", hydrogen_policy="remove")["atom_array"]


@pytest.fixture
def trna():
    """Small single-chain RNA (1ehz): chain A, res_id 1-76."""
    arr = cached_parse("1ehz", hydrogen_policy="remove")["atom_array"]
    return arr[arr.chain_id == "A"]


@pytest.fixture
def two_chain_complex():
    """Small two-protein-chain complex (1mso): chain A (res_id 1-21), chain B (res_id 1-30).

    1mso is a crystallographic insulin hexamer -- the raw parse has 3 symmetry copies
    of each chain sharing the same `chain_id`/`res_id` (disambiguated only by
    `transformation_id`). Restricted to a single copy (`transformation_id == "1"`) so
    `chain_id`+`res_id` alone unambiguously addresses one residue, matching what a
    contig token assumes. Also carries small extra chains (C-F) that tests can safely
    ignore, since `contig` subsetting drops anything not explicitly referenced.
    """
    arr = cached_parse("1mso", hydrogen_policy="remove")["atom_array"]
    return arr[arr.transformation_id == "1"]


@pytest.fixture
def lysozyme_cif_path(lysozyme, tmp_path):
    """A real, on-disk CIF file for the lysozyme fixture (for `load_condition_spec_inputs`)."""
    path = tmp_path / "6lyz.cif"
    to_cif_file(lysozyme, path)
    return path
