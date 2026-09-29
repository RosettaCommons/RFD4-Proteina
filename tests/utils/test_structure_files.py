"""Compression-independent structure filenames and ambiguity checks."""

import pytest

from rfproteina.utils.structure_files import (
    find_structure_file,
    is_structure_file,
    structure_file_stem,
    unique_structure_files,
)


@pytest.mark.parametrize("suffix", [".cif", ".cif.gz", ".cif.zst", ".bcif", ".bcif.gz", ".bcif.zst"])
def test_structure_file_stem_preserves_dots(suffix):
    assert structure_file_stem(f"1abc.def_generated{suffix}") == "1abc.def_generated"


@pytest.mark.parametrize(
    "filename",
    [
        "design.txt",
        "design.cif.bak",
        "design.cif.zst.tmp",
        "design.json",
        "design.json.gz",
        "design.json.gzip",
        "design.json.zst",
    ],
)
def test_structure_file_stem_rejects_unsupported_suffix(filename):
    assert not is_structure_file(filename)
    assert not is_structure_file(filename, include_pdb=True)
    with pytest.raises(ValueError, match="Unsupported structure filename"):
        structure_file_stem(filename)


@pytest.mark.parametrize("suffix", [".cif.gz", ".cif.zst"])
def test_unique_structure_files_rejects_sibling_encodings(tmp_path, suffix):
    paths = [tmp_path / "design.cif", tmp_path / f"design{suffix}"]
    with pytest.raises(ValueError, match="Ambiguous structure files.*Keep one encoding") as error:
        unique_structure_files(paths)
    assert all(str(path) in str(error.value) for path in paths)


def test_find_structure_file_missing_parent(tmp_path):
    assert find_structure_file(tmp_path / "missing" / "design.cif") is None
