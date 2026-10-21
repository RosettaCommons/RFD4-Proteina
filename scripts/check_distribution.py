"""Check built wheel/sdist contents without importing the model runtime."""

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement

# Size ceilings keep bulk data out of the published artifacts.
MAX_BYTES = {".whl": 2_000_000, ".gz": 2_000_000}


def check_distribution(path: Path) -> None:
    """Reject incomplete/private artifacts and direct URL dependencies."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata = archive.read(next(name for name in names if name.endswith(".dist-info/METADATA")))
            entry_points = archive.read(next(name for name in names if name.endswith(".dist-info/entry_points.txt")))
        required = {
            "rfd4_proteina/__init__.py",
            "rfproteina/proteina.py",
            "rfproteina/metrics/physics/score_pose_rosetta.py",
            "script_utils/entrypoints.py",
            "rfproteina/configs/inference.yaml",
            "rfproteina/configs/training_proteina.yaml",
            "rfproteina/.project-root",
            "rfproteina/data/cath_label_mapping.pkl",
        }
        assert (
            "rfproteina/data/benchmarks/monomer/monomer-short.cif.zst" in names
        ), "Missing the quickstart example input"
        benchmark_dirs = {PurePosixPath(n).parts[3] for n in names if n.startswith("rfproteina/data/benchmarks/")}
        assert benchmark_dirs == {"monomer"}, f"Only the quickstart examples may ship in the wheel: {benchmark_dirs}"
        for command in ("rfd4-generate", "rfd4-evaluate", "rfd4-train", "rfd4-train-ae"):
            assert f"{command} = ".encode() in entry_points, f"Missing entry point: {command}"
    else:
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            assert all(member.isfile() or member.isdir() for member in members), "Archive contains links"
            names = [member.name.split("/", 1)[1] for member in members if "/" in member.name]
            metadata = archive.extractfile(
                next(member for member in members if member.name.endswith("/PKG-INFO"))
            ).read()
        required = {"pyproject.toml", "README.md", "LICENSE", ".project-root", "src/rfd4_proteina/__init__.py"}
    assert required <= set(names), f"Missing package files: {required - set(names)}"
    for name in names:
        parts = PurePosixPath(name).parts
        assert not set(parts) & {".ipd", ".git", ".github", ".agents", ".claude", "__pycache__"}, name
        assert not any(part.startswith((".env", ".user_info_")) for part in parts), name
        assert "custom_metrics" not in parts, name
        assert "cdr_torsion_data" not in parts, name
    assert path.stat().st_size < MAX_BYTES[path.suffix], f"{path.name} is {path.stat().st_size / 1e6:.1f} MB"
    meta = BytesParser().parsebytes(metadata)
    assert meta["Name"] == "rfd4-proteina", meta["Name"]
    assert meta["License-Expression"] == "BSD-3-Clause"
    assert meta["License-File"] == "LICENSE"
    assert meta["Description-Content-Type"] == "text/markdown"
    for dependency in meta.get_all("Requires-Dist", []):
        assert Requirement(dependency).url is None, f"PyPI cannot accept a direct URL dependency: {dependency}"
    print(f"Validated {path.name}: {len(names)} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    for archive in args.archives:
        check_distribution(archive)
