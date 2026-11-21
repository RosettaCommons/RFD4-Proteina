"""Add query-matched MSA and substitution-family arms to a frozen input set."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from atomworks.io import parse
from atomworks.io.utils.io_utils import to_cif_file
from loguru import logger

from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS
from rfproteina.conditions_reporting import run_pipeline_validation
from rfproteina.surface_engineering import (
    NOTION_BIASES,
    XTR_BIASES,
    annotate_probabilities,
    complete_probability_profile,
    reweight_probabilities,
)
from rfproteina.utils.parse_presets import ANNOTATED_CIF
from rfproteina.utils.sequence_sampling import _UNIPROT_AA_PERCENT


def extend(target_dir: Path, msa_file: Path, *, family_bias_only: bool = False) -> None:
    """Validate profile provenance and add arms without touching existing inputs."""
    manifest_path = target_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    msa = json.loads(msa_file.read_text())
    if msa["target"] != manifest["target"]:
        raise ValueError("MSA profile belongs to a different target")
    atoms = parse(target_dir / "baseline.cif", config=ANNOTATED_CIF)["assemblies"]["1"][0]
    eligible = {tuple(r["key"]) for r in manifest["residues"] if r["eligible"]}
    profile_keys = [tuple(r["key"]) for r in msa["profiles"]]
    if len(profile_keys) != len(set(profile_keys)) or set(profile_keys) != {
        tuple(r["key"]) for r in manifest["residues"]
    }:
        raise ValueError("MSA profiles do not map exactly to reference residues")
    policies = {
        "msa_native": None,
        "asn_to_asp": ("ASN", "ASP"),
        "gln_to_glu": ("GLN", "GLU"),
        "ser_to_thr": ("SER", "THR"),
        "gly_to_ala": ("GLY", "ALA"),
    }
    if family_bias_only:
        policies = {"msa_completed": None, "msa_xtr": None, "msa_notion": None}
    for name, replacement in policies.items():
        child = atoms.copy()
        selected = []
        if replacement is None:
            for row in msa["profiles"]:
                key = tuple(row["key"])
                if key in eligible:
                    profile = row["probabilities"]
                    if family_bias_only:
                        background = {
                            aa: value / sum(_UNIPROT_AA_PERCENT.values()) for aa, value in _UNIPROT_AA_PERCENT.items()
                        }
                        profile = complete_probability_profile(profile, background)
                        if name != "msa_completed":
                            biases = XTR_BIASES if name == "msa_xtr" else NOTION_BIASES
                            profile = reweight_probabilities(profile, biases, 1.0)
                    annotate_probabilities(child, [key], profile)
                    selected.append(key)
        else:
            source, target = replacement
            selected = [tuple(r["key"]) for r in manifest["residues"] if r["eligible"] and r["name"] == source]
            if not selected:
                logger.warning("{} has no eligible {} sites; omit arm {}", manifest["target"], source, name)
                continue
            annotate_probabilities(child, selected, {target: 0.7, source: 0.1})
        problems = run_pipeline_validation(child)
        if problems:
            raise ValueError(f"{name}: {problems}")
        path = target_dir / f"{name}.cif"
        if path.exists():
            raise FileExistsError(path)
        to_cif_file(child, path, save_standard_annotations=True, chain_disambiguation="chain_iid")
        loaded = parse(path, config=ANNOTATED_CIF)["assemblies"]["1"][0]
        for condition in SEQUENCE_PROBABILITY_CONDITIONS:
            np.testing.assert_allclose(
                np.asarray(condition.annotation(child, default="generate")),
                np.asarray(condition.annotation(loaded, default="generate")),
                equal_nan=True,
            )
        manifest["arms"].append(
            {
                "name": name,
                "input": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "selected": [list(k) for k in selected],
            }
        )
        logger.info("{} / {}: {} selected sites", manifest["target"], name, len(selected))
    if family_bias_only:
        manifest["msa_completion_policy"] = (
            "Unspecified mass uses UniProt background; specified values and zeros retained before tilting"
        )
    manifest["msa"] = {key: value for key, value in msa.items() if key != "profiles"}
    manifest["deferred"].pop("msa_native", None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Extend one target using portable profiles resolved by the XT implementation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("msa_file", type=Path)
    parser.add_argument("--family-bias-only", action="store_true")
    args = parser.parse_args()
    extend(args.target_dir, args.msa_file, family_bias_only=args.family_bias_only)


if __name__ == "__main__":
    main()
