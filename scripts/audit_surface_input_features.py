"""Verify the lipase surface mask and formal charge in actual RFD4 input tensors.

This CPU-only boundary check runs the full native input featurizer. It catches
CCD-only charges that disappear before the model, without loading model weights.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from atomworks.ml.utils.token import get_token_starts
from hydra import compose, initialize_config_dir
from loguru import logger

from rfproteina.generate import get_dataset_from_cfg
from rfproteina.surface_engineering import atom_residue_keys


def main() -> None:
    """Write a self-checking feature receipt and fail on either boundary mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.prepared = args.prepared.resolve()
    # Any is restricted to the native heterogeneous dataset/JSON boundary;
    # identities, token indices and actual tensor values are checked below.
    results: list[dict[str, Any]] = []
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        for target in ["tll", "bacillus"]:
            path = args.prepared / target / "baseline.cif"
            row: dict[str, Any] = {
                "target": target,
                "input": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            try:
                cfg = compose(config_name="inference", overrides=[f"inputs={path}", "seed=20261120", "~metrics"])
                sample = get_dataset_from_cfg(cfg, 0)[0]
                atoms = sample["atom_array"]
                starts = get_token_starts(atoms)
                keys = atom_residue_keys(atoms)
                tokens = [keys[int(i)] for i in starts]
                manifest = json.loads((args.prepared / target / "manifest.json").read_text())
                expected = {tuple(residue["key"]) for residue in manifest["residues"] if residue["eligible"]}
                indices = manifest["eligible_token_indices"]
                selected = {tokens[i] for i in indices}
                if not expected or selected != expected or len(selected) != len(indices):
                    raise ValueError("Selected residue keys do not map uniquely to native tokens")
                oa = [
                    i
                    for i, (name, residue) in enumerate(zip(atoms.atom_name, atoms.res_name, strict=True))
                    if name == "OA" and residue == "ZMT"
                ]
                if len(oa) != 1 or int(atoms.charge[oa[0]]) != -1:
                    raise ValueError("Native parsed OA formal charge is not -1")
                oa_tokens = [i for i, start in enumerate(starts) if int(start) == oa[0]]
                if len(oa_tokens) != 1:
                    raise ValueError("OA does not map uniquely to its atomized native token")
                charges = sample["feats"]["charge"]
                if not isinstance(charges, torch.Tensor) or charges.ndim != 2:
                    raise ValueError("Unexpected native charge feature tensor")
                feature = charges[oa_tokens[0]]
                if (
                    not bool(torch.isfinite(charges).all())
                    or int(torch.count_nonzero(feature)) != 1
                    or float(feature.sum()) != -1.0
                ):
                    raise ValueError("Actual model OA charge feature does not encode -1")
                row.update(
                    passed=True,
                    n_selected=len(selected),
                    n_tokens=len(tokens),
                    oa_token=oa_tokens[0],
                    oa_model_charge=float(feature.sum()),
                    charge_feature_nonzero=charges.nonzero().tolist(),
                )
            except (ValueError, AssertionError) as exc:
                row.update(passed=False, error=str(exc))
            results.append(row)
            logger.info("{} input feature guard: {}", target, row["passed"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "passed": all(row["passed"] for row in results),
                "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "targets": results,
            },
            indent=2,
        )
        + "\n"
    )
    if not all(row["passed"] for row in results):
        raise RuntimeError("Native input feature boundary failed; do not generate this cohort")


if __name__ == "__main__":
    main()
