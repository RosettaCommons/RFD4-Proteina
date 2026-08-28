#!/usr/bin/env python3
"""Score one design CIF with Rosetta SAP + ExposedHydrophobics.

Runs inside a PyRosetta apptainer, which ships only pyrosetta + the stdlib, so
this stays argparse-only (no third-party CLI deps). Writes a JSON object to
``--output_json`` and echoes it to stdout.
"""

import argparse
import json
from pathlib import Path

import pyrosetta
from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import calculate_sap
from pyrosetta.rosetta.core.select.residue_selector import TrueResidueSelector
from pyrosetta.rosetta.protocols.denovo_design.filters import ExposedHydrophobicsFilter


def _score(cif: Path, sasa_cutoff: float) -> dict[str, float]:
    if not pyrosetta.rosetta.basic.was_init_called():
        pyrosetta.init("-mute all -ignore_unrecognized_res true -ignore_zero_occupancy false")

    pose = pyrosetta.pose_from_file(str(cif))
    sel = TrueResidueSelector()
    exposed = ExposedHydrophobicsFilter()
    exposed.set_sasa_cutoff(sasa_cutoff)
    return {
        "sap_score": float(calculate_sap(pose, sel, sel, sel)),
        "hydrophobic_exposure_sasa": float(exposed.compute(pose)),
        "n_residues": int(pose.total_residue()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif", required=True, type=Path)
    parser.add_argument("--sasa_cutoff", type=float, default=20.0)
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    try:
        payload = _score(args.cif, args.sasa_cutoff)
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    text = json.dumps(payload)
    if args.output_json is not None:
        args.output_json.write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
