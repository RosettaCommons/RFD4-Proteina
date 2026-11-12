"""Prepare a matched PD-L1 target and fixed-length binder spec for both models."""

import argparse
from pathlib import Path

import numpy as np
import yaml
from atomworks.io import parse
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import filter_amino_acids
from biotite.structure.io.pdb import PDBFile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = root / "docs/examples/inputs/5o45_cropped.cif.zst"
    target = parse(source)["assemblies"]["1"][0]
    target = target[filter_amino_acids(target) & (target.chain_id == "A")]
    if not np.array_equal(np.unique(target.res_id), np.arange(17, 132)):
        raise ValueError("Expected the checked-in PD-L1 A17-131 target; inspect the source before adapting.")
    resolved = np.isfinite(target.coord).all(axis=-1)
    backbone = np.isin(target.atom_name, ["N", "CA", "C", "O"])
    if not resolved[backbone].all():
        raise ValueError("The target contains unresolved backbone coordinates.")
    # PDB cannot carry NaN placeholders. Do not reconstruct missing atoms; both
    # predictors read the same resolved structure, with all target residues retained.
    print(f"Exporting resolved atoms; omitting {int((~resolved).sum())} unresolved non-backbone placeholders.")
    target = target[resolved]
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pdb_path, cif_path, spec_path = out / "target.pdb", out / "target.cif", out / "binder.yaml"
    if any(path.exists() for path in (pdb_path, cif_path, spec_path)):
        raise FileExistsError("Choose a fresh directory; existing example inputs are preserved.")
    pdb = PDBFile()
    pdb.set_structure(target)
    pdb.write(pdb_path)
    # RFD4 requires CIF: retain the source CIF's bond and charge annotations.
    to_cif_file(target, cif_path, chain_disambiguation="transformation_id")
    spec = {
        "pdl1_binder": {
            "input": str(cif_path),
            "contig": "60, /0, A17-131",
            "conditions": {
                "C_SEQ": {True: [{"select": "sel('A')"}]},
                "C_CRDGRP": {1: [{"select": "sel('A/*/*/[N,CA,C,O]')"}]},
            },
        }
    }
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f"RFD4 input: {spec_path}\nColabDesign target: {pdb_path}")


if __name__ == "__main__":
    main()
