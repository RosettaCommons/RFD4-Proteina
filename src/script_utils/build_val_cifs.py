#!/usr/bin/env python
"""Write post-cutoff PDB interfaces through the training pipeline as annotated CIFs.

python src/script_utils/build_val_cifs.py --n 128   # from repo root, in the container
"""

import os
from pathlib import Path

import fire
import hydra
import pyarrow.parquet as pq
from atomworks.io.utils.io_utils import to_cif_file

from rfproteina.utils.hydra import register_resolvers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CUTOFF = "2024-01-01"  # training cutoff; the holdout is the inverse


def _plinder_leaked_pdb_ids(data_path: str) -> set[str]:
    """pdb_ids plinder trains on that were deposited >= CUTOFF (plinder has no date to filter)."""
    plinder = pq.read_table(
        f"{data_path}/train_datasets/plinder/2026_05_20_plinder_systems.parquet", columns=["pdb_id"]
    )
    pdb = pq.read_table(f"{data_path}/train_datasets/pdb/pdb_interfaces.parquet", columns=["pdb_id", "deposition_date"])
    post = {
        p
        for p, d in zip(pdb.column("pdb_id").to_pylist(), pdb.column("deposition_date").to_pylist(), strict=False)
        if p and d and str(d)[:10] >= CUTOFF
    }
    return {p for p in plinder.column("pdb_id").to_pylist() if p in post}


def main(n: int = 128, out: str | None = None, experiment: str = "ncorley/hero_runs_2026_07_02_lw_2p5_2p5_0") -> None:
    """Write n holdout CIFs.

    Args:
      n: number of holdout structures to write.
      out: output dir (default: $DATA_PATH/val_datasets/pdb_temporal_holdout).
      experiment: experiment whose train PDB set + pipeline to reuse.
    """
    register_resolvers()
    data_path = os.environ["DATA_PATH"]
    out_dir = Path(out or f"{data_path}/val_datasets/pdb_temporal_holdout")
    out_dir.mkdir(parents=True, exist_ok=True)

    with hydra.initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = hydra.compose("training_proteina", overrides=[f"experiment={experiment}"])
    ds_cfg = cfg.datasets.train.pdb_interfaces.dataset  # reuse the training PDB set + pipeline
    date_idx = next(i for i, f in enumerate(ds_cfg.filters) if "deposition_date" in f)  # don't assume position
    ds_cfg.filters[date_idx] = f"deposition_date >= '{CUTOFF}'"  # flip only the date (train uses <)
    ds_cfg.filters.append(f"pdb_id not in {sorted(_plinder_leaked_pdb_ids(data_path))}")  # drop leak
    dataset = hydra.utils.instantiate(ds_cfg)

    written = 0
    for idx in range(len(dataset)):  # skip structures the pipeline rejects (e.g. odd modified residues)
        if written >= n:
            break
        try:
            to_cif_file(dataset[idx]["atom_array"], out_dir / f"holdout_{written:04d}.cif")
            written += 1
        except Exception as e:
            print(f"skip idx {idx}: {type(e).__name__}: {str(e)[:80]}")
    print(f"wrote {written} CIFs to {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
