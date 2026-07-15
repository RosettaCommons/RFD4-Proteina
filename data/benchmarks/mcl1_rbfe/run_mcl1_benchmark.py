#!/usr/bin/env python3
"""MCL1 benchmark: MACE vs FEP (OpenFE) vs Experiment.

Runs all 42 MCL1 ligands through the MACE interaction energy metric,
downloads OpenFE FEP reference predictions, and computes pairwise
correlations (Pearson, Spearman, R², RMSE, MAE).

Usage::

    python run_mcl1_benchmark.py                 # run the benchmark
    python run_mcl1_benchmark.py --resume        # resume partial run
    python run_mcl1_benchmark.py --correlations-only  # recompute from CSV

Environment variables:
    MACE_POLAR_MODEL_PATH  Path to MACE-POLAR model (or ``--mace-model``)
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import os
import sys
import time
import urllib.request
from collections.abc import Generator
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mcl1_benchmark")

SCRIPT_DIR = Path(__file__).resolve().parent
LIGANDS_SDF = SCRIPT_DIR / "ligands.sdf"
PROTEIN_PDB = SCRIPT_DIR / "mcl1_structure.pdb"
RESULTS_CSV = SCRIPT_DIR / "mcl1_benchmark_results.csv"
CORRELATION_CSV = SCRIPT_DIR / "mcl1_correlation_summary.csv"

EV_TO_KCAL = 23.0609

OPENFE_CSV_URL = (
    "https://raw.githubusercontent.com/OpenFreeEnergy/IndustryBenchmarks2024/"
    "main/industry_benchmarks/analysis/processed_results/"
    "combined_pymbar3_calculated_dg_data.csv"
)

CSV_COLUMNS = [
    "ligand_name",
    "is_flip",
    "exp_dg_kcal",
    "fep_dg_kcal",
    "fep_uncertainty_kcal",
    "mmgbsa_kcal",
    "mace_interaction_eV",
    "mace_interaction_kcal",
]


# ---------------------------------------------------------------------------
# Structure preparation
# ---------------------------------------------------------------------------


def load_structure_file(path: str | Path) -> AtomArray:
    """Load a structure from any format via ``atomworks.io.parser.parse``."""
    from atomworks.io.parser import parse

    result = parse(source=path, config="minimal")
    structure = result["asym_unit"]
    if isinstance(structure, AtomArrayStack):
        structure = structure[0]
    return structure


def load_ligands() -> Generator[tuple[str, float, Any], None, None]:
    """Yield ``(name, exp_dg, rdkit_mol)`` for each ligand in the SDF."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(LIGANDS_SDF), removeHs=False)
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name").strip()
        exp_dg = float(mol.GetProp("kcal/mol experimental dG"))
        yield name, exp_dg, mol


def build_complex(protein: AtomArray, mol: Any) -> AtomArray:
    """Combine protein + ligand into an annotated AtomArray (H stripped)."""
    from atomworks.enums import ChainType
    from atomworks.io.tools.rdkit import atom_array_from_rdkit

    lig = atom_array_from_rdkit(mol, remove_hydrogens=False)
    lig.chain_id[:] = "L"
    lig.res_name[:] = "LIG"

    prot = copy.deepcopy(protein)

    # Align annotations so biotite concatenation preserves all fields
    for annot in lig.get_annotation_categories():
        if annot != "coord" and annot not in prot.get_annotation_categories():
            prot.set_annotation(annot, np.zeros(len(prot), dtype=lig.get_annotation(annot).dtype))
    for annot in prot.get_annotation_categories():
        if annot != "coord" and annot not in lig.get_annotation_categories():
            lig.set_annotation(annot, np.zeros(len(lig), dtype=prot.get_annotation(annot).dtype))

    combined = prot + lig
    chain_type = np.full(len(combined), ChainType.POLYPEPTIDE_L, dtype=int)
    chain_type[len(prot) :] = ChainType.NON_POLYMER
    combined.set_annotation("chain_type", chain_type)

    return combined[combined.element != "H"]


# ---------------------------------------------------------------------------
# FEP reference data
# ---------------------------------------------------------------------------


def download_fep_data() -> dict[str, dict[str, float]]:
    """Download OpenFE JACS MCL1 per-ligand FEP predictions."""
    logger.info("Downloading OpenFE FEP reference data...")
    with urllib.request.urlopen(OPENFE_CSV_URL) as resp:
        text = resp.read().decode("utf-8")

    fep: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(StringIO(text)):
        if row["system name"] != "mcl1" or row["system group"] != "jacs_set":
            continue
        name = row["ligand name"].strip().replace("-flip", " flip")
        fep[name] = {
            "fep_dg": float(row["DG (kcal/mol)"]),
            "fep_unc": float(row["uncertainty (kcal/mol)"]),
            "mmgbsa": float(row["DG MMGBSA"]),
        }

    logger.info("  Found FEP data for %d ligands", len(fep))
    return fep


# ---------------------------------------------------------------------------
# MACE benchmark
# ---------------------------------------------------------------------------


def _ensure_on_path(*dirs: Path) -> None:
    """Add directories to ``sys.path`` if absent."""
    for d in dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)


def run_mace_benchmark(
    complexes: list[tuple[str, AtomArray]],
    model_path: str,
    debug: bool = True,
) -> dict[str, dict[str, float]]:
    """Run ``MACEInteractionEnergyMetric`` on each complex."""
    _ensure_on_path(Path(__file__).resolve().parents[2])

    from rfproteina.metrics.physics.mace_interaction_energy_metric import MACEInteractionEnergyMetric

    metric = MACEInteractionEnergyMetric(
        model_path=model_path,
        max_opt_steps=50,
        fmax=0.05,
        device="cuda",
        hydrogen_policy="infer",
        infer_charge="atom_array",
        max_energy_magnitude=None,
        debug=debug,
        debug_dir=str(SCRIPT_DIR / "mace-debug"),
    )
    metric.setup()

    results: dict[str, dict[str, float]] = {}
    try:
        for name, atom_array in complexes:
            logger.info("MACE: processing ligand %s", name)
            t0 = time.time()
            try:
                result = metric.compute(atom_array)
                if result["n_structures"] == 0:
                    logger.warning("MACE: no result for %s", name)
                    continue
                e = result["interaction_energies"][0]
                results[name] = {"interaction_eV": e, "interaction_kcal": e * EV_TO_KCAL}
                logger.info("  %s: %.4f eV (%.2f kcal/mol) [%.1fs]", name, e, e * EV_TO_KCAL, time.time() - t0)
            except Exception as exc:
                logger.error("MACE failed for %s: %s", name, exc)
    finally:
        metric.teardown()
    return results


def write_results_csv(
    ligand_info: list[tuple[str, float]],
    fep_data: dict[str, dict[str, float]],
    mace_results: dict[str, dict[str, float]],
    path: Path,
) -> list[dict[str, Any]]:
    """Write combined results CSV and return the rows."""
    rows: list[dict[str, Any]] = []
    for name, exp_dg in ligand_info:
        fep = fep_data.get(name, {})
        mace = mace_results.get(name, {})
        rows.append(
            {
                "ligand_name": name,
                "is_flip": "flip" in name.lower(),
                "exp_dg_kcal": exp_dg,
                "fep_dg_kcal": fep.get("fep_dg", ""),
                "fep_uncertainty_kcal": fep.get("fep_unc", ""),
                "mmgbsa_kcal": fep.get("mmgbsa", ""),
                "mace_interaction_eV": mace.get("interaction_eV", ""),
                "mace_interaction_kcal": mace.get("interaction_kcal", ""),
            }
        )

    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerows(rows)

    logger.info("Wrote results to %s (%d rows)", path, len(rows))
    return rows


def load_existing_results(path: Path) -> dict[str, dict[str, float]]:
    """Load existing CSV for ``--resume``."""
    mace: dict[str, dict[str, float]] = {}
    if not path.exists():
        return mace

    with open(path) as f:
        for row in csv.DictReader(f):
            name = row["ligand_name"]
            if row.get("mace_interaction_eV"):
                mace[name] = {
                    "interaction_eV": float(row["mace_interaction_eV"]),
                    "interaction_kcal": float(row["mace_interaction_kcal"]),
                }

    logger.info("Loaded %d MACE results from %s", len(mace), path)
    return mace


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def compute_correlations(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Compute Pearson R, Spearman rho, R², RMSE, MAE for paired arrays."""
    from scipy import stats

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return {"n": n, "pearson_r": np.nan, "spearman_rho": np.nan, "r2": np.nan, "rmse": np.nan, "mae": np.nan}

    pearson_r, _ = stats.pearsonr(x, y)
    spearman_rho, _ = stats.spearmanr(x, y)
    ss_res = np.sum((y - x) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    return {
        "n": n,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "mae": float(np.mean(np.abs(x - y))),
    }


def run_correlation_analysis(rows: list[dict[str, Any]], label: str = "all") -> list[dict[str, Any]]:
    """Compute and print pairwise correlations."""
    exp = np.array([r["exp_dg_kcal"] for r in rows], dtype=float)

    def _col(key: str) -> np.ndarray:
        return np.array([float(r[key]) if r[key] != "" else np.nan for r in rows], dtype=float)

    methods = {
        "FEP (OpenFE)": _col("fep_dg_kcal"),
        "MACE": _col("mace_interaction_kcal"),
        "MM/GBSA": _col("mmgbsa_kcal"),
    }

    pairs = [(f"Exp vs {m}", exp, v) for m, v in methods.items()]
    names = list(methods)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((f"{names[i]} vs {names[j]}", methods[names[i]], methods[names[j]]))

    print(f"\n{'=' * 80}\nCorrelation analysis ({label}, {len(rows)} ligands)\n{'=' * 80}")
    print(f"{'Pair':<25s} {'N':>4s} {'Pearson':>8s} {'Spearman':>9s} {'R²':>7s} {'RMSE':>7s} {'MAE':>7s}")
    print("-" * 80)

    summary: list[dict[str, Any]] = []
    for pair_label, a, b in pairs:
        c = compute_correlations(a, b)
        print(
            f"{pair_label:<25s} {c['n']:>4d} {c['pearson_r']:>8.3f} "
            f"{c['spearman_rho']:>9.3f} {c['r2']:>7.3f} {c['rmse']:>7.3f} {c['mae']:>7.3f}"
        )
        summary.append({"subset": label, "pair": pair_label, **c})
    return summary


def _write_correlation_csv(summary_rows: list[dict[str, Any]]) -> None:
    """Save correlation summary to CSV."""
    fields = ["subset", "pair", "n", "pearson_r", "spearman_rho", "r2", "rmse", "mae"]
    with open(CORRELATION_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    logger.info("Wrote correlation summary to %s", CORRELATION_CSV)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="MCL1 benchmark: MACE vs FEP")
    parser.add_argument("--mace-model", default=None, help="Path to MACE-POLAR model")
    parser.add_argument("--resume", action="store_true", help="Resume from existing CSV")
    parser.add_argument("--correlations-only", action="store_true", help="Recompute correlations only")
    args = parser.parse_args()

    mace_model = args.mace_model or os.environ.get("MACE_POLAR_MODEL_PATH") or os.environ.get("MACE_MODEL_PATH")

    if not mace_model:
        parser.error("MACE model path required (--mace-model or MACE_POLAR_MODEL_PATH)")

    fep_data = download_fep_data()

    if args.correlations_only:
        if not RESULTS_CSV.exists():
            parser.error(f"No results CSV at {RESULTS_CSV}")
        rows: list[dict[str, Any]] = []
        with open(RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                row["exp_dg_kcal"] = float(row["exp_dg_kcal"])
                row["is_flip"] = row["is_flip"] == "True"
                rows.append(row)
        corr = run_correlation_analysis(rows, "all")
        corr += run_correlation_analysis([r for r in rows if not r["is_flip"]], "no_flip")
        _write_correlation_csv(corr)
        return

    # Load structures
    protein = load_structure_file(PROTEIN_PDB)
    logger.info("Protein: %d atoms", len(protein))

    ligand_info: list[tuple[str, float]] = []
    complexes: list[tuple[str, AtomArray]] = []
    for name, exp_dg, mol in load_ligands():
        ligand_info.append((name, exp_dg))
        complexes.append((name, build_complex(protein, mol)))
        logger.info(
            "  %-10s  exp_dG=%.2f kcal/mol  %s  (%d atoms)",
            name,
            exp_dg,
            "[FLIP]" if "flip" in name.lower() else "",
            len(complexes[-1][1]),
        )
    logger.info("Prepared %d complexes", len(complexes))

    mace_results: dict[str, dict[str, float]] = {}
    if args.resume:
        mace_results = load_existing_results(RESULTS_CSV)

    remaining = [(n, a) for n, a in complexes if n not in mace_results]
    if remaining:
        mace_results.update(run_mace_benchmark(remaining, mace_model))

    rows = write_results_csv(ligand_info, fep_data, mace_results, RESULTS_CSV)
    corr = run_correlation_analysis(rows, "all")
    corr += run_correlation_analysis([r for r in rows if not r["is_flip"]], "no_flip")
    _write_correlation_csv(corr)


if __name__ == "__main__":
    main()
