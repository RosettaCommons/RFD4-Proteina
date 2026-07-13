"""Metrics for protein structure evaluation.

Metrics are organized into three categories:

**Refolding Oracles** — Refold the generated structure with an independent structure
prediction model and compare. Slower but provides strong validation of designability.

- ``RF3Oracle``: RF3 folding oracle (confidence scores + refolded CIF).
- ``AF3Oracle``: AlphaFold 3 folding oracle (one-shot subprocess, not differentiable).
- ``AF2Metric``: AlphaFold2 refolding confidence (pLDDT, pTM, pAE). Differentiable.

**Force Field Metrics** — Evaluate physical plausibility using energy functions. These
use either the Rosetta scoring function (via TMol) or MACE, a machine-learning force
field trained on quantum-mechanical calculations.

- ``TMolBlockPairMetric``: General region-pair TMol interface metric (ddG, or per-term
  energy/count). Binder/target ddG, ligand ddG, PPI H-bonds, electrostatics, … are
  expressed as Hydra configs over this one class; see
  ``configs/metrics/inline_metrics/tmol_block_pair_metrics.yaml``.
- ``ChargeComplementarityMetric``: Salt bridge scoring across interfaces. Differentiable.
- ``MACEEnergyMetric``: Potential energy evaluation using MACE (ML force field). Differentiable.
- ``QMInteractionEnergyIPA``: Pairwise ligand-residue interaction energies
  via GFN2-xTB (In-Pocket Analysis). Requires ``IPA_PATH``.

**FEP Metrics** — Alchemical free energy perturbation for binding affinity prediction.
Require TMD (https://github.com/tmd-industries/tmd) with CUDA custom ops.

- ``FEPMetric``: Single-edge RBFE between two ligands in a protein pocket.
- ``FEPBatchMetric``: Batch RBFE with MPS parallelism for higher throughput.

**Structure Quality Metrics** — Fast geometric and chemical validation of the generated
structure, no refolding or energy computation required.

- ``PoseBustersMetric``: Stereochemical validation of ligand molecules.
- ``AtomPairDistanceMetric``: Atom-pair distance constraints with soft reward scores.
- ``BondLengthMetric``: Unreasonable bond length detection.
- ``ClashMetric``: Fraction of atoms in a steric clash, by clash type, at a configurable threshold.
- ``SSEMetrics``: Secondary structure element fractions.
- ``RASAMetrics``: Relative accessible surface area.
- ``MolprobityMetrics``: Clashscore, Ramachandran, rotamer outliers.
- ``HbondMetrics``: Hydrogen bond counts and energies.
- ``NucleicSSSimilarityMetrics``: nucleic acid base pair conditioning adherence.
"""

from proteinfoundation.metrics.af3_oracle import AF3Oracle
from proteinfoundation.metrics.oracle_metric import OracleMetric
from proteinfoundation.metrics.rf3_oracle import RF3Oracle

# Optional imports that require additional dependencies
__all__ = ["AF3Oracle", "OracleMetric", "RF3Oracle"]

# --- Refolding Metrics ---
try:
    from proteinfoundation.metrics.af2_metric import AF2Metric

    __all__.append("AF2Metric")
except ImportError:
    pass

# --- Force Field Metrics ---
try:
    from proteinfoundation.metrics.atom_pair_distance_metric import AtomPairDistanceMetric

    __all__.append("AtomPairDistanceMetric")
    from proteinfoundation.metrics.tmol_block_pair_metric import TMolBlockPairMetric

    __all__.append("TMolBlockPairMetric")
except ImportError:
    pass

try:
    from proteinfoundation.metrics.ddg_salt_metrics import ChargeComplementarityMetric

    __all__.append("ChargeComplementarityMetric")
except ImportError:
    pass

try:
    from proteinfoundation.metrics.mace_energy_metric import MACEEnergyMetric, compute_mace_energy

    __all__.extend(["MACEEnergyMetric", "compute_mace_energy"])
except ImportError:
    pass

try:
    from proteinfoundation.metrics.ipa_interaction_energy_metric import (
        QMInteractionEnergyIPA,
        compute_ipa_interaction_energy,
    )

    __all__.extend(["QMInteractionEnergyIPA", "compute_ipa_interaction_energy"])
except ImportError:
    pass

# --- FEP Metrics ---
try:
    from proteinfoundation.metrics.fep_metric import FEPBatchMetric, FEPMetric

    __all__.extend(["FEPBatchMetric", "FEPMetric"])
except ImportError:
    pass

# --- Structure Quality Metrics ---
try:
    from proteinfoundation.metrics.posebusters_metrics import PoseBustersMetric

    __all__.append("PoseBustersMetric")
except ImportError:
    pass

# --- Conditioning Metrics ---
try:
    from proteinfoundation.metrics.nucleic_ss_metrics import NucleicSSSimilarityMetrics

    __all__.append("NucleicSSSimilarityMetrics")
except ImportError:
    pass
