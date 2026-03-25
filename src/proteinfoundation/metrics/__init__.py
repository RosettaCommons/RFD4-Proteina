"""Metrics for protein structure evaluation.

Metrics are organized into three categories:

**Refolding Metrics** — Refold the generated structure with an independent structure
prediction model and compare. Slower but provides strong validation of designability.

- ``RF3Metric``: RF3 refolding confidence (pLDDT, ipTM, ipAE, min_ipAE). Differentiable.
- ``AF2Metric``: AlphaFold2 refolding confidence (pLDDT, pTM, pAE). Differentiable.

**Force Field Metrics** — Evaluate physical plausibility using energy functions. These
use either the Rosetta scoring function (via TMol) or MACE, a machine-learning force
field trained on quantum-mechanical calculations.

- ``TMOLMetric``: Interface H-bond and electrostatic energies via TMol (Rosetta scoring function). Differentiable.
- ``SoftDDGMetric``: Binding ddG with soft repulsion via TMol (Rosetta scoring function). Differentiable.
- ``ChargeComplementarityMetric``: Salt bridge scoring across interfaces. Differentiable.
- ``SoftDDGWithSaltMetric``: Combined soft ddG + charge complementarity. Differentiable.
- ``MACEEnergyMetric``: Potential energy evaluation using MACE (ML force field). Differentiable.

**Structure Quality Metrics** — Fast geometric and chemical validation of the generated
structure, no refolding or energy computation required.

- ``PoseBustersMetric``: Stereochemical validation of ligand molecules.
- ``AtomArrayMetrics``: Secondary structure, radius of gyration, composition.
- ``BackboneMetrics``: Chain breaks, clashes, floating atoms.
- ``SidechainMetrics``: Sidechain validity.
- ``SSEMetrics``: Secondary structure element fractions.
- ``RASAMetrics``: Relative accessible surface area.
- ``MolprobityMetrics``: Clashscore, Ramachandran, rotamer outliers.
- ``HbondMetrics``: Hydrogen bond counts and energies.
"""

from proteinfoundation.metrics.rf3_metric import RF3Metric

# Optional imports that require additional dependencies
__all__ = ["RF3Metric"]

# --- Refolding Metrics ---
try:
    from proteinfoundation.metrics.af2_metric import AF2Metric

    __all__.append("AF2Metric")
except ImportError:
    pass

# --- Force Field Metrics ---
try:
    from proteinfoundation.metrics.tmol_metric import TMOLMetric

    __all__.append("TMOLMetric")
except ImportError:
    pass

try:
    from proteinfoundation.metrics.ddg_salt_metrics import (
        ChargeComplementarityMetric,
        SoftDDGMetric,
        SoftDDGWithSaltMetric,
    )

    __all__.extend(["ChargeComplementarityMetric", "SoftDDGMetric", "SoftDDGWithSaltMetric"])
except ImportError:
    pass

try:
    from proteinfoundation.metrics.mace_energy_metric import MACEEnergyMetric, compute_mace_energy

    __all__.extend(["MACEEnergyMetric", "compute_mace_energy"])
except ImportError:
    pass

# --- Structure Quality Metrics ---
try:
    from proteinfoundation.metrics.posebusters_metrics import PoseBustersMetric

    __all__.append("PoseBustersMetric")
except ImportError:
    pass
