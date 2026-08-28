"""Developability filters: GRAVY, instability index, SAP, and exposed hydrophobics.

Sequence metrics (GRAVY, Guruprasad instability index) are computed in-process from
the designed-chain sequence. Structure metrics (Rosetta SAP score and
ExposedHydrophobics) run via a nested PyRosetta apptainer, following the same pattern
as HBDesigner PyRosetta scoring.
"""

import json
import os
import shutil
import subprocess
import tempfile
from functools import cached_property
from pathlib import Path
from typing import Any

from atomworks.constants import STANDARD_AA_ONE_LETTER
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.sequence import aa_chem_comp_3to1
from biotite.structure import AtomArray

from rfproteina.metrics.metric import Metric
from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.structure_export import write_input_structure_cif

logger = RankedLogger(__name__, rank_zero_only=True)

_STANDARD_AA = frozenset(STANDARD_AA_ONE_LETTER)
_AA3_TO_1 = aa_chem_comp_3to1(standard_only=True)

# fmt: off
# Kyte-Doolittle hydropathy (ExPASy / BioPython ProtParam).
_KYTE_DOOLITTLE: dict[str, float] = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}  # noqa: E501

# Guruprasad DIWV dipeptide weights (ExPASy / BioPython ProtParamData.DIWV).
_DIWV: dict[str, dict[str, float]] = {
    "A": {"A": 1.0, "C": 44.94, "E": 1.0, "D": -7.49, "G": 1.0, "F": 1.0, "I": 1.0, "H": -7.49, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": 1.0, "Y": 1.0},  # noqa: E501
    "C": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 20.26, "G": 1.0, "F": 1.0, "I": 1.0, "H": 33.6, "K": 1.0, "M": 33.6, "L": 20.26, "N": 1.0, "Q": -6.54, "P": 20.26, "S": 1.0, "R": 1.0, "T": 33.6, "W": 24.68, "V": -6.54, "Y": 1.0},  # noqa: E501
    "E": {"A": 1.0, "C": 44.94, "E": 33.6, "D": 20.26, "G": 1.0, "F": 1.0, "I": 20.26, "H": -6.54, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 20.26, "R": 1.0, "T": 1.0, "W": -14.03, "V": 1.0, "Y": 1.0},  # noqa: E501
    "D": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": -6.54, "I": 1.0, "H": 1.0, "K": -7.49, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 1.0, "S": 20.26, "R": -6.54, "T": -14.03, "W": 1.0, "V": 1.0, "Y": 1.0},  # noqa: E501
    "G": {"A": -7.49, "C": 1.0, "E": -6.54, "D": 1.0, "G": 13.34, "F": 1.0, "I": -7.49, "H": 1.0, "K": -7.49, "M": 1.0, "L": 1.0, "N": -7.49, "Q": 1.0, "P": 1.0, "S": 1.0, "R": 1.0, "T": -7.49, "W": 13.34, "V": 1.0, "Y": -7.49},  # noqa: E501
    "F": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 13.34, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": -14.03, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": 1.0, "Y": 33.601},  # noqa: E501
    "I": {"A": 1.0, "C": 1.0, "E": 44.94, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 13.34, "K": -7.49, "M": 1.0, "L": 20.26, "N": 1.0, "Q": 1.0, "P": -1.88, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": -7.49, "Y": 1.0},  # noqa: E501
    "H": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -9.37, "F": -9.37, "I": 44.94, "H": 1.0, "K": 24.68, "M": 1.0, "L": 1.0, "N": 24.68, "Q": 1.0, "P": -1.88, "S": 1.0, "R": 1.0, "T": -6.54, "W": -1.88, "V": 1.0, "Y": 44.94},  # noqa: E501
    "K": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -7.49, "F": 1.0, "I": -7.49, "H": 1.0, "K": 1.0, "M": 33.6, "L": -7.49, "N": 1.0, "Q": 24.64, "P": -6.54, "S": 1.0, "R": 33.6, "T": 1.0, "W": 1.0, "V": -7.49, "Y": 1.0},  # noqa: E501
    "M": {"A": 13.34, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 58.28, "K": 1.0, "M": -1.88, "L": 1.0, "N": 1.0, "Q": -6.54, "P": 44.94, "S": 44.94, "R": -6.54, "T": -1.88, "W": 1.0, "V": 1.0, "Y": 24.68},  # noqa: E501
    "L": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": -7.49, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 33.6, "P": 20.26, "S": 1.0, "R": 20.26, "T": 1.0, "W": 24.68, "V": 1.0, "Y": 1.0},  # noqa: E501
    "N": {"A": 1.0, "C": -1.88, "E": 1.0, "D": 1.0, "G": -14.03, "F": -14.03, "I": 44.94, "H": 1.0, "K": 24.68, "M": 1.0, "L": 1.0, "N": 1.0, "Q": -6.54, "P": -1.88, "S": 1.0, "R": 1.0, "T": -7.49, "W": -9.37, "V": 1.0, "Y": 1.0},  # noqa: E501
    "Q": {"A": 1.0, "C": -6.54, "E": 20.26, "D": 20.26, "G": 1.0, "F": -6.54, "I": 1.0, "H": 1.0, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 44.94, "R": 1.0, "T": 1.0, "W": 1.0, "V": -6.54, "Y": -6.54},  # noqa: E501
    "P": {"A": 20.26, "C": -6.54, "E": 18.38, "D": -6.54, "G": 1.0, "F": 20.26, "I": 1.0, "H": 1.0, "K": 1.0, "M": -6.54, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 20.26, "R": -6.54, "T": 1.0, "W": -1.88, "V": 20.26, "Y": 1.0},  # noqa: E501
    "S": {"A": 1.0, "C": 33.6, "E": 20.26, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 44.94, "S": 20.26, "R": 20.26, "T": 1.0, "W": 1.0, "V": 1.0, "Y": 1.0},  # noqa: E501
    "R": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -7.49, "F": 1.0, "I": 1.0, "H": 20.26, "K": 1.0, "M": 1.0, "L": 1.0, "N": 13.34, "Q": 20.26, "P": 20.26, "S": 44.94, "R": 58.28, "T": 1.0, "W": 58.28, "V": 1.0, "Y": -6.54},  # noqa: E501
    "T": {"A": 1.0, "C": 1.0, "E": 20.26, "D": 1.0, "G": -7.49, "F": 13.34, "I": 1.0, "H": 1.0, "K": 1.0, "M": 1.0, "L": 1.0, "N": -14.03, "Q": -6.54, "P": 1.0, "S": 1.0, "R": 1.0, "T": 1.0, "W": -14.03, "V": 1.0, "Y": 1.0},  # noqa: E501
    "W": {"A": -14.03, "C": 1.0, "E": 1.0, "D": 1.0, "G": -9.37, "F": 1.0, "I": 1.0, "H": 24.68, "K": 1.0, "M": 24.68, "L": 13.34, "N": 13.34, "Q": 1.0, "P": 1.0, "S": 1.0, "R": 1.0, "T": -14.03, "W": 1.0, "V": -7.49, "Y": 1.0},  # noqa: E501
    "V": {"A": 1.0, "C": 1.0, "E": 1.0, "D": -14.03, "G": -7.49, "F": 1.0, "I": 1.0, "H": 1.0, "K": -1.88, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": -7.49, "W": 1.0, "V": 1.0, "Y": -6.54},  # noqa: E501
    "Y": {"A": 24.68, "C": 1.0, "E": -6.54, "D": 24.68, "G": -7.49, "F": 1.0, "I": 1.0, "H": 13.34, "K": 1.0, "M": 44.94, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 13.34, "S": 1.0, "R": -15.91, "T": -7.49, "W": -9.37, "V": 1.0, "Y": 13.34},  # noqa: E501
}
# fmt: on


def _standard_one_letter(sequence: str) -> str:
    """Uppercase/strip a one-letter sequence, raising on non-standard residues."""
    normalized = sequence.strip().upper()
    unknown = set(normalized) - _STANDARD_AA
    if unknown:
        raise ValueError(f"Sequence contains non-standard residues: {sorted(unknown)}")
    return normalized


def compute_gravy(sequence: str) -> float:
    """Grand average of hydropathicity (Kyte-Doolittle).

    Args:
        sequence: One-letter amino-acid sequence.

    Returns:
        GRAVY value (sum of hydropathy / length).

    Raises:
        ValueError: If the sequence contains non-standard residues.
    """
    sequence = _standard_one_letter(sequence)
    if not sequence:
        return float("nan")
    return float(sum(_KYTE_DOOLITTLE[aa] for aa in sequence) / len(sequence))


def compute_instability_index(sequence: str) -> float:
    """Guruprasad instability index (ProtParam). Values < 40 predict stability.

    Args:
        sequence: One-letter amino-acid sequence.

    Returns:
        Instability index. ``nan`` if sequence length < 2.

    Raises:
        ValueError: If the sequence contains non-standard residues.
    """
    sequence = _standard_one_letter(sequence)
    if len(sequence) < 2:
        return float("nan")
    score = 0.0
    for i in range(len(sequence) - 1):
        score += _DIWV[sequence[i]][sequence[i + 1]]
    return float((10.0 / len(sequence)) * score)


def _chain_attr(atom_array: AtomArray) -> str:
    return "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"


def _resolve_chain_of_interest(atom_array: AtomArray, chain_of_interest: str | None) -> str | None:
    if chain_of_interest is not None:
        return chain_of_interest
    coi, _ = get_chain_of_interest(atom_array)
    return coi


def subset_to_designed_protein(
    atom_array: AtomArray,
    chain_of_interest: str | None = None,
) -> AtomArray:
    """Return protein atoms on the designed chain (chain of interest).

    Args:
        atom_array: Full generated structure.
        chain_of_interest: Optional chain id/iid; inferred if omitted.

    Returns:
        Protein-only AtomArray for the designed chain. Empty if none found.
    """
    if atom_array.array_length() == 0:
        return atom_array
    ensure_annotations(atom_array, "is_protein")
    protein = atom_array[atom_array.is_protein]
    if protein.array_length() == 0:
        return protein
    coi = _resolve_chain_of_interest(protein, chain_of_interest)
    if coi is None:
        return protein
    attr = _chain_attr(protein)
    return protein[protein.get_annotation(attr) == coi]


def protein_one_letter_sequence(atom_array: AtomArray) -> str:
    """Extract the one-letter sequence, skipping non-standard residues."""
    if atom_array.array_length() == 0:
        return ""
    res_names = atom_array.res_name[get_residue_starts(atom_array)]
    return "".join(_AA3_TO_1[name] for name in res_names if name in _AA3_TO_1)


class SequenceDevelopabilityMetric(Metric):
    """Sequence-only developability scores for the designed chain.

    Reports:
        - ``gravy``: Kyte-Doolittle grand average of hydropathicity
        - ``instability_index``: Guruprasad instability index (ProtParam; <40 ~ stable)
    """

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array": "predicted_atom_array",
            "chain_of_interest": ("extra_info", "chain_of_interest"),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset(["chain_of_interest"])

    def compute(
        self,
        predicted_atom_array: AtomArray,
        chain_of_interest: str | None = None,
    ) -> dict[str, float]:
        designed = subset_to_designed_protein(predicted_atom_array, chain_of_interest)
        sequence = protein_one_letter_sequence(designed)
        if not sequence:
            return {"gravy": float("nan"), "instability_index": float("nan")}
        return {
            "gravy": compute_gravy(sequence),
            "instability_index": compute_instability_index(sequence),
        }


class RosettaDevelopabilityMetric(Metric):
    """Rosetta SAP score and ExposedHydrophobics via a nested PyRosetta apptainer.

    Reports:
        - ``sap_score``: whole-pose SAP (``calculate_sap``)
        - ``hydrophobic_exposure_sasa``: Rosetta ExposedHydrophobics filter score
          (sum of ``max(0, SASA - sasa_cutoff)`` over hydrophobic residues)

    Both are computed on the designed protein chain only.
    """

    def __init__(
        self,
        container: str | None = None,
        score_script: str | None = None,
        apptainer_bin: str | None = None,
        sasa_cutoff: float = 20.0,
        timeout_s: float = 180.0,
        binds: list[str] | None = None,
        scratch_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        container = container or os.environ.get("PYROSETTA_CONTAINER")
        if container is None:
            raise OSError("PyRosetta container not set: pass container=... or set $PYROSETTA_CONTAINER.")
        self.container = Path(container)
        default_script = Path(__file__).resolve().parents[4] / "scripts" / "developability" / "score_pose_rosetta.py"
        self.score_script = Path(score_script or default_script)
        self.apptainer_bin = apptainer_bin or shutil.which("apptainer") or "apptainer"
        self.sasa_cutoff = float(sasa_cutoff)
        self.timeout_s = float(timeout_s)
        # Extra binds come from the project-wide APPTAINER_BIND_PATHS (set per-cluster in
        # .ipd/.env.<cluster>) -- no cluster paths are hardcoded here.
        if binds is None:
            raw = os.environ.get("APPTAINER_BIND_PATHS", "")
            binds = [spec.strip() for spec in raw.split(";") if spec.strip()]
        self.binds = list(binds)
        self.scratch_dir = Path(
            scratch_dir
            or os.environ.get(
                "DEVELOPABILITY_METRIC_SCRATCH",
                os.environ.get("TMPDIR", "/tmp"),
            )
        )
        # The CIF/JSON handoff lives under scratch_dir, so it must be visible inside the container.
        if str(self.scratch_dir) not in self.binds:
            self.binds.append(str(self.scratch_dir))
        super().__init__(**kwargs)

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array": "predicted_atom_array",
            "chain_of_interest": ("extra_info", "chain_of_interest"),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset(["chain_of_interest"])

    def compute(
        self,
        predicted_atom_array: AtomArray,
        chain_of_interest: str | None = None,
    ) -> dict[str, float]:
        if not self.container.is_file():
            raise FileNotFoundError(f"PyRosetta container missing: {self.container}")
        if not self.score_script.is_file():
            raise FileNotFoundError(f"Rosetta score script missing: {self.score_script}")

        designed = subset_to_designed_protein(predicted_atom_array, chain_of_interest)
        if designed.array_length() == 0:
            return {"sap_score": float("nan"), "hydrophobic_exposure_sasa": float("nan")}

        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rfd4_developability_", dir=self.scratch_dir) as tmp:
            cif_path = Path(tmp) / "design.cif"
            out_json = Path(tmp) / "score.json"
            write_input_structure_cif(designed, cif_path)
            command = [
                self.apptainer_bin,
                "exec",
                *[arg for bind in self.binds for arg in ("--bind", bind)],
                str(self.container),
                "python",
                str(self.score_script),
                "--cif",
                str(cif_path),
                "--sasa_cutoff",
                str(self.sasa_cutoff),
                "--output_json",
                str(out_json),
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"Rosetta developability score timed out after {self.timeout_s}s; returning nan.")
                return {"sap_score": float("nan"), "hydrophobic_exposure_sasa": float("nan")}
            if out_json.is_file():
                payload = json.loads(out_json.read_text(encoding="utf-8"))
            else:
                try:
                    payload = json.loads(completed.stdout.strip().splitlines()[-1])
                except Exception as exc:
                    raise RuntimeError(
                        "Rosetta developability subprocess failed without JSON output: "
                        f"returncode={completed.returncode} stderr={completed.stderr[-2000:]}"
                    ) from exc
            if completed.returncode != 0 or "error" in payload:
                raise RuntimeError(
                    f"Rosetta developability score failed (rc={completed.returncode}): "
                    f"{payload.get('error') or completed.stderr[-2000:]}"
                )
            return {
                "sap_score": float(payload["sap_score"]),
                "hydrophobic_exposure_sasa": float(payload["hydrophobic_exposure_sasa"]),
            }
