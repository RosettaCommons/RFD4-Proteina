"""CATH superfamily annotation metric using Foldseek structural search.

Required environment variables:
- FOLDSEEK_EXEC: path to foldseek binary
- CATH_FOLDSEEK_DB: path to CATH SSG95 foldseek database prefix (cath_v4_4_0_s95_db)
- CATH_DOMAIN_LIST: path to CathDomainList.S95.v4.4.0 file
"""

import os
import subprocess
import tempfile
from functools import cache
from typing import Any

import numpy as np
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray, AtomArrayStack

from proteinfoundation.metrics import Metric

# Classification thresholds
_H_EVALUE = 0.061144
_H_COV = 0.459063
_H_TM = 0.634640
_T_EVALUE = 0.476641
_T_COV = 0.741603
_T_TM = 0.513274

_CONVERTALIS_COLS = "query,target,fident,evalue,qlen,tlen,qtmscore,ttmscore,qcov,tcov"
_SEARCH_FLAGS = ["--cov-mode", "5", "--alignment-type", "2", "-e", "0.476641", "-s", "10", "-c", "0.459063", "-a"]

_NO_HIT: dict[str, Any] = {"C": None, "A": None, "T": None, "H": None, "type": None, "evalue": None, "tmscore": None}


@cache
def _parse_cath_domain_list(cath_list_file: str) -> dict[str, str]:
    """Parse CathDomainList file into {domain_id: sfam_id} mapping.

    Cached by file path — called once per unique path per process.

    Args:
      cath_list_file: Path to CathDomainList.S95.v4.4.0 file.

    Returns:
      Dict mapping domain IDs to CATH superfamily strings (e.g. ``'3.40.50.300'``).
    """
    domain_lookup: dict[str, str] = {}
    with open(cath_list_file) as fh:
        for line in fh:
            parts = line.split()
            domain_id = parts[0]
            sfam_id = f"{parts[1]}.{parts[2]}.{parts[3]}.{parts[4]}"
            domain_lookup[domain_id] = sfam_id
    return domain_lookup


def _hit_type(evalue: float, coverage: float, tmscore: float) -> str | None:
    """Classify a Foldseek hit as H (homologous), T (topology), N (neighbor), or None."""
    if evalue < _H_EVALUE and coverage > _H_COV and tmscore > _H_TM:
        return "H"
    if evalue < _T_EVALUE and coverage > _T_COV and tmscore > _T_TM:
        return "T"
    if tmscore > _T_TM:
        return "N"
    return None


def _parse_foldseek_m8(result_file: str, domain_lookup: dict[str, str]) -> dict[str, Any] | None:
    """Parse foldseek convertalis m8 output and return the best hit.

    Args:
      result_file: Path to convertalis output file.
      domain_lookup: Mapping from domain ID to CATH superfamily string.

    Returns:
      Best-hit dict with keys ``code``, ``type``, ``evalue``, ``tmscore``,
      ``qcov``, ``tcov``; or ``None`` if no qualifying hits.
    """
    hits: list[dict[str, Any]] = []
    with open(result_file) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 10:
                continue
            target_id = parts[1]
            evalue = float(parts[3])
            qlen = int(float(parts[4]))
            tlen = int(float(parts[5]))
            qtmscore = float(parts[6])
            ttmscore = float(parts[7])
            qcov = float(parts[8])
            tcov = float(parts[9])

            if target_id not in domain_lookup:
                continue
            sfam_id = domain_lookup[target_id]
            coverage = min(qlen, tlen) / max(qlen, tlen)
            tmscore = max(qtmscore, ttmscore)
            htype = _hit_type(evalue, coverage, tmscore)

            if htype in ["H", "T", "N"]:
                code: str | None = sfam_id
            else:
                code = None

            hits.append(
                {
                    "code": code,
                    "type": htype,
                    "evalue": evalue,
                    "tmscore": tmscore,
                    "qcov": qcov,
                    "tcov": tcov,
                }
            )

    if not hits:
        return None

    h_hits = sorted([h for h in hits if h["type"] == "H"], key=lambda x: (x["evalue"], -x["tmscore"]))
    t_hits = sorted([h for h in hits if h["type"] == "T"], key=lambda x: (-x["tmscore"], x["evalue"]))
    n_hits = sorted([h for h in hits if h["type"] == "N"], key=lambda x: (-x["tmscore"], x["evalue"]))

    for group in (h_hits, t_hits, n_hits):
        if group:
            return group[0]
    return None


def _cath_levels(code: str | None) -> dict[str, str | None]:
    """Return cumulative CATH level strings given the best-hit code.

    Each level value is the full prefix up to that level, e.g. ``'1.20.59.10'``
    for H, ``'1.20.59'`` for T, ``'1.20'`` for A, ``'1'`` for C.
    """
    if code is None:
        return {"C": None, "A": None, "T": None, "H": None}
    parts = code.split(".")
    prefixes = [".".join(parts[:i]) for i in range(1, 5)]
    return {
        "C": prefixes[0] if len(parts) > 0 else None,
        "A": prefixes[1] if len(parts) > 1 else None,
        "T": prefixes[2] if len(parts) > 2 else None,
        "H": prefixes[3] if len(parts) > 3 else None,
    }


def annotate_cath(atom_array: AtomArray) -> dict[str, Any]:
    """Compute CATH superfamily annotation for a single-chain AtomArray via Foldseek.

    Runs the full Foldseek createdb -> search -> convertalis pipeline against the
    CATH SSG95 database, then classifies the best hit using thresholds from the
    domain-annotation-pipeline.

    Args:
      atom_array: Single-chain protein structure (chain of interest, amino acids only).

    Returns:
      Dict with keys:

        - C: CATH class prefix (e.g. ``'1'``) or ``None``
        - A: CATH architecture prefix (e.g. ``'1.20'``) or ``None``
        - T: CATH topology prefix (e.g. ``'1.20.59'``) or ``None``
        - H: CATH homologous superfamily (e.g. ``'1.20.59.10'``) or ``None``
        - type: ``'H'`` (homologous), ``'T'`` (topology), ``'N'`` (neighbor), or ``None``
        - evalue: E-value of the best Foldseek hit, or ``None``
        - tmscore: TM-score of the best Foldseek hit, or ``None``

    Raises:
      RuntimeError: If FOLDSEEK_EXEC, CATH_FOLDSEEK_DB, or CATH_DOMAIN_LIST are unset.
    """
    foldseek_exec = os.getenv("FOLDSEEK_EXEC")
    cath_db = os.getenv("CATH_FOLDSEEK_DB")
    cath_domain_list = os.getenv("CATH_DOMAIN_LIST")

    if not foldseek_exec:
        raise RuntimeError("FOLDSEEK_EXEC environment variable not set")
    if not cath_db:
        raise RuntimeError("CATH_FOLDSEEK_DB environment variable not set")
    if not cath_domain_list:
        raise RuntimeError("CATH_DOMAIN_LIST environment variable not set")

    domain_lookup = _parse_cath_domain_list(cath_domain_list)

    with tempfile.TemporaryDirectory(prefix="cath_metric_") as tmpdir:
        query_dir = os.path.join(tmpdir, "query_dir")
        os.makedirs(query_dir)

        # Foldseek's mmCIF parser requires occupancy and B-factor fields.
        cif_array = atom_array.copy()
        cif_array.set_annotation("occupancy", np.ones(cif_array.array_length(), dtype=np.float32))
        cif_array.set_annotation("b_factor", np.zeros(cif_array.array_length(), dtype=np.float32))
        to_cif_file(cif_array, os.path.join(query_dir, "query.cif"), save_standard_annotations=False, extra_fields=[])

        query_db = os.path.join(tmpdir, "query_db")
        result_db = os.path.join(tmpdir, "result_db")
        result_m8 = os.path.join(tmpdir, "results.m8")

        subprocess.run(
            [foldseek_exec, "createdb", query_dir, query_db, "--input-format", "2"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [foldseek_exec, "search", query_db, cath_db, result_db, tmpdir, *_SEARCH_FLAGS],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                foldseek_exec,
                "convertalis",
                query_db,
                cath_db,
                result_db,
                result_m8,
                "--format-output",
                _CONVERTALIS_COLS,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        best_hit = _parse_foldseek_m8(result_m8, domain_lookup)

    if best_hit is None:
        return dict(_NO_HIT)

    levels = _cath_levels(best_hit["code"])
    return {
        **levels,
        "type": best_hit["type"],
        "evalue": best_hit["evalue"],
        "tmscore": best_hit["tmscore"],
    }


class CathMetric(Metric):
    """Computes CATH superfamily label for the chain of interest using Foldseek.

    Required environment variables:
    - ``FOLDSEEK_EXEC``: path to foldseek binary
    - ``CATH_FOLDSEEK_DB``: path to CATH SSG95 foldseek database prefix
    - ``CATH_DOMAIN_LIST``: path to ``CathDomainList.S95.v4.4.0`` file
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map validation inputs to compute() parameters."""
        return {
            "atom_array": "predicted_atom_array",
            "chain_of_interest": ("extra_info", "chain_of_interest"),
        }

    def compute(  # type: ignore[override]
        self,
        atom_array: AtomArray | AtomArrayStack,
        chain_of_interest: str,
    ) -> dict[str, Any]:
        """Compute CATH superfamily label for the chain of interest.

        Args:
          atom_array: Full predicted structure. When a stack is provided,
            only the first model is used.
          chain_of_interest: Chain label (chain_id or chain_iid) identifying the
            designed chain.

        Returns:
          Dict with keys C, A, T, H (cumulative CATH prefixes or ``None``),
          plus ``type``, ``evalue``, and ``tmscore``.
        """
        if isinstance(atom_array, AtomArrayStack):
            atom_array = atom_array[0]

        chain_attr = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
        coi_array = atom_array[atom_array.get_annotation(chain_attr) == chain_of_interest]

        ensure_annotations(coi_array, "is_protein")
        if not coi_array.is_protein.all():
            raise ValueError(f"Chain {chain_of_interest!r} contains non-protein residues")

        return annotate_cath(coi_array)
