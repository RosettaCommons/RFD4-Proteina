"""CATH superfamily annotation metric, via Foldseek structural search or Proteina's classifier.

Two annotators are available (see :py:class:`CathMetric`):

- ``foldseek``: the Foldseek search against the CATH SSG95 database, which assigns a
  full C/A/T/H code but only for structures that match a T-level fold already in CATH.
- ``proteina_classifier``: the GearNet fold classifier released with
  `NVIDIA-BioNeMo/proteina <https://github.com/NVIDIA-BioNeMo/proteina>`_, which always
  returns a C/A/T call (no H level), never fails to answer, and costs ~0.2 s/call against
  Foldseek's ~3.7 s on a single CPU core (both annotators run on CPU only).

Required environment variables (Foldseek):
- FOLDSEEK_EXEC: path to foldseek binary
- CATH_FOLDSEEK_DB: path to CATH SSG95 foldseek database prefix (cath_v4_4_0_s95_db)
- CATH_DOMAIN_LIST: path to CathDomainList.S95.v4.4.0 file

Required environment variables (Proteina classifier):
- GEARNET_CATH_CKPT_PATH: path to ``gearnet_ca.pth`` (NVIDIA's public release only includes
  the CA-only checkpoint)
- GEARNET_CATH_LABEL_MAPPING_PATH: path to ``cath_label_mapping.pt``

Both classifier files come from the ``proteina_additional_files`` bundle of
NVIDIA-BioNeMo/proteina (NVIDIA non-commercial research license).
"""

import logging
import os
import subprocess
import tempfile
from functools import cache
from typing import Any, Literal

import numpy as np
import torch
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise
from biotite.structure import AtomArray, AtomArrayStack, spread_residue_wise
from rfproteina.metrics.gearnet import GearNet

from rfproteina.conditions import C_IDXGRP
from rfproteina.metrics import Metric

logger = logging.getLogger(__name__)

# Annotator choices for CathMetric
FOLDSEEK = "foldseek"
PROTEINA_CLASSIFIER = "proteina_classifier"
FOLDSEEK_WITH_FALLBACK = "foldseek_with_fallback"
_ANNOTATORS = (FOLDSEEK, PROTEINA_CLASSIFIER, FOLDSEEK_WITH_FALLBACK)

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


def annotate_cath(atom_array: AtomArray, threads: int | None = None) -> dict[str, Any]:
    """Compute CATH superfamily annotation for a single-chain AtomArray via Foldseek.

    Runs the full Foldseek createdb -> search -> convertalis pipeline against the
    CATH SSG95 database, then classifies the best hit using thresholds from the
    domain-annotation-pipeline.

    Args:
      atom_array: Single-chain protein structure (chain of interest, amino acids only).
      threads: ``--threads`` for foldseek. ``None`` leaves foldseek to its own default,
        which is every core it can see.

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
    thread_flags = [] if threads is None else ["--threads", str(threads)]

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
            [foldseek_exec, "createdb", query_dir, query_db, "--input-format", "2", *thread_flags],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [foldseek_exec, "search", query_db, cath_db, result_db, tmpdir, *_SEARCH_FLAGS, *thread_flags],
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
                *thread_flags,
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


CLASSIFIER_LEVELS: tuple[str, ...] = ("C", "A", "T")


def _assemble_cath_classifier_inputs(atom_array: AtomArray) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the dense per-residue CA array the classifier consumes, indexed by ``res_id``.

    The classifier takes only ``(coords, coord_mask, batch)``; a residue's sequence position is
    implicitly its **row index** in this dense array. Rows advance by one per residue, except across
    an unbonded sequence gap, where they advance by the ``res_id`` difference. Unresolved CA atoms
    are also excluded. If any residue lacks a backbone CA, returns empty inputs.

    Args:
      atom_array: One chain of one IndexGroup, amino acids only. ``res_id`` is used to determine
        sequence position.

    Returns:
      Tuple of ``coords`` (``[n_rows, 3]``, zero-filled and unused where unresolved) and
      ``mask`` (``[n_rows]`` bool). ``n_rows`` can exceed the number of residues present in the input.

    Raises:
      ValueError: If the array spans more than one chain or IndexGroup.
    """
    if atom_array.array_length() == 0:
        return torch.zeros((0, 3)), torch.zeros(0, dtype=torch.bool)

    chain_attr = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
    index_groups = C_IDXGRP.annotation(atom_array, default="generate")
    if len(np.unique(atom_array.get_annotation(chain_attr))) > 1 or len(np.unique(index_groups)) > 1:
        raise ValueError("_assemble_cath_classifier_inputs needs a single chain in a single IndexGroup")

    ensure_annotations(atom_array, "is_protein_backbone")
    is_ca = atom_array.is_protein_backbone & (atom_array.atom_name == "CA")
    if not apply_and_spread_residue_wise(atom_array, is_ca, np.any).all():
        logger.warning("Not every residue carries a backbone CA, so no CATH call is made for this chain")
        return torch.zeros((0, 3)), torch.zeros(0, dtype=torch.bool)

    residue_starts = get_residue_starts(atom_array)
    n_residues = len(residue_starts)
    res_ids = atom_array.res_id[residue_starts].astype(np.int64)
    if (np.diff(res_ids) < 1).any():
        raise ValueError("res_id must be strictly increasing along the chain to serve as a sequence axis")

    # A res_id jump only means missing residues when the two are not bonded to each other
    atom_to_residue = spread_residue_wise(atom_array, np.arange(n_residues))
    joined_to_next = np.zeros(n_residues, dtype=bool)
    if atom_array.bonds is not None and len(atom_array.bonds.as_array()):
        atom1, atom2, _ = atom_array.bonds.as_array().T
        residue1, residue2 = atom_to_residue[atom1], atom_to_residue[atom2]
        consecutive = np.abs(residue1 - residue2) == 1
        joined_to_next[np.minimum(residue1[consecutive], residue2[consecutive])] = True

    # Rows to insert after each residue, for residues the chain is missing at that point.
    extra_rows_after = np.zeros(n_residues, dtype=np.int64)
    extra_rows_after[:-1] = np.where(~joined_to_next[:-1] & (np.diff(res_ids) > 1), np.diff(res_ids) - 1, 0)

    residue_rows = np.arange(n_residues) + np.concatenate([[0], np.cumsum(extra_rows_after)[:-1]])
    n_rows = int(residue_rows[-1]) + 1

    # One CA per residue, and only where it is actually resolved.
    is_resolved_ca = is_ca & np.isfinite(atom_array.coord).all(axis=-1)
    rows_of_ca = spread_residue_wise(atom_array, residue_rows)[is_resolved_ca]

    coords = np.zeros((n_rows, 3), dtype=np.float32)
    mask = np.zeros(n_rows, dtype=bool)
    coords[rows_of_ca] = atom_array.coord[is_resolved_ca]
    mask[rows_of_ca] = True
    return torch.from_numpy(coords), torch.from_numpy(mask)


class CathMetric(Metric):
    """Computes the CATH label for the chain of interest, via Foldseek and/or Proteina's classifier.

    Which annotator runs is set by ``annotator``:

    - ``"foldseek"``: Foldseek structural search against the CATH SSG95 database.
      Assigns a full C/A/T/H code, but only for chains matching a fold already in CATH;
      everything else comes back all-``None``.
    - ``"proteina_classifier"``: the GearNet fold classifier from NVIDIA-BioNeMo/proteina.
      C/A/T only, and always returns a call.
    - ``"foldseek_with_fallback"``: run Foldseek first, and fall back to the classifier only
      for chains where Foldseek finds no match. Gives Foldseek's H-level precision where it
      applies, and a C/A/T call rather than nothing for novel folds.

    Required environment variables (``foldseek``, and the first pass of
    ``foldseek_with_fallback``):

    - ``FOLDSEEK_EXEC``: path to foldseek binary
    - ``CATH_FOLDSEEK_DB``: path to CATH SSG95 foldseek database prefix
    - ``CATH_DOMAIN_LIST``: path to ``CathDomainList.S95.v4.4.0`` file

    Required environment variables (``proteina_classifier`` and ``foldseek_with_fallback``):

    - ``GEARNET_CATH_CKPT_PATH``: path to ``gearnet_ca.pth``
    - ``GEARNET_CATH_LABEL_MAPPING_PATH``: path to ``cath_label_mapping.pt``
    """

    def __init__(
        self,
        annotator: Literal["foldseek", "proteina_classifier", "foldseek_with_fallback"] = FOLDSEEK_WITH_FALLBACK,
        classifier_ckpt_path: str | None = None,
        classifier_label_mapping_path: str | None = None,
        foldseek_threads: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CathMetric.

        Args:
          annotator: Which annotator to use -- see the class docstring. Defaults to
            ``"foldseek_with_fallback"``.
          classifier_ckpt_path: Path to ``gearnet_ca.pth``. Falls back to
            ``GEARNET_CATH_CKPT_PATH``. Only needed when the classifier can run.
          classifier_label_mapping_path: Path to ``cath_label_mapping.pt``. Falls back to
            ``GEARNET_CATH_LABEL_MAPPING_PATH``. Only needed when the classifier can run.
          foldseek_threads: ``--threads`` passed to foldseek -- see :py:func:`annotate_cath`.

        Raises:
          ValueError: If ``annotator`` is unrecognized. The classifier paths are only resolved and
            checked in :py:meth:`setup`.
        """
        if annotator not in _ANNOTATORS:
            raise ValueError(f"annotator must be one of {_ANNOTATORS}, got {annotator!r}")

        super().__init__(**kwargs)
        self.annotator = annotator
        self.classifier_ckpt_path = classifier_ckpt_path
        self.classifier_label_mapping_path = classifier_label_mapping_path
        self.foldseek_threads = foldseek_threads
        self._classifier: GearNet | None = None
        self._label_mapping: dict[str, dict[int, str]] = {}

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map validation inputs to compute() parameters."""
        return {
            "atom_array": "predicted_atom_array",
            "chain_of_interest": ("extra_info", "chain_of_interest"),
        }

    def setup(self) -> None:
        """Resolve the classifier paths and load it once for the whole validation epoch, if it can run."""
        if self.annotator == FOLDSEEK:
            return

        self.classifier_ckpt_path = self.classifier_ckpt_path or os.getenv("GEARNET_CATH_CKPT_PATH")
        self.classifier_label_mapping_path = self.classifier_label_mapping_path or os.getenv(
            "GEARNET_CATH_LABEL_MAPPING_PATH"
        )
        if not self.classifier_ckpt_path:
            raise ValueError(
                f"annotator={self.annotator!r} needs the Proteina classifier checkpoint: pass "
                "classifier_ckpt_path or set GEARNET_CATH_CKPT_PATH"
            )
        if not self.classifier_label_mapping_path:
            raise ValueError(
                f"annotator={self.annotator!r} needs the Proteina classifier label mapping: pass "
                "classifier_label_mapping_path or set GEARNET_CATH_LABEL_MAPPING_PATH"
            )

        model = GearNet()
        model.load_state_dict(torch.load(self.classifier_ckpt_path, map_location="cpu", weights_only=True))
        self._classifier = model.eval().requires_grad_(False)
        mapping = torch.load(self.classifier_label_mapping_path, map_location="cpu", weights_only=True)
        self._label_mapping = {
            level: {idx: code for code, idx in mapping[level].items()} for level in CLASSIFIER_LEVELS
        }
        logger.info(f"Loaded Proteina CATH classifier from {self.classifier_ckpt_path}")

    def teardown(self) -> None:
        """Free the Proteina classifier after validation."""
        self._classifier = None
        self._label_mapping = {}

    def compute(  # type: ignore[override]
        self,
        atom_array: AtomArray | AtomArrayStack,
        chain_of_interest: str,
    ) -> dict[str, Any]:
        """Compute the CATH label for the chain of interest.

        Args:
          atom_array: Full predicted structure. When a stack is provided,
            only the first model is used.
          chain_of_interest: Chain label (chain_id or chain_iid) identifying the
            designed chain.

        Returns:
          Dict with keys C, A, T, H (cumulative CATH prefixes or ``None``), plus ``type``,
          ``evalue``, ``tmscore``, and ``annotator`` -- which of ``"foldseek"`` /
          ``"proteina_classifier"`` produced the label. Under ``"foldseek_with_fallback"``
          that key is the only way to tell the two apart, so it is always reported.
          The classifier leaves ``H``, ``type``, ``evalue``, and ``tmscore`` as ``None``.
        """
        if isinstance(atom_array, AtomArrayStack):
            atom_array = atom_array[0]

        chain_attr = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
        coi_array = atom_array[atom_array.get_annotation(chain_attr) == chain_of_interest]

        ensure_annotations(coi_array, "is_protein")
        if not coi_array.is_protein.all():
            raise ValueError(f"Chain {chain_of_interest!r} contains non-protein residues")

        if self.annotator != PROTEINA_CLASSIFIER:
            foldseek_result = annotate_cath(coi_array, threads=self.foldseek_threads)
            # A no-match leaves every level None, so C is enough to tell a hit from a miss.
            if self.annotator == FOLDSEEK or foldseek_result["C"] is not None:
                return {**foldseek_result, "annotator": FOLDSEEK}

        if self._classifier is None:
            raise RuntimeError("Call CathMetric.setup() before computing classifier annotations")
        result = {**_NO_HIT, "annotator": PROTEINA_CLASSIFIER}
        coords, coord_mask = _assemble_cath_classifier_inputs(coi_array)
        if coord_mask.any():
            with torch.no_grad():
                output = self._classifier(coords, coord_mask, torch.zeros(len(coords), dtype=torch.long))
            for level in CLASSIFIER_LEVELS:
                result[level] = self._label_mapping[level][int(output[f"pred_{level}"][0].argmax())]
        return result
