"""AF3 structure prediction oracle."""

import io
import json
import os
import shutil
import socket
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import biotite.structure as struc
import numpy as np
import torch.distributed as dist
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.io.utils.io_utils import CIFWriteConfig, _to_cif_or_bcif, to_cif_file, to_cif_string
from atomworks.ml.utils.annotator import ANNOTATOR_REGISTRY
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile
from biotite.structure.residues import apply_residue_wise, residue_iter, spread_residue_wise

from foundry.utils.ddp import RankedLogger
from proteinfoundation.metrics.oracle_metric import OracleJobInput, OracleMetric, extract_confidence_metrics
from proteinfoundation.transforms.utils.conditioning_utils import align_atom_array_to_input_frame
from proteinfoundation.utils.chain_utils import get_chain_of_interest
from scripts.af3_apptainer import build_apptainer_fold_command

logger = RankedLogger(__name__, rank_zero_only=True)


def _resolve_local_rank() -> int:
    """Per-node local rank (0..gpus_per_node-1). Used to index per-node
    ``CUDA_VISIBLE_DEVICES`` and the AF3 daemon socket template — both are
    per-node, so global rank would be wrong on multi-node runs.

    Reads ``LOCAL_RANK`` (set by Lightning / torchrun) then ``SLURM_LOCALID``,
    then falls back to 0 (single-process: tests, scripts).
    """
    local = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
    if local is not None:
        return int(local)
    return 0


def _resolve_gpu_device(local_rank: int) -> int:
    """Driver-level GPU index for this rank. Reads the rank's slot from
    ``CUDA_VISIBLE_DEVICES``; falls back to GPU 0 in single-process settings
    where the env var isn't set (tests, scripts)."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible:
        return 0

    visible = cuda_visible.split(",")
    if local_rank >= len(visible):
        raise RuntimeError(
            f"Rank {local_rank} has no slot in CUDA_VISIBLE_DEVICES={cuda_visible!r} "
            f"(only {len(visible)} GPUs visible)."
        )

    return int(visible[local_rank])


def _send_to_daemon(daemon_sock: str, request: dict, *, timeout_s: float = 3600.0) -> dict:
    """Send one JSON request to the launcher daemon over a Unix socket."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    try:
        sock.connect(daemon_sock)
        sock.sendall((json.dumps(request) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
    finally:
        sock.close()
    return json.loads(buf.decode().strip())


@dataclass(slots=True)
class _AF3JobState:
    """In-flight bookkeeping for one design between manifest-build and finalise.

    Lives in memory only — never serialised, never seen outside :class:`AF3Oracle`.

    Attributes:
        input_atom_array: The original (pre-disambiguation) input AtomArray;
            the canvas the final remapped coords get written into.
        example_id: Caller-supplied identifier; used for canonical filenames.
        out_dir: Canonical val output dir for this design (where the refolded
            CIF and confidence JSON go).
        disambiguated_atom_array: ``input_atom_array`` with chain_ids made
            unique per chain_iid; used by :py:meth:`_build_atom_index_mapping`.
        af3_chain_id_to_original_iid: Mapping from the chain_ids AF3 saw to
            the original chain_iids, ``None`` if no disambiguation happened.
        af3_output_dir: Per-design tempdir under which AF3 wrote its files.
        job_name: Sanitised ``example_id`` (spaces → underscores) used as the
            AF3 job name and CIF id.
        chain_of_interest: Pre-computed chain of interest, or ``None``.
    """

    input_atom_array: AtomArray
    example_id: str
    out_dir: Path
    disambiguated_atom_array: AtomArray
    af3_chain_id_to_original_iid: dict[str, str] | None
    af3_output_dir: Path
    job_name: str
    chain_of_interest: str | None


class AF3Oracle(OracleMetric):
    """AlphaFold 3 oracle.

    Folds run via :py:meth:`fold_batch`: one AF3 subprocess per rank folds
    every applicable design in a single warm process. The per-item
    :py:meth:`compute` (inherited) then reads the cached confidence JSON.
    :py:meth:`fold_one` exists for symmetry with :class:`OracleMetric` but
    pays full per-batch overheads.

    Two dispatch routes (selected automatically by ``fold_batch``):

    - **Daemon route** (Lightning DDP). When
      ``AF3_LAUNCHER_SOCKET_TEMPLATE`` is set, the rank's fold request is
      sent over the Unix socket served by ``af3_launcher_daemon``. The
      slurm launcher pre-spawns these daemons via
      ``scripts/af3_daemons.py``.
    - **Direct route** (ad-hoc). When the env var is unset,
      ``fold_batch`` runs the same apptainer command directly via
      ``subprocess.run``. The caller must be on a host where
      ``apptainer`` is on ``PATH``.

    Args:
        oracle_name: Output subdirectory label. Defaults to ``"af3"``.
        template_annotations: Annotation name(s) to use for templating.
        extra_af3_flags: Additional flags forwarded to the AF3 subprocess.
    """

    def __init__(
        self,
        oracle_name: str = "af3",
        template_annotations: str | list[str] | None = None,
        extra_af3_flags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(oracle_name=oracle_name, template_annotations=template_annotations, **kwargs)
        self.extra_flags = extra_af3_flags or []

    # ------------------------------------------------------------------
    # OracleMetric framework hooks
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """No-op. The AF3 worker is owned by the sibling launcher daemon."""
        return

    def teardown(self) -> None:
        """No-op. The daemon is shut down by the slurm script's EXIT trap."""
        if dist.is_initialized():
            dist.barrier()

    # ------------------------------------------------------------------
    # Fold path
    # ------------------------------------------------------------------

    def fold_one(self, job: OracleJobInput) -> dict:
        """Fold one design through the daemon.

        Inefficient, since pays per-batch costs; use :py:meth:`fold_batch`
        directly when possible.
        """
        n_ok = self.fold_batch([job])
        if n_ok != 1:
            raise RuntimeError(f"AF3Oracle.fold_one: fold_batch returned {n_ok} for {job.example_id!r}")
        return self._load_confidence_json(self.confidence_path(job.out_dir, job.example_id))

    def fold_batch(self, jobs: list[OracleJobInput]) -> int:
        """Fold every applicable design in one AF3 subprocess (per rank).

        Jobs already cached on disk are skipped. The remaining designs are
        serialised into a single JSON manifest, dispatched to AF3 via the
        daemon (Lightning DDP) or a direct ``apptainer run`` (ad-hoc), and
        post-processed back into the canonical ``confidence_path`` /
        ``refolded_cif_path`` locations.
        """
        pending = [j for j in jobs if not self.already_refolded(j.out_dir, j.example_id)]
        if not pending:
            return 0

        rank = _resolve_local_rank()
        gpu_device = _resolve_gpu_device(rank)

        batch_dir = Path(tempfile.mkdtemp(prefix="af3_batch_"))
        af3_outputs_root = batch_dir / "af3_outputs"
        af3_outputs_root.mkdir(parents=True, exist_ok=True)

        manifest_entries, job_states = self._build_manifest(pending, af3_outputs_root)
        manifest_path = batch_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_entries))

        try:
            self._dispatch_fold(manifest_path, rank, gpu_device, len(pending))

            n_ok = 0
            for state in job_states:
                if self._finalize_af3_result(state) is not None:
                    n_ok += 1
            return n_ok
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    def _dispatch_fold(self, manifest_path: Path, rank: int, gpu_device: int, n_designs: int) -> None:
        """Run the AF3 fold via the daemon if configured, else ``apptainer run`` directly.

        Raises on subprocess failure so the caller's ``finalize`` loop can
        log per-design issues without silently swallowing the batch.
        """
        sock_template = os.environ.get("AF3_LAUNCHER_SOCKET_TEMPLATE")

        if sock_template:
            daemon_sock = sock_template.format(rank=rank)
            # Extremely conservative limit of 2 min per design
            timeout_s = max(3600.0, n_designs * 120.0)
            logger.info(
                f"AF3 fold_batch: routing {n_designs} designs via daemon at {daemon_sock} "
                f"(gpu_device={gpu_device}, timeout={timeout_s / 3600:.1f}h)"
            )
            response = _send_to_daemon(
                daemon_sock,
                {
                    "cmd": "fold",
                    "manifest": str(manifest_path),
                    "gpu_device": gpu_device,
                    "save_full_outputs": False,
                },
                timeout_s=timeout_s,
            )
            if response.get("status") != "ok":
                raise RuntimeError(f"AF3 daemon error: {response.get('message')}")
            return

        logger.info(f"AF3 fold_batch: routing {n_designs} designs via direct apptainer call (gpu_device={gpu_device})")
        cmd = build_apptainer_fold_command(
            manifest_path=manifest_path,
            gpu_device=gpu_device,
            save_full_outputs=False,
        )
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise RuntimeError(f"AF3 apptainer subprocess failed with rc={rc}")

    def _build_manifest(
        self, pending: list[OracleJobInput], af3_outputs_root: Path
    ) -> tuple[list[dict], list[_AF3JobState]]:
        """Translate pending jobs into AF3-manifest entries plus per-item state.

        For each design we (1) sanitise the AtomArray for AF3's CIF parser
        (chain disambiguation, NaN-coord stripping, occupancy/b_factor
        defaults), (2) build the manifest entry the daemon will hand to AF3
        (inline CIF + optional template + optional user CCD), and (3) record a
        :class:`_AF3JobState` carrying everything :py:meth:`_finalize_af3_result`
        will need once AF3 finishes.

        Args:
            pending: Jobs not yet refolded on disk.
            af3_outputs_root: Per-batch tempdir under which AF3 will write each
                design's output subdirectory.

        Returns:
            ``(manifest_entries, job_states)`` lists in 1:1 order with ``pending``.
        """
        manifest_entries: list[dict] = []
        job_states: list[_AF3JobState] = []

        for job in pending:
            input_atom_array = job.atom_array
            if self.template_annotations is not None:
                annotations_to_ensure = [annot for annot in self.template_annotations if annot in ANNOTATOR_REGISTRY]
                if annotations_to_ensure:
                    ensure_annotations(input_atom_array, *annotations_to_ensure)
            example_id = job.example_id
            job_name = example_id.replace(" ", "_")

            # Disambiguate chain IDs so AF3 can distinguish chain_iid variants and keep an id_map for remapping outputs
            disambiguated_atom_array, af3_chain_id_to_original_iid = self._disambiguate_chain_ids(input_atom_array)

            # AF3's CIF parser refuses NaN coordinates, so drop atoms with any NaN component.
            # (AF3 will internally pad missing atoms back; we reconcile in _finalize_af3_result.)
            valid_mask = np.all(np.isfinite(disambiguated_atom_array.coord), axis=-1)
            disambiguated_valid = disambiguated_atom_array[valid_mask]

            # Biotite's CIF writer requires occupancy + b_factor columns;
            # designs don't have meaningful values, so fill in defaults.
            disambiguated_valid.set_annotation("occupancy", np.ones(disambiguated_valid.array_length()))
            disambiguated_valid.set_annotation("b_factor", np.zeros(disambiguated_valid.array_length()))

            # AF3 reads `_struct_conn` partner seq_ids as 1-based chain
            # indices (folding_input.py:1420); renumber so bonds survive.
            disambiguated_valid = self._renumber_res_ids_per_chain(disambiguated_valid)

            # Inline CIF in the manifest (avoid file I/O)
            cif_string = to_cif_string(disambiguated_valid.copy(), config=CIFWriteConfig(id=job_name))
            af3_output_dir = af3_outputs_root / job_name

            entry: dict = {
                "name": job_name,
                "output_dir": str(af3_output_dir),
                "cif": cif_string,
            }

            # Optional payloads when the design needs them:
            #   template — structural template constraint (chain coords) for the design
            #   user_ccd — connectivity for any non-standard residues so AF3 can bond them
            if template := self._build_af3_template_dict(disambiguated_valid):
                entry["template"] = template
            if user_ccd := self._build_user_ccd_string(disambiguated_valid):
                entry["user_ccd"] = user_ccd
            manifest_entries.append(entry)

            job_states.append(
                _AF3JobState(
                    input_atom_array=input_atom_array,
                    example_id=example_id,
                    out_dir=job.out_dir,
                    disambiguated_atom_array=disambiguated_atom_array,
                    af3_chain_id_to_original_iid=af3_chain_id_to_original_iid,
                    af3_output_dir=af3_output_dir,
                    job_name=job_name,
                    chain_of_interest=job.chain_of_interest,
                )
            )

        return manifest_entries, job_states

    def _finalize_af3_result(self, state: _AF3JobState) -> dict | None:
        """Read AF3's outputs for one design, remap coords back to the input
        ordering, save the canonical refolded CIF + confidence JSON, return metrics.

        Raises if AF3's summary-confidences JSON is missing — that means the
        subprocess silently failed for this design and we don't want to mask it.
        """
        summary_confidences = self._parse_af3_confidences(state.af3_output_dir, state.job_name)
        if summary_confidences is None:
            raise RuntimeError(
                f"AF3 produced no summary_confidences for {state.example_id!r} "
                f"(expected under {state.af3_output_dir}/{state.job_name}_summary_confidences.json) — "
                "the subprocess silently failed; check AF3 stderr."
            )

        predicted_atom_array = self._parse_af3_predicted_structure(state.af3_output_dir, state.job_name)
        if predicted_atom_array is not None:
            try:
                # AF3 adds missing atoms within residues but drops residues with all-NaN coords.
                # Match that by dropping all-NaN residues into `disambiguated_partial`; dropped residues keep NaNs
                nan_coord_mask = np.isnan(state.disambiguated_atom_array.coord).any(axis=-1)
                missing_residue_mask = apply_residue_wise(state.disambiguated_atom_array, nan_coord_mask, np.all)
                missing_residue_mask = spread_residue_wise(state.disambiguated_atom_array, missing_residue_mask)
                disambiguated_partial = state.disambiguated_atom_array[~missing_residue_mask]

                # Strip AF3 OXTs at residues where the input has no OXT
                # (E.g., preserve ligand OXTs)
                af3_oxt_mask = predicted_atom_array.atom_name == "OXT"
                if af3_oxt_mask.any():
                    extra_oxt_mask = np.zeros_like(af3_oxt_mask)
                    # NOTE: We can loop at chain-id level since we disambiguated above
                    for chain in np.unique(predicted_atom_array.chain_id):
                        af3_in = predicted_atom_array.chain_id == chain
                        inp_in = disambiguated_partial.chain_id == chain
                        if not inp_in.any():
                            continue

                        # Per-residue: does this residue have an OXT? (input vs AF3, by within-chain position)
                        af3_res_oxt = apply_residue_wise(predicted_atom_array[af3_in], af3_oxt_mask[af3_in], np.any)
                        inp_res_oxt = apply_residue_wise(
                            disambiguated_partial[inp_in], disambiguated_partial.atom_name[inp_in] == "OXT", np.any
                        )

                        # AF3 has OXT at residue r but input doesn't (residues without an input counterpart count too).
                        n = min(len(af3_res_oxt), len(inp_res_oxt))
                        extra_per_res = np.concatenate([af3_res_oxt[:n] & ~inp_res_oxt[:n], af3_res_oxt[n:]])
                        extra_oxt_mask[af3_in] = (
                            spread_residue_wise(predicted_atom_array[af3_in], extra_per_res) & af3_oxt_mask[af3_in]
                        )

                    # Only strip if it reconciles the totals; otherwise let the mismatch error fire.
                    if disambiguated_partial.array_length() == predicted_atom_array[~extra_oxt_mask].array_length():
                        predicted_atom_array = predicted_atom_array[~extra_oxt_mask]

                # Fail loud if AF3 dropped or rerouted an inter-residue bond
                self._validate_inter_residue_bonds(disambiguated_partial, predicted_atom_array)

                # Map every input atom to its AF3 counterpart and copy coords
                # only into residues that aren't all-NaN; dropped residues
                # keep their original NaN coords in the final result
                input_to_af3 = self._build_atom_index_mapping(disambiguated_partial, predicted_atom_array)
                result = as_atom_array_plus(state.input_atom_array.copy())
                result.coord[~missing_residue_mask] = predicted_atom_array.coord[input_to_af3]
                predicted_atom_array = result
            except (ValueError, KeyError) as e:
                logger.warning(f"AF3 coord transfer failed for {state.example_id}: {e}")
                predicted_atom_array = None

        if predicted_atom_array is not None:
            cif_path = self.refolded_cif_path(state.out_dir, state.example_id)
            cif_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                to_cif_file(align_atom_array_to_input_frame(predicted_atom_array), cif_path)
            except Exception as e:
                logger.warning(f"Failed to save AF3 refolded CIF for {state.example_id}: {e}")

        chain_of_interest = state.chain_of_interest
        if chain_of_interest is None:
            chain_of_interest, coi_unambiguous = get_chain_of_interest(state.input_atom_array)
            if not coi_unambiguous:
                chain_of_interest = None

        af3_chain_ids_in_order: list[str] = summary_confidences.get("chain_ids", [])
        if state.af3_chain_id_to_original_iid is not None:
            chains_in_order = [state.af3_chain_id_to_original_iid.get(c) for c in af3_chain_ids_in_order]
        else:
            chains_in_order = af3_chain_ids_in_order

        metrics = extract_confidence_metrics(
            summary_confidences=summary_confidences,
            chain_of_interest=chain_of_interest,
            chains_in_order=chains_in_order,
        )

        # Note the COI used to compute the metrics, even if it was ultimately None due to ambiguity.
        metrics["chain_of_interest"] = chain_of_interest

        # Cache canonical confidence JSON so per-item compute() resumability returns instantly.
        self._save_confidence_json(metrics, self.confidence_path(state.out_dir, state.example_id))
        return metrics

    # ------------------------------------------------------------------
    # AF3 input prep helpers
    # ------------------------------------------------------------------

    def _build_af3_template_dict(self, atom_array: AtomArray) -> dict[str, dict] | None:
        """Build the AF3 per-chain template dict from :attr:`template_annotations`.

        Args:
            atom_array: Input structure, with template annotation(s) set.

        Returns:
            Dict mapping chain ID → template dict, or ``None`` if no annotations.
        """
        if not self.template_annotations:
            return None

        template_mask = np.zeros(atom_array.array_length(), dtype=bool)
        for annot_name in self.template_annotations:
            assert annot_name in atom_array.get_annotation_categories(), (
                f"Annotation '{annot_name}' not found in AtomArray"
            )
            annot = atom_array.get_annotation(annot_name)
            assert annot.dtype == bool, f"Annotation '{annot_name}' must be boolean, got {annot.dtype}"
            template_mask |= annot

        if not np.any(template_mask):
            return None

        ensure_annotations(atom_array, "is_protein")
        non_protein_templated = template_mask & ~atom_array.is_protein
        if np.any(non_protein_templated):
            bad_res = np.unique(atom_array.res_name[non_protein_templated])
            raise ValueError(
                f"Template mask covers non-protein residues: {bad_res}. "
                "AF3 structural templating is only supported for protein chains."
            )

        template_dict: dict[str, dict] = {}
        for chain_id in np.unique(atom_array.chain_id):
            chain_mask = atom_array.chain_id == chain_id
            chain_atoms = atom_array[chain_mask]
            chain_template_mask = template_mask[chain_mask]
            if not np.any(chain_template_mask):
                continue

            residue_is_templated_all = apply_residue_wise(chain_atoms, chain_template_mask, np.all)
            residue_is_templated_any = apply_residue_wise(chain_atoms, chain_template_mask, np.any)
            if not np.array_equal(residue_is_templated_all, residue_is_templated_any):
                logger.warning(
                    f"Some residues on chain {chain_id} are partially templated. "
                    "AF3 structural templating only applies to full residues; "
                    "any partially-templated residue will be fully templated."
                )

            query_indices = np.where(residue_is_templated_any)[0].tolist()
            config = CIFWriteConfig(
                id=f"template_chain_{chain_id}",
                extra_categories={"pdbx_audit_revision_history": {"revision_date": date.today().isoformat()}},
            )
            template_dict[chain_id] = {
                "mmcif": to_cif_string(chain_atoms, config=config),
                "query_indices": query_indices,
                "template_indices": query_indices,
            }
        return template_dict or None

    def _build_user_ccd_string(self, atom_array: AtomArray) -> str | None:
        """Build the user CCD string for AF3 from the input AtomArray.

        Anything that is not a standard polymer residue is added to the user CCD.
        This ensures that reusing residue names from the CCD is handled correctly.

        NOTE: This will not handle the situation where a given residue name means
        two different things within the same AtomArray. That should never be done.

        Args:
            atom_array: Input structure as an AtomArray.

        Returns:
            Multi-block CIF string for non-standard residues, or ``None``.
        """
        ensure_annotations(atom_array, "is_standard_aa", "is_standard_dna", "is_standard_rna")
        nonstandard_mask = ~(atom_array.is_standard_aa | atom_array.is_standard_dna | atom_array.is_standard_rna)
        if not np.any(nonstandard_mask):
            return None

        user_ccd_arrays: dict[str, AtomArray] = {}
        for res_array in residue_iter(atom_array[nonstandard_mask]):
            if res_array.array_length() == 1:
                continue  # single-atom residues don't get a custom CCD code
            res_name = res_array.res_name[0]
            user_ccd_arrays.setdefault(res_name, res_array)

        config = CIFWriteConfig(
            include_entity_categories=True,
            id="user_ccd_entry",
            extra_categories={"pdbx_audit_revision_history": {"revision_date": date.today().isoformat()}},
        )
        user_ccd_cif_blocks = {
            name: _to_cif_or_bcif(arr, config=config)["user_ccd_entry"] for name, arr in user_ccd_arrays.items()
        }

        buffer = io.StringIO()
        CIFFile(blocks=user_ccd_cif_blocks).write(buffer)
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # AF3 output parse helpers
    # ------------------------------------------------------------------

    def _parse_af3_predicted_structure(self, af3_output_dir: Path, job_name: str) -> AtomArray | None:
        """Parse the best-ranking AF3 predicted structure from CIF output.

        Args:
            af3_output_dir: Directory where AF3 wrote its output files directly.
            job_name: Sanitised job name (spaces replaced with underscores).

        Returns:
            Predicted :class:`~biotite.structure.AtomArray`, or ``None`` if the
            expected CIF file is missing.
        """
        cif_path = af3_output_dir / f"{job_name}_model.cif"
        if not cif_path.exists():
            logger.warning(f"AF3 output CIF not found: {cif_path}")
            return None
        config = ParseConfig.from_preset("minimal", extra_fields=("b_factor", "occupancy"))
        result = parse(str(cif_path), config=config)["asym_unit"][0]
        return as_atom_array_plus(result)

    def _parse_af3_confidences(self, af3_output_dir: Path, job_name: str) -> dict | None:
        """Parse the AF3 summary-confidences JSON.

        Args:
            af3_output_dir: Directory where AF3 wrote its output files directly.
            job_name: Sanitised job name.

        Returns:
            Parsed confidences dict, or ``None`` if the file is missing.
        """
        json_path = af3_output_dir / f"{job_name}_summary_confidences.json"
        if not json_path.exists():
            logger.warning(f"AF3 summary confidences JSON not found: {json_path}")
            return None
        with open(json_path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Chain / atom remapping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _disambiguate_chain_ids(atom_array: AtomArray) -> tuple[AtomArray, dict[str, str] | None]:
        """Assign unique chain IDs to chains that share a chain_id but differ by chain_iid.

        Args:
            atom_array: Input structure, optionally with a ``chain_iid`` annotation.

        Returns:
            AtomArray with renamed chains and a mapping from chain_iids to their
            af3 chain_ids (``None`` if no ``chain_iid`` annotation).
        """
        if "chain_iid" not in atom_array.get_annotation_categories():
            return atom_array, None

        seen: set[tuple[str, str]] = set()
        unique_pairs: list[tuple[str, str]] = []
        for chain_id, chain_iid in zip(atom_array.chain_id, atom_array.chain_iid, strict=False):
            pair = (str(chain_id), str(chain_iid))
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)

        chain_id_iids: dict[str, list[str]] = {}
        for chain_id, iid in unique_pairs:
            chain_id_iids.setdefault(chain_id, []).append(iid)
        ambiguous = {chain_id for chain_id, iids in chain_id_iids.items() if len(iids) > 1}
        af3_chain_id_to_original_iid = {
            chain_id: iids[0] for chain_id, iids in chain_id_iids.items() if chain_id not in ambiguous
        }
        if not ambiguous:
            return atom_array, af3_chain_id_to_original_iid

        existing = set(np.unique(atom_array.chain_id).tolist())
        gen = create_chain_id_generator(unavailable_chain_ids=existing)
        pair_to_new: dict[tuple[str, str], str] = {}
        for chain_id, iid in unique_pairs:
            if chain_id in ambiguous:
                new_name = next(gen)
                pair_to_new[(chain_id, iid)] = new_name
                existing.add(new_name)

        renamed = atom_array.copy()
        for (orig_chain_id, iid), new_name in pair_to_new.items():
            mask = (atom_array.chain_id == orig_chain_id) & (atom_array.chain_iid == iid)
            renamed.chain_id[mask] = new_name
            af3_chain_id_to_original_iid[new_name] = iid
        return renamed, af3_chain_id_to_original_iid

    @staticmethod
    def _renumber_res_ids_per_chain(atom_array: AtomArray) -> AtomArray:
        """Renumber ``res_id`` to ``1..N`` within each chain.

        NOTE: Chains are identified by ``chain_id`` alone, so callers must
        ensure chains are distinguishable by ``chain_id`` for the renumbering
        to be meaningful.
        """
        # Avoid modifying input AtomArray
        out = atom_array.copy()
        new_res_id = out.res_id.copy()

        # Loop through chains in their first-occurrence order
        _, first_occ = np.unique(out.chain_id, return_index=True)
        for chain_id in out.chain_id[np.sort(first_occ)]:
            in_chain = np.where(out.chain_id == chain_id)[0]
            bounds = struc.get_residue_starts(out[in_chain], add_exclusive_stop=True)
            for new_idx in range(len(bounds) - 1):
                start, stop = int(bounds[new_idx]), int(bounds[new_idx + 1])
                new_res_id[in_chain[start:stop]] = new_idx + 1

        # Update the residue ids in-place
        out.res_id = new_res_id

        return out

    @staticmethod
    def _validate_inter_residue_bonds(input_array: AtomArray, af3_array: AtomArray) -> None:
        """Assert input and AF3 inter-residue bond graphs are identical.

        Raises:
          ValueError: If the input and AF3 inter-residue bond sets
            differ in either direction (drop or addition).
        """

        def _keys(arr: AtomArray) -> set[tuple]:
            """Convert an AtomArray into a set of inter-residue bond endpoint pairs."""
            if arr.bonds is None:
                return set()

            # Renumber for safety (in the event AF3 may have renumbered differently from us)
            renumbered = AF3Oracle._renumber_res_ids_per_chain(arr)
            chain_id = renumbered.chain_id
            res_id = renumbered.res_id
            atom_name = renumbered.atom_name

            # Build up sorted tuples of ((chain_id, res_id, atom_name), (chain_id, res_id, atom_name)) for each bond
            out: set[tuple] = set()
            for i, j, _btype in renumbered.bonds.as_array():
                if chain_id[i] == chain_id[j] and res_id[i] == res_id[j]:
                    continue
                ep1 = (str(chain_id[i]), int(res_id[i]), str(atom_name[i]))
                ep2 = (str(chain_id[j]), int(res_id[j]), str(atom_name[j]))
                out.add(tuple(sorted([ep1, ep2])))

            return out

        # Ensure the sets of inter-residue bonds match exactly
        inp = _keys(input_array)
        af3 = _keys(af3_array)
        if inp != af3:
            missing = sorted(inp - af3)
            extra = sorted(af3 - inp)
            raise ValueError(
                f"AF3 inter-residue bond graph differs from input (missing in AF3: {missing}; extra in AF3: {extra})"
            )

    @staticmethod
    def _build_atom_index_mapping(input_array: AtomArray, af3_array: AtomArray) -> np.ndarray:
        """Build a per-atom index mapping from input to AF3 output.

        Walks each chain residue-by-residue and matches atom names within the
        residue. Raises :class:`ValueError` on any structural mismatch (atom
        count, residue count, residue sequence, atom names, or duplicates).

        Args:
            input_array: Normalised input structure.
            af3_array: Parsed AF3 output structure.

        Returns:
            ``input_to_af3`` of shape ``(n_atoms,)`` such that
            ``af3_array[input_to_af3[i]]`` is the AF3 atom corresponding to
            ``input_array[i]``.
        """
        if input_array.array_length() != af3_array.array_length():
            raise ValueError(
                f"Atom count mismatch: input has {input_array.array_length()}, "
                f"AF3 output has {af3_array.array_length()}"
            )

        input_to_af3 = np.empty(input_array.array_length(), dtype=np.intp)
        _, first_occ = np.unique(input_array.chain_id, return_index=True)
        chain_ids = input_array.chain_id[np.sort(first_occ)]

        for chain_id in chain_ids:
            inp_global = np.where(input_array.chain_id == chain_id)[0]
            af3_global = np.where(af3_array.chain_id == chain_id)[0]
            inp_chain = input_array[inp_global]
            af3_chain = af3_array[af3_global]

            inp_bounds = struc.get_residue_starts(inp_chain, add_exclusive_stop=True)
            af3_bounds = struc.get_residue_starts(af3_chain, add_exclusive_stop=True)
            n_inp_res = len(inp_bounds) - 1
            n_af3_res = len(af3_bounds) - 1
            if n_inp_res != n_af3_res:
                raise ValueError(f"Chain '{chain_id}': {n_inp_res} residues in input vs {n_af3_res} in AF3 output")

            inp_seq = inp_chain.res_name[inp_bounds[:-1]]
            af3_seq = af3_chain.res_name[af3_bounds[:-1]]
            if not np.array_equal(inp_seq, af3_seq):
                raise ValueError(
                    f"Chain '{chain_id}': residue sequence mismatch\n"
                    f"  input: {inp_seq.tolist()}\n  AF3:   {af3_seq.tolist()}"
                )

            for res_i in range(n_inp_res):
                i0, i1 = int(inp_bounds[res_i]), int(inp_bounds[res_i + 1])
                a0, a1 = int(af3_bounds[res_i]), int(af3_bounds[res_i + 1])
                inp_names = inp_chain.atom_name[i0:i1]
                af3_names = af3_chain.atom_name[a0:a1]
                if len(inp_names) != len(af3_names):
                    raise ValueError(
                        f"Chain '{chain_id}', residue {res_i} ({inp_chain.res_name[i0]}): "
                        f"{len(inp_names)} atoms in input vs {len(af3_names)} in AF3"
                    )

                af3_name_to_local: dict[str, int] = {}
                for local_j, name in enumerate(af3_names):
                    if name in af3_name_to_local:
                        raise ValueError(
                            f"Chain '{chain_id}', residue {res_i}: duplicate atom name '{name}' in AF3 output"
                        )
                    af3_name_to_local[name] = local_j

                seen_inp: set[str] = set()
                for local_i, name in enumerate(inp_names):
                    if name in seen_inp:
                        raise ValueError(f"Chain '{chain_id}', residue {res_i}: duplicate atom name '{name}' in input")
                    seen_inp.add(name)
                    if name not in af3_name_to_local:
                        raise ValueError(f"Chain '{chain_id}', residue {res_i}: atom '{name}' not found in AF3 output")
                    input_to_af3[inp_global[i0 + local_i]] = af3_global[a0 + af3_name_to_local[name]]

        return input_to_af3
