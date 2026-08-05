"""AF3 structure prediction oracle."""

import atexit
import contextlib
import errno
import gzip
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import biotite.structure as struc
import numpy as np
import torch.distributed as dist
from atomworks.constants import STANDARD_AA
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.io.utils.io_utils import CIFWriteConfig, _to_cif_or_bcif, to_cif_file, to_cif_string
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile
from biotite.structure.residues import apply_residue_wise, residue_iter, spread_residue_wise

from rfproteina.af3.af3_apptainer import build_apptainer_fold_command
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput, OracleMetric, extract_confidence_metrics
from rfproteina.transforms.utils.conditioning_utils import align_atom_array_to_input_frame
from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.mask_spec import resolve_mask

logger = RankedLogger(__name__, rank_zero_only=True)

# AF3 CCD validator requires _chem_comp_bond.* keys to be present even when a
# residue has no intra-residue bonds (e.g. sulfide ions in Fe-S clusters).
# An empty loop_ with just column headers satisfies the check; AF3's C++ CIF
# parser registers the keys with empty value lists, and parse_bond_data()
# handles empty lists gracefully.
_CHEM_COMP_BOND_EMPTY_LOOP = (
    "loop_\n"
    "_chem_comp_bond.atom_id_1\n"
    "_chem_comp_bond.atom_id_2\n"
    "_chem_comp_bond.value_order\n"
    "_chem_comp_bond.pdbx_aromatic_flag\n"
)


def _inject_empty_chem_comp_bond(cif_str: str) -> str:
    """Inject an empty _chem_comp_bond loop into any CIF block that lacks one."""

    def _process_block(block: str) -> str:
        if "_chem_comp_bond.atom_id_1" in block:
            return block
        stripped = block.rstrip()
        if stripped.endswith("#"):
            return stripped[:-1] + _CHEM_COMP_BOND_EMPTY_LOOP + "#\n"
        return stripped + "\n" + _CHEM_COMP_BOND_EMPTY_LOOP

    parts = re.split(r"(?=^data_)", cif_str, flags=re.MULTILINE)
    return "".join(_process_block(p) if p.startswith("data_") else p for p in parts)


_AF3_DAEMON_SCRIPT = Path(__file__).resolve().parents[2] / "af3" / "af3_daemon.py"


def _shutdown_daemon(proc: subprocess.Popen, sock_dir: Path) -> None:
    """Stop a daemon we started. Killing it closes the warm worker's stdin, so AF3
    sees EOF, exits, and frees its GPU memory."""
    proc.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)
    shutil.rmtree(sock_dir, ignore_errors=True)


def _resolve_local_rank() -> int:
    """Per-node local rank (0..gpus_per_node-1). Used to index this process's
    slot in ``CUDA_VISIBLE_DEVICES`` (per-node, so global rank would be wrong
    on multi-node runs).

    Reads ``LOCAL_RANK`` (set by Lightning / torchrun) then ``SLURM_LOCALID``,
    then falls back to 0 (single-process: tests, scripts).
    """
    local = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
    if local is not None:
        return int(local)
    return 0


def _resolve_daemon_rank(local_rank: int) -> int:
    """Daemon socket rank: AF3_DAEMON_RANK (set per-GPU by fan-out launchers, where
    every worker sees local rank 0), else local_rank (correct under DDP)."""
    explicit = os.environ.get("AF3_DAEMON_RANK")
    if explicit is not None:
        return int(explicit)
    return local_rank


def _resolve_gpu_device(local_rank: int) -> int:
    """Driver GPU index for the direct (no-daemon) fold path only. Reads the rank's
    slot from CUDA_VISIBLE_DEVICES; falls back to GPU 0 when unset."""
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


def _send_to_daemon(
    daemon_sock: str,
    request: dict,
    *,
    timeout_s: float = 3600.0,
    connect_retry_s: float = 600.0,
    connect_retry_interval_s: float = 10.0,
) -> dict:
    """Send one JSON request to the launcher daemon over a Unix socket.

    Retries on ECONNREFUSED/EAGAIN for up to ``connect_retry_s`` seconds so
    callers that start before the daemon has finished warming up (e.g.
    evaluate.py, which has no model-load delay) do not fail immediately.
    EAGAIN shows up alongside ECONNREFUSED because AF_UNIX listen backlogs are
    small by default: when several rank processes connect to the same daemon
    socket in a burst (e.g. right as a validation epoch starts), the kernel
    returns EAGAIN for connections that arrive while the backlog is full,
    rather than queuing them like AF_INET does.
    """
    deadline = time.monotonic() + connect_retry_s
    while True:
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
            return json.loads(buf.decode().strip())
        except OSError as exc:
            if exc.errno in (errno.ECONNREFUSED, errno.EAGAIN, errno.EWOULDBLOCK) and time.monotonic() < deadline:
                sock.close()
                time.sleep(connect_retry_interval_s)
                continue
            raise
        finally:
            sock.close()


@dataclass(slots=True)
class _AF3JobState:
    """In-flight bookkeeping for one design between manifest-build and finalise.

    Lives in memory only — never serialised, never seen outside :class:`AF3Oracle`.

    Attributes:
        input_atom_array: The original (pre-disambiguation) input AtomArray;
            the canvas the final remapped coords get written into.
        example_id: Caller-supplied identifier; used for canonical filenames.
        save_id: Which ``pre_oracle_transform`` output this is, or ``None``;
            distinguishes the filenames of one design's several refolds.
        out_dir: Canonical val output dir for this design (where the refolded
            CIF and confidence JSON go).
        disambiguated_atom_array: ``input_atom_array`` with chain_ids made
            unique per chain_iid; used by :py:meth:`_build_atom_index_mapping`.
        af3_chain_id_to_original_iid: Mapping from the chain_ids AF3 saw to
            the original chain_iids, ``None`` if no disambiguation happened.
        af3_output_dir: Per-design tempdir under which AF3 wrote its files.
        job_name: Sanitised ``example_id`` (plus ``save_id``, spaces →
            underscores) used as the AF3 job name and CIF id.
        chain_of_interest: Pre-computed chain of interest, or ``None``.
    """

    input_atom_array: AtomArray
    example_id: str
    save_id: str | None
    out_dir: Path
    disambiguated_atom_array: AtomArray
    af3_chain_id_to_original_iid: dict[str, str] | None
    af3_output_dir: Path
    job_name: str
    chain_of_interest: str | None


# Atom-name pairs that make an inter-residue bond a polymer backbone link, matching how
# OpenFold3's tokenizer classifies them (core/data/resources/residues.py).
_PEPTIDE_BOND_ATOMS = frozenset({"N", "C"})
_PHOSPHODIESTER_BOND_ATOMS = frozenset({"P", "O3'", "O5'"})


def _is_backbone_link(atom_name_i: str, atom_name_j: str) -> bool:
    """Whether a bond between these two atoms is a peptide or phosphodiester backbone link."""
    pair = {atom_name_i, atom_name_j}
    return pair <= _PEPTIDE_BOND_ATOMS or pair <= _PHOSPHODIESTER_BOND_ATOMS


def _is_implicit_backbone_link(res_name_i: str, atom_name_i: str, res_name_j: str, atom_name_j: str) -> bool:
    """Whether AF3 leaves this backbone link implicit while our writer emits it.

    Atomizing a residue means giving it a custom CCD code, and our writer then emits its
    flanking peptide bonds.
    """
    if not _is_backbone_link(atom_name_i, atom_name_j):
        return False
    return (res_name_i in STANDARD_AA) != (res_name_j in STANDARD_AA)


_OF3_PROVENANCE_FILENAME = "of3_ported_weights.provenance.json"
# Custom CCD codes shaped like the one tests/e2e uses for its user-CCD heme.
_ATOMIZED_CODE_FORMAT = "L:{index}"


def _using_of3_weights() -> bool:
    """Whether ``AF3_MODEL_DIR`` holds converted OpenFold3 parameters, per its provenance stamp.

    Duplicated in ``run_alphafold_batch``, which runs in the AF3 image without ``rfproteina``.
    """
    model_dir = os.environ.get("AF3_MODEL_DIR", "").strip()
    provenance = Path(model_dir) / _OF3_PROVENANCE_FILENAME if model_dir else None
    if provenance is None or not provenance.is_file():
        return False
    # Deliberately unguarded: a corrupt stamp must not quietly fall back to AF3 weights.
    return json.loads(provenance.read_text()).get("weight_family") == "openfold3"


def _atomize_ligand_bonded_residues(atom_array: AtomArray) -> AtomArray:
    """Give a custom CCD code to standard residues covalently bonded to a non-polymer.

    OpenFold3 tokenizes those per atom; a custom code is the only lever AF3 exposes.
    Returns a copy, or the input if there are none.
    """
    if atom_array.bonds is None:
        return atom_array
    ensure_annotations(atom_array, "is_protein", "is_nucleic_acid")
    is_polymer = atom_array.is_protein | atom_array.is_nucleic_acid

    targets: set[tuple[str, int]] = set()
    for i, j, _bond_type in atom_array.bonds.as_array():
        if is_polymer[i] == is_polymer[j]:
            continue
        polymer_idx = i if is_polymer[i] else j
        # Non-standard residues are already atomized by AF3, so leave their identity alone.
        if str(atom_array.res_name[polymer_idx]) not in STANDARD_AA:
            continue
        targets.add((str(atom_array.chain_id[polymer_idx]), int(atom_array.res_id[polymer_idx])))

    if not targets:
        return atom_array
    renamed = atom_array.copy()
    for index, (chain_id, res_id) in enumerate(sorted(targets), start=1):
        mask = (renamed.chain_id == chain_id) & (renamed.res_id == res_id)
        renamed.res_name[mask] = _ATOMIZED_CODE_FORMAT.format(index=index)
    return renamed


class AF3Oracle(OracleMetric):
    """AlphaFold 3 oracle.

    Folds run via :py:meth:`fold_batch`: one AF3 subprocess per rank folds
    every applicable design in a single warm process. The per-item
    :py:meth:`compute` (inherited) then reads the cached confidence JSON.
    :py:meth:`fold_one` exists for symmetry with :class:`OracleMetric` but
    pays full per-batch overheads.

    Two dispatch routes (selected automatically by ``fold_batch``):

    - **Daemon route** (fast). When ``AF3_LAUNCHER_SOCKET_TEMPLATE`` is set,
      the rank's fold request is sent over the Unix socket served by
      ``src/rfproteina/af3/af3_daemon.py``, whose warm worker keeps AF3
      loaded across folds. The slurm launchers pre-spawn these daemons
      (``emit_af3_daemon_block`` in ``scripts/common.sh``); without a launcher,
      :py:meth:`setup` starts one for this rank.
    - **Direct route** (fallback). When the env var is unset, ``fold_batch``
      runs the same apptainer command directly via ``subprocess.run``,
      reloading AF3 on every call. The caller must be on a host where
      ``apptainer`` is on ``PATH``.

    Args:
        oracle_name: Output subdirectory label. Defaults to ``"af3"``.
        template_query: AtomWorks query string selecting which atoms to template
            (e.g. ``"mask_coordinate_1_atom & ~is_nucleic_acid"``).
        extra_af3_flags: Additional flags forwarded to the AF3 subprocess.
        seed: Pins AF3's diffusion seed for reproducibility; leave ``None`` in
            production so each fold samples its own.
    """

    def __init__(
        self,
        oracle_name: str = "af3",
        template_query: str | None = None,
        extra_af3_flags: list[str] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(oracle_name=oracle_name, template_query=template_query, **kwargs)
        self.extra_flags = extra_af3_flags or []
        self.seed = seed

    # ------------------------------------------------------------------
    # OracleMetric framework hooks
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Start this rank's warm AF3 daemon unless one is already available.

        Called only when this oracle's tags matched a validation set, so a run that
        never refolds never pays the worker's ~10 GB.
        """
        super().setup()

        # A slurm launcher already pre-spawned the daemons.
        if os.environ.get("AF3_LAUNCHER_SOCKET_TEMPLATE"):
            return

        local_rank = _resolve_local_rank()
        rank = _resolve_daemon_rank(local_rank)
        # Node-local: the socket must be, and the log is only useful while we're on it.
        sock_dir = Path(tempfile.mkdtemp(prefix="af3_daemon_"))
        sock = sock_dir / f"af3_daemon_rank{rank}.sock"
        log = sock_dir / f"af3_daemon_rank{rank}.log"
        with open(log, "wb") as log_fh:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(_AF3_DAEMON_SCRIPT),
                    "serve",
                    "--skip-if-no-env",
                    "--socket-path",
                    str(sock),
                    "--gpu-device",
                    str(_resolve_gpu_device(local_rank)),
                ],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )

        # `serve` binds before it loads AF3, so this waits about a second and the
        # minutes-long warmup then overlaps whatever the caller does next.
        while not sock.exists():
            if proc.poll() is not None:
                # Leave the env var unset so fold_batch falls back to the direct route.
                shutil.rmtree(sock_dir, ignore_errors=True)
                logger.info(f"No AF3 daemon started; folds will use the direct route. See {log}.")
                return
            time.sleep(0.1)

        atexit.register(_shutdown_daemon, proc, sock_dir)
        # Inherited by DDP ranks forked later; sibling oracles reuse this daemon.
        os.environ["AF3_LAUNCHER_SOCKET_TEMPLATE"] = f"{sock_dir}/af3_daemon_rank{{rank}}.sock"
        logger.info(f"AF3 daemon started for rank {rank} (log: {log})")

    def teardown(self) -> None:
        """Tear down :attr:`pre_oracle_transform`. A launcher's daemon dies with
        its slurm step; ours dies at exit."""
        super().teardown()
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
        return self._load_confidence_json(self.confidence_path(job.out_dir, job.example_id, job.save_id))

    def fold_batch(self, jobs: list[OracleJobInput]) -> int:
        """Fold every applicable design in one AF3 subprocess (per rank).

        Jobs already cached on disk are skipped. The remaining designs are
        serialised into a single JSON manifest, dispatched to AF3 via the
        daemon (Lightning DDP) or a direct ``apptainer run`` (ad-hoc), and
        post-processed back into the canonical ``confidence_path`` /
        ``refolded_cif_path`` locations.

        Returns:
          Number of jobs that have a refold on disk once this call returns,
          counting both cached hits and freshly folded designs.
        """
        pending = [j for j in jobs if not self.already_refolded(j.out_dir, j.example_id, j.save_id)]
        n_cached = len(jobs) - len(pending)
        if not pending:
            return n_cached

        daemon_rank = _resolve_daemon_rank(_resolve_local_rank())

        batch_dir = Path(tempfile.mkdtemp(prefix="af3_batch_"))
        af3_outputs_root = batch_dir / "af3_outputs"
        af3_outputs_root.mkdir(parents=True, exist_ok=True)

        manifest_entries, job_states = self._build_manifest(pending, af3_outputs_root)
        manifest_path = batch_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_entries))

        try:
            self._dispatch_fold(manifest_path, daemon_rank, len(pending))

            n_ok = 0
            for state in job_states:
                if self._finalize_af3_result(state) is not None:
                    n_ok += 1
            return n_cached + n_ok
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    def _dispatch_fold(self, manifest_path: Path, daemon_rank: int, n_designs: int) -> None:
        """Fold via the daemon (routes by daemon_rank, which pins the GPU) if
        AF3_LAUNCHER_SOCKET_TEMPLATE is set, else a direct apptainer call. Raises on failure."""
        sock_template = os.environ.get("AF3_LAUNCHER_SOCKET_TEMPLATE")

        if sock_template:
            daemon_sock = sock_template.format(rank=daemon_rank)
            # Extremely conservative limit of 2 min per design
            timeout_s = max(3600.0, n_designs * 120.0)
            logger.info(
                f"AF3 fold_batch: routing {n_designs} designs via daemon at {daemon_sock} "
                f"(timeout={timeout_s / 3600:.1f}h)"
            )
            response = _send_to_daemon(
                daemon_sock,
                {
                    "cmd": "fold",
                    "manifest": str(manifest_path),
                    "save_full_outputs": False,
                },
                timeout_s=timeout_s,
            )
            if response.get("status") != "ok":
                raise RuntimeError(f"AF3 daemon error: {response.get('message')}")
            return

        # Direct path: no pinned daemon worker, so resolve our own GPU.
        gpu_device = _resolve_gpu_device(_resolve_local_rank())
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
            if np.any(input_atom_array.res_name == "UNL"):
                raise ValueError(
                    f"Input structure for {job.example_id!r} contains residues named 'UNL'. "
                    "AF3 treats 'UNL' as a reserved token and silently drops most ligand atoms. "
                    "Rename UNL residues to a different code (e.g. 'L;G') before evaluating with AF3Oracle."
                )
            example_id = job.example_id
            # The save_id keeps sibling refolds of one design in AF3 output
            # subdirectories of their own, just as it does on the canonical side.
            job_name = (f"{example_id}_{job.save_id}" if job.save_id is not None else example_id).replace(" ", "_")

            # Disambiguate chain IDs so AF3 can distinguish chain_iid variants and keep an id_map for remapping outputs
            disambiguated_atom_array, af3_chain_id_to_original_iid = self._disambiguate_chain_ids(input_atom_array)

            # Match OF3's tokenization before anything derives from this array; AF3's own
            # weights learned the single-token convention, so gate it.
            if _using_of3_weights():
                disambiguated_atom_array = _atomize_ligand_bonded_residues(disambiguated_atom_array)

            # AF3's CIF parser refuses NaN coordinates; it pads them back and we reconcile later.
            valid_mask = np.all(np.isfinite(disambiguated_atom_array.coord), axis=-1)
            disambiguated_valid = disambiguated_atom_array[valid_mask]

            # Biotite's CIF writer requires occupancy + b_factor; designs have no real values.
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
            #   template     — structural template constraint (chain coords) for the design
            #   user_ccd     — connectivity for any non-standard residues so AF3 can bond them
            #   unpaired_msa — per-chain a3m for designs that fold better than single-sequence
            if template := self._build_af3_template_dict(disambiguated_valid):
                entry["template"] = template
            if user_ccd := self._build_user_ccd_string(disambiguated_valid):
                entry["user_ccd"] = user_ccd
            if unpaired_msa := self._build_unpaired_msa_dict(job, disambiguated_atom_array):
                entry["unpaired_msa"] = unpaired_msa
            if self.seed is not None:
                entry["seed"] = self.seed
            manifest_entries.append(entry)

            job_states.append(
                _AF3JobState(
                    input_atom_array=input_atom_array,
                    example_id=example_id,
                    save_id=job.save_id,
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

                # Normalize the AF3 output to match the input's atom set (strip extra OXTs,
                # strip sidechains for backbone/CA-only inputs) before mapping.
                predicted_atom_array = self._normalize_af3_output(disambiguated_partial, predicted_atom_array)

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
            cif_path = self.refolded_cif_path(state.out_dir, state.example_id, state.save_id)
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
        self._save_confidence_json(metrics, self.confidence_path(state.out_dir, state.example_id, state.save_id))
        return metrics

    # ------------------------------------------------------------------
    # AF3 input prep helpers
    # ------------------------------------------------------------------

    def _build_af3_template_dict(self, atom_array: AtomArray) -> dict[str, dict] | None:
        """Build the AF3 per-chain template dict from :attr:`template_query`.

        Args:
            atom_array: Input structure, with any query annotations set.

        Returns:
            Dict mapping chain ID → template dict, or ``None`` if the query is
            unset or selects no atoms.
        """
        if not self.template_query:
            return None

        template_mask = resolve_mask(atom_array, self.template_query)
        if not np.any(template_mask):
            return None

        ensure_annotations(atom_array, "is_protein")
        non_protein_templated = template_mask & ~atom_array.is_protein
        if np.any(non_protein_templated):
            bad_res = np.unique(atom_array.res_name[non_protein_templated])
            raise ValueError(
                f"Template query {self.template_query!r} selects non-protein residues: {bad_res}. "
                "AF3 structural templating is only supported for protein chains; "
                "exclude them in the query (e.g. '& ~is_nucleic_acid')."
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

    @staticmethod
    def _build_unpaired_msa_dict(job: OracleJobInput, disambiguated_atom_array: AtomArray) -> dict[str, str] | None:
        """Inline each chain's a3m, keyed by the chain ID AF3 will see.

        The MSA must travel as text: AF3 runs in its own container, and a path that
        does not resolve there is silently taken as the MSA itself
        (``run_alphafold_from_cif.py:_resolve_path_or_string``).

        Args:
            job: Queued design; ``msa_paths_by_chain_id`` is keyed by input ``chain_id``.
            disambiguated_atom_array: ``job.atom_array`` after chain-ID disambiguation.

        Returns:
            Dict mapping every AF3 chain ID → a3m text, or ``None`` when the job carries no MSA.

        Raises:
            ValueError: If a keyed chain ID is not in the design.
        """
        if not job.msa_paths_by_chain_id:
            return None

        # Disambiguation only renames chains, so the two arrays line up atom-for-atom.
        # One input chain can yield several AF3 chains (multi-copy); all of them need the a3m.
        pairs = set(zip(job.atom_array.chain_id, disambiguated_atom_array.chain_id, strict=True))

        unpaired_msa: dict[str, str] = {}
        for chain_id, msa_path in job.msa_paths_by_chain_id.items():
            af3_chain_ids = [af3 for original, af3 in pairs if original == chain_id]
            if not af3_chain_ids:
                raise ValueError(
                    f"MSA given for chain {chain_id!r}, which is not in design {job.example_id!r}; "
                    f"available chains: {sorted(set(job.atom_array.chain_id))}"
                )
            path = Path(msa_path)
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt") as fh:
                unpaired_msa.update(dict.fromkeys(af3_chain_ids, fh.read()))
        return unpaired_msa

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
        return _inject_empty_chem_comp_bond(buffer.getvalue())

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
        """Assert input and AF3 inter-residue bond graphs match, atomized-to-unatomized backbone links aside.

        Raises:
          ValueError: If the compared bond sets differ in either direction.
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
            res_name = renumbered.res_name
            for i, j, _btype in renumbered.bonds.as_array():
                if chain_id[i] == chain_id[j] and res_id[i] == res_id[j]:
                    continue
                if _is_implicit_backbone_link(str(res_name[i]), str(atom_name[i]), str(res_name[j]), str(atom_name[j])):
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
    def _normalize_af3_output(input_array: AtomArray, af3_array: AtomArray) -> AtomArray:
        """Strip AF3 output atoms that have no counterpart in the input.

        Two passes, applied in order so that both interact correctly:

        1. **OXT stripping** — AF3 adds OXT to chain C-termini; remove it for
           any residue where the input does not have an OXT
        2. **Per-residue sidechain trimming** — AF3 fills in atoms missing from
           the input. For canonical protein and nucleic acid residues, we remove these
           extra atoms, as they were likely unresolved in crystal-derived input conditioning.

        Args:
            input_array: Normalised input structure (after NaN-residue removal).
            af3_array: Parsed AF3 full-atom output.

        Returns:
            ``af3_array`` with extra atoms removed, ready for
            :py:meth:`_build_atom_index_mapping`.
        """
        # Pass 1: strip OXT atoms AF3 added but the input doesn't have.
        # We loop per chain since chains were disambiguated upstream.
        af3_oxt_mask = af3_array.atom_name == "OXT"
        if af3_oxt_mask.any():
            extra_oxt_mask = np.zeros_like(af3_oxt_mask)
            for chain in np.unique(af3_array.chain_id):
                af3_in = af3_array.chain_id == chain
                inp_in = input_array.chain_id == chain
                if not inp_in.any():
                    continue
                af3_res_oxt = apply_residue_wise(af3_array[af3_in], af3_oxt_mask[af3_in], np.any)
                inp_res_oxt = apply_residue_wise(input_array[inp_in], input_array.atom_name[inp_in] == "OXT", np.any)
                extra_per_res = af3_res_oxt & ~inp_res_oxt
                extra_oxt_mask[af3_in] = spread_residue_wise(af3_array[af3_in], extra_per_res) & af3_oxt_mask[af3_in]
            af3_array = af3_array[~extra_oxt_mask]

        # Pass 2: per-residue sidechain trimming
        if input_array.array_length() != af3_array.array_length():
            ensure_annotations(input_array, "is_standard_aa", "is_standard_dna", "is_standard_rna")
            is_standard_polymer = input_array.is_standard_aa | input_array.is_standard_dna | input_array.is_standard_rna

            inp_bounds = struc.get_residue_starts(input_array, add_exclusive_stop=True)
            af3_bounds = struc.get_residue_starts(af3_array, add_exclusive_stop=True)
            if len(inp_bounds) == len(af3_bounds):
                keep_mask = np.ones(af3_array.array_length(), dtype=bool)
                for i in range(len(inp_bounds) - 1):
                    i0, i1 = inp_bounds[i], inp_bounds[i + 1]
                    a0, a1 = af3_bounds[i], af3_bounds[i + 1]
                    n_extra = (a1 - a0) - (i1 - i0)
                    if n_extra <= 0 or not is_standard_polymer[i0]:
                        continue
                    inp_names = set(input_array.atom_name[i0:i1].tolist())
                    af3_names = af3_array.atom_name[a0:a1]
                    extra = ~np.isin(af3_names, list(inp_names))
                    if extra.sum() == n_extra:
                        keep_mask[a0:a1] = ~extra
                af3_array = af3_array[keep_mask]

        return af3_array

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
