"""AlphaFold 2 structure-prediction oracle."""

import atexit
import contextlib
import errno
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import weakref
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch.distributed as dist
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import CIFWriteConfig, to_cif_file, to_cif_string
from biotite.structure import AtomArray, concatenate
from biotite.structure.residues import apply_residue_wise, get_residue_starts, spread_residue_wise

from rfproteina.af2.af2_apptainer import DEFAULT_RESIDENT_MODELS, build_container_fold_command
from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput, OracleMetric, extract_confidence_metrics
from rfproteina.transforms.utils.conditioning_utils import align_atom_array_to_input_frame
from rfproteina.utils.chain import get_chain_of_interest
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.mask_spec import resolve_mask
from rfproteina.utils.standard_annotations import SA_CWI

logger = RankedLogger(__name__, rank_zero_only=True)

_AF2_DAEMON_SCRIPT = Path(__file__).resolve().parents[2] / "af2" / "af2_daemon.py"
_MONOMER_MODELS = tuple(f"model_{i}_ptm" for i in range(1, 6))
_MULTIMER_MODELS = tuple(f"model_{i}_multimer_v3" for i in range(1, 6))
_OFFICIAL_MODELS = frozenset((*_MONOMER_MODELS, *_MULTIMER_MODELS))
_AF2_ORACLE_INSTANCES: weakref.WeakSet = weakref.WeakSet()


def _resident_models_for_owned_daemon(required: tuple[str, ...], explicit: str | None) -> str | None:
    """Return a validated resident set for an oracle-owned daemon."""
    if not required:
        return explicit
    if not explicit:
        return ",".join(required)

    resident = {name.strip() for name in explicit.split(",") if name.strip()}
    missing = [name for name in required if name not in resident]
    if missing:
        raise RuntimeError("AF2_RESIDENT_MODELS is missing required AF2 models: " + ",".join(missing))
    return explicit


def _effective_chain_count(atom_array: AtomArray) -> int:
    """Count chains that retain at least one atom with finite coordinates."""
    finite = np.all(np.isfinite(atom_array.coord), axis=-1)
    return len(np.unique(atom_array.chain_id[finite]))


def _shutdown_daemon(proc: subprocess.Popen, sock_path: Path, sock_dir: Path) -> None:
    """Gracefully stop an oracle-owned daemon, then force it only if needed."""
    try:
        _send_to_daemon(
            str(sock_path),
            {"cmd": "shutdown"},
            timeout_s=10.0,
            connect_retry_s=0.0,
        )
        proc.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
    shutil.rmtree(sock_dir, ignore_errors=True)


def _resolve_local_rank() -> int:
    local = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
    return int(local) if local is not None else 0


def _resolve_daemon_rank(local_rank: int) -> int:
    explicit = os.environ.get("AF2_DAEMON_RANK")
    return int(explicit) if explicit is not None else local_rank


def _resolve_gpu_device(local_rank: int) -> int:
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
    """Send one request, retrying while a daemon starts or its backlog is full."""
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
            if (
                exc.errno in (errno.ENOENT, errno.ECONNREFUSED, errno.EAGAIN, errno.EWOULDBLOCK)
                and time.monotonic() < deadline
            ):
                time.sleep(connect_retry_interval_s)
                continue
            raise
        finally:
            sock.close()


@dataclass(slots=True)
class _AF2JobState:
    input_atom_array: AtomArray
    example_id: str
    save_id: str | None
    out_dir: Path
    disambiguated_atom_array: AtomArray
    af2_chain_id_to_original_iid: dict[str, str] | None
    af2_output_dir: Path
    job_name: str
    chain_of_interest: str | None
    model_name: str | None


class AF2Oracle(OracleMetric):
    """Protein-only AlphaFold 2 refolding oracle.

    Batches are routed to a resident AF2 daemon when
    ``AF2_LAUNCHER_SOCKET_TEMPLATE`` is set, otherwise they use a direct
    container-backend invocation. ``model_name=None`` selects monomer or multimer
    weights in the worker from the input chain count.
    """

    def __init__(
        self,
        oracle_name: str = "af2",
        template_query: str | None = None,
        model_name: str | None = None,
        seed: int | None = None,
        extra_af2_flags: list[str] | None = None,
        initial_guess: bool = False,
        pct_seq_mask: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(oracle_name=oracle_name, template_query=template_query, **kwargs)
        if not 0.0 <= float(pct_seq_mask) <= 1.0:
            raise ValueError(f"pct_seq_mask must be in [0, 1], got {pct_seq_mask!r}")
        if model_name is not None and model_name not in _OFFICIAL_MODELS:
            raise ValueError(f"unsupported official AF2 model: {model_name!r}")

        self.model_name = model_name
        self.seed = seed
        self.extra_flags = extra_af2_flags or []
        self.initial_guess = bool(initial_guess)
        self.pct_seq_mask = float(pct_seq_mask)
        _AF2_ORACLE_INSTANCES.add(self)

    def required_model_names(self) -> tuple[str, ...]:
        """Return the official model weights this oracle may request."""
        if self.model_name is not None:
            return (self.model_name,)
        return DEFAULT_RESIDENT_MODELS

    @staticmethod
    def _process_required_model_names() -> tuple[str, ...]:
        """Union resident requirements for every daemon-backed oracle instance."""
        required = {
            model_name
            for oracle in tuple(_AF2_ORACLE_INSTANCES)
            if not oracle.extra_flags
            for model_name in oracle.required_model_names()
        }
        return tuple(model_name for model_name in (*_MONOMER_MODELS, *_MULTIMER_MODELS) if model_name in required)

    def setup(self) -> None:
        """Start this rank's warm daemon unless a launcher supplied one."""
        super().setup()
        if self.extra_flags:
            logger.info("AF2 extra_af2_flags require direct dispatch; not starting an owned daemon.")
            return
        if os.environ.get("AF2_LAUNCHER_SOCKET_TEMPLATE"):
            configured = os.environ.get("AF2_RESIDENT_MODELS") or ",".join(DEFAULT_RESIDENT_MODELS)
            _resident_models_for_owned_daemon(self._process_required_model_names(), configured)
            return

        local_rank = _resolve_local_rank()
        rank = _resolve_daemon_rank(local_rank)
        sock_dir = Path(tempfile.mkdtemp(prefix="af2_daemon_"))
        sock = sock_dir / f"af2_daemon_rank{rank}.sock"
        log = sock_dir / f"af2_daemon_rank{rank}.log"
        resident_models = _resident_models_for_owned_daemon(
            self._process_required_model_names(),
            os.environ.get("AF2_RESIDENT_MODELS"),
        )
        daemon_env = os.environ.copy()
        if resident_models:
            daemon_env["AF2_RESIDENT_MODELS"] = resident_models
            os.environ["AF2_RESIDENT_MODELS"] = resident_models
        with open(log, "wb") as log_fh:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(_AF2_DAEMON_SCRIPT),
                    "serve",
                    "--skip-if-no-env",
                    "--socket-path",
                    str(sock),
                    "--gpu-device",
                    str(_resolve_gpu_device(local_rank)),
                ],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=daemon_env,
            )

        while not sock.exists():
            if proc.poll() is not None:
                shutil.rmtree(sock_dir, ignore_errors=True)
                logger.info(f"No AF2 daemon started; folds will use the direct route. See {log}.")
                return
            time.sleep(0.1)

        atexit.register(_shutdown_daemon, proc, sock, sock_dir)
        os.environ["AF2_LAUNCHER_SOCKET_TEMPLATE"] = f"{sock_dir}/af2_daemon_rank{{rank}}.sock"
        logger.info(f"AF2 daemon started for rank {rank} (log: {log})")

    def teardown(self) -> None:
        super().teardown()
        if dist.is_initialized():
            dist.barrier()

    def fold_one(self, job: OracleJobInput) -> dict:
        n_ok = self.fold_batch([job])
        if n_ok != 1:
            raise RuntimeError(f"AF2Oracle.fold_one: fold_batch returned {n_ok} for {job.example_id!r}")
        return self._load_confidence_json(self.confidence_path(job.out_dir, job.example_id, job.save_id))

    def raw_summary_path(self, out_dir: Path, example_id: str, save_id: str | None = None) -> Path:
        """Return the canonical sharded AF2 worker-summary artifact path."""
        confidence_path = self.confidence_path(out_dir, example_id, save_id)
        return confidence_path.with_name(f"{self.output_stem(example_id, save_id)}_summary_confidences.json")

    def already_refolded(self, out_dir: Path, example_id: str, save_id: str | None = None) -> bool:
        confidence_path = self.confidence_path(out_dir, example_id, save_id)
        if not confidence_path.is_file():
            return False
        try:
            cached_model_name = self._load_confidence_json(confidence_path).get("af2_model_name")
        except (OSError, TypeError, ValueError):
            return False
        return cached_model_name is not None and (self.model_name is None or cached_model_name == self.model_name)

    def fold_batch(self, jobs: list[OracleJobInput]) -> int:
        pending = [job for job in jobs if not self.already_refolded(job.out_dir, job.example_id, job.save_id)]
        n_cached = len(jobs) - len(pending)
        if not pending:
            return n_cached

        daemon_rank = _resolve_daemon_rank(_resolve_local_rank())
        batch_dir = Path(tempfile.mkdtemp(prefix="af2_batch_"))
        outputs_root = batch_dir / "af2_outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        manifest, states = self._build_manifest(pending, outputs_root)
        manifest_path = batch_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        try:
            self._dispatch_fold(manifest_path, daemon_rank, len(pending))
            n_ok = sum(self._finalize_af2_result(state) is not None for state in states)
            return n_cached + n_ok
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    def _dispatch_fold(self, manifest_path: Path, daemon_rank: int, n_designs: int) -> None:
        if capture_dir_raw := os.environ.get("AF2_MANIFEST_CAPTURE_DIR"):
            capture_dir = Path(capture_dir_raw)
            capture_dir.mkdir(parents=True, exist_ok=True)
            captured_manifest = capture_dir / f"{manifest_path.parent.name}-{manifest_path.name}"
            shutil.copy2(manifest_path, captured_manifest)
            logger.info(f"AF2 fold_batch: captured dispatched manifest at {captured_manifest}")

        sock_template = os.environ.get("AF2_LAUNCHER_SOCKET_TEMPLATE")
        if sock_template:
            if self.extra_flags:
                raise RuntimeError(
                    "AF2 extra_af2_flags cannot change JIT flags through a prestarted daemon; "
                    "use direct execution or start the daemon with the desired flags."
                )
            daemon_sock = sock_template.format(rank=daemon_rank)
            fold_timeout_s = max(3600.0, n_designs * 120.0)
            logger.info(f"AF2 fold_batch: routing {n_designs} designs via daemon at {daemon_sock}")
            response = _send_to_daemon(
                daemon_sock,
                {
                    "cmd": "fold",
                    "manifest": str(manifest_path),
                    "timeout_s": fold_timeout_s,
                    "deadline_unix_s": time.time() + fold_timeout_s,
                },
                timeout_s=fold_timeout_s + 30.0,
            )
            if response.get("status") != "ok":
                raise RuntimeError(f"AF2 daemon error: {response.get('message')}")
            return

        gpu_device = _resolve_gpu_device(_resolve_local_rank())
        logger.info(f"AF2 fold_batch: routing {n_designs} designs via direct container call")
        resident_models = _resident_models_for_owned_daemon(
            self.required_model_names(),
            os.environ.get("AF2_RESIDENT_MODELS"),
        )
        cmd = build_container_fold_command(
            manifest_path=manifest_path,
            gpu_device=gpu_device,
            resident_models=resident_models,
        )
        cmd.extend(self.extra_flags)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            raise RuntimeError(f"AF2 container subprocess failed with rc={rc}")

    def _build_manifest(
        self,
        pending: list[OracleJobInput],
        outputs_root: Path,
    ) -> tuple[list[dict], list[_AF2JobState]]:
        entries: list[dict] = []
        states: list[_AF2JobState] = []
        for job in pending:
            input_atom_array = job.atom_array
            if np.any(input_atom_array.res_name == "UNL"):
                raise ValueError(f"Input structure for {job.example_id!r} contains unsupported residue 'UNL'.")
            ensure_annotations(input_atom_array, "is_protein")
            if not np.all(input_atom_array.is_protein):
                bad_res = np.unique(input_atom_array.res_name[~input_atom_array.is_protein])
                raise ValueError(
                    f"AF2Oracle only supports proteins; {job.example_id!r} contains non-protein residues: {bad_res}."
                )

            # The save_id keeps sibling refolds of one design in distinct AF2 worker
            # output dirs, mirroring AF3Oracle's naming.
            job_name = (f"{job.example_id}_{job.save_id}" if job.save_id is not None else job.example_id).replace(
                " ", "_"
            )
            disambiguated, chain_map = self._disambiguate_chain_ids(input_atom_array)
            valid = disambiguated[np.all(np.isfinite(disambiguated.coord), axis=-1)]
            valid.set_annotation("occupancy", np.ones(valid.array_length()))
            valid.set_annotation("b_factor", np.zeros(valid.array_length()))
            valid = self._renumber_res_ids_per_chain(valid)
            output_dir = outputs_root / job_name
            chain_count = _effective_chain_count(disambiguated)
            if chain_count == 0:
                raise ValueError(f"AF2 input {job.example_id!r} has no atoms with finite coordinates.")
            if self.model_name in _MONOMER_MODELS and chain_count != 1:
                raise ValueError(f"AF2 model {self.model_name!r} requires a monomer input, got {chain_count} chains.")
            if self.model_name in _MULTIMER_MODELS and chain_count < 2:
                raise ValueError(f"AF2 model {self.model_name!r} requires a multimer input, got {chain_count} chain.")
            entry: dict[str, Any] = {
                "name": job_name,
                "output_dir": str(output_dir),
                "cif": to_cif_string(valid.copy(), config=CIFWriteConfig(id=job_name)),
                "chain_count": chain_count,
                "num_res": len(get_residue_starts(valid)),
                "initial_guess": bool(self.initial_guess),
                "pct_seq_mask": float(self.pct_seq_mask),
            }
            if template := self._build_af2_template_dict(valid):
                entry["template"] = template
            if self.seed is not None:
                entry["seed"] = self.seed
            if self.model_name is not None:
                entry["model_name"] = self.model_name
            entries.append(entry)
            states.append(
                _AF2JobState(
                    input_atom_array=input_atom_array,
                    example_id=job.example_id,
                    save_id=job.save_id,
                    out_dir=job.out_dir,
                    disambiguated_atom_array=disambiguated,
                    af2_chain_id_to_original_iid=chain_map,
                    af2_output_dir=output_dir,
                    job_name=job_name,
                    chain_of_interest=job.chain_of_interest,
                    model_name=self.model_name,
                )
            )
        return entries, states

    def _build_af2_template_dict(self, atom_array: AtomArray) -> dict[str, dict] | None:
        if not self.template_query:
            return None
        template_mask = resolve_mask(atom_array, self.template_query)
        if not np.any(template_mask):
            return None

        template_dict: dict[str, dict] = {}
        for chain_id in np.unique(atom_array.chain_id):
            chain_mask = atom_array.chain_id == chain_id
            chain_atoms = atom_array[chain_mask]
            selected = template_mask[chain_mask]
            if not np.any(selected):
                continue
            selected_all = apply_residue_wise(chain_atoms, selected, np.all)
            selected_any = apply_residue_wise(chain_atoms, selected, np.any)
            if not np.array_equal(selected_all, selected_any):
                logger.warning(
                    f"Some residues on chain {chain_id} are partially templated; " "AF2 will template the full residue."
                )
            indices = np.where(selected_any)[0].tolist()
            template_atoms = chain_atoms.copy()
            if SA_CWI.has_annotation(template_atoms):
                conditioned_coords = SA_CWI.annotation(template_atoms)
                if conditioned_coords.ndim == 3:
                    conditioned_coords = conditioned_coords[0]
                use_conditioned = selected & np.all(np.isfinite(conditioned_coords), axis=-1)
                template_atoms.coord[use_conditioned] = conditioned_coords[use_conditioned]
            config = CIFWriteConfig(
                id=f"template_chain_{chain_id}",
                extra_categories={"pdbx_audit_revision_history": {"revision_date": date.today().isoformat()}},
            )
            template_dict[chain_id] = {
                "mmcif": to_cif_string(template_atoms, config=config),
                "query_indices": indices,
                "template_indices": indices,
            }
        return template_dict or None

    def _finalize_af2_result(self, state: _AF2JobState) -> dict | None:
        summary = self._parse_af2_confidences(state.af2_output_dir, state.job_name)
        if summary is None:
            raise RuntimeError(
                f"AF2 produced no summary_confidences for {state.example_id!r} "
                f"(expected {state.af2_output_dir}/{state.job_name}_summary_confidences.json)."
            )
        actual_model_name = self._validated_model_name(
            summary.get("af2_model_name"),
            state.model_name,
            state.example_id,
        )
        cif_path = self.refolded_cif_path(state.out_dir, state.example_id, state.save_id)

        predicted = self._parse_af2_predicted_structure(state.af2_output_dir, state.job_name)
        if predicted is not None:
            try:
                predicted = self._label_af2_output_chains(predicted, summary.get("chain_ids", []))
                nan_mask = np.isnan(state.disambiguated_atom_array.coord).any(axis=-1)
                missing_residue_mask = apply_residue_wise(state.disambiguated_atom_array, nan_mask, np.all)
                missing_residue_mask = spread_residue_wise(state.disambiguated_atom_array, missing_residue_mask)
                partial = state.disambiguated_atom_array[~missing_residue_mask]
                # AF2 emits chains in post-assembly order (e.g. A,C,B for identical
                # A/C sequences). Normalize/mapping walk residues in array order, so
                # reorder predicted atoms to match the input chain sequence first.
                predicted = self._reorder_chains_like(partial, predicted)
                predicted = self._normalize_af2_output(partial, predicted)
                # AF2 emits no OXT, so an input carrying one (a target read from a crystal
                # structure, say) has an atom the output cannot match. Drop those from the
                # mapping and NaN them: keeping the input coordinate would strand a C-terminal
                # atom at its pre-refold position, a chainbreak against its own residue.
                omitted = partial.atom_name == "OXT"
                if omitted.any():
                    logger.warning(
                        f"AF2 emits no OXT; writing {int(omitted.sum())} input OXT atom(s) of "
                        f"{state.example_id!r} as NaN. RMSDs whose mask covers them are skipped."
                    )
                partial = partial[~omitted]
                self._validate_inter_residue_bonds(partial, predicted)
                input_to_af2 = self._build_atom_index_mapping(partial, predicted)
                result = as_atom_array_plus(state.input_atom_array.copy())
                kept = np.where(~missing_residue_mask)[0]
                result.coord[kept[~omitted]] = predicted.coord[input_to_af2]
                result.coord[kept[omitted]] = np.nan
                predicted = result
            except (ValueError, KeyError) as exc:
                logger.warning(f"AF2 coord transfer failed for {state.example_id}: {exc}")
                predicted = None

        if predicted is not None:
            cif_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                to_cif_file(align_atom_array_to_input_frame(predicted), cif_path)
            except Exception as exc:
                logger.warning(f"Failed to save AF2 refolded CIF for {state.example_id}: {exc}")
                cif_path.unlink(missing_ok=True)
        else:
            cif_path.unlink(missing_ok=True)

        chain_of_interest = state.chain_of_interest
        if chain_of_interest is None:
            chain_of_interest, unambiguous = get_chain_of_interest(state.input_atom_array)
            if not unambiguous:
                chain_of_interest = None
        af2_chain_ids = summary.get("chain_ids", [])
        if state.af2_chain_id_to_original_iid is not None:
            chains_in_order = [state.af2_chain_id_to_original_iid.get(chain) for chain in af2_chain_ids]
        else:
            chains_in_order = af2_chain_ids
        metrics = extract_confidence_metrics(
            summary_confidences=summary,
            chain_of_interest=chain_of_interest,
            chains_in_order=chains_in_order,
        )
        metrics["af2_model_name"] = actual_model_name
        metrics["chain_of_interest"] = chain_of_interest
        raw_summary_path = self.raw_summary_path(state.out_dir, state.example_id, state.save_id)
        raw_summary_path.parent.mkdir(parents=True, exist_ok=True)
        raw_summary_path.write_text(json.dumps(summary))
        self._save_confidence_json(metrics, self.confidence_path(state.out_dir, state.example_id, state.save_id))
        return metrics

    @staticmethod
    def _validated_model_name(
        actual_model_name: str | None,
        requested_model_name: str | None,
        example_id: str,
    ) -> str:
        """Require the worker to identify exactly the model allocated to this request."""
        if actual_model_name is None:
            raise RuntimeError(f"AF2 confidence output for {example_id!r} does not identify the selected model.")
        if requested_model_name is not None and actual_model_name != requested_model_name:
            raise RuntimeError(
                f"AF2 returned model {actual_model_name!r} for a request allocated to {requested_model_name!r}."
            )
        return actual_model_name

    @staticmethod
    def _label_af2_output_chains(predicted: AtomArray, original_chain_ids: list[str]) -> AtomArray:
        """Replace AF2's positional chain labels with post-assembly auth chain IDs."""
        _, first_occurrence = np.unique(predicted.chain_id, return_index=True)
        emitted_chain_ids = predicted.chain_id[np.sort(first_occurrence)].tolist()
        if len(emitted_chain_ids) != len(original_chain_ids):
            raise ValueError(
                f"AF2 output has {len(emitted_chain_ids)} chains but confidence metadata has "
                f"{len(original_chain_ids)}"
            )
        if len(set(original_chain_ids)) != len(original_chain_ids):
            raise ValueError(f"AF2 confidence metadata has duplicate chain IDs: {original_chain_ids}")

        relabeled = predicted.copy()
        emitted_labels = predicted.chain_id.copy()
        for emitted, original in zip(emitted_chain_ids, original_chain_ids, strict=True):
            relabeled.chain_id[emitted_labels == emitted] = original
        return relabeled

    @staticmethod
    def _reorder_chains_like(reference: AtomArray, predicted: AtomArray) -> AtomArray:
        """Reorder ``predicted`` chains to match ``reference`` chain appearance order.

        AF2 may emit chains in assembly order that differs from the input CIF
        (identical sequences are grouped). Downstream normalize/mapping helpers
        assume matching global residue order, so restore input chain order here.
        """
        _, ref_first = np.unique(reference.chain_id, return_index=True)
        reference_order = reference.chain_id[np.sort(ref_first)].tolist()
        _, pred_first = np.unique(predicted.chain_id, return_index=True)
        predicted_order = predicted.chain_id[np.sort(pred_first)].tolist()
        if predicted_order == reference_order:
            return predicted
        if set(predicted_order) != set(reference_order):
            raise ValueError(f"AF2 chain IDs {predicted_order} do not match input chain IDs {reference_order}")
        segments = [predicted[predicted.chain_id == chain_id] for chain_id in reference_order]
        return concatenate(segments)

    @staticmethod
    def _parse_af2_predicted_structure(output_dir: Path, job_name: str) -> AtomArray | None:
        cif_path = output_dir / f"{job_name}_model.cif"
        if not cif_path.exists():
            logger.warning(f"AF2 output CIF not found: {cif_path}")
            return None
        config = ParseConfig.from_preset("minimal", extra_fields=("b_factor", "occupancy"))
        return as_atom_array_plus(parse(str(cif_path), config=config)["asym_unit"][0])

    @staticmethod
    def _parse_af2_confidences(output_dir: Path, job_name: str) -> dict | None:
        json_path = output_dir / f"{job_name}_summary_confidences.json"
        if not json_path.exists():
            logger.warning(f"AF2 summary confidences JSON not found: {json_path}")
            return None
        with open(json_path) as handle:
            return json.load(handle)

    # AF2 and AF3 emit the same chain/residue/atom organization. Keep these
    # mature remapping helpers shared until they move to a neutral module.
    _disambiguate_chain_ids = staticmethod(AF3Oracle._disambiguate_chain_ids)
    _renumber_res_ids_per_chain = staticmethod(AF3Oracle._renumber_res_ids_per_chain)
    _validate_inter_residue_bonds = staticmethod(AF3Oracle._validate_inter_residue_bonds)
    _normalize_af2_output = staticmethod(AF3Oracle._normalize_af3_output)
    _build_atom_index_mapping = staticmethod(AF3Oracle._build_atom_index_mapping)
