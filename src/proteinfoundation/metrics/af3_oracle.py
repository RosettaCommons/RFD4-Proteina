"""AF3 structure prediction oracle.

Folds one designed structure with AlphaFold 3 and returns confidence scores.
Inherits resumability from OracleMetric.

AF3 weights are loaded once via a persistent server subprocess started in
:py:meth:`AF3Oracle.setup` and shut down in :py:meth:`AF3Oracle.teardown`.
Multiple oracle instances with the same configuration share one server process.
AF3 is not differentiable — it cannot propagate gradients back to the input.
"""

import io
import json
import os
import select
import subprocess
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import biotite.structure as struc
import numpy as np
import torch
import torch.distributed
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.io.utils.io_utils import CIFWriteConfig, _to_cif_or_bcif, to_cif_file, to_cif_string
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile
from biotite.structure.residues import apply_residue_wise, residue_iter, spread_residue_wise

from foundry.utils.ddp import RankedLogger
from proteinfoundation.metrics.oracle_metric import OracleMetric
from proteinfoundation.utils.chain_utils import get_chain_of_interest

logger = RankedLogger(__name__, rank_zero_only=True)

_EMPTY_METRICS = {"overall_plddt": 0.0, "overall_ptm": 0.0, "overall_iptm": 0.0, "ranking_score": 0.0}


class AF3Oracle(OracleMetric):
    """AlphaFold 3 structure prediction oracle (not differentiable).

    Launches a persistent AF3 server subprocess on :py:meth:`setup` so that
    model weights are loaded once and reused across all fold calls.  Falls back
    to a one-shot subprocess when no server is running (e.g. in tests).

    Saves the refolded CIF via the canonical path from
    :py:meth:`refolded_cif_path` and returns all AF3 confidence metrics.

    Args:
        num_recycles: Number of recycling iterations. Defaults to ``5``.
        oracle_name: Output subdirectory label. Defaults to ``"af3"``.
        template_annotations: Annotation name(s) to use for templating.
            Defaults to ``None``.
        extra_af3_flags: Additional flags to pass to the AF3 subprocess.

    Example:
        >>> oracle = AF3Oracle(num_recycles=5)
        >>> result = oracle.compute(
        ...     input_atom_array=structure,
        ...     example_id="design_001",
        ...     out_dir=Path("/outputs"),
        ... )
        >>> # Returns: {"overall_plddt": 85.0, "overall_iptm": 90.0, ...}
    """

    # Class-level server pool.  Each entry is a persistent apptainer subprocess
    # that has already loaded AF3 weights.  Keys are tuples of the compile-time
    # parameters that distinguish server configurations (num_recycles, …).
    # Reference counts track how many AF3Oracle instances share each server so
    # that teardown() only kills the process when the last user is done.
    _server_cache: ClassVar[dict[tuple, subprocess.Popen]] = {}
    _server_ref_counts: ClassVar[dict[tuple, int]] = {}

    # Maximum seconds to wait for any single response from the AF3 server.
    # Applied both to the initial READY signal (weight loading + JIT warmup)
    # and to every per-structure inference response.
    _SERVER_IO_TIMEOUT: ClassVar[float] = 600.0

    def __init__(
        self,
        num_recycles: int = 5,
        oracle_name: str = "af3",
        template_annotations: str | list[str] | None = None,
        extra_af3_flags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(oracle_name=oracle_name, template_annotations=template_annotations, **kwargs)
        self.num_recycles = num_recycles
        self.extra_flags = extra_af3_flags or []

    @property
    def _server_key(self) -> tuple:
        """Unique key identifying this oracle's AF3 server configuration.

        Only :attr:`num_recycles` is used because it is compiled into the JAX
        XLA graph at trace time and cannot be changed without reloading the model.
        """
        return (self.num_recycles,)

    @staticmethod
    def _readline_timeout(file_obj: Any, timeout: float, context: str) -> str:
        """Read one line from *file_obj*, raising :exc:`TimeoutError` if it takes too long.

        Args:
            file_obj: Readable file object backed by a file descriptor.
            timeout: Maximum seconds to wait for data to arrive.
            context: Short description of what we are waiting for.

        Returns:
            The line read from *file_obj*, or an empty string on EOF.

        Raises:
            TimeoutError: If no data arrives within *timeout* seconds.
        """
        ready = select.select([file_obj], [], [], timeout)[0]
        if not ready:
            raise TimeoutError(
                f"AF3 server did not respond after {timeout:.0f}s "
                f"while waiting for: {context}. "
                "The server process may have crashed or hung — check stderr logs."
            )
        return file_obj.readline()

    def _build_af3_server_command(self) -> list[str]:
        """Build the apptainer command that launches the persistent AF3 server.

        The GPU device is chosen based on ``LOCAL_RANK`` so that each DDP rank
        uses its own GPU.

        Returns:
            Command list suitable for :class:`subprocess.Popen`.
        """
        server_script_path = self.scripts_dir / "run_alphafold_server.py"

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        # If the launcher already masked CUDA_VISIBLE_DEVICES to a single GPU,
        # the only valid device index inside the container is 0.
        gpu_device = 0 if (cuda_visible and "," not in cuda_visible) else local_rank

        # Bind the host TMPDIR so the server can write AF3 output files there.
        # tempfile.TemporaryDirectory() (used in fold_single) creates dirs under
        # this path, and the server receives those paths via the JSON protocol.
        host_tmpdir = tempfile.gettempdir()

        cmd = [
            "apptainer",
            "run",
            "--nv",
            # Force JAX not to pre-allocate GPU memory.  Using apptainer's
            # --env flag is required because the container's %environment
            # section can override values inherited from the parent process.
            "--env",
            "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "--bind",
            str(self.model_dir),
            "--bind",
            host_tmpdir,
            "--bind",
            str(self.scripts_dir),
        ]
        if self.jax_cache_dir:
            cmd += ["--bind", self.jax_cache_dir]
        cmd += [
            self.apptainer_path,
            "python",
            str(server_script_path),
            f"--num_recycles={self.num_recycles}",
            "--num_diffusion_samples=1",
            "--save_full_outputs=false",
            f"--gpu_device={gpu_device}",
        ]
        if self.jax_cache_dir:
            cmd.append(f"--jax_compilation_cache_dir={self.jax_cache_dir}")
        for flag in self.extra_flags:
            if not flag.startswith("--gpu_device="):
                cmd.append(flag)
        return cmd

    def setup(self) -> None:
        """Launch the persistent AF3 server subprocess (once per configuration).

        Called automatically by the foundry framework before validation begins.
        Increments a reference count so that multiple :class:`AF3Oracle`
        instances sharing the same configuration share one server process.
        Blocks until the server signals it has finished loading weights.
        """
        # Resolve and validate all fixed env vars once — fail loudly here rather than at fold time.
        scripts_dir = os.environ.get("AF3_SCRIPTS_DIR")
        if not scripts_dir or not os.path.isdir(scripts_dir):
            raise RuntimeError(
                f"AF3_SCRIPTS_DIR is not set or does not exist: {scripts_dir!r}. "
                "Set it to the directory containing run_alphafold_server.py and run_alphafold_from_cif.py."
            )
        self.scripts_dir = Path(scripts_dir)

        apptainer_path = os.environ.get("AF3_APPTAINER_PATH", "/path/to/shared/storage")
        if not os.path.isfile(apptainer_path):
            raise RuntimeError(f"AF3 apptainer image not found: {apptainer_path!r} (set AF3_APPTAINER_PATH)")
        self.apptainer_path = apptainer_path

        model_dir = Path(os.environ.get("AF3_MODEL_DIR", "/path/to/shared/storage"))
        if not model_dir.is_dir():
            raise RuntimeError(f"AF3 model directory not found: {model_dir!r} (set AF3_MODEL_DIR)")
        self.model_dir = model_dir

        self.jax_cache_dir: str | None = os.environ.get("AF3_JAX_CACHE_DIR")

        key = self._server_key
        AF3Oracle._server_ref_counts[key] = AF3Oracle._server_ref_counts.get(key, 0) + 1
        if key in AF3Oracle._server_cache:
            return  # another instance already started this server

        cmd = self._build_af3_server_command()
        gpu_device_flag = next((f for f in cmd if f.startswith("--gpu_device=")), "--gpu_device=?")
        logger.info(f"Launching AF3 server (num_recycles={self.num_recycles}, " f"{gpu_device_flag}): {' '.join(cmd)}")

        # Release PyTorch's cached-but-free CUDA allocations before AF3 starts.
        # PyTorch's caching allocator never voluntarily returns memory to CUDA,
        # so without this call AF3 (JAX) would see artificially reduced free
        # VRAM and be more likely to OOM.
        torch.cuda.empty_cache()

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit parent stderr so AF3 logs appear in training logs
            text=True,
            bufsize=1,  # line-buffered for responsive protocol messages
            env=os.environ.copy(),
        )
        ready_line = self._readline_timeout(
            proc.stdout,
            timeout=AF3Oracle._SERVER_IO_TIMEOUT,
            context="READY signal (AF3 weight loading and JIT warmup)",
        )
        if ready_line.strip() != "READY":
            returncode = proc.wait(timeout=10)
            raise RuntimeError(
                f"AF3 server failed to start (exit code {returncode}). " f"Expected 'READY', got: {ready_line!r}"
            )
        AF3Oracle._server_cache[key] = proc
        logger.info("AF3 server is ready.")

    def teardown(self) -> None:
        """Shut down the AF3 server and synchronise ranks.

        Called automatically by the foundry framework after validation ends.
        Decrements the reference count; the server process is terminated only
        when the count reaches zero.
        """
        key = self._server_key
        count = AF3Oracle._server_ref_counts.get(key, 0) - 1
        AF3Oracle._server_ref_counts[key] = max(count, 0)
        if count <= 0:
            proc = AF3Oracle._server_cache.pop(key, None)
            AF3Oracle._server_ref_counts.pop(key, None)
            if proc is not None:
                try:
                    proc.stdin.write('{"action": "exit"}\n')
                    proc.stdin.flush()
                    proc.stdin.close()
                    proc.wait(timeout=30)
                except Exception:
                    proc.kill()
                logger.info("AF3 server shut down.")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------

    def _build_af3_command(self, job_output_dir: Path) -> list[str]:
        """Build the AF3 subprocess command.

        Args:
            job_output_dir: Output directory for this AF3 job.

        Returns:
            Command list suitable for :func:`subprocess.run`.
        """
        script_path = self.scripts_dir / "run_alphafold_from_cif.py"
        cmd = [
            "apptainer",
            "run",
            "--nv",
            "--bind",
            str(self.model_dir),
            "--bind",
            str(job_output_dir.parent),  # ensure parent is writable inside container
            "--bind",
            str(self.scripts_dir),
        ]
        if self.jax_cache_dir:
            cmd += ["--bind", self.jax_cache_dir]
        cmd += [
            self.apptainer_path,
            "python",
            str(script_path),
            f"--output_dir={job_output_dir}",
            f"--num_recycles={self.num_recycles}",
            "--num_diffusion_samples=1",
            "--save_full_outputs=false",
            "--cif_info_stdin_json=-",
        ]
        if self.jax_cache_dir:
            cmd.append(f"--jax_compilation_cache_dir={self.jax_cache_dir}")
        cmd.extend(self.extra_flags)
        return cmd

    def _launch_af3_subprocess(
        self,
        cif_string: str,
        job_output_dir: Path,
        paired_msa: dict[str, str] | None = None,
        unpaired_msa: dict[str, str] | None = None,
        template: dict[str, Any] | None = None,
        user_ccd: str | None = None,
    ) -> bool:
        """Launch the AF3 subprocess and wait for completion.

        Args:
            cif_string: CIF-formatted input structure as a string.
            job_output_dir: Output directory for this AF3 job.
            paired_msa: Per-chain paired MSA dict.
            unpaired_msa: Per-chain unpaired MSA dict.
            template: Per-chain template dict.
            user_ccd: Multi-block CIF string for non-standard residues.

        Returns:
            True if the subprocess exited successfully, False otherwise.
        """
        cmd = self._build_af3_command(job_output_dir=job_output_dir)

        stdin_payload: dict[str, Any] = {"cif": cif_string}
        if paired_msa:
            stdin_payload["paired_msa"] = paired_msa
        if unpaired_msa:
            stdin_payload["unpaired_msa"] = unpaired_msa
        if template:
            stdin_payload["template"] = template
        if user_ccd:
            stdin_payload["user_ccd"] = user_ccd

        logger.info(f"Launching AF3: {' '.join(cmd)} ...")

        result = subprocess.run(
            cmd,
            input=json.dumps(stdin_payload),
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            stderr_msg = result.stderr.strip() if result.stderr else "<no stderr>"
            logger.error(f"AF3 subprocess failed with return code {result.returncode}:\n{stderr_msg}")
            return False
        return True

    def _send_to_af3_server(
        self,
        cif_string: str,
        job_output_dir: Path,
        paired_msa: dict[str, str] | None = None,
        unpaired_msa: dict[str, str] | None = None,
        template: dict[str, Any] | None = None,
        user_ccd: str | None = None,
    ) -> bool:
        """Send one inference request to the persistent AF3 server.

        Writes a JSON line to the server's stdin and reads back the status
        response.  The server writes AF3 outputs directly to *job_output_dir*
        (using ``force_output_dir=True``).

        Args:
            cif_string: CIF-formatted input structure.
            job_output_dir: Directory for AF3 output files (must already exist).
            paired_msa: Per-chain paired MSA dict.
            unpaired_msa: Per-chain unpaired MSA dict.
            template: Per-chain template dict.
            user_ccd: User CCD CIF string for non-standard residues.

        Returns:
            True on success, False if the server reported an error.
        """
        proc = AF3Oracle._server_cache.get(self._server_key)
        if proc is None:
            raise RuntimeError("AF3 server is not running. Ensure setup() is called before compute().")

        payload: dict[str, Any] = {"cif": cif_string, "output_dir": str(job_output_dir)}
        if paired_msa:
            payload["paired_msa"] = paired_msa
        if unpaired_msa:
            payload["unpaired_msa"] = unpaired_msa
        if template:
            payload["template"] = template
        if user_ccd:
            payload["user_ccd"] = user_ccd

        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

        response_line = self._readline_timeout(
            proc.stdout,
            timeout=AF3Oracle._SERVER_IO_TIMEOUT,
            context="inference response",
        )
        if not response_line:
            logger.error("AF3 server closed its stdout unexpectedly (process may have crashed).")
            return False

        response = json.loads(response_line)
        if response.get("status") != "success":
            logger.error(f"AF3 server error: {response.get('message', 'unknown error')}")
            return False
        return True

    def _run_af3(
        self,
        cif_string: str,
        job_output_dir: Path,
        paired_msa: dict[str, str] | None = None,
        unpaired_msa: dict[str, str] | None = None,
        template: dict[str, Any] | None = None,
        user_ccd: str | None = None,
    ) -> bool:
        """Dispatch an AF3 inference request to the server or a one-off subprocess.

        Uses the persistent server when :meth:`setup` has been called (warm
        path).  Falls back to launching a new subprocess when no server is
        running (cold path — backwards compatible with tests and standalone use).

        The server receives *job_output_dir* directly and writes outputs there
        with ``force_output_dir=True``; the one-shot subprocess receives the
        parent directory and creates the *job_name* subdirectory itself.  Both
        paths result in outputs at ``{job_output_dir}/{job_name}_*.cif``.

        Args:
            cif_string: CIF-formatted input structure.
            job_output_dir: Directory where AF3 should write its outputs.
            paired_msa: Per-chain paired MSA dict.
            unpaired_msa: Per-chain unpaired MSA dict.
            template: Per-chain template dict.
            user_ccd: User CCD CIF string for non-standard residues.

        Returns:
            True if inference succeeded, False otherwise.
        """
        if self._server_key in AF3Oracle._server_cache:
            job_output_dir.mkdir(parents=True, exist_ok=True)
            return self._send_to_af3_server(
                cif_string=cif_string,
                job_output_dir=job_output_dir,
                paired_msa=paired_msa,
                unpaired_msa=unpaired_msa,
                template=template,
                user_ccd=user_ccd,
            )
        return self._launch_af3_subprocess(
            cif_string=cif_string,
            job_output_dir=job_output_dir.parent,
            paired_msa=paired_msa,
            unpaired_msa=unpaired_msa,
            template=template,
            user_ccd=user_ccd,
        )

    # ------------------------------------------------------------------
    # Input preparation helpers
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
            assert (
                annot_name in atom_array.get_annotation_categories()
            ), f"Annotation '{annot_name}' not found in AtomArray"
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
                    "AF3 structural templating only applies to full residues. "
                    "Residues with any templated atoms will be fully templated to AF3."
                )

            # The full chain CIF is the template, so query and template indices are identical
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

        nonstandard_aa = atom_array[nonstandard_mask]
        user_ccd_arrays: dict[str, AtomArray] = {}
        for res_array in residue_iter(nonstandard_aa):
            # Single-atom residues don't get a custom CCD code
            if res_array.array_length() == 1:
                continue
            res_name = res_array.res_name[0]
            if res_name not in user_ccd_arrays:
                user_ccd_arrays[res_name] = res_array

        config = CIFWriteConfig(
            include_entity_categories=True,
            id="user_ccd_entry",
            extra_categories={"pdbx_audit_revision_history": {"revision_date": date.today().isoformat()}},
        )
        user_ccd_cif_blocks = {}
        for res_name, res_array in user_ccd_arrays.items():
            cif_block = _to_cif_or_bcif(res_array, config=config)["user_ccd_entry"]
            user_ccd_cif_blocks[res_name] = cif_block

        output_file_obj = CIFFile(blocks=user_ccd_cif_blocks)
        buffer = io.StringIO()
        output_file_obj.write(buffer)
        buffer.seek(0)
        return buffer.getvalue()

    def _parse_af3_predicted_structure(self, af3_output_dir: Path, job_name: str) -> AtomArray | None:
        """Parse the best-ranking AF3 predicted structure from CIF output.

        Args:
            af3_output_dir: Directory where AF3 wrote its output files directly.
            job_name: Sanitised job name (spaces replaced with underscores).

        Returns:
            Predicted :class:`~biotite.structure.AtomArray`, or ``None``.
        """
        cif_path = af3_output_dir / f"{job_name}_model.cif"
        if not cif_path.exists():
            logger.warning(f"AF3 output CIF not found: {cif_path}")
            return None

        result = parse(cif_path, config=ParseConfig.from_preset("minimal"))["asym_unit"][0]
        return as_atom_array_plus(result)

    def _parse_af3_confidences(self, af3_output_dir: Path, job_name: str) -> dict | None:
        """Parse the AF3 summary confidences JSON.

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

    def _coi_interface_vals(self, matrix: list, coi_idx: int, n_chains: int) -> list:
        """Collect bidirectional interface values between the COI and every other chain."""
        vals = []
        for j in range(n_chains):
            if j == coi_idx:
                continue
            v_forward = matrix[coi_idx][j]
            v_reverse = matrix[j][coi_idx]
            if v_forward is not None and v_reverse is not None:
                vals.append((v_forward + v_reverse) / 2.0)
            elif v_forward is not None:
                vals.append(v_forward)
            elif v_reverse is not None:
                vals.append(v_reverse)
        return vals

    def _extract_metrics_from_confidences(
        self,
        summary_confidences: dict,
        chain_of_interest: str | None,
        chains_in_order: list[str],
    ) -> dict[str, float]:
        """Extract scalar metrics from AF3 summary confidences.

        Args:
            summary_confidences: Augmented summary confidences dict from AF3.
            chain_of_interest: Chain label for COI metrics, or ``None``.
            chains_in_order: Ordered list of chain labels.

        Returns:
            Dictionary of scalar confidence metrics.
        """
        metrics: dict[str, float] = {}

        if summary_confidences.get("overall_plddt") is not None:
            metrics["overall_plddt"] = float(summary_confidences["overall_plddt"]) * 100.0
        if summary_confidences.get("overall_pae") is not None:
            metrics["overall_pae"] = float(summary_confidences["overall_pae"])
        if summary_confidences.get("ptm") is not None:
            metrics["overall_ptm"] = float(summary_confidences["ptm"]) * 100.0
        if summary_confidences.get("iptm") is not None:
            metrics["overall_iptm"] = float(summary_confidences["iptm"]) * 100.0
        if summary_confidences.get("ranking_score") is not None:
            metrics["ranking_score"] = float(summary_confidences["ranking_score"])
        if summary_confidences.get("fraction_disordered") is not None:
            metrics["fraction_disordered"] = float(summary_confidences["fraction_disordered"])
        if summary_confidences.get("has_clash") is not None:
            metrics["has_clash"] = float(summary_confidences["has_clash"])

        if chain_of_interest is not None and chain_of_interest in chains_in_order:
            coi_idx = list(chains_in_order).index(chain_of_interest)
            n_chains = len(chains_in_order)

            if summary_confidences.get("chain_plddt") is not None:
                metrics["plddt_coi"] = float(summary_confidences["chain_plddt"][coi_idx]) * 100.0
            if summary_confidences.get("chain_ptm") is not None:
                metrics["ptm_coi"] = float(summary_confidences["chain_ptm"][coi_idx]) * 100.0
            if summary_confidences.get("chain_pae") is not None:
                metrics["pae_coi"] = float(summary_confidences["chain_pae"][coi_idx])
            if summary_confidences.get("chain_pair_pae") is not None:
                vals = self._coi_interface_vals(summary_confidences["chain_pair_pae"], coi_idx, n_chains)
                if vals:
                    metrics["ipae_coi"] = float(np.mean(vals))
            if summary_confidences.get("chain_pair_pae_min") is not None:
                vals = self._coi_interface_vals(summary_confidences["chain_pair_pae_min"], coi_idx, n_chains)
                if vals:
                    metrics["min_ipae_coi"] = float(np.min(vals))
            if summary_confidences.get("chain_pair_iptm") is not None:
                vals = self._coi_interface_vals(summary_confidences["chain_pair_iptm"], coi_idx, n_chains)
                if vals:
                    metrics["iptm_coi"] = float(np.mean(vals)) * 100.0

        return metrics

    @staticmethod
    def _disambiguate_chain_ids(atom_array: AtomArray) -> tuple[AtomArray, dict[str, str]]:
        """Assign unique chain IDs to chains that share a chain_id but differ by chain_iid.

        Args:
            atom_array: Input structure, optionally with a ``chain_iid`` annotation.

        Returns:
            AtomArray with renamed chains and a mapping from chain_iids to their af3 chain_ids.
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
    def _build_atom_index_mapping(input_array: AtomArray, af3_array: AtomArray) -> np.ndarray:
        """Build a per-atom index mapping from input to AF3 output.

        Args:
            input_array: Normalised input structure.
            af3_array: Parsed AF3 output structure.

        Returns:
            ``input_to_af3`` of shape ``(n_atoms,)``.
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

                # Build AF3 atom_name → local index; check for duplicates
                af3_name_to_local: dict[str, int] = {}
                for local_j, name in enumerate(af3_names):
                    if name in af3_name_to_local:
                        raise ValueError(
                            f"Chain '{chain_id}', residue {res_i}: duplicate atom name '{name}' in AF3 output"
                        )
                    af3_name_to_local[name] = local_j

                # Map each input atom to its AF3 counterpart
                seen_inp: set[str] = set()
                for local_i, name in enumerate(inp_names):
                    if name in seen_inp:
                        raise ValueError(f"Chain '{chain_id}', residue {res_i}: duplicate atom name '{name}' in input")
                    seen_inp.add(name)
                    if name not in af3_name_to_local:
                        raise ValueError(f"Chain '{chain_id}', residue {res_i}: atom '{name}' not found in AF3 output")
                    input_to_af3[inp_global[i0 + local_i]] = af3_global[a0 + af3_name_to_local[name]]

        return input_to_af3

    # ------------------------------------------------------------------
    # OracleMetric implementation
    # ------------------------------------------------------------------

    def fold_single_atom_array(
        self,
        atom_array: AtomArray,
        example_id: str,
        out_dir: Path | None,
        chain_of_interest: str | None,
    ) -> dict:
        """Fold one structure with AF3 and save the refolded CIF.

        Args:
            atom_array: Design structure to fold.
            example_id: Identifier used for file naming.
            out_dir: Base output directory. CIF saved via
                :py:meth:`refolded_cif_path`. If ``None``, no CIF is written.
            chain_of_interest: Pre-computed chain of interest, or ``None`` to
                infer from the structure.

        Returns:
            Flat dict with all AF3 confidence values.
        """
        job_name = example_id.replace(" ", "_")

        # Disambiguate chain IDs: if any chain_id has multiple chain_iids, assign
        # fresh alphabetical names so AF3 can distinguish them.
        disambiguated, af3_chain_id_to_original_iid = self._disambiguate_chain_ids(atom_array)
        valid_coord_mask = np.all(np.isfinite(disambiguated.coord), axis=-1)
        disambiguated_valid = disambiguated[valid_coord_mask]
        disambiguated_valid.set_annotation("occupancy", np.ones(disambiguated_valid.array_length()))
        disambiguated_valid.set_annotation("b_factor", np.zeros(disambiguated_valid.array_length()))

        config = CIFWriteConfig(id=job_name)
        cif_string = to_cif_string(disambiguated_valid.copy(), config=config)
        template = self._build_af3_template_dict(disambiguated_valid)
        user_ccd = self._build_user_ccd_string(disambiguated_valid)

        predicted_atom_array = None
        summary_confidences = None

        with tempfile.TemporaryDirectory() as _tmp:
            tmp_dir = Path(_tmp)
            # af3_output_dir is where AF3 writes its output files directly.
            # Both the server (force_output_dir=True) and the one-shot subprocess
            # end up writing to this subdirectory.
            af3_output_dir = tmp_dir / job_name

            try:
                success = self._run_af3(
                    cif_string=cif_string,
                    job_output_dir=af3_output_dir,
                    template=template,
                    user_ccd=user_ccd,
                )
            except Exception as e:
                logger.error(f"AF3 inference failed for {example_id}: {e}")
                return _EMPTY_METRICS.copy()

            if not success:
                return _EMPTY_METRICS.copy()

            summary_confidences = self._parse_af3_confidences(af3_output_dir, job_name)
            if summary_confidences is None:
                logger.error(f"AF3 confidences not found for {example_id}")
                return _EMPTY_METRICS.copy()

            predicted_atom_array = self._parse_af3_predicted_structure(af3_output_dir, job_name)

            if predicted_atom_array is not None:
                try:
                    # NOTE: AF3 will add missing atoms within a residue, but will remove all-nan residues.
                    # We therefore follow the same procedure for the purposes of coordinate transfer.
                    nan_coord_mask = np.isnan(disambiguated.coord).any(axis=-1)
                    missing_residue_mask = apply_residue_wise(disambiguated, nan_coord_mask, np.all)
                    missing_residue_mask = spread_residue_wise(disambiguated, missing_residue_mask)
                    disambiguated_partial = disambiguated[~missing_residue_mask]

                    # We remove OXT atoms from AF3 if these are the cause of a length mismatch (designs don't have OXT)
                    oxt_mask = predicted_atom_array.atom_name == "OXT"
                    if disambiguated_partial.array_length() == predicted_atom_array[~oxt_mask].array_length():
                        predicted_atom_array = predicted_atom_array[~oxt_mask]

                    input_to_af3 = self._build_atom_index_mapping(disambiguated_partial, predicted_atom_array)
                    result = as_atom_array_plus(atom_array.copy())

                    # Write AF3 coords only for atoms in residues that are not all NaN coordinates;
                    # all-NaN residues keep their NaN coords in the result.
                    result.coord[~missing_residue_mask] = predicted_atom_array.coord[input_to_af3]
                    predicted_atom_array = result
                except (ValueError, KeyError) as e:
                    logger.warning(f"AF3 coordinate transfer failed for {example_id}: {e}")
                    predicted_atom_array = None

        if predicted_atom_array is not None and out_dir is not None:
            cif_path = self.refolded_cif_path(out_dir, example_id)
            cif_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                to_cif_file(predicted_atom_array, cif_path)
            except Exception as e:
                logger.warning(f"Failed to save AF3 refolded CIF for {example_id}: {e}")

        if chain_of_interest is None:
            chain_of_interest, coi_unambiguous = get_chain_of_interest(atom_array)
            if not coi_unambiguous:
                chain_of_interest = None

        af3_chain_ids_in_order: list[str] = summary_confidences.get("chain_ids", [])

        if af3_chain_id_to_original_iid is not None:
            original_chain_iids_in_order = [
                af3_chain_id_to_original_iid.get(chain_id) for chain_id in af3_chain_ids_in_order
            ]
        else:
            original_chain_iids_in_order = af3_chain_ids_in_order

        metrics = self._extract_metrics_from_confidences(
            summary_confidences=summary_confidences,
            chain_of_interest=chain_of_interest,
            chains_in_order=original_chain_iids_in_order,
        )
        metrics["chain_of_interest"] = chain_of_interest
        return metrics
