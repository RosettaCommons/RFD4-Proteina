"""Structural novelty against an append-only archive of validated passing designs."""

import logging
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from biotite.structure import AtomArray

from rfproteina.metrics.metric import Metric
from rfproteina.metrics.utils.metrics_utils import _write_chain_only_cif_if_unambiguous

logger = logging.getLogger(__name__)


class FoldseekNoveltyMetric(Metric):
    """Report whether a designed chain has no sufficiently similar archived passer.

    Exhaustively searches a local Foldseek database, rebuilt when archive files change.
    Archive files are immutable, single-chain CIFs; use one archive per target and pass
    definition. Queries never publish designs. Teardown releases the cached database.

    Args:
        archive_dir: Directory populated by a PassCriterionMetric with archive_dir set.
        tm_threshold: Minimum query- and target-normalized TM scores for a match.
        min_coverage: Minimum aligned fraction of both query and target for a match.
        threads: CPU threads per Foldseek invocation.
        timeout: Maximum seconds for each Foldseek invocation.
        **kwargs: Standard metric settings, including selection tags.
    """

    def __init__(
        self,
        archive_dir: str,
        tm_threshold: float = 0.5,
        min_coverage: float = 0.8,
        threads: int = 1,
        timeout: float = 300.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not archive_dir or not 0 < tm_threshold <= 1 or not 0 < min_coverage <= 1:
            raise ValueError("Novelty requires an archive_dir and TM-score/coverage thresholds in (0, 1].")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1 or timeout <= 0:
            raise ValueError("Novelty requires positive integer threads and a positive timeout.")
        self.archive_dir = Path(archive_dir)
        self.tm_threshold = tm_threshold
        self.min_coverage = min_coverage
        self.threads = threads
        self.timeout = timeout
        self._workspace: TemporaryDirectory | None = None
        self._files: tuple[Path, ...] = ()

    @property
    def kwargs_to_compute_args(self) -> dict:
        return {"atom_array": "predicted_atom_array"}

    def teardown(self) -> None:
        """Release the local database; the shared passer archive is unaffected."""
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None
        self._files = ()

    def _run(self, *args: object) -> None:
        executable = os.environ.get("FOLDSEEK_EXEC")
        if not executable:
            raise ValueError("Novelty requires FOLDSEEK_EXEC pointing to the Foldseek executable.")
        result = subprocess.run(
            [executable, *map(str, args), "--threads", str(self.threads), "-v", "1"],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"Foldseek {args[0]} failed: {result.stderr or result.stdout}")

    def compute(self, atom_array: AtomArray) -> dict[str, float]:
        """Treat the design as novel if no archived passer matches or the novelty check fails."""
        try:
            return self._compute(atom_array)
        except Exception as exc:
            logger.warning("Novelty check failed; treating design as novel: %s", exc)
            self._files = ()
            return {"is_novel": 1.0}

    def _compute(self, atom_array: AtomArray) -> dict[str, float]:
        archived = tuple(sorted(self.archive_dir.glob("[!.]*.cif")))
        if not archived:
            return {"is_novel": 1.0}
        if archived != self._files:
            self.teardown()
            self._workspace = TemporaryDirectory(prefix="foldseek_novelty_db_")
            database = Path(self._workspace.name) / "archive"
            self._run("createdb", *archived, database)
            if len(database.with_suffix(".lookup").read_text().splitlines()) != len(archived):
                raise RuntimeError("Foldseek could not read every archived passer.")
            self._files = archived
        database = Path(self._workspace.name) / "archive"
        with TemporaryDirectory(prefix="foldseek_novelty_query_") as directory:
            directory = Path(directory)
            query = _write_chain_only_cif_if_unambiguous(atom_array, "query", directory)
            if query is None:
                raise ValueError("Novelty requires an unambiguous chain of interest.")
            hits = directory / "hits.tsv"
            self._run(
                "easy-search",
                query,
                database,
                hits,
                directory / "tmp",
                "--alignment-type",
                1,
                "--exhaustive-search",
                1,
                "--max-seqs",
                len(archived),
                "-e",
                1000000,
                "--format-output",
                "qtmscore,ttmscore,qcov,tcov",
            )
            for line in hits.read_text().splitlines():
                qtm, ttm, qcov, tcov = map(float, line.split())
                if min(qtm, ttm) >= self.tm_threshold and min(qcov, tcov) >= self.min_coverage:
                    return {"is_novel": 0.0}
        return {"is_novel": 1.0}
