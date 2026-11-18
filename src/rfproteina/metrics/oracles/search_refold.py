"""Invocation-local handoff of AF3/OF3 search refolds to final evaluation."""

import hashlib
import json
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from atomworks.io.parser import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from biotite.structure import AtomArray

from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput, OracleMetric
from rfproteina.utils.coordinate_transfer import remove_terminal_oxygen_atoms
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.parse_presets import ANNOTATED_CIF
from rfproteina.utils.structure_files import structure_file_stem

logger = RankedLogger(__name__, rank_zero_only=True)


def _identity(oracle: OracleMetric, array: AtomArray) -> tuple | None:
    # Be conservative: arbitrary oracle subclasses and pre-oracle transforms may
    # depend on additional state. Only the built-in, untransformed AF3/OF3 path is supported.
    if type(oracle) is not AF3Oracle or oracle.pre_oracle_transform is not None:
        return None
    array = as_atom_array_plus(remove_terminal_oxygen_atoms(array.copy()))
    digest = hashlib.sha256()
    values = [array.coord, array.box, array.bonds.as_array() if array.bonds is not None else None]
    for n_body in (0, 1, 2):
        for name in sorted(array.get_annotation_categories(n_body=n_body)):
            values.extend((n_body, name, array.get_annotation(name, n_body=n_body)))
    for value in values:
        # Serialize each value separately so shared Python references cannot affect identity.
        digest.update(pickle.dumps(value, protocol=5))
    settings = (
        oracle.oracle_name,
        oracle.template_query,
        oracle.seed,
        oracle.preserve_res_id,
        tuple(oracle.extra_flags),
        tuple(getattr(oracle, "exclude_ccds", ())),
        tuple(os.environ.get(key) for key in ("AF3_MODEL_DIR", "OF3_MODEL_DIR", "AF3_IMAGE")),
    )
    return digest.hexdigest(), settings


@dataclass(frozen=True)
class SearchRefold:
    """A reference to one evaluated candidate, never a cache shared between searches."""

    output_dir: Path
    example_id: str
    identity: tuple

    @classmethod
    def capture(cls, oracle: OracleMetric, array: AtomArray, output_dir: Path, example_id: str) -> Self | None:
        identity = _identity(oracle, array)
        return cls(output_dir, example_id, identity) if identity is not None else None

    def reuse(self, oracle: OracleMetric, job: OracleJobInput) -> bool:
        """Publish a complete refold under the final ID, or leave a cache miss.

        Called with the final staged array, before fold_batch. Confidence JSON is
        the completion marker and is published last; final metrics still run normally.
        """
        marker = oracle.confidence_path(job.out_dir, job.example_id)
        # A previous run's final cache is not evidence for this newly generated candidate.
        marker.unlink(missing_ok=True)
        old_cif = oracle.existing_refolded_cif_path(job.out_dir, job.example_id)
        if old_cif is not None:
            old_cif.unlink()  # A fresh fold may use a different compression suffix.
        try:
            if self.identity != _identity(oracle, job.atom_array):
                return False
            source = oracle.existing_refolded_cif_path(self.output_dir, self.example_id)
            confidence = oracle.confidence_path(self.output_dir, self.example_id)
            if source is None:
                return False
            confidences = json.loads(confidence.read_text())
            if not isinstance(confidences, dict) or not confidences:
                return False
            array = parse(source, config=ANNOTATED_CIF)["assemblies"]["1"][0]
            if len(array) == 0:
                return False
            # Preserve the actual encoding, including compression, rather than renaming bytes.
            suffix = source.name[len(structure_file_stem(source)) :]
            destination = oracle.refolded_cif_path(job.out_dir, job.example_id)
            destination = destination.with_name(structure_file_stem(destination) + suffix)
            marker.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=marker.parent, prefix=".search-refold-") as staging:
                staged_cif = Path(staging) / destination.name
                staged_json = Path(staging) / marker.name
                shutil.copyfile(source, staged_cif)
                shutil.copyfile(confidence, staged_json)
                os.replace(staged_cif, destination)
                os.replace(staged_json, marker)
            return True
        except Exception as exc:
            logger.warning(f"{job.example_id}: search refold reuse failed; folding again: {exc}")
            return False
