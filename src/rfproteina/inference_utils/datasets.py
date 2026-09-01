"""Dataset wrappers for design task inference."""

import json
import pathlib
import time
from collections.abc import Callable, Sequence
from os import PathLike
from typing import Any

import yaml
from atomworks.common import exists
from atomworks.io.parser import parse
from atomworks.ml.datasets import FileDataset, MolecularDataset
from atomworks.ml.transforms.base import Compose, Transform, TransformedDict
from atomworks.ml.transforms.realize_expandable_segments import RealizeExpandableSegments
from omegaconf import DictConfig, ListConfig, OmegaConf

from rfproteina.conditions_reporting import run_pipeline_validation
from rfproteina.inference_utils.condition_spec import (
    build_annotated_atom_array_from_condition_spec,
    is_legacy_dialect,
    split_global_args,
)
from rfproteina.transforms.inference_annotations import AddSrcComponentAnnotation
from rfproteina.utils.common import STRUCTURE_FILE_SUFFIXES, is_structure_file, validate_design_id
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.parse_presets import ANNOTATED_CIF

logger = RankedLogger(__name__, rank_zero_only=True)
all_ranks_logger = RankedLogger(__name__, rank_zero_only=False)


def take_shard(items: Sequence, shard_index: int | None, num_shards: int | None, num_replicates: int) -> list:
    """Keep every ``num_shards``-th item, starting at ``shard_index``; no-op if unsharded.

    The stride is deliberate: entries are in name order, which usually correlates with length, so
    each shard gets a similar length mix (and runtime) where a contiguous slice would not.

    Args:
        items: Entries to divide, already expanded to one per (example, replicate).
        shard_index: Which shard to keep, in ``[0, num_shards)``.
        num_shards: Number of shards to divide `items` into.
        num_replicates: Number of times each example is replicated.

    Raises:
        ValueError: If `shard_index` is out of range, or this shard would be empty.
    """
    if not exists(num_shards) or num_shards <= 1:
        return list(items)
    if not exists(shard_index) or not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index!r}")
    shard = list(items)[shard_index::num_shards]
    if not shard:
        raise ValueError(
            f"Shard {shard_index} of {num_shards} is empty: only {len(items)} example-replicate(s) "
            f"to divide. Lower gen_njobs to at most {max(len(items), num_replicates)}."
        )
    return shard


class ConditionSpecDataset(MolecularDataset):
    """Loads condition_spec YAML/JSON files (docs/condition_spec_yaml.md).

    The source is a single JSON/YAML file (a dict of ``{example_id: spec}``); each top-level key
    becomes one dataset example, built directly into an AtomArray -- no CIF round trip through disk.
    """

    def __init__(
        self,
        *,
        data: PathLike | dict,
        transform: Transform | Compose | None,
        name: str | None,
        sampling_transforms: list[Transform] | None = None,
        subset_to_keys: list[str] | None = None,
        num_replicates: int = 1,
        global_prefix: str | None = None,
        source_path: PathLike | str | None = None,
        prevalidate: bool = False,
        shard_index: int | None = None,
        num_shards: int | None = None,
    ):
        """
        Args:
            - data: path to the JSON/YAML file containing the example data, or an already-loaded dict
            - transform: transform to apply to each built example
            - name: name of the dataset
            - sampling_transforms: optional transforms that edit the built ``atom_array`` (typically
              resampling a condition per replicate) after expandable segments are realized and
              before the featurization ``transform`` runs. They are deliberately excluded from
              ``prevalidate``, which checks the input as authored.
            - subset_to_keys: list of keys to subset the data to
            - num_replicates: number of times to repeat each example in the dataset
            - global_prefix: optional global prefix for output example ids
            - source_path: original JSON/YAML path when ``data`` is an in-memory dict
            - prevalidate: if True, build + validate each unique example once during init
            - shard_index: which shard this instance serves, in ``[0, num_shards)``
            - num_shards: number of shards to divide the replicated examples into; see
              :py:func:`take_shard`
        """
        # Necessary for working with expandable segments; could be incorporated into the standard
        # Transform pipeline in future.
        self.pre_transform = AddSrcComponentAnnotation() + RealizeExpandableSegments()
        # Sampling runs per __getitem__ call, so each replicate of a spec draws its own condition.
        sampled = self.pre_transform
        for sampling_transform in sampling_transforms or []:
            sampled = sampled + sampling_transform
        transform = sampled + transform if transform is not None else sampled

        if isinstance(data, PathLike | str):
            self.json_path = data
            original_data = self._load_from_path(data)
        elif isinstance(data, DictConfig):
            self.json_path = str(source_path) if source_path is not None else None
            original_data = OmegaConf.to_object(data)
        else:
            self.json_path = str(source_path) if source_path is not None else None
            original_data = data

        self._data = self._extract_examples(original_data)

        # Subset to requested keys
        if subset_to_keys is not None:
            assert len(subset_to_keys) > 0, "subset_to_keys must be a non-empty list of keys."
            self._data = {k: v for k, v in self._data.items() if k in subset_to_keys}

        if exists(global_prefix):
            self._data = {f"{global_prefix}_{k}": v for k, v in self._data.items()}

        self._check_json_keys()

        # Validate each unique example once, before replication fans them out.
        if prevalidate:
            self._prevalidate()

        # Create the requested number of replicates
        if num_replicates >= 1:
            new_data = {f"{k}_{i}": v for k, v in self._data.items() for i in range(num_replicates)}
            self._data = new_data
        else:
            raise ValueError(f"num_replicates must be an integer greater than or equal to 1, got {num_replicates}")

        # Shard after replication, so both axes are divided by the same stride.
        self._data = dict(take_shard(list(self._data.items()), shard_index, num_shards, num_replicates))

        self.name = name if name is not None else "condition-spec-dataset"
        self.transform = transform

        if len(self) == 0:
            raise ValueError(f"{type(self).__name__} is empty, data: {data}. Names: {self.names}")

        # Condense output: show first 3 and last 3 names if more than 8 examples
        if len(self.names) <= 8:
            fmt_names = ", ".join(self.names)
        else:
            fmt_names = ", ".join(self.names[:3]) + ", ... " + ", ".join(self.names[-3:])
        logger.info(f"Dataset {self.name}: {len(self):,} examples ({fmt_names})")

    @staticmethod
    def _load_from_path(data: str | PathLike) -> dict:
        """Load data from a JSON or YAML file."""
        path = pathlib.Path(data)
        assert path.exists(), f"Input file {path} does not exist."
        with open(path) as f:
            if path.suffix == ".json":
                return json.load(f)
            if path.suffix in (".yaml", ".yml"):
                return yaml.safe_load(f)
        raise ValueError(f"Input file {path} must be a JSON or YAML file.")

    def _extract_examples(self, data: dict) -> dict:
        if is_legacy_dialect(data):
            raise NotImplementedError(
                f"{self.json_path or 'input'} uses the legacy contig-JSON (`InputSpecification`) "
                "dialect, which has been removed. Convert it to Condition-annotated CIFs "
                "(scripts/json2cif/build_cif.py) or rewrite it in the condition_spec format "
                "(docs/condition_spec_yaml.md)."
            )
        self.spec_dir = pathlib.Path(self.json_path).parent if self.json_path is not None else pathlib.Path()
        # `global_args` holds shared defaults folded into every example (per-example values win).
        return split_global_args(data)

    def _check_json_keys(self) -> None:
        """Check that each key names one output directory and maps to a spec dict.

        Every design writes its outputs to ``<output_base_dir>/<example_id>/``, so an id
        carrying a path separator (or ``..``) would escape the run's output tree; catch it
        here rather than partway through generation.
        """
        for k, data in self.data.items():
            validate_design_id(k)
            if not isinstance(data, dict):
                raise ValueError("Each item in the JSON data must be a dictionary.")

    @property
    def data(self) -> dict:
        """Expose underlying dataframe as property to discourage changing it.

        Changing it can lead to unexpected behavior with torch ConcatDatasets.
        """
        return self._data

    @property
    def names(self) -> list[str]:
        return list(self.data.keys())

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, example_id: str) -> bool:
        return example_id in self.names

    def id_to_idx(self, example_id: str) -> int:
        return self.names.index(example_id)

    def idx_to_id(self, idx: int) -> str:
        return self.names[idx]

    def _prevalidate(self) -> None:
        """Build each unique example with condition_spec's own pre-save safeguards (fail fast)."""
        for example_id, spec in self.data.items():
            logger.info(f"Prevalidating input example: {example_id}")
            data = build_annotated_atom_array_from_condition_spec(
                spec, spec_dir=self.spec_dir, example_id=example_id, validate=True
            )
            # Validate again after realizing expandable segments
            realized = self.pre_transform(TransformedDict(data))["atom_array"]
            violations = run_pipeline_validation(realized)
            if violations:
                raise ValueError(
                    f"Prevalidation failed for input example {example_id!r} after realizing "
                    "expandable segments:\n" + "\n".join(violations)
                )

    def _build_example(self, example_id: str, spec: dict) -> dict:
        data = TransformedDict(
            build_annotated_atom_array_from_condition_spec(spec, spec_dir=self.spec_dir, example_id=example_id)
        )
        if self.json_path is not None:
            data["specification"] = {"input_json_path": str(pathlib.Path(self.json_path).resolve())}
        return data

    def __getitem__(self, idx: int) -> Any:
        """Build one example, then send it through the transform pipeline."""
        example_id = self.idx_to_id(idx)
        spec = self.data[example_id]

        data = self._build_example(example_id, spec)
        data["example_id"] = example_id
        data["task"] = "_".join(example_id.split("_")[:-1])

        data = self.transform(data)
        return data


class DesignProblemDataset(FileDataset):
    """
    Loads a design problem specification from a set of CIF files that adhere to the
    `DesignSpecification` format.
    """

    def __init__(
        self,
        dir_path_or_file_paths: list[PathLike] | PathLike,
        *,
        filter_fn: Callable[[PathLike], bool] | None = is_structure_file,
        transform: Transform | None = None,
        sampling_transforms: list[Transform] | None = None,
        num_replicates: int | None = None,
        global_prefix: str | None = None,
        subset_to_keys: list[str] | None = None,
        prevalidate: bool = False,
        shard_index: int | None = None,
        num_shards: int | None = None,
    ):
        """Initialize a dataset for loading design problem specifications from CIF files.


        Args:
            dir_path_or_file_paths: Path to directory with CIF/JSON/YAML files, or list of file paths.
            filter_fn: Optional function to filter which files to include based on filename.
            transform: Optional transform to apply to loaded data.
            sampling_transforms: Optional transforms that edit the loaded ``atom_array`` (typically
                resampling a condition per replicate) after expandable segments are realized and
                before the featurization ``transform`` runs. They are deliberately excluded from
                ``prevalidate``, which checks the input as authored.
            global_prefix: Optional global prefix for output example ids.
            subset_to_keys: Optional list of base ids (the file stem, before any
                ``global_prefix``) to keep; useful for running a few examples out of a large
                input directory.
            prevalidate: if True, load + validate each unique CIF once during init.
            num_replicates: Optional number of times to repeat each file in the dataset
            shard_index: Which shard this instance serves, in ``[0, num_shards)``.
            num_shards: Number of shards to divide the replicated example list into. See
                :py:func:`take_shard`.

        Raises:
            ValueError: If a key in ``subset_to_keys`` matches none of the discovered files.
        """
        # Ensure input is a list
        if not isinstance(dir_path_or_file_paths, list | ListConfig):
            dir_path_or_file_paths = [dir_path_or_file_paths]

        # Assemble list of file paths
        if len(dir_path_or_file_paths) == 1 and pathlib.Path(dir_path_or_file_paths[0]).is_dir():
            dir_path = pathlib.Path(dir_path_or_file_paths[0])
            file_paths = [p for suffix in STRUCTURE_FILE_SUFFIXES for p in dir_path.rglob(f"*{suffix}")]
        else:
            dir_path = None
            file_paths = [pathlib.Path(p) for p in dir_path_or_file_paths]

        dataset_name = dir_path if exists(dir_path) else file_paths[0].parent

        super().__init__(
            file_paths=file_paths,
            name=dataset_name,
            filter_fn=filter_fn,
        )

        # Annotate SA_SRC before expansion so original res_ids are preserved,
        # then realize expandable segments and continue with transform pipeline.
        self.pre_transform = AddSrcComponentAnnotation() + RealizeExpandableSegments()
        # Sampling runs per __getitem__ call, so each replicate of a file draws its own condition.
        sampled = self.pre_transform
        for sampling_transform in sampling_transforms or []:
            sampled = sampled + sampling_transform
        self.transform = sampled + transform if transform is not None else sampled

        self.global_prefix = global_prefix

        # Split each path into its components (parent directories plus the file stem).
        def get_base_id_parts(path: PathLike) -> tuple[str, ...]:
            return pathlib.Path(path).with_suffix("").parts

        all_parts = [get_base_id_parts(p) for p in self.file_paths]
        max_parts = max(len(parts) for parts in all_parts)
        # Grow from the stem, prepending one parent dir at a time, until every id is unique.
        for k in range(1, max_parts + 1):
            candidate_ids = ["-".join(parts[-k:]) for parts in all_parts]
            if len(set(candidate_ids)) == len(candidate_ids):
                break
        else:
            raise ValueError(
                f"Could not build unique base_ids for {len(self.file_paths)} files even "
                "using their full paths; remove duplicate input paths."
            )
        # Subset before prevalidation and replication, so discarded inputs cost nothing.
        if exists(subset_to_keys):
            keys = set(subset_to_keys)
            if missing := keys - set(candidate_ids):
                raise ValueError(
                    f"subset_to_keys entries {sorted(missing)} match none of the "
                    f"{len(candidate_ids)} discovered inputs (e.g. {candidate_ids[0]!r})."
                )
            kept = [i for i, cid in enumerate(candidate_ids) if cid in keys]
            self.file_paths = [self.file_paths[i] for i in kept]
            candidate_ids = [candidate_ids[i] for i in kept]

        if exists(self.global_prefix):
            candidate_ids = [f"{self.global_prefix}_{cid}" for cid in candidate_ids]
        self.base_ids = candidate_ids

        # Validate each unique CIF once, before replication fans them out.
        if prevalidate:
            self._prevalidate()

        # Allow for multiple replicates of the same validation example
        if exists(num_replicates):
            self.num_replicates = num_replicates
        else:
            self.num_replicates = 1

        if self.num_replicates > 1:
            self.file_paths = self.file_paths * self.num_replicates
            self.base_ids = self.base_ids * self.num_replicates

        # Shard after replication, so both axes are divided by the same stride.
        kept = take_shard(range(len(self.file_paths)), shard_index, num_shards, self.num_replicates)
        self.file_paths = [self.file_paths[i] for i in kept]
        self.base_ids = [self.base_ids[i] for i in kept]

    def _prevalidate(self) -> None:
        """Load each unique CIF, realize its expandable segments, and validate it (fail fast)."""
        for path, base_id in zip(self.file_paths, self.base_ids, strict=True):
            logger.info(f"Prevalidating input example: {base_id}")
            data = self._load_from_path(path)
            atom_array = self.pre_transform(TransformedDict(data))["atom_array"]
            violations = run_pipeline_validation(atom_array)
            if violations:
                raise ValueError(f"Prevalidation failed for input example {base_id!r}:\n" + "\n".join(violations))

    def _load_from_path(self, path: PathLike) -> Any:
        """
        Loads a design problem specification from a .ipd file
        """

        result_dict = parse(path, config=ANNOTATED_CIF)

        # Ensure that we have charge information in the output
        atom_array = result_dict["assemblies"]["1"][0]

        result_dict["atom_array"] = atom_array
        _ = result_dict.pop("asym_unit")
        _ = result_dict.pop("assemblies")

        return result_dict

    def __getitem__(self, idx: int) -> Any:
        """
        Loads a design problem specification from a .ipd file
        """
        path = self.file_paths[idx]
        previous_occurrences = self.file_paths[:idx].count(path)
        _start_parse_time = time.time()
        data = self._load_from_path(path)
        # --- Use unique base_id logic ---
        base_id = self.base_ids[idx]
        data["example_id"] = f"{base_id}_rep{previous_occurrences}" if self.num_replicates > 1 else base_id
        data["task"] = base_id
        _stop_parse_time = time.time()

        data = TransformedDict(data)
        data.__transform_history__.append(
            {
                "name": "load_from_path",
                "instance": hex(id(self._load_from_path)),
                "start_time": _start_parse_time,
                "end_time": _stop_parse_time,
                "processing_time": _stop_parse_time - _start_parse_time,
            }
        )

        data = self.transform(data)

        data["specification"] = {
            "input_cif": path,
        }
        return data
