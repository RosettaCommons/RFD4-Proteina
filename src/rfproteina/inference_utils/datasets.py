"""Dataset wrappers for design task inference."""

import json
import pathlib
import time
from collections.abc import Callable
from os import PathLike
from typing import Any

import biotite.structure as struc
import numpy as np
import yaml
from atomworks.common import exists
from atomworks.io.config import PrepareConfig
from atomworks.io.parser import parse, parse_atom_array
from atomworks.ml.datasets import FileDataset, MolecularDataset
from atomworks.ml.transforms.base import Compose, Transform, TransformedDict
from atomworks.ml.transforms.realize_expandable_segments import RealizeExpandableSegments
from biotite.structure import AtomArray
from omegaconf import DictConfig, ListConfig, OmegaConf

from rfproteina.conditions_reporting import run_pipeline_validation
from rfproteina.inference_utils.condition_spec import (
    build_annotated_atom_array_from_condition_spec,
    is_legacy_dialect,
    split_global_args,
)
from rfproteina.inference_utils.inference_utils import (
    REQUIRED_INFERENCE_ANNOTATIONS,
    STRUCTURE_FILE_SUFFIXES,
    check_has_required_conditioning_annotations,
    convert_existing_annotations_to_bool,
    ensure_input_is_abspath,
    is_structure_file,
)
from rfproteina.inference_utils.input_parsing import (
    adapt_legacy_atom_array_for_condition_pipeline,
    create_atom_array_from_design_specification,
)
from rfproteina.transforms.inference_annotations import AddSrcComponentAnnotation
from rfproteina.utils.common import validate_design_id
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.parse_presets import ANNOTATED_CIF

logger = RankedLogger(__name__, rank_zero_only=True)
all_ranks_logger = RankedLogger(__name__, rank_zero_only=False)


class JsonDataset(MolecularDataset):
    """Base for datasets whose source is one JSON/YAML file: a dict of `{example_id: spec}`."""

    def __init__(
        self,
        *,
        data: PathLike | dict,
        transform: Transform | Compose | None,
        name: str | None,
        subset_to_keys: list[str] | None = None,
        num_replicates: int = 1,
        global_prefix: str | None = None,
        source_path: PathLike | str | None = None,
        prevalidate: bool = False,
    ):
        """
        Args:
            - data: path to the JSON/YAML file containing the example data, or an already-loaded dict
            - transform: transform to apply to the data
            - name: name of the dataset
            - subset_to_keys: list of keys to subset the data to
            - num_replicates: number of times to repeat each example in the dataset
            - global_prefix: optional global prefix for output example ids
            - source_path: original JSON/YAML path when ``data`` is an in-memory dict
            - prevalidate: if True, build + validate each unique example once during init
        """

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

        # ...basic assignments
        self.name = name if name is not None else "json-dataset"
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
        """Subclass-specific preprocessing hook: raw loaded dict -> `{example_id: spec}`.

        The default is a no-op; override to drop reserved top-level keys or fold
        file-level defaults into each example before the shared subset/replicate/lookup
        machinery below runs.
        """
        return data

    def _build_example(self, example_id: str, spec: dict) -> dict:
        """Build the `{atom_array, chain_info, ligand_info, metadata}` dict for one example."""
        raise NotImplementedError

    def _prevalidate(self) -> None:
        """Build each unique example's AtomArray and run the validation pipeline on it (fail fast)."""
        for example_id, spec in self.data.items():
            logger.info(f"Prevalidating input example: {example_id}")
            atom_array = self._build_example(example_id, spec)["atom_array"]
            violations = run_pipeline_validation(atom_array)
            if violations:
                raise ValueError(f"Prevalidation failed for input example {example_id!r}:\n" + "\n".join(violations))

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
        """Pass through the length of the wrapped dataset."""
        return len(self.names)

    def __contains__(self, example_id: str) -> bool:
        """Pass through the contains method of the wrapped dataset."""
        return example_id in self.names

    def id_to_idx(self, example_id: str) -> int:
        """Pass through the id_to_idx method of the wrapped dataset."""
        return self.names.index(example_id)

    def idx_to_id(self, idx: int) -> str:
        """Pass through the idx_to_id method of the wrapped dataset."""
        return self.names[idx]

    def __getitem__(self, idx: int) -> Any:
        """Build one example, then send it through the transform pipeline."""
        example_id = self.idx_to_id(idx)
        spec = self.data[example_id]

        data = self._build_example(example_id, spec)
        data["example_id"] = example_id
        data["task"] = "_".join(example_id.split("_")[:-1])

        data = self.transform(data)
        return data


class ContigJsonDataset(JsonDataset):
    """
    Enables loading of JSON files containing contig data for benchmark design tasks,
    or the passing of examples through analogously-structured hydra configs.

    Deprecated: this is the legacy `InputSpecification`-based dialect (see
    :py:func:`~rfproteina.inference_utils.condition_spec.is_legacy_dialect`). Switch to
    the condition_spec format (`docs/condition_spec_yaml.md`,
    :py:class:`ConditionSpecDataset`) for future compatibility.
    """

    def __init__(self, *, cif_parser_overrides: dict | None = None, **kwargs):
        """
        Args:
            - cif_parser_overrides: overrides to the default CIF parsing arguments
            - **kwargs: forwarded to :py:class:`JsonDataset`
        """
        self.cif_parser_overrides = cif_parser_overrides
        super().__init__(**kwargs)

    def _extract_examples(self, data: dict) -> dict:
        # These will have already been added at inference time, but this block is useful for validation.
        if "global_args" in data:
            global_args = data.pop("global_args")
            for k in data:
                data[k].update(global_args)
        return data

    def _prevalidate(self) -> None:
        logger.warning(
            "prevalidate_inputs is not supported for the legacy contig-JSON format; skipping. "
            "Migrate to the condition_spec format (docs/condition_spec_yaml.md) to enable it."
        )

    def _build_example(self, example_id: str, spec: dict) -> dict:
        # if 'input' in metadata and not abspath, prepend the source json directory to the file path
        spec = ensure_input_is_abspath(spec, self.json_path)

        # ... Create atom array with conditioning annotations
        atom_array, spec_dict = create_atom_array_from_design_specification(
            cif_parser_args=self.cif_parser_overrides, **spec
        )

        # ... Adapt to Condition-based pipeline if needed and run parse
        data = prepare_pipeline_input_from_atom_array(atom_array)

        # ... Wrap up with additional features
        if "extra" not in spec_dict:
            spec_dict["extra"] = {}
        spec_dict["extra"]["example_id"] = example_id
        if self.json_path is not None:
            spec_dict["input_json_path"] = str(pathlib.Path(self.json_path).resolve())
        data["specification"] = spec_dict
        return data


class ConditionSpecDataset(JsonDataset):
    """Loads condition_spec YAML/JSON files (docs/condition_spec_yaml.md): one dataset example
    per top-level key, built directly into an AtomArray -- no CIF round trip through disk.
    """

    def __init__(self, *, transform: Transform | Compose | None, **kwargs):
        """
        Args:
            - transform: transform to apply to each built example
            - **kwargs: forwarded to :py:class:`JsonDataset`
        """
        # Necessary for working with expandable segments. Once the ContigJsonDataset is deprecated,
        # this can be incorporated into the standard Transform pipeline.
        self.pre_transform = AddSrcComponentAnnotation() + RealizeExpandableSegments()
        super().__init__(
            transform=self.pre_transform + transform if transform is not None else self.pre_transform, **kwargs
        )

    def _extract_examples(self, data: dict) -> dict:
        if is_legacy_dialect(data):
            raise ValueError(
                f"{self.json_path or 'input'} uses the legacy `InputSpecification`-based JSON/YAML "
                "dialect, which ConditionSpecDataset does not consume. Use ContigJsonDataset instead "
                "(generate.py's default json/yaml routing already does this automatically)."
            )
        self.spec_dir = pathlib.Path(self.json_path).parent if self.json_path is not None else pathlib.Path()
        # `global_args` holds shared defaults folded into every example (per-example values win).
        return split_global_args(data)

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


def prepare_pipeline_input_from_atom_array(atom_array_orig: AtomArray) -> dict:
    """Prepare atom array for inference pipeline.

    Args:
        atom_array_orig: Atom array with conditioning annotations.

    Returns:
        Dict with keys ``'atom_array'``, ``'chain_info'``, ``'ligand_info'``, ``'metadata'``.
    """

    # Adapt legacy annotations to Condition API
    # TODO: This should be refactored to avoid the two-step process
    atom_array_orig = adapt_legacy_atom_array_for_condition_pipeline(atom_array_orig)

    # HACK: Set empty bond graph:
    if atom_array_orig.bonds is None:
        atom_array_orig.bonds = struc.BondList(atom_array_orig.array_length())

    # HACK: Infer bonds and add to existing bond list
    inferred_bonds = struc.connect_via_residue_names(atom_array_orig, inter_residue=True)
    for atom_index1, atom_index2, bond_type in inferred_bonds.as_array():
        atom_array_orig.bonds.add_bond(atom_index1, atom_index2, bond_type)

    prepare_config = PrepareConfig(
        remove_ccds=[],
        fix_arginines=False,
        add_missing_atoms=False,
        convert_mse_to_met=False,
        hydrogen_policy="remove",
    )

    result_dict = parse_atom_array(atom_array_orig, config=prepare_config)
    atom_array = result_dict["assemblies"]["1"][0]  # First model of first assembly
    # Ensure that we have charge information in the output
    if "charge" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("charge", np.zeros(atom_array.array_length(), dtype=int))
    # HACK: Set iid information manually
    # We currently do not preserve this information from the input,
    # if you want these we'd need to remove the spoofing here
    check_has_required_conditioning_annotations(atom_array, required=REQUIRED_INFERENCE_ANNOTATIONS)
    atom_array = convert_existing_annotations_to_bool(atom_array)
    atom_array.set_annotation("chain_iid", [f"{c}_1" for c in atom_array.chain_id])
    atom_array.set_annotation("pn_unit_iid", [f"{c}_1" for c in atom_array.pn_unit_id])
    data = {
        "atom_array": atom_array,  # First model
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }
    data = TransformedDict(data)
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
        num_replicates: int | None = None,
        global_prefix: str | None = None,
        prevalidate: bool = False,
    ):
        """Initialize a dataset for loading design problem specifications from CIF files.


        Args:
            dir_path_or_file_paths: Path to directory with CIF/JSON/YAML files, or list of file paths.
            filter_fn: Optional function to filter which files to include based on filename.
            transform: Optional transform to apply to loaded data.
            global_prefix: Optional global prefix for output example ids.
            subset_to_keys: Optional list of keys to subset the data to.
            prevalidate: if True, load + validate each unique CIF once during init.
            num_replicates: Optional number of times to repeat each file in the dataset
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
        self.transform = self.pre_transform + transform if transform is not None else self.pre_transform

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
