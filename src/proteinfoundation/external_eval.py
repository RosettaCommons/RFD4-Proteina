#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../../../.ipd/shebang/latent_exec.sh" "$0" "$@"'

import argparse
import os
import json
import pandas as pd
from pathlib import Path
from biotite.structure import AtomArray, AtomArrayStack, infer_elements
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from atomworks.io.utils.io_utils import load_any, to_cif_file
from atomworks.io.utils.atom_array_plus import as_atom_array_plus, concatenate_atom_array_plus
from atomworks.io.transforms.atom_array import add_chain_type_annotation, add_polymer_annotation
from atomworks.io.utils.non_rcsb import initialize_chain_info_from_atom_array
from atomworks.io.transforms.atom_array import add_charge_from_ccd_codes
from atomworks.common import exists
from atomworks.ml.conditions import C_CRD, C_IDX, C_DIS
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.ml.utils.token import apply_and_spread_token_wise
import yaml
from foundry.metrics import MetricManager
import numpy as np
import torch
from proteinfoundation.conditions import C_IDXGRP, C_INTCRD, C_HOT
from proteinfoundation.utils.chain_utils import get_chain_of_interest
from proteinfoundation.transforms.unindexing import process_unindexed_outputs
from proteinfoundation.generate import get_contig_or_design_problem_dataset
from proteinfoundation.inference_utils.datasets import DesignProblemDataset, ContigJsonDataset
from proteinfoundation.inference_utils.inference_utils import set_com
from proteinfoundation.metrics.rf3_metric import RF3Metric
from proteinfoundation.inference_utils.concatenate_external_eval_results import (
    aggregate_metrics_from_per_example_df,
)
from foundry.utils.logging import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


def setup_argparse():
    parser = argparse.ArgumentParser(
        description="Run external evaluation and save the outputs to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-parquet",
        type=str,
        required=True,
        help="Path to the input parquet file with 'generated_structure' and 'example_id' columns",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the outputs of the evaluation will be saved",
    )
    parser.add_argument(
        "--pfm-dataset-input",
        type=str,
        required=True,
        help="Path to a pfm-formatted json/yaml, Condition-annotated CIF, or directory of CIFs",
    )
    parser.add_argument(
        "--metric-manager-config",
        type=str,
        default=None,
        help="Path to the config file for the MetricManager. If not specified, no metrics are computed.",
    )
    parser.add_argument(
        "--centering-method",
        type=str,
        required=True,
        help="How the motif coordinates were translated to produce the coords that the model saw.",
    )

    def _str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "1", "y"):
            return True
        if v.lower() in ("no", "false", "f", "0", "n"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument(
        "--use-sequence-redesign",
        type=_str_to_bool,
        required=True,
        help="Whether the sequences for refolding should be obtained from an auxiliary model, e.g. MPNN.",
    )

    parser.add_argument(
        "--subset-to-indices",
        type=int,
        nargs="+",
        default=None,
        help="List of indices to subset the input dataframe (optional, mutually exclusive with chunk args)",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=None,
        help="Chunk size: integer for absolute count, or float 0-1 for fraction of total (optional)",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=None,
        help="Job ID for chunk selection (0-indexed, optional)",
    )
    parser.add_argument(
        "--annotated-design-save-dir",
        type=str,
        default=None,
        help="Directory to save annotated designs as CIFs with mapping and Condition annotations (optional)",
    )

    args = parser.parse_args()

    # Validate mutually exclusive arguments
    if args.subset_to_indices is not None and (args.chunk_size is not None or args.job_id is not None):
        parser.error("--subset_to_indices cannot be used with --chunk_size or --job_id")

    if (args.chunk_size is None) != (args.job_id is None):
        parser.error("--chunk_size and --job_id must be specified together")

    return args


def _ensure_not_atom_array_stack(arr):
    if isinstance(arr, AtomArrayStack):
        assert arr.stack_depth() == 1, "Obtained a stacked AtomArray with depth > 1. "
        arr = arr[0]
    return arr


def _clean_metrics_dict(metrics: dict) -> dict | None:
    """Convert a metrics dict to clean floats for gathering across ranks."""
    if metrics is None:
        return None

    # Standardize types
    cleaned = {}
    chain_of_interest_values = []
    for key, value in metrics.items():
        if key in ("dataset_name", "example_id", "designed_structure_path", "sequence_redesigned_structure_path", "task"):
            cleaned[key] = str(value)
            continue
        elif key.endswith("chain_of_interest"):
            chain_of_interest_values.append(str(value))
            continue
        if isinstance(value, int | float):
            cleaned[key] = float(value)
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            cleaned[key] = value.item()
        elif isinstance(value, np.ndarray) and value.size == 1:
            cleaned[key] = value.item()

    # Add chain of interest column
    if len(set(chain_of_interest_values)) == 1:
        cleaned["chain_of_interest"] = chain_of_interest_values[0]
    elif len(set(chain_of_interest_values)) > 1:
        logger.warning(f"Multiple different chains of interest reported: {set(chain_of_interest_values)}")
        cleaned["chain_of_interest"] = None
    else:
        cleaned["chain_of_interest"] = None

    # Return None if only metadata (no actual metrics)
    metric_keys = [
        k
        for k in cleaned
        if k not in ("dataset_name", "example_id", "designed_structure_path", "sequence_redesigned_structure_path", "task", "chain_of_interest")
    ]
    return cleaned if metric_keys else None


def _add_basic_annots(atom_array):
    atom_array = _ensure_not_atom_array_stack(atom_array)
    chain_info = initialize_chain_info_from_atom_array(atom_array)
    atom_array = add_chain_type_annotation(atom_array, chain_info)
    atom_array = add_polymer_annotation(atom_array, chain_info)
    atom_array = as_atom_array_plus(atom_array)
    ensure_annotations(atom_array, "is_protein", "is_ligand")
    return atom_array


def save_metrics(all_metric_outputs: list[dict], output_dir: str, job_id: int | None = None):
    # 1. Collect and clean per-example metrics
    all_example_metrics = [cleaned for m in all_metric_outputs if (cleaned := _clean_metrics_dict(m)) is not None]

    if not all_example_metrics:
        raise ValueError("No valid metrics were computed for any examples.")

    # 2. Save per-example CSV (one row per example)
    metrics_dir = Path(output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    job_id_str = f"_job{job_id}" if job_id is not None else ""

    per_example_df = pd.DataFrame(all_example_metrics)
    # Reorder columns: example_id first, then dataset_name, then metrics
    cols = per_example_df.columns.tolist()
    priority = ["designed_structure_path", "sequence_redesigned_structure_path", "example_id", "task", "dataset_name", "chain_of_interest"]
    ordered_cols = [c for c in priority if c in cols] + [c for c in cols if c not in priority]
    per_example_df = per_example_df[ordered_cols]

    per_example_path = metrics_dir / f"per_example_metrics{job_id_str}.csv"
    per_example_df.to_csv(per_example_path, index=False)
    print(f"Saved per-example metrics to {per_example_path}")

    # 3. If single job, aggregate and save final metrics
    if job_id is None:
        agg_df = aggregate_metrics_from_per_example_df(per_example_df, title="Validation Metrics")
        agg_path = metrics_dir / "aggregated_metrics.csv"

        agg_df.to_csv(agg_path, index=False)
        print(f"Saved aggregate metrics to {agg_path}")


def correct_index_groups_for_length_variability(atom_array, spec: dict) -> AtomArray:
    """Infer where to place index group breaks for mapping indexed components to model outputs.
    This is necessary because variable-length regions between two indexed groups can make the index gaps variable.
    However, if the gaps are fixed-length, we want to enforce that the model respects the relative indices.

    Args:
        atom_array (AtomArray): The input AtomArray to be modified.
        spec (dict): The design problem specification dictionary.
    Returns:
        AtomArray: The modified AtomArray with corrected index groups.
    """
    contig = spec.get("contig")
    if contig is None:
        return atom_array

    contig_parts = contig.split(",")
    increment_index_group = []

    in_variable_length_diffused_region = False
    in_indexed_motif_region = False
    first_element_of_chain = True
    for part in contig_parts:
        if part[0].isalpha():
            # Record that we are in an indexed motif region
            if not in_indexed_motif_region:
                in_indexed_motif_region = True

                # If we were previously in a variable-length diffused region, add an index group break
                if in_variable_length_diffused_region:
                    in_variable_length_diffused_region = False
                    increment_index_group.append(True)
                else:
                    increment_index_group.append(False)

            first_element_of_chain = False

        # If the chain changes, note this and reset
        elif part == "/0":
            in_indexed_motif_region = False
            in_variable_length_diffused_region = False
            first_element_of_chain = True

        else:
            # We don't need to change the index group for diffused regions
            if in_indexed_motif_region or first_element_of_chain:
                in_indexed_motif_region = False
                increment_index_group.append(False)

            # If we are in a variable-length diffused region, note this
            if "-" in part:
                in_variable_length_diffused_region = True

            first_element_of_chain = False

    chain_identifier = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
    chain_annot = atom_array.get_annotation(chain_identifier)
    indexed_motif_mask = C_IDX.mask(atom_array, default="raise") & C_CRD.mask(atom_array, default="raise")
    chain_or_indexed_motif_changes = (chain_annot[1:] != chain_annot[:-1]) | (
        indexed_motif_mask[1:] != indexed_motif_mask[:-1]
    )
    change_indices = np.where(chain_or_indexed_motif_changes)[0] + 1
    change_indices = np.concatenate(([0], change_indices, [atom_array.array_length()]))

    if len(change_indices) != len(increment_index_group) + 1:
        # Sampling a diffused region of length zero can lead to no detected changes where one was expected
        # Cases where this is the first or last region in the contig are easily detected and cover all existing benchmarks
        # If other cases arise in the future, this function will need to be revisited
        if contig_parts[0].startswith("0-") and indexed_motif_mask[0]:
            increment_index_group = increment_index_group[1:]
        if contig_parts[-1].startswith("0-") and indexed_motif_mask[-1]:
            increment_index_group = increment_index_group[:-1]

        # If the lengths still don't match, we raise an error
        if len(change_indices) != len(increment_index_group) + 1:
            raise ValueError(
                "The number of detected chain/indexed motif changes does not match the number of expected index group breaks. "
            )

    index_groups = C_IDXGRP.annotation(atom_array, default="generate")
    current_max_idxgrp = index_groups.max()
    for i, (start, stop) in enumerate(zip(change_indices[:-1], change_indices[1:])):
        if increment_index_group[i]:
            index_groups[start:stop] += current_max_idxgrp + 1
            current_max_idxgrp += 1
    C_IDXGRP.set_annotation(atom_array, index_groups)

    return atom_array


def apply_centering_method(
    row,
    centering_method,
    coord_conditioning_atom_array,
    loaded_atom_array,
    dataset,
    pfm_dataset_input,
):
    allow_non_motif_centered = True
    ori_token = None
    infer_ori_strategy = None

    if centering_method == "no_centering":
        infer_ori_strategy = "preserve"
    elif centering_method == "motif_centered":
        allow_non_motif_centered = False
    elif centering_method == "protein_motif_centered":
        allow_non_motif_centered = True
        ori_token = coord_conditioning_atom_array[coord_conditioning_atom_array.is_protein].coord.mean(axis=0)
    elif centering_method == "ori_token":
        # Get the complete input structure file to locate the ORI token (before subsetting to contig)
        if isinstance(dataset, ContigJsonDataset):
            with open(pfm_dataset_input, "r") as f:
                pfm_json_data = json.load(f)
            input_file = pfm_json_data[row.example_id]["input"]
            if not os.path.isabs(input_file):
                input_file = os.path.abspath(os.path.join(os.path.dirname(pfm_dataset_input), input_file))

            full_input_atom_array = load_any(input_file)
            full_input_atom_array = _ensure_not_atom_array_stack(full_input_atom_array)
        else:
            full_input_atom_array = loaded_atom_array

        ori_token_mask = full_input_atom_array.res_name == "ORI"
        if not ori_token_mask.any():
            raise ValueError("When using 'ori_token' centering method, the motif must contain an ORI token.")
        elif ori_token_mask.sum() > 1:
            raise ValueError("Multiple ORI tokens found in the motif; expected only one.")
        ori_token = full_input_atom_array[ori_token_mask].coord[0]
    elif centering_method == "hotspots":
        if C_HOT.full_name not in coord_conditioning_atom_array.get_annotation_categories():
            logger.warning("No hotspots detected. Falling back to 'motif_centered' centering strategy.")
            allow_non_motif_centered = False
        else:
            infer_ori_strategy = "hotspots"
    elif centering_method == "ori_from_rfd3_json":
        rfd3_json_path = row.rfd3_json
        with open(rfd3_json_path, "r") as f:
            rfd3_json_data = json.load(f)
        ori_token = rfd3_json_data.get("ori_token")
        infer_ori_strategy = rfd3_json_data.get("infer_ori_strategy")
        if not exists(ori_token) and not exists(infer_ori_strategy):
            raise ValueError(
                f"RFD3 JSON file {rfd3_json_path} does not specify either 'ori_token' or 'infer_ori_strategy'."
            )
    elif centering_method == "unknown":
        # We will enforce a single candidate mapping per group, making the centering irrelevant
        return coord_conditioning_atom_array
    else:
        raise ValueError(f"Unknown centering method: {centering_method}")

    coord_conditioning_atom_array = set_com(
        coord_conditioning_atom_array, ori_token, infer_ori_strategy, allow_non_motif_centered
    )
    return coord_conditioning_atom_array


# TODO: If more annotations are needed down the line, this will need to be updated accordingly.
def transfer_annotations_to_redesigned_structure(generated_atom_array, sequence_redesigned_atom_array):
    """Transfer annotations from the generated structure to the sequence-redesigned structure in the mapped region."""
    # Subset to the region mapped to the unindexed coordinate conditioning
    matched_mapped_indices = []
    for atom in generated_atom_array[generated_atom_array.mapped_to_coordinate_conditioned_region]:
        match_mask = (
            (sequence_redesigned_atom_array.chain_id == atom.chain_id)
            & (sequence_redesigned_atom_array.res_id == atom.res_id)
            & (sequence_redesigned_atom_array.atom_name == atom.atom_name)
        )
        if match_mask.sum() != 1:
            raise ValueError(
                "Expected exactly one match for each atom in the mapped region, but found "
                f"{match_mask.sum()} matches for atom {f'{atom.chain_id} {atom.res_id} {atom.res_name} {atom.atom_name}'}"
            )

        matched_idx = np.where(match_mask)[0][0]
        matched_mapped_indices.append(matched_idx)

    # Annotate regions that mapped to the inserted regions
    matched_mapped_indices = np.array(matched_mapped_indices, dtype=int)
    mapped_to_coordinate_conditioned_region_redesigned = np.zeros(sequence_redesigned_atom_array.array_length(), dtype=bool)
    mapped_to_coordinate_conditioned_region_redesigned[matched_mapped_indices] = True
    sequence_redesigned_atom_array.set_annotation("mapped_to_coordinate_conditioned_region", mapped_to_coordinate_conditioned_region_redesigned)
    mapped_to_coordinate_conditioned_region_token_lvl = apply_and_spread_token_wise(
        sequence_redesigned_atom_array, sequence_redesigned_atom_array.mapped_to_coordinate_conditioned_region, np.any
    )
    sequence_redesigned_atom_array.set_annotation(
        "mapped_to_coordinate_conditioned_region_token_lvl", mapped_to_coordinate_conditioned_region_token_lvl
    )

    # Copy over coords with insertion
    coords_with_insertion = np.full_like(sequence_redesigned_atom_array.coord, np.nan)
    coords_with_insertion[matched_mapped_indices] = generated_atom_array.coord[
        generated_atom_array.mapped_to_coordinate_conditioned_region
    ]
    sequence_redesigned_atom_array.set_annotation("coords_with_insertion", coords_with_insertion)

    # If present, copy over src_component and src_coordinate_conditioned_region annotations
    # NOTE: These are optional because they are not strictly necessary for downstream tasks
    if "src_component" in generated_atom_array.get_annotation_categories():
        src_component = np.full(sequence_redesigned_atom_array.array_length(), fill_value=".", dtype="<U10")
        src_component[matched_mapped_indices] = generated_atom_array.src_component[
            generated_atom_array.mapped_to_coordinate_conditioned_region
        ]
        sequence_redesigned_atom_array.set_annotation("src_component", src_component)

        src_coordinate_conditioned_region = np.full(sequence_redesigned_atom_array.array_length(), fill_value=".", dtype="<U10")
        src_coordinate_conditioned_region[matched_mapped_indices] = generated_atom_array.src_coordinate_conditioned_region[
            generated_atom_array.mapped_to_coordinate_conditioned_region
        ]
        sequence_redesigned_atom_array.set_annotation("src_coordinate_conditioned_region", src_coordinate_conditioned_region)


    # Copy over relevant masks within the mapped regions, using the defaults everywhere else
    for cond in (C_CRD, C_IDX, C_INTCRD):
        cond_mask = cond.default_mask(sequence_redesigned_atom_array)
        cond_mask[matched_mapped_indices] = generated_atom_array[
            generated_atom_array.mapped_to_coordinate_conditioned_region
        ].get_annotation(cond.mask_name)
        cond.set_mask(sequence_redesigned_atom_array, cond_mask)

    return sequence_redesigned_atom_array


def convert_to_pfm_format(generated_atom_array, coord_conditioning_atom_array, inputs_in_same_frame):
    """Convert the generated AtomArray to the format/annotations expected by the MetricManager."""

    # Infer element from atom name for RFD3 outputs, which contain 'VX'
    known_elements_mask = np.isin(generated_atom_array.element, list(ELEMENT_NAME_TO_ATOMIC_NUMBER.keys()))
    generated_atom_array.element[~known_elements_mask] = infer_elements(generated_atom_array[~known_elements_mask])

    # Spoof a uniform index group for the generated atoms (unused but allows the annotation to be preserved through concatenation)
    coord_cond_index_groups = C_IDXGRP.annotation(coord_conditioning_atom_array, default="generate")
    max_conditioning_idxgrp = coord_cond_index_groups.max()
    C_IDXGRP.set_annotation(
        generated_atom_array, np.full(generated_atom_array.array_length(), max_conditioning_idxgrp + 1, dtype=int)
    )

    # For mapping purposes, consider the generated atoms indexed and the in-context conditioning unindexed
    C_IDX.set_mask(generated_atom_array, np.ones(generated_atom_array.array_length(), dtype=bool))
    C_IDX.set_mask(coord_conditioning_atom_array, np.zeros(coord_conditioning_atom_array.array_length(), dtype=bool))

    # Generated regions are annotated as not being coordinate-conditioned to mimic the unindexed insertion setting
    C_CRD.set_mask(generated_atom_array, np.zeros(generated_atom_array.array_length(), dtype=bool))
    C_CRD.set_mask(coord_conditioning_atom_array, np.ones(coord_conditioning_atom_array.array_length(), dtype=bool))

    # Prepare the `coords_with_insertion` annotation
    generated_atom_array.set_annotation("coords_with_insertion", generated_atom_array.coord.copy())
    coord_conditioning_atom_array.set_annotation(
        "coords_with_insertion", np.full_like(coord_conditioning_atom_array.coord, np.nan)
    )

    # Subset to coordinate-conditioned protein regions, as only these need to be mapped
    coord_conditioning_atom_array = coord_conditioning_atom_array[coord_conditioning_atom_array.is_protein]

    # The coordinate conditioning must have distinct res_ids from the generated array to facilitate the mapping
    modified_coord_conditioning_atom_array = coord_conditioning_atom_array.copy()
    max_res_id_by_chain = {
        chain_id: np.max(generated_atom_array.res_id[generated_atom_array.chain_id == chain_id])
        for chain_id in np.unique(generated_atom_array.chain_id)
    }
    for chain_id, max_res_id in max_res_id_by_chain.items():
        chain_mask = modified_coord_conditioning_atom_array.chain_id == chain_id
        if chain_mask.any():
            modified_coord_conditioning_atom_array.res_id[chain_mask] += max_res_id

    # HACK: Rfd3 doesn't output OXT atoms which breaks the mapping code, so if they are not present in the generated
    # structure we remove them from the conditioning as well
    if not np.isin("OXT", generated_atom_array.atom_name).any():
        oxt_mask = modified_coord_conditioning_atom_array.atom_name == "OXT"
        if oxt_mask.any():
            modified_coord_conditioning_atom_array = modified_coord_conditioning_atom_array[~oxt_mask]

    # Concatenate the coordinate conditioning to the generated structure
    concatenated_array = concatenate_atom_array_plus([generated_atom_array, modified_coord_conditioning_atom_array])

    # Add a within index group res idx. This is used to determine desired index gaps within the region to be inserted
    ensure_annotations(concatenated_array, "within_chain_res_idx")
    within_index_group_res_idx = concatenated_array.within_chain_res_idx.copy()
    index_groups = C_IDXGRP.annotation(concatenated_array, default="raise")
    for index_group in np.unique(index_groups):
        group_mask = index_groups == index_group
        within_index_group_res_idx[group_mask] -= within_index_group_res_idx[group_mask].min()
    concatenated_array.set_annotation("within_index_group_res_idx", within_index_group_res_idx)

    index_groups = C_IDXGRP.annotation(concatenated_array, default="raise")
    for index_group in np.unique(index_groups):
        # We can just use within_chain_res_idx for the generated structure
        if index_group > max_conditioning_idxgrp:
            continue

        # NOTE: In the current input parsing setup, user-requested index gaps are only preserved as res_id gaps
        group_mask = index_groups == index_group
        group_res_ids = concatenated_array.res_id[group_mask]
        concatenated_array.within_index_group_res_idx[group_mask] = group_res_ids - group_res_ids.min()

    # Map the coordinate conditioning to the corresponding regions in the generated structure
    annotated_model_generated_array, insertion_metadata = process_unindexed_outputs(
        concatenated_array,
        insert_unindexed_regions=False,
        inputs_in_same_frame=inputs_in_same_frame,
    )
    coord_cond_mask = C_CRD.mask(annotated_model_generated_array, default="raise")
    coord_cond_mask[annotated_model_generated_array.mapped_to_coordinate_conditioned_region] = True
    C_CRD.set_mask(annotated_model_generated_array, coord_cond_mask)

    # If the model has failed to create valid reproductions of the motif, warn and give NaN values for coords_with_insertion
    coord_conditioned_count = modified_coord_conditioning_atom_array.array_length()
    if annotated_model_generated_array.mapped_to_coordinate_conditioned_region.sum() != coord_conditioned_count:
        print(
            f"WARNING: Unindexed coord conditioning count ({coord_conditioned_count}) does not match "
            f"remapped count ({annotated_model_generated_array.mapped_to_coordinate_conditioned_region.sum()}). "
            "The model has failed to reproduce the atom names of the input motif."
        )
        valid_mapping_found = False
    else:
        valid_mapping_found = True

    if not valid_mapping_found:
        annotated_model_generated_array.set_annotation(
            "coords_with_insertion", np.full_like(annotated_model_generated_array.coord, np.nan)
        )

    # This information is used when determining which backbone atoms to align on for the AME benchmark
    mapped_to_coordinate_conditioned_region_token_lvl = apply_and_spread_token_wise(
        annotated_model_generated_array, annotated_model_generated_array.mapped_to_coordinate_conditioned_region, np.any
    )
    annotated_model_generated_array.set_annotation(
        "mapped_to_coordinate_conditioned_region_token_lvl", mapped_to_coordinate_conditioned_region_token_lvl
    )

    return annotated_model_generated_array, insertion_metadata


def run_external_eval(
    input_parquet,
    output_dir,
    pfm_dataset_input,
    metric_manager_config,
    centering_method,
    use_sequence_redesign,
    annotated_design_save_dir=None,
    subset_to_indices=None,
    chunk_size=None,
    job_id=None,
):
    """Run external evaluation and save the outputs to disk.

    NOTE: This function will only add annotations related to abslute 3d coordinate conditioning (indexed or unindexed).
    More sophisticated conditioning methods (e.g. distances, internal coordinates, etc) are not currently supported.

    Args:
        input_parquet (str): Path to the input parquet file, which must contain the following columns:
            - generated_structure: Paths to the cleaned final outputs of the model. Can be CIF or PDB, optionally gzipped.
            - example_id: Unique identifier for each example. Either json/yaml key or CIF filepath.
                If the `pfm_dataset_input` is a directory path, the CIF paths should be relative to that directory.
            - (optional) sequence_redesigned_structure: Path to the sequence-redesigned output file, e.g. from MPNN.
                Required if `use_sequence_redesign` is True.
            - (optional) rfd3_json: Path to the RFD3-style json file. Required if using `ori_from_rfd3_json` centering method.
        pfm_dataset_input(str | list[str]): Path to a pfm-formatted json/yaml, path to a Condition-annotated CIF,
            or a list or directory of such CIFs.
        output_dir (str): Directory where the outputs of the evaluation will be saved
        metric_manager_config (str | None): Path to the config file for the MetricManager. If None, no metrics are computed.
            Note that the metrics must only rely on the generated AtomArray, and possibly the input coordinate conditioning.
        centering_method (str): How the motif coordinates were translated to produce the coords that the model saw.
            Options are 'no_centering', 'motif_centered', 'protein_motif_centered', 'hotspots', 'ori_token', 'ori_from_rfd3_json', and 'unknown'.
            The in-house latent model always uses 'motif_centered', and complexa uses 'protein_motif_centered'. For RFD3, the
            most common is 'motif_centered', but 'hotspots' is used for PPI tasks. For the MCSA, 'motif_centered' was used for the RFD3
            paper benchmark but the ori tokens specified in the MCSA input files were actually 'protein_motif_centered'.
            For boltzgen, all conditioning is distance-based, so this is actually akin to internal coordinate conditioning.
            Since this is not supported, we instead use 'unknown' for the centering method, which errors if the mapping is ambiguous.
        use_sequence_redesign (bool, optional): Whether the sequences for refolding should be obtained from an auxiliary model, e.g. MPNN.
        annotated_design_save_dir (str | None, optional): Path to save the annotated designs as CIFs, including mapping annotations.
        subset_to_indices (list, optional): A list of indices to subset the input dataframe.
        chunk_size (float, optional): Chunk size - integer for absolute count, or float 0-1 for fraction.
        job_id (int, optional): Job ID for chunk selection (0-indexed). Must be specified with chunk_size.

    Return:
        None. The outputs of the evaluation will be saved to disk in the specified output directory.
    """

    df = pd.read_parquet(input_parquet)

    # Disable centering now, as it will be performed later based on the centering_method
    specification_overrides = {"allow_non_motif_centered": True, "infer_ori_strategy": "preserve"}

    dataset = get_contig_or_design_problem_dataset(
        json_or_design_problem_inputs=pfm_dataset_input, transform=None, specification_overrides=specification_overrides
    )

    if "generated_structure" not in df.columns or "example_id" not in df.columns:
        raise ValueError("The input parquet file must contain columns named 'generated_structure' and 'example_id'.")

    # Handle subsetting
    if exists(subset_to_indices):
        df = df.iloc[subset_to_indices]
    elif chunk_size is not None and job_id is not None:
        total_size = len(df)

        # Convert chunk_size to absolute count if it's a fraction
        if 0 < chunk_size < 1:
            chunk_size_abs = int(np.ceil(chunk_size * total_size))
        else:
            chunk_size_abs = int(chunk_size)

        # Calculate chunk boundaries
        start_idx = job_id * chunk_size_abs
        end_idx = min(start_idx + chunk_size_abs, total_size)

        print(f"Processing chunk {job_id}: indices {start_idx} to {end_idx-1} (total size: {total_size})")
        df = df.iloc[start_idx:end_idx]

    if df.empty:
        raise ValueError(
            "The input dataframe is empty after subsetting. Please check the provided indices or chunk parameters."
        )

    # Instantiate the MetricManager
    if exists(metric_manager_config):
        with open(metric_manager_config, "r") as f:
            metric_manager_config_data = yaml.safe_load(f)
        metric_manager = MetricManager.instantiate_from_hydra(metrics_cfg=metric_manager_config_data)
        for metric in metric_manager.metrics.values():
            if isinstance(metric, RF3Metric):
                metric.setup()
    else:
        logger.warning("metric_manager_config is None -- no metrics will be computed.")
        metric_manager = None

    all_metric_outputs = []
    for row in df.itertuples():
        # Load the generated structure
        generated_structure_path = row.generated_structure
        generated_atom_array = load_any(generated_structure_path)

        # Load the annotated input structure and validate
        if isinstance(dataset, DesignProblemDataset):
            idx = dataset.file_paths.index(row.example_id)
            loaded_atom_array = dataset[idx]["atom_array"]
        elif isinstance(dataset, ContigJsonDataset):
            if row.example_id in dataset.names:
                idx = dataset.id_to_idx(row.example_id)
                spec = dataset.data[row.example_id]
            else:
                idx = dataset.id_to_idx(f"{row.example_id}_0")
                spec = dataset.data[f"{row.example_id}_0"]
            loaded_atom_array = dataset[idx]["atom_array"]
        else:
            raise ValueError("Expected dataset to be a DesignProblemDataset or a ContigJsonDataset.")

        # Add basic annotations that would ordinarily have been added by `parse` (e.g. chain type)
        loaded_atom_array = _add_basic_annots(loaded_atom_array)
        generated_atom_array = _add_basic_annots(generated_atom_array)

        # Modify index groups to account for variable-length diffused regions
        # Note that DesignProblemDatasets necessarily have fixed-length diffused regions
        if isinstance(dataset, ContigJsonDataset):
            loaded_atom_array = correct_index_groups_for_length_variability(loaded_atom_array, spec)

        # Validate obtained arrays
        if C_INTCRD.mask(loaded_atom_array, default="generate").any():
            raise ValueError("Internal coordinate conditioning is not supported for external evaluation.")
        C_INTCRD.set_mask(generated_atom_array, np.zeros(generated_atom_array.array_length(), dtype=bool))
        distance_mask_values = C_DIS.mask(loaded_atom_array, default="generate").values
        if distance_mask_values.size > 0 and distance_mask_values.any():
            raise ValueError("Distance-based conditioning is not supported for external evaluation.")

        # Subset to regions with absolute coordinate conditioning
        coord_conditioning_atom_array = loaded_atom_array[C_CRD.mask(loaded_atom_array, default="raise")]

        # Apply appropriate global translation to the coordinate conditioning
        coord_conditioning_atom_array = apply_centering_method(
            row,
            centering_method,
            coord_conditioning_atom_array=coord_conditioning_atom_array,
            loaded_atom_array=loaded_atom_array,
            pfm_dataset_input=pfm_dataset_input,
            dataset=dataset,
        )

        # Add annotations expected by the MetricManager
        inputs_in_same_frame = centering_method != "unknown"
        formatted_generated_atom_array, insertion_metadata = convert_to_pfm_format(
            generated_atom_array,
            coord_conditioning_atom_array,
            inputs_in_same_frame=inputs_in_same_frame,
        )

        # Optionally save the annotated designs
        if exists(annotated_design_save_dir):
            annotated_design_save_path = Path(annotated_design_save_dir) / f"{str(Path(generated_structure_path).stem).split('.')[0]}_annotated.cif"
            to_cif_file(
                formatted_generated_atom_array,
                annotated_design_save_path,
                extra_fields=[
                    "mapped_to_coordinate_conditioned_region",
                    "mapped_to_coordinate_conditioned_region_token_lvl",
                    "coords_with_insertion",
                    "src_coordinate_conditioned_region",
                    "src_component",
                ],
            )

        # If using a sequence-redesigned structure, we will first transfer annotations in the mapped region.
        if use_sequence_redesign:
            sequence_redesigned_atom_array = load_any(row.sequence_redesigned_structure)
            sequence_redesigned_atom_array = _add_basic_annots(sequence_redesigned_atom_array)
            try:
                sequence_redesigned_atom_array = transfer_annotations_to_redesigned_structure(
                    generated_atom_array=formatted_generated_atom_array,
                    sequence_redesigned_atom_array=sequence_redesigned_atom_array,
                )
                rf3_input_atom_array = sequence_redesigned_atom_array

            except ValueError as e:
                logger.warning(
                    f"Error transferring annotations: {e}.\n If a remapped atom was not found in the sequence-redesigned structure, "
                    "this may result from the associated residue having been sequence-redesigned, possibly due to use of a "
                    "different mapping algorithm to decide which residues should be preserved. This example will return NaN metrics."
                )
                all_metric_outputs.append(
                    {"example_id": str(Path(generated_structure_path).stem).split(".")[0], "task": row.example_id}
                )
                continue
        else:
            rf3_input_atom_array = formatted_generated_atom_array

        # RF3 requires charges in the outputs
        rf3_input_atom_array = add_charge_from_ccd_codes(rf3_input_atom_array)

        # Optionally run the MetricManager and collect the outputs
        if not exists(metric_manager):
            continue

        # Pre-compute chain of interest so it is shared across all metrics
        coi, coi_unambiguous = get_chain_of_interest(rf3_input_atom_array)
        chain_of_interest = coi if coi_unambiguous else None

        metric_output = metric_manager(
            network_input=None,
            network_output=None,
            extra_info={
                "example_id": str(Path(generated_structure_path).stem).split(".")[0],
                "output_base_dir": Path(output_dir),
                "chain_of_interest": chain_of_interest,
            },
            predicted_atom_array=[rf3_input_atom_array],
        )
        metric_output["task"] = row.example_id
        metric_output["designed_structure_path"] = generated_structure_path
        if use_sequence_redesign:
            metric_output["sequence_redesigned_structure_path"] = row.sequence_redesigned_structure
        metric_output["insertion_rmsd"] = insertion_metadata.get("insertion_rmsd", np.nan)
        all_metric_outputs.append(metric_output)

    # Clean/organize results and save to disk
    if exists(metric_manager):
        save_metrics(all_metric_outputs, output_dir, job_id)


def main():
    args = setup_argparse()

    run_external_eval(
        input_parquet=args.input_parquet,
        output_dir=args.output_dir,
        pfm_dataset_input=args.pfm_dataset_input,
        metric_manager_config=args.metric_manager_config,
        subset_to_indices=args.subset_to_indices,
        centering_method=args.centering_method,
        use_sequence_redesign=args.use_sequence_redesign,
        chunk_size=args.chunk_size,
        job_id=args.job_id,
        annotated_design_save_dir=args.annotated_design_save_dir,
    )


if __name__ == "__main__":
    main()
