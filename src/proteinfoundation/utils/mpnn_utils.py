"""Utilities for running ProteinMPNN/LigandMPNN on model outputs.

This module provides functionality for:
- Loading structures and detecting fixed residues based on model annotations
- Generating MPNN input JSON files (pdb_path_multi and fixed_residues_multi)
- Chunking structures for parallel SLURM submission
- Creating SLURM submission scripts

Example usage:
    # Prepare MPNN jobs from a directory of generated structures
    from proteinfoundation.utils.mpnn_utils import prepare_mpnn_jobs

    prepare_mpnn_jobs(
        input_dir="./outputs/generated/",
        output_dir="./mpnn_outputs/",
        num_chunks=10,
        num_seqs_per_structure=4,
    )
"""

from __future__ import annotations

import json
import math
from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
from atomworks.io.parser import parse
from atomworks.ml.utils.token import apply_and_spread_token_wise
from biotite.structure import AtomArray

from foundry.utils.standard_annotations import SA_CRDSRC


def get_fixed_residues_from_structure(
    structure_path: PathLike,
) -> tuple[AtomArray, list[str]]:
    """Load a structure and determine which residues should be fixed for MPNN.

    Fixed residues are determined by checking if any of these annotations are True
    for a given residue:

    - :py:attr:`~foundry.utils.standard_annotations.SrcCoordinateConditionedRegion` mask
      (``mapped_to_coordinate_conditioned_region``), spread to token level:
      residue was mapped to a coordinate-conditioned motif.
      Includes residues where only a subset of atoms were mapped (e.g. tip-atom motifs).

    Args:
        structure_path: Path to the CIF file with condition annotations.

    Returns:
        Tuple of (AtomArray, fixed_residues) where fixed_residues is a list
        of strings in format f"{chain_id}{res_id}" (e.g., ["A73", "A10"]).
    """
    structure_path = Path(structure_path)

    # SA_CRDSRC requires load_standard_annotations=True; add_missing_atoms=False is required for compatibility.
    atom_array = parse(
        structure_path,
        load_standard_annotations=True,
        add_missing_atoms=False,
    )["asym_unit"][0]

    # Determine fixed residues at the atom level
    n_atoms = atom_array.array_length()
    fixed_mask = np.zeros(n_atoms, dtype=bool)

    # Construct token-level mask from atom-level SA_CRDSRC
    if SA_CRDSRC.has_mask(atom_array):
        token_level_mask = apply_and_spread_token_wise(atom_array, SA_CRDSRC.mask(atom_array), np.any)
        fixed_mask |= token_level_mask.astype(bool)
    else:
        raise ValueError(f"Structure {structure_path} is missing '{SA_CRDSRC.mask_name}' annotation.")

    # Convert atom-level mask to residue-level fixed residue list
    # Get unique (chain_id, res_id) pairs for fixed atoms
    fixed_residues_set = set()
    for i in np.where(fixed_mask)[0]:
        chain_id = atom_array.chain_id[i]
        res_id = atom_array.res_id[i]
        fixed_residues_set.add(f"{chain_id}{res_id}")

    return atom_array, sorted(fixed_residues_set)


def collect_structures_from_path(
    input_path: PathLike | list[PathLike],
    recursive: bool = True,
) -> list[Path]:
    """Collect CIF structure files from a path or list of paths.

    Args:
        input_path: Directory, file, or list of files/directories.
        recursive: Whether to search directories recursively.

    Returns:
        List of resolved Path objects to CIF files.
    """
    if isinstance(input_path, list):
        all_files = []
        for p in input_path:
            all_files.extend(collect_structures_from_path(p, recursive=recursive))
        return all_files

    input_path = Path(input_path)

    if input_path.is_file():
        if str(input_path.name).endswith((".cif", ".cif.gz")):
            return [input_path.absolute()]
        return []

    if input_path.is_dir():
        pattern = "**/*.cif*" if recursive else "*.cif*"
        files = list(input_path.glob(pattern))
        # Filter to only .cif and .cif.gz files
        files = [f for f in files if f.name.endswith(".cif") or f.name.endswith(".cif.gz")]
        return sorted([f.absolute() for f in files])

    return []


def chunk_list(items: list, num_chunks: int) -> list[list]:
    """Split a list into approximately equal chunks.

    Args:
        items: List to split.
        num_chunks: Number of chunks to create.

    Returns:
        List of lists, each containing a subset of items.
    """
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")

    n = len(items)
    if n == 0:
        return [[] for _ in range(num_chunks)]

    chunk_size = math.ceil(n / num_chunks)
    chunks = []
    for i in range(0, n, chunk_size):
        chunks.append(items[i : i + chunk_size])

    # Pad with empty lists if needed
    while len(chunks) < num_chunks:
        chunks.append([])

    return chunks


def generate_mpnn_json_file(
    structure_paths: list[Path],
    output_dir: PathLike,
    chunk_idx: int | None = None,
) -> Path:
    """Generate JSON file for MPNN pdb_path_multi and fixed_residues_multi.

    Args:
        structure_paths: List of paths to CIF structures.
        output_dir: Directory to save JSON files.
        chunk_idx: Optional chunk index for naming files.

    Returns:
        Path to the generated JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the combined JSON (same format for both pdb_path_multi and fixed_residues_multi)
    combined_data = {}

    for structure_path in structure_paths:
        try:
            _, fixed_residues = get_fixed_residues_from_structure(structure_path)
            combined_data[str(structure_path)] = fixed_residues
        except Exception as e:
            print(f"Warning: Failed to process {structure_path}: {e}")
            continue

    # Generate filename with optional chunk index
    suffix = f"_chunk{chunk_idx}" if chunk_idx is not None else ""
    json_path = output_dir / f"pdb_position_fixed{suffix}.json"

    with open(json_path, "w") as f:
        json.dump(combined_data, f, indent=None)

    return json_path


def generate_mpnn_command(
    pdb_json_path: PathLike,
    fixed_residues_json_path: PathLike,
    mpnn_script_path: str,
    output_dir: PathLike,
    model_type: Literal["ligand_mpnn", "protein_mpnn"] = "ligand_mpnn",
    batch_size: int = 1,
    pack_side_chains: bool = True,
    temperature: float = 0.1,
    packed_suffix: str = "",
    omit_aa: str = "XC",
    repack_everything: bool = False,
    force_hetatm: bool = True,
    zero_indexed: bool = True,
    ligand_mpnn_use_side_chain_context: bool = True,
    num_seqs_per_structure: int = 1,
) -> str:
    """Generate the MPNN Python command (without container wrapping).

    The container wrapping (apptainer/docker) should be handled by the launch script
    based on cluster type, not here.

    Args:
        pdb_json_path: Path to JSON with structure paths.
        fixed_residues_json_path: Path to JSON with fixed residues.
        output_dir: Directory for MPNN outputs.
        mpnn_script_path: Path to the MPNN run script.
        model_type: MPNN model type ("ligand_mpnn" or "protein_mpnn").
        batch_size: Batch size for MPNN.
        pack_side_chains: Whether to pack side chains.
        temperature: Sampling temperature.
        packed_suffix: Suffix for packed structures.
        omit_aa: Amino acids to omit from design.
        repack_everything: Whether to repack all residues.
        force_hetatm: Whether to force HETATM records.
        zero_indexed: Whether residue numbering is zero-indexed.
        ligand_mpnn_use_side_chain_context: Use side chain context for ligand MPNN.
        num_seqs_per_structure: Number of sequences to generate per structure.

    Returns:
        Python command string for running MPNN (no container wrapper).
    """
    cmd_parts = [
        "python -u",
        mpnn_script_path,
        f"--pdb_path_multi {pdb_json_path}",
        f"--fixed_residues_multi {fixed_residues_json_path}",
        f"--out_folder {output_dir}",
        f"--model_type {model_type}",
        f"--batch_size {batch_size}",
        f"--pack_side_chains {1 if pack_side_chains else 0}",
        f"--temperature {temperature}",
        f'--packed_suffix "{packed_suffix}"',
        f"--omit_AA {omit_aa}",
        f"--repack_everything {1 if repack_everything else 0}",
        f"--force_hetatm {1 if force_hetatm else 0}",
        f"--zero_indexed {1 if zero_indexed else 0}",
        f"--ligand_mpnn_use_side_chain_context {1 if ligand_mpnn_use_side_chain_context else 0}",
        f"--number_of_batches {num_seqs_per_structure}",
    ]

    return " ".join(cmd_parts)


def generate_slurm_script(
    job_name: str,
    commands: list[str],
    output_dir: PathLike,
    partition: str = "cpu",
    time: str = "1:00:00",
    cpus: int = 1,
    mem: str = "16G",
    array_range: str | None = None,
) -> str:
    """Generate a SLURM submission script.

    Args:
        job_name: Name for the SLURM job.
        commands: List of commands (one per array task, or single command).
        output_dir: Directory for log files.
        partition: SLURM partition.
        time: Wall time limit.
        cpus: Number of CPUs per task.
        mem: Memory per task.
        array_range: Array range (e.g., "0-9") if using array jobs.

    Returns:
        SLURM script content as a string.
    """
    output_dir = Path(output_dir)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output_dir}/logs/%x_%A_%a.log",
        f"#SBATCH --error={output_dir}/logs/%x_%A_%a.log",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
    ]

    if array_range is not None:
        lines.append(f"#SBATCH --array={array_range}")

    lines.extend(
        [
            "",
            "set -e",
            "",
        ]
    )

    if array_range is not None and len(commands) > 1:
        # Array job: select command based on SLURM_ARRAY_TASK_ID
        lines.append("# Commands indexed by SLURM_ARRAY_TASK_ID")
        lines.append("COMMANDS=(")
        for cmd in commands:
            # Escape quotes in command
            escaped_cmd = cmd.replace('"', '\\"')
            lines.append(f'    "{escaped_cmd}"')
        lines.append(")")
        lines.append("")
        lines.append('eval "${COMMANDS[$SLURM_ARRAY_TASK_ID]}"')
    else:
        # Single command or no array
        for cmd in commands:
            lines.append(cmd)

    return "\n".join(lines)


def prepare_mpnn_jobs(
    input_path: PathLike | list[PathLike],
    output_dir: PathLike,
    mpnn_script_path: str,
    num_chunks: int = 1,
    num_seqs_per_structure: int = 1,
    model_type: Literal["ligand_mpnn", "protein_mpnn"] = "ligand_mpnn",
    batch_size: int = 1,
    pack_side_chains: bool = True,
    temperature: float = 0.1,
    packed_suffix: str = "",
    omit_aa: str = "XC",
    repack_everything: bool = False,
    force_hetatm: bool = True,
    zero_indexed: bool = True,
    ligand_mpnn_use_side_chain_context: bool = True,
    slurm_partition: str = "cpu",
    slurm_time: str = "1:00:00",
    slurm_cpus: int = 4,
    slurm_mem: str = "16G",
    recursive: bool = True,
) -> dict:
    """Prepare MPNN jobs with chunking and SLURM support.

    This function:
    1. Collects all CIF structures from the input path(s)
    2. Extracts fixed residues based on model annotations
    3. Chunks structures for parallel processing
    4. Generates JSON files for each chunk
    5. Creates MPNN commands (without container wrapping - that's handled by launch script)

    Note:
        Container wrapping (apptainer/docker) is handled by the launch_mpnn.sh script
        based on cluster type, not by this function.

    Args:
        input_path: Directory, file, or list of CIF files.
        output_dir: Directory for all outputs.
        mpnn_script_path: Path to MPNN run script.
        num_chunks: Number of chunks for parallel processing.
        num_seqs_per_structure: Number of sequences per structure.
        model_type: MPNN model type.
        batch_size: Batch size for MPNN.
        pack_side_chains: Whether to pack side chains.
        temperature: Sampling temperature.
        packed_suffix: Suffix for packed structures.
        omit_aa: Amino acids to omit.
        repack_everything: Whether to repack all residues.
        force_hetatm: Whether to force HETATM records.
        zero_indexed: Whether residue numbering is zero-indexed.
        ligand_mpnn_use_side_chain_context: Use side chain context.
        slurm_partition: SLURM partition.
        slurm_time: SLURM time limit.
        slurm_cpus: CPUs per task.
        slurm_mem: Memory per task.
        recursive: Search directories recursively.

    Returns:
        Dictionary with keys:
        - "structure_paths": List of all structure paths
        - "chunks": List of chunked structure paths
        - "json_files": List of generated JSON file paths
        - "commands": List of MPNN commands (without container wrapping)
        - "slurm_script_path": Path to SLURM submission script
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    (output_dir / "logs").mkdir(exist_ok=True)

    # Collect all structures
    print(f"Collecting structures from {input_path}...")
    structure_paths = collect_structures_from_path(input_path, recursive=recursive)
    print(f"Found {len(structure_paths)} structures")

    if len(structure_paths) == 0:
        raise ValueError(f"No CIF structures found in {input_path}")

    # Chunk structures
    chunks = chunk_list(structure_paths, num_chunks)
    print(f"Split into {len(chunks)} chunks")

    # Generate JSON files and commands for each chunk
    json_files = []
    commands = []

    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue

        # Generate JSON files
        json_path = generate_mpnn_json_file(
            structure_paths=chunk,
            output_dir=output_dir,
            chunk_idx=i,
        )
        json_files.append(json_path)

        # Generate command (without container wrapping - handled by launch script)
        cmd = generate_mpnn_command(
            pdb_json_path=json_path,
            fixed_residues_json_path=json_path,
            output_dir=output_dir,
            model_type=model_type,
            batch_size=batch_size,
            pack_side_chains=pack_side_chains,
            temperature=temperature,
            packed_suffix=packed_suffix,
            omit_aa=omit_aa,
            repack_everything=repack_everything,
            force_hetatm=force_hetatm,
            zero_indexed=zero_indexed,
            ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
            num_seqs_per_structure=num_seqs_per_structure,
            mpnn_script_path=mpnn_script_path,
        )
        commands.append(cmd)

    # Generate SLURM script
    array_range = f"0-{len(commands) - 1}" if len(commands) > 1 else None
    slurm_content = generate_slurm_script(
        job_name="mpnn",
        commands=commands,
        output_dir=output_dir,
        partition=slurm_partition,
        time=slurm_time,
        cpus=slurm_cpus,
        mem=slurm_mem,
        array_range=array_range,
    )

    slurm_script_path = output_dir / "submit_mpnn.sh"
    with open(slurm_script_path, "w") as f:
        f.write(slurm_content)
    print(f"Wrote SLURM script to {slurm_script_path}")

    return {
        "structure_paths": structure_paths,
        "chunks": chunks,
        "json_files": json_files,
        "commands": commands,
        "slurm_script_path": slurm_script_path,
    }
