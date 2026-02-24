#!/usr/bin/env python
"""CLI tool for preparing ProteinMPNN/LigandMPNN jobs on model outputs.

This script prepares MPNN jobs by:
1. Loading generated structures with condition annotations
2. Detecting fixed residues based on model annotations
3. Creating chunked JSON files for parallel processing
4. Generating SLURM submission scripts (container handling done by launch_mpnn.sh)

Example usage:
    # Basic usage with defaults
    python prepare_mpnn.py --input-dir ./outputs/generated/ --output-dir ./mpnn/

    # With chunking for parallel processing
    python prepare_mpnn.py \\
        --input-dir ./outputs/generated/ \\
        --output-dir ./mpnn/ \\
        --num-chunks 10 \\
        --num-seqs 4

Note:
    This script generates MPNN commands without container wrapping.
    Use launch_mpnn.sh for cluster-portable execution with proper container handling.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare ProteinMPNN/LigandMPNN jobs on model outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input/output
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing generated CIF structures, or path to a single CIF file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for MPNN outputs (JSON files, SLURM scripts, results).",
    )

    # Chunking
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
        help="Number of chunks for parallel SLURM jobs. Default: 1",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search input directory recursively for CIF files. Default: True",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Don't search recursively.",
    )

    # MPNN parameters
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=1,
        help="Number of sequences to generate per structure. Default: 1",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["ligand_mpnn", "protein_mpnn"],
        default="ligand_mpnn",
        help="MPNN model type. Default: ligand_mpnn",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for MPNN. Default: 1",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature. Default: 0.1",
    )
    parser.add_argument(
        "--omit-aa",
        type=str,
        default="XC",
        help="Amino acids to omit from design. Default: XC",
    )
    parser.add_argument(
        "--pack-side-chains",
        action="store_true",
        default=True,
        help="Pack side chains. Default: True",
    )
    parser.add_argument(
        "--no-pack-side-chains",
        action="store_false",
        dest="pack_side_chains",
        help="Don't pack side chains.",
    )
    parser.add_argument(
        "--repack-everything",
        action="store_true",
        default=False,
        help="Repack all residues (ignore fixed residues). Default: False",
    )
    parser.add_argument(
        "--force-hetatm",
        action="store_true",
        default=True,
        help="Force HETATM records. Default: True",
    )
    parser.add_argument(
        "--no-force-hetatm",
        action="store_false",
        dest="force_hetatm",
        help="Don't force HETATM records.",
    )
    parser.add_argument(
        "--zero-indexed",
        action="store_true",
        default=True,
        help="Use zero-indexed residue numbering. Default: True",
    )
    parser.add_argument(
        "--one-indexed",
        action="store_false",
        dest="zero_indexed",
        help="Use one-indexed residue numbering.",
    )
    parser.add_argument(
        "--ligand-mpnn-use-side-chain-context",
        action="store_true",
        default=True,
        help="Use side chain context for LigandMPNN. Default: True",
    )
    parser.add_argument(
        "--no-ligand-mpnn-use-side-chain-context",
        action="store_false",
        dest="ligand_mpnn_use_side_chain_context",
        help="Don't use side chain context for LigandMPNN.",
    )

    # Paths
    parser.add_argument(
        "--mpnn-script",
        type=str,
        required=True,
        help="Path to the MPNN run script (e.g., /path/to/fused_mpnn/run.py).",
    )

    # SLURM parameters (defaults for CPU-based MPNN)
    parser.add_argument(
        "--slurm-partition",
        type=str,
        default="cpu",
        help="SLURM partition. Default: cpu",
    )
    parser.add_argument(
        "--slurm-time",
        type=str,
        default="1:00:00",
        help="SLURM time limit. Default: 1:00:00",
    )
    parser.add_argument(
        "--slurm-cpus",
        type=int,
        default=4,
        help="Number of CPUs per task. Default: 4",
    )
    parser.add_argument(
        "--slurm-mem",
        type=str,
        default="16G",
        help="Memory per task. Default: 16G",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    from proteinfoundation.utils.mpnn_utils import prepare_mpnn_jobs

    try:
        result = prepare_mpnn_jobs(
            input_path=args.input_dir,
            output_dir=args.output_dir,
            num_chunks=args.num_chunks,
            num_seqs_per_structure=args.num_seqs,
            model_type=args.model_type,
            batch_size=args.batch_size,
            pack_side_chains=args.pack_side_chains,
            temperature=args.temperature,
            packed_suffix="",
            omit_aa=args.omit_aa,
            repack_everything=args.repack_everything,
            force_hetatm=args.force_hetatm,
            zero_indexed=args.zero_indexed,
            ligand_mpnn_use_side_chain_context=args.ligand_mpnn_use_side_chain_context,
            mpnn_script_path=args.mpnn_script,
            slurm_partition=args.slurm_partition,
            slurm_time=args.slurm_time,
            slurm_cpus=args.slurm_cpus,
            slurm_mem=args.slurm_mem,
            recursive=args.recursive,
        )

        print("\n" + "=" * 60)
        print("MPNN job preparation complete!")
        print("=" * 60)
        print(f"\nStructures found: {len(result['structure_paths'])}")
        print(f"Chunks created: {len([c for c in result['chunks'] if c])}")
        print(f"Commands generated: {len(result['commands'])}")
        print(f"\nSLURM script: {result['slurm_script_path']}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
