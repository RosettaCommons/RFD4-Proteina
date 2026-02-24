#!/bin/bash
#SBATCH --job-name=populate_pdb_cache
#SBATCH --account project
#SBATCH --output=populate_cache_%A_%a.out
#SBATCH --error=populate_cache_%A_%a.err
#SBATCH --partition cpu,cpu_long
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --array=0-10

# SLURM Array Job to pre-populate PDB cache using 100 parallel workers
# This should be run once before training to avoid slow first epoch
#
# Features:
#   - Parses PDB structures in parallel using SLURM array jobs
#   - Each worker processes a different subset of structures
#   - Caches parsed structures for fast training
#   - Generates ligand conformers using RDKit (if enabled)
#
# Requirements:
#   - RDKit installed (for conformer generation)
#   - atomworks package
#   - CCD and PDB mirrors accessible
#
# Usage:
#   sbatch scripts/populate_pdb_cache.sbatch
#
# Array Job Configuration:
#   --array=0-99: 100 parallel workers
#   Each worker processes every 100th structure (worker 0: 0,100,200..., worker 1: 1,101,201...)
#
# Adjust resources as needed:
#   --array=0-49: Use 50 workers instead of 100
#   --mem: Increase if you get OOM errors (16GB per worker should be enough)
#   --time: Adjust based on dataset size (4 hours for ~100k structures)

echo "=========================================="
echo "PDB Cache Population Job (Array Worker)"
echo "=========================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Total Array Tasks: $SLURM_ARRAY_TASK_COUNT"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo ""

# Activate environment (adjust as needed)
# source /path/to/your/conda/activate your_env
# or
# module load python/3.10
# module load cuda/12.1

# Set environment variables for atomworks parsing
export CCD_MIRROR_PATH=/path/to/shared/storage
export PDB_MIRROR_PATH=/path/to/shared/storage

# Conformer generation settings
# Set to 1 to enable RDKit-based conformer generation for ligands during parsing
# This generates 3D conformers for ligands that don't have coordinates or have poor quality
export ATOMWORKS_GENERATE_CONFORMERS=1

# Number of conformers to generate per ligand (default: 10)
export ATOMWORKS_MAX_CONFORMERS=10

# Conformer generation timeout in seconds (default: 30)
export ATOMWORKS_CONFORMER_TIMEOUT=30

# Conformer RMSD threshold for pruning similar conformers (default: 0.5 Angstrom)
export ATOMWORKS_CONFORMER_RMSD_THRESHOLD=0.5

# Configuration
PARQUET_PATH="/path/to/shared/storage"
CACHE_DIR="/path/to/shared/storage"

# Array job configuration
TASK_ID=${SLURM_ARRAY_TASK_ID}
TOTAL_TASKS=${SLURM_ARRAY_TASK_COUNT:-100}

echo "Configuration:"
echo "  Parquet: $PARQUET_PATH"
echo "  Cache dir: $CACHE_DIR"
echo "  CCD mirror: $CCD_MIRROR_PATH"
echo "  PDB mirror: $PDB_MIRROR_PATH"
echo ""
echo "Array Job Configuration:"
echo "  This worker (task ID): $TASK_ID"
echo "  Total workers: $TOTAL_TASKS"
echo "  Processing every ${TOTAL_TASKS}th structure starting at index $TASK_ID"
echo ""
echo "Conformer Generation:"
echo "  Enabled: $ATOMWORKS_GENERATE_CONFORMERS"
echo "  Max conformers: $ATOMWORKS_MAX_CONFORMERS"
echo "  Timeout: $ATOMWORKS_CONFORMER_TIMEOUT sec"
echo "  RMSD threshold: $ATOMWORKS_CONFORMER_RMSD_THRESHOLD Å"
echo ""

# Run cache population for this worker's subset
cd /path/to/shared/storage

/path/to/shared/storage scripts/populate_pdb_cache.py \
    --parquet "$PARQUET_PATH" \
    --cache-dir "$CACHE_DIR" \
    --ccd-mirror "$CCD_MIRROR_PATH" \
    --array-task-id "$TASK_ID" \
    --array-total-tasks "$TOTAL_TASKS"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE

