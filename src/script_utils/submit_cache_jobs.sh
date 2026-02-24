#!/bin/bash
#
# Submit SLURM job array to generate cache files for PDB interfaces dataset
#
# This script submits a job array where each job processes a chunk of examples
# from the filtered dataset. Jobs run in parallel across the cluster.
#
# Usage:
#   bash submit_cache_jobs.sh

set -e  # Exit on error

# Configuration
EXAMPLES_PER_JOB=1000
TOTAL_EXAMPLES=232752  # Total after filtering and deduplication on (pdb_id, assembly_id)
CACHE_DIR="/path/to/shared/storage"
PARQUET_PATH="/path/to/shared/storage"
LOG_DIR="./slurm_cache_logs"

# Calculate number of jobs needed
NUM_JOBS=$(( (TOTAL_EXAMPLES + EXAMPLES_PER_JOB - 1) / EXAMPLES_PER_JOB ))  # Ceiling division
MAX_ARRAY_ID=$(( NUM_JOBS - 1 ))

echo "============================================"
echo "Cache Generation Job Submission"
echo "============================================"
echo "Total examples:      ${TOTAL_EXAMPLES}"
echo "Examples per job:    ${EXAMPLES_PER_JOB}"
echo "Number of jobs:      ${NUM_JOBS}"
echo "Array indices:       0-${MAX_ARRAY_ID}"
echo "Cache directory:     ${CACHE_DIR}"
echo "Log directory:       ${LOG_DIR}"
echo "============================================"
echo ""

# Create log directory
mkdir -p "${LOG_DIR}"

# Get the project directory (where this script is located)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create the SLURM batch script
BATCH_SCRIPT="${LOG_DIR}/cache_generation_job.sh"

cat > "${BATCH_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=cache_gen
#SBATCH --account project
#SBATCH -p cpu,cpu_long
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=./slurm_cache_logs/cache_%a.out
#SBATCH --error=./slurm_cache_logs/cache_%a.err

# Calculate start and end indices for this job
START_IDX=\$(( SLURM_ARRAY_TASK_ID * ${EXAMPLES_PER_JOB} ))
END_IDX=\$(( START_IDX + ${EXAMPLES_PER_JOB} ))

echo "=========================================="
echo "Job \${SLURM_ARRAY_TASK_ID} started"
echo "Start index: \${START_IDX}"
echo "End index:   \${END_IDX}"
echo "Hostname:    \$(hostname)"
echo "Project dir: ${PROJECT_DIR}"
echo "=========================================="
echo ""

# Change to project directory
cd "${PROJECT_DIR}"

# Activate virtual environment (from proteinfoundation, go up 2 levels to reach latent/.venv)
# source ../../.venv/bin/activate

# Run the cache generation script
/path/to/shared/storage scripts/generate_cache.py \\
    --start-idx \${START_IDX} \\
    --end-idx \${END_IDX} \\
    --cache-dir "${CACHE_DIR}" \\
    --parquet-path "${PARQUET_PATH}"

echo ""
echo "Job \${SLURM_ARRAY_TASK_ID} completed"
EOF

# Make the batch script executable
chmod +x "${BATCH_SCRIPT}"

echo "Created batch script: ${BATCH_SCRIPT}"
echo ""

# Submit the job array
echo "Submitting job array..."
sbatch --array=0-${MAX_ARRAY_ID} "${BATCH_SCRIPT}"

echo ""
echo "Jobs submitted successfully!"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in:"
echo "  ${LOG_DIR}/cache_*.out"
echo "  ${LOG_DIR}/cache_*.err"
echo ""
echo "Cancel all jobs with:"
echo "  scancel -u \$USER -n cache_gen"
