#!/bin/bash
# =============================================================================
# launch_search.sh - Direct cluster launch for latent model search
# =============================================================================
# Run this script directly on the cluster (not from local machine)
#
# Usage:
#   ./launch_search.sh <cluster> [options] [extra_args...]
#
# Options:
#   --gpus-per-job N  GPUs per SLURM job (default: 1). Each GPU runs independent search.
#
# Examples:
#   ./launch_search.sh ord
#   ./launch_search.sh digs
#   ./launch_search.sh gb200 --gpus-per-job=4
#
# Note: If gen_njobs or eval_njobs > 1 in config, uses SLURM job arrays.
#       With --gpus-per-job=N, each array task runs N parallel searches.
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
strip_inherited_slurm_env

# Parse arguments
CLUSTER=""
GPUS_PER_JOB=1
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus-per-job=*)
            GPUS_PER_JOB="${1#*=}"
            shift
            ;;
        --gpus-per-job)
            GPUS_PER_JOB="$2"
            shift 2
            ;;
        -*)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        *)
            if [ -z "$CLUSTER" ]; then
                CLUSTER="$1"
            else
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift
            ;;
    esac
done

if [ -z "$CLUSTER" ]; then
    echo "Usage: $0 <cluster> [--gpus-per-job=N] [extra_args...]"
    echo "Available clusters: digs, ord, gb200"
    exit 1
fi

# Load cluster configuration
load_user_info "$CLUSTER" || exit 1
print_cluster_config

# Find repository root
REPO_ROOT=$(find_repo_root "$SCRIPT_DIR")
if [ -z "$REPO_ROOT" ]; then
    echo "Error: Could not find .project-root"
    exit 1
fi
cd "$REPO_ROOT"
echo "Repository root: $REPO_ROOT"

source_venv "$REPO_ROOT"
setup_pythonpath "$REPO_ROOT"
setup_env_vars

# Read configuration from search yaml
SEARCH_CONFIG_FILE="./configs/search.yaml"
if [ ! -f "$SEARCH_CONFIG_FILE" ]; then
    echo "Error: Config file not found: $SEARCH_CONFIG_FILE"
    exit 1
fi

RUN_NAME=$(grep 'run_name_' $SEARCH_CONFIG_FILE | awk '{print $2}')
RUN_NAME="${RUN_NAME}-search"
CPUS_PER_TASK=$(grep 'ncpus_' $SEARCH_CONFIG_FILE | awk '{print $2}')
GEN_NJOBS=$(grep 'gen_njobs' $SEARCH_CONFIG_FILE | awk '{print $2}')
EVAL_NJOBS=$(grep 'eval_njobs' $SEARCH_CONFIG_FILE | awk '{print $2}')

# Default to 1 if not found
GEN_NJOBS="${GEN_NJOBS:-1}"
EVAL_NJOBS="${EVAL_NJOBS:-1}"

# Calculate total logical jobs (max of gen and eval jobs)
TOTAL_LOGICAL_JOBS=$((GEN_NJOBS > EVAL_NJOBS ? GEN_NJOBS : EVAL_NJOBS))

# Calculate SLURM array size based on gpus-per-job
NUM_ARRAY_TASKS=$(( (TOTAL_LOGICAL_JOBS + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))

# Create run directory structure
RUN_DIR="$ROOT_REMOTE/$RUN_NAME"
mkdir -p "$RUN_DIR/slurm_run_outputs"
mkdir -p "$RUN_DIR/search"

# Create .env file
cat > "$RUN_DIR/.env" << EOF
# Search Environment Variables
DATA_PATH=$DATA_PATH
CACHE_DIR=$CACHE_DIR
PDB_MIRROR_PATH=$PDB_MIRROR_PATH
CCD_MIRROR_PATH=$CCD_MIRROR_PATH
FOLDSEEK_EXEC=$FOLDSEEK_REMOTE
LOGURU_LEVEL='INFO'
EOF

print_header "Job Configuration"
echo "  Run name: $RUN_NAME"
echo "  Run directory: $RUN_DIR"
echo "  CPUs per task: $CPUS_PER_TASK"
echo "  Generation jobs: $GEN_NJOBS"
echo "  Evaluation jobs: $EVAL_NJOBS"
echo "  Total logical jobs: $TOTAL_LOGICAL_JOBS"
echo "  GPUs per SLURM job: $GPUS_PER_JOB"
echo "  SLURM array tasks: $NUM_ARRAY_TASKS"
echo "  Extra args: $EXTRA_ARGS"
echo ""

# Build SBATCH script for search (single GPU per task)
SBATCH_FILE="$RUN_DIR/slurm_batch_script.sh"

cat > "$SBATCH_FILE" << EOF
#!/bin/bash
EOF

# Add account if not DIGS
if [ "$SKIP_ACCOUNT" != "true" ] && [ -n "$ACCOUNT" ]; then
    echo "#SBATCH --account $ACCOUNT" >> "$SBATCH_FILE"
fi

# Add array directive if using multiple tasks
if [ "$NUM_ARRAY_TASKS" -gt 1 ]; then
    echo "#SBATCH --array=0-$((NUM_ARRAY_TASKS - 1))" >> "$SBATCH_FILE"
fi

# Search typically runs longer
SEARCH_RUNTIME="${RUNTIME:-08:00:00}"

cat >> "$SBATCH_FILE" << EOF
#SBATCH --job-name $RUN_NAME
#SBATCH --partition $PARTITION
#SBATCH --time $SEARCH_RUNTIME
#SBATCH --nodes 1
#SBATCH --gpus-per-node $GPUS_PER_JOB
#SBATCH --ntasks-per-node $GPUS_PER_JOB
#SBATCH --cpus-per-task $CPUS_PER_TASK
#SBATCH --open-mode append
#SBATCH --requeue
#SBATCH --output ./slurm_run_outputs/slurm_%A_%a.out
#SBATCH --error ./slurm_run_outputs/slurm_%A_%a.err

# Environment setup
export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/lib/atomworks/src:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CACHE_DIR=$CACHE_DIR

# Job array and multi-GPU configuration
ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}
GPUS_PER_JOB=$GPUS_PER_JOB
GEN_NJOBS=$GEN_NJOBS
EVAL_NJOBS=$EVAL_NJOBS
TOTAL_LOGICAL_JOBS=$TOTAL_LOGICAL_JOBS

echo "Array task: \$ARRAY_TASK_ID, GPUs per job: \$GPUS_PER_JOB"
EOF

# Add container execution command
SCRIPT_SEARCH="src/proteinfoundation/generate.py"
SCRIPT_EVALUATION="src/proteinfoundation/evaluate.py"
CONTAINER_CMD=$(get_srun_cmd)

if [ "$CLUSTER_TYPE" == "digs" ]; then
    cat >> "$SBATCH_FILE" << 'EOF'

# Run search - each GPU gets its own job_id
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $GEN_NJOBS ]; then
        echo "Running search with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        CUDA_VISIBLE_DEVICES=\$LOCAL_ID $CONTAINER_CMD python $SCRIPT_SEARCH +job_id=\$JOB_ID gen_njobs=$GEN_NJOBS $EXTRA_ARGS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait

# Run evaluation - each GPU gets its own job_id
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $EVAL_NJOBS ]; then
        echo "Running evaluation with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        CUDA_VISIBLE_DEVICES=\$LOCAL_ID $CONTAINER_CMD python $SCRIPT_EVALUATION +job_id=\$JOB_ID njobs=$EVAL_NJOBS $EXTRA_ARGS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait
EOF
else
    cat >> "$SBATCH_FILE" << 'EOF'

# Run search - each GPU gets its own job_id
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $GEN_NJOBS ]; then
        echo "Running search with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        srun --exclusive --ntasks=1 --gpus=1 $CONTAINER_CMD python $SCRIPT_SEARCH +job_id=\$JOB_ID gen_njobs=$GEN_NJOBS $EXTRA_ARGS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait

# Run evaluation - each GPU gets its own job_id
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $EVAL_NJOBS ]; then
        echo "Running evaluation with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        srun --exclusive --ntasks=1 --gpus=1 $CONTAINER_CMD python $SCRIPT_EVALUATION +job_id=\$JOB_ID njobs=$EVAL_NJOBS $EXTRA_ARGS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait
EOF
fi

# Submit job
print_header "Submitting Job"
echo "SBATCH script: $SBATCH_FILE"
cd "$RUN_DIR"
sbatch slurm_batch_script.sh
echo "Job submitted!"
