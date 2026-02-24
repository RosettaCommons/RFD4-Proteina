#!/bin/bash
# =============================================================================
# launch_infer.sh - Direct cluster launch for latent model inference
# =============================================================================
# Run this script directly on the cluster for inference/generation.
#
# Usage:
#   ./launch_infer.sh <cluster> [options] [extra_args...]
#
# Options:
#   --no-copy          Skip copying code to run directory (use current repo)
#   --overwrite-rundir Allow overwriting existing code directory in run directory
#   --follow, -f       Follow log output after job submission (Ctrl+C to exit)
#   --gpus-per-job N   GPUs per SLURM job (default: 1). Each GPU runs independent inference.
#
# Examples:
#   ./launch_infer.sh digs ckpt_path=./checkpoints ckpt_name=model.ckpt
#   ./launch_infer.sh ord inference_experiment=myexp nsamples=100
#   ./launch_infer.sh gb200 --gpus-per-job=4 gen_njobs=8
#
# Note: If gen_njobs or eval_njobs > 1 in config, uses SLURM job arrays.
#       With --gpus-per-job=N, each array task runs N parallel inferences.
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Parse arguments
CLUSTER=""
GPUS_PER_JOB=1
COPY_CODE="true"
FOLLOW_LOGS="false"
OVERWRITE_RUNDIR="false"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-copy)
            COPY_CODE="false"
            shift
            ;;
        --overwrite-rundir)
            OVERWRITE_RUNDIR="true"
            shift
            ;;
        --follow|-f)
            FOLLOW_LOGS="true"
            shift
            ;;
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
    echo "Usage: $0 <cluster> [--gpus-per-job=N] [options] [extra_args...]"
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

# Setup PYTHONPATH
setup_pythonpath "$REPO_ROOT"
setup_env_vars

# Resolve configuration using Hydra (handles composition, interpolations, overrides)
CONFIG_NAME="inference"
print_header "Resolving Hydra Config"
echo "  Config name: $CONFIG_NAME"
echo "  Extra args: $EXTRA_ARGS"

eval $(resolve_hydra_config "$REPO_ROOT" "$CONFIG_NAME" $EXTRA_ARGS)

# Validate required values
if [ -z "$RUN_NAME" ]; then
    echo "Error: run_name_ not found in resolved config"
    exit 1
fi
RUN_NAME="${RUN_NAME}-inference"

# Read gen_njobs / eval_njobs from the resolved config
# (resolve_config.py doesn't emit these, so we still grep the config for now)
INF_CONFIG_FILE="./projects/proteinfoundation/configs/inference.yaml"
GEN_NJOBS=$(grep 'gen_njobs' "$INF_CONFIG_FILE" 2>/dev/null | awk '{print $2}')
EVAL_NJOBS=$(grep 'eval_njobs' "$INF_CONFIG_FILE" 2>/dev/null | awk '{print $2}')
GEN_NJOBS="${GEN_NJOBS:-1}"
EVAL_NJOBS="${EVAL_NJOBS:-1}"

# Calculate total logical jobs (max of gen and eval jobs)
TOTAL_LOGICAL_JOBS=$((GEN_NJOBS > EVAL_NJOBS ? GEN_NJOBS : EVAL_NJOBS))

# Calculate SLURM array size based on gpus-per-job
NUM_ARRAY_TASKS=$(( (TOTAL_LOGICAL_JOBS + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))

# Set output directory
RUN_DIR="$ROOT_REMOTE/inference/$RUN_NAME"
mkdir -p "$RUN_DIR/slurm_run_outputs"

# Clear old slurm logs to start fresh
rm -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err"

# Create symlink in active directory for easy browsing
mkdir -p "$ROOT_REMOTE/active"
ln -sfn "$RUN_DIR" "$ROOT_REMOTE/active/$RUN_NAME"

# Create symlink in repo for easy local browsing
mkdir -p "$REPO_ROOT/runs"
ln -sfn "$RUN_DIR" "$REPO_ROOT/runs/$RUN_NAME"
echo "  Local symlink: $REPO_ROOT/runs/$RUN_NAME"

# Copy code to run directory (or use current repo)
if [ "$COPY_CODE" == "true" ]; then
    if [ -d "$RUN_DIR/src" ] || [ -d "$RUN_DIR/projects/proteinfoundation/proteinfoundation" ]; then
        if [ "$OVERWRITE_RUNDIR" != "true" ]; then
            echo "Error: Code directory already exists at $RUN_DIR"
            echo "  To overwrite, use the --overwrite-rundir flag"
            echo "  To skip copying and use existing code, use the --no-copy flag"
            exit 1
        fi
        echo "Warning: Overwriting existing code directory at $RUN_DIR"
    fi

    print_header "Copying Code"
    echo "  $REPO_ROOT -> $RUN_DIR"

    rsync -a "$REPO_ROOT/src/" "$RUN_DIR/src/"
    mkdir -p "$RUN_DIR/models/rf3" && rsync -a "$REPO_ROOT/models/rf3/src/" "$RUN_DIR/models/rf3/src/"
    mkdir -p "$RUN_DIR/lib/atomworks" && rsync -a "$REPO_ROOT/lib/atomworks/src/" "$RUN_DIR/lib/atomworks/src/"

    echo "  Copying projects/proteinfoundation..."
    mkdir -p "$RUN_DIR/projects/proteinfoundation"
    rsync -a --exclude='store' --exclude='outputs' --exclude='__pycache__' --exclude='*.pyc' \
        "$REPO_ROOT/projects/proteinfoundation/proteinfoundation/" "$RUN_DIR/projects/proteinfoundation/proteinfoundation/"
    rsync -a "$REPO_ROOT/projects/proteinfoundation/configs/" "$RUN_DIR/projects/proteinfoundation/configs/"
    rsync -a "$REPO_ROOT/projects/proteinfoundation/benchmarks/" "$RUN_DIR/projects/proteinfoundation/benchmarks/"
    rsync -a "$REPO_ROOT/projects/proteinfoundation/data/" "$RUN_DIR/projects/proteinfoundation/data/"

    cp "$REPO_ROOT/.project-root" "$RUN_DIR/"
    mkdir -p "$RUN_DIR/.ipd/shebang" && rsync -a "$REPO_ROOT/.ipd/shebang/" "$RUN_DIR/.ipd/shebang/"
    mkdir -p "$RUN_DIR/.ipd/apptainer"

    # DIGS: recreate container symlink with /mnt/net prefix for compute nodes
    if [ "$CLUSTER_TYPE" = "digs" ] && [ -L "$REPO_ROOT/.ipd/apptainer/latent-dev.sif" ]; then
        CONTAINER_TARGET="$(readlink -f "$REPO_ROOT/.ipd/apptainer/latent-dev.sif" 2>/dev/null || true)"
        if [ -n "$CONTAINER_TARGET" ]; then
            [[ "$CONTAINER_TARGET" == /net/* ]] && CONTAINER_TARGET="/mnt${CONTAINER_TARGET}"
            ln -sf "$CONTAINER_TARGET" "$RUN_DIR/.ipd/apptainer/latent-dev.sif"
        fi
    fi
    CODE_DIR="$RUN_DIR"
else
    print_header "Using Code In-Place"
    echo "  $REPO_ROOT"
    CODE_DIR="$REPO_ROOT"
fi

# Create .env file
cat > "$RUN_DIR/.env" << EOF
# Proteina Inference Environment Variables
DATA_PATH=$DATA_PATH
CACHE_DIR=$CACHE_DIR
PDB_MIRROR_PATH=$PDB_MIRROR_PATH
CCD_MIRROR_PATH=$CCD_MIRROR_PATH
FOLDSEEK_EXEC=$FOLDSEEK_REMOTE
HBPLUS_PATH=$HBPLUS_PATH
MOLPROBITY_PATH=$MOLPROBITY_PATH
MACE_MODEL_PATH=$MACE_MODEL_PATH
LOGURU_LEVEL='INFO'
EOF

print_header "Job Configuration"
echo "  Run name: $RUN_NAME"
echo "  Run directory: $RUN_DIR"
echo "  Extra args: $EXTRA_ARGS"
echo "  Generation jobs: $GEN_NJOBS"
echo "  Evaluation jobs: $EVAL_NJOBS"
echo "  Total logical jobs: $TOTAL_LOGICAL_JOBS"
echo "  GPUs per SLURM job: $GPUS_PER_JOB"
echo "  SLURM array tasks: $NUM_ARRAY_TASKS"
echo "  CPUs per task: $CPUS_PER_TASK"
echo ""

# Build SBATCH script
SBATCH_FILE="$RUN_DIR/slurm_batch_script.sh"
JOB_NAME="$RUN_NAME"
NODES=1
GPUS_PER_NODE="$GPUS_PER_JOB"
NTASKS_PER_NODE=1
build_sbatch_header > "$SBATCH_FILE"

# Add array directive if using multiple tasks
if [ "$NUM_ARRAY_TASKS" -gt 1 ]; then
    echo "#SBATCH --array=0-$((NUM_ARRAY_TASKS - 1))" >> "$SBATCH_FILE"
fi

# Add output/error paths
cat >> "$SBATCH_FILE" << EOF
#SBATCH --output ./slurm_run_outputs/slurm.out
#SBATCH --error ./slurm_run_outputs/slurm.err

# Environment setup
export PYTHONPATH=$CODE_DIR/src:$CODE_DIR/models/rf3/src:$CODE_DIR/projects/proteinfoundation:$CODE_DIR/lib/atomworks/src:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Job array and multi-GPU configuration
ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}
GPUS_PER_JOB=$GPUS_PER_JOB
GEN_NJOBS=$GEN_NJOBS
EVAL_NJOBS=$EVAL_NJOBS
TOTAL_LOGICAL_JOBS=$TOTAL_LOGICAL_JOBS

echo "Array task: \$ARRAY_TASK_ID, GPUs per job: \$GPUS_PER_JOB"

# Verify scontrol works (needed for auto-requeue)
scontrol show job \$SLURM_JOB_ID > /dev/null 2>&1 && echo '[slurm] scontrol works - auto-requeue enabled' || echo '[slurm] WARNING: scontrol not working - auto-requeue disabled'
EOF

# Add DATASET_SQUASH_MOUNTS if specified
if [ -n "$DATASET_SQUASH_MOUNTS" ]; then
    echo "export DATASET_SQUASH_MOUNTS=\"$DATASET_SQUASH_MOUNTS\"" >> "$SBATCH_FILE"
fi

# Add cuEquivariance if specified
if [ -n "$USE_CUEQUIVARIANCE" ]; then
    echo "export SHOULD_USE_CUEQUIVARIANCE=$USE_CUEQUIVARIANCE" >> "$SBATCH_FILE"
fi

# Add distributed generation coordination
cat >> "$SBATCH_FILE" << 'EOF'

# Distributed generation coordination (unique port per job, master = first node)
export MASTER_PORT=$((1024 + RANDOM % 64512))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
EOF

# Add container execution commands
SCRIPT_GENERATION="$CODE_DIR/projects/proteinfoundation/proteinfoundation/generate.py"
SCRIPT_EVALUATION="$CODE_DIR/projects/proteinfoundation/proteinfoundation/evaluate.py"

if [ "$CLUSTER_TYPE" == "digs" ]; then
    cat >> "$SBATCH_FILE" << 'EOF'

# Run generation - each GPU gets its own job_id
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $GEN_NJOBS ]; then
        echo "Running generation with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        srun --exclusive --ntasks=1 --gpus=1 --cpu-bind=none \\
            --kill-on-bad-exit $PYTHON $SCRIPT_GENERATION \\
            $EXTRA_ARGS \\
            hardware.ngpus_per_node_=1 \\
            hardware.nnodes_=1 \\
            job_id=\$JOB_ID \\
            gen_njobs=\$GEN_NJOBS &
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
        srun --exclusive --ntasks=1 --gpus=1 --cpu-bind=none \\
            --kill-on-bad-exit $PYTHON $SCRIPT_EVALUATION \\
            +job_id=\$JOB_ID njobs=\$EVAL_NJOBS $EXTRA_ARGS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait
EOF
else
    CONTAINER_CMD=$(get_srun_cmd)
    cat >> "$SBATCH_FILE" << 'EOF'

# Run generation - each GPU gets its own job_id
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $GEN_NJOBS ]; then
        echo "Running generation with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        srun --exclusive --ntasks=1 --gpus=1 --cpu-bind=none $CONTAINER_CMD \\
            $PYTHON $SCRIPT_GENERATION \\
            $EXTRA_ARGS \\
            hardware.ngpus_per_node_=1 \\
            hardware.nnodes_=1 \\
            job_id=\$JOB_ID \\
            gen_njobs=\$GEN_NJOBS &
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
        srun --exclusive --ntasks=1 --gpus=1 --cpu-bind=none $CONTAINER_CMD \\
            $PYTHON $SCRIPT_EVALUATION \\
            +job_id=\$JOB_ID njobs=\$EVAL_NJOBS $EXTRA_ARGS &
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

# Check for already running job
EXISTING_RUNNING=$(squeue -u "$USER" --name="$RUN_NAME" -h -o "%i %T" 2>/dev/null | grep "RUNNING" | head -1)
if [ -n "$EXISTING_RUNNING" ]; then
    EXISTING_ID=$(echo "$EXISTING_RUNNING" | awk '{print $1}')
    echo "Warning: Job '$RUN_NAME' already RUNNING (ID: $EXISTING_ID)"
    echo "  Cancel: scancel $EXISTING_ID | Monitor: tail -f $RUN_DIR/slurm_run_outputs/slurm.err"
    read -p "Continue with submission? [y/N] " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

if [ -n "$WANDBKEY" ]; then
    export WANDB_API_KEY="$WANDBKEY"
fi

SBATCH_OUTPUT=$(sbatch slurm_batch_script.sh)
JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+')
echo "$SBATCH_OUTPUT"

echo ""
echo "Monitor: squeue -u $USER --name=$RUN_NAME"
echo "Logs:    tail -f $RUN_DIR/slurm_run_outputs/slurm.err"

# Follow logs if requested
if [ "$FOLLOW_LOGS" == "true" ]; then
    echo -e "\nFollowing logs (Ctrl+C to exit)...\n"
    while [ ! -f "$RUN_DIR/slurm_run_outputs/slurm.err" ]; do
        squeue -u "$USER" --name="$RUN_NAME" -h 2>/dev/null | grep -q . || { echo "Job not in queue"; break; }
        sleep 2
    done
    tail -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err" 2>/dev/null
fi
