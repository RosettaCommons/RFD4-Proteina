#!/bin/bash
# =============================================================================
# launch_rf3_inference.sh - Run RF3 inference on any supported cluster
# =============================================================================
# Usage:
#   ./launch_rf3_inference.sh <cluster> [options] [input_file] [extra_args...]
#
# Options:
#   --no-copy      Skip copying code to run directory (use current repo)
#   --follow, -f   Follow log output after job submission (Ctrl+C to exit)
#   --gpus N       Number of GPUs to request (default: 4)
#
# Examples:
#   ./launch_rf3_inference.sh ord
#   ./launch_rf3_inference.sh ord /path/to/my_input.json
#   ./launch_rf3_inference.sh ord /path/to/my_input.cif num_steps=50
#   ./launch_rf3_inference.sh ord --gpus 1 /path/to/input.json
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Parse arguments
CLUSTER=""
GPUS=4
COPY_CODE="true"
FOLLOW_LOGS="false"
INPUT_FILE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-copy)
            COPY_CODE="false"
            shift
            ;;
        --follow|-f)
            FOLLOW_LOGS="true"
            shift
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --gpus=*)
            GPUS="${1#*=}"
            shift
            ;;
        -*)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        *)
            if [ -z "$CLUSTER" ]; then
                CLUSTER="$1"
            elif [ -z "$INPUT_FILE" ] && [[ "$1" != *"="* ]]; then
                INPUT_FILE="$1"
            else
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift
            ;;
    esac
done

if [ -z "$CLUSTER" ]; then
    echo "Usage: $0 <cluster> [--gpus N] [--no-copy] [--follow] [input_file] [extra_args...]"
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

# Default input file (RF3 example)
if [ -z "$INPUT_FILE" ]; then
    if [ -f "$REPO_ROOT/models/rf3/tests/data/5vht_from_json.json" ]; then
        INPUT_FILE="$REPO_ROOT/models/rf3/tests/data/5vht_from_json.json"
    else
        echo "Error: No input file specified and default example not found"
        exit 1
    fi
fi

# Create run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="rf3_infer_${TIMESTAMP}"
RUN_DIR="$ROOT_REMOTE/rf3_inference/$RUN_NAME"
mkdir -p "$RUN_DIR/slurm_run_outputs"
mkdir -p "$RUN_DIR/outputs"

# Create symlink in active directory for easy browsing
mkdir -p "$ROOT_REMOTE/active"
ln -sfn "$RUN_DIR" "$ROOT_REMOTE/active/$RUN_NAME"

# Create symlink in repo for easy local browsing
mkdir -p "$REPO_ROOT/runs"
ln -sfn "$RUN_DIR" "$REPO_ROOT/runs/$RUN_NAME"
echo "  Local symlink: $REPO_ROOT/runs/$RUN_NAME"

# Copy code to run directory (or use current repo)
if [ "$COPY_CODE" == "true" ]; then
    print_header "Copying Code"
    echo "  $REPO_ROOT -> $RUN_DIR"

    copy_code_to_dir "$REPO_ROOT" "$RUN_DIR" "$CLUSTER"
    CODE_DIR="$RUN_DIR"
else
    print_header "Using Code In-Place"
    echo "  $REPO_ROOT"
    CODE_DIR="$REPO_ROOT"
fi

# Create .env file from ENV_VARS array
write_env_file "$RUN_DIR/.env"

print_header "RF3 Inference Configuration"
echo "  Run name: $RUN_NAME"
echo "  Run directory: $RUN_DIR"
echo "  Input file: $INPUT_FILE"
echo "  GPUs: $GPUS"
echo "  CPUs per task: $CPUS_PER_TASK"
print_mount_summary
echo "  Extra args: $EXTRA_ARGS"
echo ""

# Build SBATCH script
SBATCH_FILE="$RUN_DIR/slurm_batch_script.sh"
JOB_NAME="rf3_inference"
NODES=1
GPUS_PER_NODE="$GPUS"
NTASKS_PER_NODE=1
build_sbatch_header > "$SBATCH_FILE"

cat >> "$SBATCH_FILE" << EOF
#SBATCH --output ./slurm_run_outputs/slurm.out
#SBATCH --error ./slurm_run_outputs/slurm.err

# Environment setup
export PYTHONPATH=$CODE_DIR/src:$CODE_DIR/models/rf3/src:$CODE_DIR/lib/atomworks/src:\$PYTHONPATH
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "=============================================="
echo "RF3 Inference Job Started"
echo "=============================================="
echo "  Node:          \$(hostname)"
echo "  Date:          \$(date)"
echo "  Input:         $INPUT_FILE"
echo "  Output dir:    $RUN_DIR/outputs"
echo "=============================================="
EOF

# Path to inference script
INFERENCE_SCRIPT="$CODE_DIR/models/rf3/src/rf3/inference.py"

if [ "$CLUSTER_TYPE" == "digs" ]; then
    # DIGS: Use shebang/apptainer
    cat >> "$SBATCH_FILE" << EOF

# Run RF3 inference
srun --cpu-bind=none --kill-on-bad-exit \\
    python $INFERENCE_SCRIPT \\
    inputs='$INPUT_FILE' \\
    out_dir='$RUN_DIR/outputs' \\
    ckpt_path='$RF3_CKPT_PATH' \\
    $EXTRA_ARGS
EOF
else
    CONTAINER_CMD=$(get_srun_cmd)
    CONTAINER_MOUNTS_STR=$(build_container_mounts_string)
    CONTAINER_ENV_CMDS=$(build_container_env_cmds)

    cat >> "$SBATCH_FILE" << EOFSCRIPT
# Container mounts (auto-generated from CONTAINER_MOUNTS array)
CONTAINER_MOUNTS="$CONTAINER_MOUNTS_STR"

# Run RF3 inference with container
srun --cpu-bind=none --kill-on-bad-exit $CONTAINER_CMD \\
    --container-mounts="\${CONTAINER_MOUNTS}" \\
    --container-workdir=$RUN_DIR \\
    bash -c "\\
        $CONTAINER_ENV_CMDS && \\
        python -c \"import torch; print('torch=' + torch.__version__); print('cuda_available=' + str(torch.cuda.is_available()))\" || true && \\
        exec python $INFERENCE_SCRIPT \\
            inputs='$INPUT_FILE' \\
            out_dir='$RUN_DIR/outputs' \\
            $EXTRA_ARGS\\
    "
EOFSCRIPT
fi

cat >> "$SBATCH_FILE" << 'EOF'

echo ""
echo "=============================================="
echo "RF3 Inference Complete"
echo "=============================================="
echo "  Date:          $(date)"
echo "=============================================="
EOF

chmod +x "$SBATCH_FILE"

# Submit job
print_header "Submitting Job"
echo "SBATCH script: $SBATCH_FILE"
cd "$RUN_DIR"

# Check for already running job
EXISTING_RUNNING=$(squeue -u "$USER" --name="rf3_inference" -h -o "%i %T" 2>/dev/null | grep "RUNNING" | head -1)
if [ -n "$EXISTING_RUNNING" ]; then
    EXISTING_ID=$(echo "$EXISTING_RUNNING" | awk '{print $1}')
    echo "Warning: RF3 inference job already RUNNING (ID: $EXISTING_ID)"
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
echo "Monitor: squeue -j $JOB_ID"
echo "Logs:    tail -f $RUN_DIR/slurm_run_outputs/slurm.err"
echo "Output:  ls -la $RUN_DIR/outputs/"

# Follow logs if requested
if [ "$FOLLOW_LOGS" == "true" ]; then
    echo -e "\nFollowing logs (Ctrl+C to exit)...\n"
    while [ ! -f "$RUN_DIR/slurm_run_outputs/slurm.err" ]; do
        squeue -u "$USER" --name="rf3_inference" -h 2>/dev/null | grep -q . || { echo "Job not in queue"; break; }
        sleep 2
    done
    tail -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err" 2>/dev/null
fi
