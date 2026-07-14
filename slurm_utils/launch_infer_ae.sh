#!/bin/bash
# =============================================================================
# launch_infer_ae.sh - Direct cluster launch for autoencoder inference
# =============================================================================
# Run this script directly on the cluster. Supports --input_dir for a 
# directory of CIF/PDB files to evaluate reconstruction quality.
#
# Usage:
#   ./launch_infer_ae.sh <cluster> [options] [extra_args...]
#
# Options:
#   --input_dir=<path>   Directory containing CIF/PDB files to evaluate
#   --output_dir=<path>  Directory for results (default: ./ae_inference)
#   --ckpt=<path>        Autoencoder checkpoint path
#   --njobs=<int>        Number of parallel jobs (default: 1)
#   --gpus-per-job=<int> GPUs per SLURM job (default: 1)
#
# Examples:
#   ./launch_infer_ae.sh ord --input_dir=/path/to/structures
#   ./launch_infer_ae.sh digs --input_dir=./my_pdbs --ckpt=/path/to/ae.ckpt
#   ./launch_infer_ae.sh gb200 --input_dir=./cifs --njobs=16 --gpus-per-job=4
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
strip_inherited_slurm_env

# Parse arguments
CLUSTER=""
INPUT_DIR=""
OUTPUT_DIR=""
CKPT_PATH=""
NJOBS=1
GPUS_PER_JOB=1
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir=*)
            INPUT_DIR="${1#*=}"
            shift
            ;;
        --output_dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --ckpt=*)
            CKPT_PATH="${1#*=}"
            shift
            ;;
        --njobs=*)
            NJOBS="${1#*=}"
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
    echo "Usage: $0 <cluster> [--input_dir=<path>] [--njobs=<int>] [--gpus-per-job=<int>] [extra_args...]"
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
source_venv "$REPO_ROOT"
setup_pythonpath "$REPO_ROOT"
setup_env_vars

# Read configuration
AE_CONFIG_FILE="./configs/inference_ae.yaml"

# Use provided checkpoint or default from config
if [ -z "$CKPT_PATH" ]; then
    CKPT_PATH=$(grep 'ckpt_file:' "$AE_CONFIG_FILE" | head -1 | awk '{print $2}')
fi

# Set output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$ROOT_REMOTE/ae_inference_$(date +%Y%m%d_%H%M%S)"
fi

# Calculate SLURM array size based on gpus-per-job
NUM_ARRAY_TASKS=$(( (NJOBS + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))

# Create output directory
mkdir -p "$OUTPUT_DIR/slurm_run_outputs"

print_header "Job Configuration"
echo "  Output directory: $OUTPUT_DIR"
echo "  Checkpoint: $CKPT_PATH"
echo "  Total jobs: $NJOBS"
echo "  GPUs per SLURM job: $GPUS_PER_JOB"
echo "  SLURM array tasks: $NUM_ARRAY_TASKS"
if [ -n "$INPUT_DIR" ]; then
    echo "  Input directory: $INPUT_DIR"
    num_files=$(find "$INPUT_DIR" -type f \( -name "*.cif" -o -name "*.pdb" \) 2>/dev/null | wc -l)
    echo "  Number of structure files: $num_files"
fi
echo "  Extra args: $EXTRA_ARGS"
echo ""

# Build SBATCH script
SBATCH_FILE="$OUTPUT_DIR/slurm_batch_script.sh"

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

cat >> "$SBATCH_FILE" << EOF
#SBATCH --job-name ae_infer
#SBATCH --partition $PARTITION
#SBATCH --time $RUNTIME
#SBATCH --nodes 1
#SBATCH --gpus-per-node $GPUS_PER_JOB
#SBATCH --ntasks-per-node $GPUS_PER_JOB
#SBATCH --cpus-per-task 8
#SBATCH --open-mode append
#SBATCH --requeue
#SBATCH --output ./slurm_run_outputs/slurm_%A_%a.out
#SBATCH --error ./slurm_run_outputs/slurm_%A_%a.err

# Environment setup
export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/lib/atomworks/src:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4

# Job array and multi-GPU configuration
ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}
GPUS_PER_JOB=$GPUS_PER_JOB
NJOBS=$NJOBS

echo "Array task: \$ARRAY_TASK_ID, GPUs per job: \$GPUS_PER_JOB"
EOF

# Build inference command
SCRIPT_AE_INFER="src/proteinfoundation/partial_autoencoder/inference.py"
CONTAINER_CMD=$(get_srun_cmd)

# Build Python arguments
PYTHON_ARGS="ckpt_file=$CKPT_PATH"
if [ -n "$INPUT_DIR" ]; then
    PYTHON_ARGS="$PYTHON_ARGS input_dir=$INPUT_DIR"
fi

if [ "$CLUSTER_TYPE" == "digs" ]; then
    cat >> "$SBATCH_FILE" << 'EOF'

# Run AE inference - each GPU gets its own job_id
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $NJOBS ]; then
        echo "Running AE inference with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        CUDA_VISIBLE_DEVICES=\$LOCAL_ID $CONTAINER_CMD python $SCRIPT_AE_INFER $PYTHON_ARGS +job_id=\$JOB_ID njobs=$NJOBS $EXTRA_ARGS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait
EOF
else
    cat >> "$SBATCH_FILE" << 'EOF'

# Run AE inference - each GPU gets its own job_id
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $NJOBS ]; then
        echo "Running AE inference with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        srun --exclusive --ntasks=1 --gpus=1 $CONTAINER_CMD python $SCRIPT_AE_INFER $PYTHON_ARGS +job_id=\$JOB_ID njobs=$NJOBS $EXTRA_ARGS &
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
cd "$OUTPUT_DIR"
sbatch slurm_batch_script.sh
echo "Job submitted!"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"
