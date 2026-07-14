#!/bin/bash
# =============================================================================
# launch_external_eval.sh - Direct cluster launch for external evaluation
# =============================================================================
# Run this script directly on the cluster for external evaluation
#
# Usage:
#   ./launch_external_eval.sh <cluster> [options]
#
# Required arguments (via options):
#   --input-parquet <path>              Path to input parquet file
#   --output-dir <path>                 Output directory for results
#   --pfm-dataset-input <path>          Path to PFM dataset input
#   --centering-method <method>         Centering method for motif coordinates
#   --use-sequence-redesign <bool>      Whether to use sequence redesign (true/false)
#
# Optional arguments:
#   --metric-manager-config <path>      Path to metric manager config (if omitted, no metrics computed)
#   --annotated-design-save-dir <path>  Directory to save annotated designs as CIFs
#   --no-copy                           Skip copying code to run directory
#   --overwrite-rundir                  Allow overwriting existing code directory in run directory
#   --follow, -f                        Follow log output after submission
#   --no-cleanup                        Don't wait for jobs to complete (for multi-job runs)
#   --chunk-size <size>                 Chunk size (integer or 0-1 fraction)
#   --njobs <n>                         Number of array jobs (default: 1)
#   --gpus-per-job <n>                  GPUs per job (default: 1)
#   --time <time>                       Time limit per job (default: 6:00:00)
#   --partition <name>                  SLURM partition (overrides profile/cluster default)
#   --gres <spec>                       GPU resource specification (e.g., gpu:a100:2, overrides profile)
#   --mem <size>                        Memory per node (e.g., 64G, 512G; overrides profile default)
#   --cpus-per-task <n>                 CPUs per task (overrides profile default)
#   --run-name <name>                   Custom run name (default: auto-generated)
#
# Examples:
#   # Basic usage with default profile
#   ./launch_external_eval.sh digs \
#       --input-parquet /path/to/data.parquet \
#       --output-dir /path/to/output \
#       --pfm-dataset-input /path/to/dataset.json \
#       --metric-manager-config /path/to/config.yaml
#
#   # With chunking and custom resources (overriding profile defaults)
#   ./launch_external_eval.sh digs \
#       --input-parquet /path/to/data.parquet \
#       --output-dir /path/to/output \
#       --pfm-dataset-input /path/to/dataset.json \
#       --metric-manager-config /path/to/config.yaml \
#       --chunk-size 100 --njobs 10 --time 12:00:00 \
#       --partition gpu-eval --gres gpu:a100:2 --mem 64G --cpus-per-task 16
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
strip_inherited_slurm_env

# Parse arguments
CLUSTER=""
COPY_CODE="true"
OVERWRITE_RUNDIR="false"
FOLLOW_LOGS="false"
WAIT_FOR_COMPLETION=""  # Will be set based on NJOBS if not explicitly specified
INPUT_PARQUET=""
OUTPUT_DIR=""
PFM_DATASET_INPUT=""
METRIC_MANAGER_CONFIG=""
CHUNK_SIZE=""
NJOBS=1
GPUS_PER_JOB=1
TIME="6:00:00"
CUSTOM_PARTITION=""
CUSTOM_GRES=""
CUSTOM_MEM=""
CUSTOM_CPUS_PER_TASK=""
RUN_NAME=""
CENTERING_METHOD=""
USE_SEQUENCE_REDESIGN="false"
ANNOTATED_DESIGN_SAVE_DIR=""

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
        --no-cleanup)
            WAIT_FOR_COMPLETION="false"
            shift
            ;;
        --input-parquet)
            INPUT_PARQUET="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --pfm-dataset-input)
            PFM_DATASET_INPUT="$2"
            shift 2
            ;;
        --metric-manager-config)
            METRIC_MANAGER_CONFIG="$2"
            shift 2
            ;;
        --centering-method)
            CENTERING_METHOD="$2"
            shift 2
            ;;
        --use-sequence-redesign)
            USE_SEQUENCE_REDESIGN="$2"
            shift 2
            ;;
        --annotated-design-save-dir)
            ANNOTATED_DESIGN_SAVE_DIR="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --njobs)
            NJOBS="$2"
            shift 2
            ;;
        --gpus-per-job)
            GPUS_PER_JOB="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --partition)
            CUSTOM_PARTITION="$2"
            shift 2
            ;;
        --gres)
            CUSTOM_GRES="$2"
            shift 2
            ;;
        --mem)
            CUSTOM_MEM="$2"
            shift 2
            ;;
        --cpus-per-task)
            CUSTOM_CPUS_PER_TASK="$2"
            shift 2
            ;;
        --run-name)
            RUN_NAME="$2"
            shift 2
            ;;
        *)
            if [ -z "$CLUSTER" ]; then
                CLUSTER="$1"
            else
                echo "Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$CLUSTER" ]; then
    echo "Usage: $0 <cluster> --input-parquet <path> --output-dir <path> --pfm-dataset-input <path> --centering-method <method> --use-sequence-redesign <bool> [options]"
    echo "Available clusters: digs, ord, gb200"
    exit 1
fi

if [ -z "$INPUT_PARQUET" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$PFM_DATASET_INPUT" ] || [ -z "$CENTERING_METHOD" ] || [ -z "$USE_SEQUENCE_REDESIGN" ]; then
    echo "Error: Missing required arguments"
    echo "Required: --input-parquet, --output-dir, --pfm-dataset-input, --centering-method, --use-sequence-redesign"
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

# Generate run name if not specified
if [ -z "$RUN_NAME" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="external_eval_${TIMESTAMP}"
fi

# Set default for WAIT_FOR_COMPLETION based on NJOBS
if [ -z "$WAIT_FOR_COMPLETION" ]; then
    if [ "$NJOBS" -gt 1 ]; then
        WAIT_FOR_COMPLETION="true"
    else
        WAIT_FOR_COMPLETION="false"
    fi
fi

# Create run directory structure
RUN_DIR="$ROOT_REMOTE/external_eval/$RUN_NAME"
mkdir -p "$RUN_DIR/slurm_run_outputs"

# Clear old slurm logs to start fresh
rm -f "$RUN_DIR/slurm_run_outputs/slurm-"*.out "$RUN_DIR/slurm_run_outputs/slurm-"*.err

# Create symlink in active directory for easy browsing
mkdir -p "$ROOT_REMOTE/active"
ln -sfn "$RUN_DIR" "$ROOT_REMOTE/active/$RUN_NAME"

# Create symlink in repo for easy local browsing
mkdir -p "$REPO_ROOT/runs"
ln -sfn "$RUN_DIR" "$REPO_ROOT/runs/$RUN_NAME"
echo "  Local symlink: $REPO_ROOT/runs/$RUN_NAME"

# Copy code to run directory (or use current repo)
if [ "$COPY_CODE" == "true" ]; then
    # Check if code directory already exists
    if [ -d "$RUN_DIR/src" ] || [ -d "$RUN_DIR/projects/proteinfoundation/proteinfoundation" ]; then
        if [ "$OVERWRITE_RUNDIR" != "true" ]; then
            echo "Error: Code directory already exists at $RUN_DIR"
            echo "  The run directory contains existing code from a previous run."
            echo "  To create a new run directory, choose a different --run-name"
            echo "  To overwrite, use the --overwrite-rundir flag"
            echo "  To skip copying and use existing code, use the --no-copy flag"
            exit 1
        fi
        echo "Warning: Overwriting existing code directory at $RUN_DIR"
    fi

    print_header "Copying Code"
    echo "  $REPO_ROOT -> $RUN_DIR"
    copy_code_to_dir "$REPO_ROOT" "$RUN_DIR" "$CLUSTER"
    CODE_DIR="$RUN_DIR"
else
    print_header "Using Code In-Place"
    echo "  $REPO_ROOT"
    CODE_DIR="$REPO_ROOT"
fi

# Create .env file from ENV_VARS array (defined in .user_info)
write_env_file "$RUN_DIR/.env"

print_header "Job Configuration"
echo "  Run name: $RUN_NAME"
echo "  Run directory: $RUN_DIR"
echo "  Input parquet: $INPUT_PARQUET"
echo "  Output directory: $OUTPUT_DIR"
echo "  PFM dataset input: $PFM_DATASET_INPUT"
echo "  Metric manager config: ${METRIC_MANAGER_CONFIG:-None}"
echo "  Annotated design save dir: ${ANNOTATED_DESIGN_SAVE_DIR:-None}"
echo "  Centering method: $CENTERING_METHOD"
echo "  Use sequence redesign: $USE_SEQUENCE_REDESIGN"
echo "  Chunk size: ${CHUNK_SIZE:-N/A}"
echo "  Number of jobs: $NJOBS"
echo "  GPUs per job: $GPUS_PER_JOB"
echo "  Time per job: $TIME"

# Show partition (custom or from profile)
if [ -n "$CUSTOM_PARTITION" ]; then
    echo "  Partition: $CUSTOM_PARTITION (custom override)"
elif [ -n "$PARTITION" ]; then
    echo "  Partition: $PARTITION (from profile)"
fi

# Show GRES info
if [ -n "$CUSTOM_GRES" ]; then
    echo "  GRES: $CUSTOM_GRES (custom override)"
elif [ -n "$GRES" ]; then
    echo "  GRES: $GRES (from profile)"
fi

# Show memory
if [ -n "$CUSTOM_MEM" ]; then
    echo "  Memory: $CUSTOM_MEM (custom override)"
elif [ -n "$MEM" ]; then
    echo "  Memory: $MEM (from profile)"
fi

# Show CPUs
if [ -n "$CUSTOM_CPUS_PER_TASK" ]; then
    echo "  CPUs per task: $CUSTOM_CPUS_PER_TASK (custom override)"
elif [ -n "$CPUS_PER_TASK" ] && [ "$CPUS_PER_TASK" != "8" ]; then
    echo "  CPUs per task: $CPUS_PER_TASK (from profile)"
fi
print_mount_summary
echo ""

# Build SBATCH script
SBATCH_FILE="$RUN_DIR/slurm_batch_script.sh"
JOB_NAME="$RUN_NAME"
NODES="1"
GPUS_PER_NODE="$GPUS_PER_JOB"
NTASKS_PER_NODE=1
CPUS_PER_TASK="${CUSTOM_CPUS_PER_TASK:-${CPUS_PER_TASK:-4}}"
RUNTIME="$TIME"

# Override settings with custom values if provided
if [ -n "$CUSTOM_PARTITION" ]; then
    PARTITION="$CUSTOM_PARTITION"
fi

if [ -n "$CUSTOM_MEM" ]; then
    MEM="$CUSTOM_MEM"
fi

if [ -n "$CUSTOM_GRES" ]; then
    GRES="$CUSTOM_GRES"
fi

# Build the SBATCH header - common.sh handles GRES vs GPUS_PER_NODE conflict
build_sbatch_header > "$SBATCH_FILE"

# Add array directive if using multiple jobs
if [ "$NJOBS" -gt 1 ]; then
    echo "#SBATCH --array=0-$((NJOBS - 1))" >> "$SBATCH_FILE"
fi

# Add output/error paths (with array task ID if applicable)
if [ "$NJOBS" -gt 1 ]; then
    cat >> "$SBATCH_FILE" << EOF
#SBATCH --output ./slurm_run_outputs/slurm-%A_%a.out
#SBATCH --error ./slurm_run_outputs/slurm-%A_%a.err
EOF
else
    cat >> "$SBATCH_FILE" << EOF
#SBATCH --output ./slurm_run_outputs/slurm.out
#SBATCH --error ./slurm_run_outputs/slurm.err
EOF
fi

# Add environment setup
cat >> "$SBATCH_FILE" << EOF

# Environment setup (all clusters)
export PYTHONPATH=$CODE_DIR/projects/proteinfoundation:$CODE_DIR/projects/proteinfoundation/lib/atomworks/src:\$PYTHONPATH
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
EOF

# Apptainer clusters: pass cluster identity to exec_core.sh (via shebang)
if [ "$CONTAINER_TYPE" == "apptainer" ]; then
    cat >> "$SBATCH_FILE" << EOF

# DIGS-specific environment settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

# Distributed coordination (unique port per job, master = first node)
export MASTER_PORT=$((1024 + RANDOM % 64512))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Cluster identity — read by exec_core.sh (shebang) to source .ipd/.env.\${CLUSTER}
export CLUSTER=$CLUSTER
export CONTAINER_TYPE=$CONTAINER_TYPE
EOF
fi

# Add DATASET_SQUASH_MOUNTS if specified
if [ -n "$DATASET_SQUASH_MOUNTS" ]; then
    echo "export DATASET_SQUASH_MOUNTS=\"$DATASET_SQUASH_MOUNTS\"" >> "$SBATCH_FILE"
fi

# Add cuEquivariance control
if [ -n "$USE_CUEQUIVARIANCE" ]; then
    echo "export SHOULD_USE_CUEQUIVARIANCE=$USE_CUEQUIVARIANCE" >> "$SBATCH_FILE"
fi

# DDP coordination (unique port per job, master = first node)
cat >> "$SBATCH_FILE" << 'EOF'

export MASTER_PORT=$((1024 + RANDOM % 64512))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
EOF

# Pre-spawn AF3 daemon block when the metric-manager-config YAML references AF3Oracle.
AF3_NEEDED="false"
if [ -n "$METRIC_MANAGER_CONFIG" ] && [ -f "$METRIC_MANAGER_CONFIG" ] && \
   grep -qE '^[[:space:]]*_target_:[[:space:]]*proteinfoundation\.metrics\.af3_oracle\.AF3Oracle' "$METRIC_MANAGER_CONFIG"; then
    AF3_NEEDED="true"
fi
if [ "$CONTAINER_TYPE" == "apptainer" ] && [ "$AF3_NEEDED" == "true" ]; then
    echo "  AF3Oracle detected in metric-manager-config — emitting AF3 daemon pre-spawn block"
    emit_af3_daemon_block "$CODE_DIR" "$GPUS_PER_NODE" >> "$SBATCH_FILE"
fi

# Build the command
SCRIPT_EVAL="$CODE_DIR/projects/proteinfoundation/proteinfoundation/external_eval.py"

# Add job array logic if needed
if [ "$NJOBS" -gt 1 ]; then
    cat >> "$SBATCH_FILE" << 'EOF'

# Get job ID from SLURM array task ID
JOB_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "Running job ID: $JOB_ID"
EOF
fi

# Build command arguments
CMD_ARGS="--input-parquet $INPUT_PARQUET --output-dir $OUTPUT_DIR --pfm-dataset-input $PFM_DATASET_INPUT --centering-method $CENTERING_METHOD --use-sequence-redesign $USE_SEQUENCE_REDESIGN"

if [ -n "$METRIC_MANAGER_CONFIG" ]; then
    CMD_ARGS="$CMD_ARGS --metric-manager-config $METRIC_MANAGER_CONFIG"
fi

if [ -n "$ANNOTATED_DESIGN_SAVE_DIR" ]; then
    CMD_ARGS="$CMD_ARGS --annotated-design-save-dir $ANNOTATED_DESIGN_SAVE_DIR"
fi

if [ -n "$CHUNK_SIZE" ] && [ "$NJOBS" -gt 1 ]; then
    CMD_ARGS="$CMD_ARGS --chunk-size $CHUNK_SIZE --job-id \$JOB_ID"
fi

# Add container execution command
if [ "$CONTAINER_TYPE" == "apptainer" ]; then
    # Apptainer: shebang in external_eval.py handles container execution via exec_core.sh
    cat >> "$SBATCH_FILE" << EOF

# Run external evaluation (shebang handles apptainer container)
srun --cpu-bind=none --kill-on-bad-exit $SCRIPT_EVAL $CMD_ARGS
EOF
else
    # OCI/GB200: Docker/Pyxis container via srun --container-image
    CONTAINER_CMD=$(get_srun_cmd)
    CONTAINER_MOUNTS_STR=$(build_container_mounts_string)
    SQFS_MOUNT_CMDS=$(build_squashfs_mount_cmds)
    CONTAINER_ENV_CMDS=$(build_container_env_cmds)

    if [ "${USE_SQUASHFS_MOUNTS:-false}" == "true" ] && [ -n "$SQFS_MOUNT_CMDS" ]; then
        cat >> "$SBATCH_FILE" << EOFSCRIPT
# Container mounts (auto-generated from CONTAINER_MOUNTS and SQUASHFS_MOUNTS arrays)
CONTAINER_MOUNTS="$CONTAINER_MOUNTS_STR"

# Run external evaluation with container (squashfs mounting enabled)
srun --cpu-bind=none --kill-on-bad-exit $CONTAINER_CMD \\
    --container-mounts="\${CONTAINER_MOUNTS}" \\
    --container-workdir=$RUN_DIR \\
    bash -c "\\
        $CONTAINER_ENV_CMDS && \\
        $SQFS_MOUNT_CMDS && \\
        exec python $SCRIPT_EVAL $CMD_ARGS\\
    "
EOFSCRIPT
    else
        cat >> "$SBATCH_FILE" << EOFSCRIPT
# Container mounts (auto-generated from CONTAINER_MOUNTS array)
CONTAINER_MOUNTS="$CONTAINER_MOUNTS_STR"

# Run external evaluation with container
srun --cpu-bind=none --kill-on-bad-exit $CONTAINER_CMD \\
    --container-mounts="\${CONTAINER_MOUNTS}" \\
    --container-workdir=$RUN_DIR \\
    bash -c "$CONTAINER_ENV_CMDS && exec python $SCRIPT_EVAL $CMD_ARGS"
EOFSCRIPT
    fi
fi

print_header "Submitting Job"
echo "SBATCH script: $SBATCH_FILE"
cd "$RUN_DIR"

# Check for already running job
EXISTING_RUNNING=$(squeue -u "$USER" --name="$RUN_NAME" -h -o "%i %T" 2>/dev/null | grep "RUNNING" | head -1)
if [ -n "$EXISTING_RUNNING" ]; then
    EXISTING_ID=$(echo "$EXISTING_RUNNING" | awk '{print $1}')
    echo "Warning: Job '$RUN_NAME' already RUNNING (ID: $EXISTING_ID)"
    echo "  Cancel: scancel $EXISTING_ID | Monitor: tail -f $RUN_DIR/slurm_run_outputs/slurm*.err"
    read -p "Continue with submission? [y/N] " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

# Submit job
SBATCH_OUTPUT=$(sbatch slurm_batch_script.sh)
JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+')
echo "$SBATCH_OUTPUT"

echo ""
echo "Monitor: squeue -u $USER --name=$RUN_NAME"
if [ "$NJOBS" -gt 1 ]; then
    echo "Logs:    ls $RUN_DIR/slurm_run_outputs/slurm-*.err"
else
    echo "Logs:    tail -f $RUN_DIR/slurm_run_outputs/slurm.err"
fi

# Wait for jobs to complete and concatenate results
if [ "$WAIT_FOR_COMPLETION" == "true" ] && [ "$NJOBS" -gt 1 ]; then
    print_header "Waiting for Jobs to Complete"
    echo "Will concatenate results once all jobs finish..."
    echo ""

    WAIT_COUNT=0
    while true; do
        # Check if any jobs are still running or pending
        INCOMPLETE_JOBS=$(squeue -u "$USER" --name="$RUN_NAME" -h 2>/dev/null | wc -l)

        if [ "$INCOMPLETE_JOBS" -eq 0 ]; then
            echo "All jobs finished running!"
            break
        fi

        # Show status every 10 minutes
        if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
            STATES=$(squeue -u "$USER" --name="$RUN_NAME" -h -o "%T" 2>/dev/null | sort | uniq -c)
            echo "$(date '+%H:%M:%S') - Jobs still active: $INCOMPLETE_JOBS"
            echo "$STATES" | sed 's/^/  /'
        fi

        sleep 60
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done

    echo ""
    print_header "Concatenating Results"

    # Run the concatenation script
    CONCAT_SCRIPT="$CODE_DIR/projects/proteinfoundation/proteinfoundation/inference_utils/aggregate_metrics.py"

    if [ "$CLUSTER_TYPE" == "digs" ]; then
        python "$CONCAT_SCRIPT" --input "$OUTPUT_DIR" --cluster-with-foldseek
    else
        CONTAINER_CMD=$(get_srun_cmd)
        CONTAINER_MOUNTS_STR=$(build_container_mounts_string)
        CONTAINER_ENV_CMDS=$(build_container_env_cmds)
        srun --ntasks=1 --kill-on-bad-exit $CONTAINER_CMD \
            --container-mounts="$CONTAINER_MOUNTS_STR" \
            --container-workdir=$RUN_DIR \
            python "$CONCAT_SCRIPT" --input "$OUTPUT_DIR"
    fi

    echo ""
    echo "Results concatenated successfully!"
    echo "  Per-example metrics: $OUTPUT_DIR/per_example_metrics.csv"
    echo "  Aggregated metrics:  $OUTPUT_DIR/aggregated_metrics.csv"
    echo "  Individual job results moved to: $OUTPUT_DIR/individual_job_results/"
fi

# Follow logs if requested
if [ "$FOLLOW_LOGS" == "true" ]; then
    echo -e "\nFollowing logs (Ctrl+C to exit)...\n"
    while [ ! -f "$RUN_DIR/slurm_run_outputs/slurm"*.err ]; do
        squeue -u "$USER" --name="$RUN_NAME" -h 2>/dev/null | grep -q . || { echo "Job not in queue"; break; }
        sleep 2
    done
    tail -f "$RUN_DIR/slurm_run_outputs/slurm"*.out "$RUN_DIR/slurm_run_outputs/slurm"*.err 2>/dev/null
fi
