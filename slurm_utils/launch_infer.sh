#!/bin/bash
# =============================================================================
# launch_infer.sh - Unified cluster launch for inference and offline validation
# =============================================================================
# Consolidates inference and offline validation into a single script. Handles:
#   - Pure inference/generation (--mode infer, the default)
#   - Offline validation on a val dataset with metrics (--mode val)
#   - Optional result concatenation via a dependent SLURM job (non-blocking)
#   - Sweep-friendly: each config gets its own run directory and concat job
#
# Usage:
#   ./launch_infer.sh <cluster> [options] [extra_hydra_args...]
#
# Positional:
#   <cluster>               Cluster name (e.g. digs, ord, gb200)
#
# Mode:
#   --mode <infer|val>      Run mode:
#                             infer (default) - pure inference/generation
#                             val             - offline validation with metrics
#                                              (automatically sets inference_experiment
#                                               and offline_validation=true)
#
# Parallelism:
#   --njobs N               Number of parallel generation jobs. Each job runs on
#                           its own GPU and processes a slice of the dataset.
#                           Default: reads gen_njobs from the Hydra config, or 1.
#   --gpus-per-job N        GPUs per SLURM array task (default: 1). Each GPU runs
#                           an independent generation job within the task.
#
# Checkpoint (convenience wrappers; can also be passed as Hydra overrides):
#   --ckpt-path PATH        Checkpoint directory (equivalent to ckpt_path=PATH)
#   --ckpt-name NAME        Checkpoint file name (equivalent to ckpt_name=NAME)
#
# Concatenation (aggregates per-job CSV results into a single file):
#   --concatenate           [infer mode] Submit a dependent SLURM job to concatenate
#                           results. Useful when metrics are configured and njobs > 1.
#   --no-concatenate        [val mode]  Skip the automatic concatenation step.
#   --concatenate-foldseek  Run Foldseek structural clustering during concatenation.
#                           Implies --concatenate.
#
# Config:
#   --config-name NAME      Hydra config name (default: inference). Can be a path
#                           relative to configs/, e.g. sweep_configs/my_sweep/inf_000.
#
# Job setup:
#   --no-copy               Skip copying code to a new run directory; use current repo.
#   --overwrite-rundir      Allow overwriting an existing code directory in the run dir.
#   --follow, -f            Follow log output after submission (Ctrl+C to exit).
#
# Hydra overrides (extra_hydra_args):
#   Any remaining arguments are passed as Hydra overrides to generate.py.
#   Examples:
#     run_name_=my_run
#     num_replicates=10
#     generation.args.nsteps=200
#     +datasets/val@generation=mcsa_full   (val mode: specify val dataset)
#     metrics=all_metrics                  (add metrics to inference)
#
# -----------------------------------------------------------------------
# EXAMPLES
# -----------------------------------------------------------------------
#
#   # Basic inference
#   ./launch_infer.sh digs \
#       ckpt_path=/path/to/ckpts ckpt_name=model.ckpt \
#       inputs=benchmarks/monomer.json run_name_=my_run
#
#   # Inference with 8 parallel jobs and result concatenation
#   ./launch_infer.sh digs --njobs 8 --concatenate \
#       ckpt_path=/path/to/ckpts ckpt_name=model.ckpt \
#       inputs=benchmarks/monomer.json run_name_=my_run
#
#   # Offline validation on MCSA subset
#   ./launch_infer.sh digs --mode val \
#       --ckpt-path /path/to/ckpts --ckpt-name model.ckpt \
#       +datasets/val@generation=mcsa_subset \
#       run_name_=mcsa_val
#
#   # Offline validation with 8 parallel jobs and Foldseek clustering
#   ./launch_infer.sh digs --mode val --njobs 8 --concatenate-foldseek \
#       --ckpt-path /path/to/ckpts --ckpt-name model.ckpt \
#       +datasets/val@generation=mcsa_subset \
#       run_name_=mcsa_val_foldseek
#
#   # Load a custom/sweep config
#   ./launch_infer.sh digs --config-name sweep_configs/my_sweep/inf_000 \
#       run_name_=sweep_run_000
#
# -----------------------------------------------------------------------
# NOTES
# -----------------------------------------------------------------------
#
#   - gen_njobs / gpus-per-job: Total logical jobs = njobs. Each SLURM array task
#     handles --gpus-per-job of those jobs. Array size = ceil(njobs / gpus-per-job).
#
#   - Concatenation: submitted as a separate SLURM job with --dependency=afterany
#     so the launch script never blocks. Results are written to:
#       <run_dir>/generation_outputs/val_metrics/
#
#   - Sweep integration: launch_sweep.sh calls this script once per config.
#     Concatenation runs independently per config (not across configs).
#
#   - Val mode adds these Hydra overrides automatically:
#       inference_experiment=inference_on_val_set
#       offline_validation=true
#     You must still pass +datasets/val@generation=<name> and run_name_=<name>.
#
#   - out_dir is always set explicitly to <run_dir>/generation_outputs so that
#     concatenation knows where to look. Pass out_dir=... to override.
#
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ============================================================
# Argument parsing
# ============================================================
CLUSTER=""
MODE="infer"           # infer | val
NJOBS=""               # empty = read from config's gen_njobs
GPUS_PER_JOB=1
CONFIG_NAME="inference"
CKPT_PATH=""
CKPT_NAME=""
DO_CONCATENATE=""      # empty = use mode default; "true"/"false" = explicit
CONCATENATE_FOLDSEEK="false"
COPY_CODE="true"
FOLLOW_LOGS="false"
OVERWRITE_RUNDIR="false"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --njobs)
            NJOBS="$2"
            shift 2
            ;;
        --gpus-per-job=*)
            GPUS_PER_JOB="${1#*=}"
            shift
            ;;
        --gpus-per-job)
            GPUS_PER_JOB="$2"
            shift 2
            ;;
        --config-name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --ckpt-path)
            CKPT_PATH="$2"
            shift 2
            ;;
        --ckpt-name)
            CKPT_NAME="$2"
            shift 2
            ;;
        --concatenate)
            DO_CONCATENATE="true"
            shift
            ;;
        --no-concatenate)
            DO_CONCATENATE="false"
            shift
            ;;
        --concatenate-foldseek)
            DO_CONCATENATE="true"
            CONCATENATE_FOLDSEEK="true"
            shift
            ;;
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

# ============================================================
# Validate arguments
# ============================================================
if [ -z "$CLUSTER" ]; then
    echo "Usage: $0 <cluster> [--mode infer|val] [--njobs N] [--gpus-per-job N] [options] [extra_hydra_args...]"
    echo "Available clusters: digs, ord, gb200"
    echo "See script header for full documentation."
    exit 1
fi

if [ "$MODE" != "infer" ] && [ "$MODE" != "val" ]; then
    echo "Error: --mode must be 'infer' or 'val', got '$MODE'"
    exit 1
fi

# Apply mode-specific default for concatenation
if [ -z "$DO_CONCATENATE" ]; then
    if [ "$MODE" == "val" ]; then
        DO_CONCATENATE="true"    # val mode: concatenate by default (non-blocking)
    else
        DO_CONCATENATE="false"   # infer mode: no concatenation by default
    fi
fi

# ============================================================
# Load cluster configuration
# ============================================================
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

# ============================================================
# Resolve Hydra config
# ============================================================
# When CONFIG_NAME contains path separators, split into --config-path + --config-name
# so Hydra sees only the bare filename and places the config at the root level.
CONFIGS_ROOT="$REPO_ROOT/projects/proteinfoundation/configs"
if [[ "$CONFIG_NAME" == */* ]]; then
    CONFIG_SUBDIR="${CONFIG_NAME%/*}"
    CONFIG_BASE="${CONFIG_NAME##*/}"
    CONFIG_HYDRA_ARGS="--config-path $CONFIGS_ROOT/$CONFIG_SUBDIR --config-name $CONFIG_BASE"
else
    CONFIG_BASE="$CONFIG_NAME"
    CONFIG_HYDRA_ARGS="--config-name $CONFIG_NAME"
fi

# For val mode, add the val experiment to config resolution so that
# val-specific requirements (e.g. run_name_: ???) are enforced up front.
RESOLVE_MODE_ARGS=""
if [ "$MODE" == "val" ]; then
    RESOLVE_MODE_ARGS="inference_experiment=inference_on_val_set"
fi

# Add ckpt_path to resolve args if provided (needed for some interpolations)
RESOLVE_CKPT_ARGS=""
if [ -n "$CKPT_PATH" ]; then
    RESOLVE_CKPT_ARGS="ckpt_path='$CKPT_PATH'"
fi

print_header "Resolving Hydra Config"
echo "  Config name: $CONFIG_NAME"
echo "  Mode:        $MODE"
echo "  Extra args:  $EXTRA_ARGS"

# Pass CONFIG_NAME (not CONFIG_BASE) so resolve_config.py can locate sweep configs
# that live in subdirectories (e.g. sweep_configs/my_sweep/inf_000).
eval $(resolve_hydra_config "$REPO_ROOT" "$CONFIG_NAME" $RESOLVE_MODE_ARGS $RESOLVE_CKPT_ARGS $EXTRA_ARGS)

# Validate that run_name_ was resolved
if [ -z "$RUN_NAME" ]; then
    echo ""
    echo "Error: run_name_ was not found in the resolved config."
    if [ "$MODE" == "val" ]; then
        echo "  For val mode, pass run_name_=<name> as an argument."
        echo "  Example: ./launch_infer.sh $CLUSTER --mode val run_name_=my_val_run ..."
    else
        echo "  Pass run_name_=<name> as an argument, or set run_name_: in your config."
    fi
    exit 1
fi

# ============================================================
# Determine number of parallel jobs
# ============================================================
# Priority: --njobs CLI flag > gen_njobs from resolved config > 1
if [ -z "$NJOBS" ]; then
    # GEN_NJOBS is emitted by resolve_config.py
    NJOBS="${GEN_NJOBS:-1}"
fi

# Calculate SLURM array task count
NUM_ARRAY_TASKS=$(( (NJOBS + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))

# ============================================================
# Set up run directory
# ============================================================
if [ "$MODE" == "val" ]; then
    RUN_DIR="$ROOT_REMOTE/validation/${RUN_NAME}"
else
    RUN_DIR="$ROOT_REMOTE/inference/${RUN_NAME}"
fi

# All generated outputs land here; concatenation also reads from here
GENERATION_OUTPUT_DIR="$RUN_DIR/generation_outputs"

mkdir -p "$RUN_DIR/slurm_run_outputs"

# Clear old SLURM logs so that a re-run starts fresh
rm -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err"

# Create symlink in active directory for easy browsing
mkdir -p "$ROOT_REMOTE/active"
ln -sfn "$RUN_DIR" "$ROOT_REMOTE/active/${RUN_NAME}"

# Create symlink in local repo for easy browsing without SSH
mkdir -p "$REPO_ROOT/runs"
ln -sfn "$RUN_DIR" "$REPO_ROOT/runs/${RUN_NAME}"
echo "  Local symlink: $REPO_ROOT/runs/${RUN_NAME}"

# ============================================================
# Copy code to run directory (or use repo in-place)
# ============================================================
if [ "$COPY_CODE" == "true" ]; then
    if [ -d "$RUN_DIR/src" ] || [ -d "$RUN_DIR/projects/proteinfoundation/proteinfoundation" ]; then
        if [ "$OVERWRITE_RUNDIR" != "true" ]; then
            echo "Error: Code directory already exists at $RUN_DIR"
            echo "  To overwrite, use --overwrite-rundir"
            echo "  To use existing code, use --no-copy"
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

# ============================================================
# Create .env file
# ============================================================
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

# ============================================================
# Assemble Hydra arguments for generate.py
# ============================================================

# Checkpoint convenience args (placed early so EXTRA_ARGS can override)
CKPT_HYDRA_ARGS=""
if [ -n "$CKPT_PATH" ]; then
    CKPT_HYDRA_ARGS="ckpt_path='$CKPT_PATH'"
fi
if [ -n "$CKPT_NAME" ]; then
    CKPT_HYDRA_ARGS="$CKPT_HYDRA_ARGS 'ckpt_name=\"$CKPT_NAME\"'"
fi

# Val-mode overrides (added before EXTRA_ARGS so users can still override them)
VAL_HYDRA_ARGS=""
if [ "$MODE" == "val" ]; then
    VAL_HYDRA_ARGS="inference_experiment=inference_on_val_set offline_validation=true"
fi

# Always set out_dir explicitly so concatenation knows where to look.
# Place before EXTRA_ARGS so users can override with out_dir=... if needed.
OUT_DIR_HYDRA="out_dir=$GENERATION_OUTPUT_DIR"

# ============================================================
# Print configuration summary
# ============================================================
print_header "Job Configuration"
echo "  Mode:             $MODE"
echo "  Run name:         $RUN_NAME"
echo "  Run directory:    $RUN_DIR"
echo "  Output directory: $GENERATION_OUTPUT_DIR"
echo "  Parallel jobs:    $NJOBS"
echo "  GPUs per job:     $GPUS_PER_JOB"
echo "  SLURM array tasks: $NUM_ARRAY_TASKS"
echo "  CPUs per task:    $CPUS_PER_TASK"
echo "  Concatenate:      $DO_CONCATENATE"
[ "$CONCATENATE_FOLDSEEK" == "true" ] && echo "  Foldseek:         true"
echo ""

# ============================================================
# Build main generation SBATCH script
# ============================================================
SBATCH_FILE="$RUN_DIR/slurm_batch_script.sh"
JOB_NAME="$RUN_NAME"
NODES=1
GPUS_PER_NODE="$GPUS_PER_JOB"
NTASKS_PER_NODE=1
build_sbatch_header > "$SBATCH_FILE"

# Array directive if multiple tasks are needed
if [ "$NUM_ARRAY_TASKS" -gt 1 ]; then
    echo "#SBATCH --array=0-$((NUM_ARRAY_TASKS - 1))" >> "$SBATCH_FILE"
fi

# Output/error paths and common environment
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

# Job array configuration
ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}
GPUS_PER_JOB=$GPUS_PER_JOB
NJOBS=$NJOBS

echo "Array task: \$ARRAY_TASK_ID, GPUs per job: \$GPUS_PER_JOB, total jobs: \$NJOBS"
EOF

# Optional mounts
if [ -n "$DATASET_SQUASH_MOUNTS" ]; then
    echo "export DATASET_SQUASH_MOUNTS=\"$DATASET_SQUASH_MOUNTS\"" >> "$SBATCH_FILE"
fi
if [ -n "$USE_CUEQUIVARIANCE" ]; then
    echo "export SHOULD_USE_CUEQUIVARIANCE=$USE_CUEQUIVARIANCE" >> "$SBATCH_FILE"
fi

# DDP coordination (unique port per job, master = first node)
cat >> "$SBATCH_FILE" << 'EOF'

export MASTER_PORT=$((1024 + RANDOM % 64512))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Verify scontrol works (needed for auto-requeue)
scontrol show job $SLURM_JOB_ID > /dev/null 2>&1 && echo '[slurm] scontrol works - auto-requeue enabled' || echo '[slurm] WARNING: scontrol not working - auto-requeue disabled'
EOF

# Platform-specific generation command
SCRIPT_GENERATE="$CODE_DIR/projects/proteinfoundation/proteinfoundation/generate.py"

if [ "$CLUSTER_TYPE" == "digs" ]; then
    cat >> "$SBATCH_FILE" << 'EOF'

# Run generation - each GPU gets its own job_id slice of the dataset
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $NJOBS ]; then
        echo "Running generation with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        srun --exclusive --ntasks=1 --gpus=1 --cpu-bind=none \\
            --kill-on-bad-exit $PYTHON $SCRIPT_GENERATE \\
            $CONFIG_HYDRA_ARGS \\
            $OUT_DIR_HYDRA \\
            $VAL_HYDRA_ARGS \\
            $CKPT_HYDRA_ARGS \\
            $EXTRA_ARGS \\
            hardware.ngpus_per_node_=1 \\
            hardware.nnodes_=1 \\
            job_id=\$JOB_ID \\
            gen_njobs=\$NJOBS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait
EOF
else
    # OCI/GB200: Docker/Pyxis container via srun --container-image
    CONTAINER_CMD=$(get_srun_cmd)
    cat >> "$SBATCH_FILE" << 'EOF'

# Run generation - each GPU gets its own job_id slice of the dataset
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $NJOBS ]; then
        echo "Running generation with job_id=$JOB_ID on GPU $LOCAL_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        srun --exclusive --ntasks=1 --gpus=1 --cpu-bind=none $CONTAINER_CMD \\
            $PYTHON $SCRIPT_GENERATE \\
            $CONFIG_HYDRA_ARGS \\
            $OUT_DIR_HYDRA \\
            $VAL_HYDRA_ARGS \\
            $CKPT_HYDRA_ARGS \\
            $EXTRA_ARGS \\
            hardware.ngpus_per_node_=1 \\
            hardware.nnodes_=1 \\
            job_id=\$JOB_ID \\
            gen_njobs=\$NJOBS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
    fi
done
wait
EOF
fi

# ============================================================
# Build concatenation SBATCH script (if enabled)
# ============================================================
# The concat job runs concatenate_external_eval_results.py after all generation
# jobs finish. It is submitted with --dependency=afterany:<gen_job_id> so the
# launch script never has to wait.
if [ "$DO_CONCATENATE" == "true" ]; then
    CONCAT_SBATCH_FILE="$RUN_DIR/slurm_concat_script.sh"
    CONCAT_SCRIPT="$CODE_DIR/projects/proteinfoundation/proteinfoundation/inference_utils/concatenate_external_eval_results.py"

    # Metrics CSVs are written by generate.py under <out_dir>/val_metrics/
    CONCAT_DIR="$GENERATION_OUTPUT_DIR/val_metrics"

    FOLDSEEK_FLAG=""
    if [ "$CONCATENATE_FOLDSEEK" == "true" ]; then
        FOLDSEEK_FLAG="--cluster-with-foldseek"
    fi

    # Build concat sbatch header (CPU-only, minimal resources)
    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name concat-${RUN_NAME}"
        if [ "$SKIP_ACCOUNT" != "true" ] && [ -n "$ACCOUNT" ]; then
            echo "#SBATCH --account $ACCOUNT"
        fi
        echo "#SBATCH --partition ${PARTITION:-batch}"
        echo "#SBATCH --time 01:00:00"
        echo "#SBATCH --nodes 1"
        echo "#SBATCH --ntasks-per-node 1"
        echo "#SBATCH --mem 16g"
        echo "#SBATCH --output ./slurm_run_outputs/slurm_concat.out"
        echo "#SBATCH --error ./slurm_run_outputs/slurm_concat.err"
        echo "#SBATCH --open-mode append"
        echo ""
        echo "export PYTHONPATH=$CODE_DIR/src:$CODE_DIR/models/rf3/src:$CODE_DIR/projects/proteinfoundation:$CODE_DIR/lib/atomworks/src:\$PYTHONPATH"
        echo "export PYTHONFAULTHANDLER=1"
        echo ""
        echo "echo \"Running result concatenation for: $RUN_NAME\""
        echo "echo \"Concat dir: $CONCAT_DIR\""
    } > "$CONCAT_SBATCH_FILE"

    if [ "$CLUSTER_TYPE" == "digs" ]; then
        cat >> "$CONCAT_SBATCH_FILE" << EOF
$PYTHON "$CONCAT_SCRIPT" --output-dir "$CONCAT_DIR" $FOLDSEEK_FLAG
EOF
    else
        CONTAINER_CMD_CONCAT=$(get_srun_cmd)
        cat >> "$CONCAT_SBATCH_FILE" << EOF
srun --ntasks=1 $CONTAINER_CMD_CONCAT \\
    $PYTHON "$CONCAT_SCRIPT" --output-dir "$CONCAT_DIR" $FOLDSEEK_FLAG
EOF
    fi

    cat >> "$CONCAT_SBATCH_FILE" << 'EOF'
echo "Concatenation complete!"
EOF
fi

# ============================================================
# Submit jobs
# ============================================================
print_header "Submitting Job"
echo "  SBATCH script: $SBATCH_FILE"
cd "$RUN_DIR"

# Warn if a job with this name is already running (avoids silent duplicates)
EXISTING_RUNNING=$(squeue -u "$USER" --name="$RUN_NAME" -h -o "%i %T" 2>/dev/null | grep "RUNNING" | head -1)
if [ -n "$EXISTING_RUNNING" ]; then
    EXISTING_ID=$(echo "$EXISTING_RUNNING" | awk '{print $1}')
    echo "Warning: Job '$RUN_NAME' is already RUNNING (ID: $EXISTING_ID)"
    echo "  Cancel: scancel $EXISTING_ID"
    echo "  Monitor: tail -f $RUN_DIR/slurm_run_outputs/slurm.err"
    read -p "Continue with submission? [y/N] " -n 1 -r; echo
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi

if [ -n "$WANDBKEY" ]; then
    export WANDB_API_KEY="$WANDBKEY"
fi

# Submit main generation job
SBATCH_OUTPUT=$(sbatch slurm_batch_script.sh)
MAIN_JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+')
echo "$SBATCH_OUTPUT"

# Submit concatenation job with dependency (non-blocking)
CONCAT_JOB_ID=""
if [ "$DO_CONCATENATE" == "true" ] && [ -n "$MAIN_JOB_ID" ]; then
    echo ""
    echo "Submitting concatenation job (waits for job $MAIN_JOB_ID to finish)..."
    CONCAT_SBATCH_OUTPUT=$(sbatch --dependency=afterany:$MAIN_JOB_ID slurm_concat_script.sh)
    CONCAT_JOB_ID=$(echo "$CONCAT_SBATCH_OUTPUT" | grep -oP '\d+')
    echo "$CONCAT_SBATCH_OUTPUT"
fi

# ============================================================
# Print monitoring information
# ============================================================
echo ""
echo "=============================================="
echo "Jobs submitted"
echo "=============================================="
echo "  Monitor:     squeue -u $USER --name=$RUN_NAME"
echo "  Gen logs:    tail -f $RUN_DIR/slurm_run_outputs/slurm.err"
if [ -n "$CONCAT_JOB_ID" ]; then
    echo "  Concat job:  squeue -j $CONCAT_JOB_ID"
    echo "  Concat logs: tail -f $RUN_DIR/slurm_run_outputs/slurm_concat.out"
    echo "  Results:     $GENERATION_OUTPUT_DIR/val_metrics/"
fi
echo ""

# Follow logs if requested (tail until Ctrl+C)
if [ "$FOLLOW_LOGS" == "true" ]; then
    echo -e "Following logs (Ctrl+C to exit)...\n"
    while [ ! -f "$RUN_DIR/slurm_run_outputs/slurm.err" ]; do
        squeue -u "$USER" --name="$RUN_NAME" -h 2>/dev/null | grep -q . || { echo "Job not in queue"; break; }
        sleep 2
    done
    tail -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err" 2>/dev/null
fi
