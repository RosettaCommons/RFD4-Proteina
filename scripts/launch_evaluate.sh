#!/bin/bash
# =============================================================================
# launch_evaluate.sh - Cluster launch for standalone structure evaluation
# =============================================================================
# Runs evaluate.py on a directory of designed structures and computes metrics
# via a MetricManager config. Supports job arrays for parallel evaluation.
#
# Usage:
#   ./launch_evaluate.sh <cluster> [options] [hydra_overrides...]
#
# Positional:
#   <cluster>               Cluster name (e.g. digs, ord, gb200)
#
# Options:
#   --njobs N               Parallel evaluation jobs (default: reads gen_njobs
#                           from Hydra config, fallback 1)
#   --gpus-per-job N        GPUs per SLURM array task (default: 1)
#   --config-name NAME      Hydra config name (default: evaluate). Same meaning
#                           as evaluate.py's --config-name.
#   --config-path DIR       Hydra search-path root (same as evaluate.py's
#                           --config-path). Defaults to the repo configs/ tree;
#                           set it to load a config that lives elsewhere on the
#                           filesystem. Such a config must inherit a base config
#                           and add a hydra.searchpath entry — see
#                           configs/inference_experiment/TEMPLATE.yaml Pattern C.
#   --gres <spec>           SLURM --gres override (e.g. gpu:l40:1)
#   --time <HH:MM:SS>       SLURM --time override (default: 6:00:00)
#   --partition <name>      SLURM --partition override
#   --mem <size>            SLURM --mem override (e.g. 64G)
#   --cpus-per-task N       SLURM --cpus-per-task override; takes precedence
#                           over .user_info CPUS_PER_TASK
#   --no-concatenate        Skip automatic per-job CSV concatenation
#   --no-foldseek           Disable Foldseek clustering in concatenation (on by default)
#   --copy                  Copy repo to run directory before submitting
#   --code-dir <path>       Use a pre-existing code directory
#   --overwrite-rundir      Allow overwriting an existing code directory
#   --follow, -f            Follow logs after submission
#
# Hydra overrides:
#   Any remaining arguments are passed directly to evaluate.py. Examples:
#     inputs=/path/to/structures
#     out_dir=/path/to/output
#     metrics_tags=unconditional_refolding
#     run_name_=my_eval
#
# Examples:
#   # Basic evaluation
#   ./launch_evaluate.sh digs \
#       inputs=/path/to/structures \
#       out_dir=/path/to/output \
#       run_name_=my_eval
#
#   # Parallel evaluation with 8 jobs
#   ./launch_evaluate.sh digs --njobs 8 \
#       inputs=/path/to/structures \
#       out_dir=/path/to/output \
#       run_name_=my_eval \
#       gen_njobs=8
#
#   # Config that lives outside the repo configs/ tree
#   ./launch_evaluate.sh coreweave \
#       --config-path /path/to/my/configs --config-name my_eval
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
strip_inherited_slurm_env

# ============================================================
# Argument parsing
# ============================================================
CLUSTER=""
NJOBS=""
GPUS_PER_JOB=1
CONFIG_NAME="evaluate"
CONFIG_PATH=""         # optional dir for configs outside the repo configs/ tree
DO_CONCATENATE="true"
CONCATENATE_FOLDSEEK="true"  # foldseek on by default
COPY_CODE="false"
CODE_DIR_OVERRIDE=""
FOLLOW_LOGS="false"
OVERWRITE_RUNDIR="false"
EXTRA_ARGS=""
CLI_GRES=""
CLI_TIME="6:00:00"
CLI_PARTITION=""
CLI_MEM=""
CLI_CPUS_PER_TASK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --njobs=*)          NJOBS="${1#*=}"; shift ;;
        --njobs)            NJOBS="$2"; shift 2 ;;
        --gpus-per-job=*)   GPUS_PER_JOB="${1#*=}"; shift ;;
        --gpus-per-job)     GPUS_PER_JOB="$2"; shift 2 ;;
        --config-name)      CONFIG_NAME="$2"; shift 2 ;;
        --config-path)      CONFIG_PATH="$2"; shift 2 ;;
        --config-path=*)    CONFIG_PATH="${1#*=}"; shift ;;
        --no-concatenate)   DO_CONCATENATE="false"; shift ;;
        --no-foldseek)      CONCATENATE_FOLDSEEK="false"; shift ;;
        --copy)             COPY_CODE="true"; shift ;;
        --code-dir)         CODE_DIR_OVERRIDE="$2"; COPY_CODE="false"; shift 2 ;;
        --overwrite-rundir) OVERWRITE_RUNDIR="true"; shift ;;
        --gres)             CLI_GRES="$2"; shift 2 ;;
        --time)             CLI_TIME="$2"; shift 2 ;;
        --partition)        CLI_PARTITION="$2"; shift 2 ;;
        --mem)              CLI_MEM="$2"; shift 2 ;;
        --cpus-per-task)    CLI_CPUS_PER_TASK="$2"; shift 2 ;;
        --follow|-f)        FOLLOW_LOGS="true"; shift ;;
        *)
            if [ -z "$CLUSTER" ]; then
                CLUSTER="$1"
            else
                EXTRA_ARGS="$EXTRA_ARGS $1"
            fi
            shift ;;
    esac
done

# ============================================================
# Validate
# ============================================================
if [ -z "$CLUSTER" ]; then
    echo "Usage: $0 <cluster> [--njobs N] [options] [hydra_overrides...]"
    echo "Available clusters: digs, ord, gb200"
    echo "See script header for full documentation."
    exit 1
fi

# ============================================================
# Load cluster configuration
# ============================================================
load_user_info "$CLUSTER" || exit 1

[ -n "$CLI_GRES" ]           && GRES="$CLI_GRES"
[ -n "$CLI_TIME" ]           && RUNTIME="$CLI_TIME"
[ -n "$CLI_PARTITION" ]      && PARTITION="$CLI_PARTITION"
[ -n "$CLI_MEM" ]            && MEM="$CLI_MEM"
[ -n "$CLI_CPUS_PER_TASK" ]  && CPUS_PER_TASK="$CLI_CPUS_PER_TASK"

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

# ============================================================
# Resolve Hydra config
# ============================================================
# Mirror Hydra exactly: --config-name is the config name (default: evaluate),
# and --config-path optionally overrides the search-path root for configs that
# live outside the repo configs/ tree. Both are passed straight through, just
# as evaluate.py / Hydra expect them — no path rewriting.
RESOLVE_PATH_ARG=""
if [ -n "$CONFIG_PATH" ]; then
    RESOLVE_PATH_ARG="--config-path $CONFIG_PATH"
fi
CONFIG_HYDRA_ARGS="$RESOLVE_PATH_ARG --config-name $CONFIG_NAME"

print_header "Resolving Hydra Config"
echo "  Config name: $CONFIG_NAME"
[ -n "$CONFIG_PATH" ] && echo "  Config path: $CONFIG_PATH"
echo "  Extra args:  $EXTRA_ARGS"

# resolve_config.py's CPUS_PER_TASK is training-oriented and falls back to 8
# for evaluate configs; save/restore ours so the eval below doesn't clobber it.
_CLUSTER_CPUS_PER_TASK="${CPUS_PER_TASK:-}"
eval $(resolve_hydra_config "$REPO_ROOT" "$CONFIG_NAME" $RESOLVE_PATH_ARG $EXTRA_ARGS)
if [ -n "$_CLUSTER_CPUS_PER_TASK" ]; then
    CPUS_PER_TASK="$_CLUSTER_CPUS_PER_TASK"
fi
unset _CLUSTER_CPUS_PER_TASK

if [ -z "$RUN_NAME" ]; then
    echo ""
    echo "Error: run_name_ was not found in the resolved config."
    echo "  Pass run_name_=<name> as a Hydra override."
    exit 1
fi

# ============================================================
# Determine number of parallel jobs
# ============================================================
if [ -z "$NJOBS" ]; then
    NJOBS="${GEN_NJOBS:-1}"
fi
NUM_ARRAY_TASKS=$(( (NJOBS + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))

# ============================================================
# Set up run directory
# ============================================================
if [ -n "$OUT_DIR" ]; then
    RUN_DIR="$OUT_DIR"
else
    RUN_DIR="$ROOT_REMOTE/evaluate/${RUN_NAME}"
fi
RUN_DIR="${RUN_DIR%/}"

mkdir -p "$RUN_DIR/slurm_run_outputs"
rm -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err"

mkdir -p "$ROOT_REMOTE/active"
ln -sfn "$RUN_DIR" "$ROOT_REMOTE/active/${RUN_NAME}"

mkdir -p "$REPO_ROOT/runs"
ln -sfn "$RUN_DIR" "$REPO_ROOT/runs/${RUN_NAME}"
echo "  Local symlink: $REPO_ROOT/runs/${RUN_NAME}"

# ============================================================
# Code directory
# ============================================================
if [ -n "$CODE_DIR_OVERRIDE" ]; then
    print_header "Using Pre-existing Code Directory"
    echo "  $CODE_DIR_OVERRIDE"
    CODE_DIR="$CODE_DIR_OVERRIDE"
elif [ "$COPY_CODE" == "true" ]; then
    CODE_DIR="$RUN_DIR/code"
    if [ -d "$CODE_DIR/src" ] || [ -d "$CODE_DIR/rfproteina" ]; then
        if [ "$OVERWRITE_RUNDIR" != "true" ]; then
            echo "Error: Code directory already exists at $CODE_DIR. Use --overwrite-rundir."
            exit 1
        fi
        echo "Warning: Overwriting existing code directory at $CODE_DIR"
    fi
    print_header "Copying Code"
    echo "  $REPO_ROOT -> $CODE_DIR"
    copy_code_to_dir "$REPO_ROOT" "$CODE_DIR" "$CLUSTER"
else
    print_header "Using Code In-Place"
    echo "  $REPO_ROOT"
    CODE_DIR="$REPO_ROOT"
fi

write_env_file "$RUN_DIR/.env"

# ============================================================
# Print configuration summary
# ============================================================
# Rule of thumb: never fewer than 2 CPUs per GPU. The whole job (1 task,
# CPUS_PER_TASK cpus) is split across GPUS_PER_JOB srun steps below, so it must
# request >= 2*GPUS_PER_JOB up front or the per-GPU steps starve.
if [ "${CPUS_PER_TASK:-0}" -lt "$(( 2 * GPUS_PER_JOB ))" ]; then
    CPUS_PER_TASK=$(( 2 * GPUS_PER_JOB ))
fi

print_header "Job Configuration"
echo "  Run name:      $RUN_NAME"
echo "  Run directory: $RUN_DIR"
echo "  Parallel jobs: $NJOBS"
echo "  GPUs per job:  $GPUS_PER_JOB"
echo "  Concatenate:   $DO_CONCATENATE (foldseek: $CONCATENATE_FOLDSEEK)"
echo ""

# ============================================================
# Build SBATCH script
# ============================================================
SBATCH_FILE="$RUN_DIR/slurm_batch_script.sh"
JOB_NAME="$RUN_NAME"
NODES=1
GPUS_PER_NODE="$GPUS_PER_JOB"
NTASKS_PER_NODE=1
build_sbatch_header > "$SBATCH_FILE"

if [ "$NUM_ARRAY_TASKS" -gt 1 ]; then
    echo "#SBATCH --array=0-$((NUM_ARRAY_TASKS - 1))" >> "$SBATCH_FILE"
fi

cat >> "$SBATCH_FILE" << EOF
#SBATCH --output ./slurm_run_outputs/slurm.out
#SBATCH --error ./slurm_run_outputs/slurm.err
EOF

emit_ulimit_block >> "$SBATCH_FILE"

cat >> "$SBATCH_FILE" << EOF

export PYTHONPATH=$CODE_DIR:$CODE_DIR/lib/atomworks/src:\$PYTHONPATH
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

ARRAY_TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}
GPUS_PER_JOB=$GPUS_PER_JOB
NJOBS=$NJOBS

# --exclusive below gives each step its own GPU, but caps it at one CPU unless asked.
CPUS_PER_STEP=\$(( \${SLURM_CPUS_ON_NODE:-8} / GPUS_PER_JOB ))
CPUS_PER_STEP=\$(( CPUS_PER_STEP >= 2 ? CPUS_PER_STEP : 2 ))  # rule of thumb: never < 2 CPUs/GPU

echo "Array task: \$ARRAY_TASK_ID, GPUs per job: \$GPUS_PER_JOB, total jobs: \$NJOBS, CPUs per step: \$CPUS_PER_STEP"
EOF

if [ "$CONTAINER_TYPE" == "apptainer" ]; then
    cat >> "$SBATCH_FILE" << EOF
export CLUSTER=$CLUSTER
export CONTAINER_TYPE=$CONTAINER_TYPE
EOF
fi

[ -n "$DATASET_SQUASH_MOUNTS" ] && echo "export DATASET_SQUASH_MOUNTS=\"$DATASET_SQUASH_MOUNTS\"" >> "$SBATCH_FILE"

# Pre-spawn AF3 daemon block when the resolved config requests AF3 
if [ "$CONTAINER_TYPE" == "apptainer" ] && [ "$AF3_NEEDED" == "true" ]; then
    echo "  AF3Oracle detected in resolved config — emitting AF3 daemon pre-spawn block"
    emit_af3_daemon_block "$CODE_DIR" "$GPUS_PER_NODE" >> "$SBATCH_FILE"
fi
if should_spawn_af2_daemons; then
    echo "  AF2Oracle detected in resolved config — emitting AF2 daemon pre-spawn block"
    emit_af2_daemon_block "$CODE_DIR" "$GPUS_PER_NODE" >> "$SBATCH_FILE"
fi

SCRIPT_EVALUATE="$CODE_DIR/src/rfproteina/evaluate.py"

if [ "$CONTAINER_TYPE" == "apptainer" ]; then
    cat >> "$SBATCH_FILE" << 'EOF'

GEN_PIDS=()
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $NJOBS ]; then
        echo "Running evaluate.py with job_id=$JOB_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        AF3_DAEMON_RANK=\$LOCAL_ID AF2_DAEMON_RANK=\$LOCAL_ID srun --exclusive --ntasks=1 --gpus=1 \\
            --cpus-per-task=\$CPUS_PER_STEP --cpu-bind=none \\
            --kill-on-bad-exit $SCRIPT_EVALUATE \\
            $CONFIG_HYDRA_ARGS \\
            $EXTRA_ARGS \\
            out_dir=$RUN_DIR \\
            job_id=\$JOB_ID \\
            gen_njobs=\$NJOBS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
        GEN_PIDS+=($!)
    fi
done
wait "${GEN_PIDS[@]}"
EOF
else
    CONTAINER_CMD=$(get_srun_cmd)
    cat >> "$SBATCH_FILE" << 'EOF'

GEN_PIDS=()
for LOCAL_ID in $(seq 0 $((GPUS_PER_JOB - 1))); do
    JOB_ID=$((ARRAY_TASK_ID * GPUS_PER_JOB + LOCAL_ID))
    if [ $JOB_ID -lt $NJOBS ]; then
        echo "Running evaluate.py with job_id=$JOB_ID"
EOF
    cat >> "$SBATCH_FILE" << EOF
        AF2_DAEMON_RANK=\$LOCAL_ID srun --exclusive --ntasks=1 --kill-on-bad-exit $CONTAINER_CMD \\
            --container-mounts="/lustre:/lustre,/tmp:/tmp" \\
            --container-workdir=$RUN_DIR \\
            $SCRIPT_EVALUATE \\
            $CONFIG_HYDRA_ARGS \\
            $EXTRA_ARGS \\
            out_dir=$RUN_DIR \\
            job_id=\$JOB_ID \\
            gen_njobs=\$NJOBS &
EOF
    cat >> "$SBATCH_FILE" << 'EOF'
        GEN_PIDS+=($!)
    fi
done
wait "${GEN_PIDS[@]}"
EOF
fi

# ============================================================
# Submit
# ============================================================
print_header "Submitting Job"
echo "SBATCH script: $SBATCH_FILE"
cd "$RUN_DIR"

EXISTING_RUNNING=$(squeue -u "$USER" --name="$RUN_NAME" -h -o "%i %T" 2>/dev/null | grep "RUNNING" | head -1)
if [ -n "$EXISTING_RUNNING" ]; then
    EXISTING_ID=$(echo "$EXISTING_RUNNING" | awk '{print $1}')
    echo "Warning: Job '$RUN_NAME' is already RUNNING (ID: $EXISTING_ID)"
    echo "  Cancel: scancel $EXISTING_ID"
    echo "  Monitor: tail -f $RUN_DIR/slurm_run_outputs/slurm.err"
    if [ -t 1 ]; then
        read -p "Continue with submission? [y/N] " -n 1 -r; echo
        [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
    else
        echo "  (Non-interactive mode: proceeding with submission)"
    fi
fi

# ============================================================
# Build concatenation SBATCH script (if enabled)
# ============================================================
if [ "$DO_CONCATENATE" == "true" ]; then
    CONCAT_SBATCH_FILE="$RUN_DIR/slurm_concat_script.sh"
    CONCAT_SCRIPT="$CODE_DIR/src/rfproteina/inference_utils/aggregate_metrics.py"
    CONCAT_DIR="$RUN_DIR/evaluation_metrics"

    FOLDSEEK_FLAG=""
    [ "$CONCATENATE_FOLDSEEK" == "true" ] && FOLDSEEK_FLAG="--cluster-with-foldseek"

    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name concat-${RUN_NAME}"
        if [ "$SKIP_ACCOUNT" != "true" ] && [ -n "$ACCOUNT" ]; then
            echo "#SBATCH --account $ACCOUNT"
        fi
        echo "#SBATCH --partition ${CPU_PARTITION:-${PARTITION:-batch}}"
        echo "#SBATCH --time 04:00:00"
        echo "#SBATCH --nodes 1"
        echo "#SBATCH --ntasks-per-node 1"
        echo "#SBATCH --cpus-per-task 4"
        echo "#SBATCH --mem 16g"
        echo "#SBATCH --output ./slurm_run_outputs/slurm_concat.out"
        echo "#SBATCH --error ./slurm_run_outputs/slurm_concat.err"
        echo "#SBATCH --open-mode append"
        echo ""
        echo "export PYTHONPATH=$CODE_DIR:$CODE_DIR/lib/atomworks/src:\$PYTHONPATH"
        echo "export PYTHONFAULTHANDLER=1"
        echo ""
        echo "echo \"Running result concatenation for: $RUN_NAME\""
        echo "echo \"Concat dir: $CONCAT_DIR\""
    } > "$CONCAT_SBATCH_FILE"

    if [ "$CONTAINER_TYPE" == "apptainer" ]; then
        cat >> "$CONCAT_SBATCH_FILE" << EOF
export CLUSTER=$CLUSTER
export CONTAINER_TYPE=$CONTAINER_TYPE

"$CONCAT_SCRIPT" --input "$CONCAT_DIR" $FOLDSEEK_FLAG
EOF
    else
        CONTAINER_CMD_CONCAT=$(get_srun_cmd)
        CONTAINER_MOUNTS_STR_CONCAT=$(build_container_mounts_string)
        CONTAINER_ENV_CMDS_CONCAT=$(build_container_env_cmds)
        cat >> "$CONCAT_SBATCH_FILE" << EOF
srun --ntasks=1 $CONTAINER_CMD_CONCAT \\
    --container-mounts="$CONTAINER_MOUNTS_STR_CONCAT" \\
    --container-workdir=$RUN_DIR \\
    bash -c "$CONTAINER_ENV_CMDS_CONCAT && python \"$CONCAT_SCRIPT\" --input \"$CONCAT_DIR\" $FOLDSEEK_FLAG"
EOF
    fi

    cat >> "$CONCAT_SBATCH_FILE" << 'EOF'
echo "Concatenation complete!"
EOF
fi

# ============================================================
# Submit jobs
# ============================================================
SBATCH_OUTPUT=$(sbatch slurm_batch_script.sh)
SUBMITTED_JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+')
echo "$SBATCH_OUTPUT"

# Submit concatenation job with dependency (non-blocking)
CONCAT_JOB_ID=""
if [ "$DO_CONCATENATE" == "true" ] && [ -n "$SUBMITTED_JOB_ID" ]; then
    echo ""
    echo "Submitting concatenation job (waits for job $SUBMITTED_JOB_ID to finish)..."
    CONCAT_SBATCH_OUTPUT=$(sbatch --dependency=afterany:$SUBMITTED_JOB_ID slurm_concat_script.sh)
    CONCAT_JOB_ID=$(echo "$CONCAT_SBATCH_OUTPUT" | grep -oP '\d+')
    echo "$CONCAT_SBATCH_OUTPUT"
    echo "LAUNCH_EVALUATE_CONCAT_JOB_ID=$CONCAT_JOB_ID"
fi

echo ""
echo "=============================================="
echo "Jobs submitted"
echo "=============================================="
echo "  Monitor:     squeue -u $USER --name=$RUN_NAME"
echo "  Eval logs:   tail -f $RUN_DIR/slurm_run_outputs/slurm.err"
if [ -n "$CONCAT_JOB_ID" ]; then
    echo "  Concat job:  squeue -j $CONCAT_JOB_ID"
    echo "  Concat logs: tail -f $RUN_DIR/slurm_run_outputs/slurm_concat.out"
    echo "  Results:     $RUN_DIR/evaluation_metrics/"
fi
echo ""

if [ "$FOLLOW_LOGS" == "true" ]; then
    echo -e "Following logs (Ctrl+C to exit)...\n"
    while [ ! -f "$RUN_DIR/slurm_run_outputs/slurm.err" ]; do
        squeue -u "$USER" --name="$RUN_NAME" -h 2>/dev/null | grep -q . || { echo "Job not in queue"; break; }
        sleep 2
    done
    tail -f "$RUN_DIR/slurm_run_outputs/slurm.out" "$RUN_DIR/slurm_run_outputs/slurm.err" 2>/dev/null
fi
