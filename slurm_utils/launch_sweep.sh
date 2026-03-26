#!/bin/bash
# =============================================================================
# launch_sweep.sh - Generate sweep configs and submit SLURM jobs for each
# =============================================================================
# Runs sweep.py to generate a set of Hydra inference configs, then calls
# launch_infer.sh once per config. Each config gets its own run directory
# and concatenation job, so results are aggregated per-config.
#
# Usage:
#   ./launch_sweep.sh <cluster> --sweep-config <path> [options] [hydra_overrides...]
#
# Required:
#   <cluster>                    Cluster name (e.g. digs, ord, gb200)
#   --sweep-config <path>        Path to YAML sweep config file
#
# Options:
#   --gpus-per-job N             GPUs per SLURM job (default: 1)
#   --njobs N                    Parallel jobs per config (default: reads
#                                gen_njobs from each per-config Hydra config)
#   --gres <spec>                SLURM --gres override (e.g. gpu:l40:1); takes
#                                precedence over .user_info setting
#   --time <HH:MM:SS>            SLURM --time override; takes precedence over
#                                .user_info RUNTIME
#   --partition <name>           SLURM --partition override; takes precedence
#                                over .user_info PARTITION
#   --mem <size>                 SLURM --mem override (e.g. 64g); takes
#                                precedence over .user_info MEM
#   --no-concatenate             Skip the per-config concatenation job
#   --no-foldseek                Disable Foldseek clustering in concatenation (on by default)
#   --base-out-dir <path>        Base directory for all sweep runs; each config
#                                gets its own subdirectory at <path>/<config_name>
#   --dry-run                    Generate configs only; do not submit jobs
#   --copy                       Copy code to the sweep directory (default: use repo in-place)
#   --overwrite-rundir           Allow overwriting existing run directories
#   --delay <seconds>            Delay between job submissions (default: 2)
#   --no-sweep-concatenate       Skip the final sweep-level concatenation job
#
# Hydra overrides:
#   Any remaining arguments are forwarded to every launch_infer.sh call.
#   This is the idiomatic way to share settings across all configs in a sweep:
#     ckpt_path=/path/to/ckpts ckpt_name=model.ckpt
#     num_replicates=5
#     inference_experiment=inference_on_val_set
#
# Examples:
#   # Benchmark sweep
#   ./launch_sweep.sh digs --sweep-config configs/sweeps/ppi_benchmark.yaml \
#       ckpt_path=/path/to/checkpoints
#
#   # Validation sweep with 8 parallel jobs (Foldseek on by default)
#   ./launch_sweep.sh ord --sweep-config configs/sweeps/val_ckpt_sweep.yaml \
#       --njobs 8 \
#       inference_experiment=inference_on_val_set
#
#   # Dry run: show configs without submitting
#   ./launch_sweep.sh digs --sweep-config configs/sweeps/ppi_benchmark.yaml \
#       --dry-run
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Parse arguments
CLUSTER=""
SWEEP_CONFIG=""
GPUS_PER_JOB=1
NJOBS=""    # empty = each per-config launch_infer.sh reads gen_njobs from the config
DO_CONCATENATE="true"
CONCATENATE_FOLDSEEK="true"  # foldseek on by default
DRY_RUN="false"
NO_COPY="true"  # use repo in-place by default
OVERWRITE_RUNDIR="false"
SWEEP_CONFIG_OUTPUT_DIR=""
BASE_OUT_DIR=""
DELAY=2
DO_SWEEP_CONCATENATE="true"
EXTRA_ARGS=""
CLI_GRES=""
CLI_TIME=""
CLI_PARTITION=""
CLI_MEM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep-config)
            SWEEP_CONFIG="$2"
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
        --njobs)
            NJOBS="$2"
            shift 2
            ;;
        --no-concatenate)
            DO_CONCATENATE="false"
            shift
            ;;
        --no-foldseek)
            CONCATENATE_FOLDSEEK="false"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --copy)
            NO_COPY="false"
            shift
            ;;
        --no-sweep-concatenate)
            DO_SWEEP_CONCATENATE="false"
            shift
            ;;
        --overwrite-rundir)
            OVERWRITE_RUNDIR="true"
            shift
            ;;
        --base-out-dir)
            BASE_OUT_DIR="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --gres)
            CLI_GRES="$2"
            shift 2
            ;;
        --time)
            CLI_TIME="$2"
            shift 2
            ;;
        --partition)
            CLI_PARTITION="$2"
            shift 2
            ;;
        --mem)
            CLI_MEM="$2"
            shift 2
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

if [ -z "$CLUSTER" ] || [ -z "$SWEEP_CONFIG" ]; then
    echo "Usage: $0 <cluster> --sweep-config <path> [options] [hydra_overrides...]"
    echo ""
    echo "Required:"
    echo "  <cluster>              Cluster name: digs, ord, gb200"
    echo "  --sweep-config <path>  Path to YAML sweep config file"
    echo ""
    echo "Options:"
    echo "  --gpus-per-job N       GPUs per SLURM job (default: 1)"
    echo "  --njobs N              Parallel jobs per config (default: reads gen_njobs from each config)"
    echo "  --no-concatenate       Skip per-config concatenation job"
    echo "  --no-foldseek          Disable Foldseek in concatenation (on by default)"
    echo "  --dry-run              Generate configs only, don't submit"
    echo "  --copy                 Copy code to sweep dir (default: use repo in-place)"
  echo "  --no-sweep-concatenate Skip the final sweep-level concatenation step"
    echo "  --base-out-dir <path>  Base dir for all sweep runs"
    echo "  --delay <seconds>      Delay between submissions (default: 2)"
    exit 1
fi

if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "Error: Sweep config not found: $SWEEP_CONFIG"
    exit 1
fi

# Resolve to absolute paths before cd changes CWD
SWEEP_CONFIG=$(realpath "$SWEEP_CONFIG")
[ -n "$BASE_OUT_DIR" ] && BASE_OUT_DIR=$(realpath "$BASE_OUT_DIR")

# Save generated sweep configs into $BASE_OUT_DIR/configs/
SWEEP_CONFIG_OUTPUT_DIR="$BASE_OUT_DIR/configs"

# Load cluster configuration
# Save the original cluster arg before load_user_info overwrites $CLUSTER.
# .user_info files set CLUSTER to the runtime cluster name (e.g. "coreweave")
# so that exec_core.sh can source .ipd/.env.${CLUSTER} inside the job.
CLUSTER_ARG="$CLUSTER"
load_user_info "$CLUSTER" || exit 1

# Apply CLI overrides (take precedence over .user_info)
[ -n "$CLI_GRES" ]      && GRES="$CLI_GRES"
[ -n "$CLI_TIME" ]      && RUNTIME="$CLI_TIME"
[ -n "$CLI_PARTITION" ] && PARTITION="$CLI_PARTITION"
[ -n "$CLI_MEM" ]       && MEM="$CLI_MEM"

# Find repository root
REPO_ROOT=$(find_repo_root "$SCRIPT_DIR")
if [ -z "$REPO_ROOT" ]; then
    echo "Error: Could not find .project-root"
    exit 1
fi
cd "$REPO_ROOT"

# Setup PYTHONPATH
setup_pythonpath "$REPO_ROOT"
setup_env_vars

# =============================================================================
# Step 0: Optionally copy code once to sweep directory
# =============================================================================
SWEEP_CODE_DIR=""
if [ "$NO_COPY" != "true" ] && [ -n "$BASE_OUT_DIR" ]; then
    SWEEP_CODE_DIR="$BASE_OUT_DIR/code"
    if [ -d "$SWEEP_CODE_DIR/src" ] || [ -d "$SWEEP_CODE_DIR/projects/proteinfoundation/proteinfoundation" ]; then
        if [ "$OVERWRITE_RUNDIR" == "true" ]; then
            echo "Warning: Overwriting existing sweep code directory at $SWEEP_CODE_DIR"
        else
            echo "Reusing existing sweep code directory: $SWEEP_CODE_DIR"
        fi
    else
        print_header "Copying Code for Sweep"
        echo "  $REPO_ROOT -> $SWEEP_CODE_DIR"
        copy_code_to_dir "$REPO_ROOT" "$SWEEP_CODE_DIR" "$CLUSTER"
        echo "  Done."
    fi
fi

# =============================================================================
# Save sweep config to output directory
# =============================================================================
if [ -n "$BASE_OUT_DIR" ]; then
    mkdir -p "$BASE_OUT_DIR"
    cp "$SWEEP_CONFIG" "$BASE_OUT_DIR/sweep_config.yaml"
    echo "  Saved sweep config to: $BASE_OUT_DIR/sweep_config.yaml"
fi

# =============================================================================
# Step 1: Generate sweep configs
# =============================================================================
print_header "Generating Sweep Configs"
echo "  Sweep config: $SWEEP_CONFIG"

SWEEP_SCRIPT="$REPO_ROOT/projects/proteinfoundation/script_utils/sweep.py"

SWEEP_CMD="python $SWEEP_SCRIPT --sweep-config $SWEEP_CONFIG"
if [ -n "$SWEEP_CONFIG_OUTPUT_DIR" ]; then
    SWEEP_CMD="$SWEEP_CMD --output-dir $SWEEP_CONFIG_OUTPUT_DIR"
fi

if [ "$DRY_RUN" == "true" ]; then
    echo ""
    echo "=== DRY RUN: showing what would be generated ==="
    (cd "$REPO_ROOT/projects/proteinfoundation" && python "$SWEEP_SCRIPT" --sweep-config "$SWEEP_CONFIG" --dry-run)
    echo ""
    echo "To generate and submit, remove --dry-run"
    exit 0
fi

# Run the sweep generator from proteinfoundation/ so output_dir resolves correctly
(cd "$REPO_ROOT/projects/proteinfoundation" && eval $SWEEP_CMD)

# Find generated configs
if [ -n "$SWEEP_CONFIG_OUTPUT_DIR" ]; then
    CONFIG_DIR="$SWEEP_CONFIG_OUTPUT_DIR"
else
    CONFIG_DIR=$(cd "$REPO_ROOT/projects/proteinfoundation" && python3 -c "
import yaml, os
with open('$SWEEP_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(os.path.realpath(cfg.get('output_dir', './configs/sweep_configs')))
")
fi

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Generated config directory not found: $CONFIG_DIR"
    exit 1
fi

NUM_CONFIGS=$(find "$CONFIG_DIR" -name "*.yaml" | wc -l)
if [ "$NUM_CONFIGS" -eq 0 ]; then
    echo "Error: No configs generated in $CONFIG_DIR"
    exit 1
fi

echo ""
echo "Generated $NUM_CONFIGS config(s) in: $CONFIG_DIR"

# =============================================================================
# Step 2: Submit each config as a SLURM job via launch_infer.sh
# =============================================================================
print_header "Submitting Jobs"
echo "  GPUs per job: $GPUS_PER_JOB"
echo "  Jobs per config: ${NJOBS:-from config gen_njobs}"
echo "  Concatenate: $DO_CONCATENATE"
[ "$CONCATENATE_FOLDSEEK" == "true" ] && echo "  Foldseek: true"
echo "  Configs to submit: $NUM_CONFIGS"
echo "  Delay between submissions: ${DELAY}s"
echo ""

SUBMITTED=0
FAILED=0
ALL_CONCAT_JOB_IDS=""  # collect per-config concat job IDs for sweep-level dependency

for config_file in $(find "$CONFIG_DIR" -name "*.yaml" | sort); do
    config_name=$(basename "$config_file" .yaml)
    echo "--- Submitting: $config_name ---"

    # Compute the Hydra config name: relative from configs/ root when inside it,
    # or an absolute dir/basename path when outside.
    config_hydra_name=$(python3 -c "
import os
cf = os.path.realpath('$config_file')
hd = os.path.realpath('$REPO_ROOT/projects/proteinfoundation/configs')
rel = os.path.relpath(cf, hd)
if rel.startswith('..'):
    # Config is outside the repo configs dir; use absolute dir + basename
    print(os.path.dirname(cf) + '/' + os.path.splitext(os.path.basename(cf))[0])
else:
    print(os.path.splitext(rel)[0])
")

    # Build per-config out_dir Hydra override if a base directory was specified.
    # This is passed as a Hydra override so launch_infer.sh's resolve_config.py
    # picks it up as OUT_DIR and uses it as the run directory.
    OUT_DIR_OVERRIDE=""
    if [ -n "$BASE_OUT_DIR" ]; then
        OUT_DIR_OVERRIDE="out_dir=$BASE_OUT_DIR/$config_name"
    fi

    # Build common flags for launch_infer.sh
    # --njobs is only passed if explicitly set; otherwise launch_infer.sh reads
    # gen_njobs from the per-config Hydra config (allowing per-config control).
    # Code is always handled at the sweep level
    COMMON_FLAGS="--gpus-per-job $GPUS_PER_JOB"
    if [ -n "$NJOBS" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --njobs $NJOBS"
    fi
    if [ -n "$SWEEP_CODE_DIR" ]; then
        # Code was copied once at the sweep level; point each job at that copy
        COMMON_FLAGS="$COMMON_FLAGS --code-dir $SWEEP_CODE_DIR"
    # else: no --copy passed, launch_infer uses repo in-place by default
    fi
    if [ "$OVERWRITE_RUNDIR" == "true" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --overwrite-rundir"
    fi
    if [ "$DO_CONCATENATE" == "false" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --no-concatenate"
    fi
    if [ "$CONCATENATE_FOLDSEEK" == "false" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --no-foldseek"
    fi
    if [ -n "$CLI_GRES" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --gres $CLI_GRES"
    fi
    if [ -n "$CLI_TIME" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --time $CLI_TIME"
    fi
    if [ -n "$CLI_PARTITION" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --partition $CLI_PARTITION"
    fi
    if [ -n "$CLI_MEM" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --mem $CLI_MEM"
    fi

    # Run launch_infer.sh and capture output to extract the concat job ID
    infer_output=$("$SCRIPT_DIR/launch_infer.sh" "$CLUSTER_ARG" \
        $COMMON_FLAGS \
        --config-name "$config_hydra_name" \
        run_name_="$config_name" \
        $OUT_DIR_OVERRIDE \
        $EXTRA_ARGS 2>&1) && SUBMITTED=$((SUBMITTED + 1)) || FAILED=$((FAILED + 1))

    echo "$infer_output"

    # Extract per-config concat job ID emitted by launch_infer.sh
    config_concat_id=$(echo "$infer_output" | grep -oP 'LAUNCH_INFER_CONCAT_JOB_ID=\K\d+' || true)
    if [ -n "$config_concat_id" ]; then
        ALL_CONCAT_JOB_IDS="${ALL_CONCAT_JOB_IDS:+$ALL_CONCAT_JOB_IDS:}$config_concat_id"
    fi

    echo ""
    # Small delay to avoid overwhelming the SLURM scheduler
    if [ "$DELAY" -gt 0 ]; then
        sleep "$DELAY"
    fi
done

# =============================================================================
# Step 3: Submit sweep-level concatenation job
# =============================================================================
# Concatenates per_example_metrics, per_task_metrics, and aggregated_metrics
# across all sweep subdirectories once all per-config concat jobs finish.
SWEEP_CONCAT_JOB_ID=""
if [ "$DO_SWEEP_CONCATENATE" == "true" ] && [ -n "$BASE_OUT_DIR" ] && [ -n "$ALL_CONCAT_JOB_IDS" ]; then
    print_header "Submitting Sweep Concatenation Job"
    echo "  Depends on concat jobs: $ALL_CONCAT_JOB_IDS"

    SWEEP_CONCAT_SCRIPT="$BASE_OUT_DIR/slurm_sweep_concat_script.sh"
    SWEEP_CONCAT_PY="$REPO_ROOT/projects/proteinfoundation/script_utils/concatenate_sweep_results.py"
    SWEEP_CODE_PY="${SWEEP_CODE_DIR:+$SWEEP_CODE_DIR/projects/proteinfoundation/script_utils/concatenate_sweep_results.py}"
    # Use copied script if available, else fall back to repo
    [ -n "$SWEEP_CODE_DIR" ] && [ -f "$SWEEP_CODE_PY" ] && SWEEP_CONCAT_PY="$SWEEP_CODE_PY"

    SWEEP_PYTHONPATH="${SWEEP_CODE_DIR:-$REPO_ROOT}/src:${SWEEP_CODE_DIR:-$REPO_ROOT}/models/rf3/src:${SWEEP_CODE_DIR:-$REPO_ROOT}/projects/proteinfoundation:${SWEEP_CODE_DIR:-$REPO_ROOT}/lib/atomworks/src"

    {
        echo "#!/bin/bash"
        echo "#SBATCH --job-name sweep-concat-$(basename "$BASE_OUT_DIR")"
        if [ "$SKIP_ACCOUNT" != "true" ] && [ -n "$ACCOUNT" ]; then
            echo "#SBATCH --account $ACCOUNT"
        fi
        echo "#SBATCH --partition ${CPU_PARTITION:-${PARTITION:-batch}}"
        echo "#SBATCH --time 01:00:00"
        echo "#SBATCH --nodes 1"
        echo "#SBATCH --ntasks-per-node 1"
        echo "#SBATCH --mem 32g"
        echo "#SBATCH --output $BASE_OUT_DIR/slurm_sweep_concat.out"
        echo "#SBATCH --error $BASE_OUT_DIR/slurm_sweep_concat.err"
        echo "#SBATCH --open-mode append"
        echo ""
        echo "export PYTHONPATH=$SWEEP_PYTHONPATH:\$PYTHONPATH"
        echo "export PYTHONFAULTHANDLER=1"
        echo ""
        echo "echo \"Running sweep-level concatenation for: $BASE_OUT_DIR\""
        echo "python \"$SWEEP_CONCAT_PY\" --sweep-dir \"$BASE_OUT_DIR\""
        echo "echo \"Sweep concatenation complete!\""
    } > "$SWEEP_CONCAT_SCRIPT"

    SWEEP_CONCAT_OUTPUT=$(sbatch --dependency=afterany:$ALL_CONCAT_JOB_IDS "$SWEEP_CONCAT_SCRIPT")
    SWEEP_CONCAT_JOB_ID=$(echo "$SWEEP_CONCAT_OUTPUT" | grep -oP '\d+')
    echo "$SWEEP_CONCAT_OUTPUT"
    echo "  Sweep concat results: $BASE_OUT_DIR/sweep_*.csv"
elif [ "$DO_SWEEP_CONCATENATE" == "true" ] && [ -z "$ALL_CONCAT_JOB_IDS" ]; then
    echo ""
    echo "Note: No concat job IDs collected - skipping sweep concatenation."
    echo "  (This happens when --no-concatenate is used or no jobs were submitted.)"
fi

# =============================================================================
# Summary
# =============================================================================
print_header "Sweep Summary"
echo "  Total configs: $NUM_CONFIGS"
echo "  Submitted:     $SUBMITTED"
echo "  Failed:        $FAILED"
echo ""
echo "  Config directory: $CONFIG_DIR"
[ -n "$SWEEP_CODE_DIR" ] && echo "  Shared code dir:  $SWEEP_CODE_DIR"
[ -n "$SWEEP_CONCAT_JOB_ID" ] && echo "  Sweep concat job: $SWEEP_CONCAT_JOB_ID"
[ -n "$BASE_OUT_DIR" ] && echo "  Sweep results:    $BASE_OUT_DIR/sweep_*.csv"
echo "  Monitor all:      squeue -u $USER"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED job(s) failed to submit. Check output above for details."
    exit 1
fi
