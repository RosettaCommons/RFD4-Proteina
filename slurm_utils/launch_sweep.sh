#!/bin/bash
# =============================================================================
# launch_sweep.sh - Generate sweep configs and submit SLURM jobs for each
# =============================================================================
# End-to-end script that:
#   1. Runs sweep.py to generate a set of inference configs
#   2. For each generated config, submits a job via launch_infer.sh
#
# Each sweep config gets its own run directory and (optionally) its own
# concatenation job, so results are aggregated per-config, not across configs.
#
# Usage:
#   ./launch_sweep.sh <cluster> --sweep-config <path> [options] [extra_args...]
#
# Options:
#   --sweep-config <path>        Path to YAML sweep config file (required)
#   --mode <infer|val>           Submission mode: infer (default) or val
#                                  (val sets inference_experiment=inference_on_val_set
#                                   and offline_validation=true automatically)
#   --gpus-per-job N             GPUs per SLURM job (default: 1)
#   --njobs N                    Parallel jobs per config (default: 1)
#   --ckpt-path <path>           Checkpoint directory. Applied to every config.
#                                  Can also be embedded in the sweep config itself.
#   --concatenate                Submit a dependent concatenation job per config.
#                                  Default: on for val mode, off for infer mode.
#   --no-concatenate             Disable concatenation (overrides the mode default).
#   --concatenate-foldseek       Enable Foldseek structural clustering in concatenation.
#                                  Implies --concatenate.
#   --base-out-dir <path>        Base directory for model outputs; each config writes
#                                  to <path>/<config_name> (prevents overwriting).
#   --sweep-config-output-dir <p>  Override output directory for generated configs.
#   --dry-run                    Generate configs only; do not submit jobs.
#   --no-copy                    Don't copy code to the run directory.
#   --overwrite-rundir           Allow overwriting existing run directories.
#   --delay <seconds>            Delay between job submissions (default: 2).
#
# Extra hydra overrides (extra_args):
#   Any remaining arguments are passed as Hydra overrides to every config.
#   Examples:
#     ckpt_path=/path/to/ckpts   (alternative to --ckpt-path for Hydra-style passing)
#     num_replicates=5
#
# Examples:
#   # PPI benchmark sweep - generates N configs, submits each as a SLURM job
#   ./launch_sweep.sh digs --sweep-config configs/sweeps/ppi_benchmark.yaml \
#       ckpt_path=/path/to/checkpoints
#
#   # Dry run: just generate configs, don't submit
#   ./launch_sweep.sh digs --sweep-config configs/sweeps/ppi_benchmark.yaml --dry-run
#
#   # Val sweep: 8 parallel jobs per config, Foldseek clustering
#   ./launch_sweep.sh ord --sweep-config configs/sweeps/val_ckpt_sweep.yaml \
#       --mode val --ckpt-path /path/to/checkpoints \
#       --njobs 8 --gpus-per-job 4 --concatenate-foldseek
#
#   # Inference sweep with per-config result concatenation
#   ./launch_sweep.sh digs --sweep-config configs/sweeps/ppi_benchmark.yaml \
#       --concatenate ckpt_path=/path/to/checkpoints
#
# Notes:
#   - launch_infer.sh is called once per generated config.
#   - For val mode, the sweep config's base_config_overrides should include
#     +datasets/val@generation=<name> to specify the validation dataset.
#   - Concatenation runs as a separate SLURM job per config (non-blocking).
#   - Results for each config land in separate directories; there is no
#     cross-config aggregation.
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Parse arguments
CLUSTER=""
SWEEP_CONFIG=""
MODE="infer"
GPUS_PER_JOB=1
NJOBS=""    # empty = each per-config launch_infer.sh reads gen_njobs from the config
CKPT_PATH=""
DO_CONCATENATE=""      # empty = use mode default in launch_infer.sh
CONCATENATE_FOLDSEEK="false"
DRY_RUN="false"
NO_COPY="false"
OVERWRITE_RUNDIR="false"
SWEEP_CONFIG_OUTPUT_DIR=""
BASE_OUT_DIR=""
DELAY=2
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --sweep-config)
            SWEEP_CONFIG="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
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
        --ckpt-path)
            CKPT_PATH="$2"
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
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --no-copy)
            NO_COPY="true"
            shift
            ;;
        --overwrite-rundir)
            OVERWRITE_RUNDIR="true"
            shift
            ;;
        --sweep-config-output-dir)
            SWEEP_CONFIG_OUTPUT_DIR="$2"
            shift 2
            ;;
        --base-out-dir)
            BASE_OUT_DIR="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
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

if [ -z "$CLUSTER" ] || [ -z "$SWEEP_CONFIG" ]; then
    echo "Usage: $0 <cluster> --sweep-config <path> [options] [extra_args...]"
    echo ""
    echo "Required:"
    echo "  <cluster>              Cluster name: digs, ord, gb200"
    echo "  --sweep-config <path>  Path to YAML sweep config file"
    echo ""
    echo "Options:"
    echo "  --mode <infer|val>     Submission mode (default: infer)"
    echo "  --gpus-per-job N       GPUs per SLURM job (default: 1)"
    echo "  --njobs N              Parallel jobs per config (default: reads gen_njobs from each config)"
    echo "  --ckpt-path <path>     Checkpoint dir (applied to every config)"
    echo "  --concatenate          Submit per-config concatenation job"
    echo "  --no-concatenate       Disable concatenation"
    echo "  --concatenate-foldseek Enable Foldseek in concatenation"
    echo "  --dry-run              Generate configs only, don't submit"
    echo "  --no-copy              Don't copy code to run directory"
    echo "  --base-out-dir <path>  Base dir for model outputs"
    echo "  --delay <seconds>      Delay between submissions (default: 2)"
    exit 1
fi

if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "Error: Sweep config not found: $SWEEP_CONFIG"
    exit 1
fi

# Resolve to absolute paths before cd changes CWD
SWEEP_CONFIG=$(realpath "$SWEEP_CONFIG")
[ -n "$SWEEP_CONFIG_OUTPUT_DIR" ] && SWEEP_CONFIG_OUTPUT_DIR=$(realpath "$SWEEP_CONFIG_OUTPUT_DIR")
[ -n "$BASE_OUT_DIR" ] && BASE_OUT_DIR=$(realpath "$BASE_OUT_DIR")

# Load cluster configuration
load_user_info "$CLUSTER" || exit 1

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
# Step 1: Generate sweep configs
# =============================================================================
print_header "Generating Sweep Configs"
echo "  Sweep config: $SWEEP_CONFIG"
echo "  Mode: $MODE"

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
echo "  Mode: $MODE"
echo "  GPUs per job: $GPUS_PER_JOB"
echo "  Jobs per config: ${NJOBS:-from config gen_njobs}"
echo "  Configs to submit: $NUM_CONFIGS"
echo "  Delay between submissions: ${DELAY}s"
echo ""

SUBMITTED=0
FAILED=0

for config_file in $(find "$CONFIG_DIR" -name "*.yaml" | sort); do
    config_name=$(basename "$config_file" .yaml)
    echo "--- Submitting: $config_name ---"

    # Compute the Hydra-relative config name (path from configs/ root, no .yaml)
    config_hydra_name=$(python3 -c "
import os
cf = os.path.realpath('$config_file')
hd = os.path.realpath('$REPO_ROOT/projects/proteinfoundation/configs')
print(os.path.splitext(os.path.relpath(cf, hd))[0])
")

    # Build per-config out_dir override if a base directory was specified
    OUT_DIR_OVERRIDE=""
    if [ -n "$BASE_OUT_DIR" ]; then
        OUT_DIR_OVERRIDE="out_dir=$BASE_OUT_DIR/$config_name"
    fi

    # Build common flags for launch_infer.sh
    # --njobs is only passed if explicitly set; otherwise launch_infer.sh reads
    # gen_njobs from the per-config Hydra config (allowing per-config control).
    COMMON_FLAGS="--mode $MODE --gpus-per-job $GPUS_PER_JOB"
    if [ -n "$NJOBS" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --njobs $NJOBS"
    fi

    if [ "$NO_COPY" == "true" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --no-copy"
    fi
    if [ "$OVERWRITE_RUNDIR" == "true" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --overwrite-rundir"
    fi
    if [ -n "$CKPT_PATH" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --ckpt-path $CKPT_PATH"
    fi

    # Concatenation flags (if explicitly set; otherwise launch_infer.sh uses mode defaults)
    if [ "$DO_CONCATENATE" == "true" ]; then
        if [ "$CONCATENATE_FOLDSEEK" == "true" ]; then
            COMMON_FLAGS="$COMMON_FLAGS --concatenate-foldseek"
        else
            COMMON_FLAGS="$COMMON_FLAGS --concatenate"
        fi
    elif [ "$DO_CONCATENATE" == "false" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --no-concatenate"
    fi
    # If DO_CONCATENATE is empty, launch_infer.sh uses the mode default (on for val, off for infer)

    "$SCRIPT_DIR/launch_infer.sh" "$CLUSTER" \
        $COMMON_FLAGS \
        --config-name "$config_hydra_name" \
        run_name_="$config_name" \
        $OUT_DIR_OVERRIDE \
        $EXTRA_ARGS && SUBMITTED=$((SUBMITTED + 1)) || FAILED=$((FAILED + 1))

    echo ""
    # Small delay to avoid overwhelming the SLURM scheduler
    if [ "$DELAY" -gt 0 ]; then
        sleep "$DELAY"
    fi
done

# =============================================================================
# Summary
# =============================================================================
print_header "Sweep Summary"
echo "  Total configs: $NUM_CONFIGS"
echo "  Submitted:     $SUBMITTED"
echo "  Failed:        $FAILED"
echo ""
echo "  Config directory: $CONFIG_DIR"
echo "  Monitor all:      squeue -u $USER"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED job(s) failed to submit. Check output above for details."
    exit 1
fi
