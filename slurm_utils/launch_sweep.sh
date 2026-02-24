#!/bin/bash
# =============================================================================
# launch_sweep.sh - Generate sweep configs and submit SLURM jobs for each
# =============================================================================
# End-to-end script that:
#   1. Runs sweep.py to generate a set of inference configs
#   2. For each generated config, submits an inference job via launch_infer.sh
#      or launch_offline_val.sh
#
# Usage:
#   ./launch_sweep.sh <cluster> --sweep-config <path> [options] [extra_args...]
#
# Options:
#   --sweep-config <path>   Path to YAML sweep config file (required)
#   --mode <infer|val>      Submission mode: infer (default) or val (offline validation)
#   --gpus-per-job N        GPUs per SLURM job (default: 1)
#   --njobs N               Parallel jobs per config (default: 1)
#   --ckpt-path <path>      Checkpoint directory (required for val mode)
#   --dry-run               Generate configs only, don't submit jobs
#   --no-copy               Don't copy code to run directory
#   --overwrite-rundir      Allow overwriting existing run directories
#   --sweep-output-dir <p>  Override output directory for generated configs
#   --delay <seconds>       Delay between job submissions (default: 2)
#
# Examples:
#   # PPI benchmark sweep (generates 12 configs, submits each as a SLURM job)
#   ./launch_sweep.sh digs --sweep-config configs/sweeps/ppi_benchmark.yaml
#
#   # Dry run: just generate configs, don't submit
#   ./launch_sweep.sh digs --sweep-config configs/sweeps/ppi_benchmark.yaml --dry-run
#
#   # Offline validation sweep with 4 GPUs per job
#   ./launch_sweep.sh ord --sweep-config configs/sweeps/offline_val_ckpt_sweep.yaml \
#       --mode val --ckpt-path /path/to/checkpoints --gpus-per-job 4 --njobs 8
#
#   # With extra Hydra overrides applied to every config
#   ./launch_sweep.sh digs --sweep-config configs/sweeps/ppi_benchmark.yaml \
#       ckpt_path=/path/to/checkpoints
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
NJOBS=1
CKPT_PATH=""
DRY_RUN="false"
NO_COPY="false"
OVERWRITE_RUNDIR="false"
SWEEP_OUTPUT_DIR=""
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
        --sweep-output-dir)
            SWEEP_OUTPUT_DIR="$2"
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
    echo "  --njobs N              Parallel jobs per config (default: 1)"
    echo "  --ckpt-path <path>     Checkpoint dir (required for val mode)"
    echo "  --dry-run              Generate configs only, don't submit"
    echo "  --no-copy              Don't copy code to run directory"
    echo "  --delay <seconds>      Delay between submissions (default: 2)"
    exit 1
fi

if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "Error: Sweep config not found: $SWEEP_CONFIG"
    exit 1
fi

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
if [ -n "$SWEEP_OUTPUT_DIR" ]; then
    SWEEP_CMD="$SWEEP_CMD --output-dir $SWEEP_OUTPUT_DIR"
fi

if [ "$DRY_RUN" == "true" ]; then
    echo ""
    echo "=== DRY RUN: showing what would be generated ==="
    python "$SWEEP_SCRIPT" --sweep-config "$SWEEP_CONFIG" --dry-run
    echo ""
    echo "To generate and submit, remove --dry-run"
    exit 0
fi

# Run the sweep generator
eval $SWEEP_CMD

# Find generated configs
# Read output_dir from sweep config
if [ -n "$SWEEP_OUTPUT_DIR" ]; then
    CONFIG_DIR="$SWEEP_OUTPUT_DIR"
else
    CONFIG_DIR=$(python3 -c "
import yaml
with open('$SWEEP_CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('output_dir', './configs/sweep_configs'))
")
fi

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Generated config directory not found: $CONFIG_DIR"
    exit 1
fi

# Count generated configs
NUM_CONFIGS=$(find "$CONFIG_DIR" -name "*.yaml" | wc -l)
if [ "$NUM_CONFIGS" -eq 0 ]; then
    echo "Error: No configs generated in $CONFIG_DIR"
    exit 1
fi

echo ""
echo "Generated $NUM_CONFIGS config(s) in: $CONFIG_DIR"

# =============================================================================
# Step 2: Submit each config as a SLURM job
# =============================================================================
print_header "Submitting Jobs"
echo "  Mode: $MODE"
echo "  GPUs per job: $GPUS_PER_JOB"
echo "  Configs to submit: $NUM_CONFIGS"
echo "  Delay between submissions: ${DELAY}s"
echo ""

SUBMITTED=0
FAILED=0

for config_file in $(find "$CONFIG_DIR" -name "*.yaml" | sort); do
    config_name=$(basename "$config_file" .yaml)
    echo "--- Submitting: $config_name ---"

    # Build common flags
    COMMON_FLAGS=""
    if [ "$NO_COPY" == "true" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --no-copy"
    fi
    if [ "$OVERWRITE_RUNDIR" == "true" ]; then
        COMMON_FLAGS="$COMMON_FLAGS --overwrite-rundir"
    fi

    if [ "$MODE" == "val" ]; then
        # Offline validation mode
        if [ -z "$CKPT_PATH" ]; then
            echo "Error: --ckpt-path required for val mode"
            exit 1
        fi

        # Extract ckpt_name from the generated config
        CKPT_NAME=$(python3 -c "
import yaml
with open('$config_file') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('ckpt_name', ''))
")

        if [ -z "$CKPT_NAME" ]; then
            echo "  Warning: No ckpt_name in config, skipping"
            FAILED=$((FAILED + 1))
            continue
        fi

        "$SCRIPT_DIR/launch_offline_val.sh" "$CLUSTER" \
            --ckpt-path "$CKPT_PATH" \
            --ckpt-name "$CKPT_NAME" \
            --njobs "$NJOBS" \
            --gpus-per-job "$GPUS_PER_JOB" \
            --no-cleanup \
            $COMMON_FLAGS \
            run_name_="$config_name" \
            $EXTRA_ARGS && SUBMITTED=$((SUBMITTED + 1)) || FAILED=$((FAILED + 1))
    else
        # Inference mode: submit via launch_infer.sh with the config as Hydra overrides
        # We pass the generated config values as extra overrides
        "$SCRIPT_DIR/launch_infer.sh" "$CLUSTER" \
            --gpus-per-job "$GPUS_PER_JOB" \
            $COMMON_FLAGS \
            run_name_="$config_name" \
            $EXTRA_ARGS && SUBMITTED=$((SUBMITTED + 1)) || FAILED=$((FAILED + 1))
    fi

    echo ""
    # Small delay to avoid overwhelming SLURM scheduler
    if [ "$DELAY" -gt 0 ]; then
        sleep "$DELAY"
    fi
done

# =============================================================================
# Summary
# =============================================================================
print_header "Sweep Summary"
echo "  Total configs: $NUM_CONFIGS"
echo "  Submitted: $SUBMITTED"
echo "  Failed: $FAILED"
echo ""
echo "  Config directory: $CONFIG_DIR"
echo "  Monitor all: squeue -u $USER"
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED job(s) failed to submit. Check output above for details."
    exit 1
fi

