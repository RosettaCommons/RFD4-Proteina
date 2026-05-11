#!/bin/bash
# =============================================================================
# launch_mpnn.sh - Launch ProteinMPNN/LigandMPNN jobs on generated structures
# =============================================================================
# Run this script directly on the cluster. It will prepare jobs and submit to SLURM.
#
# Usage:
#   ./launch_mpnn.sh <cluster> --input-dir <dir> --output-dir <dir> [options]
#
# Required:
#   <cluster>              Cluster name (digs, ord, gb200)
#   --input-dir <dir>      Directory containing generated CIF structures
#   --output-dir <dir>     Directory for MPNN outputs
#
# Options:
#   --njobs N              Number of parallel jobs (default: 1)
#   --num-seqs N           Sequences per structure (default: 1)
#   --model-type TYPE      ligand_mpnn or protein_mpnn (default: ligand_mpnn)
#   --temperature T        Sampling temperature (default: 0.1)
#   --follow, -f           Follow log output after submission
#
# Required .user_info settings:
#   MPNN_SCRIPT_PATH       Path to the fused_mpnn run.py script
#   MPNN_CONTAINER_IMAGE   Container image for running MPNN
#
# Examples:
#   ./launch_mpnn.sh digs --input-dir ./outputs/generated/ --output-dir ./mpnn/
#   ./launch_mpnn.sh digs --input-dir ./outputs/ --output-dir ./mpnn/ --njobs 10
#   ./launch_mpnn.sh digs --input-dir ./outputs/ --output-dir ./mpnn/ --num-seqs 4 --follow
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# Parse arguments
CLUSTER=""
INPUT_DIR=""
OUTPUT_DIR=""
NUM_JOBS=1
NUM_SEQS=1
MODEL_TYPE="ligand_mpnn"
TEMPERATURE=0.1
BATCH_SIZE=1
OMIT_AA="XC"
FOLLOW_LOGS="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --njobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --num-seqs)
            NUM_SEQS="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --omit-aa)
            OMIT_AA="$2"
            shift 2
            ;;
        --follow|-f)
            FOLLOW_LOGS="true"
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
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
    echo "Usage: $0 <cluster> --input-dir <dir> --output-dir <dir> [options]"
    echo "Available clusters: digs, ord, gb200"
    exit 1
fi

if [ -z "$INPUT_DIR" ]; then
    echo "Error: --input-dir is required"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: --output-dir is required"
    exit 1
fi

# Load cluster configuration (includes MPNN settings)
load_user_info "$CLUSTER" || exit 1

# Validate required MPNN settings from user_info
if [ -z "$MPNN_SCRIPT_PATH" ]; then
    echo "Error: MPNN_SCRIPT_PATH not set in .user_info_${CLUSTER}"
    echo "Please add MPNN_SCRIPT_PATH to your user_info file, e.g.:"
    echo "  MPNN_SCRIPT_PATH=\"/path/to/fused_mpnn/run.py\""
    exit 1
fi

if [ -z "$MPNN_CONTAINER_IMAGE" ]; then
    echo "Error: MPNN_CONTAINER_IMAGE not set in .user_info_${CLUSTER}"
    echo "Please add MPNN_CONTAINER_IMAGE to your user_info file, e.g.:"
    echo "  MPNN_CONTAINER_IMAGE=\"/path/to/container.sif\"  # for DIGS (Apptainer)"
    echo "  MPNN_CONTAINER_IMAGE=\"nvcr.io/nvidia/...\"      # for OCI/GB200 (Docker)"
    exit 1
fi

# Find repository root
REPO_ROOT=$(find_repo_root "$SCRIPT_DIR")
if [ -z "$REPO_ROOT" ]; then
    echo "Error: Could not find .project-root"
    exit 1
fi
cd "$REPO_ROOT"

source_venv "$REPO_ROOT"
setup_pythonpath "$REPO_ROOT"

# Convert to absolute paths
INPUT_DIR="$(cd "$INPUT_DIR" 2>/dev/null && pwd)" || { echo "Error: Input directory does not exist: $INPUT_DIR"; exit 1; }
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

# Use MPNN-specific settings from user_info, with defaults for optional settings
MPNN_PARTITION="${MPNN_PARTITION:-cpu}"
MPNN_TIME="${MPNN_TIME:-1:00:00}"
MPNN_CPUS="${MPNN_CPUS:-1}"
MPNN_MEM="${MPNN_MEM:-16G}"

print_header "MPNN Job Configuration"
echo "  Cluster:         $CLUSTER ($CLUSTER_TYPE)"
echo "  Input dir:       $INPUT_DIR"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Num jobs:        $NUM_JOBS"
echo "  Seqs/structure:  $NUM_SEQS"
echo "  Model type:      $MODEL_TYPE"
echo "  Temperature:     $TEMPERATURE"
echo "  Partition:       $MPNN_PARTITION"
echo "  Time limit:      $MPNN_TIME"
echo "  MPNN script:     $MPNN_SCRIPT_PATH"
echo "  Container:       $MPNN_CONTAINER_IMAGE"
echo ""

# Run the Python preparation script (generates commands without container wrapping)
print_header "Preparing MPNN Jobs"

python "$REPO_ROOT/projects/proteinfoundation/proteinfoundation/prepare_mpnn.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-chunks "$NUM_JOBS" \
    --num-seqs "$NUM_SEQS" \
    --model-type "$MODEL_TYPE" \
    --temperature "$TEMPERATURE" \
    --batch-size "$BATCH_SIZE" \
    --omit-aa "$OMIT_AA" \
    --slurm-partition "$MPNN_PARTITION" \
    --slurm-time "$MPNN_TIME" \
    --slurm-cpus "$MPNN_CPUS" \
    --slurm-mem "$MPNN_MEM" \
    --mpnn-script "$MPNN_SCRIPT_PATH"

# Now wrap the SLURM script with container execution based on cluster type
print_header "Adding Container Execution"

SLURM_SCRIPT="$OUTPUT_DIR/submit_mpnn.sh"
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM script not found: $SLURM_SCRIPT"
    exit 1
fi

# Read existing script and wrap commands with container
TEMP_SCRIPT=$(mktemp)

if [ "$CLUSTER_TYPE" == "digs" ]; then
    # DIGS uses Apptainer
    MPNN_BINDS="${MPNN_CONTAINER_BINDS:-}"

    echo "  Type: Apptainer"
    echo "  Image: $MPNN_CONTAINER_IMAGE"

    # Rewrite the script to wrap python commands with apptainer
    sed "s|python -u|/usr/bin/apptainer exec $MPNN_BINDS $MPNN_CONTAINER_IMAGE python -u|g" "$SLURM_SCRIPT" > "$TEMP_SCRIPT"

elif [ "$CLUSTER_TYPE" == "ord" ] || [ "$CLUSTER_TYPE" == "gb200" ]; then
    # OCI/GB200 use Docker via srun --container-image
    MPNN_MOUNTS="${MPNN_CONTAINER_MOUNTS:-}"

    echo "  Type: Docker (via srun --container-image)"
    echo "  Image: $MPNN_CONTAINER_IMAGE"

    # For Docker clusters, we need to wrap the python command with srun --container-image
    # The command format: srun --container-image=IMAGE --container-mounts=MOUNTS python -u ...
    sed "s|python -u|srun --container-image=$MPNN_CONTAINER_IMAGE --container-mounts=$MPNN_MOUNTS python -u|g" "$SLURM_SCRIPT" > "$TEMP_SCRIPT"

else
    echo "Error: Unknown cluster type: $CLUSTER_TYPE"
    echo "Supported cluster types: digs, ord, gb200"
    exit 1
fi

mv "$TEMP_SCRIPT" "$SLURM_SCRIPT"
chmod +x "$SLURM_SCRIPT"

# Submit to SLURM
print_header "Submitting to SLURM"

cd "$OUTPUT_DIR"
SBATCH_OUTPUT=$(sbatch "$SLURM_SCRIPT")
echo "$SBATCH_OUTPUT"

JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+$')

echo ""
echo "Monitor: squeue -u $USER -j $JOB_ID"
echo "Logs:    tail -f $OUTPUT_DIR/logs/mpnn_*.log"

# Follow logs if requested
if [ "$FOLLOW_LOGS" == "true" ]; then
    echo -e "\nFollowing logs (Ctrl+C to exit)...\n"
    sleep 2
    # Wait for log files to appear
    while ! ls "$OUTPUT_DIR/logs/mpnn_"*".log" 2>/dev/null; do
        squeue -u "$USER" -j "$JOB_ID" -h 2>/dev/null | grep -q . || { echo "Job completed or not found"; break; }
        sleep 2
    done
    tail -f "$OUTPUT_DIR/logs/mpnn_"*".log" 2>/dev/null || true
fi
