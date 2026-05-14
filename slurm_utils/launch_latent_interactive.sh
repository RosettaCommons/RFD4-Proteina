#!/bin/bash
# =============================================================================
# launch_latent_interactive.sh - Interactive session with full latent dev environment
# =============================================================================
# Launches an interactive container session with environment variables
# configured from .user_info_<cluster>.
#
# Architecture:
#   1. Writes an entrypoint script to a temp dir on /lustre (shared filesystem)
#   2. Runs srun --pty with the container, passing the entrypoint script as command
#   3. The entrypoint runs INSIDE the container: sets env, mounts squashfs, drops to bash
#
#   This matches how launch_train_latents.sh works (bash -c "setup && exec python"),
#   but uses an entrypoint script instead of bash -c to avoid quoting/pty issues
#   that cause commands to run on the host instead of inside the container.
#
# Usage:
#   ./launch_latent_interactive.sh <cluster> [options]
#
# Options:
#   --gpus N        Number of GPUs (default: 1)
#   --time HH:MM:SS Job time limit (default: 4:00:00)
#   --partition P   Partition to use (default: interactive)
#   --no-squash     Disable squashfs mounting
#
# Examples:
#   ./launch_latent_interactive.sh ord
#   ./launch_latent_interactive.sh ord --gpus 2
#   ./launch_latent_interactive.sh ord --gpus 4 --time 8:00:00
#   ./launch_latent_interactive.sh ord --no-squash
# =============================================================================

set -e

# Get script directory and source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
strip_inherited_slurm_env

# Default values
CLUSTER=""
NGPUS=1
TIME="4:00:00"
PARTITION="interactive"
NO_SQUASH="false"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NGPUS="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --no-squash)
            NO_SQUASH="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 <cluster> [--gpus N] [--time HH:MM:SS] [--partition P] [--no-squash]"
            echo ""
            echo "Arguments:"
            echo "  cluster      Cluster name (ord, gb200, digs)"
            echo ""
            echo "Options:"
            echo "  --gpus N        Number of GPUs (default: 1)"
            echo "  --time T        Time limit (default: 4:00:00)"
            echo "  --partition P   Partition (default: interactive)"
            echo "  --no-squash     Disable squashfs dataset mounting"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [ -z "$CLUSTER" ]; then
                CLUSTER="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$CLUSTER" ]; then
    echo "Usage: $0 <cluster> [--gpus N] [--time HH:MM:SS] [--partition P]"
    echo "Available clusters: ord, gb200, digs"
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

# Override USE_SQUASHFS_MOUNTS based on --no-squash flag
if [ "$NO_SQUASH" == "true" ]; then
    USE_SQUASHFS_MOUNTS="false"
fi

print_header "Interactive Session Configuration"
echo "  Cluster:       $CLUSTER"
echo "  Partition:     $PARTITION"
echo "  GPUs:          $NGPUS"
echo "  Time:          $TIME"
echo "  Container:     $(basename $CONTAINER_IMAGE)"
echo "  Repo root:     $REPO_ROOT"
echo "  Squashfs:      ${USE_SQUASHFS_MOUNTS:-false}"

# Setup PYTHONPATH and env vars (same as training script)
source_venv "$REPO_ROOT"
setup_pythonpath "$REPO_ROOT"
setup_env_vars

# Build container mounts string (for --container-mounts)
CONTAINER_MOUNTS_STR=$(build_container_mounts_string)
CONTAINER_CMD=$(get_srun_cmd)

print_mount_summary

# PYTHONPATH for inside the container
CONTAINER_PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/models/rf3/src:$REPO_ROOT/projects/proteinfoundation:$REPO_ROOT/lib/atomworks/src"

# Create a temporary directory for the session on the shared filesystem
# (accessible inside the container via /lustre:/lustre:rw mount)
SESSION_DIR=$(mktemp -d "$REPO_ROOT/projects/proteinfoundation/.interactive_session_XXXXXX")
trap "rm -rf $SESSION_DIR" EXIT

# Write .env file (same as training script)
write_env_file "$SESSION_DIR/.env"

# =============================================================================
# Write entrypoint script to shared filesystem
# =============================================================================
# This script runs INSIDE the container. Using a script file instead of
# bash -c "..." avoids quoting/pty issues that cause pyxis to run commands
# on the host instead of inside the container.
#
# Note: We write CONTAINER_ENV values directly (not via build_container_env_cmds)
# because that function escapes $PATH for bash -c context. In a script file,
# we need raw values so bash expands them normally.
ENTRYPOINT="$SESSION_DIR/entrypoint.sh"

cat > "$ENTRYPOINT" << 'EOF'
#!/bin/bash
set -e

EOF

# Add container environment setup (SLURM paths etc.)
# Write raw CONTAINER_ENV values directly — no escaping needed for a script file
if [ ${#CONTAINER_ENV[@]} -gt 0 ]; then
    echo "# Container environment (SLURM paths for signal propagation)" >> "$ENTRYPOINT"
    for env_cmd in "${CONTAINER_ENV[@]}"; do
        echo "$env_cmd" >> "$ENTRYPOINT"
    done
    echo "" >> "$ENTRYPOINT"
fi

# Add squashfs mount commands if enabled
# Write directly from SQUASHFS_MOUNTS array (not build_squashfs_mount_cmds)
if [ "$USE_SQUASHFS_MOUNTS" == "true" ] && [ ${#SQUASHFS_MOUNTS[@]} -gt 0 ]; then
    echo '# Mount squashfs datasets' >> "$ENTRYPOINT"
    echo 'echo "Mounting squashfs datasets..."' >> "$ENTRYPOINT"

    # Collect all mount points for a single mkdir -p
    MKDIR_ARGS=""
    for sqfs_mount in "${SQUASHFS_MOUNTS[@]}"; do
        mount_point="${sqfs_mount#*:}"
        MKDIR_ARGS="$MKDIR_ARGS $mount_point"
    done
    echo "mkdir -p$MKDIR_ARGS" >> "$ENTRYPOINT"

    # Add squashfuse commands
    for sqfs_mount in "${SQUASHFS_MOUNTS[@]}"; do
        sqfs_file="${sqfs_mount%%:*}"
        mount_point="${sqfs_mount#*:}"
        mount_name="$(basename "$mount_point")"
        echo "squashfuse -o ro $sqfs_file $mount_point && echo '[squash] Mounted $mount_name'" >> "$ENTRYPOINT"
    done
    echo 'echo "All squashfs datasets mounted."' >> "$ENTRYPOINT"
    echo "" >> "$ENTRYPOINT"
fi

# Add PYTHONPATH, GPU check, .env copy, and drop to interactive bash
# Use a non-expanding heredoc ('INNER_EOF') so $PYTHONPATH etc. are NOT expanded now
# — they'll be expanded when the script runs inside the container.
cat >> "$ENTRYPOINT" << INNER_EOF
# Environment
export PYTHONPATH=$CONTAINER_PYTHONPATH:\$PYTHONPATH
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Copy .env to working directory
if [ -f "$SESSION_DIR/.env" ]; then
    cp "$SESSION_DIR/.env" "$REPO_ROOT/projects/proteinfoundation/.env" 2>/dev/null || true
fi

# GPU info
/usr/bin/nvidia-smi || true
echo ""
echo "=== Interactive session ready ==="
echo "  Working dir: \$(pwd)"
echo "  Python:      \$(which python) (\$(python --version 2>&1))"
echo "  GPUs:        \$(nvidia-smi -L 2>/dev/null | wc -l)"
echo "================================="
echo ""

# Drop to interactive bash
exec /usr/bin/bash
INNER_EOF

chmod +x "$ENTRYPOINT"

# Show the generated entrypoint for debugging
echo ""
echo "Entrypoint script: $ENTRYPOINT"

print_header "Launching Interactive Session"
echo "Press Ctrl+D or type 'exit' to end the session"
if [ "$USE_SQUASHFS_MOUNTS" != "true" ]; then
    echo "Note: Squashfs datasets not mounted (--no-squash)"
fi
echo ""

# Construct srun command based on cluster type
if [ "$CLUSTER_TYPE" == "digs" ]; then
    # DIGS uses Apptainer
    srun --cpu-bind=none --kill-on-bad-exit -p "$PARTITION" -t "$TIME" -J latent-dev \
        -N 1 --ntasks=1 --gres=gpu:$NGPUS \
        --pty apptainer exec --nv "$CONTAINER_IMAGE" "$ENTRYPOINT"
else
    # OCI/GB200 use Docker squash images via pyxis
    #
    # Key difference from the old script:
    #   OLD: srun --pty --container-image=... bash -c "setup_cmds && exec bash"
    #   NEW: srun --pty --container-image=... /path/to/entrypoint.sh
    #
    # The old approach used bash -c with nested quoting, which interacts badly
    # with --pty and pyxis — commands could end up running on the host (where
    # mkdir, squashfuse etc. don't exist) instead of inside the container.
    #
    # The new approach passes a script file as the command. The script lives on
    # /lustre (mounted into the container via /lustre:/lustre:rw), so it's
    # accessible inside the container. This is the same pattern as
    # launch_train_latents.sh which runs bash -c "..." inside an SBATCH script.
    srun --cpu-bind=none --kill-on-bad-exit -p "$PARTITION" -A "$ACCOUNT" -t "$TIME" -J latent-dev \
        -N 1 --ntasks=1 --gres=gpu:$NGPUS \
        $CONTAINER_CMD \
        --container-mounts="$CONTAINER_MOUNTS_STR" \
        --container-workdir="$REPO_ROOT/projects/proteinfoundation" \
        --pty "$ENTRYPOINT"
fi
