#!/bin/bash
# =============================================================================
# common.sh - Shared utilities for SLURM job launching
# =============================================================================
# This file provides container-aware utilities for running jobs on different
# clusters (DIGS with Apptainer, OCI/GB200 with Docker).
#
# Usage: source this file in your launch scripts
#   source ./scripts/common.sh
# =============================================================================

# Strip any inherited SLURM_*/SBATCH_* env vars from a surrounding allocation
# (e.g. when a launcher is invoked from inside an `salloc`/interactive job).
# Otherwise those vars leak into the sbatch submission and cause inner srun
# steps to fail with "Memory required by task is not available". Only affects
# the calling script's process — the parent shell's env is untouched.
strip_inherited_slurm_env() {
    while IFS= read -r _v; do unset "$_v"; done < <(compgen -e | grep -E '^(SLURM|SBATCH)_')
}

# Find repository root by looking for .project-root file
find_repo_root() {
    local current_dir="$1"
    while [ "$current_dir" != "/" ]; do
        if [ -f "$current_dir/.project-root" ]; then
            echo "$current_dir"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
    done
    return 1
}

# Load user info for a specific cluster
# Usage: load_user_info <cluster_name>
# Example: load_user_info ord
load_user_info() {
    local cluster="$1"
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    if [ -z "$cluster" ]; then
        echo "Error: Cluster name required (digs, ord, gb200)"
        return 1
    fi
    
    local user_info_file="$script_dir/.user_info_${cluster}"
    if [ ! -f "$user_info_file" ]; then
        echo "Error: User info file not found: $user_info_file"
        echo "Available configs:"
        ls -1 "$script_dir"/.user_info_* 2>/dev/null || echo "  (none found)"
        return 1
    fi
    
    local repo_root
    repo_root=$(find_repo_root "$script_dir")

    # Shared defaults (.ipd/.env.{cluster}), then per-user .user_info_{cluster}.
    # CONTAINER_IMAGE is set only in .user_info_{cluster}, not in .ipd/.env.*.
    if [ -n "$repo_root" ]; then
        local env_file="$repo_root/.ipd/.env.${cluster}"
        if [ -f "$env_file" ]; then
            echo "Loading shared env: $env_file"
            set -a
            source "$env_file"
            set +a
        fi
    fi

    echo "Loading cluster config: $user_info_file"
    source "$user_info_file"

    case "$cluster" in
        ord|gb200)
            if [ -z "${CONTAINER_IMAGE:-}" ]; then
                echo "Error: CONTAINER_IMAGE not set in $user_info_file" >&2
                echo "Set CONTAINER_IMAGE to a .sqsh path or GitLab registry URL." >&2
                return 1
            fi
            local _pyxis_image
            _pyxis_image=$(resolve_pyxis_container_image "$CONTAINER_IMAGE")
            if [ "$_pyxis_image" != "$CONTAINER_IMAGE" ]; then
                echo "Container image (Pyxis/Enroot): $_pyxis_image"
                CONTAINER_IMAGE="$_pyxis_image"
            fi
            populate_docker_container_env
            ;;
    esac

    return 0
}

# Convert Docker pull reference to Enroot/Pyxis form (registry#repository:tag).
# Pyxis passes the value to enroot as docker://<this_string>.
#   registry.example.com:5005/genair/.../latent-dev-v26_01:latest
#     -> registry.example.com:5005#genair/.../latent-dev-v26_01:latest
resolve_pyxis_container_image() {
    local image="${1:-}"
    [ -z "$image" ] && return 0

    if [[ "$image" == docker://* ]]; then
        image="${image#docker://}"
    fi
    if [[ "$image" == /* ]]; then
        printf '%s' "$image"
        return 0
    fi
    if [[ "$image" == *"#"* ]]; then
        printf '%s' "$image"
        return 0
    fi
    if [[ "$image" == */* ]]; then
        local registry="${image%%/*}"
        local path_and_tag="${image#*/}"
        printf '%s#%s' "$registry" "$path_and_tag"
        return 0
    fi
    printf '%s' "$image"
}

# Build CONTAINER_ENV for docker/pyxis clusters from .ipd/.env.* + user WANDBKEY.
# Users should not duplicate data-path exports in .user_info_*.
populate_docker_container_env() {
    if [ ${#CONTAINER_ENV[@]} -gt 0 ]; then
        return 0
    fi

    local slurm_conf="${SLURM_CONF:-/etc/slurm/slurm.conf}"
    local cluster_name="${CLUSTER:-ord}"

    CONTAINER_ENV=(
        'unset DEBUG 2>/dev/null || true'
        'export DEBUG=false'
        'export PATH=/usr/bin:/usr/local/bin:/cm/shared/apps/slurm/current/bin:$PATH'
        'export LD_LIBRARY_PATH=/cm/shared/apps/slurm/current/lib64/slurm:/cm/shared/apps/slurm/current/lib64:$LD_LIBRARY_PATH'
        "export SLURM_CONF=$slurm_conf"
        "export CLUSTER=$cluster_name"
        "export DATA_PATH=$DATA_PATH"
        "export CACHE_DIR=$CACHE_DIR"
        "export LOCAL_DRIVE_MOUNT=$LOCAL_DRIVE_MOUNT"
        "export SQUASH_MOUNT_PATH=$SQUASH_MOUNT_PATH"
        "export PDB_MIRROR_PATH=$PDB_MIRROR_PATH"
        "export CCD_MIRROR_PATH=$CCD_MIRROR_PATH"
        "export HBPLUS_PATH=$HBPLUS_PATH"
        "export FOLDSEEK_EXEC=$FOLDSEEK_EXEC"
        "export MACE_MODEL_PATH=$MACE_MODEL_PATH"
        "export LOGURU_LEVEL=$LOGURU_LEVEL"
        "export DISABLE_CUEQUIVARIANCE=${DISABLE_CUEQUIVARIANCE:-false}"
    )
    if [ -n "${WANDBKEY:-}" ]; then
        CONTAINER_ENV+=("export WANDB_API_KEY=$WANDBKEY")
    fi
}

# Source the project virtual environment. Host-side orchestration only (config
# resolution + submission); apptainer jobs still need it and re-exec into the SIF.
# Usage: source_venv <repo_root>
source_venv() {
    local repo_root="$1"
    local venv_activate="$repo_root/.venv/bin/activate"

    if [ ! -f "$venv_activate" ]; then
        echo "Error: Virtual environment not found at $repo_root/.venv"
        echo "Run setup.sh first:"
        echo "  bash $repo_root/setup.sh"
        exit 1
    fi

    source "$venv_activate"
    echo "Activated venv: $repo_root/.venv"
}

# Setup PYTHONPATH for all modules
# Usage: setup_pythonpath <repo_root>
setup_pythonpath() {
    local repo_root="$1"
    
    export PYTHONPATH="${repo_root}/src:${repo_root}/lib:${repo_root}/lib/atomworks/src:$PYTHONPATH"
    echo "PYTHONPATH set to include rfproteina, vendored packages, atomworks"
}

# Setup common environment variables for training/inference
setup_env_vars() {
    export PYTHONFAULTHANDLER=1

    # Unset DEBUG_PORT to avoid issues when looking for debugger on the cluster (all clusters)
    unset DEBUG_PORT
    # OpenMP/BLAS thread limits - set to 1 for multiprocess dataloaders
    # Each dataloader worker should be single-threaded to avoid memory pressure
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
    
    # CUDA memory allocation (all clusters)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # DIGS-specific NCCL settings (L40s lack NVLink, disable P2P)
    if [ "$CLUSTER" == "digs" ]; then
        export NCCL_P2P_DISABLE=1
        export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
    fi
    
    echo "Environment variables configured"
}

# Resolve Hydra config and output values as shell variable assignments
# Usage: eval $(resolve_hydra_config <repo_root> <config_name> [overrides...])
# Example: eval $(resolve_hydra_config "$REPO_ROOT" "training_proteina" $EXTRA_ARGS)
# To load a config outside the repo configs/ tree, include "--config-path <dir>"
# among the overrides; resolve_config.py parses it like Hydra's --config-path.
resolve_hydra_config() {
    local repo_root="$1"
    local config_name="$2"
    shift 2
    local overrides="$@"

    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Activate project venv if available (needed for hydra/omegaconf)
    if [ -f "$repo_root/.venv/bin/activate" ]; then
        source "$repo_root/.venv/bin/activate"
    fi

    cd "$repo_root" && python "$script_dir/resolve_config.py" --config-name "$config_name" $overrides
}

# Emit the AF3-daemon pre-spawn block to stdout (caller appends to the SBATCH file)
# Usage: emit_af3_daemon_block <code_dir> <gpus_per_node>
emit_af3_daemon_block() {
    local code_dir="$1"
    local gpus_per_node="$2"
    cat <<EOF

# Pre-launch AF3 daemons (one supervisor per node, sockets in each node's /tmp).
# Foreground srun + shell background + --overlap keeps the supervisor alive without slurm killing it.
export AF3_IMAGE="${AF3_IMAGE:-}"
export AF3_JAX_CACHE_DIR="${AF3_JAX_CACHE_DIR:-}"
export AF3_MODEL_DIR="${AF3_MODEL_DIR:-}"
AF3_OUT=\${RUN_DIR:-.}/slurm_run_outputs && mkdir -p "\$AF3_OUT"
srun --ntasks-per-node=1 --nodes=\$SLURM_JOB_NUM_NODES --overlap --kill-on-bad-exit=0 \\
  /usr/bin/python3 $code_dir/src/rfproteina/af3/af3_daemon.py launch \\
  --num-gpus \${SLURM_GPUS_PER_NODE:-${gpus_per_node}} --log-dir "\$AF3_OUT" \\
  --skip-if-no-env 2>>"\$AF3_OUT/af3_daemons.log" >/dev/null &
AF3_DAEMON_SRUN_PID=\$!
trap 'kill -TERM \$AF3_DAEMON_SRUN_PID 2>/dev/null; wait \$AF3_DAEMON_SRUN_PID 2>/dev/null' EXIT
export AF3_LAUNCHER_SOCKET_TEMPLATE="/tmp/af3_\${SLURM_JOB_ID}/af3_daemon_rank{rank}.sock"
EOF
}

# AF2 supervisors are needed only for resolved configs that select AF2.
should_spawn_af2_daemons() {
    [ "${AF2_NEEDED:-false}" = "true" ] &&
        [ -n "${AF2_IMAGE:-}" ] &&
        [ -n "${AF2_MODEL_DIR:-}" ] &&
        [ -n "${AF2_JAX_CACHE_DIR:-}" ]
}

# Emit the AF2-daemon pre-spawn block to stdout (caller appends to the SBATCH file)
# Usage: emit_af2_daemon_block <code_dir> <gpus_per_node>
emit_af2_daemon_block() {
    local code_dir="$1"
    local gpus_per_node="$2"
    local resident_quoted
    printf -v resident_quoted '%q' "${AF2_REQUIRED_MODELS:-}"
    cat <<EOF

# Pre-launch AF2 daemons (one supervisor per node, sockets in each node's /tmp).
export AF2_RESIDENT_MODELS=$resident_quoted
AF2_OUT=\${RUN_DIR:-.}/slurm_run_outputs && mkdir -p "\$AF2_OUT"
srun --ntasks-per-node=1 --nodes=\$SLURM_JOB_NUM_NODES --overlap --kill-on-bad-exit=0 \\
  /usr/bin/python3 $code_dir/src/rfproteina/af2/af2_daemon.py launch \\
  --num-gpus \${SLURM_GPUS_PER_NODE:-${gpus_per_node}} --log-dir "\$AF2_OUT" \\
  --skip-if-no-env 2>>"\$AF2_OUT/af2_daemons.log" >/dev/null &
AF2_DAEMON_SRUN_PID=\$!
trap 'for pid in \${AF3_DAEMON_SRUN_PID:-} \${AF2_DAEMON_SRUN_PID:-}; do kill -TERM "\$pid" 2>/dev/null || true; wait "\$pid" 2>/dev/null || true; done' EXIT
export AF2_LAUNCHER_SOCKET_TEMPLATE="/tmp/af2_\${SLURM_JOB_ID}/af2_daemon_rank{rank}.sock"
EOF
}

# Get the srun command with appropriate container settings
# Usage: SRUN_CMD=$(get_srun_cmd)
get_srun_cmd() {
    if [ -z "$CLUSTER" ]; then
        echo "Error: CLUSTER not set. Call load_user_info first." >&2
        return 1
    fi
    
    case "$CLUSTER" in 
        ord|gb200)
            # OCI/GB200 use Docker squash images via srun --container-image
            if [ -z "$CONTAINER_IMAGE" ]; then
                echo "Error: CONTAINER_IMAGE not set for $CLUSTER cluster" >&2
                return 1
            fi
            echo "--container-image=$CONTAINER_IMAGE"
            ;;
    esac
}

# Build SBATCH header lines
# Usage: build_sbatch_header
# Outputs SBATCH directives to stdout
build_sbatch_header() {
    local job_name="${JOB_NAME:-rfproteina_job}"
    local partition="${PARTITION:-batch}"
    local runtime="${RUNTIME:-04:00:00}"
    local nodes="${NODES:-1}"
    local gpus_per_node="${GPUS_PER_NODE:-8}"
    local gres="${GRES:-}"
    local cpus_per_task="${CPUS_PER_TASK:-}"
    
    local mem="${MEM:-}"

    # Calculate ntasks_per_node based on GPU specification
    local ntasks_per_node
    if [ -n "$NTASKS_PER_NODE" ]; then
        # Explicit NTASKS_PER_NODE takes precedence
        ntasks_per_node="$NTASKS_PER_NODE"
    elif [ -n "$gres" ]; then
        # Extract GPU count from gres spec (e.g., "gpu:2" or "gpu:a100:2")
        # Default to 1 if can't parse
        ntasks_per_node=$(echo "$gres" | grep -oP 'gpu:[^:]*:\K\d+$|gpu:\K\d+$' || echo "1")
    else
        # Default: use gpus_per_node
        ntasks_per_node="$gpus_per_node"
    fi

    echo "#!/bin/bash"

    # Account - skip for DIGS
    if [ "$SKIP_ACCOUNT" != "true" ] && [ -n "$ACCOUNT" ]; then
        echo "#SBATCH --account $ACCOUNT"
    fi

    echo "#SBATCH --job-name $job_name"
    echo "#SBATCH --partition $partition"
    echo "#SBATCH --time $runtime"
    echo "#SBATCH --nodes $nodes"

    # GPU allocation: use --gres if specified, otherwise --gpus-per-node
    # Note: SLURM doesn't allow both --gres and --gpus-per-node
    if [ -n "$gres" ]; then
        echo "#SBATCH --gres=$gres"
    else
        echo "#SBATCH --gpus-per-node $gpus_per_node"
    fi

    echo "#SBATCH --ntasks-per-node $ntasks_per_node"
    if [ -n "$cpus_per_task" ] && [ "${SKIP_CPUS_PER_TASK:-false}" != "true" ]; then
        echo "#SBATCH --cpus-per-task $cpus_per_task"
    fi
    if [ -n "$mem" ]; then
        echo "#SBATCH --mem $mem"
    fi
    echo "#SBATCH --open-mode append"
    echo "#SBATCH --requeue"

    if [ -n "$SIGNAL_TIME" ]; then
        echo "#SBATCH --signal SIGUSR1@$SIGNAL_TIME"
    fi

    # Exclude nodes if specified
    if [ -n "$EXCLUDE_NODES" ]; then
        echo "#SBATCH --exclude=$EXCLUDE_NODES"
    fi

    # Extra SLURM options
    if [ -n "$SLURM_EXTRA_OPTIONS" ]; then
        echo "#SBATCH $SLURM_EXTRA_OPTIONS"
    fi
}

# Emit the open-file ulimit bump. MUST be appended only after every #SBATCH directive
emit_ulimit_block() {
    cat <<'EOF'

# Raise the open-file soft limit to the hard cap. Low defaults (e.g. 1024 on coreweave)
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
EOF
}

# Detect GPU type and set appropriate batch size
# Usage: BATCH_SIZE=$(get_batch_size_for_gpu)
get_batch_size_for_gpu() {
    local gpu_type=""
    
    # Try to detect from SLURM
    if [ -n "$SLURM_JOB_GRES" ]; then
        gpu_type=$(echo "$SLURM_JOB_GRES" | grep -oP '(?<=gpu:)[^:]+' | head -n 1)
    fi
    
    if [ -z "$gpu_type" ] && [ -n "$SLURM_JOB_CONSTRAINTS" ]; then
        gpu_type=$(echo "$SLURM_JOB_CONSTRAINTS" | grep -oP '(L40S?|A100|H200|B200)' | head -n 1)
    fi
    
    case "$gpu_type" in
        l40|L40|l40s|L40S)
            echo "8"
            ;;
        a100|A100)
            echo "16"
            ;;
        h200|H200)
            echo "24"
            ;;
        b200|B200)
            echo "32"
            ;;
        *)
            # Default
            echo "8"
            ;;
    esac
}

# RSYNC command for copying code to remote
# Usage: get_rsync_command <local_repo_root> <remote_user> <remote_host> <remote_dir>
get_rsync_command() {
    local local_root="$1"
    local remote_user="$2"
    local remote_host="$3"
    local remote_dir="$4"
    
    echo "rsync -az \\
    --exclude='checkpoints' \\
    --exclude='ckpt' \\
    --exclude='*.ckpt' \\
    --exclude='data_orig' \\
    --exclude='/data' \\
    --exclude='*.pkl' \\
    --exclude='*.pdb' \\
    --exclude='lightning_logs' \\
    --exclude='notebooks' \\
    --exclude='__pycache__' \\
    --exclude='store' \\
    --exclude='store_keep' \\
    --exclude='tmp' \\
    --exclude='wandb' \\
    --exclude='.env' \\
    --exclude='.git' \\
    --exclude='.gitignore' \\
    --exclude='*.ipynb' \\
    --exclude='inference' \\
    --exclude='samples_baselines' \\
    --exclude='.ipd/apptainer/*.sif' \\
    --exclude='.ipd/docker/*.sqsh' \\
    $local_root/ $remote_user@$remote_host:$remote_dir"
}

# Print a section header for logging
print_header() {
    local msg="$1"
    echo ""
    echo "=============================================="
    echo "$msg"
    echo "=============================================="
}

# Print cluster configuration summary
print_cluster_config() {
    print_header "Cluster Configuration"
    echo "  Cluster:         $CLUSTER"
    echo "  Partition:       ${PARTITION:-not set}"
    echo "  GPUs per node:   ${GPUS_PER_NODE:-not set}"
    echo "  Runtime:         ${RUNTIME:-not set}"
    
    if [ "$CONTAINER_TYPE" == "apptainer" ]; then
        echo "  Container:       Apptainer (exec_core.sh → .ipd/.env.$CLUSTER)"
    else
        echo "  Container:       Docker ($CONTAINER_IMAGE)"
    fi
    
    if [ "$SKIP_ACCOUNT" == "true" ]; then
        echo "  Account:         (skipped for DIGS)"
    else
        echo "  Account:         ${ACCOUNT:-not set}"
    fi
    
    if [ -n "$DATASET_SQUASH_MOUNTS" ]; then
        echo "  Dataset mounts:  $DATASET_SQUASH_MOUNTS"
    fi
    echo ""
}

# =============================================================================
# Squashfs Dataset Mounting
# =============================================================================
# For mounting squashfs dataset files (separate from the container image)
# Set DATASET_SQUASH_MOUNTS in your .user_info file as semicolon-separated pairs:
#   DATASET_SQUASH_MOUNTS="/path/to/data.sqsh:/mount/point;/path/to/data2.sqsh:/mount/point2"
#
# Usage in sbatch script:
#   source common.sh
#   MOUNT_SCRIPT=$(generate_mount_script)
#   # Then use srun with mount wrapper: srun $MOUNT_SCRIPT python train.py

# Generate a mount wrapper script for squashfs datasets
# Returns path to the generated script
generate_mount_script() {
    local script_path="${1:-./mount_datasets.sh}"
    
    if [ -z "$DATASET_SQUASH_MOUNTS" ]; then
        # No datasets to mount, return empty
        echo ""
        return 0
    fi
    
    cat > "$script_path" << 'MOUNT_SCRIPT_EOF'
#!/bin/bash
# Auto-generated squashfs mount wrapper script
set -e

# Mount datasets specified in DATASET_SQUASH_MOUNTS
# Format: "squash_file:mount_point;squash_file2:mount_point2"
DATASET_SQUASH_MOUNTS="${DATASET_SQUASH_MOUNTS:-}"

mount_datasets() {
    if [ -z "$DATASET_SQUASH_MOUNTS" ]; then
        return 0
    fi
    
    IFS=';' read -ra MOUNTS <<< "$DATASET_SQUASH_MOUNTS"
    for mount_spec in "${MOUNTS[@]}"; do
        if [ -z "$mount_spec" ]; then
            continue
        fi
        
        IFS=':' read -r squash_file mount_point <<< "$mount_spec"
        
        if [ -z "$squash_file" ] || [ -z "$mount_point" ]; then
            echo "[$(hostname):$$] WARNING: Invalid mount spec: $mount_spec"
            continue
        fi
        
        mkdir -p "$mount_point"
        
        if mountpoint -q "$mount_point" 2>/dev/null; then
            echo "[$(hostname):$$] $mount_point already mounted, reusing..."
        else
            echo "[$(hostname):$$] Mounting $squash_file -> $mount_point"
            squashfuse -o ro "$squash_file" "$mount_point"
            if [ $? -eq 0 ]; then
                echo "[$(hostname):$$] Successfully mounted"
            else
                echo "[$(hostname):$$] ERROR: Failed to mount $squash_file"
                exit 1
            fi
        fi
    done
}

# Mount all datasets
mount_datasets

# Execute the actual command
exec "$@"
MOUNT_SCRIPT_EOF
    
    chmod +x "$script_path"
    echo "$script_path"
}

# Unmount all datasets (call at end of job)
unmount_datasets() {
    if [ -z "$DATASET_SQUASH_MOUNTS" ]; then
        return 0
    fi
    
    IFS=';' read -ra MOUNTS <<< "$DATASET_SQUASH_MOUNTS"
    for mount_spec in "${MOUNTS[@]}"; do
        if [ -z "$mount_spec" ]; then
            continue
        fi
        
        IFS=':' read -r squash_file mount_point <<< "$mount_spec"
        
        if [ -n "$mount_point" ] && mountpoint -q "$mount_point" 2>/dev/null; then
            echo "[$(hostname):$$] Unmounting $mount_point"
            fusermount -u "$mount_point" 2>/dev/null || true
        fi
    done
}

# =============================================================================
# Unified Configuration Helpers (Array-based)
# =============================================================================
# These functions process the unified array-based configuration from .user_info:
#   ENV_VARS=("KEY=VALUE" ...)           - Environment variables for .env file
#   CONTAINER_MOUNTS=("HOST:CONTAINER:MODE" ...)  - SLURM container mounts
#   SQUASHFS_MOUNTS=("SQFS:MOUNTPOINT" ...)       - Squashfs files to mount

# Write .env file with WandB key and any ENV_VARS (if defined)
# Usage: write_env_file "/path/to/.env"
write_env_file() {
    local env_file="$1"

    # Create the file with restrictive permissions before writing any content so secrets
    # (WANDB_API_KEY) are never readable by other users on shared cluster storage.
    : > "$env_file"
    chmod 600 "$env_file"

    echo "# Auto-generated environment variables" >> "$env_file"
    echo "# Generated by launch script on $(date)" >> "$env_file"
    echo "" >> "$env_file"

    # Write ENV_VARS if defined (for backwards compatibility)
    if [ -n "${ENV_VARS+x}" ] && [ ${#ENV_VARS[@]} -gt 0 ]; then
        for var in "${ENV_VARS[@]}"; do
            echo "$var" >> "$env_file"
        done
    fi

    # Add WandB API key if available
    if [ -n "${WANDBKEY:-}" ]; then
        echo "WANDB_API_KEY=$WANDBKEY" >> "$env_file"
    fi

    local count=0
    [ -n "${ENV_VARS+x}" ] && count=${#ENV_VARS[@]}
    echo "  Created .env with $count variables"
}

# Build container mounts string from CONTAINER_MOUNTS array
# Usage: MOUNTS_STRING=$(build_container_mounts_string)
# Returns: comma-separated string for --container-mounts
build_container_mounts_string() {
    local mounts=""
    
    if [ ${#CONTAINER_MOUNTS[@]} -gt 0 ]; then
        for mount in "${CONTAINER_MOUNTS[@]}"; do
            if [ -n "$mounts" ]; then
                mounts="$mounts,$mount"
            else
                mounts="$mount"
            fi
        done
    fi

    if [ -n "$EXTRA_CONTAINER_MOUNTS" ]; then
        if [ -n "$mounts" ]; then
            mounts="$mounts,$EXTRA_CONTAINER_MOUNTS"
        else
            mounts="$EXTRA_CONTAINER_MOUNTS"
        fi
    fi
    
    # Ensure enroot config is mounted when available (required by some OCI clusters)
    local enroot_host="/home/${USER}/.config/enroot"
    local enroot_mount="${enroot_host}:${enroot_host}:ro"
    if [ -d "$enroot_host" ]; then
        if [[ ",$mounts," != *",$enroot_mount,"* ]]; then
            if [ -n "$mounts" ]; then
                mounts="$mounts,$enroot_mount"
            else
                mounts="$enroot_mount"
            fi
        fi
    fi
    
    # Add /dev/fuse for squashfs mounting inside container
    # Note: We don't add .sqfs files here - they're accessible via /lustre mount
    # Adding .sqfs files to --container-mounts causes pyxis to auto-mount them
    if [ "${USE_SQUASHFS_MOUNTS:-false}" == "true" ] && [ ${#SQUASHFS_MOUNTS[@]} -gt 0 ]; then
        mounts="$mounts,/dev/fuse:/dev/fuse"
    fi
    
    echo "$mounts"
}

# Build squashfs mount commands for inside container
# Usage: SQFS_CMDS=$(build_squashfs_mount_cmds)
# Returns: bash commands to create dirs and run squashfuse
build_squashfs_mount_cmds() {
    if [ "${USE_SQUASHFS_MOUNTS:-false}" != "true" ] || [ ${#SQUASHFS_MOUNTS[@]} -eq 0 ]; then
        echo ""
        return 0
    fi
    
    local mkdir_cmd="mkdir -p"
    local mount_cmds=""
    
    for sqfs_mount in "${SQUASHFS_MOUNTS[@]}"; do
        local sqfs_file="${sqfs_mount%%:*}"
        local mount_point="${sqfs_mount#*:}"
        
        if [ -n "$sqfs_file" ] && [ -n "$mount_point" ]; then
            mkdir_cmd="$mkdir_cmd $mount_point"
            local mount_name=$(basename "$mount_point")
            if [ -n "$mount_cmds" ]; then
                mount_cmds="$mount_cmds && squashfuse -o ro $sqfs_file $mount_point && echo '[squash] Mounted $mount_name'"
            else
                mount_cmds="squashfuse -o ro $sqfs_file $mount_point && echo '[squash] Mounted $mount_name'"
            fi
        fi
    done
    
    if [ -n "$mount_cmds" ]; then
        echo "$mkdir_cmd && $mount_cmds"
    else
        echo ""
    fi
}

# Build SBATCH script variable declarations for squashfs mounts
# Usage: build_squashfs_vars >> "$SBATCH_FILE"
# Outputs: shell variable assignments for SQFS_* and MNT_* (for backward compat)
build_squashfs_vars() {
    if [ "${USE_SQUASHFS_MOUNTS:-false}" != "true" ] || [ ${#SQUASHFS_MOUNTS[@]} -eq 0 ]; then
        return 0
    fi
    
    echo "# Squashfs mount configuration (auto-generated from SQUASHFS_MOUNTS array)"
    local i=0
    for sqfs_mount in "${SQUASHFS_MOUNTS[@]}"; do
        local sqfs_file="${sqfs_mount%%:*}"
        local mount_point="${sqfs_mount#*:}"
        echo "SQFS_$i=\"$sqfs_file\""
        echo "MNT_$i=\"$mount_point\""
        ((i++))
    done
    echo "SQFS_COUNT=$i"
}

# Build container environment export commands from CONTAINER_ENV array
# Usage: CONTAINER_ENV_CMDS=$(build_container_env_cmds)
# Returns: && separated export commands for inside container bash -c
build_container_env_cmds() {
    if [ ${#CONTAINER_ENV[@]} -eq 0 ]; then
        echo ""
        return 0
    fi
    
    local cmds=""
    for env_cmd in "${CONTAINER_ENV[@]}"; do
        # Escape all $ signs so variables expand inside container's bash -c,
        # not in the heredoc/outer shell that writes the SBATCH script.
        # Note: We avoid ${var//pat/\\\${NAME}} because bash's brace matching
        # treats the } inside ${NAME} as the outer closing brace, producing
        # extra } characters in the output (corrupting SLURM_CONF etc.).
        local escaped_cmd="${env_cmd//\$/\\\$}"
        if [ -n "$cmds" ]; then
            cmds="$cmds && $escaped_cmd"
        else
            cmds="$escaped_cmd"
        fi
    done
    
    echo "$cmds"
}

# Copy codebase to a target directory (used by launch_generate.sh and launch_sweep.sh)
# Usage: copy_code_to_dir <repo_root> <code_dir> [cluster]
# Copies the project (incl. lib/atomworks) and .ipd.
# Uses --no-compress (faster on LAN/shared FS) and excludes checkpoint/text files from configs/.
copy_code_to_dir() {
    local repo_root="$1"
    local code_dir="$2"
    local cluster="${3:-}"

    mkdir -p "$code_dir/lib/atomworks" "$code_dir/src" "$code_dir/data"
    rsync -a --no-compress \
        "$repo_root/lib/atomworks/src/" "$code_dir/lib/atomworks/src/"
    rsync -a --no-compress --exclude='__pycache__' --exclude='*.pyc' \
        "$repo_root/lib/gearnet/" "$code_dir/lib/gearnet/"
    rsync -a --no-compress --exclude='store' --exclude='outputs' --exclude='__pycache__' --exclude='*.pyc' \
        "$repo_root/src/rfproteina/" "$code_dir/src/rfproteina/"
    rsync -a --no-compress --exclude='*.ckpt' --exclude='*.txt' --exclude='ckpt_files' \
        "$repo_root/configs/" "$code_dir/configs/"
    rsync -a --no-compress "$repo_root/data/benchmarks/" "$code_dir/data/benchmarks/"
    rsync -a --no-compress "$repo_root/data/" "$code_dir/data/"
    rsync -a --no-compress "$repo_root/src/script_utils/" "$code_dir/src/script_utils/"

    cp "$repo_root/.project-root" "$code_dir/"
    mkdir -p "$code_dir/.ipd/shebang" && rsync -a "$repo_root/.ipd/shebang/" "$code_dir/.ipd/shebang/"
    rsync -a "$repo_root/.ipd/".env.* "$code_dir/.ipd/" 2>/dev/null || true
    mkdir -p "$code_dir/.ipd/apptainer"

    # Apptainer: recreate container symlink with /mnt/net prefix for compute nodes
    if [ "${CONTAINER_TYPE:-}" == "apptainer" ] && [ -L "$repo_root/.ipd/apptainer/latent-dev.sif" ]; then
        local container_target
        container_target="$(readlink -f "$repo_root/.ipd/apptainer/latent-dev.sif" 2>/dev/null || true)"
        if [ -n "$container_target" ]; then
            [[ "$container_target" == /net/* ]] && container_target="/mnt${container_target}"
            ln -sf "$container_target" "$code_dir/.ipd/apptainer/latent-dev.sif"
        fi
    fi
}

# Print summary of mounts for logging
print_mount_summary() {
    if [ ${#CONTAINER_MOUNTS[@]} -gt 0 ]; then
        echo "  Container mounts: ${#CONTAINER_MOUNTS[@]} entries"
    fi
    if [ "${USE_SQUASHFS_MOUNTS:-false}" == "true" ] && [ ${#SQUASHFS_MOUNTS[@]} -gt 0 ]; then
        echo "  Squashfs mounts:  ${#SQUASHFS_MOUNTS[@]} entries"
        for sqfs_mount in "${SQUASHFS_MOUNTS[@]}"; do
            local mount_point="${sqfs_mount#*:}"
            echo "    - $mount_point"
        done
    fi
    if [ -n "${ENV_VARS+x}" ] && [ ${#ENV_VARS[@]} -gt 0 ]; then
        echo "  Environment vars: ${#ENV_VARS[@]} entries"
    fi
    if [ ${#CONTAINER_ENV[@]} -gt 0 ]; then
        echo "  Container env:    ${#CONTAINER_ENV[@]} entries"
    fi
}
