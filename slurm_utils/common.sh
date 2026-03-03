#!/bin/bash
# =============================================================================
# common.sh - Shared utilities for SLURM job launching
# =============================================================================
# This file provides container-aware utilities for running jobs on different
# clusters (DIGS with Apptainer, OCI/GB200 with Docker).
#
# Usage: source this file in your launch scripts
#   source ./slurm_utils/common.sh
# =============================================================================

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
    
    echo "Loading cluster config: $user_info_file"
    source "$user_info_file"

    return 0
}

# Setup PYTHONPATH for all modules
# Usage: setup_pythonpath <repo_root>
setup_pythonpath() {
    local repo_root="$1"
    
    export PYTHONPATH="${repo_root}/src:${repo_root}/models/rf3/src:${repo_root}/projects/proteinfoundation:${repo_root}/lib/atomworks/src:$PYTHONPATH"
    echo "PYTHONPATH set to include foundry, rf3, proteinfoundation, atomworks"
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
    
    # DIGS-specific settings
    if [ "$CLUSTER" == "digs" ]; then
        # CUDA memory allocation
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        
        # NCCL settings
        export NCCL_P2P_DISABLE=1
        export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
        
    fi
    
    echo "Environment variables configured"
}

# Resolve Hydra config and output values as shell variable assignments
# Usage: eval $(resolve_hydra_config <repo_root> <config_name> [overrides...])
# Example: eval $(resolve_hydra_config "$REPO_ROOT" "training_proteina" $EXTRA_ARGS)
resolve_hydra_config() {
    local repo_root="$1"
    local config_name="$2"
    shift 2
    local overrides="$@"

    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # Run the Python resolver script
    cd "$repo_root" && python "$script_dir/resolve_config.py" --config-name "$config_name" $overrides
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
    local job_name="${JOB_NAME:-proteinfoundation_job}"
    local partition="${PARTITION:-batch}"
    local runtime="${RUNTIME:-04:00:00}"
    local nodes="${NODES:-1}"
    local gpus_per_node="${GPUS_PER_NODE:-8}"
    local gres="${GRES:-}"
    local cpus_per_task="${CPUS_PER_TASK:-}"
    
    # Scale memory based on number of GPUs if not explicitly set
    # Default: ~60GB per GPU to stay within cluster limits (max ~235GB per GPU on OCI)
    local mem
    if [ -n "$MEM" ]; then
        mem="$MEM"
    else
        local mem_per_gpu_gb=60
        local total_mem_gb=$((gpus_per_node * mem_per_gpu_gb))
        mem="${total_mem_gb}g"
    fi

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
    echo "#SBATCH --mem $mem"
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

# Write ENV_VARS array to a .env file
# Usage: write_env_file "/path/to/.env"
write_env_file() {
    local env_file="$1"
    
    echo "# Auto-generated environment variables" > "$env_file"
    echo "# Generated by launch script on $(date)" >> "$env_file"
    echo "" >> "$env_file"
    
    if [ ${#ENV_VARS[@]} -gt 0 ]; then
        for var in "${ENV_VARS[@]}"; do
            echo "$var" >> "$env_file"
        done
    fi
    
    # Add WandB API key if available
    if [ -n "$WANDBKEY" ]; then
        echo "WANDB_API_KEY=$WANDBKEY" >> "$env_file"
    fi
    
    echo "  Created .env with ${#ENV_VARS[@]} variables"
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
    if [ ${#ENV_VARS[@]} -gt 0 ]; then
        echo "  Environment vars: ${#ENV_VARS[@]} entries"
    fi
    if [ ${#CONTAINER_ENV[@]} -gt 0 ]; then
        echo "  Container env:    ${#CONTAINER_ENV[@]} entries"
    fi
}

