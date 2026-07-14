#!/bin/bash
# =============================================================================
# docker_aliases.sh - Interactive container launch functions
# =============================================================================
# Source this file in your ~/.bashrc to get convenient docker launch commands:
#   source /path/to/scripts/docker_aliases.sh
#
# Available commands:
#   complexadocker [code_dir]  - Launch Complexa container for protein-foundation-models
#   latentdocker [code_dir]    - Launch Latent Dev container (simpler version)
#
# For full latent environment with squashfs mounts, use:
#   ./launch_latent_interactive.sh ord
# =============================================================================

# Complexa container for protein-foundation-models development
# Usage: complexadocker [code_dir]
#   code_dir: Optional path to mount as /workspace/protein-foundation-models (default: current dir)
complexadocker() {
    local CODE_DIR="${1:-$(pwd)}"
    local DATA_PATH="/path/to/shared/storage"
    local CONTAINER_IMAGE="/path/to/shared/storage"
    
    echo "Launching Complexa container..."
    echo "  Code dir: $CODE_DIR"
    echo "  Data path: $DATA_PATH"
    echo ""
    
    srun -p interactive -A project -t 4:00:00 -J complexa -N 1 --gres=gpu:1 \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts="/lustre:/lustre,${CODE_DIR}:/workspace/protein-foundation-models,/etc/passwd:/etc/passwd:ro,/etc/group:/etc/group:ro,/tmp:/tmp,/dev/shm:/dev/shm" \
        --container-workdir=/workspace/protein-foundation-models \
        --export=ALL,TMPDIR=/tmp,NUMBA_CACHE_DIR=/tmp,DATA_PATH=${DATA_PATH},PROJECT_HOME=/workspace/protein-foundation-models,PYTHONPATH=/workspace/protein-foundation-models/src:/workspace/protein-foundation-models:/workspace/protein-foundation-models/community_models,HOME=/workspace/protein-foundation-models \
        --pty bash -c 'ulimit -l unlimited; ulimit -s 67108864; ln -sf /workspace/protein-foundation-models/configs /workspace/protein-foundation-models/src/configs 2>/dev/null; exec bash'
}

# Simple latent dev container (without squashfs mounts)
# For full environment with squashfs mounts, use ./launch_latent_interactive.sh ord
# Usage: latentdocker [code_dir]
latentdocker() {
    local CODE_DIR="${1:-$(pwd)}"
    local DATA_PATH="/path/to/shared/storage"
    local CONTAINER_IMAGE="/path/to/shared/storage"
    
    echo "Launching Latent Dev container (simple mode, no squashfs)..."
    echo "  Code dir: $CODE_DIR"
    echo "  Data path: $DATA_PATH"
    echo "  For full environment with squashfs mounts, use: ./launch_latent_interactive.sh ord"
    echo ""
    
    srun -p interactive -A project -t 4:00:00 -J latent-dev -N 1 --gres=gpu:1 \
        --container-image="${CONTAINER_IMAGE}" \
        --container-mounts="/lustre:/lustre:rw,/cm/shared/apps/slurm:/cm/shared/apps/slurm:ro,/usr/lib/x86_64-linux-gnu/libmunge.so.2:/usr/lib/x86_64-linux-gnu/libmunge.so.2:ro,/etc/passwd:/etc/passwd:ro,/etc/group:/etc/group:ro,/var/run/munge:/var/run/munge:ro,/tmp:/tmp,/dev/shm:/dev/shm" \
        --container-workdir="${CODE_DIR}" \
        --export=ALL,DATA_PATH=${DATA_PATH},CACHE_DIR=${DATA_PATH}/cache,PYTHONFAULTHANDLER=1 \
        --pty bash -c 'export PATH=/cm/shared/apps/slurm/current/bin:$PATH; export LD_LIBRARY_PATH=/cm/shared/apps/slurm/current/lib64/slurm:/cm/shared/apps/slurm/current/lib64:$LD_LIBRARY_PATH; exec bash'
}

echo "Docker aliases loaded: complexadocker, latentdocker"

