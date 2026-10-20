# Helpers shared by the cpu and gpu test lanes.

# Populate BIND_ARGS with the repo bind plus every path in $APPTAINER_BIND_PATHS
# (`;`-separated, the same convention exec_core.sh uses).
build_apptainer_bind_args() {
  BIND_ARGS=(--bind "$PWD:/workspace")
  if [ -n "${APPTAINER_BIND_PATHS:-}" ]; then
    IFS=';' read -ra _EXTRA_BINDS <<< "$APPTAINER_BIND_PATHS"
    for _bind in "${_EXTRA_BINDS[@]}"; do
      [ -n "$_bind" ] && BIND_ARGS+=(--bind "$_bind")
    done
  fi
}

# Append the cluster's squashfs datasets to BIND_ARGS as apptainer image binds, the same
# way the launch scripts mount them. Without this the training datasets resolve only if
# the allocated node happens to have $SQUASH_MOUNT_PATH populated already, and a node that
# does not silently sends every dataset down its fallback path.
# Reads the SQUASHFS_MOUNTS array from the sourced cluster profile ($RFD4_CI_PROFILE).
append_squashfs_binds() {
  [ "${USE_SQUASHFS_MOUNTS:-false}" == "true" ] || return 0
  local entry sqfs mount_point
  for entry in "${SQUASHFS_MOUNTS[@]}"; do
    sqfs="${entry%%:*}"
    mount_point="${entry#*:}"
    if [ -r "$sqfs" ]; then
      BIND_ARGS+=(--bind "${sqfs}:${mount_point}:image-src=/")
    else
      echo "[squash] SKIP ${mount_point} -- unreadable: ${sqfs}" >&2
    fi
  done
}
