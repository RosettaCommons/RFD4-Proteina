#!/usr/bin/bash
# DEPRECATED alias for launch_generate.sh, kept so existing sbatch wrappers, saved
# commands and notebooks keep working. Every argument is forwarded unchanged.
# Delete once the old name has stopped appearing in job scripts.

echo "WARNING: scripts/launch_infer.sh is deprecated -- use scripts/launch_generate.sh" >&2

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_generate.sh" "$@"
