#!/bin/bash
# Launch CIF parsing jobs in parallel on SLURM cluster
#
# This script:
# 1. Reads a commands file with CIF parsing commands
# 2. Splits the commands into N job arrays
# 3. Launches SLURM job arrays on cpu-bf partition
#
# Usage:
#   bash scripts/launch_cif_parsing_jobs.sh <commands_file> <n_jobs> [slurm_args...]
#
# Example:
#   bash scripts/launch_cif_parsing_jobs.sh cif_parsing_commands.txt 1000
#   bash scripts/launch_cif_parsing_jobs.sh cif_parsing_commands.txt 1000 --time=4:00:00 --mem=16G

set -e

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <commands_file> <n_jobs> [slurm_args...]"
    echo ""
    echo "Arguments:"
    echo "  commands_file  Path to file containing parsing commands (one per line)"
    echo "  n_jobs         Number of parallel jobs to create"
    echo "  slurm_args     Optional additional SLURM arguments (e.g., --time=4:00:00)"
    echo ""
    echo "Example:"
    echo "  $0 cif_parsing_commands.txt 1000 --time=4:00:00 --mem=16G"
    exit 1
fi

unset DEBUG_PORT

COMMANDS_FILE="$1"
N_JOBS="$2"
shift 2
EXTRA_SLURM_ARGS="$@"

# Validate commands file
if [ ! -f "$COMMANDS_FILE" ]; then
    echo "Error: Commands file not found: $COMMANDS_FILE"
    exit 1
fi

# Get total number of commands (excluding comments and empty lines)
TOTAL_COMMANDS=$(grep -v '^#' "$COMMANDS_FILE" | grep -v '^[[:space:]]*$' | wc -l)

if [ "$TOTAL_COMMANDS" -eq 0 ]; then
    echo "Error: No commands found in $COMMANDS_FILE"
    exit 1
fi

echo "Found $TOTAL_COMMANDS commands in $COMMANDS_FILE"
echo "Splitting into $N_JOBS jobs"

# Calculate commands per job
CMDS_PER_JOB=$(( (TOTAL_COMMANDS + N_JOBS - 1) / N_JOBS ))
echo "Each job will process ~$CMDS_PER_JOB commands"

# Create temp directory for job scripts in current working directory
TEMP_DIR="./cif_parsing_jobs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DIR"
echo "Created temp directory: $TEMP_DIR"

# Filter out comments and empty lines
FILTERED_FILE="$TEMP_DIR/filtered_commands.txt"
grep -v '^#' "$COMMANDS_FILE" | grep -v '^[[:space:]]*$' > "$FILTERED_FILE"

# Split commands file into N batch files (one per job)
echo "Splitting commands into $N_JOBS batch files..."
split -n l/$N_JOBS --numeric-suffixes=1 --suffix-length=3 --additional-suffix=.txt "$FILTERED_FILE" "$TEMP_DIR/batch_"

# Get absolute path to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Create job script
JOB_SCRIPT="$TEMP_DIR/parse_job.sh"
cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=cif_parse
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/cif_parse_%A_%a.out
#SBATCH --error=logs/cif_parse_%A_%a.err

# Get the batch file for this array task
BATCH_DIR="\$1"
BATCH_FILE="\$BATCH_DIR/batch_\$(printf '%03d' \$SLURM_ARRAY_TASK_ID).txt"

echo "Task \$SLURM_ARRAY_TASK_ID: Processing batch file \$BATCH_FILE"

# Count lines in batch file
NUM_LINES=\$(wc -l < "\$BATCH_FILE")
echo "Processing \$NUM_LINES CIF files in one apptainer session..."

# Run parse_single_cif.py with the batch file (apptainer loaded once for all files)
cd "$PROJECT_ROOT"
./script_utils/parser_population/parse_single_cif.py --command-file "\$BATCH_FILE"

if [ \$? -eq 0 ]; then
    echo "Task \$SLURM_ARRAY_TASK_ID: Complete - successfully processed \$NUM_LINES files"
else
    echo "Task \$SLURM_ARRAY_TASK_ID: Failed with exit code \$?"
    exit 1
fi
EOF

chmod +x "$JOB_SCRIPT"

# Create logs directory if it doesn't exist
mkdir -p logs

# Build SLURM command
SLURM_CMD="sbatch"
SLURM_CMD="$SLURM_CMD --array=1-${N_JOBS}"

# Add extra SLURM args if provided
if [ -n "$EXTRA_SLURM_ARGS" ]; then
    SLURM_CMD="$SLURM_CMD $EXTRA_SLURM_ARGS"
fi

# Convert TEMP_DIR to absolute path
TEMP_DIR_ABS="$(cd "$TEMP_DIR" && pwd)"
SLURM_CMD="$SLURM_CMD $JOB_SCRIPT $TEMP_DIR_ABS"

# Show the command that would be run
echo ""
echo "Would run the following SLURM command:"
echo "$SLURM_CMD"
echo ""
echo "Job script location: $JOB_SCRIPT"
echo "Batch files location: $TEMP_DIR_ABS/batch_*.txt"
echo "Number of batch files: $N_JOBS"
echo ""
echo "Each job will:"
echo "  1. Load apptainer ONCE using the shebang in parse_single_cif.py"
echo "  2. Process ~$CMDS_PER_JOB CIF files inside that single apptainer session"
echo ""
echo "IMPORTANT: This script does NOT actually submit the job to prevent accidental large job submissions."
echo "To submit the job, run the above command manually or modify this script to remove the safety check."
echo ""
echo "After the jobs complete, you can clean up with:"
echo "  rm -rf $TEMP_DIR"

# DO NOT ACTUALLY RUN THE COMMAND - safety measure
# Uncomment the line below to actually submit jobs:
eval "$SLURM_CMD"

exit 0