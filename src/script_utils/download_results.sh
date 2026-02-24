# REMOTE="<login-host>"
# ROOT_REMOTE="/path/to/shared/storage"

# User specific variables in the user_info file. Create file and fill in with your info following 
# the ./slurm_utils/.user_info_example file
source ./slurm_utils/.user_info


# This script takes in two or three arguemtns (in order) with the second one being optional
# The first argument in the run name
# The second one is the name of the directory where results will be stored locally
# The third one sepecifies whether to compute fid or not
# Initialize default values
flag_fid=false  # if false, compute designability; if true, compute fid

if [ $# -eq 3 ]; then
    RUN_NAME=$1
    RUN_NAME_LOCAL=$2
    flag_fid=true
elif [ $# -eq 2 ]; then
    RUN_NAME=$1
    RUN_NAME_LOCAL=$2
elif [ $# -eq 1 ]; then
    RUN_NAME=$1
    RUN_NAME_LOCAL=$1
else
    echo "Usage: $0 arg1(run name) [arg2](optional directory where to store results locally, defaults to arg1"
    exit 1
fi


RUN_DIR=$ROOT_REMOTE/$RUN_NAME
RESULTS_DIR=./results_downloaded/$RUN_NAME_LOCAL

if [ -d $RESULTS_DIR ]; then
    echo "Warning: Directory $RESULTS_DIR already exists."
    exit 1
else
    mkdir -p $RESULTS_DIR
fi

cd $RESULTS_DIR

# Downlaod result files and generations
if [ "$flag_fid" = true ]; then
    # Don't download pdb files for fid
    echo "Downloading results for FID"
    rsync -avm --include="*/" --include="*.csv" --exclude="*.pdb" --exclude="*esm_*" --exclude="*" $USER@$REMOTE:$RUN_DIR/inference/ .
else
    # Download pdb files for foldseek diversity evaluation
    echo "Downloading results for traditional metrics"
    rsync -avm --include="*/" --include="*.csv" --include="*.pdb" --exclude="*esm_*" --exclude="*" $USER@$REMOTE:$RUN_DIR/inference/ .
fi

# Download yaml files used for the inference runs
mkdir configs
cd configs
rsync -avm --include="*.yaml" --exclude="*" $USER@$REMOTE:$RUN_DIR/configs/inference_configs/ .
