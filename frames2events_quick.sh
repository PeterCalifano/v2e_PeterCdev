#!/bin/bash

# Source the virtual environment in the root folder of the v2e repository
DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" && pwd )"

echo "frame2events_quick script directory: $DIR"

v2e_location=$(which v2e)
echo "v2e location: $v2e_location"

# If empty string, then v2e is not in the PATH
if [ -z "$v2e_location" ]; then
    echo "v2e is not in the PATH. Have you installed v2e by running install_v2e_linux? There should be a symlink in /usr/local/bin."
    exit 1
fi

# Save the current directory
current_dir=$(pwd)

cd $DIR
source .venvEventBased/bin/activate
cd $current_dir

# Check if exactly two arguments are passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_folder> <input_framerate> <output_folder>"
    exit 1
fi

# Assign inputs to variables
input_folder=$1
input_framerate=$2
output_folder=$3

echo "Input folder path: $input_folder"
echo "Input framerate: $input_framerate"

# Get location of v2e.py from whichs
python $v2e_location -i $input_folder --input_frame_rate $input_framerate --auto_timestamp_resolution --dvs_params 'clean' --batch_size 8 --output-folder $output_folder --input_slowmotion_factor 1 --output_height 512 --output_width 512 --dvs_exposure duration 0.1

# Deactivate the virtual environment
#deactivate