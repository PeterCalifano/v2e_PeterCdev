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
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_folder> <input_framerate>"
    exit 1
fi

# Check if the third argument is passed
if [ "$#" -lt 3 ]; then
    echo "Event threshold not provided. Using default value of 0.005."
fi
event_thr=${3:-0.05}  # Set a default value if the third argument is not provided

# Assign inputs to variables
input_folder=$1
input_framerate=$2
#output_folder=$3

echo "Input folder path: $input_folder"
echo "Input framerate: $input_framerate"
echo "Event threshold: $event_thr"

# Get location of v2e.py from whichs
#--auto_timestamp_resolution
#--timestamp_resolution 0.05
python $v2e_location -i $input_folder --input_frame_rate $input_framerate --pos_thres $event_thr --neg_thres $event_thr --timestamp_resolution 0.01 --auto_timestamp_resolution --batch_size 40 --output_height 256 --output_width 256 --slomo_stats_plot 

# Deactivate the virtual environment
#deactivate