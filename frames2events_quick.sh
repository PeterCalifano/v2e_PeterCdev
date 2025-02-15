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

echo "Sourcing virtual environment in $DIR"
cd $DIR
source .venvEventBased/bin/activate
cd $current_dir

# Check if exactly two arguments are passed
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_folder> <input_framerate>"
    exit 1
fi

# Assign inputs to variables
input_folder=$1
input_framerate=$2
output_folder=$3

# Check if the forth argument is passed
if [ "$#" -lt 4 ]; then
    echo "Event threshold not provided. Using default value of 0.05."
fi
event_thr=${4:-0.08}  # Set a default value if the third argument is not provided

echo "Input folder path: $input_folder"
echo "Input framerate: $input_framerate"
echo "Event threshold: $event_thr"

# If output folder is provided, use it
if [ -z "$output_folder" ]; then
    folder_out_option=""
    unique_output_folder_bool=0
else
    folder_out_option="-o $output_folder"
    echo "Output folder: $output_folder"
    unique_output_folder_bool=1

    # Create the output folder if it does not exist
    mkdir -p $output_folder
fi

# Get location of v2e.py from whichs
#--auto_timestamp_resolution
#--timestamp_resolution 0.05
# --dvs_exposure count 1000
python $v2e_location -i $input_folder \
--timestamp_resolution 0.05 --save_dvs_model_state --show_dvs_model_state all \
--vid_orig None --crop '256, 256, 0, 0' --dvs_exposure duration 0.05 --input_frame_rate $input_framerate \
--pos_thres $event_thr --neg_thres $event_thr --photoreceptor_noise \
--batch_size 16 --output_height 1523 --output_width 1523 --slomo_stats_plot --cutoff_hz 0.5 \
--dvs_text events_stream_txt --ddd_output $folder_out_option --no_preview --unique_output_folder $unique_output_folder_bool
# Deactivate the virtual environment
#deactivate