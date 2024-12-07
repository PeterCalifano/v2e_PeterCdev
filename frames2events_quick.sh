#!/bin/bash

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

python v2e.py -i $input_folder --input_frame_rate $input_framerate --auto_timestamp_resolution --dvs_params 'clean' --batch_size 8 --output-folder $output_folder --input_slowmotion_factor 1 --output_height 512 --output_width 512 --dvs_exposure duration 0.1