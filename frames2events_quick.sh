#!/bin/bash

# Check if exactly two arguments are passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input1> <input2>"
    exit 1
fi

# Assign inputs to variables
input_folder=$1
input_framerate=$2

echo "Input folder path: $input_folder"
echo "Input framerate: $input_framerate"

python v2e.py -i $input_folder --input_frame_rate $input_framerate --auto_timestamp_resolution --dvs128 --dvs_params 'clean' --batch_size 16