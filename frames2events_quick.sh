#!/bin/bash
# Export if running headless 
# TODO verify it works
# export QT_QPA_PLATFORM=offscreen

function usage() {
    echo -e "Usage: $0 required args [optional args] \nValid arguments: \n -i|--input_folder input_folder -f|--framerate_input input_framerate \n [-o|--output output_folder] \n [-t|--threshold event_threshold] \n [-r|--resolution_output width,height] \n [-c|--cut_off frequency] \n [-d|--deviation_thr sigma_threshold] \n [-b|--batch_size batch_size] \n"
    exit 1
}

# Parse options using getopt for short and long options
TEMP=$(getopt -o a,i:,f:,o:,t:,b:,r:,c:,h,s --long input_folder:,framerate_input:,output_folder:,threshold:,batch_size:,resolution_output:,cut_off:,deviation_thr:,help,show_video -n "$0" -- "$@")
if [ $? != 0 ]; then
    usage
fi

eval set -- "$TEMP"
while true; do
    case "$1" in
        -g|--grid_time_resolution)
            timestamp_resolution="$2"
            shift 2
            ;;
        -a|--auto_timestamp)
            auto_timestamp=true
            shift 1
            ;;
        -i|--input_folder)
            input_folder="$2"
            shift 2
            ;;
        -f|--framerate_input)
            framerate_input="$2"
            shift 2
            ;;
        -o|--output_folder)
            output_folder="$2"
            shift 2
            ;;
        -t|--threshold)
            event_thr="$2"
            shift 2
            ;;
        -d|--deviation_thr)
            sigma_thr="$2"
            shift 2
            ;;
        -b|--batch_size)
            batch_size_slomo="$2"
            shift 2
            ;;
        -r|--resolution_output)
            output_resolution="$2"
            shift 2
            ;;
        -c|--cut_off)   
            cut_off_frequency="$2"
            shift 2
            ;;
        -s|--show_video)
            shift 1
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            usage
            ;;
    esac
done

# Get the current directory
current_dir=$(pwd)

# Check required options
if [ -z "${input_folder}" ] || [ -z "${framerate_input}" ]; then
    usage
fi

# Set default values for optional arguments if not provided
event_thr=${event_thr:-0.08}
batch_size_slomo=${batch_size_slomo:-4}
sigma_thr=${sigma_thr:-0.02}
auto_timestamp=${auto_timestamp:-false}
cut_off_frequency=${cut_off_frequency:-0.5}

if ! [[ "$batch_size_slomo" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: batch_size must be a positive integer."
    exit 1
fi

# If output_resolution is specified, it should be in "width,height" format.
if [ -n "$output_resolution" ]; then
    IFS=',' read -r res_w res_h <<< "$output_resolution"
    # Trim whitespace
    res_w=$(echo "$res_w" | sed 's/^[ \t]*//;s/[ \t]*$//')
    res_h=$(echo "$res_h" | sed 's/^[ \t]*//;s/[ \t]*$//')

    if ! [[ "$res_w" =~ ^[1-9][0-9]*$ ]] || ! [[ "$res_h" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: resolution_output must be in the format 'width,height' with positive integers."
        exit 1
    fi
else
    res_w=256
    res_h=256
fi

# Get the directory of this script
DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" && pwd )"
echo "Using frame2events_quick script in: $DIR"

# Try to find v2e in the PATH
v2e_location=$(which v2e)
echo "Using v2e location: $v2e_location"

if [ -z "$v2e_location" ]; then
    echo "v2e is not in the PATH. Have you installed v2e by running install_v2e_venv?"
    exit 1
fi


### PRINT INFO
echo "Sourcing virtual environment in: $DIR"
cd "$DIR"
source .venvEventBased/bin/activate
cd "$current_dir"

echo "Input folder path: $input_folder"
echo "Input framerate: $framerate_input" fps
echo "Event threshold: $event_thr"
echo "Batch size: $batch_size_slomo"

if [ -n "$output_resolution" ]; then
    output_resolution_args="--output_width $res_w --output_height $res_h"
    echo "Output resolution: ${res_w}x${res_h}" pixels
else
    output_resolution_args=""
    echo "Output resolution: native"
fi


echo "Sigma threshold: $sigma_thr"
if [ -n "$timestamp_resolution" ]; then
    echo "Timestamp resolution: $timestamp_resolution"
    # Define command 
    timestamp_resolution="--timestamp_resolution $timestamp_resolution"
else
    echo "Timestamp resolution: auto"
    timestamp_resolution=""
    auto_timestamp=true # Override -a option
fi
echo "Auto timestamp: $auto_timestamp"
if [ -n "$cut_off_frequency" ]; then
    echo "Cut off frequency: $cut_off_frequency Hz" 
    cut_off_frequency="--cutoff_hz $cut_off_frequency"
else
    echo "Cut off frequency: not set"
    cut_off_frequency=""
fi

overwrite_output_folder_option="" # Never overwrite output folder
if ! [ -n "$output_folder" ]; then
    # Compose output folder name with default pattern
    output_folder=$(basename $input_folder)"_"$event_thr"thr_"$res_w"x"$res_h"_v2e_out"
    output_folder=${output_folder// /_}
    output_folder=${output_folder//./p} 
    output_folder=${output_folder//:/_} 
    output_folder=${output_folder//,/}  
fi
echo "Output folder: $output_folder"
mkdir -p "$output_folder"

# Determine video args based on the presence of the --show_video flag
if [[ "$@" == *"--show_video"* ]]; then
    video_args="--skip_video_output --show_dvs_model_state all"
else
    video_args=""
fi

# Compose the event stream name
event_stream_=$(basename $input_folder)"_"$framerate_input"fps_"$event_thr"thr_"$res_w"x"$res_h
event_stream_="${event_stream_// /_}" # Replace spaces with underscores
event_stream_="${event_stream_//./p}" # Replace dots with underscores
event_stream_="${event_stream_//:/_}" # Replace colons with underscores
event_stream_="${event_stream_//,/}" # Remove commas

event_stream_filename="${event_stream_}.txt"
echo "Saving event stream with filename: $event_stream_filename"

### Call python main script
python "$v2e_location" -i $input_folder \
        --save_dvs_model_state \
        --vid_orig None \
        --crop '0, 0, 0, 0' \
        --dvs_exposure count 2000 \
        --input_frame_rate $framerate_input \
        --auto_timestamp $auto_timestamp $timestamp_resolution \
        --no_preview \
        --pos_thres $event_thr \
        --neg_thres $event_thr \
        --sigma_thres $sigma_thr \
        --photoreceptor_noise \
        --dvs_text $event_stream_filename \
        --output_folder $output_folder $overwrite_output_folder_option \
        --batch_size $batch_size_slomo \
        --slomo_stats_plot \
        $video_args $cut_off_frequency $output_resolution_args
        #--ignore-gooey 
