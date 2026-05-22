#!/bin/bash
# Export if running headless 
# TODO verify it works
# export QT_QPA_PLATFORM=offscreen
set -Eeo pipefail

log_info() {
    echo "[frames2events_quick] $*"
}

function usage() {
    echo -e "Usage: $0 required args [optional args] \nValid arguments: \n -i|--input_folder input_folder -f|--framerate_input input_framerate \n [-o|--output output_folder] \n [-t|--threshold event_threshold] \n [-r|--resolution_output width,height] \n [--dvxplorer] \n [-c|--cut_off frequency] \n [-d|--deviation_thr sigma_threshold] \n [-b|--batch_size batch_size] \n [-g|--grid_time_resolution timestamp_resolution] \n [-a|--auto_timestamp]  # expands to v2e --auto_timestamp_resolution \n [-s|--show_video] \n [--disable_slomo] \n [--hdr] \n [--hdr_disable_prepro] \n [-l|--last_time last_time_value] \n"
    exit 1
}

# Parse options using getopt for short and long options
TEMP=$(getopt -o a,i:,f:,o:,t:,b:,r:,c:,d:,g:,h,s,l: --long input_folder:,framerate_input:,output_folder:,threshold:,batch_size:,resolution_output:,cut_off:,deviation_thr:,grid_time_resolution:,auto_timestamp,help,show_video,disable_slomo,last_time:,dvxplorer,hdr,hdr_disable_prepro -n "$0" -- "$@")
if [ $? != 0 ]; then
    usage
fi

disable_slomo_flag=false
show_video_flag=false
output_resolution=""
use_dvxplorer=true
hdr_flag=false
hdr_disable_prepro_flag=false

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
            use_dvxplorer=false
            shift 2
            ;;
        --dvxplorer)
            use_dvxplorer=true
            shift 1
            ;;
        -c|--cut_off)   
            cut_off_frequency="$2"
            shift 2
            ;;
        -s|--show_video)
            show_video_flag=true
            shift 1
            ;;
        --disable_slomo)
            disable_slomo_flag=true
            shift 1
            ;;
        --hdr)
            hdr_flag=true
            shift 1
            ;;
        --hdr_disable_prepro)
            hdr_disable_prepro_flag=true
            shift 1
            ;;
        -h|--help)
            usage
            ;;
        --from_slomo_output)
            from_slomo_output=true
            shift
            ;;
        -l|--last_time)
            last_time="$2"
            shift 2
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
last_time=${last_time:-""}

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
    res_w=640
    res_h=480
fi

# Get the directory of this script
DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" && pwd )"
echo "Using frame2events_quick script in: $DIR"

# Try to find v2e in the PATH
v2e_location=$(which v2e)
echo "Using v2e location: $v2e_location"

if [ -z "$v2e_location" ]; then
    echo "v2e is not in the PATH. Have you installed v2e in the active environment?"
    exit 1
fi

### PRINT INFO
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Error: could not locate conda.sh; ensure conda is installed and available."
    exit 1
fi
# shellcheck source=/dev/null
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate v2e
echo "Using conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

from_slomo_output=${from_slomo_output:-false}
disable_slomo=""
if [ "$disable_slomo_flag" = true ]; then
    disable_slomo="--disable_slomo"
fi

if [ -n "$output_resolution" ]; then
    output_resolution_args="--output_width $res_w --output_height $res_h"
    resolution_summary="${res_w}x${res_h} (explicit)"
else
    if [ "$use_dvxplorer" = true ]; then
        output_resolution_args="--dvxplorer"
        resolution_summary="DVXplorer preset (640x480)"
    else
        output_resolution_args=""
        resolution_summary="native"
    fi
fi

if [ -n "$timestamp_resolution" ]; then
    # Define command 
    timestamp_resolution="--timestamp_resolution $timestamp_resolution"
    timestamp_summary="$timestamp_resolution"
else
    timestamp_resolution=""
    auto_timestamp=true # Override -a option
    timestamp_summary="auto"
fi
if [ -n "$cut_off_frequency" ]; then
    cutoff_summary="${cut_off_frequency} Hz"
    cut_off_frequency="--cutoff_hz $cut_off_frequency"
else
    cutoff_summary="not set"
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
mkdir -p "$output_folder"

# If processing is from slomo output, set the input folder variable to point to a file named "video_slomo.avi" or "slomo.avi"
if [ "$from_slomo_output" = true ]; then
    input_folder=$(dirname "$input_folder")/video_slomo.avi
    if [ ! -f "$input_folder" ]; then
        input_folder=$(dirname "$input_folder")/slomo.avi
    fi
    echo "Found Slomo output file in input folder: $input_folder"
    # Disable slomo
    disable_slomo="--disable_slomo"
    # Set overwrite to true
    overwrite_output_folder_option="--overwrite"
fi

# Determine video args based on the parsed --show_video flag
if [ "$show_video_flag" = true ]; then
    video_args="--show_dvs_model_state all"
    model_state_summary="all"
else
    video_args="--skip_video_output"
    model_state_summary="diff_frame,log_new_frame,lp_log_frame"
fi

# Default: dump diff_frame, log_new_frame, and lp_log_frame model-state videos
model_state_args="--show_dvs_model_state diff_frame log_new_frame lp_log_frame"
if [ "$show_video_flag" = true ]; then
    model_state_args=""
fi

if [ -n "$last_time" ]; then
    last_time_args="--stop_time $last_time"
else
    last_time_args=""
fi

hdr_args=""
if [ "$hdr_flag" = true ]; then
    hdr_args="--hdr"
fi
if [ "$hdr_disable_prepro_flag" = true ]; then
    hdr_args="$hdr_args --hdr_disable_prepro"
fi

# In tmux, force non-video mode for robust headless runs
if [ -n "$TMUX" ] && [ "$show_video_flag" = true ]; then
    video_args="--skip_video_output"
fi

# Compose the event stream name
event_stream_=$(basename $input_folder)"_"$framerate_input"fps_"$event_thr"thr_"$res_w"x"$res_h
event_stream_="${event_stream_// /_}" # Replace spaces with underscores
event_stream_="${event_stream_//./p}" # Replace dots with underscores
event_stream_="${event_stream_//:/_}" # Replace colons with underscores
event_stream_="${event_stream_//,/}" # Remove commas

event_stream_filename="${event_stream_}.aedat4"

log_info "Configuration summary:"
log_info "  input=$input_folder"
log_info "  output_folder=$output_folder"
log_info "  from_slomo_output=$from_slomo_output"
log_info "  input_fps=${framerate_input}Hz, threshold=$event_thr, sigma=$sigma_thr, batch_size=$batch_size_slomo"
log_info "  resolution=$resolution_summary, timestamp=$timestamp_summary, auto_timestamp=$auto_timestamp, cutoff=$cutoff_summary"
log_info "  hdr=${hdr_flag}, hdr_disable_prepro=${hdr_disable_prepro_flag}"
log_info "  model_state_dump=$model_state_summary (saved via --save_dvs_model_state)"
log_info "  output_event_file=$event_stream_filename (AEDAT-4.0)"

### Call python main script
python "$v2e_location" -i $input_folder \
        --save_dvs_model_state \
        --vid_orig None \
        --crop '0, 0, 0, 0' \
        --dvs_exposure duration 0.01 \
        --input_frame_rate $framerate_input \
        --auto_timestamp_resolution $auto_timestamp $timestamp_resolution \
        --pos_thres $event_thr \
        --neg_thres $event_thr \
        --sigma_thres $sigma_thr \
        --photoreceptor_noise \
        --dvs_aedat4 $event_stream_filename \
        --output_folder $output_folder $overwrite_output_folder_option \
        --batch_size $batch_size_slomo \
        $model_state_args $video_args $cut_off_frequency $output_resolution_args $disable_slomo \
        $last_time_args $hdr_args
        #--ignore-gooey
        #--slomo_stats_plot
