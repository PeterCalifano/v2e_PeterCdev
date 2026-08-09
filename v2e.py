#!/usr/bin/env python
"""
Python code for extracting frames from video file and synthesizing fake DVS
events from this video after SuperSloMo has generated interpolated
frames from the original video frames.

@author: Tobi Delbruck, Yuhuang Hu, Zhe He
@contact: tobi@ini.uzh.ch, yuhuang.hu@ini.uzh.ch, zhehe@student.ethz.ch
"""
# todo refractory period for pixel

# TODO review code, comment, and dissection!

import glob
import argparse
import importlib
import sys
from pathlib import Path

import argcomplete
import cv2
import numpy as np
import os
from tempfile import TemporaryDirectory
from engineering_notation import EngNumber as eng  # only from pip
from tqdm import tqdm

import torch

import v2ecore.desktop as desktop
from v2ecore.base_synthetic_input import base_synthetic_input
from v2ecore.v2e_utils import all_images, read_image, \
    check_lowpass, v2e_quit
from v2ecore.v2e_utils import set_output_dimension
from v2ecore.v2e_utils import set_output_folder
from v2ecore.v2e_utils import ImageFolderReader
from v2ecore.v2e_args import v2e_args, write_args_info, SmartFormatter
from v2ecore.v2e_args import v2e_check_dvs_exposure_args
from v2ecore.v2e_args import NO_SLOWDOWN
from v2ecore.renderer import EventRenderer, ExposureMode
from v2ecore.slomo import SuperSloMo
from v2ecore.emulator import EventEmulator
from v2ecore.v2e_utils import inputVideoFileDialog
import logging
import time
from typing import Optional, Any

logging.basicConfig()
root = logging.getLogger()
LOGGING_LEVEL = logging.INFO

root.setLevel(LOGGING_LEVEL)  # todo move to info for production
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName(
    logging.DEBUG, "\033[1;36m%s\033[1;0m" % logging.getLevelName(
        logging.DEBUG))  # cyan foreground
logging.addLevelName(
    logging.INFO, "\033[1;34m%s\033[1;0m" % logging.getLevelName(
        logging.INFO))  # blue foreground
logging.addLevelName(
    logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(
        logging.WARNING))  # red foreground
logging.addLevelName(
    logging.ERROR, "\033[38;5;9m%s\033[1;0m" % logging.getLevelName(
        logging.ERROR))  # red background
logger = logging.getLogger(__name__)

# torch device
torch_device: str = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu').type
logger.info(f'torch device is {torch_device}')
if torch_device == 'cpu':
    logger.warning('CUDA GPU acceleration of pytorch operations is not available; '
                   'see https://pytorch.org/get-started/locally/ '
                   'to generate the correct conda install command to enable GPU-accelerated CUDA.')

# may only apply to windows
try:
    #  from scripts.regsetup import description
    from gooey import Gooey  # pip install Gooey
except Exception as e:
    logger.info(f"{e}: Gooey GUI builder not available, "
                f"will use command line arguments.\n"
                f"Install with 'pip install Gooey if you want a no-arg GUI to invoke v2e'. See README")


def get_args():
    """ proceses input arguments
    :returns: (args_namespace,other_args,command_line) """
    parser = argparse.ArgumentParser(
        description='v2e: generate simulated DVS events from video.',
        epilog='Run with no --input to open file dialog', allow_abbrev=True,
        formatter_class=SmartFormatter)

    parser = v2e_args(parser)

    #  parser.add_argument(
    #      "--rotate180", type=bool, default=False,
    #      help="rotate all output 180 deg.")
    # https://kislyuk.github.io/argcomplete/#global-completion
    # Shellcode (only necessary if global completion is not activated -
    # see Global completion below), to be put in e.g. .bashrc:
    # eval "$(register-python-argcomplete v2e.py)"
    argcomplete.autocomplete(parser)

    # change to known arguments so that synthetic input module can take arguments
    (args_namespace, other_args) = parser.parse_known_args()
    command_line = ''
    for a in sys.argv:
        command_line = command_line+' '+a
    return (args_namespace, other_args, command_line)


def resolve_dvs_emulator_seed(requested_seed: int) -> tuple[int, bool]:
    """Return effective seed and whether it was auto-generated."""
    if requested_seed > 0:
        return requested_seed, False
    if requested_seed == 0:
        auto_seed = int(np.random.randint(1, 2**31))
        return auto_seed, True
    raise ValueError(f'dvs_emulator_seed must be >=0, got {requested_seed}')


def resolve_iebcs_noise_paths(
        noise_source: str,
        noise_preset: str,
        noise_pos_path: str | None,
        noise_neg_path: str | None) -> tuple[str, str]:
    """Resolve IEBCS histogram-noise file paths from CLI configuration."""
    if noise_source == "files":
        if noise_pos_path is None or noise_neg_path is None:
            raise ValueError(
                '--iebcs_hist_noise_model with --iebcs_noise_source=files '
                'requires both --iebcs_noise_pos_path and --iebcs_noise_neg_path')
        return noise_pos_path, noise_neg_path

    preset_files = {
        "3klux": ("noise_pos_3klux.npy", "noise_neg_3klux.npy"),
        "161lux": ("noise_pos_161lux.npy", "noise_neg_161lux.npy"),
        "0.1lux": ("noise_pos_0.1lux.npy", "noise_neg_0.1lux.npy"),
    }
    if noise_preset not in preset_files:
        raise ValueError(f'unsupported IEBCS noise preset "{noise_preset}"')
    pos_name, neg_name = preset_files[noise_preset]
    base_dir = Path(__file__).resolve().parent / "input" / "iebcs_noise"
    return str(base_dir / pos_name), str(base_dir / neg_name)


def main() -> None: #--check-untyped-defs

    # %% Arguments and options handling

    # Try to start GUI if requested
    try:
        ga = Gooey(get_args, program_name="v2e", default_size=(575, 600))
        # logger.info("Use --ignore-gooey to disable GUI and run with command line arguments") # DEVNOTE there is no such an input argument...
        ga()  # type: ignore
    except Exception as e:
        logger.info(
            f'{e}: Gooey package GUI not available, using command line arguments. \n'
            f'You can try to install with "pip install Gooey"')

    # Get parsed args from shell
    (args, other_args, command_line) = get_args()

    # Set input file path
    input_file: str | None = args.input
    synthetic_input: str | None = args.synthetic_input
    input_filepath: str | None = None

    if synthetic_input is not None and input_file is not None:
        logger.error(
            f'Both input_filepath {input_file} and synthetic_input {synthetic_input} are specified - you can only specify one of them')
        v2e_quit(1)

    if synthetic_input is not None and args.output_in_place:
        logger.error(
            '--output_in_place cannot be used with --synthetic_input because there is no source file or folder path')
        v2e_quit(1)

    if synthetic_input is None and input_file is None:
        try:
            input_filepath = inputVideoFileDialog()
            if input_filepath is None:
                logger.info('no file selected, quitting')
                v2e_quit()
        except Exception as e:
            logger.error(
                f'no input file specified and cannot show input file dialog; are you running without graphical display? ({e})')
            v2e_quit(1)
    elif input_file is not None:
        input_filepath = input_file

    # Set output folder
    output_folder = set_output_folder(
        args.output_folder,
        input_filepath,
        args.unique_output_folder if not args.overwrite else False,
        args.overwrite,
        args.output_in_place,
        logger)

    # Set output width and height based on the arguments
    output_width, output_height = set_output_dimension(
        args.output_width, args.output_height,
        args.dvs128, args.dvs240, args.dvs346,
        args.dvs640, args.dvs1024, args.dvxplorer,
        logger)

    # Visualization
    avi_frame_rate = args.avi_frame_rate
    dvs_vid = args.dvs_vid if not args.skip_video_output else None
    dvs_vid_full_scale = args.dvs_vid_full_scale
    vid_orig = args.vid_orig if not args.skip_video_output else None
    vid_slomo = args.vid_slomo if not args.skip_video_output else None
    preview = not args.no_preview

    # Setup synthetic input classes and method
    synthetic_input_module = None
    synthetic_input_class = None
    synthetic_input_instance: base_synthetic_input | None = None
    synthetic_input_next_frame_method = None

    if synthetic_input is not None:
        try:
            synthetic_input_module = importlib.import_module(synthetic_input)
            if '.' in synthetic_input:
                classname = synthetic_input[synthetic_input.rindex('.')+1:]
            else:
                classname = synthetic_input

            synthetic_input_module = importlib.import_module(synthetic_input)
            synthetic_input_class: synthetic_input = getattr(
                synthetic_input_module, classname)

            vid_path = os.path.join(
                output_folder, vid_orig) if not vid_orig is None else None

            # TODO output folder might not be unique, could write to first output folder
            synthetic_input_instance: base_synthetic_input = synthetic_input_class(
                width=output_width, height=output_height, preview=not args.no_preview, arg_list=other_args, avi_path=vid_path, parent_args=args)

            if not isinstance(synthetic_input_instance, base_synthetic_input):
                logger.error(f'synthetic input instance of {synthetic_input} is of type {type(synthetic_input_instance)}, but it should be a sublass of synthetic_input;'
                             f'there is no guarentee that it implements the necessary methods')
            synthetic_input_next_frame_method = getattr(
                synthetic_input_class, 'next_frame')
            synthetic_input_total_frames_method = getattr(
                synthetic_input_class, 'total_frames')

            srcNumFramesToBeProccessed = synthetic_input_instance.total_frames()

            logger.info(
                f'successfully instanced {synthetic_input_instance} with method {synthetic_input_next_frame_method}:'
                '{synthetic_input_module.__doc__}')

        except ModuleNotFoundError as e:
            logger.error(f'Could not import {synthetic_input}: {e}')
            v2e_quit(1)
        except AttributeError as e:
            logger.error(f'{synthetic_input} method incorrect?: {e}')
            v2e_quit(1)

    # Writing the info file
    infofile = write_args_info(args, output_folder, other_args, command_line)

    fh = logging.FileHandler(infofile, mode='a')
    fh.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    num_frames = 0
    srcNumFramesToBeProccessed = 0
    srcDurationToBeProcessed = float("NaN")

   # input file checking
    #  if (not input_filepath or not os.path.isfile(input_filepath)
    #      or not os.path.isdir(input_filepath)) \
    #          and not base_synthetic_input:
    if (not synthetic_input):
        if not os.path.isfile(input_filepath) and not os.path.isdir(input_filepath):

            logger.error('input file {} does not exist'.format(input_filepath))
            v2e_quit(1)

        if os.path.isdir(input_filepath):
            if len(os.listdir(input_filepath)) == 0:
                logger.error(f'input folder {input_filepath} is empty')
                v2e_quit(1)

    # define video parameters
    # the input start and stop time, may be round to actual
    # frame timestamp
    input_start_time = args.start_time
    input_stop_time = args.stop_time

    def is_float(element: Any) -> bool:
        try:
            float(element)
            return True
        except ValueError:
            return False

    if not input_start_time is None and not input_stop_time is None and is_float(input_start_time) and is_float(input_stop_time) and input_stop_time <= input_start_time:
        logger.error(
            f'stop time {input_stop_time} must be later than start time {input_start_time}')
        v2e_quit(1)

    input_slowmotion_factor: float = args.input_slowmotion_factor
    input_frame_rate: float | None = args.input_frame_rate
    timestamp_resolution: float | None = args.timestamp_resolution
    auto_timestamp_resolution: bool = args.auto_timestamp_resolution
    disable_slomo: bool = args.disable_slomo
    
    slomo = None  # make it later on

    if not disable_slomo and auto_timestamp_resolution is False \
            and timestamp_resolution is None:
        logger.error(
            'if --auto_timestamp_resolution=False, '
            'then --timestamp_resolution must be set to '
            'some desired DVS event timestamp resolution in seconds, '
            'e.g. 0.01')
        v2e_quit()

    if auto_timestamp_resolution is True \
            and timestamp_resolution is not None:
        logger.info(
            f'auto_timestamp_resolution=True and '
            f'timestamp_resolution={timestamp_resolution}: '
            f'Limiting automatic upsampling to maximum timestamp interval.')

    # DVS pixel thresholds
    pos_thres = args.pos_thres
    neg_thres = args.neg_thres
    sigma_thres = args.sigma_thres
    record_single_pixel_states = args.record_single_pixel_states

    # Cutoff and noise frequencies
    cutoff_hz = args.cutoff_hz
    leak_rate_hz = args.leak_rate_hz
    if leak_rate_hz > 0 and sigma_thres == 0:
        logger.warning(
            'leak_rate_hz > 0 but sigma_thres == 0. All leak events will be synchronous.')
    shot_noise_rate_hz = args.shot_noise_rate_hz
    iebcs_latency_jitter_model: bool = args.iebcs_latency_jitter_model
    iebcs_latency_mean_us: float = args.iebcs_latency_mean_us
    iebcs_latency_jitter_us: float = args.iebcs_latency_jitter_us
    iebcs_resample_thresholds_on_event: bool = args.iebcs_resample_thresholds_on_event
    iebcs_contrast_latency_model: bool = args.iebcs_contrast_latency_model
    iebcs_latency_tau_us: float = args.iebcs_latency_tau_us
    iebcs_latency_clamp_us: float = args.iebcs_latency_clamp_us
    iebcs_latency_slope_jitter: bool = args.iebcs_latency_slope_jitter
    iebcs_hist_noise_model: bool = args.iebcs_hist_noise_model
    iebcs_noise_source: str = args.iebcs_noise_source
    iebcs_noise_preset: str = args.iebcs_noise_preset
    iebcs_noise_pos_path: str | None = args.iebcs_noise_pos_path
    iebcs_noise_neg_path: str | None = args.iebcs_noise_neg_path
    iebcs_refractory_state_coupling: bool = args.iebcs_refractory_state_coupling
    iebcs_refractory_us: float | None = args.iebcs_refractory_us
    v2ce_nonuniform_burst_timestamps: bool = args.v2ce_nonuniform_burst_timestamps
    v2ce_burst_timestamps_mode: str = args.v2ce_burst_timestamps_mode

    if shot_noise_rate_hz < 0:
        logger.error('shot_noise_rate_hz must be non-negative')
        v2e_quit(1)
    if iebcs_latency_mean_us < 0:
        logger.error('iebcs_latency_mean_us must be non-negative')
        v2e_quit(1)
    if iebcs_latency_jitter_us < 0:
        logger.error('iebcs_latency_jitter_us must be non-negative')
        v2e_quit(1)
    if iebcs_latency_tau_us < 0:
        logger.error('iebcs_latency_tau_us must be non-negative')
        v2e_quit(1)
    if iebcs_latency_clamp_us < 0:
        logger.error('iebcs_latency_clamp_us must be non-negative')
        v2e_quit(1)
    if iebcs_refractory_us is not None and iebcs_refractory_us < 0:
        logger.error('iebcs_refractory_us must be non-negative')
        v2e_quit(1)

    iebcs_hist_noise_pos_path: str | None = None
    iebcs_hist_noise_neg_path: str | None = None
    if iebcs_hist_noise_model:
        try:
            iebcs_hist_noise_pos_path, iebcs_hist_noise_neg_path = resolve_iebcs_noise_paths(
                noise_source=iebcs_noise_source,
                noise_preset=iebcs_noise_preset,
                noise_pos_path=iebcs_noise_pos_path,
                noise_neg_path=iebcs_noise_neg_path)
        except ValueError as e:
            logger.error(str(e))
            v2e_quit(1)
        assert iebcs_hist_noise_pos_path is not None
        assert iebcs_hist_noise_neg_path is not None
        missing_hist_noise_paths: list[str] = []
        if not os.path.isfile(iebcs_hist_noise_pos_path):
            missing_hist_noise_paths.append(iebcs_hist_noise_pos_path)
        if not os.path.isfile(iebcs_hist_noise_neg_path):
            missing_hist_noise_paths.append(iebcs_hist_noise_neg_path)
        if len(missing_hist_noise_paths) > 0:
            missing_str = ", ".join(missing_hist_noise_paths)
            logger.error(
                'IEBCS histogram noise file(s) not found: %s. '
                'Provide explicit files via '
                '--iebcs_noise_source files --iebcs_noise_pos_path ... --iebcs_noise_neg_path ..., '
                'or ensure preset files exist under input/iebcs_noise/.',
                missing_str)
            v2e_quit(1)
        logger.info(
            f'Using IEBCS histogram noise distributions: ON={iebcs_hist_noise_pos_path}, '
            f'OFF={iebcs_hist_noise_neg_path}')

    # Event saving options
    dvs_h5 = args.dvs_h5
    dvs_aedat2 = args.dvs_aedat2
    dvs_aedat4 = args.dvs_aedat4
    aedat4_camera_name = args.aedat4_camera_name

    if args.dvxplorer and aedat4_camera_name is None:
        aedat4_camera_name = "DVXplorer"

    dvs_text = args.dvs_text
    
    # Signal noise output CSV file
    label_signal_noise = args.label_signal_noise

    if label_signal_noise and dvs_text is None and dvs_aedat2 is None and dvs_aedat4 is None:
        logger.error(
            'if you specify --label_signal_noise you must specify --dvs_text and/or --dvs_aedat2 and/or --dvs_aedat4')
        v2e_quit(1)
    if label_signal_noise and args.photoreceptor_noise:
        logger.error(
            'if you specify --label_signal_noise you cannot use --photoreceptor_noise option')
        v2e_quit(1)
    if label_signal_noise and shot_noise_rate_hz == 0:
        logger.error(
            'You specified --label_signal_noise, but --shot_noise_rate=0 and there will be no noise events')
        v2e_quit(1)

    # Debug feature: if show slomo stats
    slomo_stats_plot = args.slomo_stats_plot
    #  rotate180 = args.rotate180  # never used, consider removing
    batch_size = args.batch_size

    # DVS exposure
    exposure_mode, exposure_val, area_dimension = \
        v2e_check_dvs_exposure_args(args)

    if exposure_mode == ExposureMode.DURATION:
        if exposure_val is None:
            logger.error('exposure_value must be set for duration mode')
            v2e_quit(1)
        else:
            dvsFps = 1.0 / exposure_val

    time_run_started = time.time()

    slomoTimestampResolutionS = None

    if synthetic_input is None:
        logger.info("Opening video input file " + input_filepath)

        if os.path.isdir(input_filepath):
            if input_frame_rate is None:
                logger.error(
                    "When the video is presented as a folder, "
                    "The user must set --input_frame_rate manually")
                v2e_quit(1)

            input_source = ImageFolderReader(
                input_filepath, args.input_frame_rate)
            src_fps = input_source.frame_rate
            src_num_frames = input_source.num_frames

        else:
            input_source = cv2.VideoCapture(input_filepath)
            src_fps = input_source.get(cv2.CAP_PROP_FPS)
            src_num_frames = int(input_source.get(cv2.CAP_PROP_FRAME_COUNT))
            if input_frame_rate is not None:
                logger.info(
                    f'Input video frame rate {src_fps}Hz is overridden by command line argument --input_frame_rate={args.input_frame_rate}')
                src_fps = args.input_frame_rate

        if input_source is not None:

            # Set the output width and height from first image in folder, but only if they were not already set
            set_size = False

            if output_height is None and hasattr(input_source, 'frame_height'):
                set_size = True
                output_height = input_source.frame_height

            if output_width is None and hasattr(input_source, 'frame_width'):
                set_size = True
                output_width = input_source.frame_width

            if set_size:
                logger.warning(
                    f'Auto-selected DVS output size from input frame: '
                    f'{output_width}x{output_height}. '
                    f'Use camera-size options if this is not desired.')

            elif output_height is None or output_width is None:
                logger.error(
                    'Could not read video frame size from video input and so could not automatically set DVS output size. \nCheck DVS camera sizes arguments.')
                v2e_quit(1)

        assert output_height is not None, "output_height should have been set by now, check code logic"
        assert output_width is not None, "output_width should have been set by now, check code logic"

        # Check output height and width
        if output_height > 1024 or output_width > 1024:
            logger.warning(
                'Output height or width greater than 1024 pixels. '
                'Rescaling to max 1024 while maintaining aspect ratio...')

            # Compute aspect ratio
            aspect_ratio = float(output_width) / float(output_height)
            logger.info(f'Aspect ratio of input source: {aspect_ratio}')

            # Scale max to 1024, scale the other dimension to maintain aspect ratio
            if output_width > output_height:
                # Clamp width size and scale height
                output_width = 1024
                output_height = int(output_width / aspect_ratio)

            elif output_height >= output_width:
                # Clamp height size and scale width
                output_height = 1024
                output_width = int(output_height * aspect_ratio)

            logger.info(
                f'Rescaled output size: {output_width} x {output_height}')

        # Check frame rate and number of frames
        if src_fps == 0:
            logger.error(
                'source {} fps is 0; v2e needs to have a timescale '
                'for input video'.format(input_filepath))
            v2e_quit()

        if src_num_frames < 2:
            logger.warning(
                'num frames is less than 2, probably cannot be determined '
                'from cv2.CAP_PROP_FRAME_COUNT')

        srcTotalDuration = (src_num_frames-1)/src_fps
        # the index of the frames, from 0 to src_num_frames-1
        start_frame = int(src_num_frames*(input_start_time/srcTotalDuration)) \
            if input_start_time else 0
        stop_frame = int(src_num_frames*(input_stop_time/srcTotalDuration)) \
            if input_stop_time else src_num_frames-1
        srcNumFramesToBeProccessed = stop_frame-start_frame+1
        # the duration to be processed, should subtract 1 frame when
        # calculating duration
        srcDurationToBeProcessed = (srcNumFramesToBeProccessed-1)/src_fps

        # redefining start and end time using the time calculated
        # from the frames, the minimum resolution there is
        start_time = start_frame/src_fps
        stop_time = stop_frame/src_fps

        srcFrameIntervalS = (1./src_fps)/input_slowmotion_factor

        slowdown_factor = NO_SLOWDOWN  # start with factor 1 for upsampling

        if disable_slomo:
            logger.warning(
                'slomo interpolation disabled by command line option; '
                'output DVS timestamps will have source frame interval '
                'resolution')
            # time stamp resolution equals to source frame interval
            slomoTimestampResolutionS = srcFrameIntervalS

        elif not auto_timestamp_resolution:
            slowdown_factor = int(
                np.ceil(srcFrameIntervalS/timestamp_resolution))
            if slowdown_factor < NO_SLOWDOWN:
                slowdown_factor = NO_SLOWDOWN
                logger.warning(
                    'timestamp resolution={}s is >= source '
                    'frame interval={}s, will not upsample'
                    .format(timestamp_resolution, srcFrameIntervalS))
            elif slowdown_factor > 100 and cutoff_hz == 0:
                logger.warning(
                    f'slowdown_factor={slowdown_factor} is >100 but '
                    'cutoff_hz={cutoff_hz}. We have observed that '
                    'numerical errors in SuperSloMo can cause noise '
                    'that makes fake events at the upsampling rate. '
                    'Recommend to set physical cutoff_hz, '
                    'e.g. --cutoff_hz=200 (or leave the default cutoff_hz)')
            slomoTimestampResolutionS = srcFrameIntervalS/slowdown_factor

            logger.info(
                f'--auto_timestamp_resolution is False, '
                f'src_fps={src_fps}Hz '
                f'input_slowmotion_factor={input_slowmotion_factor}, '
                f'real src FPS={src_fps*input_slowmotion_factor}Hz, '
                f'srcFrameIntervalS={eng(srcFrameIntervalS)}s, '
                f'timestamp_resolution={eng(timestamp_resolution)}s, '
                f'so SuperSloMo will use slowdown_factor={slowdown_factor} '
                f'and have '
                f'slomoTimestampResolutionS={eng(slomoTimestampResolutionS)}s')

            if slomoTimestampResolutionS > timestamp_resolution:
                logger.warning(
                    'Upsampled src frame intervals of {}s is larger than\n '
                    'the desired DVS timestamp resolution of {}s'
                    .format(slomoTimestampResolutionS, timestamp_resolution))

            check_lowpass(cutoff_hz, 1/slomoTimestampResolutionS, logger)

        else:  # auto_timestamp_resolution
            if timestamp_resolution is not None:
                slowdown_factor = int(
                    np.ceil(srcFrameIntervalS/timestamp_resolution))

                logger.info(
                    f'--auto_timestamp_resolution=True and '
                    f'timestamp_resolution={eng(timestamp_resolution)}s: '
                    f'source video will be automatically upsampled but '
                    f'with at least upsampling factor of {slowdown_factor}')
            else:
                logger.info(
                    '--auto_timestamp_resolution=True and '
                    'timestamp_resolution is not set: '
                    'source video will be automatically upsampled to '
                    'limit maximum interframe motion to 1 pixel')

        # Set SloMo model, set no SloMo model if no slowdown
        if not disable_slomo and (auto_timestamp_resolution or slowdown_factor != NO_SLOWDOWN):

            slomo = SuperSloMo(model=args.slomo_model,
                               auto_upsample=auto_timestamp_resolution,
                               upsampling_factor=slowdown_factor,
                               video_path=None if args.skip_video_output else output_folder,
                               vid_orig=None if args.skip_video_output else vid_orig,
                               vid_slomo=None if args.skip_video_output else vid_slomo,
                               preview=preview, batch_size=batch_size)

    if not synthetic_input and not auto_timestamp_resolution:
        logger.info(
            f'Effective event timestamp resolution: {eng(slomoTimestampResolutionS)}s')
        if exposure_mode == ExposureMode.DURATION \
                and dvsFps > (1 / slomoTimestampResolutionS):
            logger.warning(
                'DVS video frame rate={}Hz is larger than '
                'the effective DVS frame rate of {}Hz; '
                'DVS video will have blank frames'.format(
                    dvsFps, (1 / slomoTimestampResolutionS)))

    # %% PROCESSING
    if not synthetic_input:
        logger.info(
            'Source summary: path="{}", total_frames={}, total_duration={}s, '
            'src_fps={}Hz, input_slowmotion_factor={}, frame_interval={}s, '
            'processing_frames={} ({}..{}), time_span={}s..{}s (duration {}s)'
            .format(input_filepath, src_num_frames, eng(srcTotalDuration),
                    eng(src_fps), eng(input_slowmotion_factor),
                    eng(srcFrameIntervalS),
                    stop_frame-start_frame+1, start_frame, stop_frame,
                    start_time, stop_time, (stop_time-start_time)))

        if exposure_mode == ExposureMode.DURATION:
            dvsNumFrames = np.floor(
                dvsFps*srcDurationToBeProcessed/input_slowmotion_factor)
            dvsDuration = dvsNumFrames/dvsFps
            dvsPlaybackDuration = dvsNumFrames/avi_frame_rate
            start_time = start_frame / src_fps
            # todo something replicated here, already have start and stop times
            stop_time = stop_frame / src_fps

            logger.info('DVS frame summary: mode=duration, fps={}, accumulation={}s, '
                        'num_frames={}, dvs_duration={}s, playback_duration={}s'
                        .format(eng(dvsFps), eng(1 / dvsFps),
                                dvsNumFrames, eng(dvsDuration),
                                eng(dvsPlaybackDuration)))
        elif exposure_mode == ExposureMode.SOURCE:
            logger.info(
                f'DVS frame summary: mode=source, fps={eng(src_fps)}Hz, accumulation={eng(srcFrameIntervalS)}s')
        else:
            logger.info(
                'DVS frame summary: mode=count, events_per_frame={}'
                .format(exposure_val))

    # Check one more time that we have an output width and height
    if output_width is None or output_height is None:
        logger.error("Either or both of output_width or output_height is None,\n"
                     "which means that they were not specified or could not be inferred from the input video. \n "
                     "Please see options for DVS camera sizes. \nYou can try the option --dvs346 for DAVIS346 camera as one well-supported option.")
        v2e_quit(1)
    num_pixels = output_width*output_height

    hdr = args.hdr
    if hdr:
        if args.hdr_disable_prepro:
            logger.info('Treating input as preprocessed HDR logarithmic input (preprocessing disabled)')
        else:
            logger.info('Treating input as HDR floating-point gray scale in [0,1] with internal preprocessing')

    scidvs: bool = args.scidvs
    if scidvs:
        logger.info('Simulating SCIDVS pixel')

    try:
        effective_dvs_seed, auto_seeded = resolve_dvs_emulator_seed(
            args.dvs_emulator_seed)
        effective_dvs_seed, auto_seeded = resolve_dvs_emulator_seed(
            args.dvs_emulator_seed)
    except ValueError as e:
        logger.error(str(e))
        v2e_quit(1)
        return

    if auto_seeded:
        logger.info(
            f'Using DVS emulator seed: {effective_dvs_seed} (auto-generated)')
    else:
        logger.info(f'Using DVS emulator seed: {effective_dvs_seed}')

    # Setup DVS emulator
    emulator = EventEmulator(
        pos_thres=pos_thres, neg_thres=neg_thres,
        sigma_thres=sigma_thres, cutoff_hz=cutoff_hz,
        strict_model_validity=args.strict_model_validity,
        leak_rate_hz=leak_rate_hz, shot_noise_rate_hz=shot_noise_rate_hz, photoreceptor_noise=args.photoreceptor_noise,
        leak_jitter_fraction=args.leak_jitter_fraction,
        noise_rate_cov_decades=args.noise_rate_cov_decades,
        refractory_period_s=args.refractory_period,
        iebcs_latency_jitter_model=iebcs_latency_jitter_model,
        iebcs_latency_mean_us=iebcs_latency_mean_us,
        iebcs_latency_jitter_us=iebcs_latency_jitter_us,
        iebcs_resample_thresholds_on_event=iebcs_resample_thresholds_on_event,
        iebcs_contrast_latency_model=iebcs_contrast_latency_model,
        iebcs_latency_tau_us=iebcs_latency_tau_us,
        iebcs_latency_clamp_us=iebcs_latency_clamp_us,
        iebcs_latency_slope_jitter=iebcs_latency_slope_jitter,
        iebcs_hist_noise_model=iebcs_hist_noise_model,
        iebcs_hist_noise_pos_path=iebcs_hist_noise_pos_path,
        iebcs_hist_noise_neg_path=iebcs_hist_noise_neg_path,
        iebcs_refractory_state_coupling=iebcs_refractory_state_coupling,
        iebcs_refractory_us=iebcs_refractory_us,
        v2ce_nonuniform_burst_timestamps=v2ce_nonuniform_burst_timestamps,
        v2ce_burst_timestamps_mode=v2ce_burst_timestamps_mode,
        seed=effective_dvs_seed,
        hdr_disable_prepro=args.hdr_disable_prepro,
        output_folder=output_folder, dvs_h5=dvs_h5, dvs_aedat2=dvs_aedat2, dvs_aedat4=dvs_aedat4,
        aedat4_camera_name=aedat4_camera_name,
        dvs_text=dvs_text, show_dvs_model_state=args.show_dvs_model_state,
        save_dvs_model_state=args.save_dvs_model_state,
        output_width=output_width, output_height=output_height,
        device=torch_device,
        cs_lambda_pixels=args.cs_lambda_pixels, cs_tau_p_ms=args.cs_tau_p_ms,
        hdr=hdr,
        scidvs=scidvs,
        record_single_pixel_states=record_single_pixel_states,
        label_signal_noise=label_signal_noise
    )

    if args.dvs_params is not None:
        logger.warning(
            f'--dvs_param={args.dvs_params} option overrides your '
            f'selected options for threshold, threshold-mismatch, '
            f'leak and shot noise rates')
        emulator.set_dvs_params(args.dvs_params)

    # Setup event renderer
    eventRenderer = EventRenderer(output_path=output_folder,
                                  dvs_vid=dvs_vid, preview=preview, full_scale_count=dvs_vid_full_scale,
                                  exposure_mode=exposure_mode,
                                  exposure_value=exposure_val,
                                  area_dimension=area_dimension,
                                  color_mode=args.dvs_vid_color_mode,
                                  avi_frame_rate=args.avi_frame_rate)

    def flush_event_batch(event_chunks: list[np.ndarray]) -> np.ndarray | None:
        """Concatenate buffered event chunks once and clear the buffer."""
        if len(event_chunks) == 0:
            return None
        if len(event_chunks) == 1:
            batched_events = event_chunks[0]
        else:
            batched_events = np.concatenate(event_chunks, axis=0)
        event_chunks.clear()
        return batched_events

    if synthetic_input_next_frame_method is not None:
        # buffer chunks and concatenate once at flush to avoid O(n^2) np.append growth
        event_chunks: list[np.ndarray] = []
        (fr, fr_time) = synthetic_input_instance.next_frame()
        num_frames += 1
        i = 0
        with tqdm(total=synthetic_input_instance.total_frames(),
                  desc='dvs', unit='fr') as pbar:
            with torch.no_grad():
                while fr is not None:
                    newEvents = emulator.generate_events(fr, fr_time)
                    pbar.update(1)
                    i += 1
                    if newEvents is not None and newEvents.shape[0] > 0 \
                            and not args.skip_video_output:
                        event_chunks.append(newEvents)
                        if i % batch_size == 0:
                            batched_events = flush_event_batch(event_chunks)
                            eventRenderer.render_events_to_frames(
                                batched_events, height=output_height,
                                width=output_width)
                    (fr, fr_time) = synthetic_input_instance.next_frame()
                    num_frames += 1
            # process leftover events
            batched_events = flush_event_batch(event_chunks)
            if batched_events is not None and not args.skip_video_output:
                eventRenderer.render_events_to_frames(
                    batched_events, height=output_height,
                    width=output_width)
    else:
        # video file folder or (avi/mp4) file input
        # timestamps of DVS start at zero and end with
        # span of video we processed
        srcVideoRealProcessedDuration = (stop_time-start_time) / \
            input_slowmotion_factor
        num_frames = srcNumFramesToBeProccessed
        inputHeight = None
        inputWidth = None
        inputChannels = None

        if start_frame > 0:
            logger.info('skipping to frame {}'.format(start_frame))
            for i in tqdm(range(start_frame), unit='fr', desc='src'):
                if isinstance(input_source, ImageFolderReader):
                    if i < start_frame-1:
                        ret, _ = input_source.read(skip=True)
                    else:
                        ret, _ = input_source.read()
                else:
                    ret, _ = input_source.read()
                if not ret:
                    raise ValueError(
                        'something wrong, got to end of file before '
                        'reaching start_frame')

        logger.info(
            'processing frames {} to {} from video input'.format(
                start_frame, stop_frame))

        crop_left_pixels = 0
        crop_right_pixels = 0
        crop_top_pixels = 0
        crop_bottom_pixels = 0
        crop_right_slice_end = None
        crop_bottom_slice_end = None

        if args.crop is not None:
            crop_values = args.crop
            if len(crop_values) != 4:
                logger.error(
                    f'--crop must have 4 elements (you specified --crop={args.crop}')
                v2e_quit(1)

            crop_left_pixels = crop_values[0] if crop_values[0] > 0 else 0
            crop_right_pixels = crop_values[1] if crop_values[1] > 0 else 0
            crop_top_pixels = crop_values[2] if crop_values[2] > 0 else 0
            crop_bottom_pixels = crop_values[3] if crop_values[3] > 0 else 0
            
            crop_right_slice_end = -crop_right_pixels if crop_right_pixels > 0 else None
            
            crop_bottom_slice_end = -crop_bottom_pixels if crop_bottom_pixels > 0 else None
            
            logger.info(
                f'cropping video by (left,right,top,bottom)='
                f'({crop_left_pixels},{crop_right_pixels},{crop_top_pixels},{crop_bottom_pixels})')

        with TemporaryDirectory() as source_frames_dir:
            if os.path.isdir(input_filepath):  # folder input
                inputWidth = input_source.frame_width
                inputHeight = input_source.frame_height
                inputChannels = input_source.frame_channels
            else:
                inputWidth = int(input_source.get(cv2.CAP_PROP_FRAME_WIDTH))
                inputHeight = int(input_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
                inputChannels = 1 if int(input_source.get(cv2.CAP_PROP_MONOCHROME)) \
                    else 3
            logger.info(
                'Input video {} has W={} x H={} frames each with {} channels'
                .format(input_filepath, inputWidth, inputHeight, inputChannels))

            if (output_width is None) and (output_height is None):
                output_width = inputWidth
                output_height = inputHeight
                logger.warning(
                    'Output size auto-set to input size {}x{}; '
                    'this may be slow. Consider --output_width=346 --output_height=260 '
                    'for DAVIS346-like runs.'
                    .format(output_width, output_height))

                # set emulator output width and height for the last time
                emulator.output_width = output_width
                emulator.output_height = output_height

            logger.info(
                f'*** Stage 1/3: '
                f'Resizing {srcNumFramesToBeProccessed} input frames '
                f'to output size '
                f'(with possible RGB to luma conversion)')

            for inputFrameIndex in tqdm(
                    range(srcNumFramesToBeProccessed),
                    desc='rgb2luma', unit='fr'):
                # read frame
                ret, inputVideoFrame = input_source.read()
                num_frames += 1
                if ret == False:
                    logger.warning(
                        f'could not read frame {inputFrameIndex} from {input_source}')
                    continue
                if inputVideoFrame is None or np.shape(inputVideoFrame) == ():
                    logger.warning(
                        f'empty video frame number {inputFrameIndex} in {input_source}')
                    continue
                if not ret or inputFrameIndex + start_frame > stop_frame:
                    break

                if args.crop is not None:
                    # crop the frame, indices are y,x, UL is 0,0
                    if crop_left_pixels + crop_right_pixels >= inputWidth:
                        logger.error(
                            f'left crop {crop_left_pixels} + right crop '
                            f'{crop_right_pixels} is larger than image width {inputWidth}')
                        v2e_quit(1)
                    if crop_top_pixels + crop_bottom_pixels >= inputHeight:
                        logger.error(
                            f'top crop {crop_top_pixels} + bottom crop '
                            f'{crop_bottom_pixels} is larger than image height {inputHeight}')
                        v2e_quit(1)

                    inputVideoFrame = inputVideoFrame[
                        crop_top_pixels:crop_bottom_slice_end,
                        crop_left_pixels:crop_right_slice_end
                    ]  # https://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python

                if output_height and output_width and \
                        (inputHeight != output_height or
                         inputWidth != output_width):
                    dim = (output_width, output_height)
                    (fx, fy) = (float(output_width) / inputWidth,
                                float(output_height) / inputHeight)
                    inputVideoFrame = cv2.resize(
                        src=inputVideoFrame, dsize=dim, fx=fx, fy=fy,
                        interpolation=cv2.INTER_AREA)

                if inputChannels == 3:  # color
                    if inputFrameIndex == 0:  # print info once
                        logger.info(
                            '\nConverting input frames from RGB color to luma')
                    # TODO would break resize if input is gray frames
                    # convert RGB frame into luminance.
                    inputVideoFrame = cv2.cvtColor(
                        inputVideoFrame, cv2.COLOR_BGR2GRAY)  # much faster

                    # TODO add vid_orig output if not using slomo

                # save frame into numpy records
                save_path = os.path.join(
                    source_frames_dir, str(inputFrameIndex).zfill(8) + ".npy")
                np.save(save_path, inputVideoFrame)
                # print("Writing source frame {}".format(save_path), end="\r")
            input_source.release()

            with TemporaryDirectory() as interpFramesFolder:
                interpTimes = None
                # make input to slomo
                if slomo is not None and (auto_timestamp_resolution or slowdown_factor != NO_SLOWDOWN):
                    # interpolated frames are stored to tmpfolder as
                    # 1.png, 2.png, etc

                    logger.info(
                        f'*** Stage 2/3: SloMo upsampling from '
                        f'{source_frames_dir}')

                    # Call SloMo to interpolate the frames and produce output video
                    interpTimes, avgUpsamplingFactor = slomo.interpolate(source_frame_path=source_frames_dir,
                                                                         output_folder=interpFramesFolder,
                                                                         frame_size=(output_width, output_height))

                    avgTs = srcFrameIntervalS / avgUpsamplingFactor

                    logger.info('SloMo average upsampling factor={:5.2f}; '
                                'average DVS timestamp resolution={}s'.format(avgUpsamplingFactor, eng(avgTs)))

                    # Check for undersampling wrt the photoreceptor lowpass filtering
                    if cutoff_hz > 0:
                        logger.info('Using auto_timestamp_resolution. '
                                    'checking if cutoff hz is ok given '
                                    'sample rate {}'.format(1/avgTs))
                        check_lowpass(cutoff_hz, 1/avgTs, logger)

                    # Read back to memory
                    interpFramesFilenames = all_images(interpFramesFolder)
                    # Number of frames
                    num_frames = len(interpFramesFilenames)

                else:
                    logger.info(
                        f'*** Stage 2/3: using source-rate numpy frames '
                        f'from {source_frames_dir}')
                    interpFramesFilenames = sorted(
                        glob.glob("{}".format(source_frames_dir) + "/*.npy"))
                    num_frames = len(interpFramesFilenames)
                    interpTimes = np.array(range(num_frames))

                # compute times of output integrated frames
                nFrames = len(interpFramesFilenames)
                # interpTimes is in units of 1 per input frame,
                # normalize it to src video time range
                f = srcVideoRealProcessedDuration / \
                    (np.max(interpTimes) - np.min(interpTimes))
                # compute actual times from video times
                interpTimes = f*interpTimes

                # Debug
                if slomo_stats_plot:
                    # FIXME seemingly not working. It does not produce anything
                    from matplotlib import pyplot as plt
                    dt = np.diff(interpTimes)
                    fig = plt.figure()
                    ax1 = fig.add_subplot(111)
                    ax1.set_title(
                        'Slo-Mo frame interval stats (close to continue)')
                    ax1.plot(interpTimes)
                    ax1.plot(interpTimes, 'x')
                    ax1.set_xlabel('Frame')
                    ax1.set_ylabel('Frame timestamp (s)')
                    ax2 = ax1.twinx()
                    ax2.plot(dt * 1e3)
                    plt.show()
                    logger.info('Close plot to continue...')
                    ax2.set_ylabel('Frame interval (ms)')
                    plt.show()

                # Clean-up slomo here, no longer used but still retaining memory
                if slomo is not None:
                    slomo.cleanup()

                    # Delete slomo instance
                    del slomo

                # Buffer chunks and concatenate only when flushed.
                event_chunks: list[np.ndarray] = []

                logger.info(
                    f'*** Stage 3/3: emulating DVS events from '
                    f'{nFrames} frames')

                # Prepare extra steps for data storage before event emulation
                if args.ddd_output:
                    emulator.prepare_storage(nFrames, interpTimes)

                # Generate events from frames and accumulate events to DVS frames for output DVS video
                with tqdm(total=nFrames, desc='dvs', unit='fr') as pbar:
                    with torch.no_grad():
                        # Process each frame
                        for i in range(nFrames):

                            # Read frame
                            frame_path_ = interpFramesFilenames[i]
                            frame_ = np.load(frame_path_, allow_pickle=False) \
                                if frame_path_.endswith(".npy") \
                                else read_image(frame_path_)
                            # Get events
                            newEvents = emulator.generate_events(
                                frame_, interpTimes[i])

                            pbar.update(1)  # Update progress bar

                            if newEvents is not None and newEvents.shape[0] > 0 and not args.skip_video_output:
                                event_chunks.append(newEvents)

                                if i % batch_size == 0:
                                    batched_events = flush_event_batch(
                                        event_chunks)
                                    # Render events to frames if batch size is reached
                                    eventRenderer.render_events_to_frames(
                                        batched_events, height=output_height,
                                        width=output_width)

                    # Process leftover events
                    batched_events = flush_event_batch(event_chunks)
                    if batched_events is not None and not args.skip_video_output:
                        eventRenderer.render_events_to_frames(
                            batched_events, height=output_height,
                            width=output_width)

    # Clean up
    eventRenderer.cleanup()
    emulator.cleanup()

    if synthetic_input_instance is not None:
        synthetic_input_instance.cleanup()

    if num_frames == 0:
        logger.error('no frames read from file')

    totalTime = (time.time()-time_run_started)
    framePerS = num_frames / totalTime
    sPerFrame = totalTime / num_frames if num_frames > 0 else None
    throughputStr = (str(eng(framePerS)) + 'fr/s') \
        if framePerS > 1 else (str(eng(sPerFrame)) + 's/fr')
    timestr = 'done processing {} frames in {}s ({})\n **************** see output folder {}'.format(num_frames,
                                                                                                     eng(
                                                                                                         totalTime),
                                                                                                     throughputStr,
                                                                                                     output_folder)
    logger.info('generated total {} events ({} on, {} off)'
                .format(eng(emulator.num_events_total),
                        eng(emulator.num_events_on),
                        eng(emulator.num_events_off)))
    total_time = emulator.t_previous
    rate_total = emulator.num_events_total / total_time
    rate_on_total = emulator.num_events_on / total_time
    rate_off_total = emulator.num_events_off / total_time
    rate_per_pixel = rate_total/num_pixels
    rate_on_per_pixel = rate_on_total/num_pixels
    rate_off_per_pixel = rate_off_total/num_pixels
    logger.info(
        f'Avg event rate for N={num_pixels} px and total time ={total_time:.3f} s'
        f'\n\tTotal: {eng(rate_total)}Hz ({eng(rate_on_total)}Hz on, {eng(rate_off_total)}Hz off)'
        f'\n\tPer pixel:  {eng(rate_per_pixel)}Hz ({eng(rate_on_per_pixel)}Hz on, {eng(rate_off_per_pixel)}Hz off)')

    if totalTime > 60:
        try:
            from plyer import notification
            logger.info(f'generating desktop notification')
            notification.notify(title='v2e done', message=timestr, timeout=3)
        except Exception as e:
            logger.info(f'could not show notification: {e}')

    # try to show desktop
    # suppress folder opening if it's not necessary
    if not output_folder is None:
        try:
            logger.info(f'showing {output_folder} in desktop')
            desktop.open(os.path.abspath(output_folder))
        except Exception as e:
            logger.warning(
                '{}: could not open {} in desktop'.format(e, output_folder))
    logger.info(timestr)
    return


if __name__ == "__main__":
    main()
    v2e_quit()
