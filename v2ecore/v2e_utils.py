import logging
import os
import sys
import tempfile

import numpy as np
import cv2
import glob
import easygui
from tkinter import filedialog
from numba import njit
from engineering_notation import EngNumber as eng
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from collections import deque
import matplotlib
import matplotlib.pyplot as plt

# adjust for different sensor than DAVIS346
DVS_WIDTH, DVS_HEIGHT = 346, 260

# VIDEO_CODEC_FOURCC='RGBA' # uncompressed, >10MB for a few seconds of video

# good codec, basically mp4 with simplest compression, packed in AVI,
# only 15kB for a few seconds
OUTPUT_VIDEO_CODEC_FOURCC = 'XVID'
logger = logging.getLogger(__name__)


class ImageFolderReader(object):
    def __init__(self, image_folder_path, frame_rate):
        """ImageFolderReader.

        This class implements functions that are similar to
        VideoCapture in OpenCV.

        This class is used when the frames are available as
        a sequence of image files in a folder.

        NOTE: the folder should contain only image files and
        the files have to be ordered!!!

        All images are assumed to have the same dimension.
        """
        self.image_folder_path = image_folder_path

        self.image_file_list = sorted(
            glob.glob("{}".format(self.image_folder_path) + "/*.*"))

        self.frame_rate = frame_rate

        self.current_frame_idx = 0

        self.num_frames = len(self.image_file_list)
        if self.num_frames == 0:
            raise FileNotFoundError(
                f'input folder "{self.image_folder_path}" does not contain any image files')

        frame = cv2.imread(self.image_file_list[0])
        if frame is None:
            logger.error(f'could not read a frame from file "{self.image_file_list[0]}" in folder "{self.image_folder_path}"')
            raise FileNotFoundError(f'could not read a frame named {self.image_file_list[0]} from folder {self.image_folder_path}')
        self.frame_height, self.frame_width = frame.shape[0], frame.shape[1]
        self.frame_channels = 1 if frame.ndim < 3 else frame.shape[2]

    def read(self, skip=False):
        """
        Reads the next frame

        :param skip: skip the frame

        :returns: tuple(bool, np.ndarray | None), like cv2.VideoCapture.read()
        """
        if self.current_frame_idx >= self.num_frames:
            return False, None

        if not skip:
            frame = cv2.imread(self.image_file_list[self.current_frame_idx])
        else:
            frame = None
        self.current_frame_idx += 1

        # To match with OpenCV API
        return True, frame

    def release(self):
        """Just to match with OpenCV API."""
        pass

    def __str__(self):
        s=f'ImageFolderReader reading folder {self.image_folder_path} frame number {self.current_frame_idx}'
        try:
            s=s+f' named {self.image_file_list[self.current_frame_idx-1]}'
        except:
            pass
        return s


def v2e_quit(code=0):
    try:
        quit(code)  # not defined in pydev console, e.g. running in pycharm
    finally:
        sys.exit(code)


def make_output_folder(output_folder_base, suffix_counter,overwrite, unique_output_folder) -> str:
    """Makes the output folder if it does not exist yet, or makes unique new numbered folder
    :param output_folder_base: the base name of folder. If it is absolute path, then make folder at absolute location, otherwise relative to startup folder
    :param suffix_counter: a counter value to append
    :param overwrite: to overwrite existing folder
    :param unique_output_folder: set True to make a new uniquely named numbered folder

    :returns: output folder path
    """
    if overwrite and unique_output_folder:
        logger.error(
            "specify one or the other of "
            "--overwrite and --unique_output_folder")
        v2e_quit()

    output_folder = output_folder_base+"-{}".format(suffix_counter) \
        if suffix_counter > 0 else output_folder_base

    non_empty_folder_exists = not overwrite and \
        os.path.exists(output_folder) and os.listdir(output_folder)

    if non_empty_folder_exists and not overwrite and not unique_output_folder:
        logger.error(
            'non-empty output folder {} already exists \n '
            '- use --overwrite or --unique_output_folder'.format(
                os.path.abspath(output_folder)))
        v2e_quit()

    if non_empty_folder_exists and unique_output_folder:
        return make_output_folder(
            output_folder_base, suffix_counter+1,
            overwrite, unique_output_folder)
    else:
        logger.info('using output folder {}'.format(output_folder))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        return output_folder


def set_output_folder(output_folder,
                      input_file: str | None,
                      unique_output_folder,
                      overwrite,
                      output_in_place,
                      logger) -> str:
    """Set output folder in a single function.

    :param output_folder: path to folder, if supplied, otherwise None
    :param input_file: the input file to v2e, used for output_in_place. If folder, this folder is used.
    :param overwrite: set true to overwrite existing files in the folder
    :param output_in_place: set True to output in input_file or input folder folder
    :param logger: logger to report errors and warnings to

    :returns: the output folder path
    """

    if (not output_folder is None) and output_in_place:
        raise ValueError(f'both output_folder={output_folder} and output_in_place={output_in_place} cannot be set true at same time')

    if output_in_place:
        if input_file is None:
            raise ValueError('output_in_place=True requires a real input file or input folder path')
        ip=Path(input_file)
        if ip.is_file():
            output_folder=ip.parent.absolute()
        elif ip.is_dir():
            output_folder=ip.absolute()
        logger.info(f'output_in_place==True so output_folder={output_folder}')
    else:
        output_folder = make_output_folder(
            output_folder, 0, overwrite, unique_output_folder)
        p=Path(output_folder)
        logger.info(
            f'output_in_place==False so made output_folder={p.absolute()}')

    return output_folder


def set_output_dimension(output_width, output_height,
                         dvs128, dvs240, dvs346, dvs640, dvs1024, dvxplorer,
                         logger):
    """Return output_height and output_width based on arguments."""

    if dvs128:
        output_width, output_height = 128, 128
    elif dvs240:
        output_width, output_height = 240, 180
    elif dvs346:
        output_width, output_height = 346, 260
    elif dvs640:
        output_width, output_height = 640, 480
    elif dvs1024:
        output_width, output_height = 1024, 768
    elif dvxplorer:
        output_width, output_height = 640, 480

    if (output_width is None) or (output_height is None):
        logger.warning(
            "Either output_width is None or output_height is None,"
            "or both. Setting both of them to None. \n"
            "Dimension will be set automatically from video input if available. \n"
            "Check DVS camera size arguments.")
        output_width, output_height = None, None

    return output_width, output_height


def check_lowpass(cutoffhz, fs, logger):
    """ checks if cutoffhz is ok given sample rate fs

    """
    if cutoffhz == 0 or fs == 0:
        logger.info('lowpass filter is disabled, no need for check')
        return
    maxeps = 0.3
    tau = 1/(2*np.pi*cutoffhz)
    dt = 1/fs
    eps = dt/tau
    maxdt = tau*maxeps
    maxcutoff = maxeps/(2*np.pi*dt)
    if eps > maxeps:
        logger.warning(
            'Lowpass 3dB cutoff is f_3dB={}Hz (time constant tau={}s) with '
            'sample rate fs={}Hz (sample interval dt={}s) '
            ',\n  but this results in large IIR mixing factor '
            'eps = dt/tau = {:5.3f} > {:4.1f} (maxeps),'
            '\n which means the lowpass will filter  few or even just '
            'last sample, i.e. you will not be lowpassing as expected.'
            '\nWe recommend either'
            '\n -decreasing --timestamp_resolution of DVS events below {}s'
            '\n -decreasing --cutoff_frequency_hz below {}Hz'.format(
                eng(cutoffhz), eng(tau), eng(fs), eng(dt), eps,
                maxeps, eng(maxdt), eng(maxcutoff)))
    else:
        logger.info(
            'Lowpass cutoff is f_3dB={}Hz with tau={}s and '
            'with sample rate fs={}Hz (sample interval dt={}s)'
            ',\nIt has IIR mixing factor eps={:5.3f} which is OK '
            'because it is less than recommended maxeps={:4.1f}'.format(
                eng(cutoffhz), eng(tau), eng(fs), eng(dt), eps, maxeps))


def inputVideoFileDialog():
    return _inputFileDialog(
        [("Video/Data files", ".avi .mp4 .wmv"), ('Any type', '*')])


def inputDDDFileDialog():
    return _inputFileDialog([("DDD recordings", ".hdf5"), ('Any type', '*')])


def _inputFileDialog(types):
    LAST_FILE_NAME_FILE = 'v2e_last_file_chosen.txt'
    fn = os.path.join(tempfile.gettempdir(), LAST_FILE_NAME_FILE)
    default = None
    try:
        with open(fn, 'r') as f:
            default = f.read()
    except FileNotFoundError:
        pass
    filename = easygui.fileopenbox(msg='Select file to convert',
                                   title='v2e input video file',
                                   filetypes=[types],
                                   multiple=False,
                                   default=default
                                   )
    if filename is None:
        logger.info('no file selected, quitting')
        quit(0)
    logger.info(f'selected {filename} with file dialog')
    try:
        with open(fn, 'w') as f:
            f.write(filename)
    except:
        pass
    return filename


def checkAddSuffix(path: str, suffix: str):
    if path.endswith(suffix):
        return path
    else:
        return os.path.splitext(path)[0]+suffix


def video_writer(output_path, height, width,
                 frame_rate=30, fourcc=OUTPUT_VIDEO_CODEC_FOURCC):
    """ Return a video writer.

    Parameters
    ----------
    output_path: str,
        path to store output video.
    height: int,
        height of a frame.
    width: int,
        width of a frame.
    frame_rate: int
        playback frame rate in Hz
    fourcc: cv2.VideoWriter_fourcc
        codec, None results in default XVID
    Returns
    -------
    an instance of cv2.VideoWriter.
    """
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(
                output_path,
                fourcc,
                frame_rate,
                (width, height))
    logger.info(
        'opened {} with {} https://www.fourcc.org/ codec, {}fps, '
        'and ({}x{}) size'.format(
            output_path, OUTPUT_VIDEO_CODEC_FOURCC, frame_rate,
            width, height))
    return out


def all_images(data_path):
    """Return path of all input images. Assume that the ascending order of
    file names is the same as the order of time sequence.

    Parameters
    ----------
    data_path: str
        path of the folder which contains input images.

    Returns
    -------
    List[str]
        sorted in numerical order.
    """
    images = glob.glob(os.path.join(data_path, '*.png'))
    if len(images) == 0:
        raise ValueError(("Input folder is empty or images are not in"
                          " 'png' format."))
    images_sorted = sorted(
        images,
        key=lambda line: int(line.split(os.sep)[-1].split('.')[0]))
    return images_sorted


def read_image(path: str) -> np.ndarray:
    """Read image and returns it as grayscale np.ndarray float scaled 0-255.

    Parameters
    ----------
    path: str
        path of image.

    Returns
    -------
    img: np.ndarray scaled 0-255
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    return img


def read_aedat_txt_events(fname: str):
    """
    reads txt data DVS events
    Parameters
    ----------
    fname:str
        filename
    Returns
    -------
        np.ndarray with each row having ts,x,y,pol
        ts is in seconds
        pol is 0,1
    """
    import pandas as pd
    import numpy as np
    read_table_args = dict(
        sep=' ',  # field separator
        comment='#',  # comment
        skipinitialspace=False,
        skip_blank_lines=True,
        encoding='utf-8',
        names=['t', 'x', 'y', 'p'],
        dtype={'t': np.float64, 'x': np.int32, 'y': np.int32, 'p': np.int32},
    )
    try:
        dat = pd.read_table(fname, on_bad_lines='warn', **read_table_args)
    except TypeError:
        dat = pd.read_table(
            fname,
            error_bad_lines=False,
            warn_bad_lines=True,
            **read_table_args)

    # array[N,4] with each row having ts, x, y, pol.
    # ts is in float seconds. pol is 0,1
    return np.array(dat.values)


def select_events_in_roi(events, x, y):
    """ Select the events inside the region specified by x and y.
    including the x and y values.

    Parameters
    ----------
    events: np.ndarray, [timestamp, x, y, polarity]
    x: int or tuple, x coordinate.
    y: int or tuple, y coordinate.

    Returns
    -------
    np.ndarray, event just in ROI with the same shape as events.
    """
    x_lim = DVS_WIDTH-1  # events[:, 1].max()
    y_lim = DVS_HEIGHT-1  # events[:, 2].max()

    if isinstance(x, int):
        if x < 0 or x > x_lim:
            raise ValueError("x is not in the valid range.")
        x_region = (events[:, 1] == x)

    elif isinstance(x, tuple):
        if x[0] < 0 or x[1] < 0 or \
           x[0] > x_lim or x[1] > x_lim or \
           x[0] > x[1]:
            raise ValueError("x is not in the valid range.")
        x_region = np.logical_and(events[:, 1] >= x[0], events[:, 1] <= x[1])
    else:
        raise TypeError("x must be int or tuple.")

    if isinstance(y, int):
        if y < 0 or y > y_lim:
            raise ValueError("y is not in the valid range.")
        y_region = (events[:, 2] == y)

    elif isinstance(y, tuple):
        if y[0] < 0 or y[1] < 0 or \
           y[0] > y_lim or y[1] > y_lim or \
           y[0] > y[1]:
            raise ValueError("y is not in the valid range.")
        y_region = np.logical_and(events[:, 2] >= y[0], events[:, 2] <= y[1])
    else:
        raise TypeError("y must be int or tuple.")

    region = np.logical_and(x_region, y_region)

    return events[region]


def histogram_events_in_time_bins(
        events, start=0, stop=3.5,
        time_bin_ms=50, polarity=None):
    """ Count the amount of events in each bin.
    Parameters
    ----------
    events: np.ndarray, [timestamp, x, y, polarity].
    start: float, start time in s
    stop: float, end time in s
    polarity: int or None. If int, it must be 1 or -1.

    Returns
    -------
    histogram of counts

    """
    time_bin_s = time_bin_ms*0.001

    if start < 0 or stop < 0:
        raise ValueError("start and stop must be int.")
    if start + time_bin_s > stop:
        raise ValueError("start must be less than (stop - time_bin_s).")
    if polarity and polarity not in [1, -1]:
        raise ValueError("polarity must be 1 or -1.")

    ticks = np.arange(start, stop, time_bin_s)
    bin_num = ticks.shape[0]
    ts_cnt = np.zeros([bin_num - 1, 2])
    for i in range(bin_num - 1):
        condition = np.logical_and(events[:, 0] >= ticks[i],
                                   events[:, 0] < ticks[i + 1])
        if polarity:
            condition = np.logical_and(condition, events[:, 3] == polarity)
        cnt = events[condition].shape[0]
        ts_cnt[i][0] = (ticks[i] + ticks[i + 1]) / 2
        ts_cnt[i][1] = cnt

    return ts_cnt


@njit("float64[:, :](float64[:, :], int64[:], int64[:, :])",
      nogil=True, parallel=False)
def hist2d_numba_seq(tracks, bins, ranges):
    H = np.zeros((bins[0], bins[1]), dtype=np.float64)
    delta = 1/((ranges[:, 1] - ranges[:, 0]) / bins)

    for t in range(tracks.shape[1]):
        i = (tracks[0, t] - ranges[0, 0]) * delta[0]
        j = (tracks[1, t] - ranges[1, 0]) * delta[1]
        if 0 <= i < bins[0] and 0 <= j < bins[1]:
            H[int(i), int(j)] += 1

    return H


class EventOnlineViewer3D:
    def __init__(self,
                 xlim,
                 ylim,
                 t_window_s=0.01,
                 max_events=100_000,
                 view="t"):
        """
        xlim, ylim : tuple
            Fixed spatial limits (never changed).
        t_window : float
            Time window size [s] to keep (sliding window).
        max_events : int
            Max events rendered (visual subsampling only).
        view : {"t", "x", "y"}
            Camera view direction.
        """

        self.max_events = max_events
        self.t_window = t_window_s
        self.view = view

        # Rolling buffer of events
        self.event_buffer = deque()

        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # Apply visual style with black background
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")

        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("white")

        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.zaxis.label.set_color("white")

        self.ax.grid(True, color="white", alpha=0.2)

        # Setup axis of 3D plot
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(0.0, t_window_s)

        self.ax.set_xlabel("x [px]")
        self.ax.set_ylabel("y [px]")
        self.ax.set_zlabel("t [s]")
        self.ax.set_title("Event cloud (x, y, t)", color="white")

        if view == "t":
            self.ax.view_init(elev=90, azim=-90)
        elif view == "x":
            self.ax.view_init(elev=0, azim=0)
        elif view == "y":
            self.ax.view_init(elev=0, azim=90)

        # Image-style view from time axis
        self.ax.invert_yaxis()

        # Create empty scatter once
        self.scatter_on = self.ax.scatter(
            [], [], [],
            c="lime",
            s=1,
            alpha=1.0
        )
        self.scatter_off = self.ax.scatter(
            [], [], [],
            c="red",
            s=1,
            alpha=1.0
        )

        plt.show(block=False)

    def update(self, events):
        """
        events: ndarray [N,4] -> (t, x, y, polarity)
        polarity: +1 / 1 -> ON, 0 / -1 -> OFF
        """

        if events is None or events.size == 0:
            return

        # Append events to buffer
        for ev in events:
            self.event_buffer.append(ev)

        t_now = events[-1, 0]
        t_min = t_now - self.t_window

        # Drop old events
        while self.event_buffer and self.event_buffer[0][0] < t_min:
            self.event_buffer.popleft()

        if not self.event_buffer:
            return

        buf = np.asarray(self.event_buffer)

        # Subsample for visualization
        if buf.shape[0] > self.max_events:
            idx = np.random.choice(
                buf.shape[0], self.max_events, replace=False)
            buf = buf[idx]

        # Split polarity
        t = buf[:, 0] - t_min
        x = buf[:, 1]
        y = buf[:, 2]
        p = buf[:, 3]

        on_mask = p > 0
        off_mask = ~on_mask

        # Update ON events (green)
        self.scatter_on._offsets3d = (
            x[on_mask],
            y[on_mask],
            t[on_mask]
        )

        # Update OFF events (red)
        self.scatter_off._offsets3d = (
            x[off_mask],
            y[off_mask],
            t[off_mask]
        )

        plt.pause(0.001)

    @staticmethod
    def visualize_events_3d(events,
                            max_events=200_000,
                            title="Event cloud (x, y, t)"):
        """
        events: ndarray [N, 4] -> (t, x, y, polarity)
        """

        if events is None or events.size == 0:
            print("No events to visualize")
            return

        # Subsample for speed if needed
        if events.shape[0] > max_events:
            idx = np.random.choice(events.shape[0], max_events, replace=False)
            events = events[idx]

        t = events[:, 0]
        x = events[:, 1]
        y = events[:, 2]
        p = events[:, 3]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(
            x, y, t,
            c=p,
            cmap='coolwarm',
            s=1,
            alpha=0.8
        )

        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        ax.set_zlabel("t [s]")
        ax.set_title(title)

        plt.colorbar(sc, label="polarity")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_xt(events):
        plt.figure(figsize=(7, 4))
        plt.scatter(events[:, 1], events[:, 0],
                    c=events[:, 3], s=1, cmap='coolwarm')
        plt.xlabel("x [px]")
        plt.ylabel("t [s]")
        plt.title("x-t event projection")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_yt(events):
        plt.figure(figsize=(7, 4))
        plt.scatter(events[:, 2], events[:, 0],
                    c=events[:, 3], s=1, cmap='coolwarm')
        plt.xlabel("y [px]")
        plt.ylabel("t [s]")
        plt.title("y-t event projection")
        plt.tight_layout()
        plt.show()
