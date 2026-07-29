#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring
"""Initializes all user configuration parameters. Provides function
`read_file()` to load a configuration file from disk.

Example usage for `main.py`:
    import load_config as cfg
    cfg.read_file("config.ini")

Namespace `cfg` now contains all user settings
"""

__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/real-time-PIV-BOS-camera"
__date__ = "27-09-2026"
# pylint: disable=global-statement, broad-exception-caught

import os
import enum
import glob
import configparser
import numpy as np

# ------------------------------------------------------------------------------
#   Debugging
# ------------------------------------------------------------------------------

# When true, only processes the first image pair and halts on plotting
DEBUG = False

# Print debug info to the terminal? Slow!
DEBUG_PRINT = False

# Set to False for speed-testing code
PLOT_VECTOR_MAP_RESULTS = True

# Show detailed IW analysis at the specified pixel location (x, y)?
# - Specify a tuple as `(x, y)` to show IW analysis.
# - Set to an empty tuple `()` to skip.
DEBUG_IW_PX: tuple[int, int] | tuple = ()

# ------------------------------------------------------------------------------
#   Enumerations
# ------------------------------------------------------------------------------


class IMAGE_SOURCES(enum.IntEnum):
    DISK = 0
    WEBCAM = 1
    XIMEA = 2


class MODES(enum.IntEnum):
    PIV = 0
    PIV2 = 1
    BOS = 2


class FFT_LIBS(enum.IntEnum):
    PYFFTW = 0
    ROCKETFFT = 1
    SCIPY = 2


# ------------------------------------------------------------------------------
#   Default placeholders
# ------------------------------------------------------------------------------


# [Source]
IMAGE_SOURCE = IMAGE_SOURCES.WEBCAM
IMAGE_PATH = ""
IMAGE_FILES: list[str] = []  # Will be derived from `IMAGE_PATH`
N_IMAGES = 0  # Will be derived from `IMAGE_PATH`
CAMERA_ID = 0
WANTED_RESOLUTION = [1280, 720]

# [Processing]
MODE = MODES.BOS
IW_SIZES = [32]
IW_OVERLAP = 0.5
N_STAGES = len(IW_SIZES)

# [Plotting]
colormap_name = "jet"
colormap_max_pixel_displacement = 1.0
colormap_clip_warning = False
colormap_clip_color = (1.0, 0, 0.58)  # Vibrant pink
quiver_size = 10

# [Advanced]
FFT_LIB = FFT_LIBS.ROCKETFFT
MAX_WORKERS = 32
N_FFT_THREADS = 1


# ------------------------------------------------------------------------------
#   read_file
# ------------------------------------------------------------------------------


def read_file(filename=None):
    if filename in (None, ""):
        # Lazy loading of tkinter module
        # pylint: disable=import-outside-toplevel
        import tkinter as tk
        from tkinter import filedialog

        # pylint: enable=import-outside-toplevel

        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(
            title="Select configuration file to open",
            filetypes=(("Configuration files", "*.ini"), ("All files", "*.*")),
            initialfile="config.ini",
        )

        if filename in (None, ""):
            print(
                "\nWARNING: No configuration file was selected. Using default "
                "parameters."
            )
            return

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Could not open `{filename}`.")

    print(f"\nReading configuration file:\n  {filename}")
    parser = configparser.ConfigParser()
    parser.read(filename)

    # [Source]
    global IMAGE_SOURCE
    global IMAGE_PATH
    global IMAGE_FILES
    global N_IMAGES
    global CAMERA_ID
    global WANTED_RESOLUTION

    IMAGE_SOURCE = getattr(
        IMAGE_SOURCES,
        parser["Source"]["image_source"].upper(),
    )
    IMAGE_PATH = parser["Source"]["image_path"]
    if IMAGE_SOURCE == IMAGE_SOURCES.DISK:
        IMAGE_FILES = sorted(glob.glob(IMAGE_PATH))
        N_IMAGES = len(IMAGE_FILES)
        if N_IMAGES < 2:
            raise ValueError(
                "Less than 2 images found in the supplied image path "
                f'"{IMAGE_PATH}".\nExiting.'
            )
    CAMERA_ID = parser.getint("Source", "camera_id")
    WANTED_RESOLUTION = parse_int_list(parser["Source"]["wanted_resolution"])

    # [Processing]
    global MODE
    global IW_SIZES
    global IW_OVERLAP
    global N_STAGES

    MODE = getattr(MODES, parser["Processing"]["mode"].upper())
    IW_SIZES = parse_int_list(parser["Processing"]["IW_sizes"])
    IW_OVERLAP = np.clip(parser.getfloat("Processing", "IW_overlap"), 0.0, 0.8)
    N_STAGES = len(IW_SIZES)

    # [Plotting]
    global colormap_name
    global colormap_max_pixel_displacement
    global colormap_clip_warning
    global colormap_clip_color
    global quiver_size

    colormap_name = parser["Plotting"]["colormap_name"]
    colormap_max_pixel_displacement = parser.getfloat(
        "Plotting",
        "colormap_max_pixel_displacement",
    )
    colormap_clip_warning = parser.getboolean(
        "Plotting", "colormap_clip_warning"
    )
    colormap_clip_color = parse_float_list(
        parser["Plotting"]["colormap_clip_color"],
    )
    quiver_size = parser.getfloat("Plotting", "quiver_size")

    # [Advanced]
    global FFT_LIB
    global MAX_WORKERS
    global N_FFT_THREADS

    FFT_LIB = getattr(FFT_LIBS, parser["Advanced"]["FFT_lib"].upper())
    MAX_WORKERS = np.maximum(parser.getint("Advanced", "max_workers"), 1)
    N_FFT_THREADS = np.maximum(parser.getint("Advanced", "N_FFT_threads"), 1)


# ------------------------------------------------------------------------------
#   read_file_live_preview
# ------------------------------------------------------------------------------


def read_file_live_preview(filename=None):
    if filename in (None, ""):
        # Lazy loading of tkinter module
        # pylint: disable=import-outside-toplevel
        import tkinter as tk
        from tkinter import filedialog

        # pylint: enable=import-outside-toplevel

        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(
            title="Select configuration file to open",
            filetypes=(("Configuration files", "*.ini"), ("All files", "*.*")),
            initialfile="live_preview.ini",
        )

        if filename in (None, ""):
            print(
                "\nWARNING: No configuration file was selected. Using default "
                "parameters."
            )
            return

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Could not open `{filename}`.")

    print(f"\nReading configuration file:\n {filename}")
    parser = configparser.ConfigParser()
    parser.read(filename)

    # [Source]
    global IMAGE_SOURCE
    global IMAGE_PATH
    global IMAGE_FILES
    global N_IMAGES
    global CAMERA_ID
    global WANTED_RESOLUTION

    IMAGE_SOURCE = getattr(
        IMAGE_SOURCES,
        parser["Source"]["image_source"].upper(),
    )
    try:
        IMAGE_PATH = parser["Source"]["image_path"]
    except KeyError:
        pass
    else:
        if IMAGE_SOURCE == IMAGE_SOURCES.DISK:
            IMAGE_FILES = sorted(glob.glob(IMAGE_PATH))
            N_IMAGES = len(IMAGE_FILES)
            if N_IMAGES < 2:
                raise ValueError(
                    "Less than 2 images found in the supplied image path "
                    f'"{IMAGE_PATH}".\nExiting.'
                )
    CAMERA_ID = parser.getint("Source", "camera_id")
    WANTED_RESOLUTION = parse_int_list(parser["Source"]["wanted_resolution"])


# ------------------------------------------------------------------------------
#   parse_int_list
# ------------------------------------------------------------------------------


def parse_int_list(str_in):
    try:
        return list(int(k.strip()) for k in str_in[1:-1].split(","))
    except Exception as _err:
        raise configparser.ParsingError(str_in) from None


# ------------------------------------------------------------------------------
#   parse_float_list
# ------------------------------------------------------------------------------


def parse_float_list(str_in):
    try:
        return list(float(k.strip()) for k in str_in[1:-1].split(","))
    except Exception as _err:
        return None
