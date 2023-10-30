#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Initializes all user configuration parameters. Provides function
`read_file()` to load a configuration file from disk.

Example usage for `main.py`:
    import load_config as cfg
    cfg.read_file("config.ini")

Namespace `cfg` now contains all user settings
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "30-10-2023"
__version__ = "1.0"
# pylint: disable=missing-function-docstring

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

# Load matplotlib into memory and show vector map results?
LOAD_MPL = True

# Show detailed IW analysis at the specified pixel location (x, y)?
# - Specify a tuple as `(x, y)` to show IW analysis.
# - Set to `None` or empty tuple `()` to skip.
# DEBUG_IW_PX = (500, 500)

if not "DEBUG_IW_PX" in locals():
    DEBUG_IW_PX = None

if isinstance(DEBUG_IW_PX, tuple) and len(DEBUG_IW_PX) == 2:  # type: ignore
    LOAD_MPL = True
else:
    DEBUG_IW_PX = None

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
#   Defaults
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
QUIVER_SIZE = 10
COLOR_DIV = 1

# [Advanced]
FFT_LIB = FFT_LIBS.ROCKETFFT
MAX_WORKERS = 32
N_FFT_THREADS = 1


# ------------------------------------------------------------------------------
#   read_file
# ------------------------------------------------------------------------------


def read_file(filename=None):
    if filename in (None, ""):
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(
            title="Select configuration file to open",
            filetypes=(("Configuration files", "*.ini"), ("All files", "*.*")),
            initialfile="config.ini",
        )

        if filename in (None, ""):
            print(
                "WARNING: No configuration file was selected. Using default "
                "parameters."
            )
            return

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Could not open `{filename}`.")

    print(f"Reading configuration file: {filename}")
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
            raise Exception(
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
    global QUIVER_SIZE
    global COLOR_DIV

    QUIVER_SIZE = parser.getfloat("Plotting", "quiver_size")
    COLOR_DIV = parser.getfloat("Plotting", "color_div")

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
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(
            title="Select configuration file to open",
            filetypes=(("Configuration files", "*.ini"), ("All files", "*.*")),
            initialfile="live_preview.ini",
        )

        if filename in (None, ""):
            print(
                "WARNING: No configuration file was selected. Using default "
                "parameters."
            )
            return

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Could not open `{filename}`.")

    print(f"Reading configuration file: {filename}")
    parser = configparser.ConfigParser()
    parser.read(filename)

    # [Source]
    global IMAGE_SOURCE
    global CAMERA_ID
    global WANTED_RESOLUTION

    IMAGE_SOURCE = getattr(
        IMAGE_SOURCES,
        parser["Source"]["image_source"].upper(),
    )
    CAMERA_ID = parser.getint("Source", "camera_id")
    WANTED_RESOLUTION = parse_int_list(parser["Source"]["wanted_resolution"])


# ------------------------------------------------------------------------------
#   parse_int_list
# ------------------------------------------------------------------------------


def parse_int_list(str_in):
    try:
        return list(int(k.strip()) for k in str_in[1:-1].split(","))
    except Exception as err:
        raise configparser.ParsingError(str_in) from None
