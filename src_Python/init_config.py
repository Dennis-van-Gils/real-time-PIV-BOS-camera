#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Initializes all user configuration parameters. Provides function
`read_file()` to load a configuration file from disk.

Example usage for `main.py`:
    import load_config as cfg
    cfg.read_file("config.ini")
    # `cfg` now contains all user settings
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "16-10-2023"
__version__ = "1.0"
# pylint: disable=missing-function-docstring

import os
import enum
import glob
import configparser
import numpy as np

# [Debugging]
# Print debug info to the terminal? Slow!
DEBUG = False
# When True, also requires `load_mpl = True`. Slow!
SHOW_CORRELATION_MAPS = False
# Load matplotlib into memory and show vector map results?
LOAD_MPL = True


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

# [Plotting]
QUIVER_SIZE = 10
COLOR_DIV = 1

# [Advanced]
FFT_LIB = FFT_LIBS.ROCKETFFT
N_WORKERS = 32
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
        IMAGE_FILES = glob.glob(IMAGE_PATH)
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

    MODE = getattr(MODES, parser["Processing"]["mode"].upper())
    IW_SIZES = parse_int_list(parser["Processing"]["IW_sizes"])
    IW_OVERLAP = np.clip(parser.getfloat("Processing", "IW_overlap"), 0.0, 0.8)

    # [Plotting]
    global QUIVER_SIZE
    global COLOR_DIV

    QUIVER_SIZE = parser.getfloat("Plotting", "quiver_size")
    COLOR_DIV = parser.getfloat("Plotting", "color_div")

    # [Advanced]
    global FFT_LIB
    global N_WORKERS
    global N_FFT_THREADS

    FFT_LIB = getattr(FFT_LIBS, parser["Advanced"]["FFT_lib"].upper())
    N_WORKERS = np.maximum(parser.getint("Advanced", "N_workers"), 1)
    N_FFT_THREADS = np.maximum(parser.getint("Advanced", "N_FFT_threads"), 1)

    # [Debugging]
    # Try reading optional keywords. Use defaults when keywords don't exist.
    global DEBUG
    global SHOW_CORRELATION_MAPS
    global LOAD_MPL

    try:
        DEBUG = parser.getboolean("Debugging", "debug")
    except (configparser.NoSectionError, configparser.NoOptionError):
        pass  # Remain silent and use default as specified at the top

    try:
        SHOW_CORRELATION_MAPS = parser.getboolean(
            "Debugging", "show_correlation_maps"
        )
    except (configparser.NoSectionError, configparser.NoOptionError):
        pass

    try:
        LOAD_MPL = parser.getboolean("Debugging", "load_mpl")
    except (configparser.NoSectionError, configparser.NoOptionError):
        pass


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
