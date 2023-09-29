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
__date__ = "29-09-2023"
__version__ = "1.0"
# pylint: disable=missing-function-docstring

import os
import enum
import glob
import configparser
import numpy as np


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
CAMERA_ID = 0

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

# [Debugging]
DEBUG = False
SHOW_CORRELATION_MAPS = False
LOAD_MPL = True


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
    global CAMERA_ID

    IMAGE_SOURCE = getattr(
        IMAGE_SOURCES,
        parser["Source"]["image_source"].upper(),
    )
    IMAGE_PATH = parser["Source"]["image_path"]
    if IMAGE_SOURCE == IMAGE_SOURCES.DISK:
        IMAGE_FILES = glob.glob(IMAGE_PATH)
    CAMERA_ID = parser.getint("Source", "camera_id")

    # [Processing]
    global MODE
    global IW_SIZES
    global IW_OVERLAP

    MODE = getattr(MODES, parser["Processing"]["mode"].upper())
    IW_SIZES = parse_int_list(parser["Processing"]["IW_sizes"])
    IW_OVERLAP = np.clip(parser.getfloat("Processing", "IW_overlap"), 0.0, 0.5)

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
    global DEBUG
    global SHOW_CORRELATION_MAPS
    global LOAD_MPL

    DEBUG = parser.getboolean("Debugging", "debug")
    SHOW_CORRELATION_MAPS = parser.getboolean(
        "Debugging", "show_correlation_maps"
    )
    LOAD_MPL = parser.getboolean("Debugging", "load_mpl")


# ------------------------------------------------------------------------------
#   parse_int_list
# ------------------------------------------------------------------------------


def parse_int_list(str_in):
    try:
        return list(int(k.strip()) for k in str_in[1:-1].split(","))
    except Exception as err:
        raise configparser.ParsingError(str_in) from None


""""
# ------------------------------------------------------------------------------
#   Predefined image sets used for developing this library
# ------------------------------------------------------------------------------

_img_set = 0

if _img_set == 0:
    IMG_PATH = r"../test_imgs/PIV_rising_vortex_plume/*.png"
    IW_SIZES = [64, 32]
    QUIVER_SIZE = 2
    COLOR_DIV = 14
elif _img_set == 1:
    IMG_PATH = r"../test_imgs/swirling_vortices/*.tif"
    IW_SIZES = [128, 64, 32]
    QUIVER_SIZE = 1
    COLOR_DIV = 24
else:
    IMG_PATH = r"../test_imgs/4th_PIV-Challenge_Case_E/*.tif"
    IW_SIZES = [64, 32]
    QUIVER_SIZE = 8
    COLOR_DIV = 4
"""
