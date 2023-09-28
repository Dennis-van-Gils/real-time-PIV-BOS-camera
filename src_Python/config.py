#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Configuration constants for 2D Particle Imaging Velocimetry (PIV) and
Background Oriented Schlieren (BOS).

Used abbrevations
-----------------
IW: Interrogation window
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
# pylint: disable=missing-function-docstring

import glob
from enum import IntEnum


class FFT_LIBS(IntEnum):
    PYFFTW = 0
    ROCKETFFT = 1
    SCIPY = 2


DEBUG = False  # Print debug info to terminal?
SHOW_CORRELATION_MAPS = False  # When True, also requires `LOAD_MPL = True`
LOAD_MPL = True  # Load matplotlib into memory and show vector map results?

# ------------------------------------------------------------------------------
#   User-configurable settings
# ------------------------------------------------------------------------------

# List of square interrogation window sizes in [px] for the multigrid analysis.
# Use powers of two with each subsequent IW size the exact half of the previous
# IW size.
IW_SIZES = [64, 32]  # [px]

# Interrogation window overlap fraction [0 - 1]. Only the last multigrid stage
# will have window overlapping applied to it.
IW_OVERLAP = 0.5

# FFT library to be used for 2D correlations.
#   PYFFTW:
#     TODO: descr
#
#   ROCKETFFT:
#     TODO: descr
#
#   SCIPY:
#     TODO: descr
FFT_LIB = FFT_LIBS.ROCKETFFT

# Number of concurrent workers. Each worker will spawn a separate thread
# (concurrent multithreading) that will process a chunk of all available IWs
# over which the 2D FFT correlations are to be calculated. The chunks will get
# evenly divided over the specified number of workers.
# NOTE: This number is not limited by the number of logical CPU processors: The
# workers are running in concurrent threads, not concurrent processes.
# NOTE: 32 seems to be sufficient to reach maximum speed.
N_WORKERS = 32

# Number of threads to use for each single FFT operation.
# NOTE: Leave it to 1 thread per FFT operation, because the IWs it will operate
# on are generally too small of a size to benefit from multiple threads.
N_FFT_THREADS = 1

# Plot esthetics
QUIVER_SIZE = 3
COLOR_DIV = 14

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

IMG_FILES = glob.glob(IMG_PATH)
