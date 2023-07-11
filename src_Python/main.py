#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image processing algorithm for 2D Particle Imaging Velocimetry (PIV) and
Background Oriented Schlieren (BOS)

Used abbrevations
-----------------
IW: Interrogation window
VM: Displacement vector map
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "10-07-2023"
__version__ = "1.0"
# pylint: disable=missing-function-docstring

import os
import sys
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

# mpl.rcParams["toolbar"] = "None"

import numpy as np
from scipy.signal import correlate2d, fftconvolve
import numba

from skimage.io import imread

from my_fun import IW_Grid, lookup_IW_Idx, subpx_3pgf_2D

import_file = "E:/Work/_GitHub_repo/2D-PIV-BOS/test_imgs/PIV_rising_vortex_plume/B00001.tif"

# Set the IW sizes for multigrid analysis
# Subsequent IW sizes should be the exact half of the prev IW size
# IW_SIZES   = [128 96 64 48 32];
IW_SIZES = [64, 32]
# IW_SIZES = [64]
IW_OVERLAP = 0.5

DEBUG = False

# ------------------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Read double image and split into frames A & B
    img = imread(import_file, as_gray=True)
    img_2h, img_w = np.shape(img)
    img_h = int(img_2h / 2)
    A = img[:img_h, :]
    B = img[img_h:, :]

    # Mean background removal
    A = np.clip(A - np.mean(A), 0, None).astype(int)
    B = np.clip(B - np.mean(B), 0, None).astype(int)

    # --------------------------------------------------------------------------
    #   Initialize
    # --------------------------------------------------------------------------

    # Allocate multigrid maps
    nIW_SIZES = len(IW_SIZES)
    IW_grid_As: list[IW_Grid] = []
    IW_grid_Bs: list[IW_Grid] = []
    VMs_x = []
    VMs_y = []
    VMs_dx = []
    VMs_dy = []

    # --------------------------------------------------------------------------
    #   Walk over all interrogation window sizes
    # --------------------------------------------------------------------------
    t_0 = perf_counter()

    for iIW_size, IW_size in enumerate(IW_SIZES):
        # Create IW_grid for frame A
        # Create IW_grid for frame B
        IW_grid_A = IW_Grid(img_w, img_h, IW_size, IW_OVERLAP)
        IW_grid_B = IW_Grid(img_w, img_h, IW_size, IW_OVERLAP)

        # Already store IW_grid_A in the multigrid map for the first IW size
        if iIW_size == 1:
            IW_grid_As.append(IW_grid_A)

        # Allocate memory for displacement vector map
        VM_x = IW_grid_A.x
        VM_y = IW_grid_A.y
        VM_dx = np.zeros(IW_grid_A.x.shape)
        VM_dy = np.zeros(IW_grid_A.x.shape)

        # ----------------------------------------------------------------------
        #   Walk over all IWs
        # ----------------------------------------------------------------------

        for (iIW_y, iIW_x), IW_px_x in np.ndenumerate(IW_grid_A.x):
            # Flattened iter index
            iIW = np.ravel_multi_index((iIW_y, iIW_x), IW_grid_A.x.shape)
            IW_px_y = IW_grid_A.y[iIW_y, iIW_x]

            if DEBUG:
                print(
                    f"IW: {iIW} of {IW_grid_A.nIWs - 1} "
                    f"@px {IW_px_x}, {IW_px_y}"
                )

            # ------------------------------------------------------------------
            #   Calculate IW of frame B
            #   Apply window shifting technique
            # ------------------------------------------------------------------

            if iIW_size == 0:
                # First IW size, no pre-shift available
                shift_x = 0  # [px]
                shift_y = 0  # [px]
            else:
                # Pre-shift available
                # Calculate corresponding index of the IW in the larger parent
                # grid
                iIW_parent_x, iIW_parent_y = lookup_IW_Idx(
                    IW_grid_As[iIW_size - 1],
                    IW_px_x,
                    IW_px_y,
                )

                # Retrieve the pre-shift
                shift_x = VMs_dx[-1][iIW_parent_y, iIW_parent_x]
                shift_y = VMs_dy[-1][iIW_parent_y, iIW_parent_x]

                if np.isnan(shift_x):
                    shift_x = 0
                else:
                    shift_x = int(np.round(shift_x))

                if np.isnan(shift_y):
                    shift_y = 0
                else:
                    shift_y = int(np.round(shift_y))

                if DEBUG:
                    # Flattened iter index
                    iIW_parent = np.ravel_multi_index(
                        (iIW_parent_y, iIW_parent_x),
                        IW_grid_As[iIW_size - 1].x.shape,
                    )
                    print(f"   parent IW {iIW_parent}")
                    print(f"   shift  {shift_x:+2d}, {shift_y:+2d}")

                # Calculate new center and range of the shifted IW in frame B
                IW_grid_B.x[iIW_y, iIW_x] += shift_x
                IW_grid_B.y[iIW_y, iIW_x] += shift_y
                IW_grid_B.x_range[iIW_y, iIW_x, :] += shift_x
                IW_grid_B.y_range[iIW_y, iIW_x, :] += shift_y

                # The IW should never be shifted outside of frame B.
                # When it does, equally resize the IWs of both frames A and B
                # such that the resized IW of frame B still fits in frame B
                if IW_grid_B.x_range[iIW_y, iIW_x, 0] < 0:
                    IW_grid_B.x_range[iIW_y, iIW_x, 0] = 0
                    IW_grid_A.x_range[iIW_y, iIW_x, 0] = -shift_x
                if IW_grid_B.y_range[iIW_y, iIW_x, 0] < 0:
                    IW_grid_B.y_range[iIW_y, iIW_x, 0] = 0
                    IW_grid_A.y_range[iIW_y, iIW_x, 0] = -shift_y
                if IW_grid_B.x_range[iIW_y, iIW_x, 1] > img_w - 1:
                    IW_grid_B.x_range[iIW_y, iIW_x, 1] = img_w - 1
                    IW_grid_A.x_range[iIW_y, iIW_x, 1] = img_w - 1 - shift_x
                if IW_grid_B.y_range[iIW_y, iIW_x, 1] > img_h - 1:
                    IW_grid_B.y_range[iIW_y, iIW_x, 1] = img_h - 1
                    IW_grid_A.y_range[iIW_y, iIW_x, 1] = img_h - 1 - shift_y

            # ------------------------------------------------------------------
            #   Retrieve images of IW frame A and IW frame B
            # ------------------------------------------------------------------

            if DEBUG:
                print(
                    "   A_xrng ["
                    f"{IW_grid_A.x_range[iIW_y, iIW_x, 0]:4d}, "
                    f"{IW_grid_A.x_range[iIW_y, iIW_x, 1]:4d}]"
                )
                print(
                    "   A_yrng ["
                    f"{IW_grid_A.y_range[iIW_y, iIW_x, 0]:4d}, "
                    f"{IW_grid_A.y_range[iIW_y, iIW_x, 1]:4d}]"
                )
                print(
                    "   B_xrng ["
                    f"{IW_grid_B.x_range[iIW_y, iIW_x, 0]:4d}, "
                    f"{IW_grid_B.x_range[iIW_y, iIW_x, 1]:4d}]"
                )
                print(
                    "   B_yrng ["
                    f"{IW_grid_B.y_range[iIW_y, iIW_x, 0]:4d}, "
                    f"{IW_grid_B.y_range[iIW_y, iIW_x, 1]:4d}]"
                )

            # fmt: off
            img_IW_A = A[
                IW_grid_A.y_range[iIW_y, iIW_x, 0] :
                IW_grid_A.y_range[iIW_y, iIW_x, 1] + 1,
                IW_grid_A.x_range[iIW_y, iIW_x, 0] :
                IW_grid_A.x_range[iIW_y, iIW_x, 1]+ 1,
            ]

            img_IW_B = B[
                IW_grid_B.y_range[iIW_y, iIW_x, 0] :
                IW_grid_B.y_range[iIW_y, iIW_x, 1] + 1,
                IW_grid_B.x_range[iIW_y, iIW_x, 0] :
                IW_grid_B.x_range[iIW_y, iIW_x, 1] + 1,
            ]
            # fmt: on

            # ------------------------------------------------------------------
            #   Perform cross-correlation
            # ------------------------------------------------------------------

            if (
                img_IW_A.size == 0
                or np.max(img_IW_A) == 0
                or np.max(img_IW_B) == 0
            ):
                C = np.nan  # Save computation time
            else:
                C = fftconvolve(
                    img_IW_B, np.flipud(np.fliplr(img_IW_A)), mode="full"
                )
                C = C / np.max(C)

            # Find maximum correlation peak
            if np.isnan(C).any():
                dx = np.nan
                dy = np.nan
            else:
                iMaxC = np.argmax(C)
                peak_y, peak_x = np.unravel_index(iMaxC, C.shape, order="C")
                peak_x = int(peak_x)
                peak_y = int(peak_y)

                # Sub-pixel resolution algorithm, 3-point Gaussian fit
                peak_sub_x, peak_sub_y = subpx_3pgf_2D(C, peak_x, peak_y)

                # Calculate displacement vector
                dx = peak_sub_x - C.shape[1] // 2 + shift_x
                dy = peak_sub_y - C.shape[0] // 2 + shift_y

                if DEBUG:
                    print(f"     peak   @ {peak_x:+6.2f}, {peak_y:+6.2f}")
                    print(
                        f"     3pgf   @ {peak_sub_x:+6.2f}, {peak_sub_y:+6.2f}"
                    )
                    print(f"     dx, dy = {dx:+6.2f}, {dy:+6.2f}")

                if 0:  # DEBUG flag: Show correlation map
                    if (not "h_imshow" in locals()) or (iIW == 0):
                        fig = plt.figure()
                        h_imshow = plt.imshow(
                            C,
                            cmap="gray",
                            interpolation="none",
                            vmin=0,
                            vmax=1,
                        )
                        (h_peak,) = plt.plot(peak_x, peak_y, "xr")
                        (h_peak_sub,) = plt.plot(peak_sub_x, peak_sub_y, "xg")
                        h_title = plt.title(f"{iIW} of {IW_grid_A.nIWs - 1}")
                    else:
                        h_imshow.set_data(C)
                        h_peak.set_data(peak_x, peak_y)
                        h_peak_sub.set_data(peak_sub_x, peak_sub_y)
                        h_title.set_text(f"{iIW} of {IW_grid_A.nIWs}")

                    plt.draw()
                    plt.pause(0.0001)
                    # plt.show()

            # Store result in vector map
            VM_dx[iIW_y, iIW_x] = dx
            VM_dy[iIW_y, iIW_x] = dy

        # -----------------------------------------------------------------------
        #   Store multigrid maps
        # ----------------------------------------------------------------------

        IW_grid_As.append(IW_grid_A)
        IW_grid_Bs.append(IW_grid_B)
        VMs_x.append(VM_x)
        VMs_y.append(VM_y)
        VMs_dx.append(VM_dx)
        VMs_dy.append(VM_dy)

    duration = perf_counter() - t_0
    print(f"Finished in {duration:.3f} s")
    # scipy.signal.fftconvolve takes ~1.14 s in alacritty without printing

    # --------------------------------------------------------------------------
    #   Show original image A with unfiltered vector map on top
    # --------------------------------------------------------------------------
    quiverX = 3

    if 1:
        fig = plt.figure()
        plt.imshow(A, cmap="gray", interpolation="none")
        plt.quiver(
            VMs_x[-1],
            VMs_y[-1],
            VMs_dx[-1] * quiverX,
            VMs_dy[-1] * quiverX,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
            linewidths=2,
        )
        plt.show()
