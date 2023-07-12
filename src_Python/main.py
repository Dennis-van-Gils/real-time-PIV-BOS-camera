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

import numpy as np
from scipy.signal import correlate2d, fftconvolve
import numba

from skimage.io import imread
import matplotlib.pyplot as plt

from my_fun import IW_Grid, lookup_IW_Idx, subpx_3pgf_2D

# Set the IW sizes for multigrid analysis
# Subsequent IW sizes should be the exact half of the prev IW size
IW_SIZES = [64, 32]  # Use powers of 2 [px]
# IW_SIZES = [64]
IW_OVERLAP = 0.5  # IW overlap fraction [0 - 1]

DEBUG = False  # Print debug info to terminal?
SHOW_CORRELATION_MAP = False

import_file = "E:/Work/_GitHub_repo/2D-PIV-BOS/test_imgs/PIV_rising_vortex_plume/B00001.tif"

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

        # Allocate IW images of frames A and B
        img_IW_A = np.zeros((IW_size, IW_size), dtype=A.dtype)
        img_IW_B = np.zeros((IW_size, IW_size), dtype=B.dtype)

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

            # Undo the shift again when the IW of frame B would leave the
            # borders of frame B. If so, we will zero out the appropiate
            # section of the IW of frame B that corresponds to `particles`
            # that are definitely not present in the IW of frame A.
            IW_B_needs_zeroing_out_L = 0  # left , x = 0
            IW_B_needs_zeroing_out_R = 0  # right, x = IW_size - 1
            IW_B_needs_zeroing_out_U = 0  # up   , y = 0
            IW_B_needs_zeroing_out_D = 0  # down , y = IW_size - 1

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
                    print(f"   shift  {shift_x:+2d}, {shift_y:+2d}", end="")

                # Calculate new center and range of the shifted IW in frame B
                IW_grid_B.x[iIW_y, iIW_x] += shift_x
                IW_grid_B.y[iIW_y, iIW_x] += shift_y
                IW_grid_B.x_range[iIW_y, iIW_x, :] += shift_x
                IW_grid_B.y_range[iIW_y, iIW_x, :] += shift_y

                # Undo the shift again when the IW of frame B would leave the
                # borders of frame B. If so, we will zero out the appropiate
                # section of the IW of frame B that corresponds to `particles`
                # that are definitely not present in the IW of frame A.
                IW_B_needs_zeroing_out_L = 0  # left , x = 0
                IW_B_needs_zeroing_out_R = 0  # right, x = IW_size - 1
                IW_B_needs_zeroing_out_U = 0  # up   , y = 0
                IW_B_needs_zeroing_out_D = 0  # down , y = IW_size - 1

                if IW_grid_B.x_range[iIW_y, iIW_x, 0] < 0:
                    IW_grid_B.x[iIW_y, iIW_x] -= shift_x
                    IW_grid_B.x_range[iIW_y, iIW_x, :] -= shift_x
                    IW_B_needs_zeroing_out_R = np.abs(shift_x)
                    shift_x = 0

                if IW_grid_B.x_range[iIW_y, iIW_x, 1] > img_w - 1:
                    IW_grid_B.x[iIW_y, iIW_x] -= shift_x
                    IW_grid_B.x_range[iIW_y, iIW_x, :] -= shift_x
                    IW_B_needs_zeroing_out_L = np.abs(shift_x)
                    shift_x = 0

                if IW_grid_B.y_range[iIW_y, iIW_x, 0] < 0:
                    IW_grid_B.y[iIW_y, iIW_x] -= shift_y
                    IW_grid_B.y_range[iIW_y, iIW_x, :] -= shift_y
                    IW_B_needs_zeroing_out_D = np.abs(shift_y)
                    shift_y = 0

                if IW_grid_B.y_range[iIW_y, iIW_x, 1] > img_h - 1:
                    IW_grid_B.y[iIW_y, iIW_x] -= shift_y
                    IW_grid_B.y_range[iIW_y, iIW_x, :] -= shift_y
                    IW_B_needs_zeroing_out_U = np.abs(shift_y)
                    shift_y = 0

                if DEBUG:
                    if (IW_B_needs_zeroing_out_L > 0) or (
                        IW_B_needs_zeroing_out_R > 0
                    ):
                        print(" , undo x", end="")
                    if (IW_B_needs_zeroing_out_U > 0) or (
                        IW_B_needs_zeroing_out_D > 0
                    ):
                        print(" , undo y", end="")
                    print("")

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
            # We need a copy, because otherwise the upcoming potential zeroing
            # of the IW image borders will affect, by means of reference, the
            # original image.
            # We would need to copy anyhow when we will start using pyFFTW.
            np.copyto(
                img_IW_A,
                A[IW_grid_A.y_range[iIW_y, iIW_x, 0] :
                  IW_grid_A.y_range[iIW_y, iIW_x, 1] + 1,
                  IW_grid_A.x_range[iIW_y, iIW_x, 0] :
                  IW_grid_A.x_range[iIW_y, iIW_x, 1]+ 1]
            )

            np.copyto(
                img_IW_B,
                B[IW_grid_B.y_range[iIW_y, iIW_x, 0] :
                  IW_grid_B.y_range[iIW_y, iIW_x, 1] + 1,
                  IW_grid_B.x_range[iIW_y, iIW_x, 0] :
                  IW_grid_B.x_range[iIW_y, iIW_x, 1] + 1]
            )
            # fmt: on

            # Zero out the appropiate section of the IW of frame B that
            # corresponds to `particles` that are definitely not present in the
            # IW of frame A. Likewise, zero out the IW of frame A.
            if IW_B_needs_zeroing_out_L > 0:
                img_IW_B[:, :IW_B_needs_zeroing_out_L] = 0
                img_IW_A[:, -IW_B_needs_zeroing_out_L:] = 0
            if IW_B_needs_zeroing_out_R > 0:
                img_IW_B[:, -IW_B_needs_zeroing_out_R:] = 0
                img_IW_A[:, :IW_B_needs_zeroing_out_R] = 0
            if IW_B_needs_zeroing_out_U > 0:
                img_IW_B[:IW_B_needs_zeroing_out_U, :] = 0
                img_IW_A[-IW_B_needs_zeroing_out_U:, :] = 0
            if IW_B_needs_zeroing_out_D > 0:
                img_IW_B[-IW_B_needs_zeroing_out_D:, :] = 0
                img_IW_A[:IW_B_needs_zeroing_out_D, :] = 0

            # ------------------------------------------------------------------
            #   Perform cross-correlation
            # ------------------------------------------------------------------

            if (
                img_IW_A.size == 0
                or np.max(img_IW_A) == 0
                or np.max(img_IW_B) == 0
            ):
                # Save computation time
                C = np.nan
                dx = np.nan
                dy = np.nan

            else:
                # Perform 2D cross-correlation
                C = fftconvolve(
                    img_IW_B, np.flipud(np.fliplr(img_IW_A)), mode="full"
                )
                C = C / np.max(C)

                # Find maximum correlation peak
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

                if SHOW_CORRELATION_MAP:
                    if not (plt.fignum_exists("C_map")):
                        fig = plt.figure("C_map")
                        h_imshow = plt.imshow(
                            np.zeros((IW_size * 2 - 1, IW_size * 2 - 1)),
                            cmap="gray",
                            interpolation="none",
                            vmin=0,
                            vmax=1,
                        )
                        (h_peak,) = plt.plot(IW_size, IW_size, "xr")
                        (h_peak_sub,) = plt.plot(IW_size, IW_size, "xg")
                        h_title = plt.title(f"")

                    h_imshow.set_data(C)  # type: ignore
                    h_peak.set_data([peak_x], [peak_y])  # type: ignore
                    h_peak_sub.set_data([peak_sub_x], [peak_sub_y])  # type: ignore
                    h_title.set_text(f"{iIW} of {IW_grid_A.nIWs}")  # type: ignore

                    plt.draw()
                    plt.pause(0.0001)
                    # plt.waitforbuttonpress()
                    # plt.show(block=False)
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
