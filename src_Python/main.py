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
__date__ = "13-07-2023"
__version__ = "1.0"
# pylint: disable=missing-function-docstring

import os
import sys
from typing import Any
from time import perf_counter

import numpy as np
from scipy.signal import fftconvolve
import numba

from skimage.io import imread
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

mpl.use("TkAgg")

from my_fun import (
    remove_mean_background,
    create_IW_grid,
    lookup_iIW,
    fliplrud,
    subpx_3pgf_2D,
)
from convolve2d__my_code import FFTW_Convolver_Full2D

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
    A = remove_mean_background(A)
    B = remove_mean_background(B)

    # --------------------------------------------------------------------------
    #   Initialize
    # --------------------------------------------------------------------------
    nIW_SIZES = len(IW_SIZES)

    # IW parameters per stage of the multigrid.
    # Will hold a list of tuples:
    #   list  [stage number (``int``)]
    #   tuple [IW_size (``int``)     ,
    #          IW_overlap (``float``),
    #          nIWs_x (``int``)      ,
    #          nIWs_y (``int``)]
    IW_params: list[tuple[int, float, int, int]] = []

    # Allocate vector maps per stage of the multigrid.
    # Each will hold a list of numpy.ndarrays.
    VMs_grid_x: list[np.ndarray] = []
    VMs_grid_y: list[np.ndarray] = []
    VMs_dx: list[np.ndarray] = []
    VMs_dy: list[np.ndarray] = []

    # Plan pyFFTW ahead of time
    fftws = []
    for iIW_size, IW_size in enumerate(IW_SIZES):
        fftws.append(
            FFTW_Convolver_Full2D(
                (IW_size, IW_size), (IW_size, IW_size), fftw_threads=1
            )
        )

    # --------------------------------------------------------------------------
    #   Walk over all interrogation window sizes
    # --------------------------------------------------------------------------
    t_0 = perf_counter()

    for iIW_size, IW_size in enumerate(IW_SIZES):
        # Create interrogation windows
        (
            A_IW_grid_x,
            A_IW_grid_y,
            A_IW_xlims,
            A_IW_ylims,
            nIWs_x,
            nIWs_y,
            nIWs,
        ) = create_IW_grid(img_w, img_h, IW_size, IW_OVERLAP)

        B_IW_grid_x = np.copy(A_IW_grid_x)
        B_IW_grid_y = np.copy(A_IW_grid_y)
        B_IW_xlims = np.copy(A_IW_xlims)
        B_IW_ylims = np.copy(A_IW_ylims)

        # Store IW parameters in list
        IW_params.append((IW_size, IW_OVERLAP, nIWs_x, nIWs_y))

        # Allocate IW image subset of frames A and B
        img_IW_A = np.zeros((IW_size, IW_size), dtype=A.dtype)
        img_IW_B = np.zeros((IW_size, IW_size), dtype=B.dtype)

        # Allocate memory for displacement vector map
        VM_grid_x = np.copy(A_IW_grid_x)
        VM_grid_y = np.copy(A_IW_grid_y)
        VM_dx = np.zeros(A_IW_grid_x.shape)
        VM_dy = np.zeros(A_IW_grid_x.shape)

        # ----------------------------------------------------------------------
        #   Debug plots
        # ----------------------------------------------------------------------

        if 0:  # DEBUG flag: Examine IW meshgrid
            # fmt: off
            p = {"fillstyle": "none", "markersize": 6, "linewidth": 2}
            plt.figure()
            plt.plot(
                A_IW_xlims[:, :, 0].reshape(-1),
                A_IW_ylims[:, :, 0].reshape(-1),
                "xg", label="IW starts", **p)
            plt.plot(
                A_IW_xlims[:, :, 1].reshape(-1),
                A_IW_ylims[:, :, 1].reshape(-1),
                "xr", label="IW ends", **p)

            # Plot IW centers, but not all. Just the bottom and left chords.
            plt.plot(
                A_IW_grid_x[:, 0],
                A_IW_grid_y[:, 0],
                "ok", label="IW centers", **p)
            plt.plot(
                A_IW_grid_x[0, :],
                A_IW_grid_y[0, :],
                "ok", label="_nolegend_", **p)

            # Plot image bounding box
            plt.gca().add_patch(
                Rectangle(
                    (0, 0), img_w - 1, img_h - 1, edgecolor="k", fill=None, lw=1
                )
            )
            # fmt: on
            plt.legend()
            plt.show()

        if SHOW_CORRELATION_MAP:
            # Reset any existing plot of the correlation map, because the IW
            # size has changed and plotting on top of imshow needs a rescale.
            if plt.fignum_exists("C_map"):
                plt.close("C_map")

        # ----------------------------------------------------------------------
        #   Walk over all IWs
        # ----------------------------------------------------------------------

        for (iIW_y, iIW_x), IW_px_x in np.ndenumerate(A_IW_grid_x):
            # Flattened iter index
            iIW = np.ravel_multi_index((iIW_y, iIW_x), A_IW_grid_x.shape)

            # y-pixel position of the current IW center
            IW_px_y = A_IW_grid_y[iIW_y, iIW_x]

            if DEBUG:
                print(f"IW: {iIW} of {nIWs - 1} " f"@px {IW_px_x}, {IW_px_y}")

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
                # Pre-shift available: Look up corresponding index of the IW in
                # the larger parent grid
                iIW_parent_x, iIW_parent_y = lookup_iIW(
                    IW_px_x,
                    IW_px_y,
                    IW_params[-2],
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
                        (IW_params[-2][3], IW_params[-2][2]),
                    )
                    print(f"   parent IW {iIW_parent}")
                    print(f"   shift  {shift_x:+2d}, {shift_y:+2d}", end="")

                # Calculate new center and limits of the shifted IW in frame B
                B_IW_grid_x[iIW_y, iIW_x] += shift_x
                B_IW_grid_y[iIW_y, iIW_x] += shift_y
                B_IW_xlims[iIW_y, iIW_x, :] += shift_x
                B_IW_ylims[iIW_y, iIW_x, :] += shift_y

                # Undo the shift again when the IW of frame B would leave the
                # borders of frame B. If so, we will zero out the appropiate
                # section of the IW of frame B that corresponds to `particles`
                # that are definitely not present in the IW of frame A.
                IW_B_needs_zeroing_out_L = 0  # left , x = 0
                IW_B_needs_zeroing_out_R = 0  # right, x = IW_size - 1
                IW_B_needs_zeroing_out_U = 0  # up   , y = 0
                IW_B_needs_zeroing_out_D = 0  # down , y = IW_size - 1

                if B_IW_xlims[iIW_y, iIW_x, 0] < 0:
                    B_IW_grid_x[iIW_y, iIW_x] -= shift_x
                    B_IW_xlims[iIW_y, iIW_x, :] -= shift_x
                    IW_B_needs_zeroing_out_R = np.abs(shift_x)
                    shift_x = 0

                if B_IW_xlims[iIW_y, iIW_x, 1] > img_w - 1:
                    B_IW_grid_x[iIW_y, iIW_x] -= shift_x
                    B_IW_xlims[iIW_y, iIW_x, :] -= shift_x
                    IW_B_needs_zeroing_out_L = np.abs(shift_x)
                    shift_x = 0

                if B_IW_ylims[iIW_y, iIW_x, 0] < 0:
                    B_IW_grid_y[iIW_y, iIW_x] -= shift_y
                    B_IW_ylims[iIW_y, iIW_x, :] -= shift_y
                    IW_B_needs_zeroing_out_D = np.abs(shift_y)
                    shift_y = 0

                if B_IW_ylims[iIW_y, iIW_x, 1] > img_h - 1:
                    B_IW_grid_y[iIW_y, iIW_x] -= shift_y
                    B_IW_ylims[iIW_y, iIW_x, :] -= shift_y
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
                    "   A_xlim ["
                    f"{A_IW_xlims[iIW_y, iIW_x, 0]:4d}, "
                    f"{A_IW_xlims[iIW_y, iIW_x, 1]:4d}]"
                )
                print(
                    "   A_ylim ["
                    f"{A_IW_ylims[iIW_y, iIW_x, 0]:4d}, "
                    f"{A_IW_ylims[iIW_y, iIW_x, 1]:4d}]"
                )
                print(
                    "   B_xlim ["
                    f"{B_IW_xlims[iIW_y, iIW_x, 0]:4d}, "
                    f"{B_IW_xlims[iIW_y, iIW_x, 1]:4d}]"
                )
                print(
                    "   B_ylim ["
                    f"{B_IW_ylims[iIW_y, iIW_x, 0]:4d}, "
                    f"{B_IW_ylims[iIW_y, iIW_x, 1]:4d}]"
                )

            # fmt: off
            # We need a copy, because otherwise the upcoming potential zeroing
            # of the IW image borders will affect, by means of reference, the
            # original image.
            # We would need to copy anyhow when we will start using pyFFTW.
            np.copyto(
                img_IW_A,
                A[A_IW_ylims[iIW_y, iIW_x, 0] :
                  A_IW_ylims[iIW_y, iIW_x, 1] + 1,
                  A_IW_xlims[iIW_y, iIW_x, 0] :
                  A_IW_xlims[iIW_y, iIW_x, 1]+ 1]
            )

            np.copyto(
                img_IW_B,
                B[B_IW_ylims[iIW_y, iIW_x, 0] :
                  B_IW_ylims[iIW_y, iIW_x, 1] + 1,
                  B_IW_xlims[iIW_y, iIW_x, 0] :
                  B_IW_xlims[iIW_y, iIW_x, 1] + 1]
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
                # C = fftconvolve(img_IW_B, fliplrud(img_IW_A), mode="full")
                # C = fftw_1.convolve(img_IW_B, fliplrud(img_IW_A))
                C = fftws[iIW_size].convolve(img_IW_B, fliplrud(img_IW_A))
                np.divide(C, np.max(C), out=C)
                # C = C / np.max(C)

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
                    h_title.set_text(f"{iIW} of {nIWs}")  # type: ignore

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

        VMs_grid_x.append(VM_grid_x)
        VMs_grid_y.append(VM_grid_y)
        VMs_dx.append(VM_dx)
        VMs_dy.append(VM_dy)

    duration = perf_counter() - t_0
    print(f"Finished in {duration:.3f} s")

    # --------------------------------------------------------------------------
    #   Show original image A with unfiltered vector map on top
    # --------------------------------------------------------------------------
    quiverX = 3

    if 1:
        fig = plt.figure()
        plt.imshow(A, cmap="gray", interpolation="none")
        plt.quiver(
            VMs_grid_x[-1],
            VMs_grid_y[-1],
            VMs_dx[-1] * quiverX,
            VMs_dy[-1] * quiverX,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
            linewidths=2,
        )
        plt.show()
