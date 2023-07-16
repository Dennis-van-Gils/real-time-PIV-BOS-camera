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
__date__ = "15-07-2023"
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

from my_fun import (
    remove_mean_background,
    create_IW_grid,
    lookup_IW_idx,
    fliplrud,
    normalize_C_maps,
    compute_displacement_vectors_from_C_maps,
)
from convolve2d__my_code import FFTW_Convolver_Full2D

# Set the IW sizes for multigrid analysis
# Subsequent IW sizes should be the exact half of the prev IW size
IW_SIZES = [64, 32]  # Use powers of 2 [px]
# IW_SIZES = [64]
IW_OVERLAP = 0.5  # IW overlap fraction [0 - 1]

DEBUG = False  # Print debug info to terminal?
SHOW_CORRELATION_MAPS = False
LOAD_MPL = False
# if LOAD_MPL:
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

mpl.use("TkAgg")

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
    # Preallocate and populate lists for the upcoming multigrid analysis.
    # stage: Current multigrid stage from the largest IW size to the smallest.
    # Prefix 'l' denotes 'list' with index `stage_idx`.
    N_stages = len(IW_SIZES)

    # List of IW parameters per stage of the multigrid
    #   tuple [IW_size (``int``)     ,
    #          IW_overlap (``float``),
    #          N_IWs_x (``int``)     ,
    #          N_IWs_y (``int``)]
    lIW_params: list[tuple[int, float, int, int]] = []

    # fmt: off
    # List of IW meshgrids and limits per stage of the multigrid
    lA_IW_grid_x: list[np.ndarray] = []  # np.ndarray[N_IWs_y, N_IWs_x]
    lA_IW_grid_y: list[np.ndarray] = []  # np.ndarray[N_IWs_y, N_IWs_x]
    lA_IW_lims_x: list[np.ndarray] = []  # np.ndarray[N_IWs_y, N_IWs_x, 2]
    lA_IW_lims_y: list[np.ndarray] = []  # np.ndarray[N_IWs_y, N_IWs_x, 2]

    lB_IW_grid_x: list[np.ndarray] = []  # np.ndarray[N_IWs_y, N_IWs_x]
    lB_IW_grid_y: list[np.ndarray] = []  # np.ndarray[N_IWs_y, N_IWs_x]
    lB_IW_lims_x: list[np.ndarray] = []  # np.ndarray[N_IWs_y, N_IWs_x, 2]
    lB_IW_lims_y: list[np.ndarray] = []  # np.ndarray[N_IWs_y, N_IWs_x, 2]

    # List of computed IW shifts per stage of the multigrid
    # NOTE: List index 0, which corresponds to `stage_idx = 0`, will be
    # initialized with zeros and remain so, because no window shifts exist for
    # the first multigrid stage by design.
    lIW_shifts_x: list[np.ndarray] = []  # np.ndarray[N_IWs_y, N_IWs_x]
    lIW_shifts_y: list[np.ndarray] = []  # np.ndarray[N_IWs_y, N_IWs_x]

    # List of computed correlations maps per stage of the multigrid
    #   np.ndarray[N_IWs_y, N_IWs_x, IW_size / 2 + 1, IW_size / 2 + 1]
    lC_maps: list[np.ndarray] = []

    # List of computed displacement vector maps per stage of the multigrid
    lVM_grid_x: list[np.ndarray] = []    # np.ndarray[N_IWs_y, N_IWs_x]
    lVM_grid_y: list[np.ndarray] = []    # np.ndarray[N_IWs_y, N_IWs_x]
    lVM_dx: list[np.ndarray] = []        # np.ndarray[N_IWs_y, N_IWs_x]
    lVM_dy: list[np.ndarray] = []        # np.ndarray[N_IWs_y, N_IWs_x]
    # fmt: on

    # List of pyFFTW calculation objects per stage of the multigrid
    lfftw: list[FFTW_Convolver_Full2D] = []

    for stage_idx, IW_size in enumerate(IW_SIZES):
        # Create interrogation windows
        (
            IW_grid_x,
            IW_grid_y,
            IW_lims_x,
            IW_lims_y,
            N_IWs_x,
            N_IWs_y,
        ) = create_IW_grid(img_w, img_h, IW_size, IW_OVERLAP)

        # Populate lists
        lIW_params.append((IW_size, IW_OVERLAP, N_IWs_x, N_IWs_y))

        lA_IW_grid_x.append(np.copy(IW_grid_x))
        lA_IW_grid_y.append(np.copy(IW_grid_y))
        lA_IW_lims_x.append(np.copy(IW_lims_x))
        lA_IW_lims_y.append(np.copy(IW_lims_y))

        lB_IW_grid_x.append(np.copy(IW_grid_x))
        lB_IW_grid_y.append(np.copy(IW_grid_y))
        lB_IW_lims_x.append(np.copy(IW_lims_x))
        lB_IW_lims_y.append(np.copy(IW_lims_y))

        lIW_shifts_x.append(np.zeros((N_IWs_y, N_IWs_x)))
        lIW_shifts_y.append(np.zeros((N_IWs_y, N_IWs_x)))

        C_maps = np.zeros((N_IWs_y, N_IWs_x, IW_size * 2 - 1, IW_size * 2 - 1))
        C_maps[:] = np.nan
        lC_maps.append(C_maps)

        lVM_grid_x.append(np.copy(IW_grid_x))
        lVM_grid_y.append(np.copy(IW_grid_y))
        lVM_dx.append(np.zeros(IW_grid_x.shape))
        lVM_dy.append(np.zeros(IW_grid_x.shape))

        # Create pyFFTW calculation objects
        lfftw.append(
            FFTW_Convolver_Full2D(
                (IW_size, IW_size), (IW_size, IW_size), fftw_threads=1
            )
        )

        if 0:  # DEBUG flag: Examine IW meshgrid
            # fmt: off
            p = {"fillstyle": "none", "markersize": 6, "linewidth": 2}
            plt.figure()
            plt.plot(
                IW_lims_x[:, :, 0].reshape(-1),
                IW_lims_y[:, :, 0].reshape(-1),
                "xg", label="IW starts", **p)
            plt.plot(
                IW_lims_x[:, :, 1].reshape(-1),
                IW_lims_y[:, :, 1].reshape(-1),
                "xr", label="IW ends", **p)

            # Plot IW centers, but not all. Just the bottom and left chords.
            plt.plot(
                IW_grid_x[:, 0],
                IW_grid_y[:, 0],
                "ok", label="IW centers", **p)
            plt.plot(
                IW_grid_x[0, :],
                IW_grid_y[0, :],
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

    # --------------------------------------------------------------------------
    #   Walk over all multigrid stages
    # --------------------------------------------------------------------------
    t_0 = perf_counter()

    for stage_idx, IW_size in enumerate(IW_SIZES):
        pstage_idx = stage_idx - 1  # Previous/parent stage index
        IW_params = lIW_params[stage_idx]
        N_IWs_x = IW_params[2]
        N_IWs_y = IW_params[3]
        N_IWs = N_IWs_x * N_IWs_y

        A_IW_grid_x = lA_IW_grid_x[stage_idx]
        A_IW_grid_y = lA_IW_grid_y[stage_idx]
        A_IW_lims_x = lA_IW_lims_x[stage_idx]
        A_IW_lims_y = lA_IW_lims_y[stage_idx]

        B_IW_grid_x = lB_IW_grid_x[stage_idx]
        B_IW_grid_y = lB_IW_grid_y[stage_idx]
        B_IW_lims_x = lB_IW_lims_x[stage_idx]
        B_IW_lims_y = lB_IW_lims_y[stage_idx]

        IW_shifts_x = lIW_shifts_x[stage_idx]
        IW_shifts_y = lIW_shifts_y[stage_idx]

        C_maps = lC_maps[stage_idx]

        VM_grid_x = lVM_grid_x[stage_idx]
        VM_grid_y = lVM_grid_y[stage_idx]
        VM_dx = lVM_dx[stage_idx]
        VM_dy = lVM_dy[stage_idx]

        fftw = lfftw[stage_idx]

        # Preallocate IW image subset of frames A and B
        img_IW_shape = (IW_size, IW_size)
        img_IW_A = np.zeros(img_IW_shape, dtype=A.dtype)
        img_IW_B = np.zeros(img_IW_shape, dtype=B.dtype)

        # ----------------------------------------------------------------------
        #   Walk over all interrogation windows
        # ----------------------------------------------------------------------

        for (IW_idx_y, IW_idx_x), IW_px_x in np.ndenumerate(A_IW_grid_x):
            IW_px_y = A_IW_grid_y[IW_idx_y, IW_idx_x]
            IW_idx = np.ravel_multi_index(
                (IW_idx_y, IW_idx_x), A_IW_grid_x.shape
            )

            if DEBUG:
                print(
                    f"IW: {IW_idx} of {N_IWs - 1} " f"@px {IW_px_x}, {IW_px_y}"
                )

            # Part of the window shifting mechanism:
            # Undo the shift again when the shifted IW of frame B is leaving the
            # borders of frame B. If so, we will, later on, zero out the
            # appropiate section of the IW of frame B that corresponds to
            # `particles` that are definitely not present in the IW of frame A.
            # Likewise, we will zero out pixels in frame A that are not present
            # in frame B.
            zero_out_L = 0  # left of B , x = 0
            zero_out_R = 0  # right of B, x = IW_size - 1
            zero_out_U = 0  # up of B   , y = 0
            zero_out_D = 0  # down of B , y = IW_size - 1

            # ------------------------------------------------------------------
            #   Calculate IW of frame B
            #   Apply window shifting technique
            # ------------------------------------------------------------------

            if stage_idx == 0:
                # First stage, no pre-shift available
                shift_x = 0  # [px]
                shift_y = 0  # [px]
            else:
                # Pre-shift available: Look up corresponding index of the IW in
                # the larger parent grid
                parent_IW_idx_x, parent_IW_idx_y = lookup_IW_idx(
                    IW_px_x,
                    IW_px_y,
                    lIW_params[pstage_idx],
                )

                # Retrieve the pre-shift
                shift_x = lVM_dx[pstage_idx][parent_IW_idx_y, parent_IW_idx_x]
                shift_y = lVM_dy[pstage_idx][parent_IW_idx_y, parent_IW_idx_x]
                shift_x = 0 if np.isnan(shift_x) else int(shift_x)
                shift_y = 0 if np.isnan(shift_y) else int(shift_y)

                if DEBUG:
                    # Flattened iter index
                    parent_IW_idx = np.ravel_multi_index(
                        (parent_IW_idx_y, parent_IW_idx_x),
                        (lIW_params[pstage_idx][3], lIW_params[pstage_idx][2]),
                    )
                    print(f"   parent IW {parent_IW_idx}")
                    print(f"   shift  {shift_x:+2d}, {shift_y:+2d}", end="")

                # Calculate new center and limits of the shifted IW in frame B
                B_IW_grid_x[IW_idx_y, IW_idx_x] += shift_x
                B_IW_grid_y[IW_idx_y, IW_idx_x] += shift_y
                B_IW_lims_x[IW_idx_y, IW_idx_x, :] += shift_x
                B_IW_lims_y[IW_idx_y, IW_idx_x, :] += shift_y

                # Check and prevent the shift of IW B from moving outside of the
                # source image B. When so, we will zero out part of the IW
                # images that have moved out-of-frame, later on.
                if B_IW_lims_x[IW_idx_y, IW_idx_x, 0] < 0:
                    zero_out_R = np.abs(shift_x)
                    B_IW_grid_x[IW_idx_y, IW_idx_x] -= shift_x
                    B_IW_lims_x[IW_idx_y, IW_idx_x, :] -= shift_x
                    shift_x = 0
                else:
                    zero_out_R = 0

                if B_IW_lims_x[IW_idx_y, IW_idx_x, 1] > img_w - 1:
                    zero_out_L = np.abs(shift_x)
                    B_IW_grid_x[IW_idx_y, IW_idx_x] -= shift_x
                    B_IW_lims_x[IW_idx_y, IW_idx_x, :] -= shift_x
                    shift_x = 0
                else:
                    zero_out_L = 0

                if B_IW_lims_y[IW_idx_y, IW_idx_x, 0] < 0:
                    zero_out_D = np.abs(shift_y)
                    B_IW_grid_y[IW_idx_y, IW_idx_x] -= shift_y
                    B_IW_lims_y[IW_idx_y, IW_idx_x, :] -= shift_y
                    shift_y = 0
                else:
                    zero_out_D = 0

                if B_IW_lims_y[IW_idx_y, IW_idx_x, 1] > img_h - 1:
                    zero_out_U = np.abs(shift_y)
                    B_IW_grid_y[IW_idx_y, IW_idx_x] -= shift_y
                    B_IW_lims_y[IW_idx_y, IW_idx_x, :] -= shift_y
                    shift_y = 0
                else:
                    zero_out_U = 0

                IW_shifts_x[IW_idx_y, IW_idx_x] = shift_x
                IW_shifts_y[IW_idx_y, IW_idx_x] = shift_y

                if DEBUG:
                    if (zero_out_L > 0) or (zero_out_R > 0):
                        print(" , undo x", end="")
                    if (zero_out_U > 0) or (zero_out_D > 0):
                        print(" , undo y", end="")
                    print("")

            # ------------------------------------------------------------------
            #   Retrieve images of IW frame A and IW frame B
            # ------------------------------------------------------------------

            if DEBUG:
                print(
                    "   A_xlim ["
                    f"{A_IW_lims_x[IW_idx_y, IW_idx_x, 0]:4d}, "
                    f"{A_IW_lims_x[IW_idx_y, IW_idx_x, 1]:4d}]"
                )
                print(
                    "   A_ylim ["
                    f"{A_IW_lims_y[IW_idx_y, IW_idx_x, 0]:4d}, "
                    f"{A_IW_lims_y[IW_idx_y, IW_idx_x, 1]:4d}]"
                )
                print(
                    "   B_xlim ["
                    f"{B_IW_lims_x[IW_idx_y, IW_idx_x, 0]:4d}, "
                    f"{B_IW_lims_x[IW_idx_y, IW_idx_x, 1]:4d}]"
                )
                print(
                    "   B_ylim ["
                    f"{B_IW_lims_y[IW_idx_y, IW_idx_x, 0]:4d}, "
                    f"{B_IW_lims_y[IW_idx_y, IW_idx_x, 1]:4d}]"
                )

            # fmt: off
            # We need a copy, because otherwise the upcoming potential zeroing
            # of the IW image borders will affect, by means of reference, the
            # original image.
            np.copyto(
                img_IW_A,
                A[A_IW_lims_y[IW_idx_y, IW_idx_x, 0] :
                  A_IW_lims_y[IW_idx_y, IW_idx_x, 1] + 1,
                  A_IW_lims_x[IW_idx_y, IW_idx_x, 0] :
                  A_IW_lims_x[IW_idx_y, IW_idx_x, 1]+ 1]
            )

            np.copyto(
                img_IW_B,
                B[B_IW_lims_y[IW_idx_y, IW_idx_x, 0] :
                  B_IW_lims_y[IW_idx_y, IW_idx_x, 1] + 1,
                  B_IW_lims_x[IW_idx_y, IW_idx_x, 0] :
                  B_IW_lims_x[IW_idx_y, IW_idx_x, 1] + 1]
            )
            # fmt: on

            # Zero out the appropiate section of the IW of frame B that
            # corresponds to `particles` that are definitely not present in the
            # IW of frame A. Likewise, zero out the IW of frame A.
            if zero_out_L > 0:
                img_IW_B[:, :zero_out_L] = 0
                img_IW_A[:, -zero_out_L:] = 0
            if zero_out_R > 0:
                img_IW_B[:, -zero_out_R:] = 0
                img_IW_A[:, :zero_out_R] = 0
            if zero_out_U > 0:
                img_IW_B[:zero_out_U, :] = 0
                img_IW_A[-zero_out_U:, :] = 0
            if zero_out_D > 0:
                img_IW_B[-zero_out_D:, :] = 0
                img_IW_A[:zero_out_D, :] = 0

            # ------------------------------------------------------------------
            #   Perform cross-correlation
            # ------------------------------------------------------------------

            if (
                img_IW_A.size == 0
                or np.max(img_IW_A) == 0
                or np.max(img_IW_B) == 0
            ):
                # Save computation time
                C_maps[IW_idx_y, IW_idx_x, 0, 0] = np.nan

            else:
                # Perform 2D cross-correlation
                # C_maps[IW_idx_y, IW_idx_x, :, :] = fftconvolve(
                #     img_IW_B, fliplrud(img_IW_A), mode="full"
                # )
                C_maps[IW_idx_y, IW_idx_x, :, :] = fftw.convolve(
                    img_IW_B, fliplrud(img_IW_A)
                )

        # It is not necessary to normalize the correlation maps. Adds overhead.
        # normalize_C_maps(C_maps)  # Not necessary

        compute_displacement_vectors_from_C_maps(
            C_maps,
            IW_shifts_x,
            IW_shifts_y,
            VM_dx,
            VM_dy,
            perform_subpixel_fitting=(stage_idx == N_stages - 1),
        )

        if SHOW_CORRELATION_MAPS:
            # Reset any existing plot of the correlation map, because the IW
            # size has changed and plotting on top of imshow needs a rescale.
            if plt.fignum_exists("C_map"):
                plt.close("C_map")

            # Plotting requires normalizing correlation maps for easy comparison
            normalize_C_maps(C_maps)

            for IW_idx_y in range(C_maps.shape[0]):
                for IW_idx_x in range(C_maps.shape[1]):
                    IW_idx = np.ravel_multi_index(
                        (IW_idx_y, IW_idx_x), A_IW_grid_x.shape
                    )
                    C = C_maps[IW_idx_y, IW_idx_x, :, :]
                    if np.isnan(C[0, 0]):
                        continue

                    dx = VM_dx[IW_idx_y, IW_idx_x]
                    dy = VM_dy[IW_idx_y, IW_idx_x]
                    shift_x = IW_shifts_x[IW_idx_y, IW_idx_x]
                    shift_y = IW_shifts_y[IW_idx_y, IW_idx_x]
                    peak_x = C.shape[1] // 2 + dx - shift_x
                    peak_y = C.shape[0] // 2 + dy - shift_y

                    if DEBUG:
                        print(f"     peak   @ {peak_x:+5.1f}, {peak_y:+5.1f}")
                        print(f"     dx, dy = {dx:+5.1f}, {dy:+5.1f}")

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
                        h_title = plt.title(f"")

                    h_imshow.set_data(C)  # type: ignore
                    h_peak.set_data([peak_x], [peak_y])  # type: ignore
                    h_title.set_text(f"{IW_idx} of {N_IWs}")  # type: ignore

                    plt.draw()
                    plt.pause(0.0001)
                    # plt.waitforbuttonpress()
                    # plt.show(block=False)
                    # plt.show()

    duration = perf_counter() - t_0
    print(f"Finished in {duration:.3f} s")

    # --------------------------------------------------------------------------
    #   Show original image A with unfiltered vector map on top
    # --------------------------------------------------------------------------
    quiverX = 3

    if LOAD_MPL:
        fig = plt.figure()
        plt.imshow(A, cmap="gray", interpolation="none")
        plt.quiver(
            lVM_grid_x[-1],
            lVM_grid_y[-1],
            lVM_dx[-1] * quiverX,
            lVM_dy[-1] * quiverX,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
            linewidths=2,
        )
        plt.show()
