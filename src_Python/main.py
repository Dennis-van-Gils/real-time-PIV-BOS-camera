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
__date__ = "04-08-2023"
__version__ = "1.0"
# pylint: disable=missing-function-docstring

import os
import sys
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

FASTER = False
if FASTER:
    from convolve2d__my_code_faster import FFTW_Convolver_Full2D
else:
    from convolve2d__my_code import FFTW_Convolver_Full2D

DEBUG = False  # Print debug info to terminal?
SHOW_CORRELATION_MAPS = False
LOAD_MPL = True
# if LOAD_MPL:
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

mpl.use("TkAgg")

# Holds the IW sizes for the multigrid analysis. Powers of two are advised with
# each subsequent IW size the exact half of the previous IW size.
IW_SIZES = []  # [px]
IW_OVERLAP = 0.5  # IW overlap fraction [0 - 1]

# ------------------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    if 1:
        fn = "E:/Work/_GitHub_repo/2D-PIV-BOS/test_imgs/PIV_rising_vortex_plume/B00001.tif"
        IW_SIZES = [64, 32]

        # Read double image and split into frames A & B
        img = imread(fn, as_gray=True)
        img_2h, img_w = np.shape(img)
        img_h = int(img_2h / 2)
        A = (img[:img_h, :]).astype(np.float32)
        B = (img[img_h:, :]).astype(np.float32)
    else:
        fn1 = "E:/Work/_GitHub_repo/2D-PIV-BOS/test_imgs/a1.tif"
        fn2 = "E:/Work/_GitHub_repo/2D-PIV-BOS/test_imgs/a2.tif"
        IW_SIZES = [256, 128, 64, 32]

        A = imread(fn1, as_gray=True).astype(np.float32)
        B = imread(fn2, as_gray=True).astype(np.float32)
        img_h, img_w = A.shape

    # --------------------------------------------------------------------------
    #   Init
    # --------------------------------------------------------------------------

    # Preallocate and populate lists for the upcoming multigrid analysis.
    # stage: Current multigrid stage from the largest IW size to the smallest.
    # Prefix 'l' denotes 'list' with index `stage_idx`.

    # List of IW parameters per stage of the multigrid
    #   tuple [IW_size    (``int``),
    #          IW_overlap (``float``),
    #          N_IWs      (``int``),
    #          N_IWs_x    (``int``),
    #          N_IWs_y    (``int``)]
    lIW_params: list[tuple[int, float, int, int, int]] = []

    # fmt: off
    # List of IW meshgrids and limits per stage of the multigrid
    lA_IW_grid_x: list[np.ndarray] = []  # np.ndarray[N_IWs]
    lA_IW_grid_y: list[np.ndarray] = []  # np.ndarray[N_IWs]
    lA_IW_lims_x: list[np.ndarray] = []  # np.ndarray[N_IWs, 2]
    lA_IW_lims_y: list[np.ndarray] = []  # np.ndarray[N_IWs, 2]

    lB_IW_grid_x: list[np.ndarray] = []  # np.ndarray[N_IWs]
    lB_IW_grid_y: list[np.ndarray] = []  # np.ndarray[N_IWs]
    lB_IW_lims_x: list[np.ndarray] = []  # np.ndarray[N_IWs, 2]
    lB_IW_lims_y: list[np.ndarray] = []  # np.ndarray[N_IWs, 2]

    # List of computed IW shifts per stage of the multigrid
    # NOTE: List index 0, which corresponds to `stage_idx = 0`, will be
    # initialized with zeros and remain so, because no window shifts exist for
    # the first multigrid stage by design.
    lIW_shifts_x: list[np.ndarray] = []  # np.ndarray[N_IWs]
    lIW_shifts_y: list[np.ndarray] = []  # np.ndarray[N_IWs]

    # List of computed correlations maps per stage of the multigrid
    #   np.ndarray[N_IWs, IW_size / 2 + 1, IW_size / 2 + 1]
    lC_maps: list[np.ndarray] = []

    # List of computed displacement vector maps per stage of the multigrid
    lVM_grid_x: list[np.ndarray] = []    # np.ndarray[N_IWs]
    lVM_grid_y: list[np.ndarray] = []    # np.ndarray[N_IWs]
    lVM_dx: list[np.ndarray] = []        # np.ndarray[N_IWs]
    lVM_dy: list[np.ndarray] = []        # np.ndarray[N_IWs]
    # fmt: on

    # List of pyFFTW calculation objects per stage of the multigrid
    lfftw: list[FFTW_Convolver_Full2D] = []

    N_stages = len(IW_SIZES)
    for stage_idx, IW_size in enumerate(IW_SIZES):
        # Create interrogation windows
        (
            IW_grid_x,
            IW_grid_y,
            IW_lims_x,
            IW_lims_y,
            N_IWs,
            N_IWs_x,
            N_IWs_y,
        ) = create_IW_grid(img_w, img_h, IW_size, IW_OVERLAP)

        # Populate lists
        lIW_params.append((IW_size, IW_OVERLAP, N_IWs, N_IWs_x, N_IWs_y))

        lA_IW_grid_x.append(np.copy(IW_grid_x))
        lA_IW_grid_y.append(np.copy(IW_grid_y))
        lA_IW_lims_x.append(np.copy(IW_lims_x))
        lA_IW_lims_y.append(np.copy(IW_lims_y))

        lB_IW_grid_x.append(np.copy(IW_grid_x))
        lB_IW_grid_y.append(np.copy(IW_grid_y))
        lB_IW_lims_x.append(np.copy(IW_lims_x))
        lB_IW_lims_y.append(np.copy(IW_lims_y))

        lIW_shifts_x.append(np.zeros(N_IWs))
        lIW_shifts_y.append(np.zeros(N_IWs))

        if FASTER:
            C_maps = np.zeros((N_IWs, IW_size, IW_size))
        else:
            C_maps = np.zeros((N_IWs, IW_size * 2 - 1, IW_size * 2 - 1))
        C_maps[:] = np.nan
        lC_maps.append(C_maps)

        lVM_grid_x.append(np.copy(IW_grid_x))
        lVM_grid_y.append(np.copy(IW_grid_y))
        lVM_dx.append(np.zeros(N_IWs))
        lVM_dy.append(np.zeros(N_IWs))

        # Create pyFFTW calculation objects
        lfftw.append(FFTW_Convolver_Full2D((IW_size, IW_size), fftw_threads=1))

        if 0:  # DEBUG flag: Examine IW meshgrid
            # fmt: off
            p = {"fillstyle": "none", "markersize": 6, "linewidth": 2}
            plt.figure()
            plt.plot(
                IW_lims_x[:, 0],
                IW_lims_y[:, 0],
                "xg", label="IW starts", **p)
            plt.plot(
                IW_lims_x[:, 1],
                IW_lims_y[:, 1],
                "xr", label="IW ends", **p)
            plt.plot(
                IW_grid_x,
                IW_grid_y,
                "ok", label="IW centers", **p)

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
    #   Image preparation
    # --------------------------------------------------------------------------

    # Mean background removal
    remove_mean_background(A)
    remove_mean_background(B)

    # Flip the image of frame A left-to-right and up-to-down ahead of time for
    # the upcoming 2D cross-correlation done via convolution. Doing this ahead
    # of time instead of doing it just before the convolution operation (inside
    # the IW loop) saves many duplicate `fliplrud()` operations on identical
    # data (because of the window overlapping).
    A_ = fliplrud(A)

    # --------------------------------------------------------------------------
    #   Walk over all multigrid stages
    # --------------------------------------------------------------------------

    t_0 = perf_counter()

    for stage_idx, IW_size in enumerate(IW_SIZES):
        # Short-hand variables
        N_IWs = lIW_params[stage_idx][2]
        N_IWs_x = lIW_params[stage_idx][3]
        N_IWs_y = lIW_params[stage_idx][4]
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
        IW_shape = (IW_size, IW_size)
        IW_A_ = np.zeros(IW_shape, dtype=A.dtype)
        IW_B = np.zeros(IW_shape, dtype=B.dtype)

        # ----------------------------------------------------------------------
        #   Walk over all interrogation windows
        # ----------------------------------------------------------------------

        for IW_idx, IW_px_x in enumerate(A_IW_grid_x):
            IW_px_y = A_IW_grid_y[IW_idx]

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
            IW_needs_to_be_a_copy = False

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
                parent_IW_idx = lookup_IW_idx(
                    IW_px_x,
                    IW_px_y,
                    lIW_params[stage_idx - 1],
                )

                # Retrieve the pre-shift
                shift_x = lVM_dx[stage_idx - 1][parent_IW_idx]
                shift_y = lVM_dy[stage_idx - 1][parent_IW_idx]
                shift_x = 0 if np.isnan(shift_x) else int(shift_x)
                shift_y = 0 if np.isnan(shift_y) else int(shift_y)

                # Apply the pre-shift to IW B (eager)
                B_IW_grid_x[IW_idx] += shift_x
                B_IW_grid_y[IW_idx] += shift_y
                B_IW_lims_x[IW_idx, :] += shift_x
                B_IW_lims_y[IW_idx, :] += shift_y

                # Check and prevent the shift of IW B from moving outside of the
                # source image B. When so, we will zero out part of the IW
                # images that have moved out-of-frame, later on.
                if B_IW_lims_x[IW_idx, 0] < 0:
                    IW_needs_to_be_a_copy = True
                    zero_out_R = np.abs(shift_x)
                    B_IW_grid_x[IW_idx] -= shift_x
                    B_IW_lims_x[IW_idx, :] -= shift_x
                    shift_x = 0
                else:
                    zero_out_R = 0

                if B_IW_lims_x[IW_idx, 1] > img_w - 1:
                    IW_needs_to_be_a_copy = True
                    zero_out_L = np.abs(shift_x)
                    B_IW_grid_x[IW_idx] -= shift_x
                    B_IW_lims_x[IW_idx, :] -= shift_x
                    shift_x = 0
                else:
                    zero_out_L = 0

                if B_IW_lims_y[IW_idx, 0] < 0:
                    IW_needs_to_be_a_copy = True
                    zero_out_D = np.abs(shift_y)
                    B_IW_grid_y[IW_idx] -= shift_y
                    B_IW_lims_y[IW_idx, :] -= shift_y
                    shift_y = 0
                else:
                    zero_out_D = 0

                if B_IW_lims_y[IW_idx, 1] > img_h - 1:
                    IW_needs_to_be_a_copy = True
                    zero_out_U = np.abs(shift_y)
                    B_IW_grid_y[IW_idx] -= shift_y
                    B_IW_lims_y[IW_idx, :] -= shift_y
                    shift_y = 0
                else:
                    zero_out_U = 0

                IW_shifts_x[IW_idx] = shift_x
                IW_shifts_y[IW_idx] = shift_y

            # ------------------------------------------------------------------
            #   Retrieve images of IW frame A and IW frame B
            # ------------------------------------------------------------------

            # Note: `A_` is a flipped left-to-right and up-to-down version of
            # `A`, so we have to flip the indices as well, hence the use of
            # `A.shape[] - ...`.
            Ax0 = A.shape[1] - A_IW_lims_x[IW_idx, 1] - 1
            Ax1 = A.shape[1] - A_IW_lims_x[IW_idx, 0]
            Ay0 = A.shape[0] - A_IW_lims_y[IW_idx, 1] - 1
            Ay1 = A.shape[0] - A_IW_lims_y[IW_idx, 0]

            Bx0 = B_IW_lims_x[IW_idx, 0]
            Bx1 = B_IW_lims_x[IW_idx, 1] + 1
            By0 = B_IW_lims_y[IW_idx, 0]
            By1 = B_IW_lims_y[IW_idx, 1] + 1

            # fmt: off
            if IW_needs_to_be_a_copy:
                # We need a copy, because otherwise the upcoming zeroing of the
                # IW image borders will affect, by means of reference, the
                # original image and interfere with the correlation of upcoming
                # and overlapping IWs. Copying adds a tiny cpu overhead.
                np.copyto(IW_A_, A_[Ay0:Ay1, Ax0:Ax1])
                np.copyto(IW_B , B [By0:By1, Bx0:Bx1])

                # Zero out the appropiate section of the IW of frame B that
                # corresponds to `particles` that are definitely not present in
                # the IW of frame A. Likewise, zero out the IW of frame A. Zero
                # caries the meaning of being at the mean background level of
                # the image.
                if zero_out_L > 0:
                    IW_B [:, :zero_out_L] = 0
                    IW_A_[:, :zero_out_L] = 0
                if zero_out_R > 0:
                    IW_B [:, -zero_out_R:] = 0
                    IW_A_[:, -zero_out_R:] = 0
                if zero_out_U > 0:
                    IW_B [:zero_out_U, :] = 0
                    IW_A_[:zero_out_U, :] = 0
                if zero_out_D > 0:
                    IW_B [-zero_out_D:, :] = 0
                    IW_A_[-zero_out_D:, :] = 0
            else:
                IW_A_ = A_[Ay0:Ay1, Ax0:Ax1]  # Pass by reference
                IW_B  = B [By0:By1, Bx0:Bx1]  # Pass by reference
            # fmt: on

            # ------------------------------------------------------------------
            #   Perform cross-correlation
            # ------------------------------------------------------------------

            if (np.max(IW_A_) <= 0) and (np.max(IW_B) <= 0):
                # No details are present in the IW images. All pixels are below
                # or at the mean background --> Save computation time.
                # TODO: Make this a user config threshold? Is <= 0 even correct?
                # Must match up with 'zeroing out' mechanism, just above here.
                C_maps[IW_idx, 0, 0] = np.nan

            else:
                # Perform 2D cross-correlation
                # C_maps[IW_idx, :, :] = fftconvolve(IW_B, IW_A_, mode="full")
                C_maps[IW_idx, :, :] = fftw.convolve(IW_B, IW_A_)

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

    duration = perf_counter() - t_0
    print(f"Finished in {duration:.3f} s")

    # --------------------------------------------------------------------------
    #   Debugging output
    # --------------------------------------------------------------------------

    for stage_idx, IW_size in enumerate(IW_SIZES):
        N_IWs = lIW_params[stage_idx][2]
        N_IWs_x = lIW_params[stage_idx][3]
        N_IWs_y = lIW_params[stage_idx][4]
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

        if SHOW_CORRELATION_MAPS:
            # Reset any existing plot of the correlation map, because the IW
            # size has changed and plotting on top of imshow needs a rescale.
            if plt.fignum_exists("C_map"):
                plt.close("C_map")

            # Plotting requires normalizing correlation maps for easy comparison
            normalize_C_maps(C_maps)

        for IW_idx, IW_px_x in enumerate(A_IW_grid_x):
            # NOTE: Information on potentially zeroed-out sections inside
            # `IW_A` and `IW_B` is not stored nor accessible here.
            # Variables `zero_out_L/R/U/D` have not been stored to memory to
            # save on cpu time.

            # Short-hand variables
            IW_px_y = A_IW_grid_y[IW_idx]
            shift_x = IW_shifts_x[IW_idx]
            shift_y = IW_shifts_y[IW_idx]
            C = C_maps[IW_idx]

            # TODO: For proper debugging, I need to store the found correlations
            # peaks in a list of arrays, too.
            dx = VM_dx[IW_idx]
            dy = VM_dy[IW_idx]
            shift_x = np.nan_to_num(IW_shifts_x[IW_idx])  # NaN will become 0
            shift_y = np.nan_to_num(IW_shifts_y[IW_idx])
            qx = 1 if (C.shape[0] % 2) == 0 else 0
            qy = 1 if (C.shape[1] % 2) == 0 else 0
            peak_x = dx + C.shape[1] // 2 - qx - shift_x  # TODO: store & get,
            peak_y = dy + C.shape[0] // 2 - qy - shift_y  # do not calc again

            if DEBUG:
                print(
                    f"IW: {IW_idx} of {N_IWs - 1} " f"@px {IW_px_x}, {IW_px_y}"
                )

                if stage_idx > 0:
                    parent_IW_idx = lookup_IW_idx(
                        IW_px_x,
                        IW_px_y,
                        lIW_params[stage_idx - 1],
                    )
                    print(f"   parent IW {parent_IW_idx}")
                    print(f"   shift  {shift_x:+2.0f}, {shift_y:+2.0f}")

                print(
                    "   A_xlim ["
                    f"{A_IW_lims_x[IW_idx, 0]:4d}, "
                    f"{A_IW_lims_x[IW_idx, 1]:4d}]"
                )
                print(
                    "   A_ylim ["
                    f"{A_IW_lims_y[IW_idx, 0]:4d}, "
                    f"{A_IW_lims_y[IW_idx, 1]:4d}]"
                )
                print(
                    "   B_xlim ["
                    f"{B_IW_lims_x[IW_idx, 0]:4d}, "
                    f"{B_IW_lims_x[IW_idx, 1]:4d}]"
                )
                print(
                    "   B_ylim ["
                    f"{B_IW_lims_y[IW_idx, 0]:4d}, "
                    f"{B_IW_lims_y[IW_idx, 1]:4d}]"
                )

                if not np.isnan(C[0, 0]):
                    print(f"     peak   @ {peak_x:+5.1f}, {peak_y:+5.1f}")
                    print(f"     dx, dy = {dx:+5.1f}, {dy:+5.1f}")

            if SHOW_CORRELATION_MAPS:
                if not np.isnan(C[0, 0]):
                    if not (plt.fignum_exists("C_map")):
                        fig = plt.figure("C_map")
                        h_imshow = plt.imshow(
                            np.zeros(C.shape),
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

    # --------------------------------------------------------------------------
    #   Show original image A with unfiltered vector map on top
    # --------------------------------------------------------------------------
    quiverX = 3

    if LOAD_MPL:
        grid_x = lVM_grid_x[-1]
        grid_y = lVM_grid_y[-1]
        VM_dx = np.copy(lVM_dx[-1])
        VM_dy = np.copy(lVM_dy[-1])

        # Vector magnitude
        M = np.sqrt(np.square(VM_dx) + np.square(VM_dy))

        # Threshold on vector magnitude
        # VM_dx[M < 0.5] = np.nan
        # VM_dy[M < 0.5] = np.nan

        fig = plt.figure()
        plt.imshow(A, cmap="gray", interpolation="none")
        plt.quiver(
            grid_x,
            grid_y,
            VM_dx * quiverX,
            VM_dy * quiverX,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
            linewidths=2,
        )
        plt.show()
