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
__date__ = "20-09-2023"
__version__ = "1.0"
# pylint: disable=missing-function-docstring

import glob
from time import perf_counter
import concurrent.futures

import numpy as np
import numpy.typing as npt
from skimage.io import imread
from scipy import fft as sp_fft

# from dvg_fftw_convolve2d import FFTW_Convolver_Full2D
from dvg_rocketfftw_convolve2d import FFTW_Convolver_Full2D
from my_fun import (
    get_filename_from_full_path,
    remove_mean_background,
    create_IW_grid,
    lookup_IW_idx,
    fliplrud,
    normalize_C_maps,
    compute_displacement_vectors_from_C_maps,
)
from process_IWs import process_IWs

DEBUG = False  # Print debug info to terminal?
SHOW_CORRELATION_MAPS = False
LOAD_MPL = True

if LOAD_MPL:
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.use("TkAgg")

# Holds the IW sizes for the multigrid analysis. Powers of two are advised with
# each subsequent IW size the exact half of the previous IW size.
IW_SIZES = []  # [px]
IW_OVERLAP = 0.5  # IW overlap fraction [0 - 1]

# Number of concurrent workers. Each worker will process a chunk of all the IWs
# over which the 2D correlations are calculated. The chunks will get (near)
# evenly divided over all workers.
N_WORKERS = 16

N_FFTW_THREADS = 1

# Info: Release GIL in pyFFTW also when argument `threads=1`.
# https://github.com/pyFFTW/pyFFTW/commit/0fed0706aaa6898913d4e8716d7eb3f9a4a1351d

# ------------------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    piv_set = 0

    if piv_set == 0:
        path = r"../test_imgs/PIV_rising_vortex_plume/*.png"
        IW_SIZES = [64, 32]
        quiver_size = 3
        color_div = 14
    elif piv_set == 1:
        path = r"../test_imgs/swirling_vortices/*.tif"
        IW_SIZES = [256, 128, 64, 32]
        quiver_size = 3
        color_div = 24
    else:
        path = r"../test_imgs/4th_PIV-Challenge_Case_E/*.tif"
        IW_SIZES = [64, 32]
        quiver_size = 8
        color_div = 4

    img_files = glob.glob(path)
    N_img_files = len(img_files)

    if DEBUG:  # Overrule: Only process the first image pair
        N_img_files = 2

    # Read first image to get image width and height
    A = imread(img_files[0], as_gray=True)
    A = np.asarray(A, dtype=np.float32, order="C")
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
    lIW_grid_x: list[npt.NDArray[np.int32]] = []    # NDArray shape (N_IWs, )
    lIW_grid_y: list[npt.NDArray[np.int32]] = []    # NDArray shape (N_IWs, )
    lIW_lims_x: list[npt.NDArray[np.int32]] = []    # NDArray shape (N_IWs, 2)
    lIW_lims_y: list[npt.NDArray[np.int32]] = []    # NDArray shape (N_IWs, 2)

    lA_IW_grid_x: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, )
    lA_IW_grid_y: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, )
    lA_IW_lims_x: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, 2)
    lA_IW_lims_y: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, 2)

    lB_IW_grid_x: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, )
    lB_IW_grid_y: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, )
    lB_IW_lims_x: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, 2)
    lB_IW_lims_y: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, 2)

    # List of computed IW shifts per stage of the multigrid
    # NOTE: List index 0, which corresponds to `stage_idx = 0`, will be
    # initialized with zeros and remain so throughout, because no window shifts
    # ever exist for the first multigrid stage by design. Thats okay.
    lIW_shifts_x: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, )
    lIW_shifts_y: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, )

    # List of computed correlations maps per stage of the multigrid
    #   NDArray shape (N_IWs, IW_size * 2 - 1, IW_size * 2 - 1)
    lC_maps: list[npt.NDArray[np.float32]] = []

    # List of computed displacement vector maps per stage of the multigrid
    lVM_grid_x: list[npt.NDArray[np.int32]] = []    # NDArray shape (N_IWs, )
    lVM_grid_y: list[npt.NDArray[np.int32]] = []    # NDArray shape (N_IWs, )
    lVM_dx: list[npt.NDArray[np.float32]] = []      # NDArray shape (N_IWs, )
    lVM_dy: list[npt.NDArray[np.float32]] = []      # NDArray shape (N_IWs, )
    # fmt: on

    # List of pyFFTW calculation instances per stage of the multigrid. In
    # addition, each stage will have a multiple of identical pyFFTW instances
    # equal to the number of concurrent workers set in `N_WORKERS`.
    lfftw: list[list[FFTW_Convolver_Full2D]] = []

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

        lIW_grid_x.append(np.copy(IW_grid_x))
        lIW_grid_y.append(np.copy(IW_grid_y))
        lIW_lims_x.append(np.copy(IW_lims_x))
        lIW_lims_y.append(np.copy(IW_lims_y))

        lA_IW_grid_x.append(np.copy(IW_grid_x))
        lA_IW_grid_y.append(np.copy(IW_grid_y))
        lA_IW_lims_x.append(np.copy(IW_lims_x))
        lA_IW_lims_y.append(np.copy(IW_lims_y))

        lB_IW_grid_x.append(np.copy(IW_grid_x))
        lB_IW_grid_y.append(np.copy(IW_grid_y))
        lB_IW_lims_x.append(np.copy(IW_lims_x))
        lB_IW_lims_y.append(np.copy(IW_lims_y))

        lIW_shifts_x.append(np.zeros(N_IWs, dtype=np.int32))
        lIW_shifts_y.append(np.zeros(N_IWs, dtype=np.int32))

        C_maps = np.empty(
            (N_IWs, IW_size * 2 - 1, IW_size * 2 - 1), dtype=np.float32
        )
        C_maps[:] = np.nan
        lC_maps.append(C_maps)

        lVM_grid_x.append(np.copy(IW_grid_x))
        lVM_grid_y.append(np.copy(IW_grid_y))
        lVM_dx.append(np.zeros(N_IWs, dtype=np.float32))
        lVM_dy.append(np.zeros(N_IWs, dtype=np.float32))

        # Create pyFFTW calculation objects
        fftw_workers = []
        for worker_idx in range(N_WORKERS):
            fftw_workers.append(
                FFTW_Convolver_Full2D(
                    (IW_size, IW_size),
                    fftw_threads=N_FFTW_THREADS,
                )
            )
        lfftw.append(fftw_workers)

    # --------------------------------------------------------------------------
    #   Walk over all image pairs
    # --------------------------------------------------------------------------

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS)

    for file_idx in range(0, N_img_files - 1, 2):
        fn1 = img_files[file_idx]
        fn2 = img_files[file_idx + 1]
        print(get_filename_from_full_path(fn1))

        # Reset
        for stage_idx in range(N_stages):
            lB_IW_grid_x[stage_idx] = np.copy(lIW_grid_x[stage_idx])
            lB_IW_grid_y[stage_idx] = np.copy(lIW_grid_y[stage_idx])
            lB_IW_lims_x[stage_idx] = np.copy(lIW_lims_x[stage_idx])
            lB_IW_lims_y[stage_idx] = np.copy(lIW_lims_y[stage_idx])

            """
            # `lA_IW_grid_x/y` remain constant and do not need a reset.
            # `lA_IW_lims_x/y` remain constant and do not need a reset.

            # Reset not strictly necessary as all cells will get updated
            # one-by-one. Reset only to make debugging easier.
            lIW_shifts_x[stage_idx].fill(0)
            lIW_shifts_y[stage_idx].fill(0)

            # Reset not strictly necessary as all cells will get updated at
            # once. Reset only to make debugging easier.
            lVM_dx[stage_idx][:].fill(0)
            lVM_dy[stage_idx][:].fill(0)

            # Reset not strictly necessary as all cells will get updated
            # one-by-one. Reset only to make debugging easier.
            lC_maps[stage_idx][:].fill(np.nan)
            """

        # ----------------------------------------------------------------------
        #   Image preparation
        # ----------------------------------------------------------------------

        A = imread(fn1, as_gray=True)
        B = imread(fn2, as_gray=True)

        # Enforce type and order
        A = np.asarray(A, dtype=np.float32, order="C")
        B = np.asarray(B, dtype=np.float32, order="C")

        # Mean background removal
        remove_mean_background(A)
        remove_mean_background(B)

        # Flip the image of frame A left-to-right and up-to-down ahead of time
        # as needed for the upcoming 2D cross-correlation done via convolution.
        # Doing this ahead of time instead of doing it inside `process_IWs()`
        # for each individual IW saves many duplicate `fliplrud()` operations on
        # identical data because of the window overlapping.
        A_ = np.asarray(fliplrud(A), order="C")

        # ----------------------------------------------------------------------
        #   Walk over all multigrid stages
        # ----------------------------------------------------------------------

        t_0 = perf_counter()

        for stage_idx, IW_size in enumerate(IW_SIZES):
            # Short-hand variables
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
            fftw_workers = lfftw[stage_idx]

            # ------------------------------------------------------------------
            #   Calculate shapes for upcoming fftconvolve
            # ------------------------------------------------------------------
            # TODO: Optimize. Right now is very lazy copy-paste from obsolete
            # class `dvg_rocketfftw_convolve2d.FFTW_Convolver_Full2D`

            s1 = (IW_size, IW_size)
            s2 = (IW_size, IW_size)
            shape = (IW_size * 2 - 1, IW_size * 2 - 1)

            # Speed up FFT by padding to optimal size.
            fshape = (
                sp_fft.next_fast_len(shape[0]),
                sp_fft.next_fast_len(shape[1]),
            )
            fshape_out = (fshape[0], fshape[1] // 2 + 1)

            # Slice corresponding to the 'full' convolution elements to be
            # finally returned as convolution result
            fslice = tuple([slice(sz) for sz in shape])

            # ------------------------------------------------------------------
            #   Walk over all interrogation windows and compute the 2D
            #   correlation maps
            # ------------------------------------------------------------------

            # (Near) evenly distribute all IWs over all workers
            # IWs_slices = []
            idx_starts = np.zeros(N_WORKERS, dtype=int)
            idx_stops = np.zeros(N_WORKERS, dtype=int)
            N_IWs = lIW_params[stage_idx][2]
            for i in range(N_WORKERS):
                idx_start = int(np.floor(N_IWs / N_WORKERS * i))
                idx_stop = int(np.floor(N_IWs / N_WORKERS * (i + 1)))
                # IWs_slices.append(slice(idx_start, idx_stop))
                idx_starts[i] = idx_start
                idx_stops[i] = idx_stop

            p = (
                stage_idx,
                A_,
                B,
                lIW_params[stage_idx - 1],
                lVM_dx[stage_idx - 1],
                lVM_dy[stage_idx - 1],
                A_IW_grid_x,
                A_IW_grid_y,
                A_IW_lims_x,
                A_IW_lims_y,
                B_IW_grid_x,
                B_IW_grid_y,
                B_IW_lims_x,
                B_IW_lims_y,
                IW_shifts_x,
                IW_shifts_y,
                C_maps,
            )

            futures = []
            for worker_idx in range(N_WORKERS):
                futures.append(
                    executor.submit(
                        process_IWs,
                        *p,
                        # fftw=fftw_workers[worker_idx],
                        # IWs_slice=IWs_slices[worker_idx],
                        IWs_start=idx_starts[worker_idx],
                        IWs_stop=idx_stops[worker_idx],
                        fshape=fshape,
                        fshape_out=fshape_out,
                        fslice=fslice,
                    )
                )

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)

            # process_IWs(
            #     stage_idx,
            #     A_,
            #     B,
            #     lIW_params[stage_idx - 1],
            #     lVM_dx[stage_idx - 1],
            #     lVM_dy[stage_idx - 1],
            #     A_IW_grid_x,
            #     A_IW_grid_y,
            #     A_IW_lims_x,
            #     A_IW_lims_y,
            #     B_IW_grid_x,
            #     B_IW_grid_y,
            #     B_IW_lims_x,
            #     B_IW_lims_y,
            #     IW_shifts_x,
            #     IW_shifts_y,
            #     C_maps,
            #     fftw,
            # )

            # ------------------------------------------------------------------
            #   Compute displacement vectors
            # ------------------------------------------------------------------

            # It is not necessary to normalize the correlation maps
            # normalize_C_maps(C_maps)  # Not necessary, adds overhead

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

        # ----------------------------------------------------------------------
        #   Debugging output
        # ----------------------------------------------------------------------

        for stage_idx, IW_size in enumerate(IW_SIZES):
            N_IWs = lIW_params[stage_idx][2]
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

            if SHOW_CORRELATION_MAPS and LOAD_MPL:
                # Reset any existing plot of the correlation map, because the IW
                # size has changed and plotting on top of imshow needs a rescale.
                if plt.fignum_exists("C_map"):  # type: ignore
                    plt.close("C_map")  # type: ignore

                # Plotting requires normalizing correlation maps for easy comparison
                normalize_C_maps(C_maps)

            for IW_idx in range(N_IWs):
                # NOTE: Information on potentially zeroed-out sections inside
                # `IW_A` and `IW_B` is not stored nor accessible here.
                # Variables `zero_out_L/R/U/D` have not been stored to memory to
                # save on cpu time.

                # Short-hand variables
                IW_px_x = A_IW_grid_x[IW_idx]
                IW_px_y = A_IW_grid_y[IW_idx]
                shift_x = IW_shifts_x[IW_idx]
                shift_y = IW_shifts_y[IW_idx]
                C = C_maps[IW_idx]

                # TODO: For proper debugging, I should store the found correlations
                # peaks in a list of arrays, too. Now, I will have to calculate
                # `peak_x` and `peak_y` backwards again.
                dx = VM_dx[IW_idx]
                dy = VM_dy[IW_idx]
                shift_x = IW_shifts_x[IW_idx]
                shift_y = IW_shifts_y[IW_idx]
                qx = 1 if (C.shape[0] % 2) == 0 else 0
                qy = 1 if (C.shape[1] % 2) == 0 else 0
                peak_x = (
                    dx + C.shape[1] // 2 - qx - shift_x
                )  # TODO: store & get,
                peak_y = (
                    dy + C.shape[0] // 2 - qy - shift_y
                )  # do not calc again

                if DEBUG:
                    print(
                        f"IW: {IW_idx} of {N_IWs - 1} "
                        f"@px {IW_px_x}, {IW_px_y}"
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

                if SHOW_CORRELATION_MAPS and LOAD_MPL:
                    if not np.isnan(C[0, 0]):
                        if not (plt.fignum_exists("C_map")):  # type: ignore
                            fig = plt.figure("C_map")  # type: ignore
                            h_imshow = plt.imshow(  # type: ignore
                                C,
                                cmap="gray",
                                interpolation="none",
                                vmin=0,
                                vmax=1,
                            )
                            (h_peak,) = plt.plot([peak_x], [peak_y], "xr")  # type: ignore
                            h_title = plt.title(f"{IW_idx} of {N_IWs}")  # type: ignore

                        else:
                            h_imshow.set_data(C)  # type: ignore
                            h_peak.set_data([peak_x], [peak_y])  # type: ignore
                            h_title.set_text(f"{IW_idx} of {N_IWs}")  # type: ignore

                        plt.draw()  # type: ignore
                        plt.pause(0.0001)  # type: ignore
                        # plt.waitforbuttonpress()  # type: ignore
                        # plt.show(block=False)  # type: ignore
                        # plt.show()  # type: ignore

        # --------------------------------------------------------------------------
        #   Show original image A with unfiltered vector map on top
        # --------------------------------------------------------------------------

        if LOAD_MPL:
            grid_x = lVM_grid_x[-1]
            grid_y = lVM_grid_y[-1]
            VM_dx = lVM_dx[-1]
            VM_dy = lVM_dy[-1]

            # Vector magnitude
            M = np.sqrt(np.square(VM_dx) + np.square(VM_dy))

            # Threshold on vector magnitude
            # VM_dx[M < 0.5] = np.nan
            # VM_dy[M < 0.5] = np.nan

            colors = M / color_div
            colormap = mpl.cm.jet  # type: ignore

            if not (plt.fignum_exists("VM")):  # type: ignore
                fig = plt.figure("VM")  # type: ignore
                h_imshow = plt.imshow(A, cmap="gray", interpolation="none")  # type: ignore
                h_quiver = plt.quiver(  # type: ignore
                    grid_x,
                    grid_y,
                    np.zeros(VM_dx.shape),
                    np.zeros(VM_dy.shape),
                    angles="xy",
                    scale_units="xy",
                    scale=2,
                    # color="r",
                    color=colormap(colors),
                    linewidths=1,
                )
                h_title = plt.title(f"{get_filename_from_full_path(fn1)}")  # type: ignore

            h_imshow.set_data(A)  # type: ignore
            h_quiver.set_UVC(VM_dx * quiver_size, VM_dy * quiver_size)  # type: ignore
            h_quiver.set_color(colormap(colors))  # type: ignore
            h_title.set_text(f"{get_filename_from_full_path(fn1)}")  # type: ignore

            plt.draw()  # type: ignore
            plt.pause(0.0001)  # type: ignore

            if DEBUG:
                plt.waitforbuttonpress()  # type: ignore
                # plt.show()  # type: ignore

    executor.shutdown()
