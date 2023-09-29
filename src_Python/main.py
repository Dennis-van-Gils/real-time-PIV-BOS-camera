#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image processing algorithm for 2D Particle imaging velocimetry (PIV) and
Background-oriented Schlieren (BOS)

Used abbrevations
-----------------
IW: Interrogation window
VM: Displacement vector map
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "29-09-2023"
__version__ = "1.0"
# pylint: disable=missing-function-docstring

import os
import sys
import platform
import concurrent.futures
from time import perf_counter

import psutil
import numpy as np
import numpy.typing as npt
from skimage.io import imread

# The import order is important. We must handle the user configuration first.
import init_config as cfg

# Parse command line arguments.
# Expecting None or the filename of the configuration file to read in.
if len(sys.argv) > 1:
    config_filename = sys.argv[1]
else:
    config_filename = None
cfg.read_file(config_filename)

# Now we can import the remaining modules
from process_IWs import process_IWs
from output_debug_info import output_debug_info
from my_fun import (
    get_filename_from_full_path,
    remove_mean_background,
    create_IW_grid,
    fliplrud,
    compute_displacement_vectors_from_C_maps,
)

if cfg.FFT_LIB == cfg.FFT_LIBS.PYFFTW:
    from dvg_fftconvolver_pyfftw import FFT_Convolver2D_Full
elif cfg.FFT_LIB == cfg.FFT_LIBS.ROCKETFFT:
    from dvg_fftconvolver_rocketfft import FFT_Convolver2D_Full
elif cfg.FFT_LIB == cfg.FFT_LIBS.SCIPY:
    from dvg_fftconvolver_scipy import FFT_Convolver2D_Full
else:
    from dvg_fftconvolver_rocketfft import FFT_Convolver2D_Full

if cfg.LOAD_MPL:
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.use("TkAgg")

# ------------------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Set maximum process priority in the OS
    try:
        proc = psutil.Process(os.getpid())
        if os.name == "nt":  # Windows
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
        else:  # Other
            proc.nice(10)
    except:  # pylint: disable=bare-except
        print("Warning: Could not set process to high priority.")

    # Display info
    print(f"Running on computer `{platform.node()}`")
    print(f"  CPU: {platform.processor()}")
    print(f"  OS : {platform.system()}")
    print("\nConfiguration")
    print(f"  MODE         : {cfg.MODE.name}")
    print(f"  FFT LIB      : {cfg.FFT_LIB.name}")
    print(f"  N_WORKERS    : {cfg.N_WORKERS}")
    print(f"  N_FFT_THREADS: {cfg.N_FFT_THREADS}")
    print(f"  IW_SIZES     : {cfg.IW_SIZES}")
    print(f"  IW_OVERLAP   : {cfg.IW_OVERLAP}\n")
    print("Setting up... ", end="")
    sys.stdout.flush()
    t_0 = perf_counter()

    # List of image files
    img_files = cfg.IMAGE_FILES
    N_img_files = len(img_files)

    if cfg.DEBUG:  # Overrule: Only process the first image pair
        N_img_files = 2

    # Read first image to get image width, height and bit depth
    A = imread(img_files[0], as_gray=True)
    img_h, img_w = A.shape
    img_bit_depth = A[0, 0].nbytes * 8
    img_max_value = 2**img_bit_depth - 1

    # --------------------------------------------------------------------------
    #   Init - preallocate
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

    # List of FFT calculation instances per stage of the multigrid. In
    # addition, each stage will have a multiple of identical FFT instances
    # equal to the number of concurrent workers set in `cfg.N_WORKERS`.
    lfft: list[list[FFT_Convolver2D_Full]] = []

    # --------------------------------------------------------------------------
    #   Init - populate
    # --------------------------------------------------------------------------

    N_stages = len(cfg.IW_SIZES)
    for stage_idx, IW_size in enumerate(cfg.IW_SIZES):
        # Create interrogation windows. Only the last multigrid stage will have
        # window overlapping applied to it.
        IW_overlap = cfg.IW_OVERLAP if stage_idx == N_stages - 1 else 0.0
        (
            IW_grid_x,
            IW_grid_y,
            IW_lims_x,
            IW_lims_y,
            N_IWs,
            N_IWs_x,
            N_IWs_y,
        ) = create_IW_grid(img_w, img_h, IW_size, IW_overlap)

        # Populate lists
        lIW_params.append((IW_size, IW_overlap, N_IWs, N_IWs_x, N_IWs_y))

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
            (N_IWs, IW_size * 2 - 1, IW_size * 2 - 1),
            dtype=np.float32,
        )
        C_maps[:] = np.nan
        lC_maps.append(C_maps)

        lVM_grid_x.append(np.copy(IW_grid_x))
        lVM_grid_y.append(np.copy(IW_grid_y))
        lVM_dx.append(np.zeros(N_IWs, dtype=np.float32))
        lVM_dy.append(np.zeros(N_IWs, dtype=np.float32))

        # Create FFT calculation objects
        fft_workers = []
        for worker_idx in range(cfg.N_WORKERS):
            fft_workers.append(
                FFT_Convolver2D_Full(
                    (IW_size, IW_size),
                    (IW_size, IW_size),
                    fft_threads=cfg.N_FFT_THREADS,
                )
            )
        lfft.append(fft_workers)

    # Force-trigger an eager numba compilation to take the compilation time of
    # function `process_IWs()` out of the timeit results.
    process_IWs(
        0,
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        lIW_params[0],
        lVM_dx[0],
        lVM_dy[0],
        lA_IW_grid_x[0],
        lA_IW_grid_y[0],
        lA_IW_lims_x[0],
        lA_IW_lims_y[0],
        lB_IW_grid_x[0],
        lB_IW_grid_y[0],
        lB_IW_lims_x[0],
        lB_IW_lims_y[0],
        lIW_shifts_x[0],
        lIW_shifts_y[0],
        lC_maps[0],
        lfft[0][0],
        slice(0, 0),
    )

    # Display info
    print(f"done in {perf_counter() - t_0:.3f} sec\n")

    # --------------------------------------------------------------------------
    #   Walk over all image pairs
    # --------------------------------------------------------------------------

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.N_WORKERS)

    for file_idx in range(
        0, N_img_files - 1, 2 if cfg.MODE == cfg.MODES.PIV2 else 1
    ):
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

        # Display info
        print("Reading, processing... ", end="")
        sys.stdout.flush()
        t_0 = perf_counter()

        # Read images
        A = imread(fn1, as_gray=True)
        B = imread(fn2, as_gray=True)
        A = np.asarray(A, dtype=np.float32, order="C")
        B = np.asarray(B, dtype=np.float32, order="C")
        A = A / img_max_value
        B = B / img_max_value
        remove_mean_background(A)
        remove_mean_background(B)

        # Flip the image of frame A left-to-right and up-to-down ahead of time
        # as needed for the upcoming 2D cross-correlation done via convolution.
        # Doing this ahead of time instead of doing it inside `process_IWs()`
        # for each individual IW saves many duplicate `fliplrud()` operations on
        # identical data because of the window overlapping.
        A_ = np.asarray(fliplrud(A), order="C")

        # Display info
        print(f"done in {perf_counter() - t_0:.3f}, ", end="")
        sys.stdout.flush()
        t_0 = perf_counter()

        # ----------------------------------------------------------------------
        #   Walk over all multigrid stages
        # ----------------------------------------------------------------------

        for stage_idx, IW_size in enumerate(cfg.IW_SIZES):
            N_IWs = lIW_params[stage_idx][2]

            # ------------------------------------------------------------------
            #   Walk over all interrogation windows
            # ------------------------------------------------------------------

            # Evenly distribute the IWs over all concurrent workers
            IWs_slices = []
            for worker_idx in range(cfg.N_WORKERS):
                IWs_slices.append(
                    slice(
                        int(np.floor(N_IWs / cfg.N_WORKERS * worker_idx)),
                        int(np.floor(N_IWs / cfg.N_WORKERS * (worker_idx + 1))),
                    )
                )

            # Spawn workers, each performing 2D convolutions on a slice of IWs
            # fmt: off
            futures = []
            for worker_idx in range(cfg.N_WORKERS):
                futures.append(
                    executor.submit(
                        process_IWs,
                        stage_idx,
                        A_,
                        B,
                        lIW_params  [stage_idx - 1],
                        lVM_dx      [stage_idx - 1],
                        lVM_dy      [stage_idx - 1],
                        lA_IW_grid_x[stage_idx],
                        lA_IW_grid_y[stage_idx],
                        lA_IW_lims_x[stage_idx],
                        lA_IW_lims_y[stage_idx],
                        lB_IW_grid_x[stage_idx],
                        lB_IW_grid_y[stage_idx],
                        lB_IW_lims_x[stage_idx],
                        lB_IW_lims_y[stage_idx],
                        lIW_shifts_x[stage_idx],
                        lIW_shifts_y[stage_idx],
                        lC_maps     [stage_idx],
                        lfft        [stage_idx][worker_idx],
                        IWs_slices  [worker_idx],
                    )
                )
            # fmt: on

            # Wait for all workers to complete
            concurrent.futures.wait(futures)

            # ------------------------------------------------------------------
            #   Compute displacement vector maps
            # ------------------------------------------------------------------

            # It is not necessary to normalize the correlation maps
            # normalize_C_maps(C_maps)  # Not necessary, adds overhead

            # fmt: off
            compute_displacement_vectors_from_C_maps(
                lC_maps     [stage_idx],
                lIW_shifts_x[stage_idx],
                lIW_shifts_y[stage_idx],
                lVM_dx      [stage_idx],
                lVM_dy      [stage_idx],
                perform_subpixel_fitting=(stage_idx == N_stages - 1),
            )
            # fmt: on

        # Display info
        duration = perf_counter() - t_0
        print(f"{duration:.3f} sec")

        # ----------------------------------------------------------------------
        #   Debugging output
        # ----------------------------------------------------------------------

        if (cfg.SHOW_CORRELATION_MAPS and cfg.LOAD_MPL) or cfg.DEBUG:
            output_debug_info(
                lIW_params,
                lA_IW_grid_x,
                lA_IW_grid_y,
                lA_IW_lims_x,
                lA_IW_lims_y,
                lB_IW_grid_x,
                lB_IW_grid_y,
                lB_IW_lims_x,
                lB_IW_lims_y,
                lIW_shifts_x,
                lIW_shifts_y,
                lC_maps,
                lVM_grid_x,
                lVM_grid_y,
                lVM_dx,
                lVM_dy,
            )

        # ----------------------------------------------------------------------
        #   Show original image A with unfiltered vector map on top
        # ----------------------------------------------------------------------

        if cfg.LOAD_MPL:
            grid_x = lVM_grid_x[-1]
            grid_y = lVM_grid_y[-1]
            VM_dx = lVM_dx[-1]
            VM_dy = lVM_dy[-1]

            # Vector magnitude
            M = np.sqrt(np.square(VM_dx) + np.square(VM_dy))

            # Threshold on vector magnitude
            # VM_dx[M < 0.5] = np.nan
            # VM_dy[M < 0.5] = np.nan

            colors = M / cfg.COLOR_DIV
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
                    scale=1,  # Scales down by `scale`
                    # color="r",
                    color=colormap(colors),
                    linewidths=1,
                )
                h_title = plt.title(f"{get_filename_from_full_path(fn1)}")  # type: ignore

            h_imshow.set_data(A)  # type: ignore
            h_quiver.set_UVC(VM_dx * cfg.QUIVER_SIZE, VM_dy * cfg.QUIVER_SIZE)  # type: ignore
            h_quiver.set_color(colormap(colors))  # type: ignore
            h_title.set_text(f"{get_filename_from_full_path(fn1)}")  # type: ignore

            plt.draw()  # type: ignore
            plt.pause(0.0001)  # type: ignore

            if cfg.DEBUG:
                plt.savefig("output_VM.png", dpi=300, bbox_inches="tight")  # type: ignore
                plt.waitforbuttonpress()  # type: ignore
                # plt.show()  # type: ignore

    executor.shutdown()
