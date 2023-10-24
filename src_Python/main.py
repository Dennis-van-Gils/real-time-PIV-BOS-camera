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
__date__ = "24-10-2023"
__version__ = "1.0"
# pylint: disable=missing-function-docstring

import os
import sys
import platform
import concurrent.futures
from time import perf_counter

import msvcrt

import psutil
import numpy as np
import numpy.typing as npt
import cv2

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
from utils import debugging
from utils.FrameServer import FrameServer
from utils.process_IWs import IW_Mesh, process_IWs
from utils.my_fun import (
    remove_mean_background,
    fliplrud,
    compute_displacement_vectors_from_C_maps,
)

if cfg.FFT_LIB == cfg.FFT_LIBS.PYFFTW:
    from utils.dvg_fftconvolver_pyfftw import FFT_Convolver2D_Full
elif cfg.FFT_LIB == cfg.FFT_LIBS.ROCKETFFT:
    from utils.dvg_fftconvolver_rocketfft import FFT_Convolver2D_Full
elif cfg.FFT_LIB == cfg.FFT_LIBS.SCIPY:
    from utils.dvg_fftconvolver_scipy import FFT_Convolver2D_Full
else:
    from utils.dvg_fftconvolver_rocketfft import FFT_Convolver2D_Full

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

    # Set up the frame server and read the first image frame
    frame_server = FrameServer()
    A, frame_title = frame_server.begin()
    remove_mean_background(A)

    img_h = frame_server.img_h
    img_w = frame_server.img_w

    # Prevent 'possibly unbound variable' warnings
    B_orig = np.zeros_like(A)
    B = np.zeros_like(A)
    A_ = np.zeros_like(A)

    if cfg.MODE == cfg.MODES.PIV:
        # Particle image velocimetry using equidistantly timed frames
        #   (frame_0, frame_1), (frame_1, frame_2), (frame_2, frame_3), ...
        #
        # Directly copy frame `A` into `B` to init the upcoming loop
        B = np.copy(A)

    elif cfg.MODE == cfg.MODES.PIV2:
        # Particle image velocimetry using image pairs
        #   (frame_0, frame_1), (frame_2, frame_3), (frame_4, frame_5), ...
        pass  # No extra steps needed

    elif cfg.MODE == cfg.MODES.BOS:
        # Background-oriented Schlieren
        #   (frame_0, frame_1), (frame_0, frame_2), (frame_0, frame_3), ...
        #
        # The first frame (frame_0 == `A`) should contain the quiescent
        # background consisting of a fine grained noise pattern, used from now
        # on to cross-correlate all subsequent frames `B` against.
        # Flip the image of frame `A` left-to-right and up-to-down ahead of time
        # as needed for the upcoming 2D cross-correlation done via convolution.
        A_ = np.asarray(fliplrud(A), order="C")

    # --------------------------------------------------------------------------
    #   Preallocate and populate lists for the upcoming multigrid analysis.
    #     stage: Current multigrid stage, from largest IW size to the smallest.
    #     Prefix 'l' denotes 'list' with index `stage_idx`.
    #
    #   Preallocate lists
    # --------------------------------------------------------------------------

    # IW meshgrids and limits per stage of the multigrid
    lIW_mesh: list[IW_Mesh] = []

    # Computed IW pre-shifts per stage of the multigrid
    # NOTE: List index `stage_idx = 0` will be initialized with zeros and remain
    # so throughout, because no window pre-shifts ever exist for the first
    # multigrid stage by design. That's okay.
    lIW_shifts_x: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, )
    lIW_shifts_y: list[npt.NDArray[np.int32]] = []  # NDArray shape (N_IWs, )

    # Computed correlation maps per stage of the multigrid
    #   NDArray shape (N_IWs, IW_size * 2 - 1, IW_size * 2 - 1)
    lC_maps: list[npt.NDArray[np.float32]] = []

    # fmt: off
    # Computed displacement vector maps per stage of the multigrid
    lVM_grid_x: list[npt.NDArray[np.int32]] = []    # NDArray shape (N_IWs, )
    lVM_grid_y: list[npt.NDArray[np.int32]] = []    # NDArray shape (N_IWs, )
    lVM_dx: list[npt.NDArray[np.float32]] = []      # NDArray shape (N_IWs, )
    lVM_dy: list[npt.NDArray[np.float32]] = []      # NDArray shape (N_IWs, )
    # fmt: on

    # FFT calculation instances per stage of the multigrid
    # NOTE: In addition, each stage will have multiple copies of similar FFT
    # instances equal to the number of concurrent workers `cfg.N_WORKERS`.
    lfft: list[list[FFT_Convolver2D_Full]] = []

    # --------------------------------------------------------------------------
    #   Populate lists
    # --------------------------------------------------------------------------

    for stage_idx, IW_size in enumerate(cfg.IW_SIZES):
        # Create interrogation windows. Only the last multigrid stage will have
        # window overlapping applied to it.
        IW_mesh = IW_Mesh(
            img_w,
            img_h,
            IW_size,
            cfg.IW_OVERLAP if stage_idx == cfg.N_STAGES - 1 else 0.0,
        )
        lIW_mesh.append(IW_mesh)

        lIW_shifts_x.append(np.zeros(IW_mesh.N_IWs, dtype=np.int32))
        lIW_shifts_y.append(np.zeros(IW_mesh.N_IWs, dtype=np.int32))

        C_maps = np.empty(
            (IW_mesh.N_IWs, IW_size * 2 - 1, IW_size * 2 - 1),
            dtype=np.float32,
        )
        C_maps[:] = np.nan
        lC_maps.append(C_maps)

        lVM_grid_x.append(np.copy(IW_mesh.A_grid_x))
        lVM_grid_y.append(np.copy(IW_mesh.A_grid_y))
        lVM_dx.append(np.zeros(IW_mesh.N_IWs, dtype=np.float32))
        lVM_dy.append(np.zeros(IW_mesh.N_IWs, dtype=np.float32))

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
        lIW_mesh[0].IW_params,
        lVM_dx[0],
        lVM_dy[0],
        lIW_mesh[0].A_grid_x,
        lIW_mesh[0].A_grid_y,
        lIW_mesh[0].A_lims_x,
        lIW_mesh[0].A_lims_y,
        lIW_mesh[0].B_grid_x,
        lIW_mesh[0].B_grid_y,
        lIW_mesh[0].B_lims_x,
        lIW_mesh[0].B_lims_y,
        lIW_shifts_x[0],
        lIW_shifts_y[0],
        lC_maps[0],
        lfft[0][0],
        slice(0, 0),
    )

    # Display info
    print(f"done in {perf_counter() - t_0:.3f} sec\n")

    # --------------------------------------------------------------------------
    #   Walk over all image frames from disk / acquire frames from the camera
    # --------------------------------------------------------------------------

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=cfg.N_WORKERS)

    # Debugging overrule: Only process the first two images and be done
    if cfg.DEBUG:
        cfg.N_IMAGES = 2

    done = False
    frame_idx = 0
    while not done:
        # ----------------------------------------------------------------------
        #   Read and prepare new image frames
        # ----------------------------------------------------------------------

        t_0 = perf_counter()
        frame_title = ""
        if cfg.MODE == cfg.MODES.PIV:
            # Particle image velocimetry using equidistantly timed frames
            #   (frame_0, frame_1), (frame_1, frame_2), (frame_2, frame_3), ...
            A = np.copy(B)
            B, frame_title = frame_server.serve(frame_idx + 1)

        elif cfg.MODE == cfg.MODES.PIV2:
            # Particle image velocimetry using image pairs
            #   (frame_0, frame_1), (frame_2, frame_3), (frame_4, frame_5), ...
            A, frame_title = frame_server.serve(frame_idx)
            B, frame_title = frame_server.serve(frame_idx + 1)
            remove_mean_background(A)

        elif cfg.MODE == cfg.MODES.BOS:
            # Background-oriented Schlieren
            #   (frame_0, frame_1), (frame_0, frame_2), (frame_0, frame_3), ...
            B, frame_title = frame_server.serve(frame_idx + 1)

        np.copyto(B_orig, B)  # Keep a copy of original `B` for plotting later
        remove_mean_background(B)

        if cfg.MODE in [cfg.MODES.PIV, cfg.MODES.PIV2]:
            # Flip the image of frame `A` left-to-right and up-to-down ahead of
            # time as needed for the upcoming 2D cross-correlation done via
            # convolution. Doing this ahead of time instead of doing it inside
            # `process_IWs()` for each individual IW saves many duplicate
            # `fliplrud()` operations on identical data because of the window
            # overlapping.
            A_ = np.asarray(fliplrud(A), order="C")

        # Display info
        print(frame_title)
        print(
            f"Reading, processing... done in {perf_counter() - t_0:.3f}, ",
            end="",
        )
        sys.stdout.flush()
        t_0 = perf_counter()

        # ----------------------------------------------------------------------
        #   Walk over all multigrid stages
        # ----------------------------------------------------------------------

        for stage_idx, IW_size in enumerate(cfg.IW_SIZES):
            # Short-hands
            N_IWs = lIW_mesh[stage_idx].N_IWs

            # Reset variables in-between image pairs
            lIW_mesh[stage_idx].reset_B()

            """
            # `IW_mesh.A_grid_x/y` remain constant and do not need a reset.
            # `IW_mesh.A_lims_x/y` remain constant and do not need a reset.

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
            prev_stage_idx = np.maximum(stage_idx - 1, 0)
            futures = []
            for worker_idx in range(cfg.N_WORKERS):
                futures.append(
                    executor.submit(
                        process_IWs,
                        stage_idx,
                        A_,
                        B,
                        lIW_mesh    [stage_idx - 1].IW_params,
                        lVM_dx      [stage_idx - 1],
                        lVM_dy      [stage_idx - 1],
                        lIW_mesh    [stage_idx].A_grid_x,
                        lIW_mesh    [stage_idx].A_grid_y,
                        lIW_mesh    [stage_idx].A_lims_x,
                        lIW_mesh    [stage_idx].A_lims_y,
                        lIW_mesh    [stage_idx].B_grid_x,
                        lIW_mesh    [stage_idx].B_grid_y,
                        lIW_mesh    [stage_idx].B_lims_x,
                        lIW_mesh    [stage_idx].B_lims_y,
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
                perform_subpixel_fitting=(stage_idx == cfg.N_STAGES - 1),
            )
            # fmt: on

        # Display info
        duration = perf_counter() - t_0
        print(f"{duration:.3f} sec")

        # ----------------------------------------------------------------------
        #   Debugging output
        # ----------------------------------------------------------------------

        if cfg.DEBUG_PRINT:
            debugging.print_info(
                lIW_mesh,
                lIW_shifts_x,
                lIW_shifts_y,
                lC_maps,
                lVM_dx,
                lVM_dy,
            )

        if cfg.DEBUG_IW_PX is not None:
            debugging.IW_analysis(
                cfg.DEBUG_IW_PX[0],
                cfg.DEBUG_IW_PX[1],
                A,
                B,
                lIW_mesh,
                lIW_shifts_x,
                lIW_shifts_y,
                lC_maps,
                lVM_grid_x,
                lVM_grid_y,
                lVM_dx,
                lVM_dy,
            )

        # ----------------------------------------------------------------------
        #   Show results
        # ----------------------------------------------------------------------

        # TODO: In progress code. Contains hardcoded 'magic' constants.
        if cfg.MODE in [cfg.MODES.BOS]:
            (
                IW_size,
                IW_overlap,
                N_IWs,
                N_IWs_x,
                N_IWs_y,
            ) = lIW_mesh[-1].IW_params
            grid_x = lVM_grid_x[-1]
            grid_y = lVM_grid_y[-1]
            VM_dx = lVM_dx[-1]
            VM_dy = lVM_dy[-1]

            # Creates an image filled with zero
            # intensities with the same dimensions
            # as the frame
            mask = np.zeros((N_IWs_y, N_IWs_x, 3), dtype=np.uint8)

            # Computes the magnitude and angle of the 2D vectors
            VM_dx_2 = np.reshape(VM_dx, (N_IWs_y, N_IWs_x))
            VM_dy_2 = np.reshape(VM_dy, (N_IWs_y, N_IWs_x))
            M, angle = cv2.cartToPolar(VM_dx_2, VM_dy_2, angleInDegrees=True)

            # Sets image hue according to the optical flow direction
            angle = np.nan_to_num(angle)
            mask[..., 0] = angle / 2

            # Sets image saturation to maximum
            mask[..., 1] = 255

            # Sets image value according to the optical flow
            # magnitude (normalized)
            M = np.nan_to_num(M)
            # mask[..., 2] = cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX)
            mask[..., 2] = np.clip(M * 300, 0, 255)

            # Converts HSV to RGB (BGR) color representation
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

            display_resolution = (B.shape[1] // 2, B.shape[0] // 2)
            cv2.imshow(
                "BOS",
                cv2.resize(
                    rgb,
                    display_resolution,
                    interpolation=cv2.INTER_NEAREST,
                    # interpolation=cv2.INTER_NEAREST,
                ),
            )
            cv2.imshow(
                "Image",
                cv2.resize(
                    B_orig,
                    display_resolution,
                    interpolation=cv2.INTER_NEAREST,
                ),
            )

            cv2.setWindowTitle("BOS", f"BOS {frame_title}")

            fn_export = f"export_{frame_idx:04d}.png"
            # cv2.imwrite(fn_export, cv2.resize(rgb, display_resolution))

            if cfg.DEBUG:
                cv2.waitKey(0)

        elif cfg.LOAD_MPL:
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
                h_title = plt.title(f"{frame_title}")  # type: ignore

            h_imshow.set_data(A)  # type: ignore
            h_quiver.set_UVC(VM_dx * cfg.QUIVER_SIZE, VM_dy * cfg.QUIVER_SIZE)  # type: ignore
            h_quiver.set_color(colormap(colors))  # type: ignore
            h_title.set_text(f"{frame_title}")  # type: ignore

            plt.draw()  # type: ignore
            plt.pause(0.0001)  # type: ignore

            if cfg.DEBUG:
                plt.savefig("output_VM.png", dpi=300, bbox_inches="tight")  # type: ignore
                # plt.waitforbuttonpress()  # type: ignore
                plt.show()  # type: ignore

        # ----------------------------------------------------------------------
        #   Are we finished?
        # ----------------------------------------------------------------------

        frame_step = 2 if cfg.MODE == cfg.MODES.PIV2 else 1
        frame_idx += frame_step

        if cfg.IMAGE_SOURCE == cfg.IMAGE_SOURCES.DISK:
            if frame_idx >= cfg.N_IMAGES - 1:
                done = True

        # Check for key presses
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # NOTE: Windows only
        if msvcrt.kbhit():
            k = msvcrt.getch()
            if k == b"q":
                done = True

    cv2.destroyAllWindows()
    frame_server.close()
    executor.shutdown()
