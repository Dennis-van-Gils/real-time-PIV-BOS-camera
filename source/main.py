#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=missing-function-docstring
"""Real-time fluid flow visualization for Particle Image Velocimetry (PIV) &
Background-Oriented Schlieren (BOS) camera setups.

Used abbrevations
-----------------
IW: Interrogation window
VM: Displacement vector map
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/real-time-PIV-BOS-camera"
__date__ = "19-12-2023"
__version__ = "1.0"

w = 61
print("=" * w)
print("Real-time fluid flow visualization for".center(w))
print("Particle Image Velocimetry (PIV) &".center(w))
print("Background-Oriented Schlieren (BOS)".center(w))
print("camera setups\n".center(w))
print(f"{__url__}".center(w))
print("=" * w)

info_usage = f"""
Usage: python main.py [configuration file (optional)]
  E.g. python main.py
       python main.py config.ini
       python main.py configs/config_BOS_demo_1.ini
  Opens a file navigator when no file is supplied.

  Edit the configuration file to fit your needs. See,
  {__url__}/blob/main/source/config.ini"""

info_keypresses = """
Keypresses
  ? | Show this keypresses overview.
  b | Reacquire BOS frame 0.
  c | Colormap clip warning ON/OFF.
  o | Original video frames ON/OFF.
  r | Record frames to disk ON/OFF.
  t | Print timing info     ON/OFF.
  q | Quit."""

import os
import sys
import platform
import concurrent.futures
from time import perf_counter

import psutil
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import cv2

if os.name == "nt":
    import msvcrt

# We must process the configuration file before we import the `utils` modules
import init_config as cfg

if len(sys.argv) > 1:
    config_filename = sys.argv[1]
else:
    config_filename = None
    print(info_usage)
cfg.read_file(config_filename)

# Now we can import the `utils` modules
from utils import debugging
from utils import plotting
from utils.FrameServer import FrameServer
from utils.process_IWs import IW_Mesh, process_IWs
from utils.my_fun import (
    bool2on,
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

# Global user-interaction flags
do_colormap_clip_warning = False
do_show_original_video = False
do_record_to_disk = False
do_print_timing_info = False


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

    print(f"\nRunning on computer `{platform.node()}`")
    print(f"  CPU: {platform.processor()}")
    print(f"  OS : {platform.system()}")
    print("\nConfiguration")
    print(f"  mode         : {cfg.MODE.name}")
    print(f"  FFT_lib      : {cfg.FFT_LIB.name}")
    print(f"  max_workers  : {cfg.MAX_WORKERS}")
    print(f"  N_FFT_threads: {cfg.N_FFT_THREADS}")
    print(f"  IW_sizes     : {cfg.IW_SIZES}")
    print(f"  IW_overlap   : {cfg.IW_OVERLAP}\n")
    tick = perf_counter()

    # Set up the frame server and read the first image frame
    frame_server = FrameServer()
    A = frame_server.begin()
    frame_server.report()

    # Pre-allocate to prevent 'possibly unbound variable' warnings
    A_ = np.zeros_like(A)
    B = np.zeros_like(A)
    B_orig = np.zeros_like(A)

    # Init image processing
    remove_mean_background(A)

    if cfg.MODE == cfg.MODES.PIV:
        # Particle image velocimetry using equidistantly timed frames
        #   (frame_0, frame_1), (frame_1, frame_2), (frame_2, frame_3), ...

        # Directly copy frame `A` into `B` to init the upcoming loop
        frame_server.counter = 1
        np.copyto(B, A)

    elif cfg.MODE == cfg.MODES.PIV2:
        # Particle image velocimetry using image pairs
        #   (frame_0, frame_1), (frame_2, frame_3), (frame_4, frame_5), ...
        pass

    elif cfg.MODE == cfg.MODES.BOS:
        # Background-oriented Schlieren
        #   (frame_0, frame_1), (frame_0, frame_2), (frame_0, frame_3), ...

        # The first frame (frame_0 === `A`) should contain the quiescent
        # background consisting of a fine-grained noise pattern, used from now
        # on to cross-correlate all subsequent frames `B` against.
        # Flip the image of frame `A` left-to-right and up-to-down ahead of time
        # as needed for the upcoming 2D cross-correlation done via convolution.
        frame_server.counter = 1
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

    # `FFT_Convolver` instances per stage of the multigrid and per concurrent
    # worker.
    #
    # Mechanism: The calculation of the FFT-convolution over all IWs in a
    # specific multigrid stage will be evenly distributed over multiple
    # concurrent threads, called workers. Each worker will have a single
    # `FFT_Convolver` instance that will operate on a specific slice of the
    # available IWs. The number of workers will be calculated later, with at
    # least one IW per worker, up to a maximum of `cfg.MAX_WORKERS`.
    # Hence, the inner list holds the `FFT_Convolver` instances per worker of
    # a specific stage, and the outer list enumerates the specific stage.
    lfft: list[list[FFT_Convolver2D_Full]] = []

    # Slices of IWs to be passed to the `FFT_Convolver` instances per stage of
    # the multigrid and per concurrent worker.
    # The inner list holds the slice (as `tuple[int, int]`) of IWs that each
    # worker will operate on, and the outer list enumerates the specific stage.
    lIWs_slices: list[list[tuple[int, int]]] = []

    # fmt: off
    # Computed correlation maps per stage of the multigrid
    lC_maps: list[npt.NDArray[np.float32]] = []     # NDArray shape (N_IWs, :, :)

    # Computed displacement vector maps per stage of the multigrid
    lVM_grid_x: list[npt.NDArray[np.int32]] = []    # NDArray shape (N_IWs, )
    lVM_grid_y: list[npt.NDArray[np.int32]] = []    # NDArray shape (N_IWs, )
    lVM_dx: list[npt.NDArray[np.float32]] = []      # NDArray shape (N_IWs, )
    lVM_dy: list[npt.NDArray[np.float32]] = []      # NDArray shape (N_IWs, )

    # Computed vector magnitudes and angles per stage of the multigrid
    lVM_magn: list[npt.NDArray[np.float32]] = []    # NDArray shape (N_IWs, )
    lVM_angle: list[npt.NDArray[np.float32]] = []   # NDArray shape (N_IWs, )
    # fmt: on

    # --------------------------------------------------------------------------
    #   Populate lists
    # --------------------------------------------------------------------------

    print("\nStages")
    print("  IW_size |  N_IWs | N_workers")
    print("  --------|--------|----------")

    for stage_idx, IW_size in enumerate(cfg.IW_SIZES):
        # Only the last stage will have window overlapping applied to it
        IW_mesh = IW_Mesh(
            frame_server.img_w,
            frame_server.img_h,
            IW_size,
            cfg.IW_OVERLAP if stage_idx == cfg.N_STAGES - 1 else 0.0,
        )
        N_IWs = IW_mesh.N_IWs

        lIW_mesh.append(IW_mesh)
        lIW_shifts_x.append(np.zeros(N_IWs, dtype=np.int32))
        lIW_shifts_y.append(np.zeros(N_IWs, dtype=np.int32))

        # Determine the number of workers to divide the IWs over, per stage.
        # Ensure at least one IW per worker. The maximum number of workers is
        # set by `cfg.MAX_WORKERS`.
        N_workers = np.minimum(N_IWs, cfg.MAX_WORKERS)
        print(f"  {IW_size:7d} |{N_IWs:7d} |{N_workers:10d}")
        sys.stdout.flush()

        fft_workers = []
        IWs_slices = []
        for worker_idx in range(N_workers):
            idx_0 = int(np.floor(N_IWs / N_workers * worker_idx))
            idx_1 = int(np.floor(N_IWs / N_workers * (worker_idx + 1)))
            IWs_slices.append((idx_0, idx_1))

            # Create `FFT_Convolver` instance per worker
            fft_workers.append(
                FFT_Convolver2D_Full(
                    (IW_size, IW_size),
                    (IW_size, IW_size),
                    fft_threads=cfg.N_FFT_THREADS,
                )
            )
        lfft.append(fft_workers)
        lIWs_slices.append(IWs_slices)

        C_map_shape = lfft[-1][-1].shape_out
        C_maps = np.empty((N_IWs, *C_map_shape), dtype=np.float32)
        C_maps[:] = np.nan
        lC_maps.append(C_maps)

        # fmt: off
        lVM_grid_x.append(np.copy(IW_mesh.A_grid_x))
        lVM_grid_y.append(np.copy(IW_mesh.A_grid_y))
        lVM_dx    .append(np.zeros(N_IWs, dtype=np.float32))
        lVM_dy    .append(np.zeros(N_IWs, dtype=np.float32))
        lVM_magn  .append(np.zeros(N_IWs, dtype=np.float32))
        lVM_angle .append(np.zeros(N_IWs, dtype=np.float32))
        # fmt: on

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
        (0, 0),
    )

    print(f"\nSetting up done in {perf_counter() - tick:.1f} sec")
    print(f"{info_keypresses}\n")

    # --------------------------------------------------------------------------
    #   Walk over all image frames from disk / Acquire frames from the camera
    # --------------------------------------------------------------------------

    executor = concurrent.futures.ThreadPoolExecutor(cfg.MAX_WORKERS)

    # Debugging overrule: Only process the first two images and be done
    if cfg.DEBUG:
        cfg.N_IMAGES = 2

    tick_overall = perf_counter()
    while frame_server.has_available(2 if cfg.MODE == cfg.MODES.PIV2 else 1):
        # ----------------------------------------------------------------------
        #   Read and prepare new image frames
        # ----------------------------------------------------------------------
        tick = perf_counter()

        if cfg.MODE == cfg.MODES.PIV:
            # Particle image velocimetry using equidistantly timed frames
            #   (frame_0, frame_1), (frame_1, frame_2), (frame_2, frame_3), ...
            np.copyto(A, B)
            B = frame_server.serve()

        elif cfg.MODE == cfg.MODES.PIV2:
            # Particle image velocimetry using image pairs
            #   (frame_0, frame_1), (frame_2, frame_3), (frame_4, frame_5), ...
            A = frame_server.serve()
            B = frame_server.serve()
            remove_mean_background(A)

        elif cfg.MODE == cfg.MODES.BOS:
            # Background-oriented Schlieren
            #   (frame_0, frame_1), (frame_0, frame_2), (frame_0, frame_3), ...
            B = frame_server.serve()

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

        # Timing info
        frame_title = frame_server.title
        duration = perf_counter() - tick
        if do_print_timing_info:
            print(f"{frame_title:<30s} read {duration:.3f} | proc ", end="")
            sys.stdout.flush()

        # ----------------------------------------------------------------------
        #   Walk over all multigrid stages
        # ----------------------------------------------------------------------
        tick = perf_counter()

        for stage_idx in range(cfg.N_STAGES):
            # Reset variables in-between image pairs
            lIW_mesh[stage_idx].reset_B()

            """
            # `IW_mesh.A_grid_x/y` remain constant and do not need a reset.
            # `IW_mesh.A_lims_x/y` remain constant and do not need a reset.
            lIW_mesh[stage_idx].reset_A()

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

            # Spawn workers, each performing 2D convolutions on a slice of IWs
            # fmt: off
            prev_stage_idx = np.maximum(stage_idx - 1, 0)
            futures = []
            for worker_idx in range(len(lIWs_slices[stage_idx])):
                futures.append(
                    executor.submit(
                        process_IWs,
                        stage_idx,
                        A_,
                        B,
                        lIW_mesh    [prev_stage_idx].IW_params,
                        lVM_dx      [prev_stage_idx],
                        lVM_dy      [prev_stage_idx],
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
                        lIWs_slices [stage_idx][worker_idx],
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

            cv2.cartToPolar(
                lVM_dx   [stage_idx],
                lVM_dy   [stage_idx],
                lVM_magn [stage_idx],
                lVM_angle[stage_idx],
                angleInDegrees=True,
            )
            # fmt: on

        # Timing info
        duration = perf_counter() - tick
        if do_print_timing_info:
            print(f"{duration:.3f}")

        # ----------------------------------------------------------------------
        #   Debugging output
        # ----------------------------------------------------------------------

        if cfg.DEBUG_PRINT:
            debugging.print_IWs(
                lIW_mesh,
                lIW_shifts_x,
                lIW_shifts_y,
                lC_maps,
                lVM_dx,
                lVM_dy,
            )

        if cfg.DEBUG_IW_PX is not None:
            debugging.plot_IW_analysis(
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

        if cfg.PLOT_VECTOR_MAP_RESULTS:
            # fmt: off
            # Retrieve the end results
            IW_mesh   = lIW_mesh[-1]
            VM_grid_x = lVM_grid_x[-1]
            VM_grid_y = lVM_grid_y[-1]
            VM_dx     = lVM_dx[-1]
            VM_dy     = lVM_dy[-1]
            VM_magn   = lVM_magn[-1]
            VM_angle  = lVM_angle[-1]
            # fmt: on

            if cfg.MODE in [cfg.MODES.BOS]:
                canvas = plotting.vector_map_to_hsv_colors(
                    VM_magn,
                    VM_angle,
                    VM_grid_shape_2D=(IW_mesh.N_IWs_y, IW_mesh.N_IWs_x),
                    output_resolution=(frame_server.img_w, frame_server.img_h),
                    interpolation=cv2.INTER_NEAREST,
                    # interpolation=cv2.INTER_LINEAR,
                    # interpolation=cv2.INTER_CUBIC,
                    show_clipped_as_white=do_colormap_clip_warning,
                )
            else:  # [PIV & PIV2]
                canvas = plotting.vector_map_to_cv2_quiver_plot(
                    B_orig,
                    VM_grid_x,
                    VM_grid_y,
                    VM_dx,
                    VM_dy,
                    VM_magn,
                    show_clipped=do_colormap_clip_warning,
                )

            cv2.imshow("VM_results", canvas)
            cv2.setWindowTitle("VM_results", f"{frame_title}")

            if do_show_original_video:
                # Convert float32 [0 - 1] intensities to uint8 [0 - 255]
                img_gray = np.asarray(B_orig * 255, dtype=np.uint8)

                # Recolor clipped intensities as full red
                img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                clipped_idxs = (img_gray >= 254).nonzero()
                img_rgb[clipped_idxs] = [0, 0, 255]  # bgr

                cv2.imshow("Original", img_rgb)
                cv2.setWindowTitle("Original", f"Original: {frame_title}")

            # DEBUG: Show the slower Matplotlib quiver plot and save to disk
            if cfg.DEBUG:
                plotting.vector_map_to_mpl_quiver_plot(
                    B_orig,
                    VM_grid_x,
                    VM_grid_y,
                    VM_dx,
                    VM_dy,
                    VM_magn,
                    frame_title,
                    show_clipped=do_colormap_clip_warning,
                )

                cv2.imwrite("output_cv2.png", canvas)
                plt.savefig("output_mpl.png", dpi=300, bbox_inches="tight")
                plt.show()

            # ------------------------------------------------------------------
            #   Export frames to disk
            # ------------------------------------------------------------------

            if do_record_to_disk:
                fn_export = f"{frame_title.replace(' ', '_')}.png"
                if cv2.imwrite(fn_export, canvas):
                    print(f"Saved {fn_export}")
                else:
                    print(f"Failed to save {fn_export}")

        # ----------------------------------------------------------------------
        #   Handle keypresses
        # ----------------------------------------------------------------------

        key_pressed = None

        # fmt: off
        # Listen for keypresses from within OpenCV plot
        cv2_key = cv2.waitKey(1)
        if cv2_key == ord("q"):   key_pressed = "q"
        elif cv2_key == ord("?"): key_pressed = "?"
        elif cv2_key == ord("/"): key_pressed = "?"
        elif cv2_key == ord("b"): key_pressed = "b"
        elif cv2_key == ord("c"): key_pressed = "c"
        elif cv2_key == ord("o"): key_pressed = "o"
        elif cv2_key == ord("r"): key_pressed = "r"
        elif cv2_key == ord("t"): key_pressed = "t"

        # Listen for keypresses from within terminal, Windows only
        if os.name == "nt" and msvcrt.kbhit():
            ms_key = msvcrt.getch()
            if ms_key == b"q"  : key_pressed = "q"
            elif ms_key == b"?": key_pressed = "?"
            elif ms_key == b"/": key_pressed = "?"
            elif ms_key == b"b": key_pressed = "b"
            elif ms_key == b"c": key_pressed = "c"
            elif ms_key == b"o": key_pressed = "o"
            elif ms_key == b"r": key_pressed = "r"
            elif ms_key == b"t": key_pressed = "t"
        # fmt: on

        # Execute keypress
        if key_pressed == "q":
            print("Key q | Quit")
            break

        elif key_pressed == "?":
            print(info_keypresses)
            print()

        elif key_pressed == "b" and cfg.MODE in [cfg.MODES.BOS]:
            print("Key b | Reacquire BOS frame 0 ", end="")

            new_A = frame_server.serve()
            remove_mean_background(new_A)
            new_A_ = np.asarray(fliplrud(new_A), order="C")
            np.copyto(A, new_A)
            np.copyto(A_, new_A_)
            print("DONE")

        elif key_pressed == "c":
            do_colormap_clip_warning = not do_colormap_clip_warning
            print("Key c | Colormap clip warning ", end="")
            print(f"{bool2on(do_colormap_clip_warning)}")

        elif key_pressed == "o":
            do_show_original_video = not do_show_original_video
            print("Key o | Original video frames ", end="")
            print(f"{bool2on(do_show_original_video)}")
            if not do_show_original_video:
                cv2.destroyWindow("Original")

        elif key_pressed == "r":
            do_record_to_disk = not do_record_to_disk
            print("Key r | Record frames to disk ", end="")
            print(f"{bool2on(do_record_to_disk)}")

        elif key_pressed == "t":
            do_print_timing_info = not do_print_timing_info
            print("Key t | Print timing info ", end="")
            print(f"{bool2on(do_print_timing_info)}")

    # --------------------------------------------------------------------------
    #   Exit
    # --------------------------------------------------------------------------

    print(f"\nOverall run time: {perf_counter() - tick_overall:.1f} sec")

    cv2.destroyAllWindows()
    frame_server.close()
    executor.shutdown()
