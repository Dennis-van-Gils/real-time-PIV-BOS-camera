#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "17-11-2023"
__version__ = "1.0"

w = 60
print("=" * w)
print("Live camera preview".center(w))
print(f"{__url__}".center(w))
print("=" * w)

info_usage = f"""
Usage: python live_preview.py [configuration file (optional)]
  E.g. python live_preview.py
       python live_preview.py live_preview.ini
  Opens `live_preview.ini` when no file is supplied.

  Edit the configuration file to fit your needs. See,
  {__url__}/blob/main/src_Python/live_preview.ini"""

info_keypresses = """
Keypresses
  ? | Show this keypresses overview.
  c | Clip warning          ON/OFF.
  h | Show histogram        ON/OFF.
  p | Show camera controls when available.
  r | Record frames to disk ON/OFF.
  t | Print timing info     ON/OFF.
  z | Show zoom window      ON/OFF.
  q | Quit."""

import os
import sys
from time import perf_counter
from datetime import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

if os.name == "nt":
    import msvcrt

# We must process the configuration file before we import the `utils` modules
import init_config as cfg

if len(sys.argv) > 1:
    config_filename = sys.argv[1]
else:
    config_filename = "live_preview.ini"
    print(info_usage)
cfg.read_file_live_preview(config_filename)

# Now we can import the `utils` modules
from utils.FrameServer import FrameServer

# NOTE: Backend `TkAgg` does not work well with the histogram. The matplotlib
# window steals the keypresses away from `cv2.imshow()`
# mpl.use("TkAgg")
mpl.use("QtAgg")  # Preferred above `TkAgg`

# ------------------------------------------------------------------------------
#   Globals
# ------------------------------------------------------------------------------

ZOOM_BLOCKSIZE = 32  # [px]
CV2_WINNAME_MAIN = "Main"
CV2_WINNAME_ZOOM = f"Zoom {ZOOM_BLOCKSIZE}x{ZOOM_BLOCKSIZE}"

# Global user-interaction flags
do_clip_warning = False
do_show_histogram = False
do_record_to_disk = False
do_print_timing_info = False
do_show_zoom = True

# ------------------------------------------------------------------------------
#   Tiny helper functions
# ------------------------------------------------------------------------------


def bool2on(state: bool) -> str:
    return "ON" if state else "OFF"


# ------------------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    print()
    frame_server = FrameServer()
    frame_server.begin()
    frame_server.report()

    main_window_title = (
        "Live preview: "
        f"{frame_server.img_w}x{frame_server.img_h} "
        f"@ {frame_server.img_bit_depth} bit"
    )
    print(f"{info_keypresses}\n")

    # Experimental: Try adjusting wanted camera settings
    if cfg.IMAGE_SOURCE == cfg.IMAGE_SOURCES.WEBCAM:
        cap = frame_server.cam_cv2
        cap.set(cv2.CAP_PROP_SETTINGS, 0)  # Show camera controls when available
        # cap.set(cv2.CAP_PROP_SETTINGS, 1)
        # cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        # cap.set(cv2.CAP_PROP_EXPOSURE, -7)
        # cap.set(cv2.CAP_PROP_EXPOSUREPROGRAM, 0)
        # cap.set(cv2.CAP_PROP_CONTRAST, 31)
        # cap.set(cv2.CAP_PROP_SATURATION, 31)
        # cap.set(cv2.CAP_PROP_GAIN, 127)
        # cap.set(cv2.CAP_PROP_SHARPNESS, 63)
        # cap.set(cv2.CAP_PROP_FOCUS, 0)
        # cap.set(cv2.CAP_PROP_FPS, 30)

    # Experimental: Obtained settings
    if cfg.IMAGE_SOURCE == cfg.IMAGE_SOURCES.WEBCAM:
        obt_fps = frame_server.cam_cv2.get(cv2.CAP_PROP_FPS)
    else:
        obt_fps = 0.0

    # Region of interest for zoom window is at dead center of image
    zoom_slice = (
        slice(
            frame_server.img_h // 2 - ZOOM_BLOCKSIZE // 2,
            frame_server.img_h // 2 + ZOOM_BLOCKSIZE // 2,
        ),
        slice(
            frame_server.img_w // 2 - ZOOM_BLOCKSIZE // 2,
            frame_server.img_w // 2 + ZOOM_BLOCKSIZE // 2,
        ),
    )

    # Export folder
    export_folder = datetime.strftime(datetime.now(), r"capture_%Y%m%d_%H%M%S")
    created_export_folder = False

    # Figure numbers
    fignum_histogram = 1

    # --------------------------------------------------------------------------
    #   Acquire frames
    # --------------------------------------------------------------------------

    tick = perf_counter()
    prev_tick_histogram = tick
    while frame_server.has_available(1):
        img_gray = frame_server.serve()
        frame_title = frame_server.title

        tick = perf_counter()
        if do_print_timing_info:
            print(
                f"{frame_server.title:<30s} | ",
                f"dT {frame_server.dT*1000:.0f} ms",
            )

        # Convert float32 [0 - 1] pixel intensity range to uint8 [0 - 255]
        img_gray = np.asarray(img_gray * 255, dtype=np.uint8)

        # Recolor clipped intensities as full red
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        clipped_idxs = (img_gray >= 254).nonzero()
        img_rgb[clipped_idxs] = [0, 0, 255]  # bgr

        cv2.imshow(CV2_WINNAME_MAIN, img_rgb if do_clip_warning else img_gray)
        cv2.setWindowTitle(CV2_WINNAME_MAIN, main_window_title)

        if do_record_to_disk:
            if not created_export_folder:
                if not os.path.exists(export_folder):
                    os.makedirs(export_folder)
                created_export_folder = True

            filename = f"{frame_title.replace(' ', '_')}.png"
            fn_save = os.path.join(export_folder, filename)
            if cv2.imwrite(fn_save, img_gray):
                print(f"Saved {fn_save}")
            else:
                print(f"Failed to save {fn_save}")

        if do_show_zoom:
            img_zoom = img_gray[zoom_slice]
            img_zoom_exploded = cv2.resize(
                img_zoom,
                (8 * ZOOM_BLOCKSIZE, 8 * ZOOM_BLOCKSIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow(CV2_WINNAME_ZOOM, img_zoom_exploded)

        # ----------------------------------------------------------------------
        #   Histogram
        # ----------------------------------------------------------------------

        if do_show_histogram:
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            hist = np.asarray(hist, dtype=np.float32) / img_gray.size
            if not plt.fignum_exists(fignum_histogram):
                plt.ion()
                fig = plt.figure(fignum_histogram)
                fig.canvas.manager.set_window_title("Histogram")  # type: ignore
                fig.canvas.mpl_disconnect(
                    fig.canvas.manager.key_press_handler_id  # type: ignore
                )
                (h_hist,) = plt.plot(hist)
                # plt.title("Histogram")
                plt.xticks(np.append(np.arange(0, 255, 32), 255))
                plt.xlim(0, 255)
                plt.show(block=False)

            # Update ylim every second
            if tick - prev_tick_histogram >= 1.0:
                prev_tick_histogram = tick
                max_hist_pct = np.max(hist) * 100  # [0 - 100] %
                max_ylim = np.ceil(max_hist_pct / 5) * 5 / 100
                ax = h_hist.axes  # type: ignore
                ax.axes.set_ylim([0, max_ylim])  # type: ignore
                plt.tight_layout()

            h_hist.set_ydata(hist)  # type: ignore
            fig.canvas.draw_idle()  # type: ignore
            if plt.get_backend() == "TkAgg":
                fig.canvas.flush_events()  # type: ignore

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
        elif cv2_key == ord("c"): key_pressed = "c"
        elif cv2_key == ord("h"): key_pressed = "h"
        elif cv2_key == ord("p"): key_pressed = "p"
        elif cv2_key == ord("r"): key_pressed = "r"
        elif cv2_key == ord("t"): key_pressed = "t"
        elif cv2_key == ord("z"): key_pressed = "z"

        # Listen for keypresses from within terminal, Windows only
        if os.name == "nt" and msvcrt.kbhit():
            ms_key = msvcrt.getch()
            if ms_key == b"q"  : key_pressed = "q"
            elif ms_key == b"?": key_pressed = "?"
            elif ms_key == b"/": key_pressed = "?"
            elif ms_key == b"c": key_pressed = "c"
            elif ms_key == b"h": key_pressed = "h"
            elif ms_key == b"p": key_pressed = "p"
            elif ms_key == b"r": key_pressed = "r"
            elif ms_key == b"t": key_pressed = "t"
            elif ms_key == b"z": key_pressed = "z"
        # fmt: on

        # Execute keypress
        if key_pressed == "q":
            print("Key q | Quit")
            break

        elif key_pressed == "?":
            print(info_keypresses)
            print()

        elif key_pressed == "c":
            do_clip_warning = not do_clip_warning
            print("Key c | Clip warning ", end="")
            print(f"{bool2on(do_clip_warning)}")

        elif key_pressed == "h":
            do_show_histogram = not do_show_histogram
            print("Key h | Show histogram ", end="")
            print(f"{bool2on(do_show_histogram)}")
            if not do_show_histogram and (plt.fignum_exists(fignum_histogram)):
                plt.close(fignum_histogram)

        elif key_pressed == "p":
            print("Key p | Show camera controls")
            frame_server.cam_cv2.set(cv2.CAP_PROP_SETTINGS, 0)

        elif key_pressed == "r":
            do_record_to_disk = not do_record_to_disk
            print("Key r | Record frames to disk ", end="")
            print(f"{bool2on(do_record_to_disk)}")

        elif key_pressed == "t":
            do_print_timing_info = not do_print_timing_info
            print("Key t | Print timing info ", end="")
            print(f"{bool2on(do_print_timing_info)}")

        elif key_pressed == "z":
            do_show_zoom = not do_show_zoom
            print("Key z | Show zoom ", end="")
            print(f"{bool2on(do_show_zoom)}")
            if not do_show_zoom:
                cv2.destroyWindow(CV2_WINNAME_ZOOM)

    # --------------------------------------------------------------------------
    #   Exit
    # --------------------------------------------------------------------------

    cv2.destroyAllWindows()
    frame_server.close()
