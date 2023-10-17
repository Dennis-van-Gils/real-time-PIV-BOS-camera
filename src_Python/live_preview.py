#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "17-10-2023"
__version__ = "1.0"

import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# The import order is important. We must handle the user configuration first.
import init_config as cfg

# Parse command line arguments.
# Expecting None or the filename of the configuration file to read in.
if len(sys.argv) > 1:
    config_filename = sys.argv[1]
else:
    config_filename = "live_preview.ini"
cfg.read_file_live_preview(config_filename)

# Now we can import the remaining modules
from utils.FrameServer import FrameServer

# NOTE: Backend `TkAgg` does not work well with the histogram. The matplotlib
# window steals the keypresses away from `cv2.imshow()`
# mpl.use("TkAgg")
mpl.use("QtAgg")  # Preferred above `TkAgg`

# ------------------------------------------------------------------------------
#   Settings
# ------------------------------------------------------------------------------

# OpenCV window name
WINNAME_MAIN = "Live preview"

# Toggle to correct for Windows display scaling issue
DISPLAY_SCALING = 125  # Set equal to the display scaling used by Windows [%]
do_adjust_display_scaling = False

# Toggle to enable/disable clip warning by painting clipped pixels in red
do_show_clipped = True

# Toggle to show histogram
do_show_histogram = False

# Toggle zoom window
ZOOM_BLOCKSIZE = 32  # [px]
WINNAME_ZOOM = f"Zoom {ZOOM_BLOCKSIZE}x{ZOOM_BLOCKSIZE}"
do_show_zoom = True

# Toggle to save acquired frames to disk
do_save_frames = False

# Toggle to report frame time intervals to the terminal
do_report_frame_dT = False

# ------------------------------------------------------------------------------
#   Open video camera
# ------------------------------------------------------------------------------

print("Starting video")
print("--------------")
print(f"Camera ID: {cfg.CAMERA_ID}")

# Set up the frame server
frame_server = FrameServer()
frame_server.begin()

# Experimental: Try adjusting wanted camera settings
if cfg.IMAGE_SOURCE == cfg.IMAGE_SOURCES.WEBCAM:
    cap = frame_server.cam_cv2
    cap.set(cv2.CAP_PROP_SETTINGS, 0)  # Show the camera controls when available
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

# Obtained settings
if cfg.IMAGE_SOURCE == cfg.IMAGE_SOURCES.WEBCAM:
    obt_fps = frame_server.cam_cv2.get(cv2.CAP_PROP_FPS)
else:
    obt_fps = 0.0

# Correct Windows display scaling issue
if do_adjust_display_scaling:
    scaled_window_w = int(frame_server.img_w // (DISPLAY_SCALING / 100))
    scaled_window_h = int(frame_server.img_h // (DISPLAY_SCALING / 100))
    cv2.namedWindow(WINNAME_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINNAME_MAIN, scaled_window_w, scaled_window_h)
else:
    scaled_window_w = frame_server.img_w
    scaled_window_h = frame_server.img_h

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

# ------------------------------------------------------------------------------
#   Acquire frames
# ------------------------------------------------------------------------------

print(
    f"Obtained : {frame_server.img_w} x {frame_server.img_h} px^2 "
    f"@ {obt_fps} fps"
)
print("")
print("Keypresses registered by video window:")
print("  c | Toggle clip warning.")
print("  h | Toggle show histogram.")
print("  s | Toggle save frames to disk.")
print("  t | Toggle report frame dT in [ms].")
print("  z | Toggle show zoom.")
print("  q | Quit.")
print("")
print(f"Clip warning   : {bool(do_show_clipped)}")
print(f"Show histogram : {bool(do_show_histogram)}")
print(f"Save to disk   : {bool(do_save_frames)}")
print(f"Report frame dT: {bool(do_report_frame_dT)}")
print(f"Show zoom      : {bool(do_show_zoom)}")

tick = time.perf_counter()  # [sec]
prev_tick_histogram = tick
try:
    while True:
        # Acquire frame
        img_gray, frame_title = frame_server.serve()

        # Timer
        tick = time.perf_counter()
        if do_report_frame_dT:
            print(f"{frame_server.frame_dT*1000:.0f}")

        # Convert float32 [0 - 1] pixel intensity range to uint8 [0 - 255]
        img_gray = np.asarray(img_gray * 255, dtype=np.uint8)

        # Recolor clipped intensities as full red
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        clipped_idxs = (img_gray >= 254).nonzero()
        img_rgb[clipped_idxs] = [0, 0, 255]  # bgr

        # Show image
        cv2.imshow(WINNAME_MAIN, img_rgb if do_show_clipped else img_gray)

        # Correct Windows display scaling issue
        if do_adjust_display_scaling:
            cv2.resizeWindow(WINNAME_MAIN, scaled_window_w, scaled_window_h)

        # Save acquired frame to disk
        if do_save_frames:
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

        # ----------------------------------------------------------------------
        #   Zoom
        # ----------------------------------------------------------------------

        if do_show_zoom:
            img_zoom = img_gray[zoom_slice]
            img_zoom_exploded = cv2.resize(
                img_zoom,
                (8 * ZOOM_BLOCKSIZE, 8 * ZOOM_BLOCKSIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imshow(WINNAME_ZOOM, img_zoom_exploded)

        # ----------------------------------------------------------------------
        #   Histogram
        # ----------------------------------------------------------------------

        if do_show_histogram:
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            hist = np.asarray(hist, dtype=np.float32) / img_gray.size
            if not (plt.fignum_exists(fignum_histogram)):
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
            # fig.canvas.flush_events()  # Backend `QtAgg` requires this line

        # ----------------------------------------------------------------------
        #   Handle keypresses
        # ----------------------------------------------------------------------

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            do_show_clipped = not do_show_clipped
            print(f"Clip warning   : {bool(do_show_clipped)}")

        elif key == ord("h"):
            do_show_histogram = not do_show_histogram
            print(f"Show histogram : {bool(do_show_histogram)}")
            if (plt.fignum_exists(fignum_histogram)) and not do_show_histogram:
                plt.close(fignum_histogram)

        elif key == ord("s"):
            do_save_frames = not do_save_frames
            print(f"Save to disk   : {bool(do_save_frames)}")

        elif key == ord("t"):
            do_report_frame_dT = not do_report_frame_dT
            print(f"Report frame dT: {bool(do_report_frame_dT)}")

        elif key == ord("z"):
            do_show_zoom = not do_show_zoom
            print(f"Show zoom      : {bool(do_show_zoom)}")
            if not do_show_zoom:
                cv2.destroyWindow(WINNAME_ZOOM)

        elif key == ord("q"):
            break

except KeyboardInterrupt:
    pass

# ------------------------------------------------------------------------------
#   Close
# ------------------------------------------------------------------------------

cv2.destroyAllWindows()
print("\nStopping acquisition... ", end="")
frame_server.close()
print("done.")
