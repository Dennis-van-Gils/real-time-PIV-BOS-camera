#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "05-10-2023"
__version__ = "1.0"

import os
import time
from datetime import datetime

import cv2
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# NOTE: Backend `TkAgg` does not work well with the histogram. The matplotlib
# window steals the keypresses away from `cv2.imshow()`
# mpl.use("TkAgg")
mpl.use("QtAgg")  # Preferred above `TkAgg`

# ------------------------------------------------------------------------------
#   User settings
# ------------------------------------------------------------------------------

# OpenCV window name
WINNAME = "Webcam viewer"

# Camera ID to open
CAM_ID = 0

# Wanted resolution
# WANTED_RESOLUTION = (1280, 720)  # 0.9 MPx
# WANTED_RESOLUTION = (1600, 896)  # 1.4 MPx
WANTED_RESOLUTION = (1920, 1080)  # 2.1 MPx

# Toggle to correct for Windows display scaling issue
do_adjust_display_scaling = True
DISPLAY_SCALING = 125  # Set equal to the display scaling used by Windows [%]

# Toggle to enable/disable clip warning by painting clipped pixels in red
do_show_clipped = True

# Toggle to show histogram
do_show_histogram = False

# Toggle to save acquired frames to disk
do_save_frames = False

# Toggle to report frame time intervals to the terminal
do_report_frame_dT = False

# ------------------------------------------------------------------------------
#   Open video camera
# ------------------------------------------------------------------------------

print("Starting video")
print("--------------")
print(f"Camera ID: {CAM_ID}")

cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise Exception("Could not open camera. Check the set CAM_ID.")

# Try obtaining wanted camera settings
cap.set(cv2.CAP_PROP_SETTINGS, 0)  # Shows the camera controls when available
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WANTED_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WANTED_RESOLUTION[1])
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
obt_resolution = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)
obt_fps = cap.get(cv2.CAP_PROP_FPS)

# Correct Windows display scaling issue
if do_adjust_display_scaling:
    scaled_resolution = [
        int(x // (DISPLAY_SCALING / 100)) for x in obt_resolution
    ]
    cv2.namedWindow(WINNAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINNAME, scaled_resolution[0], scaled_resolution[1])
else:
    scaled_resolution = obt_resolution

# Export folder
export_folder = datetime.strftime(datetime.now(), r"capture_%Y%m%d_%H%M%S")
created_export_folder = False

# ------------------------------------------------------------------------------
#   Acquire frames
# ------------------------------------------------------------------------------

print(
    f"Obtained : {obt_resolution[0]} x {obt_resolution[1]} px^2 @ {obt_fps} fps"
)
print("")
print("Keypresses registered by video window:")
print("  c | Toggle clip warning.")
print("  h | Toggle show histogram.")
print("  s | Toggle save frames to disk.")
print("  t | Toggle report frame dT in [ms].")
print("  q | Quit.")
print("")
print(f"Clip warning   : {bool(do_show_clipped)}")
print(f"Show histogram : {bool(do_show_histogram)}")
print(f"Save to disk   : {bool(do_save_frames)}")
print(f"Report frame dT: {bool(do_report_frame_dT)}")

tick = time.perf_counter()  # [sec]
prev_tick_histogram = tick
prev_tick_frame = tick
frame_t0 = tick
frame_idx = 0
try:
    while True:
        # Acquire frame
        success, img_raw = cap.read()

        # Timer
        prev_tick_frame = tick
        tick = time.perf_counter()
        frame_time = tick - frame_t0
        frame_dT = tick - prev_tick_frame
        if do_report_frame_dT:
            print(f"{frame_dT*1000:.0f}")

        # Turn into grayscale
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

        # Recolor clipped intensities as full red
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        clipped_idxs = (img_gray == 255).nonzero()
        img_rgb[clipped_idxs] = [0, 0, 255]  # bgr

        # Show image
        cv2.imshow(WINNAME, img_rgb if do_show_clipped else img_gray)

        # Correct Windows display scaling issue
        if do_adjust_display_scaling:
            cv2.resizeWindow(
                WINNAME,
                scaled_resolution[0],
                scaled_resolution[1],
            )

        # Save acquired frame to disk
        if do_save_frames:
            if not created_export_folder:
                if not os.path.exists(export_folder):
                    os.makedirs(export_folder)
                created_export_folder = True

            filename = f"frame_{frame_idx:06d}_t_{frame_time:.3f}.png"
            fn_save = os.path.join(export_folder, filename)
            if cv2.imwrite(fn_save, img_gray):
                print(f"Saved {fn_save}")
            else:
                print(f"Failed to save {fn_save}")

        # Histogram
        if do_show_histogram:
            hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
            hist = np.asarray(hist, dtype=np.float32) / img_gray.size
            if not (plt.fignum_exists("Histogram")):
                plt.ion()
                fig = plt.figure("Histogram")
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
                ax.axes.set_ylim([0, max_ylim])
                plt.tight_layout()

            h_hist.set_ydata(hist)  # type: ignore
            fig.canvas.draw_idle()  # type: ignore
            # fig.canvas.flush_events()  # Backend `QtAgg` requires this line

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            do_show_clipped = not do_show_clipped
            print(f"Clip warning   : {bool(do_show_clipped)}")

        elif key == ord("h"):
            do_show_histogram = not do_show_histogram
            print(f"Show histogram : {bool(do_show_histogram)}")
            if (plt.fignum_exists("Histogram")) and not do_show_histogram:
                plt.close("Histogram")

        elif key == ord("s"):
            do_save_frames = not do_save_frames
            print(f"Save to disk   : {bool(do_save_frames)}")

        elif key == ord("t"):
            do_report_frame_dT = not do_report_frame_dT
            print(f"Report frame dT: {bool(do_report_frame_dT)}")

        elif key == ord("q"):
            break

        frame_idx += 1

except KeyboardInterrupt:
    pass

# ------------------------------------------------------------------------------
#   Close
# ------------------------------------------------------------------------------

cv2.destroyAllWindows()
print("\nStopping acquisition... ", end="")
cap.release()
print("done.")
