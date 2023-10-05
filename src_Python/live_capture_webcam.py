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

# Toggle to save acquired frames to disk
do_save_frames = False

# Toggle reporting frame time intervals to the terminal
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
print("  s | Toggle saving frames to disk.")
print("  t | Toggle reporting frame dT in [ms].")
print("  q | Quit.")
print("")
print(f"Clip warning   : {bool(do_show_clipped)}")
print(f"Saving to disk : {bool(do_save_frames)}")
print(f"Report frame dT: {bool(do_report_frame_dT)}")

frame_idx = 0
frame_t0 = time.perf_counter()  # [sec]
frame_time = frame_time_prev = frame_t0  # [sec]
try:
    while True:
        # Acquire frame
        success, img_raw = cap.read()

        frame_time_prev = frame_time
        frame_time = time.perf_counter() - frame_t0
        frame_dt = frame_time - frame_time_prev
        if do_report_frame_dT:
            print(f"{frame_dt*1000:.0f}")

        # Turn into grayscale
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

        # Recolor clipped intensities as full red
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        clipped_idxs = (img_gray == 255).nonzero()
        img_rgb[clipped_idxs] = [0, 0, 255]  # bgr

        # Show
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

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            do_show_clipped = not do_show_clipped
            print(f"Clip warning   : {bool(do_show_clipped)}")

        elif key == ord("s"):
            do_save_frames = not do_save_frames
            print(f"Saving to disk : {bool(do_save_frames)}")

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
