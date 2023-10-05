#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

# Camera ID
cam_idx = 0

# Desired resolution
# resolution = (1280, 720)  # 0.9 MPx
# resolution = (1600, 896)  # 1.4 MPx
resolution = (1920, 1080)  # 2.1 MPx

# Set equal to the display scaling used in Windows [%]
display_scaling = 125

# Toggle to enable/disable clipping warning
show_clipping = True

# OpenCV window name
WINNAME = "Webcam viewer"

# Open video camera
cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_SETTINGS, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
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

# Fix window size getting multiplied by the Windows display scaling
scaled_resolution = [int(x // (display_scaling / 100)) for x in resolution]
# cv2.namedWindow(WINNAME, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(WINNAME, scaled_resolution[0], scaled_resolution[1])

print("Starting video.")
print(
    "Obtained resolution: "
    f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
)
print(f"Obtained fps: {cap.get(cv2.CAP_PROP_FPS)}")
print("Press c to toggle clip warning.")
print("Press q to exit.")
print(f"Clip warning: {'Enabled' if show_clipping else 'Disabled'}")

try:
    while True:
        success, img_raw = cap.read()
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

        # Recolor clipped intensities as full red
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        clipped_idxs = (img_gray == 255).nonzero()
        img_rgb[clipped_idxs] = [0, 0, 255]  # bgr

        cv2.imshow(WINNAME, img_rgb if show_clipping else img_gray)
        # cv2.resizeWindow(WINNAME, scaled_resolution[0], scaled_resolution[1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            show_clipping = not show_clipping
            print(f"Clip warning: {'Enabled' if show_clipping else 'Disabled'}")

        elif key == ord("q"):
            break

except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()
print("Stopping acquisition...")
cap.release()
print("Done.")
