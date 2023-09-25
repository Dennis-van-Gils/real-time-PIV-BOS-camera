#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

resolution = (1280, 720)
resolution = (1920, 1080)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
# cap.set(cv2.CAP_PROP_SETTINGS, 1)
# cap.set(cv2.CAP_PROP_AUTO_WB, 0)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
# cap.set(cv2.CAP_PROP_CONTRAST, 31)
# cap.set(cv2.CAP_PROP_SATURATION, 31)
# cap.set(cv2.CAP_PROP_GAIN, 127)
# cap.set(cv2.CAP_PROP_SHARPNESS, 63)
# cap.set(cv2.CAP_PROP_FOCUS, 0)
# cap.set(cv2.CAP_PROP_FPS, 30)

print(f"Target fps: {cap.get(cv2.CAP_PROP_FPS)}")

# Fix window size getting multiplied by the Windows display scaling
display_scaling = 125  # Set equal to the display scaling used in Windows [%]
scaled_resolution = [int(x // (display_scaling / 100)) for x in resolution]
cv2.namedWindow("main", cv2.WINDOW_NORMAL)
cv2.resizeWindow("main", scaled_resolution[0], scaled_resolution[1])

while True:
    # Capture the video frame by frame
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.fliplr(frame)
    # frame = np.asarray(frame, dtype=np.uint8)
    # frame[100:200, 100:200] = 255 - frame[100:200, 100:200]

    cv2.imshow("main", frame)
    cv2.resizeWindow("main", scaled_resolution[0], scaled_resolution[1])

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
