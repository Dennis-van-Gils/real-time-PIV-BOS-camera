#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

vid = cv2.VideoCapture(0)  # , cv2.CAP_DSHOW)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
vid.set(cv2.CAP_PROP_CONTRAST, 31)
vid.set(cv2.CAP_PROP_SATURATION, 31)
vid.set(cv2.CAP_PROP_GAIN, 127)
vid.set(cv2.CAP_PROP_SHARPNESS, 63)
vid.set(cv2.CAP_PROP_FOCUS, 0)
vid.set(cv2.CAP_PROP_FPS, 30)

print(vid.get(cv2.CAP_PROP_FPS))

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.fliplr(frame)
    # frame = np.asarray(frame, dtype=np.uint8)
    # frame[100:200, 100:200] = 255 - frame[100:200, 100:200]
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
