#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

vid = cv2.VideoCapture(0)

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    np_frame = np.asarray(gray, dtype=np.uint8)
    np_frame[100:200, 100:200] = 255 - np_frame[100:200, 100:200]
    cv2.imshow("frame", np_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
