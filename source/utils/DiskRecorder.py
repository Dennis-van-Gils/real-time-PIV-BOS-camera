#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides class `DiskRecorder` which handles

"Record frames to disk"
- writes image frames to disk
- writes vector results to disk as text

TODO notes:

Class DiskRecorder:

Member vars
- config.MODES: mode
- array: BOS_frame_0
- int  : frame_counter_since_start_recording
- str  : export folder for vector maps
- str  : export folder for original video frames


In main:
If IMAGE_SOURCE == DISK, assume we only want to write vector maps to disk, not
the original video frames. Start export automatically directly at start of main
loop.

Methods
- start
  Creates output folders
  Writes config file to output folder (convenience)
  Updates and writes current BOS_frame_0 to disk

- stop


- write_image(canvas)


"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/real-time-PIV-BOS-camera"
__date__ = "20-12-2023"

import numpy as np
import cv2


class DiskRecorder:
    def __init__(self):
        pass
