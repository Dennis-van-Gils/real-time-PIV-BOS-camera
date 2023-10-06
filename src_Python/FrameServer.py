#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides class `FrameServer` which handles serving new image frames, either
coming from files read of disk or from a video capture device such as a
webcamera or other video camera device.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "06-10-2023"
__version__ = "1.0"

import os
import sys
import time

import numpy as np
import numpy.typing as npt
import cv2

import init_config as cfg


if cfg.IMAGE_SOURCE == cfg.IMAGE_SOURCES.XIMEA:
    from ximea import xiapi


# ------------------------------------------------------------------------------
#   FrameServer
# ------------------------------------------------------------------------------


class FrameServer:
    """Manages serving new image frames, either coming from files read of disk
    or from a video capture device such as a webcamera or other video camera
    device.

    Method `begin()` must be called to finish setting up the frame server.

    Members:
        source (``init_config.IMAGE_SOURCE``):
            Enumeration indicating the video capture source to use. It is read
            from a configuration file on disk.

        cam_cv2 (``cv2.VideoCapture``):
            OpenCV video capture instance when `source` is "webcam".

        cam_xi (``xiapi.Camera``):
            Ximea video capture instance when `source` is "ximea".

        frame_count (``int``):
            Number of frames served so far.

        frame_t0 (``float``):
            Time stamp [sec] of first frame captured when `begin()` was called.

        frame_dT (``float``):
            Time interval [sec] between the last two served frames.

        img_w (``int``):
            Obtained image frame width [px].

        img_h (``int``):
            Obtained image frame height [px].

        img_N_pixels (``int``):
            Total number of pixels of the obtained image frame [px].

        img_bit_depth (``int``):
            Bit depth of the raw obtained image frame.
    """

    def __init__(self):
        self.source = cfg.IMAGE_SOURCE
        self.cam_cv2: cv2.VideoCapture
        self.cam_xi: xiapi.Camera
        self.img_xi: xiapi.Image

        # Timing
        self.frame_count: int = 0
        self.frame_t0: float = 0.0
        self.frame_dT: float = 0.0
        self._prev_tick_frame: float = 0.0

        # To be derived in `begin()`
        self.img_w = 0
        self.img_h = 0
        self.img_N_pixels = 0
        self.img_bit_depth = 0
        self._img_max_bitval = 1

        # TODO: Turn into config parameters
        Ximea_exposure = 20000  # [us]

        if self.source == cfg.IMAGE_SOURCES.DISK:
            pass  # Nothing special to do here

        elif self.source == cfg.IMAGE_SOURCES.WEBCAM:
            if sys.platform == "win32":
                self.cam_cv2 = cv2.VideoCapture(cfg.CAMERA_ID, cv2.CAP_DSHOW)
            else:
                self.cam_cv2 = cv2.VideoCapture(cfg.CAMERA_ID)

            self.cam_cv2.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                cfg.WANTED_RESOLUTION[0],
            )
            self.cam_cv2.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                cfg.WANTED_RESOLUTION[1],
            )

        elif self.source == cfg.IMAGE_SOURCES.XIMEA:
            self.cam_xi = xiapi.Camera()
            self.cam_xi.open_device()
            self.cam_xi.set_exposure(Ximea_exposure)
            self.cam_xi.start_acquisition()
            self.img_xi = xiapi.Image()

    def begin(self) -> tuple[npt.NDArray[np.float32], str]:
        """Finish setting up the frame server by reading the first image to get
        the image width, height and bit depth. Returns the first image and a
        befitting string description.

        Returns (``tuple``):
            img (``np.ndarray(np.float32)``):
                2D numpy array containing the image bitmap in ``numpy.float32``
                grayscale values, normalized by the maximum possible grayscale
                intensity value. Hence, the output range is [0 - 1].

            frame_title (``str``):
                String description befitting the served image.
        """
        return self.serve(0)

    def serve(self, frame_idx: int = 1) -> tuple[npt.NDArray[np.float32], str]:
        """Acquire and return a new grayscale image frame and a befitting string
        description. The returned image data is rescaled to the interval [0 - 1]
        regardless of the original format.

        Args:
            frame_idx (``int``, optional):
                When the images are read from files on disk, ``frame_idx``
                indicates which file of the image list to read. Otherwise,
                ``frame_idx`` gets ignored.

                Default: 1

        Returns (``tuple``):
            img (``np.ndarray(np.float32)``):
                2D numpy array containing the image bitmap in ``numpy.float32``
                grayscale values, normalized by the maximum possible grayscale
                intensity value. Hence, the output range is [0 - 1].

            frame_title (``str``):
                String description befitting the served image.
        """

        if self.source == cfg.IMAGE_SOURCES.DISK:
            fn_img = cfg.IMAGE_FILES[frame_idx]
            img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
            tick = time.perf_counter()
            frame_title = get_filename_from_full_path(fn_img)

        elif self.source == cfg.IMAGE_SOURCES.WEBCAM:
            success, img = self.cam_cv2.read()
            tick = time.perf_counter()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frame_title = ""

        elif self.source == cfg.IMAGE_SOURCES.XIMEA:
            self.cam_xi.get_image(self.img_xi)
            tick = time.perf_counter()
            img = self.img_xi.get_image_data_numpy()
            frame_title = ""

        else:
            img = np.zeros([1, 1], dtype=np.float32)
            tick = time.perf_counter()
            frame_title = "Empty frame. Should never see me."

        # Live video sources: Keep track of time
        if self.source in [cfg.IMAGE_SOURCES.WEBCAM, cfg.IMAGE_SOURCES.XIMEA]:
            if frame_idx == 0:
                self.frame_count = 0
                self.frame_t0 = tick
                self._prev_tick_frame = tick
                frame_time = 0
            else:
                self.frame_count += 1
                self.frame_dT = tick - self._prev_tick_frame
                self._prev_tick_frame = tick
                frame_time = tick - self.frame_t0

            frame_title = f"frame {self.frame_count:06d} t {frame_time:.3f}"

        if frame_idx == 0:
            self.img_h, self.img_w = img.shape
            self.img_N_pixels = self.img_w * self.img_h
            self.img_bit_depth = img[0, 0].nbytes * 8
            self._img_max_bitval = 2**self.img_bit_depth - 1

        img = np.asarray(img, dtype=np.float32, order="C")
        img = img / self._img_max_bitval

        return img, frame_title

    def close(self):
        """Close video capture device if it was opened."""
        if self.source == cfg.IMAGE_SOURCES.WEBCAM:
            self.cam_cv2.release()

        elif self.source == cfg.IMAGE_SOURCES.XIMEA:
            self.cam_xi.stop_acquisition()
            self.cam_xi.close_device()


# ------------------------------------------------------------------------------
#   get_filename_from_full_path
# ------------------------------------------------------------------------------


def get_filename_from_full_path(p: str):
    return os.path.normpath(p).split(os.path.sep)[-1]
