#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides class `FrameServer` which handles serving new image frames, either
coming from files read of disk or from a video capture device such as a
webcamera or other video camera device.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "03-10-2023"
__version__ = "1.0"

import os
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
    def __init__(self):
        """Serves grayscale images as numpy np.float32 arrays."""
        self.source = cfg.IMAGE_SOURCE
        self.cam_cv2: cv2.VideoCapture
        self.cam_xi: xiapi.Camera
        self.img_xi: xiapi.Image
        self.frame_timer_t0: float = 0
        self.frame_counter: int = 0

        # To be derived in `begin()`
        self.img_h = 0
        self.img_w = 0
        self.img_bit_depth = 0
        self.img_max_value = 1

        # TODO: Turn into config parameters
        resolution = (1920, 1080)
        Ximea_exposure = 20000  # [us]

        if self.source == cfg.IMAGE_SOURCES.DISK:
            pass  # Nothing special to do here

        elif self.source == cfg.IMAGE_SOURCES.WEBCAM:
            self.cam_cv2 = cv2.VideoCapture(cfg.CAMERA_ID)  # , cv2.CAP_DSHOW)
            self.cam_cv2.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cam_cv2.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        elif self.source == cfg.IMAGE_SOURCES.XIMEA:
            self.cam_xi = xiapi.Camera()
            self.cam_xi.open_device()
            self.cam_xi.set_exposure(Ximea_exposure)
            self.cam_xi.start_acquisition()
            self.img_xi = xiapi.Image()

    def begin(self) -> tuple[npt.NDArray[np.float32], str]:
        """Set up the frame server by reading the first image to get the image
        width, height and bit depth. Returns the first image and a string
        description.

        Returns (``tuple``):
            img (``np.ndarray(np.float32)``):
                2D numpy array containing the image bitmap in ``numpy.float32``
                grayscale values, normalized by the maximum possible grayscale
                intensity value.

            frame_title (``str``):
                String description befitting the served image.
        """
        return self.serve(0)

    def serve(self, frame_idx: int) -> tuple[npt.NDArray[np.float32], str]:
        """Return a new image frame and a befitting string description.

        Args:
            frame_idx (``int``):
                When the images are read from files on disk, ``frame_idx``
                indicates which file of the image list to read. Otherwise,
                ``frame_idx`` gets ignored.

        Returns (``tuple``):
            img (``np.ndarray(np.float32)``):
                2D numpy array containing the image bitmap in ``numpy.float32``
                grayscale values, normalized by the maximum possible grayscale
                intensity value.

            frame_title (``str``):
                String description befitting the served image.
        """

        if self.source == cfg.IMAGE_SOURCES.DISK:
            fn_img = cfg.IMAGE_FILES[frame_idx]
            img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
            frame_title = get_filename_from_full_path(fn_img)

        elif self.source == cfg.IMAGE_SOURCES.WEBCAM:
            success, img = self.cam_cv2.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if frame_idx == 0:
                self.frame_counter = 0
                self.frame_timer_t0 = time.perf_counter()
                frame_time = 0
            else:
                self.frame_counter += 1
                frame_time = time.perf_counter() - self.frame_timer_t0

            frame_title = f"frame {self.frame_counter}: {frame_time: 5.3f} s"

        elif self.source == cfg.IMAGE_SOURCES.XIMEA:
            self.cam_xi.get_image(self.img_xi)
            img = self.img_xi.get_image_data_numpy()

            if frame_idx == 0:
                self.frame_counter = 0
                self.frame_timer_t0 = time.perf_counter()
                frame_time = 0
            else:
                self.frame_counter += 1
                frame_time = time.perf_counter() - self.frame_timer_t0

            frame_title = f"frame {self.frame_counter}: {frame_time: 5.3f} s"

        else:
            img = np.zeros([1, 1], dtype=np.float32)
            frame_title = "Empty frame. Should never see me."

        if frame_idx == 0:
            self.img_h, self.img_w = img.shape
            self.img_bit_depth = img[0, 0].nbytes * 8
            self.img_max_value = 2**self.img_bit_depth - 1

        img = np.asarray(img, dtype=np.float32, order="C")
        img = img / self.img_max_value

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
