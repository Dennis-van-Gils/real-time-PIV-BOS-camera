#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides class `FrameServer` which handles serving new image frames, either
coming from files read of disk or from a video capture device such as a
webcamera or other video camera device.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "17-11-2023"
__version__ = "1.0"

import os
import sys
import time

import numpy as np
import numpy.typing as npt
import cv2

import init_config as cfg


if cfg.IMAGE_SOURCE == cfg.IMAGE_SOURCES.XIMEA:
    from utils.ximea import xiapi


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

        count (``int``):
            Number of frames served so far.

        t0 (``float``):
            Time stamp [sec] of first frame captured when `begin()` was called.

        dT (``float``):
            Time interval [sec] between the last two served frames.

        title (``str``):
            String description befitting the last served frame.

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
        self.source: cfg.IMAGE_SOURCES = cfg.IMAGE_SOURCE
        self.cam_cv2: cv2.VideoCapture
        self.cam_xi: xiapi.Camera
        self.img_xi: xiapi.Image

        # Timing
        self.counter: int = 0
        self.t0: float = 0.0
        self.dT: float = 0.0
        self._prev_tick: float = 0.0

        self.title = "Uninitialized"

        # To be derived in `begin()`
        self.img_w: int = 0
        self.img_h: int = 0
        self.img_N_pixels: int = 0
        self.img_bit_depth: int = 0
        self._img_max_bitval: int = 1

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

    # --------------------------------------------------------------------------
    #   begin
    # --------------------------------------------------------------------------

    def begin(self) -> npt.NDArray[np.float32]:
        """Finish setting up the frame server by reading the first image to get
        the image width, height and bit depth. Returns the image in grayscale
        and resets the served images counter `counter` to 0. The counter starts
        increasing when calling `serve()`

        Returns (``np.ndarray(np.float32)``):
            2D numpy array containing the image bitmap in ``numpy.float32``
            grayscale values, normalized by the maximum possible grayscale
            intensity value. Hence, the output range is [0 - 1].
        """
        img = self.serve()
        self.counter = 0  # Don't count `begin()` as a served frame

        return img

    # --------------------------------------------------------------------------
    #   serve
    # --------------------------------------------------------------------------

    def serve(self) -> npt.NDArray[np.float32]:
        """Acquire and return a new grayscale image. Specifically, when
        `cfg.image_source = disk`, the files are returned one by one from the
        list as contained in `cfg.IMAGE_FILES` which is sorted alphabetically.
        Each call to `serve()` increases the served images counter `counter` by
        1.

        Returns (``np.ndarray(np.float32)``):
            2D numpy array containing the image bitmap in ``numpy.float32``
            grayscale values, normalized by the maximum possible grayscale
            intensity value. Hence, the output range is [0 - 1].
        """

        if self.source == cfg.IMAGE_SOURCES.DISK:
            fn_img = cfg.IMAGE_FILES[self.counter]
            img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
            self.title = get_filename_from_full_path(fn_img)

        elif self.source == cfg.IMAGE_SOURCES.WEBCAM:
            success, img = self.cam_cv2.read()
            if success:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                raise Exception("Could not obtain image from camera.")

        elif self.source == cfg.IMAGE_SOURCES.XIMEA:
            self.cam_xi.get_image(self.img_xi)
            img = self.img_xi.get_image_data_numpy()

        else:
            raise ValueError("cfg.image_source falled through without a match.")

        # Count time /after/ we have obtained the new image, not before
        tick = time.perf_counter()

        if self.counter == 0:
            self.t0 = tick
            self._prev_tick = tick

            self.img_h, self.img_w = img.shape
            self.img_N_pixels = self.img_w * self.img_h
            self.img_bit_depth = img[0, 0].nbytes * 8
            self._img_max_bitval = 2**self.img_bit_depth - 1

        # Live video sources
        if self.source in [cfg.IMAGE_SOURCES.WEBCAM, cfg.IMAGE_SOURCES.XIMEA]:
            self.title = f"frame {self.counter:06d} t {tick - self.t0:.3f}"

        self.dT = tick - self._prev_tick
        self._prev_tick = tick
        self.counter += 1

        img = np.asarray(img, dtype=np.float32, order="C")
        img = img / self._img_max_bitval
        return img

    # --------------------------------------------------------------------------
    #   report
    # --------------------------------------------------------------------------

    def report(self):
        print("Frame server")
        print(f"  source    : {self.source.name}")
        if self.source == cfg.IMAGE_SOURCES.DISK:
            print(f"  image_path: {cfg.IMAGE_PATH}")
            print(f"  N_images  : {cfg.N_IMAGES}")
        else:
            print(f"  camera_id : {cfg.CAMERA_ID}")
        print(
            f"  resolution: {self.img_w} x {self.img_h} = "
            f"{self.img_N_pixels/1e6:.1f} Mpx"
        )
        print(f"  bit_depth : {self.img_bit_depth}")

    # --------------------------------------------------------------------------
    #   has_available
    # --------------------------------------------------------------------------

    def has_available(self, number: int) -> bool:
        """When `cfg.image_source = disk`, return the boolean check if there are
        still `number` of frames available to be read from files on disk.
        Otherwise, in case of live video capture, simply return True."""
        if self.source == cfg.IMAGE_SOURCES.DISK:
            return (cfg.N_IMAGES - self.counter) >= number
        else:
            return True

    # --------------------------------------------------------------------------
    #   close
    # --------------------------------------------------------------------------

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
