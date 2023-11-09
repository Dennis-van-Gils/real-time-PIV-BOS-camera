#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides numba-accelerated functions to plot complete quiver maps into numpy
raster images. To be used in conjuction with Matplotlib or OpenCV.

- `draw_quiver_map_u8(...)` for uint8-grayscale images.
- `draw_quiver_map_u24(...)` for [uint8, uint8, uint8]-color images.

These functions outperform their Matplotlib and OpenCV counterparts,
respectively `matplotib.pyplot.quiver()` and `cv2.arrowedLine()`, because we
directly plot all quivers into the passed image in one go, all running in
no-Python mode and with the GIL released.

NOTE: In contrast to Matplotlib, here the quivers get 'baked' into the raster
image as similar to OpenCV.
NOTE: The function parameters strictly expect specific data types as argument,
such as `np.int32` and `np.uint8`, and expect "C"-style contiguous numpy arrays
to ensure maximum performance.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "09-11-2023"
__version__ = "1.0"

import numpy as np
import numpy.typing as npt
import numba as nb


MINIMUM_QUIVER_TIP_SIZE = 5

# ------------------------------------------------------------------------------
#   draw_8line_u8
# ------------------------------------------------------------------------------


@nb.njit(
    (
        nb.types.Array(nb.uint8, 2, "C"),
        nb.types.Array(nb.int32, 1, "C"),
        nb.types.Array(nb.int32, 1, "C"),
        nb.uint8,
        nb.int32,
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def draw_8line_u8(
    img: npt.NDArray[np.uint8],
    pt1: npt.NDArray[np.int32],
    pt2: npt.NDArray[np.int32],
    color: int,
    linewidth: int,
):
    """Draw an 8-connected line into a uint8-grayscale image.
    NOTE: In-place operation on `img`.
    """
    # Bresenham's line algorithm
    hw = linewidth // 2  # Half-width
    x1, y1 = pt1
    x2, y2 = pt2
    dx = np.abs(x2 - x1)
    dy = np.abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        for i in range(-hw, hw + 1):
            for j in range(-hw, hw + 1):
                px = x1 + i
                py = y1 + j

                # fmt: off
                if (px >= 0 and px < img.shape[1] and \
                    py >= 0 and py < img.shape[0]):
                    img[py, px] = color
                # fmt: on

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


# ------------------------------------------------------------------------------
#   draw_8line_u24
# ------------------------------------------------------------------------------


@nb.njit(
    (
        nb.types.Array(nb.uint8, 3, "C"),
        nb.types.Array(nb.int32, 1, "C"),
        nb.types.Array(nb.int32, 1, "C"),
        nb.types.Array(nb.uint8, 1, "C"),
        nb.int32,
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def draw_8line_u24(
    img: npt.NDArray[np.uint8],
    pt1: npt.NDArray[np.int32],
    pt2: npt.NDArray[np.int32],
    color: npt.NDArray[np.uint8],
    linewidth: int,
):
    """Draw an 8-connected line into a [uint8, uint8, uint8]-color image.
    NOTE: In-place operation on `img`.
    """
    # Bresenham's line algorithm
    hw = linewidth // 2  # Half-width
    x1, y1 = pt1
    x2, y2 = pt2
    dx = np.abs(x2 - x1)
    dy = np.abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        for i in range(-hw, hw + 1):
            for j in range(-hw, hw + 1):
                px = x1 + i
                py = y1 + j

                # fmt: off
                if (px >= 0 and px < img.shape[1] and \
                    py >= 0 and py < img.shape[0]):
                    img[py, px, 0] = color[0]
                    img[py, px, 1] = color[1]
                    img[py, px, 2] = color[2]
                # fmt: on

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


# ------------------------------------------------------------------------------
#   draw_quiver_u8
# ------------------------------------------------------------------------------


@nb.njit(
    (
        nb.types.Array(nb.uint8, 2, "C"),
        nb.types.Array(nb.int32, 1, "C"),
        nb.types.Array(nb.int32, 1, "C"),
        nb.uint8,
        nb.int32,
        nb.float32,
        nb.float32,
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def draw_quiver_u8(
    img: npt.NDArray[np.uint8],
    pt1: npt.NDArray[np.int32],
    pt2: npt.NDArray[np.int32],
    color: int,
    linewidth: int,
    tip_size: float,
    tip_angle: float,
):
    """Draw a quiver (a line ending in an arrow tip) into a uint8-grayscale
    image.
    NOTE: In-place operation on `img`.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(-dy, -dx)
    tip_size = length * tip_size

    if tip_size < MINIMUM_QUIVER_TIP_SIZE and tip_size != 0:
        tip_size = MINIMUM_QUIVER_TIP_SIZE

    draw_8line_u8(img, pt1, pt2, color, linewidth)
    tip = np.empty(2, dtype=np.int32)
    tip[0] = np.round(x2 + tip_size * np.cos(angle + tip_angle))
    tip[1] = np.round(y2 + tip_size * np.sin(angle + tip_angle))
    draw_8line_u8(img, tip, pt2, color, linewidth)
    tip[0] = np.round(x2 + tip_size * np.cos(angle - tip_angle))
    tip[1] = np.round(y2 + tip_size * np.sin(angle - tip_angle))
    draw_8line_u8(img, tip, pt2, color, linewidth)


# ------------------------------------------------------------------------------
#   draw_quiver_u24
# ------------------------------------------------------------------------------


@nb.njit(
    (
        nb.types.Array(nb.uint8, 3, "C"),
        nb.types.Array(nb.int32, 1, "C"),
        nb.types.Array(nb.int32, 1, "C"),
        nb.types.Array(nb.uint8, 1, "C"),
        nb.int32,
        nb.float32,
        nb.float32,
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def draw_quiver_u24(
    img: npt.NDArray[np.uint8],
    pt1: npt.NDArray[np.int32],
    pt2: npt.NDArray[np.int32],
    color: npt.NDArray[np.uint8],
    linewidth: int,
    tip_size: float,
    tip_angle: float,
):
    """Draw a quiver (a line ending in an arrow tip) into a [uint8, uint8,
    uint8]-color image.
    NOTE: In-place operation on `img`.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(-dy, -dx)
    tip_size = length * tip_size

    if tip_size < MINIMUM_QUIVER_TIP_SIZE and tip_size != 0:
        tip_size = MINIMUM_QUIVER_TIP_SIZE

    draw_8line_u24(img, pt1, pt2, color, linewidth)
    tip = np.empty(2, dtype=np.int32)
    tip[0] = np.round(x2 + tip_size * np.cos(angle + tip_angle))
    tip[1] = np.round(y2 + tip_size * np.sin(angle + tip_angle))
    draw_8line_u24(img, tip, pt2, color, linewidth)
    tip[0] = np.round(x2 + tip_size * np.cos(angle - tip_angle))
    tip[1] = np.round(y2 + tip_size * np.sin(angle - tip_angle))
    draw_8line_u24(img, tip, pt2, color, linewidth)


# ------------------------------------------------------------------------------
#   draw_quiver_map_u8
# ------------------------------------------------------------------------------


@nb.njit(
    (
        nb.types.Array(nb.uint8, 2, "C"),
        nb.types.Array(nb.int32, 2, "C"),
        nb.types.Array(nb.int32, 2, "C"),
        nb.types.Array(nb.uint8, 1, "C"),
        nb.int32,
        nb.float32,
        nb.float32,
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def draw_quiver_map_u8(
    img: npt.NDArray[np.uint8],
    pts1: npt.NDArray[np.int32],
    pts2: npt.NDArray[np.int32],
    colors: npt.NDArray[np.uint8],
    linewidth: int,
    tip_size: float,
    tip_angle: float,
):
    """Draw a list of quivers (quiver: a line ending in an arrow tip) into a
    uint8-grayscale image.

    Args:
        img (``numpy.ndarray[np.uint8]``):
            The image as an array of shape (N_pixels_y, N_pixels_x) containing
            uint8-grayscale values.
            NOTE: In-place operation on `img`.

        pts1 (``numpy.ndarray[np.int32]``):
            Array of shape (N_quivers, 2) containing the start coordinates per
            quiver as [x, y].

        pts2 (``numpy.ndarray[np.int32]``):
            Array of shape (N_quivers, 2) containing the end coordinates per
            quiver as [x, y].

        colors (``numpy.ndarray[np.uint8]``):
            Array of shape (N_quivers,) containing uint8-grayscale values.

        linewidth (``int``):
            Linewidth of the quiver.

        tip_size (``float``):
            The length of the arrow tip with respect to the arrow length.

        tip_angle (``float``):
            The angle of the arrow tip in radians. E.g., `np.pi/4` for a wide
            tip or `np.pi/8` for a slender tip.
    """
    for i in range(len(pts1)):
        draw_quiver_u8(
            img,
            pts1[i],
            pts2[i],
            colors[i],
            linewidth,
            tip_size,
            tip_angle,
        )


# ------------------------------------------------------------------------------
#   draw_quiver_map_u24
# ------------------------------------------------------------------------------


@nb.njit(
    (
        nb.types.Array(nb.uint8, 3, "C"),
        nb.types.Array(nb.int32, 2, "C"),
        nb.types.Array(nb.int32, 2, "C"),
        nb.types.Array(nb.uint8, 2, "C"),
        nb.int32,
        nb.float32,
        nb.float32,
    ),
    cache=True,
    nogil=True,
    fastmath=True,
)
def draw_quiver_map_u24(
    img: npt.NDArray[np.uint8],
    pts1: npt.NDArray[np.int32],
    pts2: npt.NDArray[np.int32],
    colors: npt.NDArray[np.uint8],
    linewidth: int,
    tip_size: float,
    tip_angle: float,
):
    """Draw a list of quivers (quiver: a line ending in an arrow tip) into a
    [uint8, uint8, uint8]-color image.

    Args:
        img (``numpy.ndarray[np.uint8]``):
            The image as an array of shape (N_pixels_y, N_pixels_x, 3)
            containing [uint8, uint8, uint8]-color values.
            NOTE: In-place operation on `img`.

        pts1 (``numpy.ndarray[np.int32]``):
            Array of shape (N_quivers, 2) containing the start coordinates per
            quiver as [x, y].

        pts2 (``numpy.ndarray[np.int32]``):
            Array of shape (N_quivers, 2) containing the end coordinates per
            quiver as [x, y].

        colors (``numpy.ndarray[np.uint8]``):
            Array of shape (N_quivers, 3) containing [uint8, uint8, uint8]-color
            values.

        linewidth (``int``):
            Linewidth of the quiver.

        tip_size (``float``):
            The length of the arrow tip with respect to the arrow length.

        tip_angle (``float``):
            The angle of the arrow tip in radians. E.g., `np.pi/4` for a wide
            tip or `np.pi/8` for a slender tip.
    """
    for i in range(len(pts1)):
        draw_quiver_u24(
            img,
            pts1[i],
            pts2[i],
            colors[i],
            linewidth,
            tip_size,
            tip_angle,
        )
