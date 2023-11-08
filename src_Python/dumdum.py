#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: Read https://stackoverflow.com/questions/41195973/how-to-use-bresenhams-line-drawing-algorithm-with-sub-pixel-bias
"""

import sys
from time import perf_counter

import numpy as np
import numpy.typing as npt
import numba as nb
import cv2

import matplotlib.pyplot as plt

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
    """Draw an 8-connected line into a uint8-grayscale bitmap `img` from point
    `pt1` to `pt2` using the given `color` and `linewidth`.

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
    """Draw an 8-connected line into a [uint8, uint8, uint8]-color bitmap `img`
    from point `pt1` to `pt2` using the given `color` and `linewidth`.

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

    Args:
        img (``numpy.ndarray[np.uint8]``):
            The image as a 2D numpy array containing uint8-grayscale values.
            NOTE: In-place operation on `img`.

        pt1 (``numpy.ndarray[np.uint32]``):
            Start of the quiver as (x, y) coordinate.

        pt2 (``numpy.ndarray[np.uint32]``):
            End of the quiver as (x, y) coordinate.

        color (``int``):
            Color value as `uint8`.

        linewidth (``int``):
            Linewidth of each line segment making up the quiver.

        tip_size (``float``):
            The length of the arrow tip with respect to the arrow length.

        tip_angle (``float``):
            The angle of the arrow tip in radians. E.g., `np.pi/4` for a wide
            tip or `np.pi/8` for a slender tip.
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

    Args:
        img (``numpy.ndarray[np.uint8]``):
            The image as a 2D numpy array containing [uint8, uint8, uint8]-color
            values.
            NOTE: In-place operation on `img`.

        pt1 (``numpy.ndarray[np.uint32]``):
            Start of the quiver as (x, y) coordinate.

        pt2 (``numpy.ndarray[np.uint32]``):
            End of the quiver as (x, y) coordinate.

        color (``numpy.ndarray[np.uint8]``):
            Color value as `[uint8, uint8, uint8]`.

        linewidth (``int``):
            Linewidth of each line segment making up the quiver.

        tip_size (``float``):
            The length of the arrow tip with respect to the arrow length.

        tip_angle (``float``):
            The angle of the arrow tip in radians. E.g., `np.pi/4` for a wide
            tip or `np.pi/8` for a slender tip.
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
    for i in range(len(pts1)):
        pt1 = pts1[i]
        pt2 = pts2[i]
        color = colors[i]
        draw_quiver_u8(img, pt1, pt2, color, linewidth, tip_size, tip_angle)


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
    for i in range(len(pts1)):
        pt1 = pts1[i]
        pt2 = pts2[i]
        color = colors[i]
        draw_quiver_u24(img, pt1, pt2, color, linewidth, tip_size, tip_angle)


# ------------------------------------------------------------------------------
#   demo
# ------------------------------------------------------------------------------


def demo(
    img_w=600,
    img_h=400,
    use_color: bool = True,
    colormap_name: str = "jet",
):
    if use_color:
        img_empty = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    else:
        img_empty = np.zeros((img_h, img_w), dtype=np.uint8)

    # Build color lookup table (lut)
    N_COLORS_LUT = 1024
    mpl_cm = plt.get_cmap(colormap_name, N_COLORS_LUT)
    mpl_cm._init()  # type: ignore
    mpl_lut = mpl_cm._lut  # type: ignore
    cv2_lut = np.asarray(mpl_lut * 255, dtype=np.uint8)  # [0., 1.] to [0, 255]
    cv2_lut = cv2_lut[:, 2::-1]  # Drop `A` from `RGBA`and turn `RGB` into `BGR`

    img_half_w = img_w // 2
    img_half_h = img_h // 2
    img_center_to_corner_distance = np.sqrt(img_half_w**2 + img_half_h**2)

    start_radius = 50
    N_frames = 360

    quiver_kwargs = {
        "linewidth": 2,
        "tip_size": 0.2,
        "tip_angle": np.pi / 4,
    }

    # Create grid of quivers, spaced equally apart
    # --------------------------------------------

    spacing = 50
    d_spacing = np.sqrt(2 * (spacing // 2) ** 2)
    N_quivers_x = int((img_w - spacing) // spacing + 1)
    N_quivers_y = int((img_h - spacing) // spacing + 1)
    N_quivers = N_quivers_x * N_quivers_y

    arr_x = np.arange(N_quivers_x) * spacing + spacing // 2
    arr_y = np.arange(N_quivers_y) * spacing + spacing // 2
    arr_x = np.asarray(arr_x, dtype=np.int32)
    arr_y = np.asarray(arr_y, dtype=np.int32)
    grid_x = np.empty(N_quivers, dtype=np.int32)
    for i in np.arange(N_quivers_y):
        grid_x[i * N_quivers_x : (i + 1) * N_quivers_x] = arr_x
    grid_y = np.repeat(arr_y, N_quivers_x)

    # Animate all quivers by spinning each one revolution
    # ---------------------------------------------------

    img = np.copy(img_empty)
    pts1 = np.zeros((N_quivers, 2), dtype=np.int32)  # Start points
    pts2 = np.zeros((N_quivers * N_frames, 2), dtype=np.int32)  # End points
    colors_u8 = np.ones(N_quivers, dtype=np.uint8)
    colors_u8[:] = 255
    colors_u24 = np.zeros((N_quivers, 3), dtype=np.uint8)
    thetas = np.linspace(0, 2 * np.pi, N_frames)

    for theta_idx, theta in enumerate(thetas):
        for quiver_idx in range(N_quivers):
            # Start point
            x1 = grid_x[quiver_idx]
            y1 = grid_y[quiver_idx]

            # The radius `r` of each quiver depends on its distance `d` from the
            # image center. It falls of towards 0 at the quivers in the very
            # corners.
            d = np.sqrt((img_half_w - x1) ** 2 + (img_half_h - y1) ** 2)
            d = d / (img_center_to_corner_distance - d_spacing)
            r = np.maximum(start_radius * (1 - d), 0)

            # End point
            x2 = np.round(x1 + r * np.sin(theta))
            y2 = np.round(y1 - r * np.cos(theta))

            if theta_idx == 0:
                pts1[quiver_idx][0] = x1
                pts1[quiver_idx][1] = y1

                lut_idx = int(np.round(r / start_radius * (N_COLORS_LUT - 1)))
                colors_u24[quiver_idx] = cv2_lut[lut_idx]

            pts2[quiver_idx + theta_idx * N_quivers][0] = x2
            pts2[quiver_idx + theta_idx * N_quivers][1] = y2

    # Pure draw and plot
    # ------------------
    PLOT_PER_QUIVER = 0

    tick = perf_counter()
    for theta_idx, theta in enumerate(thetas):
        np.copyto(img, img_empty)

        if PLOT_PER_QUIVER:
            for quiver_idx in range(N_quivers):
                pt1 = pts1[quiver_idx]
                pt2 = pts2[quiver_idx + theta_idx * N_quivers]

                if use_color:
                    color = colors_u24[quiver_idx]
                    draw_quiver_u24(img, pt1, pt2, color, **quiver_kwargs)
                else:
                    color = colors_u8[quiver_idx]
                    draw_quiver_u8(img, pt1, pt2, color, **quiver_kwargs)

        else:
            pts2_set = pts2[theta_idx * N_quivers : (theta_idx + 1) * N_quivers]

            if use_color:
                draw_quiver_map_u24(img, pts1, pts2_set, colors_u24, **quiver_kwargs)
            else:
                draw_quiver_map_u8(img, pts1, pts2_set, colors_u8, **quiver_kwargs)

        if 1:
            cv2.imshow("output", img)
            cv2.setWindowTitle("output", f"{theta * 180 / np.pi:.1f}")
            cv2.waitKey(1)

    duration = perf_counter() - tick
    ms_per_frame = duration / N_frames * 1000
    ms_per_quiver = duration / N_frames / N_quivers * 1000
    print(f"Per frame : {ms_per_frame :.4f} ms")
    print(f"Per quiver: {ms_per_quiver:.4f} ms")


# ------------------------------------------------------------------------------
#   main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("__main__")
    sys.stdout.flush()

    img_w, img_h = 600, 400
    demo(img_w, img_h, use_color=False)
    demo(img_w, img_h, colormap_name="jet")
