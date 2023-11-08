#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO: Read https://stackoverflow.com/questions/41195973/how-to-use-bresenhams-line-drawing-algorithm-with-sub-pixel-bias
"""

from time import perf_counter
import numpy as np
import numpy.typing as npt
import numba as nb
import cv2

# ------------------------------------------------------------------------------
#   draw_8line_u8
# ------------------------------------------------------------------------------


@nb.njit(
    "(uint8[:, :], int32[:], int32[:], uint8, int32)",
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
    `pt1` to `pt2` using the given uint8 `color` and `linewidth`.

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
#   draw_quiver_u8
# ------------------------------------------------------------------------------


@nb.njit(
    "(uint8[:, :], int32[:], int32[:], uint8, int32, float32, float32)",
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
    """Draw a quiver (a line ending in an arrow tip) into a `uint8` grayscale
    image.

    Args:
        img (``numpy.ndarray[np.uint8]``):
            The grayscale image as a 2D numpy array containing `uint8` values.
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
    length = np.sqrt(dx**2 + dy**2)  # TODO: Turn into input arg
    angle = np.arctan2(-dy, -dx)  # TODO: Turn into input arg
    tip_size = length * tip_size

    draw_8line_u8(img, pt1, pt2, color, linewidth)

    tip = np.array(
        (
            np.round(x2 + tip_size * np.cos(angle + tip_angle)),
            np.round(y2 + tip_size * np.sin(angle + tip_angle)),
        ),
        dtype=np.int32,
    )
    draw_8line_u8(img, tip, pt2, color, linewidth)

    tip[0] = np.round(x2 + tip_size * np.cos(angle - tip_angle))
    tip[1] = np.round(y2 + tip_size * np.sin(angle - tip_angle))
    draw_8line_u8(img, tip, pt2, color, linewidth)


# ------------------------------------------------------------------------------
#   main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    img_w, img_h = 600, 400
    img_empty = np.zeros((img_h, img_w), dtype=np.uint8)

    img_half_w = img_w // 2
    img_half_h = img_h // 2
    img_center_to_corner_distance = np.sqrt(img_half_w**2 + img_half_h**2)

    start_radius = 50
    N_circle_points = 360

    color = 255
    linewidth = 2
    tip_size = 0.2
    tip_angle = np.pi / 4

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

    # Animate all quivers by spinning each
    # ------------------------------------

    img = np.copy(img_empty)
    pt1 = np.zeros((N_quivers, 2), dtype=np.int32)  # Start points
    pt2 = np.zeros((N_quivers * N_circle_points, 2), dtype=np.int32)  # End points
    thetas = np.linspace(0, 2 * np.pi, N_circle_points)

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
            r = start_radius * (1 - d)

            # End point
            x2 = np.round(x1 + r * np.sin(theta))
            y2 = np.round(y1 - r * np.cos(theta))

            pt1[quiver_idx][0] = x1
            pt1[quiver_idx][1] = y1
            pt2[quiver_idx + theta_idx * N_quivers][0] = x2
            pt2[quiver_idx + theta_idx * N_quivers][1] = y2

    # Pure draw and plot
    # ------------------

    tick = perf_counter()
    for theta_idx, theta in enumerate(thetas):
        np.copyto(img, img_empty)

        for quiver_idx in range(N_quivers):
            draw_quiver_u8(
                img,
                pt1=pt1[quiver_idx],
                pt2=pt2[quiver_idx + theta_idx * N_quivers],
                color=color,
                linewidth=linewidth,
                tip_size=tip_size,
                tip_angle=tip_angle,
            )

        if 1:
            cv2.imshow("output", img)
            cv2.setWindowTitle("output", f"{theta * 180 / np.pi:.1f}")
            cv2.waitKey(1)

    ms_per_quiver = (perf_counter() - tick) / N_circle_points / N_quivers * 1000
    print(f"Per quiver: {ms_per_quiver:.4f} ms")
