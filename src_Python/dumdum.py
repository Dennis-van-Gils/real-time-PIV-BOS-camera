#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
)
def draw_8line_u8(
    img: npt.NDArray[np.uint8],
    pt1: npt.NDArray[np.int32],
    pt2: npt.NDArray[np.int32],
    color: int,
    thickness: int,
):
    """Draw an 8-connected line into a uint8-grayscale bitmap `img` from point
    `pt1` to `pt2` using the given uint8 `color` and `thickness`.

    NOTE: In-place operation on `img`.
    NOTE: Type-specific drop-in replacement for `cv2.line()`.
    """
    # Bresenham's line algorithm
    hw = thickness // 2  # Half-width
    x1, y1 = pt1
    x2, y2 = pt2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
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
    "(uint8[:, :], int32[:], int32[:], uint8, int32, int32)",
    cache=True,
    nogil=True,
)
def draw_quiver_u8(
    img: npt.NDArray[np.uint8],
    pt1: npt.NDArray[np.int32],
    pt2: npt.NDArray[np.int32],
    color: int,
    thickness: int,
    tipLength: float,
):
    """Draw a quiver (an 8-connected line ending with an arrowhead) into a
    uint8-grayscale bitmap `img` from point `pt1` to `pt2` using the given uint8
    `color` and `thickness`. `tipLength` denotes the length of the arrow tip in
    relation to the arrow length.

    NOTE: In-place operation on `img`.
    NOTE: Type-specific drop-in replacement for `cv2.arrowedLine()`.
    """
    # Calculate arrowhead points
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = np.sqrt(dx**2 + dy**2)
    adx = tipLength * dx / length
    ady = tipLength * dy / length

    a1 = np.array((pt2[0] - adx + ady, pt2[1] - ady - adx), dtype=np.int32)
    a2 = np.array((pt2[0] - adx - ady, pt2[1] - ady + adx), dtype=np.int32)

    draw_8line_u8(img, pt1, pt2, color, thickness)
    draw_8line_u8(img, a1, pt2, color, thickness)
    draw_8line_u8(img, a2, pt2, color, thickness)


# ------------------------------------------------------------------------------
#   main
# ------------------------------------------------------------------------------

radius = 100
center = np.array([200, 200], dtype=np.int32)
N_points = 360

color = 255
thickness = 2
arrowhead_size = 20

img_empty = np.zeros((400, 400), dtype=np.uint8)
img = np.copy(img_empty)
pt2 = np.array([0, 0], dtype=np.int32)

tick = perf_counter()
for theta in np.linspace(0, 2 * np.pi, N_points):
    np.copyto(img, img_empty)
    pt2[0] = np.round(center[0] + radius * np.sin(theta))
    pt2[1] = np.round(center[1] + radius * np.cos(theta))

    draw_quiver_u8(
        img,
        pt1=center,
        pt2=pt2,
        color=color,
        thickness=thickness,
        tipLength=arrowhead_size,
    )

    if 1:
        cv2.imshow("output", img)
        cv2.setWindowTitle("output", f"{theta * 180 / np.pi:.1f}")
        cv2.waitKey(1)

print(f"{(perf_counter() - tick)/N_points*1000:.3f} ms")
