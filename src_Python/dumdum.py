#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import perf_counter
import numpy as np
import numpy.typing as npt
import numba as nb
import cv2

# ------------------------------------------------------------------------------
#   draw_8line
# ------------------------------------------------------------------------------


@nb.njit(
    "(uint8[:, :], int32[:], int32[:], int32)",
    cache=True,
    nogil=True,
)
def draw_8line(
    img: npt.NDArray[np.uint8],
    p1: npt.NDArray[np.int32],
    p2: npt.NDArray[np.int32],
    width: int,
):
    """Bresenham's line algorithm with support for custom linewidths."""

    # Calculate the half-width for the line
    half_width = width // 2

    # Bresenham's line algorithm
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        for i in range(-half_width, half_width + 1):
            for j in range(-half_width, half_width + 1):
                px = x1 + i
                py = y1 + j

                if px >= 0 and px < img.shape[1] and py >= 0 and py < img.shape[0]:
                    img[py, px] = 255

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
#   draw_8line_with_arrowhead
# ------------------------------------------------------------------------------


@nb.njit(
    "(uint8[:, :], int32[:], int32[:], int32, int32)",
    cache=True,
    nogil=True,
)
def draw_8line_with_arrowhead(
    img: npt.NDArray[np.uint8],
    p1: npt.NDArray[np.int32],
    p2: npt.NDArray[np.int32],
    width: int,
    arrowhead_size: int,
):
    # Calculate arrowhead points
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.sqrt(dx**2 + dy**2)
    adx = arrowhead_size * dx / length
    ady = arrowhead_size * dy / length

    a1 = np.array((p2[0] - adx + ady, p2[1] - ady - adx), dtype=np.int32)
    a2 = np.array((p2[0] - adx - ady, p2[1] - ady + adx), dtype=np.int32)

    draw_8line(img, p1, p2, width)
    draw_8line(img, a1, p2, width)
    draw_8line(img, a2, p2, width)


# ------------------------------------------------------------------------------
#   main
# ------------------------------------------------------------------------------

center = np.array([200, 200], dtype=np.int32)
radius = 100
N_points = 360
linewidth = 2
arrowhead_size = 20

img_empty = np.zeros((400, 400), dtype=np.uint8)
img = np.copy(img_empty)
p2 = np.array([0, 0], dtype=np.int32)

tick = perf_counter()
for theta in np.linspace(0, 2 * np.pi, N_points):
    np.copyto(img, img_empty)
    p2[0] = np.round(center[0] + radius * np.sin(theta))
    p2[1] = np.round(center[1] + radius * np.cos(theta))

    draw_8line_with_arrowhead(
        img,
        p1=center,
        p2=p2,
        width=linewidth,
        arrowhead_size=arrowhead_size,
    )

    if 1:
        cv2.imshow("output", img)
        cv2.setWindowTitle("output", f"{theta * 180 / np.pi:.1f}")
        cv2.waitKey(1)

print(f"{(perf_counter() - tick)/N_points*1000:.3f} ms")
