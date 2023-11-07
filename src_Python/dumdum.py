#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2


import numpy as np


def draw_8line(img, p1, p2, width):
    """Bresenham's line algorithm"""

    """
    # Check if the points are inside the image dimensions
    if (
        p1[0] < 0
        or p1[0] >= img.shape[1]
        or p1[1] < 0
        or p1[1] >= img.shape[0]
        or p2[0] < 0
        or p2[0] >= img.shape[1]
        or p2[1] < 0
        or p2[1] >= img.shape[0]
    ):
        raise ValueError("Points are outside the image dimensions")
    """

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
                # Calculate the pixel coordinates
                px = x1 + i
                py = y1 + j

                if (
                    px >= 0
                    and px < img.shape[1]
                    and py >= 0
                    and py < img.shape[0]
                ):
                    # Set the pixel at the current position to 255 (white)
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

    return img


def draw_8line_with_arrowhead(img, p1, p2, width=3, arrowhead_size=20):
    # Calculate arrowhead points
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    length = np.sqrt(dx**2 + dy**2)
    dx_unit = dx / length
    dy_unit = dy / length

    arrowhead_p1 = (
        int(p2[0] - arrowhead_size * dx_unit + arrowhead_size * dy_unit),
        int(p2[1] - arrowhead_size * dy_unit - arrowhead_size * dx_unit),
    )
    arrowhead_p2 = (
        int(p2[0] - arrowhead_size * dx_unit - arrowhead_size * dy_unit),
        int(p2[1] - arrowhead_size * dy_unit + arrowhead_size * dx_unit),
    )

    img = draw_8line(img, p1, p2, width)
    img = draw_8line(img, arrowhead_p1, p2, width)
    img = draw_8line(img, arrowhead_p2, p2, width)

    return img


if 1:
    result_image = draw_8line_with_arrowhead(
        np.zeros((400, 400), dtype=np.uint8),
        (100, 100),
        (20, 30),
        10,
    )
else:
    result_image = draw_8line(
        np.zeros((400, 400), dtype=np.uint8),
        (100, 100),
        (20, 30),
        10,
    )

cv2.imshow("output", result_image)
cv2.imwrite("output.png", result_image)
cv2.waitKey(0)
