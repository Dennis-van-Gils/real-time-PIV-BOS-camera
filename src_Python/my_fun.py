#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "10-07-2023"
__version__ = "1.0"

import numpy as np
from numba import njit

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


class IW_Grid:
    """
    Args:
        img_w (``int``):
            Width of source image [px].

        img_h (``int``):
            Height of source image [px].

        IW_size (``int``):
            Interrogation window size [px].

        overlap (``float``, optional):
            Window overlap fraction [0 - 1].
            0  : no window overlap
            0.5: 50% window overlap

            Default: 0.5

    Attributes:
        nIWs_x (``int``):
            Number of interrogation windows along the x direction.

        nIWs_y (``int``):
            Number of interrogation windows along the y direction.

        nIWs (``int``):
            Total number of interrogation windows.

        x (``np.ndarray(int)``):
            Meshgrid (TODO: mention size of array) of the x-positions of the IW
            centers [px].

        y (``np.ndarray(int)``):
            Meshgrid (TODO: mention size of array) of the y-positions of the IW
            centers [px].

        x_range (``np.ndarray(int)``):
            Array (TODO: mention size of array) containing [min max] x-pos per
            IW [px].
            3-D array (iIWs_y, iIWs_x, 2):
              (:, :, 0): starting position
              (:, :, 1): ending  position

        y_range (``np.ndarray(int)``):
            Array (TODO: mention size of array) containing [min max] y-pos per
            IW [px].
            3-D array (iIWs_y, iIWs_x, 2):
              (:, :, 0): starting position
              (:, :, 1): ending  position
    """

    def __init__(self, img_w, img_h, IW_size, overlap):
        self.IW_size = IW_size
        self.overlap = overlap

        # Calculate number of IWs that will fit in the source image
        self.nIWs_x = int((img_w - IW_size) // (IW_size * (1 - overlap))) + 1
        self.nIWs_y = int((img_h - IW_size) // (IW_size * (1 - overlap))) + 1
        self.nIWs = self.nIWs_x * self.nIWs_y

        # Calculate IW positions
        # MATLAB equivalent:
        #   arr_x = np.round((0:nIWs_x - 1) * (1 - overlap) * IW_size + np.floor(IW_size/2) + 1)
        #   arr_y = np.round((0:nIWs_y - 1) * (1 - overlap) * IW_size + np.floor(IW_size/2) + 1)
        #   [IW_grid.x, IW_grid.y] = meshgrid(arr_x, arr_y);
        half_IW_size = IW_size // 2
        arr_x = np.arange(self.nIWs_x) * (1 - overlap) * IW_size + half_IW_size
        arr_y = np.arange(self.nIWs_y) * (1 - overlap) * IW_size + half_IW_size
        arr_x = np.asarray(arr_x, dtype=int)
        arr_y = np.asarray(arr_y, dtype=int)
        self.x, self.y = np.meshgrid(arr_x, arr_y)

        # Calculate IW pixel ranges
        x_range = np.column_stack(
            (arr_x - half_IW_size, arr_x + half_IW_size - 1)
        )
        y_range = np.column_stack(
            (arr_y - half_IW_size, arr_y + half_IW_size - 1)
        )

        self.x_range = np.tile(x_range, (self.nIWs_y, 1, 1))
        self.y_range = np.tile(y_range, (self.nIWs_x, 1, 1)).swapaxes(0, 1)

        if 0:  # DEBUG flag: Examine IW_grid
            plt.figure()
            plt.plot(
                self.x_range[:, :, 0].reshape(-1),
                self.y_range[:, :, 0].reshape(-1),
                "xg",
                linewidth=2,
                label="IW starts",
            )
            plt.plot(
                self.x_range[:, :, 1].reshape(-1),
                self.y_range[:, :, 1].reshape(-1),
                "xr",
                linewidth=2,
                label="IW endings",
            )

            # Plot IW centers, but not all. Just the bottom and left chords.
            plt.plot(self.x[:, 0], self.y[:, 0], "ok", label="IW centers")
            plt.plot(self.x[0, :], self.y[0, :], "ok", label="_nolegend_")

            # Plot image bounding box
            plt.gca().add_patch(
                Rectangle(
                    (0, 0),
                    img_w - 1,
                    img_h - 1,
                    edgecolor="k",
                    fill=None,
                    lw=1,
                )
            )
            plt.legend()
            plt.show()


# @njit() DOES NOT WORK WITH CLASSES, NOR AS CLASS METHOD
def lookup_IW_Idx(IW_grid: IW_Grid, x_pixel: int, y_pixel: int):
    """Lookup the index of the IW that has its center closest to the input
    location [x_pixel, y_pixel].

    Returns:
        iIW_x: x-index of the IW.
        iIW_y: y-index of the IW.
    """

    # MATLAB equivalent:
    #  iIW_x = floor((x_pixel - floor(IW_grid.IW_size/2) - 1) / ...
    #              (IW_grid.IW_size*(1 - IW_grid.overlap)) + 1.5);
    #  iIW_y = floor((y_pixel - floor(IW_grid.IW_size/2) - 1) / ...
    #              (IW_grid.IW_size*(1 - IW_grid.overlap)) + 1.5);
    #  iIW_x = min(iIW_x, IW_grid.nIWs_x);
    #  iIW_y = min(iIW_y, IW_grid.nIWs_y);
    #  iIW   = sub2ind([IW_grid.nIWs_y IW_grid.nIWs_x], iIW_y, iIW_x);

    half_IW_size = IW_grid.IW_size // 2
    iIW_x = (
        (x_pixel - half_IW_size) / (IW_grid.IW_size * (1 - IW_grid.overlap))
        + 0.5
    ).astype(int)
    iIW_y = (
        (y_pixel - half_IW_size) / (IW_grid.IW_size * (1 - IW_grid.overlap))
        + 0.5
    ).astype(int)
    iIW_x = np.minimum(iIW_x, IW_grid.nIWs_x - 1)
    iIW_y = np.minimum(iIW_y, IW_grid.nIWs_y - 1)

    return iIW_x, iIW_y


@njit("(float64[:, :], uint16, uint16)", cache=True, nogil=True)
def subpx_3pgf_2D(C: np.ndarray, px: int, py: int):
    """Perform a 3-point Gaussian fit to the point with index (py, px)
    inside of 2-D matrix 'C' along both the x and y direction.
    """

    # Along x
    if px > 0 and px < C.shape[1] - 1:
        # Fit possible
        phi_m1 = np.maximum(C[py, px - 1], 1e-40)  # Prevent taking log of zero
        phi_p1 = np.maximum(C[py, px + 1], 1e-40)  # Prevent taking log of zero
        px_sub = (
            px
            + (np.log(phi_m1) - np.log(phi_p1))
            / (np.log(phi_m1) + np.log(phi_p1) - 2 * np.log(C[py, px]))
            / 2
        )
    else:
        # No fit possible
        px_sub = px

    # Along y
    if py > 0 and py < C.shape[0] - 1:
        # Fit possible
        phi_m1 = np.maximum(C[py - 1, px], 1e-40)  # Prevent taking log of zero
        phi_p1 = np.maximum(C[py + 1, px], 1e-40)  # Prevent taking log of zero
        py_sub = (
            py
            + (np.log(phi_m1) - np.log(phi_p1))
            / (np.log(phi_m1) + np.log(phi_p1) - 2 * np.log(C[py, px]))
            / 2
        )
    else:
        # No fit possible
        py_sub = py

    return px_sub, py_sub
