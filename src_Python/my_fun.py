#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "07-07-2023"
__version__ = "1.0"

import numpy as np

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

        y_range (``np.ndarray(int)``):
            Array (TODO: mention size of array) containing [min max] y-pos per
            IW [px].
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

        # Calculate IW ranges
        # MATLAB equivalent:
        #   x_range = [arr_x - floor(IW_size/2); array_x + floor(IW_size/2) - 1]';
        #   y_range = [arr_y - floor(IW_size/2); array_y + floor(IW_size/2) - 1]';
        #   IW_grid.x_range = sortrows(repmat(x_range, nIWs_y, 1));
        #   IW_grid.y_range = repmat(y_range, nIWs_x, 1);
        x_range = np.column_stack(
            (arr_x - half_IW_size, arr_x + half_IW_size - 1)
        )
        y_range = np.column_stack(
            (arr_y - half_IW_size, arr_y + half_IW_size - 1)
        )
        self.x_range = np.tile(x_range, (self.nIWs_y, 1))
        self.y_range = np.repeat(y_range, self.nIWs_x, axis=0)

        if 0:  # DEBUG flag: Examine IW_grid
            plt.figure()
            plt.plot(
                self.x_range[:, 0],
                self.y_range[:, 0],
                "xg",
                linewidth=2,
                label="IW starts",
            )
            plt.plot(
                self.x_range[:, 1],
                self.y_range[:, 1],
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
