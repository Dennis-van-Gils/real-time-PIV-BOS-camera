#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image processing algorithm for 2D Particle Imaging Velocimetry (PIV) and
Background Oriented Schlieren (BOS)
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "07-07-2023"
__version__ = "1.0"
# pylint: disable=bare-except, broad-except, missing-function-docstring, wrong-import-position

import os
import sys
import glob

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import numpy as np

from skimage.io import imread

from my_fun import IW_Grid

import_file = "E:/Work/_GitHub_repo/2D-PIV-BOS/test_imgs/PIV_rising_vortex_plume/B00001.tif"

# ------------------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------------------

# Used abbrevations
# -----------------
# IW: Interrogation window
# VM: Displacement vector map

if __name__ == "__main__":
    # Read double image and split into frames A & B
    img = imread(import_file, as_gray=True)
    img_2h, img_w = np.shape(img)
    img_h = int(img_2h / 2)
    A = img[:img_h, :]
    B = img[img_h:, :]
    print(A.shape)
    print(B.shape)

    # Mean background removal
    A = A - np.mean(A)
    B = B - np.mean(B)

    fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(A, cmap="gray", interpolation="none")

    # fig.add_subplot(1, 2, 2)
    # plt.imshow(
    #     B,
    #     cmap="gray",
    #     interpolation="none",
    # )

    # plt.show()

    # ----------------------------------------------------------------------
    #   Initialize
    # ----------------------------------------------------------------------

    # Set the IW sizes for multigrid analysis
    # Subsequent IW sizes should be the exact half of the prev IW size
    # IW_SIZES   = [128 96 64 48 32];
    IW_SIZES = [64, 32]
    IW_OVERLAP = 0.5

    # Allocate multigrid maps
    nIW_SIZES = len(IW_SIZES)
    IW_grid_As: list[IW_Grid] = []
    IW_grid_Bs: list[IW_Grid] = []

    # ---------------------------------------------------------------------
    #   Walk over all interrogation window sizes
    # ----------------------------------------------------------------------

    for iIW_size in range(nIW_SIZES):
        IW_size = IW_SIZES[iIW_size]

        # Create IW_grid for frame A
        # Create IW_grid for frame B
        IW_grid_A = IW_Grid(img_w, img_h, IW_size, IW_OVERLAP)
        IW_grid_B = IW_grid_A

        if 0:
            # DEBUG: Examine IW_grid

            # fig.clear()
            plt.plot(
                IW_grid_A.x_range[:, 0],
                IW_grid_A.y_range[:, 0],
                "xg",
                linewidth=2,
                label="IW starts",
            )

            plt.plot(
                IW_grid_A.x_range[:, 1],
                IW_grid_A.y_range[:, 1],
                "xr",
                linewidth=2,
                label="IW endings",
            )

            # Plot IW centers, but not all. Just the bottom and left chords.
            plt.plot(
                IW_grid_A.x[:, 0], IW_grid_A.y[:, 0], "ok", label="IW centers"
            )
            plt.plot(
                IW_grid_A.x[0, :], IW_grid_A.y[0, :], "ok", label="_nolegend_"
            )

            # Plot image bounding box
            plt.gca().add_patch(
                Rectangle(
                    (0, 0), img_w - 1, img_h - 1, edgecolor="k", fill=None, lw=1
                )
            )
            # plt.plot([0, 0], [0, img_h - 1], "-k", label="_nolegend_")
            # plt.plot([0, img_w - 1], [0, 0], "-k", label="_nolegend_")
            # plt.plot(
            #     [img_w - 1, img_w - 1], [0, img_h - 1], "-k", label="_nolegend_"
            # )
            # plt.plot(
            #     [0, img_w - 1], [img_h - 1, img_h - 1], "-k", label="_nolegend_"
            # )
            plt.legend()
            plt.show()

        # Already store IW_grid_A in the multigrid map for the first IW size
        if iIW_size == 1:
            IW_grid_As.append(IW_grid_A)

    print("The end")
