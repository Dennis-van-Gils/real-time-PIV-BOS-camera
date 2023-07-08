#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image processing algorithm for 2D Particle Imaging Velocimetry (PIV) and
Background Oriented Schlieren (BOS)

Used abbrevations
-----------------
IW: Interrogation window
VM: Displacement vector map
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "07-07-2023"
__version__ = "1.0"
# pylint: disable=missing-function-docstring

import os
import sys
import glob

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import numpy as np
from scipy.signal import correlate2d, convolve2d, fftconvolve
import numba

from skimage.io import imread

from my_fun import IW_Grid, lookup_IW_Idx

import_file = "E:/Work/_GitHub_repo/2D-PIV-BOS/test_imgs/PIV_rising_vortex_plume/B00001.tif"

# Set the IW sizes for multigrid analysis
# Subsequent IW sizes should be the exact half of the prev IW size
# IW_SIZES   = [128 96 64 48 32];
IW_SIZES = [64, 32]
IW_OVERLAP = 0.5

# ------------------------------------------------------------------------------
#   Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Read double image and split into frames A & B
    img = imread(import_file, as_gray=True)
    img_2h, img_w = np.shape(img)
    img_h = int(img_2h / 2)
    A = img[:img_h, :]
    B = img[img_h:, :]
    print(f"Img A: {A.shape}")
    print(f"Img B: {B.shape}")

    # Mean background removal
    A = A - np.mean(A)
    B = B - np.mean(B)

    if 0:  # DEBUG flag: Show image A
        fig = plt.figure()
        plt.imshow(A, cmap="gray", interpolation="none")
        plt.show()

    # --------------------------------------------------------------------------
    #   Initialize
    # --------------------------------------------------------------------------

    # Allocate multigrid maps
    nIW_SIZES = len(IW_SIZES)
    IW_grid_As: list[IW_Grid] = []
    IW_grid_Bs: list[IW_Grid] = []

    # --------------------------------------------------------------------------
    #   Walk over all interrogation window sizes
    # --------------------------------------------------------------------------

    for iIW_size in range(nIW_SIZES):
        IW_size = IW_SIZES[iIW_size]

        # Create IW_grid for frame A
        # Create IW_grid for frame B
        IW_grid_A = IW_Grid(img_w, img_h, IW_size, IW_OVERLAP)
        IW_grid_B = IW_grid_A

        # Already store IW_grid_A in the multigrid map for the first IW size
        if iIW_size == 1:
            IW_grid_As.append(IW_grid_A)

        # ----------------------------------------------------------------------
        #   Walk over all IWs
        # ----------------------------------------------------------------------

        for iIW in range(IW_grid_A.nIWs):
            print(f"iIW = {iIW}")

            # ------------------------------------------------------------------
            #   Calculate IW of frame B
            #   Apply window shifting technique
            # ------------------------------------------------------------------

            if iIW_size == 0:
                # First IW size, no pre-shift available
                shift_x = 0  # [px]
                shift_y = 0  # [px]
            else:
                # Pre-shift available
                # Calculate corresponding index of the IW in the larger parent
                # grid
                iIW_parent = lookup_IW_Idx(
                    IW_grid_As[iIW_size - 1],
                    IW_grid_A.x_1D[iIW],
                    IW_grid_A.y_1D[iIW],
                )

                """
                # MATLAB equivalent:
                % Retrieve the pre-shift
                shift_x = round(VMs{iIW_size - 1}.dx(iIW_parent));    % [px]
                shift_y = round(VMs{iIW_size - 1}.dy(iIW_parent));    % [px]
                if isnan(shift_x); shift_x = 0; end
                if isnan(shift_y); shift_y = 0; end

                % Calculate new center and range of the shifted IW in frame B
                IW_grid_B.x(iIW)          = IW_grid_B.x(iIW)          + shift_x;
                IW_grid_B.y(iIW)          = IW_grid_B.y(iIW)          + shift_y;
                IW_grid_B.x_range(iIW, :) = IW_grid_B.x_range(iIW, :) + shift_x;
                IW_grid_B.y_range(iIW, :) = IW_grid_B.y_range(iIW, :) + shift_y;

                % The IW should never be shifted outside of frame B.
                % When it does, equally resize the IWs of both frames A and B such
                % that the resized IW of frame B still fits in frame B
                if IW_grid_B.x_range(iIW, 1) < 1
                IW_grid_B.x_range(iIW, 1) = 1;
                IW_grid_A.x_range(iIW, 1) = 1 - shift_x;
                end
                if IW_grid_B.y_range(iIW, 1) < 1
                IW_grid_B.y_range(iIW, 1) = 1;
                IW_grid_A.y_range(iIW, 1) = 1 - shift_y;
                end
                if IW_grid_B.x_range(iIW, 2) > img_w
                IW_grid_B.x_range(iIW, 2) = img_w;
                IW_grid_A.x_range(iIW, 2) = img_w - shift_x;
                end
                if IW_grid_B.y_range(iIW, 2) > img_h
                IW_grid_B.y_range(iIW, 2) = img_h;
                IW_grid_A.y_range(iIW, 2) = img_h - shift_y;
                """

            # ------------------------------------------------------------------
            #   Retrieve images of IW frame A and IW frame B
            # ------------------------------------------------------------------

            img_IW_A = A[
                IW_grid_A.y_range[iIW, 0] : IW_grid_A.y_range[iIW, 1],
                IW_grid_A.x_range[iIW, 0] : IW_grid_A.x_range[iIW, 1],
            ]

            img_IW_B = B[
                IW_grid_B.y_range[iIW, 0] : IW_grid_B.y_range[iIW, 1],
                IW_grid_B.x_range[iIW, 0] : IW_grid_B.x_range[iIW, 1],
            ]

            # ------------------------------------------------------------------
            #   Perform cross-correlation
            # ------------------------------------------------------------------

            # MATLAB equivalent:
            #  if isempty(img_IW_A(:)) || ...
            #      max(img_IW_A(:)) == 0 || max(img_IW_B(:)) == 0
            #      C = nan;                        % Save computation time
            #  else
            #      %C = xcorr2(double(img_IW_B), ...
            #      %           double(img_IW_A));   % Slow but accurate
            #      C = xcorr2(single(img_IW_B), ...
            #              single(img_IW_A));   % Fast but slightly less accurate
            #      C = C/max(C(:));                % Normalize
            #  end

            # C = convolve2d(img_IW_B, img_IW_A)
            # C = correlate2d(img_IW_B, img_IW_A)
            C = fftconvolve(img_IW_B, img_IW_A)
            C = C / np.max(C)

    print("The end")
