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
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

# mpl.rcParams["toolbar"] = "None"

import numpy as np
from scipy.signal import fftconvolve
import numba

from skimage.io import imread

from my_fun import IW_Grid, lookup_IW_Idx, subpx_3pgf_2D

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
    t_0 = perf_counter()

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

        for iIW in range(18, IW_grid_A.nIWs):
            # print(f"iIW = {iIW}")

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

            C = fftconvolve(img_IW_B, img_IW_A)
            C = C / np.max(C)

            if 0:  # DEBUG flag: Show correlation map
                if iIW == 0:
                    plt.clf()
                    imshow_obj = plt.imshow(
                        C,
                        cmap="gray",
                        interpolation="none",
                        vmin=0,
                        vmax=1,
                    )
                else:
                    # imshow_obj.set_data(C)
                    plt.clf()
                    imshow_obj = plt.imshow(
                        C,
                        cmap="gray",
                        interpolation="none",
                        vmin=0,
                        vmax=1,
                    )
                plt.title(f"{iIW} of {IW_grid_A.nIWs}")
                plt.draw()
                plt.pause(0.0001)

            # Find maximum correlation peak

            # MATLAB equivalent:
            #   if isnan(max(C(:)))
            #     dx = nan; dy = nan;
            #   else
            #     [maxC, iMaxC] = max(C(:));
            #     [peak_y, peak_x] = ind2sub(size(C), iMaxC);

            #     % Sub-pixel resolution algorithm, 3-point Gaussian fit
            #     [peak_x, peak_y] = subpx_3pgf_2D(C, peak_x, peak_y);

            #     % Calculate displacement vector
            #     dx = peak_x - floor(size(C, 2)/2 + 1) + shift_x;
            #     dy = peak_y - floor(size(C, 1)/2 + 1) + shift_y;
            #   end

            iMaxC = np.argmax(C)
            peak_y, peak_x = np.unravel_index(iMaxC, C.shape, order="C")
            peak_x = int(peak_x)
            peak_y = int(peak_y)

            # Sub-pixel resolution algorithm, 3-point Gaussian fit
            peak_x, peak_y = subpx_3pgf_2D(C, peak_x, peak_y)

            if 0:  # DEBUG flag: Show correlation peak
                plt.plot(peak_x, peak_y, "xr")
                plt.plot(peak_x, peak_y, "og")
                plt.draw()
                # plt.pause(2)
                plt.show()

            # Calculate displacement vector
            dx = peak_x - C.shape[1] // 2 + 1  # + shift_x
            dy = peak_y - C.shape[0] // 2 + 1  # + shift_y

    duration = perf_counter() - t_0
    print(f"Finished in {duration:.3f} s")
    # scipy.signal.fftconvolve takes ~1.24 s in alacritty without printing
