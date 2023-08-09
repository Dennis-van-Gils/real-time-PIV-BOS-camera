#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "09-08-2023"
__version__ = "1.0.0"
# pylint: disable=invalid-name, missing-function-docstring

import sys
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy import fft as sp_fft
import cupy as cp

from numba import njit, prange
from numba.typed import List

# ------------------------------------------------------------------------------
#   FFTW_Convolver_Full2D
# ------------------------------------------------------------------------------


class FFTW_Convolver_Full2D:
    """Manages a fast-Fourier transform (FFT) convolution on 2D input arrays
    `in1` and `in2` as passed to method `convolve()`, which will return the
    result as a `numpy.ndarray` containing the 'full' convolution elements.

    Args:
        s1 (tuple):
            Shape of the upcoming input array `in1` passed to method
            `convolve()`.

        s2 (tuple):
            Shape of the upcoming input array `in2` passed to method
            `convolve()`.

            Default: 5
    """

    def __init__(self, s_img: tuple, N_IWs: int, s1: tuple, s2: tuple = ()):
        # Example:   s1 = (64, 64), s2 = (64, 64)
        # shape      evaluates to (127, 127)
        # fshape     evaluates to (128, 128)
        # fshape_out evaluates to (128, 65)
        # fslice     evaluates to ((0:127), (0:127))

        if s2 == ():
            s2 = s1

        self.s1 = s1
        self.N_IWs = N_IWs

        axes = (0, 1)
        shape = [
            max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
            for i in range(2)
        ]

        # Speed up FFT by padding to optimal size.
        # self.fshape = [sp_fft.next_fast_len(shape[a]) for a in axes]
        a = [sp_fft.next_fast_len(shape[a]) for a in axes]
        self.fshape = List()
        [self.fshape.append(x) for x in a]

        fshape_out = [self.fshape[0], self.fshape[1] // 2 + 1]

        # Slice corresponding to the 'full' convolution elements to be
        # finally returned as convolution result
        self.fslice = tuple([slice(sz) for sz in shape])

        # fmt: off
        self._rfft_in1  = cp.ndarray(self.fshape, dtype="float")
        self._rfft_in2  = cp.ndarray(self.fshape, dtype="float")
        self._irfft_out = cp.ndarray(self.fshape, dtype="float")
        # fmt: on
        self._rfft_in1[:] = 0
        self._rfft_in2[:] = 0

        self.A_ = cp.ndarray(s_img, dtype="float")
        self.B = cp.ndarray(s_img, dtype="float")

        self.IW_A_ = cp.ndarray(s1, dtype="float")
        self.IW_B = cp.ndarray(s1, dtype="float")
        self.C_maps = cp.ndarray(
            (N_IWs, s1[0] * 2 - 1, s1[1] * 2 - 1), dtype="float"
        )

    def load_frame_A_into_gpu(self, A_: npt.NDArray[np.float32]):
        self.A_ = cp.asarray(A_)

    def load_frame_B_into_gpu(self, B: npt.NDArray[np.float32]):
        self.B = cp.asarray(B)

    def construct_IW_A_B(
        self,
        Ax0,
        Ax1,
        Ay0,
        Ay1,
        Bx0,
        Bx1,
        By0,
        By1,
        zero_out_L,
        zero_out_R,
        zero_out_U,
        zero_out_D,
    ):
        # fmt: off
        #cp.copyto(self.IW_A_, self.A_[Ay0:Ay1, Ax0:Ax1])
        #cp.copyto(self.IW_B , self.B [By0:By1, Bx0:Bx1])
        self.IW_A_ = self.A_[Ay0:Ay1, Ax0:Ax1]
        self.IW_B  = self.B [By0:By1, Bx0:Bx1]

        if zero_out_L > 0:
            self.IW_A_[:, :zero_out_L] = 0
            self.IW_B [:, :zero_out_L] = 0
        if zero_out_R > 0:
            self.IW_A_[:, -zero_out_R:] = 0
            self.IW_B [:, -zero_out_R:] = 0
        if zero_out_U > 0:
            self.IW_A_[:zero_out_U, :] = 0
            self.IW_B [:zero_out_U, :] = 0
        if zero_out_D > 0:
            self.IW_A_[-zero_out_D:, :] = 0
            self.IW_B [-zero_out_D:, :] = 0
        # fmt: on

    # --------------------------------------------------------------------------
    #   convolve
    # --------------------------------------------------------------------------

    def convolve(self, IW_idx: int):
        """Performs the FFT convolution over /all/ IWs"""

        if cp.max(self.IW_A_) <= 0:
            # No details are present in the IW images. All pixels are below
            # or at the mean background --> Save computation time.
            # TODO: Make this a user config threshold? Is <= 0 even correct?
            # Must match up with 'zeroing out' mechanism of function
            # `obtain_IW_limits()`.
            self.C_maps[IW_idx, 0, 0] = np.nan

        else:
            # Perform 2D cross-correlation

            # Zero padding and forwards Fourier transformation
            self._rfft_in1[: self.s1[0], : self.s1[1]] = self.IW_A_
            self._rfft_in2[: self.s1[0], : self.s1[1]] = self.IW_B
            sp1 = cp.fft.rfftn(
                self._rfft_in1, self.fshape, axes=(0, 1), norm="forward"
            )
            sp2 = cp.fft.rfftn(
                self._rfft_in2, self.fshape, axes=(0, 1), norm="forward"
            )

            # Convolution and backwards Fourier transformation
            m = cp.multiply(sp1, sp2)
            result = cp.fft.irfftn(m, self.fshape, axes=(0, 1), norm="backward")

            # Return the 'full' elements
            # return cp.asnumpy(result[self.fslice])
            self.C_maps[IW_idx, :, :] = result[self.fslice]

    def processed_C_maps(self) -> npt.NDArray[np.float32]:
        return cp.asnumpy(self.C_maps)
