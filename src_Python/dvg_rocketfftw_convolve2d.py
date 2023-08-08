#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try:
https://github.com/styfenschaer/rocket-fft
https://github.com/styfenschaer/rocket-fft/blob/main/tests/test_scipy_testsuite.py
https://numba.discourse.group/t/rocket-fft-a-numba-extension-supporting-numpy-fft-and-scipy-fft/1657/5
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "08-08-2023"
__version__ = "1.0.0"
# pylint: disable=invalid-name, missing-function-docstring

import sys
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy import fft as sp_fft

from numba import njit, prange
from numba.typed import List

from rocket_fft import r2c, c2r
from rocket_fft import numpy_like, scipy_like

# numpy_like()
scipy_like()

# ------------------------------------------------------------------------------
#   fast_multiply
# ------------------------------------------------------------------------------


@njit(
    "(complex64[:, ::1], complex64[:, ::1], complex64[:, ::1])",
    nogil=True,
    cache=True,
)
def fast_multiply(
    in1: npt.NDArray[np.complex64],
    in2: npt.NDArray[np.complex64],
    out: npt.NDArray[np.complex64],
):
    """
    * In-place operation on `out`.
    * Faster version of `out = np.multiply(in1, in2)`.
    * Not parallelized.
    """
    for i in range(in1.shape[0]):
        for j in range(in1.shape[1]):
            out[i, j] = in1[i, j] * in2[i, j]


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

        fftw_threads (int, optional):
            Number of threads to use for the FFT transformations. When set to
            > 1, the Python GIL will not be invoked.

            Default: 5
    """

    def __init__(self, s1: tuple, s2: tuple = (), fftw_threads: int = 5):
        # Example:   s1 = (64, 64), s2 = (64, 64)
        # shape      evaluates to (127, 127)
        # fshape     evaluates to (128, 128)
        # fshape_out evaluates to (128, 65)
        # fslice     evaluates to ((0:127), (0:127))

        if s2 == ():
            s2 = s1

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
        self._rfft_in1  = np.zeros(self.fshape, dtype="float32")
        self._rfft_in2  = np.zeros(self.fshape, dtype="float32")
        self._rfft_out1 = np.zeros(fshape_out , dtype="complex64")
        self._rfft_out2 = np.zeros(fshape_out , dtype="complex64")
        self._irfft_in  = np.zeros(fshape_out , dtype="complex64")
        self._irfft_out = np.zeros(self.fshape, dtype="float32")
        # fmt: on

        """
        # Create the FFTW plans
        # fmt: off
        self._rfft_in1  = pyfftw.zeros_aligned(self.fshape, dtype="float32")
        self._rfft_in2  = pyfftw.zeros_aligned(self.fshape, dtype="float32")
        self._rfft_out1 = pyfftw.empty_aligned(fshape_out , dtype="complex64")
        self._rfft_out2 = pyfftw.empty_aligned(fshape_out , dtype="complex64")
        self._irfft_in  = pyfftw.empty_aligned(fshape_out , dtype="complex64")
        self._irfft_out = pyfftw.empty_aligned(self.fshape, dtype="float32")
        # fmt: on

        print("Creating FFTW plans for convolution...", end="")
        sys.stdout.flush()

        p = {
            "axes": (0, 1),
            "flags": ("FFTW_MEASURE", "FFTW_DESTROY_INPUT"),
            "threads": fftw_threads,
        }
        self._fftw_rfft1 = pyfftw.FFTW(self._rfft_in1, self._rfft_out1, **p)
        self._fftw_rfft2 = pyfftw.FFTW(self._rfft_in2, self._rfft_out2, **p)
        self._fftw_irfft = pyfftw.FFTW(
            self._irfft_in,
            self._irfft_out,
            direction="FFTW_BACKWARD",
            **p,
        )

        print(" done.")
        """

    # --------------------------------------------------------------------------
    #   convolve
    # --------------------------------------------------------------------------

    def convolve(
        self, in1: npt.NDArray[np.float32], in2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Performs the FFT convolution on input arrays `in1` and `in2` and
        returns the result as a `numpy.ndarray` containing the 'full'
        convolution elements.

        Returns:
            The full convolution results as a 2D numpy array with a shape
            equal to `in1 + in2 - 1`.

        NOTE: `in1` and `in2` do not necessarily have to be C-contiguous,
        because they will, internal to this method, get copied into C-contiguous
        arrays during the zero-padding operation.
        NOTE: The output matrix is not contiguous.
        """

        # Zero padding and forwards Fourier transformation
        self._rfft_in1[: in1.shape[0], : in1.shape[1]] = in1
        self._rfft_in2[: in2.shape[0], : in2.shape[1]] = in2

        # High-level
        """
        sp1 = rfftn(self._rfft_in1, self.fshape, axes=(0, 1))
        sp2 = rfftn(self._rfft_in2, self.fshape, axes=(0, 1))

        # Convolution and backwards Fourier transformation
        fast_multiply(sp1, sp2, self._irfft_in)
        result = irfftn(self._irfft_in, self.fshape, axes=(0, 1))
        """

        # Low-level rocket-fft
        # """
        rfftn(self._rfft_in1, self._rfft_out1, axes=(0, 1))
        rfftn(self._rfft_in2, self._rfft_out2, axes=(0, 1))

        # Convolution and backwards Fourier transformation
        fast_multiply(self._rfft_out1, self._rfft_out2, self._irfft_in)
        irfftn(self._irfft_in, self._irfft_out, axes=(0, 1), forward=False)
        result = self._irfft_out
        # """

        # Return the 'full' elements
        return result[self.fslice]


fft_njit = partial(
    njit,
    cache=True,
    nogil=True,
)


# @fft_njit
# def rfftn(x, s=None, axes=(0, 1), norm=None, overwrite_x=False, workers=None):
#     return sp_fft.rfftn(x, s, axes, norm, overwrite_x, workers)


# @fft_njit
# def irfftn(x, s=None, axes=(0, 1), norm=None, overwrite_x=False, workers=None):
#     return sp_fft.irfftn(x, s, axes, norm, overwrite_x, workers)


# @fft_njit
# def rfftn(a, s=None, axes=(0, 1), norm=None):
#     return np.fft.rfftn(a, s, axes, norm)


# @fft_njit
# def irfftn(a, s=None, axes=(0, 1), norm=None):
#     return np.fft.irfftn(a, s, axes, norm)


# @fft_njit
def rfftn(
    ain,
    aout,
    axes=(0, 1),
    forward=True,
    fct=np.float64(1.0),
    nthreads=np.int64(1),
):
    r2c(ain, aout, axes, forward, fct, nthreads)


# @fft_njit
def irfftn(
    ain,
    aout,
    axes=(0, 1),
    forward=True,
    fct=np.float64(1.0),
    nthreads=np.int64(1),
):
    c2r(ain, aout, axes, forward, fct, nthreads)
