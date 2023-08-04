#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performs lightning-fast convolutions on 2D input arrays.

The convolution is based on the fast-Fourier transform (FFT) as performed by the
excellent `fftw` (http://www.fftw.org) library. It will plan the transformations
ahead of time to optimize the calculations. Also, multiple threads can be
specified for the FFT and, when set to > 1, the Python GIL will not be invoked.
This results in true multithreading across multiple cores, which can result in a
huge performance gain.

This 'faster' version does not zero-append the input arrays to the next fast
length. It is hence faster, because the matrices it operates on are smaller, but
the output is less accurate due to edge effects when convoluting.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "04-08-2023"
__version__ = "1.0.0"
# pylint: disable=invalid-name, missing-function-docstring

import sys
import numpy as np
import pyfftw
from numba import njit


@njit(
    "complex128[:, :](complex128[:, :], complex128[:, :])",
    nogil=True,
    cache=True,
)
def fast_multiply(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    return np.multiply(in1, in2)


class FFTW_Convolver_Full2D:
    """Manages a fast-Fourier transform (FFT) convolution on 2D input arrays
    `in1` and `in2` as passed to method `convolve()`, which will return the
    result as a contiguous C-style `numpy.ndarray` containing the 'full'
    convolution elements.

    Here, we demand both `in1` and `in2` to be of equal size for calculation
    speed improvements.

    Args:
        s (tuple):
            Shape of the upcoming input arrays `in1` and `in2` passed to method
            `convolve()`.

        fftw_threads (int, optional):
            Number of threads to use for the FFT transformations. When set to
            > 1, the Python GIL will not be invoked.

            Default: 5
    """

    def __init__(self, s: tuple, fftw_threads: int = 5):
        self.s = s

        s_out = (s[0], s[1] // 2 + 1)
        # s_final = (s[0] * 2 - 1, s[1] * 2 - 1)
        # s_final = s_out

        # Create the FFTW plans
        # fmt: off
        self._rfft_in1  = pyfftw.zeros_aligned(s    , dtype="float64")
        self._rfft_in2  = pyfftw.zeros_aligned(s    , dtype="float64")

        self._rfft_out1 = pyfftw.empty_aligned(s_out, dtype="complex128")
        self._rfft_out2 = pyfftw.empty_aligned(s_out, dtype="complex128")

        self._irfft_in  = pyfftw.empty_aligned(s_out, dtype="complex128")
        self._irfft_out = pyfftw.empty_aligned(s    , dtype="float64")
        # fmt: on

        print("Creating FFTW plans for convolution...", end="")
        sys.stdout.flush()

        p = {
            "axes": (0, 1),
            "flags": ("FFTW_MEASURE",),
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

    # --------------------------------------------------------------------------
    #   convolve
    # --------------------------------------------------------------------------

    def convolve(self, in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
        """Performs the FFT convolution on input arrays `in1` and `in2` and
        returns the result as a contiguous C-style `numpy.ndarray` containing
        the 'full' convolution elements.

        When the shapes of the passed input arrays are not compatible with the
        convolution operation, an array full of `np.nan`s is returned.

        Returns:
            The full convolution results as a 2D numpy array.
        """
        # Force contiguous C-style numpy arrays, super fast when already so
        in1 = np.asarray(in1)
        in2 = np.asarray(in2)

        # Perform FFT convolution
        # -----------------------
        # Forwards Fourier transformation
        self._rfft_in1[:] = in1
        self._rfft_in2[:] = in2
        self._fftw_rfft1()
        self._fftw_rfft2()

        # Convolution and backwards Fourier transformation
        self._irfft_in[:] = fast_multiply(self._rfft_out1, self._rfft_out2)
        result = self._fftw_irfft()

        # Return the 'full' elements
        return fast_fftshift(result)


@njit(
    "float64[:, :](float64[:, :])",
    nogil=True,
    cache=True,
)
def fast_fftshift(C):
    """Like `numpy.fft.fftshift(), but faster."""
    rows, cols = C.shape
    half_rows = rows // 2
    half_cols = cols // 2

    shifted = np.empty_like(C)

    # Swap quadrants
    shifted[:half_rows, :half_cols] = C[half_rows:, half_cols:]
    shifted[:half_rows, half_cols:] = C[half_rows:, :half_cols]
    shifted[half_rows:, :half_cols] = C[:half_rows, half_cols:]
    shifted[half_rows:, half_cols:] = C[:half_rows, :half_cols]

    return shifted


if __name__ == "__main__":
    # TRY CuPy:
    # https://www.appsloveworld.com/numpy/100/83/how-to-do-100000-times-2d-fft-in-a-faster-way-using-python

    from scipy.signal import windows
    import matplotlib.pyplot as plt
    import pyfftw

    def gaussian_kernel(n, std, normalised=False):
        """
        Generates a n x n matrix with a centered gaussian
        of standard deviation std centered on it. If normalised,
        its volume equals 1."""
        gaussian1D = windows.gaussian(n, std)
        gaussian2D = np.outer(gaussian1D, gaussian1D)
        if normalised:
            gaussian2D /= 2 * np.pi * (std**2)
        return gaussian2D

    size_A = 64
    A = gaussian_kernel(size_A, size_A / 8)
    B = gaussian_kernel(size_A, size_A / 2)

    fftw_1 = FFTW_Convolver_Full2D(A.shape, fftw_threads=1)

    C = fftw_1.convolve(A, B)

    plt.figure(1)
    plt.imshow(A, cmap="gray")
    plt.figure(2)
    plt.imshow(B, cmap="gray")
    plt.figure(3)
    plt.imshow(C, cmap="gray")

    plt.show()
