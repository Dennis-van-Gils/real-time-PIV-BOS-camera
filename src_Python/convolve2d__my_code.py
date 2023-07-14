#!/usr/bin/env python3
# -*- coding: utf-8 -

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

    When the shapes of the passed input arrays are not compatible with the
    convolution operation, an array full of `numpy.nan`s is returned.

    Args:
        shape1 (tuple):
            Shape of the upcoming input array `in1` passed to method
            `convolve()`.

        shape2 (tuple):
            Shape of the upcoming input array `in2` passed to method
            `convolve()`.

        fftw_threads (int, optional):
            Number of threads to use for the FFT transformations. When set to
            > 1, the Python GIL will not be invoked.

            Default: 5
    """

    def __init__(self, shape1: tuple, shape2: tuple, fftw_threads: int = 5):
        self.shape1 = shape1
        self.shape2 = shape2

        shape1_in = shape1
        shape1_out = (shape1[0], shape1[1] // 2 + 1)
        shape2_in = shape2
        shape2_out = (shape2[0], shape2[1] // 2 + 1)

        # Create the FFTW plans
        # fmt: off
        self._rfft_in1  = pyfftw.empty_aligned(shape1_in , dtype="float64")
        self._rfft_in2  = pyfftw.empty_aligned(shape2_in , dtype="float64")
        self._rfft_out1 = pyfftw.empty_aligned(shape1_out, dtype="complex128")
        self._rfft_out2 = pyfftw.empty_aligned(shape2_out, dtype="complex128")
        self._irfft_in  = pyfftw.empty_aligned(shape1_out, dtype="complex128")
        self._irfft_out = pyfftw.empty_aligned(shape1_in, dtype="float64")
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

        """
        # Return np.nans when the input arrays are not fully populated yet
        if in1.shape != self.shape1 or in2.shape != self.shape2:
            return np.full(self.valid_shape, np.nan)
        """

        """
        # Check that input shapes are compatible with 'valid' mode
        if self.switch_inputs:
            in1, in2 = in2, in1
        """

        # Perform FFT convolution
        # -----------------------
        # Zero padding and forwards Fourier transformation
        self._rfft_in1[:] = in1
        self._rfft_in2[:] = in2
        self._fftw_rfft1()
        self._fftw_rfft2()

        # Convolution and backwards Fourier transformation
        self._irfft_in[:] = fast_multiply(self._rfft_out1, self._rfft_out2)
        result = self._fftw_irfft()

        # Return the 'full' elements
        # return result[self.valid_slice]
        return result


if __name__ == "__main__":
    # TRY CuPy:
    # https://www.appsloveworld.com/numpy/100/83/how-to-do-100000-times-2d-fft-in-a-faster-way-using-python

    from scipy.signal import windows

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

    size_A = 512
    A = gaussian_kernel(size_A, size_A / 8)
    B = gaussian_kernel(size_A, size_A / 2)

    fftw_1 = FFTW_Convolver_Full2D(A.shape, B.shape, fftw_threads=1)

    C = fftw_1.convolve(A, B)
