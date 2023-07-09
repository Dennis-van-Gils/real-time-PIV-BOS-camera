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


class FFTW_Convolver_Valid2D:
    """Manages a fast-Fourier transform (FFT) convolution on 2D input arrays
    `in1` and `in2` as passed to method `convolve()`, which will return the
    result as a contiguous C-style `numpy.ndarray` containing only the 'valid'
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
        # Check that input shapes are compatible with 'valid' mode
        self.switch_inputs = np.prod(shape2) > np.prod(shape1)
        if self.switch_inputs:
            shape1, shape2 = shape2, shape1

        self.shape1 = shape1
        self.shape2 = shape2

        # Speed up FFT by zero-padding to optimal size for FFTW
        self.fast_shape = tuple(
            np.maximum(
                np.array(shape1) + np.array(shape2) - 1, np.array(shape1)
            )
        )
        self.padding_in1 = np.zeros(np.subtract(self.fast_shape, self.shape1))
        self.padding_in2 = np.zeros(np.subtract(self.fast_shape, self.shape2))

        # Compute the slice containing the valid convolution results
        self.valid_shape = np.subtract(self.shape1, self.shape2) + 1
        idx_start = tuple((np.array(self.shape2) - 1) // 2)
        idx_end = tuple(np.add(idx_start, self.valid_shape))
        self.valid_slice = tuple(
            slice(start, end) for start, end in zip(idx_start, idx_end)
        )

        # Create the FFTW plans
        # fmt: off
        fast_shape2 = tuple(np.array(self.fast_shape) // 2 + 1)
        self._rfft_in1  = pyfftw.empty_aligned(self.fast_shape, dtype="float64")
        self._rfft_in2  = pyfftw.empty_aligned(self.fast_shape, dtype="float64")
        self._rfft_out1 = pyfftw.empty_aligned(fast_shape2    , dtype="complex128")
        self._rfft_out2 = pyfftw.empty_aligned(fast_shape2    , dtype="complex128")
        self._irfft_in  = pyfftw.empty_aligned(fast_shape2    , dtype="complex128")
        self._irfft_out = pyfftw.empty_aligned(self.fast_shape, dtype="float64")
        # fmt: on

        print("Creating FFTW plans for convolution...", end="")
        sys.stdout.flush()

        p = {
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
        only the 'valid' convolution elements.

        When the shapes of the passed input arrays are not compatible with the
        convolution operation, an array full of `np.nan`s is returned.

        Returns:
            The valid convolution results as a 2D numpy array.
        """
        # Force contiguous C-style numpy arrays, super fast when already so
        in1 = np.asarray(in1)
        in2 = np.asarray(in2)

        # Return np.nans when the input arrays are not fully populated yet
        if in1.shape != self.shape1 or in2.shape != self.shape2:
            return np.full(self.valid_shape, np.nan)

        # Check that input shapes are compatible with 'valid' mode
        if self.switch_inputs:
            in1, in2 = in2, in1

        # Perform FFT convolution
        # -----------------------
        # Zero padding and forwards Fourier transformation
        self._rfft_in1[:] = np.pad(
            in1,
            [(0, p) for p in np.subtract(self.fast_shape, self.shape1)],
            mode="constant",
        )
        self._rfft_in2[:] = np.pad(
            in2,
            [(0, p) for p in np.subtract(self.fast_shape, self.shape2)],
            mode="constant",
        )
        self._fftw_rfft1()
        self._fftw_rfft2()

        # Convolution and backwards Fourier transformation
        self._irfft_in[:] = fast_multiply(self._rfft_out1, self._rfft_out2)
        result = self._fftw_irfft()

        # Return only the 'valid' elements
        return result[self.valid_slice]
