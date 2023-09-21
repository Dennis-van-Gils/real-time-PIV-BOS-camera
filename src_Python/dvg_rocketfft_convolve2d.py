#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performs lightning-fast convolutions on 2D input arrays.
All code is fully jitted and running in NoPython mode.

Rocket-FFT:
https://github.com/styfenschaer/rocket-fft
https://github.com/styfenschaer/rocket-fft/blob/main/tests/test_scipy_testsuite.py
https://numba.discourse.group/t/rocket-fft-a-numba-extension-supporting-numpy-fft-and-scipy-fft/1657/5
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "21-09-2023"
__version__ = "1.0.0"
# pylint: disable=invalid-name, missing-function-docstring

import numpy as np
import numpy.typing as npt
import numba as nb
import rocket_fft

rocket_fft.scipy_like()

# ------------------------------------------------------------------------------
#   fast_multiply
# ------------------------------------------------------------------------------


@nb.njit(
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
#   FFT_Convolver_Full2D
# ------------------------------------------------------------------------------

# fmt: off
spec = [
    ("fft_threads", nb.int64),
    ("full_slice_stop", nb.int64),
    ("_rfft_in1" , nb.float32[:, ::1]),
    ("_rfft_in2" , nb.float32[:, ::1]),
    ("_rfft_out1", nb.complex64[:, ::1]),
    ("_rfft_out2", nb.complex64[:, ::1]),
    ("_irfft_in" , nb.complex64[:, ::1]),
    ("_irfft_out", nb.float32[:, ::1]),
]
# fmt: on


@nb.experimental.jitclass(spec=spec)  # type: ignore
class FFT_Convolver_Full2D:
    """Manages a fast-Fourier transform (FFT) convolution on 2D input arrays
    `in1` and `in2` as passed to method `convolve()`, which will return the
    result as a `numpy.ndarray` containing the 'full' convolution elements.
    Arrays `in1` and `in2` are each of size NxN.

    Args:
        N (int):
            NxN shape of the upcoming input arrays `in1` and `in2` passed to
            method `convolve()`.

        fft_threads (int, optional):
            Number of threads to use for each individual FFT transformation.

            Default: 1
    """

    def __init__(self, N: int, fft_threads: int = 1):
        # Example:   N = 64
        # shape      evaluates to (127, 127)
        # fshape     evaluates to (128, 128)
        # fshape_out evaluates to (128, 65)
        # fslice     evaluates to ((0:127), (0:127))

        self.fft_threads = np.int64(np.maximum(fft_threads, 1))

        # Speed up FFT by padding to optimal size
        shape = (N * 2 - 1, N * 2 - 1)
        fshape = (
            rocket_fft.good_size(np.int64(shape[0]), real=True),
            rocket_fft.good_size(np.int64(shape[1]), real=True),
        )
        fshape_out = (fshape[0], fshape[1] // 2 + 1)

        # Slice stop-point corresponding to the 'full' convolution elements to
        # be finally returned as convolution result
        self.full_slice_stop = shape[0]

        # fmt: off
        self._rfft_in1  = np.zeros(fshape    , dtype="float32")
        self._rfft_in2  = np.zeros(fshape    , dtype="float32")
        self._rfft_out1 = np.zeros(fshape_out, dtype="complex64")
        self._rfft_out2 = np.zeros(fshape_out, dtype="complex64")
        self._irfft_in  = np.zeros(fshape_out, dtype="complex64")
        self._irfft_out = np.zeros(fshape    , dtype="float32")
        # fmt: on

    # --------------------------------------------------------------------------
    #   convolve
    # --------------------------------------------------------------------------

    def convolve(
        self,
        in1: npt.NDArray[np.float32],
        in2: npt.NDArray[np.float32],
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

        # Forward Fourier transformations
        rfftn(self._rfft_in1, self._rfft_out1, nthreads=self.fft_threads)
        rfftn(self._rfft_in2, self._rfft_out2, nthreads=self.fft_threads)

        # Convolution and backwards Fourier transformation
        fast_multiply(self._rfft_out1, self._rfft_out2, self._irfft_in)
        irfftn(self._irfft_in, self._irfft_out, nthreads=self.fft_threads)
        result = self._irfft_out

        # Return the 'full' elements
        return result[0 : self.full_slice_stop, 0 : self.full_slice_stop]


# ------------------------------------------------------------------------------
#   Low-level rocket-fft
# ------------------------------------------------------------------------------


@nb.njit(
    "(float32[:, ::1], complex64[:, ::1], int64)",
    cache=True,
    nogil=True,
)
def rfftn(ain, aout, nthreads=np.int64(1)):
    axes = np.array([0, 1], dtype=np.int64)
    forward = True
    fct = np.float32(1.0)
    rocket_fft.r2c(ain, aout, axes, forward, fct, nthreads)


@nb.njit(
    "(complex64[:, ::1], float32[:, ::1], int64)",
    cache=True,
    nogil=True,
)
def irfftn(ain, aout, nthreads=np.int64(1)):
    axes = np.array([0, 1], dtype=np.int64)
    forward = False
    fct = np.float32(1.0)
    rocket_fft.c2r(ain, aout, axes, forward, fct, nthreads)
