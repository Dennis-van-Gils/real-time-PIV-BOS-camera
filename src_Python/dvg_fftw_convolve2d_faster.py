#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DO NOT USE: Suffers from edge effects, because we do not zero-append.
Performs lightning-fast convolutions on 2D input arrays.

The convolution is based on the fast-Fourier transform (FFT) as performed by the
excellent `fftw` (http://www.fftw.org) library. It will plan the transformations
ahead of time to optimize the calculations. Also, multiple threads can be
specified for the FFT and, when set to > 1, the Python GIL will not be invoked.
This results in true multithreading across multiple cores, which can result in a
huge performance gain.

This 'faster' version does not zero-append the input arrays to the next fast
length. It is hence faster, because the matrices it operates on are smaller, but
the output is less accurate due to edge effects when convoluting.

I asked ChatGPT the following
-----------------------------

I am performing a 2D correlation using FFT with zero-padding to the next fast
length. What are the differences in the output matrix if I would not zero-pad to
the next fast length? Assume the input matrices are powers of 2.

ChatGPT answer:

Zero-padding plays an important role in 2D correlation using FFT. When you
perform a 2D correlation using FFT, zero-padding to the next power of 2 (fast
length) can affect the output matrix in several ways compared to not
zero-padding to the next power of 2.

Here are the key differences between the two cases:

    Circular Convolution vs. Linear Convolution:
        Without zero-padding: If you don't zero-pad, the FFT-based convolution
        will result in circular convolution. This means that the output will
        wrap around at the edges, which might not be suitable for many
        applications.

        With zero-padding: Zero-padding allows the FFT-based convolution to
        perform linear convolution. Linear convolution results in a proper
        overlap-add method, where the output size matches the size of the input
        matrices without wrap-around effects.

    Resolution and Accuracy:
        Without zero-padding: In the case of circular convolution, you may
        encounter aliasing effects, where the high-frequency components from the
        edges wrap around and interfere with the desired results.

        With zero-padding: Zero-padding increases the resolution of the
        convolution result by reducing aliasing. It provides more accurate
        results since it effectively introduces additional "virtual" data points
        in the input matrices, allowing the convolution to better capture
        high-frequency information.

    Output Size:
        Without zero-padding: The output size of the convolution will be equal
        to the sum of the dimensions of the input matrices minus 1 (due to
        circular convolution).

        With zero-padding: The output size of the convolution will be the sum of
        the dimensions of the zero-padded matrices minus 1, resulting in a
        larger output matrix.

    Edge Effects:
        Without zero-padding: Circular convolution can produce artifacts at the
        edges due to the wrap-around behavior.

        With zero-padding: Linear convolution with zero-padding mitigates edge
        effects, resulting in cleaner results near the edges of the output
        matrix.

    Computational Efficiency:
        With zero-padding: While zero-padding may seem to introduce some
        computational overhead, it is usually not a significant factor, and the
        benefits of accurate results and reduced artifacts typically outweigh
        this minor cost.

In summary, zero-padding to the next power of 2 when performing 2D correlation
using FFT is generally recommended. It helps avoid circular convolution
artifacts, improves accuracy, reduces edge effects, and produces a more
meaningful and usable output matrix.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "06-08-2023"
__version__ = "1.0.0"
# pylint: disable=invalid-name, missing-function-docstring

import sys
import numpy as np
import numpy.typing as npt
import pyfftw
from numba import njit, prange

# ------------------------------------------------------------------------------
#   fast_multiply
# ------------------------------------------------------------------------------

"""Timeit results on computer Onera:

                [ms] per iter
shape           fast_multiply   fast_multiply_p
(32, 32)        --> 0.00150         0.01414
(64, 64)        --> 0.00409         0.01787
(128, 128)      --> 0.01537         0.02439
(256, 256)          0.05726     --> 0.03989
(512, 512)          0.22363     --> 0.08859
(1024, 1024)        1.09012     --> 0.32051
(2048, 2048)        4.37083     --> 3.89286
"""


@njit(
    "(complex64[:, :], complex64[:, :], complex64[:, :])",
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


@njit(
    "(complex64[:, :], complex64[:, :], complex64[:, :])",
    nogil=True,
    cache=True,
    parallel=True,
)
def fast_multiply_p(
    in1: npt.NDArray[np.complex64],
    in2: npt.NDArray[np.complex64],
    out: npt.NDArray[np.complex64],
):
    """
    * In-place operation on `out`.
    * Faster version of `out = np.multiply(in1, in2)`.
    * Parallelized. Only beneficial for `shape >~ (256, 256)`. Use
    `fast_multiply_o()` when shape is smaller.
    """
    for i in prange(in1.shape[0]):
        for j in prange(in1.shape[1]):
            out[i, j] = in1[i, j] * in2[i, j]


# ------------------------------------------------------------------------------
#   FFTW_Convolver_Full2D
# ------------------------------------------------------------------------------


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
        # Example: s = (64, 64)
        # s_out    evaluates to (64, 33)

        self.s = s
        s_out = (s[0], s[1] // 2 + 1)

        # Create the FFTW plans
        # fmt: off
        self._rfft_in1  = pyfftw.zeros_aligned(s    , dtype="float32")
        self._rfft_in2  = pyfftw.zeros_aligned(s    , dtype="float32")

        self._rfft_out1 = pyfftw.empty_aligned(s_out, dtype="complex64")
        self._rfft_out2 = pyfftw.empty_aligned(s_out, dtype="complex64")

        self._irfft_in  = pyfftw.empty_aligned(s_out, dtype="complex64")
        self._irfft_out = pyfftw.empty_aligned(s    , dtype="float32")
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

    def convolve(
        self, in1: npt.NDArray[np.float32], in2: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Performs the FFT convolution on input arrays `in1` and `in2` and
        returns the result as a contiguous C-style `numpy.ndarray` containing
        the 'full' convolution elements.

        When the shapes of the passed input arrays are not compatible with the
        convolution operation, an array full of `np.nan`s is returned.

        Returns:
            The full convolution results as a 2D numpy array with a shape
            equal to `in1`.
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
        fast_multiply(self._rfft_out1, self._rfft_out2, self._irfft_in)
        result = self._fftw_irfft()

        # Return the 'full' elements
        return fast_fftshift(result)


@njit(
    "float32[:, :](float32[:, :])",
    nogil=True,
    cache=True,
)
def fast_fftshift(C):
    """Like `numpy.fft.fftshift(), but faster."""
    rows, cols = C.shape
    half_rows = rows // 2
    half_cols = cols // 2

    # Swap quadrants
    shifted = np.empty_like(C)
    shifted[:half_rows, :half_cols] = C[half_rows:, half_cols:]
    shifted[:half_rows, half_cols:] = C[half_rows:, :half_cols]
    shifted[half_rows:, :half_cols] = C[:half_rows, half_cols:]
    shifted[half_rows:, half_cols:] = C[:half_rows, :half_cols]

    return shifted


if __name__ == "__main__":
    # TRY CuPy:
    # https://www.appsloveworld.com/numpy/100/83/how-to-do-100000-times-2d-fft-in-a-faster-way-using-python

    import timeit
    from scipy.signal import windows
    import matplotlib.pyplot as plt

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

    if 0:
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
