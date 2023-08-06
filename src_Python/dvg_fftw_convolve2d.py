#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performs lightning-fast convolutions on 2D input arrays.

The convolution is based on the fast-Fourier transform (FFT) as performed by the
excellent `fftw` (http://www.fftw.org) library. It will plan the transformations
ahead of time to optimize the calculations. Also, multiple threads can be
specified for the FFT and, when set to > 1, the Python GIL will not be invoked.
This results in true multithreading across multiple cores, which can result in a
huge performance gain.
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
        self.fshape = [pyfftw.next_fast_len(shape[a]) for a in axes]
        fshape_out = [self.fshape[0], self.fshape[1] // 2 + 1]

        # Slice corresponding to the 'full' convolution elements to be
        # finally returned as convolution result
        self.fslice = tuple([slice(sz) for sz in shape])

        # Create the FFTW plans
        # fmt: off
        self._rfft_in1  = pyfftw.zeros_aligned(self.fshape, dtype="float32")
        self._rfft_in2  = pyfftw.zeros_aligned(self.fshape, dtype="float32")

        self._rfft_out1 = pyfftw.empty_aligned(fshape_out, dtype="complex64")
        self._rfft_out2 = pyfftw.empty_aligned(fshape_out, dtype="complex64")

        self._irfft_in  = pyfftw.empty_aligned(fshape_out, dtype="complex64")
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
            equal to `in1 + in2 - 1`.
        """
        # Force contiguous C-style numpy arrays, super fast when already so
        in1 = np.asarray(in1)
        in2 = np.asarray(in2)

        # Perform FFT convolution
        # -----------------------
        # Zero padding and forwards Fourier transformation
        self._rfft_in1[: in1.shape[0], : in1.shape[1]] = in1
        self._rfft_in2[: in2.shape[0], : in2.shape[1]] = in2
        self._fftw_rfft1()
        self._fftw_rfft2()

        # Convolution and backwards Fourier transformation
        fast_multiply(self._rfft_out1, self._rfft_out2, self._irfft_in)
        result = self._fftw_irfft()

        # Return the 'full' elements
        return result[self.fslice]


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

        fftw_1 = FFTW_Convolver_Full2D(A.shape, B.shape, fftw_threads=1)
        C = fftw_1.convolve(A, B)

        plt.figure(1)
        plt.imshow(A, cmap="gray")
        plt.figure(2)
        plt.imshow(B, cmap="gray")
        plt.figure(3)
        plt.imshow(C, cmap="gray")
        plt.show()

    if 1:  # Timeit different `fast_multiply()` schemes
        shapes = (
            (32, 32),
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        )
        results1 = []
        results2 = []
        loop = int(1e3)

        print("Timeit different `fast_multiply()` schemes:")
        for shape in shapes:
            print(f"  shape: {shape}")
            # fmt: off
            np.random.seed(0)
            a = (np.random.uniform(-1, 1, shape) +
                 np.random.uniform( 1, 1, shape) * 1.0j)
            b = (np.random.uniform(-1, 1, shape) +
                 np.random.uniform( 1, 1, shape) * 1.0j)
            # fmt: on
            a = np.asarray(a, dtype=np.complex64)
            b = np.asarray(b, dtype=np.complex64)
            out = np.empty(a.shape, dtype=a.dtype)

            result = timeit.timeit(
                "fast_multiply(a, b, out)",
                setup=lambda: fast_multiply(a, b, out),
                globals=globals(),
                number=loop,
            )
            result = result / loop * 1000
            results1.append(result)

            result = timeit.timeit(
                "fast_multiply_p(a, b, out)",
                setup=lambda: fast_multiply_p(a, b, out),
                globals=globals(),
                number=loop,
            )
            result = result / loop * 1000
            results2.append(result)

        print("\n")
        print("                [ms] per iter")
        print("shape           fast_multiply   fast_multiply_p")
        for i in range(len(shapes)):
            print(
                f"{str(shapes[i]):16s}{results1[i]:<16.5f}{results2[i]:<16.5f}"
            )
