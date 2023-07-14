#!/usr/bin/env python3
# -*- coding: utf-8 -

import sys
import numpy as np
import pyfftw
from numba import njit, jit, prange


@njit(
    "complex128[:, :](complex128[:, :], complex128[:, :])",
    nogil=True,
    cache=True,
)
def fast_multiply(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    return np.multiply(in1, in2)


@njit(
    # parallel=True, # detrimental to performance
    nogil=True,
    cache=True,
)
def fast_zero_pad_2D(arr_in, arr_zeros):
    """
    for i in range(arr_in.shape[0]):
        for j in range(arr_in.shape[1]):
            arr_zeros[i, j] = arr_in[i, j]
    """
    arr_zeros[: arr_in.shape[0], : arr_in.shape[1]] = arr_in

    return arr_zeros


class FFTW_Convolver_Full2D:
    """Manages a fast-Fourier transform (FFT) convolution on 2D input arrays
    `in1` and `in2` as passed to method `convolve()`, which will return the
    result as a contiguous C-style `numpy.ndarray` containing the 'full'
    convolution elements.

    When the shapes of the passed input arrays are not compatible with the
    convolution operation, an array full of `numpy.nan`s is returned.

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

    def __init__(self, s1: tuple, s2: tuple, fftw_threads: int = 5):
        self.s1 = s1
        self.s2 = s2

        axes = (0, 1)
        shape = [
            max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
            for i in range(2)
        ]
        # shape = (127, 127)

        # Speed up FFT by padding to optimal size.
        self.fshape = [pyfftw.next_fast_len(shape[a]) for a in axes]
        # fshape = (128, 128)

        # Predefined zero-padding
        self.zero_padding_1 = np.zeros(self.fshape)
        self.zero_padding_2 = np.zeros(self.fshape)

        fshape_out = [self.fshape[0], self.fshape[1] // 2 + 1]

        # if calc_fast_len:
        if True:
            self.fslice = tuple([slice(sz) for sz in shape])

        # Create the FFTW plans
        # fmt: off
        self._rfft_in1  = pyfftw.empty_aligned(self.fshape, dtype="float64")
        self._rfft_in2  = pyfftw.empty_aligned(self.fshape, dtype="float64")

        self._rfft_out1 = pyfftw.empty_aligned(fshape_out, dtype="complex128")
        self._rfft_out2 = pyfftw.empty_aligned(fshape_out, dtype="complex128")

        self._irfft_in  = pyfftw.empty_aligned(fshape_out, dtype="complex128")
        self._irfft_out = pyfftw.empty_aligned(self.fshape, dtype="float64")
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
        # self._rfft_in1[:] = np.pad(
        #     in1, [(0, p) for p in np.subtract(self.fshape, self.s1)]
        # )
        self._rfft_in1[:] = fast_zero_pad_2D(in1, self.zero_padding_1)
        # self._rfft_in2[:] = np.pad(
        #     in2, [(0, p) for p in np.subtract(self.fshape, self.s2)]
        # )
        self._rfft_in2[:] = fast_zero_pad_2D(in2, self.zero_padding_2)
        self._fftw_rfft1()
        self._fftw_rfft2()

        # Convolution and backwards Fourier transformation
        self._irfft_in[:] = fast_multiply(self._rfft_out1, self._rfft_out2)
        result = self._fftw_irfft()

        # Return the 'full' elements
        # return result[self.valid_slice]
        return result[self.fslice]


if __name__ == "__main__":
    # TRY CuPy:
    # https://www.appsloveworld.com/numpy/100/83/how-to-do-100000-times-2d-fft-in-a-faster-way-using-python

    from scipy.signal import windows
    from scipy.signal import _signaltools
    from scipy import fft as sp_fft
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

    # in1, in2, axes = _signaltools._init_freq_conv_axes(
    #     A, B, mode="full", axes=None, sorted_axes=False
    # )
    in1 = A
    in2 = B
    s1 = A.shape
    s2 = B.shape
    axes = (0, 1)
    shape = [
        max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
        for i in range(in1.ndim)
    ]
    # shape = (127, 127)

    # Speed up FFT by padding to optimal size.
    # fshape = [sp_fft.next_fast_len(shape[a], not complex_result) for a in axes]
    fshape = [pyfftw.next_fast_len(shape[a]) for a in axes]
    # fshape = (128, 128)
    # sp1 = sp_fft.rfftn(in1, fshape, axes=axes)
    # sp2 = sp_fft.rfftn(in2, fshape, axes=axes)
    # sp1.shape = (128, 65)
    # sp1.shape = (128, 65)

    fftw_1 = FFTW_Convolver_Full2D(A.shape, B.shape, fftw_threads=1)

    C = fftw_1.convolve(A, B)

    plt.figure(1)
    plt.imshow(A)
    plt.figure(2)
    plt.imshow(B)
    plt.figure(3)
    plt.imshow(C)

    plt.show()
