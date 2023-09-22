#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DO NOT USE: Suffers from edge effects, because we do not zero-pad.
Performs lightning-fast convolutions.

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
__date__ = "22-09-2023"
__version__ = "1.0.0"
# pylint: disable=invalid-name, missing-function-docstring

import sys
import numpy as np
import numpy.typing as npt
import numba as nb
import pyfftw

# ------------------------------------------------------------------------------
#   fast_multiply
# ------------------------------------------------------------------------------

"""Timeit results on computer Onera:

                [ms] per iter
shape           fast_multiply   fast_multiply_p
(32, 32)        --> 0.00102         0.01334
(64, 64)        --> 0.00222         0.01491
(128, 128)      --> 0.00718         0.02000
(256, 256)      --> 0.02753         0.03023
(512, 512)          0.10537     --> 0.06094
(1024, 1024)        0.78878     --> 0.40209
(2048, 2048)        4.00985     --> 3.85964
"""


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


@nb.njit(
    "(complex64[:, ::1], complex64[:, ::1], complex64[:, ::1])",
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
      `fast_multiply()` when shape is smaller.
    """
    for i in nb.prange(in1.shape[0]):
        for j in nb.prange(in1.shape[1]):
            out[i, j] = in1[i, j] * in2[i, j]


# ------------------------------------------------------------------------------
#   fast_fftshift
# ------------------------------------------------------------------------------


@nb.njit(
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


# ------------------------------------------------------------------------------
#   FFT_Convolver2D_Full
# ------------------------------------------------------------------------------


class FFT_Convolver2D_Full:
    """Manages a fast-Fourier transform (FFT) convolution on 2D input arrays
    `in1` and `in2` to be passed to method `convolve()`, which will return the
    result as a contiguous C-style `numpy.ndarray` containing the 'full'
    convolution elements.

    Here, we demand both `in1` and `in2` to be of equal size for calculation
    speed improvements.

    Args:
        s1 (tuple[int, int]):
            Shape of the upcoming input array `in1` to be passed to method
            `convolve()`.

        s2 (tuple[int, int]):
            Shape of the upcoming input array `in2` to be passed to method
            `convolve()`.

        fft_threads (int, optional):
            Number of threads to use for each individual FFT transformation.
            When set to > 1, the Python GIL will not be invoked.

            Default: 1
    """

    def __init__(
        self,
        s1: tuple[int, ...],
        s2: tuple[int, ...],
        fft_threads: int = 1,
    ):
        # Example:   s1 = (64, 64), s2 = (64, 64)
        # s_out      evaluates to (64, 33)

        # Ensure at least 1 thread
        fft_threads = int(np.maximum(fft_threads, 1))

        # fmt: off
        # Allocate C-contiguous arrays to speed up calculations
        s_out = (s1[0], s1[1] // 2 + 1)
        self._rfft_in1  = pyfftw.zeros_aligned(s1   , dtype="float32")
        self._rfft_in2  = pyfftw.zeros_aligned(s1   , dtype="float32")
        self._rfft_out1 = pyfftw.empty_aligned(s_out, dtype="complex64")
        self._rfft_out2 = pyfftw.empty_aligned(s_out, dtype="complex64")
        self._irfft_in  = pyfftw.empty_aligned(s_out, dtype="complex64")
        self._irfft_out = pyfftw.empty_aligned(s1   , dtype="float32")
        # fmt: on

        # Create the FFTW plans
        print("Creating FFTW plans for convolution...", end="")
        sys.stdout.flush()

        p = {
            "axes": (0, 1),
            "flags": ("FFTW_MEASURE",),
            "threads": fft_threads,
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
        self,
        in1: npt.NDArray[np.float32],
        in2: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Performs the FFT convolution on input arrays `in1` and `in2` and
        returns the result as a contiguous C-style `numpy.ndarray` containing
        the 'full' convolution elements.

        Returns:
            The full convolution results as a 2D numpy array with a shape
            equal to `in1`.

        NOTE: `in1` and `in2` do not necessarily have to be C-contiguous because
        they will, internal to this method, get copied into pre-allocated
        C-contiguous arrays during the zero-padding operation.
        NOTE: The output array is C-contiguous.
        """

        # Forwards Fourier transformations
        self._rfft_in1[:] = in1
        self._rfft_in2[:] = in2
        self._fftw_rfft1()
        self._fftw_rfft2()

        # Convolution and backwards Fourier transformation
        fast_multiply(self._rfft_out1, self._rfft_out2, self._irfft_in)
        result = self._fftw_irfft()

        # Return the 'full' elements
        return fast_fftshift(result)


# ------------------------------------------------------------------------------
#   Main (timeit and demo)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import timeit
    import matplotlib.pyplot as plt

    s1 = (64, 64)  # shape 1
    s2 = (64, 64)  # shape 2

    # Create black images containing a white bar
    w = 4  # width fraction of the white bar
    h = 4  # height fraction of the white bar
    m1 = [x // 2 for x in s1]  # middle 1
    m2 = [x // 2 for x in s2]  # middle 2

    A = np.zeros(s1, dtype=np.float32)
    B = np.zeros(s2, dtype=np.float32)
    A[
        m1[0] - s1[0] // h : m1[0] + s1[0] // h,
        m1[1] - s1[1] // w : m1[1] + s1[1] // w,
    ] = 1
    B[
        m2[0] - s2[0] // w : m2[0] + s2[0] // w,
        m2[1] - s2[1] // h : m2[1] + s2[1] // h,
    ] = 1

    fft = FFT_Convolver2D_Full(A.shape, B.shape, fft_threads=1)

    # Timeit
    loop = int(1e3)
    result = timeit.timeit(
        "fft.convolve(A, B)",
        setup=lambda: fft.convolve(A, B),
        globals=globals(),
        number=loop,
    )
    result = result / loop * 1000
    print(f"{result:.2f} [ms] per iter")

    # Plot
    C = fft.convolve(A, B)

    p = {"cmap": "jet", "interpolation": "none"}
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(A, **p)
    axs[1].imshow(B, **p)
    axs[2].imshow(C, **p)
    fig.suptitle("dvg_fftconvolver_pyfftw")
    axs[0].title.set_text("A")
    axs[1].title.set_text("B")
    axs[2].title.set_text("convolve(A, B)")
    plt.show()
