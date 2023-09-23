#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performs lightning-fast convolutions.
All code is fully jitted by `numba` and running in `No Python` mode.
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "23-09-2023"
__version__ = "1.0.0"
# pylint: disable=invalid-name, missing-function-docstring

import numpy as np
import numpy.typing as npt
import numba as nb
from scipy import fft as sp_fft

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
#   FFT_Convolver2D_Full
# ------------------------------------------------------------------------------

# fmt: off
spec = [
    ("fft_threads", nb.int64),
    ("_slice_full", nb.types.UniTuple(nb.types.slice2_type, 2)),
    ("_rfft_in1"  , nb.float32  [:, ::1]),
    ("_rfft_in2"  , nb.float32  [:, ::1]),
    ("_irfft_in"  , nb.complex64[:, ::1]),
]
# fmt: on


@nb.experimental.jitclass(spec=spec)  # type: ignore
class FFT_Convolver2D_Full:
    """Manages a fast-Fourier transform (FFT) convolution on 2D input arrays
    `in1` and `in2` to be passed to method `convolve()`, which will return the
    result as a `numpy.ndarray` containing the 'full' convolution elements.

    Args:
        s1 (tuple[int, int]):
            Shape of the upcoming input array `in1` to be passed to method
            `convolve()`.

        s2 (tuple[int, int]):
            Shape of the upcoming input array `in2` to be passed to method
            `convolve()`.

        fft_threads (int, optional):
            Number of threads to use for each individual FFT transformation.

            Default: 1
    """

    def __init__(
        self,
        s1: tuple[int, ...],
        s2: tuple[int, ...],
        fft_threads: int = 1,
    ):
        # Example:   s1 = (64, 64), s2 = (64, 64)
        # shape      evaluates to (127, 127)
        # fshape     evaluates to (128, 128)
        # fshape_out evaluates to (128, 65)
        # slice_full evaluates to ((0:127), (0:127))

        # Ensure at least 1 thread
        self.fft_threads = np.int64(np.maximum(fft_threads, 1))

        # Speed up FFT by zero-padding to optimal size
        shape = (s1[0] + s2[0] - 1, s1[1] + s2[1] - 1)
        fshape = (
            sp_fft.next_fast_len(np.int64(shape[0]), real=True),
            sp_fft.next_fast_len(np.int64(shape[1]), real=True),
        )
        fshape_out = (fshape[0], fshape[1] // 2 + 1)

        # Slice corresponding to the 'full' convolution elements to be
        # finally returned as convolution result
        self._slice_full = (slice(shape[0]), slice(shape[1]))

        # fmt: off
        # Allocate C-contiguous arrays to speed up calculations
        self._rfft_in1  = np.zeros(fshape    , dtype="float32")
        self._rfft_in2  = np.zeros(fshape    , dtype="float32")
        self._irfft_in  = np.zeros(fshape_out, dtype="complex64")
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
            equal to: `(in1.shape[0] + in2.shape[0] - 1, in1.shape[1] +
            in2.shape[1] - 1)`.

        NOTE: `in1` and `in2` do not necessarily have to be C-contiguous because
        they will, internal to this method, get copied into pre-allocated
        C-contiguous arrays during the zero-padding operation.
        NOTE: The output array is not contiguous.
        """

        # Zero padding
        self._rfft_in1[: in1.shape[0], : in1.shape[1]] = in1
        self._rfft_in2[: in2.shape[0], : in2.shape[1]] = in2

        # Forwards Fourier transformations
        _rfft_out1 = sp_fft.rfft2(
            self._rfft_in1, overwrite_x=True, workers=self.fft_threads
        )
        _rfft_out2 = sp_fft.rfft2(
            self._rfft_in2, overwrite_x=True, workers=self.fft_threads
        )

        # Convolution and backwards Fourier transformation
        fast_multiply(_rfft_out1, _rfft_out2, self._irfft_in)
        _irfft_out = sp_fft.irfft2(
            self._irfft_in, overwrite_x=True, workers=self.fft_threads
        )

        # Return the 'full' elements
        return _irfft_out[self._slice_full]


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
    fig.suptitle("dvg_fftconvolver_scipy")
    axs[0].title.set_text("A")
    axs[1].title.set_text("B")
    axs[2].title.set_text("convolve(A, B)")
    plt.show()

    # Timeit different `fast_multiply()` schemes
    if 0:
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
