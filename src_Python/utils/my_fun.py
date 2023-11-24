#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/real-time-PIV-BOS-camera"
__date__ = "24-11-2023"

import numpy as np
import numpy.typing as npt
import numba as nb


# ------------------------------------------------------------------------------
#   bool2on
# ------------------------------------------------------------------------------


def bool2on(state: bool) -> str:
    return "ON" if state else "OFF"


# ------------------------------------------------------------------------------
#   remove_mean_background
# ------------------------------------------------------------------------------


@nb.njit(
    "(float32[:, ::1], )",
    parallel=True,
    cache=True,
    nogil=True,
    fastmath=True,
)
def remove_mean_background(img: npt.NDArray[np.float32]):
    """In-place operation on `img`.

    BENCHMARK on computer `Onera`:
        Vanilla numpy: return img - np.mean(img)
        Ufunc numpy  : np.subtract(img, np.mean(img), out=img)

        4096 x 4096 @ float32:
            vanilla numpy    : 32    ms per iter
            ufunc numpy      : 19    ms per iter
            numba no parallel:  7.8  ms per iter
            numba parallel   :  5.6  ms per iter

        1024 x 1024 @ float32:
            numba parallel   :  0.23 ms per iter
    """

    sigma = 0  # sum
    for y in nb.prange(img.shape[0]):
        for x in nb.prange(img.shape[1]):
            sigma += img[y, x]
    mu = sigma / img.size  # mean

    for y in nb.prange(img.shape[0]):
        for x in nb.prange(img.shape[1]):
            img[y, x] -= mu


# ------------------------------------------------------------------------------
#   fliplrud
# ------------------------------------------------------------------------------


@nb.njit(
    "float32[:, :](float32[:, :], )",
    parallel=False,  # Can't parallelize
    cache=True,
    nogil=True,
)
def fliplrud(img: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Flip the passed image left-to-right and up-to-down, necessary to use an
    FFT convolution as a 2D-correlation.
    NOTE: The returned matrix will /not/ be contiguous, regardless of whether
    the input matrix was C or Fortran-contiguous.
    """
    # Vanilla numpy:
    #   return np.flipud(np.fliplr(img))
    img = img[:, ::-1]  # fliplr
    img = img[::-1, ...]  # flipud

    return img


# ------------------------------------------------------------------------------
#   subpx_3pgf_2D
# ------------------------------------------------------------------------------


@nb.njit(
    "Tuple((float32, float32))(float32[:, :], int32, int32)",
    cache=True,
    nogil=True,
)
def subpx_3pgf_2D(
    C: npt.NDArray[np.float32], px_x: int, px_y: int
) -> tuple[float, float]:
    """Achieve sub-pixel resolution by employing a 3-point Gaussian fit to the
    point with index `(px_y, px_x)` inside of 2D matrix `C` along both the x and
    y axes.

    Returns (``tuple``):
        sub_px_x (``float``):
            Sub-pixel position along the x-axis [px].

        sub_px_y (``float``):
            Sub-pixel position along the y-axis [px].
    """

    # NOTE: We don't check for log(0) in `np.log(C[px_y, px_x])` because it
    # is garantued non-zero by the caller.
    if px_x > 0 and px_x < C.shape[1] - 1:
        m = np.log(np.maximum(C[px_y, px_x - 1], 1e-40))  # Prevent log of zero
        p = np.log(np.maximum(C[px_y, px_x + 1], 1e-40))  # Prevent log of zero
        sub_px_x = px_x + (m - p) / (m + p - 2 * np.log(C[px_y, px_x])) / 2
    else:
        sub_px_x = float(px_x)

    if px_y > 0 and px_y < C.shape[0] - 1:
        m = np.log(np.maximum(C[px_y - 1, px_x], 1e-40))  # Prevent log of zero
        p = np.log(np.maximum(C[px_y + 1, px_x], 1e-40))  # Prevent log of zero
        sub_px_y = px_y + (m - p) / (m + p - 2 * np.log(C[px_y, px_x])) / 2
    else:
        sub_px_y = float(px_y)

    return sub_px_x, sub_px_y


# ------------------------------------------------------------------------------
#   fast_max
# ------------------------------------------------------------------------------


@nb.njit(
    "float32(float32[:, :],)",
    # parallel=True,  # Not possible
    cache=True,
    nogil=True,
)
def fast_max(in1: npt.NDArray[np.float32]) -> np.float32:
    """Numba-accelerated version of `np.max()`.
    NOTE: We do not enforce the input matrix to be C-contiguous, because this
    module will pass discontiguous matrices into `fast_max()`. If we would
    enforce C-contiguity than the code executes ~2 times faster.
    """

    my_max = in1[0, 0]
    for i in range(in1.shape[0]):
        for j in range(in1.shape[1]):
            if my_max < in1[i, j]:
                my_max = in1[i, j]

    return my_max


# ------------------------------------------------------------------------------
#   fast_magnitude
#   NOTE: Don't use. Turns out native numpy outperforms this numba version.
# ------------------------------------------------------------------------------

'''
@nb.njit(
    "float32[:](float32[:], float32[:])",
    parallel=False,  # Setting to True is detrimental for this use-case
    cache=True,
    nogil=True,
)
def fast_magnitude(
    in1: npt.NDArray[np.float32],
    in2: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Numba-accelerated version of
    `M = np.sqrt(np.square(in1) + np.square(in2))`.

    NOTE: Don't use. Turns out native numpy outperforms this numba version.
    Timeit results on computer Onera
    --------------------------------
    fast_magnitude: 0.00681 ms per iter
    np_magnitude  : 0.00527 ms per iter
    """

    # Below block is commented out, because it was tested to be even slower
    # than `return np.sqrt(np.square(in1) + np.square(in2))`:
    # M = np.empty(in1.shape, dtype=np.float32)
    # for i in nb.prange(in1.shape[0]):
    #    M[i] = np.sqrt(np.square(in1[i]) + np.square(in2[i]))

    return np.sqrt(np.square(in1) + np.square(in2))
'''

# ------------------------------------------------------------------------------
#   all_smaller_or_equal_to
# ------------------------------------------------------------------------------


@nb.njit(
    "boolean(float32[:, :], float32)",
    # parallel=True,  # Not possible
    cache=True,
    nogil=True,
)
def all_smaller_or_equal_to(
    array_in: npt.NDArray[np.float32],
    value: float,
) -> bool:
    """Faster version of `np.max(array_in) <= value` because we can exit early
    here.
    NOTE: We do not enforce the input matrix to be C-contiguous, because this
    module will pass discontiguous matrices into `all_smaller_or_equal_to()`. If
    we would enforce C-contiguity than the code executes ~2 times faster.
    """
    for i in range(array_in.shape[0]):
        for j in range(array_in.shape[1]):
            if array_in[i, j] > value:
                return False

    return True


# ------------------------------------------------------------------------------
#   normalize_C_maps
# ------------------------------------------------------------------------------


@nb.njit(
    "(float32[:, :, :],)",
    parallel=True,
    cache=True,
    nogil=True,
)
def normalize_C_maps(C_maps: npt.NDArray[np.float32]):
    """In-place operation on `C_maps`.
    NOTE: It is not mandatory to normalize the correlations maps for the peak
    finding algorithm to work correctly. Normalizing wastes cpu time.
    """
    for IW_idx in nb.prange(C_maps.shape[0]):
        C = C_maps[IW_idx, :, :]
        if np.isnan(C[0, 0]):
            # Nothing sensible to normalize
            continue

        C_maps[IW_idx, :, :] = np.divide(C, np.max(C))


# ------------------------------------------------------------------------------
#   compute_displacement_vectors_from_C_maps
# ------------------------------------------------------------------------------


@nb.njit(
    "(float32[:, :, :], int32[::1], int32[::1], float32[::1], float32[::1], boolean)",
    parallel=True,
    cache=True,
    nogil=True,
)
def compute_displacement_vectors_from_C_maps(
    C_maps: npt.NDArray[np.float32],
    IW_shifts_x: npt.NDArray[np.int32],
    IW_shifts_y: npt.NDArray[np.int32],
    VM_dx: npt.NDArray[np.float32],
    VM_dy: npt.NDArray[np.float32],
    perform_subpixel_fitting: bool = False,
):
    """In-place operation on `VM_dx` and `VM_dy`.
    NOTE: The passed correlation maps do not have to be normalized for the peak
    finding algorithm to work correctly. Normalizing wastes cpu time."""
    for IW_idx in nb.prange(C_maps.shape[0]):
        C = C_maps[IW_idx, :, :]

        if np.isnan(C[0, 0]):
            dx = np.nan
            dy = np.nan
        else:
            # It is not necessary to normalize the correlation maps. Adds
            # overhead.
            # C = np.divide(C, np.max(C))  # Not necessary
            # Store back into C_maps. Adds another overhead.
            # C_maps[IW_idx, :, :] = C

            # Find maximum correlation peak
            iMaxC = int(np.argmax(C))
            peak_x = iMaxC % C.shape[1]  # unravel index
            peak_y = iMaxC // C.shape[1]  # unravel index

            if perform_subpixel_fitting:
                # Sub-pixel resolution algorithm, 3-point Gaussian fit
                peak_x, peak_y = subpx_3pgf_2D(C, peak_x, peak_y)

            # Calculate displacement vector
            """ This block is commented out. We assume zero-padding was used.
            # `qx` and `qy` are toggles for +1 or +0. They depend on whether
            # zero-padding was used for the 2D FFTW convolution (+0) or not
            # (+1).
            # With zero-padding   : correct results      , oddly  shape matrix
            # Without zero-padding: fast but edge defects, evenly shape matrix
            qx = 1 if (C.shape[0] % 2) == 0 else 0  # even / odd correction
            qy = 1 if (C.shape[1] % 2) == 0 else 0  # even / odd correction
            dx = peak_x - C.shape[1] // 2 + IW_shifts_x[IW_idx] + qx
            dy = peak_y - C.shape[0] // 2 + IW_shifts_y[IW_idx] + qy
            """
            dx = peak_x - C.shape[1] // 2 + IW_shifts_x[IW_idx]
            dy = peak_y - C.shape[0] // 2 + IW_shifts_y[IW_idx]

        # Store result in displacement vector map
        VM_dx[IW_idx] = dx
        VM_dy[IW_idx] = dy


# ------------------------------------------------------------------------------
#   main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import timeit
    from scipy.signal import fftconvolve

    print("timeit")
    print("------")

    if 1:
        img = np.random.randint(0, 255, (4096, 4096))
        # img = np.random.randint(0, 255, (1024, 1024))
        img = np.asarray(img, dtype=np.float32, order="C")

        loop = int(1e2)
        result = timeit.timeit(
            "remove_mean_background(img)",
            setup=lambda: remove_mean_background(img),  # type: ignore
            globals=globals(),
            number=loop,
        )
        print(f"remove_mean_bg: {result / loop * 1000:.5f} ms per iter")

    if 1:
        img = np.random.randint(0, 255, (1024, 1024), dtype=np.uint16)
        img = img.astype(np.float32)

        loop = int(1e6)
        result = timeit.timeit(
            "fliplrud(img)",
            setup=lambda: fliplrud(img),
            globals=globals(),
            number=loop,
        )
        print(f"fliplrud      : {result / loop * 1000:.5f} ms per iter")

    if 1:
        test_shape = (32, 32)
        A = np.random.randn(*test_shape)
        B = np.random.randn(*test_shape)
        C = fftconvolve(B, A, mode="full").astype(np.float32)
        C = C / np.max(C)

        loop = int(1e6)
        result = timeit.timeit(
            "subpx_3pgf_2D(C, 32, 32)",
            setup=lambda: subpx_3pgf_2D(C, 32, 32),
            globals=globals(),
            number=loop,
        )
        print(f"subpx_3pgf_2D : {result / loop * 1000:.5f} ms per iter")

    if 1:
        test_shape = (32, 32)
        A = np.random.randn(*test_shape)
        A = np.asarray(A, dtype=np.float32)

        loop = int(1e3)
        result = timeit.timeit(
            "fast_max(A)",
            setup=lambda: fast_max(A),
            globals=globals(),
            number=loop,
        )
        print(f"fast_max      : {result / loop * 1000:.5f} ms per iter")

    """
    if 1:
        test_shape = 4096
        in1 = np.random.randn(test_shape)
        in2 = np.random.randn(test_shape)
        in1 = np.asarray(in1, dtype=np.float32)
        in2 = np.asarray(in2, dtype=np.float32)

        loop = int(1e3)
        result = timeit.timeit(
            "fast_magnitude(in1, in2)",
            setup=lambda: fast_magnitude(in1, in2),
            globals=globals(),
            number=loop,
        )
        print(f"fast_magnitude: {result / loop * 1000:.5f} ms per iter")

        loop = int(1e3)
        result = timeit.timeit(
            "np.sqrt(np.square(in1) + np.square(in2))",
            setup=lambda: np.sqrt(np.square(in1) + np.square(in2)),
            globals=globals(),
            number=loop,
        )
        print(f"np_magnitude  : {result / loop * 1000:.5f} ms per iter")
    """
