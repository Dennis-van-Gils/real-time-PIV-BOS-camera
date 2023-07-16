#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "16-07-2023"
__version__ = "1.0"

import numpy as np
from numba import jit, njit, prange

# ------------------------------------------------------------------------------
#   remove_mean_background
# ------------------------------------------------------------------------------


# Do not supply a function signature to @njit. Timeit shows best operation
# without.
@njit(
    parallel=True,
    cache=True,
    nogil=True,
)
def remove_mean_background(img: np.ndarray):
    """In-place operation on `img`.
    Many times faster with numba than vanilla numpy.

    Vanilla numpy:
      img = np.clip(img - np.mean(img), 0, None).astype(img.dtype)

    BENCHMARK on computer `Onera`:
        4096 x 4096 @ 16 bit:
            vanilla numpy    : 112   ms per iter
            numba no parallel:  14   ms per iter
            numba parallel   :   2.5 ms per iter

        1024 x 1024 @ 16 bit:
            vanilla numpy    :   6   ms per iter
            numba no parallel:   0.8 ms per iter
            numba parallel   :   0.2 ms per iter
    """

    # We are expecting at max a 16-bit grayscale image
    mu = np.asarray(np.ceil(np.mean(img)), dtype=np.uint16)

    for y in prange(img.shape[0]):
        for x in prange(img.shape[1]):
            if img[y, x] < mu:
                img[y, x] = 0
            else:
                img[y, x] = img[y, x] - mu


# ------------------------------------------------------------------------------
#   create_IW_grid
# ------------------------------------------------------------------------------


@njit(
    cache=True,
    nogil=True,
)
def meshgrid_numba(x, y):
    """Numba-accelerated version of `np.meshgrid()` using `xy` indexing."""
    n = len(x)
    m = len(y)
    xx = np.empty((m, n), dtype=x.dtype)
    yy = np.empty((m, n), dtype=y.dtype)
    for i in range(n):
        xx[:, i] = x[i]
    for j in range(m):
        yy[j, :] = y[j]

    return xx, yy


# Can't `@njit`, because `np.tile()`, `np.repeat(..., axis=0)` and
# `np.swapaxes()` are not supported.
def create_IW_grid(
    img_w: int, img_h: int, IW_size: int, IW_overlap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Divide up the source image area given by `img_w` and `img_h` into square
    interrogation windows (IW) each of size `IW_size`.

    Args:
        img_w (``int``):
            Width of the source image [px].

        img_h (``int``):
            Height of the source image [px].

        IW_size (``int``):
            Square interrogation window size [px].

        IW_overlap (``float``, optional):
            Window overlap fraction [0 - 1].
            0  : no window overlap
            0.5: 50% window overlap

            Default: 0.5

    Returns (``tuple``):
        grid_x (``np.ndarray(int)``):
            1D linearized meshgrid containing the x-pixel positions of the IW
            centers [px].
            Array shape: [N_IWs]

        grid_y (``np.ndarray(int)``):
            1D linearized meshgrid containing the y-pixel positions of the IW
            centers [px].
            Array shape: [N_IWs]

        lims_x (``np.ndarray(int)``):
            2D array containing the x-pixel limits of each IW [px].
            (:, 0): limit start
            (:, 1): limit end
            Array shape: [N_IWs, 2]

        lims_y (``np.ndarray(int)``):
            2D array containing the y-pixel limits of each IW [px].
            (:, 0): limit start
            (:, 1): limit end
            Array shape: [N_IWs, 2]

        N_IWs (``int``):
            Obtained total number of interrogation windows.

        N_IWs_x (``int``):
            Obtained number of interrogation windows along the x-axis.

        N_IWs_y (``int``):
            Obtained number of interrogation windows along the y-axis.
    """

    # Number of IWs that will fit in the source image
    N_IWs_x = int((img_w - IW_size) // (IW_size * (1 - IW_overlap))) + 1
    N_IWs_y = int((img_h - IW_size) // (IW_size * (1 - IW_overlap))) + 1
    N_IWs = N_IWs_x * N_IWs_y

    # IW center positions
    half_IW_size = IW_size // 2
    arr_x = np.arange(N_IWs_x) * (1 - IW_overlap) * IW_size + half_IW_size
    arr_y = np.arange(N_IWs_y) * (1 - IW_overlap) * IW_size + half_IW_size
    arr_x = np.asarray(arr_x, dtype=int)
    arr_y = np.asarray(arr_y, dtype=int)

    if 1:
        # Numba accelerated, faster than `np.meshgrid()`
        grid_x, grid_y = meshgrid_numba(arr_x, arr_y)
    else:
        # Native numpy
        grid_x, grid_y = np.meshgrid(arr_x, arr_y)

    # IW limits
    lims_x = np.column_stack((arr_x - half_IW_size, arr_x + half_IW_size - 1))
    lims_y = np.column_stack((arr_y - half_IW_size, arr_y + half_IW_size - 1))
    lims_x = np.tile(lims_x, (N_IWs_y, 1, 1))
    lims_y = np.tile(lims_y, (N_IWs_x, 1, 1)).swapaxes(0, 1)

    # After-thought: We linearize the matrices again. Using linearized matrices
    # instead of multi-dim matrixes speeds up the code when iterating over all
    # elements.
    # TODO: Adjust above code to linearize from the very beginning. Not crucial
    # though.
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)
    lims_x = lims_x.reshape(N_IWs, 2)
    lims_y = lims_y.reshape(N_IWs, 2)

    return grid_x, grid_y, lims_x, lims_y, N_IWs, N_IWs_x, N_IWs_y


# ------------------------------------------------------------------------------
#   lookup_IW_idx
# ------------------------------------------------------------------------------


@njit(
    "(uint32)(uint32, uint32, Tuple((uint32, float64, uint32, uint32, uint32)))",
    cache=True,
    nogil=True,
)
def lookup_IW_idx(
    px_x: int, px_y: int, IW_params: tuple[int, float, int, int, int]
) -> int:
    """Look up and return the index of the IW, as generated by the parameters
    passed by `IW_params`, that has its center closest to the passed pixel
    position [`px_x`, `px_y`].

    Returns (``int``): Index of the IW.
    """
    (IW_size, IW_overlap, N_IWs, N_IWs_x, N_IWs_y) = IW_params
    IW_idx_x = int((px_x - IW_size // 2) / (IW_size * (1 - IW_overlap)) + 0.5)
    IW_idx_y = int((px_y - IW_size // 2) / (IW_size * (1 - IW_overlap)) + 0.5)
    IW_idx_x = np.minimum(IW_idx_x, N_IWs_x - 1)
    IW_idx_y = np.minimum(IW_idx_y, N_IWs_y - 1)

    # Linearize index, C-style
    IW_idx = IW_idx_y * N_IWs_x + IW_idx_x

    return IW_idx


# ------------------------------------------------------------------------------
#   fliplrud
# ------------------------------------------------------------------------------


@njit(
    parallel=False,  # Can't parallelize
    cache=True,
    nogil=True,
)
def fliplrud(img):
    """Flip the passed image left-to-right and up-to-down, necessary to use an
    FFT convolution as a 2D-correlation.
    """
    # Vanilla numpy:
    #   return np.flipud(np.fliplr(img))
    img = img[:, ::-1]  # fliplr
    img = img[::-1, ...]  # flipud

    return img


# ------------------------------------------------------------------------------
#   subpx_3pgf_2D
# ------------------------------------------------------------------------------


@njit(
    "Tuple((float64, float64))(float64[:, :], uint32, uint32)",
    cache=True,
    nogil=True,
)
def subpx_3pgf_2D(C: np.ndarray, px_x: int, px_y: int) -> tuple[float, float]:
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
#   normalize_C_maps
# ------------------------------------------------------------------------------


@njit(
    "(float64[:, :, :],)",
    parallel=True,
    cache=True,
    nogil=True,
)
def normalize_C_maps(C_maps: np.ndarray):
    """In-place operation on `C_maps`.
    NOTE: It is not mandatory to normalize the correlations maps for the peak
    finding algorithm to work correctly. Normalizing wastes cpu time.
    """
    for IW_idx in prange(C_maps.shape[0]):
        C = C_maps[IW_idx, :, :]
        if np.isnan(C[0, 0]):
            # Nothing sensible to normalize
            continue

        C_maps[IW_idx, :, :] = np.divide(C, np.max(C))


# ------------------------------------------------------------------------------
#   compute_displacement_vectors_from_C_maps
# ------------------------------------------------------------------------------


@njit(
    "(float64[:, :, :], float64[:], float64[:], float64[:], float64[:], boolean)",
    parallel=True,
    cache=True,
    nogil=True,
)
def compute_displacement_vectors_from_C_maps(
    C_maps: np.ndarray,
    IW_shifts_x: np.ndarray,
    IW_shifts_y: np.ndarray,
    VM_dx: np.ndarray,
    VM_dy: np.ndarray,
    perform_subpixel_fitting: bool = False,
):
    """In-place operation on `VM_dx` and `VM_dy`.
    NOTE: The passed correlation maps do not have to be normalized for the peak
    finding algorithm to work correctly. Normalizing wastes cpu time."""
    for IW_idx in prange(C_maps.shape[0]):
        C = C_maps[IW_idx, :, :]

        if np.isnan(C[0, 0]):
            dx = np.nan
            dy = np.nan
        else:
            # It is not necessary to normalize the correlation maps. Adds
            # overhead.
            # C = np.divide(C, np.max(C))  # Not necessary
            # Store back into C_maps. Adds another overhead.
            # C_maps[IW_idx_y, IW_idx_x, :, :] = C

            # Find maximum correlation peak
            iMaxC = int(np.argmax(C))
            peak_x = iMaxC % C.shape[1]  # unravel index
            peak_y = iMaxC // C.shape[1]  # unravel index

            if perform_subpixel_fitting:
                # Sub-pixel resolution algorithm, 3-point Gaussian fit
                peak_x, peak_y = subpx_3pgf_2D(C, peak_x, peak_y)

            # Calculate displacement vector
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
        # img = np.random.randint(0, 255, (1024, 1024), dtype=np.uint8)
        img = np.random.randint(0, 255, (4096, 4096), dtype=np.uint16)

        loop = int(1e2)
        result = timeit.timeit(
            "remove_mean_background(img)",
            setup=lambda: remove_mean_background(img),
            globals=globals(),
            number=loop,
        )
        print(f"remove_mean_bg: {result / loop * 1000:.5f} ms per iter")

    if 1:
        loop = int(1e3)
        result = timeit.timeit(
            "create_IW_grid(1024, 1024, 64, 0.5)",
            setup=lambda: create_IW_grid(1024, 1024, 64, 0.5),
            globals=globals(),
            number=loop,
        )
        print(f"create_IW_grid: {result / loop * 1000:.5f} ms per iter")

    if 1:
        IW_params = (64, 0.5, 3600, 60, 60)

        loop = int(1e6)
        result = timeit.timeit(
            "lookup_IW_idx(1024, 1024, IW_params)",
            setup=lambda: lookup_IW_idx(1024, 1024, IW_params),
            globals=globals(),
            number=loop,
        )
        print(f"lookup_IW_idx : {result / loop * 1000:.5f} ms per iter")

    if 1:
        img = np.random.randint(0, 255, (1024, 1024), dtype=np.uint16)

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
        C = fftconvolve(B, A, mode="full")
        C = C / np.max(C)

        loop = int(1e6)
        result = timeit.timeit(
            "subpx_3pgf_2D(C, 32, 32)",
            setup=lambda: subpx_3pgf_2D(C, 32, 32),
            globals=globals(),
            number=loop,
        )
        print(f"subpx_3pgf_2D : {result / loop * 1000:.5f} ms per iter")
