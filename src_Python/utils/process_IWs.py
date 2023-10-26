#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOTE

In this module I have experimented with the following numba-jitted class:
    @nb.experimental.jitclass
    class IW_Mesh:
        ...

Pro:
    It works great and improves the readability of the code, because a single
    object `IW_mesh` can be passed around, instead of many individual arguments
    such as arrays `A_grid_x`, `A_lims_x`, etc.

Con:
    The numba compile time is huge. This is because caching can not be done for
    numba-jitted classes. Also, jitted functions that take a jitclass as
    argument can no longer cache. Here, caching fails for `process_IWs()` and
    `obtain_IWs_from_image()` when using the jitted `IW_Mesh` class as argument.

    Our compile time went from 2.5 sec to 7.8 sec, every runtime :(

    See https://github.com/numba/numba/issues/4830#issuecomment-896819248

Taken solution:
    I kept the `IW_Mesh` class, but removed the `jitclass` decorator.

    Also, I have commented out full code blocks on `process_IWs()` and
    `obtain_IWs_from_image()` that were intended to be used by the numba-jitted
    `IW_Mesh`. They got substituted by the more lengthy functions that take the
    individual array members as arguments.

    Commented-out code blocks cary the tag `COMMENTED OUT: JITCLASSED IW_MESH`
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "24-10-2023"
__version__ = "1.0"

import numpy as np
import numpy.typing as npt
import numba as nb

from utils.my_fun import all_smaller_or_equal_to
import init_config as cfg

if cfg.FFT_LIB == cfg.FFT_LIBS.PYFFTW:
    from utils.dvg_fftconvolver_pyfftw import FFT_Convolver2D_Full
elif cfg.FFT_LIB == cfg.FFT_LIBS.ROCKETFFT:
    from utils.dvg_fftconvolver_rocketfft import FFT_Convolver2D_Full
elif cfg.FFT_LIB == cfg.FFT_LIBS.SCIPY:
    from utils.dvg_fftconvolver_scipy import FFT_Convolver2D_Full
else:
    from utils.dvg_fftconvolver_rocketfft import FFT_Convolver2D_Full


def conditional_decorator(dec, condition):
    def decorator(func):
        return dec(func) if condition else func

    return decorator


# If multigrid IW analysis is enabled, the IW pre-shift in pixels obtained from
# the larger parent grid gets multiplied with this factor.
#
# Naively, one would expect a value of 1.0 to work correctly. However, it is
# possible that the pre-shift obtained from the larger grid moves one or more of
# its smaller children IWs too far away from the original parent location. This
# is troublesome when a large gradient exists inside the parent grid IW (e.g. a
# large part of the window shifts by a large amount and a minor region shifts
# hardly). If so, then the subsequent child IW of the 'quiescent' region would
# not be able to undo the pre-shift: The IW window might have moved too far away
# such that the true correlation peak has moved out of view. Being unable to
# "find it's way back home again" would result in blocky patches of incorrect
# and too large vectors. Setting below factor to, for instance, 0.8 helps the
# child IWs to find their way back home again.
#
# After reading multiple articles: This pre-shift attenuation < 1.0 approach
# seems to be the poor man's approach to the otherwise recommended way of
# increasing the cross-correlation search area of frame B past its set IW size.
# We can't do that here, because we have numba-optimized the cross-correlation
# function to always expect equal-sized input arrays.
PRESHIFT_ATTENUATION = 0.8

# ------------------------------------------------------------------------------
#   IW_Mesh
# ------------------------------------------------------------------------------


""" COMMENTED OUT: JITCLASSED IW_MESH
@nb.experimental.jitclass
"""


class IW_Mesh:
    """Manages the interrogation window (IW) meshes for frames A and B. It
    divides up the source image area given by `img_w` and `img_h` into square
    interrogation windows each of size `IW_size` using the specified overlap
    fraction `IW_overlap`.

    Args:
        img_w (``int``):
            Width of the source image [px].

        img_h (``int``):
            Height of the source image [px].

        IW_size (``int``):
            Square interrogation window size [px].

        IW_overlap (``float``):
            Window overlap fraction [0 - 1].
            0  : no window overlap
            0.5: 50% window overlap

    Members:
        grid_x (``np.ndarray(np.int32)``):
            1D linearized meshgrid containing the x-pixel positions of the IW
            centers [px].
            Array shape: (N_IWs, )

        grid_y (``np.ndarray(np.int32)``):
            1D linearized meshgrid containing the y-pixel positions of the IW
            centers [px].
            Array shape: (N_IWs, )

        lims_x (``np.ndarray(np.int32)``):
            2D array containing the x-pixel limits of each IW [px].
            (:, 0): limit start
            (:, 1): limit end
            Array shape: (N_IWs, 2)

        lims_y (``np.ndarray(np.int32)``):
            2D array containing the y-pixel limits of each IW [px].
            (:, 0): limit start
            (:, 1): limit end
            Array shape: (N_IWs, 2)

        N_IWs (``int``):
            Obtained total number of interrogation windows.

        N_IWs_x (``int``):
            Obtained number of interrogation windows along the x-axis.

        N_IWs_y (``int``):
            Obtained number of interrogation windows along the y-axis.

        params (``tuple(int, float, int, int, int)``):
            Convenience member, combining the following into one tuple:
                (IW_size    (``int``),
                 IW_overlap (``float``),
                 N_IWs      (``int``),
                 N_IWs_x    (``int``),
                 N_IWs_y    (``int``))
    """

    """ COMMENTED OUT: JITCLASSED IW_MESH
    # fmt: off
    # Needed for jitclass signature
    img_w        : nb.int32             # type: ignore
    img_h        : nb.int32             # type: ignore
    IW_size      : nb.int32             # type: ignore
    IW_overlap   : nb.float32           # type: ignore
    _half_IW_size: nb.int32             # type: ignore
    _overlap_px  : nb.float32           # type: ignore
    N_IWs        : nb.int32             # type: ignore
    N_IWs_x      : nb.int32             # type: ignore
    N_IWs_y      : nb.int32             # type: ignore
    IW_params    : nb.types.Tuple((
                        nb.int32,
                        nb.float32,
                        nb.int32,
                        nb.int32,
                        nb.int32))      # type: ignore
    orig_grid_x  : nb.int32[:]          # type: ignore
    orig_grid_y  : nb.int32[:]          # type: ignore
    orig_lims_x  : nb.int32[:, ::1]     # type: ignore
    orig_lims_y  : nb.int32[:, ::1]     # type: ignore
    A_grid_x     : nb.int32[:]          # type: ignore
    A_grid_y     : nb.int32[:]          # type: ignore
    A_lims_x     : nb.int32[:, ::1]     # type: ignore
    A_lims_y     : nb.int32[:, ::1]     # type: ignore
    B_grid_x     : nb.int32[:]          # type: ignore
    B_grid_y     : nb.int32[:]          # type: ignore
    B_lims_x     : nb.int32[:, ::1]     # type: ignore
    B_lims_y     : nb.int32[:, ::1]     # type: ignore
    # fmt: on
    """

    def __init__(
        self,
        img_w: int,
        img_h: int,
        IW_size: int,
        IW_overlap: float,
    ):
        self.img_w = img_w
        self.img_h = img_h
        self.IW_size = IW_size
        self.IW_overlap = IW_overlap

        # Example:
        #   img_w      = 128
        #   img_h      = 64
        #   IW_size    = 32
        #   IW_overlap = 0

        half_IW_size = int(IW_size // 2)
        overlap_px = (1 - IW_overlap) * IW_size

        # Number of IWs that will fit in the source image
        N_IWs_x = int((img_w - IW_size) // overlap_px + 1)
        N_IWs_y = int((img_h - IW_size) // overlap_px + 1)
        N_IWs = N_IWs_x * N_IWs_y
        # Example: (N_IWs, N_IWs_x, N_IWs_y) = (8, 4, 2)

        # IW center positions
        arr_x = np.arange(N_IWs_x) * overlap_px + half_IW_size
        arr_y = np.arange(N_IWs_y) * overlap_px + half_IW_size
        arr_x = np.asarray(arr_x, dtype=np.int32)
        arr_y = np.asarray(arr_y, dtype=np.int32)
        # Example: arr_x  = [ 16  48  80 112]
        # Example: arr_y  = [ 16  48]

        # Create mesh grid as a linearized arrays
        grid_x = np.empty(N_IWs, dtype=np.int32)
        for i in np.arange(N_IWs_y):
            grid_x[i * N_IWs_x : (i + 1) * N_IWs_x] = arr_x
        grid_y = np.repeat(arr_y, N_IWs_x)
        # Example: grid_x = [ 16  48  80 112  16  48  80 112]
        # Example: grid_y = [ 16  16  16  16  48  48  48  48]

        # IW limits
        lims_x = np.empty((N_IWs, 2), dtype=np.int32)
        lims_y = np.empty((N_IWs, 2), dtype=np.int32)
        lims_x[:, 0] = grid_x - half_IW_size
        lims_x[:, 1] = grid_x + half_IW_size - 1
        lims_y[:, 0] = grid_y - half_IW_size
        lims_y[:, 1] = grid_y + half_IW_size - 1
        # Example: lims_x = [[  0  31]      lims_y = [[ 0 31]
        #                    [ 32  63]                [ 0 31]
        #                    [ 64  95]                [ 0 31]
        #                    [ 96 127]                [ 0 31]
        #                    [  0  31]                [32 63]
        #                    [ 32  63]                [32 63]
        #                    [ 64  95]                [32 63]
        #                    [ 96 127]]               [32 63]]

        self._half_IW_size = half_IW_size
        self._overlap_px = overlap_px

        self.N_IWs = N_IWs
        self.N_IWs_x = N_IWs_x
        self.N_IWs_y = N_IWs_y
        self.IW_params = (IW_size, IW_overlap, N_IWs, N_IWs_x, N_IWs_y)

        self.orig_grid_x = grid_x
        self.orig_grid_y = grid_y
        self.orig_lims_x = lims_x
        self.orig_lims_y = lims_y

        self.A_grid_x = np.empty_like(grid_x)
        self.A_grid_y = np.empty_like(grid_y)
        self.A_lims_x = np.empty_like(lims_x)
        self.A_lims_y = np.empty_like(lims_y)

        self.B_grid_x = np.empty_like(grid_x)
        self.B_grid_y = np.empty_like(grid_y)
        self.B_lims_x = np.empty_like(lims_x)
        self.B_lims_y = np.empty_like(lims_y)

        self.reset_A()
        self.reset_B()

    def reset_A(self):
        self.A_grid_x[:] = self.orig_grid_x[:]
        self.A_grid_y[:] = self.orig_grid_y[:]
        self.A_lims_x[:] = self.orig_lims_x[:]
        self.A_lims_y[:] = self.orig_lims_y[:]

    def reset_B(self):
        self.B_grid_x[:] = self.orig_grid_x[:]
        self.B_grid_y[:] = self.orig_grid_y[:]
        self.B_lims_x[:] = self.orig_lims_x[:]
        self.B_lims_y[:] = self.orig_lims_y[:]

    def lookup_IW_idx(self, px_x: int, px_y: int) -> int:
        """Look up and return the index of the IW that has its center closest to
        the passed pixel position [`px_x`, `px_y`].

        Returns (``int``): Index of the IW.
        """
        IW_idx_x = int((px_x - self._half_IW_size) / self._overlap_px + 0.5)
        IW_idx_y = int((px_y - self._half_IW_size) / self._overlap_px + 0.5)
        IW_idx_x = np.minimum(IW_idx_x, self.N_IWs_x - 1)
        IW_idx_y = np.minimum(IW_idx_y, self.N_IWs_y - 1)

        # Linearize index, C-style
        IW_idx = IW_idx_y * self.N_IWs_x + IW_idx_x

        return IW_idx


# ------------------------------------------------------------------------------
#   lookup_IW_idx
# ------------------------------------------------------------------------------


@nb.njit(
    "(int32)(int32, int32, Tuple((int32, float32, int32, int32, int32)))",
    cache=True,
    nogil=True,
)
def lookup_IW_idx(
    px_x: int,
    px_y: int,
    IW_params: tuple[int, float, int, int, int],
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
#   obtain_IWs_from_image
# ------------------------------------------------------------------------------


@nb.njit(
    "Tuple((float32[:, :], float32[:, :])) \
        (int32, int32, \
        float32[:, ::1], float32[:, ::1], \
        Tuple((int32, float32, int32, int32, int32)), \
        float32[::1], \
        float32[::1], \
        int32[::1], \
        int32[::1], \
        int32[:, ::1], \
        int32[:, ::1], \
        int32[::1], \
        int32[::1], \
        int32[:, ::1], \
        int32[:, ::1], \
        int32[::1], \
        int32[::1])",
    cache=True,
    nogil=True,
)
def obtain_IWs_from_image(
    # fmt: off
    IW_idx     : int,
    stage_idx  : int,
    A_         : npt.NDArray[np.float32],  # read-only
    B          : npt.NDArray[np.float32],  # read-only
    prev_IW_params: tuple[int, float, int, int, int],
                                           # read-only
    prev_VM_dx : npt.NDArray[np.float32],  # read-only
    prev_VM_dy : npt.NDArray[np.float32],  # read-only
    A_IW_grid_x: npt.NDArray[np.int32],    # read-only
    A_IW_grid_y: npt.NDArray[np.int32],    # read-only
    A_IW_lims_x: npt.NDArray[np.int32],    # read-only
    A_IW_lims_y: npt.NDArray[np.int32],    # read-only
    B_IW_grid_x: npt.NDArray[np.int32],    # in-place operation, debug output
    B_IW_grid_y: npt.NDArray[np.int32],    # in-place operation, debug output
    B_IW_lims_x: npt.NDArray[np.int32],    # in-place operation, debug output
    B_IW_lims_y: npt.NDArray[np.int32],    # in-place operation, debug output
    IW_shifts_x: npt.NDArray[np.int32],    # in-place operation, debug output
    IW_shifts_y: npt.NDArray[np.int32],  # in-place operation, debug output
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    # fmt: on
    """In-place operation on:
    * `B_IW_grid_x`
    * `B_IW_grid_y`
    * `B_IW_lims_x`
    * `B_IW_lims_y`
    * `IW_shifts_x`
    * `IW_shifts_y`

    Return the interrogation window images `IW_A_` and `IW_B` obtained from the
    source images `A_` and `B` using window pre-shifts when available from the
    previous multigrid stage.

    NOTE: The returned `IW_A_` and `IW_B` are not garantueed contiguous, because
    of the slicing within `A_` and `B`. We don't need contiguity here as `IW_A_`
    and `_IW_B` will get copied into C-contiguous arrays later on when the
    `fftw.convolve()` function is run on them. Enforcing contiguity here would
    only waste cpu time.
    """
    # --------------------------------------------------------------------------
    #   Calculate IW of frame B
    #   Apply window shifting technique
    # --------------------------------------------------------------------------

    # Part of the window shifting mechanism:
    # Undo the shift again when the shifted IW of frame B is leaving the borders
    # of frame B. If so, we will, later on, zero out the appropiate section of
    # the IW of frame B that corresponds to `particles` that are definitely not
    # present in the IW of frame A. Likewise, we will zero out pixels in frame A
    # that are not present in frame B.
    zero_out_L = 0  # left of B , x = 0
    zero_out_R = 0  # right of B, x = IW_size - 1
    zero_out_U = 0  # up of B   , y = 0
    zero_out_D = 0  # down of B , y = IW_size - 1
    IW_needs_to_be_a_copy = False

    # Check for window pre-shift
    if stage_idx == 0:
        shift_x = np.int32(0)  # [px]
        shift_y = np.int32(0)  # [px]

    else:
        # Pre-shift available: Look up corresponding index of the IW in the
        # larger parent grid
        parent_IW_idx = lookup_IW_idx(
            A_IW_grid_x[IW_idx],
            A_IW_grid_y[IW_idx],
            prev_IW_params,
        )

        # Retrieve the pre-shift
        shift_x = prev_VM_dx[parent_IW_idx]
        shift_y = prev_VM_dy[parent_IW_idx]
        shift_x = np.int32(
            0 if np.isnan(shift_x) else shift_x * PRESHIFT_ATTENUATION
        )
        shift_y = np.int32(
            0 if np.isnan(shift_y) else shift_y * PRESHIFT_ATTENUATION
        )

        # Apply the pre-shift to IW B (eager)
        B_IW_grid_x[IW_idx] += shift_x
        B_IW_grid_y[IW_idx] += shift_y
        B_IW_lims_x[IW_idx, :] += shift_x
        B_IW_lims_y[IW_idx, :] += shift_y

        # Check and prevent the shift of IW B from moving outside of the source
        # image B. When so, we will zero out part of the IW images that have
        # moved out-of-frame, later on.
        if B_IW_lims_x[IW_idx, 0] < 0:
            IW_needs_to_be_a_copy = True
            zero_out_R = np.abs(shift_x)
            B_IW_grid_x[IW_idx] -= shift_x
            B_IW_lims_x[IW_idx, :] -= shift_x
            shift_x = np.int32(0)

        if B_IW_lims_x[IW_idx, 1] > B.shape[1] - 1:
            IW_needs_to_be_a_copy = True
            zero_out_L = np.abs(shift_x)
            B_IW_grid_x[IW_idx] -= shift_x
            B_IW_lims_x[IW_idx, :] -= shift_x
            shift_x = np.int32(0)

        if B_IW_lims_y[IW_idx, 0] < 0:
            IW_needs_to_be_a_copy = True
            zero_out_D = np.abs(shift_y)
            B_IW_grid_y[IW_idx] -= shift_y
            B_IW_lims_y[IW_idx, :] -= shift_y
            shift_y = np.int32(0)

        if B_IW_lims_y[IW_idx, 1] > B.shape[0] - 1:
            IW_needs_to_be_a_copy = True
            zero_out_U = np.abs(shift_y)
            B_IW_grid_y[IW_idx] -= shift_y
            B_IW_lims_y[IW_idx, :] -= shift_y
            shift_y = np.int32(0)

        # Store
        IW_shifts_x[IW_idx] = shift_x
        IW_shifts_y[IW_idx] = shift_y

    # --------------------------------------------------------------------------
    #   Obtain images of IW frame A and IW frame B
    # --------------------------------------------------------------------------

    # Note: `A_` is a flipped left-to-right and up-to-down version of `A`, so we
    # have to flip the indices as well, hence the use of `A.shape[] - ...`.
    A_slice = (
        slice(
            A_.shape[0] - A_IW_lims_y[IW_idx, 1] - 1,
            A_.shape[0] - A_IW_lims_y[IW_idx, 0],
        ),
        slice(
            A_.shape[1] - A_IW_lims_x[IW_idx, 1] - 1,
            A_.shape[1] - A_IW_lims_x[IW_idx, 0],
        ),
    )
    B_slice = (
        slice(B_IW_lims_y[IW_idx, 0], B_IW_lims_y[IW_idx, 1] + 1),
        slice(B_IW_lims_x[IW_idx, 0], B_IW_lims_x[IW_idx, 1] + 1),
    )

    # fmt: off
    if IW_needs_to_be_a_copy:
        # We need a copy, because otherwise the upcoming zeroing of the IW image
        # borders will affect, by means of reference, the original image and
        # interfere with the correlation of upcoming and overlapping IWs.
        # Copying adds a tiny cpu overhead.
        IW_A_ = np.copy(A_[A_slice])  # C-contiguous
        IW_B  = np.copy(B [B_slice])  # C-contiguous

        # Zero out the appropiate section of the IW of frame B that corresponds
        # to `particles` that are definitely not present in the IW of frame A.
        # Likewise, zero out the IW of frame A. Zero caries the meaning of being
        # at the mean background level of the image.
        if zero_out_L > 0:
            IW_A_[:, :zero_out_L] = 0
            IW_B [:, :zero_out_L] = 0
        if zero_out_R > 0:
            IW_A_[:, -zero_out_R:] = 0
            IW_B [:, -zero_out_R:] = 0
        if zero_out_U > 0:
            IW_A_[:zero_out_U, :] = 0
            IW_B [:zero_out_U, :] = 0
        if zero_out_D > 0:
            IW_A_[-zero_out_D:, :] = 0
            IW_B [-zero_out_D:, :] = 0
    else:
        IW_A_ = A_[A_slice]  # Pass by reference, not contiguous
        IW_B  = B [B_slice]  # Pass by reference, not contiguous
    # fmt: on

    return IW_A_, IW_B


# ------------------------------------------------------------------------------
#   process_IWs
# ------------------------------------------------------------------------------


@conditional_decorator(
    nb.njit(
        cache=True,
        nogil=True,
    ),
    cfg.FFT_LIB in (cfg.FFT_LIBS.ROCKETFFT, cfg.FFT_LIBS.SCIPY),
)
def process_IWs(
    # fmt: off
    stage_idx  : int,
    A_         : npt.NDArray[np.float32],  # read-only
    B          : npt.NDArray[np.float32],  # read-only
    prev_IW_params: tuple[int, float, int, int, int],
                                           # read-only
    prev_VM_dx : npt.NDArray[np.float32],  # read-only
    prev_VM_dy : npt.NDArray[np.float32],  # read-only
    A_IW_grid_x: npt.NDArray[np.int32],    # read-only
    A_IW_grid_y: npt.NDArray[np.int32],    # read-only
    A_IW_lims_x: npt.NDArray[np.int32],    # read-only
    A_IW_lims_y: npt.NDArray[np.int32],    # read-only
    B_IW_grid_x: npt.NDArray[np.int32],    # in-place operation, debug output
    B_IW_grid_y: npt.NDArray[np.int32],    # in-place operation, debug output
    B_IW_lims_x: npt.NDArray[np.int32],    # in-place operation, debug output
    B_IW_lims_y: npt.NDArray[np.int32],    # in-place operation, debug output
    IW_shifts_x: npt.NDArray[np.int32],    # in-place operation, debug output
    IW_shifts_y: npt.NDArray[np.int32],    # in-place operation, debug output
    C_maps     : npt.NDArray[np.float32],  # in-place output
    fft        : FFT_Convolver2D_Full,
    IWs_slice  : slice = slice(None),
):
    # fmt: on
    """In-place operation on:
    * `B_IW_grid_x`
    * `B_IW_grid_y`
    * `B_IW_lims_x`
    * `B_IW_lims_y`
    * `IW_shifts_x`
    * `IW_shifts_y`
    * `C_maps`

    Walk over all interrogation windows (IW) to be taken from source images `A_`
    and `B` using window pre-shifts when available from the previous multigrid
    stage, and compute the 2D correlation maps.

    The last argument `IWs_slice` can be set to only process a certain slice out
    of all the IWs. This is useful to distribute the calculation of all IWs over
    multiple concurrent tasks. Defaults to all IWs when omitted.
    """

    idx_start = 0 if IWs_slice.start is None else IWs_slice.start
    idx_stop = C_maps.shape[0] if IWs_slice.stop is None else IWs_slice.stop
    idx_step = 1 if IWs_slice.step is None else IWs_slice.step

    for IW_idx in range(idx_start, idx_stop, idx_step):
        IW_A_, IW_B = obtain_IWs_from_image(
            IW_idx,
            stage_idx,
            A_,
            B,
            prev_IW_params,
            prev_VM_dx,
            prev_VM_dy,
            A_IW_grid_x,
            A_IW_grid_y,
            A_IW_lims_x,
            A_IW_lims_y,
            B_IW_grid_x,
            B_IW_grid_y,
            B_IW_lims_x,
            B_IW_lims_y,
            IW_shifts_x,
            IW_shifts_y,
        )

        # BOS
        if cfg.MODE == cfg.MODES.BOS:
            C_maps[IW_idx, :, :] = fft.convolve(IW_B, IW_A_)
            continue

        # PIV & PIV2
        if all_smaller_or_equal_to(IW_A_, 0) or all_smaller_or_equal_to(
            IW_B, 0
        ):
            # No details are present in the IW images. All pixels are below
            # or at the mean background --> Save computation time.
            # TODO: Make this a user config threshold? Is <= 0 even correct?
            # Must match up with 'zeroing out' mechanism of function
            # `obtain_IWs_from_image()`.
            C_maps[IW_idx, 0, 0] = np.nan

        else:
            # Perform 2D cross-correlation
            C_maps[IW_idx, :, :] = fft.convolve(IW_B, IW_A_)


''' COMMENTED OUT: JITCLASSED IW_MESH
@nb.njit(
    (nb.types.UniTuple(nb.float32[:, :], 2))(
        nb.int32,
        nb.int32,
        nb.float32[:, ::1],
        nb.float32[:, ::1],
        IW_Mesh.class_type.instance_type,  # type: ignore
        nb.float32[::1],
        nb.float32[::1],
        IW_Mesh.class_type.instance_type,  # type: ignore
        nb.int32[::1],
        nb.int32[::1],
    ),
    cache=True,
    nogil=True,
)
def obtain_IWs_from_image(
    # fmt: off
    IW_idx      : int,
    stage_idx   : int,
    A_          : npt.NDArray[np.float32],  # read-only
    B           : npt.NDArray[np.float32],  # read-only
    prev_IW_mesh: IW_Mesh,                  # read-only
    prev_VM_dx  : npt.NDArray[np.float32],  # read-only
    prev_VM_dy  : npt.NDArray[np.float32],  # read-only
    IW_mesh     : IW_Mesh,                  # in-place operation on B, read-only on A
    IW_shifts_x : npt.NDArray[np.int32],    # in-place operation, debug output
    IW_shifts_y : npt.NDArray[np.int32],  # in-place operation, debug output
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    # fmt: on
    """In-place operation on:
    * `IW_mesh.B_grid_x`
    * `IW_mesh.B_grid_y`
    * `IW_mesh.B_lims_x`
    * `IW_mesh.B_lims_y`
    * `IW_shifts_x`
    * `IW_shifts_y`

    Return the interrogation window images `IW_A_` and `IW_B` obtained from the
    source images `A_` and `B` using window pre-shifts when available from the
    previous multigrid stage.

    NOTE: The returned `IW_A_` and `IW_B` are not garantueed contiguous, because
    of the slicing within `A_` and `B`. We don't need contiguity here as `IW_A_`
    and `_IW_B` will get copied into C-contiguous arrays later on when the
    `fftw.convolve()` function is run on them. Enforcing contiguity here would
    only waste cpu time.
    """
    # --------------------------------------------------------------------------
    #   Calculate IW of frame B
    #   Apply window shifting technique
    # --------------------------------------------------------------------------

    # Part of the window shifting mechanism:
    # Undo the shift again when the shifted IW of frame B is leaving the borders
    # of frame B. If so, we will, later on, zero out the appropiate section of
    # the IW of frame B that corresponds to `particles` that are definitely not
    # present in the IW of frame A. Likewise, we will zero out pixels in frame A
    # that are not present in frame B.
    zero_out_L = 0  # left of B , x = 0
    zero_out_R = 0  # right of B, x = IW_size - 1
    zero_out_U = 0  # up of B   , y = 0
    zero_out_D = 0  # down of B , y = IW_size - 1
    IW_needs_to_be_a_copy = False

    # Check for window pre-shift
    if stage_idx == 0:
        shift_x = np.int32(0)  # [px]
        shift_y = np.int32(0)  # [px]

    else:
        # Pre-shift available: Look up corresponding index of the IW in the
        # larger parent grid
        parent_IW_idx = prev_IW_mesh.lookup_IW_idx(
            IW_mesh.A_grid_x[IW_idx],
            IW_mesh.A_grid_y[IW_idx],
        )

        # Retrieve the pre-shift
        shift_x = prev_VM_dx[parent_IW_idx]
        shift_y = prev_VM_dy[parent_IW_idx]
        shift_x = np.int32(
            0 if np.isnan(shift_x) else shift_x * PRESHIFT_ATTENUATION
        )
        shift_y = np.int32(
            0 if np.isnan(shift_y) else shift_y * PRESHIFT_ATTENUATION
        )

        # Apply the pre-shift to IW B (eager)
        IW_mesh.B_grid_x[IW_idx] += shift_x
        IW_mesh.B_grid_y[IW_idx] += shift_y
        IW_mesh.B_lims_x[IW_idx, :] += shift_x
        IW_mesh.B_lims_y[IW_idx, :] += shift_y

        # Check and prevent the shift of IW B from moving outside of the source
        # image B. When so, we will zero out part of the IW images that have
        # moved out-of-frame, later on.
        if IW_mesh.B_lims_x[IW_idx, 0] < 0:
            IW_needs_to_be_a_copy = True
            zero_out_R = np.abs(shift_x)
            IW_mesh.B_grid_x[IW_idx] -= shift_x
            IW_mesh.B_lims_x[IW_idx, :] -= shift_x
            shift_x = np.int32(0)

        if IW_mesh.B_lims_x[IW_idx, 1] > B.shape[1] - 1:
            IW_needs_to_be_a_copy = True
            zero_out_L = np.abs(shift_x)
            IW_mesh.B_grid_x[IW_idx] -= shift_x
            IW_mesh.B_lims_x[IW_idx, :] -= shift_x
            shift_x = np.int32(0)

        if IW_mesh.B_lims_y[IW_idx, 0] < 0:
            IW_needs_to_be_a_copy = True
            zero_out_D = np.abs(shift_y)
            IW_mesh.B_grid_y[IW_idx] -= shift_y
            IW_mesh.B_lims_y[IW_idx, :] -= shift_y
            shift_y = np.int32(0)

        if IW_mesh.B_lims_y[IW_idx, 1] > B.shape[0] - 1:
            IW_needs_to_be_a_copy = True
            zero_out_U = np.abs(shift_y)
            IW_mesh.B_grid_y[IW_idx] -= shift_y
            IW_mesh.B_lims_y[IW_idx, :] -= shift_y
            shift_y = np.int32(0)

        # Store
        IW_shifts_x[IW_idx] = shift_x
        IW_shifts_y[IW_idx] = shift_y

    # --------------------------------------------------------------------------
    #   Obtain images of IW frame A and IW frame B
    # --------------------------------------------------------------------------

    # Note: `A_` is a flipped left-to-right and up-to-down version of `A`, so we
    # have to flip the indices as well, hence the use of `A.shape[] - ...`.
    A_slice = (
        slice(
            A_.shape[0] - IW_mesh.A_lims_y[IW_idx, 1] - 1,
            A_.shape[0] - IW_mesh.A_lims_y[IW_idx, 0],
        ),
        slice(
            A_.shape[1] - IW_mesh.A_lims_x[IW_idx, 1] - 1,
            A_.shape[1] - IW_mesh.A_lims_x[IW_idx, 0],
        ),
    )
    B_slice = (
        slice(IW_mesh.B_lims_y[IW_idx, 0], IW_mesh.B_lims_y[IW_idx, 1] + 1),
        slice(IW_mesh.B_lims_x[IW_idx, 0], IW_mesh.B_lims_x[IW_idx, 1] + 1),
    )

    # fmt: off
    if IW_needs_to_be_a_copy:
        # We need a copy, because otherwise the upcoming zeroing of the IW image
        # borders will affect, by means of reference, the original image and
        # interfere with the correlation of upcoming and overlapping IWs.
        # Copying adds a tiny cpu overhead.
        IW_A_ = np.copy(A_[A_slice])  # C-contiguous
        IW_B  = np.copy(B [B_slice])  # C-contiguous

        # Zero out the appropiate section of the IW of frame B that corresponds
        # to `particles` that are definitely not present in the IW of frame A.
        # Likewise, zero out the IW of frame A. Zero caries the meaning of being
        # at the mean background level of the image.
        if zero_out_L > 0:
            IW_A_[:, :zero_out_L] = 0
            IW_B [:, :zero_out_L] = 0
        if zero_out_R > 0:
            IW_A_[:, -zero_out_R:] = 0
            IW_B [:, -zero_out_R:] = 0
        if zero_out_U > 0:
            IW_A_[:zero_out_U, :] = 0
            IW_B [:zero_out_U, :] = 0
        if zero_out_D > 0:
            IW_A_[-zero_out_D:, :] = 0
            IW_B [-zero_out_D:, :] = 0
    else:
        IW_A_ = A_[A_slice]  # Pass by reference, not contiguous
        IW_B  = B [B_slice]  # Pass by reference, not contiguous
    # fmt: on

    return IW_A_, IW_B
'''

''' COMMENTED OUT: JITCLASSED IW_MESH
@conditional_decorator(
    nb.njit(
        (
            nb.int32,
            nb.float32[:, ::1],
            nb.float32[:, ::1],
            IW_Mesh.class_type.instance_type,  # type: ignore
            nb.float32[::1],
            nb.float32[::1],
            IW_Mesh.class_type.instance_type,  # type: ignore
            nb.int32[::1],
            nb.int32[::1],
            nb.float32[:, :, :],
            FFT_Convolver2D_Full.class_type.instance_type,  # type: ignore
            nb.types.slice2_type,
        ),
        cache=True,
        nogil=True,
    ),
    cfg.FFT_LIB in (cfg.FFT_LIBS.ROCKETFFT, cfg.FFT_LIBS.SCIPY),
)
def process_IWs(
    # fmt: off
    stage_idx   : int,
    A_          : npt.NDArray[np.float32],  # read-only
    B           : npt.NDArray[np.float32],  # read-only
    prev_IW_mesh: IW_Mesh,                  # read-only
    prev_VM_dx  : npt.NDArray[np.float32],  # read-only
    prev_VM_dy  : npt.NDArray[np.float32],  # read-only
    IW_mesh     : IW_Mesh,                  # in-place operation on B, read-only on A
    IW_shifts_x : npt.NDArray[np.int32],    # in-place operation, debug output
    IW_shifts_y : npt.NDArray[np.int32],    # in-place operation, debug output
    C_maps      : npt.NDArray[np.float32],  # in-place output
    fft         : FFT_Convolver2D_Full,
    IWs_slice   : slice = slice(None),
):
    # fmt: on
    """In-place operation on:
    * `IW_mesh.B_grid_x`
    * `IW_mesh.B_grid_y`
    * `IW_mesh.B_lims_x`
    * `IW_mesh.B_lims_y`
    * `IW_shifts_x`
    * `IW_shifts_y`
    * `C_maps`

    Walk over all interrogation windows (IW) to be taken from source images `A_`
    and `B` using window pre-shifts when available from the previous multigrid
    stage, and compute the 2D correlation maps.

    The last argument `IWs_slice` can be set to only process a certain slice out
    of all the IWs. This is useful to distribute the calculation of all IWs over
    multiple concurrent tasks. Defaults to all IWs when omitted.
    """

    idx_start = 0 if IWs_slice.start is None else IWs_slice.start
    idx_stop = C_maps.shape[0] if IWs_slice.stop is None else IWs_slice.stop
    idx_step = 1 if IWs_slice.step is None else IWs_slice.step

    for IW_idx in range(idx_start, idx_stop, idx_step):
        IW_A_, IW_B = obtain_IWs_from_image(
            IW_idx,
            stage_idx,
            A_,
            B,
            prev_IW_mesh,
            prev_VM_dx,
            prev_VM_dy,
            IW_mesh,
            IW_shifts_x,
            IW_shifts_y,
        )

        # BOS
        if cfg.MODE == cfg.MODES.BOS:
            C_maps[IW_idx, :, :] = fft.convolve(IW_B, IW_A_)
            continue

        # PIV & PIV2
        if all_smaller_or_equal_to(IW_A_, 0) or all_smaller_or_equal_to(
            IW_B, 0
        ):
            # No details are present in the IW images. All pixels are below
            # or at the mean background --> Save computation time.
            # TODO: Make this a user config threshold? Is <= 0 even correct?
            # Must match up with 'zeroing out' mechanism of function
            # `obtain_IWs_from_image()`.
            C_maps[IW_idx, 0, 0] = np.nan

        else:
            # Perform 2D cross-correlation
            C_maps[IW_idx, :, :] = fft.convolve(IW_B, IW_A_)
'''
