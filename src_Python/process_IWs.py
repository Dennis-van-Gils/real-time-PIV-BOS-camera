#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "08-08-2023"
__version__ = "1.0"

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.signal import fftconvolve  # Only used for code validation

from dvg_cupy_convolve2d import FFTW_Convolver_Full2D
from my_fun import lookup_IW_idx, all_smaller_or_equal_to


# ------------------------------------------------------------------------------
#   obtain_IW_limits
# ------------------------------------------------------------------------------


@njit(
    "Tuple(( \
        int32, int32, int32, int32, \
        int32, int32, int32, int32, \
        int32, int32, int32, int32 \
    )) \
    (int32, int32, \
    Tuple((int32, int32)), \
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
def obtain_IW_limits(
    # fmt: off
    IW_idx     : int,
    stage_idx  : int,
    A_shape    : tuple[int, int],
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
) -> tuple[
    np.int32,
    np.int32,
    np.int32,
    np.int32,
    np.int32,
    np.int32,
    np.int32,
    np.int32,
    np.int32,
    np.int32,
    np.int32,
    np.int32,
]:
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
        shift_x = np.int32(0) if np.isnan(shift_x) else np.int32(shift_x)
        shift_y = np.int32(0) if np.isnan(shift_y) else np.int32(shift_y)

        # Apply the pre-shift to IW B (eager)
        B_IW_grid_x[IW_idx] += shift_x
        B_IW_grid_y[IW_idx] += shift_y
        B_IW_lims_x[IW_idx, :] += shift_x
        B_IW_lims_y[IW_idx, :] += shift_y

        # Check and prevent the shift of IW B from moving outside of the source
        # image B. When so, we will zero out part of the IW images that have
        # moved out-of-frame, later on.
        if B_IW_lims_x[IW_idx, 0] < 0:
            zero_out_R = np.abs(shift_x)
            B_IW_grid_x[IW_idx] -= shift_x
            B_IW_lims_x[IW_idx, :] -= shift_x
            shift_x = np.int32(0)

        if B_IW_lims_x[IW_idx, 1] > A_shape[1] - 1:
            zero_out_L = np.abs(shift_x)
            B_IW_grid_x[IW_idx] -= shift_x
            B_IW_lims_x[IW_idx, :] -= shift_x
            shift_x = np.int32(0)

        if B_IW_lims_y[IW_idx, 0] < 0:
            zero_out_D = np.abs(shift_y)
            B_IW_grid_y[IW_idx] -= shift_y
            B_IW_lims_y[IW_idx, :] -= shift_y
            shift_y = np.int32(0)

        if B_IW_lims_y[IW_idx, 1] > A_shape[0] - 1:
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
    Ax0 = A_shape[1] - A_IW_lims_x[IW_idx, 1] - 1
    Ax1 = A_shape[1] - A_IW_lims_x[IW_idx, 0]
    Ay0 = A_shape[0] - A_IW_lims_y[IW_idx, 1] - 1
    Ay1 = A_shape[0] - A_IW_lims_y[IW_idx, 0]

    Bx0 = B_IW_lims_x[IW_idx, 0]
    Bx1 = B_IW_lims_x[IW_idx, 1] + 1
    By0 = B_IW_lims_y[IW_idx, 0]
    By1 = B_IW_lims_y[IW_idx, 1] + 1

    return (
        np.int32(Ax0),
        np.int32(Ax1),
        np.int32(Ay0),
        np.int32(Ay1),
        np.int32(Bx0),
        np.int32(Bx1),
        np.int32(By0),
        np.int32(By1),
        np.int32(zero_out_L),
        np.int32(zero_out_R),
        np.int32(zero_out_U),
        np.int32(zero_out_D),
    )


# ------------------------------------------------------------------------------
#   process_IWs
# ------------------------------------------------------------------------------


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
    fftw       : FFTW_Convolver_Full2D,
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
        (
            Ax0,
            Ax1,
            Ay0,
            Ay1,
            Bx0,
            Bx1,
            By0,
            By1,
            zero_out_L,
            zero_out_R,
            zero_out_U,
            zero_out_D,
        ) = obtain_IW_limits(
            IW_idx,
            stage_idx,
            A_.shape,
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

        fftw.construct_IW_A_B(
            Ax0,
            Ax1,
            Ay0,
            Ay1,
            Bx0,
            Bx1,
            By0,
            By1,
            zero_out_L,
            zero_out_R,
            zero_out_U,
            zero_out_D,
        )

        fftw.convolve(IW_idx)

        """
        if fftw.IW_A_all_smaller_or_equal_to(0):
            # No details are present in the IW images. All pixels are below
            # or at the mean background --> Save computation time.
            # TODO: Make this a user config threshold? Is <= 0 even correct?
            # Must match up with 'zeroing out' mechanism of function
            # `obtain_IW_limits()`.
            C_maps[IW_idx, 0, 0] = np.nan

        else:
            # Perform 2D cross-correlation
            # C_maps[IW_idx, :, :] = fftconvolve(IW_B, IW_A_, mode="full")
            C_maps[IW_idx, :, :] = fftw.convolve(IW_idx)
        """

    C_maps[:] = fftw.processed_C_maps()
