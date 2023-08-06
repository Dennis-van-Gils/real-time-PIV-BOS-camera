#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "06-08-2023"
__version__ = "1.0"

import numpy as np
import numpy.typing as npt

from dvg_fftw_convolve2d import FFTW_Convolver_Full2D
from my_fun import lookup_IW_idx, fast_max


def process_IWs(
    # fmt: off
    stage_idx: int,
    A_: npt.NDArray[np.float32],         # constant, read only
    B: npt.NDArray[np.float32],          # constant, read only
    prev_IW_params: tuple[
        int, float, int, int, int
    ],  # == lIW_params[stage_idx - 1]   # constant, read only
    prev_VM_dx: npt.NDArray[
        np.float32
    ],  # == lVM_dx[stage_idx - 1]       # constant, read only
    prev_VM_dy: npt.NDArray[
        np.float32
    ],  # == lVM_dy[stage_idx - 1]       # constant, read only
    A_IW_grid_x: npt.NDArray[np.int32],  # constant, read only
    A_IW_grid_y: npt.NDArray[np.int32],  # constant, read only
    A_IW_lims_x: npt.NDArray[np.int32],  # constant, read only
    A_IW_lims_y: npt.NDArray[np.int32],  # constant, read only
    B_IW_grid_x: npt.NDArray[np.int32],  # in-place operation, not an output
    B_IW_grid_y: npt.NDArray[np.int32],  # in-place operation, not an output
    B_IW_lims_x: npt.NDArray[np.int32],  # in-place operation, not an output
    B_IW_lims_y: npt.NDArray[np.int32],  # in-place operation, not an output
    IW_A_: npt.NDArray[np.float32],      # in-place operation, not an output
    IW_B: npt.NDArray[np.float32],       # in-place operation, not an output
    IW_shifts_x: npt.NDArray[np.int32],  # in-place output
    IW_shifts_y: npt.NDArray[np.int32],  # in-place output
    C_maps: npt.NDArray[np.float32],     # in-place output
    fftw: FFTW_Convolver_Full2D,
    # fmt: on
):
    for IW_idx in range(C_maps.shape[0]):
        # ------------------------------------------------------------------
        #   Calculate IW of frame B
        #   Apply window shifting technique
        # ------------------------------------------------------------------

        # Part of the window shifting mechanism:
        # Undo the shift again when the shifted IW of frame B is leaving the
        # borders of frame B. If so, we will, later on, zero out the
        # appropiate section of the IW of frame B that corresponds to
        # `particles` that are definitely not present in the IW of frame A.
        # Likewise, we will zero out pixels in frame A that are not present
        # in frame B.
        zero_out_L = 0  # left of B , x = 0
        zero_out_R = 0  # right of B, x = IW_size - 1
        zero_out_U = 0  # up of B   , y = 0
        zero_out_D = 0  # down of B , y = IW_size - 1
        IW_needs_to_be_a_copy = False

        # Check for window pre-shift
        if stage_idx == 0:
            shift_x = 0  # [px]
            shift_y = 0  # [px]

        else:
            # Pre-shift available: Look up corresponding index of the IW in
            # the larger parent grid
            parent_IW_idx = lookup_IW_idx(
                A_IW_grid_x[IW_idx],
                A_IW_grid_y[IW_idx],
                prev_IW_params,
            )

            # Retrieve the pre-shift
            shift_x = prev_VM_dx[parent_IW_idx]
            shift_y = prev_VM_dy[parent_IW_idx]
            shift_x = 0 if np.isnan(shift_x) else int(shift_x)
            shift_y = 0 if np.isnan(shift_y) else int(shift_y)

            # Apply the pre-shift to IW B (eager)
            B_IW_grid_x[IW_idx] += shift_x
            B_IW_grid_y[IW_idx] += shift_y
            B_IW_lims_x[IW_idx, :] += shift_x
            B_IW_lims_y[IW_idx, :] += shift_y

            # Check and prevent the shift of IW B from moving outside of the
            # source image B. When so, we will zero out part of the IW
            # images that have moved out-of-frame, later on.
            if B_IW_lims_x[IW_idx, 0] < 0:
                IW_needs_to_be_a_copy = True
                zero_out_R = np.abs(shift_x)
                B_IW_grid_x[IW_idx] -= shift_x
                B_IW_lims_x[IW_idx, :] -= shift_x
                shift_x = 0
            else:
                zero_out_R = 0

            if B_IW_lims_x[IW_idx, 1] > B.shape[1] - 1:
                IW_needs_to_be_a_copy = True
                zero_out_L = np.abs(shift_x)
                B_IW_grid_x[IW_idx] -= shift_x
                B_IW_lims_x[IW_idx, :] -= shift_x
                shift_x = 0
            else:
                zero_out_L = 0

            if B_IW_lims_y[IW_idx, 0] < 0:
                IW_needs_to_be_a_copy = True
                zero_out_D = np.abs(shift_y)
                B_IW_grid_y[IW_idx] -= shift_y
                B_IW_lims_y[IW_idx, :] -= shift_y
                shift_y = 0
            else:
                zero_out_D = 0

            if B_IW_lims_y[IW_idx, 1] > B.shape[0] - 1:
                IW_needs_to_be_a_copy = True
                zero_out_U = np.abs(shift_y)
                B_IW_grid_y[IW_idx] -= shift_y
                B_IW_lims_y[IW_idx, :] -= shift_y
                shift_y = 0
            else:
                zero_out_U = 0

            # Store
            IW_shifts_x[IW_idx] = shift_x
            IW_shifts_y[IW_idx] = shift_y

        # ------------------------------------------------------------------
        #   Retrieve images of IW frame A and IW frame B
        # ------------------------------------------------------------------

        # Note: `A_` is a flipped left-to-right and up-to-down version of
        # `A`, so we have to flip the indices as well, hence the use of
        # `A.shape[] - ...`.
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
            # We need a copy, because otherwise the upcoming zeroing of the
            # IW image borders will affect, by means of reference, the
            # original image and interfere with the correlation of upcoming
            # and overlapping IWs. Copying adds a tiny cpu overhead.
            np.copyto(IW_A_, A_[A_slice])
            np.copyto(IW_B , B [B_slice])

            # Zero out the appropiate section of the IW of frame B that
            # corresponds to `particles` that are definitely not present in
            # the IW of frame A. Likewise, zero out the IW of frame A. Zero
            # caries the meaning of being at the mean background level of
            # the image.
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
            IW_A_ = A_[A_slice]  # Pass by reference
            IW_B  = B [B_slice]  # Pass by reference
        # fmt: on

        # ------------------------------------------------------------------
        #   Perform cross-correlation
        # ------------------------------------------------------------------

        if fast_max(IW_A_) <= 0:
            # No details are present in the IW images. All pixels are below
            # or at the mean background --> Save computation time.
            # TODO: Make this a user config threshold? Is <= 0 even correct?
            # Must match up with 'zeroing out' mechanism, just above here.
            C_maps[IW_idx, 0, 0] = np.nan

        else:
            # Perform 2D cross-correlation
            # C_maps[IW_idx, :, :] = fftconvolve(IW_B, IW_A_, mode="full")
            C_maps[IW_idx, :, :] = fftw.convolve(IW_B, IW_A_)
