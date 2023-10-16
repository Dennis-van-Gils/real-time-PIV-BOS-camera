#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "16-10-2023"
__version__ = "1.0"

import numpy as np

from utils.my_fun import lookup_IW_idx, normalize_C_maps
import init_config as cfg

if cfg.LOAD_MPL:
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.use("TkAgg")

# ------------------------------------------------------------------------------
#   output_debug_info
# ------------------------------------------------------------------------------


def output_debug_info(
    lIW_params,
    lA_IW_grid_x,
    lA_IW_grid_y,
    lA_IW_lims_x,
    lA_IW_lims_y,
    lB_IW_grid_x,
    lB_IW_grid_y,
    lB_IW_lims_x,
    lB_IW_lims_y,
    lIW_shifts_x,
    lIW_shifts_y,
    lC_maps,
    lVM_grid_x,
    lVM_grid_y,
    lVM_dx,
    lVM_dy,
):
    for stage_idx, IW_size in enumerate(cfg.IW_SIZES):
        # fmt: off
        N_IWs       = lIW_params  [stage_idx][2]
        A_IW_grid_x = lA_IW_grid_x[stage_idx]
        A_IW_grid_y = lA_IW_grid_y[stage_idx]
        A_IW_lims_x = lA_IW_lims_x[stage_idx]
        A_IW_lims_y = lA_IW_lims_y[stage_idx]
        B_IW_grid_x = lB_IW_grid_x[stage_idx]
        B_IW_grid_y = lB_IW_grid_y[stage_idx]
        B_IW_lims_x = lB_IW_lims_x[stage_idx]
        B_IW_lims_y = lB_IW_lims_y[stage_idx]
        IW_shifts_x = lIW_shifts_x[stage_idx]
        IW_shifts_y = lIW_shifts_y[stage_idx]
        C_maps      = lC_maps     [stage_idx]
        VM_grid_x   = lVM_grid_x  [stage_idx]
        VM_grid_y   = lVM_grid_y  [stage_idx]
        VM_dx       = lVM_dx      [stage_idx]
        VM_dy       = lVM_dy      [stage_idx]
        # fmt: on

        if cfg.SHOW_CORRELATION_MAPS and cfg.LOAD_MPL:
            # Reset any existing plot of the correlation map, because the IW
            # size has changed. Plotting on top of imshow needs a rescale.
            if plt.fignum_exists("C_map"):  # type: ignore
                plt.close("C_map")  # type: ignore

            # Plotting requires normalizing correlation maps for easy
            # comparison
            normalize_C_maps(C_maps)

        for IW_idx in range(N_IWs):
            # NOTE: Information on potentially zeroed-out sections inside
            # `IW_A` and `IW_B` is not stored nor accessible here.
            # Variables `zero_out_L/R/U/D` have not been committed to memory
            # to save on cpu time.

            # Short-hand variables
            IW_px_x = A_IW_grid_x[IW_idx]
            IW_px_y = A_IW_grid_y[IW_idx]
            shift_x = IW_shifts_x[IW_idx]
            shift_y = IW_shifts_y[IW_idx]
            C = C_maps[IW_idx]

            # We have to backwards calculate `peak_x` and `peak_y` again,
            # because they were not committed to memory to save on cpu time.
            # They represent the correlation map indices of the correlation
            # peak. We assume zero-padding was used for the FFT operations.
            dx = VM_dx[IW_idx]
            dy = VM_dy[IW_idx]
            shift_x = IW_shifts_x[IW_idx]
            shift_y = IW_shifts_y[IW_idx]
            peak_x = dx + C.shape[1] // 2 - shift_x
            peak_y = dy + C.shape[0] // 2 - shift_y

            if cfg.DEBUG:
                print(
                    f"IW: {IW_idx} of {N_IWs - 1} " f"@px {IW_px_x}, {IW_px_y}"
                )

                if stage_idx > 0:
                    parent_IW_idx = lookup_IW_idx(
                        IW_px_x,
                        IW_px_y,
                        lIW_params[stage_idx - 1],
                    )
                    print(f"   parent IW {parent_IW_idx}")
                    print(f"   shift  {shift_x:+2.0f}, {shift_y:+2.0f}")

                print(
                    "   A_xlim ["
                    f"{A_IW_lims_x[IW_idx, 0]:4d}, "
                    f"{A_IW_lims_x[IW_idx, 1]:4d}]"
                )
                print(
                    "   A_ylim ["
                    f"{A_IW_lims_y[IW_idx, 0]:4d}, "
                    f"{A_IW_lims_y[IW_idx, 1]:4d}]"
                )
                print(
                    "   B_xlim ["
                    f"{B_IW_lims_x[IW_idx, 0]:4d}, "
                    f"{B_IW_lims_x[IW_idx, 1]:4d}]"
                )
                print(
                    "   B_ylim ["
                    f"{B_IW_lims_y[IW_idx, 0]:4d}, "
                    f"{B_IW_lims_y[IW_idx, 1]:4d}]"
                )

                if not np.isnan(C[0, 0]):
                    print(f"     peak   @ {peak_x:+5.1f}, {peak_y:+5.1f}")
                    print(f"     dx, dy = {dx:+5.1f}, {dy:+5.1f}")

            if cfg.SHOW_CORRELATION_MAPS and cfg.LOAD_MPL:
                if not np.isnan(C[0, 0]):
                    if not (plt.fignum_exists("C_map")):  # type: ignore
                        fig = plt.figure("C_map")  # type: ignore
                        h_imshow = plt.imshow(  # type: ignore
                            C,
                            cmap="gray",
                            interpolation="none",
                            vmin=0,
                            vmax=1,
                        )
                        (h_peak,) = plt.plot([peak_x], [peak_y], "xr")  # type: ignore
                        h_title = plt.title(f"{IW_idx} of {N_IWs}")  # type: ignore

                    else:
                        h_imshow.set_data(C)  # type: ignore
                        h_peak.set_data([peak_x], [peak_y])  # type: ignore
                        h_title.set_text(f"{IW_idx} of {N_IWs}")  # type: ignore

                    plt.draw()  # type: ignore
                    plt.pause(0.0001)  # type: ignore
                    # plt.waitforbuttonpress()  # type: ignore
                    # plt.show(block=False)  # type: ignore
                    # plt.show()  # type: ignore
