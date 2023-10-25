#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "25-10-2023"
__version__ = "1.0"

import numpy as np

from utils.process_IWs import IW_Mesh
import init_config as cfg

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import axes

# ------------------------------------------------------------------------------
#   backwards_calculate_C_peak
# ------------------------------------------------------------------------------


def backwards_calculate_C_peak(shift_x, dx, C_map_shape_x):
    # We have to backwards calculate the `peak_x` and `peak_y` again,
    # because they were not committed to memory to save on cpu time.
    # They represent the correlation map indices of the correlation
    # peak. We assume zero-padding was used for the FFT operations.
    peak_x = dx + C_map_shape_x // 2 - shift_x
    return peak_x


# ------------------------------------------------------------------------------
#   print_IWs
# ------------------------------------------------------------------------------


def print_IWs(
    lIW_mesh: list[IW_Mesh],
    lIW_shifts_x,
    lIW_shifts_y,
    lC_maps,
    lVM_dx,
    lVM_dy,
):
    for stage_idx, IW_mesh in enumerate(lIW_mesh):
        N_IWs = IW_mesh.N_IWs

        for IW_idx in range(N_IWs):
            # NOTE: Information on potentially zeroed-out sections inside
            # `IW_A` and `IW_B` is not stored nor accessible here.
            # Variables `zero_out_L/R/U/D` have not been committed to memory
            # to save on cpu time.

            # Short-hands
            # fmt: off
            IW_px_x = IW_mesh.A_grid_x[IW_idx]
            IW_px_y = IW_mesh.A_grid_y[IW_idx]
            shift_x = lIW_shifts_x[stage_idx][IW_idx]
            shift_y = lIW_shifts_y[stage_idx][IW_idx]
            C_map   = lC_maps     [stage_idx][IW_idx]
            dx      = lVM_dx      [stage_idx][IW_idx]
            dy      = lVM_dy      [stage_idx][IW_idx]
            peak_x  = backwards_calculate_C_peak(shift_x, dx, C_map.shape[1])
            peak_y  = backwards_calculate_C_peak(shift_y, dy, C_map.shape[0])
            # fmt: on

            print(f"IW: {IW_idx} of {N_IWs - 1} " f"@px {IW_px_x}, {IW_px_y}")

            if stage_idx > 0:
                parent_IW_idx = lIW_mesh[stage_idx - 1].lookup_IW_idx(
                    IW_px_x,
                    IW_px_y,
                )
                print(f"   parent IW {parent_IW_idx}")
                print(f"   shift  {shift_x:+2.0f}, {shift_y:+2.0f}")

            print(
                "   A_xlim ["
                f"{IW_mesh.A_lims_x[IW_idx, 0]:4d}, "
                f"{IW_mesh.A_lims_x[IW_idx, 1]:4d}]"
            )
            print(
                "   A_ylim ["
                f"{IW_mesh.A_lims_y[IW_idx, 0]:4d}, "
                f"{IW_mesh.A_lims_y[IW_idx, 1]:4d}]"
            )
            print(
                "   B_xlim ["
                f"{IW_mesh.B_lims_x[IW_idx, 0]:4d}, "
                f"{IW_mesh.B_lims_x[IW_idx, 1]:4d}]"
            )
            print(
                "   B_ylim ["
                f"{IW_mesh.B_lims_y[IW_idx, 0]:4d}, "
                f"{IW_mesh.B_lims_y[IW_idx, 1]:4d}]"
            )

            if not np.isnan(C_map[0, 0]):
                print(f"     peak   @ {peak_x:+5.1f}, {peak_y:+5.1f}")
                print(f"     dx, dy = {dx:+5.1f}, {dy:+5.1f}")


# ------------------------------------------------------------------------------
#   plot_IW_analysis
# ------------------------------------------------------------------------------


def plot_IW_analysis(
    IW_px_x: int,
    IW_px_y: int,
    A,
    B,
    lIW_mesh: list[IW_Mesh],
    lIW_shifts_x,
    lIW_shifts_y,
    lC_maps,
    lVM_grid_x,
    lVM_grid_y,
    lVM_dx,
    lVM_dy,
):
    fig_1, axs_1 = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    ax_A: axes.Axes = axs_1[0]
    ax_B: axes.Axes = axs_1[1]

    ax_A.imshow(A, interpolation="nearest")
    ax_B.imshow(B, interpolation="nearest")

    fig_2, axs_2 = plt.subplots(cfg.N_STAGES, 3, figsize=(10, 6))

    print(f"\n@px {IW_px_x}, {IW_px_y}")
    print("stage | IW_idx |  pre-shift |   VM_dx,   VM_dy")
    print("------|--------|------------|-----------------")
    str_lines = []
    title_specs = {"fontsize": 8, "fontfamily": "monospace"}

    for stage_idx, IW_mesh in enumerate(lIW_mesh):
        IW_idx = IW_mesh.lookup_IW_idx(IW_px_x, IW_px_y)

        # Short-hands
        # fmt: off
        A_IW_lims_x = IW_mesh.A_lims_x[IW_idx]
        A_IW_lims_y = IW_mesh.A_lims_y[IW_idx]
        B_IW_lims_x = IW_mesh.B_lims_x[IW_idx]
        B_IW_lims_y = IW_mesh.B_lims_y[IW_idx]
        shift_x = lIW_shifts_x[stage_idx][IW_idx]
        shift_y = lIW_shifts_y[stage_idx][IW_idx]
        C_map   = lC_maps     [stage_idx][IW_idx]
        grid_x  = lVM_grid_x  [stage_idx][IW_idx]
        grid_y  = lVM_grid_y  [stage_idx][IW_idx]
        dx      = lVM_dx      [stage_idx][IW_idx]
        dy      = lVM_dy      [stage_idx][IW_idx]
        peak_x  = backwards_calculate_C_peak(shift_x, dx, C_map.shape[1])
        peak_y  = backwards_calculate_C_peak(shift_y, dy, C_map.shape[0])
        # fmt: on

        # For plot titles
        str_line = (
            f"{stage_idx:5d} | {IW_idx:6d} | "
            f"{shift_x:+4.0f}, {shift_y:+4.0f} | {dx:+7.2f}, {dy:+7.2f}"
        )
        str_line2 = f"{shift_x:+4.0f}, {shift_y:+4.0f} | {dx:+7.2f}, {dy:+7.2f}"
        str_lines.append(str_line)
        print(str_line)

        # IW outlines
        ax_A.add_patch(
            Rectangle(
                (A_IW_lims_x[0], A_IW_lims_y[0]),
                np.diff(A_IW_lims_x)[0],
                np.diff(A_IW_lims_y)[0],
                edgecolor="r",
                fill=None,
                lw=1,
            )
        )
        ax_B.add_patch(
            Rectangle(
                (B_IW_lims_x[0], B_IW_lims_y[0]),
                np.diff(B_IW_lims_x)[0],
                np.diff(B_IW_lims_y)[0],
                edgecolor="r",
                fill=None,
                lw=1,
            )
        )

        # Displacement vectors
        if not np.isnan(dx):
            ax_A.quiver(
                grid_x,
                grid_y,
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                # scale=1,
                color="r",
                # units="xy",
                # width=0.5,
            )

        # Zoom to largest IW with a margin around it
        if stage_idx == 0:
            xmin = A_IW_lims_x[0] - IW_mesh.IW_size
            xmax = A_IW_lims_x[1] + IW_mesh.IW_size
            ymin = A_IW_lims_y[0] - IW_mesh.IW_size
            ymax = A_IW_lims_y[1] + IW_mesh.IW_size
            ax_A.set_xlim(xmin, xmax)
            ax_A.set_ylim(ymax, ymin)  # Must flip ymax and ymin due to imshow

        # Obtain images of IW frame A and IW frame B
        IW_A = A[
            slice(A_IW_lims_y[0], A_IW_lims_y[1] + 1),
            slice(A_IW_lims_x[0], A_IW_lims_x[1] + 1),
        ]
        IW_B = B[
            slice(B_IW_lims_y[0], B_IW_lims_y[1] + 1),
            slice(B_IW_lims_x[0], B_IW_lims_x[1] + 1),
        ]

        if cfg.N_STAGES == 1:
            ax_IW_A = axs_2[0]
            ax_IW_B = axs_2[1]
            ax_C_map = axs_2[2]
        else:
            ax_IW_A = axs_2[stage_idx, 0]
            ax_IW_B = axs_2[stage_idx, 1]
            ax_C_map = axs_2[stage_idx, 2]

        ax_IW_A.imshow(IW_A, interpolation="nearest")
        ax_IW_B.imshow(IW_B, interpolation="nearest")
        ax_IW_A.grid(True)
        ax_IW_B.grid(True)

        C_map_h = C_map.shape[0] - 1
        C_map_w = C_map.shape[1] - 1
        ax_C_map.imshow(C_map, interpolation="nearest")
        ax_C_map.plot((0, C_map_w), np.ones(2) * C_map_h / 2, "k")
        ax_C_map.plot(np.ones(2) * C_map_w / 2, (0, C_map_h), "k")
        ax_C_map.plot(peak_x, peak_y, "xr")
        ax_C_map.grid(True)
        ax_C_map.set_title(str_line2, **title_specs)

    ax_A.set_title(f"@px {IW_px_x}, {IW_px_y}", **title_specs)
    ax_B.set_title("\n".join(str_lines), **title_specs)

    print("")
    plt.show()
