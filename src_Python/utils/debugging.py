#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "22-10-2023"
__version__ = "1.0"

import numpy as np
import numpy.typing as npt

from utils.my_fun import normalize_C_maps
from utils.process_IWs import IW_Mesh
import init_config as cfg

# if cfg.LOAD_MPL:
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import axes

mpl.use("TkAgg")

# ------------------------------------------------------------------------------
#   print_info
# ------------------------------------------------------------------------------


def print_info(
    lIW_mesh: list[IW_Mesh],
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
        prev_IW_mesh = lIW_mesh    [stage_idx - 1]
        IW_mesh      = lIW_mesh    [stage_idx]
        IW_shifts_x  = lIW_shifts_x[stage_idx]
        IW_shifts_y  = lIW_shifts_y[stage_idx]
        C_maps       = lC_maps     [stage_idx]
        VM_dx        = lVM_dx      [stage_idx]
        VM_dy        = lVM_dy      [stage_idx]
        # fmt: on

        N_IWs = IW_mesh.N_IWs
        for IW_idx in range(N_IWs):
            # NOTE: Information on potentially zeroed-out sections inside
            # `IW_A` and `IW_B` is not stored nor accessible here.
            # Variables `zero_out_L/R/U/D` have not been committed to memory
            # to save on cpu time.

            # Short-hand variables
            IW_px_x = IW_mesh.A_grid_x[IW_idx]
            IW_px_y = IW_mesh.A_grid_y[IW_idx]
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

            print(f"IW: {IW_idx} of {N_IWs - 1} " f"@px {IW_px_x}, {IW_px_y}")

            if stage_idx > 0:
                parent_IW_idx = prev_IW_mesh.lookup_IW_idx(
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

            if not np.isnan(C[0, 0]):
                print(f"     peak   @ {peak_x:+5.1f}, {peak_y:+5.1f}")
                print(f"     dx, dy = {dx:+5.1f}, {dy:+5.1f}")


# ------------------------------------------------------------------------------
#   IW_analysis
# ------------------------------------------------------------------------------


def IW_analysis(
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
    N_stages = len(cfg.IW_SIZES)

    fig_1, axs_1 = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    ax_A: axes.Axes = axs_1[0]
    ax_B: axes.Axes = axs_1[1]

    ax_A.imshow(A, interpolation="nearest")
    ax_B.imshow(B, interpolation="nearest")

    fig_2, axs_2 = plt.subplots(N_stages, 3, figsize=(10, 6))

    for stage_idx, IW_size in enumerate(cfg.IW_SIZES):
        # fmt: off
        IW_mesh     = lIW_mesh[stage_idx]
        IW_shifts_x = lIW_shifts_x[stage_idx]
        IW_shifts_y = lIW_shifts_y[stage_idx]
        C_maps      = lC_maps     [stage_idx]
        VM_grid_x   = lVM_grid_x  [stage_idx]
        VM_grid_y   = lVM_grid_y  [stage_idx]
        VM_dx       = lVM_dx      [stage_idx]
        VM_dy       = lVM_dy      [stage_idx]
        # fmt: on

        IW_idx = IW_mesh.lookup_IW_idx(IW_px_x, IW_px_y)

        # IW outlines
        ax_A.add_patch(
            Rectangle(
                (IW_mesh.A_lims_x[IW_idx, 0], IW_mesh.A_lims_y[IW_idx, 0]),
                IW_mesh.A_lims_x[IW_idx, 1] - IW_mesh.A_lims_x[IW_idx, 0],
                IW_mesh.A_lims_y[IW_idx, 1] - IW_mesh.A_lims_y[IW_idx, 0],
                edgecolor="r",
                fill=None,
                lw=1,
            )
        )
        ax_B.add_patch(
            Rectangle(
                (IW_mesh.B_lims_x[IW_idx, 0], IW_mesh.B_lims_y[IW_idx, 0]),
                IW_mesh.B_lims_x[IW_idx, 1] - IW_mesh.B_lims_x[IW_idx, 0],
                IW_mesh.B_lims_y[IW_idx, 1] - IW_mesh.B_lims_y[IW_idx, 0],
                edgecolor="r",
                fill=None,
                lw=1,
            )
        )

        # Displacement vectors
        if not np.isnan(VM_dx[IW_idx]):
            ax_A.quiver(
                VM_grid_x[IW_idx],
                VM_grid_y[IW_idx],
                VM_dx[IW_idx],
                VM_dy[IW_idx],
                angles="xy",
                scale_units="xy",
                # scale=1,
                color="r",
                # units="xy",
                # width=0.5,
            )

        # Zoom to largest IW with a margin around it
        if stage_idx == 0:
            xmin = IW_mesh.A_lims_x[IW_idx, 0] - IW_size
            xmax = IW_mesh.A_lims_x[IW_idx, 1] + IW_size
            ymin = IW_mesh.A_lims_y[IW_idx, 0] - IW_size
            ymax = IW_mesh.A_lims_y[IW_idx, 1] + IW_size
            ax_A.set_xlim(xmin, xmax)
            ax_A.set_ylim(ymax, ymin)  # Must flip ymax and ymin due to imshow

        # Obtain images of IW frame A and IW frame B
        A_slice = (
            slice(IW_mesh.A_lims_y[IW_idx, 0], IW_mesh.A_lims_y[IW_idx, 1] + 1),
            slice(IW_mesh.A_lims_x[IW_idx, 0], IW_mesh.A_lims_x[IW_idx, 1] + 1),
        )
        B_slice = (
            slice(IW_mesh.B_lims_y[IW_idx, 0], IW_mesh.B_lims_y[IW_idx, 1] + 1),
            slice(IW_mesh.B_lims_x[IW_idx, 0], IW_mesh.B_lims_x[IW_idx, 1] + 1),
        )

        IW_A = A[A_slice]
        IW_B = B[B_slice]

        C_map = C_maps[IW_idx]
        C_map_h = C_map.shape[0] - 1
        C_map_w = C_map.shape[1] - 1

        # We have to backwards calculate `peak_x` and `peak_y` again,
        # because they were not committed to memory to save on cpu time.
        # They represent the correlation map indices of the correlation
        # peak. We assume zero-padding was used for the FFT operations.
        dx = VM_dx[IW_idx]
        dy = VM_dy[IW_idx]
        shift_x = IW_shifts_x[IW_idx]
        shift_y = IW_shifts_y[IW_idx]
        peak_x = dx + C_map_w // 2 - shift_x
        peak_y = dy + C_map_h // 2 - shift_y

        if N_stages == 1:
            ax_IW_A = axs_2[0]
            ax_IW_B = axs_2[1]
            ax_C_map = axs_2[2]
        else:
            ax_IW_A = axs_2[stage_idx, 0]
            ax_IW_B = axs_2[stage_idx, 1]
            ax_C_map = axs_2[stage_idx, 2]

        ax_IW_A.imshow(IW_A, interpolation="nearest")
        ax_IW_B.imshow(IW_B, interpolation="nearest")
        ax_C_map.imshow(C_map, interpolation="nearest")
        ax_C_map.plot((0, C_map_w), np.ones(2) * C_map_h / 2, "k")
        ax_C_map.plot(np.ones(2) * C_map_w / 2, (0, C_map_h), "k")
        ax_C_map.plot(peak_x, peak_y, "xr")

        ax_IW_A.grid(True)
        ax_IW_B.grid(True)
        ax_C_map.grid(True)

    plt.show()
