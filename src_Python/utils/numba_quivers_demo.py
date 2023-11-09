#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import perf_counter

from numba_quivers import draw_quiver_map_u8, draw_quiver_map_u24
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import cv2


# ------------------------------------------------------------------------------
#   build_quiver_arrays
# ------------------------------------------------------------------------------


def build_quiver_arrays(
    img_w: int,
    img_h: int,
    quiver_spacing: int = 50,
    start_radius: int = 50,
    N_frames: int = 360,
    colormap_name: str = "jet",
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
]:
    img_half_w = img_w // 2
    img_half_h = img_h // 2
    img_center_to_corner_distance = np.sqrt(img_half_w**2 + img_half_h**2)

    # Create grid of quivers, spaced equally apart
    # --------------------------------------------

    spacing = quiver_spacing
    d_spacing = np.sqrt(2 * (spacing // 2) ** 2)
    N_quivers_x = int((img_w - spacing) // spacing + 1)
    N_quivers_y = int((img_h - spacing) // spacing + 1)
    N_quivers = N_quivers_x * N_quivers_y

    arr_x = np.arange(N_quivers_x) * spacing + spacing // 2
    arr_y = np.arange(N_quivers_y) * spacing + spacing // 2
    arr_x = np.asarray(arr_x, dtype=np.int32)
    arr_y = np.asarray(arr_y, dtype=np.int32)
    grid_x = np.empty(N_quivers, dtype=np.int32)
    for i in np.arange(N_quivers_y):
        grid_x[i * N_quivers_x : (i + 1) * N_quivers_x] = arr_x
    grid_y = np.repeat(arr_y, N_quivers_x)

    # Build color lookup table (lut)
    # ------------------------------

    N_COLORS_LUT = 1024
    mpl_cm = plt.get_cmap(colormap_name, N_COLORS_LUT)
    mpl_cm._init()  # type: ignore
    mpl_lut = mpl_cm._lut  # type: ignore
    cv2_lut = np.asarray(mpl_lut * 255, dtype=np.uint8)  # [0., 1.] to [0, 255]
    cv2_lut = cv2_lut[:, 2::-1]  # Drop `A` from `RGBA`and turn `RGB` into `BGR`

    colors_u8 = np.ones(N_quivers, dtype=np.uint8)
    colors_u8[:] = 255
    colors_u24 = np.zeros((N_quivers, 3), dtype=np.uint8)

    # Animate all quivers by spinning each one revolution
    # ---------------------------------------------------

    pts1 = np.zeros((N_quivers, 2), dtype=np.int32)
    pts2 = np.zeros((N_quivers * N_frames, 2), dtype=np.int32)
    thetas = np.linspace(0, 2 * np.pi, N_frames)

    for frame_idx, theta in enumerate(thetas):
        for quiver_idx in range(N_quivers):
            # Start point
            x1 = grid_x[quiver_idx]
            y1 = grid_y[quiver_idx]

            # The radius `r` of each quiver depends on its distance `d` from the
            # image center. It falls of towards 0 at the quivers in the very
            # corners.
            d = np.sqrt((img_half_w - x1) ** 2 + (img_half_h - y1) ** 2)
            d = d / (img_center_to_corner_distance - d_spacing)
            r = np.maximum(start_radius * (1 - d), 0)

            # End point
            x2 = np.round(x1 + r * np.sin(theta))
            y2 = np.round(y1 - r * np.cos(theta))

            if frame_idx == 0:
                pts1[quiver_idx][0] = x1
                pts1[quiver_idx][1] = y1

                lut_idx = int(np.round(r / start_radius * (N_COLORS_LUT - 1)))
                colors_u24[quiver_idx] = cv2_lut[lut_idx]

            pts2[quiver_idx + frame_idx * N_quivers][0] = x2
            pts2[quiver_idx + frame_idx * N_quivers][1] = y2

    return thetas, pts1, pts2, colors_u8, colors_u24


# ------------------------------------------------------------------------------
#   main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    USE_COLOR = 1
    img_w, img_h = 900, 600

    thetas, pts1, pts2, colors_u8, colors_u24 = build_quiver_arrays(
        img_w,
        img_h,
        quiver_spacing=50,
        start_radius=50,
        N_frames=360,
        colormap_name="jet",
    )

    quiver_kwargs = {
        "linewidth": 2,
        "tip_size": 0.2,
        "tip_angle": np.pi / 4,
    }

    N_quivers = len(pts1)
    N_frames = len(thetas)
    print(f"{img_w} x {img_h}")
    print(f"N_quivers = {N_quivers}")
    print(f"N_frames  = {N_frames}\n")

    # Pure draw and plot
    # ------------------
    img_empty = np.zeros(
        (img_h, img_w, 3) if USE_COLOR else (img_h, img_w),
        dtype=np.uint8,
    )
    img = np.copy(img_empty)

    T = 0
    for frame_idx, theta in enumerate(thetas):
        np.copyto(img, img_empty)
        pts2_set = pts2[frame_idx * N_quivers : (frame_idx + 1) * N_quivers]

        tick = perf_counter()
        if USE_COLOR:
            draw_quiver_map_u24(img, pts1, pts2_set, colors_u24, **quiver_kwargs)
        else:
            draw_quiver_map_u8(img, pts1, pts2_set, colors_u8, **quiver_kwargs)
        T += perf_counter() - tick

        cv2.imshow("output", img)
        cv2.setWindowTitle("output", f"{theta * 180 / np.pi:.1f}")
        # cv2.imwrite(f"export_{frame_idx:04d}.png", img)
        cv2.waitKey(10)

    ms_per_frame = T / N_frames * 1000
    ms_per_quiver = T / N_frames / N_quivers * 1000
    print(f"Per frame : {ms_per_frame :.4f} ms")
    print(f"Per quiver: {ms_per_quiver:.4f} ms")
