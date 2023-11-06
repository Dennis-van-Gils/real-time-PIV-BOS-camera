#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "03-11-2023"
__version__ = "1.0"

import init_config as cfg

import numpy as np
import numpy.typing as npt
import cv2

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors

mpl.use("TkAgg")


# ------------------------------------------------------------------------------
#   vector_map_to_hsv_colors
# ------------------------------------------------------------------------------


def vector_map_to_hsv_colors(
    VM_magn: npt.NDArray[np.float32],
    VM_angle: npt.NDArray[np.float32],
    VM_grid_shape_2D: tuple[int, int],
    VM_magn_multiplier: float = 255,
    output_resolution: tuple[int, int] | None = None,
    interpolation: int = cv2.INTER_CUBIC,
) -> npt.NDArray[np.uint8]:
    """Generate and return an `uint8` RGB image where the passed vector data
    gets interpreted as an HSV image as follows:
        - The Hue        channel is set by the vector angles.
        - The Saturation channel is set at constant max value of 255.
        - The Value      channel is set by the vector magnitudes.

    Args:
        VM_magn (``numpy.NDArray[np.float32]``):
            Flattened array containing the vector magnitudes per IW.

        VM_angle (``numpy.NDArray[np.float32]``):
            Flattened array containing the vector angles in degrees per IW.

        VM_grid_shape_2D (``tuple[int, int]``):
            2D-shape of the grid corresponding to the above arrays, i.e.
            [N_IWs_y, N_IWs_x].

        VM_magn_multiplier (``float``, optional):
            The vector magnitudes get multiplied with this factor to make up the
            Value channel, in turn being clipped over the range [0, 255].

            Default: 255

        output_resolution (``tuple[int, int] | None``, optional)
            The returned RGB image nominally has a 2D-raster shape given
            by `VM_grid_shape_2D`, i.e. the number of IWs along the x and y
            directions. One can upscale to a larger resolution by supplying a
            custom resolution here. When omitted or set to `None` it will
            default to the nominal resolution.

            Default: None

        interpolation (``int``, optional):
            Interpolation scheme to be used for rescaling to a different output
            resolution, see cv2.InterpolationFlags.
            https://docs.opencv.org/4.8.0/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
            Common values: cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST

            Default: cv2.INTER_CUBIC

    Returns:
        The RGB image as a 3D numpy array of `uint8` values.

    """
    if output_resolution is None:
        output_resolution = VM_grid_shape_2D

    VM_magn = np.nan_to_num(np.reshape(VM_magn, VM_grid_shape_2D))
    VM_angle = np.nan_to_num(np.reshape(VM_angle, VM_grid_shape_2D))

    # Create an HSV canvas and color it in
    canvas = np.zeros((*VM_grid_shape_2D, 3), dtype=np.uint8)
    canvas[..., 0] = np.floor(VM_angle / (360 / 179))  # Range [0, 179]
    canvas[..., 1] = 255
    canvas[..., 2] = np.clip(VM_magn * VM_magn_multiplier, 0, 255)

    cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR, dst=canvas)
    if output_resolution != VM_grid_shape_2D:
        canvas = cv2.resize(canvas, output_resolution, interpolation=interpolation)

    return np.asarray(canvas, dtype=np.uint8)


# ------------------------------------------------------------------------------
#   vector_map_to_quiver_plot
# ------------------------------------------------------------------------------


def vector_map_to_quiver_plot(
    background_img: npt.NDArray[np.float32],
    VM_grid_x: npt.NDArray[np.int32],
    VM_grid_y: npt.NDArray[np.int32],
    VM_dx: npt.NDArray[np.float32],
    VM_dy: npt.NDArray[np.float32],
    VM_magn: npt.NDArray[np.float32],
    frame_title: str,
    colormap: mpl_colors.Colormap = plt.get_cmap("jet"),
    # colormap: mpl_colors.Colormap = mpl.cm.jet,
):
    self = vector_map_to_quiver_plot
    VM_colors = VM_magn / cfg.COLOR_DIV

    if not plt.fignum_exists("VM_quiver_plot"):  # type: ignore
        plt.figure("VM_quiver_plot")
        self.h_imshow = plt.imshow(
            background_img,
            cmap="gray",
            interpolation="none",
        )
        self.h_quiver = plt.quiver(
            VM_grid_x,
            VM_grid_y,
            np.zeros(VM_dx.shape),
            np.zeros(VM_dy.shape),
            angles="xy",
            scale_units="xy",
            scale=1,  # Scales down by `scale`
            color=colormap(VM_colors),
            linewidths=1,
        )
        self.h_title = plt.title(f"{frame_title}")

    self.h_imshow.set_data(background_img)
    self.h_quiver.set_UVC(VM_dx * cfg.QUIVER_SIZE, VM_dy * cfg.QUIVER_SIZE)
    self.h_quiver.set_color(colormap(VM_colors))  # type: ignore
    self.h_title.set_text(f"{frame_title}")

    plt.draw()
    plt.pause(0.0001)
