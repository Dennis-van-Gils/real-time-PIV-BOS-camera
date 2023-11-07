#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "07-11-2023"
__version__ = "1.0"

import sys

import init_config as cfg

import numpy as np
import numpy.typing as npt
import matplotlib as mpl
from matplotlib import pyplot as plt
import cv2

mpl.use("TkAgg")

N_COLORS_LUT = 1024


# ------------------------------------------------------------------------------
#   build_mpl_colormap
# ------------------------------------------------------------------------------


def build_mpl_colormap(
    mpl_colormap_name: str = "jet",
    mpl_set_under=None,
    mpl_set_over=None,
):
    """Build and return an instance to a Matplotlib colormap as configured.

    Args:
        mpl_colormap_name (``str``, optional):
            Name of a built-in Matplotlib colormap, e.g. "jet" or "hot", etc.

            Default: "jet"

        mpl_set_under (matplotlib color, optional):
            Matplotlib color specification (e.g. "k" or a RGBA color tuple) to
            indicate out-of-range under values.

            Default: None

        mpl_set_over (matplotlib color, optional):
            Matplotlib color specification (e.g. "k" or a RGBA color tuple to
            indicate out-of-range over values.

            Default: None
    """
    mpl_cm = plt.get_cmap(mpl_colormap_name)

    if mpl_set_under is not None:
        mpl_cm.set_under(mpl_set_under)

    if mpl_set_over is not None:
        mpl_cm.set_over(mpl_set_over)

    return mpl_cm


# ------------------------------------------------------------------------------
#   build_cv2_colormap_lut
# ------------------------------------------------------------------------------


def build_cv2_colormap_lut(
    mpl_colormap_name: str = "jet",
    mpl_set_under=None,
    mpl_set_over=None,
) -> list[list[int]]:
    """Build and return a color lookup table taken from MatplotLib and converted
    for use in OpenCV. Matplotlib uses float [0 - 1] RGBA colors. OpenCV uses
    [0 - 255] BGR colors.

    Args:
        mpl_colormap_name (``str``, optional):
            Name of a built-in Matplotlib colormap, e.g. "jet" or "hot", etc.

            Default: "jet"

        mpl_set_under (matplotlib color, optional):
            Matplotlib color specification (e.g. "k" or a RGBA color tuple) to
            indicate out-of-range under values.

            Default: None

        mpl_set_over (matplotlib color, optional):
            Matplotlib color specification (e.g. "k" or a RGBA color tuple to
            indicate out-of-range over values.

            Default: None

    Returns (``list[list[int]]``):
        List containing BGR color values as `[color_idx][B, G, R]` with a shape
        equal to `(N_COLORS_LUT + 2, 3)`. The colormap proper has a length of
        `N_COLORS_LUT`. The extra last three entries are special colors:
            N_COLORS_LUT    : out-of-range under color
            N_COLORS_LUT + 1: out-of-range over color
            N_COLORS_LUT + 2: mask color (unused)
    """
    mpl_cm = plt.get_cmap(mpl_colormap_name, N_COLORS_LUT)

    if mpl_set_under is not None:
        mpl_cm.set_under(mpl_set_under)

    if mpl_set_over is not None:
        mpl_cm.set_over(mpl_set_over)

    mpl_cm._init()  # Build the lut # type: ignore
    mpl_lut = mpl_cm._lut  # type: ignore
    cv2_lut = np.asarray(mpl_lut * 255, dtype=np.uint8)  # [0., 1.] to [0, 255]
    cv2_lut = cv2_lut[:, 2::-1]  # Drop `A` from `RGBA`and turn `RGB` into `BGR`

    return cv2_lut.tolist()


# ------------------------------------------------------------------------------
#   Module-level defined colormaps
# ------------------------------------------------------------------------------

this = sys.modules[__name__]

this.mpl_colormap = build_mpl_colormap(  # type: ignore
    mpl_colormap_name=cfg.COLORMAP_NAME,
    mpl_set_under=cfg.COLORMAP_OUT_OF_RANGE_UNDER,
    mpl_set_over=cfg.COLORMAP_OUT_OF_RANGE_OVER,
)

this.cv2_colormap_lut = build_cv2_colormap_lut(  # type: ignore
    mpl_colormap_name=cfg.COLORMAP_NAME,
    mpl_set_under=cfg.COLORMAP_OUT_OF_RANGE_UNDER,
    mpl_set_over=cfg.COLORMAP_OUT_OF_RANGE_OVER,
)

# ------------------------------------------------------------------------------
#   get_color_from_cv2_colormap_lut
# ------------------------------------------------------------------------------


def get_color_from_cv2_colormap_lut(value: float):
    """Look up and return the BGR color value from the colormap as defined at
    this module level (`[plotting.py].cv2_colormap_lut`).

    Args:
        value (``float``):
            Normalized color lookup value.
                0.0 <= value <= 1.0: colormap proper
                value < 0.0        : out-of-range under color
                value > 1.0        : out-of-range over color
    """
    if value < 0.0:  # Out-of-range under
        lut_idx = N_COLORS_LUT
    elif value > 1.0:  # Out-of-range over
        lut_idx = N_COLORS_LUT + 1
    else:  # Colormap proper
        lut_idx = int(np.round(value * (N_COLORS_LUT - 1)))

    return this.cv2_colormap_lut[lut_idx]


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
        canvas = cv2.resize(
            canvas, output_resolution, interpolation=interpolation
        )

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
):
    """ """
    self = vector_map_to_quiver_plot
    colormap = this.mpl_colormap
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
    self.h_quiver.set_color(colormap(VM_colors))
    self.h_title.set_text(f"{frame_title}")

    plt.draw()
    plt.pause(0.0001)


# ------------------------------------------------------------------------------
#   vector_map_to_cv2_quiver_plot
# ------------------------------------------------------------------------------


def vector_map_to_cv2_quiver_plot(
    background_img: npt.NDArray[np.float32],
    VM_grid_x: npt.NDArray[np.int32],
    VM_grid_y: npt.NDArray[np.int32],
    VM_dx: npt.NDArray[np.float32],
    VM_dy: npt.NDArray[np.float32],
    VM_magn: npt.NDArray[np.float32],
    output_resolution: tuple[int, int] | None = None,
    interpolation: int = cv2.INTER_AREA,
) -> npt.NDArray[np.uint8]:
    """ """
    if output_resolution is None:
        output_resolution = (background_img.shape[1], background_img.shape[0])

    canvas = cv2.cvtColor(background_img * 255, cv2.COLOR_GRAY2BGR)

    for IW_idx in range(len(VM_grid_x)):
        # fmt: off
        x    = VM_grid_x[IW_idx]
        y    = VM_grid_y[IW_idx]
        dx   = VM_dx[IW_idx]
        dy   = VM_dy[IW_idx]
        magn = VM_magn[IW_idx]
        # fmt: on

        if not np.isnan(dx):
            canvas = cv2.arrowedLine(
                canvas,
                pt1=(x, y),
                pt2=(
                    int(np.round(x + dx * cfg.QUIVER_SIZE)),
                    int(np.round(y + dy * cfg.QUIVER_SIZE)),
                ),
                color=get_color_from_cv2_colormap_lut(magn / cfg.COLOR_DIV),
                thickness=2,
                tipLength=0.3,
            )

    if output_resolution != (background_img.shape[1], background_img.shape[0]):
        canvas = cv2.resize(
            canvas,
            output_resolution,
            interpolation=interpolation,
        )

    return np.asarray(canvas, dtype=np.uint8)
