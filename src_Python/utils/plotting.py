#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/2D-PIV-BOS"
__date__ = "13-11-2023"
__version__ = "1.0"

import sys

import init_config as cfg
from utils import numba_quivers

import numpy as np
import numpy.typing as npt
import numba as nb
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
) -> npt.NDArray[np.uint8]:
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

    Returns (``numpy.ndarray[np.uint8]``):
        Array containing BGR color values as `[color_idx][B, G, R]` with a shape
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

    return np.asarray(cv2_lut, order="C")


# ------------------------------------------------------------------------------
#   Module-level defined colormaps
# ------------------------------------------------------------------------------

this = sys.modules[__name__]

this.mpl_colormap = build_mpl_colormap(  # type: ignore
    mpl_colormap_name=cfg.COLORMAP_NAME,
    mpl_set_over=cfg.COLORMAP_CLIP_COLOR,
)

this.cv2_colormap_lut = build_cv2_colormap_lut(  # type: ignore
    mpl_colormap_name=cfg.COLORMAP_NAME,
    mpl_set_over=cfg.COLORMAP_CLIP_COLOR,
)

# ------------------------------------------------------------------------------
#   get_color(s)_from_cv2_colormap_lut
# ------------------------------------------------------------------------------


@nb.njit(
    cache=True,
    nogil=True,
)
def get_color_from_cv2_colormap_lut(value: float) -> npt.NDArray[np.uint8]:
    """Look up and return the BGR color value from the colormap as defined at
    this module level (`[plotting.py].cv2_colormap_lut`).

    Args:
        value (``float``):
            Normalized color lookup value.
                0.0 <= value <= 1.0: Colormap proper
                value < 0.0        : Out-of-range under color
                value > 1.0        : Out-of-range over color
                value == np.nan    : Set to (0, 0, 0) color

    Returns (``np.ndarray[uint8]``):
        The [uint8, uint8, uint8]-color value.
    """
    if np.isnan(value):
        return np.zeros(3, dtype=np.uint8)

    if value < 0.0:
        # Out-of-range under
        lut_idx = N_COLORS_LUT
    elif value > 1.0:
        # Out-of-range over
        lut_idx = N_COLORS_LUT + 1
    else:
        # Colormap proper
        lut_idx = int(np.round(value * (N_COLORS_LUT - 1)))

    return this.cv2_colormap_lut[lut_idx]


@nb.njit(
    cache=True,
    nogil=True,
)
def get_colors_from_cv2_colormap_lut(
    values: npt.NDArray[np.float32],
) -> npt.NDArray[np.uint8]:
    """Look up and return the BGR color values from the colormap as defined at
    this module level (`[plotting.py].cv2_colormap_lut`).

    Args:
        values (``numpy.ndarray[np.float32]``):
            Array containing normalized color lookup values.
                0.0 <= value <= 1.0: Colormap proper
                value < 0.0        : Out-of-range under color
                value > 1.0        : Out-of-range over color
                value == np.nan    : Set to (0, 0, 0) color

    Returns (``np.ndarray[uint8]``):
        Array containing [uint8, uint8, uint8]-color values.
    """
    colors = np.zeros((len(values), 3), dtype=np.uint8)
    for idx, value in enumerate(values):
        if not np.isnan(value):
            colors[idx] = get_color_from_cv2_colormap_lut(value)

    return colors


# ------------------------------------------------------------------------------
#   vector_map_to_hsv_colors
# ------------------------------------------------------------------------------


@nb.njit(
    (nb.types.Array(nb.uint8, 3, "C"))(
        nb.types.Array(nb.float32, 1, "C"),
        nb.types.Array(nb.float32, 1, "C"),
        nb.types.UniTuple(nb.int64, 2),
        nb.float32,
        nb.boolean,
    ),
    cache=True,
    nogil=True,
)
def _vector_map_to_hsv_colors(
    VM_magn: npt.NDArray[np.float32],
    VM_angle: npt.NDArray[np.float32],
    VM_grid_shape_2D: tuple[int, int],
    pixel_displacement_at_max_colormap_value: float,
    show_clipped_as_white: bool,
) -> npt.NDArray[np.uint8]:
    """Numba-accelerated core for function `vector_map_to_hsv_colors()`."""
    # NOTE: Argument `pixel_displacement_at_max_colormap_value` must be passed
    # in. We can not reference to `cfg.PIXEL_DISPLACEMENT_AT_MAX_COLORMAP_VALUE`
    # inside of this jitted function, because otherwise the reference /value/
    # gets baked in instead of the reference itself.
    VM_magn = np.nan_to_num(VM_magn)
    VM_angle = np.nan_to_num(VM_angle)

    # Check for 'out-of-range over' values
    HSV_sat = np.ones(VM_magn.shape, dtype=np.uint8) * 255
    HSV_val = VM_magn / pixel_displacement_at_max_colormap_value * 255
    mask = HSV_val > 255
    HSV_val[mask] = 255
    if show_clipped_as_white:
        HSV_sat[mask] = 0

    # Create a linearized HSV canvas and color it in
    canvas = np.empty((len(VM_magn), 3), dtype=np.uint8)
    canvas[:, 0] = np.floor(VM_angle / (360 / 179))  # Range [0, 179]
    canvas[:, 1] = HSV_sat
    canvas[:, 2] = HSV_val

    # Reshape linear canvas to 2D canvas
    canvas = np.reshape(canvas, (VM_grid_shape_2D[0], VM_grid_shape_2D[1], 3))

    return canvas


def vector_map_to_hsv_colors(
    VM_magn: npt.NDArray[np.float32],
    VM_angle: npt.NDArray[np.float32],
    VM_grid_shape_2D: tuple[int, int],
    output_resolution: tuple[int, int] | None = None,
    interpolation: int = cv2.INTER_CUBIC,
    show_clipped_as_white: bool = False,
) -> npt.NDArray[np.uint8]:
    """Generate and return an `uint8` RGB image where the passed vector data
    gets interpreted as an HSV image as follows:
        - The Hue        channel is set by the vector angles.
        - The Saturation channel is set at constant max value of 255.
        - The Value      channel is set by the vector magnitudes.

    Args:
        VM_magn (``numpy.ndarray[np.float32]``):
            Flattened array containing the vector magnitudes per IW.

        VM_angle (``numpy.ndarray[np.float32]``):
            Flattened array containing the vector angles in degrees per IW.

        VM_grid_shape_2D (``tuple[int, int]``):
            2D-shape of the grid corresponding to the above arrays, i.e.
            [N_IWs_y, N_IWs_x].

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

        show_clipped_as_white (``bool``, optional):
            TODO: descr.

            Default: False

    Returns:
        The RGB image as a 2D numpy array containing `[uint8, uint8, uint8]`
        RGB color values.
    """
    canvas = _vector_map_to_hsv_colors(
        VM_magn,
        VM_angle,
        VM_grid_shape_2D,
        cfg.PIXEL_DISPLACEMENT_AT_MAX_COLORMAP_VALUE,
        show_clipped_as_white,
    )
    cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR, dst=canvas)

    if output_resolution is not None and output_resolution != VM_grid_shape_2D:
        canvas = cv2.resize(
            canvas,
            output_resolution,
            interpolation=interpolation,
        )
        canvas = np.asarray(canvas, dtype=np.uint8)

    return canvas


# ------------------------------------------------------------------------------
#   vector_map_to_mpl_quiver_plot
# ------------------------------------------------------------------------------


def vector_map_to_mpl_quiver_plot(
    background_img: npt.NDArray[np.float32],
    VM_grid_x: npt.NDArray[np.int32],
    VM_grid_y: npt.NDArray[np.int32],
    VM_dx: npt.NDArray[np.float32],
    VM_dy: npt.NDArray[np.float32],
    VM_magn: npt.NDArray[np.float32],
    plot_title: str,
    show_clipped: bool = False,
):
    """ """
    self = vector_map_to_mpl_quiver_plot
    colormap = this.mpl_colormap

    VM_colors = VM_magn / cfg.PIXEL_DISPLACEMENT_AT_MAX_COLORMAP_VALUE
    if not show_clipped:
        VM_colors[VM_colors > 1] = 1  # Disable clip warning by clamping to 1

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
        self.h_title = plt.title(f"{plot_title}")

    self.h_imshow.set_data(background_img)
    self.h_quiver.set_UVC(VM_dx * cfg.QUIVER_SIZE, VM_dy * cfg.QUIVER_SIZE)
    self.h_quiver.set_color(colormap(VM_colors))
    self.h_title.set_text(f"{plot_title}")

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
    linewidth: int = 2,
    tip_size: float = 0.5,
    tip_angle: float = np.pi / 10,
    show_clipped: bool = False,
) -> npt.NDArray[np.uint8]:
    """ """
    if output_resolution is None:
        output_resolution = (background_img.shape[1], background_img.shape[0])

    canvas = cv2.cvtColor(background_img * 255, cv2.COLOR_GRAY2BGR)
    canvas = np.asarray(canvas, dtype=np.uint8)

    VM_colors = VM_magn / cfg.PIXEL_DISPLACEMENT_AT_MAX_COLORMAP_VALUE
    if not show_clipped:
        VM_colors[VM_colors > 1] = 1  # Disable clip warning by clamping to 1

    numba_quivers.draw_quiver_map_u24(
        img=canvas,
        x=VM_grid_x,
        y=VM_grid_y,
        dx=VM_dx * cfg.QUIVER_SIZE,
        dy=VM_dy * cfg.QUIVER_SIZE,
        colors=get_colors_from_cv2_colormap_lut(VM_colors),
        linewidth=linewidth,
        tip_size=tip_size,
        tip_angle=tip_angle,
    )

    if output_resolution != (background_img.shape[1], background_img.shape[0]):
        canvas = cv2.resize(
            canvas,
            output_resolution,
            interpolation=interpolation,
        )
        canvas = np.asarray(canvas, dtype=np.uint8)

    return canvas
