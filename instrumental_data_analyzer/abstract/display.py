"""
``instrumental_data_analyzer.abstract.display`` --- 绘图参数配置
================================================================

提供预定义的绘图参数 dataclass:

- :class:`SignalPlotArgs` : 通用绘图参数 (图大小、配色方案、图例)
- :class:`Signal1DPlotArgs` : 1D 信号专用绘图参数 (+ 绘图模式)

绘图模式 (mode):
  0: 所有信号共享坐标轴, 叠加显示 (适合同量纲信号比较)
  1: 共享 x 轴, 每个信号独立 y 轴 (多轴对比, 适合不同量纲信号)
  2: 每个信号绘制在单独的子图中 (分面绘图)
"""

from typing import Callable, Iterable, Annotated
from dataclasses import dataclass
from matplotlib.colors import Colormap
from matplotlib import colormaps


@dataclass(slots=True)
class SignalPlotArgs:
    """General plotting parameters for signal collections.

    Parameters
    ----------
    axis_shift : float
        Normalised spacing between twin y-axis spines in mode 1.
    ax_size : tuple[float, float] | None
        Size of the Axes data area (between spines) *in centimetres*.
        When set, the Figure is sized as *ax_size* + margins for labels,
        legend, and title.  When ``None``, matplotlib default sizing is used.

        .. note::

            ``ax_size`` controls the **Axes**, not the Figure.  The Figure
            is roughly ``ax_size + (8 cm, 2.5 cm)`` for the extra space
            taken by y-labels, legend, and title.  For a ~25 × 13 cm
            Figure, try ``ax_size = (17, 10)``.
    cmap : str | Colormap | None
        Colour map for signal lines.  See :attr:`cmap` setter.
    cmap_limit : tuple[float, float]
        Value range mapped to the colour map.
    legend_cols : int
        Number of columns in the legend.
    legend_bbox_to_anchor : tuple[float, float]
        Bounding-box anchor for the legend, in Axes-normalised coordinates.
    """

    #: Normalised gap between twin y-axis spines (mode 1)
    axis_shift: float = 0.2
    #: Axes data-area size (width, height) in cm.  None → matplotlib default.
    ax_size: Annotated[tuple[float, float] | None, "cm"] = None
    _cmap: Colormap = None
    #: (min, max) for colour-map value range
    cmap_limit: tuple[float, float] = (0, 1)
    #: Number of legend columns
    legend_cols: int = 1
    #: Legend anchor in Axes-normalised coordinates (x, y)
    legend_bbox_to_anchor: tuple[float, float] = (1.05, 1)

    @property
    def cmap(self) -> Colormap:
        return self._cmap

    @cmap.setter
    def cmap(self, value: str | Colormap):
        if isinstance(value, str):
            self._cmap = colormaps.get_cmap(value)
        elif isinstance(value, Colormap) or value is None:
            self._cmap = value
        elif isinstance(value, Callable):
            self._cmap = value
        elif isinstance(value, Iterable):
            self._cmap = value
        else:
            raise ValueError(
                "cmap must be any of str, Colormap, Callable, Iterable or None"
            )


@dataclass(slots=True)
class Signal1DPlotArgs(SignalPlotArgs):
    """1D signal plotting parameters.

    Parameters
    ----------
    mode : int
        - 0 = overlaid (all signals share one pair of axes);
        - 1 = twin y-axes (share x-axis, each signal gets its own y-axis);
        - 2 = faceted (each signal in its own subplot).
    """

    #: 0=overlaid, 1=twin y-axes, 2=faceted
    mode: int = 0
