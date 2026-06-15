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

from typing import Callable, Iterable
from dataclasses import dataclass
from matplotlib.colors import Colormap
from matplotlib import colormaps


@dataclass
class SignalPlotArgs:
    """
    PlotArgs is used to store the arguments for plotting a signal collection.
    It contains the following properties:

    Properties
    ----------
    figsize: tuple[float, float] | None
        The size of the figure. If None, the default size will be used.
    axis_shift: float | None
        The shift in mode 1
    colormap: str
        The colormap to be used for plotting. Default is 'default'.
    colormap_min: float
        The minimum value for the colormap. Default is 0.
    colormap_max: float
        The maximum value for the colormap. Default is 1.
    """

    mode = 0
    axis_shift = 0.2
    figsize: tuple[float, float] = None
    _cmap: Colormap = None
    cmap_limit: tuple[float, float] = (0, 1)
    legend_cols: int = 1
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


@dataclass
class Signal1DPlotArgs(SignalPlotArgs):
    """
    Properties
    ----------
    mode: int
        0: Plot with collection annotations
        1: Plot with all signal annotations
        2: Plot in separate subplots
    """

    mode: int = 0
