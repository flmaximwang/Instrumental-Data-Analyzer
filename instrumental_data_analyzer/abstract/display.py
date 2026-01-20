from dataclasses import dataclass
from matplotlib.colors import Colormap
from matplotlib.cm import get_cmap


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

    @property
    def cmap(self) -> Colormap:
        return self._cmap

    @cmap.setter
    def cmap(self, value: str | Colormap):
        if isinstance(value, str):
            self._cmap = get_cmap(value)
        elif isinstance(value, Colormap) or value is None:
            self._cmap = value
        else:
            raise ValueError("cmap must be any of str, Colormap or None")


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
