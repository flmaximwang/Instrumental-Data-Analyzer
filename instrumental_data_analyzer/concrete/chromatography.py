from dataclasses import dataclass, field
from ..abstract import (
    Signal1D,
    ContinuousSignal1D,
    DiscreteSignal1D,
    Signal1DCollection,
    Signal1DPlotArgs,
)


@dataclass
class ChromPlotArgs(Signal1DPlotArgs):

    figsize: tuple[float, float] = (20, 6)
    mode: int = 1


@dataclass
class ChromSig(ContinuousSignal1D):
    pass


@dataclass
class ChromLog(DiscreteSignal1D):
    pass


@dataclass
class Chrom(Signal1DCollection):

    signal_type: type = ChromSig | ChromLog
    plot_args: ChromPlotArgs = field(default_factory=ChromPlotArgs)

    def correct_conc(self, conc_delay):
        """
        校正浓度信号, 因为浓度信号和实际的盐浓度有一定的延迟
        """
        data = self["Conc B"].get_data()
        data.iloc[:, 0] += conc_delay

    def get_signal(
        self, signal_name: str
    ) -> Signal1D | ContinuousSignal1D | DiscreteSignal1D:
        return super().get_signal(signal_name)
