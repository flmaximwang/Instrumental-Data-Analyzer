from dataclasses import dataclass, field
from ..abstract import ContinuousSignal1D, ContinuousSignal1DCollection, ContDescAnno


@dataclass
class AbsorbSpec(ContinuousSignal1D):

    pass


@dataclass
class AbsorbSpecCollection(ContinuousSignal1DCollection):

    pass
