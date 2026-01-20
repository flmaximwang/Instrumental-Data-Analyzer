"""
Matrices are 2D arrays of values
"""

from dataclasses import dataclass, field
import pandas as pd
from .signal import DescAnno


@dataclass
class Matrix:

    name: str = None
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    description_annotations: list[DescAnno] = field(default_factory=list)

    @classmethod
    def from_csv(cls, csv_file: str, header=None, index_col=None, sep=","):
        return cls(pd.read_csv(csv_file, header=header, index_col=index_col, sep=sep))

    @property
    def loc(self):
        return self.data.loc

    @property
    def iloc(self):
        return self.data.iloc
