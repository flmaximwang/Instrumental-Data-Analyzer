import os, re, time, warnings
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import instrumental_data_analyzer.utils.path_utils as path_utils
from ..utils import transform_utils


@dataclass
class DescAnno:
    """
    Every signal contains multiple descriptions, which is continuous / discrete.
    This class is used to annotate each description for its
    type (like Volume), unit (like mL), limit (like (0, 10)) and ticks (like tick_number = 5)
    """

    name: str = None
    unit: str = None

    @property
    def label(self):
        if self.unit is None:
            return self.name
        else:
            return f"{self.name} ({self.unit})"


@dataclass
class ContDescAnno(DescAnno):
    """
    Continuous description annotation

    Properties
    ----------
    limit: tuple[float, float]
        The (min, max) limits for the continuous description.
    tick_number: int
        The number of ticks to be displayed on the axis.
    margin: tuple[float, float]
        The (lower_margin, upper_margin) as fractions of the total range.
    digits: int
        The number of decimal places for tick labels.
    """

    limit: tuple[float, float] = None
    margin: tuple[float, float] = None
    ticklabel_floor: float = None
    _ticklabel_space: float = None
    ticklabel_digits: int = 0

    @property
    def ticklabel_space(self):
        return self._ticklabel_space

    @ticklabel_space.setter
    def ticklabel_space(self, value):
        self._ticklabel_space = value
        if value > 1:
            self.ticklabel_digits = 0
        elif value > 0.1:
            self.ticklabel_digits = 1
        elif value > 0.01:
            self.ticklabel_digits = 2
        elif value > 0.001:
            self.ticklabel_digits = 3
        else:
            self.ticklabel_digits = 4

    @property
    def ticklabels(self):
        assert self.limit is not None, "limit is not set"
        assert self.ticklabel_space is not None, "ticklabel_space is not set"
        assert self.ticklabel_digits is not None, "ticklabel_digits is not set"

        if self.ticklabel_floor is None:
            ticklabel_floor = self.limit[0]
        else:
            ticklabel_floor = self.ticklabel_floor

        numbers = []
        c_number = ticklabel_floor
        while c_number <= self.limit[1]:
            numbers.append(c_number)
            c_number += self.ticklabel_space

        strings = [f"{num:.{self.ticklabel_digits}f}" for num in numbers]
        return strings

    @property
    def ticks(self):
        assert self.margin is not None, "margin is not set"

        ticklabels = np.array(list(map(float, self.ticklabels)))
        ticks = (ticklabels - self.limit[0]) / (self.limit[1] - self.limit[0]) * (
            self.margin[1] - self.margin[0]
        ) + self.margin[0]
        return ticks


@dataclass
class DiscDescAnno(DescAnno):

    limit: tuple[float, float] = (0, 1)
    tick_number: int = 0
    ticks: list = field(default_factory=lambda: [0])
    ticklabels: list = field(default_factory=lambda: [""])


@dataclass
class Signal:
    """
    a signal is composed of multiple descriptions.
    At least one description is continuous for all instruments.
    These descriptions are represented by columns in a pandas.DataFrame and therefore have equal lengths.
    More specific signals are subclasses that have defined descriptions.

    Properties
    ----------
    data: pd.DataFrame
        A pandas DataFrame containing the signal data. Each column represents a different description.
    name: str
        The name of the signal.
    description_annotations: list[DescAnno]
        A list of DescAnno objects that annotate each description in the data.
    """

    data: pd.DataFrame = None
    name: str = None
    description_annotations: list[DescAnno] = field(default_factory=list)

    @staticmethod
    def get_type_and_unit_from_header(header):
        """
        Returns type and unit from header string
        """
        type_unit_pattern = re.match(r"(.*) \((.*)\)", header)
        if type_unit_pattern is None:
            # No unit found
            return header, None
        else:
            return type_unit_pattern.group(1), type_unit_pattern.group(2)

    @classmethod
    def from_csv(cls, path, name=None, **kwargs):
        """
        See pandas.read_csv for supported kwargs
        """
        if name is None:
            name = path_utils.get_name_from_path(path)
        data = pd.read_csv(path, **kwargs)
        return cls(data, name)

    @staticmethod
    def rescale(
        array: np.ndarray, limit: tuple[float, float], margin: tuple[float, float]
    ):
        """
        Rescale [lower_limit, upper_limit] to [lower_margin, upper_margin] linearly.

        Parameters
        ----------
        array : np.ndarray
        limit : tuple[float, float]
            A tuple specifying the (min, max) limits for rescaling.
        margin : tuple[float, float]
            A tuple specifying the (lower_margin, upper_margin) as fractions of the total range.
        """

        scaled_to_0_1 = transform_utils.rescale_to_0_1(array, limit[0], limit[1])
        scaled_to_margin_range = (margin[1] - margin[0]) * scaled_to_0_1 + margin[0]
        return scaled_to_margin_range

    def to_csv(self, csv: str | Path):
        """
        Export signal to export_path
        """
        csv = Path(csv)
        directory = csv.parent

        if not directory.exists():
            directory.mkdir(parents=True)
        if csv.exists():
            time_stamp = time.strftime("%Y%m%d_%H%M%S")
            csv = csv.with_name(f"{csv.stem}_{time_stamp}{csv.suffix}")
            warnings.warn(f"File {csv} already exists. Renamed to {csv.name}.")
        columns = []
        for desc_anno in self.description_annotations:
            columns.append(desc_anno.label)
        self.data.columns = columns
        self.data.to_csv(csv, index=False)
