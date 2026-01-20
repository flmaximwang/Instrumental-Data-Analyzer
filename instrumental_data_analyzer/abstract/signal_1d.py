import typing
from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .signal import *
from ..utils import path_utils, transform_utils

chromatographic_units_map_types = {
    "min": "Time",
    "mL": "Volume",
    "ml": "Volume",
    "mAU": "Absorbance",
}

spectroscopic_units_map_types = {
    "nm": "Wavelength",
    "": "Absorbance",
}

units_map_types = [chromatographic_units_map_types, spectroscopic_units_map_types]


class _Interpolate:

    def __init__(self, axis: typing.Iterable[float], value: typing.Iterable[float]):
        self.axis = np.array(axis)
        self.value = np.array(value)

    def __getitem__(self, axis_i: list[int, float] | int | float | slice):
        """
        Get the value at the given axis value using 1D interpolation.
        """
        if isinstance(axis_i, slice):
            axis_i = list(range(axis_i.start, axis_i.stop, axis_i.step))
        return np.interp(axis_i, self.axis, self.value)


@dataclass
class Signal1D(Signal):
    axis_annotation_type = ContDescAnno
    value_annotation_type = DescAnno

    """
    1D signals are signals with two descriptions.

    - The first is called axis (which is continuous),
    - The second is called value (which may be continuous or discrete).

    Value describes the property of every point on the axis, like the absorbance of a sample at a certain wavelength.

    Properties
    ----------
    axis_name: str
        The name of the axis description, like "Time", "Volume", "Wavelength".
    axis_unit: str
        The unit of the axis description, like "min", "mL", "nm".
    value_name: str
        The name of the value description, like "Absorbance", "Intensity".
    value_unit: str
        The unit of the value description, like "mAU", "a.u.".
    """

    @property
    def axis(self):
        return self.data.iloc[:, 0]

    @axis.setter
    def axis(self, new_axis):
        self.data.iloc[:, 0] = new_axis

    @property
    def axis_annotation(self):
        axis_annotation: ContDescAnno = self.description_annotations[0]
        return axis_annotation

    @axis_annotation.setter
    def axis_annotation(self, new_axis_annotation):
        self.description_annotations[0] = new_axis_annotation

    @property
    def value_annotation(self):
        value_annotation: DescAnno = self.description_annotations[1]
        return value_annotation

    @value_annotation.setter
    def value_annotation(self, new_value_annotation):
        self.description_annotations[1] = new_value_annotation

    @property
    def value(self):
        return self.data.iloc[:, 1]

    @value.setter
    def value(self, new_value):
        self.data.iloc[:, 1] = new_value

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        axis_name=None,
        axis_unit=None,
        value_name=None,
        value_unit=None,
        detect_axis_name_and_unit=False,
        detect_value_name_and_unit=False,
    ):
        """
        Given a dataframe with two/three columns, create a Signal1D object.
        """
        if detect_axis_name_and_unit:
            axis_name_temp, axis_unit_temp = cls.get_type_and_unit_from_header(
                data.columns[0]
            )
        else:
            axis_name_temp = data.columns[0]
            axis_unit_temp = None
        if not axis_name and axis_name_temp:
            axis_name = axis_name_temp
        if not axis_unit and axis_unit_temp:
            axis_unit = axis_unit_temp

        if detect_value_name_and_unit:
            value_name_temp, value_unit_temp = cls.get_type_and_unit_from_header(
                data.columns[1]
            )
        else:
            value_name_temp = data.columns[1]
            value_unit_temp = None
        if not value_name and value_name_temp:
            value_name = value_name_temp
        if not value_unit and value_unit_temp:
            value_unit = value_unit_temp

        description_annotations = [
            ContDescAnno(
                name=axis_name,
                unit=axis_unit,
            ),
            ContDescAnno(
                name=value_name,
                unit=value_unit,
            ),
        ]
        res = cls(data=data, name=name, description_annotations=description_annotations)
        res.update_data_headers()
        return res

    @classmethod
    def from_csv(cls, csv: str | Path):
        csv_path = Path(csv)
        data = pd.read_csv(csv_path)
        name = csv_path.stem
        result = cls.from_data(
            data=data,
            detect_axis_name_and_unit=True,
            detect_value_name_and_unit=True,
        )
        result.name = name
        return result

    def slice_axis(self, left_limit: float, right_limit: float, include_limit=True):
        res = self.data.iloc[:, 0][
            (self.data.iloc[:, 0] >= left_limit) & (self.data.iloc[:, 0] <= right_limit)
        ].to_numpy()
        if include_limit:
            if left_limit < min(res):
                res = np.concatenate([[left_limit], res])
            if right_limit > max(res):
                res = np.concatenate([res, [right_limit]])
        return res

    def update_data_headers(self):
        old_columns = self.data.columns
        new_columns = [self.axis_annotation.label, self.value_annotation.label]
        self.data.rename(columns=dict(zip(old_columns, new_columns)), inplace=True)

    def update_type_and_unit_from_data(
        self, detect_axis_name_and_unit=True, detect_value_name_and_unit=True
    ):
        if detect_axis_name_and_unit:
            axis_name, axis_unit = self.get_type_and_unit_from_header(
                self.data.columns[0]
            )
        else:
            axis_name, axis_unit = self.data.columns[0], None
        if detect_value_name_and_unit:
            value_name, value_unit = self.get_type_and_unit_from_header(
                self.data.columns[1]
            )
        else:
            value_name, value_unit = self.data.columns[1], None
        self.axis_annotation.name = axis_name
        self.axis_annotation.unit = axis_unit
        self.value_annotation.name = value_name
        self.value_annotation.unit = value_unit

    def plot_at(self, ax: plt.Axes, label=None, **kwargs):
        """
        The method to plot a signal should only be implemented when the form of signal has been well defined.
        Such a method should retrun a Line2D
        """
        (handle,) = ax.plot([0], [0], label=label)
        return handle


@dataclass
class ContinuousSignal1D(Signal1D):
    """
    ContinuousSignal1D 是 Signal1D 的子类, 用于表示 axis 与 value 都是连续的 Signal1D
    """

    value_annotation_type = ContDescAnno

    @property
    def value_std(self):
        try:
            return self.data.iloc[:, 2]
        except IndexError:
            return None

    @property
    def value_annotation(self):
        value_annotation: ContDescAnno = self.description_annotations[1]
        return value_annotation

    @property
    def value_limit(self):
        return self.value_annotation.limit

    @value_limit.setter
    def value_limit(self, new_value_limit):
        self.value_annotation.limit = new_value_limit

    @property
    def value_margin(self):
        return self.value_annotation.margin

    @value_margin.setter
    def value_margin(self, new_value_margin):
        self.value_annotation.margin = new_value_margin

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        axis_name=None,
        axis_unit=None,
        value_name=None,
        value_unit=None,
        detect_axis_name_and_unit=False,
        detect_value_name_and_unit=False,
    ):
        """
        Given a dataframe with two/three columns, create a Signal1D object.
        """
        if detect_axis_name_and_unit:
            axis_name_temp, axis_unit_temp = cls.get_type_and_unit_from_header(
                data.columns[0]
            )
        else:
            axis_name_temp = data.columns[0]
            axis_unit_temp = None
        if not axis_name and axis_name_temp:
            axis_name = axis_name_temp
        if not axis_unit and axis_unit_temp:
            axis_unit = axis_unit_temp

        if detect_value_name_and_unit:
            value_name_temp, value_unit_temp = cls.get_type_and_unit_from_header(
                data.columns[1]
            )
        else:
            value_name_temp = data.columns[1]
            value_unit_temp = None
        if not value_name and value_name_temp:
            value_name = value_name_temp
        if not value_unit and value_unit_temp:
            value_unit = value_unit_temp

        description_annotations = [
            ContDescAnno(
                name=axis_name,
                unit=axis_unit,
                limit=(min(data.iloc[:, 0]), max(data.iloc[:, 0])),
                margin=(0, 1),
            ),
            ContDescAnno(
                name=value_name,
                unit=value_unit,
                limit=(min(data.iloc[:, 1]), max(data.iloc[:, 1])),
                margin=(0, 1),
            ),
        ]
        res = cls(data=data, description_annotations=description_annotations)
        res.update_data_headers()
        return res

    def __getitem__(self, axis_i: list[int, float] | int | float | slice):
        """
        Get the value at the given axis value using 1D interpolation.
        """
        interpolater = _Interpolate(self.axis, self.value)
        return interpolater[axis_i]

    @property
    def std(self):
        return _Interpolate(self.axis, self.value_std)

    @classmethod
    def average(
        cls,
        signals: list["ContinuousSignal1D"],
    ):
        """
        Averaging multiple ContinuousSignal1D signals into one ContinuousSignal1D signal.
        An extra std column will be added to the data to represent the standard deviation of the values at the same axis.
        """
        list_of_axes = [signal.axis for signal in signals]
        merged_axis = np.unique(np.concatenate(list_of_axes))
        merged_axis.sort()
        merged_value = np.zeros_like(merged_axis)
        value_std = np.zeros_like(merged_axis)
        for i, axis_i in enumerate(merged_axis):
            values = [signal[axis_i] for signal in signals]
            merged_value[i] = np.mean(values)
            value_std[i] = np.std(values)
        axis_name = signals[0].axis_annotation.name
        axis_unit = signals[0].axis_annotation.unit
        value_name = signals[0].value_annotation.name
        value_unit = signals[0].value_annotation.unit
        merged_data = pd.DataFrame(
            {"Axis": merged_axis, "Values": merged_value, "Value_std": value_std}
        )
        return cls.from_data(
            merged_data,
            axis_name=axis_name,
            axis_unit=axis_unit,
            value_name=value_name,
            value_unit=value_unit,
        )

    def __add__(self, the_other, name=None):
        if not isinstance(the_other, ContinuousSignal1D):
            raise TypeError(
                "Only ContinuousSignal1D can be added to ContinuousSignal1D"
            )
        if (
            self.axis_annotation.name != the_other.axis_annotation.name
            or self.axis_annotation.unit != the_other.axis_annotation.unit
            or self.value_annotation.name != the_other.value_annotation.name
            or self.value_annotation.unit != the_other.value_annotation.unit
        ):
            raise ValueError(
                "Only ContinuousSignal1D with the same axis and value name and unit can be added"
            )
        list_of_axes = [self.axis, the_other.axis]
        merged_axis = np.unique(np.concatenate(list_of_axes))
        merged_axis.sort()
        merged_value = np.zeros_like(merged_axis)
        for i, axis_i in enumerate(merged_axis):
            merged_value[i] = self[axis_i] + the_other[axis_i]
        merged_data = pd.DataFrame({"Axis": merged_axis, "Values": merged_value})
        result = ContinuousSignal1D.from_data(
            merged_data,
            axis_name=self.axis_annotation.name,
            axis_unit=self.axis_annotation.unit,
            value_name=self.value_annotation.name,
            value_unit=self.value_annotation.unit,
        )
        result.name = f"({self.name}) + ({the_other.name})" if name is None else name
        return result

    def __neg__(self, name=None):
        negated_data = self.data.copy()
        negated_data.iloc[:, 1] = -negated_data.iloc[:, 1]
        if self.value_std is not None:
            negated_data.iloc[:, 2] = self.value_std
        return type(self)(
            data=negated_data,
            name=f"-({self.name})" if name is None else name,
            description_annotations=deepcopy(self.description_annotations),
        )

    def __sub__(self, the_other, name=None):
        return self.__add__(-the_other, name=name)

    def blank_with(self, blank, name=None):
        return self.__sub__(
            blank, name=f"{self.name} blanked" if name is None else name
        )

    def rescale_between(self, target_0, target_1, inplace=False):
        values = (self.get_values() - target_0) / (target_1 - target_0)
        if inplace:
            self.set_values(values)
        return values

    def get_peak_between(self, axis_left, axis_right) -> tuple[float, float]:
        """
        Get the peak (axis, value) between the given axis range.
        """
        sliced_axis = self.slice_axis(axis_left, axis_right)
        if len(sliced_axis) == 0:
            raise ValueError("No data in the given range")
        sliced_value: np.ndarray = self[sliced_axis]
        peak_idx = sliced_value.argmax()
        peak_axis = sliced_axis[peak_idx]
        peak_value = sliced_value[peak_idx]
        return (peak_axis, peak_value)

    def plot_at(self, ax: plt.Axes, **kwargs):
        """
        Plot the signal at the given ax and return an artist handle.
        You can provide extra arguments to the plot function by **kwargs. Available arguments are listed below:
        - color: str
        - linewidth: float
        - linestyle: str
        - label: str
        - alpha: float
        - marker: str
        - horizontalalignment: center | right | left
        - verticalalignment: center | top | bottom
        """
        kwargs.pop("text_shift", None)
        label = kwargs.pop("label", None)
        if not label:
            label = self.name
        kwargs_for_annotate = kwargs.copy()
        kwargs_for_Line2D = kwargs.copy()

        kwargs_for_Line2D.pop("fontsize", None)
        kwargs_for_Line2D.pop("rotation", None)

        assert self.axis_annotation.limit is not None, "Axis limit is not set"
        assert self.axis_annotation.margin is not None, "Axis margin is not set"
        assert self.value_annotation.limit is not None, "Value limit is not set"
        assert self.value_annotation.margin is not None, "Value margin is not set"
        (handle,) = ax.plot(
            self.rescale(
                self.axis,
                self.axis_annotation.limit,
                self.axis_annotation.margin,
            ),
            self.rescale(
                self.value, self.value_annotation.limit, self.value_annotation.margin
            ),
            label=label,
            **kwargs_for_Line2D,
        )
        if self.value_std is not None:
            ax.fill_between(
                self.rescale(
                    self.axis,
                    self.axis_annotation.limit,
                    self.axis_annotation.margin,
                ),
                self.rescale(
                    self.value - self.value_std,
                    self.value_annotation.limit,
                    self.value_annotation.margin,
                ),
                self.rescale(
                    self.value + self.value_std,
                    self.value_annotation.limit,
                    self.value_annotation.margin,
                ),
                alpha=0.2,
                **kwargs_for_annotate,
            )
        return handle

    def plot_peak_at(
        self,
        ax: plt.Axes,
        axis_left: float,
        axis_right: float,
        type="vline",
        text_shift=(0, 0),
        text_digits=2,
        **kwargs,
    ):
        """
        type can be "vline" or "annotation"
        You can provide extra arguments to the plot function by **kwargs. Available arguments are listed below:
        - color: str
        - linewidth: float
        - linestyle: str
        - label: str
        - alpha: float
        - marker: str
        - horizontalalignment: center | right | left
        - verticalalignment: center | top | bottom
        """
        peak_axis, peak_value = self.get_peak_between(axis_left, axis_right)
        axis_limit = self.axis_annotation.limit
        value_limit = self.value_annotation.limit
        rescaled_peak_axis = transform_utils.rescale_to_0_1(
            peak_axis, axis_limit[0], axis_limit[1]
        )
        rescaled_peak_value = transform_utils.rescale_to_0_1(
            peak_value, value_limit[0], value_limit[1]
        )
        if type == "vline":
            self._plot_peak_at_with_vline(
                ax,
                rescaled_peak_axis=rescaled_peak_axis,
                rescaled_peak_value=rescaled_peak_value,
                peak_axis=peak_axis,
                peak_value=peak_value,
                text_shift=text_shift,
                text_digits=text_digits,
                **kwargs,
            )
        elif type == "annotation":
            self._plot_peak_at_with_annotation(
                ax,
                peak_axis=peak_axis,
                peak_value=peak_value,
                rescaled_peak_axis=rescaled_peak_axis,
                rescaled_peak_value=rescaled_peak_value,
                text_shift=text_shift,
                **kwargs,
            )

    def _plot_peak_at_with_vline(
        self,
        ax: plt.Axes,
        rescaled_peak_axis,
        rescaled_peak_value,
        peak_axis,
        peak_value,
        text_shift,
        text_digits=2,
        **kwargs,
    ):
        linestyle = kwargs.pop("linestyle", "dashed")
        ax.vlines(
            rescaled_peak_axis, 0, rescaled_peak_value, linestyles=linestyle, **kwargs
        )
        ax.annotate(
            f"{peak_axis:.{text_digits}f} {self.axis_annotation.unit}: {peak_value:.{text_digits}f}",
            xy=(rescaled_peak_axis, rescaled_peak_value),
            xytext=(
                rescaled_peak_axis + text_shift[0],
                rescaled_peak_value + text_shift[1],
            ),
            **kwargs,
        )

    def _plot_peak_at_with_annotation(
        self,
        ax: plt.Axes,
        rescaled_peak_axis,
        rescaled_peak_value,
        peak_axis,
        peak_value,
        text_shift,
        **kwargs,
    ):
        ax.annotate(
            f"{peak_axis:.2f} {self.axis_annotation.unit}",
            xy=(rescaled_peak_axis, rescaled_peak_value),
            xytext=(
                rescaled_peak_axis + text_shift[0],
                rescaled_peak_value + text_shift[1],
            ),
            **kwargs,
        )

    def integrate_between(self, start, end):
        signal_data: pd.DataFrame = self.data
        signal_data = signal_data[
            (signal_data.iloc[:, 0] >= start) & (signal_data.iloc[:, 0] <= end)
        ]

        if not isinstance(baseline, typing.Iterable):
            baseline = np.array([baseline for _ in range(len(signal_data))])
        else:
            if len(baseline) != len(signal_data):
                print("Baseline length should be equal to the length of signal data")
                return None
            else:
                baseline = np.array(baseline)

        signal_height = signal_data.iloc[:, 1] - baseline
        peak_area = np.trapz(signal_height, signal_data.iloc[:, 0])

        if ax:
            ax.vlines(
                [start, end],
                0,
                1,
                colors=color,
                linestyles=linestyles,
                linewidths=linewidths,
                alpha=max(1, alpha * 2),
            )
            ax.fill_between(
                signal_data.iloc[:, 0],
                rescale_signal(
                    baseline, self.y_limits[signal][0], self.y_limits[signal][1]
                ),
                rescale_signal(
                    signal_data.iloc[:, 1].copy(),
                    self.y_limits[signal][0],
                    self.y_limits[signal][1],
                ),
                color=color,
                alpha=alpha,
            )
        print(
            "Peak area = {} {}·ml from {} ml to {} ml".format(
                peak_area, signal_data.columns[1], start, end
            )
        )

        return peak_area


@dataclass
class DiscreteSignal1D(Signal1D):
    """
    DiscreteSignal1D represents a Signal1D with continuous axis but discrete value.
    """

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        name: str,
        axis_name=None,
        axis_unit=None,
        value_name=None,
        value_unit=None,
        detect_axis_name_and_unit=False,
        detect_value_name_and_unit=False,
    ):
        """
        Given a dataframe with two/three columns, create a Signal1D object.
        """
        if detect_axis_name_and_unit:
            axis_name_temp, axis_unit_temp = cls.get_type_and_unit_from_header(
                data.columns[0]
            )
        else:
            axis_name_temp = data.columns[0]
            axis_unit_temp = None
        if not axis_name and axis_name_temp:
            axis_name = axis_name_temp
        if not axis_unit and axis_unit_temp:
            axis_unit = axis_unit_temp

        if detect_value_name_and_unit:
            value_name_temp, value_unit_temp = cls.get_type_and_unit_from_header(
                data.columns[1]
            )
        else:
            value_name_temp = data.columns[1]
            value_unit_temp = None
        if not value_name and value_name_temp:
            value_name = value_name_temp
        if not value_unit and value_unit_temp:
            value_unit = value_unit_temp

        description_annotations = [
            ContDescAnno(
                name=axis_name,
                unit=axis_unit,
                limit=(min(data.iloc[:, 0]), max(data.iloc[:, 0])),
                margin=(0, 1),
            ),
            DescAnno(
                name=value_name,
                unit=value_unit,
            ),
        ]
        res = cls(data=data, name=name, description_annotations=description_annotations)
        res.update_data_headers()
        return res

    @classmethod
    def from_csv(
        cls,
        path,
        name=None,
        detect_axis_name_and_unit=True,
        detect_value_name_and_unit=False,
        axis_name=None,
        axis_unit=None,
        value_name="Fraction",
        value_unit=None,
        **kwargs,
    ):
        return super().from_csv(
            path,
            name,
            detect_axis_name_and_unit=detect_axis_name_and_unit,
            detect_value_name_and_unit=detect_value_name_and_unit,
            axis_name=axis_name,
            axis_unit=axis_unit,
            value_name=value_name,
            value_unit=value_unit,
            **kwargs,
        )

    # def __init__(
    #     self,
    #     data,
    #     name,
    #     axis_name=None,
    #     axis_unit=None,
    #     value_name=None,
    #     value_unit=None,
    #     update="to_data",
    #     detect_axis_name_and_unit=False,
    #     detect_value_name_and_unit=False,
    # ):
    #     super().__init__(
    #         data=data,
    #         name=name,
    #         axis_name=axis_name,
    #         axis_unit=axis_unit,
    #         value_name=value_name,
    #         value_unit=value_unit,
    #         update=update,
    #         detect_axis_name_and_unit=detect_axis_name_and_unit,
    #         detect_value_name_and_unit=detect_value_name_and_unit,
    #     )
    #     self.description_annotations = [
    #         ContDescAnno(
    #             self.get_axis_name(),
    #             self.get_axis_unit(),
    #             (self.get_axis().min(), self.get_axis().max()),
    #             10,
    #         ),
    #         DiscDescAnno(self.get_value_name(), self.get_value_unit()),
    #     ]
    #     self.arrowprops = {}

    def plot_at(
        self, ax: plt.Axes, label=None, text_shift=(0, 0), mark_height=0.5, **kwargs
    ):
        """
        Plot the signal at the given ax and return an artist handle.
        You can provide extra arguments to the plot function by **kwargs. Available arguments are listed below:
        - color: str
        - linewidth: float
        - linestyle: str
        - label: str
        - alpha: float
        - marker: str
        - horizontalalignment: center | right | left
        - verticalalignment: center | top | bottom
        """
        if not label:
            label = self.name

        kwargs_for_annotate = kwargs.copy()
        if not "rotation" in kwargs_for_annotate.keys():
            kwargs_for_annotate["rotation"] = 90
        if not "fontsize" in kwargs_for_annotate.keys():
            kwargs_for_annotate["fontsize"] = 10

        kwargs_for_Line2D = kwargs.copy()
        kwargs_for_Line2D.pop("rotation", None)
        kwargs_for_Line2D.pop("fontsize", None)
        axis_to_plot = transform_utils.rescale_to_0_1(
            self.axis, self.axis_annotation.limit[0], self.axis_annotation.limit[1]
        )
        ax.vlines(
            axis_to_plot, 0, mark_height, linestyles="dashed", **kwargs_for_Line2D
        )
        for axis, value in zip(axis_to_plot, self.value):
            ax.annotate(
                f"{value}",
                xy=(axis, mark_height),
                xytext=(axis + text_shift[0], 0.5 + text_shift[1]),
                ha="center",
                **kwargs_for_annotate,
            )

        handle = plt.Line2D(
            [0], [0], label=label, **kwargs_for_Line2D
        )  # Generate a virtual handle for legend
        return handle

    def set_arrowprops(self, arrowprops):
        """
        Set the arrowprops for annotations. See ax.annotate for supported arrowprops
        """
        self.arrowprops = arrowprops


@dataclass
class SegmentedSignal1D(Signal1D):

    # def __init__(self, **kwargs):
    #     """
    #     A segmented signal 1d should look like\n
    #     Axis,Values,Segment,Others\n
    #     0.2,0.2,1,1\n
    #     0.22,0.3,1,2\n
    #     ...\n
    #     0.4,0.4,1,9\n
    #     0.38,0.38,2,10\n
    #     ...\n
    #     """

    #     self.active_segments = "all"

    def get_segment_num(self):
        return set(self.data["Segment"])

    def get_active_segments(self):
        return self.active_segments

    def get_indices_for_segments(self, segments):
        if segments == "all":
            return self.data.index
        elif isinstance(segments, int):
            return self.data.index[self.data["Segment"] == segments]
        elif isinstance(segments, list):
            return self.data.index[self.data["Segment"].isin(segments)]
        else:
            raise ValueError(
                f"Segments should be 'all', int or list of int, but got {segments}, which is {type(segments)}"
            )

    def get_indices_for_active_segments(self):
        return self.get_indices_for_segments(self.get_active_segments())

    def set_active_segments(self, segments):
        if isinstance(segments, str):
            if segments != "all":
                raise ValueError("segments should be 'all', int or list of int")
        elif isinstance(segments, int):
            if segments < 0 or segments > max(self.data["Segment"]):
                raise ValueError(
                    "segments should be positive int and less than the max segment number"
                )
        elif isinstance(segments, list):
            if any([x < 0 or x > max(self.data["Segment"]) for x in segments]):
                raise ValueError(
                    "segments should be positive int and less than the max segment number"
                )
        else:
            raise ValueError("segments should be 'all', int or list of int")
        self.active_segments = segments

    def get_data_for_segments(self, segments):
        # return self.data.loc[]
        if segments == "all":
            return self.data
        elif isinstance(segments, int):
            return self.data[self.data["Segment"] == segments]
        elif isinstance(segments, list):
            return self.data[self.data["Segment"].isin(segments)]
        else:
            raise ValueError(
                f"Segments should be 'all', int or list of int, but got {segments}, which is {type(segments)}"
            )

    def get_data_for_active_segments(self):
        return self.get_data_for_segments(self.get_active_segments())

    def get_series_for_segements(self, segments, i):
        """
        i = 0 for axis, i = 1 for values
        """
        segement_data = self.get_data_for_segments(segments)
        return segement_data.iloc[:, i]

    def get_axis_for_segments(self, segments):
        return self.get_series_for_segements(segments, 0)

    def get_values_for_segments(self, segments):
        return self.get_series_for_segements(segments, 1)

    def get_series_for_active_segments(self, i):
        """
        i = 0 for axis, i = 1 for values
        """
        return self.get_series_for_segements(self.get_active_segments(), i)

    def get_axis_for_active_segments(self):
        return self.get_series_for_active_segments(0)

    def get_values_for_active_segments(self):
        return self.get_series_for_active_segments(1)

    def plot(self, ax: plt.Axes, color="C0"):

        def pplot(data):
            ax.plot(
                (data["Potential (V)"] - self.get_voltage_limit()[0])
                / (self.get_voltage_limit()[1] - self.get_voltage_limit()[0]),
                (data["Current (A)"] - self.get_current_limit()[0])
                / (self.get_current_limit()[1] - self.get_current_limit()[0]),
                label=self.name,
                color=color,
            )

        if self.active_segments == "all":
            pplot(self.data)
        elif isinstance(self.active_segments, int):
            segment_data = self.data[self.data["Segment"] == self.active_segments]
            pplot(segment_data)
        elif isinstance(self.active_segments, list):
            segment_data = self.data[self.data["Segment"].isin(self.active_segments)]
            pplot(segment_data)
        else:
            raise ValueError(
                f"Segments should be 'all', int or list of int, but got {self.active_segments}, which is {type(self.active_segments)}"
            )


@dataclass
class SegmentedContinuousSignal1D(SegmentedSignal1D, ContinuousSignal1D):

    # def __init__(
    #     self,
    #     data,
    #     name,
    #     axis_name=None,
    #     axis_unit=None,
    #     value_name=None,
    #     value_unit=None,
    #     update="to_data",
    #     detect_axis_name_and_unit=False,
    #     detect_value_name_and_unit=False,
    #     axis_limit=None,
    #     axis_margin=(0.1, 0.1),
    #     axis_digits=11,
    #     value_limit=None,
    #     value_margin=(0.1, 0.1),
    #     value_digits=11,
    # ):
    #     # print(axis_margin)
    #     # print(value_margin)
    #     ContinuousSignal1D.__init__(
    #         self,
    #         data=data,
    #         name=name,
    #         axis_name=axis_name,
    #         axis_unit=axis_unit,
    #         value_name=value_name,
    #         value_unit=value_unit,
    #         update=update,
    #         detect_axis_name_and_unit=detect_axis_name_and_unit,
    #         detect_value_name_and_unit=detect_value_name_and_unit,
    #         axis_limit=axis_limit,
    #         axis_margin=axis_margin,
    #         value_limit=value_limit,
    #         value_margin=value_margin,
    #     )
    #     SegmentedSignal1D.__init__(self)

    def set_active_segments(self, segments, update_value_limit=True):
        res = super().set_active_segments(segments)
        if update_value_limit:
            self.set_default_relative_value_limit()

    def set_default_relative_value_limit(self):
        segment_indices = self.get_indices_for_active_segments()
        axis_indices = self.get_indices_for_current_axis_limit()
        indices = list(set(segment_indices) & set(axis_indices))
        indices.sort()
        my_data = self.data.loc[indices, :]
        value_limit = (my_data.iloc[:, 1].min(), my_data.iloc[:, 1].max())
        self.description_annotations[1].set_limit(value_limit)

    def set_default_limit_and_margin(self):
        # 分析 signal 的 axis 与 value 范围, 设定 axis 和 value 的默认范围

        segment_data = self.get_data_for_active_segments()

        axis_limit = (min(segment_data.iloc[:, 0]), max(segment_data.iloc[:, 0]))
        value_limit = (min(segment_data.iloc[:, 1]), max(segment_data.iloc[:, 1]))
        self.description_annotations[0].set_limit(axis_limit)
        self.description_annotations[1].set_limit(value_limit)

    def plot_at(self, ax: plt.Axes, **kwargs):
        """
        Plot the signal at the given ax and return an artist handle.
        You can provide extra arguments to the plot function by **kwargs. Available arguments are listed below:
        - color: str
        - linewidth: float
        - linestyle: str
        - label: str
        - alpha: float
        - marker: str
        - horizontalalignment: center | right | left
        - verticalalignment: center | top | bottom
        """
        kwargs.pop("text_shift", None)
        label = kwargs.pop("label", None)
        if not label:
            label = self.name

        kwargs_for_annotate = kwargs.copy()
        kwargs_for_Line2D = kwargs.copy()

        kwargs_for_Line2D.pop("fontsize", None)
        kwargs_for_Line2D.pop("rotation", None)

        (handle,) = ax.plot(
            self.rescale(
                self.get_axis_for_active_segments(),
                self.get_axis_limit(),
                self.get_axis_margin(),
            ),
            self.rescale(
                self.get_values_for_active_segments(),
                self.get_value_limit(),
                self.get_value_margin(),
            ),
            label=label,
            **kwargs_for_Line2D,
        )
        return handle


@dataclass
class FractionSignal(DiscreteSignal1D):
    @classmethod
    def from_csv(
        cls,
        path,
        name=None,
        detect_axis_name_and_unit=True,
        detect_value_name_and_unit=False,
        axis_name=None,
        axis_unit=None,
        value_name="Fraction",
        value_unit=None,
        **kwargs,
    ):
        return super().from_csv(
            path=path,
            name=name,
            detect_axis_name_and_unit=detect_axis_name_and_unit,
            detect_value_name_and_unit=detect_value_name_and_unit,
            axis_name=axis_name,
            axis_unit=axis_unit,
            value_name=value_name,
            value_unit=value_unit,
            **kwargs,
        )

    # def __init__(
    #     self,
    #     data,
    #     name,
    #     axis_name=None,
    #     axis_unit=None,
    #     value_name="Fraction",
    #     value_unit=None,
    #     update="to_data",
    #     detect_axis_name_and_unit=False,
    #     detect_value_name_and_unit=False,
    # ):
    #     super().__init__(
    #         data=data,
    #         name=name,
    #         axis_name=axis_name,
    #         axis_unit=axis_unit,
    #         update=update,
    #         value_name=value_name,
    #         value_unit=value_unit,
    #         detect_axis_name_and_unit=detect_axis_name_and_unit,
    #         detect_value_name_and_unit=detect_value_name_and_unit,
    #     )
    #     if not axis_unit in chromatographic_units_map_types.keys():
    #         raise ValueError(
    #             f"Expected axis_unit to be one of {chromatographic_units_map_types.keys()}, but got {axis_unit}"
    #         )

    #     if axis_name == "undefined":
    #         self.set_axis_name(chromatographic_units_map_types[axis_unit])
