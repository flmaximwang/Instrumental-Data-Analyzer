from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from .signal import ContDescAnno, DiscDescAnno
from .signal_collection import SignalCollection
from .signal_1d import Signal1D, ContinuousSignal1D
from .display import Signal1DPlotArgs
import matplotlib.pyplot as plt


@dataclass
class Signal1DCollection(SignalCollection):

    signal_type: type = Signal1D
    signals: list[Signal1D] = field(default_factory=list)
    plot_args: Signal1DPlotArgs = field(default_factory=Signal1DPlotArgs)
    # display_modes = ["main_signal_axis", "all_axis", "separate", "denoted_axis"]

    # ==================== Properties for axis description ====================

    @property
    def axis_annotation(self):
        axis_annotation: ContDescAnno = self.description_annotations[0]
        return axis_annotation

    # ==================== Properties for value description ====================

    @property
    def value_annotation(self):
        return self.description_annotations[1]

    @value_annotation.setter
    def value_annotation(self, new_value_description):
        self.description_annotations[1] = new_value_description

    # ============================ Constructors ============================

    @classmethod
    def from_similar_signals(
        cls,
        signals: list[Signal1D],
    ):
        "Quickly create a Signal1DCollection from a list of similar Signal1D"

        axis_annotation = deepcopy(signals[0].axis_annotation)
        value_annotation = deepcopy(signals[0].value_annotation)

        result = cls(
            signals=signals,
            description_annotations=[axis_annotation, value_annotation],
            visible_signal_names=[signal.name for signal in signals],
        )
        return result

    @classmethod
    def merge(
        cls,
        signal_collections: list["Signal1DCollection"],
        name="Merged Signal1DCollection",
        signal_renaming=True,
    ) -> "Signal1DCollection":
        signals: list[Signal1D] = []
        for signal_collection in signal_collections:
            for signal in signal_collection.signals:
                if signal_renaming:
                    signals.append(signal)
                    signal.name = f"{signal_collection.name}_{signal.name}"
        return cls(signals, name=name)

    # ============================ Aligner ============================

    def align_axes(self, *axis_limit) -> None:
        """
        Align the axis limits of all signals in the collection to the same axis limits.
        """
        for signal in self.signals:
            signal.axis_annotation.limit = [*axis_limit]
        self.axis_annotation.limit = [*axis_limit]

    def align_values(self, signal_names: list[str], *value_limit) -> None:
        """
        Align the value limits of selected signals in the collection to the same value limits.
        """
        if not signal_names:
            # Align values for all continuous signals
            signal_names = []
            for signal in self.signals:
                if isinstance(signal, ContinuousSignal1D):
                    signal_names.append(signal.name)
        # Align selected signals
        for signal_name in signal_names:
            signal_value_annotation: ContDescAnno = self[signal_name].value_annotation
            signal_value_annotation.limit = [*value_limit]
        value_annotation: ContDescAnno = self.value_annotation
        value_annotation.limit = [*value_limit]

    @classmethod
    def from_folder(cls, folder: str | Path, **kwargs):
        """
        Import multiple .csv files from one folder
        """
        folder = Path(folder)
        signals = []
        for file in sorted(folder.glob("*.csv")):
            signal = cls.signal_type.from_csv(file, **kwargs)
            signals.append(signal)
        result = cls.from_similar_signals(signals)
        result.name = folder.name
        return result

    def plot_with_collection_annotations(self, **kwargs):

        figsize = self.plot_args.figsize

        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.plot_with_collection_annotations_at(ax, **kwargs)

        return fig, ax

    def plot_with_collection_annotations_at(self, ax: plt.Axes, **kwargs):

        cmap = self.plot_args.cmap
        cmap_limit = self.plot_args.cmap_limit
        legend_cols = self.plot_args.legend_cols

        handles = []

        if cmap is None:
            for i, signal_name in enumerate(self.visible_signal_names):
                signal: Signal1D = self[signal_name]
                handles.append(signal.plot_at(ax, color=f"C{i}", **kwargs))
        else:
            my_len = len(self.visible_signal_names)
            if my_len > 1:
                for i, signal_name in enumerate(self.visible_signal_names):
                    signal = self[signal_name]
                    handles.append(
                        signal.plot_at(
                            ax,
                            color=cmap(
                                i / (my_len - 1) * (cmap_limit[1] - cmap_limit[0])
                                + cmap_limit[0]
                            ),
                            **kwargs,
                        )
                    )
            else:
                signal = self[self.visible_signal_names[0]]
                handles.append(signal.plot_at(ax, color=cmap(1.0), **kwargs))
        xticks = self.axis_annotation.ticks
        xticklabels = self.axis_annotation.ticklabels
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        if isinstance(self.value_annotation, ContDescAnno):
            yticks = self.value_annotation.ticks
            yticklabels = self.value_annotation.ticklabels
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(self.axis_annotation.label)
        ax.set_ylabel(self.value_annotation.label)
        ax.legend(handles=handles, ncols=legend_cols)
        ax.set_title(self.name)

    # def plot_with_main_annotations(self, **kwargs):

    #     ax: plt.Axes
    #     fig, ax = self.subplots(1, 1)
    #     axes = [ax]
    #     handles = []
    #     my_colormap = self.colormap
    #     my_colormap_min = self.colormap_min
    #     my_colormap_max = self.colormap_max
    #     need_legend = kwargs.pop("legend", True)
    #     legend_cols = kwargs.pop("legend_cols", 1)
    #     axis_digits = kwargs.pop("axis_digits", 1)
    #     value_digits = kwargs.pop("value_digits", 1)
    #     if my_colormap == "default":
    #         for i, signal_name in enumerate(self.visible_signal_names):
    #             signal = self.signals[signal_name]
    #             handles.append(signal.plot_at(ax, color=f"C{i}", **kwargs))
    #     else:
    #         if isinstance(my_colormap, list):
    #             if len(my_colormap) != len(self.visible_signal_names):
    #                 raise ValueError(
    #                     "The length of colormap should be the same as the number of visible signals"
    #                 )
    #             for i, signal_name in enumerate(self.visible_signal_names):
    #                 signal = self.signals[signal_name]
    #                 handles.append(signal.plot_at(ax, color=my_colormap[i], **kwargs))
    #         else:
    #             my_len = len(self.visible_signal_names)
    #             if my_len > 1:
    #                 for i, signal_name in enumerate(self.visible_signal_names):
    #                     signal = self.signals[signal_name]
    #                     handles.append(
    #                         signal.plot_at(
    #                             ax,
    #                             color=my_colormap(
    #                                 i
    #                                 / (my_len - 1)
    #                                 * (my_colormap_max - my_colormap_min)
    #                                 + my_colormap_min
    #                             ),
    #                             **kwargs,
    #                         )
    #                     )
    #             else:
    #                 signal = self.signals[self.visible_signal_names[0]]
    #                 handles.append(signal.plot_at(ax, color=my_colormap(0.6), **kwargs))
    #     main_signal = self[self.main_signal_name]
    #     xticks = main_signal.get_axis_ticks()
    #     xticklabels = main_signal.get_axis_ticklabels(digits=axis_digits)
    #     yticks = main_signal.get_value_ticks()
    #     yticklabels = main_signal.get_value_tick_labels(digits=value_digits)
    #     ax.set_xticks(xticks)
    #     ax.set_yticks(yticks)
    #     ax.set_xticklabels(xticklabels)
    #     ax.set_yticklabels(yticklabels)
    #     ax.set_xlim([0, 1])
    #     ax.set_ylim([0, 1])
    #     ax.set_xlabel(self.get_axis_label())
    #     ax.set_ylabel(main_signal.get_value_label())
    #     if need_legend:
    #         ax.legend(handles=handles, ncols=legend_cols)
    #     ax.set_title(self.get_name())
    #     fig.tight_layout()
    #     return fig, axes

    def plot_with_all_annotations(self, axis_shift, **kwargs):
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=self.plot_args.figsize)
        twins: list[plt.Axes] = []
        handles: list[plt.Line2D] = []
        counter = 0
        for i, signal_name in enumerate(self.visible_signal_names):
            signal = self[signal_name]
            if i > 0:  # Other signals share the same x axis
                if isinstance(signal, ContinuousSignal1D):
                    twins.append(ax.twinx())
                    twins[-1].spines.right.set_position(
                        ("axes", 1 + axis_shift * counter)
                    )
                    counter += 1
                    ax_to_plot = twins[-1]
                    handle = signal.plot_at(ax_to_plot, color=f"C{i}", **kwargs)
                else:
                    ax_to_plot = ax
                    handle = signal.plot_at(ax_to_plot, color=f"C{i}", **kwargs)
            else:  # The first signal controls the x axis
                if isinstance(signal, ContinuousSignal1D):
                    ax_to_plot = ax
                    handle = signal.plot_at(ax_to_plot, color=f"C{i}", **kwargs)
                else:
                    raise TypeError("The first signal should be a ContinuousSignal1D")
            if isinstance(signal, ContinuousSignal1D):

                ax_to_plot.tick_params(axis="y", colors=handle.get_color())
                ax_to_plot.yaxis.label.set_color(handle.get_color())
                ax_to_plot.set_ylim([0, 1])
                ax_to_plot.set_ylabel(signal.value_annotation.label)
                ax_to_plot.set_yticks(signal.value_annotation.ticks)
                ax_to_plot.set_yticklabels(signal.value_annotation.ticklabels)

            handles.append(handle)

        ax.set_xlabel(self.axis_annotation.label)
        ax.set_xticks(self.axis_annotation.ticks)
        ax.set_xticklabels(self.axis_annotation.ticklabels)
        ax.set_xlim(0, 1)
        ax.legend(handles=handles)
        ax.set_title(self.name)
        fig.tight_layout()
        # 避免右侧额外的坐标轴跑到画布以外
        fig.subplots_adjust(right=1 - axis_shift * counter)
        return fig, [ax] + twins

    def plot_separately(self, **kwargs) -> tuple[plt.Figure, list[plt.Axes]]:
        row_num = kwargs.pop("row", 0)
        col_num = kwargs.pop("col", 0)
        if row_num == 0 and col_num == 0:
            # 尽可能按照正方形进行 plot
            row_num = (
                int(np.sqrt(len(self.visible_signal_names))) + 1
                if np.sqrt(len(self.visible_signal_names)) % 1 != 0
                else int(np.sqrt(len(self.visible_signal_names)))
            )
            col_num = (
                len(self.visible_signal_names) // row_num + 1
                if len(self.visible_signal_names) % row_num != 0
                else len(self.visible_signal_names) // row_num
            )
        elif row_num == 0:
            # 按照 col_num 调整 row_num
            row_num = (
                len(self.visible_signal_names) // col_num + 1
                if len(self.visible_signal_names) % col_num != 0
                else len(self.visible_signal_names) // col_num
            )
        elif col_num == 0:
            # 按照 row_num 调整 col_num
            col_num = (
                len(self.visible_signal_names) // row_num + 1
                if len(self.visible_signal_names) % row_num != 0
                else len(self.visible_signal_names) // row_num
            )

        if row_num * col_num < len(self.visible_signal_names):
            raise ValueError(
                f"{row_num} rows and {col_num} columns are not enough to plot {len(self.visible_signal_names)} signals."
            )

        if row_num == 1 and col_num == 1:
            fig, ax = self.subplots(1, 1)
            axes = [[ax]]
        elif row_num == 1:
            fig, axes = self.subplots(1, col_num)
            axes = [axes]
        elif col_num == 1:
            fig, axes = self.subplots(row_num, 1)
            axes = [[ax] for ax in axes]
        else:
            fig, axes = self.subplots(row_num, col_num)

        for i, signal_name in enumerate(self.visible_signal_names):
            signal = self.signals[signal_name]
            row_index = i // col_num
            col_index = i % col_num
            signal.plot_at(axes[row_index][col_index], **kwargs)
            axes[row_index][col_index].set_title(signal_name)
            axes[row_index][col_index].set_xlabel(self.get_axis_label())
            axes[row_index][col_index].set_ylabel(signal.get_value_label())
            xticks, xticklabels = signal.get_axis_ticks(), signal.get_axis_ticklabels()
            axes[row_index][col_index].set_xticks(xticks)
            axes[row_index][col_index].set_xticklabels(xticklabels)
            yticks, yticklabels = (
                signal.get_value_ticks(),
                signal.get_value_tick_labels(),
            )
            axes[row_index][col_index].set_yticks(yticks)
            axes[row_index][col_index].set_yticklabels(yticklabels)
            axes[row_index][col_index].set_xlim(0, 1)
            axes[row_index][col_index].set_ylim(0, 1)
        fig.tight_layout()
        return fig, axes

    def plot(self, **kwargs):
        """
        mode = 0: plot with collection annotations
        mode = 1: plot with all value labels;
        mode = 2: plot separately;
        legend_cols: int, default 1, number of columns in the legend
        """
        axes: list[plt.Axes]
        if self.plot_args.mode in [0, 1]:
            # Axes containing only 1 subplot
            if self.plot_args.mode == 0:
                fig, ax = self.plot_with_collection_annotations(**kwargs)
                ax.set_title(self.name)
                return fig, ax
            elif self.plot_args.mode == 1:
                axis_shift = self.plot_args.axis_shift
                fig, axes = self.plot_with_all_annotations(
                    axis_shift=axis_shift, **kwargs
                )
                axes[0].set_title(self.name)
                return fig, axes
        elif self.plot_args.mode in [2]:
            # Axes containing multiple subplots
            if self.plot_args.mode == 2:
                fig, axes = self.plot_separately(**kwargs)
                return fig, axes
        else:
            raise Exception("Unknown display mode")

    def set_default_annotations(self):
        axis_limit = None
        for signal in self.signals:
            axis_data = signal.data.iloc[:, 0]
            axis_min = min(axis_data)
            axis_max = max(axis_data)
            axis_limit = (
                (axis_min, axis_max)
                if axis_limit is None
                else (min(axis_limit[0], axis_min), max(axis_limit[1], axis_max))
            )
        self.align_axes(*axis_limit)
        ticklabel_space = (axis_limit[1] - axis_limit[0]) / 10
        if ticklabel_space == 0:
            raise ValueError("All signals have the same axis data.")
        self.axis_annotation.ticklabel_space = ticklabel_space
        for signal in self.signals:
            if isinstance(signal, ContinuousSignal1D):
                value_data = signal.data.iloc[:, 1]
                value_min = min(value_data)
                value_max = max(value_data)
                signal.value_annotation.limit = (value_min, value_max)
                ticklabel_space = (value_max - value_min) / 10
                if ticklabel_space == 0:
                    ticklabel_space = 1.0
                signal.value_annotation.ticklabel_space = ticklabel_space


@dataclass
class ContinuousSignal1DCollection(Signal1DCollection):
    """
    ContinuousSignal1DCollection is a collection of only ContinuousSignal1D
    """

    signal_type = ContinuousSignal1D
    signals: list[ContinuousSignal1D] = field(default_factory=list)

    @property
    def value_annotation(self) -> ContDescAnno:
        value_description: ContDescAnno = self.description_annotations[1]
        return value_description

    @value_annotation.setter
    def value_annotation(self, new_value_description: ContDescAnno):
        self.description_annotations[1] = new_value_description

    @classmethod
    def merge(
        cls,
        signal_collections: list["ContinuousSignal1DCollection"],
        name="Merged ContinuousSignal1DCollection",
    ):
        signals = []
        for signal_collection in signal_collections:
            for signal in signal_collection.signals:
                signals.append(signal)
        return cls(signals, name=name)

    def average_similar_signals(
        self, func_to_define_similarity=lambda x: x.split()[0]
    ) -> "ContinuousSignal1DCollection":
        """
        Average the signals with similar names in the collection

        Details
        -------
        - Similar names are defined as the names sharing the name.split()[0] part.
        """
        name_dict = {}
        for signal in self.signals:
            base_name = func_to_define_similarity(signal.name)
            if base_name not in name_dict:
                name_dict[base_name] = []
            name_dict[base_name].append(signal)
        new_signals = []
        for base_name, signal_list in name_dict.items():
            if len(signal_list) == 1:
                new_signals.append(signal_list[0])
                new_signals[-1].name = base_name
            else:
                new_signal = type(self.signals[0]).average(signal_list)
                new_signal.name = base_name
                new_signals.append(new_signal)

        result = type(self).from_similar_signals(new_signals)
        return result

    def set_default_annotations(self):
        axis_limit = None
        for signal in self.signals:
            axis_data = signal.data.iloc[:, 0]
            axis_min = min(axis_data)
            axis_max = max(axis_data)
            axis_limit = (
                (axis_min, axis_max)
                if axis_limit is None
                else (min(axis_limit[0], axis_min), max(axis_limit[1], axis_max))
            )
        self.align_axes(*axis_limit)
        self.axis_annotation.ticklabel_space = (axis_limit[1] - axis_limit[0]) / 10

        value_limit = None
        for signal in self.signals:
            value_data = signal.data.iloc[:, 1]
            value_min = min(value_data)
            value_max = max(value_data)
            value_limit = (
                (value_min, value_max)
                if value_limit is None
                else (min(value_limit[0], value_min), max(value_limit[1], value_max))
            )
        self.align_values(self.signal_names, *value_limit)
        self.value_annotation.ticklabel_space = (value_limit[1] - value_limit[0]) / 10
