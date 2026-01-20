import os, time, warnings
from dataclasses import dataclass, field
from pathlib import Path
from .signal import Signal, DescAnno
from .display import SignalPlotArgs


@dataclass
class SignalCollection:
    """
    A SignalCollection contains multiple signals and is designed to
    easily compare and visualize them.

    - Dimensions of signals in a signal collection must be the same.
    - Signals in a collection is stored by a dictionary, you can find every signal with its name like SignalCollection[signal_name]

    Properties
    ----------
    signals: list[Signal]
        A dictionary of signals in the collection, with signal names as keys.
    name: str
        The name of the signal collection.
    plot_args: PlotArgs
        The arguments for plotting the signal collection.
    """

    signals: list[Signal] = None
    signal_type: type = Signal
    visible_signal_names: list[str] = None
    name: str = None
    description_annotations: list[DescAnno] = None
    plot_args: SignalPlotArgs = None
    __index_cache__: dict = None

    @property
    def signal_names(self):
        return [signal.name for signal in self.signals]

    @staticmethod
    def merge(
        signal_collections: list["SignalCollection"], name="Merged_signal_collection"
    ) -> "SignalCollection":
        signals = []
        for signal_collection in signal_collections:
            for signal in signal_collection.signals:
                signals.append(signal)
        return SignalCollection(signals, name=name)

    def keys(self):
        return [signal.name for signal in self.signals]

    def index(self, signal_name: str):
        if not hasattr(self, "__index_cache"):
            keys = self.keys()
            self.__index_cache__ = {key: i for i, key in enumerate(keys)}
        if signal_name not in self.__index_cache__:
            raise KeyError(
                f"Signal name {signal_name} does not exist in the collection"
            )
        return self.__index_cache__[signal_name]

    def __contains__(self, signal_name: str):
        try:
            self.index(signal_name)
            return True
        except KeyError:
            return False

    def __getitem__(self, signal_name: str):
        signal_index = self.index(signal_name)
        return self.signals[signal_index]

    def __setitem__(self, signal_name: str, signal: Signal):
        try:
            signal_index = self.index(signal_name)
            self.signals[signal_index] = signal
            signal.name = signal_name
        except KeyError:
            signal_index = len(self.signals)
            self.signals.append(signal)
            signal.name = signal_name

    def __delitem__(self, signal_name: str):
        signal_index = self.index(signal_name)
        del self.signals[signal_index]
        del self.__index_cache__[signal_name]

    def append(self, signal: Signal) -> None:
        keys = self.keys()
        if signal.name in keys:
            raise KeyError(
                f"Signal name {signal.name} already exists in the collection"
            )
        self.signals.append(signal)
        if self.__index_cache__ is not None:
            self.__index_cache__[signal.name] = len(self.signals) - 1

    def remove_signal(self, signal_name: str) -> None:
        signal_index = self.index(signal_name)
        del self.signals[signal_index]
        if self.__index_cache__ is not None:
            self.__index_cache__ = None
        try:
            self.visible_signal_names.remove(signal_name)
        except ValueError:
            pass

    def rename_signal(self, old_signal_name, new_signal_name):
        signal = self[old_signal_name]
        signal.name = new_signal_name
        if self.__index_cache__ is not None:
            self.__index_cache__[new_signal_name] = self.__index_cache__.pop(
                old_signal_name
            )
        self.visible_signal_names.remove(old_signal_name)
        self.visible_signal_names.append(new_signal_name)

    def to_folder(self, directory: str | Path):
        """
        Export the collection to `root_directory/self.name`

        Parameters
        ----------
        mode : str
            "append", "write" or "replace"
        """
        directory = Path(directory)
        time_stamp = time.strftime("%Y%m%d_%H%M%S")

        if directory.exists():
            directory = directory.with_name(f"{directory.name}_{time_stamp}")
            warnings.warn(
                f"Directory {directory} already exists. Renamed to {directory.name}."
            )
        directory.mkdir(parents=True)
        for signal in self.signals:
            if "/" in signal.name:
                warnings.warn(
                    f"Signal name {signal.name} contains '/', which is not allowed in file names. They are replaced with '_'."
                )
                self.rename_signal(signal.name, signal.name.replace("/", "_"))
        for signal in self.signals:
            signal.to_csv(directory / (signal.name + ".csv"))

    def __len__(self):
        return len(self.signals)
