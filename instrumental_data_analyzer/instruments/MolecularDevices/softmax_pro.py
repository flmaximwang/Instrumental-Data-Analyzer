import re, tempfile, warnings
from io import StringIO
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from ...abstract.signal import ContDescAnno
from ...abstract.signal_1d import ContinuousSignal1D
from ...abstract.signal_1d_collection import (
    Signal1DCollection,
    ContinuousSignal1DCollection,
)
from ...concrete.multi_well_plates import (
    MultiWellPlateSeries,
    MultiWellPlate,
    PLATE_COL_NAMES,
    PLATE_ROW_NAMES,
)
from ...concrete.kinetics import KineticCurve, KineticCurveCollection


def well_num_to_row_num(well_num: int) -> int:
    if well_num == 6:
        return 2
    elif well_num == 12:
        return 3
    elif well_num == 24:
        return 4
    elif well_num == 48:
        return 6
    elif well_num == 96:
        return 8
    elif well_num == 384:
        return 16
    else:
        raise ValueError("Unsupported plate well number")


def well_num_to_col_num(well_num: int) -> int:
    if well_num == 6:
        return 3
    elif well_num == 12:
        return 4
    elif well_num == 24:
        return 6
    elif well_num == 48:
        return 8
    elif well_num == 96:
        return 12
    elif well_num == 384:
        return 24
    else:
        raise ValueError("Unsupported plate well number")


def time_to_seconds(time: str):
    if "." in time:
        day = int(time.split(".")[0])
        time = time.split(".")[1]
    else:
        day = 0
    h, m, s = map(int, time.split(":"))
    return day * 3600 * 24 + h * 3600 + m * 60 + s


class SoftMaxPro_Plate:
    """
    An object to represent a plate in Molecular Devices SoftMax Pro software. The data is usually organized
    in a specific format, which can be parsed to extract the kinetic or endpoint data for each plate. The extracted
    data is stored as a MultiWellPlate object, which contains the data matrix and the corresponding metadata.
    """

    @staticmethod
    def parse_metadata(text: str) -> dict[str, str]:
        """
        For every plate (block), there is a line of text containing the metadata of the plate, which is organized
        in a specific format. This function parses the metadata line and extracts the relevant information,
        such as plate name, read method, read mode, data type, data point number, duration, interval,
        wavelength number, excitation wavelengths, column start and number, plate well number, emission wavelengths,
        cutoff type and wavelength, flashes per read, PMT gain, row start and number. The extracted information
        is returned as a dictionary.
        """
        components = text.split("\t")
        _ = components.pop(0)  # "Plate:"
        plate_name = components.pop(0)
        _ = components.pop(0)  # A strange number 1.3
        _ = components.pop(0)  # "PlateFormat"
        method = components.pop(0)  # Kinetic, Endpoint, or Spectrum
        mode = components.pop(0)  # Absorbance, Fluorescence, or Luminescence)
        if method == "Kinetic":
            if mode == "Fluorescence":
                read_from_bottom = components.pop(0)
                data_type = components.pop(0)
                assert data_type in [
                    "Raw",
                    "Reduced",
                ], f"Unsupported data type {data_type} for kinetic fluorescence method"
                calibrate = components.pop(0)
                data_point_num = int(components.pop(0))
                duration_seconds = int(components.pop(0))
                interval_seconds = int(components.pop(0))
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                wavelength_num = int(components.pop(0))  # Wavelength number
                excitation_wavelengths = list(
                    map(int, components.pop(0).split())
                )  # Excitation wavelengths
                col_start = int(components.pop(0))
                col_num = int(components.pop(0))
                plate_well_num = int(components.pop(0))
                emission_wavelengths = list(
                    map(int, components.pop(0).split())
                )  # Emission wavelengths
                cutoff_type = components.pop(0)  # Cutoff type
                cutoff_wavelength = int(components.pop(0))  # Cutoff wavelength
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                flashes_per_read = int(components.pop(0))  # Flashes per read
                PMT_gain = components.pop(0)  # PMT gain
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                row_start = int(components.pop(0))
                row_num = int(components.pop(0))
                _ = components.pop(0)  # Unknown
                return {
                    "plate_name": plate_name,
                    "method": method,
                    "mode": mode,
                    "read_from_bottom": read_from_bottom,
                    "calibrate": calibrate,
                    "data_type": data_type,
                    "data_point_num": data_point_num,
                    "duration_seconds": duration_seconds,
                    "interval_seconds": interval_seconds,
                    "wavelength_num": wavelength_num,
                    "excitation_wavelengths": excitation_wavelengths,
                    "col_start": col_start,
                    "col_num": col_num,
                    "plate_well_num": plate_well_num,
                    "emission_wavelengths": emission_wavelengths,
                    "cutoff_type": cutoff_type,
                    "cutoff_wavelength": cutoff_wavelength,
                    "flashes_per_read": flashes_per_read,
                    "PMT_gain": PMT_gain,
                    "row_start": row_start,
                    "row_num": row_num,
                }
            elif mode == "Absorbance":
                data_type = components.pop(0)
                assert data_type in [
                    "Raw",
                    "Reduced",
                ], f"Unsupported data type {data_type} for kinetic absorbance method"
                calibrate = components.pop(0)
                data_point_num = int(components.pop(0))
                duration_seconds = int(components.pop(0))
                interval_seconds = int(components.pop(0))
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                wavelength_num = int(components.pop(0))  # Wavelength number
                wavelengths = list(map(int, components.pop(0).split()))  # Wavelengths
                col_start = int(components.pop(0))
                col_num = int(components.pop(0))
                plate_well_num = int(components.pop(0))
                row_start = int(components.pop(0))
                row_num = int(components.pop(0))
                _ = components.pop(0)  # Unknown
                return {
                    "plate_name": plate_name,
                    "method": method,
                    "mode": mode,
                    "data_type": data_type,
                    "calibrate": calibrate,
                    "data_point_num": data_point_num,
                    "duration_seconds": duration_seconds,
                    "interval_seconds": interval_seconds,
                    "wavelength_num": wavelength_num,
                    "wavelengths": wavelengths,
                    "col_start": col_start,
                    "col_num": col_num,
                    "plate_well_num": plate_well_num,
                    "row_start": row_start,
                    "row_num": row_num,
                }
            else:
                raise ValueError(f"Unsupported read mode {mode} for kinetic method")
        elif method == "Endpoint":
            if mode == "Absorbance":
                data_type = components.pop(0)
                calibrate = components.pop(0)
                data_point_num = int(components.pop(0))
                _ = components.pop(0)
                _ = components.pop(0)
                _ = components.pop(0)
                _ = components.pop(0)
                _ = components.pop(0)
                wavelength_num = components.pop(0)
                wavelengths = list(map(int, components.pop(0).split()))
                col_start = int(components.pop(0))
                col_num = int(components.pop(0))
                plate_well_num = int(components.pop(0))
                row_start = int(components.pop(0))
                row_num = int(components.pop(0))
                _ = components.pop(0)
                return {
                    "plate_name": plate_name,
                    "method": method,
                    "calibrate": calibrate,
                    "data_point_num": data_point_num,
                    "wavelength_num": wavelength_num,
                    "wavelengths": wavelengths,
                    "col_start": col_start,
                    "col_num": col_num,
                    "plate_well_num": plate_well_num,
                    "row_start": row_start,
                    "row_num": row_num,
                }
        elif method == "Spectrum":
            if mode == "Absorbance":
                data_type = components.pop(0)
                assert data_type in [
                    "Raw",
                    "Reduced",
                ], f"Unsupported data type {data_type} for spectrum absorbance method"
                calibrate = components.pop(0)
                data_point_num = int(components.pop(0))
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                wavelength_min = int(components.pop(0))
                wavelength_max = int(components.pop(0))
                wavelength_interval = int(components.pop(0))
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                col_start = int(components.pop(0))
                col_num = int(components.pop(0))
                plate_well_num = int(components.pop(0))
                row_start = int(components.pop(0))
                row_num = int(components.pop(0))
                _ = components.pop(0)  # Unknown
                return {
                    "plate_name": plate_name,
                    "method": method,
                    "mode": mode,
                    "data_type": data_type,
                    "calibrate": calibrate,
                    "data_point_num": data_point_num,
                    "wavelength_min": wavelength_min,
                    "wavelength_max": wavelength_max,
                    "wavelength_interval": wavelength_interval,
                    "col_start": col_start,
                    "col_num": col_num,
                    "plate_well_num": plate_well_num,
                    "row_start": row_start,
                    "row_num": row_num,
                }
            elif mode == "Fluorescence":
                read_from_bottom = components.pop(0)
                data_type = components.pop(0)
                assert data_type in ["Raw", "Reduced"]
                calibrate = components.pop(0)
                data_point_num = int(components.pop(0))
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                wavelength_min = int(components.pop(0))
                wavelength_max = int(components.pop(0))
                wavelength_interval = int(components.pop(0))
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                col_start = int(components.pop(0))
                col_num = int(components.pop(0))
                plate_well_num = int(components.pop(0))
                _ = components.pop(0)  # Unknown
                cutoff_type = components.pop(0)  # Cutoff type
                cutoff_wavelength = int(components.pop(0))  # Cutoff wavelength
                spectrum_type = components.pop(0)  # Scanning type
                excitation_wavelength = int(components.pop(0))  # Excitation wavelength
                flashes_per_read = int(components.pop(0))  # Flashes per read
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                _ = components.pop(0)  # Unknown
                row_start = int(components.pop(0))
                row_num = int(components.pop(0))
                _ = components.pop(0)  # Unknown
                return {
                    "plate_name": plate_name,
                    "method": method,
                    "mode": mode,
                    "data_type": data_type,
                    "read_from_bottom": read_from_bottom,
                    "calibrate": calibrate,
                    "data_point_num": data_point_num,
                    "wavelength_min": wavelength_min,
                    "wavelength_max": wavelength_max,
                    "wavelength_interval": wavelength_interval,
                    "col_start": col_start,
                    "col_num": col_num,
                    "plate_well_num": plate_well_num,
                    "cutoff_type": cutoff_type,
                    "cutoff_wavelength": cutoff_wavelength,
                    "spectrum_type": spectrum_type,
                    "excitation_wavelength": excitation_wavelength,
                    "flashes_per_read": flashes_per_read,
                    "row_start": row_start,
                    "row_num": row_num,
                }
            else:
                raise ValueError(f"Unsupported read mode {mode} for spectrum method")
        else:
            raise ValueError(f"Unsupported read method {method}")

    @staticmethod
    def parse_read_area(
        plate_well_num: int,
        col_start: int,
        col_num: int,
        row_start: int,
        row_num: int,
        wavelengths: list[int] = None,
    ) -> list[tuple[str, int]]:
        """
        Calculate the well markers (a list of well markers) for the read area based on the plate well number, column start and number, row start and number.
        The well markers are returned as a list of tuples, where each tuple contains the row name and column name corresponding to a well in the read area.
        The row names and column names are determined by the plate well number and the column and row start and number.
        """
        well_markers = []
        plate_row_names_for_well_num = PLATE_ROW_NAMES[plate_well_num]
        plate_col_names_for_well_num = PLATE_COL_NAMES[plate_well_num]
        if wavelengths is None:
            for i in range(row_start - 1, row_start + row_num - 1):
                for j in range(col_start - 1, col_start + col_num - 1):
                    well_markers.append(
                        (
                            plate_row_names_for_well_num[i],
                            plate_col_names_for_well_num[j],
                        )
                    )
        else:
            for i in range(row_start - 1, row_start + row_num - 1):
                for j in range(col_start - 1, col_start + col_num - 1):
                    for wavelength in wavelengths:
                        well_markers.append(
                            (
                                plate_row_names_for_well_num[i],
                                f"{plate_col_names_for_well_num[j]}_{wavelength}",
                            )
                        )
        return well_markers

    @staticmethod
    def parse_read(
        text: str,
        wavelengths: list[int] = None,
    ):
        """
        Each read is organized as text separated by tabs.
        This function parses the text and extracts the data matrix for this read.
        The data matrix is returned as a pandas DataFrame directly correspinding to the plate layout,
        with the row and column names corresponding to the well positions.
        """
        data = pd.read_csv(
            StringIO(text),
            header=None,
            sep="\t",
        )
        plate_data = data.iloc[0:, 2:-2]
        for i in range(len(wavelengths) - 1):
            plate_data = plate_data.drop(columns=plate_data.columns[(i + 1) * 13 - 1])
        plate_shape = plate_data.shape
        expected_col_num = {
            6: (2, 3 * len(wavelengths)),
            12: (3, 4 * len(wavelengths)),
            24: (4, 6 * len(wavelengths)),
            48: (6, 8 * len(wavelengths)),
            96: (8, 12 * len(wavelengths)),
            384: (16, 24 * len(wavelengths)),
        }
        flag = False
        for i in expected_col_num:
            if plate_shape == expected_col_num[i]:
                col_names = PLATE_COL_NAMES[i]
                row_names = PLATE_ROW_NAMES[i]
                flag = True
                break
        if not flag:
            raise ValueError(
                f"Unsupported plate shape {plate_shape} for {len(wavelengths)} wavelengths."
            )
        col_names_with_wavelengths = []
        for wavelength in wavelengths:
            for col_name in col_names:
                col_names_with_wavelengths.append(f"{col_name}_{wavelength}")
        plate_data.columns = col_names_with_wavelengths
        plate_data.index = row_names
        marker = data.iloc[0, 0]
        temperature = data.iloc[0, 1]
        return marker, temperature, plate_data

    @staticmethod
    def parse_block(
        textIO: StringIO,
    ) -> MultiWellPlate | Signal1DCollection:
        """
        Automatically parse the block of text corresponding to a plate and extract the data.
        The parsing method is determined by the read method specified in the metadata.
        For kinetic read method, the data is organized as a series of reads,
        and each read contains a data matrix corresponding to the plate layout.
        The extracted data is returned as a SoftMaxProPlateKinetic object,
        which contains the metadata and the data for this plate.
        For endpoint and spectrum read methods, the parsing method is not implemented yet.
        """
        textIO.seek(0)
        metadata_text = textIO.readline()
        metadata = SoftMaxPro_Plate.parse_metadata(metadata_text)
        if metadata["method"] == "Endpoint":
            return SoftMaxPro_Plate_Endpoint._parse_block(
                textIO=textIO, metadata=metadata
            )
        elif metadata["method"] == "Kinetic":
            return SoftMaxPro_Plate_Kinetic._parse_block(
                textIO=textIO,
                metadata=metadata,
            )
        elif metadata["method"] == "Spectrum":
            return SoftMaxPro_Plate_Spectrum._parse_block(
                textIO=textIO,
                metadata=metadata,
            )
        else:
            raise ValueError(f"Unsupported read method {metadata['method']}")


@dataclass
class SoftMaxPro_Plate_Kinetic(ContinuousSignal1DCollection):

    metadata: dict = None

    @classmethod
    def _parse_block(
        cls,
        textIO: StringIO,
        metadata: dict,
    ) -> "SoftMaxPro_Plate_Kinetic":
        """
        This function should be called by the parse_block function of the SoftMaxProPlate class when the read method is kinetic.
        Each block corresponds to a plate and contains multiple reads. This function parses the block and extracts
        the metadata and the data for this plate. The metadata is extracted using the parse_metadata function, and the data is extracted using the parse_read function. The extracted data is organized as a MultiWellPlateSeries object, which contains a list of MultiWellPlate objects corresponding to each read, and a pandas DataFrame containing the time and temperature information for each read. The metadata and the data are returned as a SoftMaxProPlateKinetic object.
        """
        header = textIO.readline()
        plate_well_num = metadata["plate_well_num"]
        row_num = len(PLATE_ROW_NAMES[plate_well_num])
        data_point_num = metadata["data_point_num"]
        data_raw = {"Time": [], "Temperature": []}
        read_area = SoftMaxPro_Plate.parse_read_area(
            plate_well_num=metadata["plate_well_num"],
            col_start=metadata["col_start"],
            col_num=metadata["col_num"],
            row_start=metadata["row_start"],
            row_num=metadata["row_num"],
            wavelengths=metadata["wavelengths"],
        )
        for wavelength in metadata["wavelengths"]:
            for well_marker in read_area:
                data_raw[well_marker] = []
        early_terminated = False
        for i in range(data_point_num):
            read_text = ""
            for j in range(row_num):
                # Read text row by row
                read_text += textIO.readline()
                if len(read_text) == 0:
                    early_terminated = True
                    break
            if early_terminated:
                warnings.warn(
                    f"Plate {metadata['plate_name']} contains less data points than expected, which indicates an interrupted experiment."
                )
                break
            time, temperature, plate_data = SoftMaxPro_Plate.parse_read(
                read_text, wavelengths=metadata["wavelengths"]
            )
            seconds = time_to_seconds(time)
            data_raw["Time"].append(seconds)
            data_raw["Temperature"].append(temperature)
            for well_marker in read_area:
                data_raw[well_marker].append(
                    plate_data.loc[well_marker[0], well_marker[1]]
                )
            textIO.readline()  # Read the empty line between reads
        signals: list[ContinuousSignal1D] = []
        for value_name in data_raw:
            if value_name == "Time":
                continue
            elif value_name == "Temperature":
                value_unit = "°C"
            else:
                if metadata["mode"] == "Absorbance":
                    value_unit = "AU"
                elif metadata["mode"] == "Fluorescence":
                    value_unit = "RFU"
                else:
                    raise ValueError(f"Unsupported read mode {metadata['mode']}")
            signal = ContinuousSignal1D.from_data(
                data=pd.DataFrame(
                    {
                        "Time": data_raw["Time"],
                        value_name: data_raw[value_name],
                    }
                ),
                axis_name="Time",
                axis_unit="s",
                value_name=metadata["mode"],
                value_unit=value_unit,
                name=(
                    f"{value_name[0]}{value_name[1]}"
                    if isinstance(value_name, tuple)
                    else value_name
                ),
            )
            signals.append(signal)
        res = cls(
            signals=signals,
            description_annotations=[
                ContDescAnno(
                    name="Time",
                    unit="s",
                ),
                ContDescAnno(
                    name=metadata["mode"],
                    unit=value_unit,
                ),
            ],
            visible_signal_names=[signal.name for signal in signals],
        )
        res.metadata = metadata
        return res

    def set_default_annotations(self):
        super().set_default_annotations()
        limit = None
        for signal in self.signals:
            if signal.name == "Temperature":
                continue
            else:
                if limit is None:
                    limit = [
                        signal.value_annotation.limit[0],
                        signal.value_annotation.limit[1],
                    ]
                else:
                    if signal.value_annotation.limit[0] < limit[0]:
                        limit[0] = signal.value_annotation.limit[0]
                    if signal.value_annotation.limit[1] > limit[1]:
                        limit[1] = signal.value_annotation.limit[1]
        self.value_annotation.limit = limit
        self.align_values(self.signal_names, limit[0], limit[1])
        self.value_annotation.ticklabel_space = (limit[1] - limit[0]) / 9


@dataclass
class SoftMaxPro_Plate_Spectrum(ContinuousSignal1DCollection):

    metadata: dict = None

    @classmethod
    def _parse_block(cls, textIO: StringIO, metadata: dict):
        """
        This function should be called by the parse_block function of the SoftMaxProPlate class when the read method is "Spectrum".
        Each block corresponds to a plate and contains multiple reads. This function parses the block and extracts
        the metadata and the data for this plate. The metadata is extracted using the parse_metadata function, and the data is extracted using the parse_read function. The extracted data is organized as a MultiWellPlateSeries object, which contains a list of MultiWellPlate objects corresponding to each read, and a pandas DataFrame containing the time and temperature information for each read. The metadata and the data are returned as a SoftMaxProPlateKinetic object.
        """
        header = textIO.readline()
        plate_well_num = metadata["plate_well_num"]
        row_num = len(PLATE_ROW_NAMES[plate_well_num])
        data_point_num = metadata["data_point_num"]
        data_raw = {"Wavelength": [], "Temperature": []}
        read_area = SoftMaxPro_Plate.parse_read_area(
            plate_well_num=metadata["plate_well_num"],
            col_start=metadata["col_start"],
            col_num=metadata["col_num"],
            row_start=metadata["row_start"],
            row_num=metadata["row_num"],
        )
        for well_marker in read_area:
            data_raw[well_marker] = []
        early_terminated = False
        for i in range(data_point_num):
            read_text = ""
            for j in range(row_num):
                read_text += textIO.readline()
                if len(read_text) == 0:
                    early_terminated = True
                    break
            if early_terminated:
                warnings.warn(
                    f"Plate {metadata['plate_name']} contains less data points than expected, which indicates an interrupted experiment."
                )
                break
            wavelength, temperature, plate_data = SoftMaxPro_Plate.parse_read(
                read_text, wavelengths=[None]
            )
            data_raw["Wavelength"].append(wavelength)
            data_raw["Temperature"].append(temperature)
            for well_marker in read_area:
                data_raw[well_marker].append(
                    plate_data.loc[well_marker[0], well_marker[1]]
                )
            textIO.readline()  # Read the empty line between reads
        signals: list[ContinuousSignal1D] = []
        for value_name in data_raw:
            if value_name == "Wavelength":
                continue
            elif value_name == "Temperature":
                value_unit = "°C"
            else:
                if metadata["mode"] == "Absorbance":
                    value_unit = "AU"
                elif metadata["mode"] == "Fluorescence":
                    value_unit = "RFU"
                else:
                    raise ValueError(f"Unsupported read mode {metadata['mode']}")
            signal = ContinuousSignal1D.from_data(
                data=pd.DataFrame(
                    {
                        "Wavelength": data_raw["Wavelength"],
                        value_name: data_raw[value_name],
                    },
                ),
                axis_name="Wavelength",
                axis_unit="nm",
                value_name=metadata["mode"],
                value_unit=value_unit,
                name=(
                    f"{value_name[0]}{value_name[1]}"
                    if isinstance(value_name, tuple)
                    else value_name
                ),
            )
            signals.append(signal)
        res = cls(
            signals=signals,
            description_annotations=[
                ContDescAnno(
                    name="Wavelength",
                    unit="nm",
                ),
                ContDescAnno(
                    name=metadata["mode"],
                    unit=value_unit,
                ),
            ],
            visible_signal_names=[signal.name for signal in signals],
        )
        res.metadata = metadata
        return res


@dataclass
class SoftMaxPro_Plate_Endpoint(MultiWellPlate):

    metadata: dict = None

    @classmethod
    def _parse_block(cls, textIO: StringIO, metadata: dict):
        """
        This function should be called by the parse_block function of the SoftMaxProPlate class when the read method is "Endpoint".
        Each block corresponds to a plate and contains a single read. This function parses the block and extracts
        the metadata and the data for this plate. The metadata is extracted using the parse_metadata function, and the data is extracted using the parse_read function. The extracted data is organized as a MultiWellPlate object, which contains the data matrix corresponding to the plate layout, and a pandas DataFrame containing the time and temperature information for this read. The metadata and the data are returned as a SoftMaxPlateEndpoint object.
        """
        header = textIO.readline()
        marker, temperature, plate_data = SoftMaxPro_Plate.parse_read(
            textIO.read(), wavelengths=metadata["wavelengths"]
        )
        res = cls(
            data=plate_data,
        )
        res.metadata = metadata
        return res


@dataclass
class SoftMaxPro_Project:
    """
    An object to read data from Molecular Devices SoftMax Pro software. The data is usually exported
    as .txt files with UTF-16 LE encoding. The file contains multiple blocks of data, each block
    corresponds to a plate. Each block starts with a line containing "##BLOCKS= " and ends with
    a line containing "~End". The data in each block is organized in a specific format, which can be parsed
    to extract the kinetic or endpoint data for each plate. The extracted data is returned as a dictionary,
    where the keys are the collection names (plate name + wavelength) and the values are either MultiWellPlate
    objects (for endpoint data) or SoftMaxProPlateKinetic objects (for kinetic data).
    """

    blocks: list[
        SoftMaxPro_Plate_Kinetic | SoftMaxPro_Plate_Spectrum | SoftMaxPro_Plate_Endpoint
    ] = None

    @classmethod
    def read_txt(
        cls,
        txt_file: str | Path,
    ) -> "SoftMaxPro_Project":
        """ """
        with open(txt_file, "r", encoding="utf-16le") as f:
            block_num_line = f.readline()
            block_num = int(re.search(r"##BLOCKS= (\d+).*", block_num_line).group(1))
            blocks = []
            for i in range(block_num):
                block = StringIO()
                while True:
                    line = f.readline()
                    if line != "~End\n":
                        block.write(line)
                    else:
                        break
                blocks.append(block)
            project_metadata = f.readline().strip()

        block_results = []
        for block in blocks:
            block_result = SoftMaxPro_Plate.parse_block(block)
            block_results.append(block_result)

        return cls(blocks=block_results)

    @property
    def metadata(self):
        return [block.metadata for block in self.blocks]

    @property
    def plate_name(self):
        return [metadata["plate_name"] for metadata in self.metadata]

    read_xls = read_txt
