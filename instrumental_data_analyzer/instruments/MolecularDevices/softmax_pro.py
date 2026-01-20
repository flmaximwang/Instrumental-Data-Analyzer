import re, tempfile, warnings
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from ...abstract.signal import ContDescAnno
from ...concrete.multi_well_plates import (
    MultiWellPlateSeries,
    MultiWellPlate,
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


class SoftMaxProPlateReader:

    @staticmethod
    def read_txt(
        txt_file: str | Path,
    ) -> dict[str, MultiWellPlate | MultiWellPlateSeries]:

        sep_num = 0
        with open(txt_file, encoding="utf-16 LE") as f:
            for line in f:
                sep_num_tmp = line.count("\t")
                if sep_num_tmp > sep_num:
                    sep_num = sep_num_tmp

        my_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        with open(txt_file, encoding="utf-16 LE") as f_in, open(
            my_temp_file.name, "w"
        ) as f_out:
            for line in f_in:
                sep_num_tmp = line.count("\t")
                extra_sep_num = sep_num - sep_num_tmp
                f_out.write(line[:-1] + ("\t" * extra_sep_num) + "\n")

        xls_data: pd.DataFrame = pd.read_csv(my_temp_file.name, header=None, sep="\t")

        block_num = int(re.search(r"##BLOCKS= (\d+).*", xls_data.iloc[0, 0]).group(1))

        plate_range = []
        counter = 0
        plate_start = None

        for i in range(1, len(xls_data)):
            if counter >= block_num:
                break
            if plate_start is None:
                plate_start = i
            if xls_data.iloc[i, 0] == "~End":
                plate_end = i
                plate_range.append((plate_start, plate_end))
                plate_start = None
                counter += 1

        def parse_plate(
            df: pd.DataFrame, wavelengths: list[int], well_num: int, read_method: str
        ):
            if read_method == "Kinetic":
                time = time_to_seconds(df.iloc[0, 0])
            elif read_method == "Endpoint":
                time = None
            else:
                raise ValueError("Unsupported read method")

            temperature = float(df.iloc[0, 1])
            row_num = well_num_to_row_num(well_num)
            col_num = well_num_to_col_num(well_num)
            plate_dict = {}
            for i, wavelength in enumerate(wavelengths):
                plate_data = df.iloc[
                    0:row_num, 2 + (col_num + 1) * i : 2 + (col_num + 1) * (i + 1) - 1
                ].reset_index(drop=True)
                plate_data = plate_data.astype(float)
                plate_dict[wavelength] = MultiWellPlate(data=plate_data)
            return time, temperature, plate_dict

        def parse_plate_series(df: pd.DataFrame):
            plate_name = df.iloc[0, 1]
            read_method = df.iloc[0, 4]
            read_mode = df.iloc[0, 5]
            data_point_num = int(df.iloc[0, 8])
            if read_method == "Kinetic":
                total_time = int(df.iloc[0, 9])
                interval_seconds = int(df.iloc[0, 10])
            else:
                total_time = None
                interval_seconds = None
            wavelength_num = int(df.iloc[0, 14])
            if wavelength_num > 1:
                wavelengths = list(map(int, df.iloc[0, 15].split()))
            else:
                wavelengths = [int(df.iloc[0, 15])]
            row_start = df.iloc[0, 16]
            row_end = df.iloc[0, 17]
            col_start = df.iloc[0, 19]
            col_end = df.iloc[0, 20]
            plate_type = df.iloc[0, 18]
            row_num = well_num_to_row_num(plate_type)

            metadata = {
                "plate_name": plate_name,
                "read_method": read_method,
                "read_mode": read_mode,
                "data_point_num": data_point_num,
                "total_time": total_time,
                "interval_seconds": interval_seconds,
                "wavelength_num": wavelength_num,
                "wavelengths": wavelengths,
                "row_start": row_start,
                "row_end": row_end,
                "col_start": col_start,
                "col_end": col_end,
                "plate_type": plate_type,
            }

            data = pd.DataFrame(columns=["Time", "Temperature"])
            plate_dict = {wavelength: [] for wavelength in wavelengths}
            actual_data_point_num = (len(df) - 2) // (row_num + 1)
            if actual_data_point_num < data_point_num:
                warnings.warn(
                    "This file contains less data points than expected, which indicates an interrupted experiment."
                )
            for plate_i in range(actual_data_point_num):
                plate_start_row = 2 + plate_i * (row_num + 1)
                plate_end_row = 2 + (plate_i + 1) * (row_num + 1) - 1
                if plate_start_row > len(df):
                    break
                if plate_end_row > len(df):
                    raise ValueError(
                        f"Incomplete plate data. Current row: {plate_start_row} - {plate_end_row}. Total rows: {len(df)}"
                    )
                plate_df = df.iloc[plate_start_row:plate_end_row, :]
                time, temperature, result_tmp = parse_plate(
                    df=plate_df,
                    wavelengths=wavelengths,
                    well_num=plate_type,
                    read_method=read_method,
                )
                for wavelength, plate in result_tmp.items():
                    plate_dict[wavelength].append(plate)
                data.loc[plate_i] = [time, temperature]
            return metadata, data, plate_dict

        result = {}
        for plate_start, plate_end in plate_range:
            df = xls_data.iloc[plate_start:plate_end, :]
            metadata, data, plate_dict = parse_plate_series(df)
            plate_name = metadata["plate_name"]

            if metadata["read_method"] == "Endpoint":
                for wavelength, plate_list in plate_dict.items():
                    collection_name = f"{plate_name}_{wavelength}"
                    result[collection_name] = plate_list[0]
            elif metadata["read_method"] == "Kinetic":
                for wavelength, plate_list in plate_dict.items():
                    collection_name = f"{plate_name}_{wavelength}"
                    result[collection_name] = SoftMaxProPlateKinetic(
                        data=data,
                        matrices=plate_list,
                        metadata=metadata,
                    )
                    if metadata["read_mode"] == "Absorbance":
                        result[collection_name].description_annotations[
                            2
                        ].name = "Absorbance"
                    elif metadata["read_mode"] == "Fluorescence":
                        result[collection_name].description_annotations[2].name = "RFU"
            else:
                raise ValueError("Unsupported read method")

        return result

    read_xls = read_txt


@dataclass
class SoftMaxProPlateKinetic(MultiWellPlateSeries):

    metadata: dict[str, str] = None
    description_annotations: list[ContDescAnno] = field(
        default_factory=lambda: [
            ContDescAnno(
                name="Time",
                unit="s",
            ),
            ContDescAnno(
                name="Temperature",
                unit="°C",
            ),
            ContDescAnno(
                name="Read",
            ),
        ]
    )
