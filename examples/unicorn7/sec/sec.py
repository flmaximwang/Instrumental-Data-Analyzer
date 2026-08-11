"""SEC chromatogram plotter.

Usage:
    python sec.py <path/to/unicorn7.txt>

Example:
    python sec.py "data/2026-07-06_Purification-001_gammaPFD-Elution.txt"
"""

import argparse
from pathlib import Path

import sys
import matplotlib.pyplot as plt
from instrumental_data_analyzer.instruments.Cytiva.unicorn7 import (
    Unicorn7Project,
    Chrom,
    ContinuousSignal1D,
)


def parse_args():

    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", type=str, required=True, help="请输入 SEC 的 txt")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="输出图片路径，默认与输入 txt 相同, 只是后缀改为 txt",
    )
    p.add_argument("--column-type", type=str, required=True, help="柱子的名称")
    p.add_argument("--buffer", type=str, required=True, help="缓冲液成分")
    p.add_argument("--flow-rate", type=str, required=True, help="流速")
    p.add_argument("--sample-name", type=str, required=True, help="样品名称")
    p.add_argument("--sample-volume", type=str, required=True, help="样品体积")
    p.add_argument(
        "--absorption-channels", type=int, default=1, help="吸收信号通道数量"
    )

    return p.parse_args()


def parse_output(input: str, output: str | None):
    if output is None:
        return Path(input).with_suffix(".png")
    else:
        return output


def parse_column_type(colume_type: str):

    COLUMN_SPECS = {
        "10/300": {
            "axis_range": 36,
            "axis_space_major": 5,
            "axis_space_minor": 1,
            "axis_digits": 0,
        },
        "16/600": {
            "axis_range": 180,
            "axis_space_major": 20,
            "axis_space_minor": 5,
            "axis_digits": 0,
        },
    }

    flag = False
    for column_spec in COLUMN_SPECS:
        if column_spec in colume_type:
            flag = True
            break
    if not flag:
        raise ValueError(
            f"COLUMN_TYPE must contain a column spec in {list(COLUMN_SPECS.keys())}"
        )

    return colume_type, COLUMN_SPECS[column_spec]


def main():

    args = parse_args()
    column_type, column_spec = parse_column_type(args.column_type)
    output = parse_output(args.input, args.output)

    my_chrom: Chrom = Unicorn7Project.read_txt(args.input)[0]
    my_chrom.set_default_annotations()

    my_chrom.align_axes(0, column_spec["axis_range"])
    my_chrom.axis_annotation.ticklabel_space_major = column_spec["axis_space_major"]
    my_chrom.axis_annotation.ticklabel_space_minor = column_spec["axis_space_minor"]
    my_chrom.axis_annotation.ticklabel_digits = column_spec["axis_digits"]

    visible_signal_names = []

    for i in range(args.absorption_channels):
        for signal_name in my_chrom.signal_names:
            if f"UV {i+1}" in signal_name:
                visible_signal_names.append(signal_name)
                break

    visible_signal_names.append("Cond")
    if "Fraction" in my_chrom.visible_signal_names:
        visible_signal_names.append("Fraction")

    my_chrom.visible_signal_names = visible_signal_names

    for signal_name in visible_signal_names:
        if signal_name in ["Fraction"]:
            continue
        signal: ContinuousSignal1D = my_chrom[signal_name]
        sliced_value = signal.slice_value_by_axis(0, column_spec["axis_range"])
        max_value = max(sliced_value)
        min_value = min(sliced_value)
        sliced_value_difference = max_value - min_value
        value_range = (
            min_value - 0.1 * sliced_value_difference,
            max_value + 0.1 * sliced_value_difference,
        )
        ticklabel_space_major = sliced_value_difference / 5
        ticklabel_space_minor = sliced_value_difference / 25
        my_chrom.align_values([signal_name], *value_range)
        signal.value_annotation.ticklabel_space = (
            ticklabel_space_major,
            ticklabel_space_minor,
        )
        if "UV" in signal_name:
            signal.value_annotation.ticklabel_digits = 2
        elif "Cond" in signal_name:
            signal.value_annotation.ticklabel_digits = 3
        else:
            raise ValueError(f"{signal_name} not supported")

    my_chrom.name = f"[{column_type}] {args.buffer} ({args.flow_rate}): {args.sample_name} ({args.sample_volume})"
    my_chrom.plot_args.ax_size = (21, 7)
    my_chrom.plot_args.axis_shift = 0.11
    fig, axes = my_chrom.plot()

    fig.savefig(output, dpi=300)


if __name__ == "__main__":
    main()
