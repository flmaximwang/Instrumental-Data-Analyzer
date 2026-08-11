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
        "10/300": 24,
        "16/600": 120,
    }

    for column_spec in COLUMN_SPECS:
        if column_spec in colume_type:
            break

    return colume_type, COLUMN_SPECS[column_spec]


def main():

    args = parse_args()
    column_type, column_volume = parse_column_type(args.column_type)
    output = parse_output(args.input, args.output)

    my_chrom: Chrom = Unicorn7Project.read_txt(args.input)[0]
    my_chrom.set_default_annotations()

    my_chrom.align_axes(0, column_volume * 1.5)
    my_chrom.axis_annotation.ticklabel_space_major = column_volume * 1.5 / 10
    my_chrom.axis_annotation.ticklabel_space_minor = column_volume * 1.5 / 50

    visible_signal_names = []

    for i in range(args.absorption_channels):
        for signal_name in my_chrom.signal_names:
            if f"UV_{i+1}" in signal_name:
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
        sliced_data = signal.slice_axis(0, column_volume * 1.5)
        max_value = max(sliced_data[:, 1])
        min_value = min(sliced_data[:, 1])
        sliced_value_difference = max_value - min_value
        value_range = (
            min_value - 0.1 * sliced_value_difference,
            max_value + 0.1 * sliced_value_difference,
        )
        ticklabel_space_major = value_range / 5
        ticklabel_space_minor = value_range / 25
        my_chrom.align_values([signal_name], *value_range)
        signal.value_annotation.ticklabel_space = (
            ticklabel_space_major,
            ticklabel_space_minor,
        )
        signal.value_annotation.ticklabel_digits = 3

    my_chrom.name = f"[{column_type}] {args.buffer} ({args.flow_rate}): {args.sample_name} ({args.sample_volume})"
    fig, axes = my_chrom.plot()

    fig.savefig(output, dpi=300)


if __name__ == "__main__":
    main()
