"""SEC 色谱图绘制 (命令行版)

从 Unicorn 7 .txt 文件读取 SEC 数据并生成定制色谱图。

Usage::

    python examples/unicorn7_txt/sec_plot.py <path/to/unicorn7.txt>

Example::

    python examples/unicorn7_txt/sec_plot.py data/2026-07-06_Purification-001_gammaPFD-Elution.txt
"""

from pathlib import Path
import sys

from instrumental_data_analyzer.instruments.Cytiva.unicorn7 import Unicorn7Chrom


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit("Error: missing path to Unicorn7 .txt file")

    txt_path = sys.argv[1]

    my_chrom = Unicorn7Chrom.from_txt(txt_path)[0]
    my_chrom.set_default_annotations()
    my_chrom.visible_signal_names = ["UV 1_280", "Cond", "Fraction"]
    my_chrom.align_axes(0, 35)
    my_chrom.axis_annotation.ticklabel_space = 5
    my_chrom.align_values(["UV 1_280"], -5, 15)
    my_chrom["UV 1_280"].value_annotation.ticklabel_space = 5
    fig, axes = my_chrom.plot()
    fig.savefig(Path(txt_path).with_suffix(".png"), dpi=300, bbox_inches="tight")

    print(f"Saved: {Path(txt_path).with_suffix('.png')}")
