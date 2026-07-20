"""SEC chromatogram plotter.

Usage:
    python sec.py <path/to/unicorn7.txt>

Example:
    python sec.py "data/2026-07-06_Purification-001_gammaPFD-Elution.txt"
"""

from pathlib import Path

import sys
from instrumental_data_analyzer.instruments.Cytiva.unicorn7 import Unicorn7Project
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit("Error: missing path to Unicorn7 .txt file")

txt_path = sys.argv[1]

my_chrom = Unicorn7Project.read_txt(txt_path)[0]
my_chrom.set_default_annotations()
my_chrom.visible_signal_names = ["UV 1_280", "Cond", "Fraction"]
my_chrom.align_axes(0, 35)
my_chrom.axis_annotation.ticklabel_space_major = 5
my_chrom.align_values(["UV 1_280"], -10, 210)
my_chrom["UV 1_280"].value_annotation.ticklabel_space_major = 20
fig, axes = my_chrom.plot()
fig.savefig(Path(txt_path).with_suffix(".png"), dpi=300, bbox_inches="tight")
