import sys

sys.path.append("..")

import unittest
import os
import pandas as pd
import numpy as np
from instrumental_data_analyzer import abstract


class TestSignal(unittest.TestCase):

    def test_from_csv(self):
        abstract.Signal.from_csv("tests/data/table_1.csv")


class TestSignal1D(unittest.TestCase):

    def test_from_csv(self):
        signal = abstract.Signal1D.from_csv("tests/data/table_1_formatted.csv")
        self.assertEqual(signal.get_axis_name(), "column 1 (unit)")
        self.assertEqual(signal.get_axis_unit(), None)
        self.assertEqual(signal.get_value_name(), "column 2 (unit)")
        self.assertEqual(signal.get_value_unit(), None)
        signal = abstract.Signal1D.from_csv(
            "tests/data/table_1_formatted.csv",
            detect_axis_name_and_unit=True,
            detect_value_name_and_unit=True,
        )
        self.assertEqual(signal.get_axis_name(), "column 1")
        self.assertEqual(signal.get_axis_unit(), "unit")
        self.assertEqual(signal.get_value_name(), "column 2")
        self.assertEqual(signal.get_value_unit(), "unit")

    def test_get_axis(self):
        signal = abstract.Signal1D.from_csv("tests/data/table_2.csv")
        self.assertEqual(signal.get_axis().iloc[0], 1)

    def test_get_values(self):
        signal = abstract.Signal1D.from_csv("tests/data/table_2.csv")
        self.assertEqual(signal.get_values().iloc[0], 2)

    def test_get_axis_between(self):
        signal = abstract.Signal1D.from_csv("tests/data/table_2.csv")
        print("")
        print(signal.get_axis_between(1, 2))
        self.assertEqual(signal.get_axis_between(1, 2).iloc[0], 1.1)

    def test_get_values_between(self):
        signal = abstract.Signal1D.from_csv(
            "tests/data/table_2.csv",
            detect_axis_name_and_unit=False,
            detect_value_name_and_unit=False,
        )
        self.assertEqual(signal.get_values_between(1, 2).iloc[0], 2)

    def test_preview(self):
        signal = abstract.Signal1D.from_csv("tests/data/table_2.csv")
        # 断言以下代码运行时会出现 TypeError
        with self.assertRaises(TypeError):
            signal.preview(
                export_path="tests/out/table_2_preview_with_Signal1D"
            )  # No curve will be observed because the plot_at method is not implemented in the abstract class

    def test_axis_setter_converts_int_to_float(self):
        """Assigning float values to an int64 axis column via setter should not raise TypeError."""
        df = pd.DataFrame({"time_s": [0, 60, 120, 180], "absorbance": [0.1, 0.2, 0.3, 0.4]})
        # Force axis column to int64 like SoftMax Pro parsed data
        df.iloc[:, 0] = df.iloc[:, 0].astype(np.int64)
        signal = abstract.Signal1D(
            data=df,
            name="test_signal",
            description_annotations=[
                abstract.ContDescAnno(name="Time", unit="s"),
                abstract.ContDescAnno(name="Absorbance", unit="AU"),
            ],
        )
        # This used to raise pandas.errors.LossySetitemError / TypeError
        signal.axis = signal.axis / 60
        self.assertEqual(signal.axis.iloc[0], 0.0)
        self.assertEqual(signal.axis.iloc[1], 1.0)
        self.assertEqual(signal.axis.dtype, np.float64)


class TestNumericSignal1D(unittest.TestCase):

    def test_get_peak_between(self):
        signal = abstract.ContinuousSignal1DBase.from_csv(
            "tests/data/table_2.csv",
            detect_axis_name_and_unit=False,
            detect_value_name_and_unit=False,
        )
        self.assertEqual(signal.get_peak_between(1, 2), (1.1, 2))

    def test_preview(self):
        signal = abstract.ContinuousSignal1DBase.from_csv("tests/data/UV_Vis_1.csv")
        self.assertEqual(signal.get_axis_name(), "Wavelength (nm)")
        self.assertEqual(signal.get_axis_unit(), None)
        self.assertEqual(signal.get_value_name(), "1mm Absorbance")
        self.assertEqual(signal.get_value_unit(), None)
        signal.set_axis_name("Wavelength")
        signal.set_axis_unit("nm")
        signal.set_value_name("1 mm Absorbance")
        signal.set_value_unit(None)
        self.assertEqual(signal.get_axis_name(), "Wavelength")
        self.assertEqual(signal.get_axis_unit(), "nm")
        self.assertEqual(signal.get_value_name(), "1 mm Absorbance")
        self.assertEqual(signal.get_value_unit(), None)
        signal.preview(export_path="tests/out/UV_Vis_1_preview_with_NumericSignal1D")
        signal = abstract.ContinuousSignal1DBase.from_csv(
            "tests/data/chromatography_UV_1.csv"
        )
        self.assertEqual(signal.get_axis_name(), "ml")
        self.assertEqual(signal.get_axis_unit(), None)
        self.assertEqual(signal.get_value_name(), "mAU")
        self.assertEqual(signal.get_value_unit(), None)
        signal.set_axis_name("Volume")
        signal.set_axis_unit("ml")
        signal.set_value_name("280 nm")
        signal.set_value_unit("mAU")
        signal.preview(
            export_path="tests/out/chromatography_UV_1_preview_with_NumericSignal1D"
        )


class TestFractionSignal(unittest.TestCase):

    def test_preview(self):
        fraction_signal = abstract.FractionSignal.from_csv(
            "tests/data/fraction_1.csv", name=None, axis_unit="ml"
        )
        fraction_signal.set_axis_name("Volume")
        fraction_signal.preview(
            rotation=90,
            text_shift=(0, 0.05),
            export_path="tests/out/fraction_1_preview_with_FractionSignal.png",
        )

    def test_export(self):
        fraction_signal = abstract.FractionSignal.from_csv(
            "tests/data/fraction_1.csv", name=None, axis_unit="ml"
        )
        fraction_signal.set_axis_name("Volume")
        with self.assertRaises(Exception):
            fraction_signal.export(
                "tests/out/fraction_1_export_with_FractionSignal.csv"
            )
        fraction_signal.export(
            "tests/out/fraction_1_export_with_FractionSignal.csv", mode="replace"
        )


if __name__ == "__main__":
    unittest.main()
