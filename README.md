# Instrumental Data Analyzer

一个用于处理、分析和可视化各种仪器数据的 Python 库，为科学实验数据提供统一的数据模型和处理框架。

## 功能特性

- **统一数据模型**：通过 `Signal` → `Signal1D` / `Signal2D` → 具体仪器信号的层次化抽象，统一管理所有仪器数据
- **内置描述注释系统**：自动管理数据的物理含义（名称、单位、范围、刻度标签），简化绘图和展示
- **自动化绘图**：支持多种绘图模式（单图多线、多轴对比、分面绘图），自动处理轴缩放和标签
- **数据操作**：支持信号代数运算（加/减/乘/除）、插值、峰值检测、区域积分、平滑滤波、基线校正
- **仪器数据解析**：内置多家厂商仪器数据导入器

## 支持的仪器

| 厂商 | 仪器 / 软件 | 数据格式 | 模块 |
|------|------------|---------|------|
| **Agilent** | ChemStation HPLC | CSV (UTF-16 LE) | `instruments.Agilent.chemstation_processor` |
| **CH Instruments** | CHI660E 电化学工作站 | TXT | `instruments.CHI.chi660e` |
| **Cytiva (GE Healthcare)** | ÄKTA / Unicorn 5, 7 | XLS, ASC, TXT (UTF-16 LE) | `instruments.Cytiva.unicorn5`, `unicorn7` |
| **Molecular Devices** | SoftMax Pro | TXT (UTF-16 LE), XLS | `instruments.MolecularDevices.softmax_pro` |
| **Thermo Fisher** | NanoDrop 2000 | TSV | `instruments.ThermoFisher.nanodrop2000` |
| **Thermo Fisher** | Multiskan FC | TXT | `instruments.ThermoFisher.multiscan_fc` |

## 安装

```bash
pip install git+https://github.com/flmaximwang/Instrumental-Data-Analyzer.git
```

依赖：Python ≥ 3.11, numpy, pandas, matplotlib, scipy, scikit-learn, impedance

## 架构总览

```
instrumental_data_analyzer/
├── abstract/          # 抽象数据模型层 —— 核心框架
│   ├── signal.py              # Signal 基类 + DescAnno 注释系统
│   ├── signal_1d.py           # 1D 信号 (Signal1D, ContinuousSignal1D, DiscreteSignal1D)
│   ├── signal_collection.py   # SignalCollection 集合基类
│   ├── signal_1d_collection.py# Signal1DCollection 集合
│   ├── signal_2d.py           # 2D 信号 (框架预留)
│   ├── matrix.py              # Matrix 矩阵
│   ├── matrix_collection.py   # MatrixSeries 矩阵序列
│   └── display.py             # SignalPlotArgs 绘图参数
├── concrete/          # 具体领域信号类型
│   ├── spectroscopy.py        # 光谱 (Spectrum, AbsorbSpec)
│   ├── chromatography.py      # 色谱 (Chrom, ChromSig, ChromLog)
│   ├── kinetics.py            # 动力学 (KineticCurve)
│   ├── voltammetry.py         # 伏安法 (Voltammegram)
│   ├── impedance_spectroscopy.py # 阻抗谱 (ImpedanceSpectrum)
│   └── multi_well_plates.py   # 多孔板 (MultiWellPlate, plate layouts)
├── instruments/       # 仪器专用数据解析器
│   ├── Agilent/
│   ├── CHI/
│   ├── Cytiva/ & GE_Heathcare/
│   ├── MolecularDevices/
│   └── ThermoFisher/
└── utils/             # 工具函数
    ├── name_utils.py          # 名称去重
    ├── path_utils.py          # 路径处理
    └── transform_utils.py     # 数据缩放
```

## 核心数据模型

### DescAnno（描述注释系统）

每个信号维度都带有描述注释，包含名称、单位、范围、刻度等信息：

```python
from instrumental_data_analyzer import ContDescAnno

axis_annotation = ContDescAnno(
    name="Wavelength",           # 物理量名称
    unit="nm",                   # 单位
    limit=(200, 800),            # 数据范围
    margin=(0.05, 0.95),         # 绘图边距
)
axis_annotation.ticklabel_space = 100  # 自动计算刻度标签
print(axis_annotation.ticklabels)      # ['200', '300', '400', ...]
print(axis_annotation.ticks)           # 缩放到 margin 后的刻度位置
```

### Signal（信号基类）

```python
from instrumental_data_analyzer import Signal

# 从 CSV 文件加载
sig = Signal.from_csv("data.csv")

# 自定义创建
import pandas as pd
sig = Signal(
    data=pd.DataFrame({"x": [1,2,3], "y": [4,5,6]}),
    name="My Signal"
)

# 导出
sig.to_csv("export.csv")
```

### Signal1D（一维信号）

所有一维仪器数据的基础类型，包含 axis（轴，连续）和 value（值，连续或离散）：

```python
from instrumental_data_analyzer import ContinuousSignal1D

# 从 CSV 读取（自动检测轴名称和单位）
sig = ContinuousSignal1D.from_csv("uv_vis_spectrum.csv")

# 从 DataFrame 创建
sig = ContinuousSignal1D.from_data(
    data=df,
    axis_name="Wavelength",
    axis_unit="nm",
    value_name="Absorbance",
    value_unit="AU"
)

# 插值访问
value_at_540nm = sig[540]

# 峰检测
peak_axis, peak_value = sig.get_peak_between(500, 600)

# 信号运算
result = sig1 + sig2      # 插值后相加
result = sig1 * 2         # 乘以标量
result = sig1 - blank     # 扣除空白
result = sig1 / sig2      # 插值后相除

# 信号滤波
sig.clean_by_moving_average(z_threshold=3)   # 去除异常值
sig.filter_by_moving_average(window=5)        # 移动平均平滑
```

### SignalCollection（信号集合）

用于管理多个信号并统一比较和可视化：

```python
from instrumental_data_analyzer import Signal1DCollection

# 从 CSV 文件夹导入
collection = Signal1DCollection.from_folder("data/folder/")

# 从相似信号列表创建
collection = Signal1DCollection.from_similar_signals([sig1, sig2, sig3])

# 访问信号
sig = collection["Signal Name"]
collection.rename_signal("old_name", "new_name")
collection.remove_signal("Signal Name")

# 对齐坐标轴
collection.align_axes(0, 30)
collection.align_values(["sig1", "sig2"], 0, 1)

# 设置默认注释
collection.set_default_annotations()

# 绘图
fig, ax = collection.plot()  # 模式 0: 单图多线
```

## 使用示例

完整的使用示例见 `examples/` 目录，每个仪器都包含可运行的示例脚本：

| 仪器 | 目录 | 数据来源 |
|------|------|---------|
| **NanoDrop 2000** (Protein A280) | `examples/nanodrop2000_protein-a280/` | 实测 TSV 数据 |
| **ChemStation HPLC** | `examples/chemstation_hplc/` | 实测 CSV 数据 (5 波长 + 馏分) |
| **ÄKTA / Unicorn 5** | `examples/unicorn5_xls/` | 合成 ASC 数据 (UV, Cond, pH, Conc) |
| **ÄKTA / Unicorn 7** | `examples/unicorn7_txt/` | 实测 TXT 数据 |
| **SoftMax Pro** (Spectrum) | `examples/softmax_pro_spectrum/` | 实测 TXT 数据 (96 孔光谱) |
| **SoftMax Pro** (Kinetics) | `examples/softmax_pro_kinetics/` | 实测 TXT 数据 (96 孔动力学，6 波长，蛋白-Hemin 结合) |
| **CHI660E** (CV) | `examples/chi660e_cv/` | 合成铁氰化物 CV 数据 |
| **CHI660E** (EIS) | `examples/chi660e_eis/` | 合成 Randles 电路 EIS 数据 |
| **Multiskan FC** | `examples/multiscan_fc/` | 合成 96 孔板数据 (BSA 标准曲线) |

运行示例：

```bash
cd examples/chemstation_hplc
python example.py
```

每项运行完成后会在同目录生成 `example.png`（动力学示例生成 `example_soret.png` 和 `example_visible.png`），展示数据可视化结果。

### Agilent ChemStation

```python
from instrumental_data_analyzer.instruments import ChemStChrom

chrom = ChemStChrom.from_exported_directory("path/to/chemstation_export/")
uv_signal = chrom["280"]
uv_signal.plot_at(ax)
```

### Unicorn (ÄKTA) 色谱

```python
from instrumental_data_analyzer.instruments import Unicorn5Chrom

chrom = Unicorn5Chrom.from_xls("chromatogram.xls")
# 或
chrom = Unicorn5Chrom.from_asc("chromatogram.asc")

uv_signal = chrom["UV"]
cond_signal = chrom["Cond"]
```

### Molecular Devices SoftMax Pro

```python
from instrumental_data_analyzer.instruments import SoftMaxPro_Project

project = SoftMaxPro_Project.read_txt("plate_reader_data.txt")
plate = project["Plate Name"]              # 获取指定板
kinetic_data = plate["A1"]                 # 动力学曲线
kinetic_data.plot_at(ax)
```

### NanoDrop 2000

```python
from instrumental_data_analyzer.instruments import Nanodrop2000Workbook

workbook = Nanodrop2000Workbook.from_tsv("spectra.tsv")
workbook.set_default_annotations(mode="UV-Vis")
fig, ax = workbook.plot()
```

### CHI660E 电化学

```python
from instrumental_data_analyzer.instruments import chi660eCV, chi660eEIS

cv = chi660eCV.from_exported_files(["file1.txt", "file2.txt"])
eis = chi660eEIS.from_exported_files({"file.txt": "Sample Name"})
```

## 绘图模式

`Signal1DCollection` 支持三种绘图模式：

| 模式 | 描述 | 使用场景 |
|------|------|---------|
| 0 | 共享坐标轴，所有信号叠加 | 快速比较多个信号 |
| 1 | 共享 x 轴，每个信号独立 y 轴 | 不同量纲的信号（如 UV + Cond + pH） |
| 2 | 分面绘图，每个信号一个子图 | 信号密度高时分别查看 |

```python
collection.plot_args.mode = 1  # 切换模式
collection.plot_args.figsize = (12, 4)
collection.plot_args.legend_cols = 2
fig, axes = collection.plot()
```

## 孔板数据处理

```python
from instrumental_data_analyzer import MultiWellPlate

# 自动检测板型 (6, 12, 24, 48, 96, 384)
plate = MultiWellPlate(data=df)

# 读取孔值
value = plate["A", 1]

# 标准曲线法校准和测量
fig, ax, result, table = plate.calibration_and_measurement(
    calibration_values=[0, 0.1, 0.5, 1.0],
    calibration_markers=[("A", 1), ("A", 2), ("A", 3), ("A", 4)],
    measurement_markers=[("B", 1), ("B", 2), ("B", 3)],
    xlabel="Concentration (mM)",
    ylabel="Absorbance",
)
```

## 开发

```bash
# 使用 poetry 安装开发依赖
poetry install

# 运行测试
python -m pytest tests/
```

## 架构模式与开发约定

### Class Hierarchy

```
Signal (dataclass)          — Base: .data (pd.DataFrame), .name, .description_annotations
  Signal1D                  — 2 columns: axis (cont) + value
    ContinuousSignal1D      — Interpolation, arithmetic, peak detection, smoothing
    DiscreteSignal1D        — Discrete values (fractions)
    SegmentedSignal1D       — Multi-segment data (voltammetry)
      SegmentedContinuousSignal1D
```

### Key Conventions

1. **dataclass + property pattern**: All signal classes are `@dataclass`. Properties expose axis/value/annotation access.
2. **Normalized plotting**: All plotting uses [0,1] normalized coordinates. Real data → plot coordinates via `rescale()` + `ContDescAnno` (limit + margin).
3. **Column annotations**: Each DataFrame column has a `DescAnno` with name, unit, limit, margin, tick settings.
4. **Collection pattern**: `SignalCollection` acts as a dict-like container with `__getitem__` by name.
5. **Instrument parsers**: Static factory methods named `from_raw_export()`, `from_exported_directory()`, `from_xls()`, etc.

### Important Notes

- Tests use `sys.path.append("..")` before imports
- Test data is in `tests/data/`, output in `tests/out/`
- Some modules have `archive` suffix (legacy code) vs processor suffix (new code)
- `GE_Heathcare/` directory has a typo (should be `Healthcare`) — maintained for compatibility

## 许可

本项目基于 [LICENSE](LICENSE) 文件中的许可条款发布。
