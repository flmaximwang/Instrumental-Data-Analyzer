# InstrumentalDataAnalyzer — CLAUDE.md

## Project Overview

InstrumentalDataAnalyzer: A Python library for analyzing and visualizing scientific instrument data.

## Directory Structure

- `instrumental_data_analyzer/abstract/` — Core abstract data model (signals, matrices, display)
- `instrumental_data_analyzer/concrete/` — Domain-specific signal types
- `instrumental_data_analyzer/instruments/` — Instrument-specific file parsers
- `instrumental_data_analyzer/utils/` — Utility functions
- `tests/` — Unit tests (pytest)
- `docs/` — Documentation

## Architecture Patterns

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
