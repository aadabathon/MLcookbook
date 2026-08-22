# ML Cookbook

A statistics, data science, and classical machine-learning workbench for arbitrary tabular datasets — and a learning environment for building one deliberately.

The core idea: you hand the library a pandas DataFrame, it gives you a typed `Dataset`, and every analysis returns a **structured, JSON-serializable result object** instead of printing or plotting. That keeps the analytical core independent of notebooks, CLIs, and web frontends, so it can later sit unchanged behind a FastAPI service.

```text
DataFrame
   ↓
Dataset / Schema          (core)
   ↓
Preprocessing             (sklearn Pipeline / ColumnTransformer, leakage-safe)
   ↓
Statistics / ML workflows (statistics, regression, ...)
   ↓
Structured Result Objects (frozen dataclasses with .to_dict())
   ↓
Python / Notebook / future REST API / future Web UI
```

## Quick start

```python
import pandas as pd
from mlcookbook import Dataset, correlation, linear_regression

data = Dataset(df)                      # infers a VarType for every column
data.schema()                           # numeric / categorical / boolean / datetime / text / unsupported
data.describe()                         # missingness, cardinality, constant columns
data.set_type("zip_code", "categorical")  # override a wrong inference

corr = correlation(data, "sqm", "price", method="spearman")
corr.statistic, corr.p_value, corr.warnings

result = linear_regression(data, target="price")
result.test_metrics.r2
result.to_dict()                        # plain dict, ready for json.dumps / FastAPI
```

Run `python examples/basic_usage.py` for a complete tour.

## Installation

Requires Python 3.12+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (`source .venv/bin/activate` on Unix)
pip install -e ".[dev]"
pytest
```

## Architecture

`src/`-layout package; each module owns one concern:

| Module | Status | Responsibility |
|---|---|---|
| `core` | ✅ | `Dataset`, schema/type inference (`VarType`), overrides, metadata, result serialization (`Result`, `to_jsonable`) |
| `preprocessing` | ✅ minimal | `build_preprocessor(dataset, columns)` → unfitted sklearn `ColumnTransformer` |
| `statistics` | ✅ one workflow | `correlation` (Pearson/Spearman) — the template for future association tests |
| `regression` | ✅ one workflow | `linear_regression` — the template for future supervised workflows |
| `evaluation` | ✅ minimal | `regression_metrics` → `RegressionMetrics` |
| `classification`, `clustering`, `dimensionality`, `visualization` | 🚧 skeletal | Documented as future work; see `ROADMAP.md` |

### Design rules

1. **Types drive applicability.** Every column gets a `VarType` (a statistical role, not a pandas dtype). Analyses validate their inputs against the schema and refuse or warn rather than silently coercing. Users can override any inference.
2. **Results are data.** Every analysis returns a frozen dataclass mixing in `Result`, which provides `to_dict()` via a recursive serializer (`core/results.py`) that handles nested dataclasses, numpy scalars/arrays, enums, and NaN → `null`. No analysis prints or plots.
3. **Warnings are part of the result.** Dropped rows, skipped columns, constant variables, and assumption issues are surfaced in `result.warnings`, not swallowed.
4. **No leakage.** Preprocessors are returned unfitted and are fitted inside a `Pipeline` on training data only (`regression/linear.py` shows the pattern).
5. **sklearn/scipy do the math.** This library orchestrates and validates; it does not reimplement numerical algorithms.
6. **Small public API.** Only `Dataset`, `VarType`, `correlation`, `linear_regression` (and their result types) are exported at the top level.

### Adding a new workflow

Copy the shape of an existing one:

- A statistical test → `statistics/association.py`: validate types from the schema, compute with scipy, return a frozen result dataclass with warnings.
- A supervised model → `regression/linear.py`: validate target/features, build a `Pipeline` with `build_preprocessor`, train/test split, evaluate both splits via `evaluation`, return a result dataclass.

## Desktop app

A PySide6 desktop shell, launched with a single entry point:

```bash
pip install -e ".[gui]"
python -m mlcookbook
```

Three layers, one seam:

- `gui/` — dumb presentation: widgets render data and emit signals (`FrontPage.recipe_selected`, screens' `back_requested`). Never imports the core. Visual constants live in `gui/theme.py`; the card grid is data-driven from `gui/recipes.py`.
- `src/mlcookbook/` — the analytical core, UI-free as ever (`mlcookbook/__main__.py` is only a trampoline to the app).
- `app.py` (repo root) — **the only module that knows both sides.** Owns the window and navigation (delete-and-redraw screen swapping) and maps a ready recipe's `key` to a screen factory in `SCREEN_FACTORIES`, where core entry points get bound to screens.

To bring a recipe alive: set its `variant="ready"` and `key` in `gui/recipes.py`, build its screen in `gui/screens/`, and register a factory in `app.py`.

A standalone Windows build (PyInstaller) is produced into `dist/MLcookbook/MLcookbook.exe` — see `packaging/` notes in ADAM.md.

## Web portability

The core has no UI or web dependency. A future service is just:

```python
@app.post("/analyze/correlation")
def analyze(...):
    result = mlcookbook.correlation(dataset, x, y)
    return result.to_dict()
```

## Where to start reading

1. `src/mlcookbook/core/schema.py` — `VarType` and inference rules
2. `src/mlcookbook/core/dataset.py` — the `Dataset` entry point
3. `src/mlcookbook/core/results.py` — how results serialize
4. `src/mlcookbook/statistics/association.py` — the statistical-workflow template
5. `src/mlcookbook/regression/linear.py` — the ML-workflow template

Then `ROADMAP.md` for what to build next.
