# Agent Guidelines — ML Cookbook

This repo is a **learning environment owned by Adam, the author**. Agents assist; they do not take over. The default posture is: small, surgical changes that preserve the existing patterns, and no unsolicited feature expansion.

## Prime directive

**Do not implement ROADMAP.md items unless explicitly asked.** The roadmap is Adam's personal curriculum — building those features is the point of the project. If asked to "help" with one, prefer explaining, reviewing, or pairing over writing the whole thing.

## Project shape

- `src/` layout, package `mlcookbook`, Python 3.12+, hatchling build.
- Install: `pip install -e ".[dev]"` · Test: `pytest` · Demo: `python examples/basic_usage.py`
- Dependencies: pandas, numpy, scipy, scikit-learn, pytest. Do not add dependencies (statsmodels, matplotlib, etc.) without asking.
- `classification/`, `clustering/`, `dimensionality/`, `visualization/` are intentionally skeletal. Leave them that way unless told otherwise.

## Architecture rules (non-negotiable)

1. **Types drive applicability.** Columns get a `VarType` (`core/schema.py`) — a statistical role, not a pandas dtype. Analyses validate inputs against `Dataset.schema()` and raise or warn; never silently coerce.
2. **Results are data.** Every analysis returns a frozen dataclass mixing in `Result` (`core/results.py`) with a `warnings: list[str]` field. No analysis prints, plots, or logs. `result.to_dict()` must survive `json.dumps`.
3. **Warnings are part of the result.** Dropped rows, skipped columns, constant variables, assumption issues → `result.warnings`.
4. **No leakage.** Preprocessors come from `preprocessing.build_preprocessor()` unfitted and are fitted inside a sklearn `Pipeline` on training data only. Never fit anything on the full dataset before splitting.
5. **sklearn/scipy do the math.** Orchestrate and validate; never reimplement numerical algorithms.
6. **Small public API.** Only add to `mlcookbook/__init__.py` deliberately. Everything else is imported from submodules.
7. **No enterprise patterns.** No factories, DI containers, deep inheritance, or clever metaprogramming. Type hints, dataclasses, composition, small modules.

## Templates to copy

- New statistical test → follow `statistics/association.py` (validate types → compute with scipy → frozen result dataclass with warnings).
- New supervised workflow → follow `regression/linear.py` (validate target/features → `build_preprocessor` in a `Pipeline` → train/test split → metrics via `evaluation/` → result dataclass).

## Testing

- Every behavior change needs pytest coverage; shared fixtures live in `tests/conftest.py` (`mixed_df`, `linear_df`).
- Every new result type needs a `json.dumps(result.to_dict())` round-trip test.
- Run `pytest` before declaring anything done.

## Windows gotcha

This machine is Windows. **Files must be UTF-8** — the original README was UTF-16 and broke the hatchling build. If a build fails with `UnicodeDecodeError`, check file encodings first.

## Docs to keep in sync

If you change the public API, architecture, or add a module: update `README.md`. If you complete or obsolete a roadmap item (only when asked): update `ROADMAP.md`.
