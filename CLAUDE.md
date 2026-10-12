# CLAUDE.md

Follow the guidelines in @AGENTS.md.

Key points: this is Adam's learning project — do not implement ROADMAP.md features unless explicitly asked (an explicit "build it" means build it end to end); use `py -3.12` on this machine (bare `python` is 3.11 without the project); the user launches the PyInstaller exe from a shortcut, so rebuild it after GUI/core changes they should see; do not commit unless asked; analyses return frozen `Result` dataclasses (never print); preprocessing must be leakage-safe; run `pytest` before finishing; keep files UTF-8.

Current data-input milestone: CSV and Parquet are supported by all four ready recipes, with PyArrow, preview/schema controls, and background import/analysis. See `README.md` for usage, `ADAM.md` for the implementation/verification record and build command, and `AGENTS.md` for architecture and tests. Future options in `ROADMAP.md` are discussion proposals, not authorization to implement them.

Random forests are now ready: `classification/forest.py` exports `random_forest` and `RandomForestResult`, with ROC/importance/forest-size reporting and the shared CSV/Parquet workflow. All three classifiers remain binary-only; regression forests and multiclass support are still roadmap work.

Session handoff: see the 2026-09-27 end-of-day note in `ADAM.md`. The last code verification passed 110 tests and the rebuilt executable was launch-tested. Work remains uncommitted; preserve earlier changes and wait for Adam to select the next implementation task.
