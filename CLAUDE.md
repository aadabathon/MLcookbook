# CLAUDE.md

Follow the guidelines in @AGENTS.md.

Key points: this is Adam's project — do not implement ROADMAP.md features unless explicitly asked (an explicit "build it" / "ship it" means build it end to end); use `py -3.12` on this machine (bare `python` is 3.11 without the project); the user launches the PyInstaller exe from a shortcut, so rebuild it after GUI/core changes they should see; do not commit unless asked; analyses return frozen `Result` dataclasses (never print); preprocessing must be leakage-safe; run `pytest` before finishing; keep files UTF-8.

Current state (2026-10-12): the data pipeline (configurable `PreprocessingConfig`, recorded `PreprocessingSummary`, training-only cross-validation, import quality checks), the modules (multiclass classifiers, tree/forest regression, SVM and neural networks for both tasks, ridge/lasso + residual diagnostics, k-means, PCA, chi-square, ANOVA/Kruskal-Wallis, association scan), nine ready GUI recipes, and CI (ruff + pytest on GitHub Actions) are built and committed. See `README.md` for usage, `ADAM.md` for the verification record and build command, and `AGENTS.md` for architecture and tests. Open items in `ROADMAP.md` are proposals, not authorization.

Session handoff: see the 2026-10-12 note in `ADAM.md`. Adam's stated priority is shipping a working product; he owns the math, and the pipeline/frontend are delegated. He plans a separate PyTorch sandbox for LLMs/modern ML; keep PyTorch out of this project.
