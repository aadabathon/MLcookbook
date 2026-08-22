# Adam's Handbook

Personal notes for the author. The scaffold is done — from here on, this project is yours. This file is your map: where things are, how to work, and what to build in what order. Edit it freely; it's a living document.

## Daily workflow

```powershell
.venv\Scripts\activate            # if using a venv (recommended)
pip install -e ".[dev,gui]"       # once, or after changing pyproject.toml
pytest                            # run everything (~5s)
pytest tests/test_dataset.py -k schema   # run one thing while iterating
python examples/basic_usage.py    # smoke-test the public API end to end
python -m mlcookbook              # launch the desktop app
```

App architecture: `gui/` is dumb (widgets + signals), `src/mlcookbook` is UI-free, and root `app.py` is the single seam that knows both — it owns the window, delete-and-redraw navigation, and the `SCREEN_FACTORIES` routing table. To wire a new recipe: flip it to `variant="ready"` with a `key` in `gui/recipes.py`, build a screen in `gui/screens/`, register a factory in `app.py`.

Packaging: `python -m PyInstaller --noconfirm --windowed --name MLcookbook --icon gui\assets\mlcookbook.ico --add-data "gui/assets;gui/assets" --paths . --paths src app.py` → `dist/MLcookbook/MLcookbook.exe`. Rebuild after significant changes; the desktop/Start Menu shortcuts point at that exe, so keep the path stable.

Work in small loops: write the test first (or alongside), run it red, make it green, run the full suite, commit.

## The codebase in one paragraph

`Dataset` (core/dataset.py) wraps a DataFrame and infers a `VarType` per column (core/schema.py) — that type, not the pandas dtype, decides which analyses apply. Every analysis is a plain function taking a `Dataset`, validating against the schema, computing with scipy/sklearn, and returning a frozen dataclass that mixes in `Result` (core/results.py) so `.to_dict()` is always JSON-safe. `preprocessing/build_preprocessor` returns an *unfitted* ColumnTransformer that gets fitted inside a Pipeline on training data only — that's the leakage guarantee. Two reference implementations exist: `statistics/association.py` (correlation) and `regression/linear.py`. Everything you build is a variation on one of those two files.

## Rules I set for myself

- Analyses **return objects, never print**. Warnings go in `result.warnings`.
- Validate against `schema()`, don't guess from dtypes.
- Fit preprocessing on train only. If I ever call `.fit` before `train_test_split`, stop.
- scipy/sklearn do the math; I do orchestration, validation, and interpretation.
- Every new result type gets a `json.dumps(result.to_dict())` test.
- Keep `mlcookbook/__init__.py` small — the public API should fit in one glance.
- Files are UTF-8 (a UTF-16 README once broke the build on this machine).

## Suggested build order (my curriculum)

Roughly increasing difficulty; full list with hints in [ROADMAP.md](ROADMAP.md).

1. **Chi-square + Cramér's V** — first solo feature; pure copy of the correlation template. Learn: contingency tables, expected counts, effect size vs p-value.
2. **Logistic regression + classification metrics** — copy of the regression template plus a new `evaluation/` result. Learn: classification metrics, class imbalance, probability vs label predictions.
3. **One-way ANOVA + Kruskal–Wallis** — first analysis with an assumption-based fallback. Learn: when parametric assumptions matter, eta-squared.
4. **Pairwise association scan** — dispatch the right test per column pair from `VarType`. This is the payoff of the schema design and the first "meta" feature.
5. **Cross-validation utility** — structured per-fold results. Learn: variance of estimates, why a single split lies.

After those: regularized regression, trees, k-means, PCA, residual diagnostics, then the FastAPI layer.

## When I use AI agents

Agents read `AGENTS.md` / `CLAUDE.md` and know not to build roadmap features for me. Good uses: reviewing my implementation, explaining a statistical concept, writing tedious test cases after I've written the logic, debugging. Bad use: "implement ANOVA" — that defeats the purpose.

## Scratch space for ideas

- (jot things here)
