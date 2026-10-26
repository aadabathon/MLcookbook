# UI roadmap

Requested by Adam on 2026-10-24. UI and dataset workflow improvements take priority over REST endpoints, the graph workshop, and further feature expansion.

## Product direction

ML Cookbook is an application for general graduate-level statistics, data science, and ML workflows. Make data, configuration, execution, and results straightforward to locate and operate. Adam's learning environment as an author does not imply a tutorial-first product.

Keep native PySide6 / Qt Widgets, the recipe catalog/cards, and the analytical widgets. The previous interface concepts are layout references; Adam's subsequent corrections are the product brief. Use a persistent sidebar with **Data at the top**, coherent typography/color/emphasis, accessible technical controls, and concise reference material. A mandatory wizard or permanent lesson panel is outside this brief.

### Design rules

- Treat the dataset as a first-class workspace object. Make import, preview, column roles, quality checks, and source identity directly accessible.
- Use one consistent sans-serif type system, limited sizes/weights, sentence-case labels, and predictable alignment. Remove gratuitous eyebrow labels, letter spacing, decorative symbols, redundant badges, and arbitrary bolding.
- Give color a purpose: selection/actions, warnings/errors, or plotted-series identity. Preserve contrast and consistent category/series mappings.
- Retain method names and graduate-level controls. Group target, predictors, preprocessing, validation and reproducibility settings coherently. Add explanations only where they help a decision.
- Keep warnings, stale-result provenance, execution status and cancellation discoverable. Visual cleanup must preserve the information needed to trust a result.
- Make examples explicit. Ordinary navigation must not silently load or train on a preset dataset.

## Delivery order

Implement one explicitly requested stage at a time, with relevant tests, native inspection, documentation, and a runnable preview build when application code changes. Each stage is independently reviewable and reversible. This document creates the plan; it does not mark the redesign as implemented.

| Stage | Scope | Review checkpoint |
|---|---|---|
| 0 | Baseline and rollback | Existing source/work, build provenance, and recovery path are accounted for |
| 1 | Shared visual system | Existing screens have consistent typography, spacing, contrast and emphasis |
| 2 | Persistent sidebar and Data workspace | Import and inspect an ordinary dataset before choosing a recipe |
| 3 | Recipe setup and explicit examples | Keep the catalog; make own-data work the normal path and examples opt-in |
| 4 | Forms, reports and technical reference | Organize controls/results without reducing analytical depth |
| 5 | Startup and first-use responsiveness | Defer or schedule expensive work, with comparable measurements |
| 6 | Compatibility and Windows delivery | Existing workflows/snapshots work and the reviewed build is recoverable |

## 0. Baseline and rollback

The initial checkpoint was taken from `main` at `68846f8` with uncommitted Movement Study integration and untracked wearable modules. HEAD alone did not capture that application; the archives below preserve that dirty working tree. A chat fork does not isolate Git changes. The delivery PR records the Movement prerequisite and UI stages separately.

- [x] Preserve all tracked and non-ignored untracked source files without changing Git: 112 files, per-file SHA-256 hashes, the base revision/status, and a tracked binary patch. The archive was read back and verified.
- [x] Establish the current test baseline: **456 pytest tests passed**, Ruff passed; 60 existing sklearn SVC deprecation warnings. No application behavior changed in this planning pass.
- [x] Record installed runtime/build versions, a package freeze, baseline validation, and the current executable's hash/timestamp with the checkpoint. This records the bundle's identity without claiming it has been copied.
- [x] Keep the existing complete `dist/MLcookbook` bundle intact while UI candidates are built into `dist/ui-preview/MLcookbook`. All 6,216 original files were hashed and verified unchanged after packaging/runtime checks. The original EXE timestamp is 2026-10-23 UTC, before the source checkpoint; it is not claimed to match that checkpoint. The executable and `_internal` stay together.
- [x] Capture representative baseline views: 16 native Windows views at 100% and 125%, covering imported data, regression/classification, trees, long tables, Local LLMs, Movement Study and dialogs. Captures and source-root verification are alongside the source checkpoint in `ui-stage1/`.

Local source checkpoint, outside Git:

```text
C:/Users/adams/.codex/visualizations/2026/10/25/01a17148-2348-7b22-af8a-4a0517a8d864/ui-baseline-2026-10-24/
    source.zip
    manifest.json
    tracked-changes.patch
```

The archive excludes ignored environments, weights, build output, `dist`, and Git metadata. It is a source recovery checkpoint, not a packaged-app backup. No checkpoint commit or branch was created.

**Rollback:** record UI-specific files/patches for every stage. Reverse only that stage's changes; inspect mixed-file changes hunk by hunk so Movement Study work and subsequent unrelated edits survive. Remove only newly introduced UI files after checking for later work. Never use a broad reset, clean, or restore of every modified file. If Adam requests commits, keep UI stages in separate commits for targeted reversion.

For a rejected preview, the existing shortcut bundle remains available. Before promoting an accepted candidate, preserve its predecessor as a complete bundle with matching source/build information. A rollback must not overwrite experiments, original datasets, or model weights.

## 1. Shared visual system

- [x] Define typography, spacing, surfaces, borders and semantic colors centrally in `gui/theme.py`.
- [x] Apply the same rules to shell, recipe cards, shared forms, dialogs, tables, plot labels and report headings. Keep the recipe layout and analytical widgets.
- [x] Remove decorative emphasis and redundant Ready pills while preserving meaningful availability/error/loading/stale states.
- [x] Make alignment and numeric formatting consistent. Retain necessary precision; show units only when actually known. Display aliases change known legacy metric captions only, leaving saved data and arbitrary names unchanged.

**Acceptance:** compare home and representative recipe/result views at normal and minimum window sizes. Text and focus are readable, emphasis is consistent, and settings/result data remain accessible. Review this small visual checkpoint before changing navigation across the application.

**Stage 1 checkpoint — 2026-10-25:** the complete visual-system preview is preserved at `dist/ui-stage1/MLcookbook/MLcookbook.exe` after Adam requested stage 2. Native Windows inspection covered 1280×860 and 1080×720, with 100%/125%/150% scaling checks for focus, dense controls, metrics and painter text. **479 pytest tests passed**, Ruff and dependency checks passed; 60 existing sklearn SVC deprecation warnings remained. All 46 checked frozen modules matched source, and the preview launched/closed normally without Python on PATH. Its frozen runtime generated real offline CPU text and reopened a source-free saved Movement Study without training or unsealing its test. Recovery preimages, hashes and the UI-specific diff live in `ui-stage1/rollback/` beside the checkpoint. No shortcut promotion or commit was made.

## 2. Persistent sidebar and Data workspace

- [x] Add a persistent sidebar to `MainWindow`, with Data first and the recipe catalog, Local LLMs and Movement Study directly accessible. Keep Save/Open experiment available.
- [x] Start with one clearly identified active ordinary dataset. Expose import, preview, column roles, quality checks and source/snapshot information. Multi-dataset project management is separate scope.
- [x] Initially show dataset/source roles read-only in Data; keep editable role overrides scoped to the existing recipe request/session. Role edits mark that recipe's displayed result stale and persist through its existing `.mlcb` state. Global role editing is a separate decision, with explicit invalidation/persistence rules before implementation.
- [x] Support CSV/Parquet import from the shell before recipe selection. Recipes use that active ordinary dataset and show its identity. Imported/restored visits skip preset dataset construction.
- [x] Distinguish the active ordinary dataset from the current recipe's effective input. Explicit examples, Movement Study and dataset-free LLM prompts show their own source context; the sidebar identifies the ordinary file independently.
- [x] Preserve background-job navigation/import locks, failed-import recovery, successful replacement invalidation and per-recipe session state. Data import options persist separately; independent study reopening preserves previous workspace/session status.
- [x] Keep Movement Study import and fixed holdouts separate. A study snapshot never automatically becomes the ordinary imported dataset; opening it retains any existing ordinary dataset.

**Acceptance:** from a fresh launch, import a personal CSV, inspect its columns, open a recipe and retain that dataset. Navigation preserves results and unrun edits without retraining. Role overrides survive Save/Open and cannot leave an old report marked current. Failed imports preserve work; successful replacement invalidates prior results correctly. Source labels distinguish ordinary files, examples, study snapshots and typed prompts. Study handoffs and sealed-test defaults remain intact.

**Source verification — 2026-10-25:** **500 pytest tests passed**, including 21 new Data/sidebar cases; Ruff and dependency checks passed. The 60 existing sklearn SVC deprecation warnings remain. Native inspection covered 63 states across 100%/125%/150% scaling with no clipping flags, Qt/Python warnings, or forced expansion at 1080×720. This includes exact unrun role/report restoration, source-free saved experiments, failed imports, minimum-size metrics/local-model controls and independent input identities. The verified 115-file source checkpoint before this stage, per-file preimages/hashes and a stage-only diff are in `ui-stage2/`; 55 core/Movement/plot source files remain byte-identical. The original shortcut bundle and complete stage 1 preview are retained.

**Windows delivery:** `dist/ui-preview/MLcookbook/MLcookbook.exe` is rebuilt and launch/close checked, with all 47 checked bundled modules matching final source. A temporary helper using the exact candidate PYZ/runtime passed Data import/navigation, leading-zero values, offline Save/Open, failure recovery and typed-prompt identity without preset/training calls or optional model imports. It also restored a stage 1 saved Movement Study exactly with zero evaluations/test sealed and generated real offline CPU text. The helper exited normally and was removed; all 6,216 original shortcut-bundle files remain unchanged. No release, shortcut promotion, commit or push was performed. The first QA helper had a shutdown-lifetime issue; one persistent QApplication fixed the helper, with no product-source change.

## 3. Recipe setup and explicit examples

- [x] Retain algorithm-based cards and browsing, concise method descriptions and source context without a beginner task picker.
- [x] Show an honest setup state with import and explicit example actions. Navigation constructs no preset and performs no analysis; empty classical setup cannot Run/Save.
- [x] Load/run examples only after **Run example**, identify their source, and provide **Use my data** for an existing active dataset.
- [x] Preserve the ordinary dataset, own/example selections, results and unrun edits separately. Failures leave the displayed result/context intact. Saved examples reopen independently without becoming ordinary data.
- [x] Keep technical settings and defaults, schema-role/core validation and visible stale results; no silent coercion or automatic best-method selection.
- [x] Preserve dataset-free typed LLM generation, with empty typed storage and explicit dataset requirements for rows/structured tasks, plus dedicated Movement Study setup/holdout isolation.

**Acceptance:** navigating to a recipe does not train a model. Own-data and explicitly requested example runs work with correct source labels and normal cancellation/recovery. Returning restores the last report and unrun edits. LLM prompts still work without a dataset.

**Source verification — 2026-10-25:** 528 pytest cases passed, Ruff and dependency checks passed; the 60 existing SVC deprecation warnings remain. Native Windows inspection covered 96 states at 100%/125%/150%, including minimum-size setup/loading/actions, own/example transitions, failures and offline saved examples. Invalid cached targets/roles can be viewed without training, repaired and rerun while ordinary work stays intact. No clipping flags, Qt/Python warnings or forced window expansion were observed. The 118-file stage 2 source checkpoint is in `ui-stage3/baseline/`; its complete build is preserved at `dist/ui-stage2/MLcookbook` alongside stage 1 and the original shortcut bundle. The PR groups the pre-existing Movement prerequisite and each UI stage into separate commits, as Adam requested.

**Windows delivery:** the rebuilt preview's 47 checked modules match final source; native startup and normal close passed. Its exact frozen runtime passed first example execution, cached-invalid repair without unrequested analysis, explicit rerun, own-data restoration, independent saved examples, source-free sealed-study restoration with zero evaluations and real offline CPU generation. The helper exited normally and was removed. All 6,216 original production files remain unchanged. Evidence: `ui-stage3/packaging-validation.json`; candidate SHA-256 `610f487e338f1e056af6584f5cc0d1837b48bf014330912e4f27300366055a44`.

## 4. Forms, reports and technical reference

- [ ] Organize forms consistently around data/columns, model settings, preprocessing, validation and reproducibility.
- [ ] Consolidate repeated status text while keeping fixed Run/Cancel/progress controls, stale-result indication and an accessible warning summary.
- [ ] Retain a short result summary and principal analytical visual. Make metrics, tables, preprocessing and provenance easy to scan/expand. Preserve train/test/CV distinctions and every table row.
- [ ] Replace API-first About content with concise reference: applicability, assumptions, parameters, metric/plot interpretation and limitations. Put callable signatures in a secondary developer section; avoid permanently displayed lessons.
- [ ] Carry these rules through Local LLMs and Movement Study. Preserve prompts/identities, requested versus applied settings, row outcomes, reference coverage, split/group lineage and explicit final-test evaluation.

**Acceptance:** inspect regression, binary/multiclass classification, unsupervised analyses, trees, long tables, LLM output and Movement Study. A graduate user can identify data, configuration, evaluation design and result limitations without decoding scattered labels or opening a function signature.

## 5. Startup and first-use responsiveness

The 2026-10-24 audit used fresh processes without flushing OS caches. Time to the first titled window was 9.25 s then 2.47–2.49 s for the source entry point, and 2.20 s then 0.87–0.88 s for the packaged app. Separate first-recipe setup took 2.56 s without training. One package-import trace spent 7.73 s in eager scientific imports; Torch was not imported. These are observations, not cold-boot/first-paint guarantees or CI thresholds.

- [ ] Profile source entry, packaged launch, first-recipe initialization and execution separately. Record source/bundle revisions and distinguish window creation from usable first paint.
- [ ] Reduce unnecessary eager imports and metadata/demo work before a screen can paint. Preserve the root public API and single `python -m mlcookbook` entry point.
- [ ] Put core work/file I/O in background jobs where appropriate; construct/update widgets on the GUI thread. Preserve progress, cancellation, recovery and clean shutdown.
- [ ] Keep optional model imports, weight loading, downloads and automatic inference out of ordinary startup/navigation.
- [ ] Use import/behavior regression tests and repeatable local timing reports. Avoid fragile wall-clock CI assertions; establish budgets after measuring the revised path.

**Acceptance:** the shell/setup remain responsive during initialization, navigation performs no unrequested analysis, and source/packaged timings are compared under the same conditions with remaining cost stated honestly.

## 6. Compatibility and Windows delivery

- [ ] Run full pytest and Ruff using `.venv/Scripts/python.exe`; distinguish baseline issues from new regressions.
- [ ] Verify Save/Open for every recipe, old `.mlcb` files, unrun controls, stale reports, roles, scroll/expanded state and offline results. Keep internal parameter/section keys stable or add explicit migrations; visible captions can change independently.
- [ ] Check import success/failure/replacement, navigation locks, worker errors, cancellation and closing during work.
- [ ] Check Movement Study isolation, group-aware evaluation, sealed defaults, and `--study`, `--movement-study` and `--experiment` handoffs. Opening/restoring must not train automatically.
- [ ] Inspect native Windows layouts at 1280×860 and the supported minimum, with process-local scaling checks for common Windows scaling settings. Check keyboard/focus, contrast, dialogs, long names, tables and numeric precision.
- [ ] Rebuild a separate candidate, verify bundled application modules against reviewed source, and launch-check it. Check real local inference when affected. Preserve the previous complete bundle before promoting a candidate to the shortcut's path.
- [ ] Update README, TUTORIAL, ADAM and relevant AGENTS conventions for implemented behavior. Keep roadmap checkboxes factual. Commit/push only when asked.

**Acceptance:** Adam can work with his own data, configure an analysis, inspect/reopen results, and return to a prior UI build without losing data, experiments or unrelated source changes.

## First implementation slice and scope

Stages 0–3 are implemented at Adam's request, with separate source checkpoints and Windows previews. Stage 4's forms/reference organization is the next reviewable slice. Recipe navigation is now setup-only; examples and analyses run explicitly.

This pass adds no new inference engine, graph execution/editor, REST service, mandatory lessons, dependency, alternate entry point, model installation/download, or multi-dataset project format. Preserve statistical/leakage boundaries, typed results, saved experiments and the existing presentation/core separation. Broader additions require a separate explicit request.
