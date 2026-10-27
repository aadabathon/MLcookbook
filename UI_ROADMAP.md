# UI roadmap

Requested by Adam on 2026-10-24. UI and dataset workflow improvements take priority over REST endpoints, the graph workshop, and further feature expansion.

## Product direction

ML Cookbook is an application for general graduate-level statistics, data science, and ML workflows. Make data, configuration, execution, and results straightforward to locate and operate. Adam's learning environment as an author does not imply a tutorial-first product.

Keep native PySide6 / Qt Widgets, the recipe catalog/cards, and the analytical widgets. The previous interface concepts are layout references; Adam's subsequent corrections are the product brief. Use a persistent sidebar with **Data at the top**, coherent typography/color/emphasis, accessible technical controls, and concise reference material. A mandatory wizard or permanent lesson panel is outside this brief.

### Design rules

- Treat the dataset as a first-class workspace object. Make import, preview, column roles, quality checks, and source identity directly accessible.
- Use one consistent sans-serif type system, limited sizes/weights, sentence-case labels, and predictable alignment. Retain Greek/math recipe marks and violet accents as product identity, per Adam's 2026-10-26 clarification. Remove gratuitous eyebrow labels, letter spacing, redundant badges, and arbitrary bolding.
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

- [x] Organize forms consistently around data/columns, model settings, preprocessing, validation and reproducibility.
- [x] Consolidate repeated status text while keeping fixed Run/Cancel/progress controls, stale-result indication and an accessible warning summary.
- [x] Retain a short result summary and principal analytical visual. Make metrics, tables, preprocessing and provenance easy to scan/expand. Preserve train/test/CV distinctions and every table row.
- [x] Replace API-first About content with concise reference: applicability, assumptions, parameters, metric/plot interpretation and limitations. Put callable signatures in a secondary developer section; avoid permanently displayed lessons.
- [x] Carry these rules through Local LLMs and Movement Study. Preserve prompts/identities, requested versus applied settings, row outcomes, reference coverage, split/group lineage and explicit final-test evaluation.

**Acceptance:** inspect regression, binary/multiclass classification, unsupervised analyses, trees, long tables, LLM output and Movement Study. A graduate user can identify data, configuration, evaluation design and result limitations without decoding scattered labels or opening a function signature.

**Source verification — 2026-10-26:** 566 pytest cases passed, Ruff/dependency checks passed; the 60 existing SVC deprecation warnings remain. Native Windows inspection covered 93 states at 100%/125%/150%, with no clipping flags, Qt/Python warnings or forced expansion at 1080×720. Preprocessing/evaluation stays outside collapsed model settings; comparison split/seed controls and fixed defaults are explicit. Unsupervised fitting is labelled as full-data fitting. Run details expansion is temporary and adds no saved-state keys. References match actual screen controls. Existing `.mlcb` states, precise report values, all 500 rows in a long-table case, typed-prompt setup and sealed Movement groups remain covered. New disclosure callbacks use QObject-bound slots after a Windows validation run exposed a native failure during worker garbage collection; focused lifecycle checks and the subsequent full suite passed without a product GC workaround. The 119-file baseline and recovery evidence are under `ui-stage4/`; the accepted complete stage 3 bundle is preserved at `dist/ui-stage3/MLcookbook`.

**Windows delivery:** all 48 checked bundled modules match final source. Native startup/responsiveness/normal close and the exact frozen runtime checks passed, including the new controls/reference, own/example/offline snapshots, a legacy source-free Movement Study with zero evaluations/final test sealed, and 24 tokens from a real offline CPU model. The temporary helper exited normally and was removed. All 6,216 original production files and the complete stage 3 inventory remain unchanged. All 52 protected core/plot/tree files match the baseline. Evidence is in `ui-stage4/packaging-validation.json`; preview SHA-256 `4d0d1200b1e8ed99fc818396cdb19905d99ba22abe269af08b1b22c101172686`. The shortcut and release remain unchanged.

## 5. Startup and first-use responsiveness

The 2026-10-24 audit used fresh processes without flushing OS caches. Time to the first titled window was 9.25 s then 2.47–2.49 s for the source entry point, and 2.20 s then 0.87–0.88 s for the packaged app. Separate first-recipe setup took 2.56 s without training. One package-import trace spent 7.73 s in eager scientific imports; Torch was not imported. These are observations, not cold-boot/first-paint guarantees or CI thresholds.

- [x] Profile source entry, packaged launch, first-recipe initialization and execution separately. Record source/bundle revisions and distinguish window creation from usable first paint.
- [x] Reduce unnecessary eager imports and metadata/demo work before a screen can paint. Preserve the root public API and single `python -m mlcookbook` entry point.
- [x] Put core work/file I/O in background jobs where appropriate; construct/update widgets on the GUI thread. Preserve progress, cancellation, recovery and clean shutdown.
- [x] Keep optional model imports, weight loading, downloads and automatic inference out of ordinary startup/navigation.
- [x] Use import/behavior regression tests and repeatable local timing reports. Avoid fragile wall-clock CI assertions; establish budgets after measuring the revised path.

**Acceptance:** the shell/setup remain responsive during initialization, navigation performs no unrequested analysis, and source/packaged timings are compared under the same conditions with remaining cost stated honestly.

**Source checkpoint — 2026-10-26:** 580 tests, Ruff, dependency checks and the library demo passed. All 44 public exports retain their original identities and signatures; an ordinary package import and Data/catalog paint load no scientific stack or optional model runtime. Cold analysis initialization and first file/snapshot reads use workers; GUI construction stays on the GUI thread. Actual-entry subprocess tests cover first paint, responsiveness, failed initialization/retry, ordered study handoffs and shutdown during initialization. Three fresh-process local samples (no OS-cache flush) measured source Data usability at a median 2.654→0.913 s. First recipe usability is 0.067→2.036 s because imports moved to first use; its deferred-work GUI heartbeat gap was about 0.062 s. These are local observations, not cold-boot guarantees or CI thresholds. Frozen measurements use a separately identified instrumented bundle runtime. The stage 4 bundle and 122-file source baseline are preserved in `dist/ui-stage4/MLcookbook` and `ui-stage5/baseline/`.

**Frozen observations:** median Data usability was 0.987→0.933 s; first recipe usability was 1.329→1.191 s. Candidate first-recipe samples were 8.382/1.191/1.179 s, so first-use initialization can still take several seconds. The median maximum GUI heartbeat gap improved from 1.314 s to 0.065 s (all candidate samples below 0.1 s). No packaged launch-speed improvement is claimed. A separate explicit warm linear example took 0.397 s overall: 0.175 s loading and 0.080 s analysis/formatting. Reports identify fresh-process/no-cache-flush conditions and the exact archived source/bundle inputs: `ui-stage5/startup-baseline-and-source.json`, `startup-candidate-source.json` and `startup-candidate-frozen.json`.

## 6. Compatibility and Windows delivery

- [x] Run full pytest and Ruff using `.venv/Scripts/python.exe`; distinguish baseline issues from new regressions.
- [x] Verify Save/Open for every recipe, old `.mlcb` files, unrun controls, stale reports, roles, scroll/expanded state and offline results. Keep internal parameter/section keys stable or add explicit migrations; visible captions can change independently.
- [x] Check import success/failure/replacement, navigation locks, worker errors, cancellation and closing during work.
- [x] Check Movement Study isolation, group-aware evaluation, sealed defaults, and `--study`, `--movement-study` and `--experiment` handoffs. Opening/restoring must not train automatically.
- [x] Inspect native Windows layouts at 1280×860 and the supported minimum, with process-local scaling checks for common Windows scaling settings. Check keyboard/focus, contrast, dialogs, long names, tables and numeric precision.
- [x] Rebuild a separate candidate, verify bundled application modules against reviewed source, and launch-check it. Check real local inference when affected. Preserve the previous complete bundle before promoting a candidate to the shortcut's path.
- [x] Update README, TUTORIAL, ADAM and relevant AGENTS conventions for implemented behavior. Keep roadmap checkboxes factual. Commit/push only when asked.

**Acceptance:** Adam can work with his own data, configure an analysis, inspect/reopen results, and return to a prior UI build without losing data, experiments or unrelated source changes.

**Delivery checkpoint — 2026-10-26:** 580 tests and the latest focused shutdown checks passed, with 60 existing SVC warnings. Stage 5 added 18 native startup states across three scale factors; all 27 GUI files remain byte-identical to the stage 4 visual review, and 51 protected analytical/plot files are unchanged (only the root import mechanism changed). The final executable's 48 checked modules match source and it passed three normal nondebug launch/close runs. Frozen checks resolved all 44 exports and verified saved/offline workflows, sealed study restoration without evaluation and CPU inference. Actual-main checks cover closing while imports are active. Explicit teardown fixes a native shutdown fault found in an earlier candidate; no GC disabling or forced-success exit was added. The hash-verified complete bundle is installed at the shortcut target `dist/MLcookbook`; the original complete app is preserved at `dist/ui-original/MLcookbook`. Evidence and rollback identities are under `ui-stage5/`.

## First implementation slice and scope

Stages 0–6 are implemented at Adam's request, with separate source checkpoints, preserved Windows bundles and a verified installation at the existing shortcut target. Recipe navigation is setup-only; examples and analyses run explicitly. Further roadmap features require a new request.

**Visual identity clarification — 2026-10-26:** Adam requested the simple UI retain its purple accents and Greek/math recipe marks. The verified build restores those marks beside card titles and violet selected navigation, actions and focus outlines. Body text, controls, analytical palette and workflow structure retain their current form; startup/shutdown code is untouched. 583 tests and native/frozen style checks passed, including 33 Windows views across common scaling settings. Adam then requested completion in this session: the build is installed at the existing shortcut target, with the complete previous neutral app preserved at `dist/ui-before-reflavor/MLcookbook` and its matching earlier build at `dist/ui-stage5/MLcookbook`. Source/bundle recovery and verification records are in `ui-reflavor/`.

This pass adds no new inference engine, graph execution/editor, REST service, mandatory lessons, dependency, alternate entry point, model installation/download, or multi-dataset project format. Preserve statistical/leakage boundaries, typed results, saved experiments and the existing presentation/core separation. Broader additions require a separate explicit request.

**Original character restoration — 2026-10-26:** at Adam's request, restore ink-black/violet surfaces and selective serif headings while retaining Segoe controls/plots, the Data-first shell and latency improvements. Remove generic subtitles, repeated input badges, permanent column-help paragraphs and duplicate empty-output instructions. Keep one effective input row with its Open action and dimensions, contextual help/reference, stale-result indication and warnings; valid edits leave a quiet footer. The eight pending accent-pass files and full source baseline are archived in `ui-restore/baseline/`; Sandbox work remains a separate discussion.

**Restoration source validation:** 592 tests, Ruff and dependency checks passed. Native inspection covered 57 views across 100%/125%/150% scaling without recorded text/control clipping or Qt warnings. Three minimum-window LLM regression cases cover adaptive form wrapping. Core algorithms, chart/tree implementations, build script and startup/shutdown function bodies retain their baseline code. Reports, warning access, strict experiment state and own/example/prompt/study boundaries remain covered. Evidence is under `ui-restore/`.

**Restoration delivery:** installed at the normal shortcut target after matching 48 bundled modules to source, three normal preview closes, frozen own-data/example/offline-reopen checks and real local CPU generation. Complete previous app preserved at `dist/ui-before-restore/MLcookbook`; full 6,216-file inventories verified both folder moves. A post-install launch/close in the inherited environment returned exit 0. Source and installation records are in `ui-restore/`. No commit or Sandbox implementation was made.
