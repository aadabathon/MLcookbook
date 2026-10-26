# Adam's Handbook

Personal notes for the author. The scaffold is done — from here on, this project is yours. This file is your map: where things are, how to work, and what to build in what order. Edit it freely; it's a living document.

For hands-on use of every recipe, start with the [recipe tutorial](TUTORIAL.md), including its practice dataset and local LLM research walkthrough.

## UI startup and delivery — 2026-10-26

The Data shell and recipe catalog now paint without importing the scientific stack. All 44 root Python API names still resolve to the original functions/classes; imports are deferred through an explicit export map. The first cold analysis prepares that existing runtime in a worker before its screen is built on the GUI thread. The current page, status and navigation locks remain visible; failures retain the page for retry. First CSV/experiment imports also prepare/read in a worker. Startup study handoffs wait for setup before loading their manifest, preserving no-training and sealed-test defaults.

In three fresh-process local samples without flushing OS caches, median source Data usability changed from 2.654 s to 0.913 s. First recipe usability changed from 0.067 s to 2.036 s because initialization moved out of launch; the maximum GUI heartbeat gap during that deferred work was about 0.062 s. This makes the shell available sooner and keeps first use responsive; it does not remove scientific initialization. Source and frozen executable measurements are recorded separately in `ui-stage5/`.

Source validation: **580 tests passed**, Ruff/dependency checks and `examples/basic_usage.py` passed; 60 existing sklearn SVC warnings remain. Tests exercise the actual module entry point and first paint without scientific imports, real first-use worker imports, timer response, failure/retry, ordered study handoffs, the unchanged public API, wildcard imports, object identity/pickling and ordinary serialized classification. Shutdown tests verify that active initialization finishes and all top-level widgets, including hidden candidate screens, are retired on the GUI thread while QApplication remains alive. `build_windows.ps1` explicitly collects core submodules for the lazy export map. The complete stage 4 preview is preserved at `dist/ui-stage4/MLcookbook`; the verified 122-file source baseline is under `ui-stage5/baseline/`.

Instrumented frozen-runtime medians were effectively unchanged for Data usability (0.987→0.933 s) and first recipe usability (1.329→1.191 s). The first candidate sample took 8.382 s to initialize the recipe; the next two took 1.191/1.179 s. Initialization remains real work. The median maximum GUI heartbeat gap during it improved from 1.314 s to 0.065 s, with candidate gaps below 0.1 s in all three samples. An explicit warm-runtime linear example took 0.397 s end to end, with 0.175 s loading data and 0.080 s in analysis/report formatting; this is a separate single observation. These measurements used fresh processes without cache flushing, not a controlled cold boot. Detailed JSON/logs and the profiling harness remain in `ui-stage5/`.

The verified final bundle is installed at `dist/MLcookbook/MLcookbook.exe`, the existing shortcut's target. Its SHA-256 is `c4b157871960a81fcfd3430765a4410de54b03c49040503d3323cfe50200f024`. A staged full copy was hash-verified before promotion; all 6,216 candidate files match, and the complete 6,216-file pre-redesign app is preserved unchanged at `dist/ui-original/MLcookbook`. The earlier stage previews and saved experiments are retained. For rollback, close the app and exchange complete bundles after checking their identities; never copy an EXE over a different `_internal` folder. Preserve the current bundle before a rollback.

Windows checks verified 48 bundled modules against source, all 44 public exports, offline snapshots, source-free sealed Movement Study restoration with zero evaluations, and real local CPU generation. The final executable passed three normal nondebug launch/close runs. An earlier candidate's native shutdown fault led to explicit GUI-thread teardown while QApplication remains alive; the failing attempt is archived under `ui-stage5/attempt-before-shutdown-fix`. Final evidence is in `ui-stage5/packaging-validation.json`, `native-exit-validation.json` and `production-promotion.json`. No runtime/model service or dependency was added.

## UI roadmap stage 4 — 2026-10-26

Recipe forms now separate data/columns, model controls and preprocessing/evaluation. Classical preprocessing and the split/seed/training-CV summary remain visible when model settings are closed; Model Comparison's editable test fraction/seed move beside those evaluation choices. Full-data unsupervised fitting is labelled accurately, including the preprocessing dialog. The banner owns stale-result provenance, while the fixed footer reports readiness, job progress or errors. Action-specific refusals and restored legacy status text remain intact.

**Run details** is a temporary disclosure of report data/settings and any recorded execution timestamp/runtime versions. It changes no saved UI keys or raw result values. **Recipe reference** covers applicability, assumptions, controls, interpretation and limitations for all twelve ready recipes, with a collapsed Python API section. Its content follows actual screen controls: PCA computes all available components, classification in the GUI supports up to 20 classes, and forest feature sampling differs by task. LLM settings are grouped and empty output no longer contains a permanent tutorial; grouped Movement controls open with the final test sealed.

The accepted stage 3 build is preserved as the complete `dist/ui-stage3/MLcookbook` bundle. Stage 4 uses a separate `dist/ui-preview/MLcookbook` candidate, retaining the shortcut bundle and earlier previews. The verified 119-file baseline, per-file hashes and recovery evidence live under `ui-stage4/` beside earlier UI checkpoints. Startup optimization is the next stage and has not been implemented in this pass.

Source validation: **566 tests passed**, Ruff and dependency checks passed; 60 existing sklearn SVC warnings remain. Native Windows checks covered 93 states across 100%/125%/150%, with no clipping flags or forced growth at 1080×720. New tests exercise modal accept/cancel, unsupervised semantics, split/seed/CV context, exact legacy state/report retention, full long tables, empty LLM setup, reference accuracy and widget cleanup. A native crash during an earlier worker-GC run prompted removal of self-retaining disclosure callbacks; QObject-bound handlers, explicit GUI deletion and repeated lifecycle checks passed before the final full run. No production GC disabling was added.

Windows delivery: all 48 checked bundled modules match source, native startup and normal close passed, and the exact frozen runtime passed the new UI/reference checks plus own/example/offline snapshots. A legacy source-free Movement Study restored exact state with zero evaluations and the final test sealed; a real offline CPU model generated 24 tokens. The helper exited normally and was removed. Original production and complete stage 3 bundles remain unchanged, as do all 52 protected core/plot/tree files. Preview SHA-256: `4d0d1200b1e8ed99fc818396cdb19905d99ba22abe269af08b1b22c101172686`. Evidence: `ui-stage4/packaging-validation.json`.

## UI roadmap stage 3 — 2026-10-25

Opening a recipe now waits for an explicit action. **Run example** first loads a bundled dataset in a background job, prepares GUI controls, then runs the analysis in a second job while navigation stays locked. Only success switches the visible input/report. **View example** restores a cached example without training, including invalid unrun settings you need to repair; **Run Analysis** reruns it. **Use my data** restores the ordinary-data result and unrun edits. Own/example settings and last views remain separate in memory; Save/Open still stores one recipe and its effective dataset, with the same strict state keys and `.mlcb` format. Saved examples reopen offline without becoming ordinary input or removing an independent study session.

Empty classical setup cannot Run/Save. Typed LLM prompts use zero-row string storage and still generate/save/reopen without a tabular import; row and structured workflows require data. No model is loaded on navigation. Core algorithms and result schemas are unchanged. The previous sidebar build is preserved in `dist/ui-stage2/MLcookbook`, and its 118-file source checkpoint is in `ui-stage3/baseline/`. The current candidate continues to use `dist/ui-preview/MLcookbook`; the original shortcut is unchanged.

Source validation: **528 tests passed**, Ruff/dependency checks passed; 60 existing sklearn SVC deprecation warnings remain. Native Windows inspection covered 96 states at 100%/125%/150%, including loading progress, exact own/example edits/results, cached-invalid-setting repair, failures, source-free saved examples, minimum-size actions/metrics, empty typed prompts and study isolation. The requested PR uses separate commits for the existing Movement prerequisite, visual system, Data sidebar and explicit examples.

Windows validation: all 47 checked candidate modules match final source, and native startup/normal close passed. The same frozen PYZ/runtime passed first example loading/analysis, cached-invalid example access without training, repair/rerun and exact own-data return; saved examples reopen independently. It also restored a legacy sealed Movement Study with zero evaluations and generated real offline CPU SmolLM2 text. The temporary helper exited normally and was removed. All 6,216 original production files remain unchanged. The final candidate SHA-256 is `610f487e338f1e056af6584f5cc0d1837b48bf014330912e4f27300366055a44`; reports are under `ui-stage3/`. No model downloads, managed services, dependency changes or shortcut promotion were added.

## UI roadmap stage 2 — 2026-10-25

The app now starts on **Data**, with persistent sidebar routes for **Recipes**, **Local LLMs** and **Movement Study**. Import CSV/Parquet before choosing an analysis, inspect the first 20 rows, read-only source roles/missing counts and data checks, then open a recipe. The sidebar identifies the active ordinary dataset, while an **Input:** line identifies the actual recipe source, including typed prompts, examples and independent movement holdouts. Imported/restored recipes skip preset construction. Navigation keeps results, stale state, overrides and unrun controls; file replacement invalidates old sessions and failed imports/opens preserve work. Data's unrun CSV controls stay separate from the applied import provenance and each recipe's saved options.

Movement Study imports and saved study opens keep fixed holdouts separate and preserve an existing ordinary dataset and its recipe sessions. Explicit accepted LLM fields become the active dataset while the LLM source run stays available. Opening a regular experiment activates its embedded snapshot without rereading the original file. Data alone is not a new project/save format. Examples remain automatic when no ordinary file is active; opt-in examples are stage 3.

The candidate remains `dist/ui-preview/MLcookbook/MLcookbook.exe` (`./build_windows.ps1 -Preview`). The complete prior visual preview is preserved at `dist/ui-stage1/MLcookbook`; the original shortcut bundle remains at `dist/MLcookbook`. Recovery source before this stage is the verified 115-file `ui-stage2/baseline/source.zip` alongside stage 1's QA directory. Revert only stage-specific hunks against that dirty-source checkpoint so the previous UI and Movement Study work survive. No commit or shortcut promotion is implied by this preview.

Source validation: **500 tests passed**, Ruff and dependency checks passed; the 60 existing SVC deprecation warnings remain. New coverage includes five Data presentation cases and sixteen coordinator/sidebar cases. Native Windows inspection covered 63 views at 100%/125%/150%, including minimum-size fields/metrics, distinct LLM/study inputs, exact role/stale-report restoration, failed imports and original-file-absent saved reopening. No clipping flags, Qt/Python warnings or forced window expansion were observed. All 55 core/Movement/plot source files checked against the prior dirty baseline remain unchanged.

Windows validation: the rebuilt candidate's 47 checked modules match source, and it opens/closes normally without Python on PATH. A temporary QA executable using the same PYZ and `_internal` passed Data import/leading-zero IDs, no-preset navigation, source-free Save/Open, failed-import recovery and typed-prompt identity before any optional model import. It restored a legacy stage 1 Movement Study with zero evaluation calls and sealed test groups, then generated real offline CPU SmolLM2 text. The helper exited normally and was removed; all 6,216 original shortcut-bundle files remain unchanged. Evidence is in `ui-stage2/packaging-validation.json` and related logs. An initial QA-only Qt shutdown issue was corrected by retaining one QApplication; no product changes were needed. No dependency, network service, model download, release, shortcut promotion or Git mutation was added.

## UI roadmap stage 1 — 2026-10-25

The first requested slice keeps the recipe catalog and analytical widgets while applying one sans-serif type system, neutral surfaces/borders, restrained accent colors, sentence-case section headings, and left-aligned cards without decorative symbols or repeated Ready labels. Plot/tree fonts share the theme, axis margins account for tick widths, and tree connections use a readable neutral line. Metric caption aliases affect display only; exact values, arbitrary names and saved report data retain their original strings. Warnings, stale results, keyboard actions and experiment controls remain visible.

The complete stage 1 candidate is now preserved at `dist/ui-stage1/MLcookbook/MLcookbook.exe`, together with its `_internal` folder. Current candidates build with `./build_windows.ps1 -Preview`. The existing shortcut still points to `dist/MLcookbook/MLcookbook.exe`; its complete bundle is preserved. Neither the shortcut nor a release was updated. At this checkpoint, Data-first navigation, explicit examples, reference content and startup work were scheduled as later [UI roadmap](UI_ROADMAP.md) stages.

Source recovery evidence is outside Git at `C:/Users/adams/.codex/visualizations/2026/10/25/01a17148-2348-7b22-af8a-4a0517a8d864/`: the verified `ui-baseline-2026-10-24/source.zip` includes the pre-existing dirty Movement Study work, while `ui-stage1/rollback/` records only changes since that checkpoint, their preimages and hashes. Revert individual UI hunks after checking subsequent edits; do not reset to HEAD, which predates the uncommitted Movement integration. Native captures and packaging checks live in `ui-stage1/`.

Verification: **479 tests passed**, Ruff, dependency checks and `git diff --check` passed. The same 60 sklearn SVC deprecation warnings remain. Native Windows inspection covered home, imported setup/results, regression/classification, complete trees, long tables, Local LLMs, Movement Study and dialogs, including minimum size, keyboard focus and metric/painter text at 100%/125%/150% scaling. The candidate's 46 checked modules match final source; it launched responsively and closed normally with Python removed from PATH. A QA helper using its exact frozen PYZ/runtime generated real offline CPU SmolLM2 text and restored a synthetic saved Movement Study with source files absent, unrun edits preserved, zero evaluation calls and test groups sealed. The exact temporary helper was removed. All 6,216 original shortcut-bundle files remain byte-for-byte unchanged. No dependencies, models, services, shortcut promotion, release, commit or push were added.

## Structured text workflows - 2026-10-19

The third roadmap milestone is implemented at my request. Open **Local LLMs**, choose a structured workflow and server engine, import text rows, check only input columns, and **Use task template with selected columns**. Templates bind `{{column name}}` literally. Fields use `new_name: text/numeric/boolean/categorical`, with optional `: label | label` for categorical values. The tutorial now walks through extracting fields from the practice housing notes.

The core submodule `llm_workflow.py` independently validates JSON and records raw answers, per-row outcomes, exact prompts, source/configuration/model identity and backend metadata. No dependencies or root API exports were added. Structured requests use Ollama `format` or llama.cpp `response_format`; validation never repairs JSON or converts quoted numbers. Unknown facts may be null. Optional `output_field=reference_column` bindings report exact agreement on valid, non-null pairs and stay outside the prompt inputs. Blank rows count toward the 1–10 source-position batch limit.

Review source text and generated values before **Use as derived dataset**. Original values/dtypes and duplicate index labels survive; generated nullable arrays join by source position plus typed snapshot fingerprint. Other recipes receive the accepted snapshot after Back; Local LLMs retains its source run. Save the LLM experiment for templates/source/history, or a downstream analysis experiment for the derived snapshot and lineage. Retry selected/failed rows creates another artifact only when source/config/model checks pass. Cancellation retains completed rows and explicitly marks unfinished ones. Compare selected compatible runs using validation, output length, wall/token metrics and reported server timings; inherited retry timings are original observations. Up to ten recent runs persist offline, and explicit JSON export includes outcomes/source values/provenance.

Research context: Silicon Badgers is my club. The existing `~/SiliconBadgers/AI_stuff/models/Qwen3.5-2B-Q4_K_M.gguf` loads successfully with the installed llama-server, unlike the previously tested Ollama 9B GGUF. Native QA used this existing 2B file on a temporary loopback CPU server: extracted item/price from two descriptions, checked exact prices, compared two prompts, exported, saved/reopened and created typed columns for another recipe. No club files were edited, dependencies installed or models downloaded. GGUF/tensor/model-architecture inspection remains separate from the next workflow graph milestone.

Verification: **440 tests passed**, Ruff and dependency checks passed. Tests cover strict JSON/roles, source fingerprints and duplicate indexes, reference exclusion/coverage, per-row errors/cancellation, identity/configuration checks on retries, changed server templates, limit-stopped outputs, exports, offline disk restore and legacy parameter states. The existing 60 sklearn SVC deprecation warnings remain.

The final Windows executable was rebuilt and launch-checked; all sixteen checked application modules match source. A temporary QA executable using the production PYZ/bootstrap and exact collected runtime verified real structured fields through both llama.cpp/Qwen 2B and Ollama/Qwen 9B, plus PyTorch/SmolLM2 generation with Python removed from PATH. The helper executable was removed. After stopping only the identity-verified QA-owned llama-server, the native app reopened the two real runs, compared them and derived typed columns offline at the minimum window size. The user's Ollama service and club files were left intact. Published releases and Git commits are unchanged.

The tested club-model setup can be started explicitly outside the app (CPU, local port 8080):

```powershell
llama-server -m "$env:USERPROFILE/SiliconBadgers/AI_stuff/models/Qwen3.5-2B-Q4_K_M.gguf" --alias qwen3.5-2b --host 127.0.0.1 --port 8080 --ctx-size 4096 --parallel 1 --n-gpu-layers 0 --threads 4 --jinja --offline --reasoning-budget 0
```

Then choose llama.cpp in Local LLMs and refresh. This does not inspect the model's internal computation graph. Context/device/thinking settings remain server controls; use a separate server profile when experimenting with them.

## Local workflow direction - 2026-10-19

Requested roadmap: [local LLM and workflow graphs](ROADMAP.md#local-llm-and-workflow-graphs). Proposed delivery order is Ollama integration, llama.cpp integration, repeatable structured text-row workflows, a UI-free graph executor with saved artifacts, then a Qt graph editor with per-node analysis/report inspection. Managed runner installation/startup follows as an optional later milestone. Each phase has a completion checkpoint and preserves the existing result, session, and training-only validation rules.

## llama.cpp integration - 2026-10-19

The second roadmap milestone is implemented at my request. **Local LLMs → llama.cpp server → Test / refresh models** connects directly to an existing single-model server. The app reads health, loaded-model identity and read-only properties, then streams user chat messages over SSE. Plain prompt/template, Ollama context/thinking and PyTorch device controls are hidden for this engine; context, offload and thinking belong to the external server. No new Python dependencies or product process management/download behavior was added.

`llamacpp.py` returns frozen JSON-safe connection/generation results. It retains source positions, submitted messages, requested and explicitly reported applied settings, reasoning/answer text, stop reasons, actual usage, millisecond server timings, wall time, model metadata and reported runtime version. Missing metrics/version remain unavailable. The shared private `_local_http.py` handles bounded loopback HTTP and idle cancellation/deadlines; SSE and Ollama NDJSON parsing stay separate. Partial, failed or cancelled runs preserve the last report/request. Save/Open needs no server/weights and migrates pre-Ollama and pre-llama.cpp states. Optional **GGUF file** hashes an explicitly selected local file; server-supplied paths are never opened automatically, and the hash does not prove binding to loaded weights, other shards or templates.

The installed llama-server reports `b11193-4e7481175`. Its loader rejected the Ollama Qwen3.5 9B GGUF's rope metadata shape; the separate club 2B file was verified successfully in the later milestone above. QA therefore downloaded `models/SmolLM2-135M-Instruct.Q4_K_M.gguf` from QuantFactory revision `476854d00ede130660aba430d15f9347ad2e7d0e` and verified SHA-256 `8030f04528538d47bda434f6f0bdf3952c40a58123e4d5e755332f23731a8684`. A temporary CPU server on port 18080 exercised discovery, real text/usage/timings, explicit file hashing, native UI, saved output and streaming cancellation. The checkpoint stays outside Git and the bundle. README has the equivalent user-started server command on the default port 8080. The small model verifies inference plumbing; it is not a quality recommendation.

Verification: **397 tests passed**, Ruff and dependency checks passed. Core contracts use a fake HTTP/SSE server and need no inference runtime/model download. Desktop tests cover loading, transient answer/reasoning previews, cancellation, rows, offline restoration and older Ollama/PyTorch states. Real cancellation leaves the test server ready. Structured text-to-dataset workflows are the next milestone; graphs and managed runtime setup remain later steps. The published v1.0.0 release is unchanged, and no commit/push has been performed.

The Windows executable was rebuilt and launched successfully; all fifteen checked application modules match the final source, with no foreign ICU DLL. A temporary QA executable reusing the production PYZ/bootstrap and exact `_internal` runtime generated through llama.cpp/SmolLM2 GGUF (including the verified file hash), Ollama/Qwen and PyTorch/SmolLM2, with Python removed from `PATH`. The server engines did not import torch/transformers. The temporary executable was removed. After stopping only the QA-owned llama-server process, the native app reopened the real saved experiment offline; a failed rerun preserved its text and report. The user's Ollama service was left running.

## Ollama integration - 2026-10-19

The first roadmap milestone is implemented at my request. Open **Local LLMs → Inference engine → Ollama → Test / refresh models**, choose an installed model, then generate. The existing local server at `http://127.0.0.1:11434` reports Ollama 0.40.6 and `qwen3.5:9b` (9.7B parameters, Q4_K_M, about 6.59 GB). A real native-app run streamed a 72-token answer in about 11 seconds including request overhead; cold loading can be considerably slower. The screen defaults to 90 seconds per row, including Ollama loading.

`src/mlcookbook/ollama.py` uses only standard-library HTTP and the existing input validators. It lists/inspects installed local models, sends raw prompts or chat messages, streams NDJSON with cancellation even while idle, and returns frozen JSON-safe data with model digest/metadata, submitted prompts/messages, requested options, server usage/timings, and local wall time. No models are pulled and the app does not own the Ollama process. The server manages device, precision, and context truncation; these applied details and exact rendered chat prompts are not verified by the connector. Thinking controls support boolean On/Off or Model default.

The live preview is transient. Errors, cancellation, deadlines, and incomplete streams retain the previous completed report/request. Save/Open keeps outputs and unrun edits without contacting Ollama. Old PyTorch experiment files migrate before validation, and the existing PyTorch API remains unchanged. No dependencies were added. The two requested cards remain at the bottom. llama.cpp is now implemented above; structured workflows and graphs remain later milestones.

Verification: **358 tests passed** in `.venv`; Ruff and dependency checks passed. Contract tests use a real fake loopback HTTP server, including fragmented UTF-8, chunked transfer, model/cloud checks, missing metrics, idle cancellation/deadlines, and import isolation from torch/transformers. Desktop tests cover row generation, transient previews, failure recovery, offline reopen, and legacy experiments. A regression test also preserves accurate runtime-version checks when the engine changes without rerunning the previous report. Native Windows inspection covered engine setup, connection metadata, real Qwen streaming/results, minimum window size, dataset mode, and reopened output. The Linux CI library correction and all current changes still await a commit/push; no new GitHub run is claimed.

The Windows executable was rebuilt with the final changes. All thirteen checked application modules in its bundle match the current source, with no foreign ICU DLL. It opened a responsive Qt window and closed normally. A temporary QA executable using the production PYZ/bootstrap and exact collected `_internal` runtime generated through the real local Ollama model without importing torch/transformers, then generated real SmolLM2 text through PyTorch, with Python removed from `PATH`. The temporary executable was removed afterward. Model files remain external; the published v1.0.0 download is unchanged.

## Local LLM / PyTorch recipe - 2026-10-19

Implemented at my request, with explicit approval for optional `torch` and `transformers` dependencies. The eleventh recipe is now named **Local LLMs**; **PyTorch / Transformers** remains its default engine. Choose a local Hugging Face safetensors model, edit a prompt, and generate; **Dataset rows** prepends the instruction to one selected string-valued column. Prompt format, token limits, temperature/top-p, seed, CPU/CUDA, row limit, and decoding time are visible controls. Cancel/failure keeps the previous report. Opening the recipe does not load or download a model.

Start with the local folder `models/SmolLM2-135M-Instruct` (gitignored), downloaded from revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. The real model generated 23 tokens in about 1.3 seconds on CPU in the initial check; that is a local observation, not a general performance guarantee. The frozen build includes CPU PyTorch, so use CPU there. A CUDA-enabled source environment can select CUDA.

`llm.py` returns frozen JSON-safe `TextGenerationResult` data: generated and formatted prompt text, input/output token counts, source row positions, stop reasons, timing, model fingerprint, parameter/context size, precision/device, settings, and runtime versions. Save/Open retains outputs and unrun edits without loading weights; reruns require the external model folder. Generated answers are not evaluated analytical results. The existing MLP recipe is unchanged.

Learning points: a tokenizer maps text to token IDs; chat templates add the model's expected role/special-token format; causal decoding appends one token at a time; `eval()` and `torch.inference_mode()` select inference behavior; temperature zero is greedy, while temperature/top-p sample from next-token probabilities. The model context holds both input and generated tokens. A seed helps reproduce sampling on the same runtime, but hardware and decoding time limits matter. Training/fine-tuning, embeddings, quantized inference, and graph composition are separate learning tasks.

The isolated `.venv` has the LLM, dev, GUI, and packaging dependencies. Use `.venv/Scripts/python.exe -m mlcookbook` / `-m pytest`. The optional extra is lazy: classical imports and the setup screen do not initialize PyTorch. Offline tests construct tiny real checkpoints; the separate LLM CI job needs no pretrained-model download. Build with `build_windows.ps1`, as described below. The published v1.0.0 download predates this feature.

Verification: **326 tests passed** in the isolated environment (pandas 3.0.6, scikit-learn 1.9.1, CPU torch 2.6.0, Transformers 4.57.6). Ruff, dependency checks, and the example workflow passed. After display refinements, 37 focused LLM/experiment checks passed. Native Windows inspection covered setup, generated text, minimum window size, dataset mode, reopened output, and the requested bottom-row placement of Local LLM and Model Comparison. The newer sklearn emits SVM probability deprecation warnings; the existing SVM tests still pass.

The Windows executable was rebuilt after the card reorder. All twelve changed application modules in the bundle match the source; no foreign ICU DLL was collected. The application opened a responsive Qt window and closed normally. A QA executable reusing the production PYZ, bootstrap hooks, and exact collected `_internal` runtime generated real SmolLM2 text offline with Python removed from `PATH`; the temporary QA executable was removed afterward. Model weights remain in `models/SmolLM2-135M-Instruct`, outside the app bundle.

CI follow-up: the pushed LLM commit passed lint, Linux core 3.12/3.13, and Windows desktop checks; the new Linux LLM job failed collection because `libEGL.so.1` was absent. The workflow now installs `libegl1`/`libopengl0` before the headless Qt checks. That workflow correction, the card reorder, and final documentation await a subsequent commit/push.

## v1.0.0 release - 2026-10-19

The first tagged release packages all ten recipes, the usability improvements, saved experiments, and regression model comparison. The package and experiment provenance now report **1.0.0**. [Release notes](CHANGELOG.md) define the scope; the [GitHub release](https://github.com/aadabathon/MLcookbook/releases/tag/v1.0.0) hosts the portable Windows x64 ZIP and its SHA-256 checksum. Extract the entire ZIP and keep `_internal` beside `MLcookbook.exe`. The local shortcut continues to use `dist/MLcookbook/MLcookbook.exe`.

Release verification: **299 tests passed** on Python 3.12 with pandas 2.2.3; Ruff and `examples/basic_usage.py` passed. The preceding milestone also passed the full suite on pandas 3.0.6. GitHub CI checks the release commit on Linux Python 3.12/3.13 and Windows Python 3.12. PyTorch and graph composition remain separate next steps.

## Saved experiments and comparison - 2026-10-17

Implemented at my request: **Save experiment / Open experiment** in the header, plus **Model Comparison** as a tenth recipe. Try the comparison demo, change a setting, save a `.mlcb`, close the app, and reopen it: the prior report and the unrun change return without retraining. The file embeds the full parsed dataset, so it still opens if the original CSV/Parquet moves. Saving is explicit and captures the current recipe only.

For your own regression study: import data, choose a numeric target/features, choose seed/test fraction and model settings, and run. Ridge and random forest receive identical training folds, with preprocessing fitted separately inside each fold. Lowest mean CV RMSE selects the pipeline, which is then refitted on training rows and evaluated on the reserved test rows. The report shows both candidates' CV curves/table and the selected model's test metrics. Change settings using CV; repeatedly consulting the test score weakens its independence.

The core additions are `comparison.py` (plain frozen comparison results, exact source row/fold positions) and `experiments.py` (versioned JSON + Parquet snapshots, fingerprints, package-version notes, atomic writes). Original import options and report run timestamp/seed/versions are retained. Current controls can differ from the saved report's request, and the stale-result banner survives reopening. Opening a valid experiment replaces the active dataset and clears other in-memory recipe sessions; failures preserve your work.

Learning points: shared seeds alone are less auditable than recorded row membership; CV selects a model while the test set estimates its performance; every fold needs a newly fitted preprocessor; repeatability depends on data, settings, and software versions. Experiment files are readable data, not pickled trained models. Classification comparison, result export, fitted-model persistence, graph composition, and PyTorch remain future work. No dependencies were added.

Validation: **299 tests passed** with the installed pandas 2.2.3 and again with isolated pandas 3.0.6 using CI's Qt offscreen mode. Ruff passed. Native Windows renders covered the home page, comparison curves/table, minimum size, reopened unrun changes, and using the saved dataset in another recipe. The final display refinements passed seven focused GUI checks.

The Windows executable was rebuilt with the isolated build `PATH`, verified to contain the current versions of all nine changed application modules, and launch-tested: a responsive Qt window opened and closed normally. No commit was made for this milestone.

## Quality-of-life update - 2026-10-16

The nine existing recipes now share a fixed Run/status/progress bar, collapsed CSV and model/validation controls, wider column-name space, an About dialog for Python details, and a persistent warning summary. Data checks remain available after a run. Short result explanations use the existing computed metrics; long tables retain all rows in a bounded scroller. Recipe cards work with Tab and Enter/Space.

Navigation remembers each recipe's configuration and last report during the session, including unrun changes, expanded controls, import options, and scroll positions. The result banner tells you when settings differ from the displayed run. Returning does not retrain. Successful file replacements clear old recipe sessions; failed imports and runs preserve previous work. First-visit demos now train in the background.

This is an in-session usability pass: reopening the app still starts fresh. Saved projects/export, repeatable workflow design, and new recipes remain separate future work. The analytical core and dependencies are unchanged.

Verification: **253 pytest tests passed** and `ruff check .` passed. All nine recipe demos plus imported-data, error, dialog, and minimum-window layouts were rendered and inspected on native Windows. The executable was rebuilt with an isolated `PATH` after diagnosing an incompatible Poppler ICU DLL in the earlier bundle. Its changed modules match the source, and the rebuilt app opened a responsive window and closed normally. No commit was made.

## Handoff - 2026-10-12

Shipped at my request in three local commits (not pushed), in this order:

1. **Data pipeline** - `PreprocessingConfig` (median/mean imputation; standard/robust/min-max/no scaling; one-hot/ordinal encoding; rare-level pooling; "(missing)" as a level; numeric missing-value flags), datetime features (days since 1970, year, month, weekday), a `PreprocessingSummary` on every supervised result showing what was fitted on which rows, training-only k-fold cross-validation in every supervised workflow (`cv_folds`, default 5), data-quality checks (constant, mostly missing, ID-like, high-cardinality), and import quality (leading zeros kept, extra missing markers, ISO dates parsed, `ImportReport`).
2. **Modules** - multiclass for all three classifiers, `decision_tree_regression`, `random_forest_regression`, ridge/lasso on `linear_regression` plus residual diagnostics, `kmeans`, `pca`, `chi_square`, `anova` (with Kruskal-Wallis), correlation confidence intervals and normality warnings, and `association_scan`.
3. **GUI** - seven ready recipes (K-Means, PCA, Association Scan are new), a Task switch on trees/forests, multiclass targets, model settings, the Preprocessing & validation dialog, extra missing markers on import, data checks, and cross-validation/preprocessing tables in every report.

Verification at the three-phase handoff: **207 pytest tests passed**; `examples/basic_usage.py` ran end to end; every recipe's demo plus multiclass-tree, regression-tree/forest, multiclass-logistic, and data-check screens were rendered and inspected; the Windows exe was rebuilt and launch-tested.

Interview prep (the parts worth being able to explain): why preprocessing must be refitted inside each CV fold; why CV runs on training rows only; ridge vs lasso geometry; what residual normality/heteroscedasticity tests do and don't imply; macro vs per-class metrics and one-vs-rest AUC; silhouette vs inertia for choosing k; why PCA needs scaling; Cramér's V vs the chi-square p-value; when Kruskal-Wallis beats ANOVA; why many tests need Benjamini-Hochberg.

Follow-up the same day: **SVM** (`svm`, `svm_regression`: rbf/linear/poly kernels, C, support vectors, Platt-scaled probabilities, target standardized for SVR so C and epsilon are unit-free) and **neural networks** (`neural_network`, `neural_network_regression`: scikit-learn MLP, Adam with batches of 32, L2 alpha, early stopping on 10% validation rows, loss and validation curves, parameter count) with recipes for both, plus **CI** (`.github/workflows/ci.yml`: ruff, library tests on Ubuntu 3.12/3.13, full app suite on Windows offscreen). 229 tests and `ruff check .` pass; the exe was rebuilt and launch-tested. Interview angles: margin vs C, the kernel trick, why SVR and MLPs want a standardized target, what early stopping's validation curve shows, parameters vs rows.

Limits: in-memory CSV/Parquet only, nothing saved across sessions, no result export, no hyperparameter search or model comparison table, no mypy.

## Daily workflow

```powershell
.venv\Scripts\activate            # if using a venv (recommended)
# No venv? This machine has Python 3.11/3.12/3.13 and bare `python` is 3.11 WITHOUT the project.
# Prefix everything with `py -3.12` instead, e.g.  py -3.12 -m pytest   /   py -3.12 -m mlcookbook
pip install -e ".[dev,gui]"       # once, or after changing pyproject.toml
pytest                            # run the full suite
pytest tests/test_dataset.py -k schema   # run one thing while iterating
python examples/basic_usage.py    # smoke-test the public API end to end
python -m mlcookbook              # launch the desktop app
```

App architecture: `gui/` is dumb (widgets + signals), `src/mlcookbook` is UI-free, and root `app.py` is the single seam that knows both — it owns the window, delete-and-redraw navigation, demo datasets (`DEMOS`), and report builders (`REPORT_BUILDERS`). To wire a new recipe: flip it to `variant="ready"` with a `key` in `gui/recipes.py`, build a screen in `gui/screens/` (declaring its task and params), and add a `DEMOS` entry and a report builder in `app.py`.

Packaging: close the app, then run this from the repository root in PowerShell. Use an isolated `PATH` for the build so PyInstaller cannot collect incompatible DLLs from other tools. A Poppler `icuuc.dll` on the tooling `PATH` caused the QtCore "specified procedure could not be found" startup error; a clean build with only Python and Windows directories fixes that collision.

```powershell
# First-time setup only; this session's .venv already has these dependencies.
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e ".[dev,gui,llm]" pyinstaller

# Rebuild after source changes, with the app closed.
./build_windows.ps1
```

The script verifies build targets, isolates `PATH`, includes the supported causal-model implementations/fast tokenizers and runtime metadata, then restores the environment. The older `--exclude-module torch` command would omit the LLM runtime; use the script now. Model weights remain in external folders, outside the bundle.

The output is `dist/MLcookbook/MLcookbook.exe`; launch it once after each build and verify generation with a local model. **The exe is a frozen snapshot**: editing `.py` files never changes it. If your shortcut shows old behavior, rebuild — or run from source with `.venv/Scripts/python.exe -m mlcookbook`. The desktop/Start Menu shortcuts point at that exe, so keep the path stable. `dist/`, `build/`, `.venv/`, and `models/` are gitignored.

Work in small loops: write the test first (or alongside), run it red, make it green, run the full suite, commit.

## What's built (status)

| Recipe | Core | GUI report (built-in example until user data is imported) |
|---|---|---|
| Local LLMs | `llm.py`, `ollama.py`, `llamacpp.py` | PyTorch, Ollama or llama.cpp prompt/text-column generation, timing/tokens, model/runtime provenance; explicit model setup |
| Linear Regression | `regression/linear.py`, `regression/diagnostics.py` | actual vs predicted, residuals, CV, residual checks, coefficients; OLS/ridge/lasso (diabetes) |
| Logistic Regression | `classification/logistic.py` | sigmoid (binary) or recall by class (multiclass), metrics, confusion matrix, coefficients (breast cancer) |
| Decision Trees | `classification/tree.py`, `regression/tree.py` | drawn tree, depth curve, importances, rules; classification or regression (breast cancer) |
| Random Forests | `classification/forest.py`, `regression/forest.py` | ROC / recall by class / actual vs predicted, structure, importances (breast cancer) |
| K-Means | `clustering/kmeans.py` | clusters on PC1/PC2, elbow + silhouette scans, sizes, profiles (wine) |
| PCA | `dimensionality/pca.py` | scree plot, scores, loadings (wine) |
| Association Scan | `statistics/scan.py` (+ `association.py`, `categorical.py`, `groups.py`) | ranked pairs with adjusted p-values, detail for the strongest pair (breast cancer) |

Every supervised report also shows cross-validation (mean ± SD) and the fitted preprocessing table.

## Random forests milestone - 2026-09-27

Implemented at my request as the fourth ready recipe. `random_forest` / `RandomForestResult` are exported from `mlcookbook` and `mlcookbook.classification`; `classification/forest.py` reuses shared binary validation, training-only preprocessing (without numeric scaling), and stratified evaluation. No new dependencies.

Defaults are 100 bootstrap trees, sqrt feature sampling, no depth cap, at least 5 training samples per leaf, a 25% test split, and seed 0. Python callers can change these settings; the GUI retains the defaults and uses its existing CSV/Parquet and column-selection controls. The report starts with a held-out ROC curve, followed by train/test metrics, a forest-size summary, feature importances, confusion matrix, and warnings.

Learning points: a forest averages many randomized trees' class probabilities; the positive-class probability must be selected by its label, not assumed to be column 1. Importance values are for encoded columns and reflect impurity reduction, with a warning about high-cardinality bias and causal interpretation. ROC coordinates are computed in the core with sklearn. The GUI never fits models or computes ROC curves.

Tests in `tests/test_forest.py` cover predictive signal, leakage prevention, class ordering, reproducibility, settings/target validation, warnings, and frozen/JSON results. Shared file and GUI integration tests exercise forest runs from both CSV and Parquet, including nullable features. Regression forests and multiclass support remain future learning work.

Verification at completion: **110 pytest tests passed**; the public API demo completed; the built-in and imported-Parquet forest reports were visually inspected; the Windows executable was rebuilt, confirmed responsive on launch, and closed normally. No commit was made.

## CSV/Parquet milestone - 2026-09-27

Implemented at my request across linear regression, logistic regression, and decision trees:

- `src/mlcookbook/io.py`: shared `load_dataset` reader, explicit CSV delimiter/encoding, PyArrow Parquet loading, and validation for headers, malformed records, empty data, and nested values.
- `gui/screens/report_screen.py`: file picker, first-20-row preview, column-role controls and missing counts, target/features, positive class, and an enabled Run Analysis flow. Reports identify the dataset and selections used.
- `app.py`: imported-data session state, report builders reused for demo/custom data, and background loading/training with recoverable errors. Switching recipes after import does not train a demo model.
- `preprocessing/builders.py`: normalize nullable categorical/string/boolean missing markers inside the pipeline, retaining training-only imputation.
- PyArrow added to `pyproject.toml` and included in the Windows build. Packaging excludes unused Torch/TensorFlow/Keras packages from the global environment; revisit those exclusions if a future workflow actually needs them.

Verification at completion: **80 pytest tests passed**, all three imported-data reports and the preview were visually inspected on native Windows, and the rebuilt `dist/MLcookbook/MLcookbook.exe` was launch-tested, confirmed responsive, and closed normally. No commit was made.

Coverage lives in `tests/test_io.py` (formats, invalid files, all three workflows, nullable data), `tests/test_app_import.py` (controls, worker completion, recovery, navigation, and deferred tree callbacks), and `tests/test_preprocessing.py` (nullable imputation fitted on training rows only). GUI integration tests require the GUI extra and skip when PySide6 is unavailable.

Limits at the CSV/Parquet milestone: local flat tables that fit in RAM; `.csv` and `.parquet` extensions only; no saved sessions or result-export UI; selections reset on navigation; existing model parameters and the 25% test split remain fixed. The two classifiers are still binary-only. Statistical-role overrides do not repair or convert source values.

Future format and workflow ideas are recorded in [ROADMAP.md](ROADMAP.md#directions-under-discussion); they are proposals, not an implementation queue or a replacement for my curriculum.

## The codebase in one paragraph

`Dataset` (core/dataset.py) wraps a DataFrame and infers a `VarType` per column (core/schema.py) — that type, not the pandas dtype, decides which analyses apply. Every analysis is a plain function taking a `Dataset`, validating against the schema, computing with scipy/sklearn, and returning a frozen dataclass that mixes in `Result` (core/results.py) so `.to_dict()` is always JSON-safe. `preprocessing/build_preprocessor` returns an *unfitted* ColumnTransformer that gets fitted inside a Pipeline on training data only — that's the leakage guarantee. Two reference implementations exist: `statistics/association.py` (correlation) and `regression/linear.py` (with `classification/logistic.py` as the binary-classifier variant, sharing `classification/_binary.py`). Everything you build is a variation on one of those files.

## Rules I set for myself

- Analyses **return objects, never print**. Warnings go in `result.warnings`.
- Validate against `schema()`, don't guess from dtypes.
- Fit preprocessing on train only. If I ever call `.fit` before `train_test_split`, stop.
- scipy/sklearn do the math; I do orchestration, validation, and interpretation.
- Every new result type gets a `json.dumps(result.to_dict())` test.
- Keep `mlcookbook/__init__.py` small — the public API should fit in one glance.
- Files are UTF-8 (a UTF-16 README once broke the build on this machine).

## Suggested build order (my curriculum)

_Status 2026-10-12: items 1-5 below were built at my request; see the handoff above. Kept as the study order for understanding the code._

Roughly increasing difficulty; full list with hints in [ROADMAP.md](ROADMAP.md).

1. **Chi-square + Cramér's V** — first solo feature; pure copy of the correlation template. Learn: contingency tables, expected counts, effect size vs p-value.
2. ✅ (agent-built) **Logistic regression + classification metrics** — copy of the regression template plus a new `evaluation/` result. Learn: classification metrics, class imbalance, probability vs label predictions.
3. **One-way ANOVA + Kruskal–Wallis** — first analysis with an assumption-based fallback. Learn: when parametric assumptions matter, eta-squared.
4. **Pairwise association scan** — dispatch the right test per column pair from `VarType`. This is the payoff of the schema design and the first "meta" feature.
5. **Cross-validation utility** — structured per-fold results. Learn: variance of estimates, why a single split lies.

After those: regularized regression, regression forests (binary decision trees and random forests already exist), k-means, PCA, residual diagnostics, then the FastAPI layer.

## When I use AI agents

Agents read `AGENTS.md` / `CLAUDE.md` and know not to build roadmap features for me — unless I explicitly say "build it", in which case they go end to end (core, tests, GUI, docs, exe rebuild) and I review. Good uses: reviewing my implementation, explaining a statistical concept, writing tedious test cases after I've written the logic, debugging. Bad use: "implement ANOVA" — that defeats the purpose.

## Scratch space for ideas

- (jot things here)


## Native wearable workflow — 2026-10-22

The PSoC belt project is now integrated directly into **Movement Study**. After collecting/auditing recordings, prepare the belt dataset and run `wearable.py export-mlcookbook` in its analysis repository. Open the resulting `export.json` in this card, then **Run grouped baselines**. Whole-person/session splits stay fixed; the final test remains sealed by default. Save an experiment to keep settings/results and the typed feature/lineage snapshot together. Reopen and rerun offline even if the original export folder moved. **Export evaluation…** retains metrics, predictions, probabilities and source times as JSON.

The native function is also called by the wearable CLI, so GUI and CLI share evaluation logic. IDs/times cannot be predictors, and test rows are not handed to the other recipe cards. Logistic regression/random forest are baselines, not the DEEPCRAFT temporal CNN. The PSoC project contains a clearly marked synthetic smoke study for trying the workflow; its scores are not evidence of real activity/fall recognition. The user still supplies physical build, real recordings, Studio training/export and on-board checks.

Verification of the movement port: **454 MLcookbook tests** and **47 wearable pipeline tests** passed, plus the four host firmware suites and actual firmware-dump import check. Ruff and dependency checks passed. Native Windows rendering was inspected at normal and minimum sizes; synthetic import/run/export/save/reopen succeeded. The rebuilt executable creates its native window with Python removed from PATH; its five integration modules were checked against final source. Evidence is in the PSoC project’s `workingDir/build/mlcookbook-port/`. No real classifier accuracy or hardware behavior is claimed, and no Git commit was made.


The same desktop entry point accepts direct offline handoffs:

```powershell
python -m mlcookbook --movement-study
python -m mlcookbook --study C:/path/to/study/mlcookbook/export.json
python -m mlcookbook --experiment C:/path/to/study.mlcb
# The portable EXE accepts the same arguments.
```

Opening verifies/imports the study or restores the saved snapshot without fitting
a model. The empty Movement Study card and new imports default to sealed test
groups; saved experiments retain their explicit unrun choices without running
them. Qt command-line options are forwarded separately. Failed imports leave
the app usable with the error in its status area. The PSoC companion now provides
`wearable.py study` to assemble prepared windows/CNN tracks, feature exports and
native grouped baselines in one new folder, plus `verify-study` and a PowerShell
launcher. Its `study.json` hashes link all derived outputs; raw sessions remain
separate evidence. Synthetic provenance propagates from sessions automatically,
and mixed real/synthetic preparation is rejected.
