# Changelog

## Unreleased

- Direct offline desktop handoff through `--movement-study`, `--study EXPORT_JSON` and `--experiment FILE_MLCB`; imports and snapshot opens never auto-train or evaluate holdouts. The PSoC companion assembles/verifies relocatable offline studies and propagates synthetic-data provenance.


- Native **Movement Study** recipe and `load_wearable_study` / `evaluate_wearable` APIs: verified fixed-group wearable imports, training-only grouped CV, validation-selected baselines, explicit final test, saved offline snapshots and probability/source-window exports. The PSoC companion CLI shares the native evaluator; no new MLcookbook dependencies.

- Repeatable Local LLM data workflows: bound prompt templates, extraction/summarization/rewriting/classification presets, independent structured JSON validation, per-row outcomes/retries/cancellation, explicit typed derived datasets with row lineage, saved-run comparisons, reference agreement and JSON export. Source data, unrun settings and recent runs reopen offline; no new dependencies.
- llama.cpp server engine in **Local LLMs**: health/loaded-model checks, chat SSE streaming and reasoning, bounded rows, idle cancellation/deadlines, usage/timing and optional explicit GGUF-file provenance, offline saved outputs, and migration of older Ollama/PyTorch experiments. Shared standard-library HTTP; no new dependencies or managed server processes.
- Ollama engine in **Local LLMs**: local connection/model discovery, quantized model support through Ollama, streamed previews, cancellation/deadlines, bounded text rows, server/model provenance, and offline experiment reopening. No new Python dependencies; existing PyTorch experiments remain compatible.
- Local LLM · PyTorch recipe with offline Hugging Face GPT-2/Llama/Qwen2/Qwen3 generation, prompt/chat controls, bounded row generation, cancellation, model/runtime provenance, and saved outputs.
- Optional `llm` dependencies, offline checkpoint tests/CI, and `build_windows.ps1` for the Windows bundle with the CPU LLM runtime.

## 1.0.0 - 2026-10-19

First tagged release of ML Cookbook, with a Python library and a portable Windows desktop app.

- Ten desktop recipes: linear regression, logistic regression, decision trees, random forests, SVM, neural networks, k-means, PCA, association scan, and regression model comparison. Trees, forests, SVM, and neural networks support both classification and regression.
- CSV/Parquet import, statistical column roles, data checks, configurable preprocessing, and training-only cross-validation. Results carry metrics, visual data, preprocessing details, and warnings as JSON-serializable data.
- Save/Open experiment: versioned `.mlcb` files embed the full typed dataset, settings, last report, and provenance. Reopening restores unrun edits and report state without retraining or needing the original source file.
- Model comparison: ridge and random forest share recorded splits and training folds. Mean cross-validation RMSE selects the model before its evaluation on the test holdout.
- Desktop usability: background demo runs, persistent Run/status controls, warning summaries, clearer result explanations, keyboard-accessible recipe cards, and per-recipe session restoration.
- Windows packaging uses an isolated build path to avoid the foreign ICU DLL collision that previously prevented QtCore from loading.

### Windows download

Extract the whole `MLcookbook-v1.0.0-windows-x64.zip` archive and run `MLcookbook/MLcookbook.exe`. Keep `_internal` beside the executable. Python and runtime dependencies are included. The release includes a SHA-256 checksum file.

### Scope

Experiments save one recipe at a time and contain data, settings, and reports, not fitted estimators. Comparison currently covers ridge versus random forest regression. Neural networks use scikit-learn's MLP. PyTorch recipes, graph composition, standalone result export, and classification comparison remain future work.
