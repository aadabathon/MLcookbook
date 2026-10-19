# Changelog

## Unreleased

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
