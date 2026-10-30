# Local LLMs research roadmap

Requested by Adam on 2026-10-29, with the initial inspection/performance implementation subsequently authorized. Expand the existing **Local LLMs → llama.cpp server** option into a model inspection, performance and experimentation workbench. Bring compatible capabilities to Ollama while preserving each engine's native information. Internal compute-graph inspection and the performance lab are the first priorities.

This document plans the work; it does not implement the milestones or authorize implementing them merely because they appear here. Deliver explicitly requested stages end to end: core, native GUI, persistence, tests, documentation and Windows preview verification. Effort sizes below are relative, not delivery dates.

## First inspection/performance delivery — 2026-10-29

The first vertical slice of stages 0–3a now has a native Research lab: bounded GGUF inventory and file comparison; actual CPU prefill/decode capture with tensor dependency exploration and phase comparison; serial server performance trials with warmups, explicit requested cache policies, separate client/backend measurements and failure/cancellation accounting; and checked offline `.mlcl` saves plus JSON export. See [the research guide](LLM_RESEARCH.md).

Real capture is verified on SmolLM2-135M and Adam's Qwen3.5-2B hybrid model. It runs a separate helper using the verified Windows x64 b11193/4e7481175 DLL set and single-file GGUFs. This establishes actual computation descriptors/dependencies, not live-server instrumentation or operation/kernel timing. Native benchmark import/process profiles, other capture ABIs/backends, bounded numerical weight/activation samples and the later token/evaluation/embedding/Kitchen stages remain planned. The detailed checklists retain that broader scope rather than declaring entire milestones complete from this first slice.

Source verification passed 931 tests, Ruff and native 1080×720 checks at 100%/125%/150% scaling. Windows preview delivery evidence is recorded in ADAM.md after packaging verification.

## Intended outcome

Open a local model, explore its actual computation, measure how it runs, compare controlled changes and reuse its outputs in analytical workflows. A useful research session should answer questions such as:

- Which operations and tensors make up this layer, and how do they connect?
- How does the graph differ between processing a prompt and generating the next token?
- Where are time and memory going, and what evidence supports that attribution?
- What changes when context length, cache policy, offload, batch size or quantization changes?
- Which tokens were plausible alternatives, and how did sampling affect selection?
- Does a faster configuration retain accuracy on a fixed evaluation dataset?
- Can embeddings and validated generated columns feed existing ML recipes and Kitchen?

Keep three kinds of graph distinct in the UI and saved artifacts:

| Graph | What it represents | Evidence |
|---|---|---|
| Architecture view | Model blocks and stored tensors | GGUF descriptors plus an explicitly identified architecture mapping |
| Captured compute graph | Operations and tensor dependencies built for a particular execution configuration | An actual instrumented llama.cpp capture |
| Kitchen workflow | User-composed data and analysis operations | The existing typed Kitchen document |

A diagram inferred from tensor names must never be labeled a captured compute graph. A captured logical operation graph must also remain distinguishable from a backend scheduling or kernel trace.

## Existing foundation

The current product already supplies local PyTorch, Ollama and llama.cpp generation; streamed text/reasoning; cancellation; bounded text-row workflows; strict structured-output validation; derived datasets; reference agreement; manual saved-run comparison; export; and offline experiment reopening.

The current llama.cpp connector talks directly to one already-running model server. It reads health, model identity and properties, then generates through chat completions. The optional GGUF selection checks the file and computes a hash; it does not parse tensors or inspect the computation. Existing timing records are useful observations, not a benchmark protocol or an operation profiler.

Preserve this working path throughout development. PyTorch remains available, and scientific/model libraries stay out of default Data/catalog startup. Navigation, session restoration and opening saved reports perform no inference, graph capture, model load or network fetch.

## Product shape

Retain one **Local LLMs** route and engine selector. Introduce subordinate views as their stages arrive:

| View | Primary work |
|---|---|
| Generate | Existing prompts, text rows and structured tasks |
| Inspect | Model files, tensor inventory, captured graphs, prompt/token inspection |
| Performance | Request measurements, benchmark trials, profiles and comparisons |
| Experiments | Evaluation datasets, configuration sweeps and quality/performance comparisons |
| Embeddings | Vector creation, similarity, projection, clustering and retrieval evaluation |

Kitchen remains the place for composing workflows. Link to relevant Kitchen artifacts from Local LLMs instead of adding another workflow editor here.

Use the current ink-black/violet theme, restrained headings, native Qt controls and QPainter plots. A graph gets room to explore, with a selectable detail inspector and compact controls. Keep Run/Capture/Cancel and job progress accessible outside scrollers. Show unsupported features with a concrete reason; distinguish editable settings from observations. Retain keyboard navigation, narrow-window containment and offline views.

## Delivery sequence and dependencies

| Stage | Scope | Relative effort | Depends on |
|---|---|---|---|
| 0 | Capability, provenance and lab artifact foundation | Medium | Existing connectors |
| 1 | Offline GGUF and tensor inspection | Large | 0 |
| 2 | Actual graph capture and exploration | Very large; early feasibility gate | 0; tensor linking uses 1 |
| 3a | Server performance lab | Large | 0; can progress alongside 1–2 |
| 3b | Native benchmark import and explicit runtime profiles | Medium–large | 3a; process launching is a separate slice |
| 4 | Prompt, token and sampling inspection | Large | 0; captures integrate with 2 |
| 5 | Controlled evaluations and configuration sweeps | Large | 3a and existing structured workflows; token analysis uses 4 |
| 6 | Graph profiling and intermediate tensor analysis | Very large | 2 and 3a; compare through 5 |
| 7 | Embeddings and retrieval lab | Large | 0 and 5's evaluation conventions |
| 8 | LLM and embedding operations in Kitchen | Large | Relevant stages 5/7 plus existing Kitchen artifact rules |
| 9 | Compatibility, scale and Windows delivery | Cross-cutting; final full pass | All delivered stages |

Start the graph-capture feasibility work in stage 2 early, rather than finishing every surrounding feature first. The first inspection/performance release should combine stages 0–1, a real stage-2 graph and stage 3a. Metadata inspection by itself does not meet Adam's compute-graph goal. Selected-model and hybrid-architecture support can expand after this first vertical slice.

## 0. Capability, provenance and lab artifact foundation

### Capabilities and requests

- [ ] Add plain capability records for the connected runtime: available, unavailable with reason, or not yet tested. Record the probe/result time and runtime/model identity. A version string alone is insufficient evidence that a particular endpoint or option works.
- [ ] Discover optional features only through explicit connection/refresh actions or the requested operation. Use bounded read-only probes where possible; an operation requiring inference must be an explicit action.
- [ ] Keep server startup options, per-request settings, requested values, explicitly reported applied values and locally observed measurements separate. Save raw backend metadata alongside normalized fields.
- [ ] Recheck relevant identity/capabilities before a run and invalidate cached assumptions after model/server changes. Absence of a full model fingerprint remains visible.
- [ ] Define explicit run, token, context, response, capture, memory and disk budgets for each lab action. Show planned trial count and estimated work before a sweep; enlarge the research limits deliberately rather than silently removing the current Generate limits.
- [ ] Keep typed-prompt research independent of ordinary active tabular data. Accepting generated tables/vectors into Data remains explicit.

### Results and persistence

- [ ] Use frozen, JSON-safe result summaries with warnings, plus separate runtime artifacts for graphs, trial tables and numeric captures. No tensors, widgets, estimators or open connections in result JSON.
- [ ] Design a dedicated versioned lab archive, provisionally `.mlcl`, with an explicit manifest, checksummed typed tables and bounded numeric/graph artifacts. Finalize its schema and extension in this stage. Save atomically and validate archive members, sizes, shapes, types and checksums before replacing current work.
- [ ] Leave `.mlcb` v1's exact `experiment.json` + `data.parquet` contract intact. Existing Generate/structured experiments keep working. Large research artifacts must not be hidden in report JSON or admitted by weakening v1 validation.
- [ ] Store input/configuration fingerprints, complete and partial outcomes, artifact relationships, runtime identities, units and measurement definitions. Record the scope of every file identity: one file, all explicitly selected shards, or the helper's loaded model set.
- [ ] Reopen saved labs with servers stopped and source models absent. Restore selected view/node/trial, settings, layouts, reports and stale associations without executing anything. New state requires explicit migration; legacy PyTorch/Ollama/llama.cpp experiments retain their current migrations.
- [ ] Add explicit export of selected tables, captures and reports. Report export can use current structured data and existing rendering; adopting another document/plotting dependency requires separate approval.

**Done when:** fake old/partial/new servers produce honest capability records; missing metrics remain unavailable; a small lab round-trips offline; corrupt or oversized artifacts preserve the current session; legacy `.mlcb` files remain valid and unchanged.

## 1. Offline GGUF and tensor inspection

### File and model inventory

- [ ] Explicitly open a GGUF or selected shard set without starting a server or allocating the full model. Index descriptors with bounded reads and cancellation. Never follow a server-supplied filesystem path automatically.
- [ ] Show architecture metadata, model dimensions, tokenizer metadata, chat-template text/identity where present, tensor count, stored parameter/storage accounting and quantization distribution. Separate declared values from totals computed from descriptors.
- [ ] Offer a searchable tensor table: exact name, shape, logical element count, storage type, byte range and layer/group. Account for quantized blocks and alignment correctly; do not use a floating-point element-size assumption for every storage type.
- [ ] Group known architectures into embeddings, repeated blocks, attention/recurrent/state components, feed-forward components and output head. Preserve generic descriptor inspection for unknown architectures.
- [ ] Add model-file comparison: changed metadata, missing/added tensors, shape/type changes and storage differences. Quantization comparisons record evidence of a common source checkpoint; matching aliases or filenames do not establish this.
- [ ] Keep unknown fields, unsupported storage types and incomplete shard sets explicit. Detect malformed dimensions/offsets, out-of-file ranges and files that change during inspection.

### Bounded weight statistics

- [ ] Make reading/dequantizing selected weights a separate bounded action after descriptor inspection. Use supported upstream readers/kernels rather than reimplementing quantization math.
- [ ] Show selected tensor ranges, norms, nonfinite counts, histograms and sampled values with exact selection policy and coverage. Stored quantized weights and decoded numeric samples are different artifacts.
- [ ] Treat stored bytes, logical parameter counts, shared/tied tensors and runtime allocation as different measures. Do not infer peak RAM/VRAM from file size.

**Done when:** a large local GGUF can be browsed responsively; malformed/incomplete files fail clearly; a tensor can be found and compared; saved descriptors reopen without the file. The user's Qwen3.5 hybrid model receives an explicit architecture-support assessment rather than being forced into a generic attention-only diagram.

## 2. Actual graph capture and exploration

### 2a. Early capture feasibility gate

- [ ] Pin a supported upstream llama.cpp revision and prove a minimal CPU capture with an explicitly selected existing model. Inspect the current graph/export and evaluation-callback APIs before choosing the hook.
- [ ] Prototype a small research helper or narrowly instrumented server build that exports an actual graph. Decide between those paths using the required capture phase, provenance and maintenance cost. Keep inference and numerical kernels in llama.cpp/GGML.
- [ ] Demonstrate prompt processing and at least one decode step. Export operation nodes, tensor dependencies and enough identity to inspect a real layer. Record unsupported/fused/opaque operations as such.
- [ ] Write the versioned capture contract and reproducible helper build instructions. Record upstream commit, local patch revision, compiler/build options and helper executable identity. No automatic runtime replacement or installation.
- [ ] Define the relationship to the connected server precisely. A separately loaded helper model is a reproducible research execution, not proof of that server process's live graph. Live-server attribution requires a capture from the instrumented process itself.

**Gate:** the real capture, its provenance and its limitations are reviewable before investing in the full viewer. If a backend cannot expose a required field, document the gap and the next instrumentation step; do not substitute an inferred diagram.

### 2b. Capture contract and native viewer

- [ ] Store graph/capture IDs; node IDs; operation codes/names/parameters; ordered input/output references; tensor shapes, strides and types; and views/aliases. Preserve unknown operation data without inventing semantics.
- [ ] Record phase, sequence/token positions, input token IDs, actual prompt/batch/microbatch shapes, context occupancy, cache state, attention/backend/offload settings and capture limits. Record tokenizer/template identity where observable, plus hashes of all helper-loaded model shards and auxiliary files in scope.
- [ ] Distinguish logical graphs before/after transformations from backend scheduling traces. Identify capture location and optimization/fusion scope. A graph for one configuration must not be presented as universal to that model.
- [ ] Provide a read-only Qt overview grouped by layer, expand/collapse, pan/zoom, fit selection, search, upstream/downstream tracing and keyboard selection. A selected node shows exact inputs, outputs, operation parameters, tensor descriptors and matching source references at the recorded revision when a mapping exists.
- [ ] Reuse suitable canvas mechanics without imposing Kitchen's editable workflow semantics. Bound initial layout/rendering work, preserve all data behind collapsed groups and keep the detail inspector within its viewport.
- [ ] Compare prefill/decode or two configurations with a structural diff. Match by explicit structural/semantic evidence; unstable allocation addresses or node order do not establish identity across captures.
- [ ] Extend the initial conventional-transformer capture to Adam's existing Qwen3.5 hybrid model, keeping architecture-specific operations visible and unknown source mappings explicit. GPU captures follow verified CPU support rather than an assumption of identical traces.

**Done when:** Adam can open a captured real-model graph, expand a layer, trace a tensor through its actual producer/consumers, inspect a prefill/decode difference and reopen it with the model/server/helper absent. Large captures, unknown operations, unsupported runtime versions and cancellation have tested behavior.

## 3. Performance lab

### 3a. Server measurements and repeatable trials

Define the measurements before drawing charts:

| Measurement | Definition and required distinction |
|---|---|
| Whole action wall time | Local monotonic duration including the action's discovery/setup/hash work |
| Generation request wall time | Local duration from submitting generation through terminal response; preserve the existing timing scope |
| First output arrival | Client-observed first nonempty reasoning and first nonempty visible-answer arrival, separately |
| Prompt processing | Backend-reported processed-token count and duration, with cached-token coverage where reported |
| Decode throughput | Backend-generated tokens divided by the matching backend decode duration |
| End-to-end throughput | Generated tokens divided by matching client wall time; retain the current workflow metric's meaning |
| Load/queue time | Separate reported phases only when observed; no invented decomposition from unrelated totals |
| Memory | Static estimates, process memory, allocated device memory and sampled peaks as separately named observations |

- [ ] Store raw measurements with unit, clock/source, scope, token-count coverage, request ID and missing-field reasons. Validate nonfinite/negative values and incompatible count/duration pairs.
- [ ] Show per-trial tables, request timelines, latency distributions, prompt/decode throughput and context/output lengths. Stream-arrival traces describe chunks; ordinary SSE chunks do not establish one-token timing.
- [ ] Track reasoning and visible-answer output where observable. Do not divide visible-answer length by a combined reasoning/answer token count and label it answer-token throughput.
- [ ] Add explicit protocols for warm model/uncached prompt, warm model/reused prefix, model load and controlled context depth. Cache conditions must be observed or labeled uncontrolled. Restarting a process alone does not establish cold filesystem-cache conditions.
- [ ] Save warmups, measured repetitions, trial order, concurrency, prompt token counts, context occupancy, requested output cap, actual output length, seed, sampling, cache policy and background-load notes. Start with serial trials.
- [ ] Use paired inputs and interleaved/randomized configuration order where useful. Report individual samples, sample count, median, mean and spread. Tail percentiles or uncertainty intervals need sufficient samples and a stated method.
- [ ] Retain timeout/error/cancelled outcomes and planned/completed trial counts. Preserve completed trials after cancellation and support explicit resume as a new linked run, with conditions revalidated. Failed/partial trials do not become successful fast observations.
- [ ] Separate fixed-work throughput tests from realistic task-generation tests, where early EOS and differing answer lengths affect workload. Record observed counts and stop reasons for both.
- [ ] Poll optional slot/server metrics with bounded overhead and capability checks. Shared server counters describe the server unless exclusivity and attribution are established; they are not automatically per-request measurements.

**Done when:** a fixed trial plan produces inspectable raw samples and correct aggregates; warm/cache conditions and failed trials remain visible; cancellation preserves completed evidence; saved plots and comparisons work offline. Fake-server fixtures verify timing arithmetic, incomplete usage and chunk/token distinctions without wall-clock CI thresholds.

### 3b. Native benchmarks and runtime profiles

- [ ] Import structured `llama-bench` output first, retaining raw rows, executable/build identity, model identity, parameters and repetition data. Keep native-runtime benchmark measurements separate from server/application latency: upstream `llama-bench` excludes tokenization and sampling.
- [ ] Compare prompt processing, generation and supported combined/context-depth workloads using upstream benchmark machinery. Do not write replacement inference benchmarks or kernels in Python.
- [ ] Define reusable server profiles: executable, selected local model/shards, context, threads, batch/microbatch, backend/offload, attention/cache settings and explicitly supported flags. Initially compare user-started profiles and record unverified startup values honestly.
- [ ] In a separately requested process-control slice, launch a selected local server/helper with structured arguments, isolated loopback port, bounded logs, readiness/deadline handling and cancellation. Track exact process ownership. Stop/restart only app-owned processes; external servers remain user-owned.
- [ ] Automate startup-setting sweeps only after that lifecycle slice is verified. No automatic dependency/runtime installation, model downloads, cloud fallback or modification of external research directories is implied.
- [ ] Add bounded concurrent-load trials later: per-request latency/failures, aggregate throughput, queue behavior and fairness. Record slot count and competing workload; do not blend these with serial baselines.

**Done when:** imported native results retain their original semantics and compare offline; a configured owned process, if implemented, starts/cancels/exits reliably while an unrelated server stays running. Startup controls are never displayed as ordinary per-request options.

## 4. Prompt, token and sampling inspection

- [ ] Explicitly render a chat template through supported llama.cpp facilities, inspect token IDs/pieces/special tokens and compare prompt variants. Record the rendering request/options and template identity. A preview establishes actual submitted formatting only when the generation path verifies the same rendering/configuration.
- [ ] Add system/multi-turn message editing with exact submitted message capture. Expose supported sampling controls such as temperature, top-p, top-k, min-p, penalties, stop sequences, seed and sampler order with defaults and task-specific applicability.
- [ ] Display selected generated tokens with available alternatives/log probabilities and token-aligned provenance. Preserve raw scores and identify pre/post-sampling distributions when known. Missing probability mass in top-k output must not be treated as a full distribution.
- [ ] Allow side-by-side sampling trials and token-level differences on the same input/configuration. Cross-model tokenizer differences need explicit alignment rules; token IDs are local to a tokenizer.
- [ ] Add context-budget inspection: rendered prompt count where verified, requested generation allowance, reported context allocation and truncation/completion outcomes. Do not infer verified remaining capacity from an incomplete template preview.
- [ ] Keep probability inspection distinct from correctness/calibration assessment. Generated reasoning text is an output artifact, not an activation trace or proof of the underlying causal computation.
- [ ] Offer Ollama parity where its installed API/model supports the information, with native-field provenance. Do not emulate unavailable tokenizer/graph introspection as if the runtime exposed it.

**Done when:** a prompt can be inspected down to token pieces, a sampled output can be reviewed with supported alternatives, and score/formatting limitations survive Save/Open. Tests cover special tokens, byte fragments, Unicode, missing scores, option rejection and changed templates/models.

## 5. Controlled evaluations and configuration sweeps

- [ ] Extend existing manual saved-run comparisons into explicit experiment plans over frozen input/reference snapshots. Preserve source positions, duplicates, skips, missing references and exact bound columns.
- [ ] Sweep model, prompt, sampling, context, reasoning options and quantization where supported. Request options can run against one server; model/startup changes use verified profiles or explicit manual transitions until owned-process control exists.
- [ ] Start with existing structured tasks. Add exact field agreement, categorical precision/recall/F1/confusion data, numeric extraction errors and coverage where the reference type supports them. Keep JSON validity, coverage and semantic correctness separate.
- [ ] Add human review/blinded side-by-side assessment for open-ended tasks. Any later model-based scoring must record the judge, rubric and limitations and remain separate from reference-grounded metrics.
- [ ] Use fixed development/evaluation partitions and repeated seeds/trials. Keep references out of prompt bindings, and reserve final assessment before tuning. LLM-derived predictors must preserve downstream training-only preprocessing/CV and target-leakage checks.
- [ ] Save experiment plan, execution order, budgets, raw outputs, trial outcomes, model/runtime identities and aggregate methods. A changed plan creates a new linked experiment rather than rewriting old evidence.
- [ ] Compare quality, latency, throughput, memory and storage together, including Pareto views. Label memory estimates/observations and unmatched model identities. Do not select a universal winner from incomparable tokenizers, tasks or cache conditions.
- [ ] Compare quantizations only with recorded evidence about source checkpoint, tokenizer, template and conversion/runtime settings. File size and a shared alias alone are insufficient.
- [ ] Provide explicit CSV/Parquet/JSON export of trial and evaluation tables plus provenance. Accepted derived data remains an independent snapshot; a research run never silently replaces active data.

**Done when:** two configurations run repeatedly on the same evaluation rows, failures and coverage are accounted for, a quality/performance comparison is reproducible from saved raw results and final-test isolation is covered by tests.

## 6. Graph profiling and intermediate tensor analysis

### Execution and memory overlays

- [ ] Attach observations to exact run/capture/graph/node or backend-event IDs. Refuse overlays with incompatible model/configuration/phase identity; do not attach timings by matching display names alone.
- [ ] Begin with a supported CPU profiling path, then add specific device backends with verified timing mechanisms. Distinguish host enqueue time, synchronized elapsed time and device-event timing. Asynchronous GPU execution cannot be timed correctly merely by surrounding an enqueue call with a host clock.
- [ ] Preserve fused/many-to-one operation-to-kernel mappings and overlapping events. A sum of operation/kernel durations need not equal total wall time. Unmapped execution stays visible.
- [ ] Show measured hot operations/layers, backend placement, transfers and allocation/lifetime observations when the capture exposes them. Keep static byte/FLOP estimates and their formulas/coverage separate from observed memory traffic or device peak allocation.
- [ ] Add optional supported process/device telemetry with sampling interval, device scope, attribution and peak-coverage limitations. No new telemetry dependency is assumed approved. Energy requires an actual supported measurement source, not TDP multiplied by time.
- [ ] Quantify instrumentation overhead with matched profiled/unprofiled trials. Synchronization and value capture can alter the workload being studied; retain both observations.

### Selected intermediate tensors

- [ ] Capture selected nodes/layers/token positions under explicit byte/time budgets. Offer summary-only capture before full values: range, mean/spread, norms, nonfinite counts and histograms.
- [ ] Store original shape/type, decoded representation, selected slices, sampling policy, completeness and omitted bytes. Distinguish weights, activations, KV/state data and output logits.
- [ ] Compare compatible captures across prompt/decode positions, layers or configurations. Record alignment and numerical comparison method; quantization/backend changes may alter both shapes and values.
- [ ] Save bounded numeric arrays/tables separately from JSON summaries, using strict non-executable readers and validated dimensions/dtypes. Reopen all views offline.

**Done when:** a real captured graph has correctly attributed measured hotspots and one bounded intermediate-value view; incompatible overlays are rejected; capture overhead and partial coverage are visible. CPU evidence precedes claims about device profiling, with supported backends listed explicitly.

Activation editing, ablation, gradient attribution, fine-tuning and custom inference kernels remain separate research proposals. This stage is read-only inspection and measurement.

## 7. Embeddings and retrieval lab

- [ ] Add capability-checked embedding requests for supported llama.cpp/Ollama models. Record model/tokenizer identity, dimension, pooling, normalization, input limits and truncation policy. Reject overlong inputs by default where supported; otherwise record the unverified condition.
- [ ] Preserve text source positions/fingerprints and skipped-row outcomes. Store vectors as typed numeric artifacts with provenance, not huge nested report dictionaries. Never combine incompatible embedding spaces merely because their dimensions match.
- [ ] Inspect norms, similarity matrices and nearest neighbors with explicit cosine/distance settings. Use existing NumPy/sklearn operations; no vector database is required for the first bounded local corpus.
- [ ] Offer PCA projections, clustering and explicit use in applicable existing recipes. Learned projection/scaling/classifier steps follow current training-only validation rules for predictive evaluations.
- [ ] Add a local retrieval experiment: explicit document/chunk snapshots, chunk size/overlap and source spans; query snapshots; known-relevant document/chunk judgments; recall@k, reciprocal rank and coverage where applicable. Exclude self/duplicate matches according to recorded rules.
- [ ] Add retrieved-context generation as a subsequent slice: inspect selected passages/source spans, exact assembled prompt, retrieval configuration and generated answer. Evaluate retrieval and answer quality separately; a cited passage alone does not verify a claim.
- [ ] Save vectors, chunks, queries, relevance judgments, retrieval results and run metadata for offline inspection. Keep embedding/reranking model requirements separate from the text-generation server profile.

**Done when:** imported text can become inspectable vectors, neighbors and a PCA view; a small labeled retrieval task compares two configurations; saved results work offline; positional lineage, embedding identity and predictive leakage boundaries have tests.

## 8. LLM and embedding operations in Kitchen

- [ ] Add explicit typed operations incrementally: generation/extraction, validation/derive, embeddings and retrieval. Reuse the delivered UI-free cores rather than creating another inference implementation.
- [ ] Design the new text/vector-matrix/run/capture artifact types and port contracts before extending the canvas. Existing one-column vectors and a matrix of embeddings are not interchangeable.
- [ ] Preserve exact source positions, configuration/model fingerprints, consumed port identity and complete runtime artifacts. Prompt/model/source changes mark descendants stale; retries create linked artifacts with compatible identity.
- [ ] Keep reference/target columns out of prompt/embedding inputs when evaluating predictions. A graph connection is subject to the existing lineage/leakage rules, not an exception to them.
- [ ] Inspect per-row outputs, validation failures, embeddings, trial summaries and links to research captures from selected nodes. **Run to selected node**, cancellation and errors retain previous completed outputs with their original provenance.
- [ ] Extend `.mlck` deliberately with explicit version migration and checked artifact storage. Preserve v1/v2 readers and ordinary/example/study independence. Research captures may be embedded by a supported format or referenced with checksums and a clear portability/export policy.
- [ ] Demonstrate two useful paths: text → extract → validate → derived columns → existing analysis; and documents → embeddings → retrieval → generation/evaluation. Explicit **Use dataset** remains the route to ordinary active data.

**Done when:** both paths run from the native canvas, inspect intermediate outputs, reject incompatible/leaky connections, retain completed work after cancellation and reopen without servers or original model files.

## Architecture and implementation boundaries

- `gui/` continues to render plain data and emit actions. `app.py` owns coordination, jobs, source/session identity and result formatting. The core imports no Qt, and no second application entry point is added.
- Keep `llamacpp.py`/`ollama.py` focused on supported transport/inference operations. Introduce small submodule-only inspection, graph, benchmark/experiment, embedding and lab-I/O modules as needed; choose final names per stage. No generic plugin framework, DI container or large base-class hierarchy.
- Use upstream llama.cpp/GGML for loading, tensor decoding, graph construction and inference. Select the descriptor reader/helper approach during stage 1; any new Python dependency, telemetry library, build runtime or distribution obligation needs explicit consideration before adoption.
- Native capture tools have a pinned, reproducible source/build contract and explicit process ownership. External runtimes/models remain external unless a later delivery decision approves packaging them. Archive readers never execute a helper, load code or download models.
- Every action runs through cancellable bounded work. Inspecting a saved artifact is distinct from rerunning it. File/network/process errors preserve the prior completed session and ordinary data.
- Large graph layouts, trial series and tensor views need bounded construction and lazy detail loading. Preserve all captured data; display downsampling/aggregation must record its scope and leave full export available within the capture budget.

## 9. Verification and delivery gates

Each implemented slice includes relevant behavior coverage and a completion note; a roadmap checkbox changes only after the evidence exists.

| Area | Required evidence |
|---|---|
| Transport/capabilities | Fake servers with old/missing endpoints, rejected options, fragmented streams, idle deadlines, terminal usage, identity changes and cancellation |
| Files/captures | Small offline fixtures for valid/unknown/malformed GGUF descriptors, graphs, aliases, shards, budget limits, corrupt archives and numeric artifacts |
| Measurement | Deterministic timing/aggregation fixtures covering units, denominators, cache policy, failed trials, incomplete counts and incompatible overlays |
| Evaluation/lineage | Duplicate indexes, missing references, isolated test labels, configuration changes, retries, embedding-space mismatch and leakage checks |
| Compatibility | Existing PyTorch/Ollama/llama.cpp `.mlcb` migration, `.mlck` versions, source-free reopen and no execution during navigation/restoration |
| Native UI | Keyboard operation, minimum-size containment, large graphs/tables, cancellation/recovery, 100%/125%/150% scaling and retained purple/math identity |
| Real runtime | Existing small CPU model, then Adam's explicitly selected Qwen3.5 hybrid model; matched benchmark/profile evidence and clearly listed supported backends |
| Windows bundle | Rebuild separate `-Preview` candidate; launch/close and verify real capture/inference/offline reopening using the frozen runtime |

Record tracked/untracked baselines before implementation/UI work and preserve unrelated changes, Movement Study holdouts and saved experiments. Do not make checkpoint commits or promote a preview over the shortcut build without Adam's instruction. Documentation-only roadmap edits require link/diff review, not an application rebuild.

## First review checkpoints

1. **Foundation:** capability/result/artifact schemas and legacy round-trip evidence.
2. **Inspection proof:** real prefill/decode capture, tensor inventory and one expandable layer with exact provenance.
3. **Useful performance lab:** repeated server trials with explicit cache conditions, raw samples and correct phase metrics.
4. **Research integration:** matched graph/profile observations, token alternatives and a fixed quality/performance evaluation.
5. **Broader reuse:** embeddings/retrieval and both Kitchen example paths, with offline portable artifacts.

The first implementation request can select stages 0–3a as a cohesive inspection/performance project, delivered in those reviewable slices. The remaining stages stay explicit follow-ups rather than being silently added to that implementation.

## Upstream basis and feasibility notes

Reviewed on 2026-10-29. These are upstream capabilities and research entry points, not a claim that the installed server exposes every feature. Pin implementation/capture versions and verify the actual local runtime.

- [llama.cpp server API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): documents template/tokenization, probabilities, embeddings, cache/timing and monitoring facilities. Per-feature support and metric scope require probing/contract tests.
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): defines metadata and stored tensor descriptors/data; it does not supply a complete runtime execution trace.
- [GGML graph API](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h): exposes graph/node access and DOT export. A production-quality capture artifact and Qt inspector are proposed work.
- [llama.cpp evaluation callback example](https://github.com/ggml-org/llama.cpp/tree/master/examples/eval-callback): demonstrates operation/tensor observation during inference. It is an instrumentation entry point, not a ready-made portable profiler.
- [llama-bench documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md): describes runtime benchmark workloads, repetitions and structured results, excluding tokenization/sampling time.
- [Ollama generation API](https://docs.ollama.com/api/generate), [running models](https://docs.ollama.com/api/ps) and [embedding API](https://docs.ollama.com/api/embed): supply native generation measurements, supported probability output, residency information and vectors. Preserve their meanings and installed-version limitations instead of manufacturing llama.cpp-equivalent internals.

Further capabilities such as multimodal inputs, tool execution, adapters and speculative decoding can be proposed after the six workbench capacities above have a verified foundation. They are not prerequisites for the requested inspection/performance project.
