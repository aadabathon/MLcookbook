# Local LLM inspection and performance lab

The native **Local LLMs → Workspace → Research lab** workspace provides a stored-tensor inventory, actual llama.cpp CPU computation captures and repeatable server performance trials. Research artifacts have their own `.mlcl` save format. Generation and structured text workflows retain their existing `.mlcb` experiments.

This is the first inspection/performance delivery from [the research roadmap](LOCAL_LLMS_ROADMAP.md). Later token-probability tools, device/kernel profiling, intermediate activation values, evaluation sweeps, embeddings and Kitchen LLM nodes remain separate stages.

## Inspect a model

1. In **Model inspection**, select a local GGUF with **Open / inspect GGUF…**. Reading descriptors and hashing the selected file run in a cancellable job; the model is not loaded for inference.
2. Search the tensor inventory by name, shape, storage type or group. Select a tensor for all recorded fields, including block size, stored bytes and absolute file offset. Metadata is independently selectable.
3. Read stored element/byte accounting separately from runtime memory. Quantized block layouts determine storage size. Views, tied tensors, padding, cache allocations and backend buffers make file size an unsuitable estimate of peak RAM or VRAM.

The core `mlcookbook.llm_lab.inspect_gguf` also accepts an explicitly selected list of shards. No adjacent shards or server-reported paths are opened automatically. Missing/incompatible shard selections and unknown storage types remain visible. Large tokenizer arrays have bounded validated previews with counts and omissions.

Inventory groups follow recorded tensor names. The inventory describes model storage; the Compute graph view supplies actual operation dependencies.

## Capture computation

1. Keep the selected GGUF path in Model inspection, then choose **Compute graph**.
2. Leave **Capture helper** empty to use the helper supplied with the app. If needed, select a different locally built ML Cookbook helper explicitly.
3. Select the installed llama.cpp runtime directory, or leave it empty to locate `llama-server` on PATH when Capture is clicked. The initial helper supports the verified Windows x64 `b11193 / 4e7481175` DLL set. An unsupported runtime produces an explicit compatibility error before the version-specific tensor ABI is used.
4. Enter a short capture prompt, choose the number of decode steps and deadline, then click **Capture graph**. This starts a separate, app-owned CPU helper process and loads the selected model there. Cancellation retires that helper; an existing inference server is left running.
5. Select a prefill or decode phase, expand a layer group, and select an operation. Inspect its type, shape, byte strides, GGML storage span, operation parameters, aliases, inputs and consumers. **Inputs** and **Consumers** trace recorded dependencies. Search and paging retain all nodes behind the displayed subset.

The helper captures actual GGML scheduler evaluation callbacks and their tensor dependency closure. Nodes marked `observed` were queried by that callback; dependency nodes can be present without that flag. Grouping follows recorded names, while edges follow captured references. A dependency closure is not a device kernel timeline, a universal graph for every input or proof of the connected server process's graph.

The initial capture uses raw text with tokenizer special tokens, greedy sampling and the recorded CPU/context/batch settings. It does not apply the server's chat template. Saved provenance records input token IDs, capture phase, helper/runtime identities, selected model hashes and configuration. A helper-loaded model identity identifies that research execution; an existing server's model alias is a separate observation.

Use **Compare phases…** to inspect structural differences between the selected phase and prefill. Comparisons match unambiguous recorded names and retain ambiguity instead of aligning unrelated nodes by allocation address or list position.

**Import capture…** reads the same versioned JSON contract without requiring a helper, runtime or model. Importing a file never executes its recorded paths. The GUI shows unavailable backend/source-reference fields explicitly. Storage spans include view/stride behavior; summing them does not establish peak allocation or memory traffic.

## File paths and server addresses

| Task | Input | Running HTTP server needed? |
|---|---|---|
| Inspect GGUF / capture computation | Selected local GGUF; capture also uses an installed runtime folder | No |
| Generate / performance trials with llama.cpp | Running server URL and its loaded model ID/alias | Yes |
| Generate with Ollama | Ollama server URL and an installed Ollama model | Yes |

A selected GGUF is not a server address. Generate's optional **GGUF fingerprint** hashes the selected file without loading it. The capture **Runtime directory** is the folder containing `llama-server.exe` and DLLs, not the model folder.

Ollama normally listens at `http://127.0.0.1:11434`; the app's llama.cpp default is `http://127.0.0.1:8080`. Use the port of the server you actually started. For example, start a separately installed llama.cpp server explicitly on port 8081:

```powershell
llama-server -m "C:/models/model.gguf" --alias local-model --host 127.0.0.1 --port 8081 --ctx-size 2048
```

Then use `http://127.0.0.1:8081` and loaded model `local-model` in Performance, or refresh models in Generate. Connection refused means no server accepted the connection at that address. Model/runtime loading compatibility is assessed by the runtime separately from the connection. Research file inspection and CPU capture can be used independently of this server.

## Run performance trials

1. Open **Performance**, enter the loopback server address and loaded model, and explicitly inspect server capabilities. An empty model field can be filled from observed single-model identity.
2. Enter the trial prompt, warmup count, measured repetitions, output cap and per-request deadline. The displayed plan accounts for all requests and output caps. Trials run serially.
3. Choose a prompt-cache policy. **Server default / uncontrolled** uses the existing chat-generation path. Explicit no-cache/prefix-reuse requests use native completion and template rendering; unsupported endpoints fail visibly. Requested cache policy, explicitly reported applied policy and reported cached tokens have separate scopes.
4. Run trials. Inspect individual outcomes, output lengths, first-answer/reasoning arrival, raw server data and measurement definitions. Warmups remain in the raw table and are excluded from measured aggregates. Cancel retains completed trials and accounts for interrupted/unattempted work.
5. Compare client wall duration and backend decode throughput using the separate plots and summary table. Missing measurements remain unavailable.

Client chunk arrival is not an individual-token or device timing. Whole-action setup, generation-request wall time, prompt processing and decoding have different boundaries. Decode throughput uses a matching backend count/duration pair. Existing end-to-end token throughput retains its original generation-request wall-time denominator. Reasoning may be included in reported combined output counts.

Warmups alone do not establish uncached, cold-storage or exclusive-server conditions. Restarting a server does not flush the operating system's file cache. Natural answers can stop at different lengths, so these trials are useful application observations rather than proof of equal computational work or output quality. Failed and timed-out trials remain in the accounting.

## Save, reopen and export

**Save lab…** atomically writes a bounded, checksummed `.mlcl` archive containing settings and completed research artifacts. **Open lab…** restores those artifacts offline with the original model files and server absent. Opening a lab performs no capture or generation. Invalid files preserve the current lab.

**Export JSON…** publishes the recorded document as inspectable JSON. Model weights, runtime DLLs and executables stay external. The lab archive is declarative data and does not execute recorded paths. Research state remains independent of ordinary active data, examples, Movement Study holdouts and Kitchen source snapshots.

The main **Save experiment** action applies to Generate and is disabled while Research lab is selected. Use the lab's own save controls for research. Switching away and back retains research controls and artifacts in memory.

## Build the Windows helper and app

The helper source, pinned upstream headers and license are under [`tools/llama_research`](tools/llama_research). It calls the installed llama.cpp/GGML numerical runtime; it does not implement model kernels. No extra Python dependency is required.

```powershell
./tools/llama_research/build.ps1
./build_windows.ps1 -Preview
```

Building the helper requires an existing Visual Studio x64 C++ toolchain; `-VcVars` can select its `vcvars64.bat`. The preview includes the helper executable and upstream license. Inference runtime DLLs and model weights are not bundled. See the helper README for the exact supported ABI/build and capture contract.

The candidate is `dist/ui-preview/MLcookbook/MLcookbook.exe`. The installed shortcut bundle remains separate until Adam reviews/promotes the candidate.

Verification covers native checks at three display scales, source/bundle matching, actual bundled-helper capture, offline reopening and a normal executable launch/close with Python removed from PATH. The exact PR validation, delivery evidence and preserved prior preview are recorded in [ADAM.md](ADAM.md).
