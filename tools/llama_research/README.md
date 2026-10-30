# llama.cpp graph capture helper

This is the native capture component of **Local LLMs → Research lab → Compute graph**. It runs an explicitly selected local GGUF through llama.cpp and records GGML operation/tensor descriptors observed by the backend scheduler's evaluation callback, including each queried operation's dependency closure. Tensor connections come from `ggml_tensor::src`, not an architecture diagram inferred from tensor names.

The captured run is separate from the connected server. It does not attach to, reconfigure, stop or replace an existing `llama-server`. No weights, servers, packages or models are downloaded. Ordinary HTTP inference remains available independently of this helper.

## Supported runtime

The first helper supports **Windows x64, CPU, llama.cpp build 11193 / upstream commit `4e7481175`**, using the public headers vendored under `vendor/`. Both the Python wrapper and the native executable verify the three ABI DLL hashes below before making version-specific calls. A different build is rejected clearly, even when its public version string happens to match. This is deliberate: the default-parameter functions return structs whose layouts depend on the compiled runtime.

| DLL | SHA-256 of the verified binary |
|---|---|
| `llama.dll` | `f99c04c7444ad30a7e306d1f1f5a2ad517e2bc7b753a6dfd654ae7931cd1bade` |
| `ggml-base.dll` | `7e4e8cf1edeeb0f8b40de183095a39a51bf6090ea09a1cf928512a250c481d60` |
| `ggml.dll` | `bb616de54ef57f8804d1d22a4bf104e2d98c8e8399b08374e70a7c096322b851` |

Other builds need an explicit supported-runtime update: review the headers/ABI, update and rebuild the helper, record the tested binaries and rerun capture/compatibility evidence. Removing the identity checks is not a runtime migration strategy. The wrapper additionally records the selected server's `--version` output, helper identity and the identities of discovered GGML backend DLLs. These are file/build observations, not proof about another running server process.

## Build

The existing Visual Studio C++ x64 toolchain is sufficient. No Python dependency or third-party capture library is required; the helper uses Windows' built-in BCrypt for SHA-256 checks.

From the repository root:

```powershell
./tools/llama_research/build.ps1
```

If Visual Studio is in another location, provide the existing toolchain script:

```powershell
./tools/llama_research/build.ps1 -VcVars 'C:/path/to/VC/Auxiliary/Build/vcvars64.bat'
```

The helper is written to `tools/llama_research/build/llama-graph-capture.exe`. Build artifacts and real-model capture evidence stay in this ignored build folder. `CMakeLists.txt` also supports an installed MSVC CMake generator; the supplied script invokes the installed compiler directly.

The Windows preview packages this executable as `tools/llama_research/llama-graph-capture.exe` alongside the vendored license. Runtime DLLs and model files remain external. The UI can use the bundled helper or an explicitly selected build, plus an explicit runtime directory. Default runtime discovery checks for `llama-server` on PATH only when Capture is requested.

## Capture contract

`mlcookbook.llm_graph.capture_graph` handles runtime checks, hashing, a hidden owned child process, deadlines, cancellation, bounded logs/output, validation and provenance. The module has no Qt or scientific/model imports. Its other APIs are `load_graph_capture`, `validate_graph_capture` and `compare_graphs`.

The version-1 JSON contract contains:

- `format: "mlcookbook-llama-graph"`, integer `version: 1`, `provenance`, `graphs`, and `warnings`.
- One `prefill` graph followed by the requested `decode` graphs. Each has a local ID, input token IDs, source token position and node list.
- Each node's local ID, recorded name, GGML operation/code, type, shape, byte strides, ordered input IDs and their GGML source-port slots, raw integer operation parameters, scheduler-query observation flag, storage-span bytes, view source and view offset.
- Helper/model/runtime file identities, source revision, capture point, prompt/tokenization policy, context/cache/batch/thread settings, timestamp, deadline and locally measured whole-action duration.

Captured node IDs are local to one phase. Structural comparison uses unique exact tensor names and reports ambiguous or unnamed identities rather than matching allocation addresses or list position. Changes to operation, shape, type, strides, storage span, parameters and alignable input dependencies remain inspectable.

Graph validation checks JSON/version/budgets, finite values, graph/node identities, ordered ports, dimensions, storage/identity records, dependency existence and cycles. Empty tensors with zero dimensions are preserved: the Qwen3.5 hybrid state graph legitimately contains them. Strides and byte spans describe tensor storage views; aliases can share an allocation, so their byte spans must not be summed as independent peak RAM.

### Current capture configuration and limits

- A new empty CPU context; four threads; context capacity 1,024; batch and microbatch capacity 512.
- Raw UTF-8 prompt with the model tokenizer's special-token handling. No server chat template is applied.
- At most 512 input tokens and 1–16 greedy decode steps. Context/cache/offload settings are recorded separately from the connected server.
- Single-file GGUF models. Split-shard capture is rejected until the complete loaded-file identity and selection contract is supported.
- A total budget of 60,000 tensor/operation nodes, a 64 MiB JSON budget, bounded logs, and a requested 1–600 second whole-action deadline. Cancellation/deadline termination affects only the owned helper.

This capture contains descriptors and operation dependencies. It does not capture intermediate values, operation timings, device kernels or GPU execution. Scheduler-query observation and dependency closure are recorded explicitly; fused backend kernels may not have a one-to-one relationship with logical operations. Source links identify the recorded upstream revision; exact per-operation source mappings are unavailable unless separately supplied.

## Verified local evidence

The existing local model files were used without modifying them:

| Model | Each phase's nodes | Each phase's input edges | Phases |
|---|---:|---:|---|
| SmolLM2-135M-Instruct Q4_K_M | 1,274 | 1,571 | Prefill and one decode |
| Qwen3.5-2B Q4_K_M | 1,786 | 2,135 | Prefill and one decode |

The Qwen capture preserves recurrent/hybrid state operations and zero-extent tensors. Complete validated evidence is stored locally under `build/smol-capture-validated.json` and `build/qwen-capture-validated.json`; those files and model weights are not repository fixtures or bundled weights. Unit tests use small declarative fixtures and ordinary owned child processes, so they do not require llama.cpp, model downloads or the helper build.

## Source and license

The vendored headers are from [llama.cpp revision `4e7481175`](https://github.com/ggml-org/llama.cpp/tree/4e7481175), covered by the accompanying [MIT license](vendor/LICENSE). The capture mechanism uses the public evaluation callback and GGML tensor descriptors. Numerical inference, tokenization, sampling, model loading and backend execution are performed by the external llama.cpp/GGML libraries.
