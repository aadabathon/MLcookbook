// ML Cookbook read-only graph capture helper.
// Public llama.cpp/GGML ABI pinned to upstream 4e7481175 (b11193).
// Numerical inference, tokenization and sampling are performed by llama.cpp.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// These are the three public-ABI DLLs from the verified Windows b11193 build.
// Check before calling any function returning a version-specific struct.
static std::string sha256(const fs::path & path) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD object_size = 0, copied = 0;
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0 ||
        BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&object_size),
                          sizeof(object_size), &copied, 0) < 0)
        throw std::runtime_error("Cannot initialize runtime identity check.");
    std::vector<unsigned char> object(object_size), buffer(1024 * 1024), digest(32);
    if (BCryptCreateHash(algorithm, &hash, object.data(), object_size, nullptr, 0, 0) < 0)
        throw std::runtime_error("Cannot initialize runtime hash.");
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("A required runtime DLL is missing.");
    while (stream) {
        stream.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
        if (stream.gcount() && BCryptHashData(hash, buffer.data(), ULONG(stream.gcount()), 0) < 0)
            throw std::runtime_error("Cannot hash runtime DLL.");
    }
    if (!stream.eof() || BCryptFinishHash(hash, digest.data(), ULONG(digest.size()), 0) < 0)
        throw std::runtime_error("Cannot finish runtime identity check.");
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    const char * hex = "0123456789abcdef";
    std::string result;
    for (unsigned char byte : digest) { result += hex[byte >> 4]; result += hex[byte & 15]; }
    return result;
}

static std::string quote(const std::string & value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
        else if (c < 32) {
            const char * hex = "0123456789abcdef";
            out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15];
        } else out += char(c);
    }
    return out + '"';
}

struct runtime {
    HMODULE llama = nullptr, ggml = nullptr, backend = nullptr;
    template<class T> T sym(const char * name) {
        auto p = GetProcAddress(llama, name);
        if (!p) p = GetProcAddress(ggml, name);
        if (!p) p = GetProcAddress(backend, name);
        if (!p) throw std::runtime_error(std::string("Runtime ABI missing ") + name);
        return reinterpret_cast<T>(p);
    }
#define PTR(name) decltype(&name) name##_p = nullptr
    PTR(llama_backend_init); PTR(llama_backend_free);
    PTR(llama_model_default_params); PTR(llama_context_default_params);
    PTR(llama_model_load_from_file); PTR(llama_model_free);
    PTR(llama_init_from_model); PTR(llama_free); PTR(llama_model_get_vocab);
    PTR(llama_tokenize); PTR(llama_batch_get_one); PTR(llama_decode);
    PTR(llama_sampler_init_greedy); PTR(llama_sampler_sample); PTR(llama_sampler_free);
    PTR(llama_n_ctx); PTR(ggml_backend_load_all_from_path);
    PTR(ggml_op_name); PTR(ggml_type_name); PTR(llama_log_set); PTR(ggml_log_set);
    PTR(ggml_nbytes);
#undef PTR
    explicit runtime(const fs::path & dir) {
        const std::map<std::wstring, std::string> verified = {
            {L"llama.dll", "f99c04c7444ad30a7e306d1f1f5a2ad517e2bc7b753a6dfd654ae7931cd1bade"},
            {L"ggml-base.dll", "7e4e8cf1edeeb0f8b40de183095a39a51bf6090ea09a1cf928512a250c481d60"},
            {L"ggml.dll", "bb616de54ef57f8804d1d22a4bf104e2d98c8e8399b08374e70a7c096322b851"},
        };
        for (const auto & entry : verified)
            if (sha256(dir / entry.first) != entry.second)
                throw std::runtime_error("Runtime DLL identity differs from the verified b11193 / 4e7481175 ABI. Capture is unsupported for this build.");
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        AddDllDirectory(dir.c_str());
        ggml = LoadLibraryExW((dir / L"ggml-base.dll").c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        backend = LoadLibraryExW((dir / L"ggml.dll").c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        llama = LoadLibraryExW((dir / L"llama.dll").c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (!llama || !ggml || !backend) throw std::runtime_error("Cannot load selected llama.cpp runtime DLLs.");
#define LOAD(name) name##_p = sym<decltype(&name)>(#name)
        LOAD(llama_backend_init); LOAD(llama_backend_free);
        LOAD(llama_model_default_params); LOAD(llama_context_default_params);
        LOAD(llama_model_load_from_file); LOAD(llama_model_free);
        LOAD(llama_init_from_model); LOAD(llama_free); LOAD(llama_model_get_vocab);
        LOAD(llama_tokenize); LOAD(llama_batch_get_one); LOAD(llama_decode);
        LOAD(llama_sampler_init_greedy); LOAD(llama_sampler_sample); LOAD(llama_sampler_free);
        LOAD(llama_n_ctx); LOAD(ggml_backend_load_all_from_path);
        LOAD(ggml_op_name); LOAD(ggml_type_name); LOAD(llama_log_set); LOAD(ggml_log_set);
        LOAD(ggml_nbytes);
#undef LOAD
    }
};

struct node {
    std::string id, name, op, dtype;
    int op_code = 0;
    std::vector<int64_t> shape;
    std::vector<size_t> strides;
    std::vector<std::string> inputs;
    std::vector<int> input_slots;
    std::vector<int32_t> parameters;
    std::string alias;
    size_t alias_offset = 0;
    size_t bytes = 0;
    bool observed = false;
};

struct graph {
    std::string id, phase;
    std::vector<int32_t> tokens;
    std::vector<node> nodes;
    std::unordered_map<const ggml_tensor *, size_t> indices;
};

struct capture {
    runtime * rt;
    graph * current = nullptr;
    size_t total_nodes = 0;
    std::string error;
    size_t visit(const ggml_tensor * t, int depth = 0) {
        auto & g = *current;
        auto found = g.indices.find(t);
        if (found != g.indices.end()) return found->second;
        if (depth > 4096 || ++total_nodes > 60000) throw std::runtime_error("Graph exceeds node/depth capture budget.");
        size_t index = g.nodes.size();
        g.indices[t] = index;
        node n;
        n.id = "n" + std::to_string(index);
        n.name = std::string(t->name, strnlen_s(t->name, GGML_MAX_NAME));
        n.op = rt->ggml_op_name_p(t->op);
        n.op_code = int(t->op);
        n.dtype = rt->ggml_type_name_p(t->type);
        n.alias_offset = t->view_offs;
        n.bytes = rt->ggml_nbytes_p(t);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) { n.shape.push_back(t->ne[i]); n.strides.push_back(t->nb[i]); }
        for (int i = 0; i < GGML_MAX_OP_PARAMS / int(sizeof(int32_t)); ++i) n.parameters.push_back(t->op_params[i]);
        g.nodes.push_back(n);
        std::vector<std::string> inputs;
        std::vector<int> input_slots;
        for (int i = 0; i < GGML_MAX_SRC; ++i) if (t->src[i]) {
            inputs.push_back("n" + std::to_string(visit(t->src[i], depth + 1)));
            input_slots.push_back(i);
        }
        std::string alias;
        if (t->view_src && t->view_src != t) alias = "n" + std::to_string(visit(t->view_src, depth + 1));
        g.nodes[index].inputs = std::move(inputs);
        g.nodes[index].input_slots = std::move(input_slots);
        g.nodes[index].alias = std::move(alias);
        return index;
    }
};

static bool observe(ggml_tensor * tensor, bool ask, void * user) {
    auto & c = *static_cast<capture *>(user);
    if (!c.current || !c.error.empty()) return false;
    if (ask) {
        try { c.current->nodes[c.visit(tensor)].observed = true; }
        catch (const std::exception & e) { c.error = e.what(); }
    }
    // Descriptor-only collection does not request tensor readback or per-node sync.
    return false;
}

static bool abort_capture(void * user) { return !static_cast<capture *>(user)->error.empty(); }

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) std::cerr << text;
}

template<class T> static void numbers(std::ostream & out, const std::vector<T> & values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}

static std::string utf8(const std::wstring & value) {
    int size = WideCharToMultiByte(CP_UTF8, 0, value.c_str(), int(value.size()), nullptr, 0, nullptr, nullptr);
    std::string out(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.c_str(), int(value.size()), out.data(), size, nullptr, nullptr);
    return out;
}

int wmain(int argc, wchar_t ** argv) {
    try {
        if (argc != 6) throw std::runtime_error("Usage: llama-graph-capture RUNTIME_DIR MODEL PROMPT_UTF8_FILE OUTPUT_JSON DECODE_STEPS");
        fs::path dir = fs::absolute(argv[1]), model_path = fs::absolute(argv[2]);
        int steps = std::stoi(argv[5]);
        if (steps < 1 || steps > 16) throw std::runtime_error("Decode steps must be 1..16.");
        if (fs::file_size(argv[3]) > 32768) throw std::runtime_error("Prompt exceeds the UTF-8 byte budget.");
        std::ifstream input(argv[3], std::ios::binary);
        std::string prompt((std::istreambuf_iterator<char>(input)), {});
        if (!input || prompt.empty() || prompt.size() > 32768) throw std::runtime_error("Prompt must contain 1..32768 UTF-8 bytes.");
        runtime rt(dir);
        rt.llama_log_set_p(quiet_log, nullptr);
        rt.ggml_log_set_p(quiet_log, nullptr);
        rt.ggml_backend_load_all_from_path_p(utf8(dir.wstring()).c_str());
        rt.llama_backend_init_p();
        auto mp = rt.llama_model_default_params_p();
        ggml_backend_dev_t cpu_only[] = {nullptr};
        mp.devices = cpu_only;
        mp.n_gpu_layers = 0;
        auto * model = rt.llama_model_load_from_file_p(utf8(model_path.wstring()).c_str(), mp);
        if (!model) throw std::runtime_error("llama.cpp could not load the selected model.");
        capture state{&rt};
        auto cp = rt.llama_context_default_params_p();
        cp.n_ctx = 1024; cp.n_batch = 512; cp.n_ubatch = 512;
        cp.n_threads = 4; cp.n_threads_batch = 4;
        cp.offload_kqv = false; cp.op_offload = false; cp.no_perf = false;
        cp.cb_eval = observe; cp.cb_eval_user_data = &state;
        cp.abort_callback = abort_capture; cp.abort_callback_data = &state;
        auto * ctx = rt.llama_init_from_model_p(model, cp);
        if (!ctx) { rt.llama_model_free_p(model); throw std::runtime_error("llama.cpp could not create the research context."); }
        auto * vocab = rt.llama_model_get_vocab_p(model);
        std::vector<llama_token> tokens(513);
        int count = rt.llama_tokenize_p(vocab, prompt.c_str(), int(prompt.size()), tokens.data(), int(tokens.size()), true, true);
        if (count <= 0 || count > 512) throw std::runtime_error("Capture prompt exceeds the 512-token budget or could not tokenize.");
        tokens.resize(count);
        auto * sampler = rt.llama_sampler_init_greedy_p();
        std::vector<graph> graphs;
        graphs.reserve(steps + 1);
        graphs.push_back({"prefill-0", "prefill", tokens});
        state.current = &graphs.back();
        if (rt.llama_decode_p(ctx, rt.llama_batch_get_one_p(tokens.data(), int(tokens.size()))) != 0) throw std::runtime_error("Prompt evaluation failed: " + state.error);
        if (!state.error.empty()) throw std::runtime_error(state.error);
        for (int i = 0; i < steps; ++i) {
            llama_token next = rt.llama_sampler_sample_p(sampler, ctx, -1);
            graphs.push_back({"decode-" + std::to_string(i), "decode", {next}});
            state.current = &graphs.back();
            if (rt.llama_decode_p(ctx, rt.llama_batch_get_one_p(&next, 1)) != 0) throw std::runtime_error("Decode evaluation failed: " + state.error);
            if (!state.error.empty()) throw std::runtime_error(state.error);
        }
        state.current = nullptr;
        std::ofstream out(argv[4], std::ios::binary);
        if (!out) throw std::runtime_error("Cannot write capture output.");
        out << "{\"format\":\"mlcookbook-llama-graph\",\"version\":1,\"provenance\":{\"capture_kind\":\"separate_helper_execution\",\"upstream_commit\":\"4e7481175\",\"capture_point\":\"GGML backend scheduler evaluation callback; dependency closure of queried operations\",\"backend\":\"CPU\",\"n_gpu_layers\":0,\"context_tokens\":" << rt.llama_n_ctx_p(ctx) << ",\"batch_tokens\":512,\"microbatch_tokens\":512,\"threads\":4,\"flash_attention_requested\":" << int(cp.flash_attn_type) << ",\"key_cache_dtype\":" << quote(rt.ggml_type_name_p(cp.type_k)) << ",\"value_cache_dtype\":" << quote(rt.ggml_type_name_p(cp.type_v)) << ",\"sampling\":\"greedy\",\"cache_policy\":\"new empty context; prefill followed by sequential decode\",\"prompt_format\":\"raw text with tokenizer special tokens; no chat template\",\"node_budget\":60000},\"graphs\":[";
        for (size_t gi = 0; gi < graphs.size(); ++gi) {
            if (gi) out << ',';
            const auto & g = graphs[gi];
            out << "{\"id\":" << quote(g.id) << ",\"phase\":" << quote(g.phase) << ",\"source_token_position\":" << (gi == 0 ? 0 : count + int(gi) - 1) << ",\"input_token_ids\":";
            numbers(out, g.tokens);
            out << ",\"nodes\":[";
            for (size_t ni = 0; ni < g.nodes.size(); ++ni) {
                if (ni) out << ',';
                const auto & n = g.nodes[ni];
                out << "{\"id\":" << quote(n.id) << ",\"name\":" << quote(n.name) << ",\"op\":" << quote(n.op) << ",\"op_code\":" << n.op_code << ",\"dtype\":" << quote(n.dtype) << ",\"shape\":";
                numbers(out, n.shape); out << ",\"strides\":"; numbers(out, n.strides);
                out << ",\"inputs\":[";
                for (size_t j = 0; j < n.inputs.size(); ++j) { if (j) out << ','; out << quote(n.inputs[j]); }
                out << "],\"input_slots\":"; numbers(out, n.input_slots);
                out << ",\"op_parameters_i32\":"; numbers(out, n.parameters);
                out << ",\"observed\":" << (n.observed ? "true" : "false") << ",\"bytes\":" << n.bytes << ",\"view_source\":" << (n.alias.empty() ? "null" : quote(n.alias)) << ",\"view_offset_bytes\":" << n.alias_offset << '}';
            }
            out << "]}";
        }
        out << "],\"warnings\":[\"This is a separate CPU research execution, not the connected server's live graph.\",\"Nodes are actual scheduler-query dependency closures; no device kernel trace, operation timings or tensor values were captured.\",\"Raw prompt tokenization is recorded; the server chat template is not applied.\"]}\n";
        out.close();
        if (!out) throw std::runtime_error("Writing capture output failed.");
        rt.llama_sampler_free_p(sampler);
        rt.llama_free_p(ctx);
        rt.llama_model_free_p(model);
        rt.llama_backend_free_p();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Graph capture: " << e.what() << '\n';
        return 1;
    }
}
