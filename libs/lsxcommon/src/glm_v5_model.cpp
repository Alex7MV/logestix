#include "glm_v5_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <spdlog/spdlog.h>

#include "lsxcommon/glm_inference.h"
#include "lsxcommon/glm_inference_load.h"
#include "lsxcommon/model_runtime.h"

namespace lsxcommon {

const std::string GlmV5Model::kType = "glm-5.2";

GlmV5Model::GlmV5Model(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)} {}

GlmV5Model::~GlmV5Model() { Unload(); }

bool GlmV5Model::Load()
{
    if (loaded_) return true;
    if (cfg_.model_path.empty()) {
        spdlog::error("[glm-model] empty model_path");
        return false;
    }
    if (!LoadTokenizerDb(cfg_.model_path)) return false;
    loaded_ = true;
    return true;
}

gpu::InferenceResult GlmV5Model::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    // ── TDD Cycle 10: drives the full frozen 79-layer execution loop ─────────
    // One owned async execution context shared by every layer pass: dense
    // layers 0-2 (post-attn RMSNorm -> gate/up -> swiglu -> down), IndexShare
    // attention + MoE router + NVFP4 expert stacks for layers 3-77, and the
    // Layer-78 MTP epilogue (eh_proj -> enorm -> router -> experts -> residual
    // -> shared_head.norm -> lm_head^T raw logits) with greedy token decode.
    // The complete memory footprint lives in the unified GlmMoEArena that
    // RunGlmGeneration allocates as ONE contiguous device block (+ one pinned
    // host staging block): KV cache slots, attention/MoE workspaces and the MTP
    // execution spaces are all carved from the same allocation, so a 79-layer
    // forward performs a single device cudaMalloc and a single host one. All
    // kernel launches run on the explicit async stream `st`; no
    // cudaDeviceSynchronize is ever issued (only per-decode-step stream syncs).
    cublasHandle_t ch = nullptr;
    cudaStream_t   st = nullptr;
    const bool ch_ok = (cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    const bool st_ok = (cudaStreamCreate(&st) == cudaSuccess);
    if (ch_ok && st_ok) cublasSetStream(ch, st);

    glm::GlmExpertDb      experts;

    auto cleanup = [&]() {
        pw_.Free();
        glm::free_glm_expert_index(experts);
        if (st) cudaStreamDestroy(st);
        if (ch) cublasDestroy(ch);
    };
    auto fail = [&](std::string msg) {
        r.error_msg = std::move(msg);
        r.ok = false;
        cleanup();
        return r;
    };

    if (!ch_ok || !st_ok) return fail("cublas/stream init failed");

    // Full pin: all 79 layers (dense 0-2 + MoE router 3-77 + MTP 78) so the
    // heterogeneous loop can read every layer's router gate / norm / indexer
    // matrices during its single pass.
    cudaError_t e = glm::LoadGlmPinnedWeights(cfg_.model_path, pw_, st);
    if (e != cudaSuccess)
        return fail(std::string("load pinned weights: ") +
                    cudaGetErrorString(e));
    if (pw_.hidden != glm::kGlmHidden)
        return fail("model hidden_size does not match kGlmHidden (6144)");
    if (pw_.layers.size() != static_cast<size_t>(glm::kGlmNumLayers))
        return fail("model does not expose all 79 layers");
    if (!pw_.d_embed || !pw_.d_head)
        return fail("model embed/lm_head weights missing");

    // On-demand routed-expert index for every MoE layer (3-78): a flat
    // memory-mapped registry whose experts are mmap'd lazily through a bounded
    // LRU and staged one at a time, so the full ~385 GiB expert set is NEVER
    // resident in VRAM — the generation loop needs only the pinned weights
    // plus ~2 × one-expert staging. The layer-78 MTP block also routes
    // through per-expert stacks.
    experts = glm::load_glm_expert_index(cfg_.model_path, glm::kGlmNumLayers,
                                         glm::kGlmNumRouted);
    if (experts.num_layers != glm::kGlmNumLayers ||
        experts.experts_per_layer != glm::kGlmNumRouted)
        return fail("expert index incomplete");

    // The faithful greedy multi-token generation pass (real GlmMoeDsaForCausalLM
    // forward: MLA + interleaved RoPE + causal per-token KV cache, sigmoid MoE
    // router + shared experts, model.norm -> lm_head head). Capacity + prompt-id
    // bounds are validated inside (against the fixed 1024-slot KV staging and
    // the embed table), making the run deterministic and memory-safe.
    glm::GlmGenerationResult gen;
    e = glm::RunGlmRealGeneration(ch, st, pw_, experts, req.input_ids,
                                  req.max_gen_tokens, gen);
    if (e != cudaSuccess || !gen.ok)
        return fail(gen.ok ? cudaGetErrorString(e) : gen.error_msg);

    r.ok = true;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    r.num_layers = glm::kGlmNumLayers;
    if (!gen.tokens.empty()) {
        r.first_gen_tok_id = gen.tokens[0];
        r.first_gen_maxv = gen.first_maxv;
    }
r.gen_tokens = std::move(gen.tokens);
    r.final_logits = std::move(gen.final_logits);
    r.kv_slots_used = gen.kv_slots_used;
    r.ttft_ms = gen.ttft_ms;
    r.total_time_ms = gen.total_ms;
    r.tokens_per_sec = (gen.total_ms > 0.0)
                           ? (r.gen_tokens.size() / (gen.total_ms / 1000.0))
                           : 0.0;
    r.eos_terminated = gen.eos_terminated;
    r.weight_resident_bytes = gen.resident_bytes;

    // Decode the generated token stream into output_text (gen.tokens carries
    // only generated tokens, never the prompt). A decode failure leaves
    // output_text empty but keeps r.ok true — generation succeeded, decode is
    // best-effort.
    if (!r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);

    cleanup();
    return r;
}

void GlmV5Model::Unload()
{
    pw_.Free();
    tok_db_.reset();
    loaded_ = false;
}

SafetensorsModel GlmV5Model::Inspect()
{
    return lsxcommon::scan_directory(cfg_.model_path);
}

gpu::InferenceResult GlmV5Model::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

gpu::BenchmarkResult GlmV5Model::Benchmark(int num_tokens)
{
    gpu::BenchmarkResult r;
    if (!loaded_) { r.error_msg = "model not loaded"; return r; }
    // Synthetic prompt: repeat token 0 (a stable vocab entry) 4 times.
    const std::vector<int32_t> prompt = {0, 0, 0, 0};

    // Mirror Infer exactly: own async context, full pin, on-demand expert index,
    // faithful real forward generation loop.
    cublasHandle_t ch = nullptr;
    cudaStream_t   st = nullptr;
    const bool ch_ok = (cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    const bool st_ok = (cudaStreamCreate(&st) == cudaSuccess);
    if (ch_ok && st_ok) cublasSetStream(ch, st);
    if (!ch_ok || !st_ok) {
        r.error_msg = "cublas/stream init failed";
        if (st) cudaStreamDestroy(st);
        if (ch) cublasDestroy(ch);
        return r;
    }

    glm::GlmExpertDb experts;
    auto cleanup = [&]() {
        pw_.Free();
        glm::free_glm_expert_index(experts);
        if (st) cudaStreamDestroy(st);
        if (ch) cublasDestroy(ch);
    };

    cudaError_t e = glm::LoadGlmPinnedWeights(cfg_.model_path, pw_, st);
    if (e != cudaSuccess) {
        r.error_msg = std::string("load pinned weights: ") +
                      cudaGetErrorString(e);
        cleanup();
        return r;
    }
    if (pw_.layers.size() != static_cast<size_t>(glm::kGlmNumLayers)) {
        r.error_msg = "model does not expose all 79 layers";
        cleanup();
        return r;
    }
    experts = glm::load_glm_expert_index(cfg_.model_path, glm::kGlmNumLayers,
                                         glm::kGlmNumRouted);
    if (experts.num_layers != glm::kGlmNumLayers ||
        experts.experts_per_layer != glm::kGlmNumRouted) {
        r.error_msg = "expert index incomplete";
        cleanup();
        return r;
    }

    glm::GlmGenerationResult gr;
    e = glm::RunGlmRealGeneration(ch, st, pw_, experts, prompt, num_tokens, gr);
    cleanup();
    if (e != cudaSuccess || !gr.ok) {
        r.error_msg = gr.ok ? cudaGetErrorString(e) : gr.error_msg;
        return r;
    }
    r.ok = true;
    r.num_tokens = static_cast<int>(gr.tokens.size());
    r.num_layers = glm::kGlmNumLayers;
    r.total_time_ms = gr.total_ms;
    r.tokens_per_sec = gr.total_ms > 0.0
        ? (static_cast<double>(r.num_tokens) / gr.total_ms) * 1000.0 : 0.0;
    r.resident_bytes = gr.resident_bytes;
    return r;
}

} // namespace lsxcommon