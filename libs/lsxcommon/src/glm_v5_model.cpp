#include "glm_v5_model.h"

#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <cctype>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <filesystem>
#include <spdlog/spdlog.h>

#include "lsxcommon/glm_inference.h"
#include "lsxcommon/glm_inference_load.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

const std::string GlmV5Model::kType = "glm-5.2";

GlmV5Model::GlmV5Model(ModelInitConfig cfg)
    : cfg_(std::move(cfg)) {}

GlmV5Model::~GlmV5Model() { Unload(); }

bool GlmV5Model::Load()
{
    if (loaded_) return true;
    if (cfg_.model_path.empty()) {
        spdlog::error("[glm-model] empty model_path");
        return false;
    }
    tok_db_ = std::make_unique<ArrowTokenizerDb>();
    auto pinned = std::filesystem::path(cfg_.model_path) / "logestix" / "pinned";
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!v.ok() || !m.ok()) {
        spdlog::error("[glm-model] cannot open tokenizer arrows in {}",
                      pinned.string());
        return false;
    }
    auto st = load_tokenizer_metadata(*v, *m, *tok_db_);
    if (!st.ok()) {
        spdlog::error("[glm-model] load_tokenizer_metadata: {}", st.ToString());
        return false;
    }
    loaded_ = true;
    return true;
}

gpu::InferenceResult GlmV5Model::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    if (req.input_ids.empty()) {
        r.error_msg = "empty input_ids";
        return r;
    }
    if (req.max_gen_tokens <= 0) {
        r.error_msg = "max_gen_tokens must be positive";
        return r;
    }

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

    glm::GlmPinnedWeights pw;
    glm::GlmExpertDb      experts;

    auto cleanup = [&]() {
        pw.Free();
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
    cudaError_t e = glm::LoadGlmPinnedWeights(cfg_.model_path, pw, st);
    if (e != cudaSuccess)
        return fail(std::string("load pinned weights: ") +
                    cudaGetErrorString(e));
    if (pw.hidden != glm::kGlmHidden)
        return fail("model hidden_size does not match kGlmHidden (6144)");
    if (pw.layers.size() != static_cast<size_t>(glm::kGlmNumLayers))
        return fail("model does not expose all 79 layers");
    if (!pw.d_embed || !pw.d_head)
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
    e = glm::RunGlmRealGeneration(ch, st, pw, experts, req.input_ids,
                                  req.max_gen_tokens, gen);
    if (e != cudaSuccess || !gen.ok)
        return fail(gen.ok ? cudaGetErrorString(e) : gen.error_msg);

    r.ok = true;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    r.num_layers = glm::kGlmNumLayers;
    // output_text stays empty: full tokenizer decode is out of scope for the
    // raw hidden/logit pass; only the decoded id stream is published.
    if (!gen.tokens.empty()) {
        r.first_gen_tok_id = gen.tokens[0];
        r.first_gen_maxv = gen.first_maxv;
    }
    cleanup();
    return r;
}

void GlmV5Model::Unload()
{
    tok_db_.reset();
    loaded_ = false;
}

SafetensorsModel GlmV5Model::Inspect()
{
    return lsxcommon::scan_directory(cfg_.model_path);
}

gpu::InferenceResult GlmV5Model::SelfCheck(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    r = Infer(req);
    if (!r.ok) {
        r.selfcheck_passed = false;
        return r;
    }
    if (r.first_gen_tok_id >= 0) {
        char buf[512];
        size_t cap = sizeof(buf);
        std::vector<int32_t> single_tok = {r.first_gen_tok_id};
        if (::lsxcommon::arrow_tokenizer_decode(
                *tok_db_, single_tok.data(), single_tok.size(), buf, &cap) == 0)
            r.first_gen_text = std::string(buf, cap);
        r.selfcheck_passed =
            (gpu::TrimAsciiWhitespace(r.first_gen_text) == "Paris");
    }
    spdlog::info("[selfcheck] top1_tok={} text='{}' passed={}",
                 r.first_gen_tok_id, r.first_gen_text,
                 r.selfcheck_passed ? "true" : "false");
    return r;
}

ArrowTokenizerDb &GlmV5Model::Tokenizer()
{
    return *tok_db_;
}

} // namespace lsxcommon