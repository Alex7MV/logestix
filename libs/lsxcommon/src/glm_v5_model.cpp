#include "glm_v5_model.h"

#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
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
    const int seq = static_cast<int>(req.input_ids.size());
    const int hidden = glm::kGlmHidden;
    const int inter  = glm::kGlmMoeIntermediate;

    // Owned async execution context for the dense-MLP + MoE forward, released
    // on every exit path.
    cublasHandle_t ch = nullptr;
    cudaStream_t   st = nullptr;
    const bool ch_ok = (cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    const bool st_ok = (cudaStreamCreate(&st) == cudaSuccess);
    if (ch_ok && st_ok) cublasSetStream(ch, st);

    glm::GlmPinnedWeights pw;
    glm::GlmExpertTables  experts;
    glm::GlmMoEArena      arena;
    int32_t *d_ids = nullptr;
    __half *d_act = nullptr, *d_alt = nullptr, *d_result = nullptr;
    __half *d_scratch = nullptr;

    auto cleanup = [&]() {
        pw.Free();
        experts.Free();
        arena.Free();
        cudaFree(d_ids); cudaFree(d_act); cudaFree(d_alt);
        cudaFree(d_result); cudaFree(d_scratch);
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

    // Full pin: layers 0-2 (dense) + 3-77 (MoE router) + 78 (MTP) so the
    // 3-77 MoE MLP stack below can read every layer's router gate / norm.
    cudaError_t e = glm::LoadGlmPinnedWeights(cfg_.model_path, pw, st);
    if (e != cudaSuccess)
        return fail(std::string("load pinned weights: ") +
                    cudaGetErrorString(e));
    // The pipeline is hard-locked to GLM-5.2 geometry; any other topology is
    // rejected up front so the kGlmHidden-strided loops can never run OOB on a
    // mismatched (synthetic/foreign) weight set.
    if (pw.hidden != hidden)
        return fail("model hidden_size does not match kGlmHidden (6144)");
    if (pw.layers.size() < static_cast<size_t>(glm::kGlmNumLayers))
        return fail("model does not expose all 79 layers");
    for (int l = 0; l < 3; ++l) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        if (!lw.d_post_attn_norm || !lw.d_mlp_gate || !lw.d_mlp_up ||
            !lw.d_mlp_down)
            return fail("layer 0-2 dense weights missing");
    }

    // Expert tables for every MoE layer (3-77); a missing expert file just
    // leaves its slot inactive, which routing deterministically skips.
    e = glm::LoadGlmExpertTables(cfg_.model_path, experts, st, nullptr,
                                 glm::kGlmNumRouted);
    if (e != cudaSuccess)
        return fail(std::string("load expert tables: ") +
                    cudaGetErrorString(e));
    if (experts.tables.size() < static_cast<size_t>(glm::kGlmNumLayers))
        return fail("expert tables incomplete");

    // Single contiguous MoE arena (256 virtual per-expert activation slices +
    // router/expert workspaces in ONE allocation): zero runtime heap churn.
    e = arena.Allocate(seq, hidden, glm::kGlmNumRouted);
    if (e != cudaSuccess)
        return fail(std::string("moe arena alloc: ") +
                    cudaGetErrorString(e));

    const size_t hid_n = static_cast<size_t>(seq) * hidden;
    if (cudaMalloc(&d_ids, static_cast<size_t>(seq) * sizeof(int32_t)) !=
            cudaSuccess ||
        cudaMalloc(&d_act, hid_n * sizeof(__half)) != cudaSuccess ||
        cudaMalloc(&d_alt, hid_n * sizeof(__half)) != cudaSuccess ||
        cudaMalloc(&d_result, hid_n * sizeof(__half)) != cudaSuccess ||
        cudaMalloc(&d_scratch,
                   (2 * hid_n + 2 * static_cast<size_t>(seq) * inter) *
                       sizeof(__half)) != cudaSuccess)
        return fail("dense scratch allocation failed");

    e = cudaMemcpyAsync(d_ids, req.input_ids.data(),
                        static_cast<size_t>(seq) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, st);
    if (e != cudaSuccess) return fail(std::string("ids upload: ") +
                                      cudaGetErrorString(e));
    e = glm::RunGlmEmbed(st, pw.d_embed, d_ids, d_act, seq, hidden);
    if (e != cudaSuccess) return fail(std::string("embed: ") +
                                      cudaGetErrorString(e));
    e = glm::RunGlmDenseMLPStack(ch, st, pw, seq, d_act, d_alt, d_result,
                                 d_scratch);
    if (e != cudaSuccess) return fail(std::string("dense stack: ") +
                                      cudaGetErrorString(e));

    // Layers 3-77 MoE MLP: dense output -> per-layer RunGlmMoERouter (top_k=8)
    // -> RunGlmMoESlices (NVFP4 expert SwiGLU) -> residual, on the same stream.
    // Dense result lives in d_result; the MoE stack ping-pongs d_alt/d_act and
    // emits its final hidden block into d_act.
    e = glm::RunGlmMoEStack(ch, st, pw, experts, 3, glm::kGlmNumLayers, seq,
                            d_result, d_alt, d_act, arena);
    if (e != cudaSuccess) return fail(std::string("moe stack: ") +
                                      cudaGetErrorString(e));

    std::vector<__half> host_out(hid_n);
    e = cudaMemcpyAsync(host_out.data(), d_act, hid_n * sizeof(__half),
                        cudaMemcpyDeviceToHost, st);
    if (e != cudaSuccess) return fail(std::string("result readback: ") +
                                      cudaGetErrorString(e));
    e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) return fail(std::string("stream sync: ") +
                                      cudaGetErrorString(e));

    r.ok = true;
    r.num_tokens = seq;
    r.num_layers = glm::kGlmNumLayers - 1; // 0-2 dense + 3-77 MoE = 78 layers
    // output_text stays empty: the MoE loop emits hidden-state vectors, not
    // decodable token text; full token decode is the 79-layer loop's job.
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
    r.error_msg = "GLM self-check not implemented (glm-5.2)";
    return r;
}

ArrowTokenizerDb &GlmV5Model::Tokenizer()
{
    return *tok_db_;
}

} // namespace lsxcommon