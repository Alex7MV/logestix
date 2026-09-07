// GLM-5.3-Flash loader suite. Dense roundtrip on the real converted model;
// expert store strictness; LRU slot bookkeeping without CUDA.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/glm_moe_flash_load.h"

using namespace lsxcommon;
using namespace lsxcommon::glm_moe_flash;

namespace fs = std::filesystem;

namespace {

std::string ModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_GLM_FLASH_ROOT"))
        return env;
    return "/mnt/llm_models/models/GLM-5.3-Flash";
}

bool ConvertedPresent()
{
    return fs::exists(ModelRoot() + "/logestix/model.index.json");
}

} // namespace

TEST_CASE("LoadGlmFlashDenseWeights: strict roundtrip")
{
    if (!ConvertedPresent()) SKIP("converted GLM-5.3-Flash not present");
    FlashPinnedWeights pw;
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    std::string error;
    REQUIRE(LoadGlmFlashDenseWeights(ModelRoot(), pw, stream, error));
    REQUIRE(error.empty());
    REQUIRE(pw.top.hidden == 4096);
    REQUIRE(pw.top.layers == 45);
    REQUIRE(pw.layers.size() == 45);
    // KDA layer 0: NVFP4 q (fp8 source requantized at conversion), conv
    // FP16, A_log + dt_bias F32.
    REQUIRE(pw.layers[0].is_kda);
    REQUIRE(pw.layers[0].kda_q.d_packed != nullptr);
    REQUIRE(pw.layers[0].kda_q.rows == 8192);
    REQUIRE(pw.layers[0].kda_q.cols == 4096);
    REQUIRE(pw.layers[0].d_kda_q_conv != nullptr);
    REQUIRE(pw.layers[0].d_A_log != nullptr);
    REQUIRE(pw.layers[0].d_dt_bias != nullptr);
    REQUIRE(pw.layers[0].d_o_norm != nullptr);
    // hc arrays per layer.
    REQUIRE(pw.layers[0].d_hc_attn_fn != nullptr);
    REQUIRE(pw.layers[0].hc_fn_rows == 24);
    REQUIRE(pw.layers[0].d_hc_attn_scale != nullptr);
    // DSA layer 3: NVFP4 q_a present, rows = q_lora.
    REQUIRE_FALSE(pw.layers[3].is_kda);
    REQUIRE(pw.layers[3].q_a.rows == 1536);
    REQUIRE(pw.layers[3].q_a.cols == 4096);
    REQUIRE(pw.layers[3].d_kv_b != nullptr);
    // Router on MoE layer 3 only.
    REQUIRE(pw.layers[3].is_moe);
    REQUIRE(pw.layers[3].d_router != nullptr);
    REQUIRE(pw.layers[3].d_router_bias != nullptr);
    REQUIRE_FALSE(pw.layers[0].is_moe);
    // Dense MLP only on layers 0..2.
    REQUIRE(pw.layers[0].mlp_gate.rows == 12288);
    REQUIRE_FALSE(pw.layers[3].mlp_gate.d_packed);
    // Shared expert NVFP4 on MoE layers.
    REQUIRE(pw.layers[3].shared_w1.rows == 2048);
    // Global: embed raw BF16 (vocab x hidden).
    REQUIRE(pw.embed_bytes ==
            static_cast<int64_t>(154880) * 4096 * 2);
    REQUIRE(pw.d_final_norm != nullptr);
    REQUIRE(pw.resident_bytes > 0);
    pw.Free();
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("LoadGlmFlashDenseWeights: missing dir is a hard error")
{
    FlashPinnedWeights pw;
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    std::string error;
    REQUIRE_FALSE(LoadGlmFlashDenseWeights("/nonexistent/model", pw, stream,
                                           error));
    REQUIRE_FALSE(error.empty());
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("LoadGlmFlashExpertStore: 43x288 pinned slabs")
{
    if (!ConvertedPresent()) SKIP("converted GLM-5.3-Flash not present");
    qwen_moe::QwenMoeExpertStore store;
    std::string error;
    REQUIRE(LoadGlmFlashExpertStore(ModelRoot(), store, error));
    REQUIRE(store.num_layers == 43);
    REQUIRE(store.num_experts == 288);
    REQUIRE(store.h_slabs.size() == 43);
    // Record sanity: every expert has non-zero w1/w2/w3 payloads and
    // matching global scales.
    REQUIRE(store.records[0][0].w1_bytes > 0);
    REQUIRE(store.records[42][287].w3_bytes > 0);
    REQUIRE(store.records[0][0].gs1 > 0.0f);
    REQUIRE(store.records[0][0].gs3 > 0.0f);
    store.Free();
}

TEST_CASE("AllocateQwenMoeExpertCache: LRU eviction bookkeeping")
{
    qwen_moe::QwenMoeExpertStore store;
    store.num_layers = 1;
    store.num_experts = 4;
    store.records.resize(1);
    store.records[0].resize(4);
    // Minimal synthetic slab: 4 records, 16B payloads each.
    uint8_t *slab = nullptr;
    REQUIRE(cudaMallocHost(reinterpret_cast<void **>(&slab), 64)
            == cudaSuccess);
    std::fill(slab, slab + 64, 0xAB);
    store.h_slabs.push_back(slab);
    store.slab_bytes.push_back(64);
    for (int e = 0; e < 4; ++e) {
        auto &r = store.records[0][e];
        r.w1_off = e * 16;
        r.w1_bytes = 16;
        r.w2_off = e * 16;
        r.w2_bytes = 0;
        r.w3_off = e * 16;
        r.w3_bytes = 0;
        r.s1_off = e * 16;
        r.s1_bytes = 0;
        r.s2_off = e * 16;
        r.s2_bytes = 0;
        r.s3_off = e * 16;
        r.s3_bytes = 0;
        r.gs1 = r.gs2 = r.gs3 = 1.0f;
    }
    gpu::ExpertDeviceCache cache;
    std::string error;
    REQUIRE(qwen_moe::AllocateQwenMoeExpertCache(store, 2, cache, error));
    REQUIRE(cache.slots == 2);
    // LRU: insert flat 0 and 1 -> touch 0 -> insert 2 evicts 1 (LRU).
    REQUIRE(qwen_moe::QwenMoeExpertCacheInsert(cache, 0) >= 0);
    REQUIRE(qwen_moe::QwenMoeExpertCacheInsert(cache, 1) >= 0);
    REQUIRE(qwen_moe::QwenMoeExpertCacheLookup(cache, 0) >= 0);
    REQUIRE(qwen_moe::QwenMoeExpertCacheInsert(cache, 2) >= 0);
    REQUIRE(qwen_moe::QwenMoeExpertCacheLookup(cache, 1) == -1);
    REQUIRE(qwen_moe::QwenMoeExpertCacheLookup(cache, 0) >= 0);
    qwen_moe::FreeQwenMoeExpertCache(cache);
    cudaFreeHost(slab);
}
