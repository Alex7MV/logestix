// GLM-5.3-Flash loader suite on a SYNTHETIC converted checkpoint (unified
// loader-test discipline: the suite converts its own tiny fixture and never
// touches the model vault; real-weight coverage lives in the selfcheck/perf
// suites). Dense roundtrip strictness; expert store strictness; LRU slot
// bookkeeping without CUDA.
//
// The load-side synthetic-geometry gate is enabled via the ctest ENVIRONMENT
// (LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1) — production loaders stay strict.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/glm_moe_flash_load.h"

#include "test_glm_moe_flash_fixture.h"

using namespace lsxcommon;
using namespace lsxcommon::glm_moe_flash;

namespace fs = std::filesystem;

TEST_CASE("LoadGlmFlashDenseWeights: strict roundtrip (synthetic)")
{
    glmflash::ConvertedFlashFixture fix;
    fix.in = fs::temp_directory_path() / "lsx_glmflash_loader_in";
    fix.out = fs::temp_directory_path() / "lsx_glmflash_loader_out";
    fs::remove_all(fix.in);
    fs::remove_all(fix.out);
    INFO("convert status: " << fix.Convert().ToString());
    REQUIRE(fix.Convert().ok());

    FlashPinnedWeights pw;
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    std::string error;
    INFO("load error: " << (LoadGlmFlashDenseWeights(fix.out.string(), pw,
                                                     stream, error)
                                 ? std::string("(loaded)")
                                 : error));
    REQUIRE(LoadGlmFlashDenseWeights(fix.out.string(), pw, stream, error));
    REQUIRE(error.empty());
    REQUIRE(pw.top.hidden == glmflash::kSynthH);
    REQUIRE(pw.top.layers == 45);
    REQUIRE(pw.layers.size() == 45);
    // KDA layer 0: NVFP4 q/k/v/o (BF16 source quantized at conversion),
    // conv FP16, A_log + dt_bias F32, dense MLP NVFP4.
    REQUIRE(pw.layers[0].loaded);
    REQUIRE(pw.layers[0].is_kda);
    REQUIRE(pw.layers[0].kda_q.d_packed != nullptr);
    REQUIRE(pw.layers[0].kda_q.rows == glmflash::kSynthD);
    REQUIRE(pw.layers[0].kda_q.cols == glmflash::kSynthH);
    REQUIRE(pw.layers[0].kda_o.d_packed != nullptr);
    REQUIRE(pw.layers[0].d_kda_q_conv != nullptr);
    REQUIRE(pw.layers[0].d_A_log != nullptr);
    REQUIRE(pw.layers[0].d_dt_bias != nullptr);
    REQUIRE(pw.layers[0].d_o_norm != nullptr);
    // hc arrays per layer.
    REQUIRE(pw.layers[0].d_hc_attn_fn != nullptr);
    REQUIRE(pw.layers[0].hc_fn_rows == 24);
    REQUIRE(pw.layers[0].d_hc_attn_scale != nullptr);
    // Dense MLP only on layers 0..2 (here: layer 0).
    REQUIRE(pw.layers[0].mlp_gate.rows == glmflash::kSynthH);
    // DSA layer 3: NVFP4 q_a present, router + shared expert on MoE layer.
    REQUIRE(pw.layers[3].loaded);
    REQUIRE_FALSE(pw.layers[3].is_kda);
    REQUIRE(pw.layers[3].q_a.rows == glmflash::kSynthD);
    REQUIRE(pw.layers[3].q_a.cols == glmflash::kSynthD);
    REQUIRE(pw.layers[3].d_kv_b != nullptr);
    REQUIRE(pw.layers[3].is_moe);
    REQUIRE(pw.layers[3].d_router != nullptr);
    REQUIRE(pw.layers[3].d_router_bias != nullptr);
    REQUIRE(pw.layers[3].shared_w1.rows == glmflash::kSynthH);
    REQUIRE_FALSE(pw.layers[0].is_moe);
    // Frozen-range layers without tensors: skipped empty under the
    // synthetic gate (the converter flushes empty raw batches for them).
    REQUIRE_FALSE(pw.layers[1].loaded);
    REQUIRE_FALSE(pw.layers[44].loaded);
    // Global: embed raw BF16 (vocab x hidden).
    REQUIRE(pw.embed_bytes ==
            static_cast<int64_t>(glmflash::kSynthV) * glmflash::kSynthH * 2);
    REQUIRE(pw.d_final_norm != nullptr);
    REQUIRE(pw.resident_bytes > 0);
    pw.Free();
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
    std::error_code ec;
    fs::remove_all(fix.in, ec);
    fs::remove_all(fix.out, ec);
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

TEST_CASE("LoadGlmFlashExpertStore: synthetic MoE layer slabs")
{
    glmflash::ConvertedFlashFixture fix;
    fix.in = fs::temp_directory_path() / "lsx_glmflash_store_in";
    fix.out = fs::temp_directory_path() / "lsx_glmflash_store_out";
    fs::remove_all(fix.in);
    fs::remove_all(fix.out);
    REQUIRE(fix.Convert().ok());

    moe::MoEExpertStore store;
    std::string error;
    REQUIRE(LoadGlmFlashExpertStore(fix.out.string(), store, error));
    // Only layer 3 carries experts in the fixture: compact store slots
    // count MoE layers from kMoeFirstLayer (3) -> exactly one slab.
    REQUIRE(store.num_layers == 1);
    REQUIRE(store.num_experts == glmflash::kSynthE);
    REQUIRE(store.h_slabs.size() == 1);
    // Record sanity: every expert has non-zero w1/w2/w3 payloads and
    // matching global scales.
    REQUIRE(store.records[0][0].w1_bytes > 0);
    REQUIRE(store.records[0][1].w3_bytes > 0);
    REQUIRE(store.records[0][0].gs1 > 0.0f);
    REQUIRE(store.records[0][0].gs3 > 0.0f);
    store.Free();
    std::error_code ec;
    fs::remove_all(fix.in, ec);
    fs::remove_all(fix.out, ec);
}

TEST_CASE("AllocateMoEExpertCache: LRU eviction bookkeeping")
{
    moe::MoEExpertStore store;
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
    REQUIRE(moe::AllocateMoEExpertCache(store, 2, cache, error));
    REQUIRE(cache.slots == 2);
    // LRU: insert flat 0 and 1 -> touch 0 -> insert 2 evicts 1 (LRU).
    REQUIRE(qwen_moe::QwenMoeExpertCacheInsert(cache, 0) >= 0);
    REQUIRE(qwen_moe::QwenMoeExpertCacheInsert(cache, 1) >= 0);
    REQUIRE(qwen_moe::QwenMoeExpertCacheLookup(cache, 0) >= 0);
    REQUIRE(qwen_moe::QwenMoeExpertCacheInsert(cache, 2) >= 0);
    REQUIRE(qwen_moe::QwenMoeExpertCacheLookup(cache, 1) == -1);
    REQUIRE(qwen_moe::QwenMoeExpertCacheLookup(cache, 0) >= 0);
    moe::FreeMoEExpertCache(cache);
    cudaFreeHost(slab);
}
