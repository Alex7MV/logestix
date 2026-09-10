#include <catch2/catch_test_macros.hpp>
#include "test_qwen_moe_exp_fixture.h"
#include "lsxcommon/qwen_moe_exp_load.h"

#include <filesystem>

using namespace lsxcommon::qwen_moe_exp;
namespace fs = std::filesystem;

namespace {

struct LoaderEnv {
    fs::path dir = fs::temp_directory_path() / "lsx_qwen_exp_loader";
    fs::path out = fs::temp_directory_path() / "lsx_qwen_exp_loader_out";
    LoaderEnv()
    {
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::remove_all(out, ec);
        qwen_moe_exp_test::ExpSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
        qwen_moe_exp_test::BuildSyntheticExpCheckpoint(dir);
        auto st = lsxcommon::qwen_moe_exp::ConvertQwen38FlashNextModel(
            dir.string(), out.string());
        REQUIRE(st.ok());
        qwen_moe_exp_test::ExpUnsetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    }
};

} // namespace

TEST_CASE("expert store loads synthetic 8-expert slabs", "[qwen-exp-loader]")
{
    LoaderEnv env;
    ExpExpertStore store;
    std::string err;
    REQUIRE(LoadQwen4ExpExpertStore(env.out.string(), store, err));
    REQUIRE(err.empty());
    REQUIRE(store.num_layers == qwen_moe_exp_test::ExpCfg().num_layers);
    REQUIRE(store.num_experts == qwen_moe_exp_test::ExpCfg().num_experts);
    REQUIRE(store.h_slabs.size() == static_cast<size_t>(store.num_layers));
    // Per-expert payload present: w1/w2/w3 + scales offsets sane.
    const auto &rec = store.records[0][0];
    REQUIRE(rec.w1_bytes > 0);
    REQUIRE(rec.w2_bytes > 0);
    REQUIRE(rec.w3_bytes > 0);
    REQUIRE(rec.s1_bytes > 0);
    REQUIRE(rec.s2_bytes > 0);
    REQUIRE(rec.s3_bytes > 0);
    REQUIRE(rec.gs1 > 0.f);
    REQUIRE(rec.gs2 > 0.f);
    REQUIRE(rec.gs3 > 0.f);
    store.Free();
}

TEST_CASE("LRU cache bookkeeping flat index across 512-expert space",
          "[qwen-exp-loader]")
{
    // Host-only: gpu::ExpertDeviceCache bookkeeping without CUDA buffers.
    // num_layers x num_experts flat index; slot budget 16 (real model has
    // 48x512 = 24576 flats; synthetic 4x8 = 32).
    lsxcommon::gpu::ExpertDeviceCache cache;
    const int flats = 4 * 8;
    const int slots = 16;
    int32_t slot_of[flats];
    int32_t owner[slots];
    int32_t lru[slots];
    for (int i = 0; i < flats; ++i) slot_of[i] = -1;
    for (int i = 0; i < slots; ++i) { owner[i] = -1; lru[i] = i; }
    cache.slots = slots;
    cache.total_experts = flats;
    cache.h_slot_of = slot_of;
    cache.h_slot_owner = owner;
    cache.h_lru = lru;

    for (int i = 0; i < slots; ++i) REQUIRE(cache.Insert(i) == i);
    REQUIRE(cache.Lookup(3) == 3);
    cache.Insert(slots);                    // evicts LRU tail = expert 0
    REQUIRE(cache.Lookup(0) == -1);
    REQUIRE(cache.Lookup(slots) >= 0);
    // Hits: Lookup(3) + Lookup(slots); misses: slots+1 inserts.
    REQUIRE(cache.hits == 2);
    REQUIRE(cache.misses == slots + 1);
}

TEST_CASE("dense weights upload + PLE meta computed", "[qwen-exp-loader]")
{
    LoaderEnv env;
    ExpPinnedWeights pw;
    cudaStream_t stream = nullptr;
    std::string err;
    {
        const bool ok = LoadQwen4ExpDenseWeights(env.out.string(), pw, stream,
                                                 err);
        if (!ok) FAIL("LoadQwen4ExpDenseWeights failed: " + err);
        REQUIRE(ok);
    }
    if (!err.empty()) FAIL(err);
    const auto &c = qwen_moe_exp_test::ExpCfg();
    // Globals.
    REQUIRE(pw.embed.d_packed != nullptr);
    REQUIRE(pw.d_head_f16 != nullptr);  // lm_head: BF16 raw -> fp16 table
    REQUIRE(pw.d_final_norm != nullptr);
    REQUIRE(pw.d_mixer_norm != nullptr);
    REQUIRE(pw.mixer_down.d_packed != nullptr);
    REQUIRE(pw.mixer_up.d_packed != nullptr);
    // Per-layer: hc norm + router + shared.
    REQUIRE(pw.layers.size() == static_cast<size_t>(c.num_layers));
    REQUIRE(pw.layers[0].d_attn_hc_norm != nullptr);
    REQUIRE(pw.layers[0].d_mlp_hc_norm != nullptr);
    REQUIRE(pw.layers[0].attn_hc_down.d_packed != nullptr);
    REQUIRE(pw.layers[0].attn_hc_up.d_packed != nullptr);
    REQUIRE(pw.layers[0].attn_hc_inject.d_packed != nullptr);
    REQUIRE(pw.layers[0].d_router != nullptr);
    REQUIRE(pw.layers[0].shared_w1.d_packed != nullptr);
    REQUIRE(pw.layers[0].d_shared_gate != nullptr);
    // PLE meta: multipliers recomputed + verified, prime table present.
    REQUIRE(pw.ple_meta.head_vocab_sizes[0] > 0);
    REQUIRE(pw.ple_meta.head_offsets[0] == 0);
    int64_t run = 0;
    for (int h = 0; h < pw.ple_meta.num_heads; ++h)
        run += pw.ple_meta.head_vocab_sizes[h];
    REQUIRE(pw.ple_meta.head_offsets[pw.ple_meta.num_heads - 1] ==
            run - pw.ple_meta.head_vocab_sizes[pw.ple_meta.num_heads - 1]);
    pw.Free();
    REQUIRE(pw.layers.empty());
}

TEST_CASE("PLE shards mmap non-empty", "[qwen-exp-loader]")
{
    LoaderEnv env;
    ExpPinnedWeights pw;
    cudaStream_t stream = nullptr;
    std::string err;
    {
        const bool ok = LoadQwen4ExpDenseWeights(env.out.string(), pw, stream,
                                                 err);
        if (!ok) FAIL("LoadQwen4ExpDenseWeights failed: " + err);
        REQUIRE(ok);
    }
    const int parts = qwen_moe_exp_test::ExpCfg().split_ngram_parts;
    std::vector<ExpPleShardView> shards(static_cast<size_t>(parts));
    REQUIRE(LoadQwen4ExpPleShards(env.out.string(), pw, shards.data(), err));
    if (!err.empty()) FAIL(err);
    for (int s = 0; s < parts; ++s) {
        REQUIRE(shards[static_cast<size_t>(s)].data != nullptr);
        REQUIRE(shards[static_cast<size_t>(s)].bytes > 0);
        REQUIRE(shards[static_cast<size_t>(s)].shard == s);
    }
    pw.Free();
}

TEST_CASE("strict errors on missing layer files", "[qwen-exp-loader]")
{
    LoaderEnv env;
    // Corrupt: remove one dense layer file.
    auto victim = env.out / "logestix" / "pinned" / "layer_2.arrow";
    std::error_code ec;
    fs::remove(victim, ec);
    ExpPinnedWeights pw;
    cudaStream_t stream = nullptr;
    std::string err;
    REQUIRE_FALSE(LoadQwen4ExpDenseWeights(env.out.string(), pw, stream, err));
    REQUIRE_FALSE(err.empty());
    pw.Free();
}

TEST_CASE("strict errors on missing expert file", "[qwen-exp-loader]")
{
    LoaderEnv env;
    auto victim = env.out / "logestix" / "experts" / "layer_1" / "exp_5.arrow";
    std::error_code ec;
    fs::remove(victim, ec);
    ExpExpertStore store;
    std::string err;
    REQUIRE_FALSE(LoadQwen4ExpExpertStore(env.out.string(), store, err));
    REQUIRE_FALSE(err.empty());
    store.Free();
}

TEST_CASE("expert cache alloc sizes to largest payload", "[qwen-exp-loader]")
{
    LoaderEnv env;
    ExpExpertStore store;
    std::string err;
    REQUIRE(LoadQwen4ExpExpertStore(env.out.string(), store, err));
    lsxcommon::gpu::ExpertDeviceCache cache{};
    REQUIRE(AllocateQwen4ExpExpertCache(store, 4, cache, err));
    REQUIRE(cache.slots == 4);
    REQUIRE(cache.total_experts == store.num_layers * store.num_experts);
    REQUIRE(cache.d_pack != nullptr);
    REQUIRE(cache.d_scale != nullptr);
    REQUIRE(cache.pack_stride > 0);
    FreeQwen4ExpExpertCache(cache);
    store.Free();
}
