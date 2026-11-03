#include <catch2/catch_test_macros.hpp>
#include "test_qwen_moe_exp_fixture.h"
#include "lsxcommon/qwen_moe_exp_load.h"
#include "qwen_moe_exp_constants.h"

#include <cstddef>

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

namespace {

// Independent replica of the post-cache generation-context device footprint
// (arena + KV) used as the sizing invariant oracle. Mirrors
// qwen_moe_exp_inference.cu RunExpGeneration arena carving and
// KvCache::Allocate for the frozen qwen38 geometry.
std::size_t OracleGenContextBytes(int seq)
{
    using namespace lsxcommon::qwen_moe_exp::frozen;
    const int H = kHidden, HCH = kHcHidden, NH = kHeads, HD = kHeadDim;
    const int NKV = kKVHeads, rot = kRotaryDim;
    const int KD = kLinearNumKeyHeads * kLinearKeyHeadDim;         // 2048
    const int VD = kLinearNumValueHeads * kLinearValueHeadDim;     // 6144
    const int S = KD * 2 + VD;                                     // 10240
    const int nv = kLinearNumValueHeads, V = kVocab;
    const int I = kMoeIntermediate, K = kTopK, SI = kSharedIntermediate;

    std::size_t half_per_tok = 2 * (std::size_t)H + 3 * (std::size_t)HCH +
                               3 * (std::size_t)(NH * HD) +
                               2 * (std::size_t)(NH * HD * 2) +
                               2 * (std::size_t)(NKV * HD) +
                               2 * (std::size_t)rot +
                               2 * (std::size_t)S + 3 * (std::size_t)VD +
                               2 * (std::size_t)nv;
    std::size_t half_fixed = (std::size_t)K * 2 * I + (std::size_t)K * I +
                             (std::size_t)K * H + (std::size_t)I + I + H + SI + H;
    std::size_t float_per_tok = 2 * (std::size_t)V + 2 * (std::size_t)H +
                                16 + 2560 + 2 * (std::size_t)nv;
    std::size_t float_fixed = 1024;

    // 36 linear + 12 full layers (kFullAttnInterval = 4, full at i%4==3).
    constexpr int kFull = 12, kLin = 36;
    std::size_t kv_bytes =
        2 * (std::size_t)kFull * seq * NKV * HD * sizeof(__half) +
        (std::size_t)kLin * seq * S * sizeof(__half) +
        (std::size_t)kLin * nv * kLinearValueHeadDim * kLinearKeyHeadDim *
            kLinearNumKeyHeads * sizeof(float);

    return (half_per_tok * seq + half_fixed) * sizeof(__half) +
           (float_per_tok * seq + float_fixed) * sizeof(float) + 4096 +
           kv_bytes;
}

} // namespace

TEST_CASE("gen-context byte estimate covers arena+kv at 2k ctx",
          "[qwen-exp-loader]")
{
    const int seq = 2053;
    const std::size_t oracle = OracleGenContextBytes(seq);
    const std::size_t est = GenContextBytes(seq);
    // Must cover the oracle (under-reserve repeats the max-gen-2048 OOM) and
    // must not wildly overshoot (would starve the LRU).
    REQUIRE(est >= oracle);
    REQUIRE(est <= oracle + (256ull << 20));
}

TEST_CASE("cache slots sizing leaves room for the gen context",
          "[qwen-exp-loader]")
{
    const std::size_t per_expert = 2611200; // frozen geometry payload
    const std::size_t free_bytes = 26009ull << 20;
    const int total = 48 * 512;
    const int slots =
        SizeExpertCacheSlots(free_bytes, per_expert, total, 2053, 100);
    REQUIRE(slots > 0);
    REQUIRE(slots < total);
    // The OOM invariant: cache + post-cache context must fit the budget.
    REQUIRE(static_cast<std::size_t>(slots) * per_expert +
                GenContextBytes(2053) <=
            free_bytes);
    // Shorter contexts allow at least as many slots.
    REQUIRE(SizeExpertCacheSlots(free_bytes, per_expert, total, 261, 100) >=
            slots);
}

