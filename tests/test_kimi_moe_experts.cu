#include <catch2/catch_test_macros.hpp>

#include "kimi_moe_kernels.h"
#include "lsxcommon/kimi_moe_expert_cache.h"
#include "lsxcommon/kimi_moe_load.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <random>
#include <cstring>
#include <vector>

namespace {

using namespace lsxcommon::kimi;

constexpr int I = kKimiMoeIntermediate;
constexpr int Hd = kKimiHidden;

float Frand(int seed, int i)
{
    uint32_t v = static_cast<uint32_t>(seed * 2654435761u + i * 40503u);
    v ^= v >> 13;
    v *= 1274126177u;
    v ^= v >> 16;
    return (static_cast<float>(v % 20001) - 10000.0f) / 10000.0f;
}

// Canonical E4M3 block-scale decode (matches nvfp4_codec::E4M3ScaleToFloat).
float HostE4M3(uint8_t b)
{
    const uint8_t e4 = (b >> 3) & 0x0F;
    if (e4 == 0) return 0.0f;
    return std::ldexp(1.0f, static_cast<int>(e4) - 7);
}

// Encodes a power-of-two scale 2^e as an E4M3 block-scale byte.
uint8_t EncodeScalePow2(int e)
{
    return static_cast<uint8_t>(((e + 7) & 0xF) << 3);
}

// Builds one full-geometry projection: packed E2M1 nibbles + E4M3 scales.
struct ProjData {
    std::vector<uint8_t> packed;
    std::vector<uint8_t> scales;
    float global_scale;
};

ProjData MakeProj(int seed, int64_t rows, int64_t cols)
{
    ProjData p;
    const int64_t numel = rows * cols;
    p.packed.assign(numel / 2, 0);
    p.scales.assign(numel / 16, 0);
    p.global_scale = 0.75f;
    for (int64_t i = 0; i < numel; ++i) {
        // Nibble pattern biased to small magnitudes (valid E2M1 codes 0..15).
        const int nib = static_cast<int>(Frand(seed, i) * 8.0f) & 0xF;
        p.packed[i >> 1] |= static_cast<uint8_t>(nib << ((i & 1) ? 4 : 0));
        if ((i & 15) == 0)
            p.scales[i >> 4] =
                EncodeScalePow2(-3 + (static_cast<int>(Frand(seed + 1, i)) & 3));
        if ((i & 1) == 0)
            p.packed[i >> 1] = static_cast<uint8_t>(p.packed[i >> 1]);
    }
    // Fix packing: rebuild cleanly lo-nibble-first.
    std::fill(p.packed.begin(), p.packed.end(), 0);
    for (int64_t i = 0; i < numel; ++i) {
        const int nib = static_cast<int>(Frand(seed, i) * 8.0f) & 0xF;
        p.packed[i >> 1] |= static_cast<uint8_t>(nib << ((i & 1) ? 4 : 0));
    }
    return p;
}

} // namespace

TEST_CASE("Kimi cache allocation tolerates dense-layer zero entries",
          "[kimi][experts][cache]")
{
    // Kimi layer 0 is dense: no expert file exists, so db.entries[0] is a
    // zero-valued KimiExpertEntry. The uniformity check must compare only
    // the NON-zero (real MoE) entries; comparing against entry 0's zero
    // strides used to disable the whole cache ("non-uniform expert
    // geometry") and silently downgrade decode to the legacy staging path.
    std::vector<KimiExpertEntry> model(3);
    // entry 0: dense hole (all zero)
    // entries 1..2: real MoE experts with uniform full-geometry slabs.
    for (int i = 1; i < 3; ++i) {
        model[static_cast<size_t>(i)].w1_bytes = 7340032;
        model[static_cast<size_t>(i)].w3_bytes = 7340032;
        model[static_cast<size_t>(i)].w2_bytes = 7340032;
        model[static_cast<size_t>(i)].w1_scales_bytes = 917504;
        model[static_cast<size_t>(i)].w3_scales_bytes = 917504;
        model[static_cast<size_t>(i)].w2_scales_bytes = 917504;
    }
    lsxcommon::gpu::expert_store::ExpertStore<KimiExpertEntry> db;
    db.entries = model;
    db.num_layers = 3;
    db.experts_per_layer = 1;

    lsxcommon::kimi::KimiMoeExpertDeviceCache *cache = nullptr;
    std::string err;
    const cudaError_t e = lsxcommon::kimi::AllocateKimiMoeExpertCache(
        db, 2, cache, err);
    REQUIRE(e == cudaSuccess);
    REQUIRE(cache != nullptr);
    REQUIRE(cache->slots == 2);
    const int64_t kExpectedPack =
        3 * static_cast<int64_t>(7340032);   // 3 aligned-up 7340032 chunks
    REQUIRE(cache->pack_stride == kExpectedPack);
    REQUIRE(cache->scale_stride == 3 * static_cast<int64_t>(917504));
    lsxcommon::kimi::FreeKimiMoeExpertDeviceCache(cache);
}

TEST_CASE("Kimi cache allocation still rejects genuinely non-uniform slabs",
          "[kimi][experts][cache]")
{
    std::vector<KimiExpertEntry> model(3);
    for (int i = 1; i < 3; ++i) {
        model[static_cast<size_t>(i)].w1_bytes = 7340032;
        model[static_cast<size_t>(i)].w3_bytes = 7340032;
        model[static_cast<size_t>(i)].w2_bytes = 7340032;
        model[static_cast<size_t>(i)].w1_scales_bytes = 917504;
        model[static_cast<size_t>(i)].w3_scales_bytes = 917504;
        model[static_cast<size_t>(i)].w2_scales_bytes = 917504;
    }
    // entry 2: a real (non-zero) expert with DIFFERENT geometry — must
    // still be rejected.
    model[2].w3_bytes = 123456;

    lsxcommon::gpu::expert_store::ExpertStore<KimiExpertEntry> db;
    db.entries = model;
    db.num_layers = 3;
    db.experts_per_layer = 1;

    lsxcommon::kimi::KimiMoeExpertDeviceCache *cache = nullptr;
    std::string err;
    REQUIRE(lsxcommon::kimi::AllocateKimiMoeExpertCache(db, 2, cache, err) ==
            cudaErrorInvalidValue);
    REQUIRE(err == "non-uniform expert geometry; cache disabled");
    REQUIRE(cache == nullptr);
}

TEST_CASE("Kimi routed-expert FFN matches CPU reference", "[kimi][experts]")
{
    // Full-geometry projections (real Kimi MoE shapes).
    ProjData w1 = MakeProj(11, I, Hd);   // gate
    ProjData w3 = MakeProj(12, I, Hd);   // up
    ProjData w2 = MakeProj(13, Hd, I);   // down

    KimiExpertEntry entry{};
    entry.host_w1 = w1.packed.data();
    entry.host_w3 = w3.packed.data();
    entry.host_w2 = w2.packed.data();
    entry.host_w1_scales = w1.scales.data();
    entry.host_w3_scales = w3.scales.data();
    entry.host_w2_scales = w2.scales.data();
    entry.w1_global_scale = w1.global_scale;
    entry.w3_global_scale = w3.global_scale;
    entry.w2_global_scale = w2.global_scale;
    entry.w1_bytes = static_cast<int64_t>(w1.packed.size());
    entry.w3_bytes = static_cast<int64_t>(w3.packed.size());
    entry.w2_bytes = static_cast<int64_t>(w2.packed.size());
    entry.w1_scales_bytes = static_cast<int64_t>(w1.scales.size());
    entry.w3_scales_bytes = static_cast<int64_t>(w3.scales.size());
    entry.w2_scales_bytes = static_cast<int64_t>(w2.scales.size());

    // Input row.
    std::vector<float> xf(Hd);
    for (int i = 0; i < Hd; ++i) xf[i] = 0.5f * Frand(7, i);
    __nv_bfloat16 *d_x = nullptr;
    {
        std::vector<__nv_bfloat16> tmp(Hd);
        for (int i = 0; i < Hd; ++i) tmp[i] = __float2bfloat16_rn(xf[i]);
        REQUIRE(cudaMalloc(&d_x, Hd * 2) == cudaSuccess);
        REQUIRE(cudaMemcpy(d_x, tmp.data(), Hd * 2,
                           cudaMemcpyHostToDevice) == cudaSuccess);
    }
    __nv_bfloat16 *d_out = nullptr;
    REQUIRE(cudaMalloc(&d_out, Hd * 2) == cudaSuccess);
    REQUIRE(cudaMemset(d_out, 0, Hd * 2) == cudaSuccess);

    const size_t staging_bytes = KimiExpertFfnStagingBytes();
    void *d_staging = nullptr;
    REQUIRE(cudaMalloc(&d_staging, staging_bytes) == cudaSuccess);


    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    cudaError_t err = RunKimiExpertFfn(stream, entry, /*weight=*/1.0f,
                                       d_x, d_out, d_staging, staging_bytes);
    REQUIRE(err == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<__nv_bfloat16> out_h(Hd);
    REQUIRE(cudaMemcpy(out_h.data(), d_out, Hd * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    // ── CPU reference ────────────────────────────────────────────────────
    auto decode = [&](const ProjData &p, std::vector<double> &out) {
        out.resize(p.packed.size() * 2);
        static const float tbl[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
        for (size_t i = 0; i < out.size(); ++i) {
            uint8_t byte = p.packed[i >> 1];
            int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
            out[i] = tbl[nib] * HostE4M3(p.scales[i >> 4]) * p.global_scale;
        }
    };
    std::vector<double> dw1, dw3, dw2;
    decode(w1, dw1);
    decode(w3, dw3);
    decode(w2, dw2);

    std::vector<double> h_act(I);
    for (int k = 0; k < I; ++k) {
        double g = 0, u = 0;
        for (int j = 0; j < Hd; ++j) {
            double xv = __bfloat162float(__float2bfloat16_rn(xf[j]));
            g += dw1[static_cast<size_t>(k) * Hd + j] * xv;
            u += dw3[static_cast<size_t>(k) * Hd + j] * xv;
        }
        h_act[k] = g / (1.0 + std::exp(-g)) * u;
    }
    double max_abs = 0.0, ref_scale = 0.0;
    for (int o = 0; o < Hd; ++o) {
        double acc = 0.0;
        for (int k = 0; k < I; ++k)
            acc += h_act[k] * dw2[static_cast<size_t>(o) * I + k];
        double got = __bfloat162float(out_h[o]);
        ref_scale = std::max(ref_scale, std::abs(acc));
        max_abs = std::max(max_abs, std::abs(got - acc));
    }
    INFO("max abs error: " << max_abs << " ref scale: " << ref_scale);
    CHECK(max_abs <= 0.01 * ref_scale + 1.0);

    cudaDeviceSynchronize();
    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_staging);
    cudaStreamDestroy(stream);
}

TEST_CASE("Kimi cache slot upload/readback byte-compare with eviction",
          "[kimi][cache][slot-bytes]")
{
    using lsxcommon::gpu::expert_store::ExpertStore;
    constexpr int N = 6;
    constexpr int64_t kProj = 7340032;    // I*H/2 packed
    constexpr int64_t kScl = 917504;      // I*H/16
    ExpertStore<KimiExpertEntry> db;
    db.entries.resize(N);
    db.num_layers = 1;
    db.experts_per_layer = 1;
    std::vector<std::vector<uint8_t>> pack(N), scale(N);
    std::mt19937 rng(7);
    for (int i = 0; i < N; ++i) {
        pack[i].resize(3 * kProj);
        scale[i].resize(3 * kScl);
        for (auto &b : pack[i]) b = static_cast<uint8_t>(rng());
        for (auto &b : scale[i]) b = static_cast<uint8_t>(rng() % 128);
        auto &e = db.entries[static_cast<size_t>(i)];
        e.host_w1 = pack[i].data();
        e.host_w3 = pack[i].data() + kProj;
        e.host_w2 = pack[i].data() + 2 * kProj;
        e.host_w1_scales = scale[i].data();
        e.host_w3_scales = scale[i].data() + kScl;
        e.host_w2_scales = scale[i].data() + 2 * kScl;
        e.w1_bytes = e.w2_bytes = e.w3_bytes = kProj;
        e.w1_scales_bytes = e.w2_scales_bytes = e.w3_scales_bytes = kScl;
        e.w1_global_scale = e.w2_global_scale = e.w3_global_scale = 0.5f;
    }
    lsxcommon::kimi::KimiMoeExpertDeviceCache *cache = nullptr;
    std::string err;
    REQUIRE(lsxcommon::kimi::AllocateKimiMoeExpertCache(db, 2, cache, err) ==
            cudaSuccess);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    lsxcommon::kimi::KimiExpertCacheState state;
    state.device = cache;
    auto upload = [&](int expert, int slot) {
        KimiExpertEntry &e = db.entries[static_cast<size_t>(expert)];
        KimiMoeCacheBatch b{};
        b.missing_count = 1;
        b.missing_idx[0] = 0;   // batch index 0
        b.missing_slot[0] = slot;
        // routed_ids[0] = expert id; single layer -> layer=0, and the flat
        // entry is entries[0*epl + expert] = entries[expert].
        const int routed_ids[1] = {expert};
        const cudaError_t ue = lsxcommon::kimi::UploadKimiMoeExpertBatch(
            db, state, b, /*layer=*/0, 1, routed_ids);
        if (ue != cudaSuccess)
            FAIL("upload e=" << expert << " slot=" << slot << " err="
                             << cudaGetErrorString(ue));
        {
            KimiExpertEntry &ee = db.entries[static_cast<size_t>(expert)];
            KimiMoeSlotView vv =
                lsxcommon::kimi::KimiMoeSlotPointers(*cache, slot, ee);
            WARN("upload e=" << expert << " slot=" << slot
                             << " p_w1_off="
                             << (vv.p_w1 - cache->d_pack));
        }
        (void)slot; (void)st;
        (void)slot; (void)st;
    };
    auto check = [&](int expert, int slot) -> bool {
        cudaDeviceSynchronize();
        KimiExpertEntry &e = db.entries[static_cast<size_t>(expert)];
        KimiMoeSlotView v =
            lsxcommon::kimi::KimiMoeSlotPointers(*cache, slot, e);
        std::vector<uint8_t> got(static_cast<size_t>(kProj));
        cudaMemcpy(got.data(), v.p_w1, static_cast<size_t>(kProj),
                   cudaMemcpyDeviceToHost);
        if (std::memcmp(got.data(), e.host_w1,
                        static_cast<size_t>(kProj)) != 0) {
            int who = -1;
            for (int i = 0; i < N; ++i)
                if (std::memcmp(got.data(), pack[static_cast<size_t>(i)].data(),
                                64) == 0)
                    who = i;
            WARN("slot=" << slot << " expected e=" << expert
                         << " first16=" << (int)got[0] << "," << (int)got[1]
                         << " matches-e=" << who);
        }
        return std::memcmp(got.data(), e.host_w1,
                           static_cast<size_t>(kProj)) == 0;
    };
    upload(0, 0);
    upload(1, 1);
    CHECK(check(0, 0));
    CHECK(check(1, 1));
    upload(2, 0);   // evicts expert 0 bytes in slot 0
    upload(3, 1);   // evicts expert 1 bytes in slot 1
    CHECK(check(2, 0));
    CHECK(check(3, 1));
    upload(0, 0);   // re-upload expert 0
    CHECK(check(0, 0));
    CHECK(check(3, 1));   // slot 1 still expert 3's bytes
    lsxcommon::kimi::FreeKimiMoeExpertDeviceCache(cache);
    cudaStreamDestroy(st);
}

// Round-2 Task 2 red test: ResolveBatch fills missing_idx with BATCH
// indices; the upload must fetch db.entries[layer*epl + h_ids[missing_idx]]
// (routed expert), not entries[layer*epl + missing_idx] (slots 0..7).
// The old upload uploaded experts 0..7 regardless of routing.
TEST_CASE("Kimi cache upload maps batch indices to routed expert ids",
          "[kimi][cache][fence]")
{
    using lsxcommon::gpu::expert_store::ExpertStore;
    constexpr int EPL = 16;   // experts per layer; routed ids 10, 13 differ
    constexpr int N = EPL;
    constexpr int64_t kProj = 7340032;    // I*H/2 packed
    constexpr int64_t kScl = 917504;      // I*H/16
    ExpertStore<KimiExpertEntry> db;
    db.entries.resize(N);
    db.num_layers = 1;
    db.experts_per_layer = EPL;
    std::vector<std::vector<uint8_t>> pack(N), scale(N);
    std::mt19937 rng(11);
    for (int i = 0; i < N; ++i) {
        pack[i].resize(3 * kProj);
        scale[i].resize(3 * kScl);
        for (auto &b : pack[i]) b = static_cast<uint8_t>(rng());
        for (auto &b : scale[i]) b = static_cast<uint8_t>(rng() % 128);
        auto &e = db.entries[static_cast<size_t>(i)];
        e.host_w1 = pack[i].data();
        e.host_w3 = pack[i].data() + kProj;
        e.host_w2 = pack[i].data() + 2 * kProj;
        e.host_w1_scales = scale[i].data();
        e.host_w3_scales = scale[i].data() + kScl;
        e.host_w2_scales = scale[i].data() + 2 * kScl;
        e.w1_bytes = e.w2_bytes = e.w3_bytes = kProj;
        e.w1_scales_bytes = e.w2_scales_bytes = e.w3_scales_bytes = kScl;
        e.w1_global_scale = e.w2_global_scale = e.w3_global_scale = 0.5f;
    }
    lsxcommon::kimi::KimiMoeExpertDeviceCache *cache = nullptr;
    std::string err;
    // 8 slots: 6 unique routed experts must all fit (missing experts
    // beyond the slot count get missing_slot=-1 by design).
    REQUIRE(lsxcommon::kimi::AllocateKimiMoeExpertCache(db, 8, cache, err) ==
            cudaSuccess);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    lsxcommon::kimi::KimiExpertCacheState state;
    state.device = cache;

    // Routed batch: batch slots 0..7 -> expert ids {10, 13, 2, 15, 10, 5, 9, 13}.
    // Dup coverage: 10 twice, 13 twice. Deduped missing set: {10,13,2,15,5,9}.
    const int top_ids[8] = {10, 13, 2, 15, 10, 5, 9, 13};
    lsxcommon::kimi::KimiMoeExpertCachePlan plan;
    plan.Init(8, EPL);
    lsxcommon::kimi::KimiMoeCacheBatch b = lsxcommon::kimi::KimiMoeResolveBatch(
        plan, 0, EPL, top_ids, 8);
    REQUIRE_FALSE(b.invalid);
    REQUIRE(b.missing_count == 6);

    const cudaError_t ue = lsxcommon::kimi::UploadKimiMoeExpertBatch(
        db, state, b, /*layer=*/0, /*experts_per_layer=*/EPL, top_ids);
    REQUIRE(ue == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    // Every assigned slot must now hold the ROUTED expert's w1 bytes.
    auto check_slot = [&](int batch_i) {
        const int slot = b.resident_slot[batch_i] >= 0
                             ? b.resident_slot[batch_i]
                             : [&] {
                                 for (int m = 0; m < b.missing_count; ++m)
                                     if (b.missing_idx[m] == batch_i)
                                         return b.missing_slot[m];
                                 return -1;
                             }();
        REQUIRE(slot >= 0);
        const KimiExpertEntry &e =
            db.entries[static_cast<size_t>(top_ids[batch_i])];
        KimiMoeSlotView v = KimiMoeSlotPointers(*cache, slot, e);
        std::vector<uint8_t> got(static_cast<size_t>(kProj));
        REQUIRE(cudaMemcpy(got.data(), v.p_w1,
                           static_cast<size_t>(kProj),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        INFO("batch_i=" << batch_i << " expert=" << top_ids[batch_i]
                        << " slot=" << slot);
        CHECK(std::memcmp(got.data(), e.host_w1,
                          static_cast<size_t>(kProj)) == 0);
    };
    for (int i = 0; i < 8; ++i) check_slot(i);

    lsxcommon::kimi::FreeKimiMoeExpertDeviceCache(cache);
    cudaStreamDestroy(st);
}

// Round-2 differential: the SAME real expert through the staging FFN and
// the cached FFN must produce bit-identical outputs (same bytes, same
// dequant, same tail). Divergence pinpoints the cached-path defect.
TEST_CASE("Kimi cached FFN bit-identical to staging FFN (real expert file)",
          "[kimi][cache][diff][real-file]")
{
    const char *env = std::getenv("LSX_TEST_KIMI_MODEL_PATH");
    if (!env) SKIP("LSX_TEST_KIMI_MODEL_PATH not set");
    const std::string root = env;

    lsxcommon::kimi::KimiExpertDb db;
    db.num_layers = kKimiNumLayers;
    db.experts_per_layer = kKimiNumRouted;
    REQUIRE(db.Load(root, /*fallback=*/true));

    const int layer = 5, expert = 42;
    db.EnsureResident(layer, expert, 0);
    const KimiExpertEntry &e =
        db.entries[static_cast<size_t>(layer) * kKimiNumRouted + expert];
    REQUIRE(e.host_w1 != nullptr);

    constexpr int I = kKimiMoeIntermediate;
    constexpr int Hd = kKimiHidden;
    const int64_t kProj = static_cast<int64_t>(I) * Hd / 2;
    const int64_t kScl = static_cast<int64_t>(I) * Hd / 16;

    // Host input.
    std::vector<float> xf(Hd);
    for (int j = 0; j < Hd; ++j) xf[j] = Frand(3, j) * 0.5f;

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    __nv_bfloat16 *d_x = nullptr, *d_out_s = nullptr, *d_out_c = nullptr;
    REQUIRE(cudaMalloc(&d_x, Hd * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out_s, Hd * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out_c, Hd * 2) == cudaSuccess);
    std::vector<__nv_bfloat16> hx(Hd);
    for (int j = 0; j < Hd; ++j) hx[j] = __float2bfloat16_rn(xf[j]);
    REQUIRE(cudaMemcpy(d_x, hx.data(), Hd * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_out_s, 0, Hd * 2) == cudaSuccess);
    REQUIRE(cudaMemset(d_out_c, 0, Hd * 2) == cudaSuccess);

    // Staging scratch: full layout from RunKimiExpertFfn (w1|w3|w2 packed +
    // scales + 3 dequanted [I*Hd] bf16 + 3 [I] + [Hd]).
    const auto align_up_l = [](size_t v) { return (v + 255) / 256 * 256; };
    const size_t staging_bytes =
        align_up_l(3 * kProj + 3 * kScl) +
        3 * static_cast<size_t>(I) * Hd * 2 + 3 * I * 2 + Hd * 2;
    void *d_staging = nullptr;
    REQUIRE(cudaMalloc(&d_staging, staging_bytes) == cudaSuccess);

    // Cache slot with the same expert uploaded through the batch API.
    lsxcommon::kimi::KimiMoeExpertDeviceCache *cache = nullptr;
    std::string err;
    REQUIRE(lsxcommon::kimi::AllocateKimiMoeExpertCache(db, 1, cache, err) ==
            cudaSuccess);
    lsxcommon::kimi::KimiExpertCacheState state;
    state.device = cache;
    lsxcommon::kimi::KimiMoeCacheBatch b{};
    b.missing_count = 1;
    b.missing_idx[0] = 0;
    b.missing_slot[0] = 0;
    const int routed_ids[1] = {expert};
    REQUIRE(lsxcommon::kimi::UploadKimiMoeExpertBatch(
                db, state, b, layer, kKimiNumRouted, routed_ids) ==
            cudaSuccess);
    const KimiMoeSlotView v = KimiMoeSlotPointers(*cache, 0, e);

    const float coeff = 0.25f;
    REQUIRE(RunKimiExpertFfn(stream, e, coeff, d_x, d_out_s,
                             d_staging, staging_bytes) == cudaSuccess);
    REQUIRE(RunKimiExpertFfnCached(stream, e, coeff, d_x, d_out_c,
                                   d_staging, staging_bytes, v) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__nv_bfloat16> hs(Hd), hc(Hd);
    REQUIRE(cudaMemcpy(hs.data(), d_out_s, Hd * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(hc.data(), d_out_c, Hd * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    size_t mismatches = 0;
    int first_bad = -1;
    for (int j = 0; j < Hd; ++j) {
        if (hs[j] != hc[j]) {
            if (first_bad < 0) first_bad = j;
            ++mismatches;
        }
    }
    INFO("mismatches=" << mismatches << " first_bad=" << first_bad
                       << " staged=" << __bfloat162float(hs[first_bad < 0 ? 0 : first_bad])
                       << " cached=" << __bfloat162float(hc[first_bad < 0 ? 0 : first_bad]));
    CHECK(mismatches == 0);

    FreeKimiMoeExpertDeviceCache(cache);
    cudaFree(d_staging);
    cudaFree(d_x); cudaFree(d_out_s); cudaFree(d_out_c);
    cudaStreamDestroy(stream);
}
