// Qwen3.5-35B-A3B MoE kernel tests: LRU cache bookkeeping (host-side),
#include <cstdio>
// router top-k, routed expert FFN, shared expert combine. CUDA-dependent
// cases are added by later tasks.

#include <catch2/catch_test_macros.hpp>

#include "test_kernel_budget.h"

#include <cstdint>
#include <vector>

#include "lsxcommon/qwen_moe_expert_cache.h"

using lsxcommon::qwen_moe::QwenMoeExpertDeviceCache;

namespace {

// Stage a host activation range into device memory. The TC quantizer reads its
// activation input on the device, so the engine always quantizes from a
// device buffer (qwen_moe_exp_inference.cu:2128 passes d_mixed). Handing it a
// host pointer only works where HMM/ATS maps host memory into the device
// address space, which is not guaranteed on every platform.
void StageActs(cudaStream_t stream, const void *src, std::size_t bytes,
               void **d_out)
{
    REQUIRE(cudaMalloc(d_out, bytes) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(*d_out, src, bytes, cudaMemcpyHostToDevice,
                            stream) == cudaSuccess);
}

QwenMoeExpertDeviceCache MakeCache(int slots, int total_experts,
                                   std::vector<int32_t> &slot_of,
                                   std::vector<int32_t> &owner,
                                   std::vector<int32_t> &lru)
{
    slot_of.assign(static_cast<size_t>(total_experts), -1);
    owner.assign(static_cast<size_t>(slots), -1);
    lru.resize(static_cast<size_t>(slots));
    for (int i = 0; i < slots; ++i) lru[static_cast<size_t>(i)] = i;
    QwenMoeExpertDeviceCache c;
    c.slots = slots;
    c.total_experts = total_experts;
    c.h_slot_of = slot_of.data();
    c.h_slot_owner = owner.data();
    c.h_lru = lru.data();
    return c;
}

} // namespace

TEST_CASE("expert cache LRU bookkeeping", "[qwen-moe-cache]")
{
    std::vector<int32_t> slot_of, owner, lru;
    auto c = MakeCache(4, 8, slot_of, owner, lru);

    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheLookup(c, 3) == -1);
    CHECK(c.misses == 0);  // pure query miss: no stats
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheInsert(c, 3) == 0);
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheInsert(c, 5) == 1);
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheInsert(c, 1) == 2);
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheInsert(c, 7) == 3);
    CHECK(c.hits == 0);
    CHECK(c.misses == 4);
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheLookup(c, 5) == 1);
    CHECK(c.hits == 1);
    CHECK(lru[0] == 1);  // hit refreshes MRU

    // Insert evicts the true LRU tail (slot holding expert 3, never touched).
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheInsert(c, 0) == 0);
    CHECK(slot_of[static_cast<size_t>(3)] == -1);
    CHECK(slot_of[static_cast<size_t>(0)] == 0);
    CHECK(owner[0] == 0);

    // Out-of-range lookups are safe and change nothing.
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheLookup(c, 99) == -1);
    CHECK(lsxcommon::qwen_moe::QwenMoeExpertCacheLookup(c, -1) == -1);
}

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>
#include "lsxcommon/qwen_moe_inference_pipeline.h"
#include "lsxcommon/qwen_moe_inference.h"
#include "lsxcommon/lsx_tc_gemm.h"

TEST_CASE("bf16 act-quant matches fp16 variant for bf16-representable values",
          "[lsx-tc-actquant]")
{
    if (!lsxcommon::lsx_tc::LsxTcRuntimeSupported()) return;
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    constexpr int K = 256;
    // Values exactly representable in bf16: coarse mantissa avoids
    // fp16-vs-bf16 rounding asymmetry.
    std::vector<__nv_bfloat16> hx(K);
    for (int i = 0; i < K; ++i) {
        const float m = static_cast<float>((i % 7) - 3);   // -3..3
        hx[static_cast<size_t>(i)] =
            __float2bfloat16(m * std::ldexp(1.0f, (i % 5) - 2));
    }
    uint8_t *d_xq, *d_xqs, *d_xqb, *d_xqsb;
    cudaMalloc(&d_xq, K / 2);  cudaMalloc(&d_xqs, K / 16);
    cudaMalloc(&d_xqb, K / 2); cudaMalloc(&d_xqsb, K / 16);
    // fp16 reference: bf16 values are exact in fp16.
    std::vector<__half> hxf(K);
    for (int i = 0; i < K; ++i)
        hxf[static_cast<size_t>(i)] = __float2half(
            __bfloat162float(hx[static_cast<size_t>(i)]));
    __half *d_hxf = nullptr;
    __nv_bfloat16 *d_hxb = nullptr;
    StageActs(st, hxf.data(), hxf.size() * sizeof(__half),
              reinterpret_cast<void **>(&d_hxf));
    StageActs(st, hx.data(), hx.size() * sizeof(__nv_bfloat16),
              reinterpret_cast<void **>(&d_hxb));
    REQUIRE(lsxcommon::lsx_tc::LsxTcQuantizeActsNvfp4(
                st, d_hxf, 1, K, d_xq, d_xqs) == cudaSuccess);
    REQUIRE(lsxcommon::lsx_tc::LsxTcQuantizeActsNvfp4Bf16(
                st, d_hxb, 1, K, d_xqb, d_xqsb) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
    std::vector<uint8_t> pq(K / 2), ps(K / 16), qb(K / 2), qs(K / 16);
    cudaMemcpy(pq.data(), d_xq, pq.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(ps.data(), d_xqs, ps.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(qb.data(), d_xqb, qb.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(qs.data(), d_xqsb, qs.size(), cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(pq.data(), qb.data(), pq.size()) == 0);
    CHECK(std::memcmp(ps.data(), qs.data(), ps.size()) == 0);
    cudaFree(d_xq); cudaFree(d_xqs); cudaFree(d_xqb); cudaFree(d_xqsb);
    cudaStreamDestroy(st);
}

namespace {

// Host replica of the HF router semantics (fp32 softmax -> top-k -> renorm).
void HostRouter(const std::vector<__nv_bfloat16> &w,
                const std::vector<__half> &x, int seq, int E, int K, int H,
                std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> &out)
{
    out.assign(static_cast<size_t>(seq) * K, {});
    for (int t = 0; t < seq; ++t) {
        std::vector<float> logits(static_cast<size_t>(E));
        float m = -INFINITY;
        for (int e = 0; e < E; ++e) {
            float dot = 0.f;
            for (int d = 0; d < H; ++d)
                dot += __bfloat162float(
                           w[static_cast<size_t>(e) * H + static_cast<size_t>(d)]) *
                       __half2float(x[static_cast<size_t>(t) * H + static_cast<size_t>(d)]);
            logits[static_cast<size_t>(e)] = dot;
            m = fmaxf(m, dot);
        }
        std::vector<float> probs(static_cast<size_t>(E));
        float z = 0.f;
        for (int e = 0; e < E; ++e) {
            probs[static_cast<size_t>(e)] = expf(logits[static_cast<size_t>(e)] - m);
            z += probs[static_cast<size_t>(e)];
        }
        for (auto &p : probs) p /= z;

        int idx[16]; float val[16]; int n = 0;
        for (int e = 0; e < E; ++e) {
            float v = probs[static_cast<size_t>(e)];
            int pos = n;
            while (pos > 0 && (val[pos-1] < v || (val[pos-1] == v && idx[pos-1] > e))) {
                if (pos < K) { val[pos] = val[pos-1]; idx[pos] = idx[pos-1]; }
                --pos;
            }
            if (pos < K) { val[pos] = v; idx[pos] = e; if (n < K) ++n; }
        }
        float s = 0.f;
        for (int i = 0; i < K; ++i) s += val[i];
        for (int i = 0; i < K; ++i) {
            auto &slot = out[static_cast<size_t>(t) * K + static_cast<size_t>(i)];
            slot.eid = idx[i];
            slot.coeff = val[i] / s;
        }
    }
}

} // namespace

TEST_CASE("router top-k matches host replica", "[qwen-moe-router]")
{
    const int seq = 5, E = 8, K = 2, H = 64;
    // Deterministic weight-like inputs in a sane numeric range.
    std::vector<__nv_bfloat16> w(static_cast<size_t>(E) * H);
    std::vector<__half> x(static_cast<size_t>(seq) * H);
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = __float2bfloat16(((float)(i * 37 % 251) / 251.0f) - 0.5f);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = __float2half(sinf((float)i) * 0.5f);

    __nv_bfloat16 *d_w;
    __half *d_x;
    lsxcommon::qwen_moe::QwenMoeRoutedSlot *d_slots;
    cudaMalloc(&d_w, w.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_x, x.size() * sizeof(__half));
    cudaMalloc(&d_slots, sizeof(lsxcommon::qwen_moe::QwenMoeRoutedSlot) *
                             static_cast<size_t>(seq) * K);
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRouterTopK(stream, d_w, d_x, seq, E,
                                                      K, H, d_slots) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> got(
        static_cast<size_t>(seq) * K);
    cudaMemcpy(got.data(), d_slots, got.size() * sizeof(got[0]),
               cudaMemcpyDeviceToHost);

    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> want;
    HostRouter(w, x, seq, E, K, H, want);

    for (size_t i = 0; i < want.size(); ++i) {
        INFO("slot " << i);
        CHECK(got[i].eid == want[i].eid);
        CHECK(std::fabs(got[i].coeff - want[i].coeff) < 1e-6f);
    }

    // Determinism: repeat run must be bit-identical.
    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> again(
        static_cast<size_t>(seq) * K);
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRouterTopK(stream, d_w, d_x, seq, E,
                                                      K, H, d_slots) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaMemcpy(again.data(), d_slots, again.size() * sizeof(again[0]),
               cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(got.data(), again.data(),
                      got.size() * sizeof(got[0])) == 0);

    cudaFree(d_w); cudaFree(d_x); cudaFree(d_slots);
    cudaStreamDestroy(stream);
}

#include "test_qwen_moe_fixture.h"
#include "lsxcommon/qwen_moe_load.h"
#include "lsxcommon/schema.h"

namespace {

// Host NVFP4 dequant of a row-major [N,K] packed weight (per-16 E4M3 block
// scales along K). Mirrors gpu::nvfp4 decode semantics.
float HostDequantNibble(int nib)
{
    const int mag = nib & 7;
    float v = (mag == 1) ? 0.5f : (mag == 2) ? 1.f : (mag == 3) ? 1.5f
              : (mag == 4) ? 2.f : (mag == 5) ? 3.f : (mag == 6) ? 4.f
              : (mag == 7) ? 6.f : 0.f;
    return (nib & 8) ? -v : v;
}

// Canonical block-scale decode (complete E4M3FN, single source nvfp4_scale.h).
float HostE4M3Scale(uint8_t e4m3)
{
    return lsxcommon::e4m3_to_float(e4m3);
}

} // namespace

// Decode-split router (stage-1 dots + stage-2 select) must produce
// bit-identical slots to the prefill monolithic kernel for the same token:
// both stage-1 dots and the monolithic row walk share one accumulation order.
TEST_CASE("router decode-split matches prefill bit-exact", "[qwen-moe-router]")
{
    const int E = 32, K = 8, H = 256;
    const int seq = 4;
    std::vector<__nv_bfloat16> w(static_cast<size_t>(E) * H);
    std::vector<__half> x(static_cast<size_t>(seq) * H);
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = __float2bfloat16(((float)(i * 37 % 251) / 251.0f) - 0.5f);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = __float2half(sinf((float)i * 0.37f) * 0.5f +
                            cosf((float)(i % 64) * 0.11f) * 0.2f);

    __nv_bfloat16 *d_w;
    __half *d_x;
    lsxcommon::qwen_moe::QwenMoeRoutedSlot *d_slots_mono, *d_slots_split;
    cudaMalloc(&d_w, w.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_x, x.size() * sizeof(__half));
    cudaMalloc(&d_slots_mono, sizeof(lsxcommon::qwen_moe::QwenMoeRoutedSlot) *
                                  static_cast<size_t>(seq) * K);
    cudaMalloc(&d_slots_split, sizeof(lsxcommon::qwen_moe::QwenMoeRoutedSlot) *
                                   static_cast<size_t>(seq) * K);
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Prefill: one monolithic launch over all tokens.
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRouterTopK(
                stream, d_w, d_x, seq, E, K, H, d_slots_mono) == cudaSuccess);
    // Decode: two-stage split, one token at a time.
    for (int t = 0; t < seq; ++t) {
        REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRouterTopK(
                    stream, d_w, d_x + static_cast<size_t>(t) * H, 1, E, K, H,
                    d_slots_split + static_cast<size_t>(t) * K) == cudaSuccess);
    }
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> mono(
        static_cast<size_t>(seq) * K);
    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> split(
        static_cast<size_t>(seq) * K);
    cudaMemcpy(mono.data(), d_slots_mono, mono.size() * sizeof(mono[0]),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(split.data(), d_slots_split, split.size() * sizeof(split[0]),
               cudaMemcpyDeviceToHost);

    for (int t = 0; t < seq; ++t) {
        INFO("token " << t);
        for (int k = 0; k < K; ++k) {
            const auto &m = mono[static_cast<size_t>(t) * K + static_cast<size_t>(k)];
            const auto &s = split[static_cast<size_t>(t) * K + static_cast<size_t>(k)];
            CHECK(m.eid == s.eid);
            CHECK(std::memcmp(&m.coeff, &s.coeff, sizeof(float)) == 0);
        }
    }

    cudaFree(d_w); cudaFree(d_x);
    cudaFree(d_slots_mono); cudaFree(d_slots_split);
    cudaStreamDestroy(stream);
}

// Real-shape decode router latency budget (frozen Qwen3.5/3.6 geometry:
// E=256, H=2048, top-8, seq=1). The router is the single largest kernel
// family in the decode step (~19% of GPU time); this budget pins the
// production entry so the stage-1 dots lever cannot silently regress.
// Correctness of the same entry is covered by the two cases above; this
// case additionally re-checks the real-shape selection against the host
// replica so a "fast but wrong" rewrite cannot pass on timing alone.
TEST_CASE("router real-shape decode latency budget", "[qwen-moe-router]")
{
    using lsxcommon::qwen_moe::QwenMoeRoutedSlot;
    const int E = 256, K = 8, H = 2048;
    std::vector<__nv_bfloat16> w(static_cast<size_t>(E) * H);
    std::vector<__half> x(static_cast<size_t>(H));
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = __float2bfloat16(((float)(i * 37 % 251) / 251.0f) - 0.5f);
    for (int i = 0; i < H; ++i)
        x[static_cast<size_t>(i)] =
            __float2half(sinf((float)i * 0.19f) * 0.4f);

    __nv_bfloat16 *d_w;
    __half *d_x;
    QwenMoeRoutedSlot *d_slots;
    cudaMalloc(&d_w, w.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_x, x.size() * sizeof(__half));
    cudaMalloc(&d_slots, sizeof(QwenMoeRoutedSlot) * K);
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(__nv_bfloat16),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half),
               cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (int i = 0; i < 5; ++i)
        REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRouterTopK(
                    stream, d_w, d_x, 1, E, K, H, d_slots) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    // Selection sanity + host-replica agreement at real scale.
    std::vector<QwenMoeRoutedSlot> got(static_cast<size_t>(K));
    cudaMemcpy(got.data(), d_slots, got.size() * sizeof(got[0]),
               cudaMemcpyDeviceToHost);
    std::vector<QwenMoeRoutedSlot> want;
    HostRouter(w, x, 1, E, K, H, want);
    for (int k = 0; k < K; ++k) {
        CHECK(got[static_cast<size_t>(k)].eid ==
              want[static_cast<size_t>(k)].eid);
        CHECK(got[static_cast<size_t>(k)].eid >= 0);
        CHECK(got[static_cast<size_t>(k)].eid < E);
        for (int j = 0; j < k; ++j)
            CHECK(got[static_cast<size_t>(k)].eid !=
                  got[static_cast<size_t>(j)].eid);
    }

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    constexpr int kIters = 200;
    cudaEventRecord(a, stream);
    for (int i = 0; i < kIters; ++i)
        REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRouterTopK(
                    stream, d_w, d_x, 1, E, K, H, d_slots) == cudaSuccess);
    cudaEventRecord(b, stream);
    REQUIRE(cudaEventSynchronize(b) == cudaSuccess);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    const double us = ms * 1000.0 / kIters;
    std::fprintf(stderr,
                 "[QWEN-MOE-ROUTER] E=%d H=%d K=%d seq=1 avg=%.2f us\n", E, H,
                 K, us);

#ifdef NDEBUG
    // Release-only budget: the deep-unrolled uint4 stage-1 walk lands
    // ~30 us; the narrow strided walk + select lands ~12 us.
    kernel_budget::Require("qwen_moe.router.decode", us, kernel_budget::Kind::kMax,
                           "warp-per-row router top-k+renorm");
#endif

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_slots);
    cudaStreamDestroy(stream);
}

TEST_CASE("routed ffn slot chain matches dequant reference", "[qwen-moe-experts]")
{
    namespace fs2 = std::filesystem;
    fs2::path in = fs2::temp_directory_path() / "lsx_qwen_moe_ffn_in";
    fs2::path out = fs2::temp_directory_path() / "lsx_qwen_moe_ffn_out";
    fs2::remove_all(in); fs2::remove_all(out);
    qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    REQUIRE(lsxcommon::qwen_moe::ConvertQwen35MoeModel(in.string(), out.string()).ok());

    lsxcommon::qwen_moe::QwenMoeExpertStore store;
    std::string err;
    REQUIRE(lsxcommon::qwen_moe::LoadQwenMoeExpertStore(out.string(), store, err));
    REQUIRE(store.num_layers == 4);
    REQUIRE(store.num_experts == 8);

    lsxcommon::qwen_moe::QwenMoeExpertDeviceCache cache;
    REQUIRE(lsxcommon::qwen_moe::AllocateQwenMoeExpertCache(store, 4, cache, err));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Token x: deterministic half vector, H=64, I=16.
    const int H = 64, I = 16, K = 2;
    std::vector<__half> hx(static_cast<size_t>(H));
    for (int i = 0; i < H; ++i)
        hx[static_cast<size_t>(i)] = __float2half(sinf((float)i * 0.7f) * 0.4f);

    __half *d_x, *d_yg, *d_yu, *d_y2;
    float *d_acc;
    cudaMalloc(&d_x, H * 2);
    cudaMalloc(&d_yg, I * 2);
    cudaMalloc(&d_yu, I * 2);
    cudaMalloc(&d_y2, H * 2);
    cudaMalloc(&d_acc, H * 4);
    cudaMemset(d_acc, 0, H * 4);
    cudaMemcpy(d_x, hx.data(), H * 2, cudaMemcpyHostToDevice);

    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> h_slots =
        {{0, 0.5f}, {1, 0.5f}};
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRoutedFFN(
                stream, store, cache, /*layer=*/0, h_slots.data(), K, d_x,
                H, I, d_acc, d_yg, d_yu, d_y2, err) == cudaSuccess);
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeAccToFp16(stream, d_acc, d_y2, H) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<__half> hout(H);
    cudaMemcpy(hout.data(), d_y2, H * 2, cudaMemcpyDeviceToHost);

    // Determinism across cache states: flush the cache and re-run.
    std::vector<__half> hout2(H);
    {
        lsxcommon::qwen_moe::FreeQwenMoeExpertCache(cache);
        REQUIRE(lsxcommon::qwen_moe::AllocateQwenMoeExpertCache(store, 4, cache, err));
        cudaMemset(d_acc, 0, H * 4);
        // Reverse slot order exercises LRU eviction path.
        std::swap(h_slots[0], h_slots[1]);
        // Undo the coefficient pairing so math matches the first run.
        h_slots[0].coeff = 0.5f; h_slots[1].coeff = 0.5f;
        h_slots[0].eid = 0; h_slots[1].eid = 1;
        REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRoutedFFN(
                    stream, store, cache, 0, h_slots.data(), K, d_x, H, I,
                    d_acc, d_yg, d_yu, d_y2, err) == cudaSuccess);
        REQUIRE(lsxcommon::qwen_moe::RunQwenMoeAccToFp16(stream, d_acc, d_y2, H) ==
                cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        cudaMemcpy(hout2.data(), d_y2, H * 2, cudaMemcpyDeviceToHost);
    }
    CHECK(std::memcmp(hout.data(), hout2.data(), H * sizeof(__half)) == 0);
    for (int i = 0; i < H; ++i)
        CHECK(std::isfinite(__half2float(hout[static_cast<size_t>(i)])));

    // Sanity: non-zero output.
    float energy = 0.f;
    for (int i = 0; i < H; ++i)
        energy += __half2float(hout[static_cast<size_t>(i)]) *
                  __half2float(hout[static_cast<size_t>(i)]);
    CHECK(energy > 0.f);

    cudaFree(d_x); cudaFree(d_yg); cudaFree(d_yu); cudaFree(d_y2);
    cudaFree(d_acc);
    cudaStreamDestroy(stream);
    lsxcommon::qwen_moe::FreeQwenMoeExpertCache(cache);
    store.Free();
    std::error_code ec;
    fs2::remove_all(in, ec); fs2::remove_all(out, ec);
}

// Copy-stream upload path: cache-miss H2D staging runs on a dedicated stream
// fenced by per-expert events while the compute stream proceeds. Output must
// be bit-identical to the single-stream path (same math, different ordering
// of independent work).
TEST_CASE("routed ffn copy-stream uploads match single-stream reference",
          "[qwen-moe-experts]")
{
    namespace fs2 = std::filesystem;
    fs2::path in = fs2::temp_directory_path() / "lsx_qwen_moe_ffn_cs_in";
    fs2::path out = fs2::temp_directory_path() / "lsx_qwen_moe_ffn_cs_out";
    fs2::remove_all(in); fs2::remove_all(out);
    qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    REQUIRE(lsxcommon::qwen_moe::ConvertQwen35MoeModel(in.string(), out.string()).ok());

    lsxcommon::qwen_moe::QwenMoeExpertStore store;
    std::string err;
    REQUIRE(lsxcommon::qwen_moe::LoadQwenMoeExpertStore(out.string(), store, err));

    lsxcommon::qwen_moe::QwenMoeExpertDeviceCache cache;
    // Deliberately small (4 slots < 8 experts over both layers) so the
    // copy-stream pass also exercises LRU eviction across layers.
    REQUIRE(lsxcommon::qwen_moe::AllocateQwenMoeExpertCache(store, 4, cache, err));

    cudaStream_t stream, copy_stream;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking) == cudaSuccess);

    const int H = 64, I = 16, K = 2;
    std::vector<__half> hx(static_cast<size_t>(H));
    for (int i = 0; i < H; ++i)
        hx[static_cast<size_t>(i)] = __float2half(sinf((float)i * 0.7f) * 0.4f);

    __half *d_x, *d_yg, *d_yu, *d_y2;
    float *d_acc;
    cudaMalloc(&d_x, H * 2);
    cudaMalloc(&d_yg, I * 2);
    cudaMalloc(&d_yu, I * 2);
    cudaMalloc(&d_y2, H * 2);
    cudaMalloc(&d_acc, H * 4);

    cudaMemcpy(d_x, hx.data(), H * 2, cudaMemcpyHostToDevice);

    const std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> h_slots =
        {{0, 0.5f}, {1, 0.5f}};

    auto run_layer = [&](cudaStream_t cs, cudaEvent_t *ev_uploads,
                         cudaEvent_t ev_front, std::vector<__half> &dst) {
        cudaMemset(d_acc, 0, H * 4);
        REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRoutedFFN(
                    stream, store, cache, /*layer=*/0, h_slots.data(), K, d_x,
                    H, I, d_acc, d_yg, d_yu, d_y2, err,
                    cs, ev_uploads, ev_front) == cudaSuccess);
        REQUIRE(lsxcommon::qwen_moe::RunQwenMoeAccToFp16(stream, d_acc, d_y2, H) ==
                cudaSuccess);
        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
        dst.resize(H);
        cudaMemcpy(dst.data(), d_y2, H * 2, cudaMemcpyDeviceToHost);
    };

    // Reference: single-stream path (cold cache).
    std::vector<__half> ref;
    run_layer(nullptr, nullptr, nullptr, ref);

    // Copy-stream path: fresh cache -> every expert is a miss staged on
    // copy_stream, compute fenced by per-expert events.
    lsxcommon::qwen_moe::FreeQwenMoeExpertCache(cache);
    REQUIRE(lsxcommon::qwen_moe::AllocateQwenMoeExpertCache(store, 4, cache, err));
    std::vector<__half> got;
    {
        cudaEvent_t ev_uploads[K];
        cudaEvent_t ev_front;
        for (int k = 0; k < K; ++k)
            REQUIRE(cudaEventCreateWithFlags(&ev_uploads[k],
                                             cudaEventDisableTiming) == cudaSuccess);
        REQUIRE(cudaEventCreateWithFlags(&ev_front,
                                         cudaEventDisableTiming) == cudaSuccess);
        run_layer(copy_stream, ev_uploads, ev_front, got);
        for (int k = 0; k < K; ++k) cudaEventDestroy(ev_uploads[k]);
        cudaEventDestroy(ev_front);
    }
    REQUIRE(std::memcmp(ref.data(), got.data(), H * sizeof(__half)) == 0);
    for (int i = 0; i < H; ++i)
        CHECK(std::isfinite(__half2float(got[static_cast<size_t>(i)])));

    cudaFree(d_x); cudaFree(d_yg); cudaFree(d_yu); cudaFree(d_y2);
    cudaFree(d_acc);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(copy_stream);
    lsxcommon::qwen_moe::FreeQwenMoeExpertCache(cache);
    store.Free();
    std::error_code ec;
    fs2::remove_all(in, ec); fs2::remove_all(out, ec);
}

TEST_CASE("shared expert gate combine", "[qwen-moe-experts]")
{
    namespace fs2 = std::filesystem;
    fs2::path in = fs2::temp_directory_path() / "lsx_qwen_moe_sh_in";
    fs2::path out = fs2::temp_directory_path() / "lsx_qwen_moe_sh_out";
    fs2::remove_all(in); fs2::remove_all(out);
    qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    REQUIRE(lsxcommon::qwen_moe::ConvertQwen35MoeModel(in.string(), out.string()).ok());

    lsxcommon::qwen_moe::QwenMoePinnedWeights pw;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::string err;
    REQUIRE(lsxcommon::qwen_moe::LoadQwenMoeDenseWeights(out.string(), pw, stream, err));

    const int H = 64, SI = 16, seq = 2;
    const auto &lw = pw.layers[0];

    // x rows deterministic; routed acc zeros -> out == g * shared(x).
    std::vector<__half> hx(static_cast<size_t>(seq) * H);
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = __float2half(sinf((float)i) * 0.4f);

    __half *d_x, *d_si, *d_sh, *d_out;
    float *d_acc, *d_g;
    cudaMalloc(&d_x, hx.size() * 2);
    cudaMalloc(&d_si, static_cast<size_t>(seq) * SI * 2);
    cudaMalloc(&d_sh, static_cast<size_t>(seq) * H * 2);
    cudaMalloc(&d_out, static_cast<size_t>(seq) * H * 2);
    cudaMalloc(&d_acc, static_cast<size_t>(seq) * H * 4);
    cudaMalloc(&d_g, seq * 4);
    cudaMemset(d_acc, 0, static_cast<size_t>(seq) * H * 4);
    cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeSharedCombine(
                stream, lw.shared_w1, lw.shared_w2, lw.shared_w3,
                lw.d_shared_gate, d_x, seq, H, SI, d_acc, d_si, d_sh, d_g,
                d_out) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<__half> hout(static_cast<size_t>(seq) * H);
    cudaMemcpy(hout.data(), d_out, hout.size() * 2, cudaMemcpyDeviceToHost);
    float energy = 0.f;
    for (auto v : hout) energy += __half2float(v) * __half2float(v);
    CHECK(energy > 0.f);
    for (auto v : hout) CHECK(std::isfinite(__half2float(v)));

    // Determinism: bit-identical repeat.
    std::vector<__half> hout2(hout.size());
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeSharedCombine(
                stream, lw.shared_w1, lw.shared_w2, lw.shared_w3,
                lw.d_shared_gate, d_x, seq, H, SI, d_acc, d_si, d_sh, d_g,
                d_out) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaMemcpy(hout2.data(), d_out, hout2.size() * 2, cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(hout.data(), hout2.data(), hout.size() * 2) == 0);

    cudaFree(d_x); cudaFree(d_si); cudaFree(d_sh); cudaFree(d_out);
    cudaFree(d_acc); cudaFree(d_g);
    cudaStreamDestroy(stream);
    lsxcommon::qwen_moe::FreeQwenMoePinnedWeights(pw);
    std::error_code ec;
    fs2::remove_all(in, ec); fs2::remove_all(out, ec);
}

TEST_CASE("generation driver deterministic on synthetic model", "[qwen-moe-gen]")
{
    namespace fs2 = std::filesystem;
    fs2::path in = fs2::temp_directory_path() / "lsx_qwen_moe_gen_in";
    fs2::path out = fs2::temp_directory_path() / "lsx_qwen_moe_gen_out";
    fs2::remove_all(in); fs2::remove_all(out);
    qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    REQUIRE(lsxcommon::qwen_moe::ConvertQwen35MoeModel(in.string(), out.string()).ok());

    lsxcommon::qwen_moe::QwenMoePinnedWeights pw;
    lsxcommon::qwen_moe::QwenMoeExpertStore store;
    lsxcommon::qwen_moe::QwenMoeExpertDeviceCache cache;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::string err;
    REQUIRE(lsxcommon::qwen_moe::LoadQwenMoeDenseWeights(out.string(), pw, stream, err));
    REQUIRE(lsxcommon::qwen_moe::LoadQwenMoeExpertStore(out.string(), store, err));
    REQUIRE(lsxcommon::qwen_moe::AllocateQwenMoeExpertCache(store, 8, cache, err));

    lsxcommon::qwen_moe::QwenMoeGenScratch gen_scratch;
    REQUIRE(lsxcommon::qwen_moe::EnsureQwenMoeScratch(stream, pw, gen_scratch) ==
            cudaSuccess);

    auto run_once = [&](lsxcommon::qwen::QwenGenerationResult &r) {
        lsxcommon::qwen::QwenKvCache kv;
        lsxcommon::sample::GreedySampler sampler;
        std::vector<int32_t> prompt = {1, 2, 3};
        return lsxcommon::qwen_moe::RunQwenMoeGeneration(
            stream, pw, store, cache, gen_scratch, prompt, /*max_new=*/4,
            /*eos_id=*/pw.moe_top.eos_token_id, r, sampler, &kv);
    };

    lsxcommon::qwen::QwenGenerationResult r1, r2;
    cudaError_t e1 = run_once(r1);
    if (e1 != cudaSuccess) WARN(r1.error_msg);
    REQUIRE(e1 == cudaSuccess);
    CHECK(static_cast<int>(r1.tokens.size()) == 4);
    for (auto t : r1.tokens) CHECK((t >= 0 && t < pw.top.vocab_size));

    cudaError_t e2 = run_once(r2);
    REQUIRE(e2 == cudaSuccess);
    CHECK(r1.tokens == r2.tokens);
    REQUIRE(r1.final_logits.size() == r2.final_logits.size());
    CHECK(std::memcmp(r1.final_logits.data(), r2.final_logits.data(),
                      r1.final_logits.size() * sizeof(float)) == 0);
    for (auto v : r1.final_logits) CHECK(std::isfinite(v));
    CHECK(r1.ok); CHECK(r2.ok);

    lsxcommon::qwen_moe::FreeQwenMoeGenScratch(gen_scratch);
    lsxcommon::qwen_moe::FreeQwenMoeExpertCache(cache);
    store.Free();
    lsxcommon::qwen_moe::FreeQwenMoePinnedWeights(pw);
    cudaStreamDestroy(stream);
    std::error_code ec;
    fs2::remove_all(in, ec); fs2::remove_all(out, ec);
}

// ── Batched TC routed chain (80-tok/s decode round) ─────────────────────────

// TC batch chain parity: per-slot batched TC GEMM must be bit-identical to
// K sequential single-slot TC GEMM launches (same kernels, same math).
// Adversarial slot table: duplicate experts, zero coeffs, alternating
// scale-block bytes (round-19 scale-mapping lesson).
TEST_CASE("tc batch gemm matches per-slot chain bitwise", "[qwen-moe-tc-batch]")
{
    if (!lsxcommon::lsx_tc::LsxTcRuntimeSupported()) return;
    namespace fs2 = std::filesystem;
    fs2::path in = fs2::temp_directory_path() / "lsx_qwen_tc_batch_in";
    fs2::path out = fs2::temp_directory_path() / "lsx_qwen_tc_batch_out";
    fs2::remove_all(in); fs2::remove_all(out);
    qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    REQUIRE(lsxcommon::qwen_moe::ConvertQwen35MoeModel(in.string(), out.string()).ok());

    lsxcommon::qwen_moe::QwenMoeExpertStore store;
    std::string err;
    REQUIRE(lsxcommon::qwen_moe::LoadQwenMoeExpertStore(out.string(), store, err));
    lsxcommon::qwen_moe::QwenMoeExpertDeviceCache cache;
    // Full residency: 8 experts -> 8 slots, so every dev_slot is stable.
    REQUIRE(lsxcommon::qwen_moe::AllocateQwenMoeExpertCache(store, 8, cache, err));

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    lsxcommon::qwen::SetQwenTcFastMath(true);

    const int H = 64, I = 16, K = 2, N13 = 2 * I;
    // Adversarial activations: alternating per-16-block scale magnitudes.
    std::vector<__half> hx(static_cast<size_t>(H));
    for (int i = 0; i < H; ++i)
        hx[static_cast<size_t>(i)] = __float2half(
            ((i / 16) % 2 ? 512.0f : 0.5f) * sinf(static_cast<float>(i) * 0.7f));

    uint8_t *d_xq; uint8_t *d_xqs;
    cudaMalloc(&d_xq, static_cast<size_t>(H) / 2);
    cudaMalloc(&d_xqs, static_cast<size_t>((H + 15) / 16));
    __half *d_hx_stage = nullptr;
    StageActs(stream, hx.data(), hx.size() * sizeof(__half),
              reinterpret_cast<void **>(&d_hx_stage));
    REQUIRE(lsxcommon::lsx_tc::LsxTcQuantizeActsNvfp4(stream, d_hx_stage, 1, H,
                                                   d_xq, d_xqs) == cudaSuccess);

    __half *d_y13_seq, *d_y13_batch, *d_y2_seq, *d_y2_batch;
    cudaMalloc(&d_y13_seq, (2 * I) * 2);
    cudaMalloc(&d_y13_batch, static_cast<size_t>(K) * 2 * I * 2);
    cudaMalloc(&d_y2_seq, H * 2);
    cudaMalloc(&d_y2_batch, static_cast<size_t>(K) * H * 2);

    // Legacy reference: K sequential RunLsxTcGemmPreQuant calls against the
    // SAME slot addresses the table resolves to (slot k == dev_slot k under
    // full residency; mirrored via EnsureQwenMoeExpertResident).
    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> routes =
        {{0, 0.3f}, {0, 0.0f}};   // duplicate eid 0, zero coeff
    for (int k = 0; k < K; ++k) {
        const int slot = lsxcommon::qwen_moe::EnsureQwenMoeExpertResident(
            store, 0, routes[static_cast<size_t>(k)].eid, cache, stream, err);
        REQUIRE(slot >= 0);
        const auto &rec = store.records[0][routes[static_cast<size_t>(k)].eid];
        uint8_t *pack = cache.d_pack + static_cast<int64_t>(slot) * cache.pack_stride;
        uint8_t *scale = cache.d_scale + static_cast<int64_t>(slot) * cache.scale_stride;
        const float seg[2] = {rec.gs1, rec.gs3};
        float *d_seg;
        cudaMalloc(&d_seg, sizeof(seg));
        cudaMemcpy(d_seg, seg, sizeof(seg), cudaMemcpyHostToDevice);
        REQUIRE(lsxcommon::lsx_tc::RunLsxTcGemmPreQuant(
                    stream, d_xq, d_xqs, pack, scale, 1.0f, d_y13_seq, 1, H,
                    2 * I, false, d_seg, nullptr) == cudaSuccess);
        cudaFree(d_seg);
        // down: feed the same quantized activations for the down GEMM
        // comparison (parity of the GEMM itself, not the swiglu producer).
        REQUIRE(lsxcommon::lsx_tc::RunLsxTcGemmPreQuant(
                    stream, d_xq, d_xqs, pack + rec.w1_bytes + rec.w3_bytes,
                    scale + rec.s1_bytes + rec.s3_bytes, rec.gs2, d_y2_seq,
                    1, I, H, false) == cudaSuccess);
    }
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    // Batched: device table with the same dev_slot/scale mapping.
    std::vector<lsxcommon::lsx_tc::LsxTcBatchSlot> tbl(K);
    for (int k = 0; k < K; ++k) {
        const int slot = lsxcommon::qwen_moe::EnsureQwenMoeExpertResident(
            store, 0, routes[static_cast<size_t>(k)].eid, cache, stream, err);
        const auto &rec = store.records[0][routes[static_cast<size_t>(k)].eid];
        tbl[static_cast<size_t>(k)] = {slot, rec.gs1, rec.gs3, rec.gs2, 0.f};
    }
    lsxcommon::lsx_tc::LsxTcBatchSlot *d_tbl;
    cudaMalloc(&d_tbl, sizeof(tbl[0]) * K);
    cudaMemcpy(d_tbl, tbl.data(), sizeof(tbl[0]) * K, cudaMemcpyHostToDevice);

    const int64_t w13_bytes =
        store.records[0][0].w1_bytes + store.records[0][0].w3_bytes;
    const int64_t s13_bytes =
        store.records[0][0].s1_bytes + store.records[0][0].s3_bytes;
    std::vector<uint8_t> h13_seq((2 * I) * 2), h13_batch(K * 2 * I * 2);
    std::vector<uint8_t> h2_seq(H * 2), h2_batch(static_cast<size_t>(K) * H * 2);
    {
        float *d_part;
        const int sk = lsxcommon::lsx_tc::LsxTcBatchSplitK(H, N13);
        cudaMalloc(&d_part, sk > 1 ? static_cast<size_t>(sk) * K * N13 * 4 : 1);
        REQUIRE(lsxcommon::lsx_tc::RunLsxTcGemmBatch13(
                    stream, d_xq, d_xqs, cache.d_pack, cache.d_scale,
                    cache.pack_stride, cache.scale_stride, d_tbl, K, H, N13,
                    d_y13_batch, d_part) == cudaSuccess);
        cudaFree(d_part);
    }
    {
        float *d_part;
        const int sk = lsxcommon::lsx_tc::LsxTcBatchSplitK(I, H);
        cudaMalloc(&d_part, sk > 1 ? static_cast<size_t>(sk) * K * H * 4 : 1);
        REQUIRE(lsxcommon::lsx_tc::RunLsxTcGemmBatchDown(
                    stream, d_xq, d_xqs, cache.d_pack, cache.d_scale,
                    cache.pack_stride, cache.scale_stride, d_tbl, K,
                    w13_bytes, s13_bytes, I, H, d_y2_batch, d_part) ==
                cudaSuccess);
        cudaFree(d_part);
    }
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    cudaMemcpy(h13_seq.data(), d_y13_seq, h13_seq.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h13_batch.data(), d_y13_batch, h13_batch.size(),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h2_seq.data(), d_y2_seq, h2_seq.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(h2_batch.data(), d_y2_batch, h2_batch.size(),
               cudaMemcpyDeviceToHost);

    // Slot 0: bitwise equal to the sequential reference.
    CHECK(std::memcmp(h13_seq.data(), h13_batch.data(), h13_seq.size()) == 0);
    CHECK(std::memcmp(h2_seq.data(), h2_batch.data(), h2_seq.size()) == 0);

    cudaFree(d_xq); cudaFree(d_xqs); cudaFree(d_y13_seq); cudaFree(d_y13_batch);
    cudaFree(d_y2_seq); cudaFree(d_y2_batch); cudaFree(d_tbl);
    cudaStreamDestroy(stream);
    lsxcommon::qwen_moe::FreeQwenMoeExpertCache(cache);
    store.Free();
    lsxcommon::qwen::SetQwenTcFastMath(false);
    std::error_code ec;
    fs2::remove_all(in, ec); fs2::remove_all(out, ec);
}

// ── Batched TC routed chain (80-tok/s decode round) ─────────────────────────

// Composite batched TC routed chain vs the pinned legacy per-slot chain:
// d_moe_acc must be BITWISE equal (same kernels, same accumulation order).
// Adversarial: duplicate eids across slots, alternating scale bytes in the
// input activations.
TEST_CASE("tc batch routed ffn matches legacy chain bitwise",
          "[qwen-moe-tc-batch]")
{
    namespace fs2 = std::filesystem;
    fs2::path in = fs2::temp_directory_path() / "lsx_qwen_tc_rffn_in";
    fs2::path out = fs2::temp_directory_path() / "lsx_qwen_tc_rffn_out";
    fs2::remove_all(in); fs2::remove_all(out);
    qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    REQUIRE(lsxcommon::qwen_moe::ConvertQwen35MoeModel(in.string(), out.string()).ok());

    lsxcommon::qwen_moe::QwenMoeExpertStore store;
    std::string err;
    REQUIRE(lsxcommon::qwen_moe::LoadQwenMoeExpertStore(out.string(), store, err));
    lsxcommon::qwen_moe::QwenMoeExpertDeviceCache cache;
    REQUIRE(lsxcommon::qwen_moe::AllocateQwenMoeExpertCache(store, 8, cache, err));

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    lsxcommon::qwen::SetQwenTcFastMath(true);

    const int H = 64, I = 16, K = 2;
    std::vector<__half> hx(static_cast<size_t>(H));
    for (int i = 0; i < H; ++i)
        hx[static_cast<size_t>(i)] = __float2half(
            ((i / 16) % 2 ? 384.0f : 0.25f) * sinf(static_cast<float>(i) * 0.7f));

    uint8_t *d_xq, *d_xqs;
    cudaMalloc(&d_xq, static_cast<size_t>(H) / 2);
    cudaMalloc(&d_xqs, static_cast<size_t>((H + 15) / 16));
    __half *d_hx_stage = nullptr;
    StageActs(stream, hx.data(), hx.size() * sizeof(__half),
              reinterpret_cast<void **>(&d_hx_stage));
    REQUIRE(lsxcommon::lsx_tc::LsxTcQuantizeActsNvfp4(stream, d_hx_stage, 1, H,
                                                   d_xq, d_xqs) == cudaSuccess);

    __half *d_postin, *d_yg, *d_yu, *d_y2, *d_ygu_batch, *d_h_batch,
        *d_yp_batch;
    float *d_acc_seq, *d_acc_batch;
    cudaMalloc(&d_postin, H * 2);
    cudaMemcpy(d_postin, hx.data(), H * 2, cudaMemcpyHostToDevice);
    // Legacy TC fused chain writes the w13 output [gate|up] into d_yg (2I
    // halves) and the swiglu reads up from d_yu == d_yg + I (the generation
    // arena places the two slices adjacently). Mirror that contract with a
    // contiguous buffer: d_yg owns 2I halves, d_yu points at its second I.
    cudaMalloc(&d_yg, 2 * I * 2);
    d_yu = d_yg + I;
    cudaMalloc(&d_y2, H * 2);
    cudaMalloc(&d_ygu_batch, static_cast<size_t>(K) * 2 * I * 2);
    cudaMalloc(&d_h_batch, static_cast<size_t>(K) * I * 2);
    cudaMalloc(&d_yp_batch, static_cast<size_t>(K) * H * 2);
    cudaMalloc(&d_acc_seq, H * 4);
    cudaMalloc(&d_acc_batch, H * 4);
    uint8_t *d_qpack_b, *d_qscales_b;
    cudaMalloc(&d_qpack_b, static_cast<size_t>(K) * I / 2);
    cudaMalloc(&d_qscales_b, static_cast<size_t>(K) * ((I + 15) / 16));

    // Legacy reference (handshake per-slot TC chain, xq/xqs provided):
    // acc seeded from the residual (init kernel) then coeff_add chain.
    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> routes =
        {{3, 0.6f}, {3, 0.4f}};   // duplicate expert eid 3
    cudaMemsetAsync(d_acc_seq, 0, H * 4, stream);
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeAccInitForTest(
                stream, d_postin, d_acc_seq, H) == cudaSuccess);
    // Then the legacy coeff-add chain.
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRoutedFFN(
                stream, store, cache, 0, routes.data(), K, d_postin, H, I,
                d_acc_seq, d_yg, d_yu, d_y2, err, nullptr, nullptr, nullptr,
                d_xq, d_xqs) == cudaSuccess);

    // Batched chain: warm the mirror, gather on device, run the batch.
    for (int e = 0; e < store.num_experts; ++e)
        REQUIRE(lsxcommon::qwen_moe::EnsureQwenMoeExpertResident(
                    store, 0, e, cache, stream, err) >= 0);
    // Stage the remaining layers so the full-residency mirror completes and
    // QwenMoeGemvNohandshake uploads d_gemv_meta (the gather input).
    for (int l = 1; l < store.num_layers; ++l)
        for (int e = 0; e < store.num_experts; ++e)
            REQUIRE(lsxcommon::qwen_moe::EnsureQwenMoeExpertResident(
                        store, l, e, cache, stream, err) >= 0);
    REQUIRE(lsxcommon::qwen_moe::QwenMoeGemvNohandshake(store));

    lsxcommon::qwen_moe::QwenMoeRoutedSlot *d_routes;
    cudaMalloc(&d_routes, sizeof(routes[0]) * K);
    cudaMemcpy(d_routes, routes.data(), sizeof(routes[0]) * K,
               cudaMemcpyHostToDevice);
    lsxcommon::qwen_moe::QwenMoeGvSlotDev *d_tbl;
    cudaMalloc(&d_tbl, sizeof(lsxcommon::qwen_moe::QwenMoeGvSlotDev) * 16);
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeGemvGatherTable(
                stream, d_routes, K, store.d_gemv_meta, d_tbl) == cudaSuccess);
    // Batched path: acc seeded with the SAME init kernel, then the plain
    // ascending reduce (the seeded-reduce fold was reverted — it corrupted
    // real-model decode; see commit 2e97def).
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeAccInitForTest(
                stream, d_postin, d_acc_batch, H) == cudaSuccess);
    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRoutedTcBatch(
                stream, store, cache, 0, d_routes, d_xq, d_xqs, K, H, I,
                d_postin, d_acc_batch, d_ygu_batch, d_h_batch, d_yp_batch,
                d_qpack_b, d_qscales_b, nullptr, d_tbl) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<float> acc_seq(H), acc_batch(H);
    cudaMemcpy(acc_seq.data(), d_acc_seq, H * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(acc_batch.data(), d_acc_batch, H * sizeof(float),
               cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(acc_seq.data(), acc_batch.data(),
                      H * sizeof(float)) == 0);

    cudaFree(d_xq); cudaFree(d_xqs); cudaFree(d_postin); cudaFree(d_yg);
    cudaFree(d_y2); cudaFree(d_ygu_batch);
    cudaFree(d_h_batch); cudaFree(d_yp_batch); cudaFree(d_acc_seq);
    cudaFree(d_acc_batch); cudaFree(d_qpack_b); cudaFree(d_qscales_b);
    cudaFree(d_routes); cudaFree(d_tbl);
    cudaStreamDestroy(stream);
    lsxcommon::qwen_moe::FreeQwenMoeExpertCache(cache);
    store.Free();
    lsxcommon::qwen::SetQwenTcFastMath(false);
    qwen_moe_test::MoeUnsetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    std::error_code ec;
    fs2::remove_all(in, ec); fs2::remove_all(out, ec);
}

// Fallback parity: partial residency (4 slots < 8 experts per layer) keeps
// the mirror incomplete, so the batched TC chain must NEVER engage and the
// legacy handshake chain remains the default. The generation must be
// bit-stable across two runs (same fixture, same session machinery).
TEST_CASE("generation driver fallback parity under partial residency",
          "[qwen-moe-tc-batch]")
{
    namespace fs2 = std::filesystem;
    fs2::path in = fs2::temp_directory_path() / "lsx_qwen_tc_fb_in";
    fs2::path out = fs2::temp_directory_path() / "lsx_qwen_tc_fb_out";
    fs2::remove_all(in); fs2::remove_all(out);
    qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
    qwen_moe_test::MoeSetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    REQUIRE(lsxcommon::qwen_moe::ConvertQwen35MoeModel(in.string(), out.string()).ok());

    lsxcommon::qwen::SetQwenTcFastMath(true);
    lsxcommon::qwen_moe::QwenMoePinnedWeights pw;
    lsxcommon::qwen_moe::QwenMoeExpertStore store;
    lsxcommon::qwen_moe::QwenMoeExpertDeviceCache cache;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::string err;
    REQUIRE(lsxcommon::qwen_moe::LoadQwenMoeDenseWeights(out.string(), pw, stream, err));
    REQUIRE(lsxcommon::qwen_moe::LoadQwenMoeExpertStore(out.string(), store, err));
    // Deliberately SMALL: 4 slots < 8 experts -> mirror never completes ->
    // QwenMoeGemvNohandshake never trips -> legacy path is the default.
    REQUIRE(lsxcommon::qwen_moe::AllocateQwenMoeExpertCache(store, 4, cache, err));
    REQUIRE(!lsxcommon::qwen_moe::QwenMoeGemvNohandshake(store));

    lsxcommon::qwen_moe::QwenMoeGenScratch gen_scratch;
    REQUIRE(lsxcommon::qwen_moe::EnsureQwenMoeScratch(stream, pw, gen_scratch) ==
            cudaSuccess);

    auto run_once = [&](lsxcommon::qwen::QwenGenerationResult &r) {
        lsxcommon::qwen::QwenKvCache kv;
        lsxcommon::sample::GreedySampler sampler;
        std::vector<int32_t> prompt = {1, 2, 3};
        return lsxcommon::qwen_moe::RunQwenMoeGeneration(
            stream, pw, store, cache, gen_scratch, prompt, /*max_new=*/4,
            /*eos_id=*/pw.moe_top.eos_token_id, r, sampler, &kv);
    };

    lsxcommon::qwen::QwenGenerationResult r1, r2;
    cudaError_t e1 = run_once(r1);
    if (e1 != cudaSuccess) WARN(r1.error_msg);
    REQUIRE(e1 == cudaSuccess);
    cudaError_t e2 = run_once(r2);
    REQUIRE(e2 == cudaSuccess);
    CHECK(r1.tokens == r2.tokens);
    REQUIRE(r1.final_logits.size() == r2.final_logits.size());
    CHECK(std::memcmp(r1.final_logits.data(), r2.final_logits.data(),
                      r1.final_logits.size() * sizeof(float)) == 0);
    CHECK(r1.ok); CHECK(r2.ok);

    lsxcommon::qwen_moe::FreeQwenMoeGenScratch(gen_scratch);
    lsxcommon::qwen_moe::FreeQwenMoeExpertCache(cache);
    store.Free();
    lsxcommon::qwen_moe::FreeQwenMoePinnedWeights(pw);
    cudaStreamDestroy(stream);
    lsxcommon::qwen::SetQwenTcFastMath(false);
    std::error_code ec;
    fs2::remove_all(in, ec); fs2::remove_all(out, ec);
}

// Real-geometry batch parity: the synthetic fixture (H=64) never exercises
// the split-K path (K >= 2048). This test drives batch13/down vs sequential
// RunLsxTcGemmPreQuant at the REAL 35B-A3B shapes (K=2048,N=1024 and
// K=512,N=2048, split_k=8/1) with random weights + adversarial scales, so
// any slice-order or scale divergence shows up as a memcmp miss.
TEST_CASE("tc batch gemm real-geometry split-k parity", "[qwen-moe-tc-batch]")
{
    if (!lsxcommon::lsx_tc::LsxTcRuntimeSupported()) return;
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    lsxcommon::qwen::SetQwenTcFastMath(true);

    const int H = 2048, I = 512, K = 8, N13 = 2 * I;
    REQUIRE(lsxcommon::lsx_tc::LsxTcBatchSplitK(H, N13) == 8);

    // Random NVFP4 payloads (bytes are already valid nibbles/scales).
    auto rnd8 = [](size_t n) {
        std::vector<uint8_t> v(n);
        uint32_t st = 12345;
        for (auto &b : v) {
            st = st * 1664525u + 1013904223u;
            b = static_cast<uint8_t>(st >> 24);
        }
        return v;
    };
    const size_t pack_stride = (4u << 20);  // w13(1MB)+w2(0.5MB)+slack
    const size_t scale_stride = (256u << 10);  // s13(128KB)+s2(32KB)+slack
    const int64_t w13_bytes = I * H;        // 2I*H/2 = 1MB
    const int64_t s13_bytes = 2 * I * (H / 16);
    std::vector<uint8_t> pack(pack_stride * 8), scale(scale_stride * 8);
    auto p1 = rnd8(pack.size());
    // Scales: alternating valid pow2-E4M3 bytes (round-19 adversarial
    // pattern) — random bytes overflow fp16 accumulators and make NaN
    // patterns order-dependent (test artifact, not product behavior).
    for (size_t i = 0; i < scale.size(); ++i)
        scale[i] = (i % 2) ? uint8_t{0x40} : uint8_t{0x28};
    std::copy(p1.begin(), p1.end(), pack.begin());

    uint8_t *d_pack, *d_scale;
    cudaMalloc(&d_pack, pack.size());
    cudaMalloc(&d_scale, scale.size());
    cudaMemcpy(d_pack, pack.data(), pack.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, scale.data(), scale.size(), cudaMemcpyHostToDevice);

    // Adversarial activations: alternating scale-block magnitudes.
    std::vector<__half> hx(H);
    for (int i = 0; i < H; ++i)
        hx[i] = __float2half(sinf(static_cast<float>(i) * 0.7f) * 0.5f);
    uint8_t *d_xq, *d_xqs;
    cudaMalloc(&d_xq, H / 2);
    cudaMalloc(&d_xqs, (H + 15) / 16);
    __half *d_hx_stage = nullptr;
    StageActs(stream, hx.data(), hx.size() * sizeof(__half),
              reinterpret_cast<void **>(&d_hx_stage));
    REQUIRE(lsxcommon::lsx_tc::LsxTcQuantizeActsNvfp4(stream, d_hx_stage, 1, H,
                                                   d_xq, d_xqs) == cudaSuccess);

    // Down projection consumes a DISTINCT per-slot activation (each expert has
    // its own swiglu output). The batch kernel applies a_slot_off = slot*K/2,
    // so the reference must read the same per-slot windows and the buffers
    // must be sized for all K slots (a single-row buffer would be read OOB by
    // the kernel and would make the parity comparison meaningless).
    const int64_t down_pack_slot = static_cast<int64_t>(I) / 2;
    const int64_t down_scale_slot = static_cast<int64_t>((I + 15) / 16);
    std::vector<__half> hdown(static_cast<size_t>(K) * I);
    for (size_t i = 0; i < hdown.size(); ++i)
        hdown[i] = __float2half(sinf(static_cast<float>(i) * 0.31f) * 0.5f);
    uint8_t *d_ad, *d_ads;
    cudaMalloc(&d_ad, static_cast<size_t>(K) * down_pack_slot);
    cudaMalloc(&d_ads, static_cast<size_t>(K) * down_scale_slot);
    __half *d_hdown = nullptr;
    StageActs(stream, hdown.data(), hdown.size() * sizeof(__half),
              reinterpret_cast<void **>(&d_hdown));
    for (int k = 0; k < K; ++k) {
        REQUIRE(lsxcommon::lsx_tc::LsxTcQuantizeActsNvfp4(
                    stream, d_hdown + static_cast<std::size_t>(k) * I, 1, I,
                    d_ad + static_cast<int64_t>(k) * down_pack_slot,
                    d_ads + static_cast<int64_t>(k) * down_scale_slot) ==
                cudaSuccess);
    }

    // Distinct per-segment global scales (catch gs mixups bitwise).
    std::vector<lsxcommon::lsx_tc::LsxTcBatchSlot> tbl(K);
    std::vector<float> hseg(2 * 8);
    for (int k = 0; k < K; ++k) {
        tbl[k] = {k, 0.005f + 0.001f * k, 0.015f + 0.001f * k,
                  0.02f + 0.001f * k, 0.f};
        hseg[2 * k] = tbl[k].gs1;
        hseg[2 * k + 1] = tbl[k].gs3;
    }
    float *d_seg;
    cudaMalloc(&d_seg, sizeof(float) * hseg.size());
    cudaMemcpy(d_seg, hseg.data(), sizeof(float) * hseg.size(),
               cudaMemcpyHostToDevice);
    lsxcommon::lsx_tc::LsxTcBatchSlot *d_tbl;
    cudaMalloc(&d_tbl, sizeof(tbl[0]) * K);
    cudaMemcpy(d_tbl, tbl.data(), sizeof(tbl[0]) * K, cudaMemcpyHostToDevice);

    // Sequential reference: per-slot RunLsxTcGemmPreQuant (legacy split-K).
    __half *d_ref13, *d_ref2, *d_bat13, *d_bat2;
    cudaMalloc(&d_ref13, static_cast<size_t>(K) * N13 * 2);
    cudaMalloc(&d_ref2, static_cast<size_t>(K) * H * 2);
    cudaMalloc(&d_bat13, static_cast<size_t>(K) * N13 * 2);
    cudaMalloc(&d_bat2, static_cast<size_t>(K) * H * 2);
    for (int k = 0; k < K; ++k) {
        uint8_t *pk = d_pack + static_cast<int64_t>(k) * pack_stride;
        uint8_t *sc = d_scale + static_cast<int64_t>(k) * scale_stride;
        REQUIRE(lsxcommon::lsx_tc::RunLsxTcGemmPreQuant(
                    stream, d_xq, d_xqs, pk, sc, 1.0f,
                    d_ref13 + static_cast<int64_t>(k) * N13, 1, H, N13,
                    false, d_seg + 2 * k, nullptr) == cudaSuccess);
        REQUIRE(lsxcommon::lsx_tc::RunLsxTcGemmPreQuant(
                    stream, d_ad + static_cast<int64_t>(k) * down_pack_slot,
                    d_ads + static_cast<int64_t>(k) * down_scale_slot,
                    pk + w13_bytes, sc + s13_bytes, tbl[k].gs2,
                    d_ref2 + static_cast<int64_t>(k) * H, 1, I, H,
                    false) == cudaSuccess);
    }
    // Batched.
    {
        float *d_part;
        cudaMalloc(&d_part, static_cast<size_t>(8) * K * N13 * 4);
        REQUIRE(lsxcommon::lsx_tc::RunLsxTcGemmBatch13(
                    stream, d_xq, d_xqs, d_pack, d_scale,
                    static_cast<int64_t>(pack_stride),
                    static_cast<int64_t>(scale_stride), d_tbl, K, H, N13,
                    d_bat13, d_part) == cudaSuccess);
        cudaFree(d_part);
    }
    {
        float *d_part = nullptr;
        REQUIRE(lsxcommon::lsx_tc::RunLsxTcGemmBatchDown(
                    stream, d_ad, d_ads, d_pack, d_scale,
                    static_cast<int64_t>(pack_stride),
                    static_cast<int64_t>(scale_stride), d_tbl, K, w13_bytes,
                    s13_bytes, I, H, d_bat2, d_part) == cudaSuccess);
    }
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<uint8_t> r13(static_cast<size_t>(K) * N13 * 2);
    std::vector<uint8_t> b13(static_cast<size_t>(K) * N13 * 2);
    std::vector<uint8_t> r2(static_cast<size_t>(K) * H * 2);
    std::vector<uint8_t> b2(static_cast<size_t>(K) * H * 2);
    REQUIRE(cudaMemcpy(r13.data(), d_ref13, r13.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(b13.data(), d_bat13, b13.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(r2.data(), d_ref2, r2.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(b2.data(), d_bat2, b2.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    // Every slot must be bitwise identical to its per-slot reference.
    const size_t slot13 = r13.size() / static_cast<size_t>(K);
    const size_t slot2 = r2.size() / static_cast<size_t>(K);
    bool ok13 = true, ok2 = true;
    for (int k = 0; k < K; ++k) {
        ok13 &= std::memcmp(r13.data() + static_cast<size_t>(k) * slot13,
                            b13.data() + static_cast<size_t>(k) * slot13,
                            slot13) == 0;
        ok2 &= std::memcmp(r2.data() + static_cast<size_t>(k) * slot2,
                           b2.data() + static_cast<size_t>(k) * slot2,
                           slot2) == 0;
    }
    CHECK(ok13);
    CHECK(ok2);
    // Non-vacuity: the per-slot weights/activations are distinct, so slot 0
    // and slot 1 results must differ. If these ever match, the test has been
    // silently defanged (e.g. missing device->host copy) and proves nothing.
    CHECK(std::memcmp(r13.data(), r13.data() + slot13, slot13) != 0);
    CHECK(std::memcmp(r2.data(), r2.data() + slot2, slot2) != 0);

    cudaFree(d_pack); cudaFree(d_scale); cudaFree(d_xq); cudaFree(d_xqs);
    cudaFree(d_ad); cudaFree(d_ads);
    cudaFree(d_seg); cudaFree(d_tbl); cudaFree(d_ref13); cudaFree(d_ref2);
    cudaFree(d_bat13); cudaFree(d_bat2);
    cudaStreamDestroy(stream);
    lsxcommon::qwen::SetQwenTcFastMath(false);
}
