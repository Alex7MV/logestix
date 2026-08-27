// seq==1 decode-path NVFP4 GEMV: weight-streaming CUDA-core dot product.
//
// Round-18 findings lever: at seq==1 the routed/shared GEMMs read ~1.73 MB
// per expert but sustain only ~30 GB/s through the 16x16x32 tile kernel
// (M=1 wastes the tile M-dimension entirely). The GEMV kernel streams
// packed E2M1 weights straight from global memory with coalesced uint4
// loads and a shfl tree reduction -- no shared-memory staging, no MMA.
//
// Numeric contract: FP32 accumulate, k-order free (tolerance class), exactly
// like the TC fast-math paths. Determinism contract: fixed partitioning ->
// bit-stable across repeated launches (run-to-run), pinned here.
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_moe_inference.h"
#include <cmath>

namespace {

// Test-local E2M1 decode table (values match nvfp4_codec kE2M1Table).
constexpr float kE2M1TestTable[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// Bit-exact host replica of nvfp4_codec.h's E4M3ScaleToFloat (e4-field zero).
float E4M3ScaleToFloatHost(uint8_t e4m3)
{
    const uint32_t e4 = (e4m3 >> 3) & 0x0Fu;
    if (e4 == 0) return 0.0f;
    uint32_t bits = (e4 + 120u) << 23;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

void HostRefGemv(const __half *x, const uint8_t *packed,
                 const uint8_t *scales, float global_scale, float *out,
                 int K, int N)
{
    for (int n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int64_t e = static_cast<int64_t>(n) * K + k;
            const uint8_t byte = packed[e >> 1];
            const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
            const float scl =
                E4M3ScaleToFloatHost(scales[e >> 4]) * global_scale;
            acc += __half2float(x[k]) * kE2M1TestTable[nib] * scl;
        }
        out[n] = acc;
    }
}

struct GemvCase {
    int K, N;
};

void RunParity(const GemvCase &c, bool acc_in, float gs)
{
    std::vector<__half> hx(c.K);
    std::vector<uint8_t> hp(static_cast<size_t>(c.K) * c.N / 2),
        hs(static_cast<size_t>(c.K) * c.N / 16);
    for (int i = 0; i < c.K; ++i)
        hx[i] = __float2half(float((i % 13) - 6) * 0.25f);
    for (size_t i = 0; i < hp.size(); ++i) hp[i] = static_cast<uint8_t>((i * 29) & 0xFF);
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = static_cast<uint8_t>(128 + (i % 48));

    std::vector<float> h_ref(c.N);
    HostRefGemv(hx.data(), hp.data(), hs.data(), gs, h_ref.data(), c.K, c.N);

    __half *dx, *dy;
    uint8_t *dp, *ds;
    cudaMalloc(&dx, static_cast<size_t>(c.K) * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&dy, static_cast<size_t>(c.N) * 2);
    cudaMemcpy(dx, hx.data(), static_cast<size_t>(c.K) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);
    const float seed_val = acc_in ? 0.5f : 0.0f;
    std::vector<__half> hseed(c.N, __float2half(seed_val));
    cudaMemcpy(dy, hseed.data(), static_cast<size_t>(c.N) * 2,
               cudaMemcpyHostToDevice);

    cudaError_t e = lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
        nullptr, dx, dp, ds, gs, dy, c.K, c.N, acc_in);
    REQUIRE(e == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hy(c.N);
    cudaMemcpy(hy.data(), dy, static_cast<size_t>(c.N) * 2, cudaMemcpyDeviceToHost);
    for (int n = 0; n < c.N; ++n) {
        const float ref = __half2float(__float2half(h_ref[n])) + seed_val;
        INFO("n=" << n << " ref=" << ref << " got=" << __half2float(hy[n]));
        CHECK(std::abs(__half2float(hy[n]) - ref) <
              2e-3f * std::abs(ref) + 2e-4f);
    }

    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
}

}  // namespace

TEST_CASE("NVFP4 GEMV parity vs host reference", "[qwen][gemm][nvfp4][gemv]")
{
    // Real decode shapes: w13-like [K=hidden, N=moe_i] and w2-like inverse.
    const GemvCase cases[] = {{2048, 512}, {512, 2048}, {256, 96}};
    for (const auto &c : cases)
        RunParity(c, /*acc=*/false, /*gs=*/0.125f);
}

// Head-class geometry: very large N exercises the flat-row grid mapping and
// the LSX_GEMV_HEAD oversized-tensor dispatch branch.
TEST_CASE("NVFP4 GEMV large-N parity (head class)",
          "[qwen][gemm][nvfp4][gemv][head]")
{
    // The oversized-tensor branch is opt-in via env (micro-validated);
    // enable it here so the large case really exercises the streaming
    // kernel instead of silently falling back to the tile path.
    _putenv("LSX_GEMV_HEAD=1");
    lsxcommon::qwen::SetQwenSeq1Gemv(true);
    lsxcommon::qwen::SetQwenTcFastMath(true);
    const GemvCase cases[] = {{256, 65536}, {2048, 20000}};
    for (const auto &c : cases)
        RunParity(c, /*acc=*/false, /*gs=*/0.125f);
    lsxcommon::qwen::SetQwenSeq1Gemv(false);
    lsxcommon::qwen::SetQwenTcFastMath(false);
}

// Pins the within-chunk scale-block split: elements 0..15 of every
// 32-element uint4 chunk use scale byte sb, elements 16..31 use sb+1.
// Adversarial contrast (sb decodes to ~0, sb+1 to the max pow2) makes ANY
// lo/hi-nibble <-> scale-half mixup explode far beyond tolerance.
TEST_CASE("NVFP4 GEMV chunk scale-block split", "[qwen][gemm][nvfp4][gemv]")
{
    constexpr int K = 512, N = 64;
    const float gs = 1.0f;
    std::vector<__half> hx(K);
    std::vector<uint8_t> hp(static_cast<size_t>(K) * N / 2),
        hs(static_cast<size_t>(K) * N / 16);
    for (int i = 0; i < K; ++i) hx[i] = __float2half(i % 2 ? 1.0f : -1.0f);
    for (size_t i = 0; i < hp.size(); ++i) hp[i] = 0x11;   // ±1 nibbles
    for (size_t i = 0; i < hs.size(); i += 2) {
        hs[i] = 0x08;                                      // 2^-6
        if (i + 1 < hs.size()) hs[i + 1] = 0xf0;           // 2^7
    }

    std::vector<float> h_ref(N);
    HostRefGemv(hx.data(), hp.data(), hs.data(), gs, h_ref.data(), K, N);

    __half *dx, *dy; uint8_t *dp, *ds;
    cudaMalloc(&dx, static_cast<size_t>(K) * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&dy, static_cast<size_t>(N) * 2);
    cudaMemcpy(dx, hx.data(), static_cast<size_t>(K) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
                nullptr, dx, dp, ds, gs, dy, K, N, false) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hy(N);
    cudaMemcpy(hy.data(), dy, static_cast<size_t>(N) * 2, cudaMemcpyDeviceToHost);
    for (int n = 0; n < N; ++n) {
        INFO("n=" << n << " ref=" << h_ref[n] << " got=" <<
             __half2float(hy[n]));
        CHECK(std::abs(__half2float(hy[n]) - h_ref[n]) <
              2e-3f * std::abs(h_ref[n]) + 2e-4f);
    }
    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(dy);
}

TEST_CASE("NVFP4 GEMV deterministic across launches",
          "[qwen][gemm][nvfp4][gemv]")
{
    constexpr int K = 2048, N = 512;
    std::vector<__half> hx(K);
    std::vector<uint8_t> hp(static_cast<size_t>(K) * N / 2),
        hs(static_cast<size_t>(K) * N / 16);
    for (int i = 0; i < K; ++i) hx[i] = __float2half(float((i % 9) - 4));
    for (size_t i = 0; i < hp.size(); ++i) hp[i] = static_cast<uint8_t>((i * 37) & 0xFF);
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = static_cast<uint8_t>(128 + (i % 24));

    __half *dx, *d1, *d2;
    uint8_t *dp, *ds;
    cudaMalloc(&dx, static_cast<size_t>(K) * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&d1, static_cast<size_t>(N) * 2);
    cudaMalloc(&d2, static_cast<size_t>(N) * 2);
    cudaMemcpy(dx, hx.data(), static_cast<size_t>(K) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
                nullptr, dx, dp, ds, 0.25f, d1, K, N, false) == cudaSuccess);
    REQUIRE(lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
                nullptr, dx, dp, ds, 0.25f, d2, K, N, false) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> r1(N), r2(N);
    cudaMemcpy(r1.data(), d1, static_cast<size_t>(N) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(r2.data(), d2, static_cast<size_t>(N) * 2, cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(r1.data(), r2.data(), r1.size() * sizeof(__half)) == 0);

    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(d1); cudaFree(d2);
}

TEST_CASE("RunQwenNvfp4Gemm fast-math seq==1 dispatch reaches GEMV "
          "(bitwise equal)",
          "[qwen][gemm][nvfp4][gemv]")
{
    lsxcommon::qwen::SetQwenTcFastMath(true);
    lsxcommon::qwen::SetQwenSeq1Gemv(true);
    constexpr int K = 2048, N = 512;
    std::vector<__half> hx(K);
    std::vector<uint8_t> hp(static_cast<size_t>(K) * N / 2),
        hs(static_cast<size_t>(K) * N / 16);
    for (int i = 0; i < K; ++i)
        hx[i] = __float2half(float((i % 11) - 5) * 0.5f);
    for (size_t i = 0; i < hp.size(); ++i) hp[i] = static_cast<uint8_t>((i * 17) & 0xFF);
    for (size_t i = 0; i < hs.size(); ++i) hs[i] = static_cast<uint8_t>(128 + (i % 32));

    __half *dx, *d_tile, *d_gemv;
    uint8_t *dp, *ds;
    cudaMalloc(&dx, static_cast<size_t>(K) * 2);
    cudaMalloc(&dp, hp.size());
    cudaMalloc(&ds, hs.size());
    cudaMalloc(&d_tile, static_cast<size_t>(N) * 2);
    cudaMalloc(&d_gemv, static_cast<size_t>(N) * 2);
    cudaMemcpy(dx, hx.data(), static_cast<size_t>(K) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), hs.size(), cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen::RunQwenNvfp4Gemm(
                nullptr, dx, dp, ds, 0.125f, d_tile, 1, K, N) == cudaSuccess);
    REQUIRE(lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
                nullptr, dx, dp, ds, 0.125f, d_gemv, K, N, false) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> t(N), g(N);
    cudaMemcpy(t.data(), d_tile, static_cast<size_t>(N) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(g.data(), d_gemv, static_cast<size_t>(N) * 2, cudaMemcpyDeviceToHost);
    // Same kernel -> identical bits. Any mismatch means the dispatcher did
    // NOT reach the GEMV kernel for the supported seq==1 shape.
    CHECK(std::memcmp(t.data(), g.data(), N * sizeof(__half)) == 0);

    lsxcommon::qwen::SetQwenSeq1Gemv(false);
    lsxcommon::qwen::SetQwenTcFastMath(false);
    cudaFree(dx); cudaFree(dp); cudaFree(ds); cudaFree(d_tile); cudaFree(d_gemv);
}




// ── Batched routed-expert GEMV chain (round-19 next lever) ─────────────────
// Ground truth = the already-pinned single-row GEMV entry called per stage
// (gate / up slices of w13, then down), eliminating hand-math reference
// drift. Verifies batch plumbing: slot-major indexing, per-slot gs/coeff,
// fused-slab offsets, and the ascending-order coeff reduce.
namespace {

struct BatchCase {
    int H, I, S;
};

void RunBatch(const BatchCase &c, std::vector<float> *acc_out)
{
    const int H = c.H, I = c.I, S = c.S;
    const int64_t w13b = static_cast<int64_t>(2 * I) * H / 2;
    const int64_t w2b = static_cast<int64_t>(H) * I / 2;
    const int64_t s13b = static_cast<int64_t>(2 * I) * H / 16;
    const int64_t s2b = static_cast<int64_t>(H) * I / 16;
    const int64_t pack_stride = ((w13b + w2b + 63) / 64) * 64;
    const int64_t scale_stride = ((s13b + s2b + 63) / 64) * 64;

    std::vector<uint8_t> hp(static_cast<size_t>(pack_stride * S)),
        hs(static_cast<size_t>(scale_stride * S));
    for (int e = 0; e < S; ++e) {
        uint8_t *p = hp.data() + static_cast<size_t>(e) * pack_stride;
        uint8_t *s = hs.data() + static_cast<size_t>(e) * scale_stride;
        for (int64_t i = 0; i < pack_stride; ++i)
            p[i] = static_cast<uint8_t>((i * 29 + e * 17) & 0xFF);
        for (int64_t i = 0; i < scale_stride; ++i)
            s[i] = static_cast<uint8_t>(128 + ((i + 5 * e) % 48));
    }

    const float coeffs[4] = {0.75f, -0.5f, 1.25f, 0.0625f};
    const int eids[4] = {3, 0, 3, 1};
    std::vector<lsxcommon::qwen_moe::QwenMoeGvSlotDev> slots(S);
    for (int k = 0; k < S; ++k) {
        slots[k].dev_slot = eids[k];
        slots[k].gs1 = static_cast<float>(1 << (k % 4)) / 256.0f;
        slots[k].gs3 = static_cast<float>(1 << ((k + 1) % 4)) / 256.0f;
        slots[k].gs2 = static_cast<float>(1 << ((k + 2) % 4)) / 256.0f;
        slots[k].coeff = coeffs[k];
    }

    std::vector<__half> hx(H);
    for (int i = 0; i < H; ++i)
        hx[i] = __float2half(float((i % 11) - 5) * 0.25f);

    uint8_t *d_pack, *d_scale, *d_slots;
    cudaMalloc(&d_pack, hp.size());
    cudaMalloc(&d_scale, hs.size());
    cudaMalloc(&d_slots, sizeof(*slots.data()) * S);
    cudaMemcpy(d_pack, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, hs.data(), hs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slots, slots.data(), sizeof(*slots.data()) * S,
               cudaMemcpyHostToDevice);

    __half *dx, *ygu, *hbuf, *ypart;
    float *acc;
    cudaMalloc(&dx, static_cast<size_t>(H) * 2);
    cudaMalloc(&ygu, static_cast<size_t>(S) * 2 * I * 2);
    cudaMalloc(&hbuf, static_cast<size_t>(S) * I * 2);
    cudaMalloc(&ypart, static_cast<size_t>(S) * H * 2);
    cudaMalloc(&acc, static_cast<size_t>(H) * 4);
    cudaMemcpy(dx, hx.data(), static_cast<size_t>(H) * 2,
               cudaMemcpyHostToDevice);
    std::vector<float> h_acc0(H);
    for (int i = 0; i < H; ++i) h_acc0[i] = 0.125f * (i % 7);
    cudaMemcpy(acc, h_acc0.data(), static_cast<size_t>(H) * 4,
               cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeRoutedGemmVBatch(
                nullptr, d_pack, d_scale, pack_stride, scale_stride,
                reinterpret_cast<lsxcommon::qwen_moe::QwenMoeGvSlotDev *>(
                    d_slots),
                S, w13b, s13b, H, I, dx, ygu, hbuf, ypart, acc) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    // ── Ground truth via the pinned single-row GEMV entry ──────────────
    __half *d_stage_in = nullptr, *d_stage_out = nullptr;
    cudaMalloc(&d_stage_in, static_cast<size_t>(std::max(H, I)) * 2);
    cudaMalloc(&d_stage_out,
               static_cast<size_t>(std::max(2 * I, H)) * 2);

    // Host-side expected ygu (bit-exact vs entry), hbuf, ypart, acc.
    std::vector<__half> exp_gu(static_cast<size_t>(S) * 2 * I);
    std::vector<__half> exp_h(static_cast<size_t>(S) * I);
    std::vector<__half> exp_yp(static_cast<size_t>(S) * H);
    for (int k = 0; k < S; ++k) {
        const int eid = eids[k];
        const uint8_t *pbase =
            d_pack + static_cast<int64_t>(eid) * pack_stride;   // host? NO --
        (void)pbase;
        REQUIRE(lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
                    nullptr, dx, d_pack + static_cast<int64_t>(eid) *
                                      pack_stride,
                    d_scale + static_cast<int64_t>(eid) * scale_stride,
                    slots[k].gs1, d_stage_out, H, I, false) == cudaSuccess);
        cudaMemcpy(exp_gu.data() + static_cast<int64_t>(k) * 2 * I,
                   d_stage_out, static_cast<size_t>(I) * 2,
                   cudaMemcpyDeviceToHost);
        REQUIRE(lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
                    nullptr, dx, d_pack + static_cast<int64_t>(eid) *
                                      pack_stride + w13b / 2,
                    d_scale + static_cast<int64_t>(eid) * scale_stride +
                        s13b / 2,
                    slots[k].gs3, d_stage_out, H, I, false) == cudaSuccess);
        cudaMemcpy(exp_gu.data() + static_cast<int64_t>(k) * 2 * I + I,
                   d_stage_out, static_cast<size_t>(I) * 2,
                   cudaMemcpyDeviceToHost);
        for (int j = 0; j < I; ++j) {
            const size_t o = static_cast<size_t>(k) * 2 * I + j;
            const float g = __half2float(exp_gu[o]);
            const float u =
                __half2float(exp_gu[static_cast<size_t>(k) * 2 * I + I + j]);
            exp_h[static_cast<size_t>(k) * I + j] =
                __float2half((g / (1.0f + std::exp(-g))) * u);
        }
        cudaMemcpy(d_stage_in, hbuf + static_cast<int64_t>(k) * I,
                   static_cast<size_t>(I) * 2, cudaMemcpyDeviceToDevice);
        REQUIRE(lsxcommon::qwen::RunQwenNvfp4GemmSeq1GemmV(
                    nullptr, d_stage_in,
                    d_pack + static_cast<int64_t>(eid) * pack_stride +
                        w13b,
                    d_scale + static_cast<int64_t>(eid) * scale_stride +
                        s13b,
                    slots[k].gs2, d_stage_out, I, H, false) == cudaSuccess);
        cudaMemcpy(exp_yp.data() + static_cast<int64_t>(k) * H,
                   d_stage_out, static_cast<size_t>(H) * 2,
                   cudaMemcpyDeviceToHost);
    }
    std::vector<float> exp_acc(h_acc0);
    for (int k = 0; k < S; ++k)
        for (int r = 0; r < H; ++r)
            exp_acc[r] +=
                slots[k].coeff *
                __half2float(exp_yp[static_cast<size_t>(k) * H + r]);

    // Device buffers bit-compare.
    std::vector<__half> got_gu(S * 2 * I), got_h(S * I), got_yp(S * H);
    cudaMemcpy(got_gu.data(), ygu, got_gu.size() * 2,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(got_h.data(), hbuf, got_h.size() * 2,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(got_yp.data(), ypart, got_yp.size() * 2,
               cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(got_gu.data(), exp_gu.data(),
                      got_gu.size() * sizeof(__half)) == 0);
    CHECK(std::memcmp(got_h.data(), exp_h.data(),
                      got_h.size() * sizeof(__half)) == 0);
    CHECK(std::memcmp(got_yp.data(), exp_yp.data(),
                      got_yp.size() * sizeof(__half)) == 0);

    std::vector<float> got_acc(H);
    cudaMemcpy(got_acc.data(), acc, static_cast<size_t>(H) * 4,
               cudaMemcpyDeviceToHost);
    for (int r = 0; r < H; ++r) {
        INFO("r=" << r << " ref=" << exp_acc[r] << " got=" << got_acc[r]);
        CHECK(std::abs(got_acc[r] - exp_acc[r]) <
              1e-6f * std::abs(exp_acc[r]) + 1e-5f);
    }
    if (acc_out) *acc_out = got_acc;

    cudaFree(d_pack); cudaFree(d_scale); cudaFree(d_slots);
    cudaFree(dx); cudaFree(ygu); cudaFree(hbuf); cudaFree(ypart);
    cudaFree(acc); cudaFree(d_stage_in); cudaFree(d_stage_out);
}

}  // namespace

TEST_CASE("RoutedGemmVBatch matches per-stage single GEMV ground truth",
          "[qwen][gemm][nvfp4][gemv][batch]")
{
    RunBatch({128, 96, 4}, nullptr);
}

// Two identical batches on the same fixture must produce bit-equal acc.
TEST_CASE("RoutedGemmVBatch deterministic across launches",
          "[qwen][gemm][nvfp4][gemv][batch]")
{
    std::vector<float> a1, a2;
    RunBatch({64, 32, 2}, &a1);
    RunBatch({64, 32, 2}, &a2);
    REQUIRE(a1.size() == a2.size());
    CHECK(std::memcmp(a1.data(), a2.data(),
                      a1.size() * sizeof(float)) == 0);
}

// No-handshake gather: table composed from device routing results +
// per-expert meta LUT must equal the host-composed expectation bit-for-bit.
TEST_CASE("GemvGatherTable composes slot table from device inputs",
          "[qwen][gemm][nvfp4][gemv][batch]")
{
    constexpr int E = 8, K = 5;
    std::vector<lsxcommon::qwen_moe::QwenMoeRoutedSlot> routes(K);
    const int eids[K] = {7, 2, 7, 0, 5};
    for (int k = 0; k < K; ++k) {
        routes[k].eid = eids[k];
        routes[k].coeff = 0.125f * (k + 1);
    }
    std::vector<lsxcommon::qwen_moe::QwenMoeExpertMetaDev> meta(E);
    for (int e = 0; e < E; ++e) {
        meta[e].dev_slot = 100 + e;
        meta[e].gs1 = 1.0f / float(1 << (e % 6));
        meta[e].gs2 = 1.0f / float(1 << ((e + 3) % 6));
        meta[e].gs3 = 1.0f / float(1 << ((e + 5) % 6));
    }

    lsxcommon::qwen_moe::QwenMoeRoutedSlot *d_routes = nullptr;
    lsxcommon::qwen_moe::QwenMoeExpertMetaDev *d_meta = nullptr;
    lsxcommon::qwen_moe::QwenMoeGvSlotDev *d_tbl = nullptr;
    cudaMalloc(&d_routes, routes.size() * sizeof(routes[0]));
    cudaMalloc(&d_meta, meta.size() * sizeof(meta[0]));
    cudaMalloc(&d_tbl, K * sizeof(*d_tbl));
    cudaMemcpy(d_routes, routes.data(), routes.size() * sizeof(routes[0]),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_meta, meta.data(), meta.size() * sizeof(meta[0]),
               cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::qwen_moe::RunQwenMoeGemvGatherTable(
                nullptr, d_routes, K, d_meta, d_tbl) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<lsxcommon::qwen_moe::QwenMoeGvSlotDev> got(K);
    cudaMemcpy(got.data(), d_tbl, K * sizeof(*d_tbl),
               cudaMemcpyDeviceToHost);
    for (int k = 0; k < K; ++k) {
        INFO("k=" << k);
        CHECK(got[k].dev_slot == meta[eids[k]].dev_slot);
        CHECK(got[k].gs1 == meta[eids[k]].gs1);
        CHECK(got[k].gs2 == meta[eids[k]].gs2);
        CHECK(got[k].gs3 == meta[eids[k]].gs3);
        CHECK(got[k].coeff == routes[k].coeff);
    }
    cudaFree(d_routes); cudaFree(d_meta); cudaFree(d_tbl);
}

// Round-20c protocol pin: the batched chain under stream capture + graph
// replay must produce BIT-IDENTICAL outputs to the plain launches (graph
// capture is required to be launch-transparent -- no forbidden ops inside
// the section).
TEST_CASE("Batched GEMV core is CUDA-graph transparent",
          "[qwen][gemm][nvfp4][gemv][batch][graph]")
{
    using namespace lsxcommon::qwen_moe;
    const int H = 128, I = 64, S = 2;   // I=64: slice=2 warps of chunks
    const int64_t w13b = static_cast<int64_t>(2 * I) * H / 2;
    const int64_t w2b = static_cast<int64_t>(H) * I / 2;
    const int64_t s13b = static_cast<int64_t>(2 * I) * H / 16;
    const int64_t s2b = static_cast<int64_t>(H) * I / 16;
    const int64_t pack_stride = ((w13b + w2b + 63) / 64) * 64;
    const int64_t scale_stride = ((s13b + s2b + 63) / 64) * 64;

    std::vector<uint8_t> hp(static_cast<size_t>(pack_stride * S)),
        hs(static_cast<size_t>(scale_stride * S));
    for (int e = 0; e < S; ++e) {
        uint8_t *p = hp.data() + static_cast<size_t>(e) * pack_stride;
        uint8_t *sc = hs.data() + static_cast<size_t>(e) * scale_stride;
        for (int64_t i = 0; i < pack_stride; ++i)
            p[i] = static_cast<uint8_t>((i * 29 + e * 31) & 0xFF);
        for (int64_t i = 0; i < scale_stride; ++i)
            sc[i] = static_cast<uint8_t>(128 + ((i + 7 * e) % 48));
    }
    QwenMoeGvSlotDev tbl[S];
    for (int k = 0; k < S; ++k) {
        tbl[k].dev_slot = k;
        tbl[k].gs1 = 0.25f;
        tbl[k].gs3 = 0.125f;
        tbl[k].gs2 = 0.0625f;
        tbl[k].coeff = k == 0 ? 0.75f : -0.5f;
    }
    std::vector<__half> hx(H);
    for (int i = 0; i < H; ++i)
        hx[i] = __float2half(float((i % 9) - 4) * 0.25f);

    uint8_t *d_pack, *d_scale;
    QwenMoeGvSlotDev *d_tblA = nullptr, *d_tblB = nullptr;
    __half *dx, *yguA, *hbufA, *ypartA, *yguB, *hbufB, *ypartB;
    float *accA, *accB;
    cudaMalloc(&d_pack, hp.size());
    cudaMalloc(&d_scale, hs.size());
    cudaMemcpy(d_pack, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, hs.data(), hs.size(), cudaMemcpyHostToDevice);
    auto alloc_set = [&](QwenMoeGvSlotDev **t, __half **g, __half **hb,
                         __half **yp, float **ac) {
        cudaMalloc(t, sizeof(QwenMoeGvSlotDev) * S);
        cudaMalloc(reinterpret_cast<void **>(g),
                   static_cast<size_t>(S) * 2 * I * 2);
        cudaMalloc(reinterpret_cast<void **>(hb),
                   static_cast<size_t>(S) * I * 2);
        cudaMalloc(reinterpret_cast<void **>(yp),
                   static_cast<size_t>(S) * H * 2);
        cudaMalloc(reinterpret_cast<void **>(ac), static_cast<size_t>(H) * 4);
    };
    alloc_set(&d_tblA, &yguA, &hbufA, &ypartA, &accA);
    alloc_set(&d_tblB, &yguB, &hbufB, &ypartB, &accB);
    cudaMemcpy(d_tblA, tbl, sizeof(tbl), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tblB, tbl, sizeof(tbl), cudaMemcpyHostToDevice);
    cudaMalloc(&dx, static_cast<size_t>(H) * 2);
    cudaMemcpy(dx, hx.data(), static_cast<size_t>(H) * 2,
               cudaMemcpyHostToDevice);
    std::vector<float> acc_seed(H, 0.25f);
    cudaMemcpy(accA, acc_seed.data(), H * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(accB, acc_seed.data(), H * 4, cudaMemcpyHostToDevice);

    // (a) Plain chain launch.
    REQUIRE(RunQwenMoeRoutedGemmVBatchCore(
                nullptr, d_pack, d_scale, pack_stride, scale_stride, d_tblA,
                S, w13b, s13b, H, I, dx, yguA, hbufA, ypartA, accA) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    // (b) Captured + replayed through a graph on a non-blocking stream.
    cudaStream_t gs = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&gs, cudaStreamNonBlocking) ==
            cudaSuccess);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t gexec = nullptr;
    REQUIRE(cudaStreamBeginCapture(gs,
                                   cudaStreamCaptureModeThreadLocal) ==
            cudaSuccess);
    cudaError_t ce = RunQwenMoeRoutedGemmVBatchCore(
        gs, d_pack, d_scale, pack_stride, scale_stride, d_tblB, S, w13b,
        s13b, H, I, dx, yguB, hbufB, ypartB, accB);
    if (ce != cudaSuccess || cudaStreamEndCapture(gs, &graph) !=
                                 cudaSuccess) {
        INFO("capture failed: " << cudaGetErrorString(cudaGetLastError()));
        CHECK(false);
    } else {
        REQUIRE(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0) ==
                cudaSuccess);
        REQUIRE(cudaGraphLaunch(gexec, gs) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(gs) == cudaSuccess);

        std::vector<char> a(static_cast<size_t>(S) * 2 * I * 2),
            b(static_cast<size_t>(S) * 2 * I * 2);
        cudaMemcpy(a.data(), yguA, a.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), yguB, b.size(), cudaMemcpyDeviceToHost);
        CHECK(std::memcmp(a.data(), b.data(), a.size()) == 0);
        a.resize(static_cast<size_t>(S) * I * 2);
        b.resize(a.size());
        cudaMemcpy(a.data(), hbufA, a.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), hbufB, b.size(), cudaMemcpyDeviceToHost);
        CHECK(std::memcmp(a.data(), b.data(), a.size()) == 0);
        a.resize(static_cast<size_t>(S) * H * 2);
        b.resize(a.size());
        cudaMemcpy(a.data(), ypartA, a.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), ypartB, b.size(), cudaMemcpyDeviceToHost);
        CHECK(std::memcmp(a.data(), b.data(), a.size()) == 0);
        std::vector<float> fa(H), fb(H);
        cudaMemcpy(fa.data(), accA, H * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(fb.data(), accB, H * 4, cudaMemcpyDeviceToHost);
        CHECK(std::memcmp(fa.data(), fb.data(), H * sizeof(float)) == 0);
    }
    if (gexec) cudaGraphExecDestroy(gexec);
    if (graph) cudaGraphDestroy(graph);
    cudaStreamDestroy(gs);
    cudaFree(d_pack); cudaFree(d_scale); cudaFree(dx);
    cudaFree(d_tblA); cudaFree(d_tblB);
    cudaFree(yguA); cudaFree(hbufA); cudaFree(ypartA); cudaFree(accA);
    cudaFree(yguB); cudaFree(hbufB); cudaFree(ypartB); cudaFree(accB);
}

// ── Gemma routed GEMV chain (round 21) ──────────────────────────────────────
// Three SEPARATED slabs inside the staged expert slot (w1|w2|w3 packed,
// s1|s2|s3 scales), geGLU activation (gelu_tanh(g)*u), weighted reduce into
// an fp32 accumulator the driver converts/casts for its BF16 pipeline.
// Host reference mirrors the shared dot contract validated by the qwen
// suites above, including an adversarial block-alternating scale pattern.
#include "lsxcommon/gemma4_inference.h"

namespace {

float GeluTanhHost(float v)
{
    const float k0 = 0.7978845608028654f;
    return 0.5f * v * (1.0f + std::tanh(k0 * (v + 0.044715f * v * v * v)));
}

void RunGemmaBatch(const int H, const int I, const int S, bool adversarial,
                   std::vector<float> *acc_out)
{
    const int64_t b1 = static_cast<int64_t>(I) * H / 2;
    const int64_t b2 = static_cast<int64_t>(H) * I / 2;
    const int64_t s1b = static_cast<int64_t>(I) * H / 16;
    const int64_t s2b = static_cast<int64_t>(H) * I / 16;
    const int64_t pack_stride = ((2 * b1 + b2 + 63) / 64) * 64;
    const int64_t scale_stride = ((2 * s1b + s2b + 63) / 64) * 64;

    std::vector<uint8_t> hp(static_cast<size_t>(pack_stride * S)),
        hs(static_cast<size_t>(scale_stride * S));
    for (int e = 0; e < S; ++e) {
        uint8_t *p = hp.data() + static_cast<size_t>(e) * pack_stride;
        uint8_t *sc = hs.data() + static_cast<size_t>(e) * scale_stride;
        for (int64_t i = 0; i < pack_stride; ++i)
            p[i] = static_cast<uint8_t>((i * 23 + e * 41) & 0xFF);
        for (int64_t i = 0; i < scale_stride; ++i)
            sc[i] = adversarial ? static_cast<uint8_t>((i % 2) ? 0xf0 : 0x08)
                                : static_cast<uint8_t>(
                                      128 + ((i + 11 * e) % 48));
    }
    std::vector<lsxcommon::qwen_moe::QwenMoeGvSlotDev> slots(S);
    for (int k = 0; k < S; ++k) {
        slots[k].dev_slot = k;
        slots[k].gs1 = static_cast<float>(1 << (k % 4)) / 256.0f;
        slots[k].gs3 = static_cast<float>(1 << ((k + 1) % 4)) / 256.0f;
        slots[k].gs2 = static_cast<float>(1 << ((k + 2) % 4)) / 256.0f;
        slots[k].coeff = k % 2 ? -0.5f : 0.75f;
    }
    std::vector<__half> hx(H);
    for (int i = 0; i < H; ++i)
        hx[i] = __float2half(float((i % 7) - 3) * 0.25f);

    uint8_t *d_pack, *d_scale;
    lsxcommon::qwen_moe::QwenMoeGvSlotDev *d_tbl = nullptr;
    cudaMalloc(&d_pack, hp.size());
    cudaMalloc(&d_scale, hs.size());
    cudaMalloc(&d_tbl, sizeof(*slots.data()) * S);
    cudaMemcpy(d_pack, hp.data(), hp.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale, hs.data(), hs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tbl, slots.data(), sizeof(*slots.data()) * S,
               cudaMemcpyHostToDevice);

    __half *dx, *ygu, *hbuf;
    float *acc_f32;
    cudaMalloc(&dx, static_cast<size_t>(H) * 2);
    cudaMalloc(&ygu, static_cast<size_t>(S) * 2 * I * 2);
    cudaMalloc(&hbuf, static_cast<size_t>(S) * I * 2);
    cudaMalloc(&acc_f32, static_cast<size_t>(H) * 4);
    cudaMemcpy(dx, hx.data(), static_cast<size_t>(H) * 2,
               cudaMemcpyHostToDevice);
    std::vector<float> acc_seed(H);
    for (int i = 0; i < H; ++i) acc_seed[i] = 0.125f * (i % 5);
    cudaMemcpy(acc_f32, acc_seed.data(), static_cast<size_t>(H) * 4,
               cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::gemma::RunRoutedGemvBatchSeg(
                nullptr, d_pack, d_scale, pack_stride, scale_stride, d_tbl,
                S,
                /*gate*/ 0, 0,
                /*up*/ b1 + b2, s1b + s2b,
                /*down*/ b1, s1b,
                /*N_gate_up=*/I, /*K_gate_up=*/H,
                /*N_down=*/H, /*K_down=*/I,
                /*geglu=*/true, dx, ygu, hbuf, acc_f32) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    // ── host reference ────────────────────────────────────────────────────
    auto ref_stage = [&](const uint8_t *pbase, const uint8_t *sbase,
                         float gs, const __half *xin, float *out, int Kd,
                         int Nr, int64_t poff, int64_t soff) {
        for (int n = 0; n < Nr; ++n) out[n] = 0.0f;
        for (int n = 0; n < Nr; ++n)
            for (int k = 0; k < Kd; ++k) {
                const uint8_t byte =
                    pbase[poff + (static_cast<int64_t>(n) * Kd + k) / 2];
                const uint8_t nib =
                    (k & 1) ? static_cast<uint8_t>(byte >> 4)
                            : static_cast<uint8_t>(byte & 0x0F);
                const float scl =
                    E4M3ScaleToFloatHost(
                        sbase[soff +
                              (static_cast<int64_t>(n) * Kd + k) / 16]) *
                    gs;
                out[n] += __half2float(xin[k]) * kE2M1TestTable[nib] * scl;
            }
    };

    auto expect_slot = [&](int k, std::vector<float> *g_out,
                           std::vector<float> *u_out,
                           std::vector<float> *y_out) {
        const uint8_t *pk = hp.data() +
            static_cast<size_t>(slots[k].dev_slot) * pack_stride;
        const uint8_t *ps = hs.data() +
            static_cast<size_t>(slots[k].dev_slot) * scale_stride;
        std::vector<__half> h_in(I);
        ref_stage(pk, ps, slots[k].gs1, hx.data(), g_out->data(), H, I, 0, 0);
        ref_stage(pk, ps, slots[k].gs3, hx.data(), u_out->data(), H, I,
                  b1 + b2, s1b + s2b);
        for (int j = 0; j < I; ++j)
            h_in[j] = __float2half(GeluTanhHost(
                __half2float(__float2half((*g_out)[j]))) *
                __half2float(__float2half((*u_out)[j])));
        ref_stage(pk, ps, slots[k].gs2, h_in.data(), y_out->data(), I, H,
                  b1, s1b);
    };

    std::vector<__half> got_gu(static_cast<size_t>(S) * 2 * I);
    cudaMemcpy(got_gu.data(), ygu, got_gu.size() * 2,
               cudaMemcpyDeviceToHost);
    for (int k = 0; k < S; ++k) {
        std::vector<float> g(I), u(I), y(H);
        expect_slot(k, &g, &u, &y);
        for (int j = 0; j < I; ++j) {
            CHECK(std::abs(__half2float(got_gu[
                      static_cast<size_t>(k) * 2 * I + j]) -
                  __half2float(__float2half(g[j]))) <
                  2e-3f * std::abs(g[j]) + 2e-3f);
            CHECK(std::abs(__half2float(got_gu[
                      static_cast<size_t>(k) * 2 * I + I + j]) -
                  __half2float(__float2half(u[j]))) <
                  2e-3f * std::abs(u[j]) + 2e-3f);
        }
    }
    std::vector<float> got_acc(H), exp_acc(acc_seed);
    cudaMemcpy(got_acc.data(), acc_f32, static_cast<size_t>(H) * 4,
               cudaMemcpyDeviceToHost);
    for (int k = 0; k < S; ++k) {
        std::vector<float> g(I), u(I), y(H);
        expect_slot(k, &g, &u, &y);
        for (int r = 0; r < H; ++r) exp_acc[r] += slots[k].coeff * y[r];
    }
    for (int r = 0; r < H; ++r) {
        INFO("r=" << r << " ref=" << exp_acc[r] << " got=" << got_acc[r]);
        CHECK(std::abs(got_acc[r] - exp_acc[r]) <
              2e-3f * std::abs(exp_acc[r]) + 1e-3f);
    }
    if (acc_out) *acc_out = got_acc;

    cudaFree(d_pack); cudaFree(d_scale); cudaFree(d_tbl);
    cudaFree(dx); cudaFree(ygu); cudaFree(hbuf); cudaFree(acc_f32);
}

} // namespace

TEST_CASE("Gemma routed GEMV batch parity vs host reference",
          "[qwen][gemm][nvfp4][gemv][gemma]")
{
    RunGemmaBatch(128, 64, 4, false, nullptr);
}

TEST_CASE("Gemma routed GEMV chunk-scale split (adversarial)",
          "[qwen][gemm][nvfp4][gemv][gemma]")
{
    RunGemmaBatch(96, 96, 2, true, nullptr);
}

TEST_CASE("Gemma routed GEMV deterministic across launches",
          "[qwen][gemm][nvfp4][gemv][gemma]")
{
    std::vector<float> a1, a2;
    RunGemmaBatch(64, 32, 2, false, &a1);
    RunGemmaBatch(64, 32, 2, false, &a2);
    REQUIRE(a1.size() == a2.size());
    CHECK(std::memcmp(a1.data(), a2.data(),
                      a1.size() * sizeof(float)) == 0);
}
