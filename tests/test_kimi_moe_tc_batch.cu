// TDD — Kimi batched TC routed chain vs the legacy per-expert GEMV chain.
// Parity is TOLERANCE-class (TC split-K divergence, spec amendment): the
// legacy chain remains the reference for correctness gates (selfcheck),
// this test pins the TC chain to it within NVFP4 quantization tolerance.
#include <catch2/catch_test_macros.hpp>

#include "kimi_moe_kernels.h"
#include "lsxcommon/kimi_moe_load.h"
#include "lsxcommon/lsx_tc_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace lsxcommon::kimi;
namespace tc = lsxcommon::lsx_tc;

constexpr int I = kKimiMoeIntermediate;   // 2048
constexpr int H = kKimiHidden;            // 7168
constexpr int K = kKimiTopK;              // 8

struct HostProj {
    std::vector<uint8_t> packed;   // [rows*cols/2]
    std::vector<uint8_t> scales;   // [rows*cols/16]
    float global_scale = 0.75f;
};

HostProj MakeProj(int seed, int64_t rows, int64_t cols)
{
    HostProj p;
    const int64_t numel = rows * cols;
    p.packed.assign(static_cast<size_t>(numel / 2), 0);
    p.scales.assign(static_cast<size_t>(numel / 16), 0);
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_int_distribution<int> sc(-7, -4);
    for (int64_t i = 0; i < numel; ++i) {
        const int v = nib(rng);
        p.packed[static_cast<size_t>(i >> 1)] |=
            static_cast<uint8_t>(v << ((i & 1) ? 4 : 0));
        if ((i & 15) == 0)
            p.scales[static_cast<size_t>(i >> 4)] =
                static_cast<uint8_t>(((sc(rng) + 7) & 0xF) << 3);
    }
    return p;
}

struct DeviceSlabs {
    uint8_t *d_pack = nullptr;    // [w1|w3|w2] fused slab per slot
    uint8_t *d_scale = nullptr;   // [s1|s3|s2] fused slab per slot
    int64_t pack_stride = 0;
    int64_t scale_stride = 0;
};

} // namespace

TEST_CASE("Kimi batched TC routed chain: tolerance parity vs legacy GEMV",
          "[kimi][tc-batch]")
{
    if (!tc::LsxTcRuntimeSupported()) return;
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    const int64_t el = static_cast<int64_t>(I) * H;
    HostProj w1 = MakeProj(1, I, H), w3 = MakeProj(2, I, H),
             w2 = MakeProj(3, H, I);
    const int64_t w1_b = el / 2, w3_b = el / 2, w2_b = el / 2;
    const int64_t s1_b = el / 16, s3_b = el / 16, s2_b = el / 16;
    auto align_up = [](size_t v) { return (v + 255) / 256 * 256; };
    DeviceSlabs dev;
    dev.pack_stride = align_up(w1_b) + align_up(w3_b) + align_up(w2_b);
    dev.scale_stride = align_up(s1_b) + align_up(s3_b) + align_up(s2_b);

    // All 8 slots point at the same weights (duplicate-eid adversarial
    // case) with distinct coeffs.
    constexpr int SLOTS = K;
    REQUIRE(cudaMalloc(&dev.d_pack, SLOTS * dev.pack_stride) == cudaSuccess);
    REQUIRE(cudaMalloc(&dev.d_scale, SLOTS * dev.scale_stride) ==
            cudaSuccess);
    {
        std::vector<uint8_t> hpack(dev.pack_stride, 0),
            hscale(dev.scale_stride, 0);
        std::memcpy(hpack.data(), w1.packed.data(), w1_b);
        std::memcpy(hpack.data() + align_up(w1_b), w3.packed.data(), w3_b);
        std::memcpy(hpack.data() + align_up(w1_b) + align_up(w3_b),
                    w2.packed.data(), w2_b);
        std::memcpy(hscale.data(), w1.scales.data(), s1_b);
        std::memcpy(hscale.data() + align_up(s1_b), w3.scales.data(), s3_b);
        std::memcpy(hscale.data() + align_up(s1_b) + align_up(s3_b),
                    w2.scales.data(), s2_b);
        for (int s = 0; s < SLOTS; ++s) {
            cudaMemcpy(dev.d_pack + s * dev.pack_stride, hpack.data(),
                       dev.pack_stride, cudaMemcpyHostToDevice);
            cudaMemcpy(dev.d_scale + s * dev.scale_stride, hscale.data(),
                       dev.scale_stride, cudaMemcpyHostToDevice);
        }
    }

    // Hidden row: bf16 values with alternating per-16 block magnitudes.
    std::vector<__nv_bfloat16> hx(H);
    for (int i = 0; i < H; ++i)
        hx[static_cast<size_t>(i)] = __float2bfloat16(
            ((i / 16) % 2 ? 1.0f : 0.25f) *
            std::sin(static_cast<float>(i) * 0.7f));
    __nv_bfloat16 *d_x;
    REQUIRE(cudaMalloc(&d_x, H * 2) == cudaSuccess);
    cudaMemcpy(d_x, hx.data(), H * 2, cudaMemcpyHostToDevice);

    const float coeffs[K] = {0.31f, 0.02f, 0.28f, 0.01f,
                             0.15f, 0.09f, 0.11f, 0.03f};
    std::vector<tc::LsxTcBatchSlot> tbl(K);
    for (int k = 0; k < K; ++k)
        tbl[static_cast<size_t>(k)] = {0 /*dev_slot*/, 0.75f, 0.75f, 0.75f,
                                       coeffs[k]};
    tc::LsxTcBatchSlot *d_tbl;
    REQUIRE(cudaMalloc(&d_tbl, tbl.size() * sizeof(tbl[0])) == cudaSuccess);
    cudaMemcpy(d_tbl, tbl.data(), tbl.size() * sizeof(tbl[0]),
               cudaMemcpyHostToDevice);

    // ── Legacy reference: K sequential RunKimiExpertFfnCached against
    // slot 0 (same slab), bf16 accum into d_out_legacy.
    __nv_bfloat16 *d_out_legacy, *d_out_tc;
    REQUIRE(cudaMalloc(&d_out_legacy, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out_tc, H * 2) == cudaSuccess);
    REQUIRE(cudaMemset(d_out_legacy, 0, H * 2) == cudaSuccess);
    {
        KimiExpertEntry se{};
        se.w1_global_scale = se.w2_global_scale = se.w3_global_scale = 0.75f;
        se.w1_bytes = w1_b; se.w3_bytes = w3_b; se.w2_bytes = w2_b;
        se.w1_scales_bytes = s1_b; se.w3_scales_bytes = s3_b;
        se.w2_scales_bytes = s2_b;
        void *d_staging;
        const size_t staging = KimiExpertFfnStagingBytes();
        REQUIRE(cudaMalloc(&d_staging, staging) == cudaSuccess);
        for (int k = 0; k < K; ++k) {
            KimiMoeSlotView v;
            v.p_w1 = dev.d_pack;
            v.p_w3 = dev.d_pack + align_up(w1_b);
            v.p_w2 = v.p_w3 + align_up(w3_b);
            v.p_s1 = dev.d_scale;
            v.p_s3 = dev.d_scale + align_up(s1_b);
            v.p_s2 = v.p_s3 + align_up(s3_b);
            REQUIRE(RunKimiExpertFfnCached(st, se, coeffs[k], d_x,
                                           d_out_legacy, d_staging,
                                           staging, v) == cudaSuccess);
        }
        cudaFree(d_staging);
    }
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

    // ── TC batched path.
    const size_t scratch = KimiTcBatchScratchBytes();
    void *d_scratch;
    REQUIRE(cudaMalloc(&d_scratch, scratch) == cudaSuccess);
    REQUIRE(cudaMemset(d_out_tc, 0, H * 2) == cudaSuccess);
    REQUIRE(RunKimiRoutedTcBatch(st, dev.d_pack, dev.d_scale,
                                 dev.pack_stride, dev.scale_stride, d_tbl, K,
                                 d_x, d_out_tc, d_scratch, scratch) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

    std::vector<__nv_bfloat16> legacy(H), tcv(H);
    // K=1 DIAG: rebuild table with a single slot to isolate.
    cudaMemcpy(legacy.data(), d_out_legacy, H * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(tcv.data(), d_out_tc, H * 2, cudaMemcpyDeviceToHost);

    // Tolerance parity: NVFP4 weight decode is exact for both paths; the
    // divergence is act-quant (E2M1) + split-K order. Relative bound
    // derived from the E2M1 relative step (~1/3 of value) amplified by
    // the <=8-term accumulation: 0.35 rel-L2 on unit-variance outputs.
    double num = 0.0, den = 0.0;
    for (int i = 0; i < H; ++i) {
        const double a = __bfloat162float(legacy[static_cast<size_t>(i)]);
        const double b = __bfloat162float(tcv[static_cast<size_t>(i)]);
        num += (a - b) * (a - b);
        den += a * a;
    }
    double maxl = 0.0, maxt = 0.0;
    for (int i = 0; i < H; ++i) {
        maxl = std::fmax(maxl, std::fabs(__bfloat162float(legacy[static_cast<size_t>(i)])));
        maxt = std::fmax(maxt, std::fabs(__bfloat162float(tcv[static_cast<size_t>(i)])));
    }
    INFO("max|legacy|=" << maxl << " max|tc|=" << maxt);
    const double rel = std::sqrt(num) / std::sqrt(den);
    INFO("relative L2 = " << rel);
    CHECK(rel < 0.35);

    cudaFree(d_scratch); cudaFree(d_tbl); cudaFree(d_out_tc);
    cudaFree(d_out_legacy); cudaFree(d_x);
    cudaFree(dev.d_pack); cudaFree(dev.d_scale);
    cudaStreamDestroy(st);
}
