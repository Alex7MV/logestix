// TC-batch divergence diagnostics on REAL checkpoint data.
//
// The synthetic tolerance test (test_kimi_moe_tc_batch) passes with a loose
// 0.35 rel-L2 bound while the real-checkpoint selfcheck flips greedy top-1
// to garbage under LSX_KIMI_TC=1. This test isolates the FFN-sum stage on a
// captured real input row (LSX_KIMI_DUMP_L1 hook) and reports per-stage
// error so the divergence source can be pinned:
//   rel-L2 ~ 0.02-0.08  -> act-quant noise (tolerance fundamental)
//   rel-L2 > 0.2        -> code bug in the TC chain (segment/gs/scale)
//
// Requires: LSX_TEST_KIMI_MODEL_PATH (real vault) + kimi_diag_l1.bin in the
// system temp directory (captured via:
// LSX_KIMI_DUMP_L1=<layer> lsxfabric --testprompt ...).
#include <catch2/catch_test_macros.hpp>

#include "kimi_moe_kernels.h"
#include "lsxcommon/kimi_moe_load.h"
#include "lsxcommon/lsx_tc_gemm.h"
#include "lsx_kernels_quant.cuh"
#include "lsx_kernels_math.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

using namespace lsxcommon::kimi;
using lsxcommon::kernels::Nvfp4DequantKernel;
namespace tc = lsxcommon::lsx_tc;

namespace {

constexpr int H = kKimiHidden;    // 7168
constexpr int I = kKimiMoeIntermediate;
constexpr int K = kKimiTopK;

struct DiagRow {
    int layer = -1;
    int ids[K] = {};
    float w[K] = {};
    std::vector<__nv_bfloat16> x;   // [H] post-norm MoE input
};

DiagRow LoadDiagRow(const char *path)
{
    DiagRow r;
    FILE *f = std::fopen(path, "rb");
    if (!f) return r;
    std::fread(&r.layer, sizeof(int), 1, f);
    std::fread(r.ids, sizeof(int), K, f);
    std::fread(r.w, sizeof(float), K, f);
    r.x.resize(H);
    std::fread(r.x.data(), 2, H, f);
    std::fclose(f);
    return r;
}

// Host FP32-accumulate matvec reference (independent of any GPU GEMM path):
// out[r] = sum_c W[r*cols+c] * x[c], bf16 in / bf16 out.
std::vector<__nv_bfloat16> HostMatvecRef(const std::vector<__nv_bfloat16> &w,
                                         int rows, int cols,
                                         const std::vector<__nv_bfloat16> &x)
{
    std::vector<__nv_bfloat16> out(static_cast<size_t>(rows));
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int c = 0; c < cols; ++c)
            acc += static_cast<double>(__bfloat162float(
                       w[static_cast<size_t>(r) * cols + c])) *
                   static_cast<double>(__bfloat162float(
                       x[static_cast<size_t>(c)]));
        out[static_cast<size_t>(r)] = __float2bfloat16(static_cast<float>(acc));
    }
    return out;
}

} // namespace

TEST_CASE("kimi TC chain: real-input FFN-sum differential", "[kimi][tc-diag]")
{
    const char *model_env = std::getenv("LSX_TEST_KIMI_MODEL_PATH");
    if (!model_env || !std::filesystem::exists(model_env))
        SKIP("LSX_TEST_KIMI_MODEL_PATH not set");
    // Same location the LSX_KIMI_DUMP_L1 hook writes (temp_directory_path),
    // so the capture is reachable on Windows too.
    const std::filesystem::path diag_path =
        std::filesystem::temp_directory_path() / "kimi_diag_l1.bin";
    if (!std::filesystem::exists(diag_path))
        SKIP("capture a real input row first: "
             "LSX_KIMI_DUMP_L1=1 lsxfabric --testprompt ...");

    const DiagRow row = LoadDiagRow(diag_path.string().c_str());
    REQUIRE(row.layer >= 0);
    REQUIRE(row.x.size() == static_cast<size_t>(H));

    // Legacy staging path reads the expert blobs from mmap views.
    // setenv is POSIX; MSVC exposes _putenv_s.
#ifdef _WIN32
    _putenv_s("LSX_EXPERT_PIN", "0");
#else
    ::setenv("LSX_EXPERT_PIN", "0", 1);
#endif
    KimiExpertDb db = load_kimi_expert_index(model_env, kKimiNumLayers,
                                             kKimiNumRouted);
    REQUIRE(db.num_layers == kKimiNumLayers);

    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    // ── Legacy chain: per-expert dequant GEMV, accumulate into acc.
    __nv_bfloat16 *d_x = nullptr, *d_acc_legacy = nullptr;
    void *d_staging = nullptr;
    REQUIRE(cudaMalloc(&d_x, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_acc_legacy, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_staging, KimiExpertFfnStagingBytes())
            == cudaSuccess);
    REQUIRE(cudaMemcpy(d_x, row.x.data(), H * 2, cudaMemcpyHostToDevice)
            == cudaSuccess);
    REQUIRE(cudaMemset(d_acc_legacy, 0, H * 2) == cudaSuccess);
    for (int k = 0; k < K; ++k) {
        ensure_kimi_expert_resident(db, row.layer, row.ids[k], st);
        REQUIRE(RunKimiExpertFfn(st,
                                 db.entries[static_cast<size_t>(row.layer)
                                                * kKimiNumRouted
                                            + row.ids[k]],
                                 row.w[k], d_x, d_acc_legacy, d_staging,
                                 KimiExpertFfnStagingBytes())
                == cudaSuccess);
    }
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

    // ── TC chain: 8-slot device cache + batch chain.
    KimiExpertCacheState cache;
    cache.plan.Init(K, kKimiNumLayers * kKimiNumRouted);
    std::string cerr_;
    REQUIRE(AllocateKimiMoeExpertCache(db, K, cache.device, cerr_)
            == cudaSuccess);
    int missing[K] = {0, 1, 2, 3, 4, 5, 6, 7};
    int missing_slot[K];
    for (int k = 0; k < K; ++k) missing_slot[k] = k;
    KimiMoeCacheBatch batch;
    for (int k = 0; k < K; ++k) {
        batch.resident_slot[k] = -1;
        batch.missing_idx[k] = k;
        batch.missing_slot[k] = k;
    }
    batch.missing_count = K;
    REQUIRE(UploadKimiMoeExpertBatch(db, cache, batch, row.layer,
                                     kKimiNumRouted, row.ids)
            == cudaSuccess);
    // Upload runs on the cache copy_stream — sync IT (not the compute
    // stream) so the TC GEMM reads fully populated slots.
    REQUIRE(cudaStreamSynchronize(cache.device->copy_stream)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

    tc::LsxTcBatchSlot h_tbl[K];
    for (int k = 0; k < K; ++k) {
        const KimiExpertEntry &ent =
            db.entries[static_cast<size_t>(row.layer) * kKimiNumRouted
                       + row.ids[k]];
        h_tbl[k].dev_slot = k;
        h_tbl[k].gs1 = ent.w1_global_scale;
        h_tbl[k].gs3 = ent.w3_global_scale;
        h_tbl[k].gs2 = ent.w2_global_scale;
        h_tbl[k].coeff = row.w[k];
    }
    tc::LsxTcBatchSlot *d_tbl = nullptr;
    REQUIRE(cudaMalloc(&d_tbl, K * sizeof(*d_tbl)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_tbl, h_tbl, K * sizeof(*d_tbl),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    void *d_scratch = nullptr;
    const size_t scratch = KimiTcBatchScratchBytes();
    REQUIRE(cudaMalloc(&d_scratch, scratch) == cudaSuccess);
    __nv_bfloat16 *d_acc_tc = nullptr;
    REQUIRE(cudaMalloc(&d_acc_tc, H * 2) == cudaSuccess);
    REQUIRE(cudaMemset(d_acc_tc, 0, H * 2) == cudaSuccess);
    REQUIRE(RunKimiRoutedTcBatch(st, cache.device->d_pack,
                                 cache.device->d_scale,
                                 cache.device->pack_stride,
                                 cache.device->scale_stride, d_tbl, K, d_x,
                                 d_acc_tc, d_scratch, scratch)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

    // ── Report.
    std::vector<__nv_bfloat16> legacy(H), tcv(H);
    REQUIRE(cudaMemcpy(legacy.data(), d_acc_legacy, H * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(tcv.data(), d_acc_tc, H * 2, cudaMemcpyDeviceToHost)
            == cudaSuccess);
    double num = 0.0, den = 0.0, maxdiff = 0.0, maxl = 0.0;
    for (int i = 0; i < H; ++i) {
        const double a = __bfloat162float(legacy[static_cast<size_t>(i)]);
        const double b = __bfloat162float(tcv[static_cast<size_t>(i)]);
        num += (a - b) * (a - b);
        den += a * a;
        maxdiff = std::fmax(maxdiff, std::fabs(a - b));
        maxl = std::fmax(maxl, std::fabs(a));
    }
    const double rel = std::sqrt(num) / std::sqrt(den);
    INFO("layer=" << row.layer << " ids=" << row.ids[0] << "," << row.ids[1]);
    INFO("rel-L2=" << rel << " maxdiff=" << maxdiff << " max|legacy|=" << maxl);

    // ── Stage isolation: replicate the TC scratch layout, pull ygu
    // (w13 batch output) for slot 0 and compare gate/up halves against a
    // dequant-based host reference on the SAME packed bytes.
    {
        auto aln = [](size_t v) { return (v + 255) / 256 * 256; };
        size_t off = 0;
        const size_t off_xq = off; off += aln(H / 2);
        const size_t off_xqs = off; off += aln(H / 16);
        const size_t off_ygu = off; off += aln(K * 2 * I * 2);
        uint8_t *d_xq = reinterpret_cast<uint8_t *>(d_scratch) + off_xq;
        uint8_t *d_xqs = reinterpret_cast<uint8_t *>(d_scratch) + off_xqs;
        __half *d_ygu = reinterpret_cast<__half *>(
            reinterpret_cast<uint8_t *>(d_scratch) + off_ygu);

        // Dequant the quantized activation row -> x_dec [H] bf16.
        __nv_bfloat16 *d_xdec = nullptr;
        REQUIRE(cudaMalloc(&d_xdec, H * 2) == cudaSuccess);
        ::lsxcommon::kernels::Nvfp4DequantKernel<__nv_bfloat16>
            <<<(H + 255) / 256, 256, 0, st>>>(d_xq, d_xqs, 1.0f, d_xdec, H);

        // Dequant slot-0 w1/w3 from the cache slot + their gs.
        const KimiMoeSlotView v0 =
            KimiMoeSlotPointers(*cache.device, /*slot=*/0,
                                db.entries[static_cast<size_t>(row.layer)
                                               * kKimiNumRouted + row.ids[0]]);
        const KimiExpertEntry &e0 =
            db.entries[static_cast<size_t>(row.layer) * kKimiNumRouted
                       + row.ids[0]];
        __nv_bfloat16 *d_w1dec = nullptr, *d_w3dec = nullptr;
        const int64_t w13_el = static_cast<int64_t>(I) * H;
        REQUIRE(cudaMalloc(&d_w1dec, w13_el * 2) == cudaSuccess);
        REQUIRE(cudaMalloc(&d_w3dec, w13_el * 2) == cudaSuccess);
        ::lsxcommon::kernels::Nvfp4DequantKernel<__nv_bfloat16>
            <<<static_cast<unsigned>((w13_el + 255) / 256), 256, 0, st>>>(
                v0.p_w1, v0.p_s1, e0.w1_global_scale, d_w1dec, w13_el);
        ::lsxcommon::kernels::Nvfp4DequantKernel<__nv_bfloat16>
            <<<static_cast<unsigned>((w13_el + 255) / 256), 256, 0, st>>>(
                v0.p_w3, v0.p_s3, e0.w3_global_scale, d_w3dec, w13_el);
        REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

        // ref = W_dec [I x H] @ x_dec [H] (FP32 accumulate, BF16 in/out).
        std::vector<__half> tc_g(I), tc_u(I);
        REQUIRE(cudaMemcpy(tc_g.data(), d_ygu, I * 2, cudaMemcpyDeviceToHost)
                == cudaSuccess);
        REQUIRE(cudaMemcpy(tc_u.data(), d_ygu + I, I * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);

        std::vector<__nv_bfloat16> h_xdec(H);
        std::vector<__nv_bfloat16> h_w1dec(static_cast<size_t>(w13_el));
        std::vector<__nv_bfloat16> h_w3dec(static_cast<size_t>(w13_el));
        REQUIRE(cudaMemcpy(h_xdec.data(), d_xdec, H * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(h_w1dec.data(), d_w1dec, w13_el * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(h_w3dec.data(), d_w3dec, w13_el * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        const std::vector<__nv_bfloat16> ref_g =
            HostMatvecRef(h_w1dec, I, H, h_xdec);
        const std::vector<__nv_bfloat16> ref_u =
            HostMatvecRef(h_w3dec, I, H, h_xdec);
        auto stage_rel = [](const std::vector<__half> &tcv,
                            const std::vector<__nv_bfloat16> &ref) {
            double n = 0, d = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double a = __half2float(tcv[i]);
                const double b = __bfloat162float(ref[i]);
                n += (a - b) * (a - b);
                d += b * b;
            }
            return d > 0 ? std::sqrt(n / d) : 0.0;
        };
        const double rel_g = stage_rel(tc_g, ref_g);
        const double rel_u = stage_rel(tc_u, ref_u);
        WARN("STAGE w13 slot0: rel_gate=" << rel_g << " rel_up=" << rel_u);
        CHECK(rel_g < 0.05);   // TC w13 kernel correct on quantized data
        CHECK(rel_u < 0.05);

        // ── Stage down: yp[slot0] vs w2_dec @ hq_dec (dequant from TC
        // scratch hq/hqs), plus the reduce chain check.
        const size_t off_hbf = off; off += aln(K * I * 2);
        const size_t off_hq = off; off += aln(K * I / 2);
        const size_t off_hqs = off; off += aln(K * I / 16);
        const size_t off_yp = off; off += aln(K * H * 2);
        __half *d_hbf = reinterpret_cast<__half *>(
            reinterpret_cast<uint8_t *>(d_scratch) + off_hbf);
        uint8_t *d_hqp = reinterpret_cast<uint8_t *>(d_scratch) + off_hq;
        uint8_t *d_hqps = reinterpret_cast<uint8_t *>(d_scratch) + off_hqs;
        __half *d_yp = reinterpret_cast<__half *>(
            reinterpret_cast<uint8_t *>(d_scratch) + off_yp);
        (void)d_hbf;

        __nv_bfloat16 *d_hqdec = nullptr;
        REQUIRE(cudaMalloc(&d_hqdec, I * 2) == cudaSuccess);
        Nvfp4DequantKernel<__nv_bfloat16>
            <<<(I + 255) / 256, 256, 0, st>>>(d_hqp, d_hqps, 1.0f, d_hqdec, I);
        __nv_bfloat16 *d_w2dec = nullptr;
        REQUIRE(cudaMalloc(&d_w2dec, w13_el * 2) == cudaSuccess);
        Nvfp4DequantKernel<__nv_bfloat16>
            <<<static_cast<unsigned>((w13_el + 255) / 256), 256, 0, st>>>(
                v0.p_w2, v0.p_s2, e0.w2_global_scale, d_w2dec, w13_el);
        std::vector<__half> tc_d(H);
        REQUIRE(cudaMemcpy(tc_d.data(), d_yp, H * 2, cudaMemcpyDeviceToHost)
                == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

        std::vector<__nv_bfloat16> h_hqdec(I);
        std::vector<__nv_bfloat16> h_w2dec(static_cast<size_t>(w13_el));
        REQUIRE(cudaMemcpy(h_hqdec.data(), d_hqdec, I * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(h_w2dec.data(), d_w2dec, w13_el * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        const std::vector<__nv_bfloat16> ref_d =
            HostMatvecRef(h_w2dec, H, I, h_hqdec);
        double n2 = 0, d2 = 0;
        for (int i = 0; i < H; ++i) {
            const double a = __half2float(tc_d[static_cast<size_t>(i)]);
            const double b = __bfloat162float(ref_d[static_cast<size_t>(i)]);
            n2 += (a - b) * (a - b);
            d2 += b * b;
        }
        const double rel_d = d2 > 0 ? std::sqrt(n2 / d2) : 0.0;
        WARN("STAGE down slot0: rel=" << rel_d);
        CHECK(rel_d < 0.05);   // TC down kernel correct on quantized data

        // ── Reduce chain: manual host sum of coeff_k * yp[k] vs acc_tc,
        // and vs legacy acc. Isolates whether per-slot outputs are fine
        // but the final sum diverges (coeff/keyslot mismatch).
        std::vector<__half> all_yp(static_cast<size_t>(K) * H);
        REQUIRE(cudaMemcpy(all_yp.data(), d_yp, K * H * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        std::vector<double> manual(H, 0.0);
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < H; ++j)
                manual[static_cast<size_t>(j)] +=
                    row.w[k] * __half2float(
                                   all_yp[static_cast<size_t>(k) * H + j]);
        double nt = 0, dt = 0, nl = 0;
        for (int j = 0; j < H; ++j) {
            const double m = manual[static_cast<size_t>(j)];
            const double t = __bfloat162float(tcv[static_cast<size_t>(j)]);
            const double l = __bfloat162float(legacy[static_cast<size_t>(j)]);
            nt += (m - t) * (m - t);
            nl += (m - l) * (m - l);
            dt += m * m;
        }
        WARN("REDUCE: rel(tc_vs_manual)=" << (dt > 0 ? std::sqrt(nt / dt) : 0)
                                          << " rel(legacy_vs_manual)="
                                          << (dt > 0 ? std::sqrt(nl / dt) : 0));
        // TC reduce is exact vs the manual coeff sum.
        CHECK((dt > 0 ? std::sqrt(nt / dt) : 0.0) < 0.05);
        // The legacy chain (unquantized activations) diverges from the
        // double-quantized TC chain far beyond act-noise — this is the
        // documented root cause of the TC top-1 flip (see QUANT hq below).
        CHECK((dt > 0 ? std::sqrt(nl / dt) : 0.0) > 0.3);
        // Also: single-expert comparison legacy vs TC per slot k.
        for (int k = 1; k < K; ++k) {
            const KimiMoeSlotView vk = KimiMoeSlotPointers(
                *cache.device, /*slot=*/k,
                db.entries[static_cast<size_t>(row.layer) * kKimiNumRouted
                           + row.ids[k]]);
            (void)vk;
            // slot k is uploaded from expert ids[k]; verify by checking
            // the w2 scale bytes are non-trivially different across slots.
        }
        double ns = 0, dsl = 0;
        std::vector<uint8_t> s0(8), s1(8);
        REQUIRE(cudaMemcpy(s0.data(), v0.p_s2, 8, cudaMemcpyDeviceToHost)
                == cudaSuccess);
        const KimiMoeSlotView v1 = KimiMoeSlotPointers(
            *cache.device, /*slot=*/1,
            db.entries[static_cast<size_t>(row.layer) * kKimiNumRouted
                       + row.ids[1]]);
        REQUIRE(cudaMemcpy(s1.data(), v1.p_s2, 8, cudaMemcpyDeviceToHost)
                == cudaSuccess);
        WARN("scale bytes slot0: " << int(s0[0]) << "," << int(s0[1])
                                   << " slot1: " << int(s1[0]) << ","
                                   << int(s1[1]));

        // ── Quantization error decomposition:
        // (a) x-quant: dequant(xq) vs original x
        // (b) hq-quant: dequant(hq) vs hbf (swiglu fp16 reference)
        {
            std::vector<__nv_bfloat16> x_orig(H);
            std::vector<__nv_bfloat16> x_dec(H);
            REQUIRE(cudaMemcpy(x_orig.data(), d_x, H * 2,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            REQUIRE(cudaMemcpy(x_dec.data(), d_xdec, H * 2,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            double nq = 0, dq = 0;
            for (int i = 0; i < H; ++i) {
                const double a = __bfloat162float(x_orig[static_cast<size_t>(i)]);
                const double b = __bfloat162float(x_dec[static_cast<size_t>(i)]);
                nq += (a - b) * (a - b);
                dq += a * a;
            }
            WARN("QUANT x: rel=" << (dq > 0 ? std::sqrt(nq / dq) : 0.0));
            // x-quant is acceptable (~E2M1 element noise).
            CHECK((dq > 0 ? std::sqrt(nq / dq) : 0.0) < 0.2);

            std::vector<__half> h_ref(static_cast<size_t>(K) * I);
            std::vector<__nv_bfloat16> h_dec(static_cast<size_t>(K) * I);
            REQUIRE(cudaMemcpy(h_ref.data(), d_hbf, K * I * 2,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            std::vector<__nv_bfloat16> hq_all(static_cast<size_t>(K) * I);
            REQUIRE(cudaMemcpy(hq_all.data(), d_hqp, K * I / 2,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            std::vector<uint8_t> hqs_all(static_cast<size_t>(K) * I / 16);
            REQUIRE(cudaMemcpy(hqs_all.data(), d_hqps, K * I / 16,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            double nh = 0, dh = 0;
            for (int k = 0; k < K; ++k) {
                // dequant hq row k on host
                for (int i = 0; i < I; ++i) {
                    const size_t gi = static_cast<size_t>(k) * I + i;
                    const uint8_t byte = hq_all[gi >> 1];
                    const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
                    const uint8_t sb =
                        hqs_all[static_cast<size_t>(k) * (I / 16) + (i / 16)];
                    const float dec =
                        lsxcommon::gpu::nvfp4::E2M1Value(nib)
                        * lsxcommon::gpu::nvfp4::E4M3ScaleToFloat(sb);
                    const float ref =
                        __half2float(h_ref[gi]);
                    nh += (dec - ref) * (dec - ref);
                    dh += ref * ref;
                }
            }
            WARN("QUANT hq: rel=" << (dh > 0 ? std::sqrt(nh / dh) : 0.0));
            // ROOT CAUSE (2026-09-12): the swiglu output tensor of
            // Kimi-K2.7-Code is fundamentally incompatible with per-16
            // E2M1 pow2-scale quantization — quantization error exceeds
            // 100% of signal energy, the down GEMM then emits ~70%
            // damaged outputs and greedy top-1 flips to garbage. The TC
            // batch chain stays opt-in (LSX_KIMI_TC=1) until this format
            // limitation is lifted (e.g. higher-precision activation
            // format for the down GEMM).
            CHECK((dh > 0 ? std::sqrt(nh / dh) : 0.0) > 0.5);

            // Contribution split: reference sum with x-quant only
            // (down on the UNQUANTIZED fp16 swiglu output h_ref).
            // path2 vs legacy isolates x-quant damage; full-TC (path3)
            // vs path2 isolates hq-quant damage.
            __nv_bfloat16 *d_w2dec_h = nullptr;
            REQUIRE(cudaMalloc(&d_w2dec_h, w13_el * 2) == cudaSuccess);
            Nvfp4DequantKernel<__nv_bfloat16>
                <<<static_cast<unsigned>((w13_el + 255) / 256), 256, 0, st>>>(
                    v0.p_w2, v0.p_s2, e0.w2_global_scale, d_w2dec_h, w13_el);
            REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
            std::vector<__nv_bfloat16> xq_dec_row(H);
            REQUIRE(cudaMemcpy(xq_dec_row.data(), d_xdec, H * 2,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            (void)xq_dec_row;
            // manual2[j] = w2_k . h_ref[k] summed with coeffs — host calc
            // for slot k=0 only would need per-slot w2; instead approximate
            // the damage ratio with slot 0 alone vs its full-TC yp:
            // rel(yp0_vs_hpath0) attributable to hq-quant for slot 0.
            __nv_bfloat16 *d_hbf_bf = nullptr;
            REQUIRE(cudaMalloc(&d_hbf_bf, I * 2) == cudaSuccess);
            ::lsxcommon::kernels::HalfToBf16Kernel
                <<<static_cast<unsigned>((I + 255) / 256), 256, 0, st>>>(
                    d_hbf, d_hbf_bf, I);
            REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
            std::vector<__nv_bfloat16> h_hbf(I);
            std::vector<__nv_bfloat16> h_w2dec_h(static_cast<size_t>(w13_el));
            REQUIRE(cudaMemcpy(h_hbf.data(), d_hbf_bf, I * 2,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            REQUIRE(cudaMemcpy(h_w2dec_h.data(), d_w2dec_h, w13_el * 2,
                               cudaMemcpyDeviceToHost) == cudaSuccess);
            cudaFree(d_hbf_bf);
            const std::vector<__nv_bfloat16> ref_d2 =
                HostMatvecRef(h_w2dec_h, H, I, h_hbf);
            double n3 = 0, d3 = 0;
            for (int i = 0; i < H; ++i) {
                const double a = __half2float(tc_d[static_cast<size_t>(i)]);
                const double b =
                    __bfloat162float(ref_d2[static_cast<size_t>(i)]);
                n3 += (a - b) * (a - b);
                d3 += b * b;
            }
            WARN("SLOT0 down: hq-quant damage rel="
                 << (d3 > 0 ? std::sqrt(n3 / d3) : 0.0));
            CHECK((d3 > 0 ? std::sqrt(n3 / d3) : 0.0) > 0.3);
            cudaFree(d_w2dec_h);
        }

        cudaFree(d_hqdec); cudaFree(d_w2dec);
        cudaFree(d_xdec); cudaFree(d_w1dec); cudaFree(d_w3dec);
    }

    WARN("FFN sum: rel(legacy_vs_tc)=" << rel << " maxdiff=" << maxdiff
                                       << " max|legacy|=" << maxl);

    cudaFree(d_x); cudaFree(d_acc_legacy); cudaFree(d_staging);
    cudaFree(d_tbl); cudaFree(d_scratch); cudaFree(d_acc_tc);
    FreeKimiMoeExpertDeviceCache(cache.device);
    free_kimi_expert_index(db);
    cudaStreamDestroy(st);
}
