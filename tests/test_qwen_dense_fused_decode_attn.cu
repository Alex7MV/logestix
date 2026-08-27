// Fused decoder (seq=1) full-attention 3-stage parity vs the split path.
//
// Direction A: the 9-launch seq=1 full-attention chain is fused into 3 kernels
// (qkv -> online-softmax GQA attention -> o_proj). These tests verify the fused
// entry first exists (Task 1), then per-stage and full-layer parity vs the
// existing split path within a FP32 tolerance envelope (Tasks 2-4).
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <bit>
#include <vector>
#include <cmath>

#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/qwen_dense_convert.h"

namespace {

using lsxcommon::qwen::QwenNvfp4Weight;
using lsxcommon::qwen::QwenLayerWeights;
using lsxcommon::qwen::QwenTopology;
using lsxcommon::qwen::RunQwenFusedDecodeAttnKernel;

int64_t nvfp4_packed_bytes(int K, int N) { return (int64_t)K * N / 2; }
int64_t nvfp4_scale_bytes(int K, int N)  { return (int64_t)K * N / 16; }

// Deterministic NVFP4 fill: nibble-packed bytes + E4M3 block scales.
void FillNvfp4(QwenNvfp4Weight &w, int K, int N,
               uint8_t seed_pack, uint8_t seed_scale)
{
    w.rows = K; w.cols = N;
    w.packed_bytes = nvfp4_packed_bytes(K, N);
    w.scale_bytes  = nvfp4_scale_bytes(K, N);
    w.global_scale = 0.5f;
    cudaMalloc(&w.d_packed, (size_t)w.packed_bytes);
    cudaMalloc(&w.d_scales, (size_t)w.scale_bytes);
    std::vector<uint8_t> pk((size_t)w.packed_bytes), sc((size_t)w.scale_bytes);
    for (int64_t i = 0; i < w.packed_bytes; ++i) pk[(size_t)i] = (uint8_t)((i * 7 + seed_pack) & 0xFF);
    for (int64_t i = 0; i < w.scale_bytes; ++i)  sc[(size_t)i]  = (uint8_t)(128 + ((int)(i + seed_scale) % 32));
    cudaMemcpy((void*)w.d_packed, pk.data(), pk.size(), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)w.d_scales, sc.data(), sc.size(), cudaMemcpyHostToDevice);
}

// Build a tiny full-attention layer: NH=2, NKV=1, HD=8, H=16, rotary_dim=4.
QwenLayerWeights MakeTinyFullLayer(int NH, int NKV, int HD, int H, int rotary_dim)
{
    QwenLayerWeights lw;
    lw.is_linear = false;
    lw.loaded = true;
    lw.norm_size = H;
    FillNvfp4(lw.q, H, NH*HD*2, 1, 2);
    FillNvfp4(lw.k, H, NKV*HD, 3, 4);
    FillNvfp4(lw.v, H, NKV*HD, 5, 6);
    FillNvfp4(lw.o, NH*HD, H, 7, 8);
    lw.q_norm_size = HD; lw.k_norm_size = HD;
    cudaMalloc(&lw.d_q_norm, (size_t)HD*2);
    cudaMalloc(&lw.d_k_norm, (size_t)HD*2);
    cudaMalloc(&lw.d_input_norm, (size_t)H*2);
    std::vector<__half> qn(HD), kn(HD), in(H);
    for (int i = 0; i < HD; ++i) { qn[i] = __float2half(0.1f * ((i % 3))); kn[i] = __float2half(0.05f*((i%5))); }
    for (int i = 0; i < H; ++i)  in[i] = __float2half(0.5f + 0.02f * (float)(i % 7));
    cudaMemcpy(lw.d_q_norm, qn.data(), HD*2, cudaMemcpyHostToDevice);
    cudaMemcpy(lw.d_k_norm, kn.data(), HD*2, cudaMemcpyHostToDevice);
    cudaMemcpy(lw.d_input_norm, in.data(), H*2, cudaMemcpyHostToDevice);
    return lw;
}

void FreeTinyLayer(QwenLayerWeights &lw)
{
    lw.q.Free(); lw.k.Free(); lw.v.Free(); lw.o.Free();
    cudaFree(lw.d_q_norm); cudaFree(lw.d_k_norm); cudaFree(lw.d_input_norm);
}

// ── CPU FP64 reference for the full fused (seq=1) decoder layer ────────────
// Mirrors the fused kernels exactly: input RMSNorm (offset form), per-head
// inline NVFP4 q/k/v+gate decode, per-head RMSNorm(q)/(k), inline RoPE
// (rotate-half pairing), KV append, block-wide online-softmax causal GQA, gate,
// o_proj. kvk/kvv are the persistent cache (host, [rows][NKV*HD], mutable).

static const double kE2M1[16] = {
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
};

static double E4M3ScaleToFloatCpu(uint8_t e4m3)
{
    const uint32_t e4 = (e4m3 >> 3) & 0x0Fu;
    if (e4 == 0) return 0.0;
    return std::bit_cast<float>(uint32_t((e4 + 120u) << 23));
}

static double Deq(const std::vector<uint8_t> &pk, const std::vector<uint8_t> &sc,
                  double gs, uint64_t e)
{
    const uint8_t b = pk[(size_t)(e >> 1)];
    const uint8_t nib = (e & 1) ? (b >> 4) : (b & 0x0Fu);
    return kE2M1[nib] * E4M3ScaleToFloatCpu(sc[(size_t)(e >> 4)]) * gs;
}

static void CpuFusedForward(
    int NH, int NKV, int HD, int H, int rotary_dim, double theta, double eps,
    int kv_base,
    const std::vector<double> &xin,
    const std::vector<double> &in_norm,
    const std::vector<double> &qn, const std::vector<double> &kn,
    const std::vector<uint8_t> &qpk, const std::vector<uint8_t> &qsc, double qgs,
    const std::vector<uint8_t> &kpk, const std::vector<uint8_t> &ksc, double kgs,
    const std::vector<uint8_t> &vpk, const std::vector<uint8_t> &vsc, double vgs,
    const std::vector<uint8_t> &opk, const std::vector<uint8_t> &osc, double ogs,
    std::vector<std::vector<double>> &kv_k, // [rows][NKV*HD], mutable
    std::vector<std::vector<double>> &kv_v,
    std::vector<double> &out)
{
    // 1. input RMSNorm (offset form).
    std::vector<double> xn(H);
    {
        double sq = 0.0;
        for (int i = 0; i < H; ++i) sq += xin[i] * xin[i];
        const double inv = 1.0 / std::sqrt(sq / H + eps);
        for (int i = 0; i < H; ++i) xn[i] = xin[i] * inv * (1.0 + in_norm[i]);
    }
    // 2. per-head q/k/v+gate projections + head-norm + rope.
    std::vector<std::vector<double>> q(NH, std::vector<double>(HD));
    std::vector<std::vector<double>> gate(NH, std::vector<double>(HD));
    std::vector<std::vector<double>> keys(NKV, std::vector<double>(HD));
    std::vector<std::vector<double>> vals(NKV, std::vector<double>(HD));
    for (int h = 0; h < NH; ++h) {
        const int kvh = h / (NH / NKV);
        for (int d = 0; d < HD; ++d) {
            const int qcol = h * HD + d;
            const int gcol = NH * HD + h * HD + d;
            const int kcol = kvh * HD + d;
            double qv = 0, gv = 0, kv_ = 0, vv = 0;
            for (int k = 0; k < H; ++k) {
                const double xv = xn[(size_t)k];
                qv += xv * Deq(qpk, qsc, qgs, (uint64_t)qcol * H + k);
                gv += xv * Deq(qpk, qsc, qgs, (uint64_t)gcol * H + k);
                kv_ += xv * Deq(kpk, ksc, kgs, (uint64_t)kcol * H + k);
                vv += xv * Deq(vpk, vsc, vgs, (uint64_t)kcol * H + k);
            }
            q[h][d] = qv; gate[h][d] = gv; keys[kvh][d] = kv_; vals[kvh][d] = vv;
        }
    }
    // per-head RMSNorm(q) / RMSNorm(k).
    for (int h = 0; h < NH; ++h) {
        double msq = 0;
        for (int d = 0; d < HD; ++d) msq += q[h][d] * q[h][d];
        const double inv = 1.0 / std::sqrt(msq / HD + eps);
        for (int d = 0; d < HD; ++d) q[h][d] *= inv * (1.0 + qn[d]);
    }
    for (int kvh = 0; kvh < NKV; ++kvh) {
        double msq = 0;
        for (int d = 0; d < HD; ++d) msq += keys[kvh][d] * keys[kvh][d];
        const double inv = 1.0 / std::sqrt(msq / HD + eps);
        for (int d = 0; d < HD; ++d) keys[kvh][d] *= inv * (1.0 + kn[d]);
    }
    // inline RoPE (rotate-half pairing) on first rotary_dim dims.
    if (rotary_dim > 0) {
        for (int h = 0; h < NH; ++h) {
            const std::vector<double> sq0 = q[h];
            for (int d = 0; d < rotary_dim; ++d) {
                const int half = rotary_dim / 2;
                double inv = 1.0 / std::pow(theta, (double)(2 * (d % half)) / (double)rotary_dim);
                const double ang = kv_base * inv;
                const double c = std::cos(ang), s = std::sin(ang);
                if (d < half) q[h][d] = sq0[d] * c - sq0[d + half] * s;
                else          q[h][d] = sq0[d] * c + sq0[d - half] * s;
            }
        }
        for (int kvh = 0; kvh < NKV; ++kvh) {
            const std::vector<double> sk0 = keys[kvh];
            for (int d = 0; d < rotary_dim; ++d) {
                const int half = rotary_dim / 2;
                double inv = 1.0 / std::pow(theta, (double)(2 * (d % half)) / (double)rotary_dim);
                const double ang = kv_base * inv;
                const double c = std::cos(ang), s = std::sin(ang);
                if (d < half) keys[kvh][d] = sk0[d] * c - sk0[d + half] * s;
                else          keys[kvh][d] = sk0[d] * c + sk0[d - half] * s;
            }
        }
    }
    // 3. append this token's k/v to NV row kv_base.
    {
        std::vector<double> krow(NKV * HD), vrow(NKV * HD);
        for (int kvh = 0; kvh < NKV; ++kvh)
            for (int d = 0; d < HD; ++d) {
                krow[kvh * HD + d] = keys[kvh][d];
                vrow[kvh * HD + d] = vals[kvh][d];
            }
        if ((int)kv_k.size() <= kv_base) kv_k.resize(kv_base + 1);
        if ((int)kv_v.size() <= kv_base) kv_v.resize(kv_base + 1);
        kv_k[kv_base] = krow;
        kv_v[kv_base] = vrow;
    }
    // 4. block-wide online-softmax causal GQA + gate.
    const double scale = 1.0 / std::sqrt((double)HD);
    std::vector<std::vector<double>> out_h(NH, std::vector<double>(HD));
    for (int h = 0; h < NH; ++h) {
        const int kvh = h / (NH / NKV);
        const int nscores = kv_base + 1;
        std::vector<double> scores(nscores);
        for (int s = 0; s < nscores; ++s) {
            double dot = 0;
            for (int d = 0; d < HD; ++d) dot += q[h][d] * kv_k[s][kvh * HD + d];
            scores[s] = dot * scale;
        }
        double m = -1e30;
        for (int s = 0; s < nscores; ++s) m = std::fmax(m, scores[s]);
        double sum = 0;
        for (int s = 0; s < nscores; ++s) sum += std::exp(scores[s] - m);
        const double inv = 1.0 / (sum + 1e-10);
        for (int d = 0; d < HD; ++d) {
            double acc = 0;
            for (int s = 0; s < nscores; ++s)
                acc += std::exp(scores[s] - m) * inv * kv_v[s][kvh * HD + d];
            const double gv = gate[h][d];
            out_h[h][d] = acc * (1.0 / (1.0 + std::exp(-gv)));
        }
    }
    // 5. o_proj (float-accumulate semantics).
    out.assign(H, 0.0);
    const uint64_t K = (uint64_t)NH * HD;   // o_proj input rows
    std::vector<double> oh_flat((size_t)NH * HD);
    for (int h = 0; h < NH; ++h)
        for (int d = 0; d < HD; ++d) oh_flat[(size_t)h * HD + d] = out_h[h][d];
    for (int n = 0; n < H; ++n) {
        double acc = 0.0;
        for (uint64_t k = 0; k < K; ++k)
            acc += oh_flat[(size_t)k] * Deq(opk, osc, ogs, n * K + k);
        out[(size_t)n] = acc;
    }
}

}  // namespace

// Task 1: the public numeric entry exists and accepts a tiny setup.
TEST_CASE("RunQwenFusedDecodeAttnKernel accepts a tiny full-attn layer", "[qwen][fused][decode]")
{
    constexpr int NH = 2, NKV = 1, HD = 8, H = 16, rotary_dim = 4;
    constexpr int max_kv = 8;

    QwenLayerWeights lw = MakeTinyFullLayer(NH, NKV, HD, H, rotary_dim);

    lsxcommon::qwen::QwenTopology top;
    top.hidden_size = H;
    top.num_attention_heads = NH;
    top.num_key_value_heads = NKV;
    top.head_dim = HD;
    top.partial_rotary_factor = 0.25;
    top.rope_theta = 10000000.0;
    top.rms_norm_eps = 1e-6;

    // x token [H], kv buffers, out [H]
    __half *dx, *dkv_k, *dkv_v, *dout;
    cudaMalloc(&dx, H*2);
    cudaMalloc(&dkv_k, (size_t)max_kv * NKV * HD * 2);
    cudaMalloc(&dkv_v, (size_t)max_kv * NKV * HD * 2);
    cudaMalloc(&dout, H*2);
    std::vector<__half> hx(H);
    for (int i = 0; i < H; ++i) hx[i] = __float2half(0.3f + 0.01f * (float)(i % 9));
    cudaMemcpy(dx, hx.data(), H*2, cudaMemcpyHostToDevice);
    cudaMemset(dkv_k, 0, (size_t)max_kv * NKV * HD * 2);
    cudaMemset(dkv_v, 0, (size_t)max_kv * NKV * HD * 2);

    cudaError_t err = lsxcommon::qwen::RunQwenFusedDecodeAttnKernel(
        nullptr, dx, lw, top, /*kv_base=*/0, dkv_k, dkv_v, dout);
    REQUIRE(err == cudaSuccess);

    cudaFree(dx); cudaFree(dkv_k); cudaFree(dkv_v); cudaFree(dout);
    FreeTinyLayer(lw);
}

// Task 4: end-to-end parity — the fused 3-stage decoder layer (single token,
// persistent NV append) matches an independent CPU FP64 reference within a
// 1e-2 relative tolerance, both for first token (kv_base=0) and second token
// (kv_base=1, causal over the two appended rows).
TEST_CASE("fused 3-stage decoder matches CPU FP64 reference (NV persistence)",
          "[qwen][fused][decode][parity]")
{
    constexpr int NH = 2, NKV = 1, HD = 8, H = 16;
    constexpr int max_kv = 8;

    QwenLayerWeights lw = MakeTinyFullLayer(NH, NKV, HD, H, /*rotary_dim=*/4);

    lsxcommon::qwen::QwenTopology top;
    top.hidden_size = H;
    top.num_attention_heads = NH;
    top.num_key_value_heads = NKV;
    top.head_dim = HD;
    top.partial_rotary_factor = 0.5;   // rot = HD * 0.5 = 4
    top.rope_theta = 10000000.0;
    top.rms_norm_eps = 1e-6;
    const int rot = (int)(HD * top.partial_rotary_factor);

    // ── weight + norm host copies (CPU reference inputs) ──────────────────
    std::vector<uint8_t> qpk, qsc, kpk, ksc, vpk, vsc, opk, osc;
    auto rd_weight = [&](const lsxcommon::qwen::QwenNvfp4Weight &w,
                         std::vector<uint8_t> &pk, std::vector<uint8_t> &sc) {
        pk.assign((size_t)w.packed_bytes, 0);
        sc.assign((size_t)w.scale_bytes, 0);
        cudaMemcpy(pk.data(), w.d_packed, pk.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(sc.data(), w.d_scales, sc.size(), cudaMemcpyDeviceToHost);
    };
    rd_weight(lw.q, qpk, qsc);
    rd_weight(lw.k, kpk, ksc);
    rd_weight(lw.v, vpk, vsc);
    rd_weight(lw.o, opk, osc);
    auto rd_half = [](const __half *dev, int n) {
        std::vector<__half> tmp((size_t)n);
        cudaMemcpy(tmp.data(), dev, (size_t)n * 2, cudaMemcpyDeviceToHost);
        std::vector<double> r((size_t)n);
        for (int i = 0; i < n; ++i) r[(size_t)i] = (double)__half2float(tmp[(size_t)i]);
        return r;
    };
    const std::vector<double> in_norm = rd_half(lw.d_input_norm, H);
    const std::vector<double> qn = rd_half(lw.d_q_norm, HD);
    const std::vector<double> kn = rd_half(lw.d_k_norm, HD);

    // ── device buffers ────────────────────────────────────────────────────
    __half *dx0, *dx1, *dkv_k, *dkv_v, *dout0, *dout1;
    REQUIRE(cudaMalloc(&dx0, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dx1, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dkv_k, (size_t)max_kv * NKV * HD * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dkv_v, (size_t)max_kv * NKV * HD * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout0, H * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout1, H * 2) == cudaSuccess);
    cudaMemset(dkv_k, 0, (size_t)max_kv * NKV * HD * 2);
    cudaMemset(dkv_v, 0, (size_t)max_kv * NKV * HD * 2);

    // two tokens
    std::vector<double> x0(H), x1(H);
    std::vector<__half> hx0(H), hx1(H);
    for (int i = 0; i < H; ++i) {
        x0[(size_t)i] = 0.30 + 0.01 * (i % 9);
        x1[(size_t)i] = 0.55 - 0.02 * (i % 5);
        hx0[(size_t)i] = __float2half((float)x0[(size_t)i]);
        hx1[(size_t)i] = __float2half((float)x1[(size_t)i]);
    }
    cudaMemcpy(dx0, hx0.data(), H * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dx1, hx1.data(), H * 2, cudaMemcpyHostToDevice);

    // ── fused runs (persistent NV across steps) ───────────────────────────
    REQUIRE(lsxcommon::qwen::RunQwenFusedDecodeAttnKernel(
                nullptr, dx0, lw, top, /*kv_base=*/0, dkv_k, dkv_v, dout0) == cudaSuccess);
    REQUIRE(lsxcommon::qwen::RunQwenFusedDecodeAttnKernel(
                nullptr, dx1, lw, top, /*kv_base=*/1, dkv_k, dkv_v, dout1) == cudaSuccess);

    // ── CPU FP64 reference with the same persistent NV ────────────────────
    std::vector<std::vector<double>> kvk, kvv;
    std::vector<double> ref0, ref1;
    CpuFusedForward(NH, NKV, HD, H, rot, top.rope_theta, top.rms_norm_eps, /*kv_base=*/0,
                    x0, in_norm, qn, kn,
                    qpk, qsc, lw.q.global_scale,
                    kpk, ksc, lw.k.global_scale,
                    vpk, vsc, lw.v.global_scale,
                    opk, osc, lw.o.global_scale,
                    kvk, kvv, ref0);
    CpuFusedForward(NH, NKV, HD, H, rot, top.rope_theta, top.rms_norm_eps, /*kv_base=*/1,
                    x1, in_norm, qn, kn,
                    qpk, qsc, lw.q.global_scale,
                    kpk, ksc, lw.k.global_scale,
                    vpk, vsc, lw.v.global_scale,
                    opk, osc, lw.o.global_scale,
                    kvk, kvv, ref1);

    std::vector<__half> hout0(H), hout1(H);
    cudaMemcpy(hout0.data(), dout0, H * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hout1.data(), dout1, H * 2, cudaMemcpyDeviceToHost);

    for (int i = 0; i < H; ++i) {
        const double got0 = (double)__half2float(hout0[(size_t)i]);
        const double got1 = (double)__half2float(hout1[(size_t)i]);
        REQUIRE(std::isfinite(got0));
        REQUIRE(std::isfinite(got1));
        const double tol0 = 1e-2 * std::max(1.0, std::abs(ref0[(size_t)i]));
        const double tol1 = 1e-2 * std::max(1.0, std::abs(ref1[(size_t)i]));
        CAPTURE(i, got0, ref0[(size_t)i], got1, ref1[(size_t)i]);
        REQUIRE(std::abs(got0 - ref0[(size_t)i]) <= tol0);
        REQUIRE(std::abs(got1 - ref1[(size_t)i]) <= tol1);
    }

    cudaFree(dx0); cudaFree(dx1); cudaFree(dkv_k); cudaFree(dkv_v);
    cudaFree(dout0); cudaFree(dout1);
    FreeTinyLayer(lw);
}