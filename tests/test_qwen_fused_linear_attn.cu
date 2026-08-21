#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "lsxcommon/qwen_inference.h"

using namespace lsxcommon::qwen;

namespace {

__host__ float HostE4M3(uint8_t e4) {
    const uint8_t exp = (e4 >> 3) & 0xF, man = e4 & 0x7;
    if (exp == 0) return 0.0f;
    unsigned bits = ((unsigned)exp + 120u) << 23 | (unsigned)man << 20;
    float f; std::memcpy(&f, &bits, 4); return f;
}
constexpr float kHostE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__host__ float HostDeq(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales,
                       int K, int N, float gs, int k, int n) {
    const int64_t e = (int64_t)n * K + k;
    const uint8_t byte = packed[e >> 1];
    const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
    const int s = nib & 0x7;
    const float sign = (nib & 0x8) ? -1.0f : 1.0f;
    return sign * kHostE2M1[s] * HostE4M3(scales[e >> 4]) * gs;
}

void FillNvfp4(QwenNvfp4Weight &w, int K, int N, float gs) {
    std::vector<uint8_t> pk((size_t)K * N / 2), sc((size_t)K * N / 16);
    for (size_t i = 0; i < pk.size(); ++i) pk[i] = (uint8_t)((i * 37 + 5) & 0xFF);
    for (size_t i = 0; i < sc.size(); ++i) sc[i] = (uint8_t)(120 + (i % 12));
    cudaMalloc(&w.d_packed, pk.size()); cudaMemcpy(const_cast<uint8_t*>(w.d_packed), pk.data(), pk.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&w.d_scales, sc.size()); cudaMemcpy(const_cast<uint8_t*>(w.d_scales), sc.data(), sc.size(), cudaMemcpyHostToDevice);
    w.global_scale = gs; w.rows = K; w.cols = N;
    w.packed_bytes = pk.size(); w.scale_bytes = sc.size();
}

} // namespace

TEST_CASE("RunQwenFusedLinearAttnKernel entry runs on a tiny layer", "[qwen][linear][fuse]")
{
    const int H = 16, nk = 2, kd = 8, nv = 2, vd = 8, seq = 2, kernel = 3;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;

    QwenTopology top;
    top.linear_num_key_heads = nk; top.linear_key_head_dim = kd;
    top.linear_num_value_heads = nv; top.linear_value_head_dim = vd;
    top.rms_norm_eps = 1e-6f;

    QwenLayerWeights lw; lw.loaded = true; lw.is_linear = true;
    FillNvfp4(lw.qkv, H, S, 0.5f); FillNvfp4(lw.z, H, VD, 0.5f);
    FillNvfp4(lw.a, H, nv, 0.5f);  FillNvfp4(lw.b, H, nv, 0.5f);
    FillNvfp4(lw.out, VD, H, 0.5f);
    std::vector<__half> conv(S * kernel, __float2half(0.25f));
    cudaMalloc(&lw.d_conv1d, conv.size() * 2);
    cudaMemcpy(lw.d_conv1d, conv.data(), conv.size() * 2, cudaMemcpyHostToDevice);
    std::vector<float> alog(nv, -0.5f), dtb(nv, 0.1f);
    cudaMalloc(&lw.d_A_log, nv * 4); cudaMemcpy(lw.d_A_log, alog.data(), nv*4, cudaMemcpyHostToDevice);
    cudaMalloc(&lw.d_dt_bias, nv * 4); cudaMemcpy(lw.d_dt_bias, dtb.data(), nv*4, cudaMemcpyHostToDevice);
    std::vector<__half> attn_norm(vd, __float2half(1.0f));
    cudaMalloc(&lw.d_attn_norm, vd*2); cudaMemcpy(lw.d_attn_norm, attn_norm.data(), vd*2, cudaMemcpyHostToDevice);

    std::vector<__half> x(seq * H);
    for (int i = 0; i < seq*H; ++i) x[i] = __float2half(0.1f + 0.03f * ((i % 5) - 2));
    __half *dx; cudaMalloc(&dx, seq*H*2); cudaMemcpy(dx, x.data(), seq*H*2, cudaMemcpyHostToDevice);
    __half *dq, *dz, *dab, *dconv, *dcore, *dgated, *dout;
    float *dg, *dbeta;
    cudaMalloc(&dq,  (size_t)seq*S*2); cudaMalloc(&dz, (size_t)seq*VD*2);
    cudaMalloc(&dab, (size_t)seq*nv*2*2); cudaMalloc(&dconv, (size_t)seq*S*2);
    cudaMalloc(&dcore, (size_t)seq*VD*2); cudaMalloc(&dgated, (size_t)seq*VD*2);
    cudaMalloc(&dout, (size_t)seq*H*2); cudaMalloc(&dg, (size_t)seq*nv*4);
    cudaMalloc(&dbeta, (size_t)seq*nv*4);

    REQUIRE(RunQwenFusedLinearAttnKernel(nullptr, lw, top, seq, dx, dout,
        dq, dz, dab, dg, dbeta, dconv, dcore, dgated) == cudaSuccess);
    cudaDeviceSynchronize();

    lw.qkv.Free(); lw.z.Free(); lw.a.Free(); lw.b.Free(); lw.out.Free();
    cudaFree(lw.d_conv1d); cudaFree(lw.d_A_log); cudaFree(lw.d_dt_bias); cudaFree(lw.d_attn_norm);
    cudaFree(dx); cudaFree(dq); cudaFree(dz); cudaFree(dab); cudaFree(dconv);
    cudaFree(dcore); cudaFree(dgated); cudaFree(dout); cudaFree(dg); cudaFree(dbeta);
}

namespace {
// Host F1 reference: qkv raw row (per col), causal conv1d, l2norm q/k slices.
// Mirrors the fused_linear_front_kernel phase order for parity.
std::vector<__half> HostLinearFront(const QwenLayerWeights &lw, int seq, int kv_base,
                                    int nk, int kd, int nv, int vd,
                                    const std::vector<__half> &x, int H)
{
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;
    const int kk = lw.conv_kernel;
    std::vector<uint8_t> pk(lw.qkv.packed_bytes), sc(lw.qkv.scale_bytes);
    cudaMemcpy(pk.data(), lw.qkv.d_packed, pk.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(sc.data(), lw.qkv.d_scales, sc.size(), cudaMemcpyDeviceToHost);
    std::vector<__half> convw((size_t)S * kk);
    cudaMemcpy(convw.data(), lw.d_conv1d, S*kk*2, cudaMemcpyDeviceToHost);

    std::vector<__half> hist((size_t)(kv_base + seq) * S);
    // raw qkv rows
    for (int t = 0; t < seq; ++t) {
        for (int c = 0; c < S; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < H; ++k)
                acc += __half2float(x[(size_t)t * H + k]) * HostDeq(pk, sc, H, S, lw.qkv.global_scale, k, c);
            hist[((size_t)kv_base + t) * S + c] = __float2half(acc);
        }
    }
    // conv1d + silu into d_conv
    std::vector<__half> conv((size_t)seq * S);
    for (int t = 0; t < seq; ++t) {
        for (int c = 0; c < S; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < kk; ++k) {
                int tt = kv_base + t - k;
                if (tt < 0) continue;
                acc += __half2float(convw[(size_t)c * kk + (kk - 1 - k)]) *
                       __half2float(hist[(size_t)tt * S + c]);
            }
            conv[(size_t)t * S + c] = __float2half(acc / (1.0f + std::exp(-acc)));
        }
    }
    // l2norm q (1/sqrt(kd)) and k (1)
    const float q_scale = 1.0f / std::sqrt((float)kd);
    for (int t = 0; t < seq; ++t)
        for (int h = 0; h < nk; ++h) {
            for (int slice = 0; slice < 2; ++slice) {
                int off = (slice ? KD : 0) + h * kd;
                float ss = 0.0f;
                for (int l = 0; l < kd; ++l) { float v = __half2float(conv[(size_t)t*S + off + l]); ss += v*v; }
                float inv = (1.0f / std::sqrt(ss + 1e-6f)) * (slice ? 1.0f : q_scale);
                for (int l = 0; l < kd; ++l)
                    conv[(size_t)t*S + off + l] = __float2half(__half2float(conv[(size_t)t*S+off+l]) * inv);
            }
        }
    return conv;
}
} // namespace

TEST_CASE("Fused front F1 qkv+conv1d+l2norm matches host reference", "[qwen][linear][fuse]")
{
    const int H = 16, nk = 2, kd = 8, nv = 2, vd = 8, kernel = 3;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;

    QwenTopology top;
    top.linear_num_key_heads = nk; top.linear_key_head_dim = kd;
    top.linear_num_value_heads = nv; top.linear_value_head_dim = vd;
    top.rms_norm_eps = 1e-6f;

    QwenLayerWeights lw; lw.loaded = true; lw.is_linear = true;
    FillNvfp4(lw.qkv, H, S, 0.5f); FillNvfp4(lw.z, H, VD, 0.5f);
    FillNvfp4(lw.a, H, nv, 0.5f);  FillNvfp4(lw.b, H, nv, 0.5f);
    FillNvfp4(lw.out, VD, H, 0.5f);
    std::vector<__half> conv(S * kernel, __float2half(0.25f));
    cudaMalloc(&lw.d_conv1d, conv.size() * 2);
    cudaMemcpy(lw.d_conv1d, conv.data(), conv.size() * 2, cudaMemcpyHostToDevice);
    std::vector<float> alog(nv, -0.5f), dtb(nv, 0.1f);
    cudaMalloc(&lw.d_A_log, nv * 4); cudaMemcpy(lw.d_A_log, alog.data(), nv*4, cudaMemcpyHostToDevice);
    cudaMalloc(&lw.d_dt_bias, nv * 4); cudaMemcpy(lw.d_dt_bias, dtb.data(), nv*4, cudaMemcpyHostToDevice);
    std::vector<__half> attn_norm(vd, __float2half(1.0f));
    cudaMalloc(&lw.d_attn_norm, vd*2); cudaMemcpy(lw.d_attn_norm, attn_norm.data(), vd*2, cudaMemcpyHostToDevice);

    for (int seq : {1, 2}) {
        std::vector<__half> x(seq * H);
        for (int i = 0; i < seq*H; ++i) x[i] = __float2half(0.1f + 0.03f * ((i % 5) - 2));
        __half *dx; cudaMalloc(&dx, seq*H*2); cudaMemcpy(dx, x.data(), seq*H*2, cudaMemcpyHostToDevice);
        __half *dq, *dz, *dab, *dconv, *dcore, *dgated, *dout;
        float *dg, *dbeta;
        cudaMalloc(&dq,  (size_t)seq*S*2); cudaMalloc(&dz, (size_t)seq*VD*2);
        cudaMalloc(&dab, (size_t)seq*nv*2*2); cudaMalloc(&dconv, (size_t)seq*S*2);
        cudaMalloc(&dcore, (size_t)seq*VD*2); cudaMalloc(&dgated, (size_t)seq*VD*2);
        cudaMalloc(&dout, (size_t)seq*H*2); cudaMalloc(&dg, (size_t)seq*nv*4);
        cudaMalloc(&dbeta, (size_t)seq*nv*4);

        REQUIRE(RunQwenFusedLinearAttnKernel(nullptr, lw, top, seq, dx, dout,
            dq, dz, dab, dg, dbeta, dconv, dcore, dgated) == cudaSuccess);
        cudaDeviceSynchronize();

        auto ref = HostLinearFront(lw, seq, 0, nk, kd, nv, vd, x, H);
        std::vector<__half> got((size_t)seq * S);
        cudaMemcpy(got.data(), dconv, seq*S*2, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < got.size(); ++i) {
            float r = __half2float(ref[i]), g = __half2float(got[i]);
            REQUIRE(std::abs(r - g) < 1e-2f * (1.0f + std::abs(r)));
        }

        cudaFree(dx); cudaFree(dq); cudaFree(dz); cudaFree(dab); cudaFree(dconv);
        cudaFree(dcore); cudaFree(dgated); cudaFree(dout); cudaFree(dg); cudaFree(dbeta);
    }

    lw.qkv.Free(); lw.z.Free(); lw.a.Free(); lw.b.Free(); lw.out.Free();
    cudaFree(lw.d_conv1d); cudaFree(lw.d_A_log); cudaFree(lw.d_dt_bias); cudaFree(lw.d_attn_norm);
}

TEST_CASE("Fused linear-attn full layer matches split reference (seq=1,2)", "[qwen][linear][fuse]")
{
    const int H = 16, nk = 2, kd = 8, nv = 2, vd = 8, kernel = 3;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;

    QwenTopology top;
    top.linear_num_key_heads = nk; top.linear_key_head_dim = kd;
    top.linear_num_value_heads = nv; top.linear_value_head_dim = vd;
    top.rms_norm_eps = 1e-6f;

    QwenLayerWeights lw; lw.loaded = true; lw.is_linear = true;
    FillNvfp4(lw.qkv, H, S, 0.5f); FillNvfp4(lw.z, H, VD, 0.5f);
    FillNvfp4(lw.a, H, nv, 0.5f);  FillNvfp4(lw.b, H, nv, 0.5f);
    FillNvfp4(lw.out, VD, H, 0.5f);
    std::vector<__half> conv(S * kernel, __float2half(0.25f));
    cudaMalloc(&lw.d_conv1d, conv.size() * 2);
    cudaMemcpy(lw.d_conv1d, conv.data(), conv.size() * 2, cudaMemcpyHostToDevice);
    std::vector<float> alog(nv, -0.5f), dtb(nv, 0.1f);
    cudaMalloc(&lw.d_A_log, nv * 4); cudaMemcpy(lw.d_A_log, alog.data(), nv*4, cudaMemcpyHostToDevice);
    cudaMalloc(&lw.d_dt_bias, nv * 4); cudaMemcpy(lw.d_dt_bias, dtb.data(), nv*4, cudaMemcpyHostToDevice);
    std::vector<__half> attn_norm(vd, __float2half(1.0f));
    cudaMalloc(&lw.d_attn_norm, vd*2); cudaMemcpy(lw.d_attn_norm, attn_norm.data(), vd*2, cudaMemcpyHostToDevice);

    for (int seq : {1, 2}) {
        std::vector<__half> x(seq * H);
        for (int i = 0; i < seq*H; ++i) x[i] = __float2half(0.1f + 0.03f * ((i % 5) - 2));
        __half *dx; cudaMalloc(&dx, seq*H*2); cudaMemcpy(dx, x.data(), seq*H*2, cudaMemcpyHostToDevice);
        __half *dq, *dqf, *dz, *dzf, *dab, *dabf, *dconv, *dconvf, *dcore, *dcoref, *dgated, *dgatedf, *dout, *doutf;
        float *dg, *dgf, *dbeta, *dbetaf;
        cudaMalloc(&dq,  (size_t)seq*S*2); cudaMalloc(&dqf, (size_t)seq*S*2);
        cudaMalloc(&dz, (size_t)seq*VD*2); cudaMalloc(&dzf, (size_t)seq*VD*2);
        cudaMalloc(&dab, (size_t)seq*nv*2*2); cudaMalloc(&dabf, (size_t)seq*nv*2*2);
        cudaMalloc(&dconv, (size_t)seq*S*2); cudaMalloc(&dconvf, (size_t)seq*S*2);
        cudaMalloc(&dcore, (size_t)seq*VD*2); cudaMalloc(&dcoref, (size_t)seq*VD*2);
        cudaMalloc(&dgated, (size_t)seq*VD*2); cudaMalloc(&dgatedf, (size_t)seq*VD*2);
        cudaMalloc(&dout, (size_t)seq*H*2); cudaMalloc(&doutf, (size_t)seq*H*2);
        cudaMalloc(&dg, (size_t)seq*nv*4); cudaMalloc(&dgf, (size_t)seq*nv*4);
        cudaMalloc(&dbeta, (size_t)seq*nv*4); cudaMalloc(&dbetaf, (size_t)seq*nv*4);

        REQUIRE(RunQwenLinearAttention(nullptr, lw, top, seq, dx, dout,
            dq, dz, dab, dg, dbeta, dconv, dcore, dgated) == cudaSuccess);
        REQUIRE(RunQwenFusedLinearAttnKernel(nullptr, lw, top, seq, dx, doutf,
            dqf, dzf, dabf, dgf, dbetaf, dconvf, dcoref, dgatedf) == cudaSuccess);
        cudaDeviceSynchronize();

        std::vector<__half> ho(seq*H), hf(seq*H);
        cudaMemcpy(ho.data(), dout, seq*H*2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hf.data(), doutf, seq*H*2, cudaMemcpyDeviceToHost);
        for (int i = 0; i < seq*H; ++i) {
            float r = __half2float(ho[i]), g = __half2float(hf[i]);
            REQUIRE(std::abs(r - g) < 1e-2f * (1.0f + std::abs(r)));
        }

        cudaFree(dx); cudaFree(dq); cudaFree(dqf); cudaFree(dz); cudaFree(dzf);
        cudaFree(dab); cudaFree(dabf); cudaFree(dconv); cudaFree(dconvf);
        cudaFree(dcore); cudaFree(dcoref); cudaFree(dgated); cudaFree(dgatedf);
        cudaFree(dout); cudaFree(doutf); cudaFree(dg); cudaFree(dgf); cudaFree(dbeta); cudaFree(dbetaf);
    }

    lw.qkv.Free(); lw.z.Free(); lw.a.Free(); lw.b.Free(); lw.out.Free();
    cudaFree(lw.d_conv1d); cudaFree(lw.d_A_log); cudaFree(lw.d_dt_bias); cudaFree(lw.d_attn_norm);
}
