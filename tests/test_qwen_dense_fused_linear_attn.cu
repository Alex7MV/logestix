#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "lsxcommon/lsx_inference_load.h"
#include "lsxcommon/lsx_attn_inference.h"

namespace attn = lsxcommon::attn;
namespace lsx_nvfp4 = lsxcommon::lsx_nvfp4;
#include "nvfp4_codec.h"

using namespace lsxcommon::qwen;
using lsxcommon::gpu::nvfp4::E2M1Value;
using lsxcommon::gpu::nvfp4::E4M3ScaleToFloat;

namespace {

__host__ float HostDeq(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales,
                       int K, int N, float gs, int k, int n) {
    const int64_t e = (int64_t)n * K + k;
    const uint8_t byte = packed[e >> 1];
    const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
    // Tile-GEMM parity: weight dequantized then rounded to half.
    const __half wv = __float2half(E2M1Value(nib)
                                   * E4M3ScaleToFloat(scales[e >> 4]) * gs);
    return __half2float(wv);
}

void FillNvfp4(lsx_nvfp4::LsxNvfp4Weight &w, int K, int N, float gs) {
    std::vector<uint8_t> pk((size_t)K * N / 2), sc((size_t)K * N / 16);
    for (size_t i = 0; i < pk.size(); ++i) pk[i] = (uint8_t)((i * 37 + 5) & 0xFF);
    for (size_t i = 0; i < sc.size(); ++i) sc[i] = (uint8_t)(120 + (i % 12));
    cudaMalloc(&w.d_packed, pk.size()); cudaMemcpy(const_cast<uint8_t*>(w.d_packed), pk.data(), pk.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&w.d_scales, sc.size()); cudaMemcpy(const_cast<uint8_t*>(w.d_scales), sc.data(), sc.size(), cudaMemcpyHostToDevice);
    w.global_scale = gs;
    // lsx_nvfp4::LsxNvfp4Weight convention: rows = OUTPUT dim, cols = INPUT dim (K for GEMM).
    w.rows = N; w.cols = K;
    w.packed_bytes = pk.size(); w.scale_bytes = sc.size();
}

} // namespace

TEST_CASE("RunAttnFusedLinearAttnKernel entry runs on a tiny layer", "[qwen][linear][fuse]")
{
    const int H = 16, nk = 2, kd = 8, nv = 2, vd = 8, seq = 2, kernel = 3;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;

    attn::AttnGeometry geo;
    geo.linear_num_key_heads = nk; geo.linear_key_head_dim = kd;
    geo.linear_num_value_heads = nv; geo.linear_value_head_dim = vd;
    geo.rms_norm_eps = 1e-6f;

    attn::LayerWeights lw; lw.loaded = true; lw.is_linear = true;
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

    REQUIRE(RunAttnFusedLinearAttnKernel(nullptr, lw, geo, seq, dx, dout,
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
std::vector<__half> HostLinearFront(const attn::LayerWeights &lw, int seq, int kv_base,
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
    auto run_geom = [](int nk, int kd, int nv, int vd, float gs) {
        UNSCOPED_INFO("geom nk=" << nk << " kd=" << kd);
        const int H = nk * kd, kernel = 3;
        const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;

        attn::AttnGeometry geo;
        geo.linear_num_key_heads = nk; geo.linear_key_head_dim = kd;
        geo.linear_num_value_heads = nv; geo.linear_value_head_dim = vd;
        geo.rms_norm_eps = 1e-6f;

        attn::LayerWeights lw; lw.loaded = true; lw.is_linear = true;
        FillNvfp4(lw.qkv, H, S, gs); FillNvfp4(lw.z, H, VD, gs);
        FillNvfp4(lw.a, H, nv, gs);  FillNvfp4(lw.b, H, nv, gs);
        FillNvfp4(lw.out, VD, H, gs);
        std::vector<__half> conv(S * kernel, __float2half(0.25f));
        lw.conv_kernel = kernel; lw.conv_dim = S;
        cudaMalloc(&lw.d_conv1d, conv.size() * 2);
        cudaMemcpy(lw.d_conv1d, conv.data(), conv.size() * 2, cudaMemcpyHostToDevice);
        std::vector<float> alog(nv, -0.5f), dtb(nv, 0.1f);
        cudaMalloc(&lw.d_A_log, nv * 4); cudaMemcpy(lw.d_A_log, alog.data(), nv*4, cudaMemcpyHostToDevice);
        cudaMalloc(&lw.d_dt_bias, nv * 4); cudaMemcpy(lw.d_dt_bias, dtb.data(), nv*4, cudaMemcpyHostToDevice);
        std::vector<__half> attn_norm(vd, __float2half(1.0f));
        cudaMalloc(&lw.d_attn_norm, vd*2); cudaMemcpy(lw.d_attn_norm, attn_norm.data(), vd*2, cudaMemcpyHostToDevice);

        auto w2host = [](const lsx_nvfp4::LsxNvfp4Weight &w) {
            std::vector<uint8_t> pk(w.rows * w.cols / 2), sc(w.rows * w.cols / 16);
            cudaMemcpy(pk.data(), w.d_packed, pk.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(sc.data(), w.d_scales, sc.size(), cudaMemcpyDeviceToHost);
            return std::make_pair(pk, sc);
        };
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

            REQUIRE(RunAttnFusedLinearAttnKernel(nullptr, lw, geo, seq, dx, dout,
                dq, dz, dab, dg, dbeta, dconv, dcore, dgated) == cudaSuccess);
            cudaDeviceSynchronize();

            auto ref = HostLinearFront(lw, seq, 0, nk, kd, nv, vd, x, H);
            std::vector<__half> got((size_t)seq * S);
            cudaMemcpy(got.data(), dconv, seq*S*2, cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < got.size(); ++i) {
                float r = __half2float(ref[i]), g = __half2float(got[i]);
                REQUIRE(std::abs(r - g) < 1e-2f * (1.0f + std::abs(r)));
            }

            // F2 host reference: z[i], a[i], b[i] then g/beta.
            auto w2host = [](const lsx_nvfp4::LsxNvfp4Weight &w) {
                std::vector<uint8_t> pk(w.packed_bytes), sc(w.scale_bytes);
                cudaMemcpy(pk.data(), w.d_packed, pk.size(), cudaMemcpyDeviceToHost);
                cudaMemcpy(sc.data(), w.d_scales, sc.size(), cudaMemcpyDeviceToHost);
                return std::make_pair(pk, sc);
            };
            auto zp = w2host(lw.z), ap = w2host(lw.a), bp = w2host(lw.b);
            std::vector<float> alog(nv), dtb(nv);
            cudaMemcpy(alog.data(), lw.d_A_log, nv*4, cudaMemcpyDeviceToHost);
            cudaMemcpy(dtb.data(), lw.d_dt_bias, nv*4, cudaMemcpyDeviceToHost);
            std::vector<__half> hz((size_t)seq*VD), hab((size_t)seq*nv*2);
            cudaMemcpy(hz.data(),  dz,  seq*VD*2, cudaMemcpyDeviceToHost);
            cudaMemcpy(hab.data(), dab, seq*nv*2*2, cudaMemcpyDeviceToHost);
            std::vector<float> hg(seq*nv), hb(seq*nv);
            cudaMemcpy(hg.data(), dg, seq*nv*4, cudaMemcpyDeviceToHost);
            cudaMemcpy(hb.data(), dbeta, seq*nv*4, cudaMemcpyDeviceToHost);
            for (int t = 0; t < seq; ++t) {
                for (int c = 0; c < VD; ++c) {
                    float acc = 0.0f;
                    for (int k = 0; k < H; ++k) acc += __half2float(x[(size_t)t*H+k]) * HostDeq(zp.first,zp.second,H,VD,lw.z.global_scale,k,c);
                    if (std::abs(__half2float(hz[(size_t)t*VD+c]) - acc) >= 1e-2f*(1.0f+std::abs(acc)))
                        UNSCOPED_INFO("z t=" << t << " c=" << c << " dev=" << __half2float(hz[(size_t)t*VD+c]) << " host=" << acc);
                    REQUIRE(std::abs(__half2float(hz[(size_t)t*VD+c]) - acc) < 1e-2f*(1.0f+std::abs(acc)));
                }
            }
            for (int t = 0; t < seq; ++t)
                for (int c = 0; c < nv; ++c) {
                    float acca = 0.0f, accb = 0.0f;
                    for (int k = 0; k < H; ++k) {
                        acca += __half2float(x[(size_t)t*H+k]) * HostDeq(ap.first,ap.second,H,nv,lw.a.global_scale,k,c);
                        accb += __half2float(x[(size_t)t*H+k]) * HostDeq(bp.first,bp.second,H,nv,lw.b.global_scale,k,c);
                    }
                    REQUIRE(std::abs(__half2float(hab[(size_t)t*nv+c]) - acca) < 1e-2f*(1.0f+std::abs(acca)));
                    float bb = __half2float(hab[(size_t)seq*nv + (size_t)t*nv+c]);
                    REQUIRE(std::abs(bb - accb) < 1e-2f*(1.0f+std::abs(accb)));
                    float gexp = -std::exp(alog[c]) * ( (acca+dtb[c]) > 20.0f ? (acca+dtb[c]) : std::log1p(std::exp(acca+dtb[c])) );
                    REQUIRE(std::abs(hg[(size_t)t*nv+c] - gexp) < 1e-2f*(1.0f+std::abs(gexp)));
                    float be = 1.0f/(1.0f+std::exp(-bb));
                    REQUIRE(std::abs(hb[(size_t)t*nv+c] - be) < 1e-2f*(1.0f+std::abs(be)));
                }

            cudaFree(dx); cudaFree(dq); cudaFree(dz); cudaFree(dab); cudaFree(dconv);
            cudaFree(dcore); cudaFree(dgated); cudaFree(dout); cudaFree(dg); cudaFree(dbeta);
        }

        lw.qkv.Free(); lw.z.Free(); lw.a.Free(); lw.b.Free(); lw.out.Free();
        cudaFree(lw.d_conv1d); cudaFree(lw.d_A_log); cudaFree(lw.d_dt_bias); cudaFree(lw.d_attn_norm);
    };

    run_geom(2, 8, 2, 8, 0.5f);
    run_geom(16, 128, 32, 128, 0.02f);
}

TEST_CASE("Fused linear-attn full layer matches split reference (seq=1,2)", "[qwen][linear][fuse]")
{
    auto run_geom = [](int nk, int kd, int nv, int vd, float gs) {
        UNSCOPED_INFO("geom nk=" << nk << " kd=" << kd);
        const int kernel = 3;
        const int H = nk * kd;             // hidden matches key head count x dim
        const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;

        attn::AttnGeometry geo;
        geo.linear_num_key_heads = nk; geo.linear_key_head_dim = kd;
        geo.linear_num_value_heads = nv; geo.linear_value_head_dim = vd;
        geo.rms_norm_eps = 1e-6f;

        attn::LayerWeights lw; lw.loaded = true; lw.is_linear = true;
        FillNvfp4(lw.qkv, H, S, gs); FillNvfp4(lw.z, H, VD, gs);
        FillNvfp4(lw.a, H, nv, gs);  FillNvfp4(lw.b, H, nv, gs);
        FillNvfp4(lw.out, VD, H, gs);
        std::vector<__half> conv(S * kernel, __float2half(0.25f));
        lw.conv_kernel = kernel; lw.conv_dim = S;
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

            REQUIRE(RunAttnLinearAttention(nullptr, lw, geo, seq, dx, dout,
                dq, dz, dab, dg, dbeta, dconv, dcore, dgated) == cudaSuccess);
            REQUIRE(RunAttnFusedLinearAttnKernel(nullptr, lw, geo, seq, dx, doutf,
                dqf, dzf, dabf, dgf, dbetaf, dconvf, dcoref, dgatedf) == cudaSuccess);
            cudaDeviceSynchronize();

            if (seq > 1) {
                // Persistent state with a multi-token chunk has no grid barrier
                // between the col_seed read and the t == seq-1 write (audit X8):
                // the neutral entry must refuse it until it has a state pass.
                float *dstate = nullptr;
                REQUIRE(cudaMalloc(&dstate,
                                   (size_t)nv * vd * kd * sizeof(float)) ==
                        cudaSuccess);
                CHECK(RunAttnLinearAttention(nullptr, lw, geo, seq, dx, dout,
                      dq, dz, dab, dg, dbeta, dconv, dcore, dgated, 0,
                      dstate) == cudaErrorInvalidValue);
                CHECK(RunAttnFusedLinearAttnKernel(nullptr, lw, geo, seq, dx,
                      doutf, dqf, dzf, dabf, dgf, dbetaf, dconvf, dcoref,
                      dgatedf, 0, dstate) == cudaErrorInvalidValue);
                cudaFree(dstate);
            }

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
    };

    run_geom(2, 8, 2, 8, 0.5f);
    run_geom(16, 128, 32, 128, 0.02f);   // real Qwen3.5-9B linear-attn geometry
}
