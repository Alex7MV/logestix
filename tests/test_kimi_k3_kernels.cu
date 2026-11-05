// Kimi-K3 kernel host references: SiTU-GLU, the sigmoid+bias top-k router, the
// AttnRes depth mixer, the KDA (per-channel decay, lower-bound gate, conv-4)
// recurrence and the absorbed gated NoPE MLA. Every expectation is computed on
// the host from the same math the checkpoint's modeling code defines — the
// kernels are never used to produce their own golden.
#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "lsxcommon/kimi_moe_k3_kernels.h"

using namespace lsxcommon::kimi_k3;

namespace {

__half H(float v) { return __float2half(v); }
float F(__half h) { return __half2float(h); }

// E2M1 nibble table (matches the neutral codec).
float E2m1(int nib)
{
    static const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (nib & 8) ? -t[nib & 7] : t[nib & 7];
}

// E4M3FN decode over the valid scale domain.
float E4m3(int b)
{
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    if (e == 0) return static_cast<float>(m) / 512.0f;
    return std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
}

// Packs one E2M1 nibble per element (values chosen from the E2M1 grid).
std::vector<uint8_t> PackNibbles(const std::vector<float> &v)
{
    std::vector<uint8_t> out((v.size() + 1) / 2, 0);
    for (size_t i = 0; i < v.size(); i += 2) {
        const int lo = static_cast<int>(v[i]);
        const int hi = (i + 1 < v.size()) ? static_cast<int>(v[i + 1]) : 0;
        out[i / 2] = static_cast<uint8_t>(lo | (hi << 4));
    }
    return out;
}

}  // namespace

TEST_CASE("kimi-k3 situ-glu matches the reference activation",
          "[kimi-k3][kernels]")
{
    const int n = 257;
    std::vector<__half> g(n), u(n), out(n);
    for (int i = 0; i < n; ++i) {
        g[static_cast<size_t>(i)] = H(-3.0f + 0.037f * static_cast<float>(i % 101));
        u[static_cast<size_t>(i)] = H(-40.0f + 0.31f * static_cast<float>(i % 97));
    }
    __half *dg = nullptr, *du = nullptr, *dout = nullptr;
    cudaMalloc(&dg, n * sizeof(__half));
    cudaMalloc(&du, n * sizeof(__half));
    cudaMalloc(&dout, n * sizeof(__half));
    cudaMemcpy(dg, g.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(du, u.data(), n * sizeof(__half), cudaMemcpyHostToDevice);
    REQUIRE(RunKimiK3Situ(0, dg, du, dout, n, 4.0f, 25.0f) == cudaSuccess);
    cudaMemcpy(out.data(), dout, n * sizeof(__half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        const float gv = F(g[static_cast<size_t>(i)]);
        const float uv = F(u[static_cast<size_t>(i)]);
        const float a = 4.0f * std::tanh(gv / 4.0f) / (1.0f + std::exp(-gv));
        const float up = 25.0f * std::tanh(uv / 25.0f);
        INFO("i=" << i);
        // FP16 output window: relative tolerance for O(1)-magnitude values.
        REQUIRE(std::fabs(F(out[static_cast<size_t>(i)]) - a * up) <=
                1.5e-3f * (1.0f + std::fabs(a * up)));
    }
    cudaFree(dg);
    cudaFree(du);
    cudaFree(dout);
}

TEST_CASE("kimi-k3 router selects by sigmoid+bias and renormalizes the raw scores",
          "[kimi-k3][kernels]")
{
    const int H_ = 16, E = 24, K = 4;
    std::vector<__half> x(static_cast<size_t>(H_)), w(static_cast<size_t>(E) * H_);
    std::vector<float> bias(static_cast<size_t>(E));
    for (int i = 0; i < H_; ++i) x[static_cast<size_t>(i)] = H(0.05f * (i - 7));
    for (int e = 0; e < E; ++e) {
        for (int i = 0; i < H_; ++i)
            w[static_cast<size_t>(e) * H_ + i] =
                H(0.01f * ((e * 7 + i * 3) % 19) - 0.1f);
        bias[static_cast<size_t>(e)] = 0.01f * ((e * 5) % 11);
    }
    __half *dx = nullptr, *dw = nullptr;
    float *dbias = nullptr, *dsel = nullptr, *draw = nullptr;
    int32_t *didx = nullptr;
    float *dwts = nullptr;
    cudaMalloc(&dx, H_ * sizeof(__half));
    cudaMalloc(&dw, static_cast<size_t>(E) * H_ * sizeof(__half));
    cudaMalloc(&dbias, E * sizeof(float));
    cudaMalloc(&dsel, E * sizeof(float));
    cudaMalloc(&draw, E * sizeof(float));
    cudaMalloc(&didx, K * sizeof(int32_t));
    cudaMalloc(&dwts, K * sizeof(float));
    cudaMemcpy(dx, x.data(), H_ * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), static_cast<size_t>(E) * H_ * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(dbias, bias.data(), E * sizeof(float), cudaMemcpyHostToDevice);

    REQUIRE(RunKimiK3RouterScores(0, dx, dw, dbias, H_, E, dsel, draw) ==
            cudaSuccess);
    REQUIRE(RunKimiK3RouterTopK(0, dsel, draw, E, K, 1.0f, 1, didx, dwts) ==
            cudaSuccess);
    std::vector<int32_t> idx(static_cast<size_t>(K));
    std::vector<float> wts(static_cast<size_t>(K));
    cudaMemcpy(idx.data(), didx, K * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(wts.data(), dwts, K * sizeof(float), cudaMemcpyDeviceToHost);

    // Host: same selection (sel desc, index asc), raw sigmoid weights renormed.
    std::vector<float> sel(static_cast<size_t>(E)), raw(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) {
        float dot = 0.0f;
        for (int i = 0; i < H_; ++i)
            dot += F(x[static_cast<size_t>(i)]) *
                   F(w[static_cast<size_t>(e) * H_ + i]);
        raw[static_cast<size_t>(e)] = 1.0f / (1.0f + std::exp(-dot));
        sel[static_cast<size_t>(e)] = raw[static_cast<size_t>(e)] +
                                      bias[static_cast<size_t>(e)];
    }
    std::vector<int> order(static_cast<size_t>(E));
    for (int e = 0; e < E; ++e) order[static_cast<size_t>(e)] = e;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (sel[static_cast<size_t>(a)] != sel[static_cast<size_t>(b)])
            return sel[static_cast<size_t>(a)] > sel[static_cast<size_t>(b)];
        return a < b;
    });
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += raw[static_cast<size_t>(order[k])];
    for (int k = 0; k < K; ++k) {
        INFO("k=" << k);
        REQUIRE(idx[static_cast<size_t>(k)] == order[static_cast<size_t>(k)]);
        REQUIRE(std::fabs(wts[static_cast<size_t>(k)] -
                          raw[static_cast<size_t>(order[k])] / sum) < 1e-6f);
    }
    cudaFree(dx); cudaFree(dw); cudaFree(dbias); cudaFree(dsel);
    cudaFree(draw); cudaFree(didx); cudaFree(dwts);
}

TEST_CASE("kimi-k3 attnres depth mixer is an rms-dot softmax over sources",
          "[kimi-k3][kernels]")
{
    const int hidden = 32, n_src = 3;
    std::vector<__half> src(static_cast<size_t>(hidden) * n_src), nw(hidden),
        pw(hidden), out(hidden);
    for (int s = 0; s < n_src; ++s)
        for (int i = 0; i < hidden; ++i)
            src[static_cast<size_t>(s) * hidden + i] =
                H(0.1f * ((s + 1) * (i % 5) - 0.5f));
    for (int i = 0; i < hidden; ++i) {
        nw[static_cast<size_t>(i)] = H(0.5f + 0.01f * i);
        pw[static_cast<size_t>(i)] = H(0.03f * ((i % 7) - 3));
    }
    __half *d_src = nullptr, *d_nw = nullptr, *d_pw = nullptr, *d_out = nullptr;
    cudaMalloc(&d_src, src.size() * sizeof(__half));
    cudaMalloc(&d_nw, hidden * sizeof(__half));
    cudaMalloc(&d_pw, hidden * sizeof(__half));
    cudaMalloc(&d_out, hidden * sizeof(__half));
    cudaMemcpy(d_src, src.data(), src.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_nw, nw.data(), hidden * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pw, pw.data(), hidden * sizeof(__half), cudaMemcpyHostToDevice);
    REQUIRE(RunKimiK3DepthMix(0, d_src, n_src, d_nw, d_pw, d_out, hidden, 1e-5f) ==
            cudaSuccess);
    cudaMemcpy(out.data(), d_out, hidden * sizeof(__half),
               cudaMemcpyDeviceToHost);

    std::vector<float> logits(static_cast<size_t>(n_src));
    for (int s = 0; s < n_src; ++s) {
        float sq = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            const float v = F(src[static_cast<size_t>(s) * hidden + i]);
            sq += v * v;
        }
        const float rstd = 1.0f / std::sqrt(sq / hidden + 1e-5f);
        float dot = 0.0f;
        for (int i = 0; i < hidden; ++i)
            dot += F(src[static_cast<size_t>(s) * hidden + i]) * rstd *
                   F(nw[static_cast<size_t>(i)]) * F(pw[static_cast<size_t>(i)]);
        logits[static_cast<size_t>(s)] = dot;
    }
    float mx = logits[0];
    for (float v : logits) mx = std::max(mx, v);
    float sum = 0.0f;
    for (auto &v : logits) {
        v = std::exp(v - mx);
        sum += v;
    }
    for (int i = 0; i < hidden; ++i) {
        float acc = 0.0f;
        for (int s = 0; s < n_src; ++s)
            acc += logits[static_cast<size_t>(s)] / sum *
                   F(src[static_cast<size_t>(s) * hidden + i]);
        INFO("i=" << i);
        REQUIRE(std::fabs(F(out[static_cast<size_t>(i)]) - acc) < 2e-3f);
    }
    cudaFree(d_src); cudaFree(d_nw); cudaFree(d_pw); cudaFree(d_out);
}

TEST_CASE("kimi-k3 KDA conv+recurrence reproduce the channel-wise delta rule",
          "[kimi-k3][kernels]")
{
    const int KH = 2, KD = 8, VD = 8, k = 4;
    const float lower_bound = -5.0f;
    std::vector<__half> xq(static_cast<size_t>(KH) * KD),
        xk(static_cast<size_t>(KH) * KD), xv(static_cast<size_t>(KH) * VD),
        gx(static_cast<size_t>(KH) * KD), beta(static_cast<size_t>(KH));
    for (int i = 0; i < KH * KD; ++i) {
        xq[static_cast<size_t>(i)] = H(0.3f * std::sin(0.7f * i));
        xk[static_cast<size_t>(i)] = H(0.4f * std::cos(0.3f * i));
        gx[static_cast<size_t>(i)] = H(0.5f * std::sin(0.11f * i));
    }
    for (int i = 0; i < KH * VD; ++i)
        xv[static_cast<size_t>(i)] = H(0.25f * std::sin(0.5f * i + 1.0f));
    std::vector<float> alog(KD), dt(static_cast<size_t>(KH) * KD),
        conv_w(static_cast<size_t>(KH) * KD * k);
    for (int d = 0; d < KD; ++d) alog[static_cast<size_t>(d)] = 0.3f + 0.1f * d;
    for (int i = 0; i < KH * KD; ++i) dt[static_cast<size_t>(i)] = 0.1f * (i % 5);
    for (int i = 0; i < KH * KD * k; ++i)
        conv_w[static_cast<size_t>(i)] = 0.2f * std::sin(0.9f * i);
    for (int h = 0; h < KH; ++h) beta[static_cast<size_t>(h)] = H(0.4f - 0.2f * h);

    __half *dq = nullptr, *dk = nullptr, *dv = nullptr, *dg = nullptr,
        *dbeta = nullptr, *dout = nullptr, *dconv_q = nullptr, *dconv_k = nullptr,
        *dconv_v = nullptr;
    float *d_alog = nullptr, *d_dt = nullptr, *d_convw = nullptr,
        *d_state = nullptr, *d_hist = nullptr;
    cudaMalloc(&dq, KH * KD * sizeof(__half));
    cudaMalloc(&dk, KH * KD * sizeof(__half));
    cudaMalloc(&dv, KH * VD * sizeof(__half));
    cudaMalloc(&dg, KH * KD * sizeof(__half));
    cudaMalloc(&dbeta, KH * sizeof(__half));
    cudaMalloc(&dout, KH * VD * sizeof(__half));
    cudaMalloc(&dconv_q, KH * KD * sizeof(__half));
    cudaMalloc(&dconv_k, KH * KD * sizeof(__half));
    cudaMalloc(&dconv_v, KH * VD * sizeof(__half));
    cudaMalloc(&d_alog, KD * sizeof(float));
    cudaMalloc(&d_dt, KH * KD * sizeof(float));
    cudaMalloc(&d_convw, conv_w.size() * sizeof(float));
    cudaMalloc(&d_state, static_cast<size_t>(KH) * KD * VD * sizeof(float));
    cudaMalloc(&d_hist, static_cast<size_t>(3) * KH * KD * 3 * sizeof(float));
    cudaMemcpy(dq, xq.data(), KH * KD * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, xk.data(), KH * KD * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, xv.data(), KH * VD * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dg, gx.data(), KH * KD * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, beta.data(), KH * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alog, alog.data(), KD * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt, dt.data(), KH * KD * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_convw, conv_w.data(), conv_w.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(d_state, 0, static_cast<size_t>(KH) * KD * VD * sizeof(float));
    cudaMemset(d_hist, 0, static_cast<size_t>(3) * KH * KD * 3 * sizeof(float));

    REQUIRE(RunKimiK3KdaConv(0, dq, d_convw, d_hist, dconv_q, KH * KD, k) ==
            cudaSuccess);
    REQUIRE(RunKimiK3KdaConv(0, dk, d_convw, d_hist + KH * KD * 3, dconv_k,
                             KH * KD, k) == cudaSuccess);
    REQUIRE(RunKimiK3KdaConv(0, dv, d_convw, d_hist + 2 * KH * KD * 3, dconv_v,
                             KH * VD, k) == cudaSuccess);
    REQUIRE(RunKimiK3KdaRecur(0, dconv_q, dconv_k, dconv_v, dg, dbeta, d_alog,
                              d_dt, d_state, dout, KH, KH, KD, VD,
                              lower_bound) == cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(KH) * VD);
    cudaMemcpy(out.data(), dout, KH * VD * sizeof(__half),
               cudaMemcpyDeviceToHost);
    // Conv outputs feed the recurrence; compare them first so a mismatch is
    // attributed to the right kernel.
    {
        std::vector<__half> cq(static_cast<size_t>(KH) * KD), ck(cq.size()),
            cv(static_cast<size_t>(KH) * VD);
        cudaMemcpy(cq.data(), dconv_q, cq.size() * sizeof(__half),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(ck.data(), dconv_k, ck.size() * sizeof(__half),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(cv.data(), dconv_v, cv.size() * sizeof(__half),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < KH * KD; ++i) {
            const float a = conv_w[static_cast<size_t>(i) * k + (k - 1)] *
                            F(xq[static_cast<size_t>(i)]);
            const float want = F(H(a / (1.0f + std::exp(-a))));
            INFO("i=" << i << " got=" << F(cq[static_cast<size_t>(i)])
                      << " want=" << want);
            REQUIRE(std::fabs(F(cq[static_cast<size_t>(i)]) - want) < 1e-4f);
            const float ak = conv_w[static_cast<size_t>(i) * k + (k - 1)] *
                             F(xk[static_cast<size_t>(i)]);
            const float wantk = F(H(ak / (1.0f + std::exp(-ak))));
            REQUIRE(std::fabs(F(ck[static_cast<size_t>(i)]) - wantk) < 1e-4f);
        }
        for (int i = 0; i < KH * VD; ++i) {
            const float a = conv_w[static_cast<size_t>(i) * k + (k - 1)] *
                            F(xv[static_cast<size_t>(i)]);
            const float want = F(H(a / (1.0f + std::exp(-a))));
            REQUIRE(std::fabs(F(cv[static_cast<size_t>(i)]) - want) < 1e-4f);
        }
    }

    // Host reference (fresh state, one step, empty conv history).
    auto sigmoid = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    for (int h = 0; h < KH; ++h) {
        std::vector<float> q(KD), kk(KD);
        float qn = 0.0f, kn = 0.0f;
        for (int d = 0; d < KD; ++d) {
            // conv over an all-zero history keeps only the current input.
            const float cv =
                conv_w[static_cast<size_t>(h) * KD * k + d * k + (k - 1)];
            q[d] = cv * F(xq[static_cast<size_t>(h) * KD + d]);
            kk[d] = cv * F(xk[static_cast<size_t>(h) * KD + d]);
            // The conv output is stored as FP16 by the kernel; round the host
            // reference the same way before the recurrence.
            q[d] = F(H(q[d] / (1.0f + std::exp(-q[d]))));
            kk[d] = F(H(kk[d] / (1.0f + std::exp(-kk[d]))));
            qn += q[d] * q[d];
            kn += kk[d] * kk[d];
        }
        const float qs = 1.0f / std::sqrt(qn + 1e-6f) / std::sqrt(static_cast<float>(KD));
        const float ks = 1.0f / std::sqrt(kn + 1e-6f);
        std::vector<float> alp(KD);
        for (int d = 0; d < KD; ++d) {
            const float a = F(gx[static_cast<size_t>(h) * KD + d]) +
                            dt[static_cast<size_t>(h) * KD + d];
            alp[static_cast<size_t>(d)] = std::exp(lower_bound *
                sigmoid(std::exp(alog[static_cast<size_t>(d)]) * a));
        }
        const float b = sigmoid(F(beta[static_cast<size_t>(h)]));
        for (int vd = 0; vd < VD; ++vd) {
            const float cvv =
                conv_w[static_cast<size_t>(h) * VD * k + vd * k + (k - 1)];
            float vvv = cvv * F(xv[static_cast<size_t>(h) * VD + vd]);
            vvv = F(H(vvv / (1.0f + std::exp(-vvv))));
            float o = 0.0f, abssum = 0.0f;
            for (int d = 0; d < KD; ++d) {
                // Fresh state: the channel decay multiplies the OLD state, so
                // the current-token contribution carries no alpha factor.
                const float term = (q[d] * qs) * (kk[d] * ks) * (vvv * b);
                o += term;
                abssum += std::fabs(term);
            }
            // The output is FP16 and the sum can cancel; tolerance scales with
            // the term magnitude (FP16 inputs), not with the (tiny) result.
            const float tol = 3e-3f * abssum + 1e-6f;
            INFO("h=" << h << " vd=" << vd << " kernel="
                      << F(out[static_cast<size_t>(h) * VD + vd]) << " host=" << o);
            REQUIRE(std::fabs(F(out[static_cast<size_t>(h) * VD + vd]) - o) <=
                    tol);
        }
    }
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dg); cudaFree(dbeta);
    cudaFree(dout); cudaFree(dconv_q); cudaFree(dconv_k); cudaFree(dconv_v);
    cudaFree(d_alog); cudaFree(d_dt); cudaFree(d_convw); cudaFree(d_state);
    cudaFree(d_hist);
}

TEST_CASE("kimi-k3 absorbed NoPE MLA matches the reference attention",
          "[kimi-k3][kernels]")
{
    const int heads = 2, nope = 4, rope = 2, L = 8, V = 4, n_pos = 3;
    const int rows_per_head = nope + V;
    const int row_packed = L / 2;
    const int row_scales = L / 16 > 0 ? L / 16 : 1;

    // kv_b rows: per head [nope rows = W_k | V rows = W_v], each L wide.
    // Pick nibble indices so the dequantized values are exact.
    std::vector<float> wv(static_cast<size_t>(heads) * rows_per_head * L, 0.0f);
    for (int h = 0; h < heads; ++h)
        for (int n = 0; n < rows_per_head; ++n)
            for (int c = 0; c < L; ++c) {
                const int nib = (h + n + c) % 5;  // 0..4 -> 0,0.5,1,1.5,2
                wv[static_cast<size_t>((h * rows_per_head + n) * L + c)] =
                    static_cast<float>(nib);
            }
    std::vector<uint8_t> packed = PackNibbles(wv);
    std::vector<uint8_t> scales(static_cast<size_t>(heads) * rows_per_head *
                                    row_scales,
                                0x38);  // E4M3 1.0
    REQUIRE(packed.size() ==
            static_cast<size_t>(heads) * rows_per_head * row_packed);

    std::vector<__half> q_nope(static_cast<size_t>(heads) * nope),
        q_rot(static_cast<size_t>(heads) * rope),
        latent(static_cast<size_t>(n_pos) * (L + rope));
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < nope; ++i)
            q_nope[static_cast<size_t>(h) * nope + i] = H(0.1f * (i + 1 + h));
        for (int i = 0; i < rope; ++i)
            q_rot[static_cast<size_t>(h) * rope + i] = H(0.2f * (i + 1));
    }
    for (int t = 0; t < n_pos; ++t)
        for (int c = 0; c < L + rope; ++c)
            latent[static_cast<size_t>(t) * (L + rope) + c] =
                H(0.05f * ((t + 1) * (c + 2) % 13) - 0.2f);

    uint8_t *d_packed = nullptr, *d_scales = nullptr;
    __half *d_qn = nullptr, *d_qr = nullptr, *d_lat = nullptr, *d_qabs = nullptr,
        *d_out = nullptr;
    cudaMalloc(&d_packed, packed.size());
    cudaMalloc(&d_scales, scales.size());
    cudaMalloc(&d_qn, q_nope.size() * sizeof(__half));
    cudaMalloc(&d_qr, q_rot.size() * sizeof(__half));
    cudaMalloc(&d_lat, latent.size() * sizeof(__half));
    cudaMalloc(&d_qabs, static_cast<size_t>(heads) * L * sizeof(__half));
    cudaMalloc(&d_out, static_cast<size_t>(heads) * V * sizeof(__half));
    cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qn, q_nope.data(), q_nope.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_qr, q_rot.data(), q_rot.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_lat, latent.data(), latent.size() * sizeof(__half),
               cudaMemcpyHostToDevice);

    REQUIRE(RunKimiK3MlaAbsorbQ(0, d_qn, d_packed, d_scales, 1.0f, d_qabs, heads,
                                nope, L, rows_per_head, 0, nope, row_packed,
                                row_scales) == cudaSuccess);
    const float scale = 1.0f / std::sqrt(static_cast<float>(nope + rope));
    REQUIRE(RunKimiK3MlaAttention(0, d_lat, n_pos, d_qabs, d_qr, d_packed,
                                  d_scales, 1.0f, d_out, heads, L, rope, V,
                                  rows_per_head, nope, rope, scale, row_packed,
                                  row_scales, nullptr, 0) == cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(heads) * V), qabs(
        static_cast<size_t>(heads) * L);
    cudaMemcpy(out.data(), d_out, out.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(qabs.data(), d_qabs, qabs.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);

    // Host reference: q_abs, scores over the cached latents, softmax, ctx, out.
    for (int h = 0; h < heads; ++h) {
        for (int c = 0; c < L; ++c) {
            float acc = 0.0f;
            for (int n = 0; n < nope; ++n) {
                const size_t row = static_cast<size_t>(h * rows_per_head + n);
                const int nib = wv[row * L + c] == 0.0f
                                    ? 0
                                    : static_cast<int>(wv[row * L + c]);
                acc += F(q_nope[static_cast<size_t>(h) * nope + n]) *
                       E2m1(nib) * E4m3(0x38);
            }
            INFO("qabs h=" << h << " c=" << c);
            REQUIRE(std::fabs(F(qabs[static_cast<size_t>(h) * L + c]) - acc) <
                    3e-3f);
        }
        std::vector<float> score(static_cast<size_t>(n_pos));
        for (int t = 0; t < n_pos; ++t) {
            float acc = 0.0f;
            for (int c = 0; c < L; ++c) {
                const int nib = wv[static_cast<size_t>(
                    (h * rows_per_head + c % nope) * L + c)] == 0.0f
                                    ? 0
                                    : static_cast<int>(
                                          wv[static_cast<size_t>((h * rows_per_head + c % nope) * L + c)]);
                (void)nib;
                acc += F(qabs[static_cast<size_t>(h) * L + c]) *
                       F(latent[static_cast<size_t>(t) * (L + rope) + c]);
            }
            for (int r = 0; r < rope; ++r)
                acc += F(q_rot[static_cast<size_t>(h) * rope + r]) *
                       F(latent[static_cast<size_t>(t) * (L + rope) + L + r]);
            score[static_cast<size_t>(t)] = acc * scale;
        }
        float mx = score[0];
        for (float v : score) mx = std::max(mx, v);
        float sum = 0.0f;
        for (auto &v : score) {
            v = std::exp(v - mx);
            sum += v;
        }
        std::vector<float> ctx(static_cast<size_t>(L), 0.0f);
        for (int t = 0; t < n_pos; ++t)
            for (int c = 0; c < L; ++c)
                ctx[static_cast<size_t>(c)] +=
                    score[static_cast<size_t>(t)] / sum *
                    F(latent[static_cast<size_t>(t) * (L + rope) + c]);
        for (int vd = 0; vd < V; ++vd) {
            float acc = 0.0f;
            for (int c = 0; c < L; ++c) {
                const size_t row =
                    static_cast<size_t>(h * rows_per_head + nope + vd);
                const int nib = wv[row * L + c] == 0.0f
                                    ? 0
                                    : static_cast<int>(wv[row * L + c]);
                acc += E2m1(nib) * E4m3(0x38) * ctx[static_cast<size_t>(c)];
            }
            INFO("h=" << h << " vd=" << vd);
            REQUIRE(std::fabs(F(out[static_cast<size_t>(h) * V + vd]) - acc) <
                    5e-3f);
        }
    }
    cudaFree(d_packed); cudaFree(d_scales); cudaFree(d_qn); cudaFree(d_qr);
    cudaFree(d_lat); cudaFree(d_qabs); cudaFree(d_out);
}
