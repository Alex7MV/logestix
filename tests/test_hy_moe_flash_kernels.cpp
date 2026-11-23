// Hy4-flash family kernels against host references computed in the test from the
// same inputs. These are the four places a silent wrong number is otherwise
// invisible in a green end-to-end run:
//
//   * iHC: `post` is PER STREAM and multiplies the whole sublayer output before
//     it is added back, with magnitude 2.0 and no Sinkhorn;
//   * the sandwich norm applies to the SUBLAYER OUTPUT, not the residual;
//   * the router: fp32 sigmoid scores, a selection-only bias, renormalized
//     weights then the routed scaling factor;
//   * sink attention: the sink participates in the softmax denominator and is
//     dropped from the output.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>

#include "lsxcommon/hy_moe_flash_kernels.h"

namespace {

namespace gpu = lsxcommon::hy4::gpu;

template <typename T>
T *DevAlloc(size_t n)
{
    void *p = nullptr;
    REQUIRE(cudaMalloc(&p, n * sizeof(T)) == cudaSuccess);
    return static_cast<T *>(p);
}

void DevFree(void *p) { cudaFree(p); }

void Upload(void *dst, const void *src, size_t bytes)
{
    REQUIRE(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess);
}

void Download(void *dst, const void *src, size_t bytes)
{
    REQUIRE(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
}

__half H(float v) { return __float2half_rn(v); }

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace

TEST_CASE("hy4 kernels: iHC mix and per-stream post", "[hy4][kernels]")
{
    constexpr int kStreams = 4;
    constexpr int kHidden = 8;
    constexpr int kRows = 2 * kStreams;
    constexpr float kEps = 1e-6f;
    constexpr float kMag = 2.0f;

    std::vector<float> x(static_cast<size_t>(kStreams) * kHidden);
    for (int j = 0; j < kStreams; ++j)
        for (int i = 0; i < kHidden; ++i)
            x[static_cast<size_t>(j) * kHidden + i] =
                0.1f * static_cast<float>(j + 1) + 0.01f * static_cast<float>(i);
    std::vector<float> fn(static_cast<size_t>(kRows) * kStreams * kHidden);
    for (size_t i = 0; i < fn.size(); ++i)
        fn[i] = 0.001f * static_cast<float>((i * 7) % 13) - 0.004f;
    std::vector<float> base = {0.1f, -0.2f, 0.3f, 0.0f, -0.1f, 0.2f, 0.0f, 0.4f};
    std::vector<float> scale = {0.5f, 1.5f};

    float *d_x = DevAlloc<float>(x.size());
    float *d_fn = DevAlloc<float>(fn.size());
    float *d_base = DevAlloc<float>(base.size());
    float *d_scale = DevAlloc<float>(scale.size());
    float *d_agg = DevAlloc<float>(kHidden);
    float *d_pre = DevAlloc<float>(kStreams);
    float *d_post = DevAlloc<float>(kStreams);
    float *d_mix = DevAlloc<float>(kRows);
    float *d_rsqrt = DevAlloc<float>(1);
    Upload(d_x, x.data(), x.size() * 4);
    Upload(d_fn, fn.data(), fn.size() * 4);
    Upload(d_base, base.data(), base.size() * 4);
    Upload(d_scale, scale.data(), scale.size() * 4);

    REQUIRE(gpu::RunIhcMix(d_x, d_fn, d_base, d_scale, kStreams, kHidden, kEps,
                           kMag, d_agg, d_pre, d_post, d_mix, d_rsqrt,
                           nullptr) == cudaSuccess);
    std::vector<float> agg(kHidden), pre(kStreams), post(kStreams), mix(kRows);
    Download(agg.data(), d_agg, kHidden * 4);
    Download(pre.data(), d_pre, kStreams * 4);
    Download(post.data(), d_post, kStreams * 4);
    Download(mix.data(), d_mix, kRows * 4);

    // Host reference, straight from the reference formula.
    double ss = 0.0;
    for (float v : x) ss += static_cast<double>(v) * v;
    const float rsqrt =
        1.0f / std::sqrt(static_cast<float>(ss / (kStreams * kHidden)) + kEps);
    std::vector<float> mix_ref(kRows);
    for (int r = 0; r < kRows; ++r) {
        double d = 0.0;
        for (int k = 0; k < kStreams * kHidden; ++k)
            d += static_cast<double>(fn[static_cast<size_t>(r) * kStreams * kHidden + k]) * x[k];
        mix_ref[r] = static_cast<float>(d) * rsqrt;
    }
    for (int r = 0; r < kRows; ++r) CHECK(mix[r] == Approx(mix_ref[r]).margin(1e-4));
    for (int j = 0; j < kStreams; ++j) {
        const float p = Sigmoid(mix_ref[j] * scale[0] + base[j]) + kEps;
        const float po = kMag * Sigmoid(mix_ref[kStreams + j] * scale[1] +
                                        base[kStreams + j]) + kEps;
        CHECK(pre[j] == Approx(p).margin(1e-5));
        CHECK(post[j] == Approx(po).margin(1e-5));
    }
    for (int i = 0; i < kHidden; ++i) {
        float want = 0.0f;
        for (int j = 0; j < kStreams; ++j)
            want += pre[j] * x[static_cast<size_t>(j) * kHidden + i];
        CHECK(agg[i] == Approx(want).margin(1e-5));
    }

    // The post update must scale each stream by ITS OWN post value: a kernel
    // that used the mean (or one scalar) changes every layer silently.
    std::vector<float> streams = x;
    std::vector<float> agg_in(kHidden);
    for (int i = 0; i < kHidden; ++i) agg_in[i] = 1.0f;
    Upload(d_agg, agg_in.data(), kHidden * 4);
    Upload(d_x, streams.data(), streams.size() * 4);
    REQUIRE(gpu::RunIhcPost(d_agg, d_post, kStreams, kHidden, d_x, nullptr) ==
            cudaSuccess);
    Download(streams.data(), d_x, streams.size() * 4);
    for (int j = 0; j < kStreams; ++j)
        for (int i = 0; i < kHidden; ++i)
            CHECK(streams[static_cast<size_t>(j) * kHidden + i] ==
                  Approx(x[static_cast<size_t>(j) * kHidden + i] + post[j])
                      .margin(1e-5));
    // Non-uniform post by construction.
    CHECK(std::abs(post[0] - post[3]) > 1e-3f);

    DevFree(d_x);
    DevFree(d_fn);
    DevFree(d_base);
    DevFree(d_scale);
    DevFree(d_agg);
    DevFree(d_pre);
    DevFree(d_post);
    DevFree(d_mix);
    DevFree(d_rsqrt);
}

TEST_CASE("hy4 kernels: sandwich norm applies to the sublayer output",
          "[hy4][kernels]")
{
    constexpr int kDim = 256;
    constexpr float kEps = 1e-5f;
    std::vector<__half> x(kDim), w(kDim);
    for (int i = 0; i < kDim; ++i) {
        x[i] = H(0.5f + 0.01f * static_cast<float>(i));
        w[i] = H(1.0f + 0.001f * static_cast<float>(i));
    }
    __half *d_x = DevAlloc<__half>(kDim);
    __half *d_w = DevAlloc<__half>(kDim);
    Upload(d_x, x.data(), kDim * 2);
    Upload(d_w, w.data(), kDim * 2);
    REQUIRE(gpu::RunRmsNormHalf(d_x, d_w, kDim, kEps, nullptr) == cudaSuccess);
    std::vector<__half> got(kDim);
    Download(got.data(), d_x, kDim * 2);

    double ss = 0.0;
    for (auto v : x) ss += static_cast<double>(__half2float(v)) * __half2float(v);
    const float rs = 1.0f / std::sqrt(static_cast<float>(ss / kDim) + kEps);
    for (int i = 0; i < kDim; ++i) {
        const float want = __half2float(x[i]) * rs * __half2float(w[i]);
        CHECK(__half2float(got[i]) == Approx(want).margin(2e-3));
    }
    DevFree(d_x);
    DevFree(d_w);
}

TEST_CASE("hy4 kernels: router selects on bias but scales on score",
          "[hy4][kernels]")
{
    constexpr int kRouted = 512;
    constexpr int kHidden = 3072;
    constexpr int kTopK = 16;
    constexpr float kScale = 3.998f;
    std::vector<__half> x(kHidden), gate(static_cast<size_t>(kRouted) * kHidden);
    for (int i = 0; i < kHidden; ++i)
        x[i] = H(0.01f * static_cast<float>((i % 17) - 8));
    for (size_t i = 0; i < gate.size(); ++i)
        gate[i] = H(0.02f * static_cast<float>(static_cast<int>(i % 11) - 5));
    // Expert 500 has a strong positive bias: it must be SELECTED even though its
    // score is unremarkable, and its weight must come from the score alone.
    std::vector<float> bias(kRouted, 0.0f);
    bias[500] = 5.0f;
    bias[501] = -5.0f;

    __half *d_x = DevAlloc<__half>(kHidden);
    __half *d_gate = DevAlloc<__half>(gate.size());
    float *d_bias = DevAlloc<float>(kRouted);
    float *d_scores = DevAlloc<float>(kRouted);
    int32_t *d_idx = DevAlloc<int32_t>(kTopK);
    float *d_w = DevAlloc<float>(kTopK);
    Upload(d_x, x.data(), kHidden * 2);
    Upload(d_gate, gate.data(), gate.size() * 2);
    Upload(d_bias, bias.data(), kRouted * 4);
    REQUIRE(gpu::RunRouterLogits(d_x, d_gate, kRouted, kHidden, d_scores,
                                 nullptr) == cudaSuccess);
    REQUIRE(gpu::RunRouterTopK(d_scores, d_bias, kRouted, kTopK, kScale, d_idx,
                               d_w, nullptr) == cudaSuccess);
    std::vector<int32_t> idx(kTopK);
    std::vector<float> w(kTopK), scores(kRouted);
    Download(idx.data(), d_idx, kTopK * 4);
    Download(w.data(), d_w, kTopK * 4);
    Download(scores.data(), d_scores, kRouted * 4);

    INFO("scores[0..3] " << scores[0] << " " << scores[1] << " " << scores[2]
                         << " " << scores[3]);
    INFO("idx[0..3] " << idx[0] << " " << idx[1] << " " << idx[2] << " "
                      << idx[3]);
    INFO("w[0..3] " << w[0] << " " << w[1] << " " << w[2] << " " << w[3]);
    // Host reference.
    std::vector<float> choice(kRouted);
    for (int j = 0; j < kRouted; ++j) choice[j] = scores[j] + bias[j];
    std::vector<int> order(kRouted);
    for (int j = 0; j < kRouted; ++j) order[j] = j;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (choice[a] != choice[b]) return choice[a] > choice[b];
        return a < b;
    });
    std::vector<int32_t> want_idx(order.begin(), order.begin() + kTopK);
    CHECK(idx == want_idx);
    CHECK(std::find(idx.begin(), idx.end(), 500) != idx.end());
    CHECK(std::find(idx.begin(), idx.end(), 501) == idx.end());
    float sum = 0.0f;
    for (int k = 0; k < kTopK; ++k) sum += scores[idx[static_cast<size_t>(k)]];
    for (int k = 0; k < kTopK; ++k) {
        const float want =
            scores[idx[static_cast<size_t>(k)]] / (sum + 1e-20f) * kScale;
        CHECK(w[static_cast<size_t>(k)] == Approx(want).margin(1e-5));
    }
    float total = 0.0f;
    for (float v : w) total += v;
    CHECK(total == Approx(kScale).margin(1e-4));

    DevFree(d_x);
    DevFree(d_gate);
    DevFree(d_bias);
    DevFree(d_scores);
    DevFree(d_idx);
    DevFree(d_w);
}

TEST_CASE("hy4 kernels: sliding-window position list", "[hy4][kernels]")
{
    constexpr int kWindow = 512;
    int32_t *d_pos = DevAlloc<int32_t>(kWindow);
    std::vector<int32_t> pos(kWindow);
    for (int p : {0, 1, 511, 512, 513, 5000}) {
        REQUIRE(gpu::RunWindowPositions(d_pos, kWindow, p, nullptr) ==
                cudaSuccess);
        Download(pos.data(), d_pos, kWindow * 4);
        // The list is always `kWindow` long and ASCENDING: entries before the
        // first cached position are -1 and the last entry is always `p`. That is
        // what lets the caller shift by (kWindow - npos) and pass a sub-range.
        CHECK(pos[static_cast<size_t>(kWindow - 1)] == static_cast<int32_t>(p));
        int visible = 0;
        int32_t prev = 0;
        for (int t = 0; t < kWindow; ++t) {
            const int32_t v = pos[static_cast<size_t>(t)];
            if (v < 0) {
                // padding only ever precedes the visible run
                CHECK(visible == 0);
                continue;
            }
            ++visible;
            CHECK(v <= p);
            CHECK(v >= p - kWindow + 1);
            if (visible == 1)
                CHECK(v == std::max(0, p - kWindow + 1));
            else
                CHECK(v == prev + 1);   // contiguous, no gaps
            prev = v;
        }
        CHECK(visible == std::min(p + 1, kWindow));
        // The visible run ends at p, so the caller's shift is exact.
        CHECK(pos[static_cast<size_t>(kWindow - visible)] ==
              std::max(0, p - kWindow + 1));
    }
    DevFree(d_pos);
}

TEST_CASE("hy4 kernels: swiglu clamps before the activation", "[hy4][kernels]")
{
    constexpr int kN = 64;
    constexpr float kLimit = 10.0f;
    std::vector<__half> g(kN), u(kN);
    for (int i = 0; i < kN; ++i) {
        g[i] = H(-20.0f + static_cast<float>(i));
        u[i] = H(-15.0f + 0.5f * static_cast<float>(i));
    }
    __half *d_g = DevAlloc<__half>(kN);
    __half *d_u = DevAlloc<__half>(kN);
    __half *d_o = DevAlloc<__half>(kN);
    Upload(d_g, g.data(), kN * 2);
    Upload(d_u, u.data(), kN * 2);
    REQUIRE(gpu::RunSwigluClamp(d_g, d_u, d_o, kN, kLimit, nullptr) ==
            cudaSuccess);
    std::vector<__half> out(kN);
    Download(out.data(), d_o, kN * 2);
    for (int i = 0; i < kN; ++i) {
        const float gi = __half2float(g[i]);
        const float ui = __half2float(u[i]);
        const float gc = std::min(gi, kLimit);       // no lower bound
        const float uc = std::max(-kLimit, std::min(ui, kLimit));
        const float want = (gc / (1.0f + std::exp(-gc))) * uc;
        CHECK(__half2float(out[i]) == Approx(want).margin(2e-2));
    }
    DevFree(d_g);
    DevFree(d_u);
    DevFree(d_o);
}

TEST_CASE("hy4 kernels: sink softmax drops the sink and keeps the mass",
          "[hy4][kernels]")
{
    // One head, one 8-dim kv latent, three visible positions. The sink is a
    // logit: it takes probability mass away but contributes no value.
    constexpr int kHeads = 1, kKv = 8, kRope = 4, kNpos = 3;
    constexpr float kScale = 0.5f;
    std::vector<__half> latent(static_cast<size_t>(64) * (kKv + kRope), H(0.0f));
    std::vector<int32_t> pos = {0, 2, 5};
    for (int t : pos)
        for (int d = 0; d < kKv + kRope; ++d)
            latent[static_cast<size_t>(t) * (kKv + kRope) + d] =
                H(0.1f * static_cast<float>((d + t) % 5));
    std::vector<float> qk(kKv);
    for (int d = 0; d < kKv; ++d) qk[static_cast<size_t>(d)] = 0.2f * static_cast<float>(d % 3) - 0.1f;
    std::vector<__half> qrope(kRope);
    for (int d = 0; d < kRope; ++d) qrope[static_cast<size_t>(d)] = H(0.3f * static_cast<float>(d));
    std::vector<float> sink = {0.75f};

    __half *d_latent = DevAlloc<__half>(latent.size());
    float *d_qk = DevAlloc<float>(kKv);
    __half *d_qrope = DevAlloc<__half>(kRope);
    int32_t *d_pos = DevAlloc<int32_t>(kNpos);
    float *d_sink = DevAlloc<float>(kHeads);
    float *d_scores = DevAlloc<float>(static_cast<size_t>(kHeads) * kNpos);
    float *d_acc = DevAlloc<float>(static_cast<size_t>(kHeads) * kKv);
    Upload(d_latent, latent.data(), latent.size() * 2);
    Upload(d_qk, qk.data(), kKv * 4);
    Upload(d_qrope, qrope.data(), kRope * 2);
    Upload(d_pos, pos.data(), kNpos * 4);
    Upload(d_sink, sink.data(), kHeads * 4);
    REQUIRE(gpu::RunMlaSinkSoftmax(d_qk, d_qrope, kRope, 0, d_latent, kKv + kRope,
                                  d_pos, kNpos, kHeads, kKv, kRope, d_sink,
                                  kScale, d_scores, d_acc, nullptr) == cudaSuccess);
    std::vector<float> scores(kNpos), acc(kKv);
    Download(scores.data(), d_scores, kNpos * 4);
    Download(acc.data(), d_acc, kKv * 4);

    // Host reference.
    float s[kNpos];
    for (int t = 0; t < kNpos; ++t) {
        float nope = 0.0f, rope = 0.0f;
        for (int d = 0; d < kKv; ++d)
            nope += qk[static_cast<size_t>(d)] *
                    __half2float(latent[static_cast<size_t>(pos[static_cast<size_t>(t)]) * (kKv + kRope) + d]);
        for (int d = 0; d < kRope; ++d)
            rope += __half2float(qrope[static_cast<size_t>(d)]) *
                    __half2float(latent[static_cast<size_t>(pos[static_cast<size_t>(t)]) * (kKv + kRope) + kKv + d]);
        s[t] = (nope + rope) * kScale;
    }
    float m = sink[0];
    for (float v : s) m = std::max(m, v);
    float den = std::exp(sink[0] - m);
    for (float v : s) den += std::exp(v - m);
    float probs[kNpos];
    float sum = 0.0f;
    for (int t = 0; t < kNpos; ++t) {
        probs[t] = std::exp(s[t] - m) / den;
        sum += probs[t];
    }
    // Mass is missing exactly because of the sink.
    CHECK(sum < 1.0f);
    CHECK(sum == Approx(1.0f - std::exp(sink[0] - m) / den).margin(1e-5));
    for (int d = 0; d < kKv; ++d) {
        float want = 0.0f;
        for (int t = 0; t < kNpos; ++t)
            want += probs[t] *
                    __half2float(latent[static_cast<size_t>(pos[static_cast<size_t>(t)]) * (kKv + kRope) + d]);
        CHECK(acc[static_cast<size_t>(d)] == Approx(want).margin(1e-5));
    }

    DevFree(d_latent);
    DevFree(d_qk);
    DevFree(d_qrope);
    DevFree(d_pos);
    DevFree(d_sink);
    DevFree(d_scores);
    DevFree(d_acc);
}
