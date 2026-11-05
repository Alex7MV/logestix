// Kimi-K3.1-Pro Lightning Indexer (Gated DSA) kernel parity.
//
// Every expectation is computed on the host from the arithmetic the frozen
// passport records (kimi_moe_pro_constants.h § Lightning Indexer) — the kernels
// never produce their own golden:
//   k[s,:]     = LayerNorm(wk @ h_s)                        weight + bias, eps
//   q[h,:]     = (wq_b @ qr)[h*index_dim : ...] * 128^-0.5
//   w[h]       = (weights_proj @ h_t)[h] * 32^-0.5
//   score[t,s] = sum_h w[h] * relu(q[h] . k[s])             s <= t (causal)
//   top-k      = the index_topk largest, ties by ascending position
//
// The identity direction is the load-bearing one: `sel == nullptr` and
// `sel = {0..n-1}` must produce bit-identical attention output, because that is
// the whole <= index_topk protocol the engine runs on real weights.
#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "lsxcommon/kimi_moe_k3_kernels.h"
#include "nvfp4_codec.h"

using namespace lsxcommon::kimi_k3;

namespace {

__half H(float v) { return __float2half(v); }
float F(__half h) { return __half2float(h); }

// Deterministic pseudo-random value in [-1, 1) — a fixed 24-bit mantissa
// derived from the index, so every run and every machine sees the same input.
float Rand(int i)
{
    const unsigned int u = static_cast<unsigned int>(i) * 2654435761u;
    return static_cast<float>((u >> 8) & 0xFFFFFFu) / 8388608.0f - 1.0f;
}

// E2M1 / E4M3 block-scale decode come from the single-source codec: this test
// has no independent claim on the value table (its scale is a constant 1.0),
// and a local copy is exactly what tests/test_nvfp4_scale.cpp forbids.

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

// Host key reference: raw = wk @ x (double), LayerNorm(weight + bias, eps).
std::vector<float> HostKey(const std::vector<__half> &x,
                           const std::vector<__half> &wk,
                           const std::vector<__half> &knorm_w,
                           const std::vector<__half> &knorm_b, int kv_len,
                           int index_dim, float eps)
{
    std::vector<double> raw(static_cast<size_t>(index_dim), 0.0);
    for (int d = 0; d < index_dim; ++d)
        for (int i = 0; i < kv_len; ++i)
            raw[static_cast<size_t>(d)] +=
                static_cast<double>(F(wk[static_cast<size_t>(d) * kv_len + i])) *
                static_cast<double>(F(x[static_cast<size_t>(i)]));
    double mean = 0.0;
    for (double v : raw) mean += v;
    mean /= static_cast<double>(index_dim);
    double var = 0.0;
    for (double v : raw) var += (v - mean) * (v - mean);
    var /= static_cast<double>(index_dim);
    const double rstd = 1.0 / std::sqrt(var + static_cast<double>(eps));
    std::vector<float> out(static_cast<size_t>(index_dim));
    for (int d = 0; d < index_dim; ++d)
        out[static_cast<size_t>(d)] = static_cast<float>(
            (raw[static_cast<size_t>(d)] - mean) * rstd *
                static_cast<double>(F(knorm_w[static_cast<size_t>(d)])) +
            static_cast<double>(F(knorm_b[static_cast<size_t>(d)])));
    return out;
}

}  // namespace

TEST_CASE("kimi-k3.1-pro indexer: key LayerNorm matches the host reference",
          "[kimi-pro][indexer]")
{
    const int kv_len = 128, index_dim = 128, n_pos = 4;
    std::vector<__half> wk(static_cast<size_t>(index_dim) * kv_len),
        knorm_w(static_cast<size_t>(index_dim)),
        knorm_b(static_cast<size_t>(index_dim));
    for (int d = 0; d < index_dim; ++d) {
        for (int i = 0; i < kv_len; ++i)
            wk[static_cast<size_t>(d) * kv_len + i] =
                H(Rand(d * kv_len + i + 1) * 0.25f);
        knorm_w[static_cast<size_t>(d)] = H(0.5f + 0.01f * (d % 7));
        knorm_b[static_cast<size_t>(d)] = H(0.05f * ((d % 5) - 2));
    }

    __half *d_wk = nullptr, *d_nw = nullptr, *d_nb = nullptr, *d_x = nullptr,
        *d_k = nullptr;
    cudaMalloc(&d_wk, wk.size() * sizeof(__half));
    cudaMalloc(&d_nw, knorm_w.size() * sizeof(__half));
    cudaMalloc(&d_nb, knorm_b.size() * sizeof(__half));
    cudaMalloc(&d_x, static_cast<size_t>(kv_len) * sizeof(__half));
    cudaMalloc(&d_k, static_cast<size_t>(index_dim) * sizeof(__half));
    cudaMemcpy(d_wk, wk.data(), wk.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_nw, knorm_w.data(), knorm_w.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_nb, knorm_b.data(), knorm_b.size() * sizeof(__half),
               cudaMemcpyHostToDevice);

    const float eps = 1e-5f;
    for (int p = 0; p < n_pos; ++p) {
        std::vector<__half> x(static_cast<size_t>(kv_len)),
            k_out(static_cast<size_t>(index_dim));
        for (int i = 0; i < kv_len; ++i)
            x[static_cast<size_t>(i)] = H(Rand(1000 * p + i + 7) * 0.5f);
        cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half),
                   cudaMemcpyHostToDevice);
        cudaMemset(d_k, 0, k_out.size() * sizeof(__half));
        REQUIRE(RunKimiK3IndexerKey(0, d_x, d_wk, d_nw, d_nb, d_k, kv_len,
                                    index_dim, eps) == cudaSuccess);
        cudaMemcpy(k_out.data(), d_k, k_out.size() * sizeof(__half),
                   cudaMemcpyDeviceToHost);
        const std::vector<float> want =
            HostKey(x, wk, knorm_w, knorm_b, kv_len, index_dim, eps);
        float maxdev = 0.0f;
        for (int d = 0; d < index_dim; ++d)
            maxdev = std::max(
                maxdev, std::fabs(F(k_out[static_cast<size_t>(d)]) -
                                  want[static_cast<size_t>(d)]));
        INFO("pos=" << p << " maxdev=" << maxdev);
        REQUIRE(maxdev <= 1e-2f);
    }
    cudaFree(d_wk);
    cudaFree(d_nw);
    cudaFree(d_nb);
    cudaFree(d_x);
    cudaFree(d_k);
}

TEST_CASE("kimi-k3.1-pro indexer: query and head weights match the host reference",
          "[kimi-pro][indexer]")
{
    const int index_heads = 4, index_dim = 8, q_lora = 16, kv_len = 32;
    const int q_width = index_heads * index_dim;
    std::vector<__half> qa(static_cast<size_t>(q_lora)),
        x(static_cast<size_t>(kv_len)),
        wq_b(static_cast<size_t>(q_width) * q_lora),
        wproj(static_cast<size_t>(index_heads) * kv_len);
    for (int j = 0; j < q_lora; ++j) qa[static_cast<size_t>(j)] = H(Rand(j + 3));
    for (int i = 0; i < kv_len; ++i) x[static_cast<size_t>(i)] = H(Rand(i + 11));
    for (size_t r = 0; r < wq_b.size(); ++r)
        wq_b[r] = H(Rand(static_cast<int>(r) + 17) * 0.5f);
    for (size_t r = 0; r < wproj.size(); ++r)
        wproj[r] = H(Rand(static_cast<int>(r) + 29) * 0.5f);

    // The two derived scales of the Lightning Indexer, from the passport.
    const float q_scale = 0.08838834764831845f;   // 128^-0.5
    const float w_scale = 0.17677669529663687f;   // 32^-0.5

    __half *d_qa = nullptr, *d_x = nullptr, *d_wqb = nullptr, *d_wproj = nullptr,
        *d_q = nullptr, *d_w = nullptr;
    cudaMalloc(&d_qa, qa.size() * sizeof(__half));
    cudaMalloc(&d_x, x.size() * sizeof(__half));
    cudaMalloc(&d_wqb, wq_b.size() * sizeof(__half));
    cudaMalloc(&d_wproj, wproj.size() * sizeof(__half));
    cudaMalloc(&d_q, static_cast<size_t>(q_width) * sizeof(__half));
    cudaMalloc(&d_w, static_cast<size_t>(index_heads) * sizeof(__half));
    cudaMemcpy(d_qa, qa.data(), qa.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wqb, wq_b.data(), wq_b.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_wproj, wproj.data(), wproj.size() * sizeof(__half),
               cudaMemcpyHostToDevice);

    REQUIRE(RunKimiK3IndexerQuery(0, d_qa, d_x, d_wqb, d_wproj, d_q, d_w, q_lora,
                                  kv_len, index_heads, index_dim, q_scale,
                                  w_scale) == cudaSuccess);
    std::vector<__half> q_out(static_cast<size_t>(q_width)),
        w_out(static_cast<size_t>(index_heads));
    cudaMemcpy(q_out.data(), d_q, q_out.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(w_out.data(), d_w, w_out.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);

    for (int h = 0; h < index_heads; ++h) {
        for (int d = 0; d < index_dim; ++d) {
            const size_t r = static_cast<size_t>(h * index_dim + d);
            double acc = 0.0;
            for (int j = 0; j < q_lora; ++j)
                acc += static_cast<double>(F(wq_b[r * q_lora + j])) *
                       static_cast<double>(F(qa[static_cast<size_t>(j)]));
            const double want = acc * static_cast<double>(q_scale);
            const double got = static_cast<double>(F(q_out[r]));
            INFO("q h=" << h << " d=" << d << " got=" << got
                        << " want=" << want);
            REQUIRE(std::fabs(got - want) <= 2e-2 * (1.0 + std::fabs(want)));
        }
        double acc = 0.0;
        for (int i = 0; i < kv_len; ++i)
            acc += static_cast<double>(
                       F(wproj[static_cast<size_t>(h) * kv_len + i])) *
                   static_cast<double>(F(x[static_cast<size_t>(i)]));
        const double want = acc * static_cast<double>(w_scale);
        const double got = static_cast<double>(F(w_out[static_cast<size_t>(h)]));
        INFO("w h=" << h << " got=" << got << " want=" << want);
        REQUIRE(std::fabs(got - want) <= 2e-2 * (1.0 + std::fabs(want)));
    }
    cudaFree(d_qa);
    cudaFree(d_x);
    cudaFree(d_wqb);
    cudaFree(d_wproj);
    cudaFree(d_q);
    cudaFree(d_w);
}

TEST_CASE("kimi-k3.1-pro indexer: scores are the head-weighted relu sum",
          "[kimi-pro][indexer]")
{
    const int n_pos = 6, index_heads = 3, index_dim = 8;
    std::vector<__half> q(static_cast<size_t>(index_heads) * index_dim),
        w(static_cast<size_t>(index_heads)),
        k_cache(static_cast<size_t>(n_pos) * index_dim);
    // Every key component is negative, so head 0 (all-positive query) has a
    // negative dot product everywhere and its relu must contribute exactly 0,
    // while head 1 (all-negative query) is strictly positive — the score is a
    // real weighted sum, not a clamped-to-zero constant.
    for (int s = 0; s < n_pos; ++s)
        for (int d = 0; d < index_dim; ++d)
            k_cache[static_cast<size_t>(s) * index_dim + d] =
                H(-0.3f - 0.1f * ((s * index_dim + d) % 4));
    for (int d = 0; d < index_dim; ++d) {
        q[static_cast<size_t>(d)] = H(0.5f);
        q[static_cast<size_t>(index_dim) + d] = H(-1.0f);
        q[2 * static_cast<size_t>(index_dim) + d] = H(Rand(2 * index_dim + d + 5));
    }
    w[0] = H(-3.0f);
    w[1] = H(0.7f);
    w[2] = H(1.3f);

    __half *d_q = nullptr, *d_w = nullptr, *d_k = nullptr, *d_w0 = nullptr;
    float *d_score = nullptr, *d_score_zero = nullptr;
    cudaMalloc(&d_q, q.size() * sizeof(__half));
    cudaMalloc(&d_w, w.size() * sizeof(__half));
    cudaMalloc(&d_w0, w.size() * sizeof(__half));
    cudaMalloc(&d_k, k_cache.size() * sizeof(__half));
    cudaMalloc(&d_score, static_cast<size_t>(n_pos) * sizeof(float));
    cudaMalloc(&d_score_zero, static_cast<size_t>(n_pos) * sizeof(float));
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k_cache.data(), k_cache.size() * sizeof(__half),
               cudaMemcpyHostToDevice);

    REQUIRE(RunKimiK3IndexerScores(0, d_q, d_w, d_k, n_pos, index_heads,
                                   index_dim, d_score) == cudaSuccess);

    std::vector<float> score(static_cast<size_t>(n_pos)),
        score_zero(static_cast<size_t>(n_pos));
    cudaMemcpy(score.data(), d_score, score.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int s = 0; s < n_pos; ++s) {
        double acc = 0.0;
        for (int h = 0; h < index_heads; ++h) {
            double dot = 0.0;
            for (int d = 0; d < index_dim; ++d)
                dot += static_cast<double>(
                           F(q[static_cast<size_t>(h) * index_dim + d])) *
                       static_cast<double>(
                           F(k_cache[static_cast<size_t>(s) * index_dim + d]));
            acc += static_cast<double>(F(w[static_cast<size_t>(h)])) *
                   std::max(0.0, dot);
        }
        INFO("s=" << s << " got=" << score[static_cast<size_t>(s)]
                  << " want=" << acc);
        REQUIRE(std::fabs(static_cast<double>(score[static_cast<size_t>(s)]) -
                          acc) <= 1e-4 * (1.0 + std::fabs(acc)));
    }

    // The clamped head contributes exactly zero, not a tiny negative: zeroing
    // its weight must leave every score bit-identical.
    std::vector<__half> w0 = w;
    w0[0] = H(0.0f);
    cudaMemcpy(d_w0, w0.data(), w0.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    REQUIRE(RunKimiK3IndexerScores(0, d_q, d_w0, d_k, n_pos, index_heads,
                                   index_dim, d_score_zero) == cudaSuccess);
    cudaMemcpy(score_zero.data(), d_score_zero, score_zero.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    REQUIRE(std::memcmp(score.data(), score_zero.data(),
                        score.size() * sizeof(float)) == 0);

    cudaFree(d_q);
    cudaFree(d_w);
    cudaFree(d_w0);
    cudaFree(d_k);
    cudaFree(d_score);
    cudaFree(d_score_zero);
}

TEST_CASE("kimi-k3.1-pro indexer: top-k is exact and deterministic",
          "[kimi-pro][indexer]")
{
    // Integer-valued scores in [0, 1000): with 3000 positions the k-th largest
    // (2048th) sits in the middle of a large tie group, so the boundary rule
    // (ascending position) is exercised, not just the strict-greater part.
    const int n_pos = 3000, k = 2048;
    std::vector<float> score(static_cast<size_t>(n_pos));
    for (int i = 0; i < n_pos; ++i)
        score[static_cast<size_t>(i)] =
            static_cast<float>((static_cast<unsigned int>(i) * 2654435761u) %
                               1000u);

    float *d_score = nullptr;
    int32_t *d_idx = nullptr;
    int *d_n = nullptr;
    cudaMalloc(&d_score, score.size() * sizeof(float));
    cudaMalloc(&d_idx, static_cast<size_t>(k) * sizeof(int32_t));
    cudaMalloc(&d_n, sizeof(int));
    cudaMemcpy(d_score, score.data(), score.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    std::vector<int> want(static_cast<size_t>(n_pos));
    std::iota(want.begin(), want.end(), 0);
    std::partial_sort(want.begin(), want.begin() + k, want.end(),
                      [&](int a, int b) {
                          const float av = score[static_cast<size_t>(a)];
                          const float bv = score[static_cast<size_t>(b)];
                          if (av != bv) return av > bv;
                          return a < b;
                      });
    // Emission contract: every selected position strictly above the boundary
    // score in ascending order, then the boundary tie group from the lowest
    // position up. Same set as `partial_sort`, in the order the masked
    // attention consumes.
    const float thr_val = score[static_cast<size_t>(want[k - 1])];
    std::vector<int32_t> expect;
    for (int i = 0; i < n_pos; ++i)
        if (score[static_cast<size_t>(i)] > thr_val)
            expect.push_back(i);
    for (int i = 0; i < n_pos && static_cast<int>(expect.size()) < k; ++i)
        if (score[static_cast<size_t>(i)] == thr_val) expect.push_back(i);
    REQUIRE(static_cast<int>(expect.size()) == k);

    for (int run = 0; run < 2; ++run) {
        std::vector<int32_t> idx(static_cast<size_t>(k));
        int n = -1;
        cudaMemset(d_idx, 0xFF, static_cast<size_t>(k) * sizeof(int32_t));
        cudaMemset(d_n, 0xFF, sizeof(int));
        REQUIRE(RunKimiK3IndexerTopK(0, d_score, n_pos, k, d_idx, d_n) ==
                cudaSuccess);
        cudaMemcpy(idx.data(), d_idx, idx.size() * sizeof(int32_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost);
        REQUIRE(n == k);
        int mismatches = 0;
        for (int i = 0; i < k; ++i)
            if (idx[static_cast<size_t>(i)] != expect[static_cast<size_t>(i)]) {
                if (mismatches < 4)
                    INFO("run=" << run << " i=" << i << " got="
                                << idx[static_cast<size_t>(i)] << " want="
                                << expect[static_cast<size_t>(i)]);
                ++mismatches;
            }
        INFO("run=" << run << " mismatches=" << mismatches);
        REQUIRE(mismatches == 0);
        // ... and the same *set* as the reference partial_sort.
        std::vector<int32_t> got(idx);
        std::sort(got.begin(), got.end());
        std::vector<int> want_sorted(want.begin(), want.begin() + k);
        std::sort(want_sorted.begin(), want_sorted.end());
        for (int i = 0; i < k; ++i)
            REQUIRE(got[static_cast<size_t>(i)] ==
                    want_sorted[static_cast<size_t>(i)]);
    }
    cudaFree(d_score);
    cudaFree(d_idx);
    cudaFree(d_n);
}

TEST_CASE("kimi-k3.1-pro indexer: identity selection is bit-identical to dense attention",
          "[kimi-pro][indexer]")
{
    // The canonical <= index_topk protocol: no selection at all.
    const int heads = 2, L = 8, rope = 2, V = 4, n_pos = 37;
    const int rows_per_head = V;
    const int row_offset = 0;
    const int row_packed = L / 2;
    const int row_scales = 1;

    std::vector<float> wv(static_cast<size_t>(heads) * rows_per_head * L, 0.0f);
    for (int h = 0; h < heads; ++h)
        for (int n = 0; n < rows_per_head; ++n)
            for (int c = 0; c < L; ++c)
                wv[static_cast<size_t>((h * rows_per_head + n) * L + c)] =
                    static_cast<float>((h + n + c) % 5);
    const std::vector<uint8_t> packed = PackNibbles(wv);
    const std::vector<uint8_t> scales(
        static_cast<size_t>(heads) * rows_per_head * row_scales, 0x38);

    std::vector<__half> q_abs(static_cast<size_t>(heads) * L),
        q_rot(static_cast<size_t>(heads) * rope),
        latent(static_cast<size_t>(n_pos) * (L + rope));
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < L; ++i)
            q_abs[static_cast<size_t>(h) * L + i] = H(0.1f * (i + 1 + h));
        for (int i = 0; i < rope; ++i)
            q_rot[static_cast<size_t>(h) * rope + i] = H(0.2f * (i + 1));
    }
    for (int t = 0; t < n_pos; ++t)
        for (int c = 0; c < L + rope; ++c)
            latent[static_cast<size_t>(t) * (L + rope) + c] =
                H(0.05f * ((t + 1) * (c + 2) % 13) - 0.2f);

    uint8_t *d_packed = nullptr, *d_scales = nullptr;
    __half *d_qa = nullptr, *d_qr = nullptr, *d_lat = nullptr,
        *d_dense = nullptr, *d_ident = nullptr;
    int32_t *d_sel = nullptr;
    cudaMalloc(&d_packed, packed.size());
    cudaMalloc(&d_scales, scales.size());
    cudaMalloc(&d_qa, q_abs.size() * sizeof(__half));
    cudaMalloc(&d_qr, q_rot.size() * sizeof(__half));
    cudaMalloc(&d_lat, latent.size() * sizeof(__half));
    cudaMalloc(&d_dense, static_cast<size_t>(heads) * V * sizeof(__half));
    cudaMalloc(&d_ident, static_cast<size_t>(heads) * V * sizeof(__half));
    cudaMalloc(&d_sel, static_cast<size_t>(n_pos) * sizeof(int32_t));
    cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qa, q_abs.data(), q_abs.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_qr, q_rot.data(), q_rot.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_lat, latent.data(), latent.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    std::vector<int32_t> sel(static_cast<size_t>(n_pos));
    std::iota(sel.begin(), sel.end(), 0);
    cudaMemcpy(d_sel, sel.data(), sel.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    const float scale = 1.0f / std::sqrt(static_cast<float>(L + rope));
    REQUIRE(RunKimiK3MlaAttention(0, d_lat, n_pos, d_qa, d_qr, d_packed,
                                  d_scales, 1.0f, d_dense, heads, L, rope, V,
                                  rows_per_head, row_offset, rope, scale,
                                  row_packed, row_scales, nullptr, 0) ==
            cudaSuccess);
    REQUIRE(RunKimiK3MlaAttention(0, d_lat, n_pos, d_qa, d_qr, d_packed,
                                  d_scales, 1.0f, d_ident, heads, L, rope, V,
                                  rows_per_head, row_offset, rope, scale,
                                  row_packed, row_scales, d_sel, n_pos) ==
            cudaSuccess);

    std::vector<__half> dense(static_cast<size_t>(heads) * V),
        ident(static_cast<size_t>(heads) * V);
    cudaMemcpy(dense.data(), d_dense, dense.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(ident.data(), d_ident, ident.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < heads * V; ++i) {
        INFO("i=" << i << " dense=" << F(dense[static_cast<size_t>(i)])
                  << " ident=" << F(ident[static_cast<size_t>(i)]));
        REQUIRE(__half_as_ushort(dense[static_cast<size_t>(i)]) ==
                __half_as_ushort(ident[static_cast<size_t>(i)]));
    }
    REQUIRE(std::memcmp(dense.data(), ident.data(),
                        dense.size() * sizeof(__half)) == 0);

    cudaFree(d_packed);
    cudaFree(d_scales);
    cudaFree(d_qa);
    cudaFree(d_qr);
    cudaFree(d_lat);
    cudaFree(d_dense);
    cudaFree(d_ident);
    cudaFree(d_sel);
}

TEST_CASE("kimi-k3.1-pro indexer: masked attention matches the host reference",
          "[kimi-pro][indexer]")
{
    const int heads = 2, L = 8, rope = 2, V = 4, n_pos = 37;
    const int rows_per_head = V;
    const int row_offset = 0;
    const int row_packed = L / 2;
    const int row_scales = 1;
    const std::vector<int32_t> sel = {0, 2, 3, 5, 8, 13, 21, 34};
    const int n_sel = static_cast<int>(sel.size());

    std::vector<float> wv(static_cast<size_t>(heads) * rows_per_head * L, 0.0f);
    for (int h = 0; h < heads; ++h)
        for (int n = 0; n < rows_per_head; ++n)
            for (int c = 0; c < L; ++c)
                wv[static_cast<size_t>((h * rows_per_head + n) * L + c)] =
                    static_cast<float>((h + n + c) % 5);
    const std::vector<uint8_t> packed = PackNibbles(wv);
    const std::vector<uint8_t> scales(
        static_cast<size_t>(heads) * rows_per_head * row_scales, 0x38);

    std::vector<__half> q_abs(static_cast<size_t>(heads) * L),
        q_rot(static_cast<size_t>(heads) * rope),
        latent(static_cast<size_t>(n_pos) * (L + rope));
    for (int h = 0; h < heads; ++h) {
        for (int i = 0; i < L; ++i)
            q_abs[static_cast<size_t>(h) * L + i] = H(0.1f * (i + 1 + h));
        for (int i = 0; i < rope; ++i)
            q_rot[static_cast<size_t>(h) * rope + i] = H(0.2f * (i + 1));
    }
    for (int t = 0; t < n_pos; ++t)
        for (int c = 0; c < L + rope; ++c)
            latent[static_cast<size_t>(t) * (L + rope) + c] =
                H(0.05f * ((t + 1) * (c + 2) % 13) - 0.2f);

    uint8_t *d_packed = nullptr, *d_scales = nullptr;
    __half *d_qa = nullptr, *d_qr = nullptr, *d_lat = nullptr,
        *d_out = nullptr;
    int32_t *d_sel = nullptr;
    cudaMalloc(&d_packed, packed.size());
    cudaMalloc(&d_scales, scales.size());
    cudaMalloc(&d_qa, q_abs.size() * sizeof(__half));
    cudaMalloc(&d_qr, q_rot.size() * sizeof(__half));
    cudaMalloc(&d_lat, latent.size() * sizeof(__half));
    cudaMalloc(&d_out, static_cast<size_t>(heads) * V * sizeof(__half));
    cudaMalloc(&d_sel, static_cast<size_t>(n_sel) * sizeof(int32_t));
    cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qa, q_abs.data(), q_abs.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_qr, q_rot.data(), q_rot.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_lat, latent.data(), latent.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sel, sel.data(), sel.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    const float scale = 1.0f / std::sqrt(static_cast<float>(L + rope));
    REQUIRE(RunKimiK3MlaAttention(0, d_lat, n_pos, d_qa, d_qr, d_packed,
                                  d_scales, 1.0f, d_out, heads, L, rope, V,
                                  rows_per_head, row_offset, rope, scale,
                                  row_packed, row_scales, d_sel, n_sel) ==
            cudaSuccess);
    std::vector<__half> out(static_cast<size_t>(heads) * V);
    cudaMemcpy(out.data(), d_out, out.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);

    // Host reference over the same subset: softmax over the selected positions
    // only, then the absorbed value projection (E2M1 x E4M3 x ctx, FP32).
    for (int h = 0; h < heads; ++h) {
        std::vector<float> score(static_cast<size_t>(n_sel));
        for (int i = 0; i < n_sel; ++i) {
            const int t = sel[static_cast<size_t>(i)];
            float acc = 0.0f;
            for (int c = 0; c < L; ++c)
                acc += F(q_abs[static_cast<size_t>(h) * L + c]) *
                       F(latent[static_cast<size_t>(t) * (L + rope) + c]);
            for (int r = 0; r < rope; ++r)
                acc += F(q_rot[static_cast<size_t>(h) * rope + r]) *
                       F(latent[static_cast<size_t>(t) * (L + rope) + L + r]);
            score[static_cast<size_t>(i)] = acc * scale;
        }
        float mx = score[0];
        for (float v : score) mx = std::max(mx, v);
        float sum = 0.0f;
        for (auto &v : score) {
            v = std::exp(v - mx);
            sum += v;
        }
        std::vector<float> ctx(static_cast<size_t>(L), 0.0f);
        for (int i = 0; i < n_sel; ++i) {
            const int t = sel[static_cast<size_t>(i)];
            for (int c = 0; c < L; ++c)
                ctx[static_cast<size_t>(c)] +=
                    score[static_cast<size_t>(i)] / sum *
                    F(latent[static_cast<size_t>(t) * (L + rope) + c]);
        }
        for (int vd = 0; vd < V; ++vd) {
            float acc = 0.0f;
            for (int c = 0; c < L; ++c) {
                const size_t row =
                    static_cast<size_t>(h * rows_per_head + row_offset + vd);
                const int nib = static_cast<int>(
                    wv[row * L + c] == 0.0f ? 0.0f : wv[row * L + c]);
                acc += lsxcommon::gpu::nvfp4::E2M1Value(nib) *
                       lsxcommon::gpu::nvfp4::E4M3ScaleToFloat(0x38) *
                       ctx[static_cast<size_t>(c)];
            }
            INFO("h=" << h << " vd=" << vd);
            REQUIRE(std::fabs(F(out[static_cast<size_t>(h) * V + vd]) - acc) <=
                    5e-3f);
        }
    }
    cudaFree(d_packed);
    cudaFree(d_scales);
    cudaFree(d_qa);
    cudaFree(d_qr);
    cudaFree(d_lat);
    cudaFree(d_out);
    cudaFree(d_sel);
}
