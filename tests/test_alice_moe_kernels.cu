// AliceAI kernel parity tests: device kernels vs a host reference.
#include <catch2/catch_test_macros.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <lsxcommon/alice_moe_kernels.h>

#include <cmath>
#include <vector>

namespace {

__half H(float v) { return __float2half(v); }
float F(__half h) { return __half2float(h); }

std::vector<__half> Pattern(int64_t n, float scale, float offset)
{
    std::vector<__half> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] = H(scale * std::sin(0.1f * i) + offset);
    return v;
}

cudaError_t ToDev(void **d, const std::vector<__half> &h)
{
    cudaError_t e = cudaMalloc(d, h.size() * sizeof(__half));
    if (e != cudaSuccess) return e;
    return cudaMemcpy(*d, h.data(), h.size() * sizeof(__half),
                      cudaMemcpyHostToDevice);
}

} // namespace

TEST_CASE("alice zc-rmsnorm kernel matches the host reference", "[alice][kernels]")
{
    constexpr int rows = 4;
    constexpr int hidden = 64;
    auto xh = Pattern(rows * hidden, 1.0f, 0.05f);
    auto wh = Pattern(hidden, 0.5f, 0.1f);
    void *dx = nullptr, *dw = nullptr, *dout = nullptr;
    REQUIRE(ToDev(&dx, xh) == cudaSuccess);
    REQUIRE(ToDev(&dw, wh) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, xh.size() * sizeof(__half)) == cudaSuccess);

    REQUIRE(lsxcommon::alice::RunAliceZcRmsNorm(
                nullptr, reinterpret_cast<const __half *>(dx),
                reinterpret_cast<const __half *>(dw),
                reinterpret_cast<__half *>(dout), rows, hidden, 1e-6f, true) ==
            cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> got(xh.size());
    REQUIRE(cudaMemcpy(got.data(), dout, got.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int r = 0; r < rows; ++r) {
        float ms = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            float v = F(xh[r * hidden + i]);
            ms += v * v;
        }
        ms /= hidden;
        float rstd = 1.0f / std::sqrt(ms + 1e-6f);
        for (int i = 0; i < hidden; ++i) {
            float want = F(xh[r * hidden + i]) * rstd * (1.0f + F(wh[i]));
            REQUIRE(std::fabs(want - F(got[r * hidden + i])) < 2e-3f);
        }
    }
    cudaFree(dx); cudaFree(dw); cudaFree(dout);
}

TEST_CASE("alice depth-mix kernel matches the host reference", "[alice][kernels]")
{
    constexpr int S = 5;
    constexpr int hidden = 64;
    auto sh = Pattern(S * hidden, 1.0f, 0.1f);
    auto nh = Pattern(hidden, 0.3f, 0.0f);
    auto ph = Pattern(hidden, 0.2f, 0.0f);
    void *ds = nullptr, *dn = nullptr, *dp = nullptr, *dout = nullptr;
    REQUIRE(ToDev(&ds, sh) == cudaSuccess);
    REQUIRE(ToDev(&dn, nh) == cudaSuccess);
    REQUIRE(ToDev(&dp, ph) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, hidden * sizeof(__half)) == cudaSuccess);

    REQUIRE(lsxcommon::alice::RunAliceDepthMix(
                nullptr, reinterpret_cast<const __half *>(ds), S,
                reinterpret_cast<const __half *>(dn),
                reinterpret_cast<const __half *>(dp),
                reinterpret_cast<__half *>(dout), hidden, 1e-6f) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> got(hidden);
    REQUIRE(cudaMemcpy(got.data(), dout, got.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    std::vector<float> logits(S, 0.0f);
    for (int s = 0; s < S; ++s) {
        float ms = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            float v = F(sh[s * hidden + i]);
            ms += v * v;
        }
        ms /= hidden;
        float rstd = 1.0f / std::sqrt(ms + 1e-6f);
        float dot = 0.0f;
        for (int i = 0; i < hidden; ++i)
            dot += F(sh[s * hidden + i]) * rstd * F(nh[i]) * F(ph[i]);
        logits[s] = dot;
    }
    float mx = logits[0];
    for (int s = 1; s < S; ++s) mx = std::max(mx, logits[s]);
    float sum = 0.0f;
    for (int s = 0; s < S; ++s) { logits[s] = std::exp(logits[s] - mx); sum += logits[s]; }
    for (int i = 0; i < hidden; ++i) {
        float want = 0.0f;
        for (int s = 0; s < S; ++s)
            want += (logits[s] / sum) * F(sh[s * hidden + i]);
        REQUIRE(std::fabs(want - F(got[i])) < 3e-3f);
    }
    cudaFree(ds); cudaFree(dn); cudaFree(dp); cudaFree(dout);
}

TEST_CASE("alice router top-k kernel selects the reference experts", "[alice][kernels]")
{
    constexpr int hidden = 64;
    constexpr int E = 16;
    constexpr int K = 4;
    auto xh = Pattern(hidden, 1.0f, 0.1f);
    auto gh = Pattern(E * hidden, 0.5f, 0.0f);
    auto bh = Pattern(E, 0.1f, 0.0f);
    void *dx = nullptr, *dg = nullptr, *db = nullptr;
    int32_t *didx = nullptr; float *dw = nullptr;
    REQUIRE(ToDev(&dx, xh) == cudaSuccess);
    REQUIRE(ToDev(&dg, gh) == cudaSuccess);
    REQUIRE(ToDev(&db, bh) == cudaSuccess);
    REQUIRE(cudaMalloc(&didx, K * sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dw, K * sizeof(float)) == cudaSuccess);

    REQUIRE(lsxcommon::alice::RunAliceRouterTopK(
                nullptr, reinterpret_cast<const __half *>(dx),
                reinterpret_cast<const __half *>(dg),
                reinterpret_cast<const __half *>(db), hidden, E, K, didx, dw,
                nullptr) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<int32_t> idx(K);
    std::vector<float> w(K);
    REQUIRE(cudaMemcpy(idx.data(), didx, K * sizeof(int32_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(w.data(), dw, K * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    std::vector<float> scores(E);
    for (int e = 0; e < E; ++e) {
        float dot = 0.0f;
        for (int i = 0; i < hidden; ++i)
            dot += F(xh[i]) * F(gh[e * hidden + i]);
        scores[e] = 1.0f / (1.0f + std::exp(-dot)) + F(bh[e]);
    }
    for (int k = 0; k < K; ++k) {
        int best = -1; float bv = -1e30f;
        for (int e = 0; e < E; ++e) {
            bool used = false;
            for (int j = 0; j < k; ++j) if (idx[j] == e) { used = true; break; }
            if (!used && scores[e] > bv) { bv = scores[e]; best = e; }
        }
        REQUIRE(idx[k] == best);
    }
    cudaFree(dx); cudaFree(dg); cudaFree(db); cudaFree(didx); cudaFree(dw);
}

TEST_CASE("alice swiglu kernel matches silu(a)*b", "[alice][kernels]")
{
    constexpr int n = 128;
    auto ah = Pattern(n, 1.0f, 0.1f);
    auto bh = Pattern(n, 0.8f, -0.2f);
    void *da = nullptr, *db = nullptr, *dout = nullptr;
    REQUIRE(ToDev(&da, ah) == cudaSuccess);
    REQUIRE(ToDev(&db, bh) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, n * sizeof(__half)) == cudaSuccess);
    REQUIRE(lsxcommon::alice::RunAliceSwiGLU(
                nullptr, reinterpret_cast<const __half *>(da),
                reinterpret_cast<const __half *>(db),
                reinterpret_cast<__half *>(dout), n) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<__half> got(n);
    REQUIRE(cudaMemcpy(got.data(), dout, n * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < n; ++i) {
        float a = F(ah[i]), b = F(bh[i]);
        float want = (a / (1.0f + std::exp(-a))) * b;
        REQUIRE(std::fabs(want - F(got[i])) < 2e-3f);
    }
    cudaFree(da); cudaFree(db); cudaFree(dout);
}
