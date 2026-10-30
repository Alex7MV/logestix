// Hy-Memory-30B-A3B numeric kernels (Model Ingestion Protocol stage 5/6):
// plain row/per-head RMSNorm, full rotary, the sigmoid+expert-bias router and
// the routed accumulation / FP32 combine, each against a host reference.
// Label `inference`; SKIPs (4) when no CUDA device is present.
#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "lsxcommon/hy_moe_kernels.h"

using namespace lsxcommon::hy;

namespace {

bool HaveCuda()
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
    return n > 0;
}

__half H(float v) { return __float2half(v); }
float F(__half h) { return __half2float(h); }

std::vector<__half> Pattern(int64_t n, float scale, int64_t seed)
{
    std::vector<__half> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        const int64_t k = (i * 7 + seed * 13) % 23;
        v[static_cast<size_t>(i)] = H(static_cast<float>(k - 11) * scale);
    }
    return v;
}

template <typename T>
T *ToDevice(const std::vector<T> &v)
{
    T *d = nullptr;
    REQUIRE(cudaMalloc(&d, sizeof(T) * v.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, v.data(), sizeof(T) * v.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
}

} // namespace

TEST_CASE("Hy plain RMSNorm matches the host reference", "[hy][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int rows = 3, hidden = 64;
    auto x = Pattern(rows * hidden, 0.05f, 1);
    auto w = Pattern(hidden, 0.01f, 2);
    for (auto &v : w) v = H(1.0f + F(v));
    __half *dx = ToDevice(x);
    __half *dw = ToDevice(w);
    __half *dy = nullptr;
    REQUIRE(cudaMalloc(&dy, sizeof(__half) * x.size()) == cudaSuccess);
    REQUIRE(RunHyNorm(nullptr, dx, dw, dy, rows, hidden, 1e-5f) == cudaSuccess);
    std::vector<__half> y(x.size());
    REQUIRE(cudaMemcpy(y.data(), dy, sizeof(__half) * y.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int r = 0; r < rows; ++r) {
        float ss = 0.0f;
        for (int j = 0; j < hidden; ++j)
            ss += F(x[static_cast<size_t>(r * hidden + j)]) *
                  F(x[static_cast<size_t>(r * hidden + j)]);
        const float inv = 1.0f / std::sqrt(ss / hidden + 1e-5f);
        for (int j = 0; j < hidden; ++j) {
            const float want =
                F(x[static_cast<size_t>(r * hidden + j)]) * inv *
                F(w[static_cast<size_t>(j)]);
            const float got = F(y[static_cast<size_t>(r * hidden + j)]);
            CHECK(std::abs(got - want) <= 6e-3f * std::max(1.0f, std::abs(want)));
        }
    }
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
}

TEST_CASE("Hy per-head RMSNorm normalizes each head separately", "[hy][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int heads = 2, hd = 32;
    auto x = Pattern(heads * hd, 0.1f, 3);
    std::vector<__half> w(static_cast<size_t>(hd), H(1.0f));
    __half *dx = ToDevice(x);
    __half *dw = ToDevice(w);
    __half *dy = nullptr;
    REQUIRE(cudaMalloc(&dy, sizeof(__half) * x.size()) == cudaSuccess);
    REQUIRE(RunHyHeadNorm(nullptr, dx, dw, dy, 1, heads, hd, 1e-5f) ==
            cudaSuccess);
    std::vector<__half> y(x.size());
    REQUIRE(cudaMemcpy(y.data(), dy, sizeof(__half) * y.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int h = 0; h < heads; ++h) {
        float ss = 0.0f;
        for (int j = 0; j < hd; ++j)
            ss += F(x[static_cast<size_t>(h * hd + j)]) *
                  F(x[static_cast<size_t>(h * hd + j)]);
        const float inv = 1.0f / std::sqrt(ss / hd + 1e-5f);
        for (int j = 0; j < hd; ++j) {
            const float want = F(x[static_cast<size_t>(h * hd + j)]) * inv;
            const float got = F(y[static_cast<size_t>(h * hd + j)]);
            CHECK(std::abs(got - want) <= 6e-3f * std::max(1.0f, std::abs(want)));
        }
    }
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
}

TEST_CASE("Hy full rotary matches the HF rotate_half reference", "[hy][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int heads = 2, hd = 8, pos = 5;
    const float theta = 11158840.0f;
    auto q = Pattern(heads * hd, 0.3f, 4);
    auto k = Pattern(heads * hd, 0.3f, 5);
    const auto q0 = q;
    const auto k0 = k;
    __half *dq = ToDevice(q);
    __half *dk = ToDevice(k);
    REQUIRE(RunHyRope(nullptr, dq, dk, 1, heads, heads, hd, theta, pos) ==
            cudaSuccess);
    std::vector<__half> qo(q.size()), ko(k.size());
    REQUIRE(cudaMemcpy(qo.data(), dq, sizeof(__half) * qo.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(ko.data(), dk, sizeof(__half) * ko.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int h = 0; h < heads; ++h) {
        for (int j = 0; j < hd / 2; ++j) {
            const float inv = 1.0f / std::pow(theta, 2.0f * j / hd);
            const float ang = static_cast<float>(pos) * inv;
            const float c = std::cos(ang), s = std::sin(ang);
            const float x1 = F(q0[static_cast<size_t>(h * hd + j)]);
            const float x2 = F(q0[static_cast<size_t>(h * hd + j + hd / 2)]);
            const float w1 = x1 * c - x2 * s;
            const float w2 = x1 * s + x2 * c;
            CHECK(std::abs(F(qo[static_cast<size_t>(h * hd + j)]) - w1) <=
                  2e-3f);
            CHECK(std::abs(F(qo[static_cast<size_t>(h * hd + j + hd / 2)]) - w2) <=
                  2e-3f);
            const float y1 = F(k0[static_cast<size_t>(h * hd + j)]);
            const float y2 = F(k0[static_cast<size_t>(h * hd + j + hd / 2)]);
            CHECK(std::abs(F(ko[static_cast<size_t>(h * hd + j)]) -
                           (y1 * c - y2 * s)) <= 2e-3f);
        }
    }
    cudaFree(dq);
    cudaFree(dk);
}

TEST_CASE("Hy router selects by sigmoid+bias and renormalizes the scores",
          "[hy][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int E = 6, K = 2, H = 32;
    const float routed_scale = 2.826f;
    auto x = Pattern(H, 0.2f, 6);
    std::vector<float> w(static_cast<size_t>(E) * H);
    for (int e = 0; e < E; ++e)
        for (int j = 0; j < H; ++j)
            w[static_cast<size_t>(e * H + j)] =
                0.05f * static_cast<float>((e + 1) * (j % 5) - 2);
    // A bias that lifts expert 4 above a higher raw score.
    std::vector<float> bias{0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f};
    __half *dx = ToDevice(x);
    float *dw = nullptr, *db = nullptr;
    REQUIRE(cudaMalloc(&dw, sizeof(float) * w.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&db, sizeof(float) * bias.size()) == cudaSuccess);
    REQUIRE(cudaMemcpy(dw, w.data(), sizeof(float) * w.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(db, bias.data(), sizeof(float) * bias.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    HyRoutedSlot *ds = nullptr;
    REQUIRE(cudaMalloc(&ds, sizeof(HyRoutedSlot) * K) == cudaSuccess);
    REQUIRE(RunHyRouter(nullptr, dw, db, dx, 1, E, K, H, routed_scale, ds) ==
            cudaSuccess);
    HyRoutedSlot slots[2];
    REQUIRE(cudaMemcpy(slots, ds, sizeof(slots), cudaMemcpyDeviceToHost) ==
            cudaSuccess);

    // Host reference.
    std::vector<float> score(E), choice(E);
    for (int e = 0; e < E; ++e) {
        float acc = 0.0f;
        for (int j = 0; j < H; ++j)
            acc += w[static_cast<size_t>(e * H + j)] *
                   F(x[static_cast<size_t>(j)]);
        score[static_cast<size_t>(e)] = 1.0f / (1.0f + std::exp(-acc));
        choice[static_cast<size_t>(e)] = score[static_cast<size_t>(e)] +
                                         bias[static_cast<size_t>(e)];
    }
    std::vector<int> sel;
    std::vector<bool> taken(E, false);
    for (int k = 0; k < K; ++k) {
        int best = -1;
        float bv = -1e30f;
        for (int e = 0; e < E; ++e)
            if (!taken[e] && choice[static_cast<size_t>(e)] > bv) {
                bv = choice[static_cast<size_t>(e)];
                best = e;
            }
        taken[static_cast<size_t>(best)] = true;
        sel.push_back(best);
    }
    std::sort(sel.begin(), sel.end());
    float sum = 0.0f;
    for (int e : sel) sum += score[static_cast<size_t>(e)];
    // Expert 4 is force-selected by the bias.
    CHECK(std::find(sel.begin(), sel.end(), 4) != sel.end());
    for (int k = 0; k < K; ++k) {
        CHECK(slots[k].eid == sel[static_cast<size_t>(k)]);
        const float want = score[static_cast<size_t>(sel[static_cast<size_t>(k)])] /
                           (sum + 1e-20f) * routed_scale;
        CHECK(std::abs(slots[k].coeff - want) <= 2e-4f * std::max(1.0f, want));
    }
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(db);
    cudaFree(ds);
}

TEST_CASE("Hy routed accumulation and FP32 combine", "[hy][kernels]")
{
    if (!HaveCuda()) SKIP("no CUDA device");
    const int H = 32, K = 3;
    auto y = Pattern(static_cast<int64_t>(K) * H, 0.25f, 7);
    auto shared = Pattern(H, 0.25f, 8);
    HyRoutedSlot h_slots[3] = {{0, 0.5f}, {1, 0.25f}, {2, 1.0f}};
    HyRoutedSlot *ds = nullptr;
    REQUIRE(cudaMalloc(&ds, sizeof(h_slots)) == cudaSuccess);
    REQUIRE(cudaMemcpy(ds, h_slots, sizeof(h_slots),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    __half *dy = ToDevice(y);
    __half *dsh = ToDevice(shared);
    float *acc = nullptr;
    __half *dout = nullptr;
    REQUIRE(cudaMalloc(&acc, sizeof(float) * H) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, sizeof(__half) * H) == cudaSuccess);
    REQUIRE(RunHyExpertAccumulate(nullptr, acc, dy, ds, K, H) == cudaSuccess);
    REQUIRE(RunHyMoECombine(nullptr, acc, dsh, dout, 1, H) == cudaSuccess);
    std::vector<__half> out(H);
    REQUIRE(cudaMemcpy(out.data(), dout, sizeof(__half) * H,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < H; ++i) {
        float a = 0.0f;
        for (int k = 0; k < K; ++k)
            a += h_slots[k].coeff * F(y[static_cast<size_t>(k * H + i)]);
        const float want = a + F(shared[static_cast<size_t>(i)]);
        CHECK(std::abs(F(out[static_cast<size_t>(i)]) - want) <= 2e-3f);
    }
    cudaFree(ds);
    cudaFree(dy);
    cudaFree(dsh);
    cudaFree(acc);
    cudaFree(dout);
}
