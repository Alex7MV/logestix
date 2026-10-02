// TDD — GigaChat3.5 device-kernel parity vs the pinned host reference in
// gigachat_moe_device_math.h. Executed on the local RTX 5090; label inference.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "gigachat_moe_kernels.h"
#include "gigachat_moe_device_math.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace lsxcommon::gigachat;

namespace {

struct DeviceBuffer {
    void *p = nullptr;
    size_t bytes = 0;
    explicit DeviceBuffer(size_t n) : bytes(n) { cudaMalloc(&p, n); }
    ~DeviceBuffer() { if (p) cudaFree(p); }
    size_t size() const { return bytes; }
    template <typename T> T *as() { return static_cast<T *>(p); }
};

} // namespace

TEST_CASE("gigachat zero-centered gated norm parity", "[gigachat][inference]")
{
    const int rows = 3, dim = 128, rank = 16;
    std::vector<float> x(static_cast<size_t>(rows) * dim, 0.0f);
    std::vector<float> wf(dim, 0.0f), wupf(static_cast<size_t>(rank) * dim),
        wdnf(static_cast<size_t>(dim) * rank);
    for (int i = 0; i < rows * dim; ++i) x[i] = 0.01f * (i % 97) - 0.5f;
    for (int i = 0; i < dim; ++i) wf[i] = 0.001f * (i % 13);
    for (int i = 0; i < rank * dim; ++i) wupf[i] = 0.002f * (i % 11) - 0.01f;
    for (int i = 0; i < dim * rank; ++i) wdnf[i] = 0.003f * (i % 7) - 0.01f;

    auto to_bf16 = [](const std::vector<float> &v) {
        std::vector<uint8_t> b(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) {
            __nv_bfloat16 h = __float2bfloat16(v[i]);
            std::memcpy(b.data() + i * 2, &h, 2);
        }
        return b;
    };
    const auto wb = to_bf16(wf), upb = to_bf16(wupf), dnb = to_bf16(wdnf);

    DeviceBuffer dx(x.size() * 4), dw(wb.size()), dup(upb.size()),
        ddn(dnb.size()), dout(x.size() * 4);
    cudaMemcpy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw.p, wb.data(), wb.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dup.p, upb.data(), upb.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(ddn.p, dnb.data(), dnb.size(), cudaMemcpyHostToDevice);

    auto st = RunZeroCenteredGatedNorm(
        dx.as<float>(), rows, dim, rank, dw.as<uint8_t>(), dup.as<uint8_t>(),
        ddn.as<uint8_t>(), 1e-6f, dout.as<float>(), nullptr);
    REQUIRE(st == cudaSuccess);

    std::vector<float> out(x.size(), 0.0f);
    cudaMemcpy(out.data(), dout.p, out.size() * 4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::vector<float> ref(x.size(), 0.0f);
    for (int r = 0; r < rows; ++r) {
        float ss = 0.0f;
        for (int i = 0; i < dim; ++i)
            ss += x[r * dim + i] * x[r * dim + i];
        const float inv = 1.0f / std::sqrt(ss / dim + 1e-6f);
        std::vector<float> xn(dim);
        for (int i = 0; i < dim; ++i) xn[i] = x[r * dim + i] * inv;
        std::vector<float> h(rank, 0.0f);
        for (int k = 0; k < rank; ++k) {
            float a = 0.0f;
            for (int i = 0; i < dim; ++i)
                a += wupf[k * dim + i] * xn[i];
            h[k] = dev::Silu(a);
        }
        for (int i = 0; i < dim; ++i) {
            float g = 0.0f;
            for (int k = 0; k < rank; ++k) g += wdnf[i * rank + k] * h[k];
            ref[r * dim + i] =
                dev::ZeroCenteredGatedNormElement(xn[i], wf[i], g);
        }
    }
    for (int i = 0; i < rows * dim; ++i)
        CHECK(out[i] == Catch::Approx(ref[i]).margin(2e-4f));
}

TEST_CASE("gigachat zero-centered gated norm honors rank < frozen 16",
          "[gigachat][inference]")
{
    // The gate rank is data (GcGatedNorm::rank), not the frozen constant. A
    // kernel that falls back to 16 reads the (here deliberately large) tail of
    // the full-size gate buffers and diverges deterministically from the
    // rank-only reference; this pins the rank argument without relying on an
    // out-of-bounds access being observed.
    constexpr int dim = 8, rank = 3, full = 16;
    std::vector<float> x(dim), wf(dim);
    for (int i = 0; i < dim; ++i) {
        x[i] = 0.05f * static_cast<float>(i - 3);
        wf[i] = 0.01f * static_cast<float>(i);
    }

    auto to_bf16 = [](const std::vector<float> &v) {
        std::vector<uint8_t> b(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) {
            __nv_bfloat16 h = __float2bfloat16(v[i]);
            std::memcpy(b.data() + i * 2, &h, 2);
        }
        return b;
    };
    // w_up [full, dim]: rows [0,rank) meaningful, [rank,full) large.
    std::vector<float> upf(static_cast<size_t>(full) * dim, 7.0f);
    for (int k = 0; k < rank; ++k)
        for (int i = 0; i < dim; ++i)
            upf[static_cast<size_t>(k) * dim + i] =
                0.02f * static_cast<float>((k * dim + i) % 11) - 0.05f;
    // w_down: [dim, rank] prefix meaningful, the rest large.
    std::vector<float> dnf(static_cast<size_t>(full) * dim, 7.0f);
    for (int i = 0; i < dim; ++i)
        for (int k = 0; k < rank; ++k)
            dnf[static_cast<size_t>(i) * rank + k] =
                0.03f * static_cast<float>((i * rank + k) % 7) - 0.05f;

    const auto wb = to_bf16(wf), upb = to_bf16(upf), dnb = to_bf16(dnf);
    DeviceBuffer dx(dim * 4), dw(wb.size()), dup(upb.size()), ddn(dnb.size()),
        dout(dim * 4);
    REQUIRE(cudaMemcpy(dx.p, x.data(), dx.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dw.p, wb.data(), wb.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dup.p, upb.data(), upb.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(ddn.p, dnb.data(), dnb.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);

    REQUIRE(RunZeroCenteredGatedNorm(
                dx.as<float>(), 1, dim, rank, dw.as<uint8_t>(),
                dup.as<uint8_t>(), ddn.as<uint8_t>(), 1e-6f,
                dout.as<float>(), nullptr) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> out(dim, 0.0f);
    REQUIRE(cudaMemcpy(out.data(), dout.p, out.size() * 4,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    float ss = 0.0f;
    for (int i = 0; i < dim; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss / dim + 1e-6f);
    std::vector<float> xn(dim);
    for (int i = 0; i < dim; ++i) xn[i] = x[i] * inv;
    std::vector<float> h(rank, 0.0f);
    for (int k = 0; k < rank; ++k) {
        float a = 0.0f;
        for (int i = 0; i < dim; ++i)
            a += upf[static_cast<size_t>(k) * dim + i] * xn[i];
        h[k] = dev::Silu(a);
    }
    for (int i = 0; i < dim; ++i) {
        float g = 0.0f;
        for (int k = 0; k < rank; ++k)
            g += dnf[static_cast<size_t>(i) * rank + k] * h[k];
        const float ref = dev::ZeroCenteredGatedNormElement(xn[i], wf[i], g);
        INFO("i=" << i << " ref=" << ref << " got=" << out[i]);
        CHECK(out[i] == Catch::Approx(ref).margin(1e-3f));
    }
}

TEST_CASE("gigachat gdn output norm parity", "[gigachat][inference]")
{
    const int heads = 4, hd = 128;
    const float eps = 1e-6f, scale = 2.0f;
    std::vector<float> o(static_cast<size_t>(heads) * hd),
        z(static_cast<size_t>(heads) * hd), wf(hd);
    for (int i = 0; i < heads * hd; ++i) {
        o[i] = 0.02f * (i % 53) - 0.5f;
        z[i] = 0.03f * (i % 31) - 0.4f;
    }
    for (int i = 0; i < hd; ++i) wf[i] = 0.001f * (i % 17);

    std::vector<uint8_t> wb(hd * 2);
    for (int i = 0; i < hd; ++i) {
        __nv_bfloat16 h = __float2bfloat16(wf[i]);
        std::memcpy(wb.data() + i * 2, &h, 2);
    }

    DeviceBuffer d_o(o.size() * 4), d_z(z.size() * 4), d_w(wb.size()),
        d_out(o.size() * 4);
    cudaMemcpy(d_o.p, o.data(), o.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z.p, z.data(), z.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w.p, wb.data(), wb.size(), cudaMemcpyHostToDevice);

    auto st = RunGdnOutputNorm(d_o.as<float>(), d_z.as<float>(), d_w.as<uint8_t>(),
                               heads, hd, eps, scale, d_out.as<float>(), nullptr);
    REQUIRE(st == cudaSuccess);

    std::vector<float> out(o.size(), 0.0f);
    cudaMemcpy(out.data(), d_out.p, out.size() * 4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int h = 0; h < heads; ++h) {
        float ss = 0.0f;
        for (int d = 0; d < hd; ++d) {
            const float v = o[h * hd + d];
            ss += v * v;
        }
        const float inv = 1.0f / std::sqrt(ss / hd + eps);
        for (int d = 0; d < hd; ++d) {
            const float xn = o[h * hd + d] * inv;
            const float expect = dev::GdnOutputNormElement(
                xn, wf[d], z[h * hd + d], eps, scale);
            CHECK(out[h * hd + d] == Catch::Approx(expect).margin(2e-4f));
        }
    }
}

TEST_CASE("gigachat swiglu-limit and sigmoid gate parity", "[gigachat][inference]")
{
    const int n = 256;
    const float L = 10.0f;
    std::vector<float> g(n), u(n), y(n);
    for (int i = 0; i < n; ++i) {
        g[i] = 0.27f * (i % 41) - 5.0f;
        u[i] = 0.31f * (i % 37) - 5.0f;
        y[i] = 0.13f * (i % 29) - 2.0f;
    }
    DeviceBuffer d_g(n * 4), d_u(n * 4), d_m(n * 4), d_y(n * 4);
    cudaMemcpy(d_g.p, g.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u.p, u.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y.p, y.data(), n * 4, cudaMemcpyHostToDevice);

    REQUIRE(RunSwigluLimit(d_g.as<float>(), d_u.as<float>(), L, n,
                           d_m.as<float>(), nullptr) == cudaSuccess);
    std::vector<float> mid(n, 0.0f);
    cudaMemcpy(mid.data(), d_m.p, n * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        CHECK(mid[i] ==
              Catch::Approx(dev::SwigluLimitElement(g[i], u[i], L)).margin(1e-5f));

    REQUIRE(RunSigmoidGate(d_y.as<float>(), d_g.as<float>(), n, nullptr) ==
            cudaSuccess);
    std::vector<float> yy(n, 0.0f);
    cudaMemcpy(yy.data(), d_y.p, n * 4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; ++i)
        CHECK(yy[i] ==
              Catch::Approx(dev::SigmoidGateElement(y[i], g[i])).margin(1e-5f));
}

// Decode runs 161 gated norms and 37 routers per token; both launch as
// single-block kernels and are the top two terms of the decode wall
// (docs/GIGACHAT_SPEED_AUDIT.md). Pin the real-shape per-launch latency. The
// norm weights (~470 KB) and router weight (3.6 MB) are L2-warm in production
// (reused every layer/token), so a back-to-back loop measures the same thing.
TEST_CASE("gigachat gated norm decode latency budget",
          "[gigachat][inference][budget]")
{
#if defined(NDEBUG)
    constexpr int dim = 7168, rank = 16, kIters = 300;
    std::vector<float> x(dim);
    for (int i = 0; i < dim; ++i) x[i] = 0.01f * (i % 97) - 0.5f;
    std::vector<uint8_t> wb(static_cast<size_t>(dim) * 2),
        upb(static_cast<size_t>(rank) * dim * 2),
        dnb(static_cast<size_t>(dim) * rank * 2);
    for (int i = 0; i < dim; ++i) {
        __nv_bfloat16 h = __float2bfloat16(0.001f * (i % 13));
        std::memcpy(wb.data() + i * 2, &h, 2);
    }
    for (int i = 0; i < rank * dim; ++i) {
        __nv_bfloat16 h = __float2bfloat16(0.002f * (i % 11) - 0.01f);
        std::memcpy(upb.data() + i * 2, &h, 2);
    }
    for (int i = 0; i < dim * rank; ++i) {
        __nv_bfloat16 h = __float2bfloat16(0.003f * (i % 7) - 0.01f);
        std::memcpy(dnb.data() + i * 2, &h, 2);
    }
    DeviceBuffer dx(static_cast<size_t>(dim) * 4), dw(wb.size()),
        dup(upb.size()), ddn(dnb.size()), dout(static_cast<size_t>(dim) * 4);
    REQUIRE(cudaMemcpy(dx.p, x.data(), dx.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dw.p, wb.data(), wb.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dup.p, upb.data(), upb.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(ddn.p, dnb.data(), dnb.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    auto run = [&] {
        return RunZeroCenteredGatedNorm(dx.as<float>(), 1, dim, rank,
                                        dw.as<uint8_t>(), dup.as<uint8_t>(),
                                        ddn.as<uint8_t>(), 1e-6f,
                                        dout.as<float>(), nullptr);
    };
    for (int i = 0; i < 50; ++i) REQUIRE(run() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cudaEvent_t e0, e1;
    REQUIRE(cudaEventCreate(&e0) == cudaSuccess);
    REQUIRE(cudaEventCreate(&e1) == cudaSuccess);
    REQUIRE(cudaEventRecord(e0) == cudaSuccess);
    for (int i = 0; i < kIters; ++i) REQUIRE(run() == cudaSuccess);
    REQUIRE(cudaEventRecord(e1) == cudaSuccess);
    REQUIRE(cudaEventSynchronize(e1) == cudaSuccess);
    float ms = 0.0f;
    REQUIRE(cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess);
    const double us = static_cast<double>(ms) * 1000.0 / kIters;
    INFO("gated norm dim=7168 latency = " << us << " us");
    CHECK(us < 40.0);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
#else
    // Measured 26.1 us against the 40.0 us budget on an idle RTX 5090 (34%
    // headroom), and 26.2 us with a competing CUDA process resident — so the
    // budget is not sensitive to ordinary contention. An unoptimized build
    // cannot be judged against it at all, so SKIP rather than pass silently:
    // before this, the case reported "passed" with zero assertions in Debug.
    SKIP("latency budget requires an optimized build (NDEBUG)");
#endif
}

TEST_CASE("gigachat router decode latency budget",
          "[gigachat][inference][budget]")
{
#if defined(NDEBUG)
    constexpr int E = 256, H = 7168, top_k = 8, kIters = 300;
    std::vector<float> x(H), bias(E, 0.0f);
    for (int i = 0; i < H; ++i) x[i] = 0.02f * (i % 23) - 0.2f;
    std::vector<uint8_t> wb(static_cast<size_t>(E) * H * 2);
    for (size_t i = 0; i < static_cast<size_t>(E) * H; ++i) {
        __nv_bfloat16 h = __float2bfloat16(0.005f * (i % 19) - 0.05f);
        std::memcpy(wb.data() + i * 2, &h, 2);
    }
    DeviceBuffer dx(static_cast<size_t>(H) * 4), dw(wb.size()),
        db(static_cast<size_t>(E) * 4), didx(static_cast<size_t>(top_k) * 4),
        dwt(static_cast<size_t>(top_k) * 4);
    REQUIRE(cudaMemcpy(dx.p, x.data(), dx.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(dw.p, wb.data(), wb.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(db.p, bias.data(), db.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    auto run = [&] {
        return RunGigaChatRouterTopK(dx.as<float>(), dw.as<uint8_t>(),
                                     db.as<float>(), E, H, top_k, 2.5f,
                                     didx.as<int>(), dwt.as<float>(), nullptr);
    };
    for (int i = 0; i < 50; ++i) REQUIRE(run() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cudaEvent_t e0, e1;
    REQUIRE(cudaEventCreate(&e0) == cudaSuccess);
    REQUIRE(cudaEventCreate(&e1) == cudaSuccess);
    REQUIRE(cudaEventRecord(e0) == cudaSuccess);
    for (int i = 0; i < kIters; ++i) REQUIRE(run() == cudaSuccess);
    REQUIRE(cudaEventRecord(e1) == cudaSuccess);
    REQUIRE(cudaEventSynchronize(e1) == cudaSuccess);
    float ms = 0.0f;
    REQUIRE(cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess);
    const double us = static_cast<double>(ms) * 1000.0 / kIters;
    INFO("router E=256 H=7168 latency = " << us << " us");
    CHECK(us < 60.0);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
#else
    SKIP("latency budget requires an optimized build (NDEBUG)");
#endif
}

TEST_CASE("gigachat sigmoid router top-k parity", "[gigachat][inference]")
{
    const int E = 8, H = 64, top_k = 2;
    const float scale = 2.5f;
    std::vector<float> x(H), wf(static_cast<size_t>(E) * H), bias(E);
    for (int i = 0; i < H; ++i) x[i] = 0.02f * (i % 23) - 0.2f;
    for (int i = 0; i < E * H; ++i) wf[i] = 0.005f * (i % 19) - 0.05f;
    // Bias flips the top-2 relative to raw scores (expert 6 above expert 0).
    for (int e = 0; e < E; ++e) bias[e] = 0.0f;
    bias[6] = 5.0f;

    std::vector<uint8_t> wb(wf.size() * 2);
    for (size_t i = 0; i < wf.size(); ++i) {
        __nv_bfloat16 h = __float2bfloat16(wf[i]);
        std::memcpy(wb.data() + i * 2, &h, 2);
    }
    DeviceBuffer d_x(H * 4), d_w(wb.size()), d_b(E * 4), d_idx(top_k * 4),
        d_wt(top_k * 4);
    cudaMemcpy(d_x.p, x.data(), H * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w.p, wb.data(), wb.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b.p, bias.data(), E * 4, cudaMemcpyHostToDevice);

    REQUIRE(RunGigaChatRouterTopK(d_x.as<float>(), d_w.as<uint8_t>(),
                                  d_b.as<float>(), E, H, top_k, scale,
                                  d_idx.as<int>(), d_wt.as<float>(),
                                  nullptr) == cudaSuccess);
    std::vector<int> idx(top_k);
    std::vector<float> wt(top_k);
    cudaMemcpy(idx.data(), d_idx.p, top_k * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(wt.data(), d_wt.p, top_k * 4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Host reference.
    std::vector<float> scores(E), choice(E);
    for (int e = 0; e < E; ++e) {
        float a = 0.0f;
        for (int i = 0; i < H; ++i) a += wf[e * H + i] * x[i];
        scores[e] = dev::Sigmoid(a);
        choice[e] = scores[e] + bias[e];
    }
    std::vector<int> ridx(top_k);
    for (int k = 0; k < top_k; ++k) {
        int best = -1;
        float bv = -1e30f;
        for (int e = 0; e < E; ++e) {
            bool used = false;
            for (int j = 0; j < k; ++j) if (ridx[j] == e) used = true;
            if (!used && choice[e] > bv) { bv = choice[e]; best = e; }
        }
        ridx[k] = best;
    }
    float sum = 0.0f;
    for (int k = 0; k < top_k; ++k) sum += scores[ridx[k]];
    for (int k = 0; k < top_k; ++k) {
        CHECK(idx[k] == ridx[k]);
        CHECK(wt[k] ==
              Catch::Approx(scores[ridx[k]] / (sum + 1e-20f) * scale).margin(2e-3f));
    }
}
