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
    explicit DeviceBuffer(size_t bytes) { cudaMalloc(&p, bytes); }
    ~DeviceBuffer() { if (p) cudaFree(p); }
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
        dx.as<float>(), rows, dim, dw.as<uint8_t>(), dup.as<uint8_t>(),
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
