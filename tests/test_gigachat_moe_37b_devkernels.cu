// TDD — GigaChat4-37B-A4B device kernels: the variant gated norm (offset 0 vs
// the 3.5 zero-centred offset 1), the sigmoid GDN output norm and the logit
// soft cap. Device parity against a double host reference, and bit-identity of
// the legacy 3.5 entry point after the variant refactor.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "gigachat_moe_kernels.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using bf16 = uint16_t;

float Bf16ToF32(bf16 b)
{
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

bf16 F32ToBf16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    // round-to-nearest-even on the 16 dropped bits
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<bf16>(bits >> 16);
}

float Sigmoid(double v) { return static_cast<float>(1.0 / (1.0 + std::exp(-v))); }

struct DevBuffer {
    void *p = nullptr;
    explicit DevBuffer(size_t bytes)
    {
        cudaMalloc(&p, bytes);
    }
    ~DevBuffer()
    {
        if (p) cudaFree(p);
    }
    DevBuffer(const DevBuffer &) = delete;
    DevBuffer &operator=(const DevBuffer &) = delete;
    template <typename T>
    T *as() { return static_cast<T *>(p); }
};

bool HaveDevice()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

} // namespace

TEST_CASE("gigachat4 gated norm kernel honours the variant offset",
          "[gigachat4][devkernels]")
{
    if (!HaveDevice()) SKIP("no CUDA device");
    constexpr int kDim = 128;
    constexpr int kRank = 16;
    std::vector<bf16> w(kDim), w_up(kRank * kDim), w_down(kDim * kRank);
    std::vector<float> x(kDim);
    for (int i = 0; i < kDim; ++i)
    {
        x[i] = 0.02f * static_cast<float>((i % 11) - 5);
        w[i] = F32ToBf16(0.01f * static_cast<float>((i % 7) - 3));
    }
    for (int k = 0; k < kRank; ++k)
        for (int i = 0; i < kDim; ++i)
        {
            w_up[k * kDim + i] =
                F32ToBf16(0.05f * static_cast<float>(((i + k) % 9) - 4));
            w_down[i * kRank + k] =
                F32ToBf16(0.03f * static_cast<float>(((i * 3 + k) % 7) - 3));
        }

    // Host reference in double.
    double ss = 0.0;
    for (int i = 0; i < kDim; ++i) ss += double(x[i]) * x[i];
    const double inv = 1.0 / std::sqrt(ss / kDim + 1e-6);
    std::vector<double> xn(kDim);
    for (int i = 0; i < kDim; ++i) xn[i] = x[i] * inv;
    std::vector<double> gate(kRank, 0.0);
    for (int k = 0; k < kRank; ++k)
    {
        double a = 0.0;
        for (int i = 0; i < kDim; ++i) a += double(Bf16ToF32(w_up[k * kDim + i])) * xn[i];
        gate[k] = a / (1.0 + std::exp(-a));  // silu
    }
    auto reference = [&](double offset) {
        std::vector<double> out(kDim);
        for (int i = 0; i < kDim; ++i)
        {
            double g = 0.0;
            for (int k = 0; k < kRank; ++k)
                g += double(Bf16ToF32(w_down[i * kRank + k])) * gate[k];
            const double s = 1.0 / (1.0 + std::exp(-g));
            out[i] = xn[i] * (offset + double(Bf16ToF32(w[i]))) * 2.0 * s;
        }
        return out;
    };
    const auto want0 = reference(0.0);
    const auto want1 = reference(1.0);

    DevBuffer d_w(kDim * 2), d_up(kRank * kDim * 2), d_down(kDim * kRank * 2),
        d_x(kDim * 4), d_out(kDim * 4), d_legacy(kDim * 4);
    cudaMemcpy(d_w.p, w.data(), kDim * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_up.p, w_up.data(), kRank * kDim * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_down.p, w_down.data(), kDim * kRank * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x.p, x.data(), kDim * 4, cudaMemcpyHostToDevice);

    REQUIRE(lsxcommon::gigachat::RunGigaChatGatedNorm(
                d_x.as<float>(), 1, kDim, kRank,
                static_cast<const uint8_t *>(d_w.p),
                static_cast<const uint8_t *>(d_up.p),
                static_cast<const uint8_t *>(d_down.p), 1e-6f, 0.0f,
                d_out.as<float>(), nullptr) == cudaSuccess);
    std::vector<float> got0(kDim);
    cudaMemcpy(got0.data(), d_out.p, kDim * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < kDim; ++i)
        CHECK(got0[static_cast<size_t>(i)] ==
              Catch::Approx(want0[static_cast<size_t>(i)]).margin(2e-3));

    // Offset 1 must reproduce the 3.5 kernel bit-exactly: the refactor may not
    // move a single bit of the sibling's numerics.
    REQUIRE(lsxcommon::gigachat::RunGigaChatGatedNorm(
                d_x.as<float>(), 1, kDim, kRank,
                static_cast<const uint8_t *>(d_w.p),
                static_cast<const uint8_t *>(d_up.p),
                static_cast<const uint8_t *>(d_down.p), 1e-6f, 1.0f,
                d_out.as<float>(), nullptr) == cudaSuccess);
    REQUIRE(lsxcommon::gigachat::RunZeroCenteredGatedNorm(
                d_x.as<float>(), 1, kDim, kRank,
                static_cast<const uint8_t *>(d_w.p),
                static_cast<const uint8_t *>(d_up.p),
                static_cast<const uint8_t *>(d_down.p), 1e-6f,
                d_legacy.as<float>(), nullptr) == cudaSuccess);
    std::vector<float> got1(kDim), legacy(kDim);
    cudaMemcpy(got1.data(), d_out.p, kDim * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(legacy.data(), d_legacy.p, kDim * 4, cudaMemcpyDeviceToHost);
    CHECK(std::memcmp(got1.data(), legacy.data(), kDim * 4) == 0);
    for (int i = 0; i < kDim; ++i)
        CHECK(got1[static_cast<size_t>(i)] ==
              Catch::Approx(want1[static_cast<size_t>(i)]).margin(2e-3));
}

TEST_CASE("gigachat4 GDN output norm honours the variant offset",
          "[gigachat4][devkernels]")
{
    if (!HaveDevice()) SKIP("no CUDA device");
    constexpr int kHeads = 2;
    constexpr int kHd = 128;
    std::vector<float> o(kHeads * kHd), z(kHeads * kHd);
    std::vector<bf16> w(kHd);
    for (int h = 0; h < kHeads; ++h)
        for (int d = 0; d < kHd; ++d)
        {
            o[h * kHd + d] = 0.03f * static_cast<float>(((h * 7 + d) % 13) - 6);
            z[h * kHd + d] = 0.1f * static_cast<float>((d % 5) - 2);
        }
    for (int d = 0; d < kHd; ++d) w[d] = F32ToBf16(0.02f * static_cast<float>((d % 7) - 3));

    std::vector<double> want(kHeads * kHd);
    for (int h = 0; h < kHeads; ++h)
    {
        double ss = 0.0;
        for (int d = 0; d < kHd; ++d) ss += double(o[h * kHd + d]) * o[h * kHd + d];
        const double inv = 1.0 / std::sqrt(ss / kHd + 1e-6);
        for (int d = 0; d < kHd; ++d)
        {
            const double xn = o[h * kHd + d] * inv;
            const double s = 1.0 / (1.0 + std::exp(-double(z[h * kHd + d])));
            want[h * kHd + d] =
                xn * double(Bf16ToF32(w[d])) * 2.0 * s;  // offset 0
        }
    }

    DevBuffer d_o(kHeads * kHd * 4), d_z(kHeads * kHd * 4), d_w(kHd * 2),
        d_out(kHeads * kHd * 4);
    cudaMemcpy(d_o.p, o.data(), kHeads * kHd * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z.p, z.data(), kHeads * kHd * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w.p, w.data(), kHd * 2, cudaMemcpyHostToDevice);
    REQUIRE(lsxcommon::gigachat::RunGdnOutputNormOffset(
                d_o.as<float>(), d_z.as<float>(),
                static_cast<const uint8_t *>(d_w.p), kHeads, kHd, 1e-6f, 2.0f,
                0.0f, d_out.as<float>(), nullptr) == cudaSuccess);
    std::vector<float> got(kHeads * kHd);
    cudaMemcpy(got.data(), d_out.p, kHeads * kHd * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < kHeads * kHd; ++i)
        CHECK(got[static_cast<size_t>(i)] ==
              Catch::Approx(want[static_cast<size_t>(i)]).margin(2e-3));
}

TEST_CASE("gigachat4 logit softcap kernel matches the host formula",
          "[gigachat4][devkernels]")
{
    if (!HaveDevice()) SKIP("no CUDA device");
    constexpr int kCap = 30;
    constexpr int kN = 9;
    std::vector<float> logits = {-1.0e6f, -100.0f, -30.0f, -3.0f, 0.0f,
                                 3.0f,   30.0f,   100.0f, 1.0e6f};
    DevBuffer d(kN * 4);
    cudaMemcpy(d.p, logits.data(), kN * 4, cudaMemcpyHostToDevice);
    REQUIRE(lsxcommon::gigachat::RunGigaChatSoftcap(d.as<float>(), kN,
                                                    static_cast<float>(kCap),
                                                    nullptr) == cudaSuccess);
    std::vector<float> got(kN);
    cudaMemcpy(got.data(), d.p, kN * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < kN; ++i)
    {
        const double want = double(kCap) * std::tanh(double(logits[static_cast<size_t>(i)]) / kCap);
        CHECK(got[static_cast<size_t>(i)] == Catch::Approx(want).margin(1e-4));
        CHECK(std::abs(got[static_cast<size_t>(i)]) <=
              static_cast<float>(kCap) + 1e-4f);
    }
}
