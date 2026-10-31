#include <catch2/catch_all.hpp>

#include <lsxcommon/granite_dense_switch_inference.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace lsxcommon::granite_switch;

namespace {

template <typename T>
T *Dev(const std::vector<T> &h)
{
    T *d = nullptr;
    REQUIRE(cudaMalloc(&d, h.size() * sizeof(T)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
}

template <typename T>
std::vector<T> Host(const T *d, size_t n)
{
    std::vector<T> h(n);
    REQUIRE(cudaMemcpy(h.data(), d, n * sizeof(T),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    return h;
}

std::vector<__half> H(const std::vector<float> &v)
{
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half_rn(v[i]);
    return h;
}

float F(__half x) { return __half2float(x); }

} // namespace

TEST_CASE("granite-switch scale and scale-add", "[granite_switch][kernels]")
{
    std::vector<float> x(16);
    for (int i = 0; i < 16; ++i) x[i] = 1.0f + i;
    auto hx = H(x);
    __half *dx = Dev(hx);
    REQUIRE(RunGraniteSwitchScale(0, dx, 16, 12.0f) == cudaSuccess);
    auto got = Host(dx, 16);
    for (int i = 0; i < 16; ++i)
        CHECK(F(got[i]) == Catch::Approx((1.0f + i) * 12.0f).margin(1e-2));

    std::vector<float> src(16, 2.0f), dst(16, 1.0f);
    auto hs = H(src), hd = H(dst);
    __half *ds = Dev(hs), *dd = Dev(hd);
    REQUIRE(RunGraniteSwitchScaleAdd(0, dd, ds, 16, 0.175f) == cudaSuccess);
    auto added = Host(dd, 16);
    for (int i = 0; i < 16; ++i)
        CHECK(F(added[i]) == Catch::Approx(1.0f + 2.0f * 0.175f).margin(1e-3));
    cudaFree(dx); cudaFree(ds); cudaFree(dd);
}

TEST_CASE("granite-switch aLoRA shrink/expand", "[granite_switch][kernels]")
{
    const int seq = 3, K = 8, rank = 4, N = 6;
    std::vector<float> x(seq * K), A(rank * K), B(N * rank), base(seq * N, 0.5f);
    for (int i = 0; i < seq * K; ++i) x[i] = 0.1f * (i + 1);
    for (int i = 0; i < rank * K; ++i) A[i] = 0.05f * ((i % 5) - 2);
    for (int i = 0; i < N * rank; ++i) B[i] = 0.03f * ((i % 7) - 3);
    std::vector<int32_t> adapter = {0, 1, 2};

    auto hx = H(x), hA = H(A), hB = H(B), hbase = H(base);
    __half *dx = Dev(hx), *dA = Dev(hA), *dB = Dev(hB), *dout = Dev(hbase);
    __half *dy = nullptr;
    REQUIRE(cudaMalloc(&dy, seq * 64 * sizeof(__half)) == cudaSuccess);
    int32_t *dad = Dev(adapter);

    REQUIRE(RunGraniteSwitchLoraShrink(0, dx, dA, dad, dy, seq, K, rank, 64) ==
            cudaSuccess);
    REQUIRE(RunGraniteSwitchLoraExpandAdd(0, dy, dB, dad, dout, seq, rank, N,
                                          N, 0, 64) == cudaSuccess);
    auto got = Host(dout, seq * N);

    for (int t = 0; t < seq; ++t) {
        for (int n = 0; n < N; ++n) {
            double ref = 0.5;
            if (adapter[t] != 0) {
                double acc = 0;
                for (int r = 0; r < rank; ++r) {
                    double s = 0;
                    for (int k = 0; k < K; ++k) s += (double)x[t * K + k] * A[r * K + k];
                    acc += s * B[n * rank + r];
                }
                ref += acc;
            }
            CHECK(F(got[t * N + n]) == Catch::Approx(ref).margin(2e-2));
        }
    }
    cudaFree(dx); cudaFree(dA); cudaFree(dB); cudaFree(dout);
    cudaFree(dy); cudaFree(dad);
}

TEST_CASE("granite-switch norm and swiglu", "[granite_switch][kernels]")
{
    const int rows = 2, H_ = 8, I = 4;
    std::vector<float> x(rows * H_);
    for (int i = 0; i < rows * H_; ++i) x[i] = 0.1f * (i - 4);
    std::vector<float> w(H_, 1.0f);
    for (int i = 0; i < H_; ++i) w[i] = 0.9f + 0.1f * i;
    auto hx = H(x), hw = H(w);
    __half *dx = Dev(hx), *dw = Dev(hw), *dout = nullptr;
    REQUIRE(cudaMalloc(&dout, rows * H_ * sizeof(__half)) == cudaSuccess);
    REQUIRE(RunGraniteSwitchRmsNormKernel(0, dout, dx, dw, rows, H_, 1e-5f) ==
            cudaSuccess);
    auto got = Host(dout, rows * H_);
    for (int r = 0; r < rows; ++r) {
        double ss = 0;
        for (int i = 0; i < H_; ++i) ss += (double)x[r * H_ + i] * x[r * H_ + i];
        double inv = 1.0 / std::sqrt(ss / H_ + 1e-5);
        for (int i = 0; i < H_; ++i)
            CHECK(F(got[r * H_ + i]) ==
                  Catch::Approx(x[r * H_ + i] * inv * w[i]).margin(2e-2));
    }

    std::vector<float> gu(2 * I);
    for (int i = 0; i < 2 * I; ++i) gu[i] = 0.2f * (i - 3);
    auto hgu = H(gu);
    __half *dgu = Dev(hgu), *dsg = nullptr;
    REQUIRE(cudaMalloc(&dsg, I * sizeof(__half)) == cudaSuccess);
    REQUIRE(RunGraniteSwitchSwiGLU(0, dgu, dsg, I, I) == cudaSuccess);
    auto sg = Host(dsg, I);
    for (int i = 0; i < I; ++i) {
        double g = gu[i], u = gu[I + i];
        double ref = (g / (1.0 + std::exp(-g))) * u;
        CHECK(F(sg[i]) == Catch::Approx(ref).margin(2e-2));
    }
    cudaFree(dx); cudaFree(dw); cudaFree(dout); cudaFree(dgu); cudaFree(dsg);
}

TEST_CASE("granite-switch rope and attention", "[granite_switch][kernels]")
{
    const int seq = 3, nh = 2, nkv = 1, hd = 4;
    std::vector<float> q(seq * nh * hd), k(seq * nkv * hd), v(seq * nkv * hd);
    for (int i = 0; i < seq * nh * hd; ++i) q[i] = 0.05f * (i % 7);
    for (int i = 0; i < seq * nkv * hd; ++i) k[i] = 0.03f * (i % 5);
    for (int i = 0; i < seq * nkv * hd; ++i) v[i] = 0.07f * (i % 3);
    std::vector<int32_t> pos = {0, 1, 2};
    auto hq = H(q), hk = H(k), hv = H(v);
    __half *dq = Dev(hq), *dk = Dev(hk), *dv = Dev(hv), *dout = nullptr;
    REQUIRE(cudaMalloc(&dout, seq * nh * hd * sizeof(__half)) == cudaSuccess);
    REQUIRE(RunGraniteSwitchAttentionKernel(0, dq, dk, dv, dout, seq, nh, nkv,
                                            hd, 0.5f, 0) == cudaSuccess);
    auto got = Host(dout, seq * nh * hd);
    for (int t = 0; t < seq; ++t)
        for (int h = 0; h < nh; ++h) {
            std::vector<double> sc(t + 1);
            for (int s = 0; s <= t; ++s) {
                double dot = 0;
                for (int d = 0; d < hd; ++d)
                    dot += q[(t * nh + h) * hd + d] * k[s * nkv * hd + d];
                sc[s] = dot * 0.5;
            }
            double m = *std::max_element(sc.begin(), sc.end());
            double z = 0;
            std::vector<double> e(t + 1);
            for (int s = 0; s <= t; ++s) { e[s] = std::exp(sc[s] - m); z += e[s]; }
            for (int d = 0; d < hd; ++d) {
                double ref = 0;
                for (int s = 0; s <= t; ++s)
                    ref += e[s] / z * v[s * nkv * hd + d];
                CHECK(F(got[(t * nh + h) * hd + d]) ==
                      Catch::Approx(ref).margin(2e-2));
            }
        }
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
}
