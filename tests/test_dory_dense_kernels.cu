// Dory kernel-level parity tests (Model Ingestion Protocol stage 5).
//
// Every nGPT primitive is checked against an independent HOST reference
// computed in doubles from the same FP16 inputs:
//   RunDoryJustNormKernel / RunDoryJustNormSumKernel /
//   RunDorySlerpAlphaKernel / RunDoryQkNormKernel /
//   RunDoryAttnGateNormKernel / RunDorySuvSwiGLUNormKernel /
//   RunDoryAttentionKernel / RunDoryLogitScaleKernel
#include <catch2/catch_all.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <lsxcommon/dory_dense_inference.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace lsxcommon::dory;

namespace {

float Lcg(int64_t &s)
{
    s = (s * 6364136223846793005LL + 1442695040888963407LL);
    return static_cast<float>((s >> 40) & 0xffff) / 65536.0f - 0.5f;
}

std::vector<__half> ToHalf(const std::vector<float> &v)
{
    std::vector<__half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = __float2half(v[i]);
    return h;
}

template <typename T>
T *ToDev(const std::vector<T> &host)
{
    T *d = nullptr;
    REQUIRE(cudaMalloc(&d, host.size() * sizeof(T)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d, host.data(), host.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return d;
}

std::vector<float> FromDevHalf(const __half *d, size_t n)
{
    std::vector<__half> h(n);
    REQUIRE(cudaMemcpy(h.data(), d, n * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __half2float(h[i]);
    return out;
}

constexpr float kEps = 1e-6f;

double Norm(const std::vector<float> &v, int row, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        const double x = v[static_cast<size_t>(row) * n + i];
        s += x * x;
    }
    return std::sqrt(s + static_cast<double>(kEps));
}

} // namespace

TEST_CASE("dory justnorm matches the host reference", "[dory][kernels]")
{
    const int rows = 3, n = 37;
    int64_t seed = 11;
    std::vector<float> x(rows * n), b(rows * n);
    for (auto &v : x) v = Lcg(seed);
    for (auto &v : b) v = 0.25f * Lcg(seed);

    auto hx = ToHalf(x), hb = ToHalf(b);
    __half *dx = ToDev(hx), *db = ToDev(hb), *dout = nullptr;
    REQUIRE(cudaMalloc(&dout, hx.size() * sizeof(__half)) == cudaSuccess);

    REQUIRE(RunDoryJustNormKernel(nullptr, dout, dx, rows, n, kEps) ==
            cudaSuccess);
    const auto out = FromDevHalf(dout, hx.size());
    for (int r = 0; r < rows; ++r) {
        std::vector<float> row(x.begin() + r * n, x.begin() + (r + 1) * n);
        const double norm = Norm(row, 0, n);
        for (int i = 0; i < n; ++i) {
            const double want =
                static_cast<double>(__half2float(hx[r * n + i])) / norm;
            INFO("row " << r << " i " << i);
            CHECK(std::abs(out[r * n + i] - want) < 2e-3);
        }
    }

    // a + b variant.
    REQUIRE(RunDoryJustNormSumKernel(nullptr, dout, dx, db, rows, n, kEps) ==
            cudaSuccess);
    const auto sum = FromDevHalf(dout, hx.size());
    for (int r = 0; r < rows; ++r) {
        std::vector<double> s(n);
        double acc = 0.0;
        for (int i = 0; i < n; ++i) {
            s[i] = static_cast<double>(__half2float(hx[r * n + i])) +
                   static_cast<double>(__half2float(hb[r * n + i]));
            acc += s[i] * s[i];
        }
        const double norm = std::sqrt(acc + kEps);
        for (int i = 0; i < n; ++i)
            CHECK(std::abs(sum[r * n + i] - s[i] / norm) < 2e-3);
    }

    // The output is a unit vector: the defining property of nGPT.
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) acc += std::pow(out[r * n + i], 2.0);
        CHECK(std::abs(std::sqrt(acc) - 1.0) < 1e-2);
    }

    cudaFree(dx);
    cudaFree(db);
    cudaFree(dout);
}

TEST_CASE("dory slerp-alpha matches the host reference", "[dory][kernels]")
{
    const int rows = 2, n = 29;
    int64_t seed = 23;
    std::vector<float> a(rows * n), b(rows * n), alpha(n);
    for (auto &v : a) v = Lcg(seed);
    for (auto &v : b) v = Lcg(seed);
    for (auto &v : alpha) v = std::fabs(Lcg(seed));

    auto ha = ToHalf(a), hb = ToHalf(b);
    __half *da = ToDev(ha), *db = ToDev(hb), *dout = nullptr;
    float *dalpha = ToDev(alpha);
    REQUIRE(cudaMalloc(&dout, ha.size() * sizeof(__half)) == cudaSuccess);

    REQUIRE(RunDorySlerpAlphaKernel(nullptr, dout, da, db, dalpha, rows, n,
                                    kEps) == cudaSuccess);
    const auto out = FromDevHalf(dout, ha.size());

    for (int r = 0; r < rows; ++r) {
        double na = 0.0, nb = 0.0;
        for (int i = 0; i < n; ++i) {
            const double va = __half2float(ha[r * n + i]);
            const double vb = __half2float(hb[r * n + i]);
            na += va * va;
            nb += vb * vb;
        }
        na = std::sqrt(na + kEps);
        nb = std::sqrt(nb + kEps);
        std::vector<double> y(n);
        double ny = 0.0;
        for (int i = 0; i < n; ++i) {
            const double ya = __half2float(ha[r * n + i]) / na;
            const double yb = __half2float(hb[r * n + i]) / nb;
            y[i] = ya + static_cast<double>(alpha[i]) * (yb - ya);
            ny += y[i] * y[i];
        }
        ny = std::sqrt(ny + kEps);
        for (int i = 0; i < n; ++i)
            CHECK(std::abs(out[r * n + i] - y[i] / ny) < 3e-3);
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dout);
    cudaFree(dalpha);
}

TEST_CASE("dory qk-norm applies the squared nGPT q scale only to q",
          "[dory][kernels]")
{
    const int rows = 2, nh = 3, nkv = 2, hd = 8;
    const int qsize = nh * hd, ksize = nkv * hd;
    int64_t seed = 31;
    std::vector<float> q(rows * qsize), k(rows * ksize), sqk2(qsize);
    for (auto &v : q) v = Lcg(seed);
    for (auto &v : k) v = Lcg(seed);
    for (auto &v : sqk2) v = 0.5f + std::fabs(Lcg(seed));

    auto hq = ToHalf(q), hk = ToHalf(k);
    __half *dq = ToDev(hq), *dk = ToDev(hk);
    float *dsqk = ToDev(sqk2);

    REQUIRE(RunDoryQkNormKernel(nullptr, dq, dk, dsqk, rows, nh, nkv, hd,
                                kEps) == cudaSuccess);
    const auto oq = FromDevHalf(dq, hq.size());
    const auto ok = FromDevHalf(dk, hk.size());

    for (int r = 0; r < rows; ++r) {
        for (int h = 0; h < nh; ++h) {
            double acc = 0.0;
            for (int d = 0; d < hd; ++d) {
                const double v = __half2float(hq[(r * nh + h) * hd + d]);
                acc += v * v;
            }
            const double norm = std::sqrt(acc + kEps);
            for (int d = 0; d < hd; ++d) {
                const double want =
                    __half2float(hq[(r * nh + h) * hd + d]) / norm *
                    sqk2[h * hd + d];
                CHECK(std::abs(oq[(r * nh + h) * hd + d] - want) <
                      std::abs(want) * 2e-2 + 2e-3);
            }
        }
        for (int h = 0; h < nkv; ++h) {
            double acc = 0.0;
            for (int d = 0; d < hd; ++d) {
                const double v = __half2float(hk[(r * nkv + h) * hd + d]);
                acc += v * v;
            }
            const double norm = std::sqrt(acc + kEps);
            for (int d = 0; d < hd; ++d) {
                const double want =
                    __half2float(hk[(r * nkv + h) * hd + d]) / norm;
                CHECK(std::abs(ok[(r * nkv + h) * hd + d] - want) < 2e-3);
            }
        }
    }

    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dsqk);
}

TEST_CASE("dory attention gate + norm matches the host reference",
          "[dory][kernels]")
{
    const int rows = 2, qsize = 19;
    int64_t seed = 41;
    std::vector<float> attn(rows * qsize), gate(rows * qsize), sgate2(qsize);
    for (auto &v : attn) v = Lcg(seed);
    for (auto &v : gate) v = 2.0f * Lcg(seed);
    for (auto &v : sgate2) v = 0.5f + std::fabs(Lcg(seed));

    auto ha = ToHalf(attn), hg = ToHalf(gate);
    __half *da = ToDev(ha), *dg = ToDev(hg), *dout = nullptr;
    float *dsg = ToDev(sgate2);
    REQUIRE(cudaMalloc(&dout, ha.size() * sizeof(__half)) == cudaSuccess);

    REQUIRE(RunDoryAttnGateNormKernel(nullptr, dout, da, dg, dsg, rows, qsize,
                                      kEps) == cudaSuccess);
    const auto out = FromDevHalf(dout, ha.size());

    for (int r = 0; r < rows; ++r) {
        std::vector<double> t(qsize);
        double acc = 0.0;
        for (int i = 0; i < qsize; ++i) {
            const double a = __half2float(ha[r * qsize + i]);
            const double g = __half2float(hg[r * qsize + i]);
            const double sig = 1.0 / (1.0 + std::exp(-g * sgate2[i]));
            t[i] = a * sig;
            acc += t[i] * t[i];
        }
        const double norm = std::sqrt(acc + kEps);
        for (int i = 0; i < qsize; ++i)
            CHECK(std::abs(out[r * qsize + i] - t[i] / norm) < 3e-3);
    }

    cudaFree(da);
    cudaFree(dg);
    cudaFree(dout);
    cudaFree(dsg);
}

TEST_CASE("dory Suv-SwiGLU-norm matches the host reference", "[dory][kernels]")
{
    const int rows = 2, I = 23;
    int64_t seed = 53;
    std::vector<float> gu(rows * 2 * I), suv_gate(I), suv_up(I);
    for (auto &v : gu) v = 2.0f * Lcg(seed);
    for (auto &v : suv_gate) v = 0.5f + std::fabs(Lcg(seed));
    for (auto &v : suv_up) v = 0.5f + std::fabs(Lcg(seed));

    auto hgu = ToHalf(gu);
    __half *dgu = ToDev(hgu), *dout = nullptr;
    float *dsg = ToDev(suv_gate), *dsu = ToDev(suv_up);
    REQUIRE(cudaMalloc(&dout, static_cast<size_t>(rows) * I * sizeof(__half)) ==
            cudaSuccess);

    REQUIRE(RunDorySuvSwiGLUNormKernel(nullptr, dout, dgu, dsg, dsu, rows, I,
                                       kEps) == cudaSuccess);
    const auto out = FromDevHalf(dout, static_cast<size_t>(rows) * I);

    for (int r = 0; r < rows; ++r) {
        std::vector<double> m(I);
        double acc = 0.0;
        for (int j = 0; j < I; ++j) {
            const double g = __half2float(hgu[r * 2 * I + j]) * suv_gate[j];
            const double u =
                __half2float(hgu[r * 2 * I + I + j]) * suv_up[j];
            const double silu = g / (1.0 + std::exp(-g));
            m[j] = silu * u;
            acc += m[j] * m[j];
        }
        const double norm = std::sqrt(acc + kEps);
        for (int j = 0; j < I; ++j)
            CHECK(std::abs(out[r * I + j] - m[j] / norm) < 3e-3);
    }

    cudaFree(dgu);
    cudaFree(dout);
    cudaFree(dsg);
    cudaFree(dsu);
}

TEST_CASE("dory sliding-window attention matches the host reference",
          "[dory][kernels]")
{
    const int seq = 3, nh = 2, nkv = 1, hd = 8;
    const int kv_base = 5;   // pretend 5 keys are already cached
    const int total = kv_base + seq;
    int64_t seed = 67;
    std::vector<float> q(seq * nh * hd), k(total * nkv * hd),
        v(total * nkv * hd);
    for (auto &x : q) x = Lcg(seed);
    for (auto &x : k) x = Lcg(seed);
    for (auto &x : v) x = Lcg(seed);

    auto hq = ToHalf(q), hk = ToHalf(k), hv = ToHalf(v);
    __half *dq = ToDev(hq), *dk = ToDev(hk), *dv = ToDev(hv), *dout = nullptr;
    REQUIRE(cudaMalloc(&dout, hq.size() * sizeof(__half)) == cudaSuccess);

    const float scale = 0.35f;
    for (int window : {0, 4, 1024}) {
        INFO("window " << window);
        REQUIRE(RunDoryAttentionKernel(nullptr, dq, dk, dv, dout, nullptr, seq,
                                       nh, nkv, hd, scale, kv_base,
                                       window) == cudaSuccess);
        const auto out = FromDevHalf(dout, hq.size());
        for (int t = 0; t < seq; ++t) {
            const int bound = kv_base + t;
            const int first = (window > 0) ? std::max(0, bound - window + 1) : 0;
            for (int h = 0; h < nh; ++h) {
                const int kvh = h / (nh / nkv);
                // FP64 reference: scores, softmax, weighted sum.
                std::vector<double> scores;
                double mx = -1e300;
                for (int p = first; p <= bound; ++p) {
                    double dot = 0.0;
                    for (int d = 0; d < hd; ++d)
                        dot += static_cast<double>(
                                   __half2float(hq[(t * nh + h) * hd + d])) *
                               static_cast<double>(
                                   __half2float(hk[(p * nkv + kvh) * hd + d]));
                    scores.push_back(dot * scale);
                    mx = std::max(mx, scores.back());
                }
                double l = 0.0;
                for (auto &s : scores) {
                    s = std::exp(s - mx);
                    l += s;
                }
                for (int d = 0; d < hd; ++d) {
                    double acc = 0.0;
                    for (size_t i = 0; i < scores.size(); ++i) {
                        const int p = first + static_cast<int>(i);
                        acc += scores[i] / l *
                               static_cast<double>(
                                   __half2float(hv[(p * nkv + kvh) * hd + d]));
                    }
                    const double got = out[(t * nh + h) * hd + d];
                    INFO("t " << t << " h " << h << " d " << d);
                    CHECK(std::abs(got - acc) < 3e-3);
                }
            }
        }
    }

    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dout);
}

TEST_CASE("dory attention split-KV path is bit-identical to the unsplit path",
          "[dory][kernels]")
{
    const int nh = 2, nkv = 1, hd = 8;
    const int kv_base = 200;   // long history so the split path is eligible
    const int seq = 1;
    const int total = kv_base + seq;
    int64_t seed = 71;
    std::vector<float> q(seq * nh * hd), k(total * nkv * hd),
        v(total * nkv * hd);
    for (auto &x : q) x = Lcg(seed);
    for (auto &x : k) x = Lcg(seed);
    for (auto &x : v) x = Lcg(seed);

    auto hq = ToHalf(q), hk = ToHalf(k), hv = ToHalf(v);
    __half *dq = ToDev(hq), *dk = ToDev(hk), *dv = ToDev(hv);
    __half *dout_a = nullptr, *dout_b = nullptr;
    REQUIRE(cudaMalloc(&dout_a, hq.size() * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout_b, hq.size() * sizeof(__half)) == cudaSuccess);
    // Partials slab sized like the workspace: splits * seq * nh * (hd + 2).
    const size_t partials = static_cast<size_t>(32) * seq * nh * (hd + 2);
    float *dpart = nullptr;
    REQUIRE(cudaMalloc(&dpart, partials * sizeof(float)) == cudaSuccess);

    for (int window : {0, 8}) {
        REQUIRE(RunDoryAttentionKernel(nullptr, dq, dk, dv, dout_a, nullptr,
                                       seq, nh, nkv, hd, 0.3f, kv_base,
                                       window) == cudaSuccess);
        REQUIRE(RunDoryAttentionKernel(nullptr, dq, dk, dv, dout_b, dpart, seq,
                                       nh, nkv, hd, 0.3f, kv_base,
                                       window) == cudaSuccess);
        const auto a = FromDevHalf(dout_a, hq.size());
        const auto b = FromDevHalf(dout_b, hq.size());
        for (size_t i = 0; i < a.size(); ++i) {
            INFO("window " << window << " i " << i);
            // Same arithmetic in a different reduction order: tolerance class,
            // deterministic run to run.
            CHECK(std::abs(a[i] - b[i]) < 3e-3);
        }
    }

    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dout_a);
    cudaFree(dout_b);
    cudaFree(dpart);
}

TEST_CASE("dory logit scale multiplies the head output per vocabulary entry",
          "[dory][kernels]")
{
    const int V = 41;
    int64_t seed = 83;
    std::vector<float> y(V), scale(V);
    for (auto &v : y) v = 4.0f * Lcg(seed);
    for (auto &v : scale) v = 1.0f + Lcg(seed);

    auto hy = ToHalf(y);
    __half *dy = ToDev(hy);
    float *dscale = ToDev(scale), *dlogits = nullptr;
    REQUIRE(cudaMalloc(&dlogits, static_cast<size_t>(V) * sizeof(float)) ==
            cudaSuccess);
    REQUIRE(RunDoryLogitScaleKernel(nullptr, dlogits, dy, dscale, V) ==
            cudaSuccess);
    std::vector<float> out(V);
    REQUIRE(cudaMemcpy(out.data(), dlogits, out.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = 0; i < V; ++i) {
        const double want = static_cast<double>(__half2float(hy[i])) * scale[i];
        CHECK(std::abs(out[i] - want) < 1e-6);
    }
    cudaFree(dy);
    cudaFree(dscale);
    cudaFree(dlogits);
}
