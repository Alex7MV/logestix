// Step-5-Preview numeric kernels against host references (Model Ingestion
// Protocol stage 5/6): FP32-residual RMSNorm with the reference BF16 narrowing
// boundary, per-head QK norm, partial NEOX RoPE, the sigmoid + selection-only
// bias router, clamped SwiGLU, the windowed GQA decode attention and the FP32
// residual combiner.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "lsxcommon/step_moe_kernels.h"

using namespace lsxcommon::step;

namespace {

__half HF(float v) { return __float2half_rn(v); }
float FH(__half v) { return __half2float(v); }
__nv_bfloat16 BF(float v) { return __float2bfloat16_rn(v); }
float FB(__nv_bfloat16 v) { return __bfloat162float(v); }

struct DevBuf {
    void *p = nullptr;
    explicit DevBuf(size_t bytes) { REQUIRE(cudaMalloc(&p, bytes) == cudaSuccess); }
    ~DevBuf() { if (p) cudaFree(p); }
    template <typename T> T *as() { return static_cast<T *>(p); }
};

template <typename T>
void Up(T *dst, const std::vector<T> &src)
{
    REQUIRE(cudaMemcpy(dst, src.data(), src.size() * sizeof(T),
                       cudaMemcpyHostToDevice) == cudaSuccess);
}

template <typename T>
void Down(std::vector<T> &dst, const T *src)
{
    REQUIRE(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
}

float RefSilu(float g) { return g / (1.0f + std::exp(-g)); }

} // namespace

TEST_CASE("Step-5 clamped SwiGLU matches the reference", "[step-kernels]")
{
    const int n = 257;
    const float limit = 7.0f;
    std::vector<__half> g(n), u(n), out(n), ref(n);
    for (int i = 0; i < n; ++i) {
        const float gv = static_cast<float>((i % 37) - 18) * 1.3f;
        const float uv = static_cast<float>((i % 23) - 11) * 2.1f;
        g[i] = HF(gv);
        u[i] = HF(uv);
        const float silu = RefSilu(FH(g[i]));
        const float a = std::fmin(silu, limit);
        const float b = std::fmin(std::fmax(FH(u[i]), -limit), limit);
        ref[i] = HF(a * b);
    }
    DevBuf dg(n * sizeof(__half)), du(n * sizeof(__half)),
        dout(n * sizeof(__half));
    Up(dg.as<__half>(), g);
    Up(du.as<__half>(), u);
    REQUIRE(RunStepSwigluClamp(nullptr, dg.as<__half>(), du.as<__half>(),
                               dout.as<__half>(), n, limit) == cudaSuccess);
    Down(out, dout.as<__half>());
    for (int i = 0; i < n; ++i) CHECK(FH(out[i]) == Catch::Approx(FH(ref[i])));
}

TEST_CASE("Step-5 router is sigmoid + selection-only bias", "[step-kernels]")
{
    const int E = 24, K = 5, H = 64;
    const float scale = 3.0f;
    std::vector<__nv_bfloat16> gate(static_cast<size_t>(E) * H);
    std::vector<float> bias(E);
    std::vector<__half> x(H);
    for (size_t i = 0; i < gate.size(); ++i)
        gate[i] = BF(static_cast<float>((i * 13) % 29) * 0.017f - 0.24f);
    for (int e = 0; e < E; ++e) bias[e] = static_cast<float>(e % 7) * 0.05f - 0.1f;
    for (int k = 0; k < H; ++k)
        x[k] = HF(static_cast<float>((k * 7) % 19) * 0.03f - 0.3f);

    // Host reference.
    std::vector<float> score(E);
    for (int e = 0; e < E; ++e) {
        float acc = 0.0f;
        for (int k = 0; k < H; ++k) acc += FB(gate[static_cast<size_t>(e) * H + k]) * FH(x[k]);
        score[e] = 1.0f / (1.0f + std::exp(-acc)) + bias[e];
    }
    std::vector<int> sel;
    std::vector<char> used(E, 0);
    for (int i = 0; i < K; ++i) {
        int best = -1;
        float bestv = -3.4e38f;
        for (int e = 0; e < E; ++e)
            if (!used[e] && score[e] > bestv) { bestv = score[e]; best = e; }
        used[best] = 1;
        sel.push_back(best);
    }
    float denom = 0.0f;
    for (int e : sel) denom += score[e] - bias[e];
    std::vector<float> ref_coeff(sel.size());
    for (size_t i = 0; i < sel.size(); ++i)
        ref_coeff[i] = (score[sel[i]] - bias[sel[i]]) / (denom + 1e-20f) * scale;

    DevBuf dg(gate.size() * sizeof(__nv_bfloat16)), db(E * sizeof(float)),
        dx(H * sizeof(__half)), ds(K * sizeof(StepRoutedSlot)),
        di(K * sizeof(int32_t));
    Up(dg.as<__nv_bfloat16>(), gate);
    Up(db.as<float>(), bias);
    Up(dx.as<__half>(), x);
    REQUIRE(RunStepRouter(nullptr, dg.as<__nv_bfloat16>(), db.as<float>(),
                          dx.as<__half>(), E, K, H, scale,
                          ds.as<StepRoutedSlot>(), di.as<int32_t>()) ==
            cudaSuccess);
    std::vector<StepRoutedSlot> slots(K);
    std::vector<int32_t> ids(K);
    Down(slots, ds.as<StepRoutedSlot>());
    Down(ids, di.as<int32_t>());
    for (int i = 0; i < K; ++i) {
        CHECK(slots[static_cast<size_t>(i)].eid == sel[static_cast<size_t>(i)]);
        CHECK(ids[static_cast<size_t>(i)] == sel[static_cast<size_t>(i)]);
        CHECK(slots[static_cast<size_t>(i)].coeff ==
              Catch::Approx(ref_coeff[static_cast<size_t>(i)]).epsilon(1e-5));
    }
}

TEST_CASE("Step-5 zero-centered norm narrows to BF16 before the final RMSNorm",
          "[step-kernels]")
{
    const int H = 128;
    std::vector<float> x(H), w(H);
    for (int i = 0; i < H; ++i) {
        x[i] = static_cast<float>((i % 17) - 8) * 0.37f;
        w[i] = 0.5f + static_cast<float>(i % 5) * 0.1f;
    }
    const float eps = 1e-5f;
    double ss = 0.0;
    std::vector<float> narrowed(H);
    for (int i = 0; i < H; ++i) {
        narrowed[i] = FB(BF(x[i]));
        ss += static_cast<double>(narrowed[i]) * narrowed[i];
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / H) + eps);
    std::vector<__nv_bfloat16> ref(H);
    for (int i = 0; i < H; ++i) ref[i] = BF(narrowed[i] * inv * (w[i] + 1.0f));

    DevBuf dx(H * sizeof(float)), dw(H * sizeof(float)),
        dout(H * sizeof(__nv_bfloat16));
    Up(dx.as<float>(), x);
    Up(dw.as<float>(), w);
    REQUIRE(RunStepNormF32ToBf16Narrowed(nullptr, dx.as<float>(), dw.as<float>(),
                                         dout.as<__nv_bfloat16>(), H, eps) ==
            cudaSuccess);
    std::vector<__nv_bfloat16> out(H);
    Down(out, dout.as<__nv_bfloat16>());
    for (int i = 0; i < H; ++i) CHECK(FB(out[i]) == Catch::Approx(FB(ref[i])));
}

TEST_CASE("Step-5 per-head QK norm normalizes each head independently",
          "[step-kernels]")
{
    const int heads = 4, HD = 32;
    std::vector<__half> x(heads * HD), ref(heads * HD);
    std::vector<float> w(HD);
    for (int d = 0; d < HD; ++d) w[d] = 0.75f + static_cast<float>(d % 3) * 0.1f;
    const float eps = 1e-5f;
    for (int h = 0; h < heads; ++h) {
        double ss = 0.0;
        for (int d = 0; d < HD; ++d) {
            x[static_cast<size_t>(h) * HD + d] =
                HF(static_cast<float>((h * 7 + d * 3) % 15) * 0.11f - 0.7f);
            ss += static_cast<double>(FH(x[static_cast<size_t>(h) * HD + d])) *
                  FH(x[static_cast<size_t>(h) * HD + d]);
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(ss / HD) + eps);
        for (int d = 0; d < HD; ++d)
            ref[static_cast<size_t>(h) * HD + d] =
                HF(FH(x[static_cast<size_t>(h) * HD + d]) * inv * (w[d] + 1.0f));
    }
    DevBuf dx(x.size() * sizeof(__half)), dw(HD * sizeof(float)),
        dout(x.size() * sizeof(__half));
    Up(dx.as<__half>(), x);
    Up(dw.as<float>(), w);
    REQUIRE(RunStepHeadNorm(nullptr, dx.as<__half>(), dw.as<float>(),
                            dout.as<__half>(), heads, HD, eps) == cudaSuccess);
    std::vector<__half> out(x.size());
    Down(out, dout.as<__half>());
    for (size_t i = 0; i < x.size(); ++i)
        CHECK(FH(out[i]) == Catch::Approx(FH(ref[i])).epsilon(1e-5));
}

TEST_CASE("Step-5 partial NEOX RoPE rotates only the leading span", "[step-kernels]")
{
    const int heads = 2, HD = 32, span = 16;
    const float theta = 10000.0f;
    const int pos = 5;
    std::vector<__half> q(heads * HD), ref(heads * HD);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = HF(static_cast<float>((i * 5) % 13) * 0.07f - 0.4f);
    ref = q;
    const int half = span / 2;
    for (int h = 0; h < heads; ++h) {
        for (int j = 0; j < half; ++j) {
            const float inv =
                1.0f / std::pow(theta, 2.0f * static_cast<float>(j) /
                                           static_cast<float>(span));
            const float ang = static_cast<float>(pos) * inv;
            const float c = std::cos(ang), s = std::sin(ang);
            const size_t a = static_cast<size_t>(h) * HD + j;
            const size_t b = a + half;
            const float av = FH(q[a]), bv = FH(q[b]);
            ref[a] = HF(av * c - bv * s);
            ref[b] = HF(av * s + bv * c);
        }
    }
    DevBuf dq(q.size() * sizeof(__half));
    Up(dq.as<__half>(), q);
    REQUIRE(RunStepPartialRope(nullptr, dq.as<__half>(), heads, nullptr, 0, HD,
                               span, theta, pos) == cudaSuccess);
    std::vector<__half> out(q.size());
    Down(out, dq.as<__half>());
    for (size_t i = 0; i < out.size(); ++i)
        CHECK(FH(out[i]) == Catch::Approx(FH(ref[i])).epsilon(2e-3));
}

TEST_CASE("Step-5 windowed decode attention matches a host softmax",
          "[step-kernels]")
{
    const int heads = 4, kv_heads = 2, HD = 32, pos = 6, window = 3;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    std::vector<__half> q(heads * HD), k((pos + 1) * kv_heads * HD),
        v((pos + 1) * kv_heads * HD), out(heads * HD), ref(heads * HD);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = HF(static_cast<float>((i * 3) % 11) * 0.05f - 0.25f);
    for (size_t i = 0; i < k.size(); ++i)
        k[i] = HF(static_cast<float>((i * 7) % 17) * 0.04f - 0.3f);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = HF(static_cast<float>((i * 5) % 19) * 0.03f - 0.25f);

    const int hpg = heads / kv_heads;
    const int lo = std::max(0, pos + 1 - window);
    for (int h = 0; h < heads; ++h) {
        const int kv = h / hpg;
        std::vector<float> score;
        float mx = -3.4e38f;
        for (int s = lo; s <= pos; ++s) {
            float dot = 0.0f;
            for (int d = 0; d < HD; ++d)
                dot += FH(q[static_cast<size_t>(h) * HD + d]) *
                       FH(k[(static_cast<size_t>(s) * kv_heads + kv) * HD + d]);
            score.push_back(dot * scale);
            mx = std::max(mx, score.back());
        }
        float denom = 0.0f;
        for (float sc : score) denom += std::exp(sc - mx);
        for (int d = 0; d < HD; ++d) {
            float acc = 0.0f;
            for (size_t i = 0; i < score.size(); ++i) {
                const int s = lo + static_cast<int>(i);
                acc += std::exp(score[i] - mx) *
                       FH(v[(static_cast<size_t>(s) * kv_heads + kv) * HD + d]);
            }
            ref[static_cast<size_t>(h) * HD + d] = HF(acc / denom);
        }
    }

    DevBuf dq(q.size() * sizeof(__half)), dk(k.size() * sizeof(__half)),
        dv(v.size() * sizeof(__half)), dout(out.size() * sizeof(__half));
    Up(dq.as<__half>(), q);
    Up(dk.as<__half>(), k);
    Up(dv.as<__half>(), v);
    REQUIRE(RunStepWindowedAttnDecode(nullptr, dq.as<__half>(), dk.as<__half>(),
                                      dv.as<__half>(), dout.as<__half>(), pos,
                                      heads, kv_heads, HD, scale,
                                      window) == cudaSuccess);
    Down(out, dout.as<__half>());
    for (size_t i = 0; i < out.size(); ++i)
        CHECK(FH(out[i]) == Catch::Approx(FH(ref[i])).epsilon(2e-3));
}

TEST_CASE("Step-5 per-head gate and FP32 residual combine", "[step-kernels]")
{
    const int heads = 3, HD = 32;
    std::vector<__half> a(heads * HD), g(heads), ref(heads * HD);
    for (size_t i = 0; i < a.size(); ++i) a[i] = HF(static_cast<float>(i) * 0.01f);
    for (int h = 0; h < heads; ++h) g[h] = HF(static_cast<float>(h) - 1.0f);
    ref = a;
    for (int h = 0; h < heads; ++h) {
        const float gate = 1.0f / (1.0f + std::exp(-FH(g[h])));
        for (int d = 0; d < HD; ++d) {
            const size_t i = static_cast<size_t>(h) * HD + d;
            ref[i] = HF(FH(a[i]) * gate);
        }
    }
    DevBuf da(a.size() * sizeof(__half)), dg(heads * sizeof(__half));
    Up(da.as<__half>(), a);
    Up(dg.as<__half>(), g);
    REQUIRE(RunStepApplyHeadGate(nullptr, da.as<__half>(), dg.as<__half>(),
                                 heads, HD) == cudaSuccess);
    std::vector<__half> out(a.size());
    Down(out, da.as<__half>());
    for (size_t i = 0; i < out.size(); ++i)
        CHECK(FH(out[i]) == Catch::Approx(FH(ref[i])).epsilon(1e-5));

    // Combine: residual += bf16(routed) + shared.
    const int n = 40;
    std::vector<float> routed(n), resid(n), rref(n);
    std::vector<__half> shared(n);
    for (int i = 0; i < n; ++i) {
        routed[i] = static_cast<float>(i) * 0.37f - 3.0f;
        shared[i] = HF(static_cast<float>(i) * 0.013f);
        resid[i] = 1.0f;
        rref[i] = resid[i] + FB(BF(routed[i])) + FH(shared[i]);
    }
    DevBuf dr(n * sizeof(float)), dsh(n * sizeof(__half)), dres(n * sizeof(float));
    Up(dr.as<float>(), routed);
    Up(dsh.as<__half>(), shared);
    Up(dres.as<float>(), resid);
    REQUIRE(RunStepCombineF32(nullptr, dr.as<float>(), dsh.as<__half>(),
                              dres.as<float>(), n) == cudaSuccess);
    std::vector<float> res(n);
    Down(res, dres.as<float>());
    for (int i = 0; i < n; ++i) CHECK(res[i] == Catch::Approx(rref[i]));
}
