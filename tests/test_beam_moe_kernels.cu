// Beam-1.0 kernel stage: every device kernel against the host reference in
// beam_moe_device_math.h (label fast, GPU resource lock). Shapes are tiny; the
// point is the math contract, not throughput.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/beam_moe_device_math.h"
#include "lsxcommon/beam_moe_kernels.h"

#include "beam_moe_constants.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace lsxcommon::beam;
using namespace lsxcommon::beam::frozen;
using Catch::Approx;

namespace {

bool HasCuda()
{
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

struct Dev {
    template <typename T>
    T *alloc(const std::vector<T> &h)
    {
        T *d = nullptr;
        REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d),
                           h.size() * sizeof(T)) == cudaSuccess);
        REQUIRE(cudaMemcpy(d, h.data(), h.size() * sizeof(T),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        return d;
    }
    template <typename T>
    T *alloc(size_t n)
    {
        T *d = nullptr;
        REQUIRE(cudaMalloc(reinterpret_cast<void **>(&d), n * sizeof(T)) ==
                cudaSuccess);
        return d;
    }
    template <typename T>
    std::vector<T> get(const T *d, size_t n)
    {
        std::vector<T> h(n);
        REQUIRE(cudaMemcpy(h.data(), d, n * sizeof(T),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        return h;
    }
    ~Dev()
    {
        for (void *p : owned) cudaFree(p);
    }
    std::vector<void *> owned;
};

float F(__half h) { return __half2float(h); }

} // namespace

TEST_CASE("beam RMSNorm matches the host reference", "[beam][kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int rows = 2, hidden = 64;
    std::vector<__half> x(rows * hidden), w(hidden), out(rows * hidden);
    std::vector<float> xf(rows * hidden), wf(hidden), rf(rows * hidden);
    for (int i = 0; i < rows * hidden; ++i) {
        xf[i] = std::sin(0.17f * i) * 2.0f;
        x[i] = __float2half(xf[i]);
        xf[i] = F(x[i]);
    }
    for (int i = 0; i < hidden; ++i) {
        wf[i] = 1.0f + 0.01f * i;
        w[i] = __float2half(wf[i]);
        wf[i] = F(w[i]);
    }
    for (int r = 0; r < rows; ++r)
        ref::RmsNorm(xf.data() + r * hidden, wf.data(), rf.data() + r * hidden,
                     hidden, kRmsEps);
    Dev dev;
    auto *dx = dev.alloc(x);
    auto *dw = dev.alloc(w);
    auto *dout = dev.alloc(out);
    REQUIRE(RunBeamNorm(nullptr, dx, dw, dout, rows, hidden, kRmsEps) ==
            cudaSuccess);
    cudaDeviceSynchronize();
    out = dev.get(dout, out.size());
    for (size_t i = 0; i < out.size(); ++i)
        CHECK(F(out[i]) == Approx(rf[i]).margin(2e-3));
}

TEST_CASE("beam per-head QK norm matches the host reference", "[beam][kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int rows = 1, heads = 4, hd = 32;
    std::vector<__half> x(rows * heads * hd), w(hd), out(rows * heads * hd);
    std::vector<float> xf(x.size()), wf(hd), rf(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        xf[i] = std::cos(0.11f * i);
        x[i] = __float2half(xf[i]);
        xf[i] = F(x[i]);
    }
    for (int i = 0; i < hd; ++i) {
        wf[i] = 0.9f + 0.02f * i;
        w[i] = __float2half(wf[i]);
        wf[i] = F(w[i]);
    }
    ref::HeadNorm(xf.data(), wf.data(), rf.data(), rows, heads, hd, kRmsEps);
    Dev dev;
    auto *dx = dev.alloc(x);
    auto *dw = dev.alloc(w);
    auto *dout = dev.alloc(out);
    REQUIRE(RunBeamHeadNorm(nullptr, dx, dw, dout, rows, heads, hd, kRmsEps) ==
            cudaSuccess);
    cudaDeviceSynchronize();
    out = dev.get(dout, out.size());
    for (size_t i = 0; i < out.size(); ++i)
        CHECK(F(out[i]) == Approx(rf[i]).margin(2e-3));
}

TEST_CASE("beam RoPE and SSMax match the host reference", "[beam][kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int qh = 2, kh = 1, hd = 32, pos = 37;
    std::vector<__half> q(qh * hd), k(kh * hd);
    std::vector<float> qf(q.size()), kf(k.size());
    for (size_t i = 0; i < q.size(); ++i) {
        qf[i] = std::sin(0.3f * i);
        q[i] = __float2half(qf[i]);
        qf[i] = F(q[i]);
    }
    for (size_t i = 0; i < k.size(); ++i) {
        kf[i] = std::cos(0.2f * i);
        k[i] = __float2half(kf[i]);
        kf[i] = F(k[i]);
    }
    ref::Rope(qf.data(), qh, hd, pos, kRopeTheta);
    ref::Rope(kf.data(), kh, hd, pos, kRopeTheta);
    Dev dev;
    auto *dq = dev.alloc(q);
    auto *dk = dev.alloc(k);
    REQUIRE(RunBeamRope(nullptr, dq, dk, 1, qh, kh, hd, kRopeTheta, pos) ==
            cudaSuccess);
    cudaDeviceSynchronize();
    q = dev.get(dq, q.size());
    k = dev.get(dk, k.size());
    for (size_t i = 0; i < q.size(); ++i)
        CHECK(F(q[i]) == Approx(qf[i]).margin(2e-3));
    for (size_t i = 0; i < k.size(); ++i)
        CHECK(F(k[i]) == Approx(kf[i]).margin(2e-3));

    // NoPE full-attention path: SSMax scales q in FP32.
    std::vector<float> before(q.size());
    for (size_t i = 0; i < q.size(); ++i) before[i] = F(q[i]);
    ref::SsmScaleQ(before.data(), qh, hd, 1500);
    REQUIRE(RunBeamSsmScale(nullptr, dq, qh, hd, 1500) == cudaSuccess);
    cudaDeviceSynchronize();
    q = dev.get(dq, q.size());
    for (size_t i = 0; i < q.size(); ++i)
        CHECK(F(q[i]) == Approx(before[i]).margin(3e-3));
}

TEST_CASE("beam depth-scaled residual and SwiGLU match the reference",
          "[beam][kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int n = 32;
    std::vector<__half> b(n), r(n), out(n);
    std::vector<float> bf(n), rf(n), of(n);
    for (int i = 0; i < n; ++i) {
        bf[i] = std::sin(0.7f * i);
        rf[i] = std::cos(0.4f * i);
        b[i] = __float2half(bf[i]);
        r[i] = __float2half(rf[i]);
        bf[i] = F(b[i]);
        rf[i] = F(r[i]);
    }
    const int idx = 7;
    ref::DepthResidual(bf.data(), rf.data(), of.data(), n, idx);
    Dev dev;
    auto *db = dev.alloc(b);
    auto *dr = dev.alloc(r);
    auto *dout = dev.alloc(out);
    REQUIRE(RunBeamDepthResidual(nullptr, db, dr, dout, n, idx) == cudaSuccess);
    cudaDeviceSynchronize();
    out = dev.get(dout, out.size());
    for (int i = 0; i < n; ++i)
        CHECK(F(out[i]) == Approx(of[i]).margin(3e-3));

    std::vector<__half> up(n), sw(n);
    std::vector<float> upf(n);
    for (int i = 0; i < n; ++i) {
        upf[i] = 0.5f + 0.1f * i;
        up[i] = __float2half(upf[i]);
        upf[i] = F(up[i]);
    }
    auto *dup = dev.alloc(up);
    auto *dsw = dev.alloc<__half>(static_cast<size_t>(n));
    REQUIRE(RunBeamSwiGLU(nullptr, db, dup, dsw, n) == cudaSuccess);
    cudaDeviceSynchronize();
    sw = dev.get(dsw, static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float g = bf[i];
        const float want = (g / (1.0f + std::exp(-g))) * upf[i];
        CHECK(F(sw[i]) == Approx(want).margin(3e-3));
    }
}

TEST_CASE("beam group-limited sigmoid router matches the reference",
          "[beam][kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int E = 8, K = 2, H = 32, G = 2, TG = 1;
    std::vector<float> w(E * H), bias(E);
    std::vector<__half> x(H);
    std::vector<float> xf(H), logits(E);
    for (int e = 0; e < E; ++e) {
        for (int i = 0; i < H; ++i)
            w[static_cast<size_t>(e) * H + i] = 0.05f * ((e * 3 + i) % 11 - 5);
        bias[e] = 0.01f * (e - 3);
    }
    for (int i = 0; i < H; ++i) {
        xf[i] = std::sin(0.3f * i);
        x[i] = __float2half(xf[i]);
        xf[i] = F(x[i]);
    }
    for (int e = 0; e < E; ++e) {
        float acc = 0.0f;
        for (int i = 0; i < H; ++i)
            acc += w[static_cast<size_t>(e) * H + i] * xf[i];
        logits[e] = acc;
    }
    const auto want = ref::Router(logits.data(), bias.data(), E, K, G, TG,
                                  kRoutedScale);

    Dev dev;
    auto *dw = dev.alloc(w);
    auto *dbias = dev.alloc(bias);
    auto *dx = dev.alloc(x);
    auto *dslots = dev.alloc<BeamRoutedSlot>(K);
    REQUIRE(RunBeamRouter(nullptr, dw, dbias, dx, 1, E, K, H, G, TG,
                          kRoutedScale, dslots) == cudaSuccess);
    cudaDeviceSynchronize();
    const auto got = dev.get(dslots, static_cast<size_t>(K));
    REQUIRE(want.ids.size() == static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        CHECK(got[k].eid == want.ids[static_cast<size_t>(k)]);
        CHECK(got[k].coeff == Approx(want.weights[static_cast<size_t>(k)])
                                  .margin(2e-4));
    }
}

TEST_CASE("beam router handles the real 256/8/8/4 register shape",
          "[beam][kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int E = frozen::kNumExperts, K = frozen::kTopK, H = 64;
    const int G = frozen::kNGroups, TG = frozen::kTopkGroup;
    std::vector<float> w(static_cast<size_t>(E) * H), bias(E);
    std::vector<__half> x(H);
    std::vector<float> xf(H), logits(E);
    for (int e = 0; e < E; ++e) {
        for (int i = 0; i < H; ++i)
            w[static_cast<size_t>(e) * H + i] =
                0.02f * static_cast<float>(((e * 7 + i * 5) % 23) - 11);
        bias[e] = 0.005f * static_cast<float>((e % 9) - 4);
    }
    for (int i = 0; i < H; ++i) {
        xf[i] = std::cos(0.21f * i);
        x[i] = __float2half(xf[i]);
        xf[i] = F(x[i]);
    }
    for (int e = 0; e < E; ++e) {
        double acc = 0.0;
        for (int i = 0; i < H; ++i)
            acc += static_cast<double>(w[static_cast<size_t>(e) * H + i]) * xf[i];
        logits[e] = static_cast<float>(acc);
    }
    const auto want = ref::Router(logits.data(), bias.data(), E, K, G, TG,
                                  kRoutedScale);
    Dev dev;
    auto *dslots = dev.alloc<BeamRoutedSlot>(K);
    REQUIRE(RunBeamRouter(nullptr, dev.alloc(w), dev.alloc(bias), dev.alloc(x),
                          1, E, K, H, G, TG, kRoutedScale, dslots) ==
            cudaSuccess);
    cudaDeviceSynchronize();
    const auto got = dev.get(dslots, static_cast<size_t>(K));
    REQUIRE(want.ids.size() == static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) {
        CHECK(got[k].eid == want.ids[static_cast<size_t>(k)]);
        CHECK(got[k].coeff ==
              Approx(want.weights[static_cast<size_t>(k)]).margin(2e-4));
    }
}

TEST_CASE("beam gated sliding-window decode attention matches the reference",
          "[beam][kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int NH = 4, NKV = 2, HD = 32, pos = 5, window = 3;
    std::vector<__half> q(NH * HD), gate(NH * HD), gb(NH), out(NH * HD);
    std::vector<__half> kc((pos + 1) * NKV * HD), vc((pos + 1) * NKV * HD);
    std::vector<float> qf(q.size()), gf(gate.size()), gbf(NH), of(out.size());
    std::vector<float> kf(kc.size()), vf(vc.size());
    for (size_t i = 0; i < q.size(); ++i) {
        qf[i] = std::sin(0.13f * i);
        q[i] = __float2half(qf[i]);
        qf[i] = F(q[i]);
        gf[i] = std::cos(0.09f * i);
        gate[i] = __float2half(gf[i]);
        gf[i] = F(gate[i]);
    }
    for (int h = 0; h < NH; ++h) {
        gbf[h] = 0.1f * h - 0.2f;
        gb[h] = __float2half(gbf[h]);
        gbf[h] = F(gb[h]);
    }
    for (size_t i = 0; i < kc.size(); ++i) {
        kf[i] = std::sin(0.05f * static_cast<float>(i) + 0.3f);
        vf[i] = std::cos(0.07f * static_cast<float>(i) - 0.2f);
        kc[i] = __float2half(kf[i]);
        vc[i] = __float2half(vf[i]);
        kf[i] = F(kc[i]);
        vf[i] = F(vc[i]);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    const int start = ref::WindowStart(pos, window);
    REQUIRE(start == 3);
    const int group = NH / NKV;
    for (int h = 0; h < NH; ++h) {
        const int kvh = h / group;
        float mx = -INFINITY;
        std::vector<float> l(pos + 1, 0.0f);
        for (int row = start; row <= pos; ++row) {
            double dot = 0.0;
            for (int d = 0; d < HD; ++d)
                dot += static_cast<double>(qf[h * HD + d]) *
                       kf[(static_cast<size_t>(row) * NKV + kvh) * HD + d];
            l[row] = static_cast<float>(dot) * scale;
            mx = std::max(mx, l[row]);
        }
        float sum = 0.0f;
        for (int row = start; row <= pos; ++row) {
            l[row] = std::exp(l[row] - mx);
            sum += l[row];
        }
        for (int d = 0; d < HD; ++d) {
            float acc = 0.0f;
            for (int row = start; row <= pos; ++row)
                acc += l[row] *
                       vf[(static_cast<size_t>(row) * NKV + kvh) * HD + d];
            acc /= sum;
            const float g = gf[h * HD + d] + gbf[h];
            of[h * HD + d] = acc * (1.0f / (1.0f + std::exp(-g)));
        }
    }
    Dev dev;
    auto *dq = dev.alloc(q);
    auto *dk = dev.alloc(kc);
    auto *dv = dev.alloc(vc);
    auto *dg = dev.alloc(gate);
    auto *dgb = dev.alloc(gb);
    auto *dout = dev.alloc(out);
    REQUIRE(RunBeamAttnDecode(nullptr, dq, dk, dv, dg, dgb, dout, NH, NKV, HD,
                              pos, window, scale) == cudaSuccess);
    cudaDeviceSynchronize();
    out = dev.get(dout, out.size());
    for (size_t i = 0; i < out.size(); ++i)
        CHECK(F(out[i]) == Approx(of[i]).margin(4e-3));
}

TEST_CASE("beam softcap matches tanh/logits/30", "[beam][kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    std::vector<float> logits = {-100.0f, -31.0f, 0.0f, 1.5f, 29.0f, 80.0f};
    const auto want = logits;
    Dev dev;
    auto *d = dev.alloc(logits);
    REQUIRE(RunBeamSoftcap(nullptr, d, static_cast<int>(logits.size())) ==
            cudaSuccess);
    cudaDeviceSynchronize();
    const auto got = dev.get(d, logits.size());
    for (size_t i = 0; i < got.size(); ++i)
        CHECK(got[i] == Approx(ref::Softcap(want[i])).margin(1e-5));
}

TEST_CASE("beam embedding lookup", "[beam][kernels]")
{
    if (!HasCuda()) SKIP("no CUDA device");
    const int V = 8, H = 16;
    std::vector<__half> table(V * H), out(H);
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = __float2half(0.01f * static_cast<float>(i));
    std::vector<int32_t> ids = {5};
    Dev dev;
    auto *dt = dev.alloc(table);
    auto *di = dev.alloc(ids);
    auto *dout = dev.alloc(out);
    REQUIRE(RunBeamEmbed(nullptr, dt, di, dout, 1, H) == cudaSuccess);
    cudaDeviceSynchronize();
    out = dev.get(dout, out.size());
    for (int i = 0; i < H; ++i)
        CHECK(F(out[i]) == F(table[5 * H + i]));
}
