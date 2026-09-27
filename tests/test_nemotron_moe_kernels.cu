// TDD — Nemotron-3.5 Mamba-2 kernel tests (Model Ingestion Protocol step 5).
// Device kernels are validated against the host reference in
// nemotron_moe_device_math.h.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "nemotron_moe_device_math.h"
#include "nemotron_moe_kernels.h"

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/safetensors_gpu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace lsxcommon::nemotron_moe;

namespace {

template <typename T>
T *ToDevice(const std::vector<T> &v)
{
    T *d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(T));
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

template <typename T>
std::vector<T> FromDevice(const T *d, size_t n)
{
    std::vector<T> v(n);
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

} // namespace

TEST_CASE("mamba conv step matches the causal depthwise conv", "[nemotron][kernels]")
{
    const int C = 4, K = 4;
    const std::vector<float> xBC = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> w(C * K), bias(C, 0.0f);
    for (int c = 0; c < C; ++c)
        for (int k = 0; k < K; ++k) w[c * K + k] = 0.1f * (c + 1) * (k + 1);
    std::vector<float> state0(C * (K - 1), 0.0f);

    float *d_x = ToDevice(xBC), *d_w = ToDevice(w), *d_b = ToDevice(bias);
    float *d_s0 = ToDevice(state0);
    float *d_s1 = nullptr, *d_o = nullptr;
    cudaMalloc(&d_s1, state0.size() * sizeof(float));
    cudaMalloc(&d_o, C * sizeof(float));

    RunNemotronConvStep(d_x, d_w, d_b, d_s0, d_s1, d_o, C, K, 0);
    cudaDeviceSynchronize();

    auto out = FromDevice(d_o, C);
    auto st1 = FromDevice(d_s1, state0.size());

    for (int c = 0; c < C; ++c) {
        float acc = w[c * K + (K - 1)] * xBC[c];  // history zeros
        const float expect = acc / (1.0f + std::exp(-acc));
        CHECK(out[c] == Catch::Approx(expect).margin(1e-6f));
    }
    // State shifts in the newest input.
    CHECK(st1[(K - 1) - 1] == Catch::Approx(xBC[0]));
    CHECK(st1[2 * (K - 1) + (K - 2)] == Catch::Approx(xBC[2]));

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_s0); cudaFree(d_s1); cudaFree(d_o);
}

TEST_CASE("selective state update matches the host reference", "[nemotron][kernels]")
{
    const int heads = 2, head_dim = 4, groups = 1, state_size = 3;
    std::vector<float> state0(heads * head_dim * state_size, 0.0f);
    for (size_t i = 0; i < state0.size(); ++i) state0[i] = 0.01f * (i % 7);
    std::vector<float> xs(heads * head_dim);
    for (size_t i = 0; i < xs.size(); ++i) xs[i] = 0.1f * (i + 1);
    std::vector<float> B(groups * state_size), Cc(groups * state_size);
    for (int i = 0; i < state_size; ++i) { B[i] = 0.2f * (i + 1); Cc[i] = 0.3f * (i + 1); }
    std::vector<float> D = {1.0f, 0.5f};
    std::vector<float> dt = {0.1f, -0.2f};
    std::vector<float> dt_bias = {0.05f, 0.0f};
    std::vector<float> A_log = {0.0f, std::log(2.0f)};

    std::vector<float> ref_state(state0.size()), ref_out(heads * head_dim);
    ref::SelectiveStateUpdate(state0.data(), xs.data(), B.data(), Cc.data(),
                              D.data(), dt.data(), dt_bias.data(),
                              A_log.data(), heads, head_dim, groups, state_size,
                              /*clamp_dt=*/true, ref_state.data(),
                              ref_out.data());

    float *d_s0 = ToDevice(state0), *d_xs = ToDevice(xs), *d_B = ToDevice(B);
    float *d_C = ToDevice(Cc), *d_D = ToDevice(D), *d_dt = ToDevice(dt);
    float *d_db = ToDevice(dt_bias), *d_Al = ToDevice(A_log);
    float *d_s1 = nullptr, *d_o = nullptr;
    cudaMalloc(&d_s1, state0.size() * sizeof(float));
    cudaMalloc(&d_o, ref_out.size() * sizeof(float));

    RunNemotronSelectiveStateStep(d_s0, d_xs, d_B, d_C, d_D, d_dt, d_db, d_Al,
                                  heads, head_dim, groups, state_size, /*clamp_dt=*/1,
                                  d_s1, d_o, 0);
    cudaDeviceSynchronize();

    auto got_state = FromDevice(d_s1, ref_state.size());
    auto got_out = FromDevice(d_o, ref_out.size());
    for (size_t i = 0; i < ref_state.size(); ++i)
        CHECK(got_state[i] == Catch::Approx(ref_state[i]).margin(1e-5f));
    for (size_t i = 0; i < ref_out.size(); ++i)
        CHECK(got_out[i] == Catch::Approx(ref_out[i]).margin(1e-5f));

    cudaFree(d_s0); cudaFree(d_xs); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    cudaFree(d_dt); cudaFree(d_db); cudaFree(d_Al); cudaFree(d_s1); cudaFree(d_o);
}

TEST_CASE("selective state update honors the dt clamp flag", "[nemotron][kernels]")
{
    // Prefill (chunk) clamps dt to kDtMin; decode (selective update) does not.
    // softplus(-30) ~ 9.4e-14 << kDtMin, so the two forms must differ.
    const int heads = 1, head_dim = 1, groups = 1, state_size = 1;
    const std::vector<float> state0 = {0.0f};
    const std::vector<float> xs = {1.0f};
    const std::vector<float> B = {1.0f}, Cc = {1.0f};
    const std::vector<float> D = {0.0f};
    const std::vector<float> dt = {-30.0f};
    const std::vector<float> dt_bias = {0.0f};
    const std::vector<float> A_log = {0.0f};

    float *d_s0 = ToDevice(state0), *d_xs = ToDevice(xs), *d_B = ToDevice(B);
    float *d_C = ToDevice(Cc), *d_D = ToDevice(D), *d_dt = ToDevice(dt);
    float *d_db = ToDevice(dt_bias), *d_Al = ToDevice(A_log);
    float *d_s1 = nullptr, *d_o = nullptr;
    cudaMalloc(&d_s1, state0.size() * sizeof(float));
    cudaMalloc(&d_o, xs.size() * sizeof(float));

    for (int clamp_dt = 0; clamp_dt <= 1; ++clamp_dt) {
        RunNemotronSelectiveStateStep(d_s0, d_xs, d_B, d_C, d_D, d_dt, d_db,
                                      d_Al, heads, head_dim, groups, state_size,
                                      clamp_dt, d_s1, d_o, 0);
        cudaDeviceSynchronize();
        const auto got = FromDevice(d_s1, 1);
        const float expect = clamp_dt ? frozen::kDtMin : std::exp(-30.0f);
        CHECK(got[0] == Catch::Approx(expect).margin(1e-6f));
    }

    cudaFree(d_s0); cudaFree(d_xs); cudaFree(d_B); cudaFree(d_C); cudaFree(d_D);
    cudaFree(d_dt); cudaFree(d_db); cudaFree(d_Al); cudaFree(d_s1); cudaFree(d_o);
}

TEST_CASE("gated rmsnorm matches the host reference", "[nemotron][kernels]")
{
    const int group_size = 8, n = 16;
    std::vector<float> x(n), g(n), w(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 0.1f * (i + 1);
        g[i] = (i % 2 == 0) ? 0.5f : -0.25f;
        w[i] = 0.5f + 0.01f * i;
    }
    const float eps = 1e-5f;
    std::vector<float> ref_out(n);
    ref::GatedRmsNorm(x.data(), g.data(), w.data(), n, group_size, eps,
                      ref_out.data());

    float *d_x = ToDevice(x), *d_g = ToDevice(g), *d_w = ToDevice(w);
    float *d_o = nullptr;
    cudaMalloc(&d_o, n * sizeof(float));
    RunNemotronGatedRmsNorm(d_x, d_g, d_w, n, group_size, eps, d_o, 0);
    cudaDeviceSynchronize();
    auto got = FromDevice(d_o, n);
    for (int i = 0; i < n; ++i)
        CHECK(got[i] == Catch::Approx(ref_out[i]).margin(1e-5f));

    cudaFree(d_x); cudaFree(d_g); cudaFree(d_w); cudaFree(d_o);
}

TEST_CASE("full-dimension RoPE matches the rotate_half reference",
          "[nemotron][kernels]")
{
    const int heads = 2, head_dim = 8, pos = 3;
    const float theta = 10000.0f;
    std::vector<float> x(heads * head_dim);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.1f * (i + 1);

    std::vector<float> ref = x;
    const int half = head_dim / 2;
    for (int h = 0; h < heads; ++h) {
        for (int j = 0; j < half; ++j) {
            const float freq = std::pow(theta, -2.0f * j / head_dim);
            const float a = pos * freq;
            const float c = std::cos(a), s = std::sin(a);
            const float x1 = x[h * head_dim + j];
            const float x2 = x[h * head_dim + half + j];
            ref[h * head_dim + j] = x1 * c - x2 * s;
            ref[h * head_dim + half + j] = x1 * s + x2 * c;
        }
    }

    float *d_x = ToDevice(x);
    RunNemotronRope(d_x, heads, head_dim, pos, theta, 0);
    cudaDeviceSynchronize();
    auto got = FromDevice(d_x, x.size());
    for (size_t i = 0; i < x.size(); ++i)
        CHECK(got[i] == Catch::Approx(ref[i]).margin(1e-5f));
    cudaFree(d_x);
}

TEST_CASE("GQA decode attention matches the host softmax reference",
          "[nemotron][kernels]")
{
    const int heads = 2, kv_heads = 1, head_dim = 4, seq_len = 3;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> q(heads * head_dim), k(kv_heads * seq_len * head_dim),
        v(kv_heads * seq_len * head_dim);
    for (size_t i = 0; i < q.size(); ++i) q[i] = 0.05f * (i + 1);
    for (size_t i = 0; i < k.size(); ++i) k[i] = 0.03f * (i + 2);
    for (size_t i = 0; i < v.size(); ++i) v[i] = 0.07f * (i + 1);

    std::vector<float> ref(heads * head_dim);
    for (int h = 0; h < heads; ++h) {
        const int hk = h / (heads / kv_heads);
        float scores[4];
        float mx = -INFINITY;
        for (int t = 0; t < seq_len; ++t) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
                dot += q[h * head_dim + d] *
                       k[(hk * seq_len + t) * head_dim + d];
            scores[t] = dot * scale;
            mx = std::max(mx, scores[t]);
        }
        float sum = 0.0f, p[4];
        for (int t = 0; t < seq_len; ++t) {
            p[t] = std::exp(scores[t] - mx);
            sum += p[t];
        }
        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int t = 0; t < seq_len; ++t)
                acc += p[t] * v[(hk * seq_len + t) * head_dim + d];
            ref[h * head_dim + d] = acc / sum;
        }
    }

    float *d_q = ToDevice(q), *d_k = ToDevice(k), *d_v = ToDevice(v);
    float *d_o = nullptr;
    cudaMalloc(&d_o, ref.size() * sizeof(float));
    RunNemotronAttnDecode(d_q, d_k, d_v, heads, kv_heads, head_dim, seq_len,
                          seq_len, scale, d_o, 0);
    cudaDeviceSynchronize();
    auto got = FromDevice(d_o, ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
        CHECK(got[i] == Catch::Approx(ref[i]).margin(1e-5f));
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
}

TEST_CASE("sigmoid top-k router matches the host reference", "[nemotron][kernels]")
{
    const int E = 8, k = 2;
    const float scale = 2.5f;
    std::vector<float> logits = {0.0f, 1.0f, 2.0f, -1.0f, 0.5f, -0.5f, 3.0f, -2.0f};
    std::vector<float> bias(E, 0.0f);

    std::vector<float> ref_w(k);
    std::vector<int> ref_i(k);
    ref::RouterTopK(logits.data(), bias.data(), E, k, scale, ref_w.data(),
                    ref_i.data());

    float *d_l = ToDevice(logits), *d_b = ToDevice(bias);
    float *d_w = nullptr;
    int *d_i = nullptr;
    cudaMalloc(&d_w, k * sizeof(float));
    cudaMalloc(&d_i, k * sizeof(int));
    RunNemotronRouter(d_l, d_b, E, k, scale, d_w, d_i, 0);
    cudaDeviceSynchronize();
    auto got_w = FromDevice(d_w, k);
    auto got_i = FromDevice(d_i, k);
    for (int t = 0; t < k; ++t) {
        CHECK(got_i[t] == ref_i[t]);
        CHECK(got_w[t] == Catch::Approx(ref_w[t]).margin(1e-5f));
    }
    cudaFree(d_l); cudaFree(d_b); cudaFree(d_w); cudaFree(d_i);
}

TEST_CASE("router e_score_correction_bias selects without scaling",
          "[nemotron][kernels]")
{
    const int E = 8, k = 1;
    const float scale = 2.5f;
    std::vector<float> logits = {0.0f, 1.0f, 2.0f, -1.0f, 0.5f, -0.5f, 0.0f, -2.0f};
    std::vector<float> bias(E, 0.0f);
    bias[7] = 10.0f;  // push expert 7 to the top by selection
    std::vector<float> ref_w(k);
    std::vector<int> ref_i(k);
    ref::RouterTopK(logits.data(), bias.data(), E, k, scale, ref_w.data(),
                    ref_i.data());

    float *d_l = ToDevice(logits), *d_b = ToDevice(bias);
    float *d_w = nullptr;
    int *d_i = nullptr;
    cudaMalloc(&d_w, k * sizeof(float));
    cudaMalloc(&d_i, k * sizeof(int));
    RunNemotronRouter(d_l, d_b, E, k, scale, d_w, d_i, 0);
    cudaDeviceSynchronize();
    auto got_w = FromDevice(d_w, k);
    auto got_i = FromDevice(d_i, k);
    CHECK(got_i[0] == 7);
    CHECK(got_w[0] == Catch::Approx(2.5f).margin(1e-5f));  // weight from raw score
    CHECK(got_i[0] == ref_i[0]);
    cudaFree(d_l); cudaFree(d_b); cudaFree(d_w); cudaFree(d_i);
}

TEST_CASE("relu2 activation", "[nemotron][kernels]")
{
    std::vector<float> x = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 3.0f};
    std::vector<float> ref(x.size());
    for (size_t i = 0; i < x.size(); ++i) ref[i] = ref::Relu2(x[i]);

    float *d_x = ToDevice(x);
    float *d_o = nullptr;
    cudaMalloc(&d_o, x.size() * sizeof(float));
    RunNemotronRelu2(d_x, static_cast<int>(x.size()), d_o, 0);
    cudaDeviceSynchronize();
    auto got = FromDevice(d_o, x.size());
    for (size_t i = 0; i < x.size(); ++i)
        CHECK(got[i] == Catch::Approx(ref[i]).margin(1e-6f));
    cudaFree(d_x); cudaFree(d_o);
}

TEST_CASE("FP16 relu2 and routed weighted-sum helpers match the reference",
          "[nemotron][kernels]")
{
    const int n = 8;
    std::vector<__half> x(n);
    for (int i = 0; i < n; ++i) x[i] = __float2half(0.1f * (i - 4));
    std::vector<__half> ref(n);
    for (int i = 0; i < n; ++i) {
        const float v = __half2float(x[i]);
        const float r = v > 0.0f ? v : 0.0f;
        ref[i] = __float2half(r * r);
    }
    __half *d_x = ToDevice(x);
    RunNemotronRelu2Half(d_x, n, 0);
    cudaDeviceSynchronize();
    auto got = FromDevice(d_x, n);
    for (int i = 0; i < n; ++i)
        CHECK(__half2float(got[i]) == __half2float(ref[i]));

    const int count = 3;
    std::vector<__half> y(count * n);
    for (int i = 0; i < count * n; ++i)
        y[i] = __float2half(0.05f * (i + 1));
    std::vector<float> w = {0.5f, 1.0f, 2.0f};
    std::vector<float> ref_acc(n, 0.0f);
    for (int j = 0; j < n; ++j)
        for (int e = 0; e < count; ++e)
            ref_acc[j] += w[e] * __half2float(y[e * n + j]);
    __half *d_y = ToDevice(y);
    float *d_w = ToDevice(w);
    float *d_acc = nullptr;
    cudaMalloc(&d_acc, n * sizeof(float));
    RunNemotronMoeWeightedSum(d_y, d_w, count, n, d_acc, false, 0);
    cudaDeviceSynchronize();
    auto acc = FromDevice(d_acc, n);
    for (int j = 0; j < n; ++j)
        CHECK(acc[j] == Catch::Approx(ref_acc[j]).margin(1e-6f));

    std::vector<__half> sh(n);
    for (int i = 0; i < n; ++i) sh[i] = __float2half(0.25f * (i + 1));
    __half *d_sh = ToDevice(sh);
    RunNemotronAddHalfToF32(d_sh, n, d_acc, 0);
    cudaDeviceSynchronize();
    auto acc2 = FromDevice(d_acc, n);
    for (int j = 0; j < n; ++j)
        CHECK(acc2[j] ==
              Catch::Approx(ref_acc[j] + __half2float(sh[j])).margin(1e-6f));

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_w); cudaFree(d_acc); cudaFree(d_sh);
}

TEST_CASE("batched expert GEMV is bit-identical to per-expert GEMV",
          "[nemotron][kernels]")
{
    const int K = 512, N = 64, count = 3;
    const bool fm = lsxcommon::lsx_nvfp4::LsxTcFastMath();
    const bool s1 = lsxcommon::lsx_nvfp4::LsxSeq1Gemv();
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(true);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(true);

    std::vector<std::vector<uint8_t>> packed(count), scales(count);
    std::vector<float> gs(count);
    std::vector<const uint8_t *> dp(count), ds(count);
    std::vector<uint8_t *> d_packed(count), d_scales(count);
    for (int e = 0; e < count; ++e) {
        std::vector<uint8_t> raw((size_t)K * N * 2);
        for (size_t i = 0; i < raw.size(); ++i)
            raw[i] = static_cast<uint8_t>((i * (e + 3) + 7) & 0xff);
        if (!lsxcommon::gpu::quantize_tensor_gpu(raw.data(), (int64_t)K * N,
                                                 "BF16", nullptr, {N, K}, {},
                                                 packed[e], scales[e], gs[e]))
            FAIL("quantize_tensor_gpu failed");
        cudaMalloc(&d_packed[e], packed[e].size());
        cudaMemcpy(d_packed[e], packed[e].data(), packed[e].size(),
                   cudaMemcpyHostToDevice);
        cudaMalloc(&d_scales[e], scales[e].size());
        cudaMemcpy(d_scales[e], scales[e].data(), scales[e].size(),
                   cudaMemcpyHostToDevice);
        dp[e] = d_packed[e];
        ds[e] = d_scales[e];
    }
    std::vector<__half> x(K);
    for (int i = 0; i < K; ++i) x[i] = __float2half(0.01f * (i + 1));
    __half *d_x = ToDevice(x);
    __half *d_ref = nullptr, *d_batch = nullptr;
    cudaMalloc(&d_ref, (size_t)count * N * sizeof(__half));
    cudaMalloc(&d_batch, (size_t)count * N * sizeof(__half));
    for (int e = 0; e < count; ++e)
        lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
            0, d_x, dp[e], ds[e], gs[e], d_ref + (size_t)e * N, K, N);
    cudaDeviceSynchronize();
    lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
        0, d_x, dp.data(), ds.data(), gs.data(), count, /*x_stride=*/0, d_batch,
        K, N);
    cudaDeviceSynchronize();
    auto ref = FromDevice(d_ref, (size_t)count * N);
    auto got = FromDevice(d_batch, (size_t)count * N);
    for (size_t i = 0; i < ref.size(); ++i)
        CHECK(std::memcmp(&ref[i], &got[i], sizeof(__half)) == 0);

    for (int e = 0; e < count; ++e) { cudaFree(d_packed[e]); cudaFree(d_scales[e]); }
    cudaFree(d_x); cudaFree(d_ref); cudaFree(d_batch);
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(fm);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(s1);
}

#if defined(NDEBUG)
TEST_CASE("budget: batched expert GEMV (6 experts)", "[nemotron][kernels][budget]")
{
    const int K = 2688, N = 1856, count = 6;
    const bool fm = lsxcommon::lsx_nvfp4::LsxTcFastMath();
    const bool s1 = lsxcommon::lsx_nvfp4::LsxSeq1Gemv();
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(true);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(true);

    std::vector<std::vector<uint8_t>> packed(count), scales(count);
    std::vector<float> gs(count);
    std::vector<const uint8_t *> dp(count), ds(count);
    std::vector<uint8_t *> d_packed(count), d_scales(count);
    for (int e = 0; e < count; ++e) {
        std::vector<uint8_t> raw((size_t)K * N * 2);
        for (size_t i = 0; i < raw.size(); ++i)
            raw[i] = static_cast<uint8_t>((i * (e + 3) + 11) & 0xff);
        REQUIRE(lsxcommon::gpu::quantize_tensor_gpu(
            raw.data(), (int64_t)K * N, "BF16", nullptr, {N, K}, {}, packed[e],
            scales[e], gs[e]));
        cudaMalloc(&d_packed[e], packed[e].size());
        cudaMemcpy(d_packed[e], packed[e].data(), packed[e].size(),
                   cudaMemcpyHostToDevice);
        cudaMalloc(&d_scales[e], scales[e].size());
        cudaMemcpy(d_scales[e], scales[e].data(), scales[e].size(),
                   cudaMemcpyHostToDevice);
        dp[e] = d_packed[e];
        ds[e] = d_scales[e];
    }
    std::vector<__half> x(K, __float2half(0.01f));
    __half *d_x = ToDevice(x);
    __half *d_y = nullptr;
    cudaMalloc(&d_y, (size_t)count * N * sizeof(__half));
    void *d_flush = nullptr;
    cudaMalloc(&d_flush, size_t{128} << 20);

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int reps = 100;
    float sum_single = 0.0f, sum_batch = 0.0f;
    for (int r = 0; r < reps; ++r) {
        cudaMemsetAsync(d_flush, 0, size_t{128} << 20, 0);
        cudaEventRecord(e0, 0);
        for (int e = 0; e < count; ++e)
            lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(
                0, d_x, dp[e], ds[e], gs[e], d_y + (size_t)e * N, K, N);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        sum_single += ms;

        cudaMemsetAsync(d_flush, 0, size_t{128} << 20, 0);
        cudaEventRecord(e0, 0);
        lsxcommon::lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
            0, d_x, dp.data(), ds.data(), gs.data(), count, 0, d_y, K, N);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
        sum_batch += ms;
    }
    const float single_us = sum_single / reps * 1000.0f;
    const float batch_us = sum_batch / reps * 1000.0f;
    INFO("expert GEMV x6: single=" << single_us << " us  batch=" << batch_us
                                   << " us");
    CHECK(batch_us < single_us * 0.6f);
    CHECK(batch_us < 80.0f);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_flush);
    for (int e = 0; e < count; ++e) { cudaFree(d_packed[e]); cudaFree(d_scales[e]); }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    lsxcommon::lsx_nvfp4::SetLsxTcFastMath(fm);
    lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(s1);
}
#endif

#if defined(NDEBUG)
// Perf budget (perf-tuning-a-model Step 3): the router top-k runs once per MoE
// layer per token, so its latency is on the decode critical path. Budget < 15 us
// for the real E=128/k=6 shape (baseline measured ~70 us with the serial
// single-thread selection). L2 is flushed with a >L2 memset between launches;
// the event window covers only the kernel.
TEST_CASE("budget: nemotron router latency (E=128, k=6)",
          "[nemotron][kernels][budget]")
{
    const int E = 128, k = 6;
    std::vector<float> logits(E), bias(E, 0.0f);
    for (int i = 0; i < E; ++i) logits[i] = 0.01f * static_cast<float>(i);
    float *d_l = ToDevice(logits), *d_b = ToDevice(bias);
    float *d_w = nullptr;
    int *d_i = nullptr;
    cudaMalloc(&d_w, k * sizeof(float));
    cudaMalloc(&d_i, k * sizeof(int));
    void *d_flush = nullptr;
    cudaMalloc(&d_flush, size_t{128} << 20);

    for (int i = 0; i < 5; ++i)
        RunNemotronRouter(d_l, d_b, E, k, 2.5f, d_w, d_i, 0);
    cudaDeviceSynchronize();

    cudaEvent_t e0 = nullptr, e1 = nullptr;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int N = 200;
    float sum_ms = 0.0f;
    for (int i = 0; i < N; ++i) {
        cudaMemsetAsync(d_flush, 0, size_t{128} << 20, 0);
        cudaEventRecord(e0, 0);
        RunNemotronRouter(d_l, d_b, E, k, 2.5f, d_w, d_i, 0);
        cudaEventRecord(e1, 0);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        sum_ms += ms;
    }
    const float avg_us = sum_ms / static_cast<float>(N) * 1000.0f;
    INFO("router kernel avg latency: " << avg_us << " us");
    CHECK(avg_us < 15.0f);

    cudaFree(d_l); cudaFree(d_b); cudaFree(d_w); cudaFree(d_i); cudaFree(d_flush);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}
#endif



