// TDD — MiMo attention prefill parity (GA and SWA + sink bias) against the
// family host reference. CUDA OBJECT lib.
#include <catch2/catch_test_macros.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "lsxcommon/mimo_moe_inference_attn.h"
#include "mimo_moe_device_math.h"

using namespace lsxcommon::mimo;
namespace ref = lsxcommon::mimo::ref;

namespace {

MimoAttnShape SmallShape(bool swa)
{
    MimoAttnShape p;
    p.heads = 2; p.kv_heads = 1; p.head_dim = 8; p.v_head_dim = 4; p.rope_dim = 4;
    p.is_swa = swa; p.window = 3; p.value_scale = 0.707f;
    p.scale = 1.0f / std::sqrt(8.0f); p.rope_theta = swa ? 1e4 : 1e7;
    p.q_size = p.heads * p.head_dim;   // 16
    p.k_size = p.kv_heads * p.head_dim; // 8
    p.v_size = p.kv_heads * p.v_head_dim; // 4
    return p;
}

void RunCase(bool swa, bool with_sink)
{
    const int seq = 6;
    const MimoAttnShape p = SmallShape(swa);
    const int rows = p.q_size + 2 * p.k_size;
    std::vector<float> qkv_f(seq * rows);
    for (size_t i = 0; i < qkv_f.size(); ++i)
        qkv_f[i] = std::sin(static_cast<float>(i) * 0.37f);
    std::vector<float> sink_f(p.heads);
    for (int h = 0; h < p.heads; ++h) sink_f[h] = 0.2f * (h + 1);

    ref::AttnShape rp;
    rp.heads = p.heads; rp.kv_heads = p.kv_heads; rp.head_dim = p.head_dim;
    rp.v_head_dim = p.v_head_dim; rp.rope_dim = p.rope_dim; rp.is_swa = p.is_swa;
    rp.window = p.window; rp.value_scale = p.value_scale; rp.rope_theta = p.rope_theta;
    rp.scale = p.scale; rp.q_size = p.q_size; rp.k_size = p.k_size; rp.v_size = p.v_size;
    std::vector<float> ref_out(seq * p.heads * p.v_head_dim);
    ref::AttentionRef(qkv_f, seq, rp, with_sink ? sink_f.data() : nullptr,
                      ref_out.data());

    std::vector<__half> qkv(qkv_f.size());
    for (size_t i = 0; i < qkv_f.size(); ++i) qkv[i] = __float2half(qkv_f[i]);
    // Host reference consumed rounded FP16 inputs.
    std::vector<float> qkv_r(qkv_f.size());
    for (size_t i = 0; i < qkv_f.size(); ++i) qkv_r[i] = __half2float(qkv[i]);
    ref::AttentionRef(qkv_r, seq, rp, with_sink ? sink_f.data() : nullptr,
                      ref_out.data());

    std::vector<__half> sink(p.heads);
    for (int h = 0; h < p.heads; ++h) sink[h] = __float2half(sink_f[h]);
    std::vector<__half> out(seq * p.heads * p.v_head_dim);

    __half *d_qkv, *d_sink = nullptr, *d_out;
    REQUIRE(cudaMalloc(&d_qkv, qkv.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, out.size() * 2) == cudaSuccess);
    if (with_sink) REQUIRE(cudaMalloc(&d_sink, sink.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_qkv, qkv.data(), qkv.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    if (with_sink)
        REQUIRE(cudaMemcpy(d_sink, sink.data(), sink.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(RunMimoAttentionPrefill(nullptr, d_qkv, d_sink, seq, p, d_out) == cudaSuccess);
    REQUIRE(cudaMemcpy(out.data(), d_out, out.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess);

    for (size_t i = 0; i < out.size(); ++i)
        REQUIRE(std::fabs(__half2float(out[i]) - ref_out[i]) < 5e-3f);
    cudaFree(d_qkv); cudaFree(d_out); if (with_sink) cudaFree(d_sink);
}

} // namespace

TEST_CASE("mimo attention GA parity", "[mimo][attn]") { RunCase(false, false); }
TEST_CASE("mimo attention SWA parity with sink bias", "[mimo][attn]") { RunCase(true, true); }

TEST_CASE("mimo attention decode matches prefill", "[mimo][attn]")
{
    const int seq = 6;
    const MimoAttnShape p = SmallShape(true);  // SWA + sink exercises the window
    const int rows = p.q_size + 2 * p.k_size;
    std::vector<__half> qkv(seq * rows);
    for (size_t i = 0; i < qkv.size(); ++i)
        qkv[i] = __float2half(std::sin(static_cast<float>(i) * 0.19f));
    std::vector<__half> sink(p.heads);
    for (int h = 0; h < p.heads; ++h) sink[h] = __float2half(0.1f * (h + 1));

    __half *d_qkv, *d_sink, *d_pref, *d_kc, *d_vc, *d_out;
    REQUIRE(cudaMalloc(&d_qkv, qkv.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_sink, sink.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_pref, seq * p.heads * p.v_head_dim * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_kc, static_cast<size_t>(seq) * p.kv_heads * p.head_dim * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_vc, static_cast<size_t>(seq) * p.kv_heads * p.v_head_dim * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, p.heads * p.v_head_dim * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_qkv, qkv.data(), qkv.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_sink, sink.data(), sink.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(RunMimoAttentionPrefill(nullptr, d_qkv, d_sink, seq, p, d_pref) == cudaSuccess);

    for (int t = 0; t < seq; ++t)
        REQUIRE(RunMimoAttentionDecode(nullptr, d_qkv + static_cast<int64_t>(t) * rows,
                                       t, d_sink, p, d_kc, d_vc, d_out) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> pref(seq * p.heads * p.v_head_dim), dec(p.heads * p.v_head_dim);
    REQUIRE(cudaMemcpy(pref.data(), d_pref, pref.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(dec.data(), d_out, dec.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess);
    const int last = seq - 1;
    for (int i = 0; i < p.heads * p.v_head_dim; ++i)
        REQUIRE(std::fabs(__half2float(dec[i]) -
                          __half2float(pref[last * p.heads * p.v_head_dim + i])) < 5e-3f);
    cudaFree(d_qkv); cudaFree(d_sink); cudaFree(d_pref);
    cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_out);
}
