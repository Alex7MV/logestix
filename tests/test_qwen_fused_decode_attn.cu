// Fused decoder (seq=1) full-attention 3-stage parity vs the split path.
//
// Direction A: the 9-launch seq=1 full-attention chain is fused into 3 kernels
// (qkv -> online-softmax GQA attention -> o_proj). These tests verify the fused
// entry first exists (Task 1), then per-stage and full-layer parity vs the
// existing split path within a FP32 tolerance envelope (Tasks 2-4).
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <cmath>

#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/qwen_convert.h"

namespace {

using lsxcommon::qwen::QwenNvfp4Weight;
using lsxcommon::qwen::QwenLayerWeights;
using lsxcommon::qwen::QwenTopology;
using lsxcommon::qwen::RunQwenFusedDecodeAttnKernel;

int64_t nvfp4_packed_bytes(int K, int N) { return (int64_t)K * N / 2; }
int64_t nvfp4_scale_bytes(int K, int N)  { return (int64_t)K * N / 16; }

// Deterministic NVFP4 fill: nibble-packed bytes + E4M3 block scales.
void FillNvfp4(QwenNvfp4Weight &w, int K, int N,
               uint8_t seed_pack, uint8_t seed_scale)
{
    w.rows = K; w.cols = N;
    w.packed_bytes = nvfp4_packed_bytes(K, N);
    w.scale_bytes  = nvfp4_scale_bytes(K, N);
    w.global_scale = 0.5f;
    cudaMalloc(&w.d_packed, (size_t)w.packed_bytes);
    cudaMalloc(&w.d_scales, (size_t)w.scale_bytes);
    std::vector<uint8_t> pk((size_t)w.packed_bytes), sc((size_t)w.scale_bytes);
    for (int64_t i = 0; i < w.packed_bytes; ++i) pk[(size_t)i] = (uint8_t)((i * 7 + seed_pack) & 0xFF);
    for (int64_t i = 0; i < w.scale_bytes; ++i)  sc[(size_t)i]  = (uint8_t)(128 + ((int)(i + seed_scale) % 32));
    cudaMemcpy((void*)w.d_packed, pk.data(), pk.size(), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)w.d_scales, sc.data(), sc.size(), cudaMemcpyHostToDevice);
}

// Build a tiny full-attention layer: NH=2, NKV=1, HD=8, H=16, rotary_dim=4.
QwenLayerWeights MakeTinyFullLayer(int NH, int NKV, int HD, int H, int rotary_dim)
{
    QwenLayerWeights lw;
    lw.is_linear = false;
    lw.loaded = true;
    lw.norm_size = H;
    FillNvfp4(lw.q, H, NH*HD*2, 1, 2);
    FillNvfp4(lw.k, H, NKV*HD, 3, 4);
    FillNvfp4(lw.v, H, NKV*HD, 5, 6);
    FillNvfp4(lw.o, NH*HD, H, 7, 8);
    lw.q_norm_size = HD; lw.k_norm_size = HD;
    cudaMalloc(&lw.d_q_norm, (size_t)HD*2);
    cudaMalloc(&lw.d_k_norm, (size_t)HD*2);
    cudaMalloc(&lw.d_input_norm, (size_t)H*2);
    std::vector<__half> qn(HD), kn(HD), in(H);
    for (int i = 0; i < HD; ++i) { qn[i] = __float2half(0.1f * ((i % 3))); kn[i] = __float2half(0.05f*((i%5))); }
    for (int i = 0; i < H; ++i)  in[i] = __float2half(0.5f + 0.02f * (float)(i % 7));
    cudaMemcpy(lw.d_q_norm, qn.data(), HD*2, cudaMemcpyHostToDevice);
    cudaMemcpy(lw.d_k_norm, kn.data(), HD*2, cudaMemcpyHostToDevice);
    cudaMemcpy(lw.d_input_norm, in.data(), H*2, cudaMemcpyHostToDevice);
    return lw;
}

void FreeTinyLayer(QwenLayerWeights &lw)
{
    lw.q.Free(); lw.k.Free(); lw.v.Free(); lw.o.Free();
    cudaFree(lw.d_q_norm); cudaFree(lw.d_k_norm); cudaFree(lw.d_input_norm);
}

}  // namespace

// Task 1: the public numeric entry exists and accepts a tiny setup.
TEST_CASE("RunQwenFusedDecodeAttnKernel accepts a tiny full-attn layer", "[qwen][fused][decode]")
{
    constexpr int NH = 2, NKV = 1, HD = 8, H = 16, rotary_dim = 4;
    constexpr int max_kv = 8;

    QwenLayerWeights lw = MakeTinyFullLayer(NH, NKV, HD, H, rotary_dim);

    lsxcommon::qwen::QwenTopology top;
    top.hidden_size = H;
    top.num_attention_heads = NH;
    top.num_key_value_heads = NKV;
    top.head_dim = HD;
    top.partial_rotary_factor = 0.25;
    top.rope_theta = 10000000.0;
    top.rms_norm_eps = 1e-6;

    // x token [H], kv buffers, out [H]
    __half *dx, *dkv_k, *dkv_v, *dout;
    cudaMalloc(&dx, H*2);
    cudaMalloc(&dkv_k, (size_t)max_kv * NKV * HD * 2);
    cudaMalloc(&dkv_v, (size_t)max_kv * NKV * HD * 2);
    cudaMalloc(&dout, H*2);
    std::vector<__half> hx(H);
    for (int i = 0; i < H; ++i) hx[i] = __float2half(0.3f + 0.01f * (float)(i % 9));
    cudaMemcpy(dx, hx.data(), H*2, cudaMemcpyHostToDevice);
    cudaMemset(dkv_k, 0, (size_t)max_kv * NKV * HD * 2);
    cudaMemset(dkv_v, 0, (size_t)max_kv * NKV * HD * 2);

    cudaError_t err = lsxcommon::qwen::RunQwenFusedDecodeAttnKernel(
        nullptr, dx, lw, top, /*kv_base=*/0, dkv_k, dkv_v, dout);
    REQUIRE(err == cudaSuccess);

    cudaFree(dx); cudaFree(dkv_k); cudaFree(dkv_v); cudaFree(dout);
    FreeTinyLayer(lw);
}