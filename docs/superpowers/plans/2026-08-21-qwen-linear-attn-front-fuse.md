# Qwen Linear-Attention (GatedDeltaNet) Front-End Fuse — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the linear-attention layer from 11 launches to 5 by fusing the input-side launches into 2 kernels (F1 qkv+conv1d+l2norm, F2 z+a+b+prep_gating), matching the split path within `1e-2`.

**Architecture:** One block per token row (`grid (seq,1)`, `blockDim=pow2(kd)`) for both fused kernels. F1 writes the raw qkv GEMM row to the persistent history (`d_qkv`), runs the causal conv1d+silu, then l2normalizes the q/k slices in place. F2 computes z/a/b projections and prep_gating. The recurrence (`qwen_gated_delta_kernel`), RMSNormGated, and out_proj stay unchanged. `RunQwenLinearAttention` is kept as the split reference; a new public `RunQwenFusedLinearAttnKernel` entry composes F1/F2 + the unchanged tail, and the two generator `is_linear` sites switch to it.

**Tech Stack:** C++20 / CUDA 13.3 (Blackwell sm_120/121), NVFP4 codec from `nvfp4_codec.h` (`E4M3ScaleToFloat`, `kE2M1Table`), Catch2 v3.7.0.

---

## File structure

- `libs/lsxcommon/include/lsxcommon/qwen_inference.h` — declare `RunQwenFusedLinearAttnKernel` (mirrors `RunQwenLinearAttention`).
- `libs/lsxcommon/src/qwen_inference.cu` — add `fused_linear_front_kernel` (F1) + `fused_linear_zab_kernel` (F2) in anonymous ns, and external `RunQwenFusedLinearAttnKernel` after the second anon-ns close (~:837); rewire the two generator sites (`:1583`, `:1674`).
- `tests/test_qwen_fused_linear_attn.cu` — F1 unit parity + full-layer parity (seq=1, seq=2).
- `tests/CMakeLists.txt` — register the test in `test_qwen_cu` OBJECT + `foreach(qt ...)`.

## Existing references to mirror

- `qwen_nvfp4_gemm_kernel` (`qwen_inference.cu:394`): NVFP4 dequant dot — `e=n*K+k; nib = (e&1)?byte>>4:byte&0x0F; scl = E4M3ScaleToFloat(scales[e>>4])*global_scale; acc += half(xrow[k])*kE2M1Table[nib]*scl`.
- `qwen_conv1d_kernel` (`:203`): `acc += w[c*kernel+(kernel-1-k)] * mixed[(kv_base+t-k)*conv_dim+c]`, `out=silu(acc)`.
- `qwen_l2norm_rows_kernel` (`:589`): power-of-2 block reduction, `inv=rsqrtf(sum+1e-6f)*extra_scale`.
- `qwen_prep_gating_kernel` (`:572`): `beta=sigmoid(b)`, `g=-expf(A_log[h])*softplus(a+dt_bias[h])`.
- `RunQwenLinearAttention` (`:1163`): split reference signature + stage order.
- `QwenLayerWeights` (`qwen_inference_load.h:33`): fields `qkv/z/a/b/out` (`QwenNvfp4Weight`: `.rows`=input dim K, `.cols`=output dim N; raw `d_packed`, `d_scales`, `global_scale`), `d_conv1d`/`conv_dim`/`conv_kernel`, `d_A_log`/`d_dt_bias` (float, `[nv]`), `d_attn_norm`.

Key helper names in file scope: `E4M3ScaleToFloat`, `kE2M1Table`, `silu_dev`, `sigmoid_dev`, `softplus_dev` (all reachable from `lsxcommon::qwen` after anon-ns close).

---

### Task 1: Fused kernels F1 + F2 + public entry (split parity base)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_inference.h`
- Modify: `libs/lsxcommon/src/qwen_inference.cu`
- Test: `tests/test_qwen_fused_linear_attn.cu`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Register the test skeleton (RED on entry symbol)**

Add `test_qwen_fused_linear_attn.cu` to the `test_qwen_cu` OBJECT library and to the `foreach(qt ...)` list in `tests/CMakeLists.txt`, next to the other Qwen CUDA tests (mirror how `test_qwen_fused_mlp.cu` is registered).

`tests/test_qwen_fused_linear_attn.cu` — starting with a minimal case that calls the (not-yet-declared) public entry and the split reference, so the build is RED until the symbols exist. Header + helpers:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "lsxcommon/qwen_inference.h"

using namespace lsxcommon::qwen;

namespace {

// Host replica of E4M3 per-block scale (match nvfp4_codec.h E4M3ScaleToFloat).
__host__ float HostE4M3(uint8_t e4) {
    const uint8_t sign = e4 >> 4, exp = (e4 >> 3) & 0xF, man = e4 & 0x7;
    if (exp == 0) return 0.0f;
    unsigned bits = (sign << 31) | ((exp + 120) << 23) | (man << 20);
    float f; __builtin_memcpy(&f, &bits, 4); return f;
}
constexpr float kHostE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Host NVFP4 dequant weight element (row-major [K x N] -> flat e = n*K + k).
__host__ float HostDeq(const std::vector<uint8_t>& packed, const std::vector<uint8_t>& scales,
                       int K, int N, float gs, int k, int n) {
    const int64_t e = (int64_t)n * K + k;
    const uint8_t byte = packed[e >> 1];
    const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
    int s = nib & 0x7;              // E2M1 low 3 bits magnitude
    float sign = (nib & 0x8) ? -1.0f : 1.0f;
    return sign * kHostE2M1[s] * HostE4M3(scales[e >> 4]) * gs;
}

// Fill QwenNvfp4Weight on device from deterministic vectors + misc fields.
void FillNvfp4(QwenNvfp4Weight &w, int K, int N, float gs) {
    std::vector<uint8_t> pk((size_t)K * N / 2), sc((size_t)K * N / 16);
    for (size_t i = 0; i < pk.size(); ++i) pk[i] = (uint8_t)((i * 37 + 5) & 0xFF);
    for (size_t i = 0; i < sc.size(); ++i) sc[i] = (uint8_t)(120 + (i % 12));
    cudaMalloc(&w.d_packed, pk.size()); cudaMemcpy(w.d_packed, pk.data(), pk.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&w.d_scales, sc.size()); cudaMemcpy(w.d_scales, sc.data(), sc.size(), cudaMemcpyHostToDevice);
    w.global_scale = gs; w.rows = K; w.cols = N;
    w.packed_bytes = pk.size(); w.scale_bytes = sc.size();
}

// Host SwiGLU-free arithmetic helpers.
__host__ float HostSilu(float x) { return x / (1.0f + std::exp(-x)); }
__host__ float HostSoftplus(float x) { return std::log1p(std::exp(x)); }
__host__ float HostSigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

} // namespace
```

- [ ] **Step 2: Build to confirm RED**

Run: `& $cmake --build build --config Debug --target test_qwen_fused_linear_attn`
Expected: FAIL — `RunQwenFusedLinearAttnKernel` / `RunQwenLinearAttention` not declared (header not yet edited) or test file has no cases (implement Step 1 test cases before building).

- [ ] **Step 3: Declare `RunQwenFusedLinearAttnKernel` in the header**

In `libs/lsxcommon/include/lsxcommon/qwen_inference.h`, directly below `RunQwenLinearAttention` (after `:96`), add:

```cpp
// Linear-attention layer with a fused front-end (F1 qkv+conv1d+l2norm, F2
// z+a+b+prep_gating), reducing the split path's 8 front launches to 2. The
// gated-delta recurrence, RMSNormGated, and out_proj tail are unchanged.
// Call signature and scratch layout identical to RunQwenLinearAttention.
cudaError_t RunQwenFusedLinearAttnKernel(
    cudaStream_t stream,
    const QwenLayerWeights &lw, const QwenTopology &top, int seq,
    const __half *d_normed, __half *d_out,
    __half *d_qkv, __half *d_z, __half *d_ab,
    float *d_g, float *d_beta,
    __half *d_conv, __half *d_core, __half *d_gated,
    int kv_base = 0, float *d_gd_state = nullptr);
```

- [ ] **Step 4: Implement F1 fused kernel**

In `libs/lsxcommon/src/qwen_inference.cu`, inside the second anonymous namespace (before it closes ~`:837`), add:

```cpp
// Fused linear-attention front (F1): qkv NVFP4 GEMM -> causal conv1d+silu ->
// l2norm(q,k). One block per token row t; blockDim = pow2(kd). The raw qkv GEMM
// row is written to the persistent d_qkv history (kv_base offsets), then conv
// reads history rows [(kv_base+t-kernel+1) .. (kv_base+t)], then q/k slices of
// the conv buffer are l2normalized in place (q scaled by 1/sqrt(kd)).
// Mirrors qwen_nvfp4_gemm_kernel + qwen_conv1d_kernel + qwen_l2norm_rows_kernel.
__global__ void fused_linear_front_kernel(
    const __half * __restrict__ x,          // d_normed [seq, H]
    const QwenNvfp4Weight qkv,              // rows=H, cols=S
    const __half * __restrict__ conv_w,     // [S, kernel]
    int seq, int nk, int kd, int nv, int vd, int H, int kernel, int kv_base,
    __half * __restrict__ d_qkv,            // [max?, S] persistent raw history
    __half * __restrict__ d_conv)           // [seq, S]
{
    const int t = blockIdx.x;
    const int S = nk * kd * 2 + nv * vd;
    const int KD = nk * kd;
    const int N = S;                        // qkv output cols
    const int K = H;
    const int nth = blockDim.x;

    // Phase A: raw qkv GEMM row -> d_qkv[kv_base+t].
    for (int c = threadIdx.x; c < N; c += nth) {
        float acc = 0.0f;
        const int64_t cbase = (int64_t)c * K;
        for (int k = 0; k < K; ++k) {
            const int64_t e = cbase + k;
            const uint8_t byte = qkv.d_packed[e >> 1];
            const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
            const float scl = E4M3ScaleToFloat(qkv.d_scales[e >> 4]) * qkv.global_scale;
            const int s = nib & 0x7;
            const float sign = (nib & 0x8) ? -1.0f : 1.0f;
            acc += __half2float(x[(int64_t)t * K + k]) * (sign * kE2M1Table[s]) * scl;
        }
        d_qkv[((int64_t)kv_base + t) * S + c] = __float2half(acc);
    }
    __syncthreads();

    // Phase B: causal conv1d+silu over S channels, writing d_conv row t.
    for (int c = threadIdx.x; c < S; c += nth) {
        float acc = 0.0f;
        for (int k = 0; k < kernel; ++k) {
            const int tt = kv_base + t - k;
            if (tt < 0) continue;
            float wv = __half2float(conv_w[(int64_t)c * kernel + (kernel - 1 - k)]);
            float mv = __half2float(d_qkv[(int64_t)tt * S + c]);
            acc += wv * mv;
        }
        d_conv[(int64_t)t * S + c] = __float2half(silu_dev(acc));
    }
    __syncthreads();

    // Phase C: l2norm q (scale 1/sqrt(kd)) and k (scale 1) slices, in place.
    __shared__ float svals[256], sred[256], sinv;
    const float q_scale = 1.0f / sqrtf((float)kd);
    for (int h = 0; h < nk; ++h) {
        for (int slice = 0; slice < 2; ++slice) {
            const int head_off = (slice == 0) ? 0 : KD;
            const float extra = (slice == 0) ? q_scale : 1.0f;
            float v = 0.0f;
            const int lane = threadIdx.x;
            const int gidx = lane + head_off + h * kd;    // global col within S? no: row-local col
            // row-local col = head_off + h*kd + lane
            const int c = head_off + h * kd + lane;
            if (lane < kd) v = __half2float(d_conv[(int64_t)t * S + c]);
            svals[lane] = v; sred[lane] = v * v;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (lane < s) sred[lane] += sred[lane + s];
                __syncthreads();
            }
            if (lane == 0) sinv = rsqrtf(sred[0] + 1e-6f) * extra;
            __syncthreads();
            if (lane < kd) d_conv[(int64_t)t * S + c] = __float2half(svals[lane] * sinv);
            __syncthreads();
        }
    }
}
```

- [ ] **Step 5: Implement F2 fused kernel**

In the same anonymous namespace, add:

```cpp
// Fused linear-attention front (F2): z + a + b NVFP4 GEMMs and prep_gating.
// One block per token row t; blockDim = pow2(kd). a/b write d_ab (a then b),
// g/beta (float) from A_log/dt_bias, z writes d_z.
__global__ void fused_linear_zab_kernel(
    const __half * __restrict__ x,          // d_normed [seq, H]
    const QwenNvfp4Weight z, const QwenNvfp4Weight a, const QwenNvfp4Weight b,
    const float * __restrict__ A_log, const float * __restrict__ dt_bias,
    int seq, int H, int vd, int nv, int VD,
    __half * __restrict__ d_z,              // [seq, VD]
    __half * __restrict__ d_ab)             // [seq*nv*2] a then b
{
    const int t = blockIdx.x;
    const int nth = blockDim.x;
    const __half *xrow = x + (int64_t)t * H;

    // z: [seq, VD]
    for (int c = threadIdx.x; c < VD; c += nth) {
        float acc = 0.0f;
        for (int k = 0; k < H; ++k) {
            const int64_t e = (int64_t)c * H + k;
            const uint8_t byte = z.d_packed[e >> 1];
            const uint8_t nib = (e & 1) ? (byte >> 4) : (byte & 0x0F);
            const float scl = E4M3ScaleToFloat(z.d_scales[e >> 4]) * z.global_scale;
            const int s = nib & 0x7;
            const float sign = (nib & 0x8) ? -1.0f : 1.0f;
            acc += __half2float(xrow[k]) * (sign * kE2M1Table[s]) * scl;
        }
        d_z[(int64_t)t * VD + c] = __float2half(acc);
    }

    // a then b: [seq*nv]
    __half *a_buf = d_ab;
    __half *b_buf = d_ab + (int64_t)seq * nv;
    for (int c = threadIdx.x; c < nv; c += nth) {
        float acca = 0.0f, accb = 0.0f;
        for (int k = 0; k < H; ++k) {
            const int64_t ea = (int64_t)c * H + k;
            const uint8_t ba = a.d_packed[ea >> 1];
            const uint8_t na = (ea & 1) ? (ba >> 4) : (ba & 0x0F);
            const float sa = E4M3ScaleToFloat(a.d_scales[ea >> 4]) * a.global_scale;
            const int ssa = na & 0x7; const float sga = (na & 0x8) ? -1.0f : 1.0f;
            acca += __half2float(xrow[k]) * (sga * kE2M1Table[ssa]) * sa;
            const int64_t eb = (int64_t)c * H + k;
            const uint8_t bb = b.d_packed[eb >> 1];
            const uint8_t nb = (eb & 1) ? (bb >> 4) : (bb & 0x0F);
            const float sb = E4M3ScaleToFloat(b.d_scales[eb >> 4]) * b.global_scale;
            const int ssb = nb & 0x7; const float sgb = (nb & 0x8) ? -1.0f : 1.0f;
            accb += __half2float(xrow[k]) * (sgb * kE2M1Table[ssb]) * sb;
        }
        a_buf[(int64_t)t * nv + c] = __float2half(acca);
        b_buf[(int64_t)t * nv + c] = __float2half(accb);
    }
    __syncthreads();

    // prep_gating -> g, beta (float), per v-head.
    for (int c = threadIdx.x; c < nv; c += nth) {
        float a_v = __half2float(a_buf[(int64_t)t * nv + c]);
        float b_v = __half2float(b_buf[(int64_t)t * nv + c]);
        // NOTE: g/beta are written by caller-owned d_g/d_beta — passed implicitly
        // through a side param below. See Step 6 (the entry passes pointers).
    }
}
```

- [ ] **Step 6: Implement the public entry `RunQwenFusedLinearAttnKernel`**

Replace the incomplete F2 body from Step 5 with the full F2 (need `d_g`/`d_beta` params), then add the external entry. **Final F2 signature** (add `d_g`, `d_beta` float pointers):

```cpp
__global__ void fused_linear_zab_kernel(
    const __half * __restrict__ x, const QwenNvfp4Weight z,
    const QwenNvfp4Weight a, const QwenNvfp4Weight b,
    const float * __restrict__ A_log, const float * __restrict__ dt_bias,
    int seq, int H, int vd, int nv, int VD,
    __half * __restrict__ d_z, __half * __restrict__ d_ab,
    float * __restrict__ d_g, float * __restrict__ d_beta)
{
    const int t = blockIdx.x;
    const int nth = blockDim.x;
    const __half *xrow = x + (int64_t)t * H;
    for (int c = threadIdx.x; c < VD; c += nth) { /* z GEMM as Step 5 */ }
    __half *a_buf = d_ab;
    __half *b_buf = d_ab + (int64_t)seq * nv;
    for (int c = threadIdx.x; c < nv; c += nth) {
        /* a,b dots as Step 5, store a_buf/b_buf */
    }
    __syncthreads();
    for (int c = threadIdx.x; c < nv; c += nth) {
        float a_v = __half2float(a_buf[(int64_t)t * nv + c]);
        float b_v = __half2float(b_buf[(int64_t)t * nv + c]);
        d_beta[(int64_t)t * nv + c] = sigmoid_dev(b_v);
        d_g[(int64_t)t * nv + c] = -__expf(A_log[c]) * softplus_dev(a_v + dt_bias[c]);
    }
}
```

Then add the public entry **after** the second anonymous namespace closes (`~:837`), at file scope in `lsxcommon::qwen`:

```cpp
cudaError_t RunQwenFusedLinearAttnKernel(
    cudaStream_t stream,
    const QwenLayerWeights &lw, const QwenTopology &top, int seq,
    const __half *d_normed, __half *d_out,
    __half *d_qkv, __half *d_z, __half *d_ab,
    float *d_g, float *d_beta,
    __half *d_conv, __half *d_core, __half *d_gated,
    int kv_base /* = 0 */, float *d_gd_state /* = nullptr */)
{
    if (!lw.loaded || seq <= 0) return cudaErrorInvalidValue;
    const int nk = top.linear_num_key_heads;
    const int kd = top.linear_key_head_dim;
    const int nv = top.linear_num_value_heads;
    const int vd = top.linear_value_head_dim;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;
    if (nk <= 0 || kd <= 0 || nv <= 0 || vd <= 0) return cudaErrorInvalidValue;
    cudaError_t e;

    // F1: qkv GEMM + conv1d + l2norm(q,k).
    int nthreads = 1;
    while (nthreads < kd) nthreads <<= 1;
    fused_linear_front_kernel<<<seq, nthreads, 0, stream>>>(
        d_normed, lw.qkv, lw.d_conv1d, seq, nk, kd, nv, vd,
        (lw.qkv.cols > 0 ? lw.qkv.rows : 0), lw.conv_kernel,
        kv_base, d_qkv, d_conv);
    e = cudaGetLastError(); if (e) return e;

    // F2: z + a + b + prep_gating.
    fused_linear_zab_kernel<<<seq, nthreads, 0, stream>>>(
        d_normed, lw.z, lw.a, lw.b, lw.d_A_log, lw.d_dt_bias,
        seq, lw.z.rows, vd, nv, VD, d_z, d_ab, d_g, d_beta);
    e = cudaGetLastError(); if (e) return e;

    // Tail (unchanged, mirrors RunQwenLinearAttention stages 4-6).
    {
        const __half *qsplit = d_conv;
        const __half *ksplit = d_conv + KD;
        const __half *vsplit = d_conv + (int64_t)2 * KD;
        const int r = nv / nk;
        const dim3 grid(seq, nv);
        const int threads = std::min(vd, 256);
        qwen_gated_delta_kernel<<<grid, threads, 0, stream>>>(
            qsplit, ksplit, vsplit, d_g, d_beta, d_core, seq,
            nk, kd, nv, vd, r, S, d_gd_state);
        e = cudaGetLastError(); if (e) return e;
    }
    {
        const int threads = std::min(vd, 256);
        rmsnorm_gated_kernel<<<seq * nv, threads, vd * sizeof(float), stream>>>(
            d_core, d_z, lw.d_attn_norm, d_gated, seq * nv, vd,
            top.rms_norm_eps);
        e = cudaGetLastError(); if (e) return e;
    }
    return LaunchNvfp4Gemm(lw.out, d_gated, seq, d_out, stream);
}
```

> The F1 `H` argument is passed as `lw.qkv.rows` (guard: if `qkv.cols<=0` return invalid). The `nv`/`vd`-based `S` computed above must equal `lw.qkv.cols` (assert in test).

- [ ] **Step 7: Build + run the skeleton (GREEN on 1 case)**

Add a minimal TEST_CASE to `tests/test_qwen_fused_linear_attn.cu`:

```cpp
TEST_CASE("RunQwenFusedLinearAttnKernel entry runs on a tiny layer", "[qwen][linear][fuse]")
{
    const int H = 16, nk = 2, kd = 8, nv = 2, vd = 8, seq = 2, kernel = 3;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;

    QwenTopology top;
    top.linear_num_key_heads = nk; top.linear_key_head_dim = kd;
    top.linear_num_value_heads = nv; top.linear_value_head_dim = vd;
    top.rms_norm_eps = 1e-6f;

    QwenLayerWeights lw; lw.loaded = true; lw.is_linear = true;
    FillNvfp4(lw.qkv, H, S, 0.5f); FillNvfp4(lw.z, H, VD, 0.5f);
    FillNvfp4(lw.a, H, nv, 0.5f);  FillNvfp4(lw.b, H, nv, 0.5f);
    FillNvfp4(lw.out, VD, H, 0.5f);
    std::vector<__half> conv(S * kernel, __float2half(0.25f));
    cudaMalloc(&lw.d_conv1d, conv.size() * 2);
    cudaMemcpy(lw.d_conv1d, conv.data(), conv.size() * 2, cudaMemcpyHostToDevice);
    std::vector<float> alog(nv, -0.5f), dtb(nv, 0.1f);
    cudaMalloc(&lw.d_A_log, nv * 4); cudaMemcpy(lw.d_A_log, alog.data(), nv*4, cudaMemcpyHostToDevice);
    cudaMalloc(&lw.d_dt_bias, nv * 4); cudaMemcpy(lw.d_dt_bias, dtb.data(), nv*4, cudaMemcpyHostToDevice);
    std::vector<__half> attn_norm(vd, __float2half(1.0f));
    cudaMalloc(&lw.d_attn_norm, vd*2); cudaMemcpy(lw.d_attn_norm, attn_norm.data(), vd*2, cudaMemcpyHostToDevice);

    std::vector<__half> x(seq * H);
    for (int i = 0; i < seq*H; ++i) x[i] = __float2half(0.1f + 0.03f * ((i % 5) - 2));
    __half *dx; cudaMalloc(&dx, seq*H*2); cudaMemcpy(dx, x.data(), seq*H*2, cudaMemcpyHostToDevice);
    __half *dq, *dz, *dab, *dconv, *dcore, *dgated, *dout;
    float *dg, *dbeta;
    cudaMalloc(&dq,  (size_t)seq*S*2); cudaMalloc(&dz, (size_t)seq*VD*2);
    cudaMalloc(&dab, (size_t)seq*nv*2*2); cudaMalloc(&dconv, (size_t)seq*S*2);
    cudaMalloc(&dcore, (size_t)seq*VD*2); cudaMalloc(&dgated, (size_t)seq*VD*2);
    cudaMalloc(&dout, (size_t)seq*H*2); cudaMalloc(&dg, (size_t)seq*nv*4);
    cudaMalloc(&dbeta, (size_t)seq*nv*4);

    REQUIRE(RunQwenFusedLinearAttnKernel(nullptr, lw, top, seq, dx, dout,
        dq, dz, dab, dg, dbeta, dconv, dcore, dgated) == cudaSuccess);
    cudaDeviceSynchronize();

    lw.qkv.Free(); lw.z.Free(); lw.a.Free(); lw.b.Free(); lw.out.Free();
    cudaFree(lw.d_conv1d); cudaFree(lw.d_A_log); cudaFree(lw.d_dt_bias); cudaFree(lw.d_attn_norm);
    cudaFree(dx); cudaFree(dq); cudaFree(dz); cudaFree(dab); cudaFree(dconv);
    cudaFree(dcore); cudaFree(dgated); cudaFree(dout); cudaFree(dg); cudaFree(dbeta);
}
```

Build + run. Expected: PASS (entry symbol + tail executes; the case is a smoke test — real parity next task).

- [ ] **Step 8: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/qwen_inference.h libs/lsxcommon/src/qwen_inference.cu tests/test_qwen_fused_linear_attn.cu tests/CMakeLists.txt
git commit -m "feat: linear-attn fused front kernels (F1 qkv+conv+norm, F2 z+a+b+gating)"
```

---

### Task 2: Wire generator + F1/F2 parity tests

**Files:**
- Modify: `libs/lsxcommon/src/qwen_inference.cu`
- Modify: `tests/test_qwen_fused_linear_attn.cu`

- [ ] **Step 1: Wire both `is_linear` sites**

In `RunQwenGeneration`, both `if (lw.is_linear)` branches (`qwen_inference.cu:1583` and `:1674`) currently call `RunQwenLinearAttention(stream, lw, top, seq, d_normed, d_attn_out, d_qkv, d_z, d_ab, d_gf, d_betaf, d_conv, d_core, d_gated);` — the transient site also passes `/*kv_base=*/0, /*gd_state=*/nullptr` and the persistent site passes `kv_base=cur` and its `d_gd_state` slice. Change **only the function name** to `RunQwenFusedLinearAttnKernel`, leaving all arguments in identical order. Verify the `d_gf`/`d_betaf`/`d_qkv`/`d_conv`/`d_core`/`d_gated` buffer pointers used there are the same ones the fused entry expects (they are — same layout as the split path).

- [ ] **Step 2: Add F1 parity test case**

Add to `tests/test_qwen_fused_linear_attn.cu` a case that runs seq=1 AND seq=2, computing the F1 front via the **split** kernels exactly (qkv GEMM through `RunQwenNvfp4Gemm` into `d_qkv` at base `kv_base`, then `qwen_conv1d_kernel`-equivalent host/direct conv, then l2norm) and compares to the fused `RunQwenFusedLinearAttnKernel`'s `d_conv` q/k slices. Because the exact split F1 kernels (`qwen_conv1d_kernel`, `qwen_l2norm_rows_kernel`, `qwen_nvfp4_gemm_kernel`) are anonymous-ns (not callable from the test), implement the **host reference** for F1 and compare within `1e-2`:

```cpp
// Host F1 reference for one (t, col) using HostDeq + explicit conv + l2norm.
#include <limits>
namespace {
__host__ float HostConv1d(const std::vector<__half>& hist, const std::vector<__half>& w,
                          int S, int kernel, int tt_base, int c) {
    float acc = 0.0f;
    for (int k = 0; k < kernel; ++k) {
        int tt = tt_base - k;
        if (tt < 0) continue;
        acc += __half2float(w[c*kernel + (kernel-1-k)]) * __half2float(hist[(int64_t)tt*S + c]);
    }
    return __half2float(__float2half(acc)); // full half, matches silu input precision? keep acc float
}
}
```

Build a case that, for a given seq/kv_base, produces the raw qkv rows (host dot + quantize to half), runs host conv+silu+l2norm, then runs `RunQwenFusedLinearAttnKernel(..., kv_base, nullptr)` on the same buffers prepopulated with garbage and reads `d_conv` back, asserting each q/k element within `1e-2` relative. **Abstract helper**: implement `HostLinearFront(seq, kv_base, ...)` returning host `d_conv`; call it for seq=1 and seq=2. (Note: `RunQwenFusedLinearAttnKernel` writes raw rows to `d_qkv` at `kv_base` and reads history recursively — for the test kv_base=0, transient semantics.)

- [ ] **Step 3: Build + run F1 parity**

Run: `& $cmake --build build --config Debug --target test_qwen_fused_linear_attn` then `& "build/bin/Debug/test_qwen_fused_linear_attn.exe"`.
Expected: all cases PASS (F1 host-parity + smoke entry case).

- [ ] **Step 4: Add full-layer parity test case**

Compare split `RunQwenLinearAttention` vs fused `RunQwenFusedLinearAttnKernel` end-to-end on the same tiny layer (reuse the Task 1 setup builders via a helper), for seq=1 and seq=2, `kv_base=0`, no gd_state. Read `d_out` from both, assert each element within `1e-2` relative. Because `RunQwenLinearAttention` is declared in the header, the test can call it directly as the reference.

- [ ] **Step 5: Build + run full parity**

Run: `& "build/bin/Debug/test_qwen_fused_linear_attn.exe"`. Expected: all PASS.

- [ ] **Step 6: Build + run `test_qwen_model` (generator now uses fused entry)**

Run: `& $cmake --build build --config Debug --target test_qwen_model` then `& "build/bin/Debug/test_qwen_model.exe"`.
Expected: PASS (1 case, 13 assertions; real-model selfcheck SKIPs without a checkpoint).
Then `& $cmake --build build --config Debug --target test_qwen_nvfp4_tile_gemm test_qwen_fused_decode_attn test_qwen_fused_mlp` and run each — expected PASS (unchanged paths unaffected).

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/src/qwen_inference.cu tests/test_qwen_fused_linear_attn.cu
git commit -m "feat: wire fused linear-attn front into generator + F1 and full-layer parity"
```

---

### Task 3: Full verification

**Files:** none (build/test only).

- [ ] **Step 1: Full build**

Run: `& $cmake --build build --config Debug`
Expected: completes with no errors (warnings OK).

- [ ] **Step 2: Qwen ctest suite**

Run: `& $ctest --preset windows-debug --test-dir build --output-on-failure --tests-regex "qwen"`
Expected: all `qwen*` tests pass (including `test_qwen_fused_linear_attn`).

- [ ] **Step 3: Confirm launch-count reduction**

Grep the fused entry site: `RunQwenFusedLinearAttnKernel` used in both generator branches, `RunQwenLinearAttention` still present only as reference (header decl + definition + tests). Confirm no dead `RunQwenLinearAttention` call remains in the generator.

- [ ] **Step 4: Commit (if any changes)**

```bash
git add -u
git commit -m "refactor: verify linear-attn fused front in full Qwen suite"  # only if modified files, else skip
```

- [ ] **Step 5: Push**

```bash
git push origin main
```

---

## Self-review notes

- `RunQwenFusedLinearAttnKernel` signature matches `RunQwenLinearAttention` exactly (checked against `qwen_inference.h:89-96`), so reusing the generator argument list is safe.
- F1 handles `kd<=256` (split enforces `kd>256 -> invalid` at `:1221`); `blockDim=pow2(kd)` matches l2norm's `__shared__ float svals[256]` constraint.
- The tail reuses `qwen_gated_delta_kernel`/`rmsnorm_gated_kernel`/`LaunchNvfp4Gemm` verbatim — the entrance only replaces stages 1-4-with-norm. Buffer pointers (`qsplit/ksplit/vsplit`) are recomputed from the same `S` constant.
- Numeric contract: FP32 accumulate, free reduction order, `1e-2` vs split (per AGENTS.md Qwen contract).
