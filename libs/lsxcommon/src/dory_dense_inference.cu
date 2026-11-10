// NVIDIA-Nemotron-Labs-Dory1.0-4B runtime (Model Ingestion Protocol stage 5).
//
// The Dory graph is a looped nGPT decoder: every physical layer is ONE mixer
// (attention XOR mlp) whose output is combined with the residual stream by a
// spherical interpolation, and the middle block is applied kRecurrentLoops
// times with the input block's output re-injected. There is no RMSNorm and no
// final norm anywhere.
//
// One stream, no per-token cudaDeviceSynchronize. Projections go through the
// model-neutral NVFP4 layer (lsx_nvfp4::RunLsxNvfp4Gemm); RoPE, the fused
// q/k/v split and the embedding gather are the model-neutral dense/kernels
// primitives. The family-local kernels are the four nGPT normalizations and the
// sliding-window causal GQA attention (the neutral dense attention has no
// window predicate). All block reductions are shared-memory trees behind
// __syncthreads(): no atomics feed final_logits, so every run is bit-identical.

#include "lsxcommon/dory_dense_inference.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "lsxcommon/dory_dense_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_dense_inference.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

#include "lsx_kernels_common.cuh"
#include "lsx_kernels_math.cuh"

namespace lsxcommon {
namespace dory {

using lsx_nvfp4::RunLsxNvfp4Gemm;

namespace {

constexpr int kThreads = 128;
constexpr int kNormThreads = 256;
constexpr int kAttnSplitMax = 32;
constexpr int kAttnSplitKeys = 64;

// Deterministic block-wide sum: warp shuffles, then one warp over the per-warp
// partials, then a broadcast through shared memory. The order is fixed, so the
// result does not depend on the launch geometry.
__device__ __forceinline__ float BlockSum(float v, float *warp_sums)
{
    const unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1)
        v = __fadd_rn(v, __shfl_down_sync(mask, v, off));
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    const int warps = (static_cast<int>(blockDim.x) + 31) >> 5;
    float t = (static_cast<int>(threadIdx.x) < warps) ? warp_sums[threadIdx.x]
                                                      : 0.f;
    if (warp == 0) {
        for (int off = 16; off > 0; off >>= 1)
            t = __fadd_rn(t, __shfl_down_sync(mask, t, off));
        if (lane == 0) warp_sums[0] = t;
    }
    __syncthreads();
    return warp_sums[0];
}

// ── nGPT normalization kernels ─────────────────────────────────────────────

// out[r] = x[r] * rsqrt(sum x[r]^2 + eps); with `b` non-null the input is
// x[r]+b[r] (the loop reinjection combine).
__global__ void justnorm_kernel(const __half *__restrict__ x,
                                const __half *__restrict__ b,
                                __half *__restrict__ out, int rows, int n,
                                float eps)
{
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    __shared__ float warp_sums[32];
    const int64_t base = static_cast<int64_t>(row) * n;
    float s = 0.f;
    for (int i = static_cast<int>(threadIdx.x); i < n;
         i += static_cast<int>(blockDim.x)) {
        float v = __half2float(x[base + i]);
        if (b != nullptr) v = __fadd_rn(v, __half2float(b[base + i]));
        s = __fmaf_rn(v, v, s);
    }
    const float inv = rsqrtf(__fadd_rn(BlockSum(s, warp_sums), eps));
    for (int i = static_cast<int>(threadIdx.x); i < n;
         i += static_cast<int>(blockDim.x)) {
        float v = __half2float(x[base + i]);
        if (b != nullptr) v = __fadd_rn(v, __half2float(b[base + i]));
        out[base + i] = __float2half(__fmul_rn(v, inv));
    }
}

// out = justnorm( justnorm(a) + lr*(justnorm(b) - justnorm(a)) ), lr per-dim.
__global__ void slerp_alpha_kernel(const __half *__restrict__ a,
                                   const __half *__restrict__ b,
                                   const float *__restrict__ alpha,
                                   __half *__restrict__ out, int rows, int n,
                                   float eps)
{
    extern __shared__ float y[];   // n floats
    __shared__ float warp_sums[32];
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const int64_t base = static_cast<int64_t>(row) * n;

    float sa = 0.f, sb = 0.f;
    for (int i = static_cast<int>(threadIdx.x); i < n;
         i += static_cast<int>(blockDim.x)) {
        const float va = __half2float(a[base + i]);
        const float vb = __half2float(b[base + i]);
        sa = __fmaf_rn(va, va, sa);
        sb = __fmaf_rn(vb, vb, sb);
    }
    const float na = BlockSum(sa, warp_sums);
    const float nb = BlockSum(sb, warp_sums);
    const float ia = rsqrtf(__fadd_rn(na, eps));
    const float ib = rsqrtf(__fadd_rn(nb, eps));

    float sy = 0.f;
    for (int i = static_cast<int>(threadIdx.x); i < n;
         i += static_cast<int>(blockDim.x)) {
        const float ya = __fmul_rn(__half2float(a[base + i]), ia);
        const float yb = __fmul_rn(__half2float(b[base + i]), ib);
        const float yi = __fmaf_rn(alpha[i], __fsub_rn(yb, ya), ya);
        y[i] = yi;
        sy = __fmaf_rn(yi, yi, sy);
    }
    const float inv = rsqrtf(__fadd_rn(BlockSum(sy, warp_sums), eps));
    for (int i = static_cast<int>(threadIdx.x); i < n;
         i += static_cast<int>(blockDim.x))
        out[base + i] = __float2half(__fmul_rn(y[i], inv));
}

// q: per-head justnorm then the squared nGPT q scale; k: per-head justnorm.
// One block per (row, head) so head_dim is always reduced on its own.
__global__ void qk_norm_kernel(__half *__restrict__ q, __half *__restrict__ k,
                               const float *__restrict__ sqk2, int rows, int nh,
                               int nkv, int hd, float eps)
{
    const int heads = nh + nkv;
    const int row = static_cast<int>(blockIdx.x) / heads;
    const int slot = static_cast<int>(blockIdx.x) % heads;
    if (row >= rows) return;
    __shared__ float warp_sums[32];
    const bool is_q = slot < nh;
    const int head = is_q ? slot : slot - nh;
    const int width = is_q ? nh : nkv;
    __half *ptr = is_q ? q : k;
    const int64_t base =
        (static_cast<int64_t>(row) * width + head) * static_cast<int64_t>(hd);

    float s = 0.f;
    for (int d = static_cast<int>(threadIdx.x); d < hd;
         d += static_cast<int>(blockDim.x)) {
        const float v = __half2float(ptr[base + d]);
        s = __fmaf_rn(v, v, s);
    }
    const float inv = rsqrtf(__fadd_rn(BlockSum(s, warp_sums), eps));
    for (int d = static_cast<int>(threadIdx.x); d < hd;
         d += static_cast<int>(blockDim.x)) {
        float v = __fmul_rn(__half2float(ptr[base + d]), inv);
        if (is_q) v = __fmul_rn(v, sqk2[head * hd + d]);
        ptr[base + d] = __float2half(v);
    }
}

// out = justnorm( attn * sigmoid(gate * sgate2) ) over the whole head stack.
__global__ void attn_gate_norm_kernel(const __half *__restrict__ attn,
                                      const __half *__restrict__ gate,
                                      const float *__restrict__ sgate2,
                                      __half *__restrict__ out, int rows,
                                      int n, float eps)
{
    extern __shared__ float t[];   // n floats
    __shared__ float warp_sums[32];
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const int64_t base = static_cast<int64_t>(row) * n;
    float s = 0.f;
    for (int i = static_cast<int>(threadIdx.x); i < n;
         i += static_cast<int>(blockDim.x)) {
        const float a = __half2float(attn[base + i]);
        const float g = __half2float(gate[base + i]);
        const float sig = __fdiv_rn(1.f, __fadd_rn(1.f, __expf(-__fmul_rn(g, sgate2[i]))));
        const float v = __fmul_rn(a, sig);
        t[i] = v;
        s = __fmaf_rn(v, v, s);
    }
    const float inv = rsqrtf(__fadd_rn(BlockSum(s, warp_sums), eps));
    for (int i = static_cast<int>(threadIdx.x); i < n;
         i += static_cast<int>(blockDim.x))
        out[base + i] = __float2half(__fmul_rn(t[i], inv));
}

// out[r][j] = justnorm_j( silu(gu[r][j]*suv_gate[j]) * gu[r][I+j]*suv_up[j] )
__global__ void suv_swiglu_norm_kernel(const __half *__restrict__ gu,
                                       const float *__restrict__ suv_gate,
                                       const float *__restrict__ suv_up,
                                       __half *__restrict__ out, int rows,
                                       int I, float eps)
{
    extern __shared__ float t[];   // I floats
    __shared__ float warp_sums[32];
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const int64_t base = static_cast<int64_t>(row) * 2 * I;
    float s = 0.f;
    for (int j = static_cast<int>(threadIdx.x); j < I;
         j += static_cast<int>(blockDim.x)) {
        const float g = __fmul_rn(__half2float(gu[base + j]), suv_gate[j]);
        const float u = __fmul_rn(__half2float(gu[base + I + j]), suv_up[j]);
        const float silu = __fdiv_rn(g, __fadd_rn(1.f, __expf(-g)));
        const float m = __fmul_rn(silu, u);
        t[j] = m;
        s = __fmaf_rn(m, m, s);
    }
    const float inv = rsqrtf(__fadd_rn(BlockSum(s, warp_sums), eps));
    const int64_t obase = static_cast<int64_t>(row) * I;
    for (int j = static_cast<int>(threadIdx.x); j < I;
         j += static_cast<int>(blockDim.x))
        out[obase + j] = __float2half(__fmul_rn(t[j], inv));
}

// ── Sliding-window causal GQA attention ────────────────────────────────────

// One block per (query row, head). `window <= 0` = full attention. The online
// softmax is the neutral dense kernel's (shared-memory tree reductions, one
// rescale per key block), only the key range is clipped to
// [bound-window+1, bound].
__global__ void dory_attn_kernel(const __half *__restrict__ q,
                                 const __half *__restrict__ k,
                                 const __half *__restrict__ v,
                                 __half *__restrict__ out, int seq, int nh,
                                 int nkv, int hd, float scale, int kv_base,
                                 int window)
{
    const int t = static_cast<int>(blockIdx.x) / nh;
    const int h = static_cast<int>(blockIdx.x) % nh;
    if (t >= seq) return;
    const int kvh = h / (nh / nkv);
    const int tid = static_cast<int>(threadIdx.x);

    extern __shared__ float smem[];
    float *q_s = smem;              // [hd]
    float *o_s = smem + hd;         // [hd]
    float *sc = smem + 2 * hd;      // [blockDim]
    float *rd = sc + blockDim.x;    // [blockDim]

    const __half *qrow = q + (static_cast<int64_t>(t) * nh + h) * hd;
    for (int d = tid; d < hd; d += static_cast<int>(blockDim.x))
        q_s[d] = __half2float(qrow[d]);
    for (int d = tid; d < hd; d += static_cast<int>(blockDim.x)) o_s[d] = 0.f;
    __syncthreads();

    const int bound = kv_base + t;   // inclusive last attended key
    const int first = (window > 0) ? max(0, bound - window + 1) : 0;
    float m = -CUDART_INF_F;
    float l = 0.f;
    for (int start = first; start <= bound; start += static_cast<int>(blockDim.x)) {
        const int cnt = min(static_cast<int>(blockDim.x), bound + 1 - start);
        float score = -CUDART_INF_F;
        if (tid < cnt) {
            const __half *krow =
                k + (static_cast<int64_t>(start + tid) * nkv + kvh) * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; ++d)
                dot = __fmaf_rn(q_s[d], __half2float(krow[d]), dot);
            score = __fmul_rn(dot, scale);
        }
        sc[tid] = score;
        __syncthreads();
        for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
            if (tid < s && sc[tid] < sc[tid + s]) sc[tid] = sc[tid + s];
            __syncthreads();
        }
        const float m_new = sc[0];
        __syncthreads();
        const float e = (tid < cnt) ? __expf(__fsub_rn(score, m_new)) : 0.f;
        sc[tid] = e;
        rd[tid] = e;
        __syncthreads();
        for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
            if (tid < s) rd[tid] = __fadd_rn(rd[tid], rd[tid + s]);
            __syncthreads();
        }
        const float l_new = rd[0];
        __syncthreads();
        const float rescale =
            (m == -CUDART_INF_F) ? 0.f : __expf(__fsub_rn(m, m_new));
        for (int d = tid; d < hd; d += static_cast<int>(blockDim.x)) {
            float acc = __fmul_rn(o_s[d], rescale);
            for (int j = 0; j < cnt; ++j) {
                const __half *vrow =
                    v + (static_cast<int64_t>(start + j) * nkv + kvh) * hd;
                acc = __fmaf_rn(sc[j], __half2float(vrow[d]), acc);
            }
            o_s[d] = acc;
        }
        m = m_new;
        l = __fmaf_rn(l, rescale, l_new);
        __syncthreads();
    }

    const float inv = (l > 0.f) ? __frcp_rn(l) : 0.f;
    __half *orow = out + (static_cast<int64_t>(t) * nh + h) * hd;
    for (int d = tid; d < hd; d += static_cast<int>(blockDim.x))
        orow[d] = __float2half(__fmul_rn(o_s[d], inv));
}

// Split-KV variant: the causal+window range [first, bound] is partitioned into
// `splits` contiguous chunks with fixed boundaries; each block writes one
// (m, l, o) partial. Fixed boundaries and a fixed merge order keep the decode
// run-to-run deterministic while giving the 1-row decode enough blocks.
__global__ void dory_attn_split_kernel(const __half *__restrict__ q,
                                       const __half *__restrict__ k,
                                       const __half *__restrict__ v,
                                       float *__restrict__ pm,
                                       float *__restrict__ pl,
                                       float *__restrict__ po, int seq, int nh,
                                       int nkv, int hd, float scale,
                                       int kv_base, int window, int splits)
{
    const int t = static_cast<int>(blockIdx.x) / nh;
    const int h = static_cast<int>(blockIdx.x) % nh;
    const int sv = static_cast<int>(blockIdx.y);
    if (t >= seq) return;
    const int kvh = h / (nh / nkv);
    const int tid = static_cast<int>(threadIdx.x);

    extern __shared__ float smem[];
    float *q_s = smem;
    float *o_s = smem + hd;
    float *sc = smem + 2 * hd;
    float *rd = sc + blockDim.x;

    const int bound = kv_base + t;
    const int first = (window > 0) ? max(0, bound - window + 1) : 0;
    const int keys = bound + 1 - first;
    const int n0 = first + static_cast<int>(static_cast<int64_t>(keys) * sv / splits);
    const int n1 =
        first + static_cast<int>(static_cast<int64_t>(keys) * (sv + 1) / splits);
    const int64_t slot = (static_cast<int64_t>(sv) * seq + t) * nh + h;

    const __half *qrow = q + (static_cast<int64_t>(t) * nh + h) * hd;
    for (int d = tid; d < hd; d += static_cast<int>(blockDim.x))
        q_s[d] = __half2float(qrow[d]);
    for (int d = tid; d < hd; d += static_cast<int>(blockDim.x)) o_s[d] = 0.f;
    __syncthreads();

    float m = -CUDART_INF_F;
    float l = 0.f;
    for (int start = n0; start < n1; start += static_cast<int>(blockDim.x)) {
        const int cnt = min(static_cast<int>(blockDim.x), n1 - start);
        float score = -CUDART_INF_F;
        if (tid < cnt) {
            const __half *krow =
                k + (static_cast<int64_t>(start + tid) * nkv + kvh) * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; ++d)
                dot = __fmaf_rn(q_s[d], __half2float(krow[d]), dot);
            score = __fmul_rn(dot, scale);
        }
        sc[tid] = score;
        __syncthreads();
        for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
            if (tid < s && sc[tid] < sc[tid + s]) sc[tid] = sc[tid + s];
            __syncthreads();
        }
        const float m_new = sc[0];
        __syncthreads();
        const float e = (tid < cnt) ? __expf(__fsub_rn(score, m_new)) : 0.f;
        sc[tid] = e;
        rd[tid] = e;
        __syncthreads();
        for (int s = static_cast<int>(blockDim.x) / 2; s > 0; s >>= 1) {
            if (tid < s) rd[tid] = __fadd_rn(rd[tid], rd[tid + s]);
            __syncthreads();
        }
        const float l_new = rd[0];
        __syncthreads();
        const float rescale =
            (m == -CUDART_INF_F) ? 0.f : __expf(__fsub_rn(m, m_new));
        for (int d = tid; d < hd; d += static_cast<int>(blockDim.x)) {
            float acc = __fmul_rn(o_s[d], rescale);
            for (int j = 0; j < cnt; ++j) {
                const __half *vrow =
                    v + (static_cast<int64_t>(start + j) * nkv + kvh) * hd;
                acc = __fmaf_rn(sc[j], __half2float(vrow[d]), acc);
            }
            o_s[d] = acc;
        }
        m = m_new;
        l = __fmaf_rn(l, rescale, l_new);
        __syncthreads();
    }

    if (tid == 0) {
        pm[slot] = m;
        pl[slot] = l;
    }
    for (int d = tid; d < hd; d += static_cast<int>(blockDim.x))
        po[slot * hd + d] = o_s[d];
}

// Merges the `splits` partials per (row, head) in fixed split order.
__global__ void dory_attn_combine_kernel(const float *__restrict__ pm,
                                         const float *__restrict__ pl,
                                         const float *__restrict__ po,
                                         __half *__restrict__ out, int seq,
                                         int nh, int hd, int splits)
{
    const int t = static_cast<int>(blockIdx.x) / nh;
    const int h = static_cast<int>(blockIdx.x) % nh;
    if (t >= seq) return;
    const int tid = static_cast<int>(threadIdx.x);
    const int64_t base = static_cast<int64_t>(t) * nh + h;

    float m = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) {
        const float ms = pm[static_cast<int64_t>(s) * seq * nh + base];
        m = fmaxf(m, ms);
    }
    float l = 0.f;
    for (int s = 0; s < splits; ++s) {
        const int64_t slot = static_cast<int64_t>(s) * seq * nh + base;
        l = __fadd_rn(l, __fmul_rn(pl[slot], __expf(__fsub_rn(pm[slot], m))));
    }
    __half *orow = out + base * hd;
    for (int d = tid; d < hd; d += static_cast<int>(blockDim.x)) {
        float a = 0.f;
        for (int s = 0; s < splits; ++s) {
            const int64_t slot = static_cast<int64_t>(s) * seq * nh + base;
            a = __fadd_rn(a, __fmul_rn(po[slot * hd + d],
                                       __expf(__fsub_rn(pm[slot], m))));
        }
        orow[d] = __float2half((l > 0.f) ? __fdiv_rn(a, l) : 0.f);
    }
}

// logits[v] = half2float(y[v]) * logit_scale[v]
__global__ void logit_scale_kernel(float *__restrict__ logits,
                                   const __half *__restrict__ y,
                                   const float *__restrict__ logit_scale,
                                   int vocab)
{
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
    if (i >= vocab) return;
    logits[i] = __fmul_rn(__half2float(y[i]), logit_scale[i]);
}

// ── Workspace ──────────────────────────────────────────────────────────────

struct Workspace {
    __half *x = nullptr;        // residual stream
    __half *input_h = nullptr;  // input block output (re-injected each loop)
    __half *state = nullptr;    // recurrent block state between loops
    __half *proj = nullptr;     // mixer projection output
    __half *qkv = nullptr;      // fused q|k|v
    __half *q = nullptr;
    __half *k = nullptr;
    __half *v = nullptr;
    __half *gate = nullptr;     // g_proj output
    __half *attn = nullptr;     // attention out (pre-gate) then gated+normed
    __half *gate_up = nullptr;  // fused gate|up
    __half *act = nullptr;      // swiglu output
    __half *logits_h = nullptr;
    float *logits = nullptr;
    int32_t *ids = nullptr;
    int32_t *pos = nullptr;
    int32_t *tok = nullptr;
    float *maxv = nullptr;
    float *attn_part = nullptr;
    int max_seq = 0;

    cudaError_t Alloc(const DoryTopology &top, int max_seq_);
    void Free();
};

void Workspace::Free()
{
    cudaFree(x); cudaFree(input_h); cudaFree(state); cudaFree(proj);
    cudaFree(qkv); cudaFree(q); cudaFree(k); cudaFree(v);
    cudaFree(gate); cudaFree(attn); cudaFree(gate_up); cudaFree(act);
    cudaFree(logits_h); cudaFree(logits);
    cudaFree(ids); cudaFree(pos); cudaFree(tok); cudaFree(maxv);
    cudaFree(attn_part);
    *this = Workspace{};
}

cudaError_t Workspace::Alloc(const DoryTopology &top, int max_seq_)
{
    max_seq = max_seq_;
    const int H = top.hidden_size;
    const int I = top.intermediate_size;
    const int V = top.vocab_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;
    const int Q = NH * HD;
    const int KV = NKV * HD;
    const int64_t S = max_seq;

    auto alloc = [](void **p, int64_t bytes) -> cudaError_t {
        return cudaMalloc(p, static_cast<size_t>(bytes));
    };
    cudaError_t e = cudaSuccess;
    if ((e = alloc(reinterpret_cast<void **>(&x), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&input_h), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&state), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&proj), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&qkv), S * (Q + 2 * KV) * 2)))
        return e;
    if ((e = alloc(reinterpret_cast<void **>(&q), S * Q * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&k), S * KV * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&v), S * KV * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&gate), S * Q * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&attn), S * Q * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&gate_up), S * 2 * I * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&act), S * I * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&logits_h),
                   static_cast<int64_t>(V) * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&logits),
                   static_cast<int64_t>(V) * 4))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&ids), S * sizeof(int32_t))))
        return e;
    if ((e = alloc(reinterpret_cast<void **>(&pos), S * sizeof(int32_t))))
        return e;
    if ((e = alloc(reinterpret_cast<void **>(&tok), sizeof(int32_t)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&maxv), sizeof(float)))) return e;
    const int64_t sp_seq = std::min<int64_t>(S, 8);
    if ((e = alloc(reinterpret_cast<void **>(&attn_part),
                   static_cast<int64_t>(kAttnSplitMax) * sp_seq * NH * (HD + 2) *
                       sizeof(float))))
        return e;
    return cudaSuccess;
}

// Shared-memory bytes for one of the per-row norm kernels.
size_t NormSmemBytes(int n) { return static_cast<size_t>(n) * sizeof(float); }

// ── Launchers ──────────────────────────────────────────────────────────────

cudaError_t JustNorm(cudaStream_t stream, __half *out, const __half *x,
                     const __half *b, int rows, int n, float eps)
{
    justnorm_kernel<<<rows, kNormThreads, 0, stream>>>(x, b, out, rows, n, eps);
    return cudaGetLastError();
}

cudaError_t SlerpAlpha(cudaStream_t stream, __half *out, const __half *a,
                       const __half *b, const float *alpha, int rows, int n,
                       float eps)
{
    slerp_alpha_kernel<<<rows, kNormThreads, NormSmemBytes(n), stream>>>(
        a, b, alpha, out, rows, n, eps);
    return cudaGetLastError();
}

cudaError_t QkNorm(cudaStream_t stream, __half *q, __half *k,
                   const float *sqk2, int rows, int nh, int nkv, int hd,
                   float eps)
{
    qk_norm_kernel<<<rows * (nh + nkv), kThreads, 0, stream>>>(
        q, k, sqk2, rows, nh, nkv, hd, eps);
    return cudaGetLastError();
}

cudaError_t AttnGateNorm(cudaStream_t stream, __half *out, const __half *attn,
                         const __half *gate, const float *sgate2, int rows,
                         int n, float eps)
{
    attn_gate_norm_kernel<<<rows, kNormThreads, NormSmemBytes(n), stream>>>(
        attn, gate, sgate2, out, rows, n, eps);
    return cudaGetLastError();
}

cudaError_t SuvSwiGLUNorm(cudaStream_t stream, __half *out, const __half *gu,
                          const float *suv_gate, const float *suv_up, int rows,
                          int I, float eps)
{
    suv_swiglu_norm_kernel<<<rows, kNormThreads, NormSmemBytes(I), stream>>>(
        gu, suv_gate, suv_up, out, rows, I, eps);
    return cudaGetLastError();
}

cudaError_t Attention(cudaStream_t stream, const __half *q, const __half *k,
                      const __half *v, __half *out, float *partials, int seq,
                      int nh, int nkv, int hd, float scale, int kv_base,
                      int window)
{
    const int bound = kv_base + seq - 1;
    const int first = (window > 0) ? std::max(0, bound - window + 1) : 0;
    const int keys = bound + 1 - first;
    // Split only when there are few query rows (decode) and the range pays for
    // a second kernel launch.
    const int splits =
        std::min(kAttnSplitMax, std::max(1, keys / kAttnSplitKeys));
    if (partials != nullptr && splits > 1 && seq * nh < splits * 4) {
        const int64_t slots = static_cast<int64_t>(splits) * seq * nh;
        const size_t pm_b = static_cast<size_t>(slots) * sizeof(float);
        const size_t po_b = static_cast<size_t>(slots) * hd * sizeof(float);
        float *pm = partials;
        float *pl = pm + slots;
        float *po = pl + slots;
        (void)pm_b;
        (void)po_b;
        const size_t smem = static_cast<size_t>(2 * hd + 2 * kThreads) *
                            sizeof(float);
        dim3 grid(seq * nh, splits);
        dory_attn_split_kernel<<<grid, kThreads, smem, stream>>>(
            q, k, v, pm, pl, po, seq, nh, nkv, hd, scale, kv_base, window,
            splits);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        dory_attn_combine_kernel<<<seq * nh, kThreads, 0, stream>>>(
            pm, pl, po, out, seq, nh, hd, splits);
        return cudaGetLastError();
    }
    const size_t smem =
        static_cast<size_t>(2 * hd + 2 * kThreads) * sizeof(float);
    dory_attn_kernel<<<seq * nh, kThreads, smem, stream>>>(
        q, k, v, out, seq, nh, nkv, hd, scale, kv_base, window);
    return cudaGetLastError();
}

} // namespace

// ── Testable kernel wrappers ───────────────────────────────────────────────

cudaError_t RunDoryJustNormKernel(cudaStream_t stream, __half *out,
                                  const __half *x, int rows, int n, float eps)
{
    return JustNorm(stream, out, x, nullptr, rows, n, eps);
}

cudaError_t RunDoryJustNormSumKernel(cudaStream_t stream, __half *out,
                                     const __half *a, const __half *b, int rows,
                                     int n, float eps)
{
    return JustNorm(stream, out, a, b, rows, n, eps);
}

cudaError_t RunDorySlerpAlphaKernel(cudaStream_t stream, __half *out,
                                    const __half *a, const __half *b,
                                    const float *alpha, int rows, int n,
                                    float eps)
{
    return SlerpAlpha(stream, out, a, b, alpha, rows, n, eps);
}

cudaError_t RunDoryQkNormKernel(cudaStream_t stream, __half *q, __half *k,
                                const float *sqk2, int rows, int nh, int nkv,
                                int hd, float eps)
{
    return QkNorm(stream, q, k, sqk2, rows, nh, nkv, hd, eps);
}

cudaError_t RunDoryAttnGateNormKernel(cudaStream_t stream, __half *out,
                                      const __half *attn, const __half *gate,
                                      const float *sgate2, int rows, int q_size,
                                      float eps)
{
    return AttnGateNorm(stream, out, attn, gate, sgate2, rows, q_size, eps);
}

cudaError_t RunDorySuvSwiGLUNormKernel(cudaStream_t stream, __half *out,
                                       const __half *gu, const float *suv_gate,
                                       const float *suv_up, int rows, int I,
                                       float eps)
{
    return SuvSwiGLUNorm(stream, out, gu, suv_gate, suv_up, rows, I, eps);
}

cudaError_t RunDoryAttentionKernel(cudaStream_t stream, const __half *q,
                                   const __half *k, const __half *v,
                                   __half *out, float *partials, int seq,
                                   int nh, int nkv, int hd, float scale,
                                   int kv_base, int window)
{
    return Attention(stream, q, k, v, out, partials, seq, nh, nkv, hd, scale,
                     kv_base, window);
}

cudaError_t RunDoryLogitScaleKernel(cudaStream_t stream, float *logits,
                                    const __half *y, const float *logit_scale,
                                    int vocab)
{
    const int blocks = (vocab + kThreads - 1) / kThreads;
    logit_scale_kernel<<<blocks, kThreads, 0, stream>>>(logits, y, logit_scale,
                                                        vocab);
    return cudaGetLastError();
}

// ── KV cache ───────────────────────────────────────────────────────────────

void DoryKvCache::Free()
{
    cudaFree(d_k);
    cudaFree(d_v);
    d_k = nullptr;
    d_v = nullptr;
    last_tok = -1;
    kv_rows = 0;
    max_total = 0;
    cache_bytes = 0;
}

cudaError_t DoryKvCache::Allocate(const DoryPinnedWeights &pw, int max_total_,
                                  cudaStream_t stream)
{
    Free();
    if (pw.kv_slots <= 0) return cudaErrorInvalidValue;
    const int64_t row = static_cast<int64_t>(pw.top.num_key_value_heads) *
                        pw.top.head_dim;
    const int64_t bytes =
        static_cast<int64_t>(pw.kv_slots) * max_total_ * row * sizeof(__half);
    cudaError_t e =
        cudaMalloc(reinterpret_cast<void **>(&d_k), static_cast<size_t>(bytes));
    if (e != cudaSuccess) return e;
    e = cudaMalloc(reinterpret_cast<void **>(&d_v), static_cast<size_t>(bytes));
    if (e != cudaSuccess) {
        Free();
        return e;
    }
    e = cudaMemsetAsync(d_k, 0, static_cast<size_t>(bytes), stream);
    if (e == cudaSuccess)
        e = cudaMemsetAsync(d_v, 0, static_cast<size_t>(bytes), stream);
    if (e != cudaSuccess) {
        Free();
        return e;
    }
    max_total = max_total_;
    cache_bytes = bytes * 2;
    return cudaSuccess;
}

cudaError_t DoryKvCache::EnsureCapacity(const DoryPinnedWeights &pw,
                                        int needed, cudaStream_t stream)
{
    if (d_k && needed <= max_total) return cudaSuccess;
    const int new_total = std::max(needed, max_total * 2);
    const int64_t row = static_cast<int64_t>(pw.top.num_key_value_heads) *
                        pw.top.head_dim;
    const int rows = kv_rows;
    __half *old_k = d_k;
    __half *old_v = d_v;
    const int old_total = max_total;
    d_k = nullptr;
    d_v = nullptr;
    max_total = 0;
    cache_bytes = 0;
    cudaError_t e = Allocate(pw, new_total, stream);
    if (e != cudaSuccess) {
        cudaFree(old_k);
        cudaFree(old_v);
        return e;
    }
    if (rows > 0 && old_k) {
        const size_t copy_bytes = static_cast<size_t>(rows) * row * sizeof(__half);
        const int64_t old_stride = static_cast<int64_t>(old_total) * row;
        const int64_t new_stride = static_cast<int64_t>(new_total) * row;
        for (int s = 0; s < pw.kv_slots; ++s) {
            e = cudaMemcpyAsync(d_k + s * new_stride, old_k + s * old_stride,
                                copy_bytes, cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) break;
            e = cudaMemcpyAsync(d_v + s * new_stride, old_v + s * old_stride,
                                copy_bytes, cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) break;
        }
    }
    cudaFree(old_k);
    cudaFree(old_v);
    return e;
}

// ── Layer / generation ─────────────────────────────────────────────────────

namespace {

// Sliding-window mask: full attention layers get window <= 0.
int LayerWindow(const DoryTopology &top, int layer)
{
    if (frozen::LayerIsFullAttention(layer)) return 0;
    return top.sliding_window;
}

double LayerRopeTheta(const DoryTopology &top, int layer)
{
    return frozen::LayerIsFullAttention(layer) ? top.rope_theta_full
                                               : top.rope_theta_sliding;
}

cudaError_t RunSlerp(cudaStream_t stream, const DoryTopology &top,
                     const DoryLayerWeights &lw, Workspace &ws, int seq)
{
    return SlerpAlpha(stream, ws.x, ws.x, ws.proj, lw.d_alpha, seq,
                      top.hidden_size, static_cast<float>(frozen::kJustNormEps));
}

cudaError_t RunAttentionMixer(cudaStream_t stream, const DoryPinnedWeights &pw,
                              const DoryLayerWeights &lw, int layer, int seq,
                              int kv_base, const int32_t *d_pos, Workspace &ws,
                              __half *cache_k, __half *cache_v)
{
    const auto &top = pw.top;
    const int H = top.hidden_size;
    const int I = top.intermediate_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;
    const int Q = NH * HD;
    const int KV = NKV * HD;
    const float eps = static_cast<float>(frozen::kJustNormEps);
    (void)I;

    cudaError_t e = RunLsxNvfp4Gemm(stream, ws.x, lw.qkv.d_packed,
                                    lw.qkv.d_scales, lw.qkv.global_scale,
                                    ws.qkv, seq, H, Q + 2 * KV);
    if (e != cudaSuccess) return e;
    if (lw.qkv_res.d_packed) {
        e = RunLsxNvfp4Gemm(stream, ws.x, lw.qkv_res.d_packed,
                            lw.qkv_res.d_scales, lw.qkv_res.global_scale,
                            ws.qkv, seq, H, Q + 2 * KV, true);
        if (e != cudaSuccess) return e;
    }
    e = RunLsxNvfp4Gemm(stream, ws.x, lw.g.d_packed, lw.g.d_scales,
                        lw.g.global_scale, ws.gate, seq, H, Q);
    if (e != cudaSuccess) return e;
    if (lw.g_res.d_packed) {
        e = RunLsxNvfp4Gemm(stream, ws.x, lw.g_res.d_packed, lw.g_res.d_scales,
                            lw.g_res.global_scale, ws.gate, seq, H, Q, true);
        if (e != cudaSuccess) return e;
    }

    e = dense::RunDenseSplitQKVKernel(stream, ws.qkv, ws.q, ws.k, ws.v, seq,
                                      Q, KV);
    if (e != cudaSuccess) return e;
    e = dense::RunDenseRopeKernel(stream, ws.q, ws.k, d_pos, seq, NH, NKV, HD,
                                  LayerRopeTheta(top, layer));
    if (e != cudaSuccess) return e;
    e = QkNorm(stream, ws.q, ws.k, lw.d_sqk2, seq, NH, NKV, HD, eps);
    if (e != cudaSuccess) return e;

    const int64_t kv_row = static_cast<int64_t>(KV);
    e = cudaMemcpyAsync(cache_k + static_cast<int64_t>(kv_base) * kv_row, ws.k,
                        static_cast<size_t>(seq) * kv_row * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(cache_v + static_cast<int64_t>(kv_base) * kv_row, ws.v,
                        static_cast<size_t>(seq) * kv_row * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;

    e = Attention(stream, ws.q, cache_k, cache_v, ws.attn, ws.attn_part, seq, NH,
                  NKV, HD, frozen::kAttentionScale, kv_base,
                  LayerWindow(top, layer));
    if (e != cudaSuccess) return e;
    e = AttnGateNorm(stream, ws.attn, ws.attn, ws.gate, lw.d_sgate2, seq, Q, eps);
    if (e != cudaSuccess) return e;

    e = RunLsxNvfp4Gemm(stream, ws.attn, lw.o.d_packed, lw.o.d_scales,
                        lw.o.global_scale, ws.proj, seq, Q, H);
    if (e != cudaSuccess) return e;
    if (lw.o_res.d_packed) {
        e = RunLsxNvfp4Gemm(stream, ws.attn, lw.o_res.d_packed,
                            lw.o_res.d_scales, lw.o_res.global_scale, ws.proj,
                            seq, Q, H, true);
        if (e != cudaSuccess) return e;
    }
    return RunSlerp(stream, top, lw, ws, seq);
}

cudaError_t RunMlpMixer(cudaStream_t stream, const DoryPinnedWeights &pw,
                        const DoryLayerWeights &lw, int seq, Workspace &ws)
{
    const auto &top = pw.top;
    const int H = top.hidden_size;
    const int I = top.intermediate_size;
    const float eps = static_cast<float>(frozen::kJustNormEps);

    cudaError_t e = RunLsxNvfp4Gemm(stream, ws.x, lw.gate_up.d_packed,
                                    lw.gate_up.d_scales, lw.gate_up.global_scale,
                                    ws.gate_up, seq, H, 2 * I);
    if (e != cudaSuccess) return e;
    if (lw.gate_up_res.d_packed) {
        e = RunLsxNvfp4Gemm(stream, ws.x, lw.gate_up_res.d_packed,
                            lw.gate_up_res.d_scales,
                            lw.gate_up_res.global_scale, ws.gate_up, seq, H,
                            2 * I, true);
        if (e != cudaSuccess) return e;
    }
    e = SuvSwiGLUNorm(stream, ws.act, ws.gate_up, lw.d_suv_gate, lw.d_suv_up,
                      seq, I, eps);
    if (e != cudaSuccess) return e;
    e = RunLsxNvfp4Gemm(stream, ws.act, lw.down.d_packed, lw.down.d_scales,
                        lw.down.global_scale, ws.proj, seq, I, H);
    if (e != cudaSuccess) return e;
    if (lw.down_res.d_packed) {
        e = RunLsxNvfp4Gemm(stream, ws.act, lw.down_res.d_packed,
                            lw.down_res.d_scales, lw.down_res.global_scale,
                            ws.proj, seq, I, H, true);
        if (e != cudaSuccess) return e;
    }
    return RunSlerp(stream, top, lw, ws, seq);
}

cudaError_t RunMixer(cudaStream_t stream, const DoryPinnedWeights &pw,
                     const DoryKvCache *cache, int layer, int loop, int seq,
                     int kv_base, const int32_t *d_pos, Workspace &ws)
{
    const auto &lw = pw.layers[static_cast<size_t>(layer)];
    if (!lw.is_attention) return RunMlpMixer(stream, pw, lw, seq, ws);

    int slot = pw.slot_of_layer[static_cast<size_t>(layer)];
    const bool recurrent = frozen::LayerIsRecurrent(layer);
    if (recurrent) slot += loop;
    const int64_t row = static_cast<int64_t>(pw.top.num_key_value_heads) *
                        pw.top.head_dim;
    const int64_t stride = static_cast<int64_t>(cache->max_total) * row;
    return RunAttentionMixer(stream, pw, lw, layer, seq, kv_base, d_pos, ws,
                             cache->d_k + static_cast<int64_t>(slot) * stride,
                             cache->d_v + static_cast<int64_t>(slot) * stride);
}

cudaError_t SampleNext(cudaStream_t stream, const DoryPinnedWeights &pw,
                       Workspace &ws, int seq, lsxcommon::sample::ISampler &sampler,
                       int32_t &tok, float &maxv)
{
    const int H = pw.top.hidden_size;
    const int V = pw.top.vocab_size;
    const __half *last = ws.x + static_cast<int64_t>(seq - 1) * H;
    cudaError_t e = lsxcommon::gemv::RunLxGemmVec(
        stream, pw.d_head_f16, last, 1, V, H, lsxcommon::gemv::LxDtype::kF16,
        lsxcommon::gemv::LxDtype::kF16, 1.0f, ws.logits_h,
        lsxcommon::gemv::LxDtype::kF16, 0.0f);
    if (e != cudaSuccess) return e;
    e = RunDoryLogitScaleKernel(stream, ws.logits, ws.logits_h,
                                pw.d_logit_scale, V);
    if (e != cudaSuccess) return e;
    e = sampler.Sample(stream, ws.logits, V, lsxcommon::sample::LogitType::kF32,
                       ws.tok, ws.maxv);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(&tok, ws.tok, sizeof(int32_t), cudaMemcpyDeviceToHost,
                        stream);
    if (e != cudaSuccess) return e;
    return cudaMemcpyAsync(&maxv, ws.maxv, sizeof(float),
                           cudaMemcpyDeviceToHost, stream);
}

// One token step (or one chunk) over the full Dory graph. The KV cache is
// appended at kv_base by each attention mixer.
cudaError_t RunGraph(cudaStream_t stream, const DoryPinnedWeights &pw,
                     const DoryKvCache *cache, const __half *embed_table,
                     int seq, int kv_base, const int32_t *d_ids,
                     const int32_t *d_pos, Workspace &ws)
{
    const auto &top = pw.top;
    const int H = top.hidden_size;
    const float eps = static_cast<float>(frozen::kJustNormEps);
    const int n_in = top.input_layers;
    const int n_rec = top.recurrent_layers;
    const int n_out = top.output_layers;

    const int total = seq * H;
    kernels::EmbedGatherKernel<__half>
        <<<(total + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            embed_table, d_ids, ws.x, seq, H);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    for (int l = 0; l < n_in && e == cudaSuccess; ++l)
        e = RunMixer(stream, pw, cache, l, 0, seq, kv_base, d_pos, ws);
    if (e != cudaSuccess) return e;

    e = cudaMemcpyAsync(ws.input_h, ws.x, static_cast<size_t>(total) * 2,
                        cudaMemcpyDeviceToDevice, stream);
    if (e == cudaSuccess)
        e = cudaMemcpyAsync(ws.state, ws.x, static_cast<size_t>(total) * 2,
                            cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;

    for (int loop = 0; loop < top.recurrent_loops && e == cudaSuccess; ++loop) {
        e = JustNorm(stream, ws.x, ws.state, ws.input_h, seq, H, eps);
        for (int l = n_in; l < n_in + n_rec && e == cudaSuccess; ++l)
            e = RunMixer(stream, pw, cache, l, loop, seq, kv_base, d_pos, ws);
        if (e != cudaSuccess) break;
        e = JustNorm(stream, ws.state, ws.x, nullptr, seq, H, eps);
    }
    if (e != cudaSuccess) return e;

    e = cudaMemcpyAsync(ws.x, ws.state, static_cast<size_t>(total) * 2,
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    for (int l = n_in + n_rec; l < n_in + n_rec + n_out && e == cudaSuccess; ++l)
        e = RunMixer(stream, pw, cache, l, 0, seq, kv_base, d_pos, ws);
    return e;
}

} // namespace

cudaError_t RunDoryGeneration(cudaStream_t stream, const DoryPinnedWeights &pw,
                              const std::vector<int32_t> &prompt, int max_new,
                              int eos_id, DoryGenerationResult &result,
                              lsxcommon::sample::ISampler &sampler,
                              DoryKvCache *kv,
                              const lsxcommon::GenerationStream *gs)
{
    const auto t_start = std::chrono::steady_clock::now();
    result = DoryGenerationResult{};
    const auto &top = pw.top;
    const int V = top.vocab_size;
    const int P = static_cast<int>(prompt.size());
    if (P <= 0) {
        result.error_msg = "empty prompt";
        return cudaErrorInvalidValue;
    }
    if (top.num_layers <= 0 ||
        pw.layers.size() < static_cast<size_t>(top.num_layers)) {
        result.error_msg = "model has no loaded layers";
        return cudaErrorInvalidValue;
    }
    if (pw.d_embed_f16 == nullptr || pw.d_head_f16 == nullptr ||
        pw.d_logit_scale == nullptr) {
        result.error_msg = "embed/head/logit-scale missing";
        return cudaErrorInvalidValue;
    }

    Workspace ws;
    const int work_seq = kv ? P : P + std::max(max_new, 1);
    cudaError_t e = ws.Alloc(top, work_seq);
    if (e != cudaSuccess) {
        ws.Free();
        result.error_msg = "workspace alloc";
        return e;
    }

    DoryKvCache local_kv;
    DoryKvCache *cache = kv;
    const int pos0 = cache ? cache->kv_rows : 0;
    const int needed = pos0 + P + std::max(max_new, 1);
    if (cache == nullptr) {
        e = local_kv.Allocate(pw, needed, stream);
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "kv alloc";
            return e;
        }
        cache = &local_kv;
    } else {
        e = cache->EnsureCapacity(pw, needed, stream);
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "kv grow";
            return e;
        }
    }

    // ── Chunk (prefill, and every decode step) ──
    auto step = [&](const int32_t *ids, int seq, int base,
                    int32_t &tok) -> cudaError_t {
        std::vector<int32_t> pos(static_cast<size_t>(seq));
        for (int i = 0; i < seq; ++i) pos[static_cast<size_t>(i)] = base + i;
        cudaError_t ce = cudaMemcpyAsync(ws.ids, ids,
                                         static_cast<size_t>(seq) * sizeof(int32_t),
                                         cudaMemcpyHostToDevice, stream);
        if (ce == cudaSuccess)
            ce = cudaMemcpyAsync(ws.pos, pos.data(),
                                 static_cast<size_t>(seq) * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream);
        if (ce != cudaSuccess) return ce;
        ce = RunGraph(stream, pw, cache, pw.d_embed_f16, seq, base, ws.ids,
                      ws.pos, ws);
        if (ce != cudaSuccess) return ce;
        float maxv = 0.f;
        return SampleNext(stream, pw, ws, seq, sampler, tok, maxv);
    };

    int32_t tok = -1;
    e = step(prompt.data(), P, pos0, tok);
    if (e != cudaSuccess) {
        ws.Free();
        result.error_msg = "prefill";
        return e;
    }
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) {
        ws.Free();
        result.error_msg = "prefill sync";
        return e;
    }
    cache->kv_rows = pos0 + P;
    cache->last_tok = tok;
    result.first_token = tok;
    result.tokens.push_back(tok);
    if (gs) gs->Emit(static_cast<std::uint32_t>(tok), 0u);
    result.ttft_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t_start)
                         .count();
    if (tok == eos_id) result.eos_terminated = true;

    for (int n = 1; n < max_new && !result.eos_terminated; ++n) {
        const int32_t next = cache->last_tok;
        const int base = cache->kv_rows;
        e = step(&next, 1, base, tok);
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "decode step";
            return e;
        }
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "decode sync";
            return e;
        }
        cache->last_tok = tok;
        cache->kv_rows += 1;
        result.tokens.push_back(tok);
        if (gs)
            gs->Emit(static_cast<std::uint32_t>(tok),
                     static_cast<std::uint32_t>(result.tokens.size() - 1));
        if (tok == eos_id) result.eos_terminated = true;
        if (gs && gs->Aborted()) break;
    }

    // Final logits row (determinism evidence for SelfCheck).
    {
        std::vector<float> h(static_cast<size_t>(V));
        e = cudaMemcpyAsync(h.data(), ws.logits,
                            static_cast<size_t>(V) * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e == cudaSuccess) result.final_logits = std::move(h);
    }

    result.ok = !result.tokens.empty() && e == cudaSuccess;
    if (!result.ok && result.error_msg.empty())
        result.error_msg = (e == cudaSuccess) ? "no tokens generated"
                                             : "logits copy failed";
    result.total_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t_start)
                          .count();
    result.resident_bytes = pw.resident_bytes;

    ws.Free();
    if (kv == nullptr) local_kv.Free();
    return e;
}

} // namespace dory
} // namespace lsxcommon
