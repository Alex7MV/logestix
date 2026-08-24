// Qwen3.5-35B-A3B MoE inference kernels: router top-k, routed expert FFN
// slot chain, shared expert combine, single-stream generation driver.
//
// Router semantics are pinned to the HF reference implementation
// (transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py):
//   probs = softmax(logits); top-k on probabilities; renormalize by sum.
// Expert FFN: NVFP4 decode -> SwiGLU(w1.x, w3.x) -> w2 -> coeff-weighted
// scatter-add. Shared expert added with sigmoid scalar gate.
// Protocol invariants: one stream end-to-end (no cudaDeviceSynchronize in the
// generation loop), shared-memory RMW only by thread 0 behind barriers.

#include "lsxcommon/qwen_moe_inference.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_moe_inference_load.h"
#include "nvfp4_codec.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lsxcommon {
namespace qwen_moe {
namespace {

constexpr int kRouterBlock = 128;
constexpr int kMaxTopK = 16;

// ── Router ────────────────────────────────────────────────────────────────

// Gemma-router etalon structure (gemma_inference.cu): one block per token,
// cooperative logits, thread-0 selection. Dynamic smem holds logits[E] and
// probs[E] so E up to 256 is supported.
__global__ void qwen_moe_router_topk_kernel(
    const __nv_bfloat16 *__restrict__ proj, const __half *__restrict__ x,
    int H, int K, int E, QwenMoeRoutedSlot *__restrict__ slots)
{
    const int t = blockIdx.x;
    if (t >= gridDim.x) return;
    const __half *xr = x + static_cast<int64_t>(t) * H;

    extern __shared__ float sh[];
    float *logits = sh;             // [E]
    float *probs = sh + E;          // [E]

    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        float dot = 0.f;
        for (int d = 0; d < H; ++d)
            dot += __bfloat162float(proj[static_cast<int64_t>(e) * H + d]) *
                   __half2float(xr[d]);
        logits[e] = dot;
    }
    __syncthreads();

    // Thread 0 only: softmax -> top-k on probabilities -> renorm (HF parity).
    // Deterministic lower-index tie-break.
    if (threadIdx.x == 0) {
        float m = logits[0];
        for (int e = 1; e < E; ++e) m = fmaxf(m, logits[e]);
        float z = 0.f;
        for (int e = 0; e < E; ++e) {
            probs[e] = expf(logits[e] - m);
            z += probs[e];
        }
        for (int e = 0; e < E; ++e) probs[e] /= z;

        int idx[kMaxTopK];
        float val[kMaxTopK];
        int n = 0;
        for (int e = 0; e < E; ++e) {
            float v = probs[e];
            int pos = n;
            while (pos > 0 &&
                   (val[pos - 1] < v || (val[pos - 1] == v && idx[pos - 1] > e))) {
                if (pos < K) { val[pos] = val[pos - 1]; idx[pos] = idx[pos - 1]; }
                --pos;
            }
            if (pos < K) {
                val[pos] = v;
                idx[pos] = e;
                if (n < K) ++n;
            }
        }
        float s = 0.f;
        for (int i = 0; i < K && i <= n; ++i) s += val[i];
        QwenMoeRoutedSlot *out = slots + static_cast<int64_t>(t) * K;
        for (int i = 0; i < K; ++i) {
            out[i].eid = idx[i];
            out[i].coeff = val[i] / s;
        }
    }
}

} // namespace

cudaError_t RunQwenMoeRouterTopK(cudaStream_t stream,                                 const __nv_bfloat16 *router_w,
                                 const __half *normed,
                                 int seq, int E, int K, int H,
                                 QwenMoeRoutedSlot *d_slots)
{
    if (K > kMaxTopK)
        return cudaErrorInvalidValue;
    const size_t smem = static_cast<size_t>(E) * 2 * sizeof(float);
    if (smem > 48 * 1024)
        return cudaErrorInvalidValue;
    qwen_moe_router_topk_kernel<<<seq, kRouterBlock, smem, stream>>>(
        router_w, normed, H, K, E, d_slots);
    return cudaGetLastError();
}

// ── Routed expert FFN slot chain ──────────────────────────────────────────
namespace {

// y = silu(gate) * up  (HF: act_fn(gate) * up, act = silu)
__global__ void qwen_moe_swiglu_kernel(const __half *__restrict__ g,
                                       const __half *__restrict__ u,
                                       __half *__restrict__ y, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = __half2float(g[i]);
    y[i] = __float2half(gv / (1.f + expf(-gv)) * __half2float(u[i]));
}

// Deterministic ascending-k accumulation into an fp32 accumulator.
__global__ void qwen_moe_coeff_add_kernel(const __half *__restrict__ y2,
                                          float coeff, float *__restrict__ acc,
                                          int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] += coeff * __half2float(y2[i]);
}

__global__ void qwen_moe_f2h_kernel(const float *__restrict__ src,
                                    __half *__restrict__ dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(src[i]);
}

constexpr int kEltBlock = 256;

inline int EltGrid(int n) { return (n + kEltBlock - 1) / kEltBlock; }

} // namespace

cudaError_t RunQwenMoeRoutedFFN(cudaStream_t stream, QwenMoeExpertStore &store,
                                QwenMoeExpertDeviceCache &cache, int layer,
                                const QwenMoeRoutedSlot *h_slots, int K,
                                const __half *d_x, int H, int I,
                                float *d_acc, __half *d_yg, __half *d_yu,
                                __half *d_y2, std::string &error)
{
    for (int k = 0; k < K; ++k) {
        // Ascending-k accumulation keeps the fp32 add order deterministic.
        const QwenMoeRoutedSlot &slot = h_slots[k];
        const int dev_slot =
            EnsureQwenMoeExpertResident(store, layer, slot.eid, cache, stream,
                                        error);
        if (dev_slot < 0) return cudaErrorInvalidValue;

        const auto &rec = store.records[static_cast<size_t>(layer)]
                                        [static_cast<size_t>(slot.eid)];
        uint8_t *pack =
            cache.d_pack + static_cast<int64_t>(dev_slot) * cache.pack_stride;
        uint8_t *scale =
            cache.d_scale + static_cast<int64_t>(dev_slot) * cache.scale_stride;
        // Slot layout mirrors the record: w1|w2|w3 packed, then s1|s2|s3.
        const uint8_t *w1 = pack;
        const uint8_t *w2 = pack + rec.w1_bytes;
        const uint8_t *w3 = pack + rec.w1_bytes + rec.w2_bytes;
        const uint8_t *s1 = scale;
        const uint8_t *s2 = scale + rec.s1_bytes;
        const uint8_t *s3 = scale + rec.s1_bytes + rec.s2_bytes;

        // y_g = w1 . x   [I,H]; y_u = w3 . x; y_2 = w2 . silu(y_g)*y_u   [H,I]
        if (qwen::RunQwenNvfp4Gemm(stream, d_x, w1, s1, rec.gs1, d_yg, 1, H, I,
                             false) != cudaSuccess)
            { error = "w1 gemm failed"; return cudaErrorInvalidValue; }
        if (qwen::RunQwenNvfp4Gemm(stream, d_x, w3, s3, rec.gs3, d_yu, 1, H, I,
                             false) != cudaSuccess)
            { error = "w3 gemm failed"; return cudaErrorInvalidValue; }
        qwen_moe_swiglu_kernel<<<EltGrid(I), kEltBlock, 0, stream>>>(
            d_yg, d_yu, d_yg, I);
        if (qwen::RunQwenNvfp4Gemm(stream, d_yg, w2, s2, rec.gs2, d_y2, 1, I, H,
                             false) != cudaSuccess)
            { error = "w2 gemm failed"; return cudaErrorInvalidValue; }
        qwen_moe_coeff_add_kernel<<<EltGrid(H), kEltBlock, 0, stream>>>(
            d_y2, slot.coeff, d_acc, H);
    }
    return cudaGetLastError();
}

cudaError_t RunQwenMoeAccToFp16(cudaStream_t stream, const float *d_acc,
                                __half *d_out, int n)
{
    qwen_moe_f2h_kernel<<<EltGrid(n), kEltBlock, 0, stream>>>(d_acc, d_out, n);
    return cudaGetLastError();
}

} // namespace qwen_moe
} // namespace lsxcommon
