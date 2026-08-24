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

cudaError_t RunQwenMoeRouterTopK(cudaStream_t stream,
                                 const __nv_bfloat16 *router_w,
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

} // namespace qwen_moe
} // namespace lsxcommon
