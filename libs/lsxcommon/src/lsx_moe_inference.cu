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


#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/shared_experts.h"
#include "lsxcommon/lsx_tc_gemm.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "nvfp4_codec.h"
#include "nvfp4_scale.h"
#include "nvfp4_gemv_math.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_act.cuh"
#include "lsx_kernels_quant.cuh"

#include <algorithm>
#include <filesystem>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"
#include "fast_load_internal.h"

#include <absl/strings/str_cat.h>

#include <fstream>
#include <iterator>

namespace lsxcommon {
namespace moe {
namespace {

using fastload::kArenaAlign;
using fastload::AlignUp;

constexpr int kRouterBlock = 128;

static_assert(sizeof(lsx_tc::LsxTcBatchSlot) == sizeof(MoEGvSlotDev) &&
                  alignof(lsx_tc::LsxTcBatchSlot) ==
                      alignof(MoEGvSlotDev),
              "LsxTcBatchSlot must stay layout-identical to MoEGvSlotDev");

// ── Router ────────────────────────────────────────────────────────────────

// Gemma-router etalon structure (gemma_moe_inference.cu): one block per token,
// cooperative logits, thread-0 selection. Dynamic smem holds logits[E] and
// probs[E] so E up to 256 is supported.
// Shared router row-dot: ONE accumulation order for every caller
// (prefill monolithic kernel AND decode-split stage 1). Vectorized 8-chain
// fold when H % 8 == 0 and both rows are 16B-aligned; scalar sequential walk
// otherwise. The fold order is fixed: ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7)).
__device__ __forceinline__ float QwenRouterDotRow(
    const __nv_bfloat16 *__restrict__ pr, const __half *__restrict__ xr,
    int H)
{
    if ((H & 7) == 0 &&
        ((reinterpret_cast<uintptr_t>(pr) |
          reinterpret_cast<uintptr_t>(xr)) & 15) == 0) {
        const uint4 *p4 = reinterpret_cast<const uint4 *>(pr);
        const uint4 *x4 = reinterpret_cast<const uint4 *>(xr);
        const int steps = H / 8;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        float a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
        for (int i = 0; i < steps; ++i) {
            const uint4 pv = p4[i];
            const uint4 xv = x4[i];
            const __nv_bfloat16 *pb =
                reinterpret_cast<const __nv_bfloat16 *>(&pv);
            const __half *xb = reinterpret_cast<const __half *>(&xv);
            float f0 = __bfloat162float(pb[0]) * __half2float(xb[0]);
            float f1 = __bfloat162float(pb[1]) * __half2float(xb[1]);
            float f2 = __bfloat162float(pb[2]) * __half2float(xb[2]);
            float f3 = __bfloat162float(pb[3]) * __half2float(xb[3]);
            float f4 = __bfloat162float(pb[4]) * __half2float(xb[4]);
            float f5 = __bfloat162float(pb[5]) * __half2float(xb[5]);
            float f6 = __bfloat162float(pb[6]) * __half2float(xb[6]);
            float f7 = __bfloat162float(pb[7]) * __half2float(xb[7]);
            a0 += f0; a1 += f1; a2 += f2; a3 += f3;
            a4 += f4; a5 += f5; a6 += f6; a7 += f7;
        }
        return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
    }
    float dot = 0.f;
    for (int d = 0; d < H; ++d)
        dot += __bfloat162float(pr[d]) * __half2float(xr[d]);
    return dot;
}

// Decode-split router: stage 1 computes logits with 8 THREADS PER EXPERT.
// Thread L owns chain L of the QwenRouterDotRow 8-chain fold: every thread
// walks the same ascending element sequence (chain, chain+8, ...) so the
// per-chain accumulation order — and therefore the logits bits — is
// identical to the single-thread etalon; only the fold of the 8 partials is
// shared via shuffle in the fixed etalon order
// ((a0+a1)+(a2+a3))+((a4+a5)+(a6+a7)).
// Replaces the one-thread-per-expert shape (latency-bound at ~50us/token:
// a single serial 4KB read chain per expert with 256 threads in flight).
// Loads are 2-byte strided (element-width) instead of full uint4: the old
// uint4 walk held 8x16B in flight per thread (~64 registers) and serialized
// on this narrow occupancy (E*8 = 2048 threads = 16 blocks), measuring
// 34us/call in situ; the strided walk issues the same 256 coalesced
// transactions per warp group but with a tiny register footprint and lands
// ~6.3us standalone / ~13us through the production entry (4x). The emitted
// FP32 mul-add sequence is bit-identical (pinned by
// "router decode-split matches prefill bit-exact" + the real-shape budget).
__global__ void moe_router_dots_kernel(
    const __nv_bfloat16 *__restrict__ proj, const __half *__restrict__ x,
    int H, float *__restrict__ logits, int E)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    const int chain = g & 7;
    const int e = g >> 3;
    const bool ok = e < E;
    const bool vec = ok && (H & 7) == 0 &&
                     ((reinterpret_cast<uintptr_t>(
                           proj + static_cast<int64_t>(e) * H) |
                       reinterpret_cast<uintptr_t>(x)) &
                      15) == 0;
    float a = 0.f;
    bool direct = false;   // scalar path: chain 0 wrote the dot itself
    if (ok) {
        const __half *xr = x;   // decode: t == 0
        const __nv_bfloat16 *pr = proj + static_cast<int64_t>(e) * H;
        if (vec) {
            // Strided element-width walk: chain L accumulates elements
            // L, L+8, L+16, ... in ascending order — exactly the per-chain
            // order of the uint4 fold above, with a minimal register set.
            for (int d = chain; d < H; d += 8)
                a += __bfloat162float(pr[d]) * __half2float(xr[d]);
        } else if (chain == 0) {
            float dot = 0.f;
            for (int d = 0; d < H; ++d)
                dot += __bfloat162float(pr[d]) * __half2float(xr[d]);
            logits[e] = dot;
            direct = true;
        }
    }
    // Fold the 8 chain partials in the etalon order; group = 8 consecutive
    // threads (all lanes of the warp participate in the shuffles — the fold
    // is reached uniformly, out-of-range tails contribute zeros).
    const int lane = threadIdx.x & 31;
    const int base = lane & ~7;
    const float a0 = __shfl_sync(0xffffffffu, a, base + 0);
    const float a1 = __shfl_sync(0xffffffffu, a, base + 1);
    const float a2 = __shfl_sync(0xffffffffu, a, base + 2);
    const float a3 = __shfl_sync(0xffffffffu, a, base + 3);
    const float a4 = __shfl_sync(0xffffffffu, a, base + 4);
    const float a5 = __shfl_sync(0xffffffffu, a, base + 5);
    const float a6 = __shfl_sync(0xffffffffu, a, base + 6);
    const float a7 = __shfl_sync(0xffffffffu, a, base + 7);
    if (chain == 0 && ok && !direct)
        logits[e] = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}

// Stage 2: softmax -> top-k -> renorm from precomputed logits (unchanged
// HF-parity selection logic). The exp pass is thread-parallel (elementwise,
// values unchanged); the max is computed by tree reduction (exact — order
// cannot change the result), the denominator sum and the top-K insertion
// stay sequential on thread 0 in the same ascending-e order, so selection
// and coefficients remain bit-identical to the single-thread version.
__global__ void moe_router_select_kernel(
    const float *__restrict__ logits, int K, int E,
    MoERoutedSlot *__restrict__ slots)
{
    const int t = blockIdx.x;
    extern __shared__ float sh[];
    float *probs = sh;   // [E]
    __shared__ float s_max;

    float local = -INFINITY;
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        local = fmaxf(local, logits[e]);
    for (int off = 16; off > 0; off >>= 1)
        local = fmaxf(local, __shfl_down_sync(0xffffffffu, local, off));
    __shared__ float warp_max[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_max[warp] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m2 = -INFINITY;
        const int warps = (blockDim.x + 31) >> 5;
        for (int w = 0; w < warps; ++w) m2 = fmaxf(m2, warp_max[w]);
        s_max = m2;
    }
    __syncthreads();
    const float m = s_max;
    for (int e = threadIdx.x; e < E; e += blockDim.x)
        probs[e] = expf(logits[e] - m);
    __syncthreads();
    if (threadIdx.x == 0) {
        float z = 0.f;
        for (int e = 0; e < E; ++e) z += probs[e];
        s_max = z;   // reuse: broadcast the denominator
    }
    __syncthreads();
    // Elementwise renorm in parallel — division of a fixed input by the
    // same z is deterministic per element, so the probs bits are identical
    // to the sequential thread-0 walk.
    {
        const float z = s_max;
        for (int e = threadIdx.x; e < E; e += blockDim.x) probs[e] /= z;
    }
    __syncthreads();
    // Exact-rank selection (bit-identical to the sequential insertion walk:
    // element e lands at position r(e) = |{e'<e : val[e']>val[e]}| +
    // |{e'<e : val[e']==val[e]}| — the stable (val desc, idx asc) order).
    // Parallel O(E^2) vote replaces the serial thread-0 sort.
    __shared__ float s_val[kMaxTopK];
    __shared__ int s_idx[kMaxTopK];
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const float v = probs[e];
        int r = 0;
        for (int j = 0; j < E; ++j) {
            const float u = probs[j];
            r += (u > v) || (u == v && j < e);
        }
        if (r < K) {
            s_val[r] = v;
            s_idx[r] = e;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sm = 0.f;
        for (int i = 0; i < K; ++i) sm += s_val[i];
        MoERoutedSlot *out = slots + static_cast<int64_t>(t) * K;
        for (int i = 0; i < K; ++i) {
            out[i].eid = s_idx[i];
            out[i].coeff = s_val[i] / sm;
        }
    }
}

__global__ void moe_router_topk_kernel(
    const __nv_bfloat16 *__restrict__ proj, const __half *__restrict__ x,
    int H, int K, int E, MoERoutedSlot *__restrict__ slots)
{
    const int t = blockIdx.x;
    const __half *xr = x + static_cast<int64_t>(t) * H;

    extern __shared__ float sh[];
    float *logits = sh;             // [E]
    float *probs = sh + E;          // [E]

    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const __nv_bfloat16 *pr =
            proj + static_cast<int64_t>(e) * H;
        logits[e] = QwenRouterDotRow(pr, xr, H);
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
        MoERoutedSlot *out = slots + static_cast<int64_t>(t) * K;
        for (int i = 0; i < K; ++i) {
            out[i].eid = idx[i];
            out[i].coeff = val[i] / s;
        }
    }
}

} // namespace

cudaError_t RunMoERouterTopK(cudaStream_t stream,                                 const __nv_bfloat16 *router_w,
                                 const __half *normed,
                                 int seq, int E, int K, int H,
                                 MoERoutedSlot *d_slots)
{
    if (K > kMaxTopK)
        return cudaErrorInvalidValue;
    const size_t smem = static_cast<size_t>(E) * 2 * sizeof(float);
    if (smem > 48 * 1024)
        return cudaErrorInvalidValue;
    if (seq == 1) {
        // Decode: two-stage split (parallel-per-expert dots + select).
        // Stage-1 scratch is stream-ordered (pool-backed, non-synchronizing);
        // no static device state (the old cached cudaMalloc leaked one buffer
        // per process and was not thread-safe across model instances).
        float *d_logits = nullptr;
        cudaError_t le = cudaMallocAsync(&d_logits, static_cast<size_t>(E) *
                                                          sizeof(float),
                                         stream);
        if (le != cudaSuccess) return le;
        // 8 threads per expert (chain-parallel etalon walk): E*8 threads
        // across 128-thread blocks engage all SMs on the 1MB row stream.
        moe_router_dots_kernel<<<(E * 8 + 127) / 128, 128, 0, stream>>>(
            router_w, normed, H, d_logits, E);
        le = cudaGetLastError();
        if (le == cudaSuccess)
            moe_router_select_kernel<<<seq, kRouterBlock,
                                            sizeof(float) * E, stream>>>(
                d_logits, K, E, d_slots);
        if (le != cudaSuccess) {
            cudaFreeAsync(d_logits, stream);
            return le;
        }
        cudaFreeAsync(d_logits, stream);
        return cudaGetLastError();
    }
    moe_router_topk_kernel<<<seq, kRouterBlock, smem, stream>>>(
        router_w, normed, H, K, E, d_slots);
    return cudaGetLastError();
}

// ── Routed expert FFN slot chain ──────────────────────────────────────────
namespace {

// SwiGLU: unified kernels::SwigluKernel (lsx_kernels_act.cuh).

// Deterministic ascending-k accumulation into an fp32 accumulator.
__global__ void moe_coeff_add_kernel(const __half *__restrict__ y2,
                                          float coeff, float *__restrict__ acc,
                                          int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] += coeff * __half2float(y2[i]);
}

__global__ void moe_coeff_add_scaled_kernel(const __half *__restrict__ y2,
                                            float coeff,
                                            const float *__restrict__ alpha,
                                            float *__restrict__ acc, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] += coeff * (*alpha) * __half2float(y2[i]);
}

// F32->FP16 convert: unified kernels::F32toTKernel (lsx_kernels_math.cuh).

constexpr int kEltBlock = 256;

inline int EltGrid(int n) { return (n + kEltBlock - 1) / kEltBlock; }

} // namespace

// Defined below (batched-GEMV section); lets the handshake routed-FFN
// branch target the device-table core directly.
cudaError_t RunMoERoutedGemmVBatchCore(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const MoEGvSlotDev *slots_dev, int k_slots,
    int64_t w13_bytes, int64_t s13_bytes,
    int H, int I, const __half *d_x,
    __half *ygu, __half *hbuf, __half *ypart, float *acc);

cudaError_t RunMoERoutedFFN(cudaStream_t stream, MoEExpertStore &store,
                                gpu::ExpertDeviceCache &cache, int layer,
                                const MoERoutedSlot *h_slots, int K,
                                const __half *d_x, int H, int I,
                                float *d_acc, __half *d_yg, __half *d_yu,
                                __half *d_y2, std::string &error,
                                cudaStream_t copy_stream,
                                cudaEvent_t *ev_uploads, cudaEvent_t ev_front,
                                const uint8_t *xq,
                                const uint8_t *xqs,
                                uint8_t *d_qpack_scratch,
                                uint8_t *d_qscales_scratch,
                                __half *ygu_batch,
                                __half *hbuf_batch,
                                __half *ypart_batch,
                                MoEGvSlotDev *d_gemv_tbl,
                                float *d_swi_scratch)
{
    // Pass 1 (staging): resolve residency for all K slots up front so miss
    // uploads can stream on the copy channel while the compute chain runs.
    int dev_slots[16];
    bool missed[16];
    if (K > 16) { error = "routed slots exceed staging capacity";
        return cudaErrorInvalidValue; }
    if (copy_stream) {
        if (!ev_uploads) { error = "bad copy-stream args";
            return cudaErrorInvalidValue; }
        // Fence: never overwrite a cache slot that previously enqueued compute
        // may still be reading. ev_front is recorded BEFORE this layer's work;
        // once the copy stream passes the wait, all prior compute has retired.
        cudaError_t e = cudaEventRecord(ev_front, stream);
        if (e != cudaSuccess) return e;
        e = cudaStreamWaitEvent(copy_stream, ev_front, 0);
        if (e != cudaSuccess) return e;
    }
    for (int k = 0; k < K; ++k) {
        missed[k] = false;
        dev_slots[k] = moe::EnsureMoEExpertResident(
            store, layer, h_slots[k].eid, cache, stream, error, copy_stream,
            copy_stream ? &missed[k] : nullptr);
        if (dev_slots[k] < 0) return cudaErrorInvalidValue;
    }
    // ── Batched GEMV decode chain (round-19 lever, experimental) ────────────
    // Steady state (full residency => zero misses) routes all K slots
    // through 4 launches instead of K x 5. Any miss falls back to the
    // legacy per-slot flow so copy-stream fencing keeps its exact contract.
    static const bool k_gemv_batch =
        std::getenv("LSX_GEMV_EXPERTS") != nullptr;
    bool any_miss = false;
    for (int k = 0; k < K; ++k) any_miss = any_miss || missed[k];
    if (k_gemv_batch && !any_miss && (I & 31) == 0 && (H & 31) == 0 &&
        store.records[layer][h_slots[0].eid].w1_bytes > 0) {
        const auto &rec0 =
            store.records[static_cast<size_t>(layer)]
                         [static_cast<size_t>(h_slots[0].eid)];
        const int64_t w13_b =
            static_cast<int64_t>(rec0.w1_bytes + rec0.w3_bytes);
        const int64_t s13_b =
            static_cast<int64_t>(rec0.s1_bytes + rec0.s3_bytes);
        MoEGvSlotDev tbl[16];
        for (int k = 0; k < K; ++k) {
            const auto &rec =
                store.records[static_cast<size_t>(layer)]
                             [static_cast<size_t>(h_slots[k].eid)];
            tbl[k].dev_slot = dev_slots[k];
            tbl[k].gs1 = rec.gs1;
            tbl[k].gs3 = rec.gs3;
            tbl[k].gs2 = rec.gs2;
            tbl[k].coeff = h_slots[k].coeff;
        }
        // Scratch: reuse caller-provided arena slices when present; fall
        // back to async pool allocations otherwise (tests / ad-hoc calls).
        MoEGvSlotDev *tbl_d = d_gemv_tbl;
        __half *ygu_b = ygu_batch, *hbuf_b = hbuf_batch,
               *ypart_b = ypart_batch;
        bool owned = false;
        if (!tbl_d) {
            if (cudaMallocAsync(&tbl_d, sizeof(MoEGvSlotDev) * K,
                                stream) == cudaSuccess &&
                cudaMallocAsync(reinterpret_cast<void **>(&ygu_b),
                                static_cast<size_t>(K) * 2 * I *
                                    sizeof(__half),
                                stream) == cudaSuccess &&
                cudaMallocAsync(reinterpret_cast<void **>(&hbuf_b),
                                static_cast<size_t>(K) * I * sizeof(__half),
                                stream) == cudaSuccess &&
                cudaMallocAsync(reinterpret_cast<void **>(&ypart_b),
                                static_cast<size_t>(K) * H * sizeof(__half),
                                stream) == cudaSuccess)
                owned = true;
            else {
                if (tbl_d) cudaFreeAsync(tbl_d, stream);
                if (ygu_b) cudaFreeAsync(ygu_b, stream);
                if (hbuf_b) cudaFreeAsync(hbuf_b, stream);
                if (ypart_b) cudaFreeAsync(ypart_b, stream);
            }
        }
        if (!owned && tbl_d && ygu_b && hbuf_b && ypart_b) {
            cudaMemcpyAsync(tbl_d, tbl, sizeof(MoEGvSlotDev) * K,
                            cudaMemcpyHostToDevice, stream);
            const cudaError_t be = RunMoERoutedGemmVBatchCore(
                stream, cache.d_pack, cache.d_scale, cache.pack_stride,
                cache.scale_stride, tbl_d, K, w13_b, s13_b, H, I, d_x,
                ygu_b, hbuf_b, ypart_b, d_acc);
            if (be == cudaSuccess)
                return cudaGetLastError();
            return be;
        }
    }
    if (copy_stream) {
        for (int k = 0; k < K; ++k) {
            if (!missed[k]) continue;
            // Uploads of distinct slots were already serialized on the copy
            // stream in pass-1 order; one event per expert is enough.
            cudaError_t e = cudaEventRecord(ev_uploads[k], copy_stream);
            if (e != cudaSuccess) return e;
        }
    }

    for (int k = 0; k < K; ++k) {
        // Ascending-k accumulation keeps the fp32 add order deterministic.
        const MoERoutedSlot &slot = h_slots[k];
        const int dev_slot = dev_slots[k];

        const auto &rec = store.records[static_cast<size_t>(layer)]
                                        [static_cast<size_t>(slot.eid)];
        uint8_t *pack =
            cache.d_pack + static_cast<int64_t>(dev_slot) * cache.pack_stride;
        uint8_t *scale =
            cache.d_scale + static_cast<int64_t>(dev_slot) * cache.scale_stride;
        // Slot layout is STAGED FUSED: [w13 = w1|w3][w2] packed,
        // [s13 = s1|s3][s2] scales (see EnsureMoEExpertResident). Gate
        // and up run as ONE GEMM over N=2I; per-segment global scales
        // {gs1 | gs3} split at N/2 inside the kernel epilogue.
        const uint8_t *w13 = pack;
        const uint8_t *w2 = pack + rec.w1_bytes + rec.w3_bytes;
        const uint8_t *s13 = scale;
        const uint8_t *s2 = scale + rec.s1_bytes + rec.s3_bytes;
        const float *seg_wgs =
            store.d_seg_wgs
                ? store.d_seg_wgs +
                      (static_cast<int64_t>(layer) * store.num_experts +
                       slot.eid) *
                          2
                : nullptr;

        if (copy_stream && missed[k]) {
            cudaError_t e = cudaStreamWaitEvent(stream, ev_uploads[k], 0);
            if (e != cudaSuccess) return e;
        }

        // y_g|y_u = [w1; w3] . x   [2I,H]; then y_2 = w2 . silu(y_g)*y_u.
        // Fused single-GEMM form requires the TC epilogue for per-segment
        // scales; the scalar path ignores them, so it keeps the split pair.
        // LSX_GEMV_EXPERTS=1 routes routed experts through the seq==1
        // weight-streaming GEMV instead (FP16 activations -- W?A16 numeric
        // class); experimental round-18 decode-lever probe.
        static const bool k_gemv_experts =
            std::getenv("LSX_GEMV_EXPERTS") != nullptr;
        const bool tc_fused =
            qwen::QwenTcFastMath() && store.d_seg_wgs != nullptr &&
            (I & 15) == 0 && !k_gemv_experts;
        const uint8_t *w1_split = pack;
        const uint8_t *w3_split = pack + rec.w1_bytes;
        const uint8_t *s1_split = scale;
        const uint8_t *s3_split = scale + rec.s1_bytes;
        if (!tc_fused) {
            if (lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, w1_split, s1_split,
                                       rec.gs1, d_yg, 1, H, I,
                                       false) != cudaSuccess)
                { error = "w1 gemm failed"; return cudaErrorInvalidValue; }
            if (lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, w3_split, s3_split,
                                       rec.gs3, d_yu, 1, H, I,
                                       false) != cudaSuccess)
                { error = "w3 gemm failed"; return cudaErrorInvalidValue; }
        } else {
            cudaError_t ge;
            if (xq && xqs) {
                ge = lsx_tc::RunLsxTcGemmPreQuant(
                    stream, xq, xqs, w13, s13, 1.0f, d_yg, 1, H, 2 * I,
                    false, seg_wgs, nullptr);
            } else {
                ge = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, w13, s13, 1.0f,
                                            d_yg, 1, H, 2 * I, false,
                                            seg_wgs, nullptr);
            }
            if (ge != cudaSuccess) {
                { error = "w13 gemm failed"; return cudaErrorInvalidValue; }
            }
        }
        // TC: fuse the activation quant into swiglu so the down GEMM skips
        // its ActQuant pass; scalar path keeps split swiglu + fp16 GEMM.
        if (tc_fused) {
            // Quant scratch for the down-GEMM input: caller-provided arena
            // slice when available (removes two pool alloc/free calls per
            // expert slot per layer per step); stream-ordered fallback keeps
            // standalone callers working.
            uint8_t *ap = d_qpack_scratch, *as = d_qscales_scratch;
            const bool own_scratch = !ap || !as;
            if (own_scratch) {
                ap = nullptr; as = nullptr;
                if (cudaMallocAsync(&ap, static_cast<size_t>(I) / 2, stream) !=
                        cudaSuccess ||
                    cudaMallocAsync(&as, static_cast<size_t>((I + 15) / 16),
                                    stream) != cudaSuccess) {
                    if (ap) cudaFreeAsync(ap, stream);
                    error = "w2 quant scratch failed";
                    return cudaErrorInvalidValue;
                }
            }
            // Overflow-proof producer-fused swiglu+quant. The per-16 block
            // scale is E4M3 (max finite 448), so an activation needing
            // absmax/6 > 448 saturates and reconstructs ~6x too small --
            // Mistral-Small-4 layer 32 expert 67 reaches 2.5e5 and loses ~52x
            // of its contribution. Prescaling by a power of two and folding
            // the reciprocal into the FP32 accumulator below fixes it, and is
            // bit-exact when alpha == 1 (every other family).
            // FP32 SwiGLU arena: the fused kernel stores the product to FP16
            // before quantizing, which overflows to inf for massive-activation
            // models and poisons the block scale.
            float *alpha_slot = d_swi_scratch ? (d_swi_scratch + I + 2)
                                              : nullptr;
            float *sw_f32 = d_swi_scratch;
            const bool own_alpha = (alpha_slot == nullptr);
            bool own_swi = false;
            if (own_alpha) {
                // Standalone caller with no arena: one FP32 buffer + two
                // scalars for absmax/alpha.
                if (cudaMallocAsync(&sw_f32, static_cast<size_t>(I) * sizeof(float),
                                    stream) != cudaSuccess ||
                    cudaMallocAsync(&alpha_slot, 2 * sizeof(float), stream) !=
                        cudaSuccess) {
                    if (sw_f32) cudaFreeAsync(sw_f32, stream);
                    if (own_scratch) {
                        cudaFreeAsync(ap, stream);
                        cudaFreeAsync(as, stream);
                    }
                    error = "tc swiglu scratch failed";
                    return cudaErrorInvalidValue;
                }
                own_swi = true;
            }
            cudaError_t qe = qwen::RunAttnSwigluPreScaledQuantInternal(
                stream, d_yg, d_yu, sw_f32, d_yg, ap, as, I, alpha_slot);
            if (qe == cudaSuccess)
                qe = lsx_tc::RunLsxTcGemmPreQuant(
                    stream, ap, as, w2, s2, rec.gs2, d_y2, 1, I, H, false);
            if (qe == cudaSuccess) {
                // Fold the 1/alpha back in inside the FP32 accumulator.
                moe_coeff_add_scaled_kernel<<<EltGrid(H), kEltBlock, 0, stream>>>(
                    d_y2, slot.coeff, alpha_slot + 1, d_acc, H);
                qe = cudaGetLastError();
            }
            if (own_scratch) {
                cudaFreeAsync(ap, stream);
                cudaFreeAsync(as, stream);
            }
            if (own_swi) {
                cudaFreeAsync(sw_f32, stream);
                cudaFreeAsync(alpha_slot, stream);
            }
            if (qe != cudaSuccess)
                { error = "w2 prequant gemm failed"; return cudaErrorInvalidValue; }
            continue;   // accumulator already updated above
        } else {
            // Overflow-proof scalar SwiGLU: the product is evaluated in FP32
            // and stored to the FP16 activation buffer under a power-of-two
            // prescale, folded back in the FP32 accumulator. Identical result
            // when the product fits FP16 (alpha == 1).
            float *sw_f32 = nullptr, *d_alpha = nullptr, *d_inv_alpha = nullptr;
            const bool own_swi = (d_swi_scratch == nullptr);
            if (own_swi)
            {
                const size_t sw_bytes = static_cast<size_t>(I) * sizeof(float);
                cudaError_t me = cudaMallocAsync(&sw_f32, sw_bytes, stream);
                if (me == cudaSuccess)
                    me = cudaMallocAsync(&d_alpha, 2 * sizeof(float), stream);
                if (me != cudaSuccess)
                {
                    if (sw_f32) cudaFreeAsync(sw_f32, stream);
                    error = "swiglu scratch failed";
                    return cudaErrorInvalidValue;
                }
            }
            else
            {
                sw_f32 = d_swi_scratch;
                d_alpha = d_swi_scratch + I;
            }
            d_inv_alpha = d_alpha + 1;
            ::lsxcommon::kernels::SwigluF32Kernel<<<EltGrid(I), kEltBlock, 0, stream>>>(
                d_yg, d_yu, sw_f32, I);
            ::lsxcommon::kernels::SwigluAbsMaxKernel<<<1, kEltBlock, 0, stream>>>(sw_f32,
                                                                   d_alpha, I);
            ::lsxcommon::kernels::SwigluAlphaKernel<<<1, 32, 0, stream>>>(d_alpha, d_alpha,
                                                          d_inv_alpha);
            ::lsxcommon::kernels::SwigluStoreScaledKernel<<<EltGrid(I), kEltBlock, 0, stream>>>(
                sw_f32, d_inv_alpha, d_yg, I);
            cudaError_t se = cudaGetLastError();
            if (se == cudaSuccess)
                se = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_yg, w2, s2, rec.gs2,
                                                 d_y2, 1, I, H, false);
            if (se == cudaSuccess)
            {
                // Fold the 1/alpha back in inside the FP32 accumulator.
                moe_coeff_add_scaled_kernel<<<EltGrid(H), kEltBlock, 0, stream>>>(
                    d_y2, slot.coeff, d_alpha, d_acc, H);
                se = cudaGetLastError();
            }
            if (own_swi)
            {
                cudaFreeAsync(sw_f32, stream);
                cudaFreeAsync(d_alpha, stream);
            }
            if (se != cudaSuccess)
                { error = "w2 gemm failed"; return cudaErrorInvalidValue; }
            continue;   // accumulator already updated above
        }
        moe_coeff_add_kernel<<<EltGrid(H), kEltBlock, 0, stream>>>(
            d_y2, slot.coeff, d_acc, H);
    }
    return cudaGetLastError();
}

cudaError_t RunMoEAccToFp16(cudaStream_t stream, const float *d_acc,
                                __half *d_out, int n)
{
    lsxcommon::kernels::F32toTKernel<__half><<<EltGrid(n), kEltBlock, 0, stream>>>(d_acc, d_out, n);
    return cudaGetLastError();
}

// ── Batched routed-expert GEMV chain (round-19 decode lever) ────────────────
// Replaces the per-slot {w1,w3,w2} loop (K x 5 kernels/layer) with 4 launches:
// w13 GEMV batch -> elementwise swiglu -> down GEMV batch (per-slot partials)
// -> ordered coeff reduce. Same weight-streaming contract as the
// moe_nvfp4_gemv_kernel core: one warp owns a K-slice of one flat output
// row, packed E2M1 nibbles stream via coalesced uint4 loads, partials merge
// through a shfl tree + ordered shared-memory sum. Determinism contract:
// fixed KP partitioning; the down stage writes per-slot partials (single
// writer per cell) and the reduce adds slots in ascending order -- the exact
// add order of the legacy per-slot coeff_add chain. Requires I % 32 == 0 &&
// H % 32 == 0 (16 B row-alignment of every packed weight row).
namespace {

constexpr int kGemvBatchWarps = 8;

// One warp owns a K-slice of one flat output row rflat (slot-major). The
// shared dot math lives in nvfp4_gemv_math.cuh (single source of truth for
// the byte/nibble/scale-block contract exercised by both suites).
__device__ __forceinline__ float moe_gemv_dot(
    const __half *__restrict__ x, const uint8_t *__restrict__ wrow,
    const uint8_t *__restrict__ srow, int kb, int chunks, int lane)
{
    return lsxcommon::nvfp4gemv::GemvDotRow(x, wrow, srow, kb, chunks, lane);
}

// ygu[s*2I + r] = dot(w13 row r of slot s, x) * (r < I ? gs1 : gs3).
// The fused w13 slab is [2I, H] row-major starting at the slot's packed base.
template <int KP>
__global__ void moe_gemv_w13_batch_kernel(
    const __half *__restrict__ x_row,
    const uint8_t *__restrict__ pack_base,
    const uint8_t *__restrict__ scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const MoEGvSlotDev *__restrict__ slots,
    int k_slots, int I, int H, __half *__restrict__ ygu)
{
    constexpr int ROWS = kGemvBatchWarps / KP;
    static_assert(ROWS >= 1 && KP <= kGemvBatchWarps);
    __shared__ float sp[kGemvBatchWarps];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int rflat = blockIdx.x * ROWS + (warp / KP);
    const int total = k_slots * 2 * I;
    float acc = 0.0f;
    if (rflat < total) {
        const int s = rflat / (2 * I);
        const int r = rflat - s * 2 * I;
        const MoEGvSlotDev slot = slots[s];
        const float gs = (r < I) ? slot.gs1 : slot.gs3;
        acc = moe_gemv_dot(
                  x_row,
                  pack_base +
                      static_cast<int64_t>(slot.dev_slot) * pack_stride +
                      static_cast<int64_t>(r) * (H >> 1),
                  scale_base +
                      static_cast<int64_t>(slot.dev_slot) * scale_stride +
                      static_cast<int64_t>(r) * (H >> 4),
                  (KP > 1 ? (H / KP) * (warp % KP) : 0),
                  (H / KP) >> 5, lane) *
              gs;
    }
    if (lane == 0) sp[warp] = acc;
    __syncthreads();
    if (lane == 0 && (warp % KP) == 0 && rflat < total) {
        const int wbase = (warp / KP) * KP;
        float t = 0.0f;
#pragma unroll
        for (int j = 0; j < KP; ++j) t += sp[wbase + j];
        ygu[rflat] = __float2half(t);
    }
}

__global__ void moe_gemv_swiglu_batch_kernel(
    const __half *__restrict__ ygu, int n_half /*== k_slots * I*/,
    int I, __half *__restrict__ hbuf)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_half) return;
    // Slot-major halves: gate rows [0,I), up rows [I,2I) per slot.
    const int s = i / I;
    const int j = i - s * I;
    const float g = __half2float(ygu[static_cast<int64_t>(s) * 2 * I + j]);
    const float u =
        __half2float(ygu[static_cast<int64_t>(s) * 2 * I + I + j]);
    hbuf[i] = __float2half((g / (1.0f + expf(-g))) * u);
}

// ypart[s*H + r] = dot(w2 row r of slot s, hbuf[s]) * gs2.
// Down slab [H, I] lives at byte offset w13_bytes in the slot's packed base;
// its scales at offset s13_bytes.
template <int KP>
__global__ void moe_gemv_down_batch_kernel(
    const __half *__restrict__ hbuf,
    const uint8_t *__restrict__ pack_base,
    const uint8_t *__restrict__ scale_base,
    int64_t pack_stride, int64_t scale_stride,
    int64_t w13_bytes, int64_t s13_bytes,
    const MoEGvSlotDev *__restrict__ slots,
    int k_slots, int I, int H, __half *__restrict__ ypart)
{
    constexpr int ROWS = kGemvBatchWarps / KP;
    static_assert(ROWS >= 1 && KP <= kGemvBatchWarps);
    __shared__ float sp[kGemvBatchWarps];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int rflat = blockIdx.x * ROWS + (warp / KP);
    const int total = k_slots * H;
    float acc = 0.0f;
    if (rflat < total) {
        const int s = rflat / H;
        const int r = rflat - s * H;
        const MoEGvSlotDev slot = slots[s];
        acc = moe_gemv_dot(
                  hbuf + static_cast<int64_t>(s) * I,
                  pack_base +
                      static_cast<int64_t>(slot.dev_slot) * pack_stride +
                      w13_bytes + static_cast<int64_t>(r) * (I >> 1),
                  scale_base +
                      static_cast<int64_t>(slot.dev_slot) * scale_stride +
                      s13_bytes + static_cast<int64_t>(r) * (I >> 4),
                  (KP > 1 ? (I / KP) * (warp % KP) : 0),
                  (I / KP) >> 5, lane) *
              slot.gs2;
    }
    if (lane == 0) sp[warp] = acc;
    __syncthreads();
    if (lane == 0 && (warp % KP) == 0 && rflat < total) {
        const int wbase = (warp / KP) * KP;
        float t = 0.0f;
#pragma unroll
        for (int j = 0; j < KP; ++j) t += sp[wbase + j];
        ypart[rflat] = __float2half(t);
    }
}

// acc[r] += sum_s coeff_s * ypart[s][r], slots in ascending order -- the
// identical fp32 add order as the legacy per-slot coeff_add chain.
__global__ void moe_gemv_reduce_batch_kernel(
    const __half *__restrict__ ypart,
    const MoEGvSlotDev *__restrict__ slots, int k_slots, int n_elems,
    float *__restrict__ acc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;
    float t = acc[i];
    for (int s = 0; s < k_slots; ++s)
        t += slots[s].coeff * __half2float(ypart[static_cast<int64_t>(s) *
                                                       n_elems + i]);
    acc[i] = t;
}

// Slot-major batched swiglu-quant: gate ygu[s*2I + j], up ygu[s*2I + I + j]
// -> h[s*I + j] + NVFP4 pack/scales at slot offsets. Per-16-group math is
// the exact clone of moe_swiglu_precise_quant_kernel (bit-identical).
__global__ void moe_swiglu_quant_batch_kernel(const __half *__restrict__ ygu,
                                              int k_slots, int I,
                                              __half *__restrict__ h,
                                              uint8_t *__restrict__ packed,
                                              uint8_t *__restrict__ scales)
{
    const int64_t groups_per_slot = (I + 15) / 16;
    const int64_t g = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (g >= static_cast<int64_t>(k_slots) * groups_per_slot) return;
    const int s = static_cast<int>(g / groups_per_slot);
    const int j0 = static_cast<int>((g - static_cast<int64_t>(s) *
                                              groups_per_slot) * 16);
    const int len = (j0 + 16 <= I) ? 16 : (I - j0);

    float vals[16];
    float absmax = 0.0f;
    for (int i = 0; i < len; ++i) {
        const float gv = __half2float(
            ygu[static_cast<int64_t>(s) * 2 * I + j0 + i]);
        const float uv = __half2float(
            ygu[static_cast<int64_t>(s) * 2 * I + I + j0 + i]);
        const float sig = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gv)));
        const __half hv =
            __float2half(__fmul_rn(sig, __fmul_rn(gv, uv)));
        h[static_cast<int64_t>(s) * I + j0 + i] = hv;
        vals[i] = __half2float(hv);
        absmax = fmaxf(absmax, fabsf(vals[i]));
    }
    const uint8_t sb =
        lsxcommon::nvfp4::FloatToE4m3(absmax * (1.0f / 6.0f));
    scales[g] = sb;
    const float inv_s = gpu::nvfp4::E4M3ScaleToFloat(sb);
    const float inv = inv_s > 0.0f ? 1.0f / inv_s : 0.0f;
    const int64_t slot_base = static_cast<int64_t>(s) * I;
    for (int i = 0; i < len; i += 2) {
        const uint8_t lo =
            lsxcommon::kernels::QuantizeE2m1TieUp(vals[i] * inv);
        const uint8_t hi = (i + 1 < len)
                               ? lsxcommon::kernels::QuantizeE2m1TieUp(
                                     vals[i + 1] * inv)
                               : 0;
        packed[(slot_base + j0 + i) / 2] =
            static_cast<uint8_t>((hi << 4) | lo);
    }
}

int MoeGemvPickKp(int cols, int rows)
{
    for (int cand : {8, 4, 2}) {
        if (cols / cand >= 1024 &&
            static_cast<int64_t>(rows) * cand >= 1024)
            return cand;
    }
    return 1;
}

// tbl[k] = {meta[d_routes[k].eid].*, coeff} -- composed fully on device so
// the no-handshake decode path never reads router output back to host.
__global__ void moe_gemv_gather_table_kernel(
    const MoERoutedSlot *__restrict__ routes,
    const MoEExpertMetaDev *__restrict__ meta,
    int k_slots,
    MoEGvSlotDev *__restrict__ out)
{
    const int k = threadIdx.x;
    if (k >= k_slots) return;   // launch width fixed at 32, K <= 16
    const MoERoutedSlot r = routes[k];
    const MoEExpertMetaDev m = meta[r.eid];
    out[k].dev_slot = m.dev_slot;
    out[k].gs1 = m.gs1;
    out[k].gs2 = m.gs2;
    out[k].gs3 = m.gs3;
    out[k].coeff = r.coeff;
}

} // namespace

cudaError_t RunMoEGemvGatherTable(
    cudaStream_t stream,
    const MoERoutedSlot *d_slots, int k_slots,
    const MoEExpertMetaDev *d_meta,
    MoEGvSlotDev *d_tbl_out)
{
    if (!d_slots || !d_meta || !d_tbl_out || k_slots <= 0 || k_slots > 16)
        return cudaErrorInvalidValue;
    moe_gemv_gather_table_kernel<<<1, 32, 0, stream>>>(d_slots, d_meta,
                                                       k_slots, d_tbl_out);
    return cudaGetLastError();
}

cudaError_t RunMoESwigluQuantBatch(cudaStream_t stream, const __half *ygu,
                                       int k_slots, int I, __half *h,
                                       uint8_t *packed, uint8_t *scales)
{
    if (!ygu || !h || !packed || !scales || k_slots <= 0 || k_slots > 16 ||
        I <= 0)
        return cudaErrorInvalidValue;
    const int64_t groups =
        static_cast<int64_t>(k_slots) * ((I + 15) / 16);
    moe_swiglu_quant_batch_kernel<<<static_cast<int>((groups + 255) / 256),
                                    256, 0, stream>>>(ygu, k_slots, I, h,
                                                      packed, scales);
    return cudaGetLastError();
}

namespace {

// Chain shared by the router-gather and device-table batched TC entry points.
// Preserves the w13 -> swiglu-quant -> w2 -> ascending reduce op order of the
// legacy per-slot path (bit-exact per slot). The ev2..ev5 handles, when
// non-null, mark stage boundaries for the LSX_PROBE_FFN diagnostic.
cudaError_t RunMoERoutedTcChain(
    cudaStream_t stream, const gpu::ExpertDeviceCache &cache,
    const MoEGvSlotDev *d_tbl, const uint8_t *xq, const uint8_t *xqs,
    int K, int H, int I, int64_t w13_bytes, int64_t s13_bytes, float *d_acc,
    __half *ygu_batch, __half *h_batch, __half *yp_batch,
    uint8_t *qpack_batch, uint8_t *qscales_batch, float *splitk13,
    cudaEvent_t ev2, cudaEvent_t ev3, cudaEvent_t ev4, cudaEvent_t ev5)
{
    const auto *tbl =
        reinterpret_cast<const lsx_tc::LsxTcBatchSlot *>(d_tbl);
    cudaError_t e = lsx_tc::RunLsxTcGemmBatch13(
        stream, xq, xqs, cache.d_pack, cache.d_scale, cache.pack_stride,
        cache.scale_stride, tbl, K, H, 2 * I, ygu_batch, splitk13);
    if (e != cudaSuccess) return e;
    if (ev2) cudaEventRecord(ev2, stream);
    e = RunMoESwigluQuantBatch(stream, ygu_batch, K, I, h_batch,
                                   qpack_batch, qscales_batch);
    if (e != cudaSuccess) return e;
    if (ev3) cudaEventRecord(ev3, stream);
    e = lsx_tc::RunLsxTcGemmBatchDown(
        stream, qpack_batch, qscales_batch, cache.d_pack, cache.d_scale,
        cache.pack_stride, cache.scale_stride, tbl, K, w13_bytes, s13_bytes,
        I, H, yp_batch, nullptr);
    if (e != cudaSuccess) return e;
    if (ev4) cudaEventRecord(ev4, stream);
    // Ascending-slot fp32 reduce -- the legacy coeff_add order (k=0..K-1);
    // acc already seeded by the caller (moe_init_acc_kernel upstream or
    // a zero memset).
    moe_gemv_reduce_batch_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
        yp_batch, d_tbl, K, H, d_acc);
    if (ev5) cudaEventRecord(ev5, stream);
    return cudaGetLastError();
}

} // namespace

cudaError_t RunMoERoutedTcBatch(
    cudaStream_t stream, MoEExpertStore &store,
    const gpu::ExpertDeviceCache &cache, int layer,
    const MoERoutedSlot *d_routes, const uint8_t *xq, const uint8_t *xqs,
    int K, int H, int I, const __half *d_resid, float *d_acc,
    __half *ygu_batch, __half *h_batch, __half *yp_batch,
    uint8_t *qpack_batch, uint8_t *qscales_batch, float *splitk13,
    MoEGvSlotDev *d_tbl)
{
    if (!d_routes || !xq || !xqs || !d_resid || !d_acc || !ygu_batch ||
        !h_batch || !yp_batch || !qpack_batch || !qscales_batch || !d_tbl ||
        K <= 0 || K > kMaxTopK || H <= 0 || I <= 0 || (I & 15) != 0 ||
        !store.d_gemv_meta || layer < 0 || layer >= store.num_layers)
        return cudaErrorInvalidValue;
    const int E = store.num_experts;
    const int64_t w13_bytes =
        store.records[static_cast<size_t>(layer)][0].w1_bytes +
        store.records[static_cast<size_t>(layer)][0].w3_bytes;
    const int64_t s13_bytes =
        store.records[static_cast<size_t>(layer)][0].s1_bytes +
        store.records[static_cast<size_t>(layer)][0].s3_bytes;

    const bool kProbe = std::getenv("LSX_PROBE_FFN") != nullptr;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr, ev3 = nullptr,
                ev4 = nullptr, ev5 = nullptr;
    if (kProbe) {
        for (auto *ev : {&ev0, &ev1, &ev2, &ev3, &ev4, &ev5})
            cudaEventCreate(ev);
        cudaEventRecord(ev0, stream);
    }
    cudaError_t e = RunMoEGemvGatherTable(
        stream, d_routes, K,
        store.d_gemv_meta + static_cast<int64_t>(layer) * E, d_tbl);
    if (e != cudaSuccess) return e;
    if (kProbe) cudaEventRecord(ev1, stream);
    e = RunMoERoutedTcChain(stream, cache, d_tbl, xq, xqs, K, H, I,
                                w13_bytes, s13_bytes, d_acc, ygu_batch,
                                h_batch, yp_batch, qpack_batch, qscales_batch,
                                splitk13, ev2, ev3, ev4, ev5);
    if (e != cudaSuccess) return e;
    if (kProbe) {
        cudaEventSynchronize(ev5);
        float t[5];
        cudaEventElapsedTime(&t[0], ev0, ev1);
        cudaEventElapsedTime(&t[1], ev1, ev2);
        cudaEventElapsedTime(&t[2], ev2, ev3);
        cudaEventElapsedTime(&t[3], ev3, ev4);
        cudaEventElapsedTime(&t[4], ev4, ev5);
        std::fprintf(stderr,
                     "[FFN-PROBE] L=%d gather=%.3f w13=%.3f swiglu=%.3f "
                     "w2=%.3f reduce=%.3f ms\n",
                     layer, t[0], t[1], t[2], t[3], t[4]);
        for (auto *ev : {&ev0, &ev1, &ev2, &ev3, &ev4, &ev5})
            cudaEventDestroy(*ev);
    }
    return cudaGetLastError();
}

cudaError_t RunMoERoutedTcBatchTbl(
    cudaStream_t stream, const gpu::ExpertDeviceCache &cache,
    const MoEGvSlotDev *d_tbl, const uint8_t *xq, const uint8_t *xqs,
    int K, int H, int I, int64_t w13_bytes, int64_t s13_bytes,
    float *d_acc, __half *ygu_batch, __half *h_batch, __half *yp_batch,
    uint8_t *qpack_batch, uint8_t *qscales_batch, float *splitk13)
{
    if (!d_tbl || !xq || !xqs || !d_acc || !ygu_batch || !h_batch ||
        !yp_batch || !qpack_batch || !qscales_batch || K <= 0 ||
        K > kMaxTopK || H <= 0 || I <= 0 || (I & 15) != 0 ||
        w13_bytes <= 0 || s13_bytes <= 0)
        return cudaErrorInvalidValue;
    return RunMoERoutedTcChain(stream, cache, d_tbl, xq, xqs, K, H, I,
                                   w13_bytes, s13_bytes, d_acc, ygu_batch,
                                   h_batch, yp_batch, qpack_batch,
                                   qscales_batch, splitk13, nullptr, nullptr,
                                   nullptr, nullptr);
}

// Device-table variant consumed by the no-handshake decode path. Requires
// the table to already live in device memory (built by the host in the
// handshake flow or by moe_gemv_gather_table_kernel otherwise).
cudaError_t RunMoERoutedGemmVBatchCore(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const MoEGvSlotDev *slots_dev, int k_slots,
    int64_t w13_bytes, int64_t s13_bytes,
    int H, int I, const __half *d_x,
    __half *ygu, __half *hbuf, __half *ypart, float *acc)
{
    if (!d_pack_base || !d_scale_base || !slots_dev || !d_x || !ygu ||
        !hbuf || !ypart || !acc || k_slots <= 0 || k_slots > 16 ||
        H <= 0 || I <= 0 || w13_bytes <= 0 || s13_bytes <= 0 ||
        (H & 31) != 0 || (I & 31) != 0)
        return cudaErrorInvalidValue;

    cudaError_t e;
    {
        const int rows = k_slots * 2 * I;
        const int kp = MoeGemvPickKp(H, rows);
        constexpr int WARPS = kGemvBatchWarps;
        auto launch = [&](auto tag) -> cudaError_t {
            constexpr int KPc = decltype(tag)::value;
            const int ROWSp = WARPS / KPc;
            moe_gemv_w13_batch_kernel<KPc><<<(rows + ROWSp - 1) / ROWSp,
                                             WARPS * 32, 0, stream>>>(
                d_x, d_pack_base, d_scale_base, pack_stride, scale_stride,
                slots_dev, k_slots, I, H, ygu);
            return cudaGetLastError();
        };
        switch (kp) {
        case 8:  e = launch(std::integral_constant<int, 8>{}); break;
        case 4:  e = launch(std::integral_constant<int, 4>{}); break;
        case 2:  e = launch(std::integral_constant<int, 2>{}); break;
        default: e = launch(std::integral_constant<int, 1>{}); break;
        }
    }
    if (e != cudaSuccess) return e;

    const int n_swiglu = k_slots * I;
    moe_gemv_swiglu_batch_kernel<<<(n_swiglu + 255) / 256, 256, 0, stream>>>(
        ygu, n_swiglu, I, hbuf);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    {
        const int rows = k_slots * H;
        const int kp = MoeGemvPickKp(I, rows);
        constexpr int WARPS = kGemvBatchWarps;
        auto launch = [&](auto tag) -> cudaError_t {
            constexpr int KPc = decltype(tag)::value;
            const int ROWSp = WARPS / KPc;
            moe_gemv_down_batch_kernel<KPc><<<(rows + ROWSp - 1) / ROWSp,
                                              WARPS * 32, 0, stream>>>(
                hbuf, d_pack_base, d_scale_base, pack_stride, scale_stride,
                w13_bytes, s13_bytes, slots_dev, k_slots, I, H, ypart);
            return cudaGetLastError();
        };
        switch (kp) {
        case 8:  e = launch(std::integral_constant<int, 8>{}); break;
        case 4:  e = launch(std::integral_constant<int, 4>{}); break;
        case 2:  e = launch(std::integral_constant<int, 2>{}); break;
        default: e = launch(std::integral_constant<int, 1>{}); break;
        }
    }
    if (e != cudaSuccess) return e;

    moe_gemv_reduce_batch_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
        ypart, slots_dev, k_slots, H, acc);
    return cudaGetLastError();
}

cudaError_t RunMoERoutedGemmVBatch(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const MoEGvSlotDev *slots_host_tbl, int k_slots,
    int64_t w13_bytes, int64_t s13_bytes,
    int H, int I, const __half *d_x,
    __half *ygu, __half *hbuf, __half *ypart, float *acc)
{
    // Handshake wrapper: builds the slot table on the host and uploads it
    // (tests / callers that already hold a host-side table).
    if (!slots_host_tbl) return cudaErrorInvalidValue;
    MoEGvSlotDev *d_tbl = nullptr;
    cudaError_t e = cudaMalloc(&d_tbl,
                               sizeof(MoEGvSlotDev) * k_slots);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(d_tbl, slots_host_tbl,
                        sizeof(MoEGvSlotDev) * k_slots,
                        cudaMemcpyHostToDevice, stream);
    if (e == cudaSuccess)
        e = RunMoERoutedGemmVBatchCore(stream, d_pack_base, d_scale_base, pack_stride,
                              scale_stride, d_tbl, k_slots, w13_bytes,
                              s13_bytes, H, I, d_x, ygu, hbuf, ypart, acc);
    cudaFree(d_tbl);
    return e == cudaSuccess ? cudaGetLastError() : e;
}

// No-handshake readiness: every routed expert has been staged at least once
// and its meta mirrored. With full expert residency (default since round 18)
// this flips on after one warm pass and then never reverts -- evictions are
// impossible when cache.slots >= total experts, so the device mirror stays
// authoritative without host readback.
bool MoEGemvNohandshake(const MoEExpertStore &store)
{
    const size_t flats =
        static_cast<size_t>(store.num_layers) * store.num_experts;
    if (flats == 0 || store.h_gemv_meta.size() != flats ||
        store.gemv_ready)
        return store.gemv_ready && !store.h_gemv_meta.empty();
    for (const auto &m : store.h_gemv_meta)
        if (m.dev_slot < 0) return false;   // warm-up incomplete
    // First complete mirror: upload once.
    auto &s = const_cast<MoEExpertStore &>(store);
    if (cudaMalloc(reinterpret_cast<void **>(&s.d_gemv_meta),
                   flats * sizeof(moe::MoEExpertMetaDev)) != cudaSuccess) {
        s.d_gemv_meta = nullptr;
        return false;
    }
    if (cudaMemcpy(s.d_gemv_meta, s.h_gemv_meta.data(),
                   flats * sizeof(moe::MoEExpertMetaDev),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(s.d_gemv_meta);
        s.d_gemv_meta = nullptr;
        return false;
    }
    s.gemv_ready = true;
    return true;
}

// Router-readback-free routed FFN for decode: composes the GEMV slot table
// from d_slots + the device meta mirror, runs the batched chain, no host
// synchronization at any point. Scratch slices are caller-owned when the
// generation arena provides them; nulls fall back to cudaMallocAsync.
cudaError_t RunMoERoutedGemmVNoHandshake(
    cudaStream_t stream, MoEExpertStore &store,
    gpu::ExpertDeviceCache &cache, int layer,
    const MoERoutedSlot *d_slots, int K,
    const __half *d_x, int H, int I, float *d_acc,
    __half *ygu_b, __half *hbuf_b, __half *ypart_b,
    MoEGvSlotDev *d_tbl,
    std::string &error)
{
    const size_t flat_base =
        static_cast<size_t>(layer) * static_cast<size_t>(store.num_experts);
    const auto &rec0 = store.records[static_cast<size_t>(layer)][0];
    const int64_t w13_b =
        static_cast<int64_t>(rec0.w1_bytes + rec0.w3_bytes);
    const int64_t s13_b =
        static_cast<int64_t>(rec0.s1_bytes + rec0.s3_bytes);

    if (!ygu_b || !hbuf_b || !ypart_b || !d_tbl) {
        error = "no-handshake scratch missing";
        return cudaErrorInvalidValue;
    }
    cudaError_t e = RunMoEGemvGatherTable(stream, d_slots, K,
                                              store.d_gemv_meta + flat_base,
                                              d_tbl);
    if (e == cudaSuccess)
        e = RunMoERoutedGemmVBatchCore(stream, cache.d_pack, cache.d_scale,
                              cache.pack_stride, cache.scale_stride, d_tbl,
                              K, w13_b, s13_b, H, I, d_x, ygu_b, hbuf_b,
                              ypart_b, d_acc);
    return e == cudaSuccess ? cudaGetLastError() : e;
}

// ── Shared expert + gate combine ──────────────────────────────────────────
namespace {

// g[t] = sigmoid(dot(shared_gate_row, x[t])) — one block per token.
// RMW isolation: the single output write happens on thread 0 after a
// block-wide tree reduction in shared memory.
__global__ void moe_shared_gate_kernel(const __half *__restrict__ gate,
                                            const __half *__restrict__ x,
                                            int H, float *__restrict__ g_out)
{
    const int t = blockIdx.x;
    const __half *xr = x + static_cast<int64_t>(t) * H;
    extern __shared__ float sh[];  // [blockDim.x] partial dots
    const int tid = threadIdx.x;
    float part = 0.f;
    for (int d = tid; d < H; d += blockDim.x)
        part += __half2float(gate[d]) * __half2float(xr[d]);
    sh[tid] = part;
    __syncthreads();
    if (tid == 0) {
        float s = 0.f;
        for (int i = 0; i < blockDim.x; ++i) s += sh[i];
        g_out[t] = 1.f / (1.f + expf(-s));
    }
}

// out[j] = fp2h(acc[j]) + g[j/H] * y_shared[j]
__global__ void moe_combine_kernel(const float *__restrict__ acc,
                                        const float *__restrict__ g,
                                        const __half *__restrict__ y_shared,
                                        __half *__restrict__ out, int total,
                                        int H)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= total) return;
    out[j] = __float2half(acc[j] +
                          g[j / H] * __half2float(y_shared[j]));
}

} // namespace

#define LSX_MOE_STAGE(name, expr)                                             \
    do {                                                                      \
        cudaError_t se_ = (expr);                                             \
        cudaError_t le_ = cudaGetLastError();                                 \
        if (se_ == cudaSuccess) se_ = le_;                                    \
        if (se_ != cudaSuccess) {                                             \
            if (std::getenv("LSX_TEST_MOE_STAGE_DEBUG"))                      \
                std::fprintf(stderr, "[stage] %s: %s\n", name,               \
                             cudaGetErrorString(se_));                        \
            return se_;                                                       \
        }                                                                     \
    } while (0)

bool MoETruncateTokensAtEos(std::vector<int32_t> &tokens, int eos_id)
{
    if (eos_id < 0) return false;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == eos_id) {
            tokens.resize(i + 1);   // legacy break keeps the EOS token
            return true;
        }
    }
    return false;
}

cudaError_t RunMoESharedCombine(
    cudaStream_t stream, const lsx_nvfp4::LsxNvfp4Weight &sw1,
    const lsx_nvfp4::LsxNvfp4Weight &sw2, const lsx_nvfp4::LsxNvfp4Weight &sw3,
    const __half *d_shared_gate, const __half *d_x, int seq, int H, int SI,
    const float *d_routed_acc, __half *d_scratch_i, __half *d_scratch_h,
    float *d_gate_val, __half *d_out, const uint8_t *xq,
    const uint8_t *xqs)
{
    // Shared expert MLP on all tokens at once. TC path fuses gate+up into
    // one N=2*SI GEMM (cached N-concatenation of the two packed weights,
    // per-segment scales via the N/2 half-split); output lands in
    // d_scratch_i extended over d_scratch_h (arena places them adjacently,
    // SI <= H keeps 2*SI within the pair). Scalar path: split pair.
    const bool tc_shared = qwen::QwenTcFastMath() && sw1.d_packed &&
                           sw3.d_packed && ((2 * SI) & 15) == 0 && SI <= H;
    __half *up_base = d_scratch_h;
    if (tc_shared) {
        static std::mutex sh_mu;
        struct FusedShared {
            const uint8_t *p;
            const uint8_t *s;
            const float *w;
            int n;
        };
        static std::unordered_map<const void *, FusedShared> sh_cache;
        const uint8_t *fp = nullptr, *fs = nullptr;
        const float *fw = nullptr;
        const int *fb_dummy = nullptr;
        int fn = 0;
        {
            std::lock_guard<std::mutex> lk(sh_mu);
            auto it = sh_cache.find(sw1.d_packed);
            if (it == sh_cache.end()) {
                cudaError_t be = lsx_tc::LsxTcBuildFusedGemmWeights(
                    stream, sw1.d_packed, sw1.d_scales, SI, sw3.d_packed,
                    sw3.d_scales, SI, nullptr, nullptr, 0, nullptr, nullptr,
                    0, H / 2, (H + 15) / 16, sw1.global_scale,
                    sw3.global_scale, 1.0f, 1.0f, &fp, &fs, &fw, &fb_dummy,
                    &fn);
                if (be != cudaSuccess) return be;
                sh_cache.emplace(sw1.d_packed, FusedShared{fp, fs, fw, fn});
            } else {
                fp = it->second.p;
                fs = it->second.s;
                fw = it->second.w;
                fn = it->second.n;
            }
        }
        if (xq && xqs) {
            LSX_MOE_STAGE("shared_fused_gemm_pq",
                          lsx_tc::RunLsxTcGemmPreQuant(
                              stream, xq, xqs, fp, fs, 1.0f, d_scratch_i,
                              seq, H, fn, false, fw, nullptr));
        } else {
            LSX_MOE_STAGE("shared_fused_gemm",
                          lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, fp, fs, 1.0f,
                                                 d_scratch_i, seq, H, fn,
                                                 false, fw, nullptr));
        }
        up_base = d_scratch_i + static_cast<int64_t>(seq) * SI;
    } else {
        if (lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, sw1.d_packed, sw1.d_scales,
                                   sw1.global_scale, d_scratch_i, seq, H, SI,
                                   false) != cudaSuccess)
            return cudaErrorInvalidValue;
        if (lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_x, sw3.d_packed, sw3.d_scales,
                                   sw3.global_scale, d_scratch_h, seq, H, SI,
                                   false) != cudaSuccess)
            return cudaErrorInvalidValue;
    }
    {
        ::lsxcommon::kernels::SwigluKernel<__half,
                                           ::lsxcommon::kernels::ExpFlavor::Precise, false>
            <<<EltGrid(seq * SI), kEltBlock, 0, stream>>>(
                d_scratch_i, up_base, d_scratch_i, seq * SI, 0.f);
        cudaError_t se_ = cudaGetLastError();
        if (se_ != cudaSuccess) {
            if (std::getenv("LSX_TEST_MOE_STAGE_DEBUG"))
                std::fprintf(stderr, "[stage] shared_swiglu: %s\n",
                             cudaGetErrorString(se_));
            return se_;
        }
    }
    if (lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_scratch_i, sw2.d_packed, sw2.d_scales,
                               sw2.global_scale, d_scratch_h, seq, SI, H,
                               false) != cudaSuccess)
        return cudaErrorInvalidValue;
    // Scalar gate per token.
    moe_shared_gate_kernel<<<seq, kEltBlock, kEltBlock * sizeof(float),
                                  stream>>>(d_shared_gate, d_x, H, d_gate_val);
    cudaError_t ge = cudaGetLastError();
    if (std::getenv("LSX_TEST_MOE_STAGE_DEBUG") && ge != cudaSuccess)
        std::fprintf(stderr, "[stage] shared_gate: %s\n",
                     cudaGetErrorString(ge));
    // Combine routed accumulation + gated shared expert.
    moe_combine_kernel<<<EltGrid(seq * H), kEltBlock, 0, stream>>>(
        d_routed_acc, d_gate_val, d_scratch_h, d_out, seq * H, H);
    ge = cudaGetLastError();
    if (std::getenv("LSX_TEST_MOE_STAGE_DEBUG") && ge != cudaSuccess)
        std::fprintf(stderr, "[stage] combine: %s\n",
                     cudaGetErrorString(ge));
    return ge;
}

// ── Generation driver (incremental path, MoE FFN) ─────────────────────────
namespace {

// acc[i] = f32(residual[i]) — seeds the MoE fp32 accumulator with the
// running residual so the final combine writes act + moe_out in one step.
__global__ void moe_init_acc_kernel(const __half *__restrict__ resid,
                                         float *__restrict__ acc, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] = __half2float(resid[i]);
}


// Residual add: unified kernels::AddInPlaceKernel (lsx_kernels_math.cuh).
} // namespace

cudaError_t RunMoEAccInitForTest(cudaStream_t stream,
                                     const __half *d_resid, float *d_acc,
                                     int n)
{
    if (!d_resid || !d_acc || n <= 0) return cudaErrorInvalidValue;
    moe_init_acc_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        d_resid, d_acc, n);
    return cudaGetLastError();
}

void EnsureMoEExpertView(moe::MoEExpertDb &db, int layer, int eid,
                         cudaStream_t stream)
{
    if (layer < 0 || layer >= db.num_layers || eid < 0 ||
        eid >= db.experts_per_layer)
        return;
    const size_t idx = static_cast<size_t>(layer) *
                           static_cast<size_t>(db.experts_per_layer) +
                       static_cast<size_t>(eid);
    if (idx < db.entries.size()) db.EnsureResident(layer, eid, stream);
}

int EnsureMoEExpertResident(moe::MoEExpertStore &store, int layer, int eid,
                                gpu::ExpertDeviceCache &cache,
                                cudaStream_t stream, std::string &error,
                                cudaStream_t h2d_stream, bool *missed,
                                int *evicted_slot)
{
    if (missed) *missed = false;
    if (evicted_slot) *evicted_slot = -1;
    if (layer < 0 || layer >= store.num_layers || eid < 0 ||
        eid >= store.num_experts) {
        error = absl::StrCat("expert out of range: layer ", layer, " eid ", eid);
        return -1;
    }
    const int flat = layer * store.num_experts + eid;
    int slot = cache.Lookup(flat);
    if (slot >= 0) return slot;
    slot = cache.Insert(flat, evicted_slot);
    if (missed) *missed = true;

    // No-handshake decode bookkeeping: mirror residency + scales so the
    // batched GEMV path can compose its table entirely on device.
    if (static_cast<int>(store.h_gemv_meta.size()) <
        store.num_layers * store.num_experts)
        store.h_gemv_meta.assign(
            static_cast<size_t>(store.num_layers) * store.num_experts,
            MoEExpertMetaDev{-1, 0.f, 0.f, 0.f});
    {
        const auto &rec =
            store.records[static_cast<size_t>(layer)]
                         [static_cast<size_t>(eid)];
        store.h_gemv_meta[static_cast<size_t>(flat)] =
            moe::MoEExpertMetaDev{slot, rec.gs1, rec.gs2, rec.gs3};
        store.gemv_ready = false;   // invalidate until re-uploaded
    }

    const size_t eidx = static_cast<size_t>(layer) *
                            static_cast<size_t>(store.num_experts) +
                        static_cast<size_t>(eid);
    cudaStream_t copy_stream = h2d_stream ? h2d_stream : stream;
    uint8_t *dst_pack =
        cache.d_pack + static_cast<int64_t>(slot) * cache.pack_stride;
    uint8_t *dst_scale =
        cache.d_scale + static_cast<int64_t>(slot) * cache.scale_stride;

    // Two storage representations share this record type: the neutral view
    // store (qwen MoE, ornith, mistral) and the legacy pinned arena (glm-5.3-
    // flash, docs/OPEN_TASKS.md §V). Pick whichever actually holds the payload
    // rather than assuming one, or a family that was not migrated stops staging
    // entirely and the failure surfaces as cudaErrorUnknown from the kernel.
    // Decide by WHICH STORE IS POPULATED, then map before probing: a cold entry
    // has a null pointer until EnsureMoEExpertView maps it, so probing first
    // makes every first-touch look "unavailable" (ornith prestage died on
    // layer 0 eid 0 that way).
    const bool view_backed = eidx < store.experts.entries.size() &&
                             !store.experts.entries.empty();
    if (view_backed) {
        EnsureMoEExpertView(store.experts, layer, eid, stream);
        if (store.experts.entries[eidx].host_w1 == nullptr) {
            error = absl::StrCat("expert host view unavailable: layer ", layer,
                                 " eid ", eid);
            return -1;
        }
        const MoEExpertEntry &v = store.experts.entries[eidx];
        // Segment ORDER is load-bearing: the device slot is FUSED for
        // inference, [w13 = w1|w3][w2] and [s13 = s1|s3][s2], which is what the
        // batched kernels size off (records[..][0].w1_bytes + w3_bytes for the
        // w13 region). The pre-ExpertStore staging was already segment-wise
        // with this same order, so nothing about the layout moves.
        struct Seg { const uint8_t *src; int64_t bytes; };
        const Seg psegs[3] = {{v.host_w1, v.w1_bytes},
                              {v.host_w3, v.w3_bytes},
                              {v.host_w2, v.w2_bytes}};
        const Seg ssegs[3] = {{v.host_w1_scales, v.w1_scales_bytes},
                              {v.host_w3_scales, v.w3_scales_bytes},
                              {v.host_w2_scales, v.w2_scales_bytes}};
        uint8_t *pk = dst_pack;
        for (const auto &seg : psegs) {
            if (cudaMemcpyAsync(pk, seg.src, static_cast<size_t>(seg.bytes),
                                cudaMemcpyHostToDevice,
                                copy_stream) != cudaSuccess) {
                error = "expert H2D pack staging failed";
                return -1;
            }
            pk += seg.bytes;
        }
        uint8_t *sc = dst_scale;
        for (const auto &seg : ssegs) {
            if (cudaMemcpyAsync(sc, seg.src, static_cast<size_t>(seg.bytes),
                                cudaMemcpyHostToDevice,
                                copy_stream) != cudaSuccess) {
                error = "expert H2D scale staging failed";
                return -1;
            }
            sc += seg.bytes;
        }
    } else {
        const uint8_t *slab = static_cast<size_t>(layer) < store.h_slabs.size()
                                 ? store.h_slabs[static_cast<size_t>(layer)]
                                 : nullptr;
        if (!slab) {
            error = absl::StrCat("expert host payload unavailable: layer ",
                                 layer, " eid ", eid);
            return -1;
        }
        const auto &r = store.records[static_cast<size_t>(layer)]
                                      [static_cast<size_t>(eid)];
        struct Seg { int64_t off; int64_t bytes; };
        const Seg psegs[3] = {{r.slab_w1_off, r.w1_bytes},
                              {r.slab_w3_off, r.w3_bytes},
                              {r.slab_w2_off, r.w2_bytes}};
        const Seg ssegs[3] = {{r.slab_s1_off, r.s1_bytes},
                              {r.slab_s3_off, r.s3_bytes},
                              {r.slab_s2_off, r.s2_bytes}};
        uint8_t *pk = dst_pack;
        for (const auto &seg : psegs) {
            if (cudaMemcpyAsync(pk, slab + seg.off,
                                static_cast<size_t>(seg.bytes),
                                cudaMemcpyHostToDevice,
                                copy_stream) != cudaSuccess) {
                error = "expert H2D pack staging failed";
                return -1;
            }
            pk += seg.bytes;
        }
        uint8_t *sc = dst_scale;
        for (const auto &seg : ssegs) {
            if (cudaMemcpyAsync(sc, slab + seg.off,
                                static_cast<size_t>(seg.bytes),
                                cudaMemcpyHostToDevice,
                                copy_stream) != cudaSuccess) {
                error = "expert H2D scale staging failed";
                return -1;
            }
            sc += seg.bytes;
        }
    }
    return slot;
}

// ── Routed-expert host store ───────────────────────────────────────────────

void MoEExpertStore::Free()
{
    for (auto &p : h_slabs)
        if (p) cudaFreeHost(p);
    h_slabs.clear();
    slab_bytes.clear();
    records.clear();
    if (d_seg_wgs) cudaFree(d_seg_wgs);
    if (d_seg_bounds) cudaFree(d_seg_bounds);
    d_seg_wgs = nullptr;
    d_seg_bounds = nullptr;
    if (d_gemv_meta) cudaFree(d_gemv_meta);
    d_gemv_meta = nullptr;
    h_gemv_meta.clear();
    gemv_ready = false;
    num_layers = 0;
    num_experts = 0;
}

// ── Routed-expert store: load + device cache ────────────────────────────────

bool LoadMoEExpertStore(const std::string &model_path,
                            MoEExpertStore &store, std::string &error)
{
    namespace fs = std::filesystem;
    store.Free();
    error.clear();

    fs::path base = fs::path(model_path) / "logestix";
    if (!fs::exists(base))
        base = fs::path(model_path) / "logestix-qwen35moe";  // legacy layout
    const fs::path experts_dir = base / "experts";

    // Discover layer count from directory names.
    int max_layer = -1;
    for (fs::directory_iterator it(experts_dir), end; it != end; ++it) {
        const std::string name = it->path().filename().string();
        if (name.rfind("layer_", 0) == 0) {
            int l = 0;
            if (absl::SimpleAtoi(name.substr(6), &l))
                max_layer = std::max(max_layer, l);
        }
    }
    if (max_layer < 0) {
        error = "no expert layer dirs under " + experts_dir.string();
        return false;
    }
    store.num_layers = max_layer + 1;
    store.records.resize(static_cast<size_t>(store.num_layers));

    // Expert count from the files in the first layer dir; every later layer
    // must have the same count or the store geometry would lie.
    if (store.num_experts <= 0) {
        int e_max = -1;
        const fs::path first_dir = experts_dir / absl::StrCat("layer_", 0);
        for (fs::directory_iterator it(first_dir), dit; it != dit; ++it) {
            const std::string name = it->path().filename().string();
            if (name.rfind("exp_", 0) != 0) continue;
            const std::size_t dot = name.find('.');
            if (dot == std::string::npos) continue;
            int e = 0;
            if (absl::SimpleAtoi(name.substr(4, dot - 4), &e))
                e_max = std::max(e_max, e);
        }
        store.num_experts = e_max + 1;
    }
    if (store.num_experts <= 0) {
        error = absl::StrCat("no expert files in layer_0 under ",
                             experts_dir.string());
        return false;
    }

    // Geometry pass only. The payload is NOT staged: the neutral store maps it
    // on demand (max_open entries), so host memory tracks the cap instead of
    // the whole expert set (17 GiB for qwen MoE / ornith, 61 GiB for mistral —
    // docs/OPEN_TASKS.md §V).
    moe::MoEExpertDb &db = store.experts;
    db.num_layers = store.num_layers;
    db.experts_per_layer = store.num_experts;
    // max_open IS the host-memory knob now: peak mapped bytes ~ max_open x
    // expert size (and warm-up cost scales with it). Overridable so an operator
    // can trade RSS against re-mapping churn without a rebuild.
    db.max_open = kMoEExpertMaxOpen;
    if (const char *env = std::getenv("LSX_MOE_EXPERT_MAX_OPEN")) {
        const long v = std::atol(env);
        if (v > 0) db.max_open = static_cast<size_t>(v);
    }
    db.first_moe_layer = 0;  // every layer routes in these families
    if (!db.Load(model_path, /*fallback=*/true)) {
        error = absl::StrCat("expert store load failed: ", model_path);
        return false;
    }

    cudaStream_t map_stream = nullptr;
    for (int l = 0; l < store.num_layers; ++l) {
        auto &recs = store.records[static_cast<size_t>(l)];
        recs.assign(static_cast<size_t>(store.num_experts), {});
        for (int e = 0; e < store.num_experts; ++e) {
            EnsureMoEExpertView(db, l, e, map_stream);
            const size_t idx = static_cast<size_t>(l) *
                                   static_cast<size_t>(store.num_experts) +
                               static_cast<size_t>(e);
            if (idx >= db.entries.size() || !db.entries[idx].host_w1) {
                error = absl::StrCat("expert read failed: layer_", l, "/exp_", e);
                return false;
            }
            const MoEExpertEntry &v = db.entries[idx];
            auto &r = recs[static_cast<size_t>(e)];
            r.w1_bytes = v.w1_bytes;
            r.w2_bytes = v.w2_bytes;
            r.w3_bytes = v.w3_bytes;
            r.s1_bytes = v.w1_scales_bytes;
            r.s2_bytes = v.w2_scales_bytes;
            r.s3_bytes = v.w3_scales_bytes;
            r.gs1 = v.w1_global_scale;
            r.gs2 = v.w2_global_scale;
            r.gs3 = v.w3_global_scale;
        }
        log::info(absl::StrCat("[moe_load] layer_", l, ": ",
                               store.num_experts, " experts mapped (views)"));
    }

    // Device-side per-expert {gs1, gs3} pairs for the fused w13 GEMM
    // epilogue: the kernel treats a non-null seg_wgs with null bounds as a
    // two-segment split at N/2 (gate | up).
    const int flats = store.num_layers * store.num_experts;
    if (flats > 0) {
        std::vector<float> hw(static_cast<size_t>(flats) * 2);
        for (int l = 0; l < store.num_layers; ++l)
            for (int e = 0; e < store.num_experts; ++e) {
                const auto &r =
                    store.records[static_cast<size_t>(l)]
                                 [static_cast<size_t>(e)];
                hw[(static_cast<size_t>(l) * store.num_experts + e) * 2] =
                    static_cast<float>(r.gs1);
                hw[(static_cast<size_t>(l) * store.num_experts + e) * 2 + 1] =
                    static_cast<float>(r.gs3);
            }
        if (cudaMalloc(&store.d_seg_wgs, hw.size() * sizeof(float)) !=
            cudaSuccess)
            return false;
        if (cudaMemcpy(store.d_seg_wgs, hw.data(), hw.size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return false;
    }
    return true;
}

bool AllocateMoEExpertCache(MoEExpertStore &store, int slots,
                                gpu::ExpertDeviceCache &cache,
                                std::string &error)
{
    if (slots <= 0 || store.num_layers <= 0 || store.num_experts <= 0 ||
        store.records.empty() ||
        (store.experts.entries.empty() && store.h_slabs.empty())) {
        error = "expert store not loaded";
        return false;
    }
    // Slot strides sized to the largest expert payload across layers.
    size_t max_pack = 0, max_scale = 0;
    for (int l = 0; l < store.num_layers; ++l) {
        for (const auto &r : store.records[static_cast<size_t>(l)]) {
            const size_t pack = static_cast<size_t>(r.w1_bytes + r.w2_bytes +
                                                    r.w3_bytes);
            const size_t scale = static_cast<size_t>(r.s1_bytes + r.s2_bytes +
                                                     r.s3_bytes);
            max_pack = std::max(max_pack, pack);
            max_scale = std::max(max_scale, scale);
        }
    }
    cache.slots = slots;
    cache.total_experts = store.num_layers * store.num_experts;
    cache.pack_stride = AlignUp(max_pack, kArenaAlign);
    cache.scale_stride = AlignUp(max_scale, kArenaAlign);
    const size_t bytes = cache.DeviceBytes();

    void *base = nullptr;
    if (cudaMalloc(&base, bytes) != cudaSuccess) {
        error = absl::StrCat("expert device block alloc failed: ",
                             cudaGetErrorString(cudaGetLastError()),
                             " (", (bytes >> 20), " MiB requested)");
        return false;
    }
    cache.d_base = static_cast<uint8_t *>(base);
    cache.d_pack = cache.d_base;
    cache.d_scale = cache.d_base +
                    static_cast<int64_t>(slots) * cache.pack_stride;
    // Host LRU bookkeeping owned by the cache allocation.
    cache.h_slot_of = new int32_t[static_cast<size_t>(cache.total_experts)];
    cache.h_slot_owner = new int32_t[static_cast<size_t>(slots)];
    cache.h_lru = new int32_t[static_cast<size_t>(slots)];
    for (int i = 0; i < cache.total_experts; ++i)
        cache.h_slot_of[i] = -1;
    for (int i = 0; i < slots; ++i) {
        cache.h_slot_owner[i] = -1;
        cache.h_lru[i] = i;
    }
    cache.hits = 0;
    cache.misses = 0;
    return true;
}

void FreeMoEExpertCache(gpu::ExpertDeviceCache &cache)
{
    if (cache.d_base) {
        cudaFree(cache.d_base);
        cache.d_base = nullptr;
    }
    delete[] cache.h_slot_of;
    delete[] cache.h_slot_owner;
    delete[] cache.h_lru;
    cache.h_slot_of = nullptr;
    cache.h_slot_owner = nullptr;
    cache.h_lru = nullptr;
    cache.d_pack = nullptr;
    cache.d_scale = nullptr;
    cache.slots = 0;
}

} // namespace moe
} // namespace lsxcommon
