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
#include "lsxcommon/qwen_moe_load.h"
#include "lsxcommon/shared_experts.h"
#include "lsxcommon/qwen_tc_gemm.h"
#include "nvfp4_codec.h"
#include "nvfp4_scale.h"
#include "nvfp4_gemv_math.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_act.cuh"
#include "lsx_kernels_quant.cuh"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lsxcommon {
namespace qwen_moe {
namespace {

constexpr int kRouterBlock = 128;
constexpr int kMaxTopK = 16;

static_assert(sizeof(qwen::tc::QwenTcBatchSlot) == sizeof(QwenMoeGvSlotDev) &&
                  alignof(qwen::tc::QwenTcBatchSlot) ==
                      alignof(QwenMoeGvSlotDev),
              "QwenTcBatchSlot must stay layout-identical to QwenMoeGvSlotDev");

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

// Decode-split router: stage 1 computes logits with ONE BLOCK PER EXPERT
// running the exact same row walk as the monolithic kernel (bit-identical
// accumulation order via QwenRouterDotRow); the E blocks stream their rows
// from different SMs instead of funneling 1 MB through a single block. The
// 8 independent accumulator chains also cut the dependent-add latency chain
// of the scalar loop (~265us -> tens of us per token at E=256/H=2048).
__global__ void qwen_moe_router_dots_kernel(
    const __nv_bfloat16 *__restrict__ proj, const __half *__restrict__ x,
    int H, float *__restrict__ logits)
{
    const int e = blockIdx.x;
    if (threadIdx.x != 0) return;
    const __half *xr = x;   // decode: t == 0
    const __nv_bfloat16 *pr = proj + static_cast<int64_t>(e) * H;
    logits[e] = QwenRouterDotRow(pr, xr, H);
}

// Stage 2: softmax -> top-k -> renorm from precomputed logits (unchanged
// HF-parity selection logic).
__global__ void qwen_moe_router_select_kernel(
    const float *__restrict__ logits, int K, int E,
    QwenMoeRoutedSlot *__restrict__ slots)
{
    const int t = blockIdx.x;
    extern __shared__ float sh[];
    float *probs = sh;   // [E]
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
        float sm = 0.f;
        for (int i = 0; i < K && i <= n; ++i) sm += val[i];
        QwenMoeRoutedSlot *out = slots + static_cast<int64_t>(t) * K;
        for (int i = 0; i < K; ++i) {
            out[i].eid = idx[i];
            out[i].coeff = val[i] / sm;
        }
    }
}

__global__ void qwen_moe_router_topk_kernel(
    const __nv_bfloat16 *__restrict__ proj, const __half *__restrict__ x,
    int H, int K, int E, QwenMoeRoutedSlot *__restrict__ slots)
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
        qwen_moe_router_dots_kernel<<<E, 32, 0, stream>>>(router_w, normed, H,
                                                          d_logits);
        le = cudaGetLastError();
        if (le == cudaSuccess)
            qwen_moe_router_select_kernel<<<seq, kRouterBlock,
                                            sizeof(float) * E, stream>>>(
                d_logits, K, E, d_slots);
        if (le != cudaSuccess) {
            cudaFreeAsync(d_logits, stream);
            return le;
        }
        cudaFreeAsync(d_logits, stream);
        return cudaGetLastError();
    }
    qwen_moe_router_topk_kernel<<<seq, kRouterBlock, smem, stream>>>(
        router_w, normed, H, K, E, d_slots);
    return cudaGetLastError();
}

// ── Routed expert FFN slot chain ──────────────────────────────────────────
namespace {

// SwiGLU: unified kernels::SwigluKernel (lsx_kernels_act.cuh).

// Deterministic ascending-k accumulation into an fp32 accumulator.
__global__ void qwen_moe_coeff_add_kernel(const __half *__restrict__ y2,
                                          float coeff, float *__restrict__ acc,
                                          int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] += coeff * __half2float(y2[i]);
}

// F32->FP16 convert: unified kernels::F32toTKernel (lsx_kernels_math.cuh).

constexpr int kEltBlock = 256;

inline int EltGrid(int n) { return (n + kEltBlock - 1) / kEltBlock; }

} // namespace

// Defined below (batched-GEMV section); lets the handshake routed-FFN
// branch target the device-table core directly.
cudaError_t RunQwenMoeRoutedGemmVBatchCore(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const QwenMoeGvSlotDev *slots_dev, int k_slots,
    int64_t w13_bytes, int64_t s13_bytes,
    int H, int I, const __half *d_x,
    __half *ygu, __half *hbuf, __half *ypart, float *acc);

cudaError_t RunQwenMoeRoutedFFN(cudaStream_t stream, QwenMoeExpertStore &store,
                                QwenMoeExpertDeviceCache &cache, int layer,
                                const QwenMoeRoutedSlot *h_slots, int K,
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
                                QwenMoeGvSlotDev *d_gemv_tbl)
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
        dev_slots[k] = EnsureQwenMoeExpertResident(
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
        QwenMoeGvSlotDev tbl[16];
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
        QwenMoeGvSlotDev *tbl_d = d_gemv_tbl;
        __half *ygu_b = ygu_batch, *hbuf_b = hbuf_batch,
               *ypart_b = ypart_batch;
        bool owned = false;
        if (!tbl_d) {
            if (cudaMallocAsync(&tbl_d, sizeof(QwenMoeGvSlotDev) * K,
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
            cudaMemcpyAsync(tbl_d, tbl, sizeof(QwenMoeGvSlotDev) * K,
                            cudaMemcpyHostToDevice, stream);
            const cudaError_t be = RunQwenMoeRoutedGemmVBatchCore(
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
        const QwenMoeRoutedSlot &slot = h_slots[k];
        const int dev_slot = dev_slots[k];

        const auto &rec = store.records[static_cast<size_t>(layer)]
                                        [static_cast<size_t>(slot.eid)];
        uint8_t *pack =
            cache.d_pack + static_cast<int64_t>(dev_slot) * cache.pack_stride;
        uint8_t *scale =
            cache.d_scale + static_cast<int64_t>(dev_slot) * cache.scale_stride;
        // Slot layout is STAGED FUSED: [w13 = w1|w3][w2] packed,
        // [s13 = s1|s3][s2] scales (see EnsureQwenMoeExpertResident). Gate
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
            if (qwen::RunQwenNvfp4Gemm(stream, d_x, w1_split, s1_split,
                                       rec.gs1, d_yg, 1, H, I,
                                       false) != cudaSuccess)
                { error = "w1 gemm failed"; return cudaErrorInvalidValue; }
            if (qwen::RunQwenNvfp4Gemm(stream, d_x, w3_split, s3_split,
                                       rec.gs3, d_yu, 1, H, I,
                                       false) != cudaSuccess)
                { error = "w3 gemm failed"; return cudaErrorInvalidValue; }
        } else {
            cudaError_t ge;
            if (xq && xqs) {
                ge = qwen::tc::RunQwenTcGemmPreQuant(
                    stream, xq, xqs, w13, s13, 1.0f, d_yg, 1, H, 2 * I,
                    false, seg_wgs, nullptr);
            } else {
                ge = qwen::RunQwenNvfp4Gemm(stream, d_x, w13, s13, 1.0f,
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
            cudaError_t qe = qwen::RunQwenSwigluPreciseQuantInternal(
                stream, d_yg, d_yu, d_yg, ap, as, I);
            if (qe == cudaSuccess)
                qe = qwen::tc::RunQwenTcGemmPreQuant(
                    stream, ap, as, w2, s2, rec.gs2, d_y2, 1, I, H, false);
            if (own_scratch) {
                cudaFreeAsync(ap, stream);
                cudaFreeAsync(as, stream);
            }
            if (qe != cudaSuccess)
                { error = "w2 prequant gemm failed"; return cudaErrorInvalidValue; }
        } else {
            ::lsxcommon::kernels::SwigluKernel<__half, ::lsxcommon::kernels::ExpFlavor::Precise, false>
                <<<EltGrid(I), kEltBlock, 0, stream>>>(d_yg, d_yu, d_yg, I, 0.f);
            if (qwen::RunQwenNvfp4Gemm(stream, d_yg, w2, s2, rec.gs2, d_y2, 1,
                                       I, H, false) != cudaSuccess)
                { error = "w2 gemm failed"; return cudaErrorInvalidValue; }
        }
        qwen_moe_coeff_add_kernel<<<EltGrid(H), kEltBlock, 0, stream>>>(
            d_y2, slot.coeff, d_acc, H);
    }
    return cudaGetLastError();
}

cudaError_t RunQwenMoeAccToFp16(cudaStream_t stream, const float *d_acc,
                                __half *d_out, int n)
{
    lsxcommon::kernels::F32toTKernel<__half><<<EltGrid(n), kEltBlock, 0, stream>>>(d_acc, d_out, n);
    return cudaGetLastError();
}

// ── Batched routed-expert GEMV chain (round-19 decode lever) ────────────────
// Replaces the per-slot {w1,w3,w2} loop (K x 5 kernels/layer) with 4 launches:
// w13 GEMV batch -> elementwise swiglu -> down GEMV batch (per-slot partials)
// -> ordered coeff reduce. Same weight-streaming contract as the
// qwen_nvfp4_gemv_kernel core: one warp owns a K-slice of one flat output
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
    const QwenMoeGvSlotDev *__restrict__ slots,
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
        const QwenMoeGvSlotDev slot = slots[s];
        const float gs = (r < I) ? slot.gs1 : slot.gs3;
        acc = moe_gemv_dot(
                  x_row,
                  pack_base +
                      static_cast<int64_t>(slot.dev_slot) * pack_stride +
                      static_cast<int64_t>(r) * (H >> 1),
                  scale_base +
                      static_cast<int64_t>(slot.dev_slot) * scale_stride +
                      static_cast<int64_t>(r) * (H >> 4),
                  (KP > 1 ? (H / KP) * (warp % KP) : 0), H >> 5, lane) *
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
    const QwenMoeGvSlotDev *__restrict__ slots,
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
        const QwenMoeGvSlotDev slot = slots[s];
        acc = moe_gemv_dot(
                  hbuf + static_cast<int64_t>(s) * I,
                  pack_base +
                      static_cast<int64_t>(slot.dev_slot) * pack_stride +
                      w13_bytes + static_cast<int64_t>(r) * (I >> 1),
                  scale_base +
                      static_cast<int64_t>(slot.dev_slot) * scale_stride +
                      s13_bytes + static_cast<int64_t>(r) * (I >> 4),
                  (KP > 1 ? (I / KP) * (warp % KP) : 0), I >> 5, lane) *
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
    const QwenMoeGvSlotDev *__restrict__ slots, int k_slots, int n_elems,
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
// the exact clone of qwen_swiglu_precise_quant_kernel (bit-identical).
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
        lsxcommon::nvfp4::FloatToE4m3P2(absmax * (1.0f / 6.0f));
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
    const QwenMoeRoutedSlot *__restrict__ routes,
    const QwenMoeExpertMetaDev *__restrict__ meta,
    int k_slots,
    QwenMoeGvSlotDev *__restrict__ out)
{
    const int k = threadIdx.x;
    if (k >= k_slots) return;   // launch width fixed at 32, K <= 16
    const QwenMoeRoutedSlot r = routes[k];
    const QwenMoeExpertMetaDev m = meta[r.eid];
    out[k].dev_slot = m.dev_slot;
    out[k].gs1 = m.gs1;
    out[k].gs2 = m.gs2;
    out[k].gs3 = m.gs3;
    out[k].coeff = r.coeff;
}

} // namespace

cudaError_t RunQwenMoeGemvGatherTable(
    cudaStream_t stream,
    const QwenMoeRoutedSlot *d_slots, int k_slots,
    const QwenMoeExpertMetaDev *d_meta,
    QwenMoeGvSlotDev *d_tbl_out)
{
    if (!d_slots || !d_meta || !d_tbl_out || k_slots <= 0 || k_slots > 16)
        return cudaErrorInvalidValue;
    moe_gemv_gather_table_kernel<<<1, 32, 0, stream>>>(d_slots, d_meta,
                                                       k_slots, d_tbl_out);
    return cudaGetLastError();
}

cudaError_t RunQwenMoeSwigluQuantBatch(cudaStream_t stream, const __half *ygu,
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

cudaError_t RunQwenMoeRoutedTcBatch(
    cudaStream_t stream, QwenMoeExpertStore &store,
    const QwenMoeExpertDeviceCache &cache, int layer,
    const QwenMoeRoutedSlot *d_routes, const uint8_t *xq, const uint8_t *xqs,
    int K, int H, int I, const __half *d_resid, float *d_acc,
    __half *ygu_batch, __half *h_batch, __half *yp_batch,
    uint8_t *qpack_batch, uint8_t *qscales_batch, float *splitk13,
    QwenMoeGvSlotDev *d_tbl)
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
    cudaError_t e = RunQwenMoeGemvGatherTable(
        stream, d_routes, K,
        store.d_gemv_meta + static_cast<int64_t>(layer) * E, d_tbl);
    if (e != cudaSuccess) return e;
    if (kProbe) cudaEventRecord(ev1, stream);
    const auto *tbl =
        reinterpret_cast<const qwen::tc::QwenTcBatchSlot *>(d_tbl);
    e = qwen::tc::RunQwenTcGemmBatch13(
        stream, xq, xqs, cache.d_pack, cache.d_scale, cache.pack_stride,
        cache.scale_stride, tbl, K, H, 2 * I, ygu_batch, splitk13);
    if (e != cudaSuccess) return e;
    if (kProbe) cudaEventRecord(ev2, stream);
    e = RunQwenMoeSwigluQuantBatch(stream, ygu_batch, K, I, h_batch,
                                   qpack_batch, qscales_batch);
    if (e != cudaSuccess) return e;
    if (kProbe) cudaEventRecord(ev3, stream);
    e = qwen::tc::RunQwenTcGemmBatchDown(
        stream, qpack_batch, qscales_batch, cache.d_pack, cache.d_scale,
        cache.pack_stride, cache.scale_stride, tbl, K, w13_bytes, s13_bytes,
        I, H, yp_batch, nullptr);
    if (e != cudaSuccess) return e;
    if (kProbe) cudaEventRecord(ev4, stream);
    // Ascending-slot fp32 reduce -- the legacy coeff_add order (k=0..K-1);
    // acc already seeded by qwen_moe_init_acc_kernel upstream.
    moe_gemv_reduce_batch_kernel<<<(H + 255) / 256, 256, 0, stream>>>(
        yp_batch, d_tbl, K, H, d_acc);
    if (kProbe) {
        cudaEventRecord(ev5, stream);
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

// Device-table variant consumed by the no-handshake decode path. Requires
// the table to already live in device memory (built by the host in the
// handshake flow or by moe_gemv_gather_table_kernel otherwise).
cudaError_t RunQwenMoeRoutedGemmVBatchCore(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const QwenMoeGvSlotDev *slots_dev, int k_slots,
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

cudaError_t RunQwenMoeRoutedGemmVBatch(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const QwenMoeGvSlotDev *slots_host_tbl, int k_slots,
    int64_t w13_bytes, int64_t s13_bytes,
    int H, int I, const __half *d_x,
    __half *ygu, __half *hbuf, __half *ypart, float *acc)
{
    // Handshake wrapper: builds the slot table on the host and uploads it
    // (tests / callers that already hold a host-side table).
    if (!slots_host_tbl) return cudaErrorInvalidValue;
    QwenMoeGvSlotDev *d_tbl = nullptr;
    cudaError_t e = cudaMalloc(&d_tbl,
                               sizeof(QwenMoeGvSlotDev) * k_slots);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(d_tbl, slots_host_tbl,
                        sizeof(QwenMoeGvSlotDev) * k_slots,
                        cudaMemcpyHostToDevice, stream);
    if (e == cudaSuccess)
        e = RunQwenMoeRoutedGemmVBatchCore(stream, d_pack_base, d_scale_base, pack_stride,
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
bool QwenMoeGemvNohandshake(const QwenMoeExpertStore &store)
{
    const size_t flats =
        static_cast<size_t>(store.num_layers) * store.num_experts;
    if (flats == 0 || store.h_gemv_meta.size() != flats ||
        store.gemv_ready)
        return store.gemv_ready && !store.h_gemv_meta.empty();
    for (const auto &m : store.h_gemv_meta)
        if (m.dev_slot < 0) return false;   // warm-up incomplete
    // First complete mirror: upload once.
    auto &s = const_cast<QwenMoeExpertStore &>(store);
    if (cudaMalloc(reinterpret_cast<void **>(&s.d_gemv_meta),
                   flats * sizeof(QwenMoeExpertMetaDev)) != cudaSuccess) {
        s.d_gemv_meta = nullptr;
        return false;
    }
    if (cudaMemcpy(s.d_gemv_meta, s.h_gemv_meta.data(),
                   flats * sizeof(QwenMoeExpertMetaDev),
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
static cudaError_t RunQwenMoeRoutedGemmVNoHandshake(
    cudaStream_t stream, QwenMoeExpertStore &store,
    QwenMoeExpertDeviceCache &cache, int layer,
    const QwenMoeRoutedSlot *d_slots, int K,
    const __half *d_x, int H, int I, float *d_acc,
    __half *ygu_b, __half *hbuf_b, __half *ypart_b,
    QwenMoeGvSlotDev *d_tbl,
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
    cudaError_t e = RunQwenMoeGemvGatherTable(stream, d_slots, K,
                                              store.d_gemv_meta + flat_base,
                                              d_tbl);
    if (e == cudaSuccess)
        e = RunQwenMoeRoutedGemmVBatchCore(stream, cache.d_pack, cache.d_scale,
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
__global__ void qwen_moe_shared_gate_kernel(const __half *__restrict__ gate,
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
__global__ void qwen_moe_combine_kernel(const float *__restrict__ acc,
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

bool TruncateTokensAtEos(std::vector<int32_t> &tokens, int eos_id)
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

cudaError_t RunQwenMoeSharedCombine(
    cudaStream_t stream, const qwen::QwenNvfp4Weight &sw1,
    const qwen::QwenNvfp4Weight &sw2, const qwen::QwenNvfp4Weight &sw3,
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
                cudaError_t be = qwen::tc::BuildFusedGemmWeights(
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
                          qwen::tc::RunQwenTcGemmPreQuant(
                              stream, xq, xqs, fp, fs, 1.0f, d_scratch_i,
                              seq, H, fn, false, fw, nullptr));
        } else {
            LSX_MOE_STAGE("shared_fused_gemm",
                          qwen::RunQwenNvfp4Gemm(stream, d_x, fp, fs, 1.0f,
                                                 d_scratch_i, seq, H, fn,
                                                 false, fw, nullptr));
        }
        up_base = d_scratch_i + static_cast<int64_t>(seq) * SI;
    } else {
        if (qwen::RunQwenNvfp4Gemm(stream, d_x, sw1.d_packed, sw1.d_scales,
                                   sw1.global_scale, d_scratch_i, seq, H, SI,
                                   false) != cudaSuccess)
            return cudaErrorInvalidValue;
        if (qwen::RunQwenNvfp4Gemm(stream, d_x, sw3.d_packed, sw3.d_scales,
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
    if (qwen::RunQwenNvfp4Gemm(stream, d_scratch_i, sw2.d_packed, sw2.d_scales,
                               sw2.global_scale, d_scratch_h, seq, SI, H,
                               false) != cudaSuccess)
        return cudaErrorInvalidValue;
    // Scalar gate per token.
    qwen_moe_shared_gate_kernel<<<seq, kEltBlock, kEltBlock * sizeof(float),
                                  stream>>>(d_shared_gate, d_x, H, d_gate_val);
    cudaError_t ge = cudaGetLastError();
    if (std::getenv("LSX_TEST_MOE_STAGE_DEBUG") && ge != cudaSuccess)
        std::fprintf(stderr, "[stage] shared_gate: %s\n",
                     cudaGetErrorString(ge));
    // Combine routed accumulation + gated shared expert.
    qwen_moe_combine_kernel<<<EltGrid(seq * H), kEltBlock, 0, stream>>>(
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
__global__ void qwen_moe_init_acc_kernel(const __half *__restrict__ resid,
                                         float *__restrict__ acc, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] = __half2float(resid[i]);
}

// Device token feed (kimi round-25 pattern): the greedy argmax computed on
// device is written straight into the next step's id slot -- no D2H, no
// per-step fence.
__global__ void qwen_feed_token_kernel(int32_t *__restrict__ dst,
                                       const int32_t *__restrict__ src)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) dst[0] = src[0];
}

// Test/parity helper defined after the anonymous namespace (public entry).

// Append the step's argmax to the generation history for the bulk drain.
__global__ void qwen_record_token_kernel(int32_t *__restrict__ hist,
                                         int idx,
                                         const int32_t *__restrict__ src)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) hist[idx] = src[0];
}

// Residual add: unified kernels::AddInPlaceKernel (lsx_kernels_math.cuh).
} // namespace

cudaError_t RunQwenMoeAccInitForTest(cudaStream_t stream,
                                     const __half *d_resid, float *d_acc,
                                     int n)
{
    if (!d_resid || !d_acc || n <= 0) return cudaErrorInvalidValue;
    qwen_moe_init_acc_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        d_resid, d_acc, n);
    return cudaGetLastError();
}

cudaError_t EnsureQwenMoeScratch(cudaStream_t stream,
                                 const QwenMoePinnedWeights &pw,
                                 QwenMoeGenScratch &scratch)
{
    const int64_t elems = static_cast<int64_t>(pw.top.vocab_size) *
                          pw.top.hidden_size;
    if (scratch.d_embed && scratch.table_elems == elems)
        return cudaSuccess;
    FreeQwenMoeGenScratch(scratch);
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&scratch.d_embed),
                               static_cast<size_t>(elems) * sizeof(__half));
    if (e != cudaSuccess) return e;
    e = lsxcommon::gpu::dequant_shared_nvfp4(
        stream, pw.embed.d_packed, pw.embed.packed_bytes, pw.embed.d_scales,
        pw.embed.scale_bytes, pw.embed.global_scale, scratch.d_embed, elems);
    if (e != cudaSuccess) return e;
    scratch.table_elems = elems;
    ++scratch.dequant_runs;
    return cudaSuccess;
}

cudaError_t ReserveQwenMoeArena(QwenMoeGenScratch &scratch, size_t bytes)
{
    if (scratch.arena && scratch.arena_capacity >= bytes)
        return cudaSuccess;
    if (scratch.arena) {
        cudaFree(scratch.arena);
        scratch.arena = nullptr;
    }
    cudaError_t e = cudaMalloc(&scratch.arena, bytes);
    if (e != cudaSuccess) {
        scratch.arena_capacity = 0;
        return e;
    }
    scratch.arena_capacity = bytes;
    ++scratch.arena_allocs;
    return cudaSuccess;
}

void FreeQwenMoeGenScratch(QwenMoeGenScratch &scratch)
{
    if (scratch.d_embed) cudaFree(scratch.d_embed);
    if (scratch.arena) cudaFree(scratch.arena);
    scratch = {};
}

cudaError_t RunQwenMoeGeneration(
    cudaStream_t stream, const QwenMoePinnedWeights &pw,
    QwenMoeExpertStore &store, QwenMoeExpertDeviceCache &cache,
    QwenMoeGenScratch &scratch,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    qwen::QwenGenerationResult &result, lsxcommon::sample::ISampler &sampler,
    qwen::QwenKvCache *kv)
{
    if (!kv)
        return cudaErrorInvalidValue;  // MoE path is incremental-only.
    {   // Persistent embed table: dequant once, reuse across calls.
        cudaError_t pre = EnsureQwenMoeScratch(stream, pw, scratch);
        if (pre != cudaSuccess) {
            result.error_msg = "scratch ensure failed";
            return pre;
        }
    }
    const qwen::QwenTopology &top = pw.top;
    const int H = top.hidden_size;
    const int NH = top.num_attention_heads, HD = top.head_dim;
    const int NKV = top.num_key_value_heads;
    const int nk = top.linear_num_key_heads, kd = top.linear_key_head_dim;
    const int nv = top.linear_num_value_heads, vd = top.linear_value_head_dim;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;
    const int V = top.vocab_size;
    const int E = pw.moe_top.num_experts;
    const int K = pw.moe_top.num_experts_per_tok;
    const int I = pw.moe_top.moe_intermediate_size;
    const int SI = pw.moe_top.shared_expert_intermediate_size;
    const int rotary_dim = static_cast<int>(HD * top.partial_rotary_factor);
    result = qwen::QwenGenerationResult{};

    if (H <= 0 || V <= 0 || prompt.empty() || max_new <= 0 || eos_id < 0 ||
        E <= 0 || K <= 0 || K > kMaxTopK || I <= 0 || SI <= 0 ||
        !pw.embed.d_packed || !pw.head.d_packed) {
        result.error_msg = "invalid MoE generation inputs";
        return cudaErrorInvalidValue;
    }
    const int max_seq = static_cast<int>(prompt.size()) + max_new;

    // KV cache via a shallow dense view (dims/pointers only).
    qwen::QwenPinnedWeights kv_view;
    kv_view.top = top;
    kv_view.layers.resize(pw.layers.size());
    for (size_t l = 0; l < pw.layers.size(); ++l)
        kv_view.layers[l] = pw.layers[static_cast<size_t>(l)].attn;
    {
        cudaError_t ce = kv->allocated()
                             ? kv->EnsureCapacity(kv_view, kv->kv_rows + max_new,
                                                  stream)
                             : kv->Allocate(kv_view, max_seq, stream);
        if (ce != cudaSuccess) {
            result.error_msg = "kv cache alloc failed";
            return ce;
        }
    }

    // ── arena (single cudaMalloc; mirrors RunQwenGeneration minus MLP gate) ──
    size_t half_sz[] = {
        (size_t)max_seq * H,             // d_act
        (size_t)max_seq * H,             // d_normed
        (size_t)max_seq * H,             // d_attn_out
        (size_t)max_seq * H,             // d_postin
        (size_t)max_seq * NH * HD * 2,   // d_qc
        (size_t)max_seq * NH * HD,       // d_q
        (size_t)max_seq * NH * HD,       // d_qgate
        (size_t)max_seq * NKV * HD,      // d_k
        (size_t)max_seq * NKV * HD,      // d_v
        (size_t)max_seq * NH * HD,       // d_attn
        (size_t)max_seq * rotary_dim,    // d_cos
        (size_t)max_seq * rotary_dim,    // d_sin
        (size_t)max_seq * S,             // d_qkv
        (size_t)max_seq * VD,            // d_z
        (size_t)max_seq * nv * 2,        // d_ab
        (size_t)max_seq * S,             // d_conv
        (size_t)max_seq * VD,            // d_core
        (size_t)max_seq * VD,            // d_gated
        (size_t)max_seq * V,             // d_logits
        (size_t)max_seq * 2,             // d_ids
        (size_t)max_seq * 2,             // d_pos
        (size_t)I,                       // d_yg (routed scratch)
        (size_t)I,                       // d_yu
        (size_t)H,                       // d_y2
        (size_t)SI,                      // d_si (shared scratch)
        (size_t)H,                       // d_sh
        // Batched-GEMV decode scratch (no-handshake / probe paths):
        (size_t)kMaxTopK * 2 * I,        // d_ygu_batch [Ktop,2I]
        (size_t)kMaxTopK * I,            // d_h_batch   [Ktop,I]
        (size_t)kMaxTopK * H,            // d_yp_batch  [Ktop,H]
    };
    size_t float_sz[] = {
        (size_t)max_seq * nv,   // d_gf
        (size_t)max_seq * nv,   // d_betaf
        (size_t)1,              // d_maxv_scratch
        (size_t)max_seq * H,    // d_moe_acc
        (size_t)1,              // d_gate_val
    };
    size_t half_bytes = 0;
    for (auto s : half_sz) half_bytes += ((s + 7) / 8) * 8 * sizeof(__half);
    half_bytes = (half_bytes + 15) & ~static_cast<size_t>(15);
    size_t float_bytes = 0;
    for (auto s : float_sz) float_bytes += s * sizeof(float);
    float_bytes = (float_bytes + 15) & ~static_cast<size_t>(15);
    // Tail slice: routed-expert down-GEMM quant scratch (I/2 packed bytes +
    // ceil(I/16) scale bytes), carved once instead of per expert slot.
    const size_t qpack_bytes = (static_cast<size_t>(I) / 2 + 15) & ~size_t(15);
    const size_t qscales_bytes = (static_cast<size_t>(I + 15) / 16 + 15) & ~size_t(15);
    // Persistent activation-quant buffers (round-20 launch-overhead pass):
    // replace the per-layer cudaMallocAsync/FreeAsync pairs with arena
    // slices -- seq==1 decode re-uses the same bytes every layer/step on
    // one stream, so no extra synchronization is needed.
    const size_t xact_q_bytes =
        ((static_cast<size_t>(H) / 2 + 63) / 64) * 64;
    const size_t xact_qs_bytes =
        (((static_cast<size_t>(H) + 15) / 16) + 63) / 64 * 64;
    const size_t gemv_tbl_bytes = ((sizeof(QwenMoeGvSlotDev) *
                                        static_cast<size_t>(kMaxTopK)) +
                                   63) /
                                  64 * 64;
    // Batched TC routed chain (80-tok/s decode round): per-slot quant
    // outputs for the down GEMM + w13 split-K partials (arena slices).
    const int tc_splitk = qwen::tc::QwenTcBatchSplitK(H, 2 * I);
    const size_t qpack_b_bytes =
        ((static_cast<size_t>(kMaxTopK) * I / 2) + 63) / 64 * 64;
    const size_t qscales_b_bytes =
        ((static_cast<size_t>(kMaxTopK) * ((I + 15) / 16)) + 63) / 64 * 64;
    const size_t tc_splitk_bytes =
        tc_splitk > 1
            ? ((static_cast<size_t>(tc_splitk) * kMaxTopK * 2 * I * 4 + 63) /
               64) * 64
            : 64;

    cudaError_t e = ReserveQwenMoeArena(
        scratch, half_bytes + float_bytes + qpack_bytes + qscales_bytes +
                     xact_q_bytes + xact_qs_bytes + gemv_tbl_bytes +
                     qpack_b_bytes + qscales_b_bytes + tc_splitk_bytes);
    if (e != cudaSuccess) { result.error_msg = "arena reserve failed"; return e; }
    void *base = scratch.arena;

    auto adv = [&](__half *&p, size_t hs) {
        __half *r = p;
        p += ((hs + 7) / 8) * 8;
        return r;
    };
    __half *ph = reinterpret_cast<__half *>(base);
    __half *d_act = adv(ph, half_sz[0]);
    __half *d_normed = adv(ph, half_sz[1]);
    __half *d_attn_out = adv(ph, half_sz[2]);
    __half *d_postin = adv(ph, half_sz[3]);
    __half *d_qc = adv(ph, half_sz[4]);
    __half *d_q = adv(ph, half_sz[5]);
    __half *d_qgate = adv(ph, half_sz[6]);
    __half *d_k = adv(ph, half_sz[7]);
    __half *d_v = adv(ph, half_sz[8]);
    __half *d_attn = adv(ph, half_sz[9]);
    __half *d_cos = adv(ph, half_sz[10]);
    __half *d_sin = adv(ph, half_sz[11]);
    __half *d_qkv = adv(ph, half_sz[12]);
    __half *d_z = adv(ph, half_sz[13]);
    __half *d_ab = adv(ph, half_sz[14]);
    __half *d_conv = adv(ph, half_sz[15]);
    __half *d_core = adv(ph, half_sz[16]);
    __half *d_gated = adv(ph, half_sz[17]);
    __half *d_embed_h = scratch.d_embed;
    __half *d_logits = adv(ph, half_sz[18]);
    int32_t *d_ids = reinterpret_cast<int32_t *>(adv(ph, half_sz[19]));
    int32_t *d_pos = reinterpret_cast<int32_t *>(adv(ph, half_sz[20]));
    __half *d_yg = adv(ph, half_sz[21]);
    __half *d_yu = adv(ph, half_sz[22]);
    __half *d_y2 = adv(ph, half_sz[23]);
    __half *d_si = adv(ph, half_sz[24]);
    __half *d_sh = adv(ph, half_sz[25]);
    __half *d_ygu_batch = adv(ph, half_sz[26]);
    __half *d_h_batch = adv(ph, half_sz[27]);
    __half *d_yp_batch = adv(ph, half_sz[28]);
    float *pf = reinterpret_cast<float *>((uint8_t *)base + half_bytes);
    float *d_gf = pf; pf += float_sz[0];
    float *d_betaf = pf; pf += float_sz[1];
    float *d_maxv_scratch = pf; pf += float_sz[2];
    float *d_moe_acc = pf; pf += float_sz[3];
    float *d_gate_val = pf;
    uint8_t *d_qpack_scratch =
        reinterpret_cast<uint8_t *>(base) + half_bytes + float_bytes;
    uint8_t *d_qscales_scratch = d_qpack_scratch + qpack_bytes;
    uint8_t *d_xact_q = d_qscales_scratch + qscales_bytes;
    uint8_t *d_xact_qs = d_xact_q + xact_q_bytes;
    QwenMoeGvSlotDev *d_gemv_tbl =
        reinterpret_cast<QwenMoeGvSlotDev *>(d_xact_qs + xact_qs_bytes);
    uint8_t *d_qpack_b = d_xact_qs + xact_qs_bytes + gemv_tbl_bytes;
    uint8_t *d_qscales_b = d_qpack_b + qpack_b_bytes;
    float *d_tc_splitk =
        reinterpret_cast<float *>(d_qscales_b + qscales_b_bytes);

    int32_t *d_argmax = nullptr;
    e = cudaMalloc(&d_argmax, sizeof(int32_t));
    if (e != cudaSuccess) { result.error_msg = "argmax alloc"; return e; }
    // Device-side token history (bulk drain after the decode loop).
    int32_t *d_tokens = nullptr;
    e = cudaMalloc(&d_tokens, sizeof(int32_t) * static_cast<size_t>(max_new));
    if (e != cudaSuccess) {
        result.error_msg = "token history alloc";
        cudaFree(d_argmax);
        return e;
    }
    QwenMoeRoutedSlot *d_slots = nullptr;
    e = cudaMalloc(&d_slots, sizeof(QwenMoeRoutedSlot) * static_cast<size_t>(K));
    if (e != cudaSuccess) {
        result.error_msg = "slots alloc";
        cudaFree(d_argmax);
        cudaFree(d_tokens);
        return e;
    }

    // Pinned staging: per-step {token,pos} H2D and per-layer routed slots D2H.
    int32_t *h_step = nullptr;
    QwenMoeRoutedSlot *h_slots = nullptr;
    bool have_pinned =
        cudaMallocHost(&h_step, static_cast<size_t>(2 * (prompt.size() + max_new)) *
                                    sizeof(int32_t)) == cudaSuccess &&
        cudaMallocHost(&h_slots, sizeof(QwenMoeRoutedSlot) *
                                     static_cast<size_t>(K)) == cudaSuccess;

    cudaEvent_t t_start = nullptr, t_ttft = nullptr, t_stop = nullptr;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_ttft);
    cudaEventCreate(&t_stop);
    cudaEvent_t tok_ready = nullptr;
    cudaEventCreate(&tok_ready);
    cudaEventRecord(t_start, stream);

    // Copy channel for cache-miss expert uploads (task-2 audit item): H2D
    // staging overlaps the routed-expert compute chain instead of serializing
    // with it. Events fence each upload; ev_front guards slot reuse.
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t ev_uploads[16] = {};
    cudaEvent_t ev_front = nullptr;
    bool have_copy =
        cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking) ==
            cudaSuccess &&
        cudaEventCreateWithFlags(&ev_front, cudaEventDisableTiming) ==
            cudaSuccess;
    for (int k = 0; have_copy && k < K && k < 16; ++k)
        have_copy = cudaEventCreateWithFlags(&ev_uploads[k],
                                             cudaEventDisableTiming) ==
                    cudaSuccess;

    std::string moe_error;
    // Round-20c/20e: per-layer FFN sub-graphs (no-handshake mode only).
    // Capture covers gather -> batched GEMV core -> shared combine; all
    // pointers are session-stable arena slices and the only dynamic data
    // flows through device memory (d_slots), so replay is bit-transparent
    // (pinned by test_qwen_nvfp4_gemv "[graph]"). Opt-in LSX_GRAPH_FFN=1;
    // first engaged pass captures and replays once -- capture never runs
    // kernels, so failed captures simply fall back to the plain chain.
    const bool kGraphFfn = std::getenv("LSX_GRAPH_FFN") != nullptr;
    // Batched TC routed chain (device-driven decode): engages once the
    // full-residency mirror is complete + TC fast_math + fused slot layout.
    // Geometry gates mirror the legacy tc_fused conditions (I % 16 == 0);
    // the GEMV experimental route keeps priority when explicitly requested.
    static const bool kGemvExpertsEnv =
        std::getenv("LSX_GEMV_EXPERTS") != nullptr;
    const bool kTcBatchGeom = qwen::QwenTcFastMath() &&
                              store.d_seg_wgs != nullptr && (I & 15) == 0 &&
                              !kGemvExpertsEnv;
    std::vector<cudaGraph_t> ffn_graphs(pw.layers.size(), nullptr);
    std::vector<cudaGraphExec_t> ffn_execs(pw.layers.size(), nullptr);
    auto ffn_graph_teardown = [&] {
        for (size_t gi = 0; gi < pw.layers.size(); ++gi) {
            if (ffn_execs[gi]) cudaGraphExecDestroy(ffn_execs[gi]);
            if (ffn_graphs[gi]) cudaGraphDestroy(ffn_graphs[gi]);
            ffn_execs[gi] = nullptr;
            ffn_graphs[gi] = nullptr;
        }
    };
    auto fail = [&](cudaError_t err, std::string msg) -> cudaError_t {
        result.error_msg = std::move(msg);
        ffn_graph_teardown();
        if (t_start) cudaEventDestroy(t_start);
        if (t_ttft) cudaEventDestroy(t_ttft);
        if (t_stop) cudaEventDestroy(t_stop);
        if (tok_ready) cudaEventDestroy(tok_ready);
        if (copy_stream) cudaStreamDestroy(copy_stream);
        for (int k = 0; k < 16; ++k)
            if (ev_uploads[k]) cudaEventDestroy(ev_uploads[k]);
        if (ev_front) cudaEventDestroy(ev_front);
        if (h_step) cudaFreeHost(h_step);
        if (h_slots) cudaFreeHost(h_slots);
        cudaFree(d_argmax);
        cudaFree(d_tokens);
        cudaFree(d_slots);
        return err;
    };

    const double eps = top.rms_norm_eps;
    const int threads = 256;

    // ── incremental path: one token per step (SelfCheck pattern) ──
    const int prompt_steps = static_cast<int>(prompt.size());
    const int total_steps = prompt_steps + max_new - 1;
    int32_t cur = prompt[0];
    int32_t tok = -1;
    int generated = 0;

    for (int step = 0; step < total_steps; ++step) {
        const int seq = 1;
        const int kv_base = kv->kv_rows;
        // Per-step H2D is position-only (async pinned write, no fence).
        // The token id arrives either from host (step 0 + prefill) or from
        // the previous step's device-side argmax (qwen_feed_token_kernel).
        if (h_step) {
            h_step[2 * step] = cur;
            h_step[2 * step + 1] = kv_base;
            cudaMemcpyAsync(d_pos, h_step + 2 * step + 1, sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            if (step == 0)
                cudaMemcpyAsync(d_ids, h_step + 2 * step, sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream);
        } else {
            cudaMemcpy(d_pos, &kv_base, sizeof(int32_t), cudaMemcpyHostToDevice);
            if (step == 0)
                cudaMemcpy(d_ids, &cur, sizeof(int32_t), cudaMemcpyHostToDevice);
        }

        e = qwen::RunQwenEmbed(stream, d_embed_h, d_ids, d_act, seq, H);
        if (e != cudaSuccess) return fail(e, "embed");

        int full_idx = 0, lin_idx = 0;
        for (int l = 0; l < static_cast<int>(pw.layers.size()); ++l) {
            const QwenMoeLayerWeights &lw = pw.layers[static_cast<size_t>(l)];
            e = qwen::RunQwenRMSNorm(stream, d_act, lw.attn.d_input_norm,
                                     d_normed, seq, H, eps);
            if (e != cudaSuccess) return fail(e, "inputnorm l" + std::to_string(l));
            if (lw.attn.is_linear) {
                e = qwen::RunQwenFusedLinearAttnKernel(stream, lw.attn, top, seq,
                    d_normed, d_attn_out,
                    kv->d_qkv + (int64_t)lin_idx * kv->max_total * S,
                    d_z, d_ab, d_gf, d_betaf, d_conv, d_core, d_gated,
                    kv_base,
                    kv->d_col + (int64_t)lin_idx * nv * vd * kd);
                ++lin_idx;
            } else {
                e = qwen::RunQwenFullAttention(stream, lw.attn, top, seq,
                    d_normed, d_attn_out, d_qc, d_q, d_qgate,
                    kv->d_kv_k + (int64_t)full_idx * kv->max_total * NKV * HD,
                    kv->d_kv_v + (int64_t)full_idx * kv->max_total * NKV * HD,
                    d_attn, d_cos, d_sin, d_pos, kv_base);
                ++full_idx;
            }
            if (e != cudaSuccess) return fail(e, "attn l" + std::to_string(l));
            {
                const int64_t n = seq * H;
                lsxcommon::kernels::AddInPlaceKernel<__half><<<(int)((n + threads - 1) / threads), threads,
                                       0, stream>>>(d_attn_out, d_act, n);
                e = cudaGetLastError();
                if (e) return fail(e, "res1 l" + std::to_string(l));
            }
            e = qwen::RunQwenRMSNorm(stream, d_act, lw.attn.d_post_norm,
                                     d_postin, seq, H, eps);
            if (e != cudaSuccess) return fail(e, "postnorm l" + std::to_string(l));

            // ── MoE FFN: router → routed experts → shared combine ──
            e = RunQwenMoeRouterTopK(stream, reinterpret_cast<const __nv_bfloat16 *>(lw.d_router),
                                     d_postin, seq, E, K, H, d_slots);
            if (e != cudaSuccess) return fail(e, "router l" + std::to_string(l));
            {
                // Accumulator init is independent of the routing result; enqueue
                // it behind the router so the GPU has work queued while the
                // host waits for the slot readback below. The TC-batch path
                // folds the seed into its reduce kernel (same fp32 order) and
                // skips this launch.
                const int64_t n = seq * H;
                qwen_moe_init_acc_kernel<<<(int)((n + threads - 1) / threads),
                                           threads, 0, stream>>>(d_act, d_moe_acc,
                                                                 n);
                e = cudaGetLastError();
                if (e != cudaSuccess) return fail(e, "acc-init l" + std::to_string(l));
            }
            // No-handshake mode (full residency steady state): the routed
            // chain composes its table on device from d_slots; the host
            // never reads the router output back, so no fence/poll here.
            // TC-batch steady state shares the same readiness mirror; the
            // GEMV experimental route keeps priority when explicitly
            // requested via LSX_GEMV_EXPERTS.
            const bool moe_nohs_ready =
                lsxcommon::qwen_moe::QwenMoeGemvNohandshake(store);
            const bool tc_batch = kTcBatchGeom && moe_nohs_ready;
            const bool nohandshake = kGemvExpertsEnv && moe_nohs_ready;
            if (!tc_batch && !nohandshake) {
                // Host needs the selected ids to stage experts: small D2H +
                // event fence once per layer. Busy-poll the event instead of
                // the blocking cudaEventSynchronize — with K layers × steps
                // handshakes per generation, host wake latency is pure GPU
                // idle time.
                cudaMemcpyAsync(h_slots, d_slots,
                                sizeof(QwenMoeRoutedSlot) * K,
                                cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(tok_ready, stream);
                {
                    cudaError_t q;
                    while ((q = cudaEventQuery(tok_ready)) ==
                           cudaErrorNotReady) {}
                    (void)q;
                }
            }
            QwenMoeRoutedSlot *slots_host = h_slots;
            // Quantize the layer input ONCE for all experts + shared GEMMs
            // (TC path): removes ~8 redundant ActQuant passes per layer.
            // Buffers are persistent arena slices (round-20 pass) -- the
            // old cudaMallocAsync/FreeAsync pairs cost two driver calls
            // per layer on WDDM for bytes that never outlive this layer.
            uint8_t *xq = nullptr, *xqs = nullptr;
            if (qwen::QwenTcFastMath()) {
                xq = d_xact_q;
                xqs = d_xact_qs;
                if (lsxcommon::qwen::tc::QuantizeActsNvfp4(
                        stream, d_postin, seq, H, xq, xqs) != cudaSuccess)
                    return fail(cudaGetLastError(), "actquant l" +
                                                        std::to_string(l));
            }
            bool skip_shared = false;
            if (tc_batch) {
                // ── Batched TC routed chain (device table, no host
                // readback): gather -> w13 batch TC GEMM -> batched
                // swiglu-quant -> w2 batch TC GEMM -> ascending reduce.
                // Bit-identical to the legacy per-slot chain (pinned by
                // test_qwen_moe_kernels "[qwen-moe-tc-batch]").
                e = RunQwenMoeRoutedTcBatch(
                    stream, store, cache, l, d_slots, xq, xqs, K, H, I,
                    d_act, d_moe_acc, d_ygu_batch, d_h_batch, d_yp_batch,
                    d_qpack_b, d_qscales_b,
                    tc_splitk > 1 ? d_tc_splitk : nullptr, d_gemv_tbl);
                if (e != cudaSuccess)
                    return fail(e, "tc-batch experts l" + std::to_string(l));
            } else if (nohandshake) {
                // ── FFN section: per-layer graph replay when enabled ──
                bool ffn_done = false;
                if (kGraphFfn) {
                    if (!ffn_execs[l]) {
                        const size_t flat_base =
                            static_cast<size_t>(l) *
                            static_cast<size_t>(store.num_experts);
                        const auto &rec0 =
                            store.records[static_cast<size_t>(l)][0];
                        cudaError_t cap = cudaStreamBeginCapture(
                            stream, cudaStreamCaptureModeThreadLocal);
                        if (cap == cudaSuccess) {
                            cudaError_t s1 = RunQwenMoeGemvGatherTable(
                                stream, d_slots, K,
                                store.d_gemv_meta + flat_base,
                                d_gemv_tbl);
                            cudaError_t s2 =
                                s1 == cudaSuccess
                                    ? RunQwenMoeRoutedGemmVBatchCore(
                                          stream, cache.d_pack, cache.d_scale,
                                          cache.pack_stride,
                                          cache.scale_stride, d_gemv_tbl, K,
                                          static_cast<int64_t>(rec0.w1_bytes +
                                                               rec0.w3_bytes),
                                          static_cast<int64_t>(rec0.s1_bytes +
                                                               rec0.s3_bytes),
                                          H, I, d_postin, d_ygu_batch,
                                          d_h_batch, d_yp_batch, d_moe_acc)
                                    : s1;
                            RunQwenMoeSharedCombine(stream, lw.shared_w1,
                                                    lw.shared_w2, lw.shared_w3,
                                                    lw.d_shared_gate,
                                                    d_postin, seq, H, SI,
                                                    d_moe_acc, d_si, d_sh,
                                                    d_gate_val, d_act, xq,
                                                    xqs);
                            cudaGraph_t g = nullptr;
                            if (cudaStreamEndCapture(stream, &g) ==
                                    cudaSuccess &&
                                s2 == cudaSuccess &&
                                cudaGraphInstantiate(&ffn_execs[l], g,
                                                     nullptr, nullptr, 0) ==
                                    cudaSuccess) {
                                ffn_graphs[l] = g;
                            } else {
                                // Capture/instantiate failed -> plain chain
                                // (capture itself executed nothing).
                                if (g) cudaGraphDestroy(g);
                                ffn_execs[l] = nullptr;
                            }
                        }
                    }
                    if (ffn_execs[l] &&
                        cudaGraphLaunch(ffn_execs[l], stream) ==
                            cudaSuccess) {
                        ffn_done = true;
                        skip_shared = true;   // combine lives inside graph
                    } else if (ffn_execs[l]) {
                        cudaGraphExecDestroy(ffn_execs[l]);
                        cudaGraphDestroy(ffn_graphs[l]);
                        ffn_execs[l] = nullptr;
                        ffn_graphs[l] = nullptr;
                        e = cudaGetLastError();
                    }
                }
                if (!ffn_done && e == cudaSuccess) {
                    e = RunQwenMoeRoutedGemmVNoHandshake(
                        stream, store, cache, l, d_slots, K, d_postin, H, I,
                        d_moe_acc, d_ygu_batch, d_h_batch, d_yp_batch,
                        d_gemv_tbl, moe_error);
                }
            } else {
                e = RunQwenMoeRoutedFFN(stream, store, cache, l, slots_host,
                                        K, d_postin, H, I, d_moe_acc, d_yg,
                                        d_yu, d_y2, moe_error,
                                        have_copy ? copy_stream : nullptr,
                                        have_copy ? ev_uploads : nullptr,
                                        ev_front, xq, xqs,
                                        d_qpack_scratch,
                                        d_qscales_scratch, d_ygu_batch,
                                        d_h_batch, d_yp_batch, d_gemv_tbl);
            }
            if (e != cudaSuccess)
                return fail(e, "experts l" + std::to_string(l) + ": " + moe_error);
            if (skip_shared) { /* shared combine executed inside graph */ }
            else {
            e = RunQwenMoeSharedCombine(stream, lw.shared_w1, lw.shared_w2,
                                        lw.shared_w3, lw.d_shared_gate,
                                        d_postin, seq, H, SI, d_moe_acc,
                                        d_si, d_sh, d_gate_val, d_act, xq,
                                        xqs);
            if (e != cudaSuccess) return fail(e, "shared l" + std::to_string(l));
            }
        }

        e = qwen::RunQwenRMSNorm(stream, d_act, pw.d_final_norm, d_postin,
                                 seq, H, eps);
        if (e != cudaSuccess) return fail(e, "finalnorm");
        e = qwen::RunQwenNvfp4Gemm(stream, d_postin, pw.head.d_packed,
                                   pw.head.d_scales, pw.head.global_scale,
                                   d_logits, seq, H, V, false);
        if (e != cudaSuccess) return fail(e, "head");

        kv->kv_rows += seq;

        if (step >= prompt_steps - 1) {
            e = sampler.Sample(stream, d_logits, V,
                               lsxcommon::sample::LogitType::kF16,
                               d_argmax, d_maxv_scratch);
            if (e != cudaSuccess) return fail(e, "argmax");
            // Device-side: append to the history + feed the NEXT step's id.
            // No D2H, no fence (kimi round-25 pattern; determinism pinned by
            // test_qwen_moe_model "[qwen-moe-decode]").
            qwen_record_token_kernel<<<1, 1, 0, stream>>>(
                d_tokens, generated, d_argmax);
            if (step + 1 < total_steps)
                qwen_feed_token_kernel<<<1, 1, 0, stream>>>(d_ids, d_argmax);
            if (generated == 0) cudaEventRecord(t_ttft, stream);
            ++generated;
        } else {
            // Prefill: feed the next prompt token from host staging.
            cur = prompt[step + 1];
            if (h_step) {
                h_step[2 * (step + 1)] = cur;
                cudaMemcpyAsync(d_ids, h_step + 2 * (step + 1),
                                sizeof(int32_t), cudaMemcpyHostToDevice,
                                stream);
            } else {
                cudaMemcpy(d_ids, &cur, sizeof(int32_t),
                           cudaMemcpyHostToDevice);
            }
        }
    }

    // ── Bulk token drain: ONE D2H of the full generation history ──
    {
        std::vector<int32_t> hgen(static_cast<size_t>(generated));
        if (generated > 0) {
            cudaMemcpyAsync(hgen.data(), d_tokens,
                            sizeof(int32_t) * static_cast<size_t>(generated),
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(tok_ready, stream);
            cudaEventSynchronize(tok_ready);
        }
        result.tokens.assign(hgen.begin(), hgen.end());
        result.eos_terminated = TruncateTokensAtEos(result.tokens, eos_id);
        if (!result.tokens.empty()) tok = result.tokens.back();
    }
    kv->last_tok = tok;

    // ── Stage replay audit (LSX_AUDIT_STAGES=1, stderr diagnostics only) ────
    // Replays the final decode step N times with sequential timestamp events
    // along the compute stream: elapsed between consecutive stamps is the
    // GPU-timeline duration of each stage INCLUDING submission queue gaps.
    // All kernels target fixed kv rows / scratch, so rounds are idempotent
    // and published results are untouched.
    if (std::getenv("LSX_AUDIT_STAGES") && h_slots) {
        constexpr int kAuditRounds = 24;
        const size_t nlayers = pw.layers.size();
        const int nseg = static_cast<int>(nlayers) * 2 + 2;
        std::vector<cudaEvent_t> aev(static_cast<size_t>(nseg) + 1);
        for (auto &ev : aev) cudaEventCreate(&ev);
        std::vector<double> seg_acc(nseg, 0.0);
        int rounds_done = 0;

        // Routing snapshot for the audited step (previous sampler fence has
        // already synchronized this stream).
        cudaMemcpyAsync(h_slots, d_slots, sizeof(QwenMoeRoutedSlot) * K,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        for (int r = 0; r < kAuditRounds; ++r) {
            int si = 0, fullA = 0, linA = 0;
            auto stamp = [&]() { cudaEventRecord(aev[si++], stream); };
            auto mark = [&]() {
                if (si <= nseg) stamp();
            };
            mark();   // t0: start-of-round boundary
            qwen::RunQwenEmbed(stream, d_embed_h, d_ids, d_act, 1, H);
            mark();
            for (size_t l = 0; l < nlayers; ++l) {
                const QwenMoeLayerWeights &lw = pw.layers[l];
                qwen::RunQwenRMSNorm(stream, d_act, lw.attn.d_input_norm,
                                     d_normed, 1, H, eps);
                if (lw.attn.is_linear) {
                    qwen::RunQwenFusedLinearAttnKernel(
                        stream, lw.attn, top, 1, d_normed, d_attn_out,
                        kv->d_qkv +
                            (int64_t)linA * kv->max_total * S,
                        d_z, d_ab, d_gf, d_betaf, d_conv, d_core, d_gated,
                        kv->kv_rows,
                        kv->d_col + (int64_t)linA * nv * vd * kd);
                    ++linA;
                } else {
                    qwen::RunQwenFullAttention(
                        stream, lw.attn, top, 1, d_normed, d_attn_out, d_qc,
                        d_q, d_qgate,
                        kv->d_kv_k +
                            (int64_t)fullA * kv->max_total * NKV * HD,
                        kv->d_kv_v +
                            (int64_t)fullA * kv->max_total * NKV * HD,
                        d_attn, d_cos, d_sin, d_pos, kv->kv_rows);
                    ++fullA;
                }
                lsxcommon::kernels::AddInPlaceKernel<__half>
                    <<<1, threads, 0, stream>>>(d_attn_out, d_act, H);
                stamp();   // attn segment: norm + attention + residual add
                RunQwenMoeRoutedFFN(stream, store, cache,
                                    static_cast<int>(l), h_slots, K,
                                    d_postin, H, I, d_moe_acc, d_yg, d_yu,
                                    d_y2, moe_error);
                RunQwenMoeSharedCombine(stream, lw.shared_w1, lw.shared_w2,
                                        lw.shared_w3, lw.d_shared_gate,
                                        d_postin, 1, H, SI, d_moe_acc, d_si,
                                        d_sh, d_gate_val, d_act, nullptr,
                                        nullptr);
                stamp();   // ffn segment: router + routed chain + shared
            }
            qwen::RunQwenRMSNorm(stream, d_act, pw.d_final_norm, d_postin, 1,
                                 H, eps);
            qwen::RunQwenNvfp4Gemm(stream, d_postin, pw.head.d_packed,
                                   pw.head.d_scales, pw.head.global_scale,
                                   d_logits, 1, H, V, false);
            stamp();       // tail segment: final norm + head GEMM
            // Accumulate this round's pairs NOW (events are reused next
            // round; without a fence their timestamps would be reset before
            // reading, scrambling cross-round spans).
            if (cudaStreamSynchronize(stream) == cudaSuccess) {
                for (int i = 0; i < nseg; ++i) {
                    float ms = -1.0f;
                    if (cudaEventElapsedTime(&ms, aev[i], aev[i + 1]) ==
                        cudaSuccess)
                        seg_acc[i] += ms;
                }
                ++rounds_done;
            }
        }
        const cudaError_t aerr = cudaStreamSynchronize(stream);
        std::vector<double> seg(nseg);
        double total = 0.0;
        if (aerr == cudaSuccess) {
            const double kR = rounds_done > 0 ? rounds_done : 1;
            for (int i = 0; i < nseg; ++i) seg[i] = seg_acc[i] / kR;
            // Layout with leading t0 stamp: seg0=embed, seg(1+2l)=attn_l,
            // seg(2+2l)=ffn_l, seg(nseg-1)=final norm + head GEMM.
            double emb = seg[0], attnF = 0.0, attnL = 0.0, ffn = 0.0;
            for (size_t l = 0; l < nlayers; ++l) {
                if (pw.layers[l].attn.is_linear)
                    attnL += seg[1 + 2 * static_cast<int>(l)];
                else
                    attnF += seg[1 + 2 * static_cast<int>(l)];
                ffn += seg[2 + 2 * static_cast<int>(l)];
            }
            const double headm = seg[nseg - 1];
            total = emb + attnF + attnL + ffn + headm;
            if (total > 0.0)
                fprintf(stderr,
                        "[AUDIT] rounds=%d avg_step_ms=%.3f share%% "
                        "embed=%.1f attn_full=%.1f attn_linear=%.1f "
                        "ffn=%.1f head=%.1f\n",
                        rounds_done, total,
                        100.0 * emb / total, 100.0 * attnF / total,
                        100.0 * attnL / total, 100.0 * ffn / total,
                        100.0 * headm / total);
            // Top-5 heaviest segments (layer attribution).
            struct SegRef { double ms; int idx; };
            std::vector<SegRef> top;
            for (int i = 0; i < nseg; ++i) top.push_back({seg[i], i});
            std::sort(top.begin(), top.end(),
                      [](const SegRef &a, const SegRef &b) {
                          return a.ms > b.ms;
                      });
            for (int t = 0; t < 5 && t < nseg; ++t) {
                const int i = top[t].idx;
                const char *kind =
                    i == 0 ? "embed"
                            : (i == nseg - 1 ? "head"
                                             : ((i % 2) ? "attn" : "ffn"));
                fprintf(stderr, "[AUDIT] top%d %s L=%d %.3f ms (%.1f%%)\n",
                        t + 1, kind,
                        (i - 1) / 2, top[t].ms,
                        total > 0.0 ? 100.0 * top[t].ms / total : 0.0);
            }
        }
        for (auto ev : aev) cudaEventDestroy(ev);
    }

    {
        std::vector<float> hlast_f(V);
        std::vector<__half> hlast(V);
        cudaMemcpyAsync(hlast.data(), d_logits, V * sizeof(__half),
                        cudaMemcpyDeviceToHost, stream);
        cudaError_t le = cudaGetLastError();
        if (le != cudaSuccess) return fail(le, "logits copy");
        cudaEventRecord(t_stop, stream);
        if (cudaEventSynchronize(t_stop) != cudaSuccess)
            return fail(cudaGetLastError(), "logits fence");
        for (int i = 0; i < V; ++i) hlast_f[i] = __half2float(hlast[i]);
        result.final_logits = std::move(hlast_f);
    }

    float ms_total = 0.0f, ms_ttft = 0.0f;
    cudaEventElapsedTime(&ms_total, t_start, t_stop);
    cudaEventElapsedTime(&ms_ttft, t_start, t_ttft);
    result.total_ms = ms_total;
    result.ttft_ms = ms_ttft;
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_ttft);
    cudaEventDestroy(t_stop);
    ffn_graph_teardown();
    if (tok_ready) cudaEventDestroy(tok_ready);
    if (copy_stream) cudaStreamDestroy(copy_stream);
    for (int k = 0; k < 16; ++k)
        if (ev_uploads[k]) cudaEventDestroy(ev_uploads[k]);
    if (ev_front) cudaEventDestroy(ev_front);
    if (h_step) cudaFreeHost(h_step);
    if (h_slots) cudaFreeHost(h_slots);
    cudaFree(d_argmax);
    cudaFree(d_tokens);
    result.resident_bytes = pw.resident_bytes + half_bytes + float_bytes +
                            qpack_bytes + qscales_bytes + xact_q_bytes +
                            xact_qs_bytes + gemv_tbl_bytes +
                            sizeof(int32_t) + kv->Bytes();
    result.ok = true;
    return cudaSuccess;
}

} // namespace qwen_moe
} // namespace lsxcommon
