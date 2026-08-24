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
#include "lsxcommon/shared_experts.h"
#include "nvfp4_codec.h"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_act.cuh"

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
        ::lsxcommon::kernels::SwigluKernel<__half, ::lsxcommon::kernels::ExpFlavor::Precise, false>
            <<<EltGrid(I), kEltBlock, 0, stream>>>(d_yg, d_yu, d_yg, I, 0.f);
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
    lsxcommon::kernels::F32toTKernel<__half><<<EltGrid(n), kEltBlock, 0, stream>>>(d_acc, d_out, n);
    return cudaGetLastError();
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

cudaError_t RunQwenMoeSharedCombine(
    cudaStream_t stream, const qwen::QwenNvfp4Weight &sw1,
    const qwen::QwenNvfp4Weight &sw2, const qwen::QwenNvfp4Weight &sw3,
    const __half *d_shared_gate, const __half *d_x, int seq, int H, int SI,
    const float *d_routed_acc, __half *d_scratch_i, __half *d_scratch_h,
    float *d_gate_val, __half *d_out)
{
    // Shared expert MLP on all tokens at once.
    if (qwen::RunQwenNvfp4Gemm(stream, d_x, sw1.d_packed, sw1.d_scales,
                               sw1.global_scale, d_scratch_i, seq, H, SI,
                               false) != cudaSuccess)
        return cudaErrorInvalidValue;
    if (qwen::RunQwenNvfp4Gemm(stream, d_x, sw3.d_packed, sw3.d_scales,
                               sw3.global_scale, d_scratch_h, seq, H, SI,
                               false) != cudaSuccess)
        return cudaErrorInvalidValue;
    ::lsxcommon::kernels::SwigluKernel<__half, ::lsxcommon::kernels::ExpFlavor::Precise, false>
        <<<EltGrid(seq * SI), kEltBlock, 0, stream>>>(d_scratch_i, d_scratch_h,
                                                      d_scratch_i, seq * SI, 0.f);
    if (qwen::RunQwenNvfp4Gemm(stream, d_scratch_i, sw2.d_packed, sw2.d_scales,
                               sw2.global_scale, d_scratch_h, seq, SI, H,
                               false) != cudaSuccess)
        return cudaErrorInvalidValue;
    // Scalar gate per token.
    qwen_moe_shared_gate_kernel<<<seq, kEltBlock, kEltBlock * sizeof(float),
                                  stream>>>(d_shared_gate, d_x, H, d_gate_val);
    // Combine routed accumulation + gated shared expert.
    qwen_moe_combine_kernel<<<EltGrid(seq * H), kEltBlock, 0, stream>>>(
        d_routed_acc, d_gate_val, d_scratch_h, d_out, seq * H, H);
    return cudaGetLastError();
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

// Residual add: unified kernels::AddInPlaceKernel (lsx_kernels_math.cuh).
} // namespace

cudaError_t RunQwenMoeGeneration(
    cudaStream_t stream, const QwenMoePinnedWeights &pw,
    QwenMoeExpertStore &store, QwenMoeExpertDeviceCache &cache,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    qwen::QwenGenerationResult &result, lsxcommon::sample::ISampler &sampler,
    qwen::QwenKvCache *kv)
{
    if (!kv)
        return cudaErrorInvalidValue;  // MoE path is incremental-only.
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
        (size_t)V * H,                   // d_embed_half
        (size_t)V * H,                   // d_head_half
        (size_t)max_seq * V,             // d_logits
        (size_t)max_seq * 2,             // d_ids
        (size_t)max_seq * 2,             // d_pos
        (size_t)I,                       // d_yg (routed scratch)
        (size_t)I,                       // d_yu
        (size_t)H,                       // d_y2
        (size_t)SI,                      // d_si (shared scratch)
        (size_t)H,                       // d_sh
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

    void *base = nullptr;
    cudaError_t e = cudaMalloc(&base, half_bytes + float_bytes);
    if (e != cudaSuccess) { result.error_msg = "arena alloc failed"; return e; }

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
    __half *d_embed_h = adv(ph, half_sz[18]);
    __half *d_head_h = adv(ph, half_sz[19]);
    __half *d_logits = adv(ph, half_sz[20]);
    int32_t *d_ids = reinterpret_cast<int32_t *>(adv(ph, half_sz[21]));
    int32_t *d_pos = reinterpret_cast<int32_t *>(adv(ph, half_sz[22]));
    __half *d_yg = adv(ph, half_sz[23]);
    __half *d_yu = adv(ph, half_sz[24]);
    __half *d_y2 = adv(ph, half_sz[25]);
    __half *d_si = adv(ph, half_sz[26]);
    __half *d_sh = adv(ph, half_sz[27]);
    float *pf = reinterpret_cast<float *>((uint8_t *)base + half_bytes);
    float *d_gf = pf; pf += float_sz[0];
    float *d_betaf = pf; pf += float_sz[1];
    float *d_maxv_scratch = pf; pf += float_sz[2];
    float *d_moe_acc = pf; pf += float_sz[3];
    float *d_gate_val = pf;

    e = lsxcommon::gpu::dequant_shared_nvfp4(
        stream, pw.embed.d_packed, pw.embed.packed_bytes, pw.embed.d_scales,
        pw.embed.scale_bytes, pw.embed.global_scale, d_embed_h,
        (int64_t)V * H);
    if (e != cudaSuccess) { result.error_msg = "dequant embed"; cudaFree(base); return e; }
    e = lsxcommon::gpu::dequant_shared_nvfp4(
        stream, pw.head.d_packed, pw.head.packed_bytes, pw.head.d_scales,
        pw.head.scale_bytes, pw.head.global_scale, d_head_h,
        (int64_t)V * H);
    if (e != cudaSuccess) { result.error_msg = "dequant head"; cudaFree(base); return e; }

    int32_t *d_argmax = nullptr;
    e = cudaMalloc(&d_argmax, sizeof(int32_t));
    if (e != cudaSuccess) { result.error_msg = "argmax alloc"; cudaFree(base); return e; }
    QwenMoeRoutedSlot *d_slots = nullptr;
    e = cudaMalloc(&d_slots, sizeof(QwenMoeRoutedSlot) * static_cast<size_t>(K));
    if (e != cudaSuccess) {
        result.error_msg = "slots alloc";
        cudaFree(d_argmax);
        cudaFree(base);
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

    std::string moe_error;
    auto fail = [&](cudaError_t err, std::string msg) -> cudaError_t {
        result.error_msg = std::move(msg);
        if (t_start) cudaEventDestroy(t_start);
        if (t_ttft) cudaEventDestroy(t_ttft);
        if (t_stop) cudaEventDestroy(t_stop);
        if (tok_ready) cudaEventDestroy(tok_ready);
        if (h_step) cudaFreeHost(h_step);
        if (h_slots) cudaFreeHost(h_slots);
        cudaFree(d_argmax);
        cudaFree(d_slots);
        cudaFree(base);
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
        if (h_step) {
            int32_t *slot = h_step + 2 * step;
            slot[0] = cur;
            slot[1] = kv_base;
            cudaMemcpyAsync(d_ids, slot, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_pos, slot + 1, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        } else {
            cudaMemcpy(d_ids, &cur, sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_pos, &kv_base, sizeof(int32_t), cudaMemcpyHostToDevice);
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
            // Host needs the selected ids to stage experts: small D2H +
            // event handshake once per layer (etalon: GLM routing staging).
            cudaMemcpyAsync(h_slots, d_slots, sizeof(QwenMoeRoutedSlot) * K,
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(tok_ready, stream);
            cudaEventSynchronize(tok_ready);
            QwenMoeRoutedSlot *slots_host = h_slots;
            qwen_moe_init_acc_kernel<<<(int)((seq * H + threads - 1) / threads),
                                       threads, 0, stream>>>(d_act, d_moe_acc,
                                                             seq * H);
            e = RunQwenMoeRoutedFFN(stream, store, cache, l, slots_host, K,
                                    d_postin, H, I, d_moe_acc, d_yg, d_yu,
                                    d_y2, moe_error);
            if (e != cudaSuccess)
                return fail(e, "experts l" + std::to_string(l) + ": " + moe_error);
            e = RunQwenMoeSharedCombine(stream, lw.shared_w1, lw.shared_w2,
                                        lw.shared_w3, lw.d_shared_gate,
                                        d_postin, seq, H, SI, d_moe_acc,
                                        d_si, d_sh, d_gate_val, d_act);
            if (e != cudaSuccess) return fail(e, "shared l" + std::to_string(l));
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
            int32_t gtok = 0;
            cudaMemcpyAsync(&gtok, d_argmax, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(tok_ready, stream);
            cudaEventSynchronize(tok_ready);
            tok = gtok;
            if (generated == 0) cudaEventRecord(t_ttft, stream);
            result.tokens.push_back(tok);
            ++generated;
            if (tok == eos_id) {
                result.eos_terminated = true;
                break;
            }
            cur = tok;
        } else {
            cur = prompt[step + 1];
        }
    }
    kv->last_tok = tok;

    // Publish final-row logits for the regression harness (stream-fenced).
    {
        std::vector<float> hlast_f(V);
        std::vector<__half> hlast(V);
        cudaMemcpyAsync(hlast.data(), d_logits, V * sizeof(__half),
                        cudaMemcpyDeviceToHost, stream);
        for (int i = 0; i < V; ++i) hlast_f[i] = __half2float(hlast[i]);
        result.final_logits = std::move(hlast_f);
    }

    cudaEventRecord(t_stop, stream);
    cudaEventSynchronize(t_stop);
    float ms_total = 0.0f, ms_ttft = 0.0f;
    cudaEventElapsedTime(&ms_total, t_start, t_stop);
    cudaEventElapsedTime(&ms_ttft, t_start, t_ttft);
    result.total_ms = ms_total;
    result.ttft_ms = ms_ttft;
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_ttft);
    cudaEventDestroy(t_stop);
    if (tok_ready) cudaEventDestroy(tok_ready);
    if (h_step) cudaFreeHost(h_step);
    if (h_slots) cudaFreeHost(h_slots);
    cudaFree(d_argmax);
    cudaFree(base);
    result.resident_bytes = pw.resident_bytes + half_bytes + float_bytes +
                            sizeof(int32_t) + kv->Bytes();
    result.ok = true;
    return cudaSuccess;
}

} // namespace qwen_moe
} // namespace lsxcommon
