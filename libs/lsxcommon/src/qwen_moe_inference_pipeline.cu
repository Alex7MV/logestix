#include "lsxcommon/qwen_moe_inference_pipeline.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/qwen_moe_load.h"
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
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace lsxcommon {
namespace qwen_moe {

namespace {

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
} // namespace


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
    lsxcommon::moe::MoEExpertStore &store, QwenMoeExpertDeviceCache &cache,
    QwenMoeGenScratch &scratch,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    qwen::QwenGenerationResult &result, lsxcommon::sample::ISampler &sampler,
    qwen::QwenKvCache *kv,
    const lsxcommon::GenerationStream *gs)
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
        E <= 0 || K <= 0 || K > moe::kMaxTopK || I <= 0 || SI <= 0 ||
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
        (size_t)moe::kMaxTopK * 2 * I,        // d_ygu_batch [Ktop,2I]
        (size_t)moe::kMaxTopK * I,            // d_h_batch   [Ktop,I]
        (size_t)moe::kMaxTopK * H,            // d_yp_batch  [Ktop,H]
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
    const size_t gemv_tbl_bytes = ((sizeof(moe::MoEGvSlotDev) *
                                        static_cast<size_t>(moe::kMaxTopK)) +
                                   63) /
                                  64 * 64;
    // Batched TC routed chain (80-tok/s decode round): per-slot quant
    // outputs for the down GEMM + w13 split-K partials (arena slices).
    const int tc_splitk = lsx_tc::LsxTcBatchSplitK(H, 2 * I);
    const size_t qpack_b_bytes =
        ((static_cast<size_t>(moe::kMaxTopK) * I / 2) + 63) / 64 * 64;
    const size_t qscales_b_bytes =
        ((static_cast<size_t>(moe::kMaxTopK) * ((I + 15) / 16)) + 63) / 64 * 64;
    const size_t tc_splitk_bytes =
        tc_splitk > 1
            ? ((static_cast<size_t>(tc_splitk) * moe::kMaxTopK * 2 * I * 4 + 63) /
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
    moe::MoEGvSlotDev *d_gemv_tbl =
        reinterpret_cast<moe::MoEGvSlotDev *>(d_xact_qs + xact_qs_bytes);
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
    moe::MoERoutedSlot *d_slots = nullptr;
    e = cudaMalloc(&d_slots, sizeof(moe::MoERoutedSlot) * static_cast<size_t>(K));
    if (e != cudaSuccess) {
        result.error_msg = "slots alloc";
        cudaFree(d_argmax);
        cudaFree(d_tokens);
        return e;
    }

    // Pinned staging: per-step {token,pos} H2D and per-layer routed slots D2H.
    int32_t *h_step = nullptr;
    moe::MoERoutedSlot *h_slots = nullptr;
    bool have_pinned =
        cudaMallocHost(&h_step, static_cast<size_t>(2 * (prompt.size() + max_new)) *
                                    sizeof(int32_t)) == cudaSuccess &&
        cudaMallocHost(&h_slots, sizeof(moe::MoERoutedSlot) *
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
    // Perf knob: dual-stream staging can be disabled for narrow-bus GPUs
    // (gpu_perf policy) — copy_stream stays null -> single-stream path.
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t ev_uploads[16] = {};
    cudaEvent_t ev_front = nullptr;
    bool have_copy =
        lsxcommon::lsx_nvfp4::LsxDualStreamStaging() &&
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
    static const bool kTcBatchDisable =
        std::getenv("LSX_TC_BATCH_OFF") != nullptr;
    const bool kTcBatchGeom = qwen::QwenTcFastMath() &&
                              store.d_seg_wgs != nullptr && (I & 15) == 0 &&
                              !kGemvExpertsEnv && !kTcBatchDisable;
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
    // EOS fence cadence: drain-and-check the sampled window every N generated
    // tokens; worst-case post-EOS waste is N-1 forward passes.
    constexpr int kEosFenceBatch = 8;
    const int prompt_steps = static_cast<int>(prompt.size());
    const int total_steps = prompt_steps + max_new - 1;
    int32_t cur = prompt[0];
    int32_t tok = -1;
    int generated = 0;
    int emitted = 0;
    bool abort_stream = false;

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

        e = qwen::RunAttnEmbed(stream, d_embed_h, d_ids, d_act, seq, H);
        if (e != cudaSuccess) return fail(e, "embed");

        int full_idx = 0, lin_idx = 0;
        for (int l = 0; l < static_cast<int>(pw.layers.size()); ++l) {
            const QwenMoeLayerWeights &lw = pw.layers[static_cast<size_t>(l)];
            e = qwen::RunAttnRMSNorm(stream, d_act, lw.attn.d_input_norm,
                                     d_normed, seq, H, eps);
            if (e != cudaSuccess) return fail(e, "inputnorm l" + std::to_string(l));
            if (lw.attn.is_linear) {
                e = qwen::RunAttnFusedLinearAttnKernel(stream, lw.attn, qwen::MakeAttnGeometry(top), seq,
                    d_normed, d_attn_out,
                    kv->d_qkv + (int64_t)lin_idx * kv->max_total * S,
                    d_z, d_ab, d_gf, d_betaf, d_conv, d_core, d_gated,
                    kv_base,
                    kv->d_col + (int64_t)lin_idx * nv * vd * kd);
                ++lin_idx;
            } else {
                e = qwen::RunAttnFullAttention(stream, lw.attn, qwen::MakeAttnGeometry(top), seq,
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
            e = qwen::RunAttnRMSNorm(stream, d_act, lw.attn.d_post_norm,
                                     d_postin, seq, H, eps);
            if (e != cudaSuccess) return fail(e, "postnorm l" + std::to_string(l));

            // ── MoE FFN: router → routed experts → shared combine ──
            e = moe::RunMoERouterTopK(stream, reinterpret_cast<const __nv_bfloat16 *>(lw.d_router),
                                     d_postin, seq, E, K, H, d_slots);
            if (e != cudaSuccess) return fail(e, "router l" + std::to_string(l));
            {
                // Accumulator init is independent of the routing result; enqueue
                // it behind the router so the GPU has work queued while the
                // host waits for the slot readback below. The TC-batch path
                // folds the seed into its reduce kernel (same fp32 order) and
                // skips this launch.
                const int64_t n = seq * H;
                e = moe::RunMoEAccInitForTest(stream, d_act, d_moe_acc,
                                        static_cast<int>(n));
                if (e != cudaSuccess) return fail(e, "acc-init l" + std::to_string(l));
            }
            // No-handshake mode (full residency steady state): the routed
            // chain composes its table on device from d_slots; the host
            // never reads the router output back, so no fence/poll here.
            // TC-batch steady state shares the same readiness mirror; the
            // GEMV experimental route keeps priority when explicitly
            // requested via LSX_GEMV_EXPERTS.
            const bool moe_nohs_ready =
                lsxcommon::moe::MoEGemvNohandshake(store);
            const bool tc_batch = kTcBatchGeom && moe_nohs_ready;
            const bool nohandshake = kGemvExpertsEnv && moe_nohs_ready;
            if (!tc_batch && !nohandshake) {
                // Host needs the selected ids to stage experts: small D2H +
                // event fence once per layer. Busy-poll the event instead of
                // the blocking cudaEventSynchronize — with K layers × steps
                // handshakes per generation, host wake latency is pure GPU
                // idle time.
                cudaMemcpyAsync(h_slots, d_slots,
                                sizeof(moe::MoERoutedSlot) * K,
                                cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(tok_ready, stream);
                {
                    cudaError_t q;
                    while ((q = cudaEventQuery(tok_ready)) ==
                           cudaErrorNotReady) {}
                    (void)q;
                }
            }
            moe::MoERoutedSlot *slots_host = h_slots;
            // Quantize the layer input ONCE for all experts + shared GEMMs
            // (TC path): removes ~8 redundant ActQuant passes per layer.
            // Buffers are persistent arena slices (round-20 pass) -- the
            // old cudaMallocAsync/FreeAsync pairs cost two driver calls
            // per layer on WDDM for bytes that never outlive this layer.
            uint8_t *xq = nullptr, *xqs = nullptr;
            if (qwen::QwenTcFastMath()) {
                xq = d_xact_q;
                xqs = d_xact_qs;
                if (lsxcommon::lsx_tc::LsxTcQuantizeActsNvfp4(
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
                e = moe::RunMoERoutedTcBatch(
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
                            cudaError_t s1 = moe::RunMoEGemvGatherTable(
                                stream, d_slots, K,
                                store.d_gemv_meta + flat_base,
                                d_gemv_tbl);
                            cudaError_t s2 =
                                s1 == cudaSuccess
                                    ? moe::RunMoERoutedGemmVBatchCore(
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
                            moe::RunMoESharedCombine(stream, lw.shared_w1,
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
                    e = moe::RunMoERoutedGemmVNoHandshake(
                        stream, store, cache, l, d_slots, K, d_postin, H, I,
                        d_moe_acc, d_ygu_batch, d_h_batch, d_yp_batch,
                        d_gemv_tbl, moe_error);
                }
            } else {
                e = moe::RunMoERoutedFFN(stream, store, cache, l, slots_host,
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
            e = moe::RunMoESharedCombine(stream, lw.shared_w1, lw.shared_w2,
                                        lw.shared_w3, lw.d_shared_gate,
                                        d_postin, seq, H, SI, d_moe_acc,
                                        d_si, d_sh, d_gate_val, d_act, xq,
                                        xqs);
            if (e != cudaSuccess) return fail(e, "shared l" + std::to_string(l));
            }
        }

        e = qwen::RunAttnRMSNorm(stream, d_act, pw.d_final_norm, d_postin,
                                 seq, H, eps);
        if (e != cudaSuccess) return fail(e, "finalnorm");
        e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_postin, pw.head.d_packed,
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
            // Batched EOS fence (kimi round-25 pattern keeps the per-step
            // argmax D2H off the hot path; this fence bounds only the
            // post-EOS waste): every kEosFenceBatch generated tokens, drain
            // the freshly recorded window once and stop enqueueing forward
            // passes once EOS is seen. Post-hoc check — the GPU math and
            // the recorded token stream stay bit-identical; TruncateAtEos
            // still normalizes the tail.
            if (eos_id >= 0 && generated % kEosFenceBatch == 0) {
                std::vector<int32_t> eos_win(static_cast<size_t>(kEosFenceBatch));
                cudaMemcpyAsync(eos_win.data(),
                                d_tokens + (generated - kEosFenceBatch),
                                sizeof(int32_t) * kEosFenceBatch,
                                cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(tok_ready, stream);
                cudaEventSynchronize(tok_ready);
                bool hit = false;
                if (gs) {
                    for (int b = 0; b < kEosFenceBatch; ++b) {
                        const int32_t t = eos_win[b];
                        if (t == eos_id) { hit = true; break; }
                        gs->Emit(static_cast<std::uint32_t>(t),
                                 static_cast<std::uint32_t>(emitted + b));
                    }
                    emitted = generated;
                } else {
                    for (int32_t t : eos_win)
                        if (t == eos_id) { hit = true; break; }
                }
                if (hit) break;
                if (gs && gs->Aborted()) { abort_stream = true; break; }
            }
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
        if (gs) {
            for (int i = emitted; i < generated; ++i) {
                if (hgen[i] == eos_id) break;
                gs->Emit(static_cast<std::uint32_t>(hgen[i]),
                         static_cast<std::uint32_t>(i));
            }
        }
        result.eos_terminated = moe::MoETruncateTokensAtEos(result.tokens, eos_id);
        if (!result.tokens.empty()) tok = result.tokens.back();
    }
    (void)abort_stream;
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
        cudaMemcpyAsync(h_slots, d_slots, sizeof(moe::MoERoutedSlot) * K,
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        for (int r = 0; r < kAuditRounds; ++r) {
            int si = 0, fullA = 0, linA = 0;
            auto stamp = [&]() { cudaEventRecord(aev[si++], stream); };
            auto mark = [&]() {
                if (si <= nseg) stamp();
            };
            mark();   // t0: start-of-round boundary
            qwen::RunAttnEmbed(stream, d_embed_h, d_ids, d_act, 1, H);
            mark();
            for (size_t l = 0; l < nlayers; ++l) {
                const QwenMoeLayerWeights &lw = pw.layers[l];
                qwen::RunAttnRMSNorm(stream, d_act, lw.attn.d_input_norm,
                                     d_normed, 1, H, eps);
                if (lw.attn.is_linear) {
                    qwen::RunAttnFusedLinearAttnKernel(
                        stream, lw.attn, qwen::MakeAttnGeometry(top), 1, d_normed, d_attn_out,
                        kv->d_qkv +
                            (int64_t)linA * kv->max_total * S,
                        d_z, d_ab, d_gf, d_betaf, d_conv, d_core, d_gated,
                        kv->kv_rows,
                        kv->d_col + (int64_t)linA * nv * vd * kd);
                    ++linA;
                } else {
                    qwen::RunAttnFullAttention(
                        stream, lw.attn, qwen::MakeAttnGeometry(top), 1, d_normed, d_attn_out, d_qc,
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
                moe::RunMoERoutedFFN(stream, store, cache,
                                    static_cast<int>(l), h_slots, K,
                                    d_postin, H, I, d_moe_acc, d_yg, d_yu,
                                    d_y2, moe_error);
                moe::RunMoESharedCombine(stream, lw.shared_w1, lw.shared_w2,
                                        lw.shared_w3, lw.d_shared_gate,
                                        d_postin, 1, H, SI, d_moe_acc, d_si,
                                        d_sh, d_gate_val, d_act, nullptr,
                                        nullptr);
                stamp();   // ffn segment: router + routed chain + shared
            }
            qwen::RunAttnRMSNorm(stream, d_act, pw.d_final_norm, d_postin, 1,
                                 H, eps);
            lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_postin, pw.head.d_packed,
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
