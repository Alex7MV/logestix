#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/lsx_tc_gemm.h"
#include "lsxcommon/log.h"
#include "lsxcommon/shared_experts.h"
#include "lsxcommon/sampling.h"
#include "nvfp4_codec.h"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"
#include "lsx_kernels_quant.cuh"
#include "nvfp4_scale.h"
#include "qwen_reduce.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lsxcommon {
namespace qwen {

// ── QwenKvCache (spec item 15: persistent incremental session state) ───────
// Buffers are per-layer-sliced: full-attention layers own [max_total][NKV*HD]
// k/v halves; linear layers own [max_total][KD*2+VD] raw qkv history halves
// (feed for the causal conv1d) plus a [nv*vd*kd] float GatedDeltaNet col
// accumulator. Allocate() sets up a fresh session (kv_rows == 0, col zeroed);
// EnsureCapacity() grows geometrically and preserves the written rows.

cudaError_t QwenKvCache::Allocate(const QwenPinnedWeights &pw, int max_total_in,
                                  cudaStream_t stream)
{
    Free();
    const QwenTopology &top = pw.top;
    const int NKV = top.num_key_value_heads, HD = top.head_dim;
    const int nk = top.linear_num_key_heads, kd = top.linear_key_head_dim;
    const int nv = top.linear_num_value_heads, vd = top.linear_value_head_dim;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;

    num_full_layers = 0;
    num_linear_layers = 0;
    for (const auto &lw : pw.layers) {
        if (lw.is_linear) ++num_linear_layers;
        else ++num_full_layers;
    }
    if (max_total_in <= 0 || num_full_layers <= 0 || num_linear_layers <= 0)
        return cudaErrorInvalidValue;

    max_total = max_total_in;
    const size_t kvk_bytes =
        (size_t)num_full_layers * (size_t)max_total * NKV * HD * sizeof(__half);
    const size_t kvv_bytes = kvk_bytes;
    const size_t qkv_bytes =
        (size_t)num_linear_layers * (size_t)max_total * S * sizeof(__half);
    const size_t col_bytes =
        (size_t)num_linear_layers * (size_t)nv * vd * kd * sizeof(float);

    if (cudaMalloc(&d_kv_k, kvk_bytes) != cudaSuccess)
        return cudaErrorMemoryAllocation;
    if (cudaMalloc(&d_kv_v, kvv_bytes) != cudaSuccess) { Free(); return cudaErrorMemoryAllocation; }
    if (cudaMalloc(&d_qkv, qkv_bytes) != cudaSuccess)   { Free(); return cudaErrorMemoryAllocation; }
    if (cudaMalloc(&d_col, col_bytes) != cudaSuccess)   { Free(); return cudaErrorMemoryAllocation; }
    if (cudaMemsetAsync(d_col, 0, col_bytes, stream) != cudaSuccess) {
        Free(); return cudaErrorInvalidValue;
    }
    kv_rows = 0;
    last_tok = -1;
    cache_bytes = (int64_t)(kvk_bytes + kvv_bytes + qkv_bytes + col_bytes);
    return cudaSuccess;
}

cudaError_t QwenKvCache::EnsureCapacity(const QwenPinnedWeights &pw, int needed,
                                        cudaStream_t stream)
{
    if (needed <= max_total) return cudaSuccess;
    if (max_total <= 0) return cudaErrorInvalidValue;
    const QwenTopology &top = pw.top;
    const int NKV = top.num_key_value_heads, HD = top.head_dim;
    const int nk = top.linear_num_key_heads, kd = top.linear_key_head_dim;
    const int nv = top.linear_num_value_heads, vd = top.linear_value_head_dim;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;

    const int new_max = std::max(2 * max_total, needed);
    const size_t kvk_bytes =
        (size_t)num_full_layers * (size_t)new_max * NKV * HD * sizeof(__half);
    const size_t kvv_bytes = kvk_bytes;
    const size_t qkv_bytes =
        (size_t)num_linear_layers * (size_t)new_max * S * sizeof(__half);
    const size_t col_bytes =
        (size_t)num_linear_layers * (size_t)nv * vd * kd * sizeof(float);

    __half *n_k = nullptr, *n_v = nullptr, *n_q = nullptr;
    float *n_c = nullptr;
    if (cudaMalloc(&n_k, kvk_bytes) != cudaSuccess) return cudaErrorMemoryAllocation;
    if (cudaMalloc(&n_v, kvv_bytes) != cudaSuccess) { cudaFree(n_k); return cudaErrorMemoryAllocation; }
    if (cudaMalloc(&n_q, qkv_bytes) != cudaSuccess) { cudaFree(n_k); cudaFree(n_v); return cudaErrorMemoryAllocation; }
    if (cudaMalloc(&n_c, col_bytes) != cudaSuccess) {
        cudaFree(n_k); cudaFree(n_v); cudaFree(n_q);
        return cudaErrorMemoryAllocation;
    }

    // Preserve the written rows [0..kv_rows) per layer (col state is row-free).
    const size_t row_k = (size_t)NKV * HD * sizeof(__half);
    const size_t row_q = (size_t)S * sizeof(__half);
    for (int li = 0; li < num_full_layers; ++li) {
        cudaMemcpyAsync(n_k + (size_t)li * new_max * NKV * HD,
                        d_kv_k + (size_t)li * max_total * NKV * HD,
                        (size_t)kv_rows * row_k, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(n_v + (size_t)li * new_max * NKV * HD,
                        d_kv_v + (size_t)li * max_total * NKV * HD,
                        (size_t)kv_rows * row_k, cudaMemcpyDeviceToDevice, stream);
    }
    for (int li = 0; li < num_linear_layers; ++li) {
        cudaMemcpyAsync(n_q + (size_t)li * new_max * S,
                        d_qkv + (size_t)li * max_total * S,
                        (size_t)kv_rows * row_q, cudaMemcpyDeviceToDevice, stream);
    }
    cudaMemcpyAsync(n_c, d_col, col_bytes, cudaMemcpyDeviceToDevice, stream);

    // Drain the copy stream before freeing the old buffers so no in-flight
    // D2D read outlives its source allocation.
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaFree(n_k); cudaFree(n_v); cudaFree(n_q); cudaFree(n_c);
        return cudaErrorInvalidValue;
    }

    cudaFree(d_kv_k);
    cudaFree(d_kv_v);
    cudaFree(d_qkv);
    cudaFree(d_col);
    d_kv_k = n_k;
    d_kv_v = n_v;
    d_qkv = n_q;
    d_col = n_c;
    max_total = new_max;
    cache_bytes = (int64_t)(kvk_bytes + kvv_bytes + qkv_bytes + col_bytes);
    return cudaSuccess;
}

void QwenKvCache::Free()
{
    cudaFree(d_kv_k);
    cudaFree(d_kv_v);
    cudaFree(d_qkv);
    cudaFree(d_col);
    d_kv_k = nullptr;
    d_kv_v = nullptr;
    d_qkv = nullptr;
    d_col = nullptr;
    kv_rows = 0;
    last_tok = -1;
    max_total = 0;
    num_full_layers = 0;
    num_linear_layers = 0;
    cache_bytes = 0;
}

// ── Generation loop ─────────────────────────────────────────────────────────
// Greedy argmax over the last row of the logits matrix [seq, V] is delegated to
// the injected lsxcommon::sample::ISampler (kF16), replacing the former fused
// qwen_argmax_kernel.

cudaError_t RunQwenGeneration(cudaStream_t stream,
                              const QwenPinnedWeights &pw,
                              const std::vector<int32_t> &prompt, int max_new,
                              int eos_id,
                              QwenGenerationResult &result,
                              lsxcommon::sample::ISampler &sampler,
                              QwenKvCache *kv,
                              const lsxcommon::GenerationStream *gs)
{
    const QwenTopology &top = pw.top;
    const int H = top.hidden_size, I = top.intermediate_size;
    const int NH = top.num_attention_heads, HD = top.head_dim;
    const int NKV = top.num_key_value_heads;
    const int nk = top.linear_num_key_heads, kd = top.linear_key_head_dim;
    const int nv = top.linear_num_value_heads, vd = top.linear_value_head_dim;
    const int KD = nk * kd, VD = nv * vd, S = KD * 2 + VD;
    const int V = top.vocab_size;
    const int rotary_dim = (int)(HD * top.partial_rotary_factor);
    result = QwenGenerationResult{};
    if (H <= 0 || I <= 0 || V <= 0 || prompt.empty() || max_new <= 0) {
        result.error_msg = "invalid generation inputs";
        return cudaErrorInvalidValue;
    }
    if (eos_id < 0) {
        result.error_msg = "invalid eos_id";
        return cudaErrorInvalidValue;
    }
    if (!pw.embed.d_packed || !pw.head.d_packed) {
        result.error_msg = "embed/head weights missing";
        return cudaErrorInvalidValue;
    }
    const int max_seq = (int)prompt.size() + max_new;

    // ── incremental session cache (spec item 15) ──
    // kv != nullptr selects the one-token-per-step append-only path: a fresh
    // session allocates a cache for prompt.size()+max_new rows (zeroed
    // GatedDeltaNet state); a continuation grows it if the remaining budget
    // does not fit. kv == nullptr keeps the full-recompute path untouched.
    const bool is_incr = (kv != nullptr);
    if (is_incr) {
        cudaError_t ce;
        if (!kv->allocated())
            ce = kv->Allocate(pw, max_seq, stream);
        else
            ce = kv->EnsureCapacity(pw, kv->kv_rows + max_new, stream);
        if (ce != cudaSuccess) {
            result.error_msg = "kv cache alloc failed";
            return ce;
        }
    }

    // ── arena (all halves unless noted; single cudaMalloc) ──
    size_t half_sz[] = {
        (size_t)max_seq * H,             // d_act
        (size_t)max_seq * H,             // d_normed (input)
        (size_t)max_seq * H,             // d_attn_out
        (size_t)max_seq * H,             // d_post_in
        2 * (size_t)max_seq * I,         // d_gate (fused-MLP scratch [2*seq,I])
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
        (size_t)max_seq * 2,             // d_ids (int32: 2 halves each)
        (size_t)max_seq * 2,             // d_pos (int32: 2 halves each)
    };
    size_t float_sz[] = {
        (size_t)max_seq * nv,            // d_gf
        (size_t)max_seq * nv,            // d_betaf
        (size_t)1,                       // d_maxv_scratch (sampler max-value out)
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
        p += ((hs + 7) / 8) * 8;   // align next buffer to 16 bytes (8 halves)
        return r;
    };
    __half *ph = reinterpret_cast<__half*>(base);
    __half *d_act = adv(ph, half_sz[0]);
    __half *d_normed = adv(ph, half_sz[1]);
    __half *d_attn_out = adv(ph, half_sz[2]);
    __half *d_postin = adv(ph, half_sz[3]);
    __half *d_gate = adv(ph, half_sz[4]);
    __half *d_qc = adv(ph, half_sz[5]);
    __half *d_q = adv(ph, half_sz[6]);
    __half *d_qgate = adv(ph, half_sz[7]);
    __half *d_k = adv(ph, half_sz[8]);
    __half *d_v = adv(ph, half_sz[9]);
    __half *d_attn = adv(ph, half_sz[10]);
    __half *d_cos = adv(ph, half_sz[11]);
    __half *d_sin = adv(ph, half_sz[12]);
    __half *d_qkv = adv(ph, half_sz[13]);
    __half *d_z = adv(ph, half_sz[14]);
    __half *d_ab = adv(ph, half_sz[15]);
    __half *d_conv = adv(ph, half_sz[16]);
    __half *d_core = adv(ph, half_sz[17]);
    __half *d_gated = adv(ph, half_sz[18]);
    __half *d_embed_h = adv(ph, half_sz[19]);
    __half *d_head_h = adv(ph, half_sz[20]);
    __half *d_logits = adv(ph, half_sz[21]);
    int32_t *d_ids = reinterpret_cast<int32_t*>(adv(ph, half_sz[22]));
    int32_t *d_pos = reinterpret_cast<int32_t*>(adv(ph, half_sz[23]));
    float *pf = reinterpret_cast<float*>((uint8_t*)base + half_bytes);
    float *d_gf = pf; pf += float_sz[0];
    float *d_betaf = pf; pf += float_sz[1];
    float *d_maxv_scratch = pf; pf += float_sz[2];

    // Dequant embed + head to half for gather / logits GEMM.
    e = lsxcommon::gpu::dequant_shared_nvfp4(stream, pw.embed.d_packed, pw.embed.packed_bytes,
                             pw.embed.d_scales, pw.embed.scale_bytes,
                             pw.embed.global_scale, d_embed_h,
                             (int64_t)V * H);
    if (e != cudaSuccess) { result.error_msg = "dequant embed"; cudaFree(base); return e; }
    e = lsxcommon::gpu::dequant_shared_nvfp4(stream, pw.head.d_packed, pw.head.packed_bytes,
                             pw.head.d_scales, pw.head.scale_bytes,
                             pw.head.global_scale, d_head_h,
                             (int64_t)V * H);
    if (e != cudaSuccess) { result.error_msg = "dequant head"; cudaFree(base); return e; }

    int32_t *d_argmax = nullptr;
    e = cudaMalloc(&d_argmax, sizeof(int32_t));
    if (e != cudaSuccess) { result.error_msg="argmax alloc"; cudaFree(base); return e; }

    // Pinned staging for the per-step {token, pos} H2D copies in the
    // incremental path: async copies must not read stack locals that die
    // before the stream drains. Allocated in the incremental branch once
    // total_steps is known.
    int32_t *h_step = nullptr;
    bool have_h_step = false;

    // Audit profiling hook (LSX_PERF_PHASES=1).
    const bool perf_phases_env = std::getenv("LSX_PERF_PHASES") != nullptr;

    // Wall-clock timing (stream-delta): t_start at entry, t_ttft at the first
    // generated token, t_stop after the final decode row.
    cudaEvent_t t_start = nullptr, t_ttft = nullptr, t_stop = nullptr;
   cudaEventCreate(&t_start);
   cudaEventCreate(&t_ttft);
   cudaEventCreate(&t_stop);
   // One persistent event for the per-token D2H handshake (created/destroyed
   // once per run instead of once per token — the create/destroy pair is a
   // ~µs driver call that used to run inside the generation loop).
   cudaEvent_t tok_ready = nullptr;
   cudaEventCreate(&tok_ready);
    cudaEventRecord(t_start, stream);
    auto fail = [&](cudaError_t err, std::string msg) -> cudaError_t {
        result.error_msg = std::move(msg);
        if (t_start) cudaEventDestroy(t_start);
        if (t_ttft) cudaEventDestroy(t_ttft);
        if (t_stop) cudaEventDestroy(t_stop);
        if (have_h_step) { cudaFreeHost(h_step); h_step = nullptr; }
        cudaFree(d_argmax);
        cudaFree(base);
        return err;
    };

    std::vector<int32_t> ids = prompt;
    std::vector<int32_t> pos(max_seq);
    for (int i = 0; i < max_seq; ++i) pos[i] = i;
    const double eps = top.rms_norm_eps;
    const int threads = 256;

    if (!is_incr) {
        // ── full-recompute path (unchanged) ──
        for (int step = 0; step < max_new; ++step) {
            const int seq = (int)ids.size();
            std::vector<int32_t> posb(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) posb[i] = (int)i;
            cudaMemcpyAsync(d_ids, ids.data(), ids.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_pos, posb.data(), posb.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);

            e = RunQwenEmbed(stream, d_embed_h, d_ids, d_act, seq, H);
            if (e != cudaSuccess) return fail(e, "embed");

            for (int l = 0; l < (int)pw.layers.size(); ++l) {
                const attn::LayerWeights &lw = pw.layers[l];
                // input RMSNorm
                e = RunQwenRMSNorm(stream, d_act, lw.d_input_norm, d_normed,
                                   seq, H, eps);
                if (e != cudaSuccess) return fail(e, std::string("inputnorm l")+std::to_string(l));
                if (lw.is_linear) {
                    e = RunQwenFusedLinearAttnKernel(stream, lw, top, seq,
                        d_normed, d_attn_out, d_qkv, d_z, d_ab,
                        d_gf, d_betaf, d_conv, d_core, d_gated);
                } else {
                    e = RunQwenFullAttention(stream, lw, top, seq,
                        d_normed, d_attn_out, d_qc, d_q, d_qgate,
                        d_k, d_v, d_attn, d_cos, d_sin, d_pos);
                }
                if (e != cudaSuccess) return fail(e, std::string("attn l")+std::to_string(l));
                // residual: d_act = d_act + d_attn_out
                {
                    const int64_t n = (int64_t)seq * H;
                    lsxcommon::kernels::AddInPlaceKernel<__half><<<(int)((n + threads - 1) / threads), threads,
                                      0, stream>>>(d_attn_out, d_act, n);
                    e = cudaGetLastError(); if (e) return fail(e, std::string("res1 l")+std::to_string(l));
                }
                // post RMSNorm
                e = RunQwenRMSNorm(stream, d_act, lw.d_post_norm, d_postin,
                                   seq, H, eps);
                if (e != cudaSuccess) return fail(e, std::string("postnorm l")+std::to_string(l));
                // fused MLP (down-GEMM accumulates into d_act residual)
                e = RunQwenFusedMLP(stream, lw, top, seq, d_postin, d_act,
                                    d_gate);
                if (e != cudaSuccess) return fail(e, std::string("mlp l")+std::to_string(l));
            }

            // final norm
            e = RunQwenRMSNorm(stream, d_act, pw.d_final_norm, d_postin,
                               seq, H, eps);
            if (e != cudaSuccess) return fail(e, "finalnorm");
            // head logits: LaunchNvfp4Gemm(pw.head, d_postin, seq, d_logits)
            e = LaunchNvfp4Gemm(pw.head, d_postin, seq, d_logits, stream);
            if (e != cudaSuccess) return fail(e, "head");
            // argmax last row
            {
                const int last_row = seq - 1;
                e = sampler.Sample(stream,
                                   d_logits + static_cast<size_t>(last_row) * V, V,
                                   lsxcommon::sample::LogitType::kF16,
                                   d_argmax, d_maxv_scratch);
                if (e != cudaSuccess) return fail(e, "argmax");
            }

            // Publish first generated token id (async, synchronized via event below).
            int32_t tok = 0;
            cudaMemcpyAsync(&tok, d_argmax, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(tok_ready, stream);
            cudaEventSynchronize(tok_ready);
            if (step == 0) cudaEventRecord(t_ttft, stream);
            result.tokens.push_back(tok);
            if (gs) gs->Emit(static_cast<std::uint32_t>(tok),
                             static_cast<std::uint32_t>(result.tokens.size() - 1));
            if (tok == eos_id) {
                result.eos_terminated = true;
                break;
            }
            if (gs && gs->Aborted()) {
                result.eos_terminated = false;
                break;
            }
            ids.push_back(tok);
        }
    } else {
        // ── incremental path: one token per step at kv->kv_rows ──
        // The first prompt.size()-1 steps replay the prompt rows one at a time
        // (their argmax samples are discarded); from prompt.size()-1 on every
        // sample is a generated token. Full-attn k/v rows are appended to the
        // persistent cache; the GatedDeltaNet col accumulator is seeded from
        // and written back to kv->d_col per linear layer.
        const int prompt_steps = (int)prompt.size();
        const int total_steps = prompt_steps + max_new - 1;
        // One pinned slot per step (write-once): the host may run ahead of the
        // stream during prompt replay, so slots must never be reused while a
        // copy from them may still be pending.
        have_h_step = cudaMallocHost(&h_step,
            static_cast<size_t>(2 * total_steps) * sizeof(int32_t)) ==
            cudaSuccess;
        int32_t cur = prompt[0];
        int32_t tok = -1;
        int generated = 0;
        // Audit profiling hook (LSX_PERF_PHASES=1): per-phase event pairs,
        // NO intermediate syncs (true pipeline timing). Capped at the first
        // 8 steps -- enough for a stable sample.
        const bool perf_phases_enabled = perf_phases_env;
        const int prof_steps =
            perf_phases_env ? std::min(total_steps, 8) : 0;
        const int prof_pairs_per_step =
            2 * static_cast<int>(pw.layers.size());
        std::vector<cudaEvent_t> ph_a, ph_b;
        if (perf_phases_enabled) {
            ph_a.resize(static_cast<size_t>(prof_steps) *
                        prof_pairs_per_step);
            ph_b.resize(ph_a.size());
            for (size_t i = 0; i < ph_a.size(); ++i) {
                cudaEventCreate(&ph_a[i]);
                cudaEventCreate(&ph_b[i]);
            }
        }
        int ph_idx = 0;
        const bool perf_phases_cap = perf_phases_enabled &&
                                     (ph_idx < prof_steps *
                                      prof_pairs_per_step - 1);
        for (int step = 0; step < total_steps; ++step) {
            const int seq = 1;
            const int kv_base = kv->kv_rows;
            if (have_h_step) {
                int32_t *slot = h_step + 2 * step;
                slot[0] = cur;
                slot[1] = kv_base;
                cudaMemcpyAsync(d_ids, slot, sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(d_pos, slot + 1, sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream);
            } else {
                // Pinned staging unavailable: fall back to synchronous copies
                // (pageable async H2D is staged synchronously anyway).
                cudaMemcpy(d_ids, &cur, sizeof(int32_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_pos, &kv_base, sizeof(int32_t), cudaMemcpyHostToDevice);
            }

            e = RunQwenEmbed(stream, d_embed_h, d_ids, d_act, seq, H);
            if (e != cudaSuccess) return fail(e, "embed");

            int full_idx = 0, lin_idx = 0;
            for (int l = 0; l < (int)pw.layers.size(); ++l) {
                const attn::LayerWeights &lw = pw.layers[l];
                if (step < prof_steps) cudaEventRecord(ph_a[ph_idx], stream);
                // input RMSNorm
                e = RunQwenRMSNorm(stream, d_act, lw.d_input_norm, d_normed,
                                   seq, H, eps);
                if (e != cudaSuccess) return fail(e, std::string("inputnorm l")+std::to_string(l));
                if (lw.is_linear) {
                    e = RunQwenFusedLinearAttnKernel(stream, lw, top, seq,
                        d_normed, d_attn_out,
                        kv->d_qkv + (int64_t)lin_idx * kv->max_total * S,
                        d_z, d_ab, d_gf, d_betaf, d_conv, d_core, d_gated,
                        kv_base,
                        kv->d_col + (int64_t)lin_idx * nv * vd * kd);
                    ++lin_idx;
                } else {
                    e = RunQwenFullAttention(stream, lw, top, seq,
                        d_normed, d_attn_out, d_qc, d_q, d_qgate,
                        kv->d_kv_k + (int64_t)full_idx * kv->max_total * NKV * HD,
                        kv->d_kv_v + (int64_t)full_idx * kv->max_total * NKV * HD,
                        d_attn, d_cos, d_sin, d_pos, kv_base);
                    ++full_idx;
                }
                if (step < prof_steps) cudaEventRecord(ph_b[ph_idx], stream);
                if (e != cudaSuccess) return fail(e, std::string("attn l")+std::to_string(l));
                // residual: d_act = d_act + d_attn_out
                {
                    const int64_t n = (int64_t)seq * H;
                    lsxcommon::kernels::AddInPlaceKernel<__half><<<(int)((n + threads - 1) / threads), threads,
                                      0, stream>>>(d_attn_out, d_act, n);
                    e = cudaGetLastError(); if (e) return fail(e, std::string("res1 l")+std::to_string(l));
                }
                if (step < prof_steps) cudaEventRecord(ph_a[ph_idx + 1], stream);
                // post RMSNorm
                e = RunQwenRMSNorm(stream, d_act, lw.d_post_norm, d_postin,
                                   seq, H, eps);
                if (e != cudaSuccess) return fail(e, std::string("postnorm l")+std::to_string(l));
                // fused MLP (down-GEMM accumulates into d_act residual)
                e = RunQwenFusedMLP(stream, lw, top, seq, d_postin, d_act,
                                    d_gate);
                if (step < prof_steps) {
                    cudaEventRecord(ph_b[ph_idx + 1], stream);
                    ph_idx += 2;
                }
                if (e != cudaSuccess) return fail(e, std::string("mlp l")+std::to_string(l));
            }

            // final norm
            e = RunQwenRMSNorm(stream, d_act, pw.d_final_norm, d_postin,
                               seq, H, eps);
            if (e != cudaSuccess) return fail(e, "finalnorm");
            // head logits
            e = LaunchNvfp4Gemm(pw.head, d_postin, seq, d_logits, stream);
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
                if (gs) gs->Emit(static_cast<std::uint32_t>(tok),
                                 static_cast<std::uint32_t>(result.tokens.size() - 1));
                if (tok == eos_id) {
                    result.eos_terminated = true;
                    break;
                }
                if (gs && gs->Aborted()) {
                    result.eos_terminated = false;
                    break;
                }
                cur = tok;
            } else {
                cur = prompt[step + 1];
            }
        }
        kv->last_tok = tok;

        if (perf_phases_enabled && ph_idx > 0) {
            // Drain the pipeline, then reduce event pairs (no per-phase syncs).
            cudaEventSynchronize(ph_b[ph_idx - 1]);
            double attn = 0.0, mlpnorm = 0.0;
            for (int p = 0; p < ph_idx; ++p) {
                float ms = 0.f;
                cudaEventElapsedTime(&ms, ph_a[p], ph_b[p]);
                if (p % 2 == 0) attn += ms; else mlpnorm += ms;
            }
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "[PHASES][qwen] attn_ms=%.1f mlp_norm_ms=%.1f "
                     "(pairs=%d steps=%d)",
                     attn, mlpnorm, ph_idx, total_steps);
            log::info(msg);
            for (size_t i = 0; i < ph_a.size(); ++i) {
                cudaEventDestroy(ph_a[i]);
                cudaEventDestroy(ph_b[i]);
            }
        }
    }
    {
        std::vector<float> hlast_f(V);
        // Incremental path always produces the final row at index 0 (seq == 1).
        int64_t base_idx = is_incr ? 0 : (int64_t)ids.size() - 1;
        if (base_idx < 0) base_idx = 0;
        base_idx *= V;
        std::vector<__half> hlast(V);
        // Stream-ordered async D2H: the blocking null-stream cudaMemcpy here
        // used to serialize the whole pipeline inside the timed region
        // (audit F-02). Completion is guaranteed by t_stop below, which is
        // recorded after this copy on the same stream.
        cudaMemcpyAsync(hlast.data(), d_logits + base_idx,
                        V * sizeof(__half), cudaMemcpyDeviceToHost, stream);
        for (int i = 0; i < V; ++i)
            hlast_f[i] = __half2float(hlast[i]);
        result.final_logits = std::move(hlast_f);
    }

    // Publish timing and residency before tearing down the arena. The event
    // wait doubles as the completion fence for the async logits copy above.
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
    result.resident_bytes = pw.resident_bytes + half_bytes
                            + float_bytes + sizeof(int32_t)
                            + (is_incr ? kv->Bytes() : 0);
    if (have_h_step) cudaFreeHost(h_step);
    cudaFree(d_argmax);
    cudaFree(base);
    result.ok = true;
    return cudaSuccess;
}


} // namespace qwen
} // namespace lsxcommon
