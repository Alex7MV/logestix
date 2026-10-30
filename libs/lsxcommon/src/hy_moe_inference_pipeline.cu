#include "lsxcommon/hy_moe_inference_pipeline.h"

#include "hy_moe_constants.h"
#include "lsx_kernels_act.cuh"
#include "lsx_kernels_math.cuh"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"

#include <absl/strings/str_cat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lsxcommon::hy {
namespace {

constexpr int kThreads = 256;

std::string LayerTag(const char *what, int layer)
{
    return absl::StrCat(what, " l", layer);
}

inline int Blocks(int64_t n) { return static_cast<int>(n / kThreads) + (n % kThreads != 0); }

} // namespace

std::int64_t HySession::Bytes() const
{
    if (!d_k) return 0;
    const int64_t per_layer = geom.LayerStride();
    return 2 * static_cast<int64_t>(geom.layers) * per_layer * 2 +
           static_cast<int64_t>(gen_tail.capacity()) * 4;
}

void HySession::Free()
{
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    d_k = nullptr;
    d_v = nullptr;
    geom = {};
    capacity = 0;
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

void HySession::Reset()
{
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

cudaError_t HySession::EnsureCapacity(const HyTopology &top, int64_t want,
                                      cudaStream_t stream)
{
    if (want <= capacity && d_k) return cudaSuccess;
    const int64_t cap = std::max<int64_t>(want, capacity * 2);
    const KvCacheGeometry g = HyKvGeometry(top, cap);
    const int64_t kv_elems = g.Elements();

    __half *nk = nullptr;
    __half *nv = nullptr;
    if (cudaMalloc(&nk, static_cast<size_t>(kv_elems) * sizeof(__half)) !=
        cudaSuccess)
        return cudaGetLastError();
    if (cudaMalloc(&nv, static_cast<size_t>(kv_elems) * sizeof(__half)) !=
        cudaSuccess) {
        cudaFree(nk);
        return cudaGetLastError();
    }
    // Preserve the cached prefix (session continuation / growth mid-request).
    if (d_k && kv_rows > 0) {
        for (int l = 0; l < static_cast<int>(g.layers); ++l) {
            const size_t bytes = static_cast<size_t>(kv_rows) *
                                 static_cast<size_t>(g.LayerStride()) *
                                 sizeof(__half);
            cudaMemcpyAsync(nk + l * g.LayerStride(),
                            d_k + l * geom.LayerStride(), bytes,
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(nv + l * g.LayerStride(),
                            d_v + l * geom.LayerStride(), bytes,
                            cudaMemcpyDeviceToDevice, stream);
        }
        cudaStreamSynchronize(stream);
    }
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    d_k = nk;
    d_v = nv;
    geom = g;
    capacity = cap;
    return cudaSuccess;
}

void HyScratch::Free()
{
    if (arena) cudaFree(arena);
    arena = nullptr;
    arena_bytes = 0;
    half_elems = 0;
    float_elems = 0;
    if (d_ids) cudaFree(d_ids);
    if (d_argmax) cudaFree(d_argmax);
    if (d_maxv) cudaFree(d_maxv);
    if (d_slots) cudaFree(d_slots);
    if (h_slots) cudaFreeHost(h_slots);
    if (d_logits) cudaFree(d_logits);
    d_ids = nullptr;
    d_argmax = nullptr;
    d_maxv = nullptr;
    d_slots = nullptr;
    h_slots = nullptr;
    d_logits = nullptr;
    logits_elems = 0;
}

namespace {

// Dense SwiGLU MLP (layer 0's MLP and the shared expert): h = silu(gate x) * up x,
// out = down h.
cudaError_t DenseMlp(cudaStream_t stream, const __half *x,
                     const lsx_nvfp4::LsxNvfp4Weight &gate,
                     const lsx_nvfp4::LsxNvfp4Weight &up,
                     const lsx_nvfp4::LsxNvfp4Weight &down, int H, int I,
                     __half *d_yg, __half *d_yu, __half *d_h, __half *out)
{
    cudaError_t e = lsx_nvfp4::RunLsxNvfp4Gemm(
        stream, x, gate.d_packed, gate.d_scales, gate.global_scale, d_yg, 1, H,
        I, false);
    if (e != cudaSuccess) return e;
    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, x, up.d_packed, up.d_scales,
                                   up.global_scale, d_yu, 1, H, I, false);
    if (e != cudaSuccess) return e;
    lsxcommon::kernels::SwigluKernel<__half, lsxcommon::kernels::ExpFlavor::Glibc,
                                     false>
        <<<Blocks(I), kThreads, 0, stream>>>(d_yg, d_yu, d_h, I, 0.0f);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    return lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_h, down.d_packed,
                                      down.d_scales, down.global_scale, out, 1,
                                      I, H, false);
}

} // namespace

gpu::InferenceResult RunHyInference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, const HyPinnedWeights &pw, HySession &session,
    HyScratch &scratch, sample::ISampler &sampler, cudaStream_t stream,
    bool ignore_eos, const GenerationStream *gs)
{
    gpu::InferenceResult out;
    out.num_layers = pw.top.num_layers;
    (void)model_path;

    const HyTopology &top = pw.top;
    const int H = top.hidden_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;
    const int V = top.vocab_size;
    const int E = top.num_experts;
    const int K = top.num_experts_per_tok;
    const int I = top.moe_intermediate_size;
    const int SI = top.shared_intermediate_size;
    const int DI = top.dense_intermediate_size;
    const int L = top.num_layers;
    const int imax = std::max(std::max(I, SI), DI);
    const float eps = static_cast<float>(top.rms_norm_eps);
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    const float theta = static_cast<float>(top.rope_theta);
    const float routed_scale = static_cast<float>(top.router_scaling_factor);
    const int eos_id = top.eos_token_id;

    if (H <= 0 || V <= 0 || NH <= 0 || NKV <= 0 || HD <= 0 || L <= 0 ||
        E <= 0 || K <= 0 || E > 512 || K > 32 || K > 8) {
        out.error_msg = "invalid Hy-Memory topology";
        return out;
    }
    if (input_ids.empty() || max_gen_tokens <= 0) {
        out.error_msg = "empty prompt or max_gen_tokens <= 0";
        return out;
    }

    // ── Session decision (shared incremental-session policy) ──
    const qwen::SessionDecision decision = qwen::DecideSession(
        session.allocated(), session.last_tok, session.gen_tail, input_ids);
    const std::vector<int32_t> prefill = decision.gen_input;
    if (!decision.is_continuation) session.Reset();

    const int prompt_steps = static_cast<int>(prefill.size());
    const int total_steps = prompt_steps + max_gen_tokens - 1;
    const int64_t need_cap = session.kv_rows + prompt_steps + max_gen_tokens + 8;
    if (cudaError_t ce = session.EnsureCapacity(top, need_cap, stream);
        ce != cudaSuccess) {
        out.error_msg = absl::StrCat("kv cache alloc failed: ",
                                     cudaGetErrorString(ce));
        return out;
    }

    // ── Scratch (grown on demand; one arena for halves, one for floats) ──
    size_t half_sz[] = {
        static_cast<size_t>(H),            // d_act
        static_cast<size_t>(H),            // d_normed
        static_cast<size_t>(H),            // d_attn_out
        static_cast<size_t>(H),            // d_postin
        static_cast<size_t>(NH * HD),      // d_q
        static_cast<size_t>(NKV * HD),     // d_kp
        static_cast<size_t>(NKV * HD),     // d_vp
        static_cast<size_t>(NH * HD),      // d_attn
        static_cast<size_t>(imax),         // d_yg (dense/shared gate)
        static_cast<size_t>(imax),         // d_yu
        static_cast<size_t>(imax),         // d_h
        static_cast<size_t>(H),            // d_shared
        static_cast<size_t>(H),            // d_mlp_out
        static_cast<size_t>(I) * K,        // d_exp_gate  [K][I]
        static_cast<size_t>(I) * K,        // d_exp_up    [K][I]
        static_cast<size_t>(H) * K,        // d_exp_out   [K][H]
    };
    size_t half_bytes = 0;
    for (auto s : half_sz) half_bytes += ((s + 7) / 8) * 8 * sizeof(__half);
    half_bytes = (half_bytes + 15) & ~static_cast<size_t>(15);
    size_t float_bytes = static_cast<size_t>(H) * sizeof(float);  // d_moe_acc
    float_bytes = (float_bytes + 15) & ~static_cast<size_t>(15);
    const size_t arena_need = half_bytes + float_bytes;
    if (!scratch.arena || scratch.arena_bytes < arena_need) {
        if (scratch.arena) cudaFree(scratch.arena);
        scratch.arena = nullptr;
        if (cudaMalloc(&scratch.arena, arena_need) != cudaSuccess) {
            out.error_msg = "scratch alloc failed";
            return out;
        }
        scratch.arena_bytes = arena_need;
    }
    if (!scratch.d_ids) {
        cudaMalloc(&scratch.d_ids, sizeof(int32_t));
        cudaMalloc(&scratch.d_argmax, sizeof(int32_t));
        cudaMalloc(&scratch.d_maxv, sizeof(float));
        cudaMalloc(&scratch.d_slots,
                   sizeof(HyRoutedSlot) * static_cast<size_t>(K));
        cudaMallocHost(&scratch.h_slots,
                       sizeof(HyRoutedSlot) * static_cast<size_t>(K));
    }
    if (scratch.logits_elems < V) {
        if (scratch.d_logits) cudaFree(scratch.d_logits);
        if (cudaMalloc(&scratch.d_logits,
                       static_cast<size_t>(V) * sizeof(float)) != cudaSuccess) {
            out.error_msg = "logits alloc failed";
            return out;
        }
        scratch.logits_elems = V;
    }
    __half *ph = static_cast<__half *>(scratch.arena);
    auto adv = [&](size_t elems) {
        __half *r = ph;
        ph += ((elems + 7) / 8) * 8;
        return r;
    };
    __half *d_act = adv(half_sz[0]);
    __half *d_normed = adv(half_sz[1]);
    __half *d_attn_out = adv(half_sz[2]);
    __half *d_postin = adv(half_sz[3]);
    __half *d_q = adv(half_sz[4]);
    __half *d_kp = adv(half_sz[5]);
    __half *d_vp = adv(half_sz[6]);
    __half *d_attn = adv(half_sz[7]);
    __half *d_yg = adv(half_sz[8]);
    __half *d_yu = adv(half_sz[9]);
    __half *d_h = adv(half_sz[10]);
    __half *d_shared = adv(half_sz[11]);
    __half *d_mlp_out = adv(half_sz[12]);
    __half *d_exp_gate = adv(half_sz[13]);
    __half *d_exp_up = adv(half_sz[14]);
    __half *d_exp_out = adv(half_sz[15]);
    float *d_moe_acc = reinterpret_cast<float *>(
        reinterpret_cast<uint8_t *>(scratch.arena) + half_bytes);

    int32_t *d_ids = scratch.d_ids;
    int32_t *d_argmax = scratch.d_argmax;
    float *d_maxv = scratch.d_maxv;
    HyRoutedSlot *d_slots = scratch.d_slots;
    HyRoutedSlot *h_slots = scratch.h_slots;
    float *d_logits = scratch.d_logits;

    cudaEvent_t t_start = nullptr, t_ttft = nullptr, t_stop = nullptr;
    cudaEvent_t tok_ready = nullptr;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_ttft);
    cudaEventCreate(&t_stop);
    cudaEventCreateWithFlags(&tok_ready, cudaEventDisableTiming);
    cudaEventRecord(t_start, stream);

    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(max_gen_tokens));
    float first_maxv = 0.0f;
    bool eos_terminated = false;

    auto fail = [&](cudaError_t err, const std::string &msg) {
        out.error_msg = msg;
        if (err != cudaSuccess)
            out.error_msg += absl::StrCat(" (", cudaGetErrorString(err), ")");
        if (t_start) cudaEventDestroy(t_start);
        if (t_ttft) cudaEventDestroy(t_ttft);
        if (t_stop) cudaEventDestroy(t_stop);
        if (tok_ready) cudaEventDestroy(tok_ready);
        return out;
    };

    int32_t cur = prefill[0];
    int generated = 0;
    bool aborted = false;

    // Per-slot pointer views, resolved on the host from the router readback.
    const uint8_t *pk_gate[8] = {};
    const uint8_t *pk_up[8] = {};
    const uint8_t *pk_down[8] = {};
    const uint8_t *sc_gate[8] = {};
    const uint8_t *sc_up[8] = {};
    const uint8_t *sc_down[8] = {};
    float gs_gate[8] = {};
    float gs_up[8] = {};
    float gs_down[8] = {};

    for (int step = 0; step < total_steps; ++step) {
        const int64_t pos = session.kv_rows;
        cudaMemcpyAsync(d_ids, &cur, sizeof(int32_t), cudaMemcpyHostToDevice,
                        stream);

        cudaError_t e = RunHyEmbed(stream, pw.d_embed, d_ids, d_act, 1, H);
        if (e != cudaSuccess) return fail(e, "embed");

        for (int l = 0; l < L; ++l) {
            const HyLayerWeights &lw = pw.layers[static_cast<size_t>(l)];

            // ── attention block ──
            e = RunHyNorm(stream, d_act, lw.attn.d_input_norm, d_normed, 1, H,
                          eps);
            if (e != cudaSuccess) return fail(e, LayerTag("norm", l));
            e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_normed, lw.attn.q.d_packed,
                                           lw.attn.q.d_scales,
                                           lw.attn.q.global_scale, d_q, 1, H,
                                           NH * HD, false);
            if (e != cudaSuccess) return fail(e, LayerTag("q", l));
            e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_normed, lw.attn.k.d_packed,
                                           lw.attn.k.d_scales,
                                           lw.attn.k.global_scale, d_kp, 1, H,
                                           NKV * HD, false);
            if (e != cudaSuccess) return fail(e, LayerTag("k", l));
            e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_normed, lw.attn.v.d_packed,
                                           lw.attn.v.d_scales,
                                           lw.attn.v.global_scale, d_vp, 1, H,
                                           NKV * HD, false);
            if (e != cudaSuccess) return fail(e, LayerTag("v", l));
            e = RunHyHeadNorm(stream, d_q, lw.attn.d_q_norm, d_q, 1, NH, HD, eps);
            if (e != cudaSuccess) return fail(e, LayerTag("qnorm", l));
            e = RunHyHeadNorm(stream, d_kp, lw.attn.d_k_norm, d_kp, 1, NKV, HD,
                              eps);
            if (e != cudaSuccess) return fail(e, LayerTag("knorm", l));
            e = RunHyRope(stream, d_q, d_kp, 1, NH, NKV, HD, theta,
                          static_cast<int>(pos));
            if (e != cudaSuccess) return fail(e, LayerTag("rope", l));

            // Append k/v to the layer's cache slice (pos-major rows).
            __half *k_base = session.d_k + l * session.geom.LayerStride();
            __half *v_base = session.d_v + l * session.geom.LayerStride();
            {
                const size_t row =
                    static_cast<size_t>(session.geom.pos_stride) * sizeof(__half);
                cudaMemcpyAsync(k_base + pos * session.geom.pos_stride, d_kp,
                                row, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(v_base + pos * session.geom.pos_stride, d_vp,
                                row, cudaMemcpyDeviceToDevice, stream);
            }

            // Dense-equivalent causal GQA decode attention via the neutral
            // kernels (split-KV past 512 cached rows; same policy as
            // RunAttnFullAttention).
            if (pos + 1 > 512) {
                e = qwen::RunAttnFullAttnKernelSplitKV(
                    stream, d_q, k_base, v_base, d_attn, 1, NH, HD, NKV, scale,
                    static_cast<int>(pos));
            } else {
                e = qwen::RunAttnFullAttnKernel(stream, d_q, k_base, v_base,
                                                d_attn, 1, NH, HD, NKV, scale,
                                                static_cast<int>(pos));
            }
            if (e != cudaSuccess) return fail(e, LayerTag("attn", l));

            e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_attn, lw.attn.o.d_packed,
                                           lw.attn.o.d_scales,
                                           lw.attn.o.global_scale, d_attn_out, 1,
                                           NH * HD, H, false);
            if (e != cudaSuccess) return fail(e, LayerTag("o", l));
            lsxcommon::kernels::AddInPlaceKernel<__half>
                <<<Blocks(H), kThreads, 0, stream>>>(d_attn_out, d_act, H);
            if ((e = cudaGetLastError()) != cudaSuccess)
                return fail(e, LayerTag("res1", l));

            // ── MLP block ──
            e = RunHyNorm(stream, d_act, lw.attn.d_post_norm, d_postin, 1, H,
                          eps);
            if (e != cudaSuccess) return fail(e, LayerTag("postnorm", l));

            if (!lw.is_moe) {
                e = DenseMlp(stream, d_postin, lw.mlp_gate, lw.mlp_up,
                             lw.mlp_down, H, DI, d_yg, d_yu, d_h, d_mlp_out);
                if (e != cudaSuccess) return fail(e, LayerTag("mlp", l));
            } else {
                // Router: one D2H readback of the top-k slots per layer.
                e = RunHyRouter(stream, lw.d_router, lw.d_router_bias, d_postin,
                                1, E, K, H, routed_scale, d_slots);
                if (e != cudaSuccess) return fail(e, LayerTag("router", l));
                cudaMemcpyAsync(h_slots, d_slots,
                                sizeof(HyRoutedSlot) * static_cast<size_t>(K),
                                cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(tok_ready, stream);
                cudaEventSynchronize(tok_ready);

                // Gated shared expert.
                e = DenseMlp(stream, d_postin, lw.shared_gate, lw.shared_up,
                             lw.shared_down, H, SI, d_yg, d_yu, d_h, d_shared);
                if (e != cudaSuccess) return fail(e, LayerTag("shared", l));

                for (int k = 0; k < K; ++k) {
                    const int eid = h_slots[k].eid;
                    if (eid < 0 || eid >= E)
                        return fail(cudaErrorInvalidValue,
                                    LayerTag("router eid", l));
                    const size_t ui = static_cast<size_t>(eid);
                    pk_gate[k] = lw.packed_gate[ui];
                    sc_gate[k] = lw.scales_gate[ui];
                    gs_gate[k] = lw.gs_gate[ui];
                    pk_up[k] = lw.packed_up[ui];
                    sc_up[k] = lw.scales_up[ui];
                    gs_up[k] = lw.gs_up[ui];
                    pk_down[k] = lw.packed_down[ui];
                    sc_down[k] = lw.scales_down[ui];
                    gs_down[k] = lw.gs_down[ui];
                }

                // gate/up per expert (batched), swiglu, down per expert,
                // weighted FP32 accumulate.
                e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                    stream, d_postin, pk_gate, sc_gate, gs_gate, K, 0, d_exp_gate,
                    H, I, false);
                if (e != cudaSuccess) return fail(e, LayerTag("w1", l));
                e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                    stream, d_postin, pk_up, sc_up, gs_up, K, 0, d_exp_up, H, I,
                    false);
                if (e != cudaSuccess) return fail(e, LayerTag("w3", l));
                const int64_t kn = static_cast<int64_t>(K) * I;
                lsxcommon::kernels::SwigluKernel<
                    __half, lsxcommon::kernels::ExpFlavor::Glibc, false>
                    <<<Blocks(kn), kThreads, 0, stream>>>(d_exp_gate, d_exp_up,
                                                          d_exp_gate, kn, 0.0f);
                if ((e = cudaGetLastError()) != cudaSuccess)
                    return fail(e, LayerTag("swiglu", l));
                e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                    stream, d_exp_gate, pk_down, sc_down, gs_down, K, I,
                    d_exp_out, I, H, false);
                if (e != cudaSuccess) return fail(e, LayerTag("w2", l));
                e = RunHyExpertAccumulate(stream, d_moe_acc, d_exp_out, d_slots,
                                          K, H);
                if (e != cudaSuccess) return fail(e, LayerTag("acc", l));
                e = RunHyMoECombine(stream, d_moe_acc, d_shared, d_mlp_out, 1,
                                    H);
                if (e != cudaSuccess) return fail(e, LayerTag("combine", l));
            }
            lsxcommon::kernels::AddInPlaceKernel<__half>
                <<<Blocks(H), kThreads, 0, stream>>>(d_mlp_out, d_act, H);
            if ((e = cudaGetLastError()) != cudaSuccess)
                return fail(e, LayerTag("res2", l));
        }

        // ── final norm + lm_head (FP16 weights, FP32 accumulate) ──
        e = RunHyNorm(stream, d_act, pw.d_final_norm, d_postin, 1, H, eps);
        if (e != cudaSuccess) return fail(e, "final norm");
        e = gemv::RunLxGemmVec(stream, pw.d_head, d_postin, 1, V, H,
                               gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
                               d_logits, gemv::LxDtype::kF32, 0.0f);
        if (e != cudaSuccess) return fail(e, "head");
        session.kv_rows += 1;

        if (step < prompt_steps - 1) {
            cur = prefill[static_cast<size_t>(step) + 1];
            continue;
        }

        // Cooperative cancel: a request aborted before its first decoded step
        // publishes no tokens at all.
        if (gs && gs->Aborted()) {
            aborted = true;
            break;
        }

        // ── sample ──
        e = sampler.Sample(stream, d_logits, V, sample::LogitType::kF32,
                           d_argmax, d_maxv);
        if (e != cudaSuccess) return fail(e, "argmax");
        int32_t tok = -1;
        float maxv = 0.0f;
        cudaMemcpyAsync(&tok, d_argmax, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&maxv, d_maxv, sizeof(float), cudaMemcpyDeviceToHost,
                        stream);
        cudaEventRecord(tok_ready, stream);
        cudaEventSynchronize(tok_ready);
        if (generated == 0) {
            first_maxv = maxv;
            cudaEventRecord(t_ttft, stream);
        }
        tokens.push_back(tok);
        // Diagnostic (LSX_HY_DEBUG=1): prompt ids + top-5 logits of the first
        // decoded step, for the host-reference oracle. stderr only.
        if (generated == 0 && std::getenv("LSX_HY_DEBUG")) {
            std::vector<float> hl(static_cast<size_t>(V));
            cudaMemcpyAsync(hl.data(), d_logits,
                            static_cast<size_t>(V) * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(tok_ready, stream);
            cudaEventSynchronize(tok_ready);
            std::vector<int> top(5, -1);
            std::vector<float> topv(5, -1e30f);
            for (int i = 0; i < V; ++i) {
                const float v = hl[static_cast<size_t>(i)];
                for (int k = 0; k < 5; ++k)
                    if (v > topv[static_cast<size_t>(k)]) {
                        for (int j = 4; j > k; --j) {
                            topv[static_cast<size_t>(j)] =
                                topv[static_cast<size_t>(j - 1)];
                            top[static_cast<size_t>(j)] =
                                top[static_cast<size_t>(j - 1)];
                        }
                        topv[static_cast<size_t>(k)] = v;
                        top[static_cast<size_t>(k)] = i;
                        break;
                    }
            }
            std::string pids;
            for (size_t i = 0; i < prefill.size(); ++i)
                pids += absl::StrCat(i ? "," : "", prefill[i]);
            std::string t5;
            for (int k = 0; k < 5; ++k)
                t5 += absl::StrCat(k ? "," : "", top[static_cast<size_t>(k)], ":",
                                   topv[static_cast<size_t>(k)]);
            fprintf(stderr, "[hy-debug] prompt_ids=%s\n", pids.c_str());
            fprintf(stderr, "[hy-debug] top5=%s\n", t5.c_str());
        }
        if (gs) {
            gs->Emit(static_cast<std::uint32_t>(tok),
                     static_cast<std::uint32_t>(generated), maxv);
            if (gs->Aborted()) {
                aborted = true;
                break;
            }
        }
        ++generated;
        if (!ignore_eos && eos_id >= 0 && tok == eos_id) {
            eos_terminated = true;
            break;
        }
        cur = tok;
    }

    // ── publish session state and drain results ──
    if (!tokens.empty()) session.last_tok = tokens.back();
    session.gen_tail = tokens;
    {
        std::vector<float> hlast(static_cast<size_t>(V));
        cudaMemcpyAsync(hlast.data(), d_logits,
                        static_cast<size_t>(V) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaError_t le = cudaGetLastError();
        cudaEventRecord(t_stop, stream);
        if (cudaEventSynchronize(t_stop) != cudaSuccess)
            return fail(cudaGetLastError(), "logits fence");
        if (le != cudaSuccess) return fail(le, "logits copy");
        out.final_logits = std::move(hlast);
    }
    float ms_total = 0.0f, ms_ttft = 0.0f;
    cudaEventElapsedTime(&ms_total, t_start, t_stop);
    ms_ttft = ms_total;
    if (!tokens.empty())
        (void)cudaEventElapsedTime(&ms_ttft, t_start, t_ttft);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_ttft);
    cudaEventDestroy(t_stop);
    cudaEventDestroy(tok_ready);

    out.gen_tokens = tokens;
    out.ttft_ms = static_cast<double>(ms_ttft);
    out.total_time_ms = static_cast<double>(ms_total);
    out.first_gen_tok_id = tokens.empty() ? -1 : tokens.front();
    out.first_gen_maxv = first_maxv;
    out.eos_terminated = eos_terminated;
    out.ok = true;
    out.weight_resident_bytes =
        pw.resident_bytes + scratch.arena_bytes + session.Bytes();
    out.num_tokens = static_cast<int>(tokens.size());
    if (out.total_time_ms > 0.0)
        out.tokens_per_sec =
            static_cast<double>(out.num_tokens) / (out.total_time_ms / 1000.0);
    out.kv_slots_used = static_cast<int>(session.kv_rows);
    (void)aborted;
    return out;
}

} // namespace lsxcommon::hy
