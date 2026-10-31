#include "lsxcommon/step_moe_inference_pipeline.h"

#include "step_moe_constants.h"
#include "step_moe_inference_attn.h"

#include "lsxcommon/log.h"
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

namespace lsxcommon::step {
namespace {

constexpr int kThreads = 256;

inline int Blocks(int64_t n)
{
    return static_cast<int>((n + kThreads - 1) / kThreads);
}

std::string LayerTag(const char *what, int layer)
{
    return absl::StrCat(what, " l", layer);
}

// Clamped-SwiGLU MLP used by the dense prefix and by every MoE layer's gated
// shared expert: out = down(clamp_swiglu(gate(x), up(x))).
cudaError_t ClampedMlp(cudaStream_t stream, const __half *x,
                       const lsx_nvfp4::LsxNvfp4Weight &gate,
                       const lsx_nvfp4::LsxNvfp4Weight &up,
                       const lsx_nvfp4::LsxNvfp4Weight &down, int H, int I,
                       __half *d_yg, __half *d_yu, __half *d_h, __half *out,
                       float limit)
{
    cudaError_t e = lsx_nvfp4::RunLsxNvfp4Gemm(
        stream, x, gate.d_packed, gate.d_scales, gate.global_scale, d_yg, 1, H,
        I, false);
    if (e != cudaSuccess) return e;
    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, x, up.d_packed, up.d_scales,
                                   up.global_scale, d_yu, 1, H, I, false);
    if (e != cudaSuccess) return e;
    e = RunStepSwigluClamp(stream, d_yg, d_yu, d_h, I, limit);
    if (e != cudaSuccess) return e;
    return lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_h, down.d_packed,
                                      down.d_scales, down.global_scale, out, 1,
                                      I, H, false);
}

} // namespace

std::int64_t StepSession::Bytes() const
{
    if (!d_k) return 0;
    return 2 * static_cast<std::int64_t>(geom.Elements()) * sizeof(__half) +
           static_cast<std::int64_t>(gen_tail.capacity()) * 4;
}

void StepSession::Free()
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

void StepSession::Reset()
{
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

cudaError_t StepSession::EnsureCapacity(const StepTopology &top, int64_t want,
                                        cudaStream_t stream)
{
    if (want <= capacity && d_k) return cudaSuccess;
    const int64_t cap = std::max<int64_t>(want, capacity * 2);
    const KvCacheGeometry g = StepKvGeometry(top, cap);
    const int64_t elems = g.Elements();
    __half *nk = nullptr;
    __half *nv = nullptr;
    if (cudaMalloc(&nk, static_cast<size_t>(elems) * sizeof(__half)) !=
        cudaSuccess)
        return cudaGetLastError();
    if (cudaMalloc(&nv, static_cast<size_t>(elems) * sizeof(__half)) !=
        cudaSuccess) {
        cudaFree(nk);
        return cudaGetLastError();
    }
    if (d_k && kv_rows > 0) {
        // Copy only the live rows of each layer's block (kv_rows * pos_stride
        // elements), never kv_rows * LayerStride — the latter would read past
        // the layer block and write past the allocation.
        const size_t bytes = static_cast<size_t>(kv_rows) *
                             static_cast<size_t>(g.pos_stride) * sizeof(__half);
        for (int l = 0; l < static_cast<int>(g.layers); ++l) {
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

void StepScratch::Free()
{
    if (arena) cudaFree(arena);
    arena = nullptr;
    arena_bytes = 0;
    if (d_ids) cudaFree(d_ids);
    if (d_top_ids) cudaFree(d_top_ids);
    if (d_argmax) cudaFree(d_argmax);
    if (d_maxv) cudaFree(d_maxv);
    if (d_slots) cudaFree(d_slots);
    if (h_slots) cudaFreeHost(h_slots);
    if (d_logits) cudaFree(d_logits);
    d_ids = nullptr;
    d_top_ids = nullptr;
    d_argmax = nullptr;
    d_maxv = nullptr;
    d_slots = nullptr;
    h_slots = nullptr;
    d_logits = nullptr;
    logits_elems = 0;
}

gpu::InferenceResult RunStepInference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, StepPinnedWeights &pw, StepSession &session,
    StepScratch &scratch, sample::ISampler &sampler, cudaStream_t stream,
    bool ignore_eos, const GenerationStream *gs)
{
    gpu::InferenceResult out;
    out.num_layers = pw.top.num_layers;
    (void)model_path;

    const StepTopology &top = pw.top;
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
    const float routed_scale = static_cast<float>(top.router_scaling_factor);

    if (H <= 0 || V <= 0 || NH <= 0 || NKV <= 0 || HD <= 0 || L <= 0 || E <= 0 ||
        K <= 0 || K > moe::kMaxTopK || K > 64) {
        out.error_msg = "invalid Step-5 topology";
        return out;
    }
    if (input_ids.empty() || max_gen_tokens <= 0) {
        out.error_msg = "empty prompt or max_gen_tokens <= 0";
        return out;
    }
    if (top.num_experts > 0 && pw.expert_store.records.empty()) {
        out.error_msg = "expert store not loaded";
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
    if (need_cap > frozen::kMaxContextTokens) {
        out.error_msg = absl::StrCat(
            "context ", need_cap, " exceeds the dense-equivalent bound ",
            static_cast<int>(frozen::kMaxContextTokens),
            " tokens: the DSA indexer selection path (>4096 tokens) is not "
            "implemented yet (OPEN_TASKS §R9)");
        return out;
    }
    if (cudaError_t ce = session.EnsureCapacity(top, need_cap, stream);
        ce != cudaSuccess) {
        out.error_msg = absl::StrCat("kv cache alloc failed: ",
                                     cudaGetErrorString(ce));
        return out;
    }

    // ── Scratch ──
    const size_t half_sz[] = {
        static_cast<size_t>(H),        // normed
        static_cast<size_t>(NH * HD),  // q
        static_cast<size_t>(NKV * HD), // k
        static_cast<size_t>(NKV * HD), // v
        static_cast<size_t>(NH),       // gate
        static_cast<size_t>(NH * HD),  // attn
        static_cast<size_t>(H),        // attn out
        static_cast<size_t>(imax),     // yg
        static_cast<size_t>(imax),     // yu
        static_cast<size_t>(imax),     // h
        static_cast<size_t>(H),        // shared
        static_cast<size_t>(H),        // mlp out
        static_cast<size_t>(H),        // final norm (bf16)
    };
    size_t half_bytes = 0;
    for (auto s : half_sz) half_bytes += ((s + 7) / 8) * 8 * sizeof(__half);
    half_bytes = (half_bytes + 15) & ~static_cast<size_t>(15);
    const size_t float_bytes = (((2ull * static_cast<size_t>(H)) * sizeof(float)) +
                                15) & ~static_cast<size_t>(15);
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
        cudaMalloc(&scratch.d_top_ids, sizeof(int32_t) * static_cast<size_t>(K));
        cudaMalloc(&scratch.d_argmax, sizeof(int32_t));
        cudaMalloc(&scratch.d_maxv, sizeof(float));
        cudaMalloc(&scratch.d_slots,
                   sizeof(StepRoutedSlot) * static_cast<size_t>(K));
        cudaMallocHost(&scratch.h_slots,
                       sizeof(StepRoutedSlot) * static_cast<size_t>(K));
    }
    if (scratch.logits_elems < V) {
        if (scratch.d_logits) cudaFree(scratch.d_logits);
        if (cudaMalloc(&scratch.d_logits, static_cast<size_t>(V) * sizeof(float)) !=
            cudaSuccess) {
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
    __half *d_normed = adv(half_sz[0]);
    __half *d_q = adv(half_sz[1]);
    __half *d_kp = adv(half_sz[2]);
    __half *d_vp = adv(half_sz[3]);
    __half *d_gate = adv(half_sz[4]);
    __half *d_attn = adv(half_sz[5]);
    __half *d_attn_out = adv(half_sz[6]);
    __half *d_yg = adv(half_sz[7]);
    __half *d_yu = adv(half_sz[8]);
    __half *d_h = adv(half_sz[9]);
    __half *d_shared = adv(half_sz[10]);
    __half *d_mlp = adv(half_sz[11]);
    __nv_bfloat16 *d_final =
        reinterpret_cast<__nv_bfloat16 *>(adv(half_sz[12]));
    float *d_resid = reinterpret_cast<float *>(
        reinterpret_cast<uint8_t *>(scratch.arena) + half_bytes);
    float *d_moe_acc = d_resid + H;

    int32_t *d_ids = scratch.d_ids;
    StepRoutedSlot *d_slots = scratch.d_slots;
    StepRoutedSlot *h_slots = scratch.h_slots;
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

    const StepAttnScratch as{d_normed, d_q, d_kp, d_vp,
                             d_gate,   d_attn, d_attn_out};

    const bool layer_dump = std::getenv("LSX_STEP_LAYER_DUMP") != nullptr;
    int32_t cur = prefill[0];
    int generated = 0;
    bool aborted = false;

    for (int step = 0; step < total_steps; ++step) {
        const int64_t pos = session.kv_rows;
        cudaMemcpyAsync(d_ids, &cur, sizeof(int32_t), cudaMemcpyHostToDevice,
                        stream);

        cudaError_t e = RunStepEmbedBf16(stream, pw.d_embed, d_ids, d_resid, H);
        if (e != cudaSuccess) return fail(e, "embed");

        for (int l = 0; l < L; ++l) {
            const StepLayerWeights &lw = pw.layers[static_cast<size_t>(l)];
            __half *k_base = session.d_k + l * session.geom.LayerStride();
            __half *v_base = session.d_v + l * session.geom.LayerStride();

            e = RunStepAttentionBlock(stream, lw, top, l, d_resid, as, k_base,
                                      v_base, static_cast<int>(pos));
            if (e != cudaSuccess) return fail(e, LayerTag("attn", l));
            e = RunStepAddHalfToF32(stream, d_attn_out, d_resid, H);
            if (e != cudaSuccess) return fail(e, LayerTag("res1", l));

            e = RunStepNormF32ToHalf(stream, d_resid, lw.d_post_norm, d_normed,
                                     H, eps);
            if (e != cudaSuccess) return fail(e, LayerTag("postnorm", l));

            if (!lw.is_moe) {
                e = ClampedMlp(stream, d_normed, lw.mlp_gate, lw.mlp_up,
                               lw.mlp_down, H, DI, d_yg, d_yu, d_h, d_mlp,
                               frozen::kSwigluLimit);
                if (e != cudaSuccess) return fail(e, LayerTag("mlp", l));
            } else {
                e = RunStepRouter(stream, lw.d_router, lw.d_router_bias,
                                  d_normed, E, K, H, routed_scale, d_slots,
                                  scratch.d_top_ids);
                if (e != cudaSuccess) return fail(e, LayerTag("router", l));
                e = cudaMemcpyAsync(h_slots, d_slots,
                                    sizeof(StepRoutedSlot) *
                                        static_cast<size_t>(K),
                                    cudaMemcpyDeviceToHost, stream);
                if (e != cudaSuccess) return fail(e, LayerTag("router copy", l));
                cudaEventRecord(tok_ready, stream);
                cudaEventSynchronize(tok_ready);

                e = ClampedMlp(stream, d_normed, lw.shared_gate, lw.shared_up,
                               lw.shared_down, H, SI, d_yg, d_yu, d_h, d_shared,
                               frozen::kSwigluLimit);
                if (e != cudaSuccess) return fail(e, LayerTag("shared", l));

                e = cudaMemsetAsync(d_moe_acc, 0,
                                    static_cast<size_t>(H) * sizeof(float),
                                    stream);
                if (e != cudaSuccess) return fail(e, LayerTag("acc zero", l));
                std::string err;
                e = moe::RunMoERoutedFFN(
                    stream, pw.expert_store, pw.expert_cache, l, h_slots,
                    K, d_normed, H, I, d_moe_acc, d_yg, d_yu, d_mlp, err,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    nullptr, frozen::kSwigluLimit);
                if (e != cudaSuccess) {
                    if (err.empty()) err = LayerTag("routed expert", l);
                    return fail(e, err);
                }
                e = RunStepCombineF32(stream, d_moe_acc, d_shared, d_resid, H);
                if (e != cudaSuccess) return fail(e, LayerTag("combine", l));
            }
            if (!lw.is_moe) {
                e = RunStepAddHalfToF32(stream, d_mlp, d_resid, H);
                if (e != cudaSuccess) return fail(e, LayerTag("res2", l));
            }
            if (layer_dump) {
                std::vector<float> h(static_cast<size_t>(H));
                cudaMemcpyAsync(h.data(), d_resid,
                                static_cast<size_t>(H) * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                double ss = 0.0;
                uint64_t hash = 1469598103934665603ull;
                const uint8_t *hb = reinterpret_cast<const uint8_t *>(h.data());
                for (size_t bi = 0; bi < h.size() * sizeof(float); ++bi) {
                    hash ^= hb[bi];
                    hash *= 1099511628211ull;
                }
                for (int i = 0; i < H; ++i) ss += static_cast<double>(h[i]) * h[i];
                std::fprintf(stderr,
                             "[step-layer] step=%d l=%d ss=%.17g hash=%016llx\n",
                             step, l, ss,
                             static_cast<unsigned long long>(hash));
            }
        }

        e = RunStepNormF32ToBf16Narrowed(stream, d_resid, pw.d_final_norm,
                                         d_final, H, eps);
        if (e != cudaSuccess) return fail(e, "final norm");
        e = gemv::RunLxGemmVec(stream, pw.d_head, d_final, 1, V, H,
                               gemv::LxDtype::kBF16, gemv::LxDtype::kBF16, 1.0f,
                               d_logits, gemv::LxDtype::kF32, 0.0f);
        if (e != cudaSuccess) return fail(e, "head");
        session.kv_rows += 1;

        if (step < prompt_steps - 1) {
            cur = prefill[static_cast<size_t>(step) + 1];
            continue;
        }

        if (gs && gs->Aborted()) {
            aborted = true;
            break;
        }

        e = sampler.Sample(stream, d_logits, V, sample::LogitType::kF32,
                           scratch.d_argmax, scratch.d_maxv);
        if (e != cudaSuccess) return fail(e, "argmax");
        int32_t tok = -1;
        float maxv = 0.0f;
        cudaMemcpyAsync(&tok, scratch.d_argmax, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&maxv, scratch.d_maxv, sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(tok_ready, stream);
        cudaEventSynchronize(tok_ready);
        if (generated == 0) {
            first_maxv = maxv;
            cudaEventRecord(t_ttft, stream);
        }
        tokens.push_back(tok);
        if (generated == 0 && std::getenv("LSX_STEP_DEBUG")) {
            std::vector<float> hl(static_cast<size_t>(V));
            cudaMemcpyAsync(hl.data(), d_logits,
                            static_cast<size_t>(V) * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(tok_ready, stream);
            cudaEventSynchronize(tok_ready);
            std::vector<int> top5(5, -1);
            std::vector<float> topv(5, -1e30f);
            for (int i = 0; i < V; ++i) {
                const float v = hl[static_cast<size_t>(i)];
                for (int k = 0; k < 5; ++k)
                    if (v > topv[static_cast<size_t>(k)]) {
                        for (int j = 4; j > k; --j) {
                            topv[static_cast<size_t>(j)] =
                                topv[static_cast<size_t>(j - 1)];
                            top5[static_cast<size_t>(j)] =
                                top5[static_cast<size_t>(j - 1)];
                        }
                        topv[static_cast<size_t>(k)] = v;
                        top5[static_cast<size_t>(k)] = i;
                        break;
                    }
            }
            std::string pids;
            for (size_t i = 0; i < prefill.size(); ++i)
                pids += absl::StrCat(i ? "," : "", prefill[i]);
            std::string t5;
            for (int k = 0; k < 5; ++k)
                t5 += absl::StrCat(k ? "," : "", top5[static_cast<size_t>(k)],
                                   ":", topv[static_cast<size_t>(k)]);
            std::fprintf(stderr, "[step-debug] prompt_ids=%s\n", pids.c_str());
            std::fprintf(stderr, "[step-debug] top5=%s\n", t5.c_str());
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
        if (!ignore_eos && ((top.eos_token_id >= 0 && tok == top.eos_token_id) ||
                            (top.eos_token_id2 >= 0 && tok == top.eos_token_id2))) {
            eos_terminated = true;
            break;
        }
        cur = tok;
    }

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
        pw.resident_bytes + static_cast<std::uint64_t>(pw.expert_cache_bytes) +
        scratch.arena_bytes + session.Bytes();
    out.num_tokens = static_cast<int>(tokens.size());
    if (out.total_time_ms > 0.0)
        out.tokens_per_sec =
            static_cast<double>(out.num_tokens) / (out.total_time_ms / 1000.0);
    out.kv_slots_used = static_cast<int>(session.kv_rows);
    (void)aborted;
    return out;
}

} // namespace lsxcommon::step
