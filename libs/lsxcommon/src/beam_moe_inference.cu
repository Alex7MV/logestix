// Beam-1.0 generation driver: token-by-token prefill and decode over the
// neutral NVFP4 GEMV, the neutral mmap expert store and the device slot LRU.
//
// Correctness-first: one token per step, no CUDA graph, and a per-MoE-layer
// router readback to resolve the streamed expert slots on the host (the
// documented perf debt; see docs/OPEN_TASKS.md).
#include "lsxcommon/beam_moe_inference.h"

#include "beam_moe_constants.h"

#include "lsxcommon/beam_moe_kernels.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/model_runtime.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lsxcommon::beam {
namespace {

using namespace frozen;

constexpr int kMaxTopKBeam = 16;

int64_t Align8(int64_t v) { return (v + 7) / 8 * 8; }

int ExpertCacheSlots()
{
    int slots = 512;
    if (const char *env = std::getenv("LSX_BEAM_EXPERT_CACHE_SLOTS")) {
        const long v = std::atol(env);
        if (v > 0) slots = static_cast<int>(v);
    }
    return slots;
}

std::string LayerTag(const char *what, int l)
{
    return absl::StrCat(what, " layer ", l);
}

} // namespace

std::int64_t BeamSession::Bytes() const
{
    if (!allocated()) return 0;
    return static_cast<std::int64_t>(2) *
           static_cast<std::int64_t>(geom.Elements()) * sizeof(__half);
}

void BeamSession::Free()
{
    if (d_kv_k) cudaFree(d_kv_k);
    if (d_kv_v) cudaFree(d_kv_v);
    d_kv_k = nullptr;
    d_kv_v = nullptr;
    capacity = 0;
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

void BeamSession::Reset()
{
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

cudaError_t BeamSession::EnsureCapacity(const BeamTopology &top,
                                        int64_t capacity_in,
                                        cudaStream_t /*stream*/)
{
    if (capacity_in <= capacity && allocated()) return cudaSuccess;
    const int64_t cap = std::max<int64_t>(capacity_in, capacity);
    KvCacheGeometry g = BeamKvGeometry(top, cap);
    __half *nk = nullptr, *nv = nullptr;
    const size_t elems = static_cast<size_t>(g.Elements());
    if (cudaMalloc(reinterpret_cast<void **>(&nk), elems * sizeof(__half)) !=
        cudaSuccess)
        return cudaErrorMemoryAllocation;
    if (cudaMalloc(reinterpret_cast<void **>(&nv), elems * sizeof(__half)) !=
        cudaSuccess) {
        cudaFree(nk);
        return cudaErrorMemoryAllocation;
    }
    if (d_kv_k) cudaFree(d_kv_k);
    if (d_kv_v) cudaFree(d_kv_v);
    d_kv_k = nk;
    d_kv_v = nv;
    capacity = cap;
    geom = g;
    // A grown cache keeps its rows; a fresh one starts empty.
    return cudaGetLastError();
}

void BeamScratch::Free()
{
    if (arena) cudaFree(arena);
    arena = nullptr;
    arena_bytes = 0;
    if (d_logits) cudaFree(d_logits);
    d_logits = nullptr;
    logits_elems = 0;
    if (d_ids) cudaFree(d_ids);
    if (d_argmax) cudaFree(d_argmax);
    if (d_maxv) cudaFree(d_maxv);
    d_ids = nullptr;
    d_argmax = nullptr;
    d_maxv = nullptr;
    if (d_slots) cudaFree(d_slots);
    d_slots = nullptr;
    if (h_slots) cudaFreeHost(h_slots);
    h_slots = nullptr;
    moe::FreeMoEExpertCache(cache);
}

gpu::InferenceResult RunBeamInference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, const std::vector<int32_t> &eos_ids,
    BeamPinnedWeights &pw, BeamSession &session, BeamScratch &scratch,
    sample::ISampler &sampler, cudaStream_t stream, bool ignore_eos,
    const GenerationStream *gs)
{
    (void)model_path;
    gpu::InferenceResult out;
    const BeamTopology &top = pw.top;
    if (input_ids.empty()) {
        out.error_msg = "empty prompt";
        return out;
    }
    const int L = top.num_layers;
    const int H = top.hidden_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;
    const int V = top.vocab_size;
    const int E = top.num_experts;
    const int K = top.num_experts_per_tok;
    const int I = top.moe_intermediate_size;
    const int DI = top.dense_intermediate_size;
    const float eps = static_cast<float>(top.rms_norm_eps);
    const float theta = static_cast<float>(top.rope_theta);

    // Session continuation: an input equal to the last generated tail resumes
    // from the cached last token; anything else starts a fresh session.
    const bool continuation = session.allocated() && !session.gen_tail.empty() &&
                              input_ids == session.gen_tail;
    std::vector<int32_t> prefill =
        continuation ? std::vector<int32_t>{session.last_tok} : input_ids;
    if (!continuation) {
        session.Reset();
        cudaError_t e = session.EnsureCapacity(
            top, static_cast<int64_t>(prefill.size()) + max_gen_tokens + 8,
            stream);
        if (e != cudaSuccess) {
            out.error_msg = "kv alloc failed";
            return out;
        }
    } else if (session.capacity < session.kv_rows + max_gen_tokens + 8) {
        cudaError_t e = session.EnsureCapacity(
            top, session.kv_rows + max_gen_tokens + 8, stream);
        if (e != cudaSuccess) {
            out.error_msg = "kv grow failed";
            return out;
        }
    }
    if (scratch.cache.slots <= 0) {
        std::string error;
        if (!moe::AllocateMoEExpertCache(pw.expert_store,
                                         ExpertCacheSlots(), scratch.cache,
                                         error)) {
            out.error_msg = "expert cache alloc failed: " + error;
            return out;
        }
    }

    // ── Scratch arena (half buffers + the FP32 MoE accumulator) ──
    const int64_t half_sz[] = {
        H, H, H, H, H,                     // x, normed, post, proj, branch
        NH * HD, NKV * HD, NKV * HD, NH * HD, NH * HD,  // q,k,v,gate,attn
        DI, DI, DI,                        // dense MLP gate/up/act
        std::max(I, DI), std::max(I, DI), H,  // yg, yu, y2
    };
    size_t half_bytes = 0;
    for (int64_t s : half_sz) half_bytes += static_cast<size_t>(Align8(s)) * sizeof(__half);
    half_bytes = (half_bytes + 15) & ~static_cast<size_t>(15);
    size_t float_bytes = static_cast<size_t>(H) * sizeof(float);
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
                   sizeof(BeamRoutedSlot) * static_cast<size_t>(K));
        cudaMallocHost(&scratch.h_slots,
                       sizeof(BeamRoutedSlot) * static_cast<size_t>(K));
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
    auto adv = [&](int64_t elems) {
        __half *r = ph;
        ph += Align8(elems);
        return r;
    };
    __half *d_x = adv(half_sz[0]);
    __half *d_normed = adv(half_sz[1]);
    __half *d_post = adv(half_sz[2]);
    __half *d_proj = adv(half_sz[3]);
    __half *d_branch = adv(half_sz[4]);
    __half *d_q = adv(half_sz[5]);
    __half *d_k = adv(half_sz[6]);
    __half *d_v = adv(half_sz[7]);
    __half *d_gate = adv(half_sz[8]);
    __half *d_attn = adv(half_sz[9]);
    __half *d_mlp_gate = adv(half_sz[10]);
    __half *d_mlp_up = adv(half_sz[11]);
    __half *d_mlp_act = adv(half_sz[12]);
    __half *d_yg = adv(half_sz[13]);
    __half *d_yu = adv(half_sz[14]);
    __half *d_y2 = adv(half_sz[15]);
    float *d_moe_acc = reinterpret_cast<float *>(
        reinterpret_cast<uint8_t *>(scratch.arena) + half_bytes);

    int32_t *d_ids = scratch.d_ids;
    int32_t *d_argmax = scratch.d_argmax;
    float *d_maxv = scratch.d_maxv;
    BeamRoutedSlot *d_slots = scratch.d_slots;
    BeamRoutedSlot *h_slots = scratch.h_slots;
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
        out.error_msg = absl::StrCat(msg, " @pos=", session.kv_rows);
        if (err != cudaSuccess)
            out.error_msg += absl::StrCat(" (", cudaGetErrorString(err), ")");
        if (t_start) cudaEventDestroy(t_start);
        if (t_ttft) cudaEventDestroy(t_ttft);
        if (t_stop) cudaEventDestroy(t_stop);
        if (tok_ready) cudaEventDestroy(tok_ready);
        return out;
    };

    using lsx_nvfp4::LaunchNvfp4Gemm;
    const int prompt_steps = static_cast<int>(prefill.size());
    const int total_steps = prompt_steps - 1 + max_gen_tokens;
    int32_t cur = prefill[0];
    int generated = 0;
    bool aborted = false;

    for (int step = 0; step < total_steps; ++step) {
        const int64_t pos = session.kv_rows;
        cudaMemcpyAsync(d_ids, &cur, sizeof(int32_t), cudaMemcpyHostToDevice,
                        stream);
        cudaError_t e = RunBeamEmbed(stream, pw.d_embed, d_ids, d_x, 1, H);
        if (e != cudaSuccess) return fail(e, "embed");

        for (int l = 0; l < L; ++l) {
            const BeamLayerWeights &lw = pw.layers[static_cast<size_t>(l)];
            const bool full = IsFullAttn(l);
            const int window = full ? 0 : kWindow;

            e = RunBeamNorm(stream, d_x, lw.attn.d_input_norm, d_normed, 1, H,
                            eps);
            if (e != cudaSuccess) return fail(e, LayerTag("norm", l));
            e = LaunchNvfp4Gemm(lw.attn.q, d_normed, 1, d_q, stream);
            if (e == cudaSuccess)
                e = LaunchNvfp4Gemm(lw.attn.k, d_normed, 1, d_k, stream);
            if (e == cudaSuccess)
                e = LaunchNvfp4Gemm(lw.attn.v, d_normed, 1, d_v, stream);
            if (e == cudaSuccess)
                e = LaunchNvfp4Gemm(lw.attn.gate, d_normed, 1, d_gate, stream);
            if (e != cudaSuccess) return fail(e, LayerTag("qkv-gate", l));

            e = RunBeamHeadNorm(stream, d_q, lw.attn.d_q_norm, d_q, 1, NH, HD,
                                eps);
            if (e == cudaSuccess)
                e = RunBeamHeadNorm(stream, d_k, lw.attn.d_k_norm, d_k, 1, NKV,
                                    HD, eps);
            if (e != cudaSuccess) return fail(e, LayerTag("qknorm", l));

            if (full) {
                e = RunBeamSsmScale(stream, d_q, NH, HD, static_cast<int>(pos));
            } else {
                e = RunBeamRope(stream, d_q, d_k, 1, NH, NKV, HD, theta,
                                static_cast<int>(pos));
            }
            if (e != cudaSuccess) return fail(e, LayerTag("rope", l));

            // Append this token's k/v to the layer's pos-major cache slice.
            const int64_t koff = session.geom.Offset(l, 0, pos, 0);
            const size_t kv_bytes =
                static_cast<size_t>(NKV) * HD * sizeof(__half);
            e = cudaMemcpyAsync(session.d_kv_k + koff, d_k, kv_bytes,
                                cudaMemcpyDeviceToDevice, stream);
            if (e == cudaSuccess)
                e = cudaMemcpyAsync(session.d_kv_v + koff, d_v, kv_bytes,
                                    cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) return fail(e, LayerTag("kv-append", l));

            __half *k_base = session.d_kv_k + session.geom.LayerStride() * l;
            __half *v_base = session.d_kv_v + session.geom.LayerStride() * l;
            e = RunBeamAttnDecode(stream, d_q, k_base, v_base, d_gate,
                                  lw.attn.d_gate_bias, d_attn, NH, NKV, HD,
                                  static_cast<int>(pos), window, kAttnScale);
            if (e != cudaSuccess) return fail(e, LayerTag("attn", l));

            e = LaunchNvfp4Gemm(lw.attn.o, d_attn, 1, d_proj, stream);
            if (e != cudaSuccess) return fail(e, LayerTag("o", l));
            e = RunBeamNorm(stream, d_proj, lw.attn.d_post_attn_norm, d_post, 1,
                            H, eps);
            if (e == cudaSuccess)
                e = RunBeamDepthResidual(stream, d_post, d_x, d_x, H,
                                         2 * (l + 1));
            if (e != cudaSuccess) return fail(e, LayerTag("attn-residual", l));

            e = RunBeamNorm(stream, d_x, lw.attn.d_pre_ffn_norm, d_normed, 1, H,
                            eps);
            if (e != cudaSuccess) return fail(e, LayerTag("pre-ffn-norm", l));

            if (lw.is_moe) {
                e = RunBeamRouter(stream, lw.d_router, lw.d_router_bias,
                                  d_normed, 1, E, K, H, top.n_group,
                                  top.topk_group, kRoutedScale, d_slots);
                if (e != cudaSuccess) return fail(e, LayerTag("router", l));
                // Host handshake: the streamed expert store resolves slots on
                // the host (correctness-first; documented perf debt).
                e = cudaMemcpyAsync(h_slots, d_slots,
                                    sizeof(BeamRoutedSlot) * K,
                                    cudaMemcpyDeviceToHost, stream);
                if (e != cudaSuccess) return fail(e, "router readback");
                if (cudaStreamSynchronize(stream) != cudaSuccess)
                    return fail(cudaGetLastError(), "router readback fence");
                e = cudaMemsetAsync(d_moe_acc, 0,
                                    static_cast<size_t>(H) * sizeof(float),
                                    stream);
                if (e != cudaSuccess) return fail(e, "moe acc init");
                std::string error;
                e = moe::RunMoERoutedFFN(
                    stream, pw.expert_store, scratch.cache, l,
                    reinterpret_cast<const moe::MoERoutedSlot *>(h_slots), K,
                    d_normed, H, I, d_moe_acc, d_yg, d_yu, d_y2, error);
                if (e != cudaSuccess) {
                    if (error.empty()) error = LayerTag("routed ffn", l);
                    return fail(e, error);
                }
                e = moe::RunMoEAccToFp16(stream, d_moe_acc, d_branch, H);
                if (e != cudaSuccess) return fail(e, LayerTag("moe fold", l));
            } else {
                e = LaunchNvfp4Gemm(lw.mlp_gate, d_normed, 1, d_mlp_gate,
                                    stream);
                if (e == cudaSuccess)
                    e = LaunchNvfp4Gemm(lw.mlp_up, d_normed, 1, d_mlp_up,
                                        stream);
                if (e == cudaSuccess)
                    e = RunBeamSwiGLU(stream, d_mlp_gate, d_mlp_up, d_mlp_act,
                                      DI);
                if (e == cudaSuccess)
                    e = LaunchNvfp4Gemm(lw.mlp_down, d_mlp_act, 1, d_branch,
                                        stream);
                if (e != cudaSuccess) return fail(e, LayerTag("dense mlp", l));
            }

            e = RunBeamNorm(stream, d_branch, lw.attn.d_post_ffn_norm, d_post,
                            1, H, eps);
            if (e == cudaSuccess)
                e = RunBeamDepthResidual(stream, d_post, d_x, d_x, H,
                                         2 * (l + 1) + 1);
            if (e != cudaSuccess) return fail(e, LayerTag("mlp-residual", l));
        }

        e = RunBeamNorm(stream, d_x, pw.d_final_norm, d_normed, 1, H, eps);
        if (e != cudaSuccess) return fail(e, "final norm");
        e = gemv::RunLxGemmVec(stream, pw.d_head, d_normed, 1, V, H,
                               gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
                               d_logits, gemv::LxDtype::kF32, 0.0f);
        if (e != cudaSuccess) return fail(e, "head");
        e = RunBeamSoftcap(stream, d_logits, V);
        if (e != cudaSuccess) return fail(e, "softcap");
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
                           d_argmax, d_maxv);
        if (e != cudaSuccess) return fail(e, "sample");
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
        if (gs)
            gs->Emit(static_cast<std::uint32_t>(tok),
                     static_cast<std::uint32_t>(generated), maxv);
        ++generated;
        session.last_tok = tok;
        cur = tok;
        if (!ignore_eos) {
            bool is_eos = false;
            for (int id : eos_ids)
                if (id == tok) is_eos = true;
            if (is_eos) {
                eos_terminated = true;
                break;
            }
        }
    }

    cudaEventRecord(t_stop, stream);
    if (generated == 0) cudaEventRecord(t_ttft, stream);
    cudaEventSynchronize(t_stop);
    float ttft_ms = 0.0f, total_ms = 0.0f;
    cudaEventElapsedTime(&ttft_ms, t_start, t_ttft);
    cudaEventElapsedTime(&total_ms, t_start, t_stop);

    // The last decoded step's logits row is the determinism probe.
    std::vector<float> hlast(static_cast<size_t>(V));
    cudaMemcpyAsync(hlast.data(), d_logits, hlast.size() * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaEventDestroy(t_start);
    cudaEventDestroy(t_ttft);
    cudaEventDestroy(t_stop);
    cudaEventDestroy(tok_ready);

    out.ok = true;
    out.gen_tokens = std::move(tokens);
    out.final_logits = std::move(hlast);
    out.ttft_ms = ttft_ms;
    out.total_time_ms = total_ms;
    out.num_tokens = generated;
    out.num_layers = L;
    out.kv_slots_used = static_cast<int32_t>(session.kv_rows);
    out.eos_terminated = eos_terminated;
    out.tokens_per_sec =
        total_ms > 0.0
            ? static_cast<double>(generated) / (static_cast<double>(total_ms) / 1000.0)
            : 0.0;
    out.first_gen_tok_id = out.gen_tokens.empty() ? -1 : out.gen_tokens[0];
    out.first_gen_maxv = first_maxv;
    out.weight_resident_bytes =
        pw.resident_bytes + scratch.arena_bytes +
        static_cast<std::uint64_t>(session.Bytes()) +
        static_cast<std::uint64_t>(scratch.cache.DeviceBytes());
    (void)aborted;
    return out;
}

} // namespace lsxcommon::beam
