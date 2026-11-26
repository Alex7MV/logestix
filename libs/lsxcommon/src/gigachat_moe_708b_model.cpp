// GigaChat4-708B-A41B `IModel` (Model Ingestion Protocol stage 3/5).
//
// This TU is the member's whole lifecycle: Load (index parse + frozen gate +
// device graph + expert cache + pipeline), Infer (one Emit per decoded token,
// abort honoured between steps, teardown-safe), SelfCheck and the stage-4b
// chat/tool bridge. Everything mechanical is the shared family layer; what is
// local is the frozen passport and the 708B geometry handed to it.
#include "lsxcommon/gigachat_moe_708b_model.h"

#include "gigachat_moe_708b_constants.h"

#include "lsxcommon/gigachat_tool_format.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_format.h>

#include <algorithm>
#include <cstdint>

namespace lsxcommon {
namespace {

// Owns the three timing events of one Infer run: every early error return below
// would otherwise leak them.
struct ScopedCudaEvents {
    cudaEvent_t start = nullptr;
    cudaEvent_t first = nullptr;
    cudaEvent_t stop = nullptr;

    ScopedCudaEvents()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&first);
        cudaEventCreate(&stop);
    }
    ~ScopedCudaEvents()
    {
        if (start) cudaEventDestroy(start);
        if (first) cudaEventDestroy(first);
        if (stop) cudaEventDestroy(stop);
    }
    ScopedCudaEvents(const ScopedCudaEvents &) = delete;
    ScopedCudaEvents &operator=(const ScopedCudaEvents &) = delete;
};

// Encodes `s` and appends the ids, dropping the encoder's auto-prepended BOS
// (the framing places exactly one BOS at position 0).
bool AppendEncoded(const ArrowTokenizerDb &tok, std::string_view s,
                   std::vector<int32_t> &out)
{
    for (size_t cap = 64; cap <= 8192; cap *= 2)
    {
        std::vector<int32_t> ids(cap, 0);
        size_t n = cap;
        const int32_t r =
            arrow_tokenizer_encode(tok, s.data(), s.size(), ids.data(), &n);
        if (r == -2) continue;
        if (r <= 0) return false;
        size_t off = 0;
        if (tok.bos_id >= 0 && !ids.empty() && ids[0] == tok.bos_id) off = 1;
        out.insert(out.end(), ids.begin() + static_cast<ptrdiff_t>(off),
                   ids.begin() + r);
        return true;
    }
    return false;
}

} // namespace

const std::string GigaChat4Moe708BModel::kType = gigachat4_708b::frozen::kType;

namespace gigachat4_708b {

gigachat4::Gc4FrozenExpect Gc708BExpect()
{
    using namespace frozen;
    gigachat4::Gc4FrozenExpect e;
    e.format = kFormat;
    e.geo.hidden = kHidden;
    e.geo.layers = kLayers;
    e.geo.vocab = kVocab;
    e.geo.heads = kHeads;
    e.geo.kv_heads = kKVHeads;
    e.geo.num_experts = kNumRouted;
    e.geo.top_k = kTopK;
    e.geo.moe_intermediate = kMoeIntermediate;
    e.geo.dense_intermediate = kDenseIntermediate;
    e.geo.first_k_dense = kFirstKDense;
    e.geo.q_lora = kQLoraRank;
    e.geo.kv_lora = kKvLoraRank;
    e.geo.qk_nope = kQkNopeHeadDim;
    e.geo.qk_rope = kQkRopeHeadDim;
    e.geo.v_head = kVHeadDim;
    e.geo.index_heads = kIndexHeads;
    e.geo.index_head_dim = kIndexHeadDim;
    e.geo.index_topk = kIndexTopk;
    e.geo.rms_eps = static_cast<double>(kRmsEps);
    e.geo.routed_scale = static_cast<double>(kRoutedScale);
    e.geo.swiglu_limit = static_cast<double>(kSwigluLimit);
    e.geo.logit_softcap = static_cast<double>(kLogitSoftcap);
    e.geo.bos_id = kBosId;
    e.geo.eos_id = kEosId;
    e.geo.pad_id = kPadId;
    e.full_attention_layers.assign(kFullAttnLayers,
                                   kFullAttnLayers + kNumFullAttn);
    e.gate_rank = kGateRank;
    return e;
}

} // namespace gigachat4_708b

GigaChat4Moe708BModel::GigaChat4Moe708BModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
    // Synthetic fixtures may omit the tokenizer arrows.
    tokenizer_optional_ = true;
}

GigaChat4Moe708BModel::~GigaChat4Moe708BModel() { Unload(); }

void GigaChat4Moe708BModel::TeardownWeights()
{
    if (d_logits_)
    {
        cudaFree(d_logits_);
        d_logits_ = nullptr;
    }
    pipeline_.Free();
    gigachat::FreeGcExpertCache(expert_cache_);
    weights_.Free();
    TeardownContext();
}

bool GigaChat4Moe708BModel::BuildPipelineDims(gigachat::GcPipelineDims &d) const
{
    using namespace gigachat4_708b::frozen;
    const int hidden = idx_.hidden_size;
    // q_lora / kv_lora / GDN head counts are read from the loaded weights so a
    // synthetic fixture works, and cross-checked against the passport by Load.
    int q_lora = 0, kv_lora = 0;
    const gigachat::GcLayerWeights *gdn = nullptr;
    for (const auto &l : weights_.layers)
    {
        if (l.full_attention && q_lora == 0)
        {
            q_lora = l.q_a.rows;
            kv_lora = l.kv_a.rows - kQkRopeHeadDim;
        }
        if (!l.full_attention && gdn == nullptr) gdn = &l;
    }
    if (q_lora <= 0 || kv_lora <= 0 || gdn == nullptr) return false;
    const int value_dim = gdn->gdn_out_proj.cols;
    const int key_dim = (gdn->conv_dim - value_dim) / 2;

    d.mla.hidden = hidden;
    d.mla.heads = idx_.num_attention_heads;
    d.mla.q_lora = q_lora;
    d.mla.kv_lora = kv_lora;
    d.mla.qk_nope = kQkNopeHeadDim;
    d.mla.rope = kQkRopeHeadDim;
    d.mla.v_head = kVHeadDim;
    d.gdn.hidden = hidden;
    d.gdn.k_heads = key_dim / kLinearKeyHeadDim;
    d.gdn.v_heads = value_dim / kLinearValueHeadDim;
    d.gdn.hk = kLinearKeyHeadDim;
    d.gdn.hv = kLinearValueHeadDim;
    d.gdn.conv_k = gdn->conv_kernel;
    d.vocab = idx_.vocab_size;
    d.num_experts = idx_.num_experts;
    d.top_k = idx_.num_experts_per_tok;
    d.moe_intermediate = idx_.moe_intermediate_size;
    d.first_k_dense = idx_.first_k_dense_replace;
    d.rms_eps = static_cast<float>(idx_.rms_norm_eps);
    d.routed_scale = static_cast<float>(idx_.router_scaling_factor);
    d.swiglu_limit = static_cast<float>(idx_.swiglu_limit);

    // Variant knobs: GigaChat4's GatedNorm is NOT zero-centred, its GDN output
    // norm is the plain-scale sigmoid variant, the logits are soft-capped, and
    // the 16 full-attention layers run the DSA lightning indexer.
    d.norm_w_offset = 0.0f;
    d.gdn_w_offset = 0.0f;
    // The 4.0 GatedNorm computes its gate from the ALREADY SCALED state
    // (s = xn*w, out = s * 2*sigmoid(Wdown@silu(Wup@s))), unlike 3.5's
    // gate-from-xn. Measured on the 37B checkpoint's own
    // post_self_attn_layernorm: the 3.5 form is 36x too large.
    d.norm_gate_from_scaled = true;
    d.logit_softcap = static_cast<float>(idx_.logit_softcap);
    d.attn_scale = static_cast<float>(idx_.attn_scale);
    d.yarn.theta = kRopeTheta;
    d.yarn.factor = kYarnFactor;
    d.yarn.orig = kYarnOrig;
    d.yarn.beta_fast = kYarnBetaFast;
    d.yarn.beta_slow = kYarnBetaSlow;
    d.gdn_sigmoid_gate_scale = kLinearSigmoidGateScale;

    d.index_heads = idx_.index_heads;
    d.index_head_dim = idx_.index_head_dim;
    d.index_topk = idx_.index_topk;
    d.index_rope = idx_.index_rope > 0 ? idx_.index_rope : kIndexRope;
    d.index_onorm_eps = kLinearONormEps;
    d.index_score_scale = kIndexScoreScale;
    return true;
}

bool GigaChat4Moe708BModel::Load()
{
    if (loaded_) return true;
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    const gigachat4::Gc4FrozenExpect expect = gigachat4_708b::Gc708BExpect();
    std::string err;
    if (!gigachat4::LoadGigaChat4IndexWithFormat(cfg_.model_path, expect.format,
                                                 idx_, err))
    {
        log::error(absl::StrFormat("[gigachat4-708b] index error: %s", err));
        return AbortLoad();
    }
    if (!gigachat4::ValidateGigaChat4GeometryAgainst(idx_, expect, err))
    {
        log::error(absl::StrFormat(
            "[gigachat4-708b] frozen geometry mismatch: %s", err));
        return AbortLoad();
    }
    vram_ = idx_.vram;
    if (!EnsureContext()) return AbortLoad();

    if (!gigachat4::LoadGigaChat4WeightsWithExpect(cfg_.model_path, idx_,
                                                   weights_, stream(), expect,
                                                   err))
    {
        log::error(absl::StrFormat("[gigachat4-708b] weight load error: %s",
                                   err));
        weights_.Free();
        return AbortLoad();
    }
    gigachat::GcPipelineDims dims;
    if (!BuildPipelineDims(dims))
    {
        log::error("[gigachat4-708b] cannot derive pipeline dims from weights");
        weights_.Free();
        return AbortLoad();
    }
    if (!gigachat::InitGigaChatExpertCache(
            cfg_.model_path, idx_.num_layers, idx_.first_k_dense_replace,
            idx_.num_experts, idx_.num_experts_per_tok, /*slots_override=*/0,
            stream(), expert_cache_, err))
    {
        log::error(absl::StrFormat("[gigachat4-708b] expert cache error: %s",
                                   err));
        TeardownWeights();
        return AbortLoad();
    }
    // Prompt + generated tokens share the pipeline's KV/state capacity; the
    // canonical 2048-token benchmark needs headroom for the framed prompt.
    // The 708B pays 5.4 GB of MLA KV at this cap on a 32 GB card.
    max_seq_ = 4096;
    if (!pipeline_.Init(weights_, dims, max_seq_, err))
    {
        log::error(absl::StrFormat("[gigachat4-708b] pipeline init error: %s",
                                   err));
        TeardownWeights();
        return AbortLoad();
    }
    if (cudaMalloc(&d_logits_, static_cast<size_t>(idx_.vocab_size) * 4) !=
        cudaSuccess)
    {
        log::error("[gigachat4-708b] logits buffer allocation failed");
        TeardownWeights();
        return AbortLoad();
    }
    resident_bytes_ = weights_.resident_bytes + expert_cache_.cache_bytes;

    if (!AcquirePermanentWeights(weights_.resident_bytes)) return AbortLoad();
    AcquireExpertCacheLease(expert_cache_.cache_bytes, expert_cache_.cache_bytes,
                            expert_cache_.cache_bytes);

    FinishLoad();
    log::info(absl::StrFormat(
        "[gigachat4-708b] loaded %s: %d layers, %d experts, %d indexer layers, "
        "%d expert slots, vocab %d",
        idx_.model.c_str(), idx_.num_layers, idx_.num_experts,
        idx_.index_topk > 0 ? static_cast<int>(idx_.full_attention_layers.size())
                            : 0,
        expert_cache_.device.slots, idx_.vocab_size));
    return true;
}

gpu::InferenceResult GigaChat4Moe708BModel::Infer(const ModelRequest &req)
{
    // Model-independence invariant: each model asserts its OWN fast-math policy
    // per call (the routing toggles are process-global).
    lsx_nvfp4::LsxFastMathScope scope(EffectiveFastMathMode(cfg_));
    return RunUnderResidencyLock([&] {
        gpu::InferenceResult r;
        if (!ValidateInferRequest(req, r)) return r;
        if (max_seq_ > 0 &&
            static_cast<int64_t>(req.input_ids.size()) + req.max_gen_tokens >
                max_seq_)
        {
            r.error_msg = "context length exceeded: prompt + max_gen > " +
                          std::to_string(max_seq_);
            return r;
        }
        lsxcommon::GenerationStream gs;
        gs.on_token = req.token_callback ? &req.token_callback : nullptr;
        gs.abort_flag = req.abort_flag;

        if (pipeline_.Reset(stream()) != cudaSuccess)
        {
            r.error_msg = "cache reset failed";
            return r;
        }

        std::vector<float> logits(static_cast<size_t>(idx_.vocab_size));
        ScopedCudaEvents ev;
        cudaEventRecord(ev.start, stream());

        int pos = 0;
        for (int32_t id : req.input_ids)
        {
            const cudaError_t e = pipeline_.Step(
                weights_, expert_cache_, id, pos, d_logits_, stream());
            if (e != cudaSuccess)
            {
                r.error_msg =
                    std::string("forward failed: ") + cudaGetErrorString(e);
                return r;
            }
            ++pos;
        }
        cudaEventRecord(ev.first, stream());

        const int eos = idx_.eos_token_id;
        auto &sampler = sample::SamplerFor(req.sampling);
        int generated = 0;
        for (int step = 0; step < req.max_gen_tokens; ++step)
        {
            if (cudaMemcpyAsync(logits.data(), d_logits_,
                                logits.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, stream()) !=
                    cudaSuccess ||
                cudaStreamSynchronize(stream()) != cudaSuccess)
            {
                r.error_msg = "logits readback failed";
                return r;
            }
            int next = 0;
            for (size_t i = 1; i < logits.size(); ++i)
                if (logits[i] > logits[static_cast<size_t>(next)])
                    next = static_cast<int>(i);
            if (!sampler.IsGreedy()) {
                float sampled_maxv = 0.0f;
                const cudaError_t se = sample::SampleToHost(
                    stream(), sampler, d_logits_, static_cast<int>(logits.size()),
                    sample::LogitType::kF32, &next, &sampled_maxv);
                if (se != cudaSuccess) {
                    r.error_msg = "sampling failed";
                    return r;
                }
            }
            r.gen_tokens.push_back(next);
            ++generated;
            if (step == 0)
            {
                r.first_gen_tok_id = next;
                r.first_gen_maxv = logits[static_cast<size_t>(next)];
            }
            r.final_logits = logits;
            if (!req.ignore_eos && next == eos)
            {
                r.eos_terminated = true;
                break;
            }
            if (step + 1 < req.max_gen_tokens)
            {
                const cudaError_t e = pipeline_.Step(
                    weights_, expert_cache_, next, pos, d_logits_, stream());
                if (e != cudaSuccess)
                {
                    r.error_msg = std::string("forward failed: ") +
                                  cudaGetErrorString(e);
                    return r;
                }
                ++pos;
            }
            gs.Emit(static_cast<std::uint32_t>(next),
                    static_cast<std::uint32_t>(step),
                    logits[static_cast<size_t>(next)]);
            if (gs.Aborted()) break;
        }
        r.num_tokens = generated;
        r.num_layers = idx_.num_layers;
        r.weight_resident_bytes = resident_bytes_;
        r.kv_slots_used = pos;
        if (tok_db_ && !r.gen_tokens.empty())
            r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
        cudaEventRecord(ev.stop, stream());
        cudaEventSynchronize(ev.stop);
        float ttft = 0.0f, total = 0.0f;
        cudaEventElapsedTime(&ttft, ev.start, ev.first);
        cudaEventElapsedTime(&total, ev.start, ev.stop);
        r.ttft_ms = static_cast<double>(ttft);
        r.total_time_ms = static_cast<double>(total);
        r.tokens_per_sec = total > 0.0f
                               ? static_cast<double>(generated) * 1000.0 /
                                     static_cast<double>(total)
                               : 0.0;
        r.ok = true;
        return r;
    });
}

gpu::InferenceResult GigaChat4Moe708BModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

bool GigaChat4Moe708BModel::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                                std::string_view raw_prompt,
                                                std::vector<int32_t> &out_ids) const
{
    using namespace gigachat4_708b::frozen;
    const int32_t bos = tok.bos_id >= 0 ? tok.bos_id : arrow_tokenizer_lookup(tok, "<s>", 3);
    const int32_t role_sep = arrow_tokenizer_lookup(tok, "<|role_sep|>\n", 13);
    const int32_t msg_sep = arrow_tokenizer_lookup(tok, "<|message_sep|>\n\n", 17);
    const int32_t think = arrow_tokenizer_lookup(tok, "<think>", 7);
    const int32_t think_end = arrow_tokenizer_lookup(tok, "</think>", 8);
    if (bos < 0 || role_sep < 0 || msg_sep < 0 || think < 0 || think_end < 0)
        return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
    (void)kRoleSepId;
    (void)kMessageSepId;
    (void)kThinkId;
    (void)kThinkEndId;

    // chatml v5 single turn + a prefilled empty think, so the control prompt
    // answers directly instead of opening a reasoning body.
    static constexpr std::string_view kSystem =
        "\xD0\xA2\xD1\x8B - GigaChat, \xD0\x98\xD0\x98-\xD0\xB0\xD1\x81\xD1\x81"
        "\xD0\xB8\xD1\x81\xD1\x82\xD0\xB5\xD0\xBD\xD1\x82 \xD0\xA1\xD0\xB1"
        "\xD0\xB5\xD1\x80\xD0\xB0. \xD0\x9E\xD1\x82\xD0\xB2\xD0\xB5\xD1\x87"
        "\xD0\xB0\xD0\xB5\xD1\x88\xD1\x8C \xD0\xBA\xD0\xBE\xD1\x80\xD1\x80"
        "\xD0\xB5\xD0\xBA\xD1\x82\xD0\xBD\xD0\xBE, \xD0\xBF\xD0\xBE\xD0\xBB"
        "\xD0\xB5\xD0\xB7\xD0\xBD\xD0\xBE \xD0\xB8 \xD0\xBF\xD0\xBE \xD0\xB4"
        "\xD0\xB5\xD0\xBB\xD1\x83.";

    out_ids.clear();
    out_ids.push_back(bos);
    auto text = [&](std::string_view s) {
        return AppendEncoded(tok, s, out_ids);
    };
    if (!text("system")) return false;
    out_ids.push_back(role_sep);
    if (!text(kSystem)) return false;
    out_ids.push_back(msg_sep);
    if (!text("user")) return false;
    out_ids.push_back(role_sep);
    if (!text(raw_prompt)) return false;
    out_ids.push_back(msg_sep);
    if (!text("assistant")) return false;
    out_ids.push_back(role_sep);
    out_ids.push_back(think);
    out_ids.push_back(think_end);
    return true;
}

bool GigaChat4Moe708BModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderGigaChatConversation(conv, text)) return false;
    out_ids.clear();
    if (!AppendGigaChatIds(tok, text, out_ids))
        return BuildPromptInputIds(tok, text, out_ids);
    if (tok.bos_id >= 0) out_ids.insert(out_ids.begin(), tok.bos_id);
    return true;
}

bool GigaChat4Moe708BModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return ParseGigaChatTokenCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
