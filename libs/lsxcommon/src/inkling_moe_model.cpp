#include "inkling_moe_model.h"

#include "lsxcommon/inkling_moe_inference_pipeline.h"
#include "lsxcommon/inkling_moe_load.h"
#include "lsxcommon/inkling_moe_tool_format.h"
#include "lsxcommon/prompt_bridge.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"

#include <absl/strings/str_format.h>

#include <cstdlib>
#include <fstream>
#include <utility>

namespace lsxcommon {

const std::string InklingMoeModel::kType = "inkling-2-small";

namespace {

int EnvInt(const char *name, int dflt)
{
    if (const char *v = std::getenv(name))
        if (const int parsed = std::atoi(v); parsed > 0) return parsed;
    return dflt;
}

}  // namespace

InklingMoeModel::InklingMoeModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
}

InklingMoeModel::~InklingMoeModel() { Unload(); }

bool InklingMoeModel::Load()
{
    if (loaded_) return true;
    tokenizer_optional_ = SyntheticGeometryAllowed();
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    auto st = inkling::LoadInklingTopology(cfg_.model_path, top_);
    if (!st.ok()) {
        log::error(absl::StrFormat("[inkling] topology load failed: %s",
                                   st.ToString()));
        return AbortLoad();
    }
    if (!inkling::InklingTopologyMatchesFrozen(top_)) {
        log::error("[inkling] frozen geometry mismatch");
        return AbortLoad();
    }
    if (!EnsureContext()) {
        log::error("[inkling] cuda context init failed");
        return AbortLoad();
    }
    cudaStreamSynchronize(stream());

    // The loader sizes every plane from `pw.top`, so the topology must be in
    // place before it runs (it was the one field the first cut set too late:
    // the layer vector came back empty and Forward indexed into nothing).
    pw_.top = top_;
    // The dense plane is device-resident for the whole session; the 594B routed
    // set is only ever an mmap view of the neutral ExpertStore.
    if (!inkling::LoadInklingPinnedWeights(cfg_.model_path, pw_, stream())) {
        log::error("[inkling] pinned weight load failed");
        return AbortLoad();
    }
    cudaStreamSynchronize(stream());
    experts_ = std::make_unique<inkling::InklingExpertDb>(
        inkling::LoadInklingExpertIndex(
            cfg_.model_path, top_.num_layers, top_.num_experts,
            inkling::InklingResolveExpertWindow(
                top_, EnvInt("LSX_INKLING_MAX_OPEN", 4096)),
            /*warm_on_load=*/false));

    // Decode scratch is fixed-size; allocating it here means a card that cannot
    // hold the working set fails Load instead of failing the first request.
    if (!scratch_.Allocate(top_, inkling::kMaxContext, stream())) {
        log::error("[inkling] decode scratch allocation failed");
        return AbortLoad();
    }

    weights_loaded_ = true;
    resident_bytes_ = pw_.resident_bytes + static_cast<std::uint64_t>(
                                               scratch_.arena_bytes +
                                               scratch_.expert_stage_bytes);

    // Governor accounting: the pinned dense plane plus the decode scratch are
    // the permanent lease; the routed stores are host mmaps.
    if (!AcquirePermanentWeights(resident_bytes_)) return AbortLoad();
    FinishLoad();
    return true;
}

gpu::InferenceResult InklingMoeModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!weights_loaded_) {
        r.error_msg = "inkling-2-small weights not loaded";
        return r;
    }

    // Model independence: this model's own kernel-routing policy.
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    if (!EnsureContext()) {
        r.error_msg = "cuda init failed";
        return r;
    }
    const int eos_id =
        (req.ignore_eos || !tok_db_) ? kIgnoreEosTokenId : tok_db_->eos_id;

    // Streaming + cooperative cancellation: the whole decode runs inside the
    // residency lock, and the generation stream carries the per-token callback
    // and the abort flag (enforced by test_model_independence_guard).
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    sample::ISampler &sampler = sample::SamplerFor(req.sampling);
    inkling::InklingGenerationResult gen;
    const auto ec = this->RunUnderResidencyLock([&]() {
        return inkling::RunInklingGeneration(
            stream(), pw_, *experts_, session_, scratch_, req.input_ids,
            req.max_gen_tokens, std::vector<int32_t>{eos_id},
            static_cast<float>(top_.final_logit_softcap), top_.unpadded_vocab,
            sampler, gen, &gs);
    });
    if (ec != cudaSuccess || !gen.ok) {
        r.error_msg = gen.error_msg.empty() ? "inkling-2-small cuda error"
                                            : gen.error_msg;
        return r;
    }
    r.ok = true;
    r.num_layers = top_.num_layers;
    r.gen_tokens = gen.tokens;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    r.first_gen_tok_id = gen.first_token;
    r.weight_resident_bytes = resident_bytes_;
    r.final_logits = gen.final_logits;
    r.eos_terminated = gen.eos_terminated;
    r.ttft_ms = gen.ttft_ms;
    r.total_time_ms = gen.total_ms;
    if (gen.total_ms > 0.0)
        r.tokens_per_sec =
            static_cast<double>(gen.tokens.size()) / (gen.total_ms / 1000.0);
    if (tok_db_ && !gen.tokens.empty()) {
        std::size_t cap = gen.tokens.size() * 16 + 64;
        std::string text(cap, '\0');
        if (arrow_tokenizer_decode(*tok_db_, gen.tokens.data(),
                                   gen.tokens.size(), text.data(), &cap) == 0)
            r.output_text.assign(text.data(), cap);
    }
    return r;
}

gpu::InferenceResult InklingMoeModel::SelfCheck(const ModelRequest &req)
{
    runtime::ParisSelfCheckOptions opts;
    opts.frame_prompt = false;
    opts.min_tokens = 2;
    opts.answer_window = 8;
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool InklingMoeModel::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                         std::string_view raw_prompt,
                                         std::vector<int32_t> &out_ids) const
{
    ChatConversation conv;
    conv.messages.push_back({"user", std::string(raw_prompt), std::string(), {}});
    std::vector<InklingSegment> segs;
    if (!RenderInklingConversation(conv, segs)) return false;
    // Without the template markers (a synthetic fixture) fall back to the raw
    // encode rather than emitting an unframed ChatML string.
    if (InklingMarkerId(tok, "<|message_user|>") < 0)
        return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
    return AppendInklingIds(tok, segs, out_ids);
}

bool InklingMoeModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::vector<InklingSegment> segs;
    if (!RenderInklingConversation(conv, segs)) return false;
    if (InklingMarkerId(tok, "<|message_user|>") < 0)
        return BuildPromptInputIds(tok, InklingSegmentsText(segs), out_ids);
    return AppendInklingIds(tok, segs, out_ids);
}

bool InklingMoeModel::ParseToolCalls(const ArrowTokenizerDb &tok,
                                    const std::vector<int32_t> &gen_tokens,
                                    std::vector<ToolCall> &out_calls) const
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;
    // Every marker is special:true, so decode drops it. A tool call is therefore
    // found by SCANNING THE IDS for the invoke marker and decoding only the span
    // after it (see the header note in inkling_moe_tool_format.h).
    const int32_t invoke = InklingMarkerId(tok, "<|content_invoke_tool_json|>");
    const int32_t end_msg = InklingMarkerId(tok, "<|end_message|>");
    if (invoke < 0) {
        // No native markers: fall back to the generic parser over the text.
        std::string text(4096, '\0');
        std::size_t cap = text.size();
        for (int attempt = 0; attempt < 8; ++attempt) {
            const int rc = arrow_tokenizer_decode(tok, gen_tokens.data(),
                                                 gen_tokens.size(), text.data(),
                                                 &cap);
            if (rc == 0) {
                text.resize(cap);
                out_calls = tools::ParseToolCallsText(text);
                return true;
            }
            if (rc == -2 && cap > text.size() && cap <= (1u << 20)) {
                text.resize(cap);
                continue;
            }
            break;
        }
        return false;
    }

    std::size_t i = 0;
    while (i < gen_tokens.size()) {
        if (gen_tokens[i] != invoke) {
            ++i;
            continue;
        }
        std::size_t j = i + 1;
        while (j < gen_tokens.size() && gen_tokens[j] != end_msg &&
               gen_tokens[j] != invoke)
            ++j;
        if (j <= i + 1) {
            i = j;
            continue;
        }
        std::string text((j - i) * 8 + 16, '\0');
        std::size_t cap = text.size();
        if (arrow_tokenizer_decode(tok, gen_tokens.data() + i + 1, j - i - 1,
                                   text.data(), &cap) == 0) {
            text.resize(cap);
            // Re-attach the marker spelling so the shared parser sees the
            // native payload shape it expects.
            ParseInklingToolCalls("<|content_invoke_tool_json|>" + text,
                                  out_calls);
        }
        i = j;
    }
    return true;
}

bool InklingMoeModel::TraceForward(const std::vector<int32_t> &tokens,
                                  const std::string &path)
{
    if (!weights_loaded_ || !experts_) return false;
    if (!EnsureContext()) return false;
    std::vector<float> states;
    std::vector<float> logits;
    const auto ec = inkling::RunInklingForwardTrace(
        stream(), pw_, *experts_, session_, scratch_, tokens,
        static_cast<float>(top_.final_logit_softcap), top_.unpadded_vocab, states,
        &logits);
    if (ec != cudaSuccess) {
        log::error(absl::StrFormat("[inkling] trace failed (cuda %d)",
                                   static_cast<int>(ec)));
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char *>(states.data()),
              static_cast<std::streamsize>(states.size() * sizeof(float)));
    if (!out.good()) return false;
    // The logits row rides along as `<path>.logits`: a parity trace without the
    // row the sampler actually sees cannot be compared against a reference.
    std::ofstream lo(path + ".logits", std::ios::binary);
    if (!lo) return false;
    lo.write(reinterpret_cast<const char *>(logits.data()),
             static_cast<std::streamsize>(logits.size() * sizeof(float)));
    return lo.good();
}

void InklingMoeModel::TeardownWeights()
{
    scratch_.Free();
    session_.Free();
    if (experts_) {
        inkling::FreeInklingExpertIndex(*experts_);
        experts_.reset();
    }
    if (weights_loaded_) {
        inkling::FreeInklingPinnedWeights(pw_);
        weights_loaded_ = false;
    }
    TeardownContext();
}

}  // namespace lsxcommon
