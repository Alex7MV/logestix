#include "kimi_moe_k3_model.h"

#include "lsxcommon/kimi_moe_k3_tool_format.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_format.h>

#include <cstdlib>
#include <utility>

namespace lsxcommon {

const std::string KimiK3Model::kType = "kimi-k3";

namespace {

int EnvInt(const char *name, int dflt)
{
    if (const char *v = std::getenv(name))
        if (const int parsed = std::atoi(v); parsed > 0) return parsed;
    return dflt;
}

}  // namespace

KimiK3Model::KimiK3Model(ModelInitConfig cfg) : ModelBase(std::move(cfg)) {}

KimiK3Model::~KimiK3Model() { Unload(); }

bool KimiK3Model::Load()
{
    if (loaded_) return true;
    tokenizer_optional_ = SyntheticGeometryAllowed();
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    auto st = kimi_k3::LoadKimiK3Topology(cfg_.model_path, top_);
    if (!st.ok()) {
        log::error(absl::StrFormat("[kimi-k3] topology load failed: %s",
                                   st.ToString()));
        return AbortLoad();
    }
    if (!SyntheticGeometryAllowed() &&
        !kimi_k3::KimiK3TopologyMatchesFrozen(top_)) {
        log::error("[kimi-k3] frozen geometry mismatch");
        return AbortLoad();
    }

    if (!EnsureContext()) {
        log::error("[kimi-k3] cuda context init failed");
        return AbortLoad();
    }
    cudaStreamSynchronize(stream());
    if (!kimi_k3::LoadKimiK3PinnedWeights(cfg_.model_path, pw_, stream())) {
        log::error("[kimi-k3] pinned weight load failed");
        return AbortLoad();
    }
    cudaStreamSynchronize(stream());
    experts_ = std::make_unique<kimi_k3::KimiK3ExpertDb>(
        kimi_k3::LoadKimiK3ExpertIndex(
            cfg_.model_path, top_.num_layers, top_.num_experts,
            top_.dense_layers,
            kimi_k3::KimiK3ResolveExpertWindow(
                top_, EnvInt("LSX_KIMI_K3_MAX_OPEN", 65536)),
            /*warm_on_load=*/false));
    weights_loaded_ = true;
    resident_bytes_ = pw_.resident_bytes;

    // Governor accounting: the pinned dense set + the streamed plane's device
    // staging are the permanent lease; the routed stores are host mmaps.
    if (!AcquirePermanentWeights(resident_bytes_)) return AbortLoad();
    FinishLoad();
    return true;
}

gpu::InferenceResult KimiK3Model::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!weights_loaded_) {
        r.error_msg = "kimi-k3 weights not loaded";
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

    // Streaming + cooperative cancellation.
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    sample::ISampler &sampler = sample::SamplerFor(req.sampling);
    kimi_k3::KimiK3GenerationResult gen;
    const int max_context = EnvInt("LSX_KIMI_K3_MAX_CONTEXT",
                                   kimi_k3::kDefaultMaxContext);
    auto ec = RunUnderResidencyLock([&]() {
        return kimi_k3::RunKimiK3Generation(
            stream(), pw_, *experts_, req.input_ids, req.max_gen_tokens,
            eos_id, max_context, sampler, gen, &gs);
    });
    if (ec != cudaSuccess || !gen.ok) {
        r.error_msg = gen.error_msg.empty() ? "kimi-k3 cuda error"
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

gpu::InferenceResult KimiK3Model::SelfCheck(const ModelRequest &req)
{
    // Chat model: the control prompt is framed, and the answer is the first
    // decoded token(s).
    // Chat model: the control prompt is XTML-framed, and K3 answers inside its
    // thinking channel, so the window covers the reasoning prefix as well.
    runtime::ParisSelfCheckOptions opts;
    opts.frame_prompt = true;
    opts.min_tokens = 64;
    opts.answer_window = 64;
    opts.substring_match = true;
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool KimiK3Model::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                     std::string_view raw_prompt,
                                     std::vector<int32_t> &out_ids) const
{
    ChatConversation conv;
    conv.messages.push_back(
        {"user", std::string(raw_prompt), std::string(), {}});
    std::vector<KimiK3Segment> segs;
    if (!RenderKimiK3Conversation(conv, segs)) return false;
    if (arrow_tokenizer_lookup(tok, "<|open|>", 8) < 0)
        return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
    return AppendKimiK3Ids(tok, segs, out_ids);
}

bool KimiK3Model::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::vector<KimiK3Segment> segs;
    if (!RenderKimiK3Conversation(conv, segs)) return false;
    if (arrow_tokenizer_lookup(tok, "<|open|>", 8) < 0)
        return BuildPromptInputIds(tok, KimiK3SegmentsText(segs), out_ids);
    return AppendKimiK3Ids(tok, segs, out_ids);
}

bool KimiK3Model::ParseToolCalls(const ArrowTokenizerDb &tok,
                                 const std::vector<int32_t> &gen_tokens,
                                 std::vector<ToolCall> &out_calls) const
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;
    std::string text(4096, '\0');
    std::size_t cap = text.size();
    for (int attempt = 0; attempt < 8; ++attempt) {
        const int rc = arrow_tokenizer_decode(
            tok, gen_tokens.data(), gen_tokens.size(), text.data(), &cap);
        if (rc == 0) {
            text.resize(cap);
            out_calls = ParseKimiK3ToolCalls(text);
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

void KimiK3Model::TeardownWeights()
{
    if (experts_) {
        kimi_k3::FreeKimiK3ExpertIndex(*experts_);
        experts_.reset();
    }
    if (weights_loaded_) {
        kimi_k3::FreeKimiK3PinnedWeights(pw_);
        weights_loaded_ = false;
    }
    TeardownContext();
}

}  // namespace lsxcommon
