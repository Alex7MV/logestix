#include "inkling_moe_varloop_model.h"

#include "inkling_moe_varloop_constants.h"

#include "lsxcommon/inkling_moe_loop_inference_pipeline.h"
#include "lsxcommon/inkling_moe_loop_load.h"
#include "lsxcommon/inkling_moe_loop_tool_format.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_bridge.h"
#include "lsxcommon/synthetic_geometry.h"

#include <absl/strings/str_format.h>

#include <cstdlib>
#include <utility>

namespace lsxcommon {

namespace frozen = lsxcommon::inkling::varloop::frozen;

const std::string InklingMoeVarloopModel::kType = frozen::kModelType;

namespace {

// The operator's pass-count request. 0 (unset / unparsable) means "no override":
// the count the checkpoint declares is what runs.
int RequestedPasses()
{
    if (const char *v = std::getenv("LSX_INKLING_VARLOOP_PASSES")) {
        const int parsed = std::atoi(v);
        if (parsed > 0) return parsed;
    }
    return 0;
}

int EnvInt(const char *name, int dflt)
{
    if (const char *v = std::getenv(name))
        if (const int parsed = std::atoi(v); parsed > 0) return parsed;
    return dflt;
}

}  // namespace

bool VarloopTopologyMatches(const inkling::InklingLoopTopology &top)
{
    // ── Capability first, and NEVER relaxed ────────────────────────────────
    // `varloop` is not a size: it says which KIND of checkpoint this is. The
    // synthetic-geometry escape below exists because a unit fixture is 8 layers
    // wide and declares 2 passes, which are legitimate differences in
    // MAGNITUDE; a fixed-pass tree loaded into the varloop row is a different
    // MODEL, and no fixture ever needs that to be legal. Checking it before the
    // escape is what makes test_inkling_moe_varloop_model's "refuses the
    // fixed-pass sibling's artifact" case able to fail: with both checks under
    // one early return, the refusal never ran and the case could only pass
    // vacuously.
    //
    // The declared pass count and the ceiling stay BELOW the escape on purpose —
    // they are numbers, and a synthetic checkpoint legitimately has its own.
    if (!top.varloop) {
        log::error("[varloop] index does not declare varloop: this model's "
                   "weights are trained across 1..5 passes");
        return false;
    }

    // ── Geometry, relaxed for synthetic fixtures ───────────────────────────
    if (SyntheticGeometryAllowed()) return true;
    struct Field {
        const char *name;
        long long got;
        long long want;
    };
    const Field kFields[] = {
        {"num_layers", top.num_layers, frozen::kLayers},
        {"hidden_size", top.hidden, frozen::kHidden},
        {"vocab_size", top.vocab, frozen::kVocab},
        {"unpadded_vocab_size", top.unpadded_vocab, frozen::kBaseVocab},
        {"num_attention_heads", top.num_heads, frozen::kHeads},
        {"num_key_value_heads", top.num_kv_heads, frozen::kKvHeads},
        {"head_dim", top.head_dim, frozen::kHeadDim},
        {"d_rel", top.d_rel, frozen::kDRel},
        {"rel_extent", top.rel_extent, frozen::kRelExtent},
        {"sliding_window_size", top.sliding_window, frozen::kSlidingWindow},
        {"conv_kernel", top.conv_kernel, frozen::kConvKernel},
        {"attn_res_block", top.attn_res_block, frozen::kAttnResBlock},
        {"num_experts", top.num_experts, frozen::kExperts},
        {"num_experts_per_tok", top.top_k, frozen::kTopK},
        {"moe_intermediate_size", top.moe_intermediate, frozen::kMoeIntermediate},
        {"num_shared_experts", top.shared_experts, frozen::kSharedExperts},
        {"loop_start", top.loop_start, frozen::kLoopStart},
        {"loop_end", top.loop_end, frozen::kLoopEnd},
        {"loop_gate_rank", top.loop_gate_rank, frozen::kLoopGateRank},
        {"log_scaling_n_floor", top.log_scaling_n_floor, frozen::kLogScaleFloor},
    };
    for (const Field &f : kFields) {
        if (f.got != f.want) {
            log::error(absl::StrFormat(
                "[varloop] frozen geometry mismatch: %s = %lld, passport says "
                "%lld",
                f.name, f.got, f.want));
            return false;
        }
    }
    if (static_cast<int>(top.full_layers.size()) != frozen::kFullLayerCount) {
        log::error(absl::StrFormat(
            "[varloop] frozen geometry mismatch: %zu full-attention layers, "
            "passport says %d",
            top.full_layers.size(), frozen::kFullLayerCount));
        return false;
    }
    for (std::size_t i = 0; i < top.full_layers.size(); ++i)
        if (top.full_layers[i] != frozen::kFullLayers[i]) {
            log::error(absl::StrFormat(
                "[varloop] frozen geometry mismatch: full layer %zu is %d, "
                "passport says %d",
                i, top.full_layers[i], frozen::kFullLayers[i]));
            return false;
        }
    // The DECLARED count and the CEILING. The declared value is what the
    // checkpoint was post-trained at; the ceiling is the model card's range. Both
    // are magnitudes, so they sit below the synthetic escape; a real tree still
    // cannot claim a range its weights were never trained for.
    if (top.loop_passes != frozen::kLoopPassesDefault) {
        log::error(absl::StrFormat(
            "[varloop] loop_passes is %d, passport says the checkpoint "
            "declares %d",
            top.loop_passes, frozen::kLoopPassesDefault));
        return false;
    }
    if (top.loop_passes_max != frozen::kLoopPassesMax) {
        log::error(absl::StrFormat(
            "[varloop] loop_passes_max is %d, passport says the trained range "
            "ends at %d",
            top.loop_passes_max, frozen::kLoopPassesMax));
        return false;
    }
    return true;
}

InklingMoeVarloopModel::InklingMoeVarloopModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
}

InklingMoeVarloopModel::~InklingMoeVarloopModel() { Unload(); }

bool InklingMoeVarloopModel::Load()
{
    if (loaded_) return true;
    tokenizer_optional_ = SyntheticGeometryAllowed();
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    // The format tag is this row's: the sibling's tree must be refused here, not
    // driven with a pass knob that does not exist for its weights.
    auto st = inkling::LoadInklingLoopTopology(cfg_.model_path, top_,
                                               frozen::kFormatTag);
    if (!st.ok()) {
        log::error(absl::StrFormat("[varloop] topology load failed: %s",
                                   st.ToString()));
        return AbortLoad();
    }
    if (!VarloopTopologyMatches(top_)) return AbortLoad();

    // The pass count: the declared default unless the operator asked for another
    // one, clamped into the trained range. Applied BEFORE anything is sized, so
    // the slab, the expert window and the decode all see the same number.
    const int declared = top_.loop_passes;
    const int requested = RequestedPasses();
    const int passes = frozen::ResolvePasses(requested, declared);
    if (requested > 0 && passes != requested) {
        log::warn(absl::StrFormat(
            "[varloop] requested %d passes, clamped to %d (trained range %d..%d)",
            requested, passes, frozen::kLoopPassesMin, frozen::kLoopPassesMax));
    }
    if (passes != declared) {
        top_.loop_passes = passes;
        // The slab is the ceiling's and does not move with the count, so the
        // line reports what actually ran against what was allocated.
        log::info(absl::StrFormat(
            "[varloop] running %d passes (checkpoint declares %d), %d layer "
            "applications per token, %d cache slots",
            passes, declared, top_.LayerApplications(), top_.CacheLayers()));
    }

    if (!EnsureContext()) {
        log::error("[varloop] cuda context init failed");
        return AbortLoad();
    }
    cudaStreamSynchronize(stream());

    // The loader sizes every plane from `pw.top`, so the resolved pass count must
    // be in place before it runs.
    pw_.top = top_;
    if (!inkling::LoadInklingLoopPinnedWeights(cfg_.model_path, pw_, stream())) {
        log::error("[varloop] pinned weight load failed");
        return AbortLoad();
    }
    cudaStreamSynchronize(stream());
    experts_ = std::make_unique<inkling::InklingLoopExpertDb>(
        inkling::LoadInklingLoopExpertIndex(
            cfg_.model_path, top_.num_layers, top_.num_experts,
            inkling::InklingLoopResolveExpertWindow(
                top_, EnvInt("LSX_INKLING_VARLOOP_MAX_OPEN", 4096)),
            /*warm_on_load=*/false));

    // Same residency decision as the parent: the routed set is 14.7 GiB and the
    // dense plane 3.2 GiB, so both fit beside the KV slab on a 32 GiB card and a
    // token's 384+ expert fetches become HBM reads. When the allocation does not
    // fit the cache stays disabled and host staging is used unchanged.
    if (EnvInt("LSX_INKLING_VARLOOP_DEVICE_EXPERTS", 1) != 0) {
        pw_.device_experts = inkling::LoadInklingLoopDeviceExperts(
            *experts_, top_, top_.moe_intermediate, top_.hidden, stream());
        cudaStreamSynchronize(stream());
    }

    if (!scratch_.Allocate(top_, inkling::kLoopMaxContext, stream())) {
        log::error("[varloop] decode scratch allocation failed");
        return AbortLoad();
    }

    weights_loaded_ = true;
    resident_bytes_ =
        pw_.resident_bytes + pw_.device_experts.total_bytes +
        static_cast<std::uint64_t>(scratch_.arena_bytes +
                                   scratch_.expert_stage_bytes +
                                   scratch_.fp8_stage_bytes);
    if (!AcquirePermanentWeights(resident_bytes_)) return AbortLoad();
    FinishLoad();
    return true;
}

gpu::InferenceResult InklingMoeVarloopModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!weights_loaded_) {
        r.error_msg = "tml-d24-varloop-research weights not loaded";
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

    // Streaming + cooperative cancellation inside one residency lock.
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    sample::ISampler &sampler = sample::SamplerFor(req.sampling);
    inkling::InklingLoopGenerationResult gen;
    const auto ec = this->RunUnderResidencyLock([&]() {
        return inkling::RunInklingLoopGeneration(
            stream(), pw_, *experts_, session_, scratch_, req.input_ids,
            req.max_gen_tokens, std::vector<int32_t>{eos_id},
            static_cast<float>(top_.final_logit_softcap), top_.unpadded_vocab,
            sampler, gen, &gs);
    });
    if (ec != cudaSuccess || !gen.ok) {
        r.error_msg = gen.error_msg.empty() ? "tml-d24-varloop-research cuda error"
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

gpu::InferenceResult InklingMoeVarloopModel::SelfCheck(const ModelRequest &req)
{
    // Same probe shape as the fixed-pass sibling because the checkpoint, the
    // template and the answer surface are the same; the pass count is whatever
    // this instance resolved (default 3). Stage 8 measures the real numbers on
    // this model's own weights — the framed prompt opens with the assistant and
    // thinking markers and reaches " Paris" inside a 32-token window.
    runtime::ParisSelfCheckOptions opts;
    opts.frame_prompt = true;
    opts.min_tokens = 32;
    opts.answer_window = 32;
    opts.substring_match = true;
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool InklingMoeVarloopModel::BuildPromptInputIds(
    const ArrowTokenizerDb &tok, std::string_view raw_prompt,
    std::vector<int32_t> &out_ids) const
{
    return inkling::loop::tools::EncodePromptIds(tok, raw_prompt, out_ids);
}

bool InklingMoeVarloopModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    return inkling::loop::tools::EncodeConversationIds(tok, conv, out_ids);
}

bool InklingMoeVarloopModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return inkling::loop::tools::ParseGeneratedCalls(tok, gen_tokens, out_calls);
}

void InklingMoeVarloopModel::TeardownWeights()
{
    scratch_.Free();
    session_.Free();
    if (experts_) {
        inkling::FreeInklingLoopExpertIndex(*experts_);
        experts_.reset();
    }
    if (weights_loaded_) {
        inkling::FreeInklingLoopPinnedWeights(pw_);
        weights_loaded_ = false;
    }
    TeardownContext();
}

}  // namespace lsxcommon
