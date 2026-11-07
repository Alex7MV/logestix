#include "granite_dense_switch_model.h"

#include <cuda_runtime.h>

#include <cmath>

#include "fast_load_internal.h"
#include "lsxcommon/log.h"
#include "lsxcommon/granite_switch_tool_format.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"

#include <absl/strings/str_cat.h>

namespace lsxcommon {

const std::string GraniteSwitch41_30BModel::kType =
    "granite-switch-4.1-30b";
const std::string GraniteSwitch41_8BModel::kType =
    "granite-switch-4.1-8b";

GraniteSwitch41_30BModel::GraniteSwitch41_30BModel(ModelInitConfig cfg)
    : GraniteSwitchDenseModel(std::move(cfg), granite_switch::frozen::Frozen())
{
}

GraniteSwitch41_30BModel::~GraniteSwitch41_30BModel() { Unload(); }

GraniteSwitch41_8BModel::GraniteSwitch41_8BModel(ModelInitConfig cfg)
    : GraniteSwitchDenseModel(std::move(cfg),
                              granite_switch_8b::Frozen())
{
}

GraniteSwitch41_8BModel::~GraniteSwitch41_8BModel() { Unload(); }

GraniteSwitchDenseModel::GraniteSwitchDenseModel(
    ModelInitConfig cfg, granite_switch::GraniteSwitchFrozen frozen)
    : ModelBase(std::move(cfg)), frozen_(frozen) {}

void GraniteSwitchDenseModel::TeardownWeights()
{
    pw_.Free();
    weights_loaded_ = false;
    kv_.reset();
    session_tail_.clear();
    TeardownContext();
}

bool GraniteSwitchDenseModel::Load()
{
    Unload();
    tokenizer_optional_ = false;
    const FastMathMode fm = EffectiveFastMathMode(cfg_);
    if (fm != FastMathMode::kAuto) {
        lsx_nvfp4::SetLsxTcFastMath(fm == FastMathMode::kOn);
        lsx_nvfp4::SetLsxSeq1Gemv(fm == FastMathMode::kOn);
    }
    if (!BeginLoad(cfg_.model_path)) return false;
    if (!EnsureContext()) {
        lsxcommon::log::error("[granite-switch-model] stream create failed");
        return AbortLoad();
    }
    const bool ok = granite_switch::LoadGraniteSwitchPinnedWeights(
        cfg_.model_path, pw_, stream());
    fastload::StreamDrain{stream()};
    weights_loaded_ = ok;
    if (!ok) {
        lsxcommon::log::error(
            "[granite-switch-model] LoadGraniteSwitchPinnedWeights failed");
        return AbortLoad();
    }
    resident_bytes_ = pw_.resident_bytes;
    vram_ = pw_.top.vram;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    lsxcommon::log::info(absl::StrCat(
        "[granite-switch-model] pinned weights loaded: ", pw_.layers.size(),
        " layers (", pw_.resident_bytes, " bytes resident)"));
    if (!ValidateGeometry(pw_.top)) {
        lsxcommon::log::error(absl::StrCat(
            "[granite-switch-model] geometry validation failed for ",
            cfg_.model_path));
        return AbortLoad();
    }
    FinishLoad();
    return true;
}

bool GraniteSwitchDenseModel::ValidateGeometry(
    const granite_switch::GraniteSwitchTopology &top) const
{
    if (SyntheticGeometryAllowed()) return true;
    // One checker for every member: the expectations arrive as data, so adding a
    // sibling cannot silently inherit the parent's numbers. The message names the
    // class's own model_type and every differing field, and the caller logs the
    // tree it tried to load.
    const std::string why =
        granite_switch::GraniteSwitchGeometryMismatch(frozen_, top);
    if (why.empty()) return true;
    lsxcommon::log::error(absl::StrCat(
        "[granite-switch-model] ", frozen_.model_type,
        " geometry mismatch (converted tree at ", cfg_.model_path, "):\n", why));
    return false;
}

gpu::InferenceResult GraniteSwitchDenseModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    // The checkpoint's embedding_multiplier=12 / residual_multiplier=0.175
    // produces activations up to ~6e3. The NVFP4 TC fast-math path quantizes
    // ACTIVATIONS to E2M1 with E4M3 block scales (max representable ~2.7e3),
    // so long prompts clamp and decode garbage (measured: the 197-token tool
    // prompt returned `x = x e^{x}` on the TC path vs the source tool call).
    // Pin the FP16-activation strict path for this family; the per-GPU policy
    // is still consulted so the guard's model-independence contract holds.
    const FastMathMode requested = EffectiveFastMathMode(cfg_);
    (void)requested;
    const lsx_nvfp4::LsxFastMathScope fm_scope(FastMathMode::kOff);

    if (!EnsureContext()) {
        r.error_msg = "cuda init failed";
        return r;
    }
    cudaStream_t st = stream();

    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    return RunUnderResidencyLock([&]() -> gpu::InferenceResult {
        const bool continuation =
            kv_ && kv_->allocated() && !session_tail_.empty() &&
            req.input_ids == session_tail_;
        std::vector<int32_t> gen_input;
        if (continuation) {
            gen_input = std::vector<int32_t>{kv_->last_tok};
        } else {
            kv_.reset();
            kv_.reset(new granite_switch::GraniteSwitchKvCache());
            session_tail_.clear();
            gen_input = req.input_ids;
        }

        granite_switch::GraniteSwitchGenerationResult gen;
        const int eos_id = (req.ignore_eos || !tok_db_)
                               ? lsxcommon::kIgnoreEosTokenId
                               : tok_db_->eos_id;
        const cudaError_t e = granite_switch::RunGraniteSwitchGeneration(
            st, pw_, gen_input, req.max_gen_tokens, eos_id, gen,
            lsxcommon::sample::SamplerFor(req.sampling), kv_.get(), &gs);
        if (e != cudaSuccess) {
            r.error_msg = runtime::CombineGenError(e, gen.error_msg);
            r.ok = false;
            kv_.reset();
            session_tail_.clear();
            return r;
        }

        const int kv_slots =
            (kv_ && kv_->allocated())
                ? kv_->kv_rows
                : static_cast<int>(gen.tokens.size()) +
                      static_cast<int>(req.input_ids.size());
        runtime::FillInferenceResult(
            r, {.tokens = &gen.tokens,
                .final_logits = &gen.final_logits,
                .ttft_ms = gen.ttft_ms,
                .total_ms = gen.total_ms,
                .error_msg = &gen.error_msg,
                .eos_terminated = gen.eos_terminated,
                .kv_slots_used = kv_slots,
                .first_maxv = 0.f,
                .resident_bytes = gen.resident_bytes},
            pw_.top.num_layers);
        session_tail_ = r.gen_tokens;
        if (tok_db_ && !r.gen_tokens.empty())
            r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
        return r;
    });
}

gpu::InferenceResult GraniteSwitchDenseModel::SelfCheck(
    const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

bool GraniteSwitchDenseModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderGraniteSwitchConversation(conv, text)) return false;
    if (!AppendGraniteSwitchIds(tok, text, out_ids))
        return BuildPromptInputIds(tok, text, out_ids);
    return true;
}

bool GraniteSwitchDenseModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return ParseGraniteSwitchTokenCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
