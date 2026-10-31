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

GraniteSwitch41_30BModel::GraniteSwitch41_30BModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg)) {}

GraniteSwitch41_30BModel::~GraniteSwitch41_30BModel() { Unload(); }

void GraniteSwitch41_30BModel::TeardownWeights()
{
    pw_.Free();
    weights_loaded_ = false;
    kv_.reset();
    session_tail_.clear();
    TeardownContext();
}

bool GraniteSwitch41_30BModel::Load()
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

bool GraniteSwitch41_30BModel::ValidateGeometry(
    const granite_switch::GraniteSwitchTopology &top) const
{
    if (SyntheticGeometryAllowed()) return true;
    const GeoFieldCheck fields[] = {
        {"hidden_size", kHiddenSize, top.hidden_size},
        {"num_layers", kNumLayers, top.num_layers},
        {"num_attention_heads", kNumHeads, top.num_attention_heads},
        {"num_key_value_heads", kNumKVHeads, top.num_key_value_heads},
        {"head_dim", kHeadDim, top.head_dim},
        {"intermediate_size", kIntermediateSize, top.intermediate_size},
        {"vocab_size", kVocabSize, top.vocab_size},
        {"num_adapters", granite_switch::frozen::kNumAdapters,
         top.num_adapters},
    };
    if (!CheckGeometryFields("granite-switch-4.1-30b", fields,
                             sizeof(fields) / sizeof(fields[0])))
        return false;
    auto close = [](double a, double b) {
        return std::abs(a - b) <= 1e-6 * std::max(1.0, std::abs(b));
    };
    if (!close(top.embedding_multiplier,
               granite_switch::frozen::kEmbeddingMultiplier) ||
        !close(top.residual_multiplier,
               granite_switch::frozen::kResidualMultiplier) ||
        !close(top.logits_scaling, granite_switch::frozen::kLogitsScaling) ||
        !close(top.attention_multiplier,
               granite_switch::frozen::kAttentionMultiplier)) {
        lsxcommon::log::error("[granite-switch-model] numeric multipliers differ "
                              "from the frozen passport");
        return false;
    }
    for (int a = 0; a < top.num_adapters; ++a) {
        if (top.adapter_ranks[a] !=
                granite_switch::frozen::kAdapterRanks[a] ||
            top.adapter_token_ids[a] !=
                granite_switch::frozen::kControlTokenIds[a] ||
            top.adapter_substitute_token_ids[a] !=
                granite_switch::frozen::kSubstituteTokenIds[a]) {
            lsxcommon::log::error(
                "[granite-switch-model] adapter metadata differs from passport");
            return false;
        }
    }
    return true;
}

gpu::InferenceResult GraniteSwitch41_30BModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

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

gpu::InferenceResult GraniteSwitch41_30BModel::SelfCheck(
    const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

bool GraniteSwitch41_30BModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderGraniteSwitchConversation(conv, text)) return false;
    if (!AppendGraniteSwitchIds(tok, text, out_ids))
        return BuildPromptInputIds(tok, text, out_ids);
    return true;
}

bool GraniteSwitch41_30BModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return ParseGraniteSwitchTokenCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
