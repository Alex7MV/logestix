#include "hy_moe_simulmt_model.h"

#include "hy_moe_simulmt_constants.h"
#include "lsxcommon/hy_moe_tool_format.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_cat.h>

namespace lsxcommon {
namespace {

hy::HyPassportGeometry SimulMTGeometry()
{
    using namespace hy_simulmt::frozen;
    return hy::HyPassportGeometry{
        .layers = kLayers,
        .hidden = kHidden,
        .heads = kHeads,
        .kv_heads = kKvHeads,
        .head_dim = kHeadDim,
        .rotary_dim = kRotaryDim,
        .vocab = kVocab,
        .dense_intermediate = kDenseIntermediate,
        .moe_intermediate = kMoeIntermediate,
        .shared_intermediate = kSharedIntermediate,
        .num_experts = kNumExperts,
        .top_k = kTopK,
        .first_moe_layer = kFirstMoeLayer,
    };
}

// Han characters (CJK Unified + ext A + compatibility + U+3007).
bool ContainsHan(std::string_view s)
{
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        unsigned cp = c;
        int len = 1;
        if ((c & 0xE0) == 0xC0) {
            len = 2;
            cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
            cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
            cp = c & 0x07;
        }
        for (int k = 1; k < len && i + static_cast<size_t>(k) < s.size(); ++k)
            cp = (cp << 6) |
                 (static_cast<unsigned char>(s[i + static_cast<size_t>(k)]) &
                  0x3F);
        if ((cp >= 0x2E80 && cp <= 0x2FDF) || cp == 0x3007 ||
            (cp >= 0x31C0 && cp <= 0x31EF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
            (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
            (cp >= 0x20000 && cp <= 0x3347F))
            return true;
        i += static_cast<size_t>(len);
    }
    return false;
}

// The official serving directive for the source script (zh<->en translator).
const char *DirectionInstruction(std::string_view source_text)
{
    return ContainsHan(source_text) ? hy_simulmt::frozen::kZhEnInstruction
                                    : hy_simulmt::frozen::kEnZhInstruction;
}

bool StartsWithInstruction(std::string_view text)
{
    using namespace hy_simulmt::frozen;
    return text.rfind(kZhEnInstruction, 0) == 0 ||
           text.rfind(kEnZhInstruction, 0) == 0;
}

std::string FrameTranslation(std::string_view source_text)
{
    if (StartsWithInstruction(source_text)) return std::string(source_text);
    return std::string(DirectionInstruction(source_text)) +
           std::string(source_text);
}

} // namespace

const std::string HySimulMTModel::kType = "hy-simulmt-30b-a3b";

HySimulMTModel::HySimulMTModel(ModelInitConfig cfg) : ModelBase{std::move(cfg)} {}

HySimulMTModel::~HySimulMTModel() { Unload(); }

void HySimulMTModel::TeardownWeights()
{
    scratch_.Free();
    session_.Free();
    pw_.Free();
    TeardownContext();
}

bool HySimulMTModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    hy::HyTopology top = hy::LoadHyTopology(cfg_.model_path);
    const std::string fmt = top.error_msg;
    top.error_msg.clear();
    if (top.num_layers <= 0) {
        lsxcommon::log::error(absl::StrCat(
            "[hy-simulmt-model] topology error: ",
            fmt.empty() ? "model.index.json unreadable" : fmt));
        return AbortLoad();
    }
    // Format contract: no backward compatibility (AGENTS.md).
    if (!fmt.empty() && fmt != hy_simulmt::frozen::kFormatTag) {
        lsxcommon::log::error(absl::StrCat(
            "[hy-simulmt-model] unsupported index format '", fmt,
            "' — convert via: lsxforge --hy_simulmt --input <model_dir> "
            "--output <model_dir>"));
        return AbortLoad();
    }
    // Frozen-geometry passport cross-check (Model Ingestion Protocol step 3).
    using namespace hy_simulmt::frozen;
    if (!SyntheticGeometryAllowed() &&
        (top.num_layers != kLayers || top.hidden_size != kHidden ||
         top.num_experts != kNumExperts || top.num_attention_heads != kHeads ||
         top.num_key_value_heads != kKvHeads || top.head_dim != kHeadDim ||
         top.vocab_size != kVocab ||
         top.moe_intermediate_size != kMoeIntermediate)) {
        lsxcommon::log::error(absl::StrCat(
            "[hy-simulmt-model] topology mismatches frozen passport (layers ",
            top.num_layers, "/", static_cast<int>(kLayers), ", hidden ",
            top.hidden_size, "/", static_cast<int>(kHidden), ", experts ",
            top.num_experts, "/", static_cast<int>(kNumExperts), ")"));
        return AbortLoad();
    }
    vram_ = top.vram;

    if (!EnsureContext()) {
        lsxcommon::log::error("[hy-simulmt-model] context create failed");
        return AbortLoad();
    }
    std::string error;
    if (!hy::LoadHyDenseWeights(cfg_.model_path, SimulMTGeometry(),
                                hy_simulmt::frozen::kFormatTag, pw_, stream(),
                                error)) {
        lsxcommon::log::error(
            absl::StrCat("[hy-simulmt-model] pinned weight load failed: ", error));
        return AbortLoad();
    }

    resident_bytes_ = pw_.resident_bytes;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

    lsxcommon::log::info(absl::StrCat(
        "[hy-simulmt-model] loaded: ", pw_.layers.size(), " layers, ",
        pw_.resident_bytes >> 20, " MiB pinned (all routed experts resident)"));
    FinishLoad();
    return true;
}

gpu::InferenceResult HySimulMTModel::Infer(const ModelRequest &req)
{
    // Model independence: assert THIS model's fast-math policy for the call.
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    r = RunUnderResidencyLock([&] {
        return hy::RunHyInference(
            cfg_.model_path, req.input_ids, req.max_gen_tokens, pw_, session_,
            scratch_, lsxcommon::sample::SamplerFor(req.sampling), stream(),
            req.ignore_eos, &gs);
    });
    if (r.ok) {
        r.output_text = runtime::DecodeGeneratedText(
            *tok_db_, r.gen_tokens.data(), r.gen_tokens.size());
        r.num_layers = pw_.top.num_layers;
        r.weight_resident_bytes =
            resident_bytes_ + scratch_.arena_bytes + session_.Bytes();
    }
    return r;
}

gpu::InferenceResult HySimulMTModel::SelfCheck(const ModelRequest &req)
{
    // Hy-SimulMT is a direction-bound zh<->en simultaneous translator: its
    // text decoder TRANSLATES the source it is given (with the checkpoint's
    // own serving directive) instead of answering an English question. The
    // canonical English probe ("The capital of France is", en-zh) therefore
    // yields the Chinese sentence "法国的首都是" and can never contain
    // "Paris". The family probe asks the same fact from the other direction:
    // the Chinese source "法国的首都是巴黎。" must translate to English and
    // contain "Paris". Recorded deviation (docs/SELFCHECK_MATRIX.md).
    const runtime::ParisSelfCheckOptions opts{
        .frame_prompt = true,
        .min_tokens = 16,
        .answer_window = 16,
        .substring_match = true,
        .prompt = "法国的首都是巴黎。",
    };
    return runtime::RunParisSelfCheck(*this, req, opts);
}

bool HySimulMTModel::BuildPromptInputIds(
    const ArrowTokenizerDb &tok, std::string_view raw_prompt,
    std::vector<int32_t> &out_ids) const
{
    ChatConversation conv;
    conv.messages.push_back({"user", FrameTranslation(raw_prompt), "", {}});
    if (hy::RenderHyConversation(tok, conv, out_ids, hy::HyRenderPolicy{
            .example_call_close_is_calls_end = true}))
        return true;
    out_ids.clear();
    return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
}

bool HySimulMTModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    // Translation contract: the direction directive prefixes the FIRST user
    // turn (the official client sends it once per request), chosen from the
    // LAST user turn's script. Tool/system turns are preserved as-is.
    ChatConversation framed = conv;
    const ChatMessage *last_user = nullptr;
    ChatMessage *first_user = nullptr;
    for (auto &m : framed.messages) {
        if (m.role != "user") continue;
        if (!first_user) first_user = &m;
        last_user = &m;
    }
    if (first_user && last_user) {
        if (!StartsWithInstruction(first_user->content))
            first_user->content =
                std::string(DirectionInstruction(last_user->content)) +
                first_user->content;
    }
    if (hy::RenderHyConversation(tok, framed, out_ids, hy::HyRenderPolicy{
            .example_call_close_is_calls_end = true}))
        return true;
    // Fallback: the shared Hermes renderer (a floor, not a pass).
    const std::string text = tools::RenderConversation(conv, /*include_tools=*/true);
    if (text.empty()) return false;
    out_ids.clear();
    return BuildPromptInputIds(tok, text, out_ids);
}

bool HySimulMTModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return hy::ParseHyToolCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
