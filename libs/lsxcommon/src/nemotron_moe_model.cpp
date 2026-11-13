#include "nemotron_moe_model.h"
#include "nemotron_moe_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/qwen_tool_format.h"
#include "lsxcommon/tool_format.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace lsxcommon {

const std::string NemotronHModel::kType = nemotron_moe::frozen::kType;

NemotronModelBase::NemotronModelBase(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
    // Synthetic fixtures may omit the tokenizer arrows.
    tokenizer_optional_ = true;
}

NemotronHModel::NemotronHModel(ModelInitConfig cfg)
    : NemotronModelBase(std::move(cfg))
{
}

NemotronHModel::~NemotronHModel() { Unload(); }

bool NemotronHModel::LoadAndValidateIndex(std::string &error)
{
    if (!nemotron_moe::LoadNemotronIndex(cfg_.model_path, idx_, error))
        return false;
    // Both distributions emit `logestix-nemotron-nvfp4`, so the format tag
    // alone cannot tell them apart. The `model` field can - and it has to,
    // because the two carry separate FSI/CODE/DET goldens and a mixed-up tree
    // would silently verify against the wrong one.
    using namespace nemotron_moe::frozen;
    if (idx_.model != kModelName) {
        error = absl::StrCat("model.index.json declares model '", idx_.model,
                             "' but this loader serves '", kModelName,
                             "' - the other distribution's tree is a different "
                             "model");
        return false;
    }
    if (!nemotron_moe::ValidateNemotronGeometry(idx_, error)) {
        log::error(absl::StrFormat(
            "[nemotron] frozen geometry mismatch: %s", error));
        return false;
    }
    return true;
}

bool NemotronModelBase::Load()
{
    if (loaded_) return true;
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    std::string err;
    if (!LoadAndValidateIndex(err)) {
        log::error(absl::StrFormat("[nemotron] index error: %s", err));
        return AbortLoad();
    }
    vram_ = idx_.vram;

    if (!EnsureContext()) return AbortLoad();

    {
        std::string err;
        if (!nemotron_moe::LoadNemotronWeights(cfg_.model_path, idx_, weights_,
                                               stream(), err)) {
            log::error(absl::StrFormat("[nemotron] weight load error: %s", err));
            weights_.Free();
            return AbortLoad();
        }
        const int kMaxSeq = 4096;
        if (!nemotron_moe::AllocNemotronSession(idx_, kMaxSeq, session_, err)) {
            log::error(absl::StrFormat("[nemotron] session alloc error: %s",
                                       err));
            weights_.Free();
            return AbortLoad();
        }
        // Routed experts: host mmap store + device slot LRU sized from free
        // VRAM. Only the dense backbone is permanent; the expert pool streams.
        nemotron_moe::LoadNmExpertStore(cfg_.model_path, idx_.num_layers,
                                        idx_.num_experts, expert_cache_.store);
        expert_cache_.experts_per_layer = idx_.num_experts;
        expert_cache_.moe_index_of_layer.assign(idx_.num_layers, -1);
        expert_cache_.moe_layers = 0;
        for (int l = 0; l < idx_.num_layers; ++l)
            if (idx_.layer_types[l] == "moe")
                expert_cache_.moe_index_of_layer[l] =
                    expert_cache_.moe_layers++;
        expert_cache_.total_experts =
            expert_cache_.moe_layers * idx_.num_experts;
        expert_cache_.layout =
            nemotron_moe::PlanNmExpertSlotFromStore(expert_cache_.store);
        if (!expert_cache_.layout.ok) {
            log::error("[nemotron] expert store carried no usable payload");
            TeardownWeights();
            return AbortLoad();
        }
        std::size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const std::size_t per_expert = static_cast<std::size_t>(
            nemotron_moe::NmExpertCachePerExpertBytes(expert_cache_.layout));
        const int slots = nemotron_moe::NmExpertCacheSlots(
            free_b, per_expert, expert_cache_.total_experts,
            idx_.num_experts_per_tok,
            std::getenv("LSX_NEMOTRON_EXPERT_CACHE_SLOTS"));
        log::info(absl::StrFormat(
            "[nemotron] expert cache: %d/%d slots (~%llu MiB), free=%llu MiB",
            slots, expert_cache_.total_experts,
            static_cast<unsigned long long>(slots * (per_expert >> 20)),
            static_cast<unsigned long long>(free_b >> 20)));
        if (!nemotron_moe::AllocateNmExpertCache(expert_cache_, slots, err)) {
            log::error(absl::StrFormat("[nemotron] expert cache error: %s", err));
            TeardownWeights();
            return AbortLoad();
        }
        // Fill the slots now (untimed): a full-size cache then behaves exactly
        // like the old resident loader on the first decode token.
        if (nemotron_moe::PrewarmNmExpertCache(expert_cache_, stream(), err) !=
            cudaSuccess) {
            log::error(absl::StrFormat("[nemotron] expert prewarm error: %s",
                                       err));
            TeardownWeights();
            return AbortLoad();
        }
        resident_bytes_ = weights_.resident_bytes + expert_cache_.cache_bytes;
    }

    // Governor accounting (lifecycle-unification II): the dense backbone is
    // permanent, the expert slot cache is a separate adaptive lease (both
    // released by the canonical Unload/AbortLoad).
    if (!AcquirePermanentWeights(weights_.resident_bytes)) return AbortLoad();
    AcquireExpertCacheLease(expert_cache_.cache_bytes, expert_cache_.cache_bytes,
                            expert_cache_.cache_bytes);

    FinishLoad();
    log::info(absl::StrFormat(
        "[nemotron] loaded %s: %d layers, %d experts, vocab %d",
        idx_.model.c_str(), idx_.num_layers, idx_.num_experts, idx_.vocab_size));
    return true;
}

void NemotronModelBase::TeardownWeights()
{
    nemotron_moe::FreeNmExpertCache(expert_cache_);
    session_.Free();
    weights_.Free();
    TeardownContext();
}

namespace {

int ArgMaxHost(const std::vector<float> &v)
{
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = static_cast<int>(i);
    return best;
}

// Owns the three timing events of one Infer run. The decode loop has four
// early error returns; without this guard each of them leaked its events.
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

} // namespace

gpu::InferenceResult NemotronHModel::Infer(const ModelRequest &req)
{
    // Model independence: this model's own kernel-routing policy for the
    // duration of the call (see LsxFastMathScope in lsx_inference_nvfp4.h).
    lsx_nvfp4::LsxFastMathScope scope(EffectiveFastMathMode(cfg_));
    return RunInferShared(req);
}

gpu::InferenceResult NemotronModelBase::RunInferShared(const ModelRequest &req)
{
    return RunUnderResidencyLock([&] {
        gpu::InferenceResult r;
        if (!ValidateInferRequest(req, r)) return r;

        lsxcommon::GenerationStream gs;
        gs.on_token = req.token_callback ? &req.token_callback : nullptr;
        gs.abort_flag = req.abort_flag;

        nemotron_moe::ResetNemotronSession(session_, stream());
        std::vector<float> logits(static_cast<size_t>(idx_.vocab_size));

        ScopedCudaEvents ev;
        cudaEventRecord(ev.start, stream());

        int pos = 0;
        for (int32_t id : req.input_ids) {
            cudaError_t e = nemotron_moe::NemotronForwardToken(
                idx_, weights_, session_, expert_cache_, id, pos, true, stream());
            if (e != cudaSuccess) {
                r.error_msg = std::string("forward failed: ") +
                              cudaGetErrorString(e);
                return r;
            }
            ++pos;
        }
        cudaEventRecord(ev.first, stream());

        const int eos = idx_.eos_token_id;
        auto &sampler = sample::SamplerFor(req.sampling);
        int generated = 0;
        for (int step = 0; step < req.max_gen_tokens; ++step) {
            if (cudaMemcpyAsync(logits.data(), session_.head_logits,
                                logits.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, stream()) !=
                    cudaSuccess ||
                cudaStreamSynchronize(stream()) != cudaSuccess) {
                r.error_msg = "logits readback failed";
                return r;
            }
            int next = ArgMaxHost(logits);
            if (!sampler.IsGreedy()) {
                float sampled_maxv = 0.0f;
                const cudaError_t se = sample::SampleToHost(
                    stream(), sampler, session_.head_logits,
                    static_cast<int>(logits.size()),
                    sample::LogitType::kF32, &next, &sampled_maxv);
                if (se != cudaSuccess) {
                    r.error_msg = "sampling failed";
                    return r;
                }
            }
            r.gen_tokens.push_back(next);
            ++generated;
            if (step == 0) {
                r.first_gen_tok_id = next;
                r.first_gen_maxv = logits[static_cast<size_t>(next)];
            }
            // Keep the last step's logits row (protocol: determinism check).
            r.final_logits = logits;
            if (!req.ignore_eos && next == eos) {
                r.eos_terminated = true;
                break;
            }
            if (step + 1 < req.max_gen_tokens) {
                cudaError_t e = nemotron_moe::NemotronForwardToken(
                    idx_, weights_, session_, expert_cache_, next, pos, false,
                    stream());
                if (e != cudaSuccess) {
                    r.error_msg = std::string("forward failed: ") +
                                  cudaGetErrorString(e);
                    return r;
                }
                ++pos;
            }
            gs.Emit(static_cast<std::uint32_t>(next), static_cast<std::uint32_t>(step),
                    logits[static_cast<size_t>(next)]);
            if (gs.Aborted()) break;
        }
        r.num_tokens = generated;
        r.num_layers = idx_.num_layers;
        r.weight_resident_bytes = resident_bytes_;
        r.kv_slots_used = pos;
        if (tok_db_ && !r.gen_tokens.empty())
            r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
        log::info(absl::StrFormat(
            "[nemotron] expert cache: hits=%lld misses=%lld slots=%d",
            static_cast<long long>(expert_cache_.device.hits),
            static_cast<long long>(expert_cache_.device.misses),
            expert_cache_.device.slots));
        cudaEventRecord(ev.stop, stream());
        cudaEventSynchronize(ev.stop);
        float ttft = 0.0f, total = 0.0f;
        cudaEventElapsedTime(&ttft, ev.start, ev.first);
        cudaEventElapsedTime(&total, ev.start, ev.stop);
        r.ttft_ms = static_cast<double>(ttft);
        r.total_time_ms = static_cast<double>(total);
        r.tokens_per_sec =
            total > 0.0f ? static_cast<double>(generated) * 1000.0 /
                               static_cast<double>(total)
                         : 0.0;
        r.ok = true;
        return r;
    });
}

gpu::InferenceResult NemotronModelBase::SelfCheck(const ModelRequest &req)
{
    // Unified Paris harness: the strict 1->2-token "Paris" contract plus the
    // cross-model `logits_hash` evidence and the LSX_SELFCHECK_EXPECT_HASH gate
    // (numeric-contract §1). frame_prompt stays OFF on purpose — the selfcheck
    // suite feeds its own control-prompt ids, and re-framing here would make
    // the gate depend on the chat template instead of the control token.
    return runtime::RunParisSelfCheck(*this, req);
}

bool NemotronModelBase::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                            std::string_view raw_prompt,
                                            std::vector<int32_t> &out_ids) const
{
    // A chat model must never see an unframed prompt. Render the single turn
    // through the SAME native template the conversation path uses (the
    // checkpoint's chat_template.jinja: an always-present leading system turn,
    // then `<|im_start|>role\n…<|im_end|>`, with the generation prompt opening
    // `<think>`). No system message is injected here: the template renders an
    // EMPTY system turn when the caller supplies none, and that is the frame
    // the checkpoint was trained on.
    ChatConversation conv;
    conv.messages.push_back({"user", std::string(raw_prompt), "", {}});
    std::string frame;
    if (RenderNemotronConversation(conv, frame)) {
        if (AppendQwenChatIds(tok, frame, out_ids)) return true;
        return prompt::EncodeRawPrompt(tok, frame, out_ids);
    }
    return false;
}

// ── Nemotron-3.5 ChatML/XML tool format ─────────────────────────────────────
// Verified against chat_template.jinja: a Qwen-style `<|im_start|>` ChatML with
// an XML tool declaration block and `<tool_call><function=NAME><parameter=K>V`
// calls; the call/parse syntax is shared with the Qwen codec.

namespace {

constexpr const char *kNemotronToolInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format "
    "with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second "
    "parameter\nthat can span\nmultiple lines\n</parameter>\n"
    "</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal "
    "with your current knowledge and do not tell the user about function "
    "calls\n</IMPORTANT>";

std::string JsonSerialize(simdjson::dom::element e)
{
    using simdjson::dom::element_type;
    switch (e.type()) {
        case element_type::OBJECT: {
            std::string out = "{";
            bool first = true;
            for (auto [k, v] : e.get_object()) {
                if (!first) out += ",";
                first = false;
                out += "\"" + std::string(k) + "\":" + JsonSerialize(v);
            }
            out += "}";
            return out;
        }
        case element_type::ARRAY: {
            std::string out = "[";
            bool first = true;
            for (auto v : e.get_array()) {
                if (!first) out += ",";
                first = false;
                out += JsonSerialize(v);
            }
            out += "]";
            return out;
        }
        case element_type::STRING: {
            std::string out = "\"";
            for (char c : std::string(e.get_string().value_unsafe())) {
                if (c == '"' || c == '\\') out += '\\';
                out += c;
            }
            return out + "\"";
        }
        case element_type::INT64:
            return std::to_string(e.get_int64().value_unsafe());
        case element_type::UINT64:
            return std::to_string(e.get_uint64().value_unsafe());
        case element_type::DOUBLE: {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.17g",
                          e.get_double().value_unsafe());
            return std::string(buf);
        }
        case element_type::BOOL:
            return e.get_bool().value_unsafe() ? "true" : "false";
        default:
            return "null";
    }
}

std::string RenderToolDeclaration(const ToolSpec &t)
{
    std::string o = "\n<function>\n<name>" + t.name + "</name>";
    if (!t.description.empty())
        o += "\n<description>" + t.description + "</description>";
    o += "\n<parameters>";
    simdjson::dom::parser parser;
    simdjson::dom::element params;
    simdjson::padded_string padded(t.parameters_json);
    if (!t.parameters_json.empty() &&
        parser.parse(padded).get(params) == simdjson::SUCCESS &&
        params.type() == simdjson::dom::element_type::OBJECT) {
        simdjson::dom::element props;
        if (params["properties"].get(props) == simdjson::SUCCESS &&
            props.type() == simdjson::dom::element_type::OBJECT) {
            for (auto [k, v] : props.get_object()) {
                o += "\n<parameter>\n<name>";
                o += std::string(k);
                o += "</name>";
                std::string_view type;
                if (v["type"].get_string().get(type) == simdjson::SUCCESS) {
                    o += "\n<type>";
                    o += std::string(type);
                    o += "</type>";
                }
                std::string_view desc;
                if (v["description"].get_string().get(desc) == simdjson::SUCCESS) {
                    o += "\n<description>";
                    o += std::string(desc);
                    o += "</description>";
                }
                simdjson::dom::element en;
                if (v["enum"].get(en) == simdjson::SUCCESS)
                    o += "\n<enum>" + JsonSerialize(en) + "</enum>";
                o += "\n</parameter>";
            }
        }
        simdjson::dom::element req;
        if (params["required"].get(req) == simdjson::SUCCESS)
            o += "\n<required>" + JsonSerialize(req) + "</required>";
    }
    o += "\n</parameters>\n</function>";
    return o;
}

} // namespace

bool NemotronModelBase::RenderNemotronConversation(const ChatConversation &conv,
                                                   std::string &out)
{
    out.clear();
    static const std::vector<std::string_view> kMarkers = {
        "<|im_start|>",   "<|im_end|>",     "<tool_call>",
        "</tool_call>",   "<tool_response>", "</tool_response>",
        "<think>",        "</think>",
    };
    const auto esc = [](std::string_view s) {
        return tools::NeutralizeControlMarkers(s, kMarkers);
    };

    const bool first_system =
        !conv.messages.empty() && conv.messages[0].role == "system";
    const bool has_tools = !conv.tools.empty();
    // The checkpoint's chat_template.jinja assigns `system_message = ""` when
    // messages[0] is not a system turn and then branches on
    // `system_message is defined`, which is therefore ALWAYS true. The frame
    // ALWAYS opens with a system turn — empty when the caller supplied none.
    // Gating this on `first_system || has_tools` produced a frame that opens at
    // the user turn, which the checkpoint never sees.
    out += "<|im_start|>system\n";
    if (first_system) out += esc(conv.messages[0].content);
    if (has_tools) {
        if (first_system && !conv.messages[0].content.empty()) out += "\n\n";
        out +=
            "# Tools\n\nYou have access to the following functions:\n\n"
            "<tools>";
        for (const ToolSpec &t : conv.tools)
            out += RenderToolDeclaration(t);
        out += "\n</tools>";
        out += kNemotronToolInstructions;
    }
    out += "<|im_end|>\n";

    simdjson::dom::parser parser;
    bool in_tool_turn = false;
    for (size_t i = first_system ? 1 : 0; i < conv.messages.size(); ++i) {
        const ChatMessage &m = conv.messages[i];
        if (m.role == "system") {
            out += "<|im_start|>system\n" + esc(m.content) + "<|im_end|>\n";
        } else if (m.role == "user") {
            out += "<|im_start|>user\n" + esc(m.content) + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            out += "<|im_start|>assistant\n";
            out += m.content.empty() ? std::string("<think></think>")
                                     : esc(m.content) + "\n";
            for (const ToolCall &c : m.tool_calls) {
                out += "<tool_call>\n<function=" + esc(c.name) + ">\n";
                if (!c.arguments_json.empty()) {
                    simdjson::dom::element args;
                    simdjson::padded_string padded(c.arguments_json);
                    if (parser.parse(padded).get(args) == simdjson::SUCCESS &&
                        args.type() ==
                            simdjson::dom::element_type::OBJECT) {
                        for (auto [k, v] : args.get_object()) {
                            out += "<parameter=" + std::string(k) + ">\n";
                            if (v.type() ==
                                simdjson::dom::element_type::STRING)
                                out += esc(v.get_string().value_unsafe());
                            else
                                out += JsonSerialize(v);
                            out += "\n</parameter>\n";
                        }
                    }
                }
                out += "</function>\n</tool_call>\n";
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            if (!in_tool_turn) {
                out += "<|im_start|>user\n";
                in_tool_turn = true;
            }
            out += "<tool_response>\n" + esc(m.content) + "\n</tool_response>\n";
            const bool next_is_tool =
                i + 1 < conv.messages.size() &&
                conv.messages[i + 1].role == "tool";
            if (!next_is_tool) {
                out += "<|im_end|>\n";
                in_tool_turn = false;
            }
        } else {
            return false;
        }
    }

    // enable_thinking is on by default (README): the model reasons inside
    // <think>…</think> and then emits the call.
    out += "<|im_start|>assistant\n<think>\n";
    return true;
}

bool NemotronModelBase::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderNemotronConversation(conv, text)) return false;
    if (!AppendQwenChatIds(tok, text, out_ids))
        // Already-rendered conversation: encode it verbatim, never re-wrap
        // it in the single-turn frame.
        return prompt::EncodeRawPrompt(tok, text, out_ids);
    return true;
}

bool NemotronModelBase::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return ParseQwenTokenCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
