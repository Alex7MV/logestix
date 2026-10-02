// Glm5NextFlashModel implementation (Kimi/Qwen38 ModelBase pattern).
#include "json_escape.h"
#include "lsxcommon/gpu_perf.h"
#include "glm_moe_flash_model.h"

#include "glm_moe_flash_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/tool_format.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

namespace lsxcommon {
using lsxcommon::json::JsonEscape;

const std::string Glm5NextFlashModel::kType = "glm-5.3-flash";

namespace {

// Device memory reserved for generation-time allocations on top of the
// expert cache: fp16 embed table (V*H*2B ~ 0.63 GB), head/logits, KDA/DSA
// per-step arenas, and CUDA context overhead.
constexpr size_t kFlashGenReserveBytes = size_t{3} << 30;

int FlashExpertSlotBytes(const qwen_moe::QwenMoeExpertStore &store)
{
    size_t max_pack = 0, max_scale = 0;
    for (int l = 0; l < store.num_layers; ++l) {
        for (const auto &r : store.records[static_cast<size_t>(l)]) {
            max_pack = std::max(max_pack,
                                static_cast<size_t>(r.w1_bytes + r.w2_bytes +
                                                    r.w3_bytes));
            max_scale = std::max(max_scale,
                                 static_cast<size_t>(r.s1_bytes + r.s2_bytes +
                                                     r.s3_bytes));
        }
    }
    constexpr size_t kAlign = 256;
    const size_t pack = (max_pack + kAlign - 1) / kAlign * kAlign;
    const size_t scale = (max_scale + kAlign - 1) / kAlign * kAlign;
    return static_cast<int>(pack + scale);
}

int ResolveExpertSlots(const qwen_moe::QwenMoeExpertStore &store)
{
    const int max_slots = store.num_layers * store.num_experts;
    int slots = 64;  // legacy floor: top-k per-layer working set headroom
    if (const char *env = std::getenv("LSX_FLASH_SLOTS")) {
        const int parsed = std::atoi(env);
        if (parsed > 0) slots = parsed;
    } else {
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            size_t budget =
                free_bytes > kFlashGenReserveBytes
                    ? free_bytes - kFlashGenReserveBytes
                    : 0;
            // Perf policy: scale the VRAM budget by the resolved slot
            // percentage (default 100 = unchanged behavior).
            const int pct = perf::CurrentKnobs().expert_slots_pct;
            if (pct < 100) budget = budget / 100u * static_cast<unsigned>(pct);
            const int slot_bytes = FlashExpertSlotBytes(store);
            if (slot_bytes > 0) {
                const int by_vram =
                    static_cast<int>(budget / static_cast<size_t>(slot_bytes));
                slots = std::max(slots, std::min(by_vram, max_slots));
            }
        }
    }
    return std::min(slots, max_slots);
}

} // namespace

Glm5NextFlashModel::Glm5NextFlashModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
    // Both toggles are process-wide (see Qwen35MoeModel): without them every
    // NVFP4 GEMM runs the shared-memory tile kernel even at seq==1 decode.
    const FastMathMode fm = EffectiveFastMathMode(cfg_);
    if (fm != FastMathMode::kAuto) {
        lsxcommon::lsx_nvfp4::SetLsxTcFastMath(fm == FastMathMode::kOn);
        lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(fm == FastMathMode::kOn);
    }
}

Glm5NextFlashModel::~Glm5NextFlashModel() { Unload(); }

bool Glm5NextFlashModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    if (!EnsureContext()) {
        log::error("[glm-flash-model] stream init failed");
        return AbortLoad();
    }

    std::string error;
    if (!glm_moe_flash::LoadGlmFlashDenseWeights(cfg_.model_path, pw_,
                                                 stream(), error)) {
        log::error("[glm-flash-model] dense load failed: " + error);
        return AbortLoad();
    }
    if (!glm_moe_flash::LoadGlmFlashExpertStore(cfg_.model_path, experts_,
                                                error)) {
        log::error("[glm-flash-model] expert store load failed: " + error);
        return AbortLoad();
    }
    // LRU device cache sized to the free-VRAM budget; slots bounded by the
    // top-k resident working set per layer.
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int slots = ResolveExpertSlots(experts_);
        log::info(absl::StrCat(
            "[glm-flash-model] expert cache slots=", slots, " (~",
            static_cast<long long>(static_cast<int64_t>(slots) *
                                   FlashExpertSlotBytes(experts_) >> 20),
            " MiB); pre-cache free=", static_cast<long long>(free_b >> 20),
            " MiB total=", static_cast<long long>(total_b >> 20), " MiB"));
        if (!qwen_moe::AllocateQwenMoeExpertCache(experts_, slots, cache_,
                                                  error)) {
            log::error("[glm-flash-model] expert cache alloc failed: "
                       + error);
            return AbortLoad();
        }
    }

    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

    resident_bytes_ = pw_.resident_bytes;
    log::info("[glm-flash-model] loaded: resident=" +
              std::to_string(resident_bytes_) + " layers=" +
              std::to_string(pw_.layers.size()));
    FinishLoad();
    return true;
}

gpu::InferenceResult Glm5NextFlashModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_ || pw_.layers.empty()) {
        r.error_msg = "model not loaded (call Load(); weights are no longer "
                      "reloaded per Infer call)";
        r.ok = false;
        return r;
    }

    // Model independence: assert THIS model's kernel-routing policy for the
    // duration of the call (process-global flags would otherwise leak the
    // last-constructed model's fast_math across the registry).
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    qwen::QwenGenerationResult gen;
    sample::GreedySampler greedy;
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    const cudaError_t err = RunUnderResidencyLock([&] {
        std::vector<int> eos_ids =
            req.ignore_eos ? std::vector<int>{lsxcommon::kIgnoreEosTokenId} : pw_.top.eos_ids;
        return glm_moe_flash::RunGlmFlashGeneration(
            stream(), pw_, experts_, cache_, req.input_ids, req.max_gen_tokens,
            eos_ids, gen, greedy, &gs);
    });
    if (err != cudaSuccess || !gen.ok) {
        r.error_msg = runtime::CombineGenError(err, gen.error_msg);
        r.ok = false;
        return r;
    }

    runtime::FillInferenceResult(
        r, {.tokens = &gen.tokens,
            .final_logits = &gen.final_logits,
            .ttft_ms = gen.ttft_ms,
            .total_ms = gen.total_ms,
            .error_msg = &gen.error_msg,
            .eos_terminated = gen.eos_terminated,
            .kv_slots_used = -1,
            .first_maxv = 0.0f,
            .resident_bytes = resident_bytes_},
        glm_moe_flash::frozen::kLayers);
    log::info(absl::StrCat(
        "[glm-flash-model] expert cache: hits=", cache_.hits,
        " misses=", cache_.misses,
        " hit_rate=",
        (cache_.hits + cache_.misses) > 0
            ? absl::StrFormat("%.1f%%",
                              100.0 * static_cast<double>(cache_.hits) /
                                  static_cast<double>(cache_.hits +
                                                      cache_.misses))
            : std::string("n/a")));
    if (tok_db_ && !r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
    return r;
}

gpu::InferenceResult Glm5NextFlashModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

void Glm5NextFlashModel::TeardownWeights()
{
    pw_.Free();
    experts_.Free();
    qwen_moe::FreeQwenMoeExpertCache(cache_);
}

// ── GLM-5.3-Flash native tool format ────────────────────────────────────────
// Verified against chat_template.jinja:
//   [gMASK]<sop><|system|>Reasoning Effort: Max<|system|>\n# Tools …<tools>…
//   <|user|>{content}\n
//   <|assistant|>\n<think></think>{content}<tool_call>NAME<arg_key>K</arg_key>
//       <arg_value>V</arg_value></tool_call>\n
//   <|observation|><tool_response>{content}</tool_response>
//   generation prompt: <|assistant|><think>

namespace {

bool AppendGlmEncoded(const ArrowTokenizerDb &tok, std::string_view text,
                      std::vector<int32_t> &ids)
{
    for (size_t cap = 256; cap <= 65536; cap *= 2) {
        std::vector<int32_t> buf(cap, 0);
        size_t enc_cap = cap;
        const int32_t n = arrow_tokenizer_encode(tok, text.data(), text.size(),
                                                 buf.data(), &enc_cap);
        if (n == -2) continue;
        if (n < 0) return false;
        const size_t begin = (tok.bos_id >= 0 && n > 0) ? 1 : 0;
        ids.insert(ids.end(), buf.begin() + begin, buf.begin() + n);
        return true;
    }
    return false;
}

constexpr const char *kGlmFlashSentinels[] = {
    "[gMASK]",  "<sop>",         "<|system|>",
    "<|user|>", "<|assistant|>", "<|observation|>",
};

bool AppendGlmIds(const ArrowTokenizerDb &tok, const std::string &text,
                  std::vector<int32_t> &ids)
{
    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end > run &&
            !AppendGlmEncoded(tok,
                              std::string_view(text).substr(run, end - run),
                              ids))
            return false;
        return true;
    };
    while (i < text.size()) {
        bool matched = false;
        for (const char *s : kGlmFlashSentinels) {
            const size_t n = std::strlen(s);
            if (text.compare(i, n, s) == 0) {
                if (!flush(i)) return false;
                const int32_t id = arrow_tokenizer_lookup(tok, s, n);
                if (id < 0) return false;
                ids.push_back(id);
                i += n;
                run = i;
                matched = true;
                break;
            }
        }
        if (!matched) ++i;
    }
    return flush(text.size());
}

std::string_view GlmTrim(std::string_view s)
{
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' ||
                          s.front() == '\r' || s.front() == '\n'))
        s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' ||
                          s.back() == '\r' || s.back() == '\n'))
        s.remove_suffix(1);
    return s;
}

std::string GlmSerialize(simdjson::dom::element e);

std::string GlmSerialize(simdjson::dom::element e)
{
    using simdjson::dom::element_type;
    switch (e.type()) {
        case element_type::OBJECT: {
            std::string out = "{";
            bool first = true;
            for (auto [k, v] : e.get_object()) {
                if (!first) out += ",";
                first = false;
                out += "\"" + JsonEscape(k) + "\":" + GlmSerialize(v);
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
                out += GlmSerialize(v);
            }
            out += "]";
            return out;
        }
        case element_type::STRING:
            return "\"" + JsonEscape(e.get_string().value_unsafe()) + "\"";
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

} // namespace

bool Glm5NextFlashModel::RenderGlmFlashConversation(const ChatConversation &conv,
                                                    std::string &out)
{
    out.clear();
    // Break control markers inside untrusted content (prompt injection).
    static const std::vector<std::string_view> kMarkers = [] {
        std::vector<std::string_view> m;
        for (const char *s : kGlmFlashSentinels) m.emplace_back(s);
        return m;
    }();
    const auto esc = [](std::string_view s) {
        return tools::NeutralizeControlMarkers(s, kMarkers);
    };
    out += "[gMASK]<sop>";
    out += "<|system|>Reasoning Effort: Max";
    if (!conv.tools.empty()) {
        out += "<|system|>\n# Tools\n\n"
               "You may call one or more functions to assist with the user "
               "query.\n\n"
               "You are provided with function signatures within "
               "<tools></tools> XML tags:\n<tools>";
        for (const ToolSpec &t : conv.tools) {
            out += "\n{\"name\":\"";
            out += JsonEscape(esc(t.name));
            out += "\",\"description\":\"";
            out += JsonEscape(esc(t.description));
            out += "\",\"parameters\":";
            out += t.parameters_json.empty() ? std::string("{}")
                                             : esc(t.parameters_json);
            out += "}";
        }
        out += "\n</tools>\n\n"
               "For each function call, output the function name and arguments "
               "within the following XML format:\n"
               "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>"
               "<arg_value>{arg-value-1}</arg_value>...</tool_call>";
    }

    bool in_observation = false;
    simdjson::dom::parser parser;
    for (const ChatMessage &m : conv.messages) {
        if (m.role == "system") {
            out += "<|system|>" + esc(m.content) + "\n";
            in_observation = false;
        } else if (m.role == "user") {
            out += "<|user|>" + esc(m.content) + "\n";
            in_observation = false;
        } else if (m.role == "assistant") {
            out += "<|assistant|>\n<think></think>";
            if (!m.content.empty()) out += esc(m.content);
            for (const ToolCall &c : m.tool_calls) {
                out += "<tool_call>" + esc(c.name);
                if (!c.arguments_json.empty()) {
                    simdjson::dom::element args;
                    simdjson::padded_string padded(c.arguments_json);
                    if (parser.parse(padded).get(args) == simdjson::SUCCESS &&
                        args.type() == simdjson::dom::element_type::OBJECT) {
                        for (auto [k, v] : args.get_object()) {
                            out += "<arg_key>" + esc(k) + "</arg_key>";
                            out += "<arg_value>";
                            if (v.type() == simdjson::dom::element_type::STRING)
                                out += esc(v.get_string().value_unsafe());
                            else
                                out += GlmSerialize(v);
                            out += "</arg_value>";
                        }
                    }
                }
                out += "</tool_call>";
            }
            out += "\n";
            in_observation = false;
        } else if (m.role == "tool") {
            if (!in_observation) {
                out += "<|observation|>";
                in_observation = true;
            }
            out += "<tool_response>" + esc(m.content) + "</tool_response>";
        } else {
            return false;
        }
    }
    out += "<|assistant|><think>";
    return true;
}

std::vector<ToolCall> Glm5NextFlashModel::ParseGlmFlashToolCallText(
    std::string_view text)
{
    std::vector<ToolCall> calls;
    simdjson::dom::parser parser;
    constexpr std::string_view kOpen = "<tool_call>";
    constexpr std::string_view kClose = "</tool_call>";
    constexpr std::string_view kArgKey = "<arg_key>";
    constexpr std::string_view kArgKeyEnd = "</arg_key>";
    constexpr std::string_view kArgVal = "<arg_value>";
    constexpr std::string_view kArgValEnd = "</arg_value>";
    size_t pos = 0;
    while (true) {
        const size_t b = text.find(kOpen, pos);
        if (b == std::string_view::npos) break;
        const size_t e = text.find(kClose, b + kOpen.size());
        if (e == std::string_view::npos) break;
        const std::string_view body =
            text.substr(b + kOpen.size(), e - (b + kOpen.size()));
        pos = e + kClose.size();

        std::string args = "{";
        bool first = true;
        size_t kp = 0;
        while (true) {
            const size_t kb = body.find(kArgKey, kp);
            if (kb == std::string_view::npos) break;
            const size_t kstart = kb + kArgKey.size();
            const size_t kend = body.find(kArgKeyEnd, kstart);
            if (kend == std::string_view::npos) break;
            const std::string key(
                GlmTrim(body.substr(kstart, kend - kstart)));
            const size_t vb = body.find(kArgVal, kend + kArgKeyEnd.size());
            if (vb == std::string_view::npos) break;
            const size_t vstart = vb + kArgVal.size();
            const size_t vend = body.find(kArgValEnd, vstart);
            if (vend == std::string_view::npos) break;
            const std::string_view raw =
                GlmTrim(body.substr(vstart, vend - vstart));
            kp = vend + kArgValEnd.size();
            if (!first) args += ",";
            first = false;
            args += "\"" + JsonEscape(key) + "\":";
            // The GLM XML format carries no string flag, so a string value that
            // is itself valid JSON (e.g. "42") is inherently ambiguous and
            // round-trips as that JSON type.
            simdjson::dom::element ev;
            simdjson::padded_string padded(raw);
            if (parser.parse(padded).get(ev) == simdjson::SUCCESS)
                args += GlmSerialize(ev);
            else
                args += "\"" + JsonEscape(raw) + "\"";
        }
        args += "}";

        // Name is everything before the first argument or the close tag.
        const size_t name_end = body.find(kArgKey);
        const std::string name(GlmTrim(
            name_end == std::string_view::npos ? body : body.substr(0, name_end)));

        ToolCall call;
        call.name = name;
        call.arguments_json = std::move(args);
        call.id = "call_" + std::to_string(calls.size());
        calls.push_back(std::move(call));
    }
    return calls;
}

bool Glm5NextFlashModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::string text;
    if (!RenderGlmFlashConversation(conv, text)) return false;
    out_ids.clear();
    if (arrow_tokenizer_lookup(tok, "<|system|>", 10) < 0 ||
        arrow_tokenizer_lookup(tok, "<|user|>", 8) < 0)
        return BuildPromptInputIds(tok, text, out_ids);
    return AppendGlmIds(tok, text, out_ids);
}

bool Glm5NextFlashModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;
    std::size_t cap = 4096;
    std::string text(cap, '\0');
    bool decoded = false;
    for (int attempt = 0; attempt < 8; ++attempt) {
        const int rc = arrow_tokenizer_decode(tok, gen_tokens.data(),
                                              gen_tokens.size(), text.data(),
                                              &cap);
        if (rc == 0) {
            text.resize(cap);
            decoded = true;
            break;
        }
        if (rc == -2 && cap > text.size() && cap <= (1u << 20)) {
            text.resize(cap);
            continue;
        }
        break;
    }
    if (!decoded) return false;
    out_calls = ParseGlmFlashToolCallText(text);
    if (out_calls.empty()) out_calls = tools::ParseToolCallsText(text);
    if (out_calls.empty()) out_calls = tools::ParseBareToolCallJson(text);
    return true;
}

} // namespace lsxcommon
