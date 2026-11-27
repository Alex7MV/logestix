#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "lsxcommon/inference_gpu.h"   // gpu::InferenceResult
#include "lsxcommon/safetensors.h"     // SafetensorsModel
#include "lsxcommon/tokenizer.h"       // ArrowTokenizerDb
#include "lsxcommon/vram_estimate.h"   // VramProfile
#include "lsxcommon/prompt_framing.h"  // prompt framing utilities
#include "lsxcommon/fast_math_mode.h"  // FastMathMode
#include "lsxcommon/lsx_kv_cache.h"    // KvCacheLayout / KvCacheGeometry
#include "lsxcommon/token_stream.h"    // TokenChunk / TokenCallback / GenerationStream
#include "lsxcommon/sampling.h"        // SamplingParams (opt-in sampling)
#include "lsxcommon/chat_conversation.h"  // ChatConversation / ToolCall
#include "lsxcommon/tool_format.h"     // generic Hermes render/parse

namespace lsxcommon {

struct ModelInitConfig {
    std::string model_path;   // required
    std::string model_type;   // empty = autodetect from logestix/model.index.json
    int max_gen_tokens = 64;
    // Blackwell fast path (audit Phase 3, F-01): route NVFP4 GEMMs through
    // native tensor-core mxf4nvf4 on sm_120a/sm_121a. Tolerance-class
    // numerics. Legacy opt-in: fast_math=true forces the TC path (kOn);
    // fast_math=false defers to fast_math_mode below.
    bool fast_math = false;
    // Tri-state routing (gpu_perf auto-tuning): kAuto (default) follows the
    // per-GPU perf policy resolved at engine init; kOn/kOff force the
    // numeric class (only consulted when fast_math == false).
    FastMathMode fast_math_mode = FastMathMode::kAuto;
    // KV cache layout for families that keep one: [pos][head][dim] (default) or
    // [head][pos][dim]. Read once at init and immutable afterwards: the strides
    // are captured into the CUDA graph as kernel arguments, so mutating this after
    // capture would replay with the stale value silently. Hybrid families whose
    // recurrent state is not a KV cache may ignore it - IModel::KvLayout() is what
    // makes "ignored" observable instead of a silent no-op.
    //
    // pos-major is the default because it MEASURED faster end to end on Xing-4.0,
    // even though head-major reads K/V 6-10% faster in isolation at long context:
    // 119.3 vs 116.8 t/s at 2048, 96.1 vs 92.0 at 4096, 68.7 vs 64.9 at 8191. Both
    // layouts stay reachable and both are exercised by a bit-exact parity test, so
    // the decision stays revisable - but it is a decision, not an inherited
    // default. Do not flip this per family without an end-to-end measurement.
    KvCacheLayout kv_cache_layout = KvCacheLayout::kPosMajor;

    // Synthetic fixture models (unit tests that build their own Arrow files)
    // have no artifact.fb and are exempt from the storage resolve + codec gate.
    // An explicit field rather than an env var or a silent fallback: the gate
    // must not be bypassable by anything a production binary can set.
    //
    // Declared LAST on purpose: ModelInitConfig is brace-initialised
    // positionally in several TUs, so a field inserted in the middle silently
    // rebinds every later argument (that is exactly what happened when this was
    // placed after model_path).
    bool synthetic = false;
};

// Effective routing mode: the legacy bool wins (true -> kOn); otherwise the
// tri-state field decides.
inline FastMathMode EffectiveFastMathMode(const ModelInitConfig &cfg)
{
    return cfg.fast_math ? FastMathMode::kOn : cfg.fast_math_mode;
}

// Sentinel "no EOS" token id: out of any vocab range, so drivers that require
// eos_id >= 0 still run to exactly max_gen_tokens (fixed-length measurement).
inline constexpr int32_t kIgnoreEosTokenId = 0x7fffffff;

struct ModelRequest {
    std::vector<int32_t> input_ids;   // pre-tokenized prompt
    int max_gen_tokens = 64;
    // Fixed-length measurement: when true, EOS does not terminate generation,
    // so exactly `max_gen_tokens` steps are decoded and t/s is comparable
    // across numeric modes (perf/r3) regardless of where EOS lands. Used by the
    // perf baseline; streaming/selfcheck leave it false.
    bool ignore_eos = false;
    // Streaming hook: invoked with each generated TokenChunk. Passed by const
    // reference (no store/copy). nullptr = streaming disabled.
    TokenCallback token_callback = nullptr;
    // Cooperative interruption intercept. When *abort_flag turns true during
    // generation, the engine loop stops publishing new chunks and the decode
    // run aborts. Never dereferenced when null.
    const std::atomic<bool> *abort_flag = nullptr;
    // Opt-in sampling. Default greedy (temperature 0), so every existing
    // caller and every golden keeps the bit-exact argmax class.
    sample::SamplingParams sampling;
};

class IModel {
public:
    virtual ~IModel() = default;

    virtual bool Load() = 0;
    virtual gpu::InferenceResult Infer(const ModelRequest &req) = 0;
    virtual void Unload() = 0;
    virtual SafetensorsModel Inspect() = 0;
    virtual gpu::InferenceResult SelfCheck(const ModelRequest &req) = 0;

    // Peak resident weight set (pinned + arena + staged experts), in bytes.
    // Default 0 keeps the interface additive for test fakes.
    virtual std::uint64_t ResidentWeightBytes() const { return 0; }

    // VRAM requirements computed at conversion time ("vram" block of
    // model.index.json). Default: empty profile keeps test fakes additive.
    virtual const VramProfile &VramRequirements() const
    {
        return kEmptyVramProfile;
    }

    virtual const std::string &Type() const = 0;

    // The KV cache layout this instance actually uses. The knob is neutral
    // (ModelInitConfig::kv_cache_layout), so a family that does not keep a KV
    // cache - or whose attention blocking cannot use the requested layout - has
    // to answer here anyway: silently claiming the requested layout would be a
    // lie no test could catch. Default matches ModelInitConfig's.
    virtual KvCacheLayout KvLayout() const { return KvCacheLayout::kPosMajor; }
    virtual ArrowTokenizerDb &Tokenizer() = 0;

    // Token bridge (Model Ingestion Protocol stage 4): frame a raw user
    // prompt into model input ids. Each framed model owns its chat-template
    // semantics and overrides this; the default is the raw BPE encode (the
    // tokenizer auto-prepends bos_id when it defines one).
    virtual bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                     std::string_view raw_prompt,
                                     std::vector<int32_t> &out_ids) const
    {
        return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
    }

    // Tool/function-calling bridge (Model Ingestion Protocol step 4b). Renders
    // a full conversation (optionally with tool schemas) into model input ids.
    // The default flattens the transcript with role markers — including the
    // <tools> block when tool schemas are present (the generic Hermes
    // fallback) — and delegates to the single-string BuildPromptInputIds;
    // families with a native tool format override this next to their model code.
    virtual bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                           const ChatConversation &conv,
                                           std::vector<int32_t> &out_ids) const
    {
        return BuildPromptInputIds(
            tok, tools::RenderConversation(conv, !conv.tools.empty()), out_ids);
    }

    // Parses generated tokens into tool calls. Default: decode the ids (with
    // capacity retry — arrow_tokenizer_decode drops special tokens) and run the
    // generic Hermes parser. Families whose tool syntax uses special tokens
    // override this to scan token ids.
    virtual bool ParseToolCalls(const ArrowTokenizerDb &tok,
                                const std::vector<int32_t> &gen_tokens,
                                std::vector<ToolCall> &out_calls) const
    {
        out_calls.clear();
        if (gen_tokens.empty()) return true;
        std::size_t cap = 4096;
        std::string text(cap, '\0');
        for (int attempt = 0; attempt < 8; ++attempt) {
            const int rc = arrow_tokenizer_decode(tok, gen_tokens.data(),
                                                  gen_tokens.size(), text.data(),
                                                  &cap);
            if (rc == 0) {
                text.resize(cap);
                out_calls = tools::ParseToolCallsText(text);
                if (out_calls.empty())
                    out_calls = tools::ParseBareToolCallJson(text);
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

    // Concurrency groundwork (spec: leases and residency locks). Infer()
    // takes a shared lock; unload-during-inference / parallel dispatch take
    // unique. One mutex per model instance, owned by ModelBase.
    virtual std::shared_mutex &ResidencyMutex() = 0;
};

} // namespace lsxcommon