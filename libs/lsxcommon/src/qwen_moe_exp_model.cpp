#include "qwen_moe_exp_model.h"
#include "qwen_moe_exp_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/resource_governor.h"
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/qwen_tool_format.h"

#include <absl/strings/str_cat.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace lsxcommon {

const std::string Qwen38FlashNextModel::kType = "qwen3.8-flash-next";

namespace {

constexpr int kQsaLayerCount = 12;  // frozen: layers 3,7,...,47

std::uint64_t MsSince(std::chrono::steady_clock::time_point t0)
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0)
            .count());
}

} // namespace

Qwen38FlashNextModel::Qwen38FlashNextModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
    const FastMathMode fm = EffectiveFastMathMode(cfg_);
    if (fm != FastMathMode::kAuto) {
        qwen::SetQwenTcFastMath(fm == FastMathMode::kOn);
        qwen::SetQwenSeq1Gemv(fm == FastMathMode::kOn);
    }
}

Qwen38FlashNextModel::~Qwen38FlashNextModel() = default;

bool Qwen38FlashNextModel::ValidateGeometry(
    const qwen_moe_exp::Qwen4ExpTopology &top) const
{
    using namespace qwen_moe_exp::frozen;
    if (SyntheticGeometryAllowed()) return true;

    const GeoFieldCheck fields[] = {
        {"hidden_size", kHidden, top.hidden_size},
        {"num_layers", kLayers, top.num_layers},
        {"num_attention_heads", kHeads, top.num_attention_heads},
        {"num_key_value_heads", kKVHeads, top.head_dim > 0
                                              ? top.num_key_value_heads
                                              : top.num_key_value_heads},
        {"head_dim", kHeadDim, top.head_dim},
        {"vocab_size", kVocab, top.vocab_size},
        {"num_experts", kNumExperts, top.num_experts},
        {"num_experts_per_tok", kTopK, top.num_experts_per_tok},
        {"moe_intermediate_size", kMoeIntermediate, top.moe_intermediate_size},
        {"shared_expert_intermediate_size", kSharedIntermediate,
         top.shared_expert_intermediate_size},
        {"hc_count", kHcCount, top.hc_count},
        {"hc_lowrank", kHcLowrank, top.hc_lowrank},
        {"indexer_budget", kIndexerBudget, top.indexer_budget},
        {"indexer_compress_ratio", kIndexerCompress, top.indexer_compress_ratio},
        {"ngram_size", kPleNgramSize, top.ngram_size},
        {"split_ngram_parts", kPleSplitParts, top.split_ngram_parts},
    };
    if (!CheckGeometryFields("qwen38-model", fields, 16)) return false;
    for (int l = 0; l < top.num_layers; ++l) {
        const std::string want =
            (l % kFullAttnInterval == kFullAttnInterval - 1) ? "full_attention"
                                                             : "linear_attention";
        if (top.layer_types[static_cast<size_t>(l)] != want) {
            log::error(absl::StrCat("[qwen38-model] layer_types[", l,
                                    "] != ", want));
            return false;
        }
    }
    return true;
}

void Qwen38FlashNextModel::TeardownWeights()
{
    qwen_moe_exp::FreeQwen4ExpPinnedWeights(pw_);
    qwen_moe_exp::FreeQwen4ExpExpertCache(cache_);
    experts_.Free();
    for (auto &st : qsa_) qwen_moe_exp::ExpQsaIndexerFree(st);
    qsa_.clear();
    qwen_moe_exp::ExpPleStateFree(ple_state_);
    ple_shards_.clear();  // keepalive releases the Arrow mmaps
    kv_.reset();
    session_tail_.clear();
    TeardownContext();
}

bool Qwen38FlashNextModel::Load()
{
    Unload();
    if (!BeginLoad(cfg_.model_path)) return false;

    if (!EnsureContext()) {
        log::error("[qwen38-model] stream init failed");
        return AbortLoad();
    }

    std::string err;
    const auto t_dense = std::chrono::steady_clock::now();
    if (!qwen_moe_exp::LoadQwen4ExpDenseWeights(cfg_.model_path, pw_,
                                                stream(), err)) {
        log::error("[qwen38-model] dense load failed: " + err);
        return AbortLoad();
    }
    log::info(absl::StrCat("[qwen38-model] dense load ", MsSince(t_dense),
                           " ms"));
    if (!ValidateGeometry(pw_.exp_top)) return AbortLoad();

    const auto t_experts = std::chrono::steady_clock::now();
    if (!qwen_moe_exp::LoadQwen4ExpExpertStore(cfg_.model_path, experts_,
                                               err)) {
        log::error("[qwen38-model] expert store load failed: " + err);
        return AbortLoad();
    }
    log::info(absl::StrCat("[qwen38-model] expert store ",
                           MsSince(t_experts), " ms"));

    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();

    // LRU slot budget: the 48x512 expert pool cannot be fully resident on a
    // 32 GiB board; ExpertCacheSlots sizes the LRU from free VRAM.
    const int total_experts = experts_.num_layers * experts_.num_experts;
    int slots = total_experts;
    bool explicit_slots = false;
    if (const char *env_slots =
            std::getenv("LSX_QWEN38_EXPERT_CACHE_SLOTS")) {
        const int v = std::atoi(env_slots);
        if (v > 0) {
            slots = std::min(v, total_experts);
            explicit_slots = true;
        }
    }
    constexpr size_t kArenaAlign = 256;
    const auto align_up = [](size_t v) {
        return (v + kArenaAlign - 1) & ~(kArenaAlign - 1);
    };
    size_t max_pack = 0, max_scale = 0;
    for (int l = 0; l < experts_.num_layers; ++l) {
        for (const auto &rec : experts_.records[static_cast<size_t>(l)]) {
            max_pack = std::max(max_pack,
                                static_cast<size_t>(rec.w1_bytes +
                                                    rec.w2_bytes +
                                                    rec.w3_bytes));
            max_scale = std::max(max_scale,
                                 static_cast<size_t>(rec.s1_bytes +
                                                     rec.s2_bytes +
                                                     rec.s3_bytes));
        }
    }
    const size_t per_expert = align_up(max_pack) + align_up(max_scale);
    // Largest supported generation context: the indexer budget caps retrieval
    // at 2048 tokens; +64 slack for prompt/decode overhead. The LRU reserve
    // must cover the context-scaled arena + KV (max-gen-2048 OOM root cause).
    const int kSizingMaxSeq = qwen_moe_exp::frozen::kIndexerBudget + 64;
    if (!explicit_slots) {
        size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
            // The generation arena and the fp16 embed table are process-lifetime
            // buffers (see ExpGenScratch), so a second Load in the same process
            // finds them already inside the free-VRAM complement. Pass them so
            // the context reserve is not charged twice — the harness runs the
            // "perf" and "r3" legs as two Loads in one target.
            const std::size_t resident =
                qwen_moe_exp::GenContextResidentBytes();
            slots = qwen_moe_exp::SizeExpertCacheSlots(
                free_bytes, per_expert, total_experts, kSizingMaxSeq,
                perf::CurrentKnobs().expert_slots_pct, resident);
            if (slots > 0 && slots < total_experts)
                log::info(absl::StrCat(
                    "[qwen38-model] free=", free_bytes >> 20, "MiB, resident ",
                    resident >> 20, "MiB of gen context, ", slots, "/",
                    total_experts,
                    " experts resident; LRU covers the rest"));
        }
    }
    if (slots <= 0) slots = 4;
    AcquireExpertCacheLease(
        static_cast<std::uint64_t>(slots) * per_expert, per_expert,
        static_cast<std::uint64_t>(total_experts) * per_expert);
    while (slots >= 4 &&
           !qwen_moe_exp::AllocateQwen4ExpExpertCache(experts_, slots, cache_,
                                                      err)) {
        slots /= 2;
        cache_ = {};
    }
    if (!cache_.d_base) {
        log::error("[qwen38-model] expert cache alloc failed: " + err);
        return AbortLoad();
    }

    // PLE shards (mmap-only) + incremental states.
    ple_shards_.resize(
        static_cast<size_t>(std::max(1, pw_.exp_top.split_ngram_parts)));
    if (!qwen_moe_exp::LoadQwen4ExpPleShards(cfg_.model_path, pw_,
                                             ple_shards_.data(), err)) {
        log::error("[qwen38-model] PLE shard load failed: " + err);
        return AbortLoad();
    }
    qsa_.resize(kQsaLayerCount);
    for (auto &st : qsa_) {
        if (qwen_moe_exp::ExpQsaIndexerAlloc(st, 262144, 128) !=
            cudaSuccess) {
            log::error("[qwen38-model] QSA indexer state alloc failed");
            return AbortLoad();
        }
    }
    if (qwen_moe_exp::ExpPleStateAlloc(
            ple_state_, qwen_moe_exp::frozen::kHidden,
            qwen_moe_exp::frozen::kHcCount,
            qwen_moe_exp::frozen::kPleConvStateLen) != cudaSuccess) {
        log::error("[qwen38-model] PLE state alloc failed");
        return AbortLoad();
    }

    FinishLoad();
    log::info(absl::StrCat("[qwen38-model] loaded, resident ",
                           pw_.resident_bytes >> 20, " MiB, cache slots ",
                           slots));
    return true;
}

gpu::InferenceResult Qwen38FlashNextModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    if (!ContextReady()) {
        r.error_msg = "execution context missing";
        return r;
    }

    // Model independence: this model's own kernel-routing policy for the
    // duration of the call (see LsxFastMathScope in lsx_inference_nvfp4.h).
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    if (!pw_.exp_top.layer_types.empty() && pw_.exp_top.hidden_size <= 0) {
        r.error_msg = "topology not parsed";
        return r;
    }
    qwen::GenerationResult gen;
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;
    const int eos_id = (req.ignore_eos || !tok_db_) ? lsxcommon::kIgnoreEosTokenId : tok_db_->eos_id;
    int kv_slots = 0;
    // Incremental session policy (mirrors Qwen35MoeModel), entirely inside one
    // shared-lock scope: the KV state is created, decoded through and
    // published under the lock. The lock is shared, so it orders this request against Unload()'s unique
    // lock and does NOT serialise Infer against Infer - two concurrent
    // Infer calls on one instance both hold it. One Infer per instance at
    // a time is a precondition (OPEN_TASKS E1b), and that is what keeps
    // them from freeing each other's cache.
    cudaError_t e = RunUnderResidencyLock([&] {
        const qwen::SessionDecision session = qwen::DecideSession(
            kv_ && kv_->allocated(), kv_ ? kv_->last_tok : -1, session_tail_,
            req.input_ids);
        if (!session.is_continuation) {
            kv_.reset();
            kv_ = std::make_unique<qwen::KvCache>();
            session_tail_.clear();
        }
        const cudaError_t ge = qwen_moe_exp::RunExpGeneration(
            stream(), pw_, experts_, cache_, ple_state_, qsa_, ple_shards_,
            session.gen_input, req.max_gen_tokens, eos_id, gen,
            lsxcommon::sample::SamplerFor(req.sampling), kv_.get(), &gs);
        if (ge != cudaSuccess || !gen.ok) {
            kv_.reset();
            session_tail_.clear();
            return ge;
        }
        kv_slots = (kv_ && kv_->allocated())
                       ? kv_->kv_rows + 1
                       : static_cast<int>(gen.tokens.size()) +
                             static_cast<int>(req.input_ids.size());
        session_tail_ = gen.tokens;
        return cudaSuccess;
    });
    if (e != cudaSuccess || !gen.ok) {
        r.error_msg = runtime::CombineGenError(e, gen.error_msg);
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
            .kv_slots_used = kv_slots,
            .first_maxv = 0.f,
            .resident_bytes = gen.resident_bytes,
            .expert_cache_slots = cache_.slots},
        pw_.top.num_layers);
    if (tok_db_ && !r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
    return r;
}

gpu::InferenceResult Qwen38FlashNextModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

bool Qwen38FlashNextModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::string text;
    // The checkpoint's own chat_template.jinja differs from the Qwen3.5/3.6 one
    // in three byte-level ways (verified against the template rendered by
    // transformers 5.15.1): every assistant history turn carries the closed
    // think block, the tool definitions are serialised like jinja's `tojson`,
    // and message bodies are trimmed. Framing a 3.8 model with the 3.5 wire is
    // a distribution shift it never saw in training, so the deltas are carried
    // explicitly by the named policy (see kQwen38RenderPolicy).
    if (!RenderQwenConversation(conv, text, kQwen38RenderPolicy)) return false;
    if (!AppendQwenChatIds(tok, text, out_ids))
        return BuildPromptInputIds(tok, text, out_ids);
    return true;
}

bool Qwen38FlashNextModel::ParseToolCalls(
    const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
    std::vector<ToolCall> &out_calls) const
{
    return ParseQwenTokenCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
