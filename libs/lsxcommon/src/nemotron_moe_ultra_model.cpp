#include "nemotron_moe_ultra_model.h"

#include "nemotron_moe_ultra_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/nemotron_ultra_tool_format.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/qwen_tool_format.h"

#include <absl/strings/str_format.h>

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace lsxcommon {

const std::string NemotronUltra550BModel::kType = nemotron_ultra::frozen::kType;

NemotronUltra550BModel::NemotronUltra550BModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
    // Synthetic fixtures may omit the tokenizer arrows.
    tokenizer_optional_ = true;
}

NemotronUltra550BModel::~NemotronUltra550BModel() { Unload(); }

bool NemotronUltra550BModel::LoadAndValidateIndex(std::string &error)
{
    if (!nemotron_ultra::LoadUltraIndex(cfg_.model_path, idx_, error)) return false;
    if (!nemotron_ultra::ValidateUltraGeometry(idx_, error)) {
        log::error(absl::StrFormat("[nemotron-ultra] frozen geometry mismatch: %s",
                                   error));
        return false;
    }
    return true;
}

bool NemotronUltra550BModel::Load()
{
    if (loaded_) return true;
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    std::string err;
    if (!LoadAndValidateIndex(err)) {
        log::error(absl::StrFormat("[nemotron-ultra] index error: %s", err));
        return AbortLoad();
    }
    vram_ = idx_.vram;

    const int kMaxSeq = 4096;
    if (!nemotron_ultra::AllocUltraSession(idx_, kMaxSeq, session_, err)) {
        log::error(absl::StrFormat("[nemotron-ultra] session alloc failed: %s", err));
        return AbortLoad();
    }
    if (!nemotron_ultra::LoadUltraWeights(cfg_.model_path, idx_, weights_, stream(),
                                          err)) {
        log::error(absl::StrFormat("[nemotron-ultra] weight load failed: %s", err));
        TeardownWeights();
        return AbortLoad();
    }

    // Routed experts: host store (mmap views) then the device slot LRU sized from
    // what the dense graph left free. 24 576 experts do not fit, by design.
    nemotron_ultra::LoadUltraExpertStore(cfg_.model_path, idx_.num_layers,
                                         idx_.num_experts, expert_cache_.store);
    expert_cache_.experts_per_layer = idx_.num_experts;
    expert_cache_.moe_index_of_layer.assign(
        static_cast<std::size_t>(idx_.num_layers), -1);
    for (int l = 0; l < idx_.num_layers; ++l)
        if (idx_.layer_types[static_cast<std::size_t>(l)] == "moe")
            expert_cache_.moe_index_of_layer[static_cast<std::size_t>(l)] =
                expert_cache_.moe_layers++;
    expert_cache_.layout =
        nemotron_ultra::PlanUltraExpertSlotFromStore(expert_cache_.store);
    if (!expert_cache_.layout.ok) {
        log::error("[nemotron-ultra] expert store empty or unreadable");
        TeardownWeights();
        return AbortLoad();
    }
    {
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) free_bytes = 0;
        expert_cache_.total_experts =
            expert_cache_.store.num_layers * expert_cache_.experts_per_layer;
        const auto per =
            static_cast<std::size_t>(nemotron_ultra::UltraExpertPerExpertBytes(
                expert_cache_.layout));
        const int slots = nemotron_ultra::UltraExpertCacheSlots(
            free_bytes, per, expert_cache_.total_experts, idx_.num_experts_per_tok,
            std::getenv("LSX_NEMOTRON_ULTRA_EXPERT_CACHE_SLOTS"));
        log::info(absl::StrFormat(
            "[nemotron-ultra] expert slots: %d of %d (%.1f MB each, %.1f GiB free)",
            slots, expert_cache_.total_experts, static_cast<double>(per) / 1e6,
            static_cast<double>(free_bytes) / (1ull << 30)));
        if (!nemotron_ultra::AllocateUltraExpertCache(expert_cache_, slots, err)) {
            log::error(absl::StrFormat("[nemotron-ultra] expert cache failed: %s",
                                       err));
            TeardownWeights();
            return AbortLoad();
        }
        const cudaError_t ce = nemotron_ultra::PrewarmUltraExpertCache(
            expert_cache_, stream(), err);
        if (ce != cudaSuccess) {
            log::error(absl::StrFormat("[nemotron-ultra] expert prewarm error: %s",
                                       err));
            TeardownWeights();
            return AbortLoad();
        }
        resident_bytes_ = weights_.resident_bytes + expert_cache_.cache_bytes;
    }

    // Governor accounting: the dense backbone is permanent, the expert slot cache
    // is a separate adaptive lease (both released by Unload/AbortLoad).
    if (!AcquirePermanentWeights(weights_.resident_bytes)) return AbortLoad();
    AcquireExpertCacheLease(expert_cache_.cache_bytes, expert_cache_.cache_bytes,
                            expert_cache_.cache_bytes);

    FinishLoad();
    log::info(absl::StrFormat(
        "[nemotron-ultra] loaded %s: %d layers, %d experts, latent %d, vocab %d",
        idx_.model.c_str(), idx_.num_layers, idx_.num_experts, idx_.moe_latent_size,
        idx_.vocab_size));
    return true;
}

void NemotronUltra550BModel::TeardownWeights()
{
    nemotron_ultra::FreeUltraExpertCache(expert_cache_);
    session_.Free();
    weights_.Free();
    TeardownContext();
}

namespace {

int ArgMaxHost(const std::vector<float> &v)
{
    int best = 0;
    for (std::size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[static_cast<std::size_t>(best)]) best = static_cast<int>(i);
    return best;
}

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

gpu::InferenceResult NemotronUltra550BModel::Infer(const ModelRequest &req)
{
    // Model independence: this model's own kernel-routing policy for the
    // duration of the call.
    lsx_nvfp4::LsxFastMathScope scope(EffectiveFastMathMode(cfg_));

    return RunUnderResidencyLock([&] {
        gpu::InferenceResult r;
        if (!ValidateInferRequest(req, r)) return r;

        lsxcommon::GenerationStream gs;
        gs.on_token = req.token_callback ? &req.token_callback : nullptr;
        gs.abort_flag = req.abort_flag;

        nemotron_ultra::ResetUltraSession(session_, stream());
        std::vector<float> logits(static_cast<std::size_t>(idx_.vocab_size));

        ScopedCudaEvents ev;
        cudaEventRecord(ev.start, stream());

        int pos = 0;
        for (int32_t id : req.input_ids) {
            const cudaError_t e = nemotron_ultra::UltraForwardToken(
                idx_, weights_, session_, expert_cache_, id, pos, true, stream());
            if (e != cudaSuccess) {
                r.error_msg = std::string("forward failed: ") +
                              cudaGetErrorString(e);
                return r;
            }
            ++pos;
        }
        cudaEventRecord(ev.first, stream());

        auto &sampler = sample::SamplerFor(req.sampling);
        int generated = 0;
        for (int step = 0; step < req.max_gen_tokens; ++step) {
            if (cudaMemcpyAsync(logits.data(), session_.head_logits,
                                logits.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, stream()) != cudaSuccess ||
                cudaStreamSynchronize(stream()) != cudaSuccess) {
                r.error_msg = "logits readback failed";
                return r;
            }
            int next = ArgMaxHost(logits);
            if (!sampler.IsGreedy()) {
                float sampled_maxv = 0.0f;
                const cudaError_t se = sample::SampleToHost(
                    stream(), sampler, session_.head_logits,
                    static_cast<int>(logits.size()), sample::LogitType::kF32, &next,
                    &sampled_maxv);
                if (se != cudaSuccess) {
                    r.error_msg = "sampling failed";
                    return r;
                }
            }
            r.gen_tokens.push_back(next);
            ++generated;
            if (step == 0) {
                r.first_gen_tok_id = next;
                r.first_gen_maxv = logits[static_cast<std::size_t>(next)];
            }
            r.final_logits = logits;
            // The checkpoint declares TWO stops ([2, 11]); stopping only on the
            // first lets the model generate past its own turn end.
            bool hit_stop = false;
            for (int stop : idx_.eos_token_ids)
                if (next == stop) hit_stop = true;
            if (!req.ignore_eos && hit_stop) {
                r.eos_terminated = true;
                break;
            }
            if (step + 1 < req.max_gen_tokens) {
                const cudaError_t e = nemotron_ultra::UltraForwardToken(
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
                    logits[static_cast<std::size_t>(next)]);
            if (gs.Aborted()) break;
        }
        r.num_tokens = generated;
        r.num_layers = idx_.num_layers;
        r.weight_resident_bytes = resident_bytes_;
        r.kv_slots_used = pos;
        if (tok_db_ && !r.gen_tokens.empty())
            r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
        cudaEventRecord(ev.stop, stream());
        cudaEventSynchronize(ev.stop);
        float ttft = 0.0f, total = 0.0f;
        cudaEventElapsedTime(&ttft, ev.start, ev.first);
        cudaEventElapsedTime(&total, ev.start, ev.stop);
        r.ttft_ms = static_cast<double>(ttft);
        r.total_time_ms = static_cast<double>(total);
        r.tokens_per_sec = total > 0.0f
                               ? static_cast<double>(generated) * 1000.0 /
                                     static_cast<double>(total)
                               : 0.0;
        log::info(absl::StrFormat(
            "[nemotron-ultra] expert cache: hits=%lld misses=%lld slots=%d",
            static_cast<long long>(expert_cache_.device.hits),
            static_cast<long long>(expert_cache_.device.misses),
            expert_cache_.device.slots));
        r.ok = true;
        return r;
    });
}

gpu::InferenceResult NemotronUltra550BModel::SelfCheck(const ModelRequest &req)
{
    // Unified Paris harness: the strict 1->2-token "Paris" contract plus the
    // cross-model logits_hash evidence (numeric-contract §1). frame_prompt stays
    // OFF: the suite feeds its own control-prompt ids.
    return runtime::RunParisSelfCheck(*this, req);
}

bool NemotronUltra550BModel::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                                 std::string_view raw_prompt,
                                                 std::vector<int32_t> &out_ids) const
{
    // A chat model must never see an unframed prompt: render the single turn
    // through the same native template the conversation path uses.
    ChatConversation conv;
    ChatMessage m;
    m.role = "user";
    m.content = std::string(raw_prompt);
    conv.messages.push_back(std::move(m));
    std::string frame;
    if (nemotron_ultra::RenderUltraConversation(conv, frame)) {
        if (AppendQwenChatIds(tok, frame, out_ids)) return true;
        return prompt::EncodeRawPrompt(tok, frame, out_ids);
    }
    return false;
}

bool NemotronUltra550BModel::BuildConversationInputIds(
    const ArrowTokenizerDb &tok, const ChatConversation &conv,
    std::vector<int32_t> &out_ids) const
{
    std::string frame;
    if (!nemotron_ultra::RenderUltraConversation(conv, frame)) return false;
    if (AppendQwenChatIds(tok, frame, out_ids)) return true;
    return prompt::EncodeRawPrompt(tok, frame, out_ids);
}

bool NemotronUltra550BModel::ParseToolCalls(const ArrowTokenizerDb &tok,
                                            const std::vector<int32_t> &gen_tokens,
                                            std::vector<ToolCall> &out_calls) const
{
    // The `<tool_call><function=…><parameter=…>` wire is byte-identical to the
    // 30B sibling's (verified by the template diff), so the parser is shared.
    return ParseQwenTokenCalls(tok, gen_tokens, out_calls);
}

} // namespace lsxcommon
