#include "nemotron_moe_model.h"
#include "nemotron_moe_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/prompt_framing.h"

#include <absl/strings/str_format.h>

#include <cstdlib>

namespace lsxcommon {

const std::string NemotronHModel::kType = nemotron_moe::frozen::kType;

NemotronHModel::NemotronHModel(ModelInitConfig cfg) : ModelBase(std::move(cfg))
{
    // Synthetic fixtures may omit the tokenizer arrows.
    tokenizer_optional_ = true;
}

NemotronHModel::~NemotronHModel() { Unload(); }

bool NemotronHModel::ValidateGeometry(
    const nemotron_moe::NemotronIndex &idx) const
{
    std::string why;
    if (!nemotron_moe::ValidateNemotronGeometry(idx, why)) {
        log::error(absl::StrFormat(
            "[nemotron] frozen geometry mismatch: %s", why));
        return false;
    }
    return true;
}

bool NemotronHModel::Load()
{
    if (loaded_) return true;
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    std::string err;
    if (!nemotron_moe::LoadNemotronIndex(cfg_.model_path, idx_, err)) {
        log::error(absl::StrFormat("[nemotron] index error: %s", err));
        return AbortLoad();
    }
    if (!ValidateGeometry(idx_)) return AbortLoad();
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

    FinishLoad();
    log::info(absl::StrFormat(
        "[nemotron] loaded %s: %d layers, %d experts, vocab %d",
        idx_.model.c_str(), idx_.num_layers, idx_.num_experts, idx_.vocab_size));
    return true;
}

void NemotronHModel::TeardownWeights()
{
    nemotron_moe::FreeNmExpertCache(expert_cache_);
    session_.Free();
    weights_.Free();
}

namespace {

int ArgMaxHost(const std::vector<float> &v)
{
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = static_cast<int>(i);
    return best;
}

} // namespace

gpu::InferenceResult NemotronHModel::Infer(const ModelRequest &req)
{
    lsx_nvfp4::LsxFastMathScope scope(EffectiveFastMathMode(cfg_));
    return RunUnderResidencyLock([&] {
        gpu::InferenceResult r;
        if (!ValidateInferRequest(req, r)) return r;

        nemotron_moe::ResetNemotronSession(session_, stream());
        std::vector<float> logits(static_cast<size_t>(idx_.vocab_size));

        cudaEvent_t ev0 = nullptr, ev_first = nullptr, ev1 = nullptr;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev_first);
        cudaEventCreate(&ev1);
        cudaEventRecord(ev0, stream());

        int pos = 0;
        for (int32_t id : req.input_ids) {
            cudaError_t e = nemotron_moe::NemotronForwardToken(
                idx_, weights_, session_, expert_cache_, id, pos, stream());
            if (e != cudaSuccess) {
                r.error_msg = std::string("forward failed: ") +
                              cudaGetErrorString(e);
                return r;
            }
            ++pos;
        }
        cudaEventRecord(ev_first, stream());

        const int eos = idx_.eos_token_id;
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
            const int next = ArgMaxHost(logits);
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
                    idx_, weights_, session_, expert_cache_, next, pos,
                    stream());
                if (e != cudaSuccess) {
                    r.error_msg = std::string("forward failed: ") +
                                  cudaGetErrorString(e);
                    return r;
                }
                ++pos;
            }
            if (req.token_callback) {
                TokenChunk c;
                c.token_id = static_cast<uint32_t>(next);
                c.step_t = static_cast<uint32_t>(step);
                c.logit_maxv = r.first_gen_maxv;
                req.token_callback(c);
            }
            if (req.abort_flag && req.abort_flag->load()) break;
        }
        r.num_tokens = generated;
        r.num_layers = idx_.num_layers;
        r.weight_resident_bytes = resident_bytes_;
        log::info(absl::StrFormat(
            "[nemotron] expert cache: hits=%lld misses=%lld slots=%d",
            static_cast<long long>(expert_cache_.device.hits),
            static_cast<long long>(expert_cache_.device.misses),
            expert_cache_.device.slots));
        cudaEventRecord(ev1, stream());
        cudaEventSynchronize(ev1);
        float ttft = 0.0f, total = 0.0f;
        cudaEventElapsedTime(&ttft, ev0, ev_first);
        cudaEventElapsedTime(&total, ev0, ev1);
        r.ttft_ms = static_cast<double>(ttft);
        r.total_time_ms = static_cast<double>(total);
        r.tokens_per_sec =
            total > 0.0f ? static_cast<double>(generated) * 1000.0 /
                               static_cast<double>(total)
                         : 0.0;
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev_first);
        cudaEventDestroy(ev1);
        r.ok = true;
        return r;
    });
}

gpu::InferenceResult NemotronHModel::SelfCheck(const ModelRequest &req)
{
    gpu::InferenceResult r = Infer(req);
    r.selfcheck_passed = false;
    if (!r.ok || r.gen_tokens.empty() || !tok_db_ || !tok_db_->valid())
        return r;
    // Decode the first generated token (the top-1 control token) and trim the
    // BPE leading space; the control prompt must answer "Paris".
    char buf[512];
    size_t cap = sizeof(buf);
    if (arrow_tokenizer_decode(*tok_db_, r.gen_tokens.data(), 1, buf, &cap) == 0) {
        std::string text(buf, cap);
        const auto first = text.find_first_not_of(" \t\n");
        const auto last = text.find_last_not_of(" \t\n");
        if (first != std::string::npos)
            text = text.substr(first, last - first + 1);
        else
            text.clear();
        r.first_gen_text = text;
        r.selfcheck_passed = text == "Paris";
    }
    // Full output text (all generated tokens) for callers that want it.
    if (!r.gen_tokens.empty()) {
        char obuf[1024];
        size_t ocap = sizeof(obuf);
        if (arrow_tokenizer_decode(*tok_db_, r.gen_tokens.data(),
                                   r.gen_tokens.size(), obuf, &ocap) == 0)
            r.output_text.assign(obuf, ocap);
    }
    return r;
}

bool NemotronHModel::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                         std::string_view raw_prompt,
                                         std::vector<int32_t> &out_ids) const
{
    return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);
}

} // namespace lsxcommon
