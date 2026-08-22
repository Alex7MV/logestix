#include "lsxcommon/model_runtime.h"

#include <absl/strings/ascii.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

#include "lsxcommon/model_factory.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/safetensors.h"

namespace lsxcommon {
namespace runtime {

namespace {
constexpr size_t kMaxDecodeBytes = 1u << 20;
constexpr size_t kInitialDecodeCap = 4096;
constexpr int kMaxDecodeAttempts = 8;
} // namespace

std::string DecodeGeneratedText(const ArrowTokenizerDb &tok_db,
                                const std::vector<int32_t> &ids)
{
    if (ids.empty()) return {};
    size_t cap = kInitialDecodeCap;
    std::string out(cap, '\0');
    for (int attempt = 0; attempt < kMaxDecodeAttempts; ++attempt) {
        const int dr = arrow_tokenizer_decode(tok_db, ids.data(), ids.size(),
                                              out.data(), &cap);
        if (dr == 0) {
            out.resize(cap);
            return out;
        }
        if (dr == -2 && cap > out.size() && cap <= kMaxDecodeBytes) {
            out.resize(cap);
            continue;
        }
        break;
    }
    spdlog::warn("[model_runtime] tokenizer decode failed ({} ids)", ids.size());
    return {};
}

gpu::InferenceResult RunParisSelfCheck(IModel &model, const ModelRequest &req)
{
    ModelRequest sc_req = req;
    sc_req.max_gen_tokens = std::max(2, req.max_gen_tokens);
    gpu::InferenceResult r = model.Infer(sc_req);
    if (!r.ok) {
        r.selfcheck_passed = false;
        return r;
    }
    if (!r.gen_tokens.empty()) {
        char buf[512];
        for (int ntok : {1, 2}) {
            if (ntok > static_cast<int>(r.gen_tokens.size())) break;
            size_t cap = sizeof(buf);
            if (arrow_tokenizer_decode(model.Tokenizer(), r.gen_tokens.data(),
                                       static_cast<size_t>(ntok), buf, &cap) != 0)
                continue;
            std::string t(buf, cap);
            absl::StripAsciiWhitespace(&t);
            r.first_gen_text = t;
            if (r.first_gen_text == "Paris") {
                r.selfcheck_passed = true;
                break;
            }
        }
        if (!r.selfcheck_passed && r.first_gen_text.empty()) {
            size_t cap = sizeof(buf);
            if (arrow_tokenizer_decode(model.Tokenizer(), r.gen_tokens.data(),
                                       std::min(static_cast<int>(r.gen_tokens.size()), 8),
                                       buf, &cap) == 0)
                r.first_gen_text = std::string(buf, cap);
        }
    }
    spdlog::info("[selfcheck] top1_tok={} text='{}' passed={}",
                 r.first_gen_tok_id, r.first_gen_text,
                 r.selfcheck_passed ? "true" : "false");
    return r;
}

namespace {

struct CliModel {
    std::unique_ptr<IModel> model;
    bool ok = false;
};

CliModel CreateCliModel(const std::string &model_path)
{
    CliModel out;
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_path;
    out.model = lsxcommon::LsxModelFactory::Create(cfg);
    if (!out.model || !out.model->Load()) {
        spdlog::error("Failed to create/load model: {}", model_path);
        return out;
    }
    out.ok = true;
    return out;
}

} // namespace

void PrintGenerationResult(const std::string &prompt, const gpu::InferenceResult &r)
{
    spdlog::info("=== --run Results ===");
    spdlog::info("prompt   : {}", prompt);
    spdlog::info("output   : {}", r.output_text);
    spdlog::info("tokens   : {} at {:.1f} t/s", r.num_tokens, r.tokens_per_sec);
}

int RunCliGeneration(const GenerationCliArgs &args)
{
    spdlog::info("Running generation path on prompt: {}", args.prompt);
    CliModel m = CreateCliModel(args.model_path);
    if (!m.ok) return 1;

    std::vector<int32_t> input_ids;
    if (!lsxcommon::prompt::BuildPromptInputIds(
            m.model->Type(), m.model->Tokenizer(), args.prompt, input_ids)) {
        spdlog::error("Tokenizer encode failed for prompt: {}", args.prompt);
        m.model->Unload();
        return 1;
    }
    lsxcommon::ModelRequest req{std::move(input_ids), args.max_gen_tokens};
    gpu::InferenceResult result = m.model->Infer(req);
    m.model->Unload();
    if (!result.ok) {
        spdlog::error("Inference failed: {}", result.error_msg);
        return 1;
    }
    PrintGenerationResult(args.prompt, result);
    return 0;
}

int RunCliSelfCheck(const GenerationCliArgs &args)
{
    spdlog::info("Running single-token self-check...");
    CliModel m = CreateCliModel(args.model_path);
    if (!m.ok) return 1;

    std::vector<int32_t> input_ids;
    if (!lsxcommon::prompt::BuildPromptInputIds(
            m.model->Type(), m.model->Tokenizer(), args.prompt, input_ids)) {
        spdlog::error("Tokenizer encode failed");
        m.model->Unload();
        return 1;
    }
    lsxcommon::ModelRequest req{std::move(input_ids), args.max_gen_tokens};
    gpu::InferenceResult result = m.model->SelfCheck(req);
    m.model->Unload();
    if (!result.ok) {
        spdlog::error("SelfCheck failed: {}", result.error_msg);
        return 1;
    }
    spdlog::info("=== --selfcheck Results ===");
    spdlog::info("first token : {} '{}'", result.first_gen_tok_id, result.first_gen_text);
    spdlog::info("maxv        : {:.4f}", result.first_gen_maxv);
    spdlog::info("passed      : {}", result.selfcheck_passed ? "yes" : "no");
    return result.selfcheck_passed ? 0 : 1;
}

int RunCliInspect(const std::string &model_path, bool full)
{
    CliModel m = CreateCliModel(model_path);
    if (!m.ok) return 1;
    SafetensorsModel mi = m.model->Inspect();
    if (full) {
        for (auto &shard : mi.shards) lsxcommon::print_shard(shard);
    }
    lsxcommon::print_model(mi);
    const auto &vp = m.model->VramRequirements();
    if (vp.min_bytes > 0) {
        spdlog::info("=== VRAM Profile (conversion-time, ctx={}) ===",
                     vp.context_tokens);
        spdlog::info("min     (weights only)         : {}",
                     FormatBytesHuman(vp.min_bytes));
        spdlog::info("optimal (+KV cache+workspace)  : {}",
                     FormatBytesHuman(vp.optimal_bytes));
        spdlog::info("max     (full expert placement): {}",
                     FormatBytesHuman(vp.max_bytes));
    }
    m.model->Unload();
    return 0;
}

} // namespace runtime
} // namespace lsxcommon