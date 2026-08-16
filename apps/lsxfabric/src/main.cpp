#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include "lsxcommon/benchmark_gpu.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/path_registry.h"
#include <cstring>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>
#include "validator.h"

namespace {

// Encodes the shared "capital of France" self-check prompt for the active
// model. GLM-5.2 requires its chat-template framing
// ([gMASK]<sop><|user|>{prompt}<|assistant|><think></think> for the
// non-thinking generation mode) and never auto-prepends a BOS; other models
// (e.g. DeepSeek) consume the raw prompt directly. Returns false when the
// tokenizer rejects the prompt.
bool EncodeSelfCheckPrompt(lsxcommon::IModel &model, std::vector<int32_t> &ids)
{
    auto &tok = model.Tokenizer();
    const char *prompt = "The capital of France is";
    ids.assign(512, 0);
    int32_t n_ids = -1;
    if (model.Type() == "glm-5.2") {
        const int32_t gmask =
            lsxcommon::arrow_tokenizer_lookup(tok, "[gMASK]", 7);
        const int32_t sop = lsxcommon::arrow_tokenizer_lookup(tok, "<sop>", 5);
        const int32_t usr = lsxcommon::arrow_tokenizer_lookup(tok, "<|user|>", 8);
        const int32_t asst =
            lsxcommon::arrow_tokenizer_lookup(tok, "<|assistant|>", 13);
        const int32_t think =
            lsxcommon::arrow_tokenizer_lookup(tok, "<think>", 7);
        const int32_t endthink =
            lsxcommon::arrow_tokenizer_lookup(tok, "</think>", 8);
        if (gmask >= 0 && sop >= 0 && usr >= 0 && asst >= 0 &&
            think >= 0 && endthink >= 0) {
            size_t cap = ids.size() - 6;
            n_ids = lsxcommon::arrow_tokenizer_encode(
                tok, prompt, std::strlen(prompt), ids.data() + 3, &cap);
            if (n_ids > 0) {
                ids[0] = gmask;
                ids[1] = sop;
                ids[2] = usr;
                ids[3 + n_ids] = asst;
                ids[4 + n_ids] = think;
                ids[5 + n_ids] = endthink;
                n_ids += 6;
            }
        } else {
            size_t cap = ids.size();
            n_ids = lsxcommon::arrow_tokenizer_encode(
                tok, prompt, std::strlen(prompt), ids.data(), &cap);
        }
    } else {
        size_t cap = ids.size();
        n_ids = lsxcommon::arrow_tokenizer_encode(
            tok, prompt, std::strlen(prompt), ids.data(), &cap);
    }
    if (n_ids < 0) return false;
    ids.resize(static_cast<size_t>(n_ids));
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    bool validate = false;
    bool benchmark = false;
    bool testprompt = false;
    bool selfcheck = false;
    auto model_path = (std::filesystem::path(lsxcommon::lsx_model_models_dir())
                       / "DeepSeek-V4-Flash").string();
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--localmodel") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (std::strcmp(argv[i], "--validate") == 0)
            validate = true;
        else if (std::strcmp(argv[i], "--benchmark") == 0)
            benchmark = true;
        else if (std::strcmp(argv[i], "--testprompt") == 0)
            testprompt = true;
        else if (std::strcmp(argv[i], "--selfcheck") == 0)
            selfcheck = true;
    }

    lsxcommon::Core::Initialize("logestix_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    spdlog::info("Model path: {}", model_path);
    if (testprompt) {
        spdlog::info("Running prompt inference...");
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = model_path;
        cfg.max_gen_tokens = 64;
        auto model = lsxcommon::LsxModelFactory::Create(cfg);
        if (!model || !model->Load()) {
            spdlog::error("Failed to create/load model: {}", model_path);
            lsxcommon::Core::Shutdown();
            return 1;
        }
        std::vector<int32_t> input_ids;
        if (!EncodeSelfCheckPrompt(*model, input_ids)) {
            spdlog::error("Tokenizer encode failed");
            model->Unload();
            lsxcommon::Core::Shutdown();
            return 1;
        }

        lsxcommon::ModelRequest req{std::move(input_ids), 64};
        auto result = model->Infer(req);
        if (!result.ok) {
            spdlog::error("Inference failed: {}", result.error_msg);
        } else {
            spdlog::info("=== --testprompt Results ===");
            if (!result.output_text.empty())
                spdlog::info("output   : {}", result.output_text);
            spdlog::info("tokens   : {}", result.num_tokens);
            spdlog::info("time     : {:0.1f} ms", result.total_time_ms);
            spdlog::info("TTFT     : {:0.2f} ms", result.ttft_ms);
            spdlog::info("tokens/s : {:0.1f}", result.tokens_per_sec);
        }
        model->Unload();
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    if (selfcheck) {
        spdlog::info("Running single-token self-check...");
        lsxcommon::ModelInitConfig cfg;
        cfg.model_path = model_path;
        auto model = lsxcommon::LsxModelFactory::Create(cfg);
        if (!model || !model->Load()) {
            spdlog::error("Failed to create/load model: {}", model_path);
            lsxcommon::Core::Shutdown();
            return 1;
        }
        std::vector<int32_t> input_ids;
        if (!EncodeSelfCheckPrompt(*model, input_ids)) {
            spdlog::error("Tokenizer encode failed");
            model->Unload();
            lsxcommon::Core::Shutdown();
            return 1;
        }

        lsxcommon::ModelRequest req{std::move(input_ids), 1};
        auto result = model->SelfCheck(req);
        if (result.ok) {
            spdlog::info("=== --selfcheck Results ===");
            spdlog::info("first token : {} '{}'",
                         result.first_gen_tok_id, result.first_gen_text);
            spdlog::info("maxv        : {:.4f}", result.first_gen_maxv);
            spdlog::info("passed      : {}",
                         result.selfcheck_passed ? "yes" : "no");
        } else {
            spdlog::error("SelfCheck failed: {}", result.error_msg);
        }
        model->Unload();
        lsxcommon::Core::Shutdown();
        return (result.ok && result.selfcheck_passed) ? 0 : 1;
    }
    if (validate) {
        spdlog::info("Running validation...");
        auto result = lsxfabric::RunValidator(model_path);
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    if (benchmark) {
        spdlog::info("Running model benchmark...");
        auto result = lsxcommon::gpu::RunBenchmark(model_path);
        if (!result.ok) {
            spdlog::error("Benchmark failed: {}", result.error_msg);
        } else {
            spdlog::info("=== --benchmark Results ===");
            spdlog::info("Tokens generated : {}", result.num_tokens);
            spdlog::info("Layers processed : {}",
                         result.num_layers * result.num_tokens);
            spdlog::info("Total time       : {:0.1f} ms", result.total_time_ms);
            spdlog::info("Avg per layer    : {:0.3f} ms",
                         result.avg_bus_latency_ms);
            spdlog::info("Tokens/sec       : {:0.1f} t/s",
                         result.tokens_per_sec);
            spdlog::info("DMA volume       : {} MiB",
                         result.total_bytes_transferred / (1024 * 1024));
        }
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }
    lsxcommon::Core::Shutdown();
    return 0;
}
