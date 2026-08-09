#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include "lsxcommon/testmodel_gpu.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/tokenizer.h"
#include <cstring>
#include <spdlog/spdlog.h>
#include <string>
#include "validator.h"

int main(int argc, char* argv[]) {
    bool validate = false;
    bool testmodel = false;
    bool testprompt = false;
    bool selfcheck = false;
    std::string model_path = R"(D:\Programs\huggingface.co\DeepSeek-V4-Flash)";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--localmodel") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (std::strcmp(argv[i], "--validate") == 0)
            validate = true;
        else if (std::strcmp(argv[i], "--testmodel") == 0)
            testmodel = true;
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
        auto &tok = model->Tokenizer();

        // Same prompt as the previous hardcoded pipeline prompt.
        const char *prompt = "The capital of France is";
        std::vector<int32_t> input_ids(512);
        size_t in_cap = input_ids.size();
        int32_t n_ids = lsxcommon::arrow_tokenizer_encode(
            tok, prompt, std::strlen(prompt), input_ids.data(), &in_cap);
        if (n_ids < 0) {
            spdlog::error("Tokenizer encode failed (code {})", n_ids);
            model->Unload();
            lsxcommon::Core::Shutdown();
            return 1;
        }
        input_ids.resize(static_cast<size_t>(n_ids));

        lsxcommon::ModelRequest req{std::move(input_ids), 64};
        auto result = model->Infer(req);
        if (!result.ok) {
            spdlog::error("Inference failed: {}", result.error_msg);
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
        auto &tok = model->Tokenizer();
        const char *prompt = "The capital of France is";
        std::vector<int32_t> input_ids(512);
        size_t in_cap = input_ids.size();
        int32_t n_ids = lsxcommon::arrow_tokenizer_encode(
            tok, prompt, std::strlen(prompt), input_ids.data(), &in_cap);
        if (n_ids < 0) {
            spdlog::error("Tokenizer encode failed (code {})", n_ids);
            model->Unload();
            lsxcommon::Core::Shutdown();
            return 1;
        }
        input_ids.resize(static_cast<size_t>(n_ids));

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
    if (testmodel) {
        spdlog::info("Running model benchmark...");
        auto result = lsxcommon::gpu::RunTestModel(model_path);
        if (!result.ok) {
            spdlog::error("TestModel failed: {}", result.error_msg);
        } else {
            spdlog::info("=== --testmodel Results ===");
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
