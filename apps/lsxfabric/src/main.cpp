#include "lsxcommon/core.h"
#include <absl/strings/str_format.h>
#include "lsxcommon/common.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_engine.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_runtime.h"
#include "cli.h"
#include "explain_artifact.h"
#include "fabric_server.h"
#include "validator.h"

#include "lsxcommon/hardware_target.h"
#include "lsxstorage/resolve.h"

#include <atomic>
#include <csignal>
#include <memory>
#include <string>

namespace {

std::atomic<bool>* g_serve_stop = nullptr;

void OnServeSignal(int) {
    if (g_serve_stop != nullptr) g_serve_stop->store(true);
}

std::unique_ptr<lsxcommon::IModel> LoadServeModel(const std::string& path) {
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = path;
    cfg.fast_math = false;
    cfg.fast_math_mode = lsxcommon::FastMathMode::kAuto;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    if (!model || !model->Load()) return nullptr;
    return model;
}

}  // namespace

int main(int argc, char *argv[]) {
    const lsxfabric::FabricCliArgs cli =
        lsxfabric::ParseFabricCliArgs(argc, argv);

    lsxcommon::Core::Initialize("logestix_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();

    // --explain-artifact answers a question ABOUT the store, so it runs before
    // the engine session and never loads a model: it is safe to run while
    // another process holds the GPU (design §2.7).
    if (cli.explain_artifact) {
        int explain_rc = 0;
        const std::string report = lsxfabric::ExplainArtifact(
            cli.model_path,
            lsxcommon::hw::SupportedCodecs(lsxcommon::hw::ActiveTarget()),
            lsxcommon::hw::CodecProfilePreference(), &explain_rc);
        lsxcommon::log::info(report);
        lsxcommon::Core::Shutdown();
        return explain_rc;
    }

    int rc = 0;
    // Scoped engine session: the per-GPU auto perf policy is resolved for
    // every launch mode, and ShutdownEngine (which logs) runs while the
    // spdlog default logger installed by Core::Initialize is still alive.
    {
        lsxcommon::InferenceEngine::Session engine_session;
        lsxcommon::log::info(absl::StrFormat("Model path: %s", cli.model_path));
        if (cli.serve) {
            auto model = LoadServeModel(cli.model_path);
            if (!model) {
                lsxcommon::log::error("Failed to load model for --serve");
                rc = 1;
            } else {
                std::atomic<bool> stop{false};
                g_serve_stop = &stop;
                std::signal(SIGINT, OnServeSignal);
                std::signal(SIGTERM, OnServeSignal);
                lsxfabric::FabricServer server(
                    *model, static_cast<std::uint32_t>(cli.max_gen));
                rc = server.Run(cli.listen, cli.listen_cancel, stop);
                g_serve_stop = nullptr;
                model->Unload();
            }
        } else if (cli.testprompt) {
            lsxcommon::runtime::GenerationCliArgs args;
            args.model_path = cli.model_path;
            args.prompt = cli.prompt;
            args.max_gen_tokens = cli.max_gen;
            args.fast_math_mode = lsxcommon::FastMathMode::kAuto;
            rc = lsxcommon::runtime::RunCliGeneration(args);
        } else if (cli.validate) {
            lsxcommon::log::info("Running validation...");
            rc = lsxfabric::RunModelValidation(cli.model_path) ? 0 : 1;
        } else {
            lsxcommon::log::error(
                "No run mode selected: pass --serve, --validate or --testprompt");
            rc = 1;
        }
    }
    lsxcommon::Core::Shutdown();
    return rc;
}
