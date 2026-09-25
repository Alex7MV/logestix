#include "lsxcommon/common.h"
#include "lsxcommon/core.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_bridge.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/tokenizer_arrows.h"
#include "lsxnetwork/fabric_client.h"

#include <absl/strings/str_format.h>

#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cli.h"

namespace {

std::string ReadAllStdin() {
    return std::string(std::istreambuf_iterator<char>(std::cin),
                       std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char* argv[]) {
    const taskmosaic::TaskMosaicCliArgs cli =
        taskmosaic::ParseTaskMosaicCliArgs(argc, argv);

    lsxcommon::Core::Initialize("taskmosaic_runtime.log");
    lsxcommon::version();

    int rc = 0;
    {
        if (cli.server.empty() || cli.model.empty()) {
            lsxcommon::log::error(
                "taskmosaic is a client: pass --server <endpoint> and "
                "--model <converted-model-path> [--prompt <text>|stdin]");
            rc = 2;
        } else {
            const std::string prompt =
                cli.prompt.empty() ? ReadAllStdin() : cli.prompt;

            lsxcommon::ArrowTokenizerDb tok;
            if (!lsxcommon::LoadTokenizerArrows(cli.model, tok)) {
                lsxcommon::log::error("cannot load tokenizer for " + cli.model);
                rc = 1;
            } else {
                std::vector<std::int32_t> ids;
                if (!lsxcommon::FramePromptForModel(cli.model, prompt, ids)) {
                    lsxcommon::log::error("prompt framing failed");
                    rc = 1;
                } else {
                    lsxnetwork::FabricClient client;
                    if (client.Connect(cli.server.c_str()) != 0) {
                        lsxcommon::log::error("cannot connect to " + cli.server);
                        rc = 1;
                    } else {
                        lsxnetwork::FabricResponse resp;
                        if (!client.Infer(
                                ids, static_cast<std::uint32_t>(cli.max_gen),
                                resp)) {
                            lsxcommon::log::error(
                                absl::StrFormat("inference transport failed "
                                                "on %s",
                                                cli.server));
                            rc = 1;
                        } else if (resp.status !=
                                   static_cast<std::int32_t>(
                                       lsxnetwork::FabricStatus::kOk)) {
                            lsxcommon::log::error(
                                absl::StrFormat("server error (status %d): %s",
                                                resp.status, resp.error));
                            rc = 1;
                        } else {
                            const std::string text =
                                lsxcommon::runtime::DecodeGeneratedText(
                                    tok, resp.tokens);
                            lsxcommon::log::info(text);
                            lsxcommon::log::info(absl::StrFormat(
                                "tokens=%u ttft=%.1fms total=%.1fms %.1f t/s",
                                static_cast<unsigned>(resp.tokens.size()),
                                resp.ttft_ms, resp.total_ms,
                                resp.tokens_per_sec));
                        }
                    }
                }
            }
        }
    }
    lsxcommon::Core::Shutdown();
    return rc;
}
