#include "lsxcommon/common.h"
#include "lsxcommon/core.h"
#include "lsxcommon/log.h"

#include <absl/strings/str_format.h>

#include <atomic>
#include <csignal>
#include <string>

#include "chat_bridge.h"
#include "cli.h"
#include "config.h"
#include "lsx_framer.h"

namespace {

std::atomic<bool>* g_stop = nullptr;

void OnSignal(int) {
    if (g_stop != nullptr) g_stop->store(true);
}

}  // namespace

int main(int argc, char* argv[]) {
    const taskmosaic::TaskMosaicCliArgs cli =
        taskmosaic::ParseTaskMosaicCliArgs(argc, argv);

    lsxcommon::Core::Initialize("taskmosaic_runtime.log");
    lsxcommon::version();

    int rc = 0;
    {
        if (cli.config.empty()) {
            lsxcommon::log::error("taskmosaic: pass --config <bridge.json>");
            rc = 2;
        } else {
            taskmosaic::BridgeConfig cfg;
            std::string err;
            if (!taskmosaic::LoadBridgeConfigFile(cli.config, cfg, err)) {
                lsxcommon::log::error("config error: " + err);
                rc = 2;
            } else {
                if (!cli.host.empty()) cfg.host = cli.host;
                if (cli.port > 0) cfg.port = cli.port;

                taskmosaic::LsxPromptFramer framer;
                taskmosaic::ChatBridge bridge(cfg, framer);
                if (!bridge.Initialize(err)) {
                    lsxcommon::log::error("bridge init failed: " + err);
                    rc = 1;
                } else {
                    std::atomic<bool> stop{false};
                    g_stop = &stop;
                    std::signal(SIGINT, OnSignal);
                    std::signal(SIGTERM, OnSignal);
                    lsxcommon::log::info(absl::StrFormat(
                        "taskmosaic bridge on %s:%d (%u models)", cfg.host,
                        cfg.port, static_cast<unsigned>(cfg.models.size())));
                    bridge.SetStopFlag(&stop);
                    bridge.Listen();  // blocks until Stop / signal
                    g_stop = nullptr;
                }
            }
        }
    }
    lsxcommon::Core::Shutdown();
    return rc;
}
