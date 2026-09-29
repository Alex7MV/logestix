#include "lsxcommon/core.h"
#include "lsxcommon/log.h"

#include <cuda_runtime.h>

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace lsxcommon {

void cudaFreeIgnoreShutdown(void* ptr) {
    if (ptr == nullptr) return;
    const cudaError_t e = cudaFree(ptr);
    if (e == cudaSuccess || e == cudaErrorCudartUnloading) return;
    // Do NOT log through the async spdlog logger here: this runs from static /
    // thread-local destructors, possibly after Core::Shutdown() has torn the
    // logger down.
    std::fprintf(stderr, "[core] cudaFreeIgnoreShutdown: %s\n",
                 cudaGetErrorString(e));
}

void Core::Initialize(const std::string& log_file_path) {
    try {
        spdlog::init_thread_pool(8192, 1);

        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_file_path, 1024 * 1024 * 5, 3
        );

        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};

        auto logger = std::make_shared<spdlog::async_logger>(
            "lsx_logger", sinks.begin(), sinks.end(),
            spdlog::thread_pool(), spdlog::async_overflow_policy::block
        );

        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [thread %t] [%^%l%$] %v");
        logger->set_level(spdlog::level::trace);
        spdlog::set_default_logger(logger);

        log::info("Logestix common library core initialized.");
    } catch (const spdlog::spdlog_ex& ex) {
        // spdlog itself failed to initialize — no logger exists, so fall back
        // to a raw stderr write to surface the failure.
        const std::string msg = std::string("Critical: Log initialization failed: ")
            + ex.what() + "\n";
        std::fputs(msg.c_str(), stderr);
    }
}

void Core::Shutdown() {
    log::info("Shutting down Logestix common library core...");
    spdlog::shutdown();
}

} // namespace lsxcommon
