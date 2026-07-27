#include "lsxcommon/converter.h"
#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include <cstring>
#include <spdlog/spdlog.h>
#include <string>

int main(int argc, char* argv[]) {
    std::string input_path;
    std::string output_path = "converted_model";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc)
            input_path = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_path = argv[++i];
    }

    if (input_path.empty()) {
        lsxcommon::Core::Initialize("lsxforge_runtime.log");
        lsxcommon::version();
        lsxcommon::log_gpu_info();
        lsxcommon::Core::Shutdown();
        return 0;
    }

    lsxcommon::Core::Initialize("lsxforge_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();

    spdlog::info("Converting model: {} -> {}", input_path, output_path);
    lsxcommon::Converter converter;
    converter.SetInputPath(input_path)
             .SetOutputPath(output_path)
             .SetVerbose(true);
    auto status = converter.Run();
    if (!status.ok()) {
        spdlog::error("Conversion failed: {}", status.ToString());
        lsxcommon::Core::Shutdown();
        return 1;
    }

    spdlog::info("Conversion complete");
    lsxcommon::Core::Shutdown();
    return 0;
}
