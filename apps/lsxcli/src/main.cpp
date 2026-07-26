#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string model_path = R"(D:\Programs\huggingface.co\DeepSeek-V4-Flash)";
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            model_path = argv[i + 1];
            break;
        }
    }

    lsxcommon::Core::Initialize("lsxcli_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    std::cout << "Model path: " << model_path << '\n';
    lsxcommon::Core::ExecuteDummyInference();
    lsxcommon::Core::Shutdown();
    return 0;
}
