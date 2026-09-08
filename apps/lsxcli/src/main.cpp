#include "lsxcommon/core.h"
#include <absl/strings/str_format.h>
#include "lsxcommon/common.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/path_registry.h"
#include "lsxcommon/log.h"
#include <cstring>
#include <filesystem>
#include <string>

int main(int argc, char* argv[]) {
    auto model_path = (std::filesystem::path(lsxcommon::lsx_model_models_dir())
                       / "DeepSeek-V4-Flash").string();
    bool info = false;
    bool infofull = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--info") == 0) {
            info = true;
        } else if (std::strcmp(argv[i], "--infofull") == 0) {
            infofull = true;
        }
    }

    lsxcommon::Core::Initialize("lsxcli_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    lsxcommon::log::info(absl::StrFormat("Model path: %s", model_path));

    if (info || infofull) {
        int rc = lsxcommon::runtime::RunCliInspect(model_path, infofull);
        lsxcommon::Core::Shutdown();
        return rc;
    }

    lsxcommon::Core::Shutdown();
    return 0;
}
