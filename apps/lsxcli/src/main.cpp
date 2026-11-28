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
    // A model IDENTITY, not a path. `lsx::storage::Resolve` searches the
    // artifact store for it (and reports a source-only checkpoint as the
    // fixable thing it is). The old default built
    // `<source root>/DeepSeek-V4-Flash` — without the org the store nests it
    // under, and pointing at the read-only checkpoint rather than the
    // converted tree — so the default could never load.
    auto model_path = "deepseek-v4-flash";
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
