#include "lsxcommon/model_factory.h"

#include "deepseek_v4_flash_model.h"

namespace lsxcommon {

void EnsureModelRegistry()
{
    static bool init = [] {
        // On-demand registration: Create() calls EnsureModelRegistry() before any
        // registry lookup, so this runs exactly once, after all TUs are loaded.
        // do NOT rely on namespace-scope dynamic init (cross-TU init order across
        // a STATIC lib is unspecified and the entry may be missing on first use).
        LsxModelFactory::Register(DeepSeekV4FlashModel::kType,
            [](const ModelInitConfig &cfg) {
                return std::make_unique<DeepSeekV4FlashModel>(cfg);
            });
        return true;
    }();
    (void)init;
}

} // namespace lsxcommon