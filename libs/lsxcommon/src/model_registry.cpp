#include "lsxcommon/model_factory.h"

#include "deepseek_v4_flash_model.h"

namespace lsxcommon {

namespace {
const bool kRegDeepSeekV4Flash = RegisterModel(
    DeepSeekV4FlashModel::kType,
    [](const ModelInitConfig &cfg) {
        return std::make_unique<DeepSeekV4FlashModel>(cfg);
    });
} // namespace

void EnsureModelRegistry()
{
    static bool init = [] {
        (void)kRegDeepSeekV4Flash;
        return true;
    }();
    (void)init;
}

} // namespace lsxcommon