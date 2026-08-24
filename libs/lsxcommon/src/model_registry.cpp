#include "lsxcommon/model_factory.h"

#include "deepseek_v4_flash_model.h"
#include "gemma_4_26b_a4b_model.h"
#include "glm_v5_model.h"
#include "kimi_v5_model.h"
#include "qwen_v5_model.h"
#include "qwen_3_8_27b_model.h"

namespace lsxcommon {

void EnsureModelRegistry()
{
    static bool init = [] {
        LsxModelFactory::Register(DeepSeekV4FlashModel::kType,
            [](const ModelInitConfig &cfg) {
                return std::make_unique<DeepSeekV4FlashModel>(cfg);
            });
        LsxModelFactory::Register(GlmV5Model::kType,
            [](const ModelInitConfig &cfg) {
                return std::make_unique<GlmV5Model>(cfg);
            });
        LsxModelFactory::Register(KimiK27CodeModel::kType,
            [](const ModelInitConfig &cfg) {
                return std::make_unique<KimiK27CodeModel>(cfg);
            });
        LsxModelFactory::Register(Qwen3_5_9BModel::kType,
            [](const ModelInitConfig &cfg) {
                return std::make_unique<Qwen3_5_9BModel>(cfg);
            });
        LsxModelFactory::Register(Qwen3_8_27BModel::kType,
            [](const ModelInitConfig &cfg) {
                return std::make_unique<Qwen3_8_27BModel>(cfg);
            });
        LsxModelFactory::Register(Gemma4_26BA4BModel::kType,
            [](const ModelInitConfig &cfg) {
                return std::make_unique<Gemma4_26BA4BModel>(cfg);
            });
        return true;
    }();
    (void)init;
}

} // namespace lsxcommon