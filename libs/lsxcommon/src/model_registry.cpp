#include "lsxcommon/model_factory.h"

#include "deepseek_v4_flash_model.h"
#include "gemma_4_26b_a4b_model.h"
#include "glm_v5_model.h"
#include "kimi_v5_model.h"
#include "qwen_v5_model.h"
#include "qwen_3_8_27b_model.h"
#include "qwen_3_5_moe_model.h"

namespace lsxcommon {

void EnsureModelRegistry()
{
    static bool init = [] {
        LsxModelFactory::Register<DeepSeekV4FlashModel>();
        LsxModelFactory::Register<GlmV5Model>();
        LsxModelFactory::Register<KimiK27CodeModel>();
        LsxModelFactory::Register<Qwen3_5_9BModel>();
        LsxModelFactory::Register<Qwen3_8_27BModel>();
        LsxModelFactory::Register<Gemma4_26BA4BModel>();
        LsxModelFactory::Register<Qwen35MoeModel>();
        return true;
    }();
    (void)init;
}

} // namespace lsxcommon
