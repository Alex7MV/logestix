#include "lsxcommon/model_factory.h"

#include "lsxcommon/gigachat_moe_model.h"
#include "deepseek_moe_model.h"
#include "deepseek_moe_v41_model.h"
#include "gemma_moe_model.h"
#include "granite_dense_model.h"
#include "glm_moe_flash_model.h"
#include "glm_moe_model.h"
#include "kimi_moe_model.h"
#include "nemotron_moe_model.h"
#include "qwen_dense_model.h"
#include "qwen_dense_27b_model.h"
#include "qwen_moe_model.h"
#include "qwen_moe_36_model.h"
#include "ornith_moe_model.h"
#include "qwen_moe_exp_model.h"

namespace lsxcommon {

void EnsureModelRegistry()
{
    static bool init = [] {
        LsxModelFactory::Register<DeepSeekV4FlashModel>();
        LsxModelFactory::Register<DeepSeekV41FlashModel>();
        LsxModelFactory::Register<GlmV5Model>();
        LsxModelFactory::Register<KimiK27CodeModel>();
        LsxModelFactory::Register<Qwen3_5_9BModel>();
        LsxModelFactory::Register<Qwen3_8_27BModel>();
        LsxModelFactory::Register<Gemma4_26BA4BModel>();
        LsxModelFactory::Register<Qwen35MoeModel>();
        LsxModelFactory::Register<Qwen36MoeModel>();
        LsxModelFactory::Register<OrnithMoeModel>();
        LsxModelFactory::Register<Qwen38FlashNextModel>();
        LsxModelFactory::Register<Glm5NextFlashModel>();
        LsxModelFactory::Register<NemotronHModel>();
        LsxModelFactory::Register<GigaChat35Model>();
        LsxModelFactory::Register<Granite42_30BModel>();
        return true;
    }();
    (void)init;
}

} // namespace lsxcommon
