#include <memory>
#include "lsxcommon/model_factory.h"

#include "alice_moe_model.h"
#include "xing_moe_model.h"
#include "mistral_moe_model.h"
#include "mistral_moe_large4_model.h"
#include "hy_moe_model.h"
#include "hy_moe_simulmt_model.h"
#include "flint_moe_model.h"
#include "ling_moe_model.h"
#include "step_moe_model.h"
#include "beam_moe_model.h"
#include "minimax_moe_model.h"
#include "minimax_moe_flash_model.h"
#include "lsxcommon/gigachat_moe_model.h"
#include "deepseek_moe_model.h"
#include "deepseek_moe_pro_model.h"
#include "deepseek_moe_v41_model.h"
#include "gemma_dense_model.h"
#include "gemma_moe_model.h"
#include "granite_dense_model.h"
#include "granite_dense_switch_model.h"
#include "mimo_moe_model.h"
#include "phi_dense_model.h"
#include "glm_moe_flash_model.h"
#include "glm_moe_model.h"
#include "kimi_moe_model.h"
#include "nemotron_moe_model.h"
#include "nemotron_moe_nvfp4_model.h"
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
        LsxModelFactory::Register<DeepSeekV4ProModel>();
        LsxModelFactory::Register<DeepSeekV41FlashModel>();
        LsxModelFactory::Register<GlmV5Model>();
        LsxModelFactory::Register<KimiK27CodeModel>();
        LsxModelFactory::Register<Qwen3_5_9BModel>();
        LsxModelFactory::Register<Qwen3_8_27BModel>();
        LsxModelFactory::Register<Gemma4_26BA4BModel>();
        LsxModelFactory::Register<Gemma4_31BDenseModel>();
        LsxModelFactory::Register<Qwen35MoeModel>();
        LsxModelFactory::Register<Qwen36MoeModel>();
        LsxModelFactory::Register<OrnithMoeModel>();
        LsxModelFactory::Register<Qwen38FlashNextModel>();
        LsxModelFactory::Register<Glm5NextFlashModel>();
        LsxModelFactory::Register<NemotronHModel>();
        LsxModelFactory::Register<NemotronNvfp4Model>();
        LsxModelFactory::Register<GigaChat35Model>();
        LsxModelFactory::Register<Granite42_30BModel>();
        LsxModelFactory::Register<GraniteSwitch41_30BModel>();
        LsxModelFactory::Register<Phi4Model>();
        LsxModelFactory::Register<MiMoV26FlashModel>();
        LsxModelFactory::Register<AliceFoundation80BModel>();
        LsxModelFactory::Register<XingMoeModel>();
        LsxModelFactory::Register<MistralSmall4MoeModel>();
        LsxModelFactory::Register<MinimaxM3Model>();
        LsxModelFactory::Register<MiniMaxM31FlashModel>();
        LsxModelFactory::Register<HyMemoryModel>();
        LsxModelFactory::Register<HySimulMTModel>();
        LsxModelFactory::Register<FlintMoeModel>();
        LsxModelFactory::Register<LingMoeModel>();
        LsxModelFactory::Register<BeamMoeModel>();
        LsxModelFactory::Register<Step5PreviewModel>();
        LsxModelFactory::Register<MistralLarge4MoeModel>();
        // Alias for the vault-normalized index "model" name
        // ("AliceAI-Foundation-80B-A3B-Base" -> "aliceai-foundation-80b-a3b-base"),
        // so path auto-detection (lsxbenchmark/lsxfabric) resolves the model.
        LsxModelFactory::Register(
            "aliceai-foundation-80b-a3b-base",
            [](const ModelInitConfig &c) {
                return std::make_unique<AliceFoundation80BModel>(c);
            });
        // Vault-normalized directory name for the Beam tree
        // ("Beam-1.0-501B-A23B" -> "beam-1.0-501b-a23b") so path auto-detection
        // resolves kType "beam".
        LsxModelFactory::Register(
            "beam-1.0-501b-a23b", [](const ModelInitConfig &c) {
                return std::make_unique<BeamMoeModel>(c);
            });
        // Vault-normalized directory name for the Ling tree
        // ("Ling-3.1-flash" -> "ling-3.1-flash") so path auto-detection
        // resolves kType "ling".
        LsxModelFactory::Register(
            "ling-3.1-flash", [](const ModelInitConfig &c) {
                return std::make_unique<LingMoeModel>(c);
            });
        return true;
    }();
    (void)init;
}

} // namespace lsxcommon
