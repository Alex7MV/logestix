#include "lsxcommon/prompt_bridge.h"

#include "lsxcommon/model_factory.h"
#include "lsxcommon/tokenizer_arrows.h"

namespace lsxcommon {

bool FramePromptForModel(const std::string& model_path,
                         std::string_view raw_prompt,
                         std::vector<std::int32_t>& out_ids) {
    ModelInitConfig cfg;
    cfg.model_path = model_path;
    auto model = LsxModelFactory::Create(cfg);
    if (!model) return false;

    ArrowTokenizerDb tok;
    if (!LoadTokenizerArrows(model_path, tok)) return false;

    return model->BuildPromptInputIds(tok, raw_prompt, out_ids);
}

}  // namespace lsxcommon
