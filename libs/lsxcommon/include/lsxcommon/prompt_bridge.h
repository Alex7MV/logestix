#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lsxcommon {

/// Host-only: frame a raw prompt into model input ids using the model family's
/// own chat template (`IModel::BuildPromptInputIds`). The model instance is
/// created through the factory but NOT Load()ed: no weights are read and no GPU
/// work happens. Framing ownership stays with each model (no central registry).
bool FramePromptForModel(const std::string& model_path,
                         std::string_view raw_prompt,
                         std::vector<std::int32_t>& out_ids);

}  // namespace lsxcommon
