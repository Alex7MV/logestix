#include "lsxcommon/minimax_moe_flash_convert.h"

#include "minimax_moe_flash_constants.h"

#include "lsxcommon/minimax_moe_convert.h"

namespace lsxcommon::minimax {

arrow::Status ConvertMinimaxFlashModel(const std::string &input_dir,
                                       const std::string &output_root)
{
    const MinimaxConvertVariant variant{
        .variant = MinimaxVariant::kM3Flash,
        .model = minimax_flash::frozen::kModelName,
        .model_type = minimax_flash::frozen::kType,
    };
    return ConvertMinimaxMoeModel(input_dir, output_root, variant);
}

} // namespace lsxcommon::minimax
