#include "lsxcommon/deepseek_moe_pro_convert.h"

#include "lsxcommon/converter.h"

#include "deepseek_moe_pro_constants.h"

#include <vector>

namespace lsxcommon {
namespace deepseek_pro {

arrow::Status ConvertDeepSeekProModel(const std::string &input_path,
                                      const std::string &output_path)
{
    if (input_path.empty() || output_path.empty())
        return arrow::Status::Invalid("input/output path required");

    Converter conv;
    conv.SetInputPath(input_path)
        .SetOutputPath(output_path)
        .SetVerbose(true)
        // DSpark / MTP speculative stage: stored-not-applied ballast.
        .SetSkipPrefixes({"mtp."})
        .SetModelName("DeepSeek-V4-Pro-0813")
        .SetExpectedGeometry(kLayers, kHidden, kVocab, kNumRouted);
    return conv.Run();
}

} // namespace deepseek_pro
} // namespace lsxcommon
