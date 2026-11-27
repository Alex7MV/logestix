#include "lsxcommon/index_layout.h"

namespace lsxcommon {

std::string IndexLayoutError(int64_t layout)
{
    std::string got;
    if (layout == 0) {
        got = "model.index.json declares no \"layout\" field";
    } else {
        got = "model.index.json declares layout " + std::to_string(layout);
    }
    return got + ", but this build reads layout " + std::to_string(kIndexLayout) +
           ". Layout-1 trees (the old per-model `format` tag) are not readable: "
           "reconvert the model with `lsxforge --<family> --input <dir> "
           "--codec-profile <profile>`.";
}

} // namespace lsxcommon
