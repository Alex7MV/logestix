#include "lsxcommon/path_registry.h"

#include <filesystem>

namespace lsxcommon {

namespace {
std::filesystem::path base()
{
    return std::filesystem::path(kModelRootBase);
}
} // namespace

std::string lsx_model_metadata_root()
{
    return (base() / kModelMetadataDir).string();
}

std::string lsx_model_metadata(const std::string& rel)
{
    return (base() / kModelMetadataDir / rel).string();
}

std::string lsx_model_models_dir()
{
    return base().string();
}

} // namespace lsxcommon
