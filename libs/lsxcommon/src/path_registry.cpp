#include "lsxcommon/path_registry.h"

#include <cstdlib>
#include <filesystem>

namespace lsxcommon {

namespace {
std::filesystem::path base()
{
    // Runtime override (LSX_MODEL_ROOT) takes precedence over the compile-time
    // platform default so an external user can relocate the vault.
    if (const char* env = std::getenv("LSX_MODEL_ROOT"); env && *env)
        return std::filesystem::path(env);
    return std::filesystem::path(kModelRootBase);
}
} // namespace

std::string lsx_model_metadata_root()
{
    return (base() / kModelMetadataDir).string();
}

std::string lsx_model_metadata(const std::string& rel)
{
    std::string norm = rel;
    const char sep = std::filesystem::path::preferred_separator;
    for (char &c : norm)
        if (c == '/') c = sep;
    return (base() / kModelMetadataDir / std::filesystem::path(norm)).string();
}

std::string lsx_model_models_dir()
{
    return base().string();
}

} // namespace lsxcommon
