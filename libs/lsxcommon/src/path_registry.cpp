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

std::string lsx_model_models_dir()
{
    return base().string();
}

std::string lsx_cache_dir()
{
    if (const char *env = std::getenv("LSX_CACHE_DIR"); env && *env)
        return env;
#if defined(_WIN32)
    if (const char *lad = std::getenv("LOCALAPPDATA"); lad && *lad)
        return (std::filesystem::path(lad) / "logestix" / "cache").string();
#else
    if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
        return (std::filesystem::path(xdg) / "logestix").string();
    if (const char *home = std::getenv("HOME"); home && *home)
        return (std::filesystem::path(home) / ".cache" / "logestix").string();
#endif
    return (std::filesystem::temp_directory_path() / "logestix-cache").string();
}

std::string lsx_cache_path(const std::string& rel)
{
    std::string norm = rel;
    const char sep = std::filesystem::path::preferred_separator;
    for (char &c : norm)
        if (c == '/' || c == '\\') c = sep;
    return (std::filesystem::path(lsx_cache_dir()) /
            std::filesystem::path(norm)).string();
}

} // namespace lsxcommon
