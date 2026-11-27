#include "lsxstorage/roots.h"

#include <cstdlib>

#if defined(_WIN32)
#include "lsxcommon/path_registry.h"  // only for the platform default base
#endif

namespace lsx::storage {

namespace {

// Platform default for the checkpoint vault. Kept byte-identical to
// lsxcommon::kModelRootBase so a tree configured today and a tree configured
// before the split resolve the same sources.
#if defined(_WIN32)
constexpr const char *kDefaultSourceBase = R"(D:\Programs\huggingface.co)";
#else
constexpr const char *kDefaultSourceBase = "/mnt/fast_storage/huggingface";
#endif

const char *NonEmptyEnv(const char *name)
{
    const char *v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? v : nullptr;
}

} // namespace

fs::path SourceRoot()
{
    if (const char *v = NonEmptyEnv("LSX_SOURCE_ROOT")) return fs::path(v);
    if (const char *v = NonEmptyEnv("LSX_MODEL_ROOT")) return fs::path(v);
    return fs::path(kDefaultSourceBase);
}

std::optional<fs::path> ArtifactRoot()
{
    const char *v = NonEmptyEnv("LSX_ARTIFACT_ROOT");
    if (v == nullptr) return std::nullopt;
    fs::path p(v);
    // A relative root would make the artifact layout depend on the process's
    // working directory, so two runs from different shells would write to
    // different places. Treat it as unset and let the caller report it.
    if (p.is_relative()) return std::nullopt;
    return p;
}

fs::path CacheRoot()
{
    if (const char *v = NonEmptyEnv("LSX_CACHE_DIR")) return fs::path(v);
#if defined(_WIN32)
    if (const char *lad = NonEmptyEnv("LOCALAPPDATA"))
        return fs::path(lad) / "logestix" / "cache";
#else
    if (const char *xdg = NonEmptyEnv("XDG_CACHE_HOME"))
        return fs::path(xdg) / "logestix";
    if (const char *home = NonEmptyEnv("HOME"))
        return fs::path(home) / ".cache" / "logestix";
#endif
    return fs::temp_directory_path() / "logestix-cache";
}

fs::path ArtifactDir(const std::string &family, const std::string &model_type,
                     const std::string &profile)
{
    const auto root = ArtifactRoot();
    if (!root.has_value()) return {};
    if (family.empty() || model_type.empty() || profile.empty()) return {};
    return *root / family / model_type / profile;
}

fs::path LegacyDir(const std::string &model)
{
    // The pre-split layout. Nothing loadable lives here: see roots.h.
    if (const char *v = NonEmptyEnv("LSX_MODEL_ROOT"))
        return fs::path(v) / model / "logestix";
    return fs::path(kDefaultSourceBase) / model / "logestix";
}

} // namespace lsx::storage
