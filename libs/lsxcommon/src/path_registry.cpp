#include "lsxcommon/path_registry.h"

#include "lsxstorage/roots.h"

#include <filesystem>

namespace lsxcommon {

// The roots themselves live in libs/lsxstorage (namespace lsx::storage), which
// sits below this library so ModelBase::BeginLoad can read artifact.fb without a
// dependency cycle. Only the CACHE root is forwarded from here: the two model
// roots are std::optional and belong to the storage layer, so there is no
// `lsx_model_models_dir()` handing back a std::string that a caller could mistake
// for a configured path — with no default it would have to return "" and every
// join onto it would silently become a cwd-relative path.
//
// The cache root is the one place where the string "logestix" is still a path
// segment: it is the per-user host cache (~/.cache/logestix), unrelated to the
// artifact layout and independent of whether any model store is configured.

std::string lsx_cache_dir()
{
    return lsx::storage::CacheRoot().string();
}

std::string lsx_cache_path(const std::string& rel)
{
    std::string norm = rel;
    const char sep = std::filesystem::path::preferred_separator;
    for (char &c : norm)
        if (c == '/' || c == '\\') c = sep;
    return (lsx::storage::CacheRoot() / std::filesystem::path(norm)).string();
}

} // namespace lsxcommon
