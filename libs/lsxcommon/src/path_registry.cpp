#include "lsxcommon/path_registry.h"

#include "lsxstorage/roots.h"

#include <filesystem>

namespace lsxcommon {

// The roots themselves live in libs/lsxstorage (namespace lsx::storage), which
// sits below this library so ModelBase::BeginLoad can read artifact.fb without a
// dependency cycle. These three functions stay because ~40 call sites and every
// test in the tree use them; they are forwarders, not a second implementation.
//
// The cache root is the one place where the string "logestix" is still a path
// segment: it is the per-user host cache (~/.cache/logestix), unrelated to the
// artifact layout.

std::string lsx_model_models_dir()
{
    return lsx::storage::SourceRoot().string();
}

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
