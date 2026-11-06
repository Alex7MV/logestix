#pragma once

// ModelRoot() needs lsxcommon's path registry, which the host-only guard tests
// do not link. Keeping the definition in its own header means a guard can
// include test_manifest_query.h (table accessors only) and a gate driver can
// include this one too.

#include <lsxcommon/path_registry.h>

#include "test_manifest_query.h"

namespace manifest_query {

inline std::string ModelRoot(const lsxmanifest::ModelRow &row)
{
    if (const char *env = std::getenv(row.vault_env); env && *env) return env;
    return lsxcommon::lsx_model_models_dir() + "/" + row.vault_dir;
}

} // namespace manifest_query
