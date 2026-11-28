#pragma once

// ModelRoot() needs lsxcommon's path registry and lsxstorage's artifact root,
// which the host-only guard tests do not all link. Keeping the definition in its
// own header means a guard can include test_manifest_query.h (table accessors
// only) and a gate driver can include this one too.

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>

#include <lsxcommon/path_registry.h>
#include <lsxstorage/roots.h>

#include "test_manifest_query.h"

namespace manifest_query {

// Where this model's ARTIFACT lives, whether or not it exists yet:
//   <LSX_ARTIFACT_ROOT>/<org>/<model_dir>/<profile>
// Empty when no artifact root is configured — the root has no default on
// purpose (libs/lsxstorage/roots.h), so there is nothing honest to invent.
inline std::filesystem::path ArtifactDir(const lsxmanifest::ModelRow &row)
{
    const std::optional<std::filesystem::path> root = lsx::storage::ArtifactRoot();
    if (!root) return {};
    return *root / row.vault_org / row.vault_dir / row.vault_profile;
}

// Where this model's SOURCE checkpoint lives:
//   <LSX_SOURCE_ROOT>/<org>/<model_dir>
// Exported separately because the two halves of the split store answer
// different questions, and a caller that converts reads the checkpoint while a
// caller that loads reads the artifact.
inline std::filesystem::path SourceDir(const lsxmanifest::ModelRow &row)
{
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           row.vault_org / row.vault_dir;
}

// The directory a gate should load from.
//
// Precedence, and why in this order:
//   1. LSX_TEST_<MODEL>_ROOT — an operator pointing at a tree anywhere (a second
//      artifact root, a copy on another disk). The escape hatch stays.
//   2. the artifact store — what every gate driver in this tree can actually
//      load, since `model.index.json` / `artifact.fb` are written there and
//      never into a Hugging Face download.
//
// It deliberately does NOT fall back to SourceDir(). That fallback is what made
// every real-weight target SKIP by default: the checkpoint has no
// `model.index.json`, so the gate reported a missing file instead of a missing
// conversion, and the only way to run a gate was to export the env var by hand.
// When a model is not converted yet the returned path simply does not exist,
// and the SKIP names the artifact path that was expected — which is the
// question an operator can act on.
inline std::filesystem::path ModelRoot(const lsxmanifest::ModelRow &row)
{
    if (const char *env = std::getenv(row.vault_env); env && *env) return env;
    return ArtifactDir(row);
}

} // namespace manifest_query
