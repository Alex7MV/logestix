#include "model_identity.h"

#include "lsxstorage/roots.h"

#include "model_manifest.h"

#include <filesystem>
#include <string_view>

namespace fs = std::filesystem;

namespace lsxforge {

std::optional<ModelIdentity> ModelIdentityFor(const std::string &input_path)
{
    if (input_path.empty()) return std::nullopt;

    // Two candidates: the last path component (the common case, ".../Qwen/Qwen3.5-9B")
    // and the whole string (a bare model name typed on the CLI).
    const std::string leaf = fs::path(input_path).filename().string();
    const std::string whole = input_path;

    for (const auto &row : lsxmanifest::kModels) {
        if (row.vault_dir == nullptr) continue;
        const std::string_view vault(row.vault_dir);
        if (vault != leaf && vault != whole) continue;
        ModelIdentity id;
        id.model_type = row.model_type;
        id.family = row.family;
        // Store coordinates: the org is the path component the checkpoint sits
        // under, the model dir is its own directory name.
        id.model_dir = std::string(vault);
        id.org = fs::path(input_path).parent_path().filename().string();
        if (id.org.empty() || id.org == id.model_dir) {
            // A bare name has no org in the path; recover it from the store so
            // the artifact still lands beside its source.
            std::string why;
            if (const auto src = lsx::storage::FindSourceModel(id.model_dir, &why);
                src.has_value())
                id.org = src->parent_path().filename().string();
        }
        if (id.org.empty()) return std::nullopt;   // cannot place it honestly
        return id;
    }
    return std::nullopt;
}

std::string ArtifactProfileDir(const ModelIdentity &identity,
                               const std::string &profile)
{
    const auto root = lsx::storage::ArtifactRoot();
    if (!root.has_value()) return {};
    if (identity.org.empty() || identity.model_dir.empty() || profile.empty())
        return {};
    return (*root / identity.org / identity.model_dir / profile).string();
}

} // namespace lsxforge
