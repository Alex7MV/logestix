#include "model_identity.h"

#include "lsxstorage/roots.h"

#include "model_manifest.h"

#include <simdjson.h>

#include <filesystem>
#include <fstream>
#include <iterator>
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
        // Store coordinates come from the MANIFEST, never from the path that was
        // typed: the artifact mirrors `<org>/<model>`, and only the registry
        // knows which org owns the model. Deriving the org from the input's
        // parent put the artifact under whatever directory the input happened
        // to sit in — a copy on another disk, a scratch dir, a script's temp —
        // and nothing downstream could tell that from the real location.
        id.model_dir = std::string(vault);
        id.org = row.vault_org != nullptr ? std::string(row.vault_org) : "";
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

std::string ChatTemplateFromTokenizerConfig(const std::string &config_path)
{
    std::ifstream in(config_path, std::ios::binary);
    if (!in) return {};
    std::string json((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    if (json.empty()) return {};

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) return {};
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) return {};
    std::string_view text;
    if (root["chat_template"].get_string().get(text) != simdjson::SUCCESS)
        return {};
    return std::string(text);
}

} // namespace lsxforge
