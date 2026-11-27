#pragma once

// ── Model identity and the artifact profile directory (lsxforge) ────────────
//
// An artifact lives at <ARTIFACT_ROOT>/<family>/<model_type>/<profile>, and both
// identity fields come from the model manifest, not from the download's spelling.
// "Qwen3.5-9B" is a vault directory; `qwen3.5-9b` is the model_type. Deriving the
// path from the vault name would tie the layout to how a checkpoint happens to be
// named on disk, so this looks the manifest up by vault dir instead.
//
// The lookup reads the table the manifest generator already emits
// (model_manifest.h), so there is no second source of truth for model identity.

#include <optional>
#include <string>

namespace lsxforge {

struct ModelIdentity {
    std::string model_type;  // "qwen3.5-9b" — factory id, artifact.fb field
    std::string family;      // "qwen"       — artifact.fb field
    std::string org;         // "Qwen"       — artifact store path segment 1
    std::string model_dir;   // "Qwen3.5-9B" — artifact store path segment 2
};

// Resolves a checkpoint directory (or a bare name) to its manifest identity.
// Accepts an org-nested path too: ".../Qwen/Qwen3.5-9B" matches the vault dir
// "Qwen3.5-9B". Returns nullopt when the manifest has no such model, because
// guessing an identity would put an artifact somewhere nothing looks for it.
std::optional<ModelIdentity> ModelIdentityFor(const std::string &input_path);

// <ARTIFACT_ROOT>/<org>/<model_dir>/<profile>, or empty when the root is unset.
// The path MIRRORS the source store (<org>/<model>), so an operator moves
// between a checkpoint and its artifacts by changing only the root.
std::string ArtifactProfileDir(const ModelIdentity &identity,
                               const std::string &profile);

// The chat template from a checkpoint's tokenizer_config.json, or "" when the
// file is absent, unparseable, or carries no "chat_template" string.
//
// Hugging Face allows the template in two places: a chat_template.jinja file
// beside the weights, or a "chat_template" string inside tokenizer_config.json.
// microsoft/phi-4 ships only the second, which is why copying the .jinja file
// recorded chat_template = "" for a checkpoint that does have a template — an
// artifact that then could not reproduce the framing it was converted from.
std::string ChatTemplateFromTokenizerConfig(const std::string &config_path);

} // namespace lsxforge
