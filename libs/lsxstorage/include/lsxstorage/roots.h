#pragma once

// ── Storage roots: read-only sources vs writable artifacts (invariant I5) ───
//
// Converted artifacts used to be written INSIDE the Hugging Face source
// checkout (`<vault>/<Model>/logestix/`), which made a read-only model
// download writable by a conversion and left no place for a second codec
// profile of the same model. This layer splits the two:
//
//   LSX_SOURCE_ROOT    read-only checkpoints (default: the vault base)
//   LSX_ARTIFACT_ROOT  converted artifacts   (default: UNSET)
//   LSX_MODEL_ROOT     the pre-split single vault; kept only as the source
//                      default and so LegacyDir() can name an old layout
//   LSX_CACHE_DIR      per-user host cache (perf calibration), never a model
//
// No function here reads a codec or touches a model file: this is location
// only. The load gate in ModelBase::BeginLoad asks libs/lsxcommon.
//
// Deliberately NOT a header reachable from `.cu`: it uses <filesystem> and
// <optional>, and device TUs must stay free of both.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace lsx::storage {

namespace fs = std::filesystem;

// Read-only checkpoint root. Precedence: LSX_SOURCE_ROOT, then LSX_MODEL_ROOT,
// then the compile-time platform default (the vault base).
fs::path SourceRoot();

// Writable artifact root, or nullopt when LSX_ARTIFACT_ROOT is unset or not an
// absolute existing-capable path. There is no default on purpose: writing a
// multi-gigabyte artifact into an unconfigured location is exactly the bug this
// layer exists to prevent, so an unset root is a configuration error the caller
// must report, not a fallback it may silently take.
std::optional<fs::path> ArtifactRoot();

// Per-user cache root (perf calibration and other host state).
fs::path CacheRoot();

// Where the artifact of (family, model_type, profile) lives:
//   <ArtifactRoot>/<family>/<model_type>/<profile>
//
// model_type is the manifest field (`models/<model_type>.json`), NOT the vault
// directory name: "Qwen3.5-9B" is a download spelling, `qwen3.5-9b` is the
// identity. The profile directory IS the artifact — there is no intermediate
// `logestix/` segment (design §2.4.2).
//
// Returns an EMPTY path when no artifact root is configured, so a caller that
// ignores ArtifactRoot() cannot accidentally write under an invented root.
// The artifact path MIRRORS the source store, so moving between a checkpoint and
// its artifacts means changing only the root:
//   <ArtifactRoot>/<org>/<model_dir>/<profile>
// org and model_dir are the store's own spelling ("microsoft", "Qwen3.5-9B"), not
// the manifest's family/model_type: an operator navigates by what `ls` shows.
fs::path ArtifactDir(const std::string &org, const std::string &model_dir,
                     const std::string &profile);

// The pre-split layout `<LSX_MODEL_ROOT>/<Model>/logestix`. Nothing loadable
// lives here any more, and with the org-nested store it cannot even be computed
// correctly, so it is gone: `FindSourceModel` answers the question that mattered
// ("where is the checkpoint for this name?") and `Resolve` reports a source
// directory as source-only. See design §2.4.1.

// ── Source checkpoints: <SourceRoot>/<org>/<model> ─────────────────────────
//
// Hugging Face lays a download out as <org>/<model>, and so does the store, so
// resolution searches ONE level of nesting instead of assuming a flat root.
struct SourceModelRef {
    std::string org;
    std::string model;
    fs::path dir;   // <root>/<org>/<model>
};

// Every checkpoint under the source root, sorted by (org, model). A directory
// directly under the root is an org; only one that also holds weights is a
// checkpoint, so `<root>/microsoft` never shows up as a model.
std::vector<SourceModelRef> ListSourceModels();

// Empty <org>/<model> directories: deliberate records of where an evicted
// checkpoint belongs. They are NOT checkpoints (nothing can load from them) and
// never enter ListSourceModels(); they exist so a diagnostic can distinguish
// "the directory is here, the weights are not" from "you mistyped the name".
// An empty `model` lists every placeholder, sorted.
std::vector<fs::path> ListPlaceholderDirs(const std::string &model);

// Finds exactly one checkpoint whose directory name is `model`.
//
// Returns nullopt when there is none OR when more than one org holds that name;
// *why says which (and names the orgs). An ambiguous name is never resolved by
// picking one, because the choice would depend on directory iteration order.
std::optional<fs::path> FindSourceModel(const std::string &model,
                                        std::string *why);

} // namespace lsx::storage
