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
fs::path ArtifactDir(const std::string &family, const std::string &model_type,
                     const std::string &profile);

// The pre-split layout `<LSX_MODEL_ROOT>/<Model>/logestix`. Nothing loadable
// lives here any more: it exists so a load pointed at a source checkout
// produces "this is a source directory, not an artifact" instead of "path not
// found". Diagnostic only, never a compatibility path.
fs::path LegacyDir(const std::string &model);

} // namespace lsx::storage
