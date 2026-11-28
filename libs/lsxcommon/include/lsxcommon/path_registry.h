#pragma once

#include <string>

namespace lsxcommon {

// Compile-time platform constants are gone: neither model root has a default.
// Both are configured at runtime —
//
//   LSX_SOURCE_ROOT    read-only checkpoints  <root>/<org>/<model>   (no default)
//   LSX_ARTIFACT_ROOT  converted artifacts    <root>/<org>/<model>/<profile>
//                                              (no default)
//
// — and both answer std::optional. A `constexpr kModelRootBase` here was read as
// "the vault lives at X" even after the store moved and split, so a reader could
// not tell a configured path from an invented one. Location lives in
// libs/lsxstorage/include/lsxstorage/roots.h (`lsx::storage::SourceRoot`,
// `ArtifactRoot`, `ArtifactDir`), which this library forwards from.
//
// The store is ORG-NESTED, the way Hugging Face lays a download out.
// `lsx::storage::FindSourceModel("phi-4")` searches one level and reports an
// ambiguous name rather than picking an org.

// Per-user cache root for host-side state that is NOT part of a model
// (per-GPU perf calibration). Never inside a model store: a cache that a
// converter or a test can overwrite must not sit next to model data.
//   LSX_CACHE_DIR  -> used verbatim
//   Linux          -> $XDG_CACHE_HOME/logestix, else $HOME/.cache/logestix
//   Windows        -> %LOCALAPPDATA%\logestix\cache
//   otherwise      -> <temp>/logestix-cache
std::string lsx_cache_dir();

// Appends @p rel ('/'-or-'\\' separated) to lsx_cache_dir() with platform
// separators. No other code may reconstruct the cache root.
std::string lsx_cache_path(const std::string& rel);

} // namespace lsxcommon
