#pragma once

#include <string>

namespace lsxcommon {

// Compile-time platform selector for the SOURCE root. Overridable at runtime
// with LSX_SOURCE_ROOT, then LSX_MODEL_ROOT (the pre-split spelling); when both
// are unset the platform default below is used.
#if defined(_WIN32)
constexpr const char* kModelRootBase = "D:\\Programs\\huggingface.co";
#else
// The model vault moved on 2026-10; LSX_MODEL_ROOT still overrides this.
constexpr const char* kModelRootBase = "/mnt/fast_storage/huggingface";
#endif

// "<base>" — the read-only models directory holding named checkpoints. This is
// the SOURCE root: converted artifacts live under LSX_ARTIFACT_ROOT, never in
// the vault (libs/lsxstorage, design §2.4.1). Kept here because the platform
// default is compile-time and callers include this header for it.
std::string lsx_model_models_dir();

// Per-user cache root for host-side state that is NOT part of a model
// (per-GPU perf calibration). Never inside the model vault: a cache that a
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
