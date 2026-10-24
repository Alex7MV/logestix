#pragma once

#include <string>

namespace lsxcommon {

// Compile-time platform selector for the model root (CC 12.0 build matrix).
// Overridable at runtime with the LSX_MODEL_ROOT environment variable (an
// absolute path to the models directory); when unset, the platform default
// below is used.
#if defined(_WIN32)
constexpr const char* kModelRootBase = "D:\\Programs\\huggingface.co";
#else
constexpr const char* kModelRootBase = "/mnt/llm_models/models";
#endif

// "<base>" — the models directory holding named checkpoints. Converted models
// live in "<base>/<Model>/logestix/"; nothing else is written into the vault.
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
