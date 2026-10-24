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

// Invariant metadata block appended to every metadata read/write query.
constexpr const char* kModelMetadataDir = "model-metadata";

// "<base>/model-metadata" (platform separator) — the vault root.
std::string lsx_model_metadata_root();

// Appends the invariant "model-metadata/" block to a relative subpath:
//   lsx_model_metadata("manifests/glm.json")
// returns "<base>/model-metadata/manifests/glm.json" with platform
// separators. No other code may reconstruct the vault root.
std::string lsx_model_metadata(const std::string& rel);

// "<base>" — the models directory holding named checkpoints.
std::string lsx_model_models_dir();

} // namespace lsxcommon
