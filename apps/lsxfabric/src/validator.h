#pragma once

#include <string>

namespace lsxfabric {

// Validates a converted model by loading it through the inference engine.
// Every model-specific detail (format tag, tensor/expert/shared/router layout,
// schemas) is owned by the engine's registry and per-family loaders; this
// wrapper only drives the engine lifecycle. Returns true iff the model is
// detected, created and loaded successfully.
bool RunModelValidation(const std::string &model_path);

} // namespace lsxfabric
