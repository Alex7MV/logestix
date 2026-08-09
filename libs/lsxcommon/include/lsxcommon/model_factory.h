#pragma once

#include <functional>
#include <memory>
#include <string>

#include "lsxcommon/model.h"

namespace lsxcommon {

using CreateModelFn = std::function<std::unique_ptr<IModel>(const ModelInitConfig &)>;

class LsxModelFactory {
public:
    static void Register(const std::string &type, CreateModelFn fn);
    static std::unique_ptr<IModel> Create(const ModelInitConfig &cfg);
    static std::string DetectType(const std::string &model_path);
};

// Guaranteed registry initialization before Create (needed for STATIC linking).
void EnsureModelRegistry();

// Registration helper used by model_registry.cpp.
bool RegisterModel(const char *type, CreateModelFn fn);

} // namespace lsxcommon