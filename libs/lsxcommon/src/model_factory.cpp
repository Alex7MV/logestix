#include "lsxcommon/model_factory.h"

#include <unordered_map>

#include <spdlog/spdlog.h>

#include "inference_load.cuh"

namespace lsxcommon {

namespace {
std::unordered_map<std::string, CreateModelFn> &Registry()
{
    static std::unordered_map<std::string, CreateModelFn> reg;
    return reg;
}
} // namespace

void LsxModelFactory::Register(const std::string &type, CreateModelFn fn)
{
    Registry()[type] = std::move(fn);
}

bool RegisterModel(const char *type, CreateModelFn fn)
{
    LsxModelFactory::Register(type, std::move(fn));
    return true;
}

std::string LsxModelFactory::DetectType(const std::string &model_path)
{
    return gpu::detect_model_type(model_path);
}

std::unique_ptr<IModel> LsxModelFactory::Create(const ModelInitConfig &cfg)
{
    EnsureModelRegistry();

    std::string type = cfg.model_type;
    if (type.empty()) {
        type = DetectType(cfg.model_path);
        if (type.empty()) {
            spdlog::error("[factory] cannot autodetect model type from {}",
                          cfg.model_path);
            return nullptr;
        }
    }

    auto &reg = Registry();
    auto it = reg.find(type);
    if (it == reg.end()) {
        spdlog::error("[factory] no registered model for type '{}'", type);
        return nullptr;
    }
    return it->second(cfg);
}

} // namespace lsxcommon