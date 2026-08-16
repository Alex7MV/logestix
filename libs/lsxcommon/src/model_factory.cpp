#include "lsxcommon/model_factory.h"

#include <mutex>
#include <absl/container/flat_hash_map.h>

#include <spdlog/spdlog.h>

#include "deepseek_inference_load.cuh"

namespace lsxcommon {

namespace {
absl::flat_hash_map<std::string, CreateModelFn> &Registry()
{
    static absl::flat_hash_map<std::string, CreateModelFn> reg;
    return reg;
}

// Guards Registry() against concurrent Register()/Create() calls. Today the
// only writer is EnsureModelRegistry() (itself serialized by its own
// function-local static init), invoked lazily from Create() before any
// lookup — but LsxModelFactory::Register() is a public API, and without this
// lock a hypothetical concurrent registration (e.g. a future plugin-model
// hot-reload path) racing a lookup from another thread would be a data race
// on the underlying unordered_map (rehash-during-read is undefined behavior).
std::mutex &RegistryMutex()
{
    static std::mutex m;
    return m;
}
} // namespace

void LsxModelFactory::Register(const std::string &type, CreateModelFn fn)
{
    std::lock_guard<std::mutex> lock(RegistryMutex());
    auto &reg = Registry();
    if (reg.find(type) != reg.end())
        spdlog::warn("[factory] model type '{}' re-registered, overwriting "
                     "the previous factory function", type);
    reg[type] = std::move(fn);
}

bool RegisterModel(const std::string &type, CreateModelFn fn)
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

    CreateModelFn fn;
    {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        auto &reg = Registry();
        auto it = reg.find(type);
        if (it == reg.end()) {
            spdlog::error("[factory] no registered model for type '{}'", type);
            return nullptr;
        }
        // Copy the std::function out while holding the lock, then invoke it
        // unlocked — the factory function itself (a model constructor) may
        // take arbitrarily long and must never hold RegistryMutex.
        fn = it->second;
    }
    return fn(cfg);
}

} // namespace lsxcommon