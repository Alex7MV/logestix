#include "lsxcommon/model_factory.h"

#include "lsxstorage/resolve.h"

#include <mutex>
#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_format.h>

#include "lsxcommon/log.h"
#include "lsxcommon/gpu_perf.h"

#include "deepseek_moe_load.cuh"

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
        log::warn(absl::StrFormat(
            "[factory] model type '%s' re-registered, overwriting "
            "the previous factory function",
            type));
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
    // Fixed-speed default: every model consumer gets the pinned per-GPU
    // perf policy (call-once; explicit SetCurrentKnobs overrides win).
    perf::EnsurePerfPolicyApplied();
    EnsureModelRegistry();

    // `eff` because the resolve step below may replace the path: the caller
    // named a model, and only the store knows where the artifact for it lives.
    ModelInitConfig eff = cfg;
    std::string type = eff.model_type;
    if (type.empty()) {
        type = DetectType(eff.model_path);
        if (type.empty()) {
            // Autodetect reads `<path>/model.index.json`, so it only works when
            // the caller already named an ARTIFACT directory. Everything else
            // arrives here: the registry identity (`qwen3.5-9b` — the spelling
            // the gates, the CLI defaults and every document use), a store
            // directory (`Qwen3.5-9B`), an org-qualified name, a path one level
            // inside an artifact, or a plain typo. Resolve answers all of them,
            // and for the typo it returns the reason instead of "cannot
            // autodetect model type", which named the wrong problem.
            const auto resolved = lsx::storage::Resolve(eff.model_path);
            if (resolved.status == lsx::storage::ResolveStatus::kArtifact) {
                eff.model_path = resolved.dir.string();
                type = DetectType(eff.model_path);
                if (type.empty() && !resolved.model_type.empty())
                    type = resolved.model_type;
            }
            if (type.empty()) {
                log::error(absl::StrFormat(
                    "[factory] %s is not a LogestiX artifact (%s). %s",
                    eff.model_path,
                    lsx::storage::ResolveStatusName(resolved.status),
                    resolved.detail.empty() ? "convert it with lsxforge"
                                            : resolved.detail));
                return nullptr;
            }
        }
    }

    CreateModelFn fn;
    {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        auto &reg = Registry();
        auto it = reg.find(type);
        if (it == reg.end()) {
            log::error(
                absl::StrFormat("[factory] no registered model for type '%s'",
                                type));
            return nullptr;
        }
        // Copy the std::function out while holding the lock, then invoke it
        // unlocked — the factory function itself (a model constructor) may
        // take arbitrarily long and must never hold RegistryMutex.
        fn = it->second;
    }
    // `eff`, not `cfg`: the resolved artifact path must reach the model, or it
    // would look for model.index.json under the identity string.
    return fn(eff);
}

} // namespace lsxcommon