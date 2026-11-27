#include "lsxstorage/roots.h"

#include <algorithm>
#include <cstdlib>

#if defined(_WIN32)
#include "lsxcommon/path_registry.h"  // only for the platform default base
#endif

namespace lsx::storage {

namespace {

// Platform default for the checkpoint vault. Kept byte-identical to
// lsxcommon::kModelRootBase so a tree configured today and a tree configured
// before the split resolve the same sources.
#if defined(_WIN32)
constexpr const char *kDefaultSourceBase = R"(D:\Programs\lsx_model)";
#else
constexpr const char *kDefaultSourceBase = "/mnt/fast_storage/lsx_model";
#endif

const char *NonEmptyEnv(const char *name)
{
    const char *v = std::getenv(name);
    return (v != nullptr && *v != '\0') ? v : nullptr;
}

} // namespace

fs::path SourceRoot()
{
    if (const char *v = NonEmptyEnv("LSX_SOURCE_ROOT")) return fs::path(v);
    if (const char *v = NonEmptyEnv("LSX_MODEL_ROOT")) return fs::path(v);
    return fs::path(kDefaultSourceBase);
}

std::optional<fs::path> ArtifactRoot()
{
    const char *v = NonEmptyEnv("LSX_ARTIFACT_ROOT");
    if (v == nullptr) return std::nullopt;
    fs::path p(v);
    // A relative root would make the artifact layout depend on the process's
    // working directory, so two runs from different shells would write to
    // different places. Treat it as unset and let the caller report it.
    if (p.is_relative()) return std::nullopt;
    return p;
}

fs::path CacheRoot()
{
    if (const char *v = NonEmptyEnv("LSX_CACHE_DIR")) return fs::path(v);
#if defined(_WIN32)
    if (const char *lad = NonEmptyEnv("LOCALAPPDATA"))
        return fs::path(lad) / "cache";
#else
    if (const char *xdg = NonEmptyEnv("XDG_CACHE_HOME"))
        return fs::path(xdg);
    if (const char *home = NonEmptyEnv("HOME"))
        return fs::path(home) / ".cache";
#endif
    return fs::temp_directory_path() / "logestix-cache";
}

fs::path ArtifactDir(const std::string &org, const std::string &model_dir,
                     const std::string &profile)
{
    const auto root = ArtifactRoot();
    if (!root.has_value()) return {};
    if (org.empty() || model_dir.empty() || profile.empty()) return {};
    return *root / org / model_dir / profile;
}

namespace {

// A checkpoint is a MODEL directory, not an org directory.
//
// config.json alone is not enough: Mistral's own download layout ships
// params.json + consolidated*.safetensors and no config.json at all, so a
// config.json-only predicate silently hides those models. The three signals are
// cheap and a model directory has at least one of them; an org directory has
// none, because it holds only subdirectories.
bool HoldsCheckpoint(const fs::path &dir)
{
    std::error_code ec;
    if (fs::is_regular_file(dir / "config.json", ec)) return true;
    if (fs::is_regular_file(dir / "params.json", ec)) return true;

    std::error_code iter_ec;
    for (const auto &entry : fs::directory_iterator(dir, iter_ec)) {
        if (iter_ec) break;
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".safetensors") return true;
    }
    return false;
}

} // namespace

std::vector<fs::path> ListPlaceholderDirs(const std::string &model)
{
    std::vector<fs::path> out;
    const fs::path root = SourceRoot();
    std::error_code org_ec;
    for (const auto &org_entry : fs::directory_iterator(root, org_ec)) {
        if (org_ec) break;
        if (!org_entry.is_directory()) continue;
        std::error_code model_ec;
        for (const auto &model_entry :
             fs::directory_iterator(org_entry.path(), model_ec)) {
            if (model_ec) break;
            if (!model_entry.is_directory()) continue;
            const std::string name = model_entry.path().filename().string();
            if (!model.empty() && name != model) continue;
            if (!HoldsCheckpoint(model_entry.path()))
                out.push_back(model_entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<SourceModelRef> ListSourceModels()
{
    std::vector<SourceModelRef> out;
    const fs::path root = SourceRoot();

    std::error_code org_ec;
    for (const auto &org_entry : fs::directory_iterator(root, org_ec)) {
        if (org_ec) break;
        if (!org_entry.is_directory()) continue;
        std::error_code model_ec;
        for (const auto &model_entry :
             fs::directory_iterator(org_entry.path(), model_ec)) {
            if (model_ec) break;
            if (!model_entry.is_directory()) continue;
            if (!HoldsCheckpoint(model_entry.path())) continue;
            out.push_back({org_entry.path().filename().string(),
                           model_entry.path().filename().string(),
                           model_entry.path()});
        }
    }
    std::sort(out.begin(), out.end(),
              [](const SourceModelRef &a, const SourceModelRef &b) {
                  if (a.org != b.org) return a.org < b.org;
                  return a.model < b.model;
              });
    return out;
}

std::optional<fs::path> FindSourceModel(const std::string &model,
                                        std::string *why)
{
    if (model.empty()) {
        if (why != nullptr) *why = "empty model name";
        return std::nullopt;
    }

    // "org/model" is matched exactly rather than searched: a qualified name is
    // the user telling us which one they mean, so a name that matches only by
    // model must NOT resolve.
    const auto slash = model.find('/');
    const bool qualified = slash != std::string::npos;

    std::vector<SourceModelRef> matches;
    for (const auto &ref : ListSourceModels()) {
        if (qualified) {
            if (ref.org == model.substr(0, slash) &&
                ref.model == model.substr(slash + 1))
                matches.push_back(ref);
        } else if (ref.model == model) {
            matches.push_back(ref);
        }
    }

    if (matches.empty()) {
        // Distinguish "nobody ever named this" from "a placeholder exists and
        // its contents are missing": an empty <org>/<model> directory is a
        // deliberate record of where an evicted checkpoint belongs, and telling
        // the operator to check their spelling would waste the record.
        const auto placeholders = ListPlaceholderDirs(model);
        if (why != nullptr) {
            if (!placeholders.empty()) {
                std::string list;
                for (const auto &p : placeholders) {
                    if (!list.empty()) list += ", ";
                    list += p.parent_path().filename().string() + "/" +
                            p.filename().string();
                }
                *why = "'" + model + "' is a placeholder directory (" + list +
                       ") with no weights: the checkpoint is not downloaded";
            } else {
                *why = "'" + model + "' is not a checkpoint under " +
                       SourceRoot().string() + " (<org>/<model>)";
            }
        }
        return std::nullopt;
    }
    if (matches.size() > 1) {
        // Never pick one: the choice would depend on directory iteration order,
        // so two runs could resolve different checkpoints.
        std::string list;
        for (const auto &m : matches) {
            if (!list.empty()) list += ", ";
            list += m.org + "/" + m.model;
        }
        if (why != nullptr)
            *why = "'" + model + "' is ambiguous under " + SourceRoot().string() +
                   ": " + list + " (qualify it as <org>/<model>)";
        return std::nullopt;
    }
    if (why != nullptr) why->clear();
    return matches.front().dir;
}

} // namespace lsx::storage
