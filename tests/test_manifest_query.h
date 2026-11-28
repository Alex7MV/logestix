#pragma once

// Host-only accessors over the generated model manifest table.
//
// `tests/model_manifest.h` is generated from `models/<org>/<model>.json` at configure time
// (cmake/LsxModelManifest.cmake). Every guard and gate driver reads the model
// registry through these functions instead of keeping its own hand-written
// rows — that is what makes "one file per model" true rather than aspirational.
//
// Not reachable from .cu/.cuh: it pulls <filesystem>/<optional>/<string> only,
// and it is included by tests, which are host-only by construction.

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "model_manifest.h"

namespace manifest_query {

// The row for `type`, or nullptr when no manifest declares it.
inline const lsxmanifest::ModelRow *FindModel(std::string_view type)
{
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
        if (type == lsxmanifest::kModels[i].model_type)
            return &lsxmanifest::kModels[i];
    return nullptr;
}

// Targets of `model`; pass `stage` to narrow to one lifecycle stage.
inline std::vector<const lsxmanifest::TargetRow *> TargetsOf(
    std::string_view model,
    std::optional<std::string_view> stage = std::nullopt)
{
    std::vector<const lsxmanifest::TargetRow *> out;
    for (std::size_t i = 0; i < lsxmanifest::kTargetCount; ++i)
    {
        const auto &t = lsxmanifest::kTargets[i];
        if (model != t.model_type) continue;
        if (stage && *stage != t.stage) continue;
        out.push_back(&t);
    }
    return out;
}

// The TUs that must wire the shared lifecycle contract for `model`.
inline std::vector<std::string> LifecycleTus(std::string_view model)
{
    std::vector<std::string> out;
    for (std::size_t i = 0; i < lsxmanifest::kLifecycleCount; ++i)
        if (model == lsxmanifest::kLifecycle[i].model_type)
            out.emplace_back(lsxmanifest::kLifecycle[i].tu);
    return out;
}

// True when the manifest declares a gate reason instead of a target: the
// absence is a recorded debt, not silence.
inline bool IsDeclaredDebt(std::string_view model, std::string_view stage)
{
    for (std::size_t i = 0; i < lsxmanifest::kDebtCount; ++i)
        if (model == lsxmanifest::kDebt[i].model_type &&
            stage == lsxmanifest::kDebt[i].stage)
            return true;
    return false;
}

// Where this model's ARTIFACT lives: the per-model env override first, then
// <LSX_ARTIFACT_ROOT>/<org>/<model_dir>/<profile>. Defined in
// test_manifest_query_root.h, which needs lsxstorage; this declaration lets a
// guard name the type without linking it. Tests never spell a store directory
// name themselves - that is how two models came to point at one tree.
inline std::filesystem::path ModelRoot(const lsxmanifest::ModelRow &row);


// A SKIP message that names THIS model and the directory it looked in, so a
// clean skip cannot be mistaken for a neighbour's.
inline std::string SkipMessage(const lsxmanifest::ModelRow &row,
                               std::string_view what)
{
    std::string out(row.model_type);
    out += " ";
    out += what;
    out += " not present in ";
    out += row.vault_org;
    out += "/";
    out += row.vault_dir;
    out += "/";
    out += row.vault_profile;
    return out;
}

// Every distinct `family` value in the table, sorted. The cross-family
// borrowing check iterates this instead of its own literal list, which is how
// five families (beam, hy, minimax, mistral, xing) went unchecked.
inline std::vector<std::string> Families()
{
    std::vector<std::string> out;
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
        out.emplace_back(lsxmanifest::kModels[i].family);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

} // namespace manifest_query
