#pragma once

// ── Locating an artifact from whatever the caller passed ───────────────────
//
// cfg_.model_path reaches the loader in three shapes, so resolution is one
// function with one precedence instead of ad-hoc path joins scattered across 40
// families:
//
//   1. an ARTIFACT directory      <root>/<family>/<model_type>/<profile>
//   2. a SOURCE checkpoint        <LSX_SOURCE_ROOT>/<Model>
//   3. a bare model name          "phi-4"
//
// There is no legacy branch: the pre-contract `<vault>/<Model>/logestix/`
// layout is not loadable any more, and a resolve status that could only ever
// fail would be a trap. A source directory resolves to kSourceOnly so the caller
// can say "convert it" rather than "not found".
//
// Deliberately separate from artifact.h: this decides WHERE, artifact.h decides
// WHAT, and the gate is the one place both are consulted.

#include <filesystem>
#include <string>
#include <vector>

namespace lsx::storage {

namespace fs = std::filesystem;

enum class ResolveStatus {
    kArtifact,    // dir holds artifact.fb: loadable
    kSourceOnly,  // a checkpoint, no artifact.fb: must be converted first
    kNotFound,    // nothing at that path and no artifact of that name
};

const char *ResolveStatusName(ResolveStatus s);

struct Resolved {
    ResolveStatus status = ResolveStatus::kNotFound;
    fs::path dir;          // the directory to hand the loaders
    std::string model_type;
    std::string family;
    std::string profile;
    std::string org;       // "microsoft": the artifact store's first segment
    std::string model_dir; // "phi-4": the second, in the store's spelling
    std::string detail;    // human-readable reason, always set
};

// Resolves `model_or_path`.
//
// When the argument is already an artifact directory it is returned as-is,
// which is what keeps `lsxfabric --localmodel <artifact>` and every test that
// passes an explicit tree working unchanged.
Resolved Resolve(const std::string &model_or_path);

// Every artifact profile of one model that exists under the artifact root,
// sorted (profile name ascending) for a deterministic choice. Empty when the
// model has never been converted.
// Enumerates the profiles of ONE model, identified by its store coordinates
// (<root>/<org>/<model_dir>/<profile>), sorted by profile name.
std::vector<Resolved> ListProfiles(const std::string &org,
                                   const std::string &model_dir);

} // namespace lsx::storage
