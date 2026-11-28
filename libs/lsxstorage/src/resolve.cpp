#include "lsxstorage/resolve.h"

#include "lsxstorage/artifact.h"
#include "lsxstorage/roots.h"

#include <algorithm>
#include <string>
#include <system_error>
#include <vector>

namespace lsx::storage {

namespace {

bool IsDirectory(const fs::path &p)
{
    std::error_code ec;
    return !p.empty() && fs::is_directory(p, ec);
}

// A source checkpoint: config.json is what every converter reads first, so its
// presence is the cheapest reliable signal that this is a download, not an
// artifact and not a random directory.
bool LooksLikeSourceModel(const fs::path &p)
{
    std::error_code ec;
    return fs::is_regular_file(p / "config.json", ec);
}

} // namespace

const char *ResolveStatusName(ResolveStatus s)
{
    switch (s) {
    case ResolveStatus::kArtifact: return "artifact";
    case ResolveStatus::kSourceOnly: return "source-only";
    case ResolveStatus::kNotFound: return "not-found";
    }
    return "unknown";
}

Resolved Resolve(const std::string &model_or_path)
{
    Resolved r;
    if (model_or_path.empty()) {
        r.detail = "empty model path";
        return r;
    }

    const fs::path given(model_or_path);

    // (1) An artifact directory, or a path one level inside one: `ls` hands out
    // file paths, so `--model <artifact>/model.index.json` must name the same
    // tree as `--model <artifact>`. (The pre-split layout had a `logestix`
    // subdir to accept as well; the profile directory IS the artifact now, so
    // that spelling is gone.)
    for (const auto &cand : {given, given.parent_path()}) {
        if (IsArtifactDir(cand)) {
            r.status = ResolveStatus::kArtifact;
            r.dir = cand;
            // The artifact records the store coordinates it belongs to, so the
            // identity fields come from the file rather than from whatever
            // path the caller happened to type.
            const auto read = ReadArtifact(cand);
            if (read.status == ReadStatus::kOk) {
                r.family = read.info.family;
                r.model_type = read.info.model_type;
                r.org = read.info.org;
                r.model_dir = read.info.model_dir;
                r.profile = read.info.profile;
            }
            if (r.profile.empty()) r.profile = cand.filename().string();
            return r;
        }
    }

    // (2) A directory that exists but is not an artifact: report WHICH kind of
    // non-artifact it is. "You pointed at the checkpoint" and "that path does
    // not exist" send an operator to two different fixes.
    if (IsDirectory(given)) {
        if (LooksLikeSourceModel(given)) {
            r.status = ResolveStatus::kSourceOnly;
            r.dir = given;
            r.detail = "source checkpoint (no artifact.fb): convert it with "
                       "lsxforge --<family> --input <dir> --codec-profile <p>";
            return r;
        }
        r.status = ResolveStatus::kSourceOnly;
        r.dir = given;
        r.detail = "directory exists but holds no artifact.fb and no config.json";
        return r;
    }

    // (2b) An ORG-QUALIFIED name, relative to the artifact root:
    //     "<org>/<model>"            one of its profiles
    //     "<org>/<model>/<profile>"  that profile
    // It is the store's own layout, so it is what `ls` prints and what an
    // operator copies — and it is a path relative to the ROOT, not to the cwd,
    // so it must be joined to the root explicitly. Nothing else may answer it:
    // in particular not the leaf-name search below, which would resolve
    // "WrongOrg/Qwen3.5-9B" to Qwen's tree and hide the org the caller named.
    const fs::path given_path(model_or_path);
    const std::string wanted_leaf = given_path.filename().string();
    std::vector<std::string> rel_parts;
    for (const auto &part : given_path)
        rel_parts.push_back(part.string());
    const bool org_qualified = !given_path.is_absolute() &&
                               (rel_parts.size() == 2 || rel_parts.size() == 3);
    const auto root = ArtifactRoot();
    if (org_qualified && root.has_value()) {
        const fs::path model_dir = *root / rel_parts[0] / rel_parts[1];
        std::error_code ec;
        if (fs::is_directory(model_dir, ec)) {
            // An explicit profile is a choice: it resolves or it does not, and
            // never falls back to a sibling profile.
            if (rel_parts.size() == 3) {
                const fs::path wanted = model_dir / rel_parts[2];
                if (IsArtifactDir(wanted)) {
                    r.status = ResolveStatus::kArtifact;
                    r.dir = wanted;
                    r.org = rel_parts[0];
                    r.model_dir = rel_parts[1];
                    r.profile = rel_parts[2];
                    const auto read = ReadArtifact(wanted);
                    if (read.status == ReadStatus::kOk) {
                        r.family = read.info.family;
                        r.model_type = read.info.model_type;
                    }
                    return r;
                }
                r.detail = "'" + model_dir.string() + "' has no profile '" +
                           rel_parts[2] + "'";
                return r;
            }
            // No profile named: the first one that IS an artifact, in the same
            // sorted order ListProfiles uses, so two runs agree.
            std::error_code prof_ec;
            for (const auto &profile_dir :
                 fs::directory_iterator(model_dir, prof_ec)) {
                if (prof_ec) break;
                if (!profile_dir.is_directory()) continue;
                if (!IsArtifactDir(profile_dir.path())) continue;
                r.status = ResolveStatus::kArtifact;
                r.dir = profile_dir.path();
                r.org = rel_parts[0];
                r.model_dir = rel_parts[1];
                r.profile = profile_dir.path().filename().string();
                const auto read = ReadArtifact(profile_dir.path());
                if (read.status == ReadStatus::kOk) {
                    r.family = read.info.family;
                    r.model_type = read.info.model_type;
                }
                return r;
            }
            r.status = ResolveStatus::kSourceOnly;
            r.org = rel_parts[0];
            r.model_dir = rel_parts[1];
            r.detail = "'" + model_dir.string() +
                       "' has no converted artifact (no profile holds "
                       "artifact.fb)";
            return r;
        }
    }

    // (3) A bare model name, or a path into a model: look for its artifacts.
    // The name may be either a vault directory ("Qwen3.5-9B") or a manifest
    // model_type ("qwen3.5-9b"); both are searched because both are things a
    // user types, and only artifact.fb knows the second — the registry, the
    // gates and every document spell the model that way, so comparing the
    // directory name alone answered "not found" for the tree's own vocabulary.
    if (root.has_value() && !org_qualified) {
        // One error_code per loop level: sharing one across nested iterators
        // means an inner failure (a permission error in one model directory)
        // aborts the whole search.
        std::error_code org_ec;
        for (const auto &org_dir : fs::directory_iterator(*root, org_ec)) {
            if (org_ec) break;
            if (!org_dir.is_directory()) continue;
            std::error_code model_ec;
            for (const auto &model_dir :
                 fs::directory_iterator(org_dir.path(), model_ec)) {
                if (model_ec) break;
                if (!model_dir.is_directory()) continue;
                const std::string name = model_dir.path().filename().string();
                const bool by_dir = name == model_or_path || name == wanted_leaf;
                // An identity can only be known from an artifact, but a model
                // directory with NO profile still answers the caller's question
                // when the identity matches another profile of it — and when the
                // caller typed the model_type, "not converted yet" is the fix,
                // not "no such model".
                bool by_identity_any = false;
                std::error_code prof_ec;
                for (const auto &profile_dir :
                     fs::directory_iterator(model_dir.path(), prof_ec)) {
                    if (prof_ec) break;
                    if (!profile_dir.is_directory()) continue;
                    if (!IsArtifactDir(profile_dir.path())) continue;
                    // Identity comes from the artifact, not from the path: one
                    // read is what a caller needs anyway, and it cannot disagree
                    // with the directory it sits in.
                    const auto read = ReadArtifact(profile_dir.path());
                    const bool by_identity =
                        read.status == ReadStatus::kOk &&
                        (read.info.model_type == model_or_path ||
                         read.info.model_type == wanted_leaf);
                    by_identity_any = by_identity_any || by_identity;
                    if (!by_dir && !by_identity) continue;
                    r.status = ResolveStatus::kArtifact;
                    r.dir = profile_dir.path();
                    r.org = org_dir.path().filename().string();
                    r.model_dir = name;
                    r.profile = profile_dir.path().filename().string();
                    if (read.status == ReadStatus::kOk) {
                        r.family = read.info.family;
                        r.model_type = read.info.model_type;
                        r.org = read.info.org.empty() ? r.org : read.info.org;
                        r.model_dir = read.info.model_dir.empty() ? r.model_dir
                                                                  : read.info.model_dir;
                    }
                    return r;
                }
                if (!by_dir && !by_identity_any) continue;
                // The model exists but has no converted profile.
                r.status = ResolveStatus::kSourceOnly;
                r.model_dir = name;
                r.org = org_dir.path().filename().string();
                r.detail = "'" + name + "' has no converted artifact under " +
                           root->string();
                return r;
            }
        }
    }

    // (4) A source checkpoint by name, under <SourceRoot>/<org>/<model>.
    // The name may arrive bare ("phi-4") or already qualified ("microsoft/phi-4");
    // both are things a user types, and the qualified form skips the search.
    const fs::path given_rel(model_or_path);
    if (given_rel.has_parent_path() &&
        given_rel.parent_path() != given_rel.parent_path().parent_path()) {
        const auto qualified = SourceRoot() / given_rel;
        if (LooksLikeSourceModel(qualified)) {
            r.status = ResolveStatus::kSourceOnly;
            r.dir = qualified;
            r.detail = "source checkpoint (no artifact.fb): convert it with "
                       "lsxforge --<family> --input <dir> --codec-profile <p>";
            return r;
        }
    }
    std::string why;
    if (const auto source = FindSourceModel(model_or_path, &why);
        source.has_value()) {
        r.status = ResolveStatus::kSourceOnly;
        r.dir = *source;
        r.detail = "source checkpoint (no artifact.fb): convert it with "
                   "lsxforge --<family> --input <dir> --codec-profile <p>";
        return r;
    }

    r.status = ResolveStatus::kNotFound;
    r.detail = why.empty() ? ("no artifact and no source checkpoint for '" +
                              model_or_path + "'")
                           : why;
    // A manifest model_type ("glm-5.2") is an IDENTITY, not a directory name;
    // the store spells the directory "GLM-5.2". Say so, because the search above
    // matched artifacts by identity and the source search cannot — the registry
    // that knows the mapping sits above this layer.
    if (model_or_path.find('/') == std::string::npos) {
        r.detail += " (a model_type is not a directory name: under the source "
                    "root the store spells the directory as the download does)";
    }
    return r;
}

std::vector<Resolved> ListProfiles(const std::string &org,
                                   const std::string &model_dir_name)
{
    std::vector<Resolved> out;
    const auto root = ArtifactRoot();
    if (!root.has_value() || org.empty() || model_dir_name.empty()) return out;

    const auto model_dir = *root / org / model_dir_name;
    std::error_code ec;
    if (!fs::is_directory(model_dir, ec)) return out;

    for (const auto &entry : fs::directory_iterator(model_dir, ec)) {
        if (ec) break;
        if (!entry.is_directory()) continue;
        if (!IsArtifactDir(entry.path())) continue;
        Resolved r;
        r.status = ResolveStatus::kArtifact;
        r.dir = entry.path();
        r.org = org;
        r.model_dir = model_dir_name;
        r.profile = entry.path().filename().string();
        // Identity comes from artifact.fb, not from the path: one read is what
        // the caller needs anyway, and it cannot disagree with the directory.
        const auto read = ReadArtifact(entry.path());
        if (read.status == ReadStatus::kOk) {
            r.family = read.info.family;
            r.model_type = read.info.model_type;
        }
        out.push_back(std::move(r));
    }
    std::sort(out.begin(), out.end(),
              [](const Resolved &a, const Resolved &b) {
                  return a.profile < b.profile;
              });
    return out;
}

} // namespace lsx::storage
