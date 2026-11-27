#include "lsxstorage/resolve.h"

#include "lsxstorage/artifact.h"
#include "lsxstorage/roots.h"

#include <algorithm>
#include <system_error>

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

    // (1) An artifact directory, or a path inside one: accept the profile dir
    // itself, the conventional `logestix` subdir an older caller may still
    // spell, and the parent of a trailing slash.
    for (const auto &cand : {given, given}) {
        if (IsArtifactDir(cand)) {
            r.status = ResolveStatus::kArtifact;
            r.dir = cand;
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

    // (3) A bare model name, or a path into a model: look for its artifacts.
    // The name may be either a vault directory or a manifest model_type; both
    // are searched because both are things a user types.
    const auto root = ArtifactRoot();
    if (root.has_value()) {
        // One error_code per loop level: sharing one across nested iterators
        // means an inner failure (a permission error in one model directory)
        // aborts the whole search.
        std::error_code fam_ec;
        for (const auto &family_dir : fs::directory_iterator(*root, fam_ec)) {
            if (fam_ec) break;
            if (!family_dir.is_directory()) continue;
            std::error_code model_ec;
            for (const auto &model_dir :
                 fs::directory_iterator(family_dir.path(), model_ec)) {
                if (model_ec) break;
                if (!model_dir.is_directory()) continue;
                const std::string name = model_dir.path().filename().string();
                if (name != model_or_path &&
                    name != fs::path(model_or_path).filename().string())
                    continue;
                std::error_code prof_ec;
                for (const auto &profile_dir :
                     fs::directory_iterator(model_dir.path(), prof_ec)) {
                    if (prof_ec) break;
                    if (!profile_dir.is_directory()) continue;
                    if (!IsArtifactDir(profile_dir.path())) continue;
                    r.status = ResolveStatus::kArtifact;
                    r.dir = profile_dir.path();
                    r.org = family_dir.path().filename().string();
                    r.model_dir = name;
                    r.profile = profile_dir.path().filename().string();
                    const auto read = ReadArtifact(profile_dir.path());
                    if (read.status == ReadStatus::kOk) {
                        r.family = read.info.family;
                        r.model_type = read.info.model_type;
                    }
                    return r;
                }
                // The model exists but has no converted profile.
                r.status = ResolveStatus::kSourceOnly;
                r.model_dir = name;
                r.org = family_dir.path().filename().string();
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
