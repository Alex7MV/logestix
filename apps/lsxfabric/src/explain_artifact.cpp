#include "explain_artifact.h"

#include "lsxstorage/artifact.h"
#include "lsxstorage/resolve.h"

#include <absl/strings/str_cat.h>

#include <algorithm>
#include <string>
#include <vector>

namespace lsxfabric {

namespace {

// "f32,bf16,nvfp4" for the mask, in ascending bit order — the same spelling the
// load gate uses, so an operator comparing the two logs sees the same words.
std::string MaskText(uint64_t mask)
{
    static const struct {
        uint64_t bit;
        const char *name;
    } kRows[] = {{1ull << 0, "f32"},   {1ull << 1, "bf16"},  {1ull << 2, "f16"},
                 {1ull << 3, "i8"},    {1ull << 4, "mxfp8"}, {1ull << 5, "nvfp4"},
                 {1ull << 6, "mxfp4"}};
    if (mask == 0) return "none";
    std::string out;
    for (const auto &row : kRows) {
        if ((mask & row.bit) == 0) continue;
        if (!out.empty()) out.push_back(',');
        out.append(row.name);
    }
    return out;
}

int RankOf(const std::vector<std::string> &preference, const std::string &name)
{
    for (size_t i = 0; i < preference.size(); ++i)
        if (preference[i] == name) return static_cast<int>(i);
    // Unranked profiles sort after every ranked one: the table is a preference,
    // not a filter, so a profile added later is still visible — and still loses
    // to a ranked one, which is what "preference" means.
    return static_cast<int>(preference.size());
}

} // namespace

std::string ExplainArtifact(const std::string &model_or_path, uint64_t supported,
                            const std::vector<std::string> &preference, int *rc)
{
    std::string out;
    const auto resolved =
        lsx::storage::Resolve(model_or_path, supported, preference);

    // An explicit profile is a choice: report it and stop. Re-ranking it would
    // answer a question the operator did not ask.
    const bool explicit_profile =
        resolved.status == lsx::storage::ResolveStatus::kArtifact &&
        !resolved.profile.empty() &&
        (model_or_path.find(resolved.profile) != std::string::npos);

    std::vector<lsx::storage::Resolved> candidates;
    if (!resolved.org.empty() && !resolved.model_dir.empty())
        candidates =
            lsx::storage::ListProfiles(resolved.org, resolved.model_dir);

    out += absl::StrCat("model:     ", model_or_path, "\n");
    out += absl::StrCat("target:    supports [", MaskText(supported), "]\n");
    if (resolved.org.empty() || resolved.model_dir.empty()) {
        out += absl::StrCat("resolved:  ",
                            lsx::storage::ResolveStatusName(resolved.status), " (",
                            resolved.detail, ")\n");
        out += "chosen:    none (nothing to choose from)\n";
        if (rc != nullptr) *rc = 1;
        return out;
    }
    out += absl::StrCat("store:     ", resolved.org, "/", resolved.model_dir, "\n");

    if (candidates.empty()) {
        out += "candidates: none (this model has no converted artifact)\n";
        out += "chosen:    none\n";
        if (rc != nullptr) *rc = 1;
        return out;
    }

    out += "candidates:\n";
    for (const auto &cand : candidates) {
        const auto read = lsx::storage::ReadArtifact(cand.dir);
        if (read.status != lsx::storage::ReadStatus::kOk) {
            out += absl::StrCat("  ", cand.profile, "  UNREADABLE artifact.fb (",
                                lsx::storage::ReadStatusName(read.status), ")\n");
            continue;
        }
        const uint64_t used = read.info.used_codecs;
        const bool fits = (used & ~supported) == 0;
        out += absl::StrCat("  ", cand.profile, "  codecs=[", MaskText(used),
                            "]  rank=", RankOf(preference, cand.profile),
                            "  ", fits ? "compatible" : "incompatible",
                            (explicit_profile && cand.profile == resolved.profile)
                                ? "  (named explicitly)"
                                : "",
                            "\n");
    }

    // The choice: what the loader would take for this target.
    const auto chosen = lsx::storage::SelectProfile(resolved.org, resolved.model_dir,
                                                   supported, preference);
    if (!chosen.has_value()) {
        out += "chosen:    no compatible profile\n";
        if (rc != nullptr) *rc = 1;
        return out;
    }
    out += absl::StrCat("chosen:    ", chosen->profile, "  -> ",
                        chosen->dir.string(), "\n");
    if (rc != nullptr) *rc = 0;
    return out;
}

} // namespace lsxfabric
