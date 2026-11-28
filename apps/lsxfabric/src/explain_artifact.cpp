#include "explain_artifact.h"

#include "lsxstorage/artifact.h"
#include "lsxstorage/resolve.h"

#include <absl/strings/str_cat.h>

#include "lsxcommon/hardware_target.h"

#include <algorithm>
#include <string>
#include <vector>

namespace lsxfabric {

namespace {

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

    // Is the caller's argument a CHOICE rather than a model name?
    //
    // Decided by comparing what Resolve() honoured with what selection would
    // pick, not by searching the string for a profile name: the string heuristic
    // said "explicit" whenever a profile name appeared anywhere in the path, and
    // then re-ranked it anyway — a report that disagreed with the loader.
    const auto selected = lsx::storage::SelectProfile(
        resolved.org, resolved.model_dir, supported, preference);
    const bool explicit_profile =
        resolved.status == lsx::storage::ResolveStatus::kArtifact &&
        (!selected.has_value() || selected->dir != resolved.dir);

    std::vector<lsx::storage::Resolved> candidates;
    if (!resolved.org.empty() && !resolved.model_dir.empty())
        candidates =
            lsx::storage::ListProfiles(resolved.org, resolved.model_dir);

    out += absl::StrCat("model:     ", model_or_path, "\n");
    out += absl::StrCat("target:    supports [",
                        lsxcommon::hw::FormatMask(supported), "]\n");
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
        out += absl::StrCat("  ", cand.profile, "  codecs=[",
                            lsxcommon::hw::FormatMask(used),
                            "]  rank=", RankOf(preference, cand.profile),
                            "  ", fits ? "compatible" : "incompatible",
                            (explicit_profile && cand.profile == resolved.profile)
                                ? "  (named explicitly)"
                                : "",
                            "\n");
    }

    // The choice. A named profile is what the loader will use — BeginLoad hands
    // the path straight to Resolve() — so the report must not re-rank it; and a
    // named profile the target cannot execute is a REFUSAL, not a reason to
    // quietly substitute a sibling the operator did not ask for.
    if (explicit_profile) {
        const auto read = lsx::storage::ReadArtifact(resolved.dir);
        const uint64_t used = read.status == lsx::storage::ReadStatus::kOk
                                  ? read.info.used_codecs
                                  : 0;
        if ((used & ~supported) != 0) {
            out += absl::StrCat("chosen:    ", resolved.profile,
                                "  REFUSED on this target (needs [",
                                lsxcommon::hw::FormatMask(used),
                                "], target has [",
                                lsxcommon::hw::FormatMask(supported), "])\n");
            out += absl::StrCat("           a compatible sibling exists: ",
                                selected.has_value() ? selected->profile
                                                     : std::string("none"),
                                "\n");
            if (rc != nullptr) *rc = 1;
            return out;
        }
        out += absl::StrCat("chosen:    ", resolved.profile, "  -> ",
                            resolved.dir.string(), "  (named explicitly)\n");
        if (rc != nullptr) *rc = 0;
        return out;
    }

    if (!selected.has_value()) {
        out += "chosen:    no compatible profile\n";
        if (rc != nullptr) *rc = 1;
        return out;
    }
    out += absl::StrCat("chosen:    ", selected->profile, "  -> ",
                        selected->dir.string(), "\n");
    if (rc != nullptr) *rc = 0;
    return out;
}

} // namespace lsxfabric
