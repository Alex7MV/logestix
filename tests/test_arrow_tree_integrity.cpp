// Arrow IPC tree integrity — validates the CONVERTED artifacts of every
// adapted model through the shared verifier (lsxcommon::arrow_check::
// VerifyArrowTree, also exposed as `lsxforge --verify`).
//
// The vault is optional: absent root / no converted tree -> CTest SKIP (4).
// LSX_ARROW_AUDIT_ROOT overrides the vault, LSX_ARROW_AUDIT_MAX_BYTES the
// per-tree deep-sample budget, LSX_ARROW_AUDIT_FULL=1 verifies every payload.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "lsxcommon/arrow_tree_verify.h"

namespace fs = std::filesystem;
namespace av = lsxcommon::arrow_check;

namespace {

std::string RootDir()
{
    if (const char *env = std::getenv("LSX_ARROW_AUDIT_ROOT"))
        return env;
    return "/mnt/fast_storage/huggingface";
}

int64_t BudgetBytes()
{
    if (const char *env = std::getenv("LSX_ARROW_AUDIT_MAX_BYTES")) {
        const long long v = std::atoll(env);
        if (v > 0) return v;
    }
    return int64_t{256} << 20;
}

bool FullMode()
{
    const char *env = std::getenv("LSX_ARROW_AUDIT_FULL");
    return env && env[0] == '1';
}

std::string SummaryLine(const std::string &model, const av::TreeVerifyStats &st)
{
    std::ostringstream os;
    os << model << ": files=" << st.files << " footer_ok=" << st.footer_ok
       << " deep=" << st.deep << " deep_MiB=" << (st.deep_bytes >> 20)
       << " budget_skips=" << st.budget_skips
       << " problems=" << st.problems.size();
    for (const auto &[kind, n] : st.kinds)
        os << " " << av::TreeFileKindName(kind) << "=" << n;
    if (!st.block_hist.empty()) {
        os << " block={";
        bool first = true;
        for (const auto &[b, n] : st.block_hist) {
            if (!first) os << ",";
            first = false;
            os << b << ":" << n;
        }
        os << "}";
    }
    return os.str();
}

} // namespace

TEST_CASE("Every converted model tree has a valid Arrow IPC layout",
          "[arrow][tree][real-model]")
{
    const fs::path root = RootDir();
    if (!fs::exists(root))
        SKIP("Arrow audit root not present: " << root.string());

    const auto trees = av::FindConvertedTrees(root);
    if (trees.empty())
        SKIP("no converted logestix/ tree under " << root.string());

    av::TreeVerifyOptions opts;
    opts.full = FullMode();
    opts.budget_bytes = BudgetBytes();

    std::ostringstream summary;
    std::ostringstream problems;
    int failures = 0;
    for (const auto &tree : trees) {
        const auto st = av::VerifyArrowTree(tree, opts);
        const std::string model =
            fs::path(tree).parent_path().filename().string();
        summary << SummaryLine(model, st) << "\n";
        if (!st.problems.empty()) {
            ++failures;
            for (std::size_t i = 0; i < st.problems.size() && i < 20; ++i)
                problems << "\n  " << model << ": " << st.problems[i];
            if (st.problems.size() > 20)
                problems << "\n  " << model << ": ... and "
                         << (st.problems.size() - 20) << " more";
        }
    }
    WARN(summary.str());
    INFO(problems.str());
    CHECK(failures == 0);
}
