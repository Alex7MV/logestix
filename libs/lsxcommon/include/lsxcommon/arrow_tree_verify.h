// arrow_tree_verify.h — shared Arrow IPC artifact verification.
//
// The single implementation behind the audit test (test_arrow_tree_integrity)
// and the post-conversion hook (`lsxforge --verify`). It checks a converted
// `logestix/` tree: every .arrow opens, has one known schema and one record
// batch with the expected row count, and a deterministic payload sample
// satisfies the offset/range/size invariants (lsxcommon/arrow_validate.h).
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace lsxcommon {
namespace arrow_check {

enum class TreeFileKind {
    kExpert,
    kSharedExpert,
    kDenseNvfp4,
    kPinned,
    kVocab,
    kMerges,
    kUnknown,
};

inline const char *TreeFileKindName(TreeFileKind k)
{
    switch (k) {
    case TreeFileKind::kExpert: return "expert";
    case TreeFileKind::kSharedExpert: return "shared-expert";
    case TreeFileKind::kDenseNvfp4: return "dense-nvfp4";
    case TreeFileKind::kPinned: return "pinned";
    case TreeFileKind::kVocab: return "tokenizer-vocab";
    case TreeFileKind::kMerges: return "tokenizer-merges";
    default: return "unknown";
    }
}

struct TreeVerifyOptions {
    // false = footer pass for every file + a bounded deep sample.
    bool full = false;
    // Deep-sample payload budget per tree (ignored when full).
    int64_t budget_bytes = int64_t{256} << 20;
};

struct TreeVerifyStats {
    int64_t files = 0;
    int64_t footer_ok = 0;
    int64_t deep = 0;
    int64_t deep_bytes = 0;
    int64_t budget_skips = 0;
    std::map<TreeFileKind, int64_t> kinds;
    std::map<int64_t, int64_t> block_hist;   // implied elems-per-scale
    std::vector<std::string> problems;
};

// Verifies one converted tree root (the directory that contains pinned/ and
// experts/). A missing root is reported as a problem, not an exception.
TreeVerifyStats VerifyArrowTree(const std::filesystem::path &converted_root,
                                const TreeVerifyOptions &opts = {});

// Returns <vault>/<model>/logestix for every model that has a converted tree.
std::vector<std::filesystem::path>
FindConvertedTrees(const std::filesystem::path &vault_root);

} // namespace arrow_check
} // namespace lsxcommon
