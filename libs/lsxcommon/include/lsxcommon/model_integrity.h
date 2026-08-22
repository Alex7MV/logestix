#pragma once

#include "lsxcommon/safetensors.h"
#include "lsxcommon/tokenizer.h"
#include <arrow/api.h>
#include <filesystem>
#include <string>
#include <vector>
#include <absl/container/flat_hash_set.h>

namespace lsxcommon::integrity {

// ── Tokenizer ──

// Which special tokens a model's tokenizer must carry.
struct TokenizerSpecialIds {
    bool bos_required = true;
    bool eos_required = true;
    bool pad_required = false;
};

// Build binary tokenizer arrows from tokenizer.json (mandatory), then verify:
//   - re-mmap tokenizer_vocab.arrow + tokenizer_merges.arrow
//   - load through load_tokenizer_metadata()
//   - validate required bos/eos/pad against required_specials
//   - byte-exact encode→decode round-trip on two hardcoded samples
arrow::Status BuildAndVerifyTokenizer(
    const std::string &tokenizer_json_path,
    const std::filesystem::path &pinned_dir,
    const TokenizerSpecialIds &required_specials);

// ── Source integrity ──

// Raw-read every tensor at its header-claimed offsets; confirm byte size.
// Names shard+tensor on any empty/short/OOB read.
arrow::Status VerifyAllTensorReads(
    const std::filesystem::path &input_path,
    const SafetensorsModel &model);

// Check that every name in `expected` appears in `present`.
// Missing → hard error listing each; extra → spdlog::warn.
arrow::Status CheckTensorCoverage(
    const std::vector<std::string> &expected,
    const std::vector<std::string> &present,
    const std::string &what);

// Present layers must span a contiguous range 0..max_layer
// and max_layer+1 must equal expected_layers.
arrow::Status CheckLayerContiguity(
    int expected_layers,
    const absl::flat_hash_set<int> &present_layers);

// Verify a vector of {name, {expected, actual}} geometry fields.
// Any mismatch hard-fails naming field, expected, and actual.
arrow::Status CheckGeometryFields(
    const std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> &fields);

// ── Output verification ──

struct VerificationReport {
    int64_t pinned_tensor_count = 0;
    size_t arrow_files_checked = 0;
};

// Verifies a converted model directory: index parseable, format tag known,
// every *model-tensor* arrow file reopens + ValidateFull(), record counts
// match the index total_pinned_tensors. Tokenizer arrows are reopened for
// validate but excluded from the count. Reads index at
// output_root/logestix/model.index.json only. On any failure the returned
// Result carries a non-OK Status with a message naming the cause; on success
// the report holds the verified record totals.
arrow::Result<VerificationReport> VerifyConvertedModel(
    const std::filesystem::path &output_root);

} // namespace lsxcommon::integrity