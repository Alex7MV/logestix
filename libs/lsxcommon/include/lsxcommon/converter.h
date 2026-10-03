#pragma once

#include "lsxcommon/schema.h"
#include "lsxcommon/vram_estimate.h"
#include <arrow/api.h>
#include <filesystem>
#include <memory>
#include <string>

namespace lsxcommon {

arrow::Result<PackedExpert> PackExpertNVFP4(const ExpertData &raw);

arrow::Status SaveExpertToArrow(
    const PackedExpert &expert,
    int layer_idx,
    int expert_idx,
    const std::filesystem::path &output_root);

// Reads only the shared-expert tensors (w{1,2,3}.weight + w{1,2,3}.scale)
// from the source safetensors shards and re-packs them into
// <output_root>/pinned/shared_experts.arrow. Used to regenerate the shared
// experts after a converter change without re-converting the full model.
arrow::Status RepackSharedExperts(
    const std::string &input_path,
    const std::filesystem::path &output_root);

// Reads only the routed-expert tensors (experts.N.w{1,2,3}.weight + .scale)
// from the source safetensors shards and re-writes
// <output_root>/experts/layer_XXX/exp_MMM.arrow for every expert. The source
// weights are already NVFP4 fp4-packed with F8_E8M0 scales, so this is a pure
// CPU passthrough — no re-quantization. Used to regenerate the routed experts
// after a converter change without re-converting the full model.
arrow::Status RepackRoutedExperts(
    const std::string &input_path,
    const std::filesystem::path &output_root);

// Recomputes the conversion-time VRAM profile from the SOURCE safetensors
// inventory (DeepSeek logestix-v1 family) and rewrites the "vram" block of
// <output_root>/logestix/model.index.json. Migrates pre-existing converted
// models without a full re-conversion.
arrow::Status RecomputeVramBlock(
    const std::string &input_path,
    const std::filesystem::path &output_root);

class Converter {
public:
    Converter();
    ~Converter();

    Converter(Converter &&) noexcept;
    Converter &operator=(Converter &&) noexcept;

    Converter &SetInputPath(const std::string &path);
    Converter &SetOutputPath(const std::string &path);
    Converter &SetVerbose(bool v);

    arrow::Status Run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

arrow::Status BuildTokenizerArrowFiles(
    const std::string &tokenizer_json_path,
    const std::filesystem::path &pinned_dir);

// TikToken (Kimi-K2.7-Code) tokenizer bridge. Parses the tiktoken ranks file
// ("base64 id" lines) plus the special tokens of tokenizer_config.json
// (added_tokens_decoder), reconstructs BPE merges from ranks and writes the
// same tokenizer_vocab.arrow / tokenizer_merges.arrow pair as
// BuildTokenizerArrowFiles. Sparse id ranges between base ranks and special
// ids are filled with placeholder rows.
arrow::Status BuildTokenizerArrowFilesTikToken(
    const std::string &tiktoken_model_path,
    const std::string &tokenizer_config_path,
    const std::filesystem::path &pinned_dir);

// SentencePiece-BPE bridge (AliceAI). Parses the protobuf `tokenizer.model`
// (identity normalizer, U+2581 word marker, byte fallback), derives BPE merge
// rules from the piece set + scores, and writes the same
// tokenizer_vocab.arrow / tokenizer_merges.arrow pair as BuildTokenizerArrowFiles
// with `tokenizer_kind="sp-bpe"` metadata.
// Piece names of the four special tokens to resolve their ids from. Defaults
// are the LLaMA-style names; a family whose tokenizer spells them differently
// (Xing: <_start>/<_end>/<_pad>) must pass its own — the ids are then taken from
// the piece set, never hardcoded.
struct SpSpecialTokens {
    const char *bos = "<s>";
    const char *eos = "</s>";
    const char *pad = "<pad>";
    const char *unk = "<unk>";
};

arrow::Status BuildTokenizerArrowFilesSentencePiece(
    const std::string &tokenizer_model_path,
    const std::filesystem::path &pinned_dir,
    const SpSpecialTokens &specials = {});

} // namespace lsxcommon
