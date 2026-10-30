#pragma once

#include "lsxcommon/schema.h"
#include "lsxcommon/vram_estimate.h"
#include <arrow/api.h>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

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

    // Tensor-name prefixes cut from the conversion (stored-not-applied
    // ballast, e.g. DeepSeek-V4-Pro's "mtp." DSpark stage). Every cut tensor
    // is counted in model.index.json's "deferred" block — a silent drop is
    // not a strip.
    Converter &SetSkipPrefixes(std::vector<std::string> prefixes);

    // Override the indexed "model" name. The default is the input directory
    // name, which is what the runtime's type detection normalizes; a family
    // pipeline pins its canonical spelling so a renamed download still
    // resolves to its kType.
    Converter &SetModelName(std::string name);

    // Frozen-passport gate: when set and synthetic geometry is not allowed,
    // the source config + tensor inventory must match exactly. An unset gate
    // keeps the generic converter's historical behaviour.
    Converter &SetExpectedGeometry(int layers, int hidden, int vocab,
                                   int experts);

    arrow::Status Run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Per-family override of the special-token ids the byte-level BPE bridge
// writes. INT32_MIN keeps the derivation from tokenizer_config.json. A
// checkpoint whose reference tokenizer emits NO [BOS] passes bos_id = -1 (its
// chat template spells the marker out as text, so an auto-prepended BOS would
// duplicate it); a stop id that differs from its tokenizer_config entry is
// passed explicitly. Same contract as TikTokenSpecialPolicy below.
struct TokenizerSpecialPolicy {
    int32_t bos_id = INT32_MIN;
    int32_t eos_id = INT32_MIN;
    int32_t pad_id = INT32_MIN;
    // Pre-tokenizer family of a byte-level BPE checkpoint:
    //   0 = legacy GPT-2 / cl100k 3-split (WordStream; default)
    //   1 = Llama-3 / o200k 7-alternative pattern (Llama3WordStream: case runs,
    //       no CJK split). Written as tokenizer_kind = 3.
    //   2 = Flint 3-split: \p{N}{1,3} + CJK Isolated over the o200k case-run
    //       pattern with [\p{P}\p{S}]+[\r\n]* (FlintWordStream).
    //       Written as tokenizer_kind = 4.
    int32_t pretokenizer = 0;
    // 1 = tolerate a vocabulary with no single-byte token for some bytes: the
    // missing ones map to -1 and the encoder skips them. The reference tokenizer
    // of Hy-Memory-30B-A3B drops CR ("\r" encodes to no token) and omits the 13
    // UTF-8-illegal lead bytes (0xC0, 0xC1, 0xF5..0xFF); without this the bridge
    // refuses the checkpoint. Default 0 keeps every other family strict.
    int32_t drop_unmapped_bytes = 0;
};

arrow::Status BuildTokenizerArrowFiles(
    const std::string &tokenizer_json_path,
    const std::filesystem::path &pinned_dir,
    const TokenizerSpecialPolicy &policy = {});

// TikToken (Kimi-K2.7-Code) tokenizer bridge. Parses the tiktoken ranks file
// ("base64 id" lines) plus the special tokens of tokenizer_config.json
// (added_tokens_decoder), reconstructs BPE merges from ranks and writes the
// same tokenizer_vocab.arrow / tokenizer_merges.arrow pair as
// BuildTokenizerArrowFiles. Sparse id ranges between base ranks and special
// ids are filled with placeholder rows.
// Per-family override of the special-token ids the tiktoken bridge writes.
// INT32_MIN keeps the historical derivation from tokenizer_config.json. A
// checkpoint whose reference tokenizer never emits [BOS] passes bos_id = -1;
// a stop id that differs from its tokenizer_config [EOS] entry (Kimi:
// config.json eos_token_id = <|im_end|>) is passed explicitly.
struct TikTokenSpecialPolicy {
    int32_t bos_id = INT32_MIN;
    int32_t eos_id = INT32_MIN;
    int32_t pad_id = INT32_MIN;
};
arrow::Status BuildTokenizerArrowFilesTikToken(
    const std::string &tiktoken_model_path,
    const std::string &tokenizer_config_path,
    const std::filesystem::path &pinned_dir,
    const TikTokenSpecialPolicy &policy = {});

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
