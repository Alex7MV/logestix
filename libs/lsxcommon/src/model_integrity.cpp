#include "lsxcommon/model_integrity.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/log.h"
#include <arrow/api.h>
#include <absl/strings/str_cat.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

namespace lsxcommon::integrity {

namespace {

// Encode → decode round-trip check. Returns error if the decoded bytes
// differ from source or if encode/decode signals failure.
arrow::Status round_trip_sample(
    const ArrowTokenizerDb &db,
    const std::string &sample)
{
    int32_t token_buf[512];
    size_t max_tokens = 512;
    int32_t n = arrow_tokenizer_encode(
        db, sample.data(), sample.size(), token_buf, &max_tokens);
    if (n <= 0) {
        std::string msg = (n == 0)
            ? "encode returned 0 tokens"
            : absl::StrCat("encode failed with code ", n);
        return arrow::Status::Invalid("tokenizer round-trip: ", msg);
    }

    std::vector<char> text_buf(sample.size() * 4 + 64, '\0');
    size_t max_len = text_buf.size();
    int32_t rc = arrow_tokenizer_decode(
        db, token_buf, static_cast<size_t>(n),
        text_buf.data(), &max_len);
    if (rc != 0)
        return arrow::Status::Invalid(
            "tokenizer round-trip: decode failed with code ", rc);

    if (std::string(text_buf.data(), max_len) != sample)
        return arrow::Status::Invalid(
            "tokenizer round-trip: decoded text differs");

    return arrow::Status::OK();
}

} // anonymous namespace

namespace {

arrow::Status VerifyTokenizerArtifacts(
    const std::filesystem::path &pinned_dir,
    const TokenizerSpecialIds &required_specials)
{
    namespace fs = std::filesystem;

    auto vocab_path = pinned_dir / "tokenizer_vocab.arrow";
    auto merges_path = pinned_dir / "tokenizer_merges.arrow";

    if (!fs::exists(vocab_path))
        return arrow::Status::Invalid(
            "tokenizer: vocab file not built: ", vocab_path.string());
    if (!fs::exists(merges_path))
        return arrow::Status::Invalid(
            "tokenizer: merges file not built: ", merges_path.string());

    // (2) MMAP and load.
    ARROW_ASSIGN_OR_RAISE(auto vocab_mmap,
        arrow::io::MemoryMappedFile::Open(vocab_path.string(),
                                          arrow::io::FileMode::READ));
    ARROW_ASSIGN_OR_RAISE(auto merges_mmap,
        arrow::io::MemoryMappedFile::Open(merges_path.string(),
                                          arrow::io::FileMode::READ));

    ArrowTokenizerDb db;
    ARROW_RETURN_NOT_OK(load_tokenizer_metadata(vocab_mmap, merges_mmap, db));

    if (!db.valid())
        return arrow::Status::Invalid("tokenizer: DB not valid after load");

    // (3) Validate special ids.
    if (required_specials.bos_required && db.bos_id < 0)
        return arrow::Status::Invalid("tokenizer: required bos_id is missing");
    if (required_specials.eos_required && db.eos_id < 0)
        return arrow::Status::Invalid("tokenizer: required eos_id is missing");
    if (required_specials.pad_required && db.pad_id < 0)
        return arrow::Status::Invalid("tokenizer: required pad_id is missing");

    // (4) Round-trip on two samples (ASCII + Unicode "élève").
    ARROW_RETURN_NOT_OK(round_trip_sample(db, "The capital of France is"));
    ARROW_RETURN_NOT_OK(round_trip_sample(db, "\xC3\xA9l\xC3\xA8ve"));

    return arrow::Status::OK();
}

} // namespace

arrow::Status BuildAndVerifyTokenizer(
    const std::string &tokenizer_json_path,
    const std::filesystem::path &pinned_dir,
    const TokenizerSpecialIds &required_specials)
{
    // (1) Build binary Arrow files from tokenizer.json (mandatory).
    ARROW_RETURN_NOT_OK(BuildTokenizerArrowFiles(tokenizer_json_path, pinned_dir));
    return VerifyTokenizerArtifacts(pinned_dir, required_specials);
}

arrow::Status BuildAndVerifyTokenizerSentencePiece(
    const std::string &tokenizer_model_path,
    const std::filesystem::path &pinned_dir,
    const TokenizerSpecialIds &required_specials,
    const SpSpecialTokens &specials)
{
    // (1) Build binary Arrow files from tokenizer.model (SentencePiece BPE).
    ARROW_RETURN_NOT_OK(BuildTokenizerArrowFilesSentencePiece(
        tokenizer_model_path, pinned_dir, specials));
    return VerifyTokenizerArtifacts(pinned_dir, required_specials);
}

arrow::Status VerifyAllTensorReads(
    const std::filesystem::path &input_path,
    const SafetensorsModel &model)
{
    for (auto &shard : model.shards) {
        auto sf_path = input_path / shard.filename;
        std::ifstream sf(sf_path.string(), std::ios::binary);
        if (!sf)
            return arrow::Status::IOError("cannot open ", sf_path.string());

        uint64_t hdr_len = 0;
        sf.read(reinterpret_cast<char*>(&hdr_len), 8);
        if (!sf || sf.gcount() != 8)
            return arrow::Status::IOError("cannot read header length in ",
                                          sf_path.string());

        int64_t data_off = 8 + static_cast<int64_t>(hdr_len);
        for (auto &t : shard.tensors) {
            int64_t raw_start = data_off + t.data_begin;
            int64_t expected = t.byte_size();
            if (raw_start + expected > shard.file_size)
                return arrow::Status::Invalid(
                    shard.filename, " tensor ", t.name,
                    " extends beyond file size");

            std::vector<uint8_t> buf(static_cast<size_t>(expected));
            sf.seekg(raw_start, std::ios::beg);
            sf.read(reinterpret_cast<char*>(buf.data()),
                    static_cast<std::streamsize>(expected));
            if (!sf || static_cast<int64_t>(sf.gcount()) != expected)
                return arrow::Status::IOError(
                    "short read on ", shard.filename, " tensor ", t.name,
                    ": expected ", expected, " bytes, got ", sf.gcount());
        }
    }
    return arrow::Status::OK();
}

arrow::Status CheckTensorCoverage(
    const std::vector<std::string> &expected,
    const std::vector<std::string> &present,
    const std::string &what)
{
    absl::flat_hash_set<std::string> exp_set(expected.begin(), expected.end());
    absl::flat_hash_set<std::string> pre_set(present.begin(), present.end());

    std::vector<std::string> missing;
    for (auto &e : exp_set)
        if (pre_set.find(e) == pre_set.end())
            missing.push_back(e);

    if (!missing.empty()) {
        std::string msg = absl::StrCat("missing tensors [", what, "]:");
        for (size_t i = 0; i < missing.size(); ++i)
            absl::StrAppend(&msg, " ", missing[i]);
        return arrow::Status::Invalid(msg);
    }

    for (auto &p : pre_set)
        if (exp_set.find(p) == exp_set.end())
            lsxcommon::log::warn(
                absl::StrCat("[integrity] unexpected tensor: ", p));

    return arrow::Status::OK();
}

arrow::Status CheckLayerContiguity(
    int expected_layers,
    const absl::flat_hash_set<int> &present_layers)
{
    if (present_layers.empty())
        return arrow::Status::Invalid("no layer tensors found");

    int max_layer = -1;
    for (int l : present_layers)
        max_layer = std::max(max_layer, l);

    int actual = max_layer + 1;
    if (expected_layers != actual)
        return arrow::Status::Invalid(
            "layer count mismatch: config says ",
            expected_layers, " but tensors span 0..",
            max_layer, " (", actual, " layers)");

    for (int l = 0; l <= max_layer; ++l)
        if (present_layers.find(l) == present_layers.end())
            return arrow::Status::Invalid(
                "layer hole: layer ", l, " has no tensors");

    return arrow::Status::OK();
}

arrow::Status CheckGeometryFields(
    const std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> &fields)
{
    std::string msg;
    size_t count = 0;
    for (auto &[name, pair] : fields) {
        auto [expected, actual] = pair;
        if (expected != actual) {
            if (count > 0) absl::StrAppend(&msg, "; ");
            absl::StrAppend(&msg, name, ": config=", expected, " tensor=", actual);
            ++count;
        }
    }
    if (count > 0)
        return arrow::Status::Invalid("geometry mismatch: ", msg);
    return arrow::Status::OK();
}

} // namespace lsxcommon::integrity