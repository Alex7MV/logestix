#include "lsxcommon/model_integrity.h"
#include "lsxcommon/converter.h"
#include "lsxcommon/schema.h"
#include "lsxcommon/log.h"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <simdjson.h>
#include <absl/strings/str_cat.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
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

arrow::Status BuildAndVerifyTokenizer(
    const std::string &tokenizer_json_path,
    const std::filesystem::path &pinned_dir,
    const TokenizerSpecialIds &required_specials)
{
    namespace fs = std::filesystem;

    // (1) Build binary Arrow files from tokenizer.json (mandatory).
    ARROW_RETURN_NOT_OK(BuildTokenizerArrowFiles(tokenizer_json_path, pinned_dir));

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

arrow::Result<VerificationReport> VerifyConvertedModel(
    const std::filesystem::path &output_root)
{
    namespace fs = std::filesystem;
    VerificationReport r;
    std::string error;

    // Resolve index location (mandatory logestix/ subdirectory).
    fs::path index_path;
    fs::path pinned_dir;
    fs::path logestix_dir = output_root / "logestix";
    if (fs::exists(logestix_dir / "model.index.json")) {
        index_path = logestix_dir / "model.index.json";
        pinned_dir = logestix_dir / "pinned";
    } else {
        return arrow::Status::Invalid(
            "model.index.json not found in ", output_root.string());
    }

    if (!fs::exists(pinned_dir))
        return arrow::Status::Invalid("pinned dir not found: ",
                                      pinned_dir.string());

    std::ifstream f(index_path, std::ios::binary);
    if (!f)
        return arrow::Status::IOError("cannot open ", index_path.string());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    if (json.empty())
        return arrow::Status::Invalid("index file is empty");

    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    simdjson::ondemand::object root;
    if (simdjson::error_code serr = parser.iterate(pad).get(doc);
        serr != simdjson::SUCCESS)
        return arrow::Status::Invalid("cannot parse index JSON");
    if (auto rroot = doc.get_object(); rroot.error() != simdjson::SUCCESS)
        return arrow::Status::Invalid("index root not an object");
    else
        root = rroot.value();

    // Format tag must be one of the known layouts.
    std::string_view fmt;
    if (root["format"].get_string().get(fmt) != simdjson::SUCCESS)
        return arrow::Status::Invalid("index missing 'format' tag");
    static const std::string_view kKnownFmts[] = {
        "logestix-qwen", "logestix-glm", "logestix-kimi", "logestix-v1"};
    bool known = false;
    for (auto kf : kKnownFmts)
        if (fmt == kf) { known = true; break; }
    if (!known)
        return arrow::Status::Invalid("unknown format tag: ",
                                      std::string(fmt));

    int64_t expected_pinned = 0;
    if (auto p = root["total_pinned_tensors"].get_int64(); p.error() == simdjson::SUCCESS)
        expected_pinned = p.value();

    int64_t expected_experts = -1;
    if (auto e = root["total_experts"].get_int64(); e.error() == simdjson::SUCCESS)
        expected_experts = e.value();

    // Walk pinned dir, reopen + ValidateFull(). Tokenizer arrows and the
    // shared_experts.arrow must reopen (corruption is caught), but are excluded
    // from the model-tensor record count.
    int64_t actual_pinned = 0;
    size_t checked = 0;
    for (auto &entry : fs::recursive_directory_iterator(pinned_dir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".arrow") continue;

        auto batch_res = ReadRecordBatchFromFile(entry.path());
        if (!batch_res.ok())
            return arrow::Status::Invalid(
                entry.path().filename().string(), ": ",
                batch_res.status().ToString());

        auto st = (*batch_res)->ValidateFull();
        if (!st.ok())
            return arrow::Status::Invalid(
                entry.path().filename().string(),
                " ValidateFull: ", st.ToString());

        auto fname = entry.path().filename().string();
        if (fname.rfind("tokenizer_", 0) == 0)
            continue;
        if (fname == "shared_experts.arrow")
            continue;  // validated, but not counted in total_pinned_tensors

        actual_pinned += (*batch_res)->num_rows();
        ++checked;
    }

    if (expected_pinned != actual_pinned)
        return arrow::Status::Invalid(
            "pinned tensor count mismatch: index says ",
            expected_pinned, " but actual is ", actual_pinned,
            " across ", checked, " arrow files");

    // Expert arrows (GLM/DeepSeek layouts) must total total_experts.
    if (expected_experts > 0) {
        auto experts_dir = logestix_dir / "experts";
        if (!fs::exists(experts_dir))
            return arrow::Status::Invalid(
                "experts dir missing: ", experts_dir.string());
        int64_t actual_experts = 0;
        for (auto &entry : fs::recursive_directory_iterator(experts_dir))
            if (entry.is_regular_file() &&
                entry.path().extension() == ".arrow")
                ++actual_experts;
        if (actual_experts != expected_experts)
            return arrow::Status::Invalid(
                "expert count mismatch: index says ", expected_experts,
                " but actual is ", actual_experts);
    }

    r.pinned_tensor_count = actual_pinned;
    r.arrow_files_checked = checked;
    return r;
}

} // namespace lsxcommon::integrity