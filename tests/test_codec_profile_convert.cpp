// The bf16 codec profile, in the converter (plan Task 15).
//
// The phi-4 and Qwen3.5-9B sources are already BF16, so a `bf16` profile is a
// PASS-THROUGH, not a quantizer: the weight planes keep the bytes they arrived
// with and declare `dtype = "BF16"` (rule D1). What makes it a profile rather
// than "the same conversion" is that the NVFP4 code path must not run at all —
// no nibble packing, no E4M3 scales, no residual plane — because an artifact that
// contains an nvfp4 plane cannot be gated onto Ampere.
//
// This is checked on a 2-layer fixture (cheap, no GPU weight traffic) and the
// result is verified by the artifact verifier, which recomputes the mask from the
// bytes rather than trusting the declaration.
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/codec_profile.h>
#include <lsxcommon/phi_dense_convert.h>
#include <lsxcommon/qwen_dense_convert.h>
#include <lsxcommon/schema.h>
#include <lsxstorage/artifact.h>

#include "test_phi_dense_fixture.h"
#include "test_qwen_dense_fixture.h"

#include <filesystem>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using lsxcommon::CodecProfile;

namespace {

constexpr uint64_t kF32 = 1ull << 0;
constexpr uint64_t kBF16 = 1ull << 1;
constexpr uint64_t kNvFp4 = 1ull << 5;

// Every Arrow file under `root`, relative and sorted: a test that lists what it
// found cannot pass vacuously when a directory is empty.
std::vector<std::string> ArrowFiles(const fs::path &root)
{
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto &e : fs::recursive_directory_iterator(root, ec)) {
        if (ec) break;
        if (!e.is_regular_file()) continue;
        const auto rel = fs::relative(e.path(), root).generic_string();
        if (rel.size() > 6 && rel.substr(rel.size() - 6) == ".arrow")
            out.push_back(rel);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// The dtypes a PinnedSchema file declares, deduplicated.
std::set<std::string> DtypesIn(const fs::path &p)
{
    std::set<std::string> out;
    auto batch = lsxcommon::ReadRecordBatchFromFile(p);
    if (!batch.ok()) return out;
    auto recs = lsxcommon::PinnedSchema::UnpackBatch(*batch);
    if (!recs.ok()) return out;
    for (const auto &r : *recs) out.insert(r.dtype);
    return out;
}

} // namespace

TEST_CASE("phi bf16 profile: no nvfp4 plane, bf16 declaration, verified mask",
          "[phi][convert][profile]") {
    const fs::path tmp = fs::temp_directory_path() / "phi_bf16_profile";
    const fs::path in = tmp / "in";
    const fs::path out = tmp / "out";
    fs::remove_all(tmp);
    fs::create_directories(in);
    auto tensors = phitest::BuildSyntheticPhiModel();
    phitest::WriteSafetensors(in, "model.safetensors", tensors);
    phitest::WriteConfig(in);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");

    auto st = lsxcommon::phi::ConvertPhiModel(in.string(), out.string(),
                                              CodecProfile::kBf16);
    INFO(st.ToString());
    REQUIRE(st.ok());

    // No quantized plane anywhere: the nvfp4 code path did not run.
    const auto files = ArrowFiles(out);
    REQUIRE_FALSE(files.empty());
    for (const auto &rel : files) {
        INFO("file: " << rel);
        CHECK(rel.find("nvfp4") == std::string::npos);
    }

    // The layer planes exist and declare BF16 — the pass-through, spelled so the
    // verifier can read it (rule D1).
    REQUIRE(fs::exists(out / "pinned" / "layer_0.arrow"));
    const auto dtypes = DtypesIn(out / "pinned" / "layer_0.arrow");
    INFO("dtypes: " << [&] {
        std::string s;
        for (const auto &d : dtypes) s += d + " ";
        return s;
    }());
    REQUIRE_FALSE(dtypes.empty());
    CHECK(dtypes.count("BF16") == 1);
    CHECK(dtypes.count("NVFP4") == 0);
    CHECK(dtypes.count("U8") == 0);   // the retired ambiguous spelling

    // The verifier recomputes the mask from the bytes.
    const auto scan = lsx::storage::ScanArtifactTree(out);
    std::string joined;
    for (const auto &p : scan.problems) joined += "\n  " + p;
    CAPTURE(joined);
    REQUIRE(scan.ok);
    CHECK((scan.mask & kNvFp4) == 0);
    CHECK((scan.mask & (kBF16 | kF32)) != 0);

    std::error_code ec;
    fs::remove_all(tmp, ec);
}

TEST_CASE("phi default profile still quantizes: the bf16 case is not the default",
          "[phi][convert][profile]") {
    // The complement that makes the case above meaningful: if `kDefault` had
    // silently become bf16, every existing nvfp4 artifact would have been built
    // by a different code path than the one the gates measured.
    const fs::path tmp = fs::temp_directory_path() / "phi_default_profile";
    const fs::path in = tmp / "in";
    const fs::path out = tmp / "out";
    fs::remove_all(tmp);
    fs::create_directories(in);
    auto tensors = phitest::BuildSyntheticPhiModel();
    phitest::WriteSafetensors(in, "model.safetensors", tensors);
    phitest::WriteConfig(in);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");

    REQUIRE(lsxcommon::phi::ConvertPhiModel(in.string(), out.string()).ok());
    CHECK(fs::exists(out / "pinned" / "layer_0_nvfp4.arrow"));
    const auto scan = lsx::storage::ScanArtifactTree(out);
    CHECK((scan.mask & kNvFp4) != 0);

    std::error_code ec;
    fs::remove_all(tmp, ec);
}

TEST_CASE("qwen bf16 profile: no nvfp4 plane, bf16 declaration, verified mask",
          "[qwen][convert][profile]") {
    const fs::path tmp = fs::temp_directory_path() / "qwen_bf16_profile";
    const fs::path in = tmp / "in";
    const fs::path out = tmp / "out";
    fs::remove_all(tmp);
    fs::create_directories(in);
    auto tensors = qwentest::BuildSyntheticQwenModel();
    qwentest::WriteSafetensors(in, "model.safetensors", tensors);
    qwentest::WriteConfig(in);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");

    auto st = lsxcommon::qwen::ConvertQwenModel(in.string(), out.string(),
                                                CodecProfile::kBf16);
    INFO(st.ToString());
    REQUIRE(st.ok());

    const auto files = ArrowFiles(out);
    REQUIRE_FALSE(files.empty());
    for (const auto &rel : files) {
        INFO("file: " << rel);
        CHECK(rel.find("nvfp4") == std::string::npos);
    }

    const auto scan = lsx::storage::ScanArtifactTree(out);
    std::string joined;
    for (const auto &p : scan.problems) joined += "\n  " + p;
    CAPTURE(joined);
    REQUIRE(scan.ok);
    CHECK((scan.mask & kNvFp4) == 0);
    CHECK((scan.mask & kBF16) != 0);

    std::error_code ec;
    fs::remove_all(tmp, ec);
}

TEST_CASE("the profile names are one list, not two", "[profile]") {
    // Forge parses the CLI string; the converters branch on the enum. A second
    // enum (or a second spelling) is how "--codec-profile bf16" ends up building
    // an nvfp4 tree: the flag is read in one place and the conversion in another.
    CHECK(std::string(lsxcommon::CodecProfileName(CodecProfile::kBf16)) == "bf16");
    CHECK(std::string(lsxcommon::CodecProfileName(CodecProfile::kNvFp4)) == "nvfp4");
    CHECK(std::string(lsxcommon::CodecProfileName(CodecProfile::kInt8W8A16)) ==
          "int8-w8a16");
    // kDefault IS nvfp4 for a converter: a flagless run must produce the layout
    // every existing artifact was measured with.
    CHECK(lsxcommon::ProfileUsesNvfp4Weights(CodecProfile::kDefault));
    CHECK(lsxcommon::ProfileUsesNvfp4Weights(CodecProfile::kNvFp4));
    CHECK_FALSE(lsxcommon::ProfileUsesNvfp4Weights(CodecProfile::kBf16));
    CHECK_FALSE(lsxcommon::ProfileUsesNvfp4Weights(CodecProfile::kInt8W8A16));
}
