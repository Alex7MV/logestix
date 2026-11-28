// Shared converter write helpers. The per-family copies that used to carry
// these bodies are gone, so their contract is pinned here instead.
#include <catch2/catch_test_macros.hpp>

#include <arrow/api.h>
#include <lsxcommon/converter.h>
#include <lsxcommon/schema.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path FreshDir(const char *tag)
{
    const fs::path dir =
        fs::temp_directory_path() / (std::string("logestix_conv_helpers_") + tag);
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

lsxcommon::DenseNvfp4Record DenseRecord()
{
    lsxcommon::DenseNvfp4Record r;
    r.name = "model.layers.0.self_attn.q_proj.weight";
    r.shape = {2, 16};
    r.dtype = "BF16";
    r.packed.assign(16, 0xA5);
    r.scales = {0x7E, 0x7F};
    r.global_scale = 4.0f;
    return r;
}

lsxcommon::PackedExpert SampleExpert()
{
    lsxcommon::PackedExpert pe;
    pe.w1_shape = pe.w2_shape = pe.w3_shape = {2, 16};
    pe.w1_packed.assign(16, 0x11);
    pe.w2_packed.assign(16, 0x22);
    pe.w3_packed.assign(16, 0x33);
    pe.w1_scales = {0x7E, 0x7F};
    pe.w2_scales = {0x7E, 0x7F};
    pe.w3_scales = {0x7E, 0x7F};
    pe.w1_global_scale = 1.5f;
    pe.w2_global_scale = 2.5f;
    pe.w3_global_scale = 3.5f;
    return pe;
}

} // namespace

TEST_CASE("ExpertArrowPath builds the canonical routed-expert path",
          "[converter][helpers]")
{
    const fs::path root = "/tmp/vault/model";
    CHECK(lsxcommon::ExpertArrowPath(root, 7, 3) ==
          root / "experts" / "layer_7" / "exp_3.arrow");
    CHECK(lsxcommon::ExpertArrowPath(root, 0, 0) ==
          root / "experts" / "layer_0" / "exp_0.arrow");
}

TEST_CASE("WriteBatch propagates a failed batch result", "[converter][helpers]")
{
    const fs::path dir = FreshDir("writebatch_error");
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> bad(
        arrow::Status::Invalid("synthetic batch failure"));

    const auto status = lsxcommon::WriteBatch(bad, dir / "out.arrow");
    REQUIRE_FALSE(status.ok());
    CHECK(status.ToString().find("synthetic batch failure") !=
          std::string::npos);
    CHECK_FALSE(fs::exists(dir / "out.arrow"));
}

TEST_CASE("WriteBatch round-trips a dense NVFP4 batch", "[converter][helpers]")
{
    const fs::path dir = FreshDir("writebatch_ok");
    const auto record = DenseRecord();
    auto batch = lsxcommon::DenseNvfp4Schema::MakeBatch({record});
    REQUIRE(batch.ok());

    const fs::path path = dir / "dense.arrow";
    REQUIRE(lsxcommon::WriteBatch(batch, path).ok());

    auto back = lsxcommon::ReadRecordBatchFromFile(path, 6);
    REQUIRE(back.ok());
    auto records = lsxcommon::DenseNvfp4Schema::UnpackBatch(*back);
    REQUIRE(records.ok());
    REQUIRE(records->size() == 1);
    CHECK((*records)[0].name == record.name);
    CHECK((*records)[0].shape == record.shape);
    CHECK((*records)[0].dtype == record.dtype);
    CHECK((*records)[0].packed == record.packed);
    CHECK((*records)[0].scales == record.scales);
    CHECK((*records)[0].global_scale == record.global_scale);
}

TEST_CASE("WriteExpertArrow creates the layer directory and round-trips",
          "[converter][helpers]")
{
    const fs::path root = FreshDir("expertfile_ok");
    const auto expert = SampleExpert();
    const fs::path path = lsxcommon::ExpertArrowPath(root, 7, 3);

    REQUIRE(lsxcommon::WriteExpertArrow(path, expert).ok());
    REQUIRE(fs::exists(path));

    auto batch = lsxcommon::ReadRecordBatchFromFile(
        path, lsxcommon::ExpertSchema::kNumFields);
    REQUIRE(batch.ok());
    auto back = lsxcommon::ExpertSchema::UnpackBatch(*batch, 16);
    REQUIRE(back.ok());
    CHECK(back->w1_packed == expert.w1_packed);
    CHECK(back->w2_packed == expert.w2_packed);
    CHECK(back->w3_packed == expert.w3_packed);
    CHECK(back->w1_scales == expert.w1_scales);
    CHECK(back->w2_scales == expert.w2_scales);
    CHECK(back->w3_scales == expert.w3_scales);
    CHECK(back->w1_global_scale == expert.w1_global_scale);
    CHECK(back->w2_global_scale == expert.w2_global_scale);
    CHECK(back->w3_global_scale == expert.w3_global_scale);
}

TEST_CASE("WriteExpertArrow reports an unwritable path", "[converter][helpers]")
{
    const fs::path root = FreshDir("expertfile_fail");
    const fs::path blocker = root / "blocked";
    { std::ofstream(blocker) << "not a directory"; }

    const auto expert = SampleExpert();
    const auto status =
        lsxcommon::WriteExpertArrow(blocker / "layer_0" / "exp_0.arrow", expert);
    CHECK_FALSE(status.ok());
}
