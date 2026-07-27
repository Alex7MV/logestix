#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/schema.h>
#include <lsxcommon/loader.h>
#include <arrow/api.h>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

static lsxcommon::PackedExpert MakeTestExpert()
{
    lsxcommon::PackedExpert e;

    // w1: 1024 bytes of [0..255] repeated
    e.w1_packed.resize(1024);
    for (int i = 0; i < 1024; ++i)
        e.w1_packed[i] = static_cast<uint8_t>(i);

    // w2: 1024 bytes of [255..0] repeated
    e.w2_packed.resize(1024);
    for (int i = 0; i < 1024; ++i)
        e.w2_packed[i] = static_cast<uint8_t>(255 - i);

    // w3: 512 bytes of alternating 0xAA/0xBB
    e.w3_packed.resize(512);
    for (int i = 0; i < 512; ++i)
        e.w3_packed[i] = (i % 2 == 0) ? uint8_t(0xAA) : uint8_t(0xBB);

    // E4M3-encoded scales (power-of-2): e4 = ceil(log2(s)) + 7, byte = (e4 << 3)
    // 0.5  → ceil(log2(0.5)) = -1 → e4=6 → 0x30
    // 1.0  → ceil(log2(1.0)) =  0 → e4=7 → 0x38
    // 2.0  → ceil(log2(2.0)) =  1 → e4=8 → 0x40
    e.w1_scales = {0x30, 0x38, 0x40, 0x40};
    e.w2_scales = {0x28, 0x38};
    e.w3_scales = {0x38};
    e.global_scale = 1.0f;

    return e;
}

static fs::path TempExpertPath()
{
    return fs::temp_directory_path() / "logestix_test_expert.arrow";
}

TEST_CASE("Loader round-trip with synthetic PackedExpert", "[loader]")
{
    auto expert = MakeTestExpert();

    auto batch_result = lsxcommon::ExpertSchema::MakeBatch(expert);
    REQUIRE(batch_result.ok());
    auto batch = *batch_result;
    REQUIRE(batch->ValidateFull().ok());

    auto path = TempExpertPath();
    auto st = lsxcommon::WriteRecordBatchToFile(batch, path);
    REQUIRE(st.ok());

    lsxcommon::Loader loader;
    {
        auto load_result = loader.LoadExpertFromArrow(path.string());
        REQUIRE(load_result.ok());
        auto loaded = *load_result;

        CHECK(loaded.w1_weight_size == 1024);
        CHECK(loaded.w2_weight_size == 1024);
        CHECK(loaded.w3_weight_size == 512);
        CHECK(loaded.w1_scale_count == 4);
        CHECK(loaded.w2_scale_count == 2);
        CHECK(loaded.w3_scale_count == 1);

        CHECK(memcmp(loaded.w1_weight, expert.w1_packed.data(), 1024) == 0);
        CHECK(memcmp(loaded.w2_weight, expert.w2_packed.data(), 1024) == 0);
        CHECK(memcmp(loaded.w3_weight, expert.w3_packed.data(), 512) == 0);

        CHECK(memcmp(loaded.w1_scale, expert.w1_scales.data(), 4) == 0);
        CHECK(memcmp(loaded.w2_scale, expert.w2_scales.data(), 2) == 0);
        CHECK(memcmp(loaded.w3_scale, expert.w3_scales.data(), 1) == 0);

        CHECK(loaded.global_scale == expert.global_scale);
    }
    fs::remove(path);
}

TEST_CASE("Load expert with empty w1 fields", "[loader][edge]")
{
    auto expert = MakeTestExpert();
    expert.w1_packed.clear();
    expert.w1_scales.clear();

    auto batch_result = lsxcommon::ExpertSchema::MakeBatch(expert);
    REQUIRE(batch_result.ok());
    auto batch = *batch_result;
    REQUIRE(batch->ValidateFull().ok());

    auto path = TempExpertPath();
    REQUIRE(lsxcommon::WriteRecordBatchToFile(batch, path).ok());

    lsxcommon::Loader loader;
    {
        auto load_result = loader.LoadExpertFromArrow(path.string());
        REQUIRE(load_result.ok());
        auto loaded = *load_result;

        CHECK(loaded.w1_weight_size == 0);
        CHECK(loaded.w1_scale_count == 0);

        CHECK(loaded.w2_weight_size == 1024);
        CHECK(loaded.w3_weight_size == 512);
        CHECK(loaded.w2_scale_count == 2);
        CHECK(loaded.w3_scale_count == 1);

        CHECK(memcmp(loaded.w2_weight, expert.w2_packed.data(), 1024) == 0);
        CHECK(memcmp(loaded.w3_weight, expert.w3_packed.data(), 512) == 0);

        CHECK(loaded.global_scale == expert.global_scale);
    }
    fs::remove(path);
}
