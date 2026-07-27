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

    e.w1_packed.resize(1024);
    e.w2_packed.resize(1024);
    e.w3_packed.resize(512);
    for (int i = 0; i < 1024; ++i) {
        e.w1_packed[i] = static_cast<uint8_t>(i);
        e.w2_packed[i] = static_cast<uint8_t>(255 - i);
    }
    for (int i = 0; i < 512; ++i)
        e.w3_packed[i] = (i % 2 == 0) ? uint8_t(0xAA) : uint8_t(0xBB);

    e.w1_scales = {0.5f, 1.0f, 1.5f, 2.0f};
    e.w2_scales = {0.25f, 0.75f};
    e.w3_scales = {1.0f};

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

        for (int64_t i = 0; i < 4; ++i)
            CHECK(loaded.w1_scale[i] == expert.w1_scales[static_cast<size_t>(i)]);
        for (int64_t i = 0; i < 2; ++i)
            CHECK(loaded.w2_scale[i] == expert.w2_scales[static_cast<size_t>(i)]);
        for (int64_t i = 0; i < 1; ++i)
            CHECK(loaded.w3_scale[i] == expert.w3_scales[static_cast<size_t>(i)]);
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
    }
    fs::remove(path);
}
