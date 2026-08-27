#include <catch2/catch_test_macros.hpp>

#include "kimi_moe_test_fixture.h"
#include "lsxcommon/kimi_moe_convert.h"
#include "lsxcommon/kimi_moe_load.h"
#include "lsxcommon/model_factory.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <string>

using namespace lsxcommon;

namespace {

std::string BuildConvertedFixture()
{
    namespace fs = std::filesystem;
    auto in = fs::temp_directory_path() / "kimi_model_in";
    auto out = fs::temp_directory_path() / "kimi_model_out";
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);

    kimitest::WriteKimiConfig(in);
    kimitest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                               kimitest::BuildSyntheticKimiModel(3));
    kimitest::WriteKimiTokenizerFiles(in);
    REQUIRE(lsxcommon::kimi::ConvertKimiModel(in.string(), out).ok());
    return out.string();
}

} // namespace

TEST_CASE("Kimi model type is registered in the factory", "[kimi][model]")
{
    EnsureModelRegistry();
    ModelInitConfig empty_cfg;
    empty_cfg.model_type = "kimi-k2.7-code";
    auto model = LsxModelFactory::Create(empty_cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "kimi-k2.7-code");
}

TEST_CASE("Kimi model lifecycle on synthetic fixture", "[kimi][model]")
{
    EnsureModelRegistry();
    ModelInitConfig empty_cfg;
    empty_cfg.model_type = "kimi-k2.7-code";
    auto model = LsxModelFactory::Create(empty_cfg);
    REQUIRE(model != nullptr);

    // Fixture layout (not a real vault path): Load() must fail cleanly
    // because the pinned weights do not match the locked geometry.
    ModelInitConfig cfg;
    cfg.model_path = BuildConvertedFixture();
    cfg.model_type = "kimi-k2.7-code";
    auto m = LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);

    // Geometry validation rejects the 3-layer/64-vocab fixture.
    CHECK_FALSE(m->Load());
    CHECK(m->ResidentWeightBytes() == 0);

    // Unload is safe even after failed Load.
    m->Unload();
}
