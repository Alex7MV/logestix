// TDD — MiMo-V2.6-Flash-RL converter synthetic E2E (Model Ingestion Protocol
// steps 1-2). Converts a synthetic checkpoint (no real weights) and checks the
// logestix-mimo-nvfp4 layout. Label `convert`, bounded TIMEOUT.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>

#include <lsxcommon/mimo_moe_convert.h>

#include "test_mimo_moe_fixture.h"

namespace fs = std::filesystem;

TEST_CASE("mimo converter emits a valid logestix-mimo-nvfp4 layout",
          "[mimo][convert]")
{
    const fs::path root = fs::temp_directory_path() / "lsx_mimo_convert";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path in = root / "src";
    const fs::path out = root / "dst";
    mimotest::BuildSyntheticMiMoCheckpoint(in);

    const auto st = lsxcommon::mimo::ConvertMiMoModel(in.string(), out.string());
    INFO(st.ToString());
    REQUIRE(st.ok());

    const std::string idx =
        mimotest::ReadFileText(out / "logestix" / "model.index.json");
    CHECK(idx.find("\"format\": \"logestix-mimo-nvfp4\"") != std::string::npos);
    CHECK(idx.find("\"num_layers\": 2") != std::string::npos);
    CHECK(idx.find("\"deferred\"") != std::string::npos);
    CHECK(idx.find("\"mtp\"") != std::string::npos);

    // All four routed experts of the MoE layer landed as NVFP4 ExpertSchema.
    for (int e = 0; e < 4; ++e)
        REQUIRE(mimotest::ExpertFileExists(out, 1, e));

    // Pinned layer files exist for both layers.
    CHECK(fs::exists(out / "logestix" / "pinned" / "layer_000.arrow"));
    CHECK(fs::exists(out / "logestix" / "pinned" / "layer_001.arrow"));
    // Global pinned file (embed / head / final norm).
    CHECK(fs::exists(out / "logestix" / "pinned" / "global.arrow"));

    // The multimodal tensor is stripped: its name appears nowhere in outputs.
    std::error_code ec;
    bool multimodal_seen = false;
    for (auto &e : fs::recursive_directory_iterator(out, ec))
    {
        if (!e.is_regular_file() || e.path().extension() != ".json") continue;
        if (mimotest::ReadFileText(e.path()).find("visual.blocks") !=
            std::string::npos)
            multimodal_seen = true;
    }
    CHECK_FALSE(multimodal_seen);

    fs::remove_all(root);
}
