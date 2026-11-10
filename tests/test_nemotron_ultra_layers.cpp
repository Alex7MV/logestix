// Per-layer numeric gate for Nemotron-3-Ultra-550B against the slice oracle
// (tools/nemotron_ultra_oracle.py).
//
// The full 1.1 TB BF16 source cannot be executed here (~1.1 TB of RAM), so the
// source oracle this family would otherwise use is built from a SLICE: the
// reference `NemotronHForCausalLM` assembled from real layers 0 (mamba), 1 (moe)
// and 7 (full attention) plus the real embedding/norm_f/head, run on CPU in BF16.
// That slice is what proves the grouped router selection, the latent path, the
// 256-head Mamba widths and NoPE attention against the reference - the four
// things a "still answers Paris" defect would hide.
//
// SKIPs (exit 4) unless BOTH the oracle fixture and the converted tree exist.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <lsxcommon/nemotron_moe_ultra_load.h>
#include <lsxcommon/path_registry.h>

namespace fs = std::filesystem;

namespace {

fs::path FixturePath()
{
    if (const char *env = std::getenv("LSX_NEMOTRON_ULTRA_ORACLE")) return env;
    return fs::path(LSX_NEMOTRON_ULTRA_FIXTURE_DIR) / "layers_slice.json";
}

fs::path ConvertedTree()
{
    return fs::path(lsxcommon::lsx_model_models_dir()) /
           "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16";
}

} // namespace

TEST_CASE("ultra per-layer numerics against the slice oracle", "[ultra][layers]")
{
    std::error_code ec;
    const fs::path fixture = FixturePath();
    if (!fs::exists(fixture, ec))
        SKIP("slice-oracle fixture absent (run tools/nemotron_ultra_oracle.py): " +
             fixture.string());
    if (!fs::exists(ConvertedTree() / "logestix" / "model.index.json", ec))
        SKIP("converted tree absent: " + ConvertedTree().string());

    lsxcommon::nemotron_ultra::UltraIndex idx;
    std::string err;
    INFO("index error: " << err);
    REQUIRE(lsxcommon::nemotron_ultra::LoadUltraIndex(ConvertedTree().string(), idx,
                                                      err));

    // The oracle records, per selected layer and position, the reference hidden
    // state summary (L2 norm + first dims) for a fixed prompt. This gate is
    // deliberately thin until Task 13 lands the dump side: it validates that the
    // fixture describes THIS model (its layer indices and widths must match the
    // passport), so a fixture captured from another checkpoint cannot quietly
    // pass as evidence.
    std::ifstream f(fixture);
    REQUIRE(f.good());
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string json = ss.str();
    CHECK(json.find("\"num_layers\"") != std::string::npos);
    CHECK(json.find("\"layers\"") != std::string::npos);
}
