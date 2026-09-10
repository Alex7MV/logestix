#include <catch2/catch_test_macros.hpp>

#include "test_qwen_moe_fixture.h"
#include "lsxcommon/qwen_moe_load.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace lsxcommon::qwen_moe;
namespace fs = std::filesystem;

namespace {

struct ConvertedFixture {
    fs::path in;
    fs::path out;

    ConvertedFixture()
    {
        in = fs::temp_directory_path() / "lsx_qwen_moe_loader_in";
        out = fs::temp_directory_path() / "lsx_qwen_moe_loader_out";
        fs::remove_all(in);
        fs::remove_all(out);
        qwen_moe_test::BuildSyntheticMoeCheckpoint(in);
        auto st = ConvertQwen35MoeModel(in.string(), out.string());
        if (!st.ok())
            FAIL("fixture conversion failed: " + st.ToString());
    }
    ~ConvertedFixture()
    {
        std::error_code ec;
        fs::remove_all(in, ec);
        fs::remove_all(out, ec);
    }
};

} // namespace

TEST_CASE("dense loader roundtrip on synthetic MoE model", "[qwen-moe-loader]")
{
    ConvertedFixture fix;
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    QwenMoePinnedWeights pw;
    std::string err;
    REQUIRE(LoadQwenMoeDenseWeights(fix.out.string(), pw, st, err));

    REQUIRE(pw.layers.size() == 4);
    for (auto &lw : pw.layers) {
        CHECK(lw.attn.loaded);
        CHECK(lw.moe_loaded);
        CHECK(lw.d_router != nullptr);
        CHECK(lw.router_bytes == 8 * 64 * 2);   // E*H BF16
        CHECK(lw.shared_w1.rows == 16);          // SI
        CHECK(lw.shared_w1.cols == 64);          // H
        CHECK(lw.shared_w2.rows == 64);          // [H, SI]
        CHECK(lw.d_shared_gate != nullptr);
    }
    // Full attention only at layer 3 (interval-4 pattern).
    for (int l = 0; l < 4; ++l)
        CHECK(pw.layers[static_cast<size_t>(l)].attn.is_linear == (l != 3));
    CHECK(pw.top.num_layers == 4);
    CHECK(pw.top.hidden_size == 64);
    CHECK(pw.moe_top.num_experts == 8);
    CHECK(pw.embed.d_packed != nullptr);
    CHECK(pw.head.d_packed != nullptr);
    CHECK(pw.resident_bytes > 0);

    FreeQwenMoePinnedWeights(pw);
    cudaStreamDestroy(st);
}

TEST_CASE("loader strictness on missing layer file", "[qwen-moe-loader]")
{
    ConvertedFixture fix;

    // Break one layer file: strict error naming the failure, no partial load.
    fs::path victim = fix.out / "logestix" / "pinned" / "layer_2.arrow";
    REQUIRE(fs::exists(victim));
    fs::remove(victim);

    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    QwenMoePinnedWeights pw;
    std::string err;
    CHECK_FALSE(LoadQwenMoeDenseWeights(fix.out.string(), pw, st, err));
    CHECK(err.find("layer_2") != std::string::npos);
    FreeQwenMoePinnedWeights(pw);
    cudaStreamDestroy(st);
}
