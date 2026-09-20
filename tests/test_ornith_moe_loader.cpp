#include <catch2/catch_test_macros.hpp>

#include "test_ornith_moe_fixture.h"
#include "lsxcommon/ornith_moe_load.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <string>

using namespace lsxcommon::ornith;
namespace fs = std::filesystem;

namespace {

struct ConvertedFixture {
    fs::path in, out;
    ConvertedFixture()
    {
        in = fs::temp_directory_path() / "lsx_ornith_loader_in";
        out = fs::temp_directory_path() / "lsx_ornith_loader_out";
        fs::remove_all(in);
        fs::remove_all(out);
        ornithtest::BuildSyntheticOrnithCheckpoint(in);
        auto st = ConvertOrnithMoeModel(in.string(), out.string());
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

TEST_CASE("ornith dense loader roundtrip on synthetic MoE model",
          "[ornith-moe-loader]")
{
    ConvertedFixture fix;
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    OrnithMoePinnedWeights pw;
    std::string err;
    REQUIRE(LoadOrnithMoeDenseWeights(fix.out.string(), pw, st, err));

    REQUIRE(pw.layers.size() == 4);
    for (auto &lw : pw.layers) {
        CHECK(lw.attn.loaded);
        CHECK(lw.moe_loaded);
        CHECK(lw.d_router != nullptr);
        CHECK(lw.router_bytes == 8 * 64 * 2);   // E*H BF16
        CHECK(lw.shared_w1.rows == 16);
        CHECK(lw.shared_w1.cols == 64);
        CHECK(lw.shared_w2.rows == 64);
        CHECK(lw.d_shared_gate != nullptr);
        // Conv normalization: MLX [C,K,1] -> HF [C,1,K] -> kernel 4.
        if (lw.attn.is_linear) CHECK(lw.attn.conv_kernel == 4);
    }
    for (int l = 0; l < 4; ++l)
        CHECK(pw.layers[static_cast<size_t>(l)].attn.is_linear == (l != 3));
    CHECK(pw.top.num_layers == 4);
    CHECK(pw.top.hidden_size == 64);
    CHECK(pw.moe_top.num_experts == 8);
    CHECK(pw.embed.d_packed != nullptr);
    CHECK(pw.head.d_packed != nullptr);

    FreeOrnithMoePinnedWeights(pw);
    cudaStreamDestroy(st);
}
