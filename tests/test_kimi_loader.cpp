#include <catch2/catch_test_macros.hpp>

#include "kimi_test_fixture.h"
#include "lsxcommon/kimi_convert.h"
#include "lsxcommon/kimi_inference_load.h"

#include <absl/strings/str_cat.h>

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <string>

using namespace lsxcommon;
using lsxcommon::kimi::ensure_kimi_expert_resident;
using lsxcommon::kimi::free_kimi_expert_index;
using lsxcommon::kimi::load_kimi_expert_index;

namespace {

// Converts the tiny synthetic model once and returns the output root.
std::string BuildConvertedFixture()
{
    namespace fs = std::filesystem;
    auto in = fs::temp_directory_path() / "kimi_loader_in";
    auto out = fs::temp_directory_path() / "kimi_loader_out";
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

TEST_CASE("LoadKimiPinnedWeights uploads dense and MoE layers",
          "[kimi][loader]")
{
    auto model = BuildConvertedFixture();

    lsxcommon::kimi::KimiPinnedWeights pw;
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    cudaError_t err = LoadKimiPinnedWeights(model, pw, stream);
    REQUIRE(err == cudaSuccess);
    if (err == cudaSuccess) {
        REQUIRE(pw.layers.size() == 3);
        // Dense layer 0: MLP weights present, router absent.
        REQUIRE(pw.layers[0].d_mlp_gate != nullptr);
        REQUIRE(pw.layers[0].d_mlp_down != nullptr);
        REQUIRE(pw.layers[0].d_mlp_up != nullptr);
        REQUIRE(pw.layers[0].d_router_gate == nullptr);
        // MoE layers 1-2: router + gate bias (fixture bias has 16 entries),
        // no dense MLP.
        for (int l = 1; l <= 2; ++l) {
            INFO("layer " << l);
            REQUIRE(pw.layers[l].d_router_gate != nullptr);
            REQUIRE(pw.layers[l].h_gate_bias != nullptr);
            REQUIRE(pw.layers[l].gate_bias_n == 16);
            REQUIRE(pw.layers[l].d_mlp_gate == nullptr);
        }
        // Globals uploaded.
        REQUIRE(pw.d_embed != nullptr);
        REQUIRE(pw.d_head != nullptr);
        REQUIRE(pw.d_final_norm != nullptr);
        REQUIRE(pw.vocab_size == 64);    // fixture embed rows
        REQUIRE(pw.hidden == 16);
    }
    pw.Free();
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    cudaStreamDestroy(stream);
}

TEST_CASE("LoadKimiPinnedWeights errors on missing layer file",
          "[kimi][loader]")
{
    auto model = BuildConvertedFixture();
    auto missing =
        std::filesystem::path(model) / "logestix" / "pinned" / "layer_2.arrow";
    std::filesystem::rename(missing, missing.string() + ".bak");

    lsxcommon::kimi::KimiPinnedWeights pw;
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    cudaError_t err = LoadKimiPinnedWeights(model, pw, stream);
    REQUIRE(err == cudaErrorInvalidValue);
    REQUIRE(pw.layers.empty());   // Free() ran on failure

    std::filesystem::rename(missing.string() + ".bak", missing);
    cudaStreamDestroy(stream);
}

TEST_CASE("Kimi expert index resolves routed experts", "[kimi][loader]")
{
    auto model = BuildConvertedFixture();
    auto db = load_kimi_expert_index(model, /*num_layers=*/3,
                                     /*experts_per_layer=*/2);
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    ensure_kimi_expert_resident(db, /*layer=*/1, /*expert=*/0, stream);
    const auto &e = db.entries[1 * 2 + 0];
    REQUIRE(e.host_w1 != nullptr);
    REQUIRE(e.host_w2 != nullptr);
    REQUIRE(e.host_w3 != nullptr);
    REQUIRE(e.host_w1_scales != nullptr);
    // Fixture gate proj is [4,32]: packed nibbles 4*16=64 bytes.
    CHECK(e.w1_bytes == 64);
    free_kimi_expert_index(db);
    cudaStreamDestroy(stream);
}
