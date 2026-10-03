// Xing4.0-29B-A4B loader stage test (Model Ingestion Protocol step 3).
// Converts the synthetic fixture and loads it through the real loader, then
// checks the residency contract that the inference stage depends on:
//   - the two mHC blocks survive per layer with their exact record shapes;
//   - MLA projections are NVFP4 (packed + scales + global_scale), norms BF16;
//   - the router weight is NVFP4 and the noaux_tc bias stays FP32;
//   - routed experts are all resident with a per-(layer, expert) offset table
//     (no expert cache: 64x38 experts fit in VRAM at NVFP4);
//   - resident_bytes counts what was actually allocated.
//
// The vault is never touched here (loader stage invariant): the fixture comes
// from tests/test_xing_moe_fixture.h through the real converter.

#include <catch2/catch_test_macros.hpp>

#include "test_xing_moe_fixture.h"

#include "lsxcommon/xing_moe_convert.h"
#include "lsxcommon/xing_moe_load.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <string>

using namespace lsxcommon;
using lsxcommon::xing::XingPinnedWeights;

namespace {

std::string BuildConvertedFixture()
{
    namespace fs = std::filesystem;
    auto in = fs::temp_directory_path() / "xing_loader_in";
    auto out = fs::temp_directory_path() / "xing_loader_out";
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    fs::create_directories(in);

    xingtest::WriteXingConfig(in);
    xingtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                               xingtest::BuildSyntheticXingModel(3, 2));
    REQUIRE(xing::ConvertXingModel(in.string(), out).ok());
    return out.string();
}

} // namespace

TEST_CASE("LoadXingWeights uploads mHC, MLA, router and all-resident experts",
          "[xing][loader]")
{
    const auto model = BuildConvertedFixture();

    XingPinnedWeights pw;
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    std::string error;
    const cudaError_t err = xing::LoadXingWeights(model, pw, stream, error);
    INFO("loader error: " << error);
    REQUIRE(err == cudaSuccess);
    if (err != cudaSuccess) return;

    REQUIRE(pw.num_layers == 3);
    REQUIRE(pw.num_experts == 2);
    REQUIRE(pw.first_moe_layer == 2);   // fixture mirrors first_k_dense=2
    REQUIRE(pw.embed.valid());
    REQUIRE(pw.head.valid());
    REQUIRE(pw.final_norm != nullptr);

    // Every layer: both mHC blocks, both block norms, the MLA inner norms.
    for (int l = 0; l < pw.num_layers; ++l)
    {
        const auto &ly = pw.layers[static_cast<size_t>(l)];
        INFO("layer " << l);
        REQUIRE(ly.attn_hc.valid());
        REQUIRE(ly.ffn_hc.valid());
        REQUIRE(ly.attn_hc.hc_fn_rows == 24);
        REQUIRE(ly.attn_hc.hc_fn_cols == 4 * 16);   // hc_mult * hidden (fixture)
        REQUIRE(ly.input_ln != nullptr);
        REQUIRE(ly.post_attn_ln != nullptr);
        REQUIRE(ly.q_a_ln != nullptr);
        REQUIRE(ly.kv_a_ln != nullptr);
        REQUIRE(ly.q_a.valid());
        REQUIRE(ly.q_b.valid());
        REQUIRE(ly.kv_a.valid());
        REQUIRE(ly.kv_b.valid());
        REQUIRE(ly.o_proj.valid());
        REQUIRE((ly.q_a.rows == 16));
        REQUIRE((ly.q_a.cols == 16));
        REQUIRE((ly.o_proj.rows == 16));
        REQUIRE((ly.o_proj.cols == 16));
    }

    // Dense prefix carries the SwiGLU trio and no router.
    for (int l = 0; l < pw.first_moe_layer; ++l)
    {
        const auto &ly = pw.layers[static_cast<size_t>(l)];
        REQUIRE_FALSE(ly.is_moe);
        REQUIRE(ly.dense_gate.valid());
        REQUIRE(ly.dense_up.valid());
        REQUIRE(ly.dense_down.valid());
        REQUIRE_FALSE(ly.router.valid());
        REQUIRE(ly.e_score_bias == nullptr);
    }

    // MoE layer: router (NVFP4) + FP32 bias + shared expert, no dense trio.
    {
        const auto &ly = pw.layers[static_cast<size_t>(pw.first_moe_layer)];
        REQUIRE(ly.is_moe);
        REQUIRE(ly.router.valid());
        REQUIRE(ly.router.rows == 2);      // fixture experts
        REQUIRE(ly.router.cols == 16);
        REQUIRE(ly.e_score_bias != nullptr);
        REQUIRE(ly.shared_gate.valid());
        REQUIRE(ly.shared_up.valid());
        REQUIRE(ly.shared_down.valid());
        REQUIRE_FALSE(ly.dense_gate.valid());
    }

    // Routed experts: all resident, indexed by LAYER (dense prefix keeps null
    // slots so callers never remap), offsets inside their layer slab, scales
    // after the packed nibbles.
    REQUIRE(pw.expert_slabs.size() == 3);
    REQUIRE(pw.expert_slabs[0] == nullptr);
    REQUIRE(pw.expert_slabs[1] == nullptr);
    REQUIRE(pw.expert_slabs[2] != nullptr);
    REQUIRE(pw.expert_records.size() == 3);
    REQUIRE(pw.expert_records[2].size() == 2);
    for (size_t e = 0; e < pw.expert_records[2].size(); ++e)
    {
        const auto &r = pw.expert_records[2][e];
        INFO("expert " << e);
        REQUIRE(r.w1_bytes == 128);   // BF16 [16,16] -> 128 packed bytes
        REQUIRE(r.s1_bytes == 16);
        REQUIRE(r.w1_off >= 0);
        REQUIRE(r.s1_off >= r.w1_off + r.w1_bytes);
        REQUIRE(static_cast<size_t>(r.s3_off + r.s3_bytes) <= pw.slab_bytes[2]);
        REQUIRE(r.gs1 > 0.0f);
    }

    // Resident accounting: nonzero and at least the expert slab.
    REQUIRE(pw.resident_bytes > 0);
    REQUIRE(pw.resident_bytes >= pw.slab_bytes[2]);

    cudaStreamDestroy(stream);
    pw.Free();
}

TEST_CASE("LoadXingWeights is strict about a missing layer", "[xing][loader]")
{
    const auto model = BuildConvertedFixture();
    std::error_code ec;
    std::filesystem::remove(std::filesystem::path(model) / "logestix" / "pinned" /
                                "layer_1.arrow",
                            ec);

    XingPinnedWeights pw;
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    std::string error;
    const cudaError_t err = xing::LoadXingWeights(model, pw, stream, error);
    REQUIRE(err != cudaSuccess);
    REQUIRE_FALSE(error.empty());
    cudaStreamDestroy(stream);
}