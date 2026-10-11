// Mistral-Small-4-119B loader test (Model Ingestion Protocol step 3). Converts
// a synthetic checkpoint, uploads it through the real loader, and checks the
// device-resident dense graph. No vault access (loader discipline).

#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/mistral_moe_convert.h"
#include "lsxcommon/mistral_moe_load.h"
#include "test_mistral_moe_fixture.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace {
bool GpuAvailable()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
} // namespace

TEST_CASE("Mistral loader uploads the dense graph from a converted fixture",
          "[mistral][load]")
{
    if (!GpuAvailable()) std::exit(4);

    auto in = fs::temp_directory_path() / "mistral_load_fixture";
    auto out = fs::temp_directory_path() / "mistral_load_out";
    fs::remove_all(in);
    fs::remove_all(out);
    fs::create_directories(in);

    mistraltest::MistralSynthGeom g;
    mistraltest::WriteMistralParams(in, g);
    synthtest::WriteSafetensors(in, "consolidated-00001-of-00001.safetensors",
                                mistraltest::BuildSyntheticMistralModel(g));
    REQUIRE(lsxcommon::mistral_moe::ConvertMistralModel(in.string(), out).ok());

    lsxcommon::mistral_moe::MistralPinnedWeights pw;
    std::string err;
    const cudaError_t e = lsxcommon::mistral_moe::LoadMistralWeights(
        out.string(), pw, nullptr, err);
    INFO("load error: " << err);
    REQUIRE(e == cudaSuccess);

    REQUIRE(pw.num_layers == static_cast<int>(g.layers));
    REQUIRE(pw.num_experts == static_cast<int>(g.experts));
    REQUIRE(pw.embed.rows == g.vocab);
    REQUIRE(pw.embed.cols == g.hidden);
    REQUIRE(pw.head.rows == g.vocab);
    REQUIRE(pw.final_norm != nullptr);

    const auto &l0 = pw.layers[0];
    REQUIRE(l0.wq_a.rows == g.q_lora);
    REQUIRE(l0.wq_a.cols == g.hidden);
    REQUIRE(l0.wq_b.rows == g.heads * (g.qk_nope + g.qk_rope));
    REQUIRE(l0.wq_b.cols == g.q_lora);
    REQUIRE(l0.wkv_a.rows == g.kv_lora + g.qk_rope);
    REQUIRE(l0.wkv_b.rows == g.heads * (g.qk_nope + g.v_head));
    REQUIRE(l0.wkv_b.cols == g.kv_lora);
    REQUIRE(l0.wo.rows == g.hidden);
    REQUIRE(l0.wo.cols == g.heads * g.v_head);
    REQUIRE(l0.q_a_norm != nullptr);
    REQUIRE(l0.kv_a_norm != nullptr);
    REQUIRE(l0.input_ln != nullptr);
    REQUIRE(l0.post_ln != nullptr);
    REQUIRE(l0.router != nullptr);
    REQUIRE(l0.shared_gate.d_packed != nullptr);
    REQUIRE(l0.shared_gate.d_scales != nullptr);
    REQUIRE(pw.resident_bytes > 0);

    pw.Free();
    REQUIRE(pw.layers.empty());
}
