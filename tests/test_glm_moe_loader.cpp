#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <lsxcommon/glm_moe_convert.h>
#include <lsxcommon/glm_moe_load.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <filesystem>

#include "glm_moe_test_fixture.h"

namespace fs = std::filesystem;

using lsxcommon::glm::GlmPinnedWeights;

namespace {

cudaError_t MakeStream(cudaStream_t &s)
{
    return cudaStreamCreate(&s);
}

void DestroyStream(cudaStream_t &s)
{
    if (s) cudaStreamDestroy(s);
    s = nullptr;
}

// Reads a value back from a device FP16 buffer (must be stream-synced first).
__half ReadHalf(const __half *d, cudaStream_t stream, int idx = 0)
{
    __half h;
    cudaMemcpyAsync(&h, d + idx, sizeof(__half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h;
}

float ReadFloat(const float *d, cudaStream_t stream, int idx = 0)
{
    float v;
    cudaMemcpyAsync(&v, d + idx, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return v;
}

} // namespace

TEST_CASE("GLM loader uploads global tensors BF16→FP16", "[glm][loader]")
{
    // Minimal model: global embed/head/norm with verifiable values.
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_glm_loader_global";
    fs::remove_all(base);
    auto in = base / "model";
    auto out = base / "converted";
    fs::create_directories(in);

    std::vector<glmtest::FakeTensor> t;
    t.push_back({"model.embed_tokens.weight", "BF16", {4, 2},
        glmtest::MakeBf16Values(8, [](int64_t i) { return static_cast<float>(i) + 0.5f; })});
    t.push_back({"lm_head.weight", "BF16", {4, 2},
        glmtest::MakeBf16Values(8, [](int64_t i) { return -static_cast<float>(i) - 0.5f; })});
    t.push_back({"model.norm.weight", "BF16", {2},
        glmtest::MakeBf16Values(2, [](int64_t i) { return 1.0f + static_cast<float>(i) * 0.25f; })});
    // A 1-layer model keeps the topology num_layers >= 1 (as any real model).
    // Strict loading requires every dense layer to carry both norms + MLP.
    t.push_back({"model.layers.0.input_layernorm.weight", "BF16", {2},
        glmtest::MakeBf16Values(2, [](int64_t i) { return 0.25f + static_cast<float>(i); })});
    t.push_back({"model.layers.0.post_attention_layernorm.weight", "BF16", {2},
        glmtest::MakeBf16Values(2, [](int64_t i) { return 0.5f + static_cast<float>(i); })});
    t.push_back({"model.layers.0.mlp.gate_proj.weight", "BF16", {2, 2},
        glmtest::MakeBf16Values(4, [](int64_t i) { return static_cast<float>(i); })});
    t.push_back({"model.layers.0.mlp.down_proj.weight", "BF16", {2, 2},
        glmtest::MakeBf16Values(4, [](int64_t i) { return static_cast<float>(i); })});
    t.push_back({"model.layers.0.mlp.up_proj.weight", "BF16", {2, 2},
        glmtest::MakeBf16Values(4, [](int64_t i) { return static_cast<float>(i); })});
    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors", t);
    glmtest::WriteConfig(in, /*num_layers=*/1, /*hidden=*/2, /*vocab=*/4,
                         /*experts=*/0);
    glmtest::WriteTokenizer(in);

    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    REQUIRE(st.ok());

    cudaStream_t stream = nullptr;
    REQUIRE(MakeStream(stream) == cudaSuccess);

    GlmPinnedWeights pw;
    auto err = lsxcommon::glm::LoadGlmPinnedWeights(out.string(), pw, stream);
    REQUIRE(err == cudaSuccess);
    cudaStreamSynchronize(stream);

    CHECK(pw.vocab_size == 4);
    CHECK(pw.hidden == 2);
    REQUIRE(pw.d_embed != nullptr);
    REQUIRE(pw.d_head != nullptr);
    REQUIRE(pw.d_final_norm != nullptr);
    CHECK(pw.layers.size() == 1);   // a single layer (input_layernorm)

    for (int i = 0; i < 8; ++i) {
        float exp = static_cast<float>(i) + 0.5f;
        float got = __half2float(ReadHalf(pw.d_embed, stream, i));
        CHECK(got == Catch::Approx(exp).epsilon(1e-3f));
    }
    for (int i = 0; i < 8; ++i) {
        float exp = -static_cast<float>(i) - 0.5f;
        float got = __half2float(ReadHalf(pw.d_head, stream, i));
        CHECK(got == Catch::Approx(exp).epsilon(1e-3f));
    }
    CHECK(__half2float(ReadHalf(pw.d_final_norm, stream, 0)) ==
          Catch::Approx(1.0f).epsilon(1e-3f));
    CHECK(__half2float(ReadHalf(pw.d_final_norm, stream, 1)) ==
          Catch::Approx(1.25f).epsilon(1e-3f));

    pw.Free();
    DestroyStream(stream);
    fs::remove_all(base);
}

TEST_CASE("GLM loader sizes layers from topology", "[glm][loader]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_glm_loader_topo";
    fs::remove_all(base);
    auto in = base / "model";
    auto out = base / "converted";
    fs::create_directories(in);

    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(in);
    glmtest::WriteTokenizer(in);
    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    REQUIRE(st.ok());

    cudaStream_t stream = nullptr;
    REQUIRE(MakeStream(stream) == cudaSuccess);

    GlmPinnedWeights pw;
    auto err = lsxcommon::glm::LoadGlmPinnedWeights(out.string(), pw, stream);
    REQUIRE(err == cudaSuccess);
    cudaStreamSynchronize(stream);

    // Topology-derived sizing: all 79 layer files present and loaded.
    CHECK(pw.layers.size() == 79);
    CHECK(pw.vocab_size == 64);
    CHECK(pw.hidden == 16);

    // Dense layer 0: dense MLP present, no router, no indexer.
    auto &l0 = pw.layers[0];
    CHECK(l0.d_mlp_gate != nullptr);
    CHECK(l0.d_mlp_down != nullptr);
    CHECK(l0.d_mlp_up != nullptr);
    CHECK(l0.d_router_gate == nullptr);
    CHECK(l0.d_idx_wq_b == nullptr);
    CHECK(l0.d_q_a_proj != nullptr);
    CHECK(l0.d_q_b_proj != nullptr);
    CHECK(l0.d_o_proj != nullptr);
    CHECK(l0.d_input_norm != nullptr);
    CHECK(l0.d_post_attn_norm != nullptr);

    // Dense layer 2.
    CHECK(pw.layers[2].d_mlp_gate != nullptr);
    CHECK(pw.layers[2].d_router_gate == nullptr);

    // MoE layer 3: router gate + F32 bias, no dense MLP.
    auto &l3 = pw.layers[3];
    CHECK(l3.d_router_gate != nullptr);
    REQUIRE(l3.h_gate_bias != nullptr);
    CHECK(l3.gate_bias_n == 16);
    CHECK(l3.d_mlp_gate == nullptr);
    CHECK(l3.d_q_a_proj != nullptr);

    // Indexer layer 4 (l % 4 == 0): indexer weights + F32 k_norm bias.
    auto &l4 = pw.layers[4];
    CHECK(l4.d_idx_wq_b != nullptr);
    CHECK(l4.d_idx_wk != nullptr);
    CHECK(l4.d_idx_weights_proj != nullptr);
    CHECK(l4.d_idx_k_norm != nullptr);
    REQUIRE(l4.d_idx_k_norm_bias != nullptr);
    CHECK(l4.d_router_gate != nullptr);   // indexer layers are MoE too

    // Non-indexer MoE layer 5: no indexer weights.
    CHECK(pw.layers[5].d_idx_wq_b == nullptr);

    // MTP layer 78: MTP extras present.
    auto &l78 = pw.layers[78];
    CHECK(l78.d_eh_proj != nullptr);
    CHECK(l78.d_enorm != nullptr);
    CHECK(l78.d_hnorm != nullptr);
    CHECK(l78.d_shared_head_norm != nullptr);
    CHECK(l78.d_router_gate != nullptr);

    pw.Free();
    DestroyStream(stream);
    fs::remove_all(base);
}

TEST_CASE("GLM loader writes correct FP16 values per layer", "[glm][loader]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_glm_loader_values";
    fs::remove_all(base);
    auto in = base / "model";
    auto out = base / "converted";
    fs::create_directories(in);

    // Layer 0 dense MLP gate_proj with verifiable values (plus minimal
    // global tensors so global.arrow exists, as for any real model).
    std::vector<glmtest::FakeTensor> t;
    t.push_back({"model.embed_tokens.weight", "BF16", {16, 16},
        glmtest::MakeBf16Values(256, [](int64_t i) { return 0.125f * static_cast<float>(i); })});
    t.push_back({"lm_head.weight", "BF16", {16, 16},
        glmtest::MakeBf16(256)});
    t.push_back({"model.norm.weight", "BF16", {16},
        glmtest::MakeBf16Values(16, [](int64_t i) { return 1.0f + static_cast<float>(i) * 0.25f; })});
    t.push_back({"model.layers.0.input_layernorm.weight", "BF16", {16},
        glmtest::MakeBf16Values(16, [](int64_t i) { return 0.5f + static_cast<float>(i); })});
    t.push_back({"model.layers.0.post_attention_layernorm.weight", "BF16", {16},
        glmtest::MakeBf16Values(16, [](int64_t i) { return 1.0f + static_cast<float>(i); })});
    t.push_back({"model.layers.0.mlp.gate_proj.weight", "BF16", {16, 16},
        glmtest::MakeBf16Values(256, [](int64_t i) { return static_cast<float>(i) * 0.125f; })});
    t.push_back({"model.layers.0.mlp.up_proj.weight", "BF16", {16, 16},
        glmtest::MakeBf16(256)});
    t.push_back({"model.layers.0.mlp.down_proj.weight", "BF16", {16, 16},
        glmtest::MakeBf16(256)});
    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors", t);
    glmtest::WriteConfig(in, /*num_layers=*/1, /*hidden=*/16, /*vocab=*/16,
                         /*experts=*/0);
    glmtest::WriteTokenizer(in);

    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    REQUIRE(st.ok());

    cudaStream_t stream = nullptr;
    REQUIRE(MakeStream(stream) == cudaSuccess);

    GlmPinnedWeights pw;
    auto err = lsxcommon::glm::LoadGlmPinnedWeights(out.string(), pw, stream);
    REQUIRE(err == cudaSuccess);
    cudaStreamSynchronize(stream);

    REQUIRE(pw.layers.size() == 1);
    auto &l0 = pw.layers[0];
    REQUIRE(l0.d_mlp_gate != nullptr);
    CHECK(__half2float(ReadHalf(l0.d_mlp_gate, stream, 0)) ==
          Catch::Approx(0.0f).epsilon(1e-3f));
    CHECK(__half2float(ReadHalf(l0.d_mlp_gate, stream, 1)) ==
          Catch::Approx(0.125f).epsilon(1e-3f));
    CHECK(__half2float(ReadHalf(l0.d_mlp_gate, stream, 255)) ==
          Catch::Approx(31.875f).epsilon(1e-3f));
    CHECK(__half2float(ReadHalf(l0.d_input_norm, stream, 15)) ==
          Catch::Approx(15.5f).epsilon(1e-3f));

    pw.Free();
    DestroyStream(stream);
    fs::remove_all(base);
}

TEST_CASE("GLM loader errors on missing layer file", "[glm][loader]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_glm_loader_missing";
    fs::remove_all(base);
    auto in = base / "model";
    auto out = base / "converted";
    fs::create_directories(in);

    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(in);
    glmtest::WriteTokenizer(in);
    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    REQUIRE(st.ok());

    auto victim = out / "logestix" / "pinned" / "layer_5.arrow";
    if (!fs::exists(victim))
        victim = out / "pinned" / "layer_5.arrow";
    REQUIRE(fs::exists(victim));
    fs::remove(victim);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    GlmPinnedWeights pw;
    auto err = lsxcommon::glm::LoadGlmPinnedWeights(out.string(), pw, stream);
    CHECK(err == cudaErrorInvalidValue);
    CHECK(pw.layers.empty());

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("GLM loader errors on malformed tensor shape", "[glm][loader]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_glm_loader_badshape";
    fs::remove_all(base);
    auto in = base / "model";
    auto out = base / "converted";
    fs::create_directories(in);

    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(in);
    glmtest::WriteTokenizer(in);
    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    REQUIRE(st.ok());

    std::vector<lsxcommon::TensorRecord> bad;
    bad.push_back({"input_layernorm.weight", "BF16", {}, {}});
    bad.push_back({"post_attention_layernorm.weight", "BF16", {16},
                   std::vector<uint8_t>(16 * 2)});
    auto batch_res = lsxcommon::PinnedSchema::MakeBatch(bad);
    REQUIRE(batch_res.ok());
    auto pinned_bad = out / "logestix" / "pinned" / "layer_5.arrow";
    if (!fs::exists(out / "logestix" / "pinned"))
        pinned_bad = out / "pinned" / "layer_5.arrow";
    auto wres = lsxcommon::WriteRecordBatchToFile(*batch_res, pinned_bad);
    REQUIRE(wres.ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    GlmPinnedWeights pw;
    auto err = lsxcommon::glm::LoadGlmPinnedWeights(out.string(), pw, stream);
    CHECK(err == cudaErrorInvalidValue);

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}
