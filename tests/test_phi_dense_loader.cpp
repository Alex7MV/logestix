#include <catch2/catch_all.hpp>

#include <lsxcommon/phi_dense_convert.h>
#include <lsxcommon/phi_dense_load.h>

#include "test_phi_dense_fixture.h"

#include <cuda_runtime.h>
#include <filesystem>

using namespace lsxcommon::phi;

namespace {

std::filesystem::path BuildConvertedFixture()
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "phi_loader_test";
    remove_all(tmp);
    create_directories(tmp);
    phitest::WriteSafetensors(tmp, "model.safetensors",
                              phitest::BuildSyntheticPhiModel());
    phitest::WriteConfig(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");
    auto st = ConvertPhiModel(tmp.string(), tmp.string());
    REQUIRE(st.ok());
    return tmp;
}

} // namespace

TEST_CASE("Phi topology parses logestix-phi index", "[phi][loader]")
{
    auto tmp = BuildConvertedFixture();
    auto top = load_phi_topology(
        (tmp / "logestix" / "model.index.json").string());
    CHECK(top.error_msg.empty());
    CHECK(top.num_layers == 2);
    CHECK(top.hidden_size == 64);
    CHECK(top.num_attention_heads == 4);
    CHECK(top.num_key_value_heads == 2);
    CHECK(top.head_dim == 16);
    CHECK(top.attention_scale > 0.0);
    std::filesystem::remove_all(tmp);
}

TEST_CASE("Phi pinned weights upload to device", "[phi][loader]")
{
    auto tmp = BuildConvertedFixture();
    const auto &c = phitest::TestCfg();

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    PhiPinnedWeights pw;
    REQUIRE(LoadPhiPinnedWeights(tmp.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    CHECK(pw.layers.size() == static_cast<size_t>(c.num_layers));
    CHECK(pw.resident_bytes > 0);
    CHECK(pw.d_final_norm != nullptr);
    CHECK(pw.final_norm_size == c.hidden_size);
    CHECK(pw.embed.d_packed != nullptr);
    CHECK(pw.embed.rows == c.vocab_size);
    CHECK(pw.embed.cols == c.hidden_size);
    CHECK(pw.head.d_packed != nullptr);
    CHECK(pw.head.rows == c.vocab_size);

    for (int l = 0; l < c.num_layers; ++l) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        INFO("layer " << l);
        CHECK(lw.loaded);
        CHECK(lw.d_input_norm != nullptr);
        CHECK(lw.d_post_norm != nullptr);
        CHECK(lw.norm_size == c.hidden_size);
        CHECK(lw.qkv.d_packed != nullptr);
        CHECK(lw.q.d_packed != nullptr);
        CHECK(lw.k.d_packed != nullptr);
        CHECK(lw.v.d_packed != nullptr);
        CHECK(lw.o.d_packed != nullptr);
        CHECK(lw.mlp_down.d_packed != nullptr);
        // Fused gate+up with derived non-owning gate/up slices.
        CHECK(lw.mlp_gate_up.d_packed != nullptr);
        CHECK(lw.mlp_gate.d_packed != nullptr);
        CHECK(lw.mlp_up.d_packed != nullptr);
        CHECK(lw.mlp_gate.rows == c.intermediate_size);
        CHECK(lw.mlp_up.rows == c.intermediate_size);
    }

    pw.Free();
    CHECK(pw.layers.empty());
    cudaStreamDestroy(stream);
    std::filesystem::remove_all(tmp);
}
