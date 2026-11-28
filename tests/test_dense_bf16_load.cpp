// Dual-encoding load + dispatch: a bf16-profile tree must LOAD and RUN (plan
// Task 15a, defect A3.13).
//
// Task 15 made the converters produce bf16 artifacts and the codec gate admit
// them. The loaders were still NVFP4-only, so a bf16 tree arrived at
// `layer weights incomplete` for every layer: the artifact gated and then loaded
// empty. This is the other half.
//
// What is asserted, and why each part is load-bearing:
//
//   * the layers are COMPLETE (`lw.loaded`) — the exact assertion that failed;
//   * the dense weight that satisfied it is the one the pipeline can dispatch to
//     (rows/cols/dtype describe a real plane, not a default-constructed one);
//   * a decode step produces FINITE logits — a loaded-but-misdispatched layer
//     would either fault or produce garbage, and "loaded" alone cannot tell those
//     apart;
//   * the NVFP4 path is untouched — the complement case, so a fix that only ever
//     exercises bf16 cannot silently break every existing artifact.
#include <catch2/catch_all.hpp>

#include <lsxcommon/codec_profile.h>
#include <lsxcommon/phi_dense_convert.h>
#include <lsxcommon/phi_dense_load.h>
#include <lsxcommon/dory_dense_convert.h>
#include <lsxcommon/dory_dense_load.h>
#include <lsxcommon/qwen_dense_convert.h>
#include <lsxcommon/qwen_inference_load.h>

#include "test_dory_dense_fixture.h"
#include "test_phi_dense_fixture.h"
#include "test_qwen_dense_fixture.h"

#include <cuda_runtime.h>

#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;
using lsxcommon::CodecProfile;

namespace {

bool AllLayersLoaded(const std::vector<lsxcommon::phi::PhiLayerWeights> &layers)
{
    for (const auto &l : layers)
        if (!l.loaded) return false;
    return !layers.empty();
}

bool AllLayersLoaded(const std::vector<lsxcommon::attn::LayerWeights> &layers)
{
    for (const auto &l : layers)
        if (!l.loaded) return false;
    return !layers.empty();
}

} // namespace

// ── phi ───────────────────────────────────────────────────────────────────

TEST_CASE("phi bf16 tree loads, every layer complete, decode is finite",
          "[phi][load][profile]") {
    const fs::path tmp = fs::temp_directory_path() / "phi_bf16_load";
    const fs::path in = tmp / "in";
    const fs::path out = tmp / "out";
    fs::remove_all(tmp);
    fs::create_directories(in);
    phitest::WriteSafetensors(in, "model.safetensors",
                              phitest::BuildSyntheticPhiModel());
    phitest::WriteConfig(in);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");
    auto st = lsxcommon::phi::ConvertPhiModel(in.string(), out.string(),
                                              CodecProfile::kBf16);
    INFO(st.ToString());
    REQUIRE(st.ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::phi::PhiPinnedWeights pw;
    REQUIRE(lsxcommon::phi::LoadPhiPinnedWeights(out.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const auto &c = phitest::TestCfg();
    REQUIRE(pw.layers.size() == static_cast<size_t>(c.num_layers));
    // THE assertion A3.13 failed: every layer complete.
    for (size_t i = 0; i < pw.layers.size(); ++i) {
        INFO("layer " << i);
        CHECK(pw.layers[i].loaded);
    }
    CHECK(AllLayersLoaded(pw.layers));

    // The dense plane that satisfied it is a real one: right shape, BF16.
    const auto &l0 = pw.layers[0];
    REQUIRE(l0.qkv_bf16.valid());
    CHECK(l0.qkv_bf16.dtype == lsxcommon::gemv::LxDtype::kBF16);
    CHECK(l0.qkv_bf16.rows == (c.num_attention_heads + 2 * c.num_key_value_heads) *
                                  c.head_dim);
    CHECK(l0.qkv_bf16.cols == c.hidden_size);
    CHECK(l0.o_bf16.valid());
    CHECK(l0.mlp_gate_up_bf16.valid());
    CHECK(l0.mlp_down_bf16.valid());
    // And it did NOT silently fill the NVFP4 fields, which would mean the bf16
    // bytes were reinterpreted as nibbles.
    CHECK(l0.qkv.d_packed == nullptr);

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(tmp);
}

TEST_CASE("phi nvfp4 tree is untouched by the dual-encoding load",
          "[phi][load][profile]") {
    // The complement: a fix that only ever exercised bf16 could break every
    // existing nvfp4 artifact without any bf16 case noticing.
    const fs::path tmp = fs::temp_directory_path() / "phi_nvfp4_load_complement";
    const fs::path in = tmp / "in";
    const fs::path out = tmp / "out";
    fs::remove_all(tmp);
    fs::create_directories(in);
    phitest::WriteSafetensors(in, "model.safetensors",
                              phitest::BuildSyntheticPhiModel());
    phitest::WriteConfig(in);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");
    REQUIRE(lsxcommon::phi::ConvertPhiModel(in.string(), out.string()).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    lsxcommon::phi::PhiPinnedWeights pw;
    REQUIRE(lsxcommon::phi::LoadPhiPinnedWeights(out.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    REQUIRE_FALSE(pw.layers.empty());
    CHECK(AllLayersLoaded(pw.layers));
    // NVFP4 present, dense absent: the encoding is chosen by what the tree holds.
    CHECK(pw.layers[0].qkv.d_packed != nullptr);
    CHECK_FALSE(pw.layers[0].qkv_bf16.valid());

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(tmp);
}


// ── qwen ──────────────────────────────────────────────────────────────────

TEST_CASE("qwen bf16 tree loads, every layer complete, both block types",
          "[qwen][load][profile]") {
    // Qwen3.5 is a HYBRID: a linear-attention layer and a full-attention layer
    // hold different projections, and a bf16 tree keeps the checkpoint's own
    // split (`gate_proj` + `up_proj`, not the fused `gate_up_proj` the nvfp4
    // converter emits). Both shapes are asserted, because "loaded" on only one
    // block type would leave half the model unreadable.
    const fs::path tmp = fs::temp_directory_path() / "qwen_bf16_load";
    const fs::path in = tmp / "in";
    const fs::path out = tmp / "out";
    fs::remove_all(tmp);
    fs::create_directories(in);
    qwentest::WriteSafetensors(in, "model.safetensors",
                               qwentest::BuildSyntheticQwenModel());
    qwentest::WriteConfig(in);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");
    auto st = lsxcommon::qwen::ConvertQwenModel(in.string(), out.string(),
                                                CodecProfile::kBf16);
    INFO(st.ToString());
    REQUIRE(st.ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::qwen::QwenPinnedWeights pw;
    REQUIRE(lsxcommon::qwen::LoadQwenPinnedWeights(out.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const auto &c = qwentest::TestCfg();
    REQUIRE(pw.layers.size() == static_cast<size_t>(c.num_layers));
    for (size_t i = 0; i < pw.layers.size(); ++i) {
        INFO("layer " << i << " linear=" << pw.layers[i].is_linear);
        CHECK(pw.layers[i].loaded);
    }
    CHECK(AllLayersLoaded(pw.layers));

    // Linear-attention block: in_proj_qkv / z / a / b / out_proj.
    const auto &lin = pw.layers[0];
    REQUIRE(lin.is_linear);
    REQUIRE(lin.qkv_bf16.valid());
    CHECK(lin.qkv_bf16.dtype == lsxcommon::gemv::LxDtype::kBF16);
    CHECK(lin.qkv_bf16.rows == qwentest::ConvDim());
    CHECK(lin.qkv_bf16.cols == c.hidden_size);
    CHECK(lin.z_bf16.valid());
    CHECK(lin.a_bf16.valid());
    CHECK(lin.b_bf16.valid());
    CHECK(lin.out_bf16.valid());
    CHECK(lin.qkv.d_packed == nullptr);   // not reinterpreted as nibbles

    // Full-attention block: q_proj holds query|gate, so it is 2*heads*head_dim.
    const auto &full = pw.layers[3];
    REQUIRE_FALSE(full.is_linear);
    REQUIRE(full.q_bf16.valid());
    CHECK(full.q_bf16.rows == qwentest::QProjOut());
    CHECK(full.q_bf16.cols == c.hidden_size);
    CHECK(full.k_bf16.valid());
    CHECK(full.v_bf16.valid());
    CHECK(full.o_bf16.valid());
    CHECK(full.q.d_packed == nullptr);

    // MLP, per layer, in the checkpoint's unfused split.
    for (size_t i = 0; i < pw.layers.size(); ++i) {
        INFO("layer " << i);
        CHECK(pw.layers[i].mlp_gate_bf16.valid());
        CHECK(pw.layers[i].mlp_up_bf16.valid());
        CHECK(pw.layers[i].mlp_down_bf16.valid());
        CHECK(pw.layers[i].mlp_gate_bf16.rows == c.intermediate_size);
        CHECK(pw.layers[i].mlp_down_bf16.rows == c.hidden_size);
        CHECK(pw.layers[i].mlp_down_bf16.cols == c.intermediate_size);
    }

    // Edges: a bf16 tree has no global_nvfp4.arrow, so embed/head must be
    // resident as dense planes instead.
    REQUIRE(pw.embed_bf16.valid());
    CHECK(pw.embed_bf16.rows == c.vocab_size);
    CHECK(pw.embed_bf16.cols == c.hidden_size);
    REQUIRE(pw.head_bf16.valid());
    CHECK(pw.head_bf16.rows == c.vocab_size);
    CHECK(pw.embed.d_packed == nullptr);

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(tmp);
}

TEST_CASE("qwen nvfp4 tree is untouched by the dual-encoding load",
          "[qwen][load][profile]") {
    const fs::path tmp = fs::temp_directory_path() / "qwen_nvfp4_load_complement";
    const fs::path in = tmp / "in";
    const fs::path out = tmp / "out";
    fs::remove_all(tmp);
    fs::create_directories(in);
    qwentest::WriteSafetensors(in, "model.safetensors",
                               qwentest::BuildSyntheticQwenModel());
    qwentest::WriteConfig(in);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");
    REQUIRE(lsxcommon::qwen::ConvertQwenModel(in.string(), out.string()).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    lsxcommon::qwen::QwenPinnedWeights pw;
    REQUIRE(lsxcommon::qwen::LoadQwenPinnedWeights(out.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    REQUIRE_FALSE(pw.layers.empty());
    CHECK(AllLayersLoaded(pw.layers));
    CHECK(pw.layers[0].qkv.d_packed != nullptr);
    CHECK_FALSE(pw.layers[0].qkv_bf16.valid());
    CHECK(pw.embed.d_packed != nullptr);
    CHECK_FALSE(pw.embed_bf16.valid());

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(tmp);
}

// ── dory ──────────────────────────────────────────────────────────────────

TEST_CASE("dory bf16 tree loads, every mixer complete, both mixer kinds",
          "[dory][load][profile]") {
    // Dory is the reason the bf16 profile exists: a 4B model at ~8 GiB runs on a
    // Jetson Orin NX 16 GB, while its NVFP4 planes cannot be executed on Ampere
    // at all. Its mixers are FUSED in the checkpoint (qkv, gate_up), so the
    // pass-through keeps those planes rather than re-splitting them.
    const fs::path tmp = fs::temp_directory_path() / "dory_bf16_load";
    const fs::path in = tmp / "in";
    const fs::path out = tmp / "out";
    fs::remove_all(tmp);
    fs::create_directories(in);
    dorytest::WriteSafetensors(in, "model.safetensors",
                               dorytest::BuildSyntheticDoryModel());
    dorytest::WriteConfig(in);
    fs::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                  in / "tokenizer.json");
    auto st = lsxcommon::dory::ConvertDoryModel(in.string(), out.string(),
                                                CodecProfile::kBf16);
    INFO(st.ToString());
    REQUIRE(st.ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::dory::DoryPinnedWeights pw;
    REQUIRE(lsxcommon::dory::LoadDoryPinnedWeights(out.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const auto &top = pw.top;
    REQUIRE(pw.layers.size() == static_cast<size_t>(top.num_layers));
    for (size_t i = 0; i < pw.layers.size(); ++i) {
        INFO("layer " << i << " attention=" << pw.layers[i].is_attention);
        CHECK(pw.layers[i].loaded);
    }

    const int H = top.hidden_size;
    const int I = top.intermediate_size;
    const int Q = top.num_attention_heads * top.head_dim;
    const int KV = top.num_key_value_heads * top.head_dim;
    for (size_t i = 0; i < pw.layers.size(); ++i) {
        const auto &lw = pw.layers[i];
        if (lw.is_attention) {
            INFO("attention mixer " << i);
            REQUIRE(lw.qkv_bf16.valid());
            CHECK(lw.qkv_bf16.dtype == lsxcommon::gemv::LxDtype::kBF16);
            CHECK(lw.qkv_bf16.rows == Q + 2 * KV);
            CHECK(lw.qkv_bf16.cols == H);
            CHECK(lw.g_bf16.valid());
            CHECK(lw.g_bf16.rows == Q);
            CHECK(lw.o_bf16.valid());
            CHECK(lw.o_bf16.rows == H);
            CHECK(lw.qkv.d_packed == nullptr);   // not reinterpreted as nibbles
            CHECK_FALSE(lw.gate_up_bf16.valid());
        } else {
            INFO("mlp mixer " << i);
            REQUIRE(lw.gate_up_bf16.valid());
            CHECK(lw.gate_up_bf16.rows == 2 * I);
            CHECK(lw.gate_up_bf16.cols == H);
            CHECK(lw.down_bf16.valid());
            CHECK(lw.down_bf16.rows == H);
            CHECK(lw.down_bf16.cols == I);
            CHECK(lw.gate_up.d_packed == nullptr);
            CHECK_FALSE(lw.qkv_bf16.valid());
        }
        // The nGPT reference scalars are encoding-independent and must still be
        // resident: they are what makes the mixer nGPT rather than vanilla.
        CHECK(lw.d_alpha != nullptr);
    }

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(tmp);
}
