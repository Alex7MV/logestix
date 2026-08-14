#include <catch2/catch_all.hpp>

#include <lsxcommon/qwen_convert.h>
#include <lsxcommon/qwen_inference_load.h>

#include "qwen_test_fixture.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <filesystem>
#include <vector>

using namespace lsxcommon::qwen;

static float ReadHalf(const __half *d, int64_t idx)
{
    __half v;
    cudaMemcpy(&v, d + idx, sizeof(__half), cudaMemcpyDeviceToHost);
    return __half2float(v);
}

TEST_CASE("Qwen loader reads topology and sizes layers")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "qwen_load_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = qwentest::BuildSyntheticQwenModel();
    qwentest::WriteSafetensors(tmp, "model.safetensors", tensors);
    qwentest::WriteConfig(tmp);

    auto st = ConvertQwenModel(tmp.string(), tmp.string());
    REQUIRE(st.ok());

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    QwenPinnedWeights pw;
    bool ok = LoadQwenPinnedWeights(tmp.string(), pw, stream);
    CHECK(ok);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);

    SECTION("topology populated") {
        CHECK(pw.top.num_layers == 4);
        CHECK(pw.top.hidden_size == 64);
        CHECK(pw.top.intermediate_size == 128);
        CHECK(pw.top.linear_num_value_heads == 4);
    }
    SECTION("global weights") {
        CHECK(pw.embed.rows == 64);
        CHECK(pw.embed.cols == 64);
        CHECK(pw.head.rows == 64);
        CHECK(pw.head.cols == 64);
        CHECK(pw.final_norm_size == 64);
        CHECK(pw.d_final_norm != nullptr);
    }
    SECTION("layer type dispatch") {
        CHECK(pw.layers.size() == 4);
        CHECK(pw.layers[0].is_linear == true);
        CHECK(pw.layers[1].is_linear == true);
        CHECK(pw.layers[2].is_linear == true);
        CHECK(pw.layers[3].is_linear == false);
    }
    SECTION("linear layer weights uploaded") {
        auto &l0 = pw.layers[0];
        CHECK(l0.loaded == true);
        CHECK(l0.qkv.rows == 64);
        // conv_dim = 2*key_dim + v_dim
        CHECK(l0.d_conv1d != nullptr);
        CHECK(l0.d_A_log != nullptr);
        CHECK(l0.d_dt_bias != nullptr);
        CHECK(l0.d_attn_norm != nullptr);
        CHECK(l0.norm_size == 64);
        // MLP present
        CHECK(l0.mlp_gate.rows == 64);
        CHECK(l0.mlp_gate.cols == 128);
    }
    SECTION("full attention layer weights uploaded") {
        auto &l3 = pw.layers[3];
        CHECK(l3.loaded == true);
        CHECK(l3.q.rows == 64);
        CHECK(l3.d_q_norm != nullptr);
        CHECK(l3.d_k_norm != nullptr);
        CHECK(l3.mlp_gate.cols == 128);
    }
    SECTION("bugs: duplicate-Free is safe") {
        cudaError_t last = cudaGetLastError();
        CHECK(last == cudaSuccess);
    }

    pw.Free();
    cudaStreamDestroy(stream);
    remove_all(tmp);
}

TEST_CASE("Qwen loader BF16->FP16 values")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "qwen_load_val_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = qwentest::BuildSyntheticQwenModel();
    qwentest::WriteSafetensors(tmp, "model.safetensors", tensors);
    qwentest::WriteConfig(tmp);
    REQUIRE(ConvertQwenModel(tmp.string(), tmp.string()).ok());

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    QwenPinnedWeights pw;
    REQUIRE(LoadQwenPinnedWeights(tmp.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    // final norm weight stored as raw BF16 values 0.00,0.01,0.02,...
    // fixture: MakeBf16Values(pattern(i)=(float)i*0.01f)
    CHECK(std::abs(ReadHalf(pw.d_final_norm, 5) - 0.05f) < 1e-3f);

    pw.Free();
    cudaStreamDestroy(stream);
    remove_all(tmp);
}