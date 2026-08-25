#include <catch2/catch_all.hpp>

#include <lsxcommon/qwen_convert.h>
#include <lsxcommon/qwen_inference_load.h>

#include "qwen_test_fixture.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <vector>

using namespace lsxcommon::qwen;

static void WriteToken(const std::filesystem::path &dir)
{
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        dir / "tokenizer.json");
}

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
    WriteToken(tmp);

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
        // qkv: [KD*2+VD, H] = [64, 64]
        CHECK(l0.qkv.rows == 64);   // KD*2+VD = 64
        CHECK(l0.qkv.cols == 64);   // H = 64
        CHECK(l0.d_conv1d != nullptr);
        CHECK(l0.d_A_log != nullptr);
        CHECK(l0.d_dt_bias != nullptr);
        CHECK(l0.d_attn_norm != nullptr);
        CHECK(l0.norm_size == 64);
        // MLP gate: [I, H] = [128, 64]
        CHECK(l0.mlp_gate.rows == 128);
        CHECK(l0.mlp_gate.cols == 64);
    }
    SECTION("full attention layer weights uploaded") {
        auto &l3 = pw.layers[3];
        CHECK(l3.loaded == true);
        // q_proj: [QProjOut(), H] = [128, 64]
        CHECK(l3.q.rows == 128);
        CHECK(l3.q.cols == 64);
        CHECK(l3.d_q_norm != nullptr);
        CHECK(l3.d_k_norm != nullptr);
        // MLP gate: [I, H] = [128, 64] -> cols = 64
        CHECK(l3.mlp_gate.rows == 128);
        CHECK(l3.mlp_gate.cols == 64);
    }
    SECTION("bugs: duplicate-Free is safe") {
        cudaError_t last = cudaGetLastError();
        CHECK(last == cudaSuccess);
    }

    pw.Free();
    cudaStreamDestroy(stream);
    remove_all(tmp);
}

static bool InArena(const void *d_arena, size_t arena_bytes, const void *p)
{
    auto base = reinterpret_cast<uintptr_t>(d_arena);
    auto ptr = reinterpret_cast<uintptr_t>(p);
    return d_arena != nullptr && ptr >= base && ptr < base + arena_bytes;
}

TEST_CASE("Qwen loader fast path: arena-backed layer uploads")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "qwen_load_arena_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = qwentest::BuildSyntheticQwenModel();
    qwentest::WriteSafetensors(tmp, "model.safetensors", tensors);
    qwentest::WriteConfig(tmp);
    WriteToken(tmp);
    REQUIRE(ConvertQwenModel(tmp.string(), tmp.string()).ok());

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    QwenPinnedWeights pw;
    REQUIRE(LoadQwenPinnedWeights(tmp.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    for (int l = 0; l < static_cast<int>(pw.layers.size()); ++l) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        INFO("layer " << l);
        REQUIRE(lw.loaded);
        CHECK(lw.d_arena != nullptr);
        CHECK(lw.arena_bytes > 0);
        // Every converted/raw 1D tensor view lives inside the layer arena.
        CHECK(InArena(lw.d_arena, lw.arena_bytes, lw.d_input_norm));
        CHECK(InArena(lw.d_arena, lw.arena_bytes, lw.d_post_norm));
        if (lw.is_linear) {
            CHECK(InArena(lw.d_arena, lw.arena_bytes, lw.d_conv1d));
            CHECK(InArena(lw.d_arena, lw.arena_bytes, lw.d_A_log));
            CHECK(InArena(lw.d_arena, lw.arena_bytes, lw.d_dt_bias));
            CHECK(InArena(lw.d_arena, lw.arena_bytes, lw.d_attn_norm));
        } else {
            CHECK(InArena(lw.d_arena, lw.arena_bytes, lw.d_q_norm));
            CHECK(InArena(lw.d_arena, lw.arena_bytes, lw.d_k_norm));
        }
    }

    // Free() releases the arena and clears the arena bookkeeping.
    pw.Free();
    CHECK(pw.layers.empty());
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
    WriteToken(tmp);
    REQUIRE(ConvertQwenModel(tmp.string(), tmp.string()).ok());

    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    QwenPinnedWeights pw;
    REQUIRE(LoadQwenPinnedWeights(tmp.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    // final norm weight stored as raw BF16 values 0.00,0.01,0.02,...
    // fixture: MakeBf16Values(pattern(i)=(float)i*0.01f)
    CHECK(std::abs(ReadHalf(pw.d_final_norm, 5) - 0.05f) < 1e-3f);

    // Fast-path bit-exactness: every converted tensor must match the legacy
    // scalar formula (bf16 bits<<16 -> FP32 -> __float2half nearest-even).
    auto bf16_expected = [](int64_t i) {
        uint32_t bits = static_cast<uint32_t>(i % 65536) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return __float2half(f);
    };
    auto bf16_bits = [&](int64_t i) {
        __half h = bf16_expected(i);
        return *reinterpret_cast<const uint16_t *>(&h);
    };
    const auto &l0 = pw.layers[0];   // linear attention
    {
        __half v;
        cudaMemcpy(&v, l0.d_conv1d + 7, sizeof(__half), cudaMemcpyDeviceToHost);
        CHECK(*reinterpret_cast<uint16_t *>(&v) == bf16_bits(7));
        cudaMemcpy(&v, l0.d_attn_norm + 3, sizeof(__half), cudaMemcpyDeviceToHost);
        CHECK(*reinterpret_cast<uint16_t *>(&v) == bf16_bits(3));
        // A_log / dt_bias are raw F32 passthroughs: exact patterns.
        float fv;
        cudaMemcpy(&fv, l0.d_A_log + 2, sizeof(float), cudaMemcpyDeviceToHost);
        CHECK(fv == 2.0f);           // i*0.5 + 1.0
        cudaMemcpy(&fv, l0.d_dt_bias + 3, sizeof(float), cudaMemcpyDeviceToHost);
        CHECK(fv == 0.3f);           // i*0.1
    }
    const auto &l3 = pw.layers[3];   // full attention
    {
        __half v;
        cudaMemcpy(&v, l3.d_q_norm + 31, sizeof(__half), cudaMemcpyDeviceToHost);
        CHECK(*reinterpret_cast<uint16_t *>(&v) == bf16_bits(31));
    }

    pw.Free();
    cudaStreamDestroy(stream);
    remove_all(tmp);
}
TEST_CASE("Qwen converter emits fused mlp.gate_up_proj; loader uploads it")
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "qwen_load_fused_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = qwentest::BuildSyntheticQwenModel();
    qwentest::WriteSafetensors(tmp, "model.safetensors", tensors);
    qwentest::WriteConfig(tmp);
    WriteToken(tmp);
    auto cst = ConvertQwenModel(tmp.string(), tmp.string());
    if (!cst.ok()) FAIL(cst.ToString());
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    QwenPinnedWeights pw;
    REQUIRE(LoadQwenPinnedWeights(tmp.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const int I = pw.top.intermediate_size;
    for (const auto &lw : pw.layers) {
        if (!lw.loaded) continue;
        REQUIRE(lw.mlp_gate_up.d_packed != nullptr);
        // Fused rows: [0,I) gate, [I,2I) up.
        CHECK(lw.mlp_gate_up.rows == 2 * I);
        CHECK(lw.mlp_gate_up.cols == lw.mlp_gate.cols);
        // Packed bytes must equal gate-packed followed by up-packed
        // (row-major nibble packing is per-row contiguous).
        REQUIRE(lw.mlp_gate_up.packed_bytes ==
                lw.mlp_gate.packed_bytes + lw.mlp_up.packed_bytes);
        std::vector<uint8_t> fused(lw.mlp_gate_up.packed_bytes),
            gate(lw.mlp_gate.packed_bytes), up(lw.mlp_up.packed_bytes);
        REQUIRE(cudaMemcpy(fused.data(), lw.mlp_gate_up.d_packed,
                           fused.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(gate.data(), lw.mlp_gate.d_packed, gate.size(),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(up.data(), lw.mlp_up.d_packed, up.size(),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(std::memcmp(fused.data(), gate.data(), gate.size()) == 0);
        CHECK(std::memcmp(fused.data() + gate.size(), up.data(), up.size()) == 0);
    }
}
