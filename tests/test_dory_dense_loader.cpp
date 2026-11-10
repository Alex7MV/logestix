#include <catch2/catch_all.hpp>

#include <lsxcommon/dory_dense_convert.h>
#include <lsxcommon/dory_dense_load.h>

#include "test_dory_dense_fixture.h"

#include <cuda_runtime.h>
#include <filesystem>
#include <vector>

using namespace lsxcommon::dory;

namespace {

std::filesystem::path BuildConvertedFixture()
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "dory_loader_test";
    remove_all(tmp);
    create_directories(tmp);
    dorytest::WriteSafetensors(tmp, "model.safetensors",
                               dorytest::BuildSyntheticDoryModel());
    dorytest::WriteConfig(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");
    auto st = ConvertDoryModel(tmp.string(), tmp.string());
    REQUIRE(st.ok());
    return tmp;
}

// Reads a small F32 vector back off the device for the nGPT-scale checks.
std::vector<float> DownloadF32(const float *d, size_t n)
{
    std::vector<float> out(n);
    if (d == nullptr) return {};
    cudaMemcpy(out.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

} // namespace

TEST_CASE("Dory topology parses the logestix-dory-nvfp4 index",
          "[dory][loader]")
{
    auto tmp = BuildConvertedFixture();
    const auto top =
        load_dory_topology((tmp / "logestix" / "model.index.json").string());
    INFO(top.error_msg);
    CHECK(top.error_msg.empty());
    CHECK(top.num_layers == dorytest::TestCfg().num_layers);
    CHECK(top.recurrent_loops == dorytest::TestCfg().recurrent_loops);
    CHECK(top.sliding_window == dorytest::TestCfg().sliding_window);
    CHECK(top.num_attention_layers == dorytest::TestCfg().num_layers / 2);
    std::filesystem::remove_all(tmp);
}

TEST_CASE("Dory pinned weights upload to device", "[dory][loader]")
{
    auto tmp = BuildConvertedFixture();
    const auto &c = dorytest::TestCfg();

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    DoryPinnedWeights pw;
    REQUIRE(LoadDoryPinnedWeights(tmp.string(), pw, stream));
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    CHECK(pw.layers.size() == static_cast<size_t>(c.num_layers));
    CHECK(pw.resident_bytes > 0);
    CHECK(pw.kv_slots == dorytest::CountKvSlots());
    CHECK(pw.slot_of_layer.size() == static_cast<size_t>(c.num_layers));

    // Edges: FP16 tables + the F32 per-vocabulary logit scale, pre-divided by
    // initializer_range.
    CHECK(pw.d_embed_f16 != nullptr);
    CHECK(pw.embed_rows == c.vocab_size);
    CHECK(pw.embed_cols == c.hidden_size);
    CHECK(pw.d_head_f16 != nullptr);
    CHECK(pw.head_rows == c.vocab_size);
    CHECK(pw.head_cols == c.hidden_size);
    REQUIRE(pw.d_logit_scale != nullptr);
    CHECK(pw.logit_scale_size == c.vocab_size);
    {
        const auto scale =
            DownloadF32(pw.d_logit_scale, static_cast<size_t>(c.vocab_size));
        REQUIRE(scale.size() == static_cast<size_t>(c.vocab_size));
        // The fixture writes logit_scale in [0.015, 0.03); after /std it must
        // land in [0.75, 1.5).
        CHECK(scale[0] > 0.5f);
        CHECK(scale[0] < 2.0f);
    }

    int attn = 0, mlp = 0;
    for (int l = 0; l < c.num_layers; ++l) {
        const auto &lw = pw.layers[static_cast<size_t>(l)];
        INFO("layer " << l);
        CHECK(lw.loaded);
        CHECK(lw.d_alpha != nullptr);
        if (dorytest::LayerIsAttention(l)) {
            ++attn;
            CHECK(lw.is_attention);
            CHECK(lw.qkv.d_packed != nullptr);
            CHECK(lw.qkv.rows == dorytest::QSize() + 2 * dorytest::KVSize());
            CHECK(lw.qkv.cols == c.hidden_size);
            CHECK(lw.g.d_packed != nullptr);
            CHECK(lw.g.rows == dorytest::QSize());
            CHECK(lw.o.d_packed != nullptr);
            CHECK(lw.o.rows == c.hidden_size);
            CHECK(lw.o.cols == dorytest::QSize());
            // The residual plane is present on every projection.
            CHECK(lw.qkv_res.d_packed != nullptr);
            CHECK(lw.g_res.d_packed != nullptr);
            CHECK(lw.o_res.d_packed != nullptr);
            CHECK(lw.gate_up.d_packed == nullptr);
            CHECK(lw.d_sqk2 != nullptr);
            CHECK(lw.d_sgate2 != nullptr);
            CHECK(lw.d_suv_gate == nullptr);

            // The pre-scaled nGPT vectors are derived host-side: sqk2 must be
            // the squared, factor-scaled source value (not the raw vector).
            const auto sqk2 = DownloadF32(
                lw.d_sqk2, static_cast<size_t>(dorytest::QSize()));
            REQUIRE(sqk2.size() == static_cast<size_t>(dorytest::QSize()));
            for (int i = 0; i < dorytest::QSize(); i += 7) {
                // The source value is stored as BF16, so the expected value is
                // derived from the BF16-rounded pattern, not the float literal.
                const uint16_t bits = synthtest::FloatToBf16(
                    dorytest::Pattern(i, 0.02f));
                const uint32_t f32 = static_cast<uint32_t>(bits) << 16;
                float raw;
                std::memcpy(&raw, &f32, sizeof(raw));
                const double factor =
                    std::sqrt(static_cast<double>(c.head_dim)) /
                    c.initializer_range;
                const float want =
                    static_cast<float>((raw * factor) * (raw * factor));
                INFO("sqk2[" << i << "] got=" << sqk2[i] << " want=" << want);
                CHECK(std::abs(sqk2[i] - want) < std::abs(want) * 1e-4f + 1e-3f);
            }
            const auto alpha =
                DownloadF32(lw.d_alpha, static_cast<size_t>(c.hidden_size));
            REQUIRE(alpha.size() == static_cast<size_t>(c.hidden_size));
            for (float a : alpha) CHECK(a >= 0.0f);   // |alpha * scale|
        } else {
            ++mlp;
            CHECK_FALSE(lw.is_attention);
            CHECK(lw.gate_up.d_packed != nullptr);
            CHECK(lw.gate_up.rows == 2 * c.intermediate_size);
            CHECK(lw.down.d_packed != nullptr);
            CHECK(lw.down.cols == c.intermediate_size);
            CHECK(lw.gate_up_res.d_packed != nullptr);
            CHECK(lw.down_res.d_packed != nullptr);
            CHECK(lw.qkv.d_packed == nullptr);
            CHECK(lw.d_suv_gate != nullptr);
            CHECK(lw.d_suv_up != nullptr);
            CHECK(lw.d_sqk2 == nullptr);
        }
    }
    CHECK(attn == c.num_layers / 2);
    CHECK(mlp == c.num_layers / 2);

    // KV slot map: mlp layers have no slot, a recurrent attention layer owns
    // one slot per loop (consecutive), an input/output attention layer one.
    int expected_slot = 0;
    for (int l = 0; l < c.num_layers; ++l) {
        if (!dorytest::LayerIsAttention(l)) {
            CHECK(pw.slot_of_layer[static_cast<size_t>(l)] == -1);
            continue;
        }
        CHECK(pw.slot_of_layer[static_cast<size_t>(l)] == expected_slot);
        expected_slot +=
            dorytest::LayerIsRecurrent(l) ? c.recurrent_loops : 1;
    }
    CHECK(expected_slot == pw.kv_slots);

    pw.Free();
    cudaStreamDestroy(stream);
    std::filesystem::remove_all(tmp);
}
