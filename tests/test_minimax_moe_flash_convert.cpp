// MiniMax-M3.1-Flash converter stage (Model Ingestion Protocol stage 1/2).
// The sibling's source is ALREADY quantized: dense projections are MXFP8
// (F8_E4M3 payload + per-32 E8M0 exponents) and routed experts are ModelOpt
// NVFP4 (weight_packed / weight_scale / weight_global_scale). This suite pins
// the parser/classifier, the variant discriminator and the synthetic
// end-to-end conversion (label `convert`, bounded TIMEOUT).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/minimax_moe_convert.h"
#include "lsxcommon/minimax_moe_flash_convert.h"
#include "test_minimax_moe_flash_fixture.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace lsxcommon::minimax;
namespace fs = std::filesystem;

namespace {

std::string ReadAll(const fs::path &p)
{
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

float E2m1(int nib)
{
    static const float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = kMag[nib & 0x7];
    return (nib & 0x8) ? -v : v;
}

// Host decode of a produced DenseNvfp4Record (two nibbles per byte, low
// first; per-16 E4M3 block scales multiplied by the stored global scale).
std::vector<float> DecodeNvfp4(const lsxcommon::DenseNvfp4Record &r,
                               int64_t numel)
{
    std::vector<float> out(static_cast<size_t>(numel));
    for (int64_t i = 0; i < numel; ++i) {
        const uint8_t byte = r.packed[static_cast<size_t>(i >> 1)];
        const int nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
        const float scale = lsxcommon::e4m3_to_float(
            r.scales[static_cast<size_t>(i >> 4)]);
        out[static_cast<size_t>(i)] = E2m1(nib) * scale * r.global_scale;
    }
    return out;
}

double RelL2(const std::vector<float> &a, const std::vector<float> &b)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        num += d * d;
        den += static_cast<double>(a[i]) * a[i];
    }
    return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

const lsxcommon::DenseNvfp4Record *FindRec(
    const std::vector<lsxcommon::DenseNvfp4Record> &recs,
    const std::string &needle)
{
    for (const auto &r : recs)
        if (r.name.find(needle) != std::string::npos) return &r;
    return nullptr;
}

void WriteM3Config(const fs::path &dir)
{
    // Minimal M3-contour config: 4 layers, dense attention on the first 3,
    // theta 5e6. Only the fields the discriminator reads matter.
    std::ofstream f(dir / "config.json");
    f << R"({"model_type":"minimax_m3_vl","text_config":{)"
      << R"("hidden_size":64,"intermediate_size":16,"num_hidden_layers":4,)"
      << R"("num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,)"
      << R"("rotary_dim":16,"vocab_size":64,"rms_norm_eps":1e-06,)"
      << R"("rope_theta":5000000,"dense_intermediate_size":32,)"
      << R"("shared_intermediate_size":16,"num_local_experts":8,)"
      << R"("num_experts_per_tok":2,"moe_layer_freq":[0,1,1,1],)"
      << R"("sparse_attention_config":{"sparse_num_index_heads":1,)"
      << R"("sparse_index_dim":32,"sparse_block_size":8,"sparse_topk_blocks":2,)"
      << R"("sparse_local_block":1,"sparse_attention_freq":[0,1,1,1]}}})"
      << "\n";
}

} // namespace

TEST_CASE("M3.1 config parses and the variant discriminator separates it from M3",
          "[minimax-flash-convert]")
{
    fs::path dir = fs::temp_directory_path() / "lsx_minimax_flash_cfg";
    fs::remove_all(dir);
    fs::create_directories(dir);
    minimax_flash_test::WriteConfig(dir);
    auto top = ParseMinimaxConfig(dir.string());
    REQUIRE(top.error_msg.empty());
    const auto &c = minimax_flash_test::TestCfg();
    CHECK(top.rope_theta == Catch::Approx(10000000.0));
    CHECK(top.hidden_size == c.hidden);
    CHECK(top.num_layers == c.layers);
    REQUIRE(top.layer_types.size() == static_cast<size_t>(c.layers));
    for (const auto &t : top.layer_types) CHECK(t == "sparse_attention");
    CHECK(top.mlp_types[0] == "dense");
    CHECK(top.mlp_types[1] == "sparse");
    CHECK(ClassifyMinimaxVariant(top) == MinimaxVariant::kM3Flash);

    fs::path m3 = fs::temp_directory_path() / "lsx_minimax_m3_cfg";
    fs::remove_all(m3);
    fs::create_directories(m3);
    WriteM3Config(m3);
    auto top_m3 = ParseMinimaxConfig(m3.string());
    REQUIRE(top_m3.error_msg.empty());
    CHECK(ClassifyMinimaxVariant(top_m3) == MinimaxVariant::kM3);

    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::remove_all(m3, ec);
}

TEST_CASE("M3.1 source tensors classify into the base kind + source encoding",
          "[minimax-flash-convert]")
{
    using K = MinimaxTensorKind;
    using E = MinimaxSourceEncoding;
    auto cl = [](const std::string &n) { return classify_minimax_tensor(n); };

    {
        auto p = cl("language_model.model.layers.0.self_attn.q_proj.weight");
        CHECK(p.kind == K::AttnQ);
        CHECK(p.encoding == E::kRaw);
    }
    {
        auto p = cl(
            "language_model.model.layers.0.self_attn.q_proj.weight_scale_inv");
        CHECK(p.kind == K::AttnQ);
        CHECK(p.encoding == E::kMxfp8ScaleInv);
    }
    {
        auto p = cl("language_model.model.layers.3.block_sparse_moe.shared_experts."
                    "up_proj.weight_scale_inv");
        CHECK(p.kind == K::SharedUpProj);
        CHECK(p.encoding == E::kMxfp8ScaleInv);
    }
    {
        auto p = cl("language_model.model.layers.5.block_sparse_moe.experts.42."
                    "w1.weight_packed");
        CHECK(p.kind == K::ExpertW1);
        CHECK(p.layer == 5);
        CHECK(p.expert == 42);
        CHECK(p.encoding == E::kExpertPacked);
    }
    {
        auto p = cl("language_model.model.layers.5.block_sparse_moe.experts.42."
                    "w2.weight_scale");
        CHECK(p.kind == K::ExpertW2);
        CHECK(p.encoding == E::kExpertScale);
    }
    {
        auto p = cl("language_model.model.layers.5.block_sparse_moe.experts.42."
                    "w3.weight_global_scale");
        CHECK(p.kind == K::ExpertW3);
        CHECK(p.encoding == E::kExpertGlobalScale);
    }
    {
        auto p = cl("language_model.model.layers.5.block_sparse_moe.experts.42."
                    "w3.input_global_scale");
        CHECK(p.kind == K::ExpertW3);
        CHECK(p.encoding == E::kExpertInputGlobalScale);
    }
    CHECK(cl("language_model.model.layers.0.self_attn.index_q_proj.weight").kind ==
          K::IndexQ);
}

TEST_CASE("ConvertMinimaxFlashModel converts the synthetic MXFP8+NVFP4 tree",
          "[minimax-flash-convert]")
{
    const char *prev = std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1", 1);
    fs::path in = fs::temp_directory_path() / "lsx_minimax_flash_in";
    fs::path out = fs::temp_directory_path() / "lsx_minimax_flash_out";
    fs::remove_all(in);
    fs::remove_all(out);
    REQUIRE(minimax_flash_test::BuildSyntheticCheckpoint(in) > 0);

    auto st = ConvertMinimaxFlashModel(in.string(), out.string());
    if (!st.ok()) WARN(st.ToString());
    REQUIRE(st.ok());

    const fs::path base = out;
    const auto &c = minimax_flash_test::TestCfg();

    // 1) Identity + structure: the same emitted layout tag, the sibling's name.
    {
        const std::string j = ReadAll(base / "model.index.json");
        CHECK(j.find("\"layout\": 2") !=
              std::string::npos);
        CHECK(j.find("\"model\": \"MiniMax-M3.1-Flash\"") != std::string::npos);
        CHECK(j.find("\"model_type\": \"minimax-m3.1-flash\"") !=
              std::string::npos);
        CHECK(j.find("\"rope_theta\"") != std::string::npos);
        CHECK(j.find("\"skipped_visual_tensors\": 2") != std::string::npos);
        CHECK(j.find("\"total_experts\": " +
                     std::to_string(c.experts * (c.layers - c.first_moe_layer))) !=
              std::string::npos);
    }

    // 2) Dense MXFP8 -> NVFP4: the scale_inv companion is consumed, the
    //    projection lands in the pinned NVFP4 batch, and the 4-bit payload
    //    reconstructs the MXFP8-dequantized source within a 4-bit error band.
    {
        auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(
                        lsxcommon::ReadRecordBatchFromFile(
                            base / "pinned" / "layer_0_nvfp4.arrow")
                            .ValueOrDie())
                        .ValueOrDie();
        for (const auto &r : recs)
            CHECK(r.name.find("weight_scale_inv") == std::string::npos);
        const auto *q = FindRec(recs, "self_attn.q_proj.weight");
        REQUIRE(q != nullptr);
        const int64_t rows = c.heads * c.head_dim;
        const int64_t numel = rows * c.hidden;
        const auto got = DecodeNvfp4(*q, numel);

        const auto raw = synthtest::MakeBf16Values(
            numel, [](int64_t i) {
                const int64_t k = (i * 7 + 30 * 13) % 23;
                return static_cast<float>(k - 11) * 0.011f;
            });
        std::vector<uint8_t> fp8, scales;
        minimax_flash_test::QuantizeMxfp8(raw, rows, c.hidden, fp8, scales);
        std::vector<float> ref(static_cast<size_t>(numel));
        const int64_t nb = c.hidden / 32;
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t j = 0; j < c.hidden; ++j) {
                const float s = minimax_flash_test::E8m0ToFloat(
                    scales[static_cast<size_t>(r * nb + j / 32)]);
                ref[static_cast<size_t>(r * c.hidden + j)] =
                    lsxcommon::e4m3_to_float(fp8[static_cast<size_t>(r * c.hidden + j)]) * s;
            }
        CHECK(RelL2(ref, got) < 0.25);
        CHECK(q->global_scale > 0.0f);
    }

    // 3) Every layer carries the indexer (sparse attention throughout),
    //    including layer 0 which still has the dense MLP.
    {
        auto recs = lsxcommon::DenseNvfp4Schema::UnpackBatch(
                        lsxcommon::ReadRecordBatchFromFile(
                            base / "pinned" / "layer_0_nvfp4.arrow")
                            .ValueOrDie())
                        .ValueOrDie();
        CHECK(FindRec(recs, "self_attn.index_q_proj.weight") != nullptr);
        CHECK(FindRec(recs, "self_attn.index_k_proj.weight") != nullptr);
        CHECK(FindRec(recs, "mlp.gate_proj.weight") != nullptr);
    }

    // 4) Routed experts are a VERBATIM passthrough of the ModelOpt NVFP4
    //    record, with the divisor-form global scale inverted.
    {
        auto batch = lsxcommon::ReadRecordBatchFromFile(
                         base / "experts" / "layer_1" / "exp_0.arrow")
                         .ValueOrDie();
        auto pe = lsxcommon::ExpertSchema::UnpackBatch(batch).ValueOrDie();
        const int64_t H = c.hidden, I = c.moe_inter;
        const auto raw = synthtest::MakeBf16Values(
            I * H, [](int64_t i) {
                const int64_t k = (i * 7 + (1000 + 0 * 3 + 1) * 13) % 23;
                return static_cast<float>(k - 11) * 0.011f;
            });
        std::vector<uint8_t> packed, scales;
        float gs_src = 0.0f;
        minimax_flash_test::QuantizeNvfp4Source(raw, I, H, packed, scales,
                                                gs_src);
        CHECK(pe.w1_packed == packed);
        CHECK(pe.w1_scales == scales);
        CHECK(pe.w1_global_scale ==
              Catch::Approx(1.0f / gs_src).epsilon(1e-6));
        CHECK(static_cast<int64_t>(pe.w1_packed.size()) == I * H / 2);
        CHECK(static_cast<int64_t>(pe.w1_scales.size()) == I * H / 16);
    }

    // 5) Determinism: a second conversion produces a byte-identical index.
    {
        fs::path out2 = fs::temp_directory_path() / "lsx_minimax_flash_out2";
        fs::remove_all(out2);
        REQUIRE(ConvertMinimaxFlashModel(in.string(), out2.string()).ok());
        CHECK(ReadAll(base / "model.index.json") ==
              ReadAll(out2 / "model.index.json"));
        fs::remove_all(out2);
    }

    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out, ec);
    if (prev) setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", prev, 1);
    else unsetenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY");
}
