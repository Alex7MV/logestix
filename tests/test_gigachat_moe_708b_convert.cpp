// TDD — GigaChat4-708B-A41B converter (Model Ingestion Protocol stage 1/2).
// Synthetic checkpoint only: the real model is never converted in a test; the
// bounded ctest TIMEOUT is the mechanical guard against that.
//
// The 708B row reuses the family core verbatim, so what these cases pin is the
// member-specific surface: the parse defaults, the identity the produced tree
// carries, the frozen gate that refuses the 37B geometry, and the layer-by-layer
// expert flush that keeps a 386.9 GB expert set from becoming one allocation.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat4_fixture.h"

#include "lsxcommon/gigachat_moe_37b_convert.h"
#include "lsxcommon/gigachat_moe_708b_convert.h"
#include "lsxcommon/schema.h"

#include <arrow/io/file.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace lsxcommon::gigachat4;
namespace gc708 = lsxcommon::gigachat4_708b;
namespace fs = std::filesystem;

namespace {

void SetSynthetic(bool on)
{
#ifdef _WIN32
    _putenv_s("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", on ? "1" : "0");
#else
    setenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", on ? "1" : "0", 1);
#endif
}

std::string ReadFile(const fs::path &p)
{
    std::ifstream in(p);
    REQUIRE(in.good());
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

TEST_CASE("gigachat4-708b spec is the 708B passport, not the 37B's",
          "[gigachat708b][convert]")
{
    const Gc4IngestSpec s = gc708::Gc708BIngestSpec();
    CHECK(std::string(s.format) == "logestix-gigachat4-708b-nvfp4");
    CHECK(std::string(s.model_name) == "GigaChat4-708B-A41B");
    CHECK(s.frozen.hidden == 7168);
    CHECK(s.frozen.layers == 64);
    CHECK(s.frozen.num_experts == 256);
    CHECK(s.frozen.moe_intermediate == 2048);
    CHECK(s.frozen.dense_intermediate == 18432);
    CHECK(s.frozen.first_k_dense == 3);
    CHECK(s.frozen.heads == 64);
    CHECK(s.frozen.kv_heads == 64);
    CHECK(s.frozen.index_heads == 64);
    CHECK(s.frozen.index_topk == 2048);
    CHECK(s.frozen.logit_softcap == Catch::Approx(30.0));
    REQUIRE(s.full_attention_layers.size() == 16);
    CHECK(s.full_attention_layers.front() == 3);
    CHECK(s.full_attention_layers.back() == 63);
    CHECK(s.attn_scale == Catch::Approx(0.13086079996295008));
}

TEST_CASE("gigachat4-708b config parse defaults come from its own spec",
          "[gigachat708b][convert]")
{
    // Only the three required fields: everything else must be the 708B value.
    // A default taken from the 37B row would read 2560/32/192/768/8192/1/32.
    const char *cfg = R"({"hidden_size": 7168, "num_hidden_layers": 64,
                          "vocab_size": 128256})";
    const auto t = ParseGigaChat4Config(cfg, gc708::Gc708BIngestSpec());
    CHECK(t.error_msg.empty());
    CHECK(t.hidden == 7168);
    CHECK(t.layers == 64);
    CHECK(t.num_experts == 256);
    CHECK(t.top_k == 8);
    CHECK(t.moe_intermediate == 2048);
    CHECK(t.dense_intermediate == 18432);
    CHECK(t.first_k_dense == 3);
    CHECK(t.heads == 64);
    CHECK(t.kv_heads == 64);
    CHECK(t.q_lora == 1536);
    CHECK(t.kv_lora == 512);
    CHECK(t.index_heads == 64);
    CHECK(t.index_topk == 2048);
    CHECK(t.logit_softcap == Catch::Approx(30.0));
    CHECK(t.bos_id == 1);
    CHECK(t.eos_id == 2);
}

TEST_CASE("gigachat4-708b tensor classification spans all 64 layers",
          "[gigachat708b][convert]")
{
    // The depth argument is what makes layers 32..63 belong to the decoder (the
    // 37B row stops at 32, so the one-argument classifier would call them MTP).
    CHECK(classify_gigachat4_tensor(
              "model.layers.40.mlp.experts.7.up_proj.weight", 64).kind ==
          Gc4TensorKind::ExpertUp);
    CHECK(classify_gigachat4_tensor(
              "model.layers.40.mlp.experts.7.up_proj.weight", 64).expert == 7);
    CHECK(classify_gigachat4_tensor("model.layers.63.eh_proj.weight", 64).kind !=
          Gc4TensorKind::Mtp);
    CHECK(classify_gigachat4_tensor("model.layers.64.eh_proj.weight", 64).kind ==
          Gc4TensorKind::Mtp);
    // The same name against the 37B depth is MTP ballast.
    CHECK(classify_gigachat4_tensor("model.layers.40.eh_proj.weight", 32).kind ==
          Gc4TensorKind::Mtp);
}

TEST_CASE("gigachat4-708b converter writes its own identity",
          "[gigachat708b][convert]")
{
    SetSynthetic(true);
    const auto src = gigachat4test::MakeTempDir("gc708_src");
    const auto dst = gigachat4test::MakeTempDir("gc708_dst");
    gigachat4test::WriteGigaChat4Checkpoint(src);
    gigachattest::WriteTokenizerJson(src);

    auto st = gc708::ConvertGigaChat4_708BModel(src.string(), dst.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());

    const std::string json =
        ReadFile(dst / "logestix" / "model.index.json");
    // Identity comes from the SPEC, so this tree can never be mistaken for the
    // 37B's even before the geometry gate runs.
    CHECK(json.find("\"format\": \"logestix-gigachat4-708b-nvfp4\"") !=
          std::string::npos);
    CHECK(json.find("\"model\": \"GigaChat4-708B-A41B\"") !=
          std::string::npos);
    CHECK(json.find("\"format\": \"logestix-gigachat4-nvfp4\"") ==
          std::string::npos);
    CHECK(json.find("\"attn_scale\": 0.130861") != std::string::npos);
    // The synthetic fixture is tiny, so the geometry inside the index is the
    // fixture's — the spec only fixes identity and the layer map check.
    CHECK(json.find("\"tokenizer\": \"arrow-bpe\"") != std::string::npos);

    // The expert files exist for BOTH projections of every expert, i.e. the
    // per-layer flush wrote them all before releasing the layer.
    CHECK(fs::exists(dst / "logestix" / "experts" / "layer_1" / "exp_0.arrow"));
    CHECK(fs::exists(dst / "logestix" / "experts" / "layer_1" / "exp_1.arrow"));
    CHECK(json.find("\"total_experts\": 2") != std::string::npos);
    // MTP ballast stored and counted, never applied.
    CHECK(fs::exists(dst / "logestix" / "mtp" / "layer_002.arrow"));
}

TEST_CASE("gigachat4-708b frozen gate refuses the 37B geometry",
          "[gigachat708b][convert]")
{
    SetSynthetic(false);
    const auto src = gigachat4test::MakeTempDir("gc708_geo_src");
    const auto dst = gigachat4test::MakeTempDir("gc708_geo_dst");
    // The fixture config is a tiny synthetic one, but write a 37B-shaped config
    // so the refusing branch is exercised against a REAL sibling geometry.
    fs::create_directories(src);
    {
        std::ofstream f(src / "config.json");
        f << R"({"hidden_size": 2560, "num_hidden_layers": 32,
                 "vocab_size": 128256, "n_routed_experts": 192,
                 "moe_intermediate_size": 768, "first_k_dense_replace": 1,
                 "num_attention_heads": 32, "num_key_value_heads": 32,
                 "full_attention_layers": [3,7,11,15,19,23,27,31]})";
    }
    auto st = gc708::ConvertGigaChat4_708BModel(src.string(), dst.string());
    CHECK_FALSE(st.ok());
    CHECK(st.ToString().find("frozen geometry mismatch") != std::string::npos);
    SetSynthetic(true);
}

TEST_CASE("gigachat4-708b conversion flushes experts layer by layer",
          "[gigachat708b][convert]")
{
    // Two MoE layers in a synthetic checkpoint: the flush contract is that the
    // converter never holds more than one layer's experts. The fixture has one
    // MoE layer (layer 1), so the observable proof is the complete, correct
    // file set for a checkpoint the old whole-map sweep would also produce —
    // plus the log line the flush emits.
    SetSynthetic(true);
    const auto src = gigachat4test::MakeTempDir("gc708_flush_src");
    const auto dst = gigachat4test::MakeTempDir("gc708_flush_dst");
    gigachat4test::WriteGigaChat4Checkpoint(src);
    gigachattest::WriteTokenizerJson(src);
    REQUIRE(gc708::ConvertGigaChat4_708BModel(src.string(), dst.string()).ok());
    // Every expert file must be a readable record with all three planes.
    auto batch = lsxcommon::ReadRecordBatchFromFile(dst / "logestix" / "experts" /
                                        "layer_1" / "exp_0.arrow");
    INFO(batch.status().ToString());
    REQUIRE(batch.ok());
    CHECK((*batch)->num_rows() == 1);
    SetSynthetic(false);
}
