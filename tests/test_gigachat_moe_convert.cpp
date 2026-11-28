// TDD — GigaChat3.5 converter (Model Ingestion Protocol step 1). Synthetic
// checkpoint only: the real model is never converted in a test. The ctest
// TIMEOUT is the mechanical guard against a real full conversion.
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat_moe_fixture.h"

#include "lsxcommon/gigachat_moe_convert.h"
#include "lsxcommon/tokenizer.h"

#include <arrow/io/file.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace lsxcommon::gigachat;
namespace fs = std::filesystem;

TEST_CASE("gigachat tensor classifier", "[gigachat][convert]")
{
    auto k = [](const char *n) { return classify_gigachat_tensor(n).kind; };
    CHECK(k("model.embed_tokens.weight") == GigaChatTensorKind::Embed);
    CHECK(k("lm_head.weight") == GigaChatTensorKind::LmHead);
    CHECK(k("model.norm.weight") == GigaChatTensorKind::FinalNormWeight);
    CHECK(k("model.norm.gate_up_projection.weight") ==
          GigaChatTensorKind::FinalNormGateUp);
    CHECK(k("model.norm.gate_down_projection.weight") ==
          GigaChatTensorKind::FinalNormGateDown);
    CHECK(k("model.layers.3.input_layernorm.weight") ==
          GigaChatTensorKind::LayerNormWeight);
    CHECK(k("model.layers.7.post_feedforward_layernorm.gate_down_projection.weight") ==
          GigaChatTensorKind::LayerNormGateDown);
    CHECK(k("model.layers.3.self_attn.q_a_proj.weight") ==
          GigaChatTensorKind::QaProj);
    CHECK(k("model.layers.3.self_attn.kv_a_proj_with_mqa.weight") ==
          GigaChatTensorKind::KvaProj);
    CHECK(k("model.layers.3.self_attn.gate_proj.weight") ==
          GigaChatTensorKind::AttnGateProj);
    CHECK(k("model.layers.3.self_attn.q_a_layernorm.weight") ==
          GigaChatTensorKind::LayerNormWeight);
    CHECK(k("model.layers.4.self_attn.in_proj_qkvz.weight") ==
          GigaChatTensorKind::GdnInProjQkvz);
    CHECK(k("model.layers.4.self_attn.in_proj_ba.weight") ==
          GigaChatTensorKind::GdnInProjBa);
    CHECK(k("model.layers.4.self_attn.conv1d.weight") ==
          GigaChatTensorKind::GdnConv1d);
    CHECK(k("model.layers.4.self_attn.A_log") == GigaChatTensorKind::GdnALog);
    CHECK(k("model.layers.4.self_attn.norm.weight") ==
          GigaChatTensorKind::GdnNormWeight);
    CHECK(k("model.layers.5.mlp.gate.weight") ==
          GigaChatTensorKind::RouterWeight);
    CHECK(k("model.layers.5.mlp.gate.e_score_correction_bias") ==
          GigaChatTensorKind::RouterBias);
    CHECK(k("model.layers.5.mlp.shared_experts.gate_proj.weight") ==
          GigaChatTensorKind::SharedGate);
    CHECK(k("model.layers.5.mlp.experts.200.up_proj.weight") ==
          GigaChatTensorKind::ExpertUp);
    CHECK(k("model.layers.0.mlp.down_proj.weight") ==
          GigaChatTensorKind::MlpDown);
    CHECK(k("model.layers.40.eh_proj.weight") == GigaChatTensorKind::Mtp);
    CHECK(k("model.layers.41.shared_head.head.weight") == GigaChatTensorKind::Mtp);
    CHECK(k("garbage") == GigaChatTensorKind::Unknown);

    auto e = classify_gigachat_tensor("model.layers.5.mlp.experts.200.up_proj.weight");
    CHECK(e.layer == 5);
    CHECK(e.expert == 200);
}

TEST_CASE("gigachat config parser reads the authoritative fields",
          "[gigachat][convert]")
{
    const char *cfg = R"({
      "hidden_size": 7168, "num_hidden_layers": 40, "vocab_size": 128256,
      "num_attention_heads": 64, "num_key_value_heads": 64,
      "n_routed_experts": 256, "num_experts_per_tok": 8,
      "moe_intermediate_size": 2048, "first_k_dense_replace": 3,
      "rms_norm_eps": 1e-06, "routed_scaling_factor": 2.5, "swiglu_limit": 10,
      "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 2
    })";
    auto t = ParseGigaChatConfig(cfg);
    CHECK(t.error_msg.empty());
    CHECK(t.hidden == 7168);
    CHECK(t.layers == 40);
    CHECK(t.vocab == 128256);
    CHECK(t.num_experts == 256);
    CHECK(t.top_k == 8);
    CHECK(t.bos_id == 1);
    CHECK(t.pad_id == 2);

    auto bad = ParseGigaChatConfig("{ not json ");
    CHECK_FALSE(bad.error_msg.empty());
}

TEST_CASE("gigachat converter emits the logestix-gigachat-nvfp4 layout",
          "[gigachat][convert]")
{
    const auto src = gigachattest::MakeTempDir("gc_src");
    const auto dst = gigachattest::MakeTempDir("gc_dst");
    gigachattest::WriteGigaChatCheckpoint(src);
    gigachattest::WriteTokenizerJson(src);

    auto st = ConvertGigaChatModel(src.string(), dst.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());

    std::ifstream in(dst / "model.index.json");
    REQUIRE(in.good());
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string json = ss.str();

    CHECK(json.find("\"layout\": 2") !=
          std::string::npos);
    CHECK(json.find("\"full_attention_layers\": [1]") != std::string::npos);
    // No ballast in v1: the MTP head is not stored, so the index carries no
    // `mtp: stored` declaration to go stale against a directory that is absent.
    CHECK(json.find("\"mtp\": \"stored\"") == std::string::npos);
    CHECK(json.find("\"total_experts\": 2") != std::string::npos);
    CHECK(fs::exists(dst / "pinned" / "global.arrow"));
    CHECK(fs::exists(dst / "pinned" / "layer_000_nvfp4.arrow"));
    CHECK(fs::exists(dst / "pinned" / "layer_001_nvfp4.arrow"));
    CHECK(fs::exists(dst / "experts" / "layer_1" / "exp_0.arrow"));
    CHECK_FALSE(fs::exists(dst / "mtp"));
    // Model Ingestion Protocol stage 4: the converter owns the tokenizer
    // bridge. GigaChat's checkpoint is the Llama-3/o200k case-run BPE, so the
    // emitted metadata must say tokenizer_kind 3 (not the legacy kind 0, which
    // mis-splits mixed-case identifiers).
    CHECK(json.find("\"tokenizer\": \"arrow-bpe\"") != std::string::npos);
    CHECK(fs::exists(dst / "pinned" / "tokenizer_vocab.arrow"));
    CHECK(fs::exists(dst / "pinned" / "tokenizer_merges.arrow"));

    auto vf = arrow::io::MemoryMappedFile::Open(
        (dst / "pinned" / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    auto mf = arrow::io::MemoryMappedFile::Open(
        (dst / "pinned" / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(vf.ok());
    REQUIRE(mf.ok());
    lsxcommon::ArrowTokenizerDb db;
    auto meta = lsxcommon::load_tokenizer_metadata(*vf, *mf, db);
    INFO(meta.ToString());
    REQUIRE(meta.ok());
    CHECK(db.tokenizer_kind == 3);
    CHECK(db.bos_id == 259);
    CHECK(db.eos_id == 2);
    CHECK(db.pad_id == 2);
    // The emitted bridge must encode/decode byte-exactly on a mixed-case
    // sample (its exact ids are pinned against the source tokenizer by
    // test_gigachat_moe_tokenizer_parity).
    std::vector<int32_t> ids(256, 0);
    size_t cap = ids.size();
    const char *text = "getUserName";
    const int32_t n =
        lsxcommon::arrow_tokenizer_encode(db, text, std::strlen(text),
                                          ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));
    std::vector<char> back(64, '\0');
    size_t back_len = back.size();
    REQUIRE(lsxcommon::arrow_tokenizer_decode(db, ids.data(), ids.size(),
                                              back.data(), &back_len) == 0);
    CHECK(std::string(back.data(), back_len) == text);
}
